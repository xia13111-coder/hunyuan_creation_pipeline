#!/usr/bin/env python3
"""Iteratively refine photo masks with CAD shape and visibility constraints."""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from ..evidence.part_id_projection import (
    PartIdProjectionError,
    _register_similarity_mask,
)
from .entityseg_regions import EntitySegRegionError, _boundary_metrics
from .sam3_regions import _normalized_shape_agreement


SCHEMA_VERSION = "qwen-cad-sam3-entityseg-hybrid/v6"
MODEL_SCHEMA_VERSION = "qwen-cad-sam3-entityseg-hybrid/v5"
LEGACY_SCHEMA_VERSION = "qwen-cad-sam3-entityseg-hybrid/v2"
MAXIMUM_ENTITY_TO_CAD_AREA_RATIO = 1.85
MINIMUM_ENTITY_CAD_DIRECT_IOU = 0.50
MINIMUM_ENTITY_CAD_SHAPE_IOU = 0.55
MAXIMUM_ENTITY_CAD_CENTROID_DISTANCE = 0.15
MINIMUM_ENTITY_EDGE_SUPPORT = 0.70
MINIMUM_ENTITY_EDGE_IMPROVEMENT = 0.03
MINIMUM_SAM_ENTITY_OVERLAP_SMALLER = 0.50
MINIMUM_DIRECT_IOU_WHEN_SAM_DISAGREES = 0.60
MINIMUM_CONNECTED_COMPONENT_PIXELS = 16
MAXIMUM_FINAL_TO_CAD_AREA_RATIO = 1.25
MAXIMUM_CAD_SUPPORT_RADIUS_FRACTION = 0.04
MINIMUM_CAD_SUPPORT_RADIUS_PIXELS = 2
MINIMUM_AMODAL_CANDIDATE_PRECISION = 0.88
MINIMUM_AMODAL_COMPLETION_SHAPE_IOU = 0.62
SHAPE_GUIDED_OPTIMIZATION_ITERATIONS = 5
MAXIMUM_VISIBLE_SUPPORT_RADIUS_FRACTION = 0.025
MAXIMUM_VISIBLE_SUPPORT_RADIUS_PIXELS = 12
MAXIMUM_VISIBLE_CORE_RADIUS_PIXELS = 3
MINIMUM_REFINED_TO_VISIBLE_AREA_RATIO = 0.35
MINIMUM_REFINED_AMODAL_PRECISION = 0.85
MODEL_SHAPE_NORMALIZATION_SIZE = 96


def _part_color(part_id: str) -> tuple[int, int, int]:
    """Return the deterministic RGB colour used by model Part-ID renders."""

    suffix = part_id[1:] if part_id.startswith("P") else ""
    number = int(suffix) if suffix.isdigit() else sum(map(ord, part_id))
    hue = (number * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return int(red * 255), int(green * 255), int(blue * 255)


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


def _records(
    document: Mapping[str, Any], label: str
) -> dict[tuple[str, str], Mapping[str, Any]]:
    output: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, row in enumerate(document["records"]):
        if not isinstance(row, Mapping):
            raise EntitySegRegionError(f"{label} record {index} is malformed")
        key = (str(row.get("view_id")), str(row.get("group_id")))
        if key in output:
            raise EntitySegRegionError(f"duplicate {label} region: {key}")
        output[key] = row
    return output


def _bound_request(
    document: Mapping[str, Any], *, label: str
) -> tuple[Path, dict[str, Any]]:
    binding = document.get("request")
    if not isinstance(binding, Mapping):
        raise EntitySegRegionError(f"{label} has no request binding")
    value = binding.get("path")
    expected_sha256 = binding.get("sha256")
    expected_document_sha256 = binding.get("document_sha256")
    if not all(
        isinstance(value, str)
        for value in (value, expected_sha256, expected_document_sha256)
    ):
        raise EntitySegRegionError(f"{label} request binding is malformed")
    path = Path(str(value)).expanduser().resolve(strict=True)
    if _sha256_file(path) != expected_sha256:
        raise EntitySegRegionError(f"{label} request hash mismatch")
    try:
        request = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EntitySegRegionError(f"unable to read {label} request") from exc
    if (
        not isinstance(request, dict)
        or _canonical_sha256(request) != expected_document_sha256
    ):
        raise EntitySegRegionError(f"{label} request document hash mismatch")
    return path, request


def _relation_request_records(
    *,
    request: Mapping[str, Any],
    expected_keys: set[tuple[str, str]],
    amodal_manifest_path: Path | None,
) -> tuple[
    dict[tuple[str, str], Mapping[str, Any]],
    dict[tuple[str, str], Mapping[str, Any]],
]:
    relation = request.get("relation_guidance")
    if relation is None:
        return {}, {}
    if not isinstance(relation, Mapping) or relation.get("schema_version") != (
        "qwen-part-relation-guidance/v1"
    ):
        raise EntitySegRegionError("relation-guided request contract is malformed")
    integrity = relation.get("integrity")
    unsigned_relation = dict(relation)
    unsigned_relation.pop("integrity", None)
    if not isinstance(integrity, Mapping) or integrity.get(
        "document_sha256"
    ) != _canonical_sha256(unsigned_relation):
        raise EntitySegRegionError("relation-guided request integrity mismatch")
    policy = relation.get("policy")
    if (
        not isinstance(policy, Mapping)
        or policy.get("target_first_pass_mask_used_for_own_location") is not False
        or policy.get("target_direct_cad_projection_used_when_relation_accepted")
        is not False
        or policy.get("per_mesh_pose_change_allowed") is not False
        or policy.get("whole_asset_transform_changed") is not False
        or policy.get("assembly_camera_changed") is not False
    ):
        raise EntitySegRegionError("relation-guided request policy is unsafe")
    if amodal_manifest_path is not None:
        inputs = relation.get("inputs")
        amodal_binding = (
            inputs.get("cad_amodal_templates") if isinstance(inputs, Mapping) else None
        )
        resolved_amodal = amodal_manifest_path.expanduser().resolve(strict=True)
        if (
            not isinstance(amodal_binding, Mapping)
            or amodal_binding.get("path") != str(resolved_amodal)
            or amodal_binding.get("sha256") != _sha256_file(resolved_amodal)
        ):
            raise EntitySegRegionError(
                "relation-guided request CAD model authority differs"
            )
    audits: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, row in enumerate(relation.get("records", [])):
        if not isinstance(row, Mapping):
            raise EntitySegRegionError(f"relation record {index} is malformed")
        key = (str(row.get("view_id")), str(row.get("part_id")))
        if key in audits:
            raise EntitySegRegionError(f"duplicate relation record: {key}")
        if (
            row.get("target_first_pass_mask_used") is not False
            or row.get("whole_asset_transform_changed") is not False
            or row.get("assembly_camera_changed") is not False
            or row.get("per_mesh_pose_change_allowed") is not False
        ):
            raise EntitySegRegionError(f"unsafe relation record: {key}")
        audits[key] = row
    regions: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, row in enumerate(request.get("regions", [])):
        if not isinstance(row, Mapping):
            raise EntitySegRegionError(f"relation request region {index} is malformed")
        key = (str(row.get("view_id")), str(row.get("group_id")))
        if key in regions:
            raise EntitySegRegionError(f"duplicate relation request region: {key}")
        regions[key] = row
    if set(audits) != expected_keys or set(regions) != expected_keys:
        raise EntitySegRegionError(
            "relation guidance does not exactly cover the segmentation regions"
        )
    for key, audit in audits.items():
        if audit.get("accepted") is True:
            region_audit = regions[key].get("relation_guidance")
            neighbor = regions[key].get("cad_assembly_neighbor_context")
            if (
                not isinstance(region_audit, Mapping)
                or region_audit.get("accepted") is not True
                or region_audit.get("target_first_pass_mask_used") is not False
                or not isinstance(neighbor, Mapping)
            ):
                raise EntitySegRegionError(
                    f"accepted relation request is incomplete: {key}"
                )
            _load_document_mask(neighbor, label=f"relation neighbor context {key}")
    return audits, regions


def _model_template_records(
    document: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    output: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, row in enumerate(document["records"]):
        if not isinstance(row, Mapping):
            raise EntitySegRegionError(
                f"CAD model template record {index} is malformed"
            )
        key = (str(row.get("view_id")), str(row.get("part_id")))
        if not key[0] or not key[1] or key in output:
            raise EntitySegRegionError(
                f"duplicate or invalid CAD model template: {key}"
            )
        output[key] = row
    return output


def _load_document_mask(
    document: Mapping[str, Any], *, label: str
) -> tuple[Path, np.ndarray]:
    value = document.get("path")
    expected_sha256 = document.get("sha256")
    if not isinstance(value, str) or not isinstance(expected_sha256, str):
        raise EntitySegRegionError(f"{label} binding is malformed")
    path = Path(value).expanduser().resolve(strict=True)
    if _sha256_file(path) != expected_sha256:
        raise EntitySegRegionError(f"{label} hash mismatch")
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.ndim != 2 or not np.any(mask >= 128):
        raise EntitySegRegionError(f"{label} is not a non-empty mask")
    return path, mask >= 128


def _model_domain_shape_references(
    *,
    manifest_path: Path,
    expected_keys: set[tuple[str, str]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any],]:
    """Load target-part silhouettes strictly from CAD-render image space.

    The raw isolated projection and renderer-authored visible Part-ID mask both
    remain in the model camera image.  They are never warped into a reference
    photo here.  Downstream comparison normalizes each photo candidate in its
    own coordinate system, so the model is a shape authority rather than a
    pasted pixel mask.
    """

    manifest_path = manifest_path.expanduser().resolve(strict=True)
    document = _read_manifest(manifest_path, "CAD model template manifest")
    records = _model_template_records(document)
    if set(records) != expected_keys:
        raise EntitySegRegionError(
            "CAD model templates do not exactly cover the hybrid region set"
        )
    inputs = document.get("inputs")
    registry_binding = inputs.get("registry") if isinstance(inputs, Mapping) else None
    if not isinstance(registry_binding, Mapping):
        raise EntitySegRegionError("CAD model templates have no registry binding")
    registry_value = registry_binding.get("path")
    if not isinstance(registry_value, str):
        raise EntitySegRegionError("CAD model template registry path is malformed")
    registry_path = Path(registry_value).expanduser().resolve(strict=True)
    if registry_binding.get("sha256") != _sha256_file(registry_path):
        raise EntitySegRegionError("CAD model template registry hash mismatch")
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EntitySegRegionError(
            "unable to read CAD model template registry"
        ) from exc
    if not isinstance(registry, Mapping):
        raise EntitySegRegionError("CAD model template registry is malformed")
    spatial_binding = (
        inputs.get("spatial_report") if isinstance(inputs, Mapping) else None
    )
    if not isinstance(spatial_binding, Mapping) or not isinstance(
        spatial_binding.get("path"), str
    ):
        raise EntitySegRegionError("CAD model templates have no spatial binding")
    spatial_path = Path(str(spatial_binding["path"])).expanduser().resolve(strict=True)
    if spatial_binding.get("sha256") != _sha256_file(spatial_path):
        raise EntitySegRegionError("CAD model template spatial hash mismatch")
    try:
        spatial = json.loads(spatial_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EntitySegRegionError("unable to read CAD model spatial report") from exc
    if not isinstance(spatial, Mapping):
        raise EntitySegRegionError("CAD model spatial report is malformed")
    alignments = {
        str(row.get("reference_view_id")): row
        for row in spatial.get("view_alignments", [])
        if isinstance(row, Mapping) and isinstance(row.get("reference_view_id"), str)
    }
    render_set = registry.get("render_set")
    views = {
        str(row.get("view_id")): row
        for row in (
            render_set.get("views", []) if isinstance(render_set, Mapping) else []
        )
        if isinstance(row, Mapping) and isinstance(row.get("view_id"), str)
    }
    references: dict[tuple[str, str], dict[str, Any]] = {}
    decoded_views: dict[str, tuple[Path, str, np.ndarray, np.ndarray, Path, str]] = {}
    for key in sorted(records):
        row = records[key]
        raw_document = row.get("raw_amodal_mask")
        if not isinstance(raw_document, Mapping):
            raise EntitySegRegionError(f"CAD model template has no raw mask: {key}")
        raw_path, complete_shape = _load_document_mask(
            raw_document, label=f"CAD model complete shape {key}"
        )
        render_view_id = row.get("render_view_id")
        view = views.get(str(render_view_id))
        if not isinstance(view, Mapping):
            raise EntitySegRegionError(f"CAD model view is missing: {key}")
        if str(render_view_id) not in decoded_views:
            part_ids_path = (
                Path(str(view.get("part_ids", ""))).expanduser().resolve(strict=True)
            )
            part_ids_sha256 = _sha256_file(part_ids_path)
            part_ids_image = cv2.imread(str(part_ids_path), cv2.IMREAD_COLOR)
            model_image_path = (
                Path(str(view.get("rgb", ""))).expanduser().resolve(strict=True)
            )
            model_image_sha256 = _sha256_file(model_image_path)
            if part_ids_image is None:
                raise EntitySegRegionError(
                    f"unable to decode CAD model Part-ID image: {part_ids_path}"
                )
            model_image = cv2.imread(str(model_image_path), cv2.IMREAD_COLOR)
            if model_image is None or model_image.shape[:2] != part_ids_image.shape[:2]:
                raise EntitySegRegionError(
                    f"CAD model RGB/Part-ID dimensions differ: {render_view_id}"
                )
            colours, counts = np.unique(
                part_ids_image.reshape(-1, 3), axis=0, return_counts=True
            )
            background_colour = colours[int(np.argmax(counts))]
            assembly_visible_shape = np.any(
                part_ids_image != background_colour.reshape(1, 1, 3), axis=2
            )
            if not np.any(assembly_visible_shape):
                raise EntitySegRegionError(
                    f"CAD model assembly context is empty: {render_view_id}"
                )
            decoded_views[str(render_view_id)] = (
                part_ids_path,
                part_ids_sha256,
                part_ids_image,
                assembly_visible_shape,
                model_image_path,
                model_image_sha256,
            )
        (
            part_ids_path,
            part_ids_sha256,
            part_ids_image,
            assembly_visible_shape,
            model_image_path,
            model_image_sha256,
        ) = decoded_views[str(render_view_id)]
        red, green, blue = _part_color(key[1])
        visible_shape = np.all(
            part_ids_image == np.asarray([blue, green, red], dtype=np.uint8), axis=2
        )
        if not np.any(visible_shape) or visible_shape.shape != complete_shape.shape:
            raise EntitySegRegionError(
                f"target Part-ID is absent from its CAD model image: {key}"
            )
        declared_visible = next(
            (
                item.get("pixels")
                for item in view.get("visible_parts", [])
                if isinstance(item, Mapping) and item.get("part_id") == key[1]
            ),
            None,
        )
        visible_pixels = int(np.count_nonzero(visible_shape))
        if declared_visible != visible_pixels:
            raise EntitySegRegionError(
                f"CAD model visible-pixel audit differs for {key}"
            )
        visible_inside_complete = int(np.count_nonzero(visible_shape & complete_shape))
        visible_precision = visible_inside_complete / max(visible_pixels, 1)
        if visible_precision < 0.75:
            raise EntitySegRegionError(
                f"CAD model visible shape is inconsistent with isolated mesh: {key}"
            )
        quarter_turns = row.get("quarter_turns_ccw")
        if isinstance(quarter_turns, bool) or not isinstance(quarter_turns, int):
            raise EntitySegRegionError(
                f"CAD model shape quarter-turn audit is malformed: {key}"
            )
        modal_document = row.get("modal_visibility_mask")
        if not isinstance(modal_document, Mapping):
            raise EntitySegRegionError(
                f"CAD model template has no local observation mask: {key}"
            )
        _modal_path, reference_modal = _load_document_mask(
            modal_document, label=f"reference local visibility {key}"
        )
        alignment = alignments.get(key[0])
        if not isinstance(alignment, Mapping):
            raise EntitySegRegionError(f"CAD model alignment is missing: {key}")
        bbox_affine = np.asarray(alignment.get("bbox_affine"), dtype=np.float32)
        ecc_warp = np.asarray(alignment.get("ecc_warp"), dtype=np.float32)
        if bbox_affine.shape != (2, 3) or ecc_warp.shape != (2, 3):
            raise EntitySegRegionError(f"CAD model alignment is malformed: {key}")
        component_count, component_labels = cv2.connectedComponents(
            visible_shape.astype(np.uint8), connectivity=8
        )
        selected_components: list[int] = []
        component_audits: list[dict[str, int]] = []
        for component_index in range(1, component_count):
            component = component_labels == component_index
            rotated_component = np.rot90(
                component.astype(np.uint8), quarter_turns % 4
            ).copy()
            normalized_component = cv2.warpAffine(
                rotated_component,
                bbox_affine,
                (reference_modal.shape[1], reference_modal.shape[0]),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            projected_component = (
                cv2.warpAffine(
                    normalized_component,
                    ecc_warp,
                    (reference_modal.shape[1], reference_modal.shape[0]),
                    flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                > 0
            )
            intersection = int(np.count_nonzero(projected_component & reference_modal))
            projected_pixels = int(np.count_nonzero(projected_component))
            component_audits.append(
                {
                    "component_index": component_index,
                    "model_pixels": int(np.count_nonzero(component)),
                    "projected_pixels": projected_pixels,
                    "reference_intersection_pixels": intersection,
                }
            )
            if intersection > 0:
                selected_components.append(component_index)
        if not selected_components:
            raise EntitySegRegionError(
                f"no CAD-model-image component corresponds to the local observation: {key}"
            )
        local_visible_shape = np.isin(component_labels, selected_components)
        rotated_assembly = np.rot90(
            assembly_visible_shape.astype(np.uint8), quarter_turns % 4
        ).copy()
        normalized_assembly = cv2.warpAffine(
            rotated_assembly,
            bbox_affine,
            (reference_modal.shape[1], reference_modal.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        reference_assembly = (
            cv2.warpAffine(
                normalized_assembly,
                ecc_warp,
                (reference_modal.shape[1], reference_modal.shape[0]),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            > 0
        )
        rotated_target = np.rot90(
            visible_shape.astype(np.uint8), quarter_turns % 4
        ).copy()
        normalized_target = cv2.warpAffine(
            rotated_target,
            bbox_affine,
            (reference_modal.shape[1], reference_modal.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        reference_target = (
            cv2.warpAffine(
                normalized_target,
                ecc_warp,
                (reference_modal.shape[1], reference_modal.shape[0]),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            > 0
        )
        target_y, target_x = np.where(reference_target)
        target_short_extent = (
            min(
                int(target_x.max() - target_x.min() + 1),
                int(target_y.max() - target_y.min() + 1),
            )
            if len(target_x)
            else 1
        )
        separation_radius = max(
            1,
            int(round(MAXIMUM_CAD_SUPPORT_RADIUS_FRACTION * target_short_extent)),
        )
        target_exclusion = (
            cv2.dilate(
                reference_target.astype(np.uint8),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * separation_radius + 1, 2 * separation_radius + 1),
                ),
            )
            > 0
        )
        assembly_neighbor_context = reference_assembly & ~target_exclusion
        assembly_neighbor_context = (
            cv2.erode(
                assembly_neighbor_context.astype(np.uint8),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * separation_radius + 1, 2 * separation_radius + 1),
                ),
            )
            > 0
        )
        full_visible_shape = np.rot90(
            visible_shape.astype(np.uint8), quarter_turns % 4
        ).astype(bool)
        relation_shape_variants = [full_visible_shape]
        for component_index in range(1, component_count):
            component_variant = np.rot90(
                component_labels == component_index, quarter_turns % 4
            ).copy()
            if not any(
                np.array_equal(component_variant, prior)
                for prior in relation_shape_variants
            ):
                relation_shape_variants.append(component_variant)
        comparison_variants = [np.rot90(local_visible_shape, quarter_turns % 4).copy()]
        for component_index in selected_components:
            component_variant = np.rot90(
                component_labels == component_index, quarter_turns % 4
            ).copy()
            if not any(
                np.array_equal(component_variant, prior)
                for prior in comparison_variants
            ):
                comparison_variants.append(component_variant)
        references[key] = {
            "visible_shape": comparison_variants,
            # This is the complete target Part-ID silhouette in the CAD render.
            # It deliberately does not use the old photo-space projection to
            # select connected components.  Cross-part relation localization
            # needs every part centroid in one common model-image coordinate
            # system, including targets whose former direct mapping was wrong.
            "relation_visible_shape": full_visible_shape,
            "relation_visible_shape_variants": relation_shape_variants,
            "relation_assembly_shape": rotated_assembly.astype(bool),
            "display_visible_shape": local_visible_shape,
            "complete_shape": np.rot90(complete_shape, quarter_turns % 4).copy(),
            "assembly_neighbor_context": assembly_neighbor_context,
            "audit": {
                "coordinate_domain": "cad_model_render_image",
                "comparison_mode": (
                    "location_scale_invariant_normalized_shape_no_photo_warp"
                ),
                "render_view_id": str(render_view_id),
                "quarter_turns_ccw_for_shape_comparison": quarter_turns % 4,
                "model_image": {
                    "path": str(model_image_path),
                    "sha256": model_image_sha256,
                },
                "model_part_ids": {
                    "path": str(part_ids_path),
                    "sha256": part_ids_sha256,
                },
                "model_visible_part_pixels": visible_pixels,
                "relation_visible_shape_pixels": int(
                    np.count_nonzero(full_visible_shape)
                ),
                "relation_shape_variant_count": len(relation_shape_variants),
                "model_local_visible_part_pixels": int(
                    np.count_nonzero(local_visible_shape)
                ),
                "model_visible_component_count": component_count - 1,
                "model_selected_component_indices": selected_components,
                "model_shape_variant_count": len(comparison_variants),
                "model_component_association": component_audits,
                "component_association_role": (
                    "identify_local_model_image_shape_only_not_photo_boundary"
                ),
                "assembly_context_role": (
                    "preserve_target_position_relative_to_other_visible_cad_parts"
                ),
                "assembly_visible_pixels_in_reference_domain": int(
                    np.count_nonzero(reference_assembly)
                ),
                "assembly_neighbor_context_pixels": int(
                    np.count_nonzero(assembly_neighbor_context)
                ),
                "target_neighbor_separation_radius_pixels": separation_radius,
                "model_complete_part_pixels": int(np.count_nonzero(complete_shape)),
                "model_visible_precision_against_complete": visible_precision,
                "raw_complete_shape": {
                    "path": str(raw_path),
                    "sha256": raw_document.get("sha256"),
                },
                "per_mesh_pose_change_allowed": False,
            },
            "aligned_amodal_sha256": (
                row.get("aligned_amodal_mask", {}).get("sha256")
                if isinstance(row.get("aligned_amodal_mask"), Mapping)
                else None
            ),
        }
    return references, dict(document)


def _resolved_mask_path(root: Path, row: Mapping[str, Any]) -> Path:
    mask = row.get("mask")
    if not isinstance(mask, Mapping) or not isinstance(mask.get("path"), str):
        raise EntitySegRegionError("accepted region has no mask path")
    path = Path(mask["path"])
    if not path.is_absolute():
        path = root / path
    return path.expanduser().resolve(strict=True)


def _sam_selected_shape_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    for box_audit in row.get("box_audits", []):
        if not isinstance(box_audit, Mapping) or box_audit.get("accepted") is not True:
            continue
        selected_index = box_audit.get("selected_candidate_index")
        for candidate in box_audit.get("candidates", []):
            if (
                isinstance(candidate, Mapping)
                and candidate.get("candidate_index") == selected_index
                and candidate.get("accepted") is True
            ):
                return dict(candidate)
    raise EntitySegRegionError(
        "accepted SAM3 region has no selected CAD-shape candidate"
    )


def _load_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != expected_shape:
        raise EntitySegRegionError(f"invalid mask: {path}")
    return mask >= 128


def _sam_aligned_cad_seed(
    seed: np.ndarray,
    sam_row: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replay the bounded shared-camera residual used to prompt SAM3."""
    shared = sam_row.get("view_shared_alignment")
    translation = (
        shared.get("translation_xy_pixels") if isinstance(shared, Mapping) else None
    )
    if (
        not isinstance(translation, list)
        or len(translation) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in translation
        )
        or shared.get("part_specific_translation_allowed") is not False
    ):
        raise EntitySegRegionError("SAM3 shared CAD translation is malformed")
    for box_audit in sam_row.get("box_audits", []):
        if not isinstance(box_audit, Mapping):
            continue
        refinement = box_audit.get("shape_point_refinement")
        if (
            not isinstance(refinement, Mapping)
            or refinement.get("accepted") is not True
        ):
            continue
        prompt = refinement.get("prompt_audit")
        prompt_translation = (
            prompt.get("translation_xy_pixels") if isinstance(prompt, Mapping) else None
        )
        if (
            not isinstance(prompt_translation, list)
            or len(prompt_translation) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in prompt_translation
            )
            or [float(value) for value in prompt_translation]
            != [float(value) for value in translation]
        ):
            raise EntitySegRegionError("SAM3 prompt and shared CAD translations differ")
    matrix = np.asarray(
        [
            [1.0, 0.0, float(translation[0])],
            [0.0, 1.0, float(translation[1])],
        ],
        dtype=np.float32,
    )
    aligned = (
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
    return aligned, {
        "source": "sam3_view_shared_camera_projection",
        "translation_xy_pixels": [float(translation[0]), float(translation[1])],
        "per_mesh_pose_change_allowed": False,
    }


def _entity_aligned_cad_seed(
    seed: np.ndarray,
    entity_row: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replay the selected EntitySeg candidate's bounded camera residual."""
    shared = entity_row.get("view_shared_alignment")
    shared_translation = (
        shared.get("translation_xy_pixels") if isinstance(shared, Mapping) else None
    )
    if (
        not isinstance(shared_translation, list)
        or len(shared_translation) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in shared_translation
        )
        or shared.get("part_specific_translation_allowed") is not False
    ):
        raise EntitySegRegionError("EntitySeg shared CAD translation is malformed")
    selected = entity_row.get("selected_candidate")
    alignment = (
        selected.get("cad_template_alignment")
        if isinstance(selected, Mapping)
        else None
    )
    translation = (
        alignment.get("translation_xy_pixels")
        if isinstance(alignment, Mapping)
        else None
    )
    if translation is None:
        translation = shared_translation
    if (
        not isinstance(translation, list)
        or len(translation) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in translation
        )
        or (
            isinstance(alignment, Mapping)
            and alignment.get("per_mesh_pose_change_allowed") is not False
        )
        or [float(value) for value in translation]
        != [float(value) for value in shared_translation]
    ):
        raise EntitySegRegionError(
            "EntitySeg candidate and shared CAD translations differ"
        )
    matrix = np.asarray(
        [
            [1.0, 0.0, float(translation[0])],
            [0.0, 1.0, float(translation[1])],
        ],
        dtype=np.float32,
    )
    aligned = (
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
    return aligned, {
        "source": "entityseg_view_shared_camera_projection",
        "translation_xy_pixels": [float(translation[0]), float(translation[1])],
        "per_mesh_pose_change_allowed": False,
    }


def _align_with_audit(mask: np.ndarray, audit: Mapping[str, Any]) -> np.ndarray:
    translation = audit.get("translation_xy_pixels")
    if not isinstance(translation, list) or len(translation) != 2:
        raise EntitySegRegionError("aligned CAD audit has no translation")
    matrix = np.asarray(
        [
            [1.0, 0.0, float(translation[0])],
            [0.0, 1.0, float(translation[1])],
        ],
        dtype=np.float32,
    )
    return (
        cv2.warpAffine(
            mask.astype(np.uint8),
            matrix,
            (mask.shape[1], mask.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )


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


def _trim_entity_to_cad_support(
    entity_mask: np.ndarray,
    cad_seed: np.ndarray,
    *,
    maximum_final_to_cad_area_ratio: float = MAXIMUM_FINAL_TO_CAD_AREA_RATIO,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Keep EntitySeg detail only inside a bounded CAD Part-ID support band.

    Entity segmentation is class agnostic and may merge touching CAD parts into
    one visual entity.  Intersecting with the exact CAD projection would throw
    away useful photo-boundary detail, so grow the projection by a small,
    resolution-independent radius.  Select the largest radius whose result is
    still within the caller's bounded CAD-area ratio.  EntitySeg keeps the
    1.25x default; SAM3 uses a stricter 1.15x bound after template prompting.
    """

    entity = np.asarray(entity_mask, dtype=bool)
    seed = np.asarray(cad_seed, dtype=bool)
    if entity.shape != seed.shape or not np.any(entity) or not np.any(seed):
        raise EntitySegRegionError("CAD support trim masks are empty or incompatible")
    seed_y, seed_x = np.where(seed)
    diagonal = float(
        np.hypot(
            int(seed_x.max() - seed_x.min() + 1),
            int(seed_y.max() - seed_y.min() + 1),
        )
    )
    maximum_radius = max(
        MINIMUM_CAD_SUPPORT_RADIUS_PIXELS,
        int(round(MAXIMUM_CAD_SUPPORT_RADIUS_FRACTION * diagonal)),
    )
    seed_pixels = int(np.count_nonzero(seed))
    maximum_pixels = int(np.floor(maximum_final_to_cad_area_ratio * seed_pixels))
    selected_radius = 0
    selected = entity & seed
    for radius in range(maximum_radius + 1):
        if radius == 0:
            support = seed
        else:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )
            support = cv2.dilate(seed.astype(np.uint8), kernel) > 0
        candidate = entity & support
        if int(np.count_nonzero(candidate)) > maximum_pixels:
            break
        selected_radius = radius
        selected = candidate
    selected_pixels = int(np.count_nonzero(selected))
    if selected_pixels == 0:
        raise EntitySegRegionError(
            "CAD support trim removed the complete EntitySeg mask"
        )
    entity_pixels = int(np.count_nonzero(entity))
    return selected, {
        "maximum_support_radius_pixels": maximum_radius,
        "selected_support_radius_pixels": selected_radius,
        "maximum_final_to_cad_area_ratio": maximum_final_to_cad_area_ratio,
        "untrimmed_entity_pixels": entity_pixels,
        "trimmed_entity_pixels": selected_pixels,
        "retained_entity_fraction": selected_pixels / entity_pixels,
        "final_to_cad_area_ratio": selected_pixels / seed_pixels,
    }


def _ellipse_morphology(mask: np.ndarray, *, radius: int, dilate: bool) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return source.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    operation = cv2.dilate if dilate else cv2.erode
    return operation(source.astype(np.uint8), kernel) > 0


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.count_nonzero(left | right))
    if union == 0:
        return 0.0
    return int(np.count_nonzero(left & right)) / union


def _automatic_refinement_radii(visible_seed: np.ndarray) -> tuple[int, int, int]:
    ys, xs = np.where(visible_seed)
    if not len(xs):
        raise EntitySegRegionError("visible CAD seed is empty")
    diagonal = float(
        np.hypot(
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        )
    )
    support_radius = max(
        MINIMUM_CAD_SUPPORT_RADIUS_PIXELS,
        min(
            MAXIMUM_VISIBLE_SUPPORT_RADIUS_PIXELS,
            int(round(MAXIMUM_VISIBLE_SUPPORT_RADIUS_FRACTION * diagonal)),
        ),
    )
    core_radius = max(
        1,
        min(
            MAXIMUM_VISIBLE_CORE_RADIUS_PIXELS,
            int(round(0.008 * diagonal)),
        ),
    )
    occlusion_margin = max(1, min(2, int(round(0.006 * diagonal))))
    return support_radius, core_radius, occlusion_margin


def _mask_principal_axis_degrees(mask: np.ndarray) -> float:
    """Return the unoriented principal-axis angle of a non-empty 2-D mask."""

    binary = np.asarray(mask, dtype=bool)
    ys, xs = np.where(binary)
    if len(xs) < 2:
        return 0.0
    points = np.column_stack((xs, ys)).astype(np.float64)
    covariance = np.cov(points, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    return float(np.degrees(np.arctan2(axis[1], axis[0])))


def _unoriented_angle_delta_degrees(target: float, source: float) -> float:
    """Smallest signed delta for axes whose direction is ambiguous by 180 deg."""

    return float((target - source + 90.0) % 180.0 - 90.0)


def _normalized_image_gradient(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    normalizer = float(np.percentile(gradient, 95.0))
    return np.clip(gradient / max(normalizer, 1e-6), 0.0, 1.0)


def _mask_edge_support(edge_field: np.ndarray, mask: np.ndarray) -> float:
    boundary = (
        cv2.morphologyEx(
            np.asarray(mask, dtype=np.uint8),
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        > 0
    )
    values = edge_field[boundary]
    return float(np.mean(values >= 0.25)) if values.size else 0.0


def _register_visible_template_to_photo(
    *,
    image: np.ndarray,
    visible_seed: np.ndarray,
    amodal_seed: np.ndarray,
    candidate_masks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Locally register a 2-D CAD segmentation template to photo evidence.

    The assembly camera and every CAD mesh remain immutable.  This function
    searches only a scale-derived, in-plane rigid transform of the *mask used
    by segmentation*.  The zero transform participates in the same ranking,
    so a local correction is accepted only when photo edges and independent
    segmentation candidates jointly support it.
    """

    visible = np.asarray(visible_seed, dtype=bool)
    amodal = np.asarray(amodal_seed, dtype=bool)
    candidates = [np.asarray(mask, dtype=bool) for mask in candidate_masks]
    if (
        visible.shape != image.shape[:2]
        or amodal.shape != visible.shape
        or not np.any(visible)
        or not candidates
        or any(mask.shape != visible.shape or not np.any(mask) for mask in candidates)
    ):
        raise EntitySegRegionError("local CAD/photo registration inputs are invalid")

    support_radius, _core_radius, _occlusion_margin = _automatic_refinement_radii(
        visible
    )
    ys, xs = np.where(visible)
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())
    diagonal = float(
        np.hypot(
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        )
    )
    maximum_translation = support_radius
    maximum_rotation = float(
        np.degrees(np.arctan2(float(maximum_translation), max(0.5 * diagonal, 1.0)))
    )

    union = np.logical_or.reduce(candidates)
    union_y, union_x = np.where(union)
    centroid_dx = int(round(float(union_x.mean()) - centroid_x))
    centroid_dy = int(round(float(union_y.mean()) - centroid_y))
    step = max(1, maximum_translation // 3)
    translation_values = set(range(-maximum_translation, maximum_translation + 1, step))
    translation_values.update({-maximum_translation, 0, maximum_translation})
    if abs(centroid_dx) <= maximum_translation:
        translation_values.add(centroid_dx)
    if abs(centroid_dy) <= maximum_translation:
        translation_values.add(centroid_dy)

    visible_angle = _mask_principal_axis_degrees(visible)
    angle_values = {
        -maximum_rotation,
        -0.5 * maximum_rotation,
        0.0,
        0.5 * maximum_rotation,
        maximum_rotation,
    }
    for candidate in candidates:
        delta = _unoriented_angle_delta_degrees(
            _mask_principal_axis_degrees(candidate), visible_angle
        )
        clipped = float(np.clip(delta, -maximum_rotation, maximum_rotation))
        angle_values.update({clipped, 0.5 * clipped})

    margin = maximum_translation + 3
    all_y = np.concatenate([ys, union_y])
    all_x = np.concatenate([xs, union_x])
    top = max(0, int(all_y.min()) - margin)
    bottom = min(visible.shape[0], int(all_y.max()) + margin + 1)
    left = max(0, int(all_x.min()) - margin)
    right = min(visible.shape[1], int(all_x.max()) + margin + 1)
    local_visible = visible[top:bottom, left:right]
    local_amodal = amodal[top:bottom, left:right]
    local_union = union[top:bottom, left:right]
    local_edges = _normalized_image_gradient(image)[top:bottom, left:right]
    center = (centroid_x - left, centroid_y - top)

    def evaluate(
        dx: int, dy: int, angle: float
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        matrix[0, 2] += dx
        matrix[1, 2] += dy
        size = (local_visible.shape[1], local_visible.shape[0])
        transformed_visible = (
            cv2.warpAffine(
                local_visible.astype(np.uint8),
                matrix,
                size,
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            > 0
        )
        transformed_amodal = (
            cv2.warpAffine(
                local_amodal.astype(np.uint8),
                matrix,
                size,
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            > 0
        )
        pixels = int(np.count_nonzero(transformed_visible))
        candidate_support = int(
            np.count_nonzero(transformed_visible & local_union)
        ) / max(pixels, 1)
        edge_support = _mask_edge_support(local_edges, transformed_visible)
        score = math.sqrt(max(candidate_support, 1e-9) * max(edge_support, 1e-9))
        return (
            transformed_visible,
            transformed_amodal,
            {
                "candidate_support": candidate_support,
                "image_edge_support": edge_support,
                "registration_score": score,
            },
        )

    initial_visible, initial_amodal, initial_metrics = evaluate(0, 0, 0.0)
    best_key = (
        float(initial_metrics["registration_score"]),
        float(initial_metrics["candidate_support"]),
        float(initial_metrics["image_edge_support"]),
        0.0,
        0.0,
    )
    best = (0, 0, 0.0, initial_visible, initial_amodal, initial_metrics)
    evaluated = 0
    for angle in sorted(angle_values):
        for dy in sorted(translation_values):
            for dx in sorted(translation_values):
                evaluated += 1
                transformed_visible, transformed_amodal, metrics = evaluate(
                    dx, dy, angle
                )
                # A local registration may not exchange one independent
                # source of evidence for another.  Requiring a Pareto
                # non-regression against the zero transform prevents a strong
                # but wrong neighbouring image edge from pulling the CAD mask
                # away from the segmenters (and vice versa).
                if (
                    float(metrics["candidate_support"])
                    < float(initial_metrics["candidate_support"]) - 1e-12
                    or float(metrics["image_edge_support"])
                    < float(initial_metrics["image_edge_support"]) - 1e-12
                ):
                    continue
                displacement = float(np.hypot(dx, dy))
                key = (
                    float(metrics["registration_score"]),
                    float(metrics["candidate_support"]),
                    float(metrics["image_edge_support"]),
                    -displacement,
                    -abs(angle),
                )
                if key > best_key:
                    best_key = key
                    best = (
                        dx,
                        dy,
                        float(angle),
                        transformed_visible,
                        transformed_amodal,
                        metrics,
                    )

    dx, dy, angle, selected_visible, selected_amodal, selected_metrics = best
    accepted = (
        bool(dx or dy or abs(angle) > 1e-9)
        and float(selected_metrics["registration_score"])
        > float(initial_metrics["registration_score"]) + 1e-12
    )
    if not accepted:
        dx, dy, angle = 0, 0, 0.0
        selected_visible = initial_visible
        selected_amodal = initial_amodal
        selected_metrics = initial_metrics

    output_visible = np.zeros_like(visible)
    output_amodal = np.zeros_like(amodal)
    output_visible[top:bottom, left:right] = selected_visible
    output_amodal[top:bottom, left:right] = selected_amodal
    if not np.any(output_visible) or not np.any(output_amodal):
        raise EntitySegRegionError("local CAD/photo registration removed the template")
    return (
        output_visible,
        output_amodal,
        {
            "method": "bounded_2d_cad_template_registration_to_photo_evidence",
            "accepted": accepted,
            "translation_xy_pixels": [int(dx), int(dy)],
            "rotation_degrees": float(angle),
            "maximum_translation_pixels": int(maximum_translation),
            "maximum_rotation_degrees": maximum_rotation,
            "search_bound_source": "visible_part_bbox_scale",
            "selection_contract": (
                "pareto_nonregression_of_photo_edges_and_candidate_support"
            ),
            "evaluated_transform_count": evaluated,
            "initial_metrics": initial_metrics,
            "selected_metrics": selected_metrics,
            "transformed_object": "reference_view_segmentation_template_only",
            "cad_mesh_transform_changed": False,
            "assembly_camera_changed": False,
            "per_mesh_pose_change_allowed": False,
        },
    )


def _full_resolution_similarity_fallback(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map tiny masks without downsampling either coordinate domain."""

    source_selected = np.asarray(source, dtype=bool)
    target_selected = np.asarray(target, dtype=bool)
    source_y, source_x = np.where(source_selected)
    target_y, target_x = np.where(target_selected)
    if not len(source_x) or not len(target_x):
        raise EntitySegRegionError(
            "tiny-mask similarity fallback received an empty mask"
        )

    def centroid_and_axis(
        xs: np.ndarray, ys: np.ndarray
    ) -> tuple[tuple[float, float], float]:
        centroid = (float(xs.mean()), float(ys.mean()))
        centered = np.column_stack((xs - centroid[0], ys - centroid[1]))
        covariance = centered.T @ centered / max(len(centered), 1)
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(values))]
        angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
        return centroid, angle

    source_centroid, source_angle = centroid_and_axis(source_x, source_y)
    target_centroid, target_angle = centroid_and_axis(target_x, target_y)
    area_scale = math.sqrt(float(len(target_x)) / float(len(source_x)))
    rotations = {
        0.0,
        target_angle - source_angle,
        source_angle - target_angle,
        target_angle - source_angle + 180.0,
        target_angle - source_angle - 180.0,
    }
    scales = {
        float(np.clip(area_scale * multiplier, 0.35, 3.0))
        for multiplier in (0.85, 1.0, 1.15)
    }
    audited: list[tuple[tuple[float, float, float], np.ndarray, np.ndarray]] = []
    for scale in sorted(scales):
        for rotation in sorted(rotations):
            matrix = cv2.getRotationMatrix2D(source_centroid, rotation, scale)
            matrix[0, 2] += target_centroid[0] - source_centroid[0]
            matrix[1, 2] += target_centroid[1] - source_centroid[1]
            warped = (
                cv2.warpAffine(
                    source_selected.astype(np.uint8),
                    matrix,
                    (target_selected.shape[1], target_selected.shape[0]),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                > 0
            )
            intersection = int(np.count_nonzero(warped & target_selected))
            union = int(np.count_nonzero(warped | target_selected))
            iou = intersection / max(union, 1)
            audited.append(
                (
                    (iou, -abs(scale - area_scale), -abs(rotation)),
                    warped,
                    matrix,
                )
            )
    audited.sort(key=lambda item: item[0], reverse=True)
    _key, registered, matrix = audited[0]
    return registered.astype(np.uint8) * 255, {
        "method": "full_resolution_tiny_mask_similarity_fallback",
        "affine_2x3": [[float(value) for value in row] for row in matrix],
        "iou": float(audited[0][0][0]),
        "source_pixels": int(len(source_x)),
        "target_pixels": int(len(target_x)),
        "downsampling_applied": False,
    }


def _register_model_shape_to_photo(
    *,
    image: np.ndarray,
    visible_seed: np.ndarray,
    model_shape_variants: list[np.ndarray],
    candidate_masks: list[np.ndarray],
    assembly_neighbor_context: np.ndarray | None = None,
    complete_target_shape_variants: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Register the model-image Part-ID shape as a photo-space proposal only."""

    visible = np.asarray(visible_seed, dtype=bool)
    variants = [np.asarray(value, dtype=bool) for value in model_shape_variants]
    candidates = [np.asarray(value, dtype=bool) for value in candidate_masks]
    neighbor_context = (
        np.asarray(assembly_neighbor_context, dtype=bool)
        if assembly_neighbor_context is not None
        else np.zeros_like(visible)
    )
    if (
        visible.shape != image.shape[:2]
        or not np.any(visible)
        or not variants
        or any(value.ndim != 2 or not np.any(value) for value in variants)
        or not candidates
        or any(
            value.shape != visible.shape or not np.any(value) for value in candidates
        )
        or neighbor_context.shape != visible.shape
    ):
        raise EntitySegRegionError("model-shape photo registration inputs are invalid")

    candidate_union = np.logical_or.reduce(candidates)
    edge_field = _normalized_image_gradient(image)
    audited: list[
        tuple[tuple[float, float, float, int], np.ndarray, dict[str, Any]]
    ] = []
    # Variant zero is the union of every model-image component associated with
    # the local observation. Component-only variants remain scoring aids; a
    # final proposal must not drop another visible piece of the same Part-ID.
    for variant_index, variant in enumerate(variants[:1]):
        try:
            mapped_raw, base_registration = _register_similarity_mask(
                variant.astype(np.uint8) * 255,
                visible.astype(np.uint8) * 255,
            )
        except PartIdProjectionError:
            mapped_raw, base_registration = _full_resolution_similarity_fallback(
                variant,
                visible,
            )
        mapped = mapped_raw > 0
        ys, xs = np.where(mapped)
        if not len(xs):
            continue
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        diagonal = float(np.hypot(width, height))
        short_extent = max(1, min(width, height))
        maximum_translation = max(2, int(round(0.08 * short_extent)))
        maximum_rotation = float(
            np.degrees(np.arctan2(float(maximum_translation), max(0.5 * diagonal, 1.0)))
        )
        maximum_scale_delta = min(
            0.25, 3.0 * float(maximum_translation) / float(short_extent)
        )
        translations = sorted(
            {
                int(round(value))
                for value in np.linspace(-maximum_translation, maximum_translation, 7)
            }
        )
        rotations = [
            float(value)
            for value in np.linspace(-maximum_rotation, maximum_rotation, 7)
        ]
        scales = [
            float(value)
            for value in np.linspace(
                1.0 - maximum_scale_delta,
                1.0 + maximum_scale_delta,
                7,
            )
        ]

        union_y, union_x = np.where(mapped | candidate_union)
        margin = maximum_translation + 3
        top = max(0, int(union_y.min()) - margin)
        bottom = min(visible.shape[0], int(union_y.max()) + margin + 1)
        left = max(0, int(union_x.min()) - margin)
        right = min(visible.shape[1], int(union_x.max()) + margin + 1)
        local_mapped = mapped[top:bottom, left:right]
        local_candidates = candidate_union[top:bottom, left:right]
        local_edges = edge_field[top:bottom, left:right]
        local_neighbors = neighbor_context[top:bottom, left:right]
        center = (float(xs.mean()) - left, float(ys.mean()) - top)
        output_size = (local_mapped.shape[1], local_mapped.shape[0])

        top_candidates: list[
            tuple[
                tuple[float, float, float, float, float],
                np.ndarray,
                int,
                int,
                float,
                float,
                dict[str, float],
            ]
        ] = []
        evaluated = 0
        for scale in scales:
            for rotation in rotations:
                for dy in translations:
                    for dx in translations:
                        evaluated += 1
                        matrix = cv2.getRotationMatrix2D(center, rotation, scale)
                        matrix[0, 2] += dx
                        matrix[1, 2] += dy
                        transformed = (
                            cv2.warpAffine(
                                local_mapped.astype(np.uint8),
                                matrix,
                                output_size,
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0,
                            )
                            > 0
                        )
                        pixels = int(np.count_nonzero(transformed))
                        if pixels == 0:
                            continue
                        candidate_support = (
                            int(np.count_nonzero(transformed & local_candidates))
                            / pixels
                        )
                        edge_support = _mask_edge_support(local_edges, transformed)
                        neighbor_overlap = (
                            int(np.count_nonzero(transformed & local_neighbors))
                            / pixels
                        )
                        shift_fraction = float(np.hypot(dx, dy)) / max(
                            float(np.hypot(maximum_translation, maximum_translation)),
                            1.0,
                        )
                        # Assembly context constrains the centroid. Rotation
                        # and uniform scale correct the target silhouette
                        # without moving it onto another model part.
                        assembly_position_score = float(
                            np.exp(-0.25 * shift_fraction * shift_fraction)
                        )
                        neighbor_clearance = max(1e-9, 1.0 - neighbor_overlap)
                        score = float(
                            np.power(
                                max(candidate_support, 1e-9)
                                * max(edge_support, 1e-9)
                                * neighbor_clearance,
                                1.0 / 3.0,
                            )
                        )
                        key = (
                            score,
                            candidate_support,
                            edge_support,
                            assembly_position_score,
                            -abs(scale - 1.0),
                        )
                        metrics = {
                            "candidate_support": candidate_support,
                            "image_edge_support": edge_support,
                            "assembly_position_score": assembly_position_score,
                            "assembly_centroid_shift_fraction": shift_fraction,
                            "assembly_neighbor_overlap_fraction": neighbor_overlap,
                            "assembly_neighbor_clearance_score": neighbor_clearance,
                            "registration_score": score,
                        }
                        if len(top_candidates) < 24 or key > top_candidates[-1][0]:
                            top_candidates.append(
                                (
                                    key,
                                    transformed,
                                    dx,
                                    dy,
                                    rotation,
                                    scale,
                                    metrics,
                                )
                            )
                            top_candidates.sort(key=lambda item: item[0], reverse=True)
                            del top_candidates[24:]
        if not top_candidates:
            continue
        reranked: list[
            tuple[
                tuple[float, float, float, float],
                np.ndarray,
                int,
                int,
                float,
                float,
                dict[str, float],
                dict[str, Any],
            ]
        ] = []
        for (
            _registration_key,
            local_candidate,
            candidate_dx,
            candidate_dy,
            candidate_rotation,
            candidate_scale,
            candidate_metrics,
        ) in top_candidates:
            full_candidate = np.zeros_like(visible)
            full_candidate[top:bottom, left:right] = local_candidate
            agreement = _best_model_domain_shape_agreement(full_candidate, variant)
            model_score = float(agreement["model_shape_score"])
            joint_score = float(
                np.power(
                    max(candidate_metrics["candidate_support"], 1e-9)
                    * max(candidate_metrics["image_edge_support"], 1e-9)
                    * max(model_score, 1e-9)
                    * max(
                        candidate_metrics["assembly_neighbor_clearance_score"],
                        1e-9,
                    ),
                    0.25,
                )
            )
            reranked.append(
                (
                    (
                        joint_score,
                        model_score,
                        float(candidate_metrics["registration_score"]),
                        float(candidate_metrics["assembly_position_score"]),
                    ),
                    full_candidate,
                    candidate_dx,
                    candidate_dy,
                    candidate_rotation,
                    candidate_scale,
                    candidate_metrics,
                    agreement,
                )
            )
        reranked.sort(key=lambda item: item[0], reverse=True)
        (
            _joint_key,
            selected,
            dx,
            dy,
            rotation,
            scale,
            selected_metrics,
            model_agreement,
        ) = reranked[0]
        selected_metrics = dict(selected_metrics)
        selected_metrics["joint_model_edge_candidate_score"] = float(reranked[0][0][0])

        global_center = (float(xs.mean()), float(ys.mean()))
        local_matrix = cv2.getRotationMatrix2D(global_center, rotation, scale)
        local_matrix[0, 2] += dx
        local_matrix[1, 2] += dy
        base_matrix = np.asarray(base_registration["affine_2x3"], dtype=np.float64)
        total_matrix = (
            np.vstack((local_matrix, [0.0, 0.0, 1.0]))
            @ np.vstack((base_matrix, [0.0, 0.0, 1.0]))
        )[:2]
        model_score = float(model_agreement["model_shape_score"])
        audit = {
            "variant_index": variant_index,
            "base_model_to_reference_registration": base_registration,
            "local_translation_xy_pixels": [int(dx), int(dy)],
            "local_rotation_degrees": float(rotation),
            "local_uniform_scale": float(scale),
            "maximum_translation_pixels": maximum_translation,
            "maximum_rotation_degrees": maximum_rotation,
            "maximum_scale_delta": maximum_scale_delta,
            "evaluated_transform_count": evaluated,
            "selected_metrics": selected_metrics,
            "model_domain_shape_score": model_score,
            "assembly_context_used": assembly_neighbor_context is not None,
            "model_to_photo_affine_2x3": [
                [float(value) for value in row] for row in total_matrix
            ],
        }
        audited.append(
            (
                (
                    float(selected_metrics["joint_model_edge_candidate_score"]),
                    model_score,
                    float(selected_metrics["candidate_support"]),
                    -variant_index,
                ),
                selected,
                audit,
            )
        )
    if not audited:
        raise EntitySegRegionError("no model-shape photo proposal could be registered")
    audited.sort(key=lambda item: item[0], reverse=True)
    selected_mask = audited[0][1]
    selected_audit = audited[0][2]
    selected_audit.update(
        {
            "method": "model_image_part_id_similarity_registration_to_photo",
            "model_shape_variant_count": len(variants),
            "proposal_component_policy": (
                "complete_target_part_id_union_plus_all_model_components_without_"
                "photo_mapping"
                if complete_target_shape_variants
                else "union_of_all_model_components_associated_with_local_observation"
            ),
            "transformed_object": "model_image_part_id_mask_proposal_only",
            "cad_mesh_transform_changed": False,
            "assembly_camera_changed": False,
            "per_mesh_pose_change_allowed": False,
        }
    )
    return selected_mask, selected_audit


def _refinement_metrics(
    *,
    image: np.ndarray,
    mask: np.ndarray,
    visible_seed: np.ndarray,
    amodal_seed: np.ndarray,
    candidate_masks: list[np.ndarray],
    model_visible_shape: np.ndarray | list[np.ndarray] | None = None,
) -> dict[str, float | int]:
    mask_pixels = int(np.count_nonzero(mask))
    visible_pixels = int(np.count_nonzero(visible_seed))
    amodal_pixels = int(np.count_nonzero(amodal_seed))
    visible_intersection = int(np.count_nonzero(mask & visible_seed))
    amodal_intersection = int(np.count_nonzero(mask & amodal_seed))
    candidate_agreements = [_mask_iou(mask, candidate) for candidate in candidate_masks]
    edge = _boundary_metrics(image, mask)
    visible_iou = _mask_iou(mask, visible_seed)
    mean_candidate_iou = float(np.mean(candidate_agreements))
    edge_support = float(edge["image_edge_support_fraction_025"])
    output: dict[str, float | int] = {
        "mask_pixels": mask_pixels,
        "visible_seed_pixels": visible_pixels,
        "visible_seed_recall": visible_intersection / max(visible_pixels, 1),
        "visible_seed_precision": visible_intersection / max(mask_pixels, 1),
        "visible_seed_iou": visible_iou,
        "final_to_visible_area_ratio": mask_pixels / max(visible_pixels, 1),
        "amodal_candidate_precision": amodal_intersection / max(mask_pixels, 1),
        "final_to_amodal_area_ratio": mask_pixels / max(amodal_pixels, 1),
        "mean_prior_candidate_iou": mean_candidate_iou,
        "minimum_prior_candidate_iou": min(candidate_agreements),
        "image_edge_support": edge_support,
        "normalized_image_edge_mean": float(edge["normalized_image_edge_mean"]),
    }
    if model_visible_shape is None:
        # Backward-compatible diagnostic mode.  The production lane supplies
        # a model-domain silhouette and never uses this objective.
        objective = float(
            np.cbrt(
                max(edge_support, 1e-9)
                * max(visible_iou, 1e-9)
                * max(mean_candidate_iou, 1e-9)
            )
        )
        output["objective_geometric_mean"] = objective
        return output

    shape = _best_model_domain_shape_agreement(mask, model_visible_shape)
    model_shape_iou = float(shape["cad_shape_iou"])
    model_shape_minimum_precision_recall = float(
        shape["cad_shape_minimum_precision_recall"]
    )
    model_shape_score = float(shape["model_shape_score"])
    # The projected visible mask is deliberately absent from the ranking
    # objective.  It only supplies ROI/visibility safety below.  Equal-weight
    # photo edge, model-image shape, and prior-candidate agreement make the
    # model silhouette measurably authoritative without a tuned CAD gain.
    objective = float(
        np.cbrt(
            max(edge_support, 1e-9)
            * max(model_shape_score, 1e-9)
            * max(mean_candidate_iou, 1e-9)
        )
    )
    output.update(
        {
            "model_domain_shape_iou": model_shape_iou,
            "model_domain_shape_minimum_precision_recall": (
                model_shape_minimum_precision_recall
            ),
            "model_domain_shape_overlap_smaller": float(
                shape["cad_shape_overlap_smaller"]
            ),
            "model_domain_shape_rotation_degrees": float(
                shape["cad_shape_rotation_degrees"]
            ),
            "model_domain_shape_variant_index": int(shape["model_shape_variant_index"]),
            "model_domain_shape_variant_count": int(shape["model_shape_variant_count"]),
            "model_domain_shape_score": model_shape_score,
            "model_domain_shape_boundary_precision": float(
                shape["model_shape_boundary_precision"]
            ),
            "model_domain_shape_boundary_recall": float(
                shape["model_shape_boundary_recall"]
            ),
            "model_domain_shape_boundary_score": float(
                shape["model_shape_boundary_score"]
            ),
            "model_domain_shape_location_invariant": True,
            "objective_geometric_mean": objective,
        }
    )
    return output


def _best_model_domain_shape_agreement(
    candidate: np.ndarray,
    model_visible_shape: np.ndarray | list[np.ndarray],
) -> dict[str, Any]:
    variants = (
        [np.asarray(model_visible_shape, dtype=bool)]
        if isinstance(model_visible_shape, np.ndarray)
        else [np.asarray(value, dtype=bool) for value in model_visible_shape]
    )
    if not variants or any(value.ndim != 2 or not np.any(value) for value in variants):
        raise EntitySegRegionError("model-domain shape variants are invalid")
    audited: list[tuple[tuple[float, float, float, int], dict[str, Any]]] = []
    for index, variant in enumerate(variants):
        shape = dict(_normalized_shape_agreement(candidate, variant))
        iou = float(shape["cad_shape_iou"])
        minimum_precision_recall = float(shape["cad_shape_minimum_precision_recall"])
        boundary = _normalized_boundary_agreement(
            candidate,
            variant,
            rotation_degrees=float(shape["cad_shape_rotation_degrees"]),
        )
        score = float(
            np.cbrt(
                max(iou, 1e-9)
                * max(minimum_precision_recall, 1e-9)
                * max(float(boundary["score"]), 1e-9)
            )
        )
        shape["model_shape_boundary_precision"] = boundary["precision"]
        shape["model_shape_boundary_recall"] = boundary["recall"]
        shape["model_shape_boundary_score"] = boundary["score"]
        shape["model_shape_score"] = score
        audited.append(((score, iou, minimum_precision_recall, -index), shape))
    audited.sort(key=lambda item: item[0], reverse=True)
    selected = audited[0][1]
    selected["model_shape_variant_index"] = -audited[0][0][3]
    selected["model_shape_variant_count"] = len(variants)
    return selected


def _normalize_shape_mask(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not np.any(binary):
        raise EntitySegRegionError("model-domain boundary shape is empty")
    ys, xs = np.where(binary)
    crop = binary[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.uint8)
    margin = max(4, MODEL_SHAPE_NORMALIZATION_SIZE // 12)
    available = MODEL_SHAPE_NORMALIZATION_SIZE - 2 * margin
    scale = available / max(crop.shape)
    width = max(1, int(round(crop.shape[1] * scale)))
    height = max(1, int(round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros(
        (MODEL_SHAPE_NORMALIZATION_SIZE, MODEL_SHAPE_NORMALIZATION_SIZE),
        dtype=np.uint8,
    )
    left = (MODEL_SHAPE_NORMALIZATION_SIZE - width) // 2
    top = (MODEL_SHAPE_NORMALIZATION_SIZE - height) // 2
    canvas[top : top + height, left : left + width] = resized
    return canvas > 0


def _normalized_boundary_agreement(
    candidate: np.ndarray,
    model_shape: np.ndarray,
    *,
    rotation_degrees: float,
) -> dict[str, float]:
    candidate_normalized = _normalize_shape_mask(candidate)
    model_normalized = _normalize_shape_mask(model_shape)
    center = (
        (MODEL_SHAPE_NORMALIZATION_SIZE - 1) * 0.5,
        (MODEL_SHAPE_NORMALIZATION_SIZE - 1) * 0.5,
    )
    matrix = cv2.getRotationMatrix2D(center, rotation_degrees, 1.0)
    model_normalized = (
        cv2.warpAffine(
            model_normalized.astype(np.uint8),
            matrix,
            (MODEL_SHAPE_NORMALIZATION_SIZE, MODEL_SHAPE_NORMALIZATION_SIZE),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    candidate_boundary = candidate_normalized & ~(
        cv2.erode(candidate_normalized.astype(np.uint8), kernel) > 0
    )
    model_boundary = model_normalized & ~(
        cv2.erode(model_normalized.astype(np.uint8), kernel) > 0
    )
    tolerance = max(1, MODEL_SHAPE_NORMALIZATION_SIZE // 48)
    support_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1)
    )
    candidate_support = (
        cv2.dilate(candidate_boundary.astype(np.uint8), support_kernel) > 0
    )
    model_support = cv2.dilate(model_boundary.astype(np.uint8), support_kernel) > 0
    precision = int(np.count_nonzero(candidate_boundary & model_support)) / max(
        int(np.count_nonzero(candidate_boundary)), 1
    )
    recall = int(np.count_nonzero(model_boundary & candidate_support)) / max(
        int(np.count_nonzero(model_boundary)), 1
    )
    return {
        "precision": precision,
        "recall": recall,
        "score": math.sqrt(max(precision, 0.0) * max(recall, 0.0)),
    }


def _rank_model_guided_initializers(
    *,
    image: np.ndarray,
    candidates: list[tuple[str, np.ndarray]],
    model_visible_shape: np.ndarray | list[np.ndarray],
) -> tuple[str, list[dict[str, float | str]]]:
    """Choose only the optimization initializer, never the final mask."""

    audited: list[tuple[tuple[float, float, str], dict[str, float | str]]] = []
    for source, candidate in candidates:
        boundary = _boundary_metrics(image, candidate)
        shape = _best_model_domain_shape_agreement(candidate, model_visible_shape)
        iou = float(shape["cad_shape_iou"])
        minimum_precision_recall = float(shape["cad_shape_minimum_precision_recall"])
        shape_score = float(shape["model_shape_score"])
        edge_support = float(boundary["image_edge_support_fraction_025"])
        initializer_score = math.sqrt(max(shape_score, 1e-9) * max(edge_support, 1e-9))
        row: dict[str, float | str] = {
            "source": source,
            "model_domain_shape_iou": iou,
            "model_domain_shape_minimum_precision_recall": (minimum_precision_recall),
            "model_domain_shape_score": shape_score,
            "model_domain_shape_boundary_score": float(
                shape["model_shape_boundary_score"]
            ),
            "model_domain_shape_variant_index": float(
                shape["model_shape_variant_index"]
            ),
            "image_edge_support": edge_support,
            "initializer_score": initializer_score,
        }
        audited.append(((initializer_score, shape_score, source), row))
    if not audited:
        raise EntitySegRegionError("model-guided initializer ranking has no candidates")
    audited.sort(key=lambda item: item[0], reverse=True)
    return str(audited[0][1]["source"]), [item[1] for item in audited]


def _iterative_shape_guided_refinement(
    *,
    image: np.ndarray,
    visible_seed: np.ndarray,
    amodal_seed: np.ndarray | None,
    model_visible_shape: np.ndarray | list[np.ndarray] | None = None,
    assembly_neighbor_context: np.ndarray | None = None,
    candidate_masks: list[tuple[str, np.ndarray]],
    primary_candidate_source: str,
    complete_target_shape_variants: bool = False,
    _enable_local_registration: bool = True,
    _enable_model_shape_proposal: bool = True,
) -> tuple[np.ndarray, dict[str, Any], dict[str, float | int]]:
    """Optimize one visible instance without choosing one model mask verbatim.

    The model-render Part-ID silhouette is the shape authority and remains in
    model image coordinates.  The assembled CAD projection supplies only the
    current-view ROI/visibility safety contract.  SAM3 and EntitySeg are prior
    estimates.  GrabCut updates the photo boundary inside a scale-derived band;
    each iterate is compared with a location/scale-invariant model shape rather
    than with a model mask pasted into the photograph.
    """

    visible = np.asarray(visible_seed, dtype=bool)
    amodal = (
        np.asarray(amodal_seed, dtype=bool)
        if amodal_seed is not None
        else visible.copy()
    )
    if image.shape[:2] != visible.shape or amodal.shape != visible.shape:
        raise EntitySegRegionError("shape-guided refinement inputs are incompatible")
    neighbor_context = (
        np.asarray(assembly_neighbor_context, dtype=bool)
        if assembly_neighbor_context is not None
        else None
    )
    if neighbor_context is not None and neighbor_context.shape != visible.shape:
        raise EntitySegRegionError("assembly context is incompatible with the photo")
    if not candidate_masks or not np.any(visible) or not np.any(amodal):
        raise EntitySegRegionError("shape-guided refinement has no usable authority")
    if model_visible_shape is not None:
        variants = (
            [np.asarray(model_visible_shape, dtype=bool)]
            if isinstance(model_visible_shape, np.ndarray)
            else [np.asarray(value, dtype=bool) for value in model_visible_shape]
        )
        if not variants or any(
            value.ndim != 2 or not np.any(value) for value in variants
        ):
            raise EntitySegRegionError("model-domain target shape is empty")
        model_visible_shape = variants
    candidate_by_source = {
        source: np.asarray(mask, dtype=bool) for source, mask in candidate_masks
    }
    if len(candidate_by_source) != len(candidate_masks):
        raise EntitySegRegionError("duplicate shape-guided candidate source")
    if primary_candidate_source not in candidate_by_source:
        raise EntitySegRegionError("primary candidate is absent from refinement inputs")
    if any(
        mask.shape != visible.shape or not np.any(mask)
        for mask in candidate_by_source.values()
    ):
        raise EntitySegRegionError(
            "shape-guided candidate mask is empty or incompatible"
        )

    if model_visible_shape is not None and _enable_model_shape_proposal:
        baseline = _iterative_shape_guided_refinement(
            image=image,
            visible_seed=visible,
            amodal_seed=amodal,
            model_visible_shape=model_visible_shape,
            assembly_neighbor_context=neighbor_context,
            candidate_masks=list(candidate_by_source.items()),
            primary_candidate_source=primary_candidate_source,
            complete_target_shape_variants=complete_target_shape_variants,
            _enable_local_registration=True,
            _enable_model_shape_proposal=False,
        )
        model_proposal, model_registration = _register_model_shape_to_photo(
            image=image,
            visible_seed=visible,
            model_shape_variants=model_visible_shape,
            candidate_masks=list(candidate_by_source.values()),
            assembly_neighbor_context=neighbor_context,
            complete_target_shape_variants=complete_target_shape_variants,
        )
        candidate_values = list(candidate_by_source.values())
        candidate_union = np.logical_or.reduce(candidate_values)
        proposal_pixels = int(np.count_nonzero(model_proposal))
        visible_pixels = int(np.count_nonzero(visible))
        proposal_candidate_support = int(
            np.count_nonzero(model_proposal & candidate_union)
        ) / max(proposal_pixels, 1)
        seed_candidate_support = int(np.count_nonzero(visible & candidate_union)) / max(
            visible_pixels, 1
        )
        proposal_metrics = _refinement_metrics(
            image=image,
            mask=model_proposal,
            visible_seed=model_proposal,
            amodal_seed=model_proposal,
            candidate_masks=candidate_values,
            model_visible_shape=model_visible_shape,
        )
        baseline_metrics = baseline[1]["final_metrics"]
        proposal_area_ratio = proposal_pixels / max(visible_pixels, 1)
        seed_neighbor_overlap = (
            int(np.count_nonzero(visible & neighbor_context)) / max(visible_pixels, 1)
            if neighbor_context is not None
            else 0.0
        )
        proposal_neighbor_overlap = (
            int(np.count_nonzero(model_proposal & neighbor_context))
            / max(proposal_pixels, 1)
            if neighbor_context is not None
            else 0.0
        )
        rejection_reasons: list[str] = []
        if (
            float(proposal_metrics["image_edge_support"])
            < float(baseline_metrics["image_edge_support"]) - 1e-12
        ):
            rejection_reasons.append("photo_edge_support_did_not_improve")
        if (
            float(proposal_metrics["model_domain_shape_score"])
            < float(baseline_metrics["model_domain_shape_score"]) - 1e-12
        ):
            rejection_reasons.append("model_domain_shape_did_not_improve")
        if proposal_candidate_support < seed_candidate_support - 1e-12:
            rejection_reasons.append("candidate_support_below_cad_location_floor")
        if not (
            MINIMUM_REFINED_TO_VISIBLE_AREA_RATIO
            <= proposal_area_ratio
            <= MAXIMUM_FINAL_TO_CAD_AREA_RATIO
        ):
            rejection_reasons.append("model_shape_proposal_area_outside_cad_bound")
        alias_tolerance = 1.0 / max(proposal_pixels, 1)
        if proposal_neighbor_overlap > seed_neighbor_overlap + alias_tolerance:
            rejection_reasons.append("assembly_neighbor_context_overlap_increased")
        if not (
            float(proposal_metrics["image_edge_support"])
            > float(baseline_metrics["image_edge_support"]) + 1e-12
            or float(proposal_metrics["model_domain_shape_score"])
            > float(baseline_metrics["model_domain_shape_score"]) + 1e-12
        ):
            rejection_reasons.append("model_shape_proposal_did_not_strictly_improve")

        selected = baseline
        selected_branch = "iterative_reference_projection_baseline"
        if not rejection_reasons:
            selected_audit = dict(baseline[1])
            iteration_audits = list(selected_audit.get("iterations", []))
            iteration_audits.append(
                {
                    "iteration": len(iteration_audits) + 1,
                    "lane": "model_image_shape_similarity_proposal",
                    "lane_iteration": 0,
                    "accepted": True,
                    "reason_codes": [],
                    "metrics": proposal_metrics,
                    "changed_pixels_from_previous_iteration": int(
                        np.count_nonzero(model_proposal ^ baseline[0])
                    ),
                }
            )
            selected_audit.update(
                {
                    "selected_iteration": len(iteration_audits),
                    "selected_optimization_lane": (
                        "model_image_shape_similarity_proposal"
                    ),
                    "selected_lane_iteration": 0,
                    "executed_iteration_count": len(iteration_audits),
                    "final_metrics": proposal_metrics,
                    "final_changed_pixels_from_initial": int(
                        np.count_nonzero(model_proposal ^ baseline[0])
                    ),
                    "iterations": iteration_audits,
                    "current_view_visibility_authority": (
                        "model_image_part_id_shape_registered_from_sealed_cad_location"
                    ),
                }
            )
            primary_unbounded = candidate_by_source[primary_candidate_source]
            support_audit: dict[str, float | int] = dict(baseline[2])
            support_audit.update(
                {
                    "untrimmed_entity_pixels": int(np.count_nonzero(primary_unbounded)),
                    "trimmed_entity_pixels": proposal_pixels,
                    "retained_entity_fraction": int(
                        np.count_nonzero(model_proposal & primary_unbounded)
                    )
                    / max(int(np.count_nonzero(primary_unbounded)), 1),
                    "final_to_cad_area_ratio": proposal_area_ratio,
                }
            )
            selected = (model_proposal, selected_audit, support_audit)
            selected_branch = "registered_model_image_shape_proposal"

        model_registration.update(
            {
                "accepted": not rejection_reasons,
                "selection_contract": (
                    "photo_edges_model_shape_candidate_support_and_assembly_"
                    "neighbor_nonregression"
                ),
                "selected_final_branch": selected_branch,
                "selection_rejection_reasons": rejection_reasons,
                "original_cad_candidate_support_floor": seed_candidate_support,
                "proposal_candidate_support": proposal_candidate_support,
                "proposal_to_original_visible_area_ratio": proposal_area_ratio,
                "assembly_context_used": neighbor_context is not None,
                "original_cad_neighbor_overlap_fraction": seed_neighbor_overlap,
                "proposal_neighbor_overlap_fraction": proposal_neighbor_overlap,
                "baseline_final_metrics": dict(baseline_metrics),
                "proposal_final_metrics": dict(proposal_metrics),
            }
        )
        selected[1]["model_domain_photo_registration"] = model_registration
        if selected_branch == "registered_model_image_shape_proposal":
            local_registration = selected[1].get("reference_space_local_registration")
            if isinstance(local_registration, dict):
                local_registration["superseded_by_model_shape_proposal"] = True
        return selected

    if model_visible_shape is not None and _enable_local_registration:
        (
            registered_visible,
            registered_amodal,
            proposal_audit,
        ) = _register_visible_template_to_photo(
            image=image,
            visible_seed=visible,
            amodal_seed=amodal,
            candidate_masks=list(candidate_by_source.values()),
        )
        baseline = _iterative_shape_guided_refinement(
            image=image,
            visible_seed=visible,
            amodal_seed=amodal,
            model_visible_shape=model_visible_shape,
            assembly_neighbor_context=neighbor_context,
            candidate_masks=list(candidate_by_source.items()),
            primary_candidate_source=primary_candidate_source,
            complete_target_shape_variants=complete_target_shape_variants,
            _enable_local_registration=False,
            _enable_model_shape_proposal=False,
        )
        selected = baseline
        selected_branch = "zero_transform_baseline"
        rejection_reasons: list[str] = []
        registered: tuple[
            np.ndarray, dict[str, Any], dict[str, float | int]
        ] | None = None
        if proposal_audit["accepted"] is True:
            registered = _iterative_shape_guided_refinement(
                image=image,
                visible_seed=registered_visible,
                amodal_seed=registered_amodal,
                model_visible_shape=model_visible_shape,
                assembly_neighbor_context=neighbor_context,
                candidate_masks=list(candidate_by_source.items()),
                primary_candidate_source=primary_candidate_source,
                complete_target_shape_variants=complete_target_shape_variants,
                _enable_local_registration=False,
                _enable_model_shape_proposal=False,
            )
            baseline_metrics = baseline[1]["final_metrics"]
            registered_metrics = registered[1]["final_metrics"]
            final_contract_metrics = (
                "image_edge_support",
                "model_domain_shape_score",
                "mean_prior_candidate_iou",
            )
            for metric in final_contract_metrics:
                if (
                    float(registered_metrics[metric])
                    < float(baseline_metrics[metric]) - 1e-12
                ):
                    rejection_reasons.append(f"final_{metric}_regressed")
            strictly_improved = any(
                float(registered_metrics[metric])
                > float(baseline_metrics[metric]) + 1e-12
                for metric in final_contract_metrics
            )
            if not strictly_improved:
                rejection_reasons.append("final_evidence_did_not_strictly_improve")
            if not rejection_reasons:
                selected = registered
                selected_branch = "bounded_2d_local_registration"
        else:
            rejection_reasons.append("local_registration_did_not_improve_both_sources")

        final_registration_audit = dict(proposal_audit)
        final_registration_audit.update(
            {
                "proposal_accepted_by_local_objective": bool(
                    proposal_audit["accepted"]
                ),
                "proposed_translation_xy_pixels": list(
                    proposal_audit["translation_xy_pixels"]
                ),
                "proposed_rotation_degrees": float(proposal_audit["rotation_degrees"]),
                "proposed_metrics": dict(proposal_audit["selected_metrics"]),
                "final_selection_contract": (
                    "pareto_nonregression_of_photo_edges_model_shape_and_"
                    "segmentation_candidate_agreement"
                ),
                "selected_final_branch": selected_branch,
                "final_selection_rejection_reasons": rejection_reasons,
                "zero_transform_final_metrics": dict(baseline[1]["final_metrics"]),
                "registered_final_metrics": (
                    dict(registered[1]["final_metrics"])
                    if registered is not None
                    else None
                ),
            }
        )
        if selected_branch == "zero_transform_baseline":
            final_registration_audit.update(
                {
                    "accepted": False,
                    "translation_xy_pixels": [0, 0],
                    "rotation_degrees": 0.0,
                    "selected_metrics": dict(proposal_audit["initial_metrics"]),
                }
            )
        selected[1]["reference_space_local_registration"] = final_registration_audit
        return selected

    local_registration_audit: dict[str, Any] | None = None

    support_radius, core_radius, occlusion_margin = _automatic_refinement_radii(visible)
    visible_support = _ellipse_morphology(visible, radius=support_radius, dilate=True)
    complete_support = _ellipse_morphology(amodal, radius=support_radius, dilate=True)
    optimization_support = visible_support & complete_support
    visible_with_margin = _ellipse_morphology(
        visible, radius=occlusion_margin, dilate=True
    )
    known_occluded = amodal & ~visible_with_margin
    prior_union = np.logical_or.reduce(list(candidate_by_source.values()))
    primary_unbounded = candidate_by_source[primary_candidate_source]
    initial, _initial_visible_bound = _trim_entity_to_cad_support(
        primary_unbounded,
        visible,
        maximum_final_to_cad_area_ratio=MAXIMUM_FINAL_TO_CAD_AREA_RATIO,
    )
    initial &= optimization_support
    initial &= ~known_occluded
    if not np.any(initial):
        raise EntitySegRegionError("visibility constraints removed the primary mask")

    registered_visible_core = _ellipse_morphology(
        visible, radius=core_radius, dilate=False
    )
    visible_core = registered_visible_core.copy()
    if model_visible_shape is not None:
        # The reference-space CAD projection locates the part but must not
        # paint its interior into the photo.  Hard foreground therefore needs
        # support from an actual photo candidate as well.
        visible_core &= prior_union
    if not np.any(visible_core):
        core_source = initial if model_visible_shape is not None else visible
        distance = cv2.distanceTransform(core_source.astype(np.uint8), cv2.DIST_L2, 3)
        maximum = float(distance.max())
        visible_core = core_source & (distance >= max(0.5, 0.5 * maximum))
    if not np.any(visible_core):
        raise EntitySegRegionError("visible CAD seed has no stable interior core")

    candidate_values = list(candidate_by_source.values())
    initial_metrics = _refinement_metrics(
        image=image,
        mask=initial,
        visible_seed=visible,
        amodal_seed=amodal,
        candidate_masks=candidate_values,
        model_visible_shape=model_visible_shape,
    )
    best_mask = initial
    best_metrics = initial_metrics
    selected_iteration = 0
    selected_lane = "initial_candidate"
    selected_lane_iteration = 0
    iteration_audits: list[dict[str, Any]] = []

    ys, xs = np.where(optimization_support)
    pad = 2
    top = max(0, int(ys.min()) - pad)
    bottom = min(visible.shape[0], int(ys.max()) + pad + 1)
    left = max(0, int(xs.min()) - pad)
    right = min(visible.shape[1], int(xs.max()) + pad + 1)
    crop = np.s_[top:bottom, left:right]
    local_support = optimization_support[crop]
    local_occluded = known_occluded[crop]
    minimum_visible_recall = max(
        0.50, float(initial_metrics["visible_seed_recall"]) - 0.02
    )
    lane_specs = (
        [
            ("photo_candidates_only", prior_union, visible_core),
            ("visibility_completion_proposal", prior_union | visible, visible_core),
            (
                "registered_visibility_grabcut_proposal",
                prior_union | visible,
                registered_visible_core,
            ),
        ]
        if model_visible_shape is not None
        else [("legacy_visible_cad", prior_union | visible, visible_core)]
    )
    global_iteration = 0
    for lane_name, probable_foreground, lane_core in lane_specs:
        labels = np.full(visible[crop].shape, cv2.GC_PR_BGD, dtype=np.uint8)
        labels[~local_support] = cv2.GC_BGD
        labels[
            (probable_foreground & optimization_support & ~known_occluded)[crop]
        ] = cv2.GC_PR_FGD
        labels[lane_core[crop]] = cv2.GC_FGD
        labels[local_occluded] = cv2.GC_BGD
        if not np.any(labels == cv2.GC_FGD) or not np.any(labels == cv2.GC_BGD):
            iteration_audits.append(
                {
                    "iteration": global_iteration,
                    "lane": lane_name,
                    "lane_iteration": 0,
                    "accepted": False,
                    "reason_codes": ["optimization_lane_lacks_hard_seeds"],
                }
            )
            continue
        background_model = np.zeros((1, 65), dtype=np.float64)
        foreground_model = np.zeros((1, 65), dtype=np.float64)
        cv2.setRNGSeed(0)
        previous_iterate = initial
        for lane_iteration in range(1, SHAPE_GUIDED_OPTIMIZATION_ITERATIONS + 1):
            global_iteration += 1
            try:
                cv2.grabCut(
                    image[crop],
                    labels,
                    None,
                    background_model,
                    foreground_model,
                    1,
                    cv2.GC_INIT_WITH_MASK if lane_iteration == 1 else cv2.GC_EVAL,
                )
            except cv2.error as exc:
                iteration_audits.append(
                    {
                        "iteration": global_iteration,
                        "lane": lane_name,
                        "lane_iteration": lane_iteration,
                        "accepted": False,
                        "reason_codes": ["opencv_grabcut_failed"],
                        "error": str(exc),
                    }
                )
                break
            local_mask = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
            refined = np.zeros_like(visible)
            refined[crop] = local_mask
            refined &= optimization_support
            refined &= ~known_occluded
            metrics = _refinement_metrics(
                image=image,
                mask=refined,
                visible_seed=visible,
                amodal_seed=amodal,
                candidate_masks=candidate_values,
                model_visible_shape=model_visible_shape,
            )
            reasons: list[str] = []
            if not np.any(refined):
                reasons.append("refinement_is_empty")
            if not np.all(refined[lane_core]):
                reasons.append("visible_cad_core_was_not_preserved")
            if float(metrics["visible_seed_recall"]) < minimum_visible_recall:
                reasons.append("visible_cad_recall_regressed")
            area_ratio = float(metrics["final_to_visible_area_ratio"])
            if (
                not MINIMUM_REFINED_TO_VISIBLE_AREA_RATIO
                <= area_ratio
                <= MAXIMUM_FINAL_TO_CAD_AREA_RATIO
            ):
                reasons.append("refined_area_outside_visible_cad_bound")
            if model_visible_shape is None and (
                float(metrics["amodal_candidate_precision"])
                < MINIMUM_REFINED_AMODAL_PRECISION
            ):
                reasons.append("refinement_extends_outside_complete_mesh")
            accepted = not reasons
            iteration_audits.append(
                {
                    "iteration": global_iteration,
                    "lane": lane_name,
                    "lane_iteration": lane_iteration,
                    "accepted": accepted,
                    "reason_codes": reasons,
                    "metrics": metrics,
                    "changed_pixels_from_previous_iteration": int(
                        np.count_nonzero(refined ^ previous_iterate)
                    ),
                }
            )
            if accepted and float(metrics["objective_geometric_mean"]) > float(
                best_metrics["objective_geometric_mean"]
            ):
                best_mask = refined
                best_metrics = metrics
                selected_iteration = global_iteration
                selected_lane = lane_name
                selected_lane_iteration = lane_iteration
            previous_iterate = refined

    initial_pixels = int(np.count_nonzero(initial))
    final_pixels = int(np.count_nonzero(best_mask))
    unbounded_pixels = int(np.count_nonzero(primary_unbounded))
    support_audit: dict[str, float | int] = {
        "maximum_support_radius_pixels": support_radius,
        "selected_support_radius_pixels": support_radius,
        "visible_core_radius_pixels": core_radius,
        "occlusion_margin_pixels": occlusion_margin,
        "maximum_final_to_cad_area_ratio": MAXIMUM_FINAL_TO_CAD_AREA_RATIO,
        "untrimmed_entity_pixels": unbounded_pixels,
        "trimmed_entity_pixels": final_pixels,
        "retained_entity_fraction": final_pixels / max(unbounded_pixels, 1),
        "final_to_cad_area_ratio": final_pixels
        / max(int(np.count_nonzero(visible)), 1),
    }
    audit = {
        "method": "iterative_visible_mesh_edge_optimization",
        "candidate_sources": sorted(candidate_by_source),
        "primary_candidate_source": primary_candidate_source,
        "iteration_budget": SHAPE_GUIDED_OPTIMIZATION_ITERATIONS,
        "selected_iteration": selected_iteration,
        "selected_optimization_lane": selected_lane,
        "selected_lane_iteration": selected_lane_iteration,
        "optimization_lanes": [name for name, _foreground, _core in lane_specs],
        "executed_iteration_count": len(iteration_audits),
        "optimization_converged": bool(
            iteration_audits
            and iteration_audits[-1].get("changed_pixels_from_previous_iteration") == 0
        ),
        "automatic_radii": {
            "visible_support_radius_pixels": support_radius,
            "visible_core_radius_pixels": core_radius,
            "occlusion_margin_pixels": occlusion_margin,
        },
        "complete_shape_authority": (
            "cad_model_render_target_part_id_normalized_shape"
            if model_visible_shape is not None
            else "legacy_reference_space_amodal_projection"
        ),
        "model_shape_coordinate_domain": (
            "cad_model_render_image" if model_visible_shape is not None else None
        ),
        "model_shape_photo_warp_applied": False,
        "current_view_visibility_authority": (
            "whole_assembly_part_id_projection_then_bounded_2d_photo_registration"
        ),
        "reference_space_local_registration": local_registration_audit,
        "image_boundary_authority": "current_reference_view_edges",
        "prior_candidate_role": "probable_foreground_initialization_only",
        "known_occluded_pixels": int(np.count_nonzero(known_occluded)),
        "known_occluded_primary_candidate_pixels_removed": int(
            np.count_nonzero(primary_unbounded & known_occluded)
        ),
        "initial_metrics": initial_metrics,
        "final_metrics": best_metrics,
        "final_changed_pixels_from_initial": int(np.count_nonzero(best_mask ^ initial)),
        "iterations": iteration_audits,
    }
    return best_mask, audit, support_audit


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
    if "entity_amodal_candidate_precision" in metrics and (
        float(metrics["entity_amodal_candidate_precision"])
        < MINIMUM_AMODAL_CANDIDATE_PRECISION
    ):
        reasons.append("entity_extends_outside_complete_mesh_shape")
    if "entity_amodal_shape_iou" in metrics and (
        float(metrics["entity_amodal_shape_iou"]) < MINIMUM_AMODAL_COMPLETION_SHAPE_IOU
    ):
        reasons.append("entity_occlusion_aware_amodal_shape_mismatch")
    if (
        float(metrics["entity_cad_centroid_distance"])
        > MAXIMUM_ENTITY_CAD_CENTROID_DISTANCE
    ):
        reasons.append("entity_centroid_too_far_from_cad_part")
    if float(metrics["entity_edge_support"]) < MINIMUM_ENTITY_EDGE_SUPPORT:
        reasons.append("entity_boundary_has_insufficient_image_edge_support")
    if sam_accepted:
        if float(metrics["entity_edge_improvement"]) < MINIMUM_ENTITY_EDGE_IMPROVEMENT:
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
        raise EntitySegRegionError(
            "accepted EntitySeg record has no selected candidate"
        )
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
    if "cad_amodal_candidate_precision" in selected:
        output.update(
            {
                "entity_amodal_candidate_precision": float(
                    selected["cad_amodal_candidate_precision"]
                ),
                "entity_amodal_completion_iou": float(
                    selected["cad_amodal_completion_iou"]
                ),
                "entity_amodal_shape_iou": float(selected["cad_amodal_shape_iou"]),
                "entity_to_amodal_area_ratio": float(
                    selected["candidate_to_cad_amodal_area_ratio"]
                ),
            }
        )
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
    amodal_manifest_path: Path | None = None,
    prior_hybrid_manifest_path: Path | None = None,
) -> dict[str, Any]:
    sam_manifest_path = sam_manifest_path.expanduser().resolve(strict=True)
    entity_manifest_path = entity_manifest_path.expanduser().resolve(strict=True)
    sam_document = _read_manifest(sam_manifest_path, "SAM3 manifest")
    entity_document = _read_manifest(entity_manifest_path, "EntitySeg manifest")
    if sam_document.get("request") != entity_document.get("request"):
        raise EntitySegRegionError(
            "SAM3 and EntitySeg manifests bind different requests"
        )
    sam_records = _records(sam_document, "SAM3")
    entity_records = _records(entity_document, "EntitySeg")
    if set(sam_records) != set(entity_records):
        raise EntitySegRegionError("SAM3 and EntitySeg region sets differ")
    request_path, request_document = _bound_request(sam_document, label="SAM3 manifest")
    entity_request_path, entity_request_document = _bound_request(
        entity_document, label="EntitySeg manifest"
    )
    if (
        request_path != entity_request_path
        or request_document != entity_request_document
    ):
        raise EntitySegRegionError("SAM3 and EntitySeg request documents differ")
    model_shape_references: dict[tuple[str, str], dict[str, Any]] = {}
    model_template_document: dict[str, Any] | None = None
    resolved_amodal_manifest: Path | None = None
    if amodal_manifest_path is not None:
        resolved_amodal_manifest = amodal_manifest_path.expanduser().resolve(
            strict=True
        )
        (
            model_shape_references,
            model_template_document,
        ) = _model_domain_shape_references(
            manifest_path=resolved_amodal_manifest,
            expected_keys=set(sam_records),
        )
    relation_audits, relation_regions = _relation_request_records(
        request=request_document,
        expected_keys=set(sam_records),
        amodal_manifest_path=resolved_amodal_manifest,
    )
    relation_guided = bool(relation_audits)
    if relation_guided:
        for reference in model_shape_references.values():
            reference["visible_shape"] = [reference["relation_visible_shape"]]
            reference["display_visible_shape"] = reference["relation_visible_shape"]
            reference["audit"]["model_shape_variant_count"] = len(
                reference["visible_shape"]
            )
            reference["audit"][
                "model_component_association_role"
            ] = "all_target_part_id_components_from_model_image_without_photo_mapping"
    prior_records: dict[tuple[str, str], Mapping[str, Any]] = {}
    prior_document: dict[str, Any] | None = None
    resolved_prior_manifest: Path | None = None
    if prior_hybrid_manifest_path is not None:
        resolved_prior_manifest = prior_hybrid_manifest_path.expanduser().resolve(
            strict=True
        )
        prior_document = _read_manifest(
            resolved_prior_manifest, "prior hybrid manifest"
        )
        if prior_document.get("schema_version") not in {
            MODEL_SCHEMA_VERSION,
            SCHEMA_VERSION,
        }:
            raise EntitySegRegionError("prior hybrid manifest schema is unsupported")
        prior_records = _records(prior_document, "prior hybrid")
        if set(prior_records) != set(sam_records):
            raise EntitySegRegionError(
                "prior hybrid regions do not exactly cover relation refinement"
            )
    sam_root = sam_manifest_path.parent
    entity_root = entity_manifest_path.parent
    prior_root = (
        resolved_prior_manifest.parent if resolved_prior_manifest is not None else None
    )
    output_dir = output_dir.expanduser().resolve()
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    if model_shape_references:
        model_shape_dir = output_dir / "model_shape_masks"
        model_shape_dir.mkdir(parents=True, exist_ok=True)
        for key, reference in model_shape_references.items():
            model_shape_path = model_shape_dir / f"{key[0]}__{key[1]}.png"
            display_shape = np.asarray(
                reference["display_visible_shape"], dtype=np.uint8
            )
            if not cv2.imwrite(str(model_shape_path), display_shape * 255):
                raise EntitySegRegionError(
                    f"unable to write model-domain local shape: {key}"
                )
            reference["audit"]["model_local_shape_mask"] = {
                "path": str(model_shape_path),
                "sha256": _sha256_file(model_shape_path),
                "mask_pixels": int(np.count_nonzero(display_shape)),
            }
            variant_documents: list[dict[str, Any]] = []
            for variant_index, variant in enumerate(reference["visible_shape"]):
                variant_path = (
                    model_shape_dir / f"{key[0]}__{key[1]}__variant_{variant_index}.png"
                )
                variant_mask = np.asarray(variant, dtype=np.uint8)
                if not cv2.imwrite(str(variant_path), variant_mask * 255):
                    raise EntitySegRegionError(
                        f"unable to write model-domain shape variant: {key}"
                    )
                variant_documents.append(
                    {
                        "variant_index": variant_index,
                        "path": str(variant_path),
                        "sha256": _sha256_file(variant_path),
                        "mask_pixels": int(np.count_nonzero(variant_mask)),
                    }
                )
            reference["audit"]["model_shape_variant_masks"] = variant_documents

    records: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for key in sorted(sam_records):
        sam_row = sam_records[key]
        entity_row = entity_records[key]
        model_reference = model_shape_references.get(key)
        relation_audit = relation_audits.get(key)
        relation_region = relation_regions.get(key)
        prior_row = prior_records.get(key)
        sam_shared = sam_row.get("view_shared_alignment")
        entity_shared = entity_row.get("view_shared_alignment")
        if (
            not isinstance(sam_shared, Mapping)
            or not isinstance(entity_shared, Mapping)
            or dict(sam_shared) != dict(entity_shared)
            or sam_shared.get("part_specific_translation_allowed") is not False
        ):
            raise EntitySegRegionError(
                f"SAM3 and EntitySeg do not share one whole-workpiece alignment: {key}"
            )
        authority = entity_row if entity_row.get("source_image") else sam_row
        source_path = (
            Path(str(authority["source_image"])).expanduser().resolve(strict=True)
        )
        source_sha256 = _sha256_file(source_path)
        if (
            sam_row.get("source_image_sha256") != source_sha256
            or entity_row.get("source_image_sha256") != source_sha256
        ):
            raise EntitySegRegionError(f"source image hash mismatch: {key}")
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise EntitySegRegionError(f"unable to decode source image: {source_path}")
        seed_doc = entity_row.get("cad_projection_seed") or sam_row.get(
            "cad_projection_seed"
        )
        if not isinstance(seed_doc, Mapping):
            raise EntitySegRegionError(f"missing CAD seed: {key}")
        seed_path = Path(str(seed_doc["path"])).expanduser().resolve(strict=True)
        if seed_doc.get("sha256") != _sha256_file(seed_path):
            raise EntitySegRegionError(f"CAD seed hash mismatch: {key}")
        seed = _load_mask(seed_path, image.shape[:2])
        sam_aligned_seed, sam_alignment_audit = _sam_aligned_cad_seed(seed, sam_row)
        entity_aligned_seed, entity_alignment_audit = _entity_aligned_cad_seed(
            seed, entity_row
        )
        expected_translation = [
            float(value) for value in sam_shared["translation_xy_pixels"]
        ]
        if (
            sam_alignment_audit.get("translation_xy_pixels") != expected_translation
            or entity_alignment_audit.get("translation_xy_pixels")
            != expected_translation
        ):
            raise EntitySegRegionError(
                f"candidate alignment differs from the view-shared alignment: {key}"
            )
        sam_amodal_doc = sam_row.get("cad_amodal_template")
        entity_amodal_doc = entity_row.get("cad_amodal_template")
        amodal_doc = (
            sam_amodal_doc if isinstance(sam_amodal_doc, Mapping) else entity_amodal_doc
        )
        amodal: np.ndarray | None = None
        if isinstance(amodal_doc, Mapping):
            amodal_path = (
                Path(str(amodal_doc["path"])).expanduser().resolve(strict=True)
            )
            expected_amodal_hash = amodal_doc.get("sha256")
            if (
                not isinstance(expected_amodal_hash, str)
                or _sha256_file(amodal_path) != expected_amodal_hash
            ):
                raise EntitySegRegionError(f"CAD amodal hash mismatch: {key}")
            if isinstance(entity_amodal_doc, Mapping):
                entity_path = (
                    Path(str(entity_amodal_doc["path"]))
                    .expanduser()
                    .resolve(strict=True)
                )
                if (
                    entity_path != amodal_path
                    or entity_amodal_doc.get("sha256") != expected_amodal_hash
                ):
                    raise EntitySegRegionError(
                        f"SAM3 and EntitySeg bind different CAD amodal templates: {key}"
                    )
            contract = amodal_doc.get("projection_contract")
            if (
                not isinstance(contract, Mapping)
                or contract.get("whole_asset_camera_unchanged") is not True
                or contract.get("whole_asset_transform_unchanged") is not True
                or contract.get("per_mesh_pose_change_allowed") is not False
            ):
                raise EntitySegRegionError(f"CAD amodal contract mismatch: {key}")
            amodal = _load_mask(amodal_path, image.shape[:2])
            if model_reference is not None:
                if relation_guided:
                    relation_amodal = (
                        relation_region.get("cad_amodal_template")
                        if isinstance(relation_region, Mapping)
                        else None
                    )
                    if (
                        not isinstance(relation_amodal, Mapping)
                        or relation_amodal.get("sha256") != expected_amodal_hash
                    ):
                        raise EntitySegRegionError(
                            f"relation and segmentation amodal templates differ: {key}"
                        )
                elif (
                    model_reference.get("aligned_amodal_sha256") != expected_amodal_hash
                ):
                    raise EntitySegRegionError(
                        f"model-domain and reference-space templates differ: {key}"
                    )
        sam_aligned_amodal = (
            _align_with_audit(amodal, sam_alignment_audit)
            if amodal is not None
            else None
        )
        entity_aligned_amodal = (
            _align_with_audit(amodal, entity_alignment_audit)
            if amodal is not None
            else None
        )
        sam_accepted = sam_row.get("accepted") is True
        entity_accepted = entity_row.get("accepted") is True
        sam_mask = (
            _load_mask(_resolved_mask_path(sam_root, sam_row), image.shape[:2])
            if sam_accepted
            else None
        )
        entity_mask = (
            _load_mask(_resolved_mask_path(entity_root, entity_row), image.shape[:2])
            if entity_accepted
            else None
        )
        metrics: dict[str, float | int] | None = None
        entity_reasons: list[str] = []
        if entity_mask is not None:
            metrics = _entity_metrics(
                image=image,
                seed=entity_aligned_seed,
                entity_mask=entity_mask,
                entity_row=entity_row,
                sam_mask=sam_mask,
            )
            entity_reasons = _entity_rejection_reasons(
                metrics, sam_accepted=sam_accepted
            )

        final_mask: np.ndarray | None = None
        cad_support_trim: dict[str, float | int] | None = None
        iterative_refinement: dict[str, Any] | None = None
        initializer_ranking: list[dict[str, float | str]] | None = None
        candidate_masks: list[tuple[str, np.ndarray]] = []
        primary_candidate_source: str | None = None
        entity_candidate_source = (
            "relation_entityseg" if relation_guided else "entityseg"
        )
        sam_candidate_source = "relation_sam3" if relation_guided else "sam3"
        if isinstance(prior_row, Mapping) and prior_row.get("accepted") is True:
            if prior_root is None:
                raise EntitySegRegionError("prior hybrid root is unavailable")
            if (
                prior_row.get("source_image") != str(source_path)
                or prior_row.get("source_image_sha256") != source_sha256
            ):
                raise EntitySegRegionError(f"prior hybrid source image differs: {key}")
            candidate_masks.append(
                (
                    "prior_iterative_hybrid",
                    _load_mask(
                        _resolved_mask_path(prior_root, prior_row),
                        image.shape[:2],
                    ),
                )
            )
        if entity_mask is not None and not entity_reasons:
            candidate_masks.append((entity_candidate_source, entity_mask))
            primary_candidate_source = entity_candidate_source
        if sam_mask is not None:
            candidate_masks.append((sam_candidate_source, sam_mask))
            if primary_candidate_source is None:
                primary_candidate_source = sam_candidate_source
        if not candidate_masks and relation_guided:
            # A rejected neural proposal must not abort the general pipeline.
            # The neighbour-located CAD silhouette remains a deterministic
            # initializer and is still refined against photo edges below.
            candidate_masks.append(("relation_cad_location_fallback", sam_aligned_seed))
            primary_candidate_source = "relation_cad_location_fallback"
        if candidate_masks and model_reference is not None:
            (
                primary_candidate_source,
                initializer_ranking,
            ) = _rank_model_guided_initializers(
                image=image,
                candidates=candidate_masks,
                model_visible_shape=model_reference["visible_shape"],
            )
        aligned_seed_audit = (
            entity_alignment_audit
            if primary_candidate_source == entity_candidate_source
            else sam_alignment_audit
        )
        if primary_candidate_source is not None:
            aligned_visible = (
                entity_aligned_seed
                if primary_candidate_source == entity_candidate_source
                else sam_aligned_seed
            )
            aligned_amodal = (
                entity_aligned_amodal
                if primary_candidate_source == entity_candidate_source
                else sam_aligned_amodal
            )
            # In the v3 production lane the reference-space amodal projection
            # is validated for lineage only.  It is not pasted into the photo
            # optimizer.  The visible projection remains a bounded ROI and
            # visibility constraint; shape comes from the CAD model image.
            refinement_amodal = None if model_reference is not None else aligned_amodal
            if (
                relation_guided
                and isinstance(relation_region, Mapping)
                and isinstance(relation_audit, Mapping)
                and relation_audit.get("accepted") is True
            ):
                neighbor_document = relation_region.get("cad_assembly_neighbor_context")
                if not isinstance(neighbor_document, Mapping):
                    raise EntitySegRegionError(
                        f"relation request has no neighbor context: {key}"
                    )
                _neighbor_path, neighbor_context = _load_document_mask(
                    neighbor_document,
                    label=f"relation neighbor context {key}",
                )
                assembly_neighbor_context = _align_with_audit(
                    neighbor_context,
                    aligned_seed_audit,
                )
            else:
                assembly_neighbor_context = (
                    _align_with_audit(
                        model_reference["assembly_neighbor_context"],
                        aligned_seed_audit,
                    )
                    if model_reference is not None
                    else None
                )
            try:
                (
                    final_mask,
                    iterative_refinement,
                    cad_support_trim,
                ) = _iterative_shape_guided_refinement(
                    image=image,
                    visible_seed=aligned_visible,
                    amodal_seed=refinement_amodal,
                    model_visible_shape=(
                        model_reference["visible_shape"]
                        if model_reference is not None
                        else None
                    ),
                    assembly_neighbor_context=assembly_neighbor_context,
                    candidate_masks=candidate_masks,
                    primary_candidate_source=primary_candidate_source,
                    complete_target_shape_variants=relation_guided,
                )
            except EntitySegRegionError as exc:
                final_mask = None
                selected_source = "none"
                decision = "iterative_refinement_rejected"
                iterative_refinement = {
                    "method": "iterative_visible_mesh_edge_optimization",
                    "accepted": False,
                    "reason_codes": ["shape_guided_refinement_failed"],
                    "error": str(exc),
                }
            else:
                selected_source = "shape_guided_iterative"
                candidate_sources = {source for source, _mask in candidate_masks}
                if not relation_guided and candidate_sources == {"sam3", "entityseg"}:
                    decision = "iterative_refinement_from_sam3_entityseg"
                elif not relation_guided and candidate_sources == {"entityseg"}:
                    decision = "iterative_refinement_from_entityseg"
                elif not relation_guided:
                    decision = "iterative_refinement_from_sam3"
                elif "relation_cad_location_fallback" in candidate_sources:
                    decision = "iterative_refinement_from_relation_cad_fallback"
                elif "prior_iterative_hybrid" in candidate_sources:
                    decision = "iterative_refinement_from_prior_and_relation_candidates"
                elif candidate_sources == {"relation_entityseg"}:
                    decision = "iterative_refinement_from_relation_entityseg"
                elif candidate_sources == {"relation_sam3"}:
                    decision = "iterative_refinement_from_relation_sam3"
                else:
                    decision = "iterative_refinement_from_relation_sam3_entityseg"
        else:
            selected_source = "none"
            decision = "no_safe_candidate"

        mask_document: dict[str, Any] | None = None
        shape_candidate: dict[str, Any] | None = None
        if primary_candidate_source == entity_candidate_source:
            selected_entity = entity_row.get("selected_candidate")
            if not isinstance(selected_entity, Mapping):
                raise EntitySegRegionError(
                    f"accepted EntitySeg region has no shape candidate: {key}"
                )
            shape_candidate = dict(selected_entity)
        elif primary_candidate_source == sam_candidate_source:
            shape_candidate = _sam_selected_shape_candidate(sam_row)
        elif final_mask is not None:
            shape_candidate = {
                "source": primary_candidate_source,
                "selection_role": "iterative_initializer_only",
                "final_boundary_selected_by": (
                    "photo_edges_plus_complete_cad_model_shape_plus_candidate_agreement"
                ),
            }
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
                "source_image_sha256": source_sha256,
                "view_shared_alignment": dict(sam_shared),
                "cad_projection_seed": dict(seed_doc),
                "accepted": final_mask is not None,
                "selected_source": selected_source,
                "primary_candidate_source": primary_candidate_source,
                "candidate_sources": [source for source, _mask in candidate_masks],
                "relation_guidance": (
                    dict(relation_audit)
                    if isinstance(relation_audit, Mapping)
                    else None
                ),
                "decision": decision,
                "entityseg_candidate_accepted": entity_accepted,
                "entityseg_fusion_rejection_reasons": entity_reasons,
                "fusion_metrics": metrics,
                "cad_support_trim": cad_support_trim,
                "iterative_refinement": iterative_refinement,
                "aligned_cad_template": aligned_seed_audit,
                "model_domain_shape_reference": (
                    dict(model_reference["audit"])
                    if model_reference is not None
                    else None
                ),
                "model_guided_initializer_ranking": initializer_ranking,
                "reference_space_amodal_used_for_final_boundary": (
                    model_reference is None and aligned_amodal is not None
                    if primary_candidate_source is not None
                    else False
                ),
                "shape_candidate": shape_candidate,
                "fusion_audit": {
                    "selected_source": selected_source,
                    "decision": decision,
                    "fusion_metrics": metrics,
                    "cad_support_trim": cad_support_trim,
                    "iterative_refinement": iterative_refinement,
                },
                "cad_amodal_template": (
                    {
                        **dict(amodal_doc),
                        "aligned_with_visible_template_translation": True,
                    }
                    if isinstance(amodal_doc, Mapping)
                    else None
                ),
                "mask": mask_document,
            }
        )

    result: dict[str, Any] = {
        "schema_version": (
            SCHEMA_VERSION
            if relation_guided
            else MODEL_SCHEMA_VERSION
            if model_shape_references
            else LEGACY_SCHEMA_VERSION
        ),
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
            **(
                {
                    "cad_model_templates": {
                        "path": str(resolved_amodal_manifest),
                        "sha256": _sha256_file(resolved_amodal_manifest),
                        "document_sha256": _canonical_sha256(model_template_document),
                    }
                }
                if resolved_amodal_manifest is not None
                and model_template_document is not None
                else {}
            ),
            **(
                {
                    "prior_hybrid_manifest": {
                        "path": str(resolved_prior_manifest),
                        "sha256": _sha256_file(resolved_prior_manifest),
                        "document_sha256": _canonical_sha256(prior_document),
                    }
                }
                if resolved_prior_manifest is not None and prior_document is not None
                else {}
            ),
        },
        "request": dict(sam_document.get("request", {})),
        "policy": {
            "identity_authority": "registered_cad_part_id_plus_sam3_instance",
            "sam3_role": "probable_foreground_initialization_only",
            "entityseg_role": "probable_foreground_initialization_only",
            "maximum_entity_to_cad_area_ratio": MAXIMUM_ENTITY_TO_CAD_AREA_RATIO,
            "minimum_entity_cad_direct_iou": MINIMUM_ENTITY_CAD_DIRECT_IOU,
            "minimum_entity_cad_shape_iou": MINIMUM_ENTITY_CAD_SHAPE_IOU,
            "maximum_entity_cad_centroid_distance": MAXIMUM_ENTITY_CAD_CENTROID_DISTANCE,
            "minimum_entity_edge_support": MINIMUM_ENTITY_EDGE_SUPPORT,
            "minimum_entity_edge_improvement": MINIMUM_ENTITY_EDGE_IMPROVEMENT,
            "minimum_sam_entity_overlap_smaller": MINIMUM_SAM_ENTITY_OVERLAP_SMALLER,
            "minimum_direct_iou_when_sam_disagrees": MINIMUM_DIRECT_IOU_WHEN_SAM_DISAGREES,
            "maximum_final_to_cad_area_ratio": MAXIMUM_FINAL_TO_CAD_AREA_RATIO,
            "maximum_cad_support_radius_fraction": MAXIMUM_CAD_SUPPORT_RADIUS_FRACTION,
            "minimum_cad_support_radius_pixels": MINIMUM_CAD_SUPPORT_RADIUS_PIXELS,
            "minimum_amodal_candidate_precision": MINIMUM_AMODAL_CANDIDATE_PRECISION,
            "minimum_amodal_completion_shape_iou": (
                MINIMUM_AMODAL_COMPLETION_SHAPE_IOU
            ),
            "shape_authority": (
                "cad_model_render_target_part_id_normalized_shape"
                if model_shape_references
                else "legacy_isolated_mesh_reference_space_projection"
            ),
            "shape_coordinate_domain": (
                "cad_model_render_image" if model_shape_references else None
            ),
            "model_shape_photo_warp_applied": False,
            "model_shape_photo_proposal_warp_applied": bool(model_shape_references),
            "model_image_shape_photo_proposal": (
                "bounded_similarity_registration_from_model_image_via_sealed_cad_"
                "assembly_context_then_photo_edges"
                if model_shape_references
                else None
            ),
            "model_shape_proposal_selection_contract": (
                "photo_edges_model_shape_candidate_support_and_assembly_neighbor_"
                "nonregression"
                if model_shape_references
                else None
            ),
            "assembly_context_position_authority": (
                "leave_one_target_out_multi_anchor_cad_part_relations"
                if relation_guided
                else "whole_assembly_model_part_id_neighbor_geometry"
                if model_shape_references
                else None
            ),
            "target_location_method": (
                "robust_similarity_plus_nearby_anchor_residual_voting"
                if relation_guided
                else None
            ),
            "target_first_pass_mask_used_for_own_location": (
                False if relation_guided else None
            ),
            "target_direct_cad_projection_used_when_relation_accepted": (
                False if relation_guided else None
            ),
            "relation_failure_policy": (
                "preserve_prior_hybrid_then_original_cad_edge_fallback"
                if relation_guided
                else None
            ),
            "relation_second_pass_segmentation": relation_guided,
            "prior_hybrid_candidate_preserved": bool(prior_records),
            "neural_rejection_fallback": (
                "relation_located_cad_shape_photo_edge_refinement"
                if relation_guided
                else None
            ),
            "relation_neighbor_exclusion_authority": (
                "target_specific_cad_assembly_warp_from_non_target_anchor_votes"
                if relation_guided
                else None
            ),
            "model_shape_proposal_component_policy": (
                "complete_target_part_id_union_plus_all_model_components_without_"
                "photo_mapping"
                if relation_guided
                else "union_of_all_model_components_associated_with_local_observation"
                if model_shape_references
                else None
            ),
            "visibility_authority": "whole_assembly_part_id_projection",
            "final_boundary_method": "iterative_visible_mesh_edge_optimization",
            "shape_guided_optimization_iterations": SHAPE_GUIDED_OPTIMIZATION_ITERATIONS,
            "candidate_selection_policy": (
                "joint_iterative_optimization_not_single_model_arbitration"
            ),
            "optimization_objective": (
                "unweighted_geometric_mean_of_photo_edges_model_domain_"
                "normalized_shape_and_prior_candidate_agreement"
                if model_shape_references
                else "legacy_unweighted_geometric_mean_of_photo_edges_visible_"
                "cad_and_prior_candidate_agreement"
            ),
            "reference_space_cad_role": (
                "initial_roi_and_visibility_for_bounded_2d_mask_registration"
                if model_shape_references
                else "legacy_shape_and_visibility"
            ),
            "reference_space_local_registration": (
                "scale_derived_2d_mask_rigid_search_using_photo_edges_and_"
                "segmentation_candidate_support"
                if model_shape_references
                else None
            ),
            "local_registration_transformed_object": (
                "reference_view_segmentation_template_only"
                if model_shape_references
                else None
            ),
            "local_registration_selection_contract": (
                "pareto_nonregression_of_photo_edges_and_candidate_support"
                if model_shape_references
                else None
            ),
            "local_registration_final_selection_contract": (
                "pareto_nonregression_of_photo_edges_model_shape_and_"
                "segmentation_candidate_agreement"
                if model_shape_references
                else None
            ),
            "cad_mesh_transform_changed": False,
            "assembly_camera_changed": False,
            "known_occlusion_policy": (
                "model_image_visible_components_define_normalized_shape_while_"
                "reference_projection_only_bounds_roi"
                if model_shape_references
                else "amodal_minus_current_view_visible_projection_is_background"
            ),
            "alignment_model": "one_whole_workpiece_translation_per_view",
            "per_mesh_pose_change_allowed": False,
            "part_specific_translation_allowed": False,
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
    parser.add_argument(
        "--amodal-manifest",
        type=Path,
        help=(
            "Sealed CAD template manifest providing the target Part-ID shape "
            "in the CAD model render image"
        ),
    )
    parser.add_argument(
        "--prior-hybrid-manifest",
        type=Path,
        help=(
            "First-pass iterative hybrid masks retained as candidates while "
            "the relation-located second pass is optimized"
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build_hybrid_masks(
        sam_manifest_path=args.sam_manifest,
        entity_manifest_path=args.entity_manifest,
        output_dir=args.output_dir,
        amodal_manifest_path=args.amodal_manifest,
        prior_hybrid_manifest_path=args.prior_hybrid_manifest,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
