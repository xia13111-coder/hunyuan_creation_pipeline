from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw

from qwen_material_pipeline.evidence import spatial as spatial_gate
from qwen_material_pipeline.evidence.spatial import (
    SpatialMappingError,
    apply_spatial_gate_to_batches,
    build_spatial_mapping_report,
)


IMAGE_SIZE = (256, 256)
BACKGROUND = (5, 5, 5)
CAD_FOREGROUND = (170, 170, 170)
GROUP_COLORS = {
    "G01": (45, 145, 62),
    "G02": (185, 48, 42),
}


@dataclass(frozen=True)
class SpatialCase:
    manifest: Path
    registry: Path
    view_group_id_maps: dict[str, dict[str, str]]
    palettes: dict[str, dict[str, Any]]
    audits: dict[str, dict[str, Any]]
    reference_images: tuple[Path, ...]


def _part_color(part_id: str) -> tuple[int, int, int]:
    """Mirror the stable encoding used by render_part_views."""

    import colorsys

    number = int(part_id[1:]) if part_id[1:].isdigit() else sum(map(ord, part_id))
    hue = (number * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return int(red * 255), int(green * 255), int(blue * 255)


def test_reference_foreground_removes_connected_viewer_axes() -> None:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[96:176, 52:216] = (45, 145, 62)
    # Three-pixel CAD-viewer axes intersect the solid object.  Component
    # filtering cannot remove them unless morphology severs them first.
    image[:, 126:129] = (20, 20, 180)
    image[134:137, :] = (40, 120, 40)

    foreground = spatial_gate._reference_foreground(image)
    x, y, width, height = spatial_gate._bbox(foreground, "synthetic reference")

    assert x > 0
    assert y > 0
    assert x + width < image.shape[1]
    assert y + height < image.shape[0]
    assert foreground[110, 80] == 255


def test_manifest_foreground_is_shared_geometry_authority(tmp_path: Path) -> None:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    # The RGB heuristic would select this larger rectangle.
    image[4:44, 5:59] = (45, 145, 62)
    image_path = tmp_path / "reference.png"
    Image.fromarray(image[:, :, ::-1]).save(image_path)
    sealed = np.zeros((48, 64), dtype=np.uint8)
    sealed[12:40, 18:52] = 255
    mask_path = tmp_path / "foreground.png"
    Image.fromarray(sealed).save(mask_path)

    foreground, resolved, authority = (
        spatial_gate._reference_foreground_from_manifest(
            source={"image": str(image_path), "palette_mask": str(mask_path)},
            manifest_path=None,
            image=image,
            view_id="front",
        )
    )

    assert authority == "manifest_palette_mask"
    assert resolved == mask_path.resolve()
    assert np.array_equal(foreground, sealed)


def test_declared_manifest_foreground_shape_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    mask_path = tmp_path / "wrong_shape.png"
    Image.fromarray(np.full((24, 32), 255, dtype=np.uint8)).save(mask_path)

    with pytest.raises(
        SpatialMappingError,
        match="does not match image shape",
    ):
        spatial_gate._reference_foreground_from_manifest(
            source={"palette_mask": str(mask_path)},
            manifest_path=None,
            image=image,
            view_id="front",
        )


def test_similarity_scale_contract_is_independent_of_raster_resolution() -> None:
    reference = np.zeros((128, 128), dtype=np.uint8)
    reference[32:96, 32:96] = 255
    render_full = reference.copy()
    render_half = np.zeros((64, 64), dtype=np.uint8)
    render_half[16:48, 16:48] = 255

    full = spatial_gate._refine_projection(
        reference,
        render_full,
        spatial_gate.DEFAULT_POLICY,
    )
    half = spatial_gate._refine_projection(
        reference,
        render_half,
        spatial_gate.DEFAULT_POLICY,
    )

    assert full["ecc_status"] == "success"
    assert half["ecc_status"] == "success"
    assert full["ecc_transform_audit"]["minimum_scale"] == pytest.approx(1.0)
    assert half["ecc_transform_audit"]["minimum_scale"] == pytest.approx(1.0)
    assert half["ecc_transform_audit"]["raw_uniform_scale"] == pytest.approx(2.0)
    assert half["ecc_transform_audit"][
        "resolution_scale_normalization"
    ] == pytest.approx(0.5)


def test_unique_multiview_canonical_color_supplements_omitted_local_palette() -> None:
    references = [
        {
            "view_id": "front",
            "palette_groups": [{"group_id": "F_BLACK", "base_color": "black"}],
            "group_id_map": {"F_BLACK": "G_BLACK"},
        },
        {
            "view_id": "side",
            "palette_groups": [{"group_id": "S_BLACK", "base_color": "black"}],
            "group_id_map": {"S_BLACK": "G_BLACK"},
        },
        {
            "view_id": "top",
            "palette_groups": [{"group_id": "T_GREEN", "base_color": "green"}],
            "group_id_map": {"T_GREEN": "G_GREEN"},
        },
    ]

    supplements = spatial_gate._canonical_palette_supplements(references)

    assert supplements["front"] == []
    assert supplements["side"] == []
    assert supplements["top"] == [
        {
            "canonical_group_id": "G_BLACK",
            "base_color": "black",
            "accepted_labels": ["black", "darkgray"],
            "source_view_ids": ["front", "side"],
        }
    ]


def test_ambiguous_canonical_color_family_is_not_propagated() -> None:
    references = [
        {
            "view_id": "a1",
            "palette_groups": [{"group_id": "A1", "base_color": "black"}],
            "group_id_map": {"A1": "G_BLACK_A"},
        },
        {
            "view_id": "a2",
            "palette_groups": [{"group_id": "A2", "base_color": "black"}],
            "group_id_map": {"A2": "G_BLACK_A"},
        },
        {
            "view_id": "b1",
            "palette_groups": [{"group_id": "B1", "base_color": "black"}],
            "group_id_map": {"B1": "G_BLACK_B"},
        },
        {
            "view_id": "b2",
            "palette_groups": [{"group_id": "B2", "base_color": "black"}],
            "group_id_map": {"B2": "G_BLACK_B"},
        },
        {
            "view_id": "target",
            "palette_groups": [{"group_id": "T1", "base_color": "green"}],
            "group_id_map": {"T1": "G_GREEN"},
        },
    ]

    supplements = spatial_gate._canonical_palette_supplements(references)

    assert supplements["target"] == []


def test_singleton_color_claim_does_not_veto_multiview_supplement() -> None:
    references = [
        {
            "view_id": "front",
            "palette_groups": [{"group_id": "F_BLACK", "base_color": "black"}],
            "group_id_map": {"F_BLACK": "G_DARK_MULTI"},
        },
        {
            "view_id": "side",
            "palette_groups": [{"group_id": "S_BLACK", "base_color": "black"}],
            "group_id_map": {"S_BLACK": "G_DARK_MULTI"},
        },
        {
            "view_id": "top",
            "palette_groups": [{"group_id": "T_BLACK", "base_color": "black"}],
            "group_id_map": {"T_BLACK": "G_DARK_MULTI"},
        },
        {
            "view_id": "iso",
            "palette_groups": [{"group_id": "I_BLACK", "base_color": "black"}],
            "group_id_map": {"I_BLACK": "G_BLACK_SINGLE"},
        },
    ]

    groups = spatial_gate._unique_multiview_canonical_palette_groups(references)

    assert groups == {
        "G_DARK_MULTI": {
            "canonical_group_id": "G_DARK_MULTI",
            "base_color": "black",
            "accepted_labels": ["black", "darkgray"],
            "source_view_ids": ["front", "side", "top"],
        }
    }


def test_accepted_palette_box_overlap_is_pixel_reproducible_and_tamper_bound(
    tmp_path: Path,
) -> None:
    image_rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    image_rgb[8:24, 8:24] = (40, 110, 225)
    image_path = tmp_path / "reference.png"
    Image.fromarray(image_rgb).save(image_path)
    palette_groups = [
        {
            "group_id": "L_BLUE",
            "base_color": "blue",
            "boxes": [[250, 250, 750, 750]],
        }
    ]
    audit = {
        "mask": None,
        "estimated_background_rgb": [0, 0, 0],
        "background_distance": 28.0,
        "groups": [
            {
                "group_id": "L_BLUE",
                "base_color": "blue",
                "accepted": True,
                "boxes": [
                    {
                        "box_index": 0,
                        "box": [250, 250, 750, 750],
                        "accepted": True,
                        "matching_pixel_count": 256,
                        "foreground_method": "color_distance",
                        "effective_background_distance": 28.0,
                        "accepted_color_labels": ["blue", "cyan"],
                    }
                ],
            }
        ],
    }

    regions = spatial_gate._accepted_palette_evidence_regions(
        image_path=image_path,
        image_shape=(32, 32, 3),
        palette_groups=palette_groups,
        group_id_map={"L_BLUE": "G_BLUE"},
        palette_audit=audit,
        view_id="top",
    )

    assert len(regions) == 1
    region = regions[0]
    assert region["evidence_pixel_count"] == 256
    assert np.count_nonzero(region["_mask"]) == 256

    part_ids = np.zeros((32, 32, 3), dtype=np.uint8)
    red, green, blue = _part_color("P0001")
    part_ids[8:24, 8:24] = (blue, green, red)
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    observation = spatial_gate._project_part_observation(
        reference={
            "view_id": "top",
            "image": image_rgb[:, :, ::-1].copy(),
            "foreground": np.full((32, 32), 255, dtype=np.uint8),
        },
        render={
            "view_id": "top",
            "visible_pixels": {"P0001": 256},
            "part_ids_image": part_ids,
        },
        alignment={
            "quarter_turns_ccw": 0,
            "bbox_affine": identity,
            "ecc_warp": identity,
        },
        part_id="P0001",
        palette_groups=palette_groups,
        group_id_map={"L_BLUE": "G_BLUE"},
        policy=spatial_gate.DEFAULT_POLICY,
        accepted_palette_evidence=regions,
    )
    overlap = observation["accepted_evidence_box_overlaps"][0]
    assert overlap["canonical_group_id"] == "G_BLUE"
    assert overlap["projected_overlap_pixels"] == 256
    assert overlap["projected_overlap_share"] == 1.0

    tampered = copy.deepcopy(audit)
    tampered["groups"][0]["boxes"][0]["matching_pixel_count"] = 255
    assert (
        spatial_gate._accepted_palette_evidence_regions(
            image_path=image_path,
            image_shape=(32, 32, 3),
            palette_groups=palette_groups,
            group_id_map={"L_BLUE": "G_BLUE"},
            palette_audit=tampered,
            view_id="top",
        )
        == []
    )


def test_singleton_color_claim_still_blocks_eligible_family_propagation() -> None:
    references = [
        {
            "view_id": "white_a",
            "palette_groups": [{"group_id": "WA", "base_color": "white"}],
            "group_id_map": {"WA": "G_WHITE_A"},
        },
        {
            "view_id": "white_b",
            "palette_groups": [{"group_id": "WB", "base_color": "white"}],
            "group_id_map": {"WB": "G_WHITE_A"},
        },
        {
            "view_id": "singleton",
            "palette_groups": [{"group_id": "WS", "base_color": "white"}],
            "group_id_map": {"WS": "G_WHITE_B"},
        },
        {
            "view_id": "target",
            "palette_groups": [{"group_id": "T1", "base_color": "green"}],
            "group_id_map": {"T1": "G_GREEN"},
        },
    ]

    supplements = spatial_gate._canonical_palette_supplements(references)

    assert supplements["target"] == []


def test_proof_bound_multiview_light_neutral_is_not_vetoed_by_resolved_singleton() -> None:
    unresolved = (
        "connected light neutral surface region detected from pixels; "
        "physical material unresolved"
    )
    references = [
        {
            "view_id": view_id,
            "palette_groups": [
                {
                    "group_id": local_id,
                    "base_color": "white",
                    "visual_description": unresolved,
                }
            ],
            "group_id_map": {local_id: "G_WHITE_MULTI"},
        }
        for view_id, local_id in (
            ("front", "WF"),
            ("side", "WS"),
            ("iso", "WI"),
        )
    ]
    references.extend(
        [
            {
                "view_id": "top",
                "palette_groups": [
                    {
                        "group_id": "WT",
                        "base_color": "white",
                        "visual_description": "white matte control module",
                    }
                ],
                "group_id_map": {"WT": "G_WHITE_SINGLE"},
            },
            {
                "view_id": "target",
                "palette_groups": [
                    {
                        "group_id": "TG",
                        "base_color": "green",
                        "visual_description": "green machine housing",
                    }
                ],
                "group_id_map": {"TG": "G_GREEN"},
            },
        ]
    )

    supplements = spatial_gate._canonical_palette_supplements(references)

    assert supplements["target"] == [
        {
            "canonical_group_id": "G_WHITE_MULTI",
            "base_color": "white",
            "accepted_labels": ["white"],
            "source_view_ids": ["front", "iso", "side"],
        }
    ]


def _shape(
    draw: ImageDraw.ImageDraw,
    shape_id: str,
    *,
    fill: tuple[int, int, int],
) -> None:
    # Both silhouettes are deliberately asymmetric and are not D4 equivalents.
    # This makes the synthetic reference-to-CAD association unique.
    if shape_id == "a":
        draw.polygon(
            [(25, 30), (210, 30), (210, 78), (142, 78), (142, 216), (25, 216)],
            fill=fill,
        )
        return
    if shape_id == "b":
        draw.polygon(
            [
                (35, 35),
                (185, 35),
                (185, 98),
                (222, 98),
                (222, 205),
                (96, 205),
                (96, 160),
                (35, 160),
            ],
            fill=fill,
        )
        return
    raise AssertionError(f"unknown synthetic shape: {shape_id}")


def _part_box(shape_id: str) -> tuple[int, int, int, int]:
    if shape_id == "a":
        return 160, 42, 199, 69
    if shape_id == "b":
        return 162, 125, 207, 177
    raise AssertionError(shape_id)


def _save_reference(path: Path, shape_id: str, canonical_group_id: str) -> None:
    image = Image.new("RGB", IMAGE_SIZE, BACKGROUND)
    _shape(ImageDraw.Draw(image), shape_id, fill=GROUP_COLORS[canonical_group_id])
    image.save(path)


def _save_render_pair(
    rgb_path: Path,
    ids_path: Path,
    shape_id: str,
    *,
    include_part: bool = True,
) -> int:
    rgb = Image.new("RGB", IMAGE_SIZE, BACKGROUND)
    _shape(ImageDraw.Draw(rgb), shape_id, fill=CAD_FOREGROUND)
    rgb.save(rgb_path)

    ids = Image.new("RGB", IMAGE_SIZE, (0, 0, 0))
    if include_part:
        ImageDraw.Draw(ids).rectangle(_part_box(shape_id), fill=_part_color("P0001"))
    ids.save(ids_path)
    if not include_part:
        return 0
    left, top, right, bottom = _part_box(shape_id)
    return (right - left + 1) * (bottom - top + 1)


def _palette(view_id: str, local_group_id: str, shape_id: str) -> dict[str, Any]:
    canonical_group_id = "G01" if local_group_id.startswith("L1") else "G02"
    return {
        "schema_version": "qwen-material-palette/v1",
        "source_view_id": view_id,
        "groups": [
            {
                "group_id": local_group_id,
                "family_hint": "metal",
                "base_color": "green" if canonical_group_id == "G01" else "red",
                "finish_hint": "painted",
                "visual_description": "synthetic painted region",
                "boxes": [list(_part_box(shape_id))],
                "confidence": 0.99,
            }
        ],
    }


def _palette_audit(
    image_path: Path,
    local_group_id: str,
    shape_id: str,
) -> dict[str, Any]:
    canonical_group_id = "G01" if local_group_id.startswith("L1") else "G02"
    box = list(_part_box(shape_id))
    left, top, right, bottom = box
    pixel_count = (right - left + 1) * (bottom - top + 1)
    return {
        "image": str(image_path),
        "mask": None,
        "mask_source": None,
        "mask_channel": None,
        "estimated_background_rgb": list(BACKGROUND),
        "minimum_foreground_coverage": 0.1,
        "minimum_color_match": 0.55,
        "background_distance": 28.0,
        "minimum_black_structure_coverage": 0.02,
        "accepted_group_ids": [local_group_id],
        "rejected_group_ids": [],
        "groups": [
            {
                "group_id": local_group_id,
                "base_color": "green" if canonical_group_id == "G01" else "red",
                "accepted": True,
                "source_group_ids": [local_group_id],
                "boxes": [
                    {
                        "box_index": 0,
                        "box": box,
                        "accepted": True,
                        "rejection_reasons": [],
                        "foreground_coverage": 1.0,
                        "color_match": 1.0,
                        "sampled_pixels": pixel_count,
                        "foreground_pixels": pixel_count,
                        "matching_pixels": pixel_count,
                        "matching_pixel_count": pixel_count,
                        "representative_srgb": list(GROUP_COLORS[canonical_group_id]),
                        "foreground_method": "color_distance",
                        "effective_background_distance": 28.0,
                        "accepted_color_labels": [
                            "green" if canonical_group_id == "G01" else "red"
                        ],
                        "foreground_color_counts": {
                            (
                                "green" if canonical_group_id == "G01" else "red"
                            ): pixel_count
                        },
                        "black_structure_required": False,
                        "black_structure_supported": False,
                        "black_structure_pixels": 0,
                        "largest_black_structure_pixels": 0,
                        "largest_black_structure_coverage": 0.0,
                        "largest_matching_structure_pixels": pixel_count,
                        "largest_matching_structure_coverage": 1.0,
                        "mixed_achromatic_supported": False,
                        "source_group_id": local_group_id,
                    }
                ],
            }
        ],
        "equivalent_group_merge_applied": False,
    }


def _make_case(
    tmp_path: Path,
    canonical_groups: tuple[str, ...],
) -> SpatialCase:
    tmp_path.mkdir(parents=True, exist_ok=True)
    asset = tmp_path / "asset.usda"
    asset.write_bytes(b"#usda 1.0\n")
    asset_digest = hashlib.sha256(asset.read_bytes()).hexdigest()

    source_views: list[dict[str, Any]] = []
    render_views: list[dict[str, Any]] = []
    palettes: dict[str, dict[str, Any]] = {}
    audits: dict[str, dict[str, Any]] = {}
    view_group_id_maps: dict[str, dict[str, str]] = {}
    reference_images: list[Path] = []

    for index, canonical_group_id in enumerate(canonical_groups):
        shape_id = chr(ord("a") + index)
        ref_id = f"ref_{shape_id}"
        cad_id = f"cad_{shape_id}"
        local_group_id = f"L{1 if canonical_group_id == 'G01' else 2}{index + 1:02d}"
        reference_path = tmp_path / f"{ref_id}.png"
        render_path = tmp_path / f"{cad_id}.png"
        part_ids_path = tmp_path / f"{cad_id}_part_ids.png"

        _save_reference(reference_path, shape_id, canonical_group_id)
        visible_pixels = _save_render_pair(render_path, part_ids_path, shape_id)

        source_views.append(
            {
                "id": ref_id,
                "image": str(reference_path),
                "palette_mask": None,
                "palette_status": "usable",
            }
        )
        render_views.append(
            {
                "view_id": cad_id,
                "rgb": str(render_path),
                "part_ids": str(part_ids_path),
                "visible_parts": [{"part_id": "P0001", "pixels": visible_pixels}],
                "segmentation_ids": [0, 2],
                "segmentation_labels": {
                    "0": {"class": "BACKGROUND"},
                    "2": {"part": "p0001"},
                },
            }
        )
        palettes[ref_id] = _palette(ref_id, local_group_id, shape_id)
        audits[ref_id] = _palette_audit(reference_path, local_group_id, shape_id)
        view_group_id_maps[ref_id] = {local_group_id: canonical_group_id}
        reference_images.append(reference_path)

    manifest_path = tmp_path / "reference_manifest.json"
    registry_path = tmp_path / "rendered_registry.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_views": source_views,
                "view_order_semantics": "unordered_same_asset_views",
            }
        ),
        encoding="utf-8",
    )
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "qwen-material-parts/v1",
                "asset_usd": str(asset),
                "asset_sha256": asset_digest,
                "part_count": 1,
                "parts": [
                    {
                        "part_id": "P0001",
                        "prim_path": "/Asset/P0001",
                        "face_count": 32,
                        "renders": [],
                    }
                ],
                "render_set": {
                    "asset_usd": str(asset),
                    "resolution": list(IMAGE_SIZE),
                    "views": render_views,
                },
            }
        ),
        encoding="utf-8",
    )
    return SpatialCase(
        manifest=manifest_path,
        registry=registry_path,
        view_group_id_maps=view_group_id_maps,
        palettes=palettes,
        audits=audits,
        reference_images=tuple(reference_images),
    )


def _build(case: SpatialCase) -> dict[str, Any]:
    return build_spatial_mapping_report(
        case.manifest,
        case.registry,
        case.view_group_id_maps,
        normalized_palettes_by_view=case.palettes,
        palette_audits_by_view=case.audits,
    )


def _semantic_votes() -> list[dict[str, Any]]:
    return [
        {
            "view_id": "ref_a",
            "part_id": "P0001",
            "local_group_id": "L101",
            "canonical_group_id": "G01",
            "status": "matched",
            "confidence": 0.96,
            "reason_code": "direct_visual_match",
        },
        {
            "view_id": "ref_b",
            "part_id": "P0001",
            "local_group_id": "L102",
            "canonical_group_id": "G01",
            "status": "matched",
            "confidence": 0.95,
            "reason_code": "direct_visual_match",
        },
    ]


def _build_with_semantic_votes(case: SpatialCase) -> dict[str, Any]:
    return build_spatial_mapping_report(
        case.manifest,
        case.registry,
        case.view_group_id_maps,
        _semantic_votes(),
        normalized_palettes_by_view=case.palettes,
        palette_audits_by_view=case.audits,
    )


def _resign(report: dict[str, Any]) -> None:
    report["integrity"] = {
        "report_sha256": spatial_gate._sha256_document(
            {key: value for key, value in report.items() if key != "integrity"}
        )
    }


def _mapping(
    *,
    status: str = "matched",
    group_id: str | None = "G01",
    confidence: float = 0.96,
    reason_code: str = "direct_visual_match",
) -> dict[str, Any]:
    return {
        "part_id": "P0001",
        "group_id": group_id,
        "mapping_confidence": confidence,
        "evidence_view_id": "ref_a" if group_id is not None else None,
        "evidence_box_index": 0 if group_id is not None else None,
        "status": status,
        "reason_code": reason_code,
    }


def _batches(mapping: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "qwen-part-palette-map/v1",
            "batch_id": "B01",
            "mappings": [mapping or _mapping()],
        }
    ]


def _output_mapping(result: dict[str, Any]) -> dict[str, Any]:
    return result["gate_batches"][0]["mappings"][0]


def _direct_observation(
    reference_rgb: np.ndarray,
    part_mask: np.ndarray,
    *,
    bbox_affine: np.ndarray | None = None,
    ecc_warp: np.ndarray | None = None,
    minimum_visible_pixels: int = 8,
    reference_foreground: np.ndarray | None = None,
    canonical_palette_groups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Exercise part projection without coupling a test to view association."""

    if reference_rgb.ndim != 3 or reference_rgb.shape[2] != 3:
        raise AssertionError("reference_rgb must be an HxWx3 RGB image")
    if part_mask.shape != reference_rgb.shape[:2]:
        raise AssertionError("part_mask/reference dimensions differ")
    part_ids = np.zeros_like(reference_rgb, dtype=np.uint8)
    red, green, blue = _part_color("P0001")
    part_ids[part_mask] = (blue, green, red)
    identity = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    policy = dict(spatial_gate.DEFAULT_POLICY)
    policy.update(
        {
            "minimum_visible_pixels": minimum_visible_pixels,
            "minimum_color_share": 0.70,
            "minimum_color_margin": 0.30,
        }
    )
    return spatial_gate._project_part_observation(
        reference={
            "view_id": "ref_direct",
            # The implementation samples OpenCV-style BGR pixels.
            "image": reference_rgb[:, :, ::-1].copy(),
            "foreground": (
                reference_foreground
                if reference_foreground is not None
                else np.full(part_mask.shape, 255, dtype=np.uint8)
            ),
        },
        render={
            "view_id": "cad_direct",
            "visible_pixels": {"P0001": int(np.count_nonzero(part_mask))},
            "part_ids_image": part_ids,
        },
        alignment={
            "quarter_turns_ccw": 0,
            "bbox_affine": (
                bbox_affine if bbox_affine is not None else identity
            ).tolist(),
            "ecc_warp": (ecc_warp if ecc_warp is not None else identity).tolist(),
        },
        part_id="P0001",
        palette_groups=[
            {"group_id": "L_GREEN", "base_color": "green", "boxes": [[0, 0, 1, 1]]},
            {"group_id": "L_RED", "base_color": "red", "boxes": [[0, 0, 1, 1]]},
        ],
        group_id_map={"L_GREEN": "G01", "L_RED": "G02"},
        policy=policy,
        canonical_palette_groups=canonical_palette_groups or [],
    )


def test_isolated_multiview_metadata_enables_tiny_part_diagnostic_only() -> None:
    reference_rgb = np.full((64, 64, 3), (45, 145, 62), dtype=np.uint8)
    part_mask = np.zeros((64, 64), dtype=bool)
    part_mask[28:33, 29:35] = True
    part_ids = np.zeros_like(reference_rgb, dtype=np.uint8)
    red, green, blue = _part_color("P0001")
    part_ids[part_mask] = (blue, green, red)
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    observation = spatial_gate._project_part_observation(
        reference={
            "view_id": "ref_front",
            "image": reference_rgb[:, :, ::-1].copy(),
            "foreground": np.full((64, 64), 255, dtype=np.uint8),
        },
        render={
            "view_id": "front",
            "visible_pixels": {"P0001": 30},
            "part_ids_image": part_ids,
        },
        alignment={
            "quarter_turns_ccw": 0,
            "bbox_affine": identity,
            "ecc_warp": identity,
        },
        part_id="P0001",
        palette_groups=[
            {"group_id": "L_GREEN", "base_color": "green"},
            {"group_id": "L_RED", "base_color": "red"},
        ],
        group_id_map={"L_GREEN": "G01", "L_RED": "G02"},
        policy=spatial_gate.DEFAULT_POLICY,
        isolated_evidence={
            "schema_version": "qwen-isolated-part-evidence/v1",
            "sha256": "a" * 64,
            "source_visible_pixels_by_view": {"front": 30, "rear": 24},
            "source_evidence_view_count": 2,
            "material_neutralized": True,
            "background_removed": True,
        },
    )

    assert observation["classification"] == "insufficient_visibility"
    assert observation["reason_code"] == "part_visible_pixels_below_floor"
    assert observation["evidence_mode"] == "isolated_mask_multiview_diagnostic"
    assert observation["declared_visible_pixels"] == 30
    assert observation["small_part_diagnostic"]["status"] == "resolved"
    assert observation["small_part_diagnostic"]["canonical_group_id"] == "G01"


def _dark_direct_observation(
    reference_rgb: np.ndarray,
    part_mask: np.ndarray,
    *,
    alignment_overrides: dict[str, Any] | None = None,
    canonical_palette_groups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Project one part through the strict dark-on-black diagnostic."""

    part_ids = np.zeros_like(reference_rgb, dtype=np.uint8)
    red, green, blue = _part_color("P0001")
    part_ids[part_mask] = (blue, green, red)
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    alignment: dict[str, Any] = {
        "quarter_turns_ccw": 0,
        "bbox_affine": identity,
        "ecc_warp": identity,
        "trusted": True,
        "reason_codes": [],
        "score": 0.95,
        "projection_score": 0.95,
        "projection_iou": 0.95,
        "ecc_status": "success",
        "ecc_correlation": 0.96,
        "ecc_transform_audit": {"constraints_passed": True},
    }
    alignment.update(alignment_overrides or {})
    policy = dict(spatial_gate.DEFAULT_POLICY)
    policy.update(
        {
            "minimum_visible_pixels": 64,
            "minimum_diagnostic_visible_pixels": 64,
        }
    )
    return spatial_gate._project_part_observation(
        reference={
            "view_id": "dark_direct",
            "image": reference_rgb[:, :, ::-1].copy(),
            "foreground": part_mask.astype(np.uint8) * 255,
        },
        render={
            "view_id": "cad_direct",
            "visible_pixels": {"P0001": int(np.count_nonzero(part_mask))},
            "part_ids_image": part_ids,
        },
        alignment=alignment,
        part_id="P0001",
        palette_groups=[
            {
                "group_id": "L_BLACK",
                "base_color": "black",
                "boxes": [[0, 0, 1, 1]],
            }
        ],
        group_id_map={"L_BLACK": "G_BLACK"},
        policy=policy,
        unique_canonical_palette_groups=canonical_palette_groups
        or [
            {
                "canonical_group_id": "G_BLACK",
                "base_color": "black",
                "accepted_labels": ["black", "darkgray"],
                "source_view_ids": ["ref_a", "ref_b"],
            }
        ],
    )


def _textured_dark_object(
    *,
    image_size: int = 128,
    object_slice: tuple[slice, slice] = (slice(48, 80), slice(48, 80)),
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.full((image_size, image_size, 3), 5, dtype=np.uint8)
    part_mask = np.zeros((image_size, image_size), dtype=bool)
    part_mask[object_slice] = True
    y_coordinates, x_coordinates = np.indices(part_mask.shape)
    values = np.where(
        ((x_coordinates // 2 + y_coordinates // 2) % 2) == 0,
        35,
        70,
    ).astype(np.uint8)
    reference[part_mask] = np.repeat(values[:, :, None], 3, axis=2)[part_mask]
    return reference, part_mask


def test_dark_foreground_diagnostic_resolves_textured_near_black_object() -> None:
    reference, part_mask = _textured_dark_object()

    observation = _dark_direct_observation(reference, part_mask)

    diagnostic = observation["dark_foreground_diagnostic"]
    assert observation["classification"] == "resolved"
    assert observation["canonical_group_id"] == "G_BLACK"
    assert diagnostic["status"] == "resolved"
    assert diagnostic["reason_codes"] == []
    assert diagnostic["normalization"]["long_edge_pixels"] == 512
    assert diagnostic["near_black_share"] >= 0.60
    assert diagnostic["dark_signal_share"] >= 0.20
    assert diagnostic["core_dark_signal_share"] >= 0.25
    assert diagnostic["adaptive_edge_density"] >= 0.25
    assert diagnostic["valid_null_shift_count"] >= 4
    assert diagnostic["dark_signal_null_margin"] >= 0.10
    assert diagnostic["canonical_source_view_ids"] == ["ref_a", "ref_b"]
    for field in (
        "normalized_projected_mask_sha256",
        "normalized_non_background_mask_sha256",
        "normalized_dark_signal_mask_sha256",
    ):
        assert len(diagnostic[field]) == 64
    unsigned = copy.deepcopy(diagnostic)
    signature = unsigned.pop("diagnostic_sha256")
    assert signature == spatial_gate._sha256_document(unsigned)


def test_singleton_black_group_does_not_block_multiview_dark_authority() -> None:
    reference, part_mask = _textured_dark_object()

    observation = _dark_direct_observation(
        reference,
        part_mask,
        canonical_palette_groups=[
            {
                "canonical_group_id": "G_SINGLETON",
                "base_color": "black",
                "accepted_labels": ["black", "darkgray"],
                "source_view_ids": ["iso"],
            },
            {
                "canonical_group_id": "G_BLACK",
                "base_color": "black",
                "accepted_labels": ["black", "darkgray"],
                "source_view_ids": ["front", "side", "top"],
            },
        ],
    )

    diagnostic = observation["dark_foreground_diagnostic"]
    assert observation["classification"] == "resolved"
    assert observation["canonical_group_id"] == "G_BLACK"
    assert diagnostic["status"] == "resolved"
    assert diagnostic["eligible_multiview_black_group_count"] == 1
    assert diagnostic["non_authoritative_singleton_black_group_ids"] == [
        "G_SINGLETON"
    ]


def test_local_black_projection_on_pure_black_background_is_not_resolved() -> None:
    reference = np.full((128, 128, 3), 5, dtype=np.uint8)
    part_mask = np.zeros((128, 128), dtype=bool)
    part_mask[48:80, 48:80] = True

    observation = _dark_direct_observation(reference, part_mask)

    diagnostic = observation["dark_foreground_diagnostic"]
    assert observation["classification"] == "conflict"
    assert observation["reason_code"] == "black_projection_lacks_dark_foreground_proof"
    assert observation["canonical_group_id"] == "G_BLACK"
    assert diagnostic["status"] == "rejected"
    assert diagnostic["dark_signal_pixels"] == 0
    assert "DARK_NON_BACKGROUND_PIXELS_BELOW_FLOOR" in diagnostic["reason_codes"]
    assert "DARK_NULL_Q75_MARGIN_BELOW_FLOOR" in diagnostic["reason_codes"]


def test_green_foreground_cannot_pass_near_black_diagnostic() -> None:
    reference = np.full((128, 128, 3), 5, dtype=np.uint8)
    part_mask = np.zeros((128, 128), dtype=bool)
    part_mask[48:80, 48:80] = True
    reference[part_mask] = (20, 70, 30)

    observation = _dark_direct_observation(reference, part_mask)

    diagnostic = observation["dark_foreground_diagnostic"]
    assert diagnostic["status"] == "rejected"
    assert diagnostic["near_black_share"] < 0.60
    assert "DARK_NEAR_BLACK_SHARE_BELOW_FLOOR" in diagnostic["reason_codes"]


def test_dark_diagnostic_rejects_when_bbox_shifts_lack_valid_area() -> None:
    reference, part_mask = _textured_dark_object(
        object_slice=(slice(14, 114), slice(14, 114))
    )

    observation = _dark_direct_observation(reference, part_mask)

    diagnostic = observation["dark_foreground_diagnostic"]
    assert diagnostic["status"] == "rejected"
    assert diagnostic["valid_null_shift_count"] < 4
    assert "DARK_VALID_NULL_SHIFTS_BELOW_FLOOR" in diagnostic["reason_codes"]


def test_dark_diagnostic_requires_strong_trusted_registration() -> None:
    reference, part_mask = _textured_dark_object()

    observation = _dark_direct_observation(
        reference,
        part_mask,
        alignment_overrides={"score": 0.84},
    )

    diagnostic = observation["dark_foreground_diagnostic"]
    assert diagnostic["status"] == "rejected"
    assert diagnostic["alignment"]["strong"] is False
    assert "DARK_ALIGNMENT_NOT_STRONG" in diagnostic["reason_codes"]


def test_canonical_supplement_cannot_resolve_background_only_projection() -> None:
    reference = np.zeros((48, 48, 3), dtype=np.uint8)
    reference[8:40, 4:20] = GROUP_COLORS["G01"]
    foreground = np.zeros((48, 48), dtype=np.uint8)
    foreground[8:40, 4:20] = 255
    part_mask = np.zeros((48, 48), dtype=bool)
    part_mask[8:40, 28:44] = True

    observation = _direct_observation(
        reference,
        part_mask,
        minimum_visible_pixels=64,
        reference_foreground=foreground,
        canonical_palette_groups=[
            {
                "canonical_group_id": "G_BLACK",
                "base_color": "black",
                "accepted_labels": ["black", "darkgray"],
                "source_view_ids": ["ref_a", "ref_b"],
            }
        ],
    )

    assert observation["classification"] != "resolved"
    assert observation.get("canonical_group_id") != "G_BLACK"
    diagnostic = observation["canonical_palette_diagnostic"]
    assert diagnostic["status"] == "rejected"
    assert (
        "CANONICAL_SUPPLEMENT_FOREGROUND_OVERLAP_BELOW_FLOOR"
        in diagnostic["reason_codes"]
    )


def test_canonical_supplement_is_repair_only_on_object_foreground() -> None:
    reference = np.zeros((48, 48, 3), dtype=np.uint8)
    part_mask = np.zeros((48, 48), dtype=bool)
    part_mask[8:40, 12:36] = True
    foreground = part_mask.astype(np.uint8) * 255

    observation = _direct_observation(
        reference,
        part_mask,
        minimum_visible_pixels=64,
        reference_foreground=foreground,
        canonical_palette_groups=[
            {
                "canonical_group_id": "G_BLACK",
                "base_color": "black",
                "accepted_labels": ["black", "darkgray"],
                "source_view_ids": ["ref_a", "ref_b"],
            }
        ],
    )

    assert observation["classification"] != "resolved"
    diagnostic = observation["canonical_palette_diagnostic"]
    assert diagnostic["status"] == "resolved"
    assert diagnostic["canonical_group_id"] == "G_BLACK"
    assert diagnostic["registration_label_stable"] is True
    assert diagnostic["perturbation_label_stable"] is True


@pytest.mark.parametrize(
    ("reference_id", "expected"),
    [
        ("ref_front", ["front"]),
        ("ref_rear", ["rear"]),
        ("ref_back", ["rear"]),
        ("ref_left", ["left"]),
        ("ref_right", ["right"]),
    ],
)
def test_explicit_cardinal_reference_uses_only_same_direction_render(
    reference_id: str,
    expected: list[str],
) -> None:
    candidates, paired_direction_family = spatial_gate._candidate_render_ids(
        reference_id,
        ["front", "rear", "left", "right", "top", "iso"],
    )

    assert candidates == expected
    assert paired_direction_family is False


def test_generic_side_reference_still_compares_left_and_right() -> None:
    candidates, paired_direction_family = spatial_gate._candidate_render_ids(
        "ref_side",
        ["front", "rear", "left", "right", "top", "iso"],
    )

    assert candidates == ["left", "right"]
    assert paired_direction_family is True


def test_dense_pose_bank_uses_bounded_search_then_refines_only_selected_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    metric_call = 0

    def fake_alignment_metrics(
        _reference_mask: np.ndarray,
        _render_mask: np.ndarray,
        _size: int,
    ) -> dict[str, float]:
        nonlocal metric_call
        score = (0.95, 0.45, 0.35, 0.25)[metric_call % 4]
        metric_call += 1
        return {"score": score}

    def fake_refine_projection(
        reference_mask: np.ndarray,
        render_mask: np.ndarray,
        _policy: dict[str, float | int],
    ) -> dict[str, Any]:
        call_shapes.append((reference_mask.shape, render_mask.shape))
        return {
            "bbox_affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "ecc_warp": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "ecc_status": "success",
            "ecc_correlation": 0.95,
            "projection_iou_before": 0.9,
            "projection_iou": 0.9,
            "ecc_transform_audit": {
                "constraints_passed": True,
                "constraint_failures": [],
            },
        }

    monkeypatch.setattr(spatial_gate, "_alignment_metrics", fake_alignment_metrics)
    monkeypatch.setattr(spatial_gate, "_refine_projection", fake_refine_projection)
    reference_mask = np.zeros((480, 640), dtype=np.uint8)
    reference_mask[40:440, 80:560] = 255
    render_mask = np.zeros((64, 64), dtype=np.uint8)
    render_mask[8:56, 12:52] = 255
    render_ids = [
        "front",
        "rear",
        "left",
        "right",
        "top",
        "iso",
        "pose_a000_e035",
    ]

    alignment = spatial_gate._associate_views(
        [{"view_id": "reference", "foreground": reference_mask}],
        [
            {"view_id": view_id, "foreground": render_mask}
            for view_id in render_ids
        ],
        spatial_gate.DEFAULT_POLICY,
    )[0]

    assert len(call_shapes) == len(render_ids) * 4 + 1
    assert all(
        reference_shape == (256, 256) and render_shape == (256, 256)
        for reference_shape, render_shape in call_shapes[:-1]
    )
    assert call_shapes[-1] == ((480, 640), (64, 64))
    assert alignment["pose_search_mask_size"] == 256
    assert alignment["pose_search_method"] == (
        "bounded_dense_search_then_full_resolution_selected_refinement"
    )


def test_explicit_front_label_does_not_bypass_geometric_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    per_render_scores = (0.95, 0.55, 0.45, 0.35)
    scores = iter(per_render_scores * 2)
    projection_scores = iter(per_render_scores * 2)

    def fake_alignment_metrics(
        _reference_mask: np.ndarray,
        _render_mask: np.ndarray,
        _size: int,
    ) -> dict[str, float]:
        return {"score": next(scores)}

    def fake_refine_projection(
        _reference_mask: np.ndarray,
        _render_mask: np.ndarray,
        _policy: dict[str, float | int],
    ) -> dict[str, Any]:
        projection_iou = next(projection_scores)
        return {
            "bbox_affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "ecc_warp": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "ecc_status": "success",
            "ecc_correlation": 0.95,
            "projection_iou_before": projection_iou,
            "projection_iou": projection_iou,
            "ecc_transform_audit": {
                "constraints_passed": True,
                "constraint_failures": [],
            },
        }

    monkeypatch.setattr(
        spatial_gate,
        "_alignment_metrics",
        fake_alignment_metrics,
    )
    monkeypatch.setattr(
        spatial_gate,
        "_refine_projection",
        fake_refine_projection,
    )
    mask = np.full((16, 16), 255, dtype=np.uint8)

    alignments = spatial_gate._associate_views(
        [{"view_id": "front", "foreground": mask}],
        [
            {"view_id": "front", "foreground": mask},
            {"view_id": "rear", "foreground": mask},
        ],
        spatial_gate.DEFAULT_POLICY,
    )

    alignment = alignments[0]
    assert alignment["selected_render_view_id"] == "front"
    assert alignment["paired_direction_family"] is False
    assert alignment["render_margin"] == pytest.approx(0.0)
    assert alignment["trusted"] is False
    assert "global_render_assignment_ambiguous" in alignment["reason_codes"]


def test_strong_refined_global_pose_can_override_coarse_local_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Coarse shape prefers ``front`` (0.95 > 0.80), while the refined
    # silhouette registration plus the one-to-one assignment prefers ``iso``.
    # Every quantitative trust gate still clears its fixed threshold.
    scores = iter((0.95, 0.70, 0.60, 0.50, 0.80, 0.65, 0.55, 0.45))
    projection_ious = iter((0.70, 0.65, 0.60, 0.55, 0.94, 0.70, 0.60, 0.55))

    def fake_alignment_metrics(
        _reference_mask: np.ndarray,
        _render_mask: np.ndarray,
        _size: int,
    ) -> dict[str, float]:
        return {"score": next(scores)}

    def fake_refine_projection(
        _reference_mask: np.ndarray,
        _render_mask: np.ndarray,
        _policy: dict[str, float | int],
    ) -> dict[str, Any]:
        projection_iou = next(projection_ious)
        return {
            "bbox_affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "ecc_warp": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "ecc_status": "success",
            "ecc_correlation": 0.90,
            "projection_iou_before": projection_iou,
            "projection_iou": projection_iou,
            "ecc_transform_audit": {
                "constraints_passed": True,
                "constraint_failures": [],
            },
        }

    monkeypatch.setattr(spatial_gate, "_alignment_metrics", fake_alignment_metrics)
    monkeypatch.setattr(spatial_gate, "_refine_projection", fake_refine_projection)
    mask = np.full((16, 16), 255, dtype=np.uint8)

    alignment = spatial_gate._associate_views(
        [{"view_id": "ref_iso", "foreground": mask}],
        [
            {"view_id": "front", "foreground": mask},
            {"view_id": "iso", "foreground": mask},
        ],
        spatial_gate.DEFAULT_POLICY,
    )[0]

    assert alignment["selected_render_view_id"] == "iso"
    assert alignment["score"] == pytest.approx(0.80)
    assert alignment["projection_iou"] == pytest.approx(0.94)
    assert (
        alignment["render_margin"]
        > spatial_gate.DEFAULT_POLICY["minimum_render_margin"]
    )
    assert alignment["reason_codes"] == []
    assert alignment["warning_codes"] == [
        "refined_render_direction_disagrees_with_raw_shape"
    ]
    assert alignment["registration_authority"] == (
        "whole_asset_uniform_scale_rotation_translation"
    )
    assert alignment["trusted"] is True
    assert alignment["observation_eligible"] is True


def test_spatial_progress_reports_render_and_part_counts_without_changing_report(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, ("G01", "G01"))
    events: list[dict[str, Any]] = []

    report = build_spatial_mapping_report(
        case.manifest,
        case.registry,
        case.view_group_id_maps,
        _semantic_votes(),
        normalized_palettes_by_view=case.palettes,
        palette_audits_by_view=case.audits,
        progress_callback=events.append,
    )

    render_events = [
        event for event in events if event["stage"] == "spatial_render_decode"
    ]
    assert [(event["state"], event["current"]) for event in render_events] == [
        ("start", 0),
        ("update", 1),
        ("update", 2),
        ("complete", 2),
    ]
    assert {event["total"] for event in render_events} == {2}
    alignment_events = [
        event for event in events if event["stage"] == "spatial_view_alignment"
    ]
    assert [
        (event["state"], event["current"]) for event in alignment_events
    ] == [
        ("start", 0),
        ("update", 1),
        ("update", 2),
        ("update", 3),
        ("update", 4),
        ("complete", 4),
    ]
    assert {event["total"] for event in alignment_events} == {4}
    assert {event["unit"] for event in alignment_events} == {"pairs"}
    semantic_vote_events = [
        event for event in events if event["stage"] == "spatial_semantic_votes"
    ]
    assert [
        (event["state"], event["current"]) for event in semantic_vote_events
    ] == [
        ("start", 0),
        ("update", 1),
        ("update", 2),
        ("complete", 2),
    ]
    assert {event["total"] for event in semantic_vote_events} == {2}
    assert {event["unit"] for event in semantic_vote_events} == {"votes"}
    part_events = [
        event for event in events if event["stage"] == "spatial_part_observations"
    ]
    assert [(event["state"], event["current"]) for event in part_events] == [
        ("start", 0),
        ("update", 1),
        ("complete", 1),
    ]
    assert report["schema_version"] == spatial_gate.SCHEMA_VERSION
    assert "progress" not in report


def test_spatial_alignment_progress_does_not_complete_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, Any]] = []
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[4:28, 6:22] = 255

    def fail_projection(
        _reference_mask: np.ndarray,
        _render_mask: np.ndarray,
        _policy: dict[str, float | int],
    ) -> dict[str, Any]:
        raise RuntimeError("synthetic ECC failure")

    monkeypatch.setattr(spatial_gate, "_refine_projection", fail_projection)

    with pytest.raises(RuntimeError, match="synthetic ECC failure"):
        spatial_gate._associate_views(
            [{"view_id": "ref_a", "foreground": mask}],
            [{"view_id": "cad_a", "foreground": mask}],
            spatial_gate.DEFAULT_POLICY,
            progress_callback=events.append,
        )

    alignment_events = [
        event for event in events if event["stage"] == "spatial_view_alignment"
    ]
    assert [(event["state"], event["current"]) for event in alignment_events] == [
        ("start", 0)
    ]
    assert not any(event["state"] == "complete" for event in alignment_events)


def test_spatial_semantic_vote_progress_does_not_complete_on_invalid_vote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path, ("G01",))
    events: list[dict[str, Any]] = []
    invalid_vote = {
        "view_id": "ref_a",
        "part_id": "P0001",
        "local_group_id": "L101",
        "canonical_group_id": "G01",
        "status": "matched",
        "confidence": 1.5,
        "reason_code": "direct_visual_match",
    }

    monkeypatch.setattr(
        spatial_gate,
        "_associate_views",
        lambda _references, _renders, _policy, *, progress_callback=None: [],
    )

    with pytest.raises(SpatialMappingError, match="invalid confidence"):
        build_spatial_mapping_report(
            case.manifest,
            case.registry,
            case.view_group_id_maps,
            [invalid_vote],
            normalized_palettes_by_view=case.palettes,
            palette_audits_by_view=case.audits,
            progress_callback=events.append,
        )

    semantic_vote_events = [
        event for event in events if event["stage"] == "spatial_semantic_votes"
    ]
    assert [
        (event["state"], event["current"]) for event in semantic_vote_events
    ] == [("start", 0)]
    assert not any(event["state"] == "complete" for event in semantic_vote_events)


def test_observation_mapping_persists_every_trusted_pose(tmp_path: Path) -> None:
    report = _build(_make_case(tmp_path, ("G01", "G01")))

    trusted = {
        alignment["reference_view_id"]: alignment["selected_render_view_id"]
        for alignment in report["view_alignments"]
        if alignment["trusted"]
    }
    observed_views = {
        observation["reference_view_id"]
        for observation in report["parts"][0]["observations"]
    }

    assert report["observation_view_mapping"] == trusted
    assert observed_views == set(trusted)
    assert report["summary"]["observation_eligible_alignment_count"] == len(trusted)
    for evidence in report["reference_evidence"]:
        assert evidence["alignment_observation_eligible"] is True
        assert evidence["alignment_reason_codes"] == []
        assert isinstance(evidence["alignment_warning_codes"], list)


def test_two_independent_same_group_spatial_supports_keep_matched(
    tmp_path: Path,
) -> None:
    report = _build(_make_case(tmp_path, ("G01", "G01")))

    result = apply_spatial_gate_to_batches(_batches(), report)

    row = _output_mapping(result)
    assert row["status"] == "matched"
    assert row["group_id"] == "G01"
    assert row["mapping_confidence"] == pytest.approx(0.96)
    decision = result["audit"]["decisions"][0]
    assert decision["part_id"] == "P0001"
    assert decision["decision"] in {"kept_auto", "unchanged_matched"}
    assert result["audit"]["summary"]["decision_count"] == 1


def test_distinct_content_and_pose_semantic_consensus_keeps_matched(
    tmp_path: Path,
) -> None:
    report = _build_with_semantic_votes(_make_case(tmp_path, ("G01", "G01")))
    report["parts"][0]["observations"] = []
    _resign(report)

    result = apply_spatial_gate_to_batches(_batches(), report)

    assert _output_mapping(result)["status"] == "matched"
    decision = result["audit"]["decisions"][0]
    assert decision["validation_lanes"] == ["semantic_multiview"]
    assert len(decision["semantic_supporting_content_cluster_ids"]) == 2
    assert len(decision["semantic_supporting_pose_cluster_ids"]) == 2


def test_semantic_vote_from_aligned_pose_where_part_is_invisible_cannot_validate(
    tmp_path: Path,
) -> None:
    report = _build_with_semantic_votes(_make_case(tmp_path, ("G01", "G01")))
    report["parts"][0]["observations"] = []
    semantic_votes = report["parts"][0]["semantic_votes"]
    assert all(vote["cad_part_visibility_eligible"] is True for vote in semantic_votes)

    semantic_votes[1]["cad_part_visible_pixels"] = 0
    semantic_votes[1]["cad_part_visibility_eligible"] = False
    semantic_votes[1]["cad_part_evidence_mode"] = "source_projection"
    semantic_votes[1]["isolated_evidence_sha256"] = None
    _resign(report)

    result = apply_spatial_gate_to_batches(_batches(), report)

    assert _output_mapping(result)["status"] == "review"
    decision = result["audit"]["decisions"][0]
    assert decision["validation_lanes"] == []
    assert decision["semantic_supporting_view_ids"] == ["ref_a"]


def test_near_duplicate_reference_cluster_is_counted_once(tmp_path: Path) -> None:
    report = _build_with_semantic_votes(_make_case(tmp_path, ("G01", "G01")))
    report["parts"][0]["observations"] = []
    semantic_votes = report["parts"][0]["semantic_votes"]
    semantic_votes[1]["content_cluster_id"] = semantic_votes[0]["content_cluster_id"]
    semantic_votes[1]["normalized_pixel_sha256"] = semantic_votes[0][
        "normalized_pixel_sha256"
    ]
    _resign(report)

    result = apply_spatial_gate_to_batches(_batches(), report)

    assert _output_mapping(result)["status"] == "review"
    assert result["audit"]["decisions"][0]["validation_lanes"] == []


def test_palette_confidence_caps_semantic_vote_confidence(tmp_path: Path) -> None:
    report = _build_with_semantic_votes(_make_case(tmp_path, ("G01", "G01")))
    report["parts"][0]["observations"] = []
    report["parts"][0]["semantic_votes"][1]["palette_confidence"] = 0.60
    report["parts"][0]["semantic_votes"][1]["effective_confidence"] = 0.60
    _resign(report)

    result = apply_spatial_gate_to_batches(_batches(), report)

    assert _output_mapping(result)["status"] == "review"
    assert result["audit"]["decisions"][0]["validation_lanes"] == []


def test_one_spatial_support_downgrades_matched_to_review(tmp_path: Path) -> None:
    report = _build(_make_case(tmp_path, ("G01",)))

    result = apply_spatial_gate_to_batches(_batches(), report)

    row = _output_mapping(result)
    assert row["status"] == "review"
    assert row["group_id"] == "G01"
    assert 0.6 <= row["mapping_confidence"] < 0.9
    assert result["audit"]["decisions"][0]["decision"] == "downgraded_review"


def test_conflicting_spatial_group_support_downgrades_to_unknown(
    tmp_path: Path,
) -> None:
    report = _build(_make_case(tmp_path, ("G01", "G02")))

    result = apply_spatial_gate_to_batches(_batches(), report)

    row = _output_mapping(result)
    assert row["status"] == "unknown"
    assert row["group_id"] is None
    assert row["mapping_confidence"] < 0.6
    assert result["audit"]["decisions"][0]["decision"] in {
        "downgraded_unknown",
        "downgraded_preserve",
    }


@pytest.mark.parametrize(
    "mapping",
    [
        _mapping(
            status="review",
            confidence=0.75,
            reason_code="partial_visibility",
        ),
        _mapping(
            status="unknown",
            group_id=None,
            confidence=0.20,
            reason_code="occluded",
        ),
    ],
    ids=["review", "unknown"],
)
def test_spatial_gate_never_promotes_review_or_unknown(
    tmp_path: Path,
    mapping: dict[str, Any],
) -> None:
    report = _build(_make_case(tmp_path, ("G01", "G01")))
    original = _batches(copy.deepcopy(mapping))

    result = apply_spatial_gate_to_batches(original, report)

    assert result["gate_batches"] == original


def test_changed_source_image_after_report_build_fails_closed(tmp_path: Path) -> None:
    case = _make_case(tmp_path, ("G01", "G01"))
    report = _build(case)
    Image.new("RGB", IMAGE_SIZE, (255, 0, 255)).save(case.reference_images[0])

    with pytest.raises(SpatialMappingError, match="(?i)(hash|sha256|stale|changed)"):
        apply_spatial_gate_to_batches(_batches(), report)


def test_invalid_input_or_report_schema_fails_closed(tmp_path: Path) -> None:
    case = _make_case(tmp_path, ("G01", "G01"))
    registry = json.loads(case.registry.read_text(encoding="utf-8"))
    registry["schema_version"] = "unsupported"
    case.registry.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(SpatialMappingError, match="schema_version"):
        _build(case)

    valid_case = _make_case(tmp_path / "valid", ("G01", "G01"))
    report = _build(valid_case)
    report["schema_version"] = "unsupported"
    with pytest.raises(SpatialMappingError, match="schema_version"):
        apply_spatial_gate_to_batches(_batches(), report)


def test_raw_part_id_decoded_count_mismatch_fails_closed(tmp_path: Path) -> None:
    case = _make_case(tmp_path, ("G01",))
    registry = json.loads(case.registry.read_text(encoding="utf-8"))
    view = registry["render_set"]["views"][0]
    view["part_ids_raw"] = view["part_ids"]
    # Simulate an annotated/lossy ID AOV: the manifest still declares the
    # original exact mask count, but one encoded part pixel is no longer exact.
    ids_path = Path(view["part_ids_raw"])
    ids = Image.open(ids_path).convert("RGB")
    left, top, _, _ = _part_box("a")
    ids.putpixel((left, top), (255, 255, 255))
    ids.save(ids_path)
    case.registry.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(
        SpatialMappingError,
        match=r"raw part-ID image does not exactly match visible_parts.*cad_a",
    ):
        _build(case)


def test_projected_part_mask_must_meet_configured_pixel_floor() -> None:
    reference = np.full((40, 40, 3), GROUP_COLORS["G01"], dtype=np.uint8)
    part_mask = np.zeros((40, 40), dtype=bool)
    part_mask[5:20, 5:25] = True  # 300 declared and decoded pixels.
    observation = _direct_observation(
        reference,
        part_mask,
        bbox_affine=np.asarray([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32),
        minimum_visible_pixels=256,
    )

    assert observation["decoded_part_pixels"] == 300
    assert observation["projected_part_pixels"] < 256
    assert observation["classification"] == "insufficient_visibility"
    # The projected mask is large enough for the bounded diagnostic lane but
    # remains below the unchanged automatic-assignment floor.
    assert observation["reason_code"] == "projected_part_pixels_below_floor"


@pytest.mark.parametrize(
    ("candidate_warp", "expected_failure"),
    [
        (
            np.asarray([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            "reflection_or_singular",
        ),
        (
            np.asarray([[0.50, 0.0, 0.0], [0.0, 0.50, 0.0]], dtype=np.float32),
            "minimum_scale",
        ),
        (
            np.asarray([[1.60, 0.0, 0.0], [0.0, 1.60, 0.0]], dtype=np.float32),
            "maximum_scale",
        ),
        (
            np.asarray([[1.50, 0.0, 0.0], [0.0, 0.65, 0.0]], dtype=np.float32),
            "condition_number",
        ),
        (
            np.asarray(
                [
                    [math.cos(math.radians(20.0)), -math.sin(math.radians(20.0)), 0.0],
                    [math.sin(math.radians(20.0)), math.cos(math.radians(20.0)), 0.0],
                ],
                dtype=np.float32,
            ),
            "rotation",
        ),
        (
            np.asarray([[1.0, 0.70, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            "shear",
        ),
        (
            np.asarray([[1.0, 0.0, 20.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            "translation",
        ),
    ],
    ids=[
        "reflection",
        "minimum-scale",
        "maximum-scale",
        "condition-number",
        "rotation",
        "shear",
        "translation",
    ],
)
def test_ecc_transform_outside_constraints_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    candidate_warp: np.ndarray,
    expected_failure: str,
) -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[9:53, 7:48] = 255

    def fake_find_transform_ecc(
        *_args: Any, **_kwargs: Any
    ) -> tuple[float, np.ndarray]:
        return 0.99, candidate_warp.copy()

    monkeypatch.setattr(spatial_gate.cv2, "findTransformECC", fake_find_transform_ecc)
    result = spatial_gate._refine_projection_affine_ecc(
        mask,
        mask,
        spatial_gate.DEFAULT_POLICY,
    )

    assert result["ecc_status"] == "rejected_transform_constraints"
    assert result["ecc_correlation"] == 0.0
    assert result["ecc_transform_audit"]["constraints_passed"] is False
    assert expected_failure in result["ecc_transform_audit"]["constraint_failures"]
    assert result["ecc_warp"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def test_bbox_vs_ecc_material_label_flip_is_conflict() -> None:
    reference = np.zeros((64, 64, 3), dtype=np.uint8)
    reference[:, :32] = GROUP_COLORS["G01"]
    reference[:, 32:] = GROUP_COLORS["G02"]
    part_mask = np.zeros((64, 64), dtype=bool)
    part_mask[12:44, 8:28] = True
    observation = _direct_observation(
        reference,
        part_mask,
        # WARP_INVERSE_MAP with negative x translation moves the mask right.
        ecc_warp=np.asarray([[1.0, 0.0, -32.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    )

    assert observation["bbox_canonical_group_id"] == "G01"
    assert observation["canonical_group_id"] == "G02"
    assert observation["registration_label_stable"] is False
    assert observation["classification"] == "conflict"
    assert observation["reason_code"] == "registration_material_label_flip"


def test_two_pixel_projection_perturbation_label_flip_is_conflict() -> None:
    reference = np.zeros((48, 48, 3), dtype=np.uint8)
    reference[:, :24] = GROUP_COLORS["G01"]
    reference[:, 24:] = GROUP_COLORS["G02"]
    part_mask = np.zeros((48, 48), dtype=bool)
    part_mask[8:40, 22:24] = True
    observation = _direct_observation(reference, part_mask)

    assert observation["bbox_canonical_group_id"] == "G01"
    assert observation["registration_label_stable"] is True
    assert observation["perturbation_label_stable"] is False
    assert any(
        row["offset_pixels"] == [2, 0] and row["canonical_group_id"] == "G02"
        for row in observation["projection_perturbations"]
    )
    assert observation["classification"] == "conflict"
    assert observation["reason_code"] == "projection_perturbation_material_instability"
    assert observation["small_part_diagnostic"]["status"] == "resolved"
    assert observation["small_part_diagnostic"]["canonical_group_id"] == "G01"


def test_two_view_dark_interior_consensus_recovers_black_background_case() -> None:
    def observation(view_id: str) -> dict[str, Any]:
        return {
            "reference_view_id": view_id,
            "classification": "conflict",
            "reason_code": "black_projection_lacks_dark_foreground_proof",
            "canonical_group_id": "G01",
            "registration_label_stable": True,
            "perturbation_label_stable": True,
            "bbox_canonical_group_id": "G01",
            "group_scores": [
                {
                    "canonical_group_id": "G01",
                    "base_color": "black",
                    "color_share": 0.82,
                }
            ],
            "dark_foreground_diagnostic": {
                "non_background_share": 0.70,
                "dark_signal_share": 0.55,
                "dark_signal_purity": 0.80,
                "core_dark_signal_share": 0.60,
                "dark_signal_null_margin": 0.35,
            },
        }

    observations = [observation("front"), observation("iso")]
    audit = spatial_gate._recover_multiview_dark_consensus(
        observations,
        policy=spatial_gate.DEFAULT_POLICY,
    )

    assert audit == {
        "status": "resolved",
        "canonical_group_id": "G01",
        "supporting_view_ids": ["front", "iso"],
        "minimum_independent_support_views": 2,
        "evidence_contract": (
            "stable_projection_and_dark_interior_multiview_consensus"
        ),
    }
    assert {row["classification"] for row in observations} == {"resolved"}
    assert {row["reason_code"] for row in observations} == {
        "multiview_dark_consensus_resolved"
    }


def test_single_dark_background_projection_remains_fail_closed() -> None:
    observations = [
        {
            "reference_view_id": "iso",
            "classification": "conflict",
            "reason_code": "black_projection_lacks_dark_foreground_proof",
            "canonical_group_id": "G01",
            "registration_label_stable": True,
            "perturbation_label_stable": True,
            "bbox_canonical_group_id": "G01",
            "group_scores": [
                {
                    "canonical_group_id": "G01",
                    "base_color": "black",
                    "color_share": 0.95,
                }
            ],
            "dark_foreground_diagnostic": {
                "non_background_share": 0.80,
                "dark_signal_share": 0.70,
                "dark_signal_purity": 0.90,
                "core_dark_signal_share": 0.70,
                "dark_signal_null_margin": 0.50,
            },
        }
    ]

    assert (
        spatial_gate._recover_multiview_dark_consensus(
            observations,
            policy=spatial_gate.DEFAULT_POLICY,
        )
        is None
    )
    assert observations[0]["classification"] == "conflict"
