from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from qwen_material_pipeline.materials.visual_group_annotation import (
    MINIMUM_SINGLE_VIEW_SPATIAL_PIXELS,
    EXIT_REQUIRE_UNAMBIGUOUS_FAILED,
    EXIT_SUCCESS,
    MAXIMUM_NEIGHBOR_CONFIDENCE,
    VisualGroupAnnotationError,
    annotate_visual_groups,
    main,
)
from qwen_material_pipeline.usd.material_common import normalize_face_subsets


def _fusion(*, duplicate_green_metal: bool = False) -> dict:
    groups = [
        {
            "group_id": "G01",
            "base_color": "black",
            "family_hint": "metal",
            "confidence": 0.86,
        },
        {
            "group_id": "G02",
            "base_color": "silver",
            "family_hint": "metal",
            "confidence": 0.72,
        },
        {
            "group_id": "G04",
            "base_color": "white",
            "family_hint": "plastic",
            "confidence": 0.92,
        },
        {
            "group_id": "G06",
            "base_color": "green",
            "family_hint": "metal",
            "confidence": 0.91,
        },
        {
            "group_id": "G07",
            "base_color": "orange",
            "family_hint": "metal",
            "confidence": 0.68,
        },
    ]
    if duplicate_green_metal:
        groups.append(
            {
                "group_id": "G08",
                "base_color": "green",
                "family_hint": "metal",
                "confidence": 0.91,
            }
        )
    return {"canonical_palette": {"groups": groups}}


def _assignment(
    part_id: str,
    semantic: str,
    material_id: str,
    **extra: object,
) -> dict:
    return {
        "part_id": part_id,
        "semantic": semantic,
        "material_id": material_id,
        **extra,
    }


def _cohort_registry_part(
    part_id: str,
    parent_path: str,
    *,
    geometry_marker: str = "a",
    appearance_marker: str = "b",
    subset_marker: str = "c",
) -> dict:
    return {
        "part_id": part_id,
        "prim_path": f"{parent_path}/Mesh",
        "parent_path": parent_path,
        "point_count": 8,
        "face_count": 12,
        "world_bbox": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        "existing_material_bind_face_subsets": [],
        "geometry_content_sha256": geometry_marker * 64,
        "source_appearance_sha256": appearance_marker * 64,
        "source_subset_layout_sha256": subset_marker * 64,
    }


def _cohort_registry(parts: list[dict]) -> dict:
    return {
        "schema_version": "qwen-material-parts/v1",
        "part_count": len(parts),
        "parts": parts,
    }


def _spatial_report(parts: list[dict]) -> dict:
    report = {
        "schema_version": "qwen-spatial-mapping-audit/v1",
        "policy": {"minimum_spatial_support_views": 2},
        "parts": parts,
    }
    payload = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    report["integrity"] = {
        "report_sha256": hashlib.sha256(payload).hexdigest()
    }
    return report


def test_trusted_spatial_anchor_propagates_only_lineage_to_stable_assembly() -> None:
    materials = {
        "P0001": "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless",
        "P0002": "mdl:Base/Metals/Galvanized_Steel.mdl#Galvanized_Steel",
        "P0003": "mdl:Base/Metals/Aluminum.mdl#Aluminum",
        "P0004": "mdl:Base/Metals/Copper.mdl#Copper",
    }
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                part_id,
                "neutral unresolved component",
                material_id,
                status="policy_fallback",
                parameters={"roughness": 0.5},
            )
            for part_id, material_id in materials.items()
        ],
    }
    registry = _cohort_registry(
        [
            _cohort_registry_part(
                "P0001",
                "/Asset/Assembly/P0001",
                geometry_marker="a",
            ),
            _cohort_registry_part(
                "P0002",
                "/Asset/Assembly/P0002",
                geometry_marker="d",
            ),
            _cohort_registry_part(
                "P0003",
                "/Asset/Assembly/P0003",
                geometry_marker="e",
            ),
            # Identical source appearance/layout outside the selected authored
            # assembly must not join its deepest valid cohort.
            _cohort_registry_part(
                "P0004",
                "/Asset/Other/P0004",
                geometry_marker="f",
            ),
        ]
    )
    report = _spatial_report(
        [
            {
                "part_id": "P0001",
                "resolved_support_counts": {"G06": 2},
                "observations": [
                    _resolved_observation("front", "G06"),
                    _resolved_observation("side", "G06"),
                ],
            }
        ]
    )

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=_fusion(),
        spatial_mapping_report=report,
        part_registry=registry,
    )

    by_part = {item["part_id"]: item for item in annotated["assignments"]}
    assert {
        part_id: item["material_id"] for part_id, item in by_part.items()
    } == materials
    assert all(item["parameters"] == {"roughness": 0.5} for item in by_part.values())
    assert all(
        by_part[part_id]["provenance"]["canonical_group_id"] == "G06"
        for part_id in ("P0001", "P0002", "P0003")
    )
    assert "canonical_group_id" not in by_part["P0004"].get("provenance", {})
    cohort = audit["source_appearance_cohort_propagation"]
    assert cohort["exact_cover"] is True
    assert cohort["cohort_count"] == 1
    assert cohort["propagated_member_count"] == 2
    contract = cohort["contracts"][0]
    assert contract["candidate_kind"] == "dominant_assembly"
    assert contract["cohort_signature_kind"] == (
        "source_appearance_plus_subset_layout"
    )
    assert contract["assembly_path"] == "/Asset/Assembly"
    assert contract["anchor_part_ids"] == ["P0001"]
    assert contract["expected_member_part_ids"] == ["P0001", "P0002", "P0003"]
    assert contract["propagated_member_part_ids"] == ["P0002", "P0003"]
    assert (
        annotated["provenance"]["visual_group_annotation"]
        ["source_appearance_cohort_propagation"]
        == cohort
    )


def test_authoritative_group_conflict_vetoes_whole_appearance_cohort() -> None:
    material_id = "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless"
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "neutral",
                material_id,
                status="policy_fallback",
            ),
            _assignment(
                "P0002",
                "black metal",
                material_id,
                provenance={"canonical_group_id": "G01"},
            ),
            _assignment(
                "P0003",
                "neutral",
                material_id,
                status="policy_fallback",
            ),
        ],
    }
    registry = _cohort_registry(
        [
            _cohort_registry_part(part_id, f"/Asset/Assembly/{part_id}")
            for part_id in ("P0001", "P0002", "P0003")
        ]
    )
    report = _spatial_report(
        [
            {
                "part_id": "P0001",
                "resolved_support_counts": {"G06": 2},
                "observations": [
                    _resolved_observation("front", "G06"),
                    _resolved_observation("side", "G06"),
                ],
            }
        ]
    )

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=_fusion(),
        spatial_mapping_report=report,
        part_registry=registry,
    )

    by_part = {item["part_id"]: item for item in annotated["assignments"]}
    assert by_part["P0001"]["provenance"]["canonical_group_id"] == "G06"
    assert by_part["P0002"]["provenance"]["canonical_group_id"] == "G01"
    assert "canonical_group_id" not in by_part["P0003"].get("provenance", {})
    cohort = audit["source_appearance_cohort_propagation"]
    assert cohort["cohort_count"] == 0
    assert cohort["propagated_member_count"] == 0
    assert any(
        rejection["reason_codes"] == ["AUTHORITATIVE_MEMBER_CONFLICT"]
        for rejection in cohort["rejected_candidates"]
    )


def test_globally_unique_sibling_pair_uses_bounded_cohort_lane() -> None:
    material_id = "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless"
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                part_id,
                "neutral",
                material_id,
                status="policy_fallback",
            )
            for part_id in ("P0001", "P0002", "P0003")
        ],
    }
    registry = _cohort_registry(
        [
            _cohort_registry_part("P0001", "/Asset/Assembly/Left"),
            _cohort_registry_part("P0002", "/Asset/Assembly/Right"),
            _cohort_registry_part(
                "P0003",
                "/Asset/Assembly/Other",
                geometry_marker="d",
                appearance_marker="e",
            ),
        ]
    )
    report = _spatial_report(
        [
            {
                "part_id": "P0001",
                "resolved_support_counts": {"G06": 2},
                "observations": [
                    _resolved_observation("front", "G06"),
                    _resolved_observation("side", "G06"),
                ],
            }
        ]
    )

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=_fusion(),
        spatial_mapping_report=report,
        part_registry=registry,
    )

    by_part = {item["part_id"]: item for item in annotated["assignments"]}
    assert by_part["P0002"]["provenance"]["canonical_group_id"] == "G06"
    assert "canonical_group_id" not in by_part["P0003"].get("provenance", {})
    contract = audit["source_appearance_cohort_propagation"]["contracts"][0]
    assert contract["candidate_kind"] == "rare_source_appearance_pair"
    assert contract["expected_member_part_ids"] == ["P0001", "P0002"]


def test_unique_visual_sibling_pair_can_differ_in_geometry() -> None:
    material_id = "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless"
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                part_id,
                "neutral",
                material_id,
                status="policy_fallback",
            )
            for part_id in ("P0001", "P0002", "P0003")
        ],
    }
    registry = _cohort_registry(
        [
            _cohort_registry_part(
                "P0001",
                "/Asset/Assembly/Left",
                geometry_marker="a",
            ),
            _cohort_registry_part(
                "P0002",
                "/Asset/Assembly/Right",
                geometry_marker="d",
            ),
            _cohort_registry_part(
                "P0003",
                "/Asset/Assembly/Other",
                geometry_marker="e",
                appearance_marker="f",
            ),
        ]
    )
    report = _spatial_report(
        [
            {
                "part_id": "P0001",
                "resolved_support_counts": {"G06": 2},
                "observations": [
                    _resolved_observation("front", "G06"),
                    _resolved_observation("side", "G06"),
                ],
            }
        ]
    )

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=_fusion(),
        spatial_mapping_report=report,
        part_registry=registry,
    )

    by_part = {item["part_id"]: item for item in annotated["assignments"]}
    assert by_part["P0002"]["provenance"]["canonical_group_id"] == "G06"
    assert "canonical_group_id" not in by_part["P0003"].get("provenance", {})
    contract = audit["source_appearance_cohort_propagation"]["contracts"][0]
    assert contract["candidate_kind"] == "rare_source_appearance_layout_pair"
    assert contract["cohort_signature_kind"] == (
        "source_appearance_plus_subset_layout"
    )
    assert contract["expected_member_part_ids"] == ["P0001", "P0002"]


def _resolved_observation(view_id: str, group_id: str) -> dict:
    return {
        "reference_view_id": view_id,
        "classification": "resolved",
        "canonical_group_id": group_id,
        "registration_label_stable": True,
        "perturbation_label_stable": True,
    }


def test_exact_repeated_subset_layout_gets_multiview_visual_lineage() -> None:
    material_id = "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless"
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                part_id,
                "neutral unresolved component",
                material_id,
                status="policy_fallback",
                face_subsets=[
                    {
                        "subset_name": "Body",
                        "semantic": "neutral unresolved subset",
                        "material_id": material_id,
                        "face_indices": list(range(0, 6)),
                    },
                    {
                        "subset_name": "Cap",
                        "semantic": "neutral unresolved subset",
                        "material_id": material_id,
                        "face_indices": list(range(6, 12)),
                    },
                ],
            )
            for part_id in ("P0001", "P0002")
        ],
    }
    registry_parts = [
        _cohort_registry_part(
            part_id,
            f"/Asset/Repeated/{part_id}",
            geometry_marker="a",
            subset_marker="c",
        )
        for part_id in ("P0001", "P0002")
    ]
    for part in registry_parts:
        part["existing_material_bind_face_subsets"] = [
            {"subset_name": "Body", "face_indices": list(range(0, 6))},
            {"subset_name": "Cap", "face_indices": list(range(6, 12))},
        ]
    report = _spatial_report(
        [
            {
                "part_id": "P0001",
                "observations": [_resolved_observation("front", "G06")],
            },
            {
                "part_id": "P0002",
                "observations": [_resolved_observation("side", "G06")],
            },
        ]
    )
    fusion = _fusion()
    for group in fusion["canonical_palette"]["groups"]:
        group["source_view_ids"] = (
            ["front", "side"] if group["group_id"] == "G06" else []
        )

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=fusion,
        spatial_mapping_report=report,
        part_registry=_cohort_registry(registry_parts),
    )

    for assignment in annotated["assignments"]:
        assert assignment["material_id"] == material_id
        assert assignment["provenance"]["canonical_group_id"] == "G06"
        assert assignment["provenance"][
            "face_subset_canonical_group_ids"
        ] == {"Body": "G06", "Cap": "G06"}
        assert all(
            subset["material_id"] == material_id
            and "provenance" not in subset
            for subset in assignment["face_subsets"]
        )
    subset_audit = audit["repeated_subset_visual_cohort"]
    assert subset_audit["cohort_count"] == 1
    assert subset_audit["annotated_part_count"] == 2
    assert subset_audit["annotated_face_subset_count"] == 4
    assert subset_audit["contracts"][0]["candidate_kind"] == (
        "exact_repeated_geometry_subset_layout"
    )
    assert audit["summary"]["assignment_part_ids_by_group"] == {
        "G06": ["P0001", "P0002"]
    }
    assert audit["summary"]["face_subsets_by_group"] == {
        "G06": [
            "P0001:Body",
            "P0001:Cap",
            "P0002:Body",
            "P0002:Cap",
        ]
    }


def test_repeated_subset_layout_fails_closed_on_anchor_group_conflict() -> None:
    material_id = "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless"
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                part_id,
                "neutral unresolved component",
                material_id,
                status="policy_fallback",
                face_subsets=[
                    {
                        "subset_name": "Body",
                        "semantic": "neutral",
                        "material_id": material_id,
                        "face_indices": list(range(12)),
                    }
                ],
            )
            for part_id in ("P0001", "P0002")
        ],
    }
    registry_parts = [
        _cohort_registry_part(
            part_id,
            f"/Asset/Repeated/{part_id}",
            geometry_marker="a",
            subset_marker="c",
        )
        for part_id in ("P0001", "P0002")
    ]
    for part in registry_parts:
        part["existing_material_bind_face_subsets"] = [
            {"subset_name": "Body", "face_indices": list(range(12))}
        ]
    fusion = _fusion()
    for group in fusion["canonical_palette"]["groups"]:
        group["source_view_ids"] = ["front", "side"]

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=fusion,
        spatial_mapping_report=_spatial_report(
            [
                {
                    "part_id": "P0001",
                    "observations": [_resolved_observation("front", "G06")],
                },
                {
                    "part_id": "P0002",
                    "observations": [_resolved_observation("side", "G01")],
                },
            ]
        ),
        part_registry=_cohort_registry(registry_parts),
    )

    assert all(
        "canonical_group_id" not in assignment.get("provenance", {})
        for assignment in annotated["assignments"]
    )
    subset_audit = audit["repeated_subset_visual_cohort"]
    assert subset_audit["cohort_count"] == 0
    assert subset_audit["rejected_candidates"][0]["reason_codes"] == [
        "REPEATED_SUBSET_ANCHOR_GROUP_CONFLICT"
    ]


def _high_confidence_single_view_observation(
    view_id: str,
    group_id: str,
) -> dict:
    return {
        "reference_view_id": view_id,
        "classification": "resolved",
        "canonical_group_id": group_id,
        "registration_label_stable": True,
        "perturbation_label_stable": True,
        "projected_part_pixels": 1536,
        "sampled_reference_pixels": 1536,
        "group_scores": [
            {
                "canonical_group_id": group_id,
                "matching_pixels": 1475,
                "color_share": 0.96,
            },
            {
                "canonical_group_id": "G01",
                "matching_pixels": 31,
                "color_share": 0.02,
            },
        ],
        "color_margin": 0.94,
        "bbox_canonical_group_id": group_id,
        "bbox_color_margin": 0.93,
        "canonical_palette_diagnostic": {
            "direct_sample": {
                "sampled_foreground_pixels": 1450,
                "foreground_overlap_ratio": 0.94,
            }
        },
        "projection_perturbations": [
            {
                "offset_pixels": offset,
                "canonical_group_id": group_id,
                "diagnostic_canonical_group_id": group_id,
            }
            for offset in ([-2, 0], [2, 0], [0, -2], [0, 2])
        ],
    }


def _thin_semantic_vote(
    view_id: str,
    group_id: str,
    *,
    reference_marker: str,
    content_cluster_id: str,
    alignment_trusted: bool = True,
) -> dict:
    return {
        "view_id": view_id,
        "reference_sha256": reference_marker * 64,
        "content_cluster_id": content_cluster_id,
        "canonical_group_id": group_id,
        "status": "review",
        "effective_confidence": 0.68,
        "pixel_gate_accepted": True,
        "unique_canonical_join": True,
        "cad_part_visible_pixels": 240,
        "cad_part_evidence_mode": (
            "source_projection" if alignment_trusted else "isolated_mask_multiview"
        ),
        "alignment_trusted": alignment_trusted,
        "isolated_evidence_sha256": (
            None if alignment_trusted else "f" * 64
        ),
    }


def _direct_multiview_observation(
    view_id: str,
    group_id: str,
    *,
    color_share: float,
    color_margin: float,
    pixels: int,
    stable: bool,
    bbox_group_id: str | None = None,
    classification: str = "insufficient_visibility",
) -> dict:
    other_group_id = "G06" if group_id == "G07" else "G07"
    runner_up_share = max(0.0, color_share - color_margin)
    return {
        "reference_view_id": view_id,
        "classification": classification,
        "canonical_group_id": (
            group_id if classification in {"resolved", "conflict"} else None
        ),
        "evidence_mode": (
            "isolated_mask_multiview_diagnostic"
            if view_id == "front"
            else "source_projection"
        ),
        "registration_label_stable": stable,
        "perturbation_label_stable": stable,
        "projected_part_pixels": pixels,
        "sampled_reference_pixels": pixels,
        "group_scores": [
            {
                "canonical_group_id": group_id,
                "matching_pixels": round(color_share * pixels),
                "color_share": color_share,
            },
            {
                "canonical_group_id": other_group_id,
                "matching_pixels": round(runner_up_share * pixels),
                "color_share": runner_up_share,
            },
        ],
        "color_margin": color_margin,
        "bbox_canonical_group_id": bbox_group_id,
        "bbox_color_margin": 0.9,
        "canonical_palette_diagnostic": {
            "direct_sample": {
                "sampled_foreground_pixels": pixels,
                "foreground_overlap_ratio": 1.0,
            }
        },
    }


def test_policy_fallback_gets_target_lineage_from_unique_spatial_multiview() -> None:
    material_id = "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless"
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "neutral delivery material",
                material_id,
                status="policy_fallback",
            ),
            _assignment(
                "P0002",
                "neutral delivery material",
                material_id,
                status="policy_fallback",
                face_subsets=[
                    {
                        "subset_name": "Cover",
                        "semantic": "neutral delivery material",
                        "material_id": material_id,
                        "face_indices": [0],
                    }
                ],
            ),
        ],
    }
    report = _spatial_report(
        [
            {
                "part_id": "P0001",
                "resolved_support_counts": {"G06": 2},
                "observations": [
                    _resolved_observation("front", "G06"),
                    _resolved_observation("side", "G06"),
                ],
            },
            {
                "part_id": "P0002",
                "resolved_support_counts": {"G06": 2},
                "observations": [
                    _resolved_observation("front", "G06"),
                    _resolved_observation("side", "G06"),
                ],
            },
        ]
    )

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=_fusion(),
        spatial_mapping_report=report,
    )

    first, second = annotated["assignments"]
    assert first["material_id"] == material_id
    assert first["provenance"]["canonical_group_id"] == "G06"
    annotation = first["provenance"]["canonical_group_annotation"]
    assert annotation["method"] == "trusted_multiview_spatial_projection/v1"
    assert annotation["supporting_view_ids"] == ["front", "side"]
    assert "canonical_group_id" not in second.get("provenance", {})
    assert audit["spatial_annotation"]["annotated_part_count"] == 1
    assert audit["summary"]["assignment_part_ids_by_group"] == {"G06": ["P0001"]}


def test_thin_part_gets_lineage_from_corroborated_semantic_projections() -> None:
    material_id = "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless"
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "neutral delivery material",
                material_id,
                status="policy_fallback",
            )
        ],
    }
    report = _spatial_report(
        [
            {
                "part_id": "P0001",
                "resolved_support_counts": {},
                "observations": [
                    {
                        "reference_view_id": "front",
                        "classification": "conflict",
                        "canonical_group_id": "G07",
                        "projected_part_pixels": 210,
                        "registration_label_stable": None,
                        "perturbation_label_stable": False,
                    }
                ],
                "semantic_votes": [
                    _thin_semantic_vote(
                        "front",
                        "G07",
                        reference_marker="a",
                        content_cluster_id="PH01",
                    ),
                    _thin_semantic_vote(
                        "iso",
                        "G07",
                        reference_marker="b",
                        content_cluster_id="PH02",
                        alignment_trusted=False,
                    ),
                ],
            }
        ]
    )

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=_fusion(),
        spatial_mapping_report=report,
    )

    assignment = annotated["assignments"][0]
    assert assignment["material_id"] == material_id
    assert assignment["provenance"]["canonical_group_id"] == "G07"
    annotation = assignment["provenance"]["canonical_group_annotation"]
    assert annotation["method"] == (
        "corroborated_multiview_semantic_projection/v1"
    )
    assert annotation["supporting_view_ids"] == ["front", "iso"]
    assert annotation["spatial_corroborating_view_ids"] == ["front"]
    assert audit["spatial_annotation"][
        "semantic_recovery_annotated_part_count"
    ] == 1


def test_thin_part_semantic_conflict_remains_unannotated() -> None:
    material_id = "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless"
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "neutral delivery material",
                material_id,
                status="policy_fallback",
            )
        ],
    }
    report = _spatial_report(
        [
            {
                "part_id": "P0001",
                "resolved_support_counts": {},
                "observations": [
                    {
                        "reference_view_id": "front",
                        "classification": "conflict",
                        "canonical_group_id": "G07",
                        "projected_part_pixels": 210,
                    }
                ],
                "semantic_votes": [
                    _thin_semantic_vote(
                        "front",
                        "G07",
                        reference_marker="a",
                        content_cluster_id="PH01",
                    ),
                    _thin_semantic_vote(
                        "iso",
                        "G06",
                        reference_marker="b",
                        content_cluster_id="PH02",
                    ),
                ],
            }
        ]
    )

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=_fusion(),
        spatial_mapping_report=report,
    )

    assert "canonical_group_id" not in annotated["assignments"][0].get(
        "provenance", {}
    )
    assert audit["spatial_annotation"]["annotated_part_count"] == 0


def test_small_part_diagnostic_uses_independent_multiview_palette_support() -> None:
    material_id = "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless"
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "neutral delivery material",
                material_id,
                status="policy_fallback",
            )
        ],
    }
    fusion = _fusion()
    orange = next(
        group
        for group in fusion["canonical_palette"]["groups"]
        if group["group_id"] == "G07"
    )
    orange["source_view_ids"] = ["front", "iso", "side"]
    report = _spatial_report(
        [
            {
                "part_id": "P0001",
                "resolved_support_counts": {},
                "observations": [
                    {
                        "reference_view_id": "front",
                        "classification": "insufficient_visibility",
                        "reason_code": "part_visible_pixels_below_floor",
                        "canonical_group_id": None,
                        "registration_label_stable": None,
                        "perturbation_label_stable": None,
                        "small_part_diagnostic": {
                            "status": "resolved",
                            "reason_codes": [],
                            "local_group_id": "L_ORANGE",
                            "canonical_group_id": "G07",
                            "bbox_canonical_group_id": "G07",
                            "registration_label_stable": True,
                            "resolved_sample_count": 6,
                            "target_sample_count": 6,
                            "consensus_ratio": 1.0,
                            "alternative_canonical_group_ids": [],
                        },
                    }
                ],
                "semantic_votes": [],
            }
        ]
    )

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=fusion,
        spatial_mapping_report=report,
    )

    assignment = annotated["assignments"][0]
    assert assignment["material_id"] == material_id
    assert assignment["provenance"]["canonical_group_id"] == "G07"
    annotation = assignment["provenance"]["canonical_group_annotation"]
    assert annotation["method"] == (
        "multiview_palette_bound_small_part_projection/v1"
    )
    assert annotation["supporting_view_ids"] == ["front"]
    assert annotation["canonical_palette_source_view_ids"] == [
        "front",
        "iso",
        "side",
    ]
    assert audit["spatial_annotation"][
        "palette_bound_small_part_recovery_annotated_part_count"
    ] == 1


def test_small_part_diagnostic_rejects_strong_semantic_group_conflict() -> None:
    material_id = "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless"
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "neutral delivery material",
                material_id,
                status="policy_fallback",
            )
        ],
    }
    fusion = _fusion()
    orange = next(
        group
        for group in fusion["canonical_palette"]["groups"]
        if group["group_id"] == "G07"
    )
    orange["source_view_ids"] = ["front", "iso", "side"]
    report = _spatial_report(
        [
            {
                "part_id": "P0001",
                "resolved_support_counts": {},
                "observations": [
                    {
                        "reference_view_id": "front",
                        "classification": "insufficient_visibility",
                        "reason_code": "part_visible_pixels_below_floor",
                        "small_part_diagnostic": {
                            "status": "resolved",
                            "reason_codes": [],
                            "canonical_group_id": "G07",
                            "bbox_canonical_group_id": "G07",
                            "registration_label_stable": True,
                            "resolved_sample_count": 6,
                            "target_sample_count": 6,
                            "consensus_ratio": 1.0,
                            "alternative_canonical_group_ids": [],
                        },
                    }
                ],
                "semantic_votes": [
                    {
                        "view_id": "iso",
                        "canonical_group_id": "G06",
                        "effective_confidence": 0.91,
                    }
                ],
            }
        ]
    )

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=fusion,
        spatial_mapping_report=report,
    )

    assert "canonical_group_id" not in annotated["assignments"][0].get(
        "provenance", {}
    )
    rejection = audit["spatial_annotation"][
        "palette_bound_small_part_rejections"
    ][0]
    assert rejection["reason_codes"] == [
        "SMALL_PART_DIAGNOSTIC_STRONG_SEMANTIC_GROUP_CONFLICT"
    ]


def test_thin_part_gets_lineage_from_three_view_direct_mask_consensus() -> None:
    material_id = "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless"
    parameters = {"roughness": 0.41}
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "neutral delivery material",
                material_id,
                status="policy_fallback",
                parameters=parameters,
            )
        ],
    }
    report = _spatial_report(
        [
            {
                "part_id": "P0001",
                "resolved_support_counts": {},
                "observations": [
                    _direct_multiview_observation(
                        "front",
                        "G07",
                        color_share=0.78,
                        color_margin=0.78,
                        pixels=176,
                        stable=True,
                    ),
                    _direct_multiview_observation(
                        "side",
                        "G07",
                        color_share=0.48,
                        color_margin=0.05,
                        pixels=125,
                        stable=False,
                        bbox_group_id="G06",
                    ),
                    _direct_multiview_observation(
                        "iso",
                        "G07",
                        color_share=0.68,
                        color_margin=0.44,
                        pixels=216,
                        stable=False,
                        bbox_group_id="G06",
                    ),
                ],
                "semantic_votes": [],
            }
        ]
    )

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=_fusion(),
        spatial_mapping_report=report,
    )

    assignment = annotated["assignments"][0]
    assert assignment["material_id"] == material_id
    assert assignment["parameters"] == parameters
    assert assignment["provenance"]["canonical_group_id"] == "G07"
    annotation = assignment["provenance"]["canonical_group_annotation"]
    assert annotation["method"] == (
        "corroborated_multiview_direct_mask_projection/v1"
    )
    assert annotation["supporting_view_ids"] == ["front", "iso"]
    assert annotation["direct_agreement_view_ids"] == [
        "front",
        "iso",
        "side",
    ]
    evidence = annotation["projection_evidence"]
    assert evidence["bbox_conflict_view_ids"] == ["iso", "side"]
    assert evidence["bbox_conflict_is_diagnostic_only"] is True
    assert evidence["reason_codes"] == []
    spatial_audit = audit["spatial_annotation"]
    assert spatial_audit[
        "direct_multiview_recovery_annotated_part_count"
    ] == 1
    assert spatial_audit["annotations"] == [
        {
            "part_id": "P0001",
            "canonical_group_id": "G07",
            "supporting_view_ids": ["front", "iso"],
            "method": "corroborated_multiview_direct_mask_projection/v1",
        }
    ]


def test_direct_mask_consensus_rejects_black_background_outside_sam3() -> None:
    material_id = "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless"
    observations = [
        _direct_multiview_observation(
            view_id,
            "G01",
            color_share=1.0,
            color_margin=1.0,
            pixels=512,
            stable=True,
        )
        for view_id in ("front", "side", "iso")
    ]
    for observation in observations:
        observation["canonical_palette_diagnostic"]["direct_sample"] = {
            "sampled_foreground_pixels": 0,
            "foreground_overlap_ratio": 0.0,
        }
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "neutral delivery material",
                material_id,
                status="policy_fallback",
            )
        ],
    }

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=_fusion(),
        spatial_mapping_report=_spatial_report(
            [
                {
                    "part_id": "P0001",
                    "resolved_support_counts": {},
                    "observations": observations,
                    "semantic_votes": [],
                }
            ]
        ),
    )

    assert "canonical_group_id" not in annotated["assignments"][0].get(
        "provenance", {}
    )
    assert audit["spatial_annotation"][
        "direct_multiview_recovery_annotated_part_count"
    ] == 0


def test_sam3_foreground_single_view_assigns_part_absent_from_other_views() -> None:
    material_id = "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless"
    observation = _direct_multiview_observation(
        "iso",
        "G07",
        color_share=0.93,
        color_margin=0.86,
        pixels=95,
        stable=False,
    )
    observation["perturbation_label_stable"] = True
    observation["canonical_palette_diagnostic"]["direct_sample"][
        "group_scores"
    ] = [
        {
            "canonical_group_id": "G07",
            "matching_pixels": 88,
            "color_share": 0.93,
        },
        {
            "canonical_group_id": "G06",
            "matching_pixels": 7,
            "color_share": 0.07,
        },
    ]
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "neutral delivery material",
                material_id,
                status="policy_fallback",
            )
        ],
    }

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=_fusion(),
        spatial_mapping_report=_spatial_report(
            [
                {
                    "part_id": "P0001",
                    "resolved_support_counts": {},
                    "observations": [observation],
                    "semantic_votes": [],
                }
            ]
        ),
    )

    assignment = annotated["assignments"][0]
    assert assignment["material_id"] == material_id
    assert assignment["provenance"]["canonical_group_id"] == "G07"
    annotation = assignment["provenance"]["canonical_group_annotation"]
    assert annotation["method"] == "sam3_foreground_single_view_projection/v1"
    assert annotation["supporting_view_ids"] == ["iso"]
    assert annotation["unseen_views_cast_no_vote"] is True
    spatial_audit = audit["spatial_annotation"]
    assert spatial_audit[
        "foreground_single_view_recovery_annotated_part_count"
    ] == 1


@pytest.mark.parametrize(
    ("failure_mode", "expected_reason_code"),
    [
        ("too_few_views", None),
        (
            "too_few_strong_views",
            "DIRECT_MULTIVIEW_STRONG_SUPPORT_BELOW_MINIMUM",
        ),
        ("strong_direct_conflict", "DIRECT_MULTIVIEW_STRONG_GROUP_CONFLICT"),
        (
            "strong_semantic_conflict",
            "DIRECT_MULTIVIEW_STRONG_SEMANTIC_GROUP_CONFLICT",
        ),
        (
            "stable_resolved_conflict",
            "DIRECT_MULTIVIEW_STABLE_RESOLVED_GROUP_CONFLICT",
        ),
    ],
)
def test_direct_multiview_projection_rejects_incomplete_or_conflicting_evidence(
    failure_mode: str,
    expected_reason_code: str | None,
) -> None:
    material_id = "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless"
    observations = [
        _direct_multiview_observation(
            "front",
            "G07",
            color_share=0.78,
            color_margin=0.78,
            pixels=176,
            stable=True,
        ),
        _direct_multiview_observation(
            "side",
            "G07",
            color_share=0.48,
            color_margin=0.05,
            pixels=125,
            stable=False,
            bbox_group_id="G06",
        ),
        _direct_multiview_observation(
            "iso",
            "G07",
            color_share=0.68,
            color_margin=0.44,
            pixels=216,
            stable=False,
            bbox_group_id="G06",
        ),
    ]
    semantic_votes: list[dict] = []
    if failure_mode == "too_few_views":
        observations.pop(1)
    elif failure_mode == "too_few_strong_views":
        observations[2] = _direct_multiview_observation(
            "iso",
            "G07",
            color_share=0.55,
            color_margin=0.30,
            pixels=216,
            stable=False,
            bbox_group_id="G06",
        )
    elif failure_mode == "strong_direct_conflict":
        observations.append(
            _direct_multiview_observation(
                "rear",
                "G06",
                color_share=0.90,
                color_margin=0.80,
                pixels=256,
                stable=True,
            )
        )
    elif failure_mode == "strong_semantic_conflict":
        semantic_votes.append(
            {
                "view_id": "rear",
                "canonical_group_id": "G06",
                "effective_confidence": 0.90,
                "pixel_gate_accepted": True,
                "unique_canonical_join": True,
            }
        )
    elif failure_mode == "stable_resolved_conflict":
        observations.append(
            _direct_multiview_observation(
                "rear",
                "G06",
                color_share=0.90,
                color_margin=0.80,
                pixels=256,
                stable=True,
                classification="resolved",
            )
        )
    else:
        raise AssertionError(f"unhandled failure mode {failure_mode}")

    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "neutral delivery material",
                material_id,
                status="policy_fallback",
            )
        ],
    }
    report = _spatial_report(
        [
            {
                "part_id": "P0001",
                "resolved_support_counts": {},
                "observations": observations,
                "semantic_votes": semantic_votes,
            }
        ]
    )

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=_fusion(),
        spatial_mapping_report=report,
    )

    assert "canonical_group_id" not in annotated["assignments"][0].get(
        "provenance", {}
    )
    spatial_audit = audit["spatial_annotation"]
    assert spatial_audit[
        "direct_multiview_recovery_annotated_part_count"
    ] == 0
    if expected_reason_code is None:
        assert spatial_audit["direct_multiview_rejections"] == []
    else:
        assert expected_reason_code in spatial_audit[
            "direct_multiview_rejections"
        ][0]["reason_codes"]


def test_large_single_view_projection_recovers_lineage_and_overrides_semantic_vote() -> None:
    material_id = "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless"
    parameters = {"roughness": 0.37}
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "neutral delivery material",
                material_id,
                status="policy_fallback",
                parameters=parameters,
            )
        ],
    }
    report = _spatial_report(
        [
            {
                "part_id": "P0001",
                "resolved_support_counts": {"G06": 1},
                "observations": [
                    _high_confidence_single_view_observation("iso", "G06")
                ],
                # The semantic lane disagrees.  Only the complete strong
                # spatial gate is allowed to override that weaker identity.
                "semantic_votes": [
                    {
                        "view_id": "iso",
                        "canonical_group_id": "G04",
                    }
                ],
            }
        ]
    )

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=_fusion(),
        spatial_mapping_report=report,
    )

    assignment = annotated["assignments"][0]
    assert assignment["material_id"] == material_id
    assert assignment["parameters"] == parameters
    assert assignment["provenance"]["canonical_group_id"] == "G06"
    annotation = assignment["provenance"]["canonical_group_annotation"]
    assert annotation["method"] == (
        "high_confidence_single_view_spatial_projection/v1"
    )
    evidence = annotation["projection_evidence"]
    assert evidence["semantic_conflicting_group_ids"] == ["G04"]
    assert evidence["semantic_vote_conflict_overridden"] is True
    record = audit["records"][0]
    assert record["confidence_tier"] == (
        "HIGH_CONFIDENCE_SINGLE_VIEW_SPATIAL_PROJECTION"
    )
    assert record["reason_codes"] == [
        "HIGH_CONFIDENCE_SINGLE_VIEW_SPATIAL_PROJECTION_ACCEPTED"
    ]
    spatial_audit = audit["spatial_annotation"]
    assert spatial_audit["single_view_recovery_annotated_part_count"] == 1
    assert spatial_audit["single_view_rejections"] == []
    assert spatial_audit["annotations"] == [
        {
            "part_id": "P0001",
            "canonical_group_id": "G06",
            "supporting_view_ids": ["iso"],
            "method": "high_confidence_single_view_spatial_projection/v1",
        }
    ]


@pytest.mark.parametrize(
    ("failure_mode", "expected_reason_code"),
    [
        ("insufficient_pixels", "SINGLE_VIEW_PROJECTED_PIXELS_BELOW_MINIMUM"),
        ("bbox_conflict", "SINGLE_VIEW_BBOX_CANONICAL_GROUP_MISMATCH"),
        ("perturbation_conflict", "SINGLE_VIEW_PERTURBATION_GROUP_CONFLICT"),
        ("stable_conflict", "SINGLE_VIEW_STABLE_SPATIAL_CONFLICT_PRESENT"),
        (
            "foreground_mismatch",
            "SINGLE_VIEW_FOREGROUND_OVERLAP_BELOW_MINIMUM",
        ),
    ],
)
def test_single_view_projection_rejects_incomplete_strong_evidence(
    failure_mode: str,
    expected_reason_code: str,
) -> None:
    material_id = "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless"
    observation = _high_confidence_single_view_observation("iso", "G06")
    observations = [observation]
    if failure_mode == "insufficient_pixels":
        observation["projected_part_pixels"] = (
            MINIMUM_SINGLE_VIEW_SPATIAL_PIXELS - 1
        )
        observation["sampled_reference_pixels"] = (
            MINIMUM_SINGLE_VIEW_SPATIAL_PIXELS - 1
        )
    elif failure_mode == "bbox_conflict":
        observation["bbox_canonical_group_id"] = "G07"
    elif failure_mode == "perturbation_conflict":
        observation["projection_perturbations"][0][
            "canonical_group_id"
        ] = "G07"
    elif failure_mode == "stable_conflict":
        observations.append(
            {
                "reference_view_id": "front",
                "classification": "conflict",
                "canonical_group_id": "G07",
                "registration_label_stable": True,
                "perturbation_label_stable": True,
            }
        )
    elif failure_mode == "foreground_mismatch":
        observation["canonical_palette_diagnostic"]["direct_sample"] = {
            "sampled_foreground_pixels": 0,
            "foreground_overlap_ratio": 0.0,
        }
    else:
        raise AssertionError(f"unhandled failure mode {failure_mode}")
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "neutral delivery material",
                material_id,
                status="policy_fallback",
            )
        ],
    }
    report = _spatial_report(
        [
            {
                "part_id": "P0001",
                "resolved_support_counts": {"G06": 1},
                "observations": observations,
                "semantic_votes": [],
            }
        ]
    )

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=_fusion(),
        spatial_mapping_report=report,
    )

    assert "canonical_group_id" not in annotated["assignments"][0].get(
        "provenance", {}
    )
    assert audit["spatial_annotation"][
        "single_view_recovery_annotated_part_count"
    ] == 0
    rejections = audit["spatial_annotation"]["single_view_rejections"]
    assert len(rejections) == 1
    assert expected_reason_code in rejections[0]["reason_codes"]


def test_exact_tokens_annotate_assignment_and_face_subset_without_mutation() -> None:
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "MVInverse-calibrated green painted steel",
                "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted",
                parameters={"paint_roughness": 0.42},
                face_subsets=[
                    {
                        "subset_name": "Cover",
                        "semantic": "white translucent plastic",
                        "material_id": (
                            "mdl:vMaterials_2/Plastic/"
                            "Plastic_Thick_Translucent.mdl#plastic_white"
                        ),
                        "face_indices": [1, 2, 3],
                    }
                ],
            )
        ],
    }
    original = copy.deepcopy(plan)

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=_fusion(),
    )

    assignment = annotated["assignments"][0]
    subset = assignment["face_subsets"][0]
    assert assignment["provenance"]["canonical_group_id"] == "G06"
    assert assignment["provenance"]["face_subset_canonical_group_ids"] == {
        "Cover": "G04"
    }
    assert "provenance" not in subset
    assert assignment["parameters"] == original["assignments"][0]["parameters"]
    assert assignment["material_id"] == original["assignments"][0]["material_id"]
    assert (
        subset["material_id"]
        == original["assignments"][0]["face_subsets"][0]["material_id"]
    )
    assert audit["status"] == "COMPLETED"
    assert audit["summary"]["assignment_part_ids_by_group"] == {"G06": ["P0001"]}
    assert audit["summary"]["face_subsets_by_group"] == {"G04": ["P0001:Cover"]}
    assert all(
        record["reason_codes"] == ["EXACT_COLOR_AND_FAMILY_TOKEN_MATCH"]
        for record in audit["records"]
    )
    normalized = normalize_face_subsets(
        "P0001",
        assignment["face_subsets"],
        allowed_material_ids={subset["material_id"]},
        face_count=4,
    )
    assert normalized[0]["subset_name"] == "Cover"


def test_duplicate_canonical_appearance_is_ambiguous_and_not_authored() -> None:
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "green painted steel",
                "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted",
            )
        ],
    }

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=_fusion(duplicate_green_metal=True),
    )

    assert "provenance" not in annotated["assignments"][0]
    record = audit["records"][0]
    assert record["outcome"] == "AMBIGUOUS"
    assert record["selected_group_id"] is None
    assert record["selection_margin"] == pytest.approx(0.0)
    assert record["reason_codes"] == ["CANONICAL_GROUP_CANDIDATES_AMBIGUOUS"]
    assert audit["status"] == "COMPLETED_FAIL_CLOSED_AMBIGUITY"


def test_face_subset_group_map_is_preserved_without_touching_subset_schema() -> None:
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "green painted steel",
                "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted",
                provenance={
                    "face_subset_canonical_group_ids": {"Cover": "G04"},
                },
                face_subsets=[
                    {
                        "subset_name": "Cover",
                        "semantic": "black anodized metal",
                        "material_id": (
                            "mdl:Base/Metals/Aluminum_Anodized_Black.mdl"
                            "#Aluminum_Anodized_Black"
                        ),
                        "face_indices": [0],
                    }
                ],
            )
        ],
    }

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=_fusion(),
    )

    assignment = annotated["assignments"][0]
    subset = assignment["face_subsets"][0]
    assert "provenance" not in subset
    assert assignment["provenance"]["face_subset_canonical_group_ids"] == {
        "Cover": "G04"
    }
    subset_record = next(
        record for record in audit["records"] if record["entity_kind"] == "face_subset"
    )
    assert subset_record["outcome"] == "PRESERVED_EXISTING"
    assert (
        "TOKEN_INFERENCE_NOT_ALLOWED_TO_OVERRIDE_EXISTING_GROUP"
        in subset_record["reason_codes"]
    )


def test_legacy_generated_subset_provenance_is_migrated_to_assignment() -> None:
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "green painted steel",
                "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted",
                face_subsets=[
                    {
                        "subset_name": "Cover",
                        "semantic": "white translucent plastic",
                        "material_id": (
                            "mdl:vMaterials_2/Plastic/"
                            "Plastic_Thick_Translucent.mdl#plastic_white"
                        ),
                        "face_indices": [0],
                        "provenance": {
                            "canonical_group_id": "G04",
                            "canonical_group_annotation": {
                                "method": ("semantic_material_token_consensus/v1"),
                                "confidence": 0.99,
                            },
                        },
                    }
                ],
            )
        ],
    }

    annotated, _audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=_fusion(),
    )

    assignment = annotated["assignments"][0]
    assert "provenance" not in assignment["face_subsets"][0]
    assert assignment["provenance"]["face_subset_canonical_group_ids"] == {
        "Cover": "G04"
    }


def test_brown_to_orange_is_low_confidence_and_never_overrides_existing() -> None:
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "brown painted steel",
                "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted",
            ),
            _assignment(
                "P0002",
                "brown painted steel",
                "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted",
                provenance={"canonical_group_id": "G06", "source": "reviewed"},
            ),
        ],
    }

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=_fusion(),
    )

    inferred = annotated["assignments"][0]["provenance"]
    preserved = annotated["assignments"][1]["provenance"]
    assert inferred["canonical_group_id"] == "G07"
    assert (
        inferred["canonical_group_annotation"]["confidence_tier"]
        == "LOW_CONFIDENCE_NEIGHBOR"
    )
    assert (
        inferred["canonical_group_annotation"]["confidence"]
        <= MAXIMUM_NEIGHBOR_CONFIDENCE
    )
    assert preserved == {"canonical_group_id": "G06", "source": "reviewed"}
    by_part = {record["part_id"]: record for record in audit["records"]}
    assert by_part["P0001"]["reason_codes"] == [
        "LOW_CONFIDENCE_COLOR_NEIGHBOR_ACCEPTED"
    ]
    assert by_part["P0002"]["outcome"] == "PRESERVED_EXISTING"
    assert (
        "TOKEN_INFERENCE_NOT_ALLOWED_TO_OVERRIDE_EXISTING_GROUP"
        in by_part["P0002"]["reason_codes"]
    )


def test_family_mismatch_and_conflicting_colors_are_left_unannotated() -> None:
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "black matte rubber",
                "mdl:vMaterials_2/Other/Rubber/Caoutchouc.mdl#Rubber_Black_Matte",
            ),
            _assignment(
                "P0002",
                "green plastic cover",
                "mdl:vMaterials_2/Plastic/Polypropylene.mdl#Polypropylene_Blue",
            ),
        ],
    }

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=_fusion(),
    )

    assert all("provenance" not in item for item in annotated["assignments"])
    by_part = {record["part_id"]: record for record in audit["records"]}
    assert by_part["P0001"]["outcome"] == "UNRESOLVED"
    assert by_part["P0001"]["reason_codes"] == ["NO_CANDIDATE_CLEARED_CONFIDENCE_GATE"]
    assert by_part["P0002"]["outcome"] == "AMBIGUOUS"
    assert by_part["P0002"]["reason_codes"] == ["CONFLICTING_COLOR_TOKENS"]


def test_material_identity_tokens_can_recover_silver_metal_group() -> None:
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "standard industrial hardware",
                "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless",
            )
        ],
    }

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=_fusion(),
    )

    assert annotated["assignments"][0]["provenance"]["canonical_group_id"] == "G02"
    record = audit["records"][0]
    assert record["outcome"] == "ANNOTATED"
    assert record["annotation_confidence"] >= 0.82


def test_policy_fallback_tokens_cannot_invent_assignment_or_subset_group() -> None:
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "neutral delivery material for unresolved CAD component",
                "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless",
                status="policy_fallback",
                confidence=0.0,
                evidence_views=[],
                provenance={
                    "tier": "neutral_default",
                    "output_confidence_basis": (
                        "policy_fallback_is_not_model_confidence"
                    ),
                },
                face_subsets=[
                    {
                        "subset_name": "CADDisplaySubset",
                        "semantic": "white plastic",
                        "material_id": (
                            "mdl:vMaterials_2/Plastic/"
                            "Polypropylene.mdl#Polypropylene_White"
                        ),
                        "face_indices": [0],
                    }
                ],
            )
        ],
    }

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=_fusion(),
    )

    assignment = annotated["assignments"][0]
    assert "canonical_group_id" not in assignment["provenance"]
    assert "face_subset_canonical_group_ids" not in assignment["provenance"]
    assert all(
        record["outcome"] == "UNRESOLVED" for record in audit["records"]
    )
    assert all(
        record["reason_codes"]
        == ["POLICY_FALLBACK_MATERIAL_TOKENS_ARE_NOT_REFERENCE_EVIDENCE"]
        for record in audit["records"]
    )
    assert (
        audit["policy"]["policy_fallback_token_inference_allowed"] is False
    )


def test_policy_fallback_preserves_explicit_canonical_group() -> None:
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "neutral delivery material",
                "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless",
                status="policy_fallback",
                confidence=0.0,
                evidence_views=[],
                provenance={
                    "tier": "qa_repair_candidate",
                    "canonical_group_id": "G06",
                },
            )
        ],
    }

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=_fusion(),
    )

    assert (
        annotated["assignments"][0]["provenance"]["canonical_group_id"]
        == "G06"
    )
    assert audit["records"][0]["outcome"] == "PRESERVED_EXISTING"


@pytest.mark.parametrize(
    "top_level_group_id",
    [None, "G07"],
)
def test_source_corroborated_mdl_requires_consistent_top_level_group_lineage(
    top_level_group_id: str | None,
) -> None:
    provenance = {
        "tier": "corroborated_source_visual_nvidia_mdl",
        "source_visual_corroboration": {"canonical_group_id": "G06"},
    }
    if top_level_group_id is not None:
        provenance["canonical_group_id"] = top_level_group_id
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "green photo-corroborated accent",
                "mdl:vMaterials_2/Plastic/Plastic.mdl#Green",
                provenance=provenance,
            )
        ],
    }

    with pytest.raises(
        VisualGroupAnnotationError,
        match="canonical_group_id|group lineage",
    ):
        annotate_visual_groups(material_plan=plan, palette_fusion=_fusion())


def test_unresolved_canonical_family_is_a_low_confidence_exact_color_wildcard() -> None:
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "orange painted steel",
                "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted",
            ),
            _assignment(
                "P0002",
                "brown painted steel",
                "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted",
            ),
        ],
    }
    fusion = {
        "canonical_palette": {
            "groups": [
                {
                    "group_id": "G07",
                    "base_color": "orange",
                    "family_hint": "other",
                    "confidence": 0.60,
                }
            ]
        }
    }

    annotated, audit = annotate_visual_groups(
        material_plan=plan,
        palette_fusion=fusion,
    )

    first = annotated["assignments"][0]["provenance"]
    assert first["canonical_group_id"] == "G07"
    assert (
        first["canonical_group_annotation"]["confidence_tier"]
        == "LOW_CONFIDENCE_FAMILY_WILDCARD"
    )
    # Brown->orange plus an unresolved family would compound two assumptions.
    assert "provenance" not in annotated["assignments"][1]
    by_part = {record["part_id"]: record for record in audit["records"]}
    assert by_part["P0001"]["reason_codes"] == [
        "LOW_CONFIDENCE_CANONICAL_FAMILY_WILDCARD_ACCEPTED"
    ]
    assert by_part["P0002"]["outcome"] == "UNRESOLVED"


def test_unknown_existing_group_is_rejected_instead_of_overwritten() -> None:
    plan = {
        "schema_version": "1.0",
        "assignments": [
            _assignment(
                "P0001",
                "green painted steel",
                "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted",
                provenance={"canonical_group_id": "G999"},
            )
        ],
    }

    with pytest.raises(VisualGroupAnnotationError, match="unknown canonical group"):
        annotate_visual_groups(material_plan=plan, palette_fusion=_fusion())


def test_cli_writes_plan_and_audit_before_strict_ambiguity_exit(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    fusion_path = tmp_path / "palette_fusion.json"
    output_path = tmp_path / "annotated.json"
    audit_path = tmp_path / "audit.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "assignments": [
                    _assignment(
                        "P0001",
                        "green painted steel",
                        "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted",
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    fusion_path.write_text(
        json.dumps(_fusion(duplicate_green_metal=True)),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--plan",
            str(plan_path),
            "--palette-fusion",
            str(fusion_path),
            "--output-plan",
            str(output_path),
            "--audit",
            str(audit_path),
            "--require-unambiguous",
        ]
    )

    assert exit_code == EXIT_REQUIRE_UNAMBIGUOUS_FAILED
    assert output_path.is_file()
    assert audit_path.is_file()
    assert (
        json.loads(audit_path.read_text(encoding="utf-8"))["summary"]["ambiguous_count"]
        == 1
    )

    unambiguous_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    unambiguous_plan["assignments"][0]["semantic"] = "white plastic"
    unambiguous_plan["assignments"][0]["material_id"] = (
        "mdl:vMaterials_2/Plastic/Polypropylene.mdl#Polypropylene_White"
    )
    plan_path.write_text(json.dumps(unambiguous_plan), encoding="utf-8")
    assert (
        main(
            [
                "--plan",
                str(plan_path),
                "--palette-fusion",
                str(fusion_path),
                "--output-plan",
                str(output_path),
                "--audit",
                str(audit_path),
                "--require-unambiguous",
            ]
        )
        == EXIT_SUCCESS
    )
