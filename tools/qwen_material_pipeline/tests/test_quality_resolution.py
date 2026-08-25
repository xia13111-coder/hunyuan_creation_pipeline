from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from asset_pipeline.visual_materials.orchestrator import (
    _continuous_camera_view_specs,
    _render_view_arguments,
    _require_complete_live_camera_alignment,
    _validate_live_camera_registration_provenance,
    _validate_quality_render_contract,
    _validate_quality_resolution_bundle,
)
from qwen_material_pipeline.materials.quality_resolution import (
    FAIL_CLOSED,
    LIMITED_PASS,
    PASS,
    QualityResolutionError,
    build_quality_resolution,
    main,
)
from qwen_material_pipeline.usd.material_common import (
    POLICY_FALLBACK_CONFIDENCE_BASIS,
    SOURCE_VISUAL_PRESERVE_ACTION,
    SOURCE_VISUAL_PRESERVE_TIER,
    canonical_sha256,
    source_visual_binding_sha256,
)


BLUE_GROUP = "G_BLUE"
GREEN_GROUP = "G_GREEN"
BLUE_LOCAL = "L_BLUE"
TARGET_VIEW = "top_ref"
TARGET_RENDER = "top"
PART_IDS_SHA = "a" * 64
SOURCE_SIGNATURE_SHA = "b" * 64
GEOMETRY_SIGNATURE_SHA = "c" * 64
CANDIDATES = ["P0001", "P0002", "P0003", "P0004"]
OWNER = "P0005"


def _source_preserve_assignment(part_id: str, prim_path: str) -> dict:
    source_material = "/Looks/SourceBlue"
    return {
        "part_id": part_id,
        "material_id": "mdl:Neutral",
        "semantic": "corroborated source accent",
        "confidence": 0.0,
        "evidence_views": [],
        "status": "policy_fallback",
        "provenance": {
            "tier": SOURCE_VISUAL_PRESERVE_TIER,
            "reason_codes": [
                "SOURCE_VISUAL_MATERIAL_PRESENT",
                "SOURCE_VISUAL_BINDING_HASH_BOUND",
                "PRESERVE_SOURCE_VISUAL_NOOP",
                "REFERENCE_PALETTE_MULTIVIEW_COLOR_CORROBORATION",
                "RARE_SOURCE_VISUAL_SIGNATURE",
                "REPEATED_GEOMETRY_SOURCE_LOCATOR",
            ],
            "output_confidence_basis": POLICY_FALLBACK_CONFIDENCE_BASIS,
            "sources": [],
            "source_visual_corroboration": {
                "canonical_group_id": BLUE_GROUP,
                "canonical_color_family": "blue",
                "canonical_source_view_ids": ["iso_ref", TARGET_VIEW],
                "source_visual_signature_sha256": SOURCE_SIGNATURE_SHA,
                "source_signature_count": 4,
                "registry_fraction": 0.04,
                "geometry_signature_sha256": GEOMETRY_SIGNATURE_SHA,
                "geometry_repeat_count": 4,
            },
        },
        "apply_action": SOURCE_VISUAL_PRESERVE_ACTION,
        "source_visual_material_prim_path": source_material,
        "source_visual_material_binding_sha256": (
            source_visual_binding_sha256(
                part_id=part_id,
                prim_path=prim_path,
                material_prim_path=source_material,
            )
        ),
    }


def _pass_view(view_id: str, render_view_id: str) -> dict:
    return {
        "reference_view_id": view_id,
        "render_view_id": render_view_id,
        "status": PASS,
        "reasons": [],
        "mapping": {
            "mode": "explicit",
            "selected_render_view_id": render_view_id,
            "reasons": [],
        },
        "reference": {
            "image_sha256": ("1" if view_id == "front_ref" else "2") * 64,
            "trusted_evidence": {"usable": True, "reasons": []},
        },
        "material_color": {
            "trusted_evidence_group_recall": {"groups": []},
            "trusted_evidence_dominant_mass": {
                "status": PASS,
                "families": [],
            },
            "unreferenced_render_chromatic_mass": {"status": PASS},
        },
    }


def _owner_observation(*, include_overlap: bool = True) -> dict:
    observation = {
        "reference_view_id": TARGET_VIEW,
        "render_view_id": TARGET_RENDER,
        "declared_visible_pixels": 10000,
        "decoded_part_pixels": 10000,
        "projected_part_pixels": 9000,
        "classification": "resolved",
        "reason_code": "spatial_color_projection_resolved",
        "local_group_id": "L_GREEN",
        "canonical_group_id": GREEN_GROUP,
        "group_scores": [
            {
                "local_group_id": "L_GREEN",
                "canonical_group_id": GREEN_GROUP,
                "base_color": "green",
                "matching_pixels": 7200,
                "color_share": 0.8,
            },
            {
                "local_group_id": BLUE_LOCAL,
                "canonical_group_id": BLUE_GROUP,
                "base_color": "blue",
                "matching_pixels": 200,
                "color_share": 0.02,
            },
        ],
        "color_margin": 0.78,
        "bbox_group_scores": [
            {
                "local_group_id": "L_GREEN",
                "canonical_group_id": GREEN_GROUP,
                "base_color": "green",
                "matching_pixels": 7000,
                "color_share": 0.78,
            },
            {
                "local_group_id": BLUE_LOCAL,
                "canonical_group_id": BLUE_GROUP,
                "base_color": "blue",
                "matching_pixels": 180,
                "color_share": 0.02,
            },
        ],
        "bbox_color_margin": 0.75,
        "bbox_canonical_group_id": GREEN_GROUP,
        "registration_label_stable": True,
        "perturbation_label_stable": True,
        "projection_perturbations": [
            {
                "offset_pixels": list(offset),
                "sampled_reference_pixels": 9000,
                "canonical_group_id": GREEN_GROUP,
                "diagnostic_canonical_group_id": GREEN_GROUP,
                "best_color_share": 0.76,
                "color_margin": 0.72,
            }
            for offset in ((-2, 0), (2, 0), (0, -2), (0, 2))
        ],
        "small_part_diagnostic": {
            "status": "rejected",
            "canonical_group_id": GREEN_GROUP,
        },
    }
    if include_overlap:
        evidence_payload = {
            "view_id": TARGET_VIEW,
            "local_group_id": BLUE_LOCAL,
            "canonical_group_id": BLUE_GROUP,
            "base_color": "blue",
            "accepted_boxes": [
                {
                    "box_index": 0,
                    "box": [100, 100, 200, 150],
                    "matching_pixel_count": 160,
                }
            ],
            "evidence_pixel_count": 160,
        }
        observation["accepted_evidence_box_overlaps"] = [
            {
                "local_group_id": BLUE_LOCAL,
                "canonical_group_id": BLUE_GROUP,
                "base_color": "blue",
                "evidence_pixel_count": 160,
                "projected_overlap_pixels": 100,
                "projected_overlap_share": 0.625,
                "evidence_audit_sha256": canonical_sha256(evidence_payload),
            }
        ]
    return observation


def _documents() -> dict[str, dict]:
    registry_parts = [
        {
            "part_id": part_id,
            "prim_path": f"/Asset/{part_id}/Mesh",
            "point_count": 100,
            "face_count": 96,
            "world_bbox": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            "existing_visual_material": "/Looks/SourceBlue",
        }
        for part_id in CANDIDATES
    ]
    registry_parts.append(
        {
            "part_id": OWNER,
            "prim_path": f"/Asset/{OWNER}/Mesh",
            "point_count": 1000,
            "face_count": 996,
            "world_bbox": [[0.0, 0.0, 0.0], [10.0, 10.0, 1.0]],
            "existing_visual_material": "/Looks/Green",
        }
    )
    rendered_registry = {
        "schema_version": "qwen-material-parts/v1",
        "asset_usd": "/tmp/asset.usd",
        "asset_sha256": "f" * 64,
        "part_count": 5,
        "instance_root_count": 0,
        "parts": registry_parts,
        "render_set": {
            "views": [
                {
                    "view_id": TARGET_RENDER,
                    "part_ids_sha256": PART_IDS_SHA,
                    "visible_parts": [{"part_id": OWNER, "pixels": 10000}],
                },
                {
                    "view_id": "front",
                    "part_ids_sha256": "d" * 64,
                    "visible_parts": [
                        *[{"part_id": part_id, "pixels": 10} for part_id in CANDIDATES],
                        {"part_id": OWNER, "pixels": 9000},
                    ],
                },
                {
                    "view_id": "rear",
                    "part_ids_sha256": "e" * 64,
                    "visible_parts": [
                        *[{"part_id": part_id, "pixels": 12} for part_id in CANDIDATES],
                        {"part_id": OWNER, "pixels": 8000},
                    ],
                },
            ]
        },
    }
    final_plan = {
        "schema_version": "1.0",
        "assignments": [
            *[
                _source_preserve_assignment(
                    part_id,
                    f"/Asset/{part_id}/Mesh",
                )
                for part_id in CANDIDATES
            ],
            {
                "part_id": OWNER,
                "material_id": "mdl:Green",
                "semantic": "green body",
                "confidence": 0.0,
                "evidence_views": [],
                "status": "policy_fallback",
                "provenance": {
                    "tier": "qa_repair_candidate",
                    "reason_codes": ["QA_TRUSTED_PART_GROUP_LOCALIZATION"],
                    "output_confidence_basis": (POLICY_FALLBACK_CONFIDENCE_BASIS),
                    "sources": [],
                    "canonical_group_id": GREEN_GROUP,
                },
            },
        ],
        "provenance": {"mode": "test"},
    }
    policy_audit = {
        "schema_version": "qwen-policy-exact-cover-report/v1",
        "corroborated_source_visual": {
            "state": "corroborated_source_accents_found",
            "applied_part_ids": list(CANDIDATES),
            "eligible_part_ids": list(CANDIDATES),
            "groups": [
                {
                    "group_id": BLUE_GROUP,
                    "canonical_color_family": "blue",
                    "canonical_source_view_ids": ["iso_ref", TARGET_VIEW],
                    "source_visual_signature": {"shader_id": "UsdPreviewSurface"},
                    "source_visual_signature_sha256": SOURCE_SIGNATURE_SHA,
                    "source_signature_count": 4,
                    "registry_fraction": 0.04,
                    "geometry_cohorts": [
                        {
                            "point_count": 100,
                            "face_count": 96,
                            "sorted_bbox_extents": [1.0, 1.0, 1.0],
                            "geometry_signature_sha256": (GEOMETRY_SIGNATURE_SHA),
                            "repeat_count": 4,
                            "part_ids": list(CANDIDATES),
                        }
                    ],
                    "eligible_part_ids": list(CANDIDATES),
                }
            ],
        },
    }
    palette_fusion = {
        "schema_version": "qwen-multiview-palette-fusion/v1",
        "canonical_palette": {
            "schema_version": "qwen-canonical-material-palette/v1",
            "groups": [
                {
                    "group_id": BLUE_GROUP,
                    "base_color": "blue",
                    "family_hint": "plastic",
                    "source_view_ids": ["iso_ref", TARGET_VIEW],
                    "distinct_view_count": 2,
                    "singleton": False,
                },
                {
                    "group_id": GREEN_GROUP,
                    "base_color": "green",
                    "family_hint": "painted metal",
                    "source_view_ids": ["front_ref", "side_ref"],
                    "distinct_view_count": 2,
                    "singleton": False,
                },
            ],
        },
        "view_group_id_maps": {
            TARGET_VIEW: {
                BLUE_LOCAL: BLUE_GROUP,
                "L_GREEN": GREEN_GROUP,
            }
        },
    }
    quality_report = {
        "schema_version": "qwen-reference-render-comparison/v1",
        "inputs": {
            "reference_manifest_sha256": "9" * 64,
            "mapping_mode": "explicit",
            "selected_view_mapping": {
                "front_ref": "front",
                "side_ref": "rear",
                TARGET_VIEW: TARGET_RENDER,
            },
        },
        "thresholds": {
            "minimum_evidence_group_recall": 0.5,
            "pass_color_score": 0.62,
        },
        "aggregate": {
            "status": "REVIEW",
            "material_color_score": 0.8,
            "comparable_view_count": 3,
            "passed_view_count": 2,
            "review_view_count": 0,
            "failed_view_count": 1,
        },
        "views": [
            _pass_view("front_ref", "front"),
            _pass_view("side_ref", "rear"),
            {
                "reference_view_id": TARGET_VIEW,
                "render_view_id": TARGET_RENDER,
                "status": "FAIL",
                "reasons": ["trusted_palette_group_missing_from_render"],
                "mapping": {
                    "mode": "explicit",
                    "selected_render_view_id": TARGET_RENDER,
                    "reasons": [],
                },
                "reference": {
                    "image_sha256": "3" * 64,
                    "trusted_evidence": {"usable": True, "reasons": []},
                },
                "material_color": {
                    "trusted_evidence_group_recall": {
                        "groups": [
                            {
                                "group_id": BLUE_LOCAL,
                                "base_colors": ["blue"],
                                "reference_evidence_weight": 160,
                                "reference_group_share": 0.02,
                                "required_render_share": 0.01,
                                "observed_render_share": 0.001,
                                "recall": 0.1,
                            }
                        ]
                    },
                    "trusted_evidence_dominant_mass": {
                        "status": PASS,
                        "families": [],
                    },
                    "unreferenced_render_chromatic_mass": {"status": PASS},
                },
            },
        ],
    }
    spatial_parts = [
        {
            "part_id": part_id,
            "observations": [
                {
                    "reference_view_id": TARGET_VIEW,
                    "render_view_id": TARGET_RENDER,
                    "declared_visible_pixels": 0,
                    "classification": "insufficient_visibility",
                    "reason_code": "part_visible_pixels_below_diagnostic_floor",
                }
            ],
        }
        for part_id in CANDIDATES
    ]
    spatial_parts.append(
        {
            "part_id": OWNER,
            "observations": [_owner_observation()],
        }
    )
    spatial_report = {
        "schema_version": "qwen-spatial-mapping-audit/v1",
        "policy": {
            "minimum_diagnostic_visible_pixels": 128,
            "minimum_semantic_confidence": 0.85,
        },
        "inputs": {
            "files": [
                {
                    "label": f"part_ids:{TARGET_RENDER}",
                    "path": "/tmp/top.png",
                    "sha256": PART_IDS_SHA,
                }
            ]
        },
        "reference_evidence": [
            {
                "view_id": TARGET_VIEW,
                "raw_sha256": "3" * 64,
                "normalized_pixel_sha256": "4" * 64,
                "content_cluster_id": "PH01",
                "selected_render_view_id": TARGET_RENDER,
                "pose_cluster_id": TARGET_RENDER,
                "alignment_trusted": True,
                "alignment_score": 0.8,
                "accepted_palette_evidence": [
                    {
                        "view_id": TARGET_VIEW,
                        "local_group_id": BLUE_LOCAL,
                        "canonical_group_id": BLUE_GROUP,
                        "base_color": "blue",
                        "accepted_boxes": [
                            {
                                "box_index": 0,
                                "box": [100, 100, 200, 150],
                                "matching_pixel_count": 160,
                            }
                        ],
                        "evidence_pixel_count": 160,
                        "evidence_audit_sha256": canonical_sha256(
                            {
                                "view_id": TARGET_VIEW,
                                "local_group_id": BLUE_LOCAL,
                                "canonical_group_id": BLUE_GROUP,
                                "base_color": "blue",
                                "accepted_boxes": [
                                    {
                                        "box_index": 0,
                                        "box": [100, 100, 200, 150],
                                        "matching_pixel_count": 160,
                                    }
                                ],
                                "evidence_pixel_count": 160,
                            }
                        ),
                    }
                ],
            }
        ],
        "view_alignments": [
            {
                "reference_view_id": TARGET_VIEW,
                "selected_render_view_id": TARGET_RENDER,
                "score": 0.8,
                "projection_iou": 0.85,
                "ecc_status": "success",
                "ecc_correlation": 0.9,
                "ecc_transform_audit": {
                    "constraints_passed": True,
                    "constraint_failures": [],
                },
                "trusted": True,
                "reason_codes": [],
            }
        ],
        "parts": spatial_parts,
    }
    geometry_risk = {
        "schema_version": "qwen-geometry-uniform-material-risk/v1",
        "parts": [
            {
                "part_id": part_id,
                "risk": {
                    "multi_material_risk": False,
                    "basis": "test",
                },
            }
            for part_id in [*CANDIDATES, OWNER]
        ],
    }
    return {
        "final_plan": final_plan,
        "policy_audit": policy_audit,
        "quality_report": quality_report,
        "palette_fusion": palette_fusion,
        "spatial_report": spatial_report,
        "geometry_risk": geometry_risk,
        "rendered_registry": rendered_registry,
    }


def test_strict_not_observable_geometry_pose_is_accepted() -> None:
    documents = _documents()

    report = build_quality_resolution(**documents)

    assert report["resolution_status"] == LIMITED_PASS
    assert report["material_stage_accepted"] is True
    assert report["reason_codes"] == []
    assert report["summary"]["accepted_limitation_count"] == 1
    limitation = report["limitations"][0]
    assert limitation["classification"] == "NOT_OBSERVABLE_GEOMETRY_POSE"
    assert limitation["reason_code"] == "POSE_OR_OCCLUSION_MISMATCH"
    assert limitation["candidate_geometry"]["safe_part_ids"] == CANDIDATES
    assert limitation["foreign_owner"]["part_id"] == OWNER
    assert (
        limitation["foreign_owner"]["accepted_box_overlap"]["projected_overlap_share"]
        == 0.625
    )
    assert limitation["evidence_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in limitation.items()
            if key not in {"eligible", "reason_codes", "evidence_sha256"}
        }
    )


def test_not_observable_cohort_can_use_confirmed_nvidia_mdl_representation() -> None:
    documents = _documents()
    for assignment in documents["final_plan"]["assignments"]:
        if assignment["part_id"] not in CANDIDATES:
            continue
        assignment.pop("apply_action")
        assignment.pop("source_visual_material_prim_path")
        assignment.pop("source_visual_material_binding_sha256")
        assignment["material_id"] = "mdl:BluePolycarbonate"
        assignment["provenance"] = {
            "tier": "corroborated_source_visual_nvidia_mdl",
            "reason_codes": ["QWEN_CONFIRMED_NVIDIA_MDL_SELECTION"],
            "output_confidence_basis": POLICY_FALLBACK_CONFIDENCE_BASIS,
            "sources": [],
            "source_visual_corroboration": {
                "canonical_group_id": BLUE_GROUP,
                "confirmed_material_id": "mdl:BluePolycarbonate",
            },
        }

    report = build_quality_resolution(**documents)

    assert report["resolution_status"] == LIMITED_PASS
    assert report["material_stage_accepted"] is True
    assert report["limitations"][0]["candidate_geometry"]["safe_part_ids"] == (
        CANDIDATES
    )
    assert (
        _validate_quality_resolution_bundle(
            resolution=report,
            **documents,
        )
        == LIMITED_PASS
    )


def _add_coverage_preview(documents: dict[str, dict]) -> None:
    target = documents["quality_report"]["views"][2]
    target["mapping"]["alignment_preview"] = {
        "score": 0.7,
        "silhouette_iou": 0.6,
    }


def test_visible_source_cohort_is_bounded_as_geometry_coverage() -> None:
    documents = _documents()
    _add_coverage_preview(documents)
    target_view = documents["rendered_registry"]["render_set"]["views"][0]
    target_view["visible_parts"].extend(
        {"part_id": part_id, "pixels": 10} for part_id in CANDIDATES
    )
    target_group = documents["quality_report"]["views"][2]["material_color"][
        "trusted_evidence_group_recall"
    ]["groups"][0]
    target_group.update(
        {
            "observed_render_share": 0.004,
            "recall": 0.4,
            "delivery_presence_status": "MISSING",
        }
    )

    report = build_quality_resolution(**documents)

    assert report["resolution_status"] == LIMITED_PASS
    limitation = report["limitations"][0]
    assert limitation["classification"] == ("OBSERVABLE_GEOMETRY_COVERAGE_MISMATCH")
    assert limitation["limitation_lane"] == ("source_bound_visible_repeated_geometry")
    assert limitation["candidate_geometry"]["visible_part_ids"] == CANDIDATES
    assert (
        _validate_quality_resolution_bundle(
            resolution=report,
            **documents,
        )
        == LIMITED_PASS
    )


def test_cross_view_delivered_repair_is_bounded_as_camera_coverage() -> None:
    documents = _documents()
    _add_coverage_preview(documents)
    documents["policy_audit"]["corroborated_source_visual"]["groups"] = []
    for view_id, quality_view in (
        ("front_ref", documents["quality_report"]["views"][0]),
        ("side_ref", documents["quality_report"]["views"][1]),
    ):
        documents["palette_fusion"]["view_group_id_maps"][view_id] = {
            BLUE_LOCAL: BLUE_GROUP
        }
        quality_view["material_color"]["trusted_evidence_group_recall"]["groups"] = [
            {
                "group_id": BLUE_LOCAL,
                "base_colors": ["blue"],
                "reference_evidence_weight": 180,
                "reference_group_share": 0.02,
                "required_render_share": 0.01,
                "observed_render_share": 0.012,
                "recall": 1.0,
                "delivery_presence_status": "PRESENT",
            }
        ]
    target_group = documents["quality_report"]["views"][2]["material_color"][
        "trusted_evidence_group_recall"
    ]["groups"][0]
    target_group.update(
        {
            "observed_render_share": 0.002,
            "recall": 0.2,
            "delivery_presence_status": "MISSING",
        }
    )
    owner_assignment = documents["final_plan"]["assignments"][-1]
    owner_assignment["material_id"] = "mdl:Blue"
    owner_assignment["provenance"] = {
        "tier": "qa_repair_candidate",
        "canonical_group_id": BLUE_GROUP,
        "reason_codes": [
            "QA_MISSING_CANONICAL_GROUP_MULTI_VIEW",
            "QA_TRUSTED_PART_GROUP_LOCALIZATION",
            "QA_HIGH_CONFIDENCE_WHITELIST_MATERIAL_CANDIDATE",
            "QA_POST_RENDER_VALIDATION_REQUIRED",
        ],
        "output_confidence_basis": POLICY_FALLBACK_CONFIDENCE_BASIS,
        "sources": [],
        "supporting_view_ids": ["front_ref", "side_ref"],
        "material_selection_basis": (
            "high_confidence_whitelist_candidate_pending_render_qa"
        ),
    }

    report = build_quality_resolution(**documents)

    assert report["resolution_status"] == LIMITED_PASS
    limitation = report["limitations"][0]
    assert limitation["limitation_lane"] == "cross_view_material_delivery"
    assert limitation["cross_view_delivery"]["delivered_view_count"] == 2
    assert limitation["qa_repair_assignments"][0]["part_id"] == OWNER
    assert (
        _validate_quality_resolution_bundle(
            resolution=report,
            **documents,
        )
        == LIMITED_PASS
    )


def test_orchestrator_accepts_only_the_hash_bound_limited_resolution() -> None:
    documents = _documents()
    report = build_quality_resolution(**documents)

    assert (
        _validate_quality_resolution_bundle(
            resolution=report,
            **documents,
        )
        == LIMITED_PASS
    )

    for field, mutate in (
        (
            "input hash",
            lambda value: value["input_hashes"].update(
                {"quality_report_sha256": "0" * 64}
            ),
        ),
        (
            "threshold",
            lambda value: value["thresholds"].update(
                {"minimum_accepted_box_owner_overlap": 0.49}
            ),
        ),
        (
            "summary",
            lambda value: value["summary"].update({"accepted_limitation_count": 0}),
        ),
        (
            "limitation evidence",
            lambda value: value["limitations"][0].update({"evidence_sha256": "0" * 64}),
        ),
    ):
        tampered = copy.deepcopy(report)
        mutate(tampered)
        with pytest.raises(RuntimeError, match="visual-quality|Visual-quality"):
            _validate_quality_resolution_bundle(
                resolution=tampered,
                **documents,
            )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_contract_documents(
    tmp_path: Path,
) -> tuple[dict, Path, dict, dict, dict[str, Path]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    views: list[dict] = []
    quality_views: list[dict] = []
    spatial_files: list[dict] = []
    paths: dict[str, Path] = {}
    for view_id in ("front", "top"):
        rgb = tmp_path / f"{view_id}.rgb.png"
        part_ids = tmp_path / f"{view_id}.part_ids.png"
        rgb.write_bytes(f"rgb:{view_id}".encode())
        part_ids.write_bytes(f"part_ids:{view_id}".encode())
        paths[f"{view_id}:rgb"] = rgb
        paths[f"{view_id}:part_ids"] = part_ids
        views.append(
            {
                "view_id": view_id,
                "rgb": str(rgb),
                "part_ids": str(part_ids),
            }
        )
        quality_views.append(
            {
                "reference_view_id": f"ref_{view_id}",
                "render_view_id": view_id,
                "status": "PASS",
                "render": {
                    "image": str(rgb),
                    "image_sha256": _sha256(rgb),
                    "part_ids": str(part_ids),
                    "part_ids_sha256": _sha256(part_ids),
                },
            }
        )
        spatial_files.append(
            {
                "label": f"part_ids:{view_id}",
                "path": str(part_ids),
                "sha256": _sha256(part_ids),
            }
        )
    registry = {
        "schema_version": "qwen-material-parts/v1",
        "render_set": {"views": views},
    }
    registry_path = tmp_path / "part_registry.rendered.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    quality = {
        "inputs": {
            "rendered_registry": str(registry_path),
            "rendered_registry_sha256": _sha256(registry_path),
        },
        "views": quality_views,
    }
    spatial = {"inputs": {"files": spatial_files}}
    return registry, registry_path, quality, spatial, paths


def test_final_part_id_contract_rejects_stale_pose_and_rgb(
    tmp_path: Path,
) -> None:
    registry, registry_path, quality, spatial, paths = _render_contract_documents(
        tmp_path
    )
    _validate_quality_render_contract(
        quality_report=quality,
        rendered_registry=registry,
        rendered_registry_path=registry_path,
        spatial_report=spatial,
    )

    paths["top:part_ids"].write_bytes(b"wrong-pose")
    with pytest.raises(RuntimeError, match="changed geometry"):
        _validate_quality_render_contract(
            quality_report=quality,
            rendered_registry=registry,
            rendered_registry_path=registry_path,
            spatial_report=spatial,
        )

    registry, registry_path, quality, spatial, paths = _render_contract_documents(
        tmp_path / "rgb-tamper"
    )
    paths["front:rgb"].write_bytes(b"stale-rgb")
    with pytest.raises(RuntimeError, match="not hash-bound"):
        _validate_quality_render_contract(
            quality_report=quality,
            rendered_registry=registry,
            rendered_registry_path=registry_path,
            spatial_report=spatial,
        )


def test_continuous_camera_specs_are_reused_for_quality_render(
    tmp_path: Path,
) -> None:
    registry = {
        "render_set": {
            "requested_view_tokens": ["front", "side"],
            "views": [
                {
                    "view_id": "front",
                    "analysis_direction": [1.0, 0.0, 0.0],
                    "analysis_camera_up_axis": [0.0, 0.0, 1.0],
                    "focal_length_mm": 45.0,
                    "camera_distance_multiplier": 2.65,
                    "camera_target_offset_u": 0.04,
                    "camera_target_offset_v": -0.08,
                    "camera_projection_mode": "orthographic",
                    "camera_orthographic_span_multiplier": 2.4,
                    "camera_calibration": {
                        "phase": "final",
                        "target_offset_u": 0.04,
                        "target_offset_v": -0.08,
                        "projection_mode": "orthographic",
                        "orthographic_span_multiplier": 2.4,
                    },
                },
                {
                    "view_id": "side",
                    "analysis_direction": [0.0, -1.0, 0.1],
                    "analysis_camera_up_axis": [0.0, 0.1, 1.0],
                    "focal_length_mm": 60.0,
                    "camera_distance_multiplier": 2.65,
                    "camera_calibration": {"phase": "final"},
                },
            ],
        }
    }
    output = tmp_path / "camera_view_specs.json"

    arguments = _render_view_arguments(
        baseline_registry=registry,
        view_specs_output=output,
        fallback_views="pose-bank-26",
    )

    assert arguments == ["--view-specs", str(output)]
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document == _continuous_camera_view_specs(registry)
    assert [item["view_id"] for item in document["views"]] == ["front", "side"]
    assert document["views"][1]["focal_length_mm"] == 60.0
    assert document["views"][0]["target_offset_u"] == 0.04
    assert document["views"][0]["target_offset_v"] == -0.08
    assert document["views"][0]["projection_mode"] == "orthographic"
    assert document["views"][0]["orthographic_span_multiplier"] == 2.4


def test_quality_render_uses_pose_bank_without_continuous_calibration(
    tmp_path: Path,
) -> None:
    registry = {
        "render_set": {
            "requested_view_tokens": ["front"],
            "views": [{"view_id": "front"}],
        }
    }
    output = tmp_path / "camera_view_specs.json"

    arguments = _render_view_arguments(
        baseline_registry=registry,
        view_specs_output=output,
        fallback_views="pose-bank-26",
    )

    assert arguments == ["--views", "pose-bank-26"]
    assert not output.exists()


def test_live_camera_provenance_accepts_only_current_run_two_pass_seed(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "renders" / "part_registry.rendered.json"
    registry.parent.mkdir()
    registry.write_text("{}", encoding="utf-8")
    specs = tmp_path / "camera" / "search_pass" / "final_view_specs.json"
    specs.parent.mkdir(parents=True)
    specs.write_text("{}", encoding="utf-8")
    reference_manifest = tmp_path / "annotations.json"
    reference_manifest.write_text("{}", encoding="utf-8")
    input_contract = {
        "camera_objective_version": "hierarchical_visible_part_alignment/v9",
        "camera_selection_policy_version": (
            "alignment_gate_then_canonical_camera_signature_with_view_fallback/v2"
        ),
        "initial_view_specs_sha256": canonical_sha256({}),
    }
    solution_contract = {"schema_version": "qwen-camera-solution/v1", "views": []}
    report = tmp_path / "camera" / "camera_calibration_report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "qwen-whole-asset-camera-calibration/v10",
                "source_registry": str(registry.resolve()),
                "source_registry_sha256": hashlib.sha256(b"{}").hexdigest(),
                "reference_manifest": str(reference_manifest.resolve()),
                "reference_manifest_sha256": hashlib.sha256(b"{}").hexdigest(),
                "source_spatial_mapping": None,
                "source_initial_view_specs": str(specs.resolve()),
                "source_initial_view_specs_sha256": hashlib.sha256(b"{}").hexdigest(),
                "seed_search": {"mode": "existing_continuous_camera_specs"},
                "camera_objective_version": "hierarchical_visible_part_alignment/v9",
                "camera_selection_policy_version": (
                    "alignment_gate_then_canonical_camera_signature_with_view_fallback/v2"
                ),
                "calibration_input_contract": input_contract,
                "calibration_input_fingerprint": canonical_sha256(input_contract),
                "camera_solution_contract": solution_contract,
                "camera_solution_fingerprint": canonical_sha256(solution_contract),
                "final_view_specs": str(specs.resolve()),
                "final_view_specs_sha256": hashlib.sha256(b"{}").hexdigest(),
                "whole_asset_only": True,
                "per_part_geometric_warp_applied": False,
                "camera_intrinsics_optimized": [
                    "projection_mode",
                    "focal_length_mm",
                ],
                "camera_extrinsics_optimized": [
                    "orbit_azimuth",
                    "orbit_elevation",
                    "optical_axis_target_u",
                    "optical_axis_target_v",
                ],
            }
        ),
        encoding="utf-8",
    )

    document = _validate_live_camera_registration_provenance(
        report,
        source_registry=registry,
        reference_manifest=reference_manifest,
        initial_view_specs=specs,
    )

    assert document["whole_asset_only"] is True

    reference_manifest.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="reference evidence differs"):
        _validate_live_camera_registration_provenance(
            report,
            source_registry=registry,
            reference_manifest=reference_manifest,
            initial_view_specs=specs,
        )


def test_live_camera_provenance_rejects_external_or_legacy_seed(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "part_registry.rendered.json"
    registry.write_text("{}", encoding="utf-8")
    reference_manifest = tmp_path / "annotations.json"
    reference_manifest.write_text("{}", encoding="utf-8")
    report = tmp_path / "camera_calibration_report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "qwen-whole-asset-camera-calibration/v7",
                "source_registry": str(registry.resolve()),
                "source_spatial_mapping": "/old/run/spatial_mapping.json",
                "source_initial_view_specs": "/old/run/final_view_specs.json",
                "seed_search": {"mode": "existing_spatial_mapping"},
                "whole_asset_only": True,
                "per_part_geometric_warp_applied": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="current from-zero"):
        _validate_live_camera_registration_provenance(
            report,
            source_registry=registry,
            reference_manifest=reference_manifest,
            initial_view_specs=None,
        )


def test_live_camera_alignment_gate_accepts_bounded_box_first_residual() -> None:
    report = {
        "views": [
            {
                "reference_view_id": "front",
                "final": {"projection_iou": 0.98, "boundary_p95_px": 2.0},
                "complete_alignment_target": {
                    "minimum_iou": 0.97,
                    "maximum_boundary_p95_px": 3.0,
                },
                "complete_alignment_passed": True,
            },
            {
                "reference_view_id": "side",
                "final": {"projection_iou": 0.93, "boundary_p95_px": 5.0},
                "complete_alignment_target": {
                    "minimum_iou": 0.97,
                    "maximum_boundary_p95_px": 3.0,
                },
                "complete_alignment_passed": False,
            },
        ]
    }

    acceptance = _require_complete_live_camera_alignment(
        report,
        expected_reference_ids={"front", "side"},
    )

    assert acceptance["views"]["front"]["tier"] == "strict"
    assert acceptance["views"]["front"]["evidence_weight"] == 1.0
    assert acceptance["views"]["side"]["tier"] == ("usable_box_correspondence")
    assert acceptance["views"]["side"]["evidence_weight"] == 0.8


def test_live_camera_alignment_gate_rejects_gross_pose_error() -> None:
    report = {
        "views": [
            {
                "reference_view_id": "side",
                "final": {
                    "projection_iou": 0.87,
                    "boundary_p95_px": 16.0,
                },
                "complete_alignment_target": {
                    "minimum_iou": 0.97,
                    "maximum_boundary_p95_px": 3.0,
                },
                "complete_alignment_passed": False,
            }
        ]
    }

    with pytest.raises(RuntimeError, match=r"side\(IoU=0.8700"):
        _require_complete_live_camera_alignment(
            report,
            expected_reference_ids={"side"},
        )


def test_live_camera_alignment_keeps_residual_views_for_local_part_boxes() -> None:
    def view(
        view_id: str,
        *,
        iou: float,
        boundary: float,
        recall: float,
        structure: float,
    ) -> dict[str, object]:
        return {
            "reference_view_id": view_id,
            "final": {
                "projection_iou": iou,
                "boundary_p95_px": boundary,
                "target_recall": recall,
                "structure_score": structure,
            },
            "complete_alignment_target": {
                "minimum_iou": 0.97,
                "maximum_boundary_p95_px": 3.0,
            },
            "complete_alignment_passed": False,
        }

    report = {
        "views": [
            view(
                "front",
                iou=0.94,
                boundary=7.0,
                recall=0.97,
                structure=0.88,
            ),
            view(
                "iso",
                iou=0.91,
                boundary=12.0,
                recall=0.95,
                structure=0.83,
            ),
            view(
                "top",
                iou=0.84,
                boundary=23.0,
                recall=0.96,
                structure=0.90,
            ),
        ]
    }

    acceptance = _require_complete_live_camera_alignment(
        report,
        expected_reference_ids={"front", "iso", "top"},
    )

    assert acceptance["policy"] == "two_layer_box_first_part_id_alignment/v2"
    assert acceptance["anchor_view_ids"] == ["front", "iso"]
    assert acceptance["views"]["top"]["tier"] == "local_box_refinement_only"
    assert acceptance["views"]["top"]["evidence_weight"] == 0.35
    assert acceptance["views"]["top"]["observation_eligible"] is True


def test_raw_pass_remains_pass_and_hash_binds_every_input() -> None:
    documents = _documents()
    documents["quality_report"]["aggregate"]["status"] = PASS

    report = build_quality_resolution(**documents)

    assert report["resolution_status"] == PASS
    assert report["limitations"] == []
    assert set(report["input_hashes"]) == {
        "final_plan_sha256",
        "policy_audit_sha256",
        "quality_report_sha256",
        "palette_fusion_sha256",
        "spatial_report_sha256",
        "geometry_risk_sha256",
        "rendered_registry_sha256",
    }
    for key, document_key in (
        ("final_plan_sha256", "final_plan"),
        ("policy_audit_sha256", "policy_audit"),
        ("quality_report_sha256", "quality_report"),
        ("palette_fusion_sha256", "palette_fusion"),
        ("spatial_report_sha256", "spatial_report"),
        ("geometry_risk_sha256", "geometry_risk"),
        ("rendered_registry_sha256", "rendered_registry"),
    ):
        assert report["input_hashes"][key] == canonical_sha256(documents[document_key])


def test_matching_pixels_never_substitute_for_optional_owner_diagnostic() -> None:
    documents = _documents()
    owner = documents["spatial_report"]["parts"][-1]["observations"][0]
    owner.pop("accepted_evidence_box_overlaps")
    # This deliberately exceeds reference_evidence_weight.  The resolver must
    # not divide 200 by 160 and pretend it proves spatial overlap.
    assert owner["group_scores"][1]["matching_pixels"] == 200

    report = build_quality_resolution(**documents)

    assert report["resolution_status"] == LIMITED_PASS
    candidate = report["limitations"][0]
    assert candidate["limitation_lane"] == (
        "source_bound_zero_visible_repeated_geometry"
    )
    assert candidate["foreign_owner"] is None
    owner_diagnostic = candidate["foreign_owner_diagnostics"][0]
    assert owner_diagnostic["target_matching_pixels"] == 200
    assert owner_diagnostic["accepted_box_overlap"] is None
    assert (
        "ACCEPTED_BOX_OWNER_OVERLAP_AUDIT_MISSING" in owner_diagnostic["reason_codes"]
    )
    assert owner_diagnostic["accepted_box_overlap"] is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda docs: docs["rendered_registry"]["render_set"]["views"][0][
            "visible_parts"
        ].append({"part_id": CANDIDATES[0], "pixels": 1}),
        lambda docs: docs["spatial_report"]["view_alignments"][0].update(
            {"score": 0.749999}
        ),
        lambda docs: docs["spatial_report"]["view_alignments"][0].update(
            {"projection_iou": 0.799999}
        ),
        lambda docs: docs["spatial_report"]["view_alignments"][0].update(
            {"ecc_correlation": 0.849999}
        ),
        lambda docs: docs["quality_report"]["views"][2]["material_color"][
            "trusted_evidence_group_recall"
        ]["groups"][0].update({"reference_group_share": 0.050001}),
    ],
    ids=[
        "candidate_target_visible",
        "alignment",
        "projection_iou",
        "ecc",
        "reference_share",
    ],
)
def test_strict_boundaries_fail_closed(mutate) -> None:
    documents = _documents()
    mutate(documents)

    report = build_quality_resolution(**documents)

    assert report["resolution_status"] == FAIL_CLOSED
    assert report["material_stage_accepted"] is False
    assert report["limitations"] == []


def test_risky_or_globally_invisible_repeated_geometry_is_not_a_candidate() -> None:
    documents = _documents()
    for item in documents["geometry_risk"]["parts"]:
        if item["part_id"] in CANDIDATES:
            item["risk"]["multi_material_risk"] = True

    risky = build_quality_resolution(**documents)

    assert risky["resolution_status"] == FAIL_CLOSED
    candidate = risky["limitation_candidates"][0]
    assert "NO_SAFE_REPEATED_SOURCE_GEOMETRY_COHORT" in candidate["reason_codes"]

    documents = _documents()
    for view in documents["rendered_registry"]["render_set"]["views"][1:]:
        view["visible_parts"] = [
            item for item in view["visible_parts"] if item["part_id"] not in CANDIDATES
        ]
    invisible = build_quality_resolution(**documents)
    assert invisible["resolution_status"] == FAIL_CLOSED


def test_tampered_source_binding_fails_closed() -> None:
    documents = _documents()
    documents["final_plan"]["assignments"][0][
        "source_visual_material_binding_sha256"
    ] = "0" * 64

    report = build_quality_resolution(**documents)

    assert report["resolution_status"] == FAIL_CLOSED
    candidate = report["limitation_candidates"][0]
    assert "SOURCE_VISUAL_BINDING_VALIDATION_FAILED" in candidate["reason_codes"]


def test_other_nonpass_view_is_not_hidden_by_one_limitation() -> None:
    documents = _documents()
    documents["quality_report"]["views"][1]["status"] = "REVIEW"
    documents["quality_report"]["views"][1]["reasons"] = [
        "color_score_requires_human_review"
    ]

    report = build_quality_resolution(**documents)

    assert report["resolution_status"] == FAIL_CLOSED
    assert any(
        reason.startswith("NONPASS_VIEW_WITHOUT_LIMITABLE_GROUP")
        for reason in report["reason_codes"]
    )


def test_cli_writes_once_and_refuses_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    documents = _documents()
    arguments: list[str] = []
    for option, key in (
        ("--final-plan", "final_plan"),
        ("--policy-audit", "policy_audit"),
        ("--quality-report", "quality_report"),
        ("--palette-fusion", "palette_fusion"),
        ("--spatial-report", "spatial_report"),
        ("--geometry-risk", "geometry_risk"),
        ("--rendered-registry", "rendered_registry"),
    ):
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(documents[key]), encoding="utf-8")
        arguments.extend([option, str(path)])
    output = tmp_path / "resolution.json"
    arguments.extend(["--output", str(output)])

    assert main(arguments) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["resolution_status"] == LIMITED_PASS
    assert (
        json.loads(output.read_text(encoding="utf-8"))["resolution_status"]
        == LIMITED_PASS
    )

    with pytest.raises(QualityResolutionError, match="refusing to overwrite"):
        main(arguments)


def test_invalid_exact_cover_is_rejected_as_schema_error() -> None:
    documents = _documents()
    documents["final_plan"] = copy.deepcopy(documents["final_plan"])
    documents["final_plan"]["assignments"].pop()

    with pytest.raises(
        QualityResolutionError,
        match="does not exactly cover",
    ):
        build_quality_resolution(**documents)
