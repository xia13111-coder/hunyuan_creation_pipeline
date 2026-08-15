from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from qwen_material_pipeline.evidence.part_id_projection import (
    _apply_exact_cad_instance_material_propagation,
    _apply_source_material_binding_propagation,
    _register_similarity_mask,
    _select_material_observation_index,
    build_part_id_material_plan,
    build_part_id_reference_evidence,
    build_part_id_retrieval_request,
    evaluate_part_id_color_evidence,
)
from qwen_material_pipeline.evidence.spatial import _part_color


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class PartIdProjectionTests(unittest.TestCase):
    def test_source_material_binding_propagates_only_unanimous_observed_material(
        self,
    ) -> None:
        def registry_row(part_id: str, material: str) -> dict[str, object]:
            return {
                "part_id": part_id,
                "existing_visual_material": material,
                "existing_visual_material_properties": {
                    "shader_id": "UsdPreviewSurface",
                    "diffuseColor": [0.2, 0.2, 0.2],
                },
                "source_appearance_sha256": "a" * 64,
                "source_subset_layout_sha256": "b" * 64,
            }

        assignments = [
            {
                "part_id": "P1",
                "material_id": "mdl:Steel",
                "confidence": 0.9,
                "status": "auto",
            },
            {
                "part_id": "P2",
                "material_id": "mdl:Fallback",
                "confidence": 0.0,
                "status": "policy_fallback",
                "provenance": {"tier": "neutral_default"},
            },
            {
                "part_id": "P3",
                "material_id": "mdl:Paint",
                "confidence": 0.9,
                "status": "auto",
            },
            {
                "part_id": "P4",
                "material_id": "mdl:Plastic",
                "confidence": 0.9,
                "status": "auto",
            },
            {
                "part_id": "P5",
                "material_id": "mdl:Fallback",
                "confidence": 0.0,
                "status": "policy_fallback",
                "provenance": {"tier": "neutral_default"},
            },
        ]
        audit_rows = [
            {"part_id": "P1", "status": "independently_selected"},
            {"part_id": "P2", "status": "unobserved_preserved"},
            {"part_id": "P3", "status": "independently_selected"},
            {"part_id": "P4", "status": "independently_selected"},
            {"part_id": "P5", "status": "unobserved_preserved"},
        ]
        evidence = {
            "P1": {"status": "observed"},
            "P2": {"status": "unobserved"},
            "P3": {"status": "observed"},
            "P4": {"status": "observed"},
            "P5": {"status": "unobserved"},
        }
        registry = {
            "parts": [
                registry_row("P1", "/Asset/Looks/Steel"),
                registry_row("P2", "/Asset/Looks/Steel"),
                registry_row("P3", "/Asset/Looks/Conflicting"),
                registry_row("P4", "/Asset/Looks/Conflicting"),
                registry_row("P5", "/Asset/Looks/Conflicting"),
            ]
        }

        result = _apply_source_material_binding_propagation(
            assignments=assignments,
            audit_rows=audit_rows,
            evidence_by_part=evidence,
            part_registry=registry,
        )

        assignment_by_id = {row["part_id"]: row for row in assignments}
        audit_by_id = {row["part_id"]: row for row in audit_rows}
        self.assertEqual(assignment_by_id["P2"]["material_id"], "mdl:Steel")
        self.assertEqual(
            audit_by_id["P2"]["status"],
            "unobserved_source_binding_propagated",
        )
        self.assertEqual(assignment_by_id["P5"]["material_id"], "mdl:Fallback")
        self.assertEqual(result["summary"]["propagated_part_count"], 1)
        self.assertEqual(result["summary"]["conflict_group_count"], 1)

    def test_exact_cad_instances_propagate_only_unanimous_observed_material(
        self,
    ) -> None:
        signature = {
            "geometry_content_sha256": "a" * 64,
            "source_appearance_sha256": "b" * 64,
            "source_subset_layout_sha256": "c" * 64,
            "point_count": 32,
            "face_count": 16,
        }
        conflicting_signature = {
            "geometry_content_sha256": "d" * 64,
            "source_appearance_sha256": "e" * 64,
            "source_subset_layout_sha256": "f" * 64,
            "point_count": 24,
            "face_count": 12,
        }
        assignments = [
            {
                "part_id": "P1",
                "material_id": "mdl:Paint",
                "confidence": 0.9,
                "status": "auto",
            },
            {
                "part_id": "P2",
                "material_id": "mdl:Fallback",
                "confidence": 0.0,
                "status": "policy_fallback",
                "provenance": {"tier": "neutral_default"},
            },
            {
                "part_id": "P3",
                "material_id": "mdl:Steel",
                "confidence": 0.85,
                "status": "auto",
            },
            {
                "part_id": "P4",
                "material_id": "mdl:Plastic",
                "confidence": 0.85,
                "status": "auto",
            },
            {
                "part_id": "P5",
                "material_id": "mdl:Fallback",
                "confidence": 0.0,
                "status": "policy_fallback",
                "provenance": {"tier": "neutral_default"},
            },
        ]
        audit_rows = [
            {"part_id": "P1", "status": "independently_selected"},
            {"part_id": "P2", "status": "unobserved_preserved"},
            {"part_id": "P3", "status": "independently_selected"},
            {"part_id": "P4", "status": "independently_selected"},
            {"part_id": "P5", "status": "unobserved_preserved"},
        ]
        evidence = {
            "P1": {"status": "observed"},
            "P2": {"status": "unobserved"},
            "P3": {"status": "observed"},
            "P4": {"status": "observed"},
            "P5": {"status": "unobserved"},
        }
        registry = {
            "parts": [
                {"part_id": part_id, **row}
                for part_id, row in (
                    ("P1", signature),
                    ("P2", signature),
                    ("P3", conflicting_signature),
                    ("P4", conflicting_signature),
                    ("P5", conflicting_signature),
                )
            ]
        }

        result = _apply_exact_cad_instance_material_propagation(
            assignments=assignments,
            audit_rows=audit_rows,
            evidence_by_part=evidence,
            part_registry=registry,
        )

        assignment_by_id = {row["part_id"]: row for row in assignments}
        audit_by_id = {row["part_id"]: row for row in audit_rows}
        self.assertEqual(assignment_by_id["P2"]["material_id"], "mdl:Paint")
        self.assertEqual(assignment_by_id["P2"]["status"], "review")
        self.assertEqual(
            audit_by_id["P2"]["status"],
            "unobserved_exact_instance_propagated",
        )
        self.assertEqual(assignment_by_id["P5"]["material_id"], "mdl:Fallback")
        self.assertEqual(audit_by_id["P5"]["status"], "unobserved_preserved")
        self.assertEqual(result["summary"]["propagated_part_count"], 1)
        self.assertEqual(result["summary"]["conflict_group_count"], 1)

    def test_tiny_chromatic_component_can_propose_h1_without_mvinverse_pixel(
        self,
    ) -> None:
        gate = evaluate_part_id_color_evidence(
            part_evidence={
                "part_id": "P0002",
                "status": "observed",
                "descriptor": {
                    "robust_color_evidence": {
                        "method": "cielab_medoid_fixed_radius",
                        "sample_count": 6,
                        "evaluated_sample_count": 6,
                        "inlier_delta_e_radius": 20.0,
                        "inlier_fraction": 0.5,
                        "median_delta_e": 20.0,
                        "p90_delta_e": 30.0,
                        "robust_reference_srgb": [0.2, 0.55, 0.72],
                    }
                },
                "observations": [
                    {
                        "view_id": "top",
                        "trusted_foreground_pixels": 6,
                        "sampling_projection_pixels": 16,
                        "foreground_overlap": 0.65,
                        "alignment_score": 0.9,
                        "chromatic_coverage": {
                            "applied": True,
                            "tiny_part_rescue": True,
                        },
                        "selected_for_material_inference": True,
                    }
                ],
            },
            sam3_role=(
                "human_confirmed_whole_workpiece_plus_automatic_local_part_refinement"
            ),
        )

        self.assertEqual(gate["status"], "PASS")
        self.assertTrue(gate["eligible_for_h1_color_candidate"])
        self.assertEqual(
            gate["target_color_source"],
            "single_view_chromatic_component",
        )
        self.assertEqual(gate["target_color_srgb"], [0.2, 0.55, 0.72])
        self.assertEqual(
            gate["components"]["minimum_foreground_overlap"],
            0.6,
        )

    def test_small_parts_prefer_pure_single_view_and_rescue_tiny_color(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_views = []
            render_views = []
            alignments = []
            for view_id in ("front", "side"):
                image = np.zeros((96, 96, 3), dtype=np.uint8)
                foreground = np.zeros((96, 96), dtype=np.uint8)
                part_ids = np.zeros((96, 96, 3), dtype=np.uint8)
                if view_id == "front":
                    # More pixels, but the Part-ID projection is visibly
                    # contaminated by two incompatible chromatic surfaces.
                    image[10:20, 10:16] = (35, 130, 45)
                    image[10:20, 16:20] = (190, 105, 45)
                    foreground[10:20, 10:20] = 255
                    part_ids[10:20, 10:20] = _part_color("P0001")
                    # A 20-pixel connector would fail the normal 32-pixel
                    # gate. Eight coherent cyan pixels are sufficient for the
                    # bounded single-view chromatic rescue.
                    image[30:32, 30:34] = (70, 175, 215)
                    image[32:35, 30:34] = (20, 20, 20)
                    foreground[30:35, 30:34] = 255
                    part_ids[30:35, 30:34] = _part_color("P0002")
                    # A tiny CAD projection completely hidden behind a green
                    # foreground surface must not rescue the occluder's color.
                    image[50:60, 50:60] = (35, 130, 45)
                    foreground[50:60, 50:60] = 255
                    part_ids[53:57, 53:57] = _part_color("P0003")
                else:
                    image[12:20, 12:20] = (190, 105, 45)
                    foreground[12:20, 12:20] = 255
                    part_ids[12:20, 12:20] = _part_color("P0001")
                image_path = root / f"{view_id}.png"
                foreground_path = root / f"{view_id}.mask.png"
                part_ids_path = root / f"{view_id}.parts.png"
                Image.fromarray(image, mode="RGB").save(image_path)
                Image.fromarray(foreground, mode="L").save(foreground_path)
                Image.fromarray(part_ids, mode="RGB").save(part_ids_path)
                source_views.append(
                    {
                        "id": view_id,
                        "image": str(image_path),
                        "palette_mask": str(foreground_path),
                        "palette_mask_authority": (
                            "sam3_foreground_before_material_inference"
                        ),
                    }
                )
                render_views.append(
                    {
                        "view_id": view_id,
                        "part_ids_raw": str(part_ids_path),
                    }
                )
                alignments.append(
                    {
                        "reference_view_id": view_id,
                        "selected_render_view_id": view_id,
                        "trusted": True,
                        "observation_eligible": True,
                        "quarter_turns_ccw": 0,
                        "bbox_affine": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                        ],
                        "ecc_warp": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                        ],
                        "score": 1.0,
                    }
                )
            evidence = build_part_id_reference_evidence(
                reference_manifest={"source_views": source_views},
                rendered_registry={
                    "parts": [
                        {"part_id": "P0001"},
                        {"part_id": "P0002"},
                        {"part_id": "P0003"},
                    ],
                    "render_set": {"views": render_views},
                },
                spatial_mapping_report={"view_alignments": alignments},
                output_dir=root / "evidence",
            )
            by_id = {row["part_id"]: row for row in evidence["parts"]}
            self.assertEqual(
                by_id["P0001"]["selected_observation_view_id"],
                "side",
            )
            self.assertEqual(
                by_id["P0001"]["observation_selection_audit"]["policy"],
                "small_part_single_view_chromatic_purity_first",
            )
            self.assertEqual(
                by_id["P0002"]["selected_observation_view_id"],
                "front",
            )
            tiny = by_id["P0002"]["observations"][0]
            self.assertTrue(tiny["chromatic_coverage"]["tiny_part_rescue"])
            self.assertEqual(by_id["P0003"]["status"], "unobserved")
            self.assertEqual(tiny["trusted_foreground_pixels"], 8)
            self.assertEqual(
                evidence["summary"]["tiny_chromatic_rescue_observation_count"],
                1,
            )

    def test_same_coating_parts_share_one_mdl_but_keep_native_h0(self) -> None:
        vinyl = "mdl:Plastics/Vinyl.mdl#Vinyl"
        chrome = "mdl:Metals/Chrome.mdl#Chrome"
        rubber = "mdl:Plastics/Rubber_Textured.mdl#Rubber_Textured"
        colors = {
            "P0001": [0.21, 0.52, 0.18],
            "P0002": [0.23, 0.49, 0.16],
            "P0003": [0.06, 0.08, 0.05],
        }
        pixels = {"P0001": 4000, "P0002": 1800, "P0003": 900}
        evidence = {
            "schema_version": "qwen-part-id-reference-evidence/v1",
            "sam3_role": "whole_workpiece_foreground",
            "part_segmentation_authority": "cad_part_id_projection",
            "integrity": {"document_sha256": "evidence"},
            "parts": [
                {
                    "part_id": part_id,
                    "status": "observed",
                    "descriptor": {"mvinverse_albedo_median_rgb": colors[part_id]},
                    "observations": [
                        {
                            "view_id": "front",
                            "mask_sha256": part_id.lower().ljust(64, "0"),
                            "trusted_foreground_pixels": pixels[part_id],
                            "foreground_overlap": 0.95,
                            "alignment_score": 0.95,
                            "selected_for_material_inference": True,
                        }
                    ],
                }
                for part_id in ("P0001", "P0002", "P0003")
            ],
        }
        retrieval_unsigned = {
            "schema_version": "qwen-visual-material-retrieval-result/v1",
            "groups": [
                {
                    "group_id": part_id,
                    "fused_ranking": [
                        {"rank": 1, "material_id": material_id, "score": 0.08},
                        {"rank": 2, "material_id": vinyl, "score": 0.07},
                        {"rank": 3, "material_id": chrome, "score": 0.06},
                        {"rank": 4, "material_id": rubber, "score": 0.05},
                    ],
                }
                for part_id, material_id in (
                    ("P0001", vinyl),
                    ("P0002", chrome),
                    ("P0003", rubber),
                )
            ],
        }
        retrieval = {
            **retrieval_unsigned,
            "integrity": {"result_sha256": canonical_sha256(retrieval_unsigned)},
        }
        base_plan = {
            "schema_version": "1.0",
            "assignments": [
                {
                    "part_id": part_id,
                    "material_id": "mdl:Fallback.mdl#Fallback",
                    "semantic": "fallback",
                    "confidence": 0.0,
                    "evidence_views": [],
                    "status": "policy_fallback",
                }
                for part_id in ("P0001", "P0002", "P0003")
            ],
            "provenance": {},
        }
        registry = {
            "default_prim": "/Asset",
            "parts": [
                {
                    "part_id": part_id,
                    "parent_path": f"/Asset/Asset/OuterShell/{part_id}",
                    "source_appearance_sha256": "a" * 64,
                }
                for part_id in ("P0001", "P0002", "P0003")
            ],
        }

        plan, audit = build_part_id_material_plan(
            base_plan=base_plan,
            evidence=evidence,
            retrieval_result=retrieval,
            qwen_choices={
                "P0001": vinyl,
                "P0002": chrome,
                "P0003": rubber,
            },
            qwen_confidences={
                "P0001": 0.92,
                "P0002": 0.90,
                "P0003": 0.91,
            },
            allow_color_parameters=True,
            part_registry=registry,
        )

        assignments = {row["part_id"]: row for row in plan["assignments"]}
        self.assertEqual(assignments["P0001"]["material_id"], vinyl)
        self.assertEqual(assignments["P0002"]["material_id"], vinyl)
        self.assertEqual(assignments["P0003"]["material_id"], rubber)
        self.assertNotIn("parameters", assignments["P0001"])
        self.assertNotIn("parameters", assignments["P0002"])
        self.assertEqual(
            assignments["P0001"]["provenance"]["mdl_parameter_candidates"][
                "selected_candidate_id"
            ],
            "H0",
        )
        self.assertNotIn(
            "coating_consistency",
            assignments["P0003"]["provenance"],
        )
        gate = audit["coating_consistency_gate"]
        self.assertEqual(gate["status"], "PASS")
        self.assertEqual(gate["summary"]["component_count"], 1)
        self.assertEqual(gate["summary"]["constrained_part_count"], 2)
        self.assertEqual(
            gate["summary"]["material_changed_part_ids"],
            ["P0002"],
        )
        self.assertFalse(plan["part_material_groups_used"])
        self.assertTrue(plan["coating_consistency_used"])

    def test_part_id_plan_generates_but_does_not_apply_h1_color(self) -> None:
        material_id = "mdl:Plastics/Vinyl.mdl#Vinyl"
        evidence = {
            "schema_version": "qwen-part-id-reference-evidence/v1",
            "sam3_role": "whole_workpiece_foreground",
            "part_segmentation_authority": "cad_part_id_projection",
            "integrity": {"document_sha256": "evidence"},
            "parts": [
                {
                    "part_id": "P0094",
                    "status": "observed",
                    "descriptor": {
                        "median_rgb": [0.204, 0.459, 0.247],
                        "mvinverse_albedo_median_rgb": [0.212, 0.525, 0.192],
                        "robust_color_evidence": {
                            "method": "cielab_medoid_fixed_radius",
                            "sample_count": 1200,
                            "evaluated_sample_count": 1200,
                            "inlier_delta_e_radius": 20.0,
                            "inlier_fraction": 0.94,
                            "median_delta_e": 4.2,
                            "p90_delta_e": 11.0,
                            "robust_reference_srgb": [0.204, 0.50, 0.20],
                        },
                    },
                    "observations": [
                        {
                            "view_id": "front",
                            "mask_sha256": "a" * 64,
                            "trusted_foreground_pixels": 1200,
                            "sampling_projection_pixels": 1250,
                            "foreground_overlap": 0.96,
                            "alignment_score": 0.94,
                            "selected_for_material_inference": True,
                        }
                    ],
                }
            ],
        }
        retrieval_unsigned = {
            "schema_version": "qwen-visual-material-retrieval-result/v1",
            "groups": [
                {
                    "group_id": "P0094",
                    "fused_ranking": [
                        {
                            "rank": 1,
                            "material_id": material_id,
                            "score": 0.08,
                        }
                    ],
                }
            ],
        }
        retrieval = {
            **retrieval_unsigned,
            "integrity": {"result_sha256": canonical_sha256(retrieval_unsigned)},
        }
        plan, audit = build_part_id_material_plan(
            base_plan={
                "schema_version": "1.0",
                "assignments": [
                    {
                        "part_id": "P0094",
                        "material_id": "mdl:Fallback.mdl#Fallback",
                        "semantic": "fallback",
                        "confidence": 0.0,
                        "evidence_views": [],
                        "status": "policy_fallback",
                    }
                ],
                "provenance": {},
            },
            evidence=evidence,
            retrieval_result=retrieval,
            qwen_choices={"P0094": material_id},
            qwen_confidences={"P0094": 0.91},
            allow_color_parameters=True,
        )
        assignment = plan["assignments"][0]
        self.assertNotIn("parameters", assignment)
        candidate_set = assignment["provenance"]["mdl_parameter_candidates"]
        self.assertEqual(candidate_set["selected_candidate_id"], "H0")
        self.assertEqual(len(candidate_set["candidates"]), 2)
        h1 = candidate_set["candidates"][1]
        self.assertEqual(h1["candidate_id"], "H1")
        self.assertEqual(set(h1["parameters"]), {"diffuse_tint"})
        self.assertEqual(
            assignment["provenance"]["mdl_color_parameterization"]["color_source"],
            "mvinverse_albedo_evidence_gated",
        )
        self.assertEqual(
            assignment["provenance"]["mdl_color_parameterization"][
                "color_parameter_semantics"
            ],
            "absolute_linear_color",
        )
        self.assertLess(
            max(h1["parameters"]["diffuse_tint"]),
            1.0,
        )
        self.assertEqual(audit["summary"]["color_parameterized_count"], 0)
        self.assertEqual(audit["summary"]["h1_color_candidate_count"], 1)
        self.assertEqual(
            audit["parts"][0]["mdl_color_parameterization"]["base_color_srgb"],
            [0.212, 0.525, 0.192],
        )

    def test_projection_fallback_rejects_h1_without_material_specific_rules(
        self,
    ) -> None:
        material_id = "mdl:Plastics/Vinyl.mdl#Vinyl"
        evidence = {
            "schema_version": "qwen-part-id-reference-evidence/v1",
            "sam3_role": (
                "human_confirmed_whole_workpiece_plus_automatic_local_part_refinement"
            ),
            "part_segmentation_authority": "cad_part_id_projection",
            "integrity": {"document_sha256": "evidence"},
            "parts": [
                {
                    "part_id": "P0001",
                    "status": "observed",
                    "descriptor": {
                        "mvinverse_albedo_median_rgb": [0.2, 0.6, 0.2],
                        "robust_color_evidence": {
                            "method": "cielab_medoid_fixed_radius",
                            "sample_count": 1000,
                            "evaluated_sample_count": 1000,
                            "inlier_delta_e_radius": 20.0,
                            "inlier_fraction": 0.95,
                            "median_delta_e": 3.0,
                            "p90_delta_e": 8.0,
                            "robust_reference_srgb": [0.2, 0.59, 0.2],
                        },
                    },
                    "observations": [
                        {
                            "view_id": "front",
                            "mask_sha256": "b" * 64,
                            "trusted_foreground_pixels": 1000,
                            "sampling_projection_pixels": 1000,
                            "foreground_overlap": 1.0,
                            "alignment_score": 0.98,
                            "boundary_policy": (
                                "global_cad_projection_human_foreground_fallback"
                            ),
                            "part_id_sam3_refinement": {
                                "applied": False,
                                "status": "local_sam3_missing_or_rejected",
                            },
                            "selected_for_material_inference": True,
                        }
                    ],
                }
            ],
        }
        retrieval_unsigned = {
            "schema_version": "qwen-visual-material-retrieval-result/v1",
            "groups": [
                {
                    "group_id": "P0001",
                    "fused_ranking": [
                        {
                            "rank": 1,
                            "material_id": material_id,
                            "score": 0.08,
                        }
                    ],
                }
            ],
        }
        retrieval = {
            **retrieval_unsigned,
            "integrity": {"result_sha256": canonical_sha256(retrieval_unsigned)},
        }
        plan, audit = build_part_id_material_plan(
            base_plan={
                "schema_version": "1.0",
                "assignments": [
                    {
                        "part_id": "P0001",
                        "material_id": "mdl:Fallback.mdl#Fallback",
                        "semantic": "fallback",
                        "confidence": 0.0,
                        "evidence_views": [],
                        "status": "policy_fallback",
                    }
                ],
                "provenance": {},
            },
            evidence=evidence,
            retrieval_result=retrieval,
            qwen_choices={"P0001": material_id},
            qwen_confidences={"P0001": 0.91},
            allow_color_parameters=True,
        )
        candidate_set = plan["assignments"][0]["provenance"]["mdl_parameter_candidates"]
        self.assertEqual(
            [row["candidate_id"] for row in candidate_set["candidates"]], ["H0"]
        )
        gate = audit["parts"][0]["mdl_color_parameterization"]["color_evidence_gate"]
        self.assertIn(
            "local_part_refinement_not_applied",
            gate["reason_codes"],
        )
        self.assertIn(
            "projection_fallback_not_color_authoritative",
            gate["reason_codes"],
        )

    def test_small_part_near_coverage_tie_prefers_clear_color_evidence(
        self,
    ) -> None:
        observations = [
            {
                "view_id": "top",
                "trusted_foreground_pixels": 101,
                "foreground_overlap": 1.0,
                "alignment_score": 0.95,
            },
            {
                "view_id": "front",
                "trusted_foreground_pixels": 95,
                "foreground_overlap": 0.97,
                "alignment_score": 0.93,
            },
        ]
        greenish = np.tile(np.asarray([[67, 114, 61]], dtype=np.uint8), (101, 1))
        copper = np.tile(np.asarray([[125, 40, 8]], dtype=np.uint8), (95, 1))
        selected, audit = _select_material_observation_index(
            observations=observations,
            rgb_samples=[greenish, copper],
        )
        self.assertEqual(selected, 1)
        self.assertEqual(
            audit["policy"],
            "small_part_near_coverage_tie_colorfulness_first",
        )

    def test_similarity_registration_fits_scale_rotation_and_translation(self) -> None:
        source = np.zeros((180, 220), dtype=np.uint8)
        source[55:105, 70:95] = 255
        source[85:112, 70:135] = 255
        matrix = cv2.getRotationMatrix2D((102.0, 84.0), 17.0, 1.28)
        matrix[:, 2] += np.asarray((23.0, -11.0), dtype=np.float64)
        target = cv2.warpAffine(
            source,
            matrix,
            (source.shape[1], source.shape[0]),
            flags=cv2.INTER_NEAREST,
        )

        registered, audit = _register_similarity_mask(source, target)

        intersection = np.count_nonzero((registered > 0) & (target > 0))
        union = np.count_nonzero((registered > 0) | (target > 0))
        self.assertGreater(intersection / union, 0.88)
        self.assertGreater(float(audit["precision"]), 0.90)
        self.assertGreater(float(audit["recall"]), 0.90)

    def test_similarity_registration_preserves_tiny_masks_in_large_images(self) -> None:
        source = np.zeros((2048, 2048), dtype=np.uint8)
        source[1000:1003, 900:904] = 255
        target = np.zeros_like(source)
        target[1011:1015, 918:923] = 255

        registered, audit = _register_similarity_mask(source, target)

        self.assertGreater(int(np.count_nonzero(registered)), 0)
        self.assertGreater(float(audit["iou"]), 0.55)
        self.assertEqual(
            audit["optimization"],
            "local_union_crop_bounded_uniform_scale_rotation_translation_"
            "coarse_to_fine_full_resolution",
        )

    def test_similarity_registration_accepts_different_render_and_photo_sizes(
        self,
    ) -> None:
        source = np.zeros((512, 512), dtype=np.uint8)
        source[180:330, 205:255] = 255
        source[280:350, 205:340] = 255
        source_centroid = (256.0, 265.0)
        target_centroid = (221.0, 305.0)
        matrix = cv2.getRotationMatrix2D(source_centroid, -13.0, 0.82)
        matrix[:, 2] += np.asarray(
            (
                target_centroid[0] - source_centroid[0],
                target_centroid[1] - source_centroid[1],
            ),
            dtype=np.float64,
        )
        target = cv2.warpAffine(
            source,
            matrix,
            (443, 582),
            flags=cv2.INTER_NEAREST,
        )

        registered, audit = _register_similarity_mask(source, target)

        self.assertEqual(registered.shape, target.shape)
        self.assertGreater(float(audit["iou"]), 0.85)
        self.assertEqual(audit["source_canvas_hw"], [512, 512])
        self.assertEqual(audit["target_canvas_hw"], [582, 443])
        self.assertEqual(audit["optimization_domain"], "independent_full_canvases")

    def test_each_part_id_gets_an_independent_reference_mask_and_material(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = np.full((96, 96, 3), 127, dtype=np.uint8)
            reference[16:64, 12:40] = (220, 30, 30)
            reference[20:72, 52:84] = (25, 55, 225)
            foreground = np.zeros((96, 96), dtype=np.uint8)
            foreground[16:64, 12:40] = 255
            foreground[20:72, 52:84] = 255
            part_ids = np.zeros((96, 96, 3), dtype=np.uint8)
            part_ids[16:64, 12:40] = _part_color("P0001")
            part_ids[20:72, 52:84] = _part_color("P0002")

            reference_path = root / "reference.png"
            foreground_path = root / "foreground.png"
            part_ids_path = root / "part_ids.png"
            Image.fromarray(reference, mode="RGB").save(reference_path)
            Image.fromarray(foreground, mode="L").save(foreground_path)
            Image.fromarray(part_ids, mode="RGB").save(part_ids_path)

            manifest = {
                "source_views": [
                    {
                        "id": "front",
                        "image": str(reference_path),
                        "palette_mask": str(foreground_path),
                        "palette_mask_authority": (
                            "sam3_foreground_before_material_inference"
                        ),
                    }
                ]
            }
            registry = {
                "parts": [
                    {"part_id": "P0001"},
                    {"part_id": "P0002"},
                    {"part_id": "P0003"},
                ],
                "render_set": {
                    "views": [
                        {
                            "view_id": "front",
                            "part_ids_raw": str(part_ids_path),
                        }
                    ]
                },
            }
            spatial = {
                "view_alignments": [
                    {
                        "reference_view_id": "front",
                        "selected_render_view_id": "front",
                        "trusted": True,
                        "observation_eligible": True,
                        "quarter_turns_ccw": 0,
                        "bbox_affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        "ecc_warp": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        "score": 1.0,
                    }
                ]
            }
            evidence = build_part_id_reference_evidence(
                reference_manifest=manifest,
                rendered_registry=registry,
                spatial_mapping_report=spatial,
                camera_alignment_acceptance={
                    "policy": "tiered_box_first_part_id_alignment/v1",
                    "views": {
                        "front": {
                            "tier": "downweighted_box_correspondence",
                            "evidence_weight": 0.55,
                        }
                    },
                },
                output_dir=root / "evidence",
                minimum_projected_pixels=8,
            )
            self.assertEqual(evidence["assignment_unit"], "part_id")
            self.assertEqual(evidence["sam3_role"], "whole_workpiece_foreground_only")
            self.assertFalse(evidence["cross_view_consensus_required"])
            self.assertEqual(evidence["summary"]["observed_part_count"], 2)
            by_id = {part["part_id"]: part for part in evidence["parts"]}
            self.assertEqual(by_id["P0001"]["status"], "observed")
            self.assertEqual(by_id["P0002"]["status"], "observed")
            self.assertEqual(by_id["P0003"]["status"], "unobserved")
            self.assertNotEqual(
                by_id["P0001"]["observations"][0]["mask_sha256"],
                by_id["P0002"]["observations"][0]["mask_sha256"],
            )
            for part_id in ("P0001", "P0002"):
                observation = by_id[part_id]["observations"][0]
                self.assertEqual(
                    observation["correspondence_mode"],
                    "cad_projected_part_id_bounding_box",
                )
                self.assertFalse(observation["photo_part_segmentation_applied"])
                self.assertEqual(
                    observation["camera_alignment_tier"],
                    "downweighted_box_correspondence",
                )
                self.assertEqual(observation["camera_alignment_evidence_weight"], 0.55)
                self.assertEqual(observation["alignment_score"], 0.55)
                self.assertTrue(Path(observation["box_mask"]).is_file())
                self.assertTrue(Path(observation["crop"]).is_file())
                self.assertTrue(Path(observation["isolated_crop"]).is_file())
                core = cv2.imread(observation["mask"], cv2.IMREAD_GRAYSCALE)
                box = cv2.imread(observation["box_mask"], cv2.IMREAD_GRAYSCALE)
                self.assertTrue(np.all(box[core >= 128] >= 128))
                self.assertGreaterEqual(
                    observation["box_foreground_pixels"],
                    observation["trusted_foreground_pixels"],
                )

            refinement_dir = root / "refinement"
            refinement_masks = refinement_dir / "masks"
            refinement_masks.mkdir(parents=True)
            refinement_records = []
            for part_id in ("P0001", "P0002"):
                source_mask = Path(by_id[part_id]["observations"][0]["mask"])
                target_mask = refinement_masks / f"front__{part_id}.png"
                source_array = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
                shifted = cv2.warpAffine(
                    source_array,
                    np.asarray([[1.0, 0.0, 3.0], [0.0, 1.0, 2.0]]),
                    (source_array.shape[1], source_array.shape[0]),
                    flags=cv2.INTER_NEAREST,
                )
                Image.fromarray(shifted, mode="L").save(target_mask)
                refinement_records.append(
                    {
                        "view_id": "front",
                        "group_id": part_id,
                        "accepted": True,
                        "mask": {
                            "path": f"masks/front__{part_id}.png",
                            "sha256": hashlib.sha256(
                                target_mask.read_bytes()
                            ).hexdigest(),
                        },
                    }
                )
            refinement_unsigned = {
                "schema_version": "qwen-sam3-region-result/v1",
                "records": refinement_records,
            }
            refinement = {
                **refinement_unsigned,
                "integrity": {"result_sha256": canonical_sha256(refinement_unsigned)},
            }
            refinement_path = refinement_dir / "manifest.json"
            refinement_path.write_text(json.dumps(refinement), encoding="utf-8")
            refined_evidence = build_part_id_reference_evidence(
                reference_manifest=manifest,
                rendered_registry=registry,
                spatial_mapping_report=spatial,
                part_id_sam3_manifest=refinement_path,
                output_dir=root / "refined_evidence",
                minimum_projected_pixels=8,
            )
            self.assertEqual(
                refined_evidence["sam3_role"],
                (
                    "human_confirmed_whole_workpiece_plus_"
                    "automatic_local_part_refinement"
                ),
            )
            self.assertEqual(
                refined_evidence["summary"]["sam3_refined_observation_count"],
                2,
            )
            self.assertTrue(
                all(
                    observation["part_id_sam3_refinement"] is not None
                    for part in refined_evidence["parts"]
                    for observation in part["observations"]
                )
            )
            for part in refined_evidence["parts"]:
                for observation in part["observations"]:
                    refinement_audit = observation["part_id_sam3_refinement"]
                    self.assertTrue(refinement_audit["applied"])
                    self.assertTrue(
                        refinement_audit["per_part_geometric_warp_applied"]
                    )
                    self.assertNotEqual(
                        refinement_audit["registration"]["affine_2x3"],
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    )

            catalog_path = root / "catalog.json"
            catalog_path.write_text("{}", encoding="utf-8")
            material_root = root / "Base"
            material_root.mkdir()
            request = build_part_id_retrieval_request(
                evidence=evidence,
                catalog=catalog_path,
                material_root=material_root,
            )
            self.assertEqual(request["assignment_unit"], "part_id")
            self.assertEqual(
                {entity["group_id"] for entity in request["groups"]},
                {"P0001", "P0002"},
            )
            self.assertTrue(
                all(
                    entity["part_id"] == entity["group_id"]
                    for entity in request["groups"]
                )
            )
            for entity in request["groups"]:
                observation = entity["observations"][0]
                source = by_id[entity["part_id"]]["observations"][0]
                self.assertEqual(observation["mask"], source["mask"])
                self.assertEqual(observation["core_mask"], source["mask"])
                self.assertEqual(
                    observation["correspondence_box_mask"], source["box_mask"]
                )
                self.assertEqual(
                    observation["material_sampling_mode"],
                    "registered_cad_part_id_core",
                )
                self.assertEqual(
                    observation["correspondence_mode"],
                    "cad_projected_part_id_bounding_box",
                )

            retrieval_unsigned = {
                "schema_version": "qwen-visual-material-retrieval-result/v1",
                "groups": [
                    {
                        "group_id": "P0001",
                        "fused_ranking": [
                            {
                                "rank": 1,
                                "material_id": "mdl:Red.mdl#Red",
                                "score": 0.08,
                            },
                            {
                                "rank": 2,
                                "material_id": "mdl:Alt.mdl#Alt",
                                "score": 0.07,
                            },
                        ],
                    },
                    {
                        "group_id": "P0002",
                        "fused_ranking": [
                            {
                                "rank": 1,
                                "material_id": "mdl:Blue.mdl#Blue",
                                "score": 0.09,
                            },
                            {
                                "rank": 2,
                                "material_id": "mdl:Alt.mdl#Alt",
                                "score": 0.07,
                            },
                        ],
                    },
                ],
            }
            retrieval = {
                **retrieval_unsigned,
                "integrity": {"result_sha256": canonical_sha256(retrieval_unsigned)},
            }
            base_plan = {
                "schema_version": "1.0",
                "assignments": [
                    {
                        "part_id": part_id,
                        "material_id": "mdl:Fallback.mdl#Fallback",
                        "semantic": "fallback",
                        "confidence": 0.0,
                        "evidence_views": [],
                        "status": "policy_fallback",
                        **(
                            {
                                "apply_action": "source_visual_preserve",
                                "source_visual_material_prim_path": (
                                    "/Asset/Looks/Source"
                                ),
                                "source_visual_material_binding_sha256": (
                                    "stale-source-binding-digest"
                                ),
                                "preserve_parent_material_binding": True,
                                "face_subsets": [
                                    {
                                        "subset_name": "stale",
                                        "face_indices": [0],
                                    }
                                ],
                                "parameters": {"roughness": 0.5},
                                "canonical_group_id": "G01",
                            }
                            if part_id == "P0001"
                            else {}
                        ),
                    }
                    for part_id in ("P0001", "P0002", "P0003")
                ],
                "provenance": {"mode": "test"},
            }
            plan, audit = build_part_id_material_plan(
                base_plan=base_plan,
                evidence=evidence,
                retrieval_result=retrieval,
                qwen_choices={
                    "P0001": "mdl:Red.mdl#Red",
                    "P0002": "mdl:Blue.mdl#Blue",
                },
                qwen_confidences={"P0001": 0.73, "P0002": 0.55},
                qwen_material_predictions={
                    "P0001": {
                        "part_id": "P0001",
                        "catalog_family": "paint",
                        "physical_substrate": "metal",
                        "material_species": "paint",
                        "surface_treatment": "paint",
                        "optical_behavior": "opaque",
                        "surface_finish": "matte",
                        "substrate_confidence": 0.91,
                        "species_confidence": 0.90,
                        "treatment_confidence": 0.90,
                        "confidence": 0.91,
                        "identity_resolution": "exact_material",
                        "status": "APPLYABLE",
                    },
                    "P0002": {
                        "part_id": "P0002",
                        "catalog_family": "paint",
                        "physical_substrate": "metal",
                        "material_species": "paint",
                        "surface_treatment": "paint",
                        "optical_behavior": "opaque",
                        "surface_finish": "glossy",
                        "substrate_confidence": 0.88,
                        "species_confidence": 0.87,
                        "treatment_confidence": 0.86,
                        "confidence": 0.88,
                        "identity_resolution": "exact_material",
                        "status": "APPLYABLE",
                    },
                },
            )
            assignments = {
                assignment["part_id"]: assignment for assignment in plan["assignments"]
            }
            self.assertEqual(assignments["P0001"]["material_id"], "mdl:Red.mdl#Red")
            self.assertEqual(
                assignments["P0002"]["material_id"], "mdl:Fallback.mdl#Fallback"
            )
            self.assertEqual(assignments["P0002"]["status"], "policy_fallback")
            self.assertTrue(
                assignments["P0002"]["provenance"][
                    "observed_part_id_qwen_selection_rejected"
                ]
            )
            self.assertEqual(
                assignments["P0003"]["material_id"],
                "mdl:Fallback.mdl#Fallback",
            )
            self.assertIsNone(
                assignments["P0001"]["provenance"]["material_region_group_id"]
            )
            self.assertFalse(assignments["P0001"]["provenance"]["palette_fusion_used"])
            self.assertEqual(assignments["P0001"]["confidence"], 0.73)
            self.assertEqual(
                assignments["P0001"]["provenance"]["qwen_confidence"],
                0.73,
            )
            self.assertEqual(
                assignments["P0001"]["provenance"]["material_prediction"],
                {
                    "part_id": "P0001",
                    "catalog_family": "paint",
                    "physical_substrate": "metal",
                    "material_species": "paint",
                    "surface_treatment": "paint",
                    "optical_behavior": "opaque",
                    "surface_finish": "matte",
                    "substrate_confidence": 0.91,
                    "species_confidence": 0.90,
                    "treatment_confidence": 0.90,
                    "confidence": 0.91,
                    "identity_resolution": "exact_material",
                    "status": "APPLYABLE",
                },
            )
            self.assertEqual(
                audit["summary"]["material_prediction_count"],
                2,
            )
            for stale_field in (
                "apply_action",
                "source_visual_material_prim_path",
                "source_visual_material_binding_sha256",
                "preserve_parent_material_binding",
                "face_subsets",
                "parameters",
                "canonical_group_id",
            ):
                self.assertNotIn(stale_field, assignments["P0001"])
            self.assertEqual(plan["assignment_unit"], "part_id")
            self.assertFalse(plan["palette_fusion_used"])
            self.assertFalse(plan["part_material_groups_used"])
            self.assertEqual(assignments["P0001"]["evidence_views"], ["front"])
            self.assertEqual(audit["summary"]["independently_selected_count"], 1)
            self.assertEqual(audit["summary"]["unobserved_preserved_count"], 1)
            self.assertEqual(
                audit["summary"]["observed_low_confidence_baseline_retained_count"],
                1,
            )
            self.assertTrue(audit["summary"]["exact_cover"])


if __name__ == "__main__":
    unittest.main()
