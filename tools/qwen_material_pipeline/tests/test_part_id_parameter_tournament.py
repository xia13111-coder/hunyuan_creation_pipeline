from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from qwen_material_pipeline.materials.part_id_parameter_tournament import (
    build_h1_candidate_plan,
    pending_h1_part_ids,
    score_part_id_render,
    select_parameter_tournament_winners,
)


def source_plan() -> dict:
    return {
        "schema_version": "1.0",
        "assignment_unit": "part_id",
        "palette_fusion_used": False,
        "part_material_groups_used": False,
        "assignments": [
            {
                "part_id": "P0001",
                "material_id": "mdl:Any.mdl#Any",
                "semantic": "test",
                "confidence": 0.9,
                "evidence_views": ["front"],
                "status": "auto",
                "provenance": {
                    "assignment_unit": "part_id",
                    "mdl_color_parameterization": {
                        "status": "native_h0_selected",
                        "selected_candidate_id": "H0",
                        "parameters_applied": False,
                    },
                    "mdl_parameter_candidates": {
                        "schema_version": ("qwen-part-id-parameter-candidates/v1"),
                        "part_id": "P0001",
                        "material_id": "mdl:Any.mdl#Any",
                        "selection_status": "PENDING_RENDER_COMPARISON",
                        "selected_candidate_id": "H0",
                        "native_h0_is_default": True,
                        "parameters_applied_to_plan": False,
                        "h1_status": "generated_pending_render_comparison",
                        "candidates": [
                            {
                                "candidate_id": "H0",
                                "kind": "native_mdl",
                                "material_id": "mdl:Any.mdl#Any",
                                "parameters": {},
                            },
                            {
                                "candidate_id": "H1",
                                "kind": "evidence_gated_color_only",
                                "material_id": "mdl:Any.mdl#Any",
                                "parameters": {"diffuse_tint": [0.2, 1.0, 0.1]},
                            },
                        ],
                    },
                },
            }
        ],
        "provenance": {},
    }


class PartIdParameterTournamentTests(unittest.TestCase):
    def test_h1_candidate_plan_changes_only_parameters(self) -> None:
        baseline = source_plan()
        candidate = build_h1_candidate_plan(
            source_plan=baseline,
            part_id="P0001",
        )
        self.assertNotIn("parameters", baseline["assignments"][0])
        self.assertEqual(
            candidate["assignments"][0]["parameters"],
            {"diffuse_tint": [0.2, 1.0, 0.1]},
        )
        self.assertEqual(pending_h1_part_ids(baseline), ["P0001"])

    def test_registered_actual_part_render_score_prefers_matching_color(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = np.zeros((64, 64, 3), dtype=np.uint8)
            reference[16:48, 20:44] = (45, 150, 35)
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[16:48, 20:44] = 255
            bad = np.zeros((64, 64, 3), dtype=np.uint8)
            bad[16:48, 20:44] = (125, 125, 125)
            good = np.zeros((64, 64, 3), dtype=np.uint8)
            good[16:48, 20:44] = (47, 147, 37)
            reference_path = root / "reference.png"
            mask_path = root / "mask.png"
            bad_path = root / "bad.png"
            good_path = root / "good.png"
            Image.fromarray(reference, mode="RGB").save(reference_path)
            Image.fromarray(mask, mode="L").save(mask_path)
            Image.fromarray(bad, mode="RGB").save(bad_path)
            Image.fromarray(good, mode="RGB").save(good_path)
            evidence = {
                "parts": [
                    {
                        "part_id": "P0001",
                        "observations": [
                            {
                                "view_id": "front",
                                "render_view_id": "front",
                                "image": str(reference_path),
                                "mask": str(mask_path),
                                "selected_for_material_inference": True,
                            }
                        ],
                    }
                ]
            }
            spatial = {
                "view_alignments": [
                    {
                        "reference_view_id": "front",
                        "selected_render_view_id": "front",
                        "trusted": True,
                        "bbox_affine": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                        ],
                        "ecc_warp": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                        ],
                    }
                ]
            }

            def registry(path: Path) -> dict:
                return {
                    "render_set": {"views": [{"view_id": "front", "rgb": str(path)}]}
                }

            h0 = score_part_id_render(
                part_id="P0001",
                evidence=evidence,
                spatial_mapping_report=spatial,
                rendered_registry=registry(bad_path),
            )
            h1 = score_part_id_render(
                part_id="P0001",
                evidence=evidence,
                spatial_mapping_report=spatial,
                rendered_registry=registry(good_path),
            )
            self.assertGreater(
                h1["appearance_score"],
                h0["appearance_score"] + 0.1,
            )
            output, audit = select_parameter_tournament_winners(
                source_plan=source_plan(),
                baseline_scores={"P0001": h0},
                h1_scores={"P0001": h1},
                minimum_score_improvement=0.015,
            )
            self.assertEqual(audit["h1_winner_part_ids"], ["P0001"])
            self.assertEqual(
                output["assignments"][0]["parameters"],
                {"diffuse_tint": [0.2, 1.0, 0.1]},
            )
            self.assertEqual(
                output["assignments"][0]["provenance"]["mdl_parameter_candidates"][
                    "selected_candidate_id"
                ],
                "H1",
            )

    def test_registered_score_relocates_renamed_reference_by_sealed_hash(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = np.zeros((32, 32, 3), dtype=np.uint8)
            reference[8:24, 8:24] = (45, 150, 35)
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[8:24, 8:24] = 255
            renamed = root / "renamed.png"
            missing = root / "original.png"
            mask_path = root / "mask.png"
            Image.fromarray(reference, mode="RGB").save(renamed)
            Image.fromarray(mask, mode="L").save(mask_path)
            digest = hashlib.sha256(renamed.read_bytes()).hexdigest()
            evidence = {
                "parts": [
                    {
                        "part_id": "P0001",
                        "observations": [
                            {
                                "view_id": "front",
                                "render_view_id": "front",
                                "image": str(missing),
                                "image_sha256": digest,
                                "mask": str(mask_path),
                                "mask_sha256": hashlib.sha256(
                                    mask_path.read_bytes()
                                ).hexdigest(),
                                "selected_for_material_inference": True,
                            }
                        ],
                    }
                ]
            }
            spatial = {
                "view_alignments": [
                    {
                        "reference_view_id": "front",
                        "selected_render_view_id": "front",
                        "trusted": True,
                        "bbox_affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        "ecc_warp": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    }
                ]
            }
            score = score_part_id_render(
                part_id="P0001",
                evidence=evidence,
                spatial_mapping_report=spatial,
                rendered_registry={
                    "render_set": {"views": [{"view_id": "front", "rgb": str(renamed)}]}
                },
            )
            self.assertEqual(score["comparison_pixel_count"], 256)

    def test_h0_remains_locked_without_clear_improvement(self) -> None:
        score = {"appearance_score": 0.75}
        output, audit = select_parameter_tournament_winners(
            source_plan=source_plan(),
            baseline_scores={"P0001": score},
            h1_scores={"P0001": {"appearance_score": 0.755}},
            minimum_score_improvement=0.015,
        )
        self.assertEqual(audit["h1_winner_count"], 0)
        self.assertNotIn("parameters", output["assignments"][0])
        self.assertEqual(
            output["assignments"][0]["provenance"]["mdl_parameter_candidates"][
                "selected_candidate_id"
            ],
            "H0",
        )


if __name__ == "__main__":
    unittest.main()
