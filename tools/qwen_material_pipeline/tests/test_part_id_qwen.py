from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from qwen_material_pipeline.workflows.part_id_qwen import (
    BATCH_SCHEMA_VERSION,
    MATERIAL_FAMILY_PREDICTION_BATCH_SCHEMA_VERSION,
    MATERIAL_IDENTITY_SELECTION_BATCH_SCHEMA_VERSION,
    PartIdQwenError,
    _apply_component_identity_consensus,
    _apply_part_id_selective_regression,
    _compatibility_shortlist,
    _family_filtered_ranking,
    _identity_filtered_ranking,
    _identity_shortlist,
    _promote_library_gap_candidates,
    _target_appearance,
    _validate_batch,
    run_part_id_qwen_rerank,
)


class _Generation:
    def __init__(self, text: str) -> None:
        self.text = text

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": "local-qwen-generation/v1",
            "generated_tokens": 32,
            "max_new_tokens": 128,
            "hit_token_limit": False,
            "eos_detected": True,
            "truncated": False,
            "finish_reason": "eos",
        }


class _Runner:
    model_identity = {"backend": "fake", "fingerprint": "unit-test"}

    def generate_with_metadata(self, _payload: object) -> _Generation:
        return _Generation(
            json.dumps(
                {
                    "schema_version": BATCH_SCHEMA_VERSION,
                    "selections": [
                        {
                            "part_id": "P0001",
                            "candidate_index": 1,
                            "confidence": 0.91,
                        },
                        {
                            "part_id": "P0002",
                            "candidate_index": 1,
                            "confidence": 0.89,
                        },
                    ],
                }
            )
        )


class _MaterialFamilyFirstRunner:
    model_identity = {"backend": "fake", "fingerprint": "family-first-test"}

    def __init__(self) -> None:
        self.calls = 0
        self.payloads: list[object] = []

    def generate_with_metadata(self, payload: object) -> _Generation:
        self.calls += 1
        self.payloads.append(payload)
        if self.calls == 1:
            return _Generation(
                json.dumps(
                    {
                        "schema_version": (
                            MATERIAL_FAMILY_PREDICTION_BATCH_SCHEMA_VERSION
                        ),
                        "predictions": [
                            {
                                "part_id": "P0001",
                                "physical_substrate": "metal",
                                "surface_treatment": "paint",
                                "optical_behavior": "opaque",
                                "surface_finish": "matte",
                                "substrate_confidence": 0.93,
                                "treatment_confidence": 0.92,
                            }
                        ],
                    }
                )
            )
        return _Generation(
            json.dumps(
                {
                    "schema_version": (
                        MATERIAL_IDENTITY_SELECTION_BATCH_SCHEMA_VERSION
                    ),
                    "selections": [
                        {
                            "part_id": "P0001",
                            "candidate_index": 1,
                            "match_type": "EXACT_LIBRARY_MATCH",
                            "confidence": 0.88,
                        }
                    ],
                }
            )
        )


class _ComponentIdentityRunner:
    model_identity = {"backend": "fake", "fingerprint": "component-identity-test"}

    def __init__(self) -> None:
        self.calls = 0

    def generate_with_metadata(self, _payload: object) -> _Generation:
        self.calls += 1
        if self.calls == 1:
            return _Generation(
                json.dumps(
                    {
                        "schema_version": (
                            MATERIAL_FAMILY_PREDICTION_BATCH_SCHEMA_VERSION
                        ),
                        "predictions": [
                            {
                                "part_id": "AC_1",
                                "physical_substrate": "metal",
                                "surface_treatment": "paint",
                                "optical_behavior": "opaque",
                                "surface_finish": "matte",
                                "substrate_confidence": 0.94,
                                "treatment_confidence": 0.92,
                            }
                        ],
                    }
                )
            )
        return _Generation(
            json.dumps(
                {
                    "schema_version": (
                        MATERIAL_IDENTITY_SELECTION_BATCH_SCHEMA_VERSION
                    ),
                    "selections": [
                        {
                            "part_id": "P0001",
                            "candidate_index": 1,
                            "match_type": "CORRESPONDING_MATERIAL",
                            "confidence": 0.7,
                        },
                        {
                            "part_id": "P0002",
                            "candidate_index": 2,
                            "match_type": "EXACT_LIBRARY_MATCH",
                            "confidence": 0.9,
                        },
                    ],
                }
            )
        )


class PartIdQwenTests(unittest.TestCase):
    @staticmethod
    def _surface_semantics(
        substrate: str,
        treatment: str = "bare",
        optical: str = "opaque",
        finish: str = "unknown",
    ) -> dict[str, object]:
        return {
            "compatible_substrates": [substrate],
            "surface_treatment": treatment,
            "optical_behavior": optical,
            "finish": finish,
            "confidence": "high",
        }

    def test_material_family_first_predicts_then_selects_only_same_family(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            crop = root / "P0001.png"
            Image.fromarray(np.full((24, 24, 3), (32, 110, 48), dtype=np.uint8)).save(
                crop
            )
            second_crop = root / "P0001_iso.png"
            Image.fromarray(np.full((24, 24, 3), (42, 105, 55), dtype=np.uint8)).save(
                second_crop
            )
            paint_matte = "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte"
            paint_gloss = "mdl:Miscellaneous/Paint_Gloss.mdl#Paint_Gloss"
            green_glass = "mdl:Glass/Green_Glass.mdl#Green_Glass"
            catalog = {
                "materials": [
                    {
                        "material_id": paint_matte,
                        "display_name": "Paint Matte",
                        "family": "paint",
                        "finishes": ["matte"],
                        "surface_semantics": {
                            "compatible_substrates": ["metal", "polymer", "wood"],
                            "surface_treatment": "paint",
                            "optical_behavior": "opaque",
                            "finish": "matte",
                            "confidence": "high",
                        },
                    },
                    {
                        "material_id": paint_gloss,
                        "display_name": "Paint Gloss",
                        "family": "paint",
                        "finishes": ["glossy"],
                        "surface_semantics": {
                            "compatible_substrates": ["metal", "polymer", "wood"],
                            "surface_treatment": "paint",
                            "optical_behavior": "opaque",
                            "finish": "glossy",
                            "confidence": "high",
                        },
                    },
                    {
                        "material_id": green_glass,
                        "display_name": "Green Glass",
                        "family": "glass",
                        "finishes": ["smooth"],
                        "surface_semantics": {
                            "compatible_substrates": ["glass"],
                            "surface_treatment": "bare",
                            "optical_behavior": "transparent",
                            "finish": "smooth",
                            "confidence": "high",
                        },
                    },
                ]
            }
            evidence = {
                "schema_version": "qwen-part-id-reference-evidence/v1",
                "integrity": {"document_sha256": "evidence"},
                "parts": [
                    {
                        "part_id": "P0001",
                        "status": "observed",
                        "descriptor": {"surface_class": "dielectric"},
                        "observations": [
                            {
                                "view_id": "front",
                                "crop": str(crop),
                                "selected_for_material_inference": True,
                            },
                            {
                                "view_id": "iso",
                                "crop": str(second_crop),
                                "selected_for_material_inference": False,
                            },
                        ],
                    }
                ],
            }
            retrieval = {
                "groups": [
                    {
                        "group_id": "P0001",
                        "fused_ranking": [
                            {"rank": 1, "material_id": green_glass},
                            {"rank": 2, "material_id": paint_gloss},
                            {"rank": 3, "material_id": paint_matte},
                        ],
                    }
                ]
            }
            runner = _MaterialFamilyFirstRunner()
            result = run_part_id_qwen_rerank(
                evidence=evidence,
                retrieval=retrieval,
                catalog=catalog,
                runner=runner,
                model="fake",
                output_dir=root / "qwen",
                batch_size=1,
                candidate_count=2,
                require_material_family_prediction=True,
            )

        self.assertEqual(runner.calls, 2)
        prediction_content = runner.payloads[0]["messages"][1]["content"]
        self.assertEqual(
            sum(row.get("type") == "image_url" for row in prediction_content),
            2,
        )
        selection_prompt = runner.payloads[1]["messages"][1]["content"][-1]["text"]
        self.assertIn('"visual_retrieval_scores_withheld": true', selection_prompt)
        self.assertIn('"original_retrieval_rank": null', selection_prompt)
        self.assertIn('"color_score": null', selection_prompt)
        self.assertEqual(result["material_prediction_mode"], "catalog_family_first")
        self.assertEqual(
            result["selection_order"],
            [
                "physical_material_identity_prediction_without_color",
                "exact_substrate_treatment_optical_filter",
                "exact_material_or_corresponding_material_selection_without_color",
                "appearance_component_exact_mdl_consensus",
            ],
        )
        self.assertEqual(result["material_predictions"][0]["catalog_family"], "paint")
        self.assertEqual(result["choices"], {"P0001": paint_matte})
        self.assertEqual(result["selections"][0]["confidence"], 0.88)
        self.assertEqual(
            result["visual_compatibility_gate"]["parts"][0]["authorized_material_ids"],
            [paint_matte, paint_gloss],
        )
        self.assertNotIn(
            green_glass,
            result["visual_compatibility_gate"]["parts"][0]["authorized_material_ids"],
        )
        self.assertEqual(result["summary"]["physical_cross_family_fallback_count"], 0)

    def test_material_identity_filter_requires_complete_catalog_coverage(
        self,
    ) -> None:
        paint_matte = "mdl:Paint_Matte.mdl#Paint_Matte"
        paint_gloss = "mdl:Paint_Gloss.mdl#Paint_Gloss"
        with self.assertRaisesRegex(PartIdQwenError, "complete catalog"):
            _family_filtered_ranking(
                ranking=[{"rank": 1, "material_id": paint_matte}],
                catalog_by_id={
                    paint_matte: {
                        "material_id": paint_matte,
                        "family": "paint",
                        "surface_semantics": {
                            "compatible_substrates": ["metal"],
                            "surface_treatment": "paint",
                            "optical_behavior": "opaque",
                            "finish": "matte",
                            "confidence": "high",
                        },
                    },
                    paint_gloss: {
                        "material_id": paint_gloss,
                        "family": "paint",
                        "surface_semantics": {
                            "compatible_substrates": ["metal"],
                            "surface_treatment": "paint",
                            "optical_behavior": "opaque",
                            "finish": "glossy",
                            "confidence": "high",
                        },
                    },
                },
                prediction={
                    "catalog_family": "paint",
                    "physical_substrate": "metal",
                    "surface_treatment": "paint",
                    "optical_behavior": "opaque",
                    "surface_finish": "matte",
                    "confidence": 0.9,
                    "substrate_confidence": 0.9,
                    "treatment_confidence": 0.9,
                    "identity_resolution": "exact_material",
                    "status": "APPLYABLE",
                },
            )

    def test_identity_filter_rejects_rubber_and_veneer_misfiled_as_plastic(
        self,
    ) -> None:
        plastic = "mdl:Plastic.mdl#Plastic"
        abs_plastic = "mdl:Plastic_ABS.mdl#Plastic_ABS"
        rubber = "mdl:Rubber_Textured.mdl#Rubber_Textured"
        veneer = "mdl:Veneer.mdl#Veneer_OU_Walnut"
        catalog = {
            plastic: {
                "material_id": plastic,
                "family": "plastic",
                "surface_semantics": self._surface_semantics("polymer"),
            },
            abs_plastic: {
                "material_id": abs_plastic,
                "family": "plastic",
                "surface_semantics": self._surface_semantics("polymer"),
            },
            rubber: {
                "material_id": rubber,
                "family": "plastic",
                "surface_semantics": self._surface_semantics("elastomer"),
            },
            veneer: {
                "material_id": veneer,
                "family": "plastic",
                "surface_semantics": self._surface_semantics("wood"),
            },
        }
        ranking = [
            {"rank": index, "material_id": material_id}
            for index, material_id in enumerate(
                (rubber, veneer, plastic, abs_plastic), start=1
            )
        ]

        filtered = _identity_filtered_ranking(
            ranking=ranking,
            catalog_by_id=catalog,
            prediction={
                "part_id": "P0001",
                "catalog_family": "plastic",
                "physical_substrate": "polymer",
                "surface_treatment": "bare",
                "optical_behavior": "opaque",
                "surface_finish": "unknown",
                "confidence": 0.9,
                "substrate_confidence": 0.9,
                "treatment_confidence": 0.9,
                "identity_resolution": "exact_material",
                "status": "APPLYABLE",
            },
        )

        self.assertEqual(
            {row["material_id"] for row in filtered},
            {plastic, abs_plastic},
        )

    def test_identity_shortlist_ignores_rgb_rank_and_samples_treatments(
        self,
    ) -> None:
        rows = [
            {
                "material_id": "mdl:Z_Bare.mdl#Z_Bare",
                "rank": 99,
                "color_score": 0.01,
                "catalog_surface_semantics": {"surface_treatment": "bare"},
                "predicted_finish_match": False,
                "identity_match_tier": "corresponding_material_fallback",
            },
            {
                "material_id": "mdl:A_Paint.mdl#A_Paint",
                "rank": 1,
                "color_score": 0.99,
                "catalog_surface_semantics": {"surface_treatment": "paint"},
                "predicted_finish_match": False,
                "identity_match_tier": "corresponding_material_fallback",
            },
            {
                "material_id": "mdl:B_Bare.mdl#B_Bare",
                "rank": 2,
                "color_score": 0.98,
                "catalog_surface_semantics": {"surface_treatment": "bare"},
                "predicted_finish_match": False,
                "identity_match_tier": "corresponding_material_fallback",
            },
        ]

        shortlist = _identity_shortlist(rows, candidate_count=2)

        self.assertEqual(
            [row["material_id"] for row in shortlist],
            ["mdl:B_Bare.mdl#B_Bare", "mdl:A_Paint.mdl#A_Paint"],
        )
        self.assertEqual(
            [row["original_retrieval_rank"] for row in shortlist], [None, None]
        )
        self.assertTrue(all(row["color_evidence_used"] is False for row in shortlist))

    def test_identity_filter_collapses_color_variants_to_generic_material(
        self,
    ) -> None:
        generic = "mdl:Aluminum_Anodized.mdl#Aluminum_Anodized"
        black = "mdl:Aluminum_Anodized_Black.mdl#Aluminum_Anodized_Black"
        blue = "mdl:Aluminum_Anodized_Blue.mdl#Aluminum_Anodized_Blue"
        bare = "mdl:Iron.mdl#Iron"
        anodized = self._surface_semantics("metal", "anodized")
        catalog = {
            generic: {"material_id": generic, "surface_semantics": anodized},
            black: {"material_id": black, "surface_semantics": anodized},
            blue: {"material_id": blue, "surface_semantics": anodized},
            bare: {
                "material_id": bare,
                "surface_semantics": self._surface_semantics("metal"),
            },
        }
        filtered = _identity_filtered_ranking(
            ranking=[
                {"rank": 1, "material_id": black},
                {"rank": 2, "material_id": blue},
                {"rank": 3, "material_id": generic},
                {"rank": 4, "material_id": bare},
            ],
            catalog_by_id=catalog,
            prediction={
                "part_id": "P0001",
                "catalog_family": "metal",
                "physical_substrate": "metal",
                "surface_treatment": "anodized",
                "optical_behavior": "opaque",
                "surface_finish": "unknown",
                "confidence": 0.9,
                "substrate_confidence": 0.9,
                "treatment_confidence": 0.9,
                "identity_resolution": "exact_material",
                "status": "APPLYABLE",
            },
        )

        self.assertEqual([row["material_id"] for row in filtered], [generic])

    def test_component_consensus_enforces_one_exact_mdl(self) -> None:
        selections, audit = _apply_component_identity_consensus(
            selections=[
                {"part_id": "P1", "material_id": "mdl:A", "confidence": 0.7},
                {"part_id": "P2", "material_id": "mdl:B", "confidence": 0.9},
                {"part_id": "P3", "material_id": "mdl:A", "confidence": 0.6},
            ],
            component_members={"AC_1": ["P1", "P2", "P3"]},
        )

        self.assertEqual({row["material_id"] for row in selections}, {"mdl:A"})
        self.assertEqual(audit["summary"]["component_count"], 1)
        self.assertTrue(audit["summary"]["all_components_share_one_exact_mdl"])

    def test_component_is_predicted_once_and_receives_one_exact_mdl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            part_rows = []
            retrieval_groups = []
            matte = "mdl:Paint_Matte.mdl#Paint_Matte"
            satin = "mdl:Paint_Satin.mdl#Paint_Satin"
            rubber = "mdl:Rubber_Textured.mdl#Rubber_Textured"
            for index, part_id in enumerate(("P0001", "P0002"), start=1):
                crop = root / f"{part_id}.png"
                Image.fromarray(
                    np.full((20, 20, 3), (20 * index, 100, 40), dtype=np.uint8)
                ).save(crop)
                part_rows.append(
                    {
                        "part_id": part_id,
                        "status": "observed",
                        "descriptor": {"surface_class": "dielectric"},
                        "observations": [
                            {
                                "view_id": "front",
                                "crop": str(crop),
                                "selected_for_material_inference": True,
                            }
                        ],
                    }
                )
                retrieval_groups.append(
                    {
                        "group_id": part_id,
                        "fused_ranking": [
                            {"rank": 1, "material_id": matte},
                            {"rank": 2, "material_id": satin},
                            {"rank": 3, "material_id": rubber},
                        ],
                    }
                )
            paint_semantics = {
                "compatible_substrates": ["metal", "polymer", "wood"],
                "surface_treatment": "paint",
                "optical_behavior": "opaque",
                "finish": "matte",
                "confidence": "high",
            }
            runner = _ComponentIdentityRunner()
            result = run_part_id_qwen_rerank(
                evidence={
                    "schema_version": "qwen-part-id-reference-evidence/v1",
                    "integrity": {"document_sha256": "evidence"},
                    "parts": part_rows,
                },
                retrieval={"groups": retrieval_groups},
                catalog={
                    "materials": [
                        {
                            "material_id": matte,
                            "family": "paint",
                            "finishes": ["matte"],
                            "surface_semantics": paint_semantics,
                        },
                        {
                            "material_id": satin,
                            "family": "paint",
                            "finishes": ["satin"],
                            "surface_semantics": {
                                **paint_semantics,
                                "finish": "satin",
                            },
                        },
                        {
                            "material_id": rubber,
                            "family": "plastic",
                            "surface_semantics": self._surface_semantics(
                                "elastomer"
                            ),
                        },
                    ]
                },
                appearance_components={
                    "components": [
                        {
                            "component_id": "AC_1",
                            "member_part_ids": ["P0001", "P0002"],
                        }
                    ]
                },
                runner=runner,
                model="fake",
                output_dir=root / "qwen",
                batch_size=2,
                candidate_count=2,
                require_material_family_prediction=True,
            )

            grayscale = np.asarray(
                Image.open(
                    root
                    / "qwen"
                    / "identity_evidence_grayscale"
                    / "P0001_front_01.png"
                )
            )

        self.assertEqual(runner.calls, 2)
        self.assertEqual(set(result["choices"].values()), {satin})
        self.assertEqual(
            {row["component_id"] for row in result["material_predictions"]},
            {"AC_1"},
        )
        self.assertEqual(
            result["component_identity_consensus"]["summary"][
                "constrained_part_count"
            ],
            2,
        )
        self.assertTrue(np.array_equal(grayscale[:, :, 0], grayscale[:, :, 1]))
        self.assertTrue(np.array_equal(grayscale[:, :, 1], grayscale[:, :, 2]))

    def test_tiny_chromatic_part_keeps_color_authority_in_selection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.zeros((16, 16, 3), dtype=np.uint8)
            mask = np.zeros((16, 16), dtype=np.uint8)
            image[6, 4:10] = (40, 160, 220)
            mask[6, 4:10] = 255
            image_path = root / "image.png"
            mask_path = root / "mask.png"
            Image.fromarray(image).save(image_path)
            Image.fromarray(mask).save(mask_path)
            target = _target_appearance(
                {
                    "image": str(image_path),
                    "mask": str(mask_path),
                    "chromatic_coverage": {
                        "applied": True,
                        "tiny_part_rescue": True,
                    },
                }
            )

        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target["trusted_pixels"], 6)
        self.assertTrue(target["chromatic_component_authoritative"])
        self.assertTrue(target["tiny_chromatic_rescue"])

        clear = "mdl:Plastics/Plastic_Clear.mdl#Plastic_Clear"
        vinyl = "mdl:Plastics/Vinyl.mdl#Vinyl"
        rows = _compatibility_shortlist(
            ranking=[
                {"rank": 1, "material_id": clear},
                {"rank": 2, "material_id": vinyl},
            ],
            catalog_by_id={
                clear: {"material_id": clear, "family": "plastic"},
                vinyl: {"material_id": vinyl, "family": "plastic"},
            },
            profiles_by_id={
                clear: {
                    "appearance": {
                        "neutral_iso": {
                            "median_rgb": [0.9, 0.9, 0.9],
                            "texture_gradient_energy": 0.0,
                        }
                    }
                },
                vinyl: {
                    "appearance": {
                        "neutral_iso": {
                            "median_rgb": [0.4, 0.4, 0.4],
                            "texture_gradient_energy": 0.0,
                        }
                    }
                },
            },
            target=target,
            candidate_count=2,
            allow_color_tuning=True,
        )
        by_id = {row["material_id"]: row for row in rows}
        self.assertFalse(by_id[clear]["selection_allowed"])
        self.assertTrue(by_id[vinyl]["selection_allowed"])
        self.assertTrue(by_id[vinyl]["color_tunable"])

    def test_batch_normalizes_bounded_decimal_candidate_index_string(self) -> None:
        expected = [
            {
                "part_id": "P0001",
                "candidates": [
                    {
                        "candidate_index": 1,
                        "material_id": "mdl:Metal.mdl#Metal",
                        "selection_allowed": True,
                    }
                ],
            }
        ]

        selections = _validate_batch(
            {
                "schema_version": BATCH_SCHEMA_VERSION,
                "selections": [
                    {
                        "part_id": "P0001",
                        "candidate_index": "1",
                        "confidence": 0.9,
                    }
                ],
            },
            expected=expected,
        )

        self.assertEqual(selections[0]["candidate_index"], 1)
        self.assertEqual(
            selections[0]["material_id"],
            "mdl:Metal.mdl#Metal",
        )
        self.assertEqual(selections[0]["index_resolution"], "exact")

    def test_batch_bounds_out_of_range_index_to_top_allowed_candidate(self) -> None:
        expected = [
            {
                "part_id": "P0001",
                "candidates": [
                    {
                        "candidate_index": 1,
                        "material_id": "mdl:Top.mdl#Top",
                        "selection_allowed": True,
                    },
                    {
                        "candidate_index": 2,
                        "material_id": "mdl:Second.mdl#Second",
                        "selection_allowed": True,
                    },
                ],
            }
        ]

        selections = _validate_batch(
            {
                "schema_version": BATCH_SCHEMA_VERSION,
                "selections": [
                    {
                        "part_id": "P0001",
                        "candidate_index": 7,
                        "confidence": 0.8,
                    }
                ],
            },
            expected=expected,
        )

        self.assertEqual(selections[0]["candidate_index"], 1)
        self.assertEqual(selections[0]["requested_candidate_index"], 7)
        self.assertEqual(
            selections[0]["index_resolution"],
            "bounded_top_candidate_fallback",
        )
        self.assertEqual(selections[0]["material_id"], "mdl:Top.mdl#Top")

    def test_selective_regression_is_fresh_per_part_and_color_first(self) -> None:
        jobs = [
            {
                "part_id": "P_GREEN",
                "target_appearance": {
                    "trusted_pixels": 5000,
                    "median_rgb": [0.15, 0.38, 0.18],
                },
                "candidates": [
                    {
                        "material_id": "mdl:Gray.mdl#Gray",
                        "compatibility_rank": 1,
                        "visual_compatibility_score": 0.83,
                        "color_similarity": 0.68,
                        "texture_similarity": 0.94,
                        "selection_allowed": True,
                    },
                    {
                        "material_id": "mdl:Green.mdl#Green",
                        "compatibility_rank": 2,
                        "visual_compatibility_score": 0.78,
                        "color_similarity": 0.86,
                        "texture_similarity": 0.68,
                        "selection_allowed": True,
                    },
                ],
            },
            {
                "part_id": "P_SMALL",
                "target_appearance": {
                    "trusted_pixels": 100,
                    "median_rgb": [0.15, 0.38, 0.18],
                },
                "candidates": [
                    {
                        "material_id": "mdl:Stable.mdl#Stable",
                        "compatibility_rank": 1,
                        "visual_compatibility_score": 0.80,
                        "color_similarity": 0.70,
                        "texture_similarity": 0.90,
                        "selection_allowed": True,
                    },
                    {
                        "material_id": "mdl:Risky.mdl#Risky",
                        "compatibility_rank": 2,
                        "visual_compatibility_score": 0.60,
                        "color_similarity": 0.95,
                        "texture_similarity": 0.20,
                        "selection_allowed": True,
                    },
                ],
            },
            {
                "part_id": "P_CHROMATIC",
                "target_appearance": {
                    "trusted_pixels": 8,
                    "median_rgb": [0.18, 0.48, 0.72],
                    "chromatic_component_authoritative": True,
                },
                "candidates": [
                    {
                        "material_id": "mdl:CurrentBest.mdl#CurrentBest",
                        "compatibility_rank": 1,
                        "visual_compatibility_score": 0.70,
                        "color_similarity": 1.0,
                        "texture_similarity": 0.80,
                        "selection_allowed": True,
                    },
                    {
                        "material_id": "mdl:QwenNear.mdl#QwenNear",
                        "compatibility_rank": 2,
                        "visual_compatibility_score": 0.69,
                        "color_similarity": 1.0,
                        "texture_similarity": 0.75,
                        "selection_allowed": True,
                    },
                ],
            },
        ]
        selections, audit = _apply_part_id_selective_regression(
            jobs=jobs,
            qwen_selections=[
                {
                    "part_id": "P_GREEN",
                    "material_id": "mdl:Gray.mdl#Gray",
                    "confidence": 0.9,
                },
                {
                    "part_id": "P_SMALL",
                    "material_id": "mdl:Risky.mdl#Risky",
                    "confidence": 0.8,
                },
                {
                    "part_id": "P_CHROMATIC",
                    "material_id": "mdl:QwenNear.mdl#QwenNear",
                    "confidence": 0.8,
                },
            ],
        )
        self.assertEqual(
            {row["part_id"]: row["material_id"] for row in selections},
            {
                "P_GREEN": "mdl:Green.mdl#Green",
                "P_SMALL": "mdl:Stable.mdl#Stable",
                "P_CHROMATIC": "mdl:CurrentBest.mdl#CurrentBest",
            },
        )
        self.assertFalse(audit["previous_material_plan_consulted"])
        self.assertEqual(
            audit["summary"]["fresh_local_baseline_selected_count"], 3
        )
        chromatic_audit = next(
            row
            for row in audit["parts"]
            if row["part_id"] == "P_CHROMATIC"
        )
        self.assertTrue(chromatic_audit["strict_color_nonregression"])
        self.assertEqual(chromatic_audit["maximum_local_score_regression"], 0.0)

    def test_compatibility_gate_penalizes_transmission_and_texture_mismatch(
        self,
    ) -> None:
        ranking = [
            {"rank": 1, "material_id": "mdl:Glass.mdl#Glass"},
            {"rank": 2, "material_id": "mdl:Grass.mdl#Grass"},
            {"rank": 3, "material_id": "mdl:Paint.mdl#Paint"},
        ]
        catalog = {
            "mdl:Glass.mdl#Glass": {
                "material_id": "mdl:Glass.mdl#Glass",
                "family": "glass",
            },
            "mdl:Grass.mdl#Grass": {
                "material_id": "mdl:Grass.mdl#Grass",
                "family": "natural",
            },
            "mdl:Paint.mdl#Paint": {
                "material_id": "mdl:Paint.mdl#Paint",
                "family": "paint",
            },
        }

        def profile(texture: float) -> dict[str, object]:
            return {
                "appearance": {
                    "neutral_iso": {
                        "median_rgb": [0.15, 0.35, 0.17],
                        "texture_gradient_energy": texture,
                    }
                }
            }

        rows = _compatibility_shortlist(
            ranking=ranking,
            catalog_by_id=catalog,
            profiles_by_id={
                "mdl:Glass.mdl#Glass": profile(0.002),
                "mdl:Grass.mdl#Grass": profile(0.04),
                "mdl:Paint.mdl#Paint": profile(0.003),
            },
            target={
                "median_rgb": [0.15, 0.35, 0.17],
                "texture_gradient_energy": 0.002,
                "likely_opaque": True,
                "likely_smooth": True,
            },
            candidate_count=3,
        )
        self.assertEqual(rows[0]["material_id"], "mdl:Paint.mdl#Paint")
        self.assertTrue(
            next(
                row
                for row in rows
                if row["material_id"] == "mdl:Glass.mdl#Glass"
            )["transmission_risk"]
        )
        self.assertFalse(
            next(
                row
                for row in rows
                if row["material_id"] == "mdl:Glass.mdl#Glass"
            )["selection_allowed"]
        )
        self.assertTrue(
            next(
                row
                for row in rows
                if row["material_id"] == "mdl:Grass.mdl#Grass"
            )["texture_mismatch_risk"]
        )

    def test_chromatic_gate_rejects_gray_default_but_accepts_color_tunable_mdl(
        self,
    ) -> None:
        vinyl = "mdl:Plastics/Vinyl.mdl#Vinyl"
        ranking = [
            {"rank": 1, "material_id": vinyl},
            {"rank": 2, "material_id": "mdl:Natural/Grass.mdl#Grass"},
        ]
        catalog = {
            vinyl: {"material_id": vinyl, "family": "plastic"},
            "mdl:Natural/Grass.mdl#Grass": {
                "material_id": "mdl:Natural/Grass.mdl#Grass",
                "family": "natural",
            },
        }
        profiles = {
            vinyl: {
                "appearance": {
                    "neutral_iso": {
                        "median_rgb": [0.396, 0.396, 0.396],
                        "texture_gradient_energy": 0.004,
                    }
                }
            },
            "mdl:Natural/Grass.mdl#Grass": {
                "appearance": {
                    "neutral_iso": {
                        "median_rgb": [0.23, 0.31, 0.14],
                        "texture_gradient_energy": 0.03,
                    }
                }
            },
        }
        target = {
            "trusted_pixels": 33000,
            "median_rgb": [0.204, 0.459, 0.247],
            "texture_gradient_energy": 0.004,
            "likely_opaque": True,
            "likely_smooth": True,
            "likely_unpatterned": True,
        }
        immutable_rows = _compatibility_shortlist(
            ranking=ranking,
            catalog_by_id=catalog,
            profiles_by_id=profiles,
            target=target,
            candidate_count=2,
        )
        immutable_vinyl = next(
            row for row in immutable_rows if row["material_id"] == vinyl
        )
        self.assertFalse(immutable_vinyl["color_gate_passed"])
        self.assertFalse(immutable_vinyl["selection_allowed"])
        self.assertLess(immutable_vinyl["color_similarity"], 0.5)

        mutable_rows = _compatibility_shortlist(
            ranking=ranking,
            catalog_by_id=catalog,
            profiles_by_id=profiles,
            target=target,
            candidate_count=2,
            allow_color_tuning=True,
        )
        mutable_vinyl = next(
            row for row in mutable_rows if row["material_id"] == vinyl
        )
        self.assertTrue(mutable_vinyl["color_tunable"])
        self.assertTrue(mutable_vinyl["color_gate_passed"])
        self.assertTrue(mutable_vinyl["selection_allowed"])
        self.assertEqual(mutable_vinyl["color_similarity"], 1.0)

    def test_library_gap_promotes_only_safe_parameter_candidates_first(
        self,
    ) -> None:
        rows, audit = _promote_library_gap_candidates(
            [
                {
                    "material_id": "mdl:Glass/Green_Glass.mdl#Green_Glass",
                    "selection_allowed": False,
                    "transmission_risk": True,
                    "texture_mismatch_risk": False,
                },
                {
                    "material_id": "mdl:Metals/Chrome.mdl#Chrome",
                    "selection_allowed": False,
                    "transmission_risk": False,
                    "texture_mismatch_risk": False,
                },
                {
                    "material_id": "mdl:Plastics/Vinyl.mdl#Vinyl",
                    "selection_allowed": False,
                    "transmission_risk": False,
                    "texture_mismatch_risk": False,
                },
                {
                    "material_id": "mdl:Natural/Grass_Cut.mdl#Grass_Cut",
                    "selection_allowed": False,
                    "transmission_risk": False,
                    "texture_mismatch_risk": True,
                },
            ],
            candidate_count=4,
        )

        promoted = [row for row in rows if row["selection_allowed"]]
        self.assertEqual(
            [row["material_id"] for row in promoted],
            [
                "mdl:Metals/Chrome.mdl#Chrome",
                "mdl:Plastics/Vinyl.mdl#Vinyl",
            ],
        )
        self.assertTrue(
            all(row["conditional_h1_evaluation"] for row in promoted)
        )
        self.assertTrue(
            all(
                row["relaxed_constraints"] == ["default_color_gate"]
                for row in promoted
            )
        )
        self.assertEqual(
            audit["status"],
            "LIBRARY_GAP_BOUNDED_FALLBACK",
        )
        self.assertFalse(audit["parameter_write_authorized"])

    def test_immutable_chromatic_library_gap_keeps_fixed_color_candidate(self) -> None:
        rows, audit = _promote_library_gap_candidates(
            [
                {
                    "material_id": "mdl:Glass/Green_Glass.mdl#Green_Glass",
                    "selection_allowed": False,
                    "color_gate_passed": True,
                    "visual_compatibility_score": 0.71,
                    "color_delta_e": 8.0,
                    "original_retrieval_rank": 1,
                    "transmission_risk": True,
                    "texture_mismatch_risk": False,
                },
                {
                    "material_id": "mdl:Metals/Steel_Cast.mdl#Steel_Cast",
                    "selection_allowed": False,
                    "color_gate_passed": False,
                    "visual_compatibility_score": 0.32,
                    "color_delta_e": 24.0,
                    "original_retrieval_rank": 2,
                    "transmission_risk": False,
                    "texture_mismatch_risk": False,
                },
            ],
            candidate_count=2,
            target={
                "median_rgb": [0.20, 0.46, 0.24],
                "trusted_pixels": 4096,
            },
            allow_color_tuning=False,
        )

        promoted = [row for row in rows if row["selection_allowed"]]
        self.assertEqual(
            [row["material_id"] for row in promoted],
            ["mdl:Glass/Green_Glass.mdl#Green_Glass"],
        )
        self.assertEqual(
            promoted[0]["library_gap_fallback_tier"],
            "immutable_chromatic_visual_priority",
        )
        self.assertEqual(promoted[0]["relaxed_constraints"], ["opacity_gate"])
        self.assertFalse(promoted[0]["conditional_h1_evaluation"])
        self.assertEqual(
            audit["policy"],
            "retrieved_nvidia_base_immutable_chromatic_visual_priority/v1",
        )
        self.assertFalse(audit["parameter_write_authorized"])

    def test_qwen_selects_each_part_independently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            crops: dict[str, Path] = {}
            for part_id, color in (
                ("P0001", (220, 30, 30)),
                ("P0002", (25, 55, 225)),
            ):
                crop = root / f"{part_id}.png"
                Image.fromarray(
                    np.full((16, 16, 3), color, dtype=np.uint8),
                    mode="RGB",
                ).save(crop)
                crops[part_id] = crop
            evidence = {
                "schema_version": "qwen-part-id-reference-evidence/v1",
                "integrity": {"document_sha256": "evidence"},
                "parts": [
                    {
                        "part_id": part_id,
                        "status": "observed",
                        "descriptor": {"base_color": base_color},
                        "observations": [
                            {
                                "view_id": "front",
                                "crop": str(crops[part_id]),
                                "selected_for_material_inference": True,
                            }
                        ],
                    }
                    for part_id, base_color in (
                        ("P0001", "red"),
                        ("P0002", "blue"),
                    )
                ],
            }
            retrieval = {
                "groups": [
                    {
                        "group_id": "P0001",
                        "fused_ranking": [
                            {
                                "rank": 1,
                                "material_id": "mdl:Red.mdl#Red",
                                "siglip2_score": 0.8,
                            },
                            {
                                "rank": 2,
                                "material_id": "mdl:Alt.mdl#Alt",
                                "siglip2_score": 0.7,
                            },
                        ],
                    },
                    {
                        "group_id": "P0002",
                        "fused_ranking": [
                            {
                                "rank": 1,
                                "material_id": "mdl:Blue.mdl#Blue",
                                "siglip2_score": 0.81,
                            },
                            {
                                "rank": 2,
                                "material_id": "mdl:Alt.mdl#Alt",
                                "siglip2_score": 0.69,
                            },
                        ],
                    },
                ]
            }
            catalog = {
                "materials": [
                    {
                        "material_id": material_id,
                        "display_name": name,
                        "family": "test",
                    }
                    for material_id, name in (
                        ("mdl:Red.mdl#Red", "Red"),
                        ("mdl:Blue.mdl#Blue", "Blue"),
                        ("mdl:Alt.mdl#Alt", "Alternative"),
                    )
                ]
            }
            result = run_part_id_qwen_rerank(
                evidence=evidence,
                retrieval=retrieval,
                catalog=catalog,
                runner=_Runner(),
                model="fake",
                output_dir=root / "qwen",
                batch_size=2,
                candidate_count=2,
            )
            self.assertEqual(
                result["choices"],
                {
                    "P0001": "mdl:Red.mdl#Red",
                    "P0002": "mdl:Blue.mdl#Blue",
                },
            )
            self.assertEqual(result["assignment_unit"], "part_id")
            self.assertFalse(result["palette_fusion_used"])
            self.assertFalse(result["part_material_groups_used"])
            self.assertTrue(result["summary"]["exact_cover_of_observed_parts"])


if __name__ == "__main__":
    unittest.main()
