from __future__ import annotations

import json
import hashlib
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
    _apply_repeated_assembly_role_constraints,
    _compatibility_shortlist,
    _direct_exact_library_match,
    _family_filtered_ranking,
    _identity_filtered_ranking,
    _identity_shortlist,
    _physical_pbr_evidence,
    _promote_library_gap_candidates,
    _rank_identity_candidates_with_pbr,
    _refine_component_memberships_with_final_evidence,
    _shape_guided_exact_preset_eligible,
    _target_appearance,
    _validate_batch,
    _write_grayscale_identity_crop,
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
                                "material_species": "paint",
                                "surface_treatment": "paint",
                                "optical_behavior": "opaque",
                                "surface_finish": "matte",
                                "substrate_confidence": 0.93,
                                "species_confidence": 0.93,
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
                                "material_species": "paint",
                                "surface_treatment": "paint",
                                "optical_behavior": "opaque",
                                "surface_finish": "matte",
                                "substrate_confidence": 0.94,
                                "species_confidence": 0.91,
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


class _SpecificPresetRunner:
    model_identity = {"backend": "fake", "fingerprint": "specific-preset-test"}

    def __init__(self) -> None:
        self.calls = 0

    def generate_with_metadata(self, _payload: object) -> _Generation:
        self.calls += 1
        if self.calls == 1:
            return _Generation(
                json.dumps(
                    {
                        "schema_version": MATERIAL_FAMILY_PREDICTION_BATCH_SCHEMA_VERSION,
                        "predictions": [
                            {
                                "part_id": "P0001",
                                "physical_substrate": "metal",
                                "material_species": "unknown",
                                "surface_treatment": "unknown",
                                "optical_behavior": "opaque",
                                "surface_finish": "matte",
                                "substrate_confidence": 0.94,
                                "species_confidence": 0.25,
                                "treatment_confidence": 0.35,
                            }
                        ],
                    }
                )
            )
        return _Generation(
            json.dumps(
                {
                    "schema_version": MATERIAL_IDENTITY_SELECTION_BATCH_SCHEMA_VERSION,
                    "selections": [
                        {
                            "part_id": "P0001",
                            "candidate_index": 1,
                            "match_type": "EXACT_LIBRARY_MATCH",
                            "confidence": 0.92,
                        }
                    ],
                }
            )
        )


class _TwoStageCorrespondingRunner:
    model_identity = {"backend": "fake", "fingerprint": "two-stage-test"}

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
                        "schema_version": MATERIAL_FAMILY_PREDICTION_BATCH_SCHEMA_VERSION,
                        "predictions": [
                            {
                                "part_id": "P0001",
                                "physical_substrate": "metal",
                                "material_species": "unknown",
                                "surface_treatment": "unknown",
                                "optical_behavior": "opaque",
                                "surface_finish": "matte",
                                "substrate_confidence": 0.94,
                                "species_confidence": 0.25,
                                "treatment_confidence": 0.35,
                            }
                        ],
                    }
                )
            )
        if self.calls == 2:
            return _Generation(
                json.dumps(
                    {
                        "schema_version": MATERIAL_IDENTITY_SELECTION_BATCH_SCHEMA_VERSION,
                        "selections": [
                            {
                                "part_id": "P0001",
                                "candidate_index": 1,
                                "match_type": "CORRESPONDING_MATERIAL",
                                "confidence": 0.86,
                            }
                        ],
                    }
                )
            )
        return _Generation(
            json.dumps(
                {
                    "schema_version": BATCH_SCHEMA_VERSION,
                    "selections": [
                        {
                            "part_id": "P0001",
                            "candidate_index": 1,
                            "confidence": 0.82,
                        }
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

        self.assertEqual(runner.calls, 1)
        prediction_content = runner.payloads[0]["messages"][1]["content"]
        self.assertEqual(
            sum(row.get("type") == "image_url" for row in prediction_content),
            2,
        )
        self.assertEqual(result["material_prediction_mode"], "catalog_family_first")
        self.assertEqual(
            result["material_identity_evidence_mode"],
            "isolated_target_only",
        )
        self.assertEqual(
            result["selection_order"],
            [
                "physical_material_identity_prediction_without_color",
                "final_evidence_component_membership_refinement",
                "repeated_cad_assembly_role_consistency",
                "exact_substrate_species_treatment_optical_filter",
                "full_catalog_specific_preset_preservation",
                "exact_preset_confirmation_with_bounded_color_evidence",
                "independent_grayscale_corresponding_material_selection_when_unresolved",
                "component_and_repeated_role_exact_mdl_consensus",
            ],
        )
        self.assertEqual(result["material_predictions"][0]["catalog_family"], "paint")
        self.assertEqual(result["choices"], {"P0001": paint_matte})
        self.assertEqual(result["selections"][0]["confidence"], 0.92)
        self.assertEqual(
            result["selections"][0]["selection_authority"],
            "unique_full_catalog_physical_contract",
        )
        self.assertEqual(result["summary"]["direct_exact_library_assignment_count"], 1)
        self.assertEqual(
            result["visual_compatibility_gate"]["parts"][0]["authorized_material_ids"],
            [paint_matte, paint_gloss],
        )
        self.assertNotIn(
            green_glass,
            result["visual_compatibility_gate"]["parts"][0]["authorized_material_ids"],
        )
        self.assertEqual(result["summary"]["physical_cross_family_fallback_count"], 0)

    def test_local_context_identity_sheet_is_grayscale_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            isolated = root / "isolated.png"
            context = root / "context.png"
            output = root / "sheet.png"
            Image.fromarray(np.full((18, 30, 3), (220, 30, 15), dtype=np.uint8)).save(
                isolated
            )
            Image.fromarray(np.full((30, 54, 3), (10, 180, 60), dtype=np.uint8)).save(
                context
            )

            _write_grayscale_identity_crop(
                isolated,
                output,
                context_source=context,
            )

            with Image.open(output) as sheet:
                self.assertEqual(sheet.size, (512, 280))
                array = np.asarray(sheet)
            self.assertTrue(np.array_equal(array[..., 0], array[..., 1]))
            self.assertTrue(np.array_equal(array[..., 1], array[..., 2]))

    def test_local_context_requires_material_family_prediction(self) -> None:
        with self.assertRaisesRegex(
            PartIdQwenError,
            "requires material-family prediction",
        ):
            run_part_id_qwen_rerank(
                evidence={"schema_version": "qwen-part-id-reference-evidence/v1"},
                retrieval={},
                catalog={},
                runner=_Runner(),
                model="fake",
                output_dir=Path("unused"),
                material_identity_local_context=True,
            )

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

    def test_exact_specific_library_preset_is_preserved_and_applied(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            crop = root / "P0001.png"
            Image.fromarray(np.full((24, 24, 3), (18, 20, 22), dtype=np.uint8)).save(
                crop
            )
            generic = "mdl:Aluminum_Anodized.mdl#Aluminum_Anodized"
            black = "mdl:Aluminum_Anodized_Black.mdl#Aluminum_Anodized_Black"
            blue = "mdl:Aluminum_Anodized_Blue.mdl#Aluminum_Anodized_Blue"
            semantics = self._surface_semantics(
                "metal",
                "anodized",
                finish="matte",
            )
            runner = _SpecificPresetRunner()

            result = run_part_id_qwen_rerank(
                evidence={
                    "schema_version": "qwen-part-id-reference-evidence/v1",
                    "integrity": {"document_sha256": "evidence"},
                    "parts": [
                        {
                            "part_id": "P0001",
                            "status": "observed",
                            "descriptor": {"surface_class": "conductor"},
                            "observations": [
                                {
                                    "view_id": "front",
                                    "crop": str(crop),
                                    "selected_for_material_inference": True,
                                }
                            ],
                        }
                    ],
                },
                retrieval={
                    "groups": [
                        {
                            "group_id": "P0001",
                            "fused_ranking": [
                                {"rank": 1, "material_id": black},
                                {"rank": 2, "material_id": generic},
                                {"rank": 3, "material_id": blue},
                            ],
                        }
                    ]
                },
                catalog={
                    "materials": [
                        {
                            "material_id": material_id,
                            "family": "metal",
                            "finishes": ["matte"],
                            "surface_semantics": semantics,
                        }
                        for material_id in (generic, black, blue)
                    ]
                },
                runner=runner,
                model="fake",
                output_dir=root / "qwen",
                batch_size=1,
                candidate_count=3,
                require_material_family_prediction=True,
            )

        self.assertEqual(runner.calls, 2)
        self.assertEqual(result["choices"], {"P0001": black})
        self.assertEqual(
            result["selections"][0]["match_type"],
            "EXACT_LIBRARY_MATCH",
        )
        self.assertEqual(result["summary"]["exact_library_match_count"], 1)
        self.assertEqual(result["summary"]["color_evidence_used_for_identity_count"], 1)

    def test_unconfirmed_specific_preset_uses_independent_grayscale_fallback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            crop = root / "P0001.png"
            Image.fromarray(np.full((24, 24, 3), (18, 20, 22), dtype=np.uint8)).save(
                crop
            )
            generic = "mdl:Aluminum_Anodized.mdl#Aluminum_Anodized"
            black = "mdl:Aluminum_Anodized_Black.mdl#Aluminum_Anodized_Black"
            blue = "mdl:Aluminum_Anodized_Blue.mdl#Aluminum_Anodized_Blue"
            semantics = self._surface_semantics("metal", "anodized", finish="matte")
            runner = _TwoStageCorrespondingRunner()
            result = run_part_id_qwen_rerank(
                evidence={
                    "schema_version": "qwen-part-id-reference-evidence/v1",
                    "integrity": {"document_sha256": "evidence"},
                    "parts": [
                        {
                            "part_id": "P0001",
                            "status": "observed",
                            "descriptor": {"surface_class": "conductor"},
                            "observations": [
                                {
                                    "view_id": "front",
                                    "crop": str(crop),
                                    "selected_for_material_inference": True,
                                }
                            ],
                        }
                    ],
                },
                retrieval={
                    "groups": [
                        {
                            "group_id": "P0001",
                            "fused_ranking": [
                                {"rank": 1, "material_id": black},
                                {"rank": 2, "material_id": generic},
                                {"rank": 3, "material_id": blue},
                            ],
                        }
                    ]
                },
                catalog={
                    "materials": [
                        {
                            "material_id": material_id,
                            "family": "metal",
                            "finishes": ["matte"],
                            "surface_semantics": semantics,
                        }
                        for material_id in (generic, black, blue)
                    ]
                },
                runner=runner,
                model="fake",
                output_dir=root / "qwen",
                batch_size=1,
                candidate_count=3,
                require_material_family_prediction=True,
            )

        self.assertEqual(runner.calls, 3)
        self.assertEqual(result["choices"], {"P0001": generic})
        selection = result["selections"][0]
        self.assertEqual(selection["match_type"], "CORRESPONDING_MATERIAL")
        self.assertEqual(
            selection["selection_authority"],
            "grayscale_corresponding_material_second_pass",
        )
        self.assertEqual(selection["exact_preset_decision"]["material_id"], generic)
        self.assertEqual(
            result["corresponding_material_qwen_selections"][0]["material_id"],
            generic,
        )
        fallback_prompt = runner.payloads[2]["messages"][1]["content"][-1]["text"]
        self.assertIn("Color is forbidden evidence", fallback_prompt)
        self.assertNotIn(black, fallback_prompt)
        self.assertNotIn(blue, fallback_prompt)

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

    def test_identity_shortlist_uses_rgb_only_to_surface_exact_preset_candidates(
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
            ["mdl:A_Paint.mdl#A_Paint", "mdl:B_Bare.mdl#B_Bare"],
        )
        self.assertEqual([row["original_retrieval_rank"] for row in shortlist], [1, 2])
        self.assertTrue(all(row["color_evidence_used"] is False for row in shortlist))

    def test_full_catalog_pbr_fingerprint_can_authorize_unique_exact_match(
        self,
    ) -> None:
        plastic = "mdl:Plastic.mdl#Plastic"
        abs_plastic = "mdl:Plastic_ABS.mdl#Plastic_ABS"
        rows = _rank_identity_candidates_with_pbr(
            [
                {
                    "material_id": plastic,
                    "identity_match_tier": "exact_material_contract",
                    "predicted_finish_match": False,
                },
                {
                    "material_id": abs_plastic,
                    "identity_match_tier": "exact_material_contract",
                    "predicted_finish_match": False,
                },
            ],
            descriptor={
                "roughness_hint": 0.30,
                "metallicity_hint": 0.02,
                "mvinverse_albedo_median_rgb": [0.9, 0.1, 0.1],
            },
            profiles_by_id={
                plastic: {
                    "authored_mdl": {
                        "reflection_roughness_constant": 0.50,
                        "metallic_constant": 0.0,
                    }
                },
                abs_plastic: {
                    "authored_mdl": {
                        "reflection_roughness_constant": 0.28,
                        "metallic_constant": 0.0,
                    }
                },
            },
        )
        prediction = {
            "status": "APPLYABLE",
            "identity_resolution": "exact_material",
            "confidence": 0.92,
            "surface_finish": "unknown",
        }

        match = _direct_exact_library_match(rows, prediction=prediction)

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match["material_id"], abs_plastic)
        self.assertEqual(
            match["authority"],
            "unique_full_catalog_mvinverse_pbr_fingerprint",
        )
        self.assertEqual(
            _physical_pbr_evidence(
                {
                    "roughness_hint": 0.30,
                    "metallicity_hint": 0.02,
                    "mvinverse_albedo_median_rgb": [0.9, 0.1, 0.1],
                }
            ),
            {"roughness": 0.30, "metallic": 0.02},
        )

    def test_ambiguous_pbr_fingerprint_never_directly_assigns(self) -> None:
        match = _direct_exact_library_match(
            [
                {
                    "material_id": "mdl:A",
                    "identity_match_tier": "exact_material_contract",
                    "predicted_finish_match": False,
                    "physical_pbr_mean_error": 0.03,
                },
                {
                    "material_id": "mdl:B",
                    "identity_match_tier": "exact_material_contract",
                    "predicted_finish_match": False,
                    "physical_pbr_mean_error": 0.05,
                },
            ],
            prediction={
                "status": "APPLYABLE",
                "identity_resolution": "exact_material",
                "confidence": 0.95,
                "surface_finish": "unknown",
            },
        )

        self.assertIsNone(match)

    def test_identity_filter_preserves_specific_presets_and_marks_generic_fallback(
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

        self.assertEqual(
            {row["material_id"] for row in filtered},
            {generic, black, blue},
        )
        by_id = {row["material_id"]: row for row in filtered}
        self.assertFalse(by_id[generic]["specific_library_preset"])
        self.assertTrue(by_id[black]["specific_library_preset"])
        self.assertTrue(by_id[blue]["specific_library_preset"])
        self.assertTrue(
            all(row["generic_identity_material_id"] == generic for row in filtered)
        )

    def test_material_species_hard_filter_keeps_copper_out_of_chrome_and_steel(
        self,
    ) -> None:
        copper = "mdl:Metals/Copper.mdl#Copper"
        brushed_copper = "mdl:Metals/Brushed_Antique_Copper.mdl#Brushed_Antique_Copper"
        chrome = "mdl:Metals/Chrome.mdl#Chrome"
        steel = "mdl:Metals/Steel_Stainless.mdl#Steel_Stainless"
        catalog = {
            material_id: {
                "material_id": material_id,
                "surface_semantics": self._surface_semantics("metal"),
            }
            for material_id in (copper, brushed_copper, chrome, steel)
        }

        filtered = _identity_filtered_ranking(
            ranking=[
                {"rank": 1, "material_id": chrome},
                {"rank": 2, "material_id": steel},
                {"rank": 3, "material_id": brushed_copper},
                {"rank": 4, "material_id": copper},
            ],
            catalog_by_id=catalog,
            prediction={
                "part_id": "P0161",
                "catalog_family": "metal",
                "physical_substrate": "metal",
                "material_species": "copper",
                "surface_treatment": "unknown",
                "optical_behavior": "opaque",
                "surface_finish": "unknown",
                "confidence": 0.90,
                "substrate_confidence": 0.92,
                "species_confidence": 0.88,
                "treatment_confidence": 0.30,
                "identity_resolution": "corresponding_material",
                "status": "APPLYABLE",
            },
        )

        self.assertEqual(
            {row["material_id"] for row in filtered},
            {copper, brushed_copper},
        )
        self.assertTrue(all(row["material_species"] == "copper" for row in filtered))
        self.assertTrue(all(row["exact_authored_preset_candidate"] for row in filtered))

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

    def test_final_evidence_expands_only_same_coating_and_assembly_branch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry_path = root / "registry.json"
            registry = {
                "default_prim": "/Asset",
                "parts": [
                    {
                        "part_id": "P1",
                        "prim_path": "/Asset/Asset/Housing/P1/Mesh",
                    },
                    {
                        "part_id": "P2",
                        "prim_path": "/Asset/Asset/Housing/P2/Mesh",
                    },
                    {
                        "part_id": "P3",
                        "prim_path": "/Asset/Asset/Housing/P3/Mesh",
                    },
                    {
                        "part_id": "P4",
                        "prim_path": "/Asset/Asset/Other/P4/Mesh",
                    },
                    {
                        "part_id": "P5",
                        "prim_path": "/Asset/Asset/Housing/P5/Mesh",
                    },
                ],
            }
            registry_path.write_text(json.dumps(registry))
            registry_sha = hashlib.sha256(registry_path.read_bytes()).hexdigest()

            def part(
                part_id: str,
                box: list[int],
                *,
                surface_class: str = "dielectric",
            ) -> dict[str, object]:
                return {
                    "part_id": part_id,
                    "descriptor": {
                        "surface_class": surface_class,
                        "median_rgb": [0.20, 0.42, 0.23],
                        "robust_color_evidence": {
                            "sample_count": 2048,
                            "inlier_fraction": 0.95,
                            "robust_reference_srgb": [0.20, 0.42, 0.23],
                        },
                    },
                    "observations": [
                        {
                            "view_id": "front",
                            "target_box_xyxy": box,
                            "trusted_foreground_pixels": 2048,
                            "camera_alignment_evidence_weight": 0.8,
                        }
                    ],
                }

            parts = {
                row["part_id"]: row
                for row in (
                    part("P1", [0, 0, 50, 50]),
                    part("P2", [50, 0, 100, 50]),
                    part("P3", [98, 0, 150, 50]),
                    # Same photo appearance and location, but a different CAD
                    # branch is not sufficient membership authority.
                    part("P4", [98, 0, 150, 50]),
                    # Same assembly and location, but a conductor is not the
                    # same physical coating as the dielectric housing seed.
                    part("P5", [98, 0, 150, 50], surface_class="conductor"),
                )
            }
            refined, audit = _refine_component_memberships_with_final_evidence(
                appearance_components={
                    "inputs": {
                        "rendered_registry": str(registry_path),
                        "rendered_registry_sha256": registry_sha,
                    },
                    "components": [
                        {
                            "component_id": "AC_green",
                            "member_part_ids": ["P1", "P2"],
                            "canonical_reference_rgb": [0.20, 0.42, 0.23],
                        }
                    ],
                },
                component_members={"AC_green": ["P1", "P2"]},
                part_by_id=parts,
            )

        self.assertEqual(refined["AC_green"], ["P1", "P2", "P3"])
        self.assertEqual(audit["summary"]["added_member_count"], 1)
        self.assertEqual(audit["components"][0]["added_members"][0]["part_id"], "P3")
        self.assertGreaterEqual(
            audit["components"][0]["added_members"][0]["spatial_support"],
            0.60,
        )

    def test_final_evidence_never_resolves_ambiguous_component_membership(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry_path = root / "registry.json"
            registry = {
                "default_prim": "/Asset",
                "parts": [
                    {
                        "part_id": part_id,
                        "prim_path": f"/Asset/Asset/Housing/{part_id}/Mesh",
                    }
                    for part_id in ("P1", "P2", "P3", "P4", "P5")
                ],
            }
            registry_path.write_text(json.dumps(registry))
            registry_sha = hashlib.sha256(registry_path.read_bytes()).hexdigest()
            parts = {
                part_id: {
                    "part_id": part_id,
                    "descriptor": {
                        "surface_class": "dielectric",
                        "median_rgb": [0.2, 0.4, 0.2],
                        "robust_color_evidence": {
                            "sample_count": 1024,
                            "inlier_fraction": 0.9,
                            "robust_reference_srgb": [0.2, 0.4, 0.2],
                        },
                    },
                    "observations": [
                        {
                            "view_id": "front",
                            "target_box_xyxy": [0, 0, 100, 100],
                            "trusted_foreground_pixels": 1024,
                            "camera_alignment_evidence_weight": 0.8,
                        }
                    ],
                }
                for part_id in ("P1", "P2", "P3", "P4", "P5")
            }
            refined, audit = _refine_component_memberships_with_final_evidence(
                appearance_components={
                    "inputs": {
                        "rendered_registry": str(registry_path),
                        "rendered_registry_sha256": registry_sha,
                    },
                    "components": [
                        {
                            "component_id": "AC_1",
                            "member_part_ids": ["P1", "P2"],
                            "canonical_reference_rgb": [0.2, 0.4, 0.2],
                        },
                        {
                            "component_id": "AC_2",
                            "member_part_ids": ["P4", "P5"],
                            "canonical_reference_rgb": [0.2, 0.4, 0.2],
                        },
                    ],
                },
                component_members={"AC_1": ["P1", "P2"], "AC_2": ["P4", "P5"]},
                part_by_id=parts,
            )

        self.assertNotIn("P3", refined["AC_1"])
        self.assertNotIn("P3", refined["AC_2"])
        self.assertEqual(audit["ambiguous_part_ids"], ["P3"])

    def test_repeated_cad_instances_share_each_homologous_material_role(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry_path = root / "registry.json"
            registry_parts = []
            part_by_id: dict[str, dict[str, object]] = {}
            body_ids: list[str] = []
            cap_ids: list[str] = []
            tip_ids: list[str] = []
            for instance_index, instance in enumerate(("Pen_A", "Pen_B", "Pen_C")):
                for role, prefix, geometry in (
                    ("Body", "B", "a" * 64),
                    ("Cap", "C", "b" * 64),
                    ("Tip", "T", "c" * 64),
                ):
                    part_id = f"{prefix}{instance_index + 1}"
                    registry_parts.append(
                        {
                            "part_id": part_id,
                            "prim_path": f"/Asset/Assembly/{instance}/{role}/Mesh",
                            "geometry_content_sha256": geometry,
                            "point_count": 128,
                            "face_count": 64,
                        }
                    )
                    surface = (
                        "conductor"
                        if role == "Tip" and instance_index == 0
                        else "dielectric"
                    )
                    part_by_id[part_id] = {
                        "part_id": part_id,
                        "descriptor": {"surface_class": surface},
                        "observations": [],
                    }
                    {"Body": body_ids, "Cap": cap_ids, "Tip": tip_ids}[role].append(
                        part_id
                    )
            registry = {
                "schema_version": "qwen-material-parts/v1",
                "parts": registry_parts,
            }
            registry_path.write_text(json.dumps(registry))
            document_sha256 = hashlib.sha256(
                json.dumps(
                    registry,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            (
                components,
                audit,
                strict_ids,
                scopes,
            ) = _apply_repeated_assembly_role_constraints(
                evidence={
                    "inputs": [
                        {
                            "label": "rendered_registry",
                            "path": str(registry_path),
                            "document_sha256": document_sha256,
                        }
                    ]
                },
                part_by_id=part_by_id,
                component_members={"AC_body": body_ids},
            )

        self.assertEqual(audit["summary"]["repeated_structure_count"], 1)
        self.assertEqual(audit["summary"]["candidate_role_count"], 3)
        self.assertEqual(audit["summary"]["constrained_role_count"], 2)
        self.assertEqual(audit["summary"]["new_role_component_count"], 1)
        self.assertEqual(components["AC_body"], body_ids)
        role_components = {
            component_id: members
            for component_id, members in components.items()
            if component_id != "AC_body"
        }
        self.assertEqual(list(role_components.values()), [cap_ids])
        role_component_id = next(iter(role_components))
        self.assertEqual(scopes[role_component_id], "repeated_assembly_role")
        self.assertEqual(strict_ids, {"AC_body", role_component_id})
        role_statuses = {
            role["relative_prim_path"]: role["status"]
            for role in audit["structures"][0]["roles"]
        }
        self.assertEqual(role_statuses["Body/Mesh"], "ALREADY_CONSTRAINED")
        self.assertEqual(role_statuses["Cap/Mesh"], "CREATED_ROLE_COMPONENT")
        self.assertEqual(role_statuses["Tip/Mesh"], "PHYSICAL_SURFACE_CONFLICT")

    def test_repeated_role_strict_consensus_resolves_noisy_exact_presets(
        self,
    ) -> None:
        selections, audit = _apply_component_identity_consensus(
            selections=[
                {
                    "part_id": "P1",
                    "material_id": "mdl:Matte",
                    "match_type": "EXACT_LIBRARY_MATCH",
                    "confidence": 0.92,
                },
                {
                    "part_id": "P2",
                    "material_id": "mdl:Chrome",
                    "match_type": "EXACT_LIBRARY_MATCH",
                    "confidence": 0.85,
                },
                {
                    "part_id": "P3",
                    "material_id": "mdl:Matte",
                    "match_type": "CORRESPONDING_MATERIAL",
                    "confidence": 0.75,
                },
                {
                    "part_id": "P4",
                    "material_id": "mdl:Matte",
                    "match_type": "CORRESPONDING_MATERIAL",
                    "confidence": 0.75,
                },
            ],
            component_members={"CAD_ROLE_1": ["P1", "P2", "P3", "P4"]},
            strict_consensus_component_ids={"CAD_ROLE_1"},
        )

        self.assertEqual({row["material_id"] for row in selections}, {"mdl:Matte"})
        self.assertEqual(
            audit["components"][0]["consensus_mode"],
            "REPEATED_ROLE_JOINT_CONSENSUS",
        )
        self.assertTrue(audit["components"][0]["exact_shared_material_enforced"])

    def test_component_consensus_never_overwrites_conflicting_exact_presets(
        self,
    ) -> None:
        selections, audit = _apply_component_identity_consensus(
            selections=[
                {
                    "part_id": "P1",
                    "material_id": "mdl:Black",
                    "match_type": "EXACT_LIBRARY_MATCH",
                    "confidence": 0.92,
                },
                {
                    "part_id": "P2",
                    "material_id": "mdl:Blue",
                    "match_type": "EXACT_LIBRARY_MATCH",
                    "confidence": 0.91,
                },
                {
                    "part_id": "P3",
                    "material_id": "mdl:Generic",
                    "match_type": "CORRESPONDING_MATERIAL",
                    "confidence": 0.75,
                },
            ],
            component_members={"AC_1": ["P1", "P2", "P3"]},
        )

        self.assertEqual(
            {row["part_id"]: row["material_id"] for row in selections},
            {"P1": "mdl:Black", "P2": "mdl:Blue", "P3": "mdl:Generic"},
        )
        component = audit["components"][0]
        self.assertEqual(
            component["consensus_mode"],
            "CONFLICTING_EXACT_PRESETS_PRESERVED",
        )
        self.assertFalse(component["exact_shared_material_enforced"])
        self.assertFalse(audit["summary"]["all_components_share_one_exact_mdl"])

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
                            "surface_semantics": self._surface_semantics("elastomer"),
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
                    root / "qwen" / "identity_evidence_grayscale" / "P0001_front_01.png"
                )
            )

        self.assertEqual(runner.calls, 1)
        self.assertEqual(set(result["choices"].values()), {matte})
        self.assertEqual(
            {row["component_id"] for row in result["material_predictions"]},
            {"AC_1"},
        )
        self.assertEqual(
            result["component_identity_consensus"]["summary"]["constrained_part_count"],
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

    def test_corresponding_specific_preset_normalizes_to_generic_identity(
        self,
    ) -> None:
        expected = [
            {
                "part_id": "P0001",
                "candidates": [
                    {
                        "candidate_index": 1,
                        "material_id": "mdl:Aluminum_Anodized_Black",
                        "selection_allowed": True,
                        "specific_library_preset": True,
                        "generic_identity_material_id": ("mdl:Aluminum_Anodized"),
                    },
                    {
                        "candidate_index": 2,
                        "material_id": "mdl:Aluminum_Anodized",
                        "selection_allowed": True,
                        "specific_library_preset": False,
                    },
                ],
            }
        ]

        selections = _validate_batch(
            {
                "schema_version": MATERIAL_IDENTITY_SELECTION_BATCH_SCHEMA_VERSION,
                "selections": [
                    {
                        "part_id": "P0001",
                        "candidate_index": 1,
                        "match_type": "CORRESPONDING_MATERIAL",
                        "confidence": 0.8,
                    }
                ],
            },
            expected=expected,
            require_material_identity_match=True,
        )

        self.assertEqual(selections[0]["candidate_index"], 2)
        self.assertEqual(
            selections[0]["material_id"],
            "mdl:Aluminum_Anodized",
        )
        self.assertEqual(
            selections[0]["index_resolution"],
            "specific_preset_to_generic_corresponding_fallback",
        )

    def test_exact_preset_requires_shape_guided_photo_instance_evidence(
        self,
    ) -> None:
        specific = "mdl:Aluminum_Anodized_Black"
        generic = "mdl:Aluminum_Anodized"
        expected = [
            {
                "part_id": "P0001",
                "candidates": [
                    {
                        "candidate_index": 1,
                        "material_id": specific,
                        "selection_allowed": True,
                        "specific_library_preset": True,
                        "generic_identity_material_id": generic,
                        "exact_preset_evidence_eligible": False,
                        "exact_preset_color_gate_passed": None,
                    },
                    {
                        "candidate_index": 2,
                        "material_id": generic,
                        "selection_allowed": True,
                        "specific_library_preset": False,
                        "exact_preset_evidence_eligible": False,
                    },
                ],
            }
        ]

        selections = _validate_batch(
            {
                "schema_version": MATERIAL_IDENTITY_SELECTION_BATCH_SCHEMA_VERSION,
                "selections": [
                    {
                        "part_id": "P0001",
                        "candidate_index": 1,
                        "match_type": "EXACT_LIBRARY_MATCH",
                        "confidence": 0.9,
                    }
                ],
            },
            expected=expected,
            require_material_identity_match=True,
        )

        self.assertEqual(selections[0]["material_id"], generic)
        self.assertEqual(selections[0]["match_type"], "CORRESPONDING_MATERIAL")
        self.assertEqual(
            selections[0]["index_resolution"],
            "specific_preset_to_generic_corresponding_fallback",
        )

    def test_shape_guided_exact_evidence_rejects_tiny_or_projection_only_mask(
        self,
    ) -> None:
        precise = {
            "photo_part_segmentation_applied": True,
            "trusted_foreground_pixels": 64,
            "part_id_sam3_refinement": {
                "applied": True,
                "shape_candidate": {
                    "cad_shape_location_invariant": True,
                    "cad_shape_iou": 0.82,
                },
            },
        }
        self.assertTrue(_shape_guided_exact_preset_eligible(precise, required=True))
        self.assertFalse(
            _shape_guided_exact_preset_eligible(
                {**precise, "trusted_foreground_pixels": 6},
                required=True,
            )
        )
        self.assertFalse(
            _shape_guided_exact_preset_eligible(
                {**precise, "photo_part_segmentation_applied": False},
                required=True,
            )
        )

    def test_identity_specific_preset_is_promoted_after_measured_confirmation(
        self,
    ) -> None:
        copper = "mdl:Metals/Brushed_Antique_Copper.mdl#Brushed_Antique_Copper"
        expected = [
            {
                "part_id": "P0161",
                "candidates": [
                    {
                        "candidate_index": 1,
                        "material_id": copper,
                        "material_species": "copper",
                        "selection_allowed": True,
                        "specific_library_preset": False,
                        "exact_authored_preset_candidate": True,
                        "physical_identity_applyable": True,
                        "exact_preset_color_gate_passed": True,
                        "exact_preset_color_delta_e": 7.89,
                    }
                ],
            }
        ]

        selections = _validate_batch(
            {
                "schema_version": MATERIAL_IDENTITY_SELECTION_BATCH_SCHEMA_VERSION,
                "selections": [
                    {
                        "part_id": "P0161",
                        "candidate_index": 1,
                        "match_type": "CORRESPONDING_MATERIAL",
                        "confidence": 0.85,
                    }
                ],
            },
            expected=expected,
            require_material_identity_match=True,
        )

        self.assertEqual(selections[0]["material_id"], copper)
        self.assertEqual(selections[0]["material_species"], "copper")
        self.assertEqual(selections[0]["match_type"], "EXACT_LIBRARY_MATCH")
        self.assertEqual(
            selections[0]["index_resolution"],
            "deterministic_exact_authored_preset_promotion",
        )

    def test_identity_specific_preset_is_not_promoted_on_color_mismatch(
        self,
    ) -> None:
        expected = [
            {
                "part_id": "P_ORANGE_PAINT",
                "candidates": [
                    {
                        "candidate_index": 1,
                        "material_id": "mdl:Metals/Copper.mdl#Copper",
                        "material_species": "copper",
                        "selection_allowed": True,
                        "specific_library_preset": False,
                        "exact_authored_preset_candidate": True,
                        "exact_preset_color_gate_passed": False,
                        "exact_preset_color_delta_e": 39.0,
                    }
                ],
            }
        ]

        selections = _validate_batch(
            {
                "schema_version": MATERIAL_IDENTITY_SELECTION_BATCH_SCHEMA_VERSION,
                "selections": [
                    {
                        "part_id": "P_ORANGE_PAINT",
                        "candidate_index": 1,
                        "match_type": "CORRESPONDING_MATERIAL",
                        "confidence": 0.90,
                    }
                ],
            },
            expected=expected,
            require_material_identity_match=True,
        )

        self.assertEqual(selections[0]["match_type"], "CORRESPONDING_MATERIAL")
        self.assertEqual(selections[0]["material_species"], "copper")

    def test_exact_preset_with_measured_color_mismatch_is_rejected(self) -> None:
        expected = [
            {
                "part_id": "P0001",
                "candidates": [
                    {
                        "candidate_index": 1,
                        "material_id": "mdl:Paint_Matte_Finish",
                        "selection_allowed": True,
                        "specific_library_preset": True,
                        "generic_identity_material_id": "mdl:Paint_Matte",
                        "exact_preset_color_gate_passed": False,
                        "exact_preset_color_delta_e": 48.0,
                    },
                    {
                        "candidate_index": 2,
                        "material_id": "mdl:Paint_Matte",
                        "selection_allowed": True,
                        "specific_library_preset": False,
                    },
                ],
            }
        ]

        selections = _validate_batch(
            {
                "schema_version": MATERIAL_IDENTITY_SELECTION_BATCH_SCHEMA_VERSION,
                "selections": [
                    {
                        "part_id": "P0001",
                        "candidate_index": 1,
                        "match_type": "EXACT_LIBRARY_MATCH",
                        "confidence": 0.91,
                    }
                ],
            },
            expected=expected,
            require_material_identity_match=True,
        )

        self.assertEqual(selections[0]["material_id"], "mdl:Paint_Matte")
        self.assertEqual(selections[0]["match_type"], "CORRESPONDING_MATERIAL")
        self.assertEqual(
            selections[0]["index_resolution"],
            "exact_preset_color_gate_rejected_to_generic",
        )

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
        self.assertEqual(audit["summary"]["fresh_local_baseline_selected_count"], 3)
        chromatic_audit = next(
            row for row in audit["parts"] if row["part_id"] == "P_CHROMATIC"
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
            next(row for row in rows if row["material_id"] == "mdl:Glass.mdl#Glass")[
                "transmission_risk"
            ]
        )
        self.assertFalse(
            next(row for row in rows if row["material_id"] == "mdl:Glass.mdl#Glass")[
                "selection_allowed"
            ]
        )
        self.assertTrue(
            next(row for row in rows if row["material_id"] == "mdl:Grass.mdl#Grass")[
                "texture_mismatch_risk"
            ]
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
        mutable_vinyl = next(row for row in mutable_rows if row["material_id"] == vinyl)
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
        self.assertTrue(all(row["conditional_h1_evaluation"] for row in promoted))
        self.assertTrue(
            all(
                row["relaxed_constraints"] == ["default_color_gate"] for row in promoted
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
