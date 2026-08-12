from __future__ import annotations

import hashlib
import json
import unittest

from qwen_material_pipeline.materials.component_mdl_tournament import (
    ComponentMdlTournamentError,
    build_component_candidate_plan,
    build_component_color_candidate_plan,
    component_candidate_material_ids,
    rebind_part_id_material_audit_for_component_mdl_tournament,
    select_component_color_winner,
    select_component_mdl_winner,
)


class ComponentMdlTournamentTests(unittest.TestCase):
    @staticmethod
    def _sha256(value: object) -> str:
        return hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _member_semantics(
        *,
        treatment: str = "paint",
        finish: str = "unknown",
        evidence_status: str = "observed",
    ) -> dict[str, dict]:
        return {
            part_id: {
                "schema_version": "qwen-part-material-semantics/v1",
                "substrate": "metal",
                "surface_treatment": treatment,
                "optical_behavior": "opaque",
                "finish": finish,
                "physical_source": "vision_inference",
                "evidence_status": evidence_status,
                "confidence": 0.9,
            }
            for part_id in ("P0001", "P0002")
        }

    @staticmethod
    def _catalog_record(
        material_id: str,
        *,
        family: str = "paint",
        treatment: str = "paint",
        finish: str = "matte",
        confidence: str = "high",
        compatible_substrates: list[str] | None = None,
    ) -> dict:
        return {
            "material_id": material_id,
            "family": family,
            "surface_semantics": {
                "schema_version": "qwen-catalog-surface-semantics/v1",
                "compatible_substrates": (
                    compatible_substrates
                    if compatible_substrates is not None
                    else ["metal", "polymer", "wood"]
                ),
                "surface_treatment": treatment,
                "optical_behavior": "opaque",
                "finish": finish,
                "inference_source": "unit_test/v1",
                "confidence": confidence,
            },
        }

    @classmethod
    def _strict_catalog(cls) -> dict[str, dict]:
        ids = {
            "matte": "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte",
            "matte_alias": (
                "mdl:Miscellaneous/Paint_Matte_Finish.mdl#Paint_Matte_Finish"
            ),
            "satin": "mdl:Miscellaneous/Paint_Satin.mdl#Paint_Satin",
            "satin_alias": (
                "mdl:Miscellaneous/Paint_Satin_Finish.mdl#Paint_Satin_Finish"
            ),
            "gloss": "mdl:Miscellaneous/Paint_Gloss.mdl#Paint_Gloss",
            "broken": "mdl:Miscellaneous/Paint_Broken.mdl#Paint_Broken",
            "low": "mdl:Miscellaneous/Paint_Low.mdl#Paint_Low",
            "water": "mdl:Natural/Water.mdl#Water",
            "grass": "mdl:Natural/Grass_Countryside.mdl#Grass_Countryside",
            "mirror": "mdl:Glass/Mirror.mdl#Mirror",
        }
        catalog = {
            ids["matte"]: cls._catalog_record(ids["matte"], finish="matte"),
            ids["matte_alias"]: cls._catalog_record(
                ids["matte_alias"], finish="matte"
            ),
            ids["satin"]: cls._catalog_record(ids["satin"], finish="satin"),
            ids["satin_alias"]: cls._catalog_record(
                ids["satin_alias"], finish="satin"
            ),
            ids["gloss"]: cls._catalog_record(ids["gloss"], finish="glossy"),
            ids["broken"]: {
                "material_id": ids["broken"],
                "family": "paint",
                "surface_semantics": {"surface_treatment": "paint"},
            },
            ids["low"]: cls._catalog_record(
                ids["low"], finish="satin", confidence="low"
            ),
        }
        # These records are deliberately forged as otherwise-compatible Paint.
        # Strict mode must still reject the known visual proxy identities.
        for key in ("water", "grass", "mirror"):
            catalog[ids[key]] = cls._catalog_record(ids[key], finish="satin")
        return catalog

    def _plan(self) -> dict:
        return {
            "schema_version": "1.0",
            "assignments": [
                {
                    "part_id": "P0001",
                    "material_id": "mdl:Metals/Steel_Cast.mdl#Steel_Cast",
                    "parameters": {},
                    "provenance": {"tier": "baseline"},
                },
                {
                    "part_id": "P0002",
                    "material_id": "mdl:Metals/Steel_Cast.mdl#Steel_Cast",
                },
                {
                    "part_id": "P0003",
                    "material_id": "mdl:Plastics/Plastic_ABS.mdl#Plastic_ABS",
                },
            ],
            "provenance": {"run": "test"},
        }

    def test_candidate_changes_only_component_mdl_identity(self) -> None:
        source = self._plan()
        result = build_component_candidate_plan(
            source_plan=source,
            component_id="AC_green",
            member_part_ids=["P0002", "P0001"],
            material_id="mdl:Glass/Green_Glass.mdl#Green_Glass",
        )
        assignments = {row["part_id"]: row for row in result["assignments"]}
        self.assertEqual(
            assignments["P0001"]["material_id"],
            "mdl:Glass/Green_Glass.mdl#Green_Glass",
        )
        self.assertEqual(
            assignments["P0002"]["material_id"],
            "mdl:Glass/Green_Glass.mdl#Green_Glass",
        )
        self.assertNotIn("parameters", assignments["P0001"])
        self.assertEqual(
            assignments["P0003"]["material_id"],
            "mdl:Plastics/Plastic_ABS.mdl#Plastic_ABS",
        )
        self.assertFalse(
            assignments["P0001"]["provenance"]
            ["appearance_component_actual_mdl_candidate"]
            ["mdl_parameter_mutation_allowed"]
        )
        self.assertEqual(
            source["assignments"][0]["material_id"],
            "mdl:Metals/Steel_Cast.mdl#Steel_Cast",
        )

    def test_winner_requires_actual_render_improvement(self) -> None:
        result = select_component_mdl_winner(
            component_id="AC_green",
            baseline_material_id="mdl:Glass/Green_Glass.mdl#Green_Glass",
            candidate_scores={
                "mdl:Glass/Green_Glass.mdl#Green_Glass": {"appearance_score": 0.28},
                "mdl:Natural/Grass_Countryside.mdl#Grass_Countryside": {
                    "appearance_score": 0.42
                },
            },
            minimum_score_improvement=0.015,
        )
        self.assertEqual(
            result["selected_material_id"],
            "mdl:Natural/Grass_Countryside.mdl#Grass_Countryside",
        )
        self.assertEqual(result["selection_status"], "ACTUAL_CAD_RENDER_WINNER")
        self.assertFalse(result["mdl_parameter_mutation_allowed"])

        retained = select_component_mdl_winner(
            component_id="AC_green",
            baseline_material_id="mdl:Glass/Green_Glass.mdl#Green_Glass",
            candidate_scores={
                "mdl:Glass/Green_Glass.mdl#Green_Glass": {"appearance_score": 0.28},
                "mdl:Natural/Grass_Countryside.mdl#Grass_Countryside": {
                    "appearance_score": 0.29
                },
            },
            minimum_score_improvement=0.015,
        )
        self.assertEqual(retained["selected_material_id"], "mdl:Glass/Green_Glass.mdl#Green_Glass")
        self.assertEqual(retained["selection_status"], "BASELINE_RETAINED")

    def test_candidate_shortlist_preserves_baseline_and_color_diversity(self) -> None:
        candidates = component_candidate_material_ids(
            baseline_material_id="mdl:Glass/Green_Glass.mdl#Green_Glass",
            retrieval_group={
                "color_ranking": [
                    {
                        "rank": 1,
                        "material_id": (
                            "mdl:Natural/Grass_Countryside.mdl#Grass_Countryside"
                        ),
                    },
                    {
                        "rank": 2,
                        "material_id": "mdl:Natural/Grass_Cut.mdl#Grass_Cut",
                    },
                ]
            },
            visual_compatibility={
                "shortlist": [
                    {
                        "compatibility_rank": 1,
                        "material_id": "mdl:Glass/Green_Glass.mdl#Green_Glass",
                    },
                    {
                        "compatibility_rank": 2,
                        "material_id": "mdl:Carpet/Carpet_Forest.mdl#Carpet_Forest",
                    },
                ]
            },
            maximum_candidates=4,
        )
        self.assertEqual(
            candidates,
            [
                "mdl:Glass/Green_Glass.mdl#Green_Glass",
                "mdl:Natural/Grass_Countryside.mdl#Grass_Countryside",
                "mdl:Natural/Grass_Cut.mdl#Grass_Cut",
                "mdl:Carpet/Carpet_Forest.mdl#Carpet_Forest",
            ],
        )

    def test_strict_shortlist_replaces_unsafe_baseline_and_uses_catalog_fallback(
        self,
    ) -> None:
        catalog = self._strict_catalog()
        candidates = component_candidate_material_ids(
            baseline_material_id="mdl:Natural/Water.mdl#Water",
            retrieval_group={
                "color_ranking": [
                    {
                        "rank": 0,
                        "material_id": (
                            "mdl:Miscellaneous/Paint_Broken.mdl#Paint_Broken"
                        ),
                    },
                    {
                        "rank": 1,
                        "material_id": (
                            "mdl:Natural/Grass_Countryside.mdl#Grass_Countryside"
                        ),
                    },
                    {
                        "rank": 2,
                        "material_id": "mdl:Glass/Mirror.mdl#Mirror",
                    },
                    {
                        "rank": 3,
                        "material_id": "mdl:Miscellaneous/Paint_Low.mdl#Paint_Low",
                    },
                ],
                "fused_ranking": [],
            },
            visual_compatibility=None,
            maximum_candidates=99,
            member_material_semantics=self._member_semantics(),
            catalog_materials_by_id=catalog,
            preferred_finish="satin",
        )
        self.assertEqual(
            candidates,
            [
                "mdl:Miscellaneous/Paint_Satin.mdl#Paint_Satin",
                "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte",
                "mdl:Miscellaneous/Paint_Gloss.mdl#Paint_Gloss",
            ],
        )
        self.assertEqual(len(candidates), 3)

    def test_strict_shortlist_preserves_only_a_compatible_baseline(self) -> None:
        catalog = self._strict_catalog()
        matte = "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte"
        satin = "mdl:Miscellaneous/Paint_Satin.mdl#Paint_Satin"
        candidates = component_candidate_material_ids(
            baseline_material_id=matte,
            retrieval_group={
                "color_ranking": [
                    {"rank": 1, "material_id": "mdl:Natural/Water.mdl#Water"},
                    {"rank": 2, "material_id": satin},
                ]
            },
            visual_compatibility=None,
            maximum_candidates=3,
            member_material_semantics=self._member_semantics(),
            catalog_materials_by_id=catalog,
        )
        self.assertEqual(candidates[0], matte)
        self.assertEqual(candidates[1], satin)
        self.assertNotIn("mdl:Natural/Water.mdl#Water", candidates)

    def test_strict_shortlist_fails_closed_for_unresolved_conflicting_or_few_members(
        self,
    ) -> None:
        catalog = self._strict_catalog()
        cases = []
        one_member = self._member_semantics()
        one_member.pop("P0002")
        cases.append(one_member)
        cases.append(self._member_semantics(evidence_status="unknown"))
        conflict = self._member_semantics()
        conflict["P0002"] = {
            **conflict["P0002"],
            "surface_treatment": "bare",
        }
        cases.append(conflict)
        malformed = self._member_semantics()
        malformed["P0002"] = {"surface_treatment": "paint"}
        cases.append(malformed)

        for member_semantics in cases:
            with self.subTest(member_semantics=member_semantics):
                with self.assertRaises(ComponentMdlTournamentError):
                    component_candidate_material_ids(
                        baseline_material_id=(
                            "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte"
                        ),
                        retrieval_group={},
                        visual_compatibility=None,
                        member_material_semantics=member_semantics,
                        catalog_materials_by_id=catalog,
                    )

    def test_strict_shortlist_fails_when_catalog_has_fewer_than_two_safe_ids(
        self,
    ) -> None:
        matte = "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte"
        catalog = {matte: self._catalog_record(matte)}
        with self.assertRaisesRegex(
            ComponentMdlTournamentError,
            "fewer than two compatible",
        ):
            component_candidate_material_ids(
                baseline_material_id="mdl:Natural/Water.mdl#Water",
                retrieval_group={},
                visual_compatibility=None,
                member_material_semantics=self._member_semantics(),
                catalog_materials_by_id=catalog,
            )

    def test_strict_candidate_plan_revalidates_every_component_member(self) -> None:
        catalog = self._strict_catalog()
        matte = "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte"
        result = build_component_candidate_plan(
            source_plan=self._plan(),
            component_id="AC_paint",
            member_part_ids=["P0001", "P0002"],
            material_id=matte,
            member_material_semantics=self._member_semantics(),
            catalog_materials_by_id=catalog,
        )
        gate = result["provenance"]["appearance_component_actual_mdl_candidate"][
            "semantic_compatibility_gate"
        ]
        self.assertEqual(gate["target_family"], "paint")
        self.assertEqual(
            gate["policy"],
            "all_component_members_physical_semantics_compatible/v1",
        )
        with self.assertRaisesRegex(
            ComponentMdlTournamentError,
            "not compatible with every member",
        ):
            build_component_candidate_plan(
                source_plan=self._plan(),
                component_id="AC_paint",
                member_part_ids=["P0001", "P0002"],
                material_id="mdl:Glass/Mirror.mdl#Mirror",
                member_material_semantics=self._member_semantics(),
                catalog_materials_by_id=catalog,
            )

    def test_strict_winner_rejects_proxy_or_unauthorized_score_rows(self) -> None:
        catalog = self._strict_catalog()
        matte = "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte"
        satin = "mdl:Miscellaneous/Paint_Satin.mdl#Paint_Satin"
        strict = {
            "member_material_semantics": self._member_semantics(),
            "catalog_materials_by_id": catalog,
        }
        with self.assertRaisesRegex(
            ComponentMdlTournamentError,
            "semantically incompatible",
        ):
            select_component_mdl_winner(
                component_id="AC_paint",
                baseline_material_id=matte,
                candidate_scores={
                    matte: {"appearance_score": 0.3},
                    "mdl:Natural/Water.mdl#Water": {"appearance_score": 0.99},
                },
                **strict,
            )
        with self.assertRaisesRegex(
            ComponentMdlTournamentError,
            "authorized shortlist",
        ):
            select_component_mdl_winner(
                component_id="AC_paint",
                baseline_material_id=matte,
                candidate_scores={
                    matte: {"appearance_score": 0.3},
                    satin: {"appearance_score": 0.5},
                },
                authorized_candidate_material_ids=[matte],
                **strict,
            )

        winner = select_component_mdl_winner(
            component_id="AC_paint",
            baseline_material_id=matte,
            candidate_scores={
                matte: {"appearance_score": 0.3},
                satin: {"appearance_score": 0.5},
            },
            authorized_candidate_material_ids=[matte, satin],
            **strict,
        )
        self.assertEqual(winner["selected_material_id"], satin)

    @staticmethod
    def _component_score(
        component_id: str,
        first: float,
        second: float,
        *,
        first_pixels: int = 100,
        second_pixels: int = 100,
    ) -> dict:
        aggregate = (
            first * first_pixels + second * second_pixels
        ) / (first_pixels + second_pixels)
        return {
            "schema_version": "qwen-appearance-component-actual-mdl-tournament/v1",
            "component_id": component_id,
            "member_part_ids": ["P0001", "P0002"],
            "member_score_count": 2,
            "comparison_pixel_count": first_pixels + second_pixels,
            "appearance_score": round(aggregate, 8),
            "color_score": round(aggregate, 8),
            "luma_score": round(aggregate, 8),
            "lab_delta_e": 10.0,
            "member_scores": [
                {
                    "part_id": "P0001",
                    "appearance_score": first,
                    "comparison_pixel_count": first_pixels,
                },
                {
                    "part_id": "P0002",
                    "appearance_score": second,
                    "comparison_pixel_count": second_pixels,
                },
            ],
        }

    def test_component_color_candidate_is_same_id_same_parameters_for_all_members(
        self,
    ) -> None:
        catalog = self._strict_catalog()
        matte = "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte"
        source = self._plan()
        for assignment in source["assignments"][:2]:
            assignment["material_id"] = matte
            assignment.pop("parameters", None)
        candidate = build_component_color_candidate_plan(
            source_plan=source,
            component_id="AC_paint",
            member_part_ids=["P0002", "P0001"],
            material_id=matte,
            target_srgb=[0.2, 0.45, 0.25],
            member_material_semantics=self._member_semantics(),
            catalog_materials_by_id=catalog,
        )
        assignments = {row["part_id"]: row for row in candidate["assignments"]}
        self.assertEqual(assignments["P0001"]["material_id"], matte)
        self.assertEqual(assignments["P0002"]["material_id"], matte)
        self.assertEqual(
            assignments["P0001"]["parameters"],
            assignments["P0002"]["parameters"],
        )
        self.assertEqual(set(assignments["P0001"]["parameters"]), {"diffuse_tint"})
        binding = assignments["P0001"]["provenance"][
            "appearance_component_color_candidate"
        ]
        self.assertTrue(binding["same_material_id_as_h0"])
        self.assertEqual(binding["target_family"], "paint")
        self.assertEqual(binding["parameter_mutation_scope"], "reviewed_color3f_linear_only")
        self.assertNotIn("parameters", source["assignments"][0])

    def test_component_color_candidate_rejects_identity_change_or_unreviewed_profile(
        self,
    ) -> None:
        catalog = self._strict_catalog()
        matte = "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte"
        source = self._plan()
        for assignment in source["assignments"][:2]:
            assignment["material_id"] = matte
            assignment.pop("parameters", None)
        with self.assertRaisesRegex(
            ComponentMdlTournamentError,
            "change the selected MDL",
        ):
            build_component_color_candidate_plan(
                source_plan=source,
                component_id="AC_paint",
                member_part_ids=["P0001", "P0002"],
                material_id="mdl:Miscellaneous/Paint_Satin.mdl#Paint_Satin",
                target_srgb=[0.2, 0.45, 0.25],
                member_material_semantics=self._member_semantics(),
                catalog_materials_by_id=catalog,
            )

        custom = "mdl:Coatings/Custom.mdl#Custom"
        custom_catalog = {custom: self._catalog_record(custom)}
        for assignment in source["assignments"][:2]:
            assignment["material_id"] = custom
        with self.assertRaisesRegex(
            ComponentMdlTournamentError,
            "reviewed Paint/Metal tuning profile",
        ):
            build_component_color_candidate_plan(
                source_plan=source,
                component_id="AC_paint",
                member_part_ids=["P0001", "P0002"],
                material_id=custom,
                target_srgb=[0.2, 0.45, 0.25],
                member_material_semantics=self._member_semantics(),
                catalog_materials_by_id=custom_catalog,
            )

    def test_component_color_winner_requires_aggregate_gain_and_member_safety(
        self,
    ) -> None:
        h0 = self._component_score("AC_paint", 0.4, 0.4)
        winner = select_component_color_winner(
            component_id="AC_paint",
            h0_score=h0,
            h1_score=self._component_score("AC_paint", 0.45, 0.40),
        )
        self.assertEqual(winner["selected_candidate_id"], "H1")
        self.assertEqual(winner["reason_codes"], [])

        insufficient = select_component_color_winner(
            component_id="AC_paint",
            h0_score=h0,
            h1_score=self._component_score("AC_paint", 0.41, 0.41),
        )
        self.assertEqual(insufficient["selected_candidate_id"], "H0")
        self.assertIn("INSUFFICIENT_AGGREGATE_IMPROVEMENT", insufficient["reason_codes"])

        regression = select_component_color_winner(
            component_id="AC_paint",
            h0_score=h0,
            # Aggregate rises because P0001 dominates the pixels, while P0002
            # regresses by 0.04 and must veto H1.
            h1_score=self._component_score(
                "AC_paint",
                0.48,
                0.36,
                first_pixels=900,
                second_pixels=100,
            ),
        )
        self.assertEqual(regression["selected_candidate_id"], "H0")
        self.assertIn("MEMBER_REGRESSION_ABOVE_MAXIMUM", regression["reason_codes"])

    def test_component_color_winner_rejects_mismatched_or_tampered_scores(self) -> None:
        h0 = self._component_score("AC_paint", 0.4, 0.4)
        mismatched = self._component_score("AC_other", 0.5, 0.5)
        with self.assertRaisesRegex(ComponentMdlTournamentError, "different component"):
            select_component_color_winner(
                component_id="AC_paint",
                h0_score=h0,
                h1_score=mismatched,
            )
        tampered = self._component_score("AC_paint", 0.5, 0.5)
        tampered["appearance_score"] = 0.99
        with self.assertRaisesRegex(ComponentMdlTournamentError, "internally consistent"):
            select_component_color_winner(
                component_id="AC_paint",
                h0_score=h0,
                h1_score=tampered,
            )

    def test_rebind_accepts_only_exact_hash_bound_component_color_h1(self) -> None:
        catalog = self._strict_catalog()
        matte = "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte"
        identity_plan = self._plan()
        for assignment in identity_plan["assignments"][:2]:
            assignment["material_id"] = matte
            assignment.pop("parameters", None)
        color_plan = build_component_color_candidate_plan(
            source_plan=identity_plan,
            component_id="AC_paint",
            member_part_ids=["P0001", "P0002"],
            material_id=matte,
            target_srgb=[0.2, 0.45, 0.25],
            member_material_semantics=self._member_semantics(),
            catalog_materials_by_id=catalog,
        )
        selection = select_component_color_winner(
            component_id="AC_paint",
            h0_score=self._component_score("AC_paint", 0.4, 0.4),
            h1_score=self._component_score("AC_paint", 0.45, 0.40),
        )
        parameters = color_plan["assignments"][0]["parameters"]
        source_audit = {
            "schema_version": "qwen-part-id-material-audit/v1",
            "assignment_unit": "part_id",
            "base_plan_sha256": "policy-hash",
            "output_plan_sha256": "identity-hash",
            "parts": [
                {
                    "part_id": "P0001",
                    "status": "independently_selected",
                    "material_id": matte,
                },
                {
                    "part_id": "P0002",
                    "status": "independently_selected",
                    "material_id": matte,
                },
                {
                    "part_id": "P0003",
                    "status": "unobserved_preserved",
                    "material_id": "mdl:Plastics/Plastic_ABS.mdl#Plastic_ABS",
                },
            ],
            "summary": {
                "part_count": 3,
                "independently_selected_count": 2,
                "unobserved_preserved_count": 1,
                "exact_cover": True,
            },
        }
        tournament_audit = {
            "components": [
                {
                    "component_id": "AC_paint",
                    "member_part_ids": ["P0001", "P0002"],
                    "baseline_material_id": matte,
                    "selected_material_id": matte,
                    "selection_status": "BASELINE_RETAINED",
                }
            ],
            "component_color_tournament": {
                "schema_version": "qwen-appearance-component-color-tournament/v1",
                "source_identity_plan_sha256": self._sha256(identity_plan),
                "final_plan_sha256": self._sha256(color_plan),
                "components": [
                    {
                        "component_id": "AC_paint",
                        "member_part_ids": ["P0001", "P0002"],
                        "material_id": matte,
                        "selected_candidate_id": "H1",
                        "parameters": parameters,
                        "source_plan_sha256": self._sha256(identity_plan),
                        "color_candidate_plan_sha256": self._sha256(color_plan),
                        "selection": selection,
                    }
                ],
            },
        }
        rebound = rebind_part_id_material_audit_for_component_mdl_tournament(
            source_audit=source_audit,
            final_plan=color_plan,
            tournament_audit=tournament_audit,
        )
        by_part = {row["part_id"]: row for row in rebound["parts"]}
        self.assertEqual(by_part["P0001"]["parameters"], parameters)
        self.assertTrue(
            by_part["P0001"]["appearance_component_color_tournament"][
                "parameter_mutation_allowed"
            ]
        )
        summary = rebound["appearance_component_actual_mdl_tournament"]
        self.assertEqual(summary["color_h1_component_count"], 1)
        self.assertEqual(summary["color_h1_part_count"], 2)
        self.assertEqual(
            summary["mdl_parameter_mutation_scope"],
            "reviewed_component_color3f_linear_only",
        )

        without_authorization = {"components": tournament_audit["components"]}
        with self.assertRaisesRegex(
            ComponentMdlTournamentError,
            "immutable MDL binding",
        ):
            rebind_part_id_material_audit_for_component_mdl_tournament(
                source_audit=source_audit,
                final_plan=color_plan,
                tournament_audit=without_authorization,
            )

        tampered = json.loads(json.dumps(tournament_audit))
        tampered["component_color_tournament"]["components"][0]["parameters"][
            "diffuse_tint"
        ] = [1.0, 0.0, 0.0]
        with self.assertRaisesRegex(
            ComponentMdlTournamentError,
            "differ from their color authorization",
        ):
            rebind_part_id_material_audit_for_component_mdl_tournament(
                source_audit=source_audit,
                final_plan=color_plan,
                tournament_audit=tampered,
            )

    def test_rebinds_part_id_audit_after_component_winner(self) -> None:
        source_plan = self._plan()
        final_plan = build_component_candidate_plan(
            source_plan=source_plan,
            component_id="AC_green",
            member_part_ids=["P0001", "P0002"],
            material_id="mdl:Natural/Grass_Countryside.mdl#Grass_Countryside",
        )
        source_audit = {
            "schema_version": "qwen-part-id-material-audit/v1",
            "assignment_unit": "part_id",
            "palette_fusion_used": False,
            "base_plan_sha256": "policy-hash",
            "output_plan_sha256": "old-hash",
            "parts": [
                {
                    "part_id": "P0001",
                    "status": "independently_selected",
                    "material_id": "mdl:Metals/Steel_Cast.mdl#Steel_Cast",
                },
                {
                    "part_id": "P0002",
                    "status": "independently_selected",
                    "material_id": "mdl:Metals/Steel_Cast.mdl#Steel_Cast",
                },
                {
                    "part_id": "P0003",
                    "status": "unobserved_preserved",
                    "material_id": "mdl:Plastics/Plastic_ABS.mdl#Plastic_ABS",
                },
            ],
            "summary": {
                "part_count": 3,
                "independently_selected_count": 2,
                "unobserved_preserved_count": 1,
                "exact_cover": True,
            },
        }
        tournament_audit = {
            "components": [
                {
                    "component_id": "AC_green",
                    "member_part_ids": ["P0001", "P0002"],
                    "baseline_material_id": "mdl:Metals/Steel_Cast.mdl#Steel_Cast",
                    "selected_material_id": (
                        "mdl:Natural/Grass_Countryside.mdl#Grass_Countryside"
                    ),
                    "selection_status": "ACTUAL_CAD_RENDER_WINNER",
                }
            ]
        }
        rebound = rebind_part_id_material_audit_for_component_mdl_tournament(
            source_audit=source_audit,
            final_plan=final_plan,
            tournament_audit=tournament_audit,
        )
        rows = {row["part_id"]: row for row in rebound["parts"]}
        self.assertEqual(
            rows["P0001"]["material_id"],
            "mdl:Natural/Grass_Countryside.mdl#Grass_Countryside",
        )
        self.assertEqual(
            rows["P0003"]["material_id"],
            "mdl:Plastics/Plastic_ABS.mdl#Plastic_ABS",
        )
        self.assertEqual(
            rebound["appearance_component_actual_mdl_tournament"]
            ["winner_part_count"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
