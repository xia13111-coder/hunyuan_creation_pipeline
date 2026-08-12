from __future__ import annotations

import unittest

from qwen_material_pipeline.materials.component_mdl_tournament import (
    ComponentMdlTournamentError,
    build_component_candidate_plan,
    component_candidate_material_ids,
    rebind_part_id_material_audit_for_component_mdl_tournament,
    select_component_mdl_winner,
)


class ComponentMdlTournamentTests(unittest.TestCase):
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
