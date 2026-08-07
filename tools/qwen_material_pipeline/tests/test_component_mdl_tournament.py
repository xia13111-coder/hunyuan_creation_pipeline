from __future__ import annotations

import unittest

from qwen_material_pipeline.materials.component_mdl_tournament import (
    build_component_candidate_plan,
    component_candidate_material_ids,
    rebind_part_id_material_audit_for_component_mdl_tournament,
    select_component_mdl_winner,
)


class ComponentMdlTournamentTests(unittest.TestCase):
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
