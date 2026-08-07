from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asset_pipeline.visual_materials.orchestrator import (
    _run_dominant_assembly_membership_tournaments,
    _validate_quality_repair_dominant_assembly_cohorts,
)
from asset_pipeline.visual_materials.config import canonical_sha256
from qwen_material_pipeline.materials.membership_tournament import (
    M0_CANDIDATE,
    M1_CANDIDATE,
    MembershipTournamentError,
    build_membership_candidate_plans,
    discover_dominant_assembly_cohorts,
    membership_exclusions_by_group,
    select_membership_candidate,
)


class MembershipTournamentTests(unittest.TestCase):
    maxDiff = None

    def _palette(self) -> dict:
        return {
            "schema_version": "qwen-multiview-palette-fusion/v1",
            "canonical_palette": {
                "groups": [
                    {
                        "group_id": "G07",
                        "sources": [
                            {
                                "view_id": "front",
                                "local_group_id": "L01",
                            },
                            {
                                "view_id": "side",
                                "local_group_id": "L04",
                            },
                            {
                                "view_id": "top",
                                "local_group_id": "L02",
                            },
                        ],
                    }
                ]
            },
            "view_group_id_maps": {
                "front": {"L01": "G07"},
                "side": {"L04": "G07"},
                "top": {"L02": "G07"},
            },
        }

    def _plan(self) -> dict:
        cohort_id = "a" * 64
        contract_sha256 = "b" * 64
        input_hashes = {
            key: character * 64
            for key, character in (
                ("baseline_plan_sha256", "1"),
                ("quality_report_sha256", "2"),
                ("palette_fusion_sha256", "3"),
                ("spatial_report_sha256", "4"),
                ("spatial_gate_audit_sha256", "5"),
                ("mapping_consensus_sha256", "6"),
                ("geometry_risk_sha256", "7"),
                ("group_materials_sha256", "8"),
                ("registry_sha256", "9"),
            )
        }

        def assignment(
            part_id: str,
            *,
            role: str,
            baseline_material_id: str,
        ) -> dict:
            return {
                "part_id": part_id,
                "material_id": "mdl:NVIDIA/Materials/Base/Paints/Green",
                "parameters": {},
                "face_subsets": [],
                "provenance": {
                    "canonical_group_id": "G07",
                    "dominant_assembly_cohort": {
                        "schema_version": "qwen-dominant-assembly-cohort/v1",
                        "candidate_kind": "dominant_assembly",
                        "cohort_id": cohort_id,
                        "contract_sha256": contract_sha256,
                        "canonical_group_id": "G07",
                        "assembly_path": "/World/Assembly",
                        "source_visual_stable_properties_signature_sha256": ("c" * 64),
                        "anchor_part_ids": ["P0001"],
                        "anchor_supporting_view_ids": ["front", "side"],
                        "anchor_child_branches": ["body"],
                        "cohort_part_ids": ["P0001", "P0002", "P0003"],
                        "expanded_member_part_ids": ["P0002", "P0003"],
                        "member_role": role,
                        "membership_status": ("PROVISIONAL_PENDING_GROUP_TOURNAMENT"),
                        "baseline_material_id": baseline_material_id,
                        "input_hashes": copy.deepcopy(input_hashes),
                    },
                },
            }

        return {
            "schema_version": "1.0",
            "assignments": [
                assignment(
                    "P0001",
                    role="strict_spatial_anchor",
                    baseline_material_id=("mdl:NVIDIA/Materials/Base/Paints/Green"),
                ),
                assignment(
                    "P0002",
                    role="expanded_member",
                    baseline_material_id=("mdl:NVIDIA/Materials/Base/Metals/Steel"),
                ),
                assignment(
                    "P0003",
                    role="expanded_member",
                    baseline_material_id=("mdl:NVIDIA/Materials/Base/Plastics/Black"),
                ),
            ],
            "provenance": {"asset_sha256": "d" * 64},
        }

    def _round(self) -> tuple[list[dict], dict]:
        return build_membership_candidate_plans(
            source_plan=self._plan(),
            palette_fusion=self._palette(),
            cohort_id="a" * 64,
        )

    def _rare_pair_plan(self) -> dict:
        plan = self._plan()
        plan["assignments"] = plan["assignments"][:2]
        for assignment in plan["assignments"]:
            cohort = assignment["provenance"]["dominant_assembly_cohort"]
            cohort.update(
                {
                    "candidate_kind": "rare_source_identity_pair",
                    "proposal_policy": (
                        "single_strict_anchor_bounded_signature_sibling/v1"
                    ),
                    "cohort_id": "e" * 64,
                    "contract_sha256": "f" * 64,
                    "anchor_part_ids": ["P0001"],
                    "anchor_supporting_view_ids": ["front"],
                    "anchor_child_branches": ["body"],
                    "cohort_part_ids": ["P0001", "P0002"],
                    "expanded_member_part_ids": ["P0002"],
                }
            )
        return plan

    def _quality(
        self,
        contract: dict,
        *,
        aggregate_status: str,
        appearance_score: float,
        render_share: float,
        view_statuses: dict[str, str] | None = None,
        non_target_excess: float = 0.0,
    ) -> dict:
        view_statuses = view_statuses or {}
        views = []
        for view_id in contract["reference_view_ids"]:
            local_group_id = contract["target_local_group_ids_by_view"][view_id]
            status = view_statuses.get(view_id, aggregate_status)
            views.append(
                {
                    "reference_view_id": view_id,
                    "status": status,
                    "reference": {
                        "trusted_evidence": {
                            "usable": True,
                            "target_group_filter_applied": True,
                            "target_local_group_id": local_group_id,
                        }
                    },
                    "material_color": {
                        "trusted_evidence_group_recall": {
                            "groups": [
                                {
                                    "group_id": local_group_id,
                                    "render_color_bins": ["green"],
                                    "reference_color_share": 0.70,
                                    "observed_render_share": render_share,
                                }
                            ]
                        },
                        "unreferenced_render_chromatic_mass": {
                            "bins": [
                                {
                                    "color_bin": "green",
                                    "excess_share": 0.0,
                                },
                                {
                                    "color_bin": "red",
                                    "excess_share": non_target_excess,
                                },
                            ]
                        },
                    },
                    "material_appearance_score": appearance_score,
                }
            )
        return {
            "schema_version": "qwen-reference-render-comparison/v1",
            "inputs": {
                "comparison_scope": {
                    "mode": "canonical_group_local",
                    "target_group_id": contract["canonical_group_id"],
                    "target_part_ids": contract["cohort_part_ids"],
                    "reference_view_ids": contract["reference_view_ids"],
                }
            },
            "aggregate": {
                "status": aggregate_status,
                "material_appearance_score": appearance_score,
            },
            "views": views,
        }

    def _bundles(
        self,
        planned: list[dict],
        contract: dict,
        *,
        m0_quality: dict,
        m1_quality: dict,
    ) -> list[dict]:
        quality_by_mode = {
            M0_CANDIDATE: m0_quality,
            M1_CANDIDATE: m1_quality,
        }
        return [
            {
                "candidate_id": candidate["candidate_id"],
                "plan": candidate["plan"],
                "quality_report": quality_by_mode[candidate["membership_mode"]],
            }
            for candidate in planned
        ]

    def test_discovers_hash_consistent_complete_cohort(self) -> None:
        cohorts = discover_dominant_assembly_cohorts(
            plan=self._plan(),
            palette_fusion=self._palette(),
        )
        self.assertEqual(len(cohorts), 1)
        self.assertEqual(cohorts[0]["anchor_part_ids"], ["P0001"])
        self.assertEqual(
            cohorts[0]["expanded_member_part_ids"],
            ["P0002", "P0003"],
        )
        self.assertEqual(
            cohorts[0]["target_local_group_ids_by_view"],
            {"front": "L01", "side": "L04", "top": "L02"},
        )

    def test_quality_repair_boundary_accepts_both_sealed_cohort_kinds(self) -> None:
        cases = (
            (self._plan(), "dominant_assembly_cohort_expansion"),
            (
                self._rare_pair_plan(),
                "single_strict_anchor_bounded_signature_sibling",
            ),
        )
        for plan, lane_name in cases:
            with self.subTest(lane=lane_name):
                assignments = plan["assignments"]
                first_record = copy.deepcopy(
                    assignments[0]["provenance"]["dominant_assembly_cohort"]
                )
                first_record.pop("member_role")
                first_record.pop("baseline_material_id")
                first_record["material_id"] = assignments[0]["material_id"]
                first_record["accepted_part_ids"] = list(
                    first_record["cohort_part_ids"]
                )
                first_record.pop("contract_sha256")
                first_record["contract_sha256"] = canonical_sha256(first_record)
                for assignment in assignments:
                    assignment["provenance"]["dominant_assembly_cohort"][
                        "contract_sha256"
                    ] = first_record["contract_sha256"]

                changes = {}
                lanes = {}
                anchor_part_ids = set(first_record["anchor_part_ids"])
                for assignment in assignments:
                    part_id = assignment["part_id"]
                    member_role = (
                        "strict_spatial_anchor"
                        if part_id in anchor_part_ids
                        else "expanded_member"
                    )
                    changes[part_id] = {
                        "part_id": part_id,
                        "canonical_group_id": "G07",
                        "material_id": assignment["material_id"],
                        "old_material_id": assignment["provenance"][
                            "dominant_assembly_cohort"
                        ]["baseline_material_id"],
                        "dominant_assembly_cohort_id": first_record["cohort_id"],
                        "dominant_assembly_member_role": member_role,
                    }
                    lanes[part_id] = {
                        "part_id": part_id,
                        "canonical_group_id": "G07",
                        "lane": lane_name,
                    }

                validated = _validate_quality_repair_dominant_assembly_cohorts(
                    plan=plan,
                    audit={"dominant_assembly_cohorts": [first_record]},
                    palette_fusion=self._palette(),
                    expected_input_hashes=first_record["input_hashes"],
                    changes_by_part=changes,
                    localization_lanes_by_part=lanes,
                )
                self.assertEqual(set(validated), set(changes))

    def test_quality_repair_boundary_rejects_tampered_cohort_audit(self) -> None:
        plan = self._plan()
        record = copy.deepcopy(
            plan["assignments"][0]["provenance"]["dominant_assembly_cohort"]
        )
        record.pop("member_role")
        record.pop("baseline_material_id")
        record["material_id"] = plan["assignments"][0]["material_id"]
        record["accepted_part_ids"] = list(record["cohort_part_ids"])
        record.pop("contract_sha256")
        record["contract_sha256"] = canonical_sha256(record)
        record["assembly_path"] = "/World/Tampered"
        with self.assertRaisesRegex(RuntimeError, "cohort hash is invalid"):
            _validate_quality_repair_dominant_assembly_cohorts(
                plan=plan,
                audit={"dominant_assembly_cohorts": [record]},
                palette_fusion=self._palette(),
                expected_input_hashes=record["input_hashes"],
                changes_by_part={},
                localization_lanes_by_part={
                    "P0001": {
                        "part_id": "P0001",
                        "canonical_group_id": "G07",
                        "lane": "dominant_assembly_cohort_expansion",
                    }
                },
            )

    def test_m0_restores_only_expanded_members(self) -> None:
        candidates, contract = self._round()
        self.assertEqual(
            contract["candidate_ids"][M0_CANDIDATE], candidates[0]["candidate_id"]
        )
        m0_assignments = {
            assignment["part_id"]: assignment
            for assignment in candidates[0]["plan"]["assignments"]
        }
        self.assertEqual(
            m0_assignments["P0001"]["material_id"],
            "mdl:NVIDIA/Materials/Base/Paints/Green",
        )
        self.assertEqual(
            m0_assignments["P0002"]["material_id"],
            "mdl:NVIDIA/Materials/Base/Metals/Steel",
        )
        self.assertEqual(
            m0_assignments["P0003"]["material_id"],
            "mdl:NVIDIA/Materials/Base/Plastics/Black",
        )

    def test_rare_two_member_pair_uses_one_anchor_but_same_render_gate(
        self,
    ) -> None:
        rare_plan = self._rare_pair_plan()
        cohorts = discover_dominant_assembly_cohorts(
            plan=rare_plan,
            palette_fusion=self._palette(),
        )
        self.assertEqual(len(cohorts), 1)
        self.assertEqual(
            cohorts[0]["candidate_kind"],
            "rare_source_identity_pair",
        )
        self.assertEqual(cohorts[0]["anchor_supporting_view_ids"], ["front"])
        self.assertEqual(
            cohorts[0]["minimum_anchor_supporting_view_count"],
            1,
        )
        planned, contract = build_membership_candidate_plans(
            source_plan=rare_plan,
            palette_fusion=self._palette(),
            cohort_id="e" * 64,
        )
        self.assertEqual(contract["cohort_part_ids"], ["P0001", "P0002"])
        self.assertEqual(
            contract["policy"]["minimum_improved_independent_view_count"],
            2,
        )
        m0_quality = self._quality(
            contract,
            aggregate_status="REVIEW",
            appearance_score=0.60,
            render_share=0.45,
        )
        m1_quality = self._quality(
            contract,
            aggregate_status="REVIEW",
            appearance_score=0.62,
            render_share=0.68,
        )
        output, audit = select_membership_candidate(
            contract=contract,
            candidates=self._bundles(
                planned,
                contract,
                m0_quality=m0_quality,
                m1_quality=m1_quality,
            ),
            palette_fusion=self._palette(),
        )
        self.assertEqual(audit["status"], "ACCEPTED_EXPANDED_COHORT")
        self.assertEqual(audit["candidate_kind"], "rare_source_identity_pair")
        selected = output["provenance"]["dominant_assembly_membership_tournaments"][0]
        self.assertEqual(
            selected["candidate_kind"],
            "rare_source_identity_pair",
        )

    def test_dominant_candidate_still_requires_two_anchor_views(self) -> None:
        plan = self._plan()
        for assignment in plan["assignments"]:
            assignment["provenance"]["dominant_assembly_cohort"][
                "anchor_supporting_view_ids"
            ] = ["front"]
        with self.assertRaisesRegex(
            MembershipTournamentError,
            "requires at least 2 anchor-supporting",
        ):
            discover_dominant_assembly_cohorts(
                plan=plan,
                palette_fusion=self._palette(),
            )

    def test_rare_candidate_must_be_exactly_anchor_plus_one_sibling(self) -> None:
        plan = self._plan()
        for assignment in plan["assignments"]:
            cohort = assignment["provenance"]["dominant_assembly_cohort"]
            cohort["candidate_kind"] = "rare_source_identity_pair"
            cohort["proposal_policy"] = (
                "single_strict_anchor_bounded_signature_sibling/v1"
            )
            cohort["anchor_supporting_view_ids"] = ["front"]
        with self.assertRaisesRegex(
            MembershipTournamentError,
            "rare pair must contain exactly one anchor",
        ):
            discover_dominant_assembly_cohorts(
                plan=plan,
                palette_fusion=self._palette(),
            )

    def test_rare_pair_does_not_relax_multiview_target_gate(self) -> None:
        rare_plan = self._rare_pair_plan()
        planned, contract = build_membership_candidate_plans(
            source_plan=rare_plan,
            palette_fusion=self._palette(),
            cohort_id="e" * 64,
        )
        m0_quality = self._quality(
            contract,
            aggregate_status="REVIEW",
            appearance_score=0.60,
            render_share=0.45,
        )
        m1_quality = self._quality(
            contract,
            aggregate_status="PASS",
            appearance_score=0.80,
            render_share=0.46,
        )
        # Only one view carries a material target improvement; aggregate PASS
        # cannot bypass the fixed two-independent-view membership gate.
        m1_quality["views"][0]["material_color"]["trusted_evidence_group_recall"][
            "groups"
        ][0]["observed_render_share"] = 0.68
        output, audit = select_membership_candidate(
            contract=contract,
            candidates=self._bundles(
                planned,
                contract,
                m0_quality=m0_quality,
                m1_quality=m1_quality,
            ),
            palette_fusion=self._palette(),
        )
        self.assertEqual(audit["status"], "REJECTED_EXPANSION_RESTORED_M0")
        self.assertEqual(audit["improved_independent_view_count"], 1)
        self.assertEqual(
            membership_exclusions_by_group(output),
            {"G07": {"P0002"}},
        )

    def test_rare_pair_does_not_relax_non_target_regression_gate(self) -> None:
        rare_plan = self._rare_pair_plan()
        planned, contract = build_membership_candidate_plans(
            source_plan=rare_plan,
            palette_fusion=self._palette(),
            cohort_id="e" * 64,
        )
        m0_quality = self._quality(
            contract,
            aggregate_status="REVIEW",
            appearance_score=0.60,
            render_share=0.45,
            non_target_excess=0.0,
        )
        m1_quality = self._quality(
            contract,
            aggregate_status="PASS",
            appearance_score=0.80,
            render_share=0.68,
            non_target_excess=0.051,
        )
        _, audit = select_membership_candidate(
            contract=contract,
            candidates=self._bundles(
                planned,
                contract,
                m0_quality=m0_quality,
                m1_quality=m1_quality,
            ),
            palette_fusion=self._palette(),
        )
        self.assertEqual(audit["status"], "REJECTED_EXPANSION_RESTORED_M0")
        self.assertIn(
            "NON_TARGET_CHROMATIC_EXCESS_LIMIT_EXCEEDED",
            audit["reason_codes"],
        )

    def test_accepts_expansion_from_two_view_improvement(self) -> None:
        planned, contract = self._round()
        m0_quality = self._quality(
            contract,
            aggregate_status="REVIEW",
            appearance_score=0.60,
            render_share=0.45,
        )
        m1_quality = self._quality(
            contract,
            aggregate_status="REVIEW",
            appearance_score=0.62,
            render_share=0.68,
        )
        output, audit = select_membership_candidate(
            contract=contract,
            candidates=self._bundles(
                planned,
                contract,
                m0_quality=m0_quality,
                m1_quality=m1_quality,
            ),
            palette_fusion=self._palette(),
        )
        self.assertEqual(audit["status"], "ACCEPTED_EXPANDED_COHORT")
        self.assertEqual(audit["improved_independent_view_count"], 3)
        self.assertEqual(audit["selected_membership_mode"], M1_CANDIDATE)
        self.assertEqual(membership_exclusions_by_group(output), {})

    def test_rejection_replays_m0_and_freezes_queue_exclusions(self) -> None:
        planned, contract = self._round()
        m0_quality = self._quality(
            contract,
            aggregate_status="REVIEW",
            appearance_score=0.60,
            render_share=0.45,
        )
        m1_quality = self._quality(
            contract,
            aggregate_status="REVIEW",
            appearance_score=0.61,
            render_share=0.48,
        )
        output, audit = select_membership_candidate(
            contract=contract,
            candidates=self._bundles(
                planned,
                contract,
                m0_quality=m0_quality,
                m1_quality=m1_quality,
            ),
            palette_fusion=self._palette(),
        )
        self.assertEqual(audit["status"], "REJECTED_EXPANSION_RESTORED_M0")
        self.assertTrue(audit["rejection_restores_m0"])
        assignments = {
            assignment["part_id"]: assignment for assignment in output["assignments"]
        }
        self.assertEqual(
            assignments["P0002"]["material_id"],
            "mdl:NVIDIA/Materials/Base/Metals/Steel",
        )
        self.assertEqual(
            membership_exclusions_by_group(output),
            {"G07": {"P0002", "P0003"}},
        )

    def test_any_view_regression_rejects_expansion(self) -> None:
        planned, contract = self._round()
        m0_quality = self._quality(
            contract,
            aggregate_status="REVIEW",
            appearance_score=0.60,
            render_share=0.45,
        )
        m1_quality = self._quality(
            contract,
            aggregate_status="PASS",
            appearance_score=0.75,
            render_share=0.68,
            view_statuses={"side": "FAIL"},
        )
        _, audit = select_membership_candidate(
            contract=contract,
            candidates=self._bundles(
                planned,
                contract,
                m0_quality=m0_quality,
                m1_quality=m1_quality,
            ),
            palette_fusion=self._palette(),
        )
        self.assertEqual(audit["status"], "REJECTED_EXPANSION_RESTORED_M0")
        self.assertIn("VIEW_STATUS_REGRESSION", audit["reason_codes"])

    def test_non_target_chromatic_excess_limit_rejects_expansion(self) -> None:
        planned, contract = self._round()
        m0_quality = self._quality(
            contract,
            aggregate_status="REVIEW",
            appearance_score=0.60,
            render_share=0.45,
            non_target_excess=0.01,
        )
        m1_quality = self._quality(
            contract,
            aggregate_status="PASS",
            appearance_score=0.75,
            render_share=0.68,
            non_target_excess=0.051,
        )
        _, audit = select_membership_candidate(
            contract=contract,
            candidates=self._bundles(
                planned,
                contract,
                m0_quality=m0_quality,
                m1_quality=m1_quality,
            ),
            palette_fusion=self._palette(),
        )
        self.assertEqual(audit["status"], "REJECTED_EXPANSION_RESTORED_M0")
        self.assertIn(
            "NON_TARGET_CHROMATIC_EXCESS_LIMIT_EXCEEDED",
            audit["reason_codes"],
        )

    def test_palette_mapping_tampering_fails_closed(self) -> None:
        planned, contract = self._round()
        palette = self._palette()
        palette["view_group_id_maps"]["side"]["L04"] = "G99"
        quality = self._quality(
            contract,
            aggregate_status="PASS",
            appearance_score=0.80,
            render_share=0.69,
        )
        with self.assertRaisesRegex(
            MembershipTournamentError,
            "palette fusion hash mismatch",
        ):
            select_membership_candidate(
                contract=contract,
                candidates=self._bundles(
                    planned,
                    contract,
                    m0_quality=quality,
                    m1_quality=quality,
                ),
                palette_fusion=palette,
            )

    def test_incomplete_render_evidence_restores_m0(self) -> None:
        planned, contract = self._round()
        valid_quality = self._quality(
            contract,
            aggregate_status="PASS",
            appearance_score=0.80,
            render_share=0.69,
        )
        invalid_quality = copy.deepcopy(valid_quality)
        invalid_quality["views"][0]["reference"]["trusted_evidence"]["usable"] = False
        output, audit = select_membership_candidate(
            contract=contract,
            candidates=self._bundles(
                planned,
                contract,
                m0_quality=valid_quality,
                m1_quality=invalid_quality,
            ),
            palette_fusion=self._palette(),
        )
        self.assertEqual(audit["status"], "REJECTED_EXPANSION_RESTORED_M0")
        self.assertEqual(
            audit["reason_codes"],
            ["CANDIDATE_RENDER_EVIDENCE_INVALID"],
        )
        self.assertEqual(
            membership_exclusions_by_group(output),
            {"G07": {"P0002", "P0003"}},
        )

    def test_divergent_member_contract_is_rejected(self) -> None:
        plan = self._plan()
        plan["assignments"][2]["provenance"]["dominant_assembly_cohort"][
            "assembly_path"
        ] = "/World/OtherAssembly"
        with self.assertRaisesRegex(
            MembershipTournamentError,
            "divergent member contracts",
        ):
            discover_dominant_assembly_cohorts(
                plan=plan,
                palette_fusion=self._palette(),
            )

    def test_orchestrator_renders_both_candidates_before_freezing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_plan = root / "plan.json"
            palette_path = root / "palette.json"
            source_plan.write_text(json.dumps(self._plan()), encoding="utf-8")
            palette_path.write_text(json.dumps(self._palette()), encoding="utf-8")
            planned, contract = self._round()
            self.assertEqual(len(planned), 2)

            current_apply = root / "current_apply.json"
            current_apply.write_text(
                json.dumps({"applied_count": 3, "face_subset_count": 0}),
                encoding="utf-8",
            )
            current_quality = root / "current_quality.json"
            current_quality.write_text("{}", encoding="utf-8")
            rendered_registry = root / "rendered_registry.json"
            rendered_registry.write_text("{}", encoding="utf-8")
            current_quality_registry = root / "quality_registry.json"
            current_quality_registry.write_text("{}", encoding="utf-8")
            reference_manifest = root / "reference_manifest.json"
            reference_manifest.write_text("{}", encoding="utf-8")
            source = root / "source.usd"
            source.write_text("usd", encoding="utf-8")
            current_look = root / "current.usda"
            current_look.write_text("look", encoding="utf-8")
            commands: list[list[str]] = []

            def fake_run(command: list[str], **_kwargs: object) -> None:
                commands.append(list(command))
                if "apply" in command:
                    look = Path(command[command.index("--output") + 1])
                    report = Path(command[command.index("--report") + 1])
                    look.parent.mkdir(parents=True, exist_ok=True)
                    look.write_text("look", encoding="utf-8")
                    report.write_text(
                        json.dumps(
                            {
                                "applied_count": 3,
                                "face_subset_count": 0,
                            }
                        ),
                        encoding="utf-8",
                    )
                elif "registry" in command:
                    output = Path(command[command.index("--output") + 1])
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text("{}", encoding="utf-8")
                elif "render" in command:
                    output_dir = Path(command[command.index("--output-dir") + 1])
                    output_dir.mkdir(parents=True, exist_ok=True)
                    (output_dir / "part_registry.rendered.json").write_text(
                        "{}",
                        encoding="utf-8",
                    )
                elif "compare" in command:
                    output = Path(command[command.index("--output") + 1])
                    output.parent.mkdir(parents=True, exist_ok=True)
                    if "--target-group-id" in command:
                        is_m0 = "_m0" in str(output)
                        quality = self._quality(
                            contract,
                            aggregate_status="REVIEW",
                            appearance_score=0.60 if is_m0 else 0.62,
                            render_share=0.45 if is_m0 else 0.68,
                        )
                    else:
                        quality = {
                            "schema_version": ("qwen-reference-render-comparison/v1"),
                            "aggregate": {"status": "PASS"},
                        }
                    output.write_text(json.dumps(quality), encoding="utf-8")

            with patch(
                "asset_pipeline.visual_materials.tournaments."
                "_validated_exact_mdl_tournament_mapping",
                return_value={
                    "front": "right",
                    "side": "front",
                    "top": "top",
                },
            ):
                result = _run_dominant_assembly_membership_tournaments(
                    source_plan_path=source_plan,
                    source=source,
                    apply_asset=source,
                    apply_subcommand="apply",
                    apply_asset_flag="--asset-usd",
                    effective_catalog=root / "catalog.json",
                    material_root=root / "materials",
                    rendered_registry=rendered_registry,
                    current_look_usd=current_look,
                    current_apply_report=current_apply,
                    current_quality_report=current_quality,
                    current_quality_rendered_registry=current_quality_registry,
                    reference_manifest=reference_manifest,
                    palette_fusion_path=palette_path,
                    tournament_dir=root / "membership",
                    tournament_view_map=root / "membership" / "view_map.json",
                    output_plan=root / "selected_plan.json",
                    output_audit=root / "membership_audit.json",
                    trusted_mapping={},
                    mapped_render_resolution=256,
                    render_rt_subframes=2,
                    analysis_up_axis="z",
                    analysis_front_axis="-y",
                    qwen_python=root / "qwen-python",
                    isaac=root / "isaac-python",
                    instance_root_count=0,
                    applied_count=3,
                    include_policy_fallback=False,
                    log_cb=None,
                    command_runner=fake_run,
                )
            self.assertEqual(result["status"], "COMPLETED")
            self.assertEqual(result["selected_expanded_cohort_count"], 1)
            self.assertEqual(
                sum("apply" in command for command in commands),
                2,
            )
            self.assertEqual(
                sum(
                    "compare" in command and "--target-group-id" in command
                    for command in commands
                ),
                2,
            )
            self.assertTrue(Path(result["audit"]).is_file())


if __name__ == "__main__":
    unittest.main()
