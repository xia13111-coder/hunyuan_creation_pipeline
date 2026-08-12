from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qwen_material_pipeline.core.material_stage_contract import (
    material_stage_contract_document,
)

import asset_pipeline
from asset_pipeline import visual_materials
from asset_pipeline.visual_materials.orchestrator import (
    _archive_partial_live_resume_downstream_artifacts,
    _archive_stale_policy_exact_cover_checkpoint,
    _appearance_baseline_safety_reason,
    _complete_coverage_assignment_statuses,
    _evaluate_part_id_quality_gate,
    _multigroup_local_compare_command,
    _palette_group_disagreement_contract_applies,
    _policy_checkpoint_matches_requested_overrides,
    _prepare_live_material_catalog,
    _quality_can_measure_lighting_statistics,
    _quality_has_lighting_normalized_groups,
    _run_qwen_mvinverse_with_recovery,
    _validated_exact_mdl_tournament_mapping,
    _verified_partial_live_resume_available,
)
from asset_pipeline.jobs.material import (
    CONFIG_SCHEMA_VERSION,
    ISOLATED_ENV_REMOVE,
    load_visual_material_config,
    parse_visual_references,
    run_assign_visual_materials_job,
)
from asset_pipeline.visual_materials.config import canonical_sha256
from asset_pipeline.visual_materials.quality import (
    part_id_quality_scope_from_camera_alignment,
)


class VisualMaterialBridgeTests(unittest.TestCase):
    def test_part_id_lock_ignores_palette_group_disagreement_contract(self) -> None:
        self.assertFalse(_palette_group_disagreement_contract_applies("part_id"))
        self.assertTrue(_palette_group_disagreement_contract_applies("palette_group"))

    def test_part_id_quality_gate_ignores_only_palette_group_completeness(
        self,
    ) -> None:
        report = {
            "aggregate": {
                "status": "FAIL",
                "comparable_view_count": 2,
                "material_appearance_score": 0.74,
            },
            "views": [
                {
                    "reference_view_id": "front",
                    "render_view_id": "right",
                    "status": "FAIL",
                    "material_appearance_score": 0.78,
                    "reasons": ["trusted_palette_group_missing_from_render"],
                },
                {
                    "reference_view_id": "side",
                    "render_view_id": "front",
                    "status": "FAIL",
                    "material_appearance_score": 0.70,
                    "reasons": ["trusted_palette_group_missing_from_render"],
                },
                {
                    "reference_view_id": "iso",
                    "status": "UNSCORABLE",
                    "material_appearance_score": None,
                    "reasons": ["no_trusted_accepted_evidence_boxes"],
                },
            ],
        }
        gate = _evaluate_part_id_quality_gate(
            report,
            minimum_aggregate_appearance_score=0.62,
            minimum_view_appearance_score=0.55,
        )
        self.assertEqual(gate["status"], "PASS")
        self.assertTrue(gate["acceptance_allowed"])
        self.assertEqual(
            gate["limitations"],
            [
                {
                    "code": "UNSCORABLE_REFERENCE_VIEWS",
                    "reference_view_ids": ["iso"],
                }
            ],
        )

    def test_part_id_quality_gate_rejects_non_palette_view_failure(self) -> None:
        report = {
            "aggregate": {
                "status": "FAIL",
                "comparable_view_count": 2,
                "material_appearance_score": 0.74,
            },
            "views": [
                {
                    "reference_view_id": "front",
                    "render_view_id": "right",
                    "status": "FAIL",
                    "material_appearance_score": 0.78,
                    "reasons": ["trusted_palette_group_missing_from_render"],
                },
                {
                    "reference_view_id": "side",
                    "render_view_id": "front",
                    "status": "FAIL",
                    "material_appearance_score": 0.70,
                    "reasons": ["material_color_score_below_threshold"],
                },
            ],
        }
        gate = _evaluate_part_id_quality_gate(
            report,
            minimum_aggregate_appearance_score=0.62,
            minimum_view_appearance_score=0.55,
        )
        self.assertEqual(gate["status"], "FAIL_CLOSED")
        self.assertFalse(gate["acceptance_allowed"])
        self.assertIn(
            "NON_PALETTE_VIEW_FAILURE_REASONS_PRESENT",
            gate["reason_codes"],
        )

    def test_part_id_quality_gate_enforces_camera_anchors_not_local_only_views(
        self,
    ) -> None:
        camera_scope = part_id_quality_scope_from_camera_alignment(
            {
                "policy": "two_layer_box_first_part_id_alignment/v2",
                "anchor_view_ids": ["front", "iso"],
                "views": {
                    "front": {
                        "tier": "usable_box_correspondence",
                        "observation_eligible": True,
                    },
                    "iso": {
                        "tier": "downweighted_box_correspondence",
                        "observation_eligible": True,
                    },
                    "side": {
                        "tier": "local_box_refinement_only",
                        "observation_eligible": True,
                    },
                    "top": {
                        "tier": "local_box_refinement_only",
                        "observation_eligible": True,
                    },
                },
            }
        )
        report = {
            "part_id_quality_scope": camera_scope,
            "aggregate": {
                "status": "FAIL",
                "comparable_view_count": 4,
                "material_appearance_score": 0.65,
            },
            "views": [
                {
                    "reference_view_id": "front",
                    "render_view_id": "front",
                    "status": "REVIEW",
                    "material_appearance_score": 0.69,
                    "reasons": ["foreground_value_similarity_below_pass_threshold"],
                },
                {
                    "reference_view_id": "iso",
                    "render_view_id": "iso",
                    "status": "PASS",
                    "material_appearance_score": 0.72,
                    "reasons": [],
                },
                {
                    "reference_view_id": "side",
                    "render_view_id": "side",
                    "status": "FAIL",
                    "material_appearance_score": 0.51,
                    "reasons": ["trusted_dominant_family_mass_deficit"],
                },
                {
                    "reference_view_id": "top",
                    "render_view_id": "top",
                    "status": "FAIL",
                    "material_appearance_score": 0.65,
                    "reasons": ["trusted_dominant_family_mass_deficit"],
                },
            ],
        }

        gate = _evaluate_part_id_quality_gate(
            report,
            minimum_aggregate_appearance_score=0.62,
            minimum_view_appearance_score=0.55,
        )

        self.assertEqual(gate["status"], "PASS")
        self.assertEqual(
            gate["view_scope"]["enforced_reference_view_ids"],
            ["front", "iso"],
        )
        self.assertAlmostEqual(
            gate["measurements"]["aggregate_appearance_score"],
            0.705,
        )
        self.assertEqual(gate["unsupported_view_failures"], {})
        self.assertIn(
            {
                "code": "LOCAL_EVIDENCE_ONLY_VIEWS_EXCLUDED_FROM_GLOBAL_GATE",
                "reference_view_ids": ["side", "top"],
            },
            gate["limitations"],
        )

    def test_part_id_quality_gate_requires_every_camera_anchor(self) -> None:
        report = {
            "part_id_quality_scope": {
                "schema_version": "asset-pipeline-part-id-quality-view-scope/v1",
                "mode": "camera_anchor_views",
                "source_camera_policy": "two_layer_box_first_part_id_alignment/v2",
                "enforced_reference_view_ids": ["front", "iso"],
                "local_evidence_only_reference_view_ids": ["side"],
                "rejected_reference_view_ids": [],
            },
            "aggregate": {
                "status": "PASS",
                "comparable_view_count": 2,
                "material_appearance_score": 0.8,
            },
            "views": [
                {
                    "reference_view_id": "front",
                    "render_view_id": "front",
                    "status": "PASS",
                    "material_appearance_score": 0.8,
                    "reasons": [],
                },
                {
                    "reference_view_id": "side",
                    "render_view_id": "side",
                    "status": "PASS",
                    "material_appearance_score": 0.8,
                    "reasons": [],
                },
            ],
        }

        gate = _evaluate_part_id_quality_gate(
            report,
            minimum_aggregate_appearance_score=0.62,
            minimum_view_appearance_score=0.55,
        )

        self.assertEqual(gate["status"], "FAIL_CLOSED")
        self.assertIn(
            "REQUIRED_CAMERA_ANCHOR_VIEW_MISSING_OR_UNSCORABLE",
            gate["reason_codes"],
        )
        self.assertIn("INSUFFICIENT_COMPARABLE_VIEWS", gate["reason_codes"])

    def test_part_id_quality_gate_rejects_low_appearance_score(self) -> None:
        report = {
            "aggregate": {
                "status": "FAIL",
                "comparable_view_count": 2,
                "material_appearance_score": 0.60,
            },
            "views": [
                {
                    "reference_view_id": view_id,
                    "render_view_id": view_id,
                    "status": "FAIL",
                    "material_appearance_score": score,
                    "reasons": ["trusted_palette_group_missing_from_render"],
                }
                for view_id, score in (("front", 0.70), ("side", 0.50))
            ],
        }
        gate = _evaluate_part_id_quality_gate(
            report,
            minimum_aggregate_appearance_score=0.62,
            minimum_view_appearance_score=0.55,
        )
        self.assertEqual(gate["status"], "FAIL_CLOSED")
        self.assertIn("AGGREGATE_APPEARANCE_BELOW_FLOOR", gate["reason_codes"])
        self.assertIn("VIEW_APPEARANCE_BELOW_FLOOR", gate["reason_codes"])

    def test_part_id_quality_gate_requires_supplied_coating_gate_to_pass(
        self,
    ) -> None:
        report = {
            "aggregate": {
                "status": "FAIL",
                "comparable_view_count": 2,
                "material_appearance_score": 0.75,
            },
            "views": [
                {
                    "reference_view_id": view_id,
                    "render_view_id": view_id,
                    "status": "FAIL",
                    "reasons": ["trusted_palette_group_missing_from_render"],
                    "material_appearance_score": 0.75,
                }
                for view_id in ("front", "side")
            ],
        }
        gate = _evaluate_part_id_quality_gate(
            report,
            minimum_aggregate_appearance_score=0.62,
            minimum_view_appearance_score=0.55,
            coating_consistency_audit={
                "coating_consistency_gate": {
                    "status": "FAIL_CLOSED",
                    "summary": {
                        "component_count": 1,
                        "constrained_part_count": 2,
                    },
                }
            },
        )
        self.assertEqual(gate["status"], "FAIL_CLOSED")
        self.assertIn(
            "COATING_CONSISTENCY_GATE_NOT_PASSED",
            gate["reason_codes"],
        )

    def test_part_id_provisional_exact_cover_authorizes_review_for_render_qa(
        self,
    ) -> None:
        self.assertEqual(
            _complete_coverage_assignment_statuses(
                material_assignment_unit="part_id",
                include_policy_fallback=True,
            ),
            {"auto", "approved", "review", "policy_fallback"},
        )
        self.assertEqual(
            _complete_coverage_assignment_statuses(
                material_assignment_unit="palette_group",
                include_policy_fallback=True,
            ),
            {"auto", "approved", "policy_fallback"},
        )

    def test_visual_policy_checkpoint_requires_every_requested_override(self) -> None:
        audit = {
            "policy": {
                "schema_version": "qwen-policy-exact-cover/v1",
                "source_visual_strategy": "neutralize_unverified",
                "semantic_rules": [],
                "default_strategy": "declared_material",
            }
        }
        requested = {
            "schema_version": "qwen-policy-exact-cover/v1",
            "source_visual_strategy": "neutralize_unverified",
            "semantic_rules": [],
        }
        self.assertTrue(
            _policy_checkpoint_matches_requested_overrides(
                audit=audit,
                requested_policy=requested,
            )
        )
        audit["policy"]["semantic_rules"] = [{"rule_id": "legacy"}]
        self.assertFalse(
            _policy_checkpoint_matches_requested_overrides(
                audit=audit,
                requested_policy=requested,
            )
        )

    def test_stale_policy_checkpoint_archive_is_reversible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            analysis = destination / "analysis"
            analysis.mkdir()
            policy_plan = analysis / "policy_exact_cover_plan.json"
            policy_audit = analysis / "policy_exact_cover_audit.json"
            repair_plan = analysis / "quality_repair_plan.json"
            for path in (policy_plan, policy_audit, repair_plan):
                path.write_text("{}", encoding="utf-8")

            archive = _archive_stale_policy_exact_cover_checkpoint(
                destination=destination,
                paths=(policy_plan, policy_audit, repair_plan),
                reason="test_policy_change",
            )

            self.assertFalse(policy_plan.exists())
            self.assertFalse(policy_audit.exists())
            self.assertFalse(repair_plan.exists())
            manifest = json.loads(
                (archive / "archive_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "COMPLETED")
            self.assertEqual(manifest["reason"], "test_policy_change")
            self.assertEqual(len(manifest["archived"]), 3)
            self.assertTrue(
                (archive / "analysis" / policy_plan.name).is_file()
            )

    def test_multigroup_compare_command_is_strictly_group_local(self) -> None:
        command = _multigroup_local_compare_command(
            qwen_python=Path("/runtime/qwen-python"),
            reference_manifest=Path("/run/reference_manifest.json"),
            rendered_registry=Path("/run/rendered_registry.json"),
            view_map=Path("/run/reference_view_map.json"),
            palette_fusion=Path("/run/palette_fusion.json"),
            group_id="G06",
            target_part_ids=["P0001", "P0002"],
            reference_view_ids=["front", "iso", "side", "top"],
            output=Path("/run/g06_quality.json"),
        )

        self.assertIn("--target-group-id", command)
        self.assertEqual(command[command.index("--target-group-id") + 1], "G06")
        self.assertEqual(command.count("--target-part-id"), 2)
        self.assertEqual(command.count("--target-entity-json"), 2)
        self.assertEqual(command.count("--target-reference-view-id"), 4)
        self.assertEqual(
            [
                command[index + 1]
                for index, value in enumerate(command)
                if value == "--target-reference-view-id"
            ],
            ["front", "iso", "side", "top"],
        )
        self.assertIn("--palette-fusion", command)
        self.assertEqual(
            command[command.index("--minimum-comparable-views") + 1],
            "4",
        )
        self.assertNotIn("--require-pass", command)

    def test_multigroup_compare_command_records_face_subset_target(self) -> None:
        command = _multigroup_local_compare_command(
            qwen_python=Path("/runtime/qwen-python"),
            reference_manifest=Path("/run/reference_manifest.json"),
            rendered_registry=Path("/run/rendered_registry.json"),
            view_map=Path("/run/reference_view_map.json"),
            palette_fusion=Path("/run/palette_fusion.json"),
            group_id="G04",
            target_part_ids=["P0001"],
            target_entities=[
                {
                    "entity_kind": "face_subset",
                    "part_id": "P0001",
                    "subset_name": "Cover",
                }
            ],
            reference_view_ids=["front", "top"],
            output=Path("/run/g04_quality.json"),
        )

        entity_json = command[command.index("--target-entity-json") + 1]
        self.assertEqual(
            json.loads(entity_json),
            {
                "entity_kind": "face_subset",
                "part_id": "P0001",
                "subset_name": "Cover",
            },
        )
        self.assertEqual(command.count("--target-reference-view-id"), 2)
        self.assertEqual(
            command[command.index("--minimum-comparable-views") + 1],
            "2",
        )

    def test_multigroup_compare_command_rejects_unsorted_parts(self) -> None:
        with self.assertRaisesRegex(ValueError, "sorted"):
            _multigroup_local_compare_command(
                qwen_python=Path("/runtime/qwen-python"),
                reference_manifest=Path("/run/reference_manifest.json"),
                rendered_registry=Path("/run/rendered_registry.json"),
                view_map=Path("/run/reference_view_map.json"),
                palette_fusion=Path("/run/palette_fusion.json"),
                group_id="G06",
                target_part_ids=["P0002", "P0001"],
                reference_view_ids=["front", "iso", "side", "top"],
                output=Path("/run/g06_quality.json"),
            )

    def test_exact_mdl_tournament_inherits_confident_auto_completed_views(
        self,
    ) -> None:
        mapping = {"front": "right", "side": "front", "top": "top"}
        quality = {
            "schema_version": "qwen-reference-render-comparison/v1",
            "inputs": {"selected_view_mapping": mapping},
            "thresholds": {
                "minimum_auto_alignment_score": 0.48,
                "minimum_auto_match_margin": 0.055,
            },
            "aggregate": {
                "reference_view_coverage_status": "PASS",
                "comparable_view_count": 3,
                "unscorable_view_count": 0,
                "failed_view_count": 0,
            },
            "views": [
                {
                    "reference_view_id": reference_id,
                    "render_view_id": render_id,
                    "status": "REVIEW",
                    "alignment": {},
                    "material_color": {},
                    "mapping": (
                        {
                            "mode": "auto_completion",
                            "selected_render_view_id": render_id,
                            "reasons": [],
                            "best_score": 0.79,
                            "global_assignment_margin": 0.30,
                            "global_one_to_one_assignment": True,
                        }
                        if reference_id == "top"
                        else {
                            "mode": "explicit_locked",
                            "selected_render_view_id": render_id,
                            "reasons": [],
                            "locked": True,
                        }
                    ),
                }
                for reference_id, render_id in mapping.items()
            ],
        }
        reference_manifest = {
            "source_views": [{"id": reference_id} for reference_id in mapping]
        }
        rendered_registry = {
            "render_set": {
                "views": [{"view_id": render_id} for render_id in mapping.values()]
            }
        }

        self.assertEqual(
            _validated_exact_mdl_tournament_mapping(
                quality_report=quality,
                reference_manifest=reference_manifest,
                trusted_mapping={"front": "right", "side": "front"},
                rendered_registry=rendered_registry,
            ),
            mapping,
        )
        quality["views"][2]["mapping"]["global_assignment_margin"] = 0.01
        with self.assertRaisesRegex(RuntimeError, "independently confident"):
            _validated_exact_mdl_tournament_mapping(
                quality_report=quality,
                reference_manifest=reference_manifest,
                trusted_mapping={"front": "right", "side": "front"},
                rendered_registry=rendered_registry,
            )

    def test_exact_mdl_tournament_accepts_appearance_fail_with_safe_mapping(
        self,
    ) -> None:
        mapping = {
            "front": "right",
            "side": "front",
            "top": "top",
            "iso": "pose_a135_e015",
        }
        trusted_mapping = {
            "front": "right",
            "side": "front",
            "iso": "pose_a135_e015",
        }
        quality = {
            "schema_version": "qwen-reference-render-comparison/v1",
            "inputs": {"selected_view_mapping": mapping},
            "thresholds": {
                "minimum_auto_alignment_score": 0.48,
                "minimum_auto_match_margin": 0.055,
            },
            "aggregate": {
                "reference_view_coverage_status": "PASS",
                "comparable_view_count": 4,
                "unscorable_view_count": 0,
                "failed_view_count": 4,
            },
            "views": [
                {
                    "reference_view_id": reference_id,
                    "render_view_id": render_id,
                    "status": "FAIL",
                    "reasons": ["trusted_palette_group_missing_from_render"],
                    "alignment": {"score": 0.8},
                    "material_color": {"score": 0.7},
                    "mapping": (
                        {
                            "mode": "auto_completion",
                            "selected_render_view_id": render_id,
                            "reasons": [],
                            "best_score": 0.79,
                            "global_assignment_margin": 0.30,
                            "global_one_to_one_assignment": True,
                        }
                        if reference_id == "top"
                        else {
                            "mode": "explicit_locked",
                            "selected_render_view_id": render_id,
                            "reasons": [],
                            "locked": True,
                        }
                    ),
                }
                for reference_id, render_id in mapping.items()
            ],
        }
        reference_manifest = {
            "source_views": [{"id": reference_id} for reference_id in mapping]
        }
        rendered_registry = {
            "render_set": {
                "views": [{"view_id": render_id} for render_id in mapping.values()]
            }
        }

        self.assertEqual(
            _validated_exact_mdl_tournament_mapping(
                quality_report=quality,
                reference_manifest=reference_manifest,
                trusted_mapping=trusted_mapping,
                rendered_registry=rendered_registry,
            ),
            mapping,
        )
        quality["views"][0]["status"] = "UNSCORABLE"
        with self.assertRaisesRegex(RuntimeError, "not comparison-safe"):
            _validated_exact_mdl_tournament_mapping(
                quality_report=quality,
                reference_manifest=reference_manifest,
                trusted_mapping=trusted_mapping,
                rendered_registry=rendered_registry,
            )

    def test_exact_mdl_tournament_rejects_untrusted_explicit_mapping(
        self,
    ) -> None:
        mapping = {"front": "right"}
        quality = {
            "schema_version": "qwen-reference-render-comparison/v1",
            "inputs": {"selected_view_mapping": mapping},
            "thresholds": {
                "minimum_auto_alignment_score": 0.48,
                "minimum_auto_match_margin": 0.055,
            },
            "aggregate": {
                "reference_view_coverage_status": "PASS",
                "comparable_view_count": 1,
                "unscorable_view_count": 0,
                "failed_view_count": 0,
            },
            "views": [
                {
                    "reference_view_id": "front",
                    "render_view_id": "right",
                    "status": "PASS",
                    "alignment": {},
                    "material_color": {},
                    "mapping": {
                        "mode": "explicit_locked",
                        "selected_render_view_id": "right",
                        "reasons": [],
                        "locked": True,
                    },
                }
            ],
        }

        with self.assertRaisesRegex(RuntimeError, "unsupported provenance"):
            _validated_exact_mdl_tournament_mapping(
                quality_report=quality,
                reference_manifest={"source_views": [{"id": "front"}]},
                trusted_mapping={},
                rendered_registry={"render_set": {"views": [{"view_id": "right"}]}},
            )

    def test_partial_live_resume_requires_hash_bound_unfinished_checkpoints(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            destination = root / "visual"
            analysis = destination / "analysis"
            face_dir = analysis / "face_regions"
            ledger_dir = analysis / "mvinverse"
            face_dir.mkdir(parents=True)
            ledger_dir.mkdir(parents=True)
            reference = root / "front.png"
            reference.write_bytes(b"reference-image")
            references = (("front", reference.resolve()),)
            reference_manifest = analysis / "mvinverse_reference_manifest.json"
            reference_manifest.write_text(
                json.dumps(
                    {
                        "source_views": [
                            {"id": "front", "image": str(reference.resolve())}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            evidence = face_dir / "parts" / "P0001.json"
            evidence.parent.mkdir()
            evidence.write_text("{}", encoding="utf-8")
            (face_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "qwen-face-region-evidence/v1",
                        "part_count": 1,
                        "parts": [{"part_id": "P0001", "evidence": "parts/P0001.json"}],
                    }
                ),
                encoding="utf-8",
            )
            from asset_pipeline.visual_materials.references import sha256_file

            (ledger_dir / "mvinverse_inference_ledger.json").write_text(
                json.dumps(
                    {
                        "schema_version": "qwen-mvinverse-inference-ledger/v1",
                        "status": "SUCCESS",
                        "fail_closed": True,
                        "inputs": {
                            "reference_manifest": {
                                "sha256": sha256_file(reference_manifest)
                            },
                            "source_views": [
                                {
                                    "view_id": "front",
                                    "sha256": sha256_file(reference),
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(
                _verified_partial_live_resume_available(destination, references)
            )
            config_path, _isaac, _fixture_references = self._fixture(root)
            config = load_visual_material_config(config_path)
            qwen_ledger = {
                "schema_version": "qwen-local-inference-ledger/v2",
                "requested_model_family": config.qwen_model_family,
                "requested_model_revision": config.qwen_model_revision,
                "model_identity": {
                    "model_path": str(config.qwen_model_path.resolve())
                },
                "palette_generation_policy": {
                    "initial_max_new_tokens": config.qwen_max_new_tokens,
                    "max_new_tokens_ceiling": (
                        config.qwen_max_new_tokens_ceiling
                    ),
                    "truncation_growth_factor": 2,
                    "retry_condition": "token_limit_reached_without_eos",
                    "minimum_usable_views": (
                        config.qwen_minimum_usable_palette_views
                    ),
                    "minimum_usable_view_ratio": (
                        config.qwen_minimum_usable_palette_view_ratio
                    ),
                },
            }
            qwen_ledger["integrity"] = {
                "ledger_sha256": canonical_sha256(qwen_ledger)
            }
            qwen_ledger_path = analysis / "qwen_inference_ledger.json"
            qwen_ledger_path.write_text(
                json.dumps(qwen_ledger), encoding="utf-8"
            )
            self.assertTrue(
                _verified_partial_live_resume_available(
                    destination, references, config
                )
            )
            qwen_ledger["palette_generation_policy"][
                "max_new_tokens_ceiling"
            ] += 1
            qwen_ledger["integrity"] = {
                "ledger_sha256": canonical_sha256(
                    {
                        key: value
                        for key, value in qwen_ledger.items()
                        if key != "integrity"
                    }
                )
            }
            qwen_ledger_path.write_text(
                json.dumps(qwen_ledger), encoding="utf-8"
            )
            self.assertFalse(
                _verified_partial_live_resume_available(
                    destination, references, config
                )
            )
            qwen_ledger["palette_generation_policy"][
                "max_new_tokens_ceiling"
            ] -= 1
            qwen_ledger["integrity"] = {
                "ledger_sha256": canonical_sha256(
                    {
                        key: value
                        for key, value in qwen_ledger.items()
                        if key != "integrity"
                    }
                )
            }
            qwen_ledger_path.write_text(
                json.dumps(qwen_ledger), encoding="utf-8"
            )
            premature_apply = destination / "apply_visual_materials_report.json"
            premature_apply.write_text("{}", encoding="utf-8")
            self.assertFalse(
                _verified_partial_live_resume_available(destination, references)
            )
            premature_apply.unlink()
            (analysis / "unattended_result.json").write_text("{}", encoding="utf-8")
            self.assertFalse(
                _verified_partial_live_resume_available(destination, references)
            )
            (analysis / "unattended_result.json").write_text(
                json.dumps({"state": "READY_TO_APPLY"}), encoding="utf-8"
            )
            for name in (
                "qwen_mvinverse_recovery.json",
                "staged_result.json",
                "confidence_gate.json",
                "autonomous_material_plan.json",
                "group_materials.json",
                "mvinverse_pbr_evidence.json",
                "part_mapping_multiview_audit.json",
                "spatial_mapping_report.json",
                "spatial_mapping_audit.json",
            ):
                (analysis / name).write_text("{}", encoding="utf-8")
            self.assertTrue(
                _verified_partial_live_resume_available(destination, references)
            )
            provisional_files = (
                destination / "apply_visual_materials_report.json",
                destination / "asset_look.usda",
                destination / "delivery_validation.json",
                destination / "quality_repair_instance_plan.json",
                analysis / "visual_quality_resolution.json",
                analysis / "exact_mdl_tournament_planning.json",
            )
            for path in provisional_files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            provisional_quality_directory = destination / "visual_quality"
            provisional_quality_directory.mkdir()
            (provisional_quality_directory / "partial.json").write_text(
                "{}", encoding="utf-8"
            )
            provisional_final_acceptance_directory = (
                destination / "final_visual_acceptance"
            )
            provisional_final_acceptance_directory.mkdir()
            tournament_directory = destination / "visual_exact_mdl_tournament"
            tournament_directory.mkdir()
            tournament_root_file = tournament_directory / "reference_view_map.json"
            tournament_root_file.write_text("{}", encoding="utf-8")
            retained_candidate = tournament_directory / "g01_01_deadbeef00"
            retained_candidate.mkdir()
            (retained_candidate / "plan.json").write_text("{}", encoding="utf-8")
            provisional_part_id_directory = (
                analysis / "part_id_sam3_refinement"
            )
            provisional_part_id_directory.mkdir()
            (provisional_part_id_directory / "manifest.json").write_text(
                "{}", encoding="utf-8"
            )
            provisional_part_id_plan = analysis / "part_id_material_plan.json"
            provisional_part_id_plan.write_text("{}", encoding="utf-8")
            preserved_checkpoints = (
                analysis / "nvidia_mdl_catalog.json",
                analysis / "policy_exact_cover_plan.json",
                analysis / "policy_exact_cover_audit.json",
                analysis / "quality_repair_plan.json",
                analysis / "quality_repair_audit.json",
            )
            for path in preserved_checkpoints:
                path.write_text("{}", encoding="utf-8")

            self.assertTrue(
                _verified_partial_live_resume_available(destination, references)
            )
            archive = _archive_partial_live_resume_downstream_artifacts(destination)
            self.assertIsNotNone(archive)
            assert archive is not None
            archive_manifest = json.loads(
                (archive / "archive_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(archive_manifest["status"], "COMPLETED")
            for path in provisional_files:
                self.assertFalse(path.exists())
                self.assertTrue(
                    archive.joinpath(path.relative_to(destination)).exists()
                )
            self.assertFalse(provisional_quality_directory.exists())
            self.assertTrue(
                archive.joinpath(
                    provisional_quality_directory.relative_to(destination)
                ).is_dir()
            )
            self.assertFalse(provisional_final_acceptance_directory.exists())
            self.assertTrue(
                archive.joinpath(
                    provisional_final_acceptance_directory.relative_to(destination)
                ).is_dir()
            )
            self.assertTrue(tournament_directory.is_dir())
            self.assertFalse(tournament_root_file.exists())
            self.assertTrue(
                archive.joinpath(
                    tournament_root_file.relative_to(destination)
                ).is_file()
            )
            self.assertTrue(retained_candidate.is_dir())
            self.assertTrue((retained_candidate / "plan.json").is_file())
            self.assertFalse(provisional_part_id_directory.exists())
            self.assertTrue(
                archive.joinpath(
                    provisional_part_id_directory.relative_to(destination)
                ).is_dir()
            )
            self.assertFalse(provisional_part_id_plan.exists())
            self.assertTrue(
                archive.joinpath(
                    provisional_part_id_plan.relative_to(destination)
                ).is_file()
            )
            for path in preserved_checkpoints:
                self.assertTrue(path.is_file())

            (analysis / "material_selection_lock.json").write_text(
                "{}", encoding="utf-8"
            )
            self.assertFalse(
                _verified_partial_live_resume_available(destination, references)
            )
            with self.assertRaisesRegex(RuntimeError, "final locked artifacts"):
                _archive_partial_live_resume_downstream_artifacts(destination)

    def test_public_api_uses_split_owner_package(self) -> None:
        self.assertIs(
            asset_pipeline.run_assign_visual_materials_job,
            visual_materials.run_assign_visual_materials_job,
        )
        self.assertIs(
            asset_pipeline.load_visual_material_config,
            visual_materials.load_visual_material_config,
        )

    def test_compatibility_facade_forwards_policy_fallback_opt_in(self) -> None:
        with patch(
            "asset_pipeline.jobs.material._run_assign_visual_materials_job",
            return_value={"state": "test"},
        ) as owner:
            result = run_assign_visual_materials_job(
                source_usd="asset.usd",
                source_cad="asset.stp",
                references=("front.png", "side.png"),
                allow_policy_material_fallback=True,
            )

        self.assertEqual(result, {"state": "test"})
        self.assertTrue(owner.call_args.kwargs["allow_policy_material_fallback"])
        self.assertEqual(owner.call_args.kwargs["inference_mode"], "live")

    def _fixture(self, root: Path) -> tuple[Path, Path, list[str]]:
        qwen_python = root / "qwen-python"
        mvinverse_python = root / "mvinverse-python"
        sam3_python = root / "sam3-python"
        retrieval_python = root / "retrieval-python"
        isaac_python = root / "isaac-python"
        for executable in (
            qwen_python,
            mvinverse_python,
            sam3_python,
            retrieval_python,
            isaac_python,
        ):
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o755)

        qwen_model = root / "qwen-model"
        material_root = root / "materials"
        mvinverse_repo = root / "mvinverse-repo"
        checkpoint = root / "checkpoint"
        sam3_repo = root / "sam3-repo"
        siglip2_model = root / "siglip2-model"
        dinov2_model = root / "dinov2-model"
        retrieval_cache = root / "retrieval-cache"
        for directory in (
            qwen_model,
            material_root,
            mvinverse_repo,
            checkpoint,
            sam3_repo,
            siglip2_model,
            dinov2_model,
            retrieval_cache,
        ):
            directory.mkdir()
        sam3_checkpoint = root / "sam3.pt"
        sam3_checkpoint.write_bytes(b"sam3")
        catalog = root / "catalog.json"
        whitelist = root / "whitelist.json"
        catalog.write_text("{}", encoding="utf-8")
        whitelist.write_text("{}", encoding="utf-8")
        config = root / "visual.json"
        config.write_text(
            json.dumps(
                {
                    "schema_version": CONFIG_SCHEMA_VERSION,
                    "qwen": {
                        "python": str(qwen_python),
                        "model_path": str(qwen_model),
                        "model_family": "qwen3_5",
                        "model_revision": "revision-qwen",
                        "max_new_tokens": 4096,
                        "max_new_tokens_ceiling": 4096,
                        "minimum_usable_palette_views": 2,
                        "minimum_usable_palette_view_ratio": 0.5,
                        "mapping_verification_views": 2,
                    },
                    "materials": {
                        "catalog": str(catalog),
                        "whitelist": str(whitelist),
                        "material_root": str(material_root),
                    },
                    "render": {
                        "resolution": 256,
                        "views": "front,side,iso",
                        "rt_subframes": 2,
                        "analysis_up_axis": "z",
                        "analysis_front_axis": "-y",
                    },
                    "mvinverse": {
                        "mode": "run",
                        "python": str(mvinverse_python),
                        "repository": str(mvinverse_repo),
                        "checkpoint": str(checkpoint),
                        "model_revision": "revision-1",
                        "device": "cuda",
                        "max_side": 448,
                        "oom_retry_max_sides": [392],
                        "timeout_seconds": 120,
                    },
                    "sam3": {
                        "python": str(sam3_python),
                        "repository": str(sam3_repo),
                        "checkpoint": str(sam3_checkpoint),
                        "device": "cuda",
                        "minimum_model_score": 0.45,
                        "minimum_prompt_overlap": 0.25,
                        "maximum_image_fraction": 0.8,
                        "minimum_mask_pixels": 32,
                    },
                    "retrieval": {
                        "python": str(retrieval_python),
                        "siglip2_model": str(siglip2_model),
                        "dinov2_model": str(dinov2_model),
                        "cache_dir": str(retrieval_cache),
                        "device": "cuda",
                        "siglip_top_k": 64,
                        "final_top_k": 32,
                        "batch_size": 24,
                    },
                }
            ),
            encoding="utf-8",
        )
        references = []
        for index, payload in enumerate((b"front", b"side"), start=1):
            image = root / f"ref-{index}.png"
            image.write_bytes(payload)
            references.append(f"view_{index}={image}")
        return config, isaac_python, references

    def test_config_paths_are_resolved_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, _isaac, _references = self._fixture(Path(temp_dir))
            config = load_visual_material_config(config_path)
        self.assertEqual(config.render_resolution, 256)
        self.assertEqual(config.mvinverse_oom_retry_max_sides, (392,))
        self.assertEqual(config.qwen_model_family, "qwen3_5")
        self.assertEqual(config.qwen_model_revision, "revision-qwen")
        self.assertEqual(config.qwen_max_new_tokens, 4096)
        self.assertEqual(config.qwen_max_new_tokens_ceiling, 4096)
        self.assertEqual(config.qwen_minimum_usable_palette_views, 2)
        self.assertEqual(config.qwen_minimum_usable_palette_view_ratio, 0.5)
        self.assertEqual(config.qwen_mapping_verification_views, 2)
        self.assertEqual(config.qwen_parallel_requests, 1)
        self.assertEqual(config.sam3_device, "cuda")
        self.assertEqual(config.sam3_minimum_model_score, 0.45)
        self.assertEqual(config.sam3_minimum_prompt_overlap, 0.25)
        self.assertEqual(config.sam3_maximum_image_fraction, 0.8)
        self.assertEqual(config.sam3_minimum_mask_pixels, 32)
        self.assertEqual(config.retrieval_device, "cuda")
        self.assertEqual(config.siglip_top_k, 64)
        self.assertEqual(config.retrieval_final_top_k, 32)
        self.assertEqual(config.retrieval_batch_size, 24)
        self.assertEqual(config.material_selection_pipeline_mode, "current")
        self.assertEqual(config.material_assignment_unit, "palette_group")
        self.assertEqual(config.quality_lighting_profile, "material-neutral")
        self.assertFalse(config.immutable_mdl_after_selection)
        self.assertEqual(
            config.material_selection_objective,
            "semantic_compatible_visual",
        )
        self.assertTrue(config.exact_mdl_tournament_all_groups)
        self.assertEqual(
            config.exact_mdl_tournament_minimum_score_improvement,
            0.015,
        )
        self.assertEqual(
            config.exact_mdl_tournament_minimum_winner_margin,
            0.005,
        )
        self.assertEqual(
            config.final_visual_gate_maximum_score_regression,
            0.01,
        )
        self.assertEqual(
            config.final_visual_gate_minimum_final_appearance_score,
            0.62,
        )
        self.assertEqual(
            config.final_visual_gate_minimum_significant_evidence_pixels,
            128,
        )
        self.assertEqual(
            config.final_visual_gate_maximum_policy_fallback_fraction,
            0.90,
        )
        self.assertEqual(
            config.final_visual_gate_maximum_neutral_fallback_fraction,
            0.75,
        )
        self.assertEqual(
            config.final_visual_gate_maximum_unresolved_entity_fraction,
            0.90,
        )
        self.assertEqual(
            config.final_visual_gate_maximum_unresolved_face_subset_fraction,
            0.50,
        )
        self.assertEqual(
            config.final_visual_gate_minimum_owner_local_resolved_fraction,
            0.50,
        )

    def test_config_accepts_remote_gpt_without_local_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, _isaac, _references = self._fixture(Path(temp_dir))
            document = json.loads(config_path.read_text(encoding="utf-8"))
            document["qwen"] = {
                "python": document["qwen"]["python"],
                "model_family": "openai_compatible",
                "base_url": "https://gateway.example/v1",
                "model": "gpt-5.6",
                "api_key_env": "REMOTE_GPT_KEY",
                "reasoning_effort": "medium",
                "timeout_seconds": 180,
                "max_new_tokens": 1024,
                "max_new_tokens_ceiling": 4096,
                "minimum_usable_palette_views": 2,
                "minimum_usable_palette_view_ratio": 0.5,
                "parallel_requests": 3,
            }
            config_path.write_text(json.dumps(document), encoding="utf-8")

            config = load_visual_material_config(config_path)

        self.assertEqual(config.qwen_model_family, "openai_compatible")
        self.assertIsNone(config.qwen_model_path)
        self.assertIsNone(config.qwen_model_revision)
        self.assertEqual(config.openai_base_url, "https://gateway.example/v1")
        self.assertEqual(config.openai_model, "gpt-5.6")
        self.assertEqual(config.openai_api_key_env, "REMOTE_GPT_KEY")
        self.assertEqual(config.openai_reasoning_effort, "medium")
        self.assertEqual(config.openai_timeout_seconds, 180)
        self.assertEqual(config.qwen_parallel_requests, 3)

    def test_production_config_requires_nvidia_base_scope(self) -> None:
        production_config = (
            Path(__file__).resolve().parents[1]
            / "tools"
            / "qwen_material_pipeline"
            / "configs"
            / "pipeline"
            / "manual_part_id_materials.json"
        )
        document = json.loads(production_config.read_text(encoding="utf-8"))

        self.assertEqual(
            document["materials"]["root_scope"],
            "nvidia_base",
        )
        self.assertEqual(
            document["qwen"]["model_family"],
            "qwen3_5",
        )
        self.assertEqual(
            document["qwen"]["model_path"],
            "${QWEN35_MODEL_PATH}",
        )
        self.assertEqual(document["qwen"]["python"], "${QWEN35_PYTHON}")
        self.assertNotIn("base_url", document["qwen"])
        self.assertNotIn("api_key_env", document["qwen"])
        self.assertEqual(document["qwen"]["max_new_tokens"], 1024)
        self.assertEqual(document["qwen"]["max_new_tokens_ceiling"], 4096)
        self.assertEqual(
            document["qwen"]["minimum_usable_palette_views"],
            2,
        )
        self.assertEqual(
            document["qwen"]["minimum_usable_palette_view_ratio"],
            0.5,
        )
        self.assertEqual(document["qwen"]["mapping_verification_views"], 2)
        self.assertEqual(document["qwen"]["parallel_requests"], 1)

    def test_config_rejects_palette_token_ceiling_below_initial_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, _isaac, _references = self._fixture(Path(temp_dir))
            document = json.loads(config_path.read_text(encoding="utf-8"))
            document["qwen"]["max_new_tokens_ceiling"] = 2048
            config_path.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "max_new_tokens_ceiling cannot be smaller",
            ):
                load_visual_material_config(config_path)

    def test_legacy_v2_config_keeps_single_budget_and_one_view_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, _isaac, _references = self._fixture(Path(temp_dir))
            document = json.loads(config_path.read_text(encoding="utf-8"))
            for name in (
                "max_new_tokens_ceiling",
                "minimum_usable_palette_views",
                "minimum_usable_palette_view_ratio",
                "mapping_verification_views",
            ):
                document["qwen"].pop(name)
            config_path.write_text(json.dumps(document), encoding="utf-8")

            config = load_visual_material_config(config_path)

        self.assertEqual(
            config.qwen_max_new_tokens_ceiling,
            config.qwen_max_new_tokens,
        )
        self.assertEqual(config.qwen_minimum_usable_palette_views, 1)
        self.assertEqual(config.qwen_minimum_usable_palette_view_ratio, 0.0)
        self.assertEqual(config.qwen_mapping_verification_views, 0)
        self.assertEqual(config.qwen_parallel_requests, 1)

    def test_config_rejects_unbounded_remote_parallel_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, _isaac, _references = self._fixture(Path(temp_dir))
            document = json.loads(config_path.read_text(encoding="utf-8"))
            document["qwen"]["parallel_requests"] = 9
            config_path.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "parallel_requests cannot exceed 8",
            ):
                load_visual_material_config(config_path)

    def test_config_rejects_invalid_usable_palette_view_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, _isaac, _references = self._fixture(Path(temp_dir))
            document = json.loads(config_path.read_text(encoding="utf-8"))
            document["qwen"]["minimum_usable_palette_view_ratio"] = 1.1
            config_path.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "minimum_usable_palette_view_ratio",
            ):
                load_visual_material_config(config_path)

    def test_nvidia_materials_scope_normalizes_collection_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, _isaac, _references = self._fixture(root)
            materials_root = root / "NVIDIA" / "Materials"
            (materials_root / "Base").mkdir(parents=True)
            (materials_root / "vMaterials_2").mkdir()

            for configured_root in (
                materials_root,
                materials_root / "Base",
                materials_root / "vMaterials_2",
            ):
                document = json.loads(config_path.read_text(encoding="utf-8"))
                document["materials"].update(
                    {
                        "material_root": str(configured_root),
                        "root_scope": "nvidia_materials",
                    }
                )
                config_path.write_text(json.dumps(document), encoding="utf-8")

                config = load_visual_material_config(config_path)

                self.assertEqual(config.material_root, materials_root)

    def test_nvidia_materials_scope_keeps_valid_mounted_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, _isaac, _references = self._fixture(root)
            mounted_root = root / "mounted-material-library"
            (mounted_root / "Base").mkdir(parents=True)
            (mounted_root / "vMaterials_2").mkdir()
            document = json.loads(config_path.read_text(encoding="utf-8"))
            document["materials"].update(
                {
                    "material_root": str(mounted_root),
                    "root_scope": "nvidia_materials",
                }
            )
            config_path.write_text(json.dumps(document), encoding="utf-8")

            config = load_visual_material_config(config_path)

            self.assertEqual(config.material_root, mounted_root)

    def test_nvidia_materials_scope_fails_closed_on_partial_library(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, _isaac, _references = self._fixture(root)
            partial_root = root / "partial-materials"
            (partial_root / "Base").mkdir(parents=True)
            document = json.loads(config_path.read_text(encoding="utf-8"))
            document["materials"].update(
                {
                    "material_root": str(partial_root),
                    "root_scope": "nvidia_materials",
                }
            )
            config_path.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(
                FileNotFoundError,
                "full NVIDIA Materials root.*vMaterials_2",
            ):
                load_visual_material_config(config_path)

    def test_nvidia_base_scope_normalizes_parent_and_base_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, _isaac, _references = self._fixture(root)
            materials_root = root / "NVIDIA" / "Materials"
            base_root = materials_root / "Base"
            base_root.mkdir(parents=True)
            (materials_root / "vMaterials_2").mkdir()

            for configured_root in (materials_root, base_root):
                document = json.loads(config_path.read_text(encoding="utf-8"))
                document["materials"].update(
                    {
                        "material_root": str(configured_root),
                        "root_scope": "nvidia_base",
                    }
                )
                config_path.write_text(json.dumps(document), encoding="utf-8")

                config = load_visual_material_config(config_path)

                self.assertEqual(config.material_root, base_root)

    def test_nvidia_base_scope_rejects_vmaterials_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, _isaac, _references = self._fixture(root)
            materials_root = root / "NVIDIA" / "Materials"
            (materials_root / "Base").mkdir(parents=True)
            vmaterials_root = materials_root / "vMaterials_2"
            vmaterials_root.mkdir()
            document = json.loads(config_path.read_text(encoding="utf-8"))
            document["materials"].update(
                {
                    "material_root": str(vmaterials_root),
                    "root_scope": "nvidia_base",
                }
            )
            config_path.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "rejects.*vMaterials_2"):
                load_visual_material_config(config_path)

    def test_custom_config_without_root_scope_keeps_given_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, _isaac, _references = self._fixture(root)
            configured_root = root / "materials"

            config = load_visual_material_config(config_path)

            self.assertEqual(config.material_root, configured_root)

    def test_config_rejects_unknown_material_root_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, _isaac, _references = self._fixture(Path(temp_dir))
            document = json.loads(config_path.read_text(encoding="utf-8"))
            document["materials"]["root_scope"] = "single_collection"
            config_path.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "root_scope"):
                load_visual_material_config(config_path)

    def test_config_enables_immutable_selected_mdl_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, _isaac, _references = self._fixture(Path(temp_dir))
            document = json.loads(config_path.read_text(encoding="utf-8"))
            document["materials"]["immutable_after_selection"] = True
            config_path.write_text(json.dumps(document), encoding="utf-8")
            config = load_visual_material_config(config_path)

        self.assertTrue(config.immutable_mdl_after_selection)

    def test_config_enables_visual_similarity_selection_objective(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, _isaac, _references = self._fixture(Path(temp_dir))
            document = json.loads(config_path.read_text(encoding="utf-8"))
            document["materials"]["selection_objective"] = "visual_similarity"
            config_path.write_text(json.dumps(document), encoding="utf-8")
            config = load_visual_material_config(config_path)

        self.assertEqual(
            config.material_selection_objective,
            "visual_similarity",
        )

    def test_config_enables_independent_part_id_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, _isaac, _references = self._fixture(Path(temp_dir))
            document = json.loads(config_path.read_text(encoding="utf-8"))
            document["materials"]["assignment_unit"] = "part_id"
            config_path.write_text(json.dumps(document), encoding="utf-8")
            config = load_visual_material_config(config_path)

        self.assertEqual(config.material_assignment_unit, "part_id")

    def test_config_enables_bounded_semantic_hybrid_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, _isaac, _references = self._fixture(Path(temp_dir))
            document = json.loads(config_path.read_text(encoding="utf-8"))
            document["materials"].update(
                {
                    "selection_pipeline_mode": "semantic_hybrid",
                    "assignment_unit": "part_id",
                    "immutable_after_selection": False,
                    "parameter_candidate_mode": "evidence_gated_h0_h1",
                    "selection_objective": "semantic_compatible_visual",
                    "exact_mdl_tournament_max_candidates": 3,
                }
            )
            config_path.write_text(json.dumps(document), encoding="utf-8")
            config = load_visual_material_config(config_path)

        self.assertEqual(config.material_selection_pipeline_mode, "semantic_hybrid")
        self.assertEqual(config.material_assignment_unit, "part_id")
        self.assertFalse(config.immutable_mdl_after_selection)
        self.assertEqual(
            config.material_parameter_candidate_mode,
            "evidence_gated_h0_h1",
        )
        self.assertEqual(
            config.material_selection_objective,
            "semantic_compatible_visual",
        )
        self.assertEqual(config.exact_mdl_tournament_max_candidates, 3)

    def test_config_rejects_semantic_hybrid_contract_drift(self) -> None:
        invalid_values = {
            "assignment_unit": "palette_group",
            "immutable_after_selection": True,
            "parameter_candidate_mode": "disabled",
            "selection_objective": "visual_similarity",
            "exact_mdl_tournament_max_candidates": 4,
        }
        for field, invalid_value in invalid_values.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp_dir:
                config_path, _isaac, _references = self._fixture(Path(temp_dir))
                document = json.loads(config_path.read_text(encoding="utf-8"))
                document["materials"].update(
                    {
                        "selection_pipeline_mode": "semantic_hybrid",
                        "assignment_unit": "part_id",
                        "immutable_after_selection": False,
                        "parameter_candidate_mode": "evidence_gated_h0_h1",
                        "selection_objective": "semantic_compatible_visual",
                        "exact_mdl_tournament_max_candidates": 3,
                        field: invalid_value,
                    }
                )
                config_path.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError,
                    "semantic_hybrid.*requires.*" + field,
                ):
                    load_visual_material_config(config_path)

    def test_semantic_hybrid_profile_uses_current_v2_catalog(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        profile = (
            repository
            / "tools"
            / "qwen_material_pipeline"
            / "configs"
            / "pipeline"
            / "manual_part_id_materials_semantic_hybrid.json"
        )
        document = json.loads(profile.read_text(encoding="utf-8"))
        materials = document["materials"]
        catalog_path = (profile.parent / materials["catalog"]).resolve(strict=True)
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

        self.assertEqual(materials["selection_pipeline_mode"], "semantic_hybrid")
        self.assertEqual(materials["assignment_unit"], "part_id")
        self.assertFalse(materials["immutable_after_selection"])
        self.assertEqual(
            materials["parameter_candidate_mode"],
            "evidence_gated_h0_h1",
        )
        self.assertEqual(
            materials["selection_objective"],
            "semantic_compatible_visual",
        )
        self.assertEqual(materials["exact_mdl_tournament_max_candidates"], 3)
        self.assertEqual(catalog["schema_version"], 2)

    def test_config_overrides_multigroup_tournament_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, _isaac, _references = self._fixture(Path(temp_dir))
            document = json.loads(config_path.read_text(encoding="utf-8"))
            document["materials"].update(
                {
                    "exact_mdl_tournament_all_groups": False,
                    "exact_mdl_tournament_minimum_score_improvement": 0.03,
                    "exact_mdl_tournament_minimum_winner_margin": 0.01,
                }
            )
            config_path.write_text(json.dumps(document), encoding="utf-8")
            config = load_visual_material_config(config_path)

        self.assertFalse(config.exact_mdl_tournament_all_groups)
        self.assertEqual(
            config.exact_mdl_tournament_minimum_score_improvement,
            0.03,
        )
        self.assertEqual(
            config.exact_mdl_tournament_minimum_winner_margin,
            0.01,
        )

    def test_config_rejects_unknown_material_selection_objective(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, _isaac, _references = self._fixture(Path(temp_dir))
            document = json.loads(config_path.read_text(encoding="utf-8"))
            document["materials"]["selection_objective"] = "physical_guess"
            config_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selection_objective"):
                load_visual_material_config(config_path)

    def test_config_rejects_unknown_quality_lighting_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, _isaac, _references = self._fixture(Path(temp_dir))
            document = json.loads(config_path.read_text(encoding="utf-8"))
            document["render"]["quality_lighting_profile"] = "showcase"
            config_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "quality_lighting_profile"):
                load_visual_material_config(config_path)

    def test_config_validates_final_visual_gate_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, _isaac, _references = self._fixture(Path(temp_dir))
            document = json.loads(config_path.read_text(encoding="utf-8"))
            document["render"]["final_visual_gate"] = {
                "maximum_score_regression": 0.025,
                "minimum_significant_evidence_pixels": 256,
                "maximum_policy_fallback_fraction": 0.75,
                "maximum_neutral_fallback_fraction": 0.65,
                "maximum_unresolved_face_subset_fraction": 0.25,
            }
            config_path.write_text(json.dumps(document), encoding="utf-8")
            config = load_visual_material_config(config_path)
            self.assertEqual(
                config.final_visual_gate_maximum_score_regression,
                0.025,
            )
            self.assertEqual(
                config.final_visual_gate_minimum_significant_evidence_pixels,
                256,
            )
            self.assertEqual(
                config.final_visual_gate_maximum_policy_fallback_fraction,
                0.75,
            )
            self.assertEqual(
                config.final_visual_gate_maximum_neutral_fallback_fraction,
                0.65,
            )
            self.assertEqual(
                config.final_visual_gate_maximum_unresolved_face_subset_fraction,
                0.25,
            )

            document["render"]["final_visual_gate"]["maximum_score_regression"] = 1.1
            config_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "maximum_score_regression"):
                load_visual_material_config(config_path)

    def test_reference_images_need_unique_ids_and_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.png"
            second = root / "second.png"
            first.write_bytes(b"same")
            second.write_bytes(b"same")
            with self.assertRaisesRegex(ValueError, "duplicate content"):
                parse_visual_references([str(first), str(second)])

    def test_reference_parser_assigns_stable_default_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.png"
            second = root / "second.jpg"
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            parsed = parse_visual_references([str(first), str(second)])
        self.assertEqual([item[0] for item in parsed], ["ref_01", "ref_02"])

    def test_live_catalog_scans_base_and_vmaterials_together(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            material_root = root / "NVIDIA" / "Materials"
            base = material_root / "Base" / "Metals"
            vmaterials = material_root / "vMaterials_2" / "Metal"
            base.mkdir(parents=True)
            vmaterials.mkdir(parents=True)
            (base / "Steel.mdl").write_text(
                "export material Steel(*) = material();",
                encoding="utf-8",
            )
            (vmaterials / "Steel_Painted.mdl").write_text(
                "export material Steel_Painted(*) = material();",
                encoding="utf-8",
            )
            configured_catalog = root / "configured-catalog.json"
            configured_whitelist = root / "configured-allowlist.json"
            configured_catalog.write_text("{}", encoding="utf-8")
            configured_whitelist.write_text("{}", encoding="utf-8")

            catalog_path, allowlist_path, count = _prepare_live_material_catalog(
                material_root=material_root,
                configured_catalog=configured_catalog,
                configured_whitelist=configured_whitelist,
                analysis_dir=root / "analysis",
                log_cb=None,
            )

            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            allowlist = json.loads(allowlist_path.read_text(encoding="utf-8"))
            self.assertEqual(count, 2)
            self.assertEqual(catalog["material_count"], 2)
            self.assertEqual(allowlist["material_count"], 2)
            self.assertEqual(
                set(allowlist["material_ids"]),
                {
                    "mdl:Base/Metals/Steel.mdl#Steel",
                    ("mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted"),
                },
            )

    def test_lighting_statistics_gate_requires_real_images_and_groups(self) -> None:
        aggregate_only = {"aggregate": {"status": "PASS"}}
        measurable = {
            "views": [
                {
                    "reference": {"image": "/tmp/reference.png"},
                    "render": {
                        "image": "/tmp/render.png",
                        "part_ids": "/tmp/part_ids.png",
                    },
                    "material_color": {},
                }
            ]
        }
        measured = json.loads(json.dumps(measurable))
        measured["views"][0]["material_color"]["lighting_normalized_groups"] = {
            "schema_version": ("qwen-lighting-normalized-group-statistics/v1"),
            "groups": [{"canonical_group_id": "G01"}],
        }

        self.assertFalse(_quality_can_measure_lighting_statistics(aggregate_only))
        self.assertTrue(_quality_can_measure_lighting_statistics(measurable))
        self.assertFalse(_quality_has_lighting_normalized_groups(measurable))
        self.assertTrue(_quality_has_lighting_normalized_groups(measured))
        self.assertEqual(
            _appearance_baseline_safety_reason(
                quality_gate_status=("MATERIAL_ACCEPTED_WITH_GEOMETRY_POSE_LIMITATION"),
                lighting_profile="material-neutral",
            ),
            "QUALITY_GATE_IS_NOT_PASS",
        )
        self.assertEqual(
            _appearance_baseline_safety_reason(
                quality_gate_status="PASS",
                lighting_profile="geometry",
            ),
            "BASELINE_LIGHTING_PROFILE_IS_NOT_MATERIAL_NEUTRAL",
        )
        self.assertIsNone(
            _appearance_baseline_safety_reason(
                quality_gate_status="PASS",
                lighting_profile="material-neutral",
            )
        )

    def test_appearance_candidate_is_adopted_only_after_validation_passes(
        self,
    ) -> None:
        cases = (
            ("PASS", "material-neutral"),
            ("FAIL_CLOSED", "material-neutral"),
            (None, "geometry"),
        )
        for validation_status, baseline_profile in cases:
            with self.subTest(
                validation_status=validation_status,
                baseline_profile=baseline_profile,
            ):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    config_path, isaac, references = self._fixture(root)
                    source = root / "asset.usd"
                    source.write_text("source", encoding="utf-8")
                    output = root / "visual-output"
                    commands: list[list[str]] = []

                    raw_quality = {
                        "aggregate": {
                            "status": "PASS",
                            "comparable_view_count": 2,
                            "failed_view_count": 0,
                            "unscorable_view_count": 0,
                        },
                        "views": [
                            {
                                "reference_view_id": "view_1",
                                "render_view_id": "front",
                                "reference": {"image": str(root / "ref-1.png")},
                                "render": {
                                    "image": str(root / "render.png"),
                                    "part_ids": str(root / "part-ids.png"),
                                },
                                "material_color": {},
                            }
                        ],
                    }
                    measured_quality = json.loads(json.dumps(raw_quality))
                    measured_quality["views"][0]["material_color"][
                        "lighting_normalized_groups"
                    ] = {
                        "schema_version": (
                            "qwen-lighting-normalized-group-statistics/v1"
                        ),
                        "groups": [{"canonical_group_id": "G01"}],
                    }

                    def fake_run(command, **_kwargs):
                        commands.append(list(command))
                        appearance_module = (
                            "qwen_material_pipeline.materials.appearance_optimization"
                        )
                        if appearance_module in command:
                            if "measure" in command:
                                quality_output = Path(
                                    command[
                                        command.index("--output-quality-report") + 1
                                    ]
                                )
                                report_output = Path(
                                    command[command.index("--output-report") + 1]
                                )
                                quality_output.parent.mkdir(parents=True, exist_ok=True)
                                quality_output.write_text(
                                    json.dumps(measured_quality),
                                    encoding="utf-8",
                                )
                                report_output.write_text(
                                    json.dumps({"summary": {"measured_view_count": 1}}),
                                    encoding="utf-8",
                                )
                            elif "build" in command:
                                contract = Path(command[command.index("--output") + 1])
                                contract.write_text(
                                    json.dumps({"summary": {"adjustment_count": 1}}),
                                    encoding="utf-8",
                                )
                            elif "apply" in command:
                                plan = Path(command[command.index("--output-plan") + 1])
                                report = Path(
                                    command[command.index("--output-report") + 1]
                                )
                                plan.write_text(
                                    json.dumps({"assignments": [{"part_id": "P0001"}]}),
                                    encoding="utf-8",
                                )
                                report.write_text(
                                    json.dumps({"changed_part_count": 1}),
                                    encoding="utf-8",
                                )
                            elif "validate" in command:
                                self.assertIsNotNone(validation_status)
                                report = Path(command[command.index("--output") + 1])
                                report.write_text(
                                    json.dumps({"status": validation_status}),
                                    encoding="utf-8",
                                )
                            return
                        if "registry" in command:
                            path = Path(command[command.index("--output") + 1])
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(
                                json.dumps({"instance_root_count": 0}),
                                encoding="utf-8",
                            )
                        elif "render" in command:
                            directory = Path(command[command.index("--output-dir") + 1])
                            directory.mkdir(parents=True, exist_ok=True)
                            (directory / "part_registry.rendered.json").write_text(
                                json.dumps(
                                    {
                                        "render_set": {
                                            "lighting_profile": baseline_profile
                                        }
                                    }
                                ),
                                encoding="utf-8",
                            )
                        elif "staged" in command:
                            directory = Path(command[command.index("--output-dir") + 1])
                            (directory / "mvinverse").mkdir(parents=True, exist_ok=True)
                            (
                                directory
                                / "mvinverse"
                                / "mvinverse_inference_ledger.json"
                            ).write_text("{}", encoding="utf-8")
                            (directory / "unattended_result.json").write_text(
                                json.dumps({"state": "READY_TO_APPLY"}),
                                encoding="utf-8",
                            )
                            (directory / "autonomous_material_plan.json").write_text(
                                json.dumps({"assignments": [{"part_id": "P0001"}]}),
                                encoding="utf-8",
                            )
                            for name in (
                                "mvinverse_pbr_evidence.json",
                                "palette_fusion.json",
                                "spatial_mapping_report.json",
                            ):
                                (directory / name).write_text("{}", encoding="utf-8")
                        elif "apply" in command:
                            look = Path(command[command.index("--output") + 1])
                            report = Path(command[command.index("--report") + 1])
                            look.write_text("look", encoding="utf-8")
                            report.write_text(
                                json.dumps({"applied_count": 1}),
                                encoding="utf-8",
                            )
                        elif "compare" in command:
                            report = Path(command[command.index("--output") + 1])
                            report.write_text(json.dumps(raw_quality), encoding="utf-8")

                    with (
                        patch(
                            "asset_pipeline.jobs.material.isaac_python",
                            return_value=isaac,
                        ),
                        patch(
                            "asset_pipeline.jobs.material.run_command",
                            side_effect=fake_run,
                        ),
                    ):
                        result = run_assign_visual_materials_job(
                            source_usd=str(source),
                            references=references,
                            output_dir=str(output),
                            config_path=str(config_path),
                            acknowledge_mvinverse_noncommercial=True,
                        )

                self.assertEqual(
                    result["appearance_optimization_status"],
                    (
                        "ACCEPTED"
                        if validation_status == "PASS"
                        else (
                            "REJECTED_FAIL_CLOSED"
                            if validation_status == "FAIL_CLOSED"
                            else "SKIPPED_UNSAFE_BASELINE"
                        )
                    ),
                )
                if validation_status is None:
                    self.assertEqual(
                        result["appearance_optimization_reason_codes"],
                        ["BASELINE_LIGHTING_PROFILE_IS_NOT_MATERIAL_NEUTRAL"],
                    )
                    self.assertEqual(
                        result["appearance_optimization_adjustment_count"],
                        0,
                    )
                    self.assertEqual(result["visual_quality_round_count"], 1)
                    self.assertTrue(result["effective_usd"].endswith("_look.usda"))
                    continue
                self.assertEqual(result["appearance_optimization_adjustment_count"], 1)
                self.assertEqual(
                    result["appearance_optimization_changed_part_count"],
                    1,
                )
                self.assertEqual(result["quality_repair_round_count"], 1)
                self.assertEqual(result["visual_quality_round_count"], 2)
                if validation_status == "PASS":
                    self.assertIn(
                        "_look_appearance_candidate.usda",
                        result["effective_usd"],
                    )
                    self.assertEqual(
                        result["material_plan"],
                        result["appearance_optimization_candidate_plan"],
                    )
                else:
                    self.assertTrue(result["effective_usd"].endswith("_look.usda"))
                    self.assertNotEqual(
                        result["material_plan"],
                        result["appearance_optimization_candidate_plan"],
                    )

    def test_inference_recovery_retries_fresh_before_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "analysis"
            output.mkdir()
            (output / "partial.json").write_text("{}", encoding="utf-8")
            attempts = 0

            def fake_run(_command, **_kwargs):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("preflight import failure")

            audit_path = _run_qwen_mvinverse_with_recovery(
                ["python", "--mvinverse-mode", "run"],
                output_dir=output,
                ledger=output / "mvinverse" / "ledger.json",
                face_region_manifest=output / "face_regions" / "manifest.json",
                log_cb=None,
                command_runner=fake_run,
            )
            audit = json.loads(audit_path.read_text(encoding="utf-8"))

            self.assertEqual(attempts, 2)
            self.assertEqual(audit["status"], "RECOVERED")
            self.assertEqual(audit["retry_mode"], "fresh_stage")
            self.assertTrue(
                (root / "analysis.failed_attempt_01" / "partial.json").is_file()
            )

    def test_inference_recovery_suppresses_deterministic_child_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "analysis"
            output.mkdir()
            attempts = 0

            def fake_run(_command, **_kwargs):
                nonlocal attempts
                attempts += 1
                (output / "inference_failure.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "qwen-material-inference-failure/v1",
                            "status": "FAILED",
                            "error_code": "insufficient_usable_palette_views",
                            "failed_stage": "palette",
                            "retryable": False,
                            "retry_scope": "none",
                            "detail": "only one of four views was usable",
                            "view_failures": [],
                        }
                    ),
                    encoding="utf-8",
                )
                raise RuntimeError("deterministic palette failure")

            with self.assertRaisesRegex(
                RuntimeError,
                "identical fresh-process retry was suppressed",
            ):
                _run_qwen_mvinverse_with_recovery(
                    ["python", "--mvinverse-mode", "run"],
                    output_dir=output,
                    ledger=output / "mvinverse" / "ledger.json",
                    face_region_manifest=output / "face_regions" / "manifest.json",
                    log_cb=None,
                    command_runner=fake_run,
                )

            audit = json.loads(
                (output / "qwen_mvinverse_recovery.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(attempts, 1)
        self.assertEqual(audit["status"], "FAILED_NON_RETRYABLE")
        self.assertEqual(audit["decision"], "SUPPRESS_IDENTICAL_RETRY")
        self.assertEqual(
            audit["failure_code"],
            "insufficient_usable_palette_views",
        )

    def test_inference_resume_reuses_two_existing_verified_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "analysis"
            ledger = output / "mvinverse" / "ledger.json"
            manifest = output / "face_regions" / "manifest.json"
            ledger.parent.mkdir(parents=True)
            manifest.parent.mkdir(parents=True)
            ledger.write_text('{"status":"SUCCESS"}', encoding="utf-8")
            manifest.write_text('{"status":"SUCCESS"}', encoding="utf-8")
            commands: list[list[str]] = []

            def fake_run(command, **_kwargs):
                commands.append(list(command))

            audit_path = _run_qwen_mvinverse_with_recovery(
                ["python", "--mvinverse-mode", "run"],
                output_dir=output,
                ledger=ledger,
                face_region_manifest=manifest,
                log_cb=None,
                command_runner=fake_run,
            )
            audit = json.loads(audit_path.read_text(encoding="utf-8"))

            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0][-1], "reuse")
            self.assertEqual(audit["status"], "RESUMED_FROM_VERIFIED_CHECKPOINTS")
            self.assertEqual(audit["attempt_count"], 1)

    def test_inference_resume_uses_complete_material_stage_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "analysis"
            ledger = output / "mvinverse" / "ledger.json"
            manifest = output / "face_regions" / "manifest.json"
            ledger.parent.mkdir(parents=True)
            manifest.parent.mkdir(parents=True)
            ledger.write_text('{"status":"SUCCESS"}', encoding="utf-8")
            manifest.write_text('{"status":"SUCCESS"}', encoding="utf-8")
            for name in (
                "palette.json",
                "mvinverse_pbr_evidence.json",
                "staged_result.json",
                "material_plan.json",
                "group_materials.json",
                "material_choice_audit.json",
                "view_evidence.json",
                "part_mapping_multiview_votes.json",
                "part_mapping_multiview_audit.json",
                "spatial_mapping_report.json",
                "spatial_mapping_audit.json",
            ):
                (output / name).write_text("{}", encoding="utf-8")
            (output / "material_stage_contract.json").write_text(
                json.dumps(material_stage_contract_document()),
                encoding="utf-8",
            )
            batches = output / "batches"
            batches.mkdir()
            (batches / "B01.json").write_text("{}", encoding="utf-8")
            commands: list[list[str]] = []

            audit_path = _run_qwen_mvinverse_with_recovery(
                ["python", "--mvinverse-mode", "run"],
                output_dir=output,
                ledger=ledger,
                face_region_manifest=manifest,
                log_cb=None,
                command_runner=lambda command, **_kwargs: commands.append(
                    list(command)
                ),
            )
            audit = json.loads(audit_path.read_text(encoding="utf-8"))

            self.assertEqual(len(commands), 1)
            self.assertIn("--resume-from-materials", commands[0])
            self.assertEqual(
                commands[0][commands[0].index("--mvinverse-mode") + 1],
                "reuse",
            )
            self.assertTrue(audit["material_stage_resume"])

    def test_job_recovers_qwen_after_verified_heavy_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, isaac, references = self._fixture(root)
            source = root / "asset.usd"
            source.write_text("source", encoding="utf-8")
            output = root / "visual-output"
            commands: list[list[str]] = []
            staged_attempts = 0

            def fake_run(command, **kwargs):
                nonlocal staged_attempts
                commands.append(list(command))
                self.assertEqual(tuple(kwargs["env_remove"]), ISOLATED_ENV_REMOVE)
                if "registry" in command:
                    path = Path(command[command.index("--output") + 1])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        json.dumps({"instance_root_count": 0}), encoding="utf-8"
                    )
                elif "render" in command:
                    directory = Path(command[command.index("--output-dir") + 1])
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / "part_registry.rendered.json").write_text(
                        "{}", encoding="utf-8"
                    )
                elif "staged" in command:
                    staged_attempts += 1
                    directory = Path(command[command.index("--output-dir") + 1])
                    (directory / "mvinverse").mkdir(parents=True, exist_ok=True)
                    (directory / "face_regions").mkdir(parents=True, exist_ok=True)
                    (
                        directory / "mvinverse" / "mvinverse_inference_ledger.json"
                    ).write_text("{}", encoding="utf-8")
                    (directory / "face_regions" / "manifest.json").write_text(
                        "{}", encoding="utf-8"
                    )
                    if staged_attempts == 1:
                        raise RuntimeError("transient local model import failure")
                    (directory / "unattended_result.json").write_text(
                        json.dumps({"state": "READY_TO_APPLY"}), encoding="utf-8"
                    )
                    (directory / "autonomous_material_plan.json").write_text(
                        json.dumps({"assignments": [{"part_id": "P0001"}]}),
                        encoding="utf-8",
                    )
                elif "apply" in command:
                    look = Path(command[command.index("--output") + 1])
                    report = Path(command[command.index("--report") + 1])
                    look.write_text("look", encoding="utf-8")
                    report.write_text(
                        json.dumps({"applied_count": 1}), encoding="utf-8"
                    )
                elif "compare" in command:
                    report = Path(command[command.index("--output") + 1])
                    report.write_text(
                        json.dumps(
                            {
                                "aggregate": {
                                    "status": "PASS",
                                    "comparable_view_count": 2,
                                }
                            }
                        ),
                        encoding="utf-8",
                    )

            with (
                patch("asset_pipeline.jobs.material.isaac_python", return_value=isaac),
                patch("asset_pipeline.jobs.material.run_command", side_effect=fake_run),
            ):
                result = run_assign_visual_materials_job(
                    source_usd=str(source),
                    references=references,
                    output_dir=str(output),
                    config_path=str(config_path),
                    acknowledge_mvinverse_noncommercial=True,
                )
                recovery = json.loads(
                    Path(result["qwen_mvinverse_recovery"]).read_text(encoding="utf-8")
                )

        self.assertEqual(result["state"], "APPLIED")
        self.assertFalse(result["instance_aware"])
        self.assertFalse(result["complete_coverage_required"])
        self.assertEqual(result["assignment_count"], 1)
        self.assertEqual(result["applied_count"], 1)
        self.assertEqual(result["visual_quality_decision"], "ACCEPTED")
        self.assertEqual(staged_attempts, 2)
        retry_command = [command for command in commands if "staged" in command][1]
        self.assertEqual(
            retry_command[retry_command.index("--mvinverse-mode") + 1], "reuse"
        )
        expected_multimodel_options = {
            "--qwen-model-family": "qwen3_5",
            "--qwen-model-revision": "revision-qwen",
            "--max-new-tokens": "4096",
            "--max-new-tokens-ceiling": "4096",
            "--minimum-usable-palette-views": "2",
            "--minimum-usable-palette-view-ratio": "0.5",
            "--mapping-verification-views": "2",
            "--remote-parallel-requests": "1",
            "--sam3-device": "cuda",
            "--sam3-minimum-model-score": "0.45",
            "--sam3-minimum-prompt-overlap": "0.25",
            "--sam3-maximum-image-fraction": "0.8",
            "--sam3-minimum-mask-pixels": "32",
            "--retrieval-device": "cuda",
            "--siglip-top-k": "64",
            "--retrieval-final-top-k": "32",
            "--retrieval-batch-size": "24",
        }
        for option, expected in expected_multimodel_options.items():
            self.assertEqual(
                retry_command[retry_command.index(option) + 1],
                expected,
            )
        for option in (
            "--sam3-python",
            "--sam3-repo",
            "--sam3-checkpoint",
            "--retrieval-python",
            "--siglip2-model",
            "--dinov2-model",
            "--retrieval-cache-dir",
        ):
            self.assertIn(option, retry_command)
        self.assertEqual(recovery["status"], "RECOVERED")
        render_commands = [command for command in commands if "render" in command]
        self.assertNotIn("--lighting-profile", render_commands[0])
        self.assertEqual(
            render_commands[1][render_commands[1].index("--lighting-profile") + 1],
            "material-neutral",
        )
        self.assertEqual(
            [
                next(
                    item
                    for item in (
                        "registry",
                        "render",
                        "staged",
                        "apply",
                        "compare",
                    )
                    if item in command
                )
                for command in commands
            ],
            [
                "registry",
                "render",
                "staged",
                "staged",
                "apply",
                "registry",
                "render",
                "compare",
            ],
        )

    def test_unattended_job_rejects_review_quality_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, isaac, references = self._fixture(root)
            source = root / "asset.usd"
            source.write_text("source", encoding="utf-8")
            output = root / "visual-output"
            commands: list[list[str]] = []

            def fake_run(command, **_kwargs):
                commands.append(list(command))
                if "registry" in command:
                    path = Path(command[command.index("--output") + 1])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        json.dumps({"instance_root_count": 0}), encoding="utf-8"
                    )
                elif "render" in command:
                    directory = Path(command[command.index("--output-dir") + 1])
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / "part_registry.rendered.json").write_text(
                        "{}", encoding="utf-8"
                    )
                elif "staged" in command:
                    directory = Path(command[command.index("--output-dir") + 1])
                    (directory / "mvinverse").mkdir(parents=True, exist_ok=True)
                    (directory / "unattended_result.json").write_text(
                        json.dumps({"state": "READY_TO_APPLY"}), encoding="utf-8"
                    )
                    (directory / "autonomous_material_plan.json").write_text(
                        json.dumps({"assignments": [{"part_id": "P0001"}]}),
                        encoding="utf-8",
                    )
                    (
                        directory / "mvinverse" / "mvinverse_inference_ledger.json"
                    ).write_text("{}", encoding="utf-8")
                elif "apply" in command:
                    look = Path(command[command.index("--output") + 1])
                    report = Path(command[command.index("--report") + 1])
                    look.write_text("look", encoding="utf-8")
                    report.write_text(
                        json.dumps({"applied_count": 1}), encoding="utf-8"
                    )
                elif "compare" in command:
                    report = Path(command[command.index("--output") + 1])
                    report.write_text(
                        json.dumps(
                            {
                                "aggregate": {
                                    "status": "REVIEW",
                                    "comparable_view_count": 2,
                                }
                            }
                        ),
                        encoding="utf-8",
                    )

            with (
                patch("asset_pipeline.jobs.material.isaac_python", return_value=isaac),
                patch("asset_pipeline.jobs.material.run_command", side_effect=fake_run),
                self.assertRaisesRegex(
                    RuntimeError,
                    r"inconclusive \(REVIEW\).*accepts only PASS",
                ),
            ):
                run_assign_visual_materials_job(
                    source_usd=str(source),
                    references=references,
                    output_dir=str(output),
                    config_path=str(config_path),
                    acknowledge_mvinverse_noncommercial=True,
                )

        self.assertIn("compare", commands[-1])

    def test_instanced_cad_is_expanded_and_uses_a_sealed_complete_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, isaac, references = self._fixture(root)
            source = root / "assembly.usd"
            source.write_text("source assembly", encoding="utf-8")
            output = root / "visual-output"
            commands: list[list[str]] = []

            def fake_run(command, **_kwargs):
                commands.append(list(command))
                if "registry" in command:
                    path = Path(command[command.index("--output") + 1])
                    usd = Path(command[command.index("--usd") + 1])
                    document = {
                        "instance_root_count": 1 if usd == source else 0,
                        "parts": (
                            []
                            if usd == source
                            else [
                                {"part_id": "P0001"},
                                {"part_id": "P0002"},
                            ]
                        ),
                    }
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(document), encoding="utf-8")
                elif "expand" in command:
                    editable = Path(command[command.index("--output-usd") + 1])
                    report = Path(command[command.index("--report") + 1])
                    editable.write_text("editable", encoding="utf-8")
                    report.write_text("{}", encoding="utf-8")
                elif "render" in command:
                    directory = Path(command[command.index("--output-dir") + 1])
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / "part_registry.rendered.json").write_text(
                        json.dumps(
                            {
                                "parts": [
                                    {"part_id": "P0001"},
                                    {"part_id": "P0002"},
                                ]
                            }
                        ),
                        encoding="utf-8",
                    )
                elif "staged" in command:
                    directory = Path(command[command.index("--output-dir") + 1])
                    (directory / "mvinverse").mkdir(parents=True, exist_ok=True)
                    (directory / "unattended_result.json").write_text(
                        json.dumps({"state": "READY_TO_APPLY"}), encoding="utf-8"
                    )
                    (directory / "autonomous_material_plan.json").write_text(
                        json.dumps(
                            {
                                "schema_version": "1.0",
                                "assignments": [
                                    {"part_id": "P0001", "status": "approved"},
                                    {"part_id": "P0002", "status": "approved"},
                                ],
                            }
                        ),
                        encoding="utf-8",
                    )
                    (
                        directory / "mvinverse" / "mvinverse_inference_ledger.json"
                    ).write_text("{}", encoding="utf-8")
                elif "apply-instances" in command:
                    look = Path(command[command.index("--output") + 1])
                    report = Path(command[command.index("--report") + 1])
                    plan = Path(command[command.index("--plan") + 1])
                    plan_document = json.loads(plan.read_text(encoding="utf-8"))
                    self.assertIn("provenance", plan_document)
                    look.write_text("look", encoding="utf-8")
                    report.write_text(
                        json.dumps({"applied_count": 2}), encoding="utf-8"
                    )
                elif "compare" in command:
                    report = Path(command[command.index("--output") + 1])
                    report.write_text(
                        json.dumps(
                            {
                                "aggregate": {
                                    "status": "PASS",
                                    "comparable_view_count": 2,
                                }
                            }
                        ),
                        encoding="utf-8",
                    )

            with (
                patch("asset_pipeline.jobs.material.isaac_python", return_value=isaac),
                patch("asset_pipeline.jobs.material.run_command", side_effect=fake_run),
            ):
                result = run_assign_visual_materials_job(
                    source_usd=str(source),
                    references=references,
                    output_dir=str(output),
                    config_path=str(config_path),
                    acknowledge_mvinverse_noncommercial=True,
                )

        self.assertTrue(result["instance_aware"])
        self.assertTrue(result["complete_coverage_required"])
        self.assertEqual(result["instance_root_count"], 1)
        self.assertEqual(result["assignment_count"], 2)
        self.assertEqual(
            result["schema_version"], "asset-pipeline-visual-material-result/v2"
        )
        self.assertEqual(result["inference_mode"], "qwen_mvinverse")
        self.assertIsNotNone(result["editable_usd"])
        self.assertIsNotNone(result["instance_material_plan"])
        self.assertEqual(
            [
                next(
                    item
                    for item in (
                        "registry",
                        "expand",
                        "render",
                        "staged",
                        "apply-instances",
                        "compare",
                    )
                    if item in command
                )
                for command in commands
            ],
            [
                "registry",
                "expand",
                "registry",
                "render",
                "staged",
                "apply-instances",
                "registry",
                "render",
                "compare",
            ],
        )

    def test_license_is_required_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "asset.usd"
            source.write_text("source", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "acknowledge"):
                run_assign_visual_materials_job(
                    source_usd=str(source),
                    references=[],
                    output_dir=str(root / "output"),
                )
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
