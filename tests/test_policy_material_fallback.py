from __future__ import annotations

import copy
import unittest

from asset_pipeline.visual_materials.config import canonical_sha256
from qwen_material_pipeline.materials.policy_exact_cover import (
    build_policy_exact_cover,
)
from asset_pipeline.visual_materials.orchestrator import (
    CORROBORATED_SOURCE_MDL_TIER,
    GENERIC_STEEL_PAINTED,
    POLICY_PLAN_MODE,
    POLICY_REPORT_SCHEMA_VERSION,
    POLICY_FALLBACK_CONFIDENCE_BASIS,
    QUALITY_REPAIR_PLAN_MODE,
    QUALITY_REPAIR_PROVENANCE_FIELD,
    QUALITY_REPAIR_REASON_CODES,
    QUALITY_REPAIR_REPORT_SCHEMA_VERSION,
    _validate_corroborated_source_visual_assignments,
    _validate_policy_exact_cover_bundle,
    _validate_quality_dominant_mass,
    _validate_quality_repair_bundle,
    _validate_quality_repair_outcome,
)


class PolicyMaterialFallbackTests(unittest.TestCase):
    def _bundle(self) -> dict[str, dict]:
        registry = {
            "schema_version": "qwen-material-parts/v1",
            "asset_sha256": "a" * 64,
            "part_count": 2,
            "parts": [
                {"part_id": "P0001", "prim_path": "/Asset/A/Mesh"},
                {"part_id": "P0002", "prim_path": "/Asset/B/Mesh"},
            ],
        }
        staged_result = {"schema_version": "qwen-staged-material-result/v1"}
        confidence_gate = {"schema_version": "qwen-material-confidence-gate/v1"}
        base_plan = {"schema_version": "1.0", "assignments": []}
        group_materials = {"schema_version": "qwen-palette-material/v1"}
        mvinverse = {"schema_version": "qwen-mvinverse-pbr-evidence/v1"}
        whitelist = {"schema_version": 1, "material_ids": ["mdl:test#material"]}
        policy = {"schema_version": "qwen-policy-exact-cover/v1"}
        assignments = [
            {
                "part_id": part_id,
                "material_id": "mdl:test#material",
                "status": "policy_fallback",
                "confidence": 0.0,
                "evidence_views": [],
            }
            for part_id in ("P0001", "P0002")
        ]
        provenance = {
            "mode": POLICY_PLAN_MODE,
            "registry_asset_sha256": registry["asset_sha256"],
            "registry_sha256": canonical_sha256(registry),
            "staged_result_sha256": canonical_sha256(staged_result),
            "confidence_gate_sha256": canonical_sha256(confidence_gate),
            "base_plan_sha256": canonical_sha256(base_plan),
            "group_materials_sha256": canonical_sha256(group_materials),
            "mvinverse_pbr_evidence_sha256": canonical_sha256(mvinverse),
            "whitelist_sha256": canonical_sha256(whitelist),
            "policy_sha256": canonical_sha256(policy),
        }
        plan = {
            "schema_version": "1.0",
            "assignments": assignments,
            "provenance": provenance,
        }
        audit = {
            "schema_version": POLICY_REPORT_SCHEMA_VERSION,
            "summary": {
                "registry_part_count": 2,
                "output_assignment_count": 2,
                "policy_fallback_count": 2,
                "exact_cover": True,
                "all_materials_in_industrial_whitelist": True,
            },
            "policy": policy,
            "input_hashes": copy.deepcopy(provenance),
            "output_plan_sha256": canonical_sha256(plan),
        }
        return {
            "plan": plan,
            "audit": audit,
            "registry": registry,
            "staged_result": staged_result,
            "confidence_gate": confidence_gate,
            "base_plan": base_plan,
            "group_materials": group_materials,
            "mvinverse_pbr_evidence": mvinverse,
            "whitelist": whitelist,
        }

    @staticmethod
    def _reseal_bundle(bundle: dict[str, dict]) -> None:
        provenance = bundle["plan"]["provenance"]
        provenance["registry_asset_sha256"] = bundle["registry"]["asset_sha256"]
        provenance["registry_sha256"] = canonical_sha256(bundle["registry"])
        bundle["audit"]["input_hashes"] = copy.deepcopy(provenance)
        bundle["audit"]["output_plan_sha256"] = canonical_sha256(bundle["plan"])

    @staticmethod
    def _part_id_evidence(bundle: dict[str, dict]) -> dict:
        unsigned = {
            "schema_version": "qwen-part-id-reference-evidence/v1",
            "assignment_unit": "part_id",
            "inputs": [
                {
                    "label": "rendered_registry",
                    "path": None,
                    "document_sha256": canonical_sha256(bundle["registry"]),
                }
            ],
            "parts": [
                {
                    "part_id": "P0001",
                    "status": "observed",
                    "observations": [{"view_id": "front"}],
                },
                {
                    "part_id": "P0002",
                    "status": "unobserved",
                    "observations": [],
                },
            ],
            "summary": {
                "registry_part_count": 2,
                "observed_part_count": 1,
                "unobserved_part_count": 1,
            },
        }
        return {
            **unsigned,
            "integrity": {"document_sha256": canonical_sha256(unsigned)},
        }

    def _evidence_replay_bundle(self) -> tuple[dict[str, dict], dict, dict]:
        bundle = self._bundle()
        for index, part in enumerate(bundle["registry"]["parts"]):
            part["parent_path"] = f"/Asset/Assembly_{index}"
        bundle["staged_result"] = {
            "schema_version": "qwen-staged-material-result/v1",
            "material_plan": {"schema_version": "1.0", "assignments": []},
        }
        bundle["confidence_gate"] = {
            "schema_version": "qwen-material-confidence-gate/v1",
            "decisions": [
                {"part_id": part_id, "decision": "preserve"}
                for part_id in ("P0001", "P0002")
            ],
        }
        bundle["group_materials"] = {
            "schema_version": "qwen-palette-material/v1",
            "selections": [],
        }
        policy = {
            "schema_version": "qwen-policy-exact-cover/v1",
            "source_visual_strategy": "neutralize_unverified",
            "default_material_id": "mdl:test#material",
            "semantic_rules": [],
            "ground_contact": {
                "enabled": False,
                "material_id": "mdl:test#material",
                "elevation_tolerance_ratio": 0.02,
                "minimum_lateral_span_ratio": 0.12,
                "maximum_up_span_ratio": 0.18,
            },
        }
        evidence = self._part_id_evidence(bundle)
        bundle["plan"], bundle["audit"] = build_policy_exact_cover(
            registry=bundle["registry"],
            staged_result=bundle["staged_result"],
            confidence_gate=bundle["confidence_gate"],
            whitelist=bundle["whitelist"],
            policy=policy,
            base_plan=bundle["base_plan"],
            group_materials=bundle["group_materials"],
            mvinverse_pbr_evidence=bundle["mvinverse_pbr_evidence"],
            part_id_evidence=evidence,
            acknowledge_policy_fallback=True,
        )
        return bundle, evidence, policy

    def test_valid_hash_bound_exact_cover_is_accepted(self) -> None:
        self.assertEqual(_validate_policy_exact_cover_bundle(**self._bundle()), 2)

    def test_evidence_converged_policy_requires_hidden_independent_fallback(
        self,
    ) -> None:
        bundle, evidence, source_policy = self._evidence_replay_bundle()

        self.assertEqual(
            _validate_policy_exact_cover_bundle(
                **bundle,
                part_id_evidence=evidence,
                expected_policy_overrides=source_policy,
            ),
            2,
        )

        bundle["plan"]["assignments"][1]["provenance"] = {
            "tier": "neutral_default",
            "reason_codes": ["coherently-resealed-but-not-builder-authored"],
            "output_confidence_basis": "policy fallback; not evidence confidence",
            "sources": [],
        }
        self._reseal_bundle(bundle)
        with self.assertRaisesRegex(RuntimeError, "exact trusted replay"):
            _validate_policy_exact_cover_bundle(
                **bundle,
                part_id_evidence=evidence,
                expected_policy_overrides=source_policy,
            )

    def test_evidence_convergence_rejects_coherently_rebuilt_source_policy(
        self,
    ) -> None:
        bundle, evidence, source_policy = self._evidence_replay_bundle()
        attacker_policy = copy.deepcopy(source_policy)
        attacker_policy["default_material_id"] = "mdl:test#attacker-blue"
        bundle["whitelist"]["material_ids"].append("mdl:test#attacker-blue")
        evidence = self._part_id_evidence(bundle)
        bundle["plan"], bundle["audit"] = build_policy_exact_cover(
            registry=bundle["registry"],
            staged_result=bundle["staged_result"],
            confidence_gate=bundle["confidence_gate"],
            whitelist=bundle["whitelist"],
            policy=attacker_policy,
            base_plan=bundle["base_plan"],
            group_materials=bundle["group_materials"],
            mvinverse_pbr_evidence=bundle["mvinverse_pbr_evidence"],
            part_id_evidence=evidence,
            acknowledge_policy_fallback=True,
        )

        with self.assertRaisesRegex(RuntimeError, "source_policy_sha256"):
            _validate_policy_exact_cover_bundle(
                **bundle,
                part_id_evidence=evidence,
                expected_policy_overrides=source_policy,
            )

    def test_evidence_convergence_rejects_same_part_ids_from_other_registry(
        self,
    ) -> None:
        bundle, evidence, source_policy = self._evidence_replay_bundle()
        bundle["registry"]["asset_sha256"] = "b" * 64
        bundle["registry"]["parts"][0]["prim_path"] = "/Other/A/Mesh"

        with self.assertRaisesRegex(RuntimeError, "another registry"):
            _validate_policy_exact_cover_bundle(
                **bundle,
                part_id_evidence=evidence,
                expected_policy_overrides=source_policy,
            )

    def test_source_material_bind_subsets_are_covered_before_apply(self) -> None:
        bundle = self._bundle()
        for part in bundle["registry"]["parts"]:
            part["face_count"] = 4
            part["existing_material_bind_face_subsets"] = []
        bundle["registry"]["parts"][0][
            "existing_material_bind_face_subsets"
        ] = [
            {
                "subset_name": "Diffuse_64",
                "family_name": "materialBind",
                "element_type": "face",
                "face_indices": [0, 1],
            }
        ]
        bundle["plan"]["assignments"][0]["face_subsets"] = [
            {
                "subset_name": "Diffuse_64",
                "material_id": "mdl:test#material",
                "face_indices": [0, 1],
            }
        ]
        bundle["audit"]["summary"].update(
            {
                "source_material_bind_subset_part_count": 1,
                "source_material_bind_subset_count": 1,
            }
        )
        self._reseal_bundle(bundle)

        self.assertEqual(_validate_policy_exact_cover_bundle(**bundle), 2)

        bundle["plan"]["assignments"][0].pop("face_subsets")
        self._reseal_bundle(bundle)
        with self.assertRaisesRegex(RuntimeError, "omits source materialBind subsets"):
            _validate_policy_exact_cover_bundle(**bundle)

    def test_tampered_plan_is_rejected_before_apply(self) -> None:
        bundle = self._bundle()
        bundle["plan"]["assignments"][0]["material_id"] = "mdl:tampered#material"
        with self.assertRaisesRegex(RuntimeError, "output plan hash"):
            _validate_policy_exact_cover_bundle(**bundle)

    def test_missing_registry_part_is_rejected_before_apply(self) -> None:
        bundle = self._bundle()
        bundle["plan"]["assignments"].pop()
        bundle["audit"]["output_plan_sha256"] = canonical_sha256(bundle["plan"])
        bundle["audit"]["summary"]["output_assignment_count"] = 1
        bundle["audit"]["summary"]["policy_fallback_count"] = 1
        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            _validate_policy_exact_cover_bundle(**bundle)

    def test_step_reference_policy_requires_neutralized_source_visuals(
        self,
    ) -> None:
        bundle = self._bundle()
        policy = bundle["audit"]["policy"]
        policy["source_visual_strategy"] = "neutralize_unverified"
        for assignment in bundle["plan"]["assignments"]:
            assignment["provenance"] = {
                "tier": "source_preserve_unavailable_neutral_fallback"
            }
        bundle["audit"]["source_visual_strategy"] = "neutralize_unverified"
        bundle["audit"]["summary"].update(
            {
                "source_visual_preserve_count": 0,
                "source_preserve_unavailable_neutral_fallback_count": 2,
                "neutral_default_count": 0,
            }
        )
        bundle["plan"]["provenance"]["policy_sha256"] = canonical_sha256(policy)
        bundle["audit"]["input_hashes"] = copy.deepcopy(bundle["plan"]["provenance"])
        bundle["audit"]["output_plan_sha256"] = canonical_sha256(bundle["plan"])

        self.assertEqual(
            _validate_policy_exact_cover_bundle(
                **bundle,
                expected_source_visual_strategy="neutralize_unverified",
            ),
            2,
        )

        bundle["plan"]["assignments"][0]["provenance"]["tier"] = (
            "source_visual_preserve"
        )
        bundle["audit"]["summary"].update(
            {
                "source_visual_preserve_count": 1,
                "source_preserve_unavailable_neutral_fallback_count": 1,
            }
        )
        bundle["audit"]["output_plan_sha256"] = canonical_sha256(bundle["plan"])
        with self.assertRaisesRegex(RuntimeError, "palette fusion"):
            _validate_policy_exact_cover_bundle(
                **bundle,
                expected_source_visual_strategy="neutralize_unverified",
            )

    def test_corroborated_source_visual_can_bind_confirmed_nvidia_mdl(
        self,
    ) -> None:
        part_ids = [f"P{index:04d}" for index in range(1, 5)]
        diffuse = [0.02, 0.4, 0.9]
        source_signature = {
            "shader_id": "UsdPreviewSurface",
            "diffuse_color": diffuse,
            "metallic": 0.5,
            "roughness": 0.5,
            "opacity": 1.0,
        }
        geometry_signature = {
            "point_count": 256,
            "face_count": 256,
            "sorted_bbox_extents": [5.0, 5.0, 8.0],
        }
        source_digest = canonical_sha256(source_signature)
        geometry_digest = canonical_sha256(geometry_signature)
        corroboration_record = {
            "canonical_group_id": "G05",
            "canonical_color_family": "blue",
            "canonical_group_association_basis": None,
            "canonical_source_view_ids": ["iso", "top"],
            "source_visual_signature_sha256": source_digest,
            "source_signature_count": 4,
            "registry_fraction": 0.05,
            "geometry_signature_sha256": geometry_digest,
            "geometry_repeat_count": 4,
        }
        raw_parts = []
        for index in range(1, 81):
            part_id = f"P{index:04d}"
            part = {
                "part_id": part_id,
                "prim_path": f"/Asset/{part_id}/Mesh",
                "point_count": 256,
                "face_count": 256,
                "world_bbox": [[float(index), 0.0, 0.0], [float(index + 5), 5.0, 8.0]],
            }
            if part_id in part_ids:
                part.update(
                    {
                        "existing_visual_material": f"/Asset/Looks/{part_id}",
                        "existing_visual_material_properties": {
                            "shader_id": "UsdPreviewSurface",
                            "diffuseColor": diffuse,
                            "metallic": 0.5,
                            "roughness": 0.5,
                            "opacity": 1.0,
                        },
                    }
                )
            raw_parts.append(part)

        material_id = (
            "mdl:vMaterials_2/Plastic/Polycarbonate_Opaque.mdl"
            "#Polycarbonate_Blue"
        )
        assignments = []
        for part_id in part_ids:
            part = next(item for item in raw_parts if item["part_id"] == part_id)
            source_path = part["existing_visual_material"]
            assignments.append(
                {
                    "part_id": part_id,
                    "material_id": material_id,
                    "provenance": {
                        "tier": CORROBORATED_SOURCE_MDL_TIER,
                        "canonical_group_id": "G05",
                        "supporting_view_ids": ["iso", "top"],
                        "canonical_group_assignment_basis": (
                            "photo_corroborated_rare_repeated_source_visual"
                        ),
                        "reason_codes": [
                            "REFERENCE_PALETTE_MULTIVIEW_COLOR_CORROBORATION",
                            "RARE_SOURCE_VISUAL_SIGNATURE",
                            "REPEATED_GEOMETRY_SOURCE_LOCATOR",
                            "QWEN_CONFIRMED_NVIDIA_MDL_SELECTION",
                            "SOURCE_VISUAL_SIGNATURE_REPLACED_BY_NVIDIA_MDL",
                        ],
                        "source_visual_material": {
                            "material_prim_path": source_path,
                            "binding_sha256": canonical_sha256(
                                {
                                    "part_id": part_id,
                                    "prim_path": part["prim_path"],
                                    "source_visual_material_prim_path": source_path,
                                }
                            ),
                        },
                        "source_visual_corroboration": {
                            **copy.deepcopy(corroboration_record),
                            "confirmed_material_id": material_id,
                        },
                    },
                }
            )
        audit = {
            "corroborated_source_visual": {
                "thresholds": {
                    "maximum_registry_fraction": 0.05,
                    "minimum_source_signature_count": 4,
                    "minimum_repeated_geometry_count": 4,
                    "minimum_raw_and_linear_saturation": 0.5,
                    "minimum_opacity": 0.99,
                },
                "maximum_source_signature_count": 4,
                "eligible_part_ids": part_ids,
                "applied_part_ids": part_ids,
                "preserved_part_ids": [],
                "nvidia_mdl_replacement_part_ids": part_ids,
                "groups": [
                    {
                        "group_id": "G05",
                        "canonical_color_family": "blue",
                        "canonical_group_association_basis": None,
                        "canonical_source_view_ids": ["iso", "top"],
                        "source_visual_signature_sha256": source_digest,
                        "source_signature_count": 4,
                        "geometry_cohorts": [
                            {
                                **geometry_signature,
                                "geometry_signature_sha256": geometry_digest,
                                "repeat_count": 4,
                                "part_ids": part_ids,
                            }
                        ],
                        "eligible_part_ids": part_ids,
                    }
                ],
            }
        }
        palette_fusion = {
            "schema_version": "qwen-multiview-palette-fusion/v1",
            "canonical_palette": {
                "schema_version": "qwen-canonical-material-palette/v1",
                "groups": [
                    {
                        "group_id": "G05",
                        "base_color": "blue",
                        "source_view_ids": ["iso", "top"],
                        "distinct_view_count": 2,
                        "singleton": False,
                        "family_hint": "plastic",
                    }
                ],
            },
        }
        self.assertEqual(
            _validate_corroborated_source_visual_assignments(
                assignments=assignments,
                raw_parts=raw_parts,
                audit=audit,
                palette_fusion=palette_fusion,
            ),
            (0, 4),
        )

        assignments[0]["provenance"]["canonical_group_id"] = "G_OTHER"
        with self.assertRaisesRegex(RuntimeError, "independently corroborated"):
            _validate_corroborated_source_visual_assignments(
                assignments=assignments,
                raw_parts=raw_parts,
                audit=audit,
                palette_fusion=palette_fusion,
            )
        assignments[0]["provenance"]["canonical_group_id"] = "G05"

        assignments[0]["material_id"] = "mdl:tampered#material"
        with self.assertRaisesRegex(RuntimeError, "independently corroborated"):
            _validate_corroborated_source_visual_assignments(
                assignments=assignments,
                raw_parts=raw_parts,
                audit=audit,
                palette_fusion=palette_fusion,
            )

    def _quality_repair_bundle(self) -> dict[str, dict]:
        policy = self._bundle()
        policy["whitelist"]["material_ids"].append("mdl:test#green")
        for assignment in policy["plan"]["assignments"]:
            assignment.update(
                {
                    "semantic": "neutral component",
                    "provenance": {
                        "tier": "neutral_default",
                        "reason_codes": [],
                        "output_confidence_basis": (POLICY_FALLBACK_CONFIDENCE_BASIS),
                        "sources": [],
                    },
                }
            )
        policy["plan"]["provenance"]["whitelist_sha256"] = canonical_sha256(
            policy["whitelist"]
        )
        policy["audit"]["input_hashes"] = copy.deepcopy(policy["plan"]["provenance"])
        policy["audit"]["output_plan_sha256"] = canonical_sha256(policy["plan"])

        evidence = {
            "quality_report": {"schema_version": "qwen-reference-render-comparison/v1"},
            "palette_fusion": {
                "schema_version": "qwen-multiview-palette-fusion/v1",
                "canonical_palette": {
                    "groups": [
                        {
                            "group_id": "G01",
                            "visual_description": "green painted panel",
                        }
                    ]
                },
            },
            "spatial_report": {"schema_version": "qwen-spatial-mapping-audit/v1"},
            "spatial_gate_audit": {"schema_version": "qwen-spatial-mapping-gate/v1"},
            "mapping_consensus": {"schema_version": "qwen-mapping-consensus-audit/v1"},
            "geometry_risk": {
                "schema_version": "qwen-geometry-uniform-material-risk/v1"
            },
            "group_materials": {
                "schema_version": "qwen-palette-material/v1",
                "selections": [
                    {
                        "group_id": "G01",
                        "material_id": "mdl:test#green",
                        "confirmed": True,
                    }
                ],
            },
            "mvinverse_pbr_evidence": {
                "schema_version": "qwen-mvinverse-pbr-evidence/v1"
            },
        }
        input_hashes = {
            "baseline_plan_sha256": canonical_sha256(policy["plan"]),
            "baseline_policy_audit_sha256": canonical_sha256(policy["audit"]),
            **{
                f"{name}_sha256": canonical_sha256(document)
                for name, document in evidence.items()
            },
            "registry_sha256": canonical_sha256(policy["registry"]),
            "whitelist_sha256": canonical_sha256(policy["whitelist"]),
        }
        output_plan = copy.deepcopy(policy["plan"])
        changed = output_plan["assignments"][0]
        changed["material_id"] = "mdl:test#green"
        changed["semantic"] = "green painted panel"
        changed["provenance"] = {
            "tier": "qa_repair_candidate",
            "reason_codes": list(QUALITY_REPAIR_REASON_CODES),
            "output_confidence_basis": POLICY_FALLBACK_CONFIDENCE_BASIS,
            "sources": [],
            "canonical_group_id": "G01",
            "baseline_material_id": "mdl:test#material",
            "baseline_tier": "neutral_default",
            "supporting_view_ids": ["ref_a", "ref_b"],
            "supporting_content_cluster_ids": ["CONTENT_01", "CONTENT_02"],
            "supporting_pose_cluster_ids": ["front", "side"],
        }
        output_plan["provenance"] = copy.deepcopy(policy["plan"]["provenance"])
        output_plan["provenance"][QUALITY_REPAIR_PROVENANCE_FIELD] = {
            "mode": QUALITY_REPAIR_PLAN_MODE,
            "input_hashes": input_hashes,
            "changed_part_ids": ["P0001"],
        }
        repair_audit = {
            "schema_version": QUALITY_REPAIR_REPORT_SCHEMA_VERSION,
            "summary": {
                "status": "REPAIRED",
                "changed_count": 1,
                "no_op": False,
                "exact_cover": True,
                "all_materials_in_whitelist": True,
                "maximum_orchestrator_retry_count": 1,
            },
            "input_hashes": input_hashes,
            "changes": [
                {
                    "part_id": "P0001",
                    "canonical_group_id": "G01",
                    "material_id": "mdl:test#green",
                    "supporting_view_ids": ["ref_a", "ref_b"],
                    "supporting_content_cluster_ids": [
                        "CONTENT_01",
                        "CONTENT_02",
                    ],
                    "supporting_pose_cluster_ids": ["front", "side"],
                    "old_material_id": "mdl:test#material",
                    "new_material_id": "mdl:test#green",
                }
            ],
            "localization_lanes": [
                {
                    "part_id": "P0001",
                    "canonical_group_id": "G01",
                    "lane": "stable_spatial_multiview",
                }
            ],
            "mvinverse": {
                "enabled": True,
                "parameterized_part_ids": [],
                "skipped": [],
            },
            "output_plan_sha256": canonical_sha256(output_plan),
        }
        return {
            "plan": output_plan,
            "audit": repair_audit,
            "baseline_plan": policy["plan"],
            "baseline_policy_audit": policy["audit"],
            **evidence,
            "registry": policy["registry"],
            "whitelist": policy["whitelist"],
        }

    def _parameterized_quality_repair_bundle(self) -> dict[str, dict]:
        bundle = self._quality_repair_bundle()
        source_material = (
            "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted_Army_Green"
        )
        bundle["whitelist"]["material_ids"].extend(
            [source_material, GENERIC_STEEL_PAINTED]
        )
        bundle["group_materials"]["selections"][0]["material_id"] = source_material
        evidence = {
            "schema_version": "qwen-mvinverse-pbr-evidence/v1",
            "inputs": {"integrity_verified": True},
            "views": [{"view_id": "ref_a"}, {"view_id": "ref_b"}],
            "groups": [
                {
                    "group_id": "G01",
                    "surface_class": "dielectric",
                    "contributing_view_ids": ["ref_a", "ref_b"],
                    "distinct_view_count": 2,
                    "albedo": {"sample_count": 2},
                    "metallic": {"sample_count": 2, "median": 0.1},
                    "roughness": {"sample_count": 2},
                    "suggestion": {
                        "decision": "auto",
                        "auto_parameter_eligible": True,
                        "base_color_srgb": [0.2, 0.5, 0.1],
                        "metallic": 0.0,
                        "roughness": 0.43,
                    },
                }
            ],
            "summary": {
                "view_count": 2,
                "canonical_group_count": 1,
                "auto_parameter_group_count": 1,
                "fail_closed": True,
                "usd_modified": False,
            },
        }
        bundle["mvinverse_pbr_evidence"] = evidence
        bundle["baseline_plan"]["provenance"]["whitelist_sha256"] = canonical_sha256(
            bundle["whitelist"]
        )
        bundle["baseline_policy_audit"]["input_hashes"] = copy.deepcopy(
            bundle["baseline_plan"]["provenance"]
        )
        bundle["baseline_policy_audit"]["output_plan_sha256"] = canonical_sha256(
            bundle["baseline_plan"]
        )
        input_hashes = {
            "baseline_plan_sha256": canonical_sha256(bundle["baseline_plan"]),
            "baseline_policy_audit_sha256": canonical_sha256(
                bundle["baseline_policy_audit"]
            ),
            "quality_report_sha256": canonical_sha256(bundle["quality_report"]),
            "palette_fusion_sha256": canonical_sha256(bundle["palette_fusion"]),
            "spatial_report_sha256": canonical_sha256(bundle["spatial_report"]),
            "spatial_gate_audit_sha256": canonical_sha256(bundle["spatial_gate_audit"]),
            "mapping_consensus_sha256": canonical_sha256(bundle["mapping_consensus"]),
            "geometry_risk_sha256": canonical_sha256(bundle["geometry_risk"]),
            "group_materials_sha256": canonical_sha256(bundle["group_materials"]),
            "mvinverse_pbr_evidence_sha256": canonical_sha256(evidence),
            "registry_sha256": canonical_sha256(bundle["registry"]),
            "whitelist_sha256": canonical_sha256(bundle["whitelist"]),
        }
        color_srgb = [0.2, 0.5, 0.1]
        paint_color = [
            value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
            for value in color_srgb
        ]
        parameters = {
            "paint_color": paint_color,
            "paint_roughness": 0.43,
        }
        mvinverse_audit = {
            "source_material_id": source_material,
            "output_material_id": source_material,
            "group_id": "G01",
            "tuning_profile_id": "nvidia_steel_painted",
            "parameterization_mode": "full_mvinverse_pbr",
            "contributing_view_ids": ["ref_a", "ref_b"],
            "base_color_srgb": color_srgb,
            "base_color_linear": paint_color,
            "observed_metallic": 0.1,
            "authored_metallic": 0.0,
            "roughness": 0.43,
            "authored_parameter_names": ["paint_color", "paint_roughness"],
            "reason_code": "MVINVERSE_AUTO_PARAMETER_ELIGIBLE",
        }
        changed = bundle["plan"]["assignments"][0]
        changed["material_id"] = source_material
        changed["parameters"] = parameters
        changed["provenance"]["mvinverse"] = mvinverse_audit
        bundle["plan"]["provenance"] = copy.deepcopy(
            bundle["baseline_plan"]["provenance"]
        )
        bundle["plan"]["provenance"][QUALITY_REPAIR_PROVENANCE_FIELD] = {
            "mode": QUALITY_REPAIR_PLAN_MODE,
            "input_hashes": input_hashes,
            "changed_part_ids": ["P0001"],
        }
        change = bundle["audit"]["changes"][0]
        change.update(
            {
                "material_id": source_material,
                "new_material_id": source_material,
                "confirmed_source_material_id": source_material,
                "mvinverse_parameterized": True,
            }
        )
        bundle["audit"]["input_hashes"] = input_hashes
        bundle["audit"]["mvinverse"]["parameterized_part_ids"] = ["P0001"]
        bundle["audit"]["output_plan_sha256"] = canonical_sha256(bundle["plan"])
        return bundle

    @staticmethod
    def _refresh_quality_repair_hashes(bundle: dict[str, dict]) -> None:
        input_hashes = {
            "baseline_plan_sha256": canonical_sha256(bundle["baseline_plan"]),
            "baseline_policy_audit_sha256": canonical_sha256(
                bundle["baseline_policy_audit"]
            ),
            "quality_report_sha256": canonical_sha256(bundle["quality_report"]),
            "palette_fusion_sha256": canonical_sha256(bundle["palette_fusion"]),
            "spatial_report_sha256": canonical_sha256(bundle["spatial_report"]),
            "spatial_gate_audit_sha256": canonical_sha256(bundle["spatial_gate_audit"]),
            "mapping_consensus_sha256": canonical_sha256(bundle["mapping_consensus"]),
            "geometry_risk_sha256": canonical_sha256(bundle["geometry_risk"]),
            "group_materials_sha256": canonical_sha256(bundle["group_materials"]),
            "mvinverse_pbr_evidence_sha256": canonical_sha256(
                bundle["mvinverse_pbr_evidence"]
            ),
            "registry_sha256": canonical_sha256(bundle["registry"]),
            "whitelist_sha256": canonical_sha256(bundle["whitelist"]),
        }
        bundle["audit"]["input_hashes"] = input_hashes
        bundle["plan"]["provenance"][QUALITY_REPAIR_PROVENANCE_FIELD][
            "input_hashes"
        ] = input_hashes
        bundle["audit"]["output_plan_sha256"] = canonical_sha256(bundle["plan"])

    def _dominant_review_override_quality_repair_bundle(
        self,
    ) -> dict[str, dict]:
        bundle = self._quality_repair_bundle()
        target_scores = [
            {"canonical_group_id": "G01", "color_share": 0.96},
            {"canonical_group_id": "G02", "color_share": 0.01},
        ]
        bbox_scores = [
            {"canonical_group_id": "G01", "color_share": 0.95},
            {"canonical_group_id": "G02", "color_share": 0.01},
        ]
        offsets = ((-2, 0), (2, 0), (0, -2), (0, 2))
        bundle["spatial_report"] = {
            "schema_version": "qwen-spatial-mapping-audit/v1",
            "policy": {"minimum_semantic_confidence": 0.85},
            "reference_evidence": [
                {
                    "view_id": "ref_a",
                    "raw_sha256": "1" * 64,
                    "normalized_pixel_sha256": "2" * 64,
                    "content_cluster_id": "CONTENT_01",
                    "pose_cluster_id": "front",
                    "alignment_trusted": True,
                },
                {
                    "view_id": "ref_b",
                    "raw_sha256": "3" * 64,
                    "normalized_pixel_sha256": "4" * 64,
                    "content_cluster_id": "CONTENT_02",
                    "pose_cluster_id": "side",
                    "alignment_trusted": True,
                },
            ],
            "view_alignments": [
                {
                    "reference_view_id": "ref_a",
                    "trusted": True,
                    "reason_codes": [],
                    "score": 0.9,
                    "projection_iou": 0.9,
                    "ecc_status": "success",
                    "ecc_correlation": 0.9,
                    "ecc_transform_audit": {"constraints_passed": True},
                }
            ],
            "parts": [
                {
                    "part_id": "P0001",
                    "observations": [
                        {
                            "reference_view_id": "ref_a",
                            "declared_visible_pixels": 2048,
                            "projected_part_pixels": 2048,
                            "registration_label_stable": True,
                            "perturbation_label_stable": True,
                            "group_scores": target_scores,
                            "color_margin": 0.95,
                            "bbox_group_scores": bbox_scores,
                            "bbox_canonical_group_id": "G01",
                            "bbox_color_margin": 0.94,
                            "projection_perturbations": [
                                {
                                    "offset_pixels": [x, y],
                                    "sampled_reference_pixels": 2048,
                                    "canonical_group_id": "G01",
                                    "diagnostic_canonical_group_id": "G01",
                                    "best_color_share": 0.95,
                                    "color_margin": 0.9,
                                }
                                for x, y in offsets
                            ],
                            "canonical_palette_diagnostic": {
                                "direct_sample": {"group_scores": target_scores},
                                "bbox_sample": {"group_scores": bbox_scores},
                                "projection_perturbations": [
                                    {
                                        "offset_pixels": [x, y],
                                        "group_scores": target_scores,
                                    }
                                    for x, y in offsets
                                ],
                            },
                        }
                    ],
                    "semantic_votes": [
                        {
                            "view_id": "ref_a",
                            "status": "review",
                            "canonical_group_id": "G02",
                            "alignment_trusted": True,
                            "unique_canonical_join": True,
                            "pixel_gate_accepted": True,
                            "effective_confidence": 0.9,
                        },
                        {
                            "view_id": "ref_b",
                            "status": "matched",
                            "canonical_group_id": "G01",
                            "alignment_trusted": True,
                            "unique_canonical_join": True,
                            "pixel_gate_accepted": True,
                            "effective_confidence": 0.85,
                        },
                    ],
                }
            ],
        }
        bundle["spatial_gate_audit"] = {
            "schema_version": "qwen-spatial-mapping-gate/v1",
            "decisions": [
                {
                    "part_id": "P0001",
                    "output_status": "review",
                    "output_group_id": None,
                }
            ],
        }
        bundle["mapping_consensus"] = {
            "schema_version": "qwen-mapping-consensus-audit/v1",
            "decisions": [
                {
                    "part_id": "P0001",
                    "output_status": "review",
                    "output_group_id": None,
                }
            ],
        }
        change = bundle["audit"]["changes"][0]
        change.update(
            {
                "supporting_view_ids": ["ref_a"],
                "supporting_content_cluster_ids": ["CONTENT_01"],
                "supporting_pose_cluster_ids": ["front"],
                "semantic_conflict_override_view_ids": ["ref_a"],
                "semantic_anchor_view_ids": ["ref_b"],
            }
        )
        bundle["audit"]["localization_lanes"][0]["lane"] = (
            "dominant_chromatic_residual_exact_single_view"
        )
        assignment_provenance = bundle["plan"]["assignments"][0]["provenance"]
        assignment_provenance.update(
            {
                "supporting_view_ids": ["ref_a"],
                "supporting_content_cluster_ids": ["CONTENT_01"],
                "supporting_pose_cluster_ids": ["front"],
                "semantic_review_override": {
                    "conflict_view_ids": ["ref_a"],
                    "anchor_view_ids": ["ref_b"],
                },
            }
        )
        self._refresh_quality_repair_hashes(bundle)
        return bundle

    def _dark_quality_repair_bundle(self) -> dict[str, dict]:
        bundle = self._quality_repair_bundle()
        group = bundle["palette_fusion"]["canonical_palette"]["groups"][0]
        group.update(
            {
                "visual_description": "matte black painted arm",
                "base_color": "black",
                "singleton": False,
                "distinct_view_count": 2,
            }
        )
        front_reference = {
            "view_id": "front",
            "raw_sha256": "1" * 64,
            "normalized_pixel_sha256": "2" * 64,
            "content_cluster_id": "CONTENT_FRONT",
            "pose_cluster_id": "right",
            "alignment_trusted": True,
            "alignment_score": 0.95,
        }
        bundle["spatial_report"] = {
            "schema_version": "qwen-spatial-mapping-audit/v1",
            "policy": {"minimum_semantic_confidence": 0.85},
            "reference_evidence": [
                front_reference,
                {
                    "view_id": "iso",
                    "raw_sha256": "3" * 64,
                    "normalized_pixel_sha256": "4" * 64,
                    "content_cluster_id": "CONTENT_ISO",
                    "pose_cluster_id": None,
                    "alignment_trusted": False,
                    "alignment_score": 0.4,
                },
                {
                    "view_id": "side",
                    "raw_sha256": "5" * 64,
                    "normalized_pixel_sha256": "6" * 64,
                    "content_cluster_id": "CONTENT_SIDE",
                    "pose_cluster_id": "front",
                    "alignment_trusted": True,
                    "alignment_score": 0.9,
                },
            ],
            "view_alignments": [
                {
                    "reference_view_id": "front",
                    "trusted": True,
                    "reason_codes": [],
                    "score": 0.95,
                    "projection_score": 0.95,
                    "projection_iou": 0.95,
                    "ecc_status": "success",
                    "ecc_correlation": 0.95,
                    "ecc_transform_audit": {"constraints_passed": True},
                }
            ],
            "parts": [],
        }
        diagnostic = {
            "status": "resolved",
            "reason_codes": [],
            "evidence_scope": "dark_on_black_foreground_repair_only",
            "canonical_group_id": "G01",
            "canonical_source_view_ids": ["iso", "side"],
            "alternative_canonical_group_ids": [],
            "projected_part_pixels": 1000,
            "normalized_projected_pixels": 1000,
            "normalization": {
                "long_edge_pixels": 512,
                "original_size": [1000, 500],
                "normalized_size": [512, 256],
                "scale": 0.512,
            },
            "alignment": {
                "trusted": True,
                "reason_codes_empty": True,
                "score": 0.95,
                "projection_score": 0.95,
                "projection_iou": 0.95,
                "ecc_status": "success",
                "ecc_correlation": 0.95,
                "transform_constraints_passed": True,
                "strong": True,
            },
            "background": {
                "median_bgr": [0.0, 0.0, 0.0],
                "border_distance_p95": 2.0,
                "distance_threshold": 12.0,
            },
            "thresholds": {
                "normalized_long_edge_pixels": 512,
                "minimum_normalized_projected_pixels": 96,
                "near_black_max_channel_exclusive": 97,
                "near_black_max_channel_spread": 32,
                "minimum_near_black_share": 0.6,
                "minimum_non_background_pixels": 24,
                "minimum_dark_signal_share": 0.2,
                "minimum_dark_signal_purity": 0.45,
                "core_distance_pixels": 2.2,
                "minimum_core_pixels": 16,
                "minimum_core_dark_signal_share": 0.25,
                "minimum_adaptive_edge_density": 0.25,
                "minimum_null_offset_pixels": 7,
                "minimum_null_valid_area_ratio": 0.8,
                "minimum_valid_null_shifts": 4,
                "minimum_null_q75_margin": 0.1,
                "minimum_alignment_score": 0.85,
                "minimum_projection_score": 0.85,
                "minimum_projection_iou": 0.85,
                "minimum_ecc_correlation": 0.9,
            },
            "near_black_pixels": 900,
            "near_black_share": 0.9,
            "non_background_pixels": 500,
            "non_background_share": 0.5,
            "dark_signal_pixels": 500,
            "dark_signal_share": 0.5,
            "dark_signal_purity": 1.0,
            "core_pixels": 400,
            "core_dark_signal_pixels": 200,
            "core_dark_signal_share": 0.5,
            "core_distance_pixels": 2.2,
            "adaptive_edge_pixels": 300,
            "adaptive_edge_density": 0.3,
            "adaptive_edge_threshold": 16.0,
            "border_gradient_p99": 10.0,
            "canny_low_threshold": 8,
            "canny_high_threshold": 16,
            "canny_edge_pixels": 100,
            "canny_edge_density": 0.1,
            "null_shifts": [
                {
                    "offset_pixels": [x, y],
                    "retained_pixels": 1000,
                    "valid_area_ratio": 1.0,
                    "valid": True,
                    "dark_signal_pixels": 100,
                    "dark_signal_share": 0.1,
                    "mask_sha256": f"{index:x}" * 64,
                }
                for index, (x, y) in enumerate(
                    (
                        (-20, 0),
                        (20, 0),
                        (0, -10),
                        (0, 10),
                        (-20, -10),
                        (-20, 10),
                        (20, -10),
                        (20, 10),
                    ),
                    start=1,
                )
            ],
            "valid_null_shift_count": 8,
            "null_dark_signal_share_q75": 0.1,
            "dark_signal_null_margin": 0.4,
            "normalized_reference_pixel_sha256": "7" * 64,
            "normalized_projected_mask_sha256": "8" * 64,
            "normalized_near_black_mask_sha256": "9" * 64,
            "normalized_non_background_mask_sha256": "a" * 64,
            "normalized_dark_signal_mask_sha256": "b" * 64,
            "normalized_adaptive_edge_mask_sha256": "c" * 64,
        }
        diagnostic["diagnostic_sha256"] = canonical_sha256(diagnostic)
        bundle["spatial_report"]["parts"] = [
            {
                "part_id": "P0001",
                "observations": [
                    {
                        "reference_view_id": "front",
                        "projected_part_pixels": 1000,
                        "dark_foreground_diagnostic": diagnostic,
                    }
                ],
                "semantic_votes": [
                    {
                        "status": "review",
                        "canonical_group_id": "G02",
                        "alignment_trusted": True,
                        "unique_canonical_join": True,
                        "pixel_gate_accepted": True,
                        "effective_confidence": 0.95,
                    }
                ],
            }
        ]
        bundle["spatial_gate_audit"] = {
            "schema_version": "qwen-spatial-mapping-gate/v1",
            "decisions": [
                {
                    "part_id": "P0001",
                    "output_status": "review",
                    "output_group_id": "G02",
                }
            ],
        }
        bundle["mapping_consensus"] = {
            "schema_version": "qwen-mapping-consensus-audit/v1",
            "decisions": [
                {
                    "part_id": "P0001",
                    "output_status": "review",
                    "output_group_id": "G02",
                }
            ],
        }

        def quality_view(
            view_id: str,
            render_view_id: str,
            image_sha256: str,
            *,
            failed: bool,
        ) -> dict:
            reference_categories = {
                "black": 0.125 if view_id == "front" else 0.0,
                "achromatic_dark": 0.125 if view_id == "front" else 0.0,
                "achromatic_mid": 0.75 if view_id == "front" else 1.0,
            }
            render_categories = {
                "black": 0.03125 if view_id == "front" else 0.0,
                "achromatic_dark": 0.03125 if view_id == "front" else 0.0,
                "achromatic_mid": 0.9375 if view_id == "front" else 1.0,
            }
            return {
                "reference_view_id": view_id,
                "render_view_id": render_view_id,
                "status": "FAIL" if failed else "PASS",
                "mapping": {
                    "selected_render_view_id": render_view_id,
                    "reasons": [],
                },
                "reference": {
                    "image_sha256": image_sha256,
                    "trusted_evidence": {"usable": True},
                    "foreground": {"pixel_count": 10000},
                },
                "render": {"foreground": {"pixel_count": 10000}},
                "alignment": {
                    "score": 0.95,
                    "silhouette_iou": 0.95,
                    "edge_f1_tolerance_3px": 0.95,
                    "profile_similarity": 0.95,
                    "bbox_aspect_similarity": 0.95,
                },
                "material_color": {
                    "trusted_evidence_group_recall": {"groups": []},
                    "reference_distribution": {
                        "sample_step": 1,
                        "sampled_pixels": 10000,
                        "category_distribution": reference_categories,
                    },
                    "render_distribution": {
                        "sample_step": 1,
                        "sampled_pixels": 10000,
                        "category_distribution": render_categories,
                    },
                },
            }

        bundle["quality_report"] = {
            "schema_version": "qwen-reference-render-comparison/v1",
            "inputs": {
                "reference_manifest_sha256": "d" * 64,
                "mapping_mode": "deterministic_global_registration",
                "selected_view_mapping": {"front": "right", "side": "front"},
            },
            "thresholds": {"minimum_evidence_group_recall": 0.2},
            "aggregate": {"status": "FAIL", "comparable_view_count": 2},
            "views": [
                quality_view("front", "right", "1" * 64, failed=True),
                quality_view("side", "front", "5" * 64, failed=False),
            ],
        }
        support = {
            "reference_view_id": "front",
            "local_group_id": "__canonical_dark__:G01",
            "reference_sha256": "1" * 64,
            "content_cluster_id": "CONTENT_FRONT",
            "pose_cluster_id": "right",
            "recall": 0.25,
            "mass_recall": 0.25,
            "deficit_sources": ["dark_foreground_achromatic_residual"],
            "reference_share": 0.25,
            "observed_render_share": 0.0625,
            "deficit_share": 0.1875,
            "normalized_reference_pixels": 10000,
            "render_foreground_pixels": 10000,
            "budget_pixels": 1875,
            "budget_limit_pixels": 2531,
            "alignment": {
                "bbox_aspect_similarity": 0.95,
                "edge_f1_tolerance_3px": 0.95,
                "profile_similarity": 0.95,
                "score": 0.95,
                "silhouette_iou": 0.95,
            },
        }
        diagnostic_summary = {
            "diagnostic_sha256": diagnostic["diagnostic_sha256"],
            "projected_part_pixels": 1000,
            "normalized_projected_pixels": 1000,
            "dark_signal_share": 0.5,
            "dark_signal_purity": 1.0,
            "dark_signal_null_margin": 0.4,
            "adaptive_edge_density": 0.3,
            "evidence_strength": 2.1999999999999997,
            "estimated_contribution_pixels": 500,
        }
        budget_fields = {
            "budget_pixels": 1875,
            "budget_limit_pixels": 2531,
            "existing_contribution_pixels": 0,
            "estimated_contribution_pixels": 500,
            "selected_contribution_pixels": 500,
            "cumulative_contribution_pixels": 500,
        }
        change = bundle["audit"]["changes"][0]
        change.update(
            {
                "supporting_view_ids": ["front"],
                "supporting_content_cluster_ids": ["CONTENT_FRONT"],
                "supporting_pose_cluster_ids": ["right"],
                "dark_residual_support": support,
                "dark_foreground_diagnostic": diagnostic_summary,
                **budget_fields,
            }
        )
        bundle["audit"]["localization_lanes"][0]["lane"] = (
            "dark_foreground_achromatic_residual_exact_projection"
        )
        bundle["audit"]["group_diagnostics"] = [
            {
                "canonical_group_id": "G01",
                "repairable": False,
                "supporting_views": [],
                "dark_residual_repairable": True,
                "dark_residual_reason_codes": [],
                "dark_residual_supporting_views": [support],
            }
        ]
        bundle["audit"]["dark_residual_budgets"] = [
            {
                "canonical_group_id": "G01",
                "reference_view_id": "front",
                "dark_residual_support": support,
                "budget_pixels": 1875,
                "budget_limit_pixels": 2531,
                "per_part_limit_pixels": 2343,
                "existing_contribution_pixels": 0,
                "existing_parts": [],
                "candidates": [
                    {
                        "part_id": "P0001",
                        "evidence_strength": 2.1999999999999997,
                        "diagnostic_sha256": diagnostic["diagnostic_sha256"],
                        "estimated_contribution_pixels": 500,
                        "selected": True,
                        "reason_code": None,
                        "cumulative_contribution_pixels": 500,
                    }
                ],
                "selected_part_ids": ["P0001"],
                "selected_contribution_pixels": 500,
                "total_contribution_pixels": 500,
            }
        ]
        output_assignment = bundle["plan"]["assignments"][0]
        output_assignment["semantic"] = "matte black painted arm"
        output_assignment["provenance"].update(
            {
                "reason_codes": [
                    "QA_DARK_FOREGROUND_ACHROMATIC_RESIDUAL",
                    "QA_TRUSTED_PART_GROUP_LOCALIZATION",
                    "QA_CONFIRMED_WHITELIST_MATERIAL",
                ],
                "supporting_view_ids": ["front"],
                "supporting_content_cluster_ids": ["CONTENT_FRONT"],
                "supporting_pose_cluster_ids": ["right"],
                "dark_foreground_residual": {
                    "lane": ("dark_foreground_achromatic_residual_exact_projection"),
                    "support": support,
                    "diagnostic_sha256": diagnostic["diagnostic_sha256"],
                    **budget_fields,
                },
            }
        )
        input_hashes = {
            "baseline_plan_sha256": canonical_sha256(bundle["baseline_plan"]),
            "baseline_policy_audit_sha256": canonical_sha256(
                bundle["baseline_policy_audit"]
            ),
            "quality_report_sha256": canonical_sha256(bundle["quality_report"]),
            "palette_fusion_sha256": canonical_sha256(bundle["palette_fusion"]),
            "spatial_report_sha256": canonical_sha256(bundle["spatial_report"]),
            "spatial_gate_audit_sha256": canonical_sha256(bundle["spatial_gate_audit"]),
            "mapping_consensus_sha256": canonical_sha256(bundle["mapping_consensus"]),
            "geometry_risk_sha256": canonical_sha256(bundle["geometry_risk"]),
            "group_materials_sha256": canonical_sha256(bundle["group_materials"]),
            "mvinverse_pbr_evidence_sha256": canonical_sha256(
                bundle["mvinverse_pbr_evidence"]
            ),
            "registry_sha256": canonical_sha256(bundle["registry"]),
            "whitelist_sha256": canonical_sha256(bundle["whitelist"]),
        }
        bundle["audit"]["input_hashes"] = input_hashes
        bundle["plan"]["provenance"][QUALITY_REPAIR_PROVENANCE_FIELD][
            "input_hashes"
        ] = input_hashes
        bundle["audit"]["output_plan_sha256"] = canonical_sha256(bundle["plan"])
        return bundle

    def _repeated_geometry_dark_quality_repair_bundle(self) -> dict[str, dict]:
        bundle = self._dark_quality_repair_bundle()
        group = bundle["palette_fusion"]["canonical_palette"]["groups"][0]
        group["source_view_ids"] = ["alt_a", "alt_b"]

        registry = bundle["registry"]
        baseline_assignments = bundle["baseline_plan"]["assignments"]
        output_assignments = bundle["plan"]["assignments"]
        source_properties = {
            "shader_path": "/Asset/Looks/Source/PreviewSurface",
            "shader_id": "UsdPreviewSurface",
            "diffuseColor": [1.0, 1.0, 0.0],
            "metallic": 0.5,
            "roughness": 0.5,
            "opacity": 1.0,
        }
        for index, part in enumerate(registry["parts"], start=1):
            part.update(
                {
                    "point_count": 100,
                    "face_count": 80,
                    "world_bbox": [
                        [float(index * 10), 0.0, 0.0],
                        [float(index * 10 + 1), 2.0, 3.0],
                    ],
                    "existing_visual_material_properties": copy.deepcopy(
                        source_properties
                    ),
                }
            )
        for index in range(3, 41):
            part_id = f"P{index:04d}"
            registry["parts"].append(
                {
                    "part_id": part_id,
                    "prim_path": f"/Asset/Filler{index}/Mesh",
                    "point_count": 100 + index,
                    "face_count": 80 + index,
                    "world_bbox": [
                        [float(index * 10), 0.0, 0.0],
                        [float(index * 10 + index), 2.0, 3.0],
                    ],
                    "existing_visual_material_properties": {
                        **copy.deepcopy(source_properties),
                        "diffuseColor": [0.5, 0.5, 0.5],
                    },
                }
            )
            baseline = copy.deepcopy(baseline_assignments[1])
            baseline["part_id"] = part_id
            baseline_assignments.append(baseline)
            output_assignments.append(copy.deepcopy(baseline))
        registry["part_count"] = 40

        bundle["geometry_risk"] = {
            "schema_version": "qwen-geometry-uniform-material-risk/v1",
            "parts": [
                {
                    "part_id": part["part_id"],
                    "risk": {"multi_material_risk": False},
                }
                for part in registry["parts"]
            ],
        }
        bundle["quality_report"] = {
            "schema_version": "qwen-reference-render-comparison/v1",
            "inputs": {
                "reference_manifest_sha256": "d" * 64,
                "mapping_mode": "deterministic_global_registration",
                "selected_view_mapping": {"front": "right"},
            },
            "thresholds": {"minimum_evidence_group_recall": 0.5},
            "aggregate": {"status": "FAIL", "comparable_view_count": 1},
            "views": [
                {
                    "reference_view_id": "front",
                    "render_view_id": "right",
                    "status": "FAIL",
                    "render": {"foreground": {"pixel_count": 10000}},
                    "material_color": {
                        "trusted_evidence_group_recall": {
                            "groups": [
                                {
                                    "group_id": "G05",
                                    "base_colors": ["black"],
                                    "render_color_bins": [
                                        "black",
                                        "achromatic_dark",
                                    ],
                                    "required_render_share": 0.1,
                                    "observed_render_share": 0.04,
                                    "recall": 0.4,
                                }
                            ]
                        }
                    },
                }
            ],
        }
        reference_records = [
            {
                "view_id": "front",
                "raw_sha256": "1" * 64,
                "normalized_pixel_sha256": "2" * 64,
                "content_cluster_id": "CONTENT_FRONT",
                "pose_cluster_id": "right",
                "alignment_trusted": True,
                "alignment_score": 0.82,
            },
            {
                "view_id": "alt_a",
                "raw_sha256": "3" * 64,
                "normalized_pixel_sha256": "4" * 64,
                "content_cluster_id": "CONTENT_ALT_A",
                "pose_cluster_id": "rear",
                "alignment_trusted": True,
                "alignment_score": 0.9,
            },
            {
                "view_id": "alt_b",
                "raw_sha256": "5" * 64,
                "normalized_pixel_sha256": "6" * 64,
                "content_cluster_id": "CONTENT_ALT_B",
                "pose_cluster_id": "top",
                "alignment_trusted": True,
                "alignment_score": 0.9,
            },
        ]
        alignment = {
            "reference_view_id": "front",
            "trusted": True,
            "reason_codes": [],
            "score": 0.82,
            "projection_score": 0.9,
            "projection_iou": 0.9,
            "ecc_status": "success",
            "ecc_correlation": 0.95,
            "ecc_transform_audit": {"constraints_passed": True},
        }
        base_diagnostic = copy.deepcopy(
            bundle["spatial_report"]["parts"][0]["observations"][0][
                "dark_foreground_diagnostic"
            ]
        )
        base_diagnostic.update(
            {
                "status": "rejected",
                "reason_codes": [
                    "DARK_CANONICAL_GROUP_CONFLICT",
                    "DARK_ALIGNMENT_NOT_STRONG",
                ],
                "canonical_source_view_ids": ["alt_a", "alt_b"],
                "alternative_canonical_group_ids": ["G04"],
                "projected_part_pixels": 300,
                "alignment": {
                    "trusted": True,
                    "reason_codes_empty": True,
                    "score": 0.82,
                    "projection_score": 0.9,
                    "projection_iou": 0.9,
                    "ecc_status": "success",
                    "ecc_correlation": 0.95,
                    "transform_constraints_passed": True,
                    "strong": False,
                },
            }
        )
        base_diagnostic.pop("diagnostic_sha256", None)
        base_diagnostic["diagnostic_sha256"] = canonical_sha256(base_diagnostic)

        required_offsets = ((-2, 0), (2, 0), (0, -2), (0, 2))

        def semantic_diagnostic(group_id: str) -> dict:
            scores = [{"canonical_group_id": group_id, "color_share": 0.01}]
            return {
                "direct_sample": {"group_scores": scores},
                "bbox_sample": {"group_scores": scores},
                "projection_perturbations": [
                    {
                        "offset_pixels": [x, y],
                        "group_scores": scores,
                    }
                    for x, y in required_offsets
                ],
            }

        spatial_parts = []
        member_records = []
        for part_id, alternative_view, alternative_group in (
            ("P0001", "alt_a", "G02"),
            ("P0002", "alt_b", "G03"),
        ):
            diagnostic = copy.deepcopy(base_diagnostic)
            target_scores = [
                {
                    "canonical_group_id": "G01",
                    "matching_pixels": 210,
                    "color_share": 0.7,
                },
                {
                    "canonical_group_id": "G04",
                    "matching_pixels": 0,
                    "color_share": 0.0,
                },
            ]
            target_observation = {
                "reference_view_id": "front",
                "declared_visible_pixels": 300,
                "projected_part_pixels": 300,
                "canonical_group_id": "G01",
                "registration_label_stable": True,
                "perturbation_label_stable": True,
                "group_scores": target_scores,
                "color_margin": 0.6,
                "bbox_canonical_group_id": "G01",
                "bbox_group_scores": [
                    {
                        "canonical_group_id": "G01",
                        "matching_pixels": 285,
                        "color_share": 0.95,
                    }
                ],
                "bbox_color_margin": 0.9,
                "projection_perturbations": [
                    {
                        "offset_pixels": [x, y],
                        "sampled_reference_pixels": 300,
                        "canonical_group_id": "G01",
                        "diagnostic_canonical_group_id": "G01",
                        "best_color_share": 0.7,
                        "color_margin": 0.5,
                    }
                    for x, y in required_offsets
                ],
                "dark_foreground_diagnostic": diagnostic,
            }
            spatial_parts.append(
                {
                    "part_id": part_id,
                    "observations": [
                        target_observation,
                        {
                            "reference_view_id": alternative_view,
                            "canonical_palette_diagnostic": semantic_diagnostic(
                                alternative_group
                            ),
                        },
                    ],
                    "semantic_votes": [
                        {
                            "view_id": alternative_view,
                            "status": "matched",
                            "canonical_group_id": alternative_group,
                            "alignment_trusted": True,
                            "unique_canonical_join": True,
                            "pixel_gate_accepted": True,
                            "effective_confidence": 0.9,
                        }
                    ],
                }
            )
            sample_shares = {
                "direct": 0.01,
                "bbox": 0.01,
                "offset_-2_0": 0.01,
                "offset_2_0": 0.01,
                "offset_0_-2": 0.01,
                "offset_0_2": 0.01,
            }
            member_records.append(
                {
                    "part_id": part_id,
                    "projected_part_pixels": 300,
                    "direct_target_share": 0.7,
                    "direct_target_margin": 0.6,
                    "direct_target_matching_pixels": 210,
                    "bbox_target_share": 0.95,
                    "bbox_target_margin": 0.9,
                    "perturbations": [
                        {
                            "offset_pixels": [x, y],
                            "sampled_reference_pixels": 300,
                            "target_share": 0.7,
                            "target_margin": 0.5,
                        }
                        for x, y in sorted(required_offsets)
                    ],
                    "alignment": {
                        "score": 0.82,
                        "projection_score": 0.9,
                        "projection_iou": 0.9,
                        "ecc_correlation": 0.95,
                        "ecc_status": "success",
                    },
                    "dark_diagnostic_sha256": diagnostic["diagnostic_sha256"],
                    "dark_signal_share": diagnostic["dark_signal_share"],
                    "dark_signal_purity": diagnostic["dark_signal_purity"],
                    "core_dark_signal_share": diagnostic["core_dark_signal_share"],
                    "adaptive_edge_density": diagnostic["adaptive_edge_density"],
                    "dark_signal_null_margin": diagnostic["dark_signal_null_margin"],
                    "semantic_alternative_disproofs": [
                        {
                            "view_id": alternative_view,
                            "canonical_group_id": alternative_group,
                            "status": "matched",
                            "effective_confidence": 0.9,
                            "sample_shares": sample_shares,
                        }
                    ],
                }
            )
        for member in member_records:
            member["evidence_contract"] = "dark_foreground_diagnostic"
            member["evidence_sha256"] = canonical_sha256(member)
        bundle["spatial_report"] = {
            "schema_version": "qwen-spatial-mapping-audit/v1",
            "policy": {"minimum_semantic_confidence": 0.85},
            "reference_evidence": reference_records,
            "view_alignments": [alignment],
            "parts": spatial_parts,
        }
        bundle["spatial_gate_audit"] = {
            "schema_version": "qwen-spatial-mapping-gate/v1",
            "decisions": [],
        }
        bundle["mapping_consensus"] = {
            "schema_version": "qwen-mapping-consensus-audit/v1",
            "decisions": [
                {
                    "part_id": part_id,
                    "main_group_id": "G01",
                    "main_status": "review",
                    "main_confidence": 0.8,
                    "output_group_id": None,
                    "output_status": "unknown",
                }
                for part_id in ("P0001", "P0002")
            ],
        }
        support = {
            "reference_view_id": "front",
            "local_group_id": "G05",
            "reference_sha256": "1" * 64,
            "content_cluster_id": "CONTENT_FRONT",
            "pose_cluster_id": "right",
            "recall": 0.4,
            "deficit_sources": ["group_recall"],
        }
        bundle["audit"]["group_diagnostics"] = [
            {
                "canonical_group_id": "G01",
                "repairable": False,
                "single_view_spatial_repairable": True,
                "supporting_views": [support],
            }
        ]
        geometry_payload = {
            "point_count": 100,
            "face_count": 80,
            "sorted_bbox_extents": [1.0, 2.0, 3.0],
        }
        geometry_sha = canonical_sha256(geometry_payload)
        stable_source = {
            key: value
            for key, value in source_properties.items()
            if key != "shader_path"
        }
        source_sha = canonical_sha256(stable_source)
        cohort_part_ids = ["P0001", "P0002"]
        cohort_id = canonical_sha256(
            {
                "canonical_group_id": "G01",
                "reference_view_id": "front",
                "geometry_signature_sha256": geometry_sha,
                "source_visual_stable_properties_signature_sha256": source_sha,
                "cohort_part_ids": cohort_part_ids,
            }
        )
        cohort_record = {
            "cohort_id": cohort_id,
            "canonical_group_id": "G01",
            "reference_view_id": "front",
            "geometry_signature": {
                **geometry_payload,
                "signature_sha256": geometry_sha,
            },
            "source_visual_stable_properties_signature_sha256": source_sha,
            "registry_part_count": 40,
            "cohort_size": 2,
            "registry_fraction": 0.05,
            "cohort_part_ids": cohort_part_ids,
            "required_render_share": 0.1,
            "observed_render_share": 0.04,
            "render_foreground_pixels": 10000,
            "budget_pixels": 600,
            "minimum_contribution_pixels": 450,
            "maximum_contribution_pixels": 810,
            "total_projected_part_pixels": 600,
            "total_direct_target_matching_pixels": 420,
            "selected": True,
            "reason_codes": [],
            "members": member_records,
        }
        bundle["audit"]["repeated_geometry_dark_cohorts"] = [cohort_record]
        bundle["audit"]["dark_residual_budgets"] = []
        dark_fields = {
            "dark_residual_support",
            "dark_foreground_diagnostic",
            "budget_pixels",
            "budget_limit_pixels",
            "existing_contribution_pixels",
            "estimated_contribution_pixels",
            "selected_contribution_pixels",
            "cumulative_contribution_pixels",
        }
        first_change = bundle["audit"]["changes"][0]
        for field in dark_fields:
            first_change.pop(field, None)
        first_change.update(
            {
                "repeated_geometry_dark_cohort_id": cohort_id,
                "cohort_part_ids": cohort_part_ids,
            }
        )
        second_change = copy.deepcopy(first_change)
        second_change["part_id"] = "P0002"
        bundle["audit"]["changes"] = [first_change, second_change]
        bundle["audit"]["localization_lanes"] = [
            {
                "part_id": part_id,
                "canonical_group_id": "G01",
                "lane": "repeated_geometry_dark_residual_exact_projection",
            }
            for part_id in cohort_part_ids
        ]
        bundle["audit"]["summary"]["changed_count"] = 2

        for part_id in cohort_part_ids:
            assignment = next(
                item for item in output_assignments if item["part_id"] == part_id
            )
            member = next(item for item in member_records if item["part_id"] == part_id)
            assignment.update(
                {
                    "material_id": "mdl:test#green",
                    "semantic": "matte black painted arm",
                    "confidence": 0.0,
                    "evidence_views": [],
                    "status": "policy_fallback",
                    "provenance": {
                        "tier": "qa_repair_candidate",
                        "reason_codes": [
                            "QA_MISSING_CANONICAL_GROUP_SINGLE_VIEW",
                            "QA_REPEATED_GEOMETRY_COHORT_EXACT_PROJECTION",
                            "QA_CONFIRMED_WHITELIST_MATERIAL",
                        ],
                        "output_confidence_basis": (POLICY_FALLBACK_CONFIDENCE_BASIS),
                        "sources": [],
                        "canonical_group_id": "G01",
                        "baseline_material_id": "mdl:test#material",
                        "baseline_tier": "neutral_default",
                        "supporting_view_ids": ["front"],
                        "supporting_content_cluster_ids": ["CONTENT_FRONT"],
                        "supporting_pose_cluster_ids": ["right"],
                        "repeated_geometry_dark_residual": {
                            "lane": (
                                "repeated_geometry_dark_residual_exact_projection"
                            ),
                            "cohort_id": cohort_id,
                            "canonical_group_id": "G01",
                            "reference_view_id": "front",
                            "geometry_signature_sha256": geometry_sha,
                            "source_visual_stable_properties_signature_sha256": (
                                source_sha
                            ),
                            "cohort_part_ids": cohort_part_ids,
                            "budget_pixels": 600,
                            "minimum_contribution_pixels": 450,
                            "maximum_contribution_pixels": 810,
                            "total_projected_part_pixels": 600,
                            "total_direct_target_matching_pixels": 420,
                            "member_evidence_contract": member[
                                "evidence_contract"
                            ],
                            "member_evidence_sha256": member["evidence_sha256"],
                            "dark_diagnostic_sha256": member["dark_diagnostic_sha256"],
                        },
                    },
                }
            )
        bundle["plan"]["provenance"][QUALITY_REPAIR_PROVENANCE_FIELD][
            "changed_part_ids"
        ] = cohort_part_ids
        self._refresh_quality_repair_hashes(bundle)
        return bundle

    def _anchored_quality_repair_bundle(self) -> dict[str, dict]:
        bundle = self._quality_repair_bundle()
        group = bundle["palette_fusion"]["canonical_palette"]["groups"][0]
        group.update({"distinct_view_count": 3, "singleton": False})

        bundle["registry"]["part_count"] = 3
        bundle["registry"]["parts"].append(
            {"part_id": "P0003", "prim_path": "/Asset/C/Mesh"}
        )
        third_baseline = copy.deepcopy(bundle["baseline_plan"]["assignments"][1])
        third_baseline["part_id"] = "P0003"
        bundle["baseline_plan"]["assignments"].append(third_baseline)
        bundle["baseline_plan"]["provenance"]["registry_sha256"] = canonical_sha256(
            bundle["registry"]
        )
        bundle["baseline_policy_audit"]["summary"].update(
            {
                "registry_part_count": 3,
                "output_assignment_count": 3,
                "policy_fallback_count": 3,
            }
        )
        bundle["baseline_policy_audit"]["input_hashes"] = copy.deepcopy(
            bundle["baseline_plan"]["provenance"]
        )
        bundle["baseline_policy_audit"]["output_plan_sha256"] = canonical_sha256(
            bundle["baseline_plan"]
        )

        second = bundle["plan"]["assignments"][1]
        second.update(
            {
                "material_id": "mdl:test#green",
                "semantic": "green painted panel",
                "provenance": {
                    "tier": "qa_repair_candidate",
                    "reason_codes": list(QUALITY_REPAIR_REASON_CODES),
                    "output_confidence_basis": POLICY_FALLBACK_CONFIDENCE_BASIS,
                    "sources": [],
                    "canonical_group_id": "G01",
                    "baseline_material_id": "mdl:test#material",
                    "baseline_tier": "neutral_default",
                    "supporting_view_ids": ["ref_b", "ref_c"],
                    "supporting_content_cluster_ids": [
                        "CONTENT_02",
                        "CONTENT_03",
                    ],
                    "supporting_pose_cluster_ids": ["side", "top"],
                },
            }
        )
        third = copy.deepcopy(third_baseline)
        third.update(
            {
                "material_id": "mdl:test#green",
                "semantic": "green painted panel",
                "provenance": {
                    "tier": "qa_repair_candidate",
                    "reason_codes": list(QUALITY_REPAIR_REASON_CODES),
                    "output_confidence_basis": POLICY_FALLBACK_CONFIDENCE_BASIS,
                    "sources": [],
                    "canonical_group_id": "G01",
                    "baseline_material_id": "mdl:test#material",
                    "baseline_tier": "neutral_default",
                    "supporting_view_ids": ["ref_a"],
                    "supporting_content_cluster_ids": ["CONTENT_01"],
                    "supporting_pose_cluster_ids": ["front"],
                },
            }
        )
        bundle["plan"]["assignments"].append(third)
        bundle["audit"]["summary"]["changed_count"] = 3
        bundle["audit"]["changes"].append(
            {
                "part_id": "P0002",
                "canonical_group_id": "G01",
                "material_id": "mdl:test#green",
                "supporting_view_ids": ["ref_b", "ref_c"],
                "supporting_content_cluster_ids": ["CONTENT_02", "CONTENT_03"],
                "supporting_pose_cluster_ids": ["side", "top"],
                "old_material_id": "mdl:test#material",
                "new_material_id": "mdl:test#green",
            }
        )
        bundle["audit"]["changes"].append(
            {
                "part_id": "P0003",
                "canonical_group_id": "G01",
                "material_id": "mdl:test#green",
                "supporting_view_ids": ["ref_a"],
                "supporting_content_cluster_ids": ["CONTENT_01"],
                "supporting_pose_cluster_ids": ["front"],
                "anchor_part_ids": ["P0001", "P0002"],
                "anchor_supporting_view_ids": ["ref_a", "ref_b", "ref_c"],
                "old_material_id": "mdl:test#material",
                "new_material_id": "mdl:test#green",
            }
        )
        bundle["audit"]["localization_lanes"].append(
            {
                "part_id": "P0002",
                "canonical_group_id": "G01",
                "lane": "bounded_spatial_multiview",
            }
        )
        bundle["audit"]["localization_lanes"].append(
            {
                "part_id": "P0003",
                "canonical_group_id": "G01",
                "lane": "exact_spatial_single_view_with_multiview_anchor",
            }
        )
        bundle["audit"]["group_diagnostics"] = [
            {
                "canonical_group_id": "G01",
                "repairable": True,
                "single_view_spatial_repairable": False,
                "supporting_views": [
                    {"reference_view_id": "ref_a", "local_group_id": "L01"},
                    {"reference_view_id": "ref_b", "local_group_id": "L09"},
                    {"reference_view_id": "ref_c", "local_group_id": "L11"},
                ],
            }
        ]
        input_hashes = {
            "baseline_plan_sha256": canonical_sha256(bundle["baseline_plan"]),
            "baseline_policy_audit_sha256": canonical_sha256(
                bundle["baseline_policy_audit"]
            ),
            "quality_report_sha256": canonical_sha256(bundle["quality_report"]),
            "palette_fusion_sha256": canonical_sha256(bundle["palette_fusion"]),
            "spatial_report_sha256": canonical_sha256(bundle["spatial_report"]),
            "spatial_gate_audit_sha256": canonical_sha256(bundle["spatial_gate_audit"]),
            "mapping_consensus_sha256": canonical_sha256(bundle["mapping_consensus"]),
            "geometry_risk_sha256": canonical_sha256(bundle["geometry_risk"]),
            "group_materials_sha256": canonical_sha256(bundle["group_materials"]),
            "mvinverse_pbr_evidence_sha256": canonical_sha256(
                bundle["mvinverse_pbr_evidence"]
            ),
            "registry_sha256": canonical_sha256(bundle["registry"]),
            "whitelist_sha256": canonical_sha256(bundle["whitelist"]),
        }
        bundle["audit"]["input_hashes"] = input_hashes
        bundle["plan"]["provenance"] = copy.deepcopy(
            bundle["baseline_plan"]["provenance"]
        )
        bundle["plan"]["provenance"][QUALITY_REPAIR_PROVENANCE_FIELD] = {
            "mode": QUALITY_REPAIR_PLAN_MODE,
            "input_hashes": input_hashes,
            "changed_part_ids": ["P0001", "P0002", "P0003"],
        }
        bundle["audit"]["output_plan_sha256"] = canonical_sha256(bundle["plan"])
        return bundle

    def test_hash_bound_quality_repair_bundle_is_accepted(self) -> None:
        self.assertEqual(
            _validate_quality_repair_bundle(**self._quality_repair_bundle()), 1
        )

    def test_hash_bound_quality_repair_safe_noop_is_accepted(self) -> None:
        bundle = self._quality_repair_bundle()
        bundle["plan"] = copy.deepcopy(bundle["baseline_plan"])
        bundle["audit"]["summary"].update(
            {
                "status": "SAFE_NOOP",
                "changed_count": 0,
                "no_op": True,
            }
        )
        bundle["audit"]["reason_codes"] = ["NO_ELIGIBLE_REPAIR_PARTS"]
        bundle["audit"]["changes"] = []
        bundle["audit"]["localization_lanes"] = []
        bundle["audit"]["output_plan_sha256"] = canonical_sha256(bundle["plan"])

        self.assertEqual(_validate_quality_repair_bundle(**bundle), 0)

    def test_quality_repair_safe_noop_cannot_change_the_baseline(self) -> None:
        bundle = self._quality_repair_bundle()
        bundle["plan"] = copy.deepcopy(bundle["baseline_plan"])
        bundle["plan"]["assignments"][0]["material_id"] = "unauthorized"
        bundle["audit"]["summary"].update(
            {
                "status": "SAFE_NOOP",
                "changed_count": 0,
                "no_op": True,
            }
        )
        bundle["audit"]["reason_codes"] = ["NO_ELIGIBLE_REPAIR_PARTS"]
        bundle["audit"]["changes"] = []
        bundle["audit"]["localization_lanes"] = []
        bundle["audit"]["output_plan_sha256"] = canonical_sha256(bundle["plan"])

        with self.assertRaisesRegex(RuntimeError, "preserve the sealed baseline"):
            _validate_quality_repair_bundle(**bundle)

    def test_dark_foreground_quality_repair_bundle_is_accepted(self) -> None:
        self.assertEqual(
            _validate_quality_repair_bundle(**self._dark_quality_repair_bundle()),
            1,
        )

    def test_repeated_geometry_dark_cohort_is_atomically_accepted(self) -> None:
        bundle = self._repeated_geometry_dark_quality_repair_bundle()
        self.assertEqual(_validate_quality_repair_bundle(**bundle), 2)

    def test_repeated_dark_strict_projection_overrides_nonlocal_semantic_noise(
        self,
    ) -> None:
        bundle = self._repeated_geometry_dark_quality_repair_bundle()
        members = bundle["audit"]["repeated_geometry_dark_cohorts"][0]["members"]
        assignments = {
            item["part_id"]: item for item in bundle["plan"]["assignments"]
        }
        for member in members:
            for field in (
                "dark_diagnostic_sha256",
                "dark_signal_share",
                "dark_signal_purity",
                "core_dark_signal_share",
                "adaptive_edge_density",
                "dark_signal_null_margin",
            ):
                member.pop(field)
            member["alignment"].pop("projection_score")
            member["semantic_alternative_disproofs"] = []
            member["evidence_contract"] = "strict_reference_space_projection"
            member.pop("evidence_sha256")
            member["evidence_sha256"] = canonical_sha256(member)
            provenance = assignments[member["part_id"]]["provenance"][
                "repeated_geometry_dark_residual"
            ]
            provenance.pop("dark_diagnostic_sha256")
            provenance["member_evidence_contract"] = member["evidence_contract"]
            provenance["member_evidence_sha256"] = member["evidence_sha256"]

        first_part = bundle["spatial_report"]["parts"][0]
        first_part["semantic_votes"].append(
            {
                "view_id": "unregistered_nonlocal_view",
                "status": "matched",
                "canonical_group_id": "G04",
                "alignment_trusted": True,
                "unique_canonical_join": True,
                "pixel_gate_accepted": True,
                "effective_confidence": 0.99,
            }
        )
        self._refresh_quality_repair_hashes(bundle)
        self.assertEqual(_validate_quality_repair_bundle(**bundle), 2)

    def test_repeated_geometry_dark_cohort_rejects_member_tampering(self) -> None:
        bundle = self._repeated_geometry_dark_quality_repair_bundle()
        bundle["audit"]["repeated_geometry_dark_cohorts"][0]["members"][0][
            "dark_signal_share"
        ] = 0.51
        with self.assertRaisesRegex(RuntimeError, "member audit"):
            _validate_quality_repair_bundle(**bundle)

    def test_repeated_geometry_dark_cohort_fails_closed_on_source_divergence(
        self,
    ) -> None:
        bundle = self._repeated_geometry_dark_quality_repair_bundle()
        bundle["registry"]["parts"][1]["existing_visual_material_properties"][
            "roughness"
        ] = 0.6
        self._refresh_quality_repair_hashes(bundle)
        with self.assertRaisesRegex(RuntimeError, "complete safe geometry cohort"):
            _validate_quality_repair_bundle(**bundle)

    def test_dark_foreground_repair_rejects_signed_diagnostic_tampering(
        self,
    ) -> None:
        bundle = self._dark_quality_repair_bundle()
        diagnostic = bundle["spatial_report"]["parts"][0]["observations"][0][
            "dark_foreground_diagnostic"
        ]
        diagnostic["dark_signal_pixels"] = 499
        input_hashes = bundle["audit"]["input_hashes"]
        input_hashes["spatial_report_sha256"] = canonical_sha256(
            bundle["spatial_report"]
        )
        bundle["plan"]["provenance"][QUALITY_REPAIR_PROVENANCE_FIELD][
            "input_hashes"
        ] = input_hashes
        bundle["audit"]["output_plan_sha256"] = canonical_sha256(bundle["plan"])
        with self.assertRaisesRegex(RuntimeError, "diagnostic hash"):
            _validate_quality_repair_bundle(**bundle)

    def test_dark_foreground_repair_rejects_budget_tampering(self) -> None:
        bundle = self._dark_quality_repair_bundle()
        bundle["audit"]["dark_residual_budgets"][0]["candidates"][0][
            "cumulative_contribution_pixels"
        ] = 501
        with self.assertRaisesRegex(RuntimeError, "budget"):
            _validate_quality_repair_bundle(**bundle)

    def test_dark_foreground_repair_rejects_matched_semantic_conflict(
        self,
    ) -> None:
        bundle = self._dark_quality_repair_bundle()
        bundle["spatial_gate_audit"]["decisions"][0]["output_status"] = "matched"
        input_hashes = bundle["audit"]["input_hashes"]
        input_hashes["spatial_gate_audit_sha256"] = canonical_sha256(
            bundle["spatial_gate_audit"]
        )
        bundle["plan"]["provenance"][QUALITY_REPAIR_PROVENANCE_FIELD][
            "input_hashes"
        ] = input_hashes
        bundle["audit"]["output_plan_sha256"] = canonical_sha256(bundle["plan"])
        with self.assertRaisesRegex(RuntimeError, "matched semantic group"):
            _validate_quality_repair_bundle(**bundle)

    def test_semantic_anchor_single_view_repair_bundle_is_accepted(self) -> None:
        bundle = self._quality_repair_bundle()
        assignment = bundle["plan"]["assignments"][0]
        change = bundle["audit"]["changes"][0]
        assignment["provenance"].update(
            {
                "supporting_view_ids": ["ref_a"],
                "supporting_content_cluster_ids": ["CONTENT_01"],
                "supporting_pose_cluster_ids": ["front"],
            }
        )
        change.update(
            {
                "supporting_view_ids": ["ref_a"],
                "supporting_content_cluster_ids": ["CONTENT_01"],
                "supporting_pose_cluster_ids": ["front"],
                "semantic_conflict_override_view_ids": ["ref_a"],
            }
        )
        bundle["audit"]["localization_lanes"][0]["lane"] = (
            "exact_spatial_single_qa_view_with_semantic_anchor"
        )
        bundle["audit"]["output_plan_sha256"] = canonical_sha256(bundle["plan"])

        self.assertEqual(_validate_quality_repair_bundle(**bundle), 1)

    def test_dominant_review_override_is_independently_accepted(self) -> None:
        bundle = self._dominant_review_override_quality_repair_bundle()
        self.assertEqual(_validate_quality_repair_bundle(**bundle), 1)

    def test_dominant_review_override_rejects_pixel_disproof_tampering(
        self,
    ) -> None:
        bundle = self._dominant_review_override_quality_repair_bundle()
        diagnostic = bundle["spatial_report"]["parts"][0]["observations"][0][
            "canonical_palette_diagnostic"
        ]
        diagnostic["direct_sample"]["group_scores"][1]["color_share"] = 0.2
        self._refresh_quality_repair_hashes(bundle)
        with self.assertRaisesRegex(RuntimeError, "pixel-disprove"):
            _validate_quality_repair_bundle(**bundle)

    def test_dominant_review_override_fails_closed_without_exact_anchor(
        self,
    ) -> None:
        bundle = self._dominant_review_override_quality_repair_bundle()
        bundle["spatial_report"]["parts"][0]["semantic_votes"][1][
            "effective_confidence"
        ] = 0.79
        self._refresh_quality_repair_hashes(bundle)
        with self.assertRaisesRegex(RuntimeError, "independent target anchor"):
            _validate_quality_repair_bundle(**bundle)

    def test_dominant_review_override_rejects_unstable_registration(self) -> None:
        bundle = self._dominant_review_override_quality_repair_bundle()
        bundle["spatial_report"]["parts"][0]["observations"][0][
            "registration_label_stable"
        ] = False
        self._refresh_quality_repair_hashes(bundle)
        with self.assertRaisesRegex(RuntimeError, "direct projection"):
            _validate_quality_repair_bundle(**bundle)

    def test_quality_repair_can_replace_source_visual_preserve_baseline(self) -> None:
        bundle = self._quality_repair_bundle()
        baseline = bundle["baseline_plan"]["assignments"][0]
        baseline.update(
            {
                "apply_action": "source_visual_preserve",
                "source_visual_material_prim_path": "/Asset/Looks/SourceGray",
                "source_visual_material_binding_sha256": "b" * 64,
            }
        )
        baseline["provenance"]["tier"] = "source_visual_preserve"
        bundle["baseline_policy_audit"]["output_plan_sha256"] = canonical_sha256(
            bundle["baseline_plan"]
        )
        input_hashes = bundle["audit"]["input_hashes"]
        input_hashes["baseline_plan_sha256"] = canonical_sha256(bundle["baseline_plan"])
        input_hashes["baseline_policy_audit_sha256"] = canonical_sha256(
            bundle["baseline_policy_audit"]
        )
        bundle["plan"]["provenance"][QUALITY_REPAIR_PROVENANCE_FIELD][
            "input_hashes"
        ] = copy.deepcopy(input_hashes)
        bundle["plan"]["assignments"][0]["provenance"]["baseline_tier"] = (
            "source_visual_preserve"
        )
        bundle["audit"]["input_hashes"] = copy.deepcopy(input_hashes)
        bundle["audit"]["output_plan_sha256"] = canonical_sha256(bundle["plan"])

        self.assertEqual(_validate_quality_repair_bundle(**bundle), 1)
        repaired = bundle["plan"]["assignments"][0]
        self.assertNotIn("apply_action", repaired)
        self.assertNotIn("source_visual_material_prim_path", repaired)
        self.assertNotIn("source_visual_material_binding_sha256", repaired)

    def test_hash_bound_anchored_single_view_repair_is_accepted(self) -> None:
        self.assertEqual(
            _validate_quality_repair_bundle(**self._anchored_quality_repair_bundle()),
            3,
        )

    def test_anchored_single_view_rejects_a_tampered_anchor_relation(self) -> None:
        bundle = self._anchored_quality_repair_bundle()
        bundle["audit"]["changes"][2]["anchor_part_ids"] = ["P0003"]
        with self.assertRaisesRegex(RuntimeError, "complete existing multiview anchor"):
            _validate_quality_repair_bundle(**bundle)

    def test_repair_bundle_does_not_use_local_residual_as_anchor(self) -> None:
        bundle = self._anchored_quality_repair_bundle()
        bundle["audit"]["localization_lanes"][0]["lane"] = (
            "dominant_chromatic_residual_exact_single_view"
        )
        bundle["audit"]["changes"][0].update(
            {
                "supporting_view_ids": ["ref_a"],
                "supporting_content_cluster_ids": ["CONTENT_01"],
                "supporting_pose_cluster_ids": ["front"],
            }
        )
        bundle["plan"]["assignments"][0]["provenance"].update(
            {
                "supporting_view_ids": ["ref_a"],
                "supporting_content_cluster_ids": ["CONTENT_01"],
                "supporting_pose_cluster_ids": ["front"],
            }
        )
        bundle["audit"]["output_plan_sha256"] = canonical_sha256(bundle["plan"])

        with self.assertRaisesRegex(RuntimeError, "complete existing multiview anchor"):
            _validate_quality_repair_bundle(**bundle)

    def test_quality_repair_cannot_modify_an_unlisted_part(self) -> None:
        bundle = self._quality_repair_bundle()
        bundle["plan"]["assignments"][1]["semantic"] = "unauthorized"
        bundle["audit"]["output_plan_sha256"] = canonical_sha256(bundle["plan"])
        with self.assertRaisesRegex(RuntimeError, "unauthorized part"):
            _validate_quality_repair_bundle(**bundle)

    def test_quality_repair_rejects_audit_material_mismatch(self) -> None:
        bundle = self._quality_repair_bundle()
        bundle["audit"]["changes"][0]["new_material_id"] = "mdl:test#material"
        with self.assertRaisesRegex(RuntimeError, "audit material delta"):
            _validate_quality_repair_bundle(**bundle)

    def test_quality_repair_rejects_injected_face_subsets(self) -> None:
        bundle = self._quality_repair_bundle()
        bundle["plan"]["assignments"][0]["face_subsets"] = [
            {
                "subset_name": "Injected",
                "material_id": "mdl:test#green",
                "face_indices": [0],
            }
        ]
        bundle["audit"]["output_plan_sha256"] = canonical_sha256(bundle["plan"])
        with self.assertRaisesRegex(RuntimeError, "assignment delta is unsafe"):
            _validate_quality_repair_bundle(**bundle)

    def test_quality_repair_rejects_injected_parameters(self) -> None:
        bundle = self._quality_repair_bundle()
        bundle["plan"]["assignments"][0]["parameters"] = {"paint_roughness": 0.1}
        bundle["audit"]["output_plan_sha256"] = canonical_sha256(bundle["plan"])
        with self.assertRaisesRegex(RuntimeError, "assignment delta is unsafe"):
            _validate_quality_repair_bundle(**bundle)

    def test_quality_repair_accepts_only_exact_mvinverse_parameters(self) -> None:
        bundle = self._parameterized_quality_repair_bundle()
        self.assertEqual(_validate_quality_repair_bundle(**bundle), 1)

        bundle["plan"]["assignments"][0]["parameters"]["paint_roughness"] = 0.2
        bundle["audit"]["output_plan_sha256"] = canonical_sha256(bundle["plan"])
        with self.assertRaisesRegex(RuntimeError, "assignment delta is unsafe"):
            _validate_quality_repair_bundle(**bundle)

    def test_quality_repair_rejects_an_auto_baseline_assignment(self) -> None:
        bundle = self._quality_repair_bundle()
        baseline = bundle["baseline_plan"]["assignments"][0]
        baseline.update(
            {
                "status": "auto",
                "confidence": 0.95,
                "evidence_views": ["ref_a"],
                "provenance": {
                    "tier": "gate_auto",
                    "reason_codes": ["CONFIDENCE_GATE_AUTO"],
                    "output_confidence_basis": "confidence gate",
                    "sources": [],
                },
            }
        )
        bundle["baseline_policy_audit"]["output_plan_sha256"] = canonical_sha256(
            bundle["baseline_plan"]
        )
        input_hashes = {
            "baseline_plan_sha256": canonical_sha256(bundle["baseline_plan"]),
            "baseline_policy_audit_sha256": canonical_sha256(
                bundle["baseline_policy_audit"]
            ),
            "quality_report_sha256": canonical_sha256(bundle["quality_report"]),
            "palette_fusion_sha256": canonical_sha256(bundle["palette_fusion"]),
            "spatial_report_sha256": canonical_sha256(bundle["spatial_report"]),
            "spatial_gate_audit_sha256": canonical_sha256(bundle["spatial_gate_audit"]),
            "mapping_consensus_sha256": canonical_sha256(bundle["mapping_consensus"]),
            "geometry_risk_sha256": canonical_sha256(bundle["geometry_risk"]),
            "group_materials_sha256": canonical_sha256(bundle["group_materials"]),
            "mvinverse_pbr_evidence_sha256": canonical_sha256(
                bundle["mvinverse_pbr_evidence"]
            ),
            "registry_sha256": canonical_sha256(bundle["registry"]),
            "whitelist_sha256": canonical_sha256(bundle["whitelist"]),
        }
        bundle["audit"]["input_hashes"] = input_hashes
        bundle["plan"]["provenance"][QUALITY_REPAIR_PROVENANCE_FIELD][
            "input_hashes"
        ] = input_hashes
        bundle["audit"]["output_plan_sha256"] = canonical_sha256(bundle["plan"])
        with self.assertRaisesRegex(RuntimeError, "not neutral fallback"):
            _validate_quality_repair_bundle(**bundle)

    def _dominant_quality_report(self, *, repaired: bool) -> dict:
        thresholds = {
            "minimum_evidence_group_recall": 0.2,
            "strong_alignment_score": 0.55,
            "minimum_dominant_reference_share": 0.25,
            "minimum_dominant_share_margin": 0.10,
            "minimum_dominant_mass_recall": 0.80,
            "minimum_dominant_absolute_deficit": 0.08,
            "minimum_dominant_silhouette_iou": 0.75,
        }

        def view(
            view_id: str,
            render_view_id: str,
            image_sha256: str,
            local_group_id: str,
            *,
            failed: bool,
        ) -> dict:
            observed = 0.72 if not failed else 0.5
            mass_recall = observed / 0.8
            return {
                "reference_view_id": view_id,
                "render_view_id": render_view_id,
                "status": "FAIL" if failed else "PASS",
                "reasons": (["trusted_dominant_family_mass_deficit"] if failed else []),
                "mapping": {
                    "selected_render_view_id": render_view_id,
                    "reasons": [],
                },
                "reference": {"image_sha256": image_sha256},
                "alignment": {"score": 0.9, "silhouette_iou": 0.9},
                "material_color": {
                    "trusted_evidence_group_recall": {
                        "groups": [{"group_id": local_group_id, "recall": 1.0}]
                    },
                    "reference_distribution": {
                        "category_distribution": {
                            "green": 0.8,
                            "achromatic_mid": 0.2,
                        }
                    },
                    "render_distribution": {
                        "category_distribution": {
                            "green": observed,
                            "achromatic_mid": 1.0 - observed,
                        }
                    },
                    "trusted_evidence_dominant_mass": {
                        "status": "FAIL" if failed else "PASS",
                        "eligible_family_count": 1,
                        "failed_family_count": 1 if failed else 0,
                        "families": [
                            {
                                "family_key": "green",
                                "local_group_ids": [local_group_id],
                                "base_colors": ["green"],
                                "render_color_bins": ["green"],
                                "reference_share": 0.8,
                                "runner_up_reference_share": 0.0,
                                "reference_share_margin": 0.8,
                                "observed_render_share": observed,
                                "deficit_share": max(0.0, 0.8 - observed),
                                "mass_recall": mass_recall,
                                "eligible": True,
                                "status": "FAIL" if failed else "PASS",
                                "reason_codes": (
                                    ["DOMINANT_FAMILY_MASS_DEFICIT"] if failed else []
                                ),
                            }
                        ],
                    },
                },
            }

        first_failed = not repaired
        return {
            "inputs": {
                "reference_manifest_sha256": "a" * 64,
                "mapping_mode": "deterministic_global_registration",
                "selected_view_mapping": {
                    "ref_a": "front",
                    "ref_b": "side",
                },
            },
            "thresholds": thresholds,
            "aggregate": {
                "status": "FAIL" if first_failed else "PASS",
                "comparable_view_count": 2,
                "reasons": (
                    ["single_strong_view_confirms_dominant_family_mass_deficit"]
                    if first_failed
                    else []
                ),
            },
            "views": [
                view(
                    "ref_a",
                    "front",
                    "b" * 64,
                    "L01",
                    failed=first_failed,
                ),
                view(
                    "ref_b",
                    "side",
                    "c" * 64,
                    "L09",
                    failed=False,
                ),
            ],
        }

    def test_dominant_mass_validator_recomputes_numeric_decisions(self) -> None:
        report = self._dominant_quality_report(repaired=False)
        validated = _validate_quality_dominant_mass(report)
        self.assertTrue(validated["enabled"])
        self.assertEqual(validated["failed_view_ids"], ["ref_a"])

        report["views"][0]["material_color"]["trusted_evidence_dominant_mass"][
            "families"
        ][0]["mass_recall"] = 0.7
        with self.assertRaisesRegex(RuntimeError, "numeric evidence"):
            _validate_quality_dominant_mass(report)

    def test_quality_repair_outcome_recovers_a_dominant_only_deficit(
        self,
    ) -> None:
        baseline = self._dominant_quality_report(repaired=False)
        repaired = self._dominant_quality_report(repaired=True)
        audit = {
            "changes": [
                {
                    "part_id": "P0001",
                    "canonical_group_id": "G01",
                    "supporting_view_ids": ["ref_a"],
                }
            ],
            "localization_lanes": [
                {
                    "part_id": "P0001",
                    "canonical_group_id": "G01",
                    "lane": "exact_spatial_single_qa_view",
                }
            ],
            "group_diagnostics": [
                {
                    "canonical_group_id": "G01",
                    "repairable": False,
                    "single_view_spatial_repairable": True,
                    "supporting_views": [
                        {
                            "reference_view_id": "ref_a",
                            "local_group_id": "L01",
                            "recall": 0.625,
                            "deficit_sources": ["dominant_mass"],
                            "dominant_mass_family_key": "green",
                        }
                    ],
                }
            ],
        }

        _validate_quality_repair_outcome(
            baseline_quality=baseline,
            repaired_quality=repaired,
            repair_audit=audit,
        )

        repaired["views"][0]["material_color"]["trusted_evidence_dominant_mass"][
            "families"
        ][0]["observed_render_share"] = 0.5
        with self.assertRaisesRegex(RuntimeError, "numeric evidence"):
            _validate_quality_repair_outcome(
                baseline_quality=baseline,
                repaired_quality=repaired,
                repair_audit=audit,
            )

    def test_failed_repair_may_enter_immutable_mdl_tournament(self) -> None:
        baseline = self._dominant_quality_report(repaired=False)
        repaired = copy.deepcopy(baseline)

        _validate_quality_repair_outcome(
            baseline_quality=baseline,
            repaired_quality=repaired,
            repair_audit={},
            allow_pending_immutable_tournament=True,
        )

        with self.assertRaisesRegex(RuntimeError, "accepted outcome"):
            _validate_quality_repair_outcome(
                baseline_quality=baseline,
                repaired_quality=repaired,
                repair_audit={},
            )

        tampered = copy.deepcopy(repaired)
        tampered["views"][0]["reference"]["image_sha256"] = "d" * 64
        with self.assertRaisesRegex(RuntimeError, "changed references"):
            _validate_quality_repair_outcome(
                baseline_quality=baseline,
                repaired_quality=tampered,
                repair_audit={},
                allow_pending_immutable_tournament=True,
            )

    def test_dark_foreground_outcome_requires_black_family_recovery(
        self,
    ) -> None:
        bundle = self._dark_quality_repair_bundle()
        baseline = bundle["quality_report"]
        repaired = copy.deepcopy(baseline)
        repaired["aggregate"]["status"] = "PASS"
        repaired["views"][0]["status"] = "PASS"
        repaired_categories = repaired["views"][0]["material_color"][
            "render_distribution"
        ]["category_distribution"]
        repaired_categories.update(
            {
                "black": 0.109375,
                "achromatic_dark": 0.109375,
                "achromatic_mid": 0.78125,
            }
        )
        _validate_quality_repair_outcome(
            baseline_quality=baseline,
            repaired_quality=repaired,
            repair_audit=bundle["audit"],
        )

        repaired_categories.update(
            {
                "black": 0.0625,
                "achromatic_dark": 0.0625,
                "achromatic_mid": 0.875,
            }
        )
        with self.assertRaisesRegex(
            RuntimeError, "did not recover.*dark-foreground residual"
        ):
            _validate_quality_repair_outcome(
                baseline_quality=baseline,
                repaired_quality=repaired,
                repair_audit=bundle["audit"],
            )

    def test_quality_repair_outcome_recovers_a_local_dominant_residual(
        self,
    ) -> None:
        baseline = self._dominant_quality_report(repaired=True)
        repaired = self._dominant_quality_report(repaired=True)

        baseline_family = baseline["views"][0]["material_color"][
            "trusted_evidence_dominant_mass"
        ]["families"][0]
        baseline_family.update(
            {
                "observed_render_share": 0.5,
                "deficit_share": 0.3,
                "mass_recall": 0.625,
                "eligible": False,
                "status": "NOT_APPLICABLE",
                "reason_codes": ["SILHOUETTE_IOU_BELOW_DOMINANT_FLOOR"],
            }
        )
        baseline["views"][0]["material_color"]["render_distribution"][
            "category_distribution"
        ] = {"green": 0.5, "achromatic_mid": 0.5}
        baseline["views"][0]["material_color"]["trusted_evidence_dominant_mass"].update(
            {
                "status": "NOT_APPLICABLE",
                "eligible_family_count": 0,
                "failed_family_count": 0,
            }
        )
        baseline["views"][0]["alignment"]["silhouette_iou"] = 0.7
        baseline["views"][0].update(
            {
                "status": "FAIL",
                "reasons": ["trusted_palette_group_missing_from_render"],
            }
        )
        baseline["views"][1]["material_color"]["trusted_evidence_group_recall"][
            "groups"
        ][0]["recall"] = 0.0
        baseline["views"][1].update(
            {
                "status": "FAIL",
                "reasons": ["trusted_palette_group_missing_from_render"],
            }
        )
        baseline["aggregate"].update(
            {
                "status": "FAIL",
                "reasons": ["trusted_palette_group_missing_from_render"],
            }
        )

        repaired_family = repaired["views"][0]["material_color"][
            "trusted_evidence_dominant_mass"
        ]["families"][0]
        repaired_family.update(
            {
                "eligible": False,
                "status": "NOT_APPLICABLE",
                "reason_codes": ["SILHOUETTE_IOU_BELOW_DOMINANT_FLOOR"],
            }
        )
        repaired["views"][0]["material_color"]["trusted_evidence_dominant_mass"].update(
            {
                "status": "NOT_APPLICABLE",
                "eligible_family_count": 0,
                "failed_family_count": 0,
            }
        )
        repaired["views"][0]["alignment"]["silhouette_iou"] = 0.7

        audit = {
            "changes": [
                {
                    "part_id": "P0001",
                    "canonical_group_id": "G01",
                    "supporting_view_ids": ["ref_a"],
                }
            ],
            "localization_lanes": [
                {
                    "part_id": "P0001",
                    "canonical_group_id": "G01",
                    "lane": ("dominant_chromatic_residual_exact_single_view"),
                }
            ],
            "group_diagnostics": [
                {
                    "canonical_group_id": "G01",
                    "repairable": True,
                    "single_view_spatial_repairable": False,
                    "dominant_residual_repairable": True,
                    "dominant_residual_supporting_views": [
                        {
                            "reference_view_id": "ref_a",
                            "local_group_id": "L01",
                            "recall": 0.625,
                            "deficit_sources": ["dominant_mass_local_projection"],
                            "dominant_mass_family_key": "green",
                            "requires_strict_local_projection": True,
                            "reference_share": 0.8,
                            "observed_render_share": 0.5,
                            "deficit_share": 0.3,
                            "mass_recall": 0.625,
                        }
                    ],
                    "supporting_views": [
                        {
                            "reference_view_id": "ref_b",
                            "local_group_id": "L09",
                            "recall": 0.0,
                            "deficit_sources": ["group_recall"],
                        },
                    ],
                }
            ],
        }

        _validate_quality_repair_outcome(
            baseline_quality=baseline,
            repaired_quality=repaired,
            repair_audit=audit,
        )

        tampered = copy.deepcopy(audit)
        tampered["group_diagnostics"][0]["dominant_residual_supporting_views"][0][
            "observed_render_share"
        ] = 0.51
        with self.assertRaisesRegex(RuntimeError, "does not match baseline evidence"):
            _validate_quality_repair_outcome(
                baseline_quality=baseline,
                repaired_quality=repaired,
                repair_audit=tampered,
            )

        unrecovered = copy.deepcopy(repaired)
        unrecovered_family = unrecovered["views"][0]["material_color"][
            "trusted_evidence_dominant_mass"
        ]["families"][0]
        unrecovered_family.update(
            {
                "observed_render_share": 0.55,
                "deficit_share": 0.25,
                "mass_recall": 0.6875,
            }
        )
        unrecovered["views"][0]["material_color"]["render_distribution"][
            "category_distribution"
        ] = {"green": 0.55, "achromatic_mid": 0.45}
        with self.assertRaisesRegex(
            RuntimeError, "did not recover.*local dominant residual"
        ):
            _validate_quality_repair_outcome(
                baseline_quality=baseline,
                repaired_quality=unrecovered,
                repair_audit=audit,
            )

    def test_quality_repair_outcome_requires_target_recovery_without_regression(
        self,
    ) -> None:
        def report(status: str, first: float, second: float) -> dict:
            return {
                "inputs": {
                    "reference_manifest_sha256": "a" * 64,
                    "mapping_mode": "deterministic_global_registration",
                    "selected_view_mapping": {
                        "ref_a": "front",
                        "ref_b": "side",
                    },
                },
                "thresholds": {"minimum_evidence_group_recall": 0.2},
                "aggregate": {
                    "status": status,
                    "comparable_view_count": 2,
                },
                "views": [
                    {
                        "reference_view_id": "ref_a",
                        "render_view_id": "front",
                        "status": status,
                        "mapping": {"selected_render_view_id": "front"},
                        "reference": {"image_sha256": "b" * 64},
                        "material_color": {
                            "trusted_evidence_group_recall": {
                                "groups": [
                                    {"group_id": "L01", "recall": first},
                                    {"group_id": "L02", "recall": 1.0},
                                ]
                            }
                        },
                    },
                    {
                        "reference_view_id": "ref_b",
                        "render_view_id": "side",
                        "status": status,
                        "mapping": {"selected_render_view_id": "side"},
                        "reference": {"image_sha256": "c" * 64},
                        "material_color": {
                            "trusted_evidence_group_recall": {
                                "groups": [{"group_id": "L09", "recall": second}]
                            }
                        },
                    },
                    {
                        "reference_view_id": "ref_c",
                        "render_view_id": None,
                        "status": "UNSCORABLE",
                        "mapping": {"selected_render_view_id": None},
                        "reference": {"image_sha256": "d" * 64},
                        "material_color": None,
                    },
                ],
            }

        baseline = report("FAIL", 0.0, 0.0)
        repaired = report("PASS", 1.0, 1.0)
        audit = {
            "changes": [
                {
                    "part_id": "P0001",
                    "canonical_group_id": "G01",
                    "supporting_view_ids": ["ref_a", "ref_b"],
                }
            ],
            "localization_lanes": [
                {
                    "part_id": "P0001",
                    "canonical_group_id": "G01",
                    "lane": "stable_spatial_multiview",
                }
            ],
            "group_diagnostics": [
                {
                    "canonical_group_id": "G01",
                    "repairable": True,
                    "supporting_views": [
                        {"reference_view_id": "ref_a", "local_group_id": "L01"},
                        {"reference_view_id": "ref_b", "local_group_id": "L09"},
                    ],
                }
            ],
        }
        _validate_quality_repair_outcome(
            baseline_quality=baseline,
            repaired_quality=repaired,
            repair_audit=audit,
        )

        low_evidence_baseline = report("FAIL", 0.0, 1.0)
        low_evidence_repaired = report("PASS", 0.19, 1.0)
        for quality in (low_evidence_baseline, low_evidence_repaired):
            quality["thresholds"].update(
                {
                    "minimum_reliable_group_evidence_pixels": 128,
                    "low_evidence_recall_tolerance_ratio": 0.9,
                    "minimum_low_evidence_observed_render_share": 0.001,
                }
            )
        low_evidence_group = low_evidence_repaired["views"][0]["material_color"][
            "trusted_evidence_group_recall"
        ]["groups"][0]
        low_evidence_group.update(
            {
                "reference_evidence_weight": 77,
                "observed_render_share": 0.001,
                "delivery_presence_status": (
                    "LOW_EVIDENCE_NEAR_THRESHOLD_PRESENT"
                ),
            }
        )
        low_evidence_audit = copy.deepcopy(audit)
        low_evidence_audit["changes"][0]["supporting_view_ids"] = ["ref_a"]
        low_evidence_audit["localization_lanes"][0]["lane"] = (
            "exact_spatial_single_qa_view"
        )
        low_evidence_audit["group_diagnostics"][0].update(
            {
                "repairable": False,
                "single_view_spatial_repairable": True,
            }
        )
        low_evidence_audit["group_diagnostics"][0]["supporting_views"] = [
            {
                "reference_view_id": "ref_a",
                "local_group_id": "L01",
            }
        ]
        _validate_quality_repair_outcome(
            baseline_quality=low_evidence_baseline,
            repaired_quality=low_evidence_repaired,
            repair_audit=low_evidence_audit,
        )

        low_evidence_group["observed_render_share"] = 0.0
        with self.assertRaisesRegex(
            RuntimeError, "did not recover its targeted evidence group"
        ):
            _validate_quality_repair_outcome(
                baseline_quality=low_evidence_baseline,
                repaired_quality=low_evidence_repaired,
                repair_audit=low_evidence_audit,
            )

        repaired["views"][0]["material_color"]["trusted_evidence_group_recall"][
            "groups"
        ][1]["recall"] = 0.0
        with self.assertRaisesRegex(RuntimeError, "new trusted-group deficit"):
            _validate_quality_repair_outcome(
                baseline_quality=baseline,
                repaired_quality=repaired,
                repair_audit=audit,
            )

    def test_quality_repair_outcome_validates_anchored_single_relationship(
        self,
    ) -> None:
        def report(status: str, recall: float) -> dict:
            return {
                "inputs": {
                    "reference_manifest_sha256": "a" * 64,
                    "mapping_mode": "deterministic_global_registration",
                    "selected_view_mapping": {
                        "ref_a": "front",
                        "ref_b": "side",
                    },
                },
                "thresholds": {"minimum_evidence_group_recall": 0.2},
                "aggregate": {
                    "status": status,
                    "comparable_view_count": 2,
                },
                "views": [
                    {
                        "reference_view_id": view_id,
                        "render_view_id": render_view_id,
                        "status": status,
                        "mapping": {"selected_render_view_id": render_view_id},
                        "reference": {"image_sha256": image_sha256},
                        "material_color": {
                            "trusted_evidence_group_recall": {
                                "groups": [
                                    {
                                        "group_id": local_group_id,
                                        "recall": recall,
                                    }
                                ]
                            }
                        },
                    }
                    for (
                        view_id,
                        render_view_id,
                        image_sha256,
                        local_group_id,
                    ) in (
                        ("ref_a", "front", "b" * 64, "L01"),
                        ("ref_b", "side", "c" * 64, "L09"),
                        ("ref_c", "top", "d" * 64, "L11"),
                    )
                ],
            }

        audit = {
            "changes": [
                {
                    "part_id": "P0001",
                    "canonical_group_id": "G01",
                    "supporting_view_ids": ["ref_a", "ref_b"],
                },
                {
                    "part_id": "P0002",
                    "canonical_group_id": "G01",
                    "supporting_view_ids": ["ref_b", "ref_c"],
                },
                {
                    "part_id": "P0003",
                    "canonical_group_id": "G01",
                    "supporting_view_ids": ["ref_a"],
                    "anchor_part_ids": ["P0001", "P0002"],
                    "anchor_supporting_view_ids": ["ref_a", "ref_b", "ref_c"],
                },
            ],
            "localization_lanes": [
                {
                    "part_id": "P0001",
                    "canonical_group_id": "G01",
                    "lane": "stable_spatial_multiview",
                },
                {
                    "part_id": "P0002",
                    "canonical_group_id": "G01",
                    "lane": "bounded_spatial_multiview",
                },
                {
                    "part_id": "P0003",
                    "canonical_group_id": "G01",
                    "lane": ("exact_spatial_single_view_with_multiview_anchor"),
                },
            ],
            "group_diagnostics": [
                {
                    "canonical_group_id": "G01",
                    "repairable": True,
                    "single_view_spatial_repairable": False,
                    "supporting_views": [
                        {"reference_view_id": "ref_a", "local_group_id": "L01"},
                        {"reference_view_id": "ref_b", "local_group_id": "L09"},
                        {"reference_view_id": "ref_c", "local_group_id": "L11"},
                    ],
                }
            ],
        }
        baseline = report("FAIL", 0.0)
        repaired = report("PASS", 1.0)

        _validate_quality_repair_outcome(
            baseline_quality=baseline,
            repaired_quality=repaired,
            repair_audit=audit,
        )

        audit["changes"][2]["anchor_part_ids"] = ["P0003"]
        with self.assertRaisesRegex(RuntimeError, "anchored single-view relationship"):
            _validate_quality_repair_outcome(
                baseline_quality=baseline,
                repaired_quality=repaired,
                repair_audit=audit,
            )

    def test_local_dominant_residual_cannot_bootstrap_anchored_single(
        self,
    ) -> None:
        def report(status: str, recall: float) -> dict:
            return {
                "inputs": {
                    "reference_manifest_sha256": "a" * 64,
                    "mapping_mode": "deterministic_global_registration",
                    "selected_view_mapping": {
                        "ref_a": "front",
                        "ref_b": "side",
                    },
                },
                "thresholds": {"minimum_evidence_group_recall": 0.2},
                "aggregate": {
                    "status": status,
                    "comparable_view_count": 2,
                },
                "views": [
                    {
                        "reference_view_id": view_id,
                        "render_view_id": render_view_id,
                        "status": status,
                        "mapping": {"selected_render_view_id": render_view_id},
                        "reference": {"image_sha256": image_sha256},
                        "material_color": {
                            "trusted_evidence_group_recall": {
                                "groups": [
                                    {
                                        "group_id": local_group_id,
                                        "recall": recall,
                                    }
                                ]
                            }
                        },
                    }
                    for (
                        view_id,
                        render_view_id,
                        image_sha256,
                        local_group_id,
                    ) in (
                        ("ref_a", "front", "b" * 64, "L01"),
                        ("ref_b", "side", "c" * 64, "L09"),
                    )
                ],
            }

        audit = {
            "changes": [
                {
                    "part_id": "P0001",
                    "canonical_group_id": "G01",
                    "supporting_view_ids": ["ref_a"],
                },
                {
                    "part_id": "P0002",
                    "canonical_group_id": "G01",
                    "supporting_view_ids": ["ref_b"],
                    "anchor_part_ids": ["P0001"],
                    "anchor_supporting_view_ids": ["ref_a", "ref_b"],
                },
            ],
            "localization_lanes": [
                {
                    "part_id": "P0001",
                    "canonical_group_id": "G01",
                    "lane": ("dominant_chromatic_residual_exact_single_view"),
                },
                {
                    "part_id": "P0002",
                    "canonical_group_id": "G01",
                    "lane": ("exact_spatial_single_view_with_multiview_anchor"),
                },
            ],
            "group_diagnostics": [
                {
                    "canonical_group_id": "G01",
                    "repairable": True,
                    "single_view_spatial_repairable": False,
                    "dominant_residual_repairable": True,
                    "dominant_residual_supporting_views": [
                        {
                            "reference_view_id": "ref_a",
                            "local_group_id": "L01",
                        }
                    ],
                    "supporting_views": [
                        {
                            "reference_view_id": "ref_a",
                            "local_group_id": "L01",
                        },
                        {
                            "reference_view_id": "ref_b",
                            "local_group_id": "L09",
                        },
                    ],
                }
            ],
        }

        with self.assertRaisesRegex(RuntimeError, "anchored single-view relationship"):
            _validate_quality_repair_outcome(
                baseline_quality=report("FAIL", 0.0),
                repaired_quality=report("PASS", 1.0),
                repair_audit=audit,
            )

    def test_quality_repair_outcome_rejects_changed_reference_contract(self) -> None:
        def report(status: str) -> dict:
            return {
                "inputs": {
                    "reference_manifest_sha256": "a" * 64,
                    "mapping_mode": "deterministic_global_registration",
                    "selected_view_mapping": {
                        "ref_a": "front",
                        "ref_b": "side",
                    },
                },
                "thresholds": {"minimum_evidence_group_recall": 0.2},
                "aggregate": {"status": status, "comparable_view_count": 2},
                "views": [
                    {
                        "reference_view_id": "ref_a",
                        "render_view_id": "front",
                        "status": status,
                        "mapping": {"selected_render_view_id": "front"},
                        "reference": {"image_sha256": "b" * 64},
                        "material_color": {
                            "trusted_evidence_group_recall": {
                                "groups": [{"group_id": "L01", "recall": 0.0}]
                            }
                        },
                    },
                    {
                        "reference_view_id": "ref_b",
                        "render_view_id": "side",
                        "status": status,
                        "mapping": {"selected_render_view_id": "side"},
                        "reference": {"image_sha256": "c" * 64},
                        "material_color": {
                            "trusted_evidence_group_recall": {
                                "groups": [{"group_id": "L09", "recall": 0.0}]
                            }
                        },
                    },
                ],
            }

        baseline = report("FAIL")
        repaired = report("PASS")
        repaired["views"][0]["reference"]["image_sha256"] = "d" * 64
        audit = {
            "changes": [
                {
                    "part_id": "P0001",
                    "canonical_group_id": "G01",
                    "supporting_view_ids": ["ref_a", "ref_b"],
                }
            ],
            "localization_lanes": [
                {
                    "part_id": "P0001",
                    "canonical_group_id": "G01",
                    "lane": "bounded_spatial_multiview",
                }
            ],
            "group_diagnostics": [
                {
                    "canonical_group_id": "G01",
                    "repairable": True,
                    "supporting_views": [
                        {"reference_view_id": "ref_a", "local_group_id": "L01"},
                        {"reference_view_id": "ref_b", "local_group_id": "L09"},
                    ],
                }
            ],
        }
        with self.assertRaisesRegex(RuntimeError, "changed references"):
            _validate_quality_repair_outcome(
                baseline_quality=baseline,
                repaired_quality=repaired,
                repair_audit=audit,
            )


if __name__ == "__main__":
    unittest.main()
