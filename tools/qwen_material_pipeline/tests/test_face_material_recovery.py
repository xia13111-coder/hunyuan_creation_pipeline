from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import qwen_material_pipeline.evidence.face_recovery as recovery
from qwen_material_pipeline.evidence.face_recovery import (
    _candidate_decisions,
    _face_indices,
    _filter_projection_compatible_views,
    _group_material_candidates,
    _observe_region,
    _spatial_override_candidates,
    build_face_material_recovery,
)


def _groups() -> list[dict[str, str]]:
    return [
        {"group_id": "G01", "base_color": "green"},
        {"group_id": "G02", "base_color": "black"},
    ]


def _view(image: np.ndarray) -> dict[str, Any]:
    return {
        "reference_view_id": "ref_front",
        "render_view_id": "front",
        "quarter_turns_ccw": 0,
        "bbox_affine": np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        "ecc_warp": np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        "raw_image": image,
        "albedo_image": image,
    }


def _policy() -> dict[str, float | int | bool]:
    return {
        **recovery.DEFAULT_POLICY,
        "minimum_projected_pixels": 16,
        "projection_scale_delta": 0.0,
    }


def test_region_requires_raw_and_albedo_support_under_perturbation() -> None:
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[8:24, 8:24] = 7
    green = np.zeros((32, 32, 3), dtype=np.uint8)
    green[:, :] = (20, 180, 20)

    observation = _observe_region(
        numeric_region_id=7,
        target_group_id="G01",
        groups=_groups(),
        view=_view(green),
        labels=labels,
        policy=_policy(),
    )

    assert observation["classification"] == "support"
    assert observation["support_profile"] == "strict"
    assert observation["canonical_group_id"] == "G01"
    assert observation["profiles"][0]["perturbation_stable"] is True


def test_unprojected_trusted_pose_is_excluded_without_guessing() -> None:
    manifest = {
        "projection_contract": {
            "views": [
                {"view_id": "front"},
                {"view_id": "right"},
            ]
        }
    }
    views = {
        "ref_front": {"render_view_id": "right"},
        "ref_side": {"render_view_id": "front"},
        "ref_iso": {"render_view_id": "pose_a135_e015"},
    }

    compatible, excluded = _filter_projection_compatible_views(
        manifest=manifest,
        views=views,
    )

    assert list(compatible) == ["ref_front", "ref_side"]
    assert excluded == ["pose_a135_e015"]


def test_stable_non_target_region_is_a_conflict() -> None:
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[8:24, 8:24] = 7
    black = np.zeros((32, 32, 3), dtype=np.uint8)

    observation = _observe_region(
        numeric_region_id=7,
        target_group_id="G01",
        groups=_groups(),
        view=_view(black),
        labels=labels,
        policy=_policy(),
    )

    assert observation["classification"] == "conflict"
    assert observation["reason_code"] == "stable_non_target_material_observed"
    assert observation["canonical_group_ids"] == ["G02"]


def test_face_ranges_are_expanded_and_cross_checked() -> None:
    patch = {
        "face_count": 4,
        "face_ranges": [[2, 3], [7, 8]],
        "face_indices": [2, 3, 7, 8],
    }

    assert _face_indices(patch, 10) == [2, 3, 7, 8]


def _gate_decision() -> dict[str, Any]:
    return {
        "part_id": "P0001",
        "decision": "preserve",
        "material_id": "painted-green",
        "model": {
            "status": "auto",
            "confidence": 0.95,
            "independent_reference_count": 2,
        },
        "mapping": {"group_id": "G01", "confidence": 0.85},
        "material_choice": {
            "confirmed": True,
            "forward_material_id": "painted-green",
            "reverse_material_id": "painted-green",
            "forward_confidence": 0.95,
            "reverse_confidence": 0.95,
        },
        "candidate_margin": 0.30,
        "multi_material_risk": True,
        "geometry_risk": {"risk": {"multi_material_risk": True}},
        "reason_codes": [
            "MAPPING_BELOW_AUTO",
            "MULTI_MATERIAL_RISK",
            "GEOMETRY_MULTI_MATERIAL_RISK",
        ],
    }


def test_candidate_filter_rejects_any_unrelated_failure_reason() -> None:
    gate = {
        "schema_version": "qwen-material-confidence-gate/v1",
        "policy": {
            "auto_model_confidence": 0.9,
            "review_mapping_confidence": 0.6,
            "auto_material_choice_confidence": 0.85,
            "minimum_candidate_margin": 0.15,
            "minimum_independent_references": 2,
        },
        "decisions": [_gate_decision()],
    }
    assert set(_candidate_decisions(gate)) == {"P0001"}

    gate["decisions"][0]["reason_codes"].append("CROSS_VIEW_MATERIAL_CONFLICT")
    assert _candidate_decisions(gate) == {}


def test_excluded_group_never_enters_standard_candidate_creation() -> None:
    gate = {
        "schema_version": recovery.CONFIDENCE_GATE_SCHEMA_VERSION,
        "policy": {
            "auto_model_confidence": 0.9,
            "review_mapping_confidence": 0.6,
            "auto_material_choice_confidence": 0.85,
            "minimum_candidate_margin": 0.15,
            "minimum_independent_references": 2,
        },
        "decisions": [_gate_decision()],
    }

    assert set(_candidate_decisions(gate)) == {"P0001"}
    assert _candidate_decisions(gate, excluded_group_ids={"G01"}) == {}


def _group_material_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    material_id = "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Green"
    group_materials = {
        "schema_version": recovery.GROUP_MATERIAL_SCHEMA_VERSION,
        "selections": [
            {
                "group_id": "G01",
                "material_id": material_id,
                "confidence": 0.95,
                "confirmed": True,
            }
        ],
    }
    audit = {
        "G01": {
            "chosen_retrieval_rank": 1,
            "model_choice_matches_retrieval_top": True,
            "confirmed": True,
            "forward": {
                "group_id": "G01",
                "material_id": material_id,
                "confidence": 0.95,
            },
            "reverse": {
                "group_id": "G01",
                "material_id": material_id,
                "confidence": 0.94,
            },
            "independent_view_choices": [
                {
                    "view_id": view_id,
                    "canonical_group_id": "G01",
                    "source_local_group_id": "LOCAL_GREEN",
                    "material_id": material_id,
                    "confidence": 0.97,
                    "candidate_margin": 0.25,
                    "mvinverse_association": {
                        "status": "matched",
                        "candidate_group_ids": ["LOCAL_GREEN"],
                        "matched_group_id": "LOCAL_GREEN",
                    },
                }
                for view_id in ("ref_front", "ref_top", "ref_iso")
            ],
        }
    }
    return group_materials, audit


def _spatial_override_gate() -> dict[str, Any]:
    return {
        "schema_version": recovery.CONFIDENCE_GATE_SCHEMA_VERSION,
        "decisions": [
            {
                "part_id": "P0001",
                "decision": "review",
                "multi_material_risk": False,
                "model": {
                    "status": "review",
                    "confidence": 0.60,
                    "unknown_reason_code": None,
                    "independent_reference_count": 2,
                },
                "mapping": {"group_id": "G03"},
                "geometry_risk": {"risk": {"multi_material_risk": False}},
                "reason_codes": [
                    "MODEL_STATUS_REQUIRES_REVIEW",
                    "MODEL_CONFIDENCE_BELOW_AUTO",
                    "MAPPING_BELOW_AUTO",
                ],
            }
        ],
    }


def _spatial_override_report() -> dict[str, Any]:
    observations = []
    for index, view_id in enumerate(("ref_front", "ref_top", "ref_iso")):
        observations.append(
            {
                "reference_view_id": view_id,
                "classification": "resolved",
                "canonical_group_id": "G01",
                "projected_part_pixels": 700,
                "color_margin": 0.50,
                "perturbation_label_stable": True,
                "registration_label_stable": True if index < 2 else None,
                "group_scores": [
                    {
                        "canonical_group_id": "G01",
                        "color_share": 0.75,
                    }
                ],
            }
        )
    return {
        "parts": [
            {
                "part_id": "P0001",
                "resolved_support_counts": {"G01": 3},
                "conflict_view_ids": [],
                "observations": observations,
            }
        ]
    }


def test_spatial_override_accepts_two_registered_independent_views() -> None:
    group_materials, audit = _group_material_evidence()
    material_id = group_materials["selections"][0]["material_id"]
    choices = _group_material_candidates(
        group_materials=group_materials,
        material_choice_audit=audit,
        allowed_material_ids={material_id},
        policy=recovery.DEFAULT_POLICY,
    )
    candidates = _spatial_override_candidates(
        confidence_gate=_spatial_override_gate(),
        spatial_mapping_report=_spatial_override_report(),
        group_material_candidates=choices,
        policy=recovery.DEFAULT_POLICY,
    )
    assert candidates["P0001"]["group_id"] == "G01"
    assert candidates["P0001"]["source_mapping_group_id"] == "G03"
    assert candidates["P0001"]["material_id"] == material_id
    assert candidates["P0001"]["confidence"] == 0.94
    assert candidates["P0001"]["minimum_spatial_color_share"] == 0.75

    report = _spatial_override_report()
    report["parts"][0]["observations"] = report["parts"][0]["observations"][:2]
    report["parts"][0]["resolved_support_counts"] = {"G01": 2}
    two_view_candidates = _spatial_override_candidates(
        confidence_gate=_spatial_override_gate(),
        spatial_mapping_report=report,
        group_material_candidates=choices,
        policy=recovery.DEFAULT_POLICY,
    )
    assert two_view_candidates["P0001"]["spatial_supporting_view_ids"] == [
        "ref_front",
        "ref_top",
    ]

    report["parts"][0]["observations"] = report["parts"][0]["observations"][:1]
    report["parts"][0]["resolved_support_counts"] = {"G01": 1}
    assert (
        _spatial_override_candidates(
            confidence_gate=_spatial_override_gate(),
            spatial_mapping_report=report,
            group_material_candidates=choices,
            policy=recovery.DEFAULT_POLICY,
        )
        == {}
    )


def test_excluded_group_never_enters_spatial_candidate_creation() -> None:
    group_materials, audit = _group_material_evidence()
    material_id = group_materials["selections"][0]["material_id"]
    unfiltered_choices = _group_material_candidates(
        group_materials=group_materials,
        material_choice_audit=audit,
        allowed_material_ids={material_id},
        policy=recovery.DEFAULT_POLICY,
    )

    assert (
        _group_material_candidates(
            group_materials=group_materials,
            material_choice_audit=audit,
            allowed_material_ids={material_id},
            policy=recovery.DEFAULT_POLICY,
            excluded_group_ids={"G01"},
        )
        == {}
    )
    assert (
        _spatial_override_candidates(
            confidence_gate=_spatial_override_gate(),
            spatial_mapping_report=_spatial_override_report(),
            group_material_candidates=unfiltered_choices,
            policy=recovery.DEFAULT_POLICY,
            excluded_group_ids={"G01"},
        )
        == {}
    )


def test_group_material_candidate_uses_derived_not_reported_confidence() -> None:
    group_materials, audit = _group_material_evidence()
    material_id = group_materials["selections"][0]["material_id"]
    group_materials["selections"][0]["confidence"] = 0.90
    choice = audit["G01"]
    choice["forward"]["confidence"] = 0.0
    choice["reverse"]["confidence"] = 0.0
    for view in choice["independent_view_choices"]:
        view["confidence"] = 0.0
    choice["selection_confidence"] = 0.90
    choice["confidence_derivation"] = {
        "schema_version": "qwen-derived-material-selection-confidence/v1",
        "derived_confidence": 0.90,
        "reported_confidence_is_authoritative": False,
    }

    candidates = _group_material_candidates(
        group_materials=group_materials,
        material_choice_audit=audit,
        allowed_material_ids={material_id},
        policy=recovery.DEFAULT_POLICY,
    )

    assert candidates["G01"]["confidence"] == 0.90


def test_spatial_override_recovers_unknown_only_with_three_registered_views() -> None:
    group_materials, audit = _group_material_evidence()
    material_id = group_materials["selections"][0]["material_id"]
    choices = _group_material_candidates(
        group_materials=group_materials,
        material_choice_audit=audit,
        allowed_material_ids={material_id},
        policy=recovery.DEFAULT_POLICY,
    )
    gate = _spatial_override_gate()
    decision = gate["decisions"][0]
    decision["decision"] = "preserve"
    decision["model"] = {
        "status": "unknown",
        "confidence": None,
        "unknown_reason_code": "not_in_reference",
    }
    decision["mapping"] = {
        "group_id": None,
        "status": "unknown",
        "confidence": 0.0,
    }
    decision["reason_codes"] = ["NO_MODEL_ASSIGNMENT"]
    report = _spatial_override_report()
    for observation in report["parts"][0]["observations"]:
        observation["registration_label_stable"] = True

    candidates = _spatial_override_candidates(
        confidence_gate=gate,
        spatial_mapping_report=report,
        group_material_candidates=choices,
        policy=recovery.DEFAULT_POLICY,
    )
    assert candidates["P0001"]["spatial_override_lane"] == (
        "three_view_unknown_recovery"
    )

    report["parts"][0]["observations"] = report["parts"][0]["observations"][:2]
    report["parts"][0]["resolved_support_counts"] = {"G01": 2}
    assert (
        _spatial_override_candidates(
            confidence_gate=gate,
            spatial_mapping_report=report,
            group_material_candidates=choices,
            policy=recovery.DEFAULT_POLICY,
        )
        == {}
    )


def test_group_material_override_rejects_one_conflicting_view() -> None:
    group_materials, audit = _group_material_evidence()
    material_id = group_materials["selections"][0]["material_id"]
    audit["G01"]["independent_view_choices"][2]["material_id"] = "other"
    assert (
        _group_material_candidates(
            group_materials=group_materials,
            material_choice_audit=audit,
            allowed_material_ids={material_id, "other"},
            policy=recovery.DEFAULT_POLICY,
        )
        == {}
    )


def test_group_material_override_accepts_confirmed_tunable_export_variants() -> None:
    army = (
        "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted_Army_Green"
    )
    arcadia = (
        "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted_Arcadia_Green"
    )
    group_materials, audit = _group_material_evidence()
    group_materials["selections"][0]["material_id"] = army
    group_audit = audit["G01"]
    group_audit.update(
        confirmation_basis="mvinverse_tunable_module_agreement",
        confirmed_material_id=army,
    )
    group_audit["forward"]["material_id"] = army
    group_audit["reverse"]["material_id"] = arcadia
    for view in group_audit["independent_view_choices"]:
        view["material_id"] = army

    choices = _group_material_candidates(
        group_materials=group_materials,
        material_choice_audit=audit,
        allowed_material_ids={army, arcadia},
        policy=recovery.DEFAULT_POLICY,
    )

    assert choices["G01"]["material_id"] == army


def test_spatial_override_allows_multi_material_part_for_face_recovery() -> None:
    group_materials, audit = _group_material_evidence()
    material_id = group_materials["selections"][0]["material_id"]
    choices = _group_material_candidates(
        group_materials=group_materials,
        material_choice_audit=audit,
        allowed_material_ids={material_id},
        policy=recovery.DEFAULT_POLICY,
    )
    gate = _spatial_override_gate()
    decision = gate["decisions"][0]
    decision["decision"] = "preserve"
    decision["multi_material_risk"] = True
    decision["geometry_risk"]["risk"]["multi_material_risk"] = True
    decision["reason_codes"].extend(
        ["MULTI_MATERIAL_RISK", "GEOMETRY_MULTI_MATERIAL_RISK"]
    )

    candidates = _spatial_override_candidates(
        confidence_gate=gate,
        spatial_mapping_report=_spatial_override_report(),
        group_material_candidates=choices,
        policy=recovery.DEFAULT_POLICY,
    )

    assert candidates["P0001"]["candidate_source"] == "spatial_consensus_override"


def test_recovery_emits_subset_only_assignment(
    tmp_path: Path, monkeypatch: Any
) -> None:
    part = {
        "surface_patch_method": recovery.SURFACE_PATCH_METHOD,
        "face_count": 4,
        "total_area_world": 2.0,
        "surface_patches": [
            {
                "region_id": "R0001",
                "numeric_region_id": 9,
                "face_count": 4,
                "face_ranges": [[0, 3]],
                "face_indices": [0, 1, 2, 3],
                "area_world": 2.0,
                "max_normal_deviation_degrees": 0.0,
            }
        ],
    }
    part_path = tmp_path / "P0001.json"
    part_path.write_text(json.dumps(part), encoding="utf-8")
    manifest = {
        "schema_version": "qwen-face-region-evidence/v1",
        "surface_patch_method": recovery.SURFACE_PATCH_METHOD,
        "asset_sha256": "asset",
        "registry_sha256": "registry",
        "parts": [
            {
                "part_id": "P0001",
                "face_count": 4,
                "surface_patch_count": 1,
                "evidence": "P0001.json",
            }
        ],
        "projection_contract": {
            "rendered_registry_sha256": "registry",
            "views": [{"view_id": "front"}],
        },
        "source_usd_unchanged": True,
        "source_usd_sha256_before": "asset",
        "source_usd_sha256_after": "asset",
        "projection_view_count": 1,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(recovery, "_verify_report", lambda _value: None)
    monkeypatch.setattr(recovery, "validate_mvinverse_evidence", lambda value: value)
    monkeypatch.setattr(
        recovery,
        "_load_view_inputs",
        lambda **_kwargs: {
            "ref_a": {"reference_view_id": "ref_a", "render_view_id": "front"},
            "ref_b": {"reference_view_id": "ref_b", "render_view_id": "front"},
        },
    )
    monkeypatch.setattr(
        recovery,
        "_load_region_labels",
        lambda **_kwargs: {"front": np.ones((4, 4), dtype=np.int32)},
    )
    monkeypatch.setattr(
        recovery,
        "_observe_region",
        lambda **kwargs: {
            "reference_view_id": kwargs["view"]["reference_view_id"],
            "classification": "support",
        },
    )
    monkeypatch.setattr(
        recovery,
        "parameterize_auto_material_plan",
        lambda **kwargs: {
            "material_plan": {
                "assignments": list(
                    kwargs["auto_material_plan"]["assignments"]
                )
            },
            "decisions": [{"part_id": "P0001", "parameterized": False}],
        },
    )
    gate = {
        "schema_version": "qwen-material-confidence-gate/v1",
        "policy": {
            "auto_model_confidence": 0.9,
            "review_mapping_confidence": 0.6,
            "auto_material_choice_confidence": 0.85,
            "minimum_candidate_margin": 0.15,
            "minimum_independent_references": 2,
        },
        "decisions": [_gate_decision()],
    }
    spatial = {
        "inputs": {"files": [{"label": "rendered_registry", "sha256": "registry"}]}
    }
    palette = {
        "schema_version": "qwen-material-palette/v1",
        "groups": [
            {
                "group_id": "G01",
                "base_color": "green",
                "visual_description": "green housing",
            }
        ],
    }
    mvinverse = {
        "groups": [
            {
                "group_id": "G01",
                "suggestion": {"auto_parameter_eligible": True},
            }
        ]
    }

    report = build_face_material_recovery(
        base_material_plan={"schema_version": "1.0", "assignments": []},
        confidence_gate=gate,
        face_region_manifest=manifest_path,
        spatial_mapping_report=spatial,
        canonical_palette=palette,
        mvinverse_evidence=mvinverse,
        batches=[{"mappings": []}],
        allowed_material_ids=["generic-painted", "painted-green"],
        policy={"minimum_projected_pixels": 1},
    )

    assignment = report["material_plan"]["assignments"][0]
    assert assignment["material_id"] == "painted-green"
    assert "parameters" not in assignment
    assert assignment["preserve_parent_material_binding"] is True
    assert assignment["face_subsets"][0]["face_indices"] == [0, 1, 2, 3]
    assert report["summary"]["parent_material_bindings_preserved"] is True


def test_insufficient_registered_views_preserve_base_plan_without_face_subsets(
    tmp_path: Path, monkeypatch: Any
) -> None:
    reference_path = tmp_path / "reference.png"
    Image.new("RGB", (8, 8), "green").save(reference_path)
    reference_sha256 = recovery._sha256_file(reference_path)
    manifest = {
        "schema_version": recovery.FACE_REGION_SCHEMA_VERSION,
        "surface_patch_method": recovery.SURFACE_PATCH_METHOD,
        "asset_sha256": "asset",
        "registry_sha256": "registry",
        "parts": [
            {
                "part_id": "P0001",
                "face_count": 4,
                "surface_patch_count": 1,
                "evidence": "unused-because-recovery-is-skipped.json",
            }
        ],
        "projection_contract": {
            "rendered_registry_sha256": "registry",
            "views": [{"view_id": "front"}],
        },
        "source_usd_unchanged": True,
        "source_usd_sha256_before": "asset",
        "source_usd_sha256_after": "asset",
        "projection_view_count": 1,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    gate = {
        "schema_version": recovery.CONFIDENCE_GATE_SCHEMA_VERSION,
        "policy": {
            "auto_model_confidence": 0.9,
            "review_mapping_confidence": 0.6,
            "auto_material_choice_confidence": 0.85,
            "minimum_candidate_margin": 0.15,
            "minimum_independent_references": 2,
        },
        "decisions": [_gate_decision()],
    }
    spatial = {
        "inputs": {
            "files": [
                {"label": "rendered_registry", "sha256": "registry"},
                {
                    "label": "reference_image:ref_front",
                    "path": str(reference_path),
                },
            ]
        },
        "view_alignments": [
            {
                "trusted": True,
                "reference_view_id": "ref_front",
                "selected_render_view_id": "front",
                "quarter_turns_ccw": 0,
                "bbox_affine": [[1, 0, 0], [0, 1, 0]],
                "ecc_warp": [[1, 0, 0], [0, 1, 0]],
            }
        ],
    }
    palette = {
        "schema_version": "qwen-material-palette/v1",
        "groups": [
            {
                "group_id": "G01",
                "base_color": "green",
                "visual_description": "green housing",
            }
        ],
    }
    mvinverse = {
        "views": [
            {
                "view_id": "ref_front",
                "sources": {
                    "albedo": {
                        "path": str(reference_path),
                        "sha256": reference_sha256,
                    }
                },
            }
        ],
        "groups": [
            {
                "group_id": "G01",
                "suggestion": {"auto_parameter_eligible": True},
            }
        ],
    }
    base_plan = {
        "schema_version": "1.0",
        "assignments": [
            {
                "part_id": "P_BASE",
                "material_id": "generic-painted",
                "status": "auto",
            }
        ],
    }

    monkeypatch.setattr(recovery, "_verify_report", lambda _value: None)
    monkeypatch.setattr(recovery, "validate_mvinverse_evidence", lambda value: value)

    def fail_if_face_regions_are_loaded(**_kwargs: Any) -> dict[str, np.ndarray]:
        raise AssertionError("face-region labels must not be loaded after gate skip")

    def fail_if_parameterized(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("face-level material must not be parameterized")

    monkeypatch.setattr(
        recovery, "_load_region_labels", fail_if_face_regions_are_loaded
    )
    monkeypatch.setattr(
        recovery, "parameterize_auto_material_plan", fail_if_parameterized
    )

    report = build_face_material_recovery(
        base_material_plan=base_plan,
        confidence_gate=gate,
        face_region_manifest=manifest_path,
        spatial_mapping_report=spatial,
        canonical_palette=palette,
        mvinverse_evidence=mvinverse,
        batches=[],
        allowed_material_ids=["generic-painted", "painted-green"],
    )

    assert report["recovery_gate"] == {
        "status": "SKIPPED_INSUFFICIENT_EVIDENCE",
        "decision": "preserve_base_material_plan",
        "reason_codes": [recovery.INSUFFICIENT_TRUSTED_VIEWS_REASON],
        "trusted_registered_view_count": 1,
        "minimum_trusted_registered_view_count": (
            recovery.MINIMUM_TRUSTED_REGISTERED_VIEWS
        ),
        "candidate_part_ids": ["P0001"],
        "face_region_labels_loaded": False,
        "face_parameterization_attempted": False,
        "face_subset_assignments_emitted": 0,
    }
    assert report["summary"]["recovered_part_count"] == 0
    assert report["summary"]["recovered_subset_count"] == 0
    assert report["summary"]["recovered_face_count"] == 0
    assert report["material_plan"] == base_plan
    assert report["material_plan"] is not base_plan
    assert report["material_plan"]["assignments"][0] is not base_plan["assignments"][0]
    assert all(
        "face_subsets" not in assignment
        for assignment in report["material_plan"]["assignments"]
    )
    unsigned = dict(report)
    integrity = unsigned.pop("integrity")
    assert integrity == {"report_sha256": recovery._sha256_document(unsigned)}

    excluded_report = build_face_material_recovery(
        base_material_plan=base_plan,
        confidence_gate=gate,
        face_region_manifest=manifest_path,
        spatial_mapping_report=spatial,
        canonical_palette=palette,
        mvinverse_evidence=mvinverse,
        batches=[],
        allowed_material_ids=["generic-painted", "painted-green"],
        excluded_group_ids={"G01"},
    )

    assert excluded_report["recovery_gate"]["candidate_part_ids"] == []
    assert excluded_report["summary"]["excluded_material_group_count"] == 1
    assert excluded_report["material_collapse_recovery"] == {
        "excluded_group_ids": ["G01"],
        "candidate_policy": (
            "exclude_before_standard_and_spatial_candidate_creation"
        ),
    }
    assert excluded_report["inputs"]["document_sha256"][
        "excluded_group_ids"
    ] == recovery._sha256_document(["G01"])
    assert excluded_report["material_plan"] == base_plan
