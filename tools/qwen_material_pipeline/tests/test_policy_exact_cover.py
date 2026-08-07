from __future__ import annotations

import copy

import pytest

import qwen_material_pipeline.materials.policy_exact_cover as policy_module
from qwen_material_pipeline.evidence.palette_fusion import (
    fuse_multiview_palettes,
)
from qwen_material_pipeline.materials.policy_exact_cover import (
    BLACK_PLASTIC,
    BLACK_RUBBER,
    BLACK_PAINTED_STEEL,
    BRASS,
    COPPER,
    FALLBACK_STATUS,
    GALVANIZED_STEEL,
    GENERIC_STEEL_PAINTED,
    POLICY_SCHEMA_VERSION,
    STAINLESS_STEEL_MATTE,
    PolicyExactCoverError,
    build_policy_exact_cover,
)
from qwen_material_pipeline.mvinverse.evidence import (
    validate_mvinverse_evidence as strict_validate_mvinverse_evidence,
)
from qwen_material_pipeline.usd.material_common import (
    SOURCE_VISUAL_PRESERVE_ACTION,
    SOURCE_VISUAL_PRESERVE_TIER,
    source_visual_binding_sha256,
)
from qwen_material_pipeline.usd.registry import (
    SOURCE_MATERIAL_BIND_SUBSETS_FIELD,
    source_material_bind_subset_sha256,
)


@pytest.fixture(autouse=True)
def _isolate_policy_logic_from_full_evidence_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Most tests below exercise policy joining after upstream validation."""

    monkeypatch.setattr(
        policy_module,
        "validate_mvinverse_evidence",
        lambda document: copy.deepcopy(document),
    )


GREEN = "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted_Army_Green"
WHITE = "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted_White"


def _part(
    part_id: str,
    parent: str,
    *,
    existing_material: str | None = None,
    geometry_fingerprint: str | None = None,
    visible_pixels: int = 0,
) -> dict:
    return {
        "part_id": part_id,
        "prim_path": f"{parent}/Mesh",
        "prim_name": "Mesh",
        "parent_path": parent,
        "point_count": 8,
        "face_count": 6,
        "world_bbox": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        "existing_visual_material": existing_material,
        "geometry_fingerprint": geometry_fingerprint,
        "existing_physics_material": None,
        "renders": (
            [{"view_id": "front", "visible_pixels": visible_pixels}]
            if visible_pixels
            else []
        ),
    }


def _with_source_subsets(
    part: dict,
    subsets: list[tuple[str, list[int]]],
) -> dict:
    result = copy.deepcopy(part)
    records = []
    for name, indices in sorted(subsets):
        record = {
            "subset_name": name,
            "subset_prim_path": f"{part['prim_path']}/{name}",
            "family_name": "materialBind",
            "family_type": "nonOverlapping",
            "element_type": "face",
            "face_indices": indices,
            "visual_material_prim_path": f"/Asset/Looks/{name}",
            "binding_relationship_name": "material:binding",
            "binding_targets": [f"/Asset/Looks/{name}"],
        }
        record["source_subset_binding_sha256"] = (
            source_material_bind_subset_sha256(
                part_id=str(part["part_id"]),
                prim_path=str(part["prim_path"]),
                subset_record=record,
            )
        )
        records.append(record)
    result[SOURCE_MATERIAL_BIND_SUBSETS_FIELD] = records
    return result


def _documents() -> tuple[dict, dict, dict, dict]:
    parts = [
        _part("P0001", "/Asset/Frame", visible_pixels=1000),
        _part("P0002", "/Asset/Frame"),
        _part("P0003", "/Asset/Panel"),
        _part("P0004", "/Asset/GBT97_WASHER"),
        _part("P0005", "/Asset/OccludedCustom"),
    ]
    registry = {
        "schema_version": "qwen-material-parts/v1",
        "asset_sha256": "a" * 64,
        "part_count": len(parts),
        "parts": parts,
    }
    staged = {
        "schema_version": "qwen-staged-material-result/v1",
        "material_plan": {
            "schema_version": "1.0",
            "assignments": [
                {
                    "part_id": "P0001",
                    "material_id": GREEN,
                    "semantic": "green frame",
                    "confidence": 0.95,
                    "evidence_views": ["front", "iso"],
                    "status": "auto",
                },
                {
                    "part_id": "P0003",
                    "material_id": WHITE,
                    "semantic": "white panel",
                    "confidence": 0.6,
                    "evidence_views": ["front"],
                    "status": "review",
                },
            ],
        },
    }
    gate = {
        "schema_version": "qwen-material-confidence-gate/v1",
        "decisions": [
            {
                "part_id": part["part_id"],
                "decision": "auto" if part["part_id"] == "P0001" else "preserve",
            }
            for part in parts
        ],
    }
    whitelist = {
        "schema_version": 1,
        "material_ids": [
            GREEN,
            WHITE,
            GALVANIZED_STEEL,
            STAINLESS_STEEL_MATTE,
            BLACK_RUBBER,
            BLACK_PLASTIC,
            BLACK_PAINTED_STEEL,
            COPPER,
            BRASS,
            GENERIC_STEEL_PAINTED,
        ],
    }
    return registry, staged, gate, whitelist


def test_policy_cover_is_exact_and_never_promotes_fallback_confidence() -> None:
    registry, staged, gate, whitelist = _documents()
    original_staged = copy.deepcopy(staged)

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        acknowledge_policy_fallback=True,
    )

    assert staged == original_staged
    assignments = {item["part_id"]: item for item in plan["assignments"]}
    assert set(assignments) == {f"P{index:04d}" for index in range(1, 6)}
    assert assignments["P0001"]["status"] == "auto"
    assert assignments["P0001"]["confidence"] == pytest.approx(0.95)

    unresolved = assignments["P0002"]
    assert unresolved["status"] == FALLBACK_STATUS
    assert unresolved["confidence"] == 0.0
    assert unresolved["evidence_views"] == []
    assert unresolved["material_id"] == STAINLESS_STEEL_MATTE
    assert unresolved["provenance"]["tier"] == "neutral_default"
    assert unresolved["provenance"]["sources"] == []

    review = assignments["P0003"]
    assert review["status"] == FALLBACK_STATUS
    assert review["confidence"] == 0.0
    assert review["material_id"] == STAINLESS_STEEL_MATTE
    assert review["provenance"]["tier"] == "neutral_default"
    assert review["provenance"]["sources"] == []

    fastener = assignments["P0004"]
    assert fastener["material_id"] == GALVANIZED_STEEL
    assert fastener["provenance"]["tier"] == "semantic_rule"

    defaulted = assignments["P0005"]
    assert defaulted["material_id"] == STAINLESS_STEEL_MATTE
    assert defaulted["provenance"]["reason_codes"] == [
        "POLICY_DECLARED_NEUTRAL_DEFAULT"
    ]
    assert report["summary"] == {
        "registry_part_count": 5,
        "staged_candidate_count": 2,
        "autonomous_base_assignment_count": 0,
        "confidence_gate_auto_count": 1,
        "output_assignment_count": 5,
        "policy_fallback_count": 4,
        "mvinverse_parameterized_part_count": 0,
        "neutral_default_count": 3,
        "source_visual_preserve_count": 0,
        "corroborated_source_visual_preserve_count": 0,
        "corroborated_source_visual_nvidia_mdl_count": 0,
        "source_preserve_unavailable_neutral_fallback_count": 0,
        "source_material_bind_subset_part_count": 0,
        "source_material_bind_subset_count": 0,
        "source_subset_parent_material_expansion_part_count": 0,
        "source_subset_parent_material_expansion_count": 0,
        "source_subset_explicit_topology_match_part_count": 0,
        "source_subset_explicit_topology_collapse_part_count": 0,
        "exact_cover": True,
        "all_materials_in_industrial_whitelist": True,
        "applicable_without_explicit_policy_fallback_authorization": False,
    }


def test_policy_cover_requires_explicit_acknowledgement() -> None:
    registry, staged, gate, whitelist = _documents()
    with pytest.raises(PolicyExactCoverError, match="explicit"):
        build_policy_exact_cover(
            registry=registry,
            staged_result=staged,
            confidence_gate=gate,
            whitelist=whitelist,
            acknowledge_policy_fallback=False,
        )


def test_whole_mesh_selection_expands_over_source_material_subsets() -> None:
    registry, staged, gate, whitelist = _documents()
    registry["parts"][0] = _with_source_subsets(
        registry["parts"][0],
        [("Paint", [0, 1, 2]), ("Steel", [3, 4, 5])],
    )
    selected = staged["material_plan"]["assignments"][0]
    selected["parameters"] = {"paint_color": [0.1, 0.2, 0.3]}

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        acknowledge_policy_fallback=True,
    )

    assignment = plan["assignments"][0]
    assert assignment["material_id"] == GREEN
    assert assignment["face_subsets"] == [
        {
            "subset_name": "Paint",
            "material_id": GREEN,
            "face_indices": [0, 1, 2],
            "parameters": {"paint_color": [0.1, 0.2, 0.3]},
        },
        {
            "subset_name": "Steel",
            "material_id": GREEN,
            "face_indices": [3, 4, 5],
            "parameters": {"paint_color": [0.1, 0.2, 0.3]},
        },
    ]
    assert assignment["provenance"][
        "source_material_bind_subset_contract"
    ]["action"] == "parent_material_expansion"
    assert report["summary"]["source_material_bind_subset_part_count"] == 1
    assert report["summary"]["source_material_bind_subset_count"] == 2
    assert (
        report["summary"]["source_subset_parent_material_expansion_part_count"]
        == 1
    )
    assert report["summary"]["source_subset_parent_material_expansion_count"] == 2
    assert report["source_material_bind_subsets"][
        "parent_material_expansion_part_ids"
    ] == ["P0001"]


def test_exact_explicit_source_subset_topology_preserves_subset_materials() -> None:
    registry, staged, gate, whitelist = _documents()
    registry["parts"][0] = _with_source_subsets(
        registry["parts"][0],
        [("Paint", [0, 1, 2]), ("Steel", [3, 4, 5])],
    )
    staged["material_plan"]["assignments"][0]["face_subsets"] = [
        {
            "subset_name": "Steel",
            "material_id": GREEN,
            "face_indices": [3, 4, 5],
        },
        {
            "subset_name": "Paint",
            "material_id": WHITE,
            "face_indices": [0, 1, 2],
        },
    ]

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        acknowledge_policy_fallback=True,
    )

    assignment = plan["assignments"][0]
    assert [
        (subset["subset_name"], subset["material_id"])
        for subset in assignment["face_subsets"]
    ] == [("Paint", WHITE), ("Steel", GREEN)]
    assert assignment["provenance"][
        "source_material_bind_subset_contract"
    ]["action"] == "explicit_topology_match"
    assert (
        report["summary"]["source_subset_explicit_topology_match_part_count"]
        == 1
    )
    assert report["source_material_bind_subsets"][
        "explicit_topology_match_part_ids"
    ] == ["P0001"]


def test_mismatched_explicit_source_subsets_collapse_to_parent_material() -> None:
    registry, staged, gate, whitelist = _documents()
    registry["parts"][0] = _with_source_subsets(
        registry["parts"][0],
        [("Paint", [0, 1, 2]), ("Steel", [3, 4, 5])],
    )
    staged["material_plan"]["assignments"][0]["face_subsets"] = [
        {
            "subset_name": "Paint",
            "material_id": WHITE,
            "face_indices": [0, 1],
        }
    ]
    staged["material_plan"]["assignments"][0][
        "preserve_parent_material_binding"
    ] = True

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        acknowledge_policy_fallback=True,
    )

    assignment = plan["assignments"][0]
    assert "preserve_parent_material_binding" not in assignment
    assert {
        subset["material_id"] for subset in assignment["face_subsets"]
    } == {GREEN}
    assert assignment["provenance"][
        "source_material_bind_subset_contract"
    ]["action"] == "explicit_topology_mismatch_collapse_to_parent"
    assert (
        "SOURCE_FACE_SUBSET_TOPOLOGY_MISMATCH_COLLAPSED_TO_PARENT"
        in assignment["provenance"]["reason_codes"]
    )
    assert (
        report["summary"]["source_subset_explicit_topology_collapse_part_count"]
        == 1
    )
    assert report["source_material_bind_subsets"][
        "explicit_topology_collapse_part_ids"
    ] == ["P0001"]


def test_tampered_registry_source_subset_hash_fails_closed() -> None:
    registry, staged, gate, whitelist = _documents()
    registry["parts"][0] = _with_source_subsets(
        registry["parts"][0],
        [("Paint", [0, 1, 2])],
    )
    registry["parts"][0][SOURCE_MATERIAL_BIND_SUBSETS_FIELD][0][
        "face_indices"
    ] = [0, 1]

    with pytest.raises(PolicyExactCoverError, match="source subset hash is invalid"):
        build_policy_exact_cover(
            registry=registry,
            staged_result=staged,
            confidence_gate=gate,
            whitelist=whitelist,
            acknowledge_policy_fallback=True,
        )


def test_policy_cover_rejects_non_whitelisted_candidate() -> None:
    registry, staged, gate, whitelist = _documents()
    staged["material_plan"]["assignments"][0]["material_id"] = "mdl:outside"
    with pytest.raises(PolicyExactCoverError, match="outside industrial whitelist"):
        build_policy_exact_cover(
            registry=registry,
            staged_result=staged,
            confidence_gate=gate,
            whitelist=whitelist,
            acknowledge_policy_fallback=True,
        )


def test_review_candidates_never_propagate_even_with_legacy_cluster_keys() -> None:
    registry, staged, gate, whitelist = _documents()
    registry["parts"] = [
        _part("P0001", "/A/SameLeaf"),
        _part("P0002", "/B/SameLeaf"),
        _part("P0003", "/C/SameLeaf"),
    ]
    registry["part_count"] = 3
    staged["material_plan"]["assignments"] = [
        {
            "part_id": "P0001",
            "material_id": GREEN,
            "semantic": "candidate one",
            "confidence": 0.6,
            "evidence_views": ["front"],
            "status": "review",
        },
        {
            "part_id": "P0002",
            "material_id": WHITE,
            "semantic": "candidate two",
            "confidence": 0.6,
            "evidence_views": ["side"],
            "status": "review",
        },
    ]
    gate["decisions"] = [
        {"part_id": f"P{index:04d}", "decision": "preserve"} for index in range(1, 4)
    ]
    policy = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "candidate_auto_cluster_keys": ["parent_path", "cad_leaf"],
        "review_cluster_keys": ["parent_path", "cad_leaf"],
        "default_strategy": "declared_material",
        "default_material_id": STAINLESS_STEEL_MATTE,
    }

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        policy=policy,
        acknowledge_policy_fallback=True,
    )

    by_id = {item["part_id"]: item for item in plan["assignments"]}
    assert by_id["P0003"]["material_id"] == STAINLESS_STEEL_MATTE
    assert all(item["material_id"] == STAINLESS_STEEL_MATTE for item in by_id.values())
    assert all(
        item["provenance"]["tier"] == "neutral_default" for item in by_id.values()
    )
    assert report["conflicts"] == []
    assert report["identity_propagation"]["applied_cluster_keys"] == []
    assert report["identity_propagation"]["ignored_unsafe_cluster_keys"] == [
        "cad_leaf",
        "parent_path",
    ]


def test_review_mapping_remains_provisional_and_cannot_seed_tournament() -> None:
    registry, staged, gate, whitelist = _documents()
    shared_binding = "/Asset/OccurrencePrototype/Looks/Diffuse_7"
    registry["parts"][1]["existing_visual_material"] = shared_binding
    registry["parts"][2]["existing_visual_material"] = shared_binding
    gate["policy"] = {"review_mapping_confidence": 0.6}
    gate["decisions"][2] = {
        "part_id": "P0003",
        "decision": "review",
        "mapping": {
            "batch_id": "B01",
            "group_id": "G07",
            "status": "review",
            "confidence": 0.6,
            "reason_code": "direct_visual_match",
        },
        "model": {
            "independent_reference_count": 1,
            "independent_reference_views": ["side"],
            "reference_evidence_source": "independent_view_predictions",
        },
        "threshold_profile": "strict",
        "material_choice": {
            "resolved_material_id": WHITE,
            "confirmation_basis": "forward_reverse_disagreement",
            "confirmed": False,
        },
        "reason_codes": ["MAPPING_BELOW_AUTO"],
    }
    palette_fusion = {
        "schema_version": "qwen-multiview-palette-fusion/v1",
        "canonical_palette": {
            "schema_version": "qwen-canonical-material-palette/v1",
            "groups": [
                {
                    "group_id": "G07",
                    "family_hint": "metal",
                    "base_color": "orange",
                    "finish_hint": "painted",
                    "confidence": 0.9,
                    "source_view_ids": ["front", "side"],
                    "distinct_view_count": 2,
                    "singleton": False,
                }
            ],
        },
    }

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        palette_fusion=palette_fusion,
        acknowledge_policy_fallback=True,
    )

    by_id = {item["part_id"]: item for item in plan["assignments"]}
    direct = by_id["P0003"]
    propagated = by_id["P0002"]
    for assignment in (direct, propagated):
        assert assignment["status"] == FALLBACK_STATUS
        assert assignment["confidence"] == 0.0
        assert assignment["evidence_views"] == []
        assert assignment["material_id"] == STAINLESS_STEEL_MATTE
        assert "canonical_group_id" not in assignment["provenance"]
        assert "target_mapping" not in assignment["provenance"]
        assert (
            assignment["provenance"]["provisional_canonical_group_id"]
            == "G07"
        )
        assert (
            assignment["provenance"]["provisional_target_mapping"][
                "authoring_eligible"
            ]
            is False
        )
        assert (
            assignment["provenance"]["provisional_target_mapping"][
                "downstream_target_eligible"
            ]
            is False
        )
    assert direct["material_id"] != WHITE
    assert direct["provenance"]["provisional_target_mapping"]["mode"] == (
        "confidence_gate_review_hypothesis/v2"
    )
    assert direct["provenance"]["evidence_lineage"]["staged_candidate"][
        "material_id"
    ] == WHITE
    assert propagated["provenance"]["provisional_target_mapping"]["mode"] == (
        "exact_identity_review_hypothesis_propagation/v2"
    )
    assert propagated["provenance"]["provisional_target_mapping"][
        "source_part_ids"
    ] == ["P0003"]
    assert propagated["provenance"]["evidence_lineage"]["propagated_fields"] == [
        "provisional_canonical_group_id"
    ]
    assert "canonical_group_id" in propagated["provenance"]["evidence_lineage"][
        "excluded_fields"
    ]
    assert "material_id" in propagated["provenance"]["evidence_lineage"][
        "excluded_fields"
    ]
    # Downstream annotation and multigroup planning discover targets only from
    # authoritative canonical_group_id fields; REVIEW lineage contributes none.
    assert all(
        "canonical_group_id" not in assignment["provenance"]
        for assignment in plan["assignments"]
    )
    lineage = report["provisional_group_hypotheses"]
    assert lineage["direct_review_hypotheses"]["part_ids_by_group"] == {
        "G07": ["P0003"]
    }
    assert lineage["exact_identity_hypothesis_expansion"]["part_ids_by_group"] == {
        "G07": ["P0002"]
    }
    assert lineage["direct_review_hypotheses"]["downstream_target_eligible"] is False
    assert (
        lineage["exact_identity_hypothesis_expansion"][
            "downstream_target_eligible"
        ]
        is False
    )


def test_provisional_review_lineage_changes_only_provenance() -> None:
    assignment = {
        "part_id": "P0001",
        "material_id": STAINLESS_STEEL_MATTE,
        "semantic": "neutral fallback",
        "confidence": 0.0,
        "evidence_views": [],
        "status": FALLBACK_STATUS,
        "parameters": {"roughness": 0.42},
        "provenance": {
            "tier": "neutral_default",
            "reason_codes": ["POLICY_DECLARED_NEUTRAL_DEFAULT"],
            "output_confidence_basis": "policy fallback; not evidence confidence",
            "sources": [],
        },
    }
    output = {"P0001": copy.deepcopy(assignment)}
    audit = policy_module._retain_non_authoring_mapping_lineage(
        output=output,
        candidates={
            "P0001": {
                "part_id": "P0001",
                "material_id": WHITE,
                "semantic": "weak white hypothesis",
                "confidence": 0.7,
                "evidence_views": ["front"],
                "status": "review",
            }
        },
        gate_records={
            "P0001": {
                "part_id": "P0001",
                "decision": "review",
                "mapping": {
                    "batch_id": "B01",
                    "group_id": "G02",
                    "status": "review",
                    "confidence": 0.7,
                    "reason_code": "semantic_review",
                },
                "model": {},
                "material_choice": {},
                "reason_codes": ["MAPPING_BELOW_AUTO"],
            }
        },
        confidence_gate={"policy": {"review_mapping_confidence": 0.6}},
        palette_fusion={
            "schema_version": "qwen-multiview-palette-fusion/v1",
            "canonical_palette": {
                "schema_version": "qwen-canonical-material-palette/v1",
                "groups": [
                    {
                        "group_id": "G02",
                        "family_hint": "metal",
                        "base_color": "white",
                        "confidence": 0.9,
                        "source_view_ids": ["front", "side"],
                        "distinct_view_count": 2,
                        "singleton": False,
                    }
                ],
            },
        },
    )

    for field in (
        "part_id",
        "material_id",
        "semantic",
        "confidence",
        "evidence_views",
        "status",
        "parameters",
    ):
        assert output["P0001"][field] == assignment[field]
    assert "canonical_group_id" not in output["P0001"]["provenance"]
    assert output["P0001"]["provenance"]["provisional_canonical_group_id"] == "G02"
    assert audit["authoring_effect"] == "none"
    assert audit["downstream_target_eligible"] is False


def test_single_view_review_target_is_not_retained_or_propagated() -> None:
    registry, staged, gate, whitelist = _documents()
    shared_binding = "/Asset/OccurrencePrototype/Looks/Diffuse_3"
    registry["parts"][1]["existing_visual_material"] = shared_binding
    registry["parts"][2]["existing_visual_material"] = shared_binding
    gate["decisions"][2] = {
        "part_id": "P0003",
        "decision": "review",
        "mapping": {
            "batch_id": "B01",
            "group_id": "G03",
            "status": "review",
            "confidence": 0.9,
            "reason_code": "direct_visual_match",
        },
    }
    palette_fusion = {
        "schema_version": "qwen-multiview-palette-fusion/v1",
        "canonical_palette": {
            "schema_version": "qwen-canonical-material-palette/v1",
            "groups": [
                {
                    "group_id": "G03",
                    "family_hint": "metal",
                    "base_color": "white",
                    "finish_hint": "painted",
                    "confidence": 0.9,
                    "source_view_ids": ["front"],
                    "distinct_view_count": 1,
                    "singleton": True,
                }
            ],
        },
    }

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        palette_fusion=palette_fusion,
        acknowledge_policy_fallback=True,
    )

    by_id = {item["part_id"]: item for item in plan["assignments"]}
    assert "canonical_group_id" not in by_id["P0003"]["provenance"]
    assert "canonical_group_id" not in by_id["P0002"]["provenance"]
    lineage = report["provisional_group_hypotheses"]
    assert lineage["direct_review_hypotheses"]["retained_part_count"] == 0
    assert (
        lineage["exact_identity_hypothesis_expansion"]["propagated_part_count"]
        == 0
    )
    assert lineage["direct_review_hypotheses"]["rejection_counts"][
        "TARGET_CANONICAL_GROUP_NOT_MULTIVIEW"
    ] == 1


def test_gate_rejected_staged_auto_cannot_assign_or_seed_exact_propagation() -> None:
    registry, staged, gate, whitelist = _documents()
    registry["parts"] = [
        _part("P0001", "/Asset/First", geometry_fingerprint="mesh-sha256:same"),
        _part("P0002", "/Asset/Second", geometry_fingerprint="mesh-sha256:same"),
    ]
    registry["part_count"] = 2
    staged["material_plan"]["assignments"] = [
        {
            "part_id": "P0001",
            "material_id": GREEN,
            "semantic": "weak green candidate",
            "confidence": 0.95,
            "evidence_views": ["front", "iso"],
            "status": "auto",
        }
    ]
    gate["decisions"] = [
        {"part_id": part_id, "decision": "preserve"} for part_id in ("P0001", "P0002")
    ]

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        acknowledge_policy_fallback=True,
    )

    assert {item["material_id"] for item in plan["assignments"]} == {
        STAINLESS_STEEL_MATTE
    }
    assert {item["provenance"]["tier"] for item in plan["assignments"]} == {
        "neutral_default"
    }
    assert report["summary"]["confidence_gate_auto_count"] == 0
    assert report["identity_propagation"]["trusted_source_part_ids"] == []


def test_gate_auto_may_propagate_across_exact_geometry_identity() -> None:
    registry, staged, gate, whitelist = _documents()
    registry["parts"] = [
        _part("P0001", "/Asset/First", geometry_fingerprint="mesh-sha256:same"),
        _part("P0002", "/Asset/Second", geometry_fingerprint="mesh-sha256:same"),
        _part("P0003", "/Asset/Third", geometry_fingerprint="mesh-sha256:other"),
    ]
    registry["part_count"] = 3
    staged["material_plan"]["assignments"] = [
        {
            "part_id": "P0001",
            "material_id": GREEN,
            "semantic": "verified green component",
            "confidence": 0.95,
            "evidence_views": ["front", "iso"],
            "status": "auto",
        }
    ]
    gate["decisions"] = [
        {
            "part_id": part_id,
            "decision": "auto" if part_id == "P0001" else "preserve",
        }
        for part_id in ("P0001", "P0002", "P0003")
    ]

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        policy={
            "schema_version": POLICY_SCHEMA_VERSION,
            "candidate_auto_cluster_keys": ["geometry_fingerprint"],
        },
        acknowledge_policy_fallback=True,
    )

    by_id = {item["part_id"]: item for item in plan["assignments"]}
    assert by_id["P0001"]["status"] == "auto"
    propagated = by_id["P0002"]
    assert propagated["status"] == FALLBACK_STATUS
    assert propagated["material_id"] == GREEN
    assert propagated["confidence"] == 0.0
    assert propagated["provenance"]["tier"] == ("trusted_exact_geometry_fingerprint")
    assert [source["part_id"] for source in propagated["provenance"]["sources"]] == [
        "P0001"
    ]
    assert by_id["P0003"]["material_id"] == STAINLESS_STEEL_MATTE
    assert by_id["P0003"]["provenance"]["tier"] == "neutral_default"
    assert report["identity_propagation"]["applied_cluster_keys"] == [
        "geometry_fingerprint",
    ]


def test_gate_auto_propagates_only_within_full_authored_material_binding() -> None:
    registry, staged, gate, whitelist = _documents()
    shared_binding = "/Asset/Housing/Looks/Diffuse_1"
    registry["parts"] = [
        _part("P0001", "/Asset/Housing/Main", existing_material=shared_binding),
        _part("P0002", "/Asset/Housing/Hidden", existing_material=shared_binding),
        _part(
            "P0003",
            "/Asset/Other",
            existing_material="/Asset/Other/Looks/Diffuse_1",
        ),
        _part(
            "P0004",
            "/Asset/GBT97_WASHER",
            existing_material=shared_binding,
        ),
    ]
    registry["part_count"] = 4
    staged["material_plan"]["assignments"] = [
        {
            "part_id": "P0001",
            "material_id": GREEN,
            "semantic": "verified green housing",
            "confidence": 0.95,
            "evidence_views": ["front", "side"],
            "status": "auto",
        }
    ]
    gate["decisions"] = [
        {
            "part_id": part_id,
            "decision": "auto" if part_id == "P0001" else "preserve",
        }
        for part_id in ("P0001", "P0002", "P0003", "P0004")
    ]

    plan, _report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        acknowledge_policy_fallback=True,
    )

    by_id = {item["part_id"]: item for item in plan["assignments"]}
    assert by_id["P0002"]["material_id"] == GREEN
    assert by_id["P0002"]["provenance"]["tier"] == (
        "trusted_authored_material_binding"
    )
    assert by_id["P0003"]["material_id"] == STAINLESS_STEEL_MATTE
    assert by_id["P0003"]["apply_action"] == SOURCE_VISUAL_PRESERVE_ACTION
    assert by_id["P0004"]["material_id"] == GREEN
    assert by_id["P0004"]["provenance"]["tier"] == (
        "trusted_authored_material_binding"
    )


def test_exact_geometry_propagation_preserves_trusted_base_parameters() -> None:
    registry, staged, gate, whitelist = _documents()
    registry["parts"] = [
        _part("P0001", "/Asset/First", geometry_fingerprint="mesh-sha256:same"),
        _part("P0002", "/Asset/Second", geometry_fingerprint="mesh-sha256:same"),
    ]
    registry["part_count"] = 2
    staged["material_plan"]["assignments"] = []
    gate["decisions"] = [
        {"part_id": part_id, "decision": "preserve"} for part_id in ("P0001", "P0002")
    ]
    base_plan = {
        "schema_version": "1.0",
        "assignments": [
            {
                "part_id": "P0001",
                "material_id": WHITE,
                "semantic": "verified white coating",
                "confidence": 0.92,
                "evidence_views": ["front", "side"],
                "status": "auto",
                "parameters": {"paint_roughness": 0.33},
            }
        ],
    }

    plan, _report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        base_plan=base_plan,
        policy={
            "schema_version": POLICY_SCHEMA_VERSION,
            "candidate_auto_cluster_keys": ["geometry_fingerprint"],
        },
        acknowledge_policy_fallback=True,
    )

    by_id = {item["part_id"]: item for item in plan["assignments"]}
    assert by_id["P0002"]["material_id"] == WHITE
    assert by_id["P0002"]["parameters"] == {"paint_roughness": 0.33}
    assert by_id["P0002"]["provenance"]["tier"] == (
        "trusted_exact_geometry_fingerprint"
    )


def test_legacy_dominant_default_is_accepted_but_never_executed() -> None:
    registry, staged, gate, whitelist = _documents()
    policy = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "default_strategy": "dominant_staged_auto",
        "default_material_id": STAINLESS_STEEL_MATTE,
    }

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        policy=policy,
        acknowledge_policy_fallback=True,
    )

    by_id = {item["part_id"]: item for item in plan["assignments"]}
    assert by_id["P0002"]["material_id"] == STAINLESS_STEEL_MATTE
    assert by_id["P0002"]["provenance"]["reason_codes"] == [
        "POLICY_DOMINANT_DEFAULT_DISABLED_NEUTRAL_USED"
    ]
    assert report["requested_default_strategy"] == "dominant_staged_auto"
    assert report["effective_default_strategy"] == "declared_material"


def test_valid_source_material_is_hash_bound_noop_before_neutral_fallback() -> None:
    registry, staged, gate, whitelist = _documents()
    registry["parts"][4]["existing_visual_material"] = "/Asset/Looks/OriginalBlue"

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        acknowledge_policy_fallback=True,
    )

    by_id = {item["part_id"]: item for item in plan["assignments"]}
    fallback = by_id["P0005"]
    assert fallback["material_id"] == STAINLESS_STEEL_MATTE
    assert fallback["apply_action"] == SOURCE_VISUAL_PRESERVE_ACTION
    assert fallback["provenance"]["tier"] == SOURCE_VISUAL_PRESERVE_TIER
    assert fallback["source_visual_material_prim_path"] == (
        "/Asset/Looks/OriginalBlue"
    )
    assert fallback["source_visual_material_binding_sha256"] == (
        source_visual_binding_sha256(
            part_id="P0005",
            prim_path="/Asset/OccludedCustom/Mesh",
            material_prim_path="/Asset/Looks/OriginalBlue",
        )
    )
    assert fallback["provenance"]["reason_codes"] == [
        "SOURCE_VISUAL_MATERIAL_PRESENT",
        "SOURCE_VISUAL_BINDING_HASH_BOUND",
        "PRESERVE_SOURCE_VISUAL_NOOP",
    ]
    assert report["summary"]["neutral_default_count"] == 2
    assert report["summary"]["source_visual_preserve_count"] == 1
    assert (
        report["summary"]["source_preserve_unavailable_neutral_fallback_count"]
        == 0
    )


def test_reference_photo_policy_neutralizes_unverified_source_display_colors() -> None:
    registry, staged, gate, whitelist = _documents()
    registry["parts"][3]["existing_visual_material"] = (
        "/Asset/Looks/DisplayYellowFastener"
    )
    registry["parts"][4]["existing_visual_material"] = (
        "/Asset/Looks/DisplayBlue"
    )

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        policy={
            "schema_version": POLICY_SCHEMA_VERSION,
            "source_visual_strategy": "neutralize_unverified",
        },
        acknowledge_policy_fallback=True,
    )

    by_id = {item["part_id"]: item for item in plan["assignments"]}
    assert by_id["P0004"]["provenance"]["tier"] == "semantic_rule"
    neutralized = by_id["P0005"]
    assert neutralized["material_id"] == STAINLESS_STEEL_MATTE
    assert (
        neutralized["provenance"]["tier"]
        == "source_preserve_unavailable_neutral_fallback"
    )
    assert "UNVERIFIED_SOURCE_VISUAL_MATERIAL_IGNORED" in (
        neutralized["provenance"]["reason_codes"]
    )
    assert "apply_action" not in neutralized
    assert "source_visual_material_prim_path" not in neutralized
    assert "source_visual_material_binding_sha256" not in neutralized
    assert report["source_visual_strategy"] == "neutralize_unverified"
    assert report["summary"]["source_visual_preserve_count"] == 0
    assert (
        report["summary"][
            "source_preserve_unavailable_neutral_fallback_count"
        ]
        == 1
    )
    assert report["summary"]["neutral_default_count"] == 2


def test_visual_similarity_policy_can_disable_name_based_material_rules() -> None:
    registry, staged, gate, whitelist = _documents()

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        policy={
            "schema_version": POLICY_SCHEMA_VERSION,
            "source_visual_strategy": "neutralize_unverified",
            "semantic_rules": [],
        },
        acknowledge_policy_fallback=True,
    )

    by_id = {item["part_id"]: item for item in plan["assignments"]}
    # P0004 is intentionally named like a standard fastener in the fixture.
    # With visual-only fallback enabled, its name is not appearance evidence.
    assert by_id["P0004"]["material_id"] == STAINLESS_STEEL_MATTE
    assert by_id["P0004"]["provenance"]["tier"] == "neutral_default"
    assert report["policy"]["semantic_rules"] == []
    assert all(
        item["provenance"]["tier"] != "semantic_rule"
        for item in plan["assignments"]
    )


def _source_accent_bundle(
    *,
    accent_count: int = 4,
    repeated_geometry: bool = True,
) -> tuple[dict, dict, dict, dict, dict]:
    _registry, _staged, _gate, whitelist = _documents()
    parts: list[dict] = []
    for index in range(80):
        part_id = f"P{index + 1:04d}"
        is_accent = index < accent_count
        point_count = 2780 if repeated_geometry or not is_accent else 2780 + index
        part = _part(part_id, f"/Asset/Assembly_{index:03d}")
        part.update(
            {
                "point_count": point_count,
                "face_count": 2738,
                "world_bbox": [
                    [float(index * 10), 0.0, 0.0],
                    [float(index * 10 + 1), 2.0, 3.0],
                ],
            }
        )
        if is_accent:
            part["existing_visual_material"] = (
                f"/Asset/Assembly_{index:03d}/Looks/AccentBlue"
            )
            part["existing_visual_material_properties"] = {
                "shader_path": (
                    f"/Asset/Assembly_{index:03d}/Looks/AccentBlue/PreviewSurface"
                ),
                "shader_id": "UsdPreviewSurface",
                "diffuseColor": [0.025151724, 0.420533568, 0.949308753],
                "metallic": 0.5,
                "roughness": 0.5,
                "opacity": 1.0,
            }
        parts.append(part)
    registry = {
        "schema_version": "qwen-material-parts/v1",
        "asset_sha256": "b" * 64,
        "part_count": len(parts),
        "parts": parts,
    }
    staged = {
        "schema_version": "qwen-staged-material-result/v1",
        "material_plan": {"schema_version": "1.0", "assignments": []},
    }
    gate = {
        "schema_version": "qwen-material-confidence-gate/v1",
        "decisions": [
            {"part_id": part["part_id"], "decision": "preserve"} for part in parts
        ],
    }
    palette = {
        "schema_version": "qwen-multiview-palette-fusion/v1",
        "canonical_palette": {
            "schema_version": "qwen-canonical-material-palette/v1",
            "groups": [
                {
                    "group_id": "G01",
                    "family_hint": "plastic",
                    "base_color": "blue",
                    "finish_hint": "painted",
                    "source_view_ids": ["front", "top"],
                    "distinct_view_count": 2,
                    "singleton": False,
                }
            ],
        },
    }
    return registry, staged, gate, whitelist, palette


def _verified_unresolved_blue_palette() -> dict:
    description = (
        "connected blue chromatic region detected from pixels; "
        "physical material unresolved"
    )
    palettes = []
    audits = []
    for view_id, local_group_id, image_sha256, box in (
        ("front", "G05", "1" * 64, [10, 10, 30, 30]),
        ("top", "G09", "2" * 64, [40, 40, 60, 60]),
    ):
        group = {
            "group_id": local_group_id,
            "family_hint": "other",
            "base_color": "blue",
            "finish_hint": "other",
            "visual_description": description,
            "confidence": 0.6,
            "boxes": [box],
        }
        palettes.append(
            {
                "schema_version": "qwen-material-palette/v1",
                "source_view_id": view_id,
                "groups": [group],
            }
        )
        audits.append(
            {
                "schema_version": "qwen-palette-accent-augmentation/v1",
                "source_view_id": view_id,
                "image_sha256": image_sha256,
                "added_group_ids": [local_group_id],
                "components": [
                    {
                        "base_color": "blue",
                        "decision": "added",
                        "accepted_components": [{"box": box}],
                    }
                ],
            }
        )
    return fuse_multiview_palettes(
        palettes,
        augmentation_audits=audits,
    )


def test_reference_corroborated_rare_repeated_source_accent_is_preserved() -> None:
    registry, staged, gate, whitelist, palette = _source_accent_bundle()

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        palette_fusion=palette,
        policy={
            "schema_version": POLICY_SCHEMA_VERSION,
            "source_visual_strategy": "neutralize_unverified",
        },
        acknowledge_policy_fallback=True,
    )

    by_id = {item["part_id"]: item for item in plan["assignments"]}
    expected = {f"P{index:04d}" for index in range(1, 5)}
    preserved = {
        part_id
        for part_id, assignment in by_id.items()
        if assignment["provenance"]["tier"] == SOURCE_VISUAL_PRESERVE_TIER
    }
    assert preserved == expected
    assert report["summary"]["source_visual_preserve_count"] == 4
    assert report["summary"]["corroborated_source_visual_preserve_count"] == 4
    assert report["corroborated_source_visual"]["eligible_part_ids"] == sorted(
        expected
    )
    assert report["corroborated_source_visual"]["maximum_source_signature_count"] == 4
    assert plan["provenance"]["palette_fusion_sha256"] == (
        policy_module._canonical_sha256(palette)
    )
    for part_id in expected:
        assignment = by_id[part_id]
        assert assignment["apply_action"] == SOURCE_VISUAL_PRESERVE_ACTION
        assert {
            "REFERENCE_PALETTE_MULTIVIEW_COLOR_CORROBORATION",
            "RARE_SOURCE_VISUAL_SIGNATURE",
            "REPEATED_GEOMETRY_SOURCE_LOCATOR",
        }.issubset(assignment["provenance"]["reason_codes"])
        evidence = assignment["provenance"]["source_visual_corroboration"]
        assert evidence["canonical_group_id"] == "G01"
        assert evidence["geometry_repeat_count"] == 4


def test_multiview_unresolved_pixel_color_can_locate_repeated_source_accent() -> None:
    registry, staged, gate, whitelist, _palette = _source_accent_bundle()
    palette = _verified_unresolved_blue_palette()

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        palette_fusion=palette,
        policy={
            "schema_version": POLICY_SCHEMA_VERSION,
            "source_visual_strategy": "neutralize_unverified",
        },
        acknowledge_policy_fallback=True,
    )

    expected = {f"P{index:04d}" for index in range(1, 5)}
    assert set(report["corroborated_source_visual"]["eligible_part_ids"]) == expected
    for assignment in plan["assignments"]:
        if assignment["part_id"] not in expected:
            continue
        evidence = assignment["provenance"]["source_visual_corroboration"]
        assert evidence["canonical_group_association_basis"] == (
            "identical_unresolved_pixel_chromatic_multiview"
        )


def test_unresolved_pixel_source_accent_contract_is_fail_closed() -> None:
    registry, staged, gate, whitelist, _palette = _source_accent_bundle()
    palette = _verified_unresolved_blue_palette()
    group = palette["canonical_palette"]["groups"][0]
    group["sources"][1]["visual_description"] = (
        "different unresolved blue observation"
    )

    _plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        palette_fusion=palette,
        policy={
            "schema_version": POLICY_SCHEMA_VERSION,
            "source_visual_strategy": "neutralize_unverified",
        },
        acknowledge_policy_fallback=True,
    )

    assert report["corroborated_source_visual"]["eligible_part_ids"] == []


def test_confirmed_nvidia_mdl_replaces_corroborated_source_accent() -> None:
    registry, staged, gate, whitelist, palette = _source_accent_bundle()
    blue_polycarbonate = (
        "mdl:vMaterials_2/Plastic/Polycarbonate_Opaque.mdl#Polycarbonate_Blue"
    )
    whitelist["material_ids"].append(blue_polycarbonate)

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        palette_fusion=palette,
        group_materials={
            "schema_version": "qwen-palette-material/v1",
            "selections": [
                    {
                        "group_id": "G01",
                        "material_id": blue_polycarbonate,
                        "confirmed": True,
                        "confidence": 0.70,
                    }
            ],
        },
        policy={
            "schema_version": POLICY_SCHEMA_VERSION,
            "source_visual_strategy": "neutralize_unverified",
        },
        acknowledge_policy_fallback=True,
    )

    by_id = {item["part_id"]: item for item in plan["assignments"]}
    replaced = {f"P{index:04d}" for index in range(1, 5)}
    for part_id in replaced:
        assignment = by_id[part_id]
        assert assignment["material_id"] == blue_polycarbonate
        assert assignment["provenance"]["tier"] == (
            "corroborated_source_visual_nvidia_mdl"
        )
        assert assignment["provenance"]["canonical_group_id"] == "G01"
        assert assignment["provenance"]["supporting_view_ids"] == [
            "front",
            "top",
        ]
        assert "apply_action" not in assignment
        corroboration = assignment["provenance"]["source_visual_corroboration"]
        assert corroboration["canonical_group_id"] == "G01"
        assert corroboration["confirmed_material_id"] == blue_polycarbonate
    assert report["summary"]["source_visual_preserve_count"] == 0
    assert report["summary"]["corroborated_source_visual_preserve_count"] == 0
    assert report["summary"]["corroborated_source_visual_nvidia_mdl_count"] == 4
    assert report["corroborated_source_visual"][
        "nvidia_mdl_replacement_part_ids"
    ] == sorted(replaced)


def test_immutable_high_confidence_mdl_replaces_corroborated_source_accent() -> None:
    registry, staged, gate, whitelist, palette = _source_accent_bundle()
    blue_polypropylene = (
        "mdl:vMaterials_2/Plastic/Polypropylene_Opaque.mdl"
        "#Polypropylene_Light_Blue"
    )
    whitelist["material_ids"].append(blue_polypropylene)

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        palette_fusion=palette,
        group_materials={
            "schema_version": "qwen-palette-material/v1",
            "selections": [
                {
                    "group_id": "G01",
                    "material_id": blue_polypropylene,
                    "confidence": 0.98,
                    "confirmed": False,
                }
            ],
        },
        policy={
            "schema_version": POLICY_SCHEMA_VERSION,
            "source_visual_strategy": "neutralize_unverified",
        },
        acknowledge_policy_fallback=True,
        immutable_mdl_after_selection=True,
    )

    by_id = {item["part_id"]: item for item in plan["assignments"]}
    replaced = {f"P{index:04d}" for index in range(1, 5)}
    for part_id in replaced:
        assignment = by_id[part_id]
        assert assignment["material_id"] == blue_polypropylene
        assert assignment["provenance"]["tier"] == (
            "corroborated_source_visual_nvidia_mdl"
        )
        assert assignment["provenance"]["canonical_group_id"] == "G01"
        assert "parameters" not in assignment
        corroboration = assignment["provenance"]["source_visual_corroboration"]
        assert corroboration["selected_material_id"] == blue_polypropylene
        assert corroboration["material_selection_basis"] == (
            "high_confidence_whitelist_candidate_pending_render_qa"
        )
        assert corroboration["selection_confidence"] == pytest.approx(0.98)
        assert "QA_POST_RENDER_VALIDATION_REQUIRED" in (
            assignment["provenance"]["reason_codes"]
        )
    assert report["summary"][
        "corroborated_source_visual_provisional_nvidia_mdl_count"
    ] == 4
    assert report["summary"]["selected_mdl_library_defaults_locked"] is True


def test_repeated_source_accent_anchor_expands_to_exact_signature_members() -> None:
    registry, staged, gate, whitelist, palette = _source_accent_bundle()
    # Grow the registry so a six-part signature remains within the 5% rarity
    # bound.  The first four parts retain the repeated geometry anchor while
    # two differently shaped caps share its exact authored appearance.
    for index in range(80, 120):
        part_id = f"P{index + 1:04d}"
        part = _part(part_id, f"/Asset/Assembly_{index:03d}")
        part.update(
            {
                "point_count": 5000 + index,
                "face_count": 4900 + index,
                "world_bbox": [
                    [float(index * 10), 0.0, 0.0],
                    [float(index * 10 + 1), 2.0, 3.0],
                ],
            }
        )
        registry["parts"].append(part)
        gate["decisions"].append({"part_id": part_id, "decision": "preserve"})
    for index in (4, 5):
        part = registry["parts"][index]
        part["point_count"] = 4000 + index
        part["face_count"] = 3900 + index
        part["existing_visual_material"] = (
            f"/Asset/Assembly_{index:03d}/Looks/AccentBlue"
        )
        part["existing_visual_material_properties"] = copy.deepcopy(
            registry["parts"][0]["existing_visual_material_properties"]
        )
        part["existing_visual_material_properties"]["shader_path"] = (
            f"/Asset/Assembly_{index:03d}/Looks/AccentBlue/PreviewSurface"
        )
    registry["part_count"] = len(registry["parts"])

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        palette_fusion=palette,
        policy={
            "schema_version": POLICY_SCHEMA_VERSION,
            "source_visual_strategy": "neutralize_unverified",
        },
        acknowledge_policy_fallback=True,
    )

    expected = {f"P{index:04d}" for index in range(1, 7)}
    by_id = {item["part_id"]: item for item in plan["assignments"]}
    assert set(report["corroborated_source_visual"]["eligible_part_ids"]) == expected
    for part_id in ("P0005", "P0006"):
        evidence = by_id[part_id]["provenance"]["source_visual_corroboration"]
        assert evidence["signature_expansion_basis"] == (
            "exact_source_signature_with_repeated_geometry_anchor"
        )
        assert evidence["signature_expansion_anchor_part_ids"] == [
            "P0001",
            "P0002",
            "P0003",
            "P0004",
        ]
        assert "EXACT_SOURCE_SIGNATURE_COHORT_EXPANSION" in (
            by_id[part_id]["provenance"]["reason_codes"]
        )


@pytest.mark.parametrize(
    ("accent_count", "repeated_geometry", "expected_rejection"),
    [
        (5, True, "SOURCE_SIGNATURE_TOO_COMMON"),
        (4, False, "NO_REPEATED_GEOMETRY_COHORT"),
    ],
)
def test_source_accent_preservation_fails_closed_outside_strict_bounds(
    accent_count: int,
    repeated_geometry: bool,
    expected_rejection: str,
) -> None:
    registry, staged, gate, whitelist, palette = _source_accent_bundle(
        accent_count=accent_count,
        repeated_geometry=repeated_geometry,
    )

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        palette_fusion=palette,
        policy={
            "schema_version": POLICY_SCHEMA_VERSION,
            "source_visual_strategy": "neutralize_unverified",
        },
        acknowledge_policy_fallback=True,
    )

    assert report["summary"]["source_visual_preserve_count"] == 0
    assert (
        report["corroborated_source_visual"]["rejection_counts"][
            expected_rejection
        ]
        == accent_count
    )
    for assignment in plan["assignments"][:accent_count]:
        assert assignment["provenance"]["tier"] == (
            "source_preserve_unavailable_neutral_fallback"
        )


def test_invalid_source_visual_strategy_fails_closed() -> None:
    registry, staged, gate, whitelist = _documents()

    with pytest.raises(PolicyExactCoverError, match="source_visual_strategy"):
        build_policy_exact_cover(
            registry=registry,
            staged_result=staged,
            confidence_gate=gate,
            whitelist=whitelist,
            policy={
                "schema_version": POLICY_SCHEMA_VERSION,
                "source_visual_strategy": "guess",
            },
            acknowledge_policy_fallback=True,
        )


@pytest.mark.parametrize(
    "invalid_path",
    ["", "relative/Looks/Blue", "/", "/Asset//Blue", "/Asset/Blue/"],
)
def test_invalid_source_material_binding_fails_closed_to_neutral(
    invalid_path: str,
) -> None:
    registry, staged, gate, whitelist = _documents()
    registry["parts"][4]["existing_visual_material"] = invalid_path

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        acknowledge_policy_fallback=True,
    )

    assignment = {
        item["part_id"]: item for item in plan["assignments"]
    }["P0005"]
    assert assignment["provenance"]["tier"] == "neutral_default"
    assert "apply_action" not in assignment
    assert report["summary"]["source_visual_preserve_count"] == 0


def test_gate_auto_is_stronger_than_source_visual_preserve() -> None:
    registry, staged, gate, whitelist = _documents()
    registry["parts"][0]["existing_visual_material"] = "/Asset/Looks/OldGreen"

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        acknowledge_policy_fallback=True,
    )

    assignment = {
        item["part_id"]: item for item in plan["assignments"]
    }["P0001"]
    assert assignment["provenance"]["tier"] == "gate_auto"
    assert "apply_action" not in assignment
    assert report["summary"]["source_visual_preserve_count"] == 0


def test_source_visual_preserve_is_stronger_than_name_policy_rule() -> None:
    registry, staged, gate, whitelist = _documents()
    registry["parts"][3]["existing_visual_material"] = (
        "/Asset/Looks/AuthoredRedFastener"
    )

    plan, _report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        acknowledge_policy_fallback=True,
    )

    assignment = {
        item["part_id"]: item for item in plan["assignments"]
    }["P0004"]
    assert assignment["apply_action"] == SOURCE_VISUAL_PRESERVE_ACTION
    assert assignment["provenance"]["tier"] == SOURCE_VISUAL_PRESERVE_TIER
    assert assignment["source_visual_material_prim_path"] == (
        "/Asset/Looks/AuthoredRedFastener"
    )


def _mvinverse_documents(*, eligible: bool) -> tuple[dict, dict]:
    group_materials = {
        "schema_version": "qwen-palette-material/v1",
        "selections": [
            {
                "group_id": "G01",
                "material_id": GREEN,
                "confidence": 0.95,
                "confirmed": True,
            }
        ],
    }
    evidence = {
        "schema_version": "qwen-mvinverse-pbr-evidence/v1",
        "groups": [
            {
                "group_id": "G01",
                "surface_class": "dielectric",
                "contributing_view_ids": ["front", "iso"],
                "metallic": {"median": 0.12},
                "suggestion": {
                    "decision": "auto" if eligible else "preserve",
                    "auto_parameter_eligible": eligible,
                    "base_color_srgb": [0.25, 0.5, 0.125],
                    "metallic": 0.0,
                    "roughness": 0.42,
                },
            }
        ],
    }
    return group_materials, evidence


def test_eligible_mvinverse_group_parameterizes_matching_policy_assignments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, staged, gate, whitelist = _documents()
    group_materials, evidence = _mvinverse_documents(eligible=True)
    validated: list[dict] = []

    def accept_verified(document: dict) -> dict:
        validated.append(document)
        return copy.deepcopy(document)

    monkeypatch.setattr(
        policy_module, "validate_mvinverse_evidence", accept_verified
    )

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        group_materials=group_materials,
        mvinverse_pbr_evidence=evidence,
        policy={
            "schema_version": POLICY_SCHEMA_VERSION,
            "candidate_auto_cluster_keys": ["geometry_fingerprint"],
        },
        acknowledge_policy_fallback=True,
    )

    by_id = {item["part_id"]: item for item in plan["assignments"]}
    assert by_id["P0001"]["material_id"] == GREEN
    assert by_id["P0001"]["parameters"]["paint_roughness"] == pytest.approx(0.42)
    assert by_id["P0001"]["parameters"]["paint_color"] == pytest.approx(
        [0.050876, 0.214041, 0.01435], abs=1e-5
    )
    assert validated == [evidence]
    mvinverse = by_id["P0001"]["provenance"]["mvinverse"]
    assert mvinverse["group_id"] == "G01"
    assert mvinverse["contributing_view_ids"] == ["front", "iso"]
    assert mvinverse["observed_metallic"] == pytest.approx(0.12)
    assert mvinverse["authored_metallic"] == 0.0
    assert report["summary"]["mvinverse_parameterized_part_count"] == 1
    for part_id in ("P0002", "P0003", "P0005"):
        assert by_id[part_id]["material_id"] == STAINLESS_STEEL_MATTE
        assert "parameters" not in by_id[part_id]


def test_immutable_selected_mdl_mode_keeps_catalog_defaults() -> None:
    registry, staged, gate, whitelist = _documents()
    group_materials, evidence = _mvinverse_documents(eligible=True)

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        group_materials=group_materials,
        mvinverse_pbr_evidence=evidence,
        acknowledge_policy_fallback=True,
        immutable_mdl_after_selection=True,
    )

    by_id = {item["part_id"]: item for item in plan["assignments"]}
    assert by_id["P0001"]["material_id"] == GREEN
    assert all("parameters" not in assignment for assignment in plan["assignments"])
    assert plan["provenance"]["immutable_mdl_after_selection"] is True
    assert report["summary"]["mvinverse_parameterized_part_count"] == 0
    assert report["summary"]["selected_mdl_library_defaults_locked"] is True


def test_mvinverse_tunes_base_rubber_without_changing_selected_material() -> None:
    registry, staged, gate, whitelist = _documents()
    group_materials, evidence = _mvinverse_documents(eligible=True)
    group_materials["selections"][0]["material_id"] = BLACK_RUBBER
    staged["material_plan"]["assignments"][0]["material_id"] = BLACK_RUBBER

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        group_materials=group_materials,
        mvinverse_pbr_evidence=evidence,
        acknowledge_policy_fallback=True,
    )

    assignment = next(
        item for item in plan["assignments"] if item["part_id"] == "P0001"
    )
    assert assignment["material_id"] == BLACK_RUBBER
    assert set(assignment["parameters"]) == {
        "diffuse_tint",
        "metallic_constant",
        "reflection_roughness_constant",
    }
    assert report["summary"]["mvinverse_parameterized_part_count"] == 1
    assert report["mvinverse"]["skipped"] == []


def test_mvinverse_parameters_follow_a_trusted_exact_geometry_propagation() -> None:
    registry, staged, gate, whitelist = _documents()
    registry["parts"] = [
        _part("P0001", "/Asset/First", geometry_fingerprint="mesh-sha256:same"),
        _part("P0002", "/Asset/Second", geometry_fingerprint="mesh-sha256:same"),
    ]
    registry["part_count"] = 2
    staged["material_plan"]["assignments"] = [
        {
            "part_id": "P0001",
            "material_id": GREEN,
            "semantic": "verified green component",
            "confidence": 0.95,
            "evidence_views": ["front", "iso"],
            "status": "auto",
        }
    ]
    gate["decisions"] = [
        {"part_id": "P0001", "decision": "auto"},
        {"part_id": "P0002", "decision": "preserve"},
    ]
    group_materials, evidence = _mvinverse_documents(eligible=True)

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        group_materials=group_materials,
        mvinverse_pbr_evidence=evidence,
        policy={
            "schema_version": POLICY_SCHEMA_VERSION,
            "candidate_auto_cluster_keys": ["geometry_fingerprint"],
        },
        acknowledge_policy_fallback=True,
    )

    by_id = {item["part_id"]: item for item in plan["assignments"]}
    assert report["summary"]["mvinverse_parameterized_part_count"] == 2
    assert by_id["P0001"]["material_id"] == GREEN
    assert by_id["P0002"]["material_id"] == GREEN
    assert by_id["P0002"]["provenance"]["tier"] == (
        "trusted_exact_geometry_fingerprint"
    )
    assert by_id["P0002"]["provenance"]["mvinverse"]["group_id"] == "G01"


def test_mvinverse_never_parameterizes_neutral_defaults_by_material_id() -> None:
    registry, staged, gate, whitelist = _documents()
    group_materials, evidence = _mvinverse_documents(eligible=True)
    group_materials["selections"][0]["material_id"] = STAINLESS_STEEL_MATTE

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        group_materials=group_materials,
        mvinverse_pbr_evidence=evidence,
        acknowledge_policy_fallback=True,
    )

    by_id = {item["part_id"]: item for item in plan["assignments"]}
    assert report["summary"]["mvinverse_parameterized_part_count"] == 0
    for part_id in ("P0002", "P0003", "P0005"):
        assert by_id[part_id]["material_id"] == STAINLESS_STEEL_MATTE
        assert "parameters" not in by_id[part_id]


def test_ineligible_mvinverse_group_keeps_selected_mdl_without_guessing() -> None:
    registry, staged, gate, whitelist = _documents()
    group_materials, evidence = _mvinverse_documents(eligible=False)

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        group_materials=group_materials,
        mvinverse_pbr_evidence=evidence,
        acknowledge_policy_fallback=True,
    )

    by_id = {item["part_id"]: item for item in plan["assignments"]}
    assert by_id["P0001"]["material_id"] == GREEN
    assert "parameters" not in by_id["P0001"]
    assert report["summary"]["mvinverse_parameterized_part_count"] == 0
    assert report["mvinverse"]["skipped"][0]["reason_code"] == (
        "MVINVERSE_NOT_AUTO_PARAMETER_ELIGIBLE"
    )


def test_single_view_mvinverse_color_requires_matching_multiview_palette() -> None:
    material_id = "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte"
    group_materials = {
        "schema_version": "qwen-palette-material/v1",
        "selections": [
            {
                "group_id": "G07",
                "material_id": material_id,
                "confirmed": True,
            }
        ],
    }
    evidence = {
        "schema_version": "qwen-mvinverse-pbr-evidence/v1",
        "groups": [
            {
                "group_id": "G07",
                "surface_class": "dielectric",
                "contributing_view_ids": ["top"],
                "albedo": {"median": [0.55, 0.35, 0.19]},
                "metallic": {"median": 0.16},
                "roughness": {"median": 0.38},
                "suggestion": {
                    "decision": "preserve",
                    "auto_parameter_eligible": False,
                    "reason_codes": ["insufficient_distinct_views"],
                },
            }
        ],
    }
    palette = {
        "schema_version": "qwen-multiview-palette-fusion/v1",
        "canonical_palette": {
            "schema_version": "qwen-canonical-material-palette/v1",
            "groups": [
                {
                    "group_id": "G07",
                    "base_color": "orange",
                    "sources": [
                        {"view_id": "front", "base_color": "orange"},
                        {"view_id": "top", "base_color": "brown"},
                    ],
                }
            ],
        },
    }

    result, skipped = policy_module._mvinverse_parameterizations(
        group_materials=group_materials,
        mvinverse_pbr_evidence=evidence,
        allowed_material_ids={material_id},
        palette_fusion=palette,
        key_by_group=True,
    )

    parameters, audit = result["G07"]
    assert set(parameters) == {"diffuse_tint"}
    assert audit["parameterization_mode"] == (
        "multiview_palette_corroborated_color_only"
    )
    assert skipped == []

    palette["canonical_palette"]["groups"][0]["sources"][1]["base_color"] = "blue"
    result, skipped = policy_module._mvinverse_parameterizations(
        group_materials=group_materials,
        mvinverse_pbr_evidence=evidence,
        allowed_material_ids={material_id},
        palette_fusion=palette,
        key_by_group=True,
    )
    assert result == {}
    assert skipped[0]["reason_code"] == "MVINVERSE_NOT_AUTO_PARAMETER_ELIGIBLE"


def test_forged_eligible_mvinverse_projection_is_audited_and_not_parameterized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, staged, gate, whitelist = _documents()
    group_materials, evidence = _mvinverse_documents(eligible=True)
    monkeypatch.setattr(
        policy_module,
        "validate_mvinverse_evidence",
        strict_validate_mvinverse_evidence,
    )

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        group_materials=group_materials,
        mvinverse_pbr_evidence=evidence,
        acknowledge_policy_fallback=True,
    )

    assignment = next(
        item for item in plan["assignments"] if item["part_id"] == "P0001"
    )
    assert assignment["material_id"] == GREEN
    assert "parameters" not in assignment
    assert report["summary"]["exact_cover"] is True
    assert report["summary"]["mvinverse_parameterized_part_count"] == 0
    assert report["mvinverse"]["skipped"][0]["reason_code"] == (
        "MVINVERSE_EVIDENCE_STRICT_VALIDATION_FAILED"
    )


def test_incomplete_mvinverse_input_bundle_disables_parameterization() -> None:
    registry, staged, gate, whitelist = _documents()
    group_materials, _evidence = _mvinverse_documents(eligible=True)

    _plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        group_materials=group_materials,
        mvinverse_pbr_evidence=None,
        acknowledge_policy_fallback=True,
    )

    assert report["summary"]["exact_cover"] is True
    assert report["summary"]["mvinverse_parameterized_part_count"] == 0
    assert report["mvinverse"]["skipped"] == [
        {
            "reason_code": "MVINVERSE_INPUT_BUNDLE_INCOMPLETE",
            "detail": (
                "group_materials and mvinverse_pbr_evidence must be supplied "
                "together; parameterization was disabled"
            ),
        }
    ]


def test_verified_autonomous_base_plan_takes_precedence() -> None:
    registry, staged, gate, whitelist = _documents()
    base_plan = {
        "schema_version": "1.0",
        "assignments": [
            {
                "part_id": "P0003",
                "material_id": WHITE,
                "semantic": "recovered white panel",
                "confidence": 0.92,
                "evidence_views": ["front", "side"],
                "status": "auto",
                "parameters": {"paint_roughness": 0.33},
            }
        ],
    }

    plan, report = build_policy_exact_cover(
        registry=registry,
        staged_result=staged,
        confidence_gate=gate,
        whitelist=whitelist,
        base_plan=base_plan,
        acknowledge_policy_fallback=True,
    )

    by_id = {item["part_id"]: item for item in plan["assignments"]}
    recovered = by_id["P0003"]
    assert recovered["status"] == "auto"
    assert recovered["material_id"] == WHITE
    assert recovered["parameters"]["paint_roughness"] == pytest.approx(0.33)
    assert recovered["provenance"]["tier"] == "autonomous_base_plan"
    assert report["summary"]["autonomous_base_assignment_count"] == 1
