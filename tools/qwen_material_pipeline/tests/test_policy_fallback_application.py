from __future__ import annotations

import sys
from typing import Any

import pytest

import qwen_material_pipeline.usd.apply as regular_apply
import qwen_material_pipeline.usd.apply_instances as instance_apply
from qwen_material_pipeline.materials.quality_repair import (
    REPAIR_MODE,
    REPAIR_PROVENANCE_FIELD,
    REPAIR_REASON_CODES,
)
from qwen_material_pipeline.usd.material_common import (
    APPLY_STATUSES,
    POLICY_EXACT_COVER_MODE,
    POLICY_FALLBACK_CONFIDENCE_BASIS,
    POLICY_FALLBACK_STATUS,
    SOURCE_VISUAL_PRESERVE_ACTION,
    SOURCE_VISUAL_PRESERVE_TIER,
    canonical_sha256,
    source_visual_binding_sha256,
    validate_policy_fallback_authorization,
)


def _documents() -> tuple[dict[str, Any], dict[str, Any]]:
    registry = {
        "schema_version": "qwen-material-parts/v1",
        "asset_sha256": "a" * 64,
        "part_count": 2,
        "parts": [
            {"part_id": "P0001", "prim_path": "/Asset/A/Mesh"},
            {"part_id": "P0002", "prim_path": "/Asset/B/Mesh"},
        ],
    }
    plan = {
        "schema_version": "1.0",
        "provenance": {
            "mode": POLICY_EXACT_COVER_MODE,
            "registry_asset_sha256": registry["asset_sha256"],
            "registry_sha256": canonical_sha256(registry),
        },
        "assignments": [
            {
                "part_id": "P0001",
                "material_id": "MAT_GREEN",
                "confidence": 0.0,
                "evidence_views": [],
                "status": POLICY_FALLBACK_STATUS,
                "provenance": {
                    "tier": "staged_auto_parent_path",
                    "reason_codes": [
                        "POLICY_IDENTITY_PROPAGATION",
                        "CLUSTER_PARENT_PATH",
                    ],
                    "output_confidence_basis": (POLICY_FALLBACK_CONFIDENCE_BASIS),
                    "sources": [
                        {
                            "part_id": "P0002",
                            "source_status": "auto",
                            "source_confidence": 0.95,
                            "source_evidence_views": ["front", "iso"],
                        }
                    ],
                },
            },
            {
                "part_id": "P0002",
                "material_id": "MAT_GREEN",
                "confidence": 0.95,
                "evidence_views": ["front", "iso"],
                "status": "auto",
            },
        ],
    }
    return registry, plan


def test_policy_fallback_is_not_a_default_apply_status() -> None:
    assert APPLY_STATUSES == {"auto", "approved"}
    assert POLICY_FALLBACK_STATUS not in APPLY_STATUSES


def test_policy_fallback_requires_explicit_authorization() -> None:
    registry, plan = _documents()

    with pytest.raises(ValueError, match="explicit.*include-policy-fallback"):
        validate_policy_fallback_authorization(
            plan,
            registry,
            include_policy_fallback=False,
        )

    assert (
        validate_policy_fallback_authorization(
            plan,
            registry,
            include_policy_fallback=True,
        )
        == 1
    )


def _source_preserve_documents() -> tuple[dict[str, Any], dict[str, Any]]:
    registry, plan = _documents()
    registry_part = registry["parts"][0]
    registry_part["existing_visual_material"] = "/Asset/Looks/Original"
    plan["provenance"]["registry_sha256"] = canonical_sha256(registry)
    assignment = plan["assignments"][0]
    assignment.update(
        {
            "apply_action": SOURCE_VISUAL_PRESERVE_ACTION,
            "source_visual_material_prim_path": "/Asset/Looks/Original",
            "source_visual_material_binding_sha256": (
                source_visual_binding_sha256(
                    part_id="P0001",
                    prim_path="/Asset/A/Mesh",
                    material_prim_path="/Asset/Looks/Original",
                )
            ),
            "provenance": {
                "tier": SOURCE_VISUAL_PRESERVE_TIER,
                "reason_codes": [
                    "SOURCE_VISUAL_MATERIAL_PRESENT",
                    "SOURCE_VISUAL_BINDING_HASH_BOUND",
                    "PRESERVE_SOURCE_VISUAL_NOOP",
                ],
                "output_confidence_basis": POLICY_FALLBACK_CONFIDENCE_BASIS,
                "sources": [],
            },
        }
    )
    return registry, plan


def test_source_visual_preserve_is_hash_bound_to_exact_registry_binding() -> None:
    registry, plan = _source_preserve_documents()

    assert (
        validate_policy_fallback_authorization(
            plan, registry, include_policy_fallback=True
        )
        == 1
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda registry, plan: plan["assignments"][0].update(
                {"source_visual_material_prim_path": "/Asset/Looks/Tampered"}
            ),
            "does not match.*registry",
        ),
        (
            lambda registry, plan: plan["assignments"][0].update(
                {"source_visual_material_binding_sha256": "0" * 64}
            ),
            "invalid.*SHA-256",
        ),
        (
            lambda registry, plan: plan["assignments"][0].update(
                {"parameters": {}}
            ),
            "conflicts with material-authoring fields",
        ),
        (
            lambda registry, plan: plan["assignments"][0]["provenance"].update(
                {"tier": "neutral_default"}
            ),
            "apply_action is not authorized",
        ),
    ],
)
def test_source_visual_preserve_contract_fails_closed(
    mutation, message: str
) -> None:
    registry, plan = _source_preserve_documents()
    mutation(registry, plan)

    with pytest.raises(ValueError, match=message):
        validate_policy_fallback_authorization(
            plan, registry, include_policy_fallback=True
        )


@pytest.mark.parametrize("module", [regular_apply, instance_apply])
def test_quality_repair_keeps_exact_cover_application_authorization(module) -> None:
    registry, plan = _documents()
    plan["provenance"][REPAIR_PROVENANCE_FIELD] = {
        "mode": REPAIR_MODE,
        "input_hashes": {"baseline_plan_sha256": "b" * 64},
        "changed_part_ids": ["P0001"],
    }
    plan["assignments"][0]["provenance"] = {
        "tier": "qa_repair_candidate",
        "reason_codes": list(REPAIR_REASON_CODES),
        "output_confidence_basis": POLICY_FALLBACK_CONFIDENCE_BASIS,
        "sources": [],
        "canonical_group_id": "G01",
        "baseline_material_id": "MAT_NEUTRAL",
        "baseline_tier": "neutral_default",
        "supporting_view_ids": ["ref_a", "ref_b"],
        "supporting_content_cluster_ids": ["CONTENT_01", "CONTENT_02"],
        "supporting_pose_cluster_ids": ["front", "side"],
    }

    assert (
        module._validate_policy_fallback_authorization(
            plan,
            registry,
            include_policy_fallback=True,
        )
        == 1
    )
    if module is instance_apply:
        plan["provenance"]["asset_sha256"] = "source"
        assert instance_apply._validate_plan_provenance(
            plan,
            source_sha256="source",
            registry_sha256=canonical_sha256(registry),
        ) == {
            "asset_sha256": "source",
            "registry_sha256": canonical_sha256(registry),
        }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda registry, plan: plan["provenance"].update({"mode": "unsafe"}),
            "mode",
        ),
        (
            lambda registry, plan: plan["provenance"].update(
                {"registry_asset_sha256": "b" * 64}
            ),
            "registry_asset_sha256.*does not match",
        ),
        (
            lambda registry, plan: plan["provenance"].update(
                {"registry_sha256": "b" * 64}
            ),
            "registry_sha256.*does not match",
        ),
        (
            lambda registry, plan: plan["assignments"][0].update({"confidence": 0.1}),
            "confidence must remain exactly 0.0",
        ),
        (
            lambda registry, plan: plan["assignments"][0].update(
                {"evidence_views": ["front"]}
            ),
            "evidence_views must remain empty",
        ),
        (
            lambda registry, plan: plan["assignments"][0]["provenance"].update(
                {"tier": ""}
            ),
            "tier must be a non-empty string",
        ),
        (
            lambda registry, plan: plan["assignments"][0]["provenance"].update(
                {"reason_codes": []}
            ),
            "reason_codes must be non-empty",
        ),
        (
            lambda registry, plan: plan["assignments"][0]["provenance"].update(
                {"output_confidence_basis": "model confidence"}
            ),
            "output_confidence_basis is invalid",
        ),
        (
            lambda registry, plan: plan["assignments"][0]["provenance"].update(
                {"sources": "P0002"}
            ),
            "sources must be an array",
        ),
        (
            lambda registry, plan: plan["assignments"][0]["provenance"]["sources"][
                0
            ].update({"source_status": "unknown"}),
            "source_status is invalid",
        ),
        (
            lambda registry, plan: plan["assignments"].pop(),
            "does not exactly cover registry",
        ),
    ],
)
def test_policy_fallback_provenance_fails_closed(mutation, message: str) -> None:
    registry, plan = _documents()
    mutation(registry, plan)

    with pytest.raises(ValueError, match=message):
        validate_policy_fallback_authorization(
            plan,
            registry,
            include_policy_fallback=True,
        )


@pytest.mark.parametrize(
    ("module", "command_flag"),
    [
        (regular_apply, "--asset-usd"),
        (instance_apply, "--source-usd"),
    ],
)
def test_apply_main_forwards_explicit_policy_fallback_flag(
    module, command_flag: str, monkeypatch, capsys
) -> None:
    captured: dict[str, Any] = {}

    def fake_apply(**kwargs):
        captured.update(kwargs)
        return {"applied_count": 1}

    monkeypatch.setattr(module, "_start_isaac_if_needed", lambda **_kwargs: None)
    target_name = (
        "apply_visual_materials"
        if module is regular_apply
        else "apply_instance_materials"
    )
    monkeypatch.setattr(module, target_name, fake_apply)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "apply",
            command_flag,
            "asset.usd",
            "--catalog",
            "catalog.json",
            "--registry",
            "registry.json",
            "--plan",
            "plan.json",
            "--output",
            "look.usda",
            "--material-root",
            "materials",
            "--include-policy-fallback",
        ],
    )

    assert module.main() == 0
    assert captured["include_policy_fallback"] is True
    assert captured["include_review"] is False
    assert '"applied_count": 1' in capsys.readouterr().out
