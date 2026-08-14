from __future__ import annotations

import ast
import copy
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from asset_pipeline.visual_materials import orchestrator
from asset_pipeline.visual_materials.stages import runner as stage_runner
from asset_pipeline.visual_materials.workspace import VisualMaterialWorkspace
from qwen_material_pipeline.materials import component_mdl_tournament


def test_finalize_stage_reuses_prepublish_confidence_and_plan_snapshots() -> None:
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(orchestrator._run_finalize_assignment_stage))
    )
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    def named_calls(name: str) -> list[ast.Call]:
        return [
            call
            for call in calls
            if isinstance(call.func, ast.Name) and call.func.id == name
        ]

    publish_call, = named_calls("build_publish_quality_gate")
    lock_call, = named_calls("build_material_selection_lock")
    publish_keywords = {keyword.arg: keyword.value for keyword in publish_call.keywords}
    lock_keywords = {keyword.arg: keyword.value for keyword in lock_call.keywords}

    assert ast.unparse(publish_keywords["confidence_gate"]) == (
        "prepublish_confidence_gate_document"
    )
    assert ast.unparse(publish_keywords["final_plan"]) == (
        "prepublish_final_plan_document"
    )
    assert ast.unparse(lock_keywords["plan"]) == "prepublish_final_plan_document"

    prepublish_reads = [
        ast.unparse(call.args[0])
        for call in named_calls("read_object")
        if call.args
        and ast.unparse(call.args[0]) in {"confidence_gate", "effective_material_plan"}
        and call.lineno < publish_call.lineno
    ]
    assert prepublish_reads.count("confidence_gate") == 1
    assert prepublish_reads.count("effective_material_plan") == 1


def _progress_messages(messages: list[str]) -> list[str]:
    return [message for message in messages if message.startswith("[PROGRESS]")]


def _observed_surface_part(part_id: str, surface_class: str) -> dict[str, object]:
    return {
        "part_id": part_id,
        "status": "observed",
        "descriptor": {"surface_class": surface_class},
    }


def _material_prediction(
    component_id: str,
    *,
    substrate: str = "metal",
    treatment: str = "paint",
    optical: str = "opaque",
    finish: str = "matte",
    status: str = "accepted",
) -> dict[str, object]:
    return {
        "part_id": component_id,
        "decision_status": status,
        "confidence": 0.9,
        "material_semantics": {
            "schema_version": "qwen-part-material-semantics/v1",
            "substrate": substrate,
            "surface_treatment": treatment,
            "optical_behavior": optical,
            "finish": finish,
            "physical_source": "vision_inference",
            "evidence_status": "observed" if status == "accepted" else "unknown",
            "confidence": 0.9,
        },
    }


def test_semantic_hybrid_component_contract_uses_predicted_material_first() -> None:
    semantics, contract = orchestrator._semantic_hybrid_component_contract(
        component_id="AC_dark",
        member_part_ids=["P1", "P2", "P3"],
        part_id_evidence={
            "parts": [
                _observed_surface_part("P1", "conductor"),
                _observed_surface_part("P2", "conductor"),
                _observed_surface_part("P3", "dielectric"),
            ]
        },
        material_prediction=_material_prediction(
            "AC_dark", substrate="metal", treatment="bare", finish="brushed"
        ),
    )

    assert contract["source"] == "appearance_component_qwen.material_predictions"
    assert contract["selection_order"] == (
        "material_prediction_then_mdl_identity_then_color"
    )
    assert all(value["substrate"] == "metal" for value in semantics.values())
    assert all(
        value["surface_treatment"] == "bare" for value in semantics.values()
    )


@pytest.mark.parametrize("status", ["ambiguous", "unknown"])
def test_semantic_hybrid_component_contract_fails_closed_without_prediction(
    status: str,
) -> None:
    with pytest.raises(
        orchestrator.ComponentMdlTournamentError,
        match="no accepted material prediction",
    ):
        orchestrator._semantic_hybrid_component_contract(
            component_id="AC_unresolved",
            member_part_ids=["P1", "P2"],
            part_id_evidence={
                "parts": [
                    _observed_surface_part("P1", "dielectric"),
                    _observed_surface_part("P2", "dielectric"),
                ]
            },
            material_prediction=_material_prediction(
                "AC_unresolved", status=status
            ),
        )


def test_semantic_hybrid_choice_replay_rejects_cross_material_color_proxy() -> None:
    paint = "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte"
    water = "mdl:Natural/Water.mdl#Water"
    catalog = {
        "schema_version": 2,
        "materials": [
            {
                "material_id": paint,
                "family": "paint",
                "surface_semantics": _catalog_surface(
                    treatment="paint",
                    substrates=["metal", "polymer", "wood"],
                    finish="matte",
                ),
            },
            {
                "material_id": water,
                "family": "liquid",
                "surface_semantics": {
                    **_catalog_surface(
                        treatment="bare",
                        substrates=["liquid"],
                        finish="smooth",
                    ),
                    "optical_behavior": "transparent",
                },
            },
        ],
    }
    prediction = _material_prediction("P1")
    accepted_unsigned = {
        "require_material_prediction": True,
        "choices": {"P1": paint},
        "material_predictions": {"P1": prediction},
    }
    orchestrator._require_predicted_material_choices_compatible(
        qwen_document={
            **accepted_unsigned,
            "integrity": {
                "document_sha256": orchestrator.canonical_sha256(
                    accepted_unsigned
                )
            },
        },
        catalog_document=catalog,
        assignment_label="Part-ID",
    )
    rejected_unsigned = {
        "require_material_prediction": True,
        "choices": {"P1": water},
        "material_predictions": {"P1": prediction},
    }
    with pytest.raises(RuntimeError, match="does not match its predicted material"):
        orchestrator._require_predicted_material_choices_compatible(
            qwen_document={
                **rejected_unsigned,
                "integrity": {
                    "document_sha256": orchestrator.canonical_sha256(
                        rejected_unsigned
                    )
                },
            },
            catalog_document=catalog,
            assignment_label="Part-ID",
        )


def test_semantic_hybrid_binds_prediction_to_part_id_plan_and_audit() -> None:
    material_id = "mdl:Plastics/Plastic_ABS.mdl#Plastic_ABS"
    prediction = _material_prediction(
        "P1", substrate="polymer", treatment="bare", finish="smooth"
    )
    qwen_unsigned = {
        "require_material_prediction": True,
        "selection_order": (
            "material_prediction_then_mdl_identity_lock_then_same_id_color"
        ),
        "choices": {"P1": material_id},
        "material_predictions": {"P1": prediction},
    }
    qwen_document = {
        **qwen_unsigned,
        "integrity": {
            "document_sha256": orchestrator.canonical_sha256(qwen_unsigned)
        },
    }
    plan, audit = orchestrator._bind_part_id_material_predictions(
        plan={
            "schema_version": "1.0",
            "assignments": [
                {
                    "part_id": "P1",
                    "material_id": material_id,
                    "provenance": {"assignment_unit": "part_id"},
                }
            ],
        },
        audit={
            "parts": [
                {
                    "part_id": "P1",
                    "status": "independently_selected",
                    "material_id": material_id,
                }
            ],
            "summary": {"part_count": 1},
        },
        qwen_document=qwen_document,
    )
    binding = plan["assignments"][0]["provenance"][
        "material_prediction_selection"
    ]
    assert binding["selected_material_id"] == material_id
    assert binding["material_identity_locked_before_color"] is True
    assert binding["color_stage_material_id_change_allowed"] is False
    assert audit["parts"][0]["material_prediction_selection"] == binding
    assert audit["output_plan_sha256"] == orchestrator.canonical_sha256(plan)


def _sealed_prediction_document(
    choices: dict[str, str], predictions: dict[str, dict[str, object]]
) -> dict[str, object]:
    unsigned = {
        "require_material_prediction": True,
        "selection_order": (
            "material_prediction_then_mdl_identity_lock_then_same_id_color"
        ),
        "choices": choices,
        "material_predictions": predictions,
    }
    return {
        **unsigned,
        "integrity": {
            "document_sha256": orchestrator.canonical_sha256(unsigned)
        },
    }


def test_semantic_hybrid_excludes_physically_mixed_appearance_component() -> None:
    matte = "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte"
    gloss = "mdl:Miscellaneous/Paint_Gloss.mdl#Paint_Gloss"
    part_document = _sealed_prediction_document(
        {"P1": gloss, "P2": matte, "P3": gloss},
        {
            "P1": _material_prediction("P1"),
            "P2": _material_prediction("P2", substrate="polymer"),
            "P3": _material_prediction("P3"),
        },
    )
    component_document = _sealed_prediction_document(
        {"AC_dark": gloss},
        {"AC_dark": _material_prediction("AC_dark")},
    )

    authorized, audit = orchestrator._semantic_hybrid_component_authorization(
        appearance_components={
            "components": [
                {
                    "component_id": "AC_dark",
                    "member_part_ids": ["P1", "P2", "P3"],
                }
            ]
        },
        part_qwen_document=part_document,
        component_qwen_document=component_document,
    )

    assert authorized == []
    assert audit["summary"]["excluded_component_count"] == 1
    assert audit["excluded_components"][0]["conflicting_part_ids"] == ["P2"]
    assert audit["excluded_components"][0]["resolution"] == (
        "retain_independent_part_id_material_identities"
    )


def test_semantic_hybrid_binds_homogeneous_component_identity_override() -> None:
    matte = "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte"
    gloss = "mdl:Miscellaneous/Paint_Gloss.mdl#Paint_Gloss"
    part_document = _sealed_prediction_document(
        {"P1": matte, "P2": gloss},
        {"P1": _material_prediction("P1"), "P2": _material_prediction("P2")},
    )
    component_document = _sealed_prediction_document(
        {"AC_green": gloss},
        {"AC_green": _material_prediction("AC_green")},
    )
    plan, audit = orchestrator._bind_part_id_material_predictions(
        plan={
            "assignments": [
                {
                    "part_id": part_id,
                    "material_id": gloss,
                    "provenance": {},
                }
                for part_id in ("P1", "P2")
            ],
            "provenance": {
                "appearance_component_mdl_selection": {
                    "selections": [
                        {
                            "component_id": "AC_green",
                            "member_part_ids": ["P1", "P2"],
                            "material_id": gloss,
                        }
                    ]
                }
            },
        },
        audit={
            "parts": [
                {
                    "part_id": part_id,
                    "status": "independently_selected",
                    "material_id": gloss,
                }
                for part_id in ("P1", "P2")
            ],
            "summary": {},
        },
        qwen_document=part_document,
        component_qwen_document=component_document,
    )

    binding = plan["assignments"][0]["provenance"][
        "material_prediction_selection"
    ]
    assert binding["independent_predicted_material_id"] == matte
    assert binding["selected_material_id"] == gloss
    assert binding["selection_scope"] == (
        "physically_homogeneous_appearance_component"
    )
    assert binding["appearance_component_prediction_selection"]["component_id"] == (
        "AC_green"
    )
    assert audit["summary"]["material_prediction_bound_count"] == 2


def test_semantic_hybrid_component_binding_rejects_physical_conflict() -> None:
    matte = "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte"
    gloss = "mdl:Miscellaneous/Paint_Gloss.mdl#Paint_Gloss"
    part_document = _sealed_prediction_document(
        {"P1": matte, "P2": gloss},
        {
            "P1": _material_prediction("P1", substrate="polymer"),
            "P2": _material_prediction("P2"),
        },
    )
    component_document = _sealed_prediction_document(
        {"AC_green": gloss},
        {"AC_green": _material_prediction("AC_green")},
    )
    with pytest.raises(RuntimeError, match="conflicts for P1"):
        orchestrator._bind_part_id_material_predictions(
            plan={
                "assignments": [
                    {
                        "part_id": part_id,
                        "material_id": gloss,
                        "provenance": {},
                    }
                    for part_id in ("P1", "P2")
                ],
                "provenance": {
                    "appearance_component_mdl_selection": {
                        "selections": [
                            {
                                "component_id": "AC_green",
                                "member_part_ids": ["P1", "P2"],
                                "material_id": gloss,
                            }
                        ]
                    }
                },
            },
            audit={
                "parts": [
                    {
                        "part_id": part_id,
                        "status": "independently_selected",
                        "material_id": gloss,
                    }
                    for part_id in ("P1", "P2")
                ],
                "summary": {},
            },
            qwen_document=part_document,
            component_qwen_document=component_document,
        )


def test_semantic_hybrid_catalog_expansion_is_bound_for_downstream_plan() -> None:
    material_id = "mdl:Plastics/Plastic_ABS.mdl#Plastic_ABS"
    retrieval_unsigned = {
        "schema_version": "qwen-visual-material-retrieval-result/v1",
        "groups": [{"group_id": "P1", "fused_ranking": []}],
    }
    retrieval = {
        **retrieval_unsigned,
        "integrity": {
            "result_sha256": orchestrator.canonical_sha256(retrieval_unsigned)
        },
    }
    expanded = orchestrator._retrieval_with_predicted_material_choices(
        retrieval_document=retrieval,
        qwen_document={
            "choices": {"P1": material_id},
            "visual_compatibility_gate": {
                "parts": [
                    {
                        "part_id": "P1",
                        "material_prediction_gate": {
                            "full_catalog_expansion_used": True
                        },
                    }
                ]
            },
        },
    )
    self_check = copy.deepcopy(expanded)
    integrity = self_check.pop("integrity")
    assert integrity["result_sha256"] == orchestrator.canonical_sha256(self_check)
    assert expanded["groups"][0]["fused_ranking"] == [
        {
            "rank": 1,
            "material_id": material_id,
            "score": 0.0,
            "physical_catalog_expansion": True,
            "selection_source": "predicted_material_compatible_catalog",
        }
    ]


def test_semantic_hybrid_rejects_unsealed_retrieval_expansion() -> None:
    retrieval_unsigned = {
        "schema_version": "qwen-visual-material-retrieval-result/v1",
        "groups": [{"group_id": "P1", "fused_ranking": []}],
    }
    retrieval = {
        **retrieval_unsigned,
        "integrity": {
            "result_sha256": orchestrator.canonical_sha256(retrieval_unsigned)
        },
    }
    with pytest.raises(RuntimeError, match="neither retrieved nor authorized"):
        orchestrator._retrieval_with_predicted_material_choices(
            retrieval_document=retrieval,
            qwen_document={
                "choices": {"P1": "mdl:Plastics/Plastic_ABS.mdl#Plastic_ABS"},
                "visual_compatibility_gate": {
                    "parts": [
                        {
                            "part_id": "P1",
                            "material_prediction_gate": {
                                "full_catalog_expansion_used": False
                            },
                        }
                    ]
                },
            },
        )


def _absolute_quality(statuses: list[str], aggregate_status: str) -> dict[str, object]:
    reference_count = len(statuses)
    return {
        "aggregate": {
            "status": aggregate_status,
            "reference_view_count": reference_count,
            "comparable_view_count": reference_count,
            "passed_view_count": statuses.count("PASS"),
            "review_view_count": statuses.count("REVIEW"),
            "failed_view_count": statuses.count("FAIL"),
            "unscorable_view_count": statuses.count("UNSCORABLE"),
        },
        "views": [
            {"reference_view_id": f"view_{index}", "status": status}
            for index, status in enumerate(statuses)
        ],
    }


def test_semantic_hybrid_absolute_quality_requires_every_view_pass() -> None:
    orchestrator._require_semantic_hybrid_absolute_quality_pass(
        _absolute_quality(["PASS", "PASS", "PASS", "PASS"], "PASS"),
        expected_reference_view_ids=["view_0", "view_1", "view_2", "view_3"],
    )

    with pytest.raises(RuntimeError, match="every registered reference view"):
        orchestrator._require_semantic_hybrid_absolute_quality_pass(
            _absolute_quality(["PASS", "FAIL", "PASS", "PASS"], "FAIL"),
            expected_reference_view_ids=["view_0", "view_1", "view_2", "view_3"],
        )


def test_semantic_hybrid_absolute_quality_rejects_omitted_view() -> None:
    quality = _absolute_quality(["PASS", "PASS"], "PASS")
    quality["aggregate"]["reference_view_count"] = 3  # type: ignore[index]

    with pytest.raises(RuntimeError, match="absolute visual gate failed"):
        orchestrator._require_semantic_hybrid_absolute_quality_pass(
            quality,
            expected_reference_view_ids=["view_0", "view_1", "view_2"],
        )


def test_semantic_hybrid_absolute_quality_does_not_trust_self_reported_view_count() -> None:
    forged = _absolute_quality(["PASS", "PASS"], "PASS")

    with pytest.raises(RuntimeError, match="absolute visual gate failed"):
        orchestrator._require_semantic_hybrid_absolute_quality_pass(
            forged,
            expected_reference_view_ids=["view_0", "view_1", "view_2", "view_3"],
        )


def test_semantic_hybrid_invocation_fails_before_pipeline_stages() -> None:
    hybrid = SimpleNamespace(material_selection_pipeline_mode="semantic_hybrid")

    with pytest.raises(ValueError, match="fresh fail-closed contract") as exc_info:
        orchestrator._validate_semantic_hybrid_invocation(
            config=hybrid,
            inference_mode="bundled",
            partial_live_resume=True,
            require_complete_coverage=False,
            allow_policy_material_fallback=False,
        )

    message = str(exc_info.value)
    assert "bundled is unsupported" in message
    assert "partial live resume is unsupported" in message
    assert "require_complete_coverage must be true" in message
    assert "allow_policy_material_fallback must be true" in message


def test_current_pipeline_mode_keeps_hybrid_invocation_gate_disabled() -> None:
    orchestrator._validate_semantic_hybrid_invocation(
        config=SimpleNamespace(),
        inference_mode="bundled",
        partial_live_resume=True,
        require_complete_coverage=False,
        allow_policy_material_fallback=False,
    )


def _minimal_orchestrator_job_dependencies(
    tmp_path: Path,
    *,
    pipeline_mode: str,
) -> tuple[Path, Path, SimpleNamespace]:
    source = tmp_path / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    isaac = tmp_path / "isaac-python"
    isaac.write_text("#!/bin/sh\n", encoding="utf-8")
    isaac.chmod(0o755)
    config = SimpleNamespace(material_selection_pipeline_mode=pipeline_mode)
    return source, isaac, config


def test_invalid_semantic_hybrid_rejects_before_output_or_resume_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, isaac, config = _minimal_orchestrator_job_dependencies(
        tmp_path,
        pipeline_mode="semantic_hybrid",
    )
    output = tmp_path / "hybrid-output"
    resume_calls: list[Path] = []
    monkeypatch.setenv("ASSET_PIPELINE_DISABLE_CPU_STABILITY_GUARD", "1")
    monkeypatch.setattr(
        orchestrator,
        "_verified_partial_live_resume_available",
        lambda destination, *_args: resume_calls.append(destination) or False,
    )

    with pytest.raises(ValueError, match="fresh fail-closed contract"):
        orchestrator.run_assign_visual_materials_job(
            source_usd=str(source),
            references=(),
            output_dir=str(output),
            inference_mode="live",
            acknowledge_mvinverse_noncommercial=True,
            require_complete_coverage=False,
            allow_policy_material_fallback=False,
            _config_loader=lambda _path: config,
            _isaac_python_resolver=lambda: isaac,
            _reference_parser=lambda _references: (),
            _default_config_path=tmp_path / "unused-config.json",
        )

    assert not output.exists()
    assert resume_calls == []


def test_semantic_hybrid_rejects_existing_output_without_resume_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, isaac, config = _minimal_orchestrator_job_dependencies(
        tmp_path,
        pipeline_mode="semantic_hybrid",
    )
    output = tmp_path / "hybrid-output"
    output.mkdir()
    resume_calls: list[Path] = []
    monkeypatch.setenv("ASSET_PIPELINE_DISABLE_CPU_STABILITY_GUARD", "1")
    monkeypatch.setattr(
        orchestrator,
        "_verified_partial_live_resume_available",
        lambda destination, *_args: resume_calls.append(destination) or True,
    )

    with pytest.raises(FileExistsError, match="never resumes"):
        orchestrator.run_assign_visual_materials_job(
            source_usd=str(source),
            references=(),
            output_dir=str(output),
            inference_mode="live",
            acknowledge_mvinverse_noncommercial=True,
            require_complete_coverage=True,
            allow_policy_material_fallback=True,
            _config_loader=lambda _path: config,
            _isaac_python_resolver=lambda: isaac,
            _reference_parser=lambda _references: (),
            _default_config_path=tmp_path / "unused-config.json",
        )

    assert resume_calls == []


@pytest.mark.parametrize(
    ("pipeline_mode", "existing_output", "resume_available"),
    [
        ("semantic_hybrid", False, False),
        ("current", True, True),
    ],
)
def test_valid_fresh_hybrid_and_current_resume_reach_first_pipeline_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pipeline_mode: str,
    existing_output: bool,
    resume_available: bool,
) -> None:
    source, isaac, config = _minimal_orchestrator_job_dependencies(
        tmp_path,
        pipeline_mode=pipeline_mode,
    )
    output = tmp_path / f"{pipeline_mode}-output"
    if existing_output:
        output.mkdir()
    resume_calls: list[Path] = []
    stage_calls: list[Path] = []

    class FirstStageReached(RuntimeError):
        pass

    def resume_spy(destination: Path, *_args: object) -> bool:
        resume_calls.append(destination)
        return resume_available

    def stage_spy(context: object, **_kwargs: object) -> object:
        stage_calls.append(context.destination)  # type: ignore[attr-defined]
        raise FirstStageReached

    monkeypatch.setenv("ASSET_PIPELINE_DISABLE_CPU_STABILITY_GUARD", "1")
    monkeypatch.setattr(
        orchestrator,
        "_verified_partial_live_resume_available",
        resume_spy,
    )
    monkeypatch.setattr(orchestrator, "prepare_source_evidence", stage_spy)

    with pytest.raises(FirstStageReached):
        orchestrator.run_assign_visual_materials_job(
            source_usd=str(source),
            references=(),
            output_dir=str(output),
            inference_mode="live",
            acknowledge_mvinverse_noncommercial=True,
            require_complete_coverage=pipeline_mode == "semantic_hybrid",
            allow_policy_material_fallback=pipeline_mode == "semantic_hybrid",
            _config_loader=lambda _path: config,
            _isaac_python_resolver=lambda: isaac,
            _reference_parser=lambda _references: (),
            _default_config_path=tmp_path / "unused-config.json",
        )

    assert output.is_dir()
    assert stage_calls == [output.resolve()]
    assert resume_calls == ([] if pipeline_mode == "semantic_hybrid" else [output])


def _catalog_surface(
    *,
    treatment: str,
    substrates: list[str],
    finish: str,
) -> dict[str, object]:
    return {
        "schema_version": "qwen-catalog-surface-semantics/v1",
        "surface_treatment": treatment,
        "optical_behavior": "opaque",
        "finish": finish,
        "compatible_substrates": substrates,
        "confidence": "high",
        "inference_source": "test_reviewed_catalog/v1",
    }


def _component_score(
    component_id: str,
    members: list[str],
    appearance_score: float,
) -> dict[str, object]:
    member_scores = [
        {
            "part_id": part_id,
            "comparison_pixel_count": 100,
            "appearance_score": appearance_score,
        }
        for part_id in sorted(members)
    ]
    return {
        "schema_version": "qwen-appearance-component-actual-mdl-tournament/v1",
        "component_id": component_id,
        "member_part_ids": sorted(members),
        "member_score_count": len(members),
        "comparison_pixel_count": 100 * len(members),
        "appearance_score": appearance_score,
        "color_score": appearance_score,
        "luma_score": appearance_score,
        "lab_delta_e": round(100.0 * (1.0 - appearance_score), 8),
        "member_scores": member_scores,
    }


@pytest.mark.parametrize(
    "accept_color_h1",
    [True, False],
    ids=["h1-winner", "all-h0-retained"],
)
def test_semantic_hybrid_full_component_flow_discards_unsafe_h0_and_binds_h1(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    accept_color_h1: bool,
) -> None:
    destination = tmp_path / "visual_material"
    destination.mkdir()
    source = tmp_path / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    workspace = VisualMaterialWorkspace.create(
        destination=destination,
        source=source,
    )
    paint_ids = [
        "mdl:Miscellaneous/Paint_Gloss.mdl#Paint_Gloss",
        "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte",
        "mdl:Miscellaneous/Paint_Satin.mdl#Paint_Satin",
    ]
    metal_ids = [
        "mdl:Metals/Aluminum_Cast.mdl#Aluminum_Cast",
        "mdl:Metals/Aluminum_Polished.mdl#Aluminum_Polished",
        "mdl:Metals/Brass.mdl#Brass",
    ]
    unsafe_paint_h0 = "mdl:Textiles/Linen_Blue.mdl#Linen_Blue"
    unsafe_metal_h0 = "mdl:Water/Water.mdl#Water"
    source_plan = {
        "schema_version": "1.0",
        "assignments": [
            {"part_id": "P1", "material_id": unsafe_paint_h0},
            {"part_id": "P2", "material_id": unsafe_paint_h0},
            {"part_id": "P3", "material_id": unsafe_metal_h0},
            {"part_id": "P4", "material_id": unsafe_metal_h0},
        ],
        "provenance": {"test": "native_qwen_h0"},
    }
    source_audit = {
        "output_plan_sha256": orchestrator.canonical_sha256(source_plan),
        "parts": [
            {
                "part_id": assignment["part_id"],
                "status": "independently_selected",
                "material_id": assignment["material_id"],
            }
            for assignment in source_plan["assignments"]
        ],
        "summary": {
            "part_count": 4,
            "independently_selected_count": 4,
            "unobserved_preserved_count": 0,
            "exact_cover": True,
        },
    }
    orchestrator.write_object(
        workspace.appearance.mdl_selection_audit,
        {
            "selections": [
                {
                    "component_id": "AC_01_paint",
                    "member_part_ids": ["P1", "P2"],
                    "material_id": unsafe_paint_h0,
                    "canonical_reference_rgb": [0.2, 0.45, 0.8],
                },
                {
                    "component_id": "AC_02_metal",
                    "member_part_ids": ["P3", "P4"],
                    "material_id": unsafe_metal_h0,
                    "canonical_reference_rgb": [0.25, 0.27, 0.3],
                },
            ]
        },
    )
    orchestrator.write_object(
        workspace.appearance.retrieval_result,
        {
            "groups": [
                {
                    "group_id": "AC_01_paint",
                    "color_ranking": [
                        {"rank": index, "material_id": material_id}
                        for index, material_id in enumerate(paint_ids, start=1)
                    ],
                },
                {
                    "group_id": "AC_02_metal",
                    "color_ranking": [
                        {"rank": index, "material_id": material_id}
                        for index, material_id in enumerate(metal_ids, start=1)
                    ],
                },
            ]
        },
    )
    orchestrator.write_object(
        workspace.appearance.qwen_result,
        {
            "require_material_prediction": True,
            "material_predictions": {
                "AC_01_paint": _material_prediction("AC_01_paint"),
                "AC_02_metal": _material_prediction(
                    "AC_02_metal",
                    substrate="metal",
                    treatment="bare",
                    finish="smooth",
                ),
            },
            "visual_compatibility_gate": {"parts": []},
        },
    )
    orchestrator.write_object(
        workspace.part_id.evidence,
        {
            "parts": [
                _observed_surface_part("P1", "dielectric"),
                _observed_surface_part("P2", "dielectric"),
                _observed_surface_part("P3", "conductor"),
                _observed_surface_part("P4", "conductor"),
            ]
        },
    )
    orchestrator.write_object(workspace.part_id.material_audit, source_audit)
    spatial_report = workspace.inference.root / "spatial_mapping_report.json"
    orchestrator.write_object(spatial_report, {"test": "registered"})
    orchestrator.write_object(
        workspace.source.rendered_registry,
        {"test": "source_registry"},
    )
    orchestrator.write_object(
        workspace.quality.rendered_registry,
        {"score_marker": "unsafe_initial"},
    )
    catalog_path = workspace.inference.root / "catalog.json"
    catalog_materials = [
        {
            "material_id": material_id,
            "family": "paint",
            "surface_semantics": _catalog_surface(
                treatment="paint",
                substrates=["metal", "polymer", "wood"],
                finish=finish,
            ),
        }
        for material_id, finish in zip(
            paint_ids,
            ["glossy", "matte", "satin"],
            strict=True,
        )
    ] + [
        {
            "material_id": material_id,
            "family": "metal",
            "surface_semantics": _catalog_surface(
                treatment="bare",
                substrates=["metal"],
                finish=finish,
            ),
        }
        for material_id, finish in zip(
            metal_ids,
            ["smooth", "polished", "brushed"],
            strict=True,
        )
    ]
    orchestrator.write_object(
        catalog_path,
        {"schema_version": 2, "materials": catalog_materials},
    )

    events: list[str] = []

    def fake_stage(
        stage_name: str,
        _command: list[str],
        _log_cb,
        **kwargs,
    ) -> None:
        events.append(stage_name)
        required_files = tuple(kwargs.get("required_files", ()))

        def command_path(flag: str) -> Path:
            index = _command.index(flag)
            return Path(_command[index + 1])

        for path in required_files:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix.lower() in {".usd", ".usda", ".usdc"}:
                path.write_text("#usda 1.0\n", encoding="utf-8")
        for path in required_files:
            if path.suffix.lower() in {".usd", ".usda", ".usdc"}:
                continue
            if stage_name.endswith("_apply"):
                apply_plan = command_path("--plan")
                output_usd = command_path("--output").resolve(strict=True)
                orchestrator.write_object(
                    path,
                    {
                        "applied_count": 4,
                        "plan_sha256": orchestrator.canonical_sha256(
                            orchestrator.read_object(apply_plan, "fake apply plan")
                        ),
                        "output_usd": str(output_usd),
                    },
                )
            elif stage_name.endswith("_registry"):
                output_usd = command_path("--usd").resolve(strict=True)
                orchestrator.write_object(
                    path,
                    {
                        "asset_usd": str(output_usd),
                        "asset_sha256": orchestrator.sha256_file(output_usd),
                    },
                )
            elif stage_name.endswith("_render"):
                rendered = orchestrator.read_object(
                    command_path("--registry"),
                    "fake input registry",
                )
                rendered["score_marker"] = stage_name
                orchestrator.write_object(path, rendered)
            else:
                orchestrator.write_object(path, {"test": stage_name})

    unsafe_baseline_score_calls: list[str] = []

    def fake_score_component_render(
        *,
        component_id: str,
        member_part_ids: list[str],
        rendered_registry: dict[str, object],
        **_kwargs,
    ) -> dict[str, object]:
        marker = str(rendered_registry.get("score_marker"))
        if marker == "unsafe_initial":
            unsafe_baseline_score_calls.append(component_id)
            score = 0.99
        elif "identity_final" in marker:
            score = 0.70 if component_id == "AC_01_paint" else 0.60
        elif "color_h1" in marker:
            score = (
                0.74 if component_id == "AC_01_paint" and accept_color_h1 else 0.59
            )
        elif "identity_1_render" in marker:
            score = 0.40 if component_id == "AC_01_paint" else 0.60
        elif "identity_2_render" in marker:
            score = 0.70 if component_id == "AC_01_paint" else 0.55
        elif "identity_3_render" in marker:
            score = 0.50
        else:  # pragma: no cover - makes unexpected ordering immediately visible
            raise AssertionError(f"unexpected score marker: {marker}")
        return _component_score(component_id, member_part_ids, score)

    monkeypatch.setattr(orchestrator, "_run_stage", fake_stage)
    monkeypatch.setattr(
        orchestrator,
        "score_component_render",
        fake_score_component_render,
    )
    monkeypatch.setattr(
        component_mdl_tournament,
        "score_component_render",
        fake_score_component_render,
    )
    real_rebind = (
        orchestrator.rebind_part_id_material_audit_for_component_mdl_tournament
    )
    trusted_score_evidence_calls: list[
        orchestrator.ComponentColorScoreEvidence
    ] = []

    def rebind_spy(
        *,
        source_audit: dict[str, object],
        source_plan: dict[str, object] | None = None,
        final_plan: dict[str, object],
        tournament_audit: dict[str, object],
        trusted_color_score_evidence: (
            orchestrator.ComponentColorScoreEvidence | None
        ) = None,
    ) -> dict[str, object]:
        assert isinstance(
            trusted_color_score_evidence,
            orchestrator.ComponentColorScoreEvidence,
        )
        color_contract = tournament_audit["component_color_tournament"]
        assert isinstance(color_contract, dict)
        color_components = color_contract["components"]
        assert isinstance(color_components, list)
        component_ids = {
            component["component_id"] for component in color_components
        }
        assert (
            set(trusted_color_score_evidence.h1_artifacts_by_component)
            == component_ids
        )
        artifact_root = workspace.appearance.actual_mdl_tournament_dir
        assert Path(trusted_color_score_evidence.artifact_root) == artifact_root
        h0 = trusted_color_score_evidence.h0_artifact
        identity_dir = artifact_root / "identity_final"
        assert Path(h0.plan) == identity_dir / "plan.json"
        assert Path(h0.apply_plan) == identity_dir / "plan.json"
        assert Path(h0.apply_report) == identity_dir / "apply_report.json"
        assert Path(h0.look_usd) == identity_dir / "look.usda"
        assert Path(h0.rendered_registry) == (
            identity_dir / "renders" / "part_registry.rendered.json"
        )
        for component_id, h1 in (
            trusted_color_score_evidence.h1_artifacts_by_component.items()
        ):
            color_dir = artifact_root / component_id / "H1_color"
            assert Path(h1.plan) == color_dir / "plan.json"
            assert Path(h1.apply_plan) == color_dir / "plan.json"
            assert Path(h1.apply_report) == color_dir / "apply_report.json"
            assert Path(h1.look_usd) == color_dir / "look.usda"
            assert Path(h1.rendered_registry) == (
                color_dir / "renders" / "part_registry.rendered.json"
            )
        trusted_score_evidence_calls.append(trusted_color_score_evidence)
        return real_rebind(
            source_audit=source_audit,
            source_plan=source_plan,
            final_plan=final_plan,
            tournament_audit=tournament_audit,
            trusted_color_score_evidence=trusted_color_score_evidence,
        )

    monkeypatch.setattr(
        orchestrator,
        "rebind_part_id_material_audit_for_component_mdl_tournament",
        rebind_spy,
    )
    config = SimpleNamespace(
        material_root=tmp_path / "materials",
        exact_mdl_tournament_max_candidates=3,
        exact_mdl_tournament_minimum_score_improvement=0.015,
        render_resolution=32,
        render_rt_subframes=1,
        quality_lighting_profile="studio_softbox_v1",
        analysis_up_axis="Z",
        analysis_front_axis="-Y",
    )
    context = SimpleNamespace(
        config=config,
        source=source,
        isaac_python=tmp_path / "isaac_python",
        destination=destination,
        workspace=workspace,
    )
    prepared = SimpleNamespace(
        rendered_registry=workspace.source.rendered_registry,
        instance_root_count=0,
    )
    planning = SimpleNamespace(
        effective_catalog=catalog_path,
        use_policy_fallback=True,
    )
    look = SimpleNamespace(
        rendered_registry_document={"parts": []},
        apply_subcommand="apply-part-plan",
        apply_asset_flag="--asset",
        apply_asset=source,
    )

    result = orchestrator._run_semantic_hybrid_component_tournament(
        context,
        prepared_source=prepared,
        planning=planning,
        look=look,
        source_plan=source_plan,
        spatial_report_path=spatial_report,
        quality_render_view_arguments=("--view", "front=0,0,1"),
        expected_applied_count=4,
        log_cb=None,
        command_runner=lambda *_args, **_kwargs: None,
    )

    assert unsafe_baseline_score_calls == []
    assert len(trusted_score_evidence_calls) == 1
    trusted_score_evidence = trusted_score_evidence_calls[0]
    assert trusted_score_evidence.evidence == orchestrator.read_object(
        workspace.part_id.evidence,
        "trusted test evidence",
    )
    assert trusted_score_evidence.spatial_mapping_report == {"test": "registered"}
    assert orchestrator.read_object(
        Path(trusted_score_evidence.h0_artifact.rendered_registry),
        "trusted H0 test registry",
    )["score_marker"] == "semantic_component_identity_final_render"
    audit = result.tournament_document
    assert audit["candidate_count"] == 6
    assert audit["actual_candidate_render_count"] == 6
    assert audit["winner_count"] == 2
    assert audit["component_color_candidate_count"] == 2
    assert audit["component_color_h1_winner_count"] == int(accept_color_h1)
    assert all(
        component["unsafe_qwen_baseline_discarded"] is True
        for component in audit["components"]
    )
    final_by_part = {
        assignment["part_id"]: assignment for assignment in result.assignments
    }
    assert final_by_part["P1"]["material_id"] == paint_ids[1]
    assert final_by_part["P2"]["material_id"] == paint_ids[1]
    assert final_by_part["P3"]["material_id"] == metal_ids[0]
    assert final_by_part["P4"]["material_id"] == metal_ids[0]
    if accept_color_h1:
        assert final_by_part["P1"]["parameters"] == final_by_part["P2"][
            "parameters"
        ]
        assert final_by_part["P1"]["parameters"]
    else:
        assert not final_by_part["P1"].get("parameters")
        assert not final_by_part["P2"].get("parameters")
    assert not final_by_part["P3"].get("parameters")
    assert not final_by_part["P4"].get("parameters")
    assert events.index("semantic_component_identity_final_render") > max(
        index
        for index, event in enumerate(events)
        if "_identity_" in event and event.endswith("_render")
        and event != "semantic_component_identity_final_render"
    )
    assert events.index("semantic_component_1_color_h1_render") > events.index(
        "semantic_component_identity_final_render"
    )
    assert events[-1] == "semantic_component_final_render"

    if accept_color_h1:
        tampered_plan = copy.deepcopy(result.plan)
        tampered_assignment = next(
            assignment
            for assignment in tampered_plan["assignments"]
            if assignment["part_id"] == "P1"
        )
        tampered_assignment["provenance"][
            "appearance_component_color_candidate"
        ]["source_plan_sha256"] = "0" * 64
        tampered_audit = copy.deepcopy(audit)
        tampered_audit["component_color_tournament"]["final_plan_sha256"] = (
            orchestrator.canonical_sha256(tampered_plan)
        )
        with pytest.raises(
            orchestrator.ComponentMdlTournamentError,
            match="exact color candidate binding",
        ):
            orchestrator.rebind_part_id_material_audit_for_component_mdl_tournament(
                source_audit=source_audit,
                final_plan=tampered_plan,
                tournament_audit=tampered_audit,
                trusted_color_score_evidence=trusted_score_evidence,
            )

    forged_score_audit = copy.deepcopy(audit)
    forged_color_record = forged_score_audit["component_color_tournament"][
        "components"
    ][0]
    forged_color_record["h0_score"] = _component_score(
        forged_color_record["component_id"],
        forged_color_record["member_part_ids"],
        0.01,
    )
    with pytest.raises(
        orchestrator.ComponentMdlTournamentError,
        match="scores do not match trusted render evidence",
    ):
        orchestrator.rebind_part_id_material_audit_for_component_mdl_tournament(
            source_audit=source_audit,
            final_plan=result.plan,
            tournament_audit=forged_score_audit,
            trusted_color_score_evidence=trusted_score_evidence,
        )


def test_run_stage_reports_start_and_complete_with_real_elapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    times = iter((10.0, 12.375))
    monkeypatch.setattr(stage_runner, "monotonic", lambda: next(times))

    stage_runner._run_stage(
        "render_locked",
        ["/fake/isaac/python.sh"],
        messages.append,
        command_runner=lambda *_args, **_kwargs: None,
    )

    progress = _progress_messages(messages)
    assert len(progress) == 2
    assert progress[0] == (
        "[PROGRESS] visual_materials/render_locked START elapsed=0.000s"
    )
    assert progress[1] == (
        "[PROGRESS] visual_materials/render_locked COMPLETE elapsed=2.375s"
    )


def test_run_stage_restores_full_affinity_for_native_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def runner(command, **_kwargs) -> None:
        captured.append(command)

    monkeypatch.setattr(stage_runner, "run_command", runner)
    monkeypatch.setattr(stage_runner, "_TASKSET_EXECUTABLE", "/usr/bin/taskset")
    monkeypatch.setattr(
        stage_runner,
        "_VISUAL_CONTROL_CHILD_CPU_AFFINITY",
        (0, 2, 4),
    )

    stage_runner._run_stage(
        "native_child",
        ["/fake/isaac/python.sh", "--help"],
        None,
        command_runner=runner,
    )

    assert captured == [
        [
            "/usr/bin/taskset",
            "-c",
            "0,2,4",
            "/fake/isaac/python.sh",
            "--help",
        ]
    ]


def test_run_stage_does_not_taskset_local_python_model_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def runner(command, **_kwargs) -> None:
        captured.append(command)

    monkeypatch.setattr(stage_runner, "run_command", runner)
    monkeypatch.setattr(stage_runner, "_TASKSET_EXECUTABLE", "/usr/bin/taskset")
    monkeypatch.setattr(stage_runner, "_VISUAL_CONTROL_CHILD_CPU_AFFINITY", (0, 2))

    stage_runner._run_stage(
        "local_model",
        ["/env/bin/python", "-m", "qwen_material_pipeline"],
        None,
        command_runner=runner,
    )

    assert captured == [["/env/bin/python", "-m", "qwen_material_pipeline"]]


def test_stable_native_child_affinity_keeps_only_hyperthreaded_cores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePath:
        def __init__(self, path: str) -> None:
            self.path = path

        def read_text(self, *, encoding: str) -> str:
            cpu = int(self.path.rsplit("cpu", 1)[1].split("/")[0])
            return {0: "0-1\n", 1: "0-1\n", 2: "2\n", 3: "3\n"}[cpu]

    monkeypatch.setattr(stage_runner, "Path", FakePath)
    assert stage_runner._stable_native_child_cpu_affinity((0, 1, 2, 3)) == (0, 1)


def test_run_stage_reports_failed_retry_and_cumulative_elapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    calls = 0
    times = iter((20.0, 21.25, 23.5))
    monkeypatch.setattr(stage_runner, "monotonic", lambda: next(times))

    def runner(_command, *, log_cb, **_kwargs) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            log_cb("Segmentation fault (core dumped)")
            raise RuntimeError("native crash")

    stage_runner._run_stage(
        "apply_locked",
        ["/fake/isaac/python.sh"],
        messages.append,
        command_runner=runner,
        retry_native_crash=True,
    )

    progress = _progress_messages(messages)
    assert [
        next(
            state
            for state in ("START", "FAILED", "RETRY", "COMPLETE")
            if f" {state} " in message
        )
        for message in progress
    ] == ["START", "FAILED", "RETRY", "COMPLETE"]
    assert "elapsed=1.250s" in progress[1]
    assert "elapsed=1.250s" in progress[2]
    assert "elapsed=3.500s" in progress[3]


def test_run_stage_reports_terminal_failure_without_false_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    times = iter((30.0, 30.625))
    monkeypatch.setattr(stage_runner, "monotonic", lambda: next(times))

    def runner(*_args, **_kwargs) -> None:
        raise RuntimeError("invalid material plan")

    with pytest.raises(RuntimeError, match="Visual material stage failed"):
        stage_runner._run_stage(
            "apply_invalid_plan",
            ["/fake/isaac/python.sh"],
            messages.append,
            command_runner=runner,
            retry_native_crash=True,
        )

    progress = _progress_messages(messages)
    assert progress == [
        "[PROGRESS] visual_materials/apply_invalid_plan START elapsed=0.000s",
        (
            "[PROGRESS] visual_materials/apply_invalid_plan FAILED "
            "elapsed=0.625s attempt=1"
        ),
    ]


def test_run_stage_missing_required_output_is_not_reported_complete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    messages: list[str] = []
    times = iter((40.0, 40.25))
    monkeypatch.setattr(stage_runner, "monotonic", lambda: next(times))

    with pytest.raises(RuntimeError, match="did not create expected file"):
        stage_runner._run_stage(
            "apply_usd",
            ["/fake/isaac/python.sh"],
            messages.append,
            command_runner=lambda *_args, **_kwargs: None,
            required_files=(tmp_path / "look.usda", tmp_path / "apply.json"),
        )

    assert _progress_messages(messages) == [
        "[PROGRESS] visual_materials/apply_usd START elapsed=0.000s",
        "[PROGRESS] visual_materials/apply_usd FAILED elapsed=0.250s attempt=1",
    ]


def test_exact_mdl_candidate_progress_spans_the_global_tournament() -> None:
    messages: list[str] = []

    orchestrator._log_exact_mdl_candidate_progress(
        messages.append,
        state="start",
        group_index=1,
        group_total=4,
        candidate_index=1,
        candidate_total=32,
        candidate_id="candidate_green_01",
        global_current=0,
        global_total=128,
    )
    orchestrator._log_exact_mdl_candidate_progress(
        messages.append,
        state="complete",
        group_index=1,
        group_total=4,
        candidate_index=1,
        candidate_total=32,
        candidate_id="candidate_green_01",
        global_current=1,
        global_total=128,
        cache_status="CACHE_HIT",
    )

    assert " 0.0% " in messages[0]
    assert "group 1/4 candidate 1/32 id=candidate_green_01" in messages[0]
    assert messages[0].endswith("(candidate 0/128)")
    assert "visual_materials.exact_mdl_tournament/candidate COMPLETE" in messages[1]
    assert "cache=CACHE_HIT" in messages[1]
    assert messages[1].endswith("(candidate 1/128)")


def test_exact_mdl_group_progress_spans_all_groups() -> None:
    messages: list[str] = []

    orchestrator._log_exact_mdl_group_progress(
        messages.append,
        state="start",
        current=0,
        total=4,
        group_id="G06",
    )
    orchestrator._log_exact_mdl_group_progress(
        messages.append,
        state="complete",
        current=4,
        total=4,
        group_id="G05",
    )

    assert " 0.0% " in messages[0]
    assert messages[0].endswith("(group 0/4)")
    assert " 100.0% " in messages[1]
    assert messages[1].endswith("(group 4/4)")
