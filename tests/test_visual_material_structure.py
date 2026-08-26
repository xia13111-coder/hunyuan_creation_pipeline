from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from asset_pipeline.visual_materials.commands import (
    cad_mesh_template_command,
    camera_registration_command,
    policy_exact_cover_command,
    staged_material_command,
    usd_expand_instances_command,
    usd_registry_command,
    usd_render_command,
)
from asset_pipeline.visual_materials.camera import (
    continuous_camera_view_specs,
    require_complete_live_camera_alignment,
)
from asset_pipeline.visual_materials.quality import evaluate_part_id_quality_gate
from asset_pipeline.visual_materials.stages import part_id_evidence
from asset_pipeline.visual_materials.stages.part_id_evidence import (
    _entityseg_region_command,
    _hybrid_mask_command,
    _require_complete_reference_views,
    _sam3_region_command,
)
from asset_pipeline.visual_materials.workspace import VisualMaterialWorkspace


def test_workspace_centralizes_the_existing_artifact_layout(tmp_path: Path) -> None:
    destination = tmp_path / "visual_material"
    source = tmp_path / "machine.usd"

    workspace = VisualMaterialWorkspace.create(
        destination=destination,
        source=source,
    )

    assert workspace.source.source_registry == (
        destination / "source_part_registry.json"
    )
    assert workspace.source.editable_usd == destination / "machine_editable.usda"
    assert workspace.inference.policy_plan == (
        destination / "analysis" / "policy_exact_cover_plan.json"
    )
    assert workspace.look.locked_usd == destination / "machine_look_locked.usda"
    assert workspace.quality.report == (
        destination / "visual_quality" / "reference_render_comparison.json"
    )
    assert workspace.part_id.material_audit == (
        destination / "analysis" / "part_id_material_audit.json"
    )
    assert workspace.part_id.amodal_template_manifest == (
        destination / "analysis" / "part_id_cad_amodal_templates" / "manifest.json"
    )
    assert workspace.part_id.initial_hybrid_mask_manifest == (
        destination / "analysis" / "part_id_hybrid_masks_initial" / "manifest.json"
    )
    assert workspace.part_id.relation_guided_request == (
        destination / "analysis" / "part_id_relation_guidance" / "request.json"
    )
    assert workspace.part_id.relation_sam3_manifest == (
        destination / "analysis" / "part_id_relation_sam3_regions" / "manifest.json"
    )
    assert workspace.part_id.relation_entityseg_manifest == (
        destination
        / "analysis"
        / "part_id_relation_entityseg_regions"
        / "manifest.json"
    )
    assert workspace.source.camera_acceptance == (
        destination / "camera_calibration" / "camera_alignment_acceptance.json"
    )
    assert not destination.exists(), "building the path map must have no side effects"

    with pytest.raises(FrozenInstanceError):
        workspace.destination = tmp_path  # type: ignore[misc]


def test_usd_command_builders_keep_the_cli_contract() -> None:
    isaac = Path("/runtime/python.sh")

    assert usd_registry_command(
        isaac_python=isaac,
        usd=Path("/run/source.usd"),
        output=Path("/run/registry.json"),
    ) == [
        "/runtime/python.sh",
        "-m",
        "qwen_material_pipeline",
        "usd",
        "registry",
        "--usd",
        "/run/source.usd",
        "--output",
        "/run/registry.json",
    ]
    assert usd_expand_instances_command(
        isaac_python=isaac,
        source_usd=Path("/run/source.usd"),
        output_usd=Path("/run/editable.usda"),
        report=Path("/run/expand.json"),
    ) == [
        "/runtime/python.sh",
        "-m",
        "qwen_material_pipeline",
        "usd",
        "expand",
        "--source-usd",
        "/run/source.usd",
        "--output-usd",
        "/run/editable.usda",
        "--report",
        "/run/expand.json",
    ]


def test_render_and_camera_builders_handle_optional_arguments() -> None:
    render = usd_render_command(
        isaac_python=Path("/runtime/python.sh"),
        registry=Path("/run/registry.json"),
        output_dir=Path("/run/renders"),
        resolution=512,
        views="26",
        rt_subframes=4,
        analysis_up_axis="z",
        analysis_front_axis="-y",
        rgb_only=True,
    )
    assert render[-2:] == ["--analysis-front-axis=-y", "--rgb-only"]

    camera = camera_registration_command(
        python=Path("/models/python"),
        registry=Path("/run/registry.json"),
        reference_manifest=Path("/run/annotations.json"),
        isaac_python=Path("/runtime/python.sh"),
        output_dir=Path("/run/camera"),
        search_resolution=256,
        final_resolution=512,
        rt_subframes=4,
        analysis_up_axis="z",
        analysis_front_axis="-y",
        initial_view_specs=Path("/run/camera/search/final_view_specs.json"),
        search_phases=("orthographic", "micro", "pico"),
    )
    assert camera[:5] == [
        "/models/python",
        "-m",
        "qwen_material_pipeline",
        "calibrate-cameras",
        "--registry",
    ]
    assert camera[camera.index("--initial-view-specs") + 1] == (
        "/run/camera/search/final_view_specs.json"
    )
    assert camera[camera.index("--search-phases") + 1] == ("orthographic,micro,pico")
    assert camera[camera.index("--fast-search") + 1] == "auto"
    assert camera[-1] == "--analysis-front-axis=-y"

    templates = cad_mesh_template_command(
        isaac_python=Path("/runtime/python.sh"),
        registry=Path("/run/registry.json"),
        spatial_report=Path("/run/spatial.json"),
        evidence=Path("/run/coarse-evidence.json"),
        output_dir=Path("/run/amodal"),
    )
    assert templates[0] == "/runtime/python.sh"
    assert templates[1].endswith(
        "/tools/qwen_material_pipeline/segmentation/cad_mesh_templates.py"
    )
    assert templates[templates.index("--registry") + 1] == "/run/registry.json"
    assert templates[templates.index("--evidence") + 1] == (
        "/run/coarse-evidence.json"
    )
    assert templates[-2:] == ["--output-dir", "/run/amodal"]


def test_part_id_evidence_commands_keep_the_two_pass_runtime_contract() -> None:
    config = SimpleNamespace(
        sam3_python=Path("/runtime/sam3-python"),
        sam3_repository=Path("/models/sam3"),
        sam3_checkpoint=Path("/models/sam3.pt"),
        sam3_device="cuda",
        sam3_minimum_model_score=0.45,
        sam3_minimum_prompt_overlap=0.25,
        sam3_maximum_image_fraction=0.8,
        sam3_minimum_mask_pixels=32,
        entityseg_python=Path("/runtime/entityseg-python"),
        entityseg_cropformer_root=Path("/models/CropFormer"),
        entityseg_config=Path("/models/entityseg.yaml"),
        entityseg_checkpoint=Path("/models/entityseg.pth"),
        entityseg_minimum_model_score=0.3,
    )
    request = Path("/run/relation/request.json")

    sam3 = _sam3_region_command(
        config,
        request=request,
        output_dir=Path("/run/relation/sam3"),
    )
    assert sam3[0] == "/runtime/sam3-python"
    assert sam3[1].endswith("/segmentation/sam3_regions.py")
    assert sam3[sam3.index("--request") + 1] == str(request)
    assert sam3[-2:] == ["--seed", "0"]

    entityseg = _entityseg_region_command(
        config,
        request=request,
        output_dir=Path("/run/relation/entityseg"),
    )
    assert entityseg[:3] == [
        "/runtime/entityseg-python",
        "-m",
        "qwen_material_pipeline.segmentation.entityseg_regions",
    ]
    assert entityseg[entityseg.index("--request") + 1] == str(request)
    assert entityseg[-2:] == ["--seed", "0"]

    hybrid = _hybrid_mask_command(
        config,
        sam_manifest=Path("/run/relation/sam3/manifest.json"),
        entityseg_manifest=Path("/run/relation/entityseg/manifest.json"),
        amodal_manifest=Path("/run/amodal/manifest.json"),
        prior_hybrid_manifest=Path("/run/initial/manifest.json"),
        output_dir=Path("/run/final"),
    )
    assert hybrid[:3] == [
        "/runtime/sam3-python",
        "-m",
        "qwen_material_pipeline.segmentation.hybrid_part_masks",
    ]
    assert hybrid[hybrid.index("--prior-hybrid-manifest") + 1] == (
        "/run/initial/manifest.json"
    )
    assert hybrid[-2:] == ["--output-dir", "/run/final"]


def test_part_id_evidence_requires_every_registered_view() -> None:
    evidence = {
        "summary": {
            "trusted_reference_view_count": 2,
            "selected_reference_view_coverage": {
                "front": {"visible_part_count": 1, "selected_part_count": 1},
                "side": {"visible_part_count": 1, "selected_part_count": 1},
            },
        },
        "parts": [
            {
                "part_id": "P0001",
                "observations": [
                    {
                        "view_id": "front",
                        "selected_for_material_inference": True,
                    },
                    {
                        "view_id": "side",
                        "selected_for_material_inference": True,
                    },
                ],
            }
        ],
    }

    _require_complete_reference_views(
        evidence=evidence,
        expected_view_ids={"front", "side"},
        label="refined Part-ID evidence",
    )
    evidence["parts"][0]["observations"][1]["selected_for_material_inference"] = False
    with pytest.raises(RuntimeError, match="do not use every registered"):
        _require_complete_reference_views(
            evidence=evidence,
            expected_view_ids={"front", "side"},
            label="refined Part-ID evidence",
        )


@pytest.mark.parametrize("entityseg_enabled", [False, True])
def test_part_id_evidence_stage_selects_the_expected_final_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entityseg_enabled: bool,
) -> None:
    destination = tmp_path / "visual_material"
    workspace = VisualMaterialWorkspace.create(
        destination=destination,
        source=tmp_path / "source.usd",
    )
    config = SimpleNamespace(
        material_prediction_mode="catalog_family_first",
        entityseg_enabled=entityseg_enabled,
        sam3_python=Path("/runtime/sam3-python"),
        sam3_repository=Path("/models/sam3"),
        sam3_checkpoint=Path("/models/sam3.pt"),
        sam3_device="cuda",
        sam3_minimum_model_score=0.45,
        sam3_minimum_prompt_overlap=0.25,
        sam3_maximum_image_fraction=0.8,
        sam3_minimum_mask_pixels=32,
        entityseg_python=Path("/runtime/entityseg-python"),
        entityseg_cropformer_root=Path("/models/CropFormer"),
        entityseg_config=Path("/models/entityseg.yaml"),
        entityseg_checkpoint=Path("/models/entityseg.pth"),
        entityseg_minimum_model_score=0.3,
    )
    context = SimpleNamespace(
        config=config,
        workspace=workspace,
        references=(("front", tmp_path / "front.png"),),
        isaac_python=Path("/runtime/isaac-python"),
    )
    stage_names: list[str] = []
    stage_commands: dict[str, list[str]] = {}
    evidence_calls: list[dict[str, object]] = []

    def fake_stage(
        name: str,
        command: list[str],
        _log_cb: object,
        **kwargs: object,
    ) -> None:
        stage_names.append(name)
        stage_commands[name] = command
        required = kwargs.get("required_files", ())
        assert isinstance(required, tuple)
        for path in required:
            assert isinstance(path, Path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")

    def fake_evidence(**kwargs: object) -> dict[str, object]:
        evidence_calls.append(kwargs)
        return {
            "parts": [
                {
                    "part_id": "P0001",
                    "observations": [
                        {
                            "view_id": "front",
                            "selected_for_material_inference": True,
                        }
                    ],
                }
            ],
            "summary": {
                "trusted_reference_view_count": 1,
                "selected_reference_view_coverage": {
                    "front": {"visible_part_count": 1, "selected_part_count": 1}
                },
            },
        }

    monkeypatch.setattr(part_id_evidence, "_run_stage", fake_stage)
    monkeypatch.setattr(
        part_id_evidence,
        "build_part_id_reference_evidence",
        fake_evidence,
    )
    monkeypatch.setattr(
        part_id_evidence,
        "build_part_id_sam3_request",
        lambda *_args, **_kwargs: {"regions": []},
    )
    monkeypatch.setattr(
        part_id_evidence,
        "build_relation_guided_request",
        lambda **_kwargs: {"regions": []},
    )
    monkeypatch.setattr(part_id_evidence, "log_message", lambda *_args: None)

    result = part_id_evidence.run_part_id_evidence_stage(
        context,  # type: ignore[arg-type]
        rendered_registry=tmp_path / "rendered_registry.json",
        mvinverse_ledger=tmp_path / "mvinverse.json",
        log_cb=lambda _message: None,
        command_runner=lambda *_args, **_kwargs: None,
    )

    assert len(evidence_calls) == 2
    assert result["parts"][0]["part_id"] == "P0001"
    final_call = evidence_calls[-1]
    if entityseg_enabled:
        assert stage_names == [
            "part_id_cad_amodal_templates",
            "part_id_sam3_local_refinement",
            "part_id_entityseg_boundary_candidates",
            "part_id_initial_sam3_entityseg_fusion",
            "part_id_relation_guided_sam3_refinement",
            "part_id_relation_guided_entityseg_boundaries",
            "part_id_relation_guided_iterative_fusion",
        ]
        assert final_call["part_id_sam3_manifest"] is None
        assert (
            final_call["part_id_hybrid_manifest"]
            == workspace.part_id.hybrid_mask_manifest
        )
        assert (
            "--prior-hybrid-manifest"
            in stage_commands["part_id_relation_guided_iterative_fusion"]
        )
    else:
        assert stage_names == [
            "part_id_cad_amodal_templates",
            "part_id_sam3_local_refinement",
        ]
        assert final_call["part_id_sam3_manifest"] == workspace.part_id.sam3_manifest
        assert final_call["part_id_hybrid_manifest"] is None


def test_staged_material_command_owns_the_model_runtime_contract() -> None:
    config = SimpleNamespace(
        qwen_python=Path("/runtime/qwen-python"),
        material_root=Path("/materials/Base"),
        qwen_model_family="local_qwen",
        qwen_max_new_tokens=512,
        qwen_max_new_tokens_ceiling=1024,
        qwen_minimum_usable_palette_views=2,
        qwen_minimum_usable_palette_view_ratio=0.5,
        qwen_mapping_verification_views=2,
        qwen_parallel_requests=1,
        material_assignment_unit="part_id",
        sam3_python=Path("/runtime/sam3-python"),
        sam3_repository=Path("/models/sam3"),
        sam3_checkpoint=Path("/models/sam3.pt"),
        sam3_device="cuda",
        sam3_minimum_model_score=0.5,
        sam3_minimum_prompt_overlap=0.2,
        sam3_maximum_image_fraction=0.95,
        sam3_minimum_mask_pixels=32,
        retrieval_python=Path("/runtime/retrieval-python"),
        siglip2_model_path=Path("/models/siglip2"),
        dinov2_model_path=Path("/models/dinov2"),
        retrieval_cache_dir=Path("/cache/retrieval"),
        retrieval_device="cuda",
        siglip_top_k=64,
        retrieval_final_top_k=32,
        retrieval_batch_size=8,
        retrieval_observation_bank_dir=Path("/cache/observations"),
        mvinverse_mode="run",
        mvinverse_repository=Path("/models/mvinverse/repo"),
        mvinverse_python=Path("/runtime/mvinverse-python"),
        mvinverse_checkpoint=Path("/models/mvinverse/checkpoint"),
        mvinverse_model_revision="revision-1",
        mvinverse_device="cuda",
        mvinverse_max_side=448,
        mvinverse_timeout_seconds=1800,
        mvinverse_oom_retry_max_sides=(392, 336),
        immutable_mdl_after_selection=True,
        exact_mdl_tournament_max_candidates=32,
        material_selection_objective="visual_similarity",
        qwen_model_path=Path("/models/qwen"),
        qwen_model_revision="revision-2",
        openai_base_url=None,
        openai_model=None,
        openai_api_key_env=None,
        openai_reasoning_effort=None,
        openai_timeout_seconds=None,
    )
    command = staged_material_command(
        config=config,  # type: ignore[arg-type]
        registry=Path("/run/registry.json"),
        references=(("front", Path("/photos/front.jpg")),),
        foreground_annotations=Path("/run/foreground.json"),
        catalog=Path("/run/catalog.json"),
        whitelist=Path("/run/allowlist.json"),
        output_dir=Path("/run/analysis"),
        isaac_python=Path("/runtime/isaac-python"),
    )

    assert command[:4] == [
        "/runtime/qwen-python",
        "-m",
        "qwen_material_pipeline",
        "staged",
    ]
    assert command[command.index("--reference") + 1] == ("front=/photos/front.jpg")
    assert command[command.index("--model-path") + 1] == "/models/qwen"
    assert command.count("--mvinverse-oom-retry-max-side") == 2
    assert "--immutable-mdl-after-selection" in command


def test_policy_command_is_a_small_deterministic_contract() -> None:
    command = policy_exact_cover_command(
        python=Path("/runtime/python"),
        registry=Path("/run/registry.json"),
        staged_result=Path("/run/staged.json"),
        confidence_gate=Path("/run/gate.json"),
        whitelist=Path("/run/allowlist.json"),
        base_plan=Path("/run/base.json"),
        group_materials=Path("/run/groups.json"),
        mvinverse_pbr_evidence=Path("/run/pbr.json"),
        palette_fusion=Path("/run/palette.json"),
        policy=Path("/run/policy.json"),
        output_plan=Path("/run/output.json"),
        audit=Path("/run/audit.json"),
        immutable_mdl_after_selection=True,
    )

    assert command[:4] == [
        "/runtime/python",
        "-m",
        "qwen_material_pipeline",
        "policy-exact-cover",
    ]
    assert command[-2:] == [
        "--acknowledge-policy-fallback",
        "--immutable-mdl-after-selection",
    ]


def test_camera_contract_reconstructs_specs_and_tiers_alignment() -> None:
    registry = {
        "render_set": {
            "requested_view_tokens": ["front"],
            "views": [
                {
                    "view_id": "front",
                    "analysis_direction": [0.0, -1.0, 0.0],
                    "analysis_camera_up_axis": [0.0, 0.0, 1.0],
                    "focal_length_mm": 55.0,
                    "camera_distance_multiplier": 1.2,
                    "camera_calibration": {
                        "target_offset_u": 0.1,
                        "target_offset_v": -0.1,
                        "projection_mode": "perspective",
                        "orthographic_span_multiplier": 2.0,
                    },
                }
            ],
        }
    }
    specs = continuous_camera_view_specs(registry)
    assert specs is not None
    assert specs["views"][0]["focal_length_mm"] == 55.0

    acceptance = require_complete_live_camera_alignment(
        {
            "views": [
                {
                    "reference_view_id": "front",
                    "complete_alignment_passed": False,
                    "complete_alignment_target": {},
                    "final": {
                        "projection_iou": 0.93,
                        "boundary_p95_px": 8.0,
                    },
                }
            ]
        },
        expected_reference_ids={"front"},
    )
    assert acceptance["views"]["front"]["tier"] == ("usable_box_correspondence")


def test_part_id_quality_gate_ignores_only_obsolete_palette_failures() -> None:
    report = {
        "aggregate": {
            "status": "FAIL",
            "comparable_view_count": 2,
            "material_appearance_score": 0.8,
        },
        "views": [
            {
                "reference_view_id": "front",
                "render_view_id": "render_front",
                "status": "FAIL",
                "material_appearance_score": 0.8,
                "reasons": ["trusted_palette_group_missing_from_render"],
            },
            {
                "reference_view_id": "side",
                "render_view_id": "render_side",
                "status": "PASS",
                "material_appearance_score": 0.75,
                "reasons": [],
            },
        ],
    }
    accepted = evaluate_part_id_quality_gate(
        report,
        minimum_aggregate_appearance_score=0.7,
        minimum_view_appearance_score=0.7,
    )
    assert accepted["status"] == "PASS"

    report["views"][0]["reasons"].append("multiple_aligned_views_confirm_mismatch")
    rejected = evaluate_part_id_quality_gate(
        report,
        minimum_aggregate_appearance_score=0.7,
        minimum_view_appearance_score=0.7,
    )
    assert rejected["status"] == "FAIL_CLOSED"
    assert "NON_PALETTE_VIEW_FAILURE_REASONS_PRESENT" in rejected["reason_codes"]
