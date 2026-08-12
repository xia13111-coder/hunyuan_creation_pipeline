from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from asset_pipeline.visual_materials.commands import (
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
    assert camera[camera.index("--render-backend") + 1] == "supervisor"
    assert camera[-1] == "--analysis-front-axis=-y"

    inprocess = camera_registration_command(
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
        render_backend="inprocess",
    )
    assert inprocess[0] == "/runtime/python.sh"
    assert inprocess[inprocess.index("--render-backend") + 1] == "inprocess"

    legacy = camera_registration_command(
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
        render_backend="subprocess",
    )
    assert legacy[0] == "/models/python"
    assert legacy[legacy.index("--render-backend") + 1] == "subprocess"


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
