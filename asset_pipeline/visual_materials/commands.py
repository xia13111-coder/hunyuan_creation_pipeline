"""Pure command builders for cross-runtime visual-material stages.

Keeping argv construction here makes the orchestrator read as a stage graph and
lets tests validate command contracts without starting Isaac Sim or model
runtimes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from ..project_layout import SOURCE_LAYOUT

if TYPE_CHECKING:
    from .config import VisualMaterialConfig


def usd_registry_command(
    *,
    isaac_python: Path,
    usd: Path,
    output: Path,
) -> list[str]:
    return [
        str(isaac_python),
        "-m",
        "qwen_material_pipeline",
        "usd",
        "registry",
        "--usd",
        str(usd),
        "--output",
        str(output),
    ]


def usd_expand_instances_command(
    *,
    isaac_python: Path,
    source_usd: Path,
    output_usd: Path,
    report: Path,
) -> list[str]:
    return [
        str(isaac_python),
        "-m",
        "qwen_material_pipeline",
        "usd",
        "expand",
        "--source-usd",
        str(source_usd),
        "--output-usd",
        str(output_usd),
        "--report",
        str(report),
    ]


def usd_render_command(
    *,
    isaac_python: Path,
    registry: Path,
    output_dir: Path,
    resolution: int,
    views: str,
    rt_subframes: int,
    analysis_up_axis: str,
    analysis_front_axis: str,
    rgb_only: bool = False,
) -> list[str]:
    command = [
        str(isaac_python),
        "-m",
        "qwen_material_pipeline",
        "usd",
        "render",
        "--registry",
        str(registry),
        "--output-dir",
        str(output_dir),
        "--resolution",
        str(resolution),
        "--views",
        views,
        "--rt-subframes",
        str(rt_subframes),
        "--analysis-up-axis",
        analysis_up_axis,
        f"--analysis-front-axis={analysis_front_axis}",
    ]
    if rgb_only:
        command.append("--rgb-only")
    return command


def camera_registration_command(
    *,
    python: Path,
    registry: Path,
    reference_manifest: Path,
    isaac_python: Path,
    output_dir: Path,
    search_resolution: int,
    final_resolution: int,
    rt_subframes: int,
    analysis_up_axis: str,
    analysis_front_axis: str,
    initial_view_specs: Path | None = None,
    search_phases: Sequence[str] = (),
    fast_search_mode: str = "auto",
) -> list[str]:
    command = [
        str(python),
        "-m",
        "qwen_material_pipeline",
        "calibrate-cameras",
        "--registry",
        str(registry),
        "--reference-manifest",
        str(reference_manifest),
    ]
    if initial_view_specs is not None:
        command.extend(["--initial-view-specs", str(initial_view_specs)])
    if search_phases:
        command.extend(["--search-phases", ",".join(search_phases)])
    command.extend(
        [
            "--isaac-python",
            str(isaac_python),
            "--output-dir",
            str(output_dir),
            "--search-resolution",
            str(search_resolution),
            "--final-resolution",
            str(final_resolution),
            "--rt-subframes",
            str(rt_subframes),
            "--fast-search",
            fast_search_mode,
            "--analysis-up-axis",
            analysis_up_axis,
            f"--analysis-front-axis={analysis_front_axis}",
        ]
    )
    return command


def cad_mesh_template_command(
    *,
    isaac_python: Path,
    registry: Path,
    spatial_report: Path,
    evidence: Path,
    output_dir: Path,
) -> list[str]:
    """Build isolated-mesh amodal masks with the sealed whole-asset cameras."""

    return [
        str(isaac_python),
        str(
            SOURCE_LAYOUT.material_package
            / "segmentation"
            / "cad_mesh_templates.py"
        ),
        "--registry",
        str(registry),
        "--spatial-report",
        str(spatial_report),
        "--evidence",
        str(evidence),
        "--output-dir",
        str(output_dir),
    ]


def staged_material_command(
    *,
    config: "VisualMaterialConfig",
    registry: Path,
    references: Sequence[tuple[str, Path]],
    foreground_annotations: Path | None,
    catalog: Path,
    whitelist: Path,
    output_dir: Path,
    isaac_python: Path,
) -> list[str]:
    """Build the complete Qwen/SAM3/retrieval/MVInverse stage command."""

    command = [
        str(config.qwen_python),
        "-m",
        "qwen_material_pipeline",
        "staged",
        "--registry",
        str(registry),
    ]
    for reference_id, path in references:
        command.extend(["--reference", f"{reference_id}={path}"])
    if foreground_annotations is not None:
        command.extend(["--sam3-foreground-annotations", str(foreground_annotations)])
    command.extend(
        [
            "--catalog",
            str(catalog),
            "--material-root",
            str(config.material_root),
            "--whitelist",
            str(whitelist),
            "--qwen-model-family",
            config.qwen_model_family,
            "--max-new-tokens",
            str(config.qwen_max_new_tokens),
            "--max-new-tokens-ceiling",
            str(config.qwen_max_new_tokens_ceiling),
            "--minimum-usable-palette-views",
            str(config.qwen_minimum_usable_palette_views),
            "--minimum-usable-palette-view-ratio",
            str(config.qwen_minimum_usable_palette_view_ratio),
            "--mapping-verification-views",
            str(config.qwen_mapping_verification_views),
            "--remote-parallel-requests",
            str(config.qwen_parallel_requests),
            "--output-dir",
            str(output_dir),
            "--stop-after",
            "materials",
            "--material-assignment-unit",
            config.material_assignment_unit,
            "--sam3-python",
            str(config.sam3_python),
            "--sam3-repo",
            str(config.sam3_repository),
            "--sam3-checkpoint",
            str(config.sam3_checkpoint),
            "--sam3-device",
            config.sam3_device,
            "--sam3-minimum-model-score",
            str(config.sam3_minimum_model_score),
            "--sam3-minimum-prompt-overlap",
            str(config.sam3_minimum_prompt_overlap),
            "--sam3-maximum-image-fraction",
            str(config.sam3_maximum_image_fraction),
            "--sam3-minimum-mask-pixels",
            str(config.sam3_minimum_mask_pixels),
            "--retrieval-python",
            str(config.retrieval_python),
            "--siglip2-model",
            str(config.siglip2_model_path),
            "--dinov2-model",
            str(config.dinov2_model_path),
            "--retrieval-cache-dir",
            str(config.retrieval_cache_dir),
            "--retrieval-device",
            config.retrieval_device,
            "--siglip-top-k",
            str(config.siglip_top_k),
            "--retrieval-final-top-k",
            str(config.retrieval_final_top_k),
            "--retrieval-batch-size",
            str(config.retrieval_batch_size),
            "--mvinverse-mode",
            config.mvinverse_mode,
            "--mvinverse-repo",
            str(config.mvinverse_repository),
            "--mvinverse-python",
            str(config.mvinverse_python),
            "--mvinverse-checkpoint",
            str(config.mvinverse_checkpoint),
            "--mvinverse-model-revision",
            config.mvinverse_model_revision,
            "--mvinverse-device",
            config.mvinverse_device,
            "--mvinverse-max-side",
            str(config.mvinverse_max_side),
            "--mvinverse-timeout-seconds",
            str(config.mvinverse_timeout_seconds),
            "--acknowledge-mvinverse-noncommercial",
            "--face-region-python",
            str(isaac_python),
        ]
    )
    if config.retrieval_observation_bank_dir is not None:
        command.extend(
            [
                "--retrieval-observation-bank",
                str(config.retrieval_observation_bank_dir),
            ]
        )
    if config.qwen_model_family == "openai_compatible":
        remote_fields = (
            config.openai_base_url,
            config.openai_model,
            config.openai_api_key_env,
            config.openai_reasoning_effort,
            config.openai_timeout_seconds,
        )
        if any(value is None for value in remote_fields):
            raise RuntimeError(
                "OpenAI-compatible visual-model configuration is incomplete"
            )
        command.extend(
            [
                "--openai-base-url",
                str(config.openai_base_url),
                "--openai-model",
                str(config.openai_model),
                "--openai-api-key-env",
                str(config.openai_api_key_env),
                "--openai-reasoning-effort",
                str(config.openai_reasoning_effort),
                "--openai-timeout-seconds",
                str(config.openai_timeout_seconds),
            ]
        )
    else:
        if config.qwen_model_path is None or config.qwen_model_revision is None:
            raise RuntimeError("Local Qwen visual-model configuration is incomplete")
        command.extend(
            [
                "--model-path",
                str(config.qwen_model_path),
                "--qwen-model-revision",
                config.qwen_model_revision,
            ]
        )
    for retry_side in config.mvinverse_oom_retry_max_sides:
        command.extend(["--mvinverse-oom-retry-max-side", str(retry_side)])
    if config.immutable_mdl_after_selection:
        command.extend(
            [
                "--immutable-mdl-after-selection",
                "--exact-mdl-tournament-max-candidates",
                str(config.exact_mdl_tournament_max_candidates),
                "--material-selection-objective",
                config.material_selection_objective,
            ]
        )
    return command


def policy_exact_cover_command(
    *,
    python: Path,
    registry: Path,
    staged_result: Path,
    confidence_gate: Path,
    whitelist: Path,
    base_plan: Path,
    group_materials: Path,
    mvinverse_pbr_evidence: Path,
    palette_fusion: Path,
    policy: Path,
    output_plan: Path,
    audit: Path,
    immutable_mdl_after_selection: bool,
) -> list[str]:
    """Build the deterministic exact-cover fallback compiler command."""

    command = [
        str(python),
        "-m",
        "qwen_material_pipeline",
        "policy-exact-cover",
        "--registry",
        str(registry),
        "--staged-result",
        str(staged_result),
        "--confidence-gate",
        str(confidence_gate),
        "--whitelist",
        str(whitelist),
        "--base-plan",
        str(base_plan),
        "--group-materials",
        str(group_materials),
        "--mvinverse-pbr-evidence",
        str(mvinverse_pbr_evidence),
        "--palette-fusion",
        str(palette_fusion),
        "--policy",
        str(policy),
        "--output-plan",
        str(output_plan),
        "--audit",
        str(audit),
        "--acknowledge-policy-fallback",
    ]
    if immutable_mdl_after_selection:
        command.append("--immutable-mdl-after-selection")
    return command


__all__ = [
    "camera_registration_command",
    "policy_exact_cover_command",
    "staged_material_command",
    "usd_expand_instances_command",
    "usd_registry_command",
    "usd_render_command",
]
