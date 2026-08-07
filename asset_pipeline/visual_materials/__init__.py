"""Automatic visual-material assignment subsystem.

Call flow::

    manual_cad
      -> run_assign_visual_materials_job       (orchestrator: stage order)
         -> context                            (validated immutable run state)
         -> workspace                          (typed artifact paths)
         -> stages.source_preparation          (registry/render/camera evidence)
         -> stages.material_inference          (Qwen/MVInverse + recovery)
         -> orchestrator                       (policy/selection/Look coordination)
         -> stages.final_acceptance             (collected-USD visual gate)
         -> stages.runner                       (subprocess/retry/progress)
         -> commands                            (subprocess argv only)
         -> qwen_material_pipeline CLI         (runtime implementations)

Configuration and reference parsing also have dedicated owner modules.
Existing callers may continue to import :mod:`asset_pipeline.jobs.material`
during the compatibility period; that module contains no pipeline policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..command import LogCallback

from .config import (
    CONFIG_SCHEMA_VERSION,
    DEFAULT_CONFIG_PATH,
    VisualMaterialConfig,
    load_visual_material_config,
)
from .contracts import ISOLATED_ENV_REMOVE, RESULT_SCHEMA_VERSION
from .references import parse_visual_references


def run_assign_visual_materials_job(
    *,
    source_usd: str,
    source_cad: str | None = None,
    references: Sequence[str],
    foreground_annotations: str | None = None,
    output_dir: str | None = None,
    config_path: str | None = None,
    inference_mode: str = "live",
    acknowledge_mvinverse_noncommercial: bool = False,
    allow_policy_material_fallback: bool = False,
    require_complete_coverage: bool = False,
    log_cb: LogCallback = None,
) -> dict[str, Any]:
    """Load the heavy material orchestrator only when assignment starts."""

    from .orchestrator import run_assign_visual_materials_job as implementation

    return implementation(
        source_usd=source_usd,
        source_cad=source_cad,
        references=references,
        foreground_annotations=foreground_annotations,
        output_dir=output_dir,
        config_path=config_path,
        inference_mode=inference_mode,
        acknowledge_mvinverse_noncommercial=acknowledge_mvinverse_noncommercial,
        allow_policy_material_fallback=allow_policy_material_fallback,
        require_complete_coverage=require_complete_coverage,
        log_cb=log_cb,
    )


def run_final_visual_acceptance_job(
    *,
    collected_usd: str,
    visual_material_result: dict[str, Any],
    output_dir: str | None = None,
    config_path: str | None = None,
    log_cb: LogCallback = None,
) -> dict[str, Any]:
    """Load the heavy material orchestrator only when final QA starts."""

    from .orchestrator import run_final_visual_acceptance_job as implementation

    return implementation(
        collected_usd=collected_usd,
        visual_material_result=visual_material_result,
        output_dir=output_dir,
        config_path=config_path,
        log_cb=log_cb,
    )


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "DEFAULT_CONFIG_PATH",
    "ISOLATED_ENV_REMOVE",
    "RESULT_SCHEMA_VERSION",
    "VisualMaterialConfig",
    "load_visual_material_config",
    "parse_visual_references",
    "run_assign_visual_materials_job",
    "run_final_visual_acceptance_job",
]
