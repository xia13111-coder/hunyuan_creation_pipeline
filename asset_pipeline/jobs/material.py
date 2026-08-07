"""Compatibility facade for :mod:`asset_pipeline.visual_materials`.

New code should import the owner package directly. This module intentionally
keeps the previous symbols and monkeypatch points while forwarding all real
work to the split visual-material subsystem.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..command import LogCallback, run_command
from ..runtime import isaac_python
from ..visual_materials.config import (
    CONFIG_SCHEMA_VERSION,
    DEFAULT_CONFIG_PATH,
    VisualMaterialConfig,
    canonical_sha256 as _canonical_sha256,
    load_visual_material_config,
    read_object as _read_object,
    require_object as _object,
    require_positive_int as _positive_int,
    require_string as _string,
    resolve_path as _resolve_path,
    write_object as _write_object,
)
from ..visual_materials.orchestrator import (
    APPLICABLE_ASSIGNMENT_STATUSES,
    ISOLATED_ENV_REMOVE,
    RESULT_SCHEMA_VERSION,
    USD_SUFFIXES,
    _require_file,
    _run_stage as _run_owned_stage,
    run_assign_visual_materials_job as _run_assign_visual_materials_job,
    run_final_visual_acceptance_job as _run_final_visual_acceptance_job,
)
from ..visual_materials.references import (
    IMAGE_SUFFIXES,
    REFERENCE_ID,
    parse_visual_references,
    sha256_file as _sha256,
)
from .delivery import run_validate_visual_material_delivery_job


def _run_stage(name: str, command: list[str], log_cb: LogCallback) -> None:
    """Compatibility wrapper preserving the patchable ``run_command``."""

    _run_owned_stage(
        name,
        command,
        log_cb,
        command_runner=run_command,
    )


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
    """Forward to the owner orchestrator with legacy patch points injected."""

    return _run_assign_visual_materials_job(
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
        _command_runner=run_command,
        _isaac_python_resolver=isaac_python,
        _config_loader=load_visual_material_config,
        _reference_parser=parse_visual_references,
        _default_config_path=DEFAULT_CONFIG_PATH,
    )


def run_final_visual_acceptance_job(
    *,
    collected_usd: str,
    visual_material_result: dict[str, Any],
    output_dir: str | None = None,
    config_path: str | None = None,
    log_cb: LogCallback = None,
) -> dict[str, Any]:
    """Forward final delivery QA while preserving legacy patch points."""

    return _run_final_visual_acceptance_job(
        collected_usd=collected_usd,
        visual_material_result=visual_material_result,
        output_dir=output_dir,
        config_path=config_path,
        log_cb=log_cb,
        _command_runner=run_command,
        _isaac_python_resolver=isaac_python,
        _config_loader=load_visual_material_config,
    )


__all__ = [
    "APPLICABLE_ASSIGNMENT_STATUSES",
    "CONFIG_SCHEMA_VERSION",
    "DEFAULT_CONFIG_PATH",
    "IMAGE_SUFFIXES",
    "ISOLATED_ENV_REMOVE",
    "REFERENCE_ID",
    "RESULT_SCHEMA_VERSION",
    "USD_SUFFIXES",
    "VisualMaterialConfig",
    "_canonical_sha256",
    "_object",
    "_positive_int",
    "_read_object",
    "_require_file",
    "_resolve_path",
    "_sha256",
    "_string",
    "_write_object",
    "load_visual_material_config",
    "parse_visual_references",
    "run_assign_visual_materials_job",
    "run_final_visual_acceptance_job",
    "run_validate_visual_material_delivery_job",
]
