"""Qwen/MVInverse material inference and verified recovery stage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...command import LogCallback, log_message
from ..commands import staged_material_command
from ..config import read_object, write_object
from ..context import VisualMaterialPipelineContext
from ..references import sha256_file
from .common import require_file
from .runner import _run_stage
from qwen_material_pipeline.core.material_stage_contract import (
    material_stage_contract_document,
)
from qwen_material_pipeline.materials.catalog import build_catalog


CommandRunner = Callable[..., None]


def _prepare_live_material_catalog(
    *,
    material_root: Path,
    configured_catalog: Path,
    configured_whitelist: Path,
    analysis_dir: Path,
    log_cb: LogCallback,
) -> tuple[Path, Path, int | None]:
    """Build a run-local exact catalog from the configured NVIDIA MDL tree.

    A checked-in catalog cannot represent a different Isaac asset mount or a
    broader root such as ``NVIDIA/Materials``.  Rebuilding beneath the trusted
    configured root keeps every candidate path constrained while allowing the
    generic workflow to use Base and vMaterials_2 together.  Empty synthetic
    roots retain the configured fixtures used by compatibility tests.
    """

    catalog = build_catalog(material_root)
    material_count = len(catalog.materials)
    if material_count == 0:
        if not configured_catalog.is_file():
            raise FileNotFoundError(
                "Configured material root exports no MDLs and the fallback "
                f"catalog does not exist: {configured_catalog}"
            )
        log_message(
            log_cb,
            "Configured material root contains no exported MDL materials; "
            "using the configured catalog and allowlist.",
        )
        return configured_catalog, configured_whitelist, None

    catalog_path = analysis_dir / "nvidia_mdl_catalog.json"
    allowlist_path = analysis_dir / "nvidia_mdl_allowlist.json"
    catalog.save(catalog_path)
    write_object(allowlist_path, catalog.to_full_allowlist_dict())
    log_message(
        log_cb,
        "Built run-local NVIDIA MDL catalog: "
        f"root={material_root} exported_materials={material_count}",
    )
    return catalog_path, allowlist_path, material_count

def _replace_command_option(command: list[str], flag: str, value: str) -> list[str]:
    """Return a command copy with one required option replaced."""

    try:
        flag_index = command.index(flag)
    except ValueError as exc:
        raise RuntimeError(f"Recovery command is missing required option {flag}") from exc
    value_index = flag_index + 1
    if value_index >= len(command) or command[value_index].startswith("--"):
        raise RuntimeError(f"Recovery command has no value for required option {flag}")
    recovered = list(command)
    recovered[value_index] = value
    return recovered


def _material_stage_checkpoint_available(output_dir: Path) -> bool:
    """Return whether persisted Qwen material inference reached its tail gate."""

    required = (
        "palette.json",
        "mvinverse_pbr_evidence.json",
        "staged_result.json",
        "material_plan.json",
        "group_materials.json",
        "material_choice_audit.json",
        "view_evidence.json",
        "part_mapping_multiview_votes.json",
        "part_mapping_multiview_audit.json",
        "spatial_mapping_report.json",
        "spatial_mapping_audit.json",
        "material_stage_contract.json",
    )
    available = all((output_dir / name).is_file() for name in required) and any(
        (output_dir / "batches").glob("*.json")
    )
    if not available:
        return False
    try:
        material_stage_contract = read_object(
            output_dir / "material_stage_contract.json",
            "material-stage revision contract",
        )
    except (OSError, RuntimeError, ValueError):
        return False
    if material_stage_contract != material_stage_contract_document():
        return False
    qwen_ledger_path = output_dir / "qwen_inference_ledger.json"
    if not qwen_ledger_path.is_file():
        # Compatibility for direct legacy Qwen3-VL callers.  The production
        # v2 bridge rejects such a directory before reaching this helper.
        return True
    try:
        qwen_ledger = read_object(
            qwen_ledger_path,
            "material-stage Qwen ledger",
        )
    except (OSError, RuntimeError, ValueError):
        return False
    if qwen_ledger.get("requested_model_family") in {
        "qwen3_5",
        "openai_compatible",
    }:
        return (
            (output_dir / "sam3_foreground_request.json").is_file()
            and (output_dir / "sam3_foreground" / "manifest.json").is_file()
            and (
                output_dir
                / "foreground_inference"
                / "foreground_inference_manifest.json"
            ).is_file()
            and (output_dir / "sam3_region_request.json").is_file()
            and (output_dir / "sam3_regions" / "manifest.json").is_file()
            and (output_dir / "visual_retrieval_request.json").is_file()
            and (
                output_dir / "visual_retrieval" / "visual_retrieval.json"
            ).is_file()
        )
    return True


def _with_material_stage_resume(command: list[str], enabled: bool) -> list[str]:
    recovered = list(command)
    if enabled and "--resume-from-materials" not in recovered:
        recovered.append("--resume-from-materials")
    return recovered


def _run_qwen_mvinverse_with_recovery(
    command: list[str],
    *,
    output_dir: Path,
    ledger: Path,
    face_region_manifest: Path,
    log_cb: LogCallback,
    command_runner: CommandRunner,
) -> Path:
    """Run inference once and recover only unclassified or transient failures.

    Qwen is deliberately isolated in a subprocess.  A transient interpreter,
    CUDA, or model-load failure therefore cannot leave the parent process
    poisoned.  MVInverse and face-region products are reused only when both
    hash-bound checkpoints exist; their own readers perform the full
    provenance validation before inference is retried.  A strict child failure
    artifact suppresses an identical retry for deterministic palette/evidence
    failures.  If an eligible failure predates reusable checkpoints, the
    incomplete attempt is retained beside a fresh stage directory and the
    complete stage is tried once more.
    """

    audit_path = output_dir / "qwen_mvinverse_recovery.json"
    inference_failure_path = output_dir / "inference_failure.json"
    qwen_ledger_path = output_dir / "qwen_inference_ledger.json"
    sam3_foreground_manifest_path = (
        output_dir / "sam3_foreground" / "manifest.json"
    )
    sam3_manifest_path = output_dir / "sam3_regions" / "manifest.json"
    visual_retrieval_path = (
        output_dir / "visual_retrieval" / "visual_retrieval.json"
    )

    def clear_inference_failure() -> None:
        if inference_failure_path.exists() or inference_failure_path.is_symlink():
            inference_failure_path.unlink()

    def read_inference_failure() -> dict[str, Any] | None:
        if not inference_failure_path.is_file():
            return None
        try:
            failure = read_object(
                inference_failure_path,
                "Qwen material inference failure",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return {
                "schema_version": "invalid",
                "status": "FAILED",
                "error_code": "invalid_inference_failure_artifact",
                "retryable": False,
                "retry_scope": "none",
                "detail": str(exc),
            }
        if (
            failure.get("schema_version")
            != "qwen-material-inference-failure/v1"
            or failure.get("status") != "FAILED"
            or not isinstance(failure.get("retryable"), bool)
            or failure.get("retry_scope") not in {"none", "fresh_process"}
            or not isinstance(failure.get("error_code"), str)
            or not failure["error_code"].strip()
            or not isinstance(failure.get("failed_stage"), str)
            or not failure["failed_stage"].strip()
            or not isinstance(failure.get("detail"), str)
            or not failure["detail"].strip()
            or not isinstance(failure.get("view_failures"), list)
            or (
                "context" in failure
                and not isinstance(failure.get("context"), dict)
            )
            or (
                failure.get("retryable") is False
                and failure.get("retry_scope") != "none"
            )
            or (
                failure.get("retryable") is True
                and failure.get("retry_scope") != "fresh_process"
            )
        ):
            return {
                "schema_version": "invalid",
                "status": "FAILED",
                "error_code": "invalid_inference_failure_contract",
                "retryable": False,
                "retry_scope": "none",
                "detail": "child failure artifact violates its strict contract",
            }
        return failure

    def multimodel_hashes() -> dict[str, str | None]:
        return {
            "qwen_inference_ledger_sha256": (
                sha256_file(qwen_ledger_path)
                if qwen_ledger_path.is_file()
                else None
            ),
            "sam3_manifest_sha256": (
                sha256_file(sam3_manifest_path)
                if sam3_manifest_path.is_file()
                else None
            ),
            "sam3_foreground_manifest_sha256": (
                sha256_file(sam3_foreground_manifest_path)
                if sam3_foreground_manifest_path.is_file()
                else None
            ),
            "visual_retrieval_sha256": (
                sha256_file(visual_retrieval_path)
                if visual_retrieval_path.is_file()
                else None
            ),
        }

    verified_resume = ledger.is_file() and face_region_manifest.is_file()
    material_stage_resume = (
        verified_resume and _material_stage_checkpoint_available(output_dir)
    )
    initial_command = (
        _replace_command_option(command, "--mvinverse-mode", "reuse")
        if verified_resume
        else list(command)
    )
    initial_command = _with_material_stage_resume(
        initial_command, material_stage_resume
    )
    initial_stage = (
        "qwen_mvinverse_verified_material_stage_resume"
        if material_stage_resume
        else "qwen_mvinverse_verified_checkpoint_resume"
        if verified_resume
        else "qwen_mvinverse"
    )
    if material_stage_resume:
        log_message(
            log_cb,
            "Visual material inference found a complete material-stage "
            "checkpoint; deterministic readers will revalidate it before "
            "resuming confidence, PBR, and face-level gates.",
        )
    elif verified_resume:
        log_message(
            log_cb,
            "Visual material inference found both heavy checkpoints; their "
            "readers will verify and reuse them before Qwen resumes.",
        )
    clear_inference_failure()
    try:
        _run_stage(
            initial_stage,
            initial_command,
            log_cb,
            command_runner=command_runner,
        )
    except RuntimeError as first_error:
        classified_failure = read_inference_failure()
        if classified_failure is not None and not (
            classified_failure.get("retryable") is True
            and classified_failure.get("retry_scope") == "fresh_process"
        ):
            write_object(
                audit_path,
                {
                    "schema_version": "qwen-mvinverse-recovery/v2",
                    "status": "FAILED_NON_RETRYABLE",
                    "attempt_count": 1,
                    "first_error": str(first_error),
                    "decision": "SUPPRESS_IDENTICAL_RETRY",
                    "failure_code": classified_failure.get("error_code"),
                    "failure": classified_failure,
                    **multimodel_hashes(),
                },
            )
            raise RuntimeError(
                "Visual material inference reported a deterministic failure; "
                "an identical fresh-process retry was suppressed: "
                f"{classified_failure.get('error_code')}"
            ) from first_error

        checkpoint_reuse = ledger.is_file() and face_region_manifest.is_file()
        material_checkpoint_reuse = (
            checkpoint_reuse
            and _material_stage_checkpoint_available(output_dir)
        )
        archived_attempt: Path | None = None
        if checkpoint_reuse:
            retry_command = _replace_command_option(
                command, "--mvinverse-mode", "reuse"
            )
            retry_command = _with_material_stage_resume(
                retry_command, material_checkpoint_reuse
            )
            retry_mode = (
                "verified_material_stage_resume"
                if material_checkpoint_reuse
                else "verified_reuse"
            )
            retry_stage = (
                "qwen_mvinverse_verified_material_stage_resume_retry"
                if material_checkpoint_reuse
                else "qwen_mvinverse_verified_reuse_retry"
            )
            recovery_message = (
                "Visual material inference will retry in a fresh process "
                "using the verified material-stage checkpoint."
                if material_checkpoint_reuse
                else "Visual material inference will retry in a fresh process "
                "using verified MVInverse and face-region checkpoints."
            )
        else:
            retry_command = list(command)
            retry_mode = "fresh_stage"
            retry_stage = "qwen_mvinverse_fresh_stage_retry"
            recovery_message = (
                "Visual material inference failed before reusable checkpoints; "
                "the incomplete attempt will be archived and the complete "
                "stage retried once in a fresh process."
            )
            archived_attempt = output_dir.with_name(
                f"{output_dir.name}.failed_attempt_01"
            )
            if archived_attempt.exists():
                raise RuntimeError(
                    "Cannot retry visual material inference because the "
                    f"archive destination already exists: {archived_attempt}"
                ) from first_error
            if output_dir.exists():
                output_dir.rename(archived_attempt)
            output_dir.mkdir(parents=True, exist_ok=False)

        log_message(log_cb, recovery_message)
        clear_inference_failure()
        try:
            _run_stage(
                retry_stage,
                retry_command,
                log_cb,
                command_runner=command_runner,
            )
        except RuntimeError as retry_error:
            write_object(
                audit_path,
                {
                    "schema_version": "qwen-mvinverse-recovery/v2",
                    "status": "FAILED_AFTER_VERIFIED_REUSE_RETRY",
                    "attempt_count": 2,
                    "first_error": str(first_error),
                    "retry_error": str(retry_error),
                    "retry_mode": retry_mode,
                    "archived_attempt": (
                        str(archived_attempt)
                        if archived_attempt is not None
                        else None
                    ),
                    "mvinverse_ledger_sha256": (
                        sha256_file(ledger) if checkpoint_reuse else None
                    ),
                    "face_region_manifest_sha256": (
                        sha256_file(face_region_manifest)
                        if checkpoint_reuse
                        else None
                    ),
                    **multimodel_hashes(),
                },
            )
            raise RuntimeError(
                "Visual material inference failed again after a fresh-process "
                "verified-checkpoint retry"
            ) from retry_error
        write_object(
            audit_path,
            {
                "schema_version": "qwen-mvinverse-recovery/v2",
                "status": "RECOVERED",
                "attempt_count": 2,
                "first_error": str(first_error),
                "retry_mode": retry_mode,
                "archived_attempt": (
                    str(archived_attempt) if archived_attempt is not None else None
                ),
                "mvinverse_ledger_sha256": (
                    sha256_file(ledger) if checkpoint_reuse else None
                ),
                "face_region_manifest_sha256": (
                    sha256_file(face_region_manifest) if checkpoint_reuse else None
                ),
                **multimodel_hashes(),
            },
        )
    else:
        write_object(
            audit_path,
            (
                {
                    "schema_version": "qwen-mvinverse-recovery/v2",
                    "status": "RESUMED_FROM_VERIFIED_CHECKPOINTS",
                    "attempt_count": 1,
                    "retry_mode": "verified_reuse",
                    "material_stage_resume": material_stage_resume,
                    "mvinverse_ledger_sha256": sha256_file(ledger),
                    "face_region_manifest_sha256": sha256_file(
                        face_region_manifest
                    ),
                }
                if verified_resume
                else {
                    "schema_version": "qwen-mvinverse-recovery/v2",
                    "status": "NOT_NEEDED",
                    "attempt_count": 1,
                    **multimodel_hashes(),
                }
            ),
        )
    return audit_path.resolve(strict=True)

@dataclass(frozen=True)
class MaterialInferenceResult:
    catalog: Path
    whitelist: Path
    material_count: int | None
    unattended: dict[str, Any]


def run_material_inference(
    context: VisualMaterialPipelineContext,
    *,
    rendered_registry: Path,
    log_cb: LogCallback,
    command_runner: CommandRunner,
) -> MaterialInferenceResult:
    """Build the live MDL catalog and execute hash-verified model inference."""

    config = context.config
    paths = context.workspace.inference
    catalog, whitelist, material_count = _prepare_live_material_catalog(
        material_root=config.material_root,
        configured_catalog=config.catalog,
        configured_whitelist=config.whitelist,
        analysis_dir=paths.root,
        log_cb=log_cb,
    )
    command = staged_material_command(
        config=config,
        registry=rendered_registry,
        references=context.references,
        foreground_annotations=context.foreground_annotations,
        catalog=catalog,
        whitelist=whitelist,
        output_dir=paths.root,
        isaac_python=context.isaac_python,
    )
    completed_resume = (
        context.partial_live_resume and paths.unattended_result.is_file()
    )
    if completed_resume:
        log_message(
            log_cb,
            "Revalidating the completed visual inference checkpoint against "
            "the current Qwen model fingerprint; Qwen weights and MVInverse "
            "inference will not be rerun.",
        )
    _run_qwen_mvinverse_with_recovery(
        command,
        output_dir=paths.root,
        ledger=paths.mvinverse_ledger,
        face_region_manifest=paths.face_region_manifest,
        log_cb=log_cb,
        command_runner=command_runner,
    )
    for artifact in (
        paths.inference_recovery,
        paths.unattended_result,
        paths.staged_material_plan,
        paths.mvinverse_ledger,
    ):
        require_file(artifact, "qwen_mvinverse")
    return MaterialInferenceResult(
        catalog=catalog,
        whitelist=whitelist,
        material_count=material_count,
        unattended=read_object(paths.unattended_result, "unattended result"),
    )


__all__ = [
    "MaterialInferenceResult",
    "_prepare_live_material_catalog",
    "_run_qwen_mvinverse_with_recovery",
    "run_material_inference",
]
