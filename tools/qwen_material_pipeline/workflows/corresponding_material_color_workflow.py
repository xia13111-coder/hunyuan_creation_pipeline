"""Run the saved material-identity-first, render-calibrated colour workflow.

The workflow deliberately keeps two decisions separate:

1. the input Part-ID plan fixes every selected NVIDIA Base MDL identity; and
2. only corresponding-material assignments receive photo-derived colour.

The default controller starts from the photo-derived colour, measures every
scope in a real CAD render, and proposes one continuous bounded gain per scope.
All proposals are evaluated together in the next render.  Each scope finally
selects its best measured history before an independent all-view comparison.
No image post-processing or per-asset Part-ID rule is used here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..evidence import reference_compare
from ..materials import adaptive_corresponding_material_color as adaptive_color
from ..materials import corresponding_material_color_selection as color_selection
from ..materials.corresponding_material_color import (
    build_corresponding_material_color_plan,
)
from ..usd.material_common import canonical_sha256


SCHEMA_VERSION = "qwen-corresponding-material-color-workflow/v3"


class CorrespondingMaterialColorWorkflowError(RuntimeError):
    """Raised when the saved workflow cannot be reproduced safely."""


CommandRunner = Callable[[Sequence[str], Path, Mapping[str, str], int], None]


def _read_object(path: Path) -> dict[str, Any]:
    resolved = _regular_file(path, "JSON input")
    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise CorrespondingMaterialColorWorkflowError(
            f"JSON root must be an object: {resolved}"
        )
    return value


def _write_object(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise CorrespondingMaterialColorWorkflowError(
            f"{label} does not exist: {path}"
        ) from exc
    if not resolved.is_file() or path.expanduser().is_symlink():
        raise CorrespondingMaterialColorWorkflowError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    return resolved


def _directory(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise CorrespondingMaterialColorWorkflowError(
            f"{label} does not exist: {path}"
        ) from exc
    if not resolved.is_dir():
        raise CorrespondingMaterialColorWorkflowError(
            f"{label} must be a directory: {path}"
        )
    return resolved


def _normalized_gains(values: Sequence[float]) -> tuple[float, ...]:
    gains: list[float] = []
    for raw in values:
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or not 0.1 <= float(raw) <= 8.0
        ):
            raise CorrespondingMaterialColorWorkflowError(
                "colour gains must be finite numbers from 0.1 to 8.0"
            )
        gain = float(raw)
        if gain in gains:
            raise CorrespondingMaterialColorWorkflowError(
                f"duplicate colour gain: {gain}"
            )
        gains.append(gain)
    if len(gains) < 2:
        raise CorrespondingMaterialColorWorkflowError(
            "render-calibrated colour selection needs at least two gains"
        )
    return tuple(gains)


def _candidate_id(gain: float) -> str:
    return f"gain_{gain:.2f}".replace(".", "_")


def _run_logged_command(
    command: Sequence[str],
    log_path: Path,
    environment: Mapping[str, str],
    timeout_seconds: int,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"command: {shlex.join(command)}\n")
        handle.flush()
        try:
            result = subprocess.run(
                list(command),
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=dict(environment),
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CorrespondingMaterialColorWorkflowError(
                f"command timed out after {timeout_seconds}s; see {log_path}"
            ) from exc
    if result.returncode != 0:
        raise CorrespondingMaterialColorWorkflowError(
            f"command exited {result.returncode}; see {log_path}"
        )


def _isaac_environment() -> dict[str, str]:
    environment = dict(os.environ)
    tools_root = str(Path(__file__).resolve().parents[2])
    existing = environment.get("PYTHONPATH", "")
    paths = [value for value in existing.split(os.pathsep) if value]
    environment["PYTHONPATH"] = os.pathsep.join(
        [tools_root, *(value for value in paths if value != tools_root)]
    )
    return environment


def _apply_command(
    *,
    isaac_python: Path,
    asset_usd: Path,
    catalog: Path,
    registry: Path,
    plan: Path,
    output: Path,
    material_root: Path,
    report: Path,
) -> list[str]:
    return [
        str(isaac_python),
        "-m",
        "qwen_material_pipeline.usd.apply",
        "--asset-usd",
        str(asset_usd),
        "--catalog",
        str(catalog),
        "--registry",
        str(registry),
        "--plan",
        str(plan),
        "--output",
        str(output),
        "--material-root",
        str(material_root),
        "--include-review",
        "--include-policy-fallback",
        "--report",
        str(report),
    ]


def _registry_command(*, isaac_python: Path, asset: Path, output: Path) -> list[str]:
    return [
        str(isaac_python),
        "-m",
        "qwen_material_pipeline.usd.registry",
        "--usd",
        str(asset),
        "--output",
        str(output),
        "--headless",
    ]


def _render_command(
    *,
    isaac_python: Path,
    registry: Path,
    output_dir: Path,
    resolution: int,
    view_specs: Path,
    rt_subframes: int,
) -> list[str]:
    return [
        str(isaac_python),
        "-m",
        "qwen_material_pipeline.usd.render",
        "--registry",
        str(registry),
        "--output-dir",
        str(output_dir),
        "--resolution",
        str(resolution),
        "--view-specs",
        str(view_specs),
        "--rt-subframes",
        str(rt_subframes),
        "--lighting-profile",
        "material-neutral",
        "--rgb-only",
    ]


def _render_plan(
    *,
    directory: Path,
    plan: Path,
    isaac_python: Path,
    asset_usd: Path,
    catalog: Path,
    registry: Path,
    material_root: Path,
    view_specs: Path,
    resolution: int,
    rt_subframes: int,
    timeout_seconds: int,
    environment: Mapping[str, str],
    command_runner: CommandRunner,
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    asset = directory / "material_look.usda"
    apply_report = directory / "apply_report.json"
    rendered_registry = directory / "part_registry.json"
    renders = directory / "renders"
    for path in (asset, apply_report, rendered_registry, renders):
        if path.exists():
            raise CorrespondingMaterialColorWorkflowError(
                f"render destination is not fresh: {path}"
            )
    command_runner(
        _apply_command(
            isaac_python=isaac_python,
            asset_usd=asset_usd,
            catalog=catalog,
            registry=registry,
            plan=plan,
            output=asset,
            material_root=material_root,
            report=apply_report,
        ),
        directory / "isaac_apply.log",
        environment,
        timeout_seconds,
    )
    command_runner(
        _registry_command(
            isaac_python=isaac_python, asset=asset, output=rendered_registry
        ),
        directory / "isaac_registry.log",
        environment,
        timeout_seconds,
    )
    command_runner(
        _render_command(
            isaac_python=isaac_python,
            registry=rendered_registry,
            output_dir=renders,
            resolution=resolution,
            view_specs=view_specs,
            rt_subframes=rt_subframes,
        ),
        directory / "isaac_render.log",
        environment,
        timeout_seconds,
    )
    output = {
        "asset": asset,
        "apply_report": apply_report,
        "registry": rendered_registry,
        "rendered_registry": renders / "part_registry.rendered.json",
    }
    for label, path in output.items():
        _regular_file(path, f"rendered {label}")
    return output


def _reference_view_ids(reference_manifest: Mapping[str, Any]) -> tuple[str, ...]:
    source_views = reference_manifest.get("source_views")
    if not isinstance(source_views, list) or not source_views:
        raise CorrespondingMaterialColorWorkflowError(
            "reference manifest has no source_views"
        )
    view_ids: list[str] = []
    for index, row in enumerate(source_views):
        view_id = row.get("id") if isinstance(row, Mapping) else None
        if not isinstance(view_id, str) or not view_id or view_id in view_ids:
            raise CorrespondingMaterialColorWorkflowError(
                f"reference manifest has invalid source view at index {index}"
            )
        view_ids.append(view_id)
    return tuple(view_ids)


def _reference_image_sha256_by_view(
    part_id_evidence: Mapping[str, Any],
) -> dict[str, str]:
    hashes: dict[str, set[str]] = {}
    parts = part_id_evidence.get("parts")
    if not isinstance(parts, list):
        raise CorrespondingMaterialColorWorkflowError(
            "Part-ID evidence has no parts for reference-image binding"
        )
    for part in parts:
        observations = part.get("observations") if isinstance(part, Mapping) else None
        for observation in observations or []:
            if not isinstance(observation, Mapping):
                continue
            view_id = observation.get("view_id")
            digest = observation.get("image_sha256")
            if isinstance(view_id, str) and isinstance(digest, str):
                hashes.setdefault(view_id, set()).add(digest)
    output: dict[str, str] = {}
    for view_id, values in hashes.items():
        if len(values) != 1:
            raise CorrespondingMaterialColorWorkflowError(
                f"reference view {view_id} has conflicting sealed image hashes"
            )
        digest = next(iter(values))
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise CorrespondingMaterialColorWorkflowError(
                f"reference view {view_id} has an invalid sealed image hash"
            )
        output[view_id] = digest
    return output


def _resolve_reference_image(
    path: object,
    *,
    manifest_path: Path,
    expected_sha256: str | None,
    view_id: str,
) -> tuple[Path, bool]:
    if not isinstance(path, str) or not path.strip():
        raise CorrespondingMaterialColorWorkflowError(
            f"reference view {view_id} has no image path"
        )
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        if expected_sha256 is None:
            raise CorrespondingMaterialColorWorkflowError(
                f"reference image is missing and has no sealed hash: {candidate}"
            ) from exc
        try:
            parent = candidate.parent.resolve(strict=True)
        except FileNotFoundError as parent_exc:
            raise CorrespondingMaterialColorWorkflowError(
                f"reference image directory is missing: {candidate.parent}"
            ) from parent_exc
        matches = [
            sibling.resolve(strict=True)
            for sibling in parent.iterdir()
            if sibling.is_file()
            and not sibling.is_symlink()
            and _sha256_file(sibling) == expected_sha256
        ]
        if len(matches) != 1:
            raise CorrespondingMaterialColorWorkflowError(
                f"reference view {view_id} cannot be relocated by sealed SHA-256; "
                f"expected one sibling match, found {len(matches)}"
            ) from exc
        return matches[0], True
    if candidate.is_symlink() or not resolved.is_file():
        raise CorrespondingMaterialColorWorkflowError(
            f"reference image must be a regular non-symlink file: {candidate}"
        )
    actual_sha256 = _sha256_file(resolved)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise CorrespondingMaterialColorWorkflowError(
            f"reference view {view_id} failed its sealed image hash"
        )
    return resolved, False


def _effective_reference_manifest(
    *,
    reference_manifest: Mapping[str, Any],
    reference_manifest_path: Path,
    part_id_evidence: Mapping[str, Any],
    output_path: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    """Relocate renamed reference files only through sealed byte identity."""

    output = copy.deepcopy(dict(reference_manifest))
    source_views = output.get("source_views")
    if not isinstance(source_views, list) or not source_views:
        raise CorrespondingMaterialColorWorkflowError(
            "reference manifest has no source_views"
        )
    expected_hashes = _reference_image_sha256_by_view(part_id_evidence)
    resolutions: list[dict[str, Any]] = []
    changed = False
    for row in source_views:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise CorrespondingMaterialColorWorkflowError(
                "reference manifest has an invalid source view"
            )
        view_id = str(row["id"])
        resolved, relocated = _resolve_reference_image(
            row.get("image"),
            manifest_path=reference_manifest_path,
            expected_sha256=expected_hashes.get(view_id),
            view_id=view_id,
        )
        original_path = row.get("image")
        row["image"] = str(resolved)
        row["image_sha256"] = _sha256_file(resolved)
        changed = changed or relocated
        resolutions.append(
            {
                "view_id": view_id,
                "original_path": original_path,
                "resolved_path": str(resolved),
                "image_sha256": row["image_sha256"],
                "relocated": relocated,
            }
        )
    if not changed:
        return reference_manifest_path, resolutions
    _write_object(output_path, output)
    return output_path, resolutions


def run_corresponding_material_color_workflow(
    *,
    source_plan_path: Path,
    qwen_choices_path: Path,
    part_id_evidence_path: Path,
    spatial_mapping_report_path: Path,
    asset_usd_path: Path,
    catalog_path: Path,
    registry_path: Path,
    material_root_path: Path,
    view_specs_path: Path,
    reference_manifest_path: Path,
    isaac_python_path: Path,
    output_dir: Path,
    gains: Sequence[float] | None = None,
    max_adaptive_iterations: int = 5,
    resolution: int = 512,
    rt_subframes: int = 4,
    timeout_seconds: int = 1800,
    command_runner: CommandRunner = _run_logged_command,
) -> dict[str, Any]:
    """Execute the complete saved workflow in one fresh output directory."""

    if (
        resolution < 64
        or rt_subframes < 1
        or timeout_seconds < 1
        or max_adaptive_iterations < 1
    ):
        raise CorrespondingMaterialColorWorkflowError(
            "resolution, iteration count, rt_subframes and timeout must be positive"
        )
    calibrated_gains = None if gains is None else _normalized_gains(gains)
    optimization_mode = "adaptive_per_scope" if gains is None else "fixed_grid"
    inputs = {
        "source_plan": _regular_file(source_plan_path, "source plan"),
        "qwen_choices": _regular_file(qwen_choices_path, "Qwen choices"),
        "part_id_evidence": _regular_file(part_id_evidence_path, "Part-ID evidence"),
        "spatial_mapping_report": _regular_file(
            spatial_mapping_report_path, "spatial mapping report"
        ),
        "asset_usd": _regular_file(asset_usd_path, "source asset USD"),
        "catalog": _regular_file(catalog_path, "material catalog"),
        "registry": _regular_file(registry_path, "source Part-ID registry"),
        "view_specs": _regular_file(view_specs_path, "camera view specs"),
        "reference_manifest": _regular_file(
            reference_manifest_path, "reference manifest"
        ),
        "isaac_python": _regular_file(isaac_python_path, "Isaac Python launcher"),
    }
    material_root = _directory(material_root_path, "NVIDIA Base material root")
    destination = output_dir.expanduser().resolve()
    if destination.exists():
        raise CorrespondingMaterialColorWorkflowError(
            f"output directory already exists; use a fresh destination: {destination}"
        )
    destination.mkdir(parents=True)
    manifest_path = destination / "workflow_manifest.json"
    input_records = {
        label: {"path": str(path), "sha256": _sha256_file(path)}
        for label, path in inputs.items()
    }
    input_records["material_root"] = {"path": str(material_root)}
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "workflow_state": "RUNNING",
        "inputs": input_records,
        "policy": {
            "material_identity_mutation_allowed": False,
            "same_component_shares_material_and_colour": True,
            "actual_cad_render_selection": True,
            "local_part_scope_quality_gate": True,
            "local_quality_rejection_behavior": (
                "retain_best_rendered_candidate_and_continue_with_review"
            ),
            "optimization_mode": optimization_mode,
            "explicit_linear_intensity_gains": (
                None if calibrated_gains is None else list(calibrated_gains)
            ),
            "maximum_adaptive_iterations": max_adaptive_iterations,
            "adaptive_policy": {
                "initial_gain": adaptive_color.INITIAL_GAIN,
                "minimum_gain": adaptive_color.MINIMUM_GAIN,
                "maximum_gain": adaptive_color.MAXIMUM_GAIN,
                "final_authority": "best_registered_actual_cad_scope_score",
            },
            "resolution": resolution,
            "rt_subframes": rt_subframes,
            "lighting_profile": "material-neutral",
        },
    }
    _write_object(manifest_path, manifest)
    environment = _isaac_environment()
    try:
        source_plan = _read_object(inputs["source_plan"])
        qwen_choices = _read_object(inputs["qwen_choices"])
        part_id_evidence = _read_object(inputs["part_id_evidence"])
        spatial_mapping_report = _read_object(inputs["spatial_mapping_report"])
        candidate_dirs: list[Path] = []
        candidate_records: list[dict[str, Any]] = []
        controller_rounds: list[dict[str, Any]] = []
        candidates_root = destination / "candidates"
        candidates_root.mkdir()

        def render_candidate(
            candidate_id: str,
            plan: Mapping[str, Any],
            audit: Mapping[str, Any],
        ) -> tuple[Path, dict[str, Path]]:
            candidate_dir = candidates_root / candidate_id
            candidate_dir.mkdir()
            plan_path = candidate_dir / "part_id_material_plan.color.json"
            audit_path = candidate_dir / "corresponding_material_color_audit.json"
            _write_object(plan_path, plan)
            _write_object(audit_path, audit)
            rendered = _render_plan(
                directory=candidate_dir,
                plan=plan_path,
                isaac_python=inputs["isaac_python"],
                asset_usd=inputs["asset_usd"],
                catalog=inputs["catalog"],
                registry=inputs["registry"],
                material_root=material_root,
                view_specs=inputs["view_specs"],
                resolution=resolution,
                rt_subframes=rt_subframes,
                timeout_seconds=timeout_seconds,
                environment=environment,
                command_runner=command_runner,
            )
            candidate_dirs.append(candidate_dir)
            print(f"Colour candidate complete: {candidate_id}", flush=True)
            return candidate_dir, rendered

        selector_minimum_candidate_count = 2
        if calibrated_gains is not None:
            for gain in calibrated_gains:
                candidate_id = _candidate_id(gain)
                plan, audit = build_corresponding_material_color_plan(
                    source_plan=source_plan,
                    qwen_choices=qwen_choices,
                    part_id_evidence=part_id_evidence,
                    linear_intensity_gain=gain,
                )
                candidate_dir, rendered = render_candidate(candidate_id, plan, audit)
                candidate_records.append(
                    {
                        "candidate_id": candidate_id,
                        "linear_intensity_gain": gain,
                        "directory": str(candidate_dir),
                        "plan_sha256": canonical_sha256(plan),
                        "asset_sha256": _sha256_file(rendered["asset"]),
                        "rendered_registry_sha256": _sha256_file(
                            rendered["rendered_registry"]
                        ),
                    }
                )
        else:
            selector_minimum_candidate_count = 1
            histories: dict[str, list[Mapping[str, Any]]] = {}
            next_scope_gains: Mapping[str, float] | None = None
            source_plan_sha256 = canonical_sha256(source_plan)
            for iteration_index in range(max_adaptive_iterations):
                candidate_id = f"iteration_{iteration_index:02d}"
                if next_scope_gains is None:
                    plan, audit = build_corresponding_material_color_plan(
                        source_plan=source_plan,
                        qwen_choices=qwen_choices,
                        part_id_evidence=part_id_evidence,
                        linear_intensity_gain=adaptive_color.INITIAL_GAIN,
                    )
                else:
                    plan, audit = build_corresponding_material_color_plan(
                        source_plan=source_plan,
                        qwen_choices=qwen_choices,
                        part_id_evidence=part_id_evidence,
                        linear_intensity_gains_by_scope=next_scope_gains,
                    )
                candidate_dir, rendered = render_candidate(candidate_id, plan, audit)
                candidate = color_selection.load_rendered_color_candidate(
                    candidate_dir, source_plan_sha256
                )
                scope_scores = adaptive_color.score_adaptive_candidate_scopes(
                    candidate=candidate,
                    part_id_evidence=part_id_evidence,
                    spatial_mapping_report=spatial_mapping_report,
                )
                if histories and set(histories) != set(scope_scores):
                    raise CorrespondingMaterialColorWorkflowError(
                        "adaptive iteration changed the sealed colour scopes"
                    )
                for scope_id, score in scope_scores.items():
                    histories.setdefault(scope_id, []).append(score)
                (
                    proposed_gains,
                    controller_audit,
                ) = adaptive_color.propose_next_scope_gains(histories=histories)
                controller_audit = {
                    **controller_audit,
                    "iteration_index": iteration_index,
                    "candidate_id": candidate_id,
                }
                controller_rounds.append(controller_audit)
                iteration_audit_path = candidate_dir / "adaptive_iteration_audit.json"
                iteration_audit_unsigned = {
                    "schema_version": adaptive_color.SCHEMA_VERSION,
                    "candidate_id": candidate_id,
                    "scope_scores": scope_scores,
                    "controller": controller_audit,
                }
                _write_object(
                    iteration_audit_path,
                    {
                        **iteration_audit_unsigned,
                        "integrity": {
                            "document_sha256": canonical_sha256(
                                iteration_audit_unsigned
                            )
                        },
                    },
                )
                candidate_records.append(
                    {
                        "candidate_id": candidate_id,
                        "scope_gains": dict(candidate.gains_by_scope),
                        "directory": str(candidate_dir),
                        "plan_sha256": canonical_sha256(plan),
                        "asset_sha256": _sha256_file(rendered["asset"]),
                        "rendered_registry_sha256": _sha256_file(
                            rendered["rendered_registry"]
                        ),
                        "scope_scores": scope_scores,
                        "controller": controller_audit,
                        "adaptive_iteration_audit": {
                            "path": str(iteration_audit_path),
                            "sha256": _sha256_file(iteration_audit_path),
                        },
                    }
                )
                if controller_audit["all_scopes_converged"]:
                    break
                next_scope_gains = proposed_gains

        adaptive_completion = None
        if calibrated_gains is None:
            final_controller = controller_rounds[-1]
            adaptive_completion = {
                "executed_iteration_count": len(controller_rounds),
                "maximum_iteration_count": max_adaptive_iterations,
                "all_scopes_converged": final_controller["all_scopes_converged"],
                "remaining_active_scope_count": final_controller["active_scope_count"],
                "termination_reason": (
                    "all_scopes_converged"
                    if final_controller["all_scopes_converged"]
                    else "maximum_iterations_reached_best_history_selected"
                ),
            }

        final_dir = destination / "final_selected"
        final_dir.mkdir()
        final_plan = final_dir / "part_id_material_plan.color.selected.json"
        final_audit = final_dir / "corresponding_material_color_selection_audit.json"
        selector_arguments = [
            "--source-plan",
            str(inputs["source_plan"]),
            "--part-id-evidence",
            str(inputs["part_id_evidence"]),
            "--spatial-mapping-report",
            str(inputs["spatial_mapping_report"]),
        ]
        for candidate_dir in candidate_dirs:
            selector_arguments.extend(["--candidate-dir", str(candidate_dir)])
        selector_arguments.extend(
            [
                "--minimum-candidate-count",
                str(selector_minimum_candidate_count),
                "--output-plan",
                str(final_plan),
                "--audit",
                str(final_audit),
            ]
        )
        if color_selection.main(selector_arguments) != 0:
            raise CorrespondingMaterialColorWorkflowError(
                "render-calibrated colour selector returned non-zero"
            )
        selection_audit_document = _read_object(final_audit)
        local_quality_status = selection_audit_document.get("status")
        if local_quality_status not in {"PASS", "REVIEW"}:
            raise CorrespondingMaterialColorWorkflowError(
                "render-calibrated colour selector returned an invalid local "
                "quality status"
            )
        if local_quality_status == "REVIEW":
            print(
                "Local colour quality remains below floor for one or more scopes; "
                "retaining the best actual-CAD candidates and continuing with "
                "explicit REVIEW status.",
                flush=True,
            )
        final_render = _render_plan(
            directory=final_dir,
            plan=final_plan,
            isaac_python=inputs["isaac_python"],
            asset_usd=inputs["asset_usd"],
            catalog=inputs["catalog"],
            registry=inputs["registry"],
            material_root=material_root,
            view_specs=inputs["view_specs"],
            resolution=resolution,
            rt_subframes=rt_subframes,
            timeout_seconds=timeout_seconds,
            environment=environment,
            command_runner=command_runner,
        )
        rendered_registry = final_dir / "renders" / "part_registry.rendered.json"
        reference_manifest = _read_object(inputs["reference_manifest"])
        view_ids = _reference_view_ids(reference_manifest)
        (
            effective_reference_manifest,
            reference_image_resolutions,
        ) = _effective_reference_manifest(
            reference_manifest=reference_manifest,
            reference_manifest_path=inputs["reference_manifest"],
            part_id_evidence=part_id_evidence,
            output_path=final_dir / "reference_manifest.resolved.json",
        )
        quality_report = final_dir / "reference_render_comparison.json"
        comparison_arguments = [
            "--reference-manifest",
            str(effective_reference_manifest),
            "--rendered-registry",
            str(rendered_registry),
            "--output",
            str(quality_report),
            "--minimum-comparable-views",
            str(len(view_ids)),
        ]
        for view_id in view_ids:
            comparison_arguments.extend(["--map", f"{view_id}={view_id}"])
        if reference_compare.main(comparison_arguments) != 0:
            raise CorrespondingMaterialColorWorkflowError(
                "reference comparison returned non-zero"
            )
        quality = _read_object(quality_report)
        aggregate = quality.get("aggregate")
        if not isinstance(aggregate, Mapping):
            raise CorrespondingMaterialColorWorkflowError(
                "reference comparison did not produce aggregate metrics"
            )
        manifest.update(
            {
                "workflow_state": "COMPLETE",
                "quality_status": aggregate.get("status"),
                "local_quality_status": local_quality_status,
                "candidates": candidate_records,
                "adaptive_controller_rounds": controller_rounds,
                "adaptive_completion": adaptive_completion,
                "reference_image_resolution": {
                    "authority": "sealed_image_sha256_same_directory_only",
                    "relocated_view_count": sum(
                        bool(row["relocated"]) for row in reference_image_resolutions
                    ),
                    "views": reference_image_resolutions,
                },
                "outputs": {
                    "selected_plan": {
                        "path": str(final_plan),
                        "sha256": _sha256_file(final_plan),
                    },
                    "selection_audit": {
                        "path": str(final_audit),
                        "sha256": _sha256_file(final_audit),
                    },
                    "asset": {
                        "path": str(final_render["asset"]),
                        "sha256": _sha256_file(final_render["asset"]),
                    },
                    "apply_report": {
                        "path": str(final_render["apply_report"]),
                        "sha256": _sha256_file(final_render["apply_report"]),
                    },
                    "registry": {
                        "path": str(final_render["registry"]),
                        "sha256": _sha256_file(final_render["registry"]),
                    },
                    "rendered_registry": {
                        "path": str(rendered_registry),
                        "sha256": _sha256_file(rendered_registry),
                    },
                    "quality_report": {
                        "path": str(quality_report),
                        "sha256": _sha256_file(quality_report),
                    },
                    **(
                        {
                            "resolved_reference_manifest": {
                                "path": str(effective_reference_manifest),
                                "sha256": _sha256_file(effective_reference_manifest),
                            }
                        }
                        if effective_reference_manifest != inputs["reference_manifest"]
                        else {}
                    ),
                },
                "quality_aggregate": dict(aggregate),
            }
        )
        _write_object(manifest_path, manifest)
        return manifest
    except Exception as exc:
        manifest["workflow_state"] = "FAILED"
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_object(manifest_path, manifest)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--qwen-choices", type=Path, required=True)
    parser.add_argument("--part-id-evidence", type=Path, required=True)
    parser.add_argument("--spatial-mapping-report", type=Path, required=True)
    parser.add_argument("--asset-usd", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--material-root", type=Path, required=True)
    parser.add_argument("--view-specs", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--isaac-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--gain",
        type=float,
        action="append",
        help=(
            "explicit fixed-grid compatibility gain; repeat at least twice; "
            "omit to use automatic per-scope calibration"
        ),
    )
    parser.add_argument("--max-adaptive-iterations", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--rt-subframes", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="return status 3 when the final all-view absolute QA is not PASS",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run_corresponding_material_color_workflow(
            source_plan_path=args.source_plan,
            qwen_choices_path=args.qwen_choices,
            part_id_evidence_path=args.part_id_evidence,
            spatial_mapping_report_path=args.spatial_mapping_report,
            asset_usd_path=args.asset_usd,
            catalog_path=args.catalog,
            registry_path=args.registry,
            material_root_path=args.material_root,
            view_specs_path=args.view_specs,
            reference_manifest_path=args.reference_manifest,
            isaac_python_path=args.isaac_python,
            output_dir=args.output_dir,
            gains=args.gain,
            max_adaptive_iterations=args.max_adaptive_iterations,
            resolution=args.resolution,
            rt_subframes=args.rt_subframes,
            timeout_seconds=args.timeout_seconds,
        )
    except CorrespondingMaterialColorWorkflowError as exc:
        print(f"Corresponding-material colour workflow failed: {exc}", file=sys.stderr)
        return 2
    status = manifest.get("quality_status")
    print(
        f"Corresponding-material colour workflow complete: quality={status}; "
        f"manifest={args.output_dir.expanduser().resolve() / 'workflow_manifest.json'}"
    )
    if args.require_pass and status != "PASS":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
