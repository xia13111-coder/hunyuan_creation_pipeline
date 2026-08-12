"""Independent locked-Look and collected-USD visual acceptance stage."""

from __future__ import annotations

import copy
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ...command import LogCallback, run_command
from ...runtime import isaac_python
from ..bundled_projects import (
    validate_bundled_acceptance_contract,
    validate_bundled_acceptance_evidence,
)
from ..camera import continuous_camera_view_specs as _continuous_camera_view_specs
from ..config import (
    VisualMaterialConfig,
    canonical_sha256,
    load_visual_material_config,
    read_object,
    write_object,
)
from ..contracts import (
    RESTORED_HISTORICAL_BASELINE,
    SEALED_BASELINE_EVIDENCE_SCHEMA,
)
from ..immutable_optimum import (
    DECISION as IMMUTABLE_LIBRARY_OPTIMUM_DECISION,
    evaluate_immutable_library_optimum,
)
from ..quality import (
    PART_ID_QUALITY_GATE_SCHEMA_VERSION,
    evaluate_part_id_quality_gate as _evaluate_part_id_quality_gate,
)
from ..references import sha256_file
from ..sealed_dependencies import verify_sealed_dependency_lock
from .common import require_file as _require_file
from .runner import _run_stage
from qwen_material_pipeline.evidence.final_visual_gate import (
    FinalVisualGateError,
    require_final_visual_gate_passed,
)
from qwen_material_pipeline.evidence.sealed_project import (
    SealedProjectBindingError,
    validate_sealed_project_binding,
)


CommandRunner = Callable[..., None]
ConfigLoader = Callable[[str | Path | None], VisualMaterialConfig]
IsaacPythonResolver = Callable[[], Path]
SealedDependencyVerifier = Callable[..., dict[str, Any]]

def _final_visual_file(value: Any, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise RuntimeError(f"{label} must be a non-empty file path")
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"{label} does not exist: {value}") from exc
    if not path.is_file():
        raise RuntimeError(f"{label} must be a file: {path}")
    return path


def _final_visual_axis_token(value: Any, label: str) -> str:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            isinstance(component, bool)
            or not isinstance(component, (int, float))
            for component in value
        )
    ):
        raise RuntimeError(f"{label} must be a three-number axis vector")
    vector = tuple(round(float(component), 6) for component in value)
    tokens = {
        (1.0, 0.0, 0.0): "x",
        (-1.0, 0.0, 0.0): "-x",
        (0.0, 1.0, 0.0): "y",
        (0.0, -1.0, 0.0): "-y",
        (0.0, 0.0, 1.0): "z",
        (0.0, 0.0, -1.0): "-z",
    }
    token = tokens.get(vector)
    if token is None:
        raise RuntimeError(f"{label} is not a supported canonical axis: {value!r}")
    return token


def _final_visual_render_contract(
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    render_set = registry.get("render_set")
    if not isinstance(render_set, Mapping):
        raise RuntimeError("Final visual baseline registry lacks render_set")
    resolution = render_set.get("resolution")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(
            isinstance(component, bool)
            or not isinstance(component, int)
            or component <= 0
            for component in resolution
        )
        or resolution[0] != resolution[1]
    ):
        raise RuntimeError(
            "Final visual baseline requires a square positive resolution"
        )
    requested = render_set.get("requested_view_tokens")
    if (
        not isinstance(requested, list)
        or not requested
        or any(not isinstance(item, str) or not item for item in requested)
    ):
        raise RuntimeError(
            "Final visual baseline registry lacks requested view tokens"
        )
    lighting_profile = render_set.get("lighting_profile")
    if lighting_profile not in {"geometry", "material-neutral"}:
        raise RuntimeError(
            "Final visual baseline registry has an unsupported lighting profile"
        )
    rt_subframes = render_set.get("rt_subframes")
    if (
        isinstance(rt_subframes, bool)
        or not isinstance(rt_subframes, int)
        or rt_subframes <= 0
    ):
        raise RuntimeError(
            "Final visual baseline registry lacks positive rt_subframes"
        )
    contract: dict[str, Any] = {
        "resolution": resolution[0],
        "views": ",".join(requested),
        "rt_subframes": rt_subframes,
        "lighting_profile": lighting_profile,
        "analysis_up_axis": _final_visual_axis_token(
            render_set.get("analysis_up_axis"),
            "Final visual baseline analysis_up_axis",
        ),
        "analysis_front_axis": _final_visual_axis_token(
            render_set.get("analysis_front_axis"),
            "Final visual baseline analysis_front_axis",
        ),
    }
    # Continuous registration deliberately uses reference-role IDs such as
    # ``side`` that are not canonical renderer presets.  Reconstruct the
    # complete camera contract from the rendered registry instead of keeping
    # only those IDs: replaying them through ``--views`` would either fail or
    # silently select a different camera.  The specs are materialized afresh
    # inside each final-acceptance round, so locked and collected evidence do
    # not depend on a mutable path from the candidate workspace.
    custom_view_specs = _continuous_camera_view_specs(registry)
    if custom_view_specs is not None:
        contract["view_specs"] = custom_view_specs
    return contract


def _final_visual_mapping(
    quality_report: Mapping[str, Any],
    *,
    allow_unscorable_unmapped_views: bool = False,
) -> tuple[dict[str, str], int, Path]:
    inputs = quality_report.get("inputs")
    thresholds = quality_report.get("thresholds")
    raw_views = quality_report.get("views")
    if (
        not isinstance(inputs, Mapping)
        or not isinstance(thresholds, Mapping)
        or not isinstance(raw_views, list)
        or not raw_views
    ):
        raise RuntimeError(
            "Final visual baseline quality report lacks inputs/thresholds/views"
        )
    raw_mapping = inputs.get("selected_view_mapping")
    if not isinstance(raw_mapping, Mapping) or not raw_mapping:
        raise RuntimeError(
            "Final visual baseline lacks a selected reference-view mapping"
        )
    mapping: dict[str, str] = {}
    for reference_id, render_id in raw_mapping.items():
        if (
            not isinstance(reference_id, str)
            or not reference_id
            or not isinstance(render_id, str)
            or not render_id
        ):
            raise RuntimeError("Final visual baseline view mapping is malformed")
        mapping[reference_id] = render_id
    reference_ids = {
        item.get("reference_view_id")
        for item in raw_views
        if isinstance(item, Mapping)
        and isinstance(item.get("reference_view_id"), str)
    }
    unscorable_reference_ids = {
        item.get("reference_view_id")
        for item in raw_views
        if isinstance(item, Mapping)
        and isinstance(item.get("reference_view_id"), str)
        and item.get("status") == "UNSCORABLE"
    }
    mapping_covers_required_views = (
        set(mapping) <= reference_ids
        and reference_ids - set(mapping) <= unscorable_reference_ids
        if allow_unscorable_unmapped_views
        else set(mapping) == reference_ids
    )
    if (
        len(reference_ids) != len(raw_views)
        or not mapping_covers_required_views
        or len(set(mapping.values())) != len(mapping)
    ):
        raise RuntimeError(
            "Final visual baseline mapping is not an exact one-to-one view cover"
        )
    minimum_comparable_views = thresholds.get("minimum_comparable_views")
    if (
        isinstance(minimum_comparable_views, bool)
        or not isinstance(minimum_comparable_views, int)
        or minimum_comparable_views <= 0
        or minimum_comparable_views > len(mapping)
    ):
        raise RuntimeError(
            "Final visual baseline has invalid minimum_comparable_views"
        )
    reference_manifest = _final_visual_file(
        inputs.get("reference_manifest"),
        "Final visual reference manifest",
    )
    return mapping, minimum_comparable_views, reference_manifest


def _require_manifest_bound_absolute_view_cover(
    *,
    quality_report: Path,
    reference_manifest: Path,
    rendered_registry: Path,
    label: str,
) -> None:
    """Require fresh evidence to cover the sealed manifest, not just itself."""

    manifest = read_object(reference_manifest, f"{label} reference manifest")
    raw_source_views = manifest.get("source_views")
    if not isinstance(raw_source_views, list) or not raw_source_views:
        raise RuntimeError(
            f"{label} reference manifest lacks non-empty source_views"
        )
    reference_ids: list[str] = []
    for index, raw_view in enumerate(raw_source_views):
        reference_id = (
            raw_view.get("id") if isinstance(raw_view, Mapping) else None
        )
        if not isinstance(reference_id, str) or not reference_id:
            raise RuntimeError(
                f"{label} reference manifest source_views[{index}] has "
                "an invalid ID"
            )
        reference_ids.append(reference_id)
    if len(reference_ids) != len(set(reference_ids)):
        raise RuntimeError(f"{label} reference manifest repeats source-view IDs")
    expected_reference_ids = set(reference_ids)

    registry = read_object(rendered_registry, f"{label} rendered registry")
    render_set = registry.get("render_set")
    raw_render_views = (
        render_set.get("views") if isinstance(render_set, Mapping) else None
    )
    if not isinstance(raw_render_views, list) or not raw_render_views:
        raise RuntimeError(f"{label} rendered registry lacks actual render views")
    render_ids: list[str] = []
    for index, raw_view in enumerate(raw_render_views):
        render_id = (
            raw_view.get("view_id") if isinstance(raw_view, Mapping) else None
        )
        if not isinstance(render_id, str) or not render_id:
            raise RuntimeError(
                f"{label} rendered registry view[{index}] has an invalid ID"
            )
        render_ids.append(render_id)
    if len(render_ids) != len(set(render_ids)):
        raise RuntimeError(f"{label} rendered registry repeats render-view IDs")
    actual_render_ids = set(render_ids)

    quality = read_object(quality_report, f"{label} quality report")
    inputs = quality.get("inputs")
    aggregate = quality.get("aggregate")
    raw_quality_views = quality.get("views")
    if (
        not isinstance(inputs, Mapping)
        or not isinstance(aggregate, Mapping)
        or not isinstance(raw_quality_views, list)
    ):
        raise RuntimeError(f"{label} quality report lacks inputs/aggregate/views")
    try:
        reported_manifest = _final_visual_file(
            inputs.get("reference_manifest"),
            f"{label} reported reference manifest",
        )
        reported_registry = _final_visual_file(
            inputs.get("rendered_registry"),
            f"{label} reported rendered registry",
        )
    except RuntimeError as exc:
        raise RuntimeError(f"{label} quality report has stale input bindings") from exc
    if (
        reported_manifest != reference_manifest
        or inputs.get("reference_manifest_sha256")
        != sha256_file(reference_manifest)
        or reported_registry != rendered_registry
        or inputs.get("rendered_registry_sha256") != sha256_file(rendered_registry)
    ):
        raise RuntimeError(f"{label} quality report input bindings are stale")

    raw_mapping = inputs.get("selected_view_mapping")
    if not isinstance(raw_mapping, Mapping):
        raise RuntimeError(f"{label} quality report lacks selected_view_mapping")
    mapping: dict[str, str] = {}
    for reference_id, render_id in raw_mapping.items():
        if (
            not isinstance(reference_id, str)
            or not reference_id
            or not isinstance(render_id, str)
            or not render_id
        ):
            raise RuntimeError(f"{label} quality report has a malformed view mapping")
        mapping[reference_id] = render_id
    if (
        set(mapping) != expected_reference_ids
        or len(set(mapping.values())) != len(mapping)
        or not set(mapping.values()) <= actual_render_ids
    ):
        raise RuntimeError(
            f"{label} quality report does not exactly cover every manifest view"
        )

    quality_views: dict[str, Mapping[str, Any]] = {}
    for index, raw_view in enumerate(raw_quality_views):
        reference_id = (
            raw_view.get("reference_view_id")
            if isinstance(raw_view, Mapping)
            else None
        )
        if (
            not isinstance(reference_id, str)
            or not reference_id
            or reference_id in quality_views
        ):
            raise RuntimeError(
                f"{label} quality report view[{index}] has an invalid or "
                "duplicate reference ID"
            )
        quality_views[reference_id] = raw_view
    expected_count = len(reference_ids)
    status_counts = {
        status: sum(
            view.get("status") == status for view in quality_views.values()
        )
        for status in ("PASS", "REVIEW", "FAIL", "UNSCORABLE")
    }
    comparable_count = sum(
        status_counts[status] for status in ("PASS", "REVIEW", "FAIL")
    )
    if (
        set(quality_views) != expected_reference_ids
        or any(
            view.get("render_view_id") != mapping[reference_id]
            for reference_id, view in quality_views.items()
        )
        or aggregate.get("reference_view_count") != expected_count
        or aggregate.get("comparable_view_count") != comparable_count
        or aggregate.get("passed_view_count") != status_counts["PASS"]
        or aggregate.get("review_view_count") != status_counts["REVIEW"]
        or aggregate.get("failed_view_count") != status_counts["FAIL"]
        or aggregate.get("unscorable_view_count") != status_counts["UNSCORABLE"]
    ):
        raise RuntimeError(
            f"{label} quality report view rows/counts do not exactly match "
            "the manifest-bound comparison"
        )


def _require_fresh_quality_pass(path: Path, label: str) -> dict[str, Any]:
    report = read_object(path, label)
    aggregate = report.get("aggregate")
    raw_views = report.get("views")
    if (
        not isinstance(aggregate, Mapping)
        or aggregate.get("status") != "PASS"
        or not isinstance(raw_views, list)
        or not raw_views
        or any(
            not isinstance(view, Mapping) or view.get("status") != "PASS"
            for view in raw_views
        )
    ):
        status = (
            aggregate.get("status")
            if isinstance(aggregate, Mapping)
            else None
        )
        raise RuntimeError(
            f"{label} did not PASS every view (aggregate={status!r})"
        )
    return report


def _require_fresh_quality_accepted(
    path: Path,
    label: str,
    *,
    config: VisualMaterialConfig,
    allow_immutable_library_optimum_review: bool,
    allow_part_id_quality: bool = False,
) -> dict[str, Any]:
    report = read_object(path, label)
    if allow_part_id_quality:
        audit = _evaluate_part_id_quality_gate(
            report,
            minimum_aggregate_appearance_score=(
                config.final_visual_gate_minimum_final_appearance_score
            ),
            minimum_view_appearance_score=(
                config.final_visual_gate_minimum_final_view_appearance_score
            ),
            minimum_comparable_views=2,
        )
        if audit.get("acceptance_allowed") is not True:
            raise RuntimeError(
                f"{label} failed independent Part-ID visual QA: "
                f"{audit.get('reason_codes')!r}"
            )
        return report
    if not allow_immutable_library_optimum_review:
        return _require_fresh_quality_pass(path, label)
    aggregate = report.get("aggregate")
    if isinstance(aggregate, Mapping) and aggregate.get("status") == "PASS":
        return _require_fresh_quality_pass(path, label)
    audit = evaluate_immutable_library_optimum(
        report,
        minimum_aggregate_appearance_score=(
            config.final_visual_gate_minimum_final_appearance_score
        ),
        minimum_view_appearance_score=(
            config.final_visual_gate_minimum_final_view_appearance_score
        ),
    )
    if audit.get("acceptance_allowed") is not True:
        raise RuntimeError(
            f"{label} did not reproduce the accepted immutable-library "
            f"optimum: {audit.get('reason_codes')!r}"
        )
    return report


def _validated_immutable_library_optimum_result(
    *,
    visual_material_result: Mapping[str, Any],
    material_output: Path,
) -> bool:
    if visual_material_result.get("immutable_library_optimum_accepted") is not True:
        return False
    if (
        visual_material_result.get("visual_quality_decision")
        != IMMUTABLE_LIBRARY_OPTIMUM_DECISION
        or visual_material_result.get("visual_quality_raw_status") != "REVIEW"
        or visual_material_result.get("visual_quality_gate_status") != "PASS"
        or visual_material_result.get("immutable_mdl_after_selection") is not True
        or visual_material_result.get("material_selection_lock_verified") is not True
        or visual_material_result.get("publish_quality_gate_status") != "PASS"
    ):
        raise RuntimeError(
            "Immutable-library optimum result has an incomplete acceptance contract"
        )
    gate = _final_visual_file(
        visual_material_result.get("immutable_library_optimum_gate"),
        "Immutable-library optimum gate",
    )
    if not gate.is_relative_to(material_output):
        raise RuntimeError(
            "Immutable-library optimum gate must be inside the material output"
        )
    document = read_object(gate, "immutable-library optimum gate")
    if (
        document.get("schema_version")
        != "asset-pipeline-immutable-library-optimum/v1"
        or document.get("status") != "PASS"
        or document.get("acceptance_allowed") is not True
        or document.get("decision") != IMMUTABLE_LIBRARY_OPTIMUM_DECISION
    ):
        raise RuntimeError("Immutable-library optimum gate is not accepted")
    bindings = document.get("bindings")
    if not isinstance(bindings, Mapping):
        raise RuntimeError("Immutable-library optimum gate lacks bindings")
    quality = _final_visual_file(
        visual_material_result.get("visual_quality_report"),
        "Immutable-library optimum quality report",
    )
    if (
        bindings.get("quality_report") != str(quality)
        or bindings.get("quality_report_sha256") != sha256_file(quality)
    ):
        raise RuntimeError(
            "Immutable-library optimum gate is not bound to selected quality"
        )
    return True


def _sealed_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validated_sealed_historical_baseline_evidence(
    *,
    visual_material_result: Mapping[str, Any],
    material_output: Path,
    isaac: Path,
    dependency_verifier: SealedDependencyVerifier,
) -> Path | None:
    """Return evidence only for a fully hash-bound bundled historical result."""

    inference_mode = visual_material_result.get("inference_mode")
    quality_status = visual_material_result.get("visual_quality_status")
    has_bundled_marker = inference_mode == "bundled_project"
    has_historical_marker = quality_status == RESTORED_HISTORICAL_BASELINE
    if not has_bundled_marker and not has_historical_marker:
        return None
    if not has_bundled_marker or not has_historical_marker:
        raise RuntimeError(
            "Sealed baseline preservation requires both "
            "inference_mode='bundled_project' and "
            "visual_quality_status='RESTORED_HISTORICAL_BASELINE'"
        )
    requested_inference_mode = visual_material_result.get(
        "requested_inference_mode"
    )
    if requested_inference_mode not in {"auto", "bundled"}:
        raise RuntimeError(
            "Sealed baseline preservation requires an explicit auto or "
            "bundled request, never live"
        )
    if visual_material_result.get("staged_state") != "READY_TO_APPLY":
        raise RuntimeError(
            "Sealed baseline preservation requires a READY_TO_APPLY replay"
        )

    evidence_path = _final_visual_file(
        visual_material_result.get("sealed_qwen_mvinverse_evidence"),
        "Sealed Qwen/MVInverse evidence",
    )
    if not evidence_path.is_relative_to(material_output):
        raise RuntimeError(
            "Sealed Qwen/MVInverse evidence must be inside the material output"
        )
    evidence = read_object(evidence_path, "sealed Qwen/MVInverse evidence")
    asset_id = visual_material_result.get("material_project")
    if not isinstance(asset_id, str) or not asset_id:
        raise RuntimeError("Bundled visual result lacks material_project")
    if (
        evidence.get("schema_version") != SEALED_BASELINE_EVIDENCE_SCHEMA
        or evidence.get("asset_id") != asset_id
        or evidence.get("live_inference_repeated") is not False
    ):
        raise RuntimeError(
            "Sealed Qwen/MVInverse evidence has invalid replay provenance"
        )

    source_cad = _final_visual_file(
        visual_material_result.get("source_cad"),
        "Bundled source CAD",
    )
    source_cad_sha256 = sha256_file(source_cad)
    if (
        _sealed_sha256(
            visual_material_result.get("source_cad_sha256"),
            "Bundled result source_cad_sha256",
        )
        != source_cad_sha256
        or _sealed_sha256(
            evidence.get("source_cad_sha256"),
            "Sealed evidence source_cad_sha256",
        )
        != source_cad_sha256
    ):
        raise RuntimeError("Sealed baseline source CAD hash is stale")

    raw_references = visual_material_result.get("references")
    if not isinstance(raw_references, list) or len(raw_references) < 2:
        raise RuntimeError(
            "Sealed baseline requires at least two hash-bound references"
        )
    reference_sha256: dict[str, str] = {}
    reference_paths_by_role: dict[str, Path] = {}
    for index, raw_reference in enumerate(raw_references):
        if not isinstance(raw_reference, Mapping):
            raise RuntimeError(
                f"Bundled result reference[{index}] must be an object"
            )
        reference_id = raw_reference.get("id")
        if (
            not isinstance(reference_id, str)
            or not reference_id
            or reference_id in reference_sha256
        ):
            raise RuntimeError(
                "Sealed baseline references require unique non-empty IDs"
            )
        image = _final_visual_file(
            raw_reference.get("image"),
            f"Bundled result reference {reference_id}",
        )
        actual_sha256 = sha256_file(image)
        if (
            _sealed_sha256(
                raw_reference.get("sha256"),
                f"Bundled result reference {reference_id} SHA-256",
            )
            != actual_sha256
        ):
            raise RuntimeError(
                f"Sealed baseline reference {reference_id} hash is stale"
            )
        reference_sha256[reference_id] = actual_sha256
        reference_paths_by_role[reference_id] = image
    evidence_references = evidence.get("reference_sha256")
    if not isinstance(evidence_references, Mapping) or dict(
        evidence_references
    ) != reference_sha256:
        raise RuntimeError(
            "Sealed evidence reference hashes do not match the bundled result"
        )

    plan_path = _final_visual_file(
        visual_material_result.get("material_plan"),
        "Bundled material plan",
    )
    audit_path = _final_visual_file(
        visual_material_result.get("project_material_audit"),
        "Bundled project material audit",
    )
    if not plan_path.is_relative_to(material_output) or not audit_path.is_relative_to(
        material_output
    ):
        raise RuntimeError(
            "Bundled plan and audit must be inside the material output"
        )
    plan = read_object(plan_path, "bundled material plan")
    audit = read_object(audit_path, "bundled project material audit")
    plan_provenance = plan.get("provenance")
    if not isinstance(plan_provenance, Mapping):
        raise RuntimeError("Sealed baseline material plan lacks provenance")
    try:
        project_binding = validate_sealed_project_binding(
            evidence,
            expected_project_sha256=plan_provenance.get("project_sha256"),
            expected_project_path=audit.get("project"),
        )
    except SealedProjectBindingError as exc:
        raise RuntimeError(
            f"Sealed Qwen/MVInverse evidence has invalid project binding: {exc}"
        ) from exc
    result_project = _final_visual_file(
        visual_material_result.get("material_project_manifest"),
        "Bundled result project manifest",
    )
    if (
        str(result_project) != project_binding["project"]
        or _sealed_sha256(
            visual_material_result.get("material_project_manifest_sha256"),
            "Bundled result project manifest SHA-256",
        )
        != project_binding["project_sha256"]
    ):
        raise RuntimeError(
            "Bundled result project manifest is stale or inconsistent"
        )
    project_document = read_object(
        result_project,
        "bundled result project manifest",
    )
    try:
        project_acceptance = validate_bundled_acceptance_contract(
            project_document
        )
        project_acceptance_evidence = validate_bundled_acceptance_evidence(
            project_document,
            project_file=result_project,
            reference_paths_by_role=reference_paths_by_role,
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Bundled project acceptance contract is invalid: {exc}"
        ) from exc
    result_acceptance = visual_material_result.get(
        "material_project_acceptance"
    )
    if (
        not isinstance(result_acceptance, Mapping)
        or dict(result_acceptance) != project_acceptance
        or _sealed_sha256(
            visual_material_result.get(
                "material_project_acceptance_sha256"
            ),
            "Bundled result acceptance contract SHA-256",
        )
        != canonical_sha256(project_acceptance)
    ):
        raise RuntimeError(
            "Bundled result acceptance contract differs from its "
            "hash-bound project"
        )
    result_acceptance_evidence = visual_material_result.get(
        "material_project_acceptance_evidence"
    )
    if project_acceptance_evidence is None:
        evidence_mismatch = result_acceptance_evidence is not None
    else:
        evidence_mismatch = (
            not isinstance(result_acceptance_evidence, Mapping)
            or dict(result_acceptance_evidence) != project_acceptance_evidence
            or _sealed_sha256(
                visual_material_result.get(
                    "material_project_acceptance_evidence_sha256"
                ),
                "Bundled result acceptance evidence SHA-256",
            )
            != canonical_sha256(project_acceptance_evidence)
            or project_binding.get("acceptance_evidence")
            != project_acceptance_evidence
        )
    if evidence_mismatch:
        raise RuntimeError(
            "Bundled result acceptance evidence differs from its "
            "hash-bound project"
        )
    if (
        plan_provenance.get("template_sha256")
        != project_binding["artifact_sha256"]["template"]
        or plan_provenance.get("historical_result_sha256")
        != project_binding["historical_result_sha256"]
        or plan_provenance.get("source_cad_sha256") != source_cad_sha256
        or plan_provenance.get("reference_sha256") != reference_sha256
    ):
        raise RuntimeError(
            "Sealed baseline plan provenance does not match its project or inputs"
        )
    plan_sha256 = canonical_sha256(plan)
    audit_sha256 = canonical_sha256(audit)
    historical_sha256 = _sealed_sha256(
        evidence.get("historical_result_sha256"),
        "Sealed evidence historical_result_sha256",
    )
    if (
        _sealed_sha256(evidence.get("plan_sha256"), "Sealed evidence plan_sha256")
        != plan_sha256
        or _sealed_sha256(
            evidence.get("audit_sha256"),
            "Sealed evidence audit_sha256",
        )
        != audit_sha256
        or audit.get("status") != "PASS"
        or audit.get("asset_id") != asset_id
        or audit.get("method") != project_binding["method"]
        or audit.get("historical_result_sha256") != historical_sha256
        or audit.get("plan_sha256") != plan_sha256
        or audit.get("complete_coverage") is not True
        or audit.get("topology_verified") is not True
        or audit.get("face_subsets_verified") is not True
    ):
        raise RuntimeError(
            "Sealed baseline plan/audit provenance is invalid or stale"
        )

    assignment_count = visual_material_result.get("assignment_count")
    applied_count = visual_material_result.get("applied_count")
    if (
        isinstance(assignment_count, bool)
        or not isinstance(assignment_count, int)
        or assignment_count <= 0
        or applied_count != assignment_count
        or audit.get("part_count") != assignment_count
    ):
        raise RuntimeError(
            "Sealed baseline does not prove complete material application"
        )

    result_catalog: Path | None = None
    for artifact in ("template", "catalog"):
        artifact_path = _final_visual_file(
            evidence.get(artifact),
            f"Sealed evidence {artifact}",
        )
        artifact_sha256 = _sealed_sha256(
            evidence.get(f"{artifact}_sha256"),
            f"Sealed evidence {artifact}_sha256",
        )
        if sha256_file(artifact_path) != artifact_sha256:
            raise RuntimeError(f"Sealed baseline {artifact} hash is stale")
        if artifact == "catalog":
            result_catalog = _final_visual_file(
                visual_material_result.get("catalog"),
                "Bundled result catalog",
            )
            if result_catalog != artifact_path:
                raise RuntimeError(
                    "Sealed evidence catalog does not match the bundled result"
                )
    if result_catalog is None:
        raise AssertionError("Sealed catalog validation did not run")

    if (
        visual_material_result.get("dependency_lock_verified") is not True
        or evidence.get("dependency_lock_verified") is not True
        or visual_material_result.get("dependency_lock_verification_status")
        != "PASS"
        or evidence.get("dependency_lock_verification_status") != "PASS"
    ):
        raise RuntimeError(
            "Sealed baseline dependency verification is absent or non-PASS"
        )
    dependency_lock = _final_visual_file(
        visual_material_result.get("dependency_lock"),
        "Bundled dependency lock",
    )
    evidence_dependency_lock = _final_visual_file(
        evidence.get("dependency_lock"),
        "Sealed evidence dependency lock",
    )
    dependency_lock_sha256 = sha256_file(dependency_lock)
    if (
        evidence_dependency_lock != dependency_lock
        or _sealed_sha256(
            visual_material_result.get("dependency_lock_sha256"),
            "Bundled result dependency_lock_sha256",
        )
        != dependency_lock_sha256
        or _sealed_sha256(
            evidence.get("dependency_lock_sha256"),
            "Sealed evidence dependency_lock_sha256",
        )
        != dependency_lock_sha256
    ):
        raise RuntimeError("Sealed baseline dependency lock is stale")

    verification_report = _final_visual_file(
        visual_material_result.get("dependency_lock_verification_report"),
        "Bundled dependency verification report",
    )
    evidence_verification_report = _final_visual_file(
        evidence.get("dependency_lock_verification_report"),
        "Sealed evidence dependency verification report",
    )
    if (
        not verification_report.is_relative_to(material_output)
        or evidence_verification_report != verification_report
    ):
        raise RuntimeError(
            "Sealed dependency verification report is outside the material "
            "output or differs from the result"
        )
    verification_report_sha256 = sha256_file(verification_report)
    if (
        _sealed_sha256(
            visual_material_result.get(
                "dependency_lock_verification_report_sha256"
            ),
            "Bundled result dependency verification report SHA-256",
        )
        != verification_report_sha256
        or _sealed_sha256(
            evidence.get("dependency_lock_verification_report_sha256"),
            "Sealed evidence dependency verification report SHA-256",
        )
        != verification_report_sha256
    ):
        raise RuntimeError("Sealed dependency verification report is stale")
    verification = read_object(
        verification_report,
        "sealed dependency verification report",
    )
    embedded_verification = evidence.get("dependency_lock_verification")
    if (
        verification.get("schema_version")
        != "qwen-sealed-material-dependency-verification/v1"
        or verification.get("status") != "PASS"
        or verification.get("dependency_lock_verified") is not True
        or verification.get("lock_path") != str(dependency_lock)
        or verification.get("lock_sha256") != dependency_lock_sha256
        or verification.get("catalog_path") != str(result_catalog)
        or embedded_verification != verification
    ):
        raise RuntimeError(
            "Sealed dependency verification report is invalid or inconsistent"
        )

    material_root_value = visual_material_result.get("material_root")
    if not isinstance(material_root_value, str) or not material_root_value:
        raise RuntimeError("Bundled visual result lacks material_root")
    try:
        material_root = Path(material_root_value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"Bundled material root does not exist: {material_root_value}"
        ) from exc
    if not material_root.is_dir() or verification.get("material_root") != str(
        material_root
    ):
        raise RuntimeError(
            "Sealed dependency verification material root is invalid"
        )
    fresh_verification = dependency_verifier(
        lock_path=dependency_lock,
        expected_lock_sha256=dependency_lock_sha256,
        catalog_path=result_catalog,
        material_root=material_root,
        isaac_root=isaac.parent,
        expected_asset_id=asset_id,
    )
    if fresh_verification != verification:
        raise RuntimeError(
            "Sealed dependencies changed after material application"
        )
    return evidence_path


def _final_visual_reference_manifest_from_result(
    *,
    visual_material_result: Mapping[str, Any],
    output: Path,
) -> Path:
    raw_references = visual_material_result.get("references")
    if not isinstance(raw_references, list) or len(raw_references) < 2:
        raise RuntimeError(
            "Legacy visual result needs at least two recorded references to "
            "establish a fresh locked baseline"
        )
    raw_acceptance_evidence = visual_material_result.get(
        "material_project_acceptance_evidence"
    )
    evidence_by_id: dict[str, Mapping[str, Any]] = {}
    if raw_acceptance_evidence is not None:
        if (
            not isinstance(raw_acceptance_evidence, Mapping)
            or raw_acceptance_evidence.get("schema_version")
            != "qwen-bundled-acceptance-evidence/v1"
            or not isinstance(raw_acceptance_evidence.get("source_views"), list)
        ):
            raise RuntimeError("Bundled visual result acceptance evidence is invalid")
        for index, raw_evidence in enumerate(
            raw_acceptance_evidence["source_views"]
        ):
            if not isinstance(raw_evidence, Mapping):
                raise RuntimeError(
                    f"Bundled acceptance evidence source_views[{index}] is invalid"
                )
            evidence_id = raw_evidence.get("id")
            if (
                not isinstance(evidence_id, str)
                or not evidence_id
                or evidence_id in evidence_by_id
            ):
                raise RuntimeError(
                    "Bundled acceptance evidence IDs are invalid or duplicated"
                )
            evidence_by_id[evidence_id] = raw_evidence

    source_views: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for index, raw in enumerate(raw_references):
        if not isinstance(raw, Mapping):
            raise RuntimeError(
                f"Visual result reference[{index}] must be an object"
            )
        reference_id = raw.get("id")
        if (
            not isinstance(reference_id, str)
            or not reference_id
            or reference_id in seen_ids
        ):
            raise RuntimeError(
                "Visual result references need unique non-empty IDs"
            )
        image = _final_visual_file(
            raw.get("image"),
            f"Visual result reference {reference_id}",
        )
        if image in seen_paths:
            raise RuntimeError("Visual result references repeat an image file")
        seen_ids.add(reference_id)
        seen_paths.add(image)
        view: dict[str, Any] = {"id": reference_id, "image": str(image)}
        if evidence_by_id:
            evidence_view = evidence_by_id.get(reference_id)
            if evidence_view is None:
                raise RuntimeError(
                    f"Bundled acceptance evidence is missing {reference_id}"
                )
            reference_sha256 = sha256_file(image)
            if (
                evidence_view.get("palette_status") != "usable"
                or _sealed_sha256(
                    evidence_view.get("reference_sha256"),
                    f"Bundled acceptance evidence {reference_id} reference SHA-256",
                )
                != reference_sha256
            ):
                raise RuntimeError(
                    f"Bundled acceptance evidence {reference_id} is stale"
                )
            mask = _final_visual_file(
                evidence_view.get("palette_mask"),
                f"Bundled acceptance evidence {reference_id} mask",
            )
            palette = _final_visual_file(
                evidence_view.get("palette_path"),
                f"Bundled acceptance evidence {reference_id} palette",
            )
            raw_artifacts = evidence_view.get("palette_artifacts")
            if not isinstance(raw_artifacts, Mapping):
                raise RuntimeError(
                    f"Bundled acceptance evidence {reference_id} artifacts are invalid"
                )
            normalized = _final_visual_file(
                raw_artifacts.get("normalized"),
                f"Bundled acceptance evidence {reference_id} normalized palette",
            )
            audit = _final_visual_file(
                raw_artifacts.get("normalized_evidence_audit"),
                f"Bundled acceptance evidence {reference_id} audit",
            )
            checks = (
                (
                    mask,
                    evidence_view.get("palette_mask_sha256"),
                    "mask",
                ),
                (
                    palette,
                    evidence_view.get("palette_path_sha256"),
                    "palette",
                ),
                (
                    normalized,
                    raw_artifacts.get("normalized_sha256"),
                    "normalized palette",
                ),
                (
                    audit,
                    raw_artifacts.get("normalized_evidence_audit_sha256"),
                    "normalized evidence audit",
                ),
            )
            for artifact, expected_hash, label in checks:
                if _sealed_sha256(
                    expected_hash,
                    f"Bundled acceptance evidence {reference_id} {label} SHA-256",
                ) != sha256_file(artifact):
                    raise RuntimeError(
                        f"Bundled acceptance evidence {reference_id} {label} changed"
                    )
            if palette != normalized:
                raise RuntimeError(
                    f"Bundled acceptance evidence {reference_id} palette differs "
                    "from normalized"
                )
            view.update(
                {
                    "palette_status": "usable",
                    "palette_mask": str(mask),
                    "palette_path": str(palette),
                    "palette_artifacts": {
                        "normalized": str(normalized),
                        "normalized_evidence_audit": str(audit),
                    },
                }
            )
        source_views.append(view)
    if evidence_by_id and set(evidence_by_id) != seen_ids:
        raise RuntimeError(
            "Bundled acceptance evidence roles differ from result references"
        )
    write_object(
        output,
        {
            "schema_version": "qwen-reference-manifest/v1",
            "source_views": source_views,
            "source": (
                "sealed_project_hash_bound_acceptance_evidence"
                if evidence_by_id
                else "legacy_result_fresh_locked_baseline"
            ),
        },
    )
    return output.resolve(strict=True)


def _final_visual_config_render_contract(
    config: VisualMaterialConfig,
) -> dict[str, Any]:
    return {
        "resolution": config.render_resolution,
        "views": config.render_views,
        "rt_subframes": config.render_rt_subframes,
        "lighting_profile": config.quality_lighting_profile,
        "analysis_up_axis": config.analysis_up_axis,
        "analysis_front_axis": config.analysis_front_axis,
    }


def _run_final_visual_render_round(
    *,
    name: str,
    asset_usd: Path,
    output_dir: Path,
    reference_manifest: Path,
    view_mapping: Mapping[str, str] | None,
    minimum_comparable_views: int,
    render_contract: Mapping[str, Any],
    isaac: Path,
    qwen_python: Path,
    config: VisualMaterialConfig,
    require_absolute_pass: bool,
    require_manifest_exact_view_cover: bool,
    allow_immutable_library_optimum_review: bool,
    allow_part_id_quality: bool,
    log_cb: LogCallback,
    command_runner: CommandRunner,
    part_id_quality_scope: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    if output_dir.exists():
        raise FileExistsError(
            f"Final visual render output already exists: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    registry = output_dir / "part_registry.json"
    render_dir = output_dir / "renders"
    rendered_registry = render_dir / "part_registry.rendered.json"
    view_map = output_dir / "reference_view_map.json"
    quality_report = output_dir / "reference_render_comparison.json"
    custom_view_specs = render_contract.get("view_specs")
    if custom_view_specs is None:
        render_view_arguments = ["--views", str(render_contract["views"])]
    else:
        if not isinstance(custom_view_specs, Mapping):
            raise RuntimeError(
                f"{name} custom camera view specs must be an object"
            )
        custom_view_specs_document = copy.deepcopy(dict(custom_view_specs))
        if custom_view_specs_document.get("schema_version") != (
            "qwen-camera-view-specs/v1"
        ):
            raise RuntimeError(
                f"{name} custom camera view specs have an unsupported schema"
            )
        raw_custom_views = custom_view_specs_document.get("views")
        custom_view_ids = (
            [
                row.get("view_id")
                for row in raw_custom_views
                if isinstance(row, Mapping)
            ]
            if isinstance(raw_custom_views, list)
            else []
        )
        expected_view_ids = str(render_contract["views"]).split(",")
        if custom_view_ids != expected_view_ids:
            raise RuntimeError(
                f"{name} custom camera IDs differ from the render contract: "
                f"expected={expected_view_ids} actual={custom_view_ids}"
            )
        round_view_specs = output_dir / "camera_view_specs.json"
        write_object(round_view_specs, custom_view_specs_document)
        render_view_arguments = ["--view-specs", str(round_view_specs)]

    _run_stage(
        f"{name}_registry",
        [
            str(isaac),
            "-m",
            "qwen_material_pipeline",
            "usd",
            "registry",
            "--usd",
            str(asset_usd),
            "--output",
            str(registry),
        ],
        log_cb,
        command_runner=command_runner,
    )
    _require_file(registry, f"{name}_registry")
    _run_stage(
        f"{name}_render",
        [
            str(isaac),
            "-m",
            "qwen_material_pipeline",
            "usd",
            "render",
            "--registry",
            str(registry),
            "--output-dir",
            str(render_dir),
            "--resolution",
            str(render_contract["resolution"]),
            *render_view_arguments,
            "--rt-subframes",
            str(render_contract["rt_subframes"]),
            "--lighting-profile",
            str(render_contract["lighting_profile"]),
            "--analysis-up-axis",
            str(render_contract["analysis_up_axis"]),
            (
                "--analysis-front-axis="
                f"{render_contract['analysis_front_axis']}"
            ),
        ],
        log_cb,
        command_runner=command_runner,
        retry_native_crash=True,
    )
    _require_file(rendered_registry, f"{name}_render")

    compare_command = [
        str(qwen_python),
        "-m",
        "qwen_material_pipeline",
        "compare",
        "--reference-manifest",
        str(reference_manifest),
        "--rendered-registry",
        str(rendered_registry),
        "--minimum-comparable-views",
        str(minimum_comparable_views),
        "--output",
        str(quality_report),
    ]
    if view_mapping is not None:
        write_object(
            view_map,
            {
                "schema_version": "qwen-reference-view-map/v1",
                "mapping": dict(sorted(view_mapping.items())),
                "source": "final_locked_baseline_mapping",
            },
        )
        compare_command.extend(["--view-map", str(view_map)])
    _run_stage(
        f"{name}_compare",
        compare_command,
        log_cb,
        command_runner=command_runner,
        retry_native_crash=True,
    )
    _require_file(quality_report, f"{name}_compare")
    if allow_part_id_quality and part_id_quality_scope is not None:
        scoped_quality = read_object(quality_report, f"{name} quality report")
        scoped_quality["part_id_quality_scope"] = copy.deepcopy(
            dict(part_id_quality_scope)
        )
        write_object(quality_report, scoped_quality)
    if require_manifest_exact_view_cover:
        _require_manifest_bound_absolute_view_cover(
            quality_report=quality_report.resolve(strict=True),
            reference_manifest=reference_manifest.resolve(strict=True),
            rendered_registry=rendered_registry.resolve(strict=True),
            label=name,
        )
    if require_absolute_pass:
        _require_fresh_quality_accepted(
            quality_report,
            f"{name} quality report",
            config=config,
            allow_immutable_library_optimum_review=(
                allow_immutable_library_optimum_review
            ),
            allow_part_id_quality=allow_part_id_quality,
        )
    else:
        report = read_object(quality_report, f"{name} quality report")
        if report.get("schema_version") != "qwen-reference-render-comparison/v1":
            raise RuntimeError(
                f"{name} quality report has an unsupported schema"
            )
    return {
        "registry": registry.resolve(strict=True),
        "rendered_registry": rendered_registry.resolve(strict=True),
        "view_map": (
            view_map.resolve(strict=True) if view_map.is_file() else view_map
        ),
        "quality_report": quality_report.resolve(strict=True),
    }


def _run_part_id_final_visual_gate_stage(
    *,
    name: str,
    final_usd: Path,
    baseline_quality_report: Path,
    baseline_rendered_registry: Path,
    final_quality_report: Path,
    final_rendered_registry: Path,
    output: Path,
    config: VisualMaterialConfig,
    allow_same_baseline_asset: bool,
) -> dict[str, Any]:
    """Verify fresh Part-ID delivery renders without palette-group semantics."""

    baseline_quality = read_object(
        baseline_quality_report, f"{name} baseline quality"
    )
    final_quality = read_object(final_quality_report, f"{name} final quality")
    baseline_registry = read_object(
        baseline_rendered_registry, f"{name} baseline registry"
    )
    final_registry = read_object(final_rendered_registry, f"{name} final registry")
    reasons: list[str] = []
    if (
        baseline_quality.get("schema_version")
        != "qwen-reference-render-comparison/v1"
        or final_quality.get("schema_version")
        != "qwen-reference-render-comparison/v1"
        or baseline_registry.get("schema_version") != "qwen-material-parts/v1"
        or final_registry.get("schema_version") != "qwen-material-parts/v1"
    ):
        reasons.append("INPUT_SCHEMA_INVALID")

    baseline_gate = _evaluate_part_id_quality_gate(
        baseline_quality,
        minimum_aggregate_appearance_score=(
            config.final_visual_gate_minimum_final_appearance_score
        ),
        minimum_view_appearance_score=(
            config.final_visual_gate_minimum_final_view_appearance_score
        ),
        minimum_comparable_views=2,
    )
    final_gate = _evaluate_part_id_quality_gate(
        final_quality,
        minimum_aggregate_appearance_score=(
            config.final_visual_gate_minimum_final_appearance_score
        ),
        minimum_view_appearance_score=(
            config.final_visual_gate_minimum_final_view_appearance_score
        ),
        minimum_comparable_views=2,
    )
    if baseline_gate.get("acceptance_allowed") is not True:
        reasons.append("BASELINE_PART_ID_QUALITY_NOT_ACCEPTED")
    if final_gate.get("acceptance_allowed") is not True:
        reasons.append("FINAL_PART_ID_QUALITY_NOT_ACCEPTED")
    if baseline_gate.get("view_scope") != final_gate.get("view_scope"):
        reasons.append("PART_ID_QUALITY_VIEW_SCOPE_CHANGED")

    baseline_inputs = baseline_quality.get("inputs")
    final_inputs = final_quality.get("inputs")
    if not isinstance(baseline_inputs, Mapping) or not isinstance(
        final_inputs, Mapping
    ):
        reasons.append("QUALITY_INPUT_BINDINGS_INVALID")
    else:
        for quality_inputs, registry_path, label in (
            (baseline_inputs, baseline_rendered_registry, "BASELINE"),
            (final_inputs, final_rendered_registry, "FINAL"),
        ):
            try:
                recorded_registry = Path(
                    str(quality_inputs.get("rendered_registry"))
                ).expanduser().resolve(strict=True)
            except (OSError, RuntimeError):
                recorded_registry = None
            if (
                recorded_registry != registry_path.resolve(strict=True)
                or quality_inputs.get("rendered_registry_sha256")
                != sha256_file(registry_path)
            ):
                reasons.append(f"{label}_QUALITY_REGISTRY_BINDING_INVALID")
        if (
            baseline_inputs.get("reference_manifest_sha256")
            != final_inputs.get("reference_manifest_sha256")
        ):
            reasons.append("REFERENCE_MANIFEST_CHANGED")

    try:
        final_asset = Path(str(final_registry.get("asset_usd"))).expanduser().resolve(
            strict=True
        )
        baseline_asset = Path(
            str(baseline_registry.get("asset_usd"))
        ).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        final_asset = None
        baseline_asset = None
        reasons.append("RENDERED_ASSET_BINDING_INVALID")
    if final_asset != final_usd.resolve(strict=True):
        reasons.append("FINAL_REGISTRY_NOT_BOUND_TO_EXPECTED_USD")
    if not allow_same_baseline_asset and baseline_asset == final_asset:
        reasons.append("BASELINE_ASSET_IS_FINAL_ASSET")
    if baseline_rendered_registry.resolve() == final_rendered_registry.resolve():
        reasons.append("FINAL_RENDERED_REGISTRY_REUSED_BASELINE")
    if baseline_quality_report.resolve() == final_quality_report.resolve():
        reasons.append("FINAL_QUALITY_REPORT_REUSED_BASELINE")
    if baseline_registry.get("render_set") != final_registry.get("render_set"):
        baseline_render_set = baseline_registry.get("render_set")
        final_render_set = final_registry.get("render_set")
        contract_fields = (
            "resolution",
            "analysis_up_axis",
            "analysis_front_axis",
            "lighting_profile",
            "requested_view_tokens",
            "rt_subframes",
        )
        if (
            not isinstance(baseline_render_set, Mapping)
            or not isinstance(final_render_set, Mapping)
            or any(
                baseline_render_set.get(field) != final_render_set.get(field)
                for field in contract_fields
            )
        ):
            reasons.append("RENDER_CONTRACT_CHANGED")
    try:
        baseline_camera_specs = _continuous_camera_view_specs(baseline_registry)
        final_camera_specs = _continuous_camera_view_specs(final_registry)
    except RuntimeError:
        reasons.append("CUSTOM_CAMERA_CONTRACT_INVALID")
    else:
        if baseline_camera_specs != final_camera_specs:
            reasons.append("CUSTOM_CAMERA_CONTRACT_CHANGED")

    baseline_mapping = (
        baseline_inputs.get("selected_view_mapping")
        if isinstance(baseline_inputs, Mapping)
        else None
    )
    final_mapping = (
        final_inputs.get("selected_view_mapping")
        if isinstance(final_inputs, Mapping)
        else None
    )
    if (
        not isinstance(baseline_mapping, Mapping)
        or not isinstance(final_mapping, Mapping)
        or dict(baseline_mapping) != dict(final_mapping)
    ):
        reasons.append("SELECTED_VIEW_MAPPING_CHANGED")

    def scored_views(gate: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        measurements = gate.get("measurements")
        raw = measurements.get("views") if isinstance(measurements, Mapping) else None
        rows = raw if isinstance(raw, list) else []
        return {
            str(view["reference_view_id"]): view
            for view in rows
            if isinstance(view, Mapping)
            and isinstance(view.get("reference_view_id"), str)
            and view.get("quality_gate_enforced") is not False
        }

    baseline_views = scored_views(baseline_gate)
    final_views = scored_views(final_gate)
    if set(baseline_views) != set(final_views):
        reasons.append("COMPARABLE_VIEW_SET_CHANGED")
    score_regressions: dict[str, float] = {}
    maximum_regression = float(config.final_visual_gate_maximum_score_regression)
    for view_id in sorted(set(baseline_views) & set(final_views)):
        delta = float(final_views[view_id]["material_appearance_score"]) - float(
            baseline_views[view_id]["material_appearance_score"]
        )
        if delta < -maximum_regression:
            score_regressions[view_id] = delta
    baseline_measurements = baseline_gate.get("measurements")
    final_measurements = final_gate.get("measurements")
    baseline_aggregate_score = (
        baseline_measurements.get("aggregate_appearance_score")
        if isinstance(baseline_measurements, Mapping)
        else None
    )
    final_aggregate_score = (
        final_measurements.get("aggregate_appearance_score")
        if isinstance(final_measurements, Mapping)
        else None
    )
    aggregate_delta = (
        float(final_aggregate_score) - float(baseline_aggregate_score)
        if isinstance(baseline_aggregate_score, (int, float))
        and not isinstance(baseline_aggregate_score, bool)
        and isinstance(final_aggregate_score, (int, float))
        and not isinstance(final_aggregate_score, bool)
        else None
    )
    if aggregate_delta is None or aggregate_delta < -maximum_regression:
        reasons.append("AGGREGATE_APPEARANCE_REGRESSION")
    if score_regressions:
        reasons.append("PER_VIEW_APPEARANCE_REGRESSION")

    unique_reasons = sorted(set(reasons))
    document = {
        "schema_version": "asset-pipeline-part-id-final-visual-gate/v1",
        "status": "PASS" if not unique_reasons else "FAIL_CLOSED",
        "completion_allowed": not unique_reasons,
        "acceptance_mode": "PART_ID_VISUAL_NONREGRESSION",
        "reason_codes": unique_reasons,
        "policy": {
            "maximum_score_regression": maximum_regression,
            "minimum_final_appearance_score": (
                config.final_visual_gate_minimum_final_appearance_score
            ),
            "minimum_final_view_appearance_score": (
                config.final_visual_gate_minimum_final_view_appearance_score
            ),
            "minimum_comparable_views": 2,
            "palette_group_completeness_not_applicable": True,
        },
        "measurements": {
            "baseline_part_id_gate": baseline_gate,
            "final_part_id_gate": final_gate,
            "aggregate_appearance_delta": aggregate_delta,
            "per_view_appearance_regressions": score_regressions,
        },
        "inputs": {
            "final_usd": str(final_usd),
            "final_usd_sha256": sha256_file(final_usd),
            "baseline_quality_report": str(baseline_quality_report),
            "baseline_quality_report_sha256": sha256_file(
                baseline_quality_report
            ),
            "final_quality_report": str(final_quality_report),
            "final_quality_report_sha256": sha256_file(final_quality_report),
            "baseline_rendered_registry": str(baseline_rendered_registry),
            "baseline_rendered_registry_sha256": sha256_file(
                baseline_rendered_registry
            ),
            "final_rendered_registry": str(final_rendered_registry),
            "final_rendered_registry_sha256": sha256_file(
                final_rendered_registry
            ),
        },
    }
    write_object(output, document)
    if unique_reasons:
        raise RuntimeError(
            f"{name} rejected independent Part-ID delivery: {unique_reasons}"
        )
    return document


def _run_final_visual_gate_stage(
    *,
    name: str,
    final_usd: Path,
    baseline_quality_report: Path,
    baseline_rendered_registry: Path,
    final_quality_report: Path,
    final_rendered_registry: Path,
    output: Path,
    config: VisualMaterialConfig,
    allow_same_baseline_asset: bool,
    allow_immutable_library_optimum_review: bool,
    sealed_baseline_evidence: Path | None,
    log_cb: LogCallback,
    command_runner: CommandRunner,
) -> dict[str, Any]:
    command = [
        str(config.qwen_python),
        "-m",
        "qwen_material_pipeline",
        "final-visual-gate",
        "--collected-usd",
        str(final_usd),
        "--baseline-quality-report",
        str(baseline_quality_report),
        "--final-quality-report",
        str(final_quality_report),
        "--baseline-rendered-registry",
        str(baseline_rendered_registry),
        "--final-rendered-registry",
        str(final_rendered_registry),
        "--output",
        str(output),
        "--maximum-score-regression",
        str(config.final_visual_gate_maximum_score_regression),
        "--maximum-group-recall-regression",
        str(config.final_visual_gate_maximum_group_recall_regression),
        "--maximum-group-share-error-regression",
        str(
            config.final_visual_gate_maximum_group_share_error_regression
        ),
        "--minimum-final-appearance-score",
        str(config.final_visual_gate_minimum_final_appearance_score),
        "--minimum-final-view-appearance-score",
        str(config.final_visual_gate_minimum_final_view_appearance_score),
        "--minimum-significant-reference-share",
        str(config.final_visual_gate_minimum_significant_reference_share),
        "--minimum-significant-evidence-pixels",
        str(config.final_visual_gate_minimum_significant_evidence_pixels),
    ]
    if allow_same_baseline_asset:
        command.append("--allow-same-baseline-asset")
    if allow_immutable_library_optimum_review:
        command.append("--allow-immutable-library-optimum-review")
    if sealed_baseline_evidence is not None:
        command.extend(
            [
                "--sealed-baseline-preservation-evidence",
                str(sealed_baseline_evidence),
            ]
        )
    stage_error: RuntimeError | None = None
    try:
        _run_stage(
            name,
            command,
            log_cb,
            command_runner=command_runner,
        )
    except RuntimeError as exc:
        # The gate intentionally exits non-zero after atomically writing its
        # fail-closed report. Preserve that audit and surface its decision
        # instead of hiding it behind the subprocess error.
        if not output.is_file():
            raise
        stage_error = exc
    _require_file(output, name)
    report = read_object(output, f"{name} report")
    try:
        require_final_visual_gate_passed(report)
    except FinalVisualGateError as exc:
        raise RuntimeError(
            f"{name} rejected unattended completion: {exc}"
        ) from exc
    if stage_error is not None:
        raise RuntimeError(
            f"{name} returned a non-zero process status despite a PASS report"
        ) from stage_error
    return report


def run_final_visual_acceptance_job(
    *,
    collected_usd: str,
    visual_material_result: Mapping[str, Any],
    output_dir: str | None = None,
    config_path: str | None = None,
    log_cb: LogCallback = None,
    _command_runner: CommandRunner = run_command,
    _isaac_python_resolver: IsaacPythonResolver = isaac_python,
    _config_loader: ConfigLoader = load_visual_material_config,
    _sealed_dependency_verifier: SealedDependencyVerifier = (
        verify_sealed_dependency_lock
    ),
) -> dict[str, Any]:
    """Independently re-render the locked Look and collected delivery.

    Candidate renders are never reused.  The locked Look is first rendered
    into a new directory and checked against the accepted selection baseline.
    The collected root USD is then independently rendered and checked against
    that locked baseline.  Only the second gate may emit ``COMPLETED``.
    """

    if not isinstance(visual_material_result, Mapping):
        raise TypeError("visual_material_result must be an object")
    if visual_material_result.get("state") != "APPLIED":
        raise RuntimeError(
            "Final visual acceptance requires an APPLIED material result"
        )
    collected = _final_visual_file(collected_usd, "Collected root USD")
    locked_usd = _final_visual_file(
        visual_material_result.get("effective_usd"),
        "Final locked Look USD",
    )
    configured_path = config_path or visual_material_result.get("config")
    if configured_path is not None and not isinstance(
        configured_path, (str, Path)
    ):
        raise RuntimeError("Visual material result config path is malformed")
    config = _config_loader(configured_path)
    configured_selection_pipeline_mode = getattr(
        config,
        "material_selection_pipeline_mode",
        "current",
    )
    if (
        not isinstance(configured_selection_pipeline_mode, str)
        or not configured_selection_pipeline_mode
    ):
        raise RuntimeError(
            "Visual material config has an invalid selection pipeline mode"
        )
    result_selection_pipeline_mode = visual_material_result.get(
        "material_selection_pipeline_mode"
    )
    if result_selection_pipeline_mode is None:
        # Results written before selection-pipeline modes existed belong to
        # the historical current lane.  They must never be upgraded into the
        # stricter hybrid lane merely by supplying a different config later.
        result_selection_pipeline_mode = "current"
    if (
        not isinstance(result_selection_pipeline_mode, str)
        or not result_selection_pipeline_mode
        or result_selection_pipeline_mode != configured_selection_pipeline_mode
    ):
        raise RuntimeError(
            "Visual material result and config use different selection "
            "pipeline modes"
        )
    require_all_reference_views_absolute_pass = (
        result_selection_pipeline_mode == "semantic_hybrid"
    )
    isaac = _isaac_python_resolver().expanduser().resolve()
    if not isaac.is_file() or not os.access(isaac, os.X_OK):
        raise FileNotFoundError(f"Isaac Sim Python is unavailable: {isaac}")

    material_output_value = visual_material_result.get("output_dir")
    if not isinstance(material_output_value, str) or not material_output_value:
        raise RuntimeError("Visual material result lacks output_dir")
    material_output = Path(material_output_value).expanduser().resolve()
    allow_part_id_quality = (
        visual_material_result.get("material_assignment_unit") == "part_id"
    )
    allow_scoped_part_id_quality = bool(
        allow_part_id_quality and not require_all_reference_views_absolute_pass
    )
    part_id_quality_scope: Mapping[str, Any] | None = None
    if allow_part_id_quality:
        part_id_gate = _final_visual_file(
            visual_material_result.get("part_id_quality_gate"),
            "Part-ID quality gate",
        )
        if not part_id_gate.is_relative_to(material_output):
            raise RuntimeError("Part-ID quality gate must be inside material output")
        part_id_gate_document = read_object(part_id_gate, "Part-ID quality gate")
        selection_quality_path = _final_visual_file(
            visual_material_result.get("visual_quality_report"),
            "Part-ID selected visual quality report",
        )
        bindings = part_id_gate_document.get("bindings")
        raw_part_id_quality_scope = part_id_gate_document.get("view_scope")
        if (
            part_id_gate_document.get("schema_version")
            != PART_ID_QUALITY_GATE_SCHEMA_VERSION
            or part_id_gate_document.get("status") != "PASS"
            or part_id_gate_document.get("acceptance_allowed") is not True
            or visual_material_result.get("visual_quality_gate_status") != "PASS"
            or not isinstance(bindings, Mapping)
            or bindings.get("quality_report") != str(selection_quality_path)
            or bindings.get("quality_report_sha256")
            != sha256_file(selection_quality_path)
            or (
                raw_part_id_quality_scope is not None
                and not isinstance(raw_part_id_quality_scope, Mapping)
            )
        ):
            raise RuntimeError(
                "Part-ID visual result has an incomplete or stale acceptance contract"
            )
        part_id_quality_scope = (
            copy.deepcopy(dict(raw_part_id_quality_scope))
            if isinstance(raw_part_id_quality_scope, Mapping)
            else None
        )
        selection_quality_document = read_object(
            selection_quality_path,
            "Part-ID selected visual quality report",
        )
        if (
            part_id_quality_scope is not None
            and selection_quality_document.get("part_id_quality_scope")
            != part_id_quality_scope
        ):
            raise RuntimeError(
                "Part-ID selected quality report lost its camera-anchor view scope"
            )
        repeated_part_id_gate = _evaluate_part_id_quality_gate(
            selection_quality_document,
            minimum_aggregate_appearance_score=(
                config.final_visual_gate_minimum_final_appearance_score
            ),
            minimum_view_appearance_score=(
                config.final_visual_gate_minimum_final_view_appearance_score
            ),
            minimum_comparable_views=2,
        )
        if repeated_part_id_gate.get("acceptance_allowed") is not True:
            raise RuntimeError(
                "Part-ID selected quality no longer satisfies the configured floors"
            )
    allow_immutable_library_optimum_review = (
        _validated_immutable_library_optimum_result(
            visual_material_result=visual_material_result,
            material_output=material_output,
        )
    )
    if require_all_reference_views_absolute_pass:
        # A semantic-hybrid delivery may not inherit the immutable-library
        # REVIEW exception even if a stale or forged result advertises it.
        allow_immutable_library_optimum_review = False
    sealed_baseline_evidence = _validated_sealed_historical_baseline_evidence(
        visual_material_result=visual_material_result,
        material_output=material_output,
        isaac=isaac,
        dependency_verifier=_sealed_dependency_verifier,
    )
    preserve_sealed_baseline = sealed_baseline_evidence is not None
    destination = Path(
        output_dir or material_output / "final_visual_acceptance"
    ).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"Final visual acceptance output already exists: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=False)

    baseline_quality_value = visual_material_result.get(
        "visual_quality_report"
    )
    baseline_registry_value = visual_material_result.get(
        "visual_quality_rendered_registry"
    )
    selection_quality: Path | None = None
    selection_registry: Path | None = None
    selection_registry_document: dict[str, Any] | None = None
    acceptance_contract = (
        visual_material_result["material_project_acceptance"]
        if preserve_sealed_baseline
        else None
    )
    if preserve_sealed_baseline and isinstance(baseline_quality_value, str):
        raise RuntimeError(
            "A sealed bundled result cannot override its hash-bound acceptance "
            "contract with a selected quality report"
        )
    if not preserve_sealed_baseline and isinstance(baseline_registry_value, str):
        selection_registry = _final_visual_file(
            baseline_registry_value,
            "Selected visual rendered registry",
        )
        selection_registry_document = read_object(
            selection_registry,
            "selected visual rendered registry",
        )
    if preserve_sealed_baseline:
        reference_manifest = _final_visual_reference_manifest_from_result(
            visual_material_result=visual_material_result,
            output=destination / "reference_manifest.json",
        )
        if acceptance_contract is None:
            raise AssertionError("Validated sealed result lost its acceptance contract")
        mapping = dict(acceptance_contract["view_mapping"])
        minimum_comparable_views = acceptance_contract[
            "minimum_comparable_views"
        ]
        render_contract = dict(acceptance_contract["render"])
    elif isinstance(baseline_quality_value, str):
        if selection_registry is None or selection_registry_document is None:
            raise RuntimeError(
                "Selected quality report has no matching rendered registry"
            )
        selection_quality = _final_visual_file(
            baseline_quality_value,
            "Selected visual quality report",
        )
        selection_quality_document = read_object(
            selection_quality,
            "selected visual quality report",
        )
        mapping, minimum_comparable_views, reference_manifest = (
            _final_visual_mapping(selection_quality_document)
        )
        render_contract = _final_visual_render_contract(
            selection_registry_document
        )
    else:
        reference_manifest = _final_visual_reference_manifest_from_result(
            visual_material_result=visual_material_result,
            output=destination / "reference_manifest.json",
        )
        mapping = None
        minimum_comparable_views = 2
        render_contract = (
            _final_visual_render_contract(selection_registry_document)
            if selection_registry_document is not None
            else _final_visual_config_render_contract(config)
        )

    locked_round = _run_final_visual_render_round(
        name="final_locked_visual",
        asset_usd=locked_usd,
        output_dir=destination / "locked",
        reference_manifest=reference_manifest,
        view_mapping=mapping,
        minimum_comparable_views=minimum_comparable_views,
        render_contract=render_contract,
        isaac=isaac,
        qwen_python=config.qwen_python,
        config=config,
        require_absolute_pass=True,
        require_manifest_exact_view_cover=(
            require_all_reference_views_absolute_pass
        ),
        allow_immutable_library_optimum_review=(
            allow_immutable_library_optimum_review
        ),
        allow_part_id_quality=allow_scoped_part_id_quality,
        log_cb=log_cb,
        command_runner=_command_runner,
        part_id_quality_scope=(
            part_id_quality_scope if allow_scoped_part_id_quality else None
        ),
    )
    locked_gate_path = destination / "locked_visual_gate.json"
    locked_gate: dict[str, Any] | None = None
    if (
        selection_quality is not None
        and selection_registry is not None
        and selection_registry_document is not None
    ):
        selection_asset = selection_registry_document.get("asset_usd")
        same_selection_asset = False
        if isinstance(selection_asset, str):
            try:
                same_selection_asset = (
                    Path(selection_asset).expanduser().resolve(strict=True)
                    == locked_usd
                )
            except (OSError, RuntimeError):
                same_selection_asset = False
        if allow_scoped_part_id_quality:
            locked_gate = _run_part_id_final_visual_gate_stage(
                name="final_locked_visual_gate",
                final_usd=locked_usd,
                baseline_quality_report=selection_quality,
                baseline_rendered_registry=selection_registry,
                final_quality_report=locked_round["quality_report"],
                final_rendered_registry=locked_round["rendered_registry"],
                output=locked_gate_path,
                config=config,
                allow_same_baseline_asset=same_selection_asset,
            )
        else:
            locked_gate = _run_final_visual_gate_stage(
                name="final_locked_visual_gate",
                final_usd=locked_usd,
                baseline_quality_report=selection_quality,
                baseline_rendered_registry=selection_registry,
                final_quality_report=locked_round["quality_report"],
                final_rendered_registry=locked_round["rendered_registry"],
                output=locked_gate_path,
                config=config,
                allow_same_baseline_asset=same_selection_asset,
                allow_immutable_library_optimum_review=(
                    allow_immutable_library_optimum_review
                ),
                sealed_baseline_evidence=sealed_baseline_evidence,
                log_cb=log_cb,
                command_runner=_command_runner,
            )

    locked_quality_document = read_object(
        locked_round["quality_report"],
        "fresh locked visual quality report",
    )
    locked_mapping, locked_minimum_views, locked_reference_manifest = (
        _final_visual_mapping(
            locked_quality_document,
            allow_unscorable_unmapped_views=allow_scoped_part_id_quality,
        )
    )
    locked_registry_document = read_object(
        locked_round["rendered_registry"],
        "fresh locked visual rendered registry",
    )
    locked_render_contract = _final_visual_render_contract(
        locked_registry_document
    )
    if preserve_sealed_baseline:
        if acceptance_contract is None:
            raise AssertionError("Validated sealed result lost its acceptance contract")
        if (
            locked_mapping != dict(acceptance_contract["view_mapping"])
            or locked_minimum_views
            != acceptance_contract["minimum_comparable_views"]
            or locked_reference_manifest != reference_manifest
            or locked_render_contract != dict(acceptance_contract["render"])
        ):
            raise RuntimeError(
                "Fresh locked visual evidence differs from the hash-bound "
                "project acceptance contract"
            )
        collected_mapping = dict(acceptance_contract["view_mapping"])
        collected_minimum_views = acceptance_contract[
            "minimum_comparable_views"
        ]
        collected_reference_manifest = reference_manifest
        collected_render_contract = dict(acceptance_contract["render"])
    else:
        collected_mapping = locked_mapping
        collected_minimum_views = locked_minimum_views
        collected_reference_manifest = locked_reference_manifest
        collected_render_contract = locked_render_contract
    collected_round = _run_final_visual_render_round(
        name="final_collected_visual",
        asset_usd=collected,
        output_dir=destination / "collected",
        reference_manifest=collected_reference_manifest,
        view_mapping=collected_mapping,
        minimum_comparable_views=collected_minimum_views,
        render_contract=collected_render_contract,
        isaac=isaac,
        qwen_python=config.qwen_python,
        config=config,
        require_absolute_pass=True,
        require_manifest_exact_view_cover=(
            require_all_reference_views_absolute_pass
        ),
        allow_immutable_library_optimum_review=(
            allow_immutable_library_optimum_review
        ),
        allow_part_id_quality=allow_scoped_part_id_quality,
        log_cb=log_cb,
        command_runner=_command_runner,
        part_id_quality_scope=(
            part_id_quality_scope if allow_scoped_part_id_quality else None
        ),
    )
    collected_gate_path = destination / "collected_visual_gate.json"
    if allow_scoped_part_id_quality:
        collected_gate = _run_part_id_final_visual_gate_stage(
            name="final_collected_visual_gate",
            final_usd=collected,
            baseline_quality_report=locked_round["quality_report"],
            baseline_rendered_registry=locked_round["rendered_registry"],
            final_quality_report=collected_round["quality_report"],
            final_rendered_registry=collected_round["rendered_registry"],
            output=collected_gate_path,
            config=config,
            allow_same_baseline_asset=False,
        )
    else:
        collected_gate = _run_final_visual_gate_stage(
            name="final_collected_visual_gate",
            final_usd=collected,
            baseline_quality_report=locked_round["quality_report"],
            baseline_rendered_registry=locked_round["rendered_registry"],
            final_quality_report=collected_round["quality_report"],
            final_rendered_registry=collected_round["rendered_registry"],
            output=collected_gate_path,
            config=config,
            allow_same_baseline_asset=False,
            allow_immutable_library_optimum_review=(
                allow_immutable_library_optimum_review
            ),
            sealed_baseline_evidence=sealed_baseline_evidence,
            log_cb=log_cb,
            command_runner=_command_runner,
        )
    if collected_gate.get("completion_allowed") is not True:
        raise RuntimeError(
            "Collected visual gate did not authorize pipeline completion"
        )
    return {
        "schema_version": "asset-pipeline-final-visual-acceptance/v1",
        "state": "COMPLETED",
        "completion_allowed": True,
        "acceptance_mode": (
            "SEALED_BASELINE_PRESERVATION"
            if preserve_sealed_baseline
            else (
                "IMMUTABLE_LIBRARY_OPTIMUM"
                if allow_immutable_library_optimum_review
                else (
                    "PART_ID_VISUAL_NONREGRESSION"
                    if allow_scoped_part_id_quality
                    else "ABSOLUTE_PASS"
                )
            )
        ),
        "material_selection_pipeline_mode": result_selection_pipeline_mode,
        "all_reference_views_absolute_pass_required": (
            require_all_reference_views_absolute_pass
        ),
        "sealed_baseline_evidence": (
            str(sealed_baseline_evidence)
            if sealed_baseline_evidence is not None
            else None
        ),
        "collected_usd": str(collected),
        "locked_usd": str(locked_usd),
        "output_dir": str(destination),
        "selection_quality_report": (
            str(selection_quality) if selection_quality is not None else None
        ),
        "selection_rendered_registry": (
            str(selection_registry) if selection_registry is not None else None
        ),
        "locked_quality_report": str(locked_round["quality_report"]),
        "locked_rendered_registry": str(
            locked_round["rendered_registry"]
        ),
        "locked_visual_gate": (
            str(locked_gate_path.resolve(strict=True))
            if locked_gate is not None
            else None
        ),
        "locked_visual_gate_status": (
            locked_gate["status"]
            if locked_gate is not None
            else "ESTABLISHED_INDEPENDENT_BASELINE"
        ),
        "collected_quality_report": str(collected_round["quality_report"]),
        "collected_rendered_registry": str(
            collected_round["rendered_registry"]
        ),
        "collected_visual_gate": str(
            collected_gate_path.resolve(strict=True)
        ),
        "collected_visual_gate_status": collected_gate["status"],
    }
