"""Fail-closed visual acceptance for the independently rendered delivery USD.

Candidate renders are useful for material selection, but they cannot prove that
the locked Look survived Physics authoring and USD collection.  This module
therefore accepts only a quality report produced from a fresh registry whose
asset is the exact collected USD.  It then compares that report with the
accepted pre-delivery baseline at aggregate, per-view, and significant
reference-group levels.

The gate is deliberately renderer-agnostic.  The normal orchestration is:

1. build a registry from the collected root USD;
2. render that registry into a new directory;
3. compare the new renders with the original reference manifest; and
4. run this gate.

No candidate render or candidate quality report can be substituted for step 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .sealed_project import (
    SealedProjectBindingError,
    validate_sealed_project_binding,
)


SCHEMA_VERSION = "qwen-final-visual-gate/v1"
QUALITY_SCHEMA_VERSION = "qwen-reference-render-comparison/v1"
REGISTRY_SCHEMA_VERSION = "qwen-material-parts/v1"

PASS = "PASS"
FAIL_CLOSED = "FAIL_CLOSED"
COMPLETED = "COMPLETED"
FAILED_COMPLETION_STATE = "FINAL_VISUAL_QA_FAILED"
ABSOLUTE_PASS_MODE = "ABSOLUTE_PASS"
IMMUTABLE_LIBRARY_OPTIMUM_MODE = "IMMUTABLE_LIBRARY_OPTIMUM"
IMMUTABLE_LIBRARY_RENDER_REPEATABILITY_TOLERANCE = 0.03
SEALED_BASELINE_PRESERVATION_MODE = "SEALED_BASELINE_PRESERVATION"
SEALED_EVIDENCE_SCHEMA_VERSION = "qwen-bundled-project-evidence/v1"

DEFAULT_MAXIMUM_SCORE_REGRESSION = 0.01
DEFAULT_MAXIMUM_GROUP_RECALL_REGRESSION = 0.01
DEFAULT_MAXIMUM_GROUP_SHARE_ERROR_REGRESSION = 0.01
DEFAULT_MINIMUM_FINAL_APPEARANCE_SCORE = 0.62
DEFAULT_MINIMUM_FINAL_VIEW_APPEARANCE_SCORE = 0.55
DEFAULT_MINIMUM_SIGNIFICANT_REFERENCE_SHARE = 0.01
DEFAULT_MINIMUM_SIGNIFICANT_EVIDENCE_PIXELS = 128
_PIXEL_COUNT_ABSOLUTE_TOLERANCE = 1e-6
_FLOAT_COMPARISON_ABSOLUTE_TOLERANCE = 1e-12
_IMMUTABLE_LIBRARY_ALLOWED_REVIEW_REASONS = frozenset(
    {"foreground_value_similarity_below_pass_threshold"}
)

_RENDER_CONTRACT_FIELDS = (
    "resolution",
    "analysis_up_axis",
    "analysis_front_axis",
    "lighting_profile",
    "requested_view_tokens",
    "rt_subframes",
)

_AXIS_VECTORS = {
    "x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}


class FinalVisualGateError(ValueError):
    """Raised when a final visual gate document or binding is malformed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _input_file(value: str | Path, label: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FinalVisualGateError(f"{label} does not exist: {value}") from exc
    if not path.is_file():
        raise FinalVisualGateError(f"{label} must be a file: {path}")
    return path


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalVisualGateError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalVisualGateError(f"{label} must contain a JSON object")
    return value


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalVisualGateError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise FinalVisualGateError(f"{label} must be an array")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise FinalVisualGateError(f"{label} must be a non-empty string")
    return value


def _unit(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalVisualGateError(f"{label} must be numeric")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise FinalVisualGateError(f"{label} must be between zero and one")
    return number


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FinalVisualGateError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return value


def _sha256_text(value: Any, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise FinalVisualGateError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _sealed_baseline_evidence_provenance(path: Path) -> dict[str, Any]:
    """Validate the sealed replay evidence that authorizes preservation mode."""

    evidence = _read_object(path, "sealed baseline evidence")
    _schema(
        evidence,
        SEALED_EVIDENCE_SCHEMA_VERSION,
        "sealed baseline evidence",
    )
    asset_id = _text(evidence.get("asset_id"), "sealed evidence asset_id")
    try:
        project_binding = validate_sealed_project_binding(evidence)
    except SealedProjectBindingError as exc:
        raise FinalVisualGateError(
            f"sealed baseline evidence has invalid project binding: {exc}"
        ) from exc
    method = project_binding["method"]
    if evidence.get("live_inference_repeated") is not False:
        raise FinalVisualGateError(
            "sealed baseline preservation requires live_inference_repeated=false"
        )
    digests = {
        field: _sha256_text(
            evidence.get(field),
            f"sealed evidence {field}",
        )
        for field in (
            "historical_result_sha256",
            "source_cad_sha256",
            "template_sha256",
            "catalog_sha256",
            "plan_sha256",
            "audit_sha256",
        )
    }
    reference_sha256 = _object(
        evidence.get("reference_sha256"),
        "sealed evidence reference_sha256",
    )
    if len(reference_sha256) < 2:
        raise FinalVisualGateError(
            "sealed baseline evidence requires at least two reference hashes"
        )
    normalized_references: dict[str, str] = {}
    for reference_id, digest in reference_sha256.items():
        normalized_id = _text(reference_id, "sealed evidence reference ID")
        normalized_references[normalized_id] = _sha256_text(
            digest,
            f"sealed evidence reference {normalized_id}",
        )
    catalog_path: Path | None = None
    for artifact in ("template", "catalog"):
        artifact_path = _recorded_file(
            evidence.get(artifact),
            f"sealed evidence {artifact}",
        )
        if _sha256_file(artifact_path) != digests[f"{artifact}_sha256"]:
            raise FinalVisualGateError(
                f"sealed evidence {artifact} hash does not match the file"
            )
        if artifact == "catalog":
            catalog_path = artifact_path
    if catalog_path is None:
        raise AssertionError("sealed catalog validation did not run")
    if (
        evidence.get("dependency_lock_verified") is not True
        or evidence.get("dependency_lock_verification_status") != PASS
    ):
        raise FinalVisualGateError(
            "sealed baseline evidence lacks PASS dependency verification"
        )
    dependency_lock = _recorded_file(
        evidence.get("dependency_lock"),
        "sealed evidence dependency lock",
    )
    dependency_lock_sha256 = _sha256_text(
        evidence.get("dependency_lock_sha256"),
        "sealed evidence dependency_lock_sha256",
    )
    if _sha256_file(dependency_lock) != dependency_lock_sha256:
        raise FinalVisualGateError(
            "sealed evidence dependency lock hash does not match the file"
        )
    verification_report = _recorded_file(
        evidence.get("dependency_lock_verification_report"),
        "sealed evidence dependency verification report",
    )
    verification_report_sha256 = _sha256_text(
        evidence.get("dependency_lock_verification_report_sha256"),
        "sealed evidence dependency verification report SHA-256",
    )
    if _sha256_file(verification_report) != verification_report_sha256:
        raise FinalVisualGateError(
            "sealed dependency verification report hash does not match the file"
        )
    verification = _read_object(
        verification_report,
        "sealed dependency verification report",
    )
    if (
        verification.get("schema_version")
        != "qwen-sealed-material-dependency-verification/v1"
        or verification.get("status") != PASS
        or verification.get("dependency_lock_verified") is not True
        or verification.get("lock_path") != str(dependency_lock)
        or verification.get("lock_sha256") != dependency_lock_sha256
        or verification.get("catalog_path") != str(catalog_path)
        or evidence.get("dependency_lock_verification") != verification
    ):
        raise FinalVisualGateError(
            "sealed dependency verification report is invalid or inconsistent"
        )
    _text(evidence.get("replay_policy"), "sealed evidence replay_policy")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "asset_id": asset_id,
        "method": method,
        "project": project_binding["project"],
        "project_sha256": project_binding["project_sha256"],
        "historical_result_sha256": digests["historical_result_sha256"],
        "reference_sha256": dict(sorted(normalized_references.items())),
        "reference_roles": project_binding["reference_roles"],
        "acceptance": project_binding["acceptance"],
        "acceptance_sha256": project_binding["acceptance_sha256"],
        "acceptance_evidence": project_binding["acceptance_evidence"],
        "acceptance_evidence_sha256": project_binding[
            "acceptance_evidence_sha256"
        ],
        "dependency_lock": str(dependency_lock),
        "dependency_lock_sha256": dependency_lock_sha256,
        "dependency_lock_verification_report": str(verification_report),
        "dependency_lock_verification_report_sha256": (verification_report_sha256),
    }


def _quality_status_rank(value: Any, label: str, *, aggregate: bool) -> int:
    status = _text(value, label)
    ranks = (
        {
            "PASS": 3,
            "REVIEW": 2,
            "FAIL": 1,
            "INSUFFICIENT_EVIDENCE": 0,
        }
        if aggregate
        else {
            "PASS": 3,
            "REVIEW": 2,
            "FAIL": 1,
            "UNSCORABLE": 0,
        }
    )
    try:
        return ranks[status]
    except KeyError as exc:
        raise FinalVisualGateError(
            f"{label} has unsupported quality status {status!r}"
        ) from exc


def _schema(document: Mapping[str, Any], expected: str, label: str) -> None:
    if document.get("schema_version") != expected:
        raise FinalVisualGateError(f"{label} schema_version must be {expected!r}")


def _recorded_file(value: Any, label: str) -> Path:
    return _input_file(_text(value, label), label)


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _report_views(
    report: Mapping[str, Any],
    label: str,
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(_array(report.get("views"), f"{label}.views")):
        view = _object(raw, f"{label}.views[{index}]")
        view_id = _text(
            view.get("reference_view_id"),
            f"{label}.views[{index}].reference_view_id",
        )
        if view_id in output:
            raise FinalVisualGateError(f"{label} repeats reference view {view_id}")
        output[view_id] = view
    if not output:
        raise FinalVisualGateError(f"{label} must contain at least one view")
    return output


def _registry_views(
    registry: Mapping[str, Any],
    label: str,
) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    render_set = _object(registry.get("render_set"), f"{label}.render_set")
    output: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(
        _array(render_set.get("views"), f"{label}.render_set.views")
    ):
        view = _object(raw, f"{label}.render_set.views[{index}]")
        view_id = _text(
            view.get("view_id"),
            f"{label}.render_set.views[{index}].view_id",
        )
        if view_id in output:
            raise FinalVisualGateError(f"{label} repeats render view {view_id}")
        output[view_id] = view
    if not output:
        raise FinalVisualGateError(f"{label} must contain rendered views")
    return render_set, output


def _reference_manifest_roles(path: Path) -> tuple[str, ...]:
    """Return the ordered reference IDs from either supported manifest shape."""

    document = _read_object(path, "reference manifest")
    source_views = document.get("source_views")
    if source_views is not None:
        roles: list[str] = []
        for index, raw_view in enumerate(
            _array(source_views, "reference manifest.source_views")
        ):
            view = _object(raw_view, f"reference manifest.source_views[{index}]")
            roles.append(
                _text(
                    view.get("id"),
                    f"reference manifest.source_views[{index}].id",
                )
            )
    else:
        roles = [
            _text(role, f"reference manifest.views[{index}]")
            for index, role in enumerate(
                _array(document.get("views"), "reference manifest.views")
            )
        ]
    if not roles or len(set(roles)) != len(roles):
        raise FinalVisualGateError(
            "reference manifest roles must be non-empty and unique"
        )
    return tuple(roles)


def _validate_axis_vector(value: Any, expected: tuple[float, ...], label: str) -> None:
    vector = _array(value, label)
    if len(vector) != len(expected):
        raise FinalVisualGateError(f"{label} has the wrong dimension")
    normalized: list[float] = []
    for index, component in enumerate(vector):
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise FinalVisualGateError(f"{label}[{index}] must be numeric")
        normalized.append(float(component))
    if tuple(normalized) != expected:
        raise FinalVisualGateError(f"{label} violates the sealed project contract")


def _validate_sealed_reference_evidence(
    *,
    provenance: Mapping[str, Any],
    reference_manifest: Path,
) -> dict[str, Any] | None:
    raw_expected = provenance.get("acceptance_evidence")
    if raw_expected is None:
        return None
    expected = _object(raw_expected, "sealed project acceptance evidence")
    expected_views = _array(
        expected.get("source_views"),
        "sealed project acceptance evidence source_views",
    )
    manifest = _read_object(reference_manifest, "reference manifest")
    manifest_views = _array(
        manifest.get("source_views"),
        "reference manifest.source_views",
    )
    if len(manifest_views) != len(expected_views):
        raise FinalVisualGateError(
            "reference manifest evidence does not cover the sealed project"
        )
    verified: list[dict[str, Any]] = []
    for index, (raw_manifest_view, raw_expected_view) in enumerate(
        zip(manifest_views, expected_views, strict=True)
    ):
        manifest_view = _object(
            raw_manifest_view,
            f"reference manifest.source_views[{index}]",
        )
        expected_view = _object(
            raw_expected_view,
            f"sealed acceptance evidence.source_views[{index}]",
        )
        required_fields = {
            "id",
            "image",
            "palette_status",
            "palette_mask",
            "palette_path",
            "palette_artifacts",
        }
        if set(manifest_view) != required_fields:
            raise FinalVisualGateError(
                f"reference manifest.source_views[{index}] fields violate the "
                "sealed acceptance evidence contract"
            )
        role = _text(manifest_view.get("id"), f"reference view[{index}] id")
        if role != expected_view.get("id"):
            raise FinalVisualGateError(
                "reference manifest evidence order violates the sealed project"
            )
        image = _recorded_file(
            manifest_view.get("image"),
            f"reference image {role}",
        )
        if _sha256_file(image) != _sha256_text(
            expected_view.get("reference_sha256"),
            f"sealed acceptance evidence {role} reference_sha256",
        ):
            raise FinalVisualGateError(
                f"reference image {role} hash violates sealed acceptance evidence"
            )
        if manifest_view.get("palette_status") != "usable":
            raise FinalVisualGateError(
                f"reference view {role} lacks usable sealed palette evidence"
            )
        mask = _recorded_file(
            manifest_view.get("palette_mask"),
            f"reference view {role} palette_mask",
        )
        palette = _recorded_file(
            manifest_view.get("palette_path"),
            f"reference view {role} palette_path",
        )
        artifacts = _object(
            manifest_view.get("palette_artifacts"),
            f"reference view {role} palette_artifacts",
        )
        if set(artifacts) != {"normalized", "normalized_evidence_audit"}:
            raise FinalVisualGateError(
                f"reference view {role} palette_artifacts violate the sealed project"
            )
        normalized = _recorded_file(
            artifacts.get("normalized"),
            f"reference view {role} normalized palette",
        )
        audit = _recorded_file(
            artifacts.get("normalized_evidence_audit"),
            f"reference view {role} normalized evidence audit",
        )
        expected_artifacts = _object(
            expected_view.get("palette_artifacts"),
            f"sealed acceptance evidence {role} palette_artifacts",
        )
        checks = (
            (
                mask,
                expected_view.get("palette_mask"),
                expected_view.get("palette_mask_sha256"),
                "palette_mask",
            ),
            (
                palette,
                expected_view.get("palette_path"),
                expected_view.get("palette_path_sha256"),
                "palette_path",
            ),
            (
                normalized,
                expected_artifacts.get("normalized"),
                expected_artifacts.get("normalized_sha256"),
                "normalized palette",
            ),
            (
                audit,
                expected_artifacts.get("normalized_evidence_audit"),
                expected_artifacts.get("normalized_evidence_audit_sha256"),
                "normalized evidence audit",
            ),
        )
        for actual_path, expected_path, expected_hash, label in checks:
            if (
                str(actual_path) != expected_path
                or _sha256_file(actual_path)
                != _sha256_text(
                    expected_hash,
                    f"sealed acceptance evidence {role} {label} sha256",
                )
            ):
                raise FinalVisualGateError(
                    f"reference view {role} {label} violates sealed evidence"
                )
        if palette != normalized:
            raise FinalVisualGateError(
                f"reference view {role} palette differs from normalized evidence"
            )
        verified.append(
            {
                "id": role,
                "reference_sha256": _sha256_file(image),
                "palette_mask_sha256": _sha256_file(mask),
                "palette_sha256": _sha256_file(palette),
                "normalized_evidence_audit_sha256": _sha256_file(audit),
            }
        )
    return {
        "manifest_sha256": _sha256_file(reference_manifest),
        "acceptance_evidence_sha256": _text(
            provenance.get("acceptance_evidence_sha256"),
            "sealed project acceptance_evidence_sha256",
        ),
        "views": verified,
    }


def _validate_sealed_acceptance_contract(
    *,
    provenance: Mapping[str, Any],
    reference_manifest: Path,
    baseline_inputs: Mapping[str, Any],
    final_inputs: Mapping[str, Any],
    baseline_quality: Mapping[str, Any],
    final_quality: Mapping[str, Any],
    baseline_render_set: Mapping[str, Any],
    final_render_set: Mapping[str, Any],
    baseline_registry_views: Mapping[str, Mapping[str, Any]],
    final_registry_views: Mapping[str, Mapping[str, Any]],
    baseline_report_views: Mapping[str, Mapping[str, Any]],
    final_report_views: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind every final-gate input to the hash-owned acceptance contract."""

    acceptance = _object(
        provenance.get("acceptance"),
        "sealed project acceptance",
    )
    render = _object(
        acceptance.get("render"),
        "sealed project acceptance.render",
    )
    resolution = _integer(
        render.get("resolution"),
        "sealed project acceptance resolution",
        minimum=1,
    )
    rt_subframes = _integer(
        render.get("rt_subframes"),
        "sealed project acceptance rt_subframes",
        minimum=1,
    )
    view_tokens = tuple(
        _text(render.get("views"), "sealed project acceptance views").split(",")
    )
    mapping = dict(
        _object(
            acceptance.get("view_mapping"),
            "sealed project acceptance.view_mapping",
        )
    )
    minimum_comparable_views = _integer(
        acceptance.get("minimum_comparable_views"),
        "sealed project acceptance minimum_comparable_views",
        minimum=1,
    )
    reference_roles = tuple(
        _text(role, "sealed project reference role")
        for role in _array(
            provenance.get("reference_roles"),
            "sealed project reference_roles",
        )
    )
    if _reference_manifest_roles(reference_manifest) != reference_roles:
        raise FinalVisualGateError(
            "reference manifest roles/order violate the sealed project contract"
        )
    reference_evidence_audit = _validate_sealed_reference_evidence(
        provenance=provenance,
        reference_manifest=reference_manifest,
    )

    expected_render_values = {
        "resolution": [resolution, resolution],
        "lighting_profile": _text(
            render.get("lighting_profile"),
            "sealed project acceptance lighting_profile",
        ),
        "requested_view_tokens": list(view_tokens),
        "rt_subframes": rt_subframes,
    }
    up_axis = _text(
        render.get("analysis_up_axis"),
        "sealed project acceptance analysis_up_axis",
    )
    front_axis = _text(
        render.get("analysis_front_axis"),
        "sealed project acceptance analysis_front_axis",
    )
    try:
        expected_up_axis = _AXIS_VECTORS[up_axis]
        expected_front_axis = _AXIS_VECTORS[front_axis]
    except KeyError as exc:
        raise FinalVisualGateError(
            "sealed project acceptance contains an unsupported analysis axis"
        ) from exc

    for label, render_set, registry_views in (
        ("baseline", baseline_render_set, baseline_registry_views),
        ("final", final_render_set, final_registry_views),
    ):
        for field, expected in expected_render_values.items():
            actual = render_set.get(field)
            if field in {"resolution", "requested_view_tokens"}:
                actual = _array(actual, f"{label} render_set.{field}")
            elif field == "rt_subframes":
                actual = _integer(
                    actual,
                    f"{label} render_set.rt_subframes",
                    minimum=1,
                )
            if actual != expected:
                raise FinalVisualGateError(
                    f"{label} render_set.{field} violates the sealed project "
                    "acceptance contract"
                )
        _validate_axis_vector(
            render_set.get("analysis_up_axis"),
            expected_up_axis,
            f"{label} render_set.analysis_up_axis",
        )
        _validate_axis_vector(
            render_set.get("analysis_front_axis"),
            expected_front_axis,
            f"{label} render_set.analysis_front_axis",
        )
        if tuple(registry_views) != view_tokens:
            raise FinalVisualGateError(
                f"{label} registry views do not exactly cover the sealed project "
                "render views in contract order"
            )

    for label, inputs, quality, report_views in (
        ("baseline", baseline_inputs, baseline_quality, baseline_report_views),
        ("final", final_inputs, final_quality, final_report_views),
    ):
        selected_mapping = dict(
            _object(
                inputs.get("selected_view_mapping"),
                f"{label} inputs.selected_view_mapping",
            )
        )
        if selected_mapping != mapping:
            raise FinalVisualGateError(
                f"{label} view mapping violates the sealed project contract"
            )
        thresholds = _object(quality.get("thresholds"), f"{label}.thresholds")
        if (
            _integer(
                thresholds.get("minimum_comparable_views"),
                f"{label} thresholds.minimum_comparable_views",
                minimum=1,
            )
            != minimum_comparable_views
        ):
            raise FinalVisualGateError(
                f"{label} minimum comparable views violates the sealed project "
                "contract"
            )
        if set(report_views) != set(reference_roles):
            raise FinalVisualGateError(
                f"{label} quality report does not exactly cover sealed reference "
                "roles"
            )
        for reference_role in reference_roles:
            if report_views[reference_role].get("render_view_id") != mapping.get(
                reference_role
            ):
                raise FinalVisualGateError(
                    f"{label} quality report maps {reference_role!r} outside the "
                    "sealed project contract"
                )
        aggregate = _object(quality.get("aggregate"), f"{label}.aggregate")
        if (
            _integer(
                aggregate.get("comparable_view_count"),
                f"{label} aggregate.comparable_view_count",
            )
            < minimum_comparable_views
        ):
            raise FinalVisualGateError(
                f"{label} quality report has fewer comparable views than the "
                "sealed project contract"
            )

    return {
        "acceptance": dict(acceptance),
        "acceptance_sha256": _text(
            provenance.get("acceptance_sha256"),
            "sealed project acceptance_sha256",
        ),
        "reference_roles": list(reference_roles),
        "reference_evidence": reference_evidence_audit,
    }


def _quality_score_record(
    view: Mapping[str, Any],
    label: str,
) -> dict[str, float]:
    material_color = _object(view.get("material_color"), f"{label}.material_color")
    material_texture = _object(
        view.get("material_texture"),
        f"{label}.material_texture",
    )
    alignment = _object(view.get("alignment"), f"{label}.alignment")
    return {
        "color": _unit(material_color.get("score"), f"{label} color score"),
        "texture": _unit(
            material_texture.get("score"),
            f"{label} texture score",
        ),
        "appearance": _unit(
            view.get("material_appearance_score"),
            f"{label} appearance score",
        ),
        "alignment": _unit(
            alignment.get("score"),
            f"{label} alignment score",
        ),
        "silhouette_iou": _unit(
            alignment.get("silhouette_iou"),
            f"{label} silhouette IoU",
        ),
    }


def _aggregate_scores(
    report: Mapping[str, Any],
    label: str,
) -> tuple[Mapping[str, Any], dict[str, float]]:
    aggregate = _object(report.get("aggregate"), f"{label}.aggregate")
    return aggregate, {
        "color": _unit(
            aggregate.get("material_color_score"),
            f"{label} aggregate color score",
        ),
        "texture": _unit(
            aggregate.get("material_texture_score"),
            f"{label} aggregate texture score",
        ),
        "appearance": _unit(
            aggregate.get("material_appearance_score"),
            f"{label} aggregate appearance score",
        ),
    }


def _significant_groups(
    view: Mapping[str, Any],
    label: str,
    *,
    minimum_reference_share: float,
    minimum_evidence_pixels: int,
) -> dict[str, Mapping[str, Any]]:
    material_color = _object(view.get("material_color"), f"{label}.material_color")
    raw_recall = material_color.get("trusted_evidence_group_recall")
    if raw_recall is None:
        return {}
    recall = _object(raw_recall, f"{label}.trusted_evidence_group_recall")
    groups: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(
        _array(recall.get("groups"), f"{label}.trusted groups")
    ):
        group = _object(raw, f"{label}.trusted groups[{index}]")
        group_id = _text(
            group.get("group_id"),
            f"{label}.trusted groups[{index}].group_id",
        )
        if group_id in groups:
            raise FinalVisualGateError(f"{label} repeats trusted group {group_id}")
        reference_share = _unit(
            group.get("reference_group_share"),
            f"{label} group {group_id} reference share",
        )
        evidence_pixels = _integer(
            group.get("reference_evidence_weight"),
            f"{label} group {group_id} evidence pixels",
        )
        if (
            reference_share >= minimum_reference_share
            or evidence_pixels >= minimum_evidence_pixels
        ):
            groups[group_id] = group
    return groups


def _all_groups(
    view: Mapping[str, Any],
    label: str,
) -> dict[str, Mapping[str, Any]]:
    material_color = _object(view.get("material_color"), f"{label}.material_color")
    raw_recall = material_color.get("trusted_evidence_group_recall")
    if raw_recall is None:
        return {}
    recall = _object(raw_recall, f"{label}.trusted_evidence_group_recall")
    output: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(
        _array(recall.get("groups"), f"{label}.trusted groups")
    ):
        group = _object(raw, f"{label}.trusted groups[{index}]")
        group_id = _text(
            group.get("group_id"),
            f"{label}.trusted groups[{index}].group_id",
        )
        if group_id in output:
            raise FinalVisualGateError(f"{label} repeats trusted group {group_id}")
        output[group_id] = group
    return output


def _render_distribution_sampled_pixels(
    view: Mapping[str, Any],
    label: str,
) -> int | None:
    """Return the exact color-distribution sample count when it is recorded.

    Older comparison fixtures did not record the render distribution.  They
    remain valid gate inputs, but cannot claim pixel-quantization tolerance and
    therefore keep the strict floating-point recall check.
    """

    material_color = _object(view.get("material_color"), f"{label}.material_color")
    raw_distribution = material_color.get("render_distribution")
    if raw_distribution is None:
        return None
    distribution = _object(
        raw_distribution,
        f"{label}.material_color.render_distribution",
    )
    return _integer(
        distribution.get("sampled_pixels"),
        f"{label} render distribution sampled_pixels",
        minimum=1,
    )


def _group_recall_quantization_audit(
    *,
    baseline_view: Mapping[str, Any],
    final_view: Mapping[str, Any],
    baseline_group: Mapping[str, Any],
    final_group: Mapping[str, Any],
    baseline_recall: float,
    final_recall: float,
    maximum_group_recall_regression: float,
    minimum_significant_evidence_pixels: int,
    whole_report_quality_nonregressing: bool,
    group_identity_unchanged: bool,
    group_share_nonregressing: bool,
    group_presence_preserved: bool,
) -> dict[str, Any]:
    """Audit a bounded integer-pixel companion gate for group recall.

    Group recall is computed from a histogram whose shares are integer pixel
    counts divided by ``sampled_pixels``.  A continuous recall threshold can
    therefore fall between two representable pixel counts.  The continuous
    relative gate remains mandatory; only its sub-pixel boundary is rounded up
    to the next representable integer count, and only when all binding and
    whole-report non-regression conditions are independently satisfied.
    """

    audit: dict[str, Any] = {
        "method": "relative_recall_and_integer_pixel_loss/v1",
        "applicable": False,
        "tolerance_applied": False,
        "reason": "PIXEL_QUANTIZATION_EVIDENCE_UNAVAILABLE",
        "raw_recall_regression": (
            baseline_recall - final_recall > maximum_group_recall_regression
        ),
        "recall_loss": max(0.0, baseline_recall - final_recall),
        "configured_maximum_recall_regression": (maximum_group_recall_regression),
    }
    if not audit["raw_recall_regression"]:
        audit["reason"] = "RAW_RECALL_WITHIN_CONFIGURED_LIMIT"
        return audit
    if not whole_report_quality_nonregressing:
        audit["reason"] = "WHOLE_REPORT_QUALITY_NOT_NONREGRESSING"
        return audit
    if not group_identity_unchanged:
        audit["reason"] = "REFERENCE_GROUP_IDENTITY_CHANGED"
        return audit
    if not group_share_nonregressing:
        audit["reason"] = "GROUP_SHARE_ERROR_REGRESSED"
        return audit
    if not group_presence_preserved:
        audit["reason"] = "GROUP_PRESENCE_NOT_PRESERVED"
        return audit

    baseline_evidence_pixels = _integer(
        baseline_group.get("reference_evidence_weight"),
        "baseline group reference_evidence_weight",
    )
    final_evidence_pixels = _integer(
        final_group.get("reference_evidence_weight"),
        "final group reference_evidence_weight",
    )
    audit["reference_evidence_pixels"] = baseline_evidence_pixels
    if (
        baseline_evidence_pixels != final_evidence_pixels
        or baseline_evidence_pixels < minimum_significant_evidence_pixels
    ):
        audit["reason"] = "INSUFFICIENT_OR_CHANGED_REFERENCE_EVIDENCE_PIXELS"
        return audit

    baseline_sampled_pixels = _render_distribution_sampled_pixels(
        baseline_view,
        "baseline view",
    )
    final_sampled_pixels = _render_distribution_sampled_pixels(
        final_view,
        "final view",
    )
    audit["baseline_sampled_pixels"] = baseline_sampled_pixels
    audit["final_sampled_pixels"] = final_sampled_pixels
    if (
        baseline_sampled_pixels is None
        or final_sampled_pixels is None
        or baseline_sampled_pixels != final_sampled_pixels
    ):
        audit["reason"] = "RENDER_SAMPLE_COUNT_UNAVAILABLE_OR_CHANGED"
        return audit

    raw_baseline_required_share = baseline_group.get("required_render_share")
    raw_final_required_share = final_group.get("required_render_share")
    if raw_baseline_required_share is None or raw_final_required_share is None:
        audit["reason"] = "REQUIRED_RENDER_SHARE_UNAVAILABLE"
        return audit
    baseline_required_share = _unit(
        raw_baseline_required_share,
        "baseline group required_render_share",
    )
    final_required_share = _unit(
        raw_final_required_share,
        "final group required_render_share",
    )
    audit["baseline_required_render_share"] = baseline_required_share
    audit["final_required_render_share"] = final_required_share
    if baseline_required_share <= 0.0 or final_required_share <= 0.0:
        audit["reason"] = "REQUIRED_RENDER_SHARE_IS_NOT_POSITIVE"
        return audit
    if not math.isclose(
        baseline_required_share,
        final_required_share,
        rel_tol=0.0,
        abs_tol=_FLOAT_COMPARISON_ABSOLUTE_TOLERANCE,
    ):
        audit["reason"] = "REQUIRED_RENDER_SHARE_CHANGED"
        return audit

    baseline_observed_share = _unit(
        baseline_group.get("observed_render_share"),
        "baseline group observed_render_share",
    )
    final_observed_share = _unit(
        final_group.get("observed_render_share"),
        "final group observed_render_share",
    )
    expected_baseline_recall = min(
        1.0,
        baseline_observed_share / baseline_required_share,
    )
    expected_final_recall = min(
        1.0,
        final_observed_share / final_required_share,
    )
    audit["expected_baseline_recall"] = expected_baseline_recall
    audit["expected_final_recall"] = expected_final_recall
    if not (
        math.isclose(
            baseline_recall,
            expected_baseline_recall,
            rel_tol=0.0,
            abs_tol=_FLOAT_COMPARISON_ABSOLUTE_TOLERANCE,
        )
        and math.isclose(
            final_recall,
            expected_final_recall,
            rel_tol=0.0,
            abs_tol=_FLOAT_COMPARISON_ABSOLUTE_TOLERANCE,
        )
    ):
        audit["reason"] = "REPORTED_RECALL_INCONSISTENT_WITH_PIXEL_SHARES"
        return audit
    sampled_pixels = baseline_sampled_pixels
    baseline_observed_float = baseline_observed_share * sampled_pixels
    final_observed_float = final_observed_share * sampled_pixels
    baseline_observed_pixels = round(baseline_observed_float)
    final_observed_pixels = round(final_observed_float)
    audit["baseline_observed_pixels"] = baseline_observed_pixels
    audit["final_observed_pixels"] = final_observed_pixels
    if not (
        math.isclose(
            baseline_observed_float,
            baseline_observed_pixels,
            rel_tol=0.0,
            abs_tol=_PIXEL_COUNT_ABSOLUTE_TOLERANCE,
        )
        and math.isclose(
            final_observed_float,
            final_observed_pixels,
            rel_tol=0.0,
            abs_tol=_PIXEL_COUNT_ABSOLUTE_TOLERANCE,
        )
    ):
        audit["reason"] = "OBSERVED_RENDER_SHARE_IS_NOT_INTEGER_PIXEL_QUANTIZED"
        return audit

    required_render_pixels = sampled_pixels * baseline_required_share
    allowed_pixel_loss = math.ceil(
        maximum_group_recall_regression * required_render_pixels
        - _FLOAT_COMPARISON_ABSOLUTE_TOLERANCE
    )
    observed_pixel_loss = max(
        0,
        baseline_observed_pixels - final_observed_pixels,
    )
    audit.update(
        {
            "applicable": True,
            "required_render_pixels": required_render_pixels,
            "observed_pixel_loss": observed_pixel_loss,
            "maximum_allowed_pixel_loss": allowed_pixel_loss,
            "effective_quantized_recall_tolerance": (
                allowed_pixel_loss / required_render_pixels
                if required_render_pixels > 0.0
                else 0.0
            ),
        }
    )
    if observed_pixel_loss <= allowed_pixel_loss:
        audit["tolerance_applied"] = True
        audit["reason"] = "WITHIN_INTEGER_PIXEL_QUANTIZATION_BOUNDARY"
    else:
        audit["reason"] = "INTEGER_PIXEL_LOSS_EXCEEDS_BOUNDARY"
    return audit


def _validate_report_binding(
    *,
    report: Mapping[str, Any],
    report_label: str,
    registry_path: Path,
) -> Mapping[str, Any]:
    inputs = _object(report.get("inputs"), f"{report_label}.inputs")
    recorded_registry = _recorded_file(
        inputs.get("rendered_registry"),
        f"{report_label}.inputs.rendered_registry",
    )
    if recorded_registry != registry_path:
        raise FinalVisualGateError(
            f"{report_label} is bound to {recorded_registry}, not {registry_path}"
        )
    recorded_hash = _text(
        inputs.get("rendered_registry_sha256"),
        f"{report_label}.inputs.rendered_registry_sha256",
    )
    actual_hash = _sha256_file(registry_path)
    if recorded_hash != actual_hash:
        raise FinalVisualGateError(
            f"{report_label} rendered registry hash does not match the file"
        )
    return inputs


def _validate_reference_binding(
    baseline_inputs: Mapping[str, Any],
    final_inputs: Mapping[str, Any],
) -> tuple[Path, str]:
    baseline_manifest = _recorded_file(
        baseline_inputs.get("reference_manifest"),
        "baseline inputs.reference_manifest",
    )
    final_manifest = _recorded_file(
        final_inputs.get("reference_manifest"),
        "final inputs.reference_manifest",
    )
    baseline_hash = _text(
        baseline_inputs.get("reference_manifest_sha256"),
        "baseline reference manifest SHA-256",
    )
    final_hash = _text(
        final_inputs.get("reference_manifest_sha256"),
        "final reference manifest SHA-256",
    )
    if baseline_manifest != final_manifest or baseline_hash != final_hash:
        raise FinalVisualGateError(
            "baseline and final quality reports use different reference manifests"
        )
    if _sha256_file(baseline_manifest) != baseline_hash:
        raise FinalVisualGateError("reference manifest hash does not match the file")
    if baseline_inputs.get("selected_view_mapping") != final_inputs.get(
        "selected_view_mapping"
    ):
        raise FinalVisualGateError(
            "baseline and final reports use different selected view mappings"
        )
    return baseline_manifest, baseline_hash


def _validate_registry_asset(
    registry: Mapping[str, Any],
    registry_label: str,
) -> tuple[Path, str]:
    asset = _recorded_file(
        registry.get("asset_usd"),
        f"{registry_label}.asset_usd",
    )
    asset_hash = _text(
        registry.get("asset_sha256"),
        f"{registry_label}.asset_sha256",
    )
    if _sha256_file(asset) != asset_hash:
        raise FinalVisualGateError(f"{registry_label} asset hash is stale")
    render_set = _object(registry.get("render_set"), f"{registry_label}.render_set")
    render_asset = _recorded_file(
        render_set.get("asset_usd"),
        f"{registry_label}.render_set.asset_usd",
    )
    if render_asset != asset:
        raise FinalVisualGateError(
            f"{registry_label} top-level and render-set assets disagree"
        )
    return asset, asset_hash


def _validate_report_render_files(
    *,
    report_views: Mapping[str, Mapping[str, Any]],
    registry_views: Mapping[str, Mapping[str, Any]],
    registry_root: Path,
    label: str,
) -> set[Path]:
    used_paths: set[Path] = set()
    for reference_view_id, view in report_views.items():
        render_view_id = _text(
            view.get("render_view_id"),
            f"{label} view {reference_view_id}.render_view_id",
        )
        registry_view = registry_views.get(render_view_id)
        if registry_view is None:
            raise FinalVisualGateError(
                f"{label} view {reference_view_id} is absent from its registry"
            )
        render = _object(
            view.get("render"),
            f"{label} view {reference_view_id}.render",
        )
        for field, report_hash_field in (
            ("rgb", "image_sha256"),
            ("part_ids", "part_ids_sha256"),
        ):
            report_field = "image" if field == "rgb" else "part_ids"
            report_path = _recorded_file(
                render.get(report_field),
                f"{label} view {reference_view_id} {report_field}",
            )
            registry_path = _recorded_file(
                registry_view.get(field),
                f"{label} registry view {render_view_id} {field}",
            )
            if report_path != registry_path:
                raise FinalVisualGateError(
                    f"{label} view {reference_view_id} does not use its "
                    f"registry {field}"
                )
            if not _is_inside(report_path, registry_root):
                raise FinalVisualGateError(
                    f"{label} view {reference_view_id} escapes its render directory"
                )
            expected_hash = _text(
                render.get(report_hash_field),
                f"{label} view {reference_view_id} {report_hash_field}",
            )
            if _sha256_file(report_path) != expected_hash:
                raise FinalVisualGateError(
                    f"{label} view {reference_view_id} {field} hash is stale"
                )
            used_paths.add(report_path)
    return used_paths


def _failure_report(
    *,
    inputs: Mapping[str, Any],
    policy: Mapping[str, Any],
    error: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": FAIL_CLOSED,
        "completion_allowed": False,
        "completion_state": FAILED_COMPLETION_STATE,
        "reason_codes": ["INVALID_OR_UNVERIFIED_FINAL_VISUAL_EVIDENCE"],
        "error": str(error),
        "inputs": dict(inputs),
        "policy": dict(policy),
        "provenance": {
            "independent_final_render_verified": False,
            "collected_asset_hash_verified": False,
            "comparison_contract_verified": False,
            "sealed_baseline_evidence_verified": False,
        },
        "summary": {
            "view_count": 0,
            "passed_view_count": 0,
            "significant_group_count": 0,
            "passed_significant_group_count": 0,
            "failure_count": 1,
        },
        "views": [],
    }


def evaluate_final_visual_gate(
    *,
    collected_usd_path: Path,
    baseline_quality_path: Path,
    final_quality_path: Path,
    baseline_registry_path: Path,
    final_registry_path: Path,
    baseline_quality: Mapping[str, Any],
    final_quality: Mapping[str, Any],
    baseline_registry: Mapping[str, Any],
    final_registry: Mapping[str, Any],
    maximum_score_regression: float = DEFAULT_MAXIMUM_SCORE_REGRESSION,
    maximum_group_recall_regression: float = (DEFAULT_MAXIMUM_GROUP_RECALL_REGRESSION),
    maximum_group_share_error_regression: float = (
        DEFAULT_MAXIMUM_GROUP_SHARE_ERROR_REGRESSION
    ),
    minimum_final_appearance_score: float = (DEFAULT_MINIMUM_FINAL_APPEARANCE_SCORE),
    minimum_final_view_appearance_score: float = (
        DEFAULT_MINIMUM_FINAL_VIEW_APPEARANCE_SCORE
    ),
    minimum_significant_reference_share: float = (
        DEFAULT_MINIMUM_SIGNIFICANT_REFERENCE_SHARE
    ),
    minimum_significant_evidence_pixels: int = (
        DEFAULT_MINIMUM_SIGNIFICANT_EVIDENCE_PIXELS
    ),
    require_distinct_baseline_asset: bool = True,
    sealed_baseline_provenance: Mapping[str, Any] | None = None,
    allow_immutable_library_optimum_review: bool = False,
) -> dict[str, Any]:
    """Evaluate one hash-bound final delivery render against its baseline."""

    preservation_mode = sealed_baseline_provenance is not None
    relative_nonregression_enforced = bool(
        preservation_mode or not allow_immutable_library_optimum_review
    )
    configured_maximum_score_regression = maximum_score_regression
    configured_maximum_group_recall_regression = (
        maximum_group_recall_regression
    )
    if allow_immutable_library_optimum_review:
        # RTX/path-traced re-renders of the exact same locked MDL look are not
        # pixel deterministic at the inexpensive selection-render budget.
        # This lane is reachable only through the explicit immutable optimum
        # contract and still enforces the absolute visual floors below.
        maximum_score_regression = max(
            maximum_score_regression,
            IMMUTABLE_LIBRARY_RENDER_REPEATABILITY_TOLERANCE,
        )
        maximum_group_recall_regression = max(
            maximum_group_recall_regression,
            IMMUTABLE_LIBRARY_RENDER_REPEATABILITY_TOLERANCE,
        )
    _schema(baseline_quality, QUALITY_SCHEMA_VERSION, "baseline quality report")
    _schema(final_quality, QUALITY_SCHEMA_VERSION, "final quality report")
    _schema(baseline_registry, REGISTRY_SCHEMA_VERSION, "baseline registry")
    _schema(final_registry, REGISTRY_SCHEMA_VERSION, "final registry")

    baseline_inputs = _validate_report_binding(
        report=baseline_quality,
        report_label="baseline quality report",
        registry_path=baseline_registry_path,
    )
    final_inputs = _validate_report_binding(
        report=final_quality,
        report_label="final quality report",
        registry_path=final_registry_path,
    )
    reference_manifest, reference_manifest_hash = _validate_reference_binding(
        baseline_inputs,
        final_inputs,
    )

    if baseline_quality.get("thresholds") != final_quality.get("thresholds"):
        raise FinalVisualGateError(
            "baseline and final reports use different comparison thresholds"
        )

    baseline_asset, baseline_asset_hash = _validate_registry_asset(
        baseline_registry,
        "baseline registry",
    )
    final_asset, final_asset_hash = _validate_registry_asset(
        final_registry,
        "final registry",
    )
    collected_hash = _sha256_file(collected_usd_path)
    if final_asset != collected_usd_path or final_asset_hash != collected_hash:
        raise FinalVisualGateError(
            "final rendered registry is not bound to the exact collected USD"
        )

    baseline_render_set, baseline_registry_views = _registry_views(
        baseline_registry,
        "baseline registry",
    )
    final_render_set, final_registry_views = _registry_views(
        final_registry,
        "final registry",
    )
    if any(
        baseline_render_set.get(field) != final_render_set.get(field)
        for field in _RENDER_CONTRACT_FIELDS
    ):
        raise FinalVisualGateError(
            "baseline and final renders use different camera/lighting contracts"
        )

    baseline_views = _report_views(baseline_quality, "baseline quality report")
    final_views = _report_views(final_quality, "final quality report")
    if set(baseline_views) != set(final_views):
        raise FinalVisualGateError(
            "final quality report does not exactly cover baseline reference views"
        )
    sealed_acceptance_audit = (
        _validate_sealed_acceptance_contract(
            provenance=sealed_baseline_provenance,
            reference_manifest=reference_manifest,
            baseline_inputs=baseline_inputs,
            final_inputs=final_inputs,
            baseline_quality=baseline_quality,
            final_quality=final_quality,
            baseline_render_set=baseline_render_set,
            final_render_set=final_render_set,
            baseline_registry_views=baseline_registry_views,
            final_registry_views=final_registry_views,
            baseline_report_views=baseline_views,
            final_report_views=final_views,
        )
        if sealed_baseline_provenance is not None
        else None
    )

    baseline_render_paths = _validate_report_render_files(
        report_views=baseline_views,
        registry_views=baseline_registry_views,
        registry_root=baseline_registry_path.parent,
        label="baseline",
    )
    final_render_paths = _validate_report_render_files(
        report_views=final_views,
        registry_views=final_registry_views,
        registry_root=final_registry_path.parent,
        label="final",
    )

    reasons: list[str] = []
    if baseline_quality_path == final_quality_path:
        reasons.append("FINAL_QUALITY_REPORT_REUSED_BASELINE")
    if baseline_registry_path == final_registry_path:
        reasons.append("FINAL_RENDER_REGISTRY_REUSED_BASELINE")
    if require_distinct_baseline_asset and baseline_asset == collected_usd_path:
        reasons.append("BASELINE_ASSET_IS_COLLECTED_ASSET")
    if baseline_render_paths & final_render_paths:
        reasons.append("FINAL_RENDER_FILES_REUSED_BASELINE")
    if _sha256_file(baseline_registry_path) == _sha256_file(final_registry_path):
        reasons.append("FINAL_RENDER_REGISTRY_CONTENT_REUSED_BASELINE")
    if _sha256_file(baseline_quality_path) == _sha256_file(final_quality_path):
        reasons.append("FINAL_QUALITY_REPORT_CONTENT_REUSED_BASELINE")

    baseline_aggregate, baseline_scores = _aggregate_scores(
        baseline_quality,
        "baseline quality report",
    )
    final_aggregate, final_scores = _aggregate_scores(
        final_quality,
        "final quality report",
    )
    final_thresholds = _object(
        final_quality.get("thresholds"),
        "final quality report thresholds",
    )
    pass_color_score = _unit(
        final_thresholds.get("pass_color_score"),
        "final quality pass_color_score",
    )

    def immutable_review_view_is_accepted(
        view: Mapping[str, Any],
        scores: Mapping[str, float],
    ) -> bool:
        raw_reasons = view.get("reasons")
        material_texture = view.get("material_texture")
        return bool(
            allow_immutable_library_optimum_review
            and view.get("status") == "REVIEW"
            and isinstance(raw_reasons, list)
            and raw_reasons
            and all(isinstance(reason, str) for reason in raw_reasons)
            and set(raw_reasons).issubset(
                _IMMUTABLE_LIBRARY_ALLOWED_REVIEW_REASONS
            )
            and scores["color"] >= pass_color_score
            and isinstance(material_texture, Mapping)
            and material_texture.get("status") == PASS
        )

    final_failed_count = _integer(
        final_aggregate.get("failed_view_count"),
        "final aggregate failed_view_count",
    )
    final_unscorable_count = _integer(
        final_aggregate.get("unscorable_view_count"),
        "final aggregate unscorable_view_count",
    )
    final_aggregate_status_accepted = bool(
        final_aggregate.get("status") == PASS
        or (
            allow_immutable_library_optimum_review
            and final_aggregate.get("status") == "REVIEW"
            and final_failed_count == 0
            and final_unscorable_count == 0
            and final_scores["color"] >= pass_color_score
        )
    )
    if preservation_mode and baseline_aggregate.get("status") != PASS:
        reasons.append("BASELINE_AGGREGATE_STATUS_NOT_PASS")
    if not final_aggregate_status_accepted:
        reasons.append("FINAL_AGGREGATE_STATUS_NOT_PASS")
    if final_scores["appearance"] < minimum_final_appearance_score:
        reasons.append("FINAL_AGGREGATE_APPEARANCE_BELOW_FLOOR")
    if preservation_mode:
        if _quality_status_rank(
            final_aggregate.get("status"),
            "final aggregate status",
            aggregate=True,
        ) < _quality_status_rank(
            baseline_aggregate.get("status"),
            "baseline aggregate status",
            aggregate=True,
        ):
            reasons.append("AGGREGATE_STATUS_REGRESSION")
    observed_aggregate_regressions = [
        metric
        for metric in ("color", "texture", "appearance")
        if final_scores[metric] < baseline_scores[metric] - maximum_score_regression
    ]
    aggregate_regressions = (
        observed_aggregate_regressions if relative_nonregression_enforced else []
    )
    if aggregate_regressions:
        reasons.append("AGGREGATE_VISUAL_SCORE_REGRESSION")

    aggregate_coverage_regressed = False
    for count_field, relation in (
        ("failed_view_count", "maximum"),
        ("review_view_count", "maximum"),
        ("unscorable_view_count", "maximum"),
        ("comparable_view_count", "minimum"),
    ):
        baseline_count = _integer(
            baseline_aggregate.get(count_field),
            f"baseline aggregate {count_field}",
        )
        final_count = _integer(
            final_aggregate.get(count_field),
            f"final aggregate {count_field}",
        )
        regressed = (
            final_count > baseline_count
            if relation == "maximum"
            else final_count < baseline_count
        )
        if regressed and relative_nonregression_enforced:
            reasons.append("AGGREGATE_VIEW_COVERAGE_REGRESSION")
            aggregate_coverage_regressed = True
            break

    precomputed_view_quality: dict[
        str,
        tuple[dict[str, float], dict[str, float], list[str], list[str]],
    ] = {}
    whole_report_quality_nonregressing = (
        final_aggregate_status_accepted
        and final_scores["appearance"] >= minimum_final_appearance_score
        and not aggregate_regressions
        and not aggregate_coverage_regressed
    )
    for view_id in sorted(baseline_views):
        baseline_view_scores = _quality_score_record(
            baseline_views[view_id],
            f"baseline view {view_id}",
        )
        final_view_scores = _quality_score_record(
            final_views[view_id],
            f"final view {view_id}",
        )
        observed_regressed_metrics = [
            metric
            for metric in (
                "color",
                "texture",
                "appearance",
                "alignment",
                "silhouette_iou",
            )
            if final_view_scores[metric]
            < baseline_view_scores[metric] - maximum_score_regression
        ]
        regressed_metrics = (
            observed_regressed_metrics if relative_nonregression_enforced else []
        )
        precomputed_view_quality[view_id] = (
            baseline_view_scores,
            final_view_scores,
            regressed_metrics,
            observed_regressed_metrics,
        )
        final_view_status_accepted = bool(
            final_views[view_id].get("status") == PASS
            or immutable_review_view_is_accepted(
                final_views[view_id],
                final_view_scores,
            )
        )
        if (
            not final_view_status_accepted
            or regressed_metrics
            or final_view_scores["appearance"] < minimum_final_view_appearance_score
        ):
            whole_report_quality_nonregressing = False

    view_records: list[dict[str, Any]] = []
    significant_group_count = 0
    passed_significant_group_count = 0
    passed_view_count = 0
    for view_id in sorted(baseline_views):
        baseline_view = baseline_views[view_id]
        final_view = final_views[view_id]
        view_reasons: list[str] = []
        if baseline_view.get("render_view_id") != final_view.get("render_view_id"):
            view_reasons.append("RENDER_VIEW_ID_CHANGED")
        if preservation_mode and baseline_view.get("status") != PASS:
            view_reasons.append("BASELINE_VIEW_STATUS_NOT_PASS")
        final_view_status_accepted = bool(
            final_view.get("status") == PASS
            or immutable_review_view_is_accepted(
                final_view,
                precomputed_view_quality[view_id][1],
            )
        )
        if not final_view_status_accepted:
            view_reasons.append("FINAL_VIEW_STATUS_NOT_PASS")
        if preservation_mode:
            if _quality_status_rank(
                final_view.get("status"),
                f"final view {view_id} status",
                aggregate=False,
            ) < _quality_status_rank(
                baseline_view.get("status"),
                f"baseline view {view_id} status",
                aggregate=False,
            ):
                view_reasons.append("FINAL_VIEW_STATUS_REGRESSION")
        (
            baseline_view_scores,
            final_view_scores,
            regressed_metrics,
            observed_regressed_metrics,
        ) = precomputed_view_quality[view_id]
        if regressed_metrics:
            view_reasons.append("PER_VIEW_VISUAL_SCORE_REGRESSION")
        if final_view_scores["appearance"] < minimum_final_view_appearance_score:
            view_reasons.append("FINAL_VIEW_APPEARANCE_BELOW_FLOOR")

        baseline_groups = _significant_groups(
            baseline_view,
            f"baseline view {view_id}",
            minimum_reference_share=minimum_significant_reference_share,
            minimum_evidence_pixels=minimum_significant_evidence_pixels,
        )
        final_groups = _all_groups(final_view, f"final view {view_id}")
        group_records: list[dict[str, Any]] = []
        for group_id in sorted(baseline_groups):
            significant_group_count += 1
            baseline_group = baseline_groups[group_id]
            final_group = final_groups.get(group_id)
            group_reasons: list[str] = []
            recall_quantization_audit: dict[str, Any] | None = None
            if final_group is None:
                group_reasons.append("SIGNIFICANT_GROUP_MISSING")
                baseline_recall = _unit(
                    baseline_group.get("recall"),
                    f"baseline view {view_id} group {group_id} recall",
                )
                final_recall = None
                baseline_share_error = None
                final_share_error = None
            else:
                group_identity_unchanged = True
                for identity_field in (
                    "base_colors",
                    "reference_evidence_weight",
                    "reference_group_share",
                ):
                    if baseline_group.get(identity_field) != final_group.get(
                        identity_field
                    ):
                        group_reasons.append("REFERENCE_GROUP_IDENTITY_CHANGED")
                        group_identity_unchanged = False
                        break
                baseline_recall = _unit(
                    baseline_group.get("recall"),
                    f"baseline view {view_id} group {group_id} recall",
                )
                final_recall = _unit(
                    final_group.get("recall"),
                    f"final view {view_id} group {group_id} recall",
                )
                reference_share = _unit(
                    baseline_group.get("reference_group_share"),
                    f"baseline view {view_id} group {group_id} reference share",
                )
                baseline_observed = _unit(
                    baseline_group.get("observed_render_share"),
                    f"baseline view {view_id} group {group_id} observed share",
                )
                final_observed = _unit(
                    final_group.get("observed_render_share"),
                    f"final view {view_id} group {group_id} observed share",
                )
                baseline_share_error = abs(baseline_observed - reference_share)
                final_share_error = abs(final_observed - reference_share)
                observed_group_share_error_regressed = (
                    final_share_error
                    > baseline_share_error + maximum_group_share_error_regression
                )
                group_share_error_regressed = bool(
                    relative_nonregression_enforced
                    and observed_group_share_error_regressed
                )
                if group_share_error_regressed:
                    group_reasons.append("SIGNIFICANT_GROUP_SHARE_ERROR_REGRESSION")
                final_presence = final_group.get("delivery_presence_status")
                baseline_presence = baseline_group.get("delivery_presence_status")
                if preservation_mode:
                    if baseline_presence not in {"PRESENT", "MISSING"}:
                        raise FinalVisualGateError(
                            f"baseline view {view_id} group {group_id} has "
                            "invalid delivery presence status"
                        )
                    if final_presence not in {"PRESENT", "MISSING"}:
                        raise FinalVisualGateError(
                            f"final view {view_id} group {group_id} has "
                            "invalid delivery presence status"
                        )
                    if baseline_presence == "PRESENT" and final_presence != "PRESENT":
                        group_reasons.append("SIGNIFICANT_GROUP_PRESENCE_REGRESSION")
                elif final_presence != "PRESENT":
                    group_reasons.append("SIGNIFICANT_GROUP_NOT_PRESENT")
                recall_quantization_audit = _group_recall_quantization_audit(
                    baseline_view=baseline_view,
                    final_view=final_view,
                    baseline_group=baseline_group,
                    final_group=final_group,
                    baseline_recall=baseline_recall,
                    final_recall=final_recall,
                    maximum_group_recall_regression=(maximum_group_recall_regression),
                    minimum_significant_evidence_pixels=(
                        minimum_significant_evidence_pixels
                    ),
                    whole_report_quality_nonregressing=(
                        whole_report_quality_nonregressing
                    ),
                    group_identity_unchanged=group_identity_unchanged,
                    group_share_nonregressing=not group_share_error_regressed,
                    group_presence_preserved=(
                        baseline_presence == "PRESENT" and final_presence == "PRESENT"
                    ),
                )
                if (
                    relative_nonregression_enforced
                    and final_recall
                    < baseline_recall - maximum_group_recall_regression
                    and not recall_quantization_audit["tolerance_applied"]
                ):
                    group_reasons.append("SIGNIFICANT_GROUP_RECALL_REGRESSION")
            group_status = PASS if not group_reasons else FAIL_CLOSED
            if group_status == PASS:
                passed_significant_group_count += 1
            else:
                view_reasons.extend(group_reasons)
            group_records.append(
                {
                    "group_id": group_id,
                    "status": group_status,
                    "reason_codes": sorted(set(group_reasons)),
                    "baseline_recall": baseline_recall,
                    "final_recall": final_recall,
                    "recall_delta": (
                        final_recall - baseline_recall
                        if final_recall is not None
                        else None
                    ),
                    "baseline_share_error": baseline_share_error,
                    "final_share_error": final_share_error,
                    "observed_share_error_regressed": (
                        observed_group_share_error_regressed
                        if final_group is not None
                        else None
                    ),
                    "observed_recall_regressed": (
                        final_recall
                        < baseline_recall - maximum_group_recall_regression
                        if final_recall is not None
                        else None
                    ),
                    "recall_quantization_audit": recall_quantization_audit,
                }
            )

        view_status = PASS if not view_reasons else FAIL_CLOSED
        if view_status == PASS:
            passed_view_count += 1
        else:
            reasons.extend(view_reasons)
        view_records.append(
            {
                "reference_view_id": view_id,
                "render_view_id": final_view.get("render_view_id"),
                "status": view_status,
                "reason_codes": sorted(set(view_reasons)),
                "baseline_scores": baseline_view_scores,
                "final_scores": final_view_scores,
                "score_deltas": {
                    metric: final_view_scores[metric] - baseline_view_scores[metric]
                    for metric in baseline_view_scores
                },
                "regressed_metrics": regressed_metrics,
                "observed_regressed_metrics": observed_regressed_metrics,
                "significant_groups": group_records,
            }
        )

    unique_reasons = sorted(set(reasons))
    status = PASS if not unique_reasons else FAIL_CLOSED
    completion_allowed = status == PASS
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "completion_allowed": completion_allowed,
        "completion_state": (
            COMPLETED if completion_allowed else FAILED_COMPLETION_STATE
        ),
        "reason_codes": unique_reasons,
        "inputs": {
            "collected_usd": str(collected_usd_path),
            "collected_usd_sha256": collected_hash,
            "baseline_quality_report": str(baseline_quality_path),
            "baseline_quality_report_sha256": _sha256_file(baseline_quality_path),
            "final_quality_report": str(final_quality_path),
            "final_quality_report_sha256": _sha256_file(final_quality_path),
            "baseline_rendered_registry": str(baseline_registry_path),
            "baseline_rendered_registry_sha256": _sha256_file(baseline_registry_path),
            "final_rendered_registry": str(final_registry_path),
            "final_rendered_registry_sha256": _sha256_file(final_registry_path),
            "reference_manifest": str(reference_manifest),
            "reference_manifest_sha256": reference_manifest_hash,
        },
        "policy": {
            "maximum_score_regression": maximum_score_regression,
            "maximum_group_recall_regression": (maximum_group_recall_regression),
            "configured_maximum_score_regression": (
                configured_maximum_score_regression
            ),
            "configured_maximum_group_recall_regression": (
                configured_maximum_group_recall_regression
            ),
            "immutable_library_render_repeatability_tolerance": (
                IMMUTABLE_LIBRARY_RENDER_REPEATABILITY_TOLERANCE
                if allow_immutable_library_optimum_review
                else None
            ),
            "maximum_group_share_error_regression": (
                maximum_group_share_error_regression
            ),
            "minimum_final_appearance_score": minimum_final_appearance_score,
            "minimum_final_view_appearance_score": (
                minimum_final_view_appearance_score
            ),
            "minimum_significant_reference_share": (
                minimum_significant_reference_share
            ),
            "minimum_significant_evidence_pixels": (
                minimum_significant_evidence_pixels
            ),
            "require_distinct_baseline_asset": require_distinct_baseline_asset,
            "acceptance_mode": (
                SEALED_BASELINE_PRESERVATION_MODE
                if preservation_mode
                else (
                    IMMUTABLE_LIBRARY_OPTIMUM_MODE
                    if allow_immutable_library_optimum_review
                    else ABSOLUTE_PASS_MODE
                )
            ),
            "absolute_quality_floors_enforced": True,
            "relative_nonregression_enforced": relative_nonregression_enforced,
            "immutable_library_relative_render_scores_are_advisory": (
                allow_immutable_library_optimum_review
                and not relative_nonregression_enforced
            ),
            "sealed_contract_absolute_pass_required": preservation_mode,
            "immutable_library_review_allowed": (
                allow_immutable_library_optimum_review
            ),
            "immutable_library_allowed_review_reason_codes": sorted(
                _IMMUTABLE_LIBRARY_ALLOWED_REVIEW_REASONS
            ),
            "group_recall_regression_method": (
                "relative_recall_and_integer_pixel_loss/v1"
            ),
        },
        "provenance": {
            "independent_final_render_verified": (
                baseline_registry_path != final_registry_path
                and not (baseline_render_paths & final_render_paths)
                and final_asset == collected_usd_path
                and final_asset_hash == collected_hash
            ),
            "collected_asset_hash_verified": (
                final_asset == collected_usd_path and final_asset_hash == collected_hash
            ),
            "comparison_contract_verified": True,
            "sealed_baseline_evidence_verified": preservation_mode,
            "sealed_baseline_evidence": (
                dict(sealed_baseline_provenance)
                if sealed_baseline_provenance is not None
                else None
            ),
            "sealed_acceptance_contract": sealed_acceptance_audit,
            "baseline_asset_usd": str(baseline_asset),
            "baseline_asset_sha256": baseline_asset_hash,
            "final_asset_usd": str(final_asset),
            "final_asset_sha256": final_asset_hash,
            "render_contract_sha256": _canonical_sha256(
                {
                    field: final_render_set.get(field)
                    for field in _RENDER_CONTRACT_FIELDS
                }
            ),
        },
        "aggregate": {
            "baseline_scores": baseline_scores,
            "final_scores": final_scores,
            "score_deltas": {
                metric: final_scores[metric] - baseline_scores[metric]
                for metric in baseline_scores
            },
            "regressed_metrics": aggregate_regressions,
            "observed_regressed_metrics": observed_aggregate_regressions,
        },
        "summary": {
            "view_count": len(view_records),
            "passed_view_count": passed_view_count,
            "significant_group_count": significant_group_count,
            "passed_significant_group_count": passed_significant_group_count,
            "failure_count": len(unique_reasons),
        },
        "views": view_records,
    }


def run_final_visual_gate(
    *,
    collected_usd: str | Path,
    baseline_quality_report: str | Path,
    final_quality_report: str | Path,
    baseline_rendered_registry: str | Path,
    final_rendered_registry: str | Path,
    output: str | Path,
    maximum_score_regression: float = DEFAULT_MAXIMUM_SCORE_REGRESSION,
    maximum_group_recall_regression: float = (DEFAULT_MAXIMUM_GROUP_RECALL_REGRESSION),
    maximum_group_share_error_regression: float = (
        DEFAULT_MAXIMUM_GROUP_SHARE_ERROR_REGRESSION
    ),
    minimum_final_appearance_score: float = (DEFAULT_MINIMUM_FINAL_APPEARANCE_SCORE),
    minimum_final_view_appearance_score: float = (
        DEFAULT_MINIMUM_FINAL_VIEW_APPEARANCE_SCORE
    ),
    minimum_significant_reference_share: float = (
        DEFAULT_MINIMUM_SIGNIFICANT_REFERENCE_SHARE
    ),
    minimum_significant_evidence_pixels: int = (
        DEFAULT_MINIMUM_SIGNIFICANT_EVIDENCE_PIXELS
    ),
    require_distinct_baseline_asset: bool = True,
    sealed_baseline_evidence: str | Path | None = None,
    allow_immutable_library_optimum_review: bool = False,
) -> dict[str, Any]:
    """Read, evaluate, and atomically write one final visual gate report."""

    evaluation_policy = {
        "maximum_score_regression": maximum_score_regression,
        "maximum_group_recall_regression": maximum_group_recall_regression,
        "maximum_group_share_error_regression": (maximum_group_share_error_regression),
        "minimum_final_appearance_score": minimum_final_appearance_score,
        "minimum_final_view_appearance_score": (minimum_final_view_appearance_score),
        "minimum_significant_reference_share": (minimum_significant_reference_share),
        "minimum_significant_evidence_pixels": (minimum_significant_evidence_pixels),
        "require_distinct_baseline_asset": require_distinct_baseline_asset,
        "allow_immutable_library_optimum_review": (
            allow_immutable_library_optimum_review
        ),
    }
    policy = {
        **evaluation_policy,
        "acceptance_mode": (
            SEALED_BASELINE_PRESERVATION_MODE
            if sealed_baseline_evidence is not None
            else (
                IMMUTABLE_LIBRARY_OPTIMUM_MODE
                if allow_immutable_library_optimum_review
                else ABSOLUTE_PASS_MODE
            )
        ),
    }
    for name, value in evaluation_policy.items():
        if name in {
            "require_distinct_baseline_asset",
            "allow_immutable_library_optimum_review",
        }:
            if not isinstance(value, bool):
                raise FinalVisualGateError(
                    f"{name} must be boolean"
                )
            continue
        if name == "minimum_significant_evidence_pixels":
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise FinalVisualGateError(
                    "minimum_significant_evidence_pixels must be positive"
                )
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise FinalVisualGateError(f"{name} must be between zero and one")

    raw_inputs = {
        "collected_usd": str(collected_usd),
        "baseline_quality_report": str(baseline_quality_report),
        "final_quality_report": str(final_quality_report),
        "baseline_rendered_registry": str(baseline_rendered_registry),
        "final_rendered_registry": str(final_rendered_registry),
        "sealed_baseline_evidence": (
            str(sealed_baseline_evidence)
            if sealed_baseline_evidence is not None
            else None
        ),
    }
    try:
        collected_path = _input_file(collected_usd, "collected USD")
        baseline_quality_path = _input_file(
            baseline_quality_report,
            "baseline quality report",
        )
        final_quality_path = _input_file(
            final_quality_report,
            "final quality report",
        )
        baseline_registry_path = _input_file(
            baseline_rendered_registry,
            "baseline rendered registry",
        )
        final_registry_path = _input_file(
            final_rendered_registry,
            "final rendered registry",
        )
        sealed_baseline_provenance = (
            _sealed_baseline_evidence_provenance(
                _input_file(
                    sealed_baseline_evidence,
                    "sealed baseline evidence",
                )
            )
            if sealed_baseline_evidence is not None
            else None
        )
        report = evaluate_final_visual_gate(
            collected_usd_path=collected_path,
            baseline_quality_path=baseline_quality_path,
            final_quality_path=final_quality_path,
            baseline_registry_path=baseline_registry_path,
            final_registry_path=final_registry_path,
            baseline_quality=_read_object(
                baseline_quality_path,
                "baseline quality report",
            ),
            final_quality=_read_object(
                final_quality_path,
                "final quality report",
            ),
            baseline_registry=_read_object(
                baseline_registry_path,
                "baseline rendered registry",
            ),
            final_registry=_read_object(
                final_registry_path,
                "final rendered registry",
            ),
            sealed_baseline_provenance=sealed_baseline_provenance,
            **evaluation_policy,
        )
    except (FinalVisualGateError, OSError) as exc:
        report = _failure_report(inputs=raw_inputs, policy=policy, error=exc)

    _write_json_atomic(Path(output), report)
    return report


def require_final_visual_gate_passed(report: Mapping[str, Any]) -> None:
    """Reject pipeline completion unless the final delivery gate passed."""

    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("status") != PASS
        or report.get("completion_allowed") is not True
        or report.get("completion_state") != COMPLETED
    ):
        raise FinalVisualGateError(
            "final collected USD did not pass independent visual acceptance"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Gate completion on an independent re-render of the collected USD")
    )
    parser.add_argument("--collected-usd", type=Path, required=True)
    parser.add_argument("--baseline-quality-report", type=Path, required=True)
    parser.add_argument("--final-quality-report", type=Path, required=True)
    parser.add_argument("--baseline-rendered-registry", type=Path, required=True)
    parser.add_argument("--final-rendered-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--maximum-score-regression",
        type=float,
        default=DEFAULT_MAXIMUM_SCORE_REGRESSION,
    )
    parser.add_argument(
        "--maximum-group-recall-regression",
        type=float,
        default=DEFAULT_MAXIMUM_GROUP_RECALL_REGRESSION,
    )
    parser.add_argument(
        "--maximum-group-share-error-regression",
        type=float,
        default=DEFAULT_MAXIMUM_GROUP_SHARE_ERROR_REGRESSION,
    )
    parser.add_argument(
        "--minimum-final-appearance-score",
        type=float,
        default=DEFAULT_MINIMUM_FINAL_APPEARANCE_SCORE,
    )
    parser.add_argument(
        "--minimum-final-view-appearance-score",
        type=float,
        default=DEFAULT_MINIMUM_FINAL_VIEW_APPEARANCE_SCORE,
    )
    parser.add_argument(
        "--minimum-significant-reference-share",
        type=float,
        default=DEFAULT_MINIMUM_SIGNIFICANT_REFERENCE_SHARE,
    )
    parser.add_argument(
        "--minimum-significant-evidence-pixels",
        type=int,
        default=DEFAULT_MINIMUM_SIGNIFICANT_EVIDENCE_PIXELS,
    )
    parser.add_argument(
        "--allow-same-baseline-asset",
        action="store_true",
        help=(
            "allow two independent renders of the same immutable USD; never "
            "use this for the collected-delivery gate"
        ),
    )
    parser.add_argument(
        "--sealed-baseline-preservation-evidence",
        type=Path,
        help=(
            "enable no-regression-only acceptance for one independently "
            "validated sealed historical baseline"
        ),
    )
    parser.add_argument(
        "--allow-immutable-library-optimum-review",
        action="store_true",
        help=(
            "accept a no-regression immutable-library REVIEW only when every "
            "view clears the configured color, texture, alignment, and "
            "appearance floors and the remaining reason is photometric value"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_final_visual_gate(
        collected_usd=args.collected_usd,
        baseline_quality_report=args.baseline_quality_report,
        final_quality_report=args.final_quality_report,
        baseline_rendered_registry=args.baseline_rendered_registry,
        final_rendered_registry=args.final_rendered_registry,
        output=args.output,
        maximum_score_regression=args.maximum_score_regression,
        maximum_group_recall_regression=(args.maximum_group_recall_regression),
        maximum_group_share_error_regression=(
            args.maximum_group_share_error_regression
        ),
        minimum_final_appearance_score=args.minimum_final_appearance_score,
        minimum_final_view_appearance_score=(args.minimum_final_view_appearance_score),
        minimum_significant_reference_share=(args.minimum_significant_reference_share),
        minimum_significant_evidence_pixels=(args.minimum_significant_evidence_pixels),
        require_distinct_baseline_asset=not args.allow_same_baseline_asset,
        sealed_baseline_evidence=(args.sealed_baseline_preservation_evidence),
        allow_immutable_library_optimum_review=(
            args.allow_immutable_library_optimum_review
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["completion_allowed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
