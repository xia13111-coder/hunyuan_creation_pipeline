#!/usr/bin/env python3
"""Build the browser-safe summary consumed by the result viewer.

The material result is embedded in the root pipeline ``result.json`` for new
runs.  Older deliveries may only expose ``visual_quality_status`` or the
render-comparison report.  This module normalizes those layouts without
following artifact paths outside the selected delivery directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "qwen-material-result-viewer/v1"
LIMITED_PASS = "MATERIAL_ACCEPTED_WITH_GEOMETRY_POSE_LIMITATION"
RESTORED_BASELINE = "RESTORED_HISTORICAL_BASELINE"
RESULT_NAMES = (
    "result.json",
    "resume_result.json",
    "pipeline_result.json",
    "sam3d_pipeline_result.json",
)
QUALITY_FIELDS = (
    "visual_quality_raw_status",
    "visual_quality_gate_status",
    "visual_quality_resolution",
    "visual_quality_limitation_count",
    "visual_quality_status",
)
APPEARANCE_ACCEPTED = "ACCEPTED"
APPEARANCE_CANDIDATE_REPORT_FIELDS = (
    "appearance_optimization_candidate_quality_report",
    "appearance_optimization_candidate_raw_quality_report",
)


def _read_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _nonnegative_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _has_quality_fields(value: Mapping[str, Any]) -> bool:
    return any(field in value for field in QUALITY_FIELDS)


def _assignment_result(value: Mapping[str, Any]) -> dict[str, Any] | None:
    steps = value.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        return None
    for raw_step in steps:
        if not isinstance(raw_step, Mapping):
            continue
        if raw_step.get("step") != "assign_visual_materials":
            continue
        result = raw_step.get("result")
        if isinstance(result, Mapping):
            return dict(result)
    return None


def find_visual_material_result(
    pipeline_result: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the visual-material result from supported old and new layouts."""

    if _has_quality_fields(pipeline_result):
        return dict(pipeline_result)

    for field in ("visual_material_result", "visual_material"):
        direct = pipeline_result.get(field)
        if isinstance(direct, Mapping):
            return dict(direct)

    for container in (
        pipeline_result,
        pipeline_result.get("postprocess"),
        pipeline_result.get("result"),
    ):
        if not isinstance(container, Mapping):
            continue
        assignment = _assignment_result(container)
        if assignment is not None:
            return assignment
        nested = container.get("visual_material_result")
        if isinstance(nested, Mapping):
            return dict(nested)
    return None


def _locate_pipeline_result(
    delivery: Path,
) -> tuple[dict[str, Any] | None, Path | None]:
    for name in RESULT_NAMES:
        path = delivery / name
        document = _read_object(path)
        if document is None:
            continue
        result = find_visual_material_result(document)
        if result is not None:
            return result, path
    return None, None


def _visual_material_root(delivery: Path) -> Path:
    nested = delivery / "visual_material"
    if nested.is_dir():
        return nested
    return delivery


def _relative_href(delivery: Path, path: Path) -> str:
    relative = path.relative_to(delivery)
    return "delivery/" + relative.as_posix()


def _delivery_file(
    delivery: Path,
    value: Any,
    *,
    relative_to: Path | None = None,
) -> Path | None:
    """Resolve one recorded artifact without escaping the selected delivery."""

    raw = _text(value)
    if raw is None:
        return None
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = (relative_to or delivery) / candidate
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(delivery)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def _recorded_artifact(
    delivery: Path,
    result: Mapping[str, Any] | None,
    field: str,
) -> Path | None:
    if result is None:
        return None
    return _delivery_file(delivery, result.get(field))


def _locate_resolution(
    delivery: Path,
    visual_material_root: Path,
    result: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, Path | None]:
    # A completed pipeline result is authoritative.  Resume/recovery runs may
    # leave an older resolution report in ``analysis`` even after a later
    # absolute PASS no longer records one.  Falling back to that stale file
    # would mix evidence from two different quality decisions in the viewer.
    if result is not None and "visual_quality_resolution" in result:
        recorded = _recorded_artifact(
            delivery,
            result,
            "visual_quality_resolution",
        )
        if recorded is None:
            return None, None
        document = _read_object(recorded)
        return (document, recorded) if document is not None else (None, None)

    candidates = (
        visual_material_root / "analysis" / "visual_quality_resolution.json",
        visual_material_root / "visual_quality_resolution.json",
    )
    for path in candidates:
        document = _read_object(path)
        if document is not None:
            return document, path
    return None, None


def _locate_quality_report(
    delivery: Path,
    visual_material_root: Path,
    result: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, Path | None]:
    appearance_status = (
        _text(result.get("appearance_optimization_status"))
        if result is not None
        else None
    )
    recorded_candidate_paths = tuple(
        path
        for field in APPEARANCE_CANDIDATE_REPORT_FIELDS
        if (path := _recorded_artifact(delivery, result, field)) is not None
    )
    effective_path = _recorded_artifact(delivery, result, "visual_quality_report")
    if appearance_status == APPEARANCE_ACCEPTED:
        artifact_candidates = (
            _recorded_artifact(
                delivery,
                result,
                "appearance_optimization_candidate_quality_report",
            ),
            effective_path,
        )
    else:
        # Candidate reports are audit artifacts until the optimizer explicitly
        # accepts them.  Even a stale/malformed effective pointer must not make
        # a rejected candidate look like the final result.
        artifact_candidates = (
            effective_path if effective_path not in recorded_candidate_paths else None,
        )

    conventional_candidates = (
        visual_material_root
        / "visual_quality_repair"
        / "reference_render_comparison.json",
        visual_material_root / "visual_quality" / "reference_render_comparison.json",
    )
    seen: set[Path] = set()
    for path in (*artifact_candidates, *conventional_candidates):
        if path is None:
            continue
        try:
            path = path.resolve(strict=True)
            path.relative_to(delivery)
        except (OSError, RuntimeError, ValueError):
            continue
        if path in seen:
            continue
        seen.add(path)
        document = _read_object(path)
        if document is not None:
            return document, path
    return None, None


def _preview_images_from_registry(
    delivery: Path,
    registry_path: Path,
    *,
    accepted_root: Path,
) -> dict[str, str]:
    registry = _read_object(registry_path)
    render_set = registry.get("render_set") if registry is not None else None
    raw_views = render_set.get("views") if isinstance(render_set, Mapping) else None
    if not isinstance(raw_views, Sequence) or isinstance(raw_views, (str, bytes)):
        return {}

    images: dict[str, str] = {}
    for raw_view in raw_views:
        if not isinstance(raw_view, Mapping):
            continue
        view_id = _text(raw_view.get("view_id"))
        image_path = _delivery_file(
            delivery,
            raw_view.get("rgb"),
            relative_to=registry_path.parent,
        )
        if view_id is None or image_path is None:
            continue
        try:
            image_path.relative_to(accepted_root)
        except ValueError:
            continue
        images.setdefault(view_id, _relative_href(delivery, image_path))
    return images


def _preview_images(
    delivery: Path,
    result: Mapping[str, Any] | None,
    quality_report_path: Path | None,
) -> dict[str, str]:
    if quality_report_path is None:
        return {}

    # The report directory is the trust boundary for this accepted QA round.
    # This prevents a rejected appearance registry from being paired with the
    # previously accepted repair report.
    accepted_root = quality_report_path.parent.resolve()
    registry_path = _recorded_artifact(
        delivery,
        result,
        "visual_quality_rendered_registry",
    )
    if registry_path is not None:
        try:
            registry_path.relative_to(accepted_root)
        except ValueError:
            registry_path = None
    if registry_path is not None:
        images = _preview_images_from_registry(
            delivery,
            registry_path,
            accepted_root=accepted_root,
        )
        if images:
            return images

    # Legacy results did not record the rendered registry.  Derive the preview
    # directory from the selected report, never from an unaccepted candidate
    # directory name.
    rgb_dir = accepted_root / "renders" / "rgb"
    if not rgb_dir.is_dir():
        return {}
    images: dict[str, str] = {}
    for image in sorted(rgb_dir.glob("*.png")):
        resolved = _delivery_file(delivery, str(image))
        if resolved is None:
            continue
        try:
            resolved.relative_to(accepted_root)
        except ValueError:
            continue
        images.setdefault(image.stem, _relative_href(delivery, resolved))
    return images


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_bound_delivery_file(
    delivery: Path,
    value: Any,
    expected_path: Path,
    expected_sha256: Any,
) -> Path | None:
    """Resolve one recorded file only when its path and digest are exact."""

    path = _delivery_file(delivery, value)
    digest = _text(expected_sha256)
    if (
        path is None
        or path != expected_path
        or digest is None
        or len(digest) != 64
    ):
        return None
    try:
        int(digest, 16)
        actual = _sha256_file(path)
    except (OSError, ValueError):
        return None
    return path if actual == digest.lower() else None


def _authoritative_final_acceptance_root(
    delivery: Path,
    visual_material_root: Path,
) -> tuple[str, Path | None]:
    """Resolve the completed acceptance round without guessing retry names."""

    result_path = visual_material_root / "final_visual_acceptance_result.json"
    if result_path.exists():
        result = _read_object(result_path)
        if result is None:
            return "REJECTED", None
        raw_root = _text(result.get("output_dir"))
        raw_gate = _text(result.get("collected_visual_gate"))
        if (
            result.get("schema_version")
            != "asset-pipeline-final-visual-acceptance/v1"
            or result.get("state") != "COMPLETED"
            or result.get("completion_allowed") is not True
            or result.get("collected_visual_gate_status") != "PASS"
            or raw_root is None
            or raw_gate is None
        ):
            return "REJECTED", None
        try:
            acceptance_root = Path(raw_root).expanduser().resolve(strict=True)
            acceptance_root.relative_to(delivery)
            gate_path = Path(raw_gate).expanduser().resolve(strict=True)
            gate_path.relative_to(delivery)
        except (OSError, RuntimeError, ValueError):
            return "REJECTED", None
        if (
            not acceptance_root.is_dir()
            or not gate_path.is_file()
            or gate_path != acceptance_root / "collected_visual_gate.json"
        ):
            return "REJECTED", None
        return "AVAILABLE", acceptance_root

    acceptance_root = visual_material_root / "final_visual_acceptance"
    if not acceptance_root.exists():
        return "ABSENT", None
    if not acceptance_root.is_dir():
        return "REJECTED", None
    return "AVAILABLE", acceptance_root


def _final_collected_preview_images(
    delivery: Path,
    visual_material_root: Path,
) -> tuple[str, dict[str, str], dict[str, Any] | None, Path | None]:
    """Return reference-keyed images from one completed collected QA round.

    The state is tri-valued. ``ABSENT`` permits compatibility fallbacks for
    historical deliveries. ``REJECTED`` means final-acceptance artifacts are
    present but incomplete or invalid, so callers must not fall back to a
    candidate/preview image. ``PASS`` exposes only the hash-bound collected
    registry, remapped from render-camera IDs to reference roles.
    """

    acceptance_state, acceptance_root = _authoritative_final_acceptance_root(
        delivery,
        visual_material_root,
    )
    if acceptance_state == "ABSENT":
        return "ABSENT", {}, None, None
    if acceptance_state != "AVAILABLE" or acceptance_root is None:
        return "REJECTED", {}, None, None

    collected_root = acceptance_root / "collected"
    gate_path = acceptance_root / "collected_visual_gate.json"
    quality_path = collected_root / "reference_render_comparison.json"
    view_map_path = collected_root / "reference_view_map.json"
    registry_path = collected_root / "renders" / "part_registry.rendered.json"
    required = (gate_path, quality_path, view_map_path, registry_path)
    if any(not path.is_file() for path in required):
        return "REJECTED", {}, None, None

    gate = _read_object(gate_path)
    quality = _read_object(quality_path)
    view_map = _read_object(view_map_path)
    registry = _read_object(registry_path)
    if any(value is None for value in (gate, quality, view_map, registry)):
        return "REJECTED", {}, None, None
    assert gate is not None
    assert quality is not None
    assert view_map is not None
    assert registry is not None

    gate_inputs = gate.get("inputs")
    gate_schema = gate.get("schema_version")
    part_id_nonregression = (
        gate_schema == "asset-pipeline-part-id-final-visual-gate/v1"
        and gate.get("acceptance_mode") == "PART_ID_VISUAL_NONREGRESSION"
    )
    legacy_final_gate = (
        gate_schema == "qwen-final-visual-gate/v1"
        and gate.get("completion_state") == "COMPLETED"
    )
    if (
        not (legacy_final_gate or part_id_nonregression)
        or gate.get("status") != "PASS"
        or gate.get("completion_allowed") is not True
        or not isinstance(gate_inputs, Mapping)
        or _hash_bound_delivery_file(
            delivery,
            gate_inputs.get("final_quality_report"),
            quality_path.resolve(),
            gate_inputs.get("final_quality_report_sha256"),
        )
        is None
        or _hash_bound_delivery_file(
            delivery,
            gate_inputs.get("final_rendered_registry"),
            registry_path.resolve(),
            gate_inputs.get("final_rendered_registry_sha256"),
        )
        is None
    ):
        return "REJECTED", {}, None, None

    aggregate = quality.get("aggregate")
    quality_inputs = quality.get("inputs")
    raw_views = quality.get("views")
    raw_mapping = (
        quality_inputs.get("selected_view_mapping")
        if isinstance(quality_inputs, Mapping)
        else None
    )
    recorded_registry = (
        _hash_bound_delivery_file(
            delivery,
            quality_inputs.get("rendered_registry"),
            registry_path.resolve(),
            quality_inputs.get("rendered_registry_sha256"),
        )
        if isinstance(quality_inputs, Mapping)
        else None
    )
    policy = gate.get("policy")
    immutable_library_optimum = (
        isinstance(policy, Mapping)
        and policy.get("acceptance_mode") == "IMMUTABLE_LIBRARY_OPTIMUM"
        and policy.get("absolute_quality_floors_enforced") is True
        and policy.get("immutable_library_review_allowed") is True
    )
    if part_id_nonregression:
        allowed_aggregate_statuses = {
            "PASS",
            "REVIEW",
            "INSUFFICIENT_EVIDENCE",
        }
        allowed_view_statuses = {"PASS", "REVIEW", "UNSCORABLE"}
    else:
        allowed_aggregate_statuses = (
            {"PASS", "REVIEW"} if immutable_library_optimum else {"PASS"}
        )
        allowed_view_statuses = allowed_aggregate_statuses
    if (
        quality.get("schema_version") != "qwen-reference-render-comparison/v1"
        or registry.get("schema_version") != "qwen-material-parts/v1"
        or not isinstance(aggregate, Mapping)
        or aggregate.get("status") not in allowed_aggregate_statuses
        or not isinstance(raw_views, list)
        or not raw_views
        or not isinstance(raw_mapping, Mapping)
        or not raw_mapping
        or recorded_registry is None
    ):
        return "REJECTED", {}, None, None

    mapping: dict[str, str] = {}
    for reference_id, render_id in raw_mapping.items():
        if (
            not isinstance(reference_id, str)
            or not reference_id
            or not isinstance(render_id, str)
            or not render_id
            or reference_id in mapping
        ):
            return "REJECTED", {}, None, None
        mapping[reference_id] = render_id
    if len(set(mapping.values())) != len(mapping):
        return "REJECTED", {}, None, None

    if (
        view_map.get("schema_version") != "qwen-reference-view-map/v1"
        or view_map.get("mapping") != mapping
    ):
        return "REJECTED", {}, None, None

    ordered_reference_ids: list[str] = []
    seen_reference_ids: set[str] = set()
    status_by_reference_id: dict[str, str] = {}
    view_by_reference_id: dict[str, Mapping[str, Any]] = {}
    for raw_view in raw_views:
        if (
            not isinstance(raw_view, Mapping)
            or raw_view.get("status") not in allowed_view_statuses
        ):
            return "REJECTED", {}, None, None
        reference_id = _text(raw_view.get("reference_view_id"))
        render_id = _text(raw_view.get("render_view_id"))
        if (
            reference_id is None
            or render_id is None
            or reference_id in seen_reference_ids
            or mapping.get(reference_id) != render_id
        ):
            return "REJECTED", {}, None, None
        seen_reference_ids.add(reference_id)
        ordered_reference_ids.append(reference_id)
        status_by_reference_id[reference_id] = str(raw_view["status"])
        view_by_reference_id[reference_id] = raw_view
    if seen_reference_ids != set(mapping):
        return "REJECTED", {}, None, None

    view_count = len(mapping)
    passed_reference_ids = [
        view_id
        for view_id in ordered_reference_ids
        if status_by_reference_id[view_id] == "PASS"
    ]
    review_reference_ids = [
        view_id
        for view_id in ordered_reference_ids
        if status_by_reference_id[view_id] == "REVIEW"
    ]
    unscorable_reference_ids = [
        view_id
        for view_id in ordered_reference_ids
        if status_by_reference_id[view_id] == "UNSCORABLE"
    ]
    comparable_view_count = len(passed_reference_ids) + len(review_reference_ids)
    expected_aggregate = {
        "reference_view_count": view_count,
        "render_view_count": view_count,
        "comparable_view_count": comparable_view_count,
        "passed_view_count": len(passed_reference_ids),
        "review_view_count": len(review_reference_ids),
        "failed_view_count": 0,
        "unscorable_view_count": len(unscorable_reference_ids),
        "reference_view_coverage_status": (
            "FAIL_CLOSED" if unscorable_reference_ids else "PASS"
        ),
        "unmapped_reference_view_ids": [],
        "unscorable_reference_view_ids": unscorable_reference_ids,
    }
    if any(aggregate.get(key) != value for key, value in expected_aggregate.items()):
        return "REJECTED", {}, None, None
    if not part_id_nonregression and (
        comparable_view_count != view_count
        or (not immutable_library_optimum and review_reference_ids)
    ):
        return "REJECTED", {}, None, None

    if part_id_nonregression:
        measurements = gate.get("measurements")
        final_part_id_gate = (
            measurements.get("final_part_id_gate")
            if isinstance(measurements, Mapping)
            else None
        )
        final_measurements = (
            final_part_id_gate.get("measurements")
            if isinstance(final_part_id_gate, Mapping)
            else None
        )
        part_id_views = (
            final_measurements.get("views")
            if isinstance(final_measurements, Mapping)
            else None
        )
        minimum_comparable_views = (
            _nonnegative_integer(policy.get("minimum_comparable_views"))
            if isinstance(policy, Mapping)
            else None
        )
        if (
            not isinstance(final_part_id_gate, Mapping)
            or final_part_id_gate.get("schema_version")
            != "asset-pipeline-part-id-quality-gate/v1"
            or final_part_id_gate.get("status") != "PASS"
            or final_part_id_gate.get("acceptance_allowed") is not True
            or final_part_id_gate.get("assignment_unit") != "part_id"
            or final_part_id_gate.get("raw_quality_status")
            != aggregate.get("status")
            or final_part_id_gate.get("effective_quality_status") != "PASS"
            or not isinstance(final_measurements, Mapping)
            or final_measurements.get("comparable_view_count")
            != comparable_view_count
            or final_measurements.get("scored_view_count")
            != comparable_view_count
            or final_measurements.get("aggregate_appearance_score")
            != aggregate.get("material_appearance_score")
            or not isinstance(part_id_views, list)
            or len(part_id_views) != comparable_view_count
            or minimum_comparable_views is None
            or comparable_view_count < minimum_comparable_views
        ):
            return "REJECTED", {}, None, None
        expected_part_id_views: dict[str, tuple[str, Any]] = {}
        for part_id_view in part_id_views:
            if not isinstance(part_id_view, Mapping):
                return "REJECTED", {}, None, None
            reference_id = _text(part_id_view.get("reference_view_id"))
            render_id = _text(part_id_view.get("render_view_id"))
            if (
                reference_id is None
                or render_id is None
                or reference_id in expected_part_id_views
                or status_by_reference_id.get(reference_id)
                not in {"PASS", "REVIEW"}
                or mapping.get(reference_id) != render_id
                or part_id_view.get("raw_status")
                != status_by_reference_id[reference_id]
                or part_id_view.get("material_appearance_score")
                != view_by_reference_id[reference_id].get(
                    "material_appearance_score"
                )
                or part_id_view.get("passes_appearance_floor") is not True
            ):
                return "REJECTED", {}, None, None
            expected_part_id_views[reference_id] = (
                render_id,
                part_id_view.get("material_appearance_score"),
            )
        if set(expected_part_id_views) != (
            set(passed_reference_ids) | set(review_reference_ids)
        ):
            return "REJECTED", {}, None, None
        limitations = final_part_id_gate.get("limitations")
        recorded_unscorable: list[str] = []
        if isinstance(limitations, list):
            for limitation in limitations:
                if (
                    isinstance(limitation, Mapping)
                    and limitation.get("code") == "UNSCORABLE_REFERENCE_VIEWS"
                    and isinstance(limitation.get("reference_view_ids"), list)
                ):
                    recorded_unscorable.extend(limitation["reference_view_ids"])
        if recorded_unscorable != unscorable_reference_ids:
            return "REJECTED", {}, None, None

    if immutable_library_optimum:
        gate_summary = gate.get("summary")
        gate_views = gate.get("views")
        if (
            not isinstance(gate_summary, Mapping)
            or gate_summary.get("view_count") != view_count
            or gate_summary.get("passed_view_count") != view_count
            or gate_summary.get("failure_count") != 0
            or not isinstance(gate_views, list)
            or len(gate_views) != view_count
        ):
            return "REJECTED", {}, None, None
        gate_mapping: dict[str, str] = {}
        for raw_gate_view in gate_views:
            if (
                not isinstance(raw_gate_view, Mapping)
                or raw_gate_view.get("status") != "PASS"
            ):
                return "REJECTED", {}, None, None
            reference_id = _text(raw_gate_view.get("reference_view_id"))
            render_id = _text(raw_gate_view.get("render_view_id"))
            if (
                reference_id is None
                or render_id is None
                or reference_id in gate_mapping
            ):
                return "REJECTED", {}, None, None
            gate_mapping[reference_id] = render_id
        if gate_mapping != mapping:
            return "REJECTED", {}, None, None

    render_set = registry.get("render_set")
    registry_views = (
        render_set.get("views") if isinstance(render_set, Mapping) else None
    )
    if not isinstance(registry_views, list) or not registry_views:
        return "REJECTED", {}, None, None
    images_by_render_id: dict[str, Path] = {}
    rgb_root = (collected_root / "renders" / "rgb").resolve()
    for raw_view in registry_views:
        if not isinstance(raw_view, Mapping):
            return "REJECTED", {}, None, None
        render_id = _text(raw_view.get("view_id"))
        image = _delivery_file(
            delivery,
            raw_view.get("rgb"),
            relative_to=registry_path.parent,
        )
        if render_id is None or image is None or render_id in images_by_render_id:
            return "REJECTED", {}, None, None
        try:
            image.relative_to(rgb_root)
        except ValueError:
            return "REJECTED", {}, None, None
        images_by_render_id[render_id] = image
    if set(images_by_render_id) != set(mapping.values()):
        return "REJECTED", {}, None, None
    if len(set(images_by_render_id.values())) != len(images_by_render_id):
        return "REJECTED", {}, None, None

    return (
        "PASS",
        {
            reference_id: _relative_href(
                delivery,
                images_by_render_id[mapping[reference_id]],
            )
            for reference_id in ordered_reference_ids
        },
        quality,
        quality_path.resolve(),
    )


def _legacy_final_preview_images(delivery: Path) -> dict[str, str]:
    """Expose one unambiguous historical final preview without claiming QA."""

    candidates = [delivery / "preview_final" / "rgb"]
    final_root = delivery / "final"
    if final_root.is_dir():
        candidates.extend(sorted(final_root.glob("*/preview_final/rgb")))
    rgb_dirs = sorted(
        {candidate.resolve() for candidate in candidates if candidate.is_dir()}
    )
    if len(rgb_dirs) != 1:
        return {}

    images: dict[str, str] = {}
    for image in sorted(rgb_dirs[0].glob("*.png")):
        resolved = _delivery_file(delivery, str(image))
        if resolved is not None:
            images.setdefault(image.stem, _relative_href(delivery, resolved))
    return images


def _restored_project_preview_images(
    delivery: Path,
    visual_material_root: Path,
    result: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Expose a sealed project preview only after its replay audits pass."""

    if (
        result is None
        or result.get("inference_mode") != "bundled_project"
        or result.get("visual_quality_status") != RESTORED_BASELINE
        or result.get("assignment_count") != result.get("applied_count")
    ):
        return {}
    project_audit_path = _recorded_artifact(
        delivery,
        result,
        "project_material_audit",
    )
    sealed_evidence_path = _recorded_artifact(
        delivery,
        result,
        "sealed_qwen_mvinverse_evidence",
    )
    delivery_validation_path = visual_material_root / "delivery_validation.json"
    project_audit = (
        _read_object(project_audit_path) if project_audit_path is not None else None
    )
    sealed_evidence = (
        _read_object(sealed_evidence_path) if sealed_evidence_path is not None else None
    )
    delivery_validation = _read_object(delivery_validation_path)
    if (
        project_audit is None
        or project_audit.get("status") != "PASS"
        or project_audit.get("complete_coverage") is not True
        or project_audit.get("topology_verified") is not True
        or project_audit.get("face_subsets_verified") is not True
        or sealed_evidence is None
        or sealed_evidence.get("live_inference_repeated") is not False
        or delivery_validation is None
        or delivery_validation.get("status") != "PASS"
        or delivery_validation.get("overall_pass") is not True
        or delivery_validation.get("failure_count") != 0
    ):
        return {}
    registry_path = _recorded_artifact(
        delivery,
        result,
        "preview_rendered_registry",
    ) or _recorded_artifact(
        delivery,
        result,
        "visual_quality_rendered_registry",
    )
    if registry_path is None:
        return {}
    preview_root = (visual_material_root / "preview_final").resolve()
    try:
        registry_path.relative_to(preview_root)
    except ValueError:
        return {}
    return _preview_images_from_registry(
        delivery,
        registry_path,
        accepted_root=preview_root,
    )


def _aggregate_status(quality_report: Mapping[str, Any] | None) -> str | None:
    if quality_report is None:
        return None
    aggregate = quality_report.get("aggregate")
    if not isinstance(aggregate, Mapping):
        return None
    return _text(aggregate.get("status"))


def _limitation_reason_label(reason_code: str | None) -> str:
    labels = {
        "POSE_OR_OCCLUSION_MISMATCH": "参考图姿态或遮挡关系与可见几何不一致",
        "CAMERA_OR_ASSEMBLY_COVERAGE_MISMATCH": (
            "材质已交付，剩余差异来自相机或装配姿态的几何覆盖"
        ),
    }
    return labels.get(reason_code or "", reason_code or "未记录原因")


def _limitation_classification_label(classification: str | None) -> str:
    labels = {
        "NOT_OBSERVABLE_GEOMETRY_POSE": (
            "当前几何与相机姿态下不可观测，不能用改材质伪装修复"
        ),
        "OBSERVABLE_GEOMETRY_COVERAGE_MISMATCH": (
            "目标材质可见但投影覆盖小于参考图，继续改材质不能补全几何"
        ),
    }
    return labels.get(
        classification or "",
        classification or "未记录分类",
    )


def _percentage(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not 0.0 <= float(value) <= 1.0:
        return None
    return f"{float(value) * 100.0:.1f}%"


def _limitation_summary(raw: Mapping[str, Any]) -> dict[str, Any]:
    reason_code = _text(raw.get("reason_code"))
    classification = _text(raw.get("classification"))
    limitation_lane = _text(raw.get("limitation_lane"))
    view_id = _text(raw.get("reference_view_id"))
    group_id = _text(raw.get("canonical_group_id"))

    evidence = raw.get("reference_group_evidence")
    evidence_pixels = None
    if isinstance(evidence, Mapping):
        evidence_pixels = _nonnegative_integer(evidence.get("evidence_pixels"))

    owner = raw.get("foreign_owner")
    owner_part_id = None
    overlap_share = None
    if isinstance(owner, Mapping):
        owner_part_id = _text(owner.get("part_id"))
        overlap = owner.get("accepted_box_overlap")
        if isinstance(overlap, Mapping):
            value = overlap.get("projected_overlap_share")
            if (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and 0.0 <= float(value) <= 1.0
            ):
                overlap_share = float(value)

    subject = " / ".join(
        value
        for value in (
            f"视角 {view_id}" if view_id else None,
            f"颜色组 {group_id}" if group_id else None,
        )
        if value is not None
    )
    details = [_limitation_reason_label(reason_code)]
    lane_labels = {
        "source_bound_zero_visible_repeated_geometry": (
            "源绑定重复件在目标姿态零可见"
        ),
        "source_bound_visible_repeated_geometry": (
            "源绑定重复件可见但投影覆盖不足"
        ),
        "cross_view_material_delivery": (
            "材质已在其他参考视角交付"
        ),
    }
    if limitation_lane:
        details.append(lane_labels.get(limitation_lane, limitation_lane))
    if owner_part_id:
        owner_detail = f"参考证据主要落在 {owner_part_id} 的投影上"
        overlap_label = _percentage(overlap_share)
        if overlap_label:
            owner_detail += f"（重叠 {overlap_label}）"
        details.append(owner_detail)
    if evidence_pixels is not None:
        details.append(f"证据像素 {evidence_pixels:,}")

    return {
        "reason_code": reason_code,
        "reason_label_zh": _limitation_reason_label(reason_code),
        "classification": classification,
        "classification_label_zh": _limitation_classification_label(classification),
        "limitation_lane": limitation_lane,
        "reference_view_id": view_id,
        "canonical_group_id": group_id,
        "evidence_pixels": evidence_pixels,
        "foreign_owner_part_id": owner_part_id,
        "projected_overlap_share": overlap_share,
        "summary_zh": f"{subject + '：' if subject else ''}{'；'.join(details)}。",
    }


def build_viewer_manifest(delivery_dir: Path) -> dict[str, Any]:
    """Normalize one selected delivery into a small browser-facing manifest."""

    delivery = delivery_dir.expanduser().resolve(strict=True)
    if not delivery.is_dir():
        raise ValueError(f"delivery is not a directory: {delivery}")
    visual_material_root = _visual_material_root(delivery)
    result, result_path = _locate_pipeline_result(delivery)
    resolution, resolution_path = _locate_resolution(
        delivery,
        visual_material_root,
        result,
    )
    (
        final_collected_state,
        final_collected_images,
        final_collected_quality,
        final_collected_quality_path,
    ) = _final_collected_preview_images(
        delivery,
        visual_material_root,
    )
    if final_collected_state == "PASS":
        quality_report = final_collected_quality
        quality_report_path = final_collected_quality_path
        preview_images = final_collected_images
    else:
        quality_report, quality_report_path = _locate_quality_report(
            delivery,
            visual_material_root,
            result,
        )
        preview_images = (
            _preview_images(delivery, result, quality_report_path)
            if final_collected_state == "ABSENT"
            else {}
        )
    restored_project_preview = False
    if (
        final_collected_state == "PASS"
        and result is not None
        and result.get("inference_mode") == "bundled_project"
        and result.get("visual_quality_status") == RESTORED_BASELINE
    ):
        restored_project_preview = True
    elif (
        final_collected_state == "ABSENT"
        and not preview_images
        and quality_report_path is None
    ):
        preview_images = _restored_project_preview_images(
            delivery,
            visual_material_root,
            result,
        )
        restored_project_preview = bool(preview_images)
    legacy_final_preview = False
    if (
        final_collected_state == "ABSENT"
        and not preview_images
        and quality_report_path is None
    ):
        preview_images = _legacy_final_preview_images(delivery)
        legacy_final_preview = bool(preview_images)
    appearance_status = (
        _text(result.get("appearance_optimization_status"))
        if result is not None
        else None
    )

    legacy_status = (
        _text(result.get("visual_quality_status")) if result is not None else None
    )
    resolution_raw = (
        _text(resolution.get("raw_quality_status")) if resolution is not None else None
    )
    resolution_gate = (
        _text(resolution.get("resolution_status")) if resolution is not None else None
    )
    if final_collected_state == "PASS" and not restored_project_preview:
        # The independent, hash-bound final acceptance supersedes an earlier
        # selection/repair report.  Immutable-library optimum runs can retain
        # raw REVIEW while their constrained final gate is PASS.
        raw_status = _aggregate_status(quality_report)
        gate_status = "PASS"
    else:
        raw_status = (
            (
                _text(result.get("visual_quality_raw_status"))
                if result is not None
                else None
            )
            or resolution_raw
            or legacy_status
            or _aggregate_status(quality_report)
        )
        gate_status = (
            (
                _text(result.get("visual_quality_gate_status"))
                if result is not None
                else None
            )
            or resolution_gate
            or legacy_status
            or raw_status
        )

    raw_limitations = resolution.get("limitations") if resolution is not None else None
    limitations = (
        [dict(item) for item in raw_limitations if isinstance(item, Mapping)]
        if isinstance(raw_limitations, list)
        else []
    )
    result_limitation_count = (
        _nonnegative_integer(result.get("visual_quality_limitation_count"))
        if result is not None
        else None
    )
    limitation_count = (
        result_limitation_count
        if result_limitation_count is not None
        else len(limitations)
    )

    new_fields_present = resolution is not None or bool(
        result is not None
        and any(
            field in result
            for field in (
                "visual_quality_raw_status",
                "visual_quality_gate_status",
                "visual_quality_resolution",
                "visual_quality_limitation_count",
            )
        )
    )
    resolution_recorded = bool(
        result is not None
        and _text(result.get("visual_quality_resolution")) is not None
    )
    if restored_project_preview:
        resolution_state = "SEALED_HISTORICAL_BASELINE"
    elif resolution_path is not None:
        resolution_state = "AVAILABLE"
    elif resolution_recorded:
        resolution_state = "RECORDED_NOT_AVAILABLE"
    elif new_fields_present and raw_status == "PASS":
        resolution_state = "NOT_REQUIRED"
    elif not new_fields_present:
        resolution_state = "LEGACY_NOT_RECORDED"
    else:
        resolution_state = "NOT_AVAILABLE"

    warnings: list[str] = []
    if final_collected_state == "REJECTED":
        warnings.append(
            "检测到最终 collected 验收目录，但 gate、全视图 PASS、映射或哈希合同"
            "不完整；页面已拒绝展示候选/旧预览。"
        )
    if (
        resolution is not None
        and result_limitation_count is not None
        and result_limitation_count != len(limitations)
    ):
        warnings.append(
            "结果中的 visual_quality_limitation_count 与解析报告的 "
            "limitations 数量不一致。"
        )
    if (
        final_collected_state != "PASS"
        and resolution is not None
        and gate_status is not None
        and resolution_gate is not None
        and gate_status != resolution_gate
    ):
        warnings.append("结果中的 visual_quality_gate_status 与解析报告状态不一致。")

    accepted = gate_status in {"PASS", LIMITED_PASS, RESTORED_BASELINE}
    if gate_status == RESTORED_BASELINE and restored_project_preview:
        note = (
            "首版材质结果已按 CAD、四张参考图、Mesh 路径、拓扑和面级子集哈希"
            "完成封存恢复；本次 collected final 的 aggregate 和四个视图"
            "均为 PASS。"
        )
    elif gate_status == LIMITED_PASS:
        note = (
            f"材质门禁已接受，但保留 {limitation_count} 项经证据约束的"
            "几何姿态/遮挡限制；原始质检状态没有被改写。"
        )
    elif gate_status == "PASS":
        note = "材质视觉门禁通过，未记录几何姿态/遮挡限制。"
    elif gate_status:
        note = "材质视觉门禁未通过；该结果不应被解释为无人验收成功。"
    else:
        note = "未找到材质视觉质量结果。"
    if legacy_final_preview:
        note += " 已加载历史最终预览；它仅用于结果回看，不代表通过新版材质 QA。"
    if not new_fields_present and legacy_status and not restored_project_preview:
        note += " 此交付使用旧字段，页面已按 visual_quality_status 兼容展示。"

    return {
        "schema_version": SCHEMA_VERSION,
        "appearance_optimization_status": appearance_status,
        "visual_quality_raw_status": raw_status,
        "visual_quality_gate_status": gate_status,
        "visual_quality_resolution": {
            "state": resolution_state,
            "href": (
                _relative_href(delivery, resolution_path)
                if resolution_path is not None
                else None
            ),
        },
        "visual_quality_limitation_count": limitation_count,
        "material_stage_accepted": accepted,
        "limitation_reasons": [_limitation_summary(item) for item in limitations],
        "note_zh": note,
        "warnings": warnings,
        "source": {
            "legacy_compatibility": (
                not new_fields_present and not restored_project_preview
            ),
            "restored_project": restored_project_preview,
            "pipeline_result": (
                _relative_href(delivery, result_path)
                if result_path is not None
                else None
            ),
            "quality_report": (
                _relative_href(delivery, quality_report_path)
                if quality_report_path is not None
                else None
            ),
            "legacy_final_preview": legacy_final_preview,
            "final_collected_acceptance": final_collected_state,
            "preview_fallback_allowed": final_collected_state == "ABSENT",
            "preview_images": preview_images,
            "resolution_report": (
                _relative_href(delivery, resolution_path)
                if resolution_path is not None
                else None
            ),
        },
    }


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the local result-viewer manifest."
    )
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    _write_manifest(args.output, build_viewer_manifest(args.delivery))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
