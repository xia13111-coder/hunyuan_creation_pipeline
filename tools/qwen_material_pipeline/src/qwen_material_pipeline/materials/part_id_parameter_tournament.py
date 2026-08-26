"""Render-and-compare selection for Part-ID H0/H1 parameter candidates.

H0 is always the untouched selected NVIDIA MDL. H1, when present, changes
only a color input that already passed the Part-ID evidence gate. This module
never chooses by material name, Part ID, or named color; the registered render
of the actual CAD part is the selection authority.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


SCHEMA_VERSION = "qwen-part-id-parameter-tournament/v1"
CANDIDATE_SCHEMA_VERSION = "qwen-part-id-parameter-candidates/v1"


class PartIdParameterTournamentError(ValueError):
    """Raised when an H0/H1 render tournament violates its contract."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_sealed_file(
    path: str | Path,
    label: str,
    expected_sha256: object = None,
) -> Path:
    """Resolve a sealed artifact after a filename-only repository rename.

    Evidence documents intentionally retain their original paths.  When that
    exact name has disappeared, accept a replacement only when exactly one
    regular, non-symlink file in the same directory has the sealed byte hash.
    This keeps old evidence reproducible without guessing from view names or
    introducing asset-specific path aliases.
    """

    candidate = Path(path).expanduser()
    digest = expected_sha256 if isinstance(expected_sha256, str) else None
    if digest is not None and (
        len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise PartIdParameterTournamentError(f"{label} has an invalid SHA-256")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        if digest is None:
            raise PartIdParameterTournamentError(
                f"unable to resolve {label}: {candidate}"
            ) from exc
        try:
            parent = candidate.parent.resolve(strict=True)
        except FileNotFoundError as parent_exc:
            raise PartIdParameterTournamentError(
                f"unable to resolve {label}: {candidate}"
            ) from parent_exc
        matches = []
        for sibling in parent.iterdir():
            if sibling.is_symlink() or not sibling.is_file():
                continue
            if _sha256_file(sibling) == digest:
                matches.append(sibling.resolve(strict=True))
        if len(matches) != 1:
            raise PartIdParameterTournamentError(
                f"unable to relocate {label} by sealed SHA-256: "
                f"expected one sibling match, found {len(matches)}"
            ) from exc
        resolved = matches[0]
    if candidate.is_symlink() or not resolved.is_file():
        raise PartIdParameterTournamentError(
            f"{label} must be a regular non-symlink file: {candidate}"
        )
    if digest is not None and _sha256_file(resolved) != digest:
        raise PartIdParameterTournamentError(f"{label} failed its sealed SHA-256")
    return resolved


def _read_rgb(
    path: str | Path,
    label: str,
    expected_sha256: object = None,
) -> np.ndarray:
    resolved = _resolve_sealed_file(path, label, expected_sha256)
    try:
        with Image.open(resolved) as opened:
            return np.asarray(
                ImageOps.exif_transpose(opened).convert("RGB"),
                dtype=np.uint8,
            )
    except OSError as exc:
        raise PartIdParameterTournamentError(
            f"unable to read {label}: {resolved}: {exc}"
        ) from exc


def _read_mask(
    path: str | Path,
    label: str,
    expected_sha256: object = None,
) -> np.ndarray:
    resolved = _resolve_sealed_file(path, label, expected_sha256)
    try:
        with Image.open(resolved) as opened:
            array = np.asarray(
                ImageOps.exif_transpose(opened).convert("L"),
                dtype=np.uint8,
            )
    except OSError as exc:
        raise PartIdParameterTournamentError(
            f"unable to read {label}: {resolved}: {exc}"
        ) from exc
    return (array >= 128).astype(np.uint8) * 255


def _assignment_by_part(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = plan.get("assignments")
    if not isinstance(rows, list) or not rows:
        raise PartIdParameterTournamentError("material plan has no assignments")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        part_id = row.get("part_id") if isinstance(row, Mapping) else None
        if not isinstance(part_id, str) or not part_id or part_id in result:
            raise PartIdParameterTournamentError(
                "material plan has invalid or duplicate Part IDs"
            )
        result[part_id] = row
    return result


def pending_h1_part_ids(plan: Mapping[str, Any]) -> list[str]:
    """Return the exact Part IDs that have one unapplied H1 candidate."""

    pending: list[str] = []
    for part_id, assignment in _assignment_by_part(plan).items():
        provenance = assignment.get("provenance")
        candidate_set = (
            provenance.get("mdl_parameter_candidates")
            if isinstance(provenance, Mapping)
            else None
        )
        if not isinstance(candidate_set, Mapping):
            continue
        if candidate_set.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
            raise PartIdParameterTournamentError(
                f"{part_id} has an unsupported parameter candidate schema"
            )
        candidates = candidate_set.get("candidates")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise PartIdParameterTournamentError(
                f"{part_id} parameter candidates are invalid"
            )
        ids = [
            row.get("candidate_id") for row in candidates if isinstance(row, Mapping)
        ]
        if ids == ["H0", "H1"]:
            pending.append(part_id)
        elif ids != ["H0"]:
            raise PartIdParameterTournamentError(
                f"{part_id} candidates must be H0 or H0/H1"
            )
    return sorted(pending)


def build_h1_candidate_plan(
    *,
    source_plan: Mapping[str, Any],
    part_id: str,
) -> dict[str, Any]:
    """Apply exactly one pending H1 to a deep-copied candidate plan."""

    output = copy.deepcopy(dict(source_plan))
    assignments = _assignment_by_part(output)
    if part_id not in assignments:
        raise PartIdParameterTournamentError(f"unknown Part ID: {part_id}")
    assignment = assignments[part_id]
    provenance = assignment.get("provenance")
    candidate_set = (
        provenance.get("mdl_parameter_candidates")
        if isinstance(provenance, Mapping)
        else None
    )
    candidates = (
        candidate_set.get("candidates") if isinstance(candidate_set, Mapping) else None
    )
    h1 = next(
        (
            row
            for row in candidates or []
            if isinstance(row, Mapping) and row.get("candidate_id") == "H1"
        ),
        None,
    )
    if not isinstance(h1, Mapping):
        raise PartIdParameterTournamentError(
            f"Part ID {part_id} has no pending H1 candidate"
        )
    if h1.get("material_id") != assignment.get("material_id"):
        raise PartIdParameterTournamentError(
            f"Part ID {part_id} H1 material does not match its selected MDL"
        )
    parameters = h1.get("parameters")
    if not isinstance(parameters, Mapping) or not parameters:
        raise PartIdParameterTournamentError(
            f"Part ID {part_id} H1 has no color parameters"
        )
    assignment["parameters"] = copy.deepcopy(dict(parameters))
    mutable_provenance = dict(provenance)
    mutable_candidate_set = copy.deepcopy(dict(candidate_set))
    mutable_candidate_set["selection_status"] = "CANDIDATE_RENDER"
    mutable_candidate_set["selected_candidate_id"] = "H1"
    mutable_candidate_set["parameters_applied_to_plan"] = True
    mutable_provenance["mdl_parameter_candidates"] = mutable_candidate_set
    mutable_provenance["parameter_tournament_candidate"] = {
        "part_id": part_id,
        "candidate_id": "H1",
        "source_plan_sha256": _canonical_sha256(source_plan),
    }
    assignment["provenance"] = mutable_provenance
    output_provenance = output.get("provenance")
    if not isinstance(output_provenance, dict):
        output_provenance = {}
        output["provenance"] = output_provenance
    output_provenance["part_id_parameter_tournament_candidate"] = {
        "part_id": part_id,
        "candidate_id": "H1",
        "source_plan_sha256": _canonical_sha256(source_plan),
    }
    return output


def _selected_observation(
    evidence: Mapping[str, Any],
    part_id: str,
) -> Mapping[str, Any]:
    parts = evidence.get("parts")
    if not isinstance(parts, list):
        raise PartIdParameterTournamentError("Part-ID evidence has no parts")
    part = next(
        (
            row
            for row in parts
            if isinstance(row, Mapping) and row.get("part_id") == part_id
        ),
        None,
    )
    if not isinstance(part, Mapping):
        raise PartIdParameterTournamentError(
            f"Part-ID evidence does not cover {part_id}"
        )
    observations = part.get("observations")
    selected = [
        row
        for row in observations or []
        if isinstance(row, Mapping)
        and row.get("selected_for_material_inference") is True
    ]
    if len(selected) != 1:
        raise PartIdParameterTournamentError(
            f"Part ID {part_id} needs exactly one selected observation"
        )
    return selected[0]


def _alignment(
    spatial_mapping_report: Mapping[str, Any],
    reference_view_id: str,
    render_view_id: str,
) -> Mapping[str, Any]:
    rows = spatial_mapping_report.get("view_alignments")
    matches = [
        row
        for row in rows or []
        if isinstance(row, Mapping)
        and row.get("reference_view_id") == reference_view_id
        and row.get("selected_render_view_id") == render_view_id
        and row.get("trusted") is True
    ]
    if len(matches) != 1:
        raise PartIdParameterTournamentError(
            f"missing trusted registration for {reference_view_id}/{render_view_id}"
        )
    return matches[0]


def _render_rgb(
    rendered_registry: Mapping[str, Any],
    render_view_id: str,
) -> np.ndarray:
    render_set = rendered_registry.get("render_set")
    views = render_set.get("views") if isinstance(render_set, Mapping) else None
    matches = [
        row
        for row in views or []
        if isinstance(row, Mapping) and row.get("view_id") == render_view_id
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("rgb"), str):
        raise PartIdParameterTournamentError(
            f"render registry has no unique RGB view {render_view_id}"
        )
    return _read_rgb(matches[0]["rgb"], f"render RGB {render_view_id}")


def _warp_render_to_reference(
    render_rgb: np.ndarray,
    *,
    alignment: Mapping[str, Any],
    output_shape: tuple[int, int],
) -> np.ndarray:
    bbox_affine = np.asarray(alignment.get("bbox_affine"), dtype=np.float32)
    ecc_warp = np.asarray(alignment.get("ecc_warp"), dtype=np.float32)
    if bbox_affine.shape != (2, 3) or ecc_warp.shape != (2, 3):
        raise PartIdParameterTournamentError(
            "spatial registration has invalid affine matrices"
        )
    output_size = (int(output_shape[1]), int(output_shape[0]))
    normalized = cv2.warpAffine(
        render_rgb,
        bbox_affine,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return cv2.warpAffine(
        normalized,
        ecc_warp,
        output_size,
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _lab_medoid(samples: np.ndarray) -> tuple[np.ndarray, float, float]:
    normalized = samples.astype(np.float32) / 255.0
    lab = cv2.cvtColor(
        normalized.reshape(1, -1, 3),
        cv2.COLOR_RGB2LAB,
    ).reshape(-1, 3)
    coordinate_median = np.median(lab, axis=0)
    medoid = lab[int(np.argmin(np.linalg.norm(lab - coordinate_median, axis=1)))]
    distances = np.linalg.norm(lab - medoid, axis=1)
    return medoid, float(np.median(distances)), float(np.quantile(distances, 0.9))


def score_part_id_render(
    *,
    part_id: str,
    evidence: Mapping[str, Any],
    spatial_mapping_report: Mapping[str, Any],
    rendered_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Score one actual CAD render inside its registered reference mask."""

    observation = _selected_observation(evidence, part_id)
    reference_view_id = observation.get("view_id")
    render_view_id = observation.get("render_view_id")
    if not isinstance(reference_view_id, str) or not isinstance(render_view_id, str):
        raise PartIdParameterTournamentError(
            f"Part ID {part_id} observation has invalid view IDs"
        )
    reference = _read_rgb(
        observation.get("image"),
        f"Part ID {part_id} reference",
        observation.get("image_sha256"),
    )
    mask = _read_mask(
        observation.get("mask"),
        f"Part ID {part_id} reference mask",
        observation.get("mask_sha256"),
    )
    if reference.shape[:2] != mask.shape:
        raise PartIdParameterTournamentError(
            f"Part ID {part_id} reference and mask dimensions differ"
        )
    rendered = _render_rgb(rendered_registry, render_view_id)
    registered = _warp_render_to_reference(
        rendered,
        alignment=_alignment(
            spatial_mapping_report,
            reference_view_id,
            render_view_id,
        ),
        output_shape=mask.shape,
    )
    selected = mask > 0
    pixel_count = int(np.count_nonzero(selected))
    chromatic_coverage = observation.get("chromatic_coverage")
    tiny_chromatic_rescue = (
        isinstance(chromatic_coverage, Mapping)
        and chromatic_coverage.get("applied") is True
        and chromatic_coverage.get("tiny_part_rescue") is True
    )
    minimum_comparison_pixels = 6 if tiny_chromatic_rescue else 8
    if pixel_count < minimum_comparison_pixels:
        raise PartIdParameterTournamentError(
            f"Part ID {part_id} has insufficient registered comparison pixels"
        )
    reference_samples = reference[selected]
    render_samples = registered[selected]
    reference_medoid, reference_median_spread, reference_p90_spread = _lab_medoid(
        reference_samples
    )
    render_medoid, render_median_spread, render_p90_spread = _lab_medoid(render_samples)
    delta_e = float(np.linalg.norm(reference_medoid - render_medoid))
    color_score = math.exp(-delta_e / 30.0)
    reference_luma = float(np.median(reference_medoid[0]))
    render_luma = float(np.median(render_medoid[0]))
    luma_score = math.exp(-abs(reference_luma - render_luma) / 25.0)
    appearance_score = 0.80 * color_score + 0.20 * luma_score
    return {
        "schema_version": "qwen-part-id-registered-render-score/v1",
        "part_id": part_id,
        "reference_view_id": reference_view_id,
        "render_view_id": render_view_id,
        "comparison_pixel_count": pixel_count,
        "minimum_comparison_pixels": minimum_comparison_pixels,
        "tiny_chromatic_rescue": tiny_chromatic_rescue,
        "lab_delta_e": round(delta_e, 8),
        "color_score": round(color_score, 8),
        "luma_score": round(luma_score, 8),
        "appearance_score": round(appearance_score, 8),
        "reference_lab_medoid": [round(float(value), 8) for value in reference_medoid],
        "render_lab_medoid": [round(float(value), 8) for value in render_medoid],
        "reference_median_delta_e": round(reference_median_spread, 8),
        "reference_p90_delta_e": round(reference_p90_spread, 8),
        "render_median_delta_e": round(render_median_spread, 8),
        "render_p90_delta_e": round(render_p90_spread, 8),
    }


def select_parameter_tournament_winners(
    *,
    source_plan: Mapping[str, Any],
    baseline_scores: Mapping[str, Mapping[str, Any]],
    h1_scores: Mapping[str, Mapping[str, Any]],
    minimum_score_improvement: float = 0.015,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply H1 only when its actual-part render clearly improves over H0."""

    if (
        isinstance(minimum_score_improvement, bool)
        or not isinstance(minimum_score_improvement, (int, float))
        or not math.isfinite(float(minimum_score_improvement))
        or not 0.0 <= float(minimum_score_improvement) <= 1.0
    ):
        raise PartIdParameterTournamentError(
            "minimum_score_improvement must be a finite unit float"
        )
    expected = set(pending_h1_part_ids(source_plan))
    if set(baseline_scores) != expected or set(h1_scores) != expected:
        raise PartIdParameterTournamentError(
            "H0/H1 scores must exactly cover pending Part IDs"
        )
    output = copy.deepcopy(dict(source_plan))
    assignments = _assignment_by_part(output)
    records: list[dict[str, Any]] = []
    h1_winners: list[str] = []
    for part_id in sorted(expected):
        h0_score = float(baseline_scores[part_id].get("appearance_score", -1.0))
        h1_score = float(h1_scores[part_id].get("appearance_score", -1.0))
        if not 0.0 <= h0_score <= 1.0 or not 0.0 <= h1_score <= 1.0:
            raise PartIdParameterTournamentError(
                f"Part ID {part_id} has invalid render scores"
            )
        improvement = h1_score - h0_score
        winner = "H1" if improvement >= float(minimum_score_improvement) else "H0"
        assignment = assignments[part_id]
        provenance = assignment.get("provenance")
        if not isinstance(provenance, dict):
            raise PartIdParameterTournamentError(
                f"Part ID {part_id} has invalid provenance"
            )
        candidate_set = copy.deepcopy(dict(provenance["mdl_parameter_candidates"]))
        h1 = next(
            row for row in candidate_set["candidates"] if row["candidate_id"] == "H1"
        )
        candidate_set["selection_status"] = "LOCKED_AFTER_RENDER_COMPARISON"
        candidate_set["selected_candidate_id"] = winner
        candidate_set["parameters_applied_to_plan"] = winner == "H1"
        candidate_set["render_comparison"] = {
            "minimum_score_improvement": float(minimum_score_improvement),
            "observed_score_improvement": round(improvement, 8),
            "h0": copy.deepcopy(dict(baseline_scores[part_id])),
            "h1": copy.deepcopy(dict(h1_scores[part_id])),
        }
        provenance["mdl_parameter_candidates"] = candidate_set
        provenance["part_id_parameter_tournament"] = {
            "winner": winner,
            "score_improvement": round(improvement, 8),
            "minimum_score_improvement": float(minimum_score_improvement),
        }
        color_audit = provenance.get("mdl_color_parameterization")
        if isinstance(color_audit, dict):
            color_audit["selected_candidate_id"] = winner
            color_audit["parameters_applied"] = winner == "H1"
            color_audit["status"] = (
                "render_selected_h1_applied"
                if winner == "H1"
                else "native_h0_selected_after_render_comparison"
            )
        if winner == "H1":
            assignment["parameters"] = copy.deepcopy(dict(h1["parameters"]))
            h1_winners.append(part_id)
        else:
            assignment.pop("parameters", None)
        records.append(
            {
                "part_id": part_id,
                "winner": winner,
                "h0_appearance_score": h0_score,
                "h1_appearance_score": h1_score,
                "score_improvement": round(improvement, 8),
            }
        )
    output_provenance = output.get("provenance")
    if not isinstance(output_provenance, dict):
        output_provenance = {}
        output["provenance"] = output_provenance
    output_provenance["part_id_parameter_tournament"] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "minimum_score_improvement": float(minimum_score_improvement),
        "candidate_part_count": len(expected),
        "h1_winner_count": len(h1_winners),
        "h1_winner_part_ids": h1_winners,
    }
    audit_unsigned = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "source_plan_sha256": _canonical_sha256(source_plan),
        "output_plan_sha256": _canonical_sha256(output),
        "minimum_score_improvement": float(minimum_score_improvement),
        "candidate_part_count": len(expected),
        "h0_winner_count": len(expected) - len(h1_winners),
        "h1_winner_count": len(h1_winners),
        "h1_winner_part_ids": h1_winners,
        "records": records,
    }
    audit = {
        **audit_unsigned,
        "integrity": {"document_sha256": _canonical_sha256(audit_unsigned)},
    }
    return output, audit


def rebind_part_id_material_audit(
    *,
    source_audit: Mapping[str, Any],
    final_plan: Mapping[str, Any],
    tournament_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash-bind the existing exact-cover audit to tournament parameter deltas."""

    output = copy.deepcopy(dict(source_audit))
    rows = output.get("parts")
    if not isinstance(rows, list):
        raise PartIdParameterTournamentError("Part-ID material audit has no parts")
    assignments = _assignment_by_part(final_plan)
    audit_by_part = {
        row.get("part_id"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("part_id"), str)
    }
    if set(audit_by_part) != set(assignments):
        raise PartIdParameterTournamentError(
            "Part-ID material audit does not exactly cover the final plan"
        )
    for part_id, assignment in assignments.items():
        if audit_by_part[part_id].get("status") != "independently_selected":
            continue
        provenance = assignment.get("provenance")
        if not isinstance(provenance, Mapping):
            raise PartIdParameterTournamentError(
                f"Part ID {part_id} final provenance is invalid"
            )
        for field in (
            "mdl_parameter_candidates",
            "mdl_color_parameterization",
        ):
            if field in provenance:
                audit_by_part[part_id][field] = copy.deepcopy(provenance[field])
    summary = output.get("summary")
    if not isinstance(summary, dict):
        raise PartIdParameterTournamentError("Part-ID material audit has no summary")
    summary["color_parameterized_count"] = sum(
        isinstance(assignment.get("parameters"), Mapping)
        and bool(assignment.get("parameters"))
        for assignment in assignments.values()
    )
    summary["native_h0_selected_count"] = sum(
        (
            assignment.get("provenance", {})
            .get("mdl_parameter_candidates", {})
            .get("selected_candidate_id")
            == "H0"
        )
        for assignment in assignments.values()
        if assignment.get("status") in {"auto", "review"}
    )
    summary["h1_render_selected_count"] = sum(
        (
            assignment.get("provenance", {})
            .get("mdl_parameter_candidates", {})
            .get("selected_candidate_id")
            == "H1"
        )
        for assignment in assignments.values()
    )
    output["output_plan_sha256"] = _canonical_sha256(final_plan)
    output["part_id_parameter_tournament"] = {
        "schema_version": SCHEMA_VERSION,
        "audit_sha256": _canonical_sha256(tournament_audit),
        "h1_winner_count": tournament_audit.get("h1_winner_count"),
        "h1_winner_part_ids": copy.deepcopy(tournament_audit.get("h1_winner_part_ids")),
    }
    output.pop("integrity", None)
    output["integrity"] = {"document_sha256": _canonical_sha256(output)}
    return output


__all__ = [
    "PartIdParameterTournamentError",
    "build_h1_candidate_plan",
    "pending_h1_part_ids",
    "rebind_part_id_material_audit",
    "score_part_id_render",
    "select_parameter_tournament_winners",
]
