"""Automatic per-scope colour calibration from registered actual-CAD renders.

The controller never changes an MDL identity.  It observes the luminance
response of every sealed colour scope, proposes a bounded continuous gain for
that scope, and updates all scopes together in one subsequent CAD render.
Historical candidates remain the authority for the final appearance-score
selection, so a poor proposal cannot replace an earlier better result.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .component_mdl_tournament import score_component_render
from .corresponding_material_color_selection import Candidate
from .part_id_parameter_tournament import score_part_id_render


SCHEMA_VERSION = "qwen-adaptive-corresponding-material-color/v1"
INITIAL_GAIN = 1.0
MINIMUM_GAIN = 0.1
MAXIMUM_GAIN = 8.0
LUMINANCE_CORRECTION_DAMPING = 0.75
MAXIMUM_STEP_RATIO = 2.0
MINIMUM_RESPONSE_SLOPE = 0.15
MAXIMUM_RESPONSE_SLOPE = 3.0
RELATIVE_LUMINANCE_TOLERANCE = 0.03
RELATIVE_GAIN_TOLERANCE = 0.02
MINIMUM_APPEARANCE_IMPROVEMENT = 0.003
STALL_PATIENCE = 2


class AdaptiveCorrespondingMaterialColorError(ValueError):
    """Raised when adaptive colour evidence violates the sealed contract."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdaptiveCorrespondingMaterialColorError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise AdaptiveCorrespondingMaterialColorError(f"{label} must be an array")
    return value


def _lab_l_to_relative_luminance(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise AdaptiveCorrespondingMaterialColorError(
            "registered Lab luminance must be finite"
        )
    lightness = min(100.0, max(0.0, float(value)))
    if lightness > 8.0:
        return ((lightness + 16.0) / 116.0) ** 3
    return lightness / 903.3


def _score_luminance(score: Mapping[str, Any]) -> tuple[float, float]:
    member_scores = score.get("member_scores")
    rows = (
        _array(member_scores, "score.member_scores")
        if member_scores is not None
        else [score]
    )
    weighted_reference = 0.0
    weighted_render = 0.0
    total_pixels = 0
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"score row {index}")
        pixels = row.get("comparison_pixel_count")
        reference_lab = row.get("reference_lab_medoid")
        render_lab = row.get("render_lab_medoid")
        if (
            isinstance(pixels, bool)
            or not isinstance(pixels, int)
            or pixels <= 0
            or not isinstance(reference_lab, Sequence)
            or isinstance(reference_lab, (str, bytes))
            or len(reference_lab) != 3
            or not isinstance(render_lab, Sequence)
            or isinstance(render_lab, (str, bytes))
            or len(render_lab) != 3
        ):
            raise AdaptiveCorrespondingMaterialColorError(
                "registered score lacks valid luminance evidence"
            )
        weighted_reference += _lab_l_to_relative_luminance(reference_lab[0]) * pixels
        weighted_render += _lab_l_to_relative_luminance(render_lab[0]) * pixels
        total_pixels += pixels
    if total_pixels <= 0:
        raise AdaptiveCorrespondingMaterialColorError(
            "registered score has no comparison pixels"
        )
    return weighted_reference / total_pixels, weighted_render / total_pixels


def score_adaptive_candidate_scopes(
    *,
    candidate: Candidate,
    part_id_evidence: Mapping[str, Any],
    spatial_mapping_report: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Score every sealed scope and expose its measured luminance response."""

    scopes: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(
        _array(candidate.audit.get("scopes"), "candidate.audit.scopes")
    ):
        scope = _mapping(raw, f"candidate.audit.scopes[{index}]")
        scope_id = scope.get("scope_id")
        if not isinstance(scope_id, str) or not scope_id or scope_id in scopes:
            raise AdaptiveCorrespondingMaterialColorError(
                "candidate contains an invalid or duplicate colour scope"
            )
        scopes[scope_id] = scope
    if set(scopes) != set(candidate.gains_by_scope):
        raise AdaptiveCorrespondingMaterialColorError(
            "candidate gain vector does not cover its sealed scopes"
        )

    output: dict[str, dict[str, Any]] = {}
    for scope_id, scope in sorted(scopes.items()):
        raw_members = _array(scope.get("member_part_ids"), f"{scope_id}.members")
        members = sorted(
            str(value) for value in raw_members if isinstance(value, str) and value
        )
        if (
            len(members) != len(raw_members)
            or len(set(members)) != len(members)
            or not members
        ):
            raise AdaptiveCorrespondingMaterialColorError(
                f"scope {scope_id} has invalid members"
            )
        if len(members) == 1:
            score = score_part_id_render(
                part_id=members[0],
                evidence=part_id_evidence,
                spatial_mapping_report=spatial_mapping_report,
                rendered_registry=candidate.rendered_registry,
            )
        else:
            score = score_component_render(
                component_id=scope_id,
                member_part_ids=members,
                evidence=part_id_evidence,
                spatial_mapping_report=spatial_mapping_report,
                rendered_registry=candidate.rendered_registry,
            )
        reference_luminance, render_luminance = _score_luminance(score)
        safe_reference = max(reference_luminance, 1e-6)
        safe_render = max(render_luminance, 1e-6)
        output[scope_id] = {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": candidate.candidate_id,
            "scope_id": scope_id,
            "member_part_ids": members,
            "linear_intensity_gain": candidate.gains_by_scope[scope_id],
            "reference_relative_luminance": round(reference_luminance, 10),
            "render_relative_luminance": round(render_luminance, 10),
            "luminance_ratio": round(safe_reference / safe_render, 10),
            "log_luminance_error": round(
                math.log(safe_reference) - math.log(safe_render), 10
            ),
            "appearance_score": float(score["appearance_score"]),
            "score": score,
        }
    return output


def _best_record(history: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return min(
        history,
        key=lambda row: (
            -float(row["appearance_score"]),
            float(row["linear_intensity_gain"]),
            str(row["candidate_id"]),
        ),
    )


def _recent_distinct_record(
    history: Sequence[Mapping[str, Any]], latest: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    latest_log_gain = math.log(float(latest["linear_intensity_gain"]))
    for row in reversed(history[:-1]):
        if abs(math.log(float(row["linear_intensity_gain"])) - latest_log_gain) > 1e-4:
            return row
    return None


def propose_next_scope_gains(
    *,
    histories: Mapping[str, Sequence[Mapping[str, Any]]],
    minimum_gain: float = MINIMUM_GAIN,
    maximum_gain: float = MAXIMUM_GAIN,
    luminance_correction_damping: float = LUMINANCE_CORRECTION_DAMPING,
    maximum_step_ratio: float = MAXIMUM_STEP_RATIO,
    relative_luminance_tolerance: float = RELATIVE_LUMINANCE_TOLERANCE,
    relative_gain_tolerance: float = RELATIVE_GAIN_TOLERANCE,
    minimum_appearance_improvement: float = MINIMUM_APPEARANCE_IMPROVEMENT,
    stall_patience: int = STALL_PATIENCE,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Propose one continuous gain per scope from all measured iterations."""

    if (
        not 0.0 < minimum_gain < maximum_gain
        or not 0.0 < luminance_correction_damping <= 1.0
        or maximum_step_ratio <= 1.0
        or not 0.0 <= relative_luminance_tolerance < 1.0
        or not 0.0 <= relative_gain_tolerance < 1.0
        or minimum_appearance_improvement < 0.0
        or stall_patience < 1
    ):
        raise AdaptiveCorrespondingMaterialColorError(
            "adaptive colour policy has invalid bounds"
        )
    if not histories:
        raise AdaptiveCorrespondingMaterialColorError(
            "adaptive colour controller has no scope histories"
        )

    next_gains: dict[str, float] = {}
    decisions: list[dict[str, Any]] = []
    active_count = 0
    for scope_id, raw_history in sorted(histories.items()):
        history = list(raw_history)
        if not history:
            raise AdaptiveCorrespondingMaterialColorError(
                f"adaptive scope {scope_id} has no measurements"
            )
        latest = history[-1]
        best = _best_record(history)
        latest_gain = float(latest["linear_intensity_gain"])
        best_gain = float(best["linear_intensity_gain"])
        log_error = float(latest["log_luminance_error"])
        relative_luminance_error = math.exp(abs(log_error)) - 1.0
        stop_reason: str | None = None
        method = "best_history"
        proposed_gain = best_gain

        if relative_luminance_error <= relative_luminance_tolerance:
            stop_reason = "luminance_matched"
        elif len(history) > stall_patience:
            earlier_best = max(
                float(row["appearance_score"]) for row in history[:-stall_patience]
            )
            recent_best = max(
                float(row["appearance_score"]) for row in history[-stall_patience:]
            )
            if recent_best - earlier_best < minimum_appearance_improvement:
                stop_reason = "appearance_stalled"

        if stop_reason is None:
            previous = _recent_distinct_record(history, latest)
            proposed_log_gain: float | None = None
            response_slope: float | None = None
            if previous is not None:
                previous_log_gain = math.log(float(previous["linear_intensity_gain"]))
                latest_log_gain = math.log(latest_gain)
                previous_log_render = math.log(
                    max(float(previous["render_relative_luminance"]), 1e-6)
                )
                latest_log_render = math.log(
                    max(float(latest["render_relative_luminance"]), 1e-6)
                )
                response_slope = (latest_log_render - previous_log_render) / (
                    latest_log_gain - previous_log_gain
                )
                if MINIMUM_RESPONSE_SLOPE <= response_slope <= MAXIMUM_RESPONSE_SLOPE:
                    proposed_log_gain = latest_log_gain + log_error / response_slope
                    method = "safeguarded_log_secant"
            if proposed_log_gain is None:
                proposed_log_gain = math.log(latest_gain) + (
                    luminance_correction_damping * log_error
                )
                method = "damped_luminance_ratio"
            lower_step = max(minimum_gain, latest_gain / maximum_step_ratio)
            upper_step = min(maximum_gain, latest_gain * maximum_step_ratio)
            proposed_gain = min(
                upper_step,
                max(lower_step, math.exp(proposed_log_gain)),
            )
            relative_gain_change = abs(proposed_gain / latest_gain - 1.0)
            if relative_gain_change <= relative_gain_tolerance:
                stop_reason = "gain_step_converged"
                proposed_gain = best_gain
                method = "best_history"
            elif any(
                abs(proposed_gain / float(row["linear_intensity_gain"]) - 1.0)
                <= relative_gain_tolerance
                for row in history
            ):
                stop_reason = "proposal_already_measured"
                proposed_gain = best_gain
                method = "best_history"

        active = stop_reason is None
        if active:
            active_count += 1
        next_gains[scope_id] = round(float(proposed_gain), 10)
        decisions.append(
            {
                "scope_id": scope_id,
                "measurement_count": len(history),
                "latest_gain": latest_gain,
                "latest_appearance_score": float(latest["appearance_score"]),
                "best_candidate_id": best["candidate_id"],
                "best_gain": best_gain,
                "best_appearance_score": float(best["appearance_score"]),
                "relative_luminance_error": round(relative_luminance_error, 10),
                "proposed_gain": round(float(proposed_gain), 10),
                "proposal_method": method,
                "active": active,
                "stop_reason": stop_reason,
            }
        )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "scope_count": len(histories),
        "active_scope_count": active_count,
        "all_scopes_converged": active_count == 0,
        "policy": {
            "minimum_gain": minimum_gain,
            "maximum_gain": maximum_gain,
            "luminance_correction_damping": luminance_correction_damping,
            "maximum_step_ratio": maximum_step_ratio,
            "relative_luminance_tolerance": relative_luminance_tolerance,
            "relative_gain_tolerance": relative_gain_tolerance,
            "minimum_appearance_improvement": minimum_appearance_improvement,
            "stall_patience": stall_patience,
            "response_slope_bounds": [
                MINIMUM_RESPONSE_SLOPE,
                MAXIMUM_RESPONSE_SLOPE,
            ],
        },
        "decisions": decisions,
    }
    return next_gains, audit


__all__ = [
    "INITIAL_GAIN",
    "MINIMUM_GAIN",
    "MAXIMUM_GAIN",
    "score_adaptive_candidate_scopes",
    "propose_next_scope_gains",
]
