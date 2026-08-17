from __future__ import annotations

import math

import pytest

from qwen_material_pipeline.materials import (
    adaptive_corresponding_material_color as adaptive,
)
from qwen_material_pipeline.materials.corresponding_material_color_selection import (
    Candidate,
)


def _measurement(
    candidate_id: str,
    gain: float,
    reference: float,
    rendered: float,
    appearance: float,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "linear_intensity_gain": gain,
        "reference_relative_luminance": reference,
        "render_relative_luminance": rendered,
        "log_luminance_error": math.log(reference) - math.log(rendered),
        "appearance_score": appearance,
    }


def test_first_proposal_is_per_scope_bounded_and_keeps_matched_scope() -> None:
    next_gains, audit = adaptive.propose_next_scope_gains(
        histories={
            "COMPONENT:dark": [_measurement("i0", 1.0, 0.4, 0.1, 0.4)],
            "PART:matched": [_measurement("i0", 1.0, 0.4, 0.395, 0.9)],
        }
    )
    assert next_gains == {"COMPONENT:dark": 2.0, "PART:matched": 1.0}
    decisions = {row["scope_id"]: row for row in audit["decisions"]}
    assert decisions["COMPONENT:dark"]["proposal_method"] == ("damped_luminance_ratio")
    assert decisions["COMPONENT:dark"]["active"] is True
    assert decisions["PART:matched"]["stop_reason"] == "luminance_matched"
    assert audit["active_scope_count"] == 1


def test_second_measurement_uses_safeguarded_log_secant() -> None:
    next_gains, audit = adaptive.propose_next_scope_gains(
        histories={
            "PART:P1": [
                _measurement("i0", 1.0, 0.4, 0.1, 0.4),
                _measurement("i1", 2.0, 0.4, 0.2, 0.7),
            ]
        }
    )
    assert next_gains["PART:P1"] == pytest.approx(4.0)
    assert audit["decisions"][0]["proposal_method"] == ("safeguarded_log_secant")
    assert audit["decisions"][0]["active"] is True


def test_stalled_scope_returns_best_measured_gain() -> None:
    next_gains, audit = adaptive.propose_next_scope_gains(
        histories={
            "PART:P1": [
                _measurement("i0", 1.0, 0.4, 0.1, 0.5000),
                _measurement("i1", 2.0, 0.4, 0.15, 0.5010),
                _measurement("i2", 3.0, 0.4, 0.16, 0.5015),
            ]
        }
    )
    assert next_gains == {"PART:P1": 3.0}
    assert audit["all_scopes_converged"] is True
    assert audit["decisions"][0]["stop_reason"] == "appearance_stalled"


def test_scope_scoring_uses_registered_part_and_component_luminance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = Candidate(
        candidate_id="iteration_00",
        gains_by_scope={"PART:P1": 1.0, "COMPONENT:C1": 2.0},
        plan={},
        audit={
            "scopes": [
                {"scope_id": "PART:P1", "member_part_ids": ["P1"]},
                {
                    "scope_id": "COMPONENT:C1",
                    "member_part_ids": ["P2", "P3"],
                },
            ]
        },
        rendered_registry={},
        paths={},
        hashes={},
    )

    def part_score(**_kwargs: object) -> dict:
        return {
            "comparison_pixel_count": 10,
            "reference_lab_medoid": [50.0, 0.0, 0.0],
            "render_lab_medoid": [25.0, 0.0, 0.0],
            "appearance_score": 0.7,
        }

    def component_score(**_kwargs: object) -> dict:
        return {
            "appearance_score": 0.8,
            "member_scores": [
                {
                    "comparison_pixel_count": 10,
                    "reference_lab_medoid": [40.0, 0.0, 0.0],
                    "render_lab_medoid": [20.0, 0.0, 0.0],
                },
                {
                    "comparison_pixel_count": 30,
                    "reference_lab_medoid": [60.0, 0.0, 0.0],
                    "render_lab_medoid": [30.0, 0.0, 0.0],
                },
            ],
        }

    monkeypatch.setattr(adaptive, "score_part_id_render", part_score)
    monkeypatch.setattr(adaptive, "score_component_render", component_score)
    scores = adaptive.score_adaptive_candidate_scopes(
        candidate=candidate,
        part_id_evidence={},
        spatial_mapping_report={},
    )
    assert set(scores) == {"PART:P1", "COMPONENT:C1"}
    assert scores["PART:P1"]["linear_intensity_gain"] == 1.0
    assert scores["COMPONENT:C1"]["linear_intensity_gain"] == 2.0
    assert scores["PART:P1"]["luminance_ratio"] > 1.0
    assert scores["COMPONENT:C1"]["luminance_ratio"] > 1.0
