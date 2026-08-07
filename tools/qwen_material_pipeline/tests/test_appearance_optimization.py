from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from qwen_material_pipeline.materials.appearance_optimization import (
    DECISION_ADJUST,
    DECISION_LIGHTING_INCONSISTENT,
    EXIT_SUCCESS,
    LIGHTING_STATISTICS_SCHEMA_VERSION,
    AppearanceOptimizationError,
    _part_color,
    apply_shared_material_optimization,
    build_shared_material_optimization_contract,
    main,
    measure_lighting_normalized_group_statistics,
    validate_shared_material_optimization_result,
)
from qwen_material_pipeline.materials.parameters import srgb_to_linear
from qwen_material_pipeline.materials.tuning import (
    tune_selected_material_from_mvinverse,
)


MATERIAL_ID = "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted"
GROUP_ID = "G11"


def _mvinverse() -> dict:
    return {
        "schema_version": "qwen-mvinverse-pbr-evidence/v1",
        "groups": [
            {
                "group_id": GROUP_ID,
                "surface_class": "dielectric",
                "contributing_view_ids": ["view-a", "view-b", "view-c"],
                "albedo": {
                    "sample_count": 3,
                    "median": [0.2, 0.5, 0.2],
                    "mad": [0.01, 0.01, 0.01],
                },
                "metallic": {"sample_count": 3, "median": 0.08},
                "roughness": {"sample_count": 3, "median": 0.4},
                "suggestion": {
                    "decision": "auto",
                    "auto_parameter_eligible": True,
                    "base_color_srgb": [0.2, 0.5, 0.2],
                    "metallic": 0.0,
                    "roughness": 0.4,
                    "reason_codes": ["multi_view_evidence_sufficient"],
                    "warning_codes": [],
                },
            }
        ],
    }


def _plan() -> dict:
    evidence = _mvinverse()["groups"][0]
    parameters, _audit = tune_selected_material_from_mvinverse(
        evidence,
        group_id=GROUP_ID,
        material_id=MATERIAL_ID,
    )
    parameters["dirt_weight"] = 0.0
    return {
        "schema_version": "1.0",
        "assignments": [
            {
                "part_id": part_id,
                "material_id": MATERIAL_ID,
                "parameters": copy.deepcopy(parameters),
                "status": "auto",
                "confidence": 0.9,
                "evidence_views": ["view-a", "view-b"],
            }
            for part_id in ("left-panel", "right-panel", "rear-panel")
        ]
        + [
            {
                "part_id": "unrelated-fastener",
                "material_id": ("mdl:Metals/Steel_Stainless.mdl#Steel_Stainless"),
                "status": "policy_fallback",
                "confidence": 0.0,
                "evidence_views": [],
            }
        ],
    }


def _quality(
    values: list[tuple[str, float, float]],
    *,
    normalized: list[dict] | None = None,
) -> dict:
    normalized_by_view = {
        record["reference_view_id"]: record for record in (normalized or [])
    }
    views = []
    for view_id, reference_value, render_value in values:
        material_color = {
            "trusted_evidence_group_recall": {
                "group_count": 1,
                "macro_recall": 1.0,
                "minimum_group_recall": 1.0,
                "groups": [
                    {
                        "group_id": GROUP_ID,
                        "reference_group_share": 0.8,
                        "reference_color_share": 0.8,
                        "observed_render_share": 0.8,
                        "recall": 1.0,
                    }
                ],
            },
            "reference_distribution": {"median_value": reference_value},
            "render_distribution": {"median_value": render_value},
        }
        if view_id in normalized_by_view:
            material_color["lighting_normalized_groups"] = {
                "schema_version": LIGHTING_STATISTICS_SCHEMA_VERSION,
                "groups": [normalized_by_view[view_id]["group"]],
            }
        views.append(
            {
                "reference_view_id": view_id,
                "render_view_id": f"render-{view_id}",
                "status": "PASS",
                "reasons": [],
                "alignment": {"score": 0.9, "silhouette_iou": 0.9},
                "material_color": material_color,
            }
        )
    return {
        "schema_version": "qwen-reference-render-comparison/v1",
        "thresholds": {"strong_alignment_score": 0.55},
        "aggregate": {
            "status": "PASS",
            "comparable_view_count": len(views),
            "failed_view_count": 0,
            "unscorable_view_count": 0,
        },
        "views": views,
    }


def _scaled(rgb: list[float], scale: float) -> list[float]:
    return [channel * scale for channel in rgb]


def _normalized_records(*, material_gain: float) -> list[dict]:
    """Return two views with very different exposure but one material gain."""

    albedo = srgb_to_linear([0.2, 0.5, 0.2])
    records = []
    for view_id, reference_exposure, render_exposure in (
        ("view-a", 0.8, 0.4),
        ("view-b", 0.4, 1.2),
    ):
        records.append(
            {
                "reference_view_id": view_id,
                "group": {
                    "canonical_group_id": GROUP_ID,
                    "reference": {
                        "sampled_pixels": 512,
                        "median_linear_rgb": _scaled(albedo, reference_exposure),
                    },
                    "render": {
                        "sampled_pixels": 512,
                        "median_linear_rgb": _scaled(
                            albedo, render_exposure * material_gain
                        ),
                    },
                    "neutral_anchor": {
                        "canonical_group_ids": ["G90"],
                        "reference": {
                            "sampled_pixels": 512,
                            "median_linear_rgb": [0.5 * reference_exposure] * 3,
                        },
                        "render": {
                            "sampled_pixels": 512,
                            "median_linear_rgb": [0.5 * render_exposure] * 3,
                        },
                    },
                },
            }
        )
    return records


def _build(
    quality: dict,
) -> dict:
    return build_shared_material_optimization_contract(
        final_plan=_plan(),
        quality_report=quality,
        mvinverse_evidence=_mvinverse(),
    )


def test_raw_value_qa_detects_contradictory_view_lighting_and_blocks_tint() -> None:
    # This reproduces the important shape of the real residual: one render is
    # darker than its reference while two are much brighter.  All views may
    # still be ordinary raw-RGB QA PASS records.
    quality = _quality(
        [
            ("view-a", 0.459, 0.380),
            ("view-b", 0.353, 0.718),
            ("view-c", 0.451, 0.667),
        ]
    )

    contract = _build(quality)

    assert contract["summary"]["shared_cohort_count"] == 1
    assert contract["summary"]["adjustment_count"] == 0
    assert contract["summary"]["lighting_inconsistent_count"] == 1
    cohort = contract["cohorts"][0]
    assert cohort["decision"] == DECISION_LIGHTING_INCONSISTENT
    assert cohort["suggestion"] is None
    assert cohort["diagnostics"]["raw_gain"]["span_stops"] > 1.2
    assert "VIEW_DEPENDENT_RAW_BRIGHTNESS_RESIDUAL" in cohort["reason_codes"]
    assert "SHARED_PARAMETER_CHANGE_BLOCKED" in cohort["reason_codes"]


def test_neutral_anchors_separate_exposure_and_author_one_bounded_shared_step() -> None:
    normalized = _normalized_records(material_gain=1.25)
    quality = _quality(
        [("view-a", 0.45, 0.30), ("view-b", 0.25, 0.75)],
        normalized=normalized,
    )

    contract = _build(quality)

    cohort = contract["cohorts"][0]
    assert cohort["decision"] == DECISION_ADJUST
    assert cohort["part_ids"] == [
        "left-panel",
        "rear-panel",
        "right-panel",
    ]
    assert cohort["suggestion"]["adjustment_stops"] == pytest.approx(math.log2(0.8))
    assert cohort["suggestion"]["linear_color_scale"] == pytest.approx(0.8)
    assert cohort["diagnostics"]["normalized_gain"]["span_stops"] == pytest.approx(0.0)
    assert cohort["diagnostics"]["view_exposure"]["span_stops"] > 2.5
    assert "VIEW_LIGHTING_EXPOSURE_VARIES" in cohort["reason_codes"]


def test_normalized_opposite_residuals_fail_closed_instead_of_averaging() -> None:
    normalized = _normalized_records(material_gain=1.25)
    # The second view has the opposite target-only residual after neutral
    # normalization.  A median tint would merely move the error elsewhere.
    second = normalized[1]["group"]
    albedo = srgb_to_linear([0.2, 0.5, 0.2])
    second["render"]["median_linear_rgb"] = _scaled(albedo, 1.2 * 0.8)
    quality = _quality(
        [("view-a", 0.45, 0.30), ("view-b", 0.25, 0.75)],
        normalized=normalized,
    )

    contract = _build(quality)

    cohort = contract["cohorts"][0]
    assert cohort["decision"] == DECISION_LIGHTING_INCONSISTENT
    assert cohort["suggestion"] is None
    assert cohort["diagnostics"]["normalized_gain"]["span_stops"] > 0.6
    assert "NORMALIZED_RESIDUAL_INCONSISTENT_ACROSS_VIEWS" in cohort["reason_codes"]


def test_apply_updates_every_shared_member_and_preserves_unrelated_parameters() -> None:
    quality = _quality(
        [("view-a", 0.45, 0.30), ("view-b", 0.25, 0.75)],
        normalized=_normalized_records(material_gain=1.25),
    )
    source = _plan()
    contract = build_shared_material_optimization_contract(
        final_plan=source,
        quality_report=quality,
        mvinverse_evidence=_mvinverse(),
    )

    candidate, report = apply_shared_material_optimization(
        final_plan=source,
        contract=contract,
    )

    assert report["changed_part_count"] == 3
    assert report["changed_part_ids"] == [
        "left-panel",
        "rear-panel",
        "right-panel",
    ]
    source_by_part = {
        assignment["part_id"]: assignment for assignment in source["assignments"]
    }
    candidate_by_part = {
        assignment["part_id"]: assignment for assignment in candidate["assignments"]
    }
    for part_id in report["changed_part_ids"]:
        source_parameters = source_by_part[part_id]["parameters"]
        candidate_parameters = candidate_by_part[part_id]["parameters"]
        assert candidate_parameters["paint_color"] == pytest.approx(
            [channel * 0.8 for channel in source_parameters["paint_color"]]
        )
        assert (
            candidate_parameters["paint_roughness"]
            == source_parameters["paint_roughness"]
        )
        assert candidate_parameters["dirt_weight"] == 0.0
    assert (
        candidate_by_part["unrelated-fastener"] == source_by_part["unrelated-fastener"]
    )


def test_rerender_validation_accepts_improvement_and_rejects_plan_tampering() -> None:
    baseline_quality = _quality(
        [("view-a", 0.45, 0.30), ("view-b", 0.25, 0.75)],
        normalized=_normalized_records(material_gain=1.25),
    )
    source = _plan()
    contract = build_shared_material_optimization_contract(
        final_plan=source,
        quality_report=baseline_quality,
        mvinverse_evidence=_mvinverse(),
    )
    candidate, _report = apply_shared_material_optimization(
        final_plan=source,
        contract=contract,
    )
    candidate_quality = _quality(
        [("view-a", 0.45, 0.24), ("view-b", 0.25, 0.60)],
        normalized=_normalized_records(material_gain=1.0),
    )

    validation = validate_shared_material_optimization_result(
        source_plan=source,
        contract=contract,
        candidate_plan=candidate,
        candidate_quality_report=candidate_quality,
    )

    assert validation["status"] == "PASS"
    assert validation["cohorts"][0]["candidate_objective_stops"] == pytest.approx(
        0.0, abs=1e-12
    )
    tampered = copy.deepcopy(candidate)
    tampered["assignments"][0]["parameters"]["paint_color"][0] += 0.01
    with pytest.raises(
        AppearanceOptimizationError,
        match="exact atomic contract application",
    ):
        validate_shared_material_optimization_result(
            source_plan=source,
            contract=contract,
            candidate_plan=tampered,
            candidate_quality_report=candidate_quality,
        )


def test_rerender_validation_fails_closed_when_objective_does_not_improve() -> None:
    baseline_quality = _quality(
        [("view-a", 0.45, 0.30), ("view-b", 0.25, 0.75)],
        normalized=_normalized_records(material_gain=1.25),
    )
    source = _plan()
    contract = build_shared_material_optimization_contract(
        final_plan=source,
        quality_report=baseline_quality,
        mvinverse_evidence=_mvinverse(),
    )
    candidate, _report = apply_shared_material_optimization(
        final_plan=source,
        contract=contract,
    )

    validation = validate_shared_material_optimization_result(
        source_plan=source,
        contract=contract,
        candidate_plan=candidate,
        candidate_quality_report=baseline_quality,
    )

    assert validation["status"] == "FAIL_CLOSED"
    assert (
        "ABSOLUTE_OBJECTIVE_IMPROVEMENT_BELOW_FLOOR"
        in validation["cohorts"][0]["reason_codes"]
    )


def test_cli_builds_machine_readable_contract(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    quality_path = tmp_path / "quality.json"
    mvinverse_path = tmp_path / "mvinverse.json"
    output_path = tmp_path / "contract.json"
    plan_path.write_text(json.dumps(_plan()), encoding="utf-8")
    quality_path.write_text(
        json.dumps(
            _quality(
                [
                    ("view-a", 0.459, 0.380),
                    ("view-b", 0.353, 0.718),
                    ("view-c", 0.451, 0.667),
                ]
            )
        ),
        encoding="utf-8",
    )
    mvinverse_path.write_text(json.dumps(_mvinverse()), encoding="utf-8")

    exit_code = main(
        [
            "build",
            "--final-plan",
            str(plan_path),
            "--quality-report",
            str(quality_path),
            "--mvinverse-evidence",
            str(mvinverse_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == EXIT_SUCCESS
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["summary"]["lighting_inconsistent_count"] == 1
    assert output["cohorts"][0]["decision"] == DECISION_LIGHTING_INCONSISTENT


def test_measurement_api_emits_statistics_without_rematching_views(
    tmp_path: Path,
) -> None:
    quality = _quality([("view-a", 0.45, 0.30), ("view-b", 0.25, 0.75)])
    palette_maps = {}
    spatial_parts = []
    for index, view in enumerate(quality["views"]):
        reference_view_id = view["reference_view_id"]
        render_view_id = view["render_view_id"]
        view["material_color"]["trusted_evidence_group_recall"]["groups"][0][
            "group_id"
        ] = "local-target"
        rgb_path = tmp_path / f"{reference_view_id}.rgb.png"
        ids_path = tmp_path / f"{reference_view_id}.ids.png"
        rgb = Image.new("RGB", (64, 64), (16, 16, 16))
        ids = Image.new("RGB", (64, 64), (0, 0, 0))
        ImageDraw.Draw(rgb).rectangle((4, 4, 31, 59), fill=(45, 115, 45))
        ImageDraw.Draw(ids).rectangle((4, 4, 31, 59), fill=_part_color("left-panel"))
        ImageDraw.Draw(rgb).rectangle(
            (36, 4, 59, 59),
            fill=(160 + index * 20,) * 3,
        )
        ImageDraw.Draw(ids).rectangle(
            (36, 4, 59, 59), fill=_part_color("neutral-anchor")
        )
        rgb.save(rgb_path)
        ids.save(ids_path)
        view["reference"] = {
            "trusted_evidence": {
                "usable": True,
                "samples": [
                    {
                        "group_id": "local-target",
                        "representative_srgb": [50, 120, 50],
                        "weight_pixels": 1000,
                    },
                    {
                        "group_id": "local-neutral",
                        "representative_srgb": [200, 200, 200],
                        "weight_pixels": 500,
                    },
                ],
            }
        }
        import hashlib

        view["render"] = {
            "image": str(rgb_path),
            "image_sha256": hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
            "part_ids": str(ids_path),
            "part_ids_sha256": hashlib.sha256(ids_path.read_bytes()).hexdigest(),
        }
        palette_maps[reference_view_id] = {
            "local-target": GROUP_ID,
            "local-neutral": "G90",
        }
        spatial_parts.append(
            {
                "part_id": "neutral-anchor",
                "observations": [
                    {
                        "reference_view_id": reference_view_id,
                        "render_view_id": render_view_id,
                        "classification": "resolved",
                        "projected_part_pixels": 512,
                        "color_margin": 0.8,
                        "group_scores": [
                            {
                                "canonical_group_id": "G90",
                                "color_share": 0.9,
                            }
                        ],
                    }
                ],
            }
        )
    palette_fusion = {
        "schema_version": "qwen-multiview-palette-fusion/v1",
        "canonical_palette": {
            "groups": [
                {"group_id": GROUP_ID, "base_color": "green"},
                {"group_id": "G90", "base_color": "white"},
            ]
        },
        "view_group_id_maps": palette_maps,
    }
    spatial_report = {
        "schema_version": "qwen-spatial-mapping-audit/v1",
        "parts": spatial_parts,
    }

    measured_quality, report = measure_lighting_normalized_group_statistics(
        final_plan=_plan(),
        quality_report=quality,
        mvinverse_evidence=_mvinverse(),
        palette_fusion=palette_fusion,
        spatial_report=spatial_report,
    )

    assert report["summary"]["measured_view_count"] == 2
    assert report["summary"]["skipped_view_count"] == 0
    for view in measured_quality["views"]:
        document = view["material_color"]["lighting_normalized_groups"]
        assert document["schema_version"] == LIGHTING_STATISTICS_SCHEMA_VERSION
        statistic = document["groups"][0]
        assert statistic["canonical_group_id"] == GROUP_ID
        assert statistic["neutral_anchor"]["canonical_group_ids"] == ["G90"]
        assert statistic["render"]["sampled_pixels"] > 128
