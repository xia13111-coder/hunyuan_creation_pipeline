from __future__ import annotations

import hashlib
import importlib.util
import json
from html.parser import HTMLParser
from pathlib import Path
from types import ModuleType

import pytest


LIMITED_PASS = "MATERIAL_ACCEPTED_WITH_GEOMETRY_POSE_LIMITATION"
RESTORED_BASELINE = "RESTORED_HISTORICAL_BASELINE"


def _viewer_module() -> ModuleType:
    path = Path(__file__).parents[1] / "web" / "result_viewer" / "build_manifest.py"
    spec = importlib.util.spec_from_file_location("qwen_result_viewer_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VIEWER = _viewer_module()


class _IdCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del tag
        for name, value in attrs:
            if name == "id" and value is not None:
                self.ids.append(value)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pipeline_result(material_result: dict) -> dict:
    return {
        "postprocess": {
            "workflow": "manual_cad",
            "steps": [
                {"step": "cad_to_usd", "result": {}},
                {
                    "step": "assign_visual_materials",
                    "result": material_result,
                },
            ],
        }
    }


def _resolution() -> dict:
    return {
        "schema_version": "qwen-visual-quality-resolution/v1",
        "raw_quality_status": "FAIL",
        "resolution_status": LIMITED_PASS,
        "material_stage_accepted": True,
        "reason_codes": [],
        "limitations": [
            {
                "classification": "NOT_OBSERVABLE_GEOMETRY_POSE",
                "reason_code": "POSE_OR_OCCLUSION_MISMATCH",
                "canonical_group_id": "G05",
                "reference_view_id": "top",
                "reference_group_evidence": {"evidence_pixels": 256},
                "foreign_owner": {
                    "part_id": "P0091",
                    "accepted_box_overlap": {"projected_overlap_share": 0.88671875},
                },
            }
        ],
        "summary": {"accepted_limitation_count": 1},
    }


def test_new_result_keeps_raw_and_gate_status_separate(tmp_path: Path) -> None:
    resolution_path = (
        tmp_path / "visual_material" / "analysis" / "visual_quality_resolution.json"
    )
    _write_json(resolution_path, _resolution())
    _write_json(
        tmp_path / "result.json",
        _pipeline_result(
            {
                "visual_quality_raw_status": "FAIL",
                "visual_quality_status": "FAIL",
                "visual_quality_gate_status": LIMITED_PASS,
                "visual_quality_resolution": str(resolution_path),
                "visual_quality_limitation_count": 1,
            }
        ),
    )
    _write_json(
        tmp_path
        / "visual_material"
        / "visual_quality_repair"
        / "reference_render_comparison.json",
        {"aggregate": {"status": "FAIL"}},
    )

    manifest = VIEWER.build_viewer_manifest(tmp_path)

    assert manifest["visual_quality_raw_status"] == "FAIL"
    assert manifest["visual_quality_gate_status"] == LIMITED_PASS
    assert manifest["visual_quality_limitation_count"] == 1
    assert manifest["material_stage_accepted"] is True
    assert manifest["visual_quality_resolution"] == {
        "state": "AVAILABLE",
        "href": ("delivery/visual_material/analysis/visual_quality_resolution.json"),
    }
    limitation = manifest["limitation_reasons"][0]
    assert limitation["reason_code"] == "POSE_OR_OCCLUSION_MISMATCH"
    assert limitation["reference_view_id"] == "top"
    assert limitation["canonical_group_id"] == "G05"
    assert limitation["foreign_owner_part_id"] == "P0091"
    assert limitation["projected_overlap_share"] == 0.88671875
    assert "重叠 88.7%" in limitation["summary_zh"]
    assert "证据像素 256" in limitation["summary_zh"]
    assert manifest["source"]["legacy_compatibility"] is False


def test_completed_result_ignores_stale_unrecorded_resolution(tmp_path: Path) -> None:
    stale_resolution = (
        tmp_path / "visual_material" / "analysis" / "visual_quality_resolution.json"
    )
    _write_json(stale_resolution, _resolution())
    _write_json(
        tmp_path / "pipeline_result.json",
        _pipeline_result(
            {
                "visual_quality_raw_status": "PASS",
                "visual_quality_gate_status": "PASS",
                "visual_quality_resolution": None,
                "visual_quality_limitation_count": 0,
            }
        ),
    )

    manifest = VIEWER.build_viewer_manifest(tmp_path)

    assert manifest["visual_quality_raw_status"] == "PASS"
    assert manifest["visual_quality_gate_status"] == "PASS"
    assert manifest["visual_quality_resolution"] == {
        "state": "NOT_REQUIRED",
        "href": None,
    }
    assert manifest["warnings"] == []
    assert manifest["source"]["resolution_report"] is None


def test_legacy_visual_quality_status_remains_visible(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "result.json",
        _pipeline_result({"visual_quality_status": "PASS"}),
    )

    manifest = VIEWER.build_viewer_manifest(tmp_path)

    assert manifest["visual_quality_raw_status"] == "PASS"
    assert manifest["visual_quality_gate_status"] == "PASS"
    assert manifest["visual_quality_limitation_count"] == 0
    assert manifest["visual_quality_resolution"] == {
        "state": "LEGACY_NOT_RECORDED",
        "href": None,
    }
    assert manifest["source"]["legacy_compatibility"] is True
    assert "旧字段" in manifest["note_zh"]


def test_unattended_resume_result_supports_visual_material_alias(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "resume_result.json",
        {
            "schema_version": "asset-pipeline-manual-cad-resume/v1",
            "state": "COMPLETED",
            "visual_material": {
                "visual_quality_raw_status": "PASS",
                "visual_quality_gate_status": "PASS",
                "visual_quality_limitation_count": 0,
            },
        },
    )

    manifest = VIEWER.build_viewer_manifest(tmp_path)

    assert manifest["visual_quality_raw_status"] == "PASS"
    assert manifest["visual_quality_gate_status"] == "PASS"
    assert manifest["material_stage_accepted"] is True
    assert manifest["source"]["pipeline_result"] == "delivery/resume_result.json"
    assert manifest["source"]["legacy_compatibility"] is False


def test_canonical_visual_material_result_precedes_resume_alias(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "resume_result.json",
        {
            "visual_material_result": {
                "visual_quality_raw_status": "PASS",
                "visual_quality_gate_status": "PASS",
            },
            "visual_material": {
                "visual_quality_raw_status": "FAIL",
                "visual_quality_gate_status": "FAIL",
            },
        },
    )

    manifest = VIEWER.build_viewer_manifest(tmp_path)

    assert manifest["visual_quality_raw_status"] == "PASS"
    assert manifest["visual_quality_gate_status"] == "PASS"


def test_historical_final_preview_is_visible_without_claiming_quality_pass(
    tmp_path: Path,
) -> None:
    rgb_dir = tmp_path / "final" / "asset_phys" / "preview_final" / "rgb"
    rgb_dir.mkdir(parents=True)
    iso = rgb_dir / "iso.png"
    iso.write_bytes(b"historical preview")
    (tmp_path / "preview_final").symlink_to(
        Path("final") / "asset_phys" / "preview_final",
        target_is_directory=True,
    )

    manifest = VIEWER.build_viewer_manifest(tmp_path)

    assert manifest["visual_quality_gate_status"] is None
    assert manifest["material_stage_accepted"] is False
    assert manifest["source"]["quality_report"] is None
    assert manifest["source"]["legacy_final_preview"] is True
    assert manifest["source"]["preview_images"] == {
        "iso": "delivery/final/asset_phys/preview_final/rgb/iso.png",
    }
    assert "仅用于结果回看" in manifest["note_zh"]


def test_resolution_and_quality_report_work_without_root_result(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "analysis" / "visual_quality_resolution.json",
        _resolution(),
    )
    _write_json(
        tmp_path / "visual_quality_repair" / "reference_render_comparison.json",
        {"aggregate": {"status": "FAIL"}},
    )

    manifest = VIEWER.build_viewer_manifest(tmp_path)

    assert manifest["visual_quality_raw_status"] == "FAIL"
    assert manifest["visual_quality_gate_status"] == LIMITED_PASS
    assert manifest["visual_quality_limitation_count"] == 1
    assert manifest["visual_quality_resolution"]["href"] == (
        "delivery/analysis/visual_quality_resolution.json"
    )
    assert manifest["source"]["quality_report"] == (
        "delivery/visual_quality_repair/reference_render_comparison.json"
    )
    assert manifest["source"]["legacy_compatibility"] is False


def test_report_count_mismatch_is_not_hidden(tmp_path: Path) -> None:
    resolution_path = (
        tmp_path / "visual_material" / "analysis" / "visual_quality_resolution.json"
    )
    _write_json(resolution_path, _resolution())
    _write_json(
        tmp_path / "result.json",
        _pipeline_result(
            {
                "visual_quality_raw_status": "FAIL",
                "visual_quality_gate_status": LIMITED_PASS,
                "visual_quality_resolution": str(resolution_path),
                "visual_quality_limitation_count": 0,
            }
        ),
    )

    manifest = VIEWER.build_viewer_manifest(tmp_path)

    assert manifest["visual_quality_limitation_count"] == 0
    assert len(manifest["limitation_reasons"]) == 1
    assert len(manifest["warnings"]) == 1
    assert "数量不一致" in manifest["warnings"][0]


def _quality_round(
    root: Path,
    name: str,
    *,
    status: str,
    pixel: bytes,
) -> tuple[Path, Path, Path]:
    quality_dir = root / "visual_material" / name
    report = quality_dir / "reference_render_comparison.measured.json"
    image = quality_dir / "renders" / "rgb" / "iso.png"
    registry = quality_dir / "renders" / "part_registry.rendered.json"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(pixel)
    _write_json(report, {"aggregate": {"status": status}})
    _write_json(
        registry,
        {
            "render_set": {
                "views": [
                    {
                        "view_id": "iso",
                        "rgb": str(image),
                    }
                ]
            }
        },
    )
    return report, registry, image


def test_accepted_appearance_uses_recorded_candidate_artifacts(
    tmp_path: Path,
) -> None:
    candidate_report, candidate_registry, candidate_image = _quality_round(
        tmp_path,
        "custom-refined-round",
        status="PASS",
        pixel=b"candidate",
    )
    _quality_round(
        tmp_path,
        "visual_quality_repair",
        status="FAIL",
        pixel=b"baseline",
    )
    _write_json(
        tmp_path / "result.json",
        _pipeline_result(
            {
                "appearance_optimization_status": "ACCEPTED",
                "appearance_optimization_candidate_quality_report": str(
                    candidate_report
                ),
                "visual_quality_report": str(candidate_report),
                "visual_quality_rendered_registry": str(candidate_registry),
                "visual_quality_raw_status": "PASS",
                "visual_quality_gate_status": "PASS",
            }
        ),
    )

    manifest = VIEWER.build_viewer_manifest(tmp_path)

    assert manifest["appearance_optimization_status"] == "ACCEPTED"
    assert manifest["source"]["quality_report"] == (
        "delivery/visual_material/custom-refined-round/"
        "reference_render_comparison.measured.json"
    )
    assert manifest["source"]["preview_images"] == {
        "iso": "delivery/" + candidate_image.relative_to(tmp_path).as_posix(),
    }


@pytest.mark.parametrize(
    "appearance_status",
    ["REJECTED_FAIL_CLOSED", "NOT_APPLICABLE"],
)
def test_unaccepted_appearance_never_exposes_candidate_artifacts(
    tmp_path: Path,
    appearance_status: str,
) -> None:
    candidate_report, candidate_registry, _candidate_image = _quality_round(
        tmp_path,
        "custom-rejected-round",
        status="PASS",
        pixel=b"rejected",
    )
    baseline_report, _baseline_registry, baseline_image = _quality_round(
        tmp_path,
        "visual_quality_repair",
        status="PASS",
        pixel=b"accepted",
    )
    _write_json(
        tmp_path / "result.json",
        _pipeline_result(
            {
                "appearance_optimization_status": appearance_status,
                "appearance_optimization_candidate_quality_report": str(
                    candidate_report
                ),
                "visual_quality_report": str(baseline_report),
                # Deliberately stale: the viewer must not pair it with the
                # baseline report or expose its rejected images.
                "visual_quality_rendered_registry": str(candidate_registry),
                "visual_quality_raw_status": "PASS",
                "visual_quality_gate_status": "PASS",
            }
        ),
    )

    manifest = VIEWER.build_viewer_manifest(tmp_path)

    assert manifest["appearance_optimization_status"] == appearance_status
    assert manifest["source"]["quality_report"] == (
        "delivery/visual_material/visual_quality_repair/"
        "reference_render_comparison.measured.json"
    )
    assert manifest["source"]["preview_images"] == {
        "iso": "delivery/" + baseline_image.relative_to(tmp_path).as_posix(),
    }
    assert "custom-rejected-round" not in json.dumps(manifest)


def test_rejected_candidate_cannot_be_smuggled_through_effective_report(
    tmp_path: Path,
) -> None:
    candidate_report, candidate_registry, _candidate_image = _quality_round(
        tmp_path,
        "custom-rejected-round",
        status="PASS",
        pixel=b"rejected",
    )
    baseline_report = (
        tmp_path
        / "visual_material"
        / "visual_quality"
        / "reference_render_comparison.json"
    )
    baseline_image = (
        tmp_path / "visual_material" / "visual_quality" / "renders" / "rgb" / "iso.png"
    )
    baseline_image.parent.mkdir(parents=True, exist_ok=True)
    baseline_image.write_bytes(b"accepted")
    _write_json(baseline_report, {"aggregate": {"status": "FAIL"}})
    _write_json(
        tmp_path / "result.json",
        _pipeline_result(
            {
                "appearance_optimization_status": "REJECTED_FAIL_CLOSED",
                "appearance_optimization_candidate_quality_report": str(
                    candidate_report
                ),
                "visual_quality_report": str(candidate_report),
                "visual_quality_rendered_registry": str(candidate_registry),
            }
        ),
    )

    manifest = VIEWER.build_viewer_manifest(tmp_path)

    assert manifest["source"]["quality_report"] == (
        "delivery/visual_material/visual_quality/reference_render_comparison.json"
    )
    assert manifest["source"]["preview_images"] == {
        "iso": "delivery/" + baseline_image.relative_to(tmp_path).as_posix(),
    }
    assert "custom-rejected-round" not in json.dumps(manifest)


def _write_final_collected_acceptance(
    root: Path,
    *,
    acceptance_name: str = "final_visual_acceptance",
    immutable_library_review: bool = False,
    part_id_nonregression: bool = False,
) -> dict[str, object]:
    acceptance_root = root / "visual_material" / acceptance_name
    collected_root = acceptance_root / "collected"
    render_root = collected_root / "renders"
    rgb_root = render_root / "rgb"
    rgb_root.mkdir(parents=True)

    ordered_mapping = [
        ("front", "right"),
        ("side", "front"),
        ("top", "pose_a090_e082_toproll"),
        ("iso", "pose_a135_e015"),
    ]
    image_paths: dict[str, Path] = {}
    registry_views: list[dict[str, str]] = []
    for reference_id, render_id in ordered_mapping:
        image = rgb_root / f"{render_id}.png"
        image.write_bytes(f"collected-{reference_id}".encode())
        image_paths[reference_id] = image
        registry_views.append({"view_id": render_id, "rgb": str(image)})

    registry_path = render_root / "part_registry.rendered.json"
    _write_json(
        registry_path,
        {
            "schema_version": "qwen-material-parts/v1",
            "render_set": {"views": registry_views},
        },
    )
    mapping = dict(ordered_mapping)
    view_map_path = collected_root / "reference_view_map.json"
    _write_json(
        view_map_path,
        {
            "schema_version": "qwen-reference-view-map/v1",
            "mapping": mapping,
            "source": "final_locked_baseline_mapping",
        },
    )
    quality_path = collected_root / "reference_render_comparison.json"
    view_count = len(ordered_mapping)
    view_statuses = ["PASS", "PASS", "PASS", "PASS"]
    aggregate_status = "PASS"
    if immutable_library_review:
        view_statuses = ["REVIEW", "PASS", "REVIEW", "PASS"]
        aggregate_status = "REVIEW"
    elif part_id_nonregression:
        view_statuses = ["REVIEW", "PASS", "REVIEW", "UNSCORABLE"]
        aggregate_status = "INSUFFICIENT_EVIDENCE"
    appearance_scores = [0.91, 0.83, 0.79, None]
    comparable_view_count = sum(
        status in {"PASS", "REVIEW"} for status in view_statuses
    )
    unscorable_reference_ids = [
        reference_id
        for (reference_id, _render_id), status in zip(
            ordered_mapping,
            view_statuses,
            strict=True,
        )
        if status == "UNSCORABLE"
    ]
    _write_json(
        quality_path,
        {
            "schema_version": "qwen-reference-render-comparison/v1",
            "inputs": {
                "rendered_registry": str(registry_path),
                "rendered_registry_sha256": _sha256(registry_path),
                "selected_view_mapping": mapping,
            },
            "aggregate": {
                "status": aggregate_status,
                "material_appearance_score": 0.85,
                "reference_view_count": view_count,
                "render_view_count": view_count,
                "comparable_view_count": comparable_view_count,
                "passed_view_count": view_statuses.count("PASS"),
                "review_view_count": view_statuses.count("REVIEW"),
                "failed_view_count": 0,
                "unscorable_view_count": len(unscorable_reference_ids),
                "reference_view_coverage_status": (
                    "FAIL_CLOSED" if unscorable_reference_ids else "PASS"
                ),
                "unmapped_reference_view_ids": [],
                "unscorable_reference_view_ids": unscorable_reference_ids,
            },
            "views": [
                {
                    "reference_view_id": reference_id,
                    "render_view_id": render_id,
                    "status": status,
                    "material_appearance_score": appearance_score,
                }
                for (reference_id, render_id), status, appearance_score in zip(
                    ordered_mapping,
                    view_statuses,
                    appearance_scores,
                    strict=True,
                )
            ],
        },
    )
    gate_path = acceptance_root / "collected_visual_gate.json"
    if part_id_nonregression:
        comparable_views = [
            {
                "reference_view_id": reference_id,
                "render_view_id": render_id,
                "raw_status": status,
                "material_appearance_score": appearance_score,
                "passes_appearance_floor": True,
            }
            for (reference_id, render_id), status, appearance_score in zip(
                ordered_mapping,
                view_statuses,
                appearance_scores,
                strict=True,
            )
            if status in {"PASS", "REVIEW"}
        ]
        gate_document = {
            "schema_version": "asset-pipeline-part-id-final-visual-gate/v1",
            "status": "PASS",
            "completion_allowed": True,
            "acceptance_mode": "PART_ID_VISUAL_NONREGRESSION",
            "policy": {"minimum_comparable_views": 2},
            "measurements": {
                "final_part_id_gate": {
                    "schema_version": "asset-pipeline-part-id-quality-gate/v1",
                    "status": "PASS",
                    "acceptance_allowed": True,
                    "assignment_unit": "part_id",
                    "raw_quality_status": aggregate_status,
                    "effective_quality_status": "PASS",
                    "measurements": {
                        "comparable_view_count": 3,
                        "scored_view_count": 3,
                        "aggregate_appearance_score": 0.86,
                        "raw_aggregate_appearance_score": 0.85,
                        "views": comparable_views,
                    },
                    "limitations": [
                        {
                            "code": "UNSCORABLE_REFERENCE_VIEWS",
                            "reference_view_ids": ["iso"],
                        }
                    ],
                }
            },
            "inputs": {
                "final_quality_report": str(quality_path),
                "final_quality_report_sha256": _sha256(quality_path),
                "final_rendered_registry": str(registry_path),
                "final_rendered_registry_sha256": _sha256(registry_path),
            },
        }
    else:
        gate_document = {
            "schema_version": "qwen-final-visual-gate/v1",
            "status": "PASS",
            "completion_allowed": True,
            "completion_state": "COMPLETED",
            "policy": (
                {
                    "acceptance_mode": "IMMUTABLE_LIBRARY_OPTIMUM",
                    "absolute_quality_floors_enforced": True,
                    "immutable_library_review_allowed": True,
                }
                if immutable_library_review
                else {}
            ),
            "summary": {
                "view_count": view_count,
                "passed_view_count": view_count,
                "failure_count": 0,
            },
            "views": [
                {
                    "reference_view_id": reference_id,
                    "render_view_id": render_id,
                    "status": "PASS",
                }
                for reference_id, render_id in ordered_mapping
            ],
            "inputs": {
                "final_quality_report": str(quality_path),
                "final_quality_report_sha256": _sha256(quality_path),
                "final_rendered_registry": str(registry_path),
                "final_rendered_registry_sha256": _sha256(registry_path),
            },
        }
    _write_json(
        gate_path,
        gate_document,
    )
    return {
        "gate": gate_path,
        "quality": quality_path,
        "view_map": view_map_path,
        "registry": registry_path,
        "images": image_paths,
    }


def test_completed_retry_acceptance_exposes_immutable_library_review(
    tmp_path: Path,
) -> None:
    artifacts = _write_final_collected_acceptance(
        tmp_path,
        acceptance_name="final_visual_acceptance_retry2",
        immutable_library_review=True,
    )
    acceptance_root = (
        tmp_path / "visual_material" / "final_visual_acceptance_retry2"
    )
    _write_json(
        tmp_path / "visual_material" / "final_visual_acceptance_result.json",
        {
            "schema_version": "asset-pipeline-final-visual-acceptance/v1",
            "state": "COMPLETED",
            "completion_allowed": True,
            "output_dir": str(acceptance_root),
            "collected_visual_gate": str(artifacts["gate"]),
            "collected_visual_gate_status": "PASS",
        },
    )

    manifest = VIEWER.build_viewer_manifest(tmp_path)

    assert manifest["visual_quality_raw_status"] == "REVIEW"
    assert manifest["visual_quality_gate_status"] == "PASS"
    assert manifest["material_stage_accepted"] is True
    assert manifest["source"]["final_collected_acceptance"] == "PASS"
    assert len(manifest["source"]["preview_images"]) == 4
    assert "final_visual_acceptance_retry2" in manifest["source"]["quality_report"]


def test_part_id_nonregression_acceptance_exposes_hash_bound_images(
    tmp_path: Path,
) -> None:
    artifacts = _write_final_collected_acceptance(
        tmp_path,
        part_id_nonregression=True,
    )

    manifest = VIEWER.build_viewer_manifest(tmp_path)

    images = artifacts["images"]
    assert isinstance(images, dict)
    assert manifest["visual_quality_raw_status"] == "INSUFFICIENT_EVIDENCE"
    assert manifest["visual_quality_gate_status"] == "PASS"
    assert manifest["material_stage_accepted"] is True
    assert manifest["source"]["final_collected_acceptance"] == "PASS"
    assert manifest["source"]["preview_fallback_allowed"] is False
    assert manifest["source"]["preview_images"] == {
        reference_id: "delivery/" + image.relative_to(tmp_path).as_posix()
        for reference_id, image in images.items()
    }


def test_completed_collected_acceptance_exposes_all_reference_role_images(
    tmp_path: Path,
) -> None:
    artifacts = _write_final_collected_acceptance(tmp_path)
    _write_json(
        tmp_path / "pipeline_result.json",
        _pipeline_result(
            {
                "inference_mode": "bundled_project",
                "visual_quality_status": RESTORED_BASELINE,
                "assignment_count": 596,
                "applied_count": 596,
            }
        ),
    )

    manifest = VIEWER.build_viewer_manifest(tmp_path)

    images = artifacts["images"]
    assert isinstance(images, dict)
    assert manifest["source"]["preview_images"] == {
        reference_id: "delivery/" + image.relative_to(tmp_path).as_posix()
        for reference_id, image in images.items()
    }
    assert list(manifest["source"]["preview_images"]) == [
        "front",
        "side",
        "top",
        "iso",
    ]
    assert manifest["source"]["final_collected_acceptance"] == "PASS"
    assert manifest["source"]["preview_fallback_allowed"] is False
    assert manifest["source"]["restored_project"] is True
    assert manifest["source"]["quality_report"] == (
        "delivery/visual_material/final_visual_acceptance/collected/"
        "reference_render_comparison.json"
    )
    assert "collected final" in manifest["note_zh"]
    assert "四个视图" in manifest["note_zh"]


@pytest.mark.parametrize(
    "failure",
    [
        "gate_status",
        "gate_completion_allowed",
        "gate_completion_state",
        "quality_aggregate",
        "quality_view",
        "view_mapping",
        "quality_hash",
    ],
)
def test_final_collected_preview_fails_closed(
    tmp_path: Path,
    failure: str,
) -> None:
    artifacts = _write_final_collected_acceptance(tmp_path)
    gate_path = artifacts["gate"]
    quality_path = artifacts["quality"]
    view_map_path = artifacts["view_map"]
    assert isinstance(gate_path, Path)
    assert isinstance(quality_path, Path)
    assert isinstance(view_map_path, Path)

    if failure.startswith("gate_"):
        gate = json.loads(gate_path.read_text())
        if failure == "gate_status":
            gate["status"] = "FAIL"
        elif failure == "gate_completion_allowed":
            gate["completion_allowed"] = False
        else:
            gate["completion_state"] = "FAILED"
        _write_json(gate_path, gate)
    elif failure in {"quality_aggregate", "quality_view"}:
        quality = json.loads(quality_path.read_text())
        if failure == "quality_aggregate":
            quality["aggregate"]["status"] = "FAIL"
        else:
            quality["views"][2]["status"] = "FAIL"
        _write_json(quality_path, quality)
        gate = json.loads(gate_path.read_text())
        gate["inputs"]["final_quality_report_sha256"] = _sha256(quality_path)
        _write_json(gate_path, gate)
    elif failure == "view_mapping":
        view_map = json.loads(view_map_path.read_text())
        view_map["mapping"]["side"] = "right"
        _write_json(view_map_path, view_map)
    else:
        quality_path.write_text(
            quality_path.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )

    legacy_image = (
        tmp_path / "final" / "asset" / "preview_final" / "rgb" / "iso.png"
    )
    legacy_image.parent.mkdir(parents=True)
    legacy_image.write_bytes(b"must-not-be-exposed")

    manifest = VIEWER.build_viewer_manifest(tmp_path)

    assert manifest["source"]["final_collected_acceptance"] == "REJECTED"
    assert manifest["source"]["preview_fallback_allowed"] is False
    assert manifest["source"]["preview_images"] == {}
    assert manifest["source"]["legacy_final_preview"] is False
    assert len(manifest["warnings"]) == 1
    assert "拒绝展示" in manifest["warnings"][0]


def test_sealed_restored_project_exposes_verified_preview(tmp_path: Path) -> None:
    visual_root = tmp_path / "visual_material"
    preview_root = visual_root / "preview_final"
    image = preview_root / "rgb" / "iso.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"restored")
    registry = preview_root / "part_registry.rendered.json"
    _write_json(
        registry,
        {
            "render_set": {
                "views": [
                    {
                        "view_id": "iso",
                        "rgb": str(image),
                    }
                ]
            }
        },
    )
    audit = visual_root / "analysis" / "project_dtn100" / "audit.json"
    evidence = visual_root / "analysis" / "project_dtn100" / "evidence.json"
    _write_json(
        audit,
        {
            "status": "PASS",
            "complete_coverage": True,
            "topology_verified": True,
            "face_subsets_verified": True,
        },
    )
    _write_json(evidence, {"live_inference_repeated": False})
    _write_json(
        visual_root / "delivery_validation.json",
        {
            "status": "PASS",
            "overall_pass": True,
            "failure_count": 0,
        },
    )
    _write_json(
        tmp_path / "pipeline_result.json",
        _pipeline_result(
            {
                "inference_mode": "bundled_project",
                "visual_quality_status": RESTORED_BASELINE,
                "assignment_count": 596,
                "applied_count": 596,
                "project_material_audit": str(audit),
                "sealed_qwen_mvinverse_evidence": str(evidence),
                "preview_rendered_registry": str(registry),
            }
        ),
    )

    manifest = VIEWER.build_viewer_manifest(tmp_path)

    assert manifest["visual_quality_raw_status"] == RESTORED_BASELINE
    assert manifest["visual_quality_gate_status"] == RESTORED_BASELINE
    assert manifest["material_stage_accepted"] is True
    assert manifest["visual_quality_resolution"]["state"] == (
        "SEALED_HISTORICAL_BASELINE"
    )
    assert manifest["source"]["restored_project"] is True
    assert manifest["source"]["legacy_compatibility"] is False
    assert manifest["source"]["preview_images"] == {
        "iso": "delivery/visual_material/preview_final/rgb/iso.png"
    }
    assert "封存恢复" in manifest["note_zh"]


def test_viewer_exposes_all_quality_gate_fields_once() -> None:
    index = (
        Path(__file__).parents[1] / "web" / "result_viewer" / "index.html"
    ).read_text(encoding="utf-8")
    parser = _IdCollector()
    parser.feed(index)

    expected = {
        "quality-raw-status",
        "quality-gate-status",
        "quality-resolution-state",
        "quality-limitation-count",
        "quality-resolution-link",
        "limitation-list",
    }
    assert expected <= set(parser.ids)
    assert len(parser.ids) == len(set(parser.ids))
    assert 'fetch("viewer_manifest.json"' in index
    assert "manifest.source?.preview_images" in index
    assert "Object.keys(recordedPreviews).length > 0" in index
    assert "previewContainer.replaceChildren()" in index
    assert "for (const view of Object.keys(recordedPreviews))" in index
    assert "manifest.source?.preview_fallback_allowed !== false" in index
    assert "visual_quality_appearance_candidate" not in index
