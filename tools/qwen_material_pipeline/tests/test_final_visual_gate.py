from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from qwen_material_pipeline.evidence.final_visual_gate import (
    ABSOLUTE_PASS_MODE,
    COMPLETED,
    FAILED_COMPLETION_STATE,
    FAIL_CLOSED,
    IMMUTABLE_LIBRARY_OPTIMUM_MODE,
    IMMUTABLE_LIBRARY_RENDER_REPEATABILITY_TOLERANCE,
    PASS,
    SEALED_BASELINE_PRESERVATION_MODE,
    FinalVisualGateError,
    main,
    require_final_visual_gate_passed,
    run_final_visual_gate,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _render_view(
    *,
    reference_view_id: str,
    render_view_id: str,
    rgb: Path,
    part_ids: Path,
    color: float,
    texture: float,
    appearance: float,
    recall: float = 0.9,
    observed_share: float = 0.2,
    status: str = PASS,
) -> dict:
    return {
        "reference_view_id": reference_view_id,
        "render_view_id": render_view_id,
        "status": status,
        "alignment": {
            "score": 0.9,
            "silhouette_iou": 0.88,
        },
        "render": {
            "image": str(rgb.resolve()),
            "image_sha256": _sha256(rgb),
            "part_ids": str(part_ids.resolve()),
            "part_ids_sha256": _sha256(part_ids),
        },
        "material_color": {
            "score": color,
            "trusted_evidence_group_recall": {
                "groups": [
                    {
                        "group_id": "G01",
                        "base_colors": ["green"],
                        "reference_evidence_weight": 400,
                        "reference_group_share": 0.2,
                        "observed_render_share": observed_share,
                        "recall": recall,
                        "delivery_presence_status": "PRESENT",
                    }
                ]
            },
        },
        "material_texture": {"score": texture},
        "material_appearance_score": appearance,
    }


def _build_registry(
    root: Path,
    *,
    asset: Path,
    prefix: str,
) -> tuple[Path, dict[str, tuple[Path, Path]]]:
    render_dir = root / prefix / "renders"
    render_dir.mkdir(parents=True)
    paths: dict[str, tuple[Path, Path]] = {}
    views = []
    for view_id in ("right", "front"):
        rgb = render_dir / "rgb" / f"{view_id}.png"
        part_ids = render_dir / "part_ids" / f"{view_id}.png"
        rgb.parent.mkdir(parents=True, exist_ok=True)
        part_ids.parent.mkdir(parents=True, exist_ok=True)
        rgb.write_bytes(f"{prefix}-{view_id}-rgb".encode())
        part_ids.write_bytes(f"{prefix}-{view_id}-ids".encode())
        paths[view_id] = (rgb, part_ids)
        views.append(
            {
                "view_id": view_id,
                "rgb": str(rgb.resolve()),
                "part_ids": str(part_ids.resolve()),
            }
        )
    registry = {
        "schema_version": "qwen-material-parts/v1",
        "asset_usd": str(asset.resolve()),
        "asset_sha256": _sha256(asset),
        "render_set": {
            "asset_usd": str(asset.resolve()),
            "resolution": [512, 512],
            "analysis_up_axis": [0.0, 0.0, 1.0],
            "analysis_front_axis": [0.0, -1.0, 0.0],
            "lighting_profile": "material-neutral",
            "requested_view_tokens": ["right", "front"],
            "rt_subframes": 4,
            "views": views,
        },
    }
    registry_path = render_dir / "part_registry.rendered.json"
    _write_json(registry_path, registry)
    return registry_path, paths


def _build_quality(
    path: Path,
    *,
    reference_manifest: Path,
    registry: Path,
    render_paths: dict[str, tuple[Path, Path]],
    color: float,
    texture: float,
    appearance: float,
) -> dict:
    views = [
        _render_view(
            reference_view_id="front_ref",
            render_view_id="right",
            rgb=render_paths["right"][0],
            part_ids=render_paths["right"][1],
            color=color,
            texture=texture,
            appearance=appearance,
        ),
        _render_view(
            reference_view_id="side_ref",
            render_view_id="front",
            rgb=render_paths["front"][0],
            part_ids=render_paths["front"][1],
            color=color,
            texture=texture,
            appearance=appearance,
        ),
    ]
    quality = {
        "schema_version": "qwen-reference-render-comparison/v1",
        "inputs": {
            "reference_manifest": str(reference_manifest.resolve()),
            "reference_manifest_sha256": _sha256(reference_manifest),
            "rendered_registry": str(registry.resolve()),
            "rendered_registry_sha256": _sha256(registry),
            "selected_view_mapping": {
                "front_ref": "right",
                "side_ref": "front",
            },
        },
        "thresholds": {
            "pass_color_score": 0.62,
            "minimum_comparable_views": 2,
        },
        "aggregate": {
            "status": PASS,
            "material_color_score": color,
            "material_texture_score": texture,
            "material_appearance_score": appearance,
            "failed_view_count": 0,
            "review_view_count": 0,
            "unscorable_view_count": 0,
            "comparable_view_count": 2,
        },
        "views": views,
    }
    _write_json(path, quality)
    return quality


def _bundle(tmp_path: Path) -> dict[str, Path | dict]:
    reference_manifest = tmp_path / "reference_manifest.json"
    _write_json(
        reference_manifest,
        {
            "schema_version": "qwen-reference-manifest/v1",
            "views": ["front_ref", "side_ref"],
        },
    )
    baseline_asset = tmp_path / "baseline.usda"
    collected_asset = tmp_path / "delivery" / "asset_phys.usda"
    baseline_asset.write_text("#usda baseline\n", encoding="utf-8")
    collected_asset.parent.mkdir()
    collected_asset.write_text("#usda collected\n", encoding="utf-8")

    baseline_registry, baseline_paths = _build_registry(
        tmp_path,
        asset=baseline_asset,
        prefix="baseline",
    )
    final_registry, final_paths = _build_registry(
        tmp_path,
        asset=collected_asset,
        prefix="final",
    )
    baseline_quality_path = tmp_path / "baseline" / "quality.json"
    final_quality_path = tmp_path / "final" / "quality.json"
    baseline_quality = _build_quality(
        baseline_quality_path,
        reference_manifest=reference_manifest,
        registry=baseline_registry,
        render_paths=baseline_paths,
        color=0.8,
        texture=0.76,
        appearance=0.78,
    )
    final_quality = _build_quality(
        final_quality_path,
        reference_manifest=reference_manifest,
        registry=final_registry,
        render_paths=final_paths,
        color=0.82,
        texture=0.78,
        appearance=0.80,
    )
    return {
        "collected": collected_asset,
        "reference_manifest": reference_manifest,
        "baseline_registry": baseline_registry,
        "final_registry": final_registry,
        "baseline_quality_path": baseline_quality_path,
        "final_quality_path": final_quality_path,
        "baseline_quality": baseline_quality,
        "final_quality": final_quality,
    }


def _run(bundle: dict[str, Path | dict], output: Path) -> dict:
    return run_final_visual_gate(
        collected_usd=bundle["collected"],
        baseline_quality_report=bundle["baseline_quality_path"],
        final_quality_report=bundle["final_quality_path"],
        baseline_rendered_registry=bundle["baseline_registry"],
        final_rendered_registry=bundle["final_registry"],
        output=output,
    )


def _rewrite_final_quality(bundle: dict[str, Path | dict], quality: dict) -> None:
    path = bundle["final_quality_path"]
    assert isinstance(path, Path)
    _write_json(path, quality)


def _set_pixel_quantized_group_recall(
    bundle: dict[str, Path | dict],
    *,
    baseline_observed_pixels: int,
    final_observed_pixels: int,
    sampled_pixels: int = 73_613,
    required_render_share: float = 0.0026405244627898507,
    reference_evidence_pixels: int = 435,
) -> None:
    baseline_quality = copy.deepcopy(bundle["baseline_quality"])
    final_quality = copy.deepcopy(bundle["final_quality"])
    assert isinstance(baseline_quality, dict)
    assert isinstance(final_quality, dict)
    for quality, observed_pixels in (
        (baseline_quality, baseline_observed_pixels),
        (final_quality, final_observed_pixels),
    ):
        view = quality["views"][0]
        view["material_color"]["render_distribution"] = {
            "sampled_pixels": sampled_pixels
        }
        group = view["material_color"]["trusted_evidence_group_recall"]["groups"][0]
        observed_share = observed_pixels / sampled_pixels
        group.update(
            {
                "reference_evidence_weight": reference_evidence_pixels,
                "required_render_share": required_render_share,
                "observed_render_share": observed_share,
                "recall": min(1.0, observed_share / required_render_share),
            }
        )
    baseline_path = bundle["baseline_quality_path"]
    final_path = bundle["final_quality_path"]
    assert isinstance(baseline_path, Path)
    assert isinstance(final_path, Path)
    _write_json(baseline_path, baseline_quality)
    _write_json(final_path, final_quality)


def _sealed_evidence(tmp_path: Path, *, live_repeated: bool = False) -> Path:
    template = tmp_path / "sealed" / "template.json"
    catalog = tmp_path / "sealed" / "catalog.json"
    dependency_lock = tmp_path / "sealed" / "dependency-lock.json"
    _write_json(template, {"template": "accepted"})
    _write_json(catalog, {"catalog": "accepted"})
    _write_json(dependency_lock, {"dependencies": "accepted"})
    method = "sealed_fixture_library_default_mdl_result"
    project = tmp_path / "sealed" / "project.json"
    _write_json(
        project,
        {
            "schema_version": "qwen-material-project/v2",
            "asset_id": "sealed-fixture",
            "template": template.name,
            "template_sha256": _sha256(template),
            "catalog": catalog.name,
            "catalog_sha256": _sha256(catalog),
            "dependency_lock": dependency_lock.name,
            "dependency_lock_sha256": _sha256(dependency_lock),
            "references": [
                {"role": "front_ref"},
                {"role": "side_ref"},
            ],
            "acceptance": {
                "render": {
                    "resolution": 512,
                    "views": "right,front",
                    "rt_subframes": 4,
                    "lighting_profile": "material-neutral",
                    "analysis_up_axis": "z",
                    "analysis_front_axis": "-y",
                },
                "view_mapping": {
                    "front_ref": "right",
                    "side_ref": "front",
                },
                "minimum_comparable_views": 2,
            },
            "evidence": {
                "method": method,
                "historical_result_sha256": "1" * 64,
            },
        },
    )
    verification = {
        "schema_version": "qwen-sealed-material-dependency-verification/v1",
        "status": "PASS",
        "dependency_lock_verified": True,
        "lock_path": str(dependency_lock),
        "lock_sha256": _sha256(dependency_lock),
        "catalog_path": str(catalog),
        "material_root": str(tmp_path / "materials"),
    }
    verification_report = tmp_path / "sealed" / "dependency-verification.json"
    _write_json(verification_report, verification)
    evidence = tmp_path / "sealed" / "evidence.json"
    _write_json(
        evidence,
        {
            "schema_version": "qwen-bundled-project-evidence/v1",
            "asset_id": "sealed-fixture",
            "method": method,
            "historical_result_sha256": "1" * 64,
            "project": str(project),
            "project_sha256": _sha256(project),
            "source_cad_sha256": "2" * 64,
            "reference_sha256": {
                "front_ref": "3" * 64,
                "side_ref": "4" * 64,
            },
            "template": str(template),
            "template_sha256": _sha256(template),
            "catalog": str(catalog),
            "catalog_sha256": _sha256(catalog),
            "dependency_lock_verified": True,
            "dependency_lock": str(dependency_lock),
            "dependency_lock_sha256": _sha256(dependency_lock),
            "dependency_lock_verification": verification,
            "dependency_lock_verification_status": "PASS",
            "dependency_lock_verification_report": str(verification_report),
            "dependency_lock_verification_report_sha256": _sha256(verification_report),
            "plan_sha256": "5" * 64,
            "audit_sha256": "6" * 64,
            "live_inference_repeated": live_repeated,
            "replay_policy": "hash-bound exact sealed-project replay",
        },
    )
    return evidence


def _set_non_pass_without_regression(bundle: dict[str, Path | dict]) -> None:
    for key in ("baseline_quality", "final_quality"):
        quality = copy.deepcopy(bundle[key])
        assert isinstance(quality, dict)
        quality["aggregate"].update(
            {
                "status": "FAIL",
                "material_color_score": 0.4,
                "material_texture_score": 0.4,
                "material_appearance_score": 0.4,
                "failed_view_count": 2,
                "review_view_count": 0,
                "unscorable_view_count": 0,
                "comparable_view_count": 2,
            }
        )
        for view in quality["views"]:
            view["status"] = "FAIL"
            view["material_color"]["score"] = 0.4
            view["material_texture"]["score"] = 0.4
            view["material_appearance_score"] = 0.4
            group = view["material_color"]["trusted_evidence_group_recall"]["groups"][0]
            group["recall"] = 0.0
            group["observed_render_share"] = 0.0
            group["delivery_presence_status"] = "MISSING"
        path_key = (
            "baseline_quality_path"
            if key == "baseline_quality"
            else "final_quality_path"
        )
        path = bundle[path_key]
        assert isinstance(path, Path)
        _write_json(path, quality)


def _rewrite_registry_contract(
    bundle: dict[str, Path | dict],
    *,
    prefix: str,
    update: Callable[[dict], None],
) -> None:
    registry_key = f"{prefix}_registry"
    quality_key = f"{prefix}_quality_path"
    registry_path = bundle[registry_key]
    quality_path = bundle[quality_key]
    assert isinstance(registry_path, Path)
    assert isinstance(quality_path, Path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    update(registry["render_set"])
    _write_json(registry_path, registry)
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["inputs"]["rendered_registry_sha256"] = _sha256(registry_path)
    _write_json(quality_path, quality)


def _rewrite_quality_contract(
    bundle: dict[str, Path | dict],
    *,
    prefix: str,
    update: Callable[[dict], None],
) -> None:
    quality_path = bundle[f"{prefix}_quality_path"]
    assert isinstance(quality_path, Path)
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    update(quality)
    _write_json(quality_path, quality)


def test_accepts_only_independent_non_regressing_collected_render(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)

    report = _run(bundle, tmp_path / "gate.json")

    assert report["status"] == PASS
    assert report["completion_allowed"] is True
    assert report["completion_state"] == COMPLETED
    assert report["provenance"]["independent_final_render_verified"] is True
    assert report["provenance"]["collected_asset_hash_verified"] is True
    assert report["summary"] == {
        "view_count": 2,
        "passed_view_count": 2,
        "significant_group_count": 2,
        "passed_significant_group_count": 2,
        "failure_count": 0,
    }
    require_final_visual_gate_passed(report)


def test_immutable_library_optimum_accepts_photometric_review_only(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    for prefix in ("baseline", "final"):
        quality = copy.deepcopy(bundle[f"{prefix}_quality"])
        assert isinstance(quality, dict)
        quality["aggregate"].update(
            {
                "status": "REVIEW",
                "failed_view_count": 0,
                "review_view_count": 2,
                "unscorable_view_count": 0,
            }
        )
        for view in quality["views"]:
            view["status"] = "REVIEW"
            view["reasons"] = [
                "foreground_value_similarity_below_pass_threshold"
            ]
            view["material_texture"]["status"] = PASS
        quality_path = bundle[f"{prefix}_quality_path"]
        assert isinstance(quality_path, Path)
        _write_json(quality_path, quality)

    default_report = _run(bundle, tmp_path / "default-gate.json")
    assert default_report["status"] == FAIL_CLOSED

    report = run_final_visual_gate(
        collected_usd=bundle["collected"],
        baseline_quality_report=bundle["baseline_quality_path"],
        final_quality_report=bundle["final_quality_path"],
        baseline_rendered_registry=bundle["baseline_registry"],
        final_rendered_registry=bundle["final_registry"],
        output=tmp_path / "optimum-gate.json",
        allow_immutable_library_optimum_review=True,
    )

    assert report["status"] == PASS
    assert report["completion_allowed"] is True
    assert report["policy"]["acceptance_mode"] == IMMUTABLE_LIBRARY_OPTIMUM_MODE
    assert report["policy"]["immutable_library_review_allowed"] is True
    assert report["summary"]["passed_view_count"] == 2


def test_immutable_library_optimum_tolerates_bounded_rtx_repeatability_noise(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    jitter = IMMUTABLE_LIBRARY_RENDER_REPEATABILITY_TOLERANCE + 0.05
    for prefix in ("baseline", "final"):
        quality = copy.deepcopy(bundle[f"{prefix}_quality"])
        assert isinstance(quality, dict)
        quality["aggregate"].update(
            {
                "status": "REVIEW",
                "failed_view_count": 0,
                "review_view_count": 2,
                "unscorable_view_count": 0,
            }
        )
        for view in quality["views"]:
            view["status"] = "REVIEW"
            view["reasons"] = [
                "foreground_value_similarity_below_pass_threshold"
            ]
            view["material_texture"]["status"] = PASS
            group = view["material_color"]["trusted_evidence_group_recall"][
                "groups"
            ][0]
            if prefix == "baseline":
                group["recall"] = 0.99
                group["observed_render_share"] = 0.2
            else:
                group["recall"] = 0.9
                group["observed_render_share"] = 0.25
        if prefix == "baseline":
            quality["aggregate"]["material_texture_score"] += jitter
            quality["aggregate"]["material_appearance_score"] += jitter
            for view in quality["views"]:
                view["material_texture"]["score"] += jitter
                view["material_appearance_score"] += jitter
        quality_path = bundle[f"{prefix}_quality_path"]
        assert isinstance(quality_path, Path)
        _write_json(quality_path, quality)

    report = run_final_visual_gate(
        collected_usd=bundle["collected"],
        baseline_quality_report=bundle["baseline_quality_path"],
        final_quality_report=bundle["final_quality_path"],
        baseline_rendered_registry=bundle["baseline_registry"],
        final_rendered_registry=bundle["final_registry"],
        output=tmp_path / "repeatability-gate.json",
        maximum_score_regression=0.01,
        maximum_group_recall_regression=0.01,
        allow_immutable_library_optimum_review=True,
    )

    assert report["status"] == PASS
    assert report["policy"]["configured_maximum_score_regression"] == 0.01
    assert (
        report["policy"]["maximum_score_regression"]
        == IMMUTABLE_LIBRARY_RENDER_REPEATABILITY_TOLERANCE
    )
    assert report["policy"]["relative_nonregression_enforced"] is False
    assert report["aggregate"]["regressed_metrics"] == []
    assert report["aggregate"]["observed_regressed_metrics"]


def test_immutable_library_optimum_rejects_nonphotometric_review(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    final_quality = copy.deepcopy(bundle["final_quality"])
    assert isinstance(final_quality, dict)
    final_quality["aggregate"].update(
        {
            "status": "REVIEW",
            "failed_view_count": 0,
            "review_view_count": 2,
            "unscorable_view_count": 0,
        }
    )
    for view in final_quality["views"]:
        view["status"] = "REVIEW"
        view["reasons"] = ["trusted_color_group_missing"]
        view["material_texture"]["status"] = PASS
    _rewrite_final_quality(bundle, final_quality)

    report = run_final_visual_gate(
        collected_usd=bundle["collected"],
        baseline_quality_report=bundle["baseline_quality_path"],
        final_quality_report=bundle["final_quality_path"],
        baseline_rendered_registry=bundle["baseline_registry"],
        final_rendered_registry=bundle["final_registry"],
        output=tmp_path / "gate.json",
        allow_immutable_library_optimum_review=True,
    )

    assert report["status"] == FAIL_CLOSED
    assert "FINAL_VIEW_STATUS_NOT_PASS" in report["reason_codes"]


def test_sealed_contract_accepts_independent_absolute_pass(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    evidence = _sealed_evidence(tmp_path)

    report = run_final_visual_gate(
        collected_usd=bundle["collected"],
        baseline_quality_report=bundle["baseline_quality_path"],
        final_quality_report=bundle["final_quality_path"],
        baseline_rendered_registry=bundle["baseline_registry"],
        final_rendered_registry=bundle["final_registry"],
        output=tmp_path / "gate.json",
        sealed_baseline_evidence=evidence,
    )

    assert report["status"] == PASS
    assert report["completion_allowed"] is True
    assert report["policy"]["acceptance_mode"] == (SEALED_BASELINE_PRESERVATION_MODE)
    assert report["policy"]["absolute_quality_floors_enforced"] is True
    assert report["policy"]["sealed_contract_absolute_pass_required"] is True
    assert report["provenance"]["sealed_baseline_evidence_verified"] is True
    assert report["provenance"]["independent_final_render_verified"] is True
    assert report["provenance"]["sealed_acceptance_contract"]["acceptance"] == {
        "render": {
            "resolution": 512,
            "views": "right,front",
            "rt_subframes": 4,
            "lighting_profile": "material-neutral",
            "analysis_up_axis": "z",
            "analysis_front_axis": "-y",
        },
        "view_mapping": {"front_ref": "right", "side_ref": "front"},
        "minimum_comparable_views": 2,
    }
    assert report["summary"]["passed_view_count"] == 2


def test_sealed_contract_rejects_identical_failed_baseline_and_final(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    _set_non_pass_without_regression(bundle)
    evidence = _sealed_evidence(tmp_path)

    report = run_final_visual_gate(
        collected_usd=bundle["collected"],
        baseline_quality_report=bundle["baseline_quality_path"],
        final_quality_report=bundle["final_quality_path"],
        baseline_rendered_registry=bundle["baseline_registry"],
        final_rendered_registry=bundle["final_registry"],
        output=tmp_path / "gate.json",
        sealed_baseline_evidence=evidence,
    )

    assert report["status"] == FAIL_CLOSED
    assert report["completion_allowed"] is False
    assert "BASELINE_AGGREGATE_STATUS_NOT_PASS" in report["reason_codes"]
    assert "FINAL_AGGREGATE_STATUS_NOT_PASS" in report["reason_codes"]
    assert "BASELINE_VIEW_STATUS_NOT_PASS" in report["reason_codes"]
    assert "FINAL_VIEW_STATUS_NOT_PASS" in report["reason_codes"]


def test_sealed_contract_rejects_tampered_rt_subframes(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    evidence = _sealed_evidence(tmp_path)
    for prefix in ("baseline", "final"):
        _rewrite_registry_contract(
            bundle,
            prefix=prefix,
            update=lambda render_set: render_set.__setitem__("rt_subframes", 2),
        )

    report = run_final_visual_gate(
        collected_usd=bundle["collected"],
        baseline_quality_report=bundle["baseline_quality_path"],
        final_quality_report=bundle["final_quality_path"],
        baseline_rendered_registry=bundle["baseline_registry"],
        final_rendered_registry=bundle["final_registry"],
        output=tmp_path / "gate.json",
        sealed_baseline_evidence=evidence,
    )

    assert report["status"] == FAIL_CLOSED
    assert "rt_subframes" in report["error"]


def test_sealed_contract_rejects_tampered_view_mapping(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    evidence = _sealed_evidence(tmp_path)
    for prefix in ("baseline", "final"):
        _rewrite_quality_contract(
            bundle,
            prefix=prefix,
            update=lambda quality: quality["inputs"].__setitem__(
                "selected_view_mapping",
                {"front_ref": "front", "side_ref": "right"},
            ),
        )

    report = run_final_visual_gate(
        collected_usd=bundle["collected"],
        baseline_quality_report=bundle["baseline_quality_path"],
        final_quality_report=bundle["final_quality_path"],
        baseline_rendered_registry=bundle["baseline_registry"],
        final_rendered_registry=bundle["final_registry"],
        output=tmp_path / "gate.json",
        sealed_baseline_evidence=evidence,
    )

    assert report["status"] == FAIL_CLOSED
    assert "view mapping" in report["error"]


def test_sealed_contract_rejects_tampered_minimum_view_count(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    evidence = _sealed_evidence(tmp_path)
    for prefix in ("baseline", "final"):
        _rewrite_quality_contract(
            bundle,
            prefix=prefix,
            update=lambda quality: quality["thresholds"].__setitem__(
                "minimum_comparable_views",
                1,
            ),
        )

    report = run_final_visual_gate(
        collected_usd=bundle["collected"],
        baseline_quality_report=bundle["baseline_quality_path"],
        final_quality_report=bundle["final_quality_path"],
        baseline_rendered_registry=bundle["baseline_registry"],
        final_rendered_registry=bundle["final_registry"],
        output=tmp_path / "gate.json",
        sealed_baseline_evidence=evidence,
    )

    assert report["status"] == FAIL_CLOSED
    assert "minimum comparable views" in report["error"]


def test_sealed_contract_rejects_incomplete_rendered_view_cover(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    evidence = _sealed_evidence(tmp_path)
    for prefix in ("baseline", "final"):
        _rewrite_registry_contract(
            bundle,
            prefix=prefix,
            update=lambda render_set: render_set["views"].pop(),
        )

    report = run_final_visual_gate(
        collected_usd=bundle["collected"],
        baseline_quality_report=bundle["baseline_quality_path"],
        final_quality_report=bundle["final_quality_path"],
        baseline_rendered_registry=bundle["baseline_registry"],
        final_rendered_registry=bundle["final_registry"],
        output=tmp_path / "gate.json",
        sealed_baseline_evidence=evidence,
    )

    assert report["status"] == FAIL_CLOSED
    assert "do not exactly cover" in report["error"]


def test_sealed_contract_rejects_reference_role_order_tampering(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    evidence = _sealed_evidence(tmp_path)
    reference_manifest = bundle["reference_manifest"]
    assert isinstance(reference_manifest, Path)
    manifest = json.loads(reference_manifest.read_text(encoding="utf-8"))
    manifest["views"].reverse()
    _write_json(reference_manifest, manifest)
    for prefix in ("baseline", "final"):
        _rewrite_quality_contract(
            bundle,
            prefix=prefix,
            update=lambda quality: quality["inputs"].__setitem__(
                "reference_manifest_sha256",
                _sha256(reference_manifest),
            ),
        )

    report = run_final_visual_gate(
        collected_usd=bundle["collected"],
        baseline_quality_report=bundle["baseline_quality_path"],
        final_quality_report=bundle["final_quality_path"],
        baseline_rendered_registry=bundle["baseline_registry"],
        final_rendered_registry=bundle["final_registry"],
        output=tmp_path / "gate.json",
        sealed_baseline_evidence=evidence,
    )

    assert report["status"] == FAIL_CLOSED
    assert "roles/order" in report["error"]


def test_non_pass_reports_cannot_use_preservation_without_sealed_evidence(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    _set_non_pass_without_regression(bundle)

    report = _run(bundle, tmp_path / "gate.json")

    assert report["status"] == FAIL_CLOSED
    assert report["policy"]["acceptance_mode"] == ABSOLUTE_PASS_MODE
    assert report["policy"]["absolute_quality_floors_enforced"] is True
    assert "FINAL_AGGREGATE_STATUS_NOT_PASS" in report["reason_codes"]
    assert "FINAL_VIEW_STATUS_NOT_PASS" in report["reason_codes"]


def test_invalid_sealed_evidence_fails_closed(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    _set_non_pass_without_regression(bundle)
    evidence = _sealed_evidence(tmp_path, live_repeated=True)

    report = run_final_visual_gate(
        collected_usd=bundle["collected"],
        baseline_quality_report=bundle["baseline_quality_path"],
        final_quality_report=bundle["final_quality_path"],
        baseline_rendered_registry=bundle["baseline_registry"],
        final_rendered_registry=bundle["final_registry"],
        output=tmp_path / "gate.json",
        sealed_baseline_evidence=evidence,
    )

    assert report["status"] == FAIL_CLOSED
    assert report["completion_allowed"] is False
    assert report["reason_codes"] == ["INVALID_OR_UNVERIFIED_FINAL_VISUAL_EVIDENCE"]
    assert report["provenance"]["sealed_baseline_evidence_verified"] is False


def test_sealed_method_must_match_hash_bound_project(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    _set_non_pass_without_regression(bundle)
    evidence = _sealed_evidence(tmp_path)
    document = json.loads(evidence.read_text(encoding="utf-8"))
    document["method"] = "forged_sealed_method"
    _write_json(evidence, document)

    report = run_final_visual_gate(
        collected_usd=bundle["collected"],
        baseline_quality_report=bundle["baseline_quality_path"],
        final_quality_report=bundle["final_quality_path"],
        baseline_rendered_registry=bundle["baseline_registry"],
        final_rendered_registry=bundle["final_registry"],
        output=tmp_path / "gate.json",
        sealed_baseline_evidence=evidence,
    )

    assert report["status"] == FAIL_CLOSED
    assert report["completion_allowed"] is False
    assert report["reason_codes"] == ["INVALID_OR_UNVERIFIED_FINAL_VISUAL_EVIDENCE"]


def test_stale_dependency_lock_cannot_enable_preservation(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    _set_non_pass_without_regression(bundle)
    evidence = _sealed_evidence(tmp_path)
    evidence_document = json.loads(evidence.read_text(encoding="utf-8"))
    Path(evidence_document["dependency_lock"]).write_text(
        '{"dependencies":"changed"}',
        encoding="utf-8",
    )

    report = run_final_visual_gate(
        collected_usd=bundle["collected"],
        baseline_quality_report=bundle["baseline_quality_path"],
        final_quality_report=bundle["final_quality_path"],
        baseline_rendered_registry=bundle["baseline_registry"],
        final_rendered_registry=bundle["final_registry"],
        output=tmp_path / "gate.json",
        sealed_baseline_evidence=evidence,
    )

    assert report["status"] == FAIL_CLOSED
    assert report["completion_allowed"] is False
    assert report["reason_codes"] == ["INVALID_OR_UNVERIFIED_FINAL_VISUAL_EVIDENCE"]


def test_sealed_preservation_still_rejects_visual_and_group_regression(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    evidence = _sealed_evidence(tmp_path)
    final_quality = copy.deepcopy(bundle["final_quality"])
    assert isinstance(final_quality, dict)
    final_quality["views"][0]["material_texture"]["score"] = 0.4
    group = final_quality["views"][1]["material_color"][
        "trusted_evidence_group_recall"
    ]["groups"][0]
    group["delivery_presence_status"] = "MISSING"
    group["recall"] = 0.0
    _rewrite_final_quality(bundle, final_quality)

    report = run_final_visual_gate(
        collected_usd=bundle["collected"],
        baseline_quality_report=bundle["baseline_quality_path"],
        final_quality_report=bundle["final_quality_path"],
        baseline_rendered_registry=bundle["baseline_registry"],
        final_rendered_registry=bundle["final_registry"],
        output=tmp_path / "gate.json",
        sealed_baseline_evidence=evidence,
    )

    assert report["status"] == FAIL_CLOSED
    assert "PER_VIEW_VISUAL_SCORE_REGRESSION" in report["reason_codes"]
    assert "SIGNIFICANT_GROUP_RECALL_REGRESSION" in report["reason_codes"]
    assert "SIGNIFICANT_GROUP_PRESENCE_REGRESSION" in report["reason_codes"]


def test_per_view_visual_regression_blocks_completion(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    final_quality = copy.deepcopy(bundle["final_quality"])
    assert isinstance(final_quality, dict)
    final_quality["views"][0]["material_texture"]["score"] = 0.5
    final_quality["views"][0]["material_appearance_score"] = 0.6
    _rewrite_final_quality(bundle, final_quality)

    report = _run(bundle, tmp_path / "gate.json")

    assert report["status"] == FAIL_CLOSED
    assert report["completion_allowed"] is False
    assert report["completion_state"] == FAILED_COMPLETION_STATE
    assert "PER_VIEW_VISUAL_SCORE_REGRESSION" in report["reason_codes"]
    assert report["views"][0]["status"] == FAIL_CLOSED
    with pytest.raises(FinalVisualGateError):
        require_final_visual_gate_passed(report)


def test_significant_reference_group_regression_blocks_completion(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    final_quality = copy.deepcopy(bundle["final_quality"])
    assert isinstance(final_quality, dict)
    group = final_quality["views"][1]["material_color"][
        "trusted_evidence_group_recall"
    ]["groups"][0]
    group["recall"] = 0.6
    group["observed_render_share"] = 0.45
    _rewrite_final_quality(bundle, final_quality)

    report = _run(bundle, tmp_path / "gate.json")

    assert report["status"] == FAIL_CLOSED
    assert "SIGNIFICANT_GROUP_RECALL_REGRESSION" in report["reason_codes"]
    assert "SIGNIFICANT_GROUP_SHARE_ERROR_REGRESSION" in report["reason_codes"]
    failed_group = report["views"][1]["significant_groups"][0]
    assert failed_group["status"] == FAIL_CLOSED


def test_two_pixel_recall_boundary_noise_uses_integer_quantization_gate(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    _set_pixel_quantized_group_recall(
        bundle,
        baseline_observed_pixels=189,
        final_observed_pixels=187,
    )

    report = _run(bundle, tmp_path / "gate.json")

    assert report["status"] == PASS
    front = next(
        view for view in report["views"] if view["reference_view_id"] == "front_ref"
    )
    group = front["significant_groups"][0]
    assert group["recall_delta"] == pytest.approx(-0.010289287046531426)
    audit = group["recall_quantization_audit"]
    assert audit["raw_recall_regression"] is True
    assert audit["applicable"] is True
    assert audit["tolerance_applied"] is True
    assert audit["baseline_observed_pixels"] == 189
    assert audit["final_observed_pixels"] == 187
    assert audit["observed_pixel_loss"] == 2
    assert audit["maximum_allowed_pixel_loss"] == 2
    assert audit["reason"] == "WITHIN_INTEGER_PIXEL_QUANTIZATION_BOUNDARY"


def test_integer_quantization_gate_rejects_true_three_pixel_regression(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    _set_pixel_quantized_group_recall(
        bundle,
        baseline_observed_pixels=189,
        final_observed_pixels=186,
    )

    report = _run(bundle, tmp_path / "gate.json")

    assert report["status"] == FAIL_CLOSED
    assert "SIGNIFICANT_GROUP_RECALL_REGRESSION" in report["reason_codes"]
    group = report["views"][0]["significant_groups"][0]
    audit = group["recall_quantization_audit"]
    assert audit["tolerance_applied"] is False
    assert audit["observed_pixel_loss"] == 3
    assert audit["maximum_allowed_pixel_loss"] == 2
    assert audit["reason"] == "INTEGER_PIXEL_LOSS_EXCEEDS_BOUNDARY"


def test_integer_quantization_tolerance_requires_significant_evidence_pixels(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    _set_pixel_quantized_group_recall(
        bundle,
        baseline_observed_pixels=189,
        final_observed_pixels=187,
        reference_evidence_pixels=127,
    )

    report = _run(bundle, tmp_path / "gate.json")

    assert report["status"] == FAIL_CLOSED
    assert "SIGNIFICANT_GROUP_RECALL_REGRESSION" in report["reason_codes"]
    group = report["views"][0]["significant_groups"][0]
    audit = group["recall_quantization_audit"]
    assert audit["tolerance_applied"] is False
    assert audit["reason"] == ("INSUFFICIENT_OR_CHANGED_REFERENCE_EVIDENCE_PIXELS")


def test_integer_quantization_rejects_recall_inconsistent_with_pixel_shares(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    _set_pixel_quantized_group_recall(
        bundle,
        baseline_observed_pixels=189,
        final_observed_pixels=187,
    )
    final_path = bundle["final_quality_path"]
    assert isinstance(final_path, Path)
    final_quality = json.loads(final_path.read_text(encoding="utf-8"))
    final_quality["views"][0]["material_color"]["trusted_evidence_group_recall"][
        "groups"
    ][0]["recall"] = 0.95
    _write_json(final_path, final_quality)

    report = _run(bundle, tmp_path / "gate.json")

    assert report["status"] == FAIL_CLOSED
    assert "SIGNIFICANT_GROUP_RECALL_REGRESSION" in report["reason_codes"]
    group = report["views"][0]["significant_groups"][0]
    audit = group["recall_quantization_audit"]
    assert audit["tolerance_applied"] is False
    assert audit["reason"] == ("REPORTED_RECALL_INCONSISTENT_WITH_PIXEL_SHARES")


def test_candidate_registry_cannot_be_reused_as_final_evidence(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)

    report = run_final_visual_gate(
        collected_usd=bundle["collected"],
        baseline_quality_report=bundle["baseline_quality_path"],
        final_quality_report=bundle["baseline_quality_path"],
        baseline_rendered_registry=bundle["baseline_registry"],
        final_rendered_registry=bundle["baseline_registry"],
        output=tmp_path / "gate.json",
    )

    assert report["status"] == FAIL_CLOSED
    assert report["completion_allowed"] is False
    assert report["completion_state"] == FAILED_COMPLETION_STATE
    assert report["provenance"]["independent_final_render_verified"] is False
    assert report["reason_codes"] == ["INVALID_OR_UNVERIFIED_FINAL_VISUAL_EVIDENCE"]


def test_independent_locked_rerender_may_share_the_same_immutable_asset(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    collected = bundle["collected"]
    baseline_registry_path = bundle["baseline_registry"]
    baseline_quality_path = bundle["baseline_quality_path"]
    assert isinstance(collected, Path)
    assert isinstance(baseline_registry_path, Path)
    assert isinstance(baseline_quality_path, Path)

    baseline_registry = json.loads(baseline_registry_path.read_text())
    baseline_registry["asset_usd"] = str(collected.resolve())
    baseline_registry["asset_sha256"] = _sha256(collected)
    baseline_registry["render_set"]["asset_usd"] = str(collected.resolve())
    _write_json(baseline_registry_path, baseline_registry)
    baseline_quality = json.loads(baseline_quality_path.read_text())
    baseline_quality["inputs"]["rendered_registry_sha256"] = _sha256(
        baseline_registry_path
    )
    _write_json(baseline_quality_path, baseline_quality)

    report = run_final_visual_gate(
        collected_usd=collected,
        baseline_quality_report=baseline_quality_path,
        final_quality_report=bundle["final_quality_path"],
        baseline_rendered_registry=baseline_registry_path,
        final_rendered_registry=bundle["final_registry"],
        output=tmp_path / "gate.json",
        require_distinct_baseline_asset=False,
    )

    assert report["status"] == PASS
    assert report["completion_allowed"] is True
    assert report["policy"]["require_distinct_baseline_asset"] is False


def test_non_pass_final_quality_status_blocks_completion(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    final_quality = copy.deepcopy(bundle["final_quality"])
    assert isinstance(final_quality, dict)
    final_quality["aggregate"]["status"] = "REVIEW"
    final_quality["aggregate"]["review_view_count"] = 1
    final_quality["views"][0]["status"] = "REVIEW"
    _rewrite_final_quality(bundle, final_quality)

    report = _run(bundle, tmp_path / "gate.json")

    assert report["status"] == FAIL_CLOSED
    assert "FINAL_AGGREGATE_STATUS_NOT_PASS" in report["reason_codes"]
    assert "FINAL_VIEW_STATUS_NOT_PASS" in report["reason_codes"]
    assert "AGGREGATE_VIEW_COVERAGE_REGRESSION" in report["reason_codes"]


def test_cli_returns_nonzero_and_writes_audit_on_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = _bundle(tmp_path)
    output = tmp_path / "gate.json"

    exit_code = main(
        [
            "--collected-usd",
            str(bundle["collected"]),
            "--baseline-quality-report",
            str(bundle["baseline_quality_path"]),
            "--final-quality-report",
            str(bundle["baseline_quality_path"]),
            "--baseline-rendered-registry",
            str(bundle["baseline_registry"]),
            "--final-rendered-registry",
            str(bundle["baseline_registry"]),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 2
    assert json.loads(output.read_text())["completion_state"] != COMPLETED
    assert '"completion_allowed": false' in capsys.readouterr().out
