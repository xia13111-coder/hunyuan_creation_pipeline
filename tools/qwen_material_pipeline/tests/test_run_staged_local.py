from __future__ import annotations

import hashlib
import io
import json
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image
import pytest

import qwen_material_pipeline.workflows.staged_local as run_staged_local
from qwen_material_pipeline.workflows.staged_local import (
    DEFAULT_MIN_VISIBLE_PIXELS,
    MIN_MATCH_VISIBLE_PIXELS,
    _assign_best_evidence_views,
    _build_disagreement_tournament_candidates,
    _build_sam3_foreground_request,
    _canonicalize_primary_batches,
    _catalog_pool,
    _confirm_material_choices,
    _derive_material_selection_confidence,
    _family_hint_is_reliable,
    _legacy_palette_from_fusion,
    _make_reference_sheet,
    _material_selection_context,
    _merge_equivalent_palette_groups,
    _mvinverse_exact_default_candidates,
    _multiview_pre_filter_group_count,
    _palette_quality_metrics,
    _resolve_immutable_coating_physics_choice,
    _select_palette_candidate,
    _shortlist_materials_with_audit,
    _validate_sam3_manifest,
    _view_grouped_part_batches,
    _write_foreground_masked_references,
    build_parser,
)
from qwen_material_pipeline.segmentation.foreground_seeds import (
    build_automatic_foreground_seeds,
)
from qwen_material_pipeline.materials.catalog import MaterialCatalog
from qwen_material_pipeline.core.staged_analysis import (
    PALETTE_SCHEMA_VERSION,
    StagedAnalysisError,
)
from qwen_material_pipeline.evidence.confidence import ConfidenceGateError
from qwen_material_pipeline.evidence.palette_fusion import fuse_multiview_palettes
from qwen_material_pipeline.qwen.client import QwenContentParseError


def test_foreground_sam3_policy_is_identical_for_run_and_resume() -> None:
    assert run_staged_local._foreground_maximum_image_fraction(0.80) == 0.90
    assert run_staged_local._foreground_maximum_image_fraction(0.95) == 0.95


def test_sam3_foreground_request_precedes_and_does_not_depend_on_palette(
    tmp_path: Path,
) -> None:
    for name, rectangle in (
        ("front.png", (10, 8, 58, 52)),
        ("side.png", (24, 5, 45, 57)),
    ):
        image = Image.new("RGB", (64, 64), (0, 0, 0))
        for x in range(rectangle[0], rectangle[2]):
            for y in range(rectangle[1], rectangle[3]):
                image.putpixel((x, y), (20, 160, 35))
        image.save(tmp_path / name)
    references = [
        ("front", tmp_path / "front.png"),
        ("side", tmp_path / "side.png"),
    ]
    request = _build_sam3_foreground_request(references)

    assert request["prompt_authority"].endswith("no_vlm_no_material_assumption")
    assert request["source_views"] == [
        {"id": view_id, "image": str(path)} for view_id, path in references
    ]
    assert [region["group_id"] for region in request["regions"]] == [
        "__foreground__",
        "__foreground__",
    ]
    assert all(region["boxes"] != [[1, 1, 999, 999]] for region in request["regions"])
    assert all(
        region["prompt"] == "manufactured object" for region in request["regions"]
    )
    assert request["foreground_seed_policy"]["schema_version"].endswith("/v1")


def test_human_sam3_foreground_request_never_uses_automatic_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    front = tmp_path / "front.png"
    side = tmp_path / "side.png"
    Image.new("RGB", (16, 12), (20, 140, 30)).save(front)
    Image.new("RGB", (16, 12), (30, 120, 20)).save(side)

    def forbidden(_path: Path):
        raise AssertionError("automatic foreground seeds must not run")

    monkeypatch.setattr(run_staged_local, "build_automatic_foreground_seeds", forbidden)
    annotations = {
        "schema_version": run_staged_local.SAM3_HUMAN_ANNOTATION_SCHEMA_VERSION,
        "source_views": [
            {
                "id": view_id,
                "click_sets": [
                    {
                        "events": [
                            {"point": [500, 500], "label": 1},
                            {"point": [0, 0], "label": 0},
                        ],
                        "positive_points": [[500, 500]],
                        "negative_points": [[0, 0]],
                        "initial_candidate_index": 1,
                    }
                ],
                "confirmed_mask": {
                    "path": str(tmp_path / f"{view_id}-mask.png"),
                    "sha256": "a" * 64,
                },
            }
            for view_id in ("front", "side")
        ],
        "integrity": {"document_sha256": "b" * 64},
    }

    request = _build_sam3_foreground_request(
        [("front", front), ("side", side)],
        annotations=annotations,
    )

    assert request["schema_version"].endswith("/v3")
    assert request["prompt_authority"] == ("human_confirmed_sam3_interactive_points")
    assert all(
        region["boxes"] == [] for region in request["regions"] if "boxes" in region
    )
    assert [
        region["click_sets"][0]["positive_points"] for region in request["regions"]
    ] == [
        [[500, 500]],
        [[500, 500]],
    ]


def test_legacy_human_annotations_keep_unordered_v2_request(tmp_path: Path) -> None:
    image = tmp_path / "front.png"
    Image.new("RGB", (8, 8), "white").save(image)
    annotations = {
        "schema_version": (
            run_staged_local.SAM3_LEGACY_HUMAN_ANNOTATION_SCHEMA_VERSION
        ),
        "source_views": [
            {
                "id": "front",
                "click_sets": [
                    {
                        "positive_points": [[500, 500]],
                        "negative_points": [],
                    }
                ],
                "confirmed_mask": {"path": "mask.png", "sha256": "a" * 64},
            }
        ],
        "integrity": {"document_sha256": "b" * 64},
    }

    request = _build_sam3_foreground_request(
        [("front", image)], annotations=annotations
    )

    assert (
        request["schema_version"] == run_staged_local.SAM3_POINT_REQUEST_SCHEMA_VERSION
    )
    assert request["human_annotation"]["schema_version"] == (
        run_staged_local.SAM3_LEGACY_HUMAN_ANNOTATION_SCHEMA_VERSION
    )


def _interactive_manifest_fixture(
    tmp_path: Path, *, ordered: bool
) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    source = tmp_path / "source.png"
    confirmed = tmp_path / "confirmed.png"
    output_mask = tmp_path / "output.png"
    Image.new("RGB", (8, 8), "white").save(source)
    mask = Image.new("L", (8, 8), 0)
    for x in range(2, 6):
        for y in range(2, 6):
            mask.putpixel((x, y), 255)
    mask.save(confirmed)
    mask.save(output_mask)
    repository = tmp_path / "sam3"
    repository.mkdir()
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"sam3")
    if ordered:
        click_sets = [
            {
                "events": [
                    {"point": [500, 500], "label": 1},
                    {"point": [0, 0], "label": 0},
                ],
                "positive_points": [[500, 500]],
                "negative_points": [[0, 0]],
                "initial_candidate_index": 1,
            }
        ]
        schema = run_staged_local.SAM3_ORDERED_POINT_REQUEST_SCHEMA_VERSION
    else:
        click_sets = [
            {
                "positive_points": [[500, 500]],
                "negative_points": [[0, 0]],
            }
        ]
        schema = run_staged_local.SAM3_POINT_REQUEST_SCHEMA_VERSION
    request: dict[str, Any] = {
        "schema_version": schema,
        "source_views": [{"id": "front", "image": str(source)}],
        "regions": [
            {
                "view_id": "front",
                "group_id": "__foreground__",
                "click_sets": click_sets,
                "confirmed_mask": {
                    "path": str(confirmed),
                    "sha256": run_staged_local._sha256_file(confirmed),
                },
            }
        ],
    }
    if ordered:
        request.update(
            prompt_authority="human_confirmed_sam3_interactive_points",
            human_annotation={
                "schema_version": run_staged_local.SAM3_HUMAN_ANNOTATION_SCHEMA_VERSION,
                "document_sha256": "a" * 64,
                "all_views_confirmed": True,
                "human_mask_is_authoritative": True,
                "formal_rerun_minimum_iou": 0.995,
            },
        )
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    confirmed_digest = run_staged_local._sha256_file(confirmed)
    record: dict[str, Any] = {
        "source_image": str(source.resolve()),
        "source_image_sha256": run_staged_local._sha256_file(source),
        "view_id": "front",
        "group_id": "__foreground__",
        "prompt_mode": (
            "human_ordered_incremental_points"
            if ordered
            else "human_interactive_points"
        ),
        "click_sets": click_sets,
        "point_set_count": 1,
        "accepted_point_set_count": 1,
        "accepted": True,
        "reason_codes": [],
        "mask_pixels": 16,
        "mask": {
            "path": str(output_mask),
            "sha256": run_staged_local._sha256_file(output_mask),
        },
        "confirmed_mask_audit": {
            "sha256": confirmed_digest,
            "confirmed_mask_pixels": 16,
            "reproduction_iou": 1.0,
            "minimum_reproduction_iou": 0.995,
            "accepted": True,
            "authoritative_output": "human_confirmed_mask",
        },
    }
    if ordered:
        record.update(
            interaction_replay_mode="ordered_events_previous_logits",
            event_count=2,
            point_set_audits=[
                {
                    "click_set_index": 0,
                    "positive_points": [[500, 500]],
                    "negative_points": [[0, 0]],
                    "events": click_sets[0]["events"],
                    "initial_candidate_index": 1,
                    "accepted": True,
                    "event_count": 2,
                    "event_audits": [
                        {
                            "event_index": 0,
                            "event": click_sets[0]["events"][0],
                            "event_count": 1,
                            "multimask_output": True,
                            "used_previous_logits": False,
                            "accepted": True,
                            "selected_candidate_index": 1,
                            "candidate_selection": "persisted_initial_candidate",
                            "candidates": [{}, {}, {}],
                        },
                        {
                            "event_index": 1,
                            "event": click_sets[0]["events"][1],
                            "event_count": 2,
                            "multimask_output": False,
                            "used_previous_logits": True,
                            "accepted": True,
                            "selected_candidate_index": 0,
                            "candidate_selection": "single_mask_refinement",
                            "candidates": [{}],
                        },
                    ],
                }
            ],
        )
    policy = run_staged_local.sam3_result_policy(
        minimum_model_score=0.45,
        minimum_prompt_overlap=0.25,
        maximum_image_fraction=0.90,
        minimum_mask_pixels=1,
        human_interactive_requested=True,
        automatic_shape_interactive_requested=False,
        ordered_interaction_requested=ordered,
    )
    unsigned = {
        "schema_version": run_staged_local.SAM3_RESULT_SCHEMA_VERSION,
        "request": {
            "sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
            "document_sha256": run_staged_local._canonical_sha256(request),
        },
        "backend": {
            "repository": str(repository),
            "repository_revision": None,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": run_staged_local._sha256_file(checkpoint),
            "device": "cuda",
            "instance_interactivity_enabled": True,
        },
        "policy": policy,
        "records": [record],
        "summary": {
            "region_count": 1,
            "accepted_region_count": 1,
            "rejected_region_count": 0,
        },
    }
    manifest = {
        **unsigned,
        "integrity": {"result_sha256": run_staged_local._canonical_sha256(unsigned)},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, manifest_path, request_path, repository, checkpoint


def _validate_interactive_fixture(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    request_path: Path,
    repository: Path,
    checkpoint: Path,
) -> None:
    unsigned = {key: value for key, value in manifest.items() if key != "integrity"}
    manifest["integrity"] = {
        "result_sha256": run_staged_local._canonical_sha256(unsigned)
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _validate_sam3_manifest(
        manifest,
        manifest_path=manifest_path,
        request_path=request_path,
        repository=repository,
        checkpoint=checkpoint,
        device="cuda",
        minimum_model_score=0.45,
        minimum_prompt_overlap=0.25,
        maximum_image_fraction=0.90,
        minimum_mask_pixels=1,
    )


def test_legacy_v2_manifest_without_new_replay_fields_can_resume(
    tmp_path: Path,
) -> None:
    fixture = _interactive_manifest_fixture(tmp_path, ordered=False)
    manifest, manifest_path, request_path, repository, checkpoint = fixture

    _validate_interactive_fixture(
        manifest,
        manifest_path=manifest_path,
        request_path=request_path,
        repository=repository,
        checkpoint=checkpoint,
    )


def test_sam3_manifest_policy_binds_the_shared_inference_seed(
    tmp_path: Path,
) -> None:
    fixture = _interactive_manifest_fixture(tmp_path, ordered=False)
    manifest, manifest_path, request_path, repository, checkpoint = fixture

    assert manifest["policy"]["inference_seed"] == 0
    assert manifest["policy"]["deterministic_algorithms"] is True
    _validate_interactive_fixture(
        manifest,
        manifest_path=manifest_path,
        request_path=request_path,
        repository=repository,
        checkpoint=checkpoint,
    )

    tampered = copy.deepcopy(manifest)
    tampered["policy"]["inference_seed"] = 1
    with pytest.raises(ValueError, match="policy does not match"):
        _validate_interactive_fixture(
            tampered,
            manifest_path=manifest_path,
            request_path=request_path,
            repository=repository,
            checkpoint=checkpoint,
        )


def test_ordered_v3_manifest_validates_exact_event_replay(tmp_path: Path) -> None:
    fixture = _interactive_manifest_fixture(tmp_path, ordered=True)
    manifest, manifest_path, request_path, repository, checkpoint = fixture

    _validate_interactive_fixture(
        manifest,
        manifest_path=manifest_path,
        request_path=request_path,
        repository=repository,
        checkpoint=checkpoint,
    )

    tampered = copy.deepcopy(manifest)
    tampered["records"][0]["point_set_audits"][0]["event_audits"][0][
        "selected_candidate_index"
    ] = 2
    with pytest.raises(ValueError, match="event replay audit is invalid"):
        _validate_interactive_fixture(
            tampered,
            manifest_path=manifest_path,
            request_path=request_path,
            repository=repository,
            checkpoint=checkpoint,
        )


def test_ordered_v3_manifest_accepts_bounded_human_confirmed_replay(
    tmp_path: Path,
) -> None:
    fixture = _interactive_manifest_fixture(tmp_path, ordered=True)
    manifest, manifest_path, request_path, repository, checkpoint = fixture
    audit = manifest["records"][0]["confirmed_mask_audit"]
    audit.update(
        reproduction_iou=0.9428,
        reproduction_precision=0.9956,
        reproduction_recall=0.9468,
        bounded_minimum_precision=(
            run_staged_local.CONFIRMED_MASK_BOUNDED_MINIMUM_PRECISION
        ),
        bounded_minimum_recall=(
            run_staged_local.CONFIRMED_MASK_BOUNDED_MINIMUM_RECALL
        ),
        acceptance_mode="bounded_human_confirmed",
    )

    _validate_interactive_fixture(
        manifest,
        manifest_path=manifest_path,
        request_path=request_path,
        repository=repository,
        checkpoint=checkpoint,
    )

    audit["reproduction_precision"] = 0.98
    with pytest.raises(
        ValueError, match="confirmed-mask reproduction audit is invalid"
    ):
        _validate_interactive_fixture(
            manifest,
            manifest_path=manifest_path,
            request_path=request_path,
            repository=repository,
            checkpoint=checkpoint,
        )


def test_ordered_v3_manifest_accepts_tight_symmetric_boundary_drift(
    tmp_path: Path,
) -> None:
    fixture = _interactive_manifest_fixture(tmp_path, ordered=True)
    manifest, manifest_path, request_path, repository, checkpoint = fixture
    audit = manifest["records"][0]["confirmed_mask_audit"]
    audit.update(
        reproduction_iou=0.9878,
        symmetric_minimum_reproduction_iou=(
            run_staged_local.CONFIRMED_MASK_SYMMETRIC_MINIMUM_IOU
        ),
        acceptance_mode="symmetric_boundary_drift",
    )

    _validate_interactive_fixture(
        manifest,
        manifest_path=manifest_path,
        request_path=request_path,
        repository=repository,
        checkpoint=checkpoint,
    )

    audit["reproduction_iou"] = 0.9849
    with pytest.raises(
        ValueError, match="confirmed-mask reproduction audit is invalid"
    ):
        _validate_interactive_fixture(
            manifest,
            manifest_path=manifest_path,
            request_path=request_path,
            repository=repository,
            checkpoint=checkpoint,
        )


def test_automatic_foreground_seeds_remove_thin_grid_and_keep_components(
    tmp_path: Path,
) -> None:
    image = Image.new("RGB", (200, 160), (0, 0, 0))
    for x in range(0, 200, 20):
        for y in range(160):
            image.putpixel((x, y), (12, 12, 12))
    for y in range(0, 160, 20):
        for x in range(200):
            image.putpixel((x, y), (12, 12, 12))
    for x in range(45, 145):
        for y in range(30, 130):
            image.putpixel((x, y), (15, 145, 40))
    for x in range(165, 185):
        for y in range(110, 140):
            image.putpixel((x, y), (210, 210, 210))
    path = tmp_path / "grid.png"
    image.save(path)

    result = build_automatic_foreground_seeds(path)

    assert len(result["boxes"]) == 2
    assert all(box != [0, 0, 1000, 1000] for box in result["boxes"])
    assert result["background_rgb"] == [0, 0, 0]


def test_foreground_masked_references_neutralize_only_background(
    tmp_path: Path,
) -> None:
    source = tmp_path / "front.png"
    mask = tmp_path / "front-mask.png"
    Image.new("RGB", (4, 2), (10, 20, 30)).save(source)
    mask_image = Image.new("L", (4, 2), 0)
    for x in range(2):
        for y in range(2):
            mask_image.putpixel((x, y), 255)
    mask_image.save(mask)

    references, audit_path = _write_foreground_masked_references(
        parsed_references=[("front", source)],
        foreground_masks={"front": mask},
        output_dir=tmp_path / "foreground_inference" / "masked_views",
    )

    assert references[0][0] == "front"
    with Image.open(references[0][1]) as opened:
        pixels = opened.convert("RGB")
        assert pixels.getpixel((0, 0)) == (10, 20, 30)
        assert pixels.getpixel((3, 0)) == (127, 127, 127)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["policy"] == ("all_vlm_and_mvinverse_inputs_are_sam3_foreground_only")
    assert audit["source_views"][0]["background_fill_rgb"] == [127, 127, 127]


def test_material_selection_confidence_ignores_qwen_numeric_anchor() -> None:
    confidence, audit = _derive_material_selection_confidence(
        first={"material_id": "MAT_A", "confidence": 0.0},
        second={"material_id": "MAT_A", "confidence": 0.0},
        chosen={"material_id": "MAT_A"},
        confirmed=True,
        confirmation_basis="exact_forward_reverse_agreement",
        retrieval_audit={
            "ranking": [{"rank": 1, "material_id": "MAT_A"}],
            "margin_available": True,
            "normalized_margin": 0.12,
        },
        independent_choices=[
            {"view_id": "front", "material_id": "MAT_A", "confidence": 0.0},
            {"view_id": "side", "material_id": "MAT_A", "confidence": 0.0},
        ],
    )

    assert confidence == pytest.approx(0.90)
    assert audit["decision"] == "independent_multimodel_auto"
    assert audit["reported_forward_confidence"] == 0.0
    assert audit["reported_reverse_confidence"] == 0.0
    assert audit["reported_confidence_is_authoritative"] is False


def test_order_stability_without_retrieval_top_is_review_only() -> None:
    confidence, audit = _derive_material_selection_confidence(
        first={"material_id": "MAT_A", "confidence": 0.99},
        second={"material_id": "MAT_A", "confidence": 0.99},
        chosen={"material_id": "MAT_A"},
        confirmed=True,
        confirmation_basis="exact_forward_reverse_agreement",
        retrieval_audit={
            "ranking": [
                {"rank": 1, "material_id": "MAT_B"},
                {"rank": 2, "material_id": "MAT_A"},
            ],
            "margin_available": True,
            "normalized_margin": 0.40,
        },
        independent_choices=[
            {"view_id": "front", "material_id": "MAT_A", "confidence": 0.99},
            {"view_id": "side", "material_id": "MAT_A", "confidence": 0.99},
        ],
    )

    assert confidence == pytest.approx(0.70)
    assert audit["decision"] == "order_stable_review"
    assert "retrieval_top_disagrees" in audit["reason_codes"]


def test_unconfirmed_material_disagreement_has_zero_derived_confidence() -> None:
    confidence, audit = _derive_material_selection_confidence(
        first={"material_id": "MAT_A", "confidence": 0.99},
        second={"material_id": "MAT_B", "confidence": 0.99},
        chosen={"material_id": "MAT_A"},
        confirmed=False,
        confirmation_basis="forward_reverse_disagreement",
        retrieval_audit={
            "ranking": [{"rank": 1, "material_id": "MAT_A"}],
            "margin_available": True,
            "normalized_margin": 0.5,
        },
        independent_choices=[
            {"view_id": "front", "material_id": "MAT_A", "confidence": 0.99},
            {"view_id": "side", "material_id": "MAT_A", "confidence": 0.99},
        ],
    )

    assert confidence == 0.0
    assert audit["decision"] == "unconfirmed_preserve"


def test_main_classifies_any_structured_qwen_parse_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "analysis"
    output.mkdir()
    error = QwenContentParseError(
        "response contains prose outside JSON",
        reason="json_with_surrounding_prose",
    )
    error.stage_name = "03_material_G01_reverse"
    error.raw_output_path = output / "raw" / "03_material_G01_reverse.raw.txt"

    def fail(_argv: list[str] | None) -> int:
        raise error

    monkeypatch.setattr(run_staged_local, "_main", fail)

    with pytest.raises(QwenContentParseError):
        run_staged_local.main(["--output-dir", str(output)])

    failure = json.loads(
        (output / "inference_failure.json").read_text(encoding="utf-8")
    )
    assert failure["error_code"] == "qwen_structured_output_invalid"
    assert failure["failed_stage"] == "03_material_G01_reverse"
    assert failure["retryable"] is False
    assert failure["retry_scope"] == "none"
    assert failure["context"]["reason"] == "json_with_surrounding_prose"
    assert failure["context"]["raw_output"].endswith("03_material_G01_reverse.raw.txt")


def test_main_classifies_confidence_contract_failure_as_non_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "analysis"
    output.mkdir()

    def fail(_argv: list[str] | None) -> int:
        raise ConfidenceGateError("visual retrieval audit is inconsistent")

    monkeypatch.setattr(run_staged_local, "_main", fail)

    with pytest.raises(ConfidenceGateError):
        run_staged_local.main(["--output-dir", str(output)])

    failure = json.loads(
        (output / "inference_failure.json").read_text(encoding="utf-8")
    )
    assert failure["error_code"] == "confidence_gate_contract_invalid"
    assert failure["failed_stage"] == "confidence_gate"
    assert failure["retryable"] is False
    assert failure["retry_scope"] == "none"


def test_main_preserves_more_specific_existing_inference_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "analysis"
    output.mkdir()
    expected = {
        "schema_version": "qwen-material-inference-failure/v1",
        "status": "FAILED",
        "error_code": "insufficient_usable_palette_views",
        "failed_stage": "palette",
        "retryable": False,
        "retry_scope": "none",
        "detail": "specific palette failure",
        "view_failures": [],
    }
    (output / "inference_failure.json").write_text(
        json.dumps(expected), encoding="utf-8"
    )

    def fail(_argv: list[str] | None) -> int:
        raise StagedAnalysisError("generic outer failure")

    monkeypatch.setattr(run_staged_local, "_main", fail)

    with pytest.raises(StagedAnalysisError):
        run_staged_local.main(["--output-dir", str(output)])

    assert (
        json.loads((output / "inference_failure.json").read_text(encoding="utf-8"))
        == expected
    )


def test_main_does_not_misclassify_local_staged_analysis_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "analysis"
    output.mkdir()

    def fail(_argv: list[str] | None) -> int:
        raise StagedAnalysisError("local crop/fusion validation failed")

    monkeypatch.setattr(run_staged_local, "_main", fail)

    with pytest.raises(StagedAnalysisError):
        run_staged_local.main(["--output-dir", str(output)])

    assert not (output / "inference_failure.json").exists()


def test_main_classifies_tampered_qwen_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "analysis"
    output.mkdir()
    error = StagedAnalysisError("checkpoint integrity SHA-256 mismatch")
    error.stage_name = "02_map_B01"
    error.raw_output_path = output / "raw" / "02_map_B01.raw.txt"
    error.checkpoint_path = (
        output / "qwen_stage_checkpoints" / "02_map_B01.checkpoint.json"
    )
    error.checkpoint_context = {"stage_name": "02_map_B01"}

    def fail(_argv: list[str] | None) -> int:
        raise error

    monkeypatch.setattr(run_staged_local, "_main", fail)

    with pytest.raises(StagedAnalysisError):
        run_staged_local.main(["--output-dir", str(output)])

    failure = json.loads(
        (output / "inference_failure.json").read_text(encoding="utf-8")
    )
    assert failure["error_code"] == "qwen_stage_checkpoint_invalid"
    assert failure["failed_stage"] == "02_map_B01"
    assert failure["context"]["checkpoint"].endswith("02_map_B01.checkpoint.json")


def test_qwen35_runtime_manifest_checks_every_file_and_exact_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    files = {
        "config.json": b'{"model_type":"qwen3_5"}',
        "chat_template.jinja": b"trusted template",
        "tokenizer.json": b"trusted tokenizer",
        "model.safetensors": b"trusted weights",
    }
    records = []
    for relative, content in files.items():
        (model / relative).write_bytes(content)
        records.append(
            {
                "path": relative,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    records.sort(key=lambda item: item["path"])
    monkeypatch.setattr(
        run_staged_local,
        "QWEN35_CONTENT_MANIFEST_SHA256",
        run_staged_local._canonical_sha256(records),
    )
    identity = {
        "config_sha256": next(
            item["sha256"] for item in records if item["path"] == "config.json"
        ),
        "runtime_files": records,
    }

    assert set(
        run_staged_local._validate_qwen35_runtime_manifest(model, identity)
    ) == set(files)

    (model / "tokenizer.json").write_bytes(b"tampered tokenizer")
    with pytest.raises(ValueError, match="size/SHA-256"):
        run_staged_local._validate_qwen35_runtime_manifest(model, identity)
    (model / "tokenizer.json").write_bytes(files["tokenizer.json"])

    (model / "unmanifested.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="file set differs"):
        run_staged_local._validate_qwen35_runtime_manifest(model, identity)


def test_qwen35_ledger_requires_exact_pinned_revision() -> None:
    with pytest.raises(ValueError, match="pinned --qwen-model-revision"):
        run_staged_local._qwen_inference_ledger(
            model_identity={"model_type": "qwen3_5"},
            requested_family="qwen3_5",
            requested_revision=None,
        )
    with pytest.raises(ValueError, match="pinned --qwen-model-revision"):
        run_staged_local._qwen_inference_ledger(
            model_identity={"model_type": "qwen3_5"},
            requested_family="auto",
            requested_revision="moving-main",
        )


def test_reference_sheet_packs_four_unordered_views(tmp_path: Path) -> None:
    references = []
    for index, color in enumerate(("red", "green", "blue", "white"), start=1):
        path = tmp_path / f"view_{index}.png"
        Image.new("RGB", (80 + index, 120), color).save(path)
        references.append((f"ref_{index}", path))
    output = _make_reference_sheet(references, tmp_path / "sheet.png", cell_size=128)
    with Image.open(output) as sheet:
        assert sheet.size == (256, 256)


def test_family_reliability_accepts_three_view_majority_over_one_conflict() -> None:
    group = {
        "group_id": "G06",
        "family_hint": "metal",
        "confidence": 0.91,
    }
    fusion_group = {
        "sources": [
            {
                "view_id": "front",
                "family_hint": "metal",
                "confidence": 0.90,
            },
            {
                "view_id": "side",
                "family_hint": "metal",
                "confidence": 0.91,
            },
            {
                "view_id": "iso",
                "family_hint": "metal",
                "confidence": 0.70,
            },
            {
                "view_id": "top",
                "family_hint": "plastic",
                "confidence": 0.84,
            },
        ]
    }

    assert _family_hint_is_reliable(group, fusion_group) is True


def test_family_reliability_rejects_two_against_one_as_weak_margin() -> None:
    group = {
        "group_id": "G01",
        "family_hint": "metal",
        "confidence": 0.90,
    }
    fusion_group = {
        "sources": [
            {
                "view_id": "front",
                "family_hint": "metal",
                "confidence": 0.90,
            },
            {
                "view_id": "side",
                "family_hint": "metal",
                "confidence": 0.89,
            },
            {
                "view_id": "top",
                "family_hint": "plastic",
                "confidence": 0.60,
            },
        ]
    }

    assert _family_hint_is_reliable(group, fusion_group) is False


def test_staged_parser_accepts_repeated_reference_arguments() -> None:
    args = build_parser().parse_args(
        [
            "--registry",
            "registry.json",
            "--reference",
            "ref_a=a.png",
            "--reference",
            "ref_b=b.png",
            "--catalog",
            "catalog.json",
            "--model-path",
            "model",
            "--output-dir",
            "output",
        ]
    )
    assert args.reference == ["ref_a=a.png", "ref_b=b.png"]
    assert args.palette_reference == "auto"
    assert args.palette_mask == []
    assert args.min_visible_pixels == DEFAULT_MIN_VISIBLE_PIXELS == 64
    assert args.mvinverse_mode == "off"
    assert args.mvinverse_max_side == 448
    assert args.acknowledge_mvinverse_noncommercial is False
    assert args.material_selection_objective == "semantic_compatible_visual"


def test_catalog_exact_allowlist_rejects_any_omitted_material(tmp_path: Path) -> None:
    root = tmp_path / "Base"
    root.mkdir()
    for name in ("Steel", "Copper"):
        (root / f"{name}.mdl").write_text(
            f"export material {name}(*) = material();",
            encoding="utf-8",
        )
    catalog = MaterialCatalog.scan(root)
    allowlist = catalog.to_full_allowlist_dict()
    allowlist["material_ids"].pop()
    allowlist_path = tmp_path / "allowlist.json"
    allowlist_path.write_text(json.dumps(allowlist), encoding="utf-8")

    with pytest.raises(ValueError, match="does not equal the complete"):
        _catalog_pool(catalog, allowlist_path)


def test_reliable_metal_family_also_considers_base_paint() -> None:
    group = {
        "group_id": "G01",
        "family_hint": "metal",
        "base_color": "green",
        "finish_hint": "matte",
        "visual_description": "green painted steel enclosure",
    }
    pool = [
        {
            "material_id": "mdl:Metals/Steel_Carbon.mdl#Steel_Carbon",
            "display_name": "Steel Carbon",
            "family": "metal",
            "colors": [],
            "finishes": [],
        },
        {
            "material_id": "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte",
            "display_name": "Paint Matte",
            "family": "paint",
            "colors": [],
            "finishes": ["matte"],
        },
        {
            "material_id": "mdl:Plastics/Plastic_ABS.mdl#Plastic_ABS",
            "display_name": "Plastic ABS",
            "family": "plastic",
            "colors": [],
            "finishes": [],
        },
    ]

    candidates, audit = _shortlist_materials_with_audit(
        group,
        pool,
        family_reliable=True,
        limit=3,
    )

    assert {item["family"] for item in candidates} == {"metal", "paint"}
    assert audit["eligible_pool_count"] == 2


def test_mvinverse_tunable_paint_beats_fixed_anodized_for_painted_metal() -> None:
    group = {
        "group_id": "G06",
        "family_hint": "metal",
        "base_color": "green",
        "finish_hint": "painted",
        "visual_description": "green painted machine enclosure",
    }
    pool = [
        {
            "material_id": (
                "mdl:Base/Metals/Aluminum_Anodized_Charcoal.mdl"
                "#Aluminum_Anodized_Charcoal"
            ),
            "display_name": "Anodized Charcoal",
            "family": "metal",
            "colors": [],
            "finishes": ["coated"],
        },
        {
            "material_id": "mdl:Base/Miscellaneous/Paint_Satin.mdl#Paint_Satin",
            "display_name": "Paint Satin",
            "family": "paint",
            "colors": [],
            "finishes": ["painted", "satin"],
        },
    ]
    evidence = {
        "surface_class": "dielectric",
        "distinct_view_count": 4,
        "albedo": {
            "sample_count": 4,
            "median": [0.23, 0.54, 0.20],
            "mad": [0.01, 0.01, 0.01],
        },
        "metallic": {
            "sample_count": 4,
            "median": 0.12,
            "mad": 0.01,
            "iqr": 0.02,
        },
        "roughness": {
            "sample_count": 4,
            "median": 0.43,
            "mad": 0.01,
            "iqr": 0.02,
        },
        "suggestion": {
            "decision": "auto",
            "auto_parameter_eligible": True,
            "base_color_srgb": [0.23, 0.54, 0.20],
            "metallic": 0.0,
            "roughness": 0.43,
        },
    }

    candidates, audit = _shortlist_materials_with_audit(
        group,
        pool,
        family_reliable=False,
        mvinverse_pbr_evidence=evidence,
    )

    assert candidates[0]["material_id"].endswith("Paint_Satin.mdl#Paint_Satin")
    assert "coating_surface_family" in audit["ranking"][0]["matched_fields"]
    assert "mvinverse_tunable_template" in audit["ranking"][0]["matched_fields"]


def test_immutable_generic_plastic_rejects_unsupported_pcb_domain_preset() -> None:
    group = {
        "group_id": "G04",
        "family_hint": "plastic",
        "base_color": "white",
        "finish_hint": "painted",
        "visual_description": "white painted rectangular plastic plate",
    }
    pool = [
        {
            "material_id": (
                "mdl:vMaterials_2/Plastic/PCB_Solder_Mask.mdl" "#PCB_Solder_Mask_White"
            ),
            "display_name": "PCB Solder Mask White",
            "description": "white printed circuit board solder mask",
            "family": "plastic",
            "colors": ["white"],
            "finishes": ["painted"],
        },
        {
            "material_id": (
                "mdl:vMaterials_2/Plastic/Polyethylene_Opaque.mdl" "#Polyethylene_White"
            ),
            "display_name": "Polyethylene White",
            "description": "opaque engineering plastic",
            "family": "plastic",
            "colors": ["white"],
            "finishes": ["painted", "opaque"],
        },
    ]

    candidates, audit = _shortlist_materials_with_audit(
        group,
        pool,
        family_reliable=True,
        allow_parameter_writes=False,
    )

    assert candidates[0]["material_id"].endswith(
        "Polyethylene_Opaque.mdl#Polyethylene_White"
    )
    assert audit["niche_domain_policy"] == {
        "mode": "positive_reference_semantics_required",
        "domains": ["automotive_finish", "electronics_surface"],
    }


def test_general_anodized_material_is_not_rejected_for_broad_automotive_keyword() -> None:
    group = {
        "group_id": "G06",
        "family_hint": "metal",
        "base_color": "green",
        "finish_hint": "painted",
        "visual_description": "green painted industrial enclosure",
    }
    general_anodized = {
        "material_id": (
            "mdl:vMaterials_2/Metal/Aluminum_Anodized.mdl"
            "#Aluminum_Anodized_Grass_Green"
        ),
        "category_path": "vMaterials_2/Metal",
        "display_name": "Aluminum Anodized Grass Green",
        "description": "general coated aluminium for AEC and industrial design",
        "keywords": ["aec", "automotive", "design", "green"],
        "family": "metal",
        "colors": ["green"],
        "finishes": ["anodized", "coated"],
    }
    carpaint = {
        "material_id": (
            "mdl:vMaterials_2/Paint/Carpaint/Carpaint_Solid.mdl" "#Hunting_Green_Matte"
        ),
        "category_path": "vMaterials_2/Paint/Carpaint",
        "display_name": "Carpaint Hunting Green Matte",
        "description": "multilayer automotive carpaint",
        "keywords": ["carpaint", "automotive", "green"],
        "family": "paint",
        "colors": ["green"],
        "finishes": ["matte"],
    }

    candidates, audit = _shortlist_materials_with_audit(
        group,
        [general_anodized, carpaint],
        family_reliable=False,
        allow_parameter_writes=False,
        limit=2,
    )

    assert candidates[0]["material_id"] == general_anodized["material_id"]
    assert audit["ranking"][0]["score"] - audit["ranking"][1]["score"] >= 300


def test_mvinverse_tunable_exports_form_one_effective_retrieval_candidate() -> None:
    army = "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted_Army_Green"
    arcadia = "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted_Arcadia_Green"
    cracked = (
        "mdl:vMaterials_2/Metal/Steel_Painted_Cracked.mdl"
        "#Steel_Painted_Army_Green_Cracked_Dirty"
    )
    group = {
        "group_id": "G06",
        "family_hint": "metal",
        "base_color": "green",
        "finish_hint": "painted",
        "visual_description": "green painted machine enclosure",
    }
    pool = [
        {
            "material_id": material_id,
            "display_name": material_id.rsplit("#", 1)[-1],
            "family": "metal",
            "colors": ["green"],
            "finishes": ["painted"],
        }
        for material_id in (army, arcadia, cracked)
    ]
    evidence = {
        "surface_class": "dielectric",
        "suggestion": {
            "decision": "auto",
            "auto_parameter_eligible": True,
        },
    }

    candidates, audit = _shortlist_materials_with_audit(
        group,
        pool,
        family_reliable=True,
        mvinverse_pbr_evidence=evidence,
    )

    painted_exports = [
        item
        for item in candidates
        if item["material_id"].split("#", 1)[0].endswith("/Steel_Painted.mdl")
    ]
    assert len(painted_exports) == 1
    assert audit["mvinverse_tunable_equivalence_dedup_count"] == 1
    assert audit["runner_up_score"] == audit["ranking"][1]["score"]


def test_immutable_mdl_retrieval_uses_distinct_fixed_export_defaults() -> None:
    army = "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted_Army_Green"
    cracked = (
        "mdl:vMaterials_2/Metal/Steel_Painted_Cracked.mdl"
        "#Steel_Painted_Army_Green_Cracked_Dirty"
    )
    hammer = "mdl:vMaterials_2/Paint/Hammer_Paint.mdl#Hammer_Paint_Green"
    group = {
        "group_id": "G06",
        "family_hint": "metal",
        "base_color": "green",
        "finish_hint": "painted",
        "visual_description": "main vertical body of the machine",
    }
    pool = [
        {
            "material_id": army,
            "display_name": "Steel Painted Army Green",
            "description": "painted steel",
            "family": "metal",
            "colors": ["green"],
            "finishes": ["painted", "coated"],
            "appearance_profile": {
                "base_color_srgb": [0.28627464, 0.31372583, 0.11764688],
                "roughness": 0.32,
            },
        },
        {
            "material_id": cracked,
            "display_name": "Steel Painted Army Green Cracked Dirty",
            "description": "cracked dirty weathered painted steel",
            "family": "metal",
            "colors": ["green"],
            "finishes": ["painted", "cracked", "dirty", "worn"],
            "appearance_profile": {
                "base_color_srgb": [0.38431407, 0.41960748, 0.15686270],
                "roughness": 0.36,
            },
        },
        {
            "material_id": hammer,
            "display_name": "Hammer Paint Green",
            "description": "hammer paint",
            "family": "paint",
            "colors": ["green"],
            "finishes": ["coated"],
            "appearance_profile": {
                "base_color_srgb": [0.36078415, 0.48235258, 0.18823586],
                "roughness": 0.42,
            },
        },
    ]
    evidence = {
        "surface_class": "dielectric",
        "distinct_view_count": 4,
        "albedo": {
            "sample_count": 4,
            "median": [0.22941176, 0.53529412, 0.19803921],
            "mad": [0.01, 0.01, 0.01],
        },
        "metallic": {
            "sample_count": 4,
            "median": 0.11568628,
            "mad": 0.01,
            "iqr": 0.02,
        },
        "roughness": {
            "sample_count": 4,
            "median": 0.42745098,
            "mad": 0.0,
            "iqr": 0.001,
        },
        "suggestion": {
            "decision": "auto",
            "auto_parameter_eligible": True,
        },
    }

    candidates, audit = _shortlist_materials_with_audit(
        group,
        pool,
        family_reliable=False,
        mvinverse_pbr_evidence=evidence,
        allow_parameter_writes=False,
    )

    assert candidates[0]["material_id"] == hammer
    assert {item["material_id"] for item in candidates} == {army, cracked, hammer}
    assert audit["strategy"].endswith("/v12")
    assert audit["fixed_library_defaults_required"] is True
    assert audit["thumbnail_default_evidence_count"] == 0
    assert (
        audit["unobserved_fixed_effect_policy"]
        == "positive_reliable_semantics_required/v1"
    )
    assert audit["mvinverse_tunable_equivalence_dedup_count"] == 0
    assert all(
        "mvinverse_tunable_template" not in item["matched_fields"]
        for item in audit["ranking"]
    )


def test_immutable_confirmed_paint_keeps_substrate_aware_coating_hypotheses() -> None:
    group = {
        "group_id": "G06",
        "family_hint": "metal",
        "base_color": "green",
        "finish_hint": "painted",
        "visual_description": "green painted machine body",
    }
    pool = [
        {
            "material_id": (
                "mdl:vMaterials_2/Metal/Steel_Painted.mdl" "#Steel_Painted_Army_Green"
            ),
            "display_name": "Steel Painted Army Green",
            "description": "painted steel",
            "family": "metal",
            "colors": ["green"],
            "finishes": ["painted", "coated"],
        },
        {
            "material_id": ("mdl:vMaterials_2/Paint/Carpaint/Carpaint_Solid.mdl#Green"),
            "display_name": "Carpaint Solid Green",
            "description": "smooth green paint",
            "family": "paint",
            "colors": ["green"],
            "finishes": ["painted", "smooth"],
        },
        {
            "material_id": (
                "mdl:vMaterials_2/Metal/Aluminum_Anodized.mdl"
                "#Aluminum_Anodized_Grass_Green"
            ),
            "display_name": "Aluminum Anodized Grass Green",
            "description": "green anodized aluminum",
            "family": "metal",
            "colors": ["green"],
            "finishes": ["anodized", "coated"],
        },
        {
            "material_id": (
                "mdl:vMaterials_2/Paint/Hammer_Paint.mdl#Hammer_Paint_Green"
            ),
            "display_name": "Hammer Paint Green",
            "description": "hammer paint",
            "family": "paint",
            "colors": ["green"],
            "finishes": ["painted"],
        },
    ]
    evidence = {
        "surface_class": "dielectric",
        "distinct_view_count": 4,
        "albedo": {
            "sample_count": 4,
            "median": [0.23, 0.54, 0.20],
            "mad": [0.01, 0.01, 0.01],
        },
        "metallic": {
            "sample_count": 4,
            "median": 0.12,
            "mad": 0.01,
            "iqr": 0.02,
        },
        "roughness": {
            "sample_count": 4,
            "median": 0.43,
            "mad": 0.0,
            "iqr": 0.01,
        },
    }

    candidates, audit = _shortlist_materials_with_audit(
        group,
        pool,
        family_reliable=False,
        mvinverse_pbr_evidence=evidence,
        allow_parameter_writes=False,
    )

    policy = audit["surface_interpretation_policy"]
    assert policy["mode"] == "balanced_confirmed_applied_coating_physics"
    assert policy["required_coating_physics_templates"] == [
        "painted_engineering_metal",
        "generic_applied_paint",
        "conversion_coating",
    ]
    assert all(
        policy["selected_material_ids_by_coating_physics_template"][template]
        for template in policy["required_coating_physics_templates"]
    )
    assert policy["complete_required_coverage"] is True
    assert any(
        item["surface_interpretation"] == "conversion_coating" for item in candidates
    )


def test_immutable_confirmed_paint_never_replaces_selected_mdl() -> None:
    group = {
        "group_id": "G06",
        "family_hint": "metal",
        "base_color": "green",
        "finish_hint": "painted",
        "visual_description": "green painted industrial machine body",
    }
    pool = [
        {
            "material_id": (
                "mdl:vMaterials_2/Metal/Aluminum_Anodized.mdl"
                "#Aluminum_Anodized_Grass_Green"
            ),
            "display_name": "Aluminum Anodized Grass Green",
            "family": "metal",
            "colors": ["green"],
            "finishes": ["anodized"],
        },
        {
            "material_id": (
                "mdl:vMaterials_2/Metal/Steel_Painted.mdl" "#Steel_Painted_Army_Green"
            ),
            "display_name": "Steel Painted Army Green",
            "family": "metal",
            "colors": ["green"],
            "finishes": ["painted"],
        },
        {
            "material_id": (
                "mdl:vMaterials_2/Paint/Paint_Eggshell.mdl" "#Paint_Eggshell_Lime"
            ),
            "display_name": "Paint Eggshell Lime",
            "family": "paint",
            "colors": ["green"],
            "finishes": ["painted"],
        },
    ]
    evidence = {
        "surface_class": "dielectric",
        "distinct_view_count": 4,
        "albedo": {
            "sample_count": 4,
            "median": [0.23, 0.54, 0.20],
            "mad": [0.01, 0.01, 0.01],
        },
        "metallic": {
            "sample_count": 4,
            "median": 0.12,
            "mad": 0.01,
            "iqr": 0.02,
        },
        "roughness": {
            "sample_count": 4,
            "median": 0.43,
            "mad": 0.01,
            "iqr": 0.02,
        },
    }
    reliability = {
        "finish_hint": {
            "canonical_value": "painted",
            "selection_value": "painted",
            "reliable": True,
        },
        "visual_description": {"reliable": True},
    }
    candidates, retrieval = _shortlist_materials_with_audit(
        group,
        pool,
        limit=3,
        semantic_reliability=reliability,
        mvinverse_pbr_evidence=evidence,
        allow_parameter_writes=False,
    )
    chosen = {
        "schema_version": "qwen-palette-material-choice/v1",
        "group_id": "G06",
        "material_id": (
            "mdl:vMaterials_2/Metal/Aluminum_Anodized.mdl"
            "#Aluminum_Anodized_Grass_Green"
        ),
        "confidence": 0.99,
    }

    resolved, confirmed, basis, audit = _resolve_immutable_coating_physics_choice(
        chosen,
        confirmed=False,
        confirmation_basis="forward_reverse_disagreement",
        candidates=candidates,
        retrieval_audit=retrieval,
    )

    assert resolved["material_id"] == chosen["material_id"]
    assert confirmed is False
    assert basis == "forward_reverse_disagreement"
    assert audit["applied"] is False
    assert audit["mode"] == "immutable_selected_mdl_preserved"
    assert audit["selected_mdl_parameters_mutable"] is False


def test_fused_palette_keeps_primary_citations_and_other_view_singletons() -> None:
    def palette(view_id: str, groups: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "schema_version": PALETTE_SCHEMA_VERSION,
            "source_view_id": view_id,
            "groups": groups,
        }

    green = {
        "group_id": "G01",
        "family_hint": "metal",
        "base_color": "green",
        "finish_hint": "painted",
        "visual_description": "green enclosure",
        "boxes": [[10, 20, 400, 900]],
        "confidence": 0.9,
    }
    fusion = fuse_multiview_palettes(
        [
            palette("front", [green]),
            palette(
                "top",
                [
                    {**green, "boxes": [[30, 40, 450, 800]], "confidence": 0.95},
                    {
                        "group_id": "G02",
                        "family_hint": "plastic",
                        "base_color": "blue",
                        "finish_hint": "glossy",
                        "visual_description": "blue valve cap",
                        "boxes": [[700, 100, 760, 180]],
                        "confidence": 0.88,
                    },
                ],
            ),
        ]
    )

    legacy = _legacy_palette_from_fusion(fusion, primary_view_id="front")
    green_id = fusion["view_group_id_maps"]["front"]["G01"]
    blue_id = fusion["view_group_id_maps"]["top"]["G02"]
    groups = {group["group_id"]: group for group in legacy["groups"]}
    assert legacy["source_view_id"] == "front"
    assert groups[green_id]["boxes"] == [[10, 20, 400, 900]]
    assert groups[blue_id]["boxes"] == [[700, 100, 760, 180]]

    batches = _canonicalize_primary_batches(
        [
            {
                "batch_id": "B01",
                "mappings": [
                    {
                        "part_id": "P0001",
                        "group_id": "G01",
                        "mapping_confidence": 0.95,
                        "evidence_view_id": "front",
                        "evidence_box_index": 0,
                        "status": "matched",
                        "reason_code": "shape_and_location",
                    }
                ],
            }
        ],
        local_to_canonical=fusion["view_group_id_maps"]["front"],
        canonical_palette=legacy,
    )
    assert batches[0]["mappings"][0]["group_id"] == green_id


def test_multiview_pre_filter_count_does_not_count_cameras_as_materials() -> None:
    group = {
        "group_id": "G01",
        "family_hint": "metal",
        "base_color": "green",
        "finish_hint": "painted",
        "visual_description": "green enclosure",
        "boxes": [[0, 0, 1000, 1000]],
        "confidence": 0.9,
    }
    canonical = {
        "schema_version": PALETTE_SCHEMA_VERSION,
        "source_view_id": "front",
        "groups": [group],
    }
    candidates = [
        {
            "model_palette": {
                **canonical,
                "source_view_id": view_id,
            }
        }
        for view_id in ("front", "side", "top", "iso")
    ]

    assert (
        _multiview_pre_filter_group_count(
            candidates,
            canonical_palette=canonical,
        )
        == 1
    )


def test_multiview_pre_filter_count_retains_distinct_model_colours() -> None:
    green = {
        "group_id": "G01",
        "family_hint": "metal",
        "base_color": "green",
        "finish_hint": "painted",
        "visual_description": "green enclosure",
        "boxes": [[0, 0, 1000, 1000]],
        "confidence": 0.9,
    }
    canonical = {
        "schema_version": PALETTE_SCHEMA_VERSION,
        "source_view_id": "front",
        "groups": [green],
    }
    candidates = [
        {"model_palette": canonical},
        {
            "model_palette": {
                **canonical,
                "source_view_id": "top",
                "groups": [
                    green,
                    {
                        **green,
                        "group_id": "G02",
                        "base_color": "orange",
                        "visual_description": "orange accent",
                    },
                ],
            }
        },
    ]

    assert (
        _multiview_pre_filter_group_count(
            candidates,
            canonical_palette=canonical,
        )
        == 2
    )


def test_legacy_adapter_accepts_multiview_union_larger_than_single_view_limit() -> None:
    signatures = [
        ("metal", "red", "painted"),
        ("metal", "green", "painted"),
        ("metal", "blue", "painted"),
        ("metal", "yellow", "painted"),
        ("metal", "orange", "painted"),
        ("metal", "black", "bare"),
        ("metal", "silver", "bare"),
        ("plastic", "white", "glossy"),
        ("plastic", "black", "matte"),
        ("rubber", "black", "matte"),
        ("rubber", "gray", "matte"),
        ("glass", "clear", "glossy"),
        ("fabric", "red", "matte"),
        ("ceramic", "white", "glossy"),
        ("plastic", "cyan", "glossy"),
        ("plastic", "pink", "glossy"),
        ("metal", "brown", "bare"),
        ("metal", "gray", "brushed"),
        ("plastic", "green", "matte"),
        ("rubber", "blue", "matte"),
    ]

    def palette(view_id: str, offset: int) -> dict[str, Any]:
        return {
            "schema_version": PALETTE_SCHEMA_VERSION,
            "source_view_id": view_id,
            "groups": [
                {
                    "group_id": f"G{index + 1:02d}",
                    "family_hint": family,
                    "base_color": color,
                    "finish_hint": finish,
                    "visual_description": f"{color} {family}",
                    "boxes": [[10, 10, 20, 20]],
                    "confidence": 0.9,
                }
                for index, (family, color, finish) in enumerate(
                    signatures[offset : offset + 10]
                )
            ],
        }

    fusion = fuse_multiview_palettes([palette("front", 0), palette("rear", 10)])
    legacy = _legacy_palette_from_fusion(fusion, primary_view_id="front")

    # Red, green, and blue occur once per view but describe conflicting known
    # substances/finishes.  They must remain separate instead of being merged
    # merely because their coarse colour label matches.
    assert len(fusion["canonical_palette"]["groups"]) == 20
    assert len(legacy["groups"]) == 20


def test_staged_mvinverse_mode_runs_before_qwen_and_binds_evidence(
    tmp_path: Path, monkeypatch: Any
) -> None:
    def image(name: str, color: str) -> Path:
        path = tmp_path / name
        Image.new("RGB", (56, 70), color).save(path)
        return path

    front = image("front.png", "green")
    side = image("side.png", "darkgreen")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "parts": [
                    {
                        "part_id": "P0001",
                        "renders": [{"view_id": "front", "visible_pixels": 0}],
                    }
                ],
                "render_set": {
                    "views": [{"view_id": "front"}],
                    "best_highlights": {},
                },
            }
        ),
        encoding="utf-8",
    )
    events: list[str] = []
    adapter_calls: list[dict[str, Any]] = []
    fusion_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_adapter(**kwargs: Any) -> dict[str, Any]:
        events.append("mvinverse")
        adapter_calls.append(kwargs)
        return {
            "status": "SUCCESS",
            "inputs": {
                "source_views": [
                    {"index": 0, "view_id": "ref_front"},
                    {"index": 1, "view_id": "ref_side"},
                ]
            },
        }

    class FakeClient:
        def __init__(self, **_kwargs: Any) -> None:
            self.repair_events: list[dict[str, Any]] = []

        def extract_palette(
            self, reference_view: dict[str, str], *, run_label: str | None = None
        ) -> dict[str, Any]:
            events.append(f"qwen:{reference_view['id']}")
            return {
                "schema_version": PALETTE_SCHEMA_VERSION,
                "source_view_id": reference_view["id"],
                "groups": [
                    {
                        "group_id": "G01",
                        "family_hint": "metal",
                        "base_color": "green",
                        "finish_hint": "painted",
                        "visual_description": "green painted housing",
                        "boxes": [[100, 100, 900, 900]],
                        "confidence": 0.95,
                    }
                ],
            }

    def fake_filter(
        document: dict[str, Any],
        _path: Path,
        *,
        mask_path: Path | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del mask_path
        return document, {
            "groups": [
                {
                    "group_id": "G01",
                    "accepted": True,
                    "boxes": [
                        {
                            "box": [100, 100, 900, 900],
                            "accepted": True,
                            "sampled_pixels": 1000,
                            "matching_pixel_count": 900,
                            "foreground_pixels": 1000,
                        }
                    ],
                }
            ]
        }

    evidence = {
        "schema_version": "qwen-mvinverse-pbr-evidence/v1",
        "inputs": {"integrity_verified": True},
        "views": [],
        "groups": [],
        "summary": {},
    }

    def fake_fusion(*args: Any, **kwargs: Any) -> dict[str, Any]:
        fusion_calls.append((args, kwargs))
        return evidence

    monkeypatch.setattr(run_staged_local, "run_mvinverse_adapter", fake_adapter)
    monkeypatch.setattr(
        run_staged_local,
        "TransformersQwen3VLRunner",
        lambda *_a, **_kw: SimpleNamespace(preflight=lambda: None),
    )
    monkeypatch.setattr(run_staged_local, "LocalStagedQwenClient", FakeClient)
    monkeypatch.setattr(
        run_staged_local, "filter_palette_by_image_evidence", fake_filter
    )
    monkeypatch.setattr(
        run_staged_local,
        "build_mvinverse_evidence_from_manifest",
        fake_fusion,
    )
    monkeypatch.setattr(
        run_staged_local, "validate_mvinverse_evidence", lambda value: value
    )

    output = tmp_path / "output"
    assert (
        run_staged_local.main(
            [
                "--registry",
                str(registry_path),
                "--reference",
                f"ref_front={front}",
                "--reference",
                f"ref_side={side}",
                "--catalog",
                str(tmp_path / "unused_catalog.json"),
                "--model-path",
                str(tmp_path / "model"),
                "--output-dir",
                str(output),
                "--cad-view",
                "front",
                "--stop-after",
                "palette",
                "--mvinverse-mode",
                "run",
                "--mvinverse-repo",
                str(tmp_path / "external_repo"),
                "--mvinverse-python",
                str(tmp_path / "external_python"),
                "--mvinverse-checkpoint",
                str(tmp_path / "external_checkpoint"),
                "--acknowledge-mvinverse-noncommercial",
            ]
        )
        == 0
    )

    assert events[:3] == ["mvinverse", "qwen:ref_front", "qwen:ref_side"]
    assert adapter_calls[0]["reuse_existing"] is False
    assert adapter_calls[0]["acknowledge_noncommercial"] is True
    assert fusion_calls[0][1]["frame_indices"] == {"ref_front": 0, "ref_side": 1}
    assert fusion_calls[0][1]["inference_ledger"] == (
        output / "mvinverse" / "mvinverse_inference_ledger.json"
    )
    assert (
        json.loads((output / "mvinverse_pbr_evidence.json").read_text(encoding="utf-8"))
        == evidence
    )


def test_face_region_evidence_command_uses_rendered_registry(
    tmp_path: Path, monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text("{}\n", encoding="utf-8")
    executable = tmp_path / "isaac-python.sh"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    destination = tmp_path / "face_regions"
    calls: list[dict[str, Any]] = []

    child_progress = (
        '@@ASSET_PROGRESS {"schema_version":"asset-pipeline-progress/v1",'
        '"scope":"qwen_material_pipeline","stage":"face_topology",'
        '"state":"update","current":1,"total":1,"unit":"parts",'
        '"detail":"P0001 completed"}\n'
    )

    class FakeProcess:
        def __init__(self, command: list[str], **kwargs: Any) -> None:
            calls.append({"command": command, **kwargs})
            self.stdout = io.StringIO(child_progress + "generated\n")
            self.stderr = io.StringIO("ordinary diagnostic\n")
            self.returncode = 0
            output_index = command.index("--output-dir") + 1
            output_dir = Path(command[output_index])
            output_dir.mkdir(parents=True)
            (output_dir / "manifest.json").write_text("{}\n", encoding="utf-8")

        def wait(self, timeout: int | None = None) -> int:
            calls[0].setdefault("wait_timeouts", []).append(timeout)
            return self.returncode

        def kill(self) -> None:
            raise AssertionError("successful process must not be killed")

    monkeypatch.setattr(run_staged_local.subprocess, "Popen", FakeProcess)

    manifest = run_staged_local._run_or_reuse_face_region_evidence(
        registry=registry,
        output_dir=destination,
        python_executable=executable,
        reuse_existing=False,
        timeout_seconds=123,
    )

    assert manifest == (destination / "manifest.json").resolve()
    assert len(calls) == 1
    command = calls[0]["command"]
    resolved_registry = str(registry.resolve())
    assert command[command.index("--registry") + 1] == resolved_registry
    assert command[command.index("--rendered-registry") + 1] == resolved_registry
    assert calls[0]["wait_timeouts"] == [123]
    assert calls[0]["stdout"] is run_staged_local.subprocess.PIPE
    assert calls[0]["stderr"] is run_staged_local.subprocess.PIPE
    assert calls[0]["text"] is True
    assert calls[0]["env"]["PYTHONUNBUFFERED"] == "1"
    for name in (
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_PROMPT_MODIFIER",
        "VIRTUAL_ENV",
        "PYTHONHOME",
        "PYTHONPATH",
    ):
        assert name not in calls[0]["env"]
    assert (destination / "face_region.stdout.log").read_text(encoding="utf-8") == (
        child_progress + "generated\n"
    )
    assert (destination / "face_region.stderr.log").read_text(
        encoding="utf-8"
    ) == "ordinary diagnostic\n"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == child_progress


def test_face_region_streaming_timeout_preserves_logs_and_fails_closed(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text("{}\n", encoding="utf-8")
    executable = tmp_path / "isaac-python.sh"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    destination = tmp_path / "face_regions"
    child_progress = (
        '@@ASSET_PROGRESS {"schema_version":"asset-pipeline-progress/v1",'
        '"scope":"qwen_material_pipeline","stage":"face_topology",'
        '"state":"start","current":0,"total":2,"unit":"parts",'
        '"detail":"started"}\n'
    )
    processes: list[Any] = []

    class TimeoutProcess:
        def __init__(self, _command: list[str], **_kwargs: Any) -> None:
            self.stdout = io.StringIO("partial output\n")
            self.stderr = io.StringIO(child_progress + "partial diagnostic\n")
            self.returncode: int | None = None
            self.killed = False
            processes.append(self)

        def wait(self, timeout: int | None = None) -> int:
            if timeout is not None and not self.killed:
                raise run_staged_local.subprocess.TimeoutExpired(
                    ["face_regions.py"], timeout
                )
            self.returncode = -9
            return self.returncode

        def kill(self) -> None:
            self.killed = True

    monkeypatch.setattr(run_staged_local.subprocess, "Popen", TimeoutProcess)

    with pytest.raises(run_staged_local.subprocess.TimeoutExpired):
        run_staged_local._run_or_reuse_face_region_evidence(
            registry=registry,
            output_dir=destination,
            python_executable=executable,
            reuse_existing=False,
            timeout_seconds=7,
        )

    assert processes[0].killed is True
    assert (destination / "face_region.stdout.log").read_text(
        encoding="utf-8"
    ) == "partial output\n"
    assert (destination / "face_region.stderr.log").read_text(encoding="utf-8") == (
        child_progress + "partial diagnostic\n"
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == child_progress


def test_face_region_projection_uses_renderer_evidence_view_bank(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "render_set": {
                    "part_evidence_view_ids": ["front", "iso", "top"],
                    "views": [
                        {"view_id": "front"},
                        {"view_id": "iso"},
                        {"view_id": "top"},
                        {"view_id": "orbit_01"},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    assert run_staged_local._preferred_face_region_projection_views(registry) == [
        "front",
        "iso",
        "top",
    ]


def test_face_region_projection_intersects_stale_pose_bank_with_calibrated_views(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "camera_calibrated_registry.json"
    registry.write_text(
        json.dumps(
            {
                "render_set": {
                    "part_evidence_view_ids": [
                        "front",
                        "iso",
                        "left",
                        "rear",
                        "right",
                        "top",
                    ],
                    "views": [
                        {"view_id": "front"},
                        {"view_id": "side"},
                        {"view_id": "top"},
                        {"view_id": "iso"},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    assert run_staged_local._preferred_face_region_projection_views(registry) == [
        "front",
        "iso",
        "top",
    ]


@pytest.mark.parametrize(
    "recover_face_subset",
    [True, False],
    ids=["face-subset-recovered", "insufficient-face-evidence"],
)
def test_unattended_runner_chains_uniform_plan_through_face_recovery_or_skip(
    tmp_path: Path, monkeypatch: Any, recover_face_subset: bool
) -> None:
    def image(name: str, color: str) -> Path:
        path = tmp_path / name
        Image.new("RGB", (64, 64), color).save(path)
        return path

    reference = image("reference.png", "green")
    rgb = image("front_rgb.png", "gray")
    part_ids = image("front_ids.png", "red")
    highlight = image("P0001_front.png", "orange")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "qwen-material-parts/v1",
                "parts": [
                    {
                        "part_id": "P0001",
                        "world_bbox": [[0, 0, 0], [1, 1, 1]],
                        "renders": [
                            {
                                "view_id": "front",
                                "visible_pixels": 512,
                                "highlight_path": str(highlight),
                            }
                        ],
                    }
                ],
                "render_set": {
                    "views": [
                        {
                            "view_id": "front",
                            "rgb": str(rgb),
                            "part_ids": str(part_ids),
                        }
                    ],
                    "best_highlights": {"P0001": str(highlight)},
                },
            }
        ),
        encoding="utf-8",
    )
    events: list[str] = []
    face_manifest = tmp_path / "face_regions" / "manifest.json"
    uniform_plan = {
        "schema_version": "1.0",
        "assignments": [
            {
                "part_id": "P0001",
                "material_id": "M01",
                "status": "auto",
                "confidence": 0.99,
                "evidence_views": ["ref"],
            }
        ],
    }
    recovered_plan = {
        "schema_version": "1.0",
        "assignments": [
            {
                **uniform_plan["assignments"][0],
                "preserve_parent_material_binding": True,
                "face_indices": [0, 1, 2],
            }
        ],
    }
    final_plan = recovered_plan if recover_face_subset else uniform_plan
    recovery_summary = (
        {"recovered_part_count": 1, "recovered_face_count": 3}
        if recover_face_subset
        else {
            "status": "SKIPPED_INSUFFICIENT_EVIDENCE",
            "skip_reason_codes": ["INSUFFICIENT_TRUSTED_REGISTERED_VIEWS"],
            "recovered_part_count": 0,
            "recovered_face_count": 0,
        }
    )
    recovery_calls: list[dict[str, Any]] = []

    def fake_adapter(**_kwargs: Any) -> dict[str, Any]:
        events.append("mvinverse")
        return {
            "status": "SUCCESS",
            "inputs": {"source_views": [{"index": 0, "view_id": "ref"}]},
        }

    def fake_face_regions(**kwargs: Any) -> Path:
        events.append("face_regions")
        assert kwargs["registry"] == registry_path
        face_manifest.parent.mkdir(parents=True)
        face_manifest.write_text("{}\n", encoding="utf-8")
        return face_manifest.resolve()

    class FakeClient:
        def __init__(self, **_kwargs: Any) -> None:
            self.repair_events: list[dict[str, Any]] = []

        def extract_palette(
            self, reference_view: dict[str, str], *, run_label: str | None = None
        ) -> dict[str, Any]:
            del run_label
            return {
                "schema_version": PALETTE_SCHEMA_VERSION,
                "source_view_id": reference_view["id"],
                "groups": [
                    {
                        "group_id": "G01",
                        "family_hint": "metal",
                        "base_color": "green",
                        "finish_hint": "painted",
                        "visual_description": "green painted housing",
                        "boxes": [[100, 100, 900, 900]],
                        "confidence": 0.98,
                    }
                ],
            }

        def map_part_batch(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "batch_id": kwargs["batch_id"],
                "mappings": [
                    {
                        "part_id": "P0001",
                        "group_id": None,
                        "status": "unknown",
                        "mapping_confidence": 0.0,
                    }
                ],
            }

        def choose_group_material(self, **_kwargs: Any) -> dict[str, Any]:
            return {"material_id": "M01", "confidence": 0.96}

    def fake_palette_filter(
        document: dict[str, Any],
        _path: Path,
        *,
        mask_path: Path | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del mask_path
        return document, {
            "groups": [
                {
                    "group_id": "G01",
                    "accepted": True,
                    "boxes": [
                        {
                            "box": [100, 100, 900, 900],
                            "accepted": True,
                            "sampled_pixels": 1000,
                            "matching_pixel_count": 900,
                            "foreground_pixels": 1000,
                        }
                    ],
                }
            ]
        }

    mvinverse_evidence = {
        "schema_version": "qwen-mvinverse-pbr-evidence/v1",
        "inputs": {"integrity_verified": True},
        "views": [],
        "groups": [],
        "summary": {},
    }
    mvinverse_evidence_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_build_mvinverse_evidence(*args: Any, **kwargs: Any) -> dict[str, Any]:
        mvinverse_evidence_calls.append((args, kwargs))
        return mvinverse_evidence

    geometry_risk = {
        "summary": {
            "multi_material_risk_part_ids": [],
            "multi_material_risk_part_count": 0,
        }
    }
    mapping_consensus = {
        "gate_batches": [],
        "audit": {"summary": {"candidate_part_count": 0}},
    }
    spatial_report = {"summary": {"trusted_view_count": 1}}
    spatial_gate = {
        "gate_batches": [],
        "audit": {"summary": {"resolved_part_count": 0}},
    }
    gate_report = {
        "summary": {"auto_count": 1},
        "auto_material_plan": uniform_plan,
    }

    class FakeCatalog:
        @staticmethod
        def load(*_args: Any, **_kwargs: Any) -> object:
            return object()

    def fake_parameterize(**kwargs: Any) -> dict[str, Any]:
        events.append("parameterize")
        assert kwargs["auto_material_plan"] is uniform_plan
        return {
            "summary": {"parameterized_assignment_count": 1},
            "material_plan": uniform_plan,
        }

    def fake_recovery(**kwargs: Any) -> dict[str, Any]:
        events.append("face_recovery")
        recovery_calls.append(kwargs)
        assert kwargs["base_material_plan"] is uniform_plan
        assert kwargs["face_region_manifest"] == face_manifest.resolve()
        assert kwargs["spatial_mapping_report"] is spatial_report
        assert kwargs["group_materials"]["selections"][0]["group_id"] == "G01"
        assert kwargs["material_choice_audit"]["G01"]["confirmed"] is True
        return {
            "summary": recovery_summary,
            "material_plan": final_plan,
        }

    monkeypatch.setattr(run_staged_local, "run_mvinverse_adapter", fake_adapter)
    monkeypatch.setattr(
        run_staged_local,
        "_run_or_reuse_face_region_evidence",
        fake_face_regions,
    )
    monkeypatch.setattr(
        run_staged_local,
        "TransformersQwen3VLRunner",
        lambda *_a, **_kw: SimpleNamespace(preflight=lambda: None),
    )
    monkeypatch.setattr(run_staged_local, "LocalStagedQwenClient", FakeClient)
    monkeypatch.setattr(
        run_staged_local, "filter_palette_by_image_evidence", fake_palette_filter
    )
    monkeypatch.setattr(
        run_staged_local,
        "build_mvinverse_evidence_from_manifest",
        fake_build_mvinverse_evidence,
    )
    monkeypatch.setattr(
        run_staged_local, "validate_mvinverse_evidence", lambda value: value
    )
    monkeypatch.setattr(
        run_staged_local, "build_geometry_risk", lambda *_a, **_kw: geometry_risk
    )
    monkeypatch.setattr(
        run_staged_local, "validate_geometry_risk", lambda *_a, **_kw: None
    )
    monkeypatch.setattr(
        run_staged_local, "build_view_group_id_maps", lambda *_a, **_kw: {"ref": {}}
    )
    monkeypatch.setattr(
        run_staged_local,
        "canonicalize_view_batch_mappings",
        lambda *_a, **_kw: [],
    )
    monkeypatch.setattr(
        run_staged_local,
        "apply_mapping_consensus_to_batches",
        lambda *_a, **_kw: mapping_consensus,
    )
    monkeypatch.setattr(
        run_staged_local,
        "build_spatial_mapping_report",
        lambda *_a, **_kw: spatial_report,
    )
    monkeypatch.setattr(
        run_staged_local,
        "apply_spatial_gate_to_batches",
        lambda *_a, **_kw: spatial_gate,
    )
    monkeypatch.setattr(run_staged_local, "MaterialCatalog", FakeCatalog)
    monkeypatch.setattr(
        run_staged_local,
        "_catalog_pool",
        lambda *_a, **_kw: [
            {
                "material_id": "M01",
                "display_name": "Green Paint",
                "description": "green painted metal",
                "family": "metal",
                "colors": ["green"],
                "finishes": ["painted"],
            }
        ],
    )
    monkeypatch.setattr(
        run_staged_local,
        "merge_staged_results",
        lambda **_kw: {
            "audit": {"assignment_count": 1},
            "material_plan": uniform_plan,
        },
    )
    monkeypatch.setattr(
        run_staged_local,
        "build_part_view_evidence",
        lambda **_kw: {
            "schema_version": "qwen-material-view-evidence/v1",
            "predictions": [],
        },
    )
    monkeypatch.setattr(
        run_staged_local,
        "evaluate_confidence_gate",
        lambda *_a, **_kw: gate_report,
    )
    monkeypatch.setattr(
        run_staged_local, "parameterize_auto_material_plan", fake_parameterize
    )
    monkeypatch.setattr(run_staged_local, "build_face_material_recovery", fake_recovery)

    output = tmp_path / "output"
    assert (
        run_staged_local.main(
            [
                "--registry",
                str(registry_path),
                "--reference",
                f"ref={reference}",
                "--catalog",
                str(tmp_path / "catalog.json"),
                "--model-path",
                str(tmp_path / "model"),
                "--output-dir",
                str(output),
                "--cad-view",
                "front",
                "--mvinverse-mode",
                "run",
                "--mvinverse-repo",
                str(tmp_path / "mvinverse-repo"),
                "--mvinverse-python",
                str(tmp_path / "mvinverse-python"),
                "--mvinverse-checkpoint",
                str(tmp_path / "checkpoint"),
                "--acknowledge-mvinverse-noncommercial",
                "--face-region-python",
                str(tmp_path / "isaac-python.sh"),
            ]
        )
        == 0
    )

    assert events == ["mvinverse", "face_regions", "parameterize", "face_recovery"]
    assert len(mvinverse_evidence_calls) == 1
    assert Path(mvinverse_evidence_calls[0][0][1]) == (
        output / "mvinverse_reference_manifest.json"
    )
    assert len(recovery_calls) == 1
    assert (
        json.loads(
            (output / "autonomous_uniform_material_plan.json").read_text(
                encoding="utf-8"
            )
        )
        == uniform_plan
    )
    assert json.loads(
        (output / "face_material_recovery.json").read_text(encoding="utf-8")
    ) == {
        "summary": recovery_summary,
        "material_plan": final_plan,
    }
    assert (
        json.loads(
            (output / "autonomous_material_plan.json").read_text(encoding="utf-8")
        )
        == final_plan
    )
    unattended = json.loads(
        (output / "unattended_result.json").read_text(encoding="utf-8")
    )
    assert unattended["state"] == "READY_TO_APPLY"
    assert unattended["face_material_recovery"] == recovery_summary


def test_palette_quality_and_selection_are_deterministic() -> None:
    def candidate(
        reference_id: str,
        colors: list[str],
        *,
        pixel_fraction: float = 0.5,
        confidence: float = 0.8,
    ) -> dict[str, Any]:
        groups = [
            {
                "group_id": f"G{index:02d}",
                "base_color": color,
                "confidence": confidence,
            }
            for index, color in enumerate(colors, start=1)
        ]
        audit = {
            "groups": [
                {
                    "group_id": group["group_id"],
                    "boxes": [
                        {
                            "accepted": True,
                            "sampled_pixels": 1000,
                            "matching_pixel_count": int(1000 * pixel_fraction),
                            "foreground_pixels": 800,
                        }
                    ],
                }
                for group in groups
            ]
        }
        quality = _palette_quality_metrics({"groups": groups}, audit)
        return {
            "reference_id": reference_id,
            "status": "usable",
            "quality": quality,
        }

    first = candidate("first", ["white"], pixel_fraction=0.9, confidence=0.99)
    diverse = candidate("diverse", ["white", "black"], pixel_fraction=0.3)
    same_size_less_diverse = candidate(
        "duplicate", ["white", "white"], pixel_fraction=0.9
    )

    assert (
        _select_palette_candidate([first, same_size_less_diverse, diverse], "auto")[
            "reference_id"
        ]
        == "diverse"
    )
    assert _select_palette_candidate([diverse, first], "first") is first
    tied = candidate("tied", ["white", "black"], pixel_fraction=0.3)
    assert _select_palette_candidate([diverse, tied], "auto") is diverse
    with pytest.raises(ValueError, match="does not match"):
        _select_palette_candidate([first], "missing")


def test_required_usable_palette_view_count_combines_absolute_and_ratio_gates() -> None:
    assert (
        run_staged_local._required_usable_palette_view_count(
            reference_count=4,
            minimum_views=2,
            minimum_ratio=0.5,
        )
        == 2
    )
    assert (
        run_staged_local._required_usable_palette_view_count(
            reference_count=4,
            minimum_views=1,
            minimum_ratio=0.75,
        )
        == 3
    )
    with pytest.raises(ValueError, match="between zero and one"):
        run_staged_local._required_usable_palette_view_count(
            reference_count=4,
            minimum_views=1,
            minimum_ratio=1.1,
        )


def test_mapping_verification_prefers_strong_non_primary_views() -> None:
    candidates = [
        {
            "reference_id": "front",
            "status": "usable",
            "quality": {"selection_key": [8, 5, 0.9]},
        },
        {
            "reference_id": "side",
            "status": "usable",
            "quality": {"selection_key": [7, 6, 0.8]},
        },
        {
            "reference_id": "top",
            "status": "usable",
            "quality": {"selection_key": [9, 6, 0.95]},
        },
        {
            "reference_id": "iso",
            "status": "usable",
            "quality": {"selection_key": [6, 4, 0.7]},
        },
    ]

    selected = run_staged_local._mapping_verification_candidates(
        candidates,
        primary_reference_id="top",
        maximum_views=2,
        eligible_view_ids={"front", "side", "top", "iso"},
    )

    assert [item["reference_id"] for item in selected] == ["front", "side"]


def test_mapping_verification_exhaustive_mode_preserves_usable_order() -> None:
    candidates = [
        {
            "reference_id": "front",
            "status": "usable",
            "quality": {"selection_key": [1]},
        },
        {
            "reference_id": "side",
            "status": "failed",
            "quality": {"selection_key": [9]},
        },
        {
            "reference_id": "iso",
            "status": "usable",
            "quality": {"selection_key": [2]},
        },
    ]

    selected = run_staged_local._mapping_verification_candidates(
        candidates,
        primary_reference_id="front",
        maximum_views=0,
        eligible_view_ids={"front", "iso"},
    )

    assert [item["reference_id"] for item in selected] == ["front", "iso"]


def test_reference_manifest_separates_palette_success_from_failure() -> None:
    usable = run_staged_local._reference_manifest_source_view(
        {
            "reference_id": "top",
            "image": "/images/top.png",
            "mask": None,
            "status": "usable",
            "normalized_palette_path": "/palettes/top.json",
            "palette_failure_artifact_path": "/failures/stale.json",
        }
    )
    assert usable["palette_status"] == "usable"
    assert usable["palette_path"] == "/palettes/top.json"
    assert "palette_failure_artifact" not in usable

    unusable = run_staged_local._reference_manifest_source_view(
        {
            "reference_id": "front",
            "image": "/images/front.png",
            "mask": None,
            "status": "unusable",
            # A stale legacy path must never be advertised as normalized.
            "normalized_palette_path": "/palettes/not-a-palette.json",
            "palette_failure_artifact_path": "/failures/front.json",
            "evidence_audit_path": "/audits/front.json",
        }
    )
    assert unusable["palette_status"] == "unusable"
    assert unusable["palette_failure_artifact"] == "/failures/front.json"
    assert "palette_path" not in unusable
    assert "normalized" not in unusable["palette_artifacts"]


def test_material_shortlist_persists_deterministic_retrieval_margin() -> None:
    group = {
        "group_id": "G01",
        "family_hint": "metal",
        "base_color": "green",
        "finish_hint": "painted",
        "visual_description": "green painted machine body",
    }
    pool = [
        {
            "material_id": "mdl:green",
            "family": "metal",
            "colors": ["green"],
            "finishes": ["painted"],
            "display_name": "Green painted metal",
        },
        {
            "material_id": "mdl:black",
            "family": "metal",
            "colors": ["black"],
            "finishes": ["painted"],
            "display_name": "Black painted metal",
        },
        {
            "material_id": "mdl:plastic",
            "family": "plastic",
            "colors": ["green"],
            "finishes": ["painted"],
            "display_name": "Green painted plastic",
        },
    ]

    candidates, audit = _shortlist_materials_with_audit(group, pool, limit=2)

    assert [item["material_id"] for item in candidates] == [
        "mdl:green",
        "mdl:black",
    ]
    assert candidates[0]["retrieval_rank"] == 1
    assert candidates[0]["retrieval_score"] > candidates[1]["retrieval_score"]
    assert candidates[0]["retrieval_matched_fields"][:3] == [
        "family",
        "color",
        "finish",
    ]
    assert audit["family_pool_available"] is True
    assert audit["family_pool_used"] is False
    assert audit["eligible_pool_count"] == 3
    assert audit["runner_up_score"] == candidates[1]["retrieval_score"]
    assert audit["score_margin"] > 0
    assert audit["normalized_margin"] > 0
    assert audit["margin_available"] is True


def test_material_shortlist_rejects_missing_pool_and_bad_limit() -> None:
    group = {
        "group_id": "G01",
        "family_hint": "metal",
        "base_color": "green",
        "finish_hint": "painted",
        "visual_description": "green body",
    }

    with pytest.raises(ValueError, match="pool cannot be empty"):
        _shortlist_materials_with_audit(group, [])
    with pytest.raises(ValueError, match="positive integer"):
        _shortlist_materials_with_audit(group, [{"material_id": "mdl:a"}], limit=0)


def test_confirmed_family_excludes_same_color_wrong_family_candidate() -> None:
    group = {
        "group_id": "G05",
        "family_hint": "plastic",
        "base_color": "blue",
        "finish_hint": "painted",
        "visual_description": "blue plastic housing",
    }
    pool = [
        {
            "material_id": "mdl:painted-blue-steel",
            "family": "metal",
            "colors": ["blue"],
            "finishes": ["painted"],
            "display_name": "Blue painted steel",
        },
        {
            "material_id": "mdl:opaque-blue-plastic",
            "family": "plastic",
            "colors": ["blue"],
            "finishes": ["smooth"],
            "display_name": "Opaque blue plastic",
        },
    ]

    candidates, audit = _shortlist_materials_with_audit(
        group,
        pool,
        family_reliable=True,
    )

    assert [item["material_id"] for item in candidates] == ["mdl:opaque-blue-plastic"]
    assert audit["family_pool_used"] is True
    assert audit["eligible_pool_count"] == 1


def test_visual_shortlist_searches_across_confirmed_material_families() -> None:
    group = {
        "group_id": "G05",
        "family_hint": "plastic",
        "base_color": "blue",
        "finish_hint": "painted",
        "visual_description": "blue housing",
    }
    pool = [
        {
            "material_id": "mdl:painted-blue-steel",
            "family": "metal",
            "colors": ["blue"],
            "finishes": ["painted"],
            "display_name": "Blue painted steel",
        },
        {
            "material_id": "mdl:opaque-blue-plastic",
            "family": "plastic",
            "colors": ["blue"],
            "finishes": ["smooth"],
            "display_name": "Opaque blue plastic",
        },
    ]

    candidates, audit = _shortlist_materials_with_audit(
        group,
        pool,
        family_reliable=True,
        visual_similarity_first=True,
    )

    assert {item["material_id"] for item in candidates} == {
        "mdl:painted-blue-steel",
        "mdl:opaque-blue-plastic",
    }
    assert audit["family_pool_used"] is False
    assert audit["eligible_pool_count"] == 2
    assert audit["strategy"] == "visual_mvinverse_similarity_score/v1"


def test_material_shortlist_uses_material_identifier_for_copper_semantics() -> None:
    group = {
        "group_id": "G01",
        "family_hint": "metal",
        "base_color": "orange",
        "finish_hint": "glossy",
        "visual_description": "bare copper pneumatic tube",
    }
    pool = [
        {
            "material_id": "mdl:Base/Metals/Copper.mdl#Copper",
            "mdl_path": "Base/Metals/Copper.mdl",
            "sub_identifier": "Copper",
            "display_name": "Omni PBR",
            "description": "generic PBR",
            "family": "metal",
            "colors": [],
            "finishes": [],
        },
        {
            "material_id": "mdl:painted-orange",
            "display_name": "Orange paint",
            "description": "painted steel",
            "family": "metal",
            "colors": ["orange"],
            "finishes": ["glossy"],
        },
    ]

    candidates, _audit = _shortlist_materials_with_audit(group, pool, limit=2)

    assert candidates[0]["material_id"].endswith("Copper.mdl#Copper")


def test_material_shortlist_family_hint_does_not_exclude_copper() -> None:
    group = {
        "group_id": "G01",
        "family_hint": "plastic",
        "base_color": "orange",
        "finish_hint": "glossy",
        "visual_description": "bare copper pneumatic tube",
    }
    pool = [
        {
            "material_id": f"mdl:plastic-{index}",
            "display_name": f"Orange plastic {index}",
            "family": "plastic",
            "colors": ["orange"],
            "finishes": ["glossy"],
        }
        for index in range(6)
    ] + [
        {
            "material_id": "mdl:Base/Metals/Copper.mdl#Copper",
            "mdl_path": "Base/Metals/Copper.mdl",
            "sub_identifier": "Copper",
            "display_name": "Copper",
            "family": "metal",
            "colors": ["orange"],
            "finishes": ["bare"],
        }
    ]

    candidates, audit = _shortlist_materials_with_audit(group, pool, limit=4)

    assert candidates[0]["material_id"].endswith("Copper.mdl#Copper")
    assert audit["eligible_pool_count"] == len(pool)
    assert audit["family_pool_used"] is False


def test_material_shortlist_keeps_intrinsically_colored_base_metals_available() -> None:
    group = {
        "group_id": "G01",
        "family_hint": "metal",
        "base_color": "orange",
        "finish_hint": "painted",
        "visual_description": "orange curved tube",
    }
    pool = [
        {
            "material_id": "mdl:painted-orange",
            "display_name": "Orange painted steel",
            "family": "metal",
            "colors": ["orange"],
            "finishes": ["painted"],
        },
        *[
            {
                "material_id": f"mdl:Base/Metals/{name}.mdl#{name}",
                "display_name": name,
                "sub_identifier": name,
                "family": "metal",
                "colors": [],
                "finishes": ["bare"],
            }
            for name in ("Copper", "Brass", "Bronze")
        ],
        {
            "material_id": "mdl:plain-steel",
            "display_name": "Plain steel",
            "family": "metal",
            "colors": [],
            "finishes": ["bare"],
        },
    ]

    candidates, audit = _shortlist_materials_with_audit(group, pool, limit=4)

    assert candidates[0]["material_id"] == "mdl:painted-orange"
    assert {
        item["material_id"].split("/")[-1].split(".")[0] for item in candidates[1:]
    } == {"Copper", "Brass", "Bronze"}
    assert audit["strategy"].endswith("/v7")
    assert all("color" in item["retrieval_matched_fields"] for item in candidates)


def test_unresolved_orange_metal_shortlist_keeps_clean_intrinsic_identities() -> None:
    group = {
        "group_id": "G07",
        "family_hint": "metal",
        "base_color": "orange",
        "finish_hint": "other",
        "visual_description": "orange metal appearance region",
    }
    reliability = {
        "finish_hint": {"canonical_value": "painted", "reliable": False},
        "visual_description": {"reliable": False},
    }
    pool = [
        {
            "material_id": "mdl:painted-orange",
            "display_name": "Orange painted steel",
            "family": "paint",
            "colors": ["orange"],
            "finishes": ["painted"],
        },
        *[
            {
                "material_id": f"mdl:vMaterials_2/Metal/{name}.mdl#{name}",
                "display_name": name,
                "sub_identifier": name,
                "family": "metal",
                "colors": ["orange"],
                "finishes": ["new", "smooth"],
            }
            for name in ("Copper", "Brass", "Bronze")
        ],
        {
            "material_id": "mdl:vMaterials_2/Metal/Copper.mdl#Copper_Worn",
            "display_name": "Copper worn",
            "sub_identifier": "Copper_Worn",
            "family": "metal",
            "colors": ["orange"],
            "finishes": ["worn"],
        },
    ]

    candidates, audit = _shortlist_materials_with_audit(
        group,
        pool,
        limit=4,
        semantic_reliability=reliability,
        allow_parameter_writes=False,
    )

    assert {item["material_id"] for item in candidates} == {
        "mdl:painted-orange",
        "mdl:vMaterials_2/Metal/Copper.mdl#Copper",
        "mdl:vMaterials_2/Metal/Brass.mdl#Brass",
        "mdl:vMaterials_2/Metal/Bronze.mdl#Bronze",
    }
    policy = audit["surface_interpretation_policy"]
    assert policy["mode"] == "balanced_intrinsic_metal_identities"
    assert policy["required_intrinsic_material_identities"] == [
        "brass",
        "bronze",
        "copper",
    ]
    assert policy["complete_required_coverage"] is True


def test_confirmed_painted_metal_keeps_bare_counter_candidate() -> None:
    group = {
        "group_id": "G06",
        "family_hint": "metal",
        "base_color": "green",
        "finish_hint": "painted",
        "visual_description": "painted machine body",
    }
    pool = [
        {
            "material_id": "mdl:Metals/Copper.mdl#Copper",
            "family": "metal",
            "colors": ["orange"],
            "finishes": [],
        },
        {
            "material_id": "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte",
            "family": "paint",
            "colors": [],
            "finishes": ["matte"],
        },
        {
            "material_id": (
                "mdl:Miscellaneous/Paint_Matte_Finish.mdl#Paint_Matte_Finish"
            ),
            "family": "paint",
            "colors": [],
            "finishes": ["matte"],
        },
    ]

    candidates, audit = _shortlist_materials_with_audit(
        group,
        pool,
        mvinverse_pbr_evidence={
            "surface_class": "dielectric",
            "distinct_view_count": 2,
            "albedo": {
                "sample_count": 2,
                "median": [0.08, 0.24, 0.08],
                "mad": [0.02, 0.03, 0.02],
            },
            "metallic": {
                "sample_count": 2,
                "median": 0.08,
                "mad": 0.02,
                "iqr": 0.03,
            },
            "roughness": {
                "sample_count": 2,
                "median": 0.66,
                "mad": 0.02,
                "iqr": 0.04,
            },
        },
    )

    assert {item["material_id"] for item in candidates} == {
        "mdl:Metals/Copper.mdl#Copper",
        "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte",
    }
    assert candidates[0]["material_id"].endswith("Paint_Matte.mdl#Paint_Matte")
    assert audit["applied_coating_confirmed"] is True
    assert audit["paint_pool_used"] is False
    assert audit["duplicate_alias_dedup_count"] == 1
    assert audit["surface_interpretation_policy"]["active"] is False


def test_unreliable_painted_hint_and_dielectric_observation_keep_paint_candidate() -> (
    None
):
    group = {
        "group_id": "G07",
        "family_hint": "metal",
        "base_color": "orange",
        "finish_hint": "other",
        "visual_description": "orange metal appearance region",
    }
    reliability = {
        "finish_hint": {"canonical_value": "painted", "reliable": False},
        "visual_description": {"reliable": False},
    }
    pool = [
        {
            "material_id": "mdl:Metals/Copper.mdl#Copper",
            "family": "metal",
            "colors": ["orange"],
            "finishes": [],
        },
        {
            "material_id": "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte",
            "family": "paint",
            "colors": [],
            "finishes": ["matte"],
        },
    ]

    candidates, audit = _shortlist_materials_with_audit(
        group,
        pool,
        semantic_reliability=reliability,
        mvinverse_pbr_evidence={
            "surface_class": "dielectric",
            "distinct_view_count": 2,
            "albedo": {
                "sample_count": 2,
                "median": [0.8, 0.32, 0.08],
                "mad": [0.03, 0.03, 0.02],
            },
            "metallic": {
                "sample_count": 2,
                "median": 0.16,
                "mad": 0.02,
                "iqr": 0.04,
            },
            "roughness": {
                "sample_count": 2,
                "median": 0.62,
                "mad": 0.02,
                "iqr": 0.04,
            },
        },
    )

    assert {item["material_id"] for item in candidates} == {
        "mdl:Metals/Copper.mdl#Copper",
        "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte",
    }
    assert audit["applied_coating_confirmed"] is False
    assert audit["applied_coating_plausible"] is True


def _dark_surface_pool() -> list[dict[str, Any]]:
    return [
        {
            "material_id": "mdl:Metals/Steel_Blued.mdl#Steel_Blued",
            "display_name": "Omni PBR",
            "family": "metal",
            "colors": [],
            "finishes": [],
        },
        {
            "material_id": (
                "mdl:Metals/Aluminum_Anodized_Black.mdl#Aluminum_Anodized_Black"
            ),
            "display_name": "Aluminum Anodized Black",
            "family": "metal",
            "colors": ["black"],
            "finishes": ["anodized"],
        },
        {
            "material_id": "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte",
            "display_name": "Paint Matte",
            "family": "paint",
            "colors": [],
            "finishes": ["matte"],
        },
        {
            "material_id": "mdl:Miscellaneous/Paint_Satin.mdl#Paint_Satin",
            "display_name": "Paint Satin",
            "family": "paint",
            "colors": [],
            "finishes": ["satin"],
        },
        {
            "material_id": "mdl:Metals/Steel_Carbon.mdl#Steel_Carbon",
            "display_name": "Steel Carbon",
            "family": "metal",
            "colors": [],
            "finishes": [],
        },
        {
            "material_id": "mdl:Metals/Steel_Stainless.mdl#Steel_Stainless",
            "display_name": "Steel Stainless",
            "family": "metal",
            "colors": [],
            "finishes": [],
        },
    ]


def _dark_multiview_pbr(*, metallic: float, roughness: float) -> dict[str, Any]:
    return {
        "surface_class": "dielectric",
        "distinct_view_count": 3,
        "albedo": {
            "sample_count": 3,
            "median": [0.035, 0.049, 0.035],
            "mad": [0.02, 0.03, 0.02],
        },
        "metallic": {
            "sample_count": 3,
            "median": metallic,
            "mad": 0.03,
            "iqr": 0.05,
        },
        "roughness": {
            "sample_count": 3,
            "median": roughness,
            "mad": 0.03,
            "iqr": 0.05,
        },
    }


def test_dark_multiview_conflict_balances_treatment_paint_and_bare_metal() -> None:
    group = {
        "group_id": "G01",
        "family_hint": "metal",
        "base_color": "black",
        "finish_hint": "painted",
        "visual_description": "black metal appearance region",
    }
    reliability = {
        "finish_hint": {"canonical_value": "painted", "reliable": True},
        "visual_description": {"reliable": False},
    }

    candidates, audit = _shortlist_materials_with_audit(
        group,
        _dark_surface_pool(),
        family_reliable=True,
        semantic_reliability=reliability,
        mvinverse_pbr_evidence=_dark_multiview_pbr(
            metallic=0.68,
            roughness=0.36,
        ),
    )

    assert {item["material_id"] for item in candidates} == {
        "mdl:Metals/Aluminum_Anodized_Black.mdl#Aluminum_Anodized_Black",
        "mdl:Metals/Steel_Blued.mdl#Steel_Blued",
        "mdl:Metals/Steel_Carbon.mdl#Steel_Carbon",
        "mdl:Miscellaneous/Paint_Satin.mdl#Paint_Satin",
    }
    assert {item["surface_interpretation"] for item in candidates} == {
        "conversion_coating",
        "applied_paint",
        "bare_metal",
    }
    policy = audit["surface_interpretation_policy"]
    assert audit["strategy"].endswith("/v7")
    assert audit["paint_pool_used"] is False
    assert audit["applied_coating_confirmed"] is False
    assert audit["applied_coating_plausible"] is True
    assert policy["active"] is True
    assert policy["semantic_numeric_conflict"] is True
    assert policy["metallicity_class"] == "conductive"
    assert policy["roughness_class"] == "satin"
    assert policy["complete_required_coverage"] is True


def test_dark_dielectric_matte_semantics_keep_stainless_and_matte_competitive() -> None:
    group = {
        "group_id": "G02",
        "family_hint": "metal",
        "base_color": "black",
        "finish_hint": "painted",
        "visual_description": "black painted stainless steel enclosure",
    }

    candidates, audit = _shortlist_materials_with_audit(
        group,
        _dark_surface_pool(),
        family_reliable=True,
        mvinverse_pbr_evidence=_dark_multiview_pbr(
            metallic=0.12,
            roughness=0.68,
        ),
    )

    material_ids = {item["material_id"] for item in candidates}
    assert "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte" in material_ids
    assert "mdl:Metals/Steel_Stainless.mdl#Steel_Stainless" in material_ids
    assert any(
        item["surface_interpretation"] == "conversion_coating" for item in candidates
    )
    assert audit["applied_coating_confirmed"] is True
    assert audit["surface_interpretation_policy"]["roughness_class"] == "matte"


def test_unstable_single_view_dark_evidence_does_not_force_balanced_shortlist() -> None:
    group = {
        "group_id": "G03",
        "family_hint": "metal",
        "base_color": "black",
        "finish_hint": "painted",
        "visual_description": "black painted metal",
    }
    evidence = _dark_multiview_pbr(metallic=0.2, roughness=0.6)
    evidence["distinct_view_count"] = 1

    _candidates, audit = _shortlist_materials_with_audit(
        group,
        _dark_surface_pool(),
        family_reliable=True,
        mvinverse_pbr_evidence=evidence,
    )

    policy = audit["surface_interpretation_policy"]
    assert policy["active"] is False
    assert policy["multi_view_albedo_reliable"] is False
    assert policy["required_interpretations"] == []


def test_nvidia_base_duplicate_paint_aliases_confirm_one_canonical_choice() -> None:
    canonical = "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte"
    alias = "mdl:Miscellaneous/Paint_Matte_Finish.mdl#Paint_Matte_Finish"
    chosen, confirmed, basis = _confirm_material_choices(
        {"material_id": alias, "confidence": 0.8},
        {"material_id": canonical, "confidence": 0.9},
        [{"material_id": canonical}, {"material_id": alias}],
    )

    assert chosen["material_id"] == canonical
    assert confirmed is True
    assert basis == "nvidia_base_duplicate_paint_alias_agreement"


def test_mvinverse_roughness_resolves_base_paint_finish_disagreement() -> None:
    satin = "mdl:Miscellaneous/Paint_Satin.mdl#Paint_Satin"
    matte = "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte"
    chosen, confirmed, basis = _confirm_material_choices(
        {"material_id": satin, "confidence": 0.9},
        {"material_id": matte, "confidence": 0.9},
        [{"material_id": satin}, {"material_id": matte}],
        mvinverse_pbr_evidence={"roughness": {"median": 0.43}},
    )

    assert chosen["material_id"] == satin
    assert confirmed is True
    assert basis == "mvinverse_resolved_base_paint_finish"


def test_immutable_intrinsic_metal_disagreement_uses_stable_retrieval_score() -> None:
    copper = {
        "material_id": "mdl:Base/Metals/Copper.mdl#Copper",
        "family": "metal",
        "display_name": "Copper",
        "retrieval_score": 846.0,
    }
    brass = {
        "material_id": "mdl:vMaterials_2/Metal/Brass.mdl#Brass",
        "family": "metal",
        "display_name": "Brass",
        "retrieval_score": 824.0,
    }
    chosen, confirmed, basis = _confirm_material_choices(
        {"material_id": copper["material_id"], "confidence": 0.95},
        {"material_id": brass["material_id"], "confidence": 0.95},
        [copper, brass],
        allow_parameter_writes=False,
    )

    assert chosen["material_id"] == copper["material_id"]
    assert confirmed is True
    assert basis == "immutable_intrinsic_metal_class_agreement"


def test_visual_retrieval_does_not_semantically_confirm_metal_disagreement() -> None:
    copper = {
        "material_id": "mdl:Base/Metals/Copper.mdl#Copper",
        "family": "metal",
        "display_name": "Copper",
        "retrieval_score": 0.033,
        "retrieval_matched_fields": ["siglip2_catalog_wide_visual"],
    }
    brass = {
        "material_id": "mdl:vMaterials_2/Metal/Brass.mdl#Brass",
        "family": "metal",
        "display_name": "Brass",
        "retrieval_score": 0.031,
        "retrieval_matched_fields": ["siglip2_catalog_wide_visual"],
    }

    chosen, confirmed, basis = _confirm_material_choices(
        {"material_id": copper["material_id"], "confidence": 0.95},
        {"material_id": brass["material_id"], "confidence": 0.95},
        [copper, brass],
        allow_parameter_writes=False,
    )

    assert chosen["material_id"] == copper["material_id"]
    assert confirmed is False
    assert basis == "forward_reverse_disagreement"


def test_immutable_applied_paint_disagreement_uses_stable_retrieval_score() -> None:
    hammer = {
        "material_id": "mdl:vMaterials_2/Paint/Hammer_Paint.mdl#Hammer_Paint_Green",
        "family": "paint",
        "colors": ["green"],
        "retrieval_score": 922.0,
    }
    steel = {
        "material_id": (
            "mdl:vMaterials_2/Metal/Steel_Painted.mdl#" "Steel_Painted_Army_Green"
        ),
        "family": "metal",
        "colors": ["green"],
        "keywords": ["painted"],
        "retrieval_score": 458.0,
    }
    chosen, confirmed, basis = _confirm_material_choices(
        {"material_id": hammer["material_id"], "confidence": 0.95},
        {"material_id": steel["material_id"], "confidence": 0.99},
        [hammer, steel],
        mvinverse_pbr_evidence={"surface_class": "dielectric"},
        allow_parameter_writes=False,
    )

    assert chosen["material_id"] == hammer["material_id"]
    assert confirmed is True
    assert basis == "immutable_applied_paint_appearance_agreement"


def test_mvinverse_resolves_tunable_export_variants_to_retrieval_leader() -> None:
    army = "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted_Army_Green"
    arcadia = "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted_Arcadia_Green"
    chosen, confirmed, basis = _confirm_material_choices(
        {"material_id": arcadia, "confidence": 0.9},
        {"material_id": army, "confidence": 0.9},
        [{"material_id": army}, {"material_id": arcadia}],
        mvinverse_pbr_evidence={
            "surface_class": "dielectric",
            "suggestion": {
                "decision": "auto",
                "auto_parameter_eligible": True,
            },
        },
    )

    assert chosen["material_id"] == army
    assert confirmed is True
    assert basis == "mvinverse_tunable_module_agreement"


def test_immutable_mdl_exports_are_not_confirmed_as_tunable_equivalents() -> None:
    army = "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted_Army_Green"
    arcadia = "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted_Arcadia_Green"
    chosen, confirmed, basis = _confirm_material_choices(
        {"material_id": arcadia, "confidence": 0.9},
        {"material_id": army, "confidence": 0.9},
        [{"material_id": army}, {"material_id": arcadia}],
        mvinverse_pbr_evidence={
            "surface_class": "dielectric",
            "suggestion": {
                "decision": "auto",
                "auto_parameter_eligible": True,
            },
        },
        allow_parameter_writes=False,
    )

    assert chosen["material_id"] == arcadia
    assert confirmed is False
    assert basis == "forward_reverse_disagreement"


def test_disagreement_forces_both_choices_and_mvinverse_defaults_into_tournament() -> None:
    forward_id = "mdl:Metal.mdl#Grass_Green"
    reverse_id = "mdl:Metal.mdl#Army_Green"
    numeric_id = "mdl:Plastic.mdl#Polypropylene_Green"
    far_id = "mdl:Plastic.mdl#White"

    def candidate(material_id: str, color: list[float]) -> dict[str, Any]:
        return {
            "material_id": material_id,
            "family": "metal" if "Metal" in material_id else "plastic",
            "appearance_profile": {
                "base_color_srgb": color,
                "roughness": 0.43,
                "metallic": 0.0,
            },
        }

    forward = candidate(forward_id, [0.22, 0.65, 0.22])
    reverse = candidate(reverse_id, [0.25, 0.40, 0.18])
    numeric = candidate(numeric_id, [0.23, 0.54, 0.20])
    far = candidate(far_id, [0.95, 0.95, 0.95])
    evidence = {
        "albedo": {"median": [0.23, 0.54, 0.20]},
        "roughness": {"median": 0.43},
        "metallic": {"median": 0.0},
    }

    neighbors = _mvinverse_exact_default_candidates(
        [forward, reverse, numeric, far],
        evidence,
        limit=1,
    )
    primary, wider, contract = _build_disagreement_tournament_candidates(
        forward={"material_id": forward_id, "confidence": 0.94},
        reverse={"material_id": reverse_id, "confidence": 0.93},
        provisional_seed={"material_id": forward_id, "confidence": 0.94},
        qwen_candidates=[forward, reverse],
        tournament_candidates=[forward, reverse, far],
        pool=[forward, reverse, numeric, far],
        mvinverse_pbr_evidence=evidence,
        maximum_candidates=4,
    )

    assert neighbors[0]["material_id"] == numeric_id
    assert {item["material_id"] for item in primary} >= {
        forward_id,
        reverse_id,
        numeric_id,
    }
    assert set(contract["required_candidate_material_ids"]) <= {
        item["material_id"] for item in wider
    }
    assert contract["provisional_seed_is_final_selection"] is False
    assert contract["selected_mdl_parameters_mutable"] is False


def test_disagreement_uses_ranked_exact_default_when_mvinverse_is_unusable() -> None:
    forward_id = "mdl:Base/Wall_Board/Gypsum.mdl#Gypsum"
    reverse_id = "mdl:Base/Wall_Board/Plaster.mdl#Plaster"
    retrieval_id = "mdl:Base/Masonry/Concrete_Smooth.mdl#Concrete_Smooth"
    other_id = "mdl:Base/Masonry/Stucco.mdl#Stucco"

    def candidate(material_id: str) -> dict[str, Any]:
        return {
            "material_id": material_id,
            "family": "plastic",
        }

    forward = candidate(forward_id)
    reverse = candidate(reverse_id)
    retrieval = candidate(retrieval_id)
    other = candidate(other_id)
    primary, wider, contract = _build_disagreement_tournament_candidates(
        forward={"material_id": forward_id, "confidence": 0.0},
        reverse={"material_id": reverse_id, "confidence": 0.0},
        provisional_seed={"material_id": forward_id, "confidence": 0.0},
        qwen_candidates=[forward, reverse],
        tournament_candidates=[reverse, retrieval, forward, other],
        pool=[forward, reverse, retrieval, other],
        mvinverse_pbr_evidence={
            "surface_class": "dielectric",
            "distinct_view_count": 0,
            "albedo": None,
            "metallic": None,
            "roughness": None,
            "suggestion": {
                "decision": "preserve",
                "auto_parameter_eligible": False,
                "reason_codes": ["insufficient_distinct_views"],
            },
        },
        maximum_candidates=4,
        selection_group={
            "family_hint": "plastic",
            "base_color": "white",
            "finish_hint": "matte",
        },
    )

    assert [item["material_id"] for item in primary[:3]] == [
        forward_id,
        reverse_id,
        retrieval_id,
    ]
    fallback = next(item for item in wider if item["material_id"] == retrieval_id)
    assert fallback["tournament_candidate_basis"] == (
        "ranked_retrieval_independent_exact_library_default_fallback"
    )
    assert contract["mvinverse_exact_default_material_ids"] == []
    assert contract["retrieval_exact_default_material_ids"] == [retrieval_id]
    assert contract["mvinverse_challenger_status"] == (
        "unavailable_or_no_eligible_exact_default_candidate"
    )
    assert contract["required_candidate_material_ids"] == [
        forward_id,
        reverse_id,
        retrieval_id,
    ]


def test_disagreement_without_any_independent_exact_default_fails_closed() -> None:
    forward = {"material_id": "mdl:Only.mdl#Forward", "family": "plastic"}
    reverse = {"material_id": "mdl:Only.mdl#Reverse", "family": "plastic"}

    with pytest.raises(
        ValueError,
        match="neither an accepted MVInverse-nearest.*nor an independent",
    ):
        _build_disagreement_tournament_candidates(
            forward={"material_id": forward["material_id"], "confidence": 0.0},
            reverse={"material_id": reverse["material_id"], "confidence": 0.0},
            provisional_seed={
                "material_id": forward["material_id"],
                "confidence": 0.0,
            },
            qwen_candidates=[forward, reverse],
            tournament_candidates=[forward, reverse],
            pool=[forward, reverse],
            mvinverse_pbr_evidence=None,
            maximum_candidates=3,
        )


@pytest.mark.parametrize(
    ("color", "intrinsic_names"),
    [
        ("orange", ("Copper", "Brass", "Bronze")),
        ("brown", ("Copper", "Brass", "Bronze")),
        ("yellow", ("Brass", "Bronze", "Gold")),
    ],
)
def test_low_confidence_finish_keeps_coating_and_intrinsic_metals_ambiguous(
    color: str,
    intrinsic_names: tuple[str, ...],
) -> None:
    group = {
        "group_id": "G01",
        "family_hint": "metal",
        "base_color": color,
        "finish_hint": "painted",
        "visual_description": f"{color} painted metal component",
        "confidence": 0.60,
    }
    original_group = json.loads(json.dumps(group))
    fusion_group = {
        "sources": [
            {
                "view_id": "front",
                "finish_hint": "painted",
                "confidence": 0.60,
            },
            *[
                {
                    "view_id": view_id,
                    "finish_hint": "other",
                    "confidence": 0.60,
                }
                for view_id in ("side", "top", "iso")
            ],
        ]
    }
    pool = [
        {
            "material_id": f"mdl:painted-{color}",
            "display_name": f"{color} painted steel",
            "family": "metal",
            "colors": [color],
            "finishes": ["painted"],
        },
        *[
            {
                "material_id": f"mdl:Base/Metals/{name}.mdl#{name}",
                "display_name": name,
                "sub_identifier": name,
                "family": "metal",
                "colors": [],
                "finishes": ["bare"],
            }
            for name in intrinsic_names
        ],
    ]

    selection_group, reliability = _material_selection_context(
        group,
        fusion_group,
    )
    candidates, retrieval = _shortlist_materials_with_audit(
        selection_group,
        pool,
        semantic_reliability=reliability,
    )

    assert group == original_group
    assert selection_group["finish_hint"] == "other"
    assert "painted" not in selection_group["visual_description"]
    assert reliability["canonical_group_preserved"] is True
    assert reliability["selection_context_modified"] is True
    assert reliability["finish_hint"]["reliable"] is False
    assert reliability["finish_hint"]["supporting_view_ids"] == ["front"]
    assert "single_review_confidence_finish_source" in reliability["reason_codes"]
    assert retrieval["finish_evidence_used"] is False
    assert retrieval["description_evidence_used"] is False
    assert retrieval["intrinsic_surface_ambiguity"] is True
    assert retrieval["normalized_margin"] == 0.0
    assert {item["material_id"] for item in candidates} == {
        f"mdl:painted-{color}",
        *{f"mdl:Base/Metals/{name}.mdl#{name}" for name in intrinsic_names},
    }
    assert {item["retrieval_score"] for item in candidates} == {210}


def test_high_confidence_finish_and_description_remain_selector_evidence() -> None:
    group = {
        "group_id": "G01",
        "family_hint": "metal",
        "base_color": "orange",
        "finish_hint": "painted",
        "visual_description": "orange painted guard",
        "confidence": 0.95,
    }
    fusion_group = {
        "sources": [
            {
                "view_id": "front",
                "finish_hint": "painted",
                "confidence": 0.95,
            },
            {
                "view_id": "side",
                "finish_hint": "other",
                "confidence": 0.60,
            },
        ]
    }

    selection_group, reliability = _material_selection_context(
        group,
        fusion_group,
    )

    assert selection_group == group
    assert reliability["selection_context_modified"] is False
    assert reliability["finish_hint"]["reliable"] is True
    assert reliability["finish_hint"]["high_confidence_confirmed"] is True
    assert reliability["visual_description"]["reliable"] is True


def test_two_review_confidence_views_confirm_finish_but_not_description() -> None:
    group = {
        "group_id": "G01",
        "family_hint": "metal",
        "base_color": "orange",
        "finish_hint": "painted",
        "visual_description": "orange painted tube",
        "confidence": 0.60,
    }
    fusion_group = {
        "sources": [
            {
                "view_id": view_id,
                "finish_hint": "painted",
                "confidence": 0.60,
            }
            for view_id in ("front", "side")
        ]
    }

    selection_group, reliability = _material_selection_context(
        group,
        fusion_group,
    )

    assert selection_group["finish_hint"] == "painted"
    assert selection_group["visual_description"] != group["visual_description"]
    assert reliability["finish_hint"]["reliable"] is True
    assert reliability["finish_hint"]["multiview_confirmed"] is True
    assert reliability["visual_description"]["reliable"] is False
    assert "finish_confirmed_by_independent_views" in reliability["reason_codes"]


def test_equivalent_palette_groups_merge_before_quality_scoring() -> None:
    palette = {
        "schema_version": PALETTE_SCHEMA_VERSION,
        "source_view_id": "ref_side",
        "groups": [
            {
                "group_id": "G01",
                "family_hint": "metal",
                "base_color": "white",
                "finish_hint": "painted",
                "visual_description": "lower-confidence white panel",
                "boxes": [[10, 10, 110, 110], [20, 20, 120, 120]],
                "confidence": 0.61,
            },
            {
                "group_id": "G02",
                "family_hint": "metal",
                "base_color": "white",
                "finish_hint": "painted",
                "visual_description": "best supported white painted housing",
                "boxes": [
                    [30, 30, 130, 130],
                    [10, 10, 110, 110],
                    [40, 40, 140, 140],
                    [50, 50, 150, 150],
                ],
                "confidence": 0.89,
            },
            {
                "group_id": "G03",
                "family_hint": "plastic",
                "base_color": "black",
                "finish_hint": "matte",
                "visual_description": "black hose",
                "boxes": [[600, 100, 700, 300]],
                "confidence": 0.78,
            },
        ],
    }
    audit = {
        "image": "/tmp/ref_side.png",
        "accepted_group_ids": ["G01", "G02", "G03"],
        "rejected_group_ids": [],
        "groups": [
            {
                "group_id": group["group_id"],
                "base_color": group["base_color"],
                "accepted": True,
                "boxes": [
                    {
                        "box_index": index,
                        "box": box,
                        "accepted": True,
                        "sampled_pixels": 1000,
                        "matching_pixel_count": 700,
                        "foreground_pixels": 800,
                    }
                    for index, box in enumerate(group["boxes"])
                ],
            }
            for group in palette["groups"]
        ],
    }

    normalized, normalized_audit, merge_audit = _merge_equivalent_palette_groups(
        palette, audit
    )

    assert len(normalized["groups"]) == 2
    white = normalized["groups"][0]
    assert white["group_id"] == "G02"
    assert white["confidence"] == 0.89
    assert white["visual_description"] == "best supported white painted housing"
    assert white["boxes"] == [
        [30, 30, 130, 130],
        [10, 10, 110, 110],
        [40, 40, 140, 140],
        [50, 50, 150, 150],
    ]
    assert normalized_audit["groups"][0]["source_group_ids"] == ["G01", "G02"]
    assert normalized_audit["groups"][0]["boxes"][0]["source_group_id"] == "G02"
    assert merge_audit["input_group_count"] == 3
    assert merge_audit["unique_signature_count"] == 2
    assert merge_audit["merged_group_count"] == 1
    assert merge_audit["clusters"][0]["member_group_ids"] == ["G01", "G02"]
    assert merge_audit["clusters"][0]["representative_group_id"] == "G02"
    assert merge_audit["clusters"][0]["duplicate_box_count"] == 1
    assert merge_audit["clusters"][0]["truncated_box_count"] == 1
    quality = _palette_quality_metrics(normalized, normalized_audit)
    assert quality["group_count"] == 2
    assert quality["color_diversity_count"] == 2


def test_best_evidence_view_uses_max_pixels_and_preferred_view_only_for_ties() -> None:
    render_set = {
        "views": [
            {"view_id": "front"},
            {"view_id": "rear"},
            {"view_id": "top"},
        ]
    }
    parts = [
        {
            "part_id": "P0001",
            "renders": [
                {"view_id": "front", "visible_pixels": 120},
                {"view_id": "rear", "visible_pixels": 480},
                {"view_id": "top", "visible_pixels": 200},
            ],
        },
        {
            "part_id": "P0002",
            "renders": [
                {"view_id": "front", "visible_pixels": 320},
                {"view_id": "rear", "visible_pixels": 320},
            ],
        },
        {
            "part_id": "P0003",
            "renders": [{"view_id": "unregistered", "visible_pixels": 999}],
        },
    ]

    evidence = _assign_best_evidence_views(
        parts,
        render_set=render_set,
        preferred_view_id="rear",
    )

    assert evidence == {
        "P0001": {
            "cad_view_id": "rear",
            "visible_pixels": 480,
            "source_visible_pixels": 480,
            "effective_visible_pixels": 480,
            "evidence_mode": "source_projection",
            "source_evidence_view_count": 3,
            "isolated_evidence_sha256": None,
        },
        "P0002": {
            "cad_view_id": "rear",
            "visible_pixels": 320,
            "source_visible_pixels": 320,
            "effective_visible_pixels": 320,
            "evidence_mode": "source_projection",
            "source_evidence_view_count": 2,
            "isolated_evidence_sha256": None,
        },
        "P0003": {
            "cad_view_id": None,
            "visible_pixels": 0,
            "source_visible_pixels": 0,
            "effective_visible_pixels": 0,
            "evidence_mode": "source_projection",
            "source_evidence_view_count": 0,
            "isolated_evidence_sha256": None,
        },
    }
    assert parts[0]["evidence_cad_view_id"] == "rear"
    assert parts[0]["evidence_visible_pixels"] == 480


def test_isolated_evidence_improves_shape_readability_without_inflating_source() -> (
    None
):
    render_set = {"views": [{"view_id": "front"}, {"view_id": "rear"}]}
    parts = [
        {
            "part_id": "P0001",
            "renders": [
                {"view_id": "front", "visible_pixels": 30},
                {"view_id": "rear", "visible_pixels": 24},
            ],
            "isolated_evidence": {
                "schema_version": "qwen-isolated-part-evidence/v1",
                "path": "/tmp/P0001.png",
                "sha256": "a" * 64,
                "selected_view_ids": ["front", "rear"],
                "source_visible_pixels_by_view": {"front": 30, "rear": 24},
                "normalized_visible_pixels_by_view": {
                    "front": 5016,
                    "rear": 4074,
                },
                "source_max_visible_pixels": 30,
                "normalized_max_visible_pixels": 5016,
                "source_evidence_view_count": 2,
                "source_evidence_view_ids": ["front", "rear"],
                "source_pixel_floor": 12,
                "material_neutralized": True,
                "background_removed": True,
            },
        }
    ]

    evidence = _assign_best_evidence_views(
        parts,
        render_set=render_set,
        preferred_view_id=None,
    )

    assert evidence["P0001"]["source_visible_pixels"] == 30
    assert evidence["P0001"]["effective_visible_pixels"] == 5016
    assert evidence["P0001"]["evidence_mode"] == "isolated_mask_multiview"
    assert parts[0]["evidence_source_visible_pixels"] == 30
    assert parts[0]["evidence_visible_pixels"] == 5016


def test_continuous_calibration_uses_current_visibility_and_old_isolated_geometry() -> (
    None
):
    render_set = {
        "continuous_camera_calibration": True,
        "views": [
            {
                "view_id": "photo_front",
                "visible_parts": [{"part_id": "P0001", "pixels": 11}],
            },
            {
                "view_id": "photo_iso",
                "visible_parts": [{"part_id": "P0001", "pixels": 42}],
            },
        ],
    }
    parts = [
        {
            "part_id": "P0001",
            # This is deliberately a source-pose view that no longer exists
            # in the calibrated render set.
            "renders": [{"view_id": "rear", "visible_pixels": 30}],
            "isolated_evidence": {
                "schema_version": "qwen-isolated-part-evidence/v1",
                "path": "/tmp/P0001.png",
                "sha256": "b" * 64,
                "selected_view_ids": ["rear"],
                "source_visible_pixels_by_view": {"rear": 30},
                "normalized_visible_pixels_by_view": {"rear": 5016},
                "source_max_visible_pixels": 30,
                "normalized_max_visible_pixels": 5016,
                "source_evidence_view_count": 1,
                "source_evidence_view_ids": ["rear"],
                "source_pixel_floor": 12,
                "material_neutralized": True,
                "background_removed": True,
            },
        },
        {
            "part_id": "P0002",
            "renders": [{"view_id": "top", "visible_pixels": 15}],
        },
    ]

    evidence = _assign_best_evidence_views(
        parts,
        render_set=render_set,
        preferred_view_id="photo_front",
    )

    assert evidence["P0001"] == {
        "cad_view_id": "photo_iso",
        "visible_pixels": 5016,
        "source_visible_pixels": 42,
        "effective_visible_pixels": 5016,
        "evidence_mode": "isolated_mask_multiview",
        "source_evidence_view_count": 0,
        "isolated_evidence_sha256": "b" * 64,
    }
    # A fully hidden part still receives a valid calibrated overview so batch
    # construction remains total; its visibility remains explicitly zero.
    assert evidence["P0002"]["cad_view_id"] == "photo_front"
    assert evidence["P0002"]["source_visible_pixels"] == 0
    assert evidence["P0002"]["visible_pixels"] == 0


def test_continuous_calibration_generates_missing_part_highlight(
    tmp_path: Path,
) -> None:
    from qwen_material_pipeline.usd.render import _part_color

    rgb = Image.new("RGB", (32, 24), (170, 170, 170))
    rgb_path = tmp_path / "rgb.png"
    rgb.save(rgb_path)
    raw_ids = Image.new("RGB", (32, 24), (28, 28, 28))
    target_color = _part_color("P0139")
    for x in range(8, 20):
        for y in range(6, 16):
            raw_ids.putpixel((x, y), target_color)
    ids_path = tmp_path / "ids.png"
    raw_ids.save(ids_path)
    render_set = {
        "continuous_camera_calibration": True,
        "views": [
            {
                "view_id": "iso",
                "rgb": str(rgb_path),
                "part_ids_raw": str(ids_path),
                "visible_parts": [{"part_id": "P0139", "pixels": 120}],
            }
        ],
    }
    parts = [{"part_id": "P0139", "evidence_cad_view_id": "iso", "renders": []}]

    generated = run_staged_local._materialize_calibrated_fallback_highlights(
        parts,
        render_set=render_set,
        best_evidence={},
        output_dir=tmp_path / "highlights",
    )

    assert set(generated) == {"P0139"}
    with Image.open(generated["P0139"]) as highlight:
        assert highlight.width > 0
        assert highlight.height > 0


def test_part_batches_never_mix_best_evidence_views() -> None:
    render_set = {
        "views": [{"view_id": "front"}, {"view_id": "rear"}],
        "analysis_basis_world": {
            "up": [0.0, 0.0, 1.0],
            "right": [1.0, 0.0, 0.0],
        },
    }
    parts = [
        {
            "part_id": "P0001",
            "evidence_cad_view_id": "rear",
            "world_bbox": [[0, 0, 0], [1, 1, 1]],
        },
        {
            "part_id": "P0002",
            "evidence_cad_view_id": "front",
            "world_bbox": [[0, 0, 1], [1, 1, 2]],
        },
        {
            "part_id": "P0003",
            "evidence_cad_view_id": "front",
            "world_bbox": [[0, 0, 0], [1, 1, 1]],
        },
    ]

    batches = _view_grouped_part_batches(parts, render_set=render_set, batch_size=4)

    assert [view_id for view_id, _batch in batches] == ["front", "rear"]
    assert [part["part_id"] for part in batches[0][1]] == ["P0002", "P0003"]
    assert [part["part_id"] for part in batches[1][1]] == ["P0001"]


def test_main_uses_each_batch_best_view_overviews_and_effective_threshold(
    tmp_path: Path, monkeypatch: Any
) -> None:
    def image(name: str, color: str) -> Path:
        path = tmp_path / name
        Image.new("RGB", (48, 48), color).save(path)
        return path

    reference = image("reference.png", "white")
    reference_alt = image("reference_alt.png", "green")
    reference_alt_mask = image("reference_alt_mask.png", "white")
    front_rgb = image("front_rgb.png", "gray")
    front_ids = image("front_ids.png", "red")
    rear_rgb = image("rear_rgb.png", "silver")
    rear_ids = image("rear_ids.png", "blue")
    front_highlight = image("P0001_front.png", "orange")
    rear_highlight = image("P0002_rear.png", "purple")
    small_highlight = image("P0003_front.png", "yellow")
    registry = {
        "parts": [
            {
                "part_id": "P0001",
                "world_bbox": [[0, 0, 0], [1, 1, 1]],
                "renders": [
                    {
                        "view_id": "front",
                        "visible_pixels": 500,
                        "highlight_path": str(front_highlight),
                    },
                    {"view_id": "rear", "visible_pixels": 100},
                ],
            },
            {
                "part_id": "P0002",
                "world_bbox": [[1, 0, 0], [2, 1, 1]],
                "renders": [
                    {"view_id": "front", "visible_pixels": 80},
                    {
                        "view_id": "rear",
                        "visible_pixels": 600,
                        "highlight_path": str(rear_highlight),
                    },
                ],
            },
            {
                "part_id": "P0003",
                "world_bbox": [[2, 0, 0], [3, 1, 1]],
                "renders": [
                    {
                        "view_id": "front",
                        "visible_pixels": 200,
                        "highlight_path": str(small_highlight),
                    }
                ],
            },
            {
                "part_id": "P0004",
                "world_bbox": [[3, 0, 0], [4, 1, 1]],
                "renders": [{"view_id": "front", "visible_pixels": 40}],
            },
        ],
        "render_set": {
            "views": [
                {
                    "view_id": "front",
                    "rgb": str(front_rgb),
                    "part_ids": str(front_ids),
                },
                {
                    "view_id": "rear",
                    "rgb": str(rear_rgb),
                    "part_ids": str(rear_ids),
                },
            ],
            "best_highlights": {
                "P0001": str(front_highlight),
                "P0002": str(rear_highlight),
                "P0003": str(small_highlight),
            },
        },
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    calls: list[dict[str, Any]] = []
    extracted: list[str] = []
    filter_calls: list[tuple[str, Path | None]] = []

    class FakeClient:
        def __init__(self, **_kwargs: Any) -> None:
            self.repair_events: list[dict[str, Any]] = []

        def extract_palette(
            self,
            reference_view: dict[str, str],
            *,
            run_label: str | None = None,
        ) -> dict[str, Any]:
            extracted.append(reference_view["id"])
            assert run_label is not None
            groups = [
                {
                    "group_id": "G01",
                    "family_hint": "metal",
                    "base_color": "white",
                    "finish_hint": "painted",
                    "visual_description": "white painted panel",
                    "boxes": [[10, 10, 100, 100]],
                    "confidence": 0.8,
                }
            ]
            if reference_view["id"] == "ref_alt":
                groups.append(
                    {
                        "group_id": "G02",
                        "family_hint": "plastic",
                        "base_color": "black",
                        "finish_hint": "matte",
                        "visual_description": "black polymer handle",
                        "boxes": [[200, 200, 300, 300]],
                        "confidence": 0.7,
                    }
                )
            return {
                "schema_version": PALETTE_SCHEMA_VERSION,
                "source_view_id": reference_view["id"],
                "groups": groups,
            }

        def map_part_batch(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {"batch_id": kwargs["batch_id"], "mappings": []}

    monkeypatch.setattr(
        run_staged_local,
        "TransformersQwen3VLRunner",
        lambda *_args, **_kwargs: SimpleNamespace(preflight=lambda: None),
    )
    monkeypatch.setattr(run_staged_local, "LocalStagedQwenClient", FakeClient)

    def fake_filter(
        document: dict[str, Any],
        _path: Path,
        *,
        mask_path: Path | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        filter_calls.append((document["source_view_id"], mask_path))
        return document, {"groups": []}

    monkeypatch.setattr(
        run_staged_local,
        "filter_palette_by_image_evidence",
        fake_filter,
    )

    output = tmp_path / "output"
    exit_code = run_staged_local.main(
        [
            "--registry",
            str(registry_path),
            "--reference",
            f"ref={reference}",
            "--reference",
            f"ref_alt={reference_alt}",
            "--palette-mask",
            f"ref_alt={reference_alt_mask}",
            "--catalog",
            str(tmp_path / "unused_catalog.json"),
            "--model-path",
            str(tmp_path / "model"),
            "--output-dir",
            str(output),
            "--cad-view",
            "front",
            "--min-visible-pixels",
            "64",
            "--stop-after",
            "mapping",
        ]
    )

    assert exit_code == 0
    assert extracted == ["ref", "ref_alt"]
    assert filter_calls == [("ref", None), ("ref_alt", reference_alt_mask)]
    assert len(calls) == 2
    assert {part["part_id"] for call in calls for part in call["target_parts"]} == {
        "P0001",
        "P0002",
        "P0003",
    }
    expected_geometry = {
        "front": (str(front_rgb), str(front_ids)),
        "rear": (str(rear_rgb), str(rear_ids)),
    }
    for call in calls:
        assert call["reference_view"]["id"] == "ref_alt"
        assert call["support_reference_view"] == {
            "id": "ref",
            "image": str(reference),
        }
        target_views = {part["evidence_cad_view_id"] for part in call["target_parts"]}
        assert len(target_views) == 1
        view_id = target_views.pop()
        assert call["cad_view"] == {
            "id": f"cad_{view_id}",
            "image": expected_geometry[view_id][0],
        }
        assert call["part_id_view"] == {
            "id": f"part_ids_{view_id}",
            "image": expected_geometry[view_id][1],
        }

    plan = json.loads((output / "batch_plan.json").read_text(encoding="utf-8"))
    assert plan["requested_min_visible_pixels"] == 64
    assert plan["min_visible_pixels"] == DEFAULT_MIN_VISIBLE_PIXELS
    assert plan["minimum_matched_visible_pixels"] == MIN_MATCH_VISIBLE_PIXELS
    assert plan["forced_unknown_parts"] == {"P0004": "too_small"}
    assert plan["part_evidence"]["P0001"]["cad_view_id"] == "front"
    assert plan["part_evidence"]["P0002"]["cad_view_id"] == "rear"
    assert plan["part_evidence"]["P0003"]["mapping_eligible"] is True
    assert plan["part_evidence"]["P0003"]["automatic_match_eligible"] is False
    assert plan["part_evidence"]["P0004"]["mapping_eligible"] is False
    assert set(plan["batch_cad_views"].values()) == {"front", "rear"}
    assert plan["reference_view_id"] == "ref_alt"
    manifest = json.loads(
        (output / "reference_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model_reference_view"] == {
        "id": "ref_alt",
        "image": str(reference_alt),
        "is_contact_sheet": False,
    }
    assert manifest["mapping_support_view"]["source_view_ids"] == ["ref"]
    assert manifest["source_views"][1]["palette_mask"] == str(reference_alt_mask)
    selection = json.loads(
        (output / "palette_selection.json").read_text(encoding="utf-8")
    )
    assert selection["selected_reference_id"] == "ref_alt"
    assert (output / "palette_views" / "01_ref" / "palette.model.json").is_file()
    assert (output / "palette_views" / "02_ref_alt" / "palette.json").is_file()
