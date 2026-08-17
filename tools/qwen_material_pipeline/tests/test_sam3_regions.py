from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from qwen_material_pipeline.segmentation.sam3_regions import (
    ORDERED_POINT_SCHEMA_VERSION,
    POINT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    Sam3RegionError,
    _arbitrate_view_group_masks,
    _bounded_shared_camera_alignment,
    _box_pixels,
    _candidate_metrics,
    _dense_cad_mask_logits,
    _estimate_view_shared_translation,
    _evaluate_shape_guided_candidate,
    _load_request,
    _normalized_cxcywh,
    _occlusion_aware_amodal_agreement,
    _segment_box,
    _segment_dense_mask_points,
    _segment_ordered_points,
    _segment_points,
    _segment_shape_guided_points,
    _shape_seed_click_set,
    _validated_box,
    result_policy,
)
from qwen_material_pipeline.segmentation.human_foreground import sha256_file


def test_standalone_cli_imports_shared_replay_code_without_inherited_pythonpath(
    tmp_path: Path,
) -> None:
    script = Path(__file__).resolve().parents[1] / "segmentation" / "sam3_regions.py"
    environment = os.environ.copy()
    for name in (
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_PROMPT_MODIFIER",
        "VIRTUAL_ENV",
        "PYTHONHOME",
        "PYTHONPATH",
    ):
        environment.pop(name, None)

    completed = subprocess.run(
        [sys.executable, "-I", str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--request" in completed.stdout


def _write_image(path: Path, *, size: tuple[int, int] = (20, 10)) -> Path:
    Image.new("RGB", size, (40, 80, 120)).save(path)
    return path


def _write_request(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "sam3_request.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _valid_request() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_views": [{"id": "front", "image": "front.png"}],
        "regions": [
            {
                "view_id": "front",
                "group_id": "G01",
                "boxes": [[100, 200, 900, 800]],
                "prompt": "  painted steel  ",
            }
        ],
    }


def test_load_request_resolves_images_and_normalizes_prompt(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "front.png")
    request_path = _write_request(tmp_path, _valid_request())

    request, source_paths = _load_request(request_path)

    assert source_paths == {"front": image.resolve()}
    assert request["regions"][0]["boxes"] == [[100, 200, 900, 800]]
    assert request["regions"][0]["prompt"] == "painted steel"


def test_load_request_accepts_human_point_sets_and_confirmed_mask(
    tmp_path: Path,
) -> None:
    _write_image(tmp_path / "front.png")
    mask = Image.new("L", (20, 10), 0)
    mask.putpixel((10, 5), 255)
    mask.save(tmp_path / "front-mask.png")
    request_path = _write_request(
        tmp_path,
        {
            "schema_version": POINT_SCHEMA_VERSION,
            "source_views": [{"id": "front", "image": "front.png"}],
            "regions": [
                {
                    "view_id": "front",
                    "group_id": "__foreground__",
                    "click_sets": [
                        {
                            "positive_points": [[500, 500]],
                            "negative_points": [[0, 0]],
                        }
                    ],
                    "confirmed_mask": {
                        "path": "front-mask.png",
                        "sha256": sha256_file(tmp_path / "front-mask.png"),
                    },
                }
            ],
        },
    )

    request, _source_paths = _load_request(request_path)

    region = request["regions"][0]
    assert region["boxes"] == []
    assert region["prompt"] == "visual"
    assert region["click_sets"][0]["positive_points"] == [[500, 500]]
    assert (
        Path(region["confirmed_mask"]["path"])
        == (tmp_path / "front-mask.png").resolve()
    )


def test_load_request_accepts_ordered_incremental_point_sets(tmp_path: Path) -> None:
    _write_image(tmp_path / "front.png")
    Image.new("L", (20, 10), 255).save(tmp_path / "front-mask.png")
    request_path = _write_request(
        tmp_path,
        {
            "schema_version": ORDERED_POINT_SCHEMA_VERSION,
            "prompt_authority": "human_confirmed_sam3_interactive_points",
            "human_annotation": {
                "schema_version": "sam3-human-foreground-annotations/v2",
                "document_sha256": "a" * 64,
                "all_views_confirmed": True,
                "human_mask_is_authoritative": True,
                "formal_rerun_minimum_iou": 0.995,
            },
            "source_views": [{"id": "front", "image": "front.png"}],
            "regions": [
                {
                    "view_id": "front",
                    "group_id": "__foreground__",
                    "click_sets": [
                        {
                            "events": [
                                {"point": [500, 500], "label": 1},
                                {"point": [0, 0], "label": 0},
                            ],
                            "positive_points": [[500, 500]],
                            "negative_points": [[0, 0]],
                            "initial_candidate_index": 2,
                        }
                    ],
                    "confirmed_mask": {
                        "path": "front-mask.png",
                        "sha256": sha256_file(tmp_path / "front-mask.png"),
                    },
                }
            ],
        },
    )

    request, _source_paths = _load_request(request_path)

    click_set = request["regions"][0]["click_sets"][0]
    assert [event["label"] for event in click_set["events"]] == [1, 0]
    assert click_set["initial_candidate_index"] == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda request: request.update(prompt_authority="automatic"),
            "invalid prompt authority",
        ),
        (
            lambda request: request["human_annotation"].update(
                schema_version="sam3-human-foreground-annotations/v1"
            ),
            "invalid human annotation record",
        ),
        (
            lambda request: request["regions"][0].update(
                click_sets=None,
                confirmed_mask=None,
                boxes=[[0, 0, 1000, 1000]],
                prompt="visual",
            ),
            "must all use foreground click sets",
        ),
        (
            lambda request: request["source_views"].append(
                {"id": "side", "image": "side.png"}
            ),
            "exactly one region per source view",
        ),
    ],
)
def test_ordered_request_fails_closed_on_incomplete_human_contract(
    tmp_path: Path, mutation, message: str
) -> None:
    _write_image(tmp_path / "front.png")
    _write_image(tmp_path / "side.png")
    Image.new("L", (20, 10), 255).save(tmp_path / "mask.png")
    request = {
        "schema_version": ORDERED_POINT_SCHEMA_VERSION,
        "prompt_authority": "human_confirmed_sam3_interactive_points",
        "human_annotation": {
            "schema_version": "sam3-human-foreground-annotations/v2",
            "document_sha256": "a" * 64,
            "all_views_confirmed": True,
            "human_mask_is_authoritative": True,
            "formal_rerun_minimum_iou": 0.995,
        },
        "source_views": [{"id": "front", "image": "front.png"}],
        "regions": [
            {
                "view_id": "front",
                "group_id": "__foreground__",
                "click_sets": [
                    {
                        "events": [{"point": [500, 500], "label": 1}],
                        "positive_points": [[500, 500]],
                        "negative_points": [],
                        "initial_candidate_index": 0,
                    }
                ],
                "confirmed_mask": {
                    "path": "mask.png",
                    "sha256": sha256_file(tmp_path / "mask.png"),
                },
            }
        ],
    }
    mutation(request)

    with pytest.raises(Sam3RegionError, match=message):
        _load_request(_write_request(tmp_path, request))


def test_load_request_rejects_point_prompt_without_positive_point(
    tmp_path: Path,
) -> None:
    _write_image(tmp_path / "front.png")
    Image.new("L", (20, 10), 255).save(tmp_path / "mask.png")
    request_path = _write_request(
        tmp_path,
        {
            "schema_version": POINT_SCHEMA_VERSION,
            "source_views": [{"id": "front", "image": "front.png"}],
            "regions": [
                {
                    "view_id": "front",
                    "group_id": "__foreground__",
                    "click_sets": [
                        {"positive_points": [], "negative_points": [[0, 0]]}
                    ],
                    "confirmed_mask": {
                        "path": "mask.png",
                        "sha256": sha256_file(tmp_path / "mask.png"),
                    },
                }
            ],
        },
    )

    with pytest.raises(Sam3RegionError, match="positive_points must be non-empty"):
        _load_request(request_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda request: request.update(schema_version="unknown/v9"),
            "unsupported SAM3 request schema",
        ),
        (
            lambda request: request["source_views"].append(
                {"id": "front", "image": "front.png"}
            ),
            "source view IDs must be unique",
        ),
        (
            lambda request: request["regions"][0].update(view_id="missing"),
            "references unknown view_id",
        ),
        (
            lambda request: request["regions"].append(
                {
                    "view_id": "front",
                    "group_id": "G01",
                    "boxes": [[0, 0, 1000, 1000]],
                }
            ),
            "duplicate region identity",
        ),
        (
            lambda request: request["regions"][0].update(boxes=[[900, 0, 100, 1000]]),
            "must be ordered within",
        ),
        (
            lambda request: request["regions"][0].update(boxes=[[False, 0, 100, 100]]),
            "integer coordinates",
        ),
        (
            lambda request: request["regions"][0].update(prompt=" "),
            "prompt must be non-empty",
        ),
    ],
)
def test_load_request_rejects_ambiguous_or_unsafe_regions(
    tmp_path: Path, mutate, message: str
) -> None:
    _write_image(tmp_path / "front.png")
    request = _valid_request()
    mutate(request)
    request_path = _write_request(tmp_path, request)

    with pytest.raises(Sam3RegionError, match=message):
        _load_request(request_path)


def test_box_conversion_uses_pixel_edges_and_cxcywh() -> None:
    assert _box_pixels([100, 200, 900, 800], width=10, height=20) == (
        1,
        4,
        9,
        16,
    )
    assert _box_pixels([999, 999, 1000, 1000], width=7, height=5) == (
        6,
        4,
        7,
        5,
    )
    assert _normalized_cxcywh(
        [100, 200, 900, 800], width=10, height=20
    ) == pytest.approx([0.5, 0.5, 0.8, 0.6])
    assert _validated_box([0, 0, 1000, 1000], "box") == [0, 0, 1000, 1000]


def test_candidate_metrics_are_mask_prompt_metrics_not_box_area_guesses() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True

    metrics = _candidate_metrics(
        mask,
        box=[200, 200, 800, 800],
        width=10,
        height=10,
    )

    assert metrics["mask_pixels"] == 16
    assert metrics["intersection_pixels"] == 16
    assert metrics["image_fraction"] == pytest.approx(0.16)
    assert metrics["overlap_smaller"] == pytest.approx(1.0)
    assert metrics["prompt_coverage"] == pytest.approx(16 / 36)
    assert metrics["mask_precision"] == pytest.approx(1.0)
    assert metrics["prompt_center_inside"] is True

    with pytest.raises(Sam3RegionError, match="mask shape"):
        _candidate_metrics(
            np.zeros((9, 10), dtype=bool),
            box=[200, 200, 800, 800],
            width=10,
            height=10,
        )


def test_candidate_metrics_include_cad_seed_agreement() -> None:
    seed = np.zeros((10, 10), dtype=bool)
    seed[3:7, 3:7] = True
    candidate = seed.copy()
    candidate[2, 2] = True

    metrics = _candidate_metrics(
        candidate,
        box=[200, 200, 800, 800],
        width=10,
        height=10,
        cad_seed=seed,
    )

    assert metrics["cad_seed_pixels"] == 16
    assert metrics["cad_seed_intersection_pixels"] == 16
    assert metrics["cad_seed_iou"] == pytest.approx(16 / 17)
    assert metrics["cad_seed_precision"] == pytest.approx(16 / 17)
    assert metrics["cad_seed_recall"] == pytest.approx(1.0)


def _arbitration_record(
    group_id: str,
    mask: np.ndarray,
    *,
    score: float,
    overlap: float = 0.9,
    precision: float = 0.9,
) -> dict:
    pixels = int(np.count_nonzero(mask))
    return {
        "group_id": group_id,
        "accepted": True,
        "reason_codes": [],
        "accepted_box_count": 1,
        "box_count": 1,
        "mask_pixels": pixels,
        "image_fraction": pixels / mask.size,
        "_mask": mask.copy(),
        "box_audits": [
            {
                "accepted": True,
                "selected_candidate_index": 0,
                "candidates": [
                    {
                        "candidate_index": 0,
                        "accepted": True,
                        "model_score": score,
                        "overlap_smaller": overlap,
                        "mask_precision": precision,
                    }
                ],
            }
        ],
    }


def test_cross_group_arbitration_is_order_independent_and_quality_bound() -> None:
    full = np.ones((10, 10), dtype=bool)
    almost = full.copy()
    almost[0, :2] = False

    def arbitrate(order: list[str]) -> dict[str, dict]:
        by_id = {
            "G01": _arbitration_record("G01", full, score=0.80),
            "G02": _arbitration_record("G02", full, score=0.96),
            "G03": _arbitration_record("G03", almost, score=0.70),
        }
        result = _arbitrate_view_group_masks(
            [by_id[group_id] for group_id in order],
            minimum_intersection_pixels=8,
        )
        return {record["group_id"]: record for record in result}

    forward = arbitrate(["G01", "G02", "G03"])
    reverse = arbitrate(["G03", "G02", "G01"])
    for result in (forward, reverse):
        assert result["G02"]["accepted"] is True
        assert result["G01"]["accepted"] is False
        assert result["G03"]["accepted"] is False
        assert result["G01"]["_mask"] is None
        assert result["G01"]["reason_codes"] == ["cross_group_near_duplicate_loser"]
        assert result["G02"]["cross_group_arbitration"]["winner_group_id"] == "G02"


def test_cross_group_arbitration_preserves_containment_and_foreground() -> None:
    large = np.ones((10, 10), dtype=bool)
    small = np.zeros((10, 10), dtype=bool)
    small[3:7, 3:7] = True
    foreground = _arbitration_record("__foreground__", large, score=1.0)
    records = _arbitrate_view_group_masks(
        [
            _arbitration_record("G_LARGE", large, score=0.80),
            _arbitration_record("G_BOLT", small, score=0.75),
            foreground,
        ],
        minimum_intersection_pixels=8,
    )
    by_id = {record["group_id"]: record for record in records}

    assert by_id["G_LARGE"]["accepted"] is True
    assert by_id["G_BOLT"]["accepted"] is True
    assert by_id["__foreground__"]["accepted"] is True
    assert (
        by_id["__foreground__"]["cross_group_arbitration"]["reason"]
        == "whole_asset_foreground_excluded"
    )


def test_cross_group_arbitration_uses_group_id_only_as_final_tiebreak() -> None:
    mask = np.ones((8, 8), dtype=bool)
    records = _arbitrate_view_group_masks(
        [
            _arbitration_record("G_Z", mask, score=0.90),
            _arbitration_record("G_A", mask, score=0.90),
        ],
        minimum_intersection_pixels=8,
    )
    by_id = {record["group_id"]: record for record in records}

    assert by_id["G_A"]["accepted"] is True
    assert by_id["G_Z"]["accepted"] is False


class _ArrayTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def detach(self) -> "_ArrayTensor":
        return self

    def float(self) -> "_ArrayTensor":
        return self

    def to(self, _device: str) -> "_ArrayTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class _FakeProcessor:
    def __init__(self, masks: np.ndarray, scores: np.ndarray) -> None:
        self.masks = masks
        self.scores = scores
        self.image = None
        self.text_prompt = None
        self.geometric_prompt = None

    def set_image(self, image):
        self.image = image
        return {"image": image}

    def set_text_prompt(self, *, prompt: str, state: dict) -> None:
        assert state["image"] is self.image
        self.text_prompt = prompt

    def add_geometric_prompt(self, *, box, label: bool, state: dict) -> dict:
        assert state["image"] is self.image
        self.geometric_prompt = (box, label)
        return {
            "masks": _ArrayTensor(self.masks),
            "scores": _ArrayTensor(self.scores),
        }


def test_segment_box_selects_best_candidate_after_fail_closed_gates() -> None:
    full_image = np.ones((10, 10), dtype=bool)
    medium = np.zeros((10, 10), dtype=bool)
    medium[2:8, 2:8] = True
    best = np.zeros((10, 10), dtype=bool)
    best[3:7, 3:7] = True
    processor = _FakeProcessor(
        np.stack([full_image, medium, best])[:, None, :, :],
        np.asarray([0.99, 0.60, 0.95], dtype=np.float32),
    )
    image = Image.new("RGB", (10, 10))
    try:
        selected, audit = _segment_box(
            processor=processor,
            image=image,
            prompt="painted steel",
            box=[200, 200, 800, 800],
            minimum_model_score=0.45,
            minimum_prompt_overlap=0.25,
            maximum_image_fraction=0.80,
            minimum_mask_pixels=8,
        )
    finally:
        image.close()

    assert selected is not None
    assert np.array_equal(selected, best)
    assert audit["accepted"] is True
    assert audit["selected_candidate_index"] == 2
    assert audit["candidates"][0]["accepted"] is False
    assert "mask_too_large" in audit["candidates"][0]["reason_codes"]
    assert processor.text_prompt == "painted steel"
    assert processor.geometric_prompt == (
        pytest.approx([0.5, 0.5, 0.6, 0.6]),
        True,
    )


def test_segment_box_prefers_cad_seed_agreement_over_model_score() -> None:
    seed = np.zeros((10, 10), dtype=bool)
    seed[3:7, 3:7] = True
    merged = np.zeros((10, 10), dtype=bool)
    merged[2:8, 2:8] = True
    matching = seed.copy()
    processor = _FakeProcessor(
        np.stack([merged, matching])[:, None, :, :],
        np.asarray([0.99, 0.70], dtype=np.float32),
    )
    image = Image.new("RGB", (10, 10))
    try:
        selected, audit = _segment_box(
            processor=processor,
            image=image,
            prompt="one component",
            box=[200, 200, 800, 800],
            minimum_model_score=0.45,
            minimum_prompt_overlap=0.25,
            maximum_image_fraction=0.80,
            minimum_mask_pixels=8,
            cad_seed=seed,
        )
    finally:
        image.close()

    assert selected is not None
    assert np.array_equal(selected, matching)
    assert audit["selected_candidate_index"] == 1


def test_segment_box_uses_shape_after_translation_not_direct_projection_iou() -> None:
    seed = np.zeros((32, 32), dtype=bool)
    seed[5:13, 4:10] = True
    direct_overlap_wrong_shape = np.zeros((32, 32), dtype=bool)
    direct_overlap_wrong_shape[3:16, 3:12] = True
    translated_matching_shape = np.zeros((32, 32), dtype=bool)
    translated_matching_shape[18:26, 20:26] = True
    processor = _FakeProcessor(
        np.stack([direct_overlap_wrong_shape, translated_matching_shape])[:, None],
        np.asarray([0.99, 0.70], dtype=np.float32),
    )
    image = Image.new("RGB", (32, 32))
    try:
        selected, audit = _segment_box(
            processor=processor,
            image=image,
            prompt="one component",
            box=[0, 0, 1000, 1000],
            minimum_model_score=0.45,
            minimum_prompt_overlap=0.25,
            maximum_image_fraction=0.80,
            minimum_mask_pixels=16,
            cad_seed=seed,
        )
    finally:
        image.close()

    assert selected is not None
    assert np.array_equal(selected, translated_matching_shape)
    assert audit["selected_candidate_index"] == 1
    wrong, matching = audit["candidates"]
    assert wrong["cad_seed_iou"] > matching["cad_seed_iou"]
    assert matching["cad_shape_iou"] > wrong["cad_shape_iou"]
    assert matching["cad_shape_location_invariant"] is True


def test_segment_box_rejects_unreliable_six_pixel_cad_shape_seed() -> None:
    seed = np.zeros((16, 16), dtype=bool)
    seed[3:5, 4:7] = True
    candidate = np.zeros((16, 16), dtype=bool)
    candidate[6:10, 7:11] = True
    processor = _FakeProcessor(
        candidate[None, None],
        np.asarray([0.99], dtype=np.float32),
    )
    image = Image.new("RGB", (16, 16))
    try:
        selected, audit = _segment_box(
            processor=processor,
            image=image,
            prompt="one component",
            box=[0, 0, 1000, 1000],
            minimum_model_score=0.45,
            minimum_prompt_overlap=0.25,
            maximum_image_fraction=0.80,
            minimum_mask_pixels=8,
            cad_seed=seed,
        )
    finally:
        image.close()

    assert selected is None
    assert "cad_shape_seed_too_small" in audit["candidates"][0]["reason_codes"]


def test_segment_box_reports_gate_reasons_when_every_candidate_fails() -> None:
    too_large = np.ones((10, 10), dtype=bool)
    too_small = np.zeros((10, 10), dtype=bool)
    too_small[0, 0] = True
    low_score = np.zeros((10, 10), dtype=bool)
    low_score[2:8, 2:8] = True
    processor = _FakeProcessor(
        np.stack([too_large, too_small, low_score]),
        np.asarray([0.95, 0.95, 0.10], dtype=np.float32),
    )
    image = Image.new("RGB", (10, 10))
    try:
        selected, audit = _segment_box(
            processor=processor,
            image=image,
            prompt="visual",
            box=[200, 200, 800, 800],
            minimum_model_score=0.45,
            minimum_prompt_overlap=0.25,
            maximum_image_fraction=0.50,
            minimum_mask_pixels=8,
        )
    finally:
        image.close()

    assert selected is None
    assert audit["accepted"] is False
    reasons = [set(item["reason_codes"]) for item in audit["candidates"]]
    assert "mask_too_large" in reasons[0]
    assert "mask_too_small" in reasons[1]
    assert "insufficient_prompt_overlap" in reasons[1]
    assert "model_score_below_threshold" in reasons[2]


class _FakeInteractiveModel:
    def __init__(self, masks: np.ndarray, scores: np.ndarray) -> None:
        self.masks = masks
        self.scores = scores
        self.point_coords = None
        self.point_labels = None

    def predict_inst(
        self,
        image_state,
        *,
        point_coords,
        point_labels,
        multimask_output: bool,
    ):
        assert image_state == {"cached": True}
        assert multimask_output is True
        self.point_coords = point_coords
        self.point_labels = point_labels
        return self.masks, self.scores, np.zeros((len(self.scores), 4, 4))


def test_segment_points_uses_true_positive_and_negative_sam3_prompts() -> None:
    bad = np.ones((10, 10), dtype=bool)
    good = np.zeros((10, 10), dtype=bool)
    good[2:8, 2:8] = True
    model = _FakeInteractiveModel(
        np.stack([bad, good]),
        np.asarray([0.99, 0.80], dtype=np.float32),
    )
    image = Image.new("RGB", (10, 10))
    try:
        selected, audit = _segment_points(
            model=model,
            image_state={"cached": True},
            image=image,
            click_set={
                "positive_points": [[500, 500]],
                "negative_points": [[0, 0]],
            },
            minimum_model_score=0.45,
            minimum_prompt_overlap=1.0,
            maximum_image_fraction=0.90,
            minimum_mask_pixels=8,
        )
    finally:
        image.close()

    assert np.array_equal(selected, good)
    assert audit["selected_candidate_index"] == 1
    assert "negative_point_inside_mask" in audit["candidates"][0]["reason_codes"]
    assert np.asarray(model.point_coords) == pytest.approx(
        np.asarray([[4.5, 4.5], [0.0, 0.0]])
    )
    assert model.point_labels.tolist() == [1, 0]


def test_dense_cad_logits_keep_thin_foreground_prompt_strong() -> None:
    mask = np.zeros((443, 582), dtype=bool)
    mask[200:215, 280:284] = True

    logits = _dense_cad_mask_logits(mask, output_size=(288, 288))

    assert logits.shape == (1, 288, 288)
    assert float(logits.max()) == pytest.approx(4.0)
    assert float(logits.min()) == pytest.approx(-4.0)
    assert np.count_nonzero(logits > 0.0) > 0


class _FakeDenseInteractiveModel:
    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask
        self.mask_input = None

    def predict_inst(
        self,
        image_state,
        *,
        point_coords,
        point_labels,
        mask_input,
        multimask_output: bool,
    ):
        assert image_state == {"cached": True}
        assert multimask_output is False
        self.mask_input = np.asarray(mask_input).copy()
        return (
            self.mask[None, :, :],
            np.asarray([0.9], dtype=np.float32),
            np.zeros((1, mask_input.shape[-2], mask_input.shape[-1]), dtype=np.float32),
        )


def test_dense_cad_mask_is_passed_as_model_prompt_not_used_as_output() -> None:
    seed = np.zeros((32, 32), dtype=bool)
    seed[8:20, 10:18] = True
    photo_prediction = np.zeros_like(seed)
    photo_prediction[9:21, 11:19] = True
    model = _FakeDenseInteractiveModel(photo_prediction)
    image = Image.new("RGB", (32, 32))
    try:
        selected, audit = _segment_dense_mask_points(
            model=model,
            image_state={"cached": True},
            image=image,
            click_set={
                "positive_points": [[450, 450]],
                "negative_points": [[0, 0]],
            },
            aligned_cad_seed=seed,
            minimum_model_score=0.45,
            minimum_prompt_overlap=1.0,
            maximum_image_fraction=0.80,
            minimum_mask_pixels=16,
        )
    finally:
        image.close()

    assert np.array_equal(selected, photo_prediction)
    assert model.mask_input is not None
    assert model.mask_input.shape == (1, 256, 256)
    assert not np.array_equal(selected, seed)
    assert audit["mask_prompt"]["source"] == "aligned_whole_assembly_visible_part_id"


def test_dense_cad_automatic_negatives_are_advisory_before_shape_gate() -> None:
    seed = np.zeros((32, 32), dtype=bool)
    seed[8:24, 8:24] = True
    photo_prediction = np.zeros_like(seed)
    photo_prediction[9:23, 9:23] = True
    model = _FakeDenseInteractiveModel(photo_prediction)
    image = Image.new("RGB", (32, 32))
    try:
        selected, audit = _segment_dense_mask_points(
            model=model,
            image_state={"cached": True},
            image=image,
            click_set={
                "positive_points": [[400, 400]],
                # This synthesized negative is intentionally inside the
                # photo proposal.  It must not veto a later strict CAD gate.
                "negative_points": [[500, 500]],
            },
            aligned_cad_seed=seed,
            minimum_model_score=0.45,
            minimum_prompt_overlap=1.0,
            maximum_image_fraction=0.80,
            minimum_mask_pixels=16,
        )
    finally:
        image.close()

    assert np.array_equal(selected, photo_prediction)
    assert audit["accepted"] is False
    assert audit["automatic_negative_points_advisory"] is True
    assert "negative_point_inside_mask" in audit["candidates"][0]["reason_codes"]


def test_occluded_visible_seed_does_not_bound_amodal_candidate_area() -> None:
    visible = np.zeros((64, 64), dtype=bool)
    visible[22:27, 24:34] = True
    amodal = np.zeros_like(visible)
    amodal[16:36, 24:34] = True
    photo_prediction = np.zeros_like(visible)
    photo_prediction[17:35, 24:34] = True

    selected, audit = _evaluate_shape_guided_candidate(
        photo_prediction,
        box=[0, 0, 1000, 1000],
        width=64,
        height=64,
        aligned_seed=visible,
        aligned_amodal=amodal,
        minimum_mask_pixels=16,
    )

    assert np.array_equal(selected, photo_prediction)
    assert audit["accepted"] is True
    assert audit["shape_metrics"]["candidate_to_cad_seed_area_ratio"] > 2.0
    assert audit["shape_metrics"]["candidate_to_cad_amodal_area_ratio"] < 1.0
    assert "aligned_cad_direct_area_mismatch" not in audit["reason_codes"]


def test_shape_gate_rejects_candidate_that_claims_neighbor_part_pixels() -> None:
    visible = np.zeros((64, 64), dtype=bool)
    visible[20:40, 10:25] = True
    amodal = np.zeros_like(visible)
    amodal[20:40, 10:40] = True
    candidate = np.zeros_like(visible)
    candidate[20:40, 10:35] = True
    other_parts = np.zeros_like(visible)
    other_parts[20:40, 25:35] = True

    selected, audit = _evaluate_shape_guided_candidate(
        candidate,
        box=[0, 0, 1000, 1000],
        width=64,
        height=64,
        aligned_seed=visible,
        aligned_amodal=amodal,
        minimum_mask_pixels=16,
        other_part_seeds=other_parts,
    )

    assert selected is None
    assert audit["shape_metrics"]["other_part_overlap_fraction"] == 0.4
    assert "candidate_claims_neighboring_cad_parts" in audit["reason_codes"]


def test_automatic_shape_points_refine_a_shifted_component() -> None:
    seed = np.zeros((32, 32), dtype=bool)
    seed[8:16, 6:12] = True
    shifted = np.zeros((32, 32), dtype=bool)
    shifted[11:19, 10:16] = True
    merged = np.zeros((32, 32), dtype=bool)
    merged[7:23, 6:22] = True
    click_set, prompt_audit = _shape_seed_click_set(
        seed,
        shifted,
        box=[0, 0, 1000, 1000],
        width=32,
        height=32,
        view_shared_alignment={"translation_xy_pixels": [4.0, 3.0]},
    )
    assert prompt_audit["translation_xy_pixels"] == pytest.approx([4.0, 3.0])
    assert click_set["positive_points"]
    assert click_set["negative_points"]

    model = _FakeInteractiveModel(
        np.stack([merged, shifted]),
        np.asarray([0.99, 0.80], dtype=np.float32),
    )
    image = Image.new("RGB", (32, 32))
    try:
        selected, audit = _segment_shape_guided_points(
            model=model,
            image_state={"cached": True},
            image=image,
            cad_seed=seed,
            coarse_candidate=shifted,
            box=[0, 0, 1000, 1000],
            minimum_model_score=0.45,
            minimum_prompt_overlap=0.25,
            maximum_image_fraction=0.80,
            minimum_mask_pixels=16,
            view_shared_alignment={"translation_xy_pixels": [4.0, 3.0]},
        )
    finally:
        image.close()

    assert selected is not None
    assert np.array_equal(selected, shifted)
    assert audit["accepted"] is True
    assert audit["shape_metrics"]["cad_shape_location_invariant"] is True
    assert audit["shape_metrics"]["cad_shape_iou"] == pytest.approx(1.0)


def test_shared_camera_alignment_only_translates_the_complete_part_shape() -> None:
    seed = np.zeros((40, 48), dtype=bool)
    seed[8:20, 10:18] = True
    seed[12:15, 18:25] = True
    coarse = np.zeros_like(seed)
    coarse[12:24, 15:23] = True
    coarse[16:19, 23:30] = True

    aligned, audit = _bounded_shared_camera_alignment(
        seed,
        coarse,
        box=[0, 0, 1000, 1000],
        width=48,
        height=40,
        view_shared_alignment={"translation_xy_pixels": [5.0, 4.0]},
    )

    assert np.array_equal(aligned, coarse)
    assert audit["translation_xy_pixels"] == pytest.approx([5.0, 4.0])
    assert audit["part_local_translation_xy_pixels"] == [0.0, 0.0]
    assert audit["candidate_centroid_residual_xy_pixels"] == pytest.approx(
        [5.0, 4.0]
    )
    assert audit["per_mesh_pose_change_allowed"] is False
    assert "translation_only" in audit["alignment_model"]


def test_view_shared_alignment_cannot_follow_one_part_candidate() -> None:
    seed_a = np.zeros((40, 48), dtype=bool)
    seed_a[8:20, 10:18] = True
    seed_b = np.zeros_like(seed_a)
    seed_b[22:32, 30:40] = True
    foreground = seed_a | seed_b

    audit = _estimate_view_shared_translation(
        {"P0001": seed_a, "P0002": seed_b},
        foreground,
    )

    assert audit["translation_xy_pixels"] == [0.0, 0.0]
    assert audit["part_specific_translation_allowed"] is False
    assert audit["cad_union_foreground_recall"] == 1.0


def test_view_shared_alignment_is_not_applicable_to_foreground_without_cad_seeds(
) -> None:
    foreground = np.ones((40, 48), dtype=bool)

    audit = _estimate_view_shared_translation({}, foreground)

    assert audit == {
        "translation_xy_pixels": [0.0, 0.0],
        "maximum_translation_xy_pixels": [0, 0],
        "estimation_mode": "not_applicable_no_cad_seeds",
        "part_specific_translation_allowed": False,
        "cad_union_pixels": 0,
    }


def test_result_policy_binds_automatic_cad_shape_refinement() -> None:
    policy = result_policy(
        minimum_model_score=0.45,
        minimum_prompt_overlap=0.25,
        maximum_image_fraction=0.8,
        minimum_mask_pixels=32,
        human_interactive_requested=False,
        automatic_shape_interactive_requested=True,
        ordered_interaction_requested=False,
    )

    assert policy["per_mesh_pose_change_allowed"] is False
    assert policy["maximum_view_shared_translation_pixels"] == 12
    assert policy["automatic_shape_point_refinement"] == (
        "always_run_same_view_cad_shape_positive_negative_points"
    )
    assert "human_point_replay_policy" not in policy


def test_amodal_shape_restores_only_renderer_known_occlusion() -> None:
    amodal = np.zeros((40, 50), dtype=bool)
    amodal[10:30, 10:40] = True
    visible = amodal.copy()
    visible[14:26, 22:28] = False
    candidate = visible.copy()

    metrics = _occlusion_aware_amodal_agreement(candidate, visible, amodal)

    assert metrics["cad_known_occluded_pixels"] == 72
    assert metrics["cad_amodal_candidate_precision"] == 1.0
    assert metrics["cad_amodal_completion_iou"] == 1.0
    assert metrics["cad_amodal_shape_iou"] == 1.0
    assert metrics["cad_amodal_shape_rotation_search_degrees"] == 0.0


def test_amodal_shape_rejects_neighbor_extension_even_when_visible_seed_matches() -> None:
    amodal = np.zeros((60, 80), dtype=bool)
    amodal[15:45, 20:50] = True
    visible = amodal.copy()
    visible[25:35, 30:40] = False
    merged_neighbor = visible.copy()
    merged_neighbor[20:45, 50:75] = True

    metrics = _occlusion_aware_amodal_agreement(
        merged_neighbor, visible, amodal
    )

    assert metrics["cad_amodal_candidate_precision"] < 0.88
    assert metrics["cad_amodal_completion_iou"] < 0.75


def test_shape_points_add_neighbor_part_centers_as_negative_prompts() -> None:
    seed = np.zeros((48, 64), dtype=bool)
    seed[18:28, 24:34] = True
    neighbor = np.zeros_like(seed)
    neighbor[18:28, 38:48] = True

    click_set, audit = _shape_seed_click_set(
        seed,
        seed,
        box=[250, 250, 900, 750],
        width=64,
        height=48,
        other_part_seeds=neighbor,
    )

    negative_pixels = {
        (
            round(point[0] * 63 / 1000),
            round(point[1] * 47 / 1000),
        )
        for point in click_set["negative_points"]
    }
    assert any(38 <= x < 48 and 18 <= y < 28 for x, y in negative_pixels)
    assert audit["neighboring_part_negative_point_count"] == 1


class _FakeOrderedInteractiveModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict_inst(
        self,
        image_state,
        *,
        point_coords,
        point_labels,
        mask_input=None,
        multimask_output: bool,
    ):
        assert image_state == {"cached": True}
        self.calls.append(
            {
                "coords": np.asarray(point_coords).copy(),
                "labels": np.asarray(point_labels).copy(),
                "mask_input": (
                    None if mask_input is None else np.asarray(mask_input).copy()
                ),
                "multimask_output": multimask_output,
            }
        )
        count = 3 if multimask_output else 1
        masks = np.zeros((count, 10, 10), dtype=bool)
        for candidate in range(count):
            masks[
                candidate, 2 : 8 + min(candidate, 1), 2 : 8 + min(candidate, 1)
            ] = True
            for (x, y), label in zip(point_coords, point_labels):
                if label == 0:
                    masks[candidate, int(round(y)), int(round(x))] = False
        scores = np.asarray([0.9, 0.8, 0.7][:count], dtype=np.float32)
        logits = np.stack(
            [np.full((7, 7), len(self.calls) + index) for index in range(count)]
        )
        return masks, scores, logits


def test_formal_ordered_replay_uses_previous_logits_and_single_refinement() -> None:
    model = _FakeOrderedInteractiveModel()
    image = Image.new("RGB", (10, 10))
    try:
        selected, audit = _segment_ordered_points(
            model=model,
            image_state={"cached": True},
            image=image,
            click_set={
                "events": [
                    {"point": [500, 500], "label": 1},
                    {"point": [0, 0], "label": 0},
                ],
                "positive_points": [[500, 500]],
                "negative_points": [[0, 0]],
                "initial_candidate_index": 1,
            },
            minimum_prompt_overlap=1.0,
            maximum_image_fraction=0.90,
            minimum_mask_pixels=8,
        )
    finally:
        image.close()

    assert selected is not None
    assert audit["accepted"] is True
    assert len(audit["event_audits"]) == 2
    assert model.calls[0]["multimask_output"] is True
    assert model.calls[0]["mask_input"] is None
    assert model.calls[1]["multimask_output"] is False
    assert model.calls[1]["mask_input"].shape == (1, 7, 7)
    assert model.calls[1]["labels"].tolist() == [1, 0]
