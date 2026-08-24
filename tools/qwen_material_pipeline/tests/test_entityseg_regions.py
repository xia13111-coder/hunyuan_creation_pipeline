from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

import qwen_material_pipeline.segmentation.entityseg_regions as entityseg_regions
from qwen_material_pipeline.segmentation.entityseg_regions import (
    _cad_location_agreement,
    _expanded_crop,
    _prediction_candidates,
    _select_candidate,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_run_keeps_inference_seed_separate_from_cad_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "front.png"
    seed_path = tmp_path / "front__P0001.png"
    source = np.zeros((32, 40, 3), dtype=np.uint8)
    cad_seed = np.zeros((32, 40), dtype=np.uint8)
    cad_seed[8:24, 10:30] = 255
    assert cv2.imwrite(str(source_path), source)
    assert cv2.imwrite(str(seed_path), cad_seed)
    request = {
        "source_views": [
            {
                "id": "front",
                "image": str(source_path),
                "image_sha256": _sha256_file(source_path),
            }
        ],
        "regions": [
            {
                "view_id": "front",
                "group_id": "P0001",
                "boxes": [[250, 250, 750, 750]],
                "cad_projection_seed": {
                    "path": str(seed_path),
                    "sha256": _sha256_file(seed_path),
                },
            }
        ],
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    repository = tmp_path / "cropformer"
    repository.mkdir()
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pth"
    config.write_text("MODEL: {}\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(entityseg_regions, "_seed_inference", lambda _seed: None)
    setup_calls: list[dict[str, object]] = []

    def setup_predictor(**kwargs: object) -> object:
        setup_calls.append(kwargs)
        return object()

    monkeypatch.setattr(entityseg_regions, "_setup_predictor", setup_predictor)
    monkeypatch.setattr(
        entityseg_regions,
        "_prediction_candidates",
        lambda *_args, **_kwargs: [],
    )

    result = entityseg_regions.run(
        request_path=request_path,
        cropformer_root=repository,
        config_path=config,
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "output",
        seed=7,
    )

    assert result["policy"]["inference_seed"] == 7
    assert isinstance(result["policy"]["inference_seed"], int)
    assert result["policy"]["inference_short_edge"] == 800
    assert result["policy"]["crop_sample_count"] == 4
    assert setup_calls == [
        {
            "cropformer_root": repository,
            "config_path": config,
            "checkpoint_path": checkpoint,
            "seed": 7,
            "inference_short_edge": 800,
        }
    ]
    persisted = json.loads((tmp_path / "output" / "manifest.json").read_text())
    assert persisted["policy"]["inference_seed"] == 7
    assert persisted["policy"]["inference_short_edge"] == 800
    assert persisted["policy"]["crop_sample_count"] == 4


def test_inference_short_edge_parser_preserves_the_audited_default() -> None:
    parser = entityseg_regions._parser()
    common = [
        "--request",
        "request.json",
        "--cropformer-root",
        "cropformer",
        "--config",
        "config.yaml",
        "--checkpoint",
        "checkpoint.pth",
        "--output-dir",
        "output",
    ]

    assert parser.parse_args(common).inference_short_edge == 800
    assert (
        parser.parse_args([*common, "--inference-short-edge", "512"])
        .inference_short_edge
        == 512
    )


def test_seed_inference_enables_expandable_cuda_segments_before_torch_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        manual_seed_all=lambda seed: calls.append(("cuda_seed", seed)),
    )
    fake_cudnn = types.SimpleNamespace(benchmark=True, deterministic=False)
    fake_torch = types.SimpleNamespace(
        cuda=fake_cuda,
        backends=types.SimpleNamespace(cudnn=fake_cudnn),
        manual_seed=lambda seed: calls.append(("cpu_seed", seed)),
        use_deterministic_algorithms=lambda enabled, warn_only: calls.append(
            ("deterministic", (enabled, warn_only))
        ),
    )
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    entityseg_regions._seed_inference(9)

    assert (
        entityseg_regions.os.environ["PYTORCH_CUDA_ALLOC_CONF"]
        == "expandable_segments:True"
    )
    assert ("cpu_seed", 9) in calls
    assert ("cuda_seed", 9) in calls
    assert fake_cudnn.benchmark is False
    assert fake_cudnn.deterministic is True


def test_prediction_candidates_release_cuda_outputs_after_cpu_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_cache_calls: list[bool] = []

    class FakeTensor:
        def __init__(self, value: np.ndarray) -> None:
            self.value = value

        def detach(self) -> FakeTensor:
            return self

        def to(self, _device: str) -> FakeTensor:
            return self

        def float(self) -> FakeTensor:
            return self

        def numpy(self) -> np.ndarray:
            return self.value

    instances = types.SimpleNamespace(
        pred_masks=FakeTensor(np.ones((1, 4, 5), dtype=np.uint8)),
        scores=FakeTensor(np.asarray([0.9], dtype=np.float32)),
    )
    predictor = types.SimpleNamespace(
        run_on_image=lambda _image: {"instances": instances}
    )
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: empty_cache_calls.append(True),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    candidates = _prediction_candidates(
        predictor,
        np.zeros((4, 5, 3), dtype=np.uint8),
        source="test",
        full_shape=(4, 5),
        minimum_score=0.3,
    )

    assert len(candidates) == 1
    assert candidates[0]["mask"].shape == (4, 5)
    assert empty_cache_calls == [True]


def test_expanded_crop_is_resolution_bounded() -> None:
    assert _expanded_crop(
        [0, 0, 100, 100],
        width=200,
        height=100,
        context_fraction=0.2,
    ) == (0, 0, 64, 64)


def test_cad_location_agreement_is_normalized_by_part_scale() -> None:
    seed = np.zeros((80, 80), dtype=bool)
    candidate = np.zeros_like(seed)
    seed[20:40, 20:40] = True
    candidate[22:42, 22:42] = True

    metrics = _cad_location_agreement(candidate, seed)

    assert metrics["cad_centroid_distance_pixels"] == np.sqrt(8.0)
    assert metrics["cad_centroid_distance_normalized"] == 0.1
    assert metrics["cad_direct_intersection_pixels"] == 18 * 18


def test_selection_rejects_neighboring_identical_component() -> None:
    source = np.zeros((100, 120, 3), dtype=np.uint8)
    seed = np.zeros((100, 120), dtype=bool)
    correct = np.zeros_like(seed)
    identical_neighbor = np.zeros_like(seed)
    seed[30:60, 40:55] = True
    correct[30:60, 40:55] = True
    identical_neighbor[30:60, 60:75] = True

    selected, audit = _select_candidate(
        [
            {
                "source": "cad_local_crop",
                "prediction_index": 0,
                "model_score": 0.99,
                "mask": identical_neighbor,
            },
            {
                "source": "cad_local_crop",
                "prediction_index": 1,
                "model_score": 0.60,
                "mask": correct,
            },
        ],
        seed=seed,
        source_image=source,
        minimum_shape_iou=0.5,
        minimum_area_agreement=0.5,
        maximum_centroid_distance=0.15,
    )

    assert selected is not None
    assert selected["prediction_index"] == 1
    neighbor = next(row for row in audit if row["prediction_index"] == 0)
    assert neighbor["accepted"] is False
    assert "cad_centroid_too_far_from_registered_part" in neighbor["reason_codes"]


def test_selection_does_not_move_cad_template_per_candidate() -> None:
    source = np.zeros((100, 120, 3), dtype=np.uint8)
    seed = np.zeros((100, 120), dtype=bool)
    shifted = np.zeros_like(seed)
    seed[30:60, 40:60] = True
    shifted[32:62, 42:62] = True

    selected, audit = _select_candidate(
        [
            {
                "source": "cad_local_crop",
                "prediction_index": 0,
                "model_score": 0.8,
                "mask": shifted,
            }
        ],
        seed=seed,
        source_image=source,
        minimum_shape_iou=0.5,
        minimum_area_agreement=0.5,
        maximum_centroid_distance=0.15,
        box=[250, 200, 700, 800],
    )

    assert selected is not None
    row = audit[0]
    assert row["cad_template_alignment"]["translation_xy_pixels"] == [0.0, 0.0]
    assert row["cad_template_alignment"]["part_local_translation_xy_pixels"] == [
        0.0,
        0.0,
    ]
    assert row["cad_template_alignment"]["candidate_centroid_residual_xy_pixels"] == [
        2.0,
        2.0,
    ]
    assert row["cad_template_alignment"]["per_mesh_pose_change_allowed"] is False
    assert row["cad_direct_iou"] == pytest.approx(0.72413793)
    assert row["registered_cad_centroid_distance_normalized"] > 0.0


def test_selection_uses_one_shared_view_translation_for_every_candidate() -> None:
    source = np.zeros((100, 120, 3), dtype=np.uint8)
    seed = np.zeros((100, 120), dtype=bool)
    shifted = np.zeros_like(seed)
    seed[30:60, 40:60] = True
    shifted[33:63, 44:64] = True
    shared = {
        "translation_xy_pixels": [4.0, 3.0],
        "maximum_translation_xy_pixels": [12, 12],
        "estimation_mode": (
            "whole_workpiece_foreground_to_visible_cad_union_integer_translation"
        ),
        "part_specific_translation_allowed": False,
        "cad_union_pixels": 600,
    }

    selected, audit = _select_candidate(
        [
            {
                "source": "cad_local_crop",
                "prediction_index": 0,
                "model_score": 0.8,
                "mask": shifted,
            }
        ],
        seed=seed,
        source_image=source,
        minimum_shape_iou=0.5,
        minimum_area_agreement=0.5,
        maximum_centroid_distance=0.15,
        box=[250, 200, 700, 800],
        view_shared_alignment=shared,
    )

    assert selected is not None
    alignment = audit[0]["cad_template_alignment"]
    assert alignment["translation_xy_pixels"] == [4.0, 3.0]
    assert alignment["part_local_translation_xy_pixels"] == [0.0, 0.0]
    assert alignment["part_specific_translation_allowed"] is False
    assert alignment["per_mesh_pose_change_allowed"] is False
