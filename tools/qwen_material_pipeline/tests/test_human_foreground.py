from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading

import numpy as np
import pytest
from PIL import Image

from qwen_material_pipeline.segmentation.human_foreground import (
    ANNOTATION_SCHEMA_VERSION,
    LEGACY_ANNOTATION_SCHEMA_VERSION,
    HumanForegroundError,
    canonical_sha256,
    grid_to_pixel,
    load_annotations,
    materialize_annotation_bundle,
    pixel_to_grid,
    require_replay_policy,
    replay_ordered_click_set,
    select_point_candidate,
    sha256_file,
    validate_click_sets,
    validate_ordered_click_sets,
)
from qwen_material_pipeline.segmentation import human_foreground as human_module


def _annotation(tmp_path: Path) -> tuple[Path, list[tuple[str, Path]]]:
    repository = tmp_path / "sam3"
    repository.mkdir()
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"test checkpoint")
    source_views = []
    references = []
    for view_id, colour in (("front", (20, 140, 30)), ("side", (30, 120, 20))):
        image_path = tmp_path / f"{view_id}.png"
        image = Image.new("RGB", (10, 8), colour)
        image.save(image_path)
        pixels = np.asarray(image, dtype=np.uint8)
        mask_array = np.zeros((8, 10), dtype=np.uint8)
        mask_array[1:7, 2:8] = 1
        mask_path = tmp_path / f"{view_id}-mask.png"
        Image.fromarray(mask_array * 255).save(mask_path)
        source_views.append(
            {
                "id": view_id,
                "image": str(image_path),
                "image_sha256": sha256_file(image_path),
                "decoded_rgb_sha256": hashlib.sha256(
                    pixels.tobytes(order="C")
                ).hexdigest(),
                "width": 10,
                "height": 8,
                "click_sets": [
                    {
                        "events": [
                            {"point": [500, 500], "label": 1},
                            {"point": [0, 0], "label": 0},
                        ],
                        "positive_points": [[500, 500]],
                        "negative_points": [[0, 0]],
                        "initial_candidate_index": 0,
                    }
                ],
                "confirmed_mask": {
                    "path": mask_path.name,
                    "sha256": sha256_file(mask_path),
                    "decoded_mask_sha256": hashlib.sha256(
                        mask_array.tobytes(order="C")
                    ).hexdigest(),
                    "mask_pixels": 36,
                    "image_fraction": 0.45,
                },
            }
        )
        references.append((view_id, image_path))
    unsigned = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "prompt_authority": "human_confirmed_sam3_interactive_points",
        "coordinate_space": {
            "type": "exif_transposed_image_grid",
            "grid_size": 1000,
            "origin": "top_left",
            "axes": "x_right_y_down",
        },
        "sam3": {
            "repository": str(tmp_path / "sam3"),
            "repository_revision": None,
            "checkpoint": str(tmp_path / "sam3.pt"),
            "checkpoint_sha256": sha256_file(checkpoint),
            "device": "cuda",
            "mode": "instance_interactivity",
        },
        "policy": {
            "minimum_model_score": 0.45,
            "human_point_model_score_authority": "advisory",
            "minimum_prompt_agreement": 0.25,
            "maximum_image_fraction": 0.90,
            "minimum_mask_pixels": 32,
            "disconnected_region_policy": "incremental_instances_then_union",
            "interaction_policy": (
                "smart_outside_add_inside_refine_with_explicit_overrides"
            ),
            "ordered_replay_policy": (
                "first_multimask_then_previous_logits_single_mask"
            ),
        },
        "source_views": source_views,
        "confirmation": {
            "all_views_confirmed": True,
            "confirmed_view_ids": ["front", "side"],
            "human_mask_is_authoritative": True,
        },
    }
    document = {
        **unsigned,
        "integrity": {"document_sha256": canonical_sha256(unsigned)},
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, references


def test_annotations_are_hash_bound_to_all_reference_pixels_and_masks(
    tmp_path: Path,
) -> None:
    path, references = _annotation(tmp_path)

    document, masks = load_annotations(path, references=references)

    assert [view["id"] for view in document["source_views"]] == ["front", "side"]
    assert set(masks) == {"front", "side"}

    Image.new("RGB", (10, 8), (255, 0, 0)).save(references[0][1])
    with pytest.raises(HumanForegroundError, match="source image changed"):
        load_annotations(path, references=references)


def test_annotations_reject_changed_click_without_integrity_update(
    tmp_path: Path,
) -> None:
    path, references = _annotation(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["source_views"][0]["click_sets"][0]["events"][0]["point"] = [
        250,
        500,
    ]
    document["source_views"][0]["click_sets"][0]["positive_points"] = [[250, 500]]
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(HumanForegroundError, match="integrity mismatch"):
        load_annotations(path, references=references)


def test_legacy_annotation_v1_remains_readable(tmp_path: Path) -> None:
    path, references = _annotation(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["schema_version"] = LEGACY_ANNOTATION_SCHEMA_VERSION
    for view in document["source_views"]:
        for click_set in view["click_sets"]:
            click_set.pop("events")
            click_set.pop("initial_candidate_index")
    document["policy"].pop("interaction_policy")
    document["policy"].pop("ordered_replay_policy")
    document["policy"]["disconnected_region_policy"] = "separate_click_sets_then_union"
    unsigned = {key: value for key, value in document.items() if key != "integrity"}
    document["integrity"] = {"document_sha256": canonical_sha256(unsigned)}
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded, _masks = load_annotations(path, references=references)

    assert loaded["schema_version"] == LEGACY_ANNOTATION_SCHEMA_VERSION


@pytest.mark.parametrize(
    "click_sets",
    [
        [],
        [{"positive_points": [], "negative_points": [[0, 0]]}],
        [{"positive_points": [[1001, 0]], "negative_points": []}],
        [{"positive_points": [[500, 500]], "negative_points": [[500, 500]]}],
    ],
)
def test_click_set_contract_rejects_unsafe_prompts(click_sets) -> None:
    with pytest.raises(HumanForegroundError):
        validate_click_sets(click_sets, "click_sets")


def test_ordered_click_set_contract_rejects_reordered_derived_points() -> None:
    click_set = {
        "events": [
            {"point": [100, 100], "label": 1},
            {"point": [200, 200], "label": 0},
            {"point": [300, 300], "label": 1},
        ],
        "positive_points": [[300, 300], [100, 100]],
        "negative_points": [[200, 200]],
        "initial_candidate_index": 0,
    }

    with pytest.raises(HumanForegroundError, match="do not match ordered events"):
        validate_ordered_click_sets([click_set], "click_sets")


def test_coordinate_grid_round_trip_uses_exif_transposed_pixel_space() -> None:
    point = pixel_to_grid([9, 7], width=10, height=8)
    assert point == [1000, 1000]
    assert grid_to_pixel(point, width=10, height=8) == [9.0, 7.0]


def test_point_candidate_gate_prefers_prompt_compliance_over_raw_score() -> None:
    wrong = np.ones((8, 10), dtype=bool)
    right = np.zeros((8, 10), dtype=bool)
    right[1:7, 2:8] = True

    selected, audit = select_point_candidate(
        np.stack([wrong, right]),
        np.asarray([0.99, 0.80]),
        positive_points=[[500, 500]],
        negative_points=[[0, 0]],
        width=10,
        height=8,
        minimum_model_score=0.45,
        minimum_prompt_agreement=1.0,
        maximum_image_fraction=0.90,
        minimum_mask_pixels=8,
    )

    assert np.array_equal(selected, right)
    assert audit["selected_candidate_index"] == 1
    assert "negative_point_inside_mask" in audit["candidates"][0]["reason_codes"]


def test_rejected_point_candidate_can_be_previewed_without_becoming_accepted() -> None:
    candidate = np.zeros((8, 10), dtype=bool)
    candidate[2:6, 3:7] = True

    selected, audit = select_point_candidate(
        np.stack([candidate]),
        np.asarray([0.20]),
        positive_points=[[500, 500]],
        negative_points=[],
        width=10,
        height=8,
        minimum_model_score=0.45,
        minimum_prompt_agreement=1.0,
        maximum_image_fraction=0.90,
        minimum_mask_pixels=8,
        allow_rejected_preview=True,
    )

    assert np.array_equal(selected, candidate)
    assert audit["accepted"] is False
    assert audit["preview_only"] is True
    assert "model_score_below_threshold" in audit["candidates"][0]["reason_codes"]


class _FakeInteractiveModel:
    def __init__(self, *, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.calls: list[dict[str, object]] = []

    def predict_inst(
        self,
        _state,
        *,
        point_coords,
        point_labels,
        mask_input=None,
        multimask_output=True,
    ):
        coords = np.asarray(point_coords, dtype=np.float32)
        labels = np.asarray(point_labels, dtype=np.int32)
        self.calls.append(
            {
                "coords": coords.copy(),
                "labels": labels.copy(),
                "mask_input": (
                    None if mask_input is None else np.asarray(mask_input).copy()
                ),
                "multimask_output": multimask_output,
            }
        )
        radii = (1, 2, 3) if multimask_output else (1,)
        masks = []
        for radius in radii:
            mask = np.zeros((self.height, self.width), dtype=bool)
            for (x_raw, y_raw), label in zip(coords, labels):
                x = min(self.width - 1, max(0, int(round(float(x_raw)))))
                y = min(self.height - 1, max(0, int(round(float(y_raw)))))
                if label == 1:
                    mask[
                        max(0, y - radius) : min(self.height, y + radius + 1),
                        max(0, x - radius) : min(self.width, x + radius + 1),
                    ] = True
                else:
                    mask[y, x] = False
            masks.append(mask)
        scores = np.asarray([0.95, 0.85, 0.75][: len(masks)], dtype=np.float32)
        call_number = len(self.calls)
        logits = np.stack(
            [
                np.full((6, 6), call_number + index, dtype=np.float32)
                for index in range(len(masks))
            ]
        )
        return np.stack(masks), scores, logits


def _interactive_session(tmp_path: Path, *, width: int = 12, height: int = 10):
    image_path = tmp_path / "front.png"
    image = Image.new("RGB", (width, height), "white")
    image.save(image_path)
    view = human_module._ViewState(
        view_id="front",
        path=image_path,
        image=image,
        decoded_rgb_sha256="0" * 64,
    )
    session = human_module.InteractiveForegroundSession.__new__(
        human_module.InteractiveForegroundSession
    )
    session._lock = threading.RLock()
    session.views = {"front": view}
    session.view_order = ["front"]
    session.active_view_id = "front"
    session._active_image_state = {"image": "fake"}
    session.model = _FakeInteractiveModel(width=width, height=height)
    session.minimum_prompt_agreement = 1.0
    session.maximum_image_fraction = 0.90
    session.minimum_mask_pixels = 1
    return session, view


def test_incremental_interaction_reuses_logits_and_preserves_event_order(
    tmp_path: Path,
) -> None:
    session, view = _interactive_session(tmp_path)

    session.add_point(2, 2, mode="smart_foreground")
    session.add_point(3, 2, mode="refine_active")
    session.add_point(3, 3, mode="background")

    assert len(view.instances) == 1
    assert [event["label"] for event in view.instances[0].events] == [1, 1, 0]
    assert session.model.calls[0]["multimask_output"] is True
    assert session.model.calls[0]["mask_input"] is None
    assert session.model.calls[1]["multimask_output"] is False
    assert session.model.calls[1]["mask_input"] is not None
    assert session.model.calls[2]["multimask_output"] is False
    assert session.model.calls[2]["labels"].tolist() == [1, 1, 0]


def test_smart_outside_click_adds_instance_and_undo_restores_union(
    tmp_path: Path,
) -> None:
    session, view = _interactive_session(tmp_path)
    session.add_point(2, 2, mode="smart_foreground")
    first_union = session.union_mask("front").copy()

    session.add_point(9, 7, mode="smart_foreground")

    assert len(view.instances) == 2
    assert np.count_nonzero(session.union_mask("front")) > np.count_nonzero(first_union)
    assert session.model.calls[-1]["multimask_output"] is True
    assert session.model.calls[-1]["mask_input"] is None

    session.undo_point()
    assert len(view.instances) == 1
    assert np.array_equal(session.union_mask("front"), first_union)


def test_rejected_active_instance_must_be_resolved_before_adding_another(
    tmp_path: Path,
) -> None:
    session, view = _interactive_session(tmp_path)
    session.minimum_mask_pixels = 100
    session.add_point(2, 2, mode="smart_foreground")

    assert view.instances[0].audit["accepted"] is False
    with pytest.raises(HumanForegroundError, match="尚未通过门限"):
        session.add_point(9, 7, mode="smart_foreground")

    assert len(view.instances) == 1
    assert "仅预览" in session.status()


def test_undo_history_stores_only_packbit_compressed_affected_instance(
    tmp_path: Path,
) -> None:
    session, view = _interactive_session(tmp_path, width=80, height=60)
    session.add_point(20, 20, mode="smart_foreground")
    session.add_point(21, 20, mode="refine_active")

    action = view.undo_stack[-1]
    assert action["kind"] == "updated"
    assert "instances" not in action
    assert action["instance"]["mask_bits"].nbytes < view.instances[0].mask.nbytes

    session.undo_point()
    assert len(view.instances[0].events) == 1


def test_candidate_cycle_is_persisted_and_formal_replay_uses_same_candidate(
    tmp_path: Path,
) -> None:
    session, view = _interactive_session(tmp_path)
    session.add_point(5, 5, mode="smart_foreground")
    first_pixels = int(view.instances[0].mask.sum())

    session.cycle_active_candidate()
    click_set = session._click_set(view.instances[0])
    replayed, _logits, audit = replay_ordered_click_set(
        model=session.model,
        image_state=session._active_image_state,
        image=view.image,
        click_set=click_set,
        minimum_prompt_agreement=1.0,
        maximum_image_fraction=0.90,
        minimum_mask_pixels=1,
    )

    assert click_set["initial_candidate_index"] == 1
    assert int(view.instances[0].mask.sum()) > first_pixels
    assert np.array_equal(replayed, view.instances[0].mask)
    assert audit["event_audits"][0]["candidate_selection"] == (
        "persisted_initial_candidate"
    )


def test_visual_outputs_share_size_and_cutout_alpha_is_union(tmp_path: Path) -> None:
    session, view = _interactive_session(tmp_path)
    session.add_point(2, 2, mode="smart_foreground")

    overlay = session.render_preview()
    mask = session.render_mask()
    cutout = session.render_cutout()

    assert overlay.size == mask.size == cutout.size == view.image.size
    assert np.array_equal(np.asarray(cutout)[..., 3] > 0, session.union_mask("front"))


def test_formal_replay_rejects_different_ui_gate_policy(tmp_path: Path) -> None:
    path, references = _annotation(tmp_path)
    document, _masks = load_annotations(path, references=references)

    with pytest.raises(HumanForegroundError, match="replay policy differs"):
        require_replay_policy(
            document,
            minimum_prompt_agreement=0.25,
            maximum_image_fraction=0.80,
            minimum_mask_pixels=32,
        )


def test_confirm_view_gates_union_of_disconnected_regions(tmp_path: Path) -> None:
    image_path = tmp_path / "front.png"
    image = Image.new("RGB", (10, 10), "white")
    image.save(image_path)
    first = np.zeros((10, 10), dtype=bool)
    second = np.zeros((10, 10), dtype=bool)
    first[:, :4] = True
    second[:, 4:8] = True
    view = human_module._ViewState(
        view_id="front",
        path=image_path,
        image=image,
        decoded_rgb_sha256="0" * 64,
        instances=[
            human_module._InstanceState(
                [], 0, first, np.zeros((1, 6, 6)), {"accepted": True}
            ),
            human_module._InstanceState(
                [], 0, second, np.zeros((1, 6, 6)), {"accepted": True}
            ),
        ],
    )
    session = human_module.InteractiveForegroundSession.__new__(
        human_module.InteractiveForegroundSession
    )
    session._lock = threading.RLock()
    session.views = {"front": view}
    session.active_view_id = "front"
    session.minimum_mask_pixels = 1
    session.maximum_image_fraction = 0.50
    session.render_preview = lambda _view_id=None: image
    session.status = lambda _view_id=None: "ok"

    with pytest.raises(HumanForegroundError, match="并集覆盖范围过大"):
        session.confirm_view()


def test_materialized_bundle_keeps_confirmed_masks_with_sealed_copy(
    tmp_path: Path,
) -> None:
    path, references = _annotation(tmp_path)
    document, masks = load_annotations(path, references=references)
    destination = tmp_path / "analysis" / "sam3_foreground_annotations.json"

    sealed = materialize_annotation_bundle(
        document,
        destination=destination,
        references=references,
        repository=tmp_path / "sam3",
        checkpoint=tmp_path / "sam3.pt",
    )
    for original_mask in masks.values():
        original_mask.unlink()
    reloaded, sealed_masks = load_annotations(
        destination,
        references=references,
        repository=tmp_path / "sam3",
        checkpoint=tmp_path / "sam3.pt",
    )

    assert reloaded["integrity"] == sealed["integrity"]
    assert set(sealed_masks) == {"front", "side"}
    assert all(path.parent.name.endswith("_masks") for path in sealed_masks.values())


def test_session_rejects_stale_mask_directory_before_loading_model(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "sam3"
    repository.mkdir()
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "annotations.json"
    (tmp_path / "annotations_masks").mkdir()

    with pytest.raises(FileExistsError, match="mask directory already exists"):
        human_module.InteractiveForegroundSession(
            references=[],
            repository=repository,
            checkpoint=checkpoint,
            output=output,
        )
