from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from asset_pipeline import manual_material_cli
from qwen_material_pipeline.segmentation.human_foreground import (
    ANNOTATION_SCHEMA_VERSION,
    canonical_sha256,
    sha256_file,
)


def _annotations(root: Path) -> Path:
    views = []
    for index, view_id in enumerate(("front", "iso")):
        image = root / f"{view_id}.png"
        image_pixels = np.full((10, 12, 3), 40 + index * 80, dtype=np.uint8)
        Image.fromarray(image_pixels, mode="RGB").save(image)
        mask = root / f"{view_id}_mask.png"
        mask_pixels = np.zeros((10, 12), dtype=np.uint8)
        mask_pixels[2:8, 3:9] = 255
        Image.fromarray(mask_pixels, mode="L").save(mask)
        binary_mask = (mask_pixels > 0).astype(np.uint8)
        views.append(
            {
                "id": view_id,
                "image": str(image),
                "image_sha256": sha256_file(image),
                "decoded_rgb_sha256": hashlib.sha256(
                    image_pixels.tobytes(order="C")
                ).hexdigest(),
                "width": 12,
                "height": 10,
                "click_sets": [
                    {
                        "events": [{"point": [500, 500], "label": 1}],
                        "positive_points": [[500, 500]],
                        "negative_points": [],
                        "initial_candidate_index": 0,
                    }
                ],
                "confirmed_mask": {
                    "path": str(mask),
                    "sha256": sha256_file(mask),
                    "decoded_mask_sha256": hashlib.sha256(
                        binary_mask.tobytes(order="C")
                    ).hexdigest(),
                    "mask_pixels": 36,
                    "image_fraction": 0.3,
                },
            }
        )
    unsigned = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "prompt_authority": "human_confirmed_sam3_interactive_points",
        "coordinate_space": {
            "type": "exif_transposed_image_grid",
            "grid_size": 1000,
            "origin": "top_left",
            "axes": "x_right_y_down",
        },
        "sam3": {"mode": "instance_interactivity"},
        "policy": {
            "minimum_model_score": 0.45,
            "human_point_model_score_authority": "advisory",
            "minimum_prompt_agreement": 0.25,
            "maximum_image_fraction": 0.9,
            "minimum_mask_pixels": 32,
            "disconnected_region_policy": "incremental_instances_then_union",
            "interaction_policy": (
                "smart_outside_add_inside_refine_with_explicit_overrides"
            ),
            "ordered_replay_policy": (
                "first_multimask_then_previous_logits_single_mask"
            ),
        },
        "source_views": views,
        "confirmation": {
            "all_views_confirmed": True,
            "confirmed_view_ids": ["front", "iso"],
            "human_mask_is_authoritative": True,
        },
    }
    path = root / "annotations.json"
    path.write_text(
        json.dumps(
            {
                **unsigned,
                "integrity": {"document_sha256": canonical_sha256(unsigned)},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_references_come_from_human_sam3_annotations(tmp_path: Path) -> None:
    annotations = _annotations(tmp_path)

    references = manual_material_cli.references_from_annotations(annotations)

    assert references == [
        f"front={(tmp_path / 'front.png').resolve()}",
        f"iso={(tmp_path / 'iso.png').resolve()}",
    ]


def test_annotations_are_rejected_when_a_reference_image_changes(
    tmp_path: Path,
) -> None:
    annotations = _annotations(tmp_path)
    Image.new("RGB", (12, 10), (255, 0, 0)).save(tmp_path / "front.png")

    with pytest.raises(ValueError, match="source image changed"):
        manual_material_cli.references_from_annotations(annotations)


def test_module_entrypoint_exposes_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "asset_pipeline.manual_material_cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--sam3-annotations" in completed.stdout


def test_dedicated_cli_does_not_expose_ambiguous_auto_fallback() -> None:
    parser = manual_material_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--stp",
                "asset.stp",
                "--sam3-annotations",
                "annotations.json",
                "--visual-inference-mode",
                "auto",
            ]
        )


def test_one_command_expands_to_the_owning_manual_workflow(
    tmp_path: Path,
) -> None:
    stp = tmp_path / "asset.stp"
    stp.write_text("STEP", encoding="utf-8")
    annotations = _annotations(tmp_path)
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    output = tmp_path / "run"
    args = manual_material_cli.build_parser().parse_args(
        [
            "--stp",
            str(stp),
            "--sam3-annotations",
            str(annotations),
            "--output",
            str(output),
            "--config",
            str(config),
        ]
    )
    with (
        patch.object(
            manual_material_cli.runtime,
            "configure_runtime",
            return_value={"CONDA_ENV": "hunyuan_sam3d"},
        ),
        patch.object(manual_material_cli, "validate_cad_input_path"),
        patch.object(manual_material_cli, "load_visual_material_config"),
        patch.object(
            manual_material_cli,
            "run_manual_cad_workflow",
            return_value={"completion_state": "COMPLETED"},
        ) as workflow,
    ):
        result = manual_material_cli.run(args)

    call = workflow.call_args.kwargs
    assert call["auto_visual_materials"] is True
    assert call["visual_inference_mode"] == "live"
    assert call["visual_foreground_annotations"] == str(annotations)
    assert call["acknowledge_mvinverse_noncommercial"] is True
    assert call["allow_policy_material_fallback"] is True
    assert call["resume"] is False
    assert call["visual_material_references"][0].startswith("front=")
    assert result["mode"] == "manual_part_id_materials"
    assert (output / "pipeline_result.json").is_file()


def test_bundled_mode_uses_annotations_only_as_reference_manifest(
    tmp_path: Path,
) -> None:
    stp = tmp_path / "asset.stp"
    stp.write_text("STEP", encoding="utf-8")
    annotations = _annotations(tmp_path)
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    output = tmp_path / "run"
    args = manual_material_cli.build_parser().parse_args(
        [
            "--stp",
            str(stp),
            "--sam3-annotations",
            str(annotations),
            "--output",
            str(output),
            "--config",
            str(config),
            "--visual-inference-mode",
            "bundled",
        ]
    )
    with (
        patch.object(
            manual_material_cli.runtime,
            "configure_runtime",
            return_value={"CONDA_ENV": "hunyuan_sam3d"},
        ),
        patch.object(manual_material_cli, "validate_cad_input_path"),
        patch.object(manual_material_cli, "load_visual_material_config"),
        patch.object(
            manual_material_cli,
            "run_manual_cad_workflow",
            return_value={"completion_state": "COMPLETED"},
        ) as workflow,
    ):
        result = manual_material_cli.run(args)

    call = workflow.call_args.kwargs
    assert call["visual_inference_mode"] == "bundled"
    assert call["visual_foreground_annotations"] is None
    assert call["allow_policy_material_fallback"] is False
    assert call["visual_material_references"][0].startswith("front=")
    assert result["sam3_annotations"] == str(annotations)
    assert result["sam3_annotations_role"] == "reference_manifest_only"


def test_nonempty_output_requires_explicit_resume(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--resume"):
        manual_material_cli._require_fresh_or_resumable_output(
            output,
            resume=False,
        )


def test_resume_is_forwarded_to_the_manual_workflow(tmp_path: Path) -> None:
    stp = tmp_path / "asset.stp"
    stp.write_text("STEP", encoding="utf-8")
    annotations = _annotations(tmp_path)
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    output = tmp_path / "run"
    output.mkdir()
    args = manual_material_cli.build_parser().parse_args(
        [
            "--stp",
            str(stp),
            "--sam3-annotations",
            str(annotations),
            "--output",
            str(output),
            "--config",
            str(config),
            "--resume",
        ]
    )
    with (
        patch.object(manual_material_cli.runtime, "configure_runtime", return_value={}),
        patch.object(manual_material_cli, "validate_cad_input_path"),
        patch.object(manual_material_cli, "load_visual_material_config"),
        patch.object(
            manual_material_cli,
            "run_manual_cad_workflow",
            return_value={"completion_state": "COMPLETED"},
        ) as workflow,
    ):
        manual_material_cli.run(args)

    assert workflow.call_args.kwargs["resume"] is True
