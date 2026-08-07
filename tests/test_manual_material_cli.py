from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from asset_pipeline import manual_material_cli


def _annotations(root: Path) -> Path:
    views = []
    for view_id in ("front", "iso"):
        image = root / f"{view_id}.png"
        image.write_bytes(b"image")
        views.append({"id": view_id, "image": str(image)})
    path = root / "annotations.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "sam3-human-foreground-annotations/v2",
                "source_views": views,
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
