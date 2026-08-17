from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from qwen_material_pipeline.workflows import corresponding_material_color_workflow as workflow


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_gain_policy_is_bounded_unique_and_requires_competition() -> None:
    assert workflow._normalized_gains([0.7, 1, 8]) == (0.7, 1.0, 8.0)
    assert workflow._candidate_id(1.4) == "gain_1_40"
    with pytest.raises(workflow.CorrespondingMaterialColorWorkflowError, match="two"):
        workflow._normalized_gains([1.0])
    with pytest.raises(workflow.CorrespondingMaterialColorWorkflowError, match="duplicate"):
        workflow._normalized_gains([1.0, 1.0])
    with pytest.raises(workflow.CorrespondingMaterialColorWorkflowError, match="0.1"):
        workflow._normalized_gains([1.0, 8.1])


def test_saved_workflow_runs_candidates_final_render_and_absolute_qa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source_plan = inputs / "source_plan.json"
    qwen_choices = inputs / "qwen_choices.json"
    part_evidence = inputs / "part_evidence.json"
    spatial = inputs / "spatial.json"
    catalog = inputs / "catalog.json"
    registry = inputs / "registry.json"
    view_specs = inputs / "view_specs.json"
    reference_manifest = inputs / "reference_manifest.json"
    for path in (
        source_plan,
        qwen_choices,
        part_evidence,
        spatial,
        catalog,
        registry,
        view_specs,
    ):
        _write_json(path, {})
    _write_json(
        reference_manifest,
        {"source_views": [{"id": "front"}, {"id": "iso"}]},
    )
    asset = inputs / "asset.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    isaac_python = inputs / "python.sh"
    isaac_python.write_text("#!/bin/sh\n", encoding="utf-8")
    material_root = inputs / "materials"
    material_root.mkdir()

    def build_candidate(**kwargs: object) -> tuple[dict, dict]:
        gain = float(kwargs["linear_intensity_gain"])
        return (
            {"assignments": [], "candidate_gain": gain},
            {"summary": {}, "candidate_gain": gain},
        )

    monkeypatch.setattr(
        workflow, "build_corresponding_material_color_plan", build_candidate
    )
    commands: list[list[str]] = []

    def fake_runner(
        raw_command: Sequence[str],
        log_path: Path,
        _environment: Mapping[str, str],
        _timeout: int,
    ) -> None:
        command = [str(value) for value in raw_command]
        commands.append(command)
        log_path.write_text("ok\n", encoding="utf-8")
        module = command[2]
        if module.endswith("usd.apply"):
            output = Path(_argument(command, "--output"))
            output.write_text("#usda 1.0\n", encoding="utf-8")
            _write_json(Path(_argument(command, "--report")), {"status": "PASS"})
        elif module.endswith("usd.registry"):
            _write_json(Path(_argument(command, "--output")), {"part_count": 1})
        elif module.endswith("usd.render"):
            output_dir = Path(_argument(command, "--output-dir"))
            _write_json(
                output_dir / "part_registry.rendered.json",
                {
                    "render_set": {
                        "views": [{"view_id": "front"}, {"view_id": "iso"}]
                    }
                },
            )
        else:  # pragma: no cover - protects the test fixture itself
            raise AssertionError(module)

    def fake_select(arguments: list[str]) -> int:
        _write_json(
            Path(_argument(arguments, "--output-plan")),
            {"assignments": [], "selected": True},
        )
        _write_json(Path(_argument(arguments, "--audit")), {"status": "PASS"})
        assert arguments.count("--candidate-dir") == 2
        return 0

    def fake_compare(arguments: list[str]) -> int:
        assert arguments.count("--map") == 2
        _write_json(
            Path(_argument(arguments, "--output")),
            {
                "aggregate": {
                    "status": "PASS",
                    "material_appearance_score": 0.95,
                    "passed_view_count": 2,
                }
            },
        )
        return 0

    monkeypatch.setattr(workflow.color_selection, "main", fake_select)
    monkeypatch.setattr(workflow.reference_compare, "main", fake_compare)
    destination = tmp_path / "result"
    manifest = workflow.run_corresponding_material_color_workflow(
        source_plan_path=source_plan,
        qwen_choices_path=qwen_choices,
        part_id_evidence_path=part_evidence,
        spatial_mapping_report_path=spatial,
        asset_usd_path=asset,
        catalog_path=catalog,
        registry_path=registry,
        material_root_path=material_root,
        view_specs_path=view_specs,
        reference_manifest_path=reference_manifest,
        isaac_python_path=isaac_python,
        output_dir=destination,
        gains=[1.0, 2.0],
        command_runner=fake_runner,
    )
    assert manifest["workflow_state"] == "COMPLETE"
    assert manifest["quality_status"] == "PASS"
    assert len(manifest["candidates"]) == 2
    assert len(commands) == 9
    apply_commands = [command for command in commands if command[2].endswith("usd.apply")]
    assert len(apply_commands) == 3
    assert all("--include-review" in command for command in apply_commands)
    assert all("--include-policy-fallback" in command for command in apply_commands)
    assert (destination / "workflow_manifest.json").is_file()
    assert (destination / "final_selected" / "material_look.usda").is_file()
    assert (
        destination
        / "final_selected"
        / "renders"
        / "part_registry.rendered.json"
    ).is_file()
    with pytest.raises(
        workflow.CorrespondingMaterialColorWorkflowError,
        match="already exists",
    ):
        workflow.run_corresponding_material_color_workflow(
            source_plan_path=source_plan,
            qwen_choices_path=qwen_choices,
            part_id_evidence_path=part_evidence,
            spatial_mapping_report_path=spatial,
            asset_usd_path=asset,
            catalog_path=catalog,
            registry_path=registry,
            material_root_path=material_root,
            view_specs_path=view_specs,
            reference_manifest_path=reference_manifest,
            isaac_python_path=isaac_python,
            output_dir=destination,
            gains=[1.0, 2.0],
            command_runner=fake_runner,
        )
