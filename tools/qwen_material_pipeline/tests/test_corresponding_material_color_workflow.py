from __future__ import annotations

import json
import math
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from qwen_material_pipeline.workflows import (
    corresponding_material_color_workflow as workflow,
)
from qwen_material_pipeline.materials.corresponding_material_color_selection import (
    Candidate,
)


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
    with pytest.raises(
        workflow.CorrespondingMaterialColorWorkflowError, match="duplicate"
    ):
        workflow._normalized_gains([1.0, 1.0])
    with pytest.raises(workflow.CorrespondingMaterialColorWorkflowError, match="0.1"):
        workflow._normalized_gains([1.0, 8.1])


def _workflow_inputs(tmp_path: Path, view_ids: Sequence[str]) -> dict[str, Path]:
    root = tmp_path / "inputs"
    root.mkdir()
    output = {
        name: root / f"{name}.json"
        for name in (
            "source_plan",
            "qwen_choices",
            "part_evidence",
            "spatial",
            "catalog",
            "registry",
            "view_specs",
        )
    }
    for path in output.values():
        _write_json(path, {})
    reference_rows = []
    evidence_observations = []
    for view_id in view_ids:
        image = root / f"{view_id}.jpg"
        image.write_bytes(f"reference:{view_id}".encode("utf-8"))
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        reference_rows.append({"id": view_id, "image": str(image)})
        evidence_observations.append(
            {
                "view_id": view_id,
                "image": str(image),
                "image_sha256": digest,
            }
        )
    _write_json(
        output["part_evidence"],
        {"parts": [{"part_id": "P1", "observations": evidence_observations}]},
    )
    output["reference_manifest"] = root / "reference_manifest.json"
    _write_json(
        output["reference_manifest"],
        {"source_views": reference_rows},
    )
    output["asset"] = root / "asset.usda"
    output["asset"].write_text("#usda 1.0\n", encoding="utf-8")
    output["isaac_python"] = root / "python.sh"
    output["isaac_python"].write_text("#!/bin/sh\n", encoding="utf-8")
    output["material_root"] = root / "materials"
    output["material_root"].mkdir()
    return output


def _fake_runner(commands: list[list[str]], view_ids: Sequence[str]):
    def run(
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
            _write_json(
                Path(_argument(command, "--output-dir"))
                / "part_registry.rendered.json",
                {"render_set": {"views": [{"view_id": value} for value in view_ids]}},
            )
        else:  # pragma: no cover
            raise AssertionError(module)

    return run


def _run_arguments(paths: Mapping[str, Path], destination: Path) -> dict:
    return {
        "source_plan_path": paths["source_plan"],
        "qwen_choices_path": paths["qwen_choices"],
        "part_id_evidence_path": paths["part_evidence"],
        "spatial_mapping_report_path": paths["spatial"],
        "asset_usd_path": paths["asset"],
        "catalog_path": paths["catalog"],
        "registry_path": paths["registry"],
        "material_root_path": paths["material_root"],
        "view_specs_path": paths["view_specs"],
        "reference_manifest_path": paths["reference_manifest"],
        "isaac_python_path": paths["isaac_python"],
        "output_dir": destination,
    }


def test_saved_workflow_runs_candidates_final_render_and_absolute_qa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _workflow_inputs(tmp_path, ["front", "iso"])

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

    def fake_select(arguments: list[str]) -> int:
        _write_json(Path(_argument(arguments, "--output-plan")), {"assignments": []})
        _write_json(Path(_argument(arguments, "--audit")), {"status": "REVIEW"})
        assert arguments.count("--candidate-dir") == 2
        assert _argument(arguments, "--minimum-candidate-count") == "2"
        return 0

    def fake_compare(arguments: list[str]) -> int:
        assert arguments.count("--map") == 2
        _write_json(
            Path(_argument(arguments, "--output")),
            {"aggregate": {"status": "PASS", "material_appearance_score": 0.95}},
        )
        return 0

    monkeypatch.setattr(workflow.color_selection, "main", fake_select)
    monkeypatch.setattr(workflow.reference_compare, "main", fake_compare)
    destination = tmp_path / "result"
    manifest = workflow.run_corresponding_material_color_workflow(
        **_run_arguments(paths, destination),
        gains=[1.0, 2.0],
        command_runner=_fake_runner(commands, ["front", "iso"]),
    )
    assert manifest["workflow_state"] == "COMPLETE"
    assert manifest["local_quality_status"] == "REVIEW"
    assert manifest["policy"]["local_quality_rejection_behavior"] == (
        "retain_best_rendered_candidate_and_continue_with_review"
    )
    assert manifest["policy"]["optimization_mode"] == "fixed_grid"
    assert len(manifest["candidates"]) == 2
    assert manifest["adaptive_controller_rounds"] == []
    assert manifest["adaptive_completion"] is None
    assert len(commands) == 9
    assert (destination / "final_selected" / "material_look.usda").is_file()
    with pytest.raises(
        workflow.CorrespondingMaterialColorWorkflowError, match="already exists"
    ):
        workflow.run_corresponding_material_color_workflow(
            **_run_arguments(paths, destination),
            gains=[1.0, 2.0],
            command_runner=_fake_runner([], ["front", "iso"]),
        )


def test_saved_workflow_adapts_scope_gain_from_registered_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _workflow_inputs(tmp_path, ["front"])
    built_gain_vectors: list[dict[str, float]] = []

    def build_candidate(**kwargs: object) -> tuple[dict, dict]:
        declared = kwargs.get("linear_intensity_gains_by_scope")
        gains = (
            dict(declared)  # type: ignore[arg-type]
            if declared is not None
            else {"PART:P1": float(kwargs["linear_intensity_gain"])}
        )
        built_gain_vectors.append(gains)
        return ({"assignments": [], "gains": gains}, {"scopes": []})

    def fake_load(directory: Path, _source_sha: str) -> Candidate:
        iteration = int(directory.name.rsplit("_", 1)[1])
        return Candidate(
            candidate_id=directory.name,
            gains_by_scope=built_gain_vectors[iteration],
            plan={},
            audit={"scopes": []},
            rendered_registry={},
            paths={},
            hashes={},
        )

    def fake_score(*, candidate: Candidate, **_kwargs: object) -> dict:
        gain = candidate.gains_by_scope["PART:P1"]
        first = candidate.candidate_id == "iteration_00"
        return {
            "PART:P1": {
                "candidate_id": candidate.candidate_id,
                "linear_intensity_gain": gain,
                "reference_relative_luminance": 0.4,
                "render_relative_luminance": 0.2 if first else 0.4,
                "luminance_ratio": 2.0 if first else 1.0,
                "log_luminance_error": math.log(2.0) if first else 0.0,
                "appearance_score": 0.5 if first else 0.9,
            }
        }

    def fake_select(arguments: list[str]) -> int:
        assert arguments.count("--candidate-dir") == 2
        assert _argument(arguments, "--minimum-candidate-count") == "1"
        _write_json(Path(_argument(arguments, "--output-plan")), {"assignments": []})
        _write_json(Path(_argument(arguments, "--audit")), {"status": "PASS"})
        return 0

    def fake_compare(arguments: list[str]) -> int:
        _write_json(
            Path(_argument(arguments, "--output")),
            {"aggregate": {"status": "PASS", "material_appearance_score": 0.9}},
        )
        return 0

    monkeypatch.setattr(
        workflow, "build_corresponding_material_color_plan", build_candidate
    )
    monkeypatch.setattr(
        workflow.color_selection, "load_rendered_color_candidate", fake_load
    )
    monkeypatch.setattr(
        workflow.adaptive_color, "score_adaptive_candidate_scopes", fake_score
    )
    monkeypatch.setattr(workflow.color_selection, "main", fake_select)
    monkeypatch.setattr(workflow.reference_compare, "main", fake_compare)
    commands: list[list[str]] = []
    manifest = workflow.run_corresponding_material_color_workflow(
        **_run_arguments(paths, tmp_path / "adaptive_result"),
        max_adaptive_iterations=4,
        command_runner=_fake_runner(commands, ["front"]),
    )
    assert manifest["policy"]["optimization_mode"] == "adaptive_per_scope"
    assert len(manifest["candidates"]) == 2
    assert len(manifest["adaptive_controller_rounds"]) == 2
    assert manifest["adaptive_completion"] == {
        "executed_iteration_count": 2,
        "maximum_iteration_count": 4,
        "all_scopes_converged": True,
        "remaining_active_scope_count": 0,
        "termination_reason": "all_scopes_converged",
    }
    for candidate in manifest["candidates"]:
        assert Path(candidate["adaptive_iteration_audit"]["path"]).is_file()
    assert built_gain_vectors[0] == {"PART:P1": 1.0}
    assert built_gain_vectors[1]["PART:P1"] == pytest.approx(2.0 ** 0.75)
    assert len(commands) == 9
