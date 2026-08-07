from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from asset_pipeline.visual_materials.orchestrator import (
    _bundled_project_inference_provenance,
    run_final_visual_acceptance_job,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _canonical_sha256(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _config(qwen_python: Path) -> SimpleNamespace:
    return SimpleNamespace(
        qwen_python=qwen_python,
        render_rt_subframes=4,
        final_visual_gate_maximum_score_regression=0.01,
        final_visual_gate_maximum_group_recall_regression=0.01,
        final_visual_gate_maximum_group_share_error_regression=0.01,
        final_visual_gate_minimum_final_appearance_score=0.62,
        final_visual_gate_minimum_final_view_appearance_score=0.55,
        final_visual_gate_minimum_significant_reference_share=0.01,
        final_visual_gate_minimum_significant_evidence_pixels=128,
    )


def _fixture(tmp_path: Path) -> dict:
    isaac = tmp_path / "isaac-python"
    qwen = tmp_path / "qwen-python"
    for executable in (isaac, qwen):
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o755)
    locked = tmp_path / "visual" / "asset_look_locked.usda"
    collected = tmp_path / "final" / "asset_phys.usda"
    locked.parent.mkdir()
    collected.parent.mkdir()
    locked.write_text("locked", encoding="utf-8")
    collected.write_text("collected", encoding="utf-8")
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"reference")
    manifest = tmp_path / "reference_manifest.json"
    _write(
        manifest,
        {
            "schema_version": "qwen-reference-manifest/v1",
            "source_views": [
                {"id": "front_ref", "image": str(reference.resolve())},
                {"id": "side_ref", "image": str(reference.resolve())},
            ],
        },
    )
    baseline_registry = tmp_path / "candidate" / "part_registry.rendered.json"
    render_set = {
        "asset_usd": str(locked.resolve()),
        "resolution": [256, 256],
        "analysis_up_axis": [0.0, 0.0, 1.0],
        "analysis_front_axis": [0.0, -1.0, 0.0],
        "lighting_profile": "material-neutral",
        "rt_subframes": 4,
        "requested_view_tokens": ["right", "front"],
        "views": [],
    }
    _write(
        baseline_registry,
        {
            "schema_version": "qwen-material-parts/v1",
            "asset_usd": str(locked.resolve()),
            "asset_sha256": _sha256(locked),
            "render_set": render_set,
        },
    )
    baseline_quality = tmp_path / "candidate" / "quality.json"
    mapping = {"front_ref": "right", "side_ref": "front"}
    _write(
        baseline_quality,
        {
            "schema_version": "qwen-reference-render-comparison/v1",
            "inputs": {
                "reference_manifest": str(manifest.resolve()),
                "selected_view_mapping": mapping,
            },
            "thresholds": {"minimum_comparable_views": 2},
            "aggregate": {"status": "PASS"},
            "views": [
                {
                    "reference_view_id": reference_id,
                    "render_view_id": render_id,
                    "status": "PASS",
                }
                for reference_id, render_id in mapping.items()
            ],
        },
    )
    result = {
        "state": "APPLIED",
        "effective_usd": str(locked),
        "output_dir": str(locked.parent),
        "config": str(tmp_path / "config.json"),
        "visual_quality_report": str(baseline_quality),
        "visual_quality_rendered_registry": str(baseline_registry),
    }
    return {
        "isaac": isaac,
        "qwen": qwen,
        "locked": locked,
        "collected": collected,
        "manifest": manifest,
        "result": result,
    }


def _fake_runner(
    commands: list[list[str]],
    *,
    reject_collected_gate: bool = False,
    quality_status: str = "PASS",
    first_render_resolution: int | None = None,
    first_compare_mapping: dict[str, str] | None = None,
):
    gate_count = 0
    render_count = 0
    compare_count = 0

    def run(command, **_kwargs):
        nonlocal compare_count, gate_count, render_count
        commands.append(list(command))
        if "final-visual-gate" in command:
            gate_count += 1
            output = Path(command[command.index("--output") + 1])
            accepted = not (reject_collected_gate and gate_count == 2)
            _write(
                output,
                {
                    "schema_version": "qwen-final-visual-gate/v1",
                    "status": "PASS" if accepted else "FAIL_CLOSED",
                    "completion_allowed": accepted,
                    "completion_state": (
                        "COMPLETED" if accepted else "FINAL_VISUAL_QA_FAILED"
                    ),
                },
            )
            return
        if "registry" in command:
            output = Path(command[command.index("--output") + 1])
            asset = Path(command[command.index("--usd") + 1]).resolve()
            _write(
                output,
                {
                    "schema_version": "qwen-material-parts/v1",
                    "asset_usd": str(asset),
                    "asset_sha256": _sha256(asset),
                },
            )
            return
        if "render" in command:
            render_count += 1
            registry = Path(command[command.index("--registry") + 1])
            asset = Path(json.loads(registry.read_text())["asset_usd"])
            output_dir = Path(command[command.index("--output-dir") + 1])
            rendered = output_dir / "part_registry.rendered.json"
            rendered_views: list[dict] = []
            custom_view_specs: str | None = None
            if "--view-specs" in command:
                view_specs = Path(command[command.index("--view-specs") + 1])
                view_specs_document = json.loads(view_specs.read_text())
                requested = [
                    str(row["view_id"])
                    for row in view_specs_document["views"]
                ]
                rendered_views = [
                    {
                        "view_id": row["view_id"],
                        "analysis_direction": row["analysis_direction"],
                        "analysis_camera_up_axis": row["analysis_up_axis"],
                        "focal_length_mm": row["focal_length_mm"],
                        "camera_distance_multiplier": row["distance_multiplier"],
                        "camera_target_offset_u": row["target_offset_u"],
                        "camera_target_offset_v": row["target_offset_v"],
                        "camera_projection_mode": row["projection_mode"],
                        "camera_orthographic_span_multiplier": row[
                            "orthographic_span_multiplier"
                        ],
                        "camera_calibration": row["calibration"],
                    }
                    for row in view_specs_document["views"]
                ]
                custom_view_specs = str(view_specs.resolve())
            else:
                requested = command[command.index("--views") + 1].split(",")
            _write(
                rendered,
                {
                    "schema_version": "qwen-material-parts/v1",
                    "asset_usd": str(asset),
                    "asset_sha256": _sha256(asset),
                    "render_set": {
                        "asset_usd": str(asset),
                        "resolution": [
                            first_render_resolution
                            if render_count == 1
                            and first_render_resolution is not None
                            else int(command[command.index("--resolution") + 1])
                        ]
                        * 2,
                        "analysis_up_axis": [0.0, 0.0, 1.0],
                        "analysis_front_axis": [0.0, -1.0, 0.0],
                        "lighting_profile": command[
                            command.index("--lighting-profile") + 1
                        ],
                        "rt_subframes": int(
                            command[command.index("--rt-subframes") + 1]
                        ),
                        "requested_view_tokens": requested,
                        "custom_view_specs": custom_view_specs,
                        "views": rendered_views,
                    },
                },
            )
            return
        if "compare" in command:
            compare_count += 1
            output = Path(command[command.index("--output") + 1])
            if "--view-map" in command:
                view_map = Path(command[command.index("--view-map") + 1])
                mapping = json.loads(view_map.read_text())["mapping"]
            else:
                mapping = {"front_ref": "right", "side_ref": "front"}
            if compare_count == 1 and first_compare_mapping is not None:
                mapping = first_compare_mapping
            _write(
                output,
                {
                    "schema_version": ("qwen-reference-render-comparison/v1"),
                    "inputs": {
                        "reference_manifest": command[
                            command.index("--reference-manifest") + 1
                        ],
                        "selected_view_mapping": mapping,
                    },
                    "thresholds": {
                        "minimum_comparable_views": int(
                            command[command.index("--minimum-comparable-views") + 1]
                        )
                    },
                    "aggregate": {"status": quality_status},
                    "views": [
                        {
                            "reference_view_id": reference_id,
                            "render_view_id": render_id,
                            "status": quality_status,
                        }
                        for reference_id, render_id in mapping.items()
                    ],
                },
            )

    return run


def _sealed_result(
    tmp_path: Path,
    fixture: dict,
    *,
    requested_inference_mode: str = "bundled",
) -> dict:
    material_output = Path(fixture["locked"]).parent
    analysis = material_output / "analysis" / "project_sealed-fixture"
    source_cad = tmp_path / "source.stp"
    source_cad.write_text("sealed CAD", encoding="utf-8")
    second_reference = tmp_path / "reference-side.png"
    second_reference.write_bytes(b"reference-side")
    references = {
        "front_ref": tmp_path / "reference.png",
        "side_ref": second_reference,
    }
    template = tmp_path / "sealed-project" / "template.json"
    catalog = tmp_path / "sealed-project" / "catalog.json"
    dependency_lock = tmp_path / "sealed-project" / "dependency-lock.json"
    material_root = tmp_path / "materials"
    material_root.mkdir()
    _write(template, {"template": "accepted"})
    _write(catalog, {"catalog": "accepted"})
    _write(dependency_lock, {"dependencies": "accepted"})
    method = "sealed_fixture_library_default_mdl_result"
    acceptance_contract = {
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
    }
    project = tmp_path / "sealed-project" / "project.json"
    _write(
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
                {"role": reference_id, "sha256": _sha256(path)}
                for reference_id, path in references.items()
            ],
            "acceptance": acceptance_contract,
            "evidence": {
                "method": method,
                "historical_result_sha256": "1" * 64,
            },
        },
    )
    plan = {
        "schema_version": "qwen-autonomous-material-plan/v1",
        "provenance": {
            "project_sha256": _sha256(project),
            "template_sha256": _sha256(template),
            "historical_result_sha256": "1" * 64,
            "source_cad_sha256": _sha256(source_cad),
            "reference_sha256": {
                reference_id: _sha256(path)
                for reference_id, path in references.items()
            },
        },
        "assignments": [{"part_id": "P0001", "status": "approved"}],
    }
    plan_path = analysis / "complete_material_plan.json"
    _write(plan_path, plan)
    plan_sha256 = _canonical_sha256(plan)
    audit = {
        "schema_version": "qwen-bundled-material-project-audit/v1",
        "status": "PASS",
        "asset_id": "sealed-fixture",
        "method": method,
        "project": str(project),
        "historical_result_sha256": "1" * 64,
        "part_count": 1,
        "complete_coverage": True,
        "topology_verified": True,
        "face_subsets_verified": True,
        "plan_sha256": plan_sha256,
    }
    audit_path = analysis / "project_material_audit.json"
    _write(audit_path, audit)
    dependency_verification = {
        "schema_version": "qwen-sealed-material-dependency-verification/v1",
        "status": "PASS",
        "dependency_lock_verified": True,
        "lock_path": str(dependency_lock),
        "lock_sha256": _sha256(dependency_lock),
        "catalog_path": str(catalog),
        "material_root": str(material_root),
        "isaac_root": str(Path(fixture["isaac"]).parent),
        "historical_parameter_policy": "immutable library defaults",
    }
    dependency_verification_report = analysis / "sealed_dependency_verification.json"
    _write(dependency_verification_report, dependency_verification)
    evidence = {
        "schema_version": "qwen-bundled-project-evidence/v1",
        "asset_id": "sealed-fixture",
        "method": method,
        "historical_result_sha256": "1" * 64,
        "project": str(project),
        "project_sha256": _sha256(project),
        "source_cad_sha256": _sha256(source_cad),
        "reference_sha256": {
            reference_id: _sha256(path) for reference_id, path in references.items()
        },
        "template": str(template),
        "template_sha256": _sha256(template),
        "catalog": str(catalog),
        "catalog_sha256": _sha256(catalog),
        "dependency_lock_verified": True,
        "dependency_lock": str(dependency_lock),
        "dependency_lock_sha256": _sha256(dependency_lock),
        "dependency_lock_verification": dependency_verification,
        "dependency_lock_verification_status": "PASS",
        "dependency_lock_verification_report": str(dependency_verification_report),
        "dependency_lock_verification_report_sha256": _sha256(
            dependency_verification_report
        ),
        "plan_sha256": plan_sha256,
        "audit_sha256": _canonical_sha256(audit),
        "live_inference_repeated": False,
        "replay_policy": "hash-bound exact sealed-project replay",
    }
    evidence_path = analysis / "sealed_qwen_mvinverse_evidence.json"
    _write(evidence_path, evidence)

    result = dict(fixture["result"])
    result.pop("visual_quality_report")
    result.update(
        {
            "source_cad": str(source_cad),
            "source_cad_sha256": _sha256(source_cad),
            "references": [
                {
                    "id": reference_id,
                    "image": str(path),
                    "sha256": _sha256(path),
                }
                for reference_id, path in references.items()
            ],
            "requested_inference_mode": requested_inference_mode,
            "inference_mode": "bundled_project",
            "visual_quality_status": "RESTORED_HISTORICAL_BASELINE",
            "staged_state": "READY_TO_APPLY",
            "material_project": "sealed-fixture",
            "material_project_manifest": str(project),
            "material_project_manifest_sha256": _sha256(project),
            "material_project_acceptance": acceptance_contract,
            "material_project_acceptance_sha256": _canonical_sha256(
                acceptance_contract
            ),
            "material_plan": str(plan_path),
            "project_material_audit": str(audit_path),
            "sealed_qwen_mvinverse_evidence": str(evidence_path),
            "catalog": str(catalog),
            "material_root": str(material_root),
            "dependency_lock_verified": True,
            "dependency_lock": str(dependency_lock),
            "dependency_lock_sha256": _sha256(dependency_lock),
            "dependency_lock_verification_status": "PASS",
            "dependency_lock_verification_report": str(dependency_verification_report),
            "dependency_lock_verification_report_sha256": _sha256(
                dependency_verification_report
            ),
            "assignment_count": 1,
            "applied_count": 1,
        }
    )
    return result


@pytest.mark.parametrize("requested_inference_mode", ["auto", "bundled"])
def test_bundled_result_preserves_actual_requested_mode(
    requested_inference_mode: str,
) -> None:
    assert _bundled_project_inference_provenance(requested_inference_mode) == {
        "requested_inference_mode": requested_inference_mode,
        "inference_mode": "bundled_project",
    }


def test_live_cannot_be_recorded_as_bundled_requested_mode() -> None:
    with pytest.raises(RuntimeError, match="requires requested mode auto or bundled"):
        _bundled_project_inference_provenance("live")


def test_runs_locked_and_collected_independent_visual_rounds(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    commands: list[list[str]] = []
    messages: list[str] = []

    result = run_final_visual_acceptance_job(
        collected_usd=str(fixture["collected"]),
        visual_material_result=fixture["result"],
        log_cb=messages.append,
        _command_runner=_fake_runner(commands),
        _isaac_python_resolver=lambda: fixture["isaac"],
        _config_loader=lambda _path: _config(fixture["qwen"]),
    )

    assert result["state"] == "COMPLETED"
    assert result["completion_allowed"] is True
    assert result["locked_visual_gate_status"] == "PASS"
    assert result["collected_visual_gate_status"] == "PASS"
    assert [
        (
            "gate"
            if "final-visual-gate" in command
            else "registry"
            if "registry" in command
            else "render"
            if "render" in command
            else "compare"
        )
        for command in commands
    ] == [
        "registry",
        "render",
        "compare",
        "gate",
        "registry",
        "render",
        "compare",
        "gate",
    ]
    assert "--allow-same-baseline-asset" in commands[3]
    assert "--allow-same-baseline-asset" not in commands[7]
    progress_starts = [
        message
        for message in messages
        if message.startswith("[PROGRESS]") and " START " in message
    ]
    for stage in (
        "final_locked_visual_registry",
        "final_locked_visual_render",
        "final_locked_visual_compare",
        "final_locked_visual_gate",
        "final_collected_visual_registry",
        "final_collected_visual_render",
        "final_collected_visual_compare",
        "final_collected_visual_gate",
    ):
        assert any(
            f"visual_materials/{stage}" in message for message in progress_starts
        )


def test_replays_continuous_camera_specs_in_both_final_rounds(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    baseline_registry = Path(
        fixture["result"]["visual_quality_rendered_registry"]
    )
    registry_document = json.loads(baseline_registry.read_text())
    registered_views = [
        {
            "view_id": "front",
            "analysis_direction": [1.0, 0.0, 0.0],
            "analysis_camera_up_axis": [0.0, 0.0, 1.0],
            "focal_length_mm": 220.0,
            "camera_distance_multiplier": 12.0,
            "camera_target_offset_u": 0.01,
            "camera_target_offset_v": -0.02,
            "camera_projection_mode": "perspective",
            "camera_orthographic_span_multiplier": 2.0,
            "camera_calibration": {
                "reference_view_id": "front",
                "phase": "sealed_full_resolution_finalist",
            },
        },
        {
            "view_id": "side",
            "analysis_direction": [0.0, 1.0, 0.0],
            "analysis_camera_up_axis": [0.0, 0.0, 1.0],
            "focal_length_mm": 245.0,
            "camera_distance_multiplier": 14.0,
            "camera_target_offset_u": -0.03,
            "camera_target_offset_v": 0.04,
            "camera_projection_mode": "perspective",
            "camera_orthographic_span_multiplier": 2.0,
            "camera_calibration": {
                "reference_view_id": "side",
                "phase": "sealed_full_resolution_finalist",
            },
        },
    ]
    registry_document["render_set"]["requested_view_tokens"] = [
        "front",
        "side",
    ]
    registry_document["render_set"]["views"] = registered_views
    _write(baseline_registry, registry_document)

    baseline_quality = Path(fixture["result"]["visual_quality_report"])
    quality_document = json.loads(baseline_quality.read_text())
    mapping = {"front_ref": "front", "side_ref": "side"}
    quality_document["inputs"]["selected_view_mapping"] = mapping
    quality_document["views"] = [
        {
            "reference_view_id": reference_id,
            "render_view_id": render_id,
            "status": "PASS",
        }
        for reference_id, render_id in mapping.items()
    ]
    _write(baseline_quality, quality_document)

    commands: list[list[str]] = []
    result = run_final_visual_acceptance_job(
        collected_usd=str(fixture["collected"]),
        visual_material_result=fixture["result"],
        _command_runner=_fake_runner(commands),
        _isaac_python_resolver=lambda: fixture["isaac"],
        _config_loader=lambda _path: _config(fixture["qwen"]),
    )

    assert result["state"] == "COMPLETED"
    render_commands = [command for command in commands if "render" in command]
    assert len(render_commands) == 2
    assert all("--view-specs" in command for command in render_commands)
    assert all("--views" not in command for command in render_commands)
    for command in render_commands:
        view_specs = Path(command[command.index("--view-specs") + 1])
        document = json.loads(view_specs.read_text())
        assert [row["view_id"] for row in document["views"]] == [
            "front",
            "side",
        ]
        side = document["views"][1]
        assert side["focal_length_mm"] == 245.0
        assert side["distance_multiplier"] == 14.0


def test_collected_gate_failure_prevents_completed_state(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    commands: list[list[str]] = []

    with pytest.raises(RuntimeError, match="rejected unattended completion"):
        run_final_visual_acceptance_job(
            collected_usd=str(fixture["collected"]),
            visual_material_result=fixture["result"],
            _command_runner=_fake_runner(
                commands,
                reject_collected_gate=True,
            ),
            _isaac_python_resolver=lambda: fixture["isaac"],
            _config_loader=lambda _path: _config(fixture["qwen"]),
        )

    gate_report = (
        Path(fixture["locked"]).parent
        / "final_visual_acceptance"
        / "collected_visual_gate.json"
    )
    assert json.loads(gate_report.read_text())["completion_allowed"] is False
    assert len([command for command in commands if "final-visual-gate" in command]) == 2


def test_legacy_result_establishes_a_fresh_locked_baseline(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    second_reference = tmp_path / "reference-side.png"
    second_reference.write_bytes(b"reference-side")
    result = dict(fixture["result"])
    result.pop("visual_quality_report")
    result["references"] = [
        {"id": "front_ref", "image": str(tmp_path / "reference.png")},
        {"id": "side_ref", "image": str(second_reference)},
    ]
    commands: list[list[str]] = []

    acceptance = run_final_visual_acceptance_job(
        collected_usd=str(fixture["collected"]),
        visual_material_result=result,
        _command_runner=_fake_runner(commands),
        _isaac_python_resolver=lambda: fixture["isaac"],
        _config_loader=lambda _path: _config(fixture["qwen"]),
    )

    assert acceptance["state"] == "COMPLETED"
    assert acceptance["selection_quality_report"] is None
    assert acceptance["locked_visual_gate"] is None
    assert acceptance["locked_visual_gate_status"] == "ESTABLISHED_INDEPENDENT_BASELINE"
    assert len([command for command in commands if "final-visual-gate" in command]) == 1


@pytest.mark.parametrize("requested_inference_mode", ["auto", "bundled"])
def test_sealed_historical_result_uses_no_regression_acceptance(
    tmp_path: Path,
    requested_inference_mode: str,
) -> None:
    fixture = _fixture(tmp_path)
    result = _sealed_result(
        tmp_path,
        fixture,
        requested_inference_mode=requested_inference_mode,
    )
    commands: list[list[str]] = []
    drifted_config = _config(fixture["qwen"])
    drifted_config.render_rt_subframes = 19

    acceptance = run_final_visual_acceptance_job(
        collected_usd=str(fixture["collected"]),
        visual_material_result=result,
        _command_runner=_fake_runner(commands, quality_status="PASS"),
        _isaac_python_resolver=lambda: fixture["isaac"],
        _config_loader=lambda _path: drifted_config,
        _sealed_dependency_verifier=lambda **_kwargs: json.loads(
            Path(result["dependency_lock_verification_report"]).read_text(
                encoding="utf-8"
            )
        ),
    )

    assert acceptance["state"] == "COMPLETED"
    assert acceptance["acceptance_mode"] == "SEALED_BASELINE_PRESERVATION"
    assert (
        acceptance["sealed_baseline_evidence"]
        == result["sealed_qwen_mvinverse_evidence"]
    )
    gate_commands = [command for command in commands if "final-visual-gate" in command]
    assert len(gate_commands) == 1
    assert "--sealed-baseline-preservation-evidence" in gate_commands[0]
    evidence_index = gate_commands[0].index("--sealed-baseline-preservation-evidence")
    assert (
        gate_commands[0][evidence_index + 1] == result["sealed_qwen_mvinverse_evidence"]
    )
    render_commands = [command for command in commands if "render" in command]
    assert len(render_commands) == 2
    for command in render_commands:
        assert command[command.index("--resolution") + 1] == "512"
        assert command[command.index("--views") + 1] == "right,front"
        assert command[command.index("--rt-subframes") + 1] == "4"
        assert command[command.index("--lighting-profile") + 1] == (
            "material-neutral"
        )
    compare_commands = [command for command in commands if "compare" in command]
    assert len(compare_commands) == 2
    assert all("--view-map" in command for command in compare_commands)
    first_view_map = Path(
        compare_commands[0][compare_commands[0].index("--view-map") + 1]
    )
    assert json.loads(first_view_map.read_text(encoding="utf-8"))["mapping"] == {
        "front_ref": "right",
        "side_ref": "front",
    }


def test_sealed_historical_result_requires_absolute_four_view_pass(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    result = _sealed_result(tmp_path, fixture)
    commands: list[list[str]] = []

    with pytest.raises(RuntimeError, match="did not PASS every view"):
        run_final_visual_acceptance_job(
            collected_usd=str(fixture["collected"]),
            visual_material_result=result,
            _command_runner=_fake_runner(commands, quality_status="FAIL"),
            _isaac_python_resolver=lambda: fixture["isaac"],
            _config_loader=lambda _path: _config(fixture["qwen"]),
            _sealed_dependency_verifier=lambda **_kwargs: json.loads(
                Path(result["dependency_lock_verification_report"]).read_text(
                    encoding="utf-8"
                )
            ),
        )

    assert not any("final-visual-gate" in command for command in commands)


def test_sealed_historical_result_rejects_injected_selection_quality(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    result = _sealed_result(tmp_path, fixture)
    result["visual_quality_report"] = fixture["result"][
        "visual_quality_report"
    ]
    commands: list[list[str]] = []

    with pytest.raises(RuntimeError, match="cannot override"):
        run_final_visual_acceptance_job(
            collected_usd=str(fixture["collected"]),
            visual_material_result=result,
            _command_runner=_fake_runner(commands),
            _isaac_python_resolver=lambda: fixture["isaac"],
            _config_loader=lambda _path: _config(fixture["qwen"]),
            _sealed_dependency_verifier=lambda **_kwargs: json.loads(
                Path(result["dependency_lock_verification_report"]).read_text(
                    encoding="utf-8"
                )
            ),
        )

    assert commands == []


@pytest.mark.parametrize("tamper", ["render", "mapping"])
def test_sealed_historical_locked_round_must_match_project_contract(
    tmp_path: Path,
    tamper: str,
) -> None:
    fixture = _fixture(tmp_path)
    result = _sealed_result(tmp_path, fixture)
    commands: list[list[str]] = []
    runner = _fake_runner(
        commands,
        first_render_resolution=256 if tamper == "render" else None,
        first_compare_mapping=(
            {"front_ref": "front", "side_ref": "right"}
            if tamper == "mapping"
            else None
        ),
    )

    with pytest.raises(RuntimeError, match="hash-bound project acceptance"):
        run_final_visual_acceptance_job(
            collected_usd=str(fixture["collected"]),
            visual_material_result=result,
            _command_runner=runner,
            _isaac_python_resolver=lambda: fixture["isaac"],
            _config_loader=lambda _path: _config(fixture["qwen"]),
            _sealed_dependency_verifier=lambda **_kwargs: json.loads(
                Path(result["dependency_lock_verification_report"]).read_text(
                    encoding="utf-8"
                )
            ),
        )

    assert not any("final-visual-gate" in command for command in commands)


def test_live_non_pass_result_cannot_use_baseline_preservation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    second_reference = tmp_path / "reference-side.png"
    second_reference.write_bytes(b"reference-side")
    result = dict(fixture["result"])
    result.pop("visual_quality_report")
    result["references"] = [
        {"id": "front_ref", "image": str(tmp_path / "reference.png")},
        {"id": "side_ref", "image": str(second_reference)},
    ]
    commands: list[list[str]] = []

    with pytest.raises(RuntimeError, match="did not PASS every view"):
        run_final_visual_acceptance_job(
            collected_usd=str(fixture["collected"]),
            visual_material_result=result,
            _command_runner=_fake_runner(commands, quality_status="FAIL"),
            _isaac_python_resolver=lambda: fixture["isaac"],
            _config_loader=lambda _path: _config(fixture["qwen"]),
        )

    assert not any("final-visual-gate" in command for command in commands)


def test_sealed_result_method_must_match_hash_bound_project(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    result = _sealed_result(tmp_path, fixture)
    evidence_path = Path(result["sealed_qwen_mvinverse_evidence"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["method"] = "forged_sealed_method"
    _write(evidence_path, evidence)
    commands: list[list[str]] = []

    with pytest.raises(RuntimeError, match="invalid project binding"):
        run_final_visual_acceptance_job(
            collected_usd=str(fixture["collected"]),
            visual_material_result=result,
            _command_runner=_fake_runner(commands),
            _isaac_python_resolver=lambda: fixture["isaac"],
            _config_loader=lambda _path: _config(fixture["qwen"]),
        )

    assert commands == []


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "tampered"])
def test_sealed_result_acceptance_must_match_hash_bound_project(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _fixture(tmp_path)
    result = _sealed_result(tmp_path, fixture)
    if mutation == "missing":
        result.pop("material_project_acceptance")
    else:
        contract = json.loads(
            json.dumps(result["material_project_acceptance"])
        )
        contract["view_mapping"]["side_ref"] = (
            "right" if mutation == "duplicate" else "tampered_pose"
        )
        result["material_project_acceptance"] = contract
        result["material_project_acceptance_sha256"] = _canonical_sha256(
            contract
        )
    commands: list[list[str]] = []

    with pytest.raises(RuntimeError, match="hash-bound project"):
        run_final_visual_acceptance_job(
            collected_usd=str(fixture["collected"]),
            visual_material_result=result,
            _command_runner=_fake_runner(commands),
            _isaac_python_resolver=lambda: fixture["isaac"],
            _config_loader=lambda _path: _config(fixture["qwen"]),
        )

    assert commands == []


def test_live_result_cannot_claim_sealed_historical_status(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    result = _sealed_result(tmp_path, fixture)
    result["inference_mode"] = "qwen_mvinverse"
    commands: list[list[str]] = []

    with pytest.raises(
        RuntimeError,
        match="requires both inference_mode='bundled_project'",
    ):
        run_final_visual_acceptance_job(
            collected_usd=str(fixture["collected"]),
            visual_material_result=result,
            _command_runner=_fake_runner(commands),
            _isaac_python_resolver=lambda: fixture["isaac"],
            _config_loader=lambda _path: _config(fixture["qwen"]),
        )

    assert commands == []


def test_stale_dependency_lock_cannot_authorize_preservation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    result = _sealed_result(tmp_path, fixture)
    Path(result["dependency_lock"]).write_text(
        '{"dependencies":"changed"}',
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    with pytest.raises(RuntimeError, match="dependency_lock hash is stale"):
        run_final_visual_acceptance_job(
            collected_usd=str(fixture["collected"]),
            visual_material_result=result,
            _command_runner=_fake_runner(commands),
            _isaac_python_resolver=lambda: fixture["isaac"],
            _config_loader=lambda _path: _config(fixture["qwen"]),
            _sealed_dependency_verifier=lambda **_kwargs: {},
        )

    assert commands == []
