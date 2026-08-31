from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from qwen_material_pipeline import __main__ as entrypoint


@pytest.mark.parametrize(
    ("arguments", "expected_module"),
    [
        (["setup-models"], "qwen_material_pipeline.setup.material_models"),
        (["staged"], "qwen_material_pipeline.workflows.staged_local"),
        (["basic"], "qwen_material_pipeline.workflows.basic"),
        (["catalog"], "qwen_material_pipeline.materials.catalog"),
        (["review"], "qwen_material_pipeline.materials.review"),
        (["complete-plan"], "qwen_material_pipeline.materials.complete_plan"),
        (
            ["annotate-visual-groups"],
            "qwen_material_pipeline.materials.visual_group_annotation",
        ),
        (
            ["policy-exact-cover"],
            "qwen_material_pipeline.materials.policy_exact_cover",
        ),
        (
            ["quality-repair-plan"],
            "qwen_material_pipeline.materials.quality_repair",
        ),
        (
            ["quality-resolution"],
            "qwen_material_pipeline.materials.quality_resolution",
        ),
        (["mvinverse-run"], "qwen_material_pipeline.mvinverse.adapter"),
        (["mvinverse-evidence"], "qwen_material_pipeline.mvinverse.evidence"),
        (["compare"], "qwen_material_pipeline.evidence.reference_compare"),
        (
            ["final-visual-gate"],
            "qwen_material_pipeline.evidence.final_visual_gate",
        ),
        (["usd", "registry"], "qwen_material_pipeline.usd.registry"),
        (["usd", "render"], "qwen_material_pipeline.usd.render"),
        (["usd", "expand"], "qwen_material_pipeline.usd.instances"),
        (["usd", "apply"], "qwen_material_pipeline.usd.apply"),
        (["usd", "apply-instances"], "qwen_material_pipeline.usd.apply_instances"),
        (["usd", "validate"], "qwen_material_pipeline.usd.validate"),
        (
            ["usd", "validate-instances"],
            "qwen_material_pipeline.usd.validate_instances",
        ),
        (["usd", "validate-delivery"], "qwen_material_pipeline.usd.delivery"),
    ],
)
def test_routes_every_command_lazily(monkeypatch, arguments, expected_module):
    imported: list[str] = []

    def fake_import(module_name: str):
        imported.append(module_name)
        return SimpleNamespace(main=lambda: 0)

    monkeypatch.setattr(entrypoint.importlib, "import_module", fake_import)
    monkeypatch.setattr(entrypoint, "_stabilize_staged_ml_runtime", lambda: None)

    assert entrypoint.main(arguments) == 0
    assert imported == [expected_module]


def test_staged_runtime_guard_pins_first_allowed_cpu_and_limits_threads(
    monkeypatch,
):
    affinity_calls: list[tuple[int, set[int]]] = []
    environment: dict[str, str] = {}
    monkeypatch.setattr(entrypoint.os, "environ", environment)
    monkeypatch.setattr(
        entrypoint.os,
        "sched_getaffinity",
        lambda pid: {7, 3, 11},
    )
    monkeypatch.setattr(
        entrypoint.os,
        "sched_setaffinity",
        lambda pid, cpus: affinity_calls.append((pid, set(cpus))),
    )

    entrypoint._stabilize_staged_ml_runtime()

    assert affinity_calls == [(0, {3})]
    assert environment == {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


def test_staged_runtime_guard_can_be_disabled(monkeypatch):
    affinity_calls: list[object] = []
    environment = {"QWEN_MATERIAL_DISABLE_CPU_STABILITY_GUARD": "1"}
    monkeypatch.setattr(entrypoint.os, "environ", environment)
    monkeypatch.setattr(
        entrypoint.os,
        "sched_getaffinity",
        lambda _pid: affinity_calls.append("get"),
    )
    monkeypatch.setattr(
        entrypoint.os,
        "sched_setaffinity",
        lambda _pid, _cpus: affinity_calls.append("set"),
    )

    entrypoint._stabilize_staged_ml_runtime()

    assert affinity_calls == []


def test_forwards_arguments_and_restores_sys_argv(monkeypatch):
    original_argv = sys.argv
    observed: list[str] = []

    def implementation_main():
        observed.extend(sys.argv)
        return 17

    monkeypatch.setattr(
        entrypoint.importlib,
        "import_module",
        lambda _module_name: SimpleNamespace(main=implementation_main),
    )

    result = entrypoint.main(["review", "--review", "decisions.json"])

    assert result == 17
    assert observed == [
        "python -m qwen_material_pipeline review",
        "--review",
        "decisions.json",
    ]
    assert sys.argv is original_argv


@pytest.mark.parametrize(
    "arguments",
    [
        ["--help"],
        ["usd", "--help"],
        ["setup-models", "--help"],
        ["usd", "registry", "--help"],
        ["usd", "validate-delivery", "--help"],
    ],
)
def test_help_never_imports_a_command_module(monkeypatch, capsys, arguments):
    def fail_import(module_name: str):
        raise AssertionError(f"help unexpectedly imported {module_name}")

    monkeypatch.setattr(entrypoint.importlib, "import_module", fail_import)

    assert entrypoint.main(arguments) == 0
    assert "usage:" in capsys.readouterr().out


def test_dispatch_restores_sys_argv_after_command_error(monkeypatch):
    original_argv = sys.argv

    def failing_main():
        raise RuntimeError("command failed")

    monkeypatch.setattr(
        entrypoint.importlib,
        "import_module",
        lambda _module_name: SimpleNamespace(main=failing_main),
    )

    with pytest.raises(RuntimeError, match="command failed"):
        entrypoint.main(["compare", "--input", "report.json"])
    assert sys.argv is original_argv


def test_unknown_usd_command_is_rejected_without_import(monkeypatch):
    monkeypatch.setattr(
        entrypoint.importlib,
        "import_module",
        lambda module_name: pytest.fail(f"unexpected import: {module_name}"),
    )

    with pytest.raises(SystemExit) as error:
        entrypoint.main(["usd", "unknown"])
    assert error.value.code == 2
