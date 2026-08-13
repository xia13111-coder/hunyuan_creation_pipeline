#!/usr/bin/env python3
"""Unified, lazy command entry point for the material pipeline.

The project intentionally spans four different runtimes: the main orchestration
environment, the local Qwen environment, an isolated MVInverse CUDA environment,
and Isaac Sim Python.  Importing every command at startup would cross those
runtime boundaries and make even ``--help`` depend on optional packages such as
``pxr`` or ``transformers``.  This module keeps a small static command table and
imports exactly one implementation, only after the user has selected a command
to execute.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Callable


PROGRAM = "python -m qwen_material_pipeline"
_ML_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class CommandSpec:
    """Lazy route to an existing module-level ``main`` function."""

    module: str
    help: str


COMMANDS: dict[str, CommandSpec] = {
    "staged": CommandSpec(
        "qwen_material_pipeline.workflows.staged_local",
        "run the conservative staged Qwen/MVInverse workflow",
    ),
    "part-id-qwen": CommandSpec(
        "qwen_material_pipeline.workflows.part_id_qwen",
        "rerank independent CAD Part-ID material candidates with local Qwen",
    ),
    "basic": CommandSpec(
        "qwen_material_pipeline.workflows.basic",
        "run catalog search or the basic multi-view Qwen workflow",
    ),
    "catalog": CommandSpec(
        "qwen_material_pipeline.materials.catalog",
        "build or inspect the local NVIDIA MDL material catalog",
    ),
    "base-bank": CommandSpec(
        "qwen_material_pipeline.materials.base_observation_bank",
        "build and verify the strict NVIDIA Base rendered observation bank",
    ),
    "download-qwen": CommandSpec(
        "qwen_material_pipeline.qwen.download_model",
        "download an official Qwen3-VL checkpoint to local storage",
    ),
    "review": CommandSpec(
        "qwen_material_pipeline.materials.review",
        "resolve explicit human review decisions into a material plan",
    ),
    "complete-plan": CommandSpec(
        "qwen_material_pipeline.materials.complete_plan",
        "build a complete plan from reviewed topology and MVInverse evidence",
    ),
    "annotate-visual-groups": CommandSpec(
        "qwen_material_pipeline.materials.visual_group_annotation",
        "recover conservative canonical visual groups in a complete/restored plan",
    ),
    "policy-exact-cover": CommandSpec(
        "qwen_material_pipeline.materials.policy_exact_cover",
        "build an explicitly opted-in, audited best-effort exact-cover plan",
    ),
    "quality-repair-plan": CommandSpec(
        "qwen_material_pipeline.materials.quality_repair",
        "compile one audited material-plan repair from final visual QA",
    ),
    "exact-mdl-tournament": CommandSpec(
        "qwen_material_pipeline.materials.exact_mdl_tournament",
        "select an immutable MDL under a configured render-QA objective",
    ),
    "quality-resolution": CommandSpec(
        "qwen_material_pipeline.materials.quality_resolution",
        "resolve final material QA without hiding geometry or pose limitations",
    ),
    "mvinverse-run": CommandSpec(
        "qwen_material_pipeline.mvinverse.adapter",
        "run or reuse verified offline MVInverse inference",
    ),
    "mvinverse-evidence": CommandSpec(
        "qwen_material_pipeline.mvinverse.evidence",
        "extract and fuse hash-bound MVInverse PBR evidence",
    ),
    "compare": CommandSpec(
        "qwen_material_pipeline.evidence.reference_compare",
        "compare reference photographs with final rendered views",
    ),
    "final-visual-gate": CommandSpec(
        "qwen_material_pipeline.evidence.final_visual_gate",
        "gate completion on an independent collected-USD render",
    ),
    "sam3-foreground-ui": CommandSpec(
        "qwen_material_pipeline.web.sam3_point_selector.app",
        "interactively click and confirm SAM3 whole-workpiece foreground masks",
    ),
    "calibrate-cameras": CommandSpec(
        "qwen_material_pipeline.evidence.camera_calibration",
        "continuously calibrate whole-asset cameras against confirmed silhouettes",
    ),
    "optimize-assembly-pose": CommandSpec(
        "qwen_material_pipeline.evidence.assembly_pose",
        "repair coherent local assembly displacement with bounded rigid motion",
    ),
    "evaluate-assembly-pose-multiview": CommandSpec(
        "qwen_material_pipeline.scripts.evaluate_multiview_assembly_pose",
        "validate one assembly-pose correction across sealed cameras",
    ),
    "appearance-components": CommandSpec(
        "qwen_material_pipeline.evidence.appearance_components",
        "derive conservative photo-supported visual coating components for CAD Part IDs",
    ),
    "appearance-component-inputs": CommandSpec(
        "qwen_material_pipeline.evidence.appearance_component_material",
        "build aggregate retrieval inputs for photo-supported appearance components",
    ),
    "appearance-component-qwen": CommandSpec(
        "qwen_material_pipeline.workflows.appearance_component_qwen",
        "select one immutable Base MDL per photo-supported appearance component",
    ),
}


USD_COMMANDS: dict[str, CommandSpec] = {
    "registry": CommandSpec(
        "qwen_material_pipeline.usd.registry",
        "build a stable part registry for a USD asset",
    ),
    "render": CommandSpec(
        "qwen_material_pipeline.usd.render",
        "render canonical RGB, part-ID, and per-part evidence views",
    ),
    "expand": CommandSpec(
        "qwen_material_pipeline.usd.instances",
        "create a non-destructive editable layer for an instanced assembly",
    ),
    "apply": CommandSpec(
        "qwen_material_pipeline.usd.apply",
        "author validated MDL bindings for a non-instanced asset",
    ),
    "apply-instances": CommandSpec(
        "qwen_material_pipeline.usd.apply_instances",
        "author validated MDL bindings for an instance-heavy assembly",
    ),
    "validate": CommandSpec(
        "qwen_material_pipeline.usd.validate",
        "validate a collected material bundle against its source asset",
    ),
    "validate-instances": CommandSpec(
        "qwen_material_pipeline.usd.validate_instances",
        "validate a collected bundle made from an instanced assembly",
    ),
    "validate-delivery": CommandSpec(
        "qwen_material_pipeline.usd.delivery",
        "validate visual materials after Physics and USD collection",
    ),
}


def _root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=(
            "Material tools for CAD Converter USD intermediates. Use the main "
            "asset pipeline --manual-stp entry for production STEP/STP assets. "
            "Commands are loaded only when executed."
        ),
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, spec in COMMANDS.items():
        commands.add_parser(name, help=spec.help, description=spec.help)
    commands.add_parser(
        "usd",
        help="USD registry, rendering, material binding, and validation tools",
        description="USD registry, rendering, material binding, and validation tools",
    )
    return parser


def _usd_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{PROGRAM} usd",
        description="USD registry, rendering, material binding, and validation tools.",
    )
    commands = parser.add_subparsers(dest="usd_command", metavar="COMMAND")
    for name, spec in USD_COMMANDS.items():
        commands.add_parser(name, help=spec.help, description=spec.help)
    return parser


def _leaf_parser(command_path: str, spec: CommandSpec) -> argparse.ArgumentParser:
    """Return lightweight help without importing the implementation module."""

    parser = argparse.ArgumentParser(
        prog=f"{PROGRAM} {command_path}",
        description=spec.help,
        epilog=(
            f"All non-help arguments are forwarded unchanged to {spec.module}.main()."
        ),
    )
    parser.add_argument(
        "arguments",
        nargs="*",
        metavar="ARG",
        help="arguments accepted by the underlying command",
    )
    return parser


def _implementation_main(module: ModuleType, module_name: str) -> Callable[[], object]:
    implementation = getattr(module, "main", None)
    if not callable(implementation):
        raise RuntimeError(f"Command module has no callable main(): {module_name}")
    return implementation


def _stabilize_staged_ml_runtime() -> None:
    """Keep local ML imports on one deterministic CPU before loading Torch.

    The staged workflow is GPU-bound and starts several independent Python
    runtimes (Qwen, SAM3, MVInverse, SigLIP2 and DINOv2).  On heterogeneous or
    marginally stable desktop CPUs, letting every fresh interpreter migrate
    across all logical CPUs while native libraries initialize can surface as
    unrelated, non-deterministic Python/NumPy/Torch import corruption.  Child
    processes inherit CPU affinity, so pinning the staged owner before any ML
    module is imported gives the entire inference tree one stable control
    core.  Native math thread pools are also bounded; GPU kernels remain
    unaffected.

    Operators can disable only this compatibility guard with
    ``QWEN_MATERIAL_DISABLE_CPU_STABILITY_GUARD=1``.  No pipeline result or
    model evidence is reused or changed by the guard.
    """

    for name in _ML_THREAD_ENVIRONMENT:
        os.environ.setdefault(name, "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if os.environ.get("QWEN_MATERIAL_DISABLE_CPU_STABILITY_GUARD") == "1":
        return
    get_affinity = getattr(os, "sched_getaffinity", None)
    set_affinity = getattr(os, "sched_setaffinity", None)
    if not callable(get_affinity) or not callable(set_affinity):
        return
    try:
        allowed = sorted(get_affinity(0))
        if allowed:
            set_affinity(0, {allowed[0]})
    except OSError:
        # CPU affinity can be unavailable inside a constrained container.
        # Thread limits still provide a safe portable fallback.
        return


def _dispatch(spec: CommandSpec, command_path: str, arguments: Sequence[str]) -> int:
    """Import one command and run it with an isolated forwarded ``sys.argv``."""

    original_argv = sys.argv
    sys.argv = [f"{PROGRAM} {command_path}", *arguments]
    try:
        module = importlib.import_module(spec.module)
        result = _implementation_main(module, spec.module)()
    finally:
        sys.argv = original_argv

    if result is None:
        return 0
    if not isinstance(result, int):
        raise TypeError(
            f"{spec.module}.main() returned {type(result).__name__}, expected int or None"
        )
    return result


def _is_help_request(arguments: Sequence[str]) -> bool:
    return len(arguments) == 1 and arguments[0] in {"-h", "--help"}


def _load_checkout_local_model_environment() -> None:
    """Load the root .env when this tool is used from the complete project."""

    try:
        from asset_pipeline.local_models import configure_offline_model_environment
        from asset_pipeline.runtime import load_project_environment
    except ImportError:
        return
    load_project_environment()
    configure_offline_model_environment()


def main(argv: Sequence[str] | None = None) -> int:
    """Route a unified command without importing unrelated implementations."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    root_parser = _root_parser()
    if not arguments or _is_help_request(arguments):
        root_parser.print_help()
        return 0

    _load_checkout_local_model_environment()

    command = arguments[0]
    if command == "usd":
        usd_arguments = arguments[1:]
        usd_parser = _usd_parser()
        if not usd_arguments or _is_help_request(usd_arguments):
            usd_parser.print_help()
            return 0
        usd_command = usd_arguments[0]
        spec = USD_COMMANDS.get(usd_command)
        if spec is None:
            usd_parser.error(f"invalid command: {usd_command!r}")
        forwarded = usd_arguments[1:]
        command_path = f"usd {usd_command}"
        if _is_help_request(forwarded):
            _leaf_parser(command_path, spec).print_help()
            return 0
        return _dispatch(spec, command_path, forwarded)

    spec = COMMANDS.get(command)
    if spec is None:
        root_parser.error(f"invalid command: {command!r}")
    forwarded = arguments[1:]
    if _is_help_request(forwarded):
        _leaf_parser(command, spec).print_help()
        return 0
    if command == "staged":
        _stabilize_staged_ml_runtime()
    return _dispatch(spec, command, forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
