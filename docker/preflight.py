#!/usr/bin/env python3
"""Validate the host side of the complete Docker deployment."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = Path(__file__).with_name(".env.runtime")
COMPOSE_FILE = Path(__file__).with_name("compose.full.yaml")
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

REQUIRED_DIRECTORIES = (
    "PROJECT_ROOT",
    "ASSET_ROOT",
    "ISAAC_CACHE_ROOT",
    "MODEL_CACHE_ROOT",
    "QWEN35_RUNTIME_DIR",
    "QWEN35_MODEL_PATH",
    "MVINVERSE_REPOSITORY",
    "MVINVERSE_CHECKPOINT",
    "SAM3_REPOSITORY",
    "SAM3D_SINGLE_VIEW_ROOT",
    "SAM3D_MULTI_VIEW_ROOT",
    "SAM3D_DINOV2_REPOSITORY",
    "ENTITYSEG_RUNTIME_DIR",
    "ENTITYSEG_BASE_RUNTIME_DIR",
    "ENTITYSEG_DETECTRON2_ROOT",
    "ENTITYSEG_CROPFORMER_ROOT",
    "SIGLIP2_MODEL_PATH",
    "DINOV2_MODEL_PATH",
    "VISUAL_MATERIAL_ROOT",
    "NVIDIA_BASE_OBSERVATION_BANK",
    "VISUAL_RETRIEVAL_CACHE",
)
REQUIRED_FILES = (
    "QWEN35_PYTHON",
    "SAM3_CHECKPOINT",
    "SAM3D_PIPELINE_CONFIG",
    "SAM3D_MOGE_CHECKPOINT",
    "SAM3D_DINOV2_CHECKPOINT",
    "ENTITYSEG_PYTHON",
    "ENTITYSEG_CONFIG",
    "ENTITYSEG_CHECKPOINT",
)


class HostPreflightError(RuntimeError):
    """Raised for one actionable deployment defect."""


def _parse_value(raw: str, *, path: Path, line_number: int) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value[0] in {"'", '"'}:
        if len(value) < 2 or value[-1] != value[0]:
            raise HostPreflightError(f"unterminated quote in {path}:{line_number}")
        return value[1:-1]
    marker = value.find(" #")
    return value[:marker].rstrip() if marker >= 0 else value


def load_environment(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise HostPreflightError(
            f"runtime environment is missing: {path}; copy docker/env.runtime.example"
        )
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise HostPreflightError(f"invalid assignment in {path}:{line_number}")
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if NAME_RE.fullmatch(name) is None:
            raise HostPreflightError(f"invalid variable name in {path}:{line_number}")
        values[name] = _parse_value(raw_value, path=path, line_number=line_number)
    for name, value in os.environ.items():
        if name in values and value:
            values[name] = value
    return values


def _value(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "").strip()
    if not value:
        raise HostPreflightError(f"{name} is not configured")
    return value


def _path(environment: Mapping[str, str], name: str) -> Path:
    path = Path(_value(environment, name)).expanduser()
    if not path.is_absolute():
        raise HostPreflightError(f"{name} must be an absolute host path: {path}")
    return path


def _run(command: list[str], *, label: str, environment: Mapping[str, str]) -> None:
    merged = os.environ.copy()
    merged.update({name: value for name, value in environment.items() if value})
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            env=merged,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HostPreflightError(f"{label} could not run: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise HostPreflightError(f"{label} failed: {detail or completed.returncode}")


def _check_layout(environment: Mapping[str, str], *, minimum_free_gb: float) -> None:
    for name in REQUIRED_DIRECTORIES:
        path = _path(environment, name)
        if not path.is_dir():
            raise HostPreflightError(f"{name} directory is missing: {path}")
    for name in REQUIRED_FILES:
        path = _path(environment, name)
        if not path.is_file():
            raise HostPreflightError(f"{name} file is missing: {path}")
    for name in ("QWEN35_PYTHON", "ENTITYSEG_PYTHON"):
        path = _path(environment, name)
        if not os.access(path, os.X_OK):
            raise HostPreflightError(f"{name} is not executable: {path}")

    entityseg_python = _path(environment, "ENTITYSEG_PYTHON")
    resolved_entityseg_python = entityseg_python.resolve()
    base_runtime = _path(environment, "ENTITYSEG_BASE_RUNTIME_DIR").resolve()
    try:
        resolved_entityseg_python.relative_to(base_runtime)
    except ValueError:
        if resolved_entityseg_python != entityseg_python:
            raise HostPreflightError(
                "ENTITYSEG_BASE_RUNTIME_DIR must contain the resolved EntitySeg "
                f"interpreter: {resolved_entityseg_python}"
            )
    detectron2_root = _path(environment, "ENTITYSEG_DETECTRON2_ROOT")
    if not (detectron2_root / "detectron2" / "__init__.py").is_file():
        raise HostPreflightError(
            "ENTITYSEG_DETECTRON2_ROOT does not contain the detectron2 package: "
            f"{detectron2_root}"
        )

    project = _path(environment, "PROJECT_ROOT")
    if project.resolve() != ROOT:
        raise HostPreflightError(
            f"PROJECT_ROOT must point to this checkout: expected {ROOT}, got {project}"
        )
    for name in ("QWEN35_MODEL_PATH", "SIGLIP2_MODEL_PATH", "DINOV2_MODEL_PATH"):
        path = _path(environment, name)
        if not (path / "config.json").is_file():
            raise HostPreflightError(f"{name} has no config.json: {path}")
    if not (_path(environment, "MVINVERSE_CHECKPOINT") / "config.json").is_file():
        raise HostPreflightError("MVINVERSE_CHECKPOINT has no config.json")
    material_root = _path(environment, "VISUAL_MATERIAL_ROOT")
    base_root = material_root if material_root.name == "Base" else material_root / "Base"
    if not base_root.is_dir() or next(base_root.rglob("*.mdl"), None) is None:
        raise HostPreflightError(
            f"VISUAL_MATERIAL_ROOT has no NVIDIA Base MDL collection: {material_root}"
        )
    bank = _path(environment, "NVIDIA_BASE_OBSERVATION_BANK")
    if not (bank / "index_manifest.json").is_file():
        raise HostPreflightError(
            f"NVIDIA_BASE_OBSERVATION_BANK has no index_manifest.json: {bank}"
        )
    asset_root = _path(environment, "ASSET_ROOT")
    free_gb = shutil.disk_usage(asset_root).free / 1024**3
    if free_gb < minimum_free_gb:
        raise HostPreflightError(
            f"ASSET_ROOT has only {free_gb:.1f} GiB free; "
            f"at least {minimum_free_gb:g} GiB is required"
        )


def _check_docker(
    environment: Mapping[str, str],
    *,
    env_file: Path,
    skip_image: bool,
) -> None:
    if shutil.which("docker") is None:
        raise HostPreflightError("docker is not installed")
    _run(["docker", "info"], label="Docker daemon", environment=environment)
    info = subprocess.run(
        ["docker", "info", "--format", "{{json .Runtimes}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if info.returncode != 0 or '"nvidia"' not in info.stdout:
        raise HostPreflightError("Docker NVIDIA runtime is unavailable")
    _run(
        [
            "docker",
            "compose",
            "--env-file",
            str(env_file),
            "-f",
            str(COMPOSE_FILE),
            "config",
            "--quiet",
        ],
        label="Docker Compose configuration",
        environment=environment,
    )
    if skip_image:
        return
    image = _value(environment, "PIPELINE_IMAGE")
    hub_image = _value(environment, "HUB_IMAGE")
    _run(
        ["docker", "image", "inspect", image],
        label="pipeline image",
        environment=environment,
    )
    _run(
        ["docker", "image", "inspect", hub_image],
        label="Isaac Hub cache image",
        environment=environment,
    )
    _run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--entrypoint",
            "nvidia-smi",
            image,
            "-L",
        ],
        label="container GPU access",
        environment=environment,
    )
    project = _path(environment, "PROJECT_ROOT")
    _run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/opt/conda/envs/hunyuan_sam3d/bin/python",
            "-e",
            "PYTHONPATH=/workspace/hunyuan3.0_assets_creation:/workspace/hunyuan3.0_assets_creation/tools/qwen_material_pipeline/src",
            "-v",
            f"{project}:/workspace/hunyuan3.0_assets_creation:ro",
            image,
            "-c",
            (
                "import asset_pipeline,qwen_material_pipeline,torch,transformers;"
                "assert '/src/qwen_material_pipeline/' in "
                "qwen_material_pipeline.__file__.replace('\\\\','/')"
            ),
        ],
        label="current source and visual-material Python imports",
        environment=environment,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--skip-image", action="store_true")
    parser.add_argument("--minimum-free-gb", type=float)
    parser.add_argument(
        "--prepare-runtime",
        action="store_true",
        help="create only writable output/cache directories before validation",
    )
    return parser


def _prepare_runtime_directories(environment: Mapping[str, str]) -> None:
    for name in (
        "ASSET_ROOT",
        "ISAAC_CACHE_ROOT",
        "MODEL_CACHE_ROOT",
        "VISUAL_RETRIEVAL_CACHE",
    ):
        _path(environment, name).mkdir(parents=True, exist_ok=True)
    isaac_root = _path(environment, "ISAAC_CACHE_ROOT")
    for relative in (
        "cache/main",
        "cache/computecache",
        "logs",
        "config",
        "data",
        "pkg",
    ):
        (isaac_root / relative).mkdir(parents=True, exist_ok=True)
    model_root = _path(environment, "MODEL_CACHE_ROOT")
    for relative in ("home", "ov-hub"):
        (model_root / relative).mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env_file = args.env_file.expanduser().resolve()
    try:
        environment = load_environment(env_file)
        if args.prepare_runtime:
            _prepare_runtime_directories(environment)
        raw_minimum = (
            args.minimum_free_gb
            if args.minimum_free_gb is not None
            else environment.get("PIPELINE_MIN_FREE_GB", "50")
        )
        try:
            minimum_free_gb = float(raw_minimum)
        except (TypeError, ValueError) as exc:
            raise HostPreflightError(
                f"PIPELINE_MIN_FREE_GB must be numeric, got {raw_minimum!r}"
            ) from exc
        if minimum_free_gb <= 0:
            raise HostPreflightError("--minimum-free-gb must be greater than zero")
        _check_layout(environment, minimum_free_gb=minimum_free_gb)
        _check_docker(
            environment,
            env_file=env_file,
            skip_image=args.skip_image,
        )
    except Exception as exc:
        print(f"[HOST PREFLIGHT] FAILED: {exc}", file=sys.stderr, flush=True)
        return 2
    print("[HOST PREFLIGHT] PASS complete asset-pipeline Docker", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
