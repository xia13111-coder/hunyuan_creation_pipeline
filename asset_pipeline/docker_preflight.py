"""Fail-fast runtime checks for the complete Docker deployment."""

from __future__ import annotations

import argparse
import importlib
import os
import resource
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from . import runtime
from .visual_materials import load_visual_material_config


class DockerPreflightError(RuntimeError):
    """Raised when a container cannot run the requested pipeline profile."""


def _require_file(path: Path, label: str, *, executable: bool = False) -> Path:
    if not path.is_file():
        raise DockerPreflightError(f"{label} is missing: {path}")
    if executable and not os.access(path, os.X_OK):
        raise DockerPreflightError(f"{label} is not executable: {path}")
    return path


def _require_directory(path: Path, label: str, *, writable: bool = False) -> Path:
    if not path.is_dir():
        raise DockerPreflightError(f"{label} is missing: {path}")
    if writable and not os.access(path, os.W_OK | os.X_OK):
        raise DockerPreflightError(f"{label} is not writable: {path}")
    return path


def _require_model_config(path: Path, label: str) -> None:
    _require_directory(path, label)
    _require_file(path / "config.json", f"{label} config")


def _probe_python(
    python: Path,
    *,
    label: str,
    modules: Iterable[str],
    python_paths: Iterable[Path] = (),
) -> None:
    module_names = tuple(dict.fromkeys(modules))
    script = (
        "import importlib.util,sys; "
        f"missing=[m for m in {module_names!r} if importlib.util.find_spec(m) is None]; "
        "sys.exit('missing Python modules: '+','.join(missing) if missing else 0)"
    )
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    additional_paths = [str(path) for path in python_paths]
    if additional_paths:
        inherited = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            [*additional_paths, *([inherited] if inherited else [])]
        )
    try:
        completed = subprocess.run(
            [str(python), "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DockerPreflightError(f"Unable to probe {label}: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise DockerPreflightError(
            f"{label} failed its import probe: {detail or completed.returncode}"
        )


def _require_source_owned_module(module_name: str, expected_root: Path) -> None:
    module = importlib.import_module(module_name)
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str):
        raise DockerPreflightError(f"{module_name} has no concrete module path")
    module_path = Path(raw_path).resolve()
    try:
        module_path.relative_to(expected_root.resolve())
    except ValueError as exc:
        raise DockerPreflightError(
            f"{module_name} resolves to stale code outside {expected_root}: {module_path}"
        ) from exc


def _require_free_space(path: Path) -> None:
    raw_minimum = os.getenv("PIPELINE_MIN_FREE_GB", "50").strip()
    try:
        minimum_gb = float(raw_minimum)
    except ValueError as exc:
        raise DockerPreflightError(
            f"PIPELINE_MIN_FREE_GB must be numeric, got {raw_minimum!r}"
        ) from exc
    if minimum_gb <= 0:
        raise DockerPreflightError("PIPELINE_MIN_FREE_GB must be greater than zero")
    free_bytes = shutil.disk_usage(path).free
    minimum_bytes = int(minimum_gb * 1024**3)
    if free_bytes < minimum_bytes:
        free_gb = free_bytes / 1024**3
        raise DockerPreflightError(
            f"asset/output filesystem has only {free_gb:.1f} GiB free at {path}; "
            f"at least {minimum_gb:g} GiB is required"
        )


def _require_open_file_limit() -> None:
    raw_minimum = os.getenv("PIPELINE_MIN_NOFILE", "65536").strip()
    try:
        minimum = int(raw_minimum)
    except ValueError as exc:
        raise DockerPreflightError(
            f"PIPELINE_MIN_NOFILE must be an integer, got {raw_minimum!r}"
        ) from exc
    if minimum <= 0:
        raise DockerPreflightError("PIPELINE_MIN_NOFILE must be greater than zero")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < minimum:
        raise DockerPreflightError(
            f"open-file soft limit is {soft}, but at least {minimum} is required "
            f"for Isaac rendering (hard limit: {hard})"
        )


def run_basic_preflight() -> None:
    configured = runtime.configure_runtime()
    root = runtime.root_dir()
    _require_directory(root, "project root")
    _require_file(Path(configured["BLENDER_BIN"] or ""), "Blender", executable=True)
    _require_file(Path(configured["ISAAC_PYTHON"] or ""), "Isaac Python", executable=True)
    asset_root = Path(
        os.getenv("PIPELINE_ASSET_ROOT", "/workspace/assets")
    ).expanduser()
    _require_directory(asset_root, "asset/output root", writable=True)
    _require_free_space(asset_root)
    _require_open_file_limit()
    _require_source_owned_module("asset_pipeline", root / "asset_pipeline")
    _require_file(
        root / "docker" / "bin" / "hunyuan-asset-pipeline",
        "asset pipeline command",
        executable=True,
    )
    _require_file(
        root / "docker" / "bin" / "manual-material-pipeline",
        "manual material command",
        executable=True,
    )
    _require_file(
        root / "docker" / "bin" / "qwen-material",
        "Qwen material command",
        executable=True,
    )


def run_visual_material_preflight(*, probe_pythons: bool = True) -> None:
    run_basic_preflight()
    root = runtime.root_dir()
    config = load_visual_material_config(None)

    _require_source_owned_module(
        "qwen_material_pipeline",
        root / "tools" / "qwen_material_pipeline" / "src" / "qwen_material_pipeline",
    )
    _require_model_config(config.qwen_model_path or Path(), "Qwen3.5 model")
    _require_model_config(config.siglip2_model_path, "SigLIP2 model")
    _require_model_config(config.dinov2_model_path, "DINOv2 model")
    _require_directory(config.mvinverse_repository, "MVInverse repository")
    _require_directory(config.mvinverse_checkpoint, "MVInverse checkpoint")
    _require_file(config.sam3_checkpoint, "SAM3 checkpoint")
    _require_directory(config.sam3_repository, "SAM3 repository")
    _require_directory(config.material_root, "NVIDIA Base materials")
    if not any(config.material_root.rglob("*.mdl")):
        raise DockerPreflightError(
            f"NVIDIA Base material root contains no MDL files: {config.material_root}"
        )
    _require_directory(config.retrieval_cache_dir, "visual retrieval cache", writable=True)
    if config.retrieval_observation_bank_dir is None:
        raise DockerPreflightError("NVIDIA Base observation bank is not configured")
    _require_file(
        config.retrieval_observation_bank_dir / "index_manifest.json",
        "NVIDIA Base observation-bank manifest",
    )

    if not config.entityseg_enabled:
        raise DockerPreflightError("the production profile requires EntitySeg")
    if (
        config.entityseg_python is None
        or config.entityseg_cropformer_root is None
        or config.entityseg_config is None
        or config.entityseg_checkpoint is None
    ):
        raise DockerPreflightError("EntitySeg runtime configuration is incomplete")
    _require_file(config.entityseg_python, "EntitySeg Python", executable=True)
    _require_directory(config.entityseg_cropformer_root, "EntitySeg CropFormer root")
    _require_file(config.entityseg_config, "EntitySeg config")
    _require_file(config.entityseg_checkpoint, "EntitySeg checkpoint")
    raw_entityseg_detectron2_root = os.getenv("ENTITYSEG_DETECTRON2_ROOT", "").strip()
    if not raw_entityseg_detectron2_root:
        raise DockerPreflightError("ENTITYSEG_DETECTRON2_ROOT is not configured")
    entityseg_detectron2_root = _require_directory(
        Path(raw_entityseg_detectron2_root),
        "EntitySeg Detectron2 root",
    )
    _require_file(
        entityseg_detectron2_root / "detectron2" / "__init__.py",
        "EntitySeg Detectron2 package",
    )

    if probe_pythons:
        probes: dict[Path, tuple[str, set[str]]] = {}

        def add_probe(path: Path, label: str, modules: Iterable[str]) -> None:
            absolute = path.absolute()
            if absolute not in probes:
                probes[absolute] = (label, set())
            probes[absolute][1].update(modules)

        add_probe(config.qwen_python, "Qwen3.5 Python", ("torch", "transformers"))
        add_probe(
            config.mvinverse_python,
            "MVInverse Python",
            ("torch", "hydra", "cv2"),
        )
        add_probe(config.sam3_python, "SAM3 Python", ("torch", "cv2"))
        add_probe(
            config.retrieval_python,
            "visual retrieval Python",
            ("torch", "transformers", "cv2"),
        )
        add_probe(
            config.entityseg_python,
            "EntitySeg Python",
            ("torch", "cv2", "detectron2"),
        )
        for python, (label, modules) in probes.items():
            _probe_python(
                python,
                label=label,
                modules=sorted(modules),
                python_paths=(entityseg_detectron2_root,)
                if python == config.entityseg_python.absolute()
                else (),
            )


def _configured_path(configured: dict[str, str | None], name: str) -> Path:
    raw_path = configured.get(name)
    if not raw_path:
        raise DockerPreflightError(f"{name} is not configured")
    return Path(raw_path)


def run_complete_preflight(*, probe_pythons: bool = True) -> None:
    run_visual_material_preflight(probe_pythons=probe_pythons)
    configured = runtime.configure_runtime()
    _require_directory(
        _configured_path(configured, "SAM3D_SINGLE_VIEW_ROOT"),
        "SAM3D single-view source",
    )
    _require_directory(
        _configured_path(configured, "SAM3D_MULTI_VIEW_ROOT"),
        "SAM3D multi-view source",
    )
    _require_file(
        _configured_path(configured, "SAM3D_PIPELINE_CONFIG"),
        "SAM3D pipeline config",
    )
    _require_file(
        _configured_path(configured, "SAM3D_MOGE_CHECKPOINT"),
        "MoGe checkpoint",
    )
    _require_directory(
        _configured_path(configured, "SAM3D_DINOV2_REPOSITORY"),
        "SAM3D DINOv2 repository",
    )
    _require_file(
        _configured_path(configured, "SAM3D_DINOV2_CHECKPOINT"),
        "SAM3D DINOv2 checkpoint",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("basic", "visual-materials", "complete"),
        default="complete",
    )
    parser.add_argument(
        "--skip-python-probes",
        action="store_true",
        help="validate paths and configuration without starting external interpreters",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.profile == "basic":
            run_basic_preflight()
        elif args.profile == "visual-materials":
            run_visual_material_preflight(
                probe_pythons=not args.skip_python_probes
            )
        else:
            run_complete_preflight(probe_pythons=not args.skip_python_probes)
    except Exception as exc:
        print(f"[DOCKER PREFLIGHT] FAILED: {exc}", file=sys.stderr, flush=True)
        return 2
    print(f"[DOCKER PREFLIGHT] PASS profile={args.profile}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
