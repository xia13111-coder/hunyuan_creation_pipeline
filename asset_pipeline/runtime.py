"""Runtime discovery and project-level defaults."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIFIED_CONDA_ENV = "hunyuan_sam3d"


def project_root() -> Path:
    return PROJECT_ROOT


def root_dir() -> Path:
    return Path(os.getenv("ROOT_DIR", str(PROJECT_ROOT))).expanduser().resolve()


def first_existing_path(candidates: list[str | Path]) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return path.resolve()
    return None


def blender_bin() -> Path:
    env_value = os.getenv("BLENDER_BIN")
    if env_value:
        return Path(env_value).expanduser().resolve()
    candidate = first_existing_path(
        [
            shutil.which("blender") or "",
            "/opt/blender/blender",
            "/usr/local/bin/blender",
        ]
    )
    return candidate or Path("/opt/blender/blender").resolve()


def isaac_python() -> Path:
    env_value = os.getenv("ISAAC_PYTHON")
    if env_value:
        return Path(env_value).expanduser().resolve()

    root_value = os.getenv("ISAACSIM_ROOT")
    if root_value:
        return (Path(root_value).expanduser().resolve() / "python.sh").resolve()

    candidates: list[str | Path] = [
        Path.home() / "isaacsim601" / "python.sh",
        Path.home() / "isaacsim600" / "python.sh",
        Path.home() / "isaacsim500" / "python.sh",
        Path.home() / "isaacsim" / "python.sh",
        Path.home() / "isaac-sim" / "python.sh",
        "/isaac-sim/python.sh",
        "/opt/isaac-sim/python.sh",
    ]
    ov_pkg_dir = Path.home() / ".local" / "share" / "ov" / "pkg"
    if ov_pkg_dir.exists():
        candidates.extend(sorted(ov_pkg_dir.glob("isaac_sim*/python.sh")))
    return first_existing_path(candidates) or Path("/isaac-sim/python.sh").resolve()


def sam3d_python() -> Path:
    return Path(sys.executable).resolve()


def active_environment_name() -> str:
    return Path(sys.prefix).name


def require_unified_environment() -> None:
    active = active_environment_name()
    if active != UNIFIED_CONDA_ENV:
        raise RuntimeError(
            f"This project uses only the '{UNIFIED_CONDA_ENV}' conda environment; "
            f"current Python environment is '{active}'. Run: conda activate {UNIFIED_CONDA_ENV}"
        )


def materials_file() -> Path:
    return root_dir() / "materials.json"


def default_refine_config_path() -> Path:
    return root_dir() / "configs" / "hunyuan_reduce_local_postprocess.yaml"


def default_refine_temp_upload() -> str | None:
    value = os.getenv("REFINE_MESH_TEMP_UPLOAD", "uguu").strip()
    return value or None


def _default_sidecar_output_dir(input_path: str, suffix: str) -> str:
    base = Path(input_path).expanduser()
    if base.suffix:
        return str(base.with_suffix("").with_name(base.stem + suffix))
    return str(base.with_name(base.name + suffix))


def default_refine_output_dir(input_path: str) -> str:
    return _default_sidecar_output_dir(input_path, "_refined_mesh")


def default_cad_usd_output_dir(input_path: str) -> str:
    return _default_sidecar_output_dir(input_path, "_cad_usd")


def available_materials() -> list[str]:
    try:
        data = json.loads(materials_file().read_text(encoding="utf-8"))
        return sorted(str(name) for name in data.get("materials", {}))
    except (OSError, ValueError, TypeError):
        return []


def available_approx_types() -> list[str]:
    return [
        "sdf",
        "convexHull",
        "convexDecomposition",
        "triangleMesh",
        "meshSimplification",
        "boundingCube",
        "boundingSphere",
        "sphereApproximation",
    ]


def configure_runtime() -> dict[str, str | None]:
    require_unified_environment()
    blender = first_existing_path(
        [
            os.getenv("BLENDER_BIN") or "",
            shutil.which("blender") or "",
            "/opt/blender/blender",
            "/usr/local/bin/blender",
        ]
    )
    if blender:
        os.environ["BLENDER_BIN"] = str(blender)

    isaac = isaac_python()
    if isaac.exists():
        os.environ["ISAAC_PYTHON"] = str(isaac)
        os.environ.setdefault("ISAACSIM_ROOT", str(isaac.parent))

    os.environ.setdefault("ROOT_DIR", str(PROJECT_ROOT))
    os.environ.setdefault("REFINE_MESH_TEMP_UPLOAD", "uguu")
    return {
        "ROOT_DIR": os.getenv("ROOT_DIR"),
        "CONDA_ENV": active_environment_name(),
        "BLENDER_BIN": os.getenv("BLENDER_BIN"),
        "ISAACSIM_ROOT": os.getenv("ISAACSIM_ROOT"),
        "ISAAC_PYTHON": os.getenv("ISAAC_PYTHON"),
        "REFINE_MESH_TEMP_UPLOAD": os.getenv("REFINE_MESH_TEMP_UPLOAD"),
    }


def runtime_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "root_dir": str(root_dir()),
        "python_bin": sys.executable,
        "sam3d_python": str(sam3d_python()),
        "blender_bin": str(blender_bin()),
        "isaac_python": str(isaac_python()),
        "refine_mesh_config": str(default_refine_config_path()),
    }
    summary["blender_exists"] = blender_bin().exists()
    summary["isaac_python_exists"] = isaac_python().exists()
    summary["sam3d_python_exists"] = sam3d_python().exists()
    summary["refine_mesh_config_exists"] = default_refine_config_path().exists()
    return summary
