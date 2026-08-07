"""Runtime discovery and project-level defaults."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from .local_models import (
    configure_discovered_sam3d_environment,
    configure_offline_model_environment,
)
from .project_layout import SOURCE_LAYOUT


PROJECT_ROOT = SOURCE_LAYOUT.root
UNIFIED_CONDA_ENV = "hunyuan_sam3d"
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def project_root() -> Path:
    return PROJECT_ROOT


def _strip_dotenv_comment(value: str) -> str:
    """Remove an unquoted dotenv comment without evaluating shell syntax."""

    quote: str | None = None
    escaped = False
    for index, character in enumerate(value):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote == '"':
            escaped = True
            continue
        if character in {"'", '"'}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
            continue
        if (
            character == "#"
            and quote is None
            and (index == 0 or value[index - 1].isspace())
        ):
            return value[:index].rstrip()
    return value.strip()


def _parse_dotenv_value(raw_value: str, *, path: Path, line_number: int) -> str:
    value = _strip_dotenv_comment(raw_value.strip())
    if not value:
        return ""
    if value[0] not in {"'", '"'}:
        return value
    quote = value[0]
    if len(value) < 2 or value[-1] != quote:
        raise ValueError(f"Unterminated quoted value in {path}:{line_number}")
    inner = value[1:-1]
    if quote == "'":
        return inner
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid escape sequence in {path}:{line_number}") from exc
    if not isinstance(decoded, str):  # pragma: no cover - JSON quotes ensure this.
        raise ValueError(f"Invalid quoted value in {path}:{line_number}")
    return decoded


def load_environment_file(path: str | Path) -> tuple[str, ...]:
    """Load non-empty dotenv assignments without overriding the process.

    The parser deliberately does not execute shell expressions or interpolate
    variables. Existing process variables have priority, and blank template
    assignments are ignored so automatic runtime discovery keeps working.
    """

    env_path = Path(path).expanduser().resolve()
    if not env_path.is_file():
        return ()
    loaded: list[str] = []
    for line_number, raw_line in enumerate(
        env_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(
                f"Invalid environment assignment in {env_path}:{line_number}"
            )
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if not _ENV_NAME.fullmatch(name):
            raise ValueError(f"Invalid environment name in {env_path}:{line_number}")
        value = _parse_dotenv_value(raw_value, path=env_path, line_number=line_number)
        if not value or name in os.environ:
            continue
        os.environ[name] = value
        loaded.append(name)
    return tuple(loaded)


def load_project_environment() -> tuple[str, ...]:
    """Load the checkout-local ``.env`` file when it exists."""

    return load_environment_file(PROJECT_ROOT / ".env")


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
    load_project_environment()
    require_unified_environment()
    configure_offline_model_environment()
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
    configure_discovered_sam3d_environment(PROJECT_ROOT)
    # Qwen3-VL and MVInverse deliberately share the pinned SAM3D Torch/CUDA
    # runtime. Isaac Sim and Blender retain their vendor-managed interpreters.
    os.environ.setdefault("QWEN_PYTHON", sys.executable)
    os.environ.setdefault("MVINVERSE_PYTHON", sys.executable)
    visual_material_root = SOURCE_LAYOUT.material_pipeline
    cache_root = (
        Path(
            os.getenv(
                "MODEL_CACHE_ROOT",
                str(
                    Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache")))
                    / "hunyuan_asset_pipeline"
                ),
            )
        )
        .expanduser()
        .resolve()
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MODEL_CACHE_ROOT", str(cache_root))
    qwen35_runtime_roots = sorted(Path("/media").glob("*/WD_BLACK/qwen35_4b_runtime"))
    default_qwen35_python = first_existing_path(
        [
            os.getenv("QWEN35_PYTHON") or "",
            *(root / "env" / "bin" / "python" for root in qwen35_runtime_roots),
            cache_root / "qwen35_4b_runtime" / "env" / "bin" / "python",
        ]
    )
    default_qwen35_model = first_existing_path(
        [
            os.getenv("QWEN35_MODEL_PATH") or "",
            *(root / "model" for root in qwen35_runtime_roots),
            cache_root / "qwen35_4b_runtime" / "model",
            visual_material_root / "models" / "qwen" / "Qwen3.5-4B",
        ]
    )
    if default_qwen35_python is not None:
        os.environ.setdefault("QWEN35_PYTHON", str(default_qwen35_python))
    if default_qwen35_model is not None:
        os.environ.setdefault("QWEN35_MODEL_PATH", str(default_qwen35_model))
    removable_qwen_models = sorted(
        Path("/media").glob("*/WD_BLACK/qwen_models/Qwen3-VL-4B-Instruct")
    )
    default_qwen_model = first_existing_path(
        [
            os.getenv("QWEN_MODEL_PATH") or "",
            *removable_qwen_models,
            visual_material_root / "models" / "qwen" / "Qwen3-VL-4B-Instruct",
        ]
    )
    os.environ.setdefault(
        "QWEN_MODEL_PATH",
        str(
            default_qwen_model
            or visual_material_root / "models" / "qwen" / "Qwen3-VL-4B-Instruct"
        ),
    )
    default_mvinverse_repository = first_existing_path(
        [
            os.getenv("MVINVERSE_REPOSITORY") or "",
            visual_material_root / "third_party" / "mvinverse",
        ]
    )
    default_mvinverse_checkpoint = first_existing_path(
        [
            os.getenv("MVINVERSE_CHECKPOINT") or "",
            visual_material_root / "models" / "mvinverse" / "model",
        ]
    )
    default_sam3_repository = first_existing_path(
        [
            os.getenv("SAM3_REPOSITORY") or "",
            SOURCE_LAYOUT.sam3d_single_view / "submodules" / "sam3",
        ]
    )
    default_sam3_checkpoint = first_existing_path(
        [
            os.getenv("SAM3_CHECKPOINT") or "",
            SOURCE_LAYOUT.sam3d_single_view / "checkpoints" / "sam3.pt",
        ]
    )
    if default_mvinverse_repository is not None:
        os.environ.setdefault("MVINVERSE_REPOSITORY", str(default_mvinverse_repository))
    if default_mvinverse_checkpoint is not None:
        os.environ.setdefault("MVINVERSE_CHECKPOINT", str(default_mvinverse_checkpoint))
    if default_sam3_repository is not None:
        os.environ.setdefault("SAM3_REPOSITORY", str(default_sam3_repository))
    if default_sam3_checkpoint is not None:
        os.environ.setdefault("SAM3_CHECKPOINT", str(default_sam3_checkpoint))
    removable_siglip_models = sorted(
        Path("/media").glob("*/WD_BLACK/qwen_models/siglip2-base-patch16-224")
    )
    default_siglip_model = first_existing_path(
        [
            os.getenv("SIGLIP2_MODEL_PATH") or "",
            *removable_siglip_models,
            cache_root / "models" / "siglip2-base-patch16-224",
        ]
    )
    os.environ.setdefault(
        "SIGLIP2_MODEL_PATH",
        str(default_siglip_model or cache_root / "models" / "siglip2-base-patch16-224"),
    )
    removable_dino_models = sorted(
        Path("/media").glob(
            "*/WD_BLACK/gsfixer_reproduction/checkpoints/dinov2-with-registers-large"
        )
    )
    default_dino_model = first_existing_path(
        [
            os.getenv("DINOV2_MODEL_PATH") or "",
            *removable_dino_models,
            cache_root / "models" / "dinov2-with-registers-large",
        ]
    )
    os.environ.setdefault(
        "DINOV2_MODEL_PATH",
        str(
            default_dino_model or cache_root / "models" / "dinov2-with-registers-large"
        ),
    )
    removable_retrieval_caches = sorted(
        Path("/media").glob("*/WD_BLACK/qwen_material_cache")
    )
    default_retrieval_cache = first_existing_path(
        [
            os.getenv("VISUAL_RETRIEVAL_CACHE") or "",
            *removable_retrieval_caches,
            cache_root / "retrieval",
        ]
    )
    os.environ.setdefault(
        "VISUAL_RETRIEVAL_CACHE",
        str(default_retrieval_cache or cache_root / "retrieval"),
    )
    default_observation_bank = first_existing_path(
        [
            os.getenv("NVIDIA_BASE_OBSERVATION_BANK") or "",
            *(
                root / "nvidia_base_observation_bank_v1"
                for root in removable_retrieval_caches
            ),
            cache_root / "nvidia_base_observation_bank_v1",
        ]
    )
    os.environ.setdefault(
        "NVIDIA_BASE_OBSERVATION_BANK",
        str(default_observation_bank or cache_root / "nvidia_base_observation_bank_v1"),
    )
    default_nvidia_materials = first_existing_path(
        [
            os.getenv("NVIDIA_MDL_MATERIALS_ROOT") or "",
            (
                Path(os.getenv("NVIDIA_MDL_BASE_ROOT", "")).expanduser().parent
                if os.getenv("NVIDIA_MDL_BASE_ROOT")
                and Path(os.getenv("NVIDIA_MDL_BASE_ROOT", "")).expanduser().name
                == "Base"
                else os.getenv("NVIDIA_MDL_BASE_ROOT") or ""
            ),
            Path.home()
            / "isaacsim_assets"
            / "Assets"
            / "Isaac"
            / "4.5"
            / "NVIDIA"
            / "Materials",
            visual_material_root / "models" / "materials" / "nvidia",
        ]
    )
    os.environ.setdefault(
        "VISUAL_MATERIAL_ROOT",
        str(
            default_nvidia_materials
            or (
                Path.home()
                / "isaacsim_assets"
                / "Assets"
                / "Isaac"
                / "4.5"
                / "NVIDIA"
                / "Materials"
            )
        ),
    )
    return {
        "ROOT_DIR": os.getenv("ROOT_DIR"),
        "CONDA_ENV": active_environment_name(),
        "BLENDER_BIN": os.getenv("BLENDER_BIN"),
        "ISAACSIM_ROOT": os.getenv("ISAACSIM_ROOT"),
        "ISAAC_PYTHON": os.getenv("ISAAC_PYTHON"),
        "REFINE_MESH_TEMP_UPLOAD": os.getenv("REFINE_MESH_TEMP_UPLOAD"),
        "QWEN_PYTHON": os.getenv("QWEN_PYTHON"),
        "MVINVERSE_PYTHON": os.getenv("MVINVERSE_PYTHON"),
        "QWEN_MODEL_PATH": os.getenv("QWEN_MODEL_PATH"),
        "QWEN35_PYTHON": os.getenv("QWEN35_PYTHON"),
        "QWEN35_MODEL_PATH": os.getenv("QWEN35_MODEL_PATH"),
        "SIGLIP2_MODEL_PATH": os.getenv("SIGLIP2_MODEL_PATH"),
        "DINOV2_MODEL_PATH": os.getenv("DINOV2_MODEL_PATH"),
        "MVINVERSE_REPOSITORY": os.getenv("MVINVERSE_REPOSITORY"),
        "MVINVERSE_CHECKPOINT": os.getenv("MVINVERSE_CHECKPOINT"),
        "SAM3_REPOSITORY": os.getenv("SAM3_REPOSITORY"),
        "SAM3_CHECKPOINT": os.getenv("SAM3_CHECKPOINT"),
        "SAM3D_SINGLE_VIEW_ROOT": os.getenv("SAM3D_SINGLE_VIEW_ROOT"),
        "SAM3D_MULTI_VIEW_ROOT": os.getenv("SAM3D_MULTI_VIEW_ROOT"),
        "SAM3D_PIPELINE_CONFIG": os.getenv("SAM3D_PIPELINE_CONFIG"),
        "SAM3D_MOGE_CHECKPOINT": os.getenv("SAM3D_MOGE_CHECKPOINT"),
        "SAM3D_DINOV2_REPOSITORY": os.getenv("SAM3D_DINOV2_REPOSITORY"),
        "SAM3D_DINOV2_CHECKPOINT": os.getenv("SAM3D_DINOV2_CHECKPOINT"),
        "LOCAL_MODELS_ONLY": os.getenv("PIPELINE_LOCAL_MODELS_ONLY"),
        "MODEL_CACHE_ROOT": os.getenv("MODEL_CACHE_ROOT"),
        "VISUAL_MATERIAL_ROOT": os.getenv("VISUAL_MATERIAL_ROOT"),
    }


def runtime_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "root_dir": str(root_dir()),
        "python_bin": sys.executable,
        "sam3d_python": str(sam3d_python()),
        "blender_bin": str(blender_bin()),
        "isaac_python": str(isaac_python()),
        "refine_mesh_config": str(default_refine_config_path()),
        "qwen_python": os.getenv("QWEN_PYTHON", sys.executable),
        "mvinverse_python": os.getenv("MVINVERSE_PYTHON", sys.executable),
        "qwen_model_path": os.getenv("QWEN_MODEL_PATH"),
        "qwen35_python": os.getenv("QWEN35_PYTHON"),
        "qwen35_model_path": os.getenv("QWEN35_MODEL_PATH"),
        "siglip2_model_path": os.getenv("SIGLIP2_MODEL_PATH"),
        "dinov2_model_path": os.getenv("DINOV2_MODEL_PATH"),
        "mvinverse_repository": os.getenv("MVINVERSE_REPOSITORY"),
        "mvinverse_checkpoint": os.getenv("MVINVERSE_CHECKPOINT"),
        "sam3_repository": os.getenv("SAM3_REPOSITORY"),
        "sam3_checkpoint": os.getenv("SAM3_CHECKPOINT"),
        "sam3d_single_view_root": os.getenv("SAM3D_SINGLE_VIEW_ROOT"),
        "sam3d_multi_view_root": os.getenv("SAM3D_MULTI_VIEW_ROOT"),
        "sam3d_pipeline_config": os.getenv("SAM3D_PIPELINE_CONFIG"),
        "sam3d_moge_checkpoint": os.getenv("SAM3D_MOGE_CHECKPOINT"),
        "sam3d_dinov2_repository": os.getenv("SAM3D_DINOV2_REPOSITORY"),
        "sam3d_dinov2_checkpoint": os.getenv("SAM3D_DINOV2_CHECKPOINT"),
        "local_models_only": os.getenv("PIPELINE_LOCAL_MODELS_ONLY", "1"),
        "model_cache_root": os.getenv("MODEL_CACHE_ROOT"),
        "visual_material_root": os.getenv("VISUAL_MATERIAL_ROOT"),
    }
    summary["blender_exists"] = blender_bin().exists()
    summary["isaac_python_exists"] = isaac_python().exists()
    summary["sam3d_python_exists"] = sam3d_python().exists()
    summary["refine_mesh_config_exists"] = default_refine_config_path().exists()
    return summary
