"""Owned paths for the material-pipeline source project and local runtime."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT.parent


def _source_project_root() -> Path:
    if SOURCE_ROOT.name == "src" and (SOURCE_ROOT.parent / "pyproject.toml").is_file():
        return SOURCE_ROOT.parent
    return PACKAGE_ROOT


PROJECT_ROOT = Path(
    os.environ.get("QWEN_MATERIAL_PIPELINE_HOME", _source_project_root())
).expanduser().resolve()
RUNTIME_ROOT = Path(
    os.environ.get("QWEN_MATERIAL_RUNTIME_ROOT", PROJECT_ROOT / "runtime")
).expanduser().resolve()
MODELS_ROOT = RUNTIME_ROOT / "models"
CACHE_ROOT = RUNTIME_ROOT / "cache"
PROJECTS_ROOT = Path(
    os.environ.get("MATERIAL_PROJECTS_ROOT", RUNTIME_ROOT / "projects")
).expanduser().resolve()
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party"


def repository_root() -> Path:
    """Return the enclosing source checkout, or ``ROOT_DIR`` when deployed."""

    configured = os.environ.get("ROOT_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if PROJECT_ROOT.parent.name == "tools":
        return PROJECT_ROOT.parent.parent
    return PROJECT_ROOT


__all__ = [
    "CACHE_ROOT",
    "MODELS_ROOT",
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "PROJECTS_ROOT",
    "RUNTIME_ROOT",
    "SOURCE_ROOT",
    "THIRD_PARTY_ROOT",
    "repository_root",
]
