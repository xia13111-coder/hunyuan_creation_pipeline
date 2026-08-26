"""Namespace for hash-bound material-project plugins kept outside the wheel."""

from __future__ import annotations

from qwen_material_pipeline.core.paths import PROJECTS_ROOT


if PROJECTS_ROOT.is_dir():
    __path__.append(str(PROJECTS_ROOT))
