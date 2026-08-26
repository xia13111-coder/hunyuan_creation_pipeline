from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SOURCE_ROOT / "qwen_material_pipeline"


def _tracked_project_paths() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "tools/qwen_material_pipeline"],
        cwd=PROJECT_ROOT.parents[1],
    )
    prefix = Path("tools/qwen_material_pipeline")
    return [
        Path(os.fsdecode(value)).relative_to(prefix)
        for value in output.split(b"\0")
        if value
    ]


def test_python_sources_have_explicit_owners() -> None:
    allowed_roots = {"src", "tests", "scripts", "third_party"}
    misplaced = sorted(
        path.as_posix()
        for path in _tracked_project_paths()
        if path.suffix == ".py" and path.parts[0] not in allowed_roots
    )

    assert misplaced == []


def test_legacy_flat_package_directories_are_absent() -> None:
    old_package_entries = {
        "__init__.py",
        "__main__.py",
        "configs",
        "core",
        "evidence",
        "materials",
        "mvinverse",
        "projects",
        "qwen",
        "retrieval",
        "schemas",
        "segmentation",
        "usd",
        "web",
        "workflows",
    }

    assert PACKAGE_ROOT.is_dir()
    assert (
        sorted(name for name in old_package_entries if (PROJECT_ROOT / name).exists())
        == []
    )


def test_setuptools_builds_only_the_src_package_tree() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'package-dir = {"" = "src"}' in pyproject
    assert 'where = ["src"]' in pyproject
    assert '"qwen_material_pipeline.projects.*"' in pyproject
    assert 'pythonpath = ["src"]' in pyproject
    assert '"runtime/' not in pyproject
    assert not (SOURCE_ROOT / "runtime").exists()
