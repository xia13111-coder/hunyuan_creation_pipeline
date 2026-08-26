#!/usr/bin/env python3
"""Fail when public source candidates contain common release hazards."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


REQUIRED_FILES = (
    "README.md",
    "README.zh.md",
    "LICENSE",
    "NOTICE",
    "legal/README.md",
    "legal/README.zh.md",
    "legal/THIRD_PARTY_NOTICES.md",
    ".github/CONTRIBUTING.md",
    ".github/SECURITY.md",
    ".github/CODE_OF_CONDUCT.md",
    "CHANGELOG.md",
    ".env.example",
    "tools/qwen_material_pipeline/LICENSE",
    "tools/qwen_material_pipeline/third_party/mvinverse/LICENSE",
    "tools/qwen_material_pipeline/third_party/mvinverse/DINOV2_LICENSE",
)
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".cff",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".mjs",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
MODEL_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".engine",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}
IGNORED_BUILD_PARTS = {
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}
DIRECT_SECRET = re.compile(
    r"(?i)(?:sk-[a-z0-9_-]{20,}|AKID[a-z0-9]{20,}|bearer\s+[a-z0-9._~-]{24,})"
)
ASSIGNED_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|secret[_-]?(?:id|key)|access[_-]?token)\b"
    r"\s*[:=]\s*[\"']?([a-z0-9/+=._~-]{16,})"
)
HOST_PATH = re.compile(r"/(home|media)/([A-Za-z0-9._-]+)/")
MARKDOWN_LINK = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
SAFE_SECRET_MARKERS = (
    "dummy",
    "example",
    "placeholder",
    "redacted",
    "your-",
    "your_",
)


def _source_candidates(root: Path) -> list[Path]:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
    )
    paths: list[Path] = []
    for raw in completed.stdout.split(b"\0"):
        if raw:
            paths.append(root / raw.decode("utf-8", errors="surrogateescape"))
    return paths


def _is_allowed_output_readme(relative: Path) -> bool:
    return relative.as_posix() == "outputs/README.md"


def _inspect_text(relative: Path, text: str) -> list[str]:
    issues: list[str] = []
    if DIRECT_SECRET.search(text):
        issues.append("contains a token-shaped secret")
    for match in ASSIGNED_SECRET.finditer(text):
        value = match.group(2)
        lowered = value.lower()
        if value.isupper() or any(marker in lowered for marker in SAFE_SECRET_MARKERS):
            continue
        issues.append(f"contains a value assigned to {match.group(1)}")
        break
    for match in HOST_PATH.finditer(text):
        account = match.group(2)
        if account == "pipeline":
            continue
        issues.append(
            f"contains a host-specific absolute path under /{match.group(1)}/{account}"
        )
        break
    return issues


def _inspect_markdown_links(root: Path, path: Path, text: str) -> list[str]:
    if "third_party" in path.relative_to(root).parts:
        return []
    issues: list[str] = []
    for raw_target in MARKDOWN_LINK.findall(text):
        target = raw_target.strip().strip("<>")
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        relative_target = unquote(target.split("#", 1)[0].split("?", 1)[0])
        if not relative_target:
            continue
        destination = (
            root / relative_target.lstrip("/")
            if relative_target.startswith("/")
            else path.parent / relative_target
        )
        if not destination.exists():
            issues.append(f"contains a broken local Markdown link: {target}")
    return issues


def audit(root: Path) -> list[str]:
    issues: list[str] = []
    for name in REQUIRED_FILES:
        if not (root / name).is_file():
            issues.append(f"missing required public-release file: {name}")
    for path in sorted(root.glob("*.py")):
        issues.append(
            "Python source must be owned by a package or tool directory, not "
            f"the repository root: {path.name}"
        )

    try:
        candidates = _source_candidates(root)
    except (OSError, subprocess.CalledProcessError) as exc:
        return [f"unable to enumerate Git source candidates: {exc}"]

    for path in candidates:
        try:
            relative = path.relative_to(root)
        except ValueError:
            issues.append(f"candidate escapes project root: {path}")
            continue

        if path.is_dir():
            if (path / ".git").exists():
                issues.append(
                    f"nested Git repository is a release candidate: {relative.as_posix()}"
                )
            continue
        if not path.exists():
            continue

        parts = set(relative.parts)
        if parts & IGNORED_BUILD_PARTS:
            issues.append(f"build/cache path is a release candidate: {relative.as_posix()}")
        if relative.parts and relative.parts[0] == "outputs" and not _is_allowed_output_readme(relative):
            issues.append(f"generated output is a release candidate: {relative.as_posix()}")
        if relative.name == ".env" or (
            relative.name.startswith(".env.") and relative.name != ".env.example"
        ):
            issues.append(f"private environment file is a release candidate: {relative.as_posix()}")
        if path.suffix.lower() in MODEL_SUFFIXES:
            issues.append(f"model/binary weight is a release candidate: {relative.as_posix()}")
        if path.is_symlink() and path.readlink().is_absolute():
            issues.append(f"absolute symlink is a release candidate: {relative.as_posix()}")
            continue
        if path.stat().st_size > 20 * 1024 * 1024:
            issues.append(f"file exceeds 20 MiB: {relative.as_posix()}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for issue in _inspect_text(relative, text):
            issues.append(f"{relative.as_posix()}: {issue}")
        if path.suffix.lower() == ".md":
            for issue in _inspect_markdown_links(root, path, text):
                issues.append(f"{relative.as_posix()}: {issue}")

    return sorted(set(issues))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit tracked and non-ignored source candidates for credentials, "
            "machine paths, nested repositories, model weights, and generated files."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (default: inferred from this script)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    issues = audit(root)
    if issues:
        print(f"Public release audit FAILED ({len(issues)} issue(s)):")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("Public release audit PASSED: source candidates contain no detected hazards.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
