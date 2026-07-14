"""File discovery and output-path helpers."""

from __future__ import annotations

import re
import time
from pathlib import Path


def list_files_by_suffix(
    root: str, suffixes: set[str], *, name_contains: str | None = None
) -> list[str]:
    base = Path(root)
    if not base.exists():
        return []
    lowered_suffixes = {suffix.lower() for suffix in suffixes}
    lowered_contains = name_contains.lower() if name_contains else None
    if base.is_file():
        if lowered_contains and lowered_contains not in base.name.lower():
            return []
        return [str(base)] if base.suffix.lower() in lowered_suffixes else []

    files = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if lowered_contains and lowered_contains not in path.name.lower():
            continue
        if path.suffix.lower() in lowered_suffixes:
            files.append(str(path))
    return sorted(files)


def list_direct_image_files(root: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    if root.is_file():
        return [root] if root.suffix.lower() in suffixes else []
    if not root.exists():
        return []

    files = []
    for path in root.iterdir():
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        lowered = path.stem.lower()
        if lowered.endswith("_mask") or lowered.startswith("mask_"):
            continue
        files.append(path)
    return sorted(files)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or "asset"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10000):
        candidate = path.parent / f"{path.stem}_{index}{path.suffix}"
        if not candidate.exists():
            return candidate
    return path.parent / f"{path.stem}_{int(time.time())}{path.suffix}"


def asset_run_name(asset_path: str, input_path: str) -> str:
    path = Path(asset_path).expanduser().resolve()
    base = Path(input_path).expanduser().resolve()
    if base.is_file():
        base = base.parent
    try:
        relative = path.relative_to(base).with_suffix("")
    except ValueError:
        relative = Path(path.stem)
    return "__".join(safe_name(part) for part in relative.parts)


def cad_usd_output_path(cad_file: str, input_path: str, out_dir: str) -> str:
    input_root = Path(input_path).expanduser().resolve()
    cad_path = Path(cad_file).expanduser().resolve()
    out_root = Path(out_dir).expanduser()
    rel_stem = (
        Path(cad_path.stem)
        if input_root.is_file()
        else cad_path.relative_to(input_root).with_suffix("")
    )
    asset_dir = out_root / rel_stem.parent / rel_stem.name
    return str(asset_dir / f"{rel_stem.name}.usd")


def mirrored_output_parent(input_file: str, input_root: str, output_root: str) -> str:
    file_parent = Path(input_file).expanduser().resolve().parent
    root_path = Path(input_root).expanduser().resolve()
    try:
        relative_parent = file_parent.relative_to(root_path)
    except ValueError:
        relative_parent = Path(Path(input_file).stem)
    return str(Path(output_root).expanduser() / relative_parent)


def suffixed_file_path(input_file: str, output_dir: str, suffix: str) -> str:
    path = Path(input_file)
    return str(Path(output_dir).expanduser() / f"{path.stem}{suffix}{path.suffix}")


def converted_usd_root(input_path: str, *, overwrite: bool, suffix: str | None) -> str:
    base = Path(input_path)
    if base.is_file():
        out_dir = base.with_suffix("")
        if not overwrite:
            out_dir = out_dir.with_name(out_dir.name + (suffix or "_zup"))
        return str(out_dir)
    return input_path


def converted_usd_path(
    input_path: str,
    *,
    usd_format: str,
    overwrite: bool,
    suffix: str | None,
) -> str | None:
    base = Path(input_path)
    if not base.is_file():
        return None
    out_dir = Path(converted_usd_root(input_path, overwrite=overwrite, suffix=suffix))
    extension = f".{usd_format}" if usd_format in {"usdc", "usda"} else ".usd"
    return str(out_dir / f"{out_dir.name}{extension}")
