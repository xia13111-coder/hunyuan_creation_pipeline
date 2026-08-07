"""Small fail-closed helpers shared by visual-material stages."""

from pathlib import Path


def require_file(path: Path, stage: str) -> Path:
    if not path.is_file():
        raise RuntimeError(f"{stage} did not create expected file: {path}")
    return path.resolve(strict=True)


__all__ = ["require_file"]
