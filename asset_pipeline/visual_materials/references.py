"""Reference-image validation and content identity helpers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Sequence


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
REFERENCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_visual_references(
    values: Sequence[str],
) -> tuple[tuple[str, Path], ...]:
    """Resolve 2..4 unique ``[ID=]IMAGE`` reference specifications."""

    if isinstance(values, (str, bytes)):
        raise ValueError("Visual references must be a sequence, not one string")
    if not 2 <= len(values) <= 4:
        raise ValueError(
            "STEP/STP reference-image material assignment requires 2..4 photos "
            "of the same physical workpiece"
        )
    parsed: list[tuple[str, Path]] = []
    seen_ids: set[str] = set()
    seen_hashes: dict[str, Path] = {}
    for index, raw_value in enumerate(values, start=1):
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(f"Visual reference {index} must be a non-empty string")
        raw = raw_value.strip()
        if "=" in raw:
            reference_id, raw_path = raw.split("=", 1)
            reference_id = reference_id.strip()
        else:
            reference_id, raw_path = f"ref_{index:02d}", raw
        if not REFERENCE_ID.fullmatch(reference_id):
            raise ValueError(f"Invalid visual reference ID: {reference_id!r}")
        if reference_id in seen_ids:
            raise ValueError(f"Duplicate visual reference ID: {reference_id}")
        try:
            path = Path(raw_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise FileNotFoundError(
                f"Visual reference image does not exist: {raw_path}"
            ) from exc
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Visual reference must be a supported image: {path}")
        digest = sha256_file(path)
        if digest in seen_hashes:
            raise ValueError(
                "Visual references must be independent images; duplicate content: "
                f"{seen_hashes[digest]} and {path}"
            )
        seen_ids.add(reference_id)
        seen_hashes[digest] = path
        parsed.append((reference_id, path))
    return tuple(parsed)


__all__ = [
    "IMAGE_SUFFIXES",
    "REFERENCE_ID",
    "parse_visual_references",
    "sha256_file",
]
