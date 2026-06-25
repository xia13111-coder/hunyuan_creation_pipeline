from __future__ import annotations

import os
from pathlib import Path


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()
    if "=" not in stripped:
        return None

    key, raw_value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None

    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
        if raw_value.strip().startswith('"'):
            value = bytes(value, "utf-8").decode("unicode_escape")
    else:
        value = value.split(" #", 1)[0].strip()
    return key, value


def load_env_file(path: str | Path, *, override: bool = False) -> dict[str, str]:
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if not parsed:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def load_default_env_files(extra_paths: list[str] | None = None) -> list[Path]:
    project_root = Path(__file__).resolve().parents[1]
    candidates = [project_root / ".env", Path.cwd() / ".env"]
    candidates.extend(Path(path).expanduser() for path in extra_paths or [])

    loaded_paths: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        loaded = load_env_file(resolved)
        if loaded:
            loaded_paths.append(resolved)
    return loaded_paths
