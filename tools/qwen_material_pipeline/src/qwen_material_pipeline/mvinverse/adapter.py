#!/usr/bin/env python3
"""Fail-closed, offline adapter for external MVInverse inference.

The adapter never downloads or bundles MVInverse code or weights.  It stages
reference images in manifest order, invokes an explicitly supplied Python
environment and repository, verifies the complete five-map output contract,
and writes a hash-bound JSON ledger suitable for unattended orchestration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import PIL
from PIL import Image, UnidentifiedImageError


SCHEMA_VERSION = "qwen-mvinverse-inference-ledger/v1"
FINGERPRINT_CONTRACT = "qwen-mvinverse-content-fingerprint/v2"
LEGACY_FINGERPRINT_CONTRACT = "qwen-mvinverse-manifest-fingerprint/v1"
LICENSE_URL = "https://github.com/Maddog241/mvinverse/blob/main/LICENSE"
LICENSE_NOTICE = "MVInverse repository LICENSE: non-commercial purposes"
DEFAULT_MAX_SIDE = 448
DEFAULT_OOM_RETRY_MAX_SIDES = (392,)
PATCH_MULTIPLE = 14
MAX_UPSTREAM_SIDE = 1024
MAP_MODES = {
    "albedo": "RGB",
    "metallic": "L",
    "roughness": "L",
    "normal": "RGB",
    "shading": "RGB",
}
EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 2
EXIT_INFERENCE_ERROR = 3

_VIEW_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_DEVICE = re.compile(r"(?:cpu|cuda(?::[0-9]+)?)")
_OOM_MARKERS = (
    "cuda out of memory",
    "torch.cuda.outofmemoryerror",
    "cublas_status_alloc_failed",
)
_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


class MVInverseAdapterError(RuntimeError):
    """Base error for the external inference trust boundary."""


class MVInverseInputError(MVInverseAdapterError):
    """Raised before, or instead of, model execution for invalid evidence."""


class MVInverseExecutionError(MVInverseAdapterError):
    """Raised when inference or its output contract fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise MVInverseInputError(f"Unable to hash {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_ledger_atomic(path: Path, ledger: Mapping[str, Any]) -> None:
    base = path.parent.resolve()

    def portable(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        if isinstance(value, tuple):
            return [portable(item) for item in value]
        if isinstance(value, str):
            candidate = Path(value).expanduser()
            if candidate.is_absolute():
                return Path(os.path.relpath(candidate, start=base)).as_posix()
        return value

    _write_text_atomic(
        path, json.dumps(portable(dict(ledger)), ensure_ascii=False, indent=2) + "\n"
    )


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MVInverseInputError(f"Unable to read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MVInverseInputError(f"{label} must be a JSON object: {path}")
    return value


def _resolve_file(value: str | Path, label: str, *, executable: bool = False) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise MVInverseInputError(f"{label} does not exist: {value}") from exc
    if not path.is_file():
        raise MVInverseInputError(f"{label} must be a file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise MVInverseInputError(f"{label} is not executable: {path}")
    return path


def _resolve_directory(value: str | Path, label: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise MVInverseInputError(f"{label} does not exist: {value}") from exc
    if not path.is_dir():
        raise MVInverseInputError(f"{label} must be a directory: {path}")
    return path


def _resolve_checkpoint(value: str | Path) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise MVInverseInputError(
            f"MVInverse checkpoint does not exist: {value}"
        ) from exc
    if not path.is_file() and not path.is_dir():
        raise MVInverseInputError(
            f"MVInverse checkpoint is not a file or directory: {path}"
        )
    return path


def _load_source_views(
    manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = _read_object(manifest_path, "reference manifest")
    raw_views = document.get("source_views")
    if not isinstance(raw_views, list) or not raw_views:
        raise MVInverseInputError(
            "reference_manifest.source_views must be a non-empty array"
        )
    if len(raw_views) > 999:
        raise MVInverseInputError("MVInverse adapter supports at most 999 source views")

    views: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_content: dict[str, str] = {}
    for index, raw in enumerate(raw_views):
        if not isinstance(raw, dict):
            raise MVInverseInputError(f"source_views[{index}] must be an object")
        view_id = raw.get("id")
        if not isinstance(view_id, str) or not _VIEW_ID.fullmatch(view_id):
            raise MVInverseInputError(
                f"source_views[{index}].id must be a safe, non-empty view identifier"
            )
        if view_id in seen_ids:
            raise MVInverseInputError(f"Duplicate source view id: {view_id}")
        seen_ids.add(view_id)
        image_value = raw.get("image")
        if not isinstance(image_value, str) or not image_value.strip():
            raise MVInverseInputError(
                f"source view {view_id}.image must be a path string"
            )
        image_path = Path(image_value).expanduser()
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        image_path = _resolve_file(image_path, f"source view {view_id} image")
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise MVInverseInputError(
                f"source view {view_id} must be PNG or JPEG: {image_path}"
            )
        digest = _sha256(image_path)
        if digest in seen_content:
            raise MVInverseInputError(
                f"source views {seen_content[digest]!r} and {view_id!r} have identical content"
            )
        seen_content[digest] = view_id
        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
                image_format = image.format
        except (OSError, UnidentifiedImageError) as exc:
            raise MVInverseInputError(
                f"Unable to decode source view {view_id}: {image_path}: {exc}"
            ) from exc
        if width < PATCH_MULTIPLE or height < PATCH_MULTIPLE:
            raise MVInverseInputError(
                f"source view {view_id} is smaller than {PATCH_MULTIPLE}px: {width}x{height}"
            )
        views.append(
            {
                "index": index,
                "view_id": view_id,
                "source_path": image_path,
                "source_sha256": digest,
                "source_size": (width, height),
                "source_format": image_format,
            }
        )
    return document, views


def _processed_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    if max(width, height) > max_side:
        scale = max_side / max(width, height)
        width = int(width * scale)
        height = int(height * scale)
    width = width // PATCH_MULTIPLE * PATCH_MULTIPLE
    height = height // PATCH_MULTIPLE * PATCH_MULTIPLE
    if width < PATCH_MULTIPLE or height < PATCH_MULTIPLE:
        raise MVInverseInputError("Preprocessing produced a zero-sized patch grid")
    return width, height


def _normalize_max_sides(
    max_side: int, oom_retry_max_sides: Sequence[int]
) -> tuple[int, ...]:
    if isinstance(max_side, bool) or not isinstance(max_side, int):
        raise MVInverseInputError("max_side must be an integer")
    if not PATCH_MULTIPLE <= max_side <= MAX_UPSTREAM_SIDE:
        raise MVInverseInputError(
            f"max_side must be between {PATCH_MULTIPLE} and {MAX_UPSTREAM_SIDE}"
        )
    result = [max_side]
    for index, value in enumerate(oom_retry_max_sides):
        if isinstance(value, bool) or not isinstance(value, int):
            raise MVInverseInputError(
                f"oom_retry_max_sides[{index}] must be an integer"
            )
        if value < PATCH_MULTIPLE or value >= result[-1]:
            raise MVInverseInputError(
                "OOM retry max-side values must be strictly descending and at least 14"
            )
        result.append(value)
    return tuple(result)


def _stage_inputs(
    source_views: Sequence[dict[str, Any]], output_dir: Path, max_side: int
) -> dict[str, Any]:
    sizes = [
        _processed_size(view["source_size"][0], view["source_size"][1], max_side)
        for view in source_views
    ]
    if len(set(sizes)) != 1:
        details = ", ".join(
            f"{view['view_id']}={size[0]}x{size[1]}"
            for view, size in zip(source_views, sizes)
        )
        raise MVInverseInputError(
            "MVInverse stacks all views into one tensor; preprocessed sizes differ: "
            + details
        )
    common_size = sizes[0]
    input_dir = output_dir / f"inputs_{max_side:04d}"
    temporary = output_dir / f".inputs_{max_side:04d}.tmp"
    if temporary.exists():
        raise MVInverseInputError(
            f"Stale temporary input directory exists; inspect and remove it: {temporary}"
        )
    temporary.mkdir(parents=False, exist_ok=False)
    generated: list[dict[str, Any]] = []
    try:
        for view, size in zip(source_views, sizes):
            filename = f"{view['index']:06d}.png"
            temporary_path = temporary / filename
            try:
                with Image.open(view["source_path"]) as source:
                    image = source.convert("RGB")
                    if image.size != size:
                        image = image.resize(size, Image.Resampling.BICUBIC)
                    image.save(temporary_path, format="PNG", compress_level=6)
            except (OSError, UnidentifiedImageError) as exc:
                raise MVInverseInputError(
                    f"Unable to stage source view {view['view_id']}: {exc}"
                ) from exc
            generated.append(
                {
                    "index": view["index"],
                    "view_id": view["view_id"],
                    "filename": filename,
                    "path": str(input_dir / filename),
                    "sha256": _sha256(temporary_path),
                    "size": list(size),
                    "mode": "RGB",
                }
            )

        expected = {record["filename"]: record["sha256"] for record in generated}
        if input_dir.exists():
            if not input_dir.is_dir():
                raise MVInverseInputError(
                    f"Staged input path is not a directory: {input_dir}"
                )
            actual_paths = sorted(
                path for path in input_dir.iterdir() if path.is_file()
            )
            actual = {path.name: _sha256(path) for path in actual_paths}
            if actual != expected or any(path.is_dir() for path in input_dir.iterdir()):
                raise MVInverseInputError(
                    f"Existing staged inputs do not match current evidence: {input_dir}"
                )
            shutil.rmtree(temporary)
        else:
            temporary.replace(input_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return {
        "max_side": max_side,
        "common_size": list(common_size),
        "input_directory": str(input_dir),
        "images": generated,
        "input_set_sha256": _canonical_sha256(
            [
                {"filename": item["filename"], "sha256": item["sha256"]}
                for item in generated
            ]
        ),
    }


def _git_metadata(repo: Path) -> dict[str, Any]:
    def packaged_revision() -> dict[str, Any] | None:
        revision_file = repo / "REVISION"
        if not revision_file.is_file():
            return None
        pinned_revision = revision_file.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[0-9a-fA-F]{40}", pinned_revision):
            return None
        return {
            "git_revision": pinned_revision.lower(),
            "tracked_worktree_dirty": False,
        }

    # ``git -C`` walks up to a parent repository.  A vendored snapshot without
    # its own .git directory must therefore be handled before invoking git, or
    # its provenance would be incorrectly reported as the host project's HEAD.
    if not (repo / ".git").exists():
        return packaged_revision() or {
            "git_revision": None,
            "tracked_worktree_dirty": None,
        }

    def command(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    try:
        revision = command("rev-parse", "HEAD")
        status = command("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.TimeoutExpired):
        return {"git_revision": None, "tracked_worktree_dirty": None}
    if revision.returncode != 0:
        # A locally vendored inference-only snapshot intentionally has no
        # nested .git directory.  Preserve the audited upstream revision in a
        # plain REVISION file so offline packaging stays small and provenance
        # remains explicit.
        return packaged_revision() or {
            "git_revision": None,
            "tracked_worktree_dirty": None,
        }
    return {
        "git_revision": revision.stdout.strip(),
        "tracked_worktree_dirty": (
            bool(status.stdout.strip()) if status.returncode == 0 else None
        ),
    }


def _checkpoint_metadata(
    checkpoint: Path, model_revision: str | None
) -> dict[str, Any]:
    if checkpoint.is_dir():
        config = checkpoint / "config.json"
        weights = sorted(checkpoint.glob("*.safetensors")) + sorted(
            checkpoint.glob("pytorch_model*.bin")
        )
        if not config.is_file() or not weights:
            raise MVInverseInputError(
                "Checkpoint directory must contain config.json and local safetensors/bin weights"
            )
        files = sorted(
            path
            for path in checkpoint.iterdir()
            if path.is_file() and not path.name.startswith(".")
        )
        records = [
            {
                "relative_path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ]
        if any(record["size_bytes"] <= 0 for record in records):
            raise MVInverseInputError(
                "Checkpoint directory contains an empty top-level file"
            )
        checkpoint_hash = _canonical_sha256(records)
        checkpoint_format = "huggingface_directory"
    elif checkpoint.is_file():
        if checkpoint.suffix.lower() not in {".pt", ".pth", ".ckpt"}:
            raise MVInverseInputError(
                "Checkpoint file must be a torch-loadable .pt/.pth/.ckpt file; "
                "pass the parent directory for Hugging Face safetensors"
            )
        size = checkpoint.stat().st_size
        if size <= 0:
            raise MVInverseInputError("Checkpoint file is empty")
        checkpoint_hash = _sha256(checkpoint)
        records = [
            {
                "relative_path": checkpoint.name,
                "size_bytes": size,
                "sha256": checkpoint_hash,
            }
        ]
        checkpoint_format = "torch_checkpoint"
    else:
        raise MVInverseInputError(
            f"Checkpoint is neither a file nor a directory: {checkpoint}"
        )

    if model_revision is not None:
        model_revision = model_revision.strip()
        if not model_revision:
            raise MVInverseInputError("model_revision cannot be blank")
    return {
        "path": str(checkpoint),
        "format": checkpoint_format,
        "files": records,
        "checkpoint_sha256": checkpoint_hash,
        "declared_revision": model_revision,
        "effective_revision": model_revision or f"sha256:{checkpoint_hash}",
    }


def _validate_repository(repo: Path) -> dict[str, Any]:
    module = repo / "mvinverse" / "models" / "mvinverse.py"
    license_path = repo / "LICENSE"
    if not module.is_file():
        raise MVInverseInputError(
            f"MVInverse module is missing from repository: {module}"
        )
    if not license_path.is_file():
        raise MVInverseInputError(f"MVInverse LICENSE is missing: {license_path}")
    try:
        license_text = license_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MVInverseInputError(f"Unable to read MVInverse LICENSE: {exc}") from exc
    if "non-commercial" not in license_text.lower():
        raise MVInverseInputError(
            "External repository LICENSE does not contain the expected non-commercial notice"
        )
    return {
        "path": str(repo),
        "model_module": str(module.resolve(strict=True)),
        "model_module_sha256": _sha256(module),
        "license_path": str(license_path.resolve(strict=True)),
        "license_sha256": _sha256(license_path),
        **_git_metadata(repo),
    }


def _command(
    python: Path,
    runner_script: Path,
    repo: Path,
    staged: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    temporary_output: Path,
    device: str,
    frame_count: int,
) -> list[str]:
    return [
        str(python),
        str(runner_script),
        "--repo",
        str(repo),
        "--input-dir",
        str(staged["input_directory"]),
        "--checkpoint",
        str(checkpoint["path"]),
        "--checkpoint-format",
        str(checkpoint["format"]),
        "--output-dir",
        str(temporary_output),
        "--device",
        device,
        "--num-frames",
        str(frame_count),
    ]


def _validate_outputs(
    directory: Path,
    source_views: Sequence[Mapping[str, Any]],
    expected_size: Sequence[int],
) -> dict[str, Any]:
    if not directory.is_dir():
        raise MVInverseExecutionError(
            f"MVInverse output directory is missing: {directory}"
        )
    expected_names = {
        f"{view['index']:03d}_{map_name}.png"
        for view in source_views
        for map_name in MAP_MODES
    }
    actual_entries = list(directory.iterdir())
    if any(not path.is_file() for path in actual_entries):
        raise MVInverseExecutionError(
            "MVInverse output directory contains a subdirectory"
        )
    actual_names = {path.name for path in actual_entries}
    if actual_names != expected_names:
        raise MVInverseExecutionError(
            "MVInverse output set is incomplete or contaminated; "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    records: list[dict[str, Any]] = []
    size = tuple(int(value) for value in expected_size)
    for view in source_views:
        for map_name, expected_mode in MAP_MODES.items():
            path = directory / f"{view['index']:03d}_{map_name}.png"
            try:
                with Image.open(path) as image:
                    image.load()
                    actual_size = image.size
                    actual_mode = image.mode
            except (OSError, UnidentifiedImageError) as exc:
                raise MVInverseExecutionError(
                    f"Unable to decode output {path}: {exc}"
                ) from exc
            if actual_size != size:
                raise MVInverseExecutionError(
                    f"Output {path.name} size {actual_size} does not match {size}"
                )
            if actual_mode != expected_mode:
                raise MVInverseExecutionError(
                    f"Output {path.name} mode {actual_mode!r} must be {expected_mode!r}"
                )
            records.append(
                {
                    "index": view["index"],
                    "view_id": view["view_id"],
                    "map": map_name,
                    "path": str(path),
                    "sha256": _sha256(path),
                    "size": list(actual_size),
                    "mode": actual_mode,
                }
            )
    return {
        "directory": str(directory),
        "map_count": len(records),
        "maps": records,
        "output_set_sha256": _canonical_sha256(
            [
                {
                    "index": item["index"],
                    "map": item["map"],
                    "sha256": item["sha256"],
                }
                for item in records
            ]
        ),
    }


def _revalidate_inference_inputs(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    source_views: Sequence[Mapping[str, Any]],
    repository: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_record: Mapping[str, Any],
    runner_script: Path,
    runner_script_sha256: str,
    staged: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject a run if evidence, code, weights, or staged pixels changed in flight."""

    if _sha256(manifest_path) != manifest_sha256:
        raise MVInverseExecutionError("Reference manifest changed during inference")
    for view in source_views:
        if _sha256(Path(view["source_path"])) != view["source_sha256"]:
            raise MVInverseExecutionError(
                f"Source view changed during inference: {view['view_id']}"
            )
    current_repository = _validate_repository(Path(repository["path"]))
    for field in ("git_revision", "model_module_sha256", "license_sha256"):
        if current_repository.get(field) != repository.get(field):
            raise MVInverseExecutionError(
                f"MVInverse repository {field} changed during inference"
            )
    if _sha256(runner_script) != runner_script_sha256:
        raise MVInverseExecutionError(
            "MVInverse runner script changed during inference"
        )
    current_checkpoint = _checkpoint_metadata(
        checkpoint_path, checkpoint_record.get("declared_revision")
    )
    if (
        current_checkpoint["checkpoint_sha256"]
        != checkpoint_record["checkpoint_sha256"]
    ):
        raise MVInverseExecutionError("MVInverse checkpoint changed during inference")

    staged_directory = Path(staged["input_directory"])
    expected = {item["filename"]: item["sha256"] for item in staged["images"]}
    if not staged_directory.is_dir():
        raise MVInverseExecutionError(
            "Staged MVInverse inputs disappeared during inference"
        )
    entries = list(staged_directory.iterdir())
    if any(not path.is_file() for path in entries):
        raise MVInverseExecutionError(
            "Staged MVInverse input directory was contaminated"
        )
    actual = {path.name: _sha256(path) for path in entries}
    if actual != expected:
        raise MVInverseExecutionError(
            "Staged MVInverse inputs changed during inference"
        )
    return {
        "status": "PASS",
        "manifest_sha256_after": manifest_sha256,
        "source_set_sha256_after": _canonical_sha256(
            [
                {
                    "index": view["index"],
                    "view_id": view["view_id"],
                    "sha256": view["source_sha256"],
                }
                for view in source_views
            ]
        ),
        "repository_revision_after": current_repository["git_revision"],
        "checkpoint_sha256_after": current_checkpoint["checkpoint_sha256"],
        "staged_input_set_sha256_after": staged["input_set_sha256"],
    }


def _write_process_log(path: Path, content: str | None) -> dict[str, Any]:
    _write_text_atomic(path, content or "")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _is_cuda_oom(stdout: str | None, stderr: str | None) -> bool:
    combined = f"{stdout or ''}\n{stderr or ''}".lower()
    return any(marker in combined for marker in _OOM_MARKERS)


def _last_json_object(content: str | None) -> dict[str, Any] | None:
    if not content:
        return None
    for line in reversed(content.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def run_mvinverse_adapter(
    *,
    reference_manifest: str | Path,
    repo: str | Path,
    python_executable: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    acknowledge_noncommercial: bool,
    model_revision: str | None = None,
    device: str = "cuda",
    max_side: int = DEFAULT_MAX_SIDE,
    oom_retry_max_sides: Sequence[int] | None = None,
    reuse_existing: bool = False,
    dry_run: bool = False,
    timeout_seconds: int = 1800,
    runner_script: str | Path | None = None,
    process_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Run or safely reuse one external MVInverse inference transaction."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    ledger_path = destination / "mvinverse_inference_ledger.json"
    final_maps = destination / "maps"
    temporary_maps = destination / ".maps.tmp"
    phase = "license"
    ledger: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PLANNING",
        "fail_closed": True,
        "started_at": _utc_now(),
        "license": {
            "notice": LICENSE_NOTICE,
            "url": LICENSE_URL,
            "acknowledged_noncommercial_use": bool(acknowledge_noncommercial),
        },
        "execution": {"attempts": []},
    }
    previous_ledger: dict[str, Any] | None = None
    previous_ledger_sha256: str | None = None

    try:
        if not acknowledge_noncommercial:
            raise MVInverseInputError(
                "MVInverse non-commercial LICENSE must be explicitly acknowledged"
            )
        if (
            not isinstance(timeout_seconds, int)
            or isinstance(timeout_seconds, bool)
            or timeout_seconds < 1
        ):
            raise MVInverseInputError("timeout_seconds must be a positive integer")
        if not isinstance(device, str) or not _DEVICE.fullmatch(device):
            raise MVInverseInputError("device must be cpu, cuda, or cuda:N")
        retries = (
            tuple(value for value in DEFAULT_OOM_RETRY_MAX_SIDES if value < max_side)
            if oom_retry_max_sides is None
            else tuple(oom_retry_max_sides)
        )
        max_sides = _normalize_max_sides(max_side, retries)

        phase = "resolve_inputs"
        manifest_path = _resolve_file(reference_manifest, "reference manifest")
        repo_path = _resolve_directory(repo, "MVInverse repository")
        python_path = _resolve_file(
            python_executable, "MVInverse Python", executable=True
        )
        checkpoint_path = _resolve_checkpoint(checkpoint)
        selected_runner = _resolve_file(
            runner_script or Path(__file__).with_name("runner.py"),
            "MVInverse runner script",
        )
        _manifest, source_views = _load_source_views(manifest_path)
        repository = _validate_repository(repo_path)
        checkpoint_record = _checkpoint_metadata(checkpoint_path, model_revision)
        ledger["inputs"] = {
            "reference_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
            },
            "order_contract": "reference_manifest.source_views array order",
            "source_views": [
                {
                    "index": view["index"],
                    "view_id": view["view_id"],
                    "path": str(view["source_path"]),
                    "sha256": view["source_sha256"],
                    "size": list(view["source_size"]),
                    "format": view["source_format"],
                }
                for view in source_views
            ],
            "source_set_sha256": _canonical_sha256(
                [
                    {
                        "index": view["index"],
                        "view_id": view["view_id"],
                        "sha256": view["source_sha256"],
                    }
                    for view in source_views
                ]
            ),
        }
        ledger["model"] = {
            "repository": repository,
            "checkpoint": checkpoint_record,
            "python": str(python_path),
            "runner_script": str(selected_runner),
            "runner_script_sha256": _sha256(selected_runner),
        }

        phase = "stage_inputs"
        staged_attempts = [
            _stage_inputs(source_views, destination, candidate)
            for candidate in max_sides
        ]
        ledger["preprocessing"] = {
            "method": "RGB + Pillow bicubic resize + floor dimensions to patch multiple",
            "pillow_version": PIL.__version__,
            "requested_max_side": max_side,
            "patch_multiple": PATCH_MULTIPLE,
            "oom_retry_max_sides": list(max_sides[1:]),
            "attempts": staged_attempts,
        }
        commands = [
            _command(
                python_path,
                selected_runner,
                repo_path,
                staged,
                checkpoint_record,
                temporary_maps,
                device,
                len(source_views),
            )
            for staged in staged_attempts
        ]
        # MVInverse consumes only the ordered source images.  Bind reuse to
        # their ids/content and all executable/model/preprocessing inputs, but
        # not to the serialized manifest bytes: absolute image paths change
        # when a verified checkpoint is copied to a new version directory.
        # The source_set and staged hashes below still fail closed if even one
        # input pixel, view id/order, or resize result changes.
        fingerprint_document = {
            "fingerprint_contract": FINGERPRINT_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "source_set_sha256": ledger["inputs"]["source_set_sha256"],
            "repository_revision": repository["git_revision"],
            "repository_module_sha256": repository["model_module_sha256"],
            "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
            "model_revision": checkpoint_record["effective_revision"],
            "runner_script_sha256": ledger["model"]["runner_script_sha256"],
            "python": str(python_path),
            "device": device,
            "max_sides": list(max_sides),
            "staged_input_sha256": [
                item["input_set_sha256"] for item in staged_attempts
            ],
        }
        run_fingerprint = _canonical_sha256(fingerprint_document)
        ledger["fingerprint_contract"] = FINGERPRINT_CONTRACT
        ledger["run_fingerprint"] = run_fingerprint
        ledger["execution"].update(
            {
                "device": device,
                "timeout_seconds": timeout_seconds,
                "dry_run": bool(dry_run),
                "reuse_existing_requested": bool(reuse_existing),
                "cwd": str(repo_path),
                "offline_environment": dict(_OFFLINE_ENV),
                "planned_commands": commands,
                "retry_policy": "retry only a recognized CUDA OOM at the next fixed max-side",
            }
        )

        if ledger_path.exists():
            previous_ledger_sha256 = _sha256(ledger_path)
            previous_ledger = _read_object(ledger_path, "previous MVInverse ledger")
            ledger["previous_ledger_sha256"] = previous_ledger_sha256

        phase = "select_mode"
        if dry_run:
            if final_maps.exists():
                raise MVInverseInputError(
                    "Cannot replace a completed inference ledger with dry-run; "
                    "use a new output directory"
                )
            ledger["status"] = "DRY_RUN"
            ledger["execution"]["status"] = "NOT_EXECUTED"
            ledger["outputs"] = None
            ledger["finished_at"] = _utc_now()
            _write_ledger_atomic(ledger_path, ledger)
            return ledger

        if reuse_existing:
            if previous_ledger is None:
                raise MVInverseInputError(
                    "Reuse requires an existing successful ledger"
                )
            if previous_ledger.get("schema_version") != SCHEMA_VERSION:
                raise MVInverseInputError(
                    "Existing MVInverse ledger has an unsupported schema"
                )
            if previous_ledger.get("status") not in {"SUCCESS", "REUSED"}:
                raise MVInverseInputError(
                    "Reuse requires existing SUCCESS or REUSED status"
                )
            previous_fingerprint = previous_ledger.get("run_fingerprint")
            fingerprint_match = previous_fingerprint == run_fingerprint
            legacy_relocation_match = False
            if not fingerprint_match:
                previous_inputs = previous_ledger.get("inputs")
                previous_manifest = (
                    previous_inputs.get("reference_manifest")
                    if isinstance(previous_inputs, dict)
                    else None
                )
                previous_manifest_sha256 = (
                    previous_manifest.get("sha256")
                    if isinstance(previous_manifest, dict)
                    else None
                )
                if isinstance(previous_manifest_sha256, str):
                    legacy_fingerprint_document = {
                        key: value
                        for key, value in fingerprint_document.items()
                        if key != "fingerprint_contract"
                    }
                    legacy_fingerprint_document["manifest_sha256"] = (
                        previous_manifest_sha256
                    )
                    legacy_relocation_match = (
                        previous_fingerprint
                        == _canonical_sha256(legacy_fingerprint_document)
                    )
                    fingerprint_match = legacy_relocation_match
            if not fingerprint_match:
                raise MVInverseInputError(
                    "Existing MVInverse ledger fingerprint does not match this run"
                )
            prior_outputs = previous_ledger.get("outputs")
            if not isinstance(prior_outputs, dict):
                raise MVInverseInputError(
                    "Existing MVInverse ledger has no output records"
                )
            selected_max_side = prior_outputs.get("preprocessing_max_side")
            selected = next(
                (
                    item
                    for item in staged_attempts
                    if item["max_side"] == selected_max_side
                ),
                None,
            )
            if selected is None:
                raise MVInverseInputError(
                    "Existing output preprocessing size is not in retry schedule"
                )
            verified = _validate_outputs(
                final_maps, source_views, selected["common_size"]
            )
            if verified["output_set_sha256"] != prior_outputs.get("output_set_sha256"):
                raise MVInverseInputError(
                    "Existing MVInverse output hashes do not match its ledger"
                )
            verified["preprocessing_max_side"] = selected_max_side
            verified["preprocessed_size"] = selected["common_size"]
            ledger["status"] = "REUSED"
            ledger["execution"]["status"] = "REUSED_NOT_EXECUTED"
            ledger["execution"]["reused_from_ledger_sha256"] = previous_ledger_sha256
            ledger["execution"]["reuse_fingerprint_validation"] = {
                "status": "PASS",
                "contract": (
                    LEGACY_FINGERPRINT_CONTRACT
                    if legacy_relocation_match
                    else FINGERPRINT_CONTRACT
                ),
                "relocation_compatible": True,
                "source_set_sha256": ledger["inputs"]["source_set_sha256"],
            }
            ledger["outputs"] = verified
            ledger["finished_at"] = _utc_now()
            _write_ledger_atomic(ledger_path, ledger)
            return ledger

        if final_maps.exists():
            raise MVInverseInputError(
                "MVInverse outputs already exist; pass --reuse-existing or use a new "
                "output directory"
            )
        if previous_ledger is not None and previous_ledger.get("status") in {
            "SUCCESS",
            "REUSED",
        }:
            raise MVInverseInputError(
                "Successful ledger exists but its maps directory is missing"
            )
        if temporary_maps.exists():
            raise MVInverseInputError(
                f"Stale temporary output exists; inspect and remove it: {temporary_maps}"
            )

        phase = "inference"
        environment = os.environ.copy()
        environment.update(_OFFLINE_ENV)
        selected_staged: dict[str, Any] | None = None
        for attempt_index, (staged, command) in enumerate(
            zip(staged_attempts, commands), start=1
        ):
            attempt_record: dict[str, Any] = {
                "attempt": attempt_index,
                "max_side": staged["max_side"],
                "preprocessed_size": staged["common_size"],
                "command": command,
                "status": "RUNNING",
            }
            ledger["execution"]["attempts"].append(attempt_record)
            attempt_started = time.monotonic()
            try:
                completed = process_runner(
                    command,
                    cwd=str(repo_path),
                    env=environment,
                    text=True,
                    capture_output=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                attempt_record["elapsed_seconds"] = round(
                    time.monotonic() - attempt_started, 6
                )
                attempt_record["status"] = "TIMEOUT"
                attempt_record["timeout_seconds"] = timeout_seconds
                stdout = (
                    exc.stdout.decode(errors="replace")
                    if isinstance(exc.stdout, bytes)
                    else exc.stdout
                )
                stderr = (
                    exc.stderr.decode(errors="replace")
                    if isinstance(exc.stderr, bytes)
                    else exc.stderr
                )
                attempt_record["stdout"] = _write_process_log(
                    destination / f"attempt_{attempt_index:02d}.stdout.log", stdout
                )
                attempt_record["stderr"] = _write_process_log(
                    destination / f"attempt_{attempt_index:02d}.stderr.log", stderr
                )
                raise MVInverseExecutionError(
                    f"MVInverse inference timed out after {timeout_seconds}s"
                ) from exc
            except OSError as exc:
                attempt_record["elapsed_seconds"] = round(
                    time.monotonic() - attempt_started, 6
                )
                attempt_record["status"] = "LAUNCH_FAILED"
                raise MVInverseExecutionError(
                    f"Unable to launch MVInverse: {exc}"
                ) from exc

            stdout_log = destination / f"attempt_{attempt_index:02d}.stdout.log"
            stderr_log = destination / f"attempt_{attempt_index:02d}.stderr.log"
            attempt_record["stdout"] = _write_process_log(stdout_log, completed.stdout)
            attempt_record["stderr"] = _write_process_log(stderr_log, completed.stderr)
            attempt_record["returncode"] = completed.returncode
            attempt_record["elapsed_seconds"] = round(
                time.monotonic() - attempt_started, 6
            )
            telemetry = _last_json_object(completed.stdout)
            if telemetry is not None:
                attempt_record["runner_telemetry"] = telemetry
            oom = _is_cuda_oom(completed.stdout, completed.stderr)
            attempt_record["cuda_oom_detected"] = oom
            if completed.returncode == 0:
                attempt_record["status"] = "SUCCESS"
                selected_staged = staged
                break
            attempt_record["status"] = "CUDA_OOM" if oom else "FAILED"
            if temporary_maps.exists():
                if temporary_maps.is_dir():
                    shutil.rmtree(temporary_maps)
                else:
                    temporary_maps.unlink()
            if not oom or attempt_index == len(staged_attempts):
                stderr_tail = (completed.stderr or "").strip()[-500:]
                raise MVInverseExecutionError(
                    "MVInverse inference failed with exit code "
                    f"{completed.returncode}: {stderr_tail}"
                )

        if selected_staged is None:  # Defensive; loop either succeeds or raises.
            raise MVInverseExecutionError("MVInverse produced no successful attempt")

        phase = "verify_outputs"
        _validate_outputs(temporary_maps, source_views, selected_staged["common_size"])
        phase = "revalidate_inputs"
        ledger["input_revalidation"] = _revalidate_inference_inputs(
            manifest_path=manifest_path,
            manifest_sha256=ledger["inputs"]["reference_manifest"]["sha256"],
            source_views=source_views,
            repository=repository,
            checkpoint_path=checkpoint_path,
            checkpoint_record=checkpoint_record,
            runner_script=selected_runner,
            runner_script_sha256=ledger["model"]["runner_script_sha256"],
            staged=selected_staged,
        )
        phase = "publish_outputs"
        temporary_maps.replace(final_maps)
        verified = _validate_outputs(
            final_maps, source_views, selected_staged["common_size"]
        )
        verified["preprocessing_max_side"] = selected_staged["max_side"]
        verified["preprocessed_size"] = selected_staged["common_size"]
        ledger["status"] = "SUCCESS"
        ledger["execution"]["status"] = "SUCCESS"
        ledger["outputs"] = verified
        ledger["finished_at"] = _utc_now()
        _write_ledger_atomic(ledger_path, ledger)
        return ledger
    except Exception as exc:
        if temporary_maps.exists():
            try:
                if temporary_maps.is_dir():
                    shutil.rmtree(temporary_maps)
                else:
                    temporary_maps.unlink()
            except OSError:
                pass
        ledger["status"] = "FAILED"
        ledger["failure"] = {
            "phase": phase,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        ledger["finished_at"] = _utc_now()
        try:
            _write_ledger_atomic(ledger_path, ledger)
        except OSError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--repo", required=True, help="external MVInverse git checkout")
    parser.add_argument("--python", dest="python_executable", required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="local HF directory or torch-loadable .pt/.pth/.ckpt; never a Hub repo id",
    )
    parser.add_argument(
        "--model-revision", help="caller-declared immutable weight revision"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE)
    retry = parser.add_mutually_exclusive_group()
    retry.add_argument(
        "--oom-retry-max-side",
        type=int,
        action="append",
        help="fixed smaller max-side used only after recognized CUDA OOM; repeatable",
    )
    retry.add_argument("--no-oom-retry", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--acknowledge-mvinverse-noncommercial",
        "--acknowledge-noncommercial-license",
        dest="acknowledge_noncommercial",
        action="store_true",
        help="confirm this run is permitted under MVInverse's non-commercial license",
    )
    parser.add_argument(
        "--runner-script",
        help="advanced: compatible offline runner (default: bundled adapter runner)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.no_oom_retry:
        retries: Sequence[int] = ()
    elif args.oom_retry_max_side is None:
        retries = tuple(
            value for value in DEFAULT_OOM_RETRY_MAX_SIDES if value < args.max_side
        )
    else:
        retries = tuple(args.oom_retry_max_side)
    try:
        ledger = run_mvinverse_adapter(
            reference_manifest=args.reference_manifest,
            repo=args.repo,
            python_executable=args.python_executable,
            checkpoint=args.checkpoint,
            model_revision=args.model_revision,
            output_dir=args.output_dir,
            acknowledge_noncommercial=args.acknowledge_noncommercial,
            device=args.device,
            max_side=args.max_side,
            oom_retry_max_sides=retries,
            reuse_existing=args.reuse_existing,
            dry_run=args.dry_run,
            timeout_seconds=args.timeout_seconds,
            runner_script=args.runner_script,
        )
    except MVInverseInputError as exc:
        print(
            json.dumps(
                {"status": "INPUT_ERROR", "error": str(exc)}, ensure_ascii=False
            ),
            file=sys.stderr,
        )
        return EXIT_INPUT_ERROR
    except (MVInverseExecutionError, OSError, subprocess.SubprocessError) as exc:
        print(
            json.dumps(
                {"status": "INFERENCE_ERROR", "error": str(exc)}, ensure_ascii=False
            ),
            file=sys.stderr,
        )
        return EXIT_INFERENCE_ERROR
    print(
        json.dumps(
            {
                "status": ledger["status"],
                "ledger": str(
                    Path(args.output_dir).expanduser().resolve()
                    / "mvinverse_inference_ledger.json"
                ),
                "run_fingerprint": ledger["run_fingerprint"],
            },
            ensure_ascii=False,
        )
    )
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MAX_SIDE",
    "DEFAULT_OOM_RETRY_MAX_SIDES",
    "EXIT_INFERENCE_ERROR",
    "EXIT_INPUT_ERROR",
    "EXIT_SUCCESS",
    "MVInverseAdapterError",
    "MVInverseExecutionError",
    "MVInverseInputError",
    "SCHEMA_VERSION",
    "build_parser",
    "main",
    "run_mvinverse_adapter",
]
