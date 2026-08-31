#!/usr/bin/env python3
"""Retrieve NVIDIA MDLs with SigLIP2 and rerank masked texture with DINOv2.

SigLIP2 performs catalog-wide image/text retrieval.  DINOv2 never replaces
that index: it compares dense patch tokens only for the bounded SigLIP2
shortlist and only inside accepted SAM3 masks.  The result is a candidate
proposal for the existing exact-MDL render tournament, not a final material
decision.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# The production orchestrator intentionally launches this file directly with
# the retrieval environment.  Make the sibling package import deterministic
# without requiring users to export PYTHONPATH.
if __package__ in {None, ""}:
    package_root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if parent.name == "qwen_material_pipeline"
    )
    sys.path.insert(0, str(package_root.parent))

from qwen_material_pipeline.materials.perceptual_color import (
    perceptual_similarity,
)  # noqa: E402


REQUEST_SCHEMA_VERSION = "qwen-visual-material-retrieval-request/v1"
RESULT_SCHEMA_VERSION = "qwen-visual-material-retrieval-result/v1"
CACHE_SCHEMA_VERSION = "qwen-siglip2-mdl-index/v1"
BASE_BANK_INDEX_SCHEMA_VERSION = "nvidia-base-material-observation-index/v1"
BASE_BANK_SCOPE_SCHEMA_VERSION = "nvidia-base-material-scope/v1"
BASE_BANK_FUSION_POLICY = (
    "rrf_base_bank_siglip1.0_dino1.2_ciede2000_color0.8_mvinverse0.2_k60/v2"
)
LEGACY_FUSION_POLICY = "reciprocal_rank_fusion_siglip1.0_dino1.2_k60/v1"
BASE_BANK_RETRIEVAL_STRATEGY = (
    "base_observation_bank_siglip2_dinov2_ciede2000_mvinverse_rrf/v2"
)
LEGACY_RETRIEVAL_STRATEGY = "siglip2_full_catalog_plus_dinov2_masked_rrf/v1"
SIGLIP2_IDENTITY_SCHEMA_VERSION = "retrieval-local-checkpoint/v1"
SIGLIP2_CANONICAL_REPOSITORY = "google/siglip2-base-patch16-224"
SIGLIP2_CANONICAL_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
SIGLIP2_CONTENT_MANIFEST_SHA256 = (
    "300e35ccdc519b90ed4a4e502c1d5484dd95cb3b3fd7138638eebab86b281a43"
)
DEFAULT_SIGLIP_TOP_K = 64
DEFAULT_FINAL_TOP_K = 32
DEFAULT_BATCH_SIZE = 24
CANVAS_SIZE = 224
PROGRESS_PREFIX = "@@ASSET_PROGRESS "
PROGRESS_SCHEMA_VERSION = "asset-pipeline-progress/v1"


class VisualRetrievalError(ValueError):
    """Raised when visual retrieval cannot satisfy its frozen contract."""


def _emit_progress(
    *,
    stage: str,
    state: str,
    detail: str,
    current: int | None = None,
    total: int | None = None,
    unit: str | None = None,
) -> None:
    event = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "scope": "visual_retrieval",
        "stage": stage,
        "state": state,
        "current": current,
        "total": total,
        "unit": unit,
        "detail": detail,
    }
    print(
        PROGRESS_PREFIX
        + json.dumps(
            event,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        file=sys.stderr,
        flush=True,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VisualRetrievalError(f"unable to read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VisualRetrievalError(f"{label} must be a JSON object")
    return value


def _resolve_file(value: Any, *, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise VisualRetrievalError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise VisualRetrievalError(f"{label} does not exist: {path}") from exc
    if not resolved.is_file():
        raise VisualRetrievalError(f"{label} is not a file: {resolved}")
    return resolved


def _resolve_under_root(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise VisualRetrievalError(f"{label} must be a non-empty relative path")
    candidate = root.joinpath(*Path(raw).parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise VisualRetrievalError(
            f"{label} escapes the material root: {raw!r}"
        ) from exc
    if not resolved.is_file():
        raise VisualRetrievalError(f"{label} is not a file: {resolved}")
    return resolved


def _model_fingerprint(model_path: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    patterns = (
        "config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "model*.safetensors",
        "pytorch_model*.bin",
    )
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(model_path.glob(pattern)):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            files.append(
                {
                    "path": path.relative_to(model_path).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    if not files:
        raise VisualRetrievalError(
            f"model directory has no checkpoint files: {model_path}"
        )
    unsigned = {"path": str(model_path), "files": files}
    return {**unsigned, "fingerprint_sha256": _canonical_sha256(unsigned)}


def _verify_pinned_siglip2_identity(model_path: Path) -> dict[str, Any]:
    """Verify every local SigLIP2 runtime file against the trust anchor."""

    model_path = model_path.expanduser().resolve(strict=True)
    identity_path = model_path / "checkpoint_identity.json"
    if not identity_path.is_file() or identity_path.is_symlink():
        raise VisualRetrievalError(
            "SigLIP2 checkpoint has no regular checkpoint_identity.json; "
            "install it with scripts/qwen35/setup_qwen35_runtime.sh"
        )
    identity = _read_json(identity_path, "SigLIP2 checkpoint identity")
    runtime_files = identity.get("runtime_files")
    if (
        identity.get("schema_version") != SIGLIP2_IDENTITY_SCHEMA_VERSION
        or identity.get("repository") != SIGLIP2_CANONICAL_REPOSITORY
        or identity.get("revision") != SIGLIP2_CANONICAL_REVISION
        or identity.get("content_manifest_sha256") != SIGLIP2_CONTENT_MANIFEST_SHA256
        or not isinstance(runtime_files, list)
        or _canonical_sha256(runtime_files) != SIGLIP2_CONTENT_MANIFEST_SHA256
    ):
        raise VisualRetrievalError(
            "SigLIP2 checkpoint identity does not match the pinned canonical "
            "repository/revision/manifest"
        )

    records: dict[str, dict[str, Any]] = {}
    for raw in runtime_files:
        path_text = raw.get("path") if isinstance(raw, dict) else None
        size = raw.get("bytes") if isinstance(raw, dict) else None
        sha256 = raw.get("sha256") if isinstance(raw, dict) else None
        if (
            not isinstance(path_text, str)
            or not path_text
            or path_text in records
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        ):
            raise VisualRetrievalError(
                "SigLIP2 checkpoint identity contains an invalid runtime file record"
            )
        relative = Path(path_text)
        source_path = model_path / relative
        try:
            resolved = source_path.resolve(strict=True)
            resolved.relative_to(model_path)
        except (OSError, ValueError) as exc:
            raise VisualRetrievalError(
                f"SigLIP2 runtime file escapes the checkpoint: {path_text!r}"
            ) from exc
        if (
            relative.is_absolute()
            or source_path.is_symlink()
            or not resolved.is_file()
            or relative.as_posix() != path_text
        ):
            raise VisualRetrievalError(
                f"SigLIP2 runtime file is not a regular relative file: {path_text!r}"
            )
        records[path_text] = raw

    actual_files = {
        path.relative_to(model_path).as_posix()
        for path in model_path.rglob("*")
        if path.is_file() and path != identity_path
    }
    if actual_files != set(records):
        raise VisualRetrievalError(
            "SigLIP2 checkpoint runtime file set differs from the pinned manifest"
        )
    for path_text, record in records.items():
        path = model_path / path_text
        if (
            path.stat().st_size != record["bytes"]
            or _sha256_file(path) != record["sha256"]
        ):
            raise VisualRetrievalError(
                f"SigLIP2 runtime file failed size/SHA-256 validation: {path_text}"
            )
    if identity.get("config_sha256") != records.get("config.json", {}).get("sha256"):
        raise VisualRetrievalError(
            "SigLIP2 checkpoint config identity differs from the runtime manifest"
        )
    return identity


def _verified_siglip2_model_identity(model_path: Path) -> dict[str, Any]:
    checkpoint_identity = _verify_pinned_siglip2_identity(model_path)
    identity = _model_fingerprint(model_path)
    unsigned = {
        "path": identity["path"],
        "files": identity["files"],
        "checkpoint_identity": checkpoint_identity,
    }
    return {**unsigned, "fingerprint_sha256": _canonical_sha256(unsigned)}


def _runtime_identity() -> dict[str, Any]:
    import numpy
    import PIL
    import torch
    import transformers

    return {
        "executable": str(Path(sys.executable).resolve()),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "transformers": str(transformers.__version__),
        "numpy": str(numpy.__version__),
        "pillow": str(PIL.__version__),
    }


def _normalize_rows(array: Any) -> Any:
    import numpy as np

    values = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _material_text(record: Mapping[str, Any]) -> str:
    fields: list[str] = []
    for key in ("display_name", "description", "family", "category_path"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            fields.append(value.strip())
    for key in ("keywords", "colors", "finishes"):
        value = record.get(key)
        if isinstance(value, list):
            fields.extend(str(item) for item in value if isinstance(item, str) and item)
    return ". ".join(dict.fromkeys(fields)) or str(
        record.get("material_id", "material")
    )


def _descriptor_text(group: Mapping[str, Any]) -> str:
    descriptor = group.get("descriptor")
    if isinstance(descriptor, str) and descriptor.strip():
        return descriptor.strip()
    if not isinstance(descriptor, Mapping):
        return f"industrial material region {group.get('group_id', '')}".strip()
    ordered = (
        "visual_description",
        "family_hint",
        "base_color",
        "finish_hint",
        "surface_class",
        "roughness_hint",
        "metallicity_hint",
    )
    parts = [
        str(descriptor[key]).strip()
        for key in ordered
        if key in descriptor
        and isinstance(descriptor[key], (str, int, float))
        and str(descriptor[key]).strip()
    ]
    return ". ".join(parts) or f"industrial material region {group.get('group_id', '')}"


def _masked_square(
    image_path: Path,
    mask_path: Path | None,
    *,
    size: int = CANVAS_SIZE,
) -> tuple[Any, Any]:
    import numpy as np
    from PIL import Image, ImageFilter, ImageOps

    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    if mask_path is None:
        mask = Image.new("L", image.size, 255)
    else:
        with Image.open(mask_path) as opened:
            mask = ImageOps.exif_transpose(opened).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
    binary = np.asarray(mask, dtype=np.uint8) >= 128
    ys, xs = np.where(binary)
    if len(xs) == 0:
        image.close()
        mask.close()
        raise VisualRetrievalError(f"mask contains no foreground: {mask_path}")
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    margin = max(4, int(round(max(right - left, bottom - top) * 0.10)))
    left, top = max(0, left - margin), max(0, top - margin)
    right, bottom = min(image.width, right + margin), min(image.height, bottom + margin)
    image_crop = image.crop((left, top, right, bottom))
    mask_crop = mask.crop((left, top, right, bottom))
    neutral = Image.new("RGB", image_crop.size, (127, 127, 127))
    neutral.paste(image_crop, mask=mask_crop)
    scale = min(size / max(1, neutral.width), size / max(1, neutral.height))
    target = (
        max(1, int(round(neutral.width * scale))),
        max(1, int(round(neutral.height * scale))),
    )
    neutral = neutral.resize(target, Image.Resampling.LANCZOS)
    mask_crop = mask_crop.resize(target, Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (size, size), (127, 127, 127))
    mask_canvas = Image.new("L", (size, size), 0)
    offset = ((size - target[0]) // 2, (size - target[1]) // 2)
    canvas.paste(neutral, offset)
    mask_canvas.paste(mask_crop, offset)
    # Dense texture evidence excludes mask boundaries where geometry,
    # antialiasing, and background dominate the representation.  A thin but
    # valid CAD part can disappear completely under the fixed erosion.  In
    # that case keep its exact projected mask: black and narrow parts are
    # valid visual evidence, and only pixels outside the mask are background.
    eroded = mask_canvas.filter(ImageFilter.MinFilter(7))
    if int(np.count_nonzero(np.asarray(eroded, dtype=np.uint8) >= 128)) < 16:
        eroded.close()
        eroded = mask_canvas.copy()
    image.close()
    mask.close()
    image_crop.close()
    mask_crop.close()
    neutral.close()
    mask_canvas.close()
    return canvas, eroded


def _load_request(
    request_path: Path,
) -> tuple[dict[str, Any], Path, Path, list[dict[str, Any]]]:
    request = _read_json(request_path, "visual retrieval request")
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise VisualRetrievalError(
            f"unsupported retrieval request schema: {request.get('schema_version')!r}"
        )
    catalog_path = _resolve_file(
        request.get("catalog"), base=request_path.parent, label="request.catalog"
    )
    raw_root = request.get("material_root")
    if not isinstance(raw_root, str) or not raw_root:
        raise VisualRetrievalError("request.material_root must be a path")
    root = Path(raw_root).expanduser()
    if not root.is_absolute():
        root = request_path.parent / root
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise VisualRetrievalError("request.material_root must be a directory")
    groups = request.get("groups")
    if not isinstance(groups, list) or not groups:
        raise VisualRetrievalError("request.groups must be a non-empty array")
    seen: set[str] = set()
    normalized_groups: list[dict[str, Any]] = []
    for group_index, raw in enumerate(groups):
        if not isinstance(raw, dict):
            raise VisualRetrievalError(f"groups[{group_index}] must be an object")
        group_id = raw.get("group_id")
        observations = raw.get("observations")
        if not isinstance(group_id, str) or not group_id or group_id in seen:
            raise VisualRetrievalError("group IDs must be unique non-empty strings")
        seen.add(group_id)
        if not isinstance(observations, list):
            raise VisualRetrievalError(
                f"group {group_id} observations must be an array"
            )
        normalized_observations: list[dict[str, Any]] = []
        for observation_index, observation in enumerate(observations):
            if not isinstance(observation, Mapping):
                raise VisualRetrievalError(
                    f"group {group_id} observation {observation_index} is invalid"
                )
            image = _resolve_file(
                observation.get("image"),
                base=request_path.parent,
                label=f"group {group_id} observation image",
            )
            raw_mask = observation.get("mask")
            mask = (
                _resolve_file(
                    raw_mask,
                    base=request_path.parent,
                    label=f"group {group_id} observation mask",
                )
                if raw_mask is not None
                else None
            )
            normalized_observations.append(
                {
                    "view_id": observation.get("view_id"),
                    "image": image,
                    "mask": mask,
                }
            )
        normalized_groups.append(
            {**raw, "group_id": group_id, "observations": normalized_observations}
        )
    return request, catalog_path, root, normalized_groups


def _load_catalog(catalog_path: Path, material_root: Path) -> list[dict[str, Any]]:
    document = _read_json(catalog_path, "material catalog")
    materials = document.get("materials")
    if not isinstance(materials, list) or not materials:
        raise VisualRetrievalError("material catalog contains no materials")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(materials):
        if not isinstance(raw, dict):
            raise VisualRetrievalError(f"catalog material {index} must be an object")
        material_id = raw.get("material_id")
        if not isinstance(material_id, str) or not material_id or material_id in seen:
            raise VisualRetrievalError("catalog material IDs must be unique")
        seen.add(material_id)
        thumbnail_path = raw.get("thumbnail_path")
        thumbnail = (
            _resolve_under_root(
                material_root, thumbnail_path, f"material {material_id} thumbnail"
            )
            if isinstance(thumbnail_path, str)
            else None
        )
        normalized.append({**raw, "_thumbnail": thumbnail})
    return normalized


def _resolve_bank_file(
    bank_dir: Path,
    value: Any,
    *,
    label: str,
    expected_sha256: Any,
) -> Path:
    if not isinstance(value, str) or not value:
        raise VisualRetrievalError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise VisualRetrievalError(f"{label} is unsafe: {value!r}")
    try:
        path = (bank_dir / relative).resolve(strict=True)
        path.relative_to(bank_dir)
    except (OSError, ValueError) as exc:
        raise VisualRetrievalError(
            f"{label} is missing or outside the observation bank"
        ) from exc
    if not path.is_file():
        raise VisualRetrievalError(f"{label} is not a file: {path}")
    if (
        not isinstance(expected_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or _sha256_file(path) != expected_sha256
    ):
        raise VisualRetrievalError(f"{label} failed SHA-256 validation")
    return path


def _mdl_source_digest(material_root: Path) -> tuple[str, int]:
    records: list[dict[str, str]] = []
    for path in sorted(material_root.rglob("*.mdl")):
        try:
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(material_root).as_posix()
        except (OSError, ValueError) as exc:
            raise VisualRetrievalError(
                f"MDL source escaped the material root: {path}"
            ) from exc
        records.append({"path": relative, "sha256": _sha256_file(resolved)})
    return _canonical_sha256(records), len(records)


def _portable_model_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Compare model content without binding a bank to one machine path."""

    identity = dict(value)
    identity.pop("path", None)
    # Older banks computed this aggregate from the full record, including the
    # absolute path.  The per-file digests and checkpoint identity below are
    # the actual content proof and remain stable after relocating the models.
    identity.pop("fingerprint_sha256", None)
    return identity


def _load_base_observation_bank(
    *,
    bank_dir: Path,
    material_root: Path,
    material_ids: Sequence[str],
    siglip_model_identity: Mapping[str, Any],
    dino_model_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Load a sealed Base-only observation index and align it to the catalog."""

    import numpy as np

    try:
        bank_dir = bank_dir.expanduser().resolve(strict=True)
    except OSError as exc:
        raise VisualRetrievalError(
            f"Base observation bank does not exist: {bank_dir}"
        ) from exc
    if not bank_dir.is_dir():
        raise VisualRetrievalError(
            f"Base observation bank is not a directory: {bank_dir}"
        )
    material_root = material_root.expanduser().resolve(strict=True)
    if material_root.name != "Base" or any(
        part.casefold() == "vmaterials_2" for part in material_root.parts
    ):
        raise VisualRetrievalError(
            "Base observation bank requires the exact NVIDIA Materials/Base root"
        )

    scope = _read_json(bank_dir / "scope_report.json", "observation bank scope")
    if (
        scope.get("schema_version") != BASE_BANK_SCOPE_SCHEMA_VERSION
        or scope.get("scope") != "nvidia_base"
        or scope.get("collection_name") != "Base"
        or scope.get("exact_cover") is not True
        or scope.get("forbidden_vmaterials_2_count") != 0
    ):
        raise VisualRetrievalError(
            "observation bank scope does not match the exact current Base root"
        )
    current_source_digest, current_module_count = _mdl_source_digest(material_root)
    if (
        scope.get("mdl_sources_sha256") != current_source_digest
        or scope.get("mdl_module_count") != current_module_count
    ):
        raise VisualRetrievalError(
            "Base MDL sources changed after the observation bank was built"
        )

    manifest_path = bank_dir / "index_manifest.json"
    manifest = _read_json(manifest_path, "observation bank index manifest")
    unsigned_manifest = dict(manifest)
    manifest_seal = unsigned_manifest.pop("manifest_sha256", None)
    if (
        manifest.get("schema_version") != BASE_BANK_INDEX_SCHEMA_VERSION
        or manifest.get("scope") != "nvidia_base"
        or manifest.get("complete") is not True
        or manifest.get("forbidden_vmaterials_2_count") != 0
        or _canonical_sha256(unsigned_manifest) != manifest_seal
        or manifest.get("catalog_sha256") != scope.get("catalog_sha256")
    ):
        raise VisualRetrievalError(
            "Base observation bank index manifest failed its sealed contract"
        )
    siglip_record = manifest.get("siglip2")
    dino_record = manifest.get("dinov2")
    if (
        not isinstance(siglip_record, Mapping)
        or not isinstance(siglip_record.get("model"), Mapping)
        or _portable_model_identity(siglip_record["model"])
        != _portable_model_identity(siglip_model_identity)
        or not isinstance(dino_record, Mapping)
        or not isinstance(dino_record.get("model"), Mapping)
        or _portable_model_identity(dino_record["model"])
        != _portable_model_identity(dino_model_identity)
    ):
        raise VisualRetrievalError(
            "observation bank embeddings were built with different model weights"
        )

    embeddings_path = _resolve_bank_file(
        bank_dir,
        manifest.get("visual_embeddings"),
        label="observation bank visual embeddings",
        expected_sha256=manifest.get("visual_embeddings_sha256"),
    )
    profiles_path = _resolve_bank_file(
        bank_dir,
        manifest.get("appearance_profiles"),
        label="observation bank appearance profiles",
        expected_sha256=manifest.get("appearance_profiles_sha256"),
    )
    profiles_document = _read_json(
        profiles_path, "observation bank appearance profiles"
    )
    profile_rows = profiles_document.get("materials")
    if (
        profiles_document.get("schema_version") != BASE_BANK_INDEX_SCHEMA_VERSION
        or profiles_document.get("scope") != "nvidia_base"
        or not isinstance(profile_rows, list)
    ):
        raise VisualRetrievalError("observation bank appearance profiles are invalid")
    profiles_by_id: dict[str, dict[str, Any]] = {}
    for row in profile_rows:
        material_id = row.get("material_id") if isinstance(row, dict) else None
        if (
            not isinstance(material_id, str)
            or not material_id
            or material_id in profiles_by_id
        ):
            raise VisualRetrievalError(
                "observation bank appearance profiles contain invalid IDs"
            )
        profiles_by_id[material_id] = row

    with np.load(embeddings_path, allow_pickle=False) as archive:
        bank_ids = archive["material_ids"].astype(str).tolist()
        siglip_embeddings = np.asarray(archive["siglip2"], dtype=np.float32)
        dino_embeddings = np.asarray(archive["dinov2"], dtype=np.float32)
    expected_ids = list(material_ids)
    if (
        len(bank_ids) != len(set(bank_ids))
        or set(bank_ids) != set(expected_ids)
        or set(profiles_by_id) != set(expected_ids)
        or manifest.get("material_count") != len(expected_ids)
        or scope.get("material_count") != len(expected_ids)
    ):
        raise VisualRetrievalError(
            "observation bank does not exactly cover the current Base catalog"
        )
    if (
        siglip_embeddings.shape
        != (len(bank_ids), int(siglip_record.get("dimension", -1)))
        or dino_embeddings.shape
        != (len(bank_ids), int(dino_record.get("dimension", -1)))
        or not np.isfinite(siglip_embeddings).all()
        or not np.isfinite(dino_embeddings).all()
    ):
        raise VisualRetrievalError("observation bank embedding arrays are invalid")
    bank_index = {material_id: index for index, material_id in enumerate(bank_ids)}
    order = [bank_index[material_id] for material_id in expected_ids]
    identity = {
        "path": str(bank_dir),
        "index_manifest_sha256": _sha256_file(manifest_path),
        "index_manifest_seal": manifest_seal,
        "visual_embeddings_sha256": manifest["visual_embeddings_sha256"],
        "appearance_profiles_sha256": manifest["appearance_profiles_sha256"],
        "mdl_sources_sha256": current_source_digest,
        "material_count": len(expected_ids),
        "observation_source_counts": manifest.get("observation_source_counts"),
    }
    return {
        "siglip2": _normalize_rows(siglip_embeddings[order]),
        "dinov2": _normalize_rows(dino_embeddings[order]),
        "profiles_by_id": profiles_by_id,
        "identity": identity,
        "manifest": manifest,
    }


def _to_device(batch: Mapping[str, Any], device: str, dtype: Any) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            value = value.to(device)
            if (
                key == "pixel_values"
                and getattr(value, "is_floating_point", lambda: False)()
            ):
                value = value.to(dtype=dtype)
        output[key] = value
    return output


def _siglip_image_embeddings(
    model: Any,
    processor: Any,
    images: Sequence[Any],
    *,
    device: str,
    dtype: Any,
    batch_size: int,
) -> Any:
    import numpy as np
    import torch

    output: list[Any] = []
    for start in range(0, len(images), batch_size):
        inputs = processor(
            images=list(images[start : start + batch_size]), return_tensors="pt"
        )
        with torch.inference_mode():
            features = model.get_image_features(**_to_device(inputs, device, dtype))
        features = _siglip_pooled_features(features, "image")
        output.append(features.detach().float().cpu().numpy())
    return _normalize_rows(np.concatenate(output, axis=0))


def _siglip_text_embeddings(
    model: Any,
    processor: Any,
    texts: Sequence[str],
    *,
    device: str,
    dtype: Any,
    batch_size: int,
    progress_stage: str | None = None,
    progress_total: int | None = None,
    progress_offset: int = 0,
) -> Any:
    import numpy as np
    import torch

    output: list[Any] = []
    for start in range(0, len(texts), batch_size):
        inputs = processor(
            text=list(texts[start : start + batch_size]),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            features = model.get_text_features(**_to_device(inputs, device, dtype))
        features = _siglip_pooled_features(features, "text")
        output.append(features.detach().float().cpu().numpy())
        if progress_stage is not None and progress_total is not None:
            _emit_progress(
                stage=progress_stage,
                state="update",
                current=min(
                    progress_total, progress_offset + start + len(inputs["input_ids"])
                ),
                total=progress_total,
                unit="catalog-items",
                detail="SigLIP2 catalog text embeddings",
            )
    return _normalize_rows(np.concatenate(output, axis=0))


def _siglip_pooled_features(output: Any, modality: str) -> Any:
    """Normalize Transformers v4 tensor and v5 pooled-output APIs."""

    if hasattr(output, "detach"):
        return output
    pooled = getattr(output, "pooler_output", None)
    if hasattr(pooled, "detach"):
        return pooled
    if isinstance(output, (tuple, list)) and len(output) > 1:
        pooled = output[1]
        if hasattr(pooled, "detach"):
            return pooled
    raise VisualRetrievalError(
        f"SigLIP2 {modality} encoder returned no pooled feature tensor"
    )


def _catalog_digest(
    materials: Sequence[Mapping[str, Any]],
    *,
    emit_progress: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    total = len(materials)
    if emit_progress:
        _emit_progress(
            stage="siglip2_catalog_hash",
            state="start",
            current=0,
            total=total,
            unit="materials",
            detail="Hashing complete NVIDIA catalog and thumbnails",
        )
    for index, item in enumerate(materials, start=1):
        thumbnail = item.get("_thumbnail")
        records.append(
            {
                "material_id": item["material_id"],
                "text": _material_text(item),
                "thumbnail": (
                    {
                        "path": str(thumbnail),
                        "sha256": _sha256_file(thumbnail),
                    }
                    if isinstance(thumbnail, Path)
                    else None
                ),
            }
        )
        if emit_progress and (index == total or index % 128 == 0):
            _emit_progress(
                stage="siglip2_catalog_hash",
                state="update",
                current=index,
                total=total,
                unit="materials",
                detail="Hashing complete NVIDIA catalog and thumbnails",
            )
    if emit_progress:
        _emit_progress(
            stage="siglip2_catalog_hash",
            state="complete",
            current=total,
            total=total,
            unit="materials",
            detail="NVIDIA catalog content identity complete",
        )
    return _canonical_sha256(records), records


def _build_or_load_siglip_index(
    *,
    materials: list[dict[str, Any]],
    model_path: Path,
    cache_dir: Path,
    device: str,
    batch_size: int,
) -> tuple[Any, dict[str, Any], Any, Any]:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    model_identity = _verified_siglip2_model_identity(model_path)
    catalog_digest, catalog_records = _catalog_digest(materials, emit_progress=True)
    cache_key = _canonical_sha256(
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "model_fingerprint": model_identity["fingerprint_sha256"],
            "catalog_digest": catalog_digest,
            "preprocess": "official_processor_gallery_image_plus_text_0.9_0.1/v1",
        }
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"{cache_key}.npz"
    manifest_path = cache_dir / f"{cache_key}.json"

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
    ).to(device)
    model.eval()
    material_ids = [str(item["material_id"]) for item in materials]

    if npz_path.is_file() and manifest_path.is_file():
        manifest = _read_json(manifest_path, "SigLIP2 cache manifest")
        if (
            manifest.get("schema_version") != CACHE_SCHEMA_VERSION
            or manifest.get("cache_key") != cache_key
            or manifest.get("npz_sha256") != _sha256_file(npz_path)
        ):
            raise VisualRetrievalError(
                "SigLIP2 cache exists but failed integrity validation"
            )
        with np.load(npz_path, allow_pickle=False) as archive:
            cached_ids = archive["material_ids"].astype(str).tolist()
            embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        if cached_ids != material_ids:
            raise VisualRetrievalError("SigLIP2 cache material order changed")
        _emit_progress(
            stage="siglip2_catalog_index",
            state="complete",
            current=len(materials),
            total=len(materials),
            unit="materials",
            detail="Verified content-addressed SigLIP2 catalog cache",
        )
        return embeddings, manifest, model, processor

    texts = [_material_text(item) for item in materials]
    available_indices = [
        index for index, item in enumerate(materials) if item["_thumbnail"] is not None
    ]
    progress_total = len(materials) + len(available_indices)
    _emit_progress(
        stage="siglip2_catalog_index",
        state="start",
        current=0,
        total=progress_total,
        unit="catalog-items",
        detail="Building content-addressed SigLIP2 catalog index",
    )
    text_embeddings = _siglip_text_embeddings(
        model,
        processor,
        texts,
        device=device,
        dtype=dtype,
        batch_size=batch_size * 2,
        progress_stage="siglip2_catalog_index",
        progress_total=progress_total,
    )
    image_embeddings = np.zeros_like(text_embeddings)
    encoded_image_count = 0
    for start in range(0, len(available_indices), batch_size):
        batch_indices = available_indices[start : start + batch_size]
        available_images: list[Any] = []
        for index in batch_indices:
            with Image.open(materials[index]["_thumbnail"]) as opened:
                available_images.append(opened.convert("RGB").copy())
        encoded = _siglip_image_embeddings(
            model,
            processor,
            available_images,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )
        for image in available_images:
            image.close()
        image_embeddings[batch_indices] = encoded
        encoded_image_count += len(batch_indices)
        _emit_progress(
            stage="siglip2_catalog_index",
            state="update",
            current=len(materials) + encoded_image_count,
            total=progress_total,
            unit="catalog-items",
            detail="Encoding NVIDIA material thumbnails with SigLIP2",
        )
    has_image = np.zeros((len(materials),), dtype=bool)
    has_image[available_indices] = True
    embeddings = text_embeddings.copy()
    embeddings[has_image] = _normalize_rows(
        0.90 * image_embeddings[has_image] + 0.10 * text_embeddings[has_image]
    )
    embeddings = _normalize_rows(embeddings)
    with npz_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            material_ids=np.asarray(material_ids),
            embeddings=embeddings.astype(np.float16),
        )
    unsigned = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "model": model_identity,
        "catalog_digest": catalog_digest,
        "catalog_records": catalog_records,
        "material_count": len(materials),
        "thumbnail_count": len(available_indices),
        "text_only_count": len(materials) - len(available_indices),
        "npz": str(npz_path),
        "npz_sha256": _sha256_file(npz_path),
    }
    manifest = {**unsigned, "manifest_sha256": _canonical_sha256(unsigned)}
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _emit_progress(
        stage="siglip2_catalog_index",
        state="complete",
        current=progress_total,
        total=progress_total,
        unit="catalog-items",
        detail="SigLIP2 catalog index built and sealed",
    )
    return embeddings, manifest, model, processor


def _manual_pixels(
    images: Sequence[Any], *, mean: Sequence[float], std: Sequence[float]
) -> Any:
    import numpy as np
    import torch

    values = []
    for image in images:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        values.append(array.transpose(2, 0, 1))
    tensor = torch.from_numpy(np.stack(values))
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def _dino_tokens(
    model: Any,
    images: Sequence[Any],
    masks: Sequence[Any],
    *,
    device: str,
    dtype: Any,
    mean: Sequence[float],
    std: Sequence[float],
) -> list[Any]:
    import numpy as np
    import torch

    pixels = _manual_pixels(images, mean=mean, std=std).to(device=device, dtype=dtype)
    with torch.inference_mode():
        output = model(pixel_values=pixels)
    hidden = output.last_hidden_state.detach().float().cpu()
    patch_size = getattr(model.config, "patch_size", 14)
    if isinstance(patch_size, (tuple, list)):
        patch_size = int(patch_size[0])
    patch_size = int(patch_size)
    grid = CANVAS_SIZE // patch_size
    patch_count = grid * grid
    if hidden.shape[1] < patch_count:
        raise VisualRetrievalError(
            "DINOv2 output contains fewer tokens than image patches"
        )
    patch_tokens = hidden[:, -patch_count:, :]
    results: list[Any] = []
    for index, mask in enumerate(masks):
        mask_array = np.asarray(mask, dtype=np.float32) / 255.0
        usable_height = grid * patch_size
        usable_width = grid * patch_size
        mask_array = mask_array[:usable_height, :usable_width]
        pooled = (
            mask_array.reshape(grid, patch_size, grid, patch_size)
            .mean(axis=(1, 3))
            .reshape(-1)
        )
        selected = pooled >= 0.50
        if int(np.count_nonzero(selected)) < 4:
            selected = pooled >= 0.10
        if int(np.count_nonzero(selected)) < 1:
            # An accepted region that becomes sub-patch after erosion carries
            # no defensible texture evidence.  Never replace it with all image
            # patches, which would silently rank the background instead.
            selected = np.zeros((patch_count,), dtype=bool)
        tokens = patch_tokens[index, torch.from_numpy(selected)]
        tokens = torch.nn.functional.normalize(tokens, dim=-1)
        results.append(tokens.numpy())
    return results


def _texture_similarity(query_sets: Sequence[Any], candidate: Any) -> float:
    candidate = _normalize_rows(candidate)
    view_scores: list[float] = []
    for query in query_sets:
        query = _normalize_rows(query)
        similarities = query @ candidate.T
        symmetric_chamfer = 0.5 * (
            float(similarities.max(axis=1).mean())
            + float(similarities.max(axis=0).mean())
        )
        global_cosine = float(
            _normalize_rows(query.mean(axis=0, keepdims=True))
            @ _normalize_rows(candidate.mean(axis=0, keepdims=True)).T
        )
        view_scores.append(0.75 * symmetric_chamfer + 0.25 * global_cosine)
    return float(sum(view_scores) / max(1, len(view_scores)))


def _rank_scores(
    material_ids: Sequence[str],
    scores: Any,
) -> tuple[list[int], dict[str, int]]:
    order = sorted(
        range(len(material_ids)),
        key=lambda index: (-float(scores[index]), material_ids[index]),
    )
    return order, {
        material_ids[index]: rank for rank, index in enumerate(order, start=1)
    }


def _masked_query_rgb(images: Sequence[Any], masks: Sequence[Any]) -> Any:
    import numpy as np

    # Upstream Part-ID evidence accepts a six-pixel chromatic component only
    # after purity, foreground-overlap, alignment, and local-contrast gates.
    # Keep retrieval's color estimator on the same contract; requiring 16
    # pixels here would erase exactly the small hoses and connectors that
    # passed those stronger geometric/evidence checks.
    minimum_color_pixels = 6
    medians: list[Any] = []
    for image, mask in zip(images, masks, strict=True):
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        valid = np.asarray(mask.convert("L"), dtype=np.uint8) >= 128
        pixels = rgb[valid]
        if len(pixels) >= minimum_color_pixels:
            medians.append(np.median(pixels, axis=0))
    if not medians:
        return None
    return np.median(np.stack(medians), axis=0).astype(np.float32)


def _bank_color_vectors(
    material_ids: Sequence[str],
    profiles_by_id: Mapping[str, Mapping[str, Any]],
) -> Any:
    import numpy as np

    vectors: list[Any] = []
    for material_id in material_ids:
        record = profiles_by_id[material_id]
        appearance = record.get("appearance")
        samples: list[Any] = []
        if isinstance(appearance, Mapping):
            for profile in sorted(appearance):
                values = appearance[profile]
                rgb = values.get("median_rgb") if isinstance(values, Mapping) else None
                if (
                    isinstance(rgb, list)
                    and len(rgb) == 3
                    and all(
                        isinstance(value, (int, float)) and not isinstance(value, bool)
                        for value in rgb
                    )
                ):
                    samples.append(np.asarray(rgb, dtype=np.float32))
        if not samples:
            raise VisualRetrievalError(
                f"observation bank has no median RGB for {material_id}"
            )
        vectors.append(np.median(np.stack(samples), axis=0))
    return np.stack(vectors).astype(np.float32)


def _mvinverse_prior_scores(
    *,
    group: Mapping[str, Any],
    material_ids: Sequence[str],
    profiles_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, bool]:
    import numpy as np

    descriptor = group.get("descriptor")
    if not isinstance(descriptor, Mapping):
        return np.full((len(material_ids),), np.nan, dtype=np.float32), False
    evidence: list[tuple[str, float]] = []
    for descriptor_key, mdl_key in (
        ("roughness_hint", "reflection_roughness_constant"),
        ("metallicity_hint", "metallic_constant"),
    ):
        value = descriptor.get(descriptor_key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0.0 <= float(value) <= 1.0
        ):
            evidence.append((mdl_key, float(value)))
    if not evidence:
        return np.full((len(material_ids),), np.nan, dtype=np.float32), False
    scores: list[float] = []
    for material_id in material_ids:
        authored = profiles_by_id[material_id].get("authored_mdl")
        terms: list[float] = []
        if isinstance(authored, Mapping):
            for mdl_key, target in evidence:
                value = authored.get(mdl_key)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and 0.0 <= float(value) <= 1.0
                ):
                    terms.append(1.0 - abs(float(value) - target))
        scores.append(sum(terms) / len(terms) if terms else float("nan"))
    values = np.asarray(scores, dtype=np.float32)
    return values, bool(np.isfinite(values).any())


def _release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run(
    *,
    request_path: Path,
    siglip_model_path: Path,
    dino_model_path: Path,
    cache_dir: Path,
    output_dir: Path,
    device: str,
    siglip_top_k: int,
    final_top_k: int,
    batch_size: int,
    observation_bank_dir: Path | None = None,
) -> dict[str, Any]:
    import torch
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor

    request, catalog_path, material_root, groups = _load_request(request_path)
    materials = _load_catalog(catalog_path, material_root)
    material_by_id = {str(item["material_id"]): item for item in materials}
    material_ids = [str(item["material_id"]) for item in materials]
    if siglip_top_k > len(materials):
        siglip_top_k = len(materials)
    if final_top_k > siglip_top_k:
        raise VisualRetrievalError("final_top_k cannot exceed siglip_top_k")
    if device == "cuda" and not torch.cuda.is_available():
        raise VisualRetrievalError("CUDA retrieval requested but CUDA is unavailable")

    output_dir.mkdir(parents=True, exist_ok=False)
    siglip_model_path = siglip_model_path.expanduser().resolve(strict=True)
    dino_model_path = dino_model_path.expanduser().resolve(strict=True)
    siglip_identity = _verified_siglip2_model_identity(siglip_model_path)
    dino_identity = _model_fingerprint(dino_model_path)
    observation_bank: dict[str, Any] | None = None
    if observation_bank_dir is not None:
        observation_bank = _load_base_observation_bank(
            bank_dir=observation_bank_dir,
            material_root=material_root,
            material_ids=material_ids,
            siglip_model_identity=siglip_identity,
            dino_model_identity=dino_identity,
        )
        gallery_embeddings = observation_bank["siglip2"]
        siglip_manifest = {
            "model": siglip_identity,
            "observation_bank": observation_bank["identity"],
        }
        siglip_processor = AutoProcessor.from_pretrained(
            siglip_model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        dtype = torch.float16 if device == "cuda" else torch.float32
        siglip_model = AutoModel.from_pretrained(
            siglip_model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
        ).to(device)
        siglip_model.eval()
        _emit_progress(
            stage="base_observation_bank",
            state="complete",
            current=len(materials),
            total=len(materials),
            unit="materials",
            detail="Verified sealed Base observation embeddings",
        )
    else:
        gallery_embeddings, siglip_manifest, siglip_model, siglip_processor = (
            _build_or_load_siglip_index(
                materials=materials,
                model_path=siglip_model_path,
                cache_dir=cache_dir.expanduser().resolve(),
                device=device,
                batch_size=batch_size,
            )
        )
    dtype = torch.float16 if device == "cuda" else torch.float32
    prepared: dict[str, dict[str, Any]] = {}
    for group_index, group in enumerate(groups, start=1):
        observations = group["observations"]
        images: list[Any] = []
        masks: list[Any] = []
        observation_audit: list[dict[str, Any]] = []
        for observation in observations:
            canvas, mask = _masked_square(
                observation["image"], observation["mask"], size=CANVAS_SIZE
            )
            images.append(canvas)
            masks.append(mask)
            observation_audit.append(
                {
                    "view_id": observation.get("view_id"),
                    "image": str(observation["image"]),
                    "image_sha256": _sha256_file(observation["image"]),
                    "mask": (
                        str(observation["mask"])
                        if observation["mask"] is not None
                        else None
                    ),
                    "mask_sha256": (
                        _sha256_file(observation["mask"])
                        if observation["mask"] is not None
                        else None
                    ),
                }
            )
        if not images:
            prepared[group["group_id"]] = {
                "group": group,
                "images": [],
                "masks": [],
                "observation_audit": [],
                "siglip_ranking": [],
                "reason_codes": ["no_accepted_sam3_observations"],
            }
            continue
        image_embeddings = _siglip_image_embeddings(
            siglip_model,
            siglip_processor,
            images,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )
        text_embedding = _siglip_text_embeddings(
            siglip_model,
            siglip_processor,
            [_descriptor_text(group)],
            device=device,
            dtype=dtype,
            batch_size=1,
        )[0]
        image_weight = 0.90 if observation_bank is not None else 0.75
        query = _normalize_rows(
            (
                image_weight * image_embeddings.mean(axis=0, keepdims=True)
                + (1.0 - image_weight) * text_embedding[None, :]
            )
        )[0]
        scores = gallery_embeddings @ query
        full_order, siglip_ranks = _rank_scores(material_ids, scores)
        order = full_order[:siglip_top_k]
        prepared[group["group_id"]] = {
            "group": group,
            "images": images,
            "masks": masks,
            "observation_audit": observation_audit,
            "siglip_scores": scores,
            "siglip_ranks": siglip_ranks,
            "siglip_ranking": [
                {
                    "rank": rank,
                    "material_id": material_ids[index],
                    "score": round(float(scores[index]), 8),
                    "thumbnail_available": material_by_id[material_ids[index]][
                        "_thumbnail"
                    ]
                    is not None,
                }
                for rank, index in enumerate(order, start=1)
            ],
            "reason_codes": [],
        }
        _emit_progress(
            stage="siglip2",
            state="update",
            current=group_index,
            total=len(groups),
            unit="groups",
            detail=f"SigLIP2 group {group['group_id']}",
        )
    del siglip_processor
    del siglip_model
    _release_cuda_cache()

    dino_processor = AutoImageProcessor.from_pretrained(
        dino_model_path,
        local_files_only=True,
        trust_remote_code=False,
        backend="pil",
    )
    mean = list(getattr(dino_processor, "image_mean", [0.485, 0.456, 0.406]))
    std = list(getattr(dino_processor, "image_std", [0.229, 0.224, 0.225]))
    dino_model = AutoModel.from_pretrained(
        dino_model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
    ).to(device)
    dino_model.eval()
    results: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups, start=1):
        state = prepared[group["group_id"]]
        if not state["siglip_ranking"]:
            results.append(
                {
                    "group_id": group["group_id"],
                    "retrieval_strategy": (
                        BASE_BANK_RETRIEVAL_STRATEGY
                        if observation_bank is not None
                        else LEGACY_RETRIEVAL_STRATEGY
                    ),
                    "descriptor": group.get("descriptor"),
                    "observations": state["observation_audit"],
                    "accepted": False,
                    "reason_codes": state["reason_codes"],
                    "siglip2_ranking": [],
                    "dino_ranking": [],
                    "fused_ranking": [],
                }
            )
            continue
        query_tokens = [
            tokens
            for tokens in _dino_tokens(
                dino_model,
                state["images"],
                state["masks"],
                device=device,
                dtype=dtype,
                mean=mean,
                std=std,
            )
            if len(tokens) > 0
        ]
        query_rgb = _masked_query_rgb(state["images"], state["masks"])
        color_scored: list[dict[str, Any]] = []
        pbr_scored: list[dict[str, Any]] = []
        query_appearance: dict[str, Any] = {}
        if observation_bank is not None:
            import numpy as np

            dino_scores = np.zeros((len(material_ids),), dtype=np.float32)
            dino_ranks: dict[str, int] = {}
            dino_order: list[int] = []
            if query_tokens:
                query_dino = _normalize_rows(
                    np.stack(
                        [
                            np.asarray(tokens, dtype=np.float32).mean(axis=0)
                            for tokens in query_tokens
                        ]
                    ).mean(axis=0, keepdims=True)
                )[0]
                dino_scores = observation_bank["dinov2"] @ query_dino
                dino_order, dino_ranks = _rank_scores(material_ids, dino_scores)
            dino_scored = [
                {
                    "rank": rank,
                    "material_id": material_ids[index],
                    "score": round(float(dino_scores[index]), 8),
                }
                for rank, index in enumerate(dino_order[:siglip_top_k], start=1)
            ]

            if query_rgb is None:
                raise VisualRetrievalError(
                    f"group {group['group_id']} has no usable masked color pixels"
                )
            bank_colors = _bank_color_vectors(
                material_ids, observation_bank["profiles_by_id"]
            )
            color_scores = np.asarray(
                [
                    perceptual_similarity(query_rgb.tolist(), candidate.tolist())
                    for candidate in bank_colors
                ],
                dtype=np.float32,
            )
            color_order, color_ranks = _rank_scores(material_ids, color_scores)
            color_scored = [
                {
                    "rank": rank,
                    "material_id": material_ids[index],
                    "score": round(float(color_scores[index]), 8),
                }
                for rank, index in enumerate(color_order[:siglip_top_k], start=1)
            ]
            pbr_scores, has_pbr_prior = _mvinverse_prior_scores(
                group=group,
                material_ids=material_ids,
                profiles_by_id=observation_bank["profiles_by_id"],
            )
            pbr_ranks: dict[str, int] = {}
            if has_pbr_prior:
                pbr_order = sorted(
                    [
                        index
                        for index in range(len(material_ids))
                        if np.isfinite(pbr_scores[index])
                    ],
                    key=lambda index: (
                        -float(pbr_scores[index]),
                        material_ids[index],
                    ),
                )
                pbr_ranks = {
                    material_ids[index]: rank
                    for rank, index in enumerate(pbr_order, start=1)
                }
                pbr_scored = [
                    {
                        "rank": rank,
                        "material_id": material_ids[index],
                        "score": round(float(pbr_scores[index]), 8),
                    }
                    for rank, index in enumerate(pbr_order[:siglip_top_k], start=1)
                ]
            query_appearance = {
                "median_rgb": [round(float(value), 8) for value in query_rgb.tolist()],
                "mvinverse_prior_available": has_pbr_prior,
            }
            fused = []
            for index, material_id in enumerate(material_ids):
                siglip_rank = state["siglip_ranks"][material_id]
                dino_rank = dino_ranks.get(material_id)
                color_rank = color_ranks[material_id]
                pbr_rank = pbr_ranks.get(material_id)
                rrf = 1.0 / (60.0 + float(siglip_rank))
                if dino_rank is not None:
                    rrf += 1.20 / (60.0 + float(dino_rank))
                rrf += 0.80 / (60.0 + float(color_rank))
                if pbr_rank is not None:
                    rrf += 0.20 / (60.0 + float(pbr_rank))
                fused.append(
                    {
                        "material_id": material_id,
                        "score": round(rrf, 10),
                        "siglip2_rank": siglip_rank,
                        "siglip2_score": round(float(state["siglip_scores"][index]), 8),
                        "dino_rank": dino_rank,
                        "dino_score": (
                            round(float(dino_scores[index]), 8)
                            if dino_rank is not None
                            else None
                        ),
                        "color_rank": color_rank,
                        "color_score": round(float(color_scores[index]), 8),
                        "mvinverse_rank": pbr_rank,
                        "mvinverse_score": (
                            round(float(pbr_scores[index]), 8)
                            if pbr_rank is not None
                            else None
                        ),
                        "thumbnail_available": material_by_id[material_id]["_thumbnail"]
                        is not None,
                    }
                )
        else:
            candidate_images: list[Any] = []
            candidate_masks: list[Any] = []
            candidate_ids: list[str] = []
            if query_tokens:
                for ranked in state["siglip_ranking"]:
                    material = material_by_id[ranked["material_id"]]
                    thumbnail = material["_thumbnail"]
                    if thumbnail is None:
                        continue
                    canvas, mask = _masked_square(thumbnail, None, size=CANVAS_SIZE)
                    candidate_images.append(canvas)
                    candidate_masks.append(mask)
                    candidate_ids.append(ranked["material_id"])
            candidate_tokens_by_id: dict[str, Any] = {}
            for start in range(0, len(candidate_images), batch_size):
                tokens = _dino_tokens(
                    dino_model,
                    candidate_images[start : start + batch_size],
                    candidate_masks[start : start + batch_size],
                    device=device,
                    dtype=dtype,
                    mean=mean,
                    std=std,
                )
                candidate_tokens_by_id.update(
                    (
                        material_id,
                        candidate_tokens,
                    )
                    for material_id, candidate_tokens in zip(
                        candidate_ids[start : start + batch_size], tokens
                    )
                    if len(candidate_tokens) > 0
                )
            for image in candidate_images + candidate_masks:
                image.close()
            dino_scored = [
                {
                    "material_id": material_id,
                    "score": round(
                        _texture_similarity(query_tokens, candidate_tokens), 8
                    ),
                }
                for material_id, candidate_tokens in candidate_tokens_by_id.items()
            ]
            dino_scored.sort(key=lambda item: (-item["score"], item["material_id"]))
            for rank, item in enumerate(dino_scored, start=1):
                item["rank"] = rank
            siglip_by_id = {
                item["material_id"]: item for item in state["siglip_ranking"]
            }
            dino_by_id = {item["material_id"]: item for item in dino_scored}
            fused = []
            for material_id, siglip in siglip_by_id.items():
                dino = dino_by_id.get(material_id)
                # A thumbnail-free MDL remains searchable through SigLIP2
                # text space, but cannot receive invented texture evidence.
                rrf = 1.0 / (60.0 + float(siglip["rank"]))
                if dino is not None:
                    rrf += 1.20 / (60.0 + float(dino["rank"]))
                fused.append(
                    {
                        "material_id": material_id,
                        "score": round(rrf, 10),
                        "siglip2_rank": siglip["rank"],
                        "siglip2_score": siglip["score"],
                        "dino_rank": dino["rank"] if dino is not None else None,
                        "dino_score": dino["score"] if dino is not None else None,
                        "thumbnail_available": siglip["thumbnail_available"],
                    }
                )
        for image in state["images"] + state["masks"]:
            image.close()
        fused.sort(
            key=lambda item: (
                -item["score"],
                item["siglip2_rank"],
                item["material_id"],
            )
        )
        for rank, item in enumerate(fused, start=1):
            item["rank"] = rank
        results.append(
            {
                "group_id": group["group_id"],
                "retrieval_strategy": (
                    BASE_BANK_RETRIEVAL_STRATEGY
                    if observation_bank is not None
                    else LEGACY_RETRIEVAL_STRATEGY
                ),
                "descriptor": group.get("descriptor"),
                "observations": state["observation_audit"],
                "accepted": True,
                "reason_codes": (
                    [] if query_tokens else ["no_masked_dino_patches_siglip2_only"]
                ),
                "siglip2_ranking": state["siglip_ranking"],
                "dino_ranking": dino_scored,
                "color_ranking": color_scored,
                "mvinverse_prior_ranking": pbr_scored,
                "query_appearance": query_appearance,
                "fused_ranking": fused[:final_top_k],
            }
        )
        _emit_progress(
            stage="dinov2",
            state="update",
            current=group_index,
            total=len(groups),
            unit="groups",
            detail=f"DINOv2 group {group['group_id']}",
        )
    del dino_processor
    del dino_model
    _release_cuda_cache()

    unsigned: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "request": {
            "path": str(request_path),
            "sha256": _sha256_file(request_path),
            "document_sha256": _canonical_sha256(request),
        },
        "catalog": {
            "path": str(catalog_path),
            "sha256": _sha256_file(catalog_path),
            "material_root": str(material_root),
            "material_count": len(materials),
            "all_catalog_materials_indexed": True,
        },
        "backends": {
            "runtime": _runtime_identity(),
            "siglip2": (
                {
                    "model": siglip_manifest["model"],
                    "index_source": "nvidia_base_observation_bank",
                    "observation_bank": siglip_manifest["observation_bank"],
                    "query_aggregation": (
                        "normalized_0.90_masked_images_plus_0.10_qwen_descriptor"
                    ),
                }
                if observation_bank is not None
                else {
                    "model": siglip_manifest["model"],
                    "catalog_digest": siglip_manifest["catalog_digest"],
                    "index_cache_key": siglip_manifest["cache_key"],
                    "index_manifest_sha256": siglip_manifest["manifest_sha256"],
                }
            ),
            "dinov2": {
                "model": dino_identity,
                **(
                    {
                        "index_source": "nvidia_base_observation_bank",
                        "observation_bank": observation_bank["identity"],
                        "gallery_aggregation": (
                            "normalized_mean_of_three_standard_rig_views"
                        ),
                    }
                    if observation_bank is not None
                    else {}
                ),
                "processor": {
                    "use_fast": False,
                    "canvas_size": CANVAS_SIZE,
                    "image_mean": mean,
                    "image_std": std,
                    "token_policy": "last_patch_grid_excludes_cls_and_registers",
                    "mask_policy": "eroded_mask_dense_patch_tokens",
                },
            },
        },
        "policy": {
            "siglip_top_k": siglip_top_k,
            "final_top_k": final_top_k,
            "batch_size": batch_size,
            "device": device,
            "fusion": (
                BASE_BANK_FUSION_POLICY
                if observation_bank is not None
                else LEGACY_FUSION_POLICY
            ),
            "gallery_source": (
                "nvidia_base_observation_bank"
                if observation_bank is not None
                else "catalog_thumbnails_and_text"
            ),
            "final_authority": "exact_mdl_render_tournament",
            "missing_mask_policy": "fail_closed_to_legacy_retrieval",
        },
        "groups": results,
        "summary": {
            "group_count": len(results),
            "accepted_group_count": sum(bool(item["accepted"]) for item in results),
            "rejected_group_count": sum(not bool(item["accepted"]) for item in results),
        },
    }
    result = {**unsigned, "integrity": {"result_sha256": _canonical_sha256(unsigned)}}
    output_path = output_dir / "visual_retrieval.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise VisualRetrievalError(f"{label} must be a positive integer")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--siglip2-model", type=Path, required=True)
    parser.add_argument("--dinov2-model", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--observation-bank",
        type=Path,
        help=(
            "sealed NVIDIA Materials/Base observation bank; when supplied, "
            "its standard rig embeddings replace thumbnail gallery features"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--siglip-top-k", type=int, default=DEFAULT_SIGLIP_TOP_K)
    parser.add_argument("--final-top-k", type=int, default=DEFAULT_FINAL_TOP_K)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    siglip_top_k = _positive_int(args.siglip_top_k, "--siglip-top-k")
    final_top_k = _positive_int(args.final_top_k, "--final-top-k")
    batch_size = _positive_int(args.batch_size, "--batch-size")
    result = run(
        request_path=args.request.expanduser().resolve(strict=True),
        siglip_model_path=args.siglip2_model,
        dino_model_path=args.dinov2_model,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir.expanduser().resolve(),
        device=args.device,
        siglip_top_k=siglip_top_k,
        final_top_k=final_top_k,
        batch_size=batch_size,
        observation_bank_dir=args.observation_bank,
    )
    print(
        json.dumps(
            {
                "output": str(
                    args.output_dir.expanduser().resolve() / "visual_retrieval.json"
                ),
                **result["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
