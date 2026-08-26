#!/usr/bin/env python3
"""Read-only MVInverse PBR evidence extraction and robust multiview fusion.

MVInverse emits pixel-aligned intrinsic maps, not USD materials.  This module
keeps that boundary explicit: it samples only cited palette regions (or an
explicit per-group mask), fuses per-view medians, and emits a fail-closed
parameter *suggestion*.  It never opens or authors a USD stage.

The official MVInverse inference script names frames ``000_albedo.png``,
``000_metallic.png`` and ``000_roughness.png``.  Lossless ``.npy``/``.npz``
equivalents are preferred when present so callers can preserve model precision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from ..core.staged_analysis import (
    StagedAnalysisError,
    validate_palette,
)
from ..evidence.color_semantics import evidence_color_labels


SCHEMA_VERSION = "qwen-mvinverse-pbr-evidence/v1"
ALBEDO_COLOR_SPACE = "assumed_srgb_display_0_1"
_GROUP_ID_RE = re.compile(r"G[0-9]{2,4}")
_VIEW_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_CHANNELS = ("albedo", "metallic", "roughness")
_LEDGER_SCHEMA_VERSION = "qwen-mvinverse-inference-ledger/v1"
_LEDGER_MAP_MODES = {
    "albedo": "RGB",
    "metallic": "L",
    "roughness": "L",
    "normal": "RGB",
    "shading": "RGB",
}
class MVInverseEvidenceError(ValueError):
    """Raised when MVInverse evidence is incomplete, ambiguous, or malformed."""


@dataclass(frozen=True)
class EvidencePolicy:
    """Conservative defaults for automatic scalar PBR suggestions."""

    minimum_region_pixels: int = 128
    minimum_color_pixels: int = 64
    minimum_image_fraction: float = 0.0005
    minimum_color_match_fraction: float = 0.20
    minimum_valid_fraction: float = 0.98
    minimum_distinct_views: int = 2
    maximum_albedo_view_mad: float = 0.15
    maximum_metallic_view_iqr: float = 0.20
    maximum_metallic_view_mad: float = 0.12
    maximum_roughness_view_iqr: float = 0.20
    maximum_roughness_view_mad: float = 0.12
    dielectric_metallic_max: float = 0.35
    conductive_metallic_min: float = 0.65
    maximum_mask_aspect_ratio_error: float = 0.02

    def validate(self) -> None:
        for name in (
            "minimum_region_pixels",
            "minimum_color_pixels",
            "minimum_distinct_views",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise MVInverseEvidenceError(
                    f"policy.{name} must be a positive integer"
                )
        for name in (
            "minimum_image_fraction",
            "minimum_color_match_fraction",
            "minimum_valid_fraction",
            "maximum_albedo_view_mad",
            "maximum_metallic_view_iqr",
            "maximum_metallic_view_mad",
            "maximum_roughness_view_iqr",
            "maximum_roughness_view_mad",
            "dielectric_metallic_max",
            "conductive_metallic_min",
            "maximum_mask_aspect_ratio_error",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= float(value) <= 1.0
            ):
                raise MVInverseEvidenceError(
                    f"policy.{name} must be a finite number from 0 to 1"
                )
        if self.minimum_color_pixels > self.minimum_region_pixels:
            raise MVInverseEvidenceError(
                "policy.minimum_color_pixels cannot exceed minimum_region_pixels"
            )
        if self.dielectric_metallic_max >= self.conductive_metallic_min:
            raise MVInverseEvidenceError(
                "dielectric_metallic_max must be lower than conductive_metallic_min"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json_source(
    value: Mapping[str, Any] | str | Path, *, label: str
) -> tuple[Mapping[str, Any], Path | None, Path]:
    if isinstance(value, Mapping):
        return value, None, Path.cwd()
    path = Path(value).expanduser().resolve(strict=True)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MVInverseEvidenceError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise MVInverseEvidenceError(f"{label} must contain a JSON object")
    return document, path, path.parent


def _resolve_artifact_path(value: Any, *, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise MVInverseEvidenceError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise MVInverseEvidenceError(f"{label} does not exist: {path}") from exc


def _manifest_views(
    manifest: Mapping[str, Any], *, manifest_base: Path
) -> list[dict[str, Any]]:
    raw_views = manifest.get("source_views")
    if not isinstance(raw_views, list) or not raw_views:
        raise MVInverseEvidenceError("reference_manifest.source_views cannot be empty")
    views: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_images: set[Path] = set()
    for index, raw in enumerate(raw_views):
        if not isinstance(raw, Mapping):
            raise MVInverseEvidenceError(
                f"reference_manifest.source_views[{index}] must be an object"
            )
        view_id = raw.get("id")
        if not isinstance(view_id, str) or not _VIEW_ID_RE.fullmatch(view_id):
            raise MVInverseEvidenceError(
                f"source_views[{index}].id must be a safe view identifier"
            )
        view_id = view_id.strip()
        if view_id in seen_ids:
            raise MVInverseEvidenceError(f"duplicate source view id: {view_id}")
        seen_ids.add(view_id)
        image = _resolve_artifact_path(
            raw.get("image"), base=manifest_base, label=f"source view {view_id}.image"
        )
        if image in seen_images:
            raise MVInverseEvidenceError(
                f"source views cannot reference the same image twice: {image}"
            )
        seen_images.add(image)
        artifacts = raw.get("palette_artifacts")
        legacy_palette_candidate: Any = None
        if isinstance(artifacts, Mapping):
            # ``normalized`` is authoritative.  ``filtered`` is retained for
            # older staged workspaces which predate normalized multiview groups.
            legacy_palette_candidate = (
                artifacts.get("normalized")
                or artifacts.get("filtered")
                or artifacts.get("pixel_filtered")
            )
        explicit_palette_candidate = raw.get("palette_path")
        if "palette_status" in raw:
            palette_status = raw.get("palette_status")
            if (
                not isinstance(palette_status, str)
                or palette_status not in {"usable", "unusable"}
            ):
                raise MVInverseEvidenceError(
                    f"source view {view_id}.palette_status must be "
                    "'usable' or 'unusable'"
                )
        else:
            # Backward compatibility for manifests written before the status
            # contract: a referenced palette was implicitly usable, while a
            # view without one remained available for MVInverse frame alignment
            # and explicit-mask evidence.
            palette_status = (
                "usable"
                if explicit_palette_candidate is not None
                or legacy_palette_candidate is not None
                else "unusable"
            )

        palette_path: Path | None = None
        if palette_status == "usable":
            candidate = (
                explicit_palette_candidate
                if explicit_palette_candidate is not None
                else legacy_palette_candidate
            )
            if candidate is None:
                raise MVInverseEvidenceError(
                    f"usable source view {view_id} has no palette_path"
                )
            palette_path = _resolve_artifact_path(
                candidate,
                base=manifest_base,
                label=f"source view {view_id}.palette_path",
            )
        # An explicitly unusable view intentionally does not resolve or read
        # any legacy palette path.  Older failed runs persisted error objects at
        # those paths; they are audit artifacts, not normalized palettes.
        views.append(
            {
                "view_id": view_id,
                "image": image,
                "palette_status": palette_status,
                "palette_path": palette_path,
                "manifest_frame_index": raw.get("mvinverse_frame_index"),
            }
        )
    return views


def _frame_mapping(
    views: Sequence[Mapping[str, Any]],
    frame_indices: Mapping[str, int] | None,
) -> tuple[dict[str, int], str]:
    view_ids = {str(view["view_id"]) for view in views}
    if frame_indices is not None:
        if set(frame_indices) != view_ids:
            raise MVInverseEvidenceError(
                "frame_indices must exactly cover source views; "
                f"missing={sorted(view_ids - set(frame_indices))}, "
                f"unexpected={sorted(set(frame_indices) - view_ids)}"
            )
        mapping = dict(frame_indices)
        strategy = "explicit_argument"
    elif all(view.get("manifest_frame_index") is not None for view in views):
        mapping = {str(view["view_id"]): view["manifest_frame_index"] for view in views}
        strategy = "explicit_manifest_indices"
    else:
        basenames = [Path(view["image"]).name for view in views]
        if len(set(basenames)) != len(basenames):
            raise MVInverseEvidenceError(
                "cannot infer official MVInverse frame order from duplicate image basenames; "
                "provide frame_indices"
            )
        ordered = sorted(views, key=lambda view: Path(view["image"]).name)
        mapping = {str(view["view_id"]): index for index, view in enumerate(ordered)}
        strategy = "official_sorted_image_basename"
    values: list[int] = []
    for view_id, value in mapping.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MVInverseEvidenceError(
                f"frame index for {view_id} must be a non-negative integer"
            )
        values.append(value)
    if len(set(values)) != len(values):
        raise MVInverseEvidenceError("frame indices must be unique")
    return {key: int(value) for key, value in mapping.items()}, strategy


def _verified_inference_ledger(
    inference_ledger: Mapping[str, Any] | str | Path,
    *,
    views: Sequence[Mapping[str, Any]],
    output_dir: Path,
    explicit_frame_indices: Mapping[str, int] | None,
) -> tuple[
    dict[str, int],
    dict[tuple[int, str], dict[str, Any]],
    Path | None,
    str,
]:
    """Validate the adapter trust ledger and every member of its five-map set."""

    ledger, ledger_path, ledger_base = _read_json_source(
        inference_ledger, label="inference_ledger"
    )
    if ledger.get("schema_version") != _LEDGER_SCHEMA_VERSION:
        raise MVInverseEvidenceError(
            f"inference ledger schema_version must be {_LEDGER_SCHEMA_VERSION!r}"
        )
    if ledger.get("status") not in {"SUCCESS", "REUSED"}:
        raise MVInverseEvidenceError(
            "inference ledger status must be SUCCESS or REUSED"
        )
    raw_inputs = ledger.get("inputs")
    if not isinstance(raw_inputs, Mapping):
        raise MVInverseEvidenceError("inference_ledger.inputs must be an object")
    ledger_views = raw_inputs.get("source_views")
    if not isinstance(ledger_views, list) or len(ledger_views) != len(views):
        raise MVInverseEvidenceError(
            "inference ledger source_views must exactly cover the reference manifest"
        )
    mapping: dict[str, int] = {}
    for position, (view, record) in enumerate(zip(views, ledger_views)):
        if not isinstance(record, Mapping):
            raise MVInverseEvidenceError(
                f"inference_ledger.inputs.source_views[{position}] must be an object"
            )
        view_id = str(view["view_id"])
        if record.get("view_id") != view_id:
            raise MVInverseEvidenceError(
                "inference ledger source view order/identity differs from the manifest: "
                f"position {position}"
            )
        index = record.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index != position:
            raise MVInverseEvidenceError(
                f"inference ledger frame index must equal manifest position for {view_id}"
            )
        declared_digest = record.get("sha256")
        actual_digest = _sha256(Path(view["image"]))
        if declared_digest != actual_digest:
            raise MVInverseEvidenceError(
                f"inference ledger source image hash differs for {view_id}"
            )
        mapping[view_id] = index
    if len(set(mapping.values())) != len(mapping):
        raise MVInverseEvidenceError("inference ledger frame indices are not unique")
    if explicit_frame_indices is not None:
        explicit, _strategy = _frame_mapping(views, explicit_frame_indices)
        if explicit != mapping:
            raise MVInverseEvidenceError(
                "explicit frame_indices disagree with the verified inference ledger"
            )

    raw_outputs = ledger.get("outputs")
    if not isinstance(raw_outputs, Mapping):
        raise MVInverseEvidenceError("verified inference ledger has no outputs object")
    directory_value = raw_outputs.get("directory")
    if not isinstance(directory_value, str) or not directory_value:
        raise MVInverseEvidenceError("inference_ledger.outputs.directory is invalid")
    ledger_directory = Path(directory_value).expanduser()
    if not ledger_directory.is_absolute():
        ledger_directory = ledger_base / ledger_directory
    try:
        ledger_directory = ledger_directory.resolve(strict=True)
    except OSError as exc:
        raise MVInverseEvidenceError(
            f"inference ledger output directory is missing: {ledger_directory}"
        ) from exc
    if ledger_directory != output_dir:
        raise MVInverseEvidenceError(
            "mvinverse_output_dir must exactly equal inference_ledger.outputs.directory"
        )
    raw_maps = raw_outputs.get("maps")
    expected_count = len(views) * len(_LEDGER_MAP_MODES)
    if (
        not isinstance(raw_maps, list)
        or len(raw_maps) != expected_count
        or raw_outputs.get("map_count") != expected_count
    ):
        raise MVInverseEvidenceError(
            "inference ledger does not contain the complete five-map output set"
        )

    by_index = {index: view_id for view_id, index in mapping.items()}
    records: dict[tuple[int, str], dict[str, Any]] = {}
    expected_names = {
        f"{index:03d}_{map_name}.png"
        for index in by_index
        for map_name in _LEDGER_MAP_MODES
    }
    actual_entries = list(output_dir.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in actual_entries):
        raise MVInverseEvidenceError(
            "verified MVInverse maps directory must contain regular files only"
        )
    if {path.name for path in actual_entries} != expected_names:
        raise MVInverseEvidenceError(
            "verified MVInverse maps directory is incomplete or contaminated"
        )
    for position, raw in enumerate(raw_maps):
        if not isinstance(raw, Mapping):
            raise MVInverseEvidenceError(
                f"inference_ledger.outputs.maps[{position}] must be an object"
            )
        index = raw.get("index")
        map_name = raw.get("map")
        if index not in by_index or map_name not in _LEDGER_MAP_MODES:
            raise MVInverseEvidenceError(
                f"inference ledger contains an unknown map at position {position}"
            )
        if raw.get("view_id") != by_index[index]:
            raise MVInverseEvidenceError(
                f"inference ledger map view_id is inconsistent at position {position}"
            )
        key = (int(index), str(map_name))
        if key in records:
            raise MVInverseEvidenceError(f"duplicate inference ledger map: {key}")
        expected_path = output_dir / f"{index:03d}_{map_name}.png"
        declared_path = raw.get("path")
        if not isinstance(declared_path, str) or not declared_path:
            raise MVInverseEvidenceError(f"inference ledger map path is invalid: {key}")
        candidate_path = Path(declared_path).expanduser()
        if not candidate_path.is_absolute():
            candidate_path = ledger_base / candidate_path
        if candidate_path.resolve(strict=True) != expected_path.resolve(strict=True):
            raise MVInverseEvidenceError(
                f"inference ledger map path is inconsistent: {key}"
            )
        digest = _sha256(expected_path)
        if raw.get("sha256") != digest:
            raise MVInverseEvidenceError(f"inference ledger map hash differs: {key}")
        try:
            with Image.open(expected_path) as opened:
                opened.load()
                actual_size = list(opened.size)
                actual_mode = opened.mode
        except OSError as exc:
            raise MVInverseEvidenceError(
                f"cannot decode verified inference map: {expected_path}"
            ) from exc
        if raw.get("size") != actual_size or raw.get("mode") != actual_mode:
            raise MVInverseEvidenceError(
                f"inference ledger map dimensions/mode differ: {key}"
            )
        if actual_mode != _LEDGER_MAP_MODES[str(map_name)]:
            raise MVInverseEvidenceError(f"inference ledger map mode is unsafe: {key}")
        records[key] = {
            "path": expected_path,
            "sha256": digest,
            "mode": actual_mode,
            "size": actual_size,
        }
    if set(records) != {
        (index, map_name) for index in by_index for map_name in _LEDGER_MAP_MODES
    }:
        raise MVInverseEvidenceError(
            "inference ledger map records do not exactly cover every source frame"
        )
    # The adapter writes records in source/frame then fixed map order.  Rebuild
    # exactly that order to detect ledger list reordering as well as tampering.
    ordered_output_set = _canonical_sha256(
        [
            {
                "index": index,
                "map": map_name,
                "sha256": records[(index, map_name)]["sha256"],
            }
            for index in mapping.values()
            for map_name in _LEDGER_MAP_MODES
        ]
    )
    if raw_outputs.get("output_set_sha256") != ordered_output_set:
        raise MVInverseEvidenceError("inference ledger output_set_sha256 differs")
    if ledger_path is not None:
        ledger_digest = _sha256(ledger_path)
    else:
        ledger_digest = _canonical_sha256(ledger)
    return mapping, records, ledger_path, ledger_digest


def _load_view_palettes(
    views: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for view in views:
        status = view.get("palette_status")
        if status == "unusable":
            continue
        if status != "usable":
            raise MVInverseEvidenceError(
                f"source view {view.get('view_id')}.palette_status is unknown"
            )
        path = view["palette_path"]
        if path is None:
            raise MVInverseEvidenceError(
                f"usable source view {view['view_id']} has no palette_path"
            )
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            canonical = validate_palette(
                raw, allowed_reference_view_ids={view["view_id"]}
            )
        except (OSError, json.JSONDecodeError, StagedAnalysisError) as exc:
            raise MVInverseEvidenceError(
                f"invalid palette for source view {view['view_id']}: {path}: {exc}"
            ) from exc
        result[str(view["view_id"])] = canonical
    return result


def _surface_class(group: Mapping[str, Any]) -> str:
    family = str(group.get("family_hint", "unknown")).lower()
    finish = str(group.get("finish_hint", "unknown")).lower()
    description = str(group.get("visual_description", "")).lower()
    if finish == "painted" or family in {
        "plastic",
        "rubber",
        "glass",
        "fabric",
        "ceramic",
    }:
        return "dielectric"
    conductive_words = (
        "bare metal",
        "unpainted metal",
        "exposed metal",
        "裸金属",
        "未喷漆金属",
    )
    if family == "metal" and (
        finish in {"bare", "brushed", "polished"}
        or any(word in description for word in conductive_words)
    ):
        return "conductive"
    return "unknown"


def _finish_compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    first = str(left["finish_hint"])
    second = str(right["finish_hint"])
    if (
        first == second
        or first in {"other", "unknown"}
        or second in {"other", "unknown"}
    ):
        return True
    families = (
        frozenset({"bare", "brushed", "polished"}),
        frozenset({"matte", "rough"}),
        frozenset({"glossy", "smooth"}),
    )
    return any(first in family and second in family for family in families)


def _family_compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    first = str(left["family_hint"])
    second = str(right["family_hint"])
    if (
        first == second
        or first in {"other", "unknown"}
        or second in {"other", "unknown"}
    ):
        return True
    return _surface_class(left) != "unknown" and _surface_class(left) == _surface_class(
        right
    )


def _matching_groups(
    canonical_group: Mapping[str, Any], local_palette: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    if local_palette is None:
        return []
    return [
        dict(group)
        for group in local_palette["groups"]
        if str(group["base_color"])
        in evidence_color_labels(str(canonical_group["base_color"]))
        and _finish_compatible(canonical_group, group)
        and _family_compatible(canonical_group, group)
    ]


def _normalize_array(array: np.ndarray, *, label: str) -> tuple[np.ndarray, str]:
    raw = np.asarray(array)
    dtype_name = str(raw.dtype)
    if raw.dtype == np.bool_:
        normalized = raw.astype(np.float64)
    elif np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(raw.dtype)
        if info.min < 0:
            raise MVInverseEvidenceError(f"{label} uses a signed integer dtype")
        normalized = raw.astype(np.float64) / float(info.max)
    elif np.issubdtype(raw.dtype, np.floating):
        normalized = raw.astype(np.float64)
    else:
        raise MVInverseEvidenceError(f"{label} has unsupported dtype {raw.dtype}")
    finite = normalized[np.isfinite(normalized)]
    if finite.size and (float(finite.min()) < -1e-6 or float(finite.max()) > 1.000001):
        raise MVInverseEvidenceError(
            f"{label} values must use an unambiguous 0..1 range"
        )
    normalized = np.where(
        np.isfinite(normalized), np.clip(normalized, 0.0, 1.0), np.nan
    )
    return normalized, dtype_name


def _png_bit_depth(path: Path) -> int | None:
    try:
        header = path.read_bytes()[:25]
    except OSError:
        return None
    if (
        len(header) >= 25
        and header[:8] == b"\x89PNG\r\n\x1a\n"
        and header[12:16] == b"IHDR"
    ):
        return int(header[24])
    return None


def _read_png(path: Path, *, label: str) -> tuple[np.ndarray, str]:
    # Pillow preserves 16-bit grayscale but currently down-converts 16-bit RGB.
    # Use OpenCV for the latter when available, retaining a clear error if the
    # optional decoder is absent rather than silently discarding precision.
    bit_depth = _png_bit_depth(path)
    if bit_depth == 16:
        try:
            import cv2  # type: ignore
        except ImportError:
            cv2 = None
        if cv2 is not None:
            decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if decoded is None:
                raise MVInverseEvidenceError(f"cannot decode {label}: {path}")
            if decoded.ndim == 3:
                if decoded.shape[2] == 4:
                    decoded = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA)
                else:
                    decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            return _normalize_array(decoded, label=label)
    with Image.open(path) as opened:
        array = np.asarray(opened.copy())
    return _normalize_array(array, label=label)


def _channel_shape(array: np.ndarray, *, channel: str, label: str) -> np.ndarray:
    if channel == "albedo":
        if array.ndim != 3 or array.shape[2] not in {3, 4}:
            raise MVInverseEvidenceError(f"{label} must have shape HxWx3 or HxWx4")
        return array[:, :, :3]
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    elif array.ndim == 3 and array.shape[2] in {3, 4}:
        rgb = array[:, :, :3]
        if not np.allclose(
            rgb[:, :, 0], rgb[:, :, 1], equal_nan=True
        ) or not np.allclose(rgb[:, :, 0], rgb[:, :, 2], equal_nan=True):
            raise MVInverseEvidenceError(f"{label} must be single-channel")
        array = rgb[:, :, 0]
    if array.ndim != 2:
        raise MVInverseEvidenceError(f"{label} must have shape HxW or HxWx1")
    return array


def _source_record(
    path: Path, *, kind: str, key: str | None, dtype: str
) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "format": kind,
        "key": key,
        "dtype": dtype,
    }


def _load_npz_channels(
    path: Path, *, frame_index: int | None
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]] | None:
    with np.load(path, allow_pickle=False) as archive:
        if not all(channel in archive.files for channel in _CHANNELS):
            return None
        arrays: dict[str, np.ndarray] = {}
        sources: dict[str, dict[str, Any]] = {}
        for channel in _CHANNELS:
            raw = np.asarray(archive[channel])
            if frame_index is not None:
                if raw.ndim < 3 or frame_index >= raw.shape[0]:
                    raise MVInverseEvidenceError(
                        f"{path}:{channel} does not contain frame {frame_index}"
                    )
                raw = raw[frame_index]
            normalized, dtype = _normalize_array(raw, label=f"{path}:{channel}")
            arrays[channel] = _channel_shape(
                normalized, channel=channel, label=f"{path}:{channel}"
            )
            sources[channel] = _source_record(
                path, kind="npz", key=channel, dtype=dtype
            )
    return arrays, sources


def _load_frame(
    output_dir: Path, *, view_id: str, frame_index: int
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    prefixes = (view_id, f"{frame_index:03d}")
    for prefix in prefixes:
        path = output_dir / f"{prefix}.npz"
        if path.is_file():
            loaded = _load_npz_channels(path, frame_index=None)
            if loaded is not None:
                return loaded
    for name in ("predictions.npz", "outputs.npz", "mvinverse_outputs.npz"):
        path = output_dir / name
        if path.is_file():
            loaded = _load_npz_channels(path, frame_index=frame_index)
            if loaded is not None:
                return loaded

    arrays: dict[str, np.ndarray] = {}
    sources: dict[str, dict[str, Any]] = {}
    # Prefer a complete per-frame NPY set, then complete global NPY arrays.
    per_frame_paths: dict[str, Path] | None = None
    for prefix in prefixes:
        candidate = {
            channel: output_dir / f"{prefix}_{channel}.npy" for channel in _CHANNELS
        }
        if all(path.is_file() for path in candidate.values()):
            per_frame_paths = candidate
            break
    if per_frame_paths is not None:
        for channel, path in per_frame_paths.items():
            raw = np.load(path, allow_pickle=False)
            normalized, dtype = _normalize_array(raw, label=str(path))
            arrays[channel] = _channel_shape(
                normalized, channel=channel, label=str(path)
            )
            sources[channel] = _source_record(path, kind="npy", key=None, dtype=dtype)
    elif all((output_dir / f"{channel}.npy").is_file() for channel in _CHANNELS):
        for channel in _CHANNELS:
            path = output_dir / f"{channel}.npy"
            raw = np.load(path, allow_pickle=False)
            if raw.ndim < 3 or frame_index >= raw.shape[0]:
                raise MVInverseEvidenceError(
                    f"{path} does not contain frame {frame_index}"
                )
            normalized, dtype = _normalize_array(raw[frame_index], label=str(path))
            arrays[channel] = _channel_shape(
                normalized, channel=channel, label=str(path)
            )
            sources[channel] = _source_record(
                path, kind="npy", key=f"frame:{frame_index}", dtype=dtype
            )
    else:
        paths: dict[str, Path] | None = None
        for prefix in prefixes:
            candidate = {
                channel: output_dir / f"{prefix}_{channel}.png" for channel in _CHANNELS
            }
            if all(path.is_file() for path in candidate.values()):
                paths = candidate
                break
        if paths is None:
            expected = ", ".join(
                f"{frame_index:03d}_{channel}.png" for channel in _CHANNELS
            )
            raise MVInverseEvidenceError(
                f"missing a complete MVInverse PBR frame for {view_id}; expected {expected} "
                "or a complete NPY/NPZ equivalent"
            )
        for channel, path in paths.items():
            normalized, dtype = _read_png(path, label=str(path))
            arrays[channel] = _channel_shape(
                normalized, channel=channel, label=str(path)
            )
            sources[channel] = _source_record(path, kind="png", key=None, dtype=dtype)

    shapes = {channel: value.shape[:2] for channel, value in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise MVInverseEvidenceError(
            f"MVInverse channel dimensions disagree for {view_id}: {shapes}"
        )
    return arrays, sources


def _load_verified_ledger_frame(
    records: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    view_id: str,
    frame_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    arrays: dict[str, np.ndarray] = {}
    sources: dict[str, dict[str, Any]] = {}
    for channel in _CHANNELS:
        record = records.get((frame_index, channel))
        if record is None:
            raise MVInverseEvidenceError(
                f"verified inference ledger is missing {view_id}/{channel}"
            )
        path = Path(record["path"])
        normalized, dtype = _read_png(path, label=str(path))
        arrays[channel] = _channel_shape(normalized, channel=channel, label=str(path))
        digest_after_decode = _sha256(path)
        if digest_after_decode != record["sha256"]:
            raise MVInverseEvidenceError(
                f"verified inference map changed while reading: {path}"
            )
        sources[channel] = _source_record(path, kind="png", key=None, dtype=dtype)
    shapes = {channel: value.shape[:2] for channel, value in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise MVInverseEvidenceError(
            f"verified MVInverse channel dimensions disagree for {view_id}: {shapes}"
        )
    return arrays, sources


def _boxes_mask(boxes: Sequence[Sequence[int]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for box in boxes:
        left = max(0, int(math.floor(int(box[0]) * width / 1000)))
        top = max(0, int(math.floor(int(box[1]) * height / 1000)))
        right = min(width, int(math.ceil(int(box[2]) * width / 1000)))
        bottom = min(height, int(math.ceil(int(box[3]) * height / 1000)))
        mask[top:bottom, left:right] = True
    return mask


def _load_mask(
    path: Path, *, height: int, width: int, policy: EvidencePolicy
) -> tuple[np.ndarray, bool]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        raw = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "mask" in archive.files:
                raw = np.asarray(archive["mask"])
            elif len(archive.files) == 1:
                raw = np.asarray(archive[archive.files[0]])
            else:
                raise MVInverseEvidenceError(
                    f"mask NPZ must contain a 'mask' key or exactly one array: {path}"
                )
    else:
        with Image.open(path) as opened:
            if "A" in opened.getbands() and opened.getchannel("A").getextrema() != (
                255,
                255,
            ):
                raw = np.asarray(opened.getchannel("A").copy())
            else:
                raw = np.asarray(opened.convert("L"))
    raw = np.asarray(raw)
    if raw.ndim == 3 and raw.shape[2] == 1:
        raw = raw[:, :, 0]
    if raw.ndim != 2:
        raise MVInverseEvidenceError(f"explicit mask must be HxW: {path}")
    if np.issubdtype(raw.dtype, np.floating):
        finite = np.isfinite(raw)
        if not finite.all():
            raw = np.where(finite, raw, 0.0)
        maximum = float(np.max(raw)) if raw.size else 0.0
        threshold = 0.5 if maximum <= 1.0 else maximum * 0.5
    else:
        maximum = int(np.max(raw)) if raw.size else 0
        threshold = 0 if maximum <= 1 else maximum * 0.5
    binary = np.asarray(raw > threshold, dtype=bool)
    source_height, source_width = raw.shape
    resized = (source_height, source_width) != (height, width)
    if resized:
        source_ratio = source_width / max(1, source_height)
        target_ratio = width / max(1, height)
        relative_error = abs(source_ratio / target_ratio - 1.0)
        if relative_error > policy.maximum_mask_aspect_ratio_error:
            raise MVInverseEvidenceError(
                f"mask aspect ratio does not match MVInverse output for {path}: "
                f"{source_width}x{source_height} vs {width}x{height}"
            )
        plane = Image.fromarray(binary.astype(np.uint8) * 255)
        binary = np.asarray(plane.resize((width, height), Image.Resampling.NEAREST)) > 0
    return binary, resized


def _albedo_labels(albedo: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.rint(albedo * 255.0), 0, 255).astype(np.uint8)
    red = rgb[:, :, 0].astype(np.float64)
    green = rgb[:, :, 1].astype(np.float64)
    blue = rgb[:, :, 2].astype(np.float64)
    maximum = np.maximum(np.maximum(red, green), blue)
    minimum = np.minimum(np.minimum(red, green), blue)
    delta = maximum - minimum
    value = maximum / 255.0
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum != 0)
    labels = np.full(maximum.shape, "other", dtype="<U7")
    dark = maximum < 55
    labels[dark] = "black"
    achromatic = (~dark) & (saturation < 0.14)
    labels[achromatic & (minimum > 215)] = "white"
    labels[achromatic & (minimum <= 215) & (value > 0.68)] = "silver"
    labels[achromatic & (minimum <= 215) & (value <= 0.68)] = "gray"
    chromatic = (~dark) & (~achromatic)
    hue = np.zeros_like(maximum)
    nonzero = delta != 0
    red_max = nonzero & (maximum == red)
    green_max = nonzero & (maximum == green)
    blue_max = nonzero & (maximum == blue)
    hue[red_max] = (60.0 * ((green[red_max] - blue[red_max]) / delta[red_max])) % 360.0
    hue[green_max] = 60.0 * (
        (blue[green_max] - red[green_max]) / delta[green_max] + 2.0
    )
    hue[blue_max] = 60.0 * ((red[blue_max] - green[blue_max]) / delta[blue_max] + 4.0)
    labels[chromatic & ((hue < 15.0) | (hue >= 345.0))] = "red"
    orange_band = chromatic & (hue >= 15.0) & (hue < 45.0)
    labels[orange_band & (value < 0.55)] = "brown"
    labels[orange_band & (value >= 0.55)] = "orange"
    labels[chromatic & (hue >= 45.0) & (hue < 70.0)] = "yellow"
    labels[chromatic & (hue >= 70.0) & (hue < 170.0)] = "green"
    labels[chromatic & (hue >= 170.0) & (hue < 200.0)] = "cyan"
    labels[chromatic & (hue >= 200.0) & (hue < 260.0)] = "blue"
    labels[chromatic & (hue >= 260.0) & (hue < 345.0)] = "pink"
    return labels


def _color_mask(
    albedo: np.ndarray, expected: str, roi: np.ndarray
) -> tuple[np.ndarray, bool]:
    if expected in {"other", "unknown", "clear"}:
        return roi.copy(), False
    accepted = evidence_color_labels(expected)
    labels = _albedo_labels(albedo)
    return roi & np.isin(labels, list(accepted)), True


def _number(value: Any) -> float:
    return round(float(value), 8)


def _statistics(values: np.ndarray) -> dict[str, Any] | None:
    array = np.asarray(values, dtype=np.float64)
    if array.shape[0] == 0:
        return None
    median = np.median(array, axis=0)
    q1 = np.quantile(array, 0.25, axis=0)
    q3 = np.quantile(array, 0.75, axis=0)
    mad = np.median(np.abs(array - median), axis=0)

    def convert(value: np.ndarray | np.floating | float) -> float | list[float]:
        item = np.asarray(value)
        if item.ndim == 0:
            return _number(item)
        return [_number(component) for component in item.tolist()]

    return {
        "sample_count": int(array.shape[0]),
        "median": convert(median),
        "q1": convert(q1),
        "q3": convert(q3),
        "iqr": convert(q3 - q1),
        "mad": convert(mad),
    }


def _sample_group(
    arrays: Mapping[str, np.ndarray],
    *,
    canonical_group: Mapping[str, Any],
    local_group: Mapping[str, Any] | None,
    association_status: str,
    candidate_group_ids: list[str],
    mask_path: Path | None,
    require_explicit_mask: bool,
    policy: EvidencePolicy,
) -> dict[str, Any]:
    height, width = arrays["metallic"].shape
    mask_resized = False
    if mask_path is not None:
        roi, mask_resized = _load_mask(
            mask_path, height=height, width=width, policy=policy
        )
        source = "explicit_mask"
        boxes: list[list[int]] = []
        evidence_mask = {
            "path": str(mask_path),
            "sha256": _sha256(mask_path),
            "resized_to_output": mask_resized,
        }
    elif require_explicit_mask:
        return {
            "group_id": canonical_group["group_id"],
            "association": {
                "status": "unmatched",
                "candidate_group_ids": [],
                "matched_group_id": None,
            },
            "evidence_source": None,
            "boxes": [],
            "mask": None,
            "region_pixels": 0,
            "image_pixels": int(height * width),
            "image_fraction": 0.0,
            "color_filter_applied": canonical_group["base_color"]
            not in {"other", "unknown", "clear"},
            "color_matching_pixels": 0,
            "color_match_fraction": 0.0,
            "valid_pixels": 0,
            "valid_fraction": 0.0,
            "albedo": None,
            "metallic": None,
            "roughness": None,
            "accepted": False,
            "reason_codes": ["missing_authoritative_region_mask"],
        }
    elif local_group is not None and association_status == "matched":
        boxes = [list(box) for box in local_group["boxes"]]
        roi = _boxes_mask(boxes, height, width)
        source = "palette_boxes"
        evidence_mask = None
    else:
        return {
            "group_id": canonical_group["group_id"],
            "association": {
                "status": association_status,
                "candidate_group_ids": candidate_group_ids,
                "matched_group_id": None,
            },
            "evidence_source": None,
            "boxes": [],
            "mask": None,
            "region_pixels": 0,
            "image_pixels": int(height * width),
            "image_fraction": 0.0,
            "color_filter_applied": canonical_group["base_color"]
            not in {"other", "unknown", "clear"},
            "color_matching_pixels": 0,
            "color_match_fraction": 0.0,
            "valid_pixels": 0,
            "valid_fraction": 0.0,
            "albedo": None,
            "metallic": None,
            "roughness": None,
            "accepted": False,
            "reason_codes": [
                (
                    "ambiguous_palette_association"
                    if association_status == "ambiguous"
                    else "no_compatible_palette_group"
                )
            ],
        }

    region_pixels = int(np.count_nonzero(roi))
    color_roi, color_filter_applied = _color_mask(
        arrays["albedo"], str(canonical_group["base_color"]), roi
    )
    color_pixels = int(np.count_nonzero(color_roi))
    finite = (
        np.all(np.isfinite(arrays["albedo"]), axis=2)
        & np.isfinite(arrays["metallic"])
        & np.isfinite(arrays["roughness"])
    )
    valid = color_roi & finite
    valid_pixels = int(np.count_nonzero(valid))
    image_pixels = int(height * width)
    image_fraction = region_pixels / max(1, image_pixels)
    color_match_fraction = color_pixels / max(1, region_pixels)
    valid_fraction = valid_pixels / max(1, color_pixels)
    reasons: list[str] = []
    if region_pixels < policy.minimum_region_pixels:
        reasons.append("insufficient_region_pixels")
    if image_fraction < policy.minimum_image_fraction:
        reasons.append("insufficient_image_coverage")
    if color_pixels < policy.minimum_color_pixels:
        reasons.append("insufficient_color_pixels")
    if (
        color_filter_applied
        and color_match_fraction < policy.minimum_color_match_fraction
    ):
        reasons.append("insufficient_albedo_color_match")
    if valid_fraction < policy.minimum_valid_fraction:
        reasons.append("insufficient_valid_pbr_coverage")
    albedo_values = arrays["albedo"][valid]
    metallic_values = arrays["metallic"][valid]
    roughness_values = arrays["roughness"][valid]
    return {
        "group_id": canonical_group["group_id"],
        "association": {
            "status": "explicit_mask" if mask_path is not None else association_status,
            "candidate_group_ids": candidate_group_ids,
            "matched_group_id": (
                local_group["group_id"] if local_group is not None else None
            ),
        },
        "evidence_source": source,
        "boxes": boxes,
        "mask": evidence_mask,
        "region_pixels": region_pixels,
        "image_pixels": image_pixels,
        "image_fraction": _number(image_fraction),
        "color_filter_applied": color_filter_applied,
        "color_matching_pixels": color_pixels,
        "color_match_fraction": _number(color_match_fraction),
        "valid_pixels": valid_pixels,
        "valid_fraction": _number(valid_fraction),
        "albedo": _statistics(albedo_values),
        "metallic": _statistics(metallic_values),
        "roughness": _statistics(roughness_values),
        "accepted": not reasons,
        "reason_codes": reasons,
    }


def _maximum(value: float | list[float]) -> float:
    if isinstance(value, list):
        return max(value) if value else 0.0
    return float(value)


def _fuse_group(
    canonical_group: Mapping[str, Any],
    view_records: Sequence[Mapping[str, Any]],
    semantic_groups: Mapping[str, Mapping[str, Any] | None],
    policy: EvidencePolicy,
    *,
    integrity_verified: bool,
) -> dict[str, Any]:
    group_id = str(canonical_group["group_id"])
    records = [
        next(group for group in view["groups"] if group["group_id"] == group_id)
        for view in view_records
    ]
    accepted = [
        (view["view_id"], record)
        for view, record in zip(view_records, records)
        if record["accepted"]
    ]
    contributing_ids = [str(view_id) for view_id, _record in accepted]
    ambiguous_views = [
        str(view["view_id"])
        for view, record in zip(view_records, records)
        if record["association"]["status"] == "ambiguous"
    ]
    unmatched_views = [
        str(view["view_id"])
        for view, record in zip(view_records, records)
        if record["association"]["status"] == "unmatched"
    ]
    rejected_views = [
        {
            "view_id": str(view["view_id"]),
            "reason_codes": list(record["reason_codes"]),
        }
        for view, record in zip(view_records, records)
        if record["association"]["status"] in {"matched", "explicit_mask"}
        and not record["accepted"]
    ]
    if accepted:
        albedo = _statistics(
            np.asarray([record["albedo"]["median"] for _view, record in accepted])
        )
        metallic = _statistics(
            np.asarray([record["metallic"]["median"] for _view, record in accepted])
        )
        roughness = _statistics(
            np.asarray([record["roughness"]["median"] for _view, record in accepted])
        )
    else:
        albedo = metallic = roughness = None

    observations = [
        {
            "view_id": "canonical",
            "group_id": group_id,
            "family_hint": canonical_group["family_hint"],
            "finish_hint": canonical_group["finish_hint"],
            "surface_class": _surface_class(canonical_group),
        }
    ]
    for view_id in contributing_ids:
        semantic = semantic_groups.get(view_id)
        if semantic is not None:
            observations.append(
                {
                    "view_id": view_id,
                    "group_id": semantic["group_id"],
                    "family_hint": semantic["family_hint"],
                    "finish_hint": semantic["finish_hint"],
                    "surface_class": _surface_class(semantic),
                }
            )
    known_classes = {
        observation["surface_class"]
        for observation in observations
        if observation["surface_class"] != "unknown"
    }
    if len(known_classes) > 1:
        surface_class = "conflict"
    elif known_classes:
        surface_class = next(iter(known_classes))
    else:
        surface_class = "unknown"

    preserve_reasons: list[str] = []
    if not integrity_verified:
        preserve_reasons.append("unverified_inference_source")
    if len(accepted) < policy.minimum_distinct_views:
        preserve_reasons.append("insufficient_distinct_views")
    if surface_class == "unknown":
        preserve_reasons.append("semantic_class_unknown")
    elif surface_class == "conflict":
        preserve_reasons.append("semantic_class_conflict")
    if albedo is not None and _maximum(albedo["mad"]) > policy.maximum_albedo_view_mad:
        preserve_reasons.append("albedo_cross_view_dispersion")
    if metallic is not None and (
        float(metallic["iqr"]) > policy.maximum_metallic_view_iqr
        or float(metallic["mad"]) > policy.maximum_metallic_view_mad
    ):
        preserve_reasons.append("metallic_cross_view_dispersion")
    if roughness is not None and (
        float(roughness["iqr"]) > policy.maximum_roughness_view_iqr
        or float(roughness["mad"]) > policy.maximum_roughness_view_mad
    ):
        preserve_reasons.append("roughness_cross_view_dispersion")
    if (
        metallic is not None
        and surface_class == "dielectric"
        and float(metallic["median"]) > policy.dielectric_metallic_max
    ):
        preserve_reasons.append("dielectric_metallicity_conflict")
    if (
        metallic is not None
        and surface_class == "conductive"
        and float(metallic["median"]) < policy.conductive_metallic_min
    ):
        preserve_reasons.append("conductive_metallicity_conflict")

    warning_codes: list[str] = []
    if ambiguous_views:
        warning_codes.append("ambiguous_views_ignored")
    if unmatched_views:
        warning_codes.append("unmatched_views_ignored")
    if rejected_views:
        warning_codes.append("low_quality_views_ignored")
    auto = not preserve_reasons
    reason_codes = (
        [
            "multi_view_evidence_sufficient",
            "cross_view_dispersion_within_limits",
            "semantic_metallicity_supported",
        ]
        if auto
        else preserve_reasons
    )
    suggestion = {
        "decision": "auto" if auto else "preserve",
        "auto_parameter_eligible": auto,
        "base_color_srgb": albedo["median"] if auto and albedo is not None else None,
        "metallic": (
            0.0
            if auto and surface_class == "dielectric"
            else 1.0
            if auto and surface_class == "conductive"
            else None
        ),
        "roughness": roughness["median"] if auto and roughness is not None else None,
        "reason_codes": reason_codes,
        "warning_codes": warning_codes,
    }
    return {
        "group_id": group_id,
        "canonical_semantic": {
            "family_hint": canonical_group["family_hint"],
            "base_color": canonical_group["base_color"],
            "finish_hint": canonical_group["finish_hint"],
            "visual_description": canonical_group["visual_description"],
        },
        "surface_class": surface_class,
        "semantic_observations": observations,
        "contributing_view_ids": contributing_ids,
        "distinct_view_count": len(contributing_ids),
        "ambiguous_views": ambiguous_views,
        "unmatched_views": unmatched_views,
        "rejected_views": rejected_views,
        "albedo": albedo,
        "metallic": metallic,
        "roughness": roughness,
        "suggestion": suggestion,
    }


def _resolved_masks(
    masks: Mapping[tuple[str, str], str | Path] | None,
    *,
    view_ids: set[str],
    group_ids: set[str],
) -> dict[tuple[str, str], Path]:
    result: dict[tuple[str, str], Path] = {}
    for key, raw_path in (masks or {}).items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise MVInverseEvidenceError("mask keys must be (view_id, group_id) tuples")
        view_id, group_id = key
        if view_id not in view_ids:
            raise MVInverseEvidenceError(f"mask references unknown view_id: {view_id}")
        if group_id not in group_ids:
            raise MVInverseEvidenceError(
                f"mask references unknown canonical group_id: {group_id}"
            )
        result[(view_id, group_id)] = Path(raw_path).expanduser().resolve(strict=True)
    return result


def build_mvinverse_evidence_from_manifest(
    mvinverse_output_dir: str | Path,
    reference_manifest: Mapping[str, Any] | str | Path,
    canonical_palette: Mapping[str, Any] | str | Path,
    *,
    masks: Mapping[tuple[str, str], str | Path] | None = None,
    require_explicit_masks: bool = False,
    frame_indices: Mapping[str, int] | None = None,
    inference_ledger: Mapping[str, Any] | str | Path | None = None,
    policy: EvidencePolicy | None = None,
    path_base: str | Path | None = None,
) -> dict[str, Any]:
    """Build strict multiview PBR evidence from a staged reference manifest.

    A local palette candidate contributes only if matching the canonical group
    by base color and compatible finish/family produces exactly one candidate.
    Explicit masks are authoritative and may disambiguate a view.  Automatic
    parameter eligibility additionally requires a verified adapter ledger;
    raw-directory mode is useful for development but always preserves.
    """

    resolved_policy = policy or EvidencePolicy()
    resolved_policy.validate()
    output_dir = Path(mvinverse_output_dir).expanduser().resolve(strict=True)
    if not output_dir.is_dir():
        raise MVInverseEvidenceError(
            f"MVInverse output is not a directory: {output_dir}"
        )
    manifest, manifest_path, manifest_base = _read_json_source(
        reference_manifest, label="reference_manifest"
    )
    palette_raw, palette_path, _palette_base = _read_json_source(
        canonical_palette, label="canonical_palette"
    )
    try:
        canonical = validate_palette(palette_raw)
    except StagedAnalysisError as exc:
        raise MVInverseEvidenceError(f"invalid canonical palette: {exc}") from exc
    views = _manifest_views(manifest, manifest_base=manifest_base)
    ledger_records: dict[tuple[int, str], dict[str, Any]] | None = None
    ledger_path: Path | None = None
    ledger_digest: str | None = None
    if inference_ledger is not None:
        mapping, ledger_records, ledger_path, ledger_digest = (
            _verified_inference_ledger(
                inference_ledger,
                views=views,
                output_dir=output_dir,
                explicit_frame_indices=frame_indices,
            )
        )
        mapping_strategy = "verified_inference_ledger"
    else:
        mapping, mapping_strategy = _frame_mapping(views, frame_indices)
    palettes = _load_view_palettes(views)
    view_ids = {str(view["view_id"]) for view in views}
    group_ids = {str(group["group_id"]) for group in canonical["groups"]}
    resolved_masks = _resolved_masks(masks, view_ids=view_ids, group_ids=group_ids)

    view_records: list[dict[str, Any]] = []
    semantic_by_group: dict[str, dict[str, Mapping[str, Any] | None]] = {
        group_id: {} for group_id in group_ids
    }
    for view in views:
        view_id = str(view["view_id"])
        frame_index = mapping[view_id]
        if ledger_records is not None:
            arrays, sources = _load_verified_ledger_frame(
                ledger_records, view_id=view_id, frame_index=frame_index
            )
        else:
            arrays, sources = _load_frame(
                output_dir, view_id=view_id, frame_index=frame_index
            )
        height, width = arrays["metallic"].shape
        local_palette = palettes.get(view_id)
        group_records: list[dict[str, Any]] = []
        for canonical_group in canonical["groups"]:
            group_id = str(canonical_group["group_id"])
            candidates = _matching_groups(canonical_group, local_palette)
            candidate_ids = [str(group["group_id"]) for group in candidates]
            local_group = candidates[0] if len(candidates) == 1 else None
            if len(candidates) == 1:
                status = "matched"
            elif candidates:
                status = "ambiguous"
            else:
                status = "unmatched"
            mask_path = resolved_masks.get((view_id, group_id))
            record = _sample_group(
                arrays,
                canonical_group=canonical_group,
                local_group=local_group,
                association_status=status,
                candidate_group_ids=candidate_ids,
                mask_path=mask_path,
                require_explicit_mask=require_explicit_masks,
                policy=resolved_policy,
            )
            group_records.append(record)
            semantic_by_group[group_id][view_id] = local_group
        view_records.append(
            {
                "view_id": view_id,
                "frame_index": frame_index,
                "resolution": [width, height],
                "sources": sources,
                "groups": group_records,
            }
        )

    fused = [
        _fuse_group(
            group,
            view_records,
            semantic_by_group[str(group["group_id"])],
            resolved_policy,
            integrity_verified=ledger_records is not None,
        )
        for group in canonical["groups"]
    ]
    auto_count = sum(group["suggestion"]["auto_parameter_eligible"] for group in fused)
    report_base = (
        Path(path_base).expanduser().resolve() if path_base is not None else None
    )

    def report_path(path: Path | None) -> str | None:
        if path is None:
            return None
        if report_base is None:
            return str(path)
        return Path(os.path.relpath(path, start=report_base)).as_posix()

    if report_base is not None:
        for view in view_records:
            for source in view["sources"].values():
                source["path"] = report_path(Path(source["path"]))

    report = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "mvinverse_output_dir": report_path(output_dir),
            "reference_manifest": report_path(manifest_path),
            "canonical_palette": report_path(palette_path),
            "frame_mapping_strategy": mapping_strategy,
            "albedo_color_space": ALBEDO_COLOR_SPACE,
            "inference_ledger": report_path(ledger_path),
            "inference_ledger_sha256": ledger_digest,
            "integrity_verified": ledger_records is not None,
        },
        "policy": asdict(resolved_policy),
        "views": view_records,
        "groups": fused,
        "summary": {
            "view_count": len(view_records),
            "canonical_group_count": len(fused),
            "auto_parameter_group_count": int(auto_count),
            "preserve_group_count": len(fused) - int(auto_count),
            "fail_closed": True,
            "usd_modified": False,
        },
    }
    return validate_mvinverse_evidence(report)


def _exact(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise MVInverseEvidenceError(f"{label} must be an object")
    if set(value) != expected:
        raise MVInverseEvidenceError(
            f"{label} fields are invalid; unexpected={sorted(set(value) - expected)}, "
            f"missing={sorted(expected - set(value))}"
        )


def _probability(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise MVInverseEvidenceError(f"{label} must be a finite number from 0 to 1")
    return float(value)


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise MVInverseEvidenceError(f"{label} must be an array of non-empty strings")
    if len(value) != len(set(value)):
        raise MVInverseEvidenceError(f"{label} contains duplicates")
    return value


def _nonnegative_integer(value: Any, label: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise MVInverseEvidenceError(f"{label} must be a {qualifier} integer")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MVInverseEvidenceError(f"{label} must be a non-empty string")
    return value


def _validate_source(value: Any, label: str) -> None:
    _exact(value, {"path", "sha256", "format", "key", "dtype"}, label)
    _nonempty_string(value["path"], f"{label}.path")
    digest = _nonempty_string(value["sha256"], f"{label}.sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise MVInverseEvidenceError(f"{label}.sha256 must be lowercase SHA-256")
    if value["format"] not in {"npy", "npz", "png"}:
        raise MVInverseEvidenceError(f"{label}.format is unsupported")
    if value["key"] is not None and not isinstance(value["key"], str):
        raise MVInverseEvidenceError(f"{label}.key must be a string or null")
    _nonempty_string(value["dtype"], f"{label}.dtype")


def _validate_stats(value: Any, label: str, *, vector: bool) -> None:
    if value is None:
        return
    _exact(value, {"sample_count", "median", "q1", "q3", "iqr", "mad"}, label)
    _nonnegative_integer(value["sample_count"], f"{label}.sample_count", positive=True)
    for field in ("median", "q1", "q3", "iqr", "mad"):
        raw = value[field]
        values = raw if isinstance(raw, list) else [raw]
        if vector != isinstance(raw, list) or (vector and len(values) != 3):
            raise MVInverseEvidenceError(f"{label}.{field} has the wrong shape")
        for index, item in enumerate(values):
            _probability(item, f"{label}.{field}[{index}]")
    median = np.atleast_1d(np.asarray(value["median"], dtype=np.float64))
    q1 = np.atleast_1d(np.asarray(value["q1"], dtype=np.float64))
    q3 = np.atleast_1d(np.asarray(value["q3"], dtype=np.float64))
    iqr = np.atleast_1d(np.asarray(value["iqr"], dtype=np.float64))
    if np.any(q1 > median + 1e-7) or np.any(median > q3 + 1e-7):
        raise MVInverseEvidenceError(f"{label} quantiles are inconsistent")
    if not np.allclose(iqr, q3 - q1, atol=2e-7, rtol=0.0):
        raise MVInverseEvidenceError(f"{label}.iqr is inconsistent with q1/q3")


def validate_mvinverse_evidence(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact persisted evidence contract and return a deep copy."""

    _exact(
        document,
        {"schema_version", "inputs", "policy", "views", "groups", "summary"},
        "report",
    )
    if document["schema_version"] != SCHEMA_VERSION:
        raise MVInverseEvidenceError(f"schema_version must be {SCHEMA_VERSION!r}")
    _exact(
        document["inputs"],
        {
            "mvinverse_output_dir",
            "reference_manifest",
            "canonical_palette",
            "frame_mapping_strategy",
            "albedo_color_space",
            "inference_ledger",
            "inference_ledger_sha256",
            "integrity_verified",
        },
        "report.inputs",
    )
    if document["inputs"]["albedo_color_space"] != ALBEDO_COLOR_SPACE:
        raise MVInverseEvidenceError("report.inputs.albedo_color_space is unsupported")
    _nonempty_string(
        document["inputs"]["mvinverse_output_dir"],
        "report.inputs.mvinverse_output_dir",
    )
    for field in ("reference_manifest", "canonical_palette", "inference_ledger"):
        if document["inputs"][field] is not None:
            _nonempty_string(document["inputs"][field], f"report.inputs.{field}")
    if document["inputs"]["frame_mapping_strategy"] not in {
        "explicit_argument",
        "explicit_manifest_indices",
        "official_sorted_image_basename",
        "verified_inference_ledger",
    }:
        raise MVInverseEvidenceError("report.inputs.frame_mapping_strategy is invalid")
    integrity_verified = document["inputs"]["integrity_verified"]
    if not isinstance(integrity_verified, bool):
        raise MVInverseEvidenceError("report.inputs.integrity_verified must be boolean")
    ledger_digest = document["inputs"]["inference_ledger_sha256"]
    if integrity_verified:
        if not isinstance(ledger_digest, str) or not re.fullmatch(
            r"[0-9a-f]{64}", ledger_digest
        ):
            raise MVInverseEvidenceError(
                "verified report must contain inference_ledger_sha256"
            )
        if document["inputs"]["frame_mapping_strategy"] != "verified_inference_ledger":
            raise MVInverseEvidenceError(
                "verified report must use the inference-ledger frame mapping"
            )
    elif (
        document["inputs"]["inference_ledger"] is not None or ledger_digest is not None
    ):
        raise MVInverseEvidenceError(
            "unverified report cannot identify an inference ledger"
        )
    elif document["inputs"]["frame_mapping_strategy"] == "verified_inference_ledger":
        raise MVInverseEvidenceError(
            "unverified report cannot use the inference-ledger frame mapping"
        )
    if not isinstance(document["policy"], Mapping):
        raise MVInverseEvidenceError("report.policy must be an object")
    expected_policy = set(EvidencePolicy.__dataclass_fields__)
    _exact(document["policy"], expected_policy, "report.policy")
    try:
        EvidencePolicy(**document["policy"]).validate()
    except TypeError as exc:
        raise MVInverseEvidenceError(f"report.policy is invalid: {exc}") from exc
    views = document["views"]
    groups = document["groups"]
    if (
        not isinstance(views, list)
        or not views
        or not isinstance(groups, list)
        or not groups
    ):
        raise MVInverseEvidenceError("report views and groups cannot be empty")
    group_ids: list[str] = []
    auto_count = 0
    for index, group in enumerate(groups):
        label = f"report.groups[{index}]"
        _exact(
            group,
            {
                "group_id",
                "canonical_semantic",
                "surface_class",
                "semantic_observations",
                "contributing_view_ids",
                "distinct_view_count",
                "ambiguous_views",
                "unmatched_views",
                "rejected_views",
                "albedo",
                "metallic",
                "roughness",
                "suggestion",
            },
            label,
        )
        group_id = group["group_id"]
        if not isinstance(group_id, str) or not _GROUP_ID_RE.fullmatch(group_id):
            raise MVInverseEvidenceError(f"{label}.group_id is invalid")
        group_ids.append(group_id)
        _exact(
            group["canonical_semantic"],
            {"family_hint", "base_color", "finish_hint", "visual_description"},
            f"{label}.canonical_semantic",
        )
        for field in ("family_hint", "base_color", "finish_hint", "visual_description"):
            _nonempty_string(
                group["canonical_semantic"][field],
                f"{label}.canonical_semantic.{field}",
            )
        if group["surface_class"] not in {
            "dielectric",
            "conductive",
            "unknown",
            "conflict",
        }:
            raise MVInverseEvidenceError(f"{label}.surface_class is invalid")
        observations = group["semantic_observations"]
        if not isinstance(observations, list) or not observations:
            raise MVInverseEvidenceError(
                f"{label}.semantic_observations cannot be empty"
            )
        for observation_index, observation in enumerate(observations):
            observation_label = f"{label}.semantic_observations[{observation_index}]"
            _exact(
                observation,
                {"view_id", "group_id", "family_hint", "finish_hint", "surface_class"},
                observation_label,
            )
            _nonempty_string(observation["view_id"], f"{observation_label}.view_id")
            if not isinstance(
                observation["group_id"], str
            ) or not _GROUP_ID_RE.fullmatch(observation["group_id"]):
                raise MVInverseEvidenceError(f"{observation_label}.group_id is invalid")
            _nonempty_string(
                observation["family_hint"], f"{observation_label}.family_hint"
            )
            _nonempty_string(
                observation["finish_hint"], f"{observation_label}.finish_hint"
            )
            if observation["surface_class"] not in {
                "dielectric",
                "conductive",
                "unknown",
            }:
                raise MVInverseEvidenceError(
                    f"{observation_label}.surface_class is invalid"
                )
        _validate_stats(group["albedo"], f"{label}.albedo", vector=True)
        _validate_stats(group["metallic"], f"{label}.metallic", vector=False)
        _validate_stats(group["roughness"], f"{label}.roughness", vector=False)
        contributing = _string_list(
            group["contributing_view_ids"], f"{label}.contributing_view_ids"
        )
        _nonnegative_integer(
            group["distinct_view_count"], f"{label}.distinct_view_count"
        )
        if group["distinct_view_count"] != len(contributing):
            raise MVInverseEvidenceError(f"{label}.distinct_view_count is inconsistent")
        for stats_field in ("albedo", "metallic", "roughness"):
            stats = group[stats_field]
            if (stats is None) != (not contributing):
                raise MVInverseEvidenceError(
                    f"{label}.{stats_field} presence is inconsistent with contributing views"
                )
            if stats is not None and stats["sample_count"] != len(contributing):
                raise MVInverseEvidenceError(
                    f"{label}.{stats_field}.sample_count must count contributing views"
                )
        _string_list(group["ambiguous_views"], f"{label}.ambiguous_views")
        _string_list(group["unmatched_views"], f"{label}.unmatched_views")
        if not isinstance(group["rejected_views"], list):
            raise MVInverseEvidenceError(f"{label}.rejected_views must be an array")
        for rejected_index, rejected in enumerate(group["rejected_views"]):
            rejected_label = f"{label}.rejected_views[{rejected_index}]"
            _exact(rejected, {"view_id", "reason_codes"}, rejected_label)
            _nonempty_string(rejected["view_id"], f"{rejected_label}.view_id")
            reasons = _string_list(
                rejected["reason_codes"], f"{rejected_label}.reason_codes"
            )
            if not reasons:
                raise MVInverseEvidenceError(
                    f"{rejected_label}.reason_codes cannot be empty"
                )
        suggestion = group["suggestion"]
        _exact(
            suggestion,
            {
                "decision",
                "auto_parameter_eligible",
                "base_color_srgb",
                "metallic",
                "roughness",
                "reason_codes",
                "warning_codes",
            },
            f"{label}.suggestion",
        )
        suggestion_reasons = _string_list(
            suggestion["reason_codes"], f"{label}.suggestion.reason_codes"
        )
        if not suggestion_reasons:
            raise MVInverseEvidenceError(
                f"{label}.suggestion.reason_codes cannot be empty"
            )
        _string_list(suggestion["warning_codes"], f"{label}.suggestion.warning_codes")
        eligible = suggestion["auto_parameter_eligible"]
        if not isinstance(eligible, bool) or suggestion["decision"] != (
            "auto" if eligible else "preserve"
        ):
            raise MVInverseEvidenceError(f"{label}.suggestion decision is inconsistent")
        if eligible:
            if not integrity_verified:
                raise MVInverseEvidenceError(
                    f"{label} cannot auto-apply without verified inference integrity"
                )
            auto_count += 1
            base_color = suggestion["base_color_srgb"]
            if not isinstance(base_color, list) or len(base_color) != 3:
                raise MVInverseEvidenceError(
                    f"{label}.suggestion.base_color_srgb must have three values"
                )
            for channel, item in enumerate(base_color):
                _probability(item, f"{label}.suggestion.base_color_srgb[{channel}]")
            _probability(suggestion["metallic"], f"{label}.suggestion.metallic")
            _probability(suggestion["roughness"], f"{label}.suggestion.roughness")
        elif any(
            suggestion[field] is not None
            for field in ("base_color_srgb", "metallic", "roughness")
        ):
            raise MVInverseEvidenceError(
                f"{label}.preserve suggestion must not contain parameters"
            )
        if (
            not integrity_verified
            and "unverified_inference_source" not in suggestion["reason_codes"]
        ):
            raise MVInverseEvidenceError(
                f"{label}.raw-directory suggestion must preserve as unverified"
            )
    if len(group_ids) != len(set(group_ids)):
        raise MVInverseEvidenceError("report contains duplicate group_id values")

    view_ids: list[str] = []
    for index, view in enumerate(views):
        label = f"report.views[{index}]"
        _exact(
            view, {"view_id", "frame_index", "resolution", "sources", "groups"}, label
        )
        view_id = view["view_id"]
        if not isinstance(view_id, str) or not _VIEW_ID_RE.fullmatch(view_id):
            raise MVInverseEvidenceError(f"{label}.view_id must be a safe identifier")
        view_ids.append(view_id)
        if (
            isinstance(view["frame_index"], bool)
            or not isinstance(view["frame_index"], int)
            or view["frame_index"] < 0
        ):
            raise MVInverseEvidenceError(f"{label}.frame_index must be non-negative")
        if (
            not isinstance(view["resolution"], list)
            or len(view["resolution"]) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 1
                for item in view["resolution"]
            )
        ):
            raise MVInverseEvidenceError(f"{label}.resolution is invalid")
        _exact(view["sources"], set(_CHANNELS), f"{label}.sources")
        for channel in _CHANNELS:
            _validate_source(view["sources"][channel], f"{label}.sources.{channel}")
        if (
            not isinstance(view["groups"], list)
            or [item.get("group_id") for item in view["groups"]] != group_ids
        ):
            raise MVInverseEvidenceError(
                f"{label}.groups must exactly cover canonical groups in order"
            )
        for group_index, item in enumerate(view["groups"]):
            item_label = f"{label}.groups[{group_index}]"
            _exact(
                item,
                {
                    "group_id",
                    "association",
                    "evidence_source",
                    "boxes",
                    "mask",
                    "region_pixels",
                    "image_pixels",
                    "image_fraction",
                    "color_filter_applied",
                    "color_matching_pixels",
                    "color_match_fraction",
                    "valid_pixels",
                    "valid_fraction",
                    "albedo",
                    "metallic",
                    "roughness",
                    "accepted",
                    "reason_codes",
                },
                item_label,
            )
            _validate_stats(item["albedo"], f"{item_label}.albedo", vector=True)
            _validate_stats(item["metallic"], f"{item_label}.metallic", vector=False)
            _validate_stats(item["roughness"], f"{item_label}.roughness", vector=False)
            association = item["association"]
            _exact(
                association,
                {"status", "candidate_group_ids", "matched_group_id"},
                f"{item_label}.association",
            )
            if association["status"] not in {
                "matched",
                "ambiguous",
                "unmatched",
                "explicit_mask",
            }:
                raise MVInverseEvidenceError(
                    f"{item_label}.association.status is invalid"
                )
            candidate_ids = _string_list(
                association["candidate_group_ids"],
                f"{item_label}.association.candidate_group_ids",
            )
            if any(not _GROUP_ID_RE.fullmatch(value) for value in candidate_ids):
                raise MVInverseEvidenceError(
                    f"{item_label}.association has an invalid candidate group ID"
                )
            matched_group_id = association["matched_group_id"]
            if matched_group_id is not None and (
                not isinstance(matched_group_id, str)
                or not _GROUP_ID_RE.fullmatch(matched_group_id)
            ):
                raise MVInverseEvidenceError(
                    f"{item_label}.association.matched_group_id is invalid"
                )
            if association["status"] == "matched" and (
                len(candidate_ids) != 1 or matched_group_id != candidate_ids[0]
            ):
                raise MVInverseEvidenceError(
                    f"{item_label}.matched association is inconsistent"
                )
            if association["status"] == "ambiguous" and (
                len(candidate_ids) < 2 or matched_group_id is not None
            ):
                raise MVInverseEvidenceError(
                    f"{item_label}.ambiguous association is inconsistent"
                )
            if association["status"] == "unmatched" and (
                candidate_ids or matched_group_id is not None
            ):
                raise MVInverseEvidenceError(
                    f"{item_label}.unmatched association is inconsistent"
                )
            if item["evidence_source"] not in {
                "palette_boxes",
                "explicit_mask",
                None,
            }:
                raise MVInverseEvidenceError(f"{item_label}.evidence_source is invalid")
            if not isinstance(item["boxes"], list):
                raise MVInverseEvidenceError(f"{item_label}.boxes must be an array")
            for box_index, box in enumerate(item["boxes"]):
                if (
                    not isinstance(box, list)
                    or len(box) != 4
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or not 0 <= value <= 1000
                        for value in box
                    )
                    or box[0] >= box[2]
                    or box[1] >= box[3]
                ):
                    raise MVInverseEvidenceError(
                        f"{item_label}.boxes[{box_index}] is invalid"
                    )
            if item["mask"] is not None:
                _exact(
                    item["mask"],
                    {"path", "sha256", "resized_to_output"},
                    f"{item_label}.mask",
                )
                _nonempty_string(item["mask"]["path"], f"{item_label}.mask.path")
                if not re.fullmatch(r"[0-9a-f]{64}", str(item["mask"]["sha256"])):
                    raise MVInverseEvidenceError(
                        f"{item_label}.mask.sha256 must be lowercase SHA-256"
                    )
                if not isinstance(item["mask"]["resized_to_output"], bool):
                    raise MVInverseEvidenceError(
                        f"{item_label}.mask.resized_to_output must be boolean"
                    )
            if item["evidence_source"] == "palette_boxes" and (
                association["status"] != "matched"
                or not item["boxes"]
                or item["mask"] is not None
            ):
                raise MVInverseEvidenceError(
                    f"{item_label}.palette-box evidence is inconsistent"
                )
            if item["evidence_source"] == "explicit_mask" and (
                association["status"] != "explicit_mask"
                or item["boxes"]
                or item["mask"] is None
            ):
                raise MVInverseEvidenceError(
                    f"{item_label}.explicit-mask evidence is inconsistent"
                )
            if item["evidence_source"] is None and (
                association["status"] not in {"ambiguous", "unmatched"}
                or item["boxes"]
                or item["mask"] is not None
            ):
                raise MVInverseEvidenceError(
                    f"{item_label}.missing evidence source is inconsistent"
                )
            for field in (
                "region_pixels",
                "image_pixels",
                "color_matching_pixels",
                "valid_pixels",
            ):
                _nonnegative_integer(
                    item[field],
                    f"{item_label}.{field}",
                    positive=field == "image_pixels",
                )
            if not (
                item["valid_pixels"]
                <= item["color_matching_pixels"]
                <= item["region_pixels"]
                <= item["image_pixels"]
            ):
                raise MVInverseEvidenceError(
                    f"{item_label} pixel counts are inconsistent"
                )
            if not isinstance(item["color_filter_applied"], bool):
                raise MVInverseEvidenceError(
                    f"{item_label}.color_filter_applied must be boolean"
                )
            _string_list(item["reason_codes"], f"{item_label}.reason_codes")
            if not isinstance(item["accepted"], bool) or item["accepted"] != (
                not item["reason_codes"]
            ):
                raise MVInverseEvidenceError(f"{item_label}.accepted is inconsistent")
            for stats_field in ("albedo", "metallic", "roughness"):
                stats = item[stats_field]
                if (stats is None) != (item["valid_pixels"] == 0):
                    raise MVInverseEvidenceError(
                        f"{item_label}.{stats_field} presence is inconsistent with valid pixels"
                    )
                if stats is not None and stats["sample_count"] != item["valid_pixels"]:
                    raise MVInverseEvidenceError(
                        f"{item_label}.{stats_field}.sample_count must count valid pixels"
                    )
            for field in ("image_fraction", "color_match_fraction", "valid_fraction"):
                _probability(item[field], f"{item_label}.{field}")
            expected_fractions = {
                "image_fraction": item["region_pixels"] / item["image_pixels"],
                "color_match_fraction": item["color_matching_pixels"]
                / max(1, item["region_pixels"]),
                "valid_fraction": item["valid_pixels"]
                / max(1, item["color_matching_pixels"]),
            }
            for field, expected in expected_fractions.items():
                if not math.isclose(
                    float(item[field]), expected, rel_tol=0.0, abs_tol=1e-7
                ):
                    raise MVInverseEvidenceError(
                        f"{item_label}.{field} is inconsistent with pixel counts"
                    )
    if len(view_ids) != len(set(view_ids)):
        raise MVInverseEvidenceError("report contains duplicate view_id values")
    known_views = set(view_ids)
    for index, group in enumerate(groups):
        categories = [
            set(group["contributing_view_ids"]),
            set(group["ambiguous_views"]),
            set(group["unmatched_views"]),
            {record.get("view_id") for record in group["rejected_views"]},
        ]
        used: set[Any] = set()
        for category in categories:
            if used & category:
                raise MVInverseEvidenceError(
                    f"report.groups[{index}] places a view in multiple outcomes"
                )
            used |= category
        if used != known_views:
            raise MVInverseEvidenceError(
                f"report.groups[{index}] must partition every source view"
            )

    summary = document["summary"]
    _exact(
        summary,
        {
            "view_count",
            "canonical_group_count",
            "auto_parameter_group_count",
            "preserve_group_count",
            "fail_closed",
            "usd_modified",
        },
        "report.summary",
    )
    _nonnegative_integer(
        summary["view_count"], "report.summary.view_count", positive=True
    )
    _nonnegative_integer(
        summary["canonical_group_count"],
        "report.summary.canonical_group_count",
        positive=True,
    )
    _nonnegative_integer(
        summary["auto_parameter_group_count"],
        "report.summary.auto_parameter_group_count",
    )
    _nonnegative_integer(
        summary["preserve_group_count"],
        "report.summary.preserve_group_count",
    )
    if summary != {
        "view_count": len(views),
        "canonical_group_count": len(groups),
        "auto_parameter_group_count": auto_count,
        "preserve_group_count": len(groups) - auto_count,
        "fail_closed": True,
        "usd_modified": False,
    }:
        raise MVInverseEvidenceError("report.summary is inconsistent")
    # JSON round-trip guarantees the returned value is detached and contains
    # no numpy scalars or other non-persistable objects.
    return json.loads(json.dumps(document, ensure_ascii=False, allow_nan=False))


def write_evidence_report(report: Mapping[str, Any], path: str | Path) -> Path:
    canonical = validate_mvinverse_evidence(report)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(canonical, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def _parse_frame_indices(values: Sequence[str]) -> dict[str, int] | None:
    if not values:
        return None
    result: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise MVInverseEvidenceError("--frame-index must use VIEW_ID=INDEX")
        view_id, raw_index = value.split("=", 1)
        if not view_id or view_id in result:
            raise MVInverseEvidenceError(f"invalid or duplicate frame mapping: {value}")
        try:
            result[view_id] = int(raw_index)
        except ValueError as exc:
            raise MVInverseEvidenceError(f"invalid frame index: {value}") from exc
    return result


def _parse_masks(values: Sequence[str]) -> dict[tuple[str, str], Path]:
    result: dict[tuple[str, str], Path] = {}
    for value in values:
        if "=" not in value or ":" not in value.split("=", 1)[0]:
            raise MVInverseEvidenceError("--mask must use VIEW_ID:GROUP_ID=PATH")
        identity, raw_path = value.split("=", 1)
        view_id, group_id = identity.split(":", 1)
        key = (view_id, group_id)
        if not view_id or not group_id or key in result:
            raise MVInverseEvidenceError(f"invalid or duplicate mask mapping: {value}")
        result[key] = Path(raw_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fuse MVInverse albedo/metallic/roughness into fail-closed group evidence."
    )
    parser.add_argument("--mvinverse-output-dir", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--canonical-palette", required=True)
    parser.add_argument(
        "--inference-ledger",
        help=(
            "Verified qwen-mvinverse-inference-ledger/v1. Without it the report "
            "is development-only and every parameter suggestion is preserve."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--frame-index",
        action="append",
        default=[],
        metavar="VIEW_ID=INDEX",
        help="Explicit official-output frame mapping; repeat for every source view.",
    )
    parser.add_argument(
        "--mask",
        action="append",
        default=[],
        metavar="VIEW_ID:GROUP_ID=PATH",
        help="Optional authoritative group mask; repeat as needed.",
    )
    args = parser.parse_args(argv)
    try:
        report = build_mvinverse_evidence_from_manifest(
            args.mvinverse_output_dir,
            args.reference_manifest,
            args.canonical_palette,
            masks=_parse_masks(args.mask),
            frame_indices=_parse_frame_indices(args.frame_index),
            inference_ledger=args.inference_ledger,
            path_base=Path(args.output).expanduser().resolve().parent,
        )
        destination = write_evidence_report(report, args.output)
    except (MVInverseEvidenceError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {"output": str(destination), **report["summary"]}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALBEDO_COLOR_SPACE",
    "EvidencePolicy",
    "MVInverseEvidenceError",
    "SCHEMA_VERSION",
    "build_mvinverse_evidence_from_manifest",
    "validate_mvinverse_evidence",
    "write_evidence_report",
]
