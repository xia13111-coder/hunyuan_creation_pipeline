#!/usr/bin/env python3
"""Fail-closed topology risk screening for uniform material assignments.

The module only reads the source USD and two JSON evidence documents.  It
does not open a USD stage or author any material bindings.  Its result is a
conservative *risk proxy*: a false result means that no hard topology signal
fired, not that a part is proven to use one material in the real object.

The distinction between hard and advisory signals is intentional.  Welded
topology describes geometric connectivity after coincident CAD tessellation
vertices have been reconciled, while raw components and normal-coherent
surface patches are commonly multiplied by STEP/BREP export seams.  Surface
patch count therefore remains auditable complexity evidence but cannot, on
its own, assert that a mesh contains multiple materials.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "qwen-geometry-uniform-material-risk/v1"
FACE_REGION_SCHEMA_VERSION = "qwen-face-region-evidence/v1"
RENDERED_REGISTRY_SCHEMA_VERSION = "qwen-material-parts/v1"

MULTIPLE_WELDED_COMPONENTS = "multiple_welded_topology_components"
HIGH_RAW_COMPONENT_COUNT = "high_raw_topology_component_count"
HIGH_SURFACE_PATCH_COUNT = "high_surface_patch_count"
REASON_CODE_ORDER = (
    MULTIPLE_WELDED_COMPONENTS,
    HIGH_RAW_COMPONENT_COUNT,
    HIGH_SURFACE_PATCH_COUNT,
)
HARD_RISK_REASON_CODES = frozenset(
    {
        MULTIPLE_WELDED_COMPONENTS,
        HIGH_RAW_COMPONENT_COUNT,
    }
)
ADVISORY_REASON_CODES = frozenset({HIGH_SURFACE_PATCH_COUNT})

_PART_ID_RE = re.compile(r"P[0-9]{4,8}")
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")


class GeometryRiskError(ValueError):
    """Raised when input evidence is malformed, stale, or inconsistent."""


@dataclass(frozen=True)
class GeometryRiskPolicy:
    """Conservative thresholds for topology signals.

    ``surface_patch_risk_threshold`` emits an advisory signal only.  A raw
    component count must be extreme before it becomes a hard signal because
    pre-weld component fragmentation is routine in CAD tessellation.
    """

    maximum_welded_topology_component_count: int = 1
    raw_topology_component_risk_threshold: int = 512
    surface_patch_risk_threshold: int = 128

    def validate(self) -> None:
        for name in (
            "maximum_welded_topology_component_count",
            "raw_topology_component_risk_threshold",
            "surface_patch_risk_threshold",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise GeometryRiskError(f"policy.{name} must be a positive integer")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GeometryRiskError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise GeometryRiskError(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GeometryRiskError(f"{label} must be a non-empty string")
    return value.strip()


def _part_id(value: Any, label: str) -> str:
    part_id = _string(value, label)
    if not _PART_ID_RE.fullmatch(part_id):
        raise GeometryRiskError(f"{label} must use P followed by 4..8 digits")
    return part_id


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GeometryRiskError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GeometryRiskError(f"{label} must be a non-negative integer")
    return value


def _sha256_value(value: Any, label: str) -> str:
    digest = _string(value, label).lower()
    if not _SHA256_RE.fullmatch(digest):
        raise GeometryRiskError(f"{label} must be a 64-character SHA-256 digest")
    return digest


def _load_document(
    value: Mapping[str, Any] | str | Path,
    *,
    label: str,
) -> tuple[Mapping[str, Any], Path | None]:
    if isinstance(value, Mapping):
        return value, None
    if not isinstance(value, (str, Path)):
        raise GeometryRiskError(f"{label} must be an object or JSON path")
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise GeometryRiskError(f"{label} is not a file: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GeometryRiskError(f"Unable to read {label}: {path}: {exc}") from exc
    return _object(document, label), path


def _resolve_asset_path(value: Any, *, base: Path | None, label: str) -> Path:
    raw = Path(_string(value, label)).expanduser()
    if not raw.is_absolute() and base is not None:
        raw = base.parent / raw
    try:
        path = raw.resolve(strict=True)
    except OSError as exc:
        raise GeometryRiskError(f"{label} does not resolve to a file: {raw}") from exc
    if not path.is_file():
        raise GeometryRiskError(f"{label} is not a file: {path}")
    return path


def _face_region_parts(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if document.get("schema_version") != FACE_REGION_SCHEMA_VERSION:
        raise GeometryRiskError(
            "face_region_manifest.schema_version must be "
            f"{FACE_REGION_SCHEMA_VERSION!r}"
        )
    if document.get("source_usd_unchanged") is not True:
        raise GeometryRiskError(
            "face_region_manifest.source_usd_unchanged must be true"
        )

    raw_parts = _array(document.get("parts"), "face_region_manifest.parts")
    if not raw_parts:
        raise GeometryRiskError("face_region_manifest.parts cannot be empty")
    parts: dict[str, dict[str, Any]] = {}
    prim_paths: set[str] = set()
    for index, raw_part in enumerate(raw_parts):
        part = _object(raw_part, f"face_region_manifest.parts[{index}]")
        part_id = _part_id(
            part.get("part_id"), f"face_region_manifest.parts[{index}].part_id"
        )
        if part_id in parts:
            raise GeometryRiskError(f"duplicate face-region part_id: {part_id}")
        prim_path = _string(
            part.get("prim_path"), f"face-region part {part_id}.prim_path"
        )
        if not prim_path.startswith("/"):
            raise GeometryRiskError(
                f"face-region part {part_id}.prim_path must be an absolute USD path"
            )
        if prim_path in prim_paths:
            raise GeometryRiskError(f"duplicate face-region prim_path: {prim_path}")
        prim_paths.add(prim_path)
        parts[part_id] = {
            "part_id": part_id,
            "prim_path": prim_path,
            "face_count": _positive_integer(
                part.get("face_count"), f"face-region part {part_id}.face_count"
            ),
            "raw_topology_component_count": _positive_integer(
                part.get("raw_topology_component_count"),
                f"face-region part {part_id}.raw_topology_component_count",
            ),
            "welded_topology_component_count": _positive_integer(
                part.get("welded_topology_component_count"),
                f"face-region part {part_id}.welded_topology_component_count",
            ),
            "surface_patch_count": _positive_integer(
                part.get("surface_patch_count"),
                f"face-region part {part_id}.surface_patch_count",
            ),
        }

    declared_part_count = _positive_integer(
        document.get("part_count"), "face_region_manifest.part_count"
    )
    if declared_part_count != len(parts):
        raise GeometryRiskError(
            "face_region_manifest.part_count does not match the parts array"
        )
    declared_face_count = _positive_integer(
        document.get("face_count"), "face_region_manifest.face_count"
    )
    if declared_face_count != sum(part["face_count"] for part in parts.values()):
        raise GeometryRiskError(
            "face_region_manifest.face_count does not match the parts array"
        )
    declared_welded = _positive_integer(
        document.get("welded_topology_component_count"),
        "face_region_manifest.welded_topology_component_count",
    )
    if declared_welded != sum(
        part["welded_topology_component_count"] for part in parts.values()
    ):
        raise GeometryRiskError(
            "face_region_manifest.welded_topology_component_count does not match parts"
        )
    declared_patches = _positive_integer(
        document.get("surface_patch_count"),
        "face_region_manifest.surface_patch_count",
    )
    if declared_patches != sum(part["surface_patch_count"] for part in parts.values()):
        raise GeometryRiskError(
            "face_region_manifest.surface_patch_count does not match parts"
        )
    return parts


def _registry_parts(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if document.get("schema_version") != RENDERED_REGISTRY_SCHEMA_VERSION:
        raise GeometryRiskError(
            "rendered_registry.schema_version must be "
            f"{RENDERED_REGISTRY_SCHEMA_VERSION!r}"
        )
    raw_parts = _array(document.get("parts"), "rendered_registry.parts")
    if not raw_parts:
        raise GeometryRiskError("rendered_registry.parts cannot be empty")
    parts: dict[str, dict[str, Any]] = {}
    prim_paths: set[str] = set()
    for index, raw_part in enumerate(raw_parts):
        part = _object(raw_part, f"rendered_registry.parts[{index}]")
        part_id = _part_id(
            part.get("part_id"), f"rendered_registry.parts[{index}].part_id"
        )
        if part_id in parts:
            raise GeometryRiskError(f"duplicate rendered-registry part_id: {part_id}")
        prim_path = _string(
            part.get("prim_path"), f"rendered-registry part {part_id}.prim_path"
        )
        if not prim_path.startswith("/"):
            raise GeometryRiskError(
                f"rendered-registry part {part_id}.prim_path must be an absolute USD path"
            )
        if prim_path in prim_paths:
            raise GeometryRiskError(
                f"duplicate rendered-registry prim_path: {prim_path}"
            )
        prim_paths.add(prim_path)
        parts[part_id] = {
            "part_id": part_id,
            "prim_path": prim_path,
            "face_count": _positive_integer(
                part.get("face_count"), f"rendered-registry part {part_id}.face_count"
            ),
        }
    declared_count = _positive_integer(
        document.get("part_count"), "rendered_registry.part_count"
    )
    if declared_count != len(parts):
        raise GeometryRiskError(
            "rendered_registry.part_count does not match the parts array"
        )
    return parts


def _validate_asset_contract(
    *,
    face_document: Mapping[str, Any],
    face_path: Path | None,
    registry_document: Mapping[str, Any],
    registry_path: Path | None,
) -> tuple[Path, str]:
    face_asset = _resolve_asset_path(
        face_document.get("asset_usd"),
        base=face_path,
        label="face_region_manifest.asset_usd",
    )
    registry_asset = _resolve_asset_path(
        registry_document.get("asset_usd"),
        base=registry_path,
        label="rendered_registry.asset_usd",
    )
    if face_asset != registry_asset:
        raise GeometryRiskError(
            "asset path mismatch between face-region manifest and rendered registry"
        )
    face_digest = _sha256_value(
        face_document.get("asset_sha256"), "face_region_manifest.asset_sha256"
    )
    registry_digest = _sha256_value(
        registry_document.get("asset_sha256"), "rendered_registry.asset_sha256"
    )
    if face_digest != registry_digest:
        raise GeometryRiskError(
            "asset hash mismatch between face-region manifest and rendered registry"
        )
    actual_digest = _sha256(face_asset)
    if actual_digest != face_digest:
        raise GeometryRiskError(
            "source USD SHA-256 does not match the evidence documents; evidence is stale"
        )

    before = _sha256_value(
        face_document.get("source_usd_sha256_before"),
        "face_region_manifest.source_usd_sha256_before",
    )
    after = _sha256_value(
        face_document.get("source_usd_sha256_after"),
        "face_region_manifest.source_usd_sha256_after",
    )
    if before != actual_digest or after != actual_digest:
        raise GeometryRiskError(
            "face-region source USD before/after hashes must equal asset_sha256"
        )

    render_set = registry_document.get("render_set")
    if render_set is not None:
        render_set = _object(render_set, "rendered_registry.render_set")
        render_asset = _resolve_asset_path(
            render_set.get("asset_usd"),
            base=registry_path,
            label="rendered_registry.render_set.asset_usd",
        )
        if render_asset != face_asset:
            raise GeometryRiskError(
                "rendered_registry.render_set.asset_usd does not match the registered asset"
            )
        has_readonly_audit = any(
            key in render_set
            for key in (
                "source_usd_sha256_before",
                "source_usd_sha256_after",
                "source_usd_unchanged",
            )
        )
        if has_readonly_audit:
            if render_set.get("source_usd_unchanged") is not True:
                raise GeometryRiskError(
                    "rendered_registry.render_set.source_usd_unchanged must be true"
                )
            render_before = _sha256_value(
                render_set.get("source_usd_sha256_before"),
                "rendered_registry.render_set.source_usd_sha256_before",
            )
            render_after = _sha256_value(
                render_set.get("source_usd_sha256_after"),
                "rendered_registry.render_set.source_usd_sha256_after",
            )
            if render_before != actual_digest or render_after != actual_digest:
                raise GeometryRiskError(
                    "rendered-registry source USD before/after hashes must equal asset_sha256"
                )
    return face_asset, actual_digest


def _reason_codes(metrics: Mapping[str, int], policy: GeometryRiskPolicy) -> list[str]:
    """Return every triggered topology signal in stable audit order."""

    reasons: list[str] = []
    if (
        metrics["welded_topology_component_count"]
        > policy.maximum_welded_topology_component_count
    ):
        reasons.append(MULTIPLE_WELDED_COMPONENTS)
    if (
        metrics["raw_topology_component_count"]
        >= policy.raw_topology_component_risk_threshold
    ):
        reasons.append(HIGH_RAW_COMPONENT_COUNT)
    if metrics["surface_patch_count"] >= policy.surface_patch_risk_threshold:
        reasons.append(HIGH_SURFACE_PATCH_COUNT)
    return reasons


def _has_hard_risk(reason_codes: Sequence[str]) -> bool:
    """Return whether any signal is strong enough to block uniform binding."""

    return any(reason in HARD_RISK_REASON_CODES for reason in reason_codes)


def build_geometry_risk(
    face_region_manifest: Mapping[str, Any] | str | Path,
    rendered_registry: Mapping[str, Any] | str | Path,
    *,
    policy: GeometryRiskPolicy | None = None,
) -> dict[str, Any]:
    """Build a deterministic uniform-material topology risk report.

    Both inputs can be mappings or JSON paths.  The function verifies their
    common source asset against the current file contents and requires exact
    part coverage before evaluating any risk threshold.
    """

    selected_policy = policy or GeometryRiskPolicy()
    if not isinstance(selected_policy, GeometryRiskPolicy):
        raise GeometryRiskError("policy must be a GeometryRiskPolicy")
    selected_policy.validate()
    face_document, face_path = _load_document(
        face_region_manifest, label="face_region_manifest"
    )
    registry_document, registry_path = _load_document(
        rendered_registry, label="rendered_registry"
    )
    face_parts = _face_region_parts(face_document)
    registry_parts = _registry_parts(registry_document)
    asset_path, asset_digest = _validate_asset_contract(
        face_document=face_document,
        face_path=face_path,
        registry_document=registry_document,
        registry_path=registry_path,
    )

    face_ids = set(face_parts)
    registry_ids = set(registry_parts)
    if face_ids != registry_ids:
        raise GeometryRiskError(
            "face-region manifest does not exactly cover rendered registry; "
            f"missing={sorted(registry_ids - face_ids)}, "
            f"unexpected={sorted(face_ids - registry_ids)}"
        )
    for part_id in sorted(face_ids):
        face_part = face_parts[part_id]
        registry_part = registry_parts[part_id]
        if face_part["prim_path"] != registry_part["prim_path"]:
            raise GeometryRiskError(f"prim_path mismatch for {part_id}")
        if face_part["face_count"] != registry_part["face_count"]:
            raise GeometryRiskError(f"face_count mismatch for {part_id}")

    part_records: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    risk_part_ids: list[str] = []
    for part_id in sorted(face_parts):
        source = face_parts[part_id]
        metrics = {
            "raw_topology_component_count": source["raw_topology_component_count"],
            "welded_topology_component_count": source[
                "welded_topology_component_count"
            ],
            "surface_patch_count": source["surface_patch_count"],
        }
        reasons = _reason_codes(metrics, selected_policy)
        at_risk = _has_hard_risk(reasons)
        reason_counts.update(reasons)
        if at_risk:
            risk_part_ids.append(part_id)
        part_records.append(
            {
                "part_id": part_id,
                "prim_path": source["prim_path"],
                "face_count": source["face_count"],
                "metrics": metrics,
                "risk": {
                    "multi_material_risk": at_risk,
                    "basis": "conservative_topology_complexity_proxy",
                },
                "reason_codes": reasons,
            }
        )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "asset_usd": str(asset_path),
        "asset_sha256": asset_digest,
        "source_usd_unchanged": True,
        "face_region_manifest": str(face_path) if face_path is not None else None,
        "face_region_manifest_sha256": _sha256(face_path) if face_path else None,
        "rendered_registry": str(registry_path) if registry_path is not None else None,
        "rendered_registry_sha256": _sha256(registry_path) if registry_path else None,
        "policy": asdict(selected_policy),
        "part_count": len(part_records),
        "parts": part_records,
        "summary": {
            "part_count": len(part_records),
            "face_count": sum(record["face_count"] for record in part_records),
            "multi_material_risk_part_count": len(risk_part_ids),
            "no_detected_multi_material_risk_part_count": (
                len(part_records) - len(risk_part_ids)
            ),
            "multi_material_risk_part_ids": risk_part_ids,
            "reason_code_counts": {
                reason: reason_counts.get(reason, 0) for reason in REASON_CODE_ORDER
            },
        },
        "limitations": [
            "Topology evidence is a conservative proxy and is not a material classifier.",
            "Surface-patch count is advisory only because normal-coherent geometric patches are not inferred material subsets.",
            "Raw pre-weld connectivity can reflect CAD export seams; only an extreme raw-component count is a hard topology signal.",
            "A part without a hard topology risk still requires visual/material evidence before automatic assignment.",
        ],
    }
    return validate_geometry_risk(report)


def validate_geometry_risk(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a JSON-compatible copy of a v1 risk report."""

    report = _object(document, "geometry_risk")
    if report.get("schema_version") != SCHEMA_VERSION:
        raise GeometryRiskError(
            f"geometry_risk.schema_version must be {SCHEMA_VERSION!r}"
        )
    asset_usd = _string(report.get("asset_usd"), "geometry_risk.asset_usd")
    asset_sha256 = _sha256_value(
        report.get("asset_sha256"), "geometry_risk.asset_sha256"
    )
    if report.get("source_usd_unchanged") is not True:
        raise GeometryRiskError("geometry_risk.source_usd_unchanged must be true")

    for name in ("face_region_manifest", "rendered_registry"):
        path_value = report.get(name)
        hash_value = report.get(f"{name}_sha256")
        if path_value is None:
            if hash_value is not None:
                raise GeometryRiskError(f"geometry_risk.{name}_sha256 requires a path")
        else:
            _string(path_value, f"geometry_risk.{name}")
            _sha256_value(hash_value, f"geometry_risk.{name}_sha256")

    raw_policy = _object(report.get("policy"), "geometry_risk.policy")
    expected_policy_fields = {
        "maximum_welded_topology_component_count",
        "raw_topology_component_risk_threshold",
        "surface_patch_risk_threshold",
    }
    if set(raw_policy) != expected_policy_fields:
        raise GeometryRiskError(
            "geometry_risk.policy must contain exactly the v1 threshold fields"
        )
    policy = GeometryRiskPolicy(
        maximum_welded_topology_component_count=_positive_integer(
            raw_policy.get("maximum_welded_topology_component_count"),
            "geometry_risk.policy.maximum_welded_topology_component_count",
        ),
        raw_topology_component_risk_threshold=_positive_integer(
            raw_policy.get("raw_topology_component_risk_threshold"),
            "geometry_risk.policy.raw_topology_component_risk_threshold",
        ),
        surface_patch_risk_threshold=_positive_integer(
            raw_policy.get("surface_patch_risk_threshold"),
            "geometry_risk.policy.surface_patch_risk_threshold",
        ),
    )
    policy.validate()

    raw_parts = _array(report.get("parts"), "geometry_risk.parts")
    if not raw_parts:
        raise GeometryRiskError("geometry_risk.parts cannot be empty")
    part_ids: list[str] = []
    prim_paths: set[str] = set()
    canonical_parts: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    risk_part_ids: list[str] = []
    for index, raw_part in enumerate(raw_parts):
        part = _object(raw_part, f"geometry_risk.parts[{index}]")
        part_id = _part_id(part.get("part_id"), f"geometry_risk.parts[{index}].part_id")
        if part_id in part_ids:
            raise GeometryRiskError(f"duplicate geometry-risk part_id: {part_id}")
        part_ids.append(part_id)
        prim_path = _string(
            part.get("prim_path"), f"geometry-risk part {part_id}.prim_path"
        )
        if not prim_path.startswith("/") or prim_path in prim_paths:
            raise GeometryRiskError(
                f"geometry-risk part {part_id}.prim_path must be unique and absolute"
            )
        prim_paths.add(prim_path)
        face_count = _positive_integer(
            part.get("face_count"), f"geometry-risk part {part_id}.face_count"
        )
        raw_metrics = _object(
            part.get("metrics"), f"geometry-risk part {part_id}.metrics"
        )
        expected_metrics = {
            "raw_topology_component_count",
            "welded_topology_component_count",
            "surface_patch_count",
        }
        if set(raw_metrics) != expected_metrics:
            raise GeometryRiskError(
                f"geometry-risk part {part_id}.metrics has unsupported fields"
            )
        metrics = {
            name: _positive_integer(
                raw_metrics.get(name), f"geometry-risk part {part_id}.metrics.{name}"
            )
            for name in expected_metrics
        }
        raw_risk = _object(part.get("risk"), f"geometry-risk part {part_id}.risk")
        if set(raw_risk) != {"multi_material_risk", "basis"}:
            raise GeometryRiskError(
                f"geometry-risk part {part_id}.risk has unsupported fields"
            )
        if raw_risk.get("basis") != "conservative_topology_complexity_proxy":
            raise GeometryRiskError(
                f"geometry-risk part {part_id}.risk.basis is invalid"
            )
        if not isinstance(raw_risk.get("multi_material_risk"), bool):
            raise GeometryRiskError(
                f"geometry-risk part {part_id}.risk.multi_material_risk must be boolean"
            )
        raw_reasons = _array(
            part.get("reason_codes"), f"geometry-risk part {part_id}.reason_codes"
        )
        reasons = [
            _string(value, f"geometry-risk part {part_id}.reason_codes[{reason_index}]")
            for reason_index, value in enumerate(raw_reasons)
        ]
        expected_reasons = _reason_codes(metrics, policy)
        if reasons != expected_reasons:
            raise GeometryRiskError(
                f"geometry-risk part {part_id}.reason_codes do not match metrics/policy"
            )
        expected_risk = _has_hard_risk(expected_reasons)
        if raw_risk["multi_material_risk"] is not expected_risk:
            raise GeometryRiskError(
                f"geometry-risk part {part_id}.multi_material_risk is inconsistent"
            )
        reason_counts.update(expected_reasons)
        if expected_risk:
            risk_part_ids.append(part_id)
        canonical_parts.append(
            {
                "part_id": part_id,
                "prim_path": prim_path,
                "face_count": face_count,
                "metrics": {
                    name: metrics[name]
                    for name in (
                        "raw_topology_component_count",
                        "welded_topology_component_count",
                        "surface_patch_count",
                    )
                },
                "risk": {
                    "multi_material_risk": expected_risk,
                    "basis": "conservative_topology_complexity_proxy",
                },
                "reason_codes": expected_reasons,
            }
        )
    if part_ids != sorted(part_ids):
        raise GeometryRiskError("geometry_risk.parts must be sorted by part_id")
    if _positive_integer(report.get("part_count"), "geometry_risk.part_count") != len(
        canonical_parts
    ):
        raise GeometryRiskError("geometry_risk.part_count does not match parts")

    expected_summary = {
        "part_count": len(canonical_parts),
        "face_count": sum(part["face_count"] for part in canonical_parts),
        "multi_material_risk_part_count": len(risk_part_ids),
        "no_detected_multi_material_risk_part_count": len(canonical_parts)
        - len(risk_part_ids),
        "multi_material_risk_part_ids": risk_part_ids,
        "reason_code_counts": {
            reason: reason_counts.get(reason, 0) for reason in REASON_CODE_ORDER
        },
    }
    if report.get("summary") != expected_summary:
        raise GeometryRiskError("geometry_risk.summary is inconsistent with parts")
    limitations = _array(report.get("limitations"), "geometry_risk.limitations")
    canonical_limitations = [
        _string(value, f"geometry_risk.limitations[{index}]")
        for index, value in enumerate(limitations)
    ]
    if not canonical_limitations:
        raise GeometryRiskError("geometry_risk.limitations cannot be empty")

    canonical = {
        "schema_version": SCHEMA_VERSION,
        "asset_usd": asset_usd,
        "asset_sha256": asset_sha256,
        "source_usd_unchanged": True,
        "face_region_manifest": report.get("face_region_manifest"),
        "face_region_manifest_sha256": report.get("face_region_manifest_sha256"),
        "rendered_registry": report.get("rendered_registry"),
        "rendered_registry_sha256": report.get("rendered_registry_sha256"),
        "policy": asdict(policy),
        "part_count": len(canonical_parts),
        "parts": canonical_parts,
        "summary": expected_summary,
        "limitations": canonical_limitations,
    }
    # A JSON round trip rejects non-serializable Mapping implementations and
    # guarantees that callers receive a detached, mutation-safe result.
    try:
        return json.loads(json.dumps(canonical, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise GeometryRiskError(
            f"geometry_risk is not valid finite JSON: {exc}"
        ) from exc


def write_geometry_risk(report: Mapping[str, Any], output: str | Path) -> Path:
    """Validate and atomically write a geometry-risk report."""

    canonical = validate_geometry_risk(report)
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / f".{output_path.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.write_text(
            json.dumps(canonical, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


# Descriptive aliases make the report's exact purpose explicit to callers
# while preserving compact build/validate/write names for pipeline integration.
build_geometry_uniform_material_risk = build_geometry_risk
validate_geometry_uniform_material_risk = validate_geometry_risk
write_geometry_uniform_material_risk = write_geometry_risk


__all__ = [
    "ADVISORY_REASON_CODES",
    "FACE_REGION_SCHEMA_VERSION",
    "GeometryRiskError",
    "GeometryRiskPolicy",
    "HARD_RISK_REASON_CODES",
    "RENDERED_REGISTRY_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_geometry_risk",
    "build_geometry_uniform_material_risk",
    "validate_geometry_risk",
    "validate_geometry_uniform_material_risk",
    "write_geometry_risk",
    "write_geometry_uniform_material_risk",
]
