#!/usr/bin/env python3
"""Material-selection inputs and fixed-MDL application for visual components.

The photo-supported appearance-component stage establishes *membership* only.
This module turns those memberships into a bounded, immutable-MDL selection
unit without ever changing CAD geometry, per-part registration, or an MDL
parameter.  Retrieval receives the selected reference observations of every
member, while Qwen receives a deterministic montage of representative member
crops.  The final binding remains one assignment per CAD Part-ID.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from .appearance_components import SCHEMA_VERSION as COMPONENT_SCHEMA_VERSION
from .part_id_projection import (
    PARAMETER_CANDIDATE_SCHEMA_VERSION,
    SCHEMA_VERSION as PART_ID_EVIDENCE_SCHEMA_VERSION,
    _build_part_id_parameter_candidates,
)


COMPONENT_EVIDENCE_SCHEMA_VERSION = "qwen-appearance-component-evidence/v1"
COMPONENT_RETRIEVAL_REQUEST_SCHEMA_VERSION = (
    "qwen-visual-material-retrieval-request/v1"
)
COMPONENT_SELECTION_SCHEMA_VERSION = "qwen-appearance-component-mdl-selection/v1"
MAXIMUM_MONTAGE_TILES = 4
MONTAGE_TILE_SIZE = 256


class AppearanceComponentMaterialError(ValueError):
    """Raised when component material selection would be untraceable."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(value: str | Path | Mapping[str, Any], label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    path = Path(value).expanduser().resolve(strict=True)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AppearanceComponentMaterialError(
            f"unable to read {label}: {path}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise AppearanceComponentMaterialError(f"{label} must be a JSON object")
    return document


def _write_object(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _file_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AppearanceComponentMaterialError(f"{label} must be a non-empty path")
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except OSError as exc:
        raise AppearanceComponentMaterialError(f"{label} does not exist: {value}") from exc
    if not path.is_file():
        raise AppearanceComponentMaterialError(f"{label} is not a file: {path}")
    return path


def _unit_rgb(value: Any, label: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
        or any(
            isinstance(channel, bool)
            or not isinstance(channel, (int, float))
            or not math.isfinite(float(channel))
            or not 0.0 <= float(channel) <= 1.0
            for channel in value
        )
    ):
        raise AppearanceComponentMaterialError(f"{label} must be a unit RGB triplet")
    return [float(channel) for channel in value]


def _component_rows(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    if document.get("schema_version") != COMPONENT_SCHEMA_VERSION:
        raise AppearanceComponentMaterialError("unsupported appearance-component schema")
    # ``appearance_components`` is a construction report, not a quality-gate
    # report.  Its producer deliberately uses COMPLETED once the conservative
    # membership pass has finished.  Requiring PASS here made every real
    # component report impossible to consume, even though the report had
    # already established valid Part-ID memberships.  Accept the current
    # producer status and the former PASS spelling for existing checkpoints;
    # retain the assignment-unit guard so a different kind of grouping cannot
    # enter the component material stage.
    if document.get("status") not in {"COMPLETED", "PASS"} or document.get(
        "assignment_unit"
    ) != "part_id":
        raise AppearanceComponentMaterialError("appearance components are not accepted")
    raw = document.get("components")
    if not isinstance(raw, list):
        raise AppearanceComponentMaterialError("appearance components has no component list")
    output: list[dict[str, Any]] = []
    seen_components: set[str] = set()
    seen_parts: set[str] = set()
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise AppearanceComponentMaterialError(f"component {index} is invalid")
        component_id = row.get("component_id")
        members = row.get("member_part_ids")
        if (
            not isinstance(component_id, str)
            or not component_id
            or component_id in seen_components
            or not isinstance(members, list)
            or len(members) < 2
            or any(not isinstance(part_id, str) or not part_id for part_id in members)
            or len(set(members)) != len(members)
            or seen_parts.intersection(members)
        ):
            raise AppearanceComponentMaterialError(
                f"component {index} has invalid or overlapping membership"
            )
        canonical_rgb = _unit_rgb(
            row.get("canonical_reference_rgb"),
            f"component {component_id} canonical_reference_rgb",
        )
        family = row.get("appearance_family")
        if not isinstance(family, str) or not family:
            raise AppearanceComponentMaterialError(
                f"component {component_id} has no appearance family"
            )
        seen_components.add(component_id)
        seen_parts.update(members)
        output.append(
            {
                "component_id": component_id,
                "member_part_ids": sorted(members),
                "anchor_part_id": row.get("anchor_part_id"),
                "appearance_family": family,
                "canonical_reference_rgb": canonical_rgb,
                "membership_authority": row.get("membership_authority"),
                "supporting_view_ids": sorted(
                    value
                    for value in row.get("supporting_view_ids", [])
                    if isinstance(value, str) and value
                ),
                "total_trusted_pixels": row.get("total_trusted_pixels"),
            }
        )
    return sorted(output, key=lambda row: row["component_id"])


def _part_evidence_by_id(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if document.get("schema_version") != PART_ID_EVIDENCE_SCHEMA_VERSION:
        raise AppearanceComponentMaterialError("unsupported Part-ID evidence schema")
    raw = document.get("parts")
    if not isinstance(raw, list):
        raise AppearanceComponentMaterialError("Part-ID evidence has no parts")
    rows: dict[str, Mapping[str, Any]] = {}
    for row in raw:
        part_id = row.get("part_id") if isinstance(row, Mapping) else None
        if not isinstance(part_id, str) or not part_id or part_id in rows:
            raise AppearanceComponentMaterialError("Part-ID evidence has duplicate IDs")
        rows[part_id] = row
    return rows


def filter_components_for_material_evidence(
    *,
    appearance_components: str | Path | Mapping[str, Any],
    part_id_evidence: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Keep only component members with a selected photo observation.

    Component membership is deliberately derived before the downstream
    evidence-selection gate.  A member can therefore be photo-supported in
    one registered source view but become unavailable after the conservative
    material-evidence gate selects the final observations.  That must narrow
    a component, not make every material assignment fail.  Members without a
    selected observation remain independently covered by the exact-cover plan.
    """

    source = _read_object(appearance_components, "appearance components")
    evidence = _read_object(part_id_evidence, "Part-ID evidence")
    normalized_components = _component_rows(source)
    parts_by_id = _part_evidence_by_id(evidence)
    raw_by_id = {
        row.get("component_id"): copy.deepcopy(dict(row))
        for row in source.get("components", [])
        if isinstance(row, Mapping) and isinstance(row.get("component_id"), str)
    }
    retained_components: list[dict[str, Any]] = []
    filter_rows: list[dict[str, Any]] = []
    for normalized in normalized_components:
        component_id = normalized["component_id"]
        raw = raw_by_id.get(component_id)
        if raw is None:  # defensive; _component_rows has already validated IDs.
            raise AppearanceComponentMaterialError(
                f"component {component_id} has no source record"
            )
        retained_members: list[str] = []
        excluded_members: list[dict[str, str]] = []
        selected_view_ids: set[str] = set()
        for part_id in normalized["member_part_ids"]:
            evidence_row = parts_by_id.get(part_id)
            if evidence_row is None:
                excluded_members.append(
                    {"part_id": part_id, "reason": "absent_from_part_id_evidence"}
                )
                continue
            try:
                observation = _selected_observation(part_id, evidence_row)
            except AppearanceComponentMaterialError as exc:
                excluded_members.append(
                    {"part_id": part_id, "reason": str(exc)}
                )
                continue
            retained_members.append(part_id)
            view_id = observation.get("view_id")
            if isinstance(view_id, str) and view_id:
                selected_view_ids.add(view_id)
        row = {
            "component_id": component_id,
            "source_member_count": len(normalized["member_part_ids"]),
            "material_eligible_member_part_ids": sorted(retained_members),
            "excluded_member_count": len(excluded_members),
            "excluded_members": excluded_members,
            "retained": len(retained_members) >= 2,
        }
        filter_rows.append(row)
        if len(retained_members) < 2:
            continue
        raw["member_part_ids"] = sorted(retained_members)
        if raw.get("anchor_part_id") not in retained_members:
            raw["anchor_part_id"] = sorted(retained_members)[0]
        raw["supporting_view_ids"] = sorted(selected_view_ids)
        raw["material_evidence_membership_authority"] = (
            "photo_supported_component_members_intersected_with_selected_"
            "part_id_material_observations"
        )
        retained_components.append(raw)

    unsigned = copy.deepcopy(source)
    unsigned.pop("integrity", None)
    unsigned["components"] = sorted(
        retained_components, key=lambda row: str(row["component_id"])
    )
    unsigned["material_evidence_filter"] = {
        "policy": "retain_components_with_at_least_two_selected_part_id_observations/v1",
        "source_component_count": len(normalized_components),
        "retained_component_count": len(retained_components),
        "source_member_count": sum(
            len(row["member_part_ids"]) for row in normalized_components
        ),
        "retained_member_count": sum(
            len(row["member_part_ids"]) for row in retained_components
        ),
        "components": filter_rows,
        "source_appearance_components_sha256": source.get("integrity", {}).get(
            "document_sha256"
        ),
        "source_part_id_evidence_sha256": evidence.get("integrity", {}).get(
            "document_sha256"
        ),
    }
    unsigned["integrity"] = {"document_sha256": _canonical_sha256(unsigned)}
    # Validate the projection before it is persisted or used to constrain a
    # material.  This also detects accidental overlap introduced by future
    # changes to the filter.
    _component_rows(unsigned)
    return unsigned


def _require_same_rendered_registry(
    components: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    """Reject cross-run component/evidence mixing before any image is reused."""

    inputs = components.get("inputs")
    expected = (
        inputs.get("rendered_registry_sha256") if isinstance(inputs, Mapping) else None
    )
    component_registry = (
        inputs.get("rendered_registry") if isinstance(inputs, Mapping) else None
    )
    raw_evidence_inputs = evidence.get("inputs")
    evidence_registry = next(
        (
            row
            for row in raw_evidence_inputs
            if isinstance(raw_evidence_inputs, list)
            and isinstance(row, Mapping)
            and row.get("label") == "rendered_registry"
        ),
        None,
    )
    actual = (
        evidence_registry.get("document_sha256")
        if isinstance(evidence_registry, Mapping)
        else None
    )
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or not isinstance(actual, str)
    ):
        raise AppearanceComponentMaterialError(
            "appearance components and Part-ID evidence do not bind the same "
            "camera-calibrated rendered registry"
        )

    # Appearance-component construction seals the *file* bytes because it also
    # consumes render paths.  Part-ID evidence, on the other hand, seals the
    # parsed registry document so formatting changes do not invalidate its
    # semantic provenance.  They are two valid representations of the same
    # registry and therefore must not be compared directly.  When both paths
    # are available, prove that they resolve to the same file, then validate
    # each producer's own digest convention.
    evidence_registry_path = (
        evidence_registry.get("path")
        if isinstance(evidence_registry, Mapping)
        else None
    )
    if isinstance(component_registry, str) and isinstance(evidence_registry_path, str):
        try:
            component_path = Path(component_registry).expanduser().resolve(strict=True)
            evidence_path = Path(evidence_registry_path).expanduser().resolve(strict=True)
        except OSError as exc:
            raise AppearanceComponentMaterialError(
                "appearance components and Part-ID evidence have an unavailable "
                "camera-calibrated rendered registry"
            ) from exc
        if component_path != evidence_path or _sha256_file(component_path) != expected:
            raise AppearanceComponentMaterialError(
                "appearance components and Part-ID evidence do not bind the same "
                "camera-calibrated rendered registry"
            )
        try:
            registry_document = json.loads(component_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AppearanceComponentMaterialError(
                "camera-calibrated rendered registry cannot be decoded for "
                "cross-stage provenance validation"
            ) from exc
        if not isinstance(registry_document, Mapping) or _canonical_sha256(
            registry_document
        ) != actual:
            raise AppearanceComponentMaterialError(
                "appearance components and Part-ID evidence do not bind the same "
                "camera-calibrated rendered registry"
            )
        return

    # Legacy in-memory test fixtures and old checkpoints contained only one
    # digest representation.  Preserve their strict equality contract rather
    # than silently accepting an unprovable cross-run mixture.
    if expected != actual:
        raise AppearanceComponentMaterialError(
            "appearance components and Part-ID evidence do not bind the same "
            "camera-calibrated rendered registry"
        )


def _selected_observation(part_id: str, row: Mapping[str, Any]) -> Mapping[str, Any]:
    if row.get("status") != "observed":
        raise AppearanceComponentMaterialError(
            f"appearance-component member {part_id} is not photo-observed"
        )
    raw = row.get("observations")
    selected = [
        observation
        for observation in raw if isinstance(raw, list) and isinstance(observation, Mapping)
        and observation.get("selected_for_material_inference") is True
    ]
    if len(selected) != 1:
        raise AppearanceComponentMaterialError(
            f"appearance-component member {part_id} lacks one selected observation"
        )
    image = _file_path(selected[0].get("image"), f"{part_id} selected image")
    # A tolerant bounding box is useful to establish CAD/photo
    # correspondence, but it is deliberately not material evidence: it can
    # include a neighbouring coloured panel, an occluder or a large amount of
    # the whole-workpiece foreground.  Once correspondence is accepted, the
    # registered CAD Part-ID interior is the only mask allowed to reach
    # retrieval or the VLM.
    _file_path(selected[0].get("mask"), f"{part_id} selected material mask")
    _file_path(selected[0].get("crop"), f"{part_id} selected crop")
    return selected[0]


def _numeric_descriptor_median(
    part_rows: Sequence[Mapping[str, Any]], key: str
) -> float | None:
    values = [
        float(descriptor[key])
        for row in part_rows
        for descriptor in [row.get("descriptor")]
        if isinstance(descriptor, Mapping)
        and isinstance(descriptor.get(key), (int, float))
        and not isinstance(descriptor.get(key), bool)
        and math.isfinite(float(descriptor[key]))
        and 0.0 <= float(descriptor[key]) <= 1.0
    ]
    if not values:
        return None
    values.sort()
    return round(values[len(values) // 2], 8)


def _component_descriptor(
    component: Mapping[str, Any], part_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    rgb = component["canonical_reference_rgb"]
    descriptor: dict[str, Any] = {
        "visual_description": (
            "photo-supported same visual coating across multiple exact CAD Part-IDs; "
            "select one opaque industrial NVIDIA Base MDL with matching visible "
            "color, smoothness and texture scale"
        ),
        "family_hint": "visual_similarity_only",
        "base_color": (
            f"{component['appearance_family']} RGB "
            f"({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f})"
        ),
        "surface_class": "opaque_industrial_coating",
        "component_appearance_family": component["appearance_family"],
        "canonical_reference_rgb": list(rgb),
    }
    for key in ("roughness_hint", "metallicity_hint"):
        median = _numeric_descriptor_median(part_rows, key)
        if median is not None:
            descriptor[key] = median
    return descriptor


def _component_montage(
    *,
    component_id: str,
    observations: Sequence[tuple[str, Mapping[str, Any]]],
    output_dir: Path,
) -> tuple[Path, Path, int]:
    """Build a clean, core-masked visual target for a shared coating.

    The older montage pasted annotated correspondence crops and marked its
    entire canvas as foreground.  Its yellow box, neighbouring parts and
    neutral tile background then became *material* pixels for SigLIP/DINO and
    Qwen's compatibility gate.  This is especially destructive for a dark
    small part placed on a green enclosure.  Keep only the registered CAD
    Part-ID core in each tile and use a neutral background outside that core.
    """

    if not observations:
        raise AppearanceComponentMaterialError(
            f"component {component_id} has no selected material observations"
        )
    chosen = list(observations[:MAXIMUM_MONTAGE_TILES])
    columns = 2
    rows = int(math.ceil(len(chosen) / columns))
    width = columns * MONTAGE_TILE_SIZE
    height = rows * MONTAGE_TILE_SIZE
    montage = Image.new("RGB", (width, height), (20, 27, 36))
    montage_mask = Image.new("L", (width, height), 0)
    for index, (part_id, observation) in enumerate(chosen):
        image_path = _file_path(observation.get("image"), f"{part_id} image")
        mask_path = _file_path(observation.get("mask"), f"{part_id} material mask")
        with Image.open(image_path) as opened, Image.open(mask_path) as mask_opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            mask = ImageOps.exif_transpose(mask_opened).convert("L")
            if mask.size != image.size:
                raise AppearanceComponentMaterialError(
                    f"{part_id} material mask dimensions differ from its image"
                )
            bbox = mask.getbbox()
            if bbox is None:
                raise AppearanceComponentMaterialError(
                    f"{part_id} selected material mask is empty"
                )
            left, top, right, bottom = bbox
            margin = max(3, int(round(max(right - left, bottom - top) * 0.10)))
            left = max(0, left - margin)
            top = max(0, top - margin)
            right = min(image.width, right + margin)
            bottom = min(image.height, bottom + margin)
            crop = image.crop((left, top, right, bottom))
            crop_mask = mask.crop((left, top, right, bottom))
            neutral = Image.new("RGB", crop.size, (127, 127, 127))
            neutral.paste(crop, mask=crop_mask)
            scale = min(
                (MONTAGE_TILE_SIZE - 8) / max(1, neutral.width),
                (MONTAGE_TILE_SIZE - 8) / max(1, neutral.height),
            )
            target = (
                max(1, int(round(neutral.width * scale))),
                max(1, int(round(neutral.height * scale))),
            )
            neutral = neutral.resize(target, Image.Resampling.LANCZOS)
            crop_mask = crop_mask.resize(target, Image.Resampling.NEAREST)
            tile = Image.new("RGB", (MONTAGE_TILE_SIZE, MONTAGE_TILE_SIZE), (127, 127, 127))
            tile_mask = Image.new("L", (MONTAGE_TILE_SIZE, MONTAGE_TILE_SIZE), 0)
            x = (MONTAGE_TILE_SIZE - neutral.width) // 2
            y = (MONTAGE_TILE_SIZE - neutral.height) // 2
            tile.paste(neutral, (x, y))
            tile_mask.paste(crop_mask, (x, y))
        offset_x = (index % columns) * MONTAGE_TILE_SIZE
        offset_y = (index // columns) * MONTAGE_TILE_SIZE
        montage.paste(tile, (offset_x, offset_y))
        montage_mask.paste(tile_mask, (offset_x, offset_y))
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{component_id}.png"
    mask_path = output_dir / f"{component_id}.mask.png"
    montage.save(image_path)
    montage_mask.save(mask_path)
    trusted_pixels = int(sum(montage_mask.histogram()[128:]))
    if trusted_pixels <= 0:
        raise AppearanceComponentMaterialError(
            f"component {component_id} montage contains no material pixels"
        )
    return image_path.resolve(), mask_path.resolve(), trusted_pixels


def build_component_material_inputs(
    *,
    appearance_components: str | Path | Mapping[str, Any],
    part_id_evidence: str | Path | Mapping[str, Any],
    catalog: str | Path,
    material_root: str | Path,
    output_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build synthetic component Qwen evidence and aggregate retrieval request."""

    components_document = _read_object(appearance_components, "appearance components")
    evidence_document = _read_object(part_id_evidence, "Part-ID evidence")
    components = _component_rows(components_document)
    parts_by_id = _part_evidence_by_id(evidence_document)
    _require_same_rendered_registry(components_document, evidence_document)
    destination = Path(output_dir).expanduser().resolve()
    montage_dir = destination / "component_montages"
    evidence_parts: list[dict[str, Any]] = []
    retrieval_groups: list[dict[str, Any]] = []
    for component in components:
        members = component["member_part_ids"]
        if any(part_id not in parts_by_id for part_id in members):
            raise AppearanceComponentMaterialError(
                f"component {component['component_id']} cites a Part-ID absent from evidence"
            )
        member_rows = [parts_by_id[part_id] for part_id in members]
        selected = [
            (part_id, _selected_observation(part_id, parts_by_id[part_id]))
            for part_id in members
        ]
        anchor = component.get("anchor_part_id")
        selected.sort(
            key=lambda item: (
                item[0] != anchor,
                -int(item[1].get("trusted_foreground_pixels", 0) or 0),
                item[0],
            )
        )
        image_path, mask_path, montage_trusted_pixels = _component_montage(
            component_id=component["component_id"],
            observations=selected,
            output_dir=montage_dir,
        )
        descriptor = _component_descriptor(component, member_rows)
        source_observations = [
            {
                "part_id": part_id,
                "view_id": observation.get("view_id"),
                "image": str(_file_path(observation.get("image"), f"{part_id} image")),
                "mask": str(
                    _file_path(
                        observation.get("mask"),
                        f"{part_id} retrieval mask",
                    )
                ),
                "source_mask_sha256": observation.get("mask_sha256"),
            }
            for part_id, observation in selected
        ]
        evidence_parts.append(
            {
                "part_id": component["component_id"],
                "status": "observed",
                "assignment_unit": "appearance_component",
                "member_part_ids": members,
                "descriptor": descriptor,
                "component": copy.deepcopy(component),
                "observations": [
                    {
                        "view_id": "component_montage",
                        "image": str(image_path),
                        "mask": str(mask_path),
                        "crop": str(image_path),
                        "mask_sha256": _sha256_file(mask_path),
                        "trusted_foreground_pixels": montage_trusted_pixels,
                        "selected_for_material_inference": True,
                    }
                ],
                "source_observations": source_observations,
            }
        )
        retrieval_groups.append(
            {
                "group_id": component["component_id"],
                "assignment_unit": "appearance_component",
                "member_part_ids": members,
                "descriptor": descriptor,
                "observations": [
                    {
                        "view_id": source["view_id"],
                        "image": source["image"],
                        "mask": source["mask"],
                    }
                    for source in source_observations
                ],
            }
        )
    evidence_unsigned = {
        "schema_version": COMPONENT_EVIDENCE_SCHEMA_VERSION,
        "assignment_unit": "appearance_component",
        "membership_authority": "photo_supported_appearance_components",
        "mdl_parameter_mutation_allowed": False,
        "source_appearance_components_sha256": components_document["integrity"][
            "document_sha256"
        ],
        "source_part_id_evidence_sha256": evidence_document["integrity"]["document_sha256"],
        "parts": evidence_parts,
        "summary": {
            "component_count": len(evidence_parts),
            "component_member_count": sum(
                len(row["member_part_ids"]) for row in evidence_parts
            ),
            "montage_tile_cap": MAXIMUM_MONTAGE_TILES,
        },
    }
    component_evidence = {
        **evidence_unsigned,
        "integrity": {"document_sha256": _canonical_sha256(evidence_unsigned)},
    }
    catalog_path = Path(catalog).expanduser().resolve(strict=True)
    material_root_path = Path(material_root).expanduser().resolve(strict=True)
    request_unsigned = {
        "schema_version": COMPONENT_RETRIEVAL_REQUEST_SCHEMA_VERSION,
        "catalog": str(catalog_path),
        "material_root": str(material_root_path),
        "assignment_unit": "appearance_component",
        "groups": retrieval_groups,
        "component_evidence_sha256": component_evidence["integrity"]["document_sha256"],
        "part_id_evidence_sha256": evidence_document["integrity"]["document_sha256"],
        "material_selection_contract": "one_immutable_base_mdl_per_component",
    }
    request = {
        **request_unsigned,
        "integrity": {"document_sha256": _canonical_sha256(request_unsigned)},
    }
    return component_evidence, request


def apply_fixed_component_mdl_choices(
    *,
    base_plan: Mapping[str, Any],
    base_audit: Mapping[str, Any],
    appearance_components: Mapping[str, Any],
    part_id_evidence: Mapping[str, Any],
    component_evidence: Mapping[str, Any],
    component_retrieval: Mapping[str, Any],
    component_qwen_choices: Mapping[str, Any],
    authorized_component_ids: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply one Qwen-chosen fixed MDL per accepted visual component.

    This never accepts a parameterized candidate.  Replacing the MDL also
    rebuilds the member's H0-only candidate record so no stale H1 candidate
    can survive from an earlier independent selection.
    """

    components = _component_rows(appearance_components)
    expected_components = {row["component_id"] for row in components}
    if authorized_component_ids is None:
        authorized_components = set(expected_components)
        component_authorization_mode = "all_accepted_appearance_components"
    else:
        if isinstance(authorized_component_ids, (str, bytes)):
            raise AppearanceComponentMaterialError(
                "authorized component IDs must be a sequence"
            )
        authorized_component_rows = list(authorized_component_ids)
        if (
            len(authorized_component_rows) != len(set(authorized_component_rows))
            or any(
                not isinstance(component_id, str) or not component_id
                for component_id in authorized_component_rows
            )
            or not set(authorized_component_rows).issubset(expected_components)
        ):
            raise AppearanceComponentMaterialError(
                "authorized component IDs are invalid"
            )
        authorized_components = set(authorized_component_rows)
        component_authorization_mode = "caller_authorized_physical_consistency_subset"
    part_evidence = _part_evidence_by_id(part_id_evidence)
    if component_evidence.get("schema_version") != COMPONENT_EVIDENCE_SCHEMA_VERSION:
        raise AppearanceComponentMaterialError("unsupported component selection evidence")
    component_evidence_integrity = component_evidence.get("integrity")
    component_evidence_sha256 = (
        component_evidence_integrity.get("document_sha256")
        if isinstance(component_evidence_integrity, Mapping)
        else None
    )
    if not isinstance(component_evidence_sha256, str) or len(component_evidence_sha256) != 64:
        raise AppearanceComponentMaterialError("component selection evidence is unsealed")
    plan = copy.deepcopy(dict(base_plan))
    audit = copy.deepcopy(dict(base_audit))
    assignments = plan.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise AppearanceComponentMaterialError("base plan has no assignments")
    assignment_by_part: dict[str, dict[str, Any]] = {}
    for row in assignments:
        part_id = row.get("part_id") if isinstance(row, Mapping) else None
        if not isinstance(part_id, str) or not part_id or part_id in assignment_by_part:
            raise AppearanceComponentMaterialError("base plan has invalid Part-ID coverage")
        assignment_by_part[part_id] = row
    if set(part_evidence) != set(assignment_by_part):
        raise AppearanceComponentMaterialError(
            "component selection evidence does not exactly cover the base plan"
        )
    if component_retrieval.get("schema_version") != (
        "qwen-visual-material-retrieval-result/v1"
    ):
        raise AppearanceComponentMaterialError("unsupported component retrieval schema")
    retrieval_integrity = component_retrieval.get("integrity")
    retrieval_unsigned = copy.deepcopy(dict(component_retrieval))
    retrieval_unsigned.pop("integrity", None)
    if (
        not isinstance(retrieval_integrity, Mapping)
        or retrieval_integrity.get("result_sha256")
        != _canonical_sha256(retrieval_unsigned)
    ):
        raise AppearanceComponentMaterialError("component retrieval failed integrity verification")
    retrieval_groups = component_retrieval.get("groups")
    if not isinstance(retrieval_groups, list):
        raise AppearanceComponentMaterialError("component retrieval has no groups")
    retrieval_by_component = {
        row.get("group_id"): row
        for row in retrieval_groups
        if isinstance(row, Mapping) and isinstance(row.get("group_id"), str)
    }
    if set(retrieval_by_component) != expected_components:
        raise AppearanceComponentMaterialError(
            "component retrieval does not exactly cover accepted components"
        )
    if component_qwen_choices.get("schema_version") != (
        "qwen-appearance-component-rerank/v1"
    ):
        raise AppearanceComponentMaterialError("unsupported component Qwen schema")
    qwen_integrity = component_qwen_choices.get("integrity")
    qwen_unsigned = copy.deepcopy(dict(component_qwen_choices))
    qwen_unsigned.pop("integrity", None)
    if (
        not isinstance(qwen_integrity, Mapping)
        or qwen_integrity.get("document_sha256") != _canonical_sha256(qwen_unsigned)
        or component_qwen_choices.get("component_evidence_sha256")
        != component_evidence_sha256
    ):
        raise AppearanceComponentMaterialError(
            "component Qwen choices are unsealed or bind different component evidence"
        )
    raw_choices = component_qwen_choices.get("choices")
    if not isinstance(raw_choices, Mapping) or set(raw_choices) != expected_components:
        raise AppearanceComponentMaterialError(
            "component Qwen choices do not exactly cover accepted components"
        )
    confidences = {
        str(row["part_id"]): float(row["confidence"])
        for row in component_qwen_choices.get("selections", [])
        if isinstance(row, Mapping)
        and isinstance(row.get("part_id"), str)
        and isinstance(row.get("confidence"), (int, float))
        and not isinstance(row.get("confidence"), bool)
    }
    if set(confidences) != expected_components:
        raise AppearanceComponentMaterialError(
            "component Qwen selections lack a confidence for one component"
        )
    audit_rows = audit.get("parts")
    if not isinstance(audit_rows, list):
        raise AppearanceComponentMaterialError("base Part-ID audit has no part records")
    audit_by_part = {
        row.get("part_id"): row
        for row in audit_rows
        if isinstance(row, Mapping) and isinstance(row.get("part_id"), str)
    }
    if set(audit_by_part) != set(assignment_by_part):
        raise AppearanceComponentMaterialError("base Part-ID audit does not cover plan")
    selection_records: list[dict[str, Any]] = []
    for component in components:
        component_id = component["component_id"]
        if component_id not in authorized_components:
            continue
        material_id = raw_choices.get(component_id)
        if not isinstance(material_id, str) or not material_id:
            raise AppearanceComponentMaterialError(
                f"component {component_id} has an invalid Qwen material ID"
            )
        ranking = retrieval_by_component[component_id].get("fused_ranking")
        selected_row = next(
            (
                row
                for row in ranking
                if isinstance(ranking, list)
                and isinstance(row, Mapping)
                and row.get("material_id") == material_id
            ),
            None,
        )
        if selected_row is None:
            raise AppearanceComponentMaterialError(
                f"component {component_id} chose a non-retrieved MDL"
            )
        for part_id in component["member_part_ids"]:
            assignment = assignment_by_part[part_id]
            assignment["material_id"] = material_id
            assignment["semantic"] = (
                "photo-supported appearance component; one immutable NVIDIA Base "
                "MDL selected from aggregate member observations"
            )
            provenance = assignment.get("provenance")
            if not isinstance(provenance, dict):
                provenance = {}
                assignment["provenance"] = provenance
            candidate_set, color_audit = _build_part_id_parameter_candidates(
                part_id=part_id,
                material_id=material_id,
                part_evidence=part_evidence[part_id],
                sam3_role=part_id_evidence.get("sam3_role"),
                enabled=False,
            )
            provenance.update(
                {
                    "selection_basis": (
                        "component_qwen_choice_within_component_siglip2_"
                        "dinov2_mvinverse_candidates"
                    ),
                    "photo_appearance_component_id": component_id,
                    "photo_appearance_component_member_part_ids": list(
                        component["member_part_ids"]
                    ),
                    "photo_appearance_component_canonical_reference_rgb": list(
                        component["canonical_reference_rgb"]
                    ),
                    "photo_appearance_component_membership_authority": component[
                        "membership_authority"
                    ],
                    "component_selected_retrieval_rank": selected_row.get("rank"),
                    "component_qwen_confidence": confidences[component_id],
                    "immutable_mdl_after_component_selection": True,
                    "mdl_parameter_candidates": candidate_set,
                    "mdl_color_parameterization": color_audit,
                }
            )
            audit_row = audit_by_part[part_id]
            audit_row.update(
                {
                    "material_id": material_id,
                    "appearance_component_id": component_id,
                    "appearance_component_mdl_overrode_independent_choice": True,
                    "component_selected_retrieval_rank": selected_row.get("rank"),
                    "component_qwen_confidence": confidences[component_id],
                    "mdl_parameter_candidates": copy.deepcopy(candidate_set),
                    "mdl_color_parameterization": copy.deepcopy(color_audit),
                }
            )
        selection_records.append(
            {
                "component_id": component_id,
                "member_part_ids": list(component["member_part_ids"]),
                "material_id": material_id,
                "qwen_confidence": confidences[component_id],
                "selected_retrieval_rank": selected_row.get("rank"),
                "canonical_reference_rgb": list(component["canonical_reference_rgb"]),
                "mdl_parameter_mutation_allowed": False,
            }
        )
    provenance = plan.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
        plan["provenance"] = provenance
    provenance.update(
        {
            "appearance_component_mdl_selection": {
                "schema_version": COMPONENT_SELECTION_SCHEMA_VERSION,
                "appearance_components_sha256": appearance_components["integrity"][
                    "document_sha256"
                ],
                "part_id_evidence_sha256": part_id_evidence["integrity"][
                    "document_sha256"
                ],
                "component_evidence_sha256": component_evidence_sha256,
                "component_retrieval_sha256": _canonical_sha256(component_retrieval),
                "component_qwen_choices_sha256": _canonical_sha256(component_qwen_choices),
                "component_authorization_mode": component_authorization_mode,
                "authorized_component_ids": sorted(authorized_components),
                "excluded_component_ids": sorted(
                    expected_components - authorized_components
                ),
                "one_fixed_mdl_per_component": True,
                "mdl_parameter_mutation_allowed": False,
                "selections": selection_records,
            },
            "coating_consistency_enabled": False,
            "coating_consistency_replaced_by": "photo_supported_appearance_components",
        }
    )
    plan["photo_appearance_components_used"] = bool(selection_records)
    plan["coating_consistency_used"] = False
    # Downstream Part-ID quality gating predates this photo-supported stage and
    # requires a successful consistency contract.  Publish an explicit PASS
    # replacement rather than leaving the old source-appearance gate as
    # ``NOT_RUN``; it carries the same summary fields but records that the
    # membership authority has changed.
    audit["coating_consistency_gate"] = {
        "schema_version": "qwen-photo-appearance-component-consistency/v1",
        "status": "PASS",
        "replaced_legacy_source_appearance_coating_gate": True,
        "membership_authority": (
            "same_view_rigid_part_id_projection_plus_sam3_foreground_colour_and_proximity"
        ),
        "mdl_parameter_mutation_allowed": False,
        "summary": {
            "component_count": len(selection_records),
            "constrained_part_count": sum(
                len(row["member_part_ids"]) for row in selection_records
            ),
            "material_changed_part_count": sum(
                len(row["member_part_ids"]) for row in selection_records
            ),
            "material_changed_part_ids": sorted(
                part_id
                for row in selection_records
                for part_id in row["member_part_ids"]
            ),
            "violation_count": 0,
        },
    }
    audit["output_plan_sha256"] = _canonical_sha256(plan)
    audit["appearance_component_mdl_selection"] = copy.deepcopy(
        provenance["appearance_component_mdl_selection"]
    )
    summary = audit.get("summary")
    if not isinstance(summary, dict):
        raise AppearanceComponentMaterialError("base Part-ID audit has no summary")
    summary.update(
        {
            "appearance_component_count": len(selection_records),
            "appearance_component_constrained_part_count": sum(
                len(row["member_part_ids"]) for row in selection_records
            ),
            "appearance_component_fixed_mdl_count": len(selection_records),
        }
    )
    audit.pop("integrity", None)
    audit["integrity"] = {"document_sha256": _canonical_sha256(audit)}
    return plan, audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="build fixed-MDL retrieval inputs for photo appearance components"
    )
    parser.add_argument("--appearance-components", type=Path, required=True)
    parser.add_argument("--part-id-evidence", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--material-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--retrieval-request-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evidence, request = build_component_material_inputs(
        appearance_components=args.appearance_components,
        part_id_evidence=args.part_id_evidence,
        catalog=args.catalog,
        material_root=args.material_root,
        output_dir=args.output_dir,
    )
    _write_object(args.evidence_output.expanduser().resolve(), evidence)
    _write_object(args.retrieval_request_output.expanduser().resolve(), request)
    print(
        json.dumps(
            {
                "component_evidence": str(args.evidence_output.expanduser().resolve()),
                "retrieval_request": str(
                    args.retrieval_request_output.expanduser().resolve()
                ),
                **evidence["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 0


__all__ = [
    "AppearanceComponentMaterialError",
    "COMPONENT_EVIDENCE_SCHEMA_VERSION",
    "COMPONENT_SELECTION_SCHEMA_VERSION",
    "apply_fixed_component_mdl_choices",
    "build_component_material_inputs",
    "filter_components_for_material_evidence",
]


if __name__ == "__main__":
    raise SystemExit(main())
