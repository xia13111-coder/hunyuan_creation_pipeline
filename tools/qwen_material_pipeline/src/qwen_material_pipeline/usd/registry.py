#!/usr/bin/env python3
"""Build a stable part-id registry for Mesh prims in a USD asset.

The module is intentionally separate from the Qwen client.  It must run in an
Isaac Sim Python process because the normal project Python does not ship pxr.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "qwen-material-parts/v1"
SOURCE_MATERIAL_BIND_SUBSETS_FIELD = "existing_material_bind_face_subsets"
SOURCE_SUBSET_HASH_FIELD = "source_subset_binding_sha256"
GEOMETRY_CONTENT_HASH_FIELD = "geometry_content_sha256"
SOURCE_APPEARANCE_HASH_FIELD = "source_appearance_sha256"
SOURCE_SUBSET_LAYOUT_HASH_FIELD = "source_subset_layout_sha256"
_SOURCE_SUBSET_FIELDS = frozenset(
    {
        "subset_name",
        "subset_prim_path",
        "family_name",
        "family_type",
        "element_type",
        "face_indices",
        "visual_material_prim_path",
        "binding_relationship_name",
        "binding_targets",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def geometry_content_sha256(
    *,
    points: Sequence[Any],
    face_vertex_counts: Sequence[Any],
    face_vertex_indices: Sequence[Any],
    orientation: Any = None,
    subdivision_scheme: Any = None,
) -> str:
    """Return a path-free, translation-invariant Mesh content digest.

    Registry point/face counts and world extents are useful corruption checks,
    but they are not a safe repeated-part identity: unrelated fasteners often
    share all three.  This digest retains local vertex order and exact face
    topology while removing only a uniform translation.  Rotation and scale
    deliberately remain significant so the hash cannot silently group merely
    similar geometry.
    """

    normalized_points: list[list[float]] = []
    converted_points: list[tuple[float, float, float]] = []
    for index, raw_point in enumerate(points):
        try:
            coordinates = list(raw_point)
        except TypeError as exc:
            raise RuntimeError(f"Mesh point[{index}] is not a 3-vector") from exc
        if len(coordinates) != 3:
            raise RuntimeError(f"Mesh point[{index}] is not a 3-vector")
        converted = tuple(_finite_scalar(value) for value in coordinates)
        if any(value is None for value in converted):
            raise RuntimeError(f"Mesh point[{index}] is not finite")
        converted_points.append(
            (float(converted[0]), float(converted[1]), float(converted[2]))
        )

    if converted_points:
        # A bbox midpoint is deterministic and avoids accumulation-order drift.
        centre = tuple(
            (min(point[axis] for point in converted_points)
             + max(point[axis] for point in converted_points))
            / 2.0
            for axis in range(3)
        )
        normalized_points = [
            [round(point[axis] - centre[axis], 9) for axis in range(3)]
            for point in converted_points
        ]

    counts: list[int] = []
    for index, value in enumerate(face_vertex_counts):
        if isinstance(value, bool):
            raise RuntimeError(f"Mesh face count[{index}] is not an integer")
        try:
            converted = int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Mesh face count[{index}] is not an integer"
            ) from exc
        if converted < 0:
            raise RuntimeError(f"Mesh face count[{index}] is negative")
        counts.append(converted)

    indices: list[int] = []
    for index, value in enumerate(face_vertex_indices):
        if isinstance(value, bool):
            raise RuntimeError(f"Mesh face index[{index}] is not an integer")
        try:
            converted = int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Mesh face index[{index}] is not an integer"
            ) from exc
        indices.append(converted)
    if sum(counts) != len(indices):
        raise RuntimeError(
            "Mesh face counts do not cover the face-vertex index array: "
            f"sum={sum(counts)} indices={len(indices)}"
        )
    if any(index < 0 or index >= len(converted_points) for index in indices):
        raise RuntimeError("Mesh face-vertex index is outside the point array")

    return _canonical_sha256(
        {
            "schema_version": "qwen-mesh-content/v1",
            "points_translation_normalized_1e-9": normalized_points,
            "face_vertex_counts": counts,
            "face_vertex_indices": indices,
            "orientation": str(orientation) if orientation is not None else None,
            "subdivision_scheme": (
                str(subdivision_scheme)
                if subdivision_scheme is not None
                else None
            ),
        }
    )


def source_appearance_sha256(properties: Mapping[str, Any] | None) -> str:
    """Hash source shader evidence without prim- or file-location identity."""

    stable_properties = (
        {
            str(key): value
            for key, value in properties.items()
            if str(key) != "shader_path"
        }
        if isinstance(properties, Mapping)
        else None
    )
    return _canonical_sha256(
        {
            "schema_version": "qwen-source-appearance/v1",
            "properties": stable_properties,
        }
    )


def source_subset_layout_sha256(
    *,
    records: Sequence[Mapping[str, Any]],
    appearance_hash_by_material_path: Mapping[str, str],
) -> str:
    """Hash face-subset layout plus path-free source appearance evidence."""

    subsets: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise RuntimeError(f"Source subset[{index}] must be an object")
        face_indices = record.get("face_indices")
        if not isinstance(face_indices, list) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in face_indices
        ):
            raise RuntimeError(
                f"Source subset[{index}].face_indices must be an integer array"
            )
        material_path = record.get("visual_material_prim_path")
        appearance_sha = (
            appearance_hash_by_material_path.get(material_path)
            if isinstance(material_path, str)
            else source_appearance_sha256(None)
        )
        if not isinstance(appearance_sha, str) or not appearance_sha:
            raise RuntimeError(
                f"Source subset[{index}] has no path-free appearance hash"
            )
        subsets.append(
            {
                "element_type": record.get("element_type"),
                "family_type": record.get("family_type"),
                "face_indices": list(face_indices),
                "source_appearance_sha256": appearance_sha,
            }
        )
    subsets.sort(
        key=lambda item: (
            item["face_indices"],
            item["source_appearance_sha256"],
            str(item["element_type"]),
            str(item["family_type"]),
        )
    )
    return _canonical_sha256(
        {
            "schema_version": "qwen-source-subset-layout/v1",
            "subsets": subsets,
        }
    )


def source_material_bind_subset_sha256(
    *,
    part_id: str,
    prim_path: str,
    subset_record: Mapping[str, Any],
) -> str:
    """Hash one source subset's immutable topology and visual binding.

    The part registry itself is hash-bound by the exact-cover plan.  This
    per-subset digest additionally makes it possible to audit one Mesh without
    relying on list position.
    """

    return _canonical_sha256(
        {
            "part_id": part_id,
            "prim_path": prim_path,
            "subset": {
                field: subset_record.get(field)
                for field in sorted(_SOURCE_SUBSET_FIELDS)
            },
        }
    )


def _validated_material_bind_face_subset_records(
    *,
    part_id: str,
    prim_path: str,
    face_count: int,
    records: Sequence[Mapping[str, Any]],
    verify_hashes: bool,
) -> list[dict[str, Any]]:
    """Validate and canonically order source ``materialBind`` face subsets."""

    if (
        isinstance(face_count, bool)
        or not isinstance(face_count, int)
        or face_count < 0
    ):
        raise RuntimeError(f"Invalid face count for {part_id}: {face_count!r}")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise RuntimeError(f"{part_id} source materialBind subsets must be an array")

    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    claimed_faces: set[int] = set()
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise RuntimeError(
                f"{part_id} source materialBind subset[{index}] must be an object"
            )
        allowed_fields = _SOURCE_SUBSET_FIELDS | (
            {SOURCE_SUBSET_HASH_FIELD} if verify_hashes else set()
        )
        unexpected = set(raw) - allowed_fields
        if unexpected:
            raise RuntimeError(
                f"{part_id} source materialBind subset[{index}] has unexpected "
                f"fields: {sorted(unexpected)}"
            )
        missing = _SOURCE_SUBSET_FIELDS - set(raw)
        if missing:
            raise RuntimeError(
                f"{part_id} source materialBind subset[{index}] is missing fields: "
                f"{sorted(missing)}"
            )

        subset_name = raw.get("subset_name")
        subset_path = raw.get("subset_prim_path")
        if not isinstance(subset_name, str) or not subset_name:
            raise RuntimeError(
                f"{part_id} source materialBind subset[{index}] has no name"
            )
        expected_path = f"{prim_path}/{subset_name}"
        if subset_path != expected_path:
            raise RuntimeError(
                f"{part_id}.{subset_name} is not a direct Mesh child: "
                f"{subset_path!r} != {expected_path!r}"
            )
        if subset_name in seen_names or subset_path in seen_paths:
            raise RuntimeError(
                f"{part_id} has duplicate source materialBind subset "
                f"{subset_name!r}"
            )
        seen_names.add(subset_name)
        seen_paths.add(str(subset_path))

        if raw.get("family_name") != "materialBind":
            raise RuntimeError(
                f"{part_id}.{subset_name} is not in the materialBind family"
            )
        if raw.get("family_type") not in {
            "unrestricted",
            "nonOverlapping",
            "partition",
        }:
            raise RuntimeError(
                f"{part_id}.{subset_name} has an invalid materialBind "
                f"family type: {raw.get('family_type')!r}"
            )
        if raw.get("element_type") != "face":
            raise RuntimeError(
                f"{part_id}.{subset_name} is not a face subset"
            )

        indices = raw.get("face_indices")
        if not isinstance(indices, list) or not indices:
            raise RuntimeError(
                f"{part_id}.{subset_name} face_indices must be a non-empty array"
            )
        if any(
            isinstance(face_index, bool) or not isinstance(face_index, int)
            for face_index in indices
        ):
            raise RuntimeError(
                f"{part_id}.{subset_name} face_indices must contain integers"
            )
        if len(set(indices)) != len(indices):
            raise RuntimeError(
                f"{part_id}.{subset_name} face_indices must be unique"
            )
        out_of_range = sorted(
            face_index
            for face_index in indices
            if face_index < 0 or face_index >= face_count
        )
        if out_of_range:
            raise RuntimeError(
                f"{part_id}.{subset_name} face_indices are outside "
                f"[0, {face_count}): {out_of_range}"
            )
        overlap = sorted(claimed_faces & set(indices))
        if overlap:
            raise RuntimeError(
                f"{part_id} source materialBind subsets overlap at faces: {overlap}"
            )
        claimed_faces.update(indices)

        for field in (
            "visual_material_prim_path",
            "binding_relationship_name",
        ):
            value = raw.get(field)
            if value is not None and not isinstance(value, str):
                raise RuntimeError(
                    f"{part_id}.{subset_name}.{field} must be a string or null"
                )
        targets = raw.get("binding_targets")
        if not isinstance(targets, list) or any(
            not isinstance(target, str) or not target for target in targets
        ):
            raise RuntimeError(
                f"{part_id}.{subset_name}.binding_targets must be a string array"
            )
        if targets != sorted(set(targets)):
            raise RuntimeError(
                f"{part_id}.{subset_name}.binding_targets must be sorted and unique"
            )

        record = {
            field: (
                list(raw[field])
                if field in {"face_indices", "binding_targets"}
                else raw[field]
            )
            for field in sorted(_SOURCE_SUBSET_FIELDS)
        }
        expected_digest = source_material_bind_subset_sha256(
            part_id=part_id,
            prim_path=prim_path,
            subset_record=record,
        )
        if verify_hashes:
            if raw.get(SOURCE_SUBSET_HASH_FIELD) != expected_digest:
                raise RuntimeError(
                    f"{part_id}.{subset_name} has an invalid source subset hash"
                )
        record[SOURCE_SUBSET_HASH_FIELD] = expected_digest
        normalized.append(record)

    return sorted(normalized, key=lambda item: item["subset_prim_path"])


def validated_source_material_bind_face_subsets(
    *,
    part_id: str,
    prim_path: str,
    face_count: int,
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate a registry snapshot of immutable source face subsets.

    This public, USD-free boundary lets non-Isaac planning processes verify
    registry evidence without importing ``pxr``.  The eventual USD apply stage
    independently verifies the same records against the composed asset.
    """

    return _validated_material_bind_face_subset_records(
        part_id=part_id,
        prim_path=prim_path,
        face_count=face_count,
        records=records,
        verify_hashes=True,
    )


def _start_isaac_if_needed(headless: bool = True):
    try:
        from pxr import Usd  # noqa: F401

        return None
    except ImportError:
        try:
            from isaacsim import SimulationApp
        except ImportError as exc:
            raise RuntimeError(
                "pxr is unavailable. Run this script with Isaac Sim python.sh."
            ) from exc
        return SimulationApp({"headless": headless})


def _range_to_list(value: Any) -> list[list[float]] | None:
    if value is None or value.IsEmpty():
        return None
    minimum = value.GetMin()
    maximum = value.GetMax()
    return [
        [float(minimum[0]), float(minimum[1]), float(minimum[2])],
        [float(maximum[0]), float(maximum[1]), float(maximum[2])],
    ]


def _finite_scalar(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if converted != converted or converted in {float("inf"), float("-inf")}:
        return None
    return converted


def _finite_color(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        channels = list(value)
    except TypeError:
        return None
    if len(channels) not in {3, 4}:
        return None
    converted = [_finite_scalar(channel) for channel in channels]
    if any(channel is None for channel in converted):
        return None
    return [float(channel) for channel in converted if channel is not None]


def _preview_material_properties(material: Any) -> dict[str, Any] | None:
    """Return weak CAD appearance evidence without treating it as ground truth."""

    if not material:
        return None
    try:
        shader, _source_name, _source_type = material.ComputeSurfaceSource()
    except Exception:
        return None
    if not shader or not shader.GetPrim().IsValid():
        return None

    identifier = shader.GetIdAttr().Get()
    properties: dict[str, Any] = {
        "shader_path": shader.GetPath().pathString,
        "shader_id": str(identifier) if identifier else None,
    }
    for input_name in ("diffuseColor", "emissiveColor"):
        shader_input = shader.GetInput(input_name)
        color = _finite_color(shader_input.Get()) if shader_input else None
        if color is not None:
            properties[input_name] = color
    for input_name in ("metallic", "roughness", "opacity"):
        shader_input = shader.GetInput(input_name)
        scalar = _finite_scalar(shader_input.Get()) if shader_input else None
        if scalar is not None:
            properties[input_name] = scalar
    return properties


def _material_bind_face_subset_records(
    *,
    mesh: Any,
    part_id: str,
    prim_path: str,
    face_count: int,
) -> list[dict[str, Any]]:
    """Snapshot valid source material subsets before inference can replace them."""

    from pxr import UsdGeom, UsdShade

    raw_subsets: list[Any] = []
    for child in mesh.GetPrim().GetChildren():
        if not child.IsA(UsdGeom.Subset):
            continue
        subset = UsdGeom.Subset(child)
        if str(subset.GetFamilyNameAttr().Get()) == str(
            UsdShade.Tokens.materialBind
        ):
            raw_subsets.append(subset)
    if not raw_subsets:
        return []

    family_type = str(
        UsdGeom.Subset.GetFamilyType(mesh, UsdShade.Tokens.materialBind)
    )
    family_valid, validation_reason = UsdGeom.Subset.ValidateFamily(
        mesh,
        UsdGeom.Tokens.face,
        UsdShade.Tokens.materialBind,
    )
    if not family_valid:
        raise RuntimeError(
            f"Existing materialBind family is invalid for {part_id}: "
            f"{validation_reason}"
        )

    records: list[dict[str, Any]] = []
    for subset in sorted(
        raw_subsets, key=lambda item: item.GetPath().pathString
    ):
        subset_prim = subset.GetPrim()
        material, relationship = UsdShade.MaterialBindingAPI(
            subset_prim
        ).ComputeBoundMaterial(materialPurpose=UsdShade.Tokens.allPurpose)
        relationship_valid = bool(
            relationship
            and relationship.IsValid()
        )
        records.append(
            {
                "subset_name": subset_prim.GetName(),
                "subset_prim_path": subset_prim.GetPath().pathString,
                "family_name": str(subset.GetFamilyNameAttr().Get()),
                "family_type": family_type,
                "element_type": str(subset.GetElementTypeAttr().Get()),
                # Preserve the authored order.  USD application validates the
                # exact source attribute rather than treating it as permission
                # to rewrite topology.
                "face_indices": [
                    int(value) for value in (subset.GetIndicesAttr().Get() or [])
                ],
                "visual_material_prim_path": (
                    material.GetPath().pathString if material else None
                ),
                "binding_relationship_name": (
                    relationship.GetName() if relationship_valid else None
                ),
                "binding_targets": (
                    sorted(
                        {
                            target.pathString
                            for target in relationship.GetTargets()
                        }
                    )
                    if relationship_valid
                    else []
                ),
            }
        )
    return _validated_material_bind_face_subset_records(
        part_id=part_id,
        prim_path=prim_path,
        face_count=face_count,
        records=records,
        verify_hashes=False,
    )


def build_part_registry(asset_usd: str | Path) -> dict[str, Any]:
    from pxr import Usd, UsdGeom, UsdShade

    asset_path = Path(asset_usd).expanduser().resolve(strict=True)
    stage = Usd.Stage.Open(str(asset_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Unable to open USD stage: {asset_path}")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    mesh_prims = sorted(
        (prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)),
        key=lambda prim: prim.GetPath().pathString,
    )
    instance_root_count = sum(prim.IsInstance() for prim in stage.Traverse())

    parts: list[dict[str, Any]] = []
    for index, prim in enumerate(mesh_prims, start=1):
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or []
        face_counts = mesh.GetFaceVertexCountsAttr().Get() or []
        face_indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        part_id = f"P{index:04d}"
        prim_path = prim.GetPath().pathString
        world_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()

        visual_material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
            materialPurpose=UsdShade.Tokens.allPurpose
        )
        physics_material, physics_relationship = UsdShade.MaterialBindingAPI(
            prim
        ).ComputeBoundMaterial(materialPurpose="physics")
        # ComputeBoundMaterial falls back to the all-purpose visual material
        # when no explicit physics binding exists.  Preserve that distinction
        # in the registry so downstream inference is not given false physics
        # metadata.
        physics_relationship_name = (
            physics_relationship.GetName()
            if physics_relationship and physics_relationship.IsValid()
            else ""
        )
        has_explicit_physics_binding = (
            physics_relationship_name == "material:binding:physics"
            or physics_relationship_name.startswith(
                "material:binding:collection:physics:"
            )
        )
        source_visual_properties = _preview_material_properties(visual_material)
        source_subsets = _material_bind_face_subset_records(
            mesh=mesh,
            part_id=part_id,
            prim_path=prim_path,
            face_count=len(face_counts),
        )
        subset_appearance_hashes: dict[str, str] = {}
        for subset in source_subsets:
            subset_material_path = subset.get("visual_material_prim_path")
            if (
                not isinstance(subset_material_path, str)
                or subset_material_path in subset_appearance_hashes
            ):
                continue
            subset_material_prim = stage.GetPrimAtPath(subset_material_path)
            subset_material = (
                UsdShade.Material(subset_material_prim)
                if subset_material_prim and subset_material_prim.IsValid()
                else None
            )
            subset_appearance_hashes[subset_material_path] = (
                source_appearance_sha256(
                    _preview_material_properties(subset_material)
                )
            )
        parts.append(
            {
                "part_id": part_id,
                "prim_path": prim_path,
                "prim_name": prim.GetName(),
                "parent_path": prim.GetParent().GetPath().pathString,
                "point_count": len(points),
                "face_count": len(face_counts),
                GEOMETRY_CONTENT_HASH_FIELD: geometry_content_sha256(
                    points=points,
                    face_vertex_counts=face_counts,
                    face_vertex_indices=face_indices,
                    orientation=mesh.GetOrientationAttr().Get(),
                    subdivision_scheme=mesh.GetSubdivisionSchemeAttr().Get(),
                ),
                "world_bbox": _range_to_list(world_range),
                "existing_visual_material": (
                    visual_material.GetPath().pathString if visual_material else None
                ),
                # CAD Converter commonly authors generic UsdPreviewSurface values.
                # They are useful as weak clustering evidence for hidden/repeated
                # parts, but are not asserted to be the physical reference material.
                "existing_visual_material_properties": source_visual_properties,
                SOURCE_APPEARANCE_HASH_FIELD: source_appearance_sha256(
                    source_visual_properties
                ),
                "existing_physics_material": (
                    physics_material.GetPath().pathString
                    if physics_material and has_explicit_physics_binding
                    else None
                ),
                SOURCE_MATERIAL_BIND_SUBSETS_FIELD: source_subsets,
                SOURCE_SUBSET_LAYOUT_HASH_FIELD: source_subset_layout_sha256(
                    records=source_subsets,
                    appearance_hash_by_material_path=subset_appearance_hashes,
                ),
                "renders": [],
            }
        )

    default_prim = stage.GetDefaultPrim()
    return {
        "schema_version": SCHEMA_VERSION,
        "asset_usd": str(asset_path),
        "asset_sha256": _sha256_file(asset_path),
        "default_prim": (
            default_prim.GetPath().pathString
            if default_prim and default_prim.IsValid()
            else None
        ),
        "part_count": len(parts),
        "instance_root_count": instance_root_count,
        "parts": parts,
    }


def write_registry(registry: dict[str, Any], output: str | Path) -> Path:
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stable IDs for USD Mesh prims"
    )
    parser.add_argument("--usd", required=True, help="Input .usd/.usda/.usdc asset")
    parser.add_argument("--output", required=True, help="Output part_registry.json")
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = _start_isaac_if_needed(headless=args.headless)
    exit_code = 0
    try:
        registry = build_part_registry(args.usd)
        output = write_registry(registry, args.output)
        print(
            json.dumps(
                {"output": str(output), "part_count": registry["part_count"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        if app is not None:
            app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
