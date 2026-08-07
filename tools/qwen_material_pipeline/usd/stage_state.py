"""Public read-only state snapshots shared by USD application and validation.

The functions in this module understand instance-proxy traversal but never
author a layer.  Keeping them separate prevents the validator from depending
on implementation details of the material application command.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


GEOMETRY_ATTRIBUTES = (
    "points",
    "faceVertexCounts",
    "faceVertexIndices",
    "holeIndices",
    "orientation",
    "subdivisionScheme",
)


def canonical_sha256(value: Mapping[str, Any]) -> str:
    """Return the stable JSON content digest used by plan provenance."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def mesh_prims(stage: Any, *, instance_proxies: bool) -> dict[str, Any]:
    """Index composed Mesh prims, optionally traversing instance proxies."""

    from pxr import Usd, UsdGeom

    traversal = (
        Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
        if instance_proxies
        else stage.Traverse()
    )
    return {
        prim.GetPath().pathString: prim for prim in traversal if prim.IsA(UsdGeom.Mesh)
    }


def _value_digest(value: Any) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def geometry_state(stage: Any, mesh_paths: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Return stable digests of all protected geometry/topology attributes."""

    result: dict[str, tuple[str, ...]] = {}
    for path in sorted(mesh_paths):
        prim = stage.GetPrimAtPath(path)
        result[path] = tuple(
            _value_digest(prim.GetAttribute(name).Get()) for name in GEOMETRY_ATTRIBUTES
        )
    return result


def world_matrix_state(
    stage: Any, mesh_paths: Iterable[str]
) -> dict[str, tuple[float, ...]]:
    """Snapshot default-time local-to-world matrices for Mesh occurrences."""

    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    result: dict[str, tuple[float, ...]] = {}
    for path in sorted(mesh_paths):
        matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        result[path] = tuple(
            float(matrix[row][column]) for row in range(4) for column in range(4)
        )
    return result


def matrix_states_close(
    left: Mapping[str, tuple[float, ...]],
    right: Mapping[str, tuple[float, ...]],
) -> bool:
    """Compare two world-matrix snapshots with the established tolerance."""

    if set(left) != set(right):
        return False
    return all(
        all(
            math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
            for a, b in zip(left[path], right[path], strict=True)
        )
        for path in left
    )


def subset_state(stage: Any, mesh_paths: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Snapshot existing GeomSubset membership and exact face indices."""

    from pxr import UsdGeom

    result: dict[str, dict[str, Any]] = {}
    for mesh_path in sorted(mesh_paths):
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        for child in mesh_prim.GetChildren():
            if not child.IsA(UsdGeom.Subset):
                continue
            subset = UsdGeom.Subset(child)
            path = child.GetPath().pathString
            result[path] = {
                "mesh_path": mesh_path,
                "element_type": repr(subset.GetElementTypeAttr().Get()),
                "family_name": repr(subset.GetFamilyNameAttr().Get()),
                "indices_sha256": _value_digest(subset.GetIndicesAttr().Get()),
                "indices": tuple(
                    int(value) for value in (subset.GetIndicesAttr().Get() or [])
                ),
            }
    return result


def _all_composed_prims(stage: Any, *, instance_proxies: bool) -> dict[str, Any]:
    from pxr import Usd

    result = {prim.GetPath().pathString: prim for prim in stage.TraverseAll()}
    if instance_proxies:
        result.update(
            {
                prim.GetPath().pathString: prim
                for prim in Usd.PrimRange.Stage(
                    stage, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
                )
            }
        )
    return result


def explicit_physics_state(stage: Any, *, instance_proxies: bool) -> dict[str, Any]:
    """Snapshot only explicitly authored Physics/PhysX opinions.

    ``ComputeBoundMaterial(materialPurpose='physics')`` falls back to the visual
    all-purpose binding, so it cannot be used to prove a physics binding exists.
    """

    result: dict[str, Any] = {}
    for path, prim in sorted(
        _all_composed_prims(stage, instance_proxies=instance_proxies).items()
    ):
        attributes = {
            attribute.GetName(): _value_digest(attribute.Get())
            for attribute in prim.GetAttributes()
            if attribute.GetName().startswith(("physics:", "physx"))
        }
        schemas = sorted(
            schema
            for schema in prim.GetAppliedSchemas()
            if "physics" in schema.casefold() or "physx" in schema.casefold()
        )
        relationships = {
            relationship.GetName(): tuple(
                target.pathString for target in relationship.GetTargets()
            )
            for relationship in prim.GetRelationships()
            if relationship.GetName().startswith(("physics:", "physx"))
            or relationship.GetName() == "material:binding:physics"
        }
        if attributes or schemas or relationships:
            result[path] = {
                "attributes": attributes,
                "schemas": schemas,
                "relationships": relationships,
            }
    return result


def source_layer_is_in_stack(prim: Any, source_path: Path) -> bool:
    """Return whether a composed prim stack contains the expected source layer."""

    source_real_path = str(source_path.resolve())
    return any(
        spec.layer.realPath
        and str(Path(spec.layer.realPath).resolve()) == source_real_path
        for spec in prim.GetPrimStack()
    )


def material_binding_path(prim: Any) -> str | None:
    """Resolve the all-purpose material bound to a composed prim."""

    from pxr import UsdShade

    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
        materialPurpose=UsdShade.Tokens.allPurpose
    )
    return material.GetPath().pathString if material else None


__all__ = [
    "GEOMETRY_ATTRIBUTES",
    "canonical_sha256",
    "explicit_physics_state",
    "geometry_state",
    "material_binding_path",
    "matrix_states_close",
    "mesh_prims",
    "source_layer_is_in_stack",
    "subset_state",
    "world_matrix_state",
]
