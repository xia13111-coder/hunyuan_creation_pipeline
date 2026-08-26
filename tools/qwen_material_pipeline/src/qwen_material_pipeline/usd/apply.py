#!/usr/bin/env python3
"""Author a non-destructive USD layer containing validated MDL bindings.

Assignments may either replace a Mesh's parent visual binding or, when
``preserve_parent_material_binding`` is explicitly true, author only validated
``materialBind`` face subsets.  The latter mode intentionally leaves uncovered
faces on the source Mesh's existing (possibly inherited) visual material.

An existing ``materialBind`` subset is a descendant binding and can therefore
shadow a newly selected whole-Mesh material.  For a whole-Mesh assignment this
module explicitly rebinds every such source subset to the selected parent MDL,
using the same parameters, while leaving the subset family and face indices
entirely source-authored.  Explicit planned face subsets remain fail-closed:
their names and indices must exactly match the complete existing family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import traceback
from pathlib import Path
from typing import Any

from qwen_material_pipeline.materials.selection_lock import (
    validate_material_selection_lock as _validate_material_selection_lock,
)
from qwen_material_pipeline.usd.material_common import (
    APPLY_STATUSES,
    POLICY_FALLBACK_STATUS,
    NormalizedParameter,
    canonical_sha256 as _canonical_sha256,
    catalog_map as _catalog_map,
    get_subidentifier as _get_subidentifier,
    json_material_parameters as _json_material_parameters,
    material_parameter_policy as _material_parameter_policy,
    material_instance_key as _material_instance_key,
    normalize_face_subsets as _normalize_face_subsets,
    normalize_material_parameters as _normalize_material_parameters,
    preserve_parent_material_binding as _preserve_parent_material_binding,
    resolve_mdl_path as _resolve_mdl_path,
    safe_child_name as _safe_child_name,
    sha256_file as _sha256_file,
    validate_policy_fallback_authorization as _validate_policy_fallback_authorization,
    validate_source_visual_preserve as _validate_source_visual_preserve,
)
from qwen_material_pipeline.usd.stage_state import (
    explicit_physics_state as _explicit_physics_state,
)


VISUAL_BINDING_RELATIONSHIP_NAMES = (
    "material:binding",
    "material:binding:preview",
    "material:binding:full",
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


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {resolved}")
    return value


def _registry_map(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    parts = registry.get("parts")
    if not isinstance(parts, list):
        raise ValueError("Part registry must contain a parts list")
    result: dict[str, dict[str, Any]] = {}
    for item in parts:
        if isinstance(item, dict) and isinstance(item.get("part_id"), str):
            part_id = item["part_id"]
            if part_id in result:
                raise ValueError(f"Part registry contains duplicate part_id: {part_id}")
            result[part_id] = item
    if not result:
        raise ValueError("Part registry contains no part_id entries")
    return result


def _visual_binding_path(prim):
    from pxr import UsdShade

    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
        materialPurpose=UsdShade.Tokens.allPurpose
    )
    return material.GetPath().pathString if material else None


def _prim_invariant_state(
    stage, prim_paths: set[str], *, physics: bool
) -> dict[str, Any]:
    """Snapshot physics or xform opinions for source prims only."""

    result: dict[str, Any] = {}
    for path in sorted(prim_paths):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            result[path] = None
            continue
        if physics:
            attributes = {
                attribute.GetName(): repr(attribute.Get())
                for attribute in prim.GetAttributes()
                if attribute.GetName().startswith(("physics:", "physx"))
            }
            schemas = sorted(
                name
                for name in prim.GetAppliedSchemas()
                if "physics" in name.casefold() or "physx" in name.casefold()
            )
            relationships = {
                relationship.GetName(): [
                    target.pathString for target in relationship.GetTargets()
                ]
                for relationship in prim.GetRelationships()
                if relationship.GetName().startswith(("physics:", "physx"))
                or relationship.GetName() == "material:binding:physics"
            }
            result[path] = {
                "attributes": attributes,
                "schemas": schemas,
                "relationships": relationships,
            }
        else:
            result[path] = {
                attribute.GetName(): repr(attribute.Get())
                for attribute in prim.GetAttributes()
                if attribute.GetName() == "xformOpOrder"
                or attribute.GetName().startswith("xformOp:")
            }
    return result


def _mesh_geometry_state(stage, mesh_paths: set[str]) -> dict[str, dict[str, str]]:
    """Snapshot exact mesh geometry/topology values, not only array lengths."""

    attribute_names = (
        "points",
        "faceVertexCounts",
        "faceVertexIndices",
        "holeIndices",
        "orientation",
        "subdivisionScheme",
    )
    result: dict[str, dict[str, str]] = {}
    for path in sorted(mesh_paths):
        prim = stage.GetPrimAtPath(path)
        result[path] = {
            name: repr(prim.GetAttribute(name).Get()) for name in attribute_names
        }
    return result


def _match_existing_material_subsets(
    part_id: str,
    face_subsets: list[dict[str, Any]],
    existing_subsets: list[dict[str, Any]],
) -> int:
    """Bind planned subsets to identical source subsets, or fail closed.

    A material look layer may safely override an existing subset's material
    relationship only when the source and plan describe exactly the same
    ``materialBind`` family.  In particular, this helper never treats a
    same-named subset as permission to change its topology.
    """

    if not existing_subsets:
        return 0

    planned_by_name = {
        face_subset["subset_name"]: face_subset for face_subset in face_subsets
    }
    existing_by_name = {
        source_subset["subset_name"]: source_subset
        for source_subset in existing_subsets
    }
    if set(planned_by_name) != set(existing_by_name):
        missing_from_plan = sorted(set(existing_by_name) - set(planned_by_name))
        missing_from_source = sorted(set(planned_by_name) - set(existing_by_name))
        raise ValueError(
            "Existing materialBind subset membership does not exactly match "
            f"the plan for {part_id}: unplanned_source={missing_from_plan}, "
            f"missing_source={missing_from_source}"
        )

    for subset_name, planned in planned_by_name.items():
        source = existing_by_name[subset_name]
        if source.get("element_type") != "face":
            raise ValueError(
                f"Existing subset is not a face subset: {part_id}.{subset_name}"
            )
        if source.get("family_name") != "materialBind":
            raise ValueError(
                "Existing subset is not in the materialBind family: "
                f"{part_id}.{subset_name}"
            )
        if list(source.get("face_indices", [])) != list(planned["face_indices"]):
            raise ValueError(
                "Existing materialBind subset indices differ from the plan: "
                f"{part_id}.{subset_name}"
            )
        subset_prim_path = source.get("subset_prim_path")
        if not isinstance(subset_prim_path, str) or not subset_prim_path:
            raise ValueError(
                f"Existing subset has no valid prim path: {part_id}.{subset_name}"
            )
        planned["source_subset_rebind"] = True
        planned["source_subset_prim_path"] = subset_prim_path
        planned["explicit_plan_override"] = True
        planned["parent_assignment_propagated"] = False
    return len(existing_subsets)


def _propagate_parent_assignment_to_existing_subsets(
    part_id: str,
    existing_subsets: list[dict[str, Any]],
    *,
    material_id: str,
    parameters: dict[str, NormalizedParameter],
) -> list[dict[str, Any]]:
    """Build look-only rebind specs for source subsets under a parent assignment.

    The returned records deliberately carry only material-selection data plus
    the exact source subset contract.  The authoring path overrides each
    subset's material relationship, but never authors ``indices``,
    ``elementType``, ``familyName``, or the family's ``familyType``.
    """

    propagated: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for source in existing_subsets:
        subset_name = source.get("subset_name")
        if not isinstance(subset_name, str) or not subset_name:
            raise ValueError(f"Existing subset has no valid name: {part_id}")
        if subset_name in seen_names:
            raise ValueError(
                f"Existing materialBind subset name is duplicated: "
                f"{part_id}.{subset_name}"
            )
        seen_names.add(subset_name)
        if source.get("element_type") != "face":
            raise ValueError(
                f"Existing subset is not a face subset: {part_id}.{subset_name}"
            )
        if source.get("family_name") != "materialBind":
            raise ValueError(
                "Existing subset is not in the materialBind family: "
                f"{part_id}.{subset_name}"
            )
        subset_prim_path = source.get("subset_prim_path")
        if not isinstance(subset_prim_path, str) or not subset_prim_path:
            raise ValueError(
                f"Existing subset has no valid prim path: {part_id}.{subset_name}"
            )
        face_indices = source.get("face_indices")
        if (
            not isinstance(face_indices, list)
            or any(type(index) is not int or index < 0 for index in face_indices)
            or len(face_indices) != len(set(face_indices))
        ):
            raise ValueError(
                "Existing materialBind subset has invalid face indices: "
                f"{part_id}.{subset_name}"
            )
        propagated.append(
            {
                "subset_name": subset_name,
                "face_indices": tuple(face_indices),
                "material_id": material_id,
                "parameters": dict(parameters),
                "source_subset_rebind": True,
                "source_subset_prim_path": subset_prim_path,
                "explicit_plan_override": False,
                "parent_assignment_propagated": True,
            }
        )
    return propagated


def apply_visual_materials(
    *,
    asset_usd: str | Path,
    catalog_path: str | Path,
    registry_path: str | Path,
    plan_path: str | Path,
    output_usd: str | Path,
    material_root: str | Path,
    include_review: bool = False,
    include_policy_fallback: bool = False,
    selection_lock_path: str | Path | None = None,
) -> dict[str, Any]:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    asset_path = Path(asset_usd).expanduser().resolve(strict=True)
    output_path = Path(output_usd).expanduser().resolve()
    if output_path == asset_path:
        raise ValueError("Output layer must not overwrite the source asset")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(material_root).expanduser().resolve(strict=True)

    catalog = _catalog_map(_load_json(catalog_path))
    registry_document = _load_json(registry_path)
    registry = _registry_map(registry_document)
    plan = _load_json(plan_path)
    if plan.get("schema_version") != "1.0":
        raise ValueError("Material plan schema_version must be '1.0'")
    selection_lock_verified = False
    if selection_lock_path is not None:
        _validate_material_selection_lock(
            lock=_load_json(selection_lock_path),
            plan=plan,
            catalog_path=catalog_path,
            material_root=root,
        )
        selection_lock_verified = True
    assignments = plan.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("Material plan must contain an assignments list")

    source_stage = Usd.Stage.Open(str(asset_path), load=Usd.Stage.LoadAll)
    if source_stage is None:
        raise RuntimeError(f"Unable to open source stage: {asset_path}")
    source_default = source_stage.GetDefaultPrim()
    if not source_default or not source_default.IsValid():
        raise ValueError("Source stage has no valid default prim")
    default_path = source_default.GetPath()

    # Validate the registry belongs to this source and capture the physics state.
    registry_asset = Path(registry_document.get("asset_usd", "")).expanduser().resolve()
    if registry_asset != asset_path:
        raise ValueError(
            f"Registry asset mismatch: expected {asset_path}, found {registry_asset}"
        )
    registry_hash = registry_document.get("asset_sha256")
    if not isinstance(registry_hash, str) or registry_hash != _sha256_file(asset_path):
        raise ValueError("Registry SHA-256 does not match the current source asset")
    policy_fallback_count = _validate_policy_fallback_authorization(
        plan,
        registry_document,
        include_policy_fallback=include_policy_fallback,
    )
    source_meshes = {
        prim.GetPath().pathString: prim
        for prim in source_stage.Traverse()
        if prim.IsA(UsdGeom.Mesh)
    }
    # Do not use ComputeBoundMaterial(materialPurpose="physics") here.  USD
    # deliberately falls back to the all-purpose visual binding when no
    # purpose-specific physics binding exists.  Replacing a visual material
    # would then look like a physics mutation even though no physics opinion
    # changed.  Snapshot only explicitly authored Physics/PhysX state.
    source_explicit_physics = _explicit_physics_state(
        source_stage, instance_proxies=False
    )
    source_visual = {
        path: _visual_binding_path(prim) for path, prim in source_meshes.items()
    }
    source_geometry_state = _mesh_geometry_state(source_stage, set(source_meshes))
    source_prim_paths = {prim.GetPath().pathString for prim in source_stage.Traverse()}
    source_physics_state = _prim_invariant_state(
        source_stage, source_prim_paths, physics=True
    )
    source_xform_state = _prim_invariant_state(
        source_stage, source_prim_paths, physics=False
    )

    allowed_statuses = set(APPLY_STATUSES)
    if include_review:
        allowed_statuses.add("review")
    if policy_fallback_count:
        allowed_statuses.add(POLICY_FALLBACK_STATUS)

    validated: list[
        tuple[
            dict[str, Any],
            dict[str, Any],
            Path | None,
            str | None,
            dict[str, NormalizedParameter] | None,
            list[dict[str, Any]],
            bool,
            bool,
            list[dict[str, Any]],
            int,
            int,
            str,
        ]
    ] = []
    seen_parts: set[str] = set()
    skipped: list[dict[str, str]] = []
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise ValueError("Every assignment must be a JSON object")
        part_id = assignment.get("part_id")
        material_id = assignment.get("material_id")
        status = assignment.get("status", "review")
        if status not in allowed_statuses:
            skipped.append({"part_id": str(part_id), "reason": f"status={status}"})
            continue
        confidence = assignment.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ValueError(f"Invalid confidence for part {part_id}: {confidence!r}")
        if status == "auto" and float(confidence) < 0.85:
            raise ValueError(f"Auto assignment below 0.85 confidence: {part_id}")
        if status == "review" and float(confidence) < 0.60:
            raise ValueError(f"Review assignment below 0.60 confidence: {part_id}")
        if not isinstance(part_id, str) or part_id not in registry:
            raise ValueError(f"Unknown part_id: {part_id!r}")
        if part_id in seen_parts:
            raise ValueError(f"Duplicate assignment for part_id: {part_id}")
        seen_parts.add(part_id)
        part = registry[part_id]
        prim_path = part.get("prim_path")
        if not isinstance(prim_path, str) or prim_path not in source_meshes:
            raise ValueError(
                f"Registry part is not a source Mesh: {part_id} -> {prim_path}"
            )
        source_mesh = UsdGeom.Mesh(source_meshes[prim_path])
        source_preserve_path = _validate_source_visual_preserve(
            part_id, assignment, part
        )
        source_visual_preserved = source_preserve_path is not None
        if source_visual_preserved:
            if source_visual[prim_path] != source_preserve_path:
                raise ValueError(
                    "Source visual binding differs from the hash-bound preserve "
                    f"contract: {part_id}"
                )
            face_subsets = []
            source_visual_subset_bindings = []
            for source_subset in UsdGeom.Subset.GetGeomSubsets(
                source_mesh,
                UsdGeom.Tokens.face,
                UsdShade.Tokens.materialBind,
            ):
                subset_prim = source_subset.GetPrim()
                subset_binding = _visual_binding_path(subset_prim)
                if subset_binding is None:
                    raise ValueError(
                        "Source-visual preserve requires every existing "
                        f"materialBind subset to resolve: {subset_prim.GetPath()}"
                    )
                source_visual_subset_bindings.append(
                    {
                        "subset_prim_path": subset_prim.GetPath().pathString,
                        "source_visual_material_prim_path": subset_binding,
                        "face_indices": list(
                            source_subset.GetIndicesAttr().Get() or []
                        ),
                    }
                )
            preserve_parent_binding = True
            mdl_path = None
            subidentifier = None
            parameters = None
        else:
            source_visual_subset_bindings = []
            face_count = len(source_mesh.GetFaceVertexCountsAttr().Get() or [])
            face_subsets = _normalize_face_subsets(
                part_id,
                assignment.get("face_subsets"),
                allowed_material_ids=set(catalog),
                face_count=face_count,
            )
            preserve_parent_binding = _preserve_parent_material_binding(
                part_id,
                assignment,
                has_face_subsets=bool(face_subsets),
            )
            if not isinstance(material_id, str) or material_id not in catalog:
                raise ValueError(f"Unknown material_id: {material_id!r}")
            item = catalog[material_id]
            mdl_path = _resolve_mdl_path(item, root)
            subidentifier = _get_subidentifier(item)
            parameters = _normalize_material_parameters(
                material_id, assignment.get("parameters")
            )
        existing_subset_rebind_count = 0
        parent_assignment_subset_rebind_count = 0
        material_bind_family_type = str(UsdGeom.Tokens.nonOverlapping)
        if not source_visual_preserved:
            existing_material_subsets = UsdGeom.Subset.GetGeomSubsets(
                source_mesh,
                UsdGeom.Tokens.face,
                UsdShade.Tokens.materialBind,
            )
            if existing_material_subsets:
                family_type = UsdGeom.Subset.GetFamilyType(
                    source_mesh, UsdShade.Tokens.materialBind
                )
                material_bind_family_type = str(family_type)
                valid_family, validation_reason = UsdGeom.Subset.ValidateFamily(
                    source_mesh,
                    UsdGeom.Tokens.face,
                    UsdShade.Tokens.materialBind,
                )
                if not valid_family:
                    raise ValueError(
                        f"Existing materialBind family is invalid for {part_id}: "
                        f"{validation_reason}"
                    )
                existing_subset_contracts = []
                for source_subset in existing_material_subsets:
                    source_subset_prim = source_subset.GetPrim()
                    existing_subset_contracts.append(
                        {
                            "subset_name": source_subset_prim.GetName(),
                            "subset_prim_path": (
                                source_subset_prim.GetPath().pathString
                            ),
                            "element_type": str(
                                source_subset.GetElementTypeAttr().Get()
                            ),
                            "family_name": str(
                                source_subset.GetFamilyNameAttr().Get()
                            ),
                            "face_indices": list(
                                source_subset.GetIndicesAttr().Get() or []
                            ),
                        }
                    )
                if face_subsets:
                    existing_subset_rebind_count = _match_existing_material_subsets(
                        part_id,
                        face_subsets,
                        existing_subset_contracts,
                    )
                else:
                    if not isinstance(material_id, str) or parameters is None:
                        raise RuntimeError(
                            f"Validated parent material is missing: {part_id}"
                        )
                    face_subsets = (
                        _propagate_parent_assignment_to_existing_subsets(
                            part_id,
                            existing_subset_contracts,
                            material_id=material_id,
                            parameters=parameters,
                        )
                    )
                    existing_subset_rebind_count = len(face_subsets)
                    parent_assignment_subset_rebind_count = len(face_subsets)
            else:
                for face_subset in face_subsets:
                    face_subset["explicit_plan_override"] = True
                    face_subset["parent_assignment_propagated"] = False
        for face_subset in face_subsets:
            subset_path = source_mesh.GetPath().AppendChild(face_subset["subset_name"])
            if face_subset.get("source_subset_rebind") is True:
                if face_subset.get("source_subset_prim_path") != subset_path.pathString:
                    raise ValueError(
                        "Existing materialBind subset is not the expected direct "
                        f"Mesh child: {part_id}.{face_subset['subset_name']}"
                    )
            elif source_stage.GetPrimAtPath(subset_path).IsValid():
                raise ValueError(
                    f"Face subset prim already exists for {part_id}: {subset_path}"
                )
            subset_item = catalog[face_subset["material_id"]]
            face_subset["mdl_path"] = _resolve_mdl_path(subset_item, root)
            face_subset["subidentifier"] = _get_subidentifier(subset_item)
        validated.append(
            (
                assignment,
                part,
                mdl_path,
                subidentifier,
                parameters,
                face_subsets,
                preserve_parent_binding,
                source_visual_preserved,
                source_visual_subset_bindings,
                existing_subset_rebind_count,
                parent_assignment_subset_rebind_count,
                material_bind_family_type,
            )
        )

    if not validated:
        raise ValueError("No auto/approved material assignments to apply")

    temporary_output_path = output_path.with_name(
        "." + output_path.stem + ".tmp" + output_path.suffix
    )
    temporary_output_path.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(temporary_output_path))
    if stage is None:
        raise RuntimeError(f"Unable to create output stage: {temporary_output_path}")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source_stage))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source_stage))
    stage.SetTimeCodesPerSecond(source_stage.GetTimeCodesPerSecond())
    stage.SetFramesPerSecond(source_stage.GetFramesPerSecond())
    stage.SetStartTimeCode(source_stage.GetStartTimeCode())
    stage.SetEndTimeCode(source_stage.GetEndTimeCode())
    root_prim = stage.DefinePrim(default_path, source_default.GetTypeName() or "Xform")
    reference_path = os.path.relpath(asset_path, start=temporary_output_path.parent)
    root_prim.GetReferences().AddReference(reference_path, default_path)
    stage.SetDefaultPrim(root_prim)
    looks_path = default_path.AppendChild("QwenLooks")
    UsdGeom.Scope.Define(stage, looks_path)

    authored_materials: dict[str, Any] = {}

    def get_or_create_material(
        material_id: str,
        mdl_path: Path,
        subidentifier: str,
        parameters: dict[str, NormalizedParameter],
    ):
        instance_key = _material_instance_key(material_id, parameters)
        if instance_key in authored_materials:
            return authored_materials[instance_key]
        parameter_suffix = (
            "_" + hashlib.sha256(instance_key.encode("utf-8")).hexdigest()[:10]
            if parameters
            else ""
        )
        material_path = looks_path.AppendChild(
            _safe_child_name(material_id) + parameter_suffix
        )
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path.AppendChild("Shader"))
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(Sdf.AssetPath(str(mdl_path)), "mdl")
        shader.SetSourceAssetSubIdentifier(subidentifier, "mdl")
        parameter_policy = _material_parameter_policy(material_id)
        for parameter_name, parameter_value in sorted(parameters.items()):
            policy_kind = parameter_policy[parameter_name][0]
            if policy_kind == "color3f_linear":
                shader.CreateInput(parameter_name, Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(*parameter_value)
                )
            elif policy_kind == "float":
                shader.CreateInput(parameter_name, Sdf.ValueTypeNames.Float).Set(
                    parameter_value
                )
            elif policy_kind == "bool":
                shader.CreateInput(parameter_name, Sdf.ValueTypeNames.Bool).Set(
                    parameter_value
                )
            else:  # The trust-boundary normalizer should make this unreachable.
                raise RuntimeError(
                    f"Unsupported parameter policy for {material_id}.{parameter_name}"
                )
        shader_output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
        shader_output.SetRenderType("material")
        connectable = shader.ConnectableAPI()
        material.CreateSurfaceOutput("mdl").ConnectToSource(connectable, "out")
        material.CreateVolumeOutput("mdl").ConnectToSource(connectable, "out")
        material.CreateDisplacementOutput("mdl").ConnectToSource(connectable, "out")
        authored_materials[instance_key] = material
        return material

    applied: list[dict[str, Any]] = []
    for (
        assignment,
        part,
        mdl_path,
        subidentifier,
        parameters,
        face_subsets,
        preserve_parent_binding,
        source_visual_preserved,
        source_visual_subset_bindings,
        existing_subset_rebind_count,
        parent_assignment_subset_rebind_count,
        material_bind_family_type,
    ) in validated:
        target = stage.OverridePrim(part["prim_path"])
        material = None
        if not preserve_parent_binding:
            if mdl_path is None or subidentifier is None or parameters is None:
                raise RuntimeError(
                    f"Validated parent material is missing: {assignment['part_id']}"
                )
            material_id = assignment["material_id"]
            material = get_or_create_material(
                material_id, mdl_path, subidentifier, parameters
            )
            UsdShade.MaterialBindingAPI.Apply(target).Bind(
                material,
                UsdShade.Tokens.weakerThanDescendants,
                UsdShade.Tokens.allPurpose,
            )
        subset_records: list[dict[str, Any]] = []
        target_mesh = UsdGeom.Mesh(target)
        for face_subset in face_subsets:
            subset_material = get_or_create_material(
                face_subset["material_id"],
                face_subset["mdl_path"],
                face_subset["subidentifier"],
                face_subset["parameters"],
            )
            source_subset_rebound = face_subset.get("source_subset_rebind") is True
            if source_subset_rebound:
                subset_prim = stage.OverridePrim(
                    face_subset["source_subset_prim_path"]
                )
                geom_subset = UsdGeom.Subset(subset_prim)
            else:
                geom_subset = UsdGeom.Subset.CreateGeomSubset(
                    target_mesh,
                    face_subset["subset_name"],
                    UsdGeom.Tokens.face,
                    Vt.IntArray(face_subset["face_indices"]),
                    UsdShade.Tokens.materialBind,
                    UsdGeom.Tokens.nonOverlapping,
                )
                subset_prim = geom_subset.GetPrim()
            UsdShade.MaterialBindingAPI.Apply(subset_prim).Bind(
                subset_material,
                UsdShade.Tokens.weakerThanDescendants,
                UsdShade.Tokens.allPurpose,
            )
            subset_record: dict[str, Any] = {
                "subset_name": face_subset["subset_name"],
                "subset_prim_path": geom_subset.GetPath().pathString,
                "material_id": face_subset["material_id"],
                "mdl_path": str(face_subset["mdl_path"]),
                "subidentifier": face_subset["subidentifier"],
                "material_prim_path": subset_material.GetPath().pathString,
                "parameters": _json_material_parameters(face_subset["parameters"]),
                "face_indices": list(face_subset["face_indices"]),
                "source_subset_rebound": source_subset_rebound,
                "explicit_plan_override": bool(
                    face_subset.get("explicit_plan_override")
                ),
                "parent_assignment_propagated": bool(
                    face_subset.get("parent_assignment_propagated")
                ),
            }
            if "semantic" in face_subset:
                subset_record["semantic"] = face_subset["semantic"]
            subset_records.append(subset_record)
        record: dict[str, Any] = {
            "part_id": assignment["part_id"],
            "prim_path": part["prim_path"],
            "parent_binding_preserved": preserve_parent_binding,
            "source_visual_preserved": source_visual_preserved,
            "face_subsets": subset_records,
            "source_visual_subset_bindings": source_visual_subset_bindings,
            "existing_subset_rebind_count": existing_subset_rebind_count,
            "parent_assignment_subset_rebind_count": (
                parent_assignment_subset_rebind_count
            ),
            "source_subset_paths_rebound": [
                subset_record["subset_prim_path"]
                for subset_record in subset_records
                if subset_record["source_subset_rebound"]
            ],
            "material_bind_family_type": material_bind_family_type,
        }
        if preserve_parent_binding:
            record["source_visual_material_prim_path"] = source_visual[
                part["prim_path"]
            ]
            record["parent_binding_relationship_authored"] = False
            if source_visual_preserved:
                record["source_visual_material_binding_sha256"] = assignment[
                    "source_visual_material_binding_sha256"
                ]
        else:
            if material is None or mdl_path is None or parameters is None:
                raise RuntimeError(
                    f"Parent material was not authored: {assignment['part_id']}"
                )
            record.update(
                {
                    "material_id": assignment["material_id"],
                    "mdl_path": str(mdl_path),
                    "subidentifier": subidentifier,
                    "material_prim_path": material.GetPath().pathString,
                    "parameters": _json_material_parameters(parameters),
                    "parent_binding_relationship_authored": True,
                }
            )
        applied.append(record)

    stage.GetRootLayer().Save()

    # Reopen the composed layer and enforce the invariants that matter to physics.
    composed = Usd.Stage.Open(str(temporary_output_path), load=Usd.Stage.LoadAll)
    if composed is None:
        raise RuntimeError(f"Unable to reopen authored layer: {temporary_output_path}")
    composed_meshes = {
        prim.GetPath().pathString: prim
        for prim in composed.Traverse()
        if prim.IsA(UsdGeom.Mesh)
    }
    if set(composed_meshes) != set(source_meshes):
        raise RuntimeError("Mesh prim paths changed after material composition")
    composed_geometry_state = _mesh_geometry_state(composed, set(composed_meshes))
    if composed_geometry_state != source_geometry_state:
        changed_paths = sorted(
            path
            for path in source_geometry_state
            if composed_geometry_state.get(path) != source_geometry_state[path]
        )
        raise RuntimeError(
            "Mesh geometry or topology values changed after material composition: "
            + ", ".join(changed_paths)
        )
    if (
        _prim_invariant_state(composed, source_prim_paths, physics=True)
        != source_physics_state
    ):
        raise RuntimeError(
            "Physics schemas or properties changed after material composition"
        )
    if (
        _explicit_physics_state(composed, instance_proxies=False)
        != source_explicit_physics
    ):
        raise RuntimeError(
            "Explicit Physics/PhysX state changed after material composition"
        )
    if (
        _prim_invariant_state(composed, source_prim_paths, physics=False)
        != source_xform_state
    ):
        raise RuntimeError("Xform properties changed after material composition")

    def verify_binding_record(binding_prim, record: dict[str, Any], label: str) -> None:
        material, _ = UsdShade.MaterialBindingAPI(binding_prim).ComputeBoundMaterial(
            materialPurpose=UsdShade.Tokens.allPurpose
        )
        if (
            not material
            or material.GetPath().pathString != record["material_prim_path"]
        ):
            raise RuntimeError(
                f"Visual material binding did not resolve as authored: {label}"
            )
        shader, _source_name, _source_type = material.ComputeSurfaceSource("mdl")
        if not shader or not shader.GetPrim().IsValid():
            raise RuntimeError(f"MDL surface source is missing: {label}")
        source_asset = shader.GetSourceAsset("mdl")
        if source_asset.path != record["mdl_path"]:
            raise RuntimeError(f"MDL source asset changed: {label}")
        if shader.GetSourceAssetSubIdentifier("mdl") != record["subidentifier"]:
            raise RuntimeError(f"MDL subidentifier changed: {label}")
        for name, expected in record["parameters"].items():
            shader_input = shader.GetInput(name)
            policy_kind = _material_parameter_policy(record["material_id"])[name][0]
            expected_usd_type = {
                "color3f_linear": Sdf.ValueTypeNames.Color3f,
                "float": Sdf.ValueTypeNames.Float,
                "bool": Sdf.ValueTypeNames.Bool,
            }[policy_kind]
            if not shader_input or shader_input.GetTypeName() != expected_usd_type:
                raise RuntimeError(
                    f"MDL parameter type did not persist: {label}.{name}"
                )
            authored = shader_input.Get()
            persisted = False
            if policy_kind == "color3f_linear" and authored is not None:
                try:
                    persisted = all(
                        math.isclose(
                            float(actual),
                            float(wanted),
                            rel_tol=1e-6,
                            abs_tol=1e-7,
                        )
                        for actual, wanted in zip(authored, expected, strict=True)
                    )
                except (TypeError, ValueError):
                    persisted = False
            elif policy_kind == "float":
                persisted = (
                    authored is not None
                    and not isinstance(authored, bool)
                    and math.isclose(
                        float(authored),
                        float(expected),
                        rel_tol=1e-6,
                        abs_tol=1e-7,
                    )
                )
            elif policy_kind == "bool":
                persisted = type(authored) is bool and authored is expected
            if not persisted:
                raise RuntimeError(f"MDL parameter did not persist: {label}.{name}")

    for record in applied:
        composed_prim = composed.GetPrimAtPath(record["prim_path"])
        if record["parent_binding_preserved"]:
            authored_parent_bindings = [
                name
                for name in VISUAL_BINDING_RELATIONSHIP_NAMES
                if composed.GetRootLayer().GetPropertyAtPath(
                    Sdf.Path(record["prim_path"]).AppendProperty(name)
                )
            ]
            if authored_parent_bindings:
                raise RuntimeError(
                    "Subset-only assignment authored a parent material binding "
                    f"relationship: {record['part_id']} -> "
                    f"{authored_parent_bindings}"
                )
            if (
                _visual_binding_path(composed_prim)
                != record["source_visual_material_prim_path"]
            ):
                raise RuntimeError(
                    "Parent visual material binding changed for subset-only "
                    f"assignment: {record['part_id']}"
                )
            if (
                record.get("source_visual_preserved") is True
                and record["face_subsets"]
            ):
                raise RuntimeError(
                    "Source-visual no-op unexpectedly authored face subsets: "
                    f"{record['part_id']}"
                )
            for subset_contract in record["source_visual_subset_bindings"]:
                subset_prim = composed.GetPrimAtPath(
                    subset_contract["subset_prim_path"]
                )
                if (
                    _visual_binding_path(subset_prim)
                    != subset_contract["source_visual_material_prim_path"]
                ):
                    raise RuntimeError(
                        "Source-visual no-op changed an existing subset binding: "
                        f"{subset_contract['subset_prim_path']}"
                    )
        else:
            verify_binding_record(composed_prim, record, record["part_id"])
        subset_records = record["face_subsets"]
        if not subset_records:
            continue

        composed_mesh = UsdGeom.Mesh(composed_prim)
        actual_subsets = UsdGeom.Subset.GetGeomSubsets(
            composed_mesh,
            UsdGeom.Tokens.face,
            UsdShade.Tokens.materialBind,
        )
        expected_subset_paths = {
            subset_record["subset_prim_path"] for subset_record in subset_records
        }
        if {subset.GetPath().pathString for subset in actual_subsets} != (
            expected_subset_paths
        ):
            raise RuntimeError(
                f"materialBind subset membership changed: {record['part_id']}"
            )
        family_type = UsdGeom.Subset.GetFamilyType(
            composed_mesh, UsdShade.Tokens.materialBind
        )
        if str(family_type) != record["material_bind_family_type"]:
            raise RuntimeError(
                "materialBind family type changed: "
                f"{record['part_id']} -> {family_type}"
            )
        if record["existing_subset_rebind_count"]:
            family_type_property = Sdf.Path(record["prim_path"]).AppendProperty(
                "subsetFamily:materialBind:familyType"
            )
            if composed.GetRootLayer().GetPropertyAtPath(family_type_property):
                raise RuntimeError(
                    "Existing materialBind family type was re-authored while "
                    f"rebinding {record['part_id']}"
                )
        valid_family, validation_reason = UsdGeom.Subset.ValidateFamily(
            composed_mesh,
            UsdGeom.Tokens.face,
            UsdShade.Tokens.materialBind,
        )
        if not valid_family:
            raise RuntimeError(
                f"Invalid materialBind family for {record['part_id']}: "
                f"{validation_reason}"
            )

        for subset_record in subset_records:
            subset_prim = composed.GetPrimAtPath(subset_record["subset_prim_path"])
            geom_subset = UsdGeom.Subset(subset_prim)
            if not geom_subset or not geom_subset.GetPrim().IsValid():
                raise RuntimeError(
                    f"Authored face subset is missing: {record['part_id']}."
                    f"{subset_record['subset_name']}"
                )
            if geom_subset.GetElementTypeAttr().Get() != UsdGeom.Tokens.face:
                raise RuntimeError(
                    f"Face subset element type changed: {record['part_id']}."
                    f"{subset_record['subset_name']}"
                )
            if geom_subset.GetFamilyNameAttr().Get() != UsdShade.Tokens.materialBind:
                raise RuntimeError(
                    f"Face subset family changed: {record['part_id']}."
                    f"{subset_record['subset_name']}"
                )
            if (
                list(geom_subset.GetIndicesAttr().Get() or [])
                != subset_record["face_indices"]
            ):
                raise RuntimeError(
                    f"Face subset indices changed: {record['part_id']}."
                    f"{subset_record['subset_name']}"
                )
            if subset_record["source_subset_rebound"]:
                subset_path = Sdf.Path(subset_record["subset_prim_path"])
                authored_topology_properties = [
                    property_name
                    for property_name in ("elementType", "familyName", "indices")
                    if composed.GetRootLayer().GetPropertyAtPath(
                        subset_path.AppendProperty(property_name)
                    )
                ]
                if authored_topology_properties:
                    raise RuntimeError(
                        "Existing face-subset topology was re-authored while "
                        f"rebinding {record['part_id']}."
                        f"{subset_record['subset_name']}: "
                        f"{authored_topology_properties}"
                    )
                binding_property = subset_path.AppendProperty("material:binding")
                if not composed.GetRootLayer().GetPropertyAtPath(binding_property):
                    raise RuntimeError(
                        "Existing face subset has no direct output-layer material "
                        f"binding: {record['part_id']}."
                        f"{subset_record['subset_name']}"
                    )
                direct_targets = [
                    target.pathString
                    for target in subset_prim.GetRelationship(
                        "material:binding"
                    ).GetTargets()
                ]
                if direct_targets != [subset_record["material_prim_path"]]:
                    raise RuntimeError(
                        "Existing face-subset direct material target differs from "
                        f"the selected MDL: {record['part_id']}."
                        f"{subset_record['subset_name']}"
                    )
            verify_binding_record(
                subset_prim,
                subset_record,
                f"{record['part_id']}.{subset_record['subset_name']}",
            )

    applied_prim_paths = {record["prim_path"] for record in applied}
    for path in set(source_meshes) - applied_prim_paths:
        if _visual_binding_path(composed_meshes[path]) != source_visual[path]:
            raise RuntimeError(f"Unassigned visual material binding changed: {path}")

    # Validation happens against a sibling temporary layer.  Only a fully
    # validated file replaces the requested output path.
    composed = None
    stage = None
    temporary_output_path.replace(output_path)

    point_occurrence_count = sum(
        len(UsdGeom.Mesh(prim).GetPointsAttr().Get() or [])
        for prim in source_meshes.values()
    )
    face_occurrence_count = sum(
        len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get() or [])
        for prim in source_meshes.values()
    )
    covered_face_occurrence_count = sum(
        len(
            UsdGeom.Mesh(source_meshes[path])
            .GetFaceVertexCountsAttr()
            .Get()
            or []
        )
        for path in applied_prim_paths
    )
    return {
        "output_usd": str(output_path),
        "source_usd": str(asset_path),
        "plan_sha256": _canonical_sha256(plan),
        "registry_sha256": _canonical_sha256(registry_document),
        "mesh_occurrence_count": len(source_meshes),
        "point_occurrence_count": point_occurrence_count,
        "face_occurrence_count": face_occurrence_count,
        "covered_face_occurrence_count": covered_face_occurrence_count,
        "applied_count": len(applied),
        "policy_fallback_count": policy_fallback_count,
        "parent_binding_preserved_count": sum(
            bool(record["parent_binding_preserved"]) for record in applied
        ),
        "source_visual_preserve_count": sum(
            record["source_visual_preserved"] is True for record in applied
        ),
        "source_visual_preserved_subset_count": sum(
            len(record["source_visual_subset_bindings"]) for record in applied
        ),
        "existing_subset_rebind_count": sum(
            record["existing_subset_rebind_count"] for record in applied
        ),
        "parent_assignment_subset_rebind_count": sum(
            record["parent_assignment_subset_rebind_count"] for record in applied
        ),
        "planned_face_subset_override_count": sum(
            bool(subset["explicit_plan_override"])
            for record in applied
            for subset in record["face_subsets"]
        ),
        "verified_subset_binding_count": sum(
            len(record["face_subsets"]) for record in applied
        ),
        "face_subset_count": sum(len(record["face_subsets"]) for record in applied),
        "skipped": skipped,
        "applied": applied,
        "validation": {
            "mesh_path_count": len(source_meshes),
            "mesh_geometry_and_topology_values_unchanged": True,
            "xforms_unchanged": True,
            "physics_properties_unchanged": True,
            "physics_bindings_unchanged": True,
            "visual_bindings_resolve": True,
            "subset_only_parent_visual_bindings_unchanged": True,
            "subset_only_parent_binding_relationships_absent": True,
            "source_visual_preserve_contracts_verified": True,
            "unassigned_visual_bindings_unchanged": True,
            "mdl_sources_and_parameters_verified": True,
            "selected_mdl_lock_verified": selection_lock_verified,
            "face_subsets_verified": True,
            "existing_face_subset_topology_not_reauthored": True,
            "existing_face_subset_bindings_directly_authored": True,
            "whole_mesh_parent_material_propagated_to_existing_subsets": True,
            "source_subset_visual_bindings_verified": True,
            "policy_fallback_explicitly_authorized": bool(
                policy_fallback_count and include_policy_fallback
            ),
        },
        "source_sha256": _sha256_file(asset_path),
        "output_sha256": _sha256_file(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a USD look layer from a validated Qwen material plan"
    )
    parser.add_argument("--asset-usd", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--material-root", required=True)
    parser.add_argument("--include-review", action="store_true")
    parser.add_argument("--include-policy-fallback", action="store_true")
    parser.add_argument(
        "--selection-lock",
        help="Hash-bound immutable selected-MDL contract",
    )
    parser.add_argument("--report", help="Optional JSON report path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = _start_isaac_if_needed(headless=True)
    exit_code = 0
    try:
        report = apply_visual_materials(
            asset_usd=args.asset_usd,
            catalog_path=args.catalog,
            registry_path=args.registry,
            plan_path=args.plan,
            output_usd=args.output,
            material_root=args.material_root,
            include_review=args.include_review,
            include_policy_fallback=args.include_policy_fallback,
            selection_lock_path=args.selection_lock,
        )
        if args.report:
            report_path = Path(args.report).expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        if app is not None:
            app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
