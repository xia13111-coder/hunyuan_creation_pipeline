#!/usr/bin/env python3
"""Apply a complete material plan to an instance-heavy USD assembly.

The source USD is never edited.  The output layer references it, disables
instancing only at the composed instance roots, and binds reviewed MDL looks
at every occurrence Mesh path.  Existing GeomSubset topology is preserved.
Hash-bound face-recovery plans may also author new non-overlapping material
subsets in the look layer, without changing source geometry or topology.

This command is a trust boundary.  A plan is accepted only when its provenance
binds both the immutable source USD and the canonical JSON content of the
occurrence registry, and when it covers every registered occurrence exactly
once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import traceback
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from qwen_material_pipeline.materials.selection_lock import (
    validate_material_selection_lock as _validate_material_selection_lock,
)
from qwen_material_pipeline.usd.material_common import (
    APPLY_STATUSES,
    POLICY_FALLBACK_STATUS,
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
    canonical_sha256 as _canonical_sha256,
    explicit_physics_state as _explicit_physics_state,
    geometry_state as _geometry_state,
    material_binding_path as _material_binding_path,
    matrix_states_close as _matrix_states_close,
    mesh_prims as _mesh_prims,
    source_layer_is_in_stack as _source_layer_is_in_stack,
    subset_state as _subset_state,
    world_matrix_state as _world_matrix_state,
)


SCHEMA_VERSION = "qwen-instance-material-application/v1"


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
    with resolved.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {resolved}")
    return value


def _validate_plan_provenance(
    plan: Mapping[str, Any],
    *,
    source_sha256: str,
    registry_sha256: str,
) -> dict[str, str]:
    provenance = plan.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Material plan must contain a provenance object")
    asset_digest = provenance.get("asset_sha256")
    registry_digest = provenance.get("registry_sha256")
    if asset_digest != source_sha256:
        raise ValueError("Plan provenance asset_sha256 does not match the source USD")
    if registry_digest != registry_sha256:
        raise ValueError(
            "Plan provenance registry_sha256 does not match canonical registry JSON"
        )
    return {
        "asset_sha256": source_sha256,
        "registry_sha256": registry_sha256,
    }


def _registry_map(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    parts = registry.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("Occurrence registry must contain a non-empty parts list")
    result: dict[str, Mapping[str, Any]] = {}
    paths: set[str] = set()
    for index, raw in enumerate(parts):
        if not isinstance(raw, Mapping):
            raise ValueError(f"registry.parts[{index}] must be an object")
        part_id = raw.get("part_id")
        prim_path = raw.get("prim_path")
        if not isinstance(part_id, str) or not part_id:
            raise ValueError(f"registry.parts[{index}].part_id is invalid")
        if not isinstance(prim_path, str) or not prim_path.startswith("/"):
            raise ValueError(f"registry.parts[{index}].prim_path is invalid")
        if part_id in result:
            raise ValueError(f"Occurrence registry has duplicate part_id: {part_id}")
        if prim_path in paths:
            raise ValueError(
                f"Occurrence registry has duplicate prim_path: {prim_path}"
            )
        result[part_id] = raw
        paths.add(prim_path)
    if registry.get("part_count") != len(result):
        raise ValueError("Occurrence registry part_count does not match parts")
    return result


def _validate_confidence(
    part_id: str, assignment: Mapping[str, Any], status: str
) -> None:
    confidence = assignment.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError(f"Invalid confidence for part {part_id}: {confidence!r}")
    if status == "auto" and float(confidence) < 0.85:
        raise ValueError(f"Auto assignment below 0.85 confidence: {part_id}")
    if status == "review" and float(confidence) < 0.60:
        raise ValueError(f"Review assignment below 0.60 confidence: {part_id}")


def _verify_material_record(stage, record: Mapping[str, Any]) -> None:
    from pxr import Sdf

    material_prim = stage.GetPrimAtPath(record["material_prim_path"])
    if not material_prim or not material_prim.IsValid():
        raise RuntimeError(
            f"Authored material is missing: {record['material_prim_path']}"
        )

    from pxr import UsdShade

    material = UsdShade.Material(material_prim)
    shader, _source_name, _source_type = material.ComputeSurfaceSource("mdl")
    if not shader or not shader.GetPrim().IsValid():
        raise RuntimeError(f"MDL surface source is missing: {record['material_id']}")
    source_asset = shader.GetSourceAsset("mdl")
    if source_asset.path != record["mdl_path"]:
        raise RuntimeError(f"MDL source asset changed: {record['material_id']}")
    if shader.GetSourceAssetSubIdentifier("mdl") != record["subidentifier"]:
        raise RuntimeError(f"MDL subidentifier changed: {record['material_id']}")

    expected_parameters = record["parameters"]
    actual_parameter_names = sorted(
        shader_input.GetBaseName() for shader_input in shader.GetInputs()
    )
    if actual_parameter_names != sorted(expected_parameters):
        raise RuntimeError(f"MDL parameter set changed: {record['material_id']}")
    for name, expected in expected_parameters.items():
        shader_input = shader.GetInput(name)
        kind = _material_parameter_policy(record["material_id"])[name][0]
        expected_type = {
            "color3f_linear": Sdf.ValueTypeNames.Color3f,
            "float": Sdf.ValueTypeNames.Float,
            "bool": Sdf.ValueTypeNames.Bool,
        }[kind]
        if not shader_input or shader_input.GetTypeName() != expected_type:
            raise RuntimeError(
                f"MDL parameter type did not persist: {record['material_id']}.{name}"
            )
        actual = shader_input.Get()
        if kind == "color3f_linear":
            valid = actual is not None and all(
                math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-7)
                for a, b in zip(actual, expected, strict=True)
            )
        elif kind == "float":
            valid = actual is not None and math.isclose(
                float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-7
            )
        else:
            valid = type(actual) is bool and actual is expected
        if not valid:
            raise RuntimeError(
                f"MDL parameter did not persist: {record['material_id']}.{name}"
            )


def apply_instance_materials(
    *,
    source_usd: str | Path,
    catalog_path: str | Path,
    registry_path: str | Path,
    plan_path: str | Path,
    output_usd: str | Path,
    material_root: str | Path,
    include_review: bool = False,
    include_policy_fallback: bool = False,
    selection_lock_path: str | Path | None = None,
) -> dict[str, Any]:
    """Author and validate a complete occurrence-level look layer."""

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    source_path = Path(source_usd).expanduser().resolve(strict=True)
    output_path = Path(output_usd).expanduser().resolve()
    if output_path == source_path:
        raise ValueError("Output layer must not overwrite the source USD")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    material_root_path = Path(material_root).expanduser().resolve(strict=True)

    source_sha256_before = _sha256_file(source_path)
    catalog = _catalog_map(_load_json(catalog_path))
    registry_document = _load_json(registry_path)
    registry_sha256 = _canonical_sha256(registry_document)
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
            material_root=material_root_path,
        )
        selection_lock_verified = True
    provenance = _validate_plan_provenance(
        plan,
        source_sha256=source_sha256_before,
        registry_sha256=registry_sha256,
    )

    registry_asset_value = registry_document.get("asset_usd")
    if not isinstance(registry_asset_value, str) or not registry_asset_value:
        raise ValueError("Occurrence registry asset_usd is invalid")
    registry_asset_path = Path(registry_asset_value).expanduser().resolve(strict=True)
    registry_asset_sha256 = registry_document.get("asset_sha256")
    if registry_asset_sha256 != _sha256_file(registry_asset_path):
        raise ValueError("Occurrence registry asset_sha256 does not match its asset")
    policy_fallback_count = _validate_policy_fallback_authorization(
        plan,
        registry_document,
        include_policy_fallback=include_policy_fallback,
    )

    source_stage = Usd.Stage.Open(str(source_path), load=Usd.Stage.LoadAll)
    registry_stage = Usd.Stage.Open(str(registry_asset_path), load=Usd.Stage.LoadAll)
    if source_stage is None or registry_stage is None:
        raise RuntimeError("Unable to open source or occurrence-registry stage")
    source_default = source_stage.GetDefaultPrim()
    registry_default = registry_stage.GetDefaultPrim()
    if not source_default or not source_default.IsValid():
        raise ValueError("Source stage has no valid default prim")
    if not registry_default or not registry_default.IsValid():
        raise ValueError("Occurrence-registry stage has no valid default prim")
    if registry_default.GetPath() != source_default.GetPath():
        raise ValueError("Occurrence-registry stage default prim differs from source")
    if not _source_layer_is_in_stack(registry_default, source_path):
        raise ValueError("Occurrence-registry stage does not compose the source USD")

    source_meshes = _mesh_prims(source_stage, instance_proxies=True)
    registry_meshes = _mesh_prims(registry_stage, instance_proxies=False)
    registered_paths = {
        str(part["prim_path"]): part_id for part_id, part in registry.items()
    }
    if set(source_meshes) != set(registry_meshes):
        raise ValueError(
            "Occurrence-registry stage Mesh paths do not match source instance proxies"
        )
    if set(registered_paths) != set(source_meshes):
        raise ValueError("Occurrence registry does not exactly cover source Mesh paths")

    source_geometry = _geometry_state(source_stage, source_meshes)
    source_world_matrices = _world_matrix_state(source_stage, source_meshes)
    if _geometry_state(registry_stage, registry_meshes) != source_geometry:
        raise ValueError(
            "Occurrence-registry stage geometry/topology differs from source"
        )
    if not _matrix_states_close(
        _world_matrix_state(registry_stage, registry_meshes), source_world_matrices
    ):
        raise ValueError(
            "Occurrence-registry stage world transforms differ from source"
        )

    point_occurrence_count = 0
    face_occurrence_count = 0
    for path, source_prim in source_meshes.items():
        mesh = UsdGeom.Mesh(source_prim)
        point_count = len(mesh.GetPointsAttr().Get() or [])
        face_count = len(mesh.GetFaceVertexCountsAttr().Get() or [])
        point_occurrence_count += point_count
        face_occurrence_count += face_count
        part = registry[registered_paths[path]]
        if (
            part.get("point_count") != point_count
            or part.get("face_count") != face_count
        ):
            raise ValueError(
                "Occurrence registry point/face counts differ from source: "
                f"{registered_paths[path]}"
            )

    source_subsets = _subset_state(source_stage, source_meshes)
    source_subset_visual_bindings = {
        path: _material_binding_path(source_stage.GetPrimAtPath(path))
        for path in source_subsets
    }
    if _subset_state(registry_stage, registry_meshes) != source_subsets:
        raise ValueError("Occurrence-registry stage GeomSubsets differ from source")
    source_physics = _explicit_physics_state(source_stage, instance_proxies=True)
    instance_paths = sorted(
        prim.GetPath().pathString
        for prim in source_stage.Traverse()
        if prim.IsInstance()
    )
    if not instance_paths:
        raise ValueError("Source stage has no composed instance roots")

    assignments = plan.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("Material plan must contain an assignments list")
    allowed_statuses = set(APPLY_STATUSES)
    if include_review:
        allowed_statuses.add("review")
    if policy_fallback_count:
        allowed_statuses.add(POLICY_FALLBACK_STATUS)
    validated: dict[str, dict[str, Any]] = {}
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, Mapping):
            raise ValueError(f"assignments[{index}] must be an object")
        part_id = assignment.get("part_id")
        if not isinstance(part_id, str) or part_id not in registry:
            raise ValueError(f"Unknown part_id: {part_id!r}")
        if part_id in validated:
            raise ValueError(f"Duplicate assignment for part_id: {part_id}")
        status = assignment.get("status", "review")
        if status not in allowed_statuses:
            raise ValueError(
                f"Complete plan assignment is not applicable: {part_id} status={status}"
            )
        _validate_confidence(part_id, assignment, str(status))
        material_id = assignment.get("material_id")
        prim_path = str(registry[part_id]["prim_path"])
        face_count = int(registry[part_id]["face_count"])
        source_preserve_path = _validate_source_visual_preserve(
            part_id, assignment, registry[part_id]
        )
        source_visual_preserved = source_preserve_path is not None
        if source_visual_preserved:
            actual_source_binding = _material_binding_path(source_meshes[prim_path])
            if actual_source_binding != source_preserve_path:
                raise ValueError(
                    "Source visual binding differs from the hash-bound preserve "
                    f"contract: {part_id}"
                )
            mdl_path = None
            subidentifier = None
            parameters = None
            face_subsets = []
        else:
            if not isinstance(material_id, str) or material_id not in catalog:
                raise ValueError(f"Unknown material_id: {material_id!r}")
            catalog_item = catalog[material_id]
            mdl_path = _resolve_mdl_path(catalog_item, material_root_path)
            subidentifier = _get_subidentifier(catalog_item)
            parameters = _normalize_material_parameters(
                material_id, assignment.get("parameters")
            )
            face_subsets = _normalize_face_subsets(
                part_id,
                assignment.get("face_subsets"),
                allowed_material_ids=set(catalog),
                face_count=face_count,
            )
        existing_subset_paths = sorted(
            path
            for path, subset in source_subsets.items()
            if subset["mesh_path"] == prim_path
        )
        planned_subset_paths: set[str] = set()
        authored_subset_paths: set[str] = set()
        for face_subset in face_subsets:
            subset_path = (
                Sdf.Path(prim_path).AppendChild(face_subset["subset_name"]).pathString
            )
            source_subset = source_subsets.get(subset_path)
            if source_subset is not None:
                source_subset_prim = source_stage.GetPrimAtPath(subset_path)
                source_subset_schema = UsdGeom.Subset(source_subset_prim)
                if (
                    source_subset_schema.GetElementTypeAttr().Get()
                    != UsdGeom.Tokens.face
                    or source_subset_schema.GetFamilyNameAttr().Get()
                    != UsdShade.Tokens.materialBind
                ):
                    raise ValueError(
                        "Planned source subset is not a materialBind face subset: "
                        f"{subset_path}"
                    )
                if tuple(face_subset["face_indices"]) != source_subset["indices"]:
                    raise ValueError(
                        "Planned face indices differ from the source GeomSubset: "
                        f"{part_id}.{face_subset['subset_name']}"
                    )
            else:
                if source_stage.GetPrimAtPath(subset_path).IsValid():
                    raise ValueError(
                        "Planned face subset collides with an existing non-subset "
                        f"source prim: {part_id}.{face_subset['subset_name']}"
                    )
                authored_subset_paths.add(subset_path)
            subset_catalog_item = catalog[face_subset["material_id"]]
            face_subset["subset_prim_path"] = subset_path
            face_subset["mdl_path"] = _resolve_mdl_path(
                subset_catalog_item, material_root_path
            )
            face_subset["subidentifier"] = _get_subidentifier(subset_catalog_item)
            planned_subset_paths.add(subset_path)
        if authored_subset_paths and existing_subset_paths:
            raise ValueError(
                "New planned face subsets require a source Mesh without existing "
                f"materialBind subsets: {part_id}"
            )
        preserve_parent = source_visual_preserved or _preserve_parent_material_binding(
            part_id,
            dict(assignment),
            has_face_subsets=bool(face_subsets),
        )
        if (
            preserve_parent
            and not source_visual_preserved
            and not authored_subset_paths
        ):
            covered_faces = {
                face_index
                for face_subset in face_subsets
                for face_index in face_subset["face_indices"]
            }
            if planned_subset_paths != set(existing_subset_paths):
                raise ValueError(
                    "preserve_parent_material_binding requires every existing "
                    f"source subset to be planned: {part_id}"
                )
            if covered_faces != set(range(face_count)):
                raise ValueError(
                    "preserve_parent_material_binding requires exact full-face "
                    f"subset coverage: {part_id}"
                )
        validated[part_id] = {
            "part_id": part_id,
            "prim_path": prim_path,
            "material_id": material_id,
            "mdl_path": mdl_path,
            "subidentifier": subidentifier,
            "parameters": parameters,
            "face_subsets": face_subsets,
            "existing_subset_paths": existing_subset_paths,
            "authored_subset_paths": sorted(authored_subset_paths),
            "preserve_parent_material_binding": preserve_parent,
            "source_visual_preserved": source_visual_preserved,
            "source_visual_material_prim_path": _material_binding_path(
                source_meshes[prim_path]
            ),
            "source_visual_material_binding_sha256": assignment.get(
                "source_visual_material_binding_sha256"
            ),
        }
    if set(validated) != set(registry):
        missing = sorted(set(registry) - set(validated))
        unexpected = sorted(set(validated) - set(registry))
        raise ValueError(
            "Complete plan does not exactly cover occurrence registry; "
            f"missing={missing}, unexpected={unexpected}"
        )

    temporary = output_path.with_name(
        f".{output_path.stem}.tmp-{os.getpid()}{output_path.suffix}"
    )
    temporary.unlink(missing_ok=True)
    try:
        stage = Usd.Stage.CreateNew(str(temporary))
        if stage is None:
            raise RuntimeError(f"Unable to create output stage: {temporary}")
        UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source_stage))
        UsdGeom.SetStageMetersPerUnit(
            stage, UsdGeom.GetStageMetersPerUnit(source_stage)
        )
        stage.SetTimeCodesPerSecond(source_stage.GetTimeCodesPerSecond())
        stage.SetFramesPerSecond(source_stage.GetFramesPerSecond())
        stage.SetStartTimeCode(source_stage.GetStartTimeCode())
        stage.SetEndTimeCode(source_stage.GetEndTimeCode())

        default_path = source_default.GetPath()
        root_prim = stage.DefinePrim(
            default_path, source_default.GetTypeName() or "Xform"
        )
        reference_path = os.path.relpath(source_path, start=temporary.parent)
        root_prim.GetReferences().AddReference(reference_path, default_path)
        stage.SetDefaultPrim(root_prim)
        for path in instance_paths:
            stage.OverridePrim(path).SetInstanceable(False)

        looks_path = default_path.AppendChild("QwenInstanceLooks")
        UsdGeom.Scope.Define(stage, looks_path)
        authored_materials: dict[str, Any] = {}
        material_records: dict[str, dict[str, Any]] = {}

        def get_or_create_material(spec: Mapping[str, Any]):
            material_id = str(spec["material_id"])
            parameters = spec["parameters"]
            instance_key = _material_instance_key(material_id, parameters)
            if instance_key in authored_materials:
                return authored_materials[instance_key]
            suffix = (
                "_" + hashlib.sha256(instance_key.encode("utf-8")).hexdigest()[:10]
                if parameters
                else ""
            )
            material_path = looks_path.AppendChild(
                _safe_child_name(material_id) + suffix
            )
            material = UsdShade.Material.Define(stage, material_path)
            shader = UsdShade.Shader.Define(stage, material_path.AppendChild("Shader"))
            shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
            shader.SetSourceAsset(Sdf.AssetPath(str(spec["mdl_path"])), "mdl")
            shader.SetSourceAssetSubIdentifier(str(spec["subidentifier"]), "mdl")
            parameter_policy = _material_parameter_policy(material_id)
            for name, value in sorted(parameters.items()):
                kind = parameter_policy[name][0]
                if kind == "color3f_linear":
                    shader.CreateInput(name, Sdf.ValueTypeNames.Color3f).Set(
                        Gf.Vec3f(*value)
                    )
                elif kind == "float":
                    shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(value)
                elif kind == "bool":
                    shader.CreateInput(name, Sdf.ValueTypeNames.Bool).Set(value)
                else:
                    raise RuntimeError(f"Unsupported MDL parameter policy: {name}")
            shader_output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
            shader_output.SetRenderType("material")
            connectable = shader.ConnectableAPI()
            material.CreateSurfaceOutput("mdl").ConnectToSource(connectable, "out")
            material.CreateVolumeOutput("mdl").ConnectToSource(connectable, "out")
            material.CreateDisplacementOutput("mdl").ConnectToSource(connectable, "out")
            authored_materials[instance_key] = material
            material_records[instance_key] = {
                "material_id": material_id,
                "material_prim_path": material_path.pathString,
                "mdl_path": str(spec["mdl_path"]),
                "mdl_sha256": _sha256_file(Path(spec["mdl_path"])),
                "subidentifier": str(spec["subidentifier"]),
                "parameters": _json_material_parameters(parameters),
            }
            return material

        applied: list[dict[str, Any]] = []
        for part_id in sorted(validated):
            spec = validated[part_id]
            mesh_prim = stage.OverridePrim(spec["prim_path"])
            preserve_parent = bool(spec["preserve_parent_material_binding"])
            parent_material = None if preserve_parent else get_or_create_material(spec)
            if not preserve_parent:
                if parent_material is None:
                    raise RuntimeError(f"Parent material is missing: {part_id}")
                UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(
                    parent_material,
                    UsdShade.Tokens.weakerThanDescendants,
                    UsdShade.Tokens.allPurpose,
                )
            planned_subsets = {
                subset["subset_prim_path"]: subset for subset in spec["face_subsets"]
            }
            subset_records: list[dict[str, Any]] = []
            source_visual_subset_records: list[dict[str, Any]] = []
            for subset_path in spec["existing_subset_paths"]:
                if spec["source_visual_preserved"]:
                    subset_binding = source_subset_visual_bindings[subset_path]
                    if subset_binding is None:
                        raise ValueError(
                            "Source-visual preserve requires every existing "
                            f"materialBind subset to resolve: {subset_path}"
                        )
                    source_visual_subset_records.append(
                        {
                            "subset_prim_path": subset_path,
                            "source_visual_material_prim_path": subset_binding,
                            "face_indices": list(
                                source_subsets[subset_path]["indices"]
                            ),
                        }
                    )
                    continue
                subset_spec = planned_subsets.get(subset_path)
                subset_material = (
                    get_or_create_material(subset_spec)
                    if subset_spec is not None
                    else parent_material
                )
                if subset_material is None:
                    raise RuntimeError(
                        f"Preserved parent has an unplanned source subset: {subset_path}"
                    )
                subset_prim = stage.OverridePrim(subset_path)
                UsdShade.MaterialBindingAPI.Apply(subset_prim).Bind(
                    subset_material,
                    UsdShade.Tokens.weakerThanDescendants,
                    UsdShade.Tokens.allPurpose,
                )
                effective_spec = subset_spec or spec
                subset_record: dict[str, Any] = {
                    "subset_name": Path(subset_path).name,
                    "subset_prim_path": subset_path,
                    "material_id": effective_spec["material_id"],
                    "material_prim_path": subset_material.GetPath().pathString,
                    "mdl_path": str(effective_spec["mdl_path"]),
                    "subidentifier": effective_spec["subidentifier"],
                    "parameters": _json_material_parameters(
                        effective_spec["parameters"]
                    ),
                    "face_indices": list(source_subsets[subset_path]["indices"]),
                    "explicit_plan_override": subset_spec is not None,
                }
                if subset_spec is not None and "semantic" in subset_spec:
                    subset_record["semantic"] = subset_spec["semantic"]
                subset_records.append(subset_record)
            for subset_path in spec["authored_subset_paths"]:
                subset_spec = planned_subsets[subset_path]
                subset_material = get_or_create_material(subset_spec)
                target_mesh = UsdGeom.Mesh(mesh_prim)
                geom_subset = UsdGeom.Subset.CreateGeomSubset(
                    target_mesh,
                    subset_spec["subset_name"],
                    UsdGeom.Tokens.face,
                    Vt.IntArray(subset_spec["face_indices"]),
                    UsdShade.Tokens.materialBind,
                    UsdGeom.Tokens.nonOverlapping,
                )
                UsdShade.MaterialBindingAPI.Apply(geom_subset.GetPrim()).Bind(
                    subset_material,
                    UsdShade.Tokens.weakerThanDescendants,
                    UsdShade.Tokens.allPurpose,
                )
                subset_record = {
                    "subset_name": subset_spec["subset_name"],
                    "subset_prim_path": subset_path,
                    "material_id": subset_spec["material_id"],
                    "material_prim_path": subset_material.GetPath().pathString,
                    "mdl_path": str(subset_spec["mdl_path"]),
                    "subidentifier": subset_spec["subidentifier"],
                    "parameters": _json_material_parameters(
                        subset_spec["parameters"]
                    ),
                    "face_indices": list(subset_spec["face_indices"]),
                    "explicit_plan_override": True,
                }
                if "semantic" in subset_spec:
                    subset_record["semantic"] = subset_spec["semantic"]
                subset_records.append(subset_record)
            record: dict[str, Any] = {
                "part_id": part_id,
                "prim_path": spec["prim_path"],
                "parent_binding_preserved": preserve_parent,
                "source_visual_preserved": bool(spec["source_visual_preserved"]),
                "face_subsets": subset_records,
                "source_visual_subset_bindings": source_visual_subset_records,
                "source_subset_paths_rebound": (
                    []
                    if spec["source_visual_preserved"]
                    else list(spec["existing_subset_paths"])
                ),
                "face_count": int(registry[part_id]["face_count"]),
            }
            if preserve_parent:
                record.update(
                    {
                        "source_visual_material_prim_path": spec[
                            "source_visual_material_prim_path"
                        ],
                        "parent_binding_relationship_authored": False,
                    }
                )
                if spec["source_visual_preserved"]:
                    record["source_visual_material_binding_sha256"] = spec[
                        "source_visual_material_binding_sha256"
                    ]
            else:
                if parent_material is None:
                    raise RuntimeError(f"Parent material is missing: {part_id}")
                record.update(
                    {
                        "material_id": spec["material_id"],
                        "material_prim_path": parent_material.GetPath().pathString,
                        "mdl_path": str(spec["mdl_path"]),
                        "subidentifier": spec["subidentifier"],
                        "parameters": _json_material_parameters(spec["parameters"]),
                        "parent_binding_relationship_authored": True,
                    }
                )
            applied.append(record)
        stage.GetRootLayer().Save()
        stage = None

        composed = Usd.Stage.Open(str(temporary), load=Usd.Stage.LoadAll)
        if composed is None:
            raise RuntimeError(f"Unable to reopen authored layer: {temporary}")
        output_meshes = _mesh_prims(composed, instance_proxies=False)
        if set(output_meshes) != set(source_meshes):
            raise RuntimeError(
                "Mesh occurrence paths changed after material composition"
            )
        if _geometry_state(composed, output_meshes) != source_geometry:
            raise RuntimeError(
                "Mesh geometry or topology changed after material composition"
            )
        if not _matrix_states_close(
            _world_matrix_state(composed, output_meshes), source_world_matrices
        ):
            raise RuntimeError(
                "Mesh world transforms changed after material composition"
            )
        output_subsets = _subset_state(composed, output_meshes)
        authored_subset_specs = {
            subset["subset_prim_path"]: {
                "mesh_path": spec["prim_path"],
                "face_indices": tuple(subset["face_indices"]),
            }
            for spec in validated.values()
            for subset in spec["face_subsets"]
            if subset["subset_prim_path"] in spec["authored_subset_paths"]
        }
        if set(output_subsets) != set(source_subsets) | set(authored_subset_specs):
            raise RuntimeError(
                "Output GeomSubsets differ from preserved source plus planned subsets"
            )
        if any(output_subsets[path] != state for path, state in source_subsets.items()):
            raise RuntimeError("Source GeomSubset topology or indices changed")
        for subset_path, expected_subset in authored_subset_specs.items():
            actual_subset = output_subsets[subset_path]
            if (
                actual_subset["mesh_path"] != expected_subset["mesh_path"]
                or actual_subset["element_type"] != repr(UsdGeom.Tokens.face)
                or actual_subset["family_name"]
                != repr(UsdShade.Tokens.materialBind)
                or actual_subset["indices"] != expected_subset["face_indices"]
            ):
                raise RuntimeError(
                    f"Authored face subset differs from its plan: {subset_path}"
                )
        output_physics = _explicit_physics_state(composed, instance_proxies=False)
        if output_physics != source_physics:
            raise RuntimeError("Explicit Physics/PhysX opinions changed")
        remaining_instances = sum(prim.IsInstance() for prim in composed.Traverse())
        if remaining_instances:
            raise RuntimeError(
                f"Output still contains {remaining_instances} composed instances"
            )

        applied_by_path = {str(record["prim_path"]): record for record in applied}
        covered_face_count = 0
        verified_subset_binding_count = 0
        for path, prim in output_meshes.items():
            record = applied_by_path[path]
            if record["parent_binding_preserved"]:
                expected_parent_material = record["source_visual_material_prim_path"]
                if _material_binding_path(prim) != expected_parent_material:
                    raise RuntimeError(
                        f"Preserved parent visual binding changed: {path}"
                    )
                for relationship_name in (
                    "material:binding",
                    "material:binding:preview",
                    "material:binding:full",
                ):
                    property_path = Sdf.Path(path).AppendProperty(relationship_name)
                    if composed.GetRootLayer().GetPropertyAtPath(property_path):
                        raise RuntimeError(
                            "Preserved parent acquired a look-layer binding: "
                            f"{property_path}"
                        )
                for subset_contract in record["source_visual_subset_bindings"]:
                    subset_path = subset_contract["subset_prim_path"]
                    subset_prim = composed.GetPrimAtPath(subset_path)
                    expected_subset_binding = subset_contract[
                        "source_visual_material_prim_path"
                    ]
                    if (
                        _material_binding_path(subset_prim)
                        != expected_subset_binding
                    ):
                        raise RuntimeError(
                            "Preserved source subset visual binding changed: "
                            f"{subset_path}"
                        )
                    for relationship_name in (
                        "material:binding",
                        "material:binding:preview",
                        "material:binding:full",
                    ):
                        property_path = Sdf.Path(subset_path).AppendProperty(
                            relationship_name
                        )
                        if composed.GetRootLayer().GetPropertyAtPath(property_path):
                            raise RuntimeError(
                                "Preserved source subset acquired a look-layer "
                                f"binding: {property_path}"
                            )
            else:
                expected_parent_material = record["material_prim_path"]
                if _material_binding_path(prim) != expected_parent_material:
                    raise RuntimeError(
                        f"Visual binding did not resolve as authored: {path}"
                    )
                direct_targets = [
                    target.pathString
                    for target in prim.GetRelationship("material:binding").GetTargets()
                ]
                if direct_targets != [expected_parent_material]:
                    raise RuntimeError(f"Direct visual binding is missing: {path}")
            covered_face_count += int(record["face_count"])
            for subset_record in record["face_subsets"]:
                subset_path = subset_record["subset_prim_path"]
                expected_subset_material = subset_record["material_prim_path"]
                subset_prim = composed.GetPrimAtPath(subset_path)
                if _material_binding_path(subset_prim) != expected_subset_material:
                    raise RuntimeError(
                        f"Source subset visual binding did not resolve: {subset_path}"
                    )
                direct_subset_targets = [
                    target.pathString
                    for target in subset_prim.GetRelationship(
                        "material:binding"
                    ).GetTargets()
                ]
                if direct_subset_targets != [expected_subset_material]:
                    raise RuntimeError(
                        f"Source subset direct visual binding is missing: {subset_path}"
                    )
                verified_subset_binding_count += 1
        if covered_face_count != face_occurrence_count:
            raise RuntimeError("Not all occurrence faces have a new visual material")

        for material_record in material_records.values():
            _verify_material_record(composed, material_record)

        source_sha256_after = _sha256_file(source_path)
        registry_asset_sha256_after = _sha256_file(registry_asset_path)
        if source_sha256_after != source_sha256_before:
            raise RuntimeError("Source USD changed during material application")
        if registry_asset_sha256_after != registry_asset_sha256:
            raise RuntimeError("Occurrence-registry stage changed during application")

        material_dependencies = sorted(
            {
                (
                    str(record["mdl_path"]),
                    str(record["mdl_sha256"]),
                    os.path.relpath(record["mdl_path"], material_root_path),
                )
                for record in material_records.values()
            }
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "source_usd": str(source_path),
            "source_sha256": source_sha256_before,
            "occurrence_registry": str(Path(registry_path).expanduser().resolve()),
            "occurrence_registry_asset": str(registry_asset_path),
            "occurrence_registry_asset_sha256": registry_asset_sha256,
            "occurrence_registry_sha256": registry_sha256,
            "plan": str(Path(plan_path).expanduser().resolve()),
            "plan_sha256": _canonical_sha256(plan),
            "plan_provenance": provenance,
            "output_usd": str(output_path),
            "deinstanced_prim_count": len(instance_paths),
            "mesh_occurrence_count": len(output_meshes),
            "point_occurrence_count": point_occurrence_count,
            "face_occurrence_count": face_occurrence_count,
            "covered_face_occurrence_count": covered_face_count,
            "source_subset_occurrence_count": len(source_subsets),
            "verified_subset_binding_count": verified_subset_binding_count,
            "applied_count": len(applied),
            "policy_fallback_count": policy_fallback_count,
            "parent_binding_preserved_count": sum(
                bool(record["parent_binding_preserved"]) for record in applied
            ),
            "source_visual_preserve_count": sum(
                record["source_visual_preserved"] is True for record in applied
            ),
            "face_subset_count": sum(len(record["face_subsets"]) for record in applied),
            "authored_face_subset_count": len(authored_subset_specs),
            "source_visual_preserved_subset_count": sum(
                len(record["source_visual_subset_bindings"]) for record in applied
            ),
            "planned_face_subset_override_count": sum(
                bool(subset["explicit_plan_override"])
                for record in applied
                for subset in record["face_subsets"]
            ),
            "authored_material_count": len(material_records),
            "applied": applied,
            "materials": sorted(
                material_records.values(),
                key=lambda item: str(item["material_prim_path"]),
            ),
            "mdl_dependencies": [
                {
                    "source_path": path,
                    "sha256": digest,
                    "relative_to_material_root": relative,
                }
                for path, digest, relative in material_dependencies
            ],
            "validation": {
                "registry_composes_source": True,
                "registry_canonical_hash_verified": True,
                "plan_provenance_verified": True,
                "complete_plan_exact_cover": True,
                "source_unchanged": True,
                "registry_asset_unchanged": True,
                "mesh_paths_unchanged": True,
                "points_and_topology_unchanged": True,
                "world_transforms_unchanged": True,
                "source_subset_indices_unchanged": True,
                "planned_face_subsets_verified": True,
                "explicit_physics_opinions_unchanged": True,
                "explicit_physics_prim_count_before": len(source_physics),
                "explicit_physics_prim_count_after": len(output_physics),
                "all_occurrence_faces_have_new_visual_material": True,
                "source_subset_visual_bindings_replaced": not any(
                    record["source_visual_preserved"] is True for record in applied
                ),
                "source_subset_visual_bindings_verified": True,
                "source_visual_preserve_contracts_verified": True,
                "mdl_sources_subidentifiers_parameters_verified": True,
                "selected_mdl_lock_verified": selection_lock_verified,
                "remaining_instance_count": 0,
                "policy_fallback_explicitly_authorized": bool(
                    policy_fallback_count and include_policy_fallback
                ),
            },
        }
        composed = None
        temporary.replace(output_path)
        report["output_sha256"] = _sha256_file(output_path)
        return report
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-usd", required=True)
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
    parser.add_argument("--report")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    app = _start_isaac_if_needed(headless=True)
    exit_code = 0
    try:
        report = apply_instance_materials(
            source_usd=args.source_usd,
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
            temporary = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
            temporary.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(report_path)
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
