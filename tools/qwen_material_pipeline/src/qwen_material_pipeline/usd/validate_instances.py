#!/usr/bin/env python3
"""Validate a collected look bundle produced from an instance-heavy USD.

The immutable source is inspected with ``Usd.TraverseInstanceProxies`` while
the collected look stage is inspected as an ordinary, de-instanced stage.
This distinction is important for CAD assemblies: normal source traversal can
silently report zero occurrence Meshes even though hundreds are rendered.

The validator is read-only and fail-closed.  It binds the source, occurrence
registry, apply report, collected USD, MDL modules, and runtime textures into a
single machine-readable report.
"""

from __future__ import annotations

import argparse
import json
import traceback
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from qwen_material_pipeline.usd.apply_instances import (
    SCHEMA_VERSION as APPLY_SCHEMA_VERSION,
)
from qwen_material_pipeline.usd.material_common import sha256_file
from qwen_material_pipeline.usd.stage_state import (
    canonical_sha256,
    explicit_physics_state,
    geometry_state,
    material_binding_path,
    matrix_states_close,
    mesh_prims,
    source_layer_is_in_stack,
    subset_state,
    world_matrix_state,
)
from qwen_material_pipeline.usd.validation_common import (
    CHECK_LABELS,
    Audit,
    collect_mapping,
    is_inside,
    load_json_object,
    report_records,
    start_isaac_if_needed,
    verify_materials,
    verify_mdl_textures,
    verify_usd_dependencies,
)


SCHEMA_VERSION = "qwen-instance-final-bundle-validation/v1"
LOOKS_SCOPE_NAME = "QwenInstanceLooks"


def _is_at_or_below(path: str | None, root: str) -> bool:
    """Return whether a USD path is the root itself or one of its descendants."""

    return isinstance(path, str) and (path == root or path.startswith(root + "/"))


def _atomic_write_json(document: Mapping[str, Any], output: str | Path) -> Path:
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return output_path


def _registry_maps(
    registry: Mapping[str, Any], audit: Audit
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    raw_parts = registry.get("parts")
    if not isinstance(raw_parts, list):
        audit.fail("registry", "Occurrence registry must contain a parts list")
        return {}, {}
    by_id: dict[str, Mapping[str, Any]] = {}
    by_path: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_parts):
        if not isinstance(raw, Mapping):
            audit.fail(
                "registry",
                "Occurrence registry part must be an object",
                context={"index": index},
            )
            continue
        part_id = raw.get("part_id")
        prim_path = raw.get("prim_path")
        if not isinstance(part_id, str) or not isinstance(prim_path, str):
            audit.fail(
                "registry",
                "Occurrence registry part requires string part_id and prim_path",
                context={"index": index},
            )
            continue
        if part_id in by_id or prim_path in by_path:
            audit.fail(
                "registry",
                "Occurrence registry has a duplicate part id or Mesh path",
                context={"part_id": part_id, "prim_path": prim_path},
            )
            continue
        by_id[part_id] = raw
        by_path[prim_path] = raw
    return by_id, by_path


def _verify_registry_asset(
    *,
    registry: Mapping[str, Any],
    apply_report: Mapping[str, Any],
    source_path: Path,
    source_stage: Any,
    source_meshes: Mapping[str, Any],
    audit: Audit,
) -> tuple[Path | None, Any | None]:
    from pxr import Usd

    raw_path = registry.get("asset_usd")
    if not isinstance(raw_path, str) or not raw_path:
        audit.fail("inputs", "Occurrence registry asset_usd is invalid")
        return None, None
    registry_asset = Path(raw_path).expanduser().resolve(strict=False)
    if not registry_asset.is_file():
        audit.fail(
            "inputs",
            "Occurrence registry asset is missing",
            context={"path": str(registry_asset)},
        )
        return registry_asset, None

    actual_hash = sha256_file(registry_asset)
    audit.require(
        "inputs",
        registry.get("asset_sha256") == actual_hash,
        "Occurrence registry asset SHA-256 is stale",
        context={"expected": registry.get("asset_sha256"), "actual": actual_hash},
    )
    audit.require(
        "inputs",
        apply_report.get("occurrence_registry_asset_sha256") == actual_hash,
        "Apply report occurrence-registry asset SHA-256 differs",
        context={
            "expected": apply_report.get("occurrence_registry_asset_sha256"),
            "actual": actual_hash,
        },
    )

    stage = Usd.Stage.Open(str(registry_asset), load=Usd.Stage.LoadAll)
    if stage is None:
        audit.fail(
            "inputs",
            "Unable to open occurrence-registry asset",
            context={"path": str(registry_asset)},
        )
        return registry_asset, None
    default_prim = stage.GetDefaultPrim()
    source_default = source_stage.GetDefaultPrim()
    audit.require(
        "inputs",
        bool(default_prim)
        and bool(source_default)
        and default_prim.GetPath() == source_default.GetPath(),
        "Occurrence-registry asset default prim differs from the source",
    )
    if default_prim and default_prim.IsValid():
        audit.require(
            "inputs",
            source_layer_is_in_stack(default_prim, source_path),
            "Occurrence-registry asset does not compose the immutable source",
        )

    registry_meshes = mesh_prims(stage, instance_proxies=False)
    audit.require(
        "registry",
        set(registry_meshes) == set(source_meshes),
        "Occurrence-registry asset Mesh paths differ from source occurrences",
        context={
            "missing": sorted(set(source_meshes) - set(registry_meshes)),
            "unexpected": sorted(set(registry_meshes) - set(source_meshes)),
        },
    )
    audit.require(
        "geometry",
        geometry_state(stage, registry_meshes)
        == geometry_state(source_stage, source_meshes),
        "Occurrence-registry asset geometry/topology differs from the source",
    )
    audit.require(
        "transforms",
        matrix_states_close(
            world_matrix_state(stage, registry_meshes),
            world_matrix_state(source_stage, source_meshes),
        ),
        "Occurrence-registry asset world transforms differ from the source",
    )
    audit.require(
        "geom_subsets",
        subset_state(stage, registry_meshes)
        == subset_state(source_stage, source_meshes),
        "Occurrence-registry asset GeomSubsets differ from the source",
    )
    audit.require(
        "physics",
        explicit_physics_state(stage, instance_proxies=False)
        == explicit_physics_state(source_stage, instance_proxies=True),
        "Occurrence-registry asset Physics/PhysX state differs from the source",
    )
    return registry_asset, stage


def _verify_declared_mdl_dependencies(
    *,
    apply_report: Mapping[str, Any],
    collect_mapping: Mapping[str, Path],
    bundle_root: Path,
    audit: Audit,
) -> None:
    raw_dependencies = apply_report.get("mdl_dependencies")
    if not isinstance(raw_dependencies, list):
        audit.fail("mdl_materials", "Apply report has no mdl_dependencies list")
        return
    mapped_targets: set[Path] = set()
    for index, raw in enumerate(raw_dependencies):
        if not isinstance(raw, Mapping):
            audit.fail(
                "mdl_materials",
                "MDL dependency record must be an object",
                context={"index": index},
            )
            continue
        source = raw.get("source_path")
        if not isinstance(source, str):
            audit.fail(
                "mdl_materials",
                "MDL dependency source_path must be a string",
                context={"index": index},
            )
            continue
        source_path = Path(source).expanduser().resolve(strict=False)
        audit.require(
            "mdl_materials",
            source_path.is_file(),
            "Original MDL dependency is missing",
            context={"path": str(source_path)},
        )
        if source_path.is_file():
            audit.require(
                "mdl_materials",
                raw.get("sha256") == sha256_file(source_path),
                "Apply-report MDL dependency SHA-256 is stale",
                context={"path": str(source_path)},
            )
        target = collect_mapping.get(str(source_path))
        audit.require(
            "mdl_materials",
            target is not None,
            "Declared MDL dependency is absent from the collect mapping",
            context={"source": str(source_path)},
        )
        if target is None:
            continue
        target = target.resolve(strict=False)
        mapped_targets.add(target)
        audit.require(
            "mdl_materials",
            target.is_file() and target.suffix.casefold() == ".mdl",
            "Collected MDL target is missing or has the wrong extension",
            context={"target": str(target)},
        )
        audit.require(
            "mdl_materials",
            is_inside(target, bundle_root),
            "Collected MDL target escapes the bundle",
            context={"target": str(target)},
        )

    bundle_mdls = {path.resolve() for path in bundle_root.rglob("*.mdl")}
    audit.require(
        "mdl_materials",
        mapped_targets == bundle_mdls,
        "Bundle MDL files do not exactly match declared application dependencies",
        context={
            "unmapped_bundle_mdls": sorted(
                str(path) for path in bundle_mdls - mapped_targets
            ),
            "mapped_targets_not_in_bundle_scan": sorted(
                str(path) for path in mapped_targets - bundle_mdls
            ),
        },
    )
    audit.metric(
        "mdl_materials", "declared_mdl_dependency_count", len(raw_dependencies)
    )
    audit.metric("mdl_materials", "collected_mdl_file_count", len(bundle_mdls))


def validate_instance_bundle(
    *,
    source_usd: str | Path,
    collected_root_usd: str | Path,
    registry_path: str | Path,
    apply_report_path: str | Path,
    bundle_root: str | Path,
) -> dict[str, Any]:
    """Validate one collected instance-aware material bundle without mutation."""

    from pxr import Usd, UsdGeom

    source_path = Path(source_usd).expanduser().resolve(strict=True)
    collected_path = Path(collected_root_usd).expanduser().resolve(strict=True)
    registry_file = Path(registry_path).expanduser().resolve(strict=True)
    apply_file = Path(apply_report_path).expanduser().resolve(strict=True)
    bundle_path = Path(bundle_root).expanduser().resolve(strict=True)
    if not bundle_path.is_dir():
        raise ValueError(f"Bundle root is not a directory: {bundle_path}")

    audit = Audit()
    source_hash_before = sha256_file(source_path)
    registry = load_json_object(registry_file)
    apply_report = load_json_object(apply_file)

    audit.require(
        "inputs",
        is_inside(collected_path, bundle_path),
        "Collected root USD is outside the declared bundle",
        context={"collected": str(collected_path), "bundle": str(bundle_path)},
    )
    audit.require(
        "inputs",
        apply_report.get("schema_version") == APPLY_SCHEMA_VERSION,
        "Apply report schema version is unsupported",
        context={"actual": apply_report.get("schema_version")},
    )
    audit.require(
        "inputs",
        apply_report.get("source_sha256") == source_hash_before,
        "Apply report source SHA-256 differs from the immutable source",
        context={
            "reported": apply_report.get("source_sha256"),
            "actual": source_hash_before,
        },
    )
    report_source = apply_report.get("source_usd")
    audit.require(
        "inputs",
        isinstance(report_source, str)
        and Path(report_source).expanduser().resolve(strict=False) == source_path,
        "Apply report source path does not identify the validated source",
        context={"reported": report_source, "actual": str(source_path)},
    )
    registry_digest = canonical_sha256(registry)
    audit.require(
        "inputs",
        apply_report.get("occurrence_registry_sha256") == registry_digest,
        "Apply report canonical occurrence-registry hash differs",
        context={
            "reported": apply_report.get("occurrence_registry_sha256"),
            "actual": registry_digest,
        },
    )
    provenance = apply_report.get("plan_provenance")
    audit.require(
        "inputs",
        isinstance(provenance, Mapping)
        and provenance.get("asset_sha256") == source_hash_before
        and provenance.get("registry_sha256") == registry_digest,
        "Material-plan provenance does not bind this source and registry",
    )

    source_stage = Usd.Stage.Open(str(source_path), load=Usd.Stage.LoadAll)
    collected_stage = Usd.Stage.Open(str(collected_path), load=Usd.Stage.LoadAll)
    if source_stage is None or collected_stage is None:
        raise RuntimeError("Unable to open source or collected USD stage")
    source_default = source_stage.GetDefaultPrim()
    collected_default = collected_stage.GetDefaultPrim()
    if not source_default or not collected_default:
        raise ValueError("Source and collected stages require valid default prims")
    audit.require(
        "inputs",
        source_default.GetPath() == collected_default.GetPath(),
        "Collected default prim differs from the source",
    )

    source_meshes = mesh_prims(source_stage, instance_proxies=True)
    collected_meshes = mesh_prims(collected_stage, instance_proxies=False)
    registry_by_id, registry_by_path = _registry_maps(registry, audit)
    registered_paths = set(registry_by_path)
    source_paths = set(source_meshes)
    collected_paths = set(collected_meshes)
    audit.require(
        "registry",
        registered_paths == source_paths,
        "Occurrence registry does not exactly cover source instance Mesh paths",
        context={
            "unregistered_source": sorted(source_paths - registered_paths),
            "registry_not_source": sorted(registered_paths - source_paths),
        },
    )
    audit.require(
        "registry",
        collected_paths == source_paths,
        "Collected Mesh occurrence paths differ from the source",
        context={
            "missing_collected": sorted(source_paths - collected_paths),
            "unexpected_collected": sorted(collected_paths - source_paths),
        },
    )
    audit.require(
        "registry",
        registry.get("part_count") == len(registry_by_id) == len(source_meshes),
        "Registry part_count differs from unique Mesh occurrences",
        context={
            "declared": registry.get("part_count"),
            "valid_parts": len(registry_by_id),
            "source_meshes": len(source_meshes),
        },
    )
    for name in ("mesh_occurrence_count", "applied_count"):
        audit.require(
            "inputs",
            apply_report.get(name) == len(source_meshes),
            f"Apply report {name} differs from source occurrence count",
            context={"reported": apply_report.get(name), "actual": len(source_meshes)},
        )
    source_instance_count = sum(prim.IsInstance() for prim in source_stage.Traverse())
    collected_instance_count = sum(
        prim.IsInstance() for prim in collected_stage.Traverse()
    )
    audit.require(
        "registry",
        source_instance_count > 0
        and apply_report.get("deinstanced_prim_count") == source_instance_count,
        "Apply report de-instancing count differs from the source",
        context={
            "reported": apply_report.get("deinstanced_prim_count"),
            "actual": source_instance_count,
        },
    )
    audit.require(
        "registry",
        collected_instance_count == 0,
        "Collected look stage still contains composed instances",
        context={"remaining": collected_instance_count},
    )
    audit.metric("registry", "source_instance_count", source_instance_count)
    audit.metric("registry", "source_mesh_occurrence_count", len(source_meshes))
    audit.metric("registry", "collected_mesh_occurrence_count", len(collected_meshes))
    audit.metric(
        "registry", "remaining_collected_instance_count", collected_instance_count
    )

    point_count = 0
    face_count = 0
    for path, source_prim in source_meshes.items():
        mesh = UsdGeom.Mesh(source_prim)
        points = len(mesh.GetPointsAttr().Get() or [])
        faces = len(mesh.GetFaceVertexCountsAttr().Get() or [])
        point_count += points
        face_count += faces
        part = registry_by_path.get(path)
        if part is not None:
            audit.require(
                "registry",
                part.get("point_count") == points and part.get("face_count") == faces,
                "Occurrence registry point/face count differs from source Mesh",
                context={
                    "prim_path": path,
                    "registry_points": part.get("point_count"),
                    "source_points": points,
                    "registry_faces": part.get("face_count"),
                    "source_faces": faces,
                },
            )
    audit.require(
        "geometry",
        geometry_state(source_stage, source_meshes)
        == geometry_state(collected_stage, collected_meshes),
        "Collected Mesh points/topology differ from source occurrences",
    )
    audit.require(
        "transforms",
        matrix_states_close(
            world_matrix_state(source_stage, source_meshes),
            world_matrix_state(collected_stage, collected_meshes),
        ),
        "Collected Mesh world transforms differ from source occurrences",
    )
    audit.require(
        "geometry",
        apply_report.get("point_occurrence_count") == point_count
        and apply_report.get("face_occurrence_count") == face_count,
        "Apply-report geometry totals differ from source occurrences",
        context={
            "reported_points": apply_report.get("point_occurrence_count"),
            "actual_points": point_count,
            "reported_faces": apply_report.get("face_occurrence_count"),
            "actual_faces": face_count,
        },
    )
    audit.metric("geometry", "mesh_occurrence_count", len(source_meshes))
    audit.metric("geometry", "point_occurrence_count", point_count)
    audit.metric("geometry", "face_occurrence_count", face_count)
    audit.metric(
        "geometry", "compared_geometry_attribute_count", len(source_meshes) * 6
    )
    audit.metric("transforms", "world_matrix_comparison_count", len(source_meshes))

    source_subsets = subset_state(source_stage, source_meshes)
    collected_subsets = subset_state(collected_stage, collected_meshes)
    reported_face_subsets: dict[str, tuple[str, Mapping[str, Any]]] = {}
    reported_preserved_subsets: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for raw_record in apply_report.get("applied", []):
        if not isinstance(raw_record, Mapping):
            continue
        mesh_path = raw_record.get("prim_path")
        if not isinstance(mesh_path, str):
            continue
        for field, destination in (
            ("face_subsets", reported_face_subsets),
            ("source_visual_subset_bindings", reported_preserved_subsets),
        ):
            for raw_subset in raw_record.get(field, []):
                if not isinstance(raw_subset, Mapping):
                    continue
                subset_path = raw_subset.get("subset_prim_path")
                if isinstance(subset_path, str):
                    destination[subset_path] = (mesh_path, raw_subset)
    authored_subset_paths = set(reported_face_subsets) - set(source_subsets)
    expected_collected_subset_paths = set(source_subsets) | authored_subset_paths
    source_subsets_unchanged = all(
        collected_subsets.get(path) == state
        for path, state in source_subsets.items()
    )
    authored_subsets_match = all(
        path in collected_subsets
        and collected_subsets[path]["mesh_path"] == mesh_path
        and list(collected_subsets[path]["indices"])
        == subset_record.get("face_indices")
        for path, (mesh_path, subset_record) in reported_face_subsets.items()
        if path in authored_subset_paths
    )
    audit.require(
        "geom_subsets",
        set(collected_subsets) == expected_collected_subset_paths
        and source_subsets_unchanged
        and authored_subsets_match,
        "Collected GeomSubsets differ from preserved source plus planned subsets",
        context={
            "missing": sorted(set(source_subsets) - set(collected_subsets)),
            "unexpected": sorted(
                set(collected_subsets) - expected_collected_subset_paths
            ),
            "changed": sorted(
                path
                for path in set(source_subsets) & set(collected_subsets)
                if source_subsets[path] != collected_subsets[path]
            ),
        },
    )
    audit.require(
        "geom_subsets",
        apply_report.get("source_subset_occurrence_count") == len(source_subsets)
        and apply_report.get("authored_face_subset_count", 0)
        == len(authored_subset_paths)
        and (
            apply_report.get("face_subset_count", 0)
            + apply_report.get("source_visual_preserved_subset_count", 0)
            == len(collected_subsets)
        ),
        "Apply-report GeomSubset counts differ from collected occurrences",
        context={
            "source_subsets": len(source_subsets),
            "authored_subsets": len(authored_subset_paths),
            "reported_source_subsets": apply_report.get(
                "source_subset_occurrence_count"
            ),
            "reported_authored_subsets": apply_report.get(
                "authored_face_subset_count"
            ),
            "reported_planned_subsets": apply_report.get("face_subset_count"),
            "reported_source_preserved_subsets": apply_report.get(
                "source_visual_preserved_subset_count"
            ),
        },
    )
    audit.metric("geom_subsets", "source_subset_occurrence_count", len(source_subsets))
    audit.metric(
        "geom_subsets", "authored_face_subset_count", len(authored_subset_paths)
    )
    audit.metric(
        "geom_subsets",
        "source_subset_face_occurrence_count",
        sum(len(item["indices"]) for item in source_subsets.values()),
    )

    source_physics = explicit_physics_state(source_stage, instance_proxies=True)
    collected_physics = explicit_physics_state(collected_stage, instance_proxies=False)
    audit.require(
        "physics",
        source_physics == collected_physics,
        "Explicit Physics/PhysX state differs from the source",
        context={
            "source_paths": sorted(source_physics),
            "collected_paths": sorted(collected_physics),
        },
    )
    audit.require(
        "physics",
        apply_report.get("validation", {}).get("explicit_physics_prim_count_before")
        == len(source_physics)
        and apply_report.get("validation", {}).get("explicit_physics_prim_count_after")
        == len(collected_physics),
        "Apply-report explicit-physics counts differ",
    )
    audit.metric("physics", "source_explicit_physics_prim_count", len(source_physics))
    audit.metric(
        "physics", "collected_explicit_physics_prim_count", len(collected_physics)
    )

    _verify_registry_asset(
        registry=registry,
        apply_report=apply_report,
        source_path=source_path,
        source_stage=source_stage,
        source_meshes=source_meshes,
        audit=audit,
    )

    applied_by_path, expected_materials = report_records(
        dict(apply_report), dict(registry_by_id), audit
    )
    audit.require(
        "visual_bindings",
        set(applied_by_path) == source_paths,
        "Apply-report records do not exactly cover Mesh occurrences",
        context={
            "missing": sorted(source_paths - set(applied_by_path)),
            "unexpected": sorted(set(applied_by_path) - source_paths),
        },
    )
    look_root = source_default.GetPath().AppendChild(LOOKS_SCOPE_NAME).pathString
    covered_faces = 0
    verified_mesh_bindings = 0
    verified_subset_bindings = 0
    verified_subset_index_records = 0
    outside_look_bindings: list[dict[str, Any]] = []
    for path, prim in collected_meshes.items():
        record = applied_by_path.get(path)
        binding = material_binding_path(prim)
        expected = record.get("material_prim_path") if record else None
        direct_targets = [
            target.pathString
            for target in prim.GetRelationship("material:binding").GetTargets()
        ]
        source_visual_preserved = bool(
            record and record.get("source_visual_preserved") is True
        )
        if source_visual_preserved:
            source_prim = source_meshes[path]
            source_binding = material_binding_path(source_prim)
            source_direct_targets = [
                target.pathString
                for target in source_prim.GetRelationship(
                    "material:binding"
                ).GetTargets()
            ]
            expected = record.get("source_visual_material_prim_path")
            mesh_valid = (
                record.get("parent_binding_preserved") is True
                and isinstance(expected, str)
                and expected == source_binding
                and binding == source_binding
                and direct_targets == source_direct_targets
            )
        else:
            mesh_valid = (
                record is not None
                and record.get("parent_binding_preserved") is False
                and binding == expected
                and direct_targets == [expected]
                and _is_at_or_below(binding, look_root)
            )
        audit.require(
            "visual_bindings",
            mesh_valid,
            "Mesh does not have the exact new-look binding from the apply report",
            context={
                "prim_path": path,
                "expected": expected,
                "computed": binding,
                "direct_targets": direct_targets,
            },
        )
        if not source_visual_preserved and not _is_at_or_below(binding, look_root):
            outside_look_bindings.append({"prim_path": path, "binding": binding})
        subset_valid = True
        expected_subsets = {
            item.get("subset_prim_path"): item
            for item in (record or {}).get("face_subsets", [])
            if isinstance(item, Mapping)
        }
        expected_source_subsets = {
            item.get("subset_prim_path"): item
            for item in (record or {}).get("source_visual_subset_bindings", [])
            if isinstance(item, Mapping)
        }
        expected_all_subset_paths = set(expected_subsets) | set(
            expected_source_subsets
        )
        collected_mesh_subset_paths = {
            subset_path
            for subset_path, subset in collected_subsets.items()
            if subset["mesh_path"] == path
        }
        audit.require(
            "geom_subsets",
            expected_all_subset_paths == collected_mesh_subset_paths,
            "Apply-report subset paths do not exactly cover collected GeomSubsets",
            context={
                "mesh_path": path,
                "missing": sorted(
                    collected_mesh_subset_paths - expected_all_subset_paths
                ),
                "unexpected": sorted(
                    expected_all_subset_paths - collected_mesh_subset_paths
                ),
            },
        )
        for subset_path, collected_subset in collected_subsets.items():
            if collected_subset["mesh_path"] != path:
                continue
            subset_prim = collected_stage.GetPrimAtPath(subset_path)
            subset_binding = material_binding_path(subset_prim)
            subset_record = expected_subsets.get(subset_path)
            source_subset_record = expected_source_subsets.get(subset_path)
            effective_subset_record = subset_record or source_subset_record
            source_subset = source_subsets.get(subset_path)
            expected_indices = (
                list(source_subset["indices"])
                if source_subset is not None
                else (
                    subset_record.get("face_indices")
                    if subset_record is not None
                    else None
                )
            )
            indices_match = (
                subset_record is not None
                and subset_record.get("face_indices") == expected_indices
                and list(collected_subset["indices"]) == expected_indices
            )
            if source_subset_record is not None:
                indices_match = (
                    source_subset is not None
                    and source_subset_record.get("face_indices") == expected_indices
                    and list(collected_subset["indices"]) == expected_indices
                )
            audit.require(
                "geom_subsets",
                indices_match,
                "Apply-report GeomSubset face indices differ from source or plan",
                context={
                    "subset_prim_path": subset_path,
                    "expected": expected_indices,
                    "reported": (
                        effective_subset_record.get("face_indices")
                        if effective_subset_record
                        else None
                    ),
                },
            )
            if source_subset_record is not None:
                if source_subset is None:
                    audit.fail(
                        "geom_subsets",
                        "A source-preserved subset is absent from the source",
                        context={"subset_prim_path": subset_path},
                    )
                    subset_valid = False
                    continue
                source_subset_prim = source_stage.GetPrimAtPath(subset_path)
                source_subset_binding = material_binding_path(source_subset_prim)
                source_subset_direct_targets = [
                    target.pathString
                    for target in source_subset_prim.GetRelationship(
                        "material:binding"
                    ).GetTargets()
                ]
                collected_subset_direct_targets = [
                    target.pathString
                    for target in subset_prim.GetRelationship(
                        "material:binding"
                    ).GetTargets()
                ]
                subset_expected = source_subset_record.get(
                    "source_visual_material_prim_path"
                )
                current_valid = (
                    indices_match
                    and subset_expected == source_subset_binding
                    and subset_binding == source_subset_binding
                    and collected_subset_direct_targets
                    == source_subset_direct_targets
                )
            else:
                subset_expected = (
                    subset_record.get("material_prim_path")
                    if subset_record
                    else expected
                )
                current_valid = (
                    indices_match
                    and subset_binding == subset_expected
                    and _is_at_or_below(subset_binding, look_root)
                )
            subset_valid = subset_valid and current_valid
            audit.require(
                "visual_bindings",
                current_valid,
                "Source GeomSubset does not have its exact new-look binding",
                context={
                    "subset_prim_path": subset_path,
                    "expected": subset_expected,
                    "actual": subset_binding,
                },
            )
            if source_subset_record is None and not _is_at_or_below(
                subset_binding, look_root
            ):
                outside_look_bindings.append(
                    {"prim_path": subset_path, "binding": subset_binding}
                )
            if current_valid:
                verified_subset_bindings += 1
            if indices_match:
                verified_subset_index_records += 1
        if mesh_valid:
            verified_mesh_bindings += 1
        if mesh_valid and subset_valid:
            covered_faces += len(
                UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get() or []
            )

    audit.require(
        "visual_bindings",
        not outside_look_bindings,
        "One or more final visual bindings escape the new look scope",
        context={"bindings": outside_look_bindings},
    )
    audit.require(
        "visual_bindings",
        covered_faces == face_count
        and apply_report.get("covered_face_occurrence_count") == face_count,
        "Not every source face occurrence is covered by a verified new look",
        context={
            "source_faces": face_count,
            "verified_covered_faces": covered_faces,
            "reported_covered_faces": apply_report.get("covered_face_occurrence_count"),
        },
    )
    for material_path in expected_materials:
        audit.require(
            "visual_bindings",
            _is_at_or_below(material_path, look_root),
            "An authored material prim escapes the new look scope",
            context={"material_prim_path": material_path, "look_root": look_root},
        )
    audit.metric("visual_bindings", "look_scope", look_root)
    audit.metric(
        "visual_bindings", "verified_mesh_binding_count", verified_mesh_bindings
    )
    audit.metric(
        "visual_bindings", "verified_subset_binding_count", verified_subset_bindings
    )
    audit.metric("visual_bindings", "covered_face_occurrence_count", covered_faces)
    audit.metric(
        "geom_subsets", "actual_material_face_subset_count", len(collected_subsets)
    )
    audit.metric(
        "geom_subsets",
        "verified_face_subset_count",
        verified_subset_index_records,
    )

    collected_mapping = collect_mapping(bundle_path, audit)
    source_target = collected_mapping.get(str(source_path))
    audit.require(
        "usd_dependencies",
        source_target is not None,
        "Immutable source is absent from the collect mapping",
        context={"source": str(source_path)},
    )
    if source_target is not None:
        source_target = source_target.resolve(strict=False)
        layer_paths = {
            Path(layer.realPath).resolve(strict=False)
            for layer in collected_stage.GetUsedLayers()
            if layer.realPath
        }
        audit.require(
            "usd_dependencies",
            source_target.is_file()
            and is_inside(source_target, bundle_path)
            and source_target in layer_paths,
            "Collected source target is missing, outside the bundle, or uncomposed",
            context={"target": str(source_target)},
        )

    verify_materials(
        collected_stage, expected_materials, collected_mapping, bundle_path, audit
    )
    _verify_declared_mdl_dependencies(
        apply_report=apply_report,
        collect_mapping=collected_mapping,
        bundle_root=bundle_path,
        audit=audit,
    )
    verify_usd_dependencies(collected_path, bundle_path, audit)
    verify_mdl_textures(bundle_path, audit)

    source_hash_after = sha256_file(source_path)
    audit.require(
        "inputs",
        source_hash_after == source_hash_before,
        "Immutable source USD changed during validation",
        context={"before": source_hash_before, "after": source_hash_after},
    )
    audit.metric("inputs", "source_unchanged", source_hash_after == source_hash_before)

    inputs = {
        "source_usd": str(source_path),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "collected_root_usd": str(collected_path),
        "collected_root_sha256": sha256_file(collected_path),
        "registry": str(registry_file),
        "registry_file_sha256": sha256_file(registry_file),
        "registry_canonical_sha256": registry_digest,
        "apply_report": str(apply_file),
        "apply_report_sha256": sha256_file(apply_file),
        "bundle_root": str(bundle_path),
    }
    report = audit.to_report(inputs)
    report["schema_version"] = SCHEMA_VERSION
    report["verified_contract"] = {
        "instance_aware_source_traversal": True,
        "mesh_occurrence_count": len(source_meshes),
        "point_occurrence_count": point_count,
        "face_occurrence_count": face_count,
        "covered_face_occurrence_count": covered_faces,
        "source_subset_occurrence_count": len(source_subsets),
        "source_subset_face_occurrence_count": sum(
            len(item["indices"]) for item in source_subsets.values()
        ),
        "explicit_physics_prim_count": len(source_physics),
        "new_look_scope": look_root,
        "authored_material_prim_count": len(expected_materials),
        "declared_mdl_dependency_count": audit.checks["mdl_materials"].metrics.get(
            "declared_mdl_dependency_count", 0
        ),
        "runtime_texture_reference_count": audit.checks["mdl_textures"].metrics.get(
            "runtime_texture_reference_count", 0
        ),
        "usd_asset_dependency_count": audit.checks["usd_dependencies"].metrics.get(
            "usd_asset_dependency_count", 0
        ),
        "source_unchanged": source_hash_after == source_hash_before,
    }
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a portable material bundle against an instance-heavy source USD"
        )
    )
    parser.add_argument("--source-usd", required=True)
    parser.add_argument("--collected-root-usd", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--apply-report", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    app = None
    try:
        app = start_isaac_if_needed()
        report = validate_instance_bundle(
            source_usd=args.source_usd,
            collected_root_usd=args.collected_root_usd,
            registry_path=args.registry,
            apply_report_path=args.apply_report,
            bundle_root=args.bundle_root,
        )
        output = _atomic_write_json(report, args.output)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "overall_pass": report["overall_pass"],
                    "output": str(output),
                    "summary": report["summary"],
                    "verified_contract": report["verified_contract"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0 if report["overall_pass"] else 1
    except Exception as exc:
        traceback.print_exc()
        fatal_report = {
            "schema_version": SCHEMA_VERSION,
            "status": "FAIL",
            "overall_pass": False,
            "inputs": {
                "source_usd": str(Path(args.source_usd).expanduser()),
                "collected_root_usd": str(Path(args.collected_root_usd).expanduser()),
                "registry": str(Path(args.registry).expanduser()),
                "apply_report": str(Path(args.apply_report).expanduser()),
                "bundle_root": str(Path(args.bundle_root).expanduser()),
            },
            "summary": {
                "check_count": len(CHECK_LABELS),
                "passed_check_count": 0,
                "failed_check_count": len(CHECK_LABELS),
                "failure_count": 1,
                "warning_count": 0,
            },
            "checks": [],
            "warnings": [],
            "fatal_error": {"type": type(exc).__name__, "message": str(exc)},
        }
        try:
            output = _atomic_write_json(fatal_report, args.output)
            print(
                json.dumps(
                    {"status": "FAIL", "overall_pass": False, "output": str(output)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception:
            traceback.print_exc()
        return 2
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
