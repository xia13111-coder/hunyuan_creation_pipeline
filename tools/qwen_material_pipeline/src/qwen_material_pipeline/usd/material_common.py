"""Shared, model-independent material-plan validation helpers.

Both regular and instance-aware USD material application use this module as
their public contract.  The functions intentionally avoid importing ``pxr``
so plans and catalogs can be validated before an Isaac Sim process is started.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..materials.tuning import parameter_policy_for_material


APPLY_STATUSES = {"auto", "approved"}
POLICY_FALLBACK_STATUS = "policy_fallback"
POLICY_EXACT_COVER_MODE = "explicit_best_effort_policy_exact_cover"
POLICY_FALLBACK_CONFIDENCE_BASIS = "policy fallback; not evidence confidence"
SOURCE_VISUAL_PRESERVE_ACTION = "source_visual_preserve"
SOURCE_VISUAL_PRESERVE_TIER = "source_visual_preserve"
_SHA256 = re.compile(r"[0-9a-f]{64}")

# Policy tuple: (kind, inclusive_minimum, inclusive_maximum).  Boolean
# parameters do not have numeric bounds.
ParameterPolicy = tuple[str, float | None, float | None]
NormalizedParameter = tuple[float, float, float] | float | bool

PARAMETER_POLICIES: dict[str, dict[str, ParameterPolicy]] = {
    "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted": {
        "paint_color": ("color3f_linear", 0.0, 1.0),
        "paint_roughness": ("float", 0.0, 1.0),
        "paint_roughness_variation": ("float", 0.0, 1.0),
        "dirt_weight": ("float", 0.0, 1.0),
        "wash_weight": ("float", 0.0, 1.0),
        "paint_stroke_normal_strength": ("float", 0.0, 1.0),
        "uneven_normal_strength": ("float", 0.0, 1.0),
        "enable_rust_damage": ("bool", None, None),
    },
}


def material_parameter_policy(material_id: str) -> dict[str, ParameterPolicy]:
    """Return the bounded policy for one exact NVIDIA MDL export.

    The legacy generic ``Steel_Painted`` interface retains its additional
    reviewed controls.  Other supported presets receive only the minimal
    colour/PBR delta declared by :mod:`materials.tuning`.
    """

    policy = dict(parameter_policy_for_material(material_id))
    policy.update(PARAMETER_POLICIES.get(material_id, {}))
    return policy

SAFE_SUBSET_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    """Return the canonical JSON digest used by policy-plan provenance."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_visual_binding_sha256(
    *,
    part_id: str,
    prim_path: str,
    material_prim_path: str,
) -> str:
    """Hash one immutable source-binding identity used by preserve-only plans."""

    return canonical_sha256(
        {
            "part_id": part_id,
            "prim_path": prim_path,
            "source_visual_material_prim_path": material_prim_path,
        }
    )


def validate_source_visual_preserve(
    part_id: str,
    assignment: Mapping[str, Any],
    registry_part: Mapping[str, Any],
) -> str | None:
    """Validate and return an explicit source-visual no-op binding.

    ``None`` means the assignment is a normal material-authoring action.  The
    preserve action is deliberately hash-bound to the exact registry part and
    forbids any simultaneous material/subset mutation.
    """

    action = assignment.get("apply_action")
    if action is None:
        return None
    if action != SOURCE_VISUAL_PRESERVE_ACTION:
        raise ValueError(f"Unknown apply_action for {part_id}: {action!r}")
    if assignment.get("status") != POLICY_FALLBACK_STATUS:
        raise ValueError(
            f"{SOURCE_VISUAL_PRESERVE_ACTION} for {part_id} requires "
            f"status={POLICY_FALLBACK_STATUS}"
        )
    provenance = assignment.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("tier") != SOURCE_VISUAL_PRESERVE_TIER
    ):
        raise ValueError(
            f"{SOURCE_VISUAL_PRESERVE_ACTION} for {part_id} requires the "
            f"{SOURCE_VISUAL_PRESERVE_TIER} provenance tier"
        )
    prim_path = registry_part.get("prim_path")
    source_path = registry_part.get("existing_visual_material")
    declared_path = assignment.get("source_visual_material_prim_path")
    if (
        not isinstance(prim_path, str)
        or not prim_path.startswith("/")
        or not isinstance(source_path, str)
        or not source_path.startswith("/")
        or source_path == "/"
        or source_path.endswith("/")
        or "//" in source_path
        or any(character.isspace() for character in source_path)
    ):
        raise ValueError(
            f"{SOURCE_VISUAL_PRESERVE_ACTION} for {part_id} has no valid "
            "registry source visual binding"
        )
    if declared_path != source_path:
        raise ValueError(
            f"{SOURCE_VISUAL_PRESERVE_ACTION} for {part_id} does not match "
            "the registry source visual binding"
        )
    expected_digest = source_visual_binding_sha256(
        part_id=part_id,
        prim_path=prim_path,
        material_prim_path=source_path,
    )
    if assignment.get("source_visual_material_binding_sha256") != expected_digest:
        raise ValueError(
            f"{SOURCE_VISUAL_PRESERVE_ACTION} for {part_id} has an invalid "
            "source binding SHA-256"
        )
    forbidden = sorted(
        field
        for field in (
            "parameters",
            "face_subsets",
            "preserve_parent_material_binding",
        )
        if field in assignment
    )
    if forbidden:
        raise ValueError(
            f"{SOURCE_VISUAL_PRESERVE_ACTION} for {part_id} conflicts with "
            f"material-authoring fields: {forbidden}"
        )
    return source_path


def _policy_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _policy_unit(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{label} must be a finite number from 0 to 1")
    return float(value)


def _policy_string_array(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    result = [
        _policy_text(item, f"{label}[{index}]") for index, item in enumerate(value)
    ]
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must contain unique strings")
    return result


def validate_policy_fallback_authorization(
    plan: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    include_policy_fallback: bool,
) -> int:
    """Authorize audited policy fallbacks without widening default statuses.

    The explicit flag is only one half of the trust boundary.  A fallback plan
    must also be an exact registry cover, be hash-bound to that registry, and
    retain the zero-confidence provenance emitted by ``policy-exact-cover``.
    """

    assignments = plan.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("Material plan must contain an assignments list")
    fallback_assignments = [
        assignment
        for assignment in assignments
        if isinstance(assignment, Mapping)
        and assignment.get("status") == POLICY_FALLBACK_STATUS
    ]
    if not fallback_assignments:
        return 0
    if include_policy_fallback is not True:
        raise ValueError(
            "policy_fallback assignments require explicit "
            "--include-policy-fallback authorization"
        )

    parts = registry.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("Policy fallback registry must contain a non-empty parts list")
    registry_part_ids: list[str] = []
    registry_by_id: dict[str, Mapping[str, Any]] = {}
    for index, part in enumerate(parts):
        if not isinstance(part, Mapping):
            raise ValueError(f"registry.parts[{index}] must be an object")
        part_id = _policy_text(
            part.get("part_id"), f"registry.parts[{index}].part_id"
        )
        registry_part_ids.append(part_id)
        registry_by_id[part_id] = part
    if len(set(registry_part_ids)) != len(registry_part_ids):
        raise ValueError("Policy fallback registry contains duplicate part IDs")

    assignment_part_ids: list[str] = []
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, Mapping):
            raise ValueError(f"assignments[{index}] must be an object")
        assignment_part_ids.append(
            _policy_text(assignment.get("part_id"), f"assignments[{index}].part_id")
        )
    if len(set(assignment_part_ids)) != len(assignment_part_ids) or set(
        assignment_part_ids
    ) != set(registry_part_ids):
        missing = sorted(set(registry_part_ids) - set(assignment_part_ids))
        unexpected = sorted(set(assignment_part_ids) - set(registry_part_ids))
        raise ValueError(
            "Policy fallback plan does not exactly cover registry; "
            f"missing={missing}, unexpected={unexpected}"
        )

    provenance = plan.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Policy fallback plan must contain a provenance object")
    if provenance.get("mode") != POLICY_EXACT_COVER_MODE:
        raise ValueError(
            "Policy fallback plan provenance mode is not an audited exact cover"
        )
    expected_digests = {
        "registry_asset_sha256": registry.get("asset_sha256"),
        "registry_sha256": canonical_sha256(registry),
    }
    for field, expected in expected_digests.items():
        actual = provenance.get(field)
        if not isinstance(actual, str) or _SHA256.fullmatch(actual) is None:
            raise ValueError(
                f"Policy fallback plan provenance {field} must be a SHA-256 digest"
            )
        if actual != expected:
            raise ValueError(
                f"Policy fallback plan provenance {field} does not match registry"
            )

    for assignment in fallback_assignments:
        part_id = str(assignment["part_id"])
        confidence = _policy_unit(
            assignment.get("confidence"), f"policy fallback {part_id}.confidence"
        )
        if confidence != 0.0:
            raise ValueError(
                f"policy fallback {part_id}.confidence must remain exactly 0.0"
            )
        if assignment.get("evidence_views") != []:
            raise ValueError(
                f"policy fallback {part_id}.evidence_views must remain empty"
            )
        item_provenance = assignment.get("provenance")
        if not isinstance(item_provenance, Mapping):
            raise ValueError(
                f"policy fallback {part_id} must contain a provenance object"
            )
        _policy_text(
            item_provenance.get("tier"),
            f"policy fallback {part_id}.provenance.tier",
        )
        reason_codes = _policy_string_array(
            item_provenance.get("reason_codes"),
            f"policy fallback {part_id}.provenance.reason_codes",
        )
        if not reason_codes:
            raise ValueError(
                f"policy fallback {part_id}.provenance.reason_codes must be non-empty"
            )
        if (
            item_provenance.get("output_confidence_basis")
            != POLICY_FALLBACK_CONFIDENCE_BASIS
        ):
            raise ValueError(
                f"policy fallback {part_id}.provenance.output_confidence_basis "
                "is invalid"
            )
        sources = item_provenance.get("sources")
        if not isinstance(sources, list):
            raise ValueError(
                f"policy fallback {part_id}.provenance.sources must be an array"
            )
        source_part_ids: list[str] = []
        for index, source in enumerate(sources):
            label = f"policy fallback {part_id}.provenance.sources[{index}]"
            if not isinstance(source, Mapping):
                raise ValueError(f"{label} must be an object")
            source_part_ids.append(
                _policy_text(source.get("part_id"), f"{label}.part_id")
            )
            if source.get("source_status") not in {"auto", "approved", "review"}:
                raise ValueError(f"{label}.source_status is invalid")
            _policy_unit(source.get("source_confidence"), f"{label}.source_confidence")
            _policy_string_array(
                source.get("source_evidence_views"),
                f"{label}.source_evidence_views",
            )
        if len(set(source_part_ids)) != len(source_part_ids):
            raise ValueError(
                f"policy fallback {part_id}.provenance.sources "
                "contains duplicate part IDs"
            )
        tier = item_provenance.get("tier")
        action = assignment.get("apply_action")
        if tier == SOURCE_VISUAL_PRESERVE_TIER:
            validate_source_visual_preserve(
                part_id, assignment, registry_by_id[part_id]
            )
            required_reason_codes = {
                "SOURCE_VISUAL_MATERIAL_PRESENT",
                "SOURCE_VISUAL_BINDING_HASH_BOUND",
                "PRESERVE_SOURCE_VISUAL_NOOP",
            }
            if not required_reason_codes <= set(reason_codes):
                raise ValueError(
                    f"policy fallback {part_id} source-preserve reason_codes "
                    "are incomplete"
                )
        elif action is not None:
            raise ValueError(
                f"policy fallback {part_id} apply_action is not authorized "
                "by its provenance tier"
            )
    return len(fallback_assignments)


def preserve_parent_material_binding(
    part_id: str,
    assignment: dict[str, Any],
    *,
    has_face_subsets: bool,
) -> bool:
    """Validate and return an assignment's parent-binding policy."""

    preserve = assignment.get("preserve_parent_material_binding", False)
    if type(preserve) is not bool:
        raise ValueError(
            f"preserve_parent_material_binding for {part_id} must be a boolean"
        )
    if not preserve:
        return False
    if not has_face_subsets:
        raise ValueError(
            f"preserve_parent_material_binding for {part_id} requires face_subsets"
        )
    return True


def normalize_material_parameters(
    material_id: str, value: Any
) -> dict[str, NormalizedParameter]:
    """Validate the explicitly supported MDL parameter subset."""

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Material parameters for {material_id} must be an object")
    policy = material_parameter_policy(material_id)
    unexpected = set(value) - set(policy)
    if unexpected:
        raise ValueError(
            f"Unsupported material parameters for {material_id}: {sorted(unexpected)}"
        )
    normalized: dict[str, NormalizedParameter] = {}
    for name, raw in value.items():
        kind, minimum, maximum = policy[name]
        if kind == "color3f_linear":
            if (
                not isinstance(raw, (list, tuple))
                or len(raw) != 3
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    or minimum is None
                    or maximum is None
                    or not minimum <= float(item) <= maximum
                    for item in raw
                )
            ):
                raise ValueError(
                    f"{material_id}.{name} must be three finite linear RGB "
                    "values in [0, 1]"
                )
            normalized[name] = tuple(float(item) for item in raw)
        elif kind == "float":
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
                or minimum is None
                or maximum is None
                or not minimum <= float(raw) <= maximum
            ):
                raise ValueError(
                    f"{material_id}.{name} must be a finite number in "
                    f"[{minimum}, {maximum}]"
                )
            normalized[name] = float(raw)
        elif kind == "bool":
            if type(raw) is not bool:
                raise ValueError(f"{material_id}.{name} must be a boolean")
            normalized[name] = raw
        else:
            raise ValueError(f"Unsupported parameter policy for {material_id}.{name}")
    return normalized


def material_instance_key(
    material_id: str, parameters: dict[str, NormalizedParameter]
) -> str:
    """Return the stable cache key for one parameterized material instance."""

    canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    return f"{material_id}\0{canonical}"


def json_material_parameters(
    parameters: dict[str, NormalizedParameter],
) -> dict[str, list[float] | float | bool]:
    """Convert normalized immutable values to stable JSON-compatible values."""

    return {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in parameters.items()
    }


def normalize_face_subsets(
    part_id: str,
    value: Any,
    *,
    allowed_material_ids: set[str],
    face_count: int,
) -> list[dict[str, Any]]:
    """Validate material subsets without importing USD."""

    if value is None:
        return []
    if not isinstance(value, list) or not value:
        raise ValueError(f"face_subsets for {part_id} must be a non-empty list")
    if (
        isinstance(face_count, bool)
        or not isinstance(face_count, int)
        or face_count < 0
    ):
        raise ValueError(f"Invalid source face count for {part_id}: {face_count!r}")

    allowed_fields = {
        "subset_name",
        "material_id",
        "parameters",
        "semantic",
        "face_indices",
    }
    seen_names: set[str] = set()
    claimed_faces: set[int] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        label = f"face_subsets[{index}] for {part_id}"
        if not isinstance(raw, dict):
            raise ValueError(f"{label} must be an object")
        unexpected = set(raw) - allowed_fields
        if unexpected:
            raise ValueError(f"{label} has unexpected fields: {sorted(unexpected)}")

        subset_name = raw.get("subset_name")
        if not isinstance(subset_name, str) or not SAFE_SUBSET_NAME.fullmatch(
            subset_name
        ):
            raise ValueError(f"{label} has unsafe subset_name: {subset_name!r}")
        if subset_name in seen_names:
            raise ValueError(f"Duplicate subset_name for {part_id}: {subset_name}")
        seen_names.add(subset_name)

        material_id = raw.get("material_id")
        if not isinstance(material_id, str) or material_id not in allowed_material_ids:
            raise ValueError(f"{label} has unknown material_id: {material_id!r}")

        face_indices = raw.get("face_indices")
        if not isinstance(face_indices, list) or not face_indices:
            raise ValueError(f"{label}.face_indices must be a non-empty list")
        if any(
            isinstance(face_index, bool) or not isinstance(face_index, int)
            for face_index in face_indices
        ):
            raise ValueError(f"{label}.face_indices must contain only integers")
        if len(set(face_indices)) != len(face_indices):
            raise ValueError(f"{label}.face_indices must be unique")
        out_of_range = sorted(
            face_index
            for face_index in face_indices
            if face_index < 0 or face_index >= face_count
        )
        if out_of_range:
            raise ValueError(
                f"{label}.face_indices out of range [0, {face_count}): {out_of_range}"
            )
        overlap = sorted(claimed_faces & set(face_indices))
        if overlap:
            raise ValueError(
                f"face_subsets for {part_id} overlap at face indices: {overlap}"
            )
        claimed_faces.update(face_indices)

        parameters = normalize_material_parameters(material_id, raw.get("parameters"))
        subset: dict[str, Any] = {
            "subset_name": subset_name,
            "material_id": material_id,
            "face_indices": tuple(face_indices),
            "parameters": parameters,
        }
        if "semantic" in raw:
            semantic = raw["semantic"]
            if not isinstance(semantic, str) or not semantic.strip():
                raise ValueError(f"{label}.semantic must be a non-empty string")
            subset["semantic"] = semantic
        normalized.append(subset)
    return normalized


def safe_child_name(material_id: str) -> str:
    """Return a deterministic USD-safe child name for a material ID."""

    stem = re.sub(r"[^A-Za-z0-9_]", "_", material_id).strip("_")
    if not stem or stem[0].isdigit():
        stem = f"M_{stem}"
    digest = hashlib.sha1(material_id.encode("utf-8")).hexdigest()[:10]
    return f"{stem[:80]}_{digest}"


def sha256_file(path: Path) -> str:
    """Hash a file without loading it fully into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def catalog_map(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index and validate catalog records by material ID."""

    materials = catalog.get("materials")
    if not isinstance(materials, list):
        raise ValueError("Catalog must contain a materials list")
    result: dict[str, dict[str, Any]] = {}
    for item in materials:
        if isinstance(item, dict) and isinstance(item.get("material_id"), str):
            material_id = item["material_id"]
            if material_id in result:
                raise ValueError(
                    f"Catalog contains duplicate material_id: {material_id}"
                )
            result[material_id] = item
    if not result:
        raise ValueError("Catalog contains no material_id entries")
    return result


def resolve_mdl_path(item: dict[str, Any], material_root: Path) -> Path:
    """Resolve one catalog MDL path below the caller-controlled root."""

    raw = item.get("mdl_path") or item.get("mdl_relpath") or item.get("module_path")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Material {item.get('material_id')} has no MDL path")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = material_root / candidate
    candidate = candidate.resolve(strict=True)
    root = material_root.resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"MDL path escapes material root: {candidate}") from exc
    if candidate.suffix.lower() != ".mdl":
        raise ValueError(f"Material is not an MDL module: {candidate}")
    return candidate


def get_subidentifier(item: dict[str, Any]) -> str:
    """Return and validate the catalog's exported MDL material name."""

    value = (
        item.get("sub_identifier")
        or item.get("subidentifier")
        or item.get("entrypoint")
        or item.get("material_name")
    )
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Material {item.get('material_id')} has no subidentifier")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Unsafe MDL subidentifier: {value!r}")
    return value


__all__ = [
    "APPLY_STATUSES",
    "PARAMETER_POLICIES",
    "POLICY_EXACT_COVER_MODE",
    "POLICY_FALLBACK_CONFIDENCE_BASIS",
    "POLICY_FALLBACK_STATUS",
    "SOURCE_VISUAL_PRESERVE_ACTION",
    "SOURCE_VISUAL_PRESERVE_TIER",
    "NormalizedParameter",
    "canonical_sha256",
    "catalog_map",
    "get_subidentifier",
    "json_material_parameters",
    "material_parameter_policy",
    "material_instance_key",
    "normalize_face_subsets",
    "normalize_material_parameters",
    "preserve_parent_material_binding",
    "resolve_mdl_path",
    "safe_child_name",
    "sha256_file",
    "source_visual_binding_sha256",
    "validate_policy_fallback_authorization",
    "validate_source_visual_preserve",
]
