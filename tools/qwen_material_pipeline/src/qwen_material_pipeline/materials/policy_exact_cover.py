"""Build an audited, best-effort exact-cover material plan.

This module is deliberately separate from the conservative confidence gate.
It never promotes weak evidence to ``auto`` or ``approved``.  Parts that are
not accepted by the gate are emitted with the explicit ``policy_fallback``
status, zero output confidence, and machine-readable provenance.

The policy is intended for an explicitly opted-in unattended CAD workflow:

1. keep gate-approved assignments;
2. apply ordered, whitelist-bounded industrial name rules;
3. optionally preserve a hash-bound source visual binding when the caller
   explicitly treats CAD display materials as appearance evidence; and
4. otherwise ignore unverified CAD display colours and fill the remaining
   parts with a declared neutral material.

Exact geometry-fingerprint propagation remains available as an explicit
policy option, but is disabled by default: identical meshes may intentionally
have different colours or materials.

Review candidates and staged candidates rejected by the confidence gate are
audit inputs only.  They can never author a material or seed propagation.
Likewise, assembly ancestry and CAD leaf names are not material identities.

No USD or ``pxr`` dependency is used.  The builder consumes only JSON
artifacts already produced by the staged workflow.
"""

from __future__ import annotations

import argparse
import colorsys
import copy
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .tuning import (
    tune_selected_material_color_from_mvinverse,
    tune_selected_material_from_mvinverse,
    tuning_profile_for_material,
)
from ..core.staged_analysis import (
    StagedAnalysisError,
    collapse_recovery_group_ids,
)
from ..evidence.color_semantics import (
    fusion_color_label,
    pixel_color_label,
)
from ..evidence.palette_fusion import (
    UNRESOLVED_PIXEL_CHROMATIC_ASSOCIATION,
    is_verified_unresolved_pixel_chromatic_group,
)
from ..mvinverse.evidence import (
    MVInverseEvidenceError,
    validate_mvinverse_evidence,
)
from ..usd.material_common import (
    SOURCE_VISUAL_PRESERVE_ACTION,
    SOURCE_VISUAL_PRESERVE_TIER,
    source_visual_binding_sha256,
)
from ..usd.registry import (
    SOURCE_MATERIAL_BIND_SUBSETS_FIELD,
    SOURCE_SUBSET_HASH_FIELD,
    source_material_bind_subset_sha256,
)


POLICY_SCHEMA_VERSION = "qwen-policy-exact-cover/v1"
REPORT_SCHEMA_VERSION = "qwen-policy-exact-cover-report/v1"
PLAN_SCHEMA_VERSION = "1.0"
REGISTRY_SCHEMA_VERSION = "qwen-material-parts/v1"
STAGED_SCHEMA_VERSION = "qwen-staged-material-result/v1"
GATE_SCHEMA_VERSION = "qwen-material-confidence-gate/v1"
GROUP_MATERIALS_SCHEMA_VERSION = "qwen-palette-material/v1"
MVINVERSE_SCHEMA_VERSION = "qwen-mvinverse-pbr-evidence/v1"
PALETTE_FUSION_SCHEMA_VERSION = "qwen-multiview-palette-fusion/v1"
CANONICAL_PALETTE_SCHEMA_VERSION = "qwen-canonical-material-palette/v1"
PART_ID_EVIDENCE_SCHEMA_VERSION = "qwen-part-id-reference-evidence/v1"
FALLBACK_STATUS = "policy_fallback"
# The production policy is bounded to the configured NVIDIA ``Base`` root.
# IDs are therefore relative to that root rather than to its ``Materials``
# parent.  There is no galvanized preset in Base, so carbon steel is the
# closest physically correct unattended fallback for ordinary fasteners.
GENERIC_STEEL_PAINTED = "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte"
GALVANIZED_STEEL = "mdl:Metals/Steel_Carbon.mdl#Steel_Carbon"
STAINLESS_STEEL_MATTE = "mdl:Metals/Steel_Stainless.mdl#Steel_Stainless"
BLACK_RUBBER = "mdl:Plastics/Rubber_Smooth.mdl#Rubber_Smooth"
BLACK_PLASTIC = "mdl:Plastics/Plastic_ABS.mdl#Plastic_ABS"
COPPER = "mdl:Metals/Copper.mdl#Copper"
BRASS = "mdl:Metals/Brass.mdl#Brass"
BLACK_PAINTED_STEEL = (
    "mdl:Metals/Aluminum_Anodized_Black.mdl#Aluminum_Anodized_Black"
)


DEFAULT_POLICY: dict[str, Any] = {
    "schema_version": POLICY_SCHEMA_VERSION,
    "candidate_auto_cluster_keys": [
        "existing_visual_material",
    ],
    "review_cluster_keys": [],
    "default_strategy": "declared_material",
    "default_material_id": STAINLESS_STEEL_MATTE,
    # ``preserve`` retains compatibility for callers whose authored USD
    # materials are real appearance evidence.  Reference-photo STEP workflows
    # select ``neutralize_unverified`` automatically because CAD assembly
    # display colours are commonly only part-separation aids.
    "source_visual_strategy": "preserve",
    "ground_contact": {
        # Contact with the floor is geometry, not appearance evidence.  Keep
        # this opt-in for deployments with an explicit machine-base policy;
        # the generic unattended path must not turn pedals or fixtures black.
        "enabled": False,
        "material_id": BLACK_PAINTED_STEEL,
        "elevation_tolerance_ratio": 0.02,
        "minimum_lateral_span_ratio": 0.12,
        "maximum_up_span_ratio": 0.18,
    },
    "semantic_rules": [
        {
            "rule_id": "rubber_seal_or_o_ring",
            "pattern": r"(?:O[_ -]?RING|SEAL|GASKET|PACKING|RUBBER|GBT?3452)",
            "material_id": BLACK_RUBBER,
            "semantic": "industrial rubber seal or O-ring",
        },
        {
            "rule_id": "explicit_copper",
            "pattern": r"(?:COPPER|(?:^|[_/])CU(?:[_/]|$))",
            "material_id": COPPER,
            "semantic": "copper component",
        },
        {
            "rule_id": "explicit_brass",
            "pattern": r"(?:BRASS|HUANGTONG)",
            "material_id": BRASS,
            "semantic": "brass component",
        },
        {
            "rule_id": "hose_or_cable",
            "pattern": r"(?:HOSE|CABLE|WIRE|PNEUMATIC|FLEX(?:IBLE)?[_ -]?TUBE)",
            "material_id": BLACK_RUBBER,
            "semantic": "flexible hose or cable jacket",
        },
        {
            "rule_id": "standard_fastener",
            "pattern": (
                r"(?:GBT?\d|DIN\d|ISO\d|JIS\d|ANSI\d|"
                r"(?:^|[_/])FB\d|BOLT|SCREW|NUT|WASHER)"
            ),
            "material_id": GALVANIZED_STEEL,
            "semantic": "standard industrial fastener",
        },
        {
            "rule_id": "hose_clamp",
            "pattern": r"(?:HOUKU|HOSE[_ -]?CLAMP|CLAMP)",
            "material_id": STAINLESS_STEEL_MATTE,
            "semantic": "stainless clamp",
        },
        {
            "rule_id": "plastic_electrical",
            "pattern": (
                r"(?:PLASTIC|ABS|NYLON|CONNECTOR|SENSOR|SWITCH|"
                r"ELECTRICAL[_ -]?PLUG)"
            ),
            "material_id": BLACK_PLASTIC,
            "semantic": "industrial plastic or electrical housing",
        },
    ],
}

_CLUSTER_KEYS = frozenset(
    {"parent_path", "cad_leaf", "existing_visual_material", "geometry_fingerprint"}
)
_EXACT_IDENTITY_CLUSTER_KEYS = frozenset(
    {"geometry_fingerprint", "existing_visual_material"}
)
_MVINVERSE_TRUSTED_TIERS = frozenset(
    {
        "autonomous_base_plan",
        "corroborated_source_visual_nvidia_mdl",
        "gate_auto",
        "trusted_exact_geometry_fingerprint",
        "trusted_authored_material_binding",
    }
)
CORROBORATED_SOURCE_MDL_TIER = "corroborated_source_visual_nvidia_mdl"
CORROBORATED_SOURCE_PROVISIONAL_MATERIAL_BASIS = (
    "high_confidence_whitelist_candidate_pending_render_qa"
)
CORROBORATED_SOURCE_MIN_PROVISIONAL_CONFIDENCE = 0.85
CORROBORATED_SOURCE_MIN_CONFIRMED_CONFIDENCE = 0.60
_SOURCE_VISUAL_STRATEGIES = frozenset({"preserve", "neutralize_unverified"})
_SOURCE_ACCENT_CHROMATIC_COLORS = frozenset(
    {"red", "orange", "yellow", "green", "blue", "pink"}
)
_SOURCE_ACCENT_MAX_REGISTRY_FRACTION = 0.05
_SOURCE_ACCENT_MIN_SIGNATURE_COUNT = 4
_SOURCE_ACCENT_MIN_GEOMETRY_REPEAT_COUNT = 4
_SOURCE_ACCENT_MIN_SATURATION = 0.5
_SOURCE_ACCENT_OPACITY_FLOOR = 0.99


class PolicyExactCoverError(ValueError):
    """Raised when deterministic exact coverage cannot be audited."""


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyExactCoverError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PolicyExactCoverError(f"{label} must be an array")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PolicyExactCoverError(f"{label} must be a non-empty string")
    return value.strip()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _unit_or_zero(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0.0 <= float(value) <= 1.0
    ):
        return 0.0
    return float(value)


def _whitelist_ids(document: Mapping[str, Any]) -> set[str]:
    if document.get("schema_version") != 1:
        raise PolicyExactCoverError("whitelist schema_version must be 1")
    raw_ids = _array(document.get("material_ids"), "whitelist.material_ids")
    identifiers = [
        _text(value, f"whitelist.material_ids[{index}]")
        for index, value in enumerate(raw_ids)
    ]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise PolicyExactCoverError(
            "whitelist.material_ids must be non-empty and unique"
        )
    return set(identifiers)


def _part_id_evidence_statuses(
    document: Mapping[str, Any] | None,
    *,
    registry: Mapping[str, Any],
    registry_ids: set[str],
) -> dict[str, str]:
    """Validate the final Part-ID visibility evidence used by policy fallback.

    The exact-cover policy is normally authored before the relatively expensive
    local Part-ID projection stage.  A caller may supply that final evidence on
    a deterministic second pass.  Doing so prevents a CAD part that proved
    hidden in every reference view from inheriting any earlier palette/group
    choice while preserving the legacy first-pass behaviour when no evidence
    is supplied.
    """

    if document is None:
        return {}
    if (
        document.get("schema_version") != PART_ID_EVIDENCE_SCHEMA_VERSION
        or document.get("assignment_unit") != "part_id"
    ):
        raise PolicyExactCoverError(
            "Part-ID evidence has an unsupported schema or assignment unit"
        )
    integrity = document.get("integrity")
    unsigned = dict(document)
    unsigned.pop("integrity", None)
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("document_sha256") != _canonical_sha256(unsigned)
    ):
        raise PolicyExactCoverError("Part-ID evidence failed its integrity seal")
    raw_inputs = document.get("inputs")
    rendered_registry_inputs = [
        raw_input
        for raw_input in (raw_inputs if isinstance(raw_inputs, list) else [])
        if isinstance(raw_input, Mapping)
        and raw_input.get("label") == "rendered_registry"
    ]
    if (
        not isinstance(raw_inputs, list)
        or len(rendered_registry_inputs) != 1
        or rendered_registry_inputs[0].get("document_sha256")
        != _canonical_sha256(registry)
    ):
        raise PolicyExactCoverError(
            "Part-ID evidence is not bound to the policy rendered registry"
        )
    raw_parts = document.get("parts")
    if not isinstance(raw_parts, list):
        raise PolicyExactCoverError("Part-ID evidence has no parts")
    statuses: dict[str, str] = {}
    for index, raw_part in enumerate(raw_parts):
        if not isinstance(raw_part, Mapping):
            raise PolicyExactCoverError(
                f"Part-ID evidence part {index} is not an object"
            )
        part_id = raw_part.get("part_id")
        status = raw_part.get("status")
        observations = raw_part.get("observations")
        if (
            not isinstance(part_id, str)
            or not part_id
            or part_id in statuses
            or status not in {"observed", "unobserved"}
            or not isinstance(observations, list)
            or (status == "observed" and not observations)
            or (status == "unobserved" and observations)
        ):
            raise PolicyExactCoverError(
                f"Part-ID evidence part {index} is malformed"
            )
        statuses[part_id] = str(status)
    if set(statuses) != registry_ids:
        raise PolicyExactCoverError(
            "Part-ID evidence does not exactly cover the policy registry"
        )
    summary = document.get("summary")
    observed_count = sum(status == "observed" for status in statuses.values())
    if (
        not isinstance(summary, Mapping)
        or summary.get("registry_part_count") != len(registry_ids)
        or summary.get("observed_part_count") != observed_count
        or summary.get("unobserved_part_count")
        != len(registry_ids) - observed_count
    ):
        raise PolicyExactCoverError("Part-ID evidence summary is inconsistent")
    return statuses


def _registry_source_material_bind_subsets(
    part: Mapping[str, Any],
    *,
    part_id: str,
) -> list[dict[str, Any]]:
    """Validate the hash-bound source subset contract used by USD apply."""

    raw_value = part.get(SOURCE_MATERIAL_BIND_SUBSETS_FIELD, [])
    raw_subsets = _array(
        raw_value,
        f"registry part {part_id}.{SOURCE_MATERIAL_BIND_SUBSETS_FIELD}",
    )
    if not raw_subsets:
        return []

    prim_path = _text(
        part.get("prim_path"), f"registry part {part_id}.prim_path"
    )
    face_count = part.get("face_count")
    if (
        isinstance(face_count, bool)
        or not isinstance(face_count, int)
        or face_count < 0
    ):
        raise PolicyExactCoverError(
            f"registry part {part_id}.face_count must be a non-negative integer "
            "when source materialBind subsets are present"
        )

    expected_fields = {
        "subset_name",
        "subset_prim_path",
        "family_name",
        "family_type",
        "element_type",
        "face_indices",
        "visual_material_prim_path",
        "binding_relationship_name",
        "binding_targets",
        SOURCE_SUBSET_HASH_FIELD,
    }
    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    claimed_faces: set[int] = set()
    for index, raw in enumerate(raw_subsets):
        subset = dict(
            _object(
                raw,
                f"registry part {part_id} source materialBind subset[{index}]",
            )
        )
        if set(subset) != expected_fields:
            raise PolicyExactCoverError(
                f"registry part {part_id} source materialBind subset[{index}] "
                f"fields differ; missing={sorted(expected_fields - set(subset))}, "
                f"unexpected={sorted(set(subset) - expected_fields)}"
            )
        subset_name = _text(
            subset.get("subset_name"),
            f"registry part {part_id} source subset[{index}].subset_name",
        )
        subset_path = _text(
            subset.get("subset_prim_path"),
            f"registry part {part_id} source subset[{index}].subset_prim_path",
        )
        expected_path = f"{prim_path}/{subset_name}"
        if subset_path != expected_path:
            raise PolicyExactCoverError(
                f"registry part {part_id} source subset is not a direct Mesh child: "
                f"{subset_path!r} != {expected_path!r}"
            )
        if subset_name in seen_names or subset_path in seen_paths:
            raise PolicyExactCoverError(
                f"registry part {part_id} has duplicate source materialBind subset: "
                f"{subset_name}"
            )
        seen_names.add(subset_name)
        seen_paths.add(subset_path)
        if subset.get("family_name") != "materialBind":
            raise PolicyExactCoverError(
                f"registry part {part_id}.{subset_name} is not materialBind"
            )
        if subset.get("family_type") not in {
            "unrestricted",
            "nonOverlapping",
            "partition",
        }:
            raise PolicyExactCoverError(
                f"registry part {part_id}.{subset_name} has an invalid "
                f"materialBind family type: {subset.get('family_type')!r}"
            )
        if subset.get("element_type") != "face":
            raise PolicyExactCoverError(
                f"registry part {part_id}.{subset_name} is not a face subset"
            )

        face_indices = list(
            _array(
                subset.get("face_indices"),
                f"registry part {part_id}.{subset_name}.face_indices",
            )
        )
        if not face_indices or any(
            isinstance(face_index, bool) or not isinstance(face_index, int)
            for face_index in face_indices
        ):
            raise PolicyExactCoverError(
                f"registry part {part_id}.{subset_name}.face_indices must be a "
                "non-empty integer array"
            )
        if len(set(face_indices)) != len(face_indices):
            raise PolicyExactCoverError(
                f"registry part {part_id}.{subset_name}.face_indices are not unique"
            )
        out_of_range = sorted(
            face_index
            for face_index in face_indices
            if face_index < 0 or face_index >= face_count
        )
        if out_of_range:
            raise PolicyExactCoverError(
                f"registry part {part_id}.{subset_name}.face_indices are outside "
                f"[0, {face_count}): {out_of_range}"
            )
        overlap = sorted(claimed_faces & set(face_indices))
        if overlap:
            raise PolicyExactCoverError(
                f"registry part {part_id} source materialBind subsets overlap: "
                f"{overlap}"
            )
        claimed_faces.update(face_indices)
        subset["face_indices"] = face_indices

        for field in (
            "visual_material_prim_path",
            "binding_relationship_name",
        ):
            if subset.get(field) is not None and not isinstance(
                subset.get(field), str
            ):
                raise PolicyExactCoverError(
                    f"registry part {part_id}.{subset_name}.{field} must be a "
                    "string or null"
                )
        binding_targets = list(
            _array(
                subset.get("binding_targets"),
                f"registry part {part_id}.{subset_name}.binding_targets",
            )
        )
        if any(not isinstance(target, str) or not target for target in binding_targets):
            raise PolicyExactCoverError(
                f"registry part {part_id}.{subset_name}.binding_targets must be "
                "a string array"
            )
        if binding_targets != sorted(set(binding_targets)):
            raise PolicyExactCoverError(
                f"registry part {part_id}.{subset_name}.binding_targets must be "
                "sorted and unique"
            )
        subset["binding_targets"] = binding_targets
        expected_digest = source_material_bind_subset_sha256(
            part_id=part_id,
            prim_path=prim_path,
            subset_record=subset,
        )
        if subset.get(SOURCE_SUBSET_HASH_FIELD) != expected_digest:
            raise PolicyExactCoverError(
                f"registry part {part_id}.{subset_name} source subset hash is invalid"
            )
        normalized.append(subset)

    if [item["subset_prim_path"] for item in normalized] != sorted(seen_paths):
        raise PolicyExactCoverError(
            f"registry part {part_id} source materialBind subsets are not "
            "canonically ordered"
        )
    return normalized


def _registry_parts(
    document: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if document.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise PolicyExactCoverError("registry has an unsupported schema_version")
    normalized: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(_array(document.get("parts"), "registry.parts")):
        part = dict(_object(raw, f"registry.parts[{index}]"))
        part_id = _text(part.get("part_id"), f"registry.parts[{index}].part_id")
        if part_id in by_id:
            raise PolicyExactCoverError(f"duplicate registry part_id: {part_id}")
        for field in ("prim_path", "parent_path"):
            _text(part.get(field), f"registry part {part_id}.{field}")
        part[SOURCE_MATERIAL_BIND_SUBSETS_FIELD] = (
            _registry_source_material_bind_subsets(part, part_id=part_id)
        )
        by_id[part_id] = part
        normalized.append(part)
    if not normalized or document.get("part_count") != len(normalized):
        raise PolicyExactCoverError("registry part_count does not match parts")
    return normalized, by_id


def _candidate_assignments(
    staged_result: Mapping[str, Any],
    *,
    registry_ids: set[str],
    allowed_material_ids: set[str],
) -> dict[str, dict[str, Any]]:
    if staged_result.get("schema_version") != STAGED_SCHEMA_VERSION:
        raise PolicyExactCoverError("staged result has an unsupported schema_version")
    plan = _object(staged_result.get("material_plan"), "staged_result.material_plan")
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise PolicyExactCoverError("staged material plan schema_version must be '1.0'")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(
        _array(plan.get("assignments"), "staged material assignments")
    ):
        assignment = copy.deepcopy(dict(_object(raw, f"staged assignment[{index}]")))
        part_id = _text(
            assignment.get("part_id"), f"staged assignment[{index}].part_id"
        )
        if part_id in result:
            raise PolicyExactCoverError(f"duplicate staged assignment: {part_id}")
        if part_id not in registry_ids:
            raise PolicyExactCoverError(
                f"staged assignment is absent from registry: {part_id}"
            )
        material_id = _text(
            assignment.get("material_id"),
            f"staged assignment {part_id}.material_id",
        )
        if material_id not in allowed_material_ids:
            raise PolicyExactCoverError(
                f"staged assignment {part_id} is outside industrial whitelist: "
                f"{material_id}"
            )
        status = assignment.get("status")
        if status not in {"auto", "approved", "review"}:
            raise PolicyExactCoverError(
                f"staged assignment {part_id} has unsupported status: {status!r}"
            )
        result[part_id] = assignment
    return result


def _collapse_recovery_group_ids(
    staged_result: Mapping[str, Any],
) -> set[str]:
    """Return palette groups that cannot seed a fallback material."""

    try:
        return collapse_recovery_group_ids(staged_result)
    except StagedAnalysisError as exc:
        raise PolicyExactCoverError(
            f"staged collapse recovery audit is invalid: {exc}"
        ) from exc


def _autonomous_base_assignments(
    base_plan: Mapping[str, Any] | None,
    *,
    registry_ids: set[str],
    allowed_material_ids: set[str],
) -> dict[str, dict[str, Any]]:
    if base_plan is None:
        return {}
    if base_plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise PolicyExactCoverError("base plan schema_version must be '1.0'")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(
        _array(base_plan.get("assignments"), "base_plan.assignments")
    ):
        assignment = copy.deepcopy(
            dict(_object(raw, f"base_plan.assignments[{index}]"))
        )
        part_id = _text(
            assignment.get("part_id"), f"base_plan.assignments[{index}].part_id"
        )
        if part_id in result:
            raise PolicyExactCoverError(f"duplicate base-plan assignment: {part_id}")
        if part_id not in registry_ids:
            raise PolicyExactCoverError(
                f"base-plan assignment is absent from registry: {part_id}"
            )
        status = assignment.get("status")
        if status not in {"auto", "approved"}:
            raise PolicyExactCoverError(
                f"base-plan assignment {part_id} is not safely applicable: {status!r}"
            )
        material_id = _text(
            assignment.get("material_id"),
            f"base-plan assignment {part_id}.material_id",
        )
        if material_id not in allowed_material_ids:
            raise PolicyExactCoverError(
                f"base-plan assignment {part_id} is outside industrial whitelist: "
                f"{material_id}"
            )
        for subset_index, raw_subset in enumerate(
            _array(
                assignment.get("face_subsets", []),
                f"base-plan assignment {part_id}.face_subsets",
            )
        ):
            subset = _object(
                raw_subset,
                f"base-plan assignment {part_id}.face_subsets[{subset_index}]",
            )
            subset_material = _text(
                subset.get("material_id"),
                f"base-plan assignment {part_id} subset material_id",
            )
            if subset_material not in allowed_material_ids:
                raise PolicyExactCoverError(
                    f"base-plan face subset for {part_id} is outside industrial "
                    f"whitelist: {subset_material}"
                )
        result[part_id] = assignment
    return result


def _gate_decision_records(
    confidence_gate: Mapping[str, Any], *, registry_ids: set[str]
) -> dict[str, dict[str, Any]]:
    if confidence_gate.get("schema_version") != GATE_SCHEMA_VERSION:
        raise PolicyExactCoverError("confidence gate has an unsupported schema_version")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(
        _array(confidence_gate.get("decisions"), "confidence_gate.decisions")
    ):
        decision = _object(raw, f"confidence_gate.decisions[{index}]")
        part_id = _text(
            decision.get("part_id"), f"confidence_gate.decisions[{index}].part_id"
        )
        if part_id in result:
            raise PolicyExactCoverError(f"duplicate gate decision: {part_id}")
        value = decision.get("decision")
        if value not in {"auto", "review", "preserve"}:
            raise PolicyExactCoverError(
                f"confidence gate decision for {part_id} is invalid: {value!r}"
            )
        result[part_id] = copy.deepcopy(dict(decision))
    if set(result) != registry_ids:
        raise PolicyExactCoverError(
            "confidence gate decisions do not exactly cover registry"
        )
    return result


def _effective_policy(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    if policy is None:
        return copy.deepcopy(DEFAULT_POLICY)
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise PolicyExactCoverError("policy has an unsupported schema_version")
    merged = copy.deepcopy(DEFAULT_POLICY)
    merged.update(copy.deepcopy(dict(policy)))
    return merged


def _material_id_for_catalog(
    material_id: str, allowed_material_ids: set[str]
) -> str:
    """Resolve Base-root and Materials-root IDs without weakening the allowlist."""

    if material_id in allowed_material_ids:
        return material_id
    if material_id.startswith("mdl:Base/"):
        candidate = "mdl:" + material_id[len("mdl:Base/") :]
    elif material_id.startswith("mdl:"):
        candidate = "mdl:Base/" + material_id[len("mdl:") :]
    else:
        return material_id
    return candidate if candidate in allowed_material_ids else material_id


def _policy_for_catalog(
    policy: Mapping[str, Any], allowed_material_ids: set[str]
) -> dict[str, Any]:
    """Translate only trusted policy constants to the active catalog namespace."""

    result = copy.deepcopy(dict(policy))
    default_material = result.get("default_material_id")
    if isinstance(default_material, str):
        result["default_material_id"] = _material_id_for_catalog(
            default_material, allowed_material_ids
        )
    ground = result.get("ground_contact")
    if isinstance(ground, Mapping):
        resolved_ground = copy.deepcopy(dict(ground))
        material_id = resolved_ground.get("material_id")
        if isinstance(material_id, str):
            resolved_ground["material_id"] = _material_id_for_catalog(
                material_id, allowed_material_ids
            )
        result["ground_contact"] = resolved_ground
    rules = result.get("semantic_rules")
    if isinstance(rules, list):
        resolved_rules: list[Any] = []
        for raw_rule in rules:
            if not isinstance(raw_rule, Mapping):
                resolved_rules.append(copy.deepcopy(raw_rule))
                continue
            rule = copy.deepcopy(dict(raw_rule))
            material_id = rule.get("material_id")
            if isinstance(material_id, str):
                rule["material_id"] = _material_id_for_catalog(
                    material_id, allowed_material_ids
                )
            resolved_rules.append(rule)
        result["semantic_rules"] = resolved_rules
    return result


def _cluster_value(part: Mapping[str, Any], key: str) -> str | None:
    if key == "parent_path":
        return str(part["parent_path"])
    if key == "cad_leaf":
        return str(part["parent_path"]).rsplit("/", 1)[-1]
    if key == "existing_visual_material":
        value = part.get("existing_visual_material")
        if not isinstance(value, str) or not value:
            return None
        # Material names such as ``Diffuse_1`` are commonly repeated under
        # unrelated CAD components.  Only the full authored binding path is a
        # usable appearance identity.
        return value
    if key == "geometry_fingerprint":
        value = part.get("geometry_fingerprint")
        return value if isinstance(value, str) and value else None
    raise AssertionError(f"unsupported cluster key: {key}")


def _validated_cluster_keys(policy: Mapping[str, Any], field: str) -> list[str]:
    result = [
        _text(value, f"policy.{field}[{index}]")
        for index, value in enumerate(_array(policy.get(field), f"policy.{field}"))
    ]
    invalid = sorted(set(result) - _CLUSTER_KEYS)
    if invalid:
        raise PolicyExactCoverError(
            f"policy.{field} contains unsupported keys: {invalid}"
        )
    if len(set(result)) != len(result):
        raise PolicyExactCoverError(f"policy.{field} contains duplicate keys")
    return result


def _fallback_assignment(
    *,
    part_id: str,
    material_id: str,
    semantic: str,
    tier: str,
    reason_codes: list[str],
    sources: Sequence[Mapping[str, Any]] = (),
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    provenance_sources: list[dict[str, Any]] = []
    for source in sources:
        provenance_sources.append(
            {
                "part_id": source["part_id"],
                "source_status": source.get("status"),
                "source_confidence": _unit_or_zero(source.get("confidence")),
                "source_evidence_views": list(source.get("evidence_views", [])),
            }
        )
    result = {
        "part_id": part_id,
        "material_id": material_id,
        "semantic": semantic,
        "confidence": 0.0,
        "evidence_views": [],
        "status": FALLBACK_STATUS,
        "provenance": {
            "tier": tier,
            "reason_codes": list(reason_codes),
            "output_confidence_basis": "policy fallback; not evidence confidence",
            "sources": provenance_sources,
        },
    }
    if parameters is not None:
        result["parameters"] = copy.deepcopy(dict(parameters))
    return result


def _valid_source_visual_material_path(value: Any) -> str | None:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or value == "/"
        or value.endswith("/")
        or "//" in value
        or any(character.isspace() for character in value)
    ):
        return None
    return value


def _source_visual_preserve_assignment(
    *,
    part: Mapping[str, Any],
    fallback_material_id: str,
    corroboration: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return a hash-bound material no-op when the registry proves a binding."""

    source_material = _valid_source_visual_material_path(
        part.get("existing_visual_material")
    )
    if source_material is None:
        return None
    part_id = str(part["part_id"])
    prim_path = str(part["prim_path"])
    assignment = _fallback_assignment(
        part_id=part_id,
        material_id=fallback_material_id,
        semantic=(
            "source CAD appearance preserved after independent reference-palette "
            "and repeated-geometry corroboration"
            if corroboration is not None
            else "preserved visual material authored by the source CAD asset"
        ),
        tier=SOURCE_VISUAL_PRESERVE_TIER,
        reason_codes=[
            "SOURCE_VISUAL_MATERIAL_PRESENT",
            "SOURCE_VISUAL_BINDING_HASH_BOUND",
            "PRESERVE_SOURCE_VISUAL_NOOP",
            *(
                [
                    "REFERENCE_PALETTE_MULTIVIEW_COLOR_CORROBORATION",
                    "RARE_SOURCE_VISUAL_SIGNATURE",
                    "REPEATED_GEOMETRY_SOURCE_LOCATOR",
                    *(
                        ["EXACT_SOURCE_SIGNATURE_COHORT_EXPANSION"]
                        if corroboration.get("signature_expansion_basis")
                        == "exact_source_signature_with_repeated_geometry_anchor"
                        else []
                    ),
                ]
                if corroboration is not None
                else []
            ),
        ],
    )
    if corroboration is not None:
        assignment["provenance"]["source_visual_corroboration"] = copy.deepcopy(
            dict(corroboration)
        )
    assignment.update(
        {
            "apply_action": SOURCE_VISUAL_PRESERVE_ACTION,
            "source_visual_material_prim_path": source_material,
            "source_visual_material_binding_sha256": (
                source_visual_binding_sha256(
                    part_id=part_id,
                    prim_path=prim_path,
                    material_prim_path=source_material,
                )
            ),
        }
    )
    return assignment


def _unit_number_or_none(value: Any) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        return None
    return float(value)


def _linear_channel_to_srgb(value: float) -> float:
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * (value ** (1.0 / 2.4)) - 0.055


def _rgb_color_label(rgb: Sequence[float]) -> str:
    encoded = tuple(
        max(0, min(255, int(round(float(channel) * 255.0)))) for channel in rgb
    )
    return fusion_color_label(pixel_color_label(*encoded))


def _rgb_saturation(rgb: Sequence[float]) -> float:
    return float(colorsys.rgb_to_hsv(*(float(channel) for channel in rgb))[1])


def _source_visual_signature(
    part: Mapping[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    """Return a strict, renderer-independent source PreviewSurface signature."""

    properties = part.get("existing_visual_material_properties")
    if not isinstance(properties, Mapping):
        return None
    if properties.get("shader_id") != "UsdPreviewSurface":
        return None
    raw_diffuse = properties.get("diffuseColor")
    if (
        isinstance(raw_diffuse, (str, bytes))
        or not isinstance(raw_diffuse, Sequence)
        or len(raw_diffuse) != 3
    ):
        return None
    diffuse: list[float] = []
    for value in raw_diffuse:
        parsed = _unit_number_or_none(value)
        if parsed is None:
            return None
        diffuse.append(parsed)
    metallic = _unit_number_or_none(properties.get("metallic"))
    roughness = _unit_number_or_none(properties.get("roughness"))
    opacity = _unit_number_or_none(properties.get("opacity"))
    if metallic is None or roughness is None or opacity is None:
        return None
    if opacity < _SOURCE_ACCENT_OPACITY_FLOOR:
        return None

    srgb_interpretation = tuple(_linear_channel_to_srgb(value) for value in diffuse)
    raw_color = _rgb_color_label(diffuse)
    linear_color = _rgb_color_label(srgb_interpretation)
    raw_saturation = _rgb_saturation(diffuse)
    linear_saturation = _rgb_saturation(srgb_interpretation)
    if (
        raw_color != linear_color
        or raw_color not in _SOURCE_ACCENT_CHROMATIC_COLORS
        or raw_saturation < _SOURCE_ACCENT_MIN_SATURATION
        or linear_saturation < _SOURCE_ACCENT_MIN_SATURATION
    ):
        return None

    signature = (
        "UsdPreviewSurface",
        *(round(value, 6) for value in diffuse),
        round(metallic, 6),
        round(roughness, 6),
        round(opacity, 6),
    )
    return signature, {
        "shader_id": "UsdPreviewSurface",
        "diffuse_color": [round(value, 8) for value in diffuse],
        "metallic": round(metallic, 8),
        "roughness": round(roughness, 8),
        "opacity": round(opacity, 8),
        "raw_color_family": raw_color,
        "linear_to_srgb_color_family": linear_color,
        "raw_saturation": round(raw_saturation, 8),
        "linear_to_srgb_saturation": round(linear_saturation, 8),
    }


def _repeated_geometry_signature(
    part: Mapping[str, Any],
) -> tuple[int, int, tuple[float, float, float]] | None:
    point_count = part.get("point_count")
    face_count = part.get("face_count")
    if (
        isinstance(point_count, bool)
        or not isinstance(point_count, int)
        or point_count <= 0
        or isinstance(face_count, bool)
        or not isinstance(face_count, int)
        or face_count <= 0
    ):
        return None
    bounds = _bbox(part)
    if bounds is None:
        return None
    minimum, maximum = bounds
    extents = tuple(
        sorted(round(maximum[index] - minimum[index], 3) for index in range(3))
    )
    return point_count, face_count, extents


def _is_corroborated_unresolved_pixel_chromatic_group(
    group: Mapping[str, Any],
    *,
    canonical_color: str,
    source_view_ids: Sequence[str],
) -> bool:
    """Validate the narrow unresolved-pixel multiview fusion contract."""

    if (
        not is_verified_unresolved_pixel_chromatic_group(group)
        or group.get("association_basis")
        != UNRESOLVED_PIXEL_CHROMATIC_ASSOCIATION
        or canonical_color not in _SOURCE_ACCENT_CHROMATIC_COLORS
        or len(set(source_view_ids)) < 2
    ):
        return False
    return (
        fusion_color_label(str(group.get("base_color", ""))) == canonical_color
        and list(group.get("source_view_ids", [])) == list(source_view_ids)
    )


def _canonical_multiview_chromatic_groups(
    palette_fusion: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if palette_fusion.get("schema_version") != PALETTE_FUSION_SCHEMA_VERSION:
        raise PolicyExactCoverError(
            "palette fusion has an unsupported schema_version"
        )
    canonical = _object(
        palette_fusion.get("canonical_palette"),
        "palette_fusion.canonical_palette",
    )
    if canonical.get("schema_version") != CANONICAL_PALETTE_SCHEMA_VERSION:
        raise PolicyExactCoverError(
            "canonical palette has an unsupported schema_version"
        )
    raw_groups = _array(
        canonical.get("groups"),
        "palette_fusion.canonical_palette.groups",
    )
    groups_by_color: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for index, raw_group in enumerate(raw_groups):
        group = _object(raw_group, f"canonical_palette.groups[{index}]")
        group_id = _text(group.get("group_id"), f"canonical group {index}.group_id")
        if group_id in seen_ids:
            raise PolicyExactCoverError(f"duplicate canonical group_id: {group_id}")
        seen_ids.add(group_id)
        distinct_view_count = group.get("distinct_view_count")
        singleton = group.get("singleton")
        source_view_ids = _array(
            group.get("source_view_ids"),
            f"canonical group {group_id}.source_view_ids",
        )
        normalized_views = [
            _text(value, f"canonical group {group_id}.source_view_ids")
            for value in source_view_ids
        ]
        if (
            isinstance(distinct_view_count, bool)
            or not isinstance(distinct_view_count, int)
            or distinct_view_count != len(set(normalized_views))
            or not isinstance(singleton, bool)
        ):
            raise PolicyExactCoverError(
                f"canonical group {group_id} has inconsistent view support"
            )
        color = fusion_color_label(
            _text(group.get("base_color"), f"canonical group {group_id}.base_color")
        )
        family = str(group.get("family_hint", "")).casefold()
        unresolved_pixel_corroborated = (
            _is_corroborated_unresolved_pixel_chromatic_group(
                group,
                canonical_color=color,
                source_view_ids=normalized_views,
            )
        )
        if (
            not singleton
            and distinct_view_count >= 2
            and color in _SOURCE_ACCENT_CHROMATIC_COLORS
            and (
                family not in {"", "other", "unknown"}
                or unresolved_pixel_corroborated
            )
        ):
            groups_by_color[color].append(group)
    # A coarse source colour cannot choose between two different canonical
    # appearances.  Only a globally unique multiview chromatic family is safe.
    return {
        color: groups[0] for color, groups in groups_by_color.items() if len(groups) == 1
    }


def _reviewable_canonical_group_evidence(
    palette_fusion: Mapping[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Return hash-bound multiview groups that may retain review target lineage.

    A canonical group is only a target namespace here.  This helper does not
    approve a part mapping or authorize the group's selected material.  In
    particular, singleton palette observations are excluded: a weak per-part
    mapping must not turn one reference observation into an automatic label.
    """

    if palette_fusion is None:
        return {}, {}
    if palette_fusion.get("schema_version") != PALETTE_FUSION_SCHEMA_VERSION:
        raise PolicyExactCoverError(
            "palette fusion has an unsupported schema_version"
        )
    canonical = _object(
        palette_fusion.get("canonical_palette"),
        "palette_fusion.canonical_palette",
    )
    if canonical.get("schema_version") != CANONICAL_PALETTE_SCHEMA_VERSION:
        raise PolicyExactCoverError(
            "canonical palette has an unsupported schema_version"
        )

    groups: dict[str, dict[str, Any]] = {}
    rejection_counts: Counter[str] = Counter()
    seen: set[str] = set()
    for index, raw in enumerate(
        _array(canonical.get("groups"), "canonical_palette.groups")
    ):
        group = _object(raw, f"canonical_palette.groups[{index}]")
        group_id = _text(
            group.get("group_id"),
            f"canonical_palette.groups[{index}].group_id",
        )
        if group_id in seen:
            raise PolicyExactCoverError(
                f"duplicate canonical palette group: {group_id}"
            )
        seen.add(group_id)
        source_view_ids = [
            _text(
                value,
                f"canonical palette group {group_id}.source_view_ids[{view_index}]",
            )
            for view_index, value in enumerate(
                _array(
                    group.get("source_view_ids"),
                    f"canonical palette group {group_id}.source_view_ids",
                )
            )
        ]
        distinct_view_count = group.get("distinct_view_count")
        singleton = group.get("singleton")
        if (
            isinstance(distinct_view_count, bool)
            or not isinstance(distinct_view_count, int)
            or distinct_view_count != len(set(source_view_ids))
            or not isinstance(singleton, bool)
        ):
            raise PolicyExactCoverError(
                f"canonical group {group_id} has inconsistent view support"
            )
        if singleton or distinct_view_count < 2:
            rejection_counts["CANONICAL_GROUP_NOT_MULTIVIEW"] += 1
            continue
        canonical_confidence = _unit_or_zero(group.get("confidence"))
        if canonical_confidence < 0.6:
            rejection_counts["CANONICAL_GROUP_BELOW_REVIEW_CONFIDENCE"] += 1
            continue
        groups[group_id] = {
            "canonical_group_id": group_id,
            "source_view_ids": sorted(set(source_view_ids)),
            "distinct_view_count": distinct_view_count,
            "canonical_confidence": canonical_confidence,
            "singleton": False,
        }
    return groups, dict(sorted(rejection_counts.items()))


def _review_mapping_confidence_floor(
    confidence_gate: Mapping[str, Any],
) -> float:
    """Return a conservative floor for retaining a non-authoring target."""

    policy = confidence_gate.get("policy")
    if not isinstance(policy, Mapping):
        return 0.6
    configured = policy.get("review_mapping_confidence")
    parsed = _unit_or_zero(configured)
    # Older audit fixtures omit the policy; malformed or weakened policy
    # values must never broaden lineage retention below the production floor.
    return max(0.6, parsed)


def _retain_non_authoring_mapping_lineage(
    *,
    output: Mapping[str, dict[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
    gate_records: Mapping[str, Mapping[str, Any]],
    confidence_gate: Mapping[str, Any],
    palette_fusion: Mapping[str, Any] | None,
    excluded_part_ids: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Retain a REVIEW mapping strictly as a provisional audit hypothesis.

    The exact-cover material remains ``policy_fallback`` with confidence zero.
    REVIEW evidence must not populate ``provenance.canonical_group_id`` because
    that field is an authoring/tournament target consumed downstream.  The
    staged MDL and proposed canonical group therefore remain quarantined under
    explicitly provisional fields.
    """

    groups, group_rejections = _reviewable_canonical_group_evidence(
        palette_fusion
    )
    confidence_floor = _review_mapping_confidence_floor(confidence_gate)
    retained_by_group: dict[str, list[str]] = defaultdict(list)
    rejection_counts: Counter[str] = Counter(group_rejections)

    if palette_fusion is None:
        return {
            "state": "not_requested_without_palette_fusion",
            "authoring_effect": "none",
            "downstream_target_eligible": False,
            "retained_part_count": 0,
            "retained_group_ids": [],
            "part_ids_by_group": {},
            "rejection_counts": {},
        }

    for part_id in sorted(output):
        if part_id in excluded_part_ids:
            rejection_counts["PART_ID_UNOBSERVED"] += 1
            continue
        assignment = output[part_id]
        if assignment.get("status") != FALLBACK_STATUS:
            continue
        candidate = candidates.get(part_id)
        if candidate is None:
            rejection_counts["NO_STAGED_CANDIDATE"] += 1
            continue
        gate_record = gate_records[part_id]
        if gate_record.get("decision") != "review":
            rejection_counts["GATE_DECISION_NOT_REVIEW"] += 1
            continue
        mapping = gate_record.get("mapping")
        if not isinstance(mapping, Mapping):
            rejection_counts["TARGET_MAPPING_UNAVAILABLE"] += 1
            continue
        mapping_status = mapping.get("status")
        if mapping_status not in {"matched", "review"}:
            rejection_counts["TARGET_MAPPING_NOT_REVIEWABLE"] += 1
            continue
        mapping_confidence = _unit_or_zero(mapping.get("confidence"))
        if mapping_confidence < confidence_floor:
            rejection_counts["TARGET_MAPPING_BELOW_REVIEW_CONFIDENCE"] += 1
            continue
        group_id = mapping.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            rejection_counts["TARGET_CANONICAL_GROUP_UNAVAILABLE"] += 1
            continue
        group_evidence = groups.get(group_id)
        if group_evidence is None:
            rejection_counts["TARGET_CANONICAL_GROUP_NOT_MULTIVIEW"] += 1
            continue

        provenance = _object(
            assignment.get("provenance"),
            f"policy fallback {part_id}.provenance",
        )
        existing_group_id = provenance.get("canonical_group_id")
        if existing_group_id is not None:
            rejection_counts["AUTHORITATIVE_CANONICAL_GROUP_ALREADY_PRESENT"] += 1
            continue
        existing_provisional_group_id = provenance.get(
            "provisional_canonical_group_id"
        )
        if existing_provisional_group_id not in {None, group_id}:
            raise PolicyExactCoverError(
                f"policy fallback {part_id} has conflicting provisional groups: "
                f"{existing_provisional_group_id!r} and {group_id!r}"
            )
        if (
            provenance.get("target_mapping") is not None
            or provenance.get("provisional_target_mapping") is not None
            or provenance.get("evidence_lineage") is not None
        ):
            raise PolicyExactCoverError(
                f"policy fallback {part_id} already contains target lineage"
            )

        model = gate_record.get("model")
        model = model if isinstance(model, Mapping) else {}
        material_choice = gate_record.get("material_choice")
        material_choice = (
            material_choice if isinstance(material_choice, Mapping) else {}
        )
        mutable_provenance = copy.deepcopy(dict(provenance))
        mutable_provenance["provisional_canonical_group_id"] = group_id
        mutable_provenance["provisional_target_mapping"] = {
            "mode": "confidence_gate_review_hypothesis/v2",
            "provisional_canonical_group_id": group_id,
            "batch_id": mapping.get("batch_id"),
            "status": mapping_status,
            "confidence": mapping_confidence,
            "reason_code": mapping.get("reason_code"),
            "gate_decision": "review",
            "authoring_eligible": False,
            "downstream_target_eligible": False,
            "retained_for": "audit_only",
        }
        mutable_provenance["evidence_lineage"] = {
            "mode": "provisional_review_mapping_evidence/v2",
            "staged_candidate": {
                "material_id": candidate.get("material_id"),
                "status": candidate.get("status"),
                "confidence": _unit_or_zero(candidate.get("confidence")),
                "evidence_views": list(candidate.get("evidence_views", [])),
            },
            "confidence_gate": {
                "decision": "review",
                "reason_codes": list(gate_record.get("reason_codes", [])),
                "threshold_profile": gate_record.get("threshold_profile"),
                "independent_reference_count": model.get(
                    "independent_reference_count"
                ),
                "independent_reference_views": list(
                    model.get("independent_reference_views", [])
                ),
                "reference_evidence_source": model.get(
                    "reference_evidence_source"
                ),
            },
            "provisional_target_hypothesis": copy.deepcopy(group_evidence),
            "material_selection": {
                "resolved_material_id": material_choice.get(
                    "resolved_material_id"
                ),
                "confirmation_basis": material_choice.get(
                    "confirmation_basis"
                ),
                "confirmed": material_choice.get("confirmed"),
            },
            "authoring_effect": "none",
            "authoring_eligible": False,
            "downstream_target_eligible": False,
        }
        reason_codes = list(mutable_provenance.get("reason_codes", []))
        reason_code = "PROVISIONAL_CANONICAL_GROUP_HYPOTHESIS_RETAINED"
        if reason_code not in reason_codes:
            reason_codes.append(reason_code)
        mutable_provenance["reason_codes"] = reason_codes
        assignment["provenance"] = mutable_provenance
        retained_by_group[group_id].append(part_id)

    retained_part_count = sum(len(part_ids) for part_ids in retained_by_group.values())
    return {
        "state": (
            "provisional_review_hypotheses_retained"
            if retained_part_count
            else "no_reviewable_hypotheses"
        ),
        "authoring_effect": "none",
        "downstream_target_eligible": False,
        "mapping_confidence_floor": confidence_floor,
        "retained_part_count": retained_part_count,
        "retained_group_ids": sorted(retained_by_group),
        "part_ids_by_group": {
            group_id: sorted(part_ids)
            for group_id, part_ids in sorted(retained_by_group.items())
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }


def _propagate_non_authoring_mapping_lineage(
    *,
    parts: Sequence[Mapping[str, Any]],
    output: Mapping[str, dict[str, Any]],
    cluster_keys: Sequence[str],
    excluded_part_ids: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Propagate only a provisional audit hypothesis across exact identities.

    This never authors ``canonical_group_id``.  Consequently an
    ``existing_visual_material`` or geometry identity may help audit repeated
    REVIEW hypotheses, but can never manufacture an authoritative downstream
    tournament target.
    """

    propagated_by_group: dict[str, list[str]] = defaultdict(list)
    conflicts: list[dict[str, Any]] = []
    applied_cluster_counts: Counter[str] = Counter()
    for cluster_key in cluster_keys:
        clusters: dict[str, list[str]] = defaultdict(list)
        for part in parts:
            value = _cluster_value(part, cluster_key)
            if value is not None:
                clusters[value].append(str(part["part_id"]))

        for cluster_value in sorted(clusters):
            part_ids = sorted(clusters[cluster_value])
            if len(part_ids) < 2:
                continue
            sources: list[tuple[str, Mapping[str, Any]]] = []
            for part_id in part_ids:
                if part_id in excluded_part_ids:
                    continue
                assignment = output[part_id]
                provenance = assignment.get("provenance")
                if not isinstance(provenance, Mapping):
                    continue
                target_mapping = provenance.get("provisional_target_mapping")
                if (
                    isinstance(target_mapping, Mapping)
                    and target_mapping.get("mode")
                    == "confidence_gate_review_hypothesis/v2"
                    and target_mapping.get("authoring_eligible") is False
                    and target_mapping.get("downstream_target_eligible") is False
                    and isinstance(
                        provenance.get("provisional_canonical_group_id"),
                        str,
                    )
                ):
                    sources.append((part_id, provenance))
            source_group_ids = sorted(
                {
                    str(provenance["provisional_canonical_group_id"])
                    for _part_id, provenance in sources
                }
            )
            if not source_group_ids:
                continue
            if len(source_group_ids) != 1:
                conflicts.append(
                    {
                        "cluster_key": cluster_key,
                        "cluster_value_sha256": _canonical_sha256(
                            {
                                "cluster_key": cluster_key,
                                "cluster_value": cluster_value,
                            }
                        ),
                        "part_ids": part_ids,
                        "source_part_ids": [
                            part_id for part_id, _provenance in sources
                        ],
                        "provisional_group_ids": source_group_ids,
                        "action": "no_hypothesis_propagation",
                        "reason_code": "REVIEW_HYPOTHESIS_GROUP_CONFLICT",
                    }
                )
                continue

            group_id = source_group_ids[0]
            source_part_ids = [
                part_id
                for part_id, provenance in sources
                if provenance.get("provisional_canonical_group_id") == group_id
            ]
            source_mapping_confidences = [
                _unit_or_zero(
                    provenance["provisional_target_mapping"].get("confidence")
                )
                for _part_id, provenance in sources
                if isinstance(
                    provenance.get("provisional_target_mapping"),
                    Mapping,
                )
            ]
            source_lineage_sha256s = [
                _canonical_sha256(
                    _object(
                        provenance.get("evidence_lineage"),
                        f"policy fallback {part_id}.evidence_lineage",
                    )
                )
                for part_id, provenance in sources
            ]
            cluster_digest = _canonical_sha256(
                {
                    "cluster_key": cluster_key,
                    "cluster_value": cluster_value,
                }
            )
            for part_id in part_ids:
                if part_id in excluded_part_ids:
                    continue
                assignment = output[part_id]
                if assignment.get("status") != FALLBACK_STATUS:
                    continue
                provenance = _object(
                    assignment.get("provenance"),
                    f"policy fallback {part_id}.provenance",
                )
                authoritative_group_id = provenance.get("canonical_group_id")
                if authoritative_group_id is not None:
                    conflicts.append(
                        {
                            "cluster_key": cluster_key,
                            "cluster_value_sha256": cluster_digest,
                            "part_ids": [part_id],
                            "source_part_ids": source_part_ids,
                            "authoritative_group_id": str(
                                authoritative_group_id
                            ),
                            "provisional_group_id": group_id,
                            "action": "preserve_authoritative_target",
                            "reason_code": (
                                "PROVISIONAL_HYPOTHESIS_CANNOT_OVERRIDE_"
                                "AUTHORITATIVE_GROUP"
                            ),
                        }
                    )
                    continue
                existing_group_id = provenance.get(
                    "provisional_canonical_group_id"
                )
                if existing_group_id is not None:
                    if existing_group_id != group_id:
                        conflicts.append(
                            {
                                "cluster_key": cluster_key,
                                "cluster_value_sha256": cluster_digest,
                                "part_ids": [part_id],
                                "source_part_ids": source_part_ids,
                                "provisional_group_ids": sorted(
                                    {str(existing_group_id), group_id}
                                ),
                                "action": "preserve_existing_hypothesis",
                                "reason_code": (
                                    "PART_PROVISIONAL_GROUP_CONFLICT_WITH_"
                                    "EXACT_IDENTITY"
                                ),
                            }
                        )
                    continue
                mutable_provenance = copy.deepcopy(dict(provenance))
                mutable_provenance["provisional_canonical_group_id"] = group_id
                mutable_provenance["provisional_target_mapping"] = {
                    "mode": "exact_identity_review_hypothesis_propagation/v2",
                    "provisional_canonical_group_id": group_id,
                    "status": "review",
                    "confidence": min(source_mapping_confidences),
                    "gate_decision": "review",
                    "authoring_eligible": False,
                    "downstream_target_eligible": False,
                    "retained_for": "audit_only",
                    "identity_cluster_key": cluster_key,
                    "identity_cluster_sha256": cluster_digest,
                    "source_part_ids": source_part_ids,
                }
                mutable_provenance["evidence_lineage"] = {
                    "mode": "provisional_exact_identity_hypothesis/v2",
                    "provisional_canonical_group_id": group_id,
                    "source_part_ids": source_part_ids,
                    "source_evidence_lineage_sha256s": source_lineage_sha256s,
                    "identity_cluster_key": cluster_key,
                    "identity_cluster_sha256": cluster_digest,
                    "propagated_fields": ["provisional_canonical_group_id"],
                    "excluded_fields": [
                        "canonical_group_id",
                        "material_id",
                        "semantic",
                        "parameters",
                        "confidence",
                        "evidence_views",
                    ],
                    "authoring_effect": "none",
                    "authoring_eligible": False,
                    "downstream_target_eligible": False,
                }
                reason_codes = list(mutable_provenance.get("reason_codes", []))
                reason_code = (
                    "PROVISIONAL_GROUP_HYPOTHESIS_PROPAGATED_EXACT_IDENTITY"
                )
                if reason_code not in reason_codes:
                    reason_codes.append(reason_code)
                mutable_provenance["reason_codes"] = reason_codes
                assignment["provenance"] = mutable_provenance
                propagated_by_group[group_id].append(part_id)
                applied_cluster_counts[cluster_key] += 1

    propagated_count = sum(
        len(part_ids) for part_ids in propagated_by_group.values()
    )
    return {
        "state": (
            "exact_identity_hypotheses_propagated"
            if propagated_count
            else "no_exact_identity_hypothesis_expansion"
        ),
        "authoring_effect": "none",
        "downstream_target_eligible": False,
        "configured_cluster_keys": list(cluster_keys),
        "applied_target_count_by_cluster_key": dict(
            sorted(applied_cluster_counts.items())
        ),
        "propagated_part_count": propagated_count,
        "propagated_group_ids": sorted(propagated_by_group),
        "part_ids_by_group": {
            group_id: sorted(part_ids)
            for group_id, part_ids in sorted(propagated_by_group.items())
        },
        "conflicts": conflicts,
    }


def _corroborated_source_visual_parts(
    *,
    parts: Sequence[Mapping[str, Any]],
    palette_fusion: Mapping[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Find rare repeated source accents independently corroborated by photos.

    This is a preservation locator, not a material classifier.  It may retain
    the exact hash-bound source binding, but it never converts the colour into
    an MDL family or raises its evidence confidence.
    """

    thresholds = {
        "maximum_registry_fraction": _SOURCE_ACCENT_MAX_REGISTRY_FRACTION,
        "minimum_source_signature_count": _SOURCE_ACCENT_MIN_SIGNATURE_COUNT,
        "minimum_repeated_geometry_count": (
            _SOURCE_ACCENT_MIN_GEOMETRY_REPEAT_COUNT
        ),
        "minimum_raw_and_linear_saturation": _SOURCE_ACCENT_MIN_SATURATION,
        "minimum_opacity": _SOURCE_ACCENT_OPACITY_FLOOR,
    }
    if palette_fusion is None:
        return {}, {
            "state": "not_requested_without_palette_fusion",
            "thresholds": thresholds,
            "eligible_part_ids": [],
            "groups": [],
        }

    canonical_by_color = _canonical_multiview_chromatic_groups(palette_fusion)
    signature_clusters: dict[
        tuple[Any, ...], list[tuple[Mapping[str, Any], dict[str, Any]]]
    ] = defaultdict(list)
    for part in parts:
        if _valid_source_visual_material_path(
            part.get("existing_visual_material")
        ) is None:
            continue
        parsed = _source_visual_signature(part)
        if parsed is None:
            continue
        signature, signature_record = parsed
        color = str(signature_record["raw_color_family"])
        canonical_group = canonical_by_color.get(color)
        if canonical_group is None:
            continue
        signature_clusters[signature].append((part, signature_record))

    maximum_signature_count = math.floor(
        len(parts) * _SOURCE_ACCENT_MAX_REGISTRY_FRACTION
    )
    candidates_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejection_counts: Counter[str] = Counter()
    for signature, members in sorted(
        signature_clusters.items(), key=lambda item: repr(item[0])
    ):
        signature_count = len(members)
        if signature_count < _SOURCE_ACCENT_MIN_SIGNATURE_COUNT:
            rejection_counts["SOURCE_SIGNATURE_TOO_SMALL"] += signature_count
            continue
        if signature_count > maximum_signature_count:
            rejection_counts["SOURCE_SIGNATURE_TOO_COMMON"] += signature_count
            continue
        color = str(members[0][1]["raw_color_family"])
        canonical_group = canonical_by_color[color]
        group_id = str(canonical_group["group_id"])
        geometry_clusters: dict[
            tuple[int, int, tuple[float, float, float]],
            list[Mapping[str, Any]],
        ] = defaultdict(list)
        for part, _signature_record in members:
            geometry_signature = _repeated_geometry_signature(part)
            if geometry_signature is not None:
                geometry_clusters[geometry_signature].append(part)
        repeated_clusters = {
            geometry_signature: cluster
            for geometry_signature, cluster in geometry_clusters.items()
            if len(cluster) >= _SOURCE_ACCENT_MIN_GEOMETRY_REPEAT_COUNT
        }
        geometry_anchor_parts = [
            part
            for geometry_signature in sorted(repeated_clusters, key=repr)
            for part in sorted(
                repeated_clusters[geometry_signature],
                key=lambda item: str(item["part_id"]),
            )
        ]
        if len(geometry_anchor_parts) < _SOURCE_ACCENT_MIN_GEOMETRY_REPEAT_COUNT:
            rejection_counts["NO_REPEATED_GEOMETRY_COHORT"] += signature_count
            continue
        # Repeated geometry proves that the rare source-display signature is
        # an intentional appearance class rather than a one-off CAD selection
        # colour.  Once that signature is also uniquely corroborated by a
        # multiview reference-palette colour, include every exact signature
        # member.  Requiring every member to repeat geometrically excluded
        # larger caps/housings which deliberately shared the same authored
        # appearance as their repeated small inserts.
        signature_members = sorted(
            (part for part, _signature_record in members),
            key=lambda item: str(item["part_id"]),
        )
        signature_payload = {
            "shader_id": signature[0],
            "diffuse_color": list(signature[1:4]),
            "metallic": signature[4],
            "roughness": signature[5],
            "opacity": signature[6],
        }
        geometry_records = []
        for geometry_signature in sorted(repeated_clusters, key=repr):
            point_count, face_count, extents = geometry_signature
            cluster = repeated_clusters[geometry_signature]
            geometry_payload = {
                "point_count": point_count,
                "face_count": face_count,
                "sorted_bbox_extents": list(extents),
            }
            geometry_records.append(
                {
                    **geometry_payload,
                    "geometry_signature_sha256": _canonical_sha256(geometry_payload),
                    "repeat_count": len(cluster),
                    "part_ids": sorted(str(part["part_id"]) for part in cluster),
                }
            )
        candidates_by_group[group_id].append(
            {
                "group_id": group_id,
                "canonical_color_family": color,
                "canonical_group_association_basis": canonical_group.get(
                    "association_basis"
                ),
                "canonical_source_view_ids": sorted(
                    str(value) for value in canonical_group["source_view_ids"]
                ),
                "source_visual_signature": copy.deepcopy(members[0][1]),
                "source_visual_signature_sha256": _canonical_sha256(
                    signature_payload
                ),
                "source_signature_count": signature_count,
                "registry_fraction": round(signature_count / len(parts), 8),
                "geometry_cohorts": geometry_records,
                "geometry_anchor_part_ids": sorted(
                    str(part["part_id"]) for part in geometry_anchor_parts
                ),
                "signature_member_part_ids": sorted(
                    str(part["part_id"]) for part in signature_members
                ),
                "eligible_part_ids": sorted(
                    str(part["part_id"]) for part in signature_members
                ),
            }
        )

    result: dict[str, dict[str, Any]] = {}
    accepted_groups: list[dict[str, Any]] = []
    for group_id in sorted(candidates_by_group):
        candidates = candidates_by_group[group_id]
        if len(candidates) != 1:
            rejection_counts["AMBIGUOUS_SOURCE_SIGNATURES_FOR_CANONICAL_GROUP"] += sum(
                len(candidate["eligible_part_ids"]) for candidate in candidates
            )
            continue
        candidate = candidates[0]
        accepted_groups.append(candidate)
        for part_id in candidate["eligible_part_ids"]:
            geometry_record = next(
                (
                    record
                    for record in candidate["geometry_cohorts"]
                    if part_id in record["part_ids"]
                ),
                None,
            )
            result[part_id] = {
                "canonical_group_id": group_id,
                "canonical_color_family": candidate["canonical_color_family"],
                "canonical_group_association_basis": candidate[
                    "canonical_group_association_basis"
                ],
                "canonical_source_view_ids": list(
                    candidate["canonical_source_view_ids"]
                ),
                "source_visual_signature_sha256": candidate[
                    "source_visual_signature_sha256"
                ],
                "source_signature_count": candidate["source_signature_count"],
                "registry_fraction": candidate["registry_fraction"],
                **(
                    {
                        "geometry_signature_sha256": geometry_record[
                            "geometry_signature_sha256"
                        ],
                        "geometry_repeat_count": geometry_record["repeat_count"],
                    }
                    if geometry_record is not None
                    else {
                        "signature_expansion_basis": (
                            "exact_source_signature_with_repeated_geometry_anchor"
                        ),
                        "signature_expansion_anchor_part_ids": list(
                            candidate["geometry_anchor_part_ids"]
                        ),
                    }
                ),
            }
    return result, {
        "state": (
            "corroborated_source_accents_found"
            if result
            else "no_corroborated_source_accents"
        ),
        "thresholds": thresholds,
        "maximum_source_signature_count": maximum_signature_count,
        "canonical_unique_multiview_chromatic_group_ids": sorted(
            str(group["group_id"]) for group in canonical_by_color.values()
        ),
        "eligible_part_ids": sorted(result),
        "groups": accepted_groups,
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }


def _direct_gate_assignment(
    assignment: Mapping[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(assignment))
    result["provenance"] = {
        "tier": "gate_auto",
        "reason_codes": ["CONFIDENCE_GATE_AUTO"],
        "output_confidence_basis": "confidence gate",
        "sources": [
            {
                "part_id": assignment["part_id"],
                "source_status": assignment.get("status"),
                "source_confidence": _unit_or_zero(assignment.get("confidence")),
                "source_evidence_views": list(assignment.get("evidence_views", [])),
            }
        ],
    }
    return result


def _direct_base_assignment(assignment: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(assignment))
    upstream = result.get("provenance")
    result["provenance"] = {
        "tier": "autonomous_base_plan",
        "reason_codes": ["VERIFIED_AUTONOMOUS_BASE_PLAN"],
        "output_confidence_basis": "upstream autonomous material plan",
        "sources": [
            {
                "part_id": assignment["part_id"],
                "source_status": assignment.get("status"),
                "source_confidence": _unit_or_zero(assignment.get("confidence")),
                "source_evidence_views": list(assignment.get("evidence_views", [])),
            }
        ],
        "upstream_provenance": (
            copy.deepcopy(dict(upstream)) if isinstance(upstream, Mapping) else None
        ),
    }
    return result


def _explicit_face_subsets_match_source_topology(
    value: Any,
    source_subsets: Sequence[Mapping[str, Any]],
) -> bool:
    """Return true only for an exact, name-bound source topology match."""

    if not isinstance(value, list) or not value:
        return False
    planned_by_name: dict[str, Mapping[str, Any]] = {}
    for raw in value:
        if not isinstance(raw, Mapping):
            return False
        name = raw.get("subset_name")
        indices = raw.get("face_indices")
        if (
            not isinstance(name, str)
            or not name
            or name in planned_by_name
            or not isinstance(indices, list)
            or any(
                isinstance(face_index, bool) or not isinstance(face_index, int)
                for face_index in indices
            )
        ):
            return False
        planned_by_name[name] = raw
    source_by_name = {
        str(subset["subset_name"]): subset for subset in source_subsets
    }
    if set(planned_by_name) != set(source_by_name):
        return False
    return all(
        list(planned_by_name[name].get("face_indices", []))
        == list(source_by_name[name]["face_indices"])
        for name in source_by_name
    )


def _set_source_subset_contract_provenance(
    assignment: dict[str, Any],
    *,
    action: str,
    source_subsets: Sequence[Mapping[str, Any]],
) -> None:
    provenance = assignment.get("provenance")
    if not isinstance(provenance, Mapping):
        raise PolicyExactCoverError(
            f"assignment {assignment.get('part_id')} has no provenance object"
        )
    mutable_provenance = copy.deepcopy(dict(provenance))
    mutable_provenance["source_material_bind_subset_contract"] = {
        "action": action,
        "source_subset_prim_paths": [
            str(subset["subset_prim_path"]) for subset in source_subsets
        ],
        "source_subset_binding_sha256": [
            str(subset[SOURCE_SUBSET_HASH_FIELD]) for subset in source_subsets
        ],
    }
    assignment["provenance"] = mutable_provenance


def _prepare_source_subset_contracts(
    *,
    assignments: Sequence[dict[str, Any]],
    parts_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Collapse unsafe explicit subsets before parameter tuning.

    The collapse is visual, not geometric: the source subsets remain exactly
    as authored and are later rebound to the selected parent material.
    """

    audit = {
        "source_visual_preserve_part_ids": [],
        "parent_material_expansion_part_ids": [],
        "explicit_topology_match_part_ids": [],
        "explicit_topology_collapse_part_ids": [],
    }
    for assignment in assignments:
        part_id = str(assignment["part_id"])
        source_subsets = list(
            parts_by_id[part_id].get(SOURCE_MATERIAL_BIND_SUBSETS_FIELD, [])
        )
        if not source_subsets:
            continue
        if assignment.get("apply_action") == SOURCE_VISUAL_PRESERVE_ACTION:
            _set_source_subset_contract_provenance(
                assignment,
                action="source_visual_preserve_noop",
                source_subsets=source_subsets,
            )
            audit["source_visual_preserve_part_ids"].append(part_id)
            continue

        explicit_subsets = assignment.get("face_subsets")
        if explicit_subsets:
            if _explicit_face_subsets_match_source_topology(
                explicit_subsets, source_subsets
            ):
                by_name = {
                    str(subset["subset_name"]): copy.deepcopy(dict(subset))
                    for subset in explicit_subsets
                }
                assignment["face_subsets"] = [
                    by_name[str(source["subset_name"])]
                    for source in source_subsets
                ]
                _set_source_subset_contract_provenance(
                    assignment,
                    action="explicit_topology_match",
                    source_subsets=source_subsets,
                )
                audit["explicit_topology_match_part_ids"].append(part_id)
                continue

            assignment.pop("face_subsets", None)
            assignment.pop("preserve_parent_material_binding", None)
            provenance = assignment.get("provenance")
            if isinstance(provenance, Mapping):
                mutable_provenance = copy.deepcopy(dict(provenance))
                reason_codes = list(mutable_provenance.get("reason_codes", []))
                collapse_reason = (
                    "SOURCE_FACE_SUBSET_TOPOLOGY_MISMATCH_COLLAPSED_TO_PARENT"
                )
                if collapse_reason not in reason_codes:
                    reason_codes.append(collapse_reason)
                mutable_provenance["reason_codes"] = reason_codes
                assignment["provenance"] = mutable_provenance
            audit["explicit_topology_collapse_part_ids"].append(part_id)
        else:
            # An empty list is semantically a whole-Mesh selection and is not
            # accepted by the apply schema as an authored subset array.
            assignment.pop("face_subsets", None)
            assignment.pop("preserve_parent_material_binding", None)
            audit["parent_material_expansion_part_ids"].append(part_id)
    return audit


def _materialize_source_subset_rebinds(
    *,
    assignments: Sequence[dict[str, Any]],
    parts_by_id: Mapping[str, Mapping[str, Any]],
    audit: Mapping[str, list[str]],
) -> None:
    """Make every source subset explicitly inherit the final parent MDL."""

    collapsed_ids = set(audit["explicit_topology_collapse_part_ids"])
    expanded_ids = set(audit["parent_material_expansion_part_ids"])
    for assignment in assignments:
        part_id = str(assignment["part_id"])
        if part_id not in collapsed_ids and part_id not in expanded_ids:
            continue
        source_subsets = list(
            parts_by_id[part_id][SOURCE_MATERIAL_BIND_SUBSETS_FIELD]
        )
        parameters = assignment.get("parameters")
        face_subsets: list[dict[str, Any]] = []
        for source_subset in source_subsets:
            face_subset: dict[str, Any] = {
                "subset_name": str(source_subset["subset_name"]),
                "material_id": str(assignment["material_id"]),
                "face_indices": list(source_subset["face_indices"]),
            }
            if parameters is not None:
                face_subset["parameters"] = copy.deepcopy(parameters)
            face_subsets.append(face_subset)
        assignment["face_subsets"] = face_subsets
        _set_source_subset_contract_provenance(
            assignment,
            action=(
                "explicit_topology_mismatch_collapse_to_parent"
                if part_id in collapsed_ids
                else "parent_material_expansion"
            ),
            source_subsets=source_subsets,
        )


def _cluster_propagate(
    *,
    parts: Sequence[Mapping[str, Any]],
    source_assignments: Mapping[str, Mapping[str, Any]],
    output: dict[str, dict[str, Any]],
    cluster_key: str,
    tier: str,
    conflicts: list[dict[str, Any]],
    minimum_cluster_part_count: int = 1,
) -> None:
    clusters: dict[str, list[str]] = defaultdict(list)
    for part in parts:
        value = _cluster_value(part, cluster_key)
        if value is not None:
            clusters[value].append(str(part["part_id"]))
    for value in sorted(clusters):
        part_ids = clusters[value]
        if len(part_ids) < minimum_cluster_part_count:
            continue
        sources = [
            source_assignments[part_id]
            for part_id in part_ids
            if part_id in source_assignments
        ]
        materials = sorted({str(source["material_id"]) for source in sources})
        if len(materials) > 1:
            conflicts.append(
                {
                    "cluster_key": cluster_key,
                    "cluster_value": value,
                    "part_ids": sorted(part_ids),
                    "candidate_material_ids": materials,
                    "source_part_ids": sorted(
                        str(source["part_id"]) for source in sources
                    ),
                    "action": "no_propagation",
                }
            )
            continue
        if not sources:
            continue
        if any(source.get("face_subsets") for source in sources):
            conflicts.append(
                {
                    "cluster_key": cluster_key,
                    "cluster_value": value,
                    "part_ids": sorted(part_ids),
                    "candidate_material_ids": materials,
                    "source_part_ids": sorted(
                        str(source["part_id"]) for source in sources
                    ),
                    "action": "no_propagation",
                    "reason_code": "SOURCE_FACE_SUBSETS_REQUIRE_PART_LOCAL_EVIDENCE",
                }
            )
            continue
        parameter_signatures = {
            json.dumps(
                source.get("parameters"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for source in sources
        }
        if len(parameter_signatures) > 1:
            conflicts.append(
                {
                    "cluster_key": cluster_key,
                    "cluster_value": value,
                    "part_ids": sorted(part_ids),
                    "candidate_material_ids": materials,
                    "source_part_ids": sorted(
                        str(source["part_id"]) for source in sources
                    ),
                    "action": "no_propagation",
                    "reason_code": "SOURCE_PARAMETERIZATIONS_CONFLICT",
                }
            )
            continue
        material_id = materials[0]
        semantic = str(sources[0].get("semantic") or "propagated CAD material")
        shared_parameters = sources[0].get("parameters")
        if shared_parameters is not None and not isinstance(shared_parameters, Mapping):
            raise PolicyExactCoverError(
                f"exact-identity source parameters must be an object: "
                f"{sources[0]['part_id']}"
            )
        for part_id in sorted(part_ids):
            if part_id in output:
                continue
            output[part_id] = _fallback_assignment(
                part_id=part_id,
                material_id=material_id,
                semantic=semantic,
                tier=tier,
                reason_codes=[
                    "POLICY_IDENTITY_PROPAGATION",
                    f"CLUSTER_{cluster_key.upper()}",
                ],
                sources=sources,
                parameters=shared_parameters,
            )


def _semantic_rules(
    policy: Mapping[str, Any], allowed_material_ids: set[str]
) -> list[tuple[str, re.Pattern[str], str, str]]:
    result: list[tuple[str, re.Pattern[str], str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(
        _array(policy.get("semantic_rules"), "policy.semantic_rules")
    ):
        rule = _object(raw, f"policy.semantic_rules[{index}]")
        rule_id = _text(rule.get("rule_id"), f"semantic rule[{index}].rule_id")
        if rule_id in seen:
            raise PolicyExactCoverError(f"duplicate semantic rule_id: {rule_id}")
        seen.add(rule_id)
        material_id = _text(
            rule.get("material_id"), f"semantic rule {rule_id}.material_id"
        )
        if material_id not in allowed_material_ids:
            raise PolicyExactCoverError(
                f"semantic rule {rule_id} material is outside industrial whitelist: "
                f"{material_id}"
            )
        pattern_text = _text(rule.get("pattern"), f"semantic rule {rule_id}.pattern")
        try:
            pattern = re.compile(pattern_text, re.IGNORECASE)
        except re.error as exc:
            raise PolicyExactCoverError(
                f"semantic rule {rule_id} has invalid regex: {exc}"
            ) from exc
        semantic = _text(rule.get("semantic"), f"semantic rule {rule_id}.semantic")
        result.append((rule_id, pattern, material_id, semantic))
    return result


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyExactCoverError(f"{label} must be a finite number")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise PolicyExactCoverError(f"{label} must be a finite number")
    return result


def _ratio(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if not 0.0 <= result <= 1.0:
        raise PolicyExactCoverError(f"{label} must be between 0 and 1")
    return result


def _bbox(part: Mapping[str, Any]) -> tuple[list[float], list[float]] | None:
    raw = part.get("world_bbox")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
        return None
    bounds: list[list[float]] = []
    for bound_index, raw_bound in enumerate(raw):
        if (
            not isinstance(raw_bound, Sequence)
            or isinstance(raw_bound, (str, bytes))
            or len(raw_bound) != 3
        ):
            return None
        try:
            bound = [
                _finite_number(value, f"world_bbox[{bound_index}]")
                for value in raw_bound
            ]
        except PolicyExactCoverError:
            return None
        bounds.append(bound)
    if any(bounds[1][index] < bounds[0][index] for index in range(3)):
        return None
    return bounds[0], bounds[1]


def _ground_contact_parts(
    *,
    registry: Mapping[str, Any],
    parts: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    allowed_material_ids: set[str],
) -> tuple[set[str], str | None, dict[str, Any]]:
    raw = _object(policy.get("ground_contact"), "policy.ground_contact")
    enabled = raw.get("enabled")
    if not isinstance(enabled, bool):
        raise PolicyExactCoverError("policy.ground_contact.enabled must be boolean")
    material_id = _text(raw.get("material_id"), "policy.ground_contact.material_id")
    if material_id not in allowed_material_ids:
        raise PolicyExactCoverError(
            "policy.ground_contact.material_id is outside industrial whitelist"
        )
    settings = {
        "elevation_tolerance_ratio": _ratio(
            raw.get("elevation_tolerance_ratio"),
            "policy.ground_contact.elevation_tolerance_ratio",
        ),
        "minimum_lateral_span_ratio": _ratio(
            raw.get("minimum_lateral_span_ratio"),
            "policy.ground_contact.minimum_lateral_span_ratio",
        ),
        "maximum_up_span_ratio": _ratio(
            raw.get("maximum_up_span_ratio"),
            "policy.ground_contact.maximum_up_span_ratio",
        ),
    }
    if not enabled:
        return set(), material_id, {**settings, "state": "disabled"}
    render_set = registry.get("render_set")
    if not isinstance(render_set, Mapping):
        return (
            set(),
            material_id,
            {
                **settings,
                "state": "skipped_missing_render_set",
            },
        )
    raw_up = render_set.get("analysis_up_axis")
    if (
        not isinstance(raw_up, Sequence)
        or isinstance(raw_up, (str, bytes))
        or len(raw_up) != 3
    ):
        return (
            set(),
            material_id,
            {
                **settings,
                "state": "skipped_missing_analysis_up_axis",
            },
        )
    up = [
        _finite_number(value, f"render_set.analysis_up_axis[{index}]")
        for index, value in enumerate(raw_up)
    ]
    up_axis = max(range(3), key=lambda index: abs(up[index]))
    if abs(up[up_axis]) < 0.9 or any(
        abs(up[index]) > 0.1 for index in range(3) if index != up_axis
    ):
        return (
            set(),
            material_id,
            {
                **settings,
                "state": "skipped_non_axis_aligned_up_axis",
                "analysis_up_axis": up,
            },
        )
    valid = [
        (str(part["part_id"]), bounds)
        for part in parts
        if (bounds := _bbox(part)) is not None
    ]
    if not valid:
        return set(), material_id, {**settings, "state": "skipped_missing_bboxes"}
    sign = 1.0 if up[up_axis] >= 0.0 else -1.0
    lateral_axes = [index for index in range(3) if index != up_axis]
    global_min = [min(bounds[0][index] for _, bounds in valid) for index in range(3)]
    global_max = [max(bounds[1][index] for _, bounds in valid) for index in range(3)]
    global_span = [global_max[index] - global_min[index] for index in range(3)]
    up_extent = global_span[up_axis]
    if up_extent <= 0.0:
        return set(), material_id, {**settings, "state": "skipped_zero_up_extent"}
    ground_coordinate = global_min[up_axis] if sign > 0.0 else global_max[up_axis]
    matched: set[str] = set()
    for part_id, (minimum, maximum) in valid:
        part_ground = minimum[up_axis] if sign > 0.0 else maximum[up_axis]
        elevation = abs(part_ground - ground_coordinate)
        up_span = maximum[up_axis] - minimum[up_axis]
        lateral_ratios = [
            (
                (maximum[index] - minimum[index]) / global_span[index]
                if global_span[index] > 0.0
                else 0.0
            )
            for index in lateral_axes
        ]
        if (
            elevation <= settings["elevation_tolerance_ratio"] * up_extent
            and up_span <= settings["maximum_up_span_ratio"] * up_extent
            and max(lateral_ratios, default=0.0)
            >= settings["minimum_lateral_span_ratio"]
        ):
            matched.add(part_id)
    return (
        matched,
        material_id,
        {
            **settings,
            "state": "evaluated",
            "analysis_up_axis": up,
            "up_axis_index": up_axis,
            "matched_part_count": len(matched),
        },
    )


def _mvinverse_parameterizations(
    *,
    group_materials: Mapping[str, Any] | None,
    mvinverse_pbr_evidence: Mapping[str, Any] | None,
    allowed_material_ids: set[str],
    palette_fusion: Mapping[str, Any] | None = None,
    key_by_group: bool = False,
    excluded_group_ids: set[str] | None = None,
) -> tuple[dict[str, tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    if group_materials is None and mvinverse_pbr_evidence is None:
        return {}, []
    if group_materials is None or mvinverse_pbr_evidence is None:
        return {}, [
            {
                "reason_code": "MVINVERSE_INPUT_BUNDLE_INCOMPLETE",
                "detail": (
                    "group_materials and mvinverse_pbr_evidence must be supplied "
                    "together; parameterization was disabled"
                ),
            }
        ]
    if group_materials.get("schema_version") != GROUP_MATERIALS_SCHEMA_VERSION:
        raise PolicyExactCoverError("group_materials has an unsupported schema_version")
    try:
        verified_mvinverse_evidence = validate_mvinverse_evidence(
            mvinverse_pbr_evidence
        )
    except MVInverseEvidenceError as exc:
        return {}, [
            {
                "reason_code": "MVINVERSE_EVIDENCE_STRICT_VALIDATION_FAILED",
                "detail": str(exc),
            }
        ]
    evidence_by_group: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(
        _array(verified_mvinverse_evidence.get("groups"), "mvinverse evidence.groups")
    ):
        group = _object(raw, f"mvinverse evidence.groups[{index}]")
        group_id = _text(
            group.get("group_id"), f"mvinverse evidence.groups[{index}].group_id"
        )
        if group_id in evidence_by_group:
            raise PolicyExactCoverError(
                f"duplicate MVInverse evidence group: {group_id}"
            )
        evidence_by_group[group_id] = group

    color_corroborated_group_ids: set[str] = set()
    if palette_fusion is not None:
        if palette_fusion.get("schema_version") != PALETTE_FUSION_SCHEMA_VERSION:
            raise PolicyExactCoverError(
                "palette fusion has an unsupported schema_version"
            )
        canonical = _object(
            palette_fusion.get("canonical_palette"),
            "palette_fusion.canonical_palette",
        )
        if canonical.get("schema_version") != CANONICAL_PALETTE_SCHEMA_VERSION:
            raise PolicyExactCoverError(
                "canonical palette has an unsupported schema_version"
            )
        seen_palette_groups: set[str] = set()
        for index, raw in enumerate(
            _array(canonical.get("groups"), "canonical_palette.groups")
        ):
            group = _object(raw, f"canonical_palette.groups[{index}]")
            group_id = _text(
                group.get("group_id"),
                f"canonical_palette.groups[{index}].group_id",
            )
            if group_id in seen_palette_groups:
                raise PolicyExactCoverError(
                    f"duplicate canonical palette group: {group_id}"
                )
            seen_palette_groups.add(group_id)
            canonical_color = fusion_color_label(
                _text(
                    group.get("base_color"),
                    f"canonical palette group {group_id}.base_color",
                )
            )
            supporting_views: set[str] = set()
            raw_sources = group.get("sources")
            if (
                not isinstance(raw_sources, Sequence)
                or isinstance(raw_sources, (str, bytes))
            ):
                continue
            for source_index, raw_source in enumerate(
                raw_sources
            ):
                source = _object(
                    raw_source,
                    f"canonical palette group {group_id}.sources[{source_index}]",
                )
                source_color = fusion_color_label(
                    _text(
                        source.get("base_color"),
                        f"canonical palette group {group_id}.source.base_color",
                    )
                )
                if source_color == canonical_color:
                    supporting_views.add(
                        _text(
                            source.get("view_id"),
                            f"canonical palette group {group_id}.source.view_id",
                        )
                    )
            evidence_group = evidence_by_group.get(group_id)
            albedo = (
                evidence_group.get("albedo")
                if isinstance(evidence_group, Mapping)
                else None
            )
            raw_median = albedo.get("median") if isinstance(albedo, Mapping) else None
            if (
                len(supporting_views) >= 2
                and isinstance(raw_median, Sequence)
                and not isinstance(raw_median, (str, bytes))
                and len(raw_median) == 3
                and all(_unit_number_or_none(value) is not None for value in raw_median)
                and _rgb_color_label([float(value) for value in raw_median])
                == canonical_color
            ):
                color_corroborated_group_ids.add(group_id)

    def tune_group(
        *,
        group_id: str,
        material_id: str,
        evidence_group: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            return tune_selected_material_from_mvinverse(
                evidence_group,
                group_id=group_id,
                material_id=material_id,
            )
        except ValueError as full_error:
            if group_id not in color_corroborated_group_ids:
                raise full_error
            try:
                return tune_selected_material_color_from_mvinverse(
                    evidence_group,
                    group_id=group_id,
                    material_id=material_id,
                )
            except ValueError:
                raise full_error

    excluded_groups = excluded_group_ids or set()
    candidates_by_material: dict[str, list[tuple[str, Mapping[str, Any]]]] = (
        defaultdict(list)
    )
    result: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    skipped: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    for index, raw in enumerate(
        _array(group_materials.get("selections"), "group_materials.selections")
    ):
        selection = _object(raw, f"group_materials.selections[{index}]")
        group_id = _text(
            selection.get("group_id"), f"group_materials.selections[{index}].group_id"
        )
        if group_id in seen_group_ids:
            raise PolicyExactCoverError(
                f"duplicate group material selection: {group_id}"
            )
        seen_group_ids.add(group_id)
        material_id = _text(
            selection.get("material_id"),
            f"group_materials.selections[{index}].material_id",
        )
        if material_id not in allowed_material_ids:
            raise PolicyExactCoverError(
                f"group material {group_id} is outside industrial whitelist: "
                f"{material_id}"
            )
        if group_id in excluded_groups:
            skipped.append(
                {
                    "group_id": group_id,
                    "material_id": material_id,
                    "reason_code": "MATERIAL_COLLAPSE_RECOVERY_REQUIRED",
                }
            )
            continue
        if selection.get("confirmed") is not True:
            skipped.append(
                {
                    "group_id": group_id,
                    "material_id": material_id,
                    "reason_code": "MATERIAL_SELECTION_NOT_CONFIRMED",
                }
            )
            continue
        # MVInverse observes PBR values; it does not identify a material
        # family.  The exact Qwen-selected MDL must therefore have an explicit
        # tuning profile.  Parameterization never changes the selected
        # material family or substitutes a generic shader.
        if tuning_profile_for_material(material_id) is None:
            skipped.append(
                {
                    "group_id": group_id,
                    "material_id": material_id,
                    "reason_code": "MATERIAL_HAS_NO_BOUNDED_TUNING_PROFILE",
                }
            )
            continue
        evidence_group = evidence_by_group.get(group_id)
        if evidence_group is None:
            skipped.append(
                {
                    "group_id": group_id,
                    "material_id": material_id,
                    "reason_code": "MVINVERSE_GROUP_MISSING",
                }
            )
            continue
        if key_by_group:
            try:
                parameters, audit = tune_group(
                    group_id=group_id,
                    material_id=material_id,
                    evidence_group=evidence_group,
                )
            except ValueError as exc:
                skipped.append(
                    {
                        "group_id": group_id,
                        "material_id": material_id,
                        "reason_code": "MVINVERSE_NOT_AUTO_PARAMETER_ELIGIBLE",
                        "detail": str(exc),
                    }
                )
                continue
            result[group_id] = (parameters, audit)
            continue
        candidates_by_material[material_id].append((group_id, evidence_group))

    if key_by_group:
        return result, skipped
    for material_id in sorted(candidates_by_material):
        candidates = candidates_by_material[material_id]
        if len(candidates) != 1:
            skipped.append(
                {
                    "group_ids": sorted(group_id for group_id, _ in candidates),
                    "material_id": material_id,
                    "reason_code": "AMBIGUOUS_MATERIAL_TO_GROUP_MAPPING",
                }
            )
            continue
        group_id, evidence_group = candidates[0]
        try:
            parameters, audit = tune_group(
                group_id=group_id,
                material_id=material_id,
                evidence_group=evidence_group,
            )
        except ValueError as exc:
            skipped.append(
                {
                    "group_id": group_id,
                    "material_id": material_id,
                    "reason_code": "MVINVERSE_NOT_AUTO_PARAMETER_ELIGIBLE",
                    "detail": str(exc),
                }
            )
            continue
        result[material_id] = (parameters, audit)
    return result, skipped


def _corroborated_group_material_selections(
    *,
    group_materials: Mapping[str, Any] | None,
    allowed_material_ids: set[str],
    allow_high_confidence_provisional: bool,
    excluded_group_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return exact NVIDIA MDL choices eligible for source corroboration.

    A forward/reverse disagreement between near-equivalent named presets does
    not make a multiview photo/CAD colour binding disappear.  In immutable
    mode, a high-confidence Qwen choice may therefore replace a corroborated
    source display material before the one bounded render-QA round.  It remains
    explicitly provisional and is never promoted to model confidence.
    """

    if group_materials is None:
        return {}
    if not isinstance(allow_high_confidence_provisional, bool):
        raise PolicyExactCoverError(
            "allow_high_confidence_provisional must be boolean"
        )
    if group_materials.get("schema_version") != GROUP_MATERIALS_SCHEMA_VERSION:
        raise PolicyExactCoverError("group_materials has an unsupported schema_version")
    excluded_groups = excluded_group_ids or set()
    result: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for index, raw_selection in enumerate(
        _array(group_materials.get("selections"), "group_materials.selections")
    ):
        selection = _object(
            raw_selection,
            f"group_materials.selections[{index}]",
        )
        group_id = _text(
            selection.get("group_id"),
            f"group_materials.selections[{index}].group_id",
        )
        if group_id in seen:
            raise PolicyExactCoverError(
                f"duplicate group material selection: {group_id}"
            )
        seen.add(group_id)
        material_id = _text(
            selection.get("material_id"),
            f"group_materials.selections[{index}].material_id",
        )
        if material_id not in allowed_material_ids:
            raise PolicyExactCoverError(
                f"group material {group_id} is outside industrial whitelist: "
                f"{material_id}"
            )
        if group_id in excluded_groups:
            continue
        confirmed = selection.get("confirmed")
        confidence = selection.get("confidence")
        normalized_confidence = _unit_or_zero(confidence)
        if (
            confirmed is True
            and normalized_confidence
            >= CORROBORATED_SOURCE_MIN_CONFIRMED_CONFIDENCE
        ):
            result[group_id] = {
                "material_id": material_id,
                "confirmed": True,
                "confidence": normalized_confidence,
                "material_selection_basis": (
                    "derived_evidence_confirmed_material_choice"
                ),
            }
        elif (
            confirmed is False
            and allow_high_confidence_provisional
            and isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and math.isfinite(float(confidence))
            and CORROBORATED_SOURCE_MIN_PROVISIONAL_CONFIDENCE
            <= float(confidence)
            <= 1.0
        ):
            result[group_id] = {
                "material_id": material_id,
                "confirmed": False,
                "confidence": float(confidence),
                "material_selection_basis": (
                    CORROBORATED_SOURCE_PROVISIONAL_MATERIAL_BASIS
                ),
            }
    return result


def _apply_mvinverse_parameterizations(
    assignments: Sequence[dict[str, Any]],
    parameterizations: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    key_by_group: bool = False,
) -> list[str]:
    parameterized: list[str] = []
    for assignment in assignments:
        provenance = _object(assignment.get("provenance"), "assignment.provenance")
        if provenance.get("tier") not in _MVINVERSE_TRUSTED_TIERS:
            continue
        material_id = str(assignment["material_id"])
        parameterization_key = material_id
        if key_by_group:
            corroboration = provenance.get("source_visual_corroboration")
            group_id = (
                corroboration.get("canonical_group_id")
                if isinstance(corroboration, Mapping)
                else None
            )
            parameterization_key = group_id if isinstance(group_id, str) else ""
        spec = parameterizations.get(parameterization_key)
        if spec is None:
            continue
        if "parameters" in assignment or assignment.get("face_subsets"):
            continue
        parameters, audit = spec
        assignment["parameters"] = copy.deepcopy(dict(parameters))
        mutable_provenance = dict(provenance)
        mutable_provenance["mvinverse"] = {
            "source_material_id": material_id,
            "output_material_id": material_id,
            "group_id": audit["group_id"],
            "tuning_profile_id": audit["tuning_profile_id"],
            "parameterization_mode": audit.get(
                "parameterization_mode", "full_mvinverse_pbr"
            ),
            "contributing_view_ids": list(audit["contributing_view_ids"]),
            "base_color_srgb": list(audit["base_color_srgb"]),
            "base_color_linear": list(audit["base_color_linear"]),
            "observed_metallic": audit["observed_metallic"],
            "authored_metallic": audit["authored_metallic"],
            "roughness": audit["roughness"],
            "authored_parameter_names": list(audit["authored_parameter_names"]),
            "reason_code": (
                "MVINVERSE_AUTO_PARAMETER_ELIGIBLE"
                if audit.get("parameterization_mode") != (
                    "multiview_palette_corroborated_color_only"
                )
                else "MVINVERSE_COLOR_CORROBORATED_BY_MULTIVIEW_PALETTE"
            ),
        }
        assignment["provenance"] = mutable_provenance
        parameterized.append(str(assignment["part_id"]))
    return parameterized


def build_policy_exact_cover(
    *,
    registry: Mapping[str, Any],
    staged_result: Mapping[str, Any],
    confidence_gate: Mapping[str, Any],
    whitelist: Mapping[str, Any],
    acknowledge_policy_fallback: bool,
    policy: Mapping[str, Any] | None = None,
    base_plan: Mapping[str, Any] | None = None,
    group_materials: Mapping[str, Any] | None = None,
    mvinverse_pbr_evidence: Mapping[str, Any] | None = None,
    palette_fusion: Mapping[str, Any] | None = None,
    part_id_evidence: Mapping[str, Any] | None = None,
    immutable_mdl_after_selection: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an exact-cover plan and a hash-bound best-effort audit."""

    if acknowledge_policy_fallback is not True:
        raise PolicyExactCoverError(
            "policy fallback requires explicit acknowledge_policy_fallback=True"
        )
    if not isinstance(immutable_mdl_after_selection, bool):
        raise PolicyExactCoverError(
            "immutable_mdl_after_selection must be boolean"
        )
    if part_id_evidence is not None and policy is None:
        raise PolicyExactCoverError(
            "Part-ID evidence convergence requires an explicit source policy"
        )
    allowed_material_ids = _whitelist_ids(whitelist)
    parts, parts_by_id = _registry_parts(registry)
    registry_ids = set(parts_by_id)
    part_id_evidence_statuses = _part_id_evidence_statuses(
        part_id_evidence,
        registry=registry,
        registry_ids=registry_ids,
    )
    unobserved_part_ids = {
        part_id
        for part_id, status in part_id_evidence_statuses.items()
        if status == "unobserved"
    }
    candidates = _candidate_assignments(
        staged_result,
        registry_ids=registry_ids,
        allowed_material_ids=allowed_material_ids,
    )
    collapse_recovery_excluded_group_ids = _collapse_recovery_group_ids(
        staged_result
    )
    autonomous_base = _autonomous_base_assignments(
        base_plan,
        registry_ids=registry_ids,
        allowed_material_ids=allowed_material_ids,
    )
    gate_records = _gate_decision_records(
        confidence_gate,
        registry_ids=registry_ids,
    )
    gate = {
        part_id: str(record["decision"])
        for part_id, record in gate_records.items()
    }
    effective_policy = _policy_for_catalog(
        _effective_policy(policy), allowed_material_ids
    )
    configured_candidate_auto_keys = _validated_cluster_keys(
        effective_policy, "candidate_auto_cluster_keys"
    )
    configured_review_keys = _validated_cluster_keys(
        effective_policy, "review_cluster_keys"
    )
    candidate_auto_keys = [
        key
        for key in configured_candidate_auto_keys
        if key in _EXACT_IDENTITY_CLUSTER_KEYS
    ]
    ignored_cluster_keys = sorted(
        {
            *(set(configured_candidate_auto_keys) - _EXACT_IDENTITY_CLUSTER_KEYS),
            *configured_review_keys,
        }
    )
    rules = _semantic_rules(effective_policy, allowed_material_ids)
    mvinverse_parameterizations, mvinverse_skipped = _mvinverse_parameterizations(
        group_materials=group_materials,
        mvinverse_pbr_evidence=mvinverse_pbr_evidence,
        allowed_material_ids=allowed_material_ids,
        palette_fusion=palette_fusion,
        key_by_group=palette_fusion is not None,
        excluded_group_ids=collapse_recovery_excluded_group_ids,
    )
    corroborated_group_materials = _corroborated_group_material_selections(
        group_materials=group_materials,
        allowed_material_ids=allowed_material_ids,
        allow_high_confidence_provisional=immutable_mdl_after_selection,
        excluded_group_ids=collapse_recovery_excluded_group_ids,
    )

    output: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for part_id in sorted(autonomous_base):
        output[part_id] = _direct_base_assignment(autonomous_base[part_id])

    gate_auto = {
        part_id: assignment
        for part_id, assignment in candidates.items()
        if gate[part_id] == "auto" and assignment.get("status") in {"auto", "approved"}
    }
    for part_id in sorted(gate_auto):
        if part_id not in output:
            output[part_id] = _direct_gate_assignment(gate_auto[part_id])

    # Only assignments independently accepted by the confidence gate, or the
    # already verified autonomous base plan, may seed exact-identity
    # propagation.  A model's staged ``auto`` label is not sufficient.
    trusted_identity_sources = dict(gate_auto)
    trusted_identity_sources.update(autonomous_base)
    for part_id in unobserved_part_ids:
        trusted_identity_sources.pop(part_id, None)
    for cluster_key in (
        key for key in candidate_auto_keys if key != "existing_visual_material"
    ):
        _cluster_propagate(
            parts=parts,
            source_assignments=trusted_identity_sources,
            output=output,
            cluster_key=cluster_key,
            tier=f"trusted_exact_{cluster_key}",
            conflicts=conflicts,
        )

    # A trusted gate/base assignment may carry across an exact authored source
    # binding before any weak policy heuristic is considered.  Basename-only
    # joins remain forbidden by ``_cluster_value``.
    if "existing_visual_material" in candidate_auto_keys:
        _cluster_propagate(
            parts=parts,
            source_assignments=trusted_identity_sources,
            output=output,
            cluster_key="existing_visual_material",
            tier="trusted_authored_material_binding",
            conflicts=conflicts,
        )

    # A final Part-ID projection has stronger visibility authority than the
    # earlier palette/group and exact-identity hypotheses.  Remove every
    # hidden part from those candidate outputs so it can re-enter only through
    # the independent per-part policy rules below.  This is intentionally a
    # no-op for legacy/current callers, which do not provide Part-ID evidence.
    for part_id in unobserved_part_ids:
        output.pop(part_id, None)

    requested_strategy = effective_policy.get("default_strategy")
    if requested_strategy not in {"dominant_staged_auto", "declared_material"}:
        raise PolicyExactCoverError(
            "policy.default_strategy must be dominant_staged_auto or declared_material"
        )
    declared_default = _text(
        effective_policy.get("default_material_id"), "policy.default_material_id"
    )
    if declared_default not in allowed_material_ids:
        raise PolicyExactCoverError(
            "policy.default_material_id is outside industrial whitelist"
        )
    source_visual_strategy = effective_policy.get("source_visual_strategy")
    if source_visual_strategy not in _SOURCE_VISUAL_STRATEGIES:
        raise PolicyExactCoverError(
            "policy.source_visual_strategy must be preserve or "
            "neutralize_unverified"
        )
    # ``material_id`` remains present in preserve-only entries so a later,
    # stronger QA-repair pass can use the same exact-cover plan schema.  Apply
    # never resolves or authors this fallback material for the preserve action.
    default_material = declared_default
    corroborated_source_visual, source_corroboration_audit = (
        _corroborated_source_visual_parts(
            parts=parts,
            palette_fusion=palette_fusion,
        )
    )
    source_visual_preserve_count = 0
    corroborated_source_visual_preserve_part_ids: list[str] = []
    corroborated_source_visual_mdl_part_ids: list[str] = []
    corroborated_source_visual_provisional_mdl_part_ids: list[str] = []
    neutralized_source_visual_part_ids: set[str] = set()
    for part in parts:
        part_id = str(part["part_id"])
        if part_id in output:
            continue
        if source_visual_strategy == "neutralize_unverified":
            corroboration = (
                None
                if part_id in unobserved_part_ids
                else corroborated_source_visual.get(part_id)
            )
            if corroboration is not None:
                group_id = str(corroboration["canonical_group_id"])
                material_selection = corroborated_group_materials.get(group_id)
                if material_selection is not None:
                    selected_material_id = str(
                        material_selection["material_id"]
                    )
                    material_confirmed = material_selection["confirmed"] is True
                    source_material = _valid_source_visual_material_path(
                        part.get("existing_visual_material")
                    )
                    if source_material is None:
                        raise AssertionError(
                            "corroborated source visual has no valid authored binding"
                        )
                    assignment = _fallback_assignment(
                        part_id=part_id,
                        material_id=selected_material_id,
                        semantic=(
                            "photo-corroborated source accent represented by the "
                            "Qwen-selected NVIDIA MDL material pending bounded "
                            "render QA"
                            if not material_confirmed
                            else "photo-corroborated source accent represented by "
                            "the Qwen-confirmed NVIDIA MDL material"
                        ),
                        tier=CORROBORATED_SOURCE_MDL_TIER,
                        reason_codes=[
                            "REFERENCE_PALETTE_MULTIVIEW_COLOR_CORROBORATION",
                            "RARE_SOURCE_VISUAL_SIGNATURE",
                            "REPEATED_GEOMETRY_SOURCE_LOCATOR",
                            *(
                                ["EXACT_SOURCE_SIGNATURE_COHORT_EXPANSION"]
                                if corroboration.get("signature_expansion_basis")
                                == (
                                    "exact_source_signature_with_repeated_geometry_anchor"
                                )
                                else []
                            ),
                            (
                                "QWEN_CONFIRMED_NVIDIA_MDL_SELECTION"
                                if material_confirmed
                                else (
                                    "QWEN_HIGH_CONFIDENCE_NVIDIA_MDL_CANDIDATE"
                                )
                            ),
                            *(
                                []
                                if material_confirmed
                                else ["QA_POST_RENDER_VALIDATION_REQUIRED"]
                            ),
                            "SOURCE_VISUAL_SIGNATURE_REPLACED_BY_NVIDIA_MDL",
                        ],
                    )
                    assignment["provenance"]["source_visual_material"] = {
                        "material_prim_path": source_material,
                        "binding_sha256": source_visual_binding_sha256(
                            part_id=part_id,
                            prim_path=_text(
                                part.get("prim_path"),
                                f"registry part {part_id}.prim_path",
                            ),
                            material_prim_path=source_material,
                        ),
                    }
                    assignment["provenance"]["canonical_group_id"] = group_id
                    assignment["provenance"]["supporting_view_ids"] = list(
                        corroboration["canonical_source_view_ids"]
                    )
                    assignment["provenance"][
                        "canonical_group_assignment_basis"
                    ] = "photo_corroborated_rare_repeated_source_visual"
                    assignment["provenance"]["source_visual_corroboration"] = {
                        **copy.deepcopy(corroboration),
                        **(
                            {"confirmed_material_id": selected_material_id}
                            if material_confirmed
                            else {
                                "selected_material_id": selected_material_id,
                                "material_selection_basis": material_selection[
                                    "material_selection_basis"
                                ],
                                "selection_confidence": material_selection[
                                    "confidence"
                                ],
                            }
                        ),
                    }
                    output[part_id] = assignment
                    corroborated_source_visual_mdl_part_ids.append(part_id)
                    if not material_confirmed:
                        corroborated_source_visual_provisional_mdl_part_ids.append(
                            part_id
                        )
                    continue
                preserve_assignment = _source_visual_preserve_assignment(
                    part=part,
                    fallback_material_id=default_material,
                    corroboration=corroboration,
                )
                if preserve_assignment is None:
                    raise AssertionError(
                        "corroborated source visual has no valid authored binding"
                    )
                output[part_id] = preserve_assignment
                source_visual_preserve_count += 1
                corroborated_source_visual_preserve_part_ids.append(part_id)
                continue
            if _valid_source_visual_material_path(
                part.get("existing_visual_material")
            ) is not None:
                neutralized_source_visual_part_ids.add(part_id)
            continue
        preserve_assignment = _source_visual_preserve_assignment(
            part=part,
            fallback_material_id=default_material,
        )
        if preserve_assignment is not None:
            output[part_id] = preserve_assignment
            source_visual_preserve_count += 1

    # Name and geometry policy rules are deliberately weaker than an immutable,
    # valid source visual binding.  They only fill parts for which neither
    # evidence-driven assignment nor source-preserve is available.
    for part in parts:
        part_id = str(part["part_id"])
        if part_id in output:
            continue
        source_text = " ".join(
            str(part.get(field) or "")
            for field in ("prim_path", "parent_path", "prim_name")
        )
        for rule_id, pattern, material_id, semantic in rules:
            if pattern.search(source_text):
                output[part_id] = _fallback_assignment(
                    part_id=part_id,
                    material_id=material_id,
                    semantic=semantic,
                    tier="semantic_rule",
                    reason_codes=["POLICY_SEMANTIC_RULE", f"RULE_{rule_id.upper()}"],
                )
                break

    ground_part_ids, ground_material_id, ground_audit = _ground_contact_parts(
        registry=registry,
        parts=parts,
        policy=effective_policy,
        allowed_material_ids=allowed_material_ids,
    )
    if ground_material_id is None:
        raise AssertionError("ground-contact policy returned no material")
    for part_id in sorted(ground_part_ids):
        if part_id in output:
            continue
        output[part_id] = _fallback_assignment(
            part_id=part_id,
            material_id=ground_material_id,
            semantic="ground-contacting industrial base or support",
            tier="ground_contact_bbox",
            reason_codes=[
                "POLICY_GROUND_CONTACT_BBOX",
                "CAD_GEOMETRY_ONLY_NOT_PHOTO_EVIDENCE",
            ],
        )

    # ``dominant_staged_auto`` remains accepted as a legacy input value so old
    # policy files still parse, but it is deliberately never executed.  One
    # visible or weakly staged colour must not paint every unresolved CAD part.
    default_reason = (
        "POLICY_DOMINANT_DEFAULT_DISABLED_NEUTRAL_USED"
        if requested_strategy == "dominant_staged_auto"
        else "POLICY_DECLARED_NEUTRAL_DEFAULT"
    )
    neutral_default_count = 0
    source_preserve_unavailable_neutral_fallback_count = 0
    for part in parts:
        part_id = str(part["part_id"])
        if part_id in output:
            continue
        source_visual_neutralized = (
            part_id in neutralized_source_visual_part_ids
        )
        if source_visual_neutralized:
            source_preserve_unavailable_neutral_fallback_count += 1
        else:
            neutral_default_count += 1
        output[part_id] = _fallback_assignment(
            part_id=part_id,
            material_id=default_material,
            semantic=(
                "neutral delivery material replacing an unverified source CAD "
                "display material"
                if source_visual_neutralized
                else "neutral delivery material for unresolved CAD component"
            ),
            tier=(
                "source_preserve_unavailable_neutral_fallback"
                if source_visual_neutralized
                else "neutral_default"
            ),
            reason_codes=(
                [
                    default_reason,
                    "UNVERIFIED_SOURCE_VISUAL_MATERIAL_IGNORED",
                    "CAD_DISPLAY_COLOR_IS_NOT_PHOTO_APPEARANCE_EVIDENCE",
                ]
                if source_visual_neutralized
                else [default_reason]
            ),
        )

    if set(output) != registry_ids:
        raise AssertionError(
            "policy exact-cover builder failed internal coverage check"
        )
    retained_mapping_lineage_audit = _retain_non_authoring_mapping_lineage(
        output=output,
        candidates=candidates,
        gate_records=gate_records,
        confidence_gate=confidence_gate,
        palette_fusion=palette_fusion,
        excluded_part_ids=unobserved_part_ids,
    )
    propagated_mapping_lineage_audit = (
        _propagate_non_authoring_mapping_lineage(
            parts=parts,
            output=output,
            cluster_keys=candidate_auto_keys,
            excluded_part_ids=unobserved_part_ids,
        )
    )
    assignments = [output[str(part["part_id"])] for part in parts]
    source_subset_contract_audit = _prepare_source_subset_contracts(
        assignments=assignments,
        parts_by_id=parts_by_id,
    )
    if immutable_mdl_after_selection:
        for assignment in assignments:
            part_id = str(assignment["part_id"])
            parameters = assignment.get("parameters")
            if parameters is not None and (
                not isinstance(parameters, Mapping) or bool(parameters)
            ):
                raise PolicyExactCoverError(
                    "immutable selected-MDL mode requires library-default "
                    f"parameters before selection is sealed: {part_id}"
                )
            raw_subsets = assignment.get("face_subsets", [])
            if isinstance(raw_subsets, Sequence) and not isinstance(
                raw_subsets, (str, bytes)
            ):
                for subset in raw_subsets:
                    if not isinstance(subset, Mapping):
                        continue
                    subset_parameters = subset.get("parameters")
                    if subset_parameters is not None and (
                        not isinstance(subset_parameters, Mapping)
                        or bool(subset_parameters)
                    ):
                        raise PolicyExactCoverError(
                            "immutable selected-MDL mode requires library-default "
                            f"face-subset parameters before selection is sealed: {part_id}"
                        )
    parameterized_part_ids = (
        []
        if immutable_mdl_after_selection
        else _apply_mvinverse_parameterizations(
            assignments,
            mvinverse_parameterizations,
            key_by_group=palette_fusion is not None,
        )
    )
    _materialize_source_subset_rebinds(
        assignments=assignments,
        parts_by_id=parts_by_id,
        audit=source_subset_contract_audit,
    )
    invalid_materials = sorted(
        {
            str(material_id)
            for item in assignments
            for material_id in [
                item["material_id"],
                *[
                    subset.get("material_id")
                    for subset in item.get("face_subsets", [])
                    if isinstance(subset, Mapping)
                ],
            ]
        }
        - allowed_material_ids
    )
    if invalid_materials:
        raise AssertionError(f"non-whitelisted output materials: {invalid_materials}")

    status_counts = Counter(str(item["status"]) for item in assignments)
    tier_counts = Counter(
        str(_object(item["provenance"], "assignment.provenance")["tier"])
        for item in assignments
    )
    material_counts = Counter(str(item["material_id"]) for item in assignments)
    source_subset_parts = [
        part
        for part in parts
        if part.get(SOURCE_MATERIAL_BIND_SUBSETS_FIELD)
    ]
    source_subset_count = sum(
        len(part[SOURCE_MATERIAL_BIND_SUBSETS_FIELD])
        for part in source_subset_parts
    )
    source_subset_paths = sorted(
        str(subset["subset_prim_path"])
        for part in source_subset_parts
        for subset in part[SOURCE_MATERIAL_BIND_SUBSETS_FIELD]
    )
    parent_expansion_ids = source_subset_contract_audit[
        "parent_material_expansion_part_ids"
    ]
    explicit_match_ids = source_subset_contract_audit[
        "explicit_topology_match_part_ids"
    ]
    explicit_collapse_ids = source_subset_contract_audit[
        "explicit_topology_collapse_part_ids"
    ]
    preserve_subset_ids = source_subset_contract_audit[
        "source_visual_preserve_part_ids"
    ]
    source_subset_count_by_part = {
        str(part["part_id"]): len(part[SOURCE_MATERIAL_BIND_SUBSETS_FIELD])
        for part in source_subset_parts
    }
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "assignments": assignments,
        "provenance": {
            "mode": "explicit_best_effort_policy_exact_cover",
            "registry_asset_sha256": registry.get("asset_sha256"),
            "registry_sha256": _canonical_sha256(registry),
            "staged_result_sha256": _canonical_sha256(staged_result),
            "confidence_gate_sha256": _canonical_sha256(confidence_gate),
            "base_plan_sha256": (
                _canonical_sha256(base_plan) if base_plan is not None else None
            ),
            "group_materials_sha256": (
                _canonical_sha256(group_materials)
                if group_materials is not None
                else None
            ),
            "mvinverse_pbr_evidence_sha256": (
                _canonical_sha256(mvinverse_pbr_evidence)
                if mvinverse_pbr_evidence is not None
                else None
            ),
            **(
                {"palette_fusion_sha256": _canonical_sha256(palette_fusion)}
                if palette_fusion is not None
                else {}
            ),
            **(
                {
                    "part_id_evidence_sha256": _canonical_sha256(
                        part_id_evidence
                    ),
                    "source_policy_sha256": _canonical_sha256(policy),
                }
                if part_id_evidence is not None
                else {}
            ),
            "whitelist_sha256": _canonical_sha256(whitelist),
            "policy_sha256": _canonical_sha256(effective_policy),
            **(
                {"immutable_mdl_after_selection": True}
                if immutable_mdl_after_selection
                else {}
            ),
        },
    }
    report_summary: dict[str, Any] = {
        "registry_part_count": len(parts),
        "staged_candidate_count": len(candidates),
        "autonomous_base_assignment_count": len(autonomous_base),
        "confidence_gate_auto_count": len(gate_auto),
        "output_assignment_count": len(assignments),
        "policy_fallback_count": status_counts[FALLBACK_STATUS],
        **(
            {
                "material_collapse_recovery_excluded_group_count": len(
                    collapse_recovery_excluded_group_ids
                )
            }
            if collapse_recovery_excluded_group_ids
            else {}
        ),
        "mvinverse_parameterized_part_count": len(parameterized_part_ids),
        "neutral_default_count": neutral_default_count,
        "source_visual_preserve_count": source_visual_preserve_count,
        "corroborated_source_visual_preserve_count": len(
            corroborated_source_visual_preserve_part_ids
        ),
        "corroborated_source_visual_nvidia_mdl_count": len(
            corroborated_source_visual_mdl_part_ids
        ),
        "source_preserve_unavailable_neutral_fallback_count": (
            source_preserve_unavailable_neutral_fallback_count
        ),
        "source_material_bind_subset_part_count": len(source_subset_parts),
        "source_material_bind_subset_count": source_subset_count,
        "source_subset_parent_material_expansion_part_count": len(
            parent_expansion_ids
        ),
        "source_subset_parent_material_expansion_count": sum(
            source_subset_count_by_part[part_id]
            for part_id in parent_expansion_ids
        ),
        "source_subset_explicit_topology_match_part_count": len(
            explicit_match_ids
        ),
        "source_subset_explicit_topology_collapse_part_count": len(
            explicit_collapse_ids
        ),
        "exact_cover": True,
        "all_materials_in_industrial_whitelist": True,
        "applicable_without_explicit_policy_fallback_authorization": False,
    }
    if immutable_mdl_after_selection:
        report_summary["selected_mdl_library_defaults_locked"] = True
        report_summary[
            "corroborated_source_visual_provisional_nvidia_mdl_count"
        ] = len(corroborated_source_visual_provisional_mdl_part_ids)
    if part_id_evidence is not None:
        report_summary.update(
            {
                "part_id_evidence_observed_count": (
                    len(part_id_evidence_statuses) - len(unobserved_part_ids)
                ),
                "part_id_evidence_unobserved_count": len(unobserved_part_ids),
                "part_id_evidence_constrained_policy_fallback_count": len(
                    unobserved_part_ids
                ),
            }
        )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "summary": report_summary,
        "policy": effective_policy,
        "source_visual_strategy": source_visual_strategy,
        "source_material_bind_subsets": {
            "source_part_ids": sorted(source_subset_count_by_part),
            "source_subset_prim_paths": source_subset_paths,
            "source_visual_preserve_part_ids": sorted(preserve_subset_ids),
            "parent_material_expansion_part_ids": sorted(parent_expansion_ids),
            "explicit_topology_match_part_ids": sorted(explicit_match_ids),
            "explicit_topology_collapse_part_ids": sorted(explicit_collapse_ids),
        },
        "corroborated_source_visual": {
            **source_corroboration_audit,
            "applied_part_ids": sorted(
                [
                    *corroborated_source_visual_preserve_part_ids,
                    *corroborated_source_visual_mdl_part_ids,
                ]
            ),
            "preserved_part_ids": sorted(
                corroborated_source_visual_preserve_part_ids
            ),
            "nvidia_mdl_replacement_part_ids": sorted(
                corroborated_source_visual_mdl_part_ids
            ),
        },
        **(
            {
                "material_collapse_recovery": {
                    "excluded_group_ids": sorted(
                        collapse_recovery_excluded_group_ids
                    ),
                    "fallback_rule": (
                        "reviewed or unsafe palette groups cannot seed "
                        "source-visual material replacement"
                    ),
                }
            }
            if collapse_recovery_excluded_group_ids
            else {}
        ),
        **(
            {
                "part_id_evidence_convergence": {
                    "state": "final_visibility_applied",
                    "assignment_unit": "part_id",
                    "part_id_evidence_sha256": _canonical_sha256(
                        part_id_evidence
                    ),
                    "source_policy_sha256": _canonical_sha256(policy),
                    "observed_part_count": (
                        len(part_id_evidence_statuses) - len(unobserved_part_ids)
                    ),
                    "unobserved_part_count": len(unobserved_part_ids),
                    "unobserved_assignment_policy": (
                        "independent_policy_only_no_palette_group_or_identity_"
                        "propagation"
                    ),
                }
            }
            if part_id_evidence is not None
            else {}
        ),
        "selected_default_material_id": default_material,
        "requested_default_strategy": requested_strategy,
        "effective_default_strategy": "declared_material",
        "identity_propagation": {
            "configured_candidate_auto_cluster_keys": (configured_candidate_auto_keys),
            "configured_review_cluster_keys": configured_review_keys,
            "applied_cluster_keys": candidate_auto_keys,
            "ignored_unsafe_cluster_keys": ignored_cluster_keys,
            "trusted_source_part_ids": sorted(trusted_identity_sources),
        },
        "provisional_group_hypotheses": {
            "direct_review_hypotheses": retained_mapping_lineage_audit,
            "exact_identity_hypothesis_expansion": (
                propagated_mapping_lineage_audit
            ),
        },
        "ground_contact": ground_audit,
        "mvinverse": {
            "parameterized_part_ids": sorted(parameterized_part_ids),
            "eligible_source_material_ids": sorted(mvinverse_parameterizations),
            "skipped": mvinverse_skipped,
        },
        "status_counts": dict(sorted(status_counts.items())),
        "provenance_tier_counts": dict(sorted(tier_counts.items())),
        "material_counts": dict(sorted(material_counts.items())),
        "conflicts": sorted(
            conflicts,
            key=lambda item: (item["cluster_key"], item["cluster_value"]),
        ),
        "input_hashes": dict(plan["provenance"]),
        "output_plan_sha256": _canonical_sha256(plan),
    }
    return plan, report


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve(strict=True).read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyExactCoverError(f"unable to read {label}: {exc}") from exc
    return dict(_object(value, label))


def _write_json_new(path: Path, value: Mapping[str, Any]) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise PolicyExactCoverError(f"refusing to overwrite output: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, resolved)
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--staged-result", type=Path, required=True)
    parser.add_argument("--confidence-gate", type=Path, required=True)
    parser.add_argument("--whitelist", type=Path, required=True)
    parser.add_argument("--base-plan", type=Path)
    parser.add_argument("--group-materials", type=Path)
    parser.add_argument("--mvinverse-pbr-evidence", type=Path)
    parser.add_argument("--palette-fusion", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--acknowledge-policy-fallback", action="store_true")
    parser.add_argument(
        "--immutable-mdl-after-selection",
        action="store_true",
        help="forbid all post-selection MDL parameter overrides",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plan, report = build_policy_exact_cover(
        registry=_read_json(args.registry, "registry"),
        staged_result=_read_json(args.staged_result, "staged result"),
        confidence_gate=_read_json(args.confidence_gate, "confidence gate"),
        whitelist=_read_json(args.whitelist, "whitelist"),
        policy=(_read_json(args.policy, "policy") if args.policy is not None else None),
        base_plan=(
            _read_json(args.base_plan, "base plan")
            if args.base_plan is not None
            else None
        ),
        group_materials=(
            _read_json(args.group_materials, "group materials")
            if args.group_materials is not None
            else None
        ),
        mvinverse_pbr_evidence=(
            _read_json(args.mvinverse_pbr_evidence, "MVInverse PBR evidence")
            if args.mvinverse_pbr_evidence is not None
            else None
        ),
        palette_fusion=(
            _read_json(args.palette_fusion, "palette fusion")
            if args.palette_fusion is not None
            else None
        ),
        acknowledge_policy_fallback=args.acknowledge_policy_fallback,
        immutable_mdl_after_selection=args.immutable_mdl_after_selection,
    )
    plan_path = _write_json_new(args.output_plan, plan)
    audit_path = _write_json_new(args.audit, report)
    print(
        json.dumps(
            {
                "output_plan": str(plan_path),
                "audit": str(audit_path),
                **report["summary"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
