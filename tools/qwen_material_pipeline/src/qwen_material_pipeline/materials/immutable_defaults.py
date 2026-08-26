"""Resolve parameterized material plans to immutable NVIDIA MDL exports.

The visual pipeline may receive a previously accepted plan whose appearance was
authored by changing parameters on a generic MDL.  Immutable delivery forbids
replaying those edits.  This module deterministically replaces each
parameterized assignment with a library export from the same MDL module,
guided by the assignment semantic and the Qwen/MVInverse primary candidate
pool.  It never authors or copies material parameters.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "qwen-immutable-default-plan/v1"
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_COLOR_TOKENS = frozenset(
    {
        "black",
        "blue",
        "brown",
        "copper",
        "cyan",
        "gold",
        "gray",
        "green",
        "grey",
        "ivory",
        "orange",
        "pink",
        "purple",
        "red",
        "silver",
        "tan",
        "turquoise",
        "white",
        "yellow",
    }
)


class ImmutableDefaultResolutionError(ValueError):
    """Raised when a parameterized material has no auditable exact export."""


def _tokens(value: object) -> set[str]:
    return set(_TOKEN_RE.findall(str(value).casefold()))


def _materials(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = catalog.get("materials")
    if not isinstance(raw, list):
        raise ImmutableDefaultResolutionError("catalog.materials must be an array")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ImmutableDefaultResolutionError(
                f"catalog.materials[{index}] must be an object"
            )
        material_id = item.get("material_id")
        mdl_path = item.get("mdl_path")
        sub_identifier = item.get("sub_identifier")
        if not all(
            isinstance(value, str) and value
            for value in (material_id, mdl_path, sub_identifier)
        ):
            raise ImmutableDefaultResolutionError(
                f"catalog.materials[{index}] lacks exact MDL identity"
            )
        if material_id in seen:
            continue
        seen.add(material_id)
        output.append(item)
    return output


def load_primary_candidate_ids(paths: Iterable[str | Path]) -> set[str]:
    """Load Qwen/MVInverse primary candidates without using broad tournament order."""

    output: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve(strict=True)
        document = json.loads(path.read_text(encoding="utf-8"))
        candidates = document.get("candidates") if isinstance(document, dict) else None
        if not isinstance(candidates, list):
            raise ImmutableDefaultResolutionError(
                f"primary candidate file lacks candidates: {path}"
            )
        for candidate in candidates:
            material_id = (
                candidate.get("material_id") if isinstance(candidate, dict) else None
            )
            if isinstance(material_id, str) and material_id:
                output.add(material_id)
    return output


def choose_exact_default_export(
    *,
    assignment: Mapping[str, Any],
    catalog_materials: Sequence[Mapping[str, Any]],
    primary_candidate_ids: set[str],
) -> tuple[str, dict[str, Any]]:
    """Choose one same-module, parameter-free library export deterministically."""

    material_id = assignment.get("material_id")
    semantic = assignment.get("semantic", "")
    if not isinstance(material_id, str) or not material_id.startswith("mdl:"):
        raise ImmutableDefaultResolutionError("assignment lacks material_id")
    if "#" not in material_id:
        raise ImmutableDefaultResolutionError(
            f"assignment is not an exact MDL export: {material_id}"
        )
    module = material_id[4:].split("#", 1)[0]
    semantic_tokens = _tokens(semantic)
    semantic_colors = semantic_tokens & _COLOR_TOKENS
    scored: list[tuple[float, str, Mapping[str, Any], dict[str, Any]]] = []
    for candidate in catalog_materials:
        if candidate.get("mdl_path") != module:
            continue
        candidate_id = candidate.get("material_id")
        if not isinstance(candidate_id, str) or candidate_id == material_id:
            continue
        color_tokens = set()
        colors = candidate.get("colors", [])
        if isinstance(colors, list):
            color_tokens.update(
                token for color in colors for token in _tokens(color)
            )
        candidate_tokens = set(color_tokens)
        for field in ("display_name", "sub_identifier", "description"):
            candidate_tokens.update(_tokens(candidate.get(field, "")))
        for field in ("keywords", "finishes"):
            values = candidate.get(field, [])
            if isinstance(values, list):
                candidate_tokens.update(
                    token for value in values for token in _tokens(value)
                )
        matched_colors = semantic_colors & color_tokens
        conflicting_colors = (
            color_tokens & _COLOR_TOKENS
        ) - semantic_colors if semantic_colors else set()
        score = 100.0
        score += 80.0 * len(matched_colors)
        score -= 35.0 * len(conflicting_colors)
        score += 4.0 * len(semantic_tokens & candidate_tokens)
        if candidate_id in primary_candidate_ids:
            score += 45.0
        audit = {
            "same_mdl_module": True,
            "semantic_color_tokens": sorted(semantic_colors),
            "candidate_color_tokens": sorted(color_tokens),
            "matched_color_tokens": sorted(matched_colors),
            "conflicting_color_tokens": sorted(conflicting_colors),
            "qwen_mvinverse_primary_candidate": candidate_id
            in primary_candidate_ids,
            "score": score,
        }
        scored.append((score, candidate_id, candidate, audit))
    if not scored:
        raise ImmutableDefaultResolutionError(
            f"no alternative exact export exists in the same MDL module: {module}"
        )
    scored.sort(key=lambda item: (-item[0], item[1]))
    score, selected_id, _candidate, audit = scored[0]
    if semantic_colors and not audit["matched_color_tokens"]:
        raise ImmutableDefaultResolutionError(
            f"no same-module export matches semantic color {sorted(semantic_colors)} "
            f"for {material_id}"
        )
    audit.update(
        {
            "source_material_id": material_id,
            "selected_material_id": selected_id,
            "candidate_count": len(scored),
            "selection_score": score,
        }
    )
    return selected_id, audit


def resolve_plan_to_immutable_defaults(
    *,
    plan: Mapping[str, Any],
    catalog: Mapping[str, Any],
    primary_candidate_ids: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a complete plan containing no material parameter mutations."""

    raw_assignments = plan.get("assignments")
    if not isinstance(raw_assignments, list) or not raw_assignments:
        raise ImmutableDefaultResolutionError("plan.assignments must be non-empty")
    catalog_materials = _materials(catalog)
    allowed_ids = {
        str(material["material_id"]) for material in catalog_materials
    }
    primary = set(primary_candidate_ids or ())
    output = copy.deepcopy(dict(plan))
    assignments = output.get("assignments")
    assert isinstance(assignments, list)
    changes: list[dict[str, Any]] = []

    def resolve_record(
        record: dict[str, Any], *, part_id: str, subset_name: str | None
    ) -> None:
        material_id = record.get("material_id")
        parameters = record.get("parameters")
        if parameters in (None, {}):
            record.pop("parameters", None)
            if material_id not in allowed_ids:
                raise ImmutableDefaultResolutionError(
                    f"plan material is absent from catalog: {material_id}"
                )
            return
        if not isinstance(parameters, dict):
            raise ImmutableDefaultResolutionError(
                f"parameters must be an object for {part_id}"
            )
        selected_id, audit = choose_exact_default_export(
            assignment=record,
            catalog_materials=catalog_materials,
            primary_candidate_ids=primary,
        )
        record["material_id"] = selected_id
        record.pop("parameters", None)
        changes.append(
            {
                "part_id": part_id,
                "subset_name": subset_name,
                **audit,
                "source_parameter_names": sorted(parameters),
                "parameter_writes": 0,
            }
        )

    part_ids: set[str] = set()
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            raise ImmutableDefaultResolutionError(
                f"plan.assignments[{index}] must be an object"
            )
        part_id = assignment.get("part_id")
        if not isinstance(part_id, str) or not part_id or part_id in part_ids:
            raise ImmutableDefaultResolutionError(
                f"invalid or duplicate part_id at assignment {index}"
            )
        part_ids.add(part_id)
        resolve_record(assignment, part_id=part_id, subset_name=None)
        subsets = assignment.get("face_subsets", [])
        if not isinstance(subsets, list):
            raise ImmutableDefaultResolutionError(
                f"face_subsets must be an array for {part_id}"
            )
        for subset_index, subset in enumerate(subsets):
            if not isinstance(subset, dict):
                raise ImmutableDefaultResolutionError(
                    f"face_subsets[{subset_index}] must be an object for {part_id}"
                )
            subset_name = subset.get("subset_name")
            if not isinstance(subset_name, str) or not subset_name:
                raise ImmutableDefaultResolutionError(
                    f"face subset lacks subset_name for {part_id}"
                )
            resolve_record(subset, part_id=part_id, subset_name=subset_name)

    provenance = output.get("provenance")
    sealed_provenance = dict(provenance) if isinstance(provenance, dict) else {}
    sealed_provenance["immutable_default_resolution"] = {
        "schema_version": SCHEMA_VERSION,
        "parameterized_record_count": len(changes),
        "parameter_writes": 0,
        "same_module_exact_exports_only": True,
        "qwen_mvinverse_primary_candidate_count": len(primary),
    }
    output["provenance"] = sealed_provenance
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "RESOLVED",
        "assignment_count": len(assignments),
        "changed_record_count": len(changes),
        "parameter_write_count": 0,
        "parameters_remaining": any(
            bool(assignment.get("parameters"))
            or any(
                bool(subset.get("parameters"))
                for subset in assignment.get("face_subsets", [])
                if isinstance(subset, dict)
            )
            for assignment in assignments
        ),
        "changes": changes,
    }
    if audit["parameters_remaining"]:
        raise AssertionError("immutable-default resolver left material parameters")
    return output, audit


def rebind_verified_plan_provenance(
    *,
    plan: dict[str, Any],
    target_plan: Mapping[str, Any],
) -> None:
    """Bind a plan to an equivalent occurrence registry after strict identity checks."""

    source_provenance = plan.get("provenance")
    target_provenance = target_plan.get("provenance")
    if not isinstance(source_provenance, dict) or not isinstance(
        target_provenance, Mapping
    ):
        raise ImmutableDefaultResolutionError(
            "source and target plans require provenance objects"
        )
    source_asset = source_provenance.get("asset_sha256")
    target_asset = target_provenance.get("asset_sha256")
    if (
        not isinstance(source_asset, str)
        or not isinstance(target_asset, str)
        or source_asset != target_asset
    ):
        raise ImmutableDefaultResolutionError(
            "source and target plans do not bind the same asset"
        )
    target_registry = target_provenance.get("registry_sha256")
    if not isinstance(target_registry, str) or not target_registry:
        raise ImmutableDefaultResolutionError(
            "target plan lacks registry_sha256 provenance"
        )
    source_ids = {
        assignment.get("part_id")
        for assignment in plan.get("assignments", [])
        if isinstance(assignment, Mapping)
    }
    target_ids = {
        assignment.get("part_id")
        for assignment in target_plan.get("assignments", [])
        if isinstance(assignment, Mapping)
    }
    if (
        None in source_ids
        or None in target_ids
        or source_ids != target_ids
        or len(source_ids) != len(plan.get("assignments", []))
    ):
        raise ImmutableDefaultResolutionError(
            "source and target plans do not cover the same unique part IDs"
        )
    source_provenance["registry_sha256"] = target_registry
    source_provenance["registry_rebound_from_verified_equivalent_plan"] = True


def _read_object(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ImmutableDefaultResolutionError(f"expected JSON object: {path}")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replace parameterized plan records with exact NVIDIA MDL exports"
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--primary-candidates", type=Path, action="append", default=[])
    parser.add_argument(
        "--target-plan-provenance",
        type=Path,
        help=(
            "equivalent complete plan whose asset/registry hashes bind the "
            "current occurrence registry"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = _read_object(args.plan.expanduser().resolve(strict=True))
    catalog = _read_object(args.catalog.expanduser().resolve(strict=True))
    primary = load_primary_candidate_ids(args.primary_candidates)
    output, audit = resolve_plan_to_immutable_defaults(
        plan=plan,
        catalog=catalog,
        primary_candidate_ids=primary,
    )
    if args.target_plan_provenance is not None:
        target_plan = _read_object(
            args.target_plan_provenance.expanduser().resolve(strict=True)
        )
        rebind_verified_plan_provenance(plan=output, target_plan=target_plan)
        audit["registry_provenance_rebound"] = True
    else:
        audit["registry_provenance_rebound"] = False
    for path, document in ((args.output, output), (args.audit, audit)):
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "assignment_count": audit["assignment_count"],
                "changed_record_count": audit["changed_record_count"],
                "output": str(args.output.expanduser().resolve()),
                "audit": str(args.audit.expanduser().resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
