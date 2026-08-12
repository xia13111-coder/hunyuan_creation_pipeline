"""Actual-CAD scoring helpers for immutable appearance-component MDL choices.

Retrieval-bank renders are useful for narrowing the NVIDIA Base catalog, but
they cannot predict how a translucent or reflective MDL will look on the
actual CAD geometry under the registered camera and lighting.  This module
keeps the final decision evidence-bounded: it creates a candidate plan that
changes one appearance component's *MDL identity only*, and scores its member
Part-ID cores after a real CAD render.  It never writes MDL parameters.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .part_id_parameter_tournament import (
    PartIdParameterTournamentError,
    score_part_id_render,
)
from .semantics import (
    FINISH_CLASSES,
    MaterialSemanticsError,
    catalog_matches_part_semantics,
    normalize_catalog_surface_semantics,
    normalize_part_material_semantics,
)


SCHEMA_VERSION = "qwen-appearance-component-actual-mdl-tournament/v1"


class ComponentMdlTournamentError(ValueError):
    """Raised when a component candidate would violate immutable-MDL rules."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _assignments(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if plan.get("schema_version") != "1.0":
        raise ComponentMdlTournamentError("material plan has an invalid schema")
    raw = plan.get("assignments")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ComponentMdlTournamentError("material plan assignments are invalid")
    output: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise ComponentMdlTournamentError(
                f"material plan assignment {index} is invalid"
            )
        # ``output`` is already a deep copy. Keep its native dict object so a
        # validated candidate mutation is reflected in the serialized plan
        # rather than only in this temporary lookup table.
        row = value if isinstance(value, dict) else dict(value)
        part_id = row.get("part_id")
        material_id = row.get("material_id")
        parameters = row.get("parameters")
        if (
            not isinstance(part_id, str)
            or not part_id
            or part_id in output
            or not isinstance(material_id, str)
            or not material_id.startswith("mdl:")
            or (
                parameters is not None
                and (not isinstance(parameters, Mapping) or bool(parameters))
            )
        ):
            raise ComponentMdlTournamentError(
                f"material plan assignment {index} is not an immutable MDL binding"
            )
        output[part_id] = row
    if not output:
        raise ComponentMdlTournamentError("material plan has no assignments")
    return output


def _member_ids(member_part_ids: Sequence[str]) -> list[str]:
    if isinstance(member_part_ids, (str, bytes)):
        raise ComponentMdlTournamentError("component member Part-IDs are invalid")
    members = sorted(member_part_ids)
    if (
        len(members) < 2
        or len(members) != len(set(members))
        or any(not isinstance(part_id, str) or not part_id for part_id in members)
    ):
        raise ComponentMdlTournamentError(
            "component needs at least two unique non-empty Part-IDs"
        )
    return members


_FORBIDDEN_PROXY_TOKENS = frozenset({"grass", "mirror", "water"})
_METAL_SURFACE_TREATMENTS = frozenset(
    {
        "anodized",
        "bare",
        "conversion_coating",
        "galvanized",
        "oxidized",
        "plated",
    }
)
_FALLBACK_FINISH_ORDER = {
    "matte": 0,
    "satin": 1,
    "glossy": 2,
    "polished": 3,
    "brushed": 4,
    "smooth": 5,
    "rough": 6,
    "textured": 7,
    "weathered": 8,
    "unknown": 9,
}


def _semantic_gate_requested(
    member_material_semantics: Mapping[str, Mapping[str, Any]] | None,
    catalog_materials_by_id: Mapping[str, Mapping[str, Any]] | None,
) -> bool:
    requested = (
        member_material_semantics is not None
        or catalog_materials_by_id is not None
    )
    if requested and (
        member_material_semantics is None or catalog_materials_by_id is None
    ):
        raise ComponentMdlTournamentError(
            "strict component semantics require both member semantics and the catalog"
        )
    return requested


def _strict_member_semantics(
    member_material_semantics: Mapping[str, Mapping[str, Any]],
    *,
    expected_member_part_ids: Sequence[str] | None = None,
    preferred_finish: str | None = None,
) -> tuple[dict[str, dict[str, Any]], str, str | None]:
    """Validate the common opaque Paint/Metal contract for one component."""

    if not isinstance(member_material_semantics, Mapping):
        raise ComponentMdlTournamentError(
            "component member material semantics must be a mapping"
        )
    if len(member_material_semantics) < 2:
        raise ComponentMdlTournamentError(
            "strict component semantics need at least two members"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for part_id, value in member_material_semantics.items():
        if not isinstance(part_id, str) or not part_id or part_id in normalized:
            raise ComponentMdlTournamentError(
                "component member material semantics have an invalid Part-ID"
            )
        try:
            semantics = normalize_part_material_semantics(value)
        except (MaterialSemanticsError, TypeError) as exc:
            raise ComponentMdlTournamentError(
                f"component member semantics are malformed for {part_id}: {exc}"
            ) from exc
        if (
            semantics["evidence_status"] != "observed"
            or semantics["substrate"] == "unknown"
            or semantics["surface_treatment"] == "unknown"
            or semantics["optical_behavior"] == "unknown"
        ):
            raise ComponentMdlTournamentError(
                f"component member semantics are not observed and resolved for {part_id}"
            )
        normalized[part_id] = semantics
    if expected_member_part_ids is not None:
        expected = set(_member_ids(expected_member_part_ids))
        if set(normalized) != expected:
            raise ComponentMdlTournamentError(
                "component member semantics do not exactly cover the candidate members"
            )

    treatments = {value["surface_treatment"] for value in normalized.values()}
    optical_behaviors = {value["optical_behavior"] for value in normalized.values()}
    known_finishes = {
        value["finish"]
        for value in normalized.values()
        if value["finish"] != "unknown"
    }
    if len(treatments) != 1 or len(optical_behaviors) != 1 or len(known_finishes) > 1:
        raise ComponentMdlTournamentError(
            "component member material semantics conflict"
        )
    treatment = next(iter(treatments))
    optical = next(iter(optical_behaviors))
    substrates = {value["substrate"] for value in normalized.values()}
    if optical != "opaque":
        raise ComponentMdlTournamentError(
            "strict component tournament supports only opaque surfaces"
        )
    if treatment == "paint":
        target_family = "paint"
    elif substrates == {"metal"} and treatment in _METAL_SURFACE_TREATMENTS:
        target_family = "metal"
    else:
        raise ComponentMdlTournamentError(
            "strict component tournament supports only compatible Paint or Metal surfaces"
        )

    if preferred_finish is not None and (
        not isinstance(preferred_finish, str)
        or preferred_finish not in FINISH_CLASSES
        or preferred_finish == "unknown"
    ):
        raise ComponentMdlTournamentError(
            "preferred_finish must be a known catalog finish"
        )
    inferred_finish = next(iter(known_finishes), None)
    if (
        preferred_finish is not None
        and inferred_finish is not None
        and preferred_finish != inferred_finish
    ):
        raise ComponentMdlTournamentError(
            "preferred_finish conflicts with component member semantics"
        )
    return normalized, target_family, preferred_finish or inferred_finish


def _proxy_material_id(material_id: str) -> bool:
    tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", material_id.casefold())
        if token
    }
    return bool(tokens & _FORBIDDEN_PROXY_TOKENS)


def _catalog_candidate_semantics(
    *,
    material_id: str,
    catalog_materials_by_id: Mapping[str, Mapping[str, Any]],
    member_material_semantics: Mapping[str, Mapping[str, Any]],
    target_family: str,
) -> dict[str, Any] | None:
    """Return normalized metadata only for a fully compatible strict candidate."""

    if (
        not isinstance(material_id, str)
        or not material_id.startswith("mdl:")
        or _proxy_material_id(material_id)
    ):
        return None
    raw_record = catalog_materials_by_id.get(material_id)
    if not isinstance(raw_record, Mapping):
        return None
    recorded_id = raw_record.get("material_id")
    if recorded_id is not None and recorded_id != material_id:
        return None
    family = raw_record.get("family")
    if not isinstance(family, str) or family.casefold() != target_family:
        return None
    raw_surface = raw_record.get("surface_semantics")
    if not isinstance(raw_surface, Mapping):
        return None
    try:
        surface = normalize_catalog_surface_semantics(raw_surface)
        if surface["confidence"] != "high":
            return None
        if not all(
            catalog_matches_part_semantics(surface, semantics)
            for semantics in member_material_semantics.values()
        ):
            return None
    except (MaterialSemanticsError, TypeError):
        return None
    return {
        "family": family.casefold(),
        "surface_semantics": surface,
    }


def _paint_alias_key(material_id: str, family: str) -> str:
    if family != "paint":
        return material_id.casefold()
    sub_identifier = material_id.rsplit("#", 1)[-1].casefold()
    if sub_identifier.endswith("_finish"):
        sub_identifier = sub_identifier[: -len("_finish")]
    return f"paint:{sub_identifier}"


def _strict_component_candidate_material_ids(
    *,
    baseline_material_id: str,
    retrieval_group: Mapping[str, Any],
    visual_compatibility: Mapping[str, Any] | None,
    maximum_candidates: int,
    member_material_semantics: Mapping[str, Mapping[str, Any]],
    catalog_materials_by_id: Mapping[str, Mapping[str, Any]],
    preferred_finish: str | None,
) -> list[str]:
    normalized_members, target_family, target_finish = _strict_member_semantics(
        member_material_semantics,
        preferred_finish=preferred_finish,
    )
    if not isinstance(catalog_materials_by_id, Mapping):
        raise ComponentMdlTournamentError(
            "strict component material catalog must be a mapping"
        )
    limit = min(3, maximum_candidates)
    selected: list[str] = []
    selected_aliases: set[str] = set()

    def compatible_metadata(material_id: Any) -> dict[str, Any] | None:
        if not isinstance(material_id, str):
            return None
        return _catalog_candidate_semantics(
            material_id=material_id,
            catalog_materials_by_id=catalog_materials_by_id,
            member_material_semantics=normalized_members,
            target_family=target_family,
        )

    def add(material_id: Any) -> None:
        if len(selected) >= limit or not isinstance(material_id, str):
            return
        metadata = compatible_metadata(material_id)
        if metadata is None:
            return
        alias = _paint_alias_key(material_id, metadata["family"])
        if material_id not in selected and alias not in selected_aliases:
            selected.append(material_id)
            selected_aliases.add(alias)

    # The legacy baseline remains H0 only when it passes the same semantic gate
    # as every challenger. Otherwise the first safe ranked/fallback material is
    # the new effective H0 returned at index zero.
    add(baseline_material_id)

    raw_color_ranking = retrieval_group.get("color_ranking")
    if isinstance(raw_color_ranking, Sequence) and not isinstance(
        raw_color_ranking, (str, bytes)
    ):
        for row in sorted(
            (row for row in raw_color_ranking if isinstance(row, Mapping)),
            key=lambda row: (
                int(row["rank"])
                if isinstance(row.get("rank"), int)
                and not isinstance(row.get("rank"), bool)
                else 1_000_000,
                str(row.get("material_id", "")),
            ),
        ):
            add(row.get("material_id"))

    raw_shortlist = (
        visual_compatibility.get("shortlist")
        if isinstance(visual_compatibility, Mapping)
        else None
    )
    if isinstance(raw_shortlist, Sequence) and not isinstance(
        raw_shortlist, (str, bytes)
    ):
        for row in sorted(
            (row for row in raw_shortlist if isinstance(row, Mapping)),
            key=lambda row: (
                int(row["compatibility_rank"])
                if isinstance(row.get("compatibility_rank"), int)
                and not isinstance(row.get("compatibility_rank"), bool)
                else 1_000_000,
                str(row.get("material_id", "")),
            ),
        ):
            add(row.get("material_id"))

    raw_fused_ranking = retrieval_group.get("fused_ranking")
    if isinstance(raw_fused_ranking, Sequence) and not isinstance(
        raw_fused_ranking, (str, bytes)
    ):
        for row in raw_fused_ranking:
            if isinstance(row, Mapping):
                add(row.get("material_id"))

    def fallback_key(material_id: str) -> tuple[int, int, int, str]:
        metadata = compatible_metadata(material_id)
        if metadata is None:
            return (1, 1, 1_000_000, material_id)
        surface = metadata["surface_semantics"]
        finish = surface["finish"]
        return (
            0 if target_finish is not None and finish == target_finish else 1,
            0 if not material_id.rsplit("#", 1)[-1].casefold().endswith("_finish") else 1,
            _FALLBACK_FINISH_ORDER.get(finish, 1_000_000),
            material_id,
        )

    compatible_catalog_ids = [
        material_id
        for material_id in catalog_materials_by_id
        if isinstance(material_id, str) and compatible_metadata(material_id) is not None
    ]
    for material_id in sorted(compatible_catalog_ids, key=fallback_key):
        add(material_id)

    if len(selected) < 2:
        raise ComponentMdlTournamentError(
            "strict component semantics yielded fewer than two compatible MDL candidates"
        )
    return selected


def component_candidate_material_ids(
    *,
    baseline_material_id: str,
    retrieval_group: Mapping[str, Any],
    visual_compatibility: Mapping[str, Any] | None,
    maximum_candidates: int = 4,
    member_material_semantics: Mapping[str, Mapping[str, Any]] | None = None,
    catalog_materials_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    preferred_finish: str | None = None,
) -> list[str]:
    """Build a small, visually diverse fixed-MDL tournament shortlist.

    The retrieval-bank winner is always retained as the baseline.  The next
    two slots favour the independently computed colour ranking, since this is
    the cue that a library thumbnail often gets right even when its geometry
    or transmission is a poor proxy for the CAD part.  The final slot favours
    the compatibility gate's best different candidate.  The actual CAD render
    is the only final authority, so candidates need not already pass the
    pre-render physical-risk gate.
    """

    if not isinstance(baseline_material_id, str) or not baseline_material_id.startswith(
        "mdl:"
    ):
        raise ComponentMdlTournamentError("baseline_material_id must be an MDL ID")
    if (
        isinstance(maximum_candidates, bool)
        or not isinstance(maximum_candidates, int)
        or maximum_candidates < 2
    ):
        raise ComponentMdlTournamentError("maximum_candidates must be an integer >= 2")
    if not isinstance(retrieval_group, Mapping):
        raise ComponentMdlTournamentError("retrieval group is invalid")

    if _semantic_gate_requested(
        member_material_semantics,
        catalog_materials_by_id,
    ):
        assert member_material_semantics is not None
        assert catalog_materials_by_id is not None
        return _strict_component_candidate_material_ids(
            baseline_material_id=baseline_material_id,
            retrieval_group=retrieval_group,
            visual_compatibility=visual_compatibility,
            maximum_candidates=maximum_candidates,
            member_material_semantics=member_material_semantics,
            catalog_materials_by_id=catalog_materials_by_id,
            preferred_finish=preferred_finish,
        )
    if preferred_finish is not None:
        raise ComponentMdlTournamentError(
            "preferred_finish is available only with strict component semantics"
        )

    selected: list[str] = []

    def add(material_id: Any) -> None:
        if (
            isinstance(material_id, str)
            and material_id.startswith("mdl:")
            and material_id not in selected
            and len(selected) < maximum_candidates
        ):
            selected.append(material_id)

    add(baseline_material_id)
    raw_color_ranking = retrieval_group.get("color_ranking")
    if isinstance(raw_color_ranking, Sequence) and not isinstance(
        raw_color_ranking, (str, bytes)
    ):
        ranked_color_rows = sorted(
            (row for row in raw_color_ranking if isinstance(row, Mapping)),
            key=lambda row: (
                int(row["rank"])
                if isinstance(row.get("rank"), int)
                and not isinstance(row.get("rank"), bool)
                else 1_000_000,
                str(row.get("material_id", "")),
            ),
        )
        for row in ranked_color_rows[:2]:
            add(row.get("material_id"))

    raw_shortlist = (
        visual_compatibility.get("shortlist")
        if isinstance(visual_compatibility, Mapping)
        else None
    )
    if isinstance(raw_shortlist, Sequence) and not isinstance(
        raw_shortlist, (str, bytes)
    ):
        ranked_compatibility_rows = sorted(
            (row for row in raw_shortlist if isinstance(row, Mapping)),
            key=lambda row: (
                int(row["compatibility_rank"])
                if isinstance(row.get("compatibility_rank"), int)
                and not isinstance(row.get("compatibility_rank"), bool)
                else 1_000_000,
                str(row.get("material_id", "")),
            ),
        )
        for row in ranked_compatibility_rows:
            add(row.get("material_id"))
            if len(selected) >= maximum_candidates:
                break

    if len(selected) < 2:
        raw_fused_ranking = retrieval_group.get("fused_ranking")
        if isinstance(raw_fused_ranking, Sequence) and not isinstance(
            raw_fused_ranking, (str, bytes)
        ):
            for row in raw_fused_ranking:
                if isinstance(row, Mapping):
                    add(row.get("material_id"))
                if len(selected) >= maximum_candidates:
                    break
    if len(selected) < 2:
        raise ComponentMdlTournamentError(
            "component retrieval did not yield a second fixed-MDL candidate"
        )
    return selected


def build_component_candidate_plan(
    *,
    source_plan: Mapping[str, Any],
    component_id: str,
    member_part_ids: Sequence[str],
    material_id: str,
    member_material_semantics: Mapping[str, Mapping[str, Any]] | None = None,
    catalog_materials_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a candidate that changes one component to one fixed Base MDL."""

    if not isinstance(component_id, str) or not component_id:
        raise ComponentMdlTournamentError("component_id must be non-empty")
    if not isinstance(material_id, str) or not material_id.startswith("mdl:"):
        raise ComponentMdlTournamentError("candidate material_id must be an MDL ID")
    members = _member_ids(member_part_ids)
    semantic_gate: dict[str, Any] | None = None
    if _semantic_gate_requested(
        member_material_semantics,
        catalog_materials_by_id,
    ):
        assert member_material_semantics is not None
        assert catalog_materials_by_id is not None
        normalized_members, target_family, _target_finish = _strict_member_semantics(
            member_material_semantics,
            expected_member_part_ids=members,
        )
        if (
            _catalog_candidate_semantics(
                material_id=material_id,
                catalog_materials_by_id=catalog_materials_by_id,
                member_material_semantics=normalized_members,
                target_family=target_family,
            )
            is None
        ):
            raise ComponentMdlTournamentError(
                "component candidate MDL is not compatible with every member"
            )
        semantic_gate = {
            "policy": "all_component_members_physical_semantics_compatible/v1",
            "member_material_semantics_sha256": _canonical_sha256(
                normalized_members
            ),
            "catalog_material_record_sha256": _canonical_sha256(
                catalog_materials_by_id[material_id]
            ),
            "target_family": target_family,
        }
    output = copy.deepcopy(dict(source_plan))
    assignments = _assignments(output)
    missing = sorted(set(members) - set(assignments))
    if missing:
        raise ComponentMdlTournamentError(
            f"component {component_id} has unknown Part-IDs: {missing}"
        )
    for part_id in members:
        assignment = assignments[part_id]
        assignment["material_id"] = material_id
        assignment.pop("parameters", None)
        provenance = assignment.get("provenance")
        updated_provenance = dict(provenance) if isinstance(provenance, Mapping) else {}
        updated_provenance["appearance_component_actual_mdl_candidate"] = {
            "component_id": component_id,
            "member_part_ids": members,
            "material_id": material_id,
            "source_plan_sha256": _canonical_sha256(source_plan),
            "mdl_parameter_mutation_allowed": False,
        }
        if semantic_gate is not None:
            updated_provenance["appearance_component_actual_mdl_candidate"][
                "semantic_compatibility_gate"
            ] = copy.deepcopy(semantic_gate)
        assignment["provenance"] = updated_provenance
    provenance = output.get("provenance")
    output_provenance = dict(provenance) if isinstance(provenance, Mapping) else {}
    output_provenance["appearance_component_actual_mdl_candidate"] = {
        "component_id": component_id,
        "member_part_ids": members,
        "material_id": material_id,
        "source_plan_sha256": _canonical_sha256(source_plan),
        "mdl_parameter_mutation_allowed": False,
    }
    if semantic_gate is not None:
        output_provenance["appearance_component_actual_mdl_candidate"][
            "semantic_compatibility_gate"
        ] = copy.deepcopy(semantic_gate)
    output["provenance"] = output_provenance
    return output


def score_component_render(
    *,
    component_id: str,
    member_part_ids: Sequence[str],
    evidence: Mapping[str, Any],
    spatial_mapping_report: Mapping[str, Any],
    rendered_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a pixel-weighted real-CAD similarity score for one component."""

    if not isinstance(component_id, str) or not component_id:
        raise ComponentMdlTournamentError("component_id must be non-empty")
    members = _member_ids(member_part_ids)
    scores: list[dict[str, Any]] = []
    for part_id in members:
        try:
            score = score_part_id_render(
                part_id=part_id,
                evidence=evidence,
                spatial_mapping_report=spatial_mapping_report,
                rendered_registry=rendered_registry,
            )
        except PartIdParameterTournamentError as exc:
            raise ComponentMdlTournamentError(
                f"component {component_id} could not score {part_id}: {exc}"
            ) from exc
        scores.append(score)
    total_pixels = sum(int(row["comparison_pixel_count"]) for row in scores)
    if total_pixels <= 0:
        raise ComponentMdlTournamentError(
            f"component {component_id} has no registered comparison pixels"
        )

    def weighted(key: str) -> float:
        return sum(
            float(row[key]) * int(row["comparison_pixel_count"]) for row in scores
        ) / total_pixels

    return {
        "schema_version": SCHEMA_VERSION,
        "component_id": component_id,
        "member_part_ids": members,
        "member_score_count": len(scores),
        "comparison_pixel_count": total_pixels,
        "appearance_score": round(weighted("appearance_score"), 8),
        "color_score": round(weighted("color_score"), 8),
        "luma_score": round(weighted("luma_score"), 8),
        "lab_delta_e": round(weighted("lab_delta_e"), 8),
        "member_scores": scores,
    }


def select_component_mdl_winner(
    *,
    component_id: str,
    baseline_material_id: str,
    candidate_scores: Mapping[str, Mapping[str, Any]],
    minimum_score_improvement: float = 0.015,
    member_material_semantics: Mapping[str, Mapping[str, Any]] | None = None,
    catalog_materials_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    authorized_candidate_material_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Choose a fixed MDL only when real-CAD evidence clearly improves it."""

    if not isinstance(component_id, str) or not component_id:
        raise ComponentMdlTournamentError("component_id must be non-empty")
    if not isinstance(baseline_material_id, str) or not baseline_material_id.startswith(
        "mdl:"
    ):
        raise ComponentMdlTournamentError("baseline_material_id must be an MDL ID")
    if (
        isinstance(minimum_score_improvement, bool)
        or not isinstance(minimum_score_improvement, (int, float))
        or not math.isfinite(float(minimum_score_improvement))
        or not 0.0 <= float(minimum_score_improvement) <= 1.0
    ):
        raise ComponentMdlTournamentError(
            "minimum_score_improvement must be a finite unit number"
        )
    if baseline_material_id not in candidate_scores:
        raise ComponentMdlTournamentError("candidate scores lack the baseline MDL")
    strict_semantics: tuple[dict[str, dict[str, Any]], str] | None = None
    if _semantic_gate_requested(
        member_material_semantics,
        catalog_materials_by_id,
    ):
        assert member_material_semantics is not None
        assert catalog_materials_by_id is not None
        normalized_members, target_family, _target_finish = _strict_member_semantics(
            member_material_semantics
        )
        strict_semantics = (normalized_members, target_family)
        if not 2 <= len(candidate_scores) <= 3:
            raise ComponentMdlTournamentError(
                "strict component winner needs two or three candidate scores"
            )
        if authorized_candidate_material_ids is not None:
            if isinstance(authorized_candidate_material_ids, (str, bytes)):
                raise ComponentMdlTournamentError(
                    "authorized component candidates are invalid"
                )
            authorized = list(authorized_candidate_material_ids)
            if (
                len(authorized) != len(set(authorized))
                or set(authorized) != set(candidate_scores)
                or not 2 <= len(authorized) <= 3
            ):
                raise ComponentMdlTournamentError(
                    "candidate scores do not exactly match the authorized shortlist"
                )
    elif authorized_candidate_material_ids is not None:
        raise ComponentMdlTournamentError(
            "authorized candidate binding requires strict component semantics"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for material_id, value in candidate_scores.items():
        score = value.get("appearance_score") if isinstance(value, Mapping) else None
        if (
            not isinstance(material_id, str)
            or not material_id.startswith("mdl:")
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ComponentMdlTournamentError("candidate score is invalid")
        if strict_semantics is not None:
            assert catalog_materials_by_id is not None
            normalized_members, target_family = strict_semantics
            if (
                _catalog_candidate_semantics(
                    material_id=material_id,
                    catalog_materials_by_id=catalog_materials_by_id,
                    member_material_semantics=normalized_members,
                    target_family=target_family,
                )
                is None
            ):
                raise ComponentMdlTournamentError(
                    "candidate score map contains a semantically incompatible MDL"
                )
        normalized[material_id] = dict(value)
    winner_id = min(
        normalized,
        key=lambda material_id: (-float(normalized[material_id]["appearance_score"]), material_id),
    )
    baseline_score = float(normalized[baseline_material_id]["appearance_score"])
    winner_score = float(normalized[winner_id]["appearance_score"])
    accepted = winner_id != baseline_material_id and (
        winner_score >= baseline_score + float(minimum_score_improvement)
    )
    selected_id = winner_id if accepted else baseline_material_id
    return {
        "schema_version": SCHEMA_VERSION,
        "component_id": component_id,
        "baseline_material_id": baseline_material_id,
        "selected_material_id": selected_id,
        "winning_candidate_material_id": winner_id,
        "baseline_appearance_score": round(baseline_score, 8),
        "winning_appearance_score": round(winner_score, 8),
        "score_improvement": round(winner_score - baseline_score, 8),
        "minimum_score_improvement": float(minimum_score_improvement),
        "selection_status": "ACTUAL_CAD_RENDER_WINNER" if accepted else "BASELINE_RETAINED",
        "mdl_parameter_mutation_allowed": False,
    }


def rebind_part_id_material_audit_for_component_mdl_tournament(
    *,
    source_audit: Mapping[str, Any],
    final_plan: Mapping[str, Any],
    tournament_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a Part-ID exact-cover audit to immutable component-MDL winners.

    The publication gate deliberately verifies each observed audit row against
    the final plan. A real-CAD component winner therefore refreshes its row
    material IDs and final-plan hash, while hidden Part-IDs remain unchanged.
    """

    output = copy.deepcopy(dict(source_audit))
    raw_rows = output.get("parts")
    if not isinstance(raw_rows, list):
        raise ComponentMdlTournamentError("Part-ID material audit has no parts")
    assignments = _assignments(final_plan)
    rows_by_part: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, dict):
            raise ComponentMdlTournamentError(
                f"Part-ID material audit row {index} is invalid"
            )
        part_id = raw_row.get("part_id")
        if not isinstance(part_id, str) or not part_id or part_id in rows_by_part:
            raise ComponentMdlTournamentError(
                f"Part-ID material audit row {index} has an invalid Part-ID"
            )
        rows_by_part[part_id] = raw_row
    if set(rows_by_part) != set(assignments):
        raise ComponentMdlTournamentError(
            "Part-ID material audit does not exactly cover the final plan"
        )

    raw_components = tournament_audit.get("components")
    if not isinstance(raw_components, list):
        raise ComponentMdlTournamentError("component tournament audit has no components")
    winner_by_part: dict[str, Mapping[str, Any]] = {}
    for component in raw_components:
        if not isinstance(component, Mapping):
            raise ComponentMdlTournamentError("component tournament row is invalid")
        members = component.get("member_part_ids")
        selected_material_id = component.get("selected_material_id")
        baseline_material_id = component.get("baseline_material_id")
        if (
            not isinstance(members, list)
            or not isinstance(selected_material_id, str)
            or not selected_material_id.startswith("mdl:")
            or not isinstance(baseline_material_id, str)
            or not baseline_material_id.startswith("mdl:")
        ):
            raise ComponentMdlTournamentError("component tournament row is malformed")
        if selected_material_id == baseline_material_id:
            continue
        for part_id in members:
            if (
                not isinstance(part_id, str)
                or not part_id
                or part_id in winner_by_part
            ):
                raise ComponentMdlTournamentError(
                    "component tournament winner Part-IDs are invalid"
                )
            winner_by_part[part_id] = component

    for part_id, assignment in assignments.items():
        row = rows_by_part[part_id]
        row_status = row.get("status")
        if row_status == "unobserved_preserved":
            if row.get("material_id") != assignment.get("material_id"):
                raise ComponentMdlTournamentError(
                    f"unobserved Part-ID {part_id} changed in component tournament"
                )
            continue
        if row_status != "independently_selected":
            raise ComponentMdlTournamentError(
                f"Part-ID material audit has unsupported status for {part_id}"
            )
        material_id = assignment.get("material_id")
        if not isinstance(material_id, str) or not material_id.startswith("mdl:"):
            raise ComponentMdlTournamentError(
                f"final immutable MDL binding is invalid for {part_id}"
            )
        row["material_id"] = material_id
        winner = winner_by_part.get(part_id)
        if winner is not None:
            row["appearance_component_actual_mdl_tournament"] = {
                "component_id": winner.get("component_id"),
                "selected_material_id": material_id,
                "selection_status": winner.get("selection_status"),
                "mdl_parameter_mutation_allowed": False,
            }

    summary = output.get("summary")
    if not isinstance(summary, dict):
        raise ComponentMdlTournamentError("Part-ID material audit has no summary")
    independently_selected_count = sum(
        row.get("status") == "independently_selected" for row in rows_by_part.values()
    )
    unobserved_preserved_count = sum(
        row.get("status") == "unobserved_preserved" for row in rows_by_part.values()
    )
    if (
        summary.get("part_count") != len(rows_by_part)
        or summary.get("independently_selected_count") != independently_selected_count
        or summary.get("unobserved_preserved_count") != unobserved_preserved_count
        or summary.get("exact_cover") is not True
    ):
        raise ComponentMdlTournamentError(
            "Part-ID material audit summary does not match its rows"
        )
    output["output_plan_sha256"] = _canonical_sha256(final_plan)
    output["appearance_component_actual_mdl_tournament"] = {
        "schema_version": SCHEMA_VERSION,
        "audit_sha256": _canonical_sha256(tournament_audit),
        "winner_component_count": len(
            {str(component.get("component_id")) for component in winner_by_part.values()}
        ),
        "winner_part_count": len(winner_by_part),
        "mdl_parameter_mutation_allowed": False,
    }
    output.pop("integrity", None)
    output["integrity"] = {"document_sha256": _canonical_sha256(output)}
    return output


__all__ = [
    "SCHEMA_VERSION",
    "ComponentMdlTournamentError",
    "build_component_candidate_plan",
    "score_component_render",
    "select_component_mdl_winner",
]
