"""Coordinate immutable exact-MDL render tournaments across visual groups.

The single-group tournament deliberately changes one canonical group at a
time.  This module keeps that attribution boundary while extending it to every
declared significant group:

* the baseline is always rendered as a candidate;
* primary Qwen/MVInverse candidates are merged with the wider tournament pool
  without allowing the wider pool to evict them;
* every later round is generated from the plan accepted by the previous round;
* complete all-view-PASS evidence outranks complete non-failing REVIEW evidence;
* REVIEW evidence can replace a FAIL baseline only with complete multi-view
  scores and configured baseline/runner-up margins; and
* material parameters and face-subset parameters remain at NVIDIA library
  defaults.

Rendering remains an orchestration concern.  :func:`coordinate_descent_exact_mdl_groups`
therefore accepts a round provider which can apply, render, compare, and return
the hash-linked candidate bundles for the current plan.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .exact_mdl_tournament import (
    ExactMdlTournamentError,
    QUALITY_TIER_ALL_VIEW_PASS,
    QUALITY_TIER_COMPLETE_NONFAIL_REVIEW,
    QUALITY_TIER_INELIGIBLE,
    QUALITY_TIER_RANK,
    SELECTION_OBJECTIVE_SEMANTIC,
    SELECTION_OBJECTIVE_VISUAL,
    build_part_family_contract,
    material_entity_contract_key,
    select_and_replay_exact_mdl_candidate,
)
from .membership_tournament import (
    MembershipTournamentError,
    membership_exclusions_by_group,
)
from ..evidence.reference_compare import MIN_TRUSTED_EVIDENCE_PIXELS


SCHEMA_VERSION = "qwen-multigroup-exact-mdl-coordinate-descent/v1"
QUEUE_SCHEMA_VERSION = "qwen-multigroup-exact-mdl-queue/v1"
PLANNING_SCHEMA_VERSION = "qwen-exact-mdl-group-candidate-planning/v1"
ROUND_SCHEMA_VERSION = "qwen-exact-mdl-group-round/v1"
VISUAL_GROUP_ANNOTATION_AUDIT_SCHEMA_VERSION = (
    "qwen-visual-group-plan-annotation/v1"
)
SOURCE_APPEARANCE_COHORT_SCHEMA_VERSION = (
    "qwen-source-appearance-cohort-propagation/v1"
)
SOURCE_APPEARANCE_COHORT_CONTRACT_SCHEMA_VERSION = (
    "qwen-source-appearance-cohort-contract/v1"
)
SOURCE_APPEARANCE_COHORT_METHOD = (
    "trusted_spatial_anchor_source_appearance_cohort/v1"
)

DEFAULT_MINIMUM_SCORE_IMPROVEMENT = 0.015
DEFAULT_MINIMUM_WINNER_MARGIN = 0.005

ROUND_ACCEPTED = "ACCEPTED"
ROUND_FALLBACK_BASELINE_BEST = "FALLBACK_BASELINE_BEST"
ROUND_FALLBACK_BASELINE_INELIGIBLE = "FALLBACK_BASELINE_INELIGIBLE"
ROUND_FALLBACK_INSUFFICIENT_IMPROVEMENT = "FALLBACK_INSUFFICIENT_IMPROVEMENT"
ROUND_FALLBACK_AMBIGUOUS_WINNER = "FALLBACK_AMBIGUOUS_WINNER"
ROUND_FALLBACK_NO_ELIGIBLE_CANDIDATE = "FALLBACK_NO_ELIGIBLE_CANDIDATE"
BASELINE_GROUP_PRESENT_WITHOUT_LOCALIZATION = (
    "BASELINE_GROUP_PRESENT_WITHOUT_LOCALIZATION"
)
INSUFFICIENT_TRUSTED_SCORING_REFERENCE_VIEWS = (
    "INSUFFICIENT_TRUSTED_SCORING_REFERENCE_VIEWS"
)

_VISUAL_RETRIEVAL_STRATEGY = "siglip2_full_catalog_plus_dinov2_masked_rrf/v1"
_BASE_BANK_RETRIEVAL_STRATEGY = (
    "base_observation_bank_siglip2_dinov2_color_mvinverse_rrf/v1"
)
_VISUAL_RETRIEVAL_STRATEGIES = {
    _VISUAL_RETRIEVAL_STRATEGY,
    _BASE_BANK_RETRIEVAL_STRATEGY,
}
_SELECTION_FALLBACK_STRATEGY = "family_gated_semantic_mvinverse_similarity_score/v12"
_TOURNAMENT_FALLBACK_STRATEGY = "visual_mvinverse_similarity_score/v1"
_VISUAL_RETRIEVAL_FINAL_AUTHORITY = "exact_mdl_render_tournament"
_RRF_RANK_CONSTANT = 60
_MISSING_RANK = 2**31 - 1
_VISUAL_LANE = "visual"
_SELECTION_FALLBACK_LANE = "selection_fallback"
_TOURNAMENT_FALLBACK_LANE = "tournament_fallback"
_LANE_MINIMUM_ORDER = (
    _SELECTION_FALLBACK_LANE,
    _TOURNAMENT_FALLBACK_LANE,
    _VISUAL_LANE,
)
_LANE_GROWTH_ORDER = (
    _TOURNAMENT_FALLBACK_LANE,
    _SELECTION_FALLBACK_LANE,
    _VISUAL_LANE,
)


class MultigroupExactMdlTournamentError(ValueError):
    """Raised when the multi-group trust boundary is violated."""


RoundProvider = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Sequence[Mapping[str, Any]],
]


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MultigroupExactMdlTournamentError(
            f"document is not canonical JSON: {exc}"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _unit_interval(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise MultigroupExactMdlTournamentError(
            f"{label} must be a finite number from 0 to 1"
        )
    return float(value)


def _sorted_unique_texts(
    values: Sequence[Any],
    label: str,
    *,
    require_sorted: bool = False,
) -> list[str]:
    result: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value:
            raise MultigroupExactMdlTournamentError(
                f"{label}[{index}] must be a non-empty string"
            )
        result.append(value)
    if len(result) != len(set(result)):
        raise MultigroupExactMdlTournamentError(f"{label} must not contain duplicates")
    if require_sorted and result != sorted(result):
        raise MultigroupExactMdlTournamentError(f"{label} must be sorted")
    return result


def _assignments(
    plan: Mapping[str, Any],
    label: str,
) -> dict[str, dict[str, Any]]:
    if plan.get("schema_version") != "1.0":
        raise MultigroupExactMdlTournamentError(f"{label} has an invalid schema")
    raw_assignments = plan.get("assignments")
    if not isinstance(raw_assignments, Sequence) or isinstance(
        raw_assignments, (str, bytes)
    ):
        raise MultigroupExactMdlTournamentError(f"{label}.assignments must be an array")
    output: dict[str, dict[str, Any]] = {}
    for index, raw_assignment in enumerate(raw_assignments):
        if not isinstance(raw_assignment, Mapping):
            raise MultigroupExactMdlTournamentError(
                f"{label}.assignments[{index}] must be an object"
            )
        assignment = copy.deepcopy(dict(raw_assignment))
        part_id = assignment.get("part_id")
        material_id = assignment.get("material_id")
        if (
            not isinstance(part_id, str)
            or not part_id
            or part_id in output
            or not isinstance(material_id, str)
            or not material_id.startswith("mdl:")
        ):
            raise MultigroupExactMdlTournamentError(
                f"{label}.assignments[{index}] is invalid"
            )
        _require_library_defaults(assignment, f"{label}/{part_id}")
        output[part_id] = assignment
    if not output:
        raise MultigroupExactMdlTournamentError(f"{label} has no assignments")
    return output


def _face_subsets(
    assignment: Mapping[str, Any],
    label: str,
) -> dict[str, dict[str, Any]]:
    raw_subsets = assignment.get("face_subsets", [])
    if raw_subsets is None:
        return {}
    if not isinstance(raw_subsets, Sequence) or isinstance(
        raw_subsets,
        (str, bytes),
    ):
        raise MultigroupExactMdlTournamentError(
            f"{label}.face_subsets must be an array"
        )
    output: dict[str, dict[str, Any]] = {}
    for index, raw_subset in enumerate(raw_subsets):
        if not isinstance(raw_subset, Mapping):
            raise MultigroupExactMdlTournamentError(
                f"{label}.face_subsets[{index}] must be an object"
            )
        subset = copy.deepcopy(dict(raw_subset))
        subset_name = subset.get("subset_name")
        material_id = subset.get("material_id")
        if (
            not isinstance(subset_name, str)
            or not subset_name
            or subset_name in output
            or not isinstance(material_id, str)
            or not material_id.startswith("mdl:")
        ):
            raise MultigroupExactMdlTournamentError(
                f"{label}.face_subsets[{index}] has an invalid material binding"
            )
        _require_library_defaults(
            {"face_subsets": [subset]},
            f"{label}.face_subsets[{index}]",
        )
        output[subset_name] = subset
    return output


def _entity_key(entity: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(entity["part_id"]),
        str(entity.get("subset_name", "")),
    )


def _normalized_target_entities(
    *,
    target_part_ids: Sequence[str],
    target_entities: Sequence[Mapping[str, Any]] | None,
    assignments: Mapping[str, Mapping[str, Any]],
    label: str,
) -> list[dict[str, str]]:
    """Normalize assignment/subset targets without adding subset schema fields."""

    part_ids = _sorted_unique_texts(
        list(target_part_ids),
        f"{label}.target_part_ids",
        require_sorted=True,
    )
    if not part_ids:
        raise MultigroupExactMdlTournamentError(
            f"{label}.target_part_ids must not be empty"
        )
    missing_parts = sorted(set(part_ids) - set(assignments))
    if missing_parts:
        raise MultigroupExactMdlTournamentError(
            f"{label} targets parts absent from the plan: " + ", ".join(missing_parts)
        )
    raw_entities: Sequence[Any] = (
        [
            {
                "entity_kind": "assignment",
                "part_id": part_id,
            }
            for part_id in part_ids
        ]
        if target_entities is None
        else target_entities
    )
    if not isinstance(raw_entities, Sequence) or isinstance(
        raw_entities,
        (str, bytes),
    ):
        raise MultigroupExactMdlTournamentError(
            f"{label}.target_entities must be an array"
        )
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw_entity in enumerate(raw_entities):
        if not isinstance(raw_entity, Mapping):
            raise MultigroupExactMdlTournamentError(
                f"{label}.target_entities[{index}] must be an object"
            )
        part_id = raw_entity.get("part_id")
        subset_name = raw_entity.get("subset_name")
        raw_kind = raw_entity.get("entity_kind")
        if not isinstance(part_id, str) or not part_id or part_id not in assignments:
            raise MultigroupExactMdlTournamentError(
                f"{label}.target_entities[{index}] has an invalid part_id"
            )
        if subset_name is None:
            entity_kind = "assignment"
            entity: dict[str, str] = {
                "entity_kind": entity_kind,
                "part_id": part_id,
            }
        else:
            if not isinstance(subset_name, str) or not subset_name:
                raise MultigroupExactMdlTournamentError(
                    f"{label}.target_entities[{index}] has an invalid subset_name"
                )
            entity_kind = "face_subset"
            if subset_name not in _face_subsets(
                assignments[part_id],
                f"{label}/{part_id}",
            ):
                raise MultigroupExactMdlTournamentError(
                    f"{label}.target_entities[{index}] references absent face "
                    f"subset {part_id}:{subset_name}"
                )
            entity = {
                "entity_kind": entity_kind,
                "part_id": part_id,
                "subset_name": subset_name,
            }
        if raw_kind is not None and raw_kind != entity_kind:
            raise MultigroupExactMdlTournamentError(
                f"{label}.target_entities[{index}] has inconsistent entity_kind"
            )
        key = _entity_key(entity)
        if key in seen:
            raise MultigroupExactMdlTournamentError(
                f"{label}.target_entities contains duplicate {key}"
            )
        seen.add(key)
        normalized.append(entity)
    if {entity["part_id"] for entity in normalized} != set(part_ids):
        raise MultigroupExactMdlTournamentError(
            f"{label}.target_part_ids must exactly cover target_entities"
        )
    return sorted(normalized, key=_entity_key)


def _entity_material_id(
    assignments: Mapping[str, Mapping[str, Any]],
    entity: Mapping[str, Any],
) -> str:
    assignment = assignments[str(entity["part_id"])]
    subset_name = entity.get("subset_name")
    if subset_name is None:
        return str(assignment["material_id"])
    return str(
        _face_subsets(
            assignment,
            f"material entity {entity['part_id']}",
        )[str(subset_name)]["material_id"]
    )


def _require_entities_in_group(
    *,
    assignments: Mapping[str, Mapping[str, Any]],
    group_id: str,
    entities: Sequence[Mapping[str, Any]],
    label: str,
) -> None:
    for entity in entities:
        part_id = str(entity["part_id"])
        subset_name = entity.get("subset_name")
        provenance = assignments[part_id].get("provenance")
        if not isinstance(provenance, Mapping):
            raise MultigroupExactMdlTournamentError(
                f"{label} material entity {part_id} lacks group provenance"
            )
        if subset_name is None:
            declared_group_id = provenance.get("canonical_group_id")
        else:
            subset_groups = provenance.get("face_subset_canonical_group_ids")
            declared_group_id = (
                subset_groups.get(subset_name)
                if isinstance(subset_groups, Mapping)
                else None
            )
        if declared_group_id != group_id:
            entity_label = (
                part_id if subset_name is None else f"{part_id}:{subset_name}"
            )
            raise MultigroupExactMdlTournamentError(
                f"{label} material entity {entity_label} belongs to "
                f"{declared_group_id!r}, not {group_id!r}"
            )


def _set_entity_material_id(
    assignments: Mapping[str, dict[str, Any]],
    entity: Mapping[str, Any],
    material_id: str,
) -> None:
    assignment = assignments[str(entity["part_id"])]
    subset_name = entity.get("subset_name")
    if subset_name is None:
        assignment["material_id"] = material_id
        return
    raw_subsets = assignment.get("face_subsets")
    if not isinstance(raw_subsets, list):
        raise MultigroupExactMdlTournamentError(
            f"material entity {entity['part_id']} face_subsets must be an array"
        )
    mutable_subset = next(
        (
            subset
            for subset in raw_subsets
            if isinstance(subset, dict) and subset.get("subset_name") == subset_name
        ),
        None,
    )
    if mutable_subset is None:
        raise MultigroupExactMdlTournamentError(
            f"material entity {entity['part_id']}:{subset_name} is absent"
        )
    mutable_subset["material_id"] = material_id


def _require_library_defaults(
    assignment: Mapping[str, Any],
    label: str,
) -> None:
    parameters = assignment.get("parameters")
    if parameters is not None and (
        not isinstance(parameters, Mapping) or bool(parameters)
    ):
        raise MultigroupExactMdlTournamentError(
            f"{label} modifies selected MDL parameters"
        )
    raw_subsets = assignment.get("face_subsets", [])
    if not isinstance(raw_subsets, Sequence) or isinstance(raw_subsets, (str, bytes)):
        raise MultigroupExactMdlTournamentError(
            f"{label}.face_subsets must be an array"
        )
    for index, raw_subset in enumerate(raw_subsets):
        if not isinstance(raw_subset, Mapping):
            raise MultigroupExactMdlTournamentError(
                f"{label}.face_subsets[{index}] must be an object"
            )
        subset_parameters = raw_subset.get("parameters")
        if subset_parameters is not None and (
            not isinstance(subset_parameters, Mapping) or bool(subset_parameters)
        ):
            raise MultigroupExactMdlTournamentError(
                f"{label}.face_subsets[{index}] modifies MDL parameters"
            )


def _material_delta(
    baseline_plan: Mapping[str, Any],
    candidate_plan: Mapping[str, Any],
    *,
    label: str,
) -> list[dict[str, str]]:
    before = _assignments(baseline_plan, "group-round baseline")
    after = _assignments(candidate_plan, label)
    if set(before) != set(after):
        raise MultigroupExactMdlTournamentError(
            f"{label} does not exactly cover the baseline"
        )
    delta: list[dict[str, str]] = []
    for part_id in sorted(before):
        old_assignment = copy.deepcopy(before[part_id])
        new_assignment = copy.deepcopy(after[part_id])
        old_material_id = str(old_assignment.pop("material_id"))
        new_material_id = str(new_assignment.pop("material_id"))
        old_subsets = _face_subsets(
            {"face_subsets": old_assignment.pop("face_subsets", [])},
            f"group-round baseline/{part_id}",
        )
        new_subsets = _face_subsets(
            {"face_subsets": new_assignment.pop("face_subsets", [])},
            f"{label}/{part_id}",
        )
        if list(old_subsets) != list(new_subsets):
            raise MultigroupExactMdlTournamentError(
                f"{label}/{part_id} does not preserve face-subset order"
            )
        if old_assignment != new_assignment:
            raise MultigroupExactMdlTournamentError(
                f"{label}/{part_id} changes fields other than material bindings"
            )
        if old_material_id != new_material_id:
            delta.append(
                {
                    "part_id": part_id,
                    "old_material_id": old_material_id,
                    "new_material_id": new_material_id,
                }
            )
        for subset_name in sorted(old_subsets):
            old_subset = old_subsets[subset_name]
            new_subset = new_subsets[subset_name]
            old_subset_material_id = str(old_subset.pop("material_id"))
            new_subset_material_id = str(new_subset.pop("material_id"))
            if old_subset != new_subset:
                raise MultigroupExactMdlTournamentError(
                    f"{label}/{part_id}/{subset_name} changes fields other "
                    "than material_id"
                )
            if old_subset_material_id != new_subset_material_id:
                delta.append(
                    {
                        "part_id": part_id,
                        "subset_name": subset_name,
                        "old_material_id": old_subset_material_id,
                        "new_material_id": new_subset_material_id,
                    }
                )
    return delta


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MultigroupExactMdlTournamentError(f"{label} must be an array")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MultigroupExactMdlTournamentError(f"{label} must be an object")
    return value


def _candidate_evidence_value(
    candidate: Mapping[str, Any],
) -> tuple[float, int]:
    raw_score = candidate.get(
        "retrieval_score",
        candidate.get("score"),
    )
    score = (
        float(raw_score)
        if (
            not isinstance(raw_score, bool)
            and isinstance(raw_score, (int, float))
            and math.isfinite(float(raw_score))
        )
        else float("-inf")
    )
    raw_rank = candidate.get("retrieval_rank", candidate.get("rank"))
    rank = (
        int(raw_rank)
        if (
            not isinstance(raw_rank, bool)
            and isinstance(raw_rank, int)
            and raw_rank >= 1
        )
        else _MISSING_RANK
    )
    return score, rank


def _strict_nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MultigroupExactMdlTournamentError(
            f"{label} must be a non-negative integer"
        )
    return value


def _strict_ranked_retrieval_audit(
    raw_audit: Any,
    *,
    label: str,
    expected_strategy: str,
    allowed_material_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(raw_audit, Mapping):
        raise MultigroupExactMdlTournamentError(f"{label} must be an object")
    audit = dict(raw_audit)
    if audit.get("strategy") != expected_strategy:
        raise MultigroupExactMdlTournamentError(
            f"{label}.strategy must be {expected_strategy!r}"
        )
    if audit.get("fixed_library_defaults_required") is not True:
        raise MultigroupExactMdlTournamentError(
            f"{label} must require fixed NVIDIA library defaults"
        )
    pool_count = _strict_nonnegative_integer(
        audit.get("pool_count"),
        f"{label}.pool_count",
    )
    eligible_pool_count = _strict_nonnegative_integer(
        audit.get("eligible_pool_count"),
        f"{label}.eligible_pool_count",
    )
    if eligible_pool_count > pool_count:
        raise MultigroupExactMdlTournamentError(
            f"{label}.eligible_pool_count exceeds pool_count"
        )
    raw_limit = audit.get("limit")
    if isinstance(raw_limit, bool) or not isinstance(raw_limit, int) or raw_limit < 1:
        raise MultigroupExactMdlTournamentError(
            f"{label}.limit must be a positive integer"
        )
    ranking = _sequence(audit.get("ranking"), f"{label}.ranking")
    if not ranking:
        raise MultigroupExactMdlTournamentError(f"{label}.ranking must not be empty")
    if len(ranking) > raw_limit or len(ranking) > eligible_pool_count:
        raise MultigroupExactMdlTournamentError(
            f"{label}.ranking exceeds its declared bounds"
        )
    normalized: list[dict[str, Any]] = []
    seen_material_ids: set[str] = set()
    for expected_rank, raw_row in enumerate(ranking, start=1):
        row_label = f"{label}.ranking[{expected_rank - 1}]"
        if not isinstance(raw_row, Mapping):
            raise MultigroupExactMdlTournamentError(f"{row_label} must be an object")
        rank = raw_row.get("rank")
        material_id = raw_row.get("material_id")
        score = raw_row.get("score")
        matched_fields = raw_row.get("matched_fields")
        if rank != expected_rank or isinstance(rank, bool):
            raise MultigroupExactMdlTournamentError(
                f"{label}.ranking ranks must be contiguous from one"
            )
        if not isinstance(material_id, str) or not material_id:
            raise MultigroupExactMdlTournamentError(
                f"{row_label}.material_id must be a non-empty string"
            )
        if material_id in seen_material_ids:
            raise MultigroupExactMdlTournamentError(
                f"{label}.ranking contains duplicate material_id {material_id!r}"
            )
        if material_id not in allowed_material_ids:
            raise MultigroupExactMdlTournamentError(
                f"{label}.ranking references a material outside the catalog: "
                f"{material_id}"
            )
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise MultigroupExactMdlTournamentError(f"{row_label}.score must be finite")
        fields = _sequence(matched_fields, f"{row_label}.matched_fields")
        if (
            not fields
            or any(not isinstance(field, str) or not field for field in fields)
            or len(set(fields)) != len(fields)
        ):
            raise MultigroupExactMdlTournamentError(
                f"{row_label}.matched_fields must contain unique non-empty strings"
            )
        seen_material_ids.add(material_id)
        normalized.append(
            {
                "rank": expected_rank,
                "material_id": material_id,
                "score": float(score),
                "matched_fields": list(fields),
            }
        )
    if any(
        normalized[index]["score"] < normalized[index + 1]["score"]
        for index in range(len(normalized) - 1)
    ):
        raise MultigroupExactMdlTournamentError(
            f"{label}.ranking scores must be descending"
        )
    top_score = audit.get("top_score")
    if (
        isinstance(top_score, bool)
        or not isinstance(top_score, (int, float))
        or not math.isfinite(float(top_score))
        or not math.isclose(
            float(top_score),
            float(normalized[0]["score"]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        raise MultigroupExactMdlTournamentError(
            f"{label}.top_score does not match ranking[0]"
        )
    margin_available = audit.get("margin_available")
    if not isinstance(margin_available, bool):
        raise MultigroupExactMdlTournamentError(
            f"{label}.margin_available must be boolean"
        )
    runner_up = audit.get("runner_up_score")
    score_margin = audit.get("score_margin")
    normalized_margin = audit.get("normalized_margin")
    if margin_available:
        if len(normalized) < 2:
            raise MultigroupExactMdlTournamentError(
                f"{label} declares a margin without a runner-up"
            )
        numeric_margin_fields = (runner_up, score_margin, normalized_margin)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric_margin_fields
        ):
            raise MultigroupExactMdlTournamentError(
                f"{label} margin fields must be finite numbers"
            )
        expected_runner_up = float(normalized[1]["score"])
        expected_margin = float(normalized[0]["score"]) - expected_runner_up
        denominator = (
            max(abs(float(normalized[0]["score"])), 1e-12)
            if expected_strategy in _VISUAL_RETRIEVAL_STRATEGIES
            else max(abs(float(normalized[0]["score"])), 1.0)
        )
        expected_normalized_margin = expected_margin / denominator
        if (
            not math.isclose(
                float(runner_up),
                expected_runner_up,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(score_margin),
                expected_margin,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(normalized_margin),
                expected_normalized_margin,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise MultigroupExactMdlTournamentError(
                f"{label} margin summary is inconsistent"
            )
    elif len(normalized) != 1 or any(
        value is not None for value in (runner_up, score_margin, normalized_margin)
    ):
        raise MultigroupExactMdlTournamentError(
            f"{label} unavailable margin contract is inconsistent"
        )
    return audit, normalized


def _strict_visual_wrapper_audit(
    raw_audit: Any,
    *,
    label: str,
    expected_fallback_strategy: str,
    allowed_material_ids: set[str],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    strategy = raw_audit.get("strategy") if isinstance(raw_audit, Mapping) else None
    if strategy not in _VISUAL_RETRIEVAL_STRATEGIES:
        raise MultigroupExactMdlTournamentError(
            f"{label}.strategy is not a supported visual retrieval strategy"
        )
    audit, ranking = _strict_ranked_retrieval_audit(
        raw_audit,
        label=label,
        expected_strategy=str(strategy),
        allowed_material_ids=allowed_material_ids,
    )
    group_id = audit.get("group_id")
    if not isinstance(group_id, str) or not group_id:
        raise MultigroupExactMdlTournamentError(
            f"{label}.group_id must be a non-empty string"
        )
    if audit.get("full_catalog_indexed") is not True:
        raise MultigroupExactMdlTournamentError(
            f"{label} must index the full NVIDIA catalog"
        )
    if audit.get("final_authority") != _VISUAL_RETRIEVAL_FINAL_AUTHORITY:
        raise MultigroupExactMdlTournamentError(
            f"{label}.final_authority must be {_VISUAL_RETRIEVAL_FINAL_AUTHORITY!r}"
        )
    pool_count = int(audit["pool_count"])
    if audit.get("eligible_pool_count") != pool_count or pool_count != len(
        allowed_material_ids
    ):
        raise MultigroupExactMdlTournamentError(
            f"{label} must cover the complete effective NVIDIA catalog"
        )
    seen_siglip_ranks: set[int] = set()
    seen_dino_ranks: set[int] = set()
    seen_color_ranks: set[int] = set()
    seen_mvinverse_ranks: set[int] = set()
    raw_visual_ranking = _sequence(audit.get("ranking"), f"{label}.ranking")
    for index, (raw_row, normalized_row) in enumerate(
        zip(raw_visual_ranking, ranking, strict=True)
    ):
        assert isinstance(raw_row, Mapping)
        expected_fields = {
            "rank",
            "material_id",
            "score",
            "matched_fields",
            "siglip2_rank",
            "siglip2_score",
            "dino_rank",
            "dino_score",
        }
        if strategy == _BASE_BANK_RETRIEVAL_STRATEGY:
            expected_fields.update(
                {
                    "color_rank",
                    "color_score",
                    "mvinverse_rank",
                    "mvinverse_score",
                }
            )
        if set(raw_row) != expected_fields:
            raise MultigroupExactMdlTournamentError(
                f"{label}.ranking[{index}] visual fields are invalid"
            )
        siglip_rank = raw_row.get("siglip2_rank")
        siglip_score = raw_row.get("siglip2_score")
        dino_rank = raw_row.get("dino_rank")
        dino_score = raw_row.get("dino_score")
        if (
            isinstance(siglip_rank, bool)
            or not isinstance(siglip_rank, int)
            or not 1 <= siglip_rank <= pool_count
            or siglip_rank in seen_siglip_ranks
            or isinstance(siglip_score, bool)
            or not isinstance(siglip_score, (int, float))
            or not math.isfinite(float(siglip_score))
        ):
            raise MultigroupExactMdlTournamentError(
                f"{label}.ranking[{index}] has invalid SigLIP2 evidence"
            )
        seen_siglip_ranks.add(siglip_rank)
        if (dino_rank is None) != (dino_score is None):
            raise MultigroupExactMdlTournamentError(
                f"{label}.ranking[{index}] DINO rank/score must be supplied together"
            )
        expected_matches = (
            ["siglip2_base_bank_rig_visual"]
            if strategy == _BASE_BANK_RETRIEVAL_STRATEGY
            else ["siglip2_catalog_wide_visual"]
        )
        dino_rrf = 0.0
        if dino_rank is not None:
            if (
                isinstance(dino_rank, bool)
                or not isinstance(dino_rank, int)
                or not 1 <= dino_rank <= pool_count
                or dino_rank in seen_dino_ranks
                or isinstance(dino_score, bool)
                or not isinstance(dino_score, (int, float))
                or not math.isfinite(float(dino_score))
            ):
                raise MultigroupExactMdlTournamentError(
                    f"{label}.ranking[{index}] has invalid DINO evidence"
                )
            seen_dino_ranks.add(dino_rank)
            expected_matches.append(
                "dinov2_base_bank_surface_texture"
                if strategy == _BASE_BANK_RETRIEVAL_STRATEGY
                else "dinov2_masked_dense_texture"
            )
            dino_rrf = 1.2 / (_RRF_RANK_CONSTANT + dino_rank)
        color_rrf = 0.0
        mvinverse_rrf = 0.0
        if strategy == _BASE_BANK_RETRIEVAL_STRATEGY:
            color_rank = raw_row.get("color_rank")
            color_score = raw_row.get("color_score")
            if (
                isinstance(color_rank, bool)
                or not isinstance(color_rank, int)
                or not 1 <= color_rank <= pool_count
                or color_rank in seen_color_ranks
                or isinstance(color_score, bool)
                or not isinstance(color_score, (int, float))
                or not math.isfinite(float(color_score))
            ):
                raise MultigroupExactMdlTournamentError(
                    f"{label}.ranking[{index}] has invalid color evidence"
                )
            seen_color_ranks.add(color_rank)
            expected_matches.append("masked_color_appearance")
            color_rrf = 0.8 / (_RRF_RANK_CONSTANT + color_rank)
            mvinverse_rank = raw_row.get("mvinverse_rank")
            mvinverse_score = raw_row.get("mvinverse_score")
            if (mvinverse_rank is None) != (mvinverse_score is None):
                raise MultigroupExactMdlTournamentError(
                    f"{label}.ranking[{index}] MVInverse rank/score must be "
                    "supplied together"
                )
            if mvinverse_rank is not None:
                if (
                    isinstance(mvinverse_rank, bool)
                    or not isinstance(mvinverse_rank, int)
                    or not 1 <= mvinverse_rank <= pool_count
                    or mvinverse_rank in seen_mvinverse_ranks
                    or isinstance(mvinverse_score, bool)
                    or not isinstance(mvinverse_score, (int, float))
                    or not math.isfinite(float(mvinverse_score))
                ):
                    raise MultigroupExactMdlTournamentError(
                        f"{label}.ranking[{index}] has invalid MVInverse evidence"
                    )
                seen_mvinverse_ranks.add(mvinverse_rank)
                expected_matches.append("mvinverse_authored_pbr_prior")
                mvinverse_rrf = 0.2 / (
                    _RRF_RANK_CONSTANT + mvinverse_rank
                )
        expected_score = round(
            1.0 / (_RRF_RANK_CONSTANT + siglip_rank)
            + dino_rrf
            + color_rrf
            + mvinverse_rrf,
            10,
        )
        if normalized_row["matched_fields"] != expected_matches or not math.isclose(
            float(normalized_row["score"]),
            expected_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise MultigroupExactMdlTournamentError(
                f"{label}.ranking[{index}] visual RRF evidence is inconsistent"
            )
    fallback_audit, fallback_ranking = _strict_ranked_retrieval_audit(
        audit.get("fallback_audit"),
        label=f"{label}.fallback_audit",
        expected_strategy=expected_fallback_strategy,
        allowed_material_ids=allowed_material_ids,
    )
    if fallback_audit.get("pool_count") != pool_count:
        raise MultigroupExactMdlTournamentError(
            f"{label}.fallback_audit does not share the full catalog snapshot"
        )
    return group_id, ranking, fallback_ranking


def _strict_candidate_identities(
    candidates: Sequence[Any],
    *,
    label: str,
    allowed_material_ids: set[str],
) -> None:
    seen_material_ids: set[str] = set()
    for index, raw_candidate in enumerate(candidates):
        row_label = f"{label}[{index}]"
        if not isinstance(raw_candidate, Mapping):
            raise MultigroupExactMdlTournamentError(f"{row_label} must be an object")
        material_id = raw_candidate.get("material_id")
        if not isinstance(material_id, str) or not material_id:
            raise MultigroupExactMdlTournamentError(
                f"{row_label}.material_id must be a non-empty string"
            )
        if material_id in seen_material_ids:
            raise MultigroupExactMdlTournamentError(
                f"{label} contains duplicate material_id {material_id!r}"
            )
        if material_id not in allowed_material_ids:
            raise MultigroupExactMdlTournamentError(
                f"{label} references a material outside the catalog: {material_id}"
            )
        seen_material_ids.add(material_id)


def _visual_retrieval_fusion(
    *,
    candidate_document: Mapping[str, Any],
    primary: Sequence[Any],
    wider: Sequence[Any],
    allowed_material_ids: set[str],
) -> dict[str, Any] | None:
    selection_raw = candidate_document.get("retrieval_audit")
    tournament_raw = candidate_document.get("tournament_retrieval_audit")
    if selection_raw is None and tournament_raw is None:
        return None
    if not isinstance(selection_raw, Mapping) or not isinstance(
        tournament_raw,
        Mapping,
    ):
        raise MultigroupExactMdlTournamentError(
            "visual retrieval requires both retrieval_audit and "
            "tournament_retrieval_audit objects"
        )
    strategies = {
        selection_raw.get("strategy"),
        tournament_raw.get("strategy"),
    }
    if not strategies.intersection(_VISUAL_RETRIEVAL_STRATEGIES):
        # Retain compatibility with legacy, non-visual retrieval documents.
        return None
    if len(strategies) != 1 or not strategies.issubset(
        _VISUAL_RETRIEVAL_STRATEGIES
    ):
        raise MultigroupExactMdlTournamentError(
            "selection and tournament visual retrieval strategies disagree"
        )
    selection_group_id, selection_visual, selection_fallback = (
        _strict_visual_wrapper_audit(
            selection_raw,
            label="material candidate document.retrieval_audit",
            expected_fallback_strategy=_SELECTION_FALLBACK_STRATEGY,
            allowed_material_ids=allowed_material_ids,
        )
    )
    tournament_group_id, tournament_visual, tournament_fallback = (
        _strict_visual_wrapper_audit(
            tournament_raw,
            label="material candidate document.tournament_retrieval_audit",
            expected_fallback_strategy=_TOURNAMENT_FALLBACK_STRATEGY,
            allowed_material_ids=allowed_material_ids,
        )
    )
    if selection_group_id != tournament_group_id:
        raise MultigroupExactMdlTournamentError(
            "selection and tournament visual retrieval group_id values disagree"
        )
    selection_visual_ids = [str(row["material_id"]) for row in selection_visual]
    tournament_visual_ids = [str(row["material_id"]) for row in tournament_visual]
    if (
        len(selection_visual_ids) > len(tournament_visual_ids)
        or tournament_visual_ids[: len(selection_visual_ids)] != selection_visual_ids
    ):
        raise MultigroupExactMdlTournamentError(
            "selection visual ranking must be a prefix of the tournament visual ranking"
        )

    # Forward/reverse Qwen disagreement can deliberately add exact identities
    # to the persisted primary lists without rewriting the immutable retrieval
    # audit.  Validate those identities strictly, but keep the complete audit
    # rankings as the rank authority for RRF.
    _strict_candidate_identities(
        primary,
        label="material candidate document.candidates",
        allowed_material_ids=allowed_material_ids,
    )
    _strict_candidate_identities(
        wider,
        label="material candidate document.tournament_candidates",
        allowed_material_ids=allowed_material_ids,
    )

    fallback_by_material: dict[str, dict[str, Any]] = {}
    for source_name, rows in (
        ("selection_fallback", selection_fallback),
        ("tournament_fallback", tournament_fallback),
    ):
        for row in rows:
            material_id = str(row["material_id"])
            record = fallback_by_material.setdefault(
                material_id,
                {
                    "material_id": material_id,
                    "source_evidence": {},
                },
            )
            # Scores and matched fields describe different retrieval
            # objectives.  Preserve both without comparing their scales.
            record["source_evidence"][source_name] = copy.deepcopy(row)
    fused_fallback = sorted(
        fallback_by_material.values(),
        key=lambda record: (
            min(
                int(evidence["rank"]) for evidence in record["source_evidence"].values()
            ),
            0 if "selection_fallback" in record["source_evidence"] else 1,
            int(
                record["source_evidence"].get(
                    "selection_fallback", {"rank": _MISSING_RANK}
                )["rank"]
            ),
            int(
                record["source_evidence"].get(
                    "tournament_fallback", {"rank": _MISSING_RANK}
                )["rank"]
            ),
            str(record["material_id"]),
        ),
    )
    for fused_rank, record in enumerate(fused_fallback, start=1):
        record["rank"] = fused_rank
        record["best_source_rank"] = min(
            int(evidence["rank"]) for evidence in record["source_evidence"].values()
        )

    return {
        "group_id": selection_group_id,
        "selection_visual": selection_visual,
        "tournament_visual": tournament_visual,
        "selection_fallback": selection_fallback,
        "tournament_fallback": tournament_fallback,
        "fused_fallback": fused_fallback,
    }


def _visual_lane_rank(record: Mapping[str, Any], lane: str) -> int | None:
    if lane == _VISUAL_LANE:
        rank = record.get("visual_lane_rank")
    else:
        raw_evidence = record.get("fallback_source_evidence")
        evidence = (
            raw_evidence.get(lane) if isinstance(raw_evidence, Mapping) else None
        )
        rank = evidence.get("rank") if isinstance(evidence, Mapping) else None
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        return None
    return rank


def _reserved_visual_lane_quotas(
    *,
    slot_budget: int,
    lane_records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, int]:
    """Reserve bounded render slots for every independent evidence lane.

    Primary Qwen candidates already consume the strongest visual-RRF slots.
    The remaining budget therefore gives semantic and MVInverse fallbacks the
    first two minimum reservations.  Once every available lane has one slot,
    growth rotates through MVInverse, semantic, and visual evidence.  The
    policy is deterministic and depends only on lane availability, never on a
    material name, colour, family, or raw score shared across objectives.
    """

    quotas = {lane: 0 for lane in _LANE_MINIMUM_ORDER}
    if slot_budget <= 0:
        return quotas
    availability = {
        lane: len({str(record["material_id"]) for record in records})
        for lane, records in lane_records.items()
    }
    remaining = slot_budget
    for lane in _LANE_MINIMUM_ORDER:
        if remaining <= 0:
            break
        if availability.get(lane, 0) > 0:
            quotas[lane] = 1
            remaining -= 1
    while remaining > 0:
        progressed = False
        for lane in _LANE_GROWTH_ORDER:
            if remaining <= 0:
                break
            if quotas[lane] >= availability.get(lane, 0):
                continue
            quotas[lane] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    return quotas


def _rank_stratified_lane_order(
    records: Sequence[dict[str, Any]],
    *,
    reserved_count: int,
) -> list[dict[str, Any]]:
    """Put deterministic head/middle/lower-tail representatives first.

    MVInverse immutable-default rankings commonly contain dense preset series.
    Taking only the first adjacent rows can spend every render on nearly the
    same preset.  The reserved prefix spans the ranking through the beginning
    of its lower quartile; exact rendered QA remains the sole winner authority.
    Remaining rows retain their original rank order for quota refill.
    """

    ordered = list(records)
    if reserved_count <= 1 or len(ordered) <= 1:
        return ordered
    anchor_count = min(reserved_count, len(ordered))
    lower_quartile_size = max(1, math.ceil(len(ordered) / 4.0))
    tail_anchor = max(0, len(ordered) - lower_quartile_size)
    anchor_indices = [
        round(index * tail_anchor / max(1, anchor_count - 1))
        for index in range(anchor_count)
    ]
    unique_anchor_indices: list[int] = []
    for index in anchor_indices:
        if index not in unique_anchor_indices:
            unique_anchor_indices.append(index)
    if len(unique_anchor_indices) < anchor_count:
        for index in range(len(ordered)):
            if index not in unique_anchor_indices:
                unique_anchor_indices.append(index)
            if len(unique_anchor_indices) == anchor_count:
                break
    anchor_set = set(unique_anchor_indices)
    return [
        *[ordered[index] for index in unique_anchor_indices],
        *[
            record
            for index, record in enumerate(ordered)
            if index not in anchor_set
        ],
    ]


def _merged_ranked_candidates(
    *,
    candidate_document: Mapping[str, Any],
    source_material_ids: set[str],
    allowed_material_ids: set[str],
    allowed_families: set[str],
    maximum_candidates: int,
    visual_similarity_first: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary = _sequence(
        candidate_document.get("candidates"),
        "material candidate document.candidates",
    )
    wider_raw = candidate_document.get("tournament_candidates", [])
    wider = _sequence(
        wider_raw,
        "material candidate document.tournament_candidates",
    )
    visual_fusion = (
        _visual_retrieval_fusion(
            candidate_document=candidate_document,
            primary=primary,
            wider=wider,
            allowed_material_ids=allowed_material_ids,
        )
        if visual_similarity_first
        else None
    )
    if visual_fusion is not None:
        raw_group = candidate_document.get("group")
        if (
            isinstance(raw_group, Mapping)
            and isinstance(raw_group.get("group_id"), str)
            and raw_group["group_id"] != visual_fusion["group_id"]
        ):
            raise MultigroupExactMdlTournamentError(
                "visual retrieval group_id does not match candidate document group"
            )

    merged: dict[str, dict[str, Any]] = {}
    first_seen = 0
    invalid_count = 0
    incompatible_count = 0
    outside_catalog_count = 0
    for source_name, source_candidates in (
        ("primary", primary),
        ("tournament", wider),
    ):
        for raw_candidate in source_candidates:
            if not isinstance(raw_candidate, Mapping):
                invalid_count += 1
                continue
            material_id = raw_candidate.get("material_id")
            family = raw_candidate.get("family")
            if not isinstance(material_id, str) or not material_id:
                invalid_count += 1
                continue
            if material_id not in allowed_material_ids:
                outside_catalog_count += 1
                continue
            if not visual_similarity_first and family not in allowed_families:
                incompatible_count += 1
                continue
            score, rank = _candidate_evidence_value(raw_candidate)
            record = merged.get(material_id)
            if record is None:
                record = {
                    "material_id": material_id,
                    "family": family,
                    "sources": [],
                    "primary": False,
                    "best_retrieval_score": score,
                    "best_retrieval_rank": rank,
                    "first_seen": first_seen,
                }
                merged[material_id] = record
                first_seen += 1
            if source_name not in record["sources"]:
                record["sources"].append(source_name)
            record["primary"] = record["primary"] or source_name == "primary"
            record["best_retrieval_score"] = max(
                float(record["best_retrieval_score"]),
                score,
            )
            record["best_retrieval_rank"] = min(
                int(record["best_retrieval_rank"]),
                rank,
            )

    if visual_fusion is not None:
        visual_lane_rank: dict[str, int] = {}
        fallback_lane_rank: dict[str, int] = {}
        for row in visual_fusion["tournament_visual"]:
            material_id = str(row["material_id"])
            visual_lane_rank[material_id] = int(row["rank"])
            record = merged.setdefault(
                material_id,
                {
                    "material_id": material_id,
                    "family": None,
                    "sources": [],
                    "primary": False,
                    "best_retrieval_score": float(row["score"]),
                    "best_retrieval_rank": int(row["rank"]),
                    "first_seen": first_seen,
                },
            )
            if record["first_seen"] == first_seen:
                first_seen += 1
            if "tournament_visual_audit" not in record["sources"]:
                record["sources"].append("tournament_visual_audit")
            record["best_retrieval_score"] = max(
                float(record["best_retrieval_score"]),
                float(row["score"]),
            )
            record["best_retrieval_rank"] = min(
                int(record["best_retrieval_rank"]),
                int(row["rank"]),
            )
            record["visual_source_evidence"] = copy.deepcopy(row)
        for row in visual_fusion["fused_fallback"]:
            material_id = str(row["material_id"])
            fallback_lane_rank[material_id] = int(row["rank"])
            source_scores = [
                float(evidence["score"]) for evidence in row["source_evidence"].values()
            ]
            source_ranks = [
                int(evidence["rank"]) for evidence in row["source_evidence"].values()
            ]
            record = merged.setdefault(
                material_id,
                {
                    "material_id": material_id,
                    "family": None,
                    "sources": [],
                    "primary": False,
                    "best_retrieval_score": max(source_scores),
                    "best_retrieval_rank": min(source_ranks),
                    "first_seen": first_seen,
                },
            )
            if record["first_seen"] == first_seen:
                first_seen += 1
            for source_name in sorted(row["source_evidence"]):
                if source_name not in record["sources"]:
                    record["sources"].append(source_name)
            record["best_retrieval_score"] = max(
                float(record["best_retrieval_score"]),
                *source_scores,
            )
            record["best_retrieval_rank"] = min(
                int(record["best_retrieval_rank"]),
                *source_ranks,
            )
            record["fallback_source_evidence"] = copy.deepcopy(row["source_evidence"])
        for material_id, record in merged.items():
            visual_rank = visual_lane_rank.get(material_id)
            fallback_rank = fallback_lane_rank.get(material_id)
            record["visual_lane_rank"] = visual_rank
            record["fallback_lane_rank"] = fallback_rank
            record["two_lane_rrf_score"] = (
                1.0 / (_RRF_RANK_CONSTANT + visual_rank) if visual_rank else 0.0
            ) + (1.0 / (_RRF_RANK_CONSTANT + fallback_rank) if fallback_rank else 0.0)

    # A single current identity is the baseline itself and must not appear as
    # a no-op challenger.  For a mixed current group, however, consolidating
    # the group to either existing identity is a real rendered alternative.
    baseline_records = (
        [
            merged.pop(material_id)
            for material_id in sorted(source_material_ids)
            if material_id in merged
        ]
        if len(source_material_ids) == 1
        else []
    )
    if visual_fusion is None:
        primary_records = sorted(
            (record for record in merged.values() if record["primary"]),
            key=lambda record: (
                -float(record["best_retrieval_score"]),
                int(record["best_retrieval_rank"]),
                int(record["first_seen"]),
                str(record["material_id"]),
            ),
        )
        extended_records = sorted(
            (record for record in merged.values() if not record["primary"]),
            key=lambda record: (
                -float(record["best_retrieval_score"]),
                int(record["best_retrieval_rank"]),
                int(record["first_seen"]),
                str(record["material_id"]),
            ),
        )
    else:
        primary_order = {
            str(candidate["material_id"]): index
            for index, candidate in enumerate(primary)
            if isinstance(candidate, Mapping)
            and isinstance(candidate.get("material_id"), str)
        }
        primary_records = sorted(
            (record for record in merged.values() if record["primary"]),
            key=lambda record: (
                primary_order.get(str(record["material_id"]), _MISSING_RANK),
                str(record["material_id"]),
            ),
        )
        extended_records = sorted(
            (record for record in merged.values() if not record["primary"]),
            key=lambda record: (
                -float(record.get("two_lane_rrf_score", 0.0)),
                int(record.get("visual_lane_rank") or _MISSING_RANK),
                int(record.get("fallback_lane_rank") or _MISSING_RANK),
                int(record["first_seen"]),
                str(record["material_id"]),
            ),
        )
    required_count = 1 + len(primary_records)
    if required_count > maximum_candidates:
        raise MultigroupExactMdlTournamentError(
            "maximum_candidates cannot preserve baseline plus every primary "
            f"candidate: required={required_count}, configured={maximum_candidates}"
        )
    extension_budget = maximum_candidates - required_count
    lane_quotas = {lane: 0 for lane in _LANE_MINIMUM_ORDER}
    lane_selected_material_ids = {lane: [] for lane in _LANE_MINIMUM_ORDER}
    if visual_fusion is None:
        selected_extensions = extended_records[:extension_budget]
    else:
        lane_records = {
            lane: sorted(
                (
                    record
                    for record in extended_records
                    if _visual_lane_rank(record, lane) is not None
                ),
                key=lambda record: (
                    int(_visual_lane_rank(record, lane) or _MISSING_RANK),
                    int(record["first_seen"]),
                    str(record["material_id"]),
                ),
            )
            for lane in _LANE_MINIMUM_ORDER
        }
        lane_quotas = _reserved_visual_lane_quotas(
            slot_budget=extension_budget,
            lane_records=lane_records,
        )
        lane_orders = dict(lane_records)
        lane_orders[_TOURNAMENT_FALLBACK_LANE] = _rank_stratified_lane_order(
            lane_records[_TOURNAMENT_FALLBACK_LANE],
            reserved_count=lane_quotas[_TOURNAMENT_FALLBACK_LANE],
        )
        selected_extensions: list[dict[str, Any]] = []
        selected_extension_ids: set[str] = set()

        def selected_lane_count(lane: str) -> int:
            return sum(
                _visual_lane_rank(record, lane) is not None
                for record in selected_extensions
            )

        for lane in _LANE_MINIMUM_ORDER:
            for record in lane_orders[lane]:
                if selected_lane_count(lane) >= lane_quotas[lane]:
                    break
                material_id = str(record["material_id"])
                if material_id in selected_extension_ids:
                    continue
                if len(selected_extensions) >= extension_budget:
                    break
                selected_extensions.append(record)
                selected_extension_ids.add(material_id)
        for record in extended_records:
            if len(selected_extensions) >= extension_budget:
                break
            material_id = str(record["material_id"])
            if material_id in selected_extension_ids:
                continue
            selected_extensions.append(record)
            selected_extension_ids.add(material_id)
        lane_selected_material_ids = {
            lane: [
                str(record["material_id"])
                for record in selected_extensions
                if _visual_lane_rank(record, lane) is not None
            ]
            for lane in _LANE_MINIMUM_ORDER
        }
    selected = primary_records + selected_extensions
    baseline_sources = sorted(
        {str(source) for record in baseline_records for source in record["sources"]}
        or {"current_plan"}
    )
    baseline_primary = any(record["primary"] for record in baseline_records)
    baseline_scores = [
        float(record["best_retrieval_score"])
        for record in baseline_records
        if math.isfinite(float(record["best_retrieval_score"]))
    ]
    baseline_ranks = [
        int(record["best_retrieval_rank"])
        for record in baseline_records
        if int(record["best_retrieval_rank"]) < 2**31 - 1
    ]
    records = [
        {
            "material_id": (
                next(iter(source_material_ids))
                if len(source_material_ids) == 1
                else None
            ),
            "baseline_material_ids": sorted(source_material_ids),
            "family": None,
            "sources": baseline_sources,
            "primary": baseline_primary,
            "is_baseline": True,
            "best_retrieval_score": (max(baseline_scores) if baseline_scores else None),
            "best_retrieval_rank": (min(baseline_ranks) if baseline_ranks else None),
        },
        *[
            {
                **{key: value for key, value in record.items() if key != "first_seen"},
                "is_baseline": False,
            }
            for record in selected
        ],
    ]
    audit = {
        "primary_input_count": len(primary),
        "tournament_input_count": len(wider),
        "merged_unique_catalog_compatible_count": (len(merged) + len(baseline_records)),
        "primary_unique_nonbaseline_count": len(primary_records),
        "extended_unique_nonbaseline_count": len(extended_records),
        "invalid_candidate_count": invalid_count,
        "outside_catalog_candidate_count": outside_catalog_count,
        "semantic_incompatible_candidate_count": incompatible_count,
        "maximum_candidates": maximum_candidates,
        "baseline_always_included": True,
        "all_primary_candidates_preserved": True,
        "ordering": (
            "baseline_then_primary_then_reserved_visual_semantic_mvinverse_lanes/v2"
            if visual_fusion is not None
            else "baseline_then_primary_by_evidence_then_extended_by_evidence/v1"
        ),
        "visual_retrieval_fusion_applied": visual_fusion is not None,
        "visual_retrieval_group_id": (
            visual_fusion["group_id"] if visual_fusion is not None else None
        ),
        "visual_lane_input_count": (
            len(visual_fusion["tournament_visual"]) if visual_fusion is not None else 0
        ),
        "selection_fallback_input_count": (
            len(visual_fusion["selection_fallback"]) if visual_fusion is not None else 0
        ),
        "tournament_fallback_input_count": (
            len(visual_fusion["tournament_fallback"])
            if visual_fusion is not None
            else 0
        ),
        "fused_fallback_unique_count": (
            len(visual_fusion["fused_fallback"]) if visual_fusion is not None else 0
        ),
        "rrf_rank_constant": (
            _RRF_RANK_CONSTANT if visual_fusion is not None else None
        ),
        "raw_scores_compared_across_lanes": False,
        "extension_slot_budget": extension_budget,
        "reserved_lane_quota_policy": (
            "semantic_then_mvinverse_then_visual_minimums_growth_mvinverse_semantic_visual/v1"
            if visual_fusion is not None
            else None
        ),
        "reserved_lane_quotas": lane_quotas,
        "reserved_lane_selected_material_ids": lane_selected_material_ids,
        "reserved_lane_selected_counts": {
            lane: len(material_ids)
            for lane, material_ids in lane_selected_material_ids.items()
        },
        "reserved_lane_quota_satisfied": {
            lane: len(lane_selected_material_ids[lane]) >= quota
            for lane, quota in lane_quotas.items()
        },
        "tournament_fallback_diversity_policy": (
            "rank_stratified_head_middle_lower_quartile/v1"
            if visual_fusion is not None
            else None
        ),
        "selected_candidates": records,
    }
    return records, audit


def build_exact_mdl_group_candidate_plans(
    *,
    source_plan: Mapping[str, Any],
    group_id: str,
    target_part_ids: Sequence[str],
    target_entities: Sequence[Mapping[str, Any]] | None = None,
    candidate_document: Mapping[str, Any],
    allowed_material_ids: set[str],
    maximum_candidates: int,
    selection_objective: str = SELECTION_OBJECTIVE_VISUAL,
    allowed_families: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one hash-linked group round without dropping primary candidates."""

    if not isinstance(group_id, str) or not group_id:
        raise MultigroupExactMdlTournamentError("group_id must be a non-empty string")
    if (
        isinstance(maximum_candidates, bool)
        or not isinstance(maximum_candidates, int)
        or maximum_candidates < 2
    ):
        raise MultigroupExactMdlTournamentError(
            "maximum_candidates must be at least two"
        )
    if selection_objective not in {
        SELECTION_OBJECTIVE_SEMANTIC,
        SELECTION_OBJECTIVE_VISUAL,
    }:
        raise MultigroupExactMdlTournamentError(
            f"unsupported selection objective: {selection_objective}"
        )
    part_ids = _sorted_unique_texts(
        list(target_part_ids),
        "target_part_ids",
        require_sorted=True,
    )
    if not part_ids:
        raise MultigroupExactMdlTournamentError("target_part_ids must not be empty")
    assignments = _assignments(source_plan, "group candidate source plan")
    entities = _normalized_target_entities(
        target_part_ids=part_ids,
        target_entities=target_entities,
        assignments=assignments,
        label=f"group {group_id}",
    )
    _require_entities_in_group(
        assignments=assignments,
        group_id=group_id,
        entities=entities,
        label="group candidate planner",
    )
    source_material_ids = {
        _entity_material_id(assignments, entity) for entity in entities
    }
    if not source_material_ids.issubset(allowed_material_ids):
        raise MultigroupExactMdlTournamentError(
            f"group {group_id} baseline material is outside the catalog"
        )
    effective_allowed_families = allowed_families or set()
    candidate_records, merge_audit = _merged_ranked_candidates(
        candidate_document=candidate_document,
        source_material_ids=source_material_ids,
        allowed_material_ids=allowed_material_ids,
        allowed_families=effective_allowed_families,
        maximum_candidates=maximum_candidates,
        visual_similarity_first=(selection_objective == SELECTION_OBJECTIVE_VISUAL),
    )
    if len(candidate_records) < 2:
        raise MultigroupExactMdlTournamentError(
            f"group {group_id} has fewer than two exact MDL identities"
        )

    source_plan_sha256 = _canonical_sha256(source_plan)
    planned: list[dict[str, Any]] = []
    for index, evidence in enumerate(candidate_records, start=1):
        raw_material_id = evidence["material_id"]
        material_id = str(raw_material_id) if isinstance(raw_material_id, str) else None
        identity_key = (
            material_id
            if material_id is not None
            else "|".join(sorted(source_material_ids))
        )
        candidate_id = (
            f"{group_id.casefold()}_{index:02d}_"
            f"{hashlib.sha256(identity_key.encode('utf-8')).hexdigest()[:10]}"
        )
        if evidence["is_baseline"]:
            plan = copy.deepcopy(dict(source_plan))
            provenance = plan.setdefault("provenance", {})
            if not isinstance(provenance, dict):
                raise MultigroupExactMdlTournamentError(
                    "candidate source plan provenance is invalid"
                )
            provenance["immutable_mdl_after_selection"] = True
            provenance["exact_mdl_candidate"] = {
                "candidate_id": candidate_id,
                "source_plan_sha256": source_plan_sha256,
                "source_material_ids": sorted(source_material_ids),
                "candidate_material_id": material_id,
                "changed_part_ids": [],
                "changed_entities": [],
                "target_entities": copy.deepcopy(entities),
                "parameters_locked_to_library_defaults": True,
            }
        else:
            if material_id is None:
                raise AssertionError("nonbaseline candidate has no material_id")
            plan = copy.deepcopy(dict(source_plan))
            plan_assignments = _assignments(
                plan,
                f"group {group_id} candidate plan",
            )
            changed_entities: list[dict[str, str]] = []
            for entity in entities:
                if _entity_material_id(plan_assignments, entity) != material_id:
                    changed_entities.append(copy.deepcopy(entity))
                _set_entity_material_id(
                    plan_assignments,
                    entity,
                    material_id,
                )
            plan["assignments"] = [
                plan_assignments[str(assignment["part_id"])]
                for assignment in plan["assignments"]
            ]
            provenance = plan.setdefault("provenance", {})
            if not isinstance(provenance, dict):
                raise MultigroupExactMdlTournamentError(
                    "candidate source plan provenance is invalid"
                )
            provenance["immutable_mdl_after_selection"] = True
            provenance["exact_mdl_candidate"] = {
                "candidate_id": candidate_id,
                "source_plan_sha256": source_plan_sha256,
                "source_material_ids": sorted(source_material_ids),
                "candidate_material_id": material_id,
                "changed_part_ids": sorted(
                    {entity["part_id"] for entity in changed_entities}
                ),
                "changed_entities": changed_entities,
                "target_entities": copy.deepcopy(entities),
                "target_part_ids": part_ids,
                "parameters_locked_to_library_defaults": True,
            }
            _assignments(plan, f"group {group_id} exact MDL candidate plan")
        planned.append(
            {
                "candidate_id": candidate_id,
                "group_id": group_id,
                "material_id": material_id,
                "is_baseline": bool(evidence["is_baseline"]),
                "candidate_evidence": copy.deepcopy(evidence),
                "plan": plan,
            }
        )

    audit = {
        "schema_version": PLANNING_SCHEMA_VERSION,
        "status": "PLANNED",
        "group_id": group_id,
        "target_part_ids": part_ids,
        "target_part_count": len(part_ids),
        "target_entities": copy.deepcopy(entities),
        "target_entity_count": len(entities),
        "target_face_subset_count": sum(
            entity["entity_kind"] == "face_subset" for entity in entities
        ),
        "source_plan_sha256": source_plan_sha256,
        "source_material_id": (
            next(iter(source_material_ids)) if len(source_material_ids) == 1 else None
        ),
        "source_material_ids": sorted(source_material_ids),
        "selection_objective": selection_objective,
        "candidate_count": len(planned),
        "candidate_material_ids": [str(record["material_id"]) for record in planned],
        "baseline_candidate_id": planned[0]["candidate_id"],
        "baseline_candidate_included": True,
        "primary_and_tournament_merge": merge_audit,
        "one_group_at_a_time": True,
        "parameters_locked_to_library_defaults": True,
    }
    return planned, audit


def _reference_footprint(group: Mapping[str, Any]) -> float:
    footprint = 0.0
    raw_sources = group.get("sources", [])
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
        return footprint
    for source in raw_sources:
        if not isinstance(source, Mapping):
            continue
        confidence = source.get("confidence")
        raw_boxes = source.get("boxes", [])
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not isinstance(raw_boxes, Sequence)
            or isinstance(raw_boxes, (str, bytes))
        ):
            continue
        area = 0.0
        for box in raw_boxes:
            if (
                isinstance(box, Sequence)
                and not isinstance(box, (str, bytes))
                and len(box) == 4
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in box
                )
            ):
                left, top, right, bottom = map(float, box)
                area += max(0.0, right - left) * max(0.0, bottom - top) / 1_000_000.0
        footprint += min(1.0, area) * max(0.0, min(1.0, float(confidence)))
    return footprint


def _baseline_group_presence_evidence(
    *,
    group_id: str,
    source_view_ids: Sequence[str],
    palette_fusion: Mapping[str, Any],
    quality_report: Mapping[str, Any] | None,
    minimum_source_view_count: int = 2,
) -> dict[str, Any] | None:
    """Return trusted per-view evidence that the baseline already delivers a group.

    Whole-asset quality reports name palette groups in each view's local
    namespace.  The canonical group may therefore be inspected only through
    ``palette_fusion.view_group_id_maps``.  Material names, model semantics, and
    provisional localization are intentionally outside this contract.
    """

    if (
        isinstance(minimum_source_view_count, bool)
        or not isinstance(minimum_source_view_count, int)
        or minimum_source_view_count < 1
    ):
        return None
    expected_view_ids = sorted(set(source_view_ids))
    if (
        len(expected_view_ids) < minimum_source_view_count
        or not isinstance(quality_report, Mapping)
        or quality_report.get("schema_version") != "qwen-reference-render-comparison/v1"
    ):
        return None
    quality_inputs = quality_report.get("inputs")
    comparison_scope = (
        quality_inputs.get("comparison_scope")
        if isinstance(quality_inputs, Mapping)
        else None
    )
    if (
        not isinstance(comparison_scope, Mapping)
        or comparison_scope.get("mode") != "whole_asset"
    ):
        return None
    aggregate = quality_report.get("aggregate")
    if (
        not isinstance(aggregate, Mapping)
        or aggregate.get("reference_view_coverage_status") != "PASS"
    ):
        return None

    raw_view_maps = palette_fusion.get("view_group_id_maps")
    raw_quality_views = quality_report.get("views")
    if (
        not isinstance(raw_view_maps, Mapping)
        or not isinstance(raw_quality_views, Sequence)
        or isinstance(raw_quality_views, (str, bytes))
    ):
        return None
    quality_views: dict[str, Mapping[str, Any]] = {}
    for raw_view in raw_quality_views:
        if not isinstance(raw_view, Mapping):
            return None
        view_id = raw_view.get("reference_view_id")
        if not isinstance(view_id, str) or not view_id or view_id in quality_views:
            return None
        quality_views[view_id] = raw_view

    raw_reference_view_count = aggregate.get("reference_view_count")
    if (
        isinstance(raw_reference_view_count, bool)
        or not isinstance(raw_reference_view_count, int)
        or raw_reference_view_count != len(quality_views)
    ):
        return None

    evidence_views: list[dict[str, Any]] = []
    for view_id in expected_view_ids:
        quality_view = quality_views.get(view_id)
        view_map = raw_view_maps.get(view_id)
        if not isinstance(quality_view, Mapping) or not isinstance(view_map, Mapping):
            return None
        reference = quality_view.get("reference")
        trusted_evidence = (
            reference.get("trusted_evidence")
            if isinstance(reference, Mapping)
            else None
        )
        if (
            not isinstance(trusted_evidence, Mapping)
            or trusted_evidence.get("usable") is not True
            or trusted_evidence.get("reasons") != []
        ):
            return None

        local_group_ids = sorted(
            local_group_id
            for local_group_id, canonical_group_id in view_map.items()
            if (
                isinstance(local_group_id, str)
                and local_group_id
                and canonical_group_id == group_id
            )
        )
        if not local_group_ids:
            return None
        material_color = quality_view.get("material_color")
        group_recall = (
            material_color.get("trusted_evidence_group_recall")
            if isinstance(material_color, Mapping)
            else None
        )
        raw_groups = (
            group_recall.get("groups") if isinstance(group_recall, Mapping) else None
        )
        if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, (str, bytes)):
            return None
        group_rows: dict[str, Mapping[str, Any]] = {}
        for raw_group in raw_groups:
            if not isinstance(raw_group, Mapping):
                return None
            local_group_id = raw_group.get("group_id")
            if (
                not isinstance(local_group_id, str)
                or not local_group_id
                or local_group_id in group_rows
            ):
                return None
            group_rows[local_group_id] = raw_group
        raw_group_count = group_recall.get("group_count")
        if (
            isinstance(raw_group_count, bool)
            or not isinstance(raw_group_count, int)
            or raw_group_count != len(group_rows)
        ):
            return None

        for local_group_id in local_group_ids:
            row = group_rows.get(local_group_id)
            recall = row.get("recall") if isinstance(row, Mapping) else None
            if (
                not isinstance(row, Mapping)
                or row.get("delivery_presence_status") != "PRESENT"
                or isinstance(recall, bool)
                or not isinstance(recall, (int, float))
                or not math.isfinite(float(recall))
                or float(recall) != 1.0
            ):
                return None
        evidence_views.append(
            {
                "reference_view_id": view_id,
                "local_group_ids": local_group_ids,
                "delivery_presence_status": "PRESENT",
                "recall": 1.0,
                "trusted_reference_evidence": True,
            }
        )

    try:
        quality_report_sha256 = _canonical_sha256(quality_report)
    except MultigroupExactMdlTournamentError:
        return None
    return {
        "canonical_group_id": group_id,
        "reference_view_ids": expected_view_ids,
        "reference_view_count": len(expected_view_ids),
        "minimum_source_view_count": minimum_source_view_count,
        "quality_report_sha256": quality_report_sha256,
        "view_group_mapping_source": "palette_fusion.view_group_id_maps",
        "views": evidence_views,
        "all_source_views_present": True,
    }


def _trusted_scoring_reference_scope(
    *,
    group_id: str,
    fusion_group: Mapping[str, Any],
    palette_fusion: Mapping[str, Any],
    quality_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Seal the local tournament to sources already trusted by whole-asset QA.

    Palette fusion may retain a legitimate, multi-view accent whose footprint in
    one source view is below the deterministic comparison floor.  Such a source
    is useful semantic corroboration, but it cannot safely participate in an
    exact-MDL render score.  Intersecting the canonical sources with the
    hash-bound whole-asset ``trusted_evidence.samples`` keeps the two contracts
    consistent without lowering the pixel trust boundary.
    """

    raw_sources = fusion_group.get("sources")
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
        raise MultigroupExactMdlTournamentError(
            f"palette fusion group {group_id} sources must be an array"
        )
    local_group_ids_by_view: dict[str, str] = {}
    for index, source in enumerate(raw_sources):
        if not isinstance(source, Mapping):
            raise MultigroupExactMdlTournamentError(
                f"palette fusion group {group_id} source {index} is invalid"
            )
        view_id = source.get("view_id")
        local_group_id = source.get("local_group_id")
        if (
            not isinstance(view_id, str)
            or not view_id
            or not isinstance(local_group_id, str)
            or not local_group_id
        ):
            raise MultigroupExactMdlTournamentError(
                f"palette fusion group {group_id} source {index} lacks view/local IDs"
            )
        previous = local_group_ids_by_view.get(view_id)
        if previous is not None and previous != local_group_id:
            raise MultigroupExactMdlTournamentError(
                f"palette fusion group {group_id} repeats view {view_id}"
            )
        local_group_ids_by_view[view_id] = local_group_id

    canonical_reference_view_ids = sorted(local_group_ids_by_view)
    if quality_report is None:
        return {
            "schema_version": "qwen-trusted-scoring-reference-scope/v1",
            "canonical_group_id": group_id,
            "selection_mode": "CANONICAL_SOURCES_NO_QUALITY_REPORT",
            "canonical_reference_view_ids": canonical_reference_view_ids,
            "reference_view_ids": canonical_reference_view_ids,
            "excluded_reference_sources": [],
            "quality_report_sha256": None,
            "minimum_trusted_evidence_pixels": MIN_TRUSTED_EVIDENCE_PIXELS,
        }
    if (
        not isinstance(quality_report, Mapping)
        or quality_report.get("schema_version")
        != "qwen-reference-render-comparison/v1"
    ):
        raise MultigroupExactMdlTournamentError(
            "trusted scoring reference scope requires a valid quality report"
        )
    quality_inputs = quality_report.get("inputs")
    comparison_scope = (
        quality_inputs.get("comparison_scope")
        if isinstance(quality_inputs, Mapping)
        else None
    )
    aggregate = quality_report.get("aggregate")
    if (
        comparison_scope != {"mode": "whole_asset"}
        or not isinstance(aggregate, Mapping)
        or aggregate.get("reference_view_coverage_status") != "PASS"
    ):
        raise MultigroupExactMdlTournamentError(
            "trusted scoring reference scope requires complete whole-asset QA"
        )
    raw_quality_views = quality_report.get("views")
    if not isinstance(raw_quality_views, Sequence) or isinstance(
        raw_quality_views, (str, bytes)
    ):
        raise MultigroupExactMdlTournamentError(
            "trusted scoring reference scope quality views are invalid"
        )
    quality_views: dict[str, Mapping[str, Any]] = {}
    for raw_view in raw_quality_views:
        if not isinstance(raw_view, Mapping):
            raise MultigroupExactMdlTournamentError(
                "trusted scoring reference scope contains an invalid quality view"
            )
        view_id = raw_view.get("reference_view_id")
        if (
            not isinstance(view_id, str)
            or not view_id
            or view_id in quality_views
        ):
            raise MultigroupExactMdlTournamentError(
                "trusted scoring reference scope quality view IDs are invalid"
            )
        quality_views[view_id] = raw_view

    raw_view_maps = palette_fusion.get("view_group_id_maps")
    if not isinstance(raw_view_maps, Mapping):
        raise MultigroupExactMdlTournamentError(
            "palette fusion lacks view_group_id_maps for trusted scoring scope"
        )
    selected: list[str] = []
    excluded: list[dict[str, Any]] = []
    for view_id in canonical_reference_view_ids:
        local_group_id = local_group_ids_by_view[view_id]
        view_map = raw_view_maps.get(view_id)
        if (
            not isinstance(view_map, Mapping)
            or view_map.get(local_group_id) != group_id
        ):
            raise MultigroupExactMdlTournamentError(
                "palette fusion source/map mismatch for trusted scoring scope: "
                f"{view_id}:{local_group_id}->{group_id}"
            )
        quality_view = quality_views.get(view_id)
        if not isinstance(quality_view, Mapping):
            raise MultigroupExactMdlTournamentError(
                f"quality report omits canonical source view {view_id}"
            )
        reference = quality_view.get("reference")
        trusted = (
            reference.get("trusted_evidence")
            if isinstance(reference, Mapping)
            else None
        )
        samples = trusted.get("samples") if isinstance(trusted, Mapping) else None
        matching_samples = [
            sample
            for sample in samples or []
            if (
                isinstance(sample, Mapping)
                and sample.get("group_id") == local_group_id
                and isinstance(sample.get("weight_pixels"), int)
                and not isinstance(sample.get("weight_pixels"), bool)
                and int(sample["weight_pixels"]) >= MIN_TRUSTED_EVIDENCE_PIXELS
            )
        ]
        if (
            isinstance(trusted, Mapping)
            and trusted.get("usable") is True
            and trusted.get("reasons") == []
            and matching_samples
        ):
            selected.append(view_id)
            continue
        excluded.append(
            {
                "reference_view_id": view_id,
                "local_group_id": local_group_id,
                "reason": "LOCAL_GROUP_ABSENT_FROM_TRUSTED_REFERENCE_EVIDENCE",
                "trusted_evidence_audit": (
                    trusted.get("audit") if isinstance(trusted, Mapping) else None
                ),
                "trusted_evidence_audit_sha256": (
                    trusted.get("audit_sha256")
                    if isinstance(trusted, Mapping)
                    else None
                ),
                "matching_trusted_sample_count": len(matching_samples),
            }
        )

    return {
        "schema_version": "qwen-trusted-scoring-reference-scope/v1",
        "canonical_group_id": group_id,
        "selection_mode": "WHOLE_ASSET_TRUSTED_EVIDENCE_INTERSECTION",
        "canonical_reference_view_ids": canonical_reference_view_ids,
        "reference_view_ids": selected,
        "excluded_reference_sources": excluded,
        "quality_report_sha256": _canonical_sha256(quality_report),
        "minimum_trusted_evidence_pixels": MIN_TRUSTED_EVIDENCE_PIXELS,
    }


def _validated_source_appearance_cohort_contracts(
    *,
    source_plan: Mapping[str, Any],
    assignments: Mapping[str, Mapping[str, Any]],
    visual_group_annotation_audit: Mapping[str, Any] | None,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Validate the hash-linked cohort handoff before targets are queued."""

    disabled = {
        "enabled": False,
        "contract_count": 0,
        "expected_member_count": 0,
        "expected_member_part_ids_by_group": {},
        "contract_ids_by_group": {},
        "annotation_audit_verified": False,
        "exact_cover": True,
    }
    if visual_group_annotation_audit is None:
        return {}, disabled
    if visual_group_annotation_audit.get("schema_version") != (
        VISUAL_GROUP_ANNOTATION_AUDIT_SCHEMA_VERSION
    ):
        raise MultigroupExactMdlTournamentError(
            "visual-group annotation audit has an unsupported schema_version"
        )
    if visual_group_annotation_audit.get("annotated_plan_sha256") != (
        _canonical_sha256(source_plan)
    ):
        raise MultigroupExactMdlTournamentError(
            "visual-group annotation audit does not hash-bind the queue source plan"
        )
    raw_cohort_audit = visual_group_annotation_audit.get(
        "source_appearance_cohort_propagation"
    )
    if not isinstance(raw_cohort_audit, Mapping) or raw_cohort_audit.get(
        "schema_version"
    ) != SOURCE_APPEARANCE_COHORT_SCHEMA_VERSION:
        raise MultigroupExactMdlTournamentError(
            "visual-group annotation audit lacks a valid source-appearance "
            "cohort contract"
        )
    plan_provenance = source_plan.get("provenance")
    plan_annotation = (
        plan_provenance.get("visual_group_annotation")
        if isinstance(plan_provenance, Mapping)
        else None
    )
    plan_cohort_audit = (
        plan_annotation.get("source_appearance_cohort_propagation")
        if isinstance(plan_annotation, Mapping)
        else None
    )
    if not isinstance(plan_cohort_audit, Mapping) or _canonical_sha256(
        plan_cohort_audit
    ) != _canonical_sha256(raw_cohort_audit):
        raise MultigroupExactMdlTournamentError(
            "source-appearance cohort audit and annotated plan provenance disagree"
        )
    enabled = raw_cohort_audit.get("enabled")
    if not isinstance(enabled, bool):
        raise MultigroupExactMdlTournamentError(
            "source-appearance cohort enabled flag must be boolean"
        )
    if raw_cohort_audit.get("method") != SOURCE_APPEARANCE_COHORT_METHOD:
        raise MultigroupExactMdlTournamentError(
            "source-appearance cohort audit has an invalid method"
        )
    raw_contracts = _sequence(
        raw_cohort_audit.get("contracts"),
        "source-appearance cohort contracts",
    )
    raw_blockers = _sequence(
        raw_cohort_audit.get("coverage_blockers"),
        "source-appearance cohort coverage blockers",
    )
    if raw_cohort_audit.get("exact_cover") is not True or raw_blockers:
        raise MultigroupExactMdlTournamentError(
            "source-appearance cohort annotation did not establish exact coverage"
        )
    for field in (
        "registry_sha256",
        "spatial_report_sha256",
        "annotation_input_plan_sha256",
    ):
        if raw_cohort_audit.get("enabled") is True and not _is_sha256(
            raw_cohort_audit.get(field)
        ):
            raise MultigroupExactMdlTournamentError(
                f"source-appearance cohort audit has an invalid {field}"
            )
    if not enabled and raw_contracts:
        raise MultigroupExactMdlTournamentError(
            "disabled source-appearance cohort audit contains contracts"
        )

    expected_by_group: dict[str, set[str]] = {}
    contract_ids_by_group: dict[str, list[str]] = {}
    seen_cohort_ids: set[str] = set()
    claimed_part_ids: set[str] = set()
    claimed_propagated_part_ids: set[str] = set()
    for index, raw_contract in enumerate(raw_contracts):
        contract = _mapping(
            raw_contract,
            f"source-appearance cohort contracts[{index}]",
        )
        if contract.get("schema_version") != (
            SOURCE_APPEARANCE_COHORT_CONTRACT_SCHEMA_VERSION
        ) or contract.get("method") != SOURCE_APPEARANCE_COHORT_METHOD:
            raise MultigroupExactMdlTournamentError(
                f"source-appearance cohort contract[{index}] has an invalid schema"
            )
        candidate_kind = contract.get("candidate_kind")
        expected_signature_kind = {
            "dominant_assembly": "source_appearance_plus_subset_layout",
            "rare_source_appearance_pair": (
                "geometry_plus_appearance_plus_subset_layout"
            ),
            "rare_source_appearance_layout_pair": (
                "source_appearance_plus_subset_layout"
            ),
        }.get(candidate_kind)
        if (
            expected_signature_kind is None
            or contract.get("cohort_signature_kind")
            != expected_signature_kind
        ):
            raise MultigroupExactMdlTournamentError(
                f"source-appearance cohort contract[{index}] has an invalid "
                "lane signature"
            )
        contract_sha256 = contract.get("contract_sha256")
        unsigned_contract = copy.deepcopy(dict(contract))
        unsigned_contract.pop("contract_sha256", None)
        if (
            not _is_sha256(contract_sha256)
            or _canonical_sha256(unsigned_contract) != contract_sha256
        ):
            raise MultigroupExactMdlTournamentError(
                f"source-appearance cohort contract[{index}] failed SHA256 validation"
            )
        cohort_id = contract.get("cohort_id")
        group_id = contract.get("canonical_group_id")
        if (
            not _is_sha256(cohort_id)
            or cohort_id in seen_cohort_ids
            or not isinstance(group_id, str)
            or not group_id
        ):
            raise MultigroupExactMdlTournamentError(
                f"source-appearance cohort contract[{index}] has invalid identity"
            )
        seen_cohort_ids.add(cohort_id)
        raw_expected = _sequence(
            contract.get("expected_member_part_ids"),
            f"source-appearance cohort contract[{index}].expected_member_part_ids",
        )
        expected_part_ids = [
            str(part_id)
            for part_id in raw_expected
            if isinstance(part_id, str) and part_id
        ]
        if (
            len(expected_part_ids) != len(raw_expected)
            or expected_part_ids != sorted(set(expected_part_ids))
            or not expected_part_ids
        ):
            raise MultigroupExactMdlTournamentError(
                f"source-appearance cohort contract[{index}] member IDs are invalid"
            )
        overlap = claimed_part_ids.intersection(expected_part_ids)
        if overlap:
            raise MultigroupExactMdlTournamentError(
                "source-appearance cohort contracts overlap parts: "
                + ", ".join(sorted(overlap))
            )
        claimed_part_ids.update(expected_part_ids)
        raw_anchors = _sequence(
            contract.get("anchor_part_ids"),
            f"source-appearance cohort contract[{index}].anchor_part_ids",
        )
        anchor_part_ids = [
            str(part_id)
            for part_id in raw_anchors
            if isinstance(part_id, str) and part_id
        ]
        if (
            anchor_part_ids != sorted(set(anchor_part_ids))
            or not anchor_part_ids
            or not set(anchor_part_ids) <= set(expected_part_ids)
        ):
            raise MultigroupExactMdlTournamentError(
                f"source-appearance cohort contract[{index}] anchors are invalid"
            )
        raw_propagated = _sequence(
            contract.get("propagated_member_part_ids"),
            f"source-appearance cohort contract[{index}].propagated_member_part_ids",
        )
        propagated_part_ids = [
            str(part_id)
            for part_id in raw_propagated
            if isinstance(part_id, str) and part_id
        ]
        if (
            propagated_part_ids != sorted(set(propagated_part_ids))
            or not set(propagated_part_ids) <= set(expected_part_ids)
            or contract.get("exact_cover") is not True
            or any(
                not _is_sha256(contract.get(field))
                for field in (
                    "registry_sha256",
                    "spatial_report_sha256",
                    "annotation_input_plan_sha256",
                )
            )
            or any(
                contract.get(field) != raw_cohort_audit.get(field)
                for field in (
                    "registry_sha256",
                    "spatial_report_sha256",
                    "annotation_input_plan_sha256",
                )
            )
        ):
            raise MultigroupExactMdlTournamentError(
                f"source-appearance cohort contract[{index}] evidence is invalid"
            )
        for part_id in expected_part_ids:
            assignment = assignments.get(part_id)
            if not isinstance(assignment, Mapping):
                raise MultigroupExactMdlTournamentError(
                    f"source-appearance cohort member {part_id} is absent from plan"
                )
            provenance = assignment.get("provenance")
            lineage = (
                provenance.get("source_appearance_cohort")
                if isinstance(provenance, Mapping)
                else None
            )
            if (
                not isinstance(provenance, Mapping)
                or provenance.get("canonical_group_id") != group_id
                or not isinstance(lineage, Mapping)
                or lineage.get("schema_version")
                != SOURCE_APPEARANCE_COHORT_CONTRACT_SCHEMA_VERSION
                or lineage.get("method")
                != SOURCE_APPEARANCE_COHORT_METHOD
                or lineage.get("cohort_id") != cohort_id
                or lineage.get("contract_sha256") != contract_sha256
                or lineage.get("canonical_group_id") != group_id
                or lineage.get("anchor_part_ids") != anchor_part_ids
                or lineage.get("expected_member_part_ids") != expected_part_ids
                or lineage.get("propagated_member_part_ids")
                != propagated_part_ids
                or lineage.get("registry_sha256")
                != contract.get("registry_sha256")
                or lineage.get("spatial_report_sha256")
                != contract.get("spatial_report_sha256")
                or lineage.get("annotation_input_plan_sha256")
                != contract.get("annotation_input_plan_sha256")
                or lineage.get("exact_cover") is not True
                or lineage.get("member_role")
                != (
                    "anchor"
                    if part_id in anchor_part_ids
                    else "propagated_member"
                    if part_id in propagated_part_ids
                    else "existing_member"
                )
            ):
                raise MultigroupExactMdlTournamentError(
                    "source-appearance cohort lineage is incomplete for "
                    f"part {part_id}"
                )
        claimed_propagated_part_ids.update(propagated_part_ids)
        expected_by_group.setdefault(group_id, set()).update(expected_part_ids)
        contract_ids_by_group.setdefault(group_id, []).append(cohort_id)

    declared_contract_count = raw_cohort_audit.get("cohort_count")
    declared_member_count = raw_cohort_audit.get("expected_member_count")
    declared_propagated_count = raw_cohort_audit.get("propagated_member_count")
    if (
        isinstance(declared_contract_count, bool)
        or not isinstance(declared_contract_count, int)
        or declared_contract_count != len(raw_contracts)
        or isinstance(declared_member_count, bool)
        or not isinstance(declared_member_count, int)
        or declared_member_count != len(claimed_part_ids)
        or isinstance(declared_propagated_count, bool)
        or not isinstance(declared_propagated_count, int)
        or declared_propagated_count != len(claimed_propagated_part_ids)
    ):
        raise MultigroupExactMdlTournamentError(
            "source-appearance cohort audit summary is inconsistent"
        )
    return expected_by_group, {
        "enabled": enabled,
        "contract_count": len(raw_contracts),
        "expected_member_count": len(claimed_part_ids),
        "expected_member_part_ids_by_group": {
            group_id: sorted(part_ids)
            for group_id, part_ids in sorted(expected_by_group.items())
        },
        "contract_ids_by_group": {
            group_id: sorted(cohort_ids)
            for group_id, cohort_ids in sorted(contract_ids_by_group.items())
        },
        "annotation_audit_verified": True,
        "exact_cover": True,
    }


def build_multigroup_exact_mdl_queue(
    *,
    source_plan: Mapping[str, Any],
    material_candidates_by_group: Mapping[str, Mapping[str, Any]],
    material_choice_audit: Mapping[str, Any],
    palette_fusion: Mapping[str, Any],
    allowed_material_ids: set[str],
    maximum_candidates: int,
    selection_objective: str = SELECTION_OBJECTIVE_VISUAL,
    minimum_reference_footprint_score: float = 0.0,
    quality_report: Mapping[str, Any] | None = None,
    visual_group_annotation_audit: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Discover every candidate-bearing visual group in deterministic order."""

    if (
        isinstance(maximum_candidates, bool)
        or not isinstance(maximum_candidates, int)
        or maximum_candidates < 2
    ):
        raise MultigroupExactMdlTournamentError(
            "maximum_candidates must be at least two"
        )
    if selection_objective not in {
        SELECTION_OBJECTIVE_SEMANTIC,
        SELECTION_OBJECTIVE_VISUAL,
    }:
        raise MultigroupExactMdlTournamentError(
            f"unsupported selection objective: {selection_objective}"
        )
    minimum_footprint = _unit_interval(
        minimum_reference_footprint_score,
        "minimum_reference_footprint_score",
    )
    assignments = _assignments(source_plan, "multi-group queue source plan")
    (
        source_appearance_expected_by_group,
        source_appearance_coverage_audit,
    ) = _validated_source_appearance_cohort_contracts(
        source_plan=source_plan,
        assignments=assignments,
        visual_group_annotation_audit=visual_group_annotation_audit,
    )
    try:
        membership_exclusions = membership_exclusions_by_group(source_plan)
    except MembershipTournamentError as exc:
        raise MultigroupExactMdlTournamentError(
            f"multi-group queue received an invalid frozen membership decision: {exc}"
        ) from exc
    try:
        family_contract = build_part_family_contract(
            plan=source_plan,
            material_choice_audit=material_choice_audit,
            palette_fusion=palette_fusion,
        )
    except ExactMdlTournamentError as exc:
        raise MultigroupExactMdlTournamentError(str(exc)) from exc
    canonical_palette = palette_fusion.get("canonical_palette")
    raw_groups = (
        canonical_palette.get("groups")
        if isinstance(canonical_palette, Mapping)
        else None
    )
    groups = _sequence(raw_groups, "palette_fusion.canonical_palette.groups")
    fusion_groups: dict[str, Mapping[str, Any]] = {}
    for group in groups:
        if not isinstance(group, Mapping):
            continue
        group_id = group.get("group_id")
        if isinstance(group_id, str) and group_id:
            if group_id in fusion_groups:
                raise MultigroupExactMdlTournamentError(
                    f"palette fusion repeats group {group_id}"
                )
            fusion_groups[group_id] = group

    entities_by_group: dict[str, list[dict[str, str]]] = {}
    for part_id, assignment in assignments.items():
        provenance = assignment.get("provenance")
        group_id = (
            provenance.get("canonical_group_id")
            if isinstance(provenance, Mapping)
            else None
        )
        if isinstance(group_id, str) and group_id:
            if part_id in membership_exclusions.get(group_id, set()):
                pass
            elif (
                selection_objective == SELECTION_OBJECTIVE_VISUAL
                or material_entity_contract_key(part_id) in family_contract
            ):
                entities_by_group.setdefault(group_id, []).append(
                    {
                        "entity_kind": "assignment",
                        "part_id": part_id,
                    }
                )
        raw_subset_groups = (
            provenance.get("face_subset_canonical_group_ids")
            if isinstance(provenance, Mapping)
            else None
        )
        if raw_subset_groups is None:
            continue
        if not isinstance(raw_subset_groups, Mapping):
            raise MultigroupExactMdlTournamentError(
                f"assignment {part_id} face-subset group map must be an object"
            )
        subsets = _face_subsets(assignment, f"multi-group queue/{part_id}")
        unknown_subset_names = sorted(set(raw_subset_groups) - set(subsets))
        if unknown_subset_names:
            raise MultigroupExactMdlTournamentError(
                f"assignment {part_id} face-subset group map references absent "
                f"subsets: {unknown_subset_names}"
            )
        for subset_name, subset_group_id in sorted(raw_subset_groups.items()):
            if (
                not isinstance(subset_name, str)
                or not subset_name
                or not isinstance(subset_group_id, str)
                or not subset_group_id
            ):
                raise MultigroupExactMdlTournamentError(
                    f"assignment {part_id} has an invalid face-subset group map"
                )
            if (
                selection_objective == SELECTION_OBJECTIVE_VISUAL
                or material_entity_contract_key(part_id, subset_name) in family_contract
            ):
                entities_by_group.setdefault(subset_group_id, []).append(
                    {
                        "entity_kind": "face_subset",
                        "part_id": part_id,
                        "subset_name": subset_name,
                    }
                )

    source_appearance_coverage_blockers: list[dict[str, Any]] = []
    source_appearance_queued_by_group: dict[str, list[str]] = {}
    for group_id, expected_part_ids in sorted(
        source_appearance_expected_by_group.items()
    ):
        actual_assignment_part_ids = {
            str(entity["part_id"])
            for entity in entities_by_group.get(group_id, [])
            if entity.get("entity_kind") == "assignment"
        }
        queued_part_ids = sorted(expected_part_ids & actual_assignment_part_ids)
        missing_part_ids = sorted(expected_part_ids - actual_assignment_part_ids)
        source_appearance_queued_by_group[group_id] = queued_part_ids
        if missing_part_ids:
            source_appearance_coverage_blockers.append(
                {
                    "group_id": group_id,
                    "reason": "SOURCE_APPEARANCE_COHORT_TARGET_INCOMPLETE",
                    "expected_part_ids": sorted(expected_part_ids),
                    "queued_part_ids": queued_part_ids,
                    "missing_part_ids": missing_part_ids,
                }
            )

    queue: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    # Start from the union, rather than only groups that survived the
    # part-localization handoff.  Otherwise a strong multi-view palette group
    # can disappear before the tournament and the queue can incorrectly claim
    # complete coverage.  A significant reference group without target
    # entities is a hard coverage blocker: silently ignoring it would publish
    # a visually incomplete Look.
    discovered_group_ids = set(entities_by_group) | set(fusion_groups)
    for group_id in sorted(discovered_group_ids):
        target_entities = sorted(entities_by_group.get(group_id, []), key=_entity_key)
        target_part_ids = sorted({entity["part_id"] for entity in target_entities})
        candidate_document = material_candidates_by_group.get(group_id)
        fusion_group = fusion_groups.get(group_id)
        if not isinstance(fusion_group, Mapping):
            exclusions.append({"group_id": group_id, "reason": "MISSING_FUSION_GROUP"})
            continue
        footprint = _reference_footprint(fusion_group)
        if footprint < minimum_footprint:
            exclusions.append(
                {
                    "group_id": group_id,
                    "reason": "BELOW_REFERENCE_FOOTPRINT_THRESHOLD",
                    "reference_footprint_score": footprint,
                }
            )
            continue
        raw_sources = fusion_group.get("sources", [])
        canonical_reference_view_ids = sorted(
            {
                str(source["view_id"])
                for source in raw_sources
                if isinstance(source, Mapping)
                and isinstance(source.get("view_id"), str)
                and source["view_id"]
            }
        )
        if len(canonical_reference_view_ids) < 2:
            baseline_presence = (
                _baseline_group_presence_evidence(
                    group_id=group_id,
                    source_view_ids=canonical_reference_view_ids,
                    palette_fusion=palette_fusion,
                    quality_report=quality_report,
                    minimum_source_view_count=1,
                )
                if not target_entities
                else None
            )
            exclusions.append(
                {
                    "group_id": group_id,
                    "reason": "INSUFFICIENT_INDEPENDENT_REFERENCE_VIEWS",
                    "reference_view_ids": canonical_reference_view_ids,
                    "reference_view_count": len(canonical_reference_view_ids),
                    "baseline_preserved": True,
                    "authored_target_entity_count": len(target_entities),
                    "baseline_presence_evidence": baseline_presence,
                }
            )
            continue
        reference_scope = _trusted_scoring_reference_scope(
            group_id=group_id,
            fusion_group=fusion_group,
            palette_fusion=palette_fusion,
            quality_report=quality_report,
        )
        reference_view_ids = list(reference_scope["reference_view_ids"])
        if len(reference_view_ids) < 2:
            exclusions.append(
                {
                    "group_id": group_id,
                    "reason": INSUFFICIENT_TRUSTED_SCORING_REFERENCE_VIEWS,
                    "canonical_reference_view_ids": canonical_reference_view_ids,
                    "reference_view_ids": reference_view_ids,
                    "reference_view_count": len(reference_view_ids),
                    "baseline_preserved": True,
                    "authored_target_entity_count": len(target_entities),
                    "trusted_scoring_reference_scope": reference_scope,
                }
            )
            continue
        if not isinstance(candidate_document, Mapping):
            exclusions.append(
                {"group_id": group_id, "reason": "MISSING_CANDIDATE_DOCUMENT"}
            )
            continue
        if not target_entities:
            baseline_presence = _baseline_group_presence_evidence(
                group_id=group_id,
                source_view_ids=reference_view_ids,
                palette_fusion=palette_fusion,
                quality_report=quality_report,
            )
            if baseline_presence is not None:
                exclusions.append(
                    {
                        "group_id": group_id,
                        "reason": BASELINE_GROUP_PRESENT_WITHOUT_LOCALIZATION,
                        "reference_footprint_score": footprint,
                        "reference_view_ids": reference_view_ids,
                        "reference_view_count": len(reference_view_ids),
                        "baseline_preserved": True,
                        "authored_target_entity_count": 0,
                        "baseline_presence_evidence": baseline_presence,
                    }
                )
                continue
            exclusions.append(
                {
                    "group_id": group_id,
                    "reason": "NO_TARGET_MATERIAL_ENTITIES",
                    "reference_footprint_score": footprint,
                    "reference_view_ids": reference_view_ids,
                    "reference_view_count": len(reference_view_ids),
                }
            )
            continue
        source_material_ids = {
            _entity_material_id(assignments, entity) for entity in target_entities
        }
        allowed_families = set().union(
            *(
                family_contract.get(
                    material_entity_contract_key(
                        entity["part_id"],
                        entity.get("subset_name"),
                    ),
                    set(),
                )
                for entity in target_entities
            )
        )
        try:
            ranked_candidates, merge_audit = _merged_ranked_candidates(
                candidate_document=candidate_document,
                source_material_ids=source_material_ids,
                allowed_material_ids=allowed_material_ids,
                allowed_families=allowed_families,
                maximum_candidates=maximum_candidates,
                visual_similarity_first=(
                    selection_objective == SELECTION_OBJECTIVE_VISUAL
                ),
            )
        except MultigroupExactMdlTournamentError as exc:
            raise MultigroupExactMdlTournamentError(f"group {group_id}: {exc}") from exc
        if len(ranked_candidates) < 2:
            exclusions.append(
                {
                    "group_id": group_id,
                    "reason": "NO_ALTERNATE_EXACT_MDL_CANDIDATE",
                }
            )
            continue
        queue.append(
            {
                "group_id": group_id,
                "target_part_ids": target_part_ids,
                "target_part_count": len(target_part_ids),
                "target_entities": copy.deepcopy(target_entities),
                "target_entity_count": len(target_entities),
                "target_face_subset_count": sum(
                    entity["entity_kind"] == "face_subset" for entity in target_entities
                ),
                "source_appearance_cohort_expected_part_ids": sorted(
                    source_appearance_expected_by_group.get(group_id, set())
                ),
                "source_appearance_cohort_contract_ids": list(
                    source_appearance_coverage_audit[
                        "contract_ids_by_group"
                    ].get(group_id, [])
                ),
                "reference_footprint_score": footprint,
                "canonical_reference_view_ids": canonical_reference_view_ids,
                "reference_view_ids": reference_view_ids,
                "reference_view_count": len(reference_view_ids),
                "trusted_scoring_reference_scope": reference_scope,
                "source_material_id": (
                    next(iter(source_material_ids))
                    if len(source_material_ids) == 1
                    else None
                ),
                "source_material_ids": sorted(source_material_ids),
                "allowed_families": sorted(allowed_families),
                "candidate_count": len(ranked_candidates),
                "candidate_material_ids": [
                    record["material_id"] for record in ranked_candidates
                ],
                "primary_and_tournament_merge": merge_audit,
            }
        )
    queue.sort(
        key=lambda group: (
            -float(group["reference_footprint_score"]),
            -int(group["target_part_count"]),
            str(group["group_id"]),
        )
    )
    queued_entity_keys: set[tuple[str, str]] = set()
    for group in queue:
        group_entity_keys = {_entity_key(entity) for entity in group["target_entities"]}
        overlap = queued_entity_keys.intersection(group_entity_keys)
        if overlap:
            raise MultigroupExactMdlTournamentError(
                "significant visual groups overlap material entities: "
                + ", ".join(
                    f"{part_id}:{subset_name or '<assignment>'}"
                    for part_id, subset_name in sorted(overlap)
                )
            )
        queued_entity_keys.update(group_entity_keys)
    coverage_blockers = [
        exclusion
        for exclusion in exclusions
        if exclusion["reason"]
        not in {
            "BELOW_REFERENCE_FOOTPRINT_THRESHOLD",
            "INSUFFICIENT_INDEPENDENT_REFERENCE_VIEWS",
            INSUFFICIENT_TRUSTED_SCORING_REFERENCE_VIEWS,
            BASELINE_GROUP_PRESENT_WITHOUT_LOCALIZATION,
        }
    ]
    coverage_blockers.extend(source_appearance_coverage_blockers)
    audit = {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "status": "PLANNED",
        "source_plan_sha256": _canonical_sha256(source_plan),
        "selection_objective": selection_objective,
        "minimum_reference_footprint_score": minimum_footprint,
        "significant_group_count": len(queue),
        "significant_group_ids": [str(group["group_id"]) for group in queue],
        "groups": copy.deepcopy(queue),
        "excluded_group_count": len(exclusions),
        "excluded_groups": exclusions,
        "coverage_blocker_count": len(coverage_blockers),
        "coverage_blockers": coverage_blockers,
        "all_discovered_significant_groups_queued": not coverage_blockers,
        "all_candidate_bearing_significant_groups_queued": not coverage_blockers,
        "minimum_independent_reference_views_per_significant_group": 2,
        "insufficient_reference_groups_preserve_baseline": True,
        "insufficient_trusted_scoring_groups_preserve_baseline": True,
        "delivered_unlocalized_groups_preserve_baseline": True,
        "baseline_presence_requires_all_trusted_source_views": True,
        "baseline_presence_requires_exact_recall": 1.0,
        "baseline_presence_uses_material_or_semantic_tokens": False,
        "membership_freeze_applied": bool(membership_exclusions),
        "membership_excluded_part_count": sum(
            len(part_ids) for part_ids in membership_exclusions.values()
        ),
        "membership_excluded_part_ids_by_group": {
            group_id: sorted(part_ids)
            for group_id, part_ids in sorted(membership_exclusions.items())
        },
        "one_group_at_a_time": True,
        "face_subset_groups_supported": True,
        "target_entity_schema": ("assignment-or-face-subset-owner-reference/v1"),
        "face_subset_schema_fields_unchanged": True,
        "parameters_locked_to_library_defaults": True,
        "source_appearance_cohort_coverage": {
            **source_appearance_coverage_audit,
            "queued_member_part_ids_by_group": {
                group_id: list(part_ids)
                for group_id, part_ids in sorted(
                    source_appearance_queued_by_group.items()
                )
            },
            "coverage_blockers": copy.deepcopy(
                source_appearance_coverage_blockers
            ),
            "coverage_blocker_count": len(
                source_appearance_coverage_blockers
            ),
            "exact_cover": not source_appearance_coverage_blockers,
        },
    }
    return queue, audit


def _comparison_score(
    value: Any,
) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        return None
    return float(value)


def _local_comparison_contract(
    quality_report: Any,
    *,
    selection_objective: str,
) -> dict[str, Any]:
    """Classify score completeness independently from PASS/FAIL/REVIEW."""

    reasons: list[str] = []
    view_reason_codes: set[str] = set()
    if (
        not isinstance(quality_report, Mapping)
        or quality_report.get("schema_version") != "qwen-reference-render-comparison/v1"
    ):
        return {
            "complete": False,
            "all_view_nonfail": False,
            "selection_score": None,
            "selection_score_kind": None,
            "aggregate_status": None,
            "reference_view_ids": [],
            "reason_codes": ["QUALITY_REPORT_SCHEMA_INVALID"],
            "view_reason_codes": [],
        }
    quality_inputs = quality_report.get("inputs")
    comparison_scope = (
        quality_inputs.get("comparison_scope")
        if isinstance(quality_inputs, Mapping)
        else None
    )
    raw_expected_view_ids = (
        comparison_scope.get("reference_view_ids")
        if isinstance(comparison_scope, Mapping)
        else None
    )
    expected_view_array = isinstance(
        raw_expected_view_ids, Sequence
    ) and not isinstance(raw_expected_view_ids, (str, bytes))
    expected_view_ids: list[str] = []
    if expected_view_array:
        expected_view_ids = [
            str(view_id)
            for view_id in raw_expected_view_ids
            if isinstance(view_id, str) and view_id
        ]
    if (
        len(expected_view_ids) < 2
        or not expected_view_array
        or len(expected_view_ids) != len(raw_expected_view_ids)
        or expected_view_ids != sorted(set(expected_view_ids))
    ):
        reasons.append("LOCAL_REFERENCE_VIEW_SET_INCOMPLETE")

    raw_views = quality_report.get("views")
    views = (
        list(raw_views)
        if isinstance(raw_views, Sequence) and not isinstance(raw_views, (str, bytes))
        else []
    )
    actual_view_ids: list[str] = []
    actual_view_statuses: list[str] = []
    per_view_complete = True
    for view in views:
        if not isinstance(view, Mapping):
            per_view_complete = False
            continue
        view_id = view.get("reference_view_id")
        if not isinstance(view_id, str) or not view_id:
            per_view_complete = False
        else:
            actual_view_ids.append(view_id)
        raw_view_reasons = view.get("reasons")
        if isinstance(raw_view_reasons, Sequence) and not isinstance(
            raw_view_reasons, (str, bytes)
        ):
            view_reason_codes.update(
                reason
                for reason in raw_view_reasons
                if isinstance(reason, str) and reason
            )
        view_status = view.get("status")
        if view_status not in {"PASS", "REVIEW", "FAIL"}:
            per_view_complete = False
        else:
            actual_view_statuses.append(str(view_status))
        color = view.get("material_color")
        if (
            not isinstance(color, Mapping)
            or _comparison_score(color.get("score")) is None
        ):
            per_view_complete = False
        if selection_objective == SELECTION_OBJECTIVE_VISUAL:
            texture = view.get("material_texture")
            if (
                not isinstance(texture, Mapping)
                or _comparison_score(texture.get("score")) is None
                or _comparison_score(view.get("material_appearance_score")) is None
            ):
                per_view_complete = False
    if (
        not per_view_complete
        or sorted(actual_view_ids) != expected_view_ids
        or len(actual_view_ids) != len(set(actual_view_ids))
    ):
        reasons.append("LOCAL_PER_VIEW_SCORES_INCOMPLETE")

    aggregate = quality_report.get("aggregate")
    if not isinstance(aggregate, Mapping):
        return {
            "complete": False,
            "all_view_nonfail": False,
            "selection_score": None,
            "selection_score_kind": None,
            "aggregate_status": None,
            "reference_view_ids": expected_view_ids,
            "reason_codes": sorted(set(reasons + ["AGGREGATE_MISSING"])),
            "view_reason_codes": sorted(view_reason_codes),
        }
    color_score = _comparison_score(aggregate.get("material_color_score"))
    texture_score = _comparison_score(aggregate.get("material_texture_score"))
    appearance_score = _comparison_score(aggregate.get("material_appearance_score"))
    if color_score is None:
        reasons.append("AGGREGATE_COLOR_SCORE_INCOMPLETE")
    if selection_objective == SELECTION_OBJECTIVE_VISUAL and (
        texture_score is None or appearance_score is None
    ):
        reasons.append("AGGREGATE_TEXTURE_APPEARANCE_SCORE_INCOMPLETE")
    view_count = len(expected_view_ids)
    aggregate_status_counts = {
        "PASS": aggregate.get("passed_view_count"),
        "REVIEW": aggregate.get("review_view_count"),
        "FAIL": aggregate.get("failed_view_count"),
        "UNSCORABLE": aggregate.get("unscorable_view_count"),
    }
    status_counts_complete = (
        all(
            isinstance(count, int) and not isinstance(count, bool) and count >= 0
            for count in aggregate_status_counts.values()
        )
        and sum(int(count) for count in aggregate_status_counts.values()) == view_count
        and all(
            aggregate_status_counts[status] == actual_view_statuses.count(status)
            for status in aggregate_status_counts
        )
    )
    coverage_complete = (
        view_count >= 2
        and aggregate.get("reference_view_count") == view_count
        and aggregate.get("comparable_view_count") == view_count
        and aggregate.get("unscorable_view_count") == 0
        and aggregate.get("reference_view_coverage_status") == "PASS"
        and status_counts_complete
    )
    if selection_objective == SELECTION_OBJECTIVE_VISUAL:
        coverage_complete = (
            coverage_complete
            and aggregate.get("texture_comparable_view_count") == view_count
            and aggregate.get("texture_unscorable_view_count") == 0
        )
    if not coverage_complete:
        reasons.append("LOCAL_VIEW_COVERAGE_INCOMPLETE")
    selection_score = (
        appearance_score
        if selection_objective == SELECTION_OBJECTIVE_VISUAL
        else color_score
    )
    all_view_nonfail = (
        coverage_complete
        and aggregate.get("status") in {"PASS", "REVIEW"}
        and aggregate_status_counts["FAIL"] == 0
        and aggregate_status_counts["UNSCORABLE"] == 0
        and len(actual_view_statuses) == view_count
        and set(actual_view_statuses).issubset({"PASS", "REVIEW"})
    )
    return {
        "complete": not reasons,
        "all_view_nonfail": all_view_nonfail,
        "selection_score": selection_score,
        "selection_score_kind": (
            "material_appearance_score"
            if selection_objective == SELECTION_OBJECTIVE_VISUAL
            else "material_color_score"
        ),
        "aggregate_status": aggregate.get("status"),
        "reference_view_ids": expected_view_ids,
        "reason_codes": sorted(set(reasons)),
        "view_reason_codes": sorted(view_reason_codes),
    }


def _whole_asset_comparison_contract(
    quality_report: Any,
    *,
    selection_objective: str,
) -> dict[str, Any]:
    """Validate a complete whole-asset guard using the local score contract."""

    if not isinstance(quality_report, Mapping):
        return {
            "complete": False,
            "all_view_nonfail": False,
            "selection_score": None,
            "selection_score_kind": None,
            "aggregate_status": None,
            "reference_view_ids": [],
            "reason_codes": ["GLOBAL_QUALITY_REPORT_SCHEMA_INVALID"],
            "view_reason_codes": [],
        }
    quality_inputs = quality_report.get("inputs")
    comparison_scope = (
        quality_inputs.get("comparison_scope")
        if isinstance(quality_inputs, Mapping)
        else None
    )
    selected_mapping = (
        quality_inputs.get("selected_view_mapping")
        if isinstance(quality_inputs, Mapping)
        else None
    )
    seeded_mapping = (
        quality_inputs.get("seeded_view_mapping")
        if isinstance(quality_inputs, Mapping)
        else None
    )
    mapping_valid = (
        isinstance(selected_mapping, Mapping)
        and len(selected_mapping) >= 2
        and all(
            isinstance(reference_id, str)
            and reference_id
            and isinstance(render_id, str)
            and render_id
            for reference_id, render_id in selected_mapping.items()
        )
        and seeded_mapping == selected_mapping
    )
    if comparison_scope != {"mode": "whole_asset"} or not mapping_valid:
        return {
            "complete": False,
            "all_view_nonfail": False,
            "selection_score": None,
            "selection_score_kind": None,
            "aggregate_status": None,
            "reference_view_ids": [],
            "reason_codes": [
                (
                    "GLOBAL_COMPARISON_SCOPE_INVALID"
                    if comparison_scope != {"mode": "whole_asset"}
                    else "GLOBAL_VIEW_MAPPING_INVALID"
                )
            ],
            "view_reason_codes": [],
        }
    assert isinstance(selected_mapping, Mapping)
    expected_view_ids = sorted(str(view_id) for view_id in selected_mapping)
    raw_views = quality_report.get("views")
    if not isinstance(raw_views, Sequence) or isinstance(raw_views, (str, bytes)):
        return {
            "complete": False,
            "all_view_nonfail": False,
            "selection_score": None,
            "selection_score_kind": None,
            "aggregate_status": None,
            "reference_view_ids": expected_view_ids,
            "reason_codes": ["GLOBAL_PER_VIEW_SCORES_INCOMPLETE"],
            "view_reason_codes": [],
        }
    for raw_view in raw_views:
        if not isinstance(raw_view, Mapping):
            mapping_valid = False
            break
        reference_id = raw_view.get("reference_view_id")
        if (
            not isinstance(reference_id, str)
            or reference_id not in selected_mapping
            or raw_view.get("render_view_id") != selected_mapping[reference_id]
        ):
            mapping_valid = False
            break
    if not mapping_valid:
        return {
            "complete": False,
            "all_view_nonfail": False,
            "selection_score": None,
            "selection_score_kind": None,
            "aggregate_status": None,
            "reference_view_ids": expected_view_ids,
            "reason_codes": ["GLOBAL_VIEW_MAPPING_INVALID"],
            "view_reason_codes": [],
        }

    normalized = copy.deepcopy(dict(quality_report))
    normalized_inputs = copy.deepcopy(dict(quality_inputs))
    normalized_scope = copy.deepcopy(dict(comparison_scope))
    normalized_scope["reference_view_ids"] = expected_view_ids
    normalized_inputs["comparison_scope"] = normalized_scope
    normalized["inputs"] = normalized_inputs
    contract = _local_comparison_contract(
        normalized,
        selection_objective=selection_objective,
    )
    contract["reason_codes"] = [
        (
            "GLOBAL_" + reason.removeprefix("LOCAL_")
            if reason.startswith("LOCAL_")
            else reason
        )
        for reason in contract["reason_codes"]
    ]
    contract["comparison_scope"] = "whole_asset"
    return contract


def _round_structure(
    *,
    current_plan: Mapping[str, Any],
    group_id: str,
    target_part_ids: Sequence[str],
    target_entities: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    selection_objective: str,
) -> tuple[
    str,
    dict[str, list[dict[str, str]]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    if len(candidates) < 2:
        raise MultigroupExactMdlTournamentError(
            f"group {group_id} requires at least two candidates"
        )
    candidate_deltas: dict[str, list[dict[str, str]]] = {}
    local_comparison_contracts: dict[str, dict[str, Any]] = {}
    global_comparison_contracts: dict[str, dict[str, Any]] = {}
    baseline_candidate_ids: list[str] = []
    seen_candidate_ids: set[str] = set()
    target_parts = set(target_part_ids)
    target_entity_keys = {_entity_key(entity) for entity in target_entities}
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise MultigroupExactMdlTournamentError(
                f"group {group_id} candidate[{index}] must be an object"
            )
        candidate_id = candidate.get("candidate_id")
        plan = candidate.get("plan")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in seen_candidate_ids
            or not isinstance(plan, Mapping)
        ):
            raise MultigroupExactMdlTournamentError(
                f"group {group_id} candidate[{index}] is invalid"
            )
        seen_candidate_ids.add(candidate_id)
        quality_report = candidate.get("quality_report")
        quality_inputs = (
            quality_report.get("inputs")
            if isinstance(quality_report, Mapping)
            else None
        )
        comparison_scope = (
            quality_inputs.get("comparison_scope")
            if isinstance(quality_inputs, Mapping)
            else None
        )
        if (
            not isinstance(comparison_scope, Mapping)
            or comparison_scope.get("mode") != "canonical_group_local"
            or comparison_scope.get("target_group_id") != group_id
            or comparison_scope.get("target_part_ids") != sorted(target_parts)
        ):
            raise MultigroupExactMdlTournamentError(
                f"group {group_id} candidate {candidate_id} lacks exact "
                "group-local reference/render comparison scope"
            )
        expected_target_entities = copy.deepcopy(list(target_entities))
        comparison_target_entities = comparison_scope.get("target_entities")
        has_face_subset_target = any(
            entity.get("subset_name") is not None for entity in target_entities
        )
        if comparison_target_entities is not None and (
            comparison_target_entities != expected_target_entities
        ):
            raise MultigroupExactMdlTournamentError(
                f"group {group_id} candidate {candidate_id} comparison scope "
                "targets different material entities"
            )
        if has_face_subset_target and (
            comparison_target_entities != expected_target_entities
            or comparison_scope.get("render_mask_granularity")
            != "containing_part_proxy"
            or comparison_scope.get("face_subset_render_mask_exact") is not False
        ):
            raise MultigroupExactMdlTournamentError(
                f"group {group_id} candidate {candidate_id} lacks audited "
                "face-subset comparison proxy scope"
            )
        local_comparison_contracts[candidate_id] = _local_comparison_contract(
            quality_report,
            selection_objective=selection_objective,
        )
        global_comparison_contracts[candidate_id] = (
            _whole_asset_comparison_contract(
                candidate.get("global_quality_report"),
                selection_objective=selection_objective,
            )
        )
        delta = _material_delta(
            current_plan,
            plan,
            label=f"group {group_id} candidate {candidate_id}",
        )
        candidate_deltas[candidate_id] = delta
        is_baseline = len(delta) == 0
        if candidate.get("is_baseline") is True and not is_baseline:
            raise MultigroupExactMdlTournamentError(
                f"group {group_id} candidate {candidate_id} falsely claims baseline"
            )
        if is_baseline:
            baseline_candidate_ids.append(candidate_id)
            continue
        changed_entity_keys = {
            (
                change["part_id"],
                str(change.get("subset_name", "")),
            )
            for change in delta
        }
        if not changed_entity_keys or not changed_entity_keys.issubset(
            target_entity_keys
        ):
            raise MultigroupExactMdlTournamentError(
                f"group {group_id} candidate {candidate_id} does not change "
                "only declared target material entities"
            )
        candidate_assignments = _assignments(
            plan,
            f"group {group_id} candidate {candidate_id}",
        )
        resulting_material_ids = {
            _entity_material_id(candidate_assignments, entity)
            for entity in target_entities
        }
        if len(resulting_material_ids) != 1:
            raise MultigroupExactMdlTournamentError(
                f"group {group_id} candidate {candidate_id} assigns multiple "
                "MDL identities to one visual group"
            )
    if len(baseline_candidate_ids) != 1:
        raise MultigroupExactMdlTournamentError(
            f"group {group_id} must contain exactly one baseline candidate"
        )
    return (
        baseline_candidate_ids[0],
        candidate_deltas,
        local_comparison_contracts,
        global_comparison_contracts,
    )


def select_exact_mdl_group_step(
    *,
    current_plan: Mapping[str, Any],
    group_id: str,
    target_part_ids: Sequence[str],
    target_entities: Sequence[Mapping[str, Any]] | None = None,
    candidates: Sequence[Mapping[str, Any]],
    allowed_material_ids: set[str],
    material_families_by_id: Mapping[str, str] | None = None,
    allowed_families_by_part: Mapping[str, set[str]] | None = None,
    selection_objective: str = SELECTION_OBJECTIVE_VISUAL,
    minimum_score_improvement: float = DEFAULT_MINIMUM_SCORE_IMPROVEMENT,
    minimum_winner_margin: float = DEFAULT_MINIMUM_WINNER_MARGIN,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate one rendered group round and conservatively accept or revert."""

    improvement_threshold = _unit_interval(
        minimum_score_improvement,
        "minimum_score_improvement",
    )
    margin_threshold = _unit_interval(
        minimum_winner_margin,
        "minimum_winner_margin",
    )
    if not isinstance(group_id, str) or not group_id:
        raise MultigroupExactMdlTournamentError("group_id must be a non-empty string")
    part_ids = _sorted_unique_texts(
        list(target_part_ids),
        "target_part_ids",
        require_sorted=True,
    )
    if not part_ids:
        raise MultigroupExactMdlTournamentError("target_part_ids must not be empty")
    current_assignments = _assignments(current_plan, "group-round current plan")
    entities = _normalized_target_entities(
        target_part_ids=part_ids,
        target_entities=target_entities,
        assignments=current_assignments,
        label=f"group {group_id}",
    )
    _require_entities_in_group(
        assignments=current_assignments,
        group_id=group_id,
        entities=entities,
        label="group-round selector",
    )
    (
        baseline_candidate_id,
        candidate_deltas,
        local_comparison_contracts,
        global_comparison_contracts,
    ) = _round_structure(
        current_plan=current_plan,
        group_id=group_id,
        target_part_ids=part_ids,
        target_entities=entities,
        candidates=candidates,
        selection_objective=selection_objective,
    )
    incomplete_challengers = [
        {
            "candidate_id": candidate_id,
            "local": copy.deepcopy(local_comparison_contracts[candidate_id]),
            "global": copy.deepcopy(global_comparison_contracts[candidate_id]),
        }
        for candidate_id in sorted(local_comparison_contracts)
        if (
            candidate_id != baseline_candidate_id
            and (
                local_comparison_contracts[candidate_id]["complete"] is not True
                or local_comparison_contracts[candidate_id]["all_view_nonfail"]
                is not True
                or global_comparison_contracts[candidate_id]["complete"] is not True
            )
        )
    ]
    # Prefer the whole-asset score when at least one challenger has a
    # complete non-failing global comparison.  During coordinate descent the
    # whole asset can legitimately remain FAIL because later visual groups
    # have not been repaired yet; in that case requiring a global PASS makes
    # every candidate ineligible and prevents the first group from ever
    # changing.  Fall back to the complete group-local score, while retaining
    # the whole-asset report as a strict no-regression guard.
    use_global_selection = any(
        candidate_id != baseline_candidate_id
        and contract["complete"] is True
        and contract["all_view_nonfail"] is True
        for candidate_id, contract in global_comparison_contracts.items()
    )
    selection_contracts = (
        global_comparison_contracts
        if use_global_selection
        else local_comparison_contracts
    )
    baseline_contract = selection_contracts[baseline_candidate_id]
    baseline_global_score = global_comparison_contracts[baseline_candidate_id].get(
        "selection_score"
    )
    global_regression_exclusions: list[dict[str, Any]] = []
    tournament_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id"))
        local_contract = local_comparison_contracts[candidate_id]
        global_contract = global_comparison_contracts[candidate_id]
        if (
            candidate_id != baseline_candidate_id
            and (
                local_contract["complete"] is not True
                or local_contract["all_view_nonfail"] is not True
                or global_contract["complete"] is not True
            )
        ):
            continue
        global_quality_report = candidate.get("global_quality_report")
        if not isinstance(global_quality_report, Mapping):
            if candidate_id == baseline_candidate_id:
                raise MultigroupExactMdlTournamentError(
                    f"group {group_id} baseline candidate lacks a hash-bound "
                    "whole-asset quality guard"
                )
            continue
        candidate_global_score = global_contract.get("selection_score")
        if (
            not use_global_selection
            and candidate_id != baseline_candidate_id
            and isinstance(baseline_global_score, (int, float))
            and not isinstance(baseline_global_score, bool)
            and isinstance(candidate_global_score, (int, float))
            and not isinstance(candidate_global_score, bool)
            and float(candidate_global_score) + 0.01
            < float(baseline_global_score)
        ):
            global_regression_exclusions.append(
                {
                    "candidate_id": candidate_id,
                    "baseline_global_score": float(baseline_global_score),
                    "candidate_global_score": float(candidate_global_score),
                    "maximum_global_score_regression": 0.01,
                }
            )
            continue
        selection_candidate = dict(candidate)
        if use_global_selection:
            selection_candidate["quality_report"] = global_quality_report
        tournament_candidates.append(selection_candidate)
    current_hash = _canonical_sha256(current_plan)
    try:
        selected_plan, tournament_audit = select_and_replay_exact_mdl_candidate(
            baseline_plan=current_plan,
            target_plan=current_plan,
            candidates=tournament_candidates,
            allowed_material_ids=allowed_material_ids,
            material_families_by_id=material_families_by_id,
            allowed_families_by_part=allowed_families_by_part,
            selection_objective=selection_objective,
        )
    except ExactMdlTournamentError as exc:
        if exc.audit is None:
            raise MultigroupExactMdlTournamentError(str(exc)) from exc
        output = copy.deepcopy(dict(current_plan))
        round_audit = {
            "schema_version": ROUND_SCHEMA_VERSION,
            "status": ROUND_FALLBACK_NO_ELIGIBLE_CANDIDATE,
            "group_id": group_id,
            "target_part_ids": part_ids,
            "target_entities": copy.deepcopy(entities),
            "input_plan_sha256": current_hash,
            "output_plan_sha256": current_hash,
            "baseline_candidate_id": baseline_candidate_id,
            "selected_candidate_id": None,
            "accepted_candidate_id": None,
            "baseline_score": baseline_contract["selection_score"],
            "baseline_score_complete": baseline_contract["complete"],
            "baseline_aggregate_status": baseline_contract["aggregate_status"],
            "baseline_comparability_reason_codes": baseline_contract["reason_codes"],
            "baseline_all_view_pass_required": False,
            "baseline_complete_comparable_score_required": True,
            "selected_score": None,
            "score_improvement": None,
            "runner_up_score": None,
            "winner_margin": None,
            "minimum_score_improvement": improvement_threshold,
            "minimum_winner_margin": margin_threshold,
            "fallback_to_input_plan": True,
            "material_changes": [],
            "parameters_locked_to_library_defaults": True,
            "selection_scope": (
                "whole_asset_guard"
                if use_global_selection
                else "canonical_group_local_with_whole_asset_nonregression_guard"
            ),
            "local_nonfail_evidence_required": True,
            "global_regression_exclusion_count": len(
                global_regression_exclusions
            ),
            "global_regression_exclusions": global_regression_exclusions,
            "tournament": exc.audit,
            "incomplete_challenger_count": len(incomplete_challengers),
            "incomplete_challengers": incomplete_challengers,
            "reason_codes": ["NO_ELIGIBLE_COMPLETE_NONFAILING_CANDIDATE"],
        }
        return output, round_audit

    records = {
        str(record["candidate_id"]): record for record in tournament_audit["candidates"]
    }
    for candidate_id, record in records.items():
        record["local_comparison"] = copy.deepcopy(
            local_comparison_contracts[candidate_id]
        )
        record["global_comparison"] = copy.deepcopy(
            global_comparison_contracts[candidate_id]
        )
        source_candidate = next(
            candidate
            for candidate in candidates
            if candidate.get("candidate_id") == candidate_id
        )
        record["local_quality_report_sha256"] = _canonical_sha256(
            source_candidate["quality_report"]
        )
        record["global_quality_report_sha256"] = _canonical_sha256(
            source_candidate["global_quality_report"]
        )
        record["selection_comparison_scope"] = (
            "whole_asset" if use_global_selection else "canonical_group_local"
        )
    baseline_record = records.get(baseline_candidate_id)
    selected_candidate_id = tournament_audit.get("selected_candidate_id")
    selected_record = (
        records.get(str(selected_candidate_id))
        if isinstance(selected_candidate_id, str)
        else None
    )
    if baseline_record is None or selected_record is None:
        raise MultigroupExactMdlTournamentError(
            f"group {group_id} tournament audit lost a candidate record"
        )

    baseline_record_tier = str(
        baseline_record.get("quality_tier", QUALITY_TIER_INELIGIBLE)
    )
    if baseline_record_tier not in QUALITY_TIER_RANK:
        raise MultigroupExactMdlTournamentError(
            f"group {group_id} baseline quality tier is invalid"
        )
    baseline_all_view_pass = (
        baseline_record_tier == QUALITY_TIER_ALL_VIEW_PASS
        and baseline_record.get("all_view_pass") is True
    )
    baseline_comparable = baseline_contract["complete"] is True
    baseline_quality_tier = (
        "INCOMPLETE"
        if not baseline_comparable
        else baseline_record_tier
        if baseline_record_tier != QUALITY_TIER_INELIGIBLE
        else "COMPLETE_FAIL"
    )
    baseline_score = baseline_contract["selection_score"]
    selected_score = selected_record.get("selection_score")
    if not isinstance(selected_score, (int, float)) or isinstance(selected_score, bool):
        raise MultigroupExactMdlTournamentError(
            f"group {group_id} selected score is invalid"
        )
    selected_quality_tier = str(
        selected_record.get("quality_tier", QUALITY_TIER_INELIGIBLE)
    )
    if selected_quality_tier not in QUALITY_TIER_RANK:
        raise MultigroupExactMdlTournamentError(
            f"group {group_id} selected quality tier is invalid"
        )
    selected_all_view_pass = (
        selected_quality_tier == QUALITY_TIER_ALL_VIEW_PASS
        and selected_record.get("all_view_pass") is True
    )
    quality_tier_promotion = (
        selected_candidate_id != baseline_candidate_id
        and QUALITY_TIER_RANK[selected_quality_tier]
        > QUALITY_TIER_RANK[baseline_record_tier]
    )
    all_view_pass_promotion = (
        selected_candidate_id != baseline_candidate_id
        and selected_quality_tier == QUALITY_TIER_ALL_VIEW_PASS
        and baseline_record_tier != QUALITY_TIER_ALL_VIEW_PASS
    )
    complete_nonfail_review_promotion = (
        selected_candidate_id != baseline_candidate_id
        and selected_quality_tier == QUALITY_TIER_COMPLETE_NONFAIL_REVIEW
        and QUALITY_TIER_RANK[selected_quality_tier]
        > QUALITY_TIER_RANK[baseline_record_tier]
    )
    runner_up_scores = [
        float(record["selection_score"])
        for candidate_id, record in records.items()
        if candidate_id != selected_candidate_id
        and candidate_id != baseline_candidate_id
        and record.get("eligible") is True
        and record.get("quality_tier") == selected_quality_tier
        and isinstance(record.get("selection_score"), (int, float))
        and not isinstance(record.get("selection_score"), bool)
    ]
    if (
        selected_candidate_id != baseline_candidate_id
        and baseline_record_tier == selected_quality_tier
        and baseline_comparable
        and isinstance(baseline_score, (int, float))
        and not isinstance(baseline_score, bool)
    ):
        runner_up_scores.append(float(baseline_score))
    runner_up_score = max(runner_up_scores) if runner_up_scores else None
    improvement = (
        float(selected_score) - float(baseline_score)
        if (
            baseline_comparable
            and isinstance(baseline_score, (int, float))
            and not isinstance(baseline_score, bool)
        )
        else None
    )
    positive_complete_nonfail_review_promotion = (
        complete_nonfail_review_promotion
        and improvement is not None
        and improvement > 1e-12
    )
    score_thresholds_applicable = (
        selected_candidate_id != baseline_candidate_id
        and not all_view_pass_promotion
        and not positive_complete_nonfail_review_promotion
    )
    winner_margin = (
        float(selected_score) - runner_up_score if runner_up_score is not None else None
    )

    status = ROUND_ACCEPTED
    reasons = ["CHALLENGER_EXCEEDS_BASELINE_AND_RUNNER_UP_MARGINS"]
    if not baseline_comparable:
        status = ROUND_FALLBACK_BASELINE_INELIGIBLE
        reasons = ["BASELINE_COMPLETE_COMPARABLE_SCORE_REQUIRED"]
    elif selected_candidate_id == baseline_candidate_id:
        status = ROUND_FALLBACK_BASELINE_BEST
        reasons = ["BASELINE_REMAINS_BEST_ELIGIBLE_CANDIDATE"]
    elif all_view_pass_promotion:
        reasons = ["ALL_VIEW_PASS_QUALITY_PROMOTION_OVER_NONPASS_BASELINE"]
    elif score_thresholds_applicable and (
        improvement is None or improvement + 1e-12 < improvement_threshold
    ):
        status = ROUND_FALLBACK_INSUFFICIENT_IMPROVEMENT
        reasons = ["MINIMUM_SCORE_IMPROVEMENT_NOT_MET"]
    elif (
        score_thresholds_applicable
        and not complete_nonfail_review_promotion
        and (
        winner_margin is None or winner_margin + 1e-12 < margin_threshold
        )
    ):
        status = ROUND_FALLBACK_AMBIGUOUS_WINNER
        reasons = ["MINIMUM_RUNNER_UP_MARGIN_NOT_MET"]
    elif complete_nonfail_review_promotion and (
        winner_margin is None or winner_margin + 1e-12 < margin_threshold
    ):
        reasons = [
            "COMPLETE_NONFAIL_REVIEW_QUALITY_PROMOTION_OVER_FAIL_BASELINE"
        ]
    elif selected_quality_tier == QUALITY_TIER_COMPLETE_NONFAIL_REVIEW:
        reasons = ["COMPLETE_NONFAIL_REVIEW_CLEAR_VISUAL_WINNER"]

    accepted = status == ROUND_ACCEPTED
    output = (
        copy.deepcopy(dict(selected_plan))
        if accepted
        else copy.deepcopy(dict(current_plan))
    )
    output_hash = _canonical_sha256(output)
    material_changes = candidate_deltas[str(selected_candidate_id)] if accepted else []
    round_audit = {
        "schema_version": ROUND_SCHEMA_VERSION,
        "status": status,
        "group_id": group_id,
        "target_part_ids": part_ids,
        "target_entities": copy.deepcopy(entities),
        "input_plan_sha256": current_hash,
        "output_plan_sha256": output_hash,
        "baseline_candidate_id": baseline_candidate_id,
        "selected_candidate_id": selected_candidate_id,
        "accepted_candidate_id": (selected_candidate_id if accepted else None),
        "baseline_score": (
            float(baseline_score)
            if isinstance(baseline_score, (int, float))
            and not isinstance(baseline_score, bool)
            else None
        ),
        "baseline_score_complete": baseline_comparable,
        "baseline_all_view_pass": baseline_all_view_pass,
        "baseline_aggregate_status": baseline_contract["aggregate_status"],
        "baseline_comparability_reason_codes": baseline_contract["reason_codes"],
        "baseline_all_view_pass_required": False,
        "baseline_complete_comparable_score_required": True,
        "baseline_quality_tier": baseline_quality_tier,
        "baseline_local_comparison": copy.deepcopy(
            local_comparison_contracts[baseline_candidate_id]
        ),
        "baseline_global_comparison": copy.deepcopy(baseline_contract),
        "selected_all_view_pass": selected_all_view_pass,
        "selected_quality_tier": (
            selected_quality_tier
        ),
        "selected_local_comparison": copy.deepcopy(
            local_comparison_contracts[str(selected_candidate_id)]
        ),
        "selected_global_comparison": copy.deepcopy(
            global_comparison_contracts[str(selected_candidate_id)]
        ),
        "quality_tier_promotion": quality_tier_promotion,
        "all_view_pass_quality_promotion": all_view_pass_promotion,
        "complete_nonfail_review_quality_promotion": (
            complete_nonfail_review_promotion
        ),
        "positive_score_complete_nonfail_review_quality_promotion": (
            positive_complete_nonfail_review_promotion
        ),
        "score_thresholds_applicable": score_thresholds_applicable,
        "selected_score": float(selected_score),
        "score_improvement": improvement,
        "runner_up_score": runner_up_score,
        "winner_margin": winner_margin,
        "minimum_score_improvement": improvement_threshold,
        "minimum_winner_margin": margin_threshold,
        "fallback_to_input_plan": not accepted,
        "material_changes": material_changes,
        "parameters_locked_to_library_defaults": True,
        "selection_scope": (
            "whole_asset_guard"
            if use_global_selection
            else "canonical_group_local_with_whole_asset_nonregression_guard"
        ),
        "local_nonfail_evidence_required": True,
        "all_verified_reference_views_guarded": True,
        "tournament": tournament_audit,
        "incomplete_challenger_count": len(incomplete_challengers),
        "incomplete_challengers": incomplete_challengers,
        "global_regression_exclusion_count": len(global_regression_exclusions),
        "global_regression_exclusions": global_regression_exclusions,
        "reason_codes": reasons,
    }
    return output, round_audit


def finalize_multigroup_exact_mdl_plan(
    *,
    initial_plan: Mapping[str, Any],
    current_plan: Mapping[str, Any],
    significant_group_ids: Sequence[str],
    round_audits: Sequence[Mapping[str, Any]],
    selection_objective: str = SELECTION_OBJECTIVE_VISUAL,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate full significant-group coverage and seal the final plan."""

    expected_group_ids = _sorted_unique_texts(
        list(significant_group_ids),
        "significant_group_ids",
    )
    if not expected_group_ids:
        raise MultigroupExactMdlTournamentError(
            "significant_group_ids must not be empty"
        )
    _assignments(initial_plan, "multi-group initial plan")
    _assignments(current_plan, "multi-group current plan")
    if len(round_audits) != len(expected_group_ids):
        raise MultigroupExactMdlTournamentError(
            "not every significant group has a round audit"
        )
    expected_input_hash = _canonical_sha256(initial_plan)
    seen_group_ids: list[str] = []
    accepted_group_ids: list[str] = []
    fallback_group_ids: list[str] = []
    changed_part_ids: set[str] = set()
    changed_entity_keys: set[tuple[str, str]] = set()
    changed_entities: list[dict[str, str]] = []
    for index, raw_audit in enumerate(round_audits):
        if not isinstance(raw_audit, Mapping):
            raise MultigroupExactMdlTournamentError(
                f"round_audits[{index}] must be an object"
            )
        if raw_audit.get("schema_version") != ROUND_SCHEMA_VERSION:
            raise MultigroupExactMdlTournamentError(
                f"round_audits[{index}] has an invalid schema"
            )
        group_id = raw_audit.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise MultigroupExactMdlTournamentError(
                f"round_audits[{index}] has an invalid group_id"
            )
        if raw_audit.get("input_plan_sha256") != expected_input_hash:
            raise MultigroupExactMdlTournamentError(
                f"round {group_id} does not continue the accepted plan hash chain"
            )
        output_hash = raw_audit.get("output_plan_sha256")
        if not isinstance(output_hash, str) or not output_hash:
            raise MultigroupExactMdlTournamentError(
                f"round {group_id} lacks an output plan hash"
            )
        expected_input_hash = output_hash
        seen_group_ids.append(group_id)
        if raw_audit.get("status") == ROUND_ACCEPTED:
            accepted_group_ids.append(group_id)
            raw_changes = _sequence(
                raw_audit.get("material_changes"),
                f"round {group_id}.material_changes",
            )
            for change in raw_changes:
                if not isinstance(change, Mapping) or not isinstance(
                    change.get("part_id"), str
                ):
                    raise MultigroupExactMdlTournamentError(
                        f"round {group_id} has an invalid material change"
                    )
                part_id = str(change["part_id"])
                subset_name = change.get("subset_name")
                if subset_name is not None and (
                    not isinstance(subset_name, str) or not subset_name
                ):
                    raise MultigroupExactMdlTournamentError(
                        f"round {group_id} has an invalid face-subset change"
                    )
                entity_key = (part_id, str(subset_name or ""))
                if entity_key in changed_entity_keys:
                    raise MultigroupExactMdlTournamentError(
                        "multiple visual groups changed material entity "
                        f"{part_id}:{subset_name or '<assignment>'}"
                    )
                changed_entity_keys.add(entity_key)
                changed_part_ids.add(part_id)
                entity = {
                    "entity_kind": (
                        "face_subset" if subset_name is not None else "assignment"
                    ),
                    "part_id": part_id,
                }
                if isinstance(subset_name, str):
                    entity["subset_name"] = subset_name
                changed_entities.append(entity)
        else:
            fallback_group_ids.append(group_id)
    if seen_group_ids != expected_group_ids:
        raise MultigroupExactMdlTournamentError(
            "round audits do not cover significant groups in declared order"
        )
    if expected_input_hash != _canonical_sha256(current_plan):
        raise MultigroupExactMdlTournamentError(
            "current plan does not match the final round output hash"
        )

    output = copy.deepcopy(dict(current_plan))
    provenance = output.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise MultigroupExactMdlTournamentError("final plan provenance is invalid")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "selection_objective": selection_objective,
        "initial_plan_sha256": _canonical_sha256(initial_plan),
        "preseal_final_plan_sha256": _canonical_sha256(current_plan),
        "significant_group_ids": expected_group_ids,
        "accepted_group_ids": accepted_group_ids,
        "fallback_group_ids": fallback_group_ids,
        "round_audits_sha256": _canonical_sha256(list(round_audits)),
        "coordinate_descent": True,
        "all_significant_groups_evaluated": True,
        "face_subset_groups_supported": True,
        "face_subset_schema_fields_unchanged": True,
        "parameters_locked_to_library_defaults": True,
    }
    provenance["immutable_mdl_after_selection"] = True
    provenance["multigroup_exact_mdl_coordinate_descent"] = summary
    _assignments(output, "sealed multi-group output plan")
    final_hash = _canonical_sha256(output)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETED",
        "selection_objective": selection_objective,
        "initial_plan_sha256": _canonical_sha256(initial_plan),
        "preseal_final_plan_sha256": _canonical_sha256(current_plan),
        "final_plan_sha256": final_hash,
        "significant_group_count": len(expected_group_ids),
        "evaluated_group_count": len(round_audits),
        "accepted_group_count": len(accepted_group_ids),
        "fallback_group_count": len(fallback_group_ids),
        "significant_group_ids": expected_group_ids,
        "accepted_group_ids": accepted_group_ids,
        "fallback_group_ids": fallback_group_ids,
        "changed_part_ids": sorted(changed_part_ids),
        "changed_entities": sorted(changed_entities, key=_entity_key),
        "rounds": copy.deepcopy(list(round_audits)),
        "coordinate_descent": True,
        "all_significant_groups_evaluated": True,
        "face_subset_groups_supported": True,
        "face_subset_schema_fields_unchanged": True,
        "baseline_candidate_required_per_group": True,
        "all_view_pass_quality_precedes_complete_nonpass": True,
        "minimum_improvement_and_winner_margin_required": ("within_same_quality_tier"),
        "parameters_locked_to_library_defaults": True,
    }
    return output, audit


def coordinate_descent_exact_mdl_groups(
    *,
    initial_plan: Mapping[str, Any],
    significant_groups: Sequence[Mapping[str, Any]],
    round_provider: RoundProvider,
    allowed_material_ids: set[str],
    material_families_by_id: Mapping[str, str] | None = None,
    allowed_families_by_part: Mapping[str, set[str]] | None = None,
    selection_objective: str = SELECTION_OBJECTIVE_VISUAL,
    minimum_score_improvement: float = DEFAULT_MINIMUM_SCORE_IMPROVEMENT,
    minimum_winner_margin: float = DEFAULT_MINIMUM_WINNER_MARGIN,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run sequential rendered rounds over every declared significant group."""

    if not callable(round_provider):
        raise MultigroupExactMdlTournamentError("round_provider must be callable")
    groups: list[dict[str, Any]] = []
    initial_assignments = _assignments(initial_plan, "coordinate-descent initial plan")
    occupied_entities: set[tuple[str, str]] = set()
    for index, raw_group in enumerate(significant_groups):
        if not isinstance(raw_group, Mapping):
            raise MultigroupExactMdlTournamentError(
                f"significant_groups[{index}] must be an object"
            )
        group_id = raw_group.get("group_id")
        raw_target_parts = raw_group.get("target_part_ids")
        if not isinstance(group_id, str) or not group_id:
            raise MultigroupExactMdlTournamentError(
                f"significant_groups[{index}] has an invalid group_id"
            )
        target_part_ids = _sorted_unique_texts(
            list(
                _sequence(
                    raw_target_parts,
                    f"significant_groups[{index}].target_part_ids",
                )
            ),
            f"significant_groups[{index}].target_part_ids",
            require_sorted=True,
        )
        raw_target_entities = raw_group.get("target_entities")
        if raw_target_entities is not None and (
            not isinstance(raw_target_entities, Sequence)
            or isinstance(raw_target_entities, (str, bytes))
        ):
            raise MultigroupExactMdlTournamentError(
                f"significant_groups[{index}].target_entities must be an array"
            )
        target_entities = _normalized_target_entities(
            target_part_ids=target_part_ids,
            target_entities=raw_target_entities,
            assignments=initial_assignments,
            label=f"significant_groups[{index}]",
        )
        entity_keys = {_entity_key(entity) for entity in target_entities}
        overlap = occupied_entities.intersection(entity_keys)
        if overlap:
            raise MultigroupExactMdlTournamentError(
                "significant groups overlap material entities: "
                + ", ".join(
                    f"{part_id}:{subset_name or '<assignment>'}"
                    for part_id, subset_name in sorted(overlap)
                )
            )
        occupied_entities.update(entity_keys)
        groups.append(
            {
                **copy.deepcopy(dict(raw_group)),
                "group_id": group_id,
                "target_part_ids": target_part_ids,
                "target_entities": target_entities,
            }
        )
    group_ids = [str(group["group_id"]) for group in groups]
    if len(group_ids) != len(set(group_ids)) or not group_ids:
        raise MultigroupExactMdlTournamentError(
            "significant groups must be non-empty with unique group_ids"
        )

    current = copy.deepcopy(dict(initial_plan))
    round_audits: list[dict[str, Any]] = []
    for group in groups:
        candidates = round_provider(
            copy.deepcopy(current),
            copy.deepcopy(group),
        )
        current, round_audit = select_exact_mdl_group_step(
            current_plan=current,
            group_id=str(group["group_id"]),
            target_part_ids=group["target_part_ids"],
            target_entities=group["target_entities"],
            candidates=candidates,
            allowed_material_ids=allowed_material_ids,
            material_families_by_id=material_families_by_id,
            allowed_families_by_part=allowed_families_by_part,
            selection_objective=selection_objective,
            minimum_score_improvement=minimum_score_improvement,
            minimum_winner_margin=minimum_winner_margin,
        )
        round_audits.append(round_audit)
    return finalize_multigroup_exact_mdl_plan(
        initial_plan=initial_plan,
        current_plan=current,
        significant_group_ids=group_ids,
        round_audits=round_audits,
        selection_objective=selection_objective,
    )


__all__ = [
    "DEFAULT_MINIMUM_SCORE_IMPROVEMENT",
    "DEFAULT_MINIMUM_WINNER_MARGIN",
    "MultigroupExactMdlTournamentError",
    "PLANNING_SCHEMA_VERSION",
    "QUEUE_SCHEMA_VERSION",
    "ROUND_ACCEPTED",
    "ROUND_FALLBACK_AMBIGUOUS_WINNER",
    "ROUND_FALLBACK_BASELINE_BEST",
    "ROUND_FALLBACK_BASELINE_INELIGIBLE",
    "ROUND_FALLBACK_INSUFFICIENT_IMPROVEMENT",
    "ROUND_FALLBACK_NO_ELIGIBLE_CANDIDATE",
    "ROUND_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_exact_mdl_group_candidate_plans",
    "build_multigroup_exact_mdl_queue",
    "coordinate_descent_exact_mdl_groups",
    "finalize_multigroup_exact_mdl_plan",
    "select_exact_mdl_group_step",
]
