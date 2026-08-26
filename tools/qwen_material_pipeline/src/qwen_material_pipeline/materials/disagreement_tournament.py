"""Fail-closed contract for unresolved forward/reverse MDL choices.

Qwen's forward and reverse candidate-order passes are intentionally
independent.  When they disagree, either answer may still be a useful rendered
seed, but neither answer is a completed material selection.  This module binds
the disagreement, the exact immutable NVIDIA MDL candidates that must be
rendered, and the only tournament outcomes that may resolve it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LEGACY_SCHEMA_VERSION = "qwen-forward-reverse-exact-mdl-tournament/v1"
SCHEMA_VERSION = "qwen-forward-reverse-exact-mdl-tournament/v2"
RESOLUTION_POLICY = "render_confirmation_required_before_material_lock/v1"
CHALLENGER_POLICY = "mvinverse_then_ranked_retrieval_exact_default/v1"
MVINVERSE_CHALLENGER_BASIS = "mvinverse_nearest_exact_library_default"
RANKED_RETRIEVAL_CHALLENGER_BASIS = (
    "ranked_retrieval_independent_exact_library_default_fallback"
)


class DisagreementTournamentContractError(ValueError):
    """Raised when an unresolved material choice could bypass render QA."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DisagreementTournamentContractError(f"{label} must be non-empty text")
    return value


def _unique_texts(value: Any, label: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DisagreementTournamentContractError(f"{label} must be an array")
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        text = _text(item, f"{label}[{index}]")
        if text in seen:
            raise DisagreementTournamentContractError(
                f"{label} contains duplicate material_id: {text}"
            )
        seen.add(text)
        result.append(text)
    return result


def build_disagreement_tournament_contract(
    *,
    forward_material_id: str,
    reverse_material_id: str,
    provisional_seed_material_id: str,
    mvinverse_exact_default_material_ids: Sequence[str],
    tournament_candidate_material_ids: Sequence[str],
    retrieval_exact_default_material_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Build and validate the immutable exact-MDL render contract."""

    forward = _text(forward_material_id, "forward_material_id")
    reverse = _text(reverse_material_id, "reverse_material_id")
    seed = _text(provisional_seed_material_id, "provisional_seed_material_id")
    if forward == reverse:
        raise DisagreementTournamentContractError(
            "a disagreement contract requires distinct forward and reverse choices"
        )
    if seed not in {forward, reverse}:
        raise DisagreementTournamentContractError(
            "the provisional seed must preserve one recorded Qwen choice"
        )
    mvinverse_ids = _unique_texts(
        mvinverse_exact_default_material_ids,
        "mvinverse_exact_default_material_ids",
    )
    retrieval_ids = _unique_texts(
        retrieval_exact_default_material_ids,
        "retrieval_exact_default_material_ids",
    )
    overlap = sorted(set(mvinverse_ids) & set(retrieval_ids))
    if overlap:
        raise DisagreementTournamentContractError(
            "independent exact-default candidates have ambiguous evidence "
            f"provenance: {overlap}"
        )
    independent_candidates = [
        {
            "material_id": material_id,
            "evidence_basis": MVINVERSE_CHALLENGER_BASIS,
        }
        for material_id in mvinverse_ids
    ] + [
        {
            "material_id": material_id,
            "evidence_basis": RANKED_RETRIEVAL_CHALLENGER_BASIS,
        }
        for material_id in retrieval_ids
    ]
    if not independent_candidates:
        raise DisagreementTournamentContractError(
            "at least one independent exact library-default candidate is required"
        )
    candidate_ids = _unique_texts(
        tournament_candidate_material_ids,
        "tournament_candidate_material_ids",
    )
    independent_ids = [
        str(candidate["material_id"]) for candidate in independent_candidates
    ]
    disputed_challengers = sorted(
        set(independent_ids) & {forward, reverse}
    )
    if disputed_challengers:
        raise DisagreementTournamentContractError(
            "an independent exact-default challenger repeats a disputed Qwen "
            f"choice: {disputed_challengers}"
        )
    required_ids = list(dict.fromkeys([forward, reverse, *independent_ids]))
    if len(required_ids) < 3:
        raise DisagreementTournamentContractError(
            "a disagreement tournament requires two disputed choices and an "
            "independent exact-default challenger"
        )
    missing = sorted(set(required_ids) - set(candidate_ids))
    if missing:
        raise DisagreementTournamentContractError(
            f"tournament candidate list omits required disagreement evidence: {missing}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "required": True,
        "reason_code": "FORWARD_REVERSE_EXACT_MDL_DISAGREEMENT",
        "resolution_policy": RESOLUTION_POLICY,
        "forward_material_id": forward,
        "reverse_material_id": reverse,
        "provisional_seed_material_id": seed,
        "provisional_seed_is_final_selection": False,
        "challenger_policy": CHALLENGER_POLICY,
        "mvinverse_exact_default_material_ids": mvinverse_ids,
        "retrieval_exact_default_material_ids": retrieval_ids,
        "mvinverse_challenger_status": (
            "available"
            if mvinverse_ids
            else "unavailable_or_no_eligible_exact_default_candidate"
        ),
        "independent_exact_default_candidates": independent_candidates,
        "required_candidate_material_ids": required_ids,
        "selected_mdl_parameters_mutable": False,
        "library_default_parameters_required": True,
    }


def validate_disagreement_tournament_contract(
    contract: Mapping[str, Any],
    *,
    forward_material_id: str,
    reverse_material_id: str,
    tournament_candidate_material_ids: Sequence[str],
) -> dict[str, Any]:
    """Validate a persisted contract against its choices and actual queue."""

    schema_version = contract.get("schema_version")
    if schema_version not in {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}:
        raise DisagreementTournamentContractError(
            "disagreement tournament contract has an unsupported schema"
        )
    if (
        contract.get("required") is not True
        or contract.get("reason_code") != "FORWARD_REVERSE_EXACT_MDL_DISAGREEMENT"
        or contract.get("resolution_policy") != RESOLUTION_POLICY
        or contract.get("provisional_seed_is_final_selection") is not False
        or contract.get("selected_mdl_parameters_mutable") is not False
        or contract.get("library_default_parameters_required") is not True
    ):
        raise DisagreementTournamentContractError(
            "disagreement tournament contract weakens the fail-closed policy"
        )
    forward = _text(forward_material_id, "forward_material_id")
    reverse = _text(reverse_material_id, "reverse_material_id")
    if (
        contract.get("forward_material_id") != forward
        or contract.get("reverse_material_id") != reverse
        or forward == reverse
    ):
        raise DisagreementTournamentContractError(
            "disagreement tournament contract does not match Qwen choices"
        )
    seed = _text(
        contract.get("provisional_seed_material_id"),
        "provisional_seed_material_id",
    )
    if seed not in {forward, reverse}:
        raise DisagreementTournamentContractError(
            "persisted provisional seed is not one of the disputed choices"
        )
    mvinverse_ids = _unique_texts(
        contract.get("mvinverse_exact_default_material_ids"),
        "mvinverse_exact_default_material_ids",
    )
    if schema_version == LEGACY_SCHEMA_VERSION:
        if not mvinverse_ids:
            raise DisagreementTournamentContractError(
                "legacy disagreement contracts require an MVInverse "
                "exact-default candidate"
            )
        independent_ids = list(mvinverse_ids)
    else:
        if contract.get("challenger_policy") != CHALLENGER_POLICY:
            raise DisagreementTournamentContractError(
                "disagreement tournament contract has an unsupported "
                "independent-challenger policy"
            )
        retrieval_ids = _unique_texts(
            contract.get("retrieval_exact_default_material_ids"),
            "retrieval_exact_default_material_ids",
        )
        overlap = sorted(set(mvinverse_ids) & set(retrieval_ids))
        if overlap:
            raise DisagreementTournamentContractError(
                "independent exact-default candidates have ambiguous evidence "
                f"provenance: {overlap}"
            )
        raw_independent = contract.get("independent_exact_default_candidates")
        if not isinstance(raw_independent, Sequence) or isinstance(
            raw_independent, (str, bytes)
        ):
            raise DisagreementTournamentContractError(
                "independent_exact_default_candidates must be an array"
            )
        expected_independent = [
            {
                "material_id": material_id,
                "evidence_basis": MVINVERSE_CHALLENGER_BASIS,
            }
            for material_id in mvinverse_ids
        ] + [
            {
                "material_id": material_id,
                "evidence_basis": RANKED_RETRIEVAL_CHALLENGER_BASIS,
            }
            for material_id in retrieval_ids
        ]
        if list(raw_independent) != expected_independent:
            raise DisagreementTournamentContractError(
                "independent exact-default candidate provenance is inconsistent"
            )
        independent_ids = [
            str(candidate["material_id"]) for candidate in expected_independent
        ]
        if not independent_ids:
            raise DisagreementTournamentContractError(
                "at least one independent exact library-default candidate is "
                "required"
            )
        expected_mvinverse_status = (
            "available"
            if mvinverse_ids
            else "unavailable_or_no_eligible_exact_default_candidate"
        )
        if contract.get("mvinverse_challenger_status") != expected_mvinverse_status:
            raise DisagreementTournamentContractError(
                "MVInverse challenger availability status is inconsistent"
            )
    disputed_challengers = sorted(
        set(independent_ids) & {forward, reverse}
    )
    if disputed_challengers:
        raise DisagreementTournamentContractError(
            "an independent exact-default challenger repeats a disputed Qwen "
            f"choice: {disputed_challengers}"
        )
    required_ids = _unique_texts(
        contract.get("required_candidate_material_ids"),
        "required_candidate_material_ids",
    )
    expected_required = list(dict.fromkeys([forward, reverse, *independent_ids]))
    if len(expected_required) < 3 or required_ids != expected_required:
        raise DisagreementTournamentContractError(
            "persisted required candidates are not the two Qwen choices plus "
            "the recorded independent exact-default challengers"
        )
    queued_ids = _unique_texts(
        tournament_candidate_material_ids,
        "tournament_candidate_material_ids",
    )
    missing = sorted(set(required_ids) - set(queued_ids))
    if missing:
        raise DisagreementTournamentContractError(
            f"render queue omits required disagreement candidates: {missing}"
        )
    return dict(contract)


def disagreement_is_render_confirmed(
    contract: Mapping[str, Any],
    *,
    round_audit: Mapping[str, Any],
) -> bool:
    """Return whether a rendered tournament may resolve the disagreement.

    An accepted challenger has already passed the exact-MDL selector's visual
    eligibility contract.  The provisional seed may also survive, but only if
    it itself passed every compared view and remained the best eligible
    rendered candidate.  Threshold fallback from a non-PASS seed is never a
    confirmation.
    """

    if (
        contract.get("schema_version")
        not in {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}
        or contract.get("required") is not True
    ):
        raise DisagreementTournamentContractError(
            "render confirmation requires a validated disagreement contract"
        )
    accepted = round_audit.get("accepted_candidate_id")
    if isinstance(accepted, str) and accepted:
        return True
    baseline = round_audit.get("baseline_candidate_id")
    selected = round_audit.get("selected_candidate_id")
    return (
        isinstance(baseline, str)
        and baseline
        and selected == baseline
        and round_audit.get("baseline_all_view_pass") is True
        and round_audit.get("status") == "FALLBACK_BASELINE_BEST"
    )


__all__ = [
    "DisagreementTournamentContractError",
    "CHALLENGER_POLICY",
    "LEGACY_SCHEMA_VERSION",
    "MVINVERSE_CHALLENGER_BASIS",
    "RESOLUTION_POLICY",
    "SCHEMA_VERSION",
    "RANKED_RETRIEVAL_CHALLENGER_BASIS",
    "build_disagreement_tournament_contract",
    "disagreement_is_render_confirmed",
    "validate_disagreement_tournament_contract",
]
