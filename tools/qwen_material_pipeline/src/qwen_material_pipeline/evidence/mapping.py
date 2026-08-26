"""Fail-closed multiview consensus for part-to-palette mappings.

This module contains data-only joins.  It does not invoke Qwen, MVInverse, or
USD.  MVInverse's accepted palette associations translate the view-local group
IDs cited by independent Qwen calls into canonical group IDs.  A primary
``matched`` mapping remains eligible only when independent views agree.  A
primary review mapping may be promoted only when two independent, automatic
view-local mappings agree with its exact canonical group and no conflicting
or unresolved evidence exists.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from qwen_material_pipeline.mvinverse.evidence import (
    MVInverseEvidenceError,
    SCHEMA_VERSION as MVINVERSE_EVIDENCE_SCHEMA_VERSION,
    validate_mvinverse_evidence,
)
from qwen_material_pipeline.core.staged_analysis import (
    AUTO_THRESHOLD as BATCH_AUTO_THRESHOLD,
    BATCH_SCHEMA_VERSION,
    MAPPING_STATUSES,
    REVIEW_THRESHOLD as BATCH_REVIEW_THRESHOLD,
)


AUDIT_SCHEMA_VERSION = "qwen-mapping-consensus-audit/v1"
REVIEW_CONFIDENCE_CAP = min(BATCH_AUTO_THRESHOLD - 0.000001, 0.849999)
UNKNOWN_CONFIDENCE_CAP = BATCH_REVIEW_THRESHOLD - 0.000001

_GROUP_ID_RE = re.compile(r"G[0-9]{2,4}")
_DECISIONS = frozenset(
    {
        "kept_auto",
        "promoted_auto",
        "downgraded_review",
        "downgraded_preserve",
        "unchanged_review",
        "unchanged_preserve",
    }
)
_VOTE_FIELDS = {
    "view_id",
    "part_id",
    "local_group_id",
    "canonical_group_id",
    "status",
    "confidence",
    "reason_code",
}
_AUDIT_DECISION_FIELDS = {
    "part_id",
    "batch_id",
    "main_group_id",
    "main_status",
    "main_confidence",
    "output_group_id",
    "output_status",
    "output_confidence",
    "decision",
    "reason_codes",
    "agreeing_view_ids",
    "conflicting_view_ids",
    "unknown_view_ids",
    "vote_count",
}


class MappingConsensusError(ValueError):
    """Raised when consensus inputs are malformed, duplicated, or ambiguous."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MappingConsensusError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MappingConsensusError(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MappingConsensusError(f"{label} must be a non-empty string")
    return value.strip()


def _optional_group(value: Any, label: str) -> str | None:
    if value is None:
        return None
    group_id = _string(value, label)
    if not _GROUP_ID_RE.fullmatch(group_id):
        raise MappingConsensusError(f"{label} must be a palette group ID")
    return group_id


def _confidence(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise MappingConsensusError(f"{label} must be a finite number from 0 to 1")
    return float(value)


def _exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise MappingConsensusError(
            f"{label} fields are invalid; unexpected={sorted(actual - expected)}, "
            f"missing={sorted(expected - actual)}"
        )


def build_view_group_id_maps(
    mvinverse_evidence: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Return ``view -> local group -> canonical group`` from strict evidence.

    Only quality-accepted, unambiguous ``matched`` records participate.  An
    accepted local group may identify exactly one canonical group in a view;
    duplicate view/canonical records or a reused local group fail instead of
    being resolved by order.
    """

    try:
        document = validate_mvinverse_evidence(mvinverse_evidence)
    except MVInverseEvidenceError as exc:
        raise MappingConsensusError(f"invalid MVInverse evidence: {exc}") from exc
    if document.get("schema_version") != MVINVERSE_EVIDENCE_SCHEMA_VERSION:
        raise MappingConsensusError("unsupported MVInverse evidence schema_version")
    views = _sequence(document.get("views"), "mvinverse_evidence.views")
    result: dict[str, dict[str, str]] = {}
    for view_index, raw_view in enumerate(views):
        view = _mapping(raw_view, f"mvinverse_evidence.views[{view_index}]")
        view_id = _string(view.get("view_id"), f"views[{view_index}].view_id")
        if view_id in result:
            raise MappingConsensusError(f"duplicate MVInverse view_id: {view_id}")
        local_to_canonical: dict[str, str] = {}
        seen_canonical: set[str] = set()
        groups = _sequence(view.get("groups"), f"view {view_id}.groups")
        for group_index, raw_group in enumerate(groups):
            label = f"view {view_id}.groups[{group_index}]"
            group = _mapping(raw_group, label)
            canonical_group_id = _optional_group(
                group.get("group_id"), f"{label}.group_id"
            )
            assert canonical_group_id is not None
            if canonical_group_id in seen_canonical:
                raise MappingConsensusError(
                    f"duplicate canonical group {canonical_group_id} in view {view_id}"
                )
            seen_canonical.add(canonical_group_id)
            accepted = group.get("accepted")
            if not isinstance(accepted, bool):
                raise MappingConsensusError(f"{label}.accepted must be boolean")
            association = _mapping(group.get("association"), f"{label}.association")
            _exact_fields(
                association,
                {"status", "candidate_group_ids", "matched_group_id"},
                f"{label}.association",
            )
            status = association.get("status")
            if status not in {"matched", "ambiguous", "unmatched", "explicit_mask"}:
                raise MappingConsensusError(f"{label}.association.status is invalid")
            candidates = _sequence(
                association.get("candidate_group_ids"),
                f"{label}.association.candidate_group_ids",
            )
            candidate_ids = [
                _optional_group(value, f"{label}.association.candidate_group_ids")
                for value in candidates
            ]
            if any(value is None for value in candidate_ids):
                raise MappingConsensusError(
                    f"{label}.association candidate IDs cannot be null"
                )
            if len(candidate_ids) != len(set(candidate_ids)):
                raise MappingConsensusError(
                    f"{label}.association contains duplicate candidate group IDs"
                )
            matched_group_id = _optional_group(
                association.get("matched_group_id"),
                f"{label}.association.matched_group_id",
            )
            if status == "matched" and (
                matched_group_id is None or candidate_ids != [matched_group_id]
            ):
                raise MappingConsensusError(
                    f"{label} has an ambiguous or incomplete matched association"
                )
            if status == "ambiguous" and (
                matched_group_id is not None or len(candidate_ids) < 2
            ):
                raise MappingConsensusError(
                    f"{label} has an inconsistent ambiguous association"
                )
            if status == "unmatched" and (
                matched_group_id is not None or candidate_ids
            ):
                raise MappingConsensusError(
                    f"{label} has an inconsistent unmatched association"
                )
            if not accepted or status != "matched" or matched_group_id is None:
                continue
            previous = local_to_canonical.get(matched_group_id)
            if previous is not None:
                raise MappingConsensusError(
                    "ambiguous local-to-canonical association in view "
                    f"{view_id}: {matched_group_id} identifies both {previous} and "
                    f"{canonical_group_id}"
                )
            local_to_canonical[matched_group_id] = canonical_group_id
        result[view_id] = dict(sorted(local_to_canonical.items()))
    return dict(sorted(result.items()))


def canonicalize_view_batch_mappings(
    view_batches_by_id: Mapping[str, Sequence[Mapping[str, Any]]],
    view_group_id_maps: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Translate view-local Qwen batch mappings into canonical consensus votes."""

    batches_by_view = _mapping(view_batches_by_id, "view_batches_by_id")
    group_maps = _mapping(view_group_id_maps, "view_group_id_maps")
    normalized_maps: dict[str, dict[str, str]] = {}
    for raw_view_id, raw_map in group_maps.items():
        view_id = _string(raw_view_id, "view_group_id_maps view_id")
        local_map = _mapping(raw_map, f"view_group_id_maps[{view_id}]")
        converted: dict[str, str] = {}
        for raw_local, raw_canonical in local_map.items():
            local = _optional_group(raw_local, f"{view_id} local group")
            canonical = _optional_group(raw_canonical, f"{view_id} canonical group")
            assert local is not None and canonical is not None
            if local in converted:
                raise MappingConsensusError(
                    f"duplicate local group {local} in view group map {view_id}"
                )
            converted[local] = canonical
        normalized_maps[view_id] = converted

    votes: list[dict[str, Any]] = []
    seen_part_views: set[tuple[str, str]] = set()
    for raw_view_id, raw_batches in batches_by_view.items():
        view_id = _string(raw_view_id, "view_batches_by_id view_id")
        if view_id not in normalized_maps:
            raise MappingConsensusError(
                f"Qwen batches reference a view without an accepted group map: {view_id}"
            )
        batches = _sequence(raw_batches, f"view_batches_by_id[{view_id}]")
        for batch_index, raw_batch in enumerate(batches):
            batch = _mapping(raw_batch, f"{view_id}.batches[{batch_index}]")
            mappings = _sequence(
                batch.get("mappings"), f"{view_id}.batches[{batch_index}].mappings"
            )
            for mapping_index, raw_row in enumerate(mappings):
                label = f"{view_id}.batches[{batch_index}].mappings[{mapping_index}]"
                row = _mapping(raw_row, label)
                part_id = _string(row.get("part_id"), f"{label}.part_id")
                identity = (view_id, part_id)
                if identity in seen_part_views:
                    raise MappingConsensusError(
                        f"duplicate Qwen vote for part/view: {part_id}/{view_id}"
                    )
                seen_part_views.add(identity)
                status = row.get("status")
                if status not in MAPPING_STATUSES:
                    raise MappingConsensusError(f"{label}.status is invalid")
                confidence = _confidence(
                    row.get("mapping_confidence"), f"{label}.mapping_confidence"
                )
                reason_code = _string(row.get("reason_code"), f"{label}.reason_code")
                local_group_id = _optional_group(
                    row.get("group_id"), f"{label}.group_id"
                )
                if status == "unknown" and local_group_id is not None:
                    raise MappingConsensusError(
                        f"{label}.unknown mapping must not cite a local group"
                    )
                if status != "unknown" and local_group_id is None:
                    raise MappingConsensusError(
                        f"{label}.{status} mapping must cite a local group"
                    )
                if status != "unknown" and "evidence_view_id" in row:
                    evidence_view_id = row.get("evidence_view_id")
                    if evidence_view_id != view_id:
                        raise MappingConsensusError(
                            f"{label}.evidence_view_id must equal {view_id!r}"
                        )
                canonical_group_id = (
                    normalized_maps[view_id].get(local_group_id)
                    if local_group_id is not None
                    else None
                )
                votes.append(
                    {
                        "view_id": view_id,
                        "part_id": part_id,
                        "local_group_id": local_group_id,
                        "canonical_group_id": canonical_group_id,
                        "status": status,
                        "confidence": confidence,
                        "reason_code": reason_code,
                    }
                )
    return sorted(votes, key=lambda item: (item["part_id"], item["view_id"]))


def _validated_votes(votes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values = _sequence(votes, "votes")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw_vote in enumerate(values):
        label = f"votes[{index}]"
        vote = _mapping(raw_vote, label)
        _exact_fields(vote, _VOTE_FIELDS, label)
        view_id = _string(vote["view_id"], f"{label}.view_id")
        part_id = _string(vote["part_id"], f"{label}.part_id")
        identity = (view_id, part_id)
        if identity in seen:
            raise MappingConsensusError(
                f"duplicate consensus vote for part/view: {part_id}/{view_id}"
            )
        seen.add(identity)
        status = vote["status"]
        if status not in MAPPING_STATUSES:
            raise MappingConsensusError(f"{label}.status is invalid")
        local_group_id = _optional_group(
            vote["local_group_id"], f"{label}.local_group_id"
        )
        canonical_group_id = _optional_group(
            vote["canonical_group_id"], f"{label}.canonical_group_id"
        )
        if status == "unknown" and (
            local_group_id is not None or canonical_group_id is not None
        ):
            raise MappingConsensusError(
                f"{label}.unknown vote must use null local/canonical groups"
            )
        if status != "unknown" and local_group_id is None:
            raise MappingConsensusError(
                f"{label}.{status} vote must cite a local group"
            )
        result.append(
            {
                "view_id": view_id,
                "part_id": part_id,
                "local_group_id": local_group_id,
                "canonical_group_id": canonical_group_id,
                "status": status,
                "confidence": _confidence(vote["confidence"], f"{label}.confidence"),
                "reason_code": _string(vote["reason_code"], f"{label}.reason_code"),
            }
        )
    return sorted(result, key=lambda item: (item["part_id"], item["view_id"]))


def _validate_policy(
    minimum_agreeing_views: int,
    auto_conf: float,
    conflict_conf: float,
) -> tuple[int, float, float]:
    if (
        isinstance(minimum_agreeing_views, bool)
        or not isinstance(minimum_agreeing_views, int)
        or minimum_agreeing_views < 2
    ):
        raise MappingConsensusError("minimum_agreeing_views must be an integer >= 2")
    auto = _confidence(auto_conf, "auto_conf")
    conflict = _confidence(conflict_conf, "conflict_conf")
    if auto < BATCH_REVIEW_THRESHOLD + 0.000001:
        raise MappingConsensusError(
            "auto_conf must leave room above the batch review threshold "
            f"{BATCH_REVIEW_THRESHOLD}"
        )
    if conflict >= auto:
        raise MappingConsensusError("conflict_conf must be lower than auto_conf")
    return minimum_agreeing_views, auto, conflict


def _main_batches(
    main_batches: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[str, dict[str, Any]]], set[str]]:
    batches = copy.deepcopy(list(_sequence(main_batches, "main_batches")))
    rows: list[tuple[str, dict[str, Any]]] = []
    seen_parts: set[str] = set()
    seen_batches: set[str] = set()
    for batch_index, raw_batch in enumerate(batches):
        label = f"main_batches[{batch_index}]"
        batch = _mapping(raw_batch, label)
        _exact_fields(batch, {"schema_version", "batch_id", "mappings"}, label)
        if batch["schema_version"] != BATCH_SCHEMA_VERSION:
            raise MappingConsensusError(f"{label}.schema_version is unsupported")
        batch_id = _string(batch["batch_id"], f"{label}.batch_id")
        if batch_id in seen_batches:
            raise MappingConsensusError(f"duplicate main batch_id: {batch_id}")
        seen_batches.add(batch_id)
        mappings = _sequence(batch["mappings"], f"{label}.mappings")
        for mapping_index, raw_row in enumerate(mappings):
            row_label = f"{label}.mappings[{mapping_index}]"
            row = _mapping(raw_row, row_label)
            part_id = _string(row.get("part_id"), f"{row_label}.part_id")
            if part_id in seen_parts:
                raise MappingConsensusError(
                    f"duplicate main mapping for part_id: {part_id}"
                )
            seen_parts.add(part_id)
            status = row.get("status")
            if status not in MAPPING_STATUSES:
                raise MappingConsensusError(f"{row_label}.status is invalid")
            group_id = _optional_group(row.get("group_id"), f"{row_label}.group_id")
            if status == "unknown" and group_id is not None:
                raise MappingConsensusError(
                    f"{row_label}.unknown mapping must use a null group"
                )
            if status != "unknown" and group_id is None:
                raise MappingConsensusError(
                    f"{row_label}.{status} mapping must cite a group"
                )
            _confidence(
                row.get("mapping_confidence"), f"{row_label}.mapping_confidence"
            )
            _string(row.get("reason_code"), f"{row_label}.reason_code")
            rows.append((batch_id, row))
    return batches, rows, seen_parts


def apply_mapping_consensus_to_batches(
    main_batches: Sequence[Mapping[str, Any]],
    votes: Sequence[Mapping[str, Any]],
    minimum_agreeing_views: int = 2,
    auto_conf: float = BATCH_AUTO_THRESHOLD,
    conflict_conf: float = 0.6,
) -> dict[str, Any]:
    """Gate primary mappings with independent, canonicalized per-view votes.

    The returned batches are detached from ``main_batches``.  A review mapping
    can be promoted only by independent high-confidence agreement for its
    already-cited canonical group.  Unknown mappings are never promoted.  A
    conflict or any ``multi_material_mesh`` vote clears every positive group
    citation, including a review citation, and preserves the part as unknown.
    Keeping a contradicted review citation is unsafe in an unattended run:
    later face/material recovery stages are allowed to consume review rows and
    could otherwise amplify a dominant primary-view mistake.
    """

    minimum_views, auto_threshold, conflict_threshold = _validate_policy(
        minimum_agreeing_views, auto_conf, conflict_conf
    )
    gate_batches, main_rows, part_ids = _main_batches(main_batches)
    canonical_votes = _validated_votes(votes)
    unexpected = sorted({vote["part_id"] for vote in canonical_votes} - part_ids)
    if unexpected:
        raise MappingConsensusError(
            f"votes reference parts absent from main_batches: {unexpected}"
        )
    votes_by_part: dict[str, list[dict[str, Any]]] = {
        part_id: [] for part_id in part_ids
    }
    for vote in canonical_votes:
        votes_by_part[vote["part_id"]].append(vote)

    decisions: list[dict[str, Any]] = []
    for batch_id, row in main_rows:
        part_id = str(row["part_id"])
        main_status = str(row["status"])
        main_group_id = row.get("group_id")
        main_confidence = float(row["mapping_confidence"])
        part_votes = votes_by_part[part_id]
        agreeing = [
            vote
            for vote in part_votes
            if vote["status"] == "matched"
            and vote["canonical_group_id"] == main_group_id
            and vote["confidence"] >= auto_threshold
        ]
        conflicting = [
            vote
            for vote in part_votes
            if vote["status"] in {"matched", "review"}
            and vote["canonical_group_id"] is not None
            and vote["canonical_group_id"] != main_group_id
            and vote["confidence"] >= conflict_threshold
        ]
        unresolved = [
            vote
            for vote in part_votes
            if vote["status"] in {"matched", "review"}
            and vote["local_group_id"] is not None
            and vote["canonical_group_id"] is None
            and vote["confidence"] >= conflict_threshold
        ]
        agreeing_ids = sorted(vote["view_id"] for vote in agreeing)
        conflicting_ids = sorted(vote["view_id"] for vote in conflicting)
        classified_ids = set(agreeing_ids) | set(conflicting_ids)
        unknown_ids = sorted(
            vote["view_id"]
            for vote in part_votes
            if vote["view_id"] not in classified_ids
        )
        multi_material = any(
            vote["reason_code"] == "multi_material_mesh" for vote in part_votes
        )
        reason_codes: list[str] = []

        if (
            main_status == "review"
            and not multi_material
            and not conflicting
            and not unresolved
            and len(agreeing_ids) >= minimum_views
        ):
            decision = "promoted_auto"
            reason_codes.extend(
                [
                    "independent_multiview_review_recovery",
                    "minimum_independent_agreement_met",
                    "no_conflicting_votes",
                ]
            )
            row["mapping_confidence"] = min(
                vote["confidence"] for vote in agreeing
            )
            row["status"] = "matched"
            row["reason_code"] = "direct_visual_match"
        elif main_status in {"matched", "review"} and (
            multi_material or conflicting
        ):
            decision = "downgraded_preserve"
            if multi_material:
                reason_codes.append("multi_material_mesh_vote")
            if conflicting:
                reason_codes.append("conflicting_canonical_group_votes")
            if len(agreeing_ids) < minimum_views:
                reason_codes.append("insufficient_independent_agreement")
            row["group_id"] = None
            row["mapping_confidence"] = min(
                main_confidence,
                UNKNOWN_CONFIDENCE_CAP,
                max(0.0, conflict_threshold - 0.000001),
            )
            row["evidence_view_id"] = None
            row["evidence_box_index"] = None
            row["status"] = "unknown"
            row["reason_code"] = (
                "multi_material_mesh" if multi_material else "ambiguous"
            )
        elif main_status != "matched":
            decision = (
                "unchanged_review" if main_status == "review" else "unchanged_preserve"
            )
            reason_codes.append("main_mapping_not_matched")
        elif unresolved:
            decision = "downgraded_review"
            reason_codes.append("unresolved_high_confidence_group_vote")
            row["mapping_confidence"] = min(
                main_confidence,
                REVIEW_CONFIDENCE_CAP,
                auto_threshold - 0.000001,
            )
            row["status"] = "review"
            row["reason_code"] = "ambiguous"
        elif len(agreeing_ids) < minimum_views:
            decision = "downgraded_review"
            reason_codes.append("insufficient_independent_agreement")
            if not part_votes:
                reason_codes.append("no_independent_votes")
            row["mapping_confidence"] = min(
                main_confidence,
                REVIEW_CONFIDENCE_CAP,
                auto_threshold - 0.000001,
            )
            row["status"] = "review"
            row["reason_code"] = "ambiguous"
        else:
            decision = "kept_auto"
            reason_codes.extend(
                [
                    "minimum_independent_agreement_met",
                    "no_conflicting_votes",
                ]
            )

        decisions.append(
            {
                "part_id": part_id,
                "batch_id": batch_id,
                "main_group_id": main_group_id,
                "main_status": main_status,
                "main_confidence": main_confidence,
                "output_group_id": row.get("group_id"),
                "output_status": row["status"],
                "output_confidence": float(row["mapping_confidence"]),
                "decision": decision,
                "reason_codes": reason_codes,
                "agreeing_view_ids": agreeing_ids,
                "conflicting_view_ids": conflicting_ids,
                "unknown_view_ids": unknown_ids,
                "vote_count": len(part_votes),
            }
        )

    counts = {decision: 0 for decision in _DECISIONS}
    for record in decisions:
        counts[record["decision"]] += 1
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "policy": {
            "minimum_agreeing_views": minimum_views,
            "auto_confidence_threshold": auto_threshold,
            "conflict_confidence_threshold": conflict_threshold,
        },
        "decisions": decisions,
        "summary": {
            "part_count": len(decisions),
            "vote_count": len(canonical_votes),
            "kept_auto_count": counts["kept_auto"],
            "promoted_auto_count": counts["promoted_auto"],
            "downgraded_review_count": counts["downgraded_review"],
            "downgraded_preserve_count": counts["downgraded_preserve"],
            "unchanged_review_count": counts["unchanged_review"],
            "unchanged_preserve_count": counts["unchanged_preserve"],
            "fail_closed": True,
        },
    }
    return {
        "gate_batches": gate_batches,
        "audit": validate_mapping_consensus_audit(audit),
    }


def validate_mapping_consensus_audit(
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact persisted consensus audit contract."""

    document = _mapping(audit, "audit")
    _exact_fields(
        document, {"schema_version", "policy", "decisions", "summary"}, "audit"
    )
    if document["schema_version"] != AUDIT_SCHEMA_VERSION:
        raise MappingConsensusError("audit.schema_version is unsupported")
    policy = _mapping(document["policy"], "audit.policy")
    _exact_fields(
        policy,
        {
            "minimum_agreeing_views",
            "auto_confidence_threshold",
            "conflict_confidence_threshold",
        },
        "audit.policy",
    )
    _validate_policy(
        policy["minimum_agreeing_views"],
        policy["auto_confidence_threshold"],
        policy["conflict_confidence_threshold"],
    )
    raw_decisions = _sequence(document["decisions"], "audit.decisions")
    seen_parts: set[str] = set()
    counts = {decision: 0 for decision in _DECISIONS}
    normalized_decisions: list[dict[str, Any]] = []
    for index, raw_decision in enumerate(raw_decisions):
        label = f"audit.decisions[{index}]"
        decision = _mapping(raw_decision, label)
        _exact_fields(decision, _AUDIT_DECISION_FIELDS, label)
        part_id = _string(decision["part_id"], f"{label}.part_id")
        if part_id in seen_parts:
            raise MappingConsensusError(f"audit has duplicate part_id: {part_id}")
        seen_parts.add(part_id)
        _string(decision["batch_id"], f"{label}.batch_id")
        main_status = decision["main_status"]
        output_status = decision["output_status"]
        if main_status not in MAPPING_STATUSES or output_status not in MAPPING_STATUSES:
            raise MappingConsensusError(f"{label} contains an invalid mapping status")
        _optional_group(decision["main_group_id"], f"{label}.main_group_id")
        _optional_group(decision["output_group_id"], f"{label}.output_group_id")
        _confidence(decision["main_confidence"], f"{label}.main_confidence")
        _confidence(decision["output_confidence"], f"{label}.output_confidence")
        decision_name = decision["decision"]
        if decision_name not in _DECISIONS:
            raise MappingConsensusError(f"{label}.decision is invalid")
        counts[decision_name] += 1
        for field in (
            "reason_codes",
            "agreeing_view_ids",
            "conflicting_view_ids",
            "unknown_view_ids",
        ):
            values = _sequence(decision[field], f"{label}.{field}")
            strings = [_string(value, f"{label}.{field}") for value in values]
            if len(strings) != len(set(strings)):
                raise MappingConsensusError(f"{label}.{field} contains duplicates")
            if field != "reason_codes" and strings != sorted(strings):
                raise MappingConsensusError(f"{label}.{field} must be sorted")
            if field == "reason_codes" and not strings:
                raise MappingConsensusError(f"{label}.reason_codes cannot be empty")
        view_sets = [
            set(decision["agreeing_view_ids"]),
            set(decision["conflicting_view_ids"]),
            set(decision["unknown_view_ids"]),
        ]
        if any(
            view_sets[left] & view_sets[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise MappingConsensusError(f"{label} places a view in multiple outcomes")
        vote_count = decision["vote_count"]
        if (
            isinstance(vote_count, bool)
            or not isinstance(vote_count, int)
            or vote_count < 0
        ):
            raise MappingConsensusError(f"{label}.vote_count must be non-negative")
        if vote_count != sum(len(values) for values in view_sets):
            raise MappingConsensusError(f"{label}.vote_count is inconsistent")

        main_group = decision["main_group_id"]
        output_group = decision["output_group_id"]
        output_confidence = float(decision["output_confidence"])
        minimum_views = int(policy["minimum_agreeing_views"])
        if main_status == "unknown" and main_group is not None:
            raise MappingConsensusError(f"{label}.unknown main mapping cites a group")
        if main_status != "unknown" and main_group is None:
            raise MappingConsensusError(
                f"{label}.{main_status} main mapping lacks a group"
            )
        if output_status == "unknown" and output_group is not None:
            raise MappingConsensusError(f"{label}.unknown output mapping cites a group")
        if output_status != "unknown" and output_group is None:
            raise MappingConsensusError(
                f"{label}.{output_status} output mapping lacks a group"
            )
        if decision_name == "kept_auto" and not (
            main_status == output_status == "matched"
            and main_group == output_group
            and len(decision["agreeing_view_ids"]) >= minimum_views
            and not decision["conflicting_view_ids"]
        ):
            raise MappingConsensusError(f"{label}.kept_auto outcome is inconsistent")
        if decision_name == "promoted_auto" and not (
            main_status == "review"
            and output_status == "matched"
            and main_group == output_group
            and output_confidence >= BATCH_AUTO_THRESHOLD
            and len(decision["agreeing_view_ids"]) >= minimum_views
            and not decision["conflicting_view_ids"]
            and "independent_multiview_review_recovery"
            in decision["reason_codes"]
        ):
            raise MappingConsensusError(
                f"{label}.promoted_auto outcome is inconsistent"
            )
        if decision_name == "downgraded_review" and not (
            main_status == "matched"
            and output_status == "review"
            and main_group == output_group
            and BATCH_REVIEW_THRESHOLD
            <= output_confidence
            < min(BATCH_AUTO_THRESHOLD, float(policy["auto_confidence_threshold"]))
        ):
            raise MappingConsensusError(
                f"{label}.downgraded_review outcome is inconsistent"
            )
        if decision_name == "downgraded_preserve" and not (
            main_status in {"matched", "review"}
            and output_status == "unknown"
            and output_group is None
            and output_confidence < BATCH_REVIEW_THRESHOLD
            and (
                "multi_material_mesh_vote" in decision["reason_codes"]
                or "conflicting_canonical_group_votes" in decision["reason_codes"]
            )
        ):
            raise MappingConsensusError(
                f"{label}.downgraded_preserve outcome is inconsistent"
            )
        if decision_name == "unchanged_review" and not (
            main_status == output_status == "review" and main_group == output_group
        ):
            raise MappingConsensusError(
                f"{label}.unchanged_review outcome is inconsistent"
            )
        if decision_name == "unchanged_preserve" and not (
            main_status == output_status == "unknown"
            and main_group is None
            and output_group is None
        ):
            raise MappingConsensusError(
                f"{label}.unchanged_preserve outcome is inconsistent"
            )
        normalized_decisions.append(copy.deepcopy(dict(decision)))

    summary = _mapping(document["summary"], "audit.summary")
    expected_summary = {
        "part_count",
        "vote_count",
        "kept_auto_count",
        "promoted_auto_count",
        "downgraded_review_count",
        "downgraded_preserve_count",
        "unchanged_review_count",
        "unchanged_preserve_count",
        "fail_closed",
    }
    _exact_fields(summary, expected_summary, "audit.summary")
    expected_values = {
        "part_count": len(normalized_decisions),
        "vote_count": sum(record["vote_count"] for record in normalized_decisions),
        "kept_auto_count": counts["kept_auto"],
        "promoted_auto_count": counts["promoted_auto"],
        "downgraded_review_count": counts["downgraded_review"],
        "downgraded_preserve_count": counts["downgraded_preserve"],
        "unchanged_review_count": counts["unchanged_review"],
        "unchanged_preserve_count": counts["unchanged_preserve"],
        "fail_closed": True,
    }
    if dict(summary) != expected_values:
        raise MappingConsensusError("audit.summary is inconsistent")
    return copy.deepcopy(dict(document))
