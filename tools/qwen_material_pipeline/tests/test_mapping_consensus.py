from __future__ import annotations

import copy

import pytest

import qwen_material_pipeline.evidence.mapping as mapping_consensus
from qwen_material_pipeline.evidence.mapping import (
    AUDIT_SCHEMA_VERSION,
    MappingConsensusError,
    apply_mapping_consensus_to_batches,
    build_view_group_id_maps,
    canonicalize_view_batch_mappings,
    validate_mapping_consensus_audit,
)


@pytest.fixture(autouse=True)
def _isolate_projection_from_full_evidence_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fusion module owns the full report; these tests exercise its join."""

    monkeypatch.setattr(
        mapping_consensus,
        "validate_mvinverse_evidence",
        lambda document: copy.deepcopy(document),
    )


def _association(
    canonical: str,
    local: str | None,
    *,
    accepted: bool = True,
    status: str = "matched",
    candidates: list[str] | None = None,
) -> dict:
    if candidates is None:
        candidates = [local] if local is not None else []
    return {
        "group_id": canonical,
        "accepted": accepted,
        "association": {
            "status": status,
            "candidate_group_ids": candidates,
            "matched_group_id": local,
        },
    }


def _evidence(views: list[dict]) -> dict:
    return {
        "schema_version": "qwen-mvinverse-pbr-evidence/v1",
        "views": views,
    }


def _mapping(
    part_id: str,
    group_id: str | None,
    *,
    view_id: str | None,
    status: str = "matched",
    confidence: float = 0.95,
    reason: str = "direct_visual_match",
) -> dict:
    return {
        "part_id": part_id,
        "group_id": group_id,
        "mapping_confidence": confidence,
        "evidence_view_id": view_id,
        "evidence_box_index": 0 if group_id is not None else None,
        "status": status,
        "reason_code": reason,
    }


def _batch(batch_id: str, *mappings: dict) -> dict:
    return {
        "schema_version": "qwen-part-palette-map/v1",
        "batch_id": batch_id,
        "mappings": list(mappings),
    }


def _vote(
    view_id: str,
    part_id: str = "P0001",
    *,
    local: str | None = "G07",
    canonical: str | None = "G01",
    status: str = "matched",
    confidence: float = 0.95,
    reason: str = "direct_visual_match",
) -> dict:
    return {
        "view_id": view_id,
        "part_id": part_id,
        "local_group_id": local,
        "canonical_group_id": canonical,
        "status": status,
        "confidence": confidence,
        "reason_code": reason,
    }


def _main_batch() -> list[dict]:
    return [
        _batch(
            "B01",
            _mapping("P0001", "G01", view_id="ref_main", confidence=0.96),
        )
    ]


def test_build_view_group_maps_uses_only_accepted_unambiguous_matches() -> None:
    evidence = _evidence(
        [
            {
                "view_id": "ref_front",
                "groups": [
                    _association("G01", "G07"),
                    _association("G02", "G08", accepted=False),
                    _association(
                        "G03",
                        None,
                        accepted=False,
                        status="ambiguous",
                        candidates=["G09", "G10"],
                    ),
                    _association(
                        "G04",
                        "G11",
                        accepted=True,
                        status="explicit_mask",
                    ),
                ],
            },
            {
                "view_id": "ref_side",
                "groups": [_association("G01", "G02")],
            },
        ]
    )

    assert build_view_group_id_maps(evidence) == {
        "ref_front": {"G07": "G01"},
        "ref_side": {"G02": "G01"},
    }


def test_build_view_group_maps_rejects_duplicate_or_ambiguous_identity() -> None:
    duplicate_view = _evidence(
        [
            {"view_id": "ref_front", "groups": []},
            {"view_id": "ref_front", "groups": []},
        ]
    )
    with pytest.raises(MappingConsensusError, match="duplicate MVInverse view_id"):
        build_view_group_id_maps(duplicate_view)

    reused_local = _evidence(
        [
            {
                "view_id": "ref_front",
                "groups": [
                    _association("G01", "G07"),
                    _association("G02", "G07"),
                ],
            }
        ]
    )
    with pytest.raises(MappingConsensusError, match="ambiguous local-to-canonical"):
        build_view_group_id_maps(reused_local)


def test_canonicalize_view_batches_preserves_qwen_vote_evidence() -> None:
    votes = canonicalize_view_batch_mappings(
        {
            "ref_front": [
                _batch(
                    "B01",
                    _mapping("P0001", "G07", view_id="ref_front"),
                    _mapping(
                        "P0002",
                        None,
                        view_id=None,
                        status="unknown",
                        confidence=0.20,
                        reason="multi_material_mesh",
                    ),
                )
            ],
            "ref_side": [
                _batch(
                    "B01",
                    _mapping(
                        "P0001",
                        "G02",
                        view_id="ref_side",
                        status="review",
                        confidence=0.75,
                        reason="partial_visibility",
                    ),
                )
            ],
        },
        {
            "ref_front": {"G07": "G01"},
            "ref_side": {"G02": "G01"},
        },
    )

    assert votes == [
        {
            "view_id": "ref_front",
            "part_id": "P0001",
            "local_group_id": "G07",
            "canonical_group_id": "G01",
            "status": "matched",
            "confidence": 0.95,
            "reason_code": "direct_visual_match",
        },
        {
            "view_id": "ref_side",
            "part_id": "P0001",
            "local_group_id": "G02",
            "canonical_group_id": "G01",
            "status": "review",
            "confidence": 0.75,
            "reason_code": "partial_visibility",
        },
        {
            "view_id": "ref_front",
            "part_id": "P0002",
            "local_group_id": None,
            "canonical_group_id": None,
            "status": "unknown",
            "confidence": 0.20,
            "reason_code": "multi_material_mesh",
        },
    ]


def test_canonicalize_rejects_duplicate_part_view_across_batches() -> None:
    row = _mapping("P0001", "G07", view_id="ref_front")
    with pytest.raises(MappingConsensusError, match="duplicate Qwen vote"):
        canonicalize_view_batch_mappings(
            {"ref_front": [_batch("B01", row), _batch("B02", row)]},
            {"ref_front": {"G07": "G01"}},
        )


def test_two_high_confidence_independent_views_keep_main_mapping() -> None:
    main = _main_batch()
    original = copy.deepcopy(main)
    report = apply_mapping_consensus_to_batches(
        main,
        [
            _vote("ref_front", confidence=0.96),
            _vote("ref_side", local="G02", confidence=0.92),
            _vote(
                "ref_top",
                local="G03",
                canonical="G02",
                confidence=0.59,
            ),
        ],
    )

    assert main == original
    assert report["gate_batches"] == original
    decision = report["audit"]["decisions"][0]
    assert decision["decision"] == "kept_auto"
    assert decision["reason_codes"] == [
        "minimum_independent_agreement_met",
        "no_conflicting_votes",
    ]
    assert decision["agreeing_view_ids"] == ["ref_front", "ref_side"]
    assert decision["conflicting_view_ids"] == []
    assert decision["unknown_view_ids"] == ["ref_top"]
    assert report["audit"]["schema_version"] == AUDIT_SCHEMA_VERSION


def test_insufficient_independent_agreement_downgrades_to_review() -> None:
    report = apply_mapping_consensus_to_batches(
        _main_batch(), [_vote("ref_front", confidence=0.99)]
    )

    row = report["gate_batches"][0]["mappings"][0]
    assert row["status"] == "review"
    assert row["group_id"] == "G01"
    assert 0.6 <= row["mapping_confidence"] < 0.9
    assert row["reason_code"] == "ambiguous"
    decision = report["audit"]["decisions"][0]
    assert decision["decision"] == "downgraded_review"
    assert decision["reason_codes"] == ["insufficient_independent_agreement"]


def test_high_confidence_unresolved_local_group_blocks_auto() -> None:
    report = apply_mapping_consensus_to_batches(
        _main_batch(),
        [
            _vote("ref_front", confidence=0.97),
            _vote("ref_side", local="G02", confidence=0.94),
            _vote(
                "ref_top",
                local="G09",
                canonical=None,
                confidence=0.93,
            ),
        ],
    )

    row = report["gate_batches"][0]["mappings"][0]
    assert row["status"] == "review"
    assert row["group_id"] == "G01"
    assert row["mapping_confidence"] < 0.9
    decision = report["audit"]["decisions"][0]
    assert decision["decision"] == "downgraded_review"
    assert decision["reason_codes"] == ["unresolved_high_confidence_group_vote"]
    assert decision["unknown_view_ids"] == ["ref_top"]


def test_medium_confidence_canonical_conflict_downgrades_to_preserve() -> None:
    report = apply_mapping_consensus_to_batches(
        _main_batch(),
        [
            _vote("ref_front", confidence=0.97),
            _vote("ref_side", local="G02", confidence=0.94),
            _vote(
                "ref_top",
                local="G03",
                canonical="G02",
                status="review",
                confidence=0.60,
                reason="ambiguous",
            ),
        ],
    )

    row = report["gate_batches"][0]["mappings"][0]
    assert row == {
        "part_id": "P0001",
        "group_id": None,
        "mapping_confidence": pytest.approx(0.599999),
        "evidence_view_id": None,
        "evidence_box_index": None,
        "status": "unknown",
        "reason_code": "ambiguous",
    }
    decision = report["audit"]["decisions"][0]
    assert decision["decision"] == "downgraded_preserve"
    assert decision["agreeing_view_ids"] == ["ref_front", "ref_side"]
    assert decision["conflicting_view_ids"] == ["ref_top"]
    assert decision["reason_codes"] == ["conflicting_canonical_group_votes"]


def test_conflicting_review_citation_is_also_downgraded_to_preserve() -> None:
    """A review row remains consumable downstream and cannot retain a conflict."""

    main = [
        _batch(
            "B01",
            _mapping(
                "P0001",
                "G01",
                view_id="ref_main",
                status="review",
                confidence=0.76,
                reason="partial_visibility",
            ),
        )
    ]
    report = apply_mapping_consensus_to_batches(
        main,
        [
            _vote(
                "ref_front",
                local="G03",
                canonical="G02",
                status="review",
                confidence=0.72,
                reason="ambiguous",
            )
        ],
    )

    row = report["gate_batches"][0]["mappings"][0]
    assert row == {
        "part_id": "P0001",
        "group_id": None,
        "mapping_confidence": pytest.approx(0.599999),
        "evidence_view_id": None,
        "evidence_box_index": None,
        "status": "unknown",
        "reason_code": "ambiguous",
    }
    decision = report["audit"]["decisions"][0]
    assert decision["main_status"] == "review"
    assert decision["decision"] == "downgraded_preserve"
    assert decision["reason_codes"] == [
        "conflicting_canonical_group_votes",
        "insufficient_independent_agreement",
    ]
    assert report["audit"]["summary"]["downgraded_preserve_count"] == 1
    assert report["audit"]["summary"]["unchanged_review_count"] == 0


def test_any_multi_material_vote_overrides_two_agreeing_views() -> None:
    report = apply_mapping_consensus_to_batches(
        _main_batch(),
        [
            _vote("ref_front", confidence=0.97),
            _vote("ref_side", local="G02", confidence=0.94),
            _vote(
                "ref_iso",
                local=None,
                canonical=None,
                status="unknown",
                confidence=0.1,
                reason="multi_material_mesh",
            ),
        ],
    )

    row = report["gate_batches"][0]["mappings"][0]
    assert row["status"] == "unknown"
    assert row["group_id"] is None
    assert row["mapping_confidence"] < 0.6
    assert row["reason_code"] == "multi_material_mesh"
    decision = report["audit"]["decisions"][0]
    assert decision["decision"] == "downgraded_preserve"
    assert decision["reason_codes"] == ["multi_material_mesh_vote"]
    assert decision["unknown_view_ids"] == ["ref_iso"]


def test_independent_views_promote_review_but_never_unknown() -> None:
    main = [
        _batch(
            "B01",
            _mapping(
                "P0001",
                "G01",
                view_id="ref_main",
                status="review",
                confidence=0.75,
                reason="partial_visibility",
            ),
            _mapping(
                "P0002",
                None,
                view_id=None,
                status="unknown",
                confidence=0.2,
                reason="occluded",
            ),
        )
    ]
    report = apply_mapping_consensus_to_batches(
        main,
        [
            _vote("ref_front", "P0001"),
            _vote("ref_side", "P0001", local="G02"),
            _vote("ref_front", "P0002"),
            _vote("ref_side", "P0002", local="G02"),
        ],
    )

    recovered = report["gate_batches"][0]["mappings"]
    assert recovered[0]["status"] == "matched"
    assert recovered[0]["group_id"] == "G01"
    assert recovered[0]["mapping_confidence"] >= 0.85
    assert recovered[1] == main[0]["mappings"][1]
    assert [item["decision"] for item in report["audit"]["decisions"]] == [
        "promoted_auto",
        "unchanged_preserve",
    ]


def test_apply_rejects_duplicate_part_view_and_duplicate_main_part() -> None:
    duplicate_vote = _vote("ref_front")
    with pytest.raises(MappingConsensusError, match="duplicate consensus vote"):
        apply_mapping_consensus_to_batches(
            _main_batch(), [duplicate_vote, duplicate_vote]
        )

    duplicate_main = [
        _batch("B01", _mapping("P0001", "G01", view_id="ref_main")),
        _batch("B02", _mapping("P0001", "G01", view_id="ref_main")),
    ]
    with pytest.raises(MappingConsensusError, match="duplicate main mapping"):
        apply_mapping_consensus_to_batches(duplicate_main, [])


def test_audit_validator_rejects_schema_drift() -> None:
    audit = apply_mapping_consensus_to_batches(
        _main_batch(), [_vote("ref_front"), _vote("ref_side", local="G02")]
    )["audit"]
    audit["decisions"][0]["extra"] = True

    with pytest.raises(MappingConsensusError, match="fields are invalid"):
        validate_mapping_consensus_audit(audit)
