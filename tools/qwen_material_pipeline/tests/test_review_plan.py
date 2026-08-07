from __future__ import annotations

import pytest

from qwen_material_pipeline.materials.review import (
    REVIEW_SCHEMA_VERSION,
    _review_face_subsets,
    _whitelist_ids,
    resolve_review_decisions,
)


def _staged() -> dict:
    return {
        "material_plan": {
            "schema_version": "1.0",
            "assignments": [
                {
                    "part_id": "P0001",
                    "material_id": "MAT_GREEN",
                    "semantic": "green body",
                    "confidence": 0.8,
                    "evidence_views": ["ref_front"],
                    "status": "review",
                }
            ],
        },
        "unknown_parts": [{"part_id": "P0002", "reason_code": "too_small"}],
    }


def test_review_requires_exact_source_hash() -> None:
    review = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "source_result_sha256": "wrong",
        "decisions": [],
    }
    with pytest.raises(ValueError, match="does not match"):
        resolve_review_decisions(
            _staged(),
            review,
            source_result_sha256="expected",
            allowed_material_ids={"MAT_GREEN"},
        )


def test_complete_review_approves_and_preserves_without_mutating_source() -> None:
    staged = _staged()
    review = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "source_result_sha256": "digest",
        "decisions": [
            {"part_id": "P0001", "decision": "approve", "note": "looks right"},
            {"part_id": "P0002", "decision": "preserve_existing"},
        ],
    }
    plan, report = resolve_review_decisions(
        staged,
        review,
        source_result_sha256="digest",
        allowed_material_ids={"MAT_GREEN"},
    )
    assert plan["assignments"][0]["status"] == "approved"
    assert report["approved_count"] == 1
    assert report["preserve_existing_count"] == 1
    assert report["complete"] is True
    assert staged["material_plan"]["assignments"][0]["status"] == "review"


def test_override_unknown_part_supports_whitelisted_parameters() -> None:
    review = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "source_result_sha256": "digest",
        "decisions": [
            {"part_id": "P0001", "decision": "preserve_existing"},
            {
                "part_id": "P0002",
                "decision": "override",
                "material_id": "MAT_GREEN",
                "parameters": {"paint_color": [0.02, 0.2, 0.04]},
            },
        ],
    }
    plan, _report = resolve_review_decisions(
        _staged(),
        review,
        source_result_sha256="digest",
        allowed_material_ids={"MAT_GREEN"},
    )
    assert plan["assignments"][0]["part_id"] == "P0002"
    assert plan["assignments"][0]["parameters"]["paint_color"] == [0.02, 0.2, 0.04]


def test_review_transmits_valid_human_face_subsets() -> None:
    staged = _staged()
    review = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "source_result_sha256": "digest",
        "decisions": [
            {
                "part_id": "P0001",
                "decision": "approve",
                "face_subsets": [
                    {
                        "subset_name": "black_controls",
                        "material_id": "MAT_BLACK",
                        "semantic": "black controls",
                        "face_indices": [7, 2, 9],
                    }
                ],
            },
            {"part_id": "P0002", "decision": "preserve_existing"},
        ],
    }
    plan, report = resolve_review_decisions(
        staged,
        review,
        source_result_sha256="digest",
        allowed_material_ids={"MAT_GREEN", "MAT_BLACK"},
    )
    assert plan["assignments"][0]["face_subsets"] == [
        {
            "subset_name": "black_controls",
            "material_id": "MAT_BLACK",
            "semantic": "black controls",
            "face_indices": [7, 2, 9],
        }
    ]
    assert report["face_subset_count"] == 1
    assert report["face_subset_parts"] == ["P0001"]
    assert "face_subsets" not in staged["material_plan"]["assignments"][0]


def test_review_transmits_subset_only_parent_binding_policy() -> None:
    review = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "source_result_sha256": "digest",
        "decisions": [
            {
                "part_id": "P0001",
                "decision": "approve",
                "preserve_parent_material_binding": True,
                "face_subsets": [
                    {
                        "subset_name": "green_panel",
                        "material_id": "MAT_GREEN",
                        "face_indices": [1, 2],
                    }
                ],
            },
            {"part_id": "P0002", "decision": "preserve_existing"},
        ],
    }
    plan, _report = resolve_review_decisions(
        _staged(),
        review,
        source_result_sha256="digest",
        allowed_material_ids={"MAT_GREEN"},
    )
    assignment = plan["assignments"][0]
    assert assignment["material_id"] == "MAT_GREEN"
    assert assignment["preserve_parent_material_binding"] is True


@pytest.mark.parametrize("value", [1, "true", None])
def test_review_subset_only_parent_binding_policy_is_strict(value: object) -> None:
    review = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "source_result_sha256": "digest",
        "decisions": [
            {
                "part_id": "P0001",
                "decision": "approve",
                "preserve_parent_material_binding": value,
                "face_subsets": [
                    {
                        "subset_name": "green_panel",
                        "material_id": "MAT_GREEN",
                        "face_indices": [1],
                    }
                ],
            },
            {"part_id": "P0002", "decision": "preserve_existing"},
        ],
    }
    with pytest.raises(ValueError, match="must be a boolean"):
        resolve_review_decisions(
            _staged(),
            review,
            source_result_sha256="digest",
            allowed_material_ids={"MAT_GREEN"},
        )


def test_review_subset_only_parent_binding_requires_subsets() -> None:
    review = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "source_result_sha256": "digest",
        "decisions": [
            {
                "part_id": "P0001",
                "decision": "approve",
                "preserve_parent_material_binding": True,
            },
            {"part_id": "P0002", "decision": "preserve_existing"},
        ],
    }
    with pytest.raises(ValueError, match="requires face_subsets"):
        resolve_review_decisions(
            _staged(),
            review,
            source_result_sha256="digest",
            allowed_material_ids={"MAT_GREEN"},
        )


def test_review_expands_compact_inclusive_face_ranges() -> None:
    value = _review_face_subsets(
        [
            {
                "subset_name": "metal_rod",
                "material_id": "MAT_METAL",
                "face_ranges": [[4, 6], [10, 10]],
            }
        ],
        part_id="P0002",
        allowed_material_ids={"MAT_METAL"},
    )

    assert value == [
        {
            "subset_name": "metal_rod",
            "material_id": "MAT_METAL",
            "face_indices": [4, 5, 6, 10],
        }
    ]


@pytest.mark.parametrize(
    ("face_subsets", "message"),
    [
        ([], "non-empty list"),
        (
            [
                {
                    "subset_name": "bad/name",
                    "material_id": "MAT_BLACK",
                    "face_indices": [0],
                }
            ],
            "unsafe subset_name",
        ),
        (
            [
                {
                    "subset_name": "black",
                    "material_id": "NOT_ALLOWED",
                    "face_indices": [0],
                }
            ],
            "unknown material_id",
        ),
        (
            [
                {
                    "subset_name": "black",
                    "material_id": "MAT_BLACK",
                    "face_indices": [0, 0],
                }
            ],
            "must be unique",
        ),
        (
            [
                {
                    "subset_name": "black",
                    "material_id": "MAT_BLACK",
                    "face_indices": [-1],
                }
            ],
            "non-negative",
        ),
        (
            [
                {
                    "subset_name": "black",
                    "material_id": "MAT_BLACK",
                    "face_indices": [True],
                }
            ],
            "only integers",
        ),
        (
            [
                {
                    "subset_name": "black",
                    "material_id": "MAT_BLACK",
                    "face_indices": [0],
                    "face_ranges": [[1, 2]],
                }
            ],
            "exactly one",
        ),
        (
            [
                {
                    "subset_name": "black",
                    "material_id": "MAT_BLACK",
                    "face_ranges": [[3, 1]],
                }
            ],
            "end must be >= start",
        ),
        (
            [
                {
                    "subset_name": "black",
                    "material_id": "MAT_BLACK",
                    "face_ranges": [[0, 2], [2, 3]],
                }
            ],
            "unique faces",
        ),
    ],
)
def test_review_face_subsets_fail_closed(face_subsets: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _review_face_subsets(
            face_subsets,
            part_id="P0024",
            allowed_material_ids={"MAT_BLACK"},
        )


def test_review_face_subsets_require_unique_names() -> None:
    record = {
        "subset_name": "black",
        "material_id": "MAT_BLACK",
        "face_indices": [0],
    }
    with pytest.raises(ValueError, match="Duplicate subset_name"):
        _review_face_subsets(
            [record, {**record, "face_indices": [1]}],
            part_id="P0024",
            allowed_material_ids={"MAT_BLACK"},
        )


def test_preserve_existing_cannot_silently_ignore_face_subsets() -> None:
    review = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "source_result_sha256": "digest",
        "decisions": [
            {"part_id": "P0001", "decision": "approve"},
            {
                "part_id": "P0002",
                "decision": "preserve_existing",
                "face_subsets": [],
            },
        ],
    }
    with pytest.raises(ValueError, match="not allowed with preserve_existing"):
        resolve_review_decisions(
            _staged(),
            review,
            source_result_sha256="digest",
            allowed_material_ids={"MAT_GREEN"},
        )


def test_whitelist_is_bounded_by_catalog() -> None:
    value = _whitelist_ids(
        {"schema_version": 1, "material_ids": ["MAT_GREEN", "MAT_BLACK"]},
        {"MAT_GREEN", "MAT_BLACK", "MAT_COPPER"},
    )
    assert value == {"MAT_GREEN", "MAT_BLACK"}


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"schema_version": 2, "material_ids": ["MAT_GREEN"]}, "schema_version"),
        ({"schema_version": 1, "material_ids": []}, "non-empty"),
        (
            {"schema_version": 1, "material_ids": ["MAT_GREEN", "MAT_GREEN"]},
            "duplicate",
        ),
        (
            {"schema_version": 1, "material_ids": ["NOT_IN_CATALOG"]},
            "absent from catalog",
        ),
    ],
)
def test_whitelist_structure_and_ids_are_strict(document: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _whitelist_ids(document, {"MAT_GREEN"})


def test_complete_review_rejects_missing_parts() -> None:
    review = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "source_result_sha256": "digest",
        "decisions": [{"part_id": "P0001", "decision": "approve"}],
    }
    with pytest.raises(ValueError, match="missing=.*P0002"):
        resolve_review_decisions(
            _staged(),
            review,
            source_result_sha256="digest",
            allowed_material_ids={"MAT_GREEN"},
        )


def test_staged_result_must_exactly_cover_registry() -> None:
    review = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "source_result_sha256": "digest",
        "decisions": [],
    }
    with pytest.raises(ValueError, match="exactly cover the registry"):
        resolve_review_decisions(
            _staged(),
            review,
            source_result_sha256="digest",
            allowed_material_ids={"MAT_GREEN"},
            expected_part_ids={"P0001", "P0002", "P0003"},
        )
