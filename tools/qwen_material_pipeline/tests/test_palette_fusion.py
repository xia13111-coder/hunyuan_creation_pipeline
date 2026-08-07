from __future__ import annotations

import pytest

from qwen_material_pipeline.evidence.palette_fusion import (
    PaletteFusionError,
    UNRESOLVED_PIXEL_LIGHT_NEUTRAL_ASSOCIATION,
    UNRESOLVED_PIXEL_MASKED_DARK_ASSOCIATION,
    fuse_multiview_palettes,
    is_verified_unresolved_pixel_light_neutral_group,
    is_verified_unresolved_pixel_masked_dark_group,
)


def _group(
    group_id: str,
    *,
    family: str,
    color: str,
    finish: str,
    box: list[int],
    confidence: float = 0.9,
) -> dict:
    return {
        "group_id": group_id,
        "family_hint": family,
        "base_color": color,
        "finish_hint": finish,
        "visual_description": f"{finish} {color} {family}",
        "boxes": [box],
        "confidence": confidence,
    }


def _palette(view_id: str, groups: list[dict]) -> dict:
    return {
        "schema_version": "qwen-material-palette/v1",
        "source_view_id": view_id,
        "groups": groups,
    }


def _accent_audit(
    view_id: str,
    group: dict,
    *,
    image_sha256: str,
    mask_sha256: str | None = None,
) -> dict:
    audit = {
        "schema_version": "qwen-palette-accent-augmentation/v1",
        "source_view_id": view_id,
        "image_sha256": image_sha256,
        "added_group_ids": [group["group_id"]],
        "components": [
            {
                "base_color": group["base_color"],
                "decision": "added",
                "accepted_components": [
                    {"box": box} for box in group["boxes"]
                ],
            }
        ],
    }
    if mask_sha256 is not None:
        audit.update(
            {
                "mask": f"/evidence/{view_id}.png",
                "mask_sha256": mask_sha256,
                "masked_dark_recovery_enabled": True,
            }
        )
    return audit


def _group_by_signature(fused: dict, family: str, color: str, finish: str) -> dict:
    return next(
        group
        for group in fused["canonical_palette"]["groups"]
        if (
            group["family_hint"],
            group["base_color"],
            group["finish_hint"],
        )
        == (family, color, finish)
    )


def test_fusion_unions_all_views_and_preserves_original_citations() -> None:
    front = _palette(
        "ref_front",
        [
            _group(
                "G01",
                family="metal",
                color="green",
                finish="painted",
                box=[100, 200, 300, 400],
                confidence=0.91,
            ),
            _group(
                "G02",
                family="metal",
                color="white",
                finish="painted",
                box=[500, 100, 650, 300],
                confidence=0.88,
            ),
        ],
    )
    side = _palette(
        "ref_side",
        [
            _group(
                "G09",
                family="metal",
                color="green",
                finish="painted",
                box=[80, 220, 260, 440],
                confidence=0.96,
            ),
            _group(
                "G10",
                family="metal",
                color="orange",
                finish="painted",
                box=[600, 300, 720, 580],
                confidence=0.85,
            ),
        ],
    )
    top = _palette(
        "ref_top",
        [
            _group(
                "G04",
                family="plastic",
                color="cyan",
                finish="painted",
                box=[200, 200, 260, 280],
                confidence=0.87,
            )
        ],
    )
    iso = _palette(
        "ref_iso",
        [
            _group(
                "G03",
                family="metal",
                color="brown",
                finish="painted",
                box=[400, 350, 520, 610],
                confidence=0.93,
            ),
            _group(
                "G06",
                family="plastic",
                color="blue",
                finish="painted",
                box=[150, 180, 220, 260],
                confidence=0.94,
            ),
        ],
    )

    fused = fuse_multiview_palettes([front, side, top, iso])

    assert fused == fuse_multiview_palettes([iso, top, side, front])
    assert fused["summary"] == {
        "input_view_count": 4,
        "input_group_count": 7,
        "canonical_group_count": 4,
        "multiview_group_count": 3,
        "singleton_group_count": 1,
        "winner_view_selected": False,
        "source_boxes_preserved": True,
    }

    green = _group_by_signature(fused, "metal", "green", "painted")
    assert green["source_view_ids"] == ["ref_front", "ref_side"]
    assert green["representative_ref"] == {
        "view_id": "ref_side",
        "local_group_id": "G09",
    }
    assert [
        (source["view_id"], source["local_group_id"], source["boxes"])
        for source in green["sources"]
    ] == [
        ("ref_front", "G01", [[100, 200, 300, 400]]),
        ("ref_side", "G09", [[80, 220, 260, 440]]),
    ]
    assert "boxes" not in green

    orange = _group_by_signature(fused, "metal", "orange", "painted")
    assert {
        (source["view_id"], source["base_color"]) for source in orange["sources"]
    } == {("ref_iso", "brown"), ("ref_side", "orange")}

    blue = _group_by_signature(fused, "plastic", "blue", "painted")
    assert {
        (source["view_id"], source["base_color"]) for source in blue["sources"]
    } == {("ref_iso", "blue"), ("ref_top", "cyan")}

    white = _group_by_signature(fused, "metal", "white", "painted")
    assert white["singleton"] is True
    assert white["source_view_ids"] == ["ref_front"]

    for view_id, palette in (
        ("ref_front", front),
        ("ref_side", side),
        ("ref_top", top),
        ("ref_iso", iso),
    ):
        assert set(fused["view_group_id_maps"][view_id]) == {
            group["group_id"] for group in palette["groups"]
        }


def test_uncertain_observations_remain_auditable_singletons() -> None:
    front = _palette(
        "ref_front",
        [
            _group(
                "G01",
                family="unknown",
                color="gray",
                finish="unknown",
                box=[100, 100, 200, 200],
            )
        ],
    )
    side = _palette(
        "ref_side",
        [
            _group(
                "G01",
                family="unknown",
                color="gray",
                finish="unknown",
                box=[300, 300, 400, 400],
            )
        ],
    )

    fused = fuse_multiview_palettes([front, side])

    assert fused["summary"]["canonical_group_count"] == 2
    assert fused["summary"]["singleton_group_count"] == 2
    canonical_ids = {
        fused["view_group_id_maps"]["ref_front"]["G01"],
        fused["view_group_id_maps"]["ref_side"]["G01"],
    }
    assert len(canonical_ids) == 2
    assert all(
        group["source_count"] == 1 for group in fused["canonical_palette"]["groups"]
    )


def test_identical_unresolved_pixel_chromatic_groups_merge_across_views() -> None:
    description = (
        "connected blue chromatic region detected from pixels; "
        "physical material unresolved"
    )
    iso_group = _group(
        "G05",
        family="other",
        color="blue",
        finish="other",
        box=[428, 365, 459, 411],
        confidence=0.6,
    )
    top_group = _group(
        "G09",
        family="other",
        color="blue",
        finish="other",
        box=[582, 369, 608, 397],
        confidence=0.6,
    )
    iso_group["visual_description"] = description
    top_group["visual_description"] = description

    palettes = [_palette("iso", [iso_group]), _palette("top", [top_group])]
    audits = [
        _accent_audit("iso", iso_group, image_sha256="1" * 64),
        _accent_audit("top", top_group, image_sha256="2" * 64),
    ]
    fused = fuse_multiview_palettes(
        palettes,
        augmentation_audits=audits,
    )

    assert fused["summary"]["canonical_group_count"] == 1
    group = fused["canonical_palette"]["groups"][0]
    assert group["association_basis"] == (
        "identical_unresolved_pixel_chromatic_multiview"
    )
    assert group["family_hint"] == "other"
    assert group["finish_hint"] == "other"
    assert group["source_view_ids"] == ["iso", "top"]
    assert group["singleton"] is False
    assert group["association_evidence"]["reference_image_sha256s"] == [
        "1" * 64,
        "2" * 64,
    ]
    assert fused["view_group_id_maps"]["iso"]["G05"] == group["group_id"]
    assert fused["view_group_id_maps"]["top"]["G09"] == group["group_id"]
    assert fused == fuse_multiview_palettes(
        list(reversed(palettes)),
        augmentation_audits=list(reversed(audits)),
    )


def test_identical_unresolved_light_neutral_groups_merge_across_views() -> None:
    description = (
        "connected light neutral surface region detected from pixels; "
        "physical material unresolved"
    )
    front_group = _group(
        "G04",
        family="other",
        color="white",
        finish="other",
        box=[460, 320, 535, 455],
        confidence=0.6,
    )
    iso_group = _group(
        "G08",
        family="other",
        color="white",
        finish="other",
        box=[615, 410, 685, 540],
        confidence=0.6,
    )
    front_group["visual_description"] = description
    iso_group["visual_description"] = description
    palettes = [
        _palette("front", [front_group]),
        _palette("iso", [iso_group]),
    ]
    audits = [
        _accent_audit("front", front_group, image_sha256="3" * 64),
        _accent_audit("iso", iso_group, image_sha256="4" * 64),
    ]

    fused = fuse_multiview_palettes(
        palettes,
        augmentation_audits=audits,
    )

    assert fused["summary"]["canonical_group_count"] == 1
    group = fused["canonical_palette"]["groups"][0]
    assert (
        group["association_basis"]
        == UNRESOLVED_PIXEL_LIGHT_NEUTRAL_ASSOCIATION
    )
    assert group["base_color"] == "white"
    assert group["source_view_ids"] == ["front", "iso"]
    assert group["singleton"] is False
    assert is_verified_unresolved_pixel_light_neutral_group(group)


def test_resolved_white_singleton_does_not_absorb_or_block_unresolved_multiview_white() -> None:
    description = (
        "connected light neutral surface region detected from pixels; "
        "physical material unresolved"
    )
    front_group = _group(
        "G04",
        family="other",
        color="white",
        finish="other",
        box=[10, 10, 30, 30],
        confidence=0.6,
    )
    side_group = _group(
        "G07",
        family="other",
        color="white",
        finish="other",
        box=[20, 20, 40, 40],
        confidence=0.6,
    )
    resolved_top_group = _group(
        "G02",
        family="plastic",
        color="white",
        finish="matte",
        box=[5, 5, 25, 25],
        confidence=0.93,
    )
    front_group["visual_description"] = description
    side_group["visual_description"] = description
    resolved_top_group["visual_description"] = "white matte control module"
    palettes = [
        _palette("front", [front_group]),
        _palette("side", [side_group]),
        _palette("top", [resolved_top_group]),
    ]
    audits = [
        _accent_audit("front", front_group, image_sha256="1" * 64),
        _accent_audit("side", side_group, image_sha256="2" * 64),
        None,
    ]

    fused = fuse_multiview_palettes(
        palettes,
        augmentation_audits=audits,
    )

    assert fused["summary"]["canonical_group_count"] == 2
    unresolved_id = fused["view_group_id_maps"]["front"]["G04"]
    assert fused["view_group_id_maps"]["side"]["G07"] == unresolved_id
    assert fused["view_group_id_maps"]["top"]["G02"] != unresolved_id
    unresolved = next(
        group
        for group in fused["canonical_palette"]["groups"]
        if group["group_id"] == unresolved_id
    )
    assert unresolved["source_view_ids"] == ["front", "side"]
    assert unresolved["association_basis"] == (
        UNRESOLVED_PIXEL_LIGHT_NEUTRAL_ASSOCIATION
    )
    assert is_verified_unresolved_pixel_light_neutral_group(unresolved)


def test_identical_masked_dark_groups_merge_only_with_mask_bound_proof() -> None:
    description = (
        "connected dark surface region detected inside the trusted foreground "
        "mask; physical material unresolved"
    )
    front_group = _group(
        "G05",
        family="other",
        color="black",
        finish="other",
        box=[235, 726, 729, 764],
        confidence=0.6,
    )
    side_group = _group(
        "G09",
        family="other",
        color="black",
        finish="other",
        box=[408, 701, 705, 744],
        confidence=0.6,
    )
    front_group["visual_description"] = description
    side_group["visual_description"] = description
    palettes = [
        _palette("front", [front_group]),
        _palette("side", [side_group]),
    ]
    audits = [
        _accent_audit(
            "front",
            front_group,
            image_sha256="5" * 64,
            mask_sha256="7" * 64,
        ),
        _accent_audit(
            "side",
            side_group,
            image_sha256="6" * 64,
            mask_sha256="8" * 64,
        ),
    ]

    fused = fuse_multiview_palettes(
        palettes,
        augmentation_audits=audits,
    )

    assert fused["summary"]["canonical_group_count"] == 1
    group = fused["canonical_palette"]["groups"][0]
    assert group["association_basis"] == (
        UNRESOLVED_PIXEL_MASKED_DARK_ASSOCIATION
    )
    assert group["source_view_ids"] == ["front", "side"]
    assert group["association_evidence"]["foreground_mask_sha256s"] == [
        "7" * 64,
        "8" * 64,
    ]
    assert is_verified_unresolved_pixel_masked_dark_group(group)

    audits[1] = _accent_audit(
        "side",
        side_group,
        image_sha256="6" * 64,
    )
    with pytest.raises(PaletteFusionError, match="lacks foreground-mask proof"):
        fuse_multiview_palettes(palettes, augmentation_audits=audits)


def test_filtered_augmentation_box_subset_remains_proof_bound() -> None:
    group = _group(
        "G04",
        family="other",
        color="white",
        finish="other",
        box=[100, 100, 180, 240],
        confidence=0.6,
    )
    group["boxes"].append([400, 400, 440, 460])
    group["visual_description"] = (
        "connected light neutral surface region detected from pixels; "
        "physical material unresolved"
    )
    audit = _accent_audit("front", group, image_sha256="5" * 64)
    group["boxes"] = [group["boxes"][0]]

    fused = fuse_multiview_palettes(
        [_palette("front", [group])],
        augmentation_audits=[audit],
    )

    canonical = fused["canonical_palette"]["groups"][0]
    assert canonical["sources"][0]["boxes"] == [[100, 100, 180, 240]]
    assert canonical["sources"][0]["accent_augmentation_evidence"][
        "augmented_group_sha256"
    ]


def test_unresolved_pixel_groups_without_augmentation_proof_remain_singletons() -> None:
    description = (
        "connected blue chromatic region detected from pixels; "
        "physical material unresolved"
    )
    groups = []
    for group_id in ("G05", "G09"):
        group = _group(
            group_id,
            family="other",
            color="blue",
            finish="other",
            box=[10, 10, 30, 30],
            confidence=0.6,
        )
        group["visual_description"] = description
        groups.append(group)

    fused = fuse_multiview_palettes(
        [_palette("iso", [groups[0]]), _palette("top", [groups[1]])]
    )

    assert fused["summary"]["canonical_group_count"] == 2
    assert fused["summary"]["singleton_group_count"] == 2


def test_unresolved_pixel_groups_require_independent_reference_images() -> None:
    description = (
        "connected blue chromatic region detected from pixels; "
        "physical material unresolved"
    )
    groups = []
    for group_id in ("G05", "G09"):
        group = _group(
            group_id,
            family="other",
            color="blue",
            finish="other",
            box=[10, 10, 30, 30],
            confidence=0.6,
        )
        group["visual_description"] = description
        groups.append(group)
    palettes = [_palette("iso", [groups[0]]), _palette("top", [groups[1]])]
    audits = [
        _accent_audit("iso", groups[0], image_sha256="a" * 64),
        _accent_audit("top", groups[1], image_sha256="a" * 64),
    ]

    fused = fuse_multiview_palettes(
        palettes,
        augmentation_audits=audits,
    )

    assert fused["summary"]["canonical_group_count"] == 2
    assert fused["summary"]["singleton_group_count"] == 2


def test_tampered_augmentation_group_fails_closed() -> None:
    group = _group(
        "G05",
        family="other",
        color="blue",
        finish="other",
        box=[10, 10, 30, 30],
        confidence=0.6,
    )
    group["visual_description"] = (
        "connected blue chromatic region detected from pixels; "
        "physical material unresolved"
    )
    audit = _accent_audit("iso", group, image_sha256="b" * 64)
    group["boxes"] = [[20, 20, 40, 40]]

    with pytest.raises(PaletteFusionError, match="pixel-component audit"):
        fuse_multiview_palettes(
            [_palette("iso", [group])],
            augmentation_audits=[audit],
        )


def test_augmentation_source_view_must_match_palette() -> None:
    group = _group(
        "G05",
        family="other",
        color="blue",
        finish="other",
        box=[10, 10, 30, 30],
        confidence=0.6,
    )
    group["visual_description"] = (
        "connected blue chromatic region detected from pixels; "
        "physical material unresolved"
    )

    with pytest.raises(PaletteFusionError, match="source view"):
        fuse_multiview_palettes(
            [_palette("iso", [group])],
            augmentation_audits=[
                _accent_audit("top", group, image_sha256="c" * 64)
            ],
        )


def test_unresolved_pixel_chromatic_descriptions_must_match_exactly() -> None:
    descriptions = (
        (
            "connected blue chromatic region detected from pixels; "
            "physical material unresolved"
        ),
        (
            "connected blue chromatic region detected from pixels; "
            "physical material unresolved elsewhere"
        ),
    )
    groups = []
    for description in descriptions:
        group = _group(
            "G01",
            family="other",
            color="blue",
            finish="other",
            box=[10, 10, 30, 30],
            confidence=0.6,
        )
        group["visual_description"] = description
        groups.append(group)

    fused = fuse_multiview_palettes(
        [_palette("iso", [groups[0]]), _palette("top", [groups[1]])]
    )

    assert fused["summary"]["canonical_group_count"] == 2
    assert fused["summary"]["singleton_group_count"] == 2


def test_unique_chromatic_group_does_not_merge_conflicting_known_finish() -> None:
    palettes = [
        _palette(
            "ref_a",
            [
                _group(
                    "G01",
                    family="metal",
                    color="orange",
                    finish="painted",
                    box=[10, 10, 100, 100],
                )
            ],
        ),
        _palette(
            "ref_b",
            [
                _group(
                    "G01",
                    family="metal",
                    color="brown",
                    finish="bare",
                    box=[20, 20, 110, 110],
                )
            ],
        ),
    ]

    fused = fuse_multiview_palettes(palettes)

    assert fused["summary"]["canonical_group_count"] == 2
    assert all(
        group["distinct_view_count"] == 1
        for group in fused["canonical_palette"]["groups"]
    )


def test_unique_chromatic_group_lets_unresolved_pixel_evidence_abstain() -> None:
    palettes = [
        _palette(
            "ref_a",
            [
                _group(
                    "G01",
                    family="metal",
                    color="orange",
                    finish="painted",
                    box=[10, 10, 100, 100],
                    confidence=0.99,
                )
            ],
        ),
        *[
            _palette(
                view_id,
                [
                    _group(
                        "G01",
                        family="other",
                        color="brown",
                        finish="other",
                        box=[20, 20, 110, 110],
                        confidence=0.6,
                    )
                ],
            )
            for view_id in ("ref_b", "ref_c", "ref_d")
        ],
    ]

    group = fuse_multiview_palettes(palettes)["canonical_palette"]["groups"][0]

    assert group["family_hint"] == "metal"
    assert group["finish_hint"] == "painted"
    assert group["representative_ref"]["view_id"] == "ref_a"
    assert group["semantic_consensus"]["finish_hint_votes"] == [
        {"value": "painted", "source_count": 1, "confidence_sum": 0.99},
        {"value": "other", "source_count": 3, "confidence_sum": 1.8},
    ]


def test_unique_painted_colour_absorbs_one_substrate_family_outlier() -> None:
    palettes = [
        _palette(
            view_id,
            [
                _group(
                    "G01",
                    family="metal",
                    color="green",
                    finish="painted",
                    box=[10, 10, 100, 100],
                    confidence=0.9,
                )
            ],
        )
        for view_id in ("ref_front", "ref_side", "ref_iso")
    ]
    palettes.append(
        _palette(
            "ref_top",
            [
                _group(
                    "G07",
                    family="plastic",
                    color="green",
                    finish="painted",
                    box=[20, 20, 110, 110],
                    confidence=0.95,
                )
            ],
        )
    )

    fused = fuse_multiview_palettes(palettes)

    assert fused["summary"]["canonical_group_count"] == 1
    group = fused["canonical_palette"]["groups"][0]
    assert group["family_hint"] == "metal"
    assert group["finish_hint"] == "painted"
    assert group["distinct_view_count"] == 4
    assert {
        fused["view_group_id_maps"][view_id][local_id]
        for view_id, local_id in (
            ("ref_front", "G01"),
            ("ref_side", "G01"),
            ("ref_iso", "G01"),
            ("ref_top", "G07"),
        )
    } == {group["group_id"]}


@pytest.mark.parametrize(
    "families",
    [
        ("metal", "metal", "plastic", "plastic"),
        ("metal", "metal", "metal", "plastic", "plastic"),
        ("metal", "metal", "metal", "plastic", "rubber"),
    ],
)
def test_ambiguous_painted_substrate_families_remain_separate(
    families: tuple[str, ...],
) -> None:
    palettes = [
        _palette(
            f"ref_{index}",
            [
                _group(
                    "G01",
                    family=family,
                    color="green",
                    finish="painted",
                    box=[10, 10, 100, 100],
                )
            ],
        )
        for index, family in enumerate(families)
    ]

    fused = fuse_multiview_palettes(palettes)

    assert fused["summary"]["canonical_group_count"] == len(set(families))
    assert {
        group["family_hint"] for group in fused["canonical_palette"]["groups"]
    } == set(families)


def test_painted_family_outlier_does_not_cross_finish_disagreement() -> None:
    palettes = [
        _palette(
            "ref_a",
            [
                _group(
                    "G01",
                    family="metal",
                    color="green",
                    finish="painted",
                    box=[10, 10, 100, 100],
                )
            ],
        ),
        _palette(
            "ref_b",
            [
                _group(
                    "G01",
                    family="metal",
                    color="green",
                    finish="painted",
                    box=[10, 10, 100, 100],
                )
            ],
        ),
        _palette(
            "ref_c",
            [
                _group(
                    "G01",
                    family="plastic",
                    color="green",
                    finish="glossy",
                    box=[10, 10, 100, 100],
                )
            ],
        ),
    ]

    fused = fuse_multiview_palettes(palettes)

    assert fused["summary"]["canonical_group_count"] == 2


def test_duplicate_same_colour_in_one_view_requires_family_and_finish() -> None:
    palettes = [
        _palette(
            "ref_a",
            [
                _group(
                    "G01",
                    family="metal",
                    color="blue",
                    finish="painted",
                    box=[10, 10, 100, 100],
                ),
                _group(
                    "G02",
                    family="plastic",
                    color="cyan",
                    finish="painted",
                    box=[130, 10, 220, 100],
                ),
            ],
        ),
        _palette(
            "ref_b",
            [
                _group(
                    "G01",
                    family="plastic",
                    color="cyan",
                    finish="painted",
                    box=[20, 20, 110, 110],
                )
            ],
        ),
        _palette(
            "ref_c",
            [
                _group(
                    "G01",
                    family="metal",
                    color="cyan",
                    finish="glossy",
                    box=[30, 30, 120, 120],
                )
            ],
        ),
    ]

    fused = fuse_multiview_palettes(palettes)

    assert fused["summary"]["canonical_group_count"] == 3
    assert fused["summary"]["singleton_group_count"] == 2


def test_fusion_rejects_duplicate_source_view_ids() -> None:
    palette = _palette(
        "ref_front",
        [
            _group(
                "G01",
                family="metal",
                color="green",
                finish="painted",
                box=[100, 100, 200, 200],
            )
        ],
    )

    with pytest.raises(PaletteFusionError, match="duplicate source_view_id"):
        fuse_multiview_palettes([palette, palette])
