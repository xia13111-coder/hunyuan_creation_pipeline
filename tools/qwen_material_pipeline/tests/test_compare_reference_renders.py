from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from qwen_material_pipeline.evidence import reference_compare as comparison
from qwen_material_pipeline.evidence.reference_compare import (
    ComparisonInputError,
    DEFAULT_THRESHOLDS,
    EXIT_INPUT_ERROR,
    EXIT_REQUIRE_PASS_FAILED,
    EXIT_SUCCESS,
    _infer_reference_mask,
    _annotate_group_delivery_presence,
    _mask_metrics,
    _part_color,
    _unreferenced_render_chromatic_mass,
    compare_reference_renders,
    main,
)


def test_tiny_nonzero_group_near_recall_boundary_is_quantization_tolerant() -> None:
    group_recall = {
        "groups": [
            {
                "reference_evidence_weight": 77,
                "observed_render_share": 0.0021,
                "recall": 0.49,
            }
        ]
    }

    assert (
        _annotate_group_delivery_presence(
            group_recall,
            thresholds=DEFAULT_THRESHOLDS,
        )
        is False
    )
    assert group_recall["groups"][0]["delivery_presence_status"] == (
        "LOW_EVIDENCE_NEAR_THRESHOLD_PRESENT"
    )

    for field, value in (
        ("reference_evidence_weight", 128),
        ("observed_render_share", 0.0),
        ("recall", 0.44),
    ):
        candidate = {
            "groups": [
                {
                    "reference_evidence_weight": 77,
                    "observed_render_share": 0.0021,
                    "recall": 0.49,
                    field: value,
                }
            ]
        }
        assert _annotate_group_delivery_presence(
            candidate,
            thresholds=DEFAULT_THRESHOLDS,
        )
        assert candidate["groups"][0]["delivery_presence_status"] == "MISSING"


def _shape_box(shape: str) -> tuple[int, int, int, int]:
    if shape == "vertical":
        return (58, 20, 101, 139)
    if shape == "horizontal":
        return (20, 58, 139, 101)
    if shape == "wide":
        return (28, 35, 131, 124)
    raise AssertionError(shape)


def test_reference_mask_removes_connected_viewer_axes() -> None:
    image = Image.new("RGB", (256, 256), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((52, 96, 215, 175), fill=(45, 145, 62))
    draw.rectangle((126, 0, 128, 255), fill=(180, 20, 20))
    draw.rectangle((0, 134, 255, 136), fill=(40, 120, 40))

    mask, audit = _infer_reference_mask(image)
    metrics = _mask_metrics(mask)

    assert audit["thin_overlay_opening_size"] == 5
    assert metrics["bbox"] is not None
    left, top, right, bottom = metrics["bbox"]
    assert left > 0 and top > 0
    assert right < image.width and bottom < image.height
    assert mask.getpixel((80, 110)) == 255


def test_large_render_only_chromatic_mass_is_a_hard_failure() -> None:
    bins = (
        "black",
        "achromatic_dark",
        "achromatic_mid",
        "achromatic_light",
        "red",
        "orange_brown",
        "yellow",
        "green",
        "cyan_blue",
        "purple",
    )
    reference_categories = {label: 0.0 for label in bins}
    render_categories = {label: 0.0 for label in bins}
    reference_categories["achromatic_light"] = 0.999
    reference_categories["yellow"] = 0.001
    render_categories["achromatic_light"] = 0.89
    render_categories["yellow"] = 0.11

    result = _unreferenced_render_chromatic_mass(
        {"category_distribution": reference_categories},
        {"category_distribution": render_categories},
        alignment={"score": 0.80},
        thresholds=DEFAULT_THRESHOLDS,
    )

    assert result["status"] == "FAIL"
    assert result["failed_color_bins"] == ["yellow"]


def test_render_only_chromatic_mass_requires_strong_alignment() -> None:
    reference_categories = {
        label: 0.0
        for label in (
            "black",
            "achromatic_dark",
            "achromatic_mid",
            "achromatic_light",
            "red",
            "orange_brown",
            "yellow",
            "green",
            "cyan_blue",
            "purple",
        )
    }
    render_categories = dict(reference_categories)
    reference_categories["achromatic_light"] = 1.0
    render_categories["achromatic_light"] = 0.80
    render_categories["yellow"] = 0.20

    result = _unreferenced_render_chromatic_mass(
        {"category_distribution": reference_categories},
        {"category_distribution": render_categories},
        alignment={"score": 0.40},
        thresholds=DEFAULT_THRESHOLDS,
    )

    assert result["status"] == "PASS"


def test_color_censored_local_mask_cannot_prove_unreferenced_color_absence() -> None:
    labels = (
        "black",
        "achromatic_dark",
        "achromatic_mid",
        "achromatic_light",
        "red",
        "orange_brown",
        "yellow",
        "green",
        "cyan_blue",
        "purple",
    )
    reference_categories = {label: 0.0 for label in labels}
    render_categories = {label: 0.0 for label in labels}
    reference_categories["orange_brown"] = 1.0
    render_categories["orange_brown"] = 0.75
    render_categories["green"] = 0.25

    result = _unreferenced_render_chromatic_mass(
        {"category_distribution": reference_categories},
        {"category_distribution": render_categories},
        alignment={"score": 0.90},
        thresholds=DEFAULT_THRESHOLDS,
        evidence={"samples": [{"base_color": "orange"}]},
        reference_target_mask_audit={
            "method": "trusted_group_roi_and_color_family_intersected_foreground"
        },
    )

    assert result["status"] == "NOT_APPLICABLE"
    assert result["failed_color_bins"] == []
    green = next(item for item in result["bins"] if item["color_bin"] == "green")
    assert green["raw_excess_share"] == pytest.approx(0.25)
    assert green["effective_excess_share"] == 0.0
    assert green["status"] == "NOT_APPLICABLE"


def test_scoped_compatible_color_family_bins_are_pooled() -> None:
    labels = (
        "black",
        "achromatic_dark",
        "achromatic_mid",
        "achromatic_light",
        "red",
        "orange_brown",
        "yellow",
        "green",
        "cyan_blue",
        "purple",
    )
    reference_categories = {label: 0.0 for label in labels}
    render_categories = {label: 0.0 for label in labels}
    reference_categories["orange_brown"] = 1.0
    render_categories["orange_brown"] = 0.80
    render_categories["red"] = 0.20

    result = _unreferenced_render_chromatic_mass(
        {"category_distribution": reference_categories},
        {"category_distribution": render_categories},
        alignment={"score": 0.90},
        thresholds=DEFAULT_THRESHOLDS,
        evidence={"samples": [{"base_color": "orange"}]},
        reference_target_mask_audit={"method": "explicit_target_group_mask"},
    )

    assert result["status"] == "PASS"
    red = next(item for item in result["bins"] if item["color_bin"] == "red")
    assert red["raw_status"] == "FAIL"
    assert red["status"] == "PASS"
    assert red["trusted_family_compatible"] is True
    assert red["effective_excess_share"] == 0.0


def _write_reference(
    root: Path,
    view_id: str,
    *,
    shape: str,
    color: tuple[int, int, int],
    evidence_color: str = "green",
    usable_evidence: bool = True,
) -> dict:
    image_path = root / f"{view_id}_reference.png"
    image = Image.new("RGB", (160, 160), (0, 0, 0))
    ImageDraw.Draw(image).rectangle(_shape_box(shape), fill=color)
    image.save(image_path)
    if not usable_evidence:
        return {
            "id": view_id,
            "image": str(image_path),
            "palette_mask": None,
            "palette_status": "unusable",
        }
    evidence_path = root / f"{view_id}_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "accepted_group_ids": ["G01"],
                "rejected_group_ids": [],
                "groups": [
                    {
                        "group_id": "G01",
                        "base_color": evidence_color,
                        "accepted": True,
                        "boxes": [
                            {
                                "accepted": True,
                                "representative_srgb": list(color),
                                "matching_pixel_count": 1000,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return {
        "id": view_id,
        "image": str(image_path),
        "palette_mask": None,
        "palette_status": "usable",
        "palette_artifacts": {"normalized_evidence_audit": str(evidence_path)},
    }


def _write_render(
    root: Path,
    view_id: str,
    *,
    shape: str,
    color: tuple[int, int, int],
    declared_pixel_multiplier: float = 1.0,
) -> dict:
    rgb_path = root / f"{view_id}_render.png"
    ids_path = root / f"{view_id}_part_ids.png"
    rgb = Image.new("RGB", (160, 160), (150, 155, 160))
    ids = Image.new("RGB", (160, 160), (28, 28, 28))
    box = _shape_box(shape)
    ImageDraw.Draw(rgb).rectangle(box, fill=color)
    ImageDraw.Draw(ids).rectangle(box, fill=_part_color("P0001"))
    rgb.save(rgb_path)
    ids.save(ids_path)
    pixel_data = (
        ids.get_flattened_data()
        if hasattr(ids, "get_flattened_data")
        else ids.getdata()
    )
    actual_pixels = sum(1 for pixel in pixel_data if pixel == _part_color("P0001"))
    return {
        "view_id": view_id,
        "rgb": str(rgb_path),
        "part_ids": str(ids_path),
        "visible_parts": [
            {
                "part_id": "P0001",
                "pixels": int(round(actual_pixels * declared_pixel_multiplier)),
            }
        ],
    }


def _case(
    tmp_path: Path,
    references: list[dict],
    renders: list[dict],
) -> tuple[Path, Path]:
    manifest_path = tmp_path / "reference_manifest.json"
    registry_path = tmp_path / "rendered_registry.json"
    manifest_path.write_text(json.dumps({"source_views": references}), encoding="utf-8")
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "qwen-material-parts/v1",
                "parts": [{"part_id": "P0001", "prim_path": "/Asset/P0001"}],
                "render_set": {"views": renders},
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, registry_path


def _write_scoped_reference(
    root: Path,
    view_id: str,
    *,
    target_color: tuple[int, int, int] = (35, 115, 48),
    other_color: tuple[int, int, int] = (175, 35, 35),
) -> dict:
    image_path = root / f"{view_id}_scoped_reference.png"
    image = Image.new("RGB", (160, 160), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 25, 74, 134), fill=target_color)
    draw.rectangle((85, 25, 144, 134), fill=other_color)
    image.save(image_path)
    evidence_path = root / f"{view_id}_scoped_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "accepted_group_ids": ["G_LOCAL_TARGET", "G_LOCAL_OTHER"],
                "rejected_group_ids": [],
                "groups": [
                    {
                        "group_id": "G_LOCAL_TARGET",
                        "base_color": "green",
                        "accepted": True,
                        "boxes": [
                            {
                                "accepted": True,
                                "box": [90, 150, 475, 850],
                                "representative_srgb": list(target_color),
                                "matching_pixel_count": 6600,
                            }
                        ],
                    },
                    {
                        "group_id": "G_LOCAL_OTHER",
                        "base_color": "red",
                        "accepted": True,
                        "boxes": [
                            {
                                "accepted": True,
                                "box": [525, 150, 910, 850],
                                "representative_srgb": list(other_color),
                                "matching_pixel_count": 6600,
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return {
        "id": view_id,
        "image": str(image_path),
        "palette_mask": None,
        "palette_status": "usable",
        "palette_artifacts": {"normalized_evidence_audit": str(evidence_path)},
    }


def _write_scoped_render(
    root: Path,
    view_id: str,
    *,
    target_color: tuple[int, int, int] = (35, 115, 48),
    other_color: tuple[int, int, int] = (175, 35, 35),
    target_visible: bool = True,
) -> dict:
    rgb_path = root / f"{view_id}_scoped_render.png"
    ids_path = root / f"{view_id}_scoped_part_ids.png"
    rgb = Image.new("RGB", (160, 160), (150, 155, 160))
    ids = Image.new("RGB", (160, 160), (28, 28, 28))
    rgb_draw = ImageDraw.Draw(rgb)
    ids_draw = ImageDraw.Draw(ids)
    visible_parts: list[dict] = []
    if target_visible:
        rgb_draw.rectangle((15, 25, 74, 134), fill=target_color)
        ids_draw.rectangle((15, 25, 74, 134), fill=_part_color("P0001"))
        visible_parts.append({"part_id": "P0001", "pixels": 60 * 110})
    rgb_draw.rectangle((85, 25, 144, 134), fill=other_color)
    ids_draw.rectangle((85, 25, 144, 134), fill=_part_color("P0002"))
    visible_parts.append({"part_id": "P0002", "pixels": 60 * 110})
    rgb.save(rgb_path)
    ids.save(ids_path)
    return {
        "view_id": view_id,
        "rgb": str(rgb_path),
        "part_ids": str(ids_path),
        "visible_parts": visible_parts,
    }


def _scoped_case(
    root: Path,
    *,
    other_render_color: tuple[int, int, int] = (175, 35, 35),
    target_visible: bool = True,
    reference_view_ids: tuple[str, ...] = ("ref_a", "ref_b"),
) -> tuple[Path, Path, Path, dict[str, str]]:
    references = [
        _write_scoped_reference(root, reference_view_id)
        for reference_view_id in reference_view_ids
    ]
    renders = [
        _write_scoped_render(
            root,
            reference_view_id.replace("ref_", "view_", 1),
            other_color=other_render_color,
            target_visible=target_visible,
        )
        for reference_view_id in reference_view_ids
    ]
    manifest = root / "scoped_reference_manifest.json"
    registry = root / "scoped_rendered_registry.json"
    fusion = root / "palette_fusion.json"
    manifest.write_text(json.dumps({"source_views": references}), encoding="utf-8")
    registry.write_text(
        json.dumps(
            {
                "schema_version": "qwen-material-parts/v1",
                "parts": [
                    {"part_id": "P0001", "prim_path": "/Asset/P0001"},
                    {"part_id": "P0002", "prim_path": "/Asset/P0002"},
                ],
                "render_set": {"views": renders},
            }
        ),
        encoding="utf-8",
    )
    fusion.write_text(
        json.dumps(
            {
                "canonical_palette": {
                    "groups": [
                        {
                            "group_id": "G_TARGET",
                            "sources": [
                                {
                                    "view_id": reference_view_id,
                                    "local_group_id": "G_LOCAL_TARGET",
                                }
                                for reference_view_id in reference_view_ids
                            ],
                        }
                    ]
                },
                "view_group_id_maps": {
                    reference_view_id: {"G_LOCAL_TARGET": "G_TARGET"}
                    for reference_view_id in reference_view_ids
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest, registry, fusion, {
        reference_view_id: reference_view_id.replace("ref_", "view_", 1)
        for reference_view_id in reference_view_ids
    }


def test_scoped_group_score_ignores_non_target_render_parts(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    manifest, registry, fusion, mapping = _scoped_case(first_root)
    (
        second_manifest,
        second_registry,
        second_fusion,
        second_mapping,
    ) = _scoped_case(second_root, other_render_color=(25, 45, 190))

    first = compare_reference_renders(
        manifest,
        registry,
        view_mapping=mapping,
        target_part_ids=["P0001"],
        target_group_id="G_TARGET",
        palette_fusion=fusion,
    )
    second = compare_reference_renders(
        second_manifest,
        second_registry,
        view_mapping=second_mapping,
        target_part_ids=["P0001"],
        target_group_id="G_TARGET",
        palette_fusion=second_fusion,
    )

    assert first["aggregate"]["status"] == "PASS"
    assert second["aggregate"]["status"] == "PASS"
    assert first["aggregate"]["material_appearance_score"] == pytest.approx(
        second["aggregate"]["material_appearance_score"]
    )
    assert first["inputs"]["comparison_scope"]["mode"] == "canonical_group_local"
    assert first["inputs"]["comparison_scope"]["target_part_ids"] == ["P0001"]
    assert first["inputs"]["comparison_scope"]["reference_view_ids"] == [
        "ref_a",
        "ref_b",
    ]
    for view in first["views"]:
        assert (
            view["reference"]["foreground"]["pixel_count"]
            < view["reference"]["alignment_foreground"]["pixel_count"]
        )
        assert (
            view["render"]["foreground"]["pixel_count"]
            < view["render"]["alignment_foreground"]["pixel_count"]
        )
        assert view["material_color"]["render_distribution"]["category_distribution"][
            "green"
        ] == pytest.approx(1.0)


def test_scoped_group_explicit_reference_subset_filters_canonical_sources(
    tmp_path: Path,
) -> None:
    manifest, registry, fusion, mapping = _scoped_case(
        tmp_path,
        reference_view_ids=("ref_a", "ref_b", "ref_c"),
    )

    report = compare_reference_renders(
        manifest,
        registry,
        view_mapping=mapping,
        target_part_ids=["P0001"],
        target_group_id="G_TARGET",
        target_reference_view_ids=["ref_a", "ref_c"],
        palette_fusion=fusion,
        minimum_comparable_views=2,
    )

    scope = report["inputs"]["comparison_scope"]
    assert scope["reference_view_ids"] == ["ref_a", "ref_c"]
    assert [view["reference_view_id"] for view in report["views"]] == [
        "ref_a",
        "ref_c",
    ]
    assert report["aggregate"]["reference_view_count"] == 2
    assert report["aggregate"]["comparable_view_count"] == 2
    assert report["aggregate"]["reference_view_coverage_status"] == "PASS"


@pytest.mark.parametrize(
    ("target_reference_view_ids", "message"),
    (
        (["ref_a", "unknown"], "canonical|unknown"),
        (["ref_a", "ref_a"], "unique|duplicate"),
        (["ref_a"], "at least two"),
    ),
)
def test_scoped_group_rejects_invalid_explicit_reference_subset(
    tmp_path: Path,
    target_reference_view_ids: list[str],
    message: str,
) -> None:
    manifest, registry, fusion, mapping = _scoped_case(
        tmp_path,
        reference_view_ids=("ref_a", "ref_b", "ref_c"),
    )

    with pytest.raises(ComparisonInputError, match=message):
        compare_reference_renders(
            manifest,
            registry,
            view_mapping=mapping,
            target_part_ids=["P0001"],
            target_group_id="G_TARGET",
            target_reference_view_ids=target_reference_view_ids,
            palette_fusion=fusion,
            minimum_comparable_views=2,
        )


def test_scoped_face_subset_records_containing_part_proxy_boundary(
    tmp_path: Path,
) -> None:
    manifest, registry, fusion, mapping = _scoped_case(tmp_path)

    report = compare_reference_renders(
        manifest,
        registry,
        view_mapping=mapping,
        target_part_ids=["P0001"],
        target_entities=[
            {
                "entity_kind": "face_subset",
                "part_id": "P0001",
                "subset_name": "Cover",
            }
        ],
        target_group_id="G_TARGET",
        palette_fusion=fusion,
    )

    scope = report["inputs"]["comparison_scope"]
    assert scope["target_entities"] == [
        {
            "entity_kind": "face_subset",
            "part_id": "P0001",
            "subset_name": "Cover",
        }
    ]
    assert scope["render_mask_granularity"] == "containing_part_proxy"
    assert scope["face_subset_render_mask_exact"] is False
    assert report["aggregate"]["status"] == "PASS"


def test_scoped_group_with_invisible_target_parts_fails_closed(
    tmp_path: Path,
) -> None:
    manifest, registry, fusion, mapping = _scoped_case(
        tmp_path,
        target_visible=False,
    )

    report = compare_reference_renders(
        manifest,
        registry,
        view_mapping=mapping,
        target_part_ids=["P0001"],
        target_group_id="G_TARGET",
        palette_fusion=fusion,
    )

    assert report["aggregate"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert report["aggregate"]["material_appearance_score"] is None
    assert report["aggregate"]["comparable_view_count"] == 0
    assert all(view["status"] == "UNSCORABLE" for view in report["views"])
    assert all(
        "render_target_foreground_missing" in view["reasons"]
        for view in report["views"]
    )


def test_scoped_group_contract_rejects_partial_or_unknown_targets(
    tmp_path: Path,
) -> None:
    manifest, registry, fusion, mapping = _scoped_case(tmp_path)

    with pytest.raises(ComparisonInputError, match="requires target_part_ids"):
        compare_reference_renders(
            manifest,
            registry,
            view_mapping=mapping,
            target_part_ids=["P0001"],
        )
    with pytest.raises(ComparisonInputError, match="absent from rendered registry"):
        compare_reference_renders(
            manifest,
            registry,
            view_mapping=mapping,
            target_part_ids=["P9999"],
            target_group_id="G_TARGET",
            palette_fusion=fusion,
        )


def test_scoped_group_accepts_explicit_reference_masks_without_roi_boxes(
    tmp_path: Path,
) -> None:
    manifest, registry, fusion, mapping = _scoped_case(tmp_path)
    manifest_document = json.loads(manifest.read_text(encoding="utf-8"))
    for source_view in manifest_document["source_views"]:
        evidence_path = Path(
            source_view["palette_artifacts"]["normalized_evidence_audit"]
        )
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence["groups"][0]["boxes"][0].pop("box")
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        target_mask_path = tmp_path / f"{source_view['id']}_target_mask.png"
        target_mask = Image.new("L", (160, 160), 0)
        ImageDraw.Draw(target_mask).rectangle((15, 25, 74, 134), fill=255)
        target_mask.save(target_mask_path)
        source_view["target_group_masks"] = {"G_LOCAL_TARGET": str(target_mask_path)}
    manifest.write_text(json.dumps(manifest_document), encoding="utf-8")

    report = compare_reference_renders(
        manifest,
        registry,
        view_mapping=mapping,
        target_part_ids=["P0001"],
        target_group_id="G_TARGET",
        palette_fusion=fusion,
    )

    assert report["aggregate"]["status"] == "PASS"
    assert all(
        view["reference"]["target_foreground_audit"]["method"]
        == "explicit_target_group_mask_intersected_foreground"
        for view in report["views"]
    )


def test_scoped_group_without_reference_mask_or_roi_fails_closed(
    tmp_path: Path,
) -> None:
    manifest, registry, fusion, mapping = _scoped_case(tmp_path)
    manifest_document = json.loads(manifest.read_text(encoding="utf-8"))
    evidence_path = Path(
        manifest_document["source_views"][0]["palette_artifacts"][
            "normalized_evidence_audit"
        ]
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["groups"][0]["boxes"][0].pop("box")
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(ComparisonInputError, match="needs normalized"):
        compare_reference_renders(
            manifest,
            registry,
            view_mapping=mapping,
            target_part_ids=["P0001"],
            target_group_id="G_TARGET",
            palette_fusion=fusion,
        )


def test_cli_scoped_group_writes_local_report(tmp_path: Path) -> None:
    manifest, registry, fusion, mapping = _scoped_case(tmp_path)
    view_map = tmp_path / "view_map.json"
    output = tmp_path / "scoped_report.json"
    view_map.write_text(json.dumps({"mapping": mapping}), encoding="utf-8")

    exit_code = main(
        [
            "--reference-manifest",
            str(manifest),
            "--rendered-registry",
            str(registry),
            "--view-map",
            str(view_map),
            "--target-part-id",
            "P0001",
            "--target-group-id",
            "G_TARGET",
            "--palette-fusion",
            str(fusion),
            "--output",
            str(output),
            "--require-pass",
        ]
    )

    assert exit_code == EXIT_SUCCESS
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["aggregate"]["status"] == "PASS"
    assert report["inputs"]["comparison_scope"]["target_group_id"] == "G_TARGET"


def test_aligned_same_material_passes_with_two_explicit_views(tmp_path: Path) -> None:
    references = [
        _write_reference(tmp_path, "ref_a", shape="vertical", color=(35, 115, 48)),
        _write_reference(tmp_path, "ref_b", shape="horizontal", color=(35, 115, 48)),
    ]
    renders = [
        _write_render(tmp_path, "view_a", shape="vertical", color=(28, 92, 41)),
        _write_render(tmp_path, "view_b", shape="horizontal", color=(28, 92, 41)),
    ]
    manifest, registry = _case(tmp_path, references, renders)

    report = compare_reference_renders(
        manifest,
        registry,
        view_mapping={"ref_a": "view_a", "ref_b": "view_b"},
    )

    assert report["aggregate"]["status"] == "PASS"
    assert report["aggregate"]["comparable_view_count"] == 2
    assert report["aggregate"]["material_color_difference_score"] < 0.1
    assert all(view["status"] == "PASS" for view in report["views"])
    assert all(
        view["material_color"]["difference_score"] < 0.1 for view in report["views"]
    )
    assert all(view["alignment"]["score"] > 0.9 for view in report["views"])


def test_wrongly_aligned_pair_is_unscorable_not_material_failure(
    tmp_path: Path,
) -> None:
    references = [
        _write_reference(tmp_path, "ref_a", shape="vertical", color=(35, 115, 48))
    ]
    renders = [
        _write_render(tmp_path, "view_a", shape="horizontal", color=(180, 25, 25))
    ]
    manifest, registry = _case(tmp_path, references, renders)

    report = compare_reference_renders(
        manifest,
        registry,
        view_mapping={"ref_a": "view_a"},
        minimum_comparable_views=1,
    )

    view = report["views"][0]
    assert view["status"] == "UNSCORABLE"
    assert view["material_color"] is None
    assert "view_alignment_below_material_scoring_threshold" in view["reasons"]
    assert report["aggregate"]["status"] == "INSUFFICIENT_EVIDENCE"


def test_two_strongly_aligned_wrong_colors_can_fail_material(
    tmp_path: Path,
) -> None:
    references = [
        _write_reference(tmp_path, "ref_a", shape="vertical", color=(35, 115, 48)),
        _write_reference(tmp_path, "ref_b", shape="horizontal", color=(35, 115, 48)),
    ]
    renders = [
        _write_render(tmp_path, "view_a", shape="vertical", color=(180, 25, 25)),
        _write_render(tmp_path, "view_b", shape="horizontal", color=(180, 25, 25)),
    ]
    manifest, registry = _case(tmp_path, references, renders)

    report = compare_reference_renders(
        manifest,
        registry,
        view_mapping={"ref_a": "view_a", "ref_b": "view_b"},
    )

    assert [view["status"] for view in report["views"]] == ["FAIL", "FAIL"]
    assert report["aggregate"]["status"] == "FAIL"
    assert report["aggregate"]["material_match_conclusion"] == "FAIL"
    assert (
        "multiple_aligned_views_confirm_color_mismatch"
        in report["aggregate"]["reasons"]
    )


def test_dominant_group_is_not_satisfied_by_a_thin_matching_edge(
    tmp_path: Path,
) -> None:
    references = [
        _write_reference(tmp_path, "ref_a", shape="vertical", color=(35, 115, 48))
    ]
    render = _write_render(
        tmp_path,
        "view_a",
        shape="vertical",
        color=(155, 155, 155),
    )
    with Image.open(render["rgb"]) as opened:
        image = opened.convert("RGB")
    box = _shape_box("vertical")
    ImageDraw.Draw(image).rectangle(
        (box[0], box[1], box[0] + 2, box[3]),
        fill=(35, 115, 48),
    )
    image.save(render["rgb"])
    manifest, registry = _case(tmp_path, references, [render])

    report = compare_reference_renders(
        manifest,
        registry,
        view_mapping={"ref_a": "view_a"},
        minimum_comparable_views=1,
    )

    view = report["views"][0]
    group = view["material_color"]["trusted_evidence_group_recall"]["groups"][0]
    dominant = view["material_color"]["trusted_evidence_dominant_mass"]
    assert view["status"] == "FAIL"
    assert group["reference_color_share"] > 0.95
    assert group["observed_render_share"] < 0.10
    assert dominant["status"] == "FAIL"
    assert "trusted_dominant_family_mass_deficit" in view["reasons"]


def test_group_footprint_is_normalized_by_reference_foreground(
    tmp_path: Path,
) -> None:
    references = [
        _write_reference(
            tmp_path,
            "ref_a",
            shape="vertical",
            color=(35, 115, 48),
        )
    ]
    renders = [
        _write_render(
            tmp_path,
            "view_a",
            shape="vertical",
            color=(35, 115, 48),
        )
    ]
    manifest, registry = _case(tmp_path, references, renders)

    report = compare_reference_renders(
        manifest,
        registry,
        view_mapping={"ref_a": "view_a"},
        minimum_comparable_views=1,
    )

    group = report["views"][0]["material_color"]["trusted_evidence_group_recall"][
        "groups"
    ][0]
    assert group["reference_foreground_pixels"] > group["reference_evidence_weight"]
    assert group["reference_evidence_share"] == pytest.approx(
        group["reference_evidence_weight"] / group["reference_foreground_pixels"]
    )
    assert group["reference_group_share_basis"] == (
        "trusted_evidence_footprint_capped_color_share"
    )


def test_foreground_value_similarity_is_part_of_material_score(
    tmp_path: Path,
) -> None:
    references = [
        _write_reference(
            tmp_path,
            "ref_a",
            shape="vertical",
            color=(20, 85, 34),
        )
    ]
    matching = _write_render(
        tmp_path,
        "matching",
        shape="vertical",
        color=(20, 85, 34),
    )
    bright = _write_render(
        tmp_path,
        "bright",
        shape="vertical",
        color=(55, 235, 94),
    )
    matching_case = tmp_path / "matching_case"
    bright_case = tmp_path / "bright_case"
    matching_case.mkdir()
    bright_case.mkdir()
    manifest, matching_registry = _case(matching_case, references, [matching])
    _, bright_registry = _case(bright_case, references, [bright])

    matching_report = compare_reference_renders(
        manifest,
        matching_registry,
        view_mapping={"ref_a": "matching"},
        minimum_comparable_views=1,
    )
    bright_report = compare_reference_renders(
        manifest,
        bright_registry,
        view_mapping={"ref_a": "bright"},
        minimum_comparable_views=1,
    )

    matching_color = matching_report["views"][0]["material_color"]
    bright_color = bright_report["views"][0]["material_color"]
    assert matching_color["median_value_similarity"] == pytest.approx(1.0)
    assert bright_color["median_value_similarity"] < 0.5
    assert matching_color["score"] > bright_color["score"]


def test_multiview_source_exposure_can_resolve_bounded_value_only_reviews(
    tmp_path: Path,
) -> None:
    reference_colors = ((20, 85, 34), (25, 110, 44), (30, 125, 50))
    references = [
        _write_reference(
            tmp_path,
            f"ref_{index}",
            shape="vertical",
            color=color,
        )
        for index, color in enumerate(reference_colors)
    ]
    renders = [
        _write_render(
            tmp_path,
            f"render_{index}",
            shape="vertical",
            color=(40, 155, 62),
        )
        for index in range(len(references))
    ]
    manifest, registry = _case(tmp_path, references, renders)

    report = compare_reference_renders(
        manifest,
        registry,
        view_mapping={
            f"ref_{index}": f"render_{index}"
            for index in range(len(references))
        },
    )

    resolution = report["photometric_cohort_resolution"]
    assert resolution["status"] == "PASS"
    assert resolution["metrics"]["reference_value_span"] >= 0.08
    assert resolution["metrics"]["render_value_span"] == pytest.approx(0.0)
    assert report["aggregate"]["status"] == "PASS"
    assert report["aggregate"]["passed_view_count"] == 3
    assert all(view["status"] == "PASS" for view in report["views"])
    promoted = [
        view
        for view in report["views"]
        if "photometric_value_resolution" in view
    ]
    assert promoted
    assert all(
        view["photometric_value_resolution"]["status"] == "PASS"
        for view in promoted
    )


def test_multiview_photometric_cohort_does_not_excuse_large_value_error(
    tmp_path: Path,
) -> None:
    reference_colors = ((20, 65, 26), (20, 85, 34), (25, 105, 42))
    references = [
        _write_reference(
            tmp_path,
            f"ref_{index}",
            shape="vertical",
            color=color,
        )
        for index, color in enumerate(reference_colors)
    ]
    renders = [
        _write_render(
            tmp_path,
            f"render_{index}",
            shape="vertical",
            color=(60, 240, 96),
        )
        for index in range(len(references))
    ]
    manifest, registry = _case(tmp_path, references, renders)

    report = compare_reference_renders(
        manifest,
        registry,
        view_mapping={
            f"ref_{index}": f"render_{index}"
            for index in range(len(references))
        },
    )

    resolution = report["photometric_cohort_resolution"]
    assert resolution["status"] == "REJECTED"
    assert "PHOTOMETRIC_VALUE_OFFSET_EXCEEDS_BOUND" in resolution["reason_codes"]
    assert report["aggregate"]["status"] == "REVIEW"


def test_surface_texture_changes_visual_appearance_score_without_category_rules(
    tmp_path: Path,
) -> None:
    references = [
        _write_reference(
            tmp_path,
            "ref_a",
            shape="vertical",
            color=(35, 115, 48),
        )
    ]
    smooth = _write_render(
        tmp_path,
        "smooth",
        shape="vertical",
        color=(35, 115, 48),
    )
    textured = _write_render(
        tmp_path,
        "textured",
        shape="vertical",
        color=(35, 115, 48),
    )
    with Image.open(textured["rgb"]) as opened:
        textured_image = opened.convert("RGB")
    pixels = textured_image.load()
    left, top, right, bottom = _shape_box("vertical")
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            delta = 14 if (x + y) % 2 else -14
            pixels[x, y] = (
                max(0, min(255, 35 + delta)),
                max(0, min(255, 115 + delta)),
                max(0, min(255, 48 + delta)),
            )
    textured_image.save(textured["rgb"])

    smooth_case = tmp_path / "smooth_case"
    textured_case = tmp_path / "textured_case"
    smooth_case.mkdir()
    textured_case.mkdir()
    manifest, smooth_registry = _case(smooth_case, references, [smooth])
    _, textured_registry = _case(textured_case, references, [textured])
    smooth_report = compare_reference_renders(
        manifest,
        smooth_registry,
        view_mapping={"ref_a": "smooth"},
        minimum_comparable_views=1,
    )
    textured_report = compare_reference_renders(
        manifest,
        textured_registry,
        view_mapping={"ref_a": "textured"},
        minimum_comparable_views=1,
    )

    smooth_view = smooth_report["views"][0]
    textured_view = textured_report["views"][0]
    assert smooth_view["material_texture"]["score"] > 0.95
    assert textured_view["material_texture"]["score"] < 0.75
    assert (
        smooth_report["aggregate"]["material_appearance_score"]
        > textured_report["aggregate"]["material_appearance_score"]
    )
    assert smooth_report["aggregate"]["texture_comparable_view_count"] == 1


def test_dominant_mass_blocks_one_strong_view_until_area_recovers(
    tmp_path: Path,
) -> None:
    references = [
        _write_reference(tmp_path, "ref_a", shape="vertical", color=(35, 115, 48))
    ]
    render = _write_render(
        tmp_path,
        "view_a",
        shape="vertical",
        color=(35, 115, 48),
    )
    box = _shape_box("vertical")
    with Image.open(render["rgb"]) as opened:
        image = opened.convert("RGB")
    ImageDraw.Draw(image).rectangle(
        (box[2] - 9, box[1], box[2], box[3]),
        fill=(155, 155, 155),
    )
    image.save(render["rgb"])
    manifest, registry = _case(tmp_path, references, [render])

    failed = compare_reference_renders(
        manifest,
        registry,
        view_mapping={"ref_a": "view_a"},
    )

    view = failed["views"][0]
    group = view["material_color"]["trusted_evidence_group_recall"]["groups"][0]
    dominant = view["material_color"]["trusted_evidence_dominant_mass"]
    family = dominant["families"][0]
    assert group["recall"] == 1.0
    assert dominant["status"] == "FAIL"
    assert family["mass_recall"] < 0.80
    assert family["deficit_share"] > 0.08
    assert view["reasons"] == ["trusted_dominant_family_mass_deficit"]
    assert failed["aggregate"]["status"] == "FAIL"
    assert failed["aggregate"]["comparable_view_count"] == 1
    assert failed["aggregate"]["reasons"] == [
        "single_strong_view_confirms_dominant_family_mass_deficit"
    ]

    ImageDraw.Draw(image).rectangle(
        (box[2] - 9, box[1], box[2] - 4, box[3]),
        fill=(35, 115, 48),
    )
    image.save(render["rgb"])
    recovered = compare_reference_renders(
        manifest,
        registry,
        view_mapping={"ref_a": "view_a"},
        minimum_comparable_views=1,
    )
    recovered_dominant = recovered["views"][0]["material_color"][
        "trusted_evidence_dominant_mass"
    ]
    assert recovered_dominant["status"] == "PASS"
    assert recovered["aggregate"]["status"] == "PASS"


def test_small_accent_family_is_not_a_dominant_mass_failure(
    tmp_path: Path,
) -> None:
    references = [
        _write_reference(tmp_path, "ref_a", shape="vertical", color=(35, 115, 48))
    ]
    box = _shape_box("vertical")
    with Image.open(references[0]["image"]) as opened:
        reference_image = opened.convert("RGB")
    ImageDraw.Draw(reference_image).rectangle(
        (box[2] - 5, box[1], box[2], box[3]),
        fill=(170, 90, 35),
    )
    reference_image.save(references[0]["image"])
    evidence_path = Path(
        references[0]["palette_artifacts"]["normalized_evidence_audit"]
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["accepted_group_ids"].append("G02")
    evidence["groups"].append(
        {
            "group_id": "G02",
            "base_color": "orange",
            "accepted": True,
            "boxes": [
                {
                    "accepted": True,
                    "representative_srgb": [170, 90, 35],
                    "matching_pixel_count": 100,
                }
            ],
        }
    )
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    render = _write_render(
        tmp_path,
        "view_a",
        shape="vertical",
        color=(35, 115, 48),
    )
    with Image.open(render["rgb"]) as opened:
        render_image = opened.convert("RGB")
    ImageDraw.Draw(render_image).rectangle(
        (box[2] - 3, box[1], box[2], box[3]),
        fill=(170, 90, 35),
    )
    render_image.save(render["rgb"])
    manifest, registry = _case(tmp_path, references, [render])

    report = compare_reference_renders(
        manifest,
        registry,
        view_mapping={"ref_a": "view_a"},
        minimum_comparable_views=1,
    )

    dominant = report["views"][0]["material_color"]["trusted_evidence_dominant_mass"]
    by_key = {item["family_key"]: item for item in dominant["families"]}
    assert by_key["green"]["status"] == "PASS"
    assert by_key["orange_brown|red"]["status"] == "NOT_APPLICABLE"
    assert (
        "REFERENCE_SHARE_BELOW_DOMINANT_FLOOR"
        in by_key["orange_brown|red"]["reason_codes"]
    )
    assert report["aggregate"]["status"] == "PASS"


def test_missing_palette_evidence_fails_closed(tmp_path: Path) -> None:
    references = [
        _write_reference(
            tmp_path,
            "ref_a",
            shape="vertical",
            color=(35, 115, 48),
            usable_evidence=False,
        )
    ]
    renders = [_write_render(tmp_path, "view_a", shape="vertical", color=(35, 115, 48))]
    manifest, registry = _case(tmp_path, references, renders)

    report = compare_reference_renders(
        manifest,
        registry,
        view_mapping={"ref_a": "view_a"},
        minimum_comparable_views=1,
    )

    assert report["views"][0]["status"] == "UNSCORABLE"
    assert "palette_status_not_usable" in report["views"][0]["reasons"]
    assert report["aggregate"]["material_match_conclusion"] == "NOT_CONCLUSIVE"


def test_auto_match_rejects_equal_shape_ambiguity(tmp_path: Path) -> None:
    references = [
        _write_reference(tmp_path, "ref_a", shape="vertical", color=(35, 115, 48))
    ]
    renders = [
        _write_render(tmp_path, "view_a", shape="vertical", color=(35, 115, 48)),
        _write_render(tmp_path, "view_b", shape="vertical", color=(35, 115, 48)),
    ]
    manifest, registry = _case(tmp_path, references, renders)

    report = compare_reference_renders(manifest, registry, minimum_comparable_views=1)

    mapping = report["views"][0]["mapping"]
    assert mapping["selected_render_view_id"] is None
    assert "auto_match_margin_too_small" in mapping["reasons"]
    assert report["views"][0]["status"] == "UNSCORABLE"


def test_auto_match_accepts_unique_well_separated_shapes(tmp_path: Path) -> None:
    references = [
        _write_reference(tmp_path, "ref_tall", shape="vertical", color=(35, 115, 48)),
        _write_reference(tmp_path, "ref_wide", shape="horizontal", color=(35, 115, 48)),
    ]
    renders = [
        _write_render(tmp_path, "view_wide", shape="horizontal", color=(31, 99, 43)),
        _write_render(tmp_path, "view_tall", shape="vertical", color=(31, 99, 43)),
    ]
    manifest, registry = _case(tmp_path, references, renders)

    report = compare_reference_renders(manifest, registry)

    assert report["inputs"]["selected_view_mapping"] == {
        "ref_tall": "view_tall",
        "ref_wide": "view_wide",
    }
    assert report["aggregate"]["status"] == "PASS"
    assert all(view["mapping"]["margin"] > 0.055 for view in report["views"])


def test_auto_match_uses_global_one_to_one_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_masks = [Image.new("L", (1, 1), value) for value in (1, 2)]
    render_masks = [Image.new("L", (1, 1), value) for value in (3, 4)]
    scores = {
        (id(reference_masks[0]), id(render_masks[0])): 0.90,
        (id(reference_masks[0]), id(render_masks[1])): 0.89,
        (id(reference_masks[1]), id(render_masks[0])): 0.88,
        (id(reference_masks[1]), id(render_masks[1])): 0.50,
    }

    def fake_alignment(left: Image.Image, right: Image.Image) -> dict[str, float]:
        score = scores[(id(left), id(right))]
        return {"score": score, "difference_score": 1.0 - score}

    monkeypatch.setattr(comparison, "_alignment_metrics", fake_alignment)
    mapping, audits = comparison._auto_mapping(
        [
            {"view_id": "ref_a", "mask": reference_masks[0]},
            {"view_id": "ref_b", "mask": reference_masks[1]},
        ],
        [
            {"view_id": "view_a", "mask": render_masks[0]},
            {"view_id": "view_b", "mask": render_masks[1]},
        ],
        DEFAULT_THRESHOLDS,
    )

    # Both local greedy choices prefer view_a.  The maximum-weight injective
    # assignment instead reserves it for ref_b and still clears the global
    # edge-confidence margin for both references.
    assert mapping == {"ref_a": "view_b", "ref_b": "view_a"}
    assert all(audit["global_one_to_one_assignment"] for audit in audits.values())
    assert all(audit["global_assignment_margin"] > 0.055 for audit in audits.values())
    assert audits["ref_a"]["local_candidate_margin"] < 0.0


def test_partial_explicit_mapping_auto_completes_independent_pose(
    tmp_path: Path,
) -> None:
    references = [
        _write_reference(tmp_path, "ref_front", shape="vertical", color=(35, 115, 48)),
        _write_reference(tmp_path, "ref_iso", shape="horizontal", color=(35, 115, 48)),
    ]
    renders = [
        _write_render(tmp_path, "front", shape="vertical", color=(31, 99, 43)),
        _write_render(tmp_path, "iso", shape="horizontal", color=(31, 99, 43)),
    ]
    manifest, registry = _case(tmp_path, references, renders)

    report = compare_reference_renders(
        manifest,
        registry,
        view_mapping={"ref_front": "front"},
    )

    assert report["inputs"]["mapping_mode"] == "explicit_seeded_auto_completion"
    assert report["inputs"]["seeded_view_mapping"] == {"ref_front": "front"}
    assert report["inputs"]["selected_view_mapping"] == {
        "ref_front": "front",
        "ref_iso": "iso",
    }
    by_reference = {view["reference_view_id"]: view for view in report["views"]}
    assert by_reference["ref_front"]["mapping"]["mode"] == "explicit_locked"
    assert by_reference["ref_iso"]["mapping"]["mode"] == "auto_completion"
    assert by_reference["ref_iso"]["mapping"]["global_one_to_one_assignment"] is True
    assert by_reference["ref_iso"]["render_view_id"] == "iso"
    assert report["aggregate"]["status"] == "PASS"
    assert report["aggregate"]["reference_view_coverage_status"] == "PASS"
    assert report["aggregate"]["unmapped_reference_view_ids"] == []


def test_partial_mapping_ambiguous_pose_fails_closed(tmp_path: Path) -> None:
    references = [
        _write_reference(tmp_path, "ref_front", shape="vertical", color=(35, 115, 48)),
        _write_reference(tmp_path, "ref_top", shape="wide", color=(35, 115, 48)),
        _write_reference(tmp_path, "ref_iso", shape="horizontal", color=(35, 115, 48)),
    ]
    renders = [
        _write_render(tmp_path, "front", shape="vertical", color=(31, 99, 43)),
        _write_render(tmp_path, "top", shape="wide", color=(31, 99, 43)),
        _write_render(tmp_path, "iso_a", shape="horizontal", color=(31, 99, 43)),
        _write_render(tmp_path, "iso_b", shape="horizontal", color=(31, 99, 43)),
    ]
    manifest, registry = _case(tmp_path, references, renders)

    report = compare_reference_renders(
        manifest,
        registry,
        view_mapping={"ref_front": "front", "ref_top": "top"},
    )

    by_reference = {view["reference_view_id"]: view for view in report["views"]}
    iso = by_reference["ref_iso"]
    assert iso["render_view_id"] is None
    assert iso["status"] == "UNSCORABLE"
    assert "auto_match_margin_too_small" in iso["mapping"]["reasons"]
    assert "global_assignment_margin_too_small" in iso["mapping"]["reasons"]
    assert report["inputs"]["selected_view_mapping"] == {
        "ref_front": "front",
        "ref_top": "top",
    }
    assert report["aggregate"]["comparable_view_count"] == 2
    assert report["aggregate"]["passed_view_count"] == 2
    assert report["aggregate"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert report["aggregate"]["reference_view_coverage_status"] == "FAIL_CLOSED"
    assert report["aggregate"]["unmapped_reference_view_ids"] == ["ref_iso"]
    assert "reference_view_coverage_failed_closed" in report["aggregate"]["reasons"]
    assert (
        "not_all_reference_views_have_confident_one_to_one_mapping"
        in report["aggregate"]["reasons"]
    )


def test_registry_visible_pixel_mismatch_is_rejected(tmp_path: Path) -> None:
    references = [
        _write_reference(tmp_path, "ref_a", shape="vertical", color=(35, 115, 48))
    ]
    renders = [
        _write_render(
            tmp_path,
            "view_a",
            shape="vertical",
            color=(35, 115, 48),
            declared_pixel_multiplier=2.0,
        )
    ]
    manifest, registry = _case(tmp_path, references, renders)

    with pytest.raises(ComparisonInputError, match="inconsistent"):
        compare_reference_renders(
            manifest,
            registry,
            view_mapping={"ref_a": "view_a"},
            minimum_comparable_views=1,
        )


def test_cli_writes_machine_readable_report(tmp_path: Path) -> None:
    references = [
        _write_reference(tmp_path, "ref_a", shape="wide", color=(35, 115, 48))
    ]
    renders = [_write_render(tmp_path, "view_a", shape="wide", color=(31, 99, 43))]
    manifest, registry = _case(tmp_path, references, renders)
    output = tmp_path / "report.json"

    exit_code = main(
        [
            "--reference-manifest",
            str(manifest),
            "--rendered-registry",
            str(registry),
            "--output",
            str(output),
            "--map",
            "ref_a=view_a",
            "--minimum-comparable-views",
            "1",
            "--require-pass",
        ]
    )

    assert exit_code == EXIT_SUCCESS
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == "qwen-reference-render-comparison/v1"
    assert report["aggregate"]["status"] == "PASS"
    assert report["inputs"]["selected_view_mapping"] == {"ref_a": "view_a"}

    insufficient_output = tmp_path / "insufficient.json"
    insufficient_exit = main(
        [
            "--reference-manifest",
            str(manifest),
            "--rendered-registry",
            str(registry),
            "--output",
            str(insufficient_output),
            "--map",
            "ref_a=view_a",
            "--require-pass",
        ]
    )
    assert insufficient_exit == EXIT_REQUIRE_PASS_FAILED
    assert (
        json.loads(insufficient_output.read_text(encoding="utf-8"))["aggregate"][
            "status"
        ]
        == "INSUFFICIENT_EVIDENCE"
    )

    invalid_output = tmp_path / "invalid.json"
    invalid_exit = main(
        [
            "--reference-manifest",
            str(manifest),
            "--rendered-registry",
            str(registry),
            "--output",
            str(invalid_output),
            "--map",
            "unknown=view_a",
        ]
    )
    assert invalid_exit == EXIT_INPUT_ERROR
    assert not invalid_output.exists()
