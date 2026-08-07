from __future__ import annotations

from pathlib import Path

from qwen_material_pipeline.usd.validation_common import (
    CHECK_LABELS,
    Audit,
    is_inside,
    local_asset_candidates,
    parameter_matches,
    report_records,
    scan_mdl_document,
    strip_mdl_comments,
    verify_mdl_textures,
)


def test_apply_report_accepts_subset_only_record_without_parent_material() -> None:
    report = {
        "applied_count": 1,
        "parent_binding_preserved_count": 1,
        "face_subset_count": 1,
        "applied": [
            {
                "part_id": "P0001",
                "prim_path": "/Asset/Mesh",
                "parent_binding_preserved": True,
                "source_visual_material_prim_path": "/Asset/Looks/Original",
                "parent_binding_relationship_authored": False,
                "face_subsets": [
                    {
                        "subset_name": "green",
                        "subset_prim_path": "/Asset/Mesh/green",
                        "material_id": "MAT_GREEN",
                        "material_prim_path": "/Asset/QwenLooks/Green",
                        "mdl_path": "/materials/green.mdl",
                        "subidentifier": "Green",
                        "parameters": {},
                        "face_indices": [0],
                    }
                ],
            }
        ],
    }
    audit = Audit()
    applied, materials = report_records(
        report,
        {"P0001": {"part_id": "P0001", "prim_path": "/Asset/Mesh"}},
        audit,
    )
    result = audit.to_report({})

    assert result["status"] == "PASS"
    assert applied["/Asset/Mesh"]["parent_binding_preserved"] is True
    assert set(materials) == {"/Asset/QwenLooks/Green"}


def test_apply_report_rejects_ambiguous_subset_only_parent_material() -> None:
    report = {
        "applied_count": 1,
        "parent_binding_preserved_count": 1,
        "face_subset_count": 0,
        "applied": [
            {
                "part_id": "P0001",
                "prim_path": "/Asset/Mesh",
                "parent_binding_preserved": True,
                "source_visual_material_prim_path": None,
                "parent_binding_relationship_authored": False,
                "material_prim_path": "/Asset/QwenLooks/Unexpected",
                "face_subsets": [],
            }
        ],
    }
    audit = Audit()
    report_records(
        report,
        {"P0001": {"part_id": "P0001", "prim_path": "/Asset/Mesh"}},
        audit,
    )
    result = audit.to_report({})

    assert result["status"] == "FAIL"
    messages = [
        failure["message"]
        for check in result["checks"]
        for failure in check["failures"]
    ]
    assert any(
        "ambiguously contains parent material fields" in item for item in messages
    )
    assert any("requires at least one face subset" in item for item in messages)


def test_mdl_scanner_finds_runtime_textures_and_separates_thumbnails() -> None:
    source = r"""
        // texture_2d("ignored-line.png")
        /* texture_2d("ignored-block.png") */
        texture_2d("../textures/base color.png", ::tex::gamma_srgb);
        texture_2d();
        let name = "// this is inside a string";
        ::anno::thumbnail("./.thumbs/preview.png");
        texture_2d("../textures/normal.png", ::tex::gamma_linear);
        texture_2d("../textures/normal.png", ::tex::gamma_linear);
    """

    textures, thumbnails = scan_mdl_document(source)

    assert textures == ["../textures/base color.png", "../textures/normal.png"]
    assert thumbnails == ["./.thumbs/preview.png"]


def test_comment_stripper_preserves_newlines_and_comment_markers_in_strings() -> None:
    source = 'texture_2d("folder//asset.png"); // ignored\n/* block\ncomment */\n'

    stripped = strip_mdl_comments(source)

    assert stripped.count("\n") == source.count("\n")
    assert '"folder//asset.png"' in stripped
    assert "ignored" not in stripped
    assert "comment" not in stripped


def test_local_asset_candidates_resolve_relative_and_tiled_paths(
    tmp_path: Path,
) -> None:
    material_dir = tmp_path / "materials"
    texture_dir = tmp_path / "textures"
    material_dir.mkdir()
    texture_dir.mkdir()
    owner = material_dir / "paint.mdl"
    owner.write_text("mdl 1.7;", encoding="utf-8")
    first = texture_dir / "paint.1001.png"
    second = texture_dir / "paint.1002.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    assert local_asset_candidates("../textures/paint.1001.png", owner) == [
        first.resolve()
    ]
    assert local_asset_candidates("../textures/paint.<UDIM>.png", owner) == [
        first.resolve(),
        second.resolve(),
    ]
    assert local_asset_candidates("https://example.invalid/paint.png", owner) == []


def test_inside_bundle_follows_symlinks(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    outside = tmp_path / "outside.png"
    bundle.mkdir()
    outside.write_bytes(b"outside")
    link = bundle / "link.png"
    link.symlink_to(outside)

    assert is_inside(bundle / "nested.png", bundle)
    assert not is_inside(link, bundle)


def test_parameter_matching_handles_usd_float_rounding() -> None:
    assert parameter_matches(0.03434000164270401, 0.03434)
    assert parameter_matches((0.1, 0.2, 0.3), [0.1, 0.2, 0.3])
    assert parameter_matches(False, False)
    assert not parameter_matches(True, 1.0)
    assert not parameter_matches((0.1, 0.2, 0.4), [0.1, 0.2, 0.3])


def test_runtime_texture_is_required_but_thumbnail_is_optional(tmp_path: Path) -> None:
    material_dir = tmp_path / "materials"
    material_dir.mkdir()
    mdl = material_dir / "paint.mdl"
    mdl.write_text(
        """
        mdl 1.7;
        texture_2d("../textures/runtime.png");
        ::anno::thumbnail("../thumbs/optional.png");
        """,
        encoding="utf-8",
    )

    missing_audit = Audit()
    verify_mdl_textures(tmp_path, missing_audit)
    missing_report = missing_audit.to_report({})
    texture_check = next(
        check for check in missing_report["checks"] if check["id"] == "mdl_textures"
    )
    assert texture_check["status"] == "FAIL"
    assert missing_report["summary"]["warning_count"] == 1

    texture_dir = tmp_path / "textures"
    texture_dir.mkdir()
    (texture_dir / "runtime.png").write_bytes(b"runtime")
    present_audit = Audit()
    verify_mdl_textures(tmp_path, present_audit)
    present_report = present_audit.to_report({})
    texture_check = next(
        check for check in present_report["checks"] if check["id"] == "mdl_textures"
    )
    assert texture_check["status"] == "PASS"
    assert texture_check["metrics"]["missing_optional_thumbnail_count"] == 1


def test_audit_emits_machine_readable_pass_and_fail_status() -> None:
    passing = Audit().to_report({"fixture": True})
    assert passing["status"] == "PASS"
    assert passing["overall_pass"] is True
    assert passing["summary"]["passed_check_count"] == len(CHECK_LABELS)

    audit = Audit()
    audit.fail("geometry", "changed", context={"prim_path": "/Asset/Mesh"})
    failing = audit.to_report({"fixture": True})
    assert failing["status"] == "FAIL"
    assert failing["overall_pass"] is False
    assert failing["summary"]["failure_count"] == 1
    geometry = next(check for check in failing["checks"] if check["id"] == "geometry")
    assert geometry["status"] == "FAIL"
    assert geometry["failures"][0]["context"]["prim_path"] == "/Asset/Mesh"
