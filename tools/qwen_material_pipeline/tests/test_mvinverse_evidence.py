from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from qwen_material_pipeline.mvinverse.evidence import (
    ALBEDO_COLOR_SPACE,
    MVInverseEvidenceError,
    SCHEMA_VERSION,
    build_mvinverse_evidence_from_manifest,
    validate_mvinverse_evidence,
    write_evidence_report,
)


def _group(
    group_id: str,
    *,
    family: str = "metal",
    color: str = "green",
    finish: str = "painted",
    boxes: list[list[int]] | None = None,
) -> dict:
    return {
        "group_id": group_id,
        "family_hint": family,
        "base_color": color,
        "finish_hint": finish,
        "visual_description": f"{color} {finish} {family} surface",
        "boxes": boxes or [[50, 50, 950, 950]],
        "confidence": 0.95,
    }


def _palette(view_id: str, groups: list[dict]) -> dict:
    return {
        "schema_version": "qwen-material-palette/v1",
        "source_view_id": view_id,
        "groups": groups,
    }


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _manifest(
    tmp_path: Path,
    view_palettes: list[tuple[str, dict | None]],
) -> tuple[dict, dict[str, int]]:
    views = []
    indices = {}
    for index, (view_id, palette) in enumerate(view_palettes):
        image = tmp_path / f"{view_id}.png"
        # Independent evidence must not be represented by byte-identical
        # source files, even in synthetic tests.
        Image.new("RGB", (30, 20), (index, 0, 0)).save(image)
        record: dict = {"id": view_id, "image": str(image)}
        if palette is not None:
            path = _write_json(tmp_path / f"{view_id}_palette.json", palette)
            record["palette_artifacts"] = {"normalized": str(path)}
        views.append(record)
        indices[view_id] = index
    return {"source_views": views}, indices


def _status_manifest(
    tmp_path: Path,
    view_palettes: list[tuple[str, str, dict | str]],
) -> tuple[dict, dict[str, int]]:
    views = []
    indices = {}
    for index, (view_id, status, palette) in enumerate(view_palettes):
        image = tmp_path / f"{view_id}.png"
        Image.new("RGB", (30, 20), (index, 0, 0)).save(image)
        palette_path = tmp_path / f"{view_id}_palette.json"
        if isinstance(palette, str):
            palette_path.write_text(palette, encoding="utf-8")
        else:
            _write_json(palette_path, palette)
        record: dict = {
            "id": view_id,
            "image": str(image),
            "palette_status": status,
            # Keeping this legacy artifact even for unusable views verifies
            # that the status, rather than the file contents, is authoritative.
            "palette_artifacts": {"normalized": str(palette_path)},
        }
        if status == "usable":
            record["palette_path"] = str(palette_path)
        else:
            record["palette_failure_artifact"] = str(palette_path)
        views.append(record)
        indices[view_id] = index
    return {"source_views": views}, indices


def _save_npy_frame(
    output: Path,
    index: int,
    *,
    albedo: np.ndarray,
    metallic: np.ndarray,
    roughness: np.ndarray,
) -> None:
    np.save(output / f"{index:03d}_albedo.npy", albedo)
    np.save(output / f"{index:03d}_metallic.npy", metallic)
    np.save(output / f"{index:03d}_roughness.npy", roughness)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_ledger(output: Path, manifest: dict) -> dict:
    modes = {
        "albedo": "RGB",
        "metallic": "L",
        "roughness": "L",
        "normal": "RGB",
        "shading": "RGB",
    }
    source_views = []
    maps = []
    for index, view in enumerate(manifest["source_views"]):
        source_views.append(
            {
                "index": index,
                "view_id": view["id"],
                "path": view["image"],
                "sha256": _sha256(Path(view["image"])),
                "size": [30, 20],
                "format": "PNG",
            }
        )
        for map_name, mode in modes.items():
            path = output / f"{index:03d}_{map_name}.png"
            if map_name == "albedo":
                Image.new(mode, (30, 20), (25, 153, 51)).save(path)
            elif map_name == "metallic":
                Image.new(mode, (30, 20), 26).save(path)
            elif map_name == "roughness":
                Image.new(mode, (30, 20), 82 + index * 5).save(path)
            elif map_name == "normal":
                Image.new(mode, (30, 20), (128, 128, 255)).save(path)
            else:
                Image.new(mode, (30, 20), (128, 128, 128)).save(path)
            maps.append(
                {
                    "index": index,
                    "view_id": view["id"],
                    "map": map_name,
                    "path": str(path),
                    "sha256": _sha256(path),
                    "size": [30, 20],
                    "mode": mode,
                }
            )
    output_set = hashlib.sha256(
        json.dumps(
            [
                {"index": item["index"], "map": item["map"], "sha256": item["sha256"]}
                for item in maps
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "schema_version": "qwen-mvinverse-inference-ledger/v1",
        "status": "SUCCESS",
        "inputs": {"source_views": source_views},
        "outputs": {
            "directory": str(output),
            "map_count": len(maps),
            "maps": maps,
            "output_set_sha256": output_set,
        },
    }


def test_two_unusable_views_are_skipped_without_losing_frame_alignment(
    tmp_path: Path,
) -> None:
    output = tmp_path / "maps"
    output.mkdir()
    canonical = _palette("top", [_group("G01")])
    manifest, indices = _status_manifest(
        tmp_path,
        [
            ("front", "unusable", '{"usable": false, "error": "truncated"}'),
            ("side", "unusable", '{"groups": ['),
            ("top", "usable", _palette("top", [_group("G07")])),
            ("iso", "usable", _palette("iso", [_group("G08")])),
        ],
    )
    ledger = _verified_ledger(output, manifest)

    report = build_mvinverse_evidence_from_manifest(
        output,
        manifest,
        canonical,
        frame_indices=indices,
        inference_ledger=ledger,
    )

    assert [(view["view_id"], view["frame_index"]) for view in report["views"]] == [
        ("front", 0),
        ("side", 1),
        ("top", 2),
        ("iso", 3),
    ]
    fused = report["groups"][0]
    assert fused["contributing_view_ids"] == ["top", "iso"]
    assert fused["unmatched_views"] == ["front", "side"]
    assert fused["suggestion"]["auto_parameter_eligible"] is True


def test_explicit_usable_view_with_malformed_palette_is_rejected(
    tmp_path: Path,
) -> None:
    output = tmp_path / "maps"
    output.mkdir()
    canonical = _palette("front", [_group("G01")])
    manifest, indices = _status_manifest(
        tmp_path,
        [("front", "usable", '{"usable": false, "error": "not a palette"}')],
    )

    with pytest.raises(MVInverseEvidenceError, match="invalid palette.*front"):
        build_mvinverse_evidence_from_manifest(
            output, manifest, canonical, frame_indices=indices
        )


def test_explicit_unusable_view_with_malformed_palette_is_not_read(
    tmp_path: Path,
) -> None:
    output = tmp_path / "maps"
    output.mkdir()
    canonical = _palette("side", [_group("G01")])
    manifest, indices = _status_manifest(
        tmp_path,
        [
            ("front", "unusable", '{"groups": ['),
            ("side", "usable", _palette("side", [_group("G07")])),
        ],
    )
    for index in indices.values():
        _save_npy_frame(
            output,
            index,
            albedo=np.full((20, 30, 3), (0.1, 0.6, 0.2), dtype=np.float32),
            metallic=np.full((20, 30), 0.1, dtype=np.float32),
            roughness=np.full((20, 30), 0.3, dtype=np.float32),
        )

    report = build_mvinverse_evidence_from_manifest(
        output, manifest, canonical, frame_indices=indices
    )

    assert [view["view_id"] for view in report["views"]] == ["front", "side"]
    assert report["groups"][0]["contributing_view_ids"] == ["side"]
    assert report["groups"][0]["unmatched_views"] == ["front"]


def test_unknown_explicit_palette_status_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "maps"
    output.mkdir()
    canonical = _palette("front", [_group("G01")])
    manifest, indices = _status_manifest(
        tmp_path,
        [("front", "failed", _palette("front", [_group("G07")]))],
    )

    with pytest.raises(MVInverseEvidenceError, match="palette_status"):
        build_mvinverse_evidence_from_manifest(
            output, manifest, canonical, frame_indices=indices
        )


def test_npy_is_preferred_color_filtered_and_fuses_dielectric(tmp_path: Path) -> None:
    output = tmp_path / "mvinverse"
    output.mkdir()
    canonical = _palette("ref_a", [_group("G01")])
    manifest, indices = _manifest(
        tmp_path,
        [
            ("ref_a", _palette("ref_a", [_group("G07")])),
            # Painted plastic is compatible with painted metal because both
            # describe the dielectric visible coating.
            ("ref_b", _palette("ref_b", [_group("G09", family="plastic")])),
        ],
    )
    for index, (metallic_value, roughness_value) in enumerate(
        ((0.10, 0.30), (0.12, 0.34))
    ):
        albedo = np.zeros((20, 30, 3), dtype=np.float32)
        albedo[:] = (0.75, 0.05, 0.04)  # red background inside a loose box
        albedo[5:15, 5:25] = (0.10, 0.60, 0.20)
        metallic = np.full((20, 30), 0.90, dtype=np.float32)
        roughness = np.full((20, 30), 0.90, dtype=np.float32)
        metallic[5:15, 5:25] = metallic_value
        roughness[5:15, 5:25] = roughness_value
        _save_npy_frame(
            output,
            index,
            albedo=albedo,
            metallic=metallic,
            roughness=roughness,
        )
        # A complete PNG set exists but must lose to the higher-precision NPY.
        Image.new("RGB", (30, 20), (0, 0, 0)).save(output / f"{index:03d}_albedo.png")
        Image.new("L", (30, 20), 255).save(output / f"{index:03d}_metallic.png")
        Image.new("L", (30, 20), 255).save(output / f"{index:03d}_roughness.png")

    report = build_mvinverse_evidence_from_manifest(
        output, manifest, canonical, frame_indices=indices
    )

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["inputs"]["albedo_color_space"] == ALBEDO_COLOR_SPACE
    assert report["inputs"]["frame_mapping_strategy"] == "explicit_argument"
    assert report["views"][0]["sources"]["albedo"]["format"] == "npy"
    first = report["views"][0]["groups"][0]
    assert first["region_pixels"] == 504
    assert first["color_matching_pixels"] == 200
    assert first["metallic"]["median"] == pytest.approx(0.10)
    fused = report["groups"][0]
    assert fused["contributing_view_ids"] == ["ref_a", "ref_b"]
    assert fused["distinct_view_count"] == 2
    assert fused["albedo"]["median"] == pytest.approx([0.10, 0.60, 0.20])
    assert fused["metallic"]["median"] == pytest.approx(0.11)
    assert fused["roughness"]["median"] == pytest.approx(0.32)
    assert fused["surface_class"] == "dielectric"
    assert fused["suggestion"] == {
        "decision": "preserve",
        "auto_parameter_eligible": False,
        "base_color_srgb": None,
        "metallic": None,
        "roughness": None,
        "reason_codes": ["unverified_inference_source"],
        "warning_codes": [],
    }
    assert report["summary"]["auto_parameter_group_count"] == 0
    assert report["summary"]["usd_modified"] is False


def test_required_sam_mask_never_falls_back_to_palette_boxes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "mvinverse"
    output.mkdir()
    canonical = _palette("front", [_group("G01")])
    manifest, indices = _manifest(
        tmp_path,
        [("front", _palette("front", [_group("G77")]))],
    )
    _save_npy_frame(
        output,
        0,
        albedo=np.full((20, 30, 3), (0.1, 0.6, 0.2), dtype=np.float32),
        metallic=np.full((20, 30), 0.1, dtype=np.float32),
        roughness=np.full((20, 30), 0.4, dtype=np.float32),
    )

    report = build_mvinverse_evidence_from_manifest(
        output,
        manifest,
        canonical,
        frame_indices=indices,
        masks={},
        require_explicit_masks=True,
    )

    group = report["views"][0]["groups"][0]
    assert group["accepted"] is False
    assert group["evidence_source"] is None
    assert group["boxes"] == []
    assert group["reason_codes"] == ["missing_authoritative_region_mask"]


def test_ambiguous_duplicate_color_groups_do_not_count_as_a_view(
    tmp_path: Path,
) -> None:
    output = tmp_path / "mvinverse"
    output.mkdir()
    canonical = _palette("ref_a", [_group("G01", color="white")])
    duplicate_palette = _palette(
        "ref_a",
        [
            _group("G07", color="white"),
            _group("G08", color="white"),
        ],
    )
    unique_palette = _palette("ref_b", [_group("G09", color="white")])
    manifest, indices = _manifest(
        tmp_path, [("ref_a", duplicate_palette), ("ref_b", unique_palette)]
    )
    for index in range(2):
        _save_npy_frame(
            output,
            index,
            albedo=np.full((20, 30, 3), 0.8, dtype=np.float32),
            metallic=np.full((20, 30), 0.05, dtype=np.float32),
            roughness=np.full((20, 30), 0.4, dtype=np.float32),
        )

    report = build_mvinverse_evidence_from_manifest(
        output, manifest, canonical, frame_indices=indices
    )

    first = report["views"][0]["groups"][0]
    assert first["association"] == {
        "status": "ambiguous",
        "candidate_group_ids": ["G07", "G08"],
        "matched_group_id": None,
    }
    fused = report["groups"][0]
    assert fused["ambiguous_views"] == ["ref_a"]
    assert fused["contributing_view_ids"] == ["ref_b"]
    assert fused["suggestion"]["decision"] == "preserve"
    assert fused["suggestion"]["reason_codes"] == [
        "unverified_inference_source",
        "insufficient_distinct_views",
    ]
    assert fused["suggestion"]["warning_codes"] == ["ambiguous_views_ignored"]


def test_bare_metal_prediction_that_looks_dielectric_is_preserved(
    tmp_path: Path,
) -> None:
    output = tmp_path / "mvinverse"
    output.mkdir()
    canonical = _palette(
        "ref_a", [_group("G01", family="metal", color="gray", finish="bare")]
    )
    manifest, indices = _manifest(
        tmp_path,
        [
            (
                "ref_a",
                _palette("ref_a", [_group("G07", color="gray", finish="brushed")]),
            ),
            (
                "ref_b",
                _palette("ref_b", [_group("G08", color="gray", finish="polished")]),
            ),
        ],
    )
    for index in range(2):
        _save_npy_frame(
            output,
            index,
            albedo=np.full((20, 30, 3), 0.5, dtype=np.float32),
            metallic=np.full((20, 30), 0.2, dtype=np.float32),
            roughness=np.full((20, 30), 0.3, dtype=np.float32),
        )

    report = build_mvinverse_evidence_from_manifest(
        output, manifest, canonical, frame_indices=indices
    )

    fused = report["groups"][0]
    assert fused["surface_class"] == "conductive"
    assert fused["metallic"]["median"] == pytest.approx(0.2)
    assert fused["suggestion"]["decision"] == "preserve"
    assert "conductive_metallicity_conflict" in fused["suggestion"]["reason_codes"]
    assert fused["suggestion"]["metallic"] is None


def test_explicit_masks_and_16bit_png_are_supported(tmp_path: Path) -> None:
    output = tmp_path / "mvinverse"
    output.mkdir()
    canonical = _palette(
        "ref_a", [_group("G01", family="plastic", color="black", finish="matte")]
    )
    manifest, indices = _manifest(tmp_path, [("ref_a", None), ("ref_b", None)])
    masks = {}
    for view_id, index in indices.items():
        Image.fromarray(np.full((20, 30, 3), 10, dtype=np.uint8), mode="RGB").save(
            output / f"{index:03d}_albedo.png"
        )
        Image.fromarray(np.zeros((20, 30), dtype=np.uint16)).save(
            output / f"{index:03d}_metallic.png"
        )
        Image.fromarray(np.full((20, 30), 32768, dtype=np.uint16)).save(
            output / f"{index:03d}_roughness.png"
        )
        # Source-size masks exercise nearest-neighbour resize after MVInverse's
        # divisible-by-14 image resize.
        mask = np.zeros((40, 60), dtype=np.uint8)
        mask[:, :30] = 255
        mask_path = tmp_path / f"{view_id}_mask.png"
        Image.fromarray(mask, mode="L").save(mask_path)
        masks[(view_id, "G01")] = mask_path

    report = build_mvinverse_evidence_from_manifest(
        output,
        manifest,
        canonical,
        masks=masks,
        frame_indices=indices,
    )

    first = report["views"][0]["groups"][0]
    assert first["association"]["status"] == "explicit_mask"
    assert first["mask"]["resized_to_output"] is True
    assert first["region_pixels"] == 300
    assert report["views"][0]["sources"]["roughness"]["dtype"] == "uint16"
    fused = report["groups"][0]
    assert fused["roughness"]["median"] == pytest.approx(32768 / 65535)
    assert fused["suggestion"]["decision"] == "preserve"
    assert fused["suggestion"]["reason_codes"] == ["unverified_inference_source"]


def test_global_npz_is_preferred_and_indexed(tmp_path: Path) -> None:
    output = tmp_path / "mvinverse"
    output.mkdir()
    canonical = _palette("ref_a", [_group("G01")])
    manifest, indices = _manifest(
        tmp_path,
        [
            ("ref_a", _palette("ref_a", [_group("G07")])),
            ("ref_b", _palette("ref_b", [_group("G08")])),
        ],
    )
    np.savez(
        output / "predictions.npz",
        albedo=np.full((2, 20, 30, 3), (0.1, 0.6, 0.2), dtype=np.float32),
        metallic=np.full((2, 20, 30, 1), 0.1, dtype=np.float32),
        roughness=np.stack(
            [
                np.full((20, 30, 1), 0.3, dtype=np.float32),
                np.full((20, 30, 1), 0.4, dtype=np.float32),
            ]
        ),
    )

    report = build_mvinverse_evidence_from_manifest(
        output, manifest, canonical, frame_indices=indices
    )

    assert report["views"][0]["sources"]["albedo"]["format"] == "npz"
    assert report["views"][1]["sources"]["roughness"]["key"] == "roughness"
    assert report["groups"][0]["roughness"]["median"] == pytest.approx(0.35)
    assert report["groups"][0]["suggestion"]["decision"] == "preserve"


def test_verified_ledger_is_required_for_auto_and_hash_bound(tmp_path: Path) -> None:
    output = tmp_path / "maps"
    output.mkdir()
    canonical = _palette("ref_a", [_group("G01")])
    manifest, indices = _manifest(
        tmp_path,
        [
            ("ref_a", _palette("ref_a", [_group("G07")])),
            ("ref_b", _palette("ref_b", [_group("G08")])),
        ],
    )
    ledger = _verified_ledger(output, manifest)

    report = build_mvinverse_evidence_from_manifest(
        output,
        manifest,
        canonical,
        frame_indices=indices,
        inference_ledger=ledger,
    )

    assert report["inputs"]["integrity_verified"] is True
    assert report["inputs"]["frame_mapping_strategy"] == "verified_inference_ledger"
    assert len(report["inputs"]["inference_ledger_sha256"]) == 64
    suggestion = report["groups"][0]["suggestion"]
    assert suggestion["decision"] == "auto"
    assert suggestion["auto_parameter_eligible"] is True
    assert suggestion["metallic"] == 0.0
    assert suggestion["roughness"] == pytest.approx((82 / 255 + 87 / 255) / 2)

    injected = output / "000_albedo.npy"
    np.save(injected, np.zeros((20, 30, 3), dtype=np.float32))
    with pytest.raises(MVInverseEvidenceError, match="contaminated"):
        build_mvinverse_evidence_from_manifest(
            output,
            manifest,
            canonical,
            frame_indices=indices,
            inference_ledger=ledger,
        )
    injected.unlink()

    (output / "000_metallic.png").write_bytes(b"tampered")
    with pytest.raises(MVInverseEvidenceError, match="map hash differs"):
        build_mvinverse_evidence_from_manifest(
            output,
            manifest,
            canonical,
            frame_indices=indices,
            inference_ledger=ledger,
        )


def test_float_arrays_outside_zero_one_fail_closed(tmp_path: Path) -> None:
    output = tmp_path / "mvinverse"
    output.mkdir()
    canonical = _palette("ref_a", [_group("G01")])
    manifest, indices = _manifest(
        tmp_path, [("ref_a", _palette("ref_a", [_group("G07")]))]
    )
    _save_npy_frame(
        output,
        0,
        albedo=np.full((20, 30, 3), 128.0, dtype=np.float32),
        metallic=np.full((20, 30), 0.1, dtype=np.float32),
        roughness=np.full((20, 30), 0.3, dtype=np.float32),
    )

    with pytest.raises(MVInverseEvidenceError, match="unambiguous 0..1"):
        build_mvinverse_evidence_from_manifest(
            output, manifest, canonical, frame_indices=indices
        )


def test_strict_validator_and_atomic_writer(tmp_path: Path) -> None:
    output = tmp_path / "mvinverse"
    output.mkdir()
    canonical = _palette("ref_a", [_group("G01")])
    manifest, indices = _manifest(
        tmp_path,
        [
            ("ref_a", _palette("ref_a", [_group("G07")])),
            ("ref_b", _palette("ref_b", [_group("G08")])),
        ],
    )
    for index in range(2):
        _save_npy_frame(
            output,
            index,
            albedo=np.full((20, 30, 3), (0.1, 0.6, 0.2), dtype=np.float32),
            metallic=np.full((20, 30), 0.1, dtype=np.float32),
            roughness=np.full((20, 30), 0.3, dtype=np.float32),
        )
    report = build_mvinverse_evidence_from_manifest(
        output, manifest, canonical, frame_indices=indices
    )
    destination = write_evidence_report(report, tmp_path / "nested" / "report.json")
    assert validate_mvinverse_evidence(json.loads(destination.read_text())) == report
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = Path(__file__).parents[1] / "schemas" / "mvinverse_evidence_v1.json"
    jsonschema.validate(report, json.loads(schema_path.read_text(encoding="utf-8")))

    malformed = copy.deepcopy(report)
    malformed["groups"][0]["suggestion"]["invented"] = True
    with pytest.raises(MVInverseEvidenceError, match="fields are invalid"):
        validate_mvinverse_evidence(malformed)
