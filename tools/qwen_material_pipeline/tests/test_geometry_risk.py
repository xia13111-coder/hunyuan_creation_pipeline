from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from qwen_material_pipeline.evidence.geometry import (
    GeometryRiskError,
    GeometryRiskPolicy,
    build_geometry_risk,
    validate_geometry_risk,
    write_geometry_risk,
)


def _documents(tmp_path: Path) -> tuple[dict, dict, Path]:
    asset = tmp_path / "asset.usda"
    asset.write_bytes(b"#usda 1.0\n")
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    part_specs = [
        ("P0001", "/Asset/Safe", 10, 511, 1, 127),
        ("P0002", "/Asset/Welded", 20, 2, 2, 4),
        ("P0003", "/Asset/Raw", 30, 512, 1, 4),
        ("P0004", "/Asset/Patches", 40, 4, 1, 128),
    ]
    face_parts = [
        {
            "part_id": part_id,
            "prim_path": prim_path,
            "face_count": face_count,
            "raw_topology_component_count": raw_count,
            "welded_topology_component_count": welded_count,
            "surface_patch_count": patch_count,
            "evidence": f"parts/{part_id}.json",
        }
        for part_id, prim_path, face_count, raw_count, welded_count, patch_count in part_specs
    ]
    registry_parts = [
        {
            "part_id": part_id,
            "prim_path": prim_path,
            "face_count": face_count,
            "renders": [],
        }
        for part_id, prim_path, face_count, *_ in part_specs
    ]
    face_manifest = {
        "schema_version": "qwen-face-region-evidence/v1",
        "asset_usd": str(asset),
        "asset_sha256": digest,
        "part_count": len(face_parts),
        "face_count": sum(part["face_count"] for part in face_parts),
        "welded_topology_component_count": sum(
            part["welded_topology_component_count"] for part in face_parts
        ),
        "surface_patch_count": sum(part["surface_patch_count"] for part in face_parts),
        "parts": face_parts,
        "source_usd_sha256_before": digest,
        "source_usd_sha256_after": digest,
        "source_usd_unchanged": True,
    }
    rendered_registry = {
        "schema_version": "qwen-material-parts/v1",
        "asset_usd": str(asset),
        "asset_sha256": digest,
        "part_count": len(registry_parts),
        "parts": registry_parts,
        "render_set": {"asset_usd": str(asset), "views": []},
    }
    return face_manifest, rendered_registry, asset


def test_default_policy_flags_each_boundary_and_keeps_below_threshold_safe(
    tmp_path: Path,
) -> None:
    face_manifest, rendered_registry, _ = _documents(tmp_path)

    report = build_geometry_risk(face_manifest, rendered_registry)

    assert report["schema_version"] == "qwen-geometry-uniform-material-risk/v1"
    assert report["source_usd_unchanged"] is True
    by_id = {part["part_id"]: part for part in report["parts"]}
    assert by_id["P0001"]["risk"]["multi_material_risk"] is False
    assert by_id["P0001"]["reason_codes"] == []
    assert by_id["P0002"]["reason_codes"] == ["multiple_welded_topology_components"]
    assert by_id["P0003"]["reason_codes"] == ["high_raw_topology_component_count"]
    assert by_id["P0004"]["reason_codes"] == ["high_surface_patch_count"]
    assert by_id["P0004"]["risk"]["multi_material_risk"] is False
    assert report["summary"] == {
        "part_count": 4,
        "face_count": 100,
        "multi_material_risk_part_count": 2,
        "no_detected_multi_material_risk_part_count": 2,
        "multi_material_risk_part_ids": ["P0002", "P0003"],
        "reason_code_counts": {
            "multiple_welded_topology_components": 1,
            "high_raw_topology_component_count": 1,
            "high_surface_patch_count": 1,
        },
    }


def test_all_reasons_are_reported_in_stable_order(tmp_path: Path) -> None:
    face_manifest, rendered_registry, _ = _documents(tmp_path)
    face_manifest["parts"][0].update(
        {
            "raw_topology_component_count": 600,
            "welded_topology_component_count": 3,
            "surface_patch_count": 200,
        }
    )
    face_manifest["welded_topology_component_count"] += 2
    face_manifest["surface_patch_count"] += 73

    part = build_geometry_risk(face_manifest, rendered_registry)["parts"][0]

    assert part["risk"]["multi_material_risk"] is True
    assert part["reason_codes"] == [
        "multiple_welded_topology_components",
        "high_raw_topology_component_count",
        "high_surface_patch_count",
    ]


def test_custom_policy_is_honored_and_validated(tmp_path: Path) -> None:
    face_manifest, rendered_registry, _ = _documents(tmp_path)
    policy = GeometryRiskPolicy(
        maximum_welded_topology_component_count=4,
        raw_topology_component_risk_threshold=1024,
        surface_patch_risk_threshold=256,
    )

    report = build_geometry_risk(face_manifest, rendered_registry, policy=policy)

    assert report["summary"]["multi_material_risk_part_count"] == 0
    with pytest.raises(GeometryRiskError, match="positive integer"):
        build_geometry_risk(
            face_manifest,
            rendered_registry,
            policy=GeometryRiskPolicy(raw_topology_component_risk_threshold=0),
        )


def test_step_brep_patch_fragmentation_is_advisory_when_welded_once(
    tmp_path: Path,
) -> None:
    face_manifest, rendered_registry, _ = _documents(tmp_path)
    # These shapes mirror ordinary STEP/BREP tessellation: many raw index
    # islands and normal-coherent patches collapse to one welded component.
    face_manifest["parts"][0].update(
        {
            "raw_topology_component_count": 45,
            "surface_patch_count": 180,
        }
    )
    face_manifest["parts"][2].update(
        {
            "raw_topology_component_count": 276,
            "surface_patch_count": 193,
        }
    )
    face_manifest["surface_patch_count"] = sum(
        part["surface_patch_count"] for part in face_manifest["parts"]
    )

    by_id = {
        part["part_id"]: part
        for part in build_geometry_risk(face_manifest, rendered_registry)["parts"]
    }

    for part_id in ("P0001", "P0003"):
        assert by_id[part_id]["metrics"]["welded_topology_component_count"] == 1
        assert by_id[part_id]["reason_codes"] == ["high_surface_patch_count"]
        assert by_id[part_id]["risk"]["multi_material_risk"] is False


def test_extreme_raw_fragmentation_remains_a_hard_fail_closed_signal(
    tmp_path: Path,
) -> None:
    face_manifest, rendered_registry, _ = _documents(tmp_path)

    part = build_geometry_risk(face_manifest, rendered_registry)["parts"][2]

    assert part["metrics"] == {
        "raw_topology_component_count": 512,
        "welded_topology_component_count": 1,
        "surface_patch_count": 4,
    }
    assert part["reason_codes"] == ["high_raw_topology_component_count"]
    assert part["risk"]["multi_material_risk"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda face, registry: face.update(schema_version="unsupported"),
            "schema_version",
        ),
        (
            lambda face, registry: face.update(source_usd_unchanged=False),
            "source_usd_unchanged",
        ),
        (
            lambda face, registry: registry.update(schema_version="unsupported"),
            "schema_version",
        ),
    ],
)
def test_schema_and_readonly_contract_fail_closed(
    tmp_path: Path, mutation, message: str
) -> None:
    face_manifest, rendered_registry, _ = _documents(tmp_path)
    mutation(face_manifest, rendered_registry)

    with pytest.raises(GeometryRiskError, match=message):
        build_geometry_risk(face_manifest, rendered_registry)


def test_asset_path_mismatch_fails_closed(tmp_path: Path) -> None:
    face_manifest, rendered_registry, _ = _documents(tmp_path)
    other = tmp_path / "other.usda"
    other.write_bytes(b"#usda 1.0\n")
    rendered_registry["asset_usd"] = str(other)

    with pytest.raises(GeometryRiskError, match="asset path mismatch"):
        build_geometry_risk(face_manifest, rendered_registry)


def test_declared_or_current_asset_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    face_manifest, rendered_registry, asset = _documents(tmp_path)
    rendered_registry["asset_sha256"] = "0" * 64
    with pytest.raises(GeometryRiskError, match="asset hash mismatch"):
        build_geometry_risk(face_manifest, rendered_registry)

    face_manifest, rendered_registry, asset = _documents(tmp_path)
    asset.write_bytes(b"#usda 1.0\n# changed\n")
    with pytest.raises(GeometryRiskError, match="evidence is stale"):
        build_geometry_risk(face_manifest, rendered_registry)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda face, registry: registry["parts"].pop(),
            "does not exactly cover",
        ),
        (
            lambda face, registry: registry["parts"][0].update(
                prim_path="/Asset/Wrong"
            ),
            "prim_path mismatch",
        ),
        (
            lambda face, registry: registry["parts"][0].update(face_count=999),
            "face_count mismatch",
        ),
    ],
)
def test_part_identity_and_face_count_require_exact_cover(
    tmp_path: Path, mutation, message: str
) -> None:
    face_manifest, rendered_registry, _ = _documents(tmp_path)
    mutation(face_manifest, rendered_registry)
    if len(rendered_registry["parts"]) != rendered_registry["part_count"]:
        rendered_registry["part_count"] = len(rendered_registry["parts"])

    with pytest.raises(GeometryRiskError, match=message):
        build_geometry_risk(face_manifest, rendered_registry)


def test_manifest_summary_counts_are_checked(tmp_path: Path) -> None:
    face_manifest, rendered_registry, _ = _documents(tmp_path)
    face_manifest["face_count"] += 1

    with pytest.raises(GeometryRiskError, match="face_count does not match"):
        build_geometry_risk(face_manifest, rendered_registry)


def test_render_readonly_audit_must_be_complete_and_matching(tmp_path: Path) -> None:
    face_manifest, rendered_registry, _ = _documents(tmp_path)
    rendered_registry["render_set"]["source_usd_unchanged"] = True

    with pytest.raises(GeometryRiskError, match="source_usd_sha256_before"):
        build_geometry_risk(face_manifest, rendered_registry)


def test_path_inputs_write_atomically_and_validate_round_trip(tmp_path: Path) -> None:
    face_manifest, rendered_registry, _ = _documents(tmp_path)
    face_path = tmp_path / "face.json"
    registry_path = tmp_path / "rendered.json"
    face_path.write_text(json.dumps(face_manifest), encoding="utf-8")
    registry_path.write_text(json.dumps(rendered_registry), encoding="utf-8")

    report = build_geometry_risk(face_path, registry_path)
    output = write_geometry_risk(report, tmp_path / "nested" / "risk.json")

    assert output.is_file()
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == report
    assert validate_geometry_risk(written) == report
    assert report["face_region_manifest"] == str(face_path.resolve())
    assert report["rendered_registry"] == str(registry_path.resolve())
    assert not list(output.parent.glob(".risk.json.tmp-*"))


def test_report_validation_rejects_tampered_risk_and_summary(tmp_path: Path) -> None:
    face_manifest, rendered_registry, _ = _documents(tmp_path)
    report = build_geometry_risk(face_manifest, rendered_registry)

    tampered = copy.deepcopy(report)
    tampered["parts"][0]["risk"]["multi_material_risk"] = True
    with pytest.raises(GeometryRiskError, match="inconsistent"):
        validate_geometry_risk(tampered)

    tampered = copy.deepcopy(report)
    tampered["summary"]["multi_material_risk_part_count"] = 0
    with pytest.raises(GeometryRiskError, match="summary is inconsistent"):
        validate_geometry_risk(tampered)

    tampered = copy.deepcopy(report)
    tampered["parts"][3]["reason_codes"] = []
    with pytest.raises(GeometryRiskError, match="reason_codes do not match"):
        validate_geometry_risk(tampered)
