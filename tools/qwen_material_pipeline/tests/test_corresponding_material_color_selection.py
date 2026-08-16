from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from qwen_material_pipeline.materials import corresponding_material_color_selection as selection
from qwen_material_pipeline.materials.corresponding_material_color import (
    AUDIT_SCHEMA_VERSION as COLOR_AUDIT_SCHEMA_VERSION,
)
from qwen_material_pipeline.usd.material_common import canonical_sha256


PAINT = "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte"
METAL = "mdl:Metals/Aluminum_Anodized.mdl#Aluminum_Anodized"
EXACT = "mdl:Metals/Steel_Stainless.mdl#Steel_Stainless"


def _source_plan() -> dict:
    return {
        "assignment_unit": "part_id",
        "assignments": [
            {"part_id": "P1", "material_id": EXACT, "provenance": {}},
            {"part_id": "P2", "material_id": PAINT, "provenance": {}},
            {"part_id": "P3", "material_id": METAL, "provenance": {}},
            {"part_id": "P4", "material_id": METAL, "provenance": {}},
        ],
        "provenance": {"fixture": True},
    }


def _candidate(source: dict, candidate_id: str, gain: float) -> selection.Candidate:
    plan = copy.deepcopy(source)
    by_id = {row["part_id"]: row for row in plan["assignments"]}
    for part_id in ("P2", "P3", "P4"):
        by_id[part_id]["parameters"] = {"diffuse_tint": [0.1 * gain] * 3}
        by_id[part_id]["provenance"] = {"candidate": candidate_id}
    scopes = [
        {
            "scope_id": "PART:P2",
            "member_part_ids": ["P2"],
            "material_id": PAINT,
            "target_srgb": [0.2, 0.3, 0.4],
        },
        {
            "scope_id": "COMPONENT:C1",
            "member_part_ids": ["P3", "P4"],
            "material_id": METAL,
            "target_srgb": [0.1, 0.5, 0.2],
        },
    ]
    return selection.Candidate(
        candidate_id=candidate_id,
        gain=gain,
        plan=plan,
        audit={"scopes": scopes},
        rendered_registry={"gain": gain},
        paths={},
        hashes={"rendered_registry": f"registry-{gain}", "asset": f"asset-{gain}"},
    )


def test_selects_actual_render_gain_per_scope_and_preserves_mdl_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_plan()
    candidates = [_candidate(source, "gain_1", 1.0), _candidate(source, "gain_2", 2.0)]

    def part_score(*, rendered_registry: dict, **_kwargs: object) -> dict:
        gain = rendered_registry["gain"]
        return {"appearance_score": 0.9 if gain == 2.0 else 0.4}

    def component_score(*, rendered_registry: dict, **_kwargs: object) -> dict:
        gain = rendered_registry["gain"]
        return {"appearance_score": 0.8 if gain == 1.0 else 0.5}

    monkeypatch.setattr(selection, "score_part_id_render", part_score)
    monkeypatch.setattr(selection, "score_component_render", component_score)
    output, audit = selection.select_render_calibrated_color_plan(
        source_plan=source,
        candidates=candidates,
        part_id_evidence={},
        spatial_mapping_report={},
    )
    before = {row["part_id"]: row for row in source["assignments"]}
    after = {row["part_id"]: row for row in output["assignments"]}
    assert {part_id: row["material_id"] for part_id, row in after.items()} == {
        part_id: row["material_id"] for part_id, row in before.items()
    }
    assert "parameters" not in after["P1"]
    assert after["P2"]["parameters"] == {"diffuse_tint": [0.2, 0.2, 0.2]}
    assert after["P3"]["parameters"] == after["P4"]["parameters"] == {
        "diffuse_tint": [0.1, 0.1, 0.1]
    }
    assert audit["summary"]["selected_gain_scope_counts"] == {
        "1.0": 1,
        "2.0": 1,
    }
    assert audit["summary"]["material_identity_change_count"] == 0


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_loader_binds_plan_apply_asset_and_render_registry(
    tmp_path: Path,
) -> None:
    source = _source_plan()
    candidate = _candidate(source, "gain_1", 1.0)
    root = tmp_path / "gain_1"
    plan_path = root / "part_id_material_plan.color.json"
    audit_path = root / "corresponding_material_color_audit.json"
    asset_path = root / "dtn100_colored.usda"
    apply_path = root / "apply_report.json"
    registry_path = root / "renders" / "part_registry.rendered.json"
    _write_json(plan_path, dict(candidate.plan))
    audit_unsigned = {
        "schema_version": COLOR_AUDIT_SCHEMA_VERSION,
        "source_plan_sha256": canonical_sha256(source),
        "output_plan_sha256": canonical_sha256(candidate.plan),
        "policy": {"linear_intensity_gain": 1.0},
        "scopes": list(candidate.audit["scopes"]),
    }
    _write_json(
        audit_path,
        {
            **audit_unsigned,
            "integrity": {"document_sha256": canonical_sha256(audit_unsigned)},
        },
    )
    asset_path.write_text("#usda 1.0\n", encoding="utf-8")
    _write_json(
        apply_path,
        {
            "plan_sha256": canonical_sha256(candidate.plan),
            "output_usd": str(asset_path),
            "output_sha256": _file_sha(asset_path),
        },
    )
    _write_json(
        registry_path,
        {"asset_usd": str(asset_path), "asset_sha256": _file_sha(asset_path)},
    )
    loaded = selection._load_candidate(root, canonical_sha256(source))
    assert loaded.gain == 1.0
    assert loaded.hashes["asset"] == _file_sha(asset_path)

    tampered = json.loads(registry_path.read_text())
    tampered["asset_sha256"] = "0" * 64
    _write_json(registry_path, tampered)
    with pytest.raises(
        selection.CorrespondingMaterialColorSelectionError,
        match="registry asset hash mismatch",
    ):
        selection._load_candidate(root, canonical_sha256(source))
