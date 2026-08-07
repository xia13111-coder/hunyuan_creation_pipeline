from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from asset_pipeline.visual_materials.config import canonical_sha256
from asset_pipeline.visual_materials.orchestrator import (
    _ExactMdlCandidateCacheError,
    _archive_exact_mdl_candidate_cache_entry,
    _exact_mdl_material_application_contract,
    _validate_exact_mdl_candidate_cache,
)
from asset_pipeline.visual_materials.references import sha256_file


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _candidate_fixture(tmp_path: Path) -> dict[str, Any]:
    destination = tmp_path / "visual"
    candidate_id = "g01_01_deadbeef00"
    candidate_dir = destination / "visual_exact_mdl_tournament" / candidate_id
    render_dir = candidate_dir / "renders"
    render_dir.mkdir(parents=True)

    source = tmp_path / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    look = candidate_dir / "look.usda"
    look.write_text("#usda 1.0\n# candidate\n", encoding="utf-8")
    plan = {
        "schema_version": "qwen-material-plan/v1",
        "assignments": [
            {
                "part_id": "P0001",
                "material_id": "mdl:Base/Metals/Steel.mdl#Steel",
                "parameters": {},
            }
        ],
    }
    _write_json(candidate_dir / "plan.json", plan)

    occurrence_registry = {
        "schema_version": "qwen-material-parts/v1",
        "asset_usd": str(source.resolve()),
        "asset_sha256": sha256_file(source),
        "part_count": 1,
        "instance_root_count": 0,
        "parts": [{"part_id": "P0001"}],
    }
    apply_report = {
        "source_usd": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "registry_sha256": canonical_sha256(occurrence_registry),
        "plan_sha256": canonical_sha256(plan),
        "output_usd": str(look.resolve()),
        "output_sha256": sha256_file(look),
        "applied_count": 1,
        "face_subset_count": 0,
    }
    _write_json(candidate_dir / "apply_report.json", apply_report)

    registry = {
        "schema_version": "qwen-material-parts/v1",
        "asset_usd": str(look.resolve()),
        "asset_sha256": sha256_file(look),
        "part_count": 1,
        "instance_root_count": 0,
        "parts": [{"part_id": "P0001"}],
    }
    _write_json(candidate_dir / "part_registry.json", registry)

    render_files: dict[str, tuple[Path, Path]] = {}
    for view_id in ("right", "front"):
        rgb = render_dir / "rgb" / f"{view_id}.png"
        part_ids = render_dir / "part_ids" / f"{view_id}.png"
        rgb.parent.mkdir(parents=True, exist_ok=True)
        part_ids.parent.mkdir(parents=True, exist_ok=True)
        rgb.write_bytes(f"rgb-{view_id}".encode())
        part_ids.write_bytes(f"ids-{view_id}".encode())
        render_files[view_id] = (rgb.resolve(), part_ids.resolve())
    rendered_registry = {
        **registry,
        "render_set": {
            "resolution": [64, 64],
            "lighting_profile": "material-neutral",
            "requested_view_tokens": ["right", "front"],
            "analysis_up_axis": [0.0, 0.0, 1.0],
            "analysis_front_axis": [0.0, -1.0, 0.0],
            "views": [
                {
                    "view_id": view_id,
                    "rgb": str(render_files[view_id][0]),
                    "part_ids": str(render_files[view_id][1]),
                }
                for view_id in ("right", "front")
            ],
        },
    }
    rendered_registry_path = render_dir / "part_registry.rendered.json"
    _write_json(rendered_registry_path, rendered_registry)

    references: dict[str, Path] = {}
    for reference_id in ("front", "side"):
        image = tmp_path / f"{reference_id}.jpg"
        image.write_bytes(f"reference-{reference_id}".encode())
        references[reference_id] = image.resolve()
    reference_manifest = tmp_path / "reference_manifest.json"
    _write_json(
        reference_manifest,
        {
            "source_views": [
                {"id": reference_id, "image": str(references[reference_id])}
                for reference_id in ("front", "side")
            ]
        },
    )
    palette_fusion = tmp_path / "palette_fusion.json"
    _write_json(palette_fusion, {"schema_version": "palette-test/v1"})
    mapping = {"front": "right", "side": "front"}
    quality = {
        "schema_version": "qwen-reference-render-comparison/v1",
        "inputs": {
            "reference_manifest": str(reference_manifest.resolve()),
            "reference_manifest_sha256": sha256_file(reference_manifest),
            "rendered_registry": str(rendered_registry_path.resolve()),
            "rendered_registry_sha256": sha256_file(rendered_registry_path),
            "seeded_view_mapping": mapping,
            "selected_view_mapping": mapping,
            "comparison_scope": {
                "mode": "canonical_group_local",
                "target_group_id": "G01",
                "target_part_ids": ["P0001"],
                "target_entities": [{"entity_kind": "assignment", "part_id": "P0001"}],
                "reference_view_ids": ["front", "side"],
                "palette_fusion": str(palette_fusion.resolve()),
                "palette_fusion_sha256": sha256_file(palette_fusion),
            },
        },
        "aggregate": {"status": "PASS"},
        "views": [
            {
                "reference_view_id": reference_id,
                "render_view_id": render_id,
                "reference": {
                    "image": str(references[reference_id]),
                    "image_sha256": sha256_file(references[reference_id]),
                },
                "render": {
                    "image": str(render_files[render_id][0]),
                    "image_sha256": sha256_file(render_files[render_id][0]),
                    "part_ids": str(render_files[render_id][1]),
                    "part_ids_sha256": sha256_file(render_files[render_id][1]),
                },
            }
            for reference_id, render_id in mapping.items()
        ],
    }
    quality_path = candidate_dir / "reference_render_comparison.json"
    _write_json(quality_path, quality)

    return {
        "destination": destination,
        "candidate_id": candidate_id,
        "candidate_dir": candidate_dir,
        "source": source,
        "plan": plan,
        "occurrence_registry": occurrence_registry,
        "mapping": mapping,
        "reference_manifest": reference_manifest,
        "palette_fusion": palette_fusion,
        "rendered_registry_path": rendered_registry_path,
        "quality_path": quality_path,
        "look": look,
    }


def _validate(
    fixture: dict[str, Any],
    *,
    expected_mapping: dict[Any, Any] | None = None,
    expected_render_view_ids: list[Any] | None = None,
    expected_reference_view_ids: list[Any] | None = None,
    whole_asset_quality_path: Path | None = None,
) -> dict[str, Any]:
    return _validate_exact_mdl_candidate_cache(
        candidate_dir=fixture["candidate_dir"],
        candidate_id=fixture["candidate_id"],
        expected_plan=fixture["plan"],
        apply_asset=fixture["source"],
        occurrence_registry=fixture["occurrence_registry"],
        expected_applied_count=1,
        expected_face_subset_count=0,
        expected_mapping=(
            fixture["mapping"] if expected_mapping is None else expected_mapping
        ),
        expected_render_view_ids=(
            ["right", "front"]
            if expected_render_view_ids is None
            else expected_render_view_ids
        ),
        expected_reference_view_ids=(
            ["front", "side"]
            if expected_reference_view_ids is None
            else expected_reference_view_ids
        ),
        expected_render_resolution=64,
        expected_analysis_up_axis="z",
        expected_analysis_front_axis="-y",
        reference_manifest=fixture["reference_manifest"],
        palette_fusion=fixture["palette_fusion"],
        target_group_id="G01",
        target_part_ids=["P0001"],
        target_entities=[{"entity_kind": "assignment", "part_id": "P0001"}],
        whole_asset_quality_path=whole_asset_quality_path,
    )


def _write_whole_asset_quality(
    fixture: dict[str, Any],
    *,
    mapping: dict[str, str] | None = None,
) -> Path:
    """Derive the full-view guard from the fixture's hash-bound local report."""

    effective_mapping = fixture["mapping"] if mapping is None else mapping
    local_quality = json.loads(fixture["quality_path"].read_text(encoding="utf-8"))
    rendered_registry = json.loads(
        fixture["rendered_registry_path"].read_text(encoding="utf-8")
    )
    render_views = {
        view["view_id"]: view for view in rendered_registry["render_set"]["views"]
    }
    reference_manifest = json.loads(
        fixture["reference_manifest"].read_text(encoding="utf-8")
    )
    references = {
        source["id"]: Path(source["image"])
        for source in reference_manifest["source_views"]
    }
    whole_quality = copy.deepcopy(local_quality)
    whole_quality["inputs"]["seeded_view_mapping"] = dict(effective_mapping)
    whole_quality["inputs"]["selected_view_mapping"] = dict(effective_mapping)
    whole_quality["inputs"]["comparison_scope"] = {"mode": "whole_asset"}
    whole_quality["views"] = [
        {
            "reference_view_id": reference_id,
            "render_view_id": render_id,
            "reference": {
                "image": str(references[reference_id].resolve()),
                "image_sha256": sha256_file(references[reference_id]),
            },
            "render": {
                "image": str(Path(render_views[render_id]["rgb"]).resolve()),
                "image_sha256": sha256_file(Path(render_views[render_id]["rgb"])),
                "part_ids": str(
                    Path(render_views[render_id]["part_ids"]).resolve()
                ),
                "part_ids_sha256": sha256_file(
                    Path(render_views[render_id]["part_ids"])
                ),
            },
        }
        for reference_id, render_id in effective_mapping.items()
    ]
    path = (
        fixture["candidate_dir"]
        / "whole_asset_reference_render_comparison.json"
    )
    _write_json(path, whole_quality)
    return path


def _add_global_only_view(fixture: dict[str, Any]) -> dict[str, str]:
    """Add a tournament view which is absent from this candidate's group."""

    candidate_dir = fixture["candidate_dir"]
    render_dir = candidate_dir / "renders"
    rgb = render_dir / "rgb" / "top.png"
    part_ids = render_dir / "part_ids" / "top.png"
    rgb.write_bytes(b"rgb-top")
    part_ids.write_bytes(b"ids-top")

    rendered_registry_path = fixture["rendered_registry_path"]
    rendered_registry = json.loads(rendered_registry_path.read_text(encoding="utf-8"))
    render_set = rendered_registry["render_set"]
    render_set["requested_view_tokens"].append("top")
    render_set["views"].append(
        {
            "view_id": "top",
            "rgb": str(rgb.resolve()),
            "part_ids": str(part_ids.resolve()),
        }
    )
    _write_json(rendered_registry_path, rendered_registry)

    reference_manifest_path = fixture["reference_manifest"]
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    top_reference = reference_manifest_path.parent / "top.jpg"
    top_reference.write_bytes(b"reference-top")
    reference_manifest["source_views"].append(
        {"id": "top", "image": str(top_reference.resolve())}
    )
    _write_json(reference_manifest_path, reference_manifest)

    quality_path = fixture["quality_path"]
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["inputs"]["rendered_registry_sha256"] = sha256_file(rendered_registry_path)
    quality["inputs"]["reference_manifest_sha256"] = sha256_file(
        reference_manifest_path
    )
    _write_json(quality_path, quality)
    return {"front": "right", "side": "front", "top": "top"}


def _replace_cached_plan(
    fixture: dict[str, Any],
    plan: dict[str, Any],
) -> None:
    _write_json(fixture["candidate_dir"] / "plan.json", plan)
    apply_path = fixture["candidate_dir"] / "apply_report.json"
    apply_report = json.loads(apply_path.read_text(encoding="utf-8"))
    apply_report["plan_sha256"] = canonical_sha256(plan)
    _write_json(apply_path, apply_report)


def test_exact_mdl_candidate_cache_accepts_complete_hash_bound_bundle(
    tmp_path: Path,
) -> None:
    fixture = _candidate_fixture(tmp_path)

    cached = _validate(fixture)

    assert cached["plan"] == fixture["plan"]
    assert cached["apply_report"]["applied_count"] == 1
    assert cached["apply_report"]["face_subset_count"] == 0
    assert (
        cached["apply_report_path"]
        == (fixture["candidate_dir"] / "apply_report.json").resolve()
    )
    assert cached["cache_rebase"] is None
    assert cached["rendered_registry_file_sha256"] == sha256_file(
        fixture["rendered_registry_path"]
    )
    assert cached["whole_asset_quality_report"] is None


def test_exact_mdl_candidate_cache_accepts_hash_bound_whole_asset_guard(
    tmp_path: Path,
) -> None:
    fixture = _candidate_fixture(tmp_path)
    whole_quality_path = _write_whole_asset_quality(fixture)

    cached = _validate(
        fixture,
        whole_asset_quality_path=whole_quality_path,
    )

    assert cached["whole_asset_quality_report"]["inputs"]["comparison_scope"] == {
        "mode": "whole_asset"
    }
    assert cached["whole_asset_quality_report"]["inputs"][
        "selected_view_mapping"
    ] == fixture["mapping"]


@pytest.mark.parametrize(
    "tamper",
    (
        "mapping",
        "registry_hash",
        "missing_view",
        "render_hash",
    ),
)
def test_exact_mdl_candidate_cache_rejects_invalid_whole_asset_guard(
    tmp_path: Path,
    tamper: str,
) -> None:
    fixture = _candidate_fixture(tmp_path)
    whole_quality_path = _write_whole_asset_quality(fixture)
    whole_quality = json.loads(whole_quality_path.read_text(encoding="utf-8"))
    if tamper == "mapping":
        whole_quality["inputs"]["selected_view_mapping"] = {"front": "right"}
    elif tamper == "registry_hash":
        whole_quality["inputs"]["rendered_registry_sha256"] = "0" * 64
    elif tamper == "missing_view":
        whole_quality["views"].pop()
    elif tamper == "render_hash":
        whole_quality["views"][0]["render"]["image_sha256"] = "0" * 64
    _write_json(whole_quality_path, whole_quality)

    with pytest.raises(_ExactMdlCandidateCacheError):
        _validate(
            fixture,
            whole_asset_quality_path=whole_quality_path,
        )


def test_exact_mdl_candidate_cache_projects_global_mapping_to_group_scope(
    tmp_path: Path,
) -> None:
    fixture = _candidate_fixture(tmp_path)
    global_mapping = _add_global_only_view(fixture)

    cached = _validate(
        fixture,
        expected_mapping=global_mapping,
        expected_render_view_ids=["right", "front", "top"],
    )

    assert cached["quality_report"]["inputs"]["selected_view_mapping"] == {
        "front": "right",
        "side": "front",
    }


@pytest.mark.parametrize(
    "candidate_mapping",
    (
        {"front": "right"},
        {"front": "right", "side": "front", "top": "top"},
        {"front": "right", "side": "top"},
        {"front": "right", "side": 7},
        {"front": "right", "side": "front", "unknown": "top"},
    ),
)
@pytest.mark.parametrize(
    "mapping_field",
    ("selected_view_mapping", "seeded_view_mapping"),
)
def test_exact_mdl_candidate_cache_rejects_nonexact_scoped_mapping(
    tmp_path: Path,
    candidate_mapping: dict[str, Any],
    mapping_field: str,
) -> None:
    fixture = _candidate_fixture(tmp_path)
    global_mapping = _add_global_only_view(fixture)
    quality = json.loads(fixture["quality_path"].read_text(encoding="utf-8"))
    quality["inputs"][mapping_field] = candidate_mapping
    _write_json(fixture["quality_path"], quality)

    with pytest.raises(
        _ExactMdlCandidateCacheError,
        match="candidate quality view mapping differs",
    ):
        _validate(
            fixture,
            expected_mapping=global_mapping,
            expected_render_view_ids=["right", "front", "top"],
        )


@pytest.mark.parametrize(
    ("expected_mapping", "expected_reference_view_ids", "message"),
    (
        (
            {"front": "right"},
            ["front", "side"],
            "does not cover candidate scope",
        ),
        (
            {"front": "right", "side": 7},
            ["front", "side"],
            "current tournament view mapping is invalid",
        ),
        (
            {"front": "right", "side": "front"},
            ["front", 7],
            "expected reference-view scope is invalid",
        ),
        (
            {"front": "right", "side": "front"},
            ["front", "front"],
            "expected reference-view scope is invalid",
        ),
        (
            {"front": "right", "side": "front"},
            "front",
            "expected reference-view scope is invalid",
        ),
    ),
)
def test_exact_mdl_candidate_cache_rejects_invalid_expected_mapping_scope(
    tmp_path: Path,
    expected_mapping: dict[Any, Any],
    expected_reference_view_ids: Any,
    message: str,
) -> None:
    fixture = _candidate_fixture(tmp_path)

    with pytest.raises(_ExactMdlCandidateCacheError, match=message):
        _validate(
            fixture,
            expected_mapping=expected_mapping,
            expected_reference_view_ids=expected_reference_view_ids,
        )


@pytest.mark.parametrize(
    "tamper",
    (
        "planned_material",
        "apply_plan_hash",
        "look",
        "rendered_registry",
        "quality_registry_hash",
        "applied_count",
        "face_subset_count",
        "missing_quality",
    ),
)
def test_exact_mdl_candidate_cache_rejects_any_partial_or_mismatched_bundle(
    tmp_path: Path,
    tamper: str,
) -> None:
    fixture = _candidate_fixture(tmp_path)
    if tamper == "planned_material":
        fixture["plan"] = {
            **fixture["plan"],
            "assignments": [
                {
                    **fixture["plan"]["assignments"][0],
                    "material_id": "mdl:Base/Metals/Aluminum.mdl#Aluminum",
                }
            ],
        }
    elif tamper == "apply_plan_hash":
        path = fixture["candidate_dir"] / "apply_report.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["plan_sha256"] = "0" * 64
        _write_json(path, document)
    elif tamper == "look":
        fixture["look"].write_text("#usda 1.0\n# changed\n", encoding="utf-8")
    elif tamper == "rendered_registry":
        document = json.loads(
            fixture["rendered_registry_path"].read_text(encoding="utf-8")
        )
        document["unexpected"] = True
        _write_json(fixture["rendered_registry_path"], document)
    elif tamper == "quality_registry_hash":
        document = json.loads(fixture["quality_path"].read_text(encoding="utf-8"))
        document["inputs"]["rendered_registry_sha256"] = "0" * 64
        _write_json(fixture["quality_path"], document)
    elif tamper in {"applied_count", "face_subset_count"}:
        path = fixture["candidate_dir"] / "apply_report.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document[tamper] = 99
        _write_json(path, document)
    elif tamper == "missing_quality":
        fixture["quality_path"].unlink()

    with pytest.raises(_ExactMdlCandidateCacheError):
        _validate(fixture)


def test_exact_mdl_candidate_cache_rebases_diagnostic_only_plan_change(
    tmp_path: Path,
) -> None:
    fixture = _candidate_fixture(tmp_path)
    cached_plan = fixture["plan"]
    fixture["plan"] = {
        **cached_plan,
        "provenance": {
            "quality_report_sha256": "1" * 64,
            "quality_repair": {
                "pixel_count": 123,
                "recall": 0.453881,
            },
        },
        "diagnostics": {"replayed": True},
    }

    cached = _validate(fixture)

    assert cached["plan"] == fixture["plan"]
    assert cached["cached_plan"] == cached_plan
    rebase = cached["cache_rebase"]
    assert rebase["status"] == "RENDER_EQUIVALENT_PLAN_REBASE"
    assert rebase["cached_plan_sha256"] == canonical_sha256(cached_plan)
    assert rebase["expected_plan_sha256"] == canonical_sha256(fixture["plan"])
    assert (
        rebase["cached_material_application_contract_sha256"]
        == rebase["expected_material_application_contract_sha256"]
    )
    rebased_path = fixture["candidate_dir"] / "apply_report.cache_rebased.json"
    assert cached["apply_report_path"] == rebased_path
    assert rebased_path.is_file()
    assert cached["apply_report"]["plan_sha256"] == canonical_sha256(fixture["plan"])
    assert cached["apply_report"]["candidate_cache_rebase"] == rebase
    rebased_plan_path = fixture["candidate_dir"] / "plan.cache_rebased.json"
    assert cached["plan_path"] == rebased_plan_path
    assert json.loads(rebased_plan_path.read_text(encoding="utf-8")) == fixture["plan"]
    assert rebase["expected_plan"] == str(rebased_plan_path)
    assert rebase["expected_plan_file_sha256"] == sha256_file(rebased_plan_path)
    assert rebase["expected_plan_canonical_sha256"] == canonical_sha256(fixture["plan"])
    original_apply = json.loads(
        (fixture["candidate_dir"] / "apply_report.json").read_text(encoding="utf-8")
    )
    assert original_apply["plan_sha256"] == canonical_sha256(cached_plan)
    assert "candidate_cache_rebase" not in original_apply


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("material_id", "mdl:Base/Metals/Aluminum.mdl#Aluminum"),
        ("parameters", {"roughness": 0.5}),
        ("status", "approved"),
        ("confidence", 0.9),
        (
            "face_subsets",
            [
                {
                    "subset_name": "paint",
                    "material_id": "mdl:Base/Metals/Steel.mdl#Steel",
                    "face_indices": [0],
                }
            ],
        ),
        ("preserve_parent_material_binding", True),
        ("apply_action", "source_visual_preserve"),
        ("source_visual_material_prim_path", "/Looks/Steel"),
        ("source_visual_material_binding_sha256", "2" * 64),
    ),
)
def test_exact_mdl_candidate_cache_rejects_application_contract_change(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    fixture = _candidate_fixture(tmp_path)
    assignment = dict(fixture["plan"]["assignments"][0])
    assignment[field] = value
    fixture["plan"] = {
        **fixture["plan"],
        "assignments": [assignment],
    }

    with pytest.raises(
        _ExactMdlCandidateCacheError,
        match="material-application contract mismatch",
    ):
        _validate(fixture)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", "2.0"),
        ("mode", "different_policy_mode"),
        ("registry_asset_sha256", "3" * 64),
        ("registry_sha256", "4" * 64),
        ("asset_sha256", "5" * 64),
    ),
)
def test_exact_mdl_candidate_cache_rejects_plan_authorization_change(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    fixture = _candidate_fixture(tmp_path)
    if field == "schema_version":
        fixture["plan"] = {**fixture["plan"], field: value}
    else:
        fixture["plan"] = {
            **fixture["plan"],
            "provenance": {field: value},
        }

    with pytest.raises(
        _ExactMdlCandidateCacheError,
        match="material-application contract mismatch",
    ):
        _validate(fixture)


def test_material_application_contract_preserves_assignment_and_subset_order() -> None:
    first = {
        "part_id": "P0001",
        "material_id": "mdl:one",
        "status": "approved",
        "confidence": 1.0,
        "face_subsets": [
            {
                "subset_name": "first",
                "material_id": "mdl:first",
                "face_indices": [0],
            },
            {
                "subset_name": "second",
                "material_id": "mdl:second",
                "face_indices": [1],
            },
        ],
    }
    second = {
        "part_id": "P0002",
        "material_id": "mdl:two",
        "status": "approved",
        "confidence": 1.0,
    }
    baseline = {"schema_version": "1.0", "assignments": [first, second]}
    reversed_assignments = {
        **baseline,
        "assignments": [second, first],
    }
    reversed_subsets = copy.deepcopy(baseline)
    reversed_subsets["assignments"][0]["face_subsets"].reverse()

    baseline_contract = _exact_mdl_material_application_contract(baseline)
    assert (
        _exact_mdl_material_application_contract(reversed_assignments)
        != baseline_contract
    )
    assert (
        _exact_mdl_material_application_contract(reversed_subsets) != baseline_contract
    )


def test_exact_mdl_candidate_cache_rejects_source_preserve_tier_change(
    tmp_path: Path,
) -> None:
    fixture = _candidate_fixture(tmp_path)
    cached_plan = copy.deepcopy(fixture["plan"])
    cached_assignment = cached_plan["assignments"][0]
    cached_assignment.update(
        {
            "status": "policy_fallback",
            "confidence": 0.0,
            "evidence_views": [],
            "apply_action": "source_visual_preserve",
            "source_visual_material_prim_path": "/Looks/Steel",
            "source_visual_material_binding_sha256": "6" * 64,
            "provenance": {
                "tier": "source_visual_preserve",
                "reason_codes": ["SOURCE_VISUAL_MATERIAL_PRESENT"],
                "output_confidence_basis": ("policy fallback; not evidence confidence"),
                "sources": [],
            },
        }
    )
    _replace_cached_plan(fixture, cached_plan)
    fixture["plan"] = copy.deepcopy(cached_plan)
    fixture["plan"]["assignments"][0]["provenance"]["tier"] = "other_tier"

    with pytest.raises(
        _ExactMdlCandidateCacheError,
        match="material-application contract mismatch",
    ):
        _validate(fixture)


@pytest.mark.parametrize(
    ("location", "field", "value"),
    (
        ("assignment", "evidence_views", ["front"]),
        ("provenance", "tier", "different_tier"),
        ("provenance", "reason_codes", ["DIFFERENT_REASON"]),
        (
            "provenance",
            "output_confidence_basis",
            "different confidence basis",
        ),
        (
            "provenance",
            "sources",
            [
                {
                    "part_id": "P0002",
                    "source_status": "approved",
                    "source_confidence": 1.0,
                    "source_evidence_views": ["front"],
                }
            ],
        ),
    ),
)
def test_exact_mdl_candidate_cache_rejects_policy_fallback_authorization_change(
    tmp_path: Path,
    location: str,
    field: str,
    value: Any,
) -> None:
    fixture = _candidate_fixture(tmp_path)
    cached_plan = copy.deepcopy(fixture["plan"])
    cached_assignment = cached_plan["assignments"][0]
    cached_assignment.update(
        {
            "status": "policy_fallback",
            "confidence": 0.0,
            "evidence_views": [],
            "provenance": {
                "tier": "neutral_fallback",
                "reason_codes": ["NO_RELIABLE_EVIDENCE"],
                "output_confidence_basis": ("policy fallback; not evidence confidence"),
                "sources": [],
            },
        }
    )
    _replace_cached_plan(fixture, cached_plan)
    fixture["plan"] = copy.deepcopy(cached_plan)
    if location == "assignment":
        fixture["plan"]["assignments"][0][field] = value
    else:
        fixture["plan"]["assignments"][0]["provenance"][field] = value

    with pytest.raises(
        _ExactMdlCandidateCacheError,
        match="material-application contract mismatch",
    ):
        _validate(fixture)


def test_invalid_candidate_is_archived_as_one_reversible_bundle(
    tmp_path: Path,
) -> None:
    fixture = _candidate_fixture(tmp_path)
    fixture["quality_path"].unlink()

    with pytest.raises(_ExactMdlCandidateCacheError):
        _validate(fixture)
    archive = _archive_exact_mdl_candidate_cache_entry(
        destination=fixture["destination"],
        candidate_path=fixture["candidate_dir"],
        reason="quality report missing",
    )

    assert not fixture["candidate_dir"].exists()
    archived_candidate = (
        archive / "visual_exact_mdl_tournament" / fixture["candidate_id"]
    )
    assert archived_candidate.is_dir()
    assert (archived_candidate / "plan.json").is_file()
    assert (archived_candidate / "look.usda").is_file()
    manifest = json.loads(
        (archive / "archive_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "COMPLETED"
    assert manifest["reason"] == "quality report missing"
