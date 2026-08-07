from __future__ import annotations

import builtins
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from asset_pipeline.visual_materials.bundled_projects import (
    match_bundled_project,
    validate_bundled_acceptance_evidence,
)
from asset_pipeline.visual_materials.orchestrator import (
    _bundled_project_apply_command,
)
try:
    from qwen_material_pipeline.projects.dtn100.plan import build_plan
except ModuleNotFoundError:  # Private sealed projects are absent in source releases.
    build_plan = None
from qwen_material_pipeline.usd.registry import (
    SOURCE_MATERIAL_BIND_SUBSETS_FIELD,
    SOURCE_SUBSET_HASH_FIELD,
    source_material_bind_subset_sha256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_dtn100_bundle() -> None:
    if build_plan is None:
        pytest.skip("private DTN100 sealed project is not part of the source release")


def _topology_sha256(parts: list[dict[str, object]]) -> str:
    identities = sorted(
        (
            {
                "prim_path": item["prim_path"],
                "point_count": item["point_count"],
                "face_count": item["face_count"],
            }
            for item in parts
        ),
        key=lambda item: str(item["prim_path"]),
    )
    payload = json.dumps(
        identities,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, object]:
    source_cad = tmp_path / "assembly.stp"
    source_cad.write_bytes(b"stable-cad")
    reference_paths = []
    for index in range(4):
        path = tmp_path / f"reference-{index}.jpg"
        path.write_bytes(f"photo-{index}".encode())
        reference_paths.append(path)

    material_root = tmp_path / "NVIDIA" / "Materials"
    mdl_path = material_root / "Base" / "Fixture.mdl"
    mdl_path.parent.mkdir(parents=True)
    mdl_path.write_text(
        'mdl 1.0;\nexport material Fixture() = material();\n'
        'export texture_2d FixtureTexture() = '
        'texture_2d("./Fixture/fixture.png");\n',
        encoding="utf-8",
    )
    texture_path = material_root / "Base" / "Fixture" / "fixture.png"
    texture_path.parent.mkdir(parents=True)
    texture_path.write_bytes(b"sealed-texture")
    isaac_root = tmp_path / "isaacsim"
    (isaac_root / "VERSION").parent.mkdir(parents=True)
    (isaac_root / "VERSION").write_text("5.0.0-test\n", encoding="utf-8")
    helper_path = isaac_root / "kit" / "mdl" / "core" / "Base" / "Helper.mdl"
    helper_path.parent.mkdir(parents=True)
    helper_path.write_text("mdl 1.0;\n", encoding="utf-8")
    project_dir = tmp_path / "projects" / "fixture"
    project_dir.mkdir(parents=True)
    (project_dir / "template.json").write_text("{}", encoding="utf-8")
    material_id = "mdl:Base/Fixture.mdl#Fixture"
    catalog = {
        "schema_version": 1,
        "material_root": ".",
        "material_count": 1,
        "materials": [
            {
                "material_id": material_id,
                "mdl_path": "Base/Fixture.mdl",
                "sub_identifier": "Fixture",
            }
        ],
    }
    (project_dir / "catalog.json").write_text(
        json.dumps(catalog), encoding="utf-8"
    )
    dependency_lock = {
        "schema_version": "qwen-sealed-material-dependency-lock/v1",
        "asset_id": "fixture",
        "policy": "exact-mdl-module-runtime-resource-and-isaac-build/v1",
        "selected_materials": [
            {
                "material_id": material_id,
                "mdl_path": "Base/Fixture.mdl",
                "sub_identifier": "Fixture",
            }
        ],
        "top_level_modules": [
            {"mdl_path": "Base/Fixture.mdl", "sha256": _sha256(mdl_path)}
        ],
        "runtime_resources": [
            {
                "owner_mdl_path": "Base/Fixture.mdl",
                "authored_path": "./Fixture/fixture.png",
                "resolved_path": "Base/Fixture/fixture.png",
                "sha256": _sha256(texture_path),
            }
        ],
        "isaac_helper_modules": [
            {
                "module_name": "::Helper",
                "relative_to_isaac_root": (
                    "kit/mdl/core/Base/Helper.mdl"
                ),
                "sha256": _sha256(helper_path),
            }
        ],
        "isaac_runtime": {
            "version_file": "VERSION",
            "version": "5.0.0-test",
            "sha256": _sha256(isaac_root / "VERSION"),
        },
        "historical_parameter_policy": {
            "mode": "sealed-template-exact-parameters/v1",
            "template_hash_binds_parameters": True,
            "library_modules_immutable": True,
            "post_selection_parameter_mutation": False,
            "library_defaults_required": False,
        },
        "summary": {
            "selected_material_count": 1,
            "top_level_module_count": 1,
            "runtime_resource_count": 1,
            "isaac_helper_module_count": 1,
        },
    }
    dependency_lock_path = project_dir / "dependency_lock.json"
    dependency_lock_path.write_text(
        json.dumps(dependency_lock), encoding="utf-8"
    )
    occurrence_registry = {
        "part_count": 2,
        "parts": [
            {
                "prim_path": "/Assembly/Part1/Mesh",
                "point_count": 3,
                "face_count": 2,
            },
            {
                "prim_path": "/Assembly/Part2/Mesh",
                "point_count": 4,
                "face_count": 3,
            },
        ],
    }
    project = {
        "schema_version": "qwen-material-project/v2",
        "asset_id": "fixture",
        "source_cad": {
            "basename": source_cad.name,
            "sha256": _sha256(source_cad),
        },
        "references": [
            {"role": f"ref-{index}", "sha256": _sha256(path)}
            for index, path in enumerate(reference_paths)
        ],
        "expected_assembly": {
            "source_registry_contracts": [
                {
                    "representation_id": "fixture_instanced/v1",
                    "instance_root_count": 2,
                    "topology_role": "pre_expansion",
                },
                {
                    "representation_id": "fixture_deinstanced/v1",
                    "instance_root_count": 0,
                    "topology_role": "occurrence_equivalent",
                },
            ],
            "mesh_occurrences": 2,
            "point_occurrence_count": 7,
            "face_occurrence_count": 5,
            "occurrence_path_topology_sha256": _topology_sha256(
                occurrence_registry["parts"]
            ),
        },
        "planner_module": "fixture.plan",
        "template": "template.json",
        "template_sha256": _sha256(project_dir / "template.json"),
        "catalog": "catalog.json",
        "catalog_sha256": _sha256(project_dir / "catalog.json"),
        "dependency_lock": dependency_lock_path.name,
        "dependency_lock_sha256": _sha256(dependency_lock_path),
        "material_root_scope": "nvidia_materials",
        "render": {
            "resolution": 384,
            "views": "front,iso",
            "rt_subframes": 2,
            "analysis_up_axis": "z",
            "analysis_front_axis": "-y",
        },
        "acceptance": {
            "render": {
                "resolution": 512,
                "views": "right,front,top,iso",
                "rt_subframes": 4,
                "lighting_profile": "material-neutral",
                "analysis_up_axis": "z",
                "analysis_front_axis": "-y",
            },
            "view_mapping": {
                "ref-0": "right",
                "ref-1": "front",
                "ref-2": "top",
                "ref-3": "iso",
            },
            "minimum_comparable_views": 4,
        },
    }
    (project_dir / "project.json").write_text(
        json.dumps(project),
        encoding="utf-8",
    )
    return {
        "source_cad": source_cad,
        "references": tuple(
            (f"ref-{index}", path) for index, path in enumerate(reference_paths)
        ),
        "source_registry": {"instance_root_count": 2},
        "occurrence_registry": occurrence_registry,
        "material_root": material_root,
        "isaac_root": isaac_root,
        "texture_path": texture_path,
        "dependency_lock": dependency_lock_path,
        "project_file": project_dir / "project.json",
        "projects_root": tmp_path / "projects",
    }


def _planner_fixture(tmp_path: Path) -> dict[str, Path]:
    source_cad = tmp_path / "assembly.stp"
    source_cad.write_bytes(b"sealed-cad")
    source_usd = tmp_path / "assembly.usd"
    source_usd.write_bytes(b"sealed-usd")
    reference = tmp_path / "reference.jpg"
    reference.write_bytes(b"sealed-photo")
    prim_path = "/Assembly/Part/Mesh"
    subset_record: dict[str, object] = {
        "binding_relationship_name": "material:binding",
        "binding_targets": ["/Assembly/Looks/Original"],
        "element_type": "face",
        "face_indices": [1, 2],
        "family_name": "materialBind",
        "family_type": "unrestricted",
        "subset_name": "painted_faces",
        "subset_prim_path": f"{prim_path}/painted_faces",
        "visual_material_prim_path": "/Assembly/Looks/Original",
    }
    subset_record[SOURCE_SUBSET_HASH_FIELD] = (
        source_material_bind_subset_sha256(
            part_id="P0001",
            prim_path=prim_path,
            subset_record=subset_record,
        )
    )
    registry = {
        "asset_usd": str(source_usd),
        "asset_sha256": _sha256(source_usd),
        "part_count": 1,
        "parts": [
            {
                "part_id": "P0001",
                "prim_path": prim_path,
                "point_count": 4,
                "face_count": 3,
                SOURCE_MATERIAL_BIND_SUBSETS_FIELD: [subset_record],
            }
        ],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    template = {
        "schema_version": "qwen-dtn100-v1-material-template/v1",
        "historical_result_sha256": "historical",
        "profiles": {
            "body": {
                "material_id": "body",
                "semantic": "painted body",
            },
            "detail": {
                "material_id": "detail",
                "semantic": "painted detail",
            },
        },
        "assignments": [
            {
                "prim_path": prim_path,
                "point_count": 4,
                "face_count": 3,
                "profile": "body",
                "subsets": [
                    {
                        "subset_name": "painted_faces",
                        "profile": "detail",
                        "index_count": 2,
                        "indices_sha256": hashlib.sha256(
                            b"[1,2]"
                        ).hexdigest(),
                    }
                ],
            }
        ],
        "profile_counts": {"body": 1},
    }
    template_path = tmp_path / "template.json"
    template_path.write_text(json.dumps(template), encoding="utf-8")
    project = {
        "schema_version": "qwen-material-project/v2",
        "asset_id": "planner-fixture",
        "source_cad": {"sha256": _sha256(source_cad)},
        "references": [{"role": "front", "sha256": _sha256(reference)}],
        "expected_assembly": {
            "mesh_occurrences": 1,
            "point_occurrence_count": 4,
            "face_occurrence_count": 3,
            "subset_count": 1,
            "subset_face_count": 2,
        },
        "template": template_path.name,
        "template_sha256": _sha256(template_path),
        "evidence": {"method": "sealed-fixture"},
    }
    project_path = tmp_path / "project.json"
    project_path.write_text(json.dumps(project), encoding="utf-8")
    return {
        "source_cad": source_cad,
        "source_usd": source_usd,
        "reference": reference,
        "registry": registry_path,
        "template": template_path,
        "project": project_path,
    }


def test_exact_identity_selects_bundled_project(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    match = match_bundled_project(
        source_cad=fixture["source_cad"],
        references=fixture["references"],
        source_registry=fixture["source_registry"],
        occurrence_registry=fixture["occurrence_registry"],
        configured_material_root=fixture["material_root"] / "Base",
        isaac_root=fixture["isaac_root"],
        projects_root=fixture["projects_root"],
    )
    assert match is not None
    assert match.asset_id == "fixture"
    assert match.material_root == fixture["material_root"]
    assert match.dependency_lock_verification["status"] == "PASS"
    assert match.dependency_lock_verification["runtime_resource_count"] == 1
    assert match.acceptance["view_mapping"] == {
        "ref-0": "right",
        "ref-1": "front",
        "ref-2": "top",
        "ref-3": "iso",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda contract: contract["view_mapping"].pop("ref-3"),
            "cover every reference role exactly",
        ),
        (
            lambda contract: contract["view_mapping"].update(
                {"ref-3": "top"}
            ),
            "one-to-one",
        ),
        (
            lambda contract: contract["view_mapping"].update(
                {"ref-3": "tampered_pose"}
            ),
            "undeclared render view",
        ),
    ],
)
def test_bundled_project_rejects_invalid_acceptance_mapping(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    project_path = fixture["project_file"]
    project = json.loads(project_path.read_text(encoding="utf-8"))
    mutation(project["acceptance"])
    project_path.write_text(json.dumps(project), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        match_bundled_project(
            source_cad=fixture["source_cad"],
            references=fixture["references"],
            source_registry=fixture["source_registry"],
            occurrence_registry=fixture["occurrence_registry"],
            configured_material_root=fixture["material_root"],
            isaac_root=fixture["isaac_root"],
            projects_root=fixture["projects_root"],
        )


def test_bundled_project_rejects_acceptance_view_preset(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    project_path = fixture["project_file"]
    project = json.loads(project_path.read_text(encoding="utf-8"))
    project["acceptance"]["render"]["views"] = (
        "right,front,top,pose-bank-74"
    )
    project_path.write_text(json.dumps(project), encoding="utf-8")

    with pytest.raises(ValueError, match="explicit supported poses"):
        match_bundled_project(
            source_cad=fixture["source_cad"],
            references=fixture["references"],
            source_registry=fixture["source_registry"],
            occurrence_registry=fixture["occurrence_registry"],
            configured_material_root=fixture["material_root"],
            isaac_root=fixture["isaac_root"],
            projects_root=fixture["projects_root"],
        )


def test_bundled_project_rejects_duplicate_reference_hash(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    project_path = fixture["project_file"]
    project = json.loads(project_path.read_text(encoding="utf-8"))
    project["references"][3]["sha256"] = project["references"][2]["sha256"]
    project_path.write_text(json.dumps(project), encoding="utf-8")

    with pytest.raises(ValueError, match="malformed or duplicated"):
        match_bundled_project(
            source_cad=fixture["source_cad"],
            references=fixture["references"],
            source_registry=fixture["source_registry"],
            occurrence_registry=fixture["occurrence_registry"],
            configured_material_root=fixture["material_root"],
            isaac_root=fixture["isaac_root"],
            projects_root=fixture["projects_root"],
        )


def test_bundled_project_rejects_dependency_byte_change(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["texture_path"].write_bytes(b"changed-texture")

    with pytest.raises(ValueError, match="content changed"):
        match_bundled_project(
            source_cad=fixture["source_cad"],
            references=fixture["references"],
            source_registry=fixture["source_registry"],
            occurrence_registry=fixture["occurrence_registry"],
            configured_material_root=fixture["material_root"],
            isaac_root=fixture["isaac_root"],
            projects_root=fixture["projects_root"],
        )


def test_bundled_project_rejects_dependency_manifest_hash_change(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    lock_path = fixture["dependency_lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["tampered"] = True
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest hash changed"):
        match_bundled_project(
            source_cad=fixture["source_cad"],
            references=fixture["references"],
            source_registry=fixture["source_registry"],
            occurrence_registry=fixture["occurrence_registry"],
            configured_material_root=fixture["material_root"],
            isaac_root=fixture["isaac_root"],
            projects_root=fixture["projects_root"],
        )


def test_bundled_project_dependency_paths_cannot_escape_root(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    lock_path = fixture["dependency_lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["runtime_resources"][0]["resolved_path"] = "../escape.png"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    project_path = fixture["project_file"]
    project = json.loads(project_path.read_text(encoding="utf-8"))
    project["dependency_lock_sha256"] = _sha256(lock_path)
    project_path.write_text(json.dumps(project), encoding="utf-8")

    with pytest.raises(ValueError, match="declared root"):
        match_bundled_project(
            source_cad=fixture["source_cad"],
            references=fixture["references"],
            source_registry=fixture["source_registry"],
            occurrence_registry=fixture["occurrence_registry"],
            configured_material_root=fixture["material_root"],
            isaac_root=fixture["isaac_root"],
            projects_root=fixture["projects_root"],
        )


def test_renamed_exact_cad_still_selects_by_content(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    renamed = tmp_path / "renamed-input.step"
    renamed.write_bytes(fixture["source_cad"].read_bytes())
    match = match_bundled_project(
        source_cad=renamed,
        references=fixture["references"],
        source_registry=fixture["source_registry"],
        occurrence_registry=fixture["occurrence_registry"],
        configured_material_root=fixture["material_root"] / "Base",
        isaac_root=fixture["isaac_root"],
        projects_root=fixture["projects_root"],
    )
    assert match is not None
    assert match.asset_id == "fixture"


def test_deinstanced_596_mesh_manual_source_matches_exact_topology_contract(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    parts: list[dict[str, object]] = [
        {
            "prim_path": f"/Assembly/Occurrence{index:04d}/Mesh",
            "point_count": 1,
            "face_count": 1,
        }
        for index in range(1, 597)
    ]
    parts[0]["point_count"] = 504_669 - 595
    parts[0]["face_count"] = 546_262 - 595
    occurrence_registry = {"part_count": 596, "parts": parts}
    source_registry = {
        "instance_root_count": 0,
        "part_count": 596,
        "parts": [dict(item) for item in parts],
    }
    project_file = fixture["projects_root"] / "fixture" / "project.json"
    project = json.loads(project_file.read_text(encoding="utf-8"))
    project["expected_assembly"].update(
        {
            "mesh_occurrences": 596,
            "point_occurrence_count": 504_669,
            "face_occurrence_count": 546_262,
            "occurrence_path_topology_sha256": _topology_sha256(parts),
        }
    )
    project_file.write_text(json.dumps(project), encoding="utf-8")

    match = match_bundled_project(
        source_cad=fixture["source_cad"],
        references=fixture["references"],
        source_registry=source_registry,
        occurrence_registry=occurrence_registry,
        configured_material_root=fixture["material_root"],
        isaac_root=fixture["isaac_root"],
        projects_root=fixture["projects_root"],
    )

    assert match is not None
    assert match.source_representation_id == "fixture_deinstanced/v1"
    assert match.source_registry_topology_role == "occurrence_equivalent"


def test_dtn100_manifest_path_topology_contract_matches_sealed_template() -> None:
    _require_dtn100_bundle()
    project_dir = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "qwen_material_pipeline"
        / "projects"
        / "dtn100"
    )
    project = json.loads(
        (project_dir / "project.json").read_text(encoding="utf-8")
    )
    template = json.loads(
        (project_dir / "v1_material_template.json").read_text(
            encoding="utf-8"
        )
    )
    expected = project["expected_assembly"]
    assignments = template["assignments"]

    assert project["schema_version"] == "qwen-material-project/v2"
    assert project["acceptance"] == {
        "render": {
            "resolution": 512,
            "views": (
                "right,front,pose_a090_e082_toproll,pose_a135_e015"
            ),
            "rt_subframes": 4,
            "lighting_profile": "material-neutral",
            "analysis_up_axis": "z",
            "analysis_front_axis": "-y",
        },
        "view_mapping": {
            "front": "right",
            "side": "front",
            "top": "pose_a090_e082_toproll",
            "iso": "pose_a135_e015",
        },
        "minimum_comparable_views": 4,
    }
    assert len(assignments) == expected["mesh_occurrences"] == 596
    assert sum(item["point_count"] for item in assignments) == 504_669
    assert sum(item["face_count"] for item in assignments) == 546_262
    assert (
        _topology_sha256(assignments)
        == expected["occurrence_path_topology_sha256"]
    )
    assert {
        (item["instance_root_count"], item["topology_role"])
        for item in expected["source_registry_contracts"]
    } == {(326, "pre_expansion"), (0, "occurrence_equivalent")}


def test_dtn100_sealed_profiles_match_public_v18_library_default_lock() -> None:
    _require_dtn100_bundle()
    root = Path(__file__).resolve().parents[1]
    project_dir = (
        root / "tools" / "qwen_material_pipeline" / "projects" / "dtn100"
    )
    project = json.loads(
        (project_dir / "project.json").read_text(encoding="utf-8")
    )
    template_path = project_dir / project["template"]
    catalog_path = project_dir / project["catalog"]
    dependency_lock_path = project_dir / project["dependency_lock"]
    template = json.loads(template_path.read_text(encoding="utf-8"))
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    dependency_lock = json.loads(
        dependency_lock_path.read_text(encoding="utf-8")
    )
    public_summary_path = (
        root
        / "apps"
        / "material_audit_web"
        / "public"
        / "data"
        / "run-summary.json"
    )
    public_summary = json.loads(
        public_summary_path.read_text(encoding="utf-8")
    )
    acceptance_evidence = validate_bundled_acceptance_evidence(
        project,
        project_file=project_dir / "project.json",
        reference_paths_by_role={
            item["role"]: (
                root / "examples" / "dtn100" / "references" / item["basename"]
            )
            for item in project["references"]
        },
    )

    profiles = template["profiles"]
    profile_counts = template["profile_counts"]
    restored_counts = Counter(
        {
            profiles[profile_name]["material_id"]: assignment_count
            for profile_name, assignment_count in profile_counts.items()
        }
    )
    public_counts = Counter(
        {
            material["material_id"]: material["assignment_count"]
            for material in public_summary["materials"]
        }
    )
    catalog_ids = {item["material_id"] for item in catalog["materials"]}
    locked_ids = {
        item["material_id"] for item in dependency_lock["selected_materials"]
    }

    assert project["evidence"]["method"] == (
        "sealed_dtn100_v18_library_default_mdl_result"
    )
    assert project["evidence"]["public_audit_sha256"] == _sha256(
        public_summary_path
    )
    assert project["template_sha256"] == _sha256(template_path)
    assert project["catalog_sha256"] == _sha256(catalog_path)
    assert project["dependency_lock_sha256"] == _sha256(dependency_lock_path)
    assert acceptance_evidence is not None
    assert acceptance_evidence["manifest_sha256"] == project[
        "acceptance_evidence"
    ]["sha256"]
    assert [
        view["id"] for view in acceptance_evidence["source_views"]
    ] == ["front", "side", "top", "iso"]
    assert restored_counts == public_counts
    assert len(restored_counts) == len(catalog_ids) == len(locked_ids) == 13
    assert catalog_ids == locked_ids == set(restored_counts)
    assert all(profile.get("parameters") == {} for profile in profiles.values())
    assert dependency_lock["historical_parameter_policy"] == {
        "mode": "library-default-selected-mdl/v1",
        "template_hash_binds_parameters": True,
        "library_modules_immutable": True,
        "post_selection_parameter_mutation": False,
        "library_defaults_required": True,
    }


def test_current_v17_artifacts_match_sealed_project_and_dependencies() -> None:
    _require_dtn100_bundle()
    root = Path(__file__).resolve().parents[1]
    visual_dir = (
        root / "outputs" / "manual" / "dtn100_unattended_v17"
        / "visual_material"
    )
    material_root = (
        Path.home() / "isaacsim_assets" / "Assets" / "Isaac" / "4.5"
        / "NVIDIA" / "Materials" / "Base"
    )
    isaac_root = Path.home() / "isaacsim500"
    required = [
        root / "data" / "manual" / "dtn100-00-00_a2_asm.stp",
        visual_dir / "source_part_registry.json",
        visual_dir / "renders" / "part_registry.rendered.json",
        material_root,
        isaac_root / "VERSION",
    ]
    if not all(path.exists() for path in required):
        pytest.skip("Local v17 replay artifacts or NVIDIA dependencies unavailable")
    references = [
        (
            "front",
            root / "examples" / "dtn100" / "references" / "20260717-113202.jpg",
        ),
        (
            "side",
            root / "examples" / "dtn100" / "references" / "20260717-113159.jpg",
        ),
        (
            "top",
            root / "examples" / "dtn100" / "references" / "20260717-113206.jpg",
        ),
        (
            "iso",
            root / "examples" / "dtn100" / "references" / "20260717-113209.jpg",
        ),
    ]
    source_registry = json.loads(
        (visual_dir / "source_part_registry.json").read_text(encoding="utf-8")
    )
    occurrence_registry = json.loads(
        (visual_dir / "renders" / "part_registry.rendered.json").read_text(
            encoding="utf-8"
        )
    )

    match = match_bundled_project(
        source_cad=root / "data" / "manual" / "dtn100-00-00_a2_asm.stp",
        references=references,
        source_registry=source_registry,
        occurrence_registry=occurrence_registry,
        configured_material_root=material_root,
        isaac_root=isaac_root,
    )

    assert match is not None
    assert match.asset_id == "dtn100"
    assert match.source_representation_id == "manual_physics_deinstanced/v1"
    assert match.dependency_lock_verification["status"] == "PASS"
    assert match.dependency_lock_verification["top_level_module_count"] == 9
    assert match.dependency_lock_verification["runtime_resource_count"] == 24


def test_deinstanced_source_does_not_bypass_path_bound_topology(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    occurrence_registry = fixture["occurrence_registry"]
    source_registry = {
        "instance_root_count": 0,
        "part_count": occurrence_registry["part_count"],
        "parts": [
            dict(item) for item in occurrence_registry["parts"]
        ],
    }
    # Preserve every aggregate count while changing one occurrence identity.
    source_registry["parts"][0]["prim_path"] = "/Assembly/Replaced/Mesh"

    with pytest.raises(ValueError, match="source topology changed"):
        match_bundled_project(
            source_cad=fixture["source_cad"],
            references=fixture["references"],
            source_registry=source_registry,
            occurrence_registry=occurrence_registry,
            configured_material_root=fixture["material_root"],
            isaac_root=fixture["isaac_root"],
            projects_root=fixture["projects_root"],
        )


def test_occurrence_path_change_fails_even_when_aggregate_counts_match(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    occurrence_registry = {
        "part_count": fixture["occurrence_registry"]["part_count"],
        "parts": [
            dict(item) for item in fixture["occurrence_registry"]["parts"]
        ],
    }
    occurrence_registry["parts"][0]["prim_path"] = "/Assembly/Replaced/Mesh"

    with pytest.raises(ValueError, match="occurrence_path_topology_sha256"):
        match_bundled_project(
            source_cad=fixture["source_cad"],
            references=fixture["references"],
            source_registry=fixture["source_registry"],
            occurrence_registry=occurrence_registry,
            configured_material_root=fixture["material_root"],
            isaac_root=fixture["isaac_root"],
            projects_root=fixture["projects_root"],
        )


def test_different_photograph_uses_generic_path(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    changed = tmp_path / "changed.jpg"
    changed.write_bytes(b"different-photo")
    references = list(fixture["references"])
    references[0] = ("changed", changed)
    match = match_bundled_project(
        source_cad=fixture["source_cad"],
        references=references,
        source_registry=fixture["source_registry"],
        occurrence_registry=fixture["occurrence_registry"],
        configured_material_root=fixture["material_root"],
        isaac_root=fixture["isaac_root"],
        projects_root=fixture["projects_root"],
    )
    assert match is None


def test_exact_inputs_fail_closed_when_topology_changes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    occurrence_registry = dict(fixture["occurrence_registry"])
    occurrence_registry["part_count"] = 3
    with pytest.raises(ValueError, match="topology changed"):
        match_bundled_project(
            source_cad=fixture["source_cad"],
            references=fixture["references"],
            source_registry=fixture["source_registry"],
            occurrence_registry=occurrence_registry,
            configured_material_root=fixture["material_root"],
            isaac_root=fixture["isaac_root"],
            projects_root=fixture["projects_root"],
        )


def test_exact_inputs_fail_closed_when_template_changes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    template = fixture["projects_root"] / "fixture" / "template.json"
    template.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="template hash changed"):
        match_bundled_project(
            source_cad=fixture["source_cad"],
            references=fixture["references"],
            source_registry=fixture["source_registry"],
            occurrence_registry=fixture["occurrence_registry"],
            configured_material_root=fixture["material_root"],
            isaac_root=fixture["isaac_root"],
            projects_root=fixture["projects_root"],
        )


@pytest.mark.parametrize(
    ("instance_root_count", "expected_command", "expected_source_argument"),
    [
        (0, "apply", "--asset-usd"),
        (326, "apply-instances", "--source-usd"),
    ],
)
def test_bundled_apply_command_matches_source_representation(
    tmp_path: Path,
    instance_root_count: int,
    expected_command: str,
    expected_source_argument: str,
) -> None:
    source = tmp_path / "source.usd"
    command = _bundled_project_apply_command(
        isaac=tmp_path / "python.sh",
        source=source,
        catalog=tmp_path / "catalog.json",
        registry=tmp_path / "registry.json",
        material_plan=tmp_path / "plan.json",
        look_usd=tmp_path / "look.usda",
        material_root=tmp_path / "Materials",
        apply_report=tmp_path / "apply.json",
        instance_root_count=instance_root_count,
    )

    assert command[4] == expected_command
    assert expected_source_argument in command
    assert command[command.index(expected_source_argument) + 1] == str(source)
    rejected_argument = (
        "--source-usd"
        if expected_source_argument == "--asset-usd"
        else "--asset-usd"
    )
    assert rejected_argument not in command


def test_bundled_planner_uses_hash_bound_registry_subsets_without_pxr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_dtn100_bundle()
    fixture = _planner_fixture(tmp_path)
    original_import = builtins.__import__

    def reject_pxr(name, *args, **kwargs):
        if name == "pxr" or name.startswith("pxr."):
            raise AssertionError("bundled planner must not import pxr")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_pxr)
    plan, audit, unattended = build_plan(
        project_path=fixture["project"],
        source_cad=fixture["source_cad"],
        source_usd=fixture["source_usd"],
        registry_path=fixture["registry"],
        references=[("front", fixture["reference"])],
    )

    assert audit["status"] == "PASS"
    assert audit["face_subsets_verified"] is True
    assert plan["assignments"][0]["face_subsets"][0]["face_indices"] == [1, 2]
    assert unattended["state"] == "READY_TO_APPLY"


def test_bundled_planner_rejects_tampered_registry_subset_evidence(
    tmp_path: Path,
) -> None:
    _require_dtn100_bundle()
    fixture = _planner_fixture(tmp_path)
    registry = json.loads(fixture["registry"].read_text(encoding="utf-8"))
    registry["parts"][0][SOURCE_MATERIAL_BIND_SUBSETS_FIELD][0][
        "face_indices"
    ] = [0, 2]
    fixture["registry"].write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid source subset hash"):
        build_plan(
            project_path=fixture["project"],
            source_cad=fixture["source_cad"],
            source_usd=fixture["source_usd"],
            registry_path=fixture["registry"],
            references=[("front", fixture["reference"])],
        )
