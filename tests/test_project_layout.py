import json
from pathlib import Path

from asset_pipeline.project_layout import ProjectLayout, SOURCE_LAYOUT
from asset_pipeline.visual_materials.config import DEFAULT_CONFIG_PATH


def test_source_layout_has_one_owner_for_material_pipeline_paths() -> None:
    assert SOURCE_LAYOUT.material_pipeline == (
        SOURCE_LAYOUT.root / "tools" / "qwen_material_pipeline"
    )
    assert SOURCE_LAYOUT.material_package == (
        SOURCE_LAYOUT.material_pipeline / "src" / "qwen_material_pipeline"
    )
    assert SOURCE_LAYOUT.material_runtime == (
        SOURCE_LAYOUT.material_pipeline / "runtime"
    )
    assert SOURCE_LAYOUT.material_models == SOURCE_LAYOUT.material_runtime / "models"
    assert SOURCE_LAYOUT.material_third_party == (
        SOURCE_LAYOUT.material_pipeline / "third_party"
    )
    assert SOURCE_LAYOUT.nvidia_base_observation_bank == (
        SOURCE_LAYOUT.material_pipeline / "assets" / "nvidia_base_observation_bank_v1"
    )
    assert DEFAULT_CONFIG_PATH == SOURCE_LAYOUT.manual_part_id_material_config
    assert SOURCE_LAYOUT.material_retrieval_script == (
        SOURCE_LAYOUT.material_pipeline
        / "src"
        / "qwen_material_pipeline"
        / "retrieval"
        / "visual_materials.py"
    )
    assert SOURCE_LAYOUT.physics_materials == (
        SOURCE_LAYOUT.root / "configs" / "physics" / "materials.json"
    )
    assert SOURCE_LAYOUT.default_refine_config == (
        SOURCE_LAYOUT.root
        / "configs"
        / "refinement"
        / "hunyuan_reduce_local_postprocess.yaml"
    )
    assert SOURCE_LAYOUT.visual_material_requirements == (
        SOURCE_LAYOUT.root / "requirements" / "visual-materials.txt"
    )


def test_deployed_layout_rebases_all_runtime_paths(tmp_path: Path) -> None:
    layout = ProjectLayout.from_root(tmp_path / "deployment")

    assert layout.root == (tmp_path / "deployment").resolve()
    assert layout.material_pythonpath == (
        layout.root / "tools" / "qwen_material_pipeline" / "src"
    )
    assert layout.material_projects == (
        layout.root / "tools" / "qwen_material_pipeline" / "runtime" / "projects"
    )
    assert layout.material_configs == (
        layout.root
        / "tools"
        / "qwen_material_pipeline"
        / "src"
        / "qwen_material_pipeline"
        / "configs"
    )
    assert layout.outputs == layout.root / "outputs"
    assert layout.physics_materials == (
        layout.root / "configs" / "physics" / "materials.json"
    )
    assert layout.default_refine_config == (
        layout.root / "configs" / "refinement" / "hunyuan_reduce_local_postprocess.yaml"
    )


def test_repository_support_files_are_grouped_by_owner() -> None:
    root = SOURCE_LAYOUT.root
    expected_files = (
        "legal/README.md",
        "legal/README.zh.md",
        "legal/THIRD_PARTY_NOTICES.md",
        "requirements/visual-materials.txt",
        "docs/development/architecture.md",
        "docs/guides/manual-part-id-materials.md",
        "tools/qwen_material_pipeline/src/qwen_material_pipeline/segmentation/part_id_request.py",
        "tools/qwen_material_pipeline/scripts/qwen35/setup_qwen35_runtime.sh",
        "tools/qwen_material_pipeline/src/qwen_material_pipeline/web/result_viewer/serve.sh",
        "tools/blender/utilities/create_hunyuan_upload_proxy.py",
        "tools/blender/diagnostics/render_topology_views.py",
        "tools/isaac/utilities/apply_uniform_mdl.py",
        "tools/isaac/utilities/group_meshes_as_compound.py",
    )
    for relative in expected_files:
        assert (root / relative).is_file(), relative

    obsolete_paths = (
        "materials.json",
        "THIRD_PARTY_NOTICES.md",
        "requirements-visual-materials.txt",
        "docs/architecture.md",
        "docs/manual-part-id-materials.md",
        "tools/qwen_material_pipeline/scripts/build_part_id_sam3_request.py",
        "tools/qwen_material_pipeline/scripts/setup_qwen35_runtime.sh",
        "tools/qwen_material_pipeline/scripts/serve_results.sh",
        "tools/qwen_material_pipeline/__init__.py",
        "tools/qwen_material_pipeline/__main__.py",
        "tools/qwen_material_pipeline/configs",
        "tools/qwen_material_pipeline/segmentation",
        "tools/qwen_material_pipeline/web",
    )
    for relative in obsolete_paths:
        assert not (root / relative).exists(), relative


def test_repository_root_has_no_python_source_files() -> None:
    assert sorted(path.name for path in SOURCE_LAYOUT.root.glob("*.py")) == []


def test_production_material_config_has_no_host_specific_model_paths() -> None:
    document = json.loads(
        SOURCE_LAYOUT.manual_part_id_material_config.read_text(encoding="utf-8")
    )
    serialized = json.dumps(document, sort_keys=True)
    assert "/home/" not in serialized
    assert "/media/" not in serialized
    assert document["sam3"]["python"] == "${QWEN_PYTHON}"
    assert document["sam3"]["entityseg"]["python"] == "${ENTITYSEG_PYTHON}"
    assert document["retrieval"]["python"] == "${QWEN_PYTHON}"
    assert document["retrieval"]["siglip2_model"] == "${SIGLIP2_MODEL_PATH}"
    assert document["retrieval"]["dinov2_model"] == "${DINOV2_MODEL_PATH}"
