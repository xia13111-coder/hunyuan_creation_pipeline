from pathlib import Path

from asset_pipeline.project_layout import ProjectLayout, SOURCE_LAYOUT
from asset_pipeline.visual_materials.config import DEFAULT_CONFIG_PATH


def test_source_layout_has_one_owner_for_material_pipeline_paths() -> None:
    assert SOURCE_LAYOUT.material_pipeline == (
        SOURCE_LAYOUT.root / "tools" / "qwen_material_pipeline"
    )
    assert DEFAULT_CONFIG_PATH == SOURCE_LAYOUT.manual_part_id_material_config
    assert SOURCE_LAYOUT.material_retrieval_script == (
        SOURCE_LAYOUT.material_pipeline / "retrieval" / "visual_materials.py"
    )


def test_deployed_layout_rebases_all_runtime_paths(tmp_path: Path) -> None:
    layout = ProjectLayout.from_root(tmp_path / "deployment")

    assert layout.root == (tmp_path / "deployment").resolve()
    assert layout.material_pythonpath == layout.root / "tools"
    assert layout.material_projects == (
        layout.root / "tools" / "qwen_material_pipeline" / "projects"
    )
    assert layout.outputs == layout.root / "outputs"
