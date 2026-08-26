"""Canonical repository paths shared by pipeline layers.

Keep repository structure knowledge here.  Workflow and job modules should not
reconstruct paths such as ``tools/qwen_material_pipeline`` themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectLayout:
    """Typed path map for one checkout or deployed project root."""

    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ProjectLayout":
        return cls(Path(root).expanduser().resolve())

    @property
    def apps(self) -> Path:
        return self.root / "apps"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def physics_configs(self) -> Path:
        return self.configs / "physics"

    @property
    def physics_materials(self) -> Path:
        return self.physics_configs / "materials.json"

    @property
    def refinement_configs(self) -> Path:
        return self.configs / "refinement"

    @property
    def default_refine_config(self) -> Path:
        return self.refinement_configs / "hunyuan_reduce_local_postprocess.yaml"

    @property
    def requirements(self) -> Path:
        return self.root / "requirements"

    @property
    def visual_material_requirements(self) -> Path:
        return self.requirements / "visual-materials.txt"

    @property
    def docs(self) -> Path:
        return self.root / "docs"

    @property
    def examples(self) -> Path:
        return self.root / "examples"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def tools(self) -> Path:
        return self.root / "tools"

    @property
    def material_pipeline(self) -> Path:
        return self.tools / "qwen_material_pipeline"

    @property
    def material_configs(self) -> Path:
        return self.material_pipeline / "configs"

    @property
    def manual_part_id_material_config(self) -> Path:
        return self.material_configs / "pipeline" / "manual_part_id_materials.json"

    @property
    def material_projects(self) -> Path:
        return self.material_pipeline / "projects"

    @property
    def material_pythonpath(self) -> Path:
        """Directory placed on PYTHONPATH for ``qwen_material_pipeline``."""

        return self.tools

    @property
    def material_retrieval_script(self) -> Path:
        return self.material_pipeline / "retrieval" / "visual_materials.py"

    @property
    def sam3d_tools(self) -> Path:
        return self.tools / "sam3d"

    @property
    def sam3d_single_view(self) -> Path:
        return self.sam3d_tools / "third_party" / "sam-3d-objects"

    @property
    def sam3d_multi_view(self) -> Path:
        return self.sam3d_tools / "third_party" / "sam-3d-objects-multiview"


SOURCE_LAYOUT = ProjectLayout.from_root(Path(__file__).resolve().parents[1])


__all__ = ["ProjectLayout", "SOURCE_LAYOUT"]
