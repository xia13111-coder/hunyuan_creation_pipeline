from __future__ import annotations

import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

import asset_pipeline
import pipeline_runner
from asset_pipeline import runtime
from asset_pipeline.jobs.hunyuan import run_hunyuan_job
from asset_pipeline.jobs.isaac import run_add_physics_job
from asset_pipeline.jobs.sam3d import prepare_sam3d_input
from asset_pipeline import workflows
from asset_pipeline.cli import build_parser


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class PipelineStructureTests(unittest.TestCase):
    def test_manual_cad_sdf_resolution_defaults_to_32(self) -> None:
        args = build_parser().parse_args(["--manual-stp", "asset.stp"])
        self.assertEqual(args.manual_sdf_resolution, 32)

    @patch("asset_pipeline.jobs.isaac.run_command")
    def test_add_physics_job_passes_sdf_resolution(self, run_command_mock) -> None:
        result = run_add_physics_job(
            folder="input.usd",
            out_dir="output",
            approx="sdf",
            sdf_resolution=64,
        )

        command = run_command_mock.call_args.args[0]
        index = command.index("--sdf-res")
        self.assertEqual(command[index + 1], "64")
        self.assertEqual(result["sdf_resolution"], 64)

    def test_sam3d_reuses_the_active_python(self) -> None:
        self.assertEqual(runtime.sam3d_python(), Path(sys.executable).resolve())

    def test_runtime_rejects_a_different_conda_environment(self) -> None:
        with patch.object(runtime.sys, "prefix", "/opt/conda/envs/other"):
            with self.assertRaisesRegex(RuntimeError, "hunyuan_sam3d"):
                runtime.require_unified_environment()

    def test_compatibility_module_exports_new_owners(self) -> None:
        self.assertIs(
            pipeline_runner.run_refine_mesh_job, asset_pipeline.run_refine_mesh_job
        )
        self.assertEqual(
            pipeline_runner.run_refine_mesh_job.__module__, "asset_pipeline.jobs.refine"
        )
        self.assertEqual(
            pipeline_runner.run_process_model_job.__module__, "asset_pipeline.workflows"
        )

    def test_materials_use_explicit_presets_only(self) -> None:
        path = asset_pipeline.project_root() / "materials.json"
        materials = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(materials), {"materials"})
        self.assertIn("plastic", materials["materials"])

    def test_tencent_sdk_versions_stay_aligned(self) -> None:
        path = asset_pipeline.project_root() / "environment.yml"
        environment = yaml.safe_load(path.read_text(encoding="utf-8"))
        pip_dependencies = next(
            item["pip"]
            for item in environment["dependencies"]
            if isinstance(item, dict) and "pip" in item
        )
        versions = {
            name: version
            for dependency in pip_dependencies
            for name, separator, version in [dependency.partition("==")]
            if separator and name.startswith("tencentcloud-sdk-python")
        }

        self.assertEqual(
            versions,
            {
                "tencentcloud-sdk-python": "3.0.1462",
                "tencentcloud-sdk-python-ai3d": "3.0.1462",
                "tencentcloud-sdk-python-common": "3.0.1462",
            },
        )

    def test_sam3d_auto_mode_prepares_sorted_multiview_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            source.mkdir()
            (source / "b.png").write_bytes(PNG_1X1)
            (source / "a.png").write_bytes(PNG_1X1)
            (source / "mask_ignore.png").write_bytes(PNG_1X1)

            work_dir, mode, source_images = prepare_sam3d_input(
                str(source),
                str(root / "output"),
                "auto",
            )

            self.assertEqual(mode, "multi")
            self.assertEqual(
                [Path(path).name for path in source_images], ["a.png", "b.png"]
            )
            self.assertTrue((Path(work_dir) / "images" / "00000.png").exists())
            self.assertTrue((Path(work_dir) / "images" / "00001.png").exists())

    @patch("asset_pipeline.jobs.hunyuan.run_command")
    def test_hunyuan_job_runs_package_module(self, run_command_mock) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            run_hunyuan_job(output_dir=output_dir, verbose=0)

        command = run_command_mock.call_args.args[0]
        self.assertEqual(command[1:3], ["-m", "asset_pipeline.hunyuan_generation"])

    @patch("asset_pipeline.workflows.run_process_model_job")
    @patch("asset_pipeline.workflows.run_refine_mesh_job")
    @patch("asset_pipeline.workflows.run_generate_model_job")
    def test_generated_workflow_passes_refined_output_to_postprocess(
        self,
        generate_mock,
        refine_mock,
        process_mock,
    ) -> None:
        generate_mock.return_value = {"model_files": ["raw.glb"]}
        refine_mock.return_value = {"postprocess_input_path": "refined_glbs"}
        process_mock.return_value = {"steps": []}

        result = workflows.run_generate_and_process_model_job(
            output_dir="downloads",
            intermediate_output_dir="intermediate",
            final_output_dir="final",
            len_x=0.4,
            len_y=0.3,
            len_z=0.8,
            orientation="X=L,Y=M,Z=S",
        )

        self.assertEqual(
            result["refine_mesh"]["postprocess_input_path"], "refined_glbs"
        )
        self.assertEqual(process_mock.call_args.kwargs["input_path"], "refined_glbs")


if __name__ == "__main__":
    unittest.main()
