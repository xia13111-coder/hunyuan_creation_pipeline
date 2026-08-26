from __future__ import annotations

import base64
import ast
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

import asset_pipeline
import asset_pipeline.cli as pipeline_cli
import asset_pipeline.manual_cad as manual_cad
import pipeline_runner
from asset_pipeline import runtime
from asset_pipeline.jobs.hunyuan import run_hunyuan_job
from asset_pipeline.jobs.isaac import run_add_physics_job
from asset_pipeline.jobs.sam3d import prepare_sam3d_input
from asset_pipeline import workflows
from asset_pipeline.cli import build_parser
from asset_pipeline.cli import ensure_generation_source
from asset_pipeline.cli import validate_visual_material_args


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class PipelineStructureTests(unittest.TestCase):
    def test_publishable_tree_has_no_machine_specific_paths_or_absolute_links(
        self,
    ) -> None:
        project = Path(__file__).resolve().parents[1]
        forbidden = tuple("/".join(("", root, "user")) for root in ("home", "media"))
        excluded_roots = {
            project / ".git",
            project / "docker" / "runtime",
            project / "docker" / "offline-images",
            project / "outputs",
            project / "annotations",
            project / "downloads",
            project / "downloads_refined_mesh",
            project / "cad_usd",
            project / "output_intermediate",
            project / "output_final",
            project / "manual_output_final",
            project / "sam3d_output_final",
            project / "apps" / "material_audit_web" / "dist",
            project / "apps" / "material_audit_web" / "node_modules",
            project / "apps" / "material_audit_web" / "public" / "data",
            project
            / "apps"
            / "material_audit_web"
            / "public"
            / "images"
            / "part-id-whole-similarity-v3",
            project / "pipeline_result.json",
        }
        text_suffixes = {
            ".cfg",
            ".ini",
            ".json",
            ".md",
            ".mmd",
            ".py",
            ".sh",
            ".toml",
            ".txt",
            ".yaml",
            ".yml",
        }
        leaks: list[str] = []
        absolute_links: list[str] = []
        tracked_output = subprocess.check_output(
            ["git", "ls-files", "-z"],
            cwd=project,
        )
        tracked_paths = [
            project / os.fsdecode(raw_path)
            for raw_path in tracked_output.split(b"\0")
            if raw_path
        ]
        for path in tracked_paths:
            if not path.exists() and not path.is_symlink():
                # ``git ls-files`` includes paths deleted in the current
                # change; those files are not part of the publishable tree.
                continue
            if any(path == root or root in path.parents for root in excluded_roots):
                continue
            if path.is_symlink():
                if Path(path.readlink()).is_absolute():
                    absolute_links.append(str(path.relative_to(project)))
                continue
            if path.suffix.lower() not in text_suffixes:
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            if any(marker in content for marker in forbidden):
                leaks.append(str(path.relative_to(project)))
        self.assertEqual(leaks, [])
        self.assertEqual(absolute_links, [])

    def test_visual_material_cli_is_explicit_opt_in(self) -> None:
        args = build_parser().parse_args(
            ["--existing-glb", "asset.glb", "--visual-reference", "ref.png"]
        )
        with self.assertRaisesRegex(ValueError, "--auto-visual-materials"):
            validate_visual_material_args(args)

    def test_manual_visual_material_cli_defaults_to_live_inference(self) -> None:
        args = build_parser().parse_args(["--manual-stp", "asset.stp"])
        self.assertEqual(args.visual_inference_mode, "live")

        bundled_args = build_parser().parse_args(
            [
                "--manual-stp",
                "asset.stp",
                "--auto-visual-materials",
                "--visual-inference-mode",
                "bundled",
            ]
        )
        self.assertEqual(bundled_args.visual_inference_mode, "bundled")

    def test_policy_material_fallback_cli_is_manual_cad_only_and_opt_in(
        self,
    ) -> None:
        default_args = build_parser().parse_args(["--manual-stp", "asset.stp"])
        self.assertFalse(default_args.allow_policy_material_fallback)

        glb_args = build_parser().parse_args(
            [
                "--existing-glb",
                "asset.glb",
                "--auto-visual-materials",
                "--allow-policy-material-fallback",
            ]
        )
        with self.assertRaisesRegex(ValueError, "--manual-stp"):
            validate_visual_material_args(glb_args)

        manual_args = build_parser().parse_args(
            [
                "--manual-stp",
                "asset.stp",
                "--allow-policy-material-fallback",
            ]
        )
        with self.assertRaisesRegex(ValueError, "--auto-visual-materials"):
            validate_visual_material_args(manual_args)

    def test_policy_material_fallback_cli_reaches_manual_workflow(self) -> None:
        args = build_parser().parse_args(
            [
                "--manual-stp",
                "asset.stp",
                "--auto-visual-materials",
                "--allow-policy-material-fallback",
            ]
        )
        with patch.object(
            pipeline_cli, "run_manual_cad_workflow", return_value={}
        ) as workflow:
            result = pipeline_cli.run(args)

        self.assertEqual(result["mode"], "manual_stp")
        self.assertTrue(workflow.call_args.kwargs["allow_policy_material_fallback"])
        self.assertEqual(workflow.call_args.kwargs["visual_inference_mode"], "live")

    def test_human_sam3_foreground_annotation_reaches_manual_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            annotation = Path(temp_dir) / "sam3_foreground_annotations.json"
            annotation.write_text("{}", encoding="utf-8")
            args = build_parser().parse_args(
                [
                    "--manual-stp",
                    "asset.stp",
                    "--auto-visual-materials",
                    "--visual-foreground-annotations",
                    str(annotation),
                ]
            )
            with patch.object(
                pipeline_cli, "run_manual_cad_workflow", return_value={}
            ) as workflow:
                pipeline_cli.run(args)

        self.assertEqual(
            workflow.call_args.kwargs["visual_foreground_annotations"],
            str(annotation),
        )

    def test_human_sam3_foreground_annotation_is_manual_live_only(self) -> None:
        args = build_parser().parse_args(
            [
                "--existing-glb",
                "asset.glb",
                "--auto-visual-materials",
                "--visual-foreground-annotations",
                "annotations.json",
            ]
        )
        with self.assertRaisesRegex(ValueError, "--manual-stp"):
            validate_visual_material_args(args)

        args = build_parser().parse_args(
            [
                "--manual-stp",
                "asset.stp",
                "--auto-visual-materials",
                "--visual-inference-mode",
                "bundled",
                "--visual-foreground-annotations",
                "annotations.json",
            ]
        )
        with self.assertRaisesRegex(ValueError, "inference-mode live"):
            validate_visual_material_args(args)

    def test_visual_materials_keep_generic_inference_and_exact_sealed_projects(
        self,
    ) -> None:
        project = Path(__file__).resolve().parents[1]
        orchestrator = (
            project / "asset_pipeline" / "visual_materials" / "orchestrator.py"
        ).read_text(encoding="utf-8")
        material_cli = (
            project / "tools" / "qwen_material_pipeline" / "__main__.py"
        ).read_text(encoding="utf-8")
        command_builders = (
            project / "asset_pipeline" / "visual_materials" / "commands.py"
        ).read_text(encoding="utf-8")
        self.assertFalse(
            (project / "asset_pipeline" / "visual_materials" / "projects.py").exists()
        )
        project_manifest = (
            project
            / "tools"
            / "qwen_material_pipeline"
            / "projects"
            / "dtn100"
            / "project.json"
        )
        if project_manifest.is_file():
            manifest = json.loads(project_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "qwen-material-project/v2")
            self.assertEqual(len(manifest["references"]), 4)
            self.assertTrue(manifest["source_cad"]["sha256"])
        else:
            self.assertTrue(
                (project_manifest.parents[1] / "README.md").is_file(),
                "source releases must document how private sealed projects are added",
            )
        material_inference = (
            project
            / "asset_pipeline"
            / "visual_materials"
            / "stages"
            / "material_inference.py"
        ).read_text(encoding="utf-8")
        self.assertIn("run_material_inference", orchestrator)
        self.assertIn("staged_material_command", material_inference)
        self.assertIn('"staged"', command_builders)
        source_preparation = (
            project
            / "asset_pipeline"
            / "visual_materials"
            / "stages"
            / "source_preparation.py"
        ).read_text(encoding="utf-8")
        self.assertIn("match_bundled_project", source_preparation)
        self.assertIn("_run_bundled_project_assignment", orchestrator)
        self.assertNotIn("".join(("dtn", "100", "-plan")), material_cli)

    def test_manual_cad_sdf_resolution_defaults_to_32(self) -> None:
        args = build_parser().parse_args(["--manual-stp", "asset.stp"])
        self.assertEqual(args.manual_sdf_resolution, 32)

    def test_manual_cad_rejects_glb_and_target_dimensions_at_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for suffix in (".glb", ".usd", ".usda", ".usdc"):
                invalid = root / f"asset{suffix}"
                invalid.write_bytes(b"invalid manual input")
                args = build_parser().parse_args(["--manual-stp", str(invalid)])
                with self.assertRaisesRegex(ValueError, r"\.stp/\.step"):
                    ensure_generation_source(args)

            step = root / "asset.step"
            step.write_text("STEP", encoding="utf-8")
            args = build_parser().parse_args(
                ["--manual-stp", str(step), "--len-x", "1.0"]
            )
            with self.assertRaisesRegex(ValueError, "remove --len-x"):
                ensure_generation_source(args)

            for option in ("--len-y", "--len-z"):
                args = build_parser().parse_args(
                    ["--manual-stp", str(step), option, "1.0"]
                )
                with self.assertRaisesRegex(ValueError, f"remove {option}"):
                    ensure_generation_source(args)

            args = build_parser().parse_args(
                ["--manual-stp", str(step), "--orientation", "X=L,Y=M,Z=S"]
            )
            with self.assertRaisesRegex(ValueError, "remove --orientation"):
                ensure_generation_source(args)

            second = root / "second.step"
            second.write_text("STEP", encoding="utf-8")
            args = build_parser().parse_args(
                ["--manual-stp", str(root), "--auto-visual-materials"]
            )
            with self.assertRaisesRegex(ValueError, "exactly one STEP/STP"):
                ensure_generation_source(args)

    def test_manual_cad_visual_materials_need_no_dimensions(self) -> None:
        parameters = inspect.signature(manual_cad.run_manual_cad_workflow).parameters
        self.assertNotIn("len_x", parameters)
        self.assertNotIn("len_y", parameters)
        self.assertNotIn("len_z", parameters)
        self.assertNotIn("orientation", parameters)
        self.assertIs(workflows.run_stp_physics_job, manual_cad.run_manual_cad_workflow)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "front.png"
            second = root / "side.png"
            first.write_bytes(b"front")
            second.write_bytes(b"side")
            args = build_parser().parse_args(
                [
                    "--manual-stp",
                    "asset.stp",
                    "--auto-visual-materials",
                    "--visual-reference",
                    str(first),
                    "--visual-reference",
                    str(second),
                    "--acknowledge-mvinverse-noncommercial",
                ]
            )
            with patch("asset_pipeline.cli.load_visual_material_config") as load_config:
                validate_visual_material_args(args)
        load_config.assert_called_once_with(None)

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

    def test_environment_file_loads_values_without_overriding_shell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "# local runtime\n"
                "PIPELINE_TEST_FROM_FILE='file value'\n"
                "PIPELINE_TEST_SHELL=file-value\n"
                "PIPELINE_TEST_BLANK=\n"
                "PIPELINE_TEST_HASH=https://example.invalid/a#fragment\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"PIPELINE_TEST_SHELL": "shell-value"},
                clear=False,
            ):
                loaded = runtime.load_environment_file(env_file)
                self.assertEqual(os.environ["PIPELINE_TEST_FROM_FILE"], "file value")
                self.assertEqual(os.environ["PIPELINE_TEST_SHELL"], "shell-value")
                self.assertNotIn("PIPELINE_TEST_BLANK", os.environ)
                self.assertEqual(
                    os.environ["PIPELINE_TEST_HASH"],
                    "https://example.invalid/a#fragment",
                )
                self.assertEqual(
                    loaded,
                    ("PIPELINE_TEST_FROM_FILE", "PIPELINE_TEST_HASH"),
                )
            os.environ.pop("PIPELINE_TEST_FROM_FILE", None)
            os.environ.pop("PIPELINE_TEST_HASH", None)

    def test_environment_file_rejects_malformed_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("NOT AN ASSIGNMENT\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"\.env:1"):
                runtime.load_environment_file(env_file)

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
            run_hunyuan_job(
                output_dir=output_dir,
                input_dir=output_dir,
                verbose=0,
            )

        command = run_command_mock.call_args.args[0]
        self.assertEqual(command[1:3], ["-m", "asset_pipeline.hunyuan_generation"])
        self.assertIn("--input", command)
        self.assertNotIn("--prompt", command)

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

    def test_visual_material_stage_runs_between_convert_and_physics(self) -> None:
        events: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            converted = root / "converted" / "asset.usd"
            converted.parent.mkdir()
            converted.write_text("converted", encoding="utf-8")
            look = root / "visual" / "asset_look.usda"
            look.parent.mkdir()
            look.write_text("look", encoding="utf-8")
            intermediate = root / "intermediate"
            final = root / "final"

            def align(**_kwargs):
                events.append("align")
                return {}

            def resize(**_kwargs):
                events.append("resize")
                return {}

            def convert(**_kwargs):
                events.append("convert_usd")
                return {
                    "usd_root": str(converted.parent),
                    "usd_input_path": str(converted),
                    "usd_files": [str(converted)],
                }

            def visual(**kwargs):
                events.append("assign_visual_materials")
                self.assertEqual(kwargs["source_usd"], str(converted))
                return {
                    "effective_usd": str(look),
                    "output_dir": str(look.parent),
                    "rendered_registry": "registry.json",
                    "apply_report": "apply.json",
                    "visual_quality_gate_status": "PASS",
                }

            def physics(**kwargs):
                events.append("add_physics")
                self.assertEqual(kwargs["folder"], str(look))
                output = intermediate / "asset_look_phys.usda"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("physics", encoding="utf-8")
                return {}

            def collect(**kwargs):
                events.append("collect_usd")
                self.assertEqual(
                    kwargs["folder"], str(intermediate / "asset_look_phys.usda")
                )
                collected = final / "asset_look_phys" / "asset_look_phys.usda"
                collected.parent.mkdir(parents=True)
                collected.write_text("collected", encoding="utf-8")
                return {}

            def validate_delivery(**kwargs):
                events.append("validate_visual_material_delivery")
                self.assertEqual(kwargs["look_usd"], str(look))
                self.assertEqual(
                    kwargs["collected_root_usd"],
                    str((final / "asset_look_phys" / "asset_look_phys.usda").resolve()),
                )
                return {"overall_pass": True}

            with (
                patch.object(
                    workflows, "blender_preflight", return_value=["asset.glb"]
                ),
                patch.object(workflows, "run_align_job", side_effect=align),
                patch.object(workflows, "run_resize_job", side_effect=resize),
                patch.object(workflows, "run_convert_job", side_effect=convert),
                patch.object(
                    workflows,
                    "run_assign_visual_materials_job",
                    side_effect=visual,
                ),
                patch.object(workflows, "run_add_physics_job", side_effect=physics),
                patch.object(workflows, "run_collect_job", side_effect=collect),
                patch.object(
                    workflows,
                    "run_validate_visual_material_delivery_job",
                    side_effect=validate_delivery,
                ),
                patch.object(
                    workflows, "list_files_by_suffix", return_value=["asset.glb"]
                ),
            ):
                result = workflows.run_postprocess_job(
                    input_path="asset.glb",
                    len_x=1,
                    len_y=1,
                    len_z=1,
                    intermediate_output_dir=str(intermediate),
                    final_output_dir=str(final),
                    auto_visual_materials=True,
                    visual_material_references=("front.png", "side.png"),
                    acknowledge_mvinverse_noncommercial=True,
                )

        self.assertEqual(
            events,
            [
                "align",
                "resize",
                "convert_usd",
                "assign_visual_materials",
                "add_physics",
                "collect_usd",
                "validate_visual_material_delivery",
            ],
        )
        self.assertEqual(
            [step["step"] for step in result["steps"]],
            [
                "align",
                "resize",
                "convert_usd",
                "assign_visual_materials",
                "add_physics",
                "collect_usd",
                "validate_visual_material_delivery",
            ],
        )
        self.assertEqual(result["physics_input"], str(look))

    def test_visual_material_failure_stops_before_physics(self) -> None:
        with (
            patch.object(workflows, "blender_preflight", return_value=["asset.glb"]),
            patch.object(workflows, "run_align_job", return_value={}),
            patch.object(workflows, "run_resize_job", return_value={}),
            patch.object(
                workflows,
                "run_convert_job",
                return_value={
                    "usd_root": "converted",
                    "usd_input_path": "converted/asset.usd",
                    "usd_files": ["converted/asset.usd"],
                },
            ),
            patch.object(
                workflows,
                "run_assign_visual_materials_job",
                side_effect=RuntimeError("no safe assignments"),
            ),
            patch.object(workflows, "run_add_physics_job") as physics,
            patch.object(workflows, "run_collect_job") as collect,
        ):
            with self.assertRaisesRegex(RuntimeError, "no safe assignments"):
                workflows.run_postprocess_job(
                    input_path="asset.glb",
                    len_x=1,
                    len_y=1,
                    len_z=1,
                    intermediate_output_dir="intermediate",
                    final_output_dir="final",
                    auto_visual_materials=True,
                    visual_material_references=("front.png", "side.png"),
                    acknowledge_mvinverse_noncommercial=True,
                )
        physics.assert_not_called()
        collect.assert_not_called()

    def test_manual_cad_material_stage_runs_after_geometry_normalization_without_resize(
        self,
    ) -> None:
        events: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cad_root = root / "cad"
            converted = cad_root / "asset" / "asset.usd"
            converted.parent.mkdir(parents=True)
            converted.write_text("cad", encoding="utf-8")
            look = root / "visual" / "asset_look.usda"
            look.parent.mkdir()
            look.write_text("look", encoding="utf-8")
            intermediate = root / "intermediate"
            final = root / "final"

            def cad(**_kwargs):
                events.append("cad_to_usd")
                return {
                    "out_dir": str(cad_root),
                    "cad_files": ["asset.stp"],
                    "usd_files": [str(converted)],
                }

            def visual(**kwargs):
                events.append("assign_visual_materials")
                self.assertEqual(
                    kwargs["source_usd"],
                    str(intermediate / "asset" / "asset_phys.usd"),
                )
                self.assertEqual(kwargs["source_cad"], "asset.stp")
                self.assertTrue(kwargs["require_complete_coverage"])
                self.assertTrue(kwargs["allow_policy_material_fallback"])
                self.assertEqual(kwargs["inference_mode"], "live")
                return {
                    "effective_usd": str(look),
                    "output_dir": str(look.parent),
                    "rendered_registry": "registry.json",
                    "apply_report": "apply.json",
                    "visual_quality_gate_status": "PASS",
                }

            def physics(**kwargs):
                events.append("add_physics")
                self.assertEqual(kwargs["folder"], str(converted))
                self.assertTrue(kwargs["center_origin"])
                self.assertTrue(kwargs["sdf_remesh"])
                output = intermediate / "asset" / "asset_phys.usd"
                output.parent.mkdir(parents=True)
                output.write_text("physics", encoding="utf-8")
                return {}

            def collect(**kwargs):
                events.append("collect_usd")
                self.assertEqual(kwargs["folder"], str(look))
                collected = final / "asset_look" / "asset_look.usda"
                collected.parent.mkdir(parents=True)
                collected.write_text("collected", encoding="utf-8")
                return {}

            def validate_delivery(**kwargs):
                events.append("validate_visual_material_delivery")
                self.assertEqual(kwargs["look_usd"], str(look))
                self.assertEqual(kwargs["physics_usd"], str(look))
                return {"overall_pass": True}

            def final_visual_acceptance(**kwargs):
                events.append("final_visual_acceptance")
                self.assertEqual(
                    kwargs["collected_usd"],
                    str((final / "asset_look" / "asset_look.usda").resolve()),
                )
                return {"state": "COMPLETED", "completion_allowed": True}

            with (
                patch.object(manual_cad, "run_cad_to_usd_job", side_effect=cad),
                patch.object(
                    manual_cad,
                    "run_assign_visual_materials_job",
                    side_effect=visual,
                ),
                patch.object(manual_cad, "run_add_physics_job", side_effect=physics),
                patch.object(manual_cad, "run_collect_job", side_effect=collect),
                patch.object(
                    manual_cad,
                    "run_validate_visual_material_delivery_job",
                    side_effect=validate_delivery,
                ),
                patch.object(
                    manual_cad,
                    "run_final_visual_acceptance_job",
                    side_effect=final_visual_acceptance,
                ),
                patch.object(workflows, "run_align_job") as align,
                patch.object(workflows, "run_resize_job") as resize,
            ):
                result = manual_cad.run_manual_cad_workflow(
                    input_path="asset.stp",
                    intermediate_output_dir=str(intermediate),
                    final_output_dir=str(final),
                    auto_visual_materials=True,
                    visual_material_references=("front.png", "side.png"),
                    acknowledge_mvinverse_noncommercial=True,
                    allow_policy_material_fallback=True,
                )

        align.assert_not_called()
        resize.assert_not_called()
        self.assertEqual(
            events,
            [
                "cad_to_usd",
                "add_physics",
                "assign_visual_materials",
                "collect_usd",
                "validate_visual_material_delivery",
                "final_visual_acceptance",
            ],
        )
        self.assertEqual(
            [step["step"] for step in result["steps"]],
            [
                "cad_to_usd",
                "add_physics",
                "assign_visual_materials",
                "collect_usd",
                "validate_visual_material_delivery",
                "final_visual_acceptance",
            ],
        )
        self.assertEqual(result["physics_input_files"], [str(converted)])
        self.assertEqual(
            result["visual_material_input_files"],
            [str(intermediate / "asset" / "asset_phys.usd")],
        )
        self.assertEqual(result["completion_state"], "COMPLETED")

    def test_manual_cad_rejects_unaccepted_final_visual_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cad_root = root / "cad"
            converted = cad_root / "asset" / "asset.usd"
            look = root / "visual" / "asset_look.usda"
            intermediate = root / "intermediate"
            final = root / "final"
            converted.parent.mkdir(parents=True)
            look.parent.mkdir(parents=True)
            converted.write_text("cad", encoding="utf-8")
            look.write_text("look", encoding="utf-8")

            def physics(**kwargs):
                output = Path(kwargs["out_dir"]) / "asset_phys.usd"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("physics", encoding="utf-8")
                return {}

            def collect(**kwargs):
                source = Path(kwargs["folder"])
                output = Path(kwargs["out_dir"]) / source.stem / source.name
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("collected", encoding="utf-8")
                return {}

            with (
                patch.object(
                    manual_cad,
                    "run_cad_to_usd_job",
                    return_value={
                        "out_dir": str(cad_root),
                        "cad_files": ["asset.stp"],
                        "usd_files": [str(converted)],
                    },
                ),
                patch.object(
                    manual_cad,
                    "run_assign_visual_materials_job",
                    return_value={
                        "effective_usd": str(look),
                        "output_dir": str(look.parent),
                        "rendered_registry": "registry.json",
                        "apply_report": "apply.json",
                        "visual_quality_gate_status": "PASS",
                    },
                ),
                patch.object(
                    manual_cad,
                    "run_add_physics_job",
                    side_effect=physics,
                ),
                patch.object(
                    manual_cad,
                    "run_collect_job",
                    side_effect=collect,
                ),
                patch.object(
                    manual_cad,
                    "run_validate_visual_material_delivery_job",
                    return_value={"overall_pass": True},
                ),
                patch.object(
                    manual_cad,
                    "run_final_visual_acceptance_job",
                    return_value={
                        "state": "FINAL_VISUAL_QA_FAILED",
                        "completion_allowed": False,
                    },
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "did not authorize pipeline completion",
                ),
            ):
                manual_cad.run_manual_cad_workflow(
                    input_path="asset.stp",
                    intermediate_output_dir=str(intermediate),
                    final_output_dir=str(final),
                    auto_visual_materials=True,
                    visual_material_references=("front.png", "side.png"),
                    acknowledge_mvinverse_noncommercial=True,
                )

    def test_manual_cad_known_nonpass_live_gate_stops_before_collect(self) -> None:
        for gate_status in ("FAIL", "REVIEW", "LIMITED_PASS", "", None):
            with self.subTest(gate_status=gate_status):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    cad_root = root / "cad"
                    converted = cad_root / "asset" / "asset.usd"
                    intermediate = root / "intermediate"
                    final = root / "final"
                    converted.parent.mkdir(parents=True)
                    converted.write_text("cad", encoding="utf-8")

                    def physics(**kwargs):
                        output = Path(kwargs["out_dir"]) / "asset_phys.usd"
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_text("physics", encoding="utf-8")
                        return {}

                    with (
                        patch.object(
                            manual_cad,
                            "run_cad_to_usd_job",
                            return_value={
                                "out_dir": str(cad_root),
                                "cad_files": ["asset.stp"],
                                "usd_files": [str(converted)],
                            },
                        ),
                        patch.object(
                            manual_cad,
                            "run_add_physics_job",
                            side_effect=physics,
                        ),
                        patch.object(
                            manual_cad,
                            "run_assign_visual_materials_job",
                            return_value={
                                "effective_usd": str(root / "visual" / "look.usda"),
                                "output_dir": str(root / "visual"),
                                "rendered_registry": "registry.json",
                                "apply_report": "apply.json",
                                "inference_mode": "qwen_mvinverse",
                                "visual_quality_gate_status": gate_status,
                            },
                        ),
                        patch.object(manual_cad, "run_collect_job") as collect,
                        patch.object(
                            manual_cad,
                            "run_validate_visual_material_delivery_job",
                        ) as delivery,
                        self.assertRaisesRegex(
                            RuntimeError,
                            "quality gate did not authorize collection",
                        ),
                    ):
                        manual_cad.run_manual_cad_workflow(
                            input_path="asset.stp",
                            intermediate_output_dir=str(intermediate),
                            final_output_dir=str(final),
                            auto_visual_materials=True,
                            visual_material_references=("front.png", "side.png"),
                            acknowledge_mvinverse_noncommercial=True,
                        )

                    collect.assert_not_called()
                    delivery.assert_not_called()
                    self.assertFalse(final.exists())

    def test_manual_cad_historical_restore_can_reach_collect(self) -> None:
        compatibility_metadata = (
            {"inference_mode": "bundled_project"},
            {
                "inference_mode": "qwen_mvinverse",
                "visual_quality_status": "RESTORED_HISTORICAL_BASELINE",
            },
        )
        for metadata in compatibility_metadata:
            with self.subTest(metadata=metadata):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    cad_root = root / "cad"
                    converted = cad_root / "asset" / "asset.usd"
                    look = root / "visual" / "look.usda"
                    intermediate = root / "intermediate"
                    final = root / "final"
                    converted.parent.mkdir(parents=True)
                    converted.write_text("cad", encoding="utf-8")

                    def physics(**kwargs):
                        output = Path(kwargs["out_dir"]) / "asset_phys.usd"
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_text("physics", encoding="utf-8")
                        return {}

                    def collect_job(**kwargs):
                        source = Path(kwargs["folder"])
                        output = Path(kwargs["out_dir"]) / source.stem / source.name
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_text("collected", encoding="utf-8")
                        return {}

                    visual_result = {
                        "effective_usd": str(look),
                        "output_dir": str(look.parent),
                        "rendered_registry": "registry.json",
                        "apply_report": "apply.json",
                        "visual_quality_gate_status": "FAIL",
                        **metadata,
                    }
                    with (
                        patch.object(
                            manual_cad,
                            "run_cad_to_usd_job",
                            return_value={
                                "out_dir": str(cad_root),
                                "cad_files": ["asset.stp"],
                                "usd_files": [str(converted)],
                            },
                        ),
                        patch.object(
                            manual_cad,
                            "run_add_physics_job",
                            side_effect=physics,
                        ),
                        patch.object(
                            manual_cad,
                            "run_assign_visual_materials_job",
                            return_value=visual_result,
                        ),
                        patch.object(
                            manual_cad,
                            "run_collect_job",
                            side_effect=collect_job,
                        ) as collect,
                        patch.object(
                            manual_cad,
                            "run_validate_visual_material_delivery_job",
                            return_value={"overall_pass": True},
                        ),
                        patch.object(
                            manual_cad,
                            "run_final_visual_acceptance_job",
                            return_value={
                                "state": "COMPLETED",
                                "completion_allowed": True,
                            },
                        ),
                    ):
                        manual_cad.run_manual_cad_workflow(
                            input_path="asset.stp",
                            intermediate_output_dir=str(intermediate),
                            final_output_dir=str(final),
                            auto_visual_materials=True,
                            visual_material_references=("front.png", "side.png"),
                            acknowledge_mvinverse_noncommercial=True,
                        )

                    collect.assert_called_once_with(
                        folder=str(look),
                        out_dir=str(final),
                        headless=True,
                        log_cb=None,
                    )

    def test_manual_cad_policy_fallback_requires_visual_material_stage(self) -> None:
        with patch.object(manual_cad, "run_cad_to_usd_job") as cad:
            with self.assertRaisesRegex(ValueError, "auto_visual_materials=True"):
                manual_cad.run_manual_cad_workflow(
                    input_path="asset.stp",
                    intermediate_output_dir="intermediate",
                    final_output_dir="final",
                    allow_policy_material_fallback=True,
                )
        cad.assert_not_called()

    def test_manual_cad_foreground_annotations_require_live_visual_stage(
        self,
    ) -> None:
        with patch.object(manual_cad, "run_cad_to_usd_job") as cad:
            with self.assertRaisesRegex(ValueError, "auto_visual_materials=True"):
                manual_cad.run_manual_cad_workflow(
                    input_path="asset.stp",
                    intermediate_output_dir="intermediate",
                    final_output_dir="final",
                    visual_foreground_annotations="annotations.json",
                )
        cad.assert_not_called()

        with patch.object(manual_cad, "run_cad_to_usd_job") as cad:
            with self.assertRaisesRegex(ValueError, "inference_mode='live'"):
                manual_cad.run_manual_cad_workflow(
                    input_path="asset.stp",
                    intermediate_output_dir="intermediate",
                    final_output_dir="final",
                    auto_visual_materials=True,
                    visual_foreground_annotations="annotations.json",
                    visual_inference_mode="bundled",
                )
        cad.assert_not_called()

    def test_manual_cad_multiple_assets_fail_before_materials_and_physics(
        self,
    ) -> None:
        with (
            patch.object(
                manual_cad,
                "run_cad_to_usd_job",
                return_value={
                    "out_dir": "cad",
                    "cad_files": ["a.stp", "b.stp"],
                    "usd_files": ["cad/a.usd", "cad/b.usd"],
                },
            ),
            patch.object(manual_cad, "run_assign_visual_materials_job") as visual,
            patch.object(manual_cad, "run_add_physics_job") as physics,
            patch.object(manual_cad, "run_collect_job") as collect,
        ):
            with self.assertRaisesRegex(RuntimeError, "single-asset"):
                manual_cad.run_manual_cad_workflow(
                    input_path="cad-folder",
                    intermediate_output_dir="intermediate",
                    final_output_dir="final",
                    auto_visual_materials=True,
                    visual_material_references=("front.png", "side.png"),
                    acknowledge_mvinverse_noncommercial=True,
                )
        visual.assert_not_called()
        physics.assert_not_called()
        collect.assert_not_called()

    def test_manual_cad_physics_only_directory_processes_every_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cad_root = root / "cad"
            intermediate = root / "intermediate"
            final = root / "final"
            converted = [
                cad_root / "first" / "first.usd",
                cad_root / "nested" / "second" / "second.usd",
            ]
            for path in converted:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("usd", encoding="utf-8")

            def cad(**kwargs):
                self.assertFalse(kwargs["require_single"])
                return {
                    "out_dir": str(cad_root),
                    "cad_files": ["first.stp", "nested/second.step"],
                    "usd_files": [str(path) for path in converted],
                }

            def physics(**kwargs):
                source = Path(kwargs["folder"])
                output = Path(kwargs["out_dir"]) / f"{source.stem}_phys.usd"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("physics", encoding="utf-8")
                return {}

            def collect(**kwargs):
                source = Path(kwargs["folder"])
                output = Path(kwargs["out_dir"]) / source.stem / source.name
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("collected", encoding="utf-8")
                return {}

            with (
                patch.object(manual_cad, "run_cad_to_usd_job", side_effect=cad),
                patch.object(manual_cad, "run_add_physics_job", side_effect=physics),
                patch.object(manual_cad, "run_collect_job", side_effect=collect),
                patch.object(manual_cad, "run_assign_visual_materials_job") as visual,
                patch.object(
                    manual_cad, "run_validate_visual_material_delivery_job"
                ) as delivery,
            ):
                result = manual_cad.run_manual_cad_workflow(
                    input_path=str(root),
                    intermediate_output_dir=str(intermediate),
                    final_output_dir=str(final),
                )

        visual.assert_not_called()
        delivery.assert_not_called()
        self.assertEqual(len(result["processed_cad_files"]), 2)
        self.assertEqual(len(result["physics_input_files"]), 2)
        self.assertEqual(
            [step["step"] for step in result["steps"]],
            ["cad_to_usd", "add_physics", "collect_usd"],
        )

    def test_visual_assignment_entry_only_sequences_bounded_stages(self) -> None:
        source = (
            Path(asset_pipeline.__file__).parent
            / "visual_materials"
            / "orchestrator.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        entry = functions["run_assign_visual_materials_job"]
        stage_names = {
            "prepare_source_evidence",
            "_run_policy_part_id_stage",
            "_run_look_application_stage",
            "_run_visual_qa_stage",
            "_run_material_selection_stage",
            "_run_finalize_assignment_stage",
        }
        called = {
            node.func.id
            for node in ast.walk(entry)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

        self.assertTrue(stage_names <= called)
        self.assertLessEqual(entry.end_lineno - entry.lineno + 1, 450)
        for stage_name in stage_names - {"prepare_source_evidence"}:
            stage = functions[stage_name]
            self.assertFalse(
                any(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == stage_name
                    for node in ast.walk(stage)
                ),
                f"{stage_name} must not recursively call itself",
            )


if __name__ == "__main__":
    unittest.main()
