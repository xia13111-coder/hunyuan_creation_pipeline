from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asset_pipeline import workflows
from asset_pipeline.cli import build_parser, ensure_generation_source
from asset_pipeline.jobs.sam3d import run_sam3d_image_job


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "YAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


class Sam3dConfidenceTests(unittest.TestCase):
    def test_cli_confidence_defaults_to_half(self) -> None:
        args = build_parser().parse_args(
            ["--sam3d-input", "images", "--sam3d-prompt", "fixture"]
        )
        self.assertEqual(args.sam3d_confidence_threshold, 0.5)

    def test_cli_rejects_confidence_outside_unit_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for value in ("-0.01", "1.01"):
                args = build_parser().parse_args(
                    [
                        "--sam3d-input",
                        temp_dir,
                        "--sam3d-prompt",
                        "fixture",
                        "--sam3d-confidence-threshold",
                        value,
                    ]
                )
                with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                    ensure_generation_source(args)

    @patch(
        "asset_pipeline.workflows.run_process_model_job",
        return_value={"steps": []},
    )
    @patch(
        "asset_pipeline.workflows.run_sam3d_image_job",
        return_value={"postprocess_input_path": "result.glb"},
    )
    def test_workflow_forwards_confidence_to_sam3d_job(
        self,
        sam3d_job_mock,
        _process_model_mock,
    ) -> None:
        workflows.run_sam3d_image_and_process_model_job(
            input_path="images",
            output_dir="downloads",
            intermediate_output_dir="intermediate",
            final_output_dir="final",
            len_x=1.0,
            len_y=1.0,
            len_z=1.0,
            orientation="X=L,Y=M,Z=S",
            sam3d_prompt="fixture assembly",
            sam3d_confidence_threshold=0.05,
            refine_mesh=False,
        )

        self.assertEqual(
            sam3d_job_mock.call_args.kwargs["confidence_threshold"],
            0.05,
        )

    @patch("asset_pipeline.jobs.sam3d.list_files_by_suffix", return_value=[])
    @patch(
        "asset_pipeline.jobs.sam3d.select_sam3d_glb",
        return_value=Path("result.glb"),
    )
    @patch(
        "asset_pipeline.jobs.sam3d.sam3d_python",
        return_value=Path("/opt/conda/envs/sam3d/bin/python"),
    )
    @patch("asset_pipeline.jobs.sam3d.run_command")
    def test_job_forwards_confidence_to_wrapper(
        self,
        run_command_mock,
        _sam3d_python_mock,
        _select_glb_mock,
        _list_files_mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.png"
            source.write_bytes(PNG_1X1)
            result = run_sam3d_image_job(
                input_path=str(source),
                output_dir=str(root / "output"),
                prompt="fixture assembly",
                confidence_threshold=0.05,
            )

        command = run_command_mock.call_args.args[0]
        option_index = command.index("--confidence-threshold")
        self.assertEqual(command[option_index + 1], "0.05")
        self.assertEqual(result["confidence_threshold"], 0.05)


if __name__ == "__main__":
    unittest.main()
