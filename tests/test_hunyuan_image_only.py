from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

API_AVAILABLE = True
API_IMPORT_ERROR = ""
try:
    from fastapi import HTTPException
    from pydantic import ValidationError

    from asset_pipeline.api import (
        GenerateAndProcessModelRequest,
        GenerateModelRequest,
        validate_hunyuan_image_payload,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - depends on test runtime.
    API_AVAILABLE = False
    API_IMPORT_ERROR = str(exc)

from asset_pipeline.cli import build_parser
from asset_pipeline.hunyuan_generation import (
    build_payload_single_pro,
    build_payload_single_rapid,
    submit_job_with_retry,
)
from asset_pipeline.jobs.hunyuan import run_hunyuan_job


class HunyuanImageOnlyTests(unittest.TestCase):
    def test_public_cli_rejects_text_prompt_option(self) -> None:
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            build_parser().parse_args(["--prompt", "red chair"])

    @unittest.skipUnless(API_AVAILABLE, f"FastAPI unavailable: {API_IMPORT_ERROR}")
    def test_api_schemas_reject_prompt_field(self) -> None:
        with self.assertRaises(ValidationError):
            GenerateModelRequest(
                image_url="https://example.com/chair.png",
                prompt="red chair",
            )

        with self.assertRaises(ValidationError):
            GenerateAndProcessModelRequest(
                image_url="https://example.com/chair.png",
                prompt="red chair",
                intermediate_output_dir="intermediate",
                final_output_dir="final",
                len_x=0.4,
                len_y=0.3,
                len_z=0.8,
            )

    @unittest.skipUnless(API_AVAILABLE, f"FastAPI unavailable: {API_IMPORT_ERROR}")
    def test_api_requires_exactly_one_image_source(self) -> None:
        validate_hunyuan_image_payload({"input_dir": "images", "image_url": None})
        validate_hunyuan_image_payload(
            {"input_dir": None, "image_url": "https://example.com/chair.png"}
        )

        for payload in (
            {"input_dir": None, "image_url": None},
            {
                "input_dir": "images",
                "image_url": "https://example.com/chair.png",
            },
        ):
            with self.subTest(payload=payload), self.assertRaises(HTTPException):
                validate_hunyuan_image_payload(payload)

    def test_payload_builders_emit_only_image_sources(self) -> None:
        rapid = build_payload_single_rapid(
            image_path=None,
            result_format="GLB",
            enable_pbr=True,
            image_url="https://example.com/chair.png",
        )
        self.assertEqual(rapid["ImageUrl"], "https://example.com/chair.png")
        self.assertNotIn("Prompt", rapid)

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "chair.png"
            image_path.write_bytes(b"x" * 1024)
            pro = build_payload_single_pro(
                image_path=str(image_path),
                result_format="GLB",
                enable_pbr=True,
            )

        self.assertIn("ImageBase64", pro)
        self.assertNotIn("Prompt", pro)

    def test_submission_rejects_non_image_payloads(self) -> None:
        invalid_payloads = (
            {"Prompt": "red chair"},
            {},
            {"ImageBase64": "encoded", "ImageUrl": "https://example.com/a.png"},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                submit_job_with_retry(
                    object(),
                    payload,
                    result_format="GLB",
                    version="pro",
                    max_retry=1,
                    retry_interval=0,
                )

    @patch("asset_pipeline.jobs.hunyuan.run_command")
    def test_job_command_contains_one_image_source(self, run_command_mock) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_hunyuan_job(
                output_dir=temp_dir,
                image_url="https://example.com/chair.png",
                verbose=0,
            )

        command = run_command_mock.call_args.args[0]
        self.assertIn("--image-url", command)
        self.assertNotIn("--input", command)
        self.assertNotIn("--prompt", command)

        with self.assertRaises(ValueError):
            run_hunyuan_job(
                output_dir="out",
                input_dir="images",
                image_url="https://example.com/chair.png",
            )


if __name__ == "__main__":
    unittest.main()
