from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from fastapi import HTTPException
    from pydantic import ValidationError
except ModuleNotFoundError as exc:  # pragma: no cover - depends on test runtime.
    raise unittest.SkipTest(
        "FastAPI tests require the main pipeline environment"
    ) from exc

from asset_pipeline import api


class ManualCadApiTests(unittest.TestCase):
    def _payload(self, input_path: str) -> dict[str, object]:
        return {
            "input_path": input_path,
            "intermediate_output_dir": "intermediate",
            "final_output_dir": "final",
        }

    def test_schema_rejects_glb_transform_fields(self) -> None:
        for field, value in (
            ("len_x", 1.0),
            ("len_y", 1.0),
            ("len_z", 1.0),
            ("orientation", "X=L,Y=M,Z=S"),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValidationError):
                    api.ProcessManualCadRequest(
                        **self._payload("asset.stp"), **{field: value}
                    )

    def test_endpoint_rejects_usd_before_submitting_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "asset.usd"
            source.write_text("usd", encoding="utf-8")
            request = api.ProcessManualCadRequest(**self._payload(str(source)))
            with patch.object(api, "submit_job") as submit:
                with self.assertRaises(HTTPException) as caught:
                    api.process_manual_cad(request)
        self.assertEqual(caught.exception.status_code, 400)
        submit.assert_not_called()

    def test_endpoint_accepts_step_without_size_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "asset.step"
            source.write_text("STEP", encoding="utf-8")
            request = api.ProcessManualCadRequest(
                **self._payload(str(source)), sdf_resolution=64
            )
            with patch.object(
                api, "submit_job", return_value={"status": "queued"}
            ) as submit:
                result = api.process_manual_cad(request)

        self.assertEqual(result, {"status": "queued"})
        payload = submit.call_args.args[1]
        self.assertEqual(payload["input_path"], str(source))
        self.assertEqual(payload["sdf_resolution"], 64)
        self.assertFalse(payload["allow_policy_material_fallback"])
        for field in ("len_x", "len_y", "len_z", "orientation"):
            self.assertNotIn(field, payload)

    def test_policy_material_fallback_requires_auto_visual_materials(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "asset.step"
            source.write_text("STEP", encoding="utf-8")
            request = api.ProcessManualCadRequest(
                **self._payload(str(source)),
                allow_policy_material_fallback=True,
            )
            with patch.object(api, "submit_job") as submit:
                with self.assertRaises(HTTPException) as caught:
                    api.process_manual_cad(request)

        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("auto_visual_materials=true", caught.exception.detail)
        submit.assert_not_called()

    def test_policy_material_fallback_is_rejected_outside_manual_cad(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            api.validate_visual_material_payload(
                {
                    "auto_visual_materials": True,
                    "allow_policy_material_fallback": True,
                }
            )
        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("STEP/STP", caught.exception.detail)

    def test_endpoint_forwards_opted_in_policy_material_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "asset.step"
            front = root / "front.png"
            side = root / "side.png"
            source.write_text("STEP", encoding="utf-8")
            front.write_bytes(b"front")
            side.write_bytes(b"side")
            request = api.ProcessManualCadRequest(
                **self._payload(str(source)),
                auto_visual_materials=True,
                visual_material_references=[str(front), str(side)],
                acknowledge_mvinverse_noncommercial=True,
                allow_policy_material_fallback=True,
            )
            with (
                patch.object(api, "load_visual_material_config"),
                patch.object(
                    api, "submit_job", return_value={"status": "queued"}
                ) as submit,
            ):
                result = api.process_manual_cad(request)

        self.assertEqual(result, {"status": "queued"})
        self.assertTrue(submit.call_args.args[1]["allow_policy_material_fallback"])

    def test_endpoint_forwards_human_sam3_foreground_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "asset.step"
            front = root / "front.png"
            side = root / "side.png"
            annotations = root / "sam3_foreground_annotations.json"
            source.write_text("STEP", encoding="utf-8")
            front.write_bytes(b"front")
            side.write_bytes(b"side")
            annotations.write_text("{}", encoding="utf-8")
            request = api.ProcessManualCadRequest(
                **self._payload(str(source)),
                auto_visual_materials=True,
                visual_material_references=[str(front), str(side)],
                visual_foreground_annotations=str(annotations),
                acknowledge_mvinverse_noncommercial=True,
            )
            with (
                patch.object(api, "load_visual_material_config"),
                patch.object(
                    api, "submit_job", return_value={"status": "queued"}
                ) as submit,
            ):
                result = api.process_manual_cad(request)

        self.assertEqual(result, {"status": "queued"})
        self.assertEqual(
            submit.call_args.args[1]["visual_foreground_annotations"],
            str(annotations),
        )

    def test_non_manual_payload_cannot_silently_drop_foreground_annotations(
        self,
    ) -> None:
        with self.assertRaises(ValidationError):
            api.ProcessModelRequest(
                input_path="asset.glb",
                intermediate_output_dir="intermediate",
                final_output_dir="final",
                len_x=1.0,
                len_y=1.0,
                len_z=1.0,
                visual_foreground_annotations="annotations.json",
            )


if __name__ == "__main__":
    unittest.main()
