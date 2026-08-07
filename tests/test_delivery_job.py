from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asset_pipeline.jobs.delivery import run_validate_visual_material_delivery_job


class DeliveryJobTests(unittest.TestCase):
    def _inputs(self, root: Path) -> dict[str, str]:
        look = root / "asset_look.usda"
        physics = root / "asset_look_phys.usda"
        bundle = root / "bundle"
        collected = bundle / "asset_look_phys.usda"
        registry = root / "registry.json"
        apply_report = root / "apply.json"
        isaac = root / "python.sh"
        bundle.mkdir()
        for path in (look, physics, collected, registry, apply_report, isaac):
            path.write_text("{}", encoding="utf-8")
        isaac.chmod(0o755)
        return {
            "look_usd": str(look),
            "physics_usd": str(physics),
            "collected_root_usd": str(collected),
            "registry": str(registry),
            "apply_report": str(apply_report),
            "bundle_root": str(bundle),
            "isaac": str(isaac),
        }

    def test_builds_validate_delivery_command_and_returns_pass_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = self._inputs(root)
            output = root / "delivery.json"
            output.write_text(
                json.dumps({"overall_pass": True, "failure_count": 0}),
                encoding="utf-8",
            )
            isaac = inputs.pop("isaac")
            with (
                patch(
                    "asset_pipeline.jobs.delivery.isaac_python",
                    return_value=Path(isaac),
                ),
                patch("asset_pipeline.jobs.delivery.run_command") as run_command,
            ):
                result = run_validate_visual_material_delivery_job(
                    **inputs, output=str(output)
                )

        command = run_command.call_args.args[0]
        self.assertEqual(
            command[:5],
            [isaac, "-m", "qwen_material_pipeline", "usd", "validate-delivery"],
        )
        self.assertIn("--collected-root-usd", command)
        self.assertIn("--bundle-root", command)
        self.assertTrue(run_command.call_args.kwargs["env_remove"])
        self.assertTrue(result["overall_pass"])
        self.assertEqual(result["report"], str(output.resolve()))

    def test_missing_report_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = self._inputs(root)
            isaac = inputs.pop("isaac")
            with (
                patch(
                    "asset_pipeline.jobs.delivery.isaac_python",
                    return_value=Path(isaac),
                ),
                patch("asset_pipeline.jobs.delivery.run_command"),
            ):
                with self.assertRaisesRegex(RuntimeError, "did not create expected"):
                    run_validate_visual_material_delivery_job(
                        **inputs, output=str(root / "missing.json")
                    )

    def test_failed_report_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = self._inputs(root)
            output = root / "delivery.json"
            output.write_text(
                json.dumps({"overall_pass": False, "failure_count": 1}),
                encoding="utf-8",
            )
            isaac = inputs.pop("isaac")
            with (
                patch(
                    "asset_pipeline.jobs.delivery.isaac_python",
                    return_value=Path(isaac),
                ),
                patch("asset_pipeline.jobs.delivery.run_command"),
            ):
                with self.assertRaisesRegex(RuntimeError, "failed visual-material"):
                    run_validate_visual_material_delivery_job(
                        **inputs, output=str(output)
                    )


if __name__ == "__main__":
    unittest.main()
