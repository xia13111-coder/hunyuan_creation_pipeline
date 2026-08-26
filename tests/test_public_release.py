from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicReleaseTests(unittest.TestCase):
    def test_required_public_metadata_is_present(self) -> None:
        required = {
            ".env.example",
            "CHANGELOG.md",
            "CITATION.cff",
            ".github/CODE_OF_CONDUCT.md",
            ".github/CONTRIBUTING.md",
            "LICENSE",
            "NOTICE",
            "README.md",
            "README.zh.md",
            ".github/SECURITY.md",
            "legal/README.md",
            "legal/README.zh.md",
            "legal/THIRD_PARTY_NOTICES.md",
            "tools/qwen_material_pipeline/LICENSE",
            "tools/qwen_material_pipeline/third_party/mvinverse/LICENSE",
            "tools/qwen_material_pipeline/third_party/mvinverse/DINOV2_LICENSE",
        }
        missing = sorted(name for name in required if not (ROOT / name).is_file())
        self.assertEqual(missing, [])

    def test_repository_root_has_no_python_source_files(self) -> None:
        self.assertEqual(sorted(path.name for path in ROOT.glob("*.py")), [])

    def test_public_source_audit_passes(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools/release/check_public_tree.py")],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_production_material_config_has_no_host_specific_model_paths(self) -> None:
        path = (
            ROOT
            / "tools/qwen_material_pipeline/src/qwen_material_pipeline/configs/pipeline/manual_part_id_materials.json"
        )
        document = json.loads(path.read_text(encoding="utf-8"))
        serialized = json.dumps(document, sort_keys=True)
        self.assertNotIn("/home/", serialized)
        self.assertNotIn("/media/", serialized)
        self.assertEqual(document["sam3"]["python"], "${QWEN_PYTHON}")
        self.assertEqual(
            document["sam3"]["entityseg"]["python"],
            "${ENTITYSEG_PYTHON}",
        )
        self.assertEqual(document["retrieval"]["python"], "${QWEN_PYTHON}")
        self.assertEqual(
            document["retrieval"]["siglip2_model"],
            "${SIGLIP2_MODEL_PATH}",
        )
        self.assertEqual(
            document["retrieval"]["dinov2_model"],
            "${DINOV2_MODEL_PATH}",
        )


if __name__ == "__main__":
    unittest.main()
