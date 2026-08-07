from __future__ import annotations

# Kept inside the standalone package so the prototype remains self-contained.

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path

from qwen_material_pipeline.qwen.client import (
    QwenClientError,
    QwenMaterialClient,
    QwenResponseError,
    build_analysis_payload,
    load_image_url,
    validate_analysis_result,
    validate_material_plan,
)


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def valid_plan() -> dict:
    return {
        "schema_version": "1.0",
        "assignments": [
            {
                "part_id": "P001",
                "material_id": "MAT_STEEL",
                "semantic": "dark matte painted steel",
                "confidence": 0.91,
                "evidence_views": ["front", "side"],
                "status": "auto",
            }
        ],
    }


class QwenClientTests(unittest.TestCase):
    def test_load_image_url_accepts_urls_and_encodes_local_images(self) -> None:
        remote = "https://example.test/front.png"
        self.assertEqual(load_image_url(remote), remote)
        data_url = "data:image/png;base64," + base64.b64encode(PNG_1X1).decode()
        self.assertEqual(load_image_url(data_url), data_url)

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "view.png"
            image_path.write_bytes(PNG_1X1)
            encoded = load_image_url(image_path)

        self.assertTrue(encoded.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(encoded.split(",", 1)[1]), PNG_1X1)

    def test_build_payload_is_multiview_json_and_hides_local_paths(self) -> None:
        payload = build_analysis_payload(
            "qwen-test",
            views=[
                {"id": "front", "image": "https://example.test/front.png"},
                {"id": "side", "image": "https://example.test/side.png"},
            ],
            parts=[
                {
                    "part_id": "P001",
                    "prim_path": "/World/SecretPrim",
                    "label": "main body",
                }
            ],
            candidate_materials=[
                {
                    "material_id": "MAT_STEEL",
                    "name": "Painted steel",
                    "mdl_path": "/private/material.mdl",
                }
            ],
        )

        self.assertEqual(payload["model"], "qwen-test")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        content = payload["messages"][1]["content"]
        image_items = [item for item in content if item["type"] == "image_url"]
        self.assertEqual(len(image_items), 2)
        prompt = content[-1]["text"]
        self.assertIn("P001", prompt)
        self.assertIn("MAT_STEEL", prompt)
        self.assertNotIn("/World/SecretPrim", prompt)
        self.assertNotIn("/private/material.mdl", prompt)

    def test_build_payload_embeds_candidate_preview_without_leaking_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            preview = Path(temp_dir) / "steel.png"
            preview.write_bytes(PNG_1X1)
            payload = build_analysis_payload(
                "qwen-test",
                views=[{"id": "front", "image": "https://example.test/front.png"}],
                parts=[{"part_id": "P001"}],
                candidate_materials=[
                    {"material_id": "MAT_STEEL", "thumbnail_image": str(preview)}
                ],
            )

        content = payload["messages"][1]["content"]
        images = [item for item in content if item["type"] == "image_url"]
        self.assertEqual(len(images), 2)
        self.assertTrue(
            images[1]["image_url"]["url"].startswith("data:image/png;base64,")
        )
        prompt = content[-1]["text"]
        self.assertNotIn(str(preview), prompt)

    def test_validate_material_plan_enforces_id_whitelists(self) -> None:
        validated = validate_material_plan(valid_plan(), {"P001"}, {"MAT_STEEL"})
        self.assertEqual(validated["assignments"][0]["confidence"], 0.91)

        unknown_part = valid_plan()
        unknown_part["assignments"][0]["part_id"] = "P999"
        with self.assertRaisesRegex(QwenResponseError, "unknown part_id"):
            validate_material_plan(unknown_part, {"P001"}, {"MAT_STEEL"})

        unknown_material = valid_plan()
        unknown_material["assignments"][0]["material_id"] = "INVENTED"
        with self.assertRaisesRegex(QwenResponseError, "unknown material_id"):
            validate_material_plan(unknown_material, {"P001"}, {"MAT_STEEL"})

    def test_validate_material_plan_rejects_paths_and_extra_fields(self) -> None:
        plan = valid_plan()
        plan["assignments"][0]["prim_path"] = "/World/Hacked"
        with self.assertRaisesRegex(QwenResponseError, "unexpected=.*prim_path"):
            validate_material_plan(plan, {"P001"}, {"MAT_STEEL"})

        plan = valid_plan()
        plan["assignments"][0]["mdl_path"] = "/tmp/material.mdl"
        with self.assertRaisesRegex(QwenResponseError, "unexpected=.*mdl_path"):
            validate_material_plan(plan, {"P001"}, {"MAT_STEEL"})

    def test_validator_accepts_human_approved_status(self) -> None:
        plan = valid_plan()
        plan["assignments"][0]["status"] = "approved"
        result = validate_material_plan(plan, {"P001"}, {"MAT_STEEL"})
        self.assertEqual(result["assignments"][0]["status"], "approved")

        plan["assignments"][0]["status"] = "assigned"
        with self.assertRaisesRegex(QwenResponseError, "status must be one of"):
            validate_material_plan(plan, {"P001"}, {"MAT_STEEL"})

    def test_validator_enforces_confidence_status_thresholds(self) -> None:
        plan = valid_plan()
        plan["assignments"][0]["confidence"] = 0.70
        with self.assertRaisesRegex(QwenResponseError, "auto requires"):
            validate_material_plan(plan, {"P001"}, {"MAT_STEEL"})

        plan["assignments"][0]["status"] = "review"
        self.assertEqual(
            validate_material_plan(plan, {"P001"}, {"MAT_STEEL"})["assignments"][0][
                "status"
            ],
            "review",
        )
        plan["assignments"][0]["status"] = "unknown"
        with self.assertRaisesRegex(QwenResponseError, "unknown requires"):
            validate_material_plan(plan, {"P001"}, {"MAT_STEEL"})

    def test_unknown_may_have_no_evidence_but_other_statuses_may_not(self) -> None:
        plan = valid_plan()
        plan["assignments"][0].update(
            {"status": "unknown", "confidence": 0.0, "evidence_views": []}
        )
        result = validate_material_plan(plan, {"P001"}, {"MAT_STEEL"})
        self.assertEqual(result["assignments"][0]["evidence_views"], [])

        plan["assignments"][0].update(
            {"status": "auto", "confidence": 0.91, "evidence_views": []}
        )
        with self.assertRaisesRegex(QwenResponseError, "empty only for unknown"):
            validate_material_plan(plan, {"P001"}, {"MAT_STEEL"})

    def test_material_decisions_require_user_reference_evidence(self) -> None:
        plan = valid_plan()
        plan["assignments"][0]["evidence_views"] = ["cad_front"]
        with self.assertRaisesRegex(QwenResponseError, "no user reference evidence"):
            validate_analysis_result(
                plan,
                [{"id": "cad_front"}],
                [{"part_id": "P001", "renders": [{"view_id": "front"}]}],
                [{"material_id": "MAT_STEEL"}],
            )

    def test_invisible_part_must_be_unknown_with_empty_evidence(self) -> None:
        plan = valid_plan()
        plan["assignments"][0]["evidence_views"] = ["front"]
        with self.assertRaisesRegex(QwenResponseError, "no visible render evidence"):
            validate_analysis_result(
                plan,
                [{"id": "front"}],
                [{"part_id": "P001", "renders": []}],
                [{"material_id": "MAT_STEEL"}],
            )

        plan["assignments"][0].update(
            {"status": "unknown", "confidence": 0.0, "evidence_views": []}
        )
        result = validate_analysis_result(
            plan,
            [{"id": "front"}],
            [{"part_id": "P001", "renders": []}],
            [{"material_id": "MAT_STEEL"}],
        )
        self.assertEqual(result["assignments"][0]["status"], "unknown")

    def test_dry_run_does_not_require_api_key(self) -> None:
        client = QwenMaterialClient(api_key=None, model="qwen-test")
        payload = client.analyze(
            views=[{"id": "front", "image": "https://example.test/front.png"}],
            parts=[{"part_id": "P001", "prim_path": "/World/Body"}],
            candidate_materials=[{"material_id": "MAT_STEEL"}],
            dry_run=True,
        )
        self.assertEqual(payload["model"], "qwen-test")

    def test_analyze_posts_openai_payload_and_validates_response(self) -> None:
        envelope = {"choices": [{"message": {"content": json.dumps(valid_plan())}}]}
        captured = {}

        def opener(http_request, timeout):
            captured["url"] = http_request.full_url
            captured["authorization"] = http_request.get_header("Authorization")
            captured["payload"] = json.loads(http_request.data)
            captured["timeout"] = timeout
            return io.BytesIO(json.dumps(envelope).encode())

        client = QwenMaterialClient(
            api_key="test-key",
            base_url="https://dashscope.test/v1",
            model="qwen-test",
            timeout=7,
            opener=opener,
        )
        result = client.analyze(
            views=[
                {"id": "front", "image": "https://example.test/front.png"},
                {"id": "side", "image": "https://example.test/side.png"},
            ],
            parts=[{"part_id": "P001", "prim_path": "/World/Body"}],
            candidate_materials=[{"material_id": "MAT_STEEL"}],
        )

        self.assertEqual(result, valid_plan())
        self.assertEqual(captured["url"], "https://dashscope.test/v1/chat/completions")
        self.assertEqual(captured["authorization"], "Bearer test-key")
        self.assertEqual(captured["payload"]["model"], "qwen-test")
        self.assertEqual(captured["timeout"], 7.0)

    def test_analyze_rejects_unknown_evidence_view(self) -> None:
        plan = valid_plan()
        plan["assignments"][0]["evidence_views"] = ["invented_view"]
        envelope = {"choices": [{"message": {"content": json.dumps(plan)}}]}
        client = QwenMaterialClient(
            api_key="test-key",
            opener=lambda *_args, **_kwargs: io.BytesIO(json.dumps(envelope).encode()),
        )
        with self.assertRaisesRegex(QwenResponseError, "unknown evidence view"):
            client.analyze(
                views=[{"id": "front", "image": "https://example.test/front.png"}],
                parts=[{"part_id": "P001"}],
                candidate_materials=[{"material_id": "MAT_STEEL"}],
            )

    def test_live_call_without_api_key_has_clear_error(self) -> None:
        client = QwenMaterialClient(api_key=None)
        client.api_key = None  # Keep this test independent of the developer shell.
        with self.assertRaisesRegex(QwenClientError, "DASHSCOPE_API_KEY"):
            client.analyze(
                views=[{"id": "front", "image": "https://example.test/front.png"}],
                parts=[{"part_id": "P001"}],
                candidate_materials=[{"material_id": "MAT_STEEL"}],
            )


if __name__ == "__main__":
    unittest.main()
