from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from qwen_material_pipeline.workflows.basic import (
    _build_parser,
    _collect_input_views,
    _create_analysis_client,
    _view_id_from_filename,
    _views_from_directory,
)
from qwen_material_pipeline.qwen.local_vl import (
    LocalGenerationResult,
    LocalQwenClient,
    LocalQwen3VLClient,
    TransformersLocalQwenRunner,
    TransformersQwen3VLRunner,
    _qwen35_nonthinking_chat_template,
    decode_data_image,
    openai_payload_to_qwen_messages,
)
from qwen_material_pipeline.qwen.client import (
    QwenClientError,
    QwenMaterialClient,
    QwenResponseError,
    parse_plan_content,
    parse_plan_content_with_audit,
)


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(PNG_1X1).decode()


def valid_plan(part_id: str = "P001") -> dict:
    return {
        "schema_version": "1.0",
        "assignments": [
            {
                "part_id": part_id,
                "material_id": "MAT_STEEL",
                "semantic": "dark matte painted steel",
                "confidence": 0.91,
                "evidence_views": ["front"],
                "status": "auto",
            }
        ],
    }


def analysis_inputs() -> tuple[list[dict], list[dict], list[dict]]:
    return (
        [{"id": "front", "image": PNG_DATA_URL}],
        [{"part_id": "P001", "prim_path": "/World/Secret"}],
        [{"material_id": "MAT_STEEL", "mdl_path": "/secret/material.mdl"}],
    )


def write_model_config(directory: str | Path, model_type: str) -> None:
    architectures = {
        "qwen3_vl": "Qwen3VLForConditionalGeneration",
        "qwen3_vl_moe": "Qwen3VLMoeForConditionalGeneration",
        "qwen3_5": "Qwen3_5ForConditionalGeneration",
    }
    Path(directory, "config.json").write_text(
        json.dumps(
            {
                "architectures": [architectures.get(model_type, "UnknownModel")],
                "image_token_id": 123,
                "model_type": model_type,
                "vision_config": {"model_type": model_type},
            }
        ),
        encoding="utf-8",
    )


def fake_local_backend(
    *,
    version: str = "4.57.0",
    qwen35_class=None,
    processor_class=None,
    qwen3_vl_class=None,
) -> dict[str, types.ModuleType]:
    torch_module = types.ModuleType("torch")
    torch_module.__version__ = "2.7.0"
    torch_module.bfloat16 = object()
    torch_module.float16 = object()
    torch_module.float32 = object()
    torch_module.version = types.SimpleNamespace(cuda="12.8")
    torch_module.cuda = types.SimpleNamespace(empty_cache=lambda: None)

    transformers_module = types.ModuleType("transformers")
    transformers_module.__version__ = version
    transformers_module.AutoProcessor = processor_class or type(
        "FakeAutoProcessor", (), {}
    )
    transformers_module.AutoModelForImageTextToText = qwen3_vl_class or type(
        "FakeAutoModelForImageTextToText", (), {}
    )
    if qwen35_class is not None:
        transformers_module.Qwen3_5ForConditionalGeneration = qwen35_class
    return {"torch": torch_module, "transformers": transformers_module}


class LocalImageAdapterTests(unittest.TestCase):
    def test_decodes_rgb_image_and_preserves_message_order(self) -> None:
        payload = {
            "messages": [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
                        {"type": "text", "text": "after"},
                    ],
                },
            ]
        }
        messages = openai_payload_to_qwen_messages(payload)
        try:
            self.assertEqual(
                messages[0],
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "system"}],
                },
            )
            blocks = messages[1]["content"]
            self.assertEqual(
                [block["type"] for block in blocks], ["text", "image", "text"]
            )
            self.assertEqual(blocks[0]["text"], "before")
            self.assertEqual(blocks[2]["text"], "after")
            self.assertEqual(blocks[1]["image"].mode, "RGB")
            self.assertEqual(blocks[1]["image"].size, (1, 1))
        finally:
            messages[1]["content"][1]["image"].close()

    def test_rejects_remote_images_and_mime_mismatch(self) -> None:
        with self.assertRaisesRegex(QwenClientError, "data images only"):
            decode_data_image("https://example.test/image.png")
        mismatch = "data:image/jpeg;base64," + base64.b64encode(PNG_1X1).decode()
        with self.assertRaisesRegex(QwenClientError, "MIME mismatch"):
            decode_data_image(mismatch)

    def test_enforces_total_pixel_and_decoded_byte_budgets(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
                        {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
                    ],
                }
            ]
        }
        with self.assertRaisesRegex(QwenClientError, "max_total_pixels=1"):
            openai_payload_to_qwen_messages(payload, max_total_pixels=1)
        with self.assertRaisesRegex(QwenClientError, "max_image_bytes=1"):
            decode_data_image(PNG_DATA_URL, max_image_bytes=1)

    def test_downsamples_large_photo_in_memory_to_pixel_budget(self) -> None:
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (20, 10), (120, 80, 40)).save(buffer, format="PNG")
        image_url = "data:image/png;base64," + base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")

        image = decode_data_image(image_url, max_image_pixels=50)
        try:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (10, 5))
            self.assertLessEqual(image.width * image.height, 50)
        finally:
            image.close()

    def test_rejects_unknown_content_blocks(self) -> None:
        payload = {
            "messages": [{"role": "user", "content": [{"type": "video", "url": "x"}]}]
        }
        with self.assertRaisesRegex(QwenClientError, "Unsupported local content"):
            openai_payload_to_qwen_messages(payload)


class LocalClientTests(unittest.TestCase):
    def test_fake_runner_reuses_shared_payload_and_validation(self) -> None:
        captured = {}

        def runner(payload):
            captured["payload"] = payload
            return json.dumps(valid_plan())

        views, parts, materials = analysis_inputs()
        client = LocalQwen3VLClient(model="qwen-local-test", runner=runner)
        result = client.analyze(views, parts, materials)

        self.assertEqual(result, valid_plan())
        self.assertEqual(captured["payload"]["model"], "qwen-local-test")
        prompt = captured["payload"]["messages"][1]["content"][-1]["text"]
        self.assertNotIn("/World/Secret", prompt)
        self.assertNotIn("/secret/material.mdl", prompt)

    def test_raw_output_is_saved_before_strict_json_failure(self) -> None:
        raw = "Model result:\n```json\n{}\n```"
        views, parts, materials = analysis_inputs()
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = Path(temp_dir) / "model.raw.txt"
            client = LocalQwen3VLClient(
                runner=lambda _payload: raw,
                raw_output_path=raw_path,
            )
            with self.assertRaisesRegex(QwenResponseError, "Markdown fence"):
                client.analyze(views, parts, materials)
            self.assertEqual(raw_path.read_text(encoding="utf-8"), raw)

    def test_invalid_json_is_not_misreported_as_markdown(self) -> None:
        with self.assertRaises(QwenResponseError) as raised:
            parse_plan_content('{"schema_version": "1.0",')
        self.assertIn("invalid JSON", str(raised.exception))
        self.assertNotIn("Markdown", str(raised.exception))
        self.assertEqual(
            getattr(raised.exception, "reason", None), "invalid_json_syntax"
        )

    def test_one_exact_json_fence_is_audited_and_unwrapped(self) -> None:
        raw = " \n```json\n{\"ok\": true}\n```\t"
        document, audit = parse_plan_content_with_audit(raw)

        self.assertEqual(document, {"ok": True})
        self.assertEqual(
            audit["normalization"], "exact_markdown_json_fence_removed"
        )
        self.assertEqual(
            audit["raw_sha256"], hashlib.sha256(raw.encode()).hexdigest()
        )
        self.assertEqual(
            audit["normalized_sha256"],
            hashlib.sha256(b'{"ok": true}').hexdigest(),
        )
        self.assertEqual(audit["strict_json_status"], "valid_object")
        self.assertTrue(audit["strict_json_valid"])
        self.assertTrue(audit["top_level_object"])

    def test_one_exact_unlabelled_fence_is_accepted(self) -> None:
        self.assertEqual(parse_plan_content("```\n{}\n```"), {})

    def test_nonexact_or_unsafe_markdown_fences_are_rejected(self) -> None:
        cases = {
            "prose_before": "Result:\n```json\n{}\n```",
            "prose_after": "```json\n{}\n```\nDone",
            "multiple": "```json\n{}\n```\n```json\n{}\n```",
            "nested": "```json\n{\"text\": \"```\"}\n```",
            "unknown_language": "```javascript\n{}\n```",
        }
        for label, raw in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(QwenResponseError) as raised:
                    parse_plan_content(raw)
                self.assertIn("Markdown fence", str(raised.exception))
                audit = getattr(raised.exception, "parse_audit", None)
                self.assertIsNotNone(audit)
                self.assertFalse(audit["strict_json_valid"])
                self.assertEqual(
                    audit["strict_json_status"],
                    "not_parsed_transport_rejected",
                )

    def test_fenced_invalid_json_stays_a_strict_parse_failure(self) -> None:
        with self.assertRaises(QwenResponseError) as raised:
            parse_plan_content("```json\n{\"schema_version\":\n```")
        self.assertEqual(
            getattr(raised.exception, "reason", None), "invalid_json_syntax"
        )
        audit = raised.exception.parse_audit
        self.assertEqual(
            audit["normalization"], "exact_markdown_json_fence_removed"
        )
        self.assertEqual(audit["strict_json_status"], "invalid_json_syntax")

    def test_shared_validation_rejects_unknown_ids_and_excess_assignments(self) -> None:
        views, parts, materials = analysis_inputs()
        client = LocalQwen3VLClient(
            runner=lambda _payload: json.dumps(valid_plan("P999"))
        )
        with self.assertRaisesRegex(QwenResponseError, "unknown part_id"):
            client.analyze(views, parts, materials)

        parts.append({"part_id": "P002"})
        plan = valid_plan()
        second = dict(plan["assignments"][0])
        second["part_id"] = "P002"
        plan["assignments"].append(second)
        client = LocalQwen3VLClient(runner=lambda _payload: json.dumps(plan))
        with self.assertRaisesRegex(QwenResponseError, "max_assignments=1"):
            client.analyze(views, parts, materials, max_assignments=1)

    def test_runner_must_return_text(self) -> None:
        views, parts, materials = analysis_inputs()
        client = LocalQwen3VLClient(runner=lambda _payload: valid_plan())
        with self.assertRaisesRegex(QwenClientError, "must return text"):
            client.analyze(views, parts, materials)

    def test_live_inference_requires_a_user_reference_before_runner(self) -> None:
        called = False

        def runner(_payload):
            nonlocal called
            called = True
            return json.dumps(valid_plan())

        _views, parts, materials = analysis_inputs()
        client = LocalQwen3VLClient(runner=runner)
        with self.assertRaisesRegex(QwenClientError, "user reference view"):
            client.analyze(
                [{"id": "cad_front", "image": PNG_DATA_URL}], parts, materials
            )
        self.assertFalse(called)

    def test_dry_run_with_real_runner_object_does_not_import_or_load_model(
        self,
    ) -> None:
        views, parts, materials = analysis_inputs()
        with tempfile.TemporaryDirectory() as model_dir:
            client = LocalQwen3VLClient(model_path=model_dir)
            runner = client._runner
            self.assertIsInstance(runner, TransformersQwen3VLRunner)
            payload = client.analyze(views, parts, materials, dry_run=True)
            self.assertIsNone(runner._model)
            self.assertEqual(payload["model"], Path(model_dir).name)

    def test_model_path_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing"
            with self.assertRaises(FileNotFoundError):
                LocalQwen3VLClient(model_path=missing)


class LocalRunnerDispatchTests(unittest.TestCase):
    def test_qwen35_legacy_template_gets_hard_nonthinking_generation_prompt(
        self,
    ) -> None:
        legacy = """prefix
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n<think>\\n' }}
{%- endif %}"""
        upgraded = _qwen35_nonthinking_chat_template(legacy)
        self.assertIn("enable_thinking is defined", upgraded)
        self.assertIn("<think>\\n\\n</think>\\n\\n", upgraded)
        self.assertNotEqual(upgraded, legacy)

    def test_qwen35_template_upgrade_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(QwenClientError, "refusing"):
            _qwen35_nonthinking_chat_template("unknown template")

    def test_qwen35_current_template_is_preserved(self) -> None:
        current = "{{ enable_thinking }} stable"
        self.assertEqual(_qwen35_nonthinking_chat_template(current), current)

    def test_generic_names_preserve_legacy_class_identity(self) -> None:
        self.assertIs(LocalQwenClient, LocalQwen3VLClient)
        self.assertIs(TransformersLocalQwenRunner, TransformersQwen3VLRunner)

    def test_qwen3_vl_keeps_existing_auto_model_loader(self) -> None:
        qwen3_vl_class = type("Qwen3VLLoader", (), {})
        with tempfile.TemporaryDirectory() as model_dir:
            write_model_config(model_dir, "qwen3_vl")
            runner = TransformersQwen3VLRunner(model_dir)
            modules = fake_local_backend(qwen3_vl_class=qwen3_vl_class)

            with patch.dict(sys.modules, modules):
                runner.preflight()

        self.assertEqual(runner._model_type, "qwen3_vl")
        self.assertIs(runner._auto_model_class, qwen3_vl_class)

    def test_qwen35_uses_official_conditional_generation_class(self) -> None:
        qwen35_class = type("Qwen35Loader", (), {})
        with tempfile.TemporaryDirectory() as model_dir:
            write_model_config(model_dir, "qwen3_5")
            runner = TransformersQwen3VLRunner(model_dir)
            modules = fake_local_backend(qwen35_class=qwen35_class)

            with patch.dict(sys.modules, modules):
                runner.preflight()

        self.assertEqual(runner._model_type, "qwen3_5")
        self.assertIs(runner._auto_model_class, qwen35_class)

    def test_qwen35_missing_transformers_capability_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as model_dir:
            write_model_config(model_dir, "qwen3_5")
            runner = TransformersQwen3VLRunner(model_dir)
            modules = fake_local_backend(version="4.57.0")

            with patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(
                    QwenClientError,
                    r"Qwen3\.5 requires.*Qwen3_5ForConditionalGeneration"
                    r".*Install.*network fallback.*disabled",
                ):
                    runner.preflight()

    def test_preflight_rejects_text_only_config_with_qwen_model_type(self) -> None:
        with tempfile.TemporaryDirectory() as model_dir:
            config_path = Path(model_dir, "config.json")
            config_path.write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen3_5ForConditionalGeneration"],
                        "image_token_id": 123,
                        "model_type": "qwen3_5",
                    }
                ),
                encoding="utf-8",
            )
            runner = TransformersQwen3VLRunner(model_dir)

            with self.assertRaisesRegex(QwenClientError, "no usable vision_config"):
                runner.preflight()

    def test_generation_explicitly_disables_thinking(self) -> None:
        class FakeInputs(dict):
            def __init__(self):
                super().__init__(input_ids=[[1, 2]])
                self.input_ids = self["input_ids"]

            def to(self, _device):
                return self

        class FakeProcessor:
            template_kwargs = None
            eos_token_id = 3
            chat_template = """{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n<think>\\n' }}
{%- endif %}"""

            def apply_chat_template(self, _messages, **kwargs):
                self.template_kwargs = kwargs
                return FakeInputs()

            def batch_decode(self, *_args, **_kwargs):
                return ["model output"]

        class FakeModel:
            device = "cpu"

            def generate(self, **_kwargs):
                return [[1, 2, 3]]

        class InferenceMode:
            def __enter__(self):
                return None

            def __exit__(self, *_args):
                return False

        with tempfile.TemporaryDirectory() as model_dir:
            runner = TransformersQwen3VLRunner(model_dir)
            runner._processor = FakeProcessor()
            runner._model = FakeModel()
            runner._model_type = "qwen3_5"
            runner._qwen35_chat_template = _qwen35_nonthinking_chat_template(
                runner._processor.chat_template
            )
            runner._torch = types.SimpleNamespace(
                inference_mode=lambda: InferenceMode()
            )
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "inspect"}],
                    }
                ]
            }

            self.assertEqual(runner(payload), "model output")
            self.assertIs(
                runner._processor.template_kwargs["enable_thinking"],
                False,
            )
            self.assertIn(
                "enable_thinking is defined",
                runner._processor.template_kwargs["chat_template"],
            )
            self.assertEqual(
                runner.last_generation_metadata,
                {
                    "schema_version": "local-qwen-generation/v1",
                    "generated_tokens": 1,
                    "max_new_tokens": 8192,
                    "hit_token_limit": False,
                    "eos_detected": True,
                    "truncated": False,
                    "finish_reason": "eos",
                },
            )

    def test_generation_budget_override_reports_length_truncation(self) -> None:
        class FakeInputs(dict):
            def __init__(self):
                super().__init__(input_ids=[[1, 2]])
                self.input_ids = self["input_ids"]

            def to(self, _device):
                return self

        class FakeProcessor:
            eos_token_id = 99
            chat_template = None

            def apply_chat_template(self, _messages, **_kwargs):
                return FakeInputs()

            def batch_decode(self, *_args, **_kwargs):
                return ['{"incomplete":']

        class FakeModel:
            device = "cpu"
            generation_config = types.SimpleNamespace(eos_token_id=99)
            kwargs = None

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return [[1, 2, 10, 11]]

        class InferenceMode:
            def __enter__(self):
                return None

            def __exit__(self, *_args):
                return False

        with tempfile.TemporaryDirectory() as model_dir:
            runner = TransformersQwen3VLRunner(model_dir, max_new_tokens=1024)
            runner._processor = FakeProcessor()
            runner._model = FakeModel()
            runner._model_type = "qwen3_vl"
            runner._torch = types.SimpleNamespace(
                inference_mode=lambda: InferenceMode()
            )
            result = runner.generate_with_metadata(
                {"messages": [{"role": "user", "content": "inspect"}]},
                max_new_tokens=2,
            )

        self.assertIsInstance(result, LocalGenerationResult)
        self.assertEqual(result.generated_tokens, 2)
        self.assertEqual(result.max_new_tokens, 2)
        self.assertTrue(result.hit_token_limit)
        self.assertFalse(result.eos_detected)
        self.assertTrue(result.truncated)
        self.assertEqual(runner.max_new_tokens, 1024)

    def test_load_unload_reload_is_local_and_keeps_frozen_identity(self) -> None:
        class FakeProcessor:
            calls = []
            chat_template = """{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n<think>\\n' }}
{%- endif %}"""

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.calls.append((args, kwargs))
                return cls()

        class FakeModel:
            calls = []

            def __init__(self):
                self.evaluated = False

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.calls.append((args, kwargs))
                return cls()

            def eval(self):
                self.evaluated = True
                return self

        with tempfile.TemporaryDirectory() as model_dir:
            write_model_config(model_dir, "qwen3_5")
            Path(model_dir, "preprocessor_config.json").write_text(
                "{}",
                encoding="utf-8",
            )
            Path(model_dir, "chat_template.json").write_text(
                json.dumps({"chat_template": "test template"}),
                encoding="utf-8",
            )
            Path(model_dir, "model.safetensors").write_bytes(b"fake weights")
            runner = TransformersQwen3VLRunner(model_dir)
            modules = fake_local_backend(
                qwen35_class=FakeModel,
                processor_class=FakeProcessor,
            )
            empty_cache_calls = []
            modules["torch"].cuda = types.SimpleNamespace(
                empty_cache=lambda: empty_cache_calls.append(True)
            )

            with patch.dict(sys.modules, modules):
                runner._load()
                identity_before = runner.model_identity
                runner.unload()
                self.assertIsNone(runner._model)
                self.assertIsNone(runner._processor)
                self.assertEqual(runner.model_identity, identity_before)
                runner._load()

            expected_path = str(Path(model_dir).resolve())
            self.assertEqual(FakeProcessor.calls[0][0], (expected_path,))
            self.assertTrue(FakeProcessor.calls[0][1]["local_files_only"])
            self.assertFalse(FakeProcessor.calls[0][1]["trust_remote_code"])
            self.assertEqual(FakeModel.calls[0][0], (expected_path,))
            self.assertTrue(FakeModel.calls[0][1]["local_files_only"])
            self.assertFalse(FakeModel.calls[0][1]["trust_remote_code"])
            self.assertTrue(runner._model.evaluated)
            self.assertEqual(len(FakeProcessor.calls), 2)
            self.assertEqual(len(FakeModel.calls), 2)
            self.assertEqual(empty_cache_calls, [True])

            self.assertEqual(identity_before["backend"], "transformers-local")
            self.assertEqual(identity_before["model_type"], "qwen3_5")
            self.assertIn("fingerprint", identity_before["config"])
            self.assertIn("fingerprint", identity_before["processor"])
            self.assertIn("fingerprint", identity_before["chat_template"])
            self.assertEqual(identity_before["weights"]["file_count"], 1)
            self.assertFalse(identity_before["generation"]["enable_thinking"])
            json.dumps(identity_before)

    def test_rejects_unsupported_or_missing_local_model_config(self) -> None:
        with tempfile.TemporaryDirectory() as model_dir:
            runner = TransformersQwen3VLRunner(model_dir)
            with self.assertRaisesRegex(QwenClientError, "config does not exist"):
                runner.preflight()

            write_model_config(model_dir, "not_qwen")
            modules = fake_local_backend()
            with patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(
                    QwenClientError, "Unsupported local Qwen model_type"
                ):
                    runner.preflight()

    def test_network_model_resolution_cannot_be_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as model_dir:
            with self.assertRaisesRegex(ValueError, "local_files_only=True"):
                TransformersQwen3VLRunner(
                    model_dir,
                    local_files_only=False,
                )

    def test_client_exposes_identity_and_delegates_unload(self) -> None:
        class FakeRunner:
            model_identity = {
                "schema_version": "1.0",
                "backend": "fake-local",
            }

            def __init__(self):
                self.unload_calls = 0

            def __call__(self, _payload):
                return json.dumps(valid_plan())

            def unload(self):
                self.unload_calls += 1

        runner = FakeRunner()
        client = LocalQwen3VLClient(runner=runner)
        identity = client.model_identity
        identity["backend"] = "mutated"
        self.assertEqual(client.model_identity["backend"], "fake-local")

        client.unload()
        self.assertEqual(runner.unload_calls, 1)


class LocalCliTests(unittest.TestCase):
    def _parse(self, *extra: str):
        return _build_parser().parse_args(
            [
                "analyze",
                "--registry",
                "registry.json",
                "--catalog",
                "catalog.json",
                "--output",
                "plan.json",
                *extra,
            ]
        )

    def test_dashscope_remains_the_default_backend(self) -> None:
        args = self._parse()
        self.assertEqual(args.backend, "dashscope")
        self.assertIsInstance(_create_analysis_client(args), QwenMaterialClient)

    def test_transformers_backend_requires_model_path(self) -> None:
        args = self._parse("--backend", "transformers")
        with self.assertRaisesRegex(ValueError, "--model-path is required"):
            _create_analysis_client(args)

    def test_transformers_backend_options_reach_local_client(self) -> None:
        with tempfile.TemporaryDirectory() as model_dir:
            args = self._parse(
                "--backend",
                "transformers",
                "--model-path",
                model_dir,
                "--dtype",
                "float16",
                "--device-map",
                "cuda:0",
                "--attn-implementation",
                "eager",
                "--max-new-tokens",
                "2048",
            )
            client = _create_analysis_client(args)
            self.assertIsInstance(client, LocalQwen3VLClient)
            self.assertEqual(client._runner.dtype, "float16")
            self.assertEqual(client._runner.device_map, "cuda:0")
            self.assertEqual(client._runner.attn_implementation, "eager")
            self.assertEqual(client._runner.max_new_tokens, 2048)

    def test_backend_specific_arguments_are_rejected(self) -> None:
        args = self._parse("--model-path", "/tmp/model")
        with self.assertRaisesRegex(ValueError, "only valid"):
            _create_analysis_client(args)

    def test_view_dir_reads_supported_images_in_filename_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            filenames = [
                "06 rear.GIF",
                "01 front.PNG",
                "05 top.bmp",
                "04 iso.webp",
                "03 right.jpeg",
                "02 left.jpg",
            ]
            for filename in filenames:
                (directory / filename).write_bytes(b"image")
            (directory / "ignore.txt").write_text("not an image", encoding="utf-8")
            nested = directory / "nested"
            nested.mkdir()
            (nested / "00 hidden.png").write_bytes(b"image")

            views = _views_from_directory(directory)

            self.assertEqual(
                [view["id"] for view in views],
                [
                    "ref_01_front",
                    "ref_02_left",
                    "ref_03_right",
                    "ref_04_iso",
                    "ref_05_top",
                    "ref_06_rear",
                ],
            )
            self.assertTrue(all(Path(view["image"]).is_absolute() for view in views))
            self.assertFalse(any("hidden" in view["image"] for view in views))

    def test_view_dir_ids_are_stable_and_do_not_use_reserved_prefixes(self) -> None:
        self.assertEqual(_view_id_from_filename("cad_front.png"), "ref_cad_front")
        self.assertEqual(_view_id_from_filename("Part IDs.JPG"), "ref_part_ids")
        self.assertEqual(_view_id_from_filename("正面视图.png"), "ref_正面视图")
        fallback = _view_id_from_filename("!!!.png")
        self.assertEqual(fallback, _view_id_from_filename("!!!.png"))
        self.assertTrue(fallback.startswith("ref_image_"))

    def test_view_dir_merges_with_explicit_and_registry_views(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            (directory / "front.png").write_bytes(b"image")
            registry = {
                "render_set": {
                    "views": [
                        {
                            "view_id": "iso",
                            "rgb": "/tmp/cad.png",
                            "part_ids": "/tmp/ids.png",
                        }
                    ],
                    "contact_sheets": ["/tmp/contact.png"],
                }
            }

            views = _collect_input_views(
                explicit_views=[{"id": "ref_manual", "image": "/tmp/manual.png"}],
                view_directories=[directory],
                registry=registry,
                include_registry_renders=True,
            )

            self.assertEqual(
                [view["id"] for view in views],
                [
                    "ref_manual",
                    "ref_front",
                    "cad_iso",
                    "part_ids_iso",
                    "part_contact_01",
                ],
            )

    def test_view_dir_rejects_missing_file_and_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            with self.assertRaisesRegex(ValueError, "contains no supported images"):
                _views_from_directory(directory)

            file_path = directory / "photo.png"
            file_path.write_bytes(b"image")
            with self.assertRaisesRegex(ValueError, "must be a directory"):
                _views_from_directory(file_path)
            with self.assertRaisesRegex(ValueError, "does not exist"):
                _views_from_directory(directory / "missing")

    def test_view_dir_reports_generated_id_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            (directory / "front.jpg").write_bytes(b"image")
            (directory / "front.png").write_bytes(b"image")
            with self.assertRaisesRegex(
                ValueError, "Duplicate input view IDs: ref_front"
            ):
                _collect_input_views(
                    explicit_views=[],
                    view_directories=[directory],
                    registry={},
                    include_registry_renders=False,
                )

    def test_view_dir_reports_collision_with_explicit_view(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            (directory / "front.png").write_bytes(b"image")
            with self.assertRaisesRegex(
                ValueError, "Duplicate input view IDs: ref_front"
            ):
                _collect_input_views(
                    explicit_views=[{"id": "ref_front", "image": "/tmp/manual.png"}],
                    view_directories=[directory],
                    registry={},
                    include_registry_renders=False,
                )

    def test_parser_accepts_repeatable_view_directories(self) -> None:
        args = self._parse("--view-dir", "views-a", "--view-dir", "views-b")
        self.assertEqual(args.view_dir, [Path("views-a"), Path("views-b")])


if __name__ == "__main__":
    unittest.main()
