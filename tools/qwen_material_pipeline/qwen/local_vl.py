"""Local Hugging Face Transformers backend for Qwen vision-language models.

This module deliberately imports Pillow, PyTorch, and Transformers only when
they are needed.  Catalog operations, remote inference, and dry runs therefore
remain usable in the dependency-free base environment.  The historical
Qwen3-VL class names remain public compatibility aliases even though the
runner also supports Qwen3.5.
"""

from __future__ import annotations

import base64
import binascii
import copy
import gc
import hashlib
import io
import json
import platform
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_material_pipeline.qwen.client import (
    QwenClientError,
    build_analysis_payload,
    parse_plan_content,
    require_user_reference_views,
    validate_analysis_result,
)


DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
MIN_TRANSFORMERS_VERSION = (4, 57, 0)
_QWEN3_VL_MODEL_TYPES = frozenset({"qwen3_vl", "qwen3_vl_moe"})
_QWEN35_MODEL_TYPE = "qwen3_5"
_QWEN35_MODEL_CLASS = "Qwen3_5ForConditionalGeneration"
_QWEN35_NONTHINKING_TEMPLATE_MODE = "qwen3.5-hard-nonthinking/v1"
_QWEN35_LEGACY_GENERATION_BLOCK = """{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n<think>\\n' }}
{%- endif %}"""
_QWEN35_NONTHINKING_GENERATION_BLOCK = """{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- else %}
        {{- '<think>\\n' }}
    {%- endif %}
{%- endif %}"""
_EXPECTED_ARCHITECTURES = {
    "qwen3_vl": "Qwen3VLForConditionalGeneration",
    "qwen3_vl_moe": "Qwen3VLMoeForConditionalGeneration",
    _QWEN35_MODEL_TYPE: _QWEN35_MODEL_CLASS,
}
_PROCESSOR_ARTIFACT_NAMES = (
    "added_tokens.json",
    "checkpoint_identity.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)
_CHAT_TEMPLATE_NAMES = ("chat_template.json", "chat_template.jinja")
_WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.pth")
_WEIGHT_SAMPLE_BYTES = 64 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 1024 * 1024
DEFAULT_MAX_TOTAL_PIXELS = 16 * 1024 * 1024
DEFAULT_MAX_IMAGE_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_NEW_TOKENS = 8192
# A compressed image can otherwise expand to an unreasonable allocation before
# it is resized for the model.  This ceiling is intentionally independent from
# the inference pixel budget and still covers common 48/64 MP phone photos.
HARD_MAX_SOURCE_PIXELS = 64 * 1024 * 1024
_DATA_IMAGE_RE = re.compile(
    r"^data:(image/[a-zA-Z0-9.+-]+);base64,([a-zA-Z0-9+/=\r\n]+)$"
)
_SUPPORTED_IMAGE_MIMES = frozenset(
    {"image/png", "image/jpeg", "image/jpg", "image/gif", "image/bmp", "image/webp"}
)


@dataclass(frozen=True)
class LocalGenerationResult:
    """One decoded local generation plus bounded-completion telemetry."""

    text: str
    generated_tokens: int
    max_new_tokens: int
    hit_token_limit: bool
    eos_detected: bool | None

    @property
    def truncated(self) -> bool:
        """Return whether the backend stopped at its budget without EOS."""

        return self.hit_token_limit and self.eos_detected is not True

    def metadata(self) -> dict[str, Any]:
        if self.eos_detected is True:
            finish_reason = "eos"
        elif self.truncated:
            finish_reason = "length"
        else:
            finish_reason = "stopped"
        return {
            "schema_version": "local-qwen-generation/v1",
            "generated_tokens": self.generated_tokens,
            "max_new_tokens": self.max_new_tokens,
            "hit_token_limit": self.hit_token_limit,
            "eos_detected": self.eos_detected,
            "truncated": self.truncated,
            "finish_reason": finish_reason,
        }


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _version_tuple(value: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        raise QwenClientError(f"Cannot parse Transformers version: {value!r}")
    return tuple(int(component) for component in match.groups())


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _read_local_model_config(model_path: Path) -> tuple[dict[str, Any], bytes]:
    """Read the dispatch configuration directly from local storage."""

    config_path = model_path / "config.json"
    try:
        raw_config = config_path.read_bytes()
    except FileNotFoundError as exc:
        raise QwenClientError(
            f"Local model config does not exist: {config_path}"
        ) from exc
    except OSError as exc:
        raise QwenClientError(
            f"Could not read local model config: {config_path}: {exc}"
        ) from exc
    try:
        config = json.loads(raw_config.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QwenClientError(
            f"Local model config is not valid JSON: {config_path}: {exc}"
        ) from exc
    if not isinstance(config, dict):
        raise QwenClientError(
            f"Local model config must contain a JSON object: {config_path}"
        )
    return config, raw_config


def _validated_model_type(config: Mapping[str, Any], config_path: Path) -> str:
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type.strip():
        raise QwenClientError(
            f"Local model config must define a non-empty model_type: {config_path}"
        )
    model_type = model_type.strip()
    if model_type not in _EXPECTED_ARCHITECTURES:
        supported = ", ".join(sorted(_EXPECTED_ARCHITECTURES))
        raise QwenClientError(
            f"Unsupported local Qwen model_type={model_type!r}; supported "
            f"model types: {supported}"
        )
    return model_type


def _validate_visual_capability(
    config: Mapping[str, Any],
    *,
    model_type: str,
    config_path: Path,
) -> list[str]:
    """Reject text-only or mislabeled local configurations before loading."""

    architectures = config.get("architectures")
    if (
        isinstance(architectures, (str, bytes))
        or not isinstance(architectures, Sequence)
        or not all(isinstance(item, str) and item for item in architectures)
    ):
        raise QwenClientError(
            f"Local vision-language config has invalid architectures: {config_path}"
        )
    architecture_names = list(architectures)
    expected_architecture = _EXPECTED_ARCHITECTURES[model_type]
    if expected_architecture not in architecture_names:
        raise QwenClientError(
            f"Local config model_type={model_type!r} must declare "
            f"{expected_architecture} in architectures; found "
            f"{architecture_names!r}"
        )

    vision_config = config.get("vision_config")
    if not isinstance(vision_config, Mapping) or not vision_config:
        raise QwenClientError(
            f"Local config model_type={model_type!r} has no usable vision_config; "
            "a vision-language checkpoint is required"
        )
    image_token_id = config.get("image_token_id")
    if (
        isinstance(image_token_id, bool)
        or not isinstance(image_token_id, int)
        or image_token_id < 0
    ):
        raise QwenClientError(
            f"Local config model_type={model_type!r} has no valid image_token_id; "
            "a vision-language checkpoint is required"
        )
    return architecture_names


def _full_file_entry(path: Path, model_path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise QwenClientError(
            f"Could not fingerprint local artifact {path}: {exc}"
        ) from exc
    return {
        "path": path.relative_to(model_path).as_posix(),
        "bytes": len(raw),
        "sha256": _sha256_bytes(raw),
    }


def _processor_identity(model_path: Path) -> dict[str, Any]:
    entries = [
        _full_file_entry(path, model_path)
        for name in _PROCESSOR_ARTIFACT_NAMES
        if (path := model_path / name).is_file()
    ]
    return {
        "fingerprint": _json_fingerprint(entries),
        "files": entries,
    }


def _chat_template_identity(model_path: Path) -> dict[str, Any]:
    entries = [
        _full_file_entry(path, model_path)
        for name in _CHAT_TEMPLATE_NAMES
        if (path := model_path / name).is_file()
    ]
    if not entries:
        tokenizer_config_path = model_path / "tokenizer_config.json"
        if tokenizer_config_path.is_file():
            try:
                tokenizer_config = json.loads(
                    tokenizer_config_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise QwenClientError(
                    "Could not inspect tokenizer_config.json for a local chat "
                    f"template: {exc}"
                ) from exc
            chat_template = (
                tokenizer_config.get("chat_template")
                if isinstance(tokenizer_config, Mapping)
                else None
            )
            if chat_template is not None:
                entries.append(
                    {
                        "path": "tokenizer_config.json#chat_template",
                        "bytes": len(
                            json.dumps(chat_template, ensure_ascii=False).encode(
                                "utf-8"
                            )
                        ),
                        "sha256": _json_fingerprint(chat_template),
                    }
                )
    return {
        "enable_thinking": False,
        "fingerprint": _json_fingerprint(entries),
        "sources": entries,
    }


def _qwen35_nonthinking_chat_template(template: Any) -> str:
    """Return a Qwen3.5 template with a fail-closed hard thinking switch.

    Early Qwen3.5-4B revisions always opened ``<think>`` in their generation
    prompt even when callers supplied ``enable_thinking=False``.  Newer
    official templates expose the variable directly.  Keep newer templates
    unchanged; upgrade the one known legacy generation block in memory only.
    The checkpoint on disk remains byte-for-byte identical to its verified
    manifest.
    """

    if not isinstance(template, str) or not template.strip():
        raise QwenClientError(
            "Local Qwen3.5 processor has no usable chat template; refusing to "
            "run because hard non-thinking mode cannot be guaranteed"
        )
    if "enable_thinking" in template:
        return template
    occurrences = template.count(_QWEN35_LEGACY_GENERATION_BLOCK)
    if occurrences != 1:
        raise QwenClientError(
            "Local Qwen3.5 chat template does not expose enable_thinking and "
            "does not match the supported pinned legacy generation block; "
            "refusing to silently run in thinking mode"
        )
    return template.replace(
        _QWEN35_LEGACY_GENERATION_BLOCK,
        _QWEN35_NONTHINKING_GENERATION_BLOCK,
        1,
    )


def _sampled_weight_entry(path: Path, model_path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(_WEIGHT_SAMPLE_BYTES)
            suffix = b""
            if size > _WEIGHT_SAMPLE_BYTES:
                stream.seek(max(0, size - _WEIGHT_SAMPLE_BYTES))
                suffix = stream.read(_WEIGHT_SAMPLE_BYTES)
    except OSError as exc:
        raise QwenClientError(
            f"Could not fingerprint local model weight {path}: {exc}"
        ) from exc
    sample = b"\0".join(
        (
            path.relative_to(model_path).as_posix().encode("utf-8"),
            str(size).encode("ascii"),
            prefix,
            suffix,
        )
    )
    return {
        "path": path.relative_to(model_path).as_posix(),
        "bytes": size,
        "sample_sha256": _sha256_bytes(sample),
    }


def _weights_identity(model_path: Path) -> dict[str, Any]:
    weight_paths: set[Path] = set()
    for pattern in _WEIGHT_PATTERNS:
        weight_paths.update(path for path in model_path.glob(pattern) if path.is_file())
    entries = [
        _sampled_weight_entry(path, model_path)
        for path in sorted(weight_paths, key=lambda item: item.name)
    ]
    index_entries = [
        _full_file_entry(path, model_path)
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
        if (path := model_path / name).is_file()
    ]
    manifest = {"files": entries, "indexes": index_entries}
    return {
        "algorithm": (
            "sha256-v1(path+size+first-and-last-65536-bytes; full-index-files)"
        ),
        "fingerprint": _json_fingerprint(manifest),
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        **manifest,
    }


def decode_data_image(
    image_url: str,
    *,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> Any:
    """Decode a data URL and fit it to the local inference pixel budget.

    The returned Pillow image exists only in memory.  Oversized source photos
    are downsampled without modifying the user's original file.
    """

    max_image_pixels = _positive_int(max_image_pixels, "max_image_pixels")
    max_image_bytes = _positive_int(max_image_bytes, "max_image_bytes")
    if not isinstance(image_url, str):
        raise TypeError("image_url must be a string")
    match = _DATA_IMAGE_RE.fullmatch(image_url)
    if not match:
        if image_url.startswith(("http://", "https://")):
            raise QwenClientError(
                "Local Qwen3-VL accepts embedded data images only; provide local "
                "image files so the shared payload can embed them"
            )
        raise QwenClientError("Local image input must be a base64 image data URL")
    declared_mime = match.group(1).lower()
    if declared_mime not in _SUPPORTED_IMAGE_MIMES:
        raise QwenClientError(f"Unsupported local image MIME type: {declared_mime}")

    encoded = "".join(match.group(2).split())
    # Reject oversized input before allocating the decoded byte string.
    if len(encoded) > ((max_image_bytes + 2) // 3) * 4 + 4:
        raise QwenClientError(
            f"Encoded image exceeds max_image_bytes={max_image_bytes}"
        )
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise QwenClientError("Local image data contains invalid base64") from exc
    if not raw:
        raise QwenClientError("Local image data is empty")
    if len(raw) > max_image_bytes:
        raise QwenClientError(f"Image exceeds max_image_bytes={max_image_bytes}")

    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise QwenClientError(
            "Local Qwen3-VL requires Pillow; install requirements-local.txt"
        ) from exc

    try:
        with Image.open(io.BytesIO(raw)) as probe:
            width, height = probe.size
            detected_mime = Image.MIME.get(probe.format)
            if width < 1 or height < 1:
                raise QwenClientError("Decoded image has invalid dimensions")
            if width * height > HARD_MAX_SOURCE_PIXELS:
                raise QwenClientError(
                    f"Source image has {width * height} pixels, exceeding the "
                    f"hard safety limit of {HARD_MAX_SOURCE_PIXELS}"
                )
            probe.verify()
        if detected_mime:
            normalized_declared = (
                "image/jpeg" if declared_mime == "image/jpg" else declared_mime
            )
            if detected_mime.lower() != normalized_declared:
                raise QwenClientError(
                    f"Image MIME mismatch: declared {declared_mime}, "
                    f"decoded {detected_mime.lower()}"
                )
        from PIL import ImageOps

        with Image.open(io.BytesIO(raw)) as decoded:
            decoded.load()
            oriented = ImageOps.exif_transpose(decoded)
            try:
                image = oriented.convert("RGB")
            finally:
                if oriented is not decoded:
                    oriented.close()
        pixels = image.width * image.height
        if pixels <= max_image_pixels:
            return image

        scale = (max_image_pixels / pixels) ** 0.5
        target_width = max(1, int(image.width * scale))
        target_height = max(1, int(image.height * scale))
        # Integer rounding can overshoot the exact budget by a handful of
        # pixels for unusual aspect ratios.
        while target_width * target_height > max_image_pixels:
            if target_width >= target_height and target_width > 1:
                target_width -= 1
            elif target_height > 1:
                target_height -= 1
            else:  # pragma: no cover - max_image_pixels is always positive
                break
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        resized = image.resize((target_width, target_height), resampling)
        image.close()
        return resized
    except QwenClientError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise QwenClientError("Could not decode local image data") from exc


def openai_payload_to_qwen_messages(
    payload: Mapping[str, Any],
    *,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    max_total_pixels: int = DEFAULT_MAX_TOTAL_PIXELS,
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> list[dict[str, Any]]:
    """Convert the shared OpenAI-style payload to Qwen processor messages."""

    max_image_pixels = _positive_int(max_image_pixels, "max_image_pixels")
    max_total_pixels = _positive_int(max_total_pixels, "max_total_pixels")
    max_image_bytes = _positive_int(max_image_bytes, "max_image_bytes")
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be an object")
    source_messages = payload.get("messages")
    if isinstance(source_messages, (str, bytes)) or not isinstance(
        source_messages, Sequence
    ):
        raise QwenClientError("Payload messages must be an array")

    messages: list[dict[str, Any]] = []
    total_pixels = 0
    for message_index, message in enumerate(source_messages):
        if not isinstance(message, Mapping):
            raise QwenClientError(f"messages[{message_index}] must be an object")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise QwenClientError(f"messages[{message_index}].role is invalid")
        content = message.get("content")
        if isinstance(content, str):
            # Qwen3-VL's multimodal Processor iterates every message content
            # block, including the system message, and therefore requires the
            # structured list form instead of a bare string.
            messages.append(
                {"role": role, "content": [{"type": "text", "text": content}]}
            )
            continue
        if isinstance(content, (str, bytes)) or not isinstance(content, Sequence):
            raise QwenClientError(
                f"messages[{message_index}].content must be text or an array"
            )

        converted: list[dict[str, Any]] = []
        for block_index, block in enumerate(content):
            if not isinstance(block, Mapping):
                raise QwenClientError(
                    f"messages[{message_index}].content[{block_index}] must be an object"
                )
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if not isinstance(text, str):
                    raise QwenClientError(
                        f"messages[{message_index}].content[{block_index}].text "
                        "must be a string"
                    )
                converted.append({"type": "text", "text": text})
                continue
            if block_type != "image_url":
                raise QwenClientError(
                    f"Unsupported local content block type: {block_type!r}"
                )
            image_url = block.get("image_url")
            if not isinstance(image_url, Mapping) or not isinstance(
                image_url.get("url"), str
            ):
                raise QwenClientError(
                    f"messages[{message_index}].content[{block_index}].image_url "
                    "is invalid"
                )
            image = decode_data_image(
                image_url["url"],
                max_image_pixels=max_image_pixels,
                max_image_bytes=max_image_bytes,
            )
            pixels = image.width * image.height
            if total_pixels + pixels > max_total_pixels:
                image.close()
                for converted_message in messages:
                    _close_content_images(converted_message.get("content"))
                _close_content_images(converted)
                raise QwenClientError(
                    f"Images exceed max_total_pixels={max_total_pixels}"
                )
            total_pixels += pixels
            converted.append({"type": "image", "image": image})
        messages.append({"role": role, "content": converted})
    return messages


def _close_content_images(content: Any) -> None:
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, Mapping) and block.get("type") == "image":
            close = getattr(block.get("image"), "close", None)
            if close:
                close()


def _eos_token_ids(model: Any, processor: Any) -> set[int]:
    """Collect configured EOS IDs without assuming one Transformers layout."""

    values: list[Any] = []
    for owner in (
        getattr(model, "generation_config", None),
        getattr(model, "config", None),
        getattr(processor, "tokenizer", None),
        processor,
    ):
        if owner is not None:
            values.append(getattr(owner, "eos_token_id", None))

    result: set[int] = set()
    for value in values:
        candidates = value if isinstance(value, (list, tuple, set)) else (value,)
        for candidate in candidates:
            if (
                isinstance(candidate, int)
                and not isinstance(candidate, bool)
                and candidate >= 0
            ):
                result.add(candidate)
    return result


def _last_token_id(tokens: Any) -> int | None:
    if len(tokens) < 1:
        return None
    value = tokens[-1]
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


class TransformersQwen3VLRunner:
    """Lazy local Qwen3-VL/Qwen3.5 generator.

    The class keeps its original name so existing integrations do not need to
    change imports.  Model architecture selection is based only on the local
    ``config.json``; remote code and network model resolution are never used.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        attn_implementation: str = "sdpa",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
        max_total_pixels: int = DEFAULT_MAX_TOTAL_PIXELS,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        local_files_only: bool = True,
    ) -> None:
        resolved = Path(model_path).expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError(
                f"Local Qwen3-VL model path is not a directory: {resolved}"
            )
        if dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be auto, bfloat16, float16, or float32")
        if not isinstance(device_map, str) or not device_map.strip():
            raise ValueError("device_map must be a non-empty string")
        if attn_implementation not in {"sdpa", "flash_attention_2", "eager"}:
            raise ValueError(
                "attn_implementation must be sdpa, flash_attention_2, or eager"
            )
        if local_files_only is not True:
            raise ValueError(
                "Local Qwen inference requires local_files_only=True; network "
                "model resolution is intentionally disabled"
            )
        self.model_path = resolved
        self.dtype = dtype
        self.device_map = device_map.strip()
        self.attn_implementation = attn_implementation
        self.max_new_tokens = _positive_int(max_new_tokens, "max_new_tokens")
        self.max_image_pixels = _positive_int(max_image_pixels, "max_image_pixels")
        self.max_total_pixels = _positive_int(max_total_pixels, "max_total_pixels")
        self.max_image_bytes = _positive_int(max_image_bytes, "max_image_bytes")
        self.local_files_only = True
        self._model_type: str | None = None
        self._model_identity: dict[str, Any] | None = None
        self._torch: Any = None
        self._auto_model_class: Any = None
        self._auto_processor_class: Any = None
        self._model: Any = None
        self._processor: Any = None
        self._last_generation_result: LocalGenerationResult | None = None

    def preflight(self) -> None:
        """Import and validate the local backend without loading model weights."""

        if self._auto_model_class is not None:
            return
        config_path = self.model_path / "config.json"
        config, raw_config = _read_local_model_config(self.model_path)
        model_type = _validated_model_type(config, config_path)
        architectures = _validate_visual_capability(
            config,
            model_type=model_type,
            config_path=config_path,
        )
        try:
            import torch
            import transformers
        except ImportError as exc:  # pragma: no cover - optional environment
            raise QwenClientError(
                "Local Qwen inference requires PyTorch and Transformers; install "
                "requirements-local.txt"
            ) from exc

        transformers_version = getattr(transformers, "__version__", "unknown")
        try:
            AutoProcessor = getattr(transformers, "AutoProcessor")
            if AutoProcessor is None:
                raise AttributeError("AutoProcessor is None")
        except (AttributeError, ImportError, ModuleNotFoundError, RuntimeError) as exc:
            raise QwenClientError(
                "Installed Transformers does not expose AutoProcessor; upgrade "
                f"Transformers (found {transformers_version})"
            ) from exc

        if model_type in _QWEN3_VL_MODEL_TYPES:
            if _version_tuple(transformers_version) < MIN_TRANSFORMERS_VERSION:
                required = ".".join(str(value) for value in MIN_TRANSFORMERS_VERSION)
                raise QwenClientError(
                    f"Qwen3-VL requires transformers>={required}; found "
                    f"{transformers_version}"
                )
            try:
                model_class = getattr(transformers, "AutoModelForImageTextToText")
                if model_class is None:
                    raise AttributeError("AutoModelForImageTextToText is None")
            except (
                AttributeError,
                ImportError,
                ModuleNotFoundError,
                RuntimeError,
            ) as exc:
                raise QwenClientError(
                    "Installed Transformers does not expose AutoModelForImageTextToText"
                ) from exc
        elif model_type == _QWEN35_MODEL_TYPE:
            try:
                model_class = getattr(transformers, _QWEN35_MODEL_CLASS)
                if model_class is None:
                    raise AttributeError(f"{_QWEN35_MODEL_CLASS} is None")
            except (
                AttributeError,
                ImportError,
                ModuleNotFoundError,
                RuntimeError,
            ) as exc:
                raise QwenClientError(
                    "Local Qwen3.5 requires a newer Transformers build exposing "
                    f"{_QWEN35_MODEL_CLASS}; found transformers=="
                    f"{transformers_version}. Install the pinned isolated runtime "
                    "with scripts/setup_qwen35_runtime.sh. "
                    "Remote code and network fallback are intentionally disabled."
                ) from exc

        self._model_type = model_type
        self._torch = torch
        self._auto_model_class = model_class
        self._auto_processor_class = AutoProcessor
        identity: dict[str, Any] = {
            "schema_version": "1.0",
            "backend": "transformers-local",
            "model_path": str(self.model_path),
            "model_type": model_type,
            "model_class": getattr(model_class, "__name__", str(model_class)),
            "config": {
                "path": "config.json",
                "fingerprint": _sha256_bytes(raw_config),
                "architectures": architectures,
                "vision_config_model_type": config["vision_config"].get("model_type"),
                "image_token_id": config["image_token_id"],
            },
            "processor": {
                "class": getattr(AutoProcessor, "__name__", str(AutoProcessor)),
                **_processor_identity(self.model_path),
            },
            "chat_template": _chat_template_identity(self.model_path),
            "weights": _weights_identity(self.model_path),
            "runtime": {
                "python": platform.python_version(),
                "torch": str(getattr(torch, "__version__", "unknown")),
                "transformers": str(transformers_version),
                "cuda": getattr(getattr(torch, "version", None), "cuda", None),
                "dtype": self.dtype,
                "device_map": self.device_map,
                "attn_implementation": self.attn_implementation,
            },
            "generation": {
                "do_sample": False,
                "enable_thinking": False,
                "chat_template_mode": (
                    _QWEN35_NONTHINKING_TEMPLATE_MODE
                    if model_type == _QWEN35_MODEL_TYPE
                    else "checkpoint-native"
                ),
                "max_new_tokens": self.max_new_tokens,
                "use_cache": True,
            },
            "input_contract": {
                "max_image_pixels": self.max_image_pixels,
                "max_total_pixels": self.max_total_pixels,
                "max_image_bytes": self.max_image_bytes,
            },
        }
        identity["fingerprint"] = _json_fingerprint(identity)
        self._model_identity = identity

    @property
    def model_identity(self) -> dict[str, Any]:
        """Return a detached JSON-serializable identity for reproducibility."""

        self.preflight()
        if self._model_identity is None:  # pragma: no cover - defensive invariant
            raise QwenClientError("Local Qwen preflight did not create model identity")
        return copy.deepcopy(self._model_identity)

    def _load(self) -> None:
        if self._model is not None:
            return
        self.preflight()

        dtype: Any = self.dtype
        if self.dtype != "auto":
            dtype = getattr(self._torch, self.dtype)
        load_kwargs = {
            "dtype": dtype,
            "device_map": self.device_map,
            "attn_implementation": self.attn_implementation,
            "local_files_only": True,
            "trust_remote_code": False,
            "low_cpu_mem_usage": True,
        }
        self._processor = self._auto_processor_class.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        if self._model_type == _QWEN35_MODEL_TYPE:
            self._qwen35_chat_template = _qwen35_nonthinking_chat_template(
                getattr(self._processor, "chat_template", None)
            )
        else:
            self._qwen35_chat_template = None
        self._model = self._auto_model_class.from_pretrained(
            str(self.model_path), **load_kwargs
        )
        self._model.eval()

    def unload(self) -> None:
        """Release loaded weights while retaining the frozen backend identity."""

        model = self._model
        processor = self._processor
        self._model = None
        self._processor = None
        del model
        del processor
        gc.collect()

        cuda = getattr(self._torch, "cuda", None)
        empty_cache = getattr(cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()

    @property
    def last_generation_metadata(self) -> dict[str, Any] | None:
        """Return detached telemetry for the latest successful generation."""

        if self._last_generation_result is None:
            return None
        return copy.deepcopy(self._last_generation_result.metadata())

    def generate_with_metadata(
        self,
        payload: Mapping[str, Any],
        *,
        max_new_tokens: int | None = None,
    ) -> LocalGenerationResult:
        """Generate once with an optional per-call budget override."""

        generation_budget = (
            self.max_new_tokens
            if max_new_tokens is None
            else _positive_int(max_new_tokens, "max_new_tokens")
        )
        self._last_generation_result = None
        self._load()
        messages = openai_payload_to_qwen_messages(
            payload,
            max_image_pixels=self.max_image_pixels,
            max_total_pixels=self.max_total_pixels,
            max_image_bytes=self.max_image_bytes,
        )
        try:
            template_kwargs: dict[str, Any] = {"enable_thinking": False}
            if self._model_type == _QWEN35_MODEL_TYPE:
                template_kwargs["chat_template"] = self._qwen35_chat_template
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                **template_kwargs,
            )
            inputs = inputs.to(self._model.device)
            with self._torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=generation_budget,
                    use_cache=True,
                )
            trimmed_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            decoded = self._processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        finally:
            for message in messages:
                _close_content_images(message.get("content"))
        if len(decoded) != 1 or not isinstance(decoded[0], str):
            raise QwenClientError("Local Qwen3-VL returned an invalid text batch")
        generated_tokens = len(trimmed_ids[0])
        eos_ids = _eos_token_ids(self._model, self._processor)
        last_token_id = _last_token_id(trimmed_ids[0])
        eos_detected = (
            last_token_id in eos_ids
            if eos_ids and last_token_id is not None
            else None
        )
        result = LocalGenerationResult(
            text=decoded[0],
            generated_tokens=generated_tokens,
            max_new_tokens=generation_budget,
            hit_token_limit=generated_tokens >= generation_budget,
            eos_detected=eos_detected,
        )
        self._last_generation_result = result
        return result

    def __call__(self, payload: Mapping[str, Any]) -> str:
        return self.generate_with_metadata(payload).text


class LocalQwen3VLClient:
    """Material-analysis client with the same contract as the remote client."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        model: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        attn_implementation: str = "sdpa",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
        max_total_pixels: int = DEFAULT_MAX_TOTAL_PIXELS,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        raw_output_path: str | Path | None = None,
        runner: Callable[[Mapping[str, Any]], str] | None = None,
    ) -> None:
        if runner is None:
            if model_path is None:
                raise ValueError("model_path is required for local Qwen3-VL inference")
            runner = TransformersQwen3VLRunner(
                model_path,
                dtype=dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
                max_new_tokens=max_new_tokens,
                max_image_pixels=max_image_pixels,
                max_total_pixels=max_total_pixels,
                max_image_bytes=max_image_bytes,
            )
        if model is not None:
            model_name = model.strip()
            if not model_name:
                raise ValueError("model must be a non-empty string")
        elif model_path is not None:
            model_name = Path(model_path).expanduser().name
        else:
            model_name = DEFAULT_LOCAL_MODEL
        self.model = model_name
        self._runner = runner
        self.raw_output_path = (
            Path(raw_output_path).expanduser().resolve()
            if raw_output_path is not None
            else None
        )

    @property
    def model_identity(self) -> dict[str, Any]:
        """Expose runner identity through the client used by staged workflows."""

        identity = getattr(self._runner, "model_identity", None)
        if isinstance(identity, Mapping):
            return copy.deepcopy(dict(identity))
        return {
            "schema_version": "1.0",
            "backend": "custom-local-runner",
            "model": self.model,
        }

    def unload(self) -> None:
        """Release runner resources when the backend supports explicit unload."""

        unload = getattr(self._runner, "unload", None)
        if callable(unload):
            unload()

    def analyze(
        self,
        views: list[dict[str, Any]],
        parts: list[dict[str, Any]],
        candidate_materials: list[dict[str, Any]],
        max_assignments: int | None = None,
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        payload = build_analysis_payload(
            self.model,
            views,
            parts,
            candidate_materials,
            max_assignments=max_assignments,
        )
        if dry_run:
            return payload
        require_user_reference_views(views)
        raw_content = self._runner(payload)
        if not isinstance(raw_content, str):
            raise QwenClientError("Local Qwen3-VL runner must return text")
        if self.raw_output_path is not None:
            self.raw_output_path.parent.mkdir(parents=True, exist_ok=True)
            self.raw_output_path.write_text(raw_content, encoding="utf-8")
        plan = parse_plan_content(raw_content)
        return validate_analysis_result(
            plan,
            views,
            parts,
            candidate_materials,
            max_assignments=max_assignments,
        )


TransformersLocalQwenRunner = TransformersQwen3VLRunner
LocalQwenClient = LocalQwen3VLClient


__all__ = [
    "DEFAULT_LOCAL_MODEL",
    "LocalGenerationResult",
    "LocalQwenClient",
    "LocalQwen3VLClient",
    "TransformersLocalQwenRunner",
    "TransformersQwen3VLRunner",
    "decode_data_image",
    "openai_payload_to_qwen_messages",
]
