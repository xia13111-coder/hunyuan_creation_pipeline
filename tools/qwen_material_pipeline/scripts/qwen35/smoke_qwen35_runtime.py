#!/usr/bin/env python3
"""Run an offline CUDA load/generate/unload/reload smoke for local Qwen3.5."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

TOOLS_DIR = Path(__file__).resolve().parents[3]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from qwen_material_pipeline.qwen.local_vl import (  # noqa: E402
    TransformersQwen3VLRunner,
)


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cycles", type=_positive_int, default=2)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=24)
    parser.add_argument("--image-side", type=_positive_int, default=224)
    parser.add_argument("--minimum-free-gib", type=_positive_float, default=14.0)
    return parser.parse_args()


def _payload(side: int) -> dict[str, object]:
    from PIL import Image

    image = Image.new("RGB", (side, side), (120, 135, 150))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image.close()
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are an offline runtime smoke test.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                    {
                        "type": "text",
                        "text": "Name the dominant color in at most three words.",
                    },
                ],
            },
        ]
    }


def main() -> int:
    args = _parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the isolated Qwen3.5 runtime")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    gib = 1024**3
    if free_bytes < args.minimum_free_gib * gib:
        raise RuntimeError(
            "Insufficient free CUDA memory for Qwen3.5 smoke: "
            f"free={free_bytes / gib:.2f} GiB, "
            f"required={args.minimum_free_gib:.2f} GiB"
        )

    side = args.image_side
    payload = _payload(side)
    runner = TransformersQwen3VLRunner(
        args.model_path,
        dtype="bfloat16",
        device_map="cuda:0",
        attn_implementation="sdpa",
        max_new_tokens=args.max_new_tokens,
        max_image_pixels=side * side,
        max_total_pixels=side * side,
    )
    identity = runner.model_identity
    cycles: list[dict[str, object]] = []
    for cycle in range(1, args.cycles + 1):
        torch.cuda.reset_peak_memory_stats(0)
        output = runner(payload)
        torch.cuda.synchronize(0)
        peak_allocated = torch.cuda.max_memory_allocated(0)
        peak_reserved = torch.cuda.max_memory_reserved(0)
        runner.unload()
        torch.cuda.synchronize(0)
        cycles.append(
            {
                "cycle": cycle,
                "output": output.strip(),
                "peak_allocated_gib": round(peak_allocated / gib, 3),
                "peak_reserved_gib": round(peak_reserved / gib, 3),
                "allocated_after_unload_gib": round(
                    torch.cuda.memory_allocated(0) / gib, 3
                ),
                "reserved_after_unload_gib": round(
                    torch.cuda.memory_reserved(0) / gib, 3
                ),
            }
        )
    result = {
        "schema_version": "qwen35-cuda-smoke/v1",
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gib": round(total_bytes / gib, 3),
        "gpu_free_before_gib": round(free_bytes / gib, 3),
        "model_type": identity["model_type"],
        "model_fingerprint": identity["fingerprint"],
        "cycles": cycles,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
