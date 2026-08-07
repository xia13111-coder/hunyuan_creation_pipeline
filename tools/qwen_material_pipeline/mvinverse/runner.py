#!/usr/bin/env python3
"""Offline MVInverse subprocess runner used by :mod:`mvinverse_adapter`.

This file deliberately contains no weights and performs no download.  The
adapter supplies an external MVInverse checkout, Python interpreter, and
either a local Hugging Face checkpoint directory or a torch checkpoint file.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence


MAP_CHANNELS = {
    "albedo": 3,
    "metallic": 1,
    "roughness": 1,
    "normal": 3,
    "shading": 3,
}


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _load_model_class(repo: Path) -> Any:
    sys.path.insert(0, str(repo))
    module = importlib.import_module("mvinverse.models.mvinverse")
    module_path = Path(module.__file__).resolve(strict=True)
    if not _under(module_path, repo):
        raise RuntimeError(
            f"Imported MVInverse code is outside the requested repository: {module_path}"
        )
    return module.MVInverse


def _load_model(
    model_class: Any,
    checkpoint: Path,
    checkpoint_format: str,
    device: Any,
    torch: Any,
) -> Any:
    if checkpoint_format == "huggingface_directory":
        # Hugging Face's mixin accepts a local directory.  Offline environment
        # variables are also set by the parent adapter and again in main().
        model = model_class.from_pretrained(str(checkpoint))
    elif checkpoint_format == "torch_checkpoint":
        model = model_class()
        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint is not compatible with this MVInverse checkout; "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
    else:  # pragma: no cover - argparse constrains this for normal callers.
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_format}")
    return model.to(device).eval()


def _load_images(input_dir: Path, expected_count: int, torch: Any) -> Any:
    import numpy as np
    from PIL import Image

    paths = sorted(input_dir.glob("*.png"))
    if len(paths) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} staged PNG inputs, found {len(paths)}"
        )
    tensors = []
    common_size: tuple[int, int] | None = None
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.width % 14 or image.height % 14:
                raise RuntimeError(f"Staged input dimensions must be divisible by 14: {path}")
            if common_size is None:
                common_size = image.size
            elif image.size != common_size:
                raise RuntimeError("All staged MVInverse inputs must have identical dimensions")
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
    if not tensors:
        raise RuntimeError("No staged MVInverse input was found")
    return torch.stack(tensors, dim=0)


def _validate_prediction(
    name: str,
    value: Any,
    frame_count: int,
    expected_height: int,
    expected_width: int,
    torch: Any,
) -> dict[str, float]:
    channels = MAP_CHANNELS[name]
    if value.ndim != 5 or value.shape[0] != 1 or value.shape[1] != frame_count:
        raise RuntimeError(f"MVInverse returned an invalid {name} tensor shape: {value.shape}")
    if value.shape[2:4] != (expected_height, expected_width):
        raise RuntimeError(
            f"MVInverse returned invalid {name} spatial dimensions: {value.shape}"
        )
    if value.shape[-1] != channels:
        raise RuntimeError(f"MVInverse returned invalid {name} channels: {value.shape}")
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"MVInverse returned non-finite values in {name}")
    minimum = float(value.amin().item())
    maximum = float(value.amax().item())
    expected_minimum = -1.0 if name == "normal" else 0.0
    # A tiny tolerance accommodates low-precision kernels without hiding a
    # materially invalid model output behind PNG clipping.
    tolerance = 1e-3
    if minimum < expected_minimum - tolerance or maximum > 1.0 + tolerance:
        raise RuntimeError(
            f"MVInverse returned out-of-range {name} values: min={minimum}, max={maximum}"
        )
    return {"minimum": minimum, "maximum": maximum}


def _save_predictions(
    result: dict[str, Any],
    output_dir: Path,
    frame_count: int,
    expected_height: int,
    expected_width: int,
    torch: Any,
) -> dict[str, dict[str, float]]:
    import numpy as np
    from PIL import Image

    ranges: dict[str, dict[str, float]] = {}
    for name in MAP_CHANNELS:
        if name not in result:
            raise RuntimeError(f"MVInverse result is missing {name!r}")
        ranges[name] = _validate_prediction(
            name,
            result[name],
            frame_count,
            expected_height,
            expected_width,
            torch,
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    for index in range(frame_count):
        for name in MAP_CHANNELS:
            frame = result[name][0, index].detach().float().cpu().numpy()
            if name == "normal":
                frame = frame * 0.5 + 0.5
            frame = np.clip(frame, 0.0, 1.0)
            if MAP_CHANNELS[name] == 1:
                frame = frame[..., 0]
            image = Image.fromarray((frame * 255.0).astype(np.uint8))
            image.save(output_dir / f"{index:03d}_{name}.png")
    return ranges


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-format",
        choices=("huggingface_directory", "torch_checkpoint"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-frames", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    started = time.perf_counter()
    args = build_parser().parse_args(argv)
    # Defense in depth: local checkpoints must remain local even if this runner
    # is invoked directly instead of through the adapter.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    repo = args.repo.expanduser().resolve(strict=True)
    input_dir = args.input_dir.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    if args.num_frames < 1:
        raise ValueError("--num-frames must be positive")

    import torch

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model_class = _load_model_class(repo)
    model = _load_model(
        model_class, checkpoint, args.checkpoint_format, device, torch
    )
    images = _load_images(input_dir, args.num_frames, torch).to(device)
    autocast = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    with torch.no_grad(), autocast:
        result = model(images[None])
    prediction_ranges = _save_predictions(
        result,
        output_dir,
        args.num_frames,
        int(images.shape[-2]),
        int(images.shape[-1]),
        torch,
    )
    cuda_memory = None
    if device.type == "cuda":
        cuda_memory = {
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    print(
        json.dumps(
            {
                "status": "SUCCESS",
                "frame_count": args.num_frames,
                "image_shape_nchw": list(images.shape),
                "output_dir": str(output_dir),
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "cuda_memory": cuda_memory,
                "prediction_ranges": prediction_ranges,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
