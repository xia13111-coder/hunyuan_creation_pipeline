#!/usr/bin/env python3
"""Download an official Qwen3-VL checkpoint to an explicit local directory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence


OFFICIAL_MODELS = (
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id", choices=OFFICIAL_MODELS, default=OFFICIAL_MODELS[0]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--parallel", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.parallel < 1 or args.parallel > 16:
        raise ValueError("--parallel must be between 1 and 16")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.cache_dir is not None:
        cache_dir = args.cache_dir.expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MODELSCOPE_CACHE"] = str(cache_dir)
    # ModelScope reads this setting at import time. Parallel ranged downloads
    # are substantially faster for the two multi-GB safetensor shards.
    os.environ["MODELSCOPE_DOWNLOAD_PARALLELS"] = str(args.parallel)
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Install tools/qwen_material_pipeline/requirements-local.txt first"
        ) from exc

    resolved = Path(
        snapshot_download(
            args.model_id,
            local_dir=str(output),
            max_workers=2,
        )
    ).resolve(strict=True)
    required = ("config.json", "model.safetensors.index.json")
    missing = [name for name in required if not (resolved / name).is_file()]
    shards = sorted(resolved.glob("model-*.safetensors"))
    if missing or not shards:
        raise RuntimeError(
            f"Checkpoint download is incomplete; missing={missing}, shards={len(shards)}"
        )
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "output": str(resolved),
                "shard_count": len(shards),
                "bytes": sum(path.stat().st_size for path in shards),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
