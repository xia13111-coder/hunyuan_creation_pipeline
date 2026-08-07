#!/usr/bin/env python3

"""Run a vendor multi-view script with the project-owned local model config."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-script", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def _load_vendor(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("sam3d_multiview_vendor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load multi-view script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    vendor_script = args.vendor_script.expanduser().resolve(strict=True)
    model_config = args.model_config.expanduser().resolve(strict=True)
    if not vendor_script.is_file() or not model_config.is_file():
        raise FileNotFoundError("Vendor script and local model config must be files")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    module = _load_vendor(vendor_script)
    original_inference = getattr(module, "Inference", None)
    vendor_main = getattr(module, "main", None)
    if not callable(original_inference) or not callable(vendor_main):
        raise RuntimeError(f"Unsupported multi-view script interface: {vendor_script}")

    def local_inference(_vendor_config: str, *values, **keywords):
        return original_inference(str(model_config), *values, **keywords)

    module.Inference = local_inference
    forwarded = list(args.arguments)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    original_argv = sys.argv
    sys.argv = [str(vendor_script), *forwarded]
    try:
        result = vendor_main()
    finally:
        sys.argv = original_argv
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
