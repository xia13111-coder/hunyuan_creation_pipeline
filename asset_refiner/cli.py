from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import apply_overrides, load_config
from .env import load_default_env_files
from .exceptions import AssetRefinerError
from .runner import run_refinement


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m asset_refiner",
        description="Refine one Hunyuan3D model as a single whole asset.",
    )
    parser.add_argument("--input", help="Input Hunyuan3D model, e.g. .glb/.gltf/.obj/.fbx")
    parser.add_argument("--output", help="Output directory for the refined asset")
    parser.add_argument("--config", help="YAML config file")
    parser.add_argument("--blender", help="Override Blender executable path")
    parser.add_argument("--retopo-target", help="Use this existing retopologized model as the final topology target")
    parser.add_argument("--hunyuan-input-url", help="Public or signed URL that Tencent Hunyuan3D API can fetch")
    parser.add_argument(
        "--hunyuan-upload-input",
        help=(
            "Local model file to upload for Hunyuan API instead of --input. "
            "--input is still used as the local high/source model for QC and texture migration."
        ),
    )
    parser.add_argument(
        "--env-file",
        action="append",
        help="Load local environment variables from this file before running. Can be passed more than once.",
    )
    parser.add_argument(
        "--hunyuan-temp-upload",
        choices=["uguu"],
        help="Upload local --input to a temporary public file host for Hunyuan API input. This exposes the model URL publicly for the host retention period.",
    )
    parser.add_argument(
        "--hunyuan-local-postprocess",
        action="store_true",
        help="Use Hunyuan ReduceFace as the low-poly target, then run local Blender UV and texture migration.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the backend command without running Blender")
    parser.add_argument("--print-config", action="store_true", help="Print the resolved config and exit")
    parser.add_argument(
        "--fail-on-qc-error",
        action="store_true",
        help="Return exit code 2 if machine-readable QC status is fail",
    )
    return parser


def _overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.blender:
        overrides.setdefault("backend", {})["blender_executable"] = args.blender
    if args.retopo_target:
        retopo = overrides.setdefault("retopology", {})
        retopo["method"] = "external_target_project"
        retopo["target_path"] = args.retopo_target
    if args.hunyuan_input_url:
        overrides.setdefault("hunyuan", {})["input_url"] = args.hunyuan_input_url
    if getattr(args, "hunyuan_upload_input", None):
        overrides.setdefault("hunyuan", {})["upload_input_path"] = args.hunyuan_upload_input
    if args.hunyuan_temp_upload:
        temp_upload = overrides.setdefault("hunyuan", {}).setdefault("temp_upload", {})
        temp_upload["enabled"] = True
        temp_upload["provider"] = args.hunyuan_temp_upload
    if getattr(args, "hunyuan_local_postprocess", False):
        hunyuan = overrides.setdefault("hunyuan", {})
        hunyuan.setdefault("local_postprocess", {})["enabled"] = True
    if args.fail_on_qc_error:
        overrides.setdefault("qc", {})["fail_on_error"] = True
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    load_default_env_files(args.env_file)
    overrides = _overrides_from_args(args)

    if args.print_config:
        config = apply_overrides(load_config(args.config), overrides)
        print(json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    if not args.input or not args.output:
        parser.error("--input and --output are required unless --print-config is used")

    try:
        result = run_refinement(
            input_path=args.input,
            output_dir=Path(args.output),
            config_path=args.config,
            overrides=overrides,
            dry_run=args.dry_run,
        )
    except (AssetRefinerError, OSError, ValueError) as exc:
        print(f"asset_refiner: error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(" ".join(result.command))
        return 0

    report = result.report or {}
    status = report.get("status", "unknown")
    exported = report.get("exports", [])
    print(f"Refined asset written to: {result.output_dir}")
    print(f"QC report: {result.report_path}")
    print(f"QC status: {status}")
    for item in exported:
        if isinstance(item, dict) and item.get("path"):
            print(f"Export: {item['path']}")

    fail_on_error = bool(report.get("config", {}).get("qc", {}).get("fail_on_error"))
    if fail_on_error and status == "fail":
        return 2
    return 0
