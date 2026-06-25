#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pipeline_runner


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data"
DEFAULT_GENERATION_OUTPUT_DIR = PROJECT_ROOT / "downloads"
DEFAULT_INTERMEDIATE_OUTPUT_DIR = PROJECT_ROOT / "output_intermediate"
DEFAULT_FINAL_OUTPUT_DIR = PROJECT_ROOT / "output_final"
DEFAULT_RESULT_JSON = PROJECT_ROOT / "pipeline_result.json"


def log(message: str) -> None:
    print(message, flush=True)


def first_existing_path(candidates: list[str | Path]) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return path.resolve()
    return None


def configure_runtime() -> dict[str, str | None]:
    blender = first_existing_path(
        [
            os.getenv("BLENDER_BIN") or "",
            shutil.which("blender") or "",
            "/opt/blender/blender",
            "/usr/local/bin/blender",
        ]
    )
    if blender:
        os.environ["BLENDER_BIN"] = str(blender)

    isaac_python = first_existing_path(
        [
            os.getenv("ISAAC_PYTHON") or "",
            "/home/user/isaacsim500/python.sh",
            "/isaac-sim/python.sh",
            "/opt/isaac-sim/python.sh",
        ]
    )
    if isaac_python:
        os.environ["ISAAC_PYTHON"] = str(isaac_python)
        os.environ.setdefault("ISAACSIM_ROOT", str(isaac_python.parent))

    os.environ.setdefault("ROOT_DIR", str(PROJECT_ROOT))
    os.environ.setdefault("REFINE_MESH_TEMP_UPLOAD", "uguu")

    return {
        "ROOT_DIR": os.getenv("ROOT_DIR"),
        "BLENDER_BIN": os.getenv("BLENDER_BIN"),
        "ISAACSIM_ROOT": os.getenv("ISAACSIM_ROOT"),
        "ISAAC_PYTHON": os.getenv("ISAAC_PYTHON"),
        "REFINE_MESH_TEMP_UPLOAD": os.getenv("REFINE_MESH_TEMP_UPLOAD"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete asset pipeline: Hunyuan generation, refine mesh, "
            "Blender postprocess, Isaac physics, and USD collect."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="image folder for Hunyuan generation")
    source.add_argument("--prompt", help="text prompt for Hunyuan generation")
    source.add_argument("--image-url", help="image URL for Hunyuan generation")
    source.add_argument("--existing-glb", help="existing GLB file or folder; skips Hunyuan generation")
    source.add_argument("--manual-glb", help="hand-made/general GLB file or folder; runs GLB -> USD -> physics -> collect")

    parser.add_argument("--output-dir", default=str(DEFAULT_GENERATION_OUTPUT_DIR), help="Hunyuan generation output dir")
    parser.add_argument("--intermediate-output-dir", default=str(DEFAULT_INTERMEDIATE_OUTPUT_DIR), help="physics USD output dir")
    parser.add_argument("--final-output-dir", default=str(DEFAULT_FINAL_OUTPUT_DIR), help="collected final USD output dir")
    parser.add_argument("--result-json", default=str(DEFAULT_RESULT_JSON), help="where to write the final JSON result")

    parser.add_argument("--face-count", type=int, default=150000, help="Hunyuan pro face count")
    parser.add_argument("--download-preview", action="store_true", help="download Hunyuan preview image")

    parser.add_argument("--len-x", type=float, default=0.4, help="target X size in meters")
    parser.add_argument("--len-y", type=float, default=0.3, help="target Y size in meters")
    parser.add_argument("--len-z", type=float, default=0.3, help="target Z size in meters")
    parser.add_argument("--orientation", default="X=L,Y=M,Z=S", help="axis map, for example X=L,Y=M,Z=S")
    parser.add_argument("--manual-align", action="store_true", help="for --manual-glb, align GLB before USD conversion")
    parser.add_argument("--manual-resize", action="store_true", help="for --manual-glb, resize GLB before USD conversion")
    parser.add_argument("--usd-format", default="usd", choices=["usd", "usda", "usdc"], help="USD format for manual/general GLB mode")
    parser.add_argument("--visible-only", action="store_true", help="export only visible objects when converting GLB to USD")

    parser.add_argument("--material", default="plastic", help="material name in materials.json")
    parser.add_argument("--approx", default="sdf", help="collision approximation, for example sdf or convexDecomposition")
    parser.add_argument("--set-mass", type=float, help="total asset mass in kg; omitted means auto estimate")

    parser.add_argument("--skip-refine", action="store_true", help="skip refine mesh")
    parser.add_argument("--refine-output-dir", help="refine mesh output dir")
    parser.add_argument("--refine-config-path", help="refine mesh config path")
    parser.add_argument("--refine-temp-upload", default=None, help="temporary upload provider; use none to disable")
    parser.add_argument("--refine-fail-on-qc-error", action="store_true", help="fail when refine QC status is fail")
    return parser


def ensure_generation_source(args: argparse.Namespace) -> None:
    if args.existing_glb or args.manual_glb:
        return
    if args.prompt or args.image_url:
        return
    input_dir = Path(args.input_dir).expanduser()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input image directory does not exist: {input_dir}")


def write_result(result: dict[str, Any], path: str) -> None:
    result_path = Path(path).expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"Result JSON: {result_path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.manual_glb:
        postprocess_result = pipeline_runner.run_glb_physics_job(
            input_path=args.manual_glb,
            intermediate_output_dir=args.intermediate_output_dir,
            final_output_dir=args.final_output_dir,
            material=args.material,
            set_mass=args.set_mass,
            approx=args.approx,
            usd_format=args.usd_format,
            visible_only=args.visible_only,
            align=args.manual_align,
            axis_map=args.orientation,
            resize=args.manual_resize,
            len_x=args.len_x,
            len_y=args.len_y,
            len_z=args.len_z,
            log_cb=log,
        )
        return {
            "runtime": pipeline_runner.runtime_summary(),
            "mode": "manual_glb",
            "generation": None,
            "refine_mesh": None,
            "postprocess": postprocess_result,
        }

    if args.existing_glb:
        process_input = args.existing_glb
        refine_result = None
        if not args.skip_refine:
            refine_result = pipeline_runner.run_refine_mesh_job(
                input_path=process_input,
                output_dir=args.refine_output_dir,
                config_path=args.refine_config_path,
                temp_upload=args.refine_temp_upload,
                fail_on_qc_error=args.refine_fail_on_qc_error,
                log_cb=log,
            )
            process_input = refine_result["postprocess_input_path"]

        postprocess_result = pipeline_runner.run_process_model_job(
            input_path=process_input,
            len_x=args.len_x,
            len_y=args.len_y,
            len_z=args.len_z,
            orientation=args.orientation,
            intermediate_output_dir=args.intermediate_output_dir,
            final_output_dir=args.final_output_dir,
            set_mass=args.set_mass,
            material=args.material,
            approx=args.approx,
            log_cb=log,
        )
        return {
            "runtime": pipeline_runner.runtime_summary(),
            "generation": None,
            "refine_mesh": refine_result,
            "postprocess": postprocess_result,
        }

    result = pipeline_runner.run_generate_and_process_model_job(
        output_dir=args.output_dir,
        intermediate_output_dir=args.intermediate_output_dir,
        final_output_dir=args.final_output_dir,
        len_x=args.len_x,
        len_y=args.len_y,
        len_z=args.len_z,
        orientation=args.orientation,
        set_mass=args.set_mass,
        material=args.material,
        approx=args.approx,
        input_dir=args.input_dir,
        prompt=args.prompt,
        image_url=args.image_url,
        face_count=args.face_count,
        download_preview=args.download_preview,
        refine_mesh=not args.skip_refine,
        refine_output_dir=args.refine_output_dir,
        refine_config_path=args.refine_config_path,
        refine_temp_upload=args.refine_temp_upload,
        refine_fail_on_qc_error=args.refine_fail_on_qc_error,
        log_cb=log,
    )
    result["runtime"] = pipeline_runner.runtime_summary()
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = configure_runtime()
    log("Runtime:")
    for key, value in runtime.items():
        log(f"  {key}={value}")

    try:
        ensure_generation_source(args)
        result = run(args)
        write_result(result, args.result_json)
        log("Pipeline finished.")
        return 0
    except Exception as exc:
        log(f"Pipeline failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
