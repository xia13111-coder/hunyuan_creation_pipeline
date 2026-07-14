#!/usr/bin/env python3

"""User-facing asset pipeline CLI implementation.

Call flow:
main -> runtime.configure_runtime -> ensure_generation_source -> run

run branches:
- manual_stp -> workflows.run_stp_physics_job
- sam3d_input -> workflows.run_sam3d_image_and_process_model_job
- sam3d_glb / existing_glb -> refine job -> workflows.run_process_model_job
- generated input -> workflows.run_generate_and_process_model_job
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import runtime
from .jobs.refine import run_refine_mesh_job
from .workflows import (
    DEFAULT_MANUAL_SDF_RESOLUTION,
    run_generate_and_process_model_job,
    run_process_model_job,
    run_sam3d_image_and_process_model_job,
    run_stp_physics_job,
)


PROJECT_ROOT = runtime.project_root()

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data"
DEFAULT_GENERATION_OUTPUT_DIR = PROJECT_ROOT / "downloads"
DEFAULT_INTERMEDIATE_OUTPUT_DIR = PROJECT_ROOT / "output_intermediate"
DEFAULT_FINAL_OUTPUT_DIR = PROJECT_ROOT / "output_final"
DEFAULT_RESULT_JSON = PROJECT_ROOT / "pipeline_result.json"


def log(message: str) -> None:
    print(message, flush=True)


configure_runtime = runtime.configure_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete asset pipeline: Hunyuan generation, refine mesh, "
            "Blender postprocess, Isaac physics, and USD collect."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="image folder for Hunyuan generation",
    )
    source.add_argument("--prompt", help="text prompt for Hunyuan generation")
    source.add_argument("--image-url", help="image URL for Hunyuan generation")
    source.add_argument(
        "--existing-glb", help="existing GLB file or folder; skips Hunyuan generation"
    )
    source.add_argument(
        "--sam3d-input",
        help="SAM3D source image or image folder (PNG/JPG/WEBP/BMP); segments, reconstructs, then refines",
    )
    source.add_argument(
        "--sam3d-glb",
        help="existing SAM3D GLB file or folder; skips image reconstruction but still runs Hunyuan refine",
    )
    source.add_argument(
        "--manual-stp",
        help="hand-modeled STEP/STP CAD file or folder; runs CAD -> USD -> physics -> collect",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_GENERATION_OUTPUT_DIR),
        help="Hunyuan generation directory or SAM3D workspace root",
    )
    parser.add_argument(
        "--intermediate-output-dir",
        default=str(DEFAULT_INTERMEDIATE_OUTPUT_DIR),
        help="physics USD output dir",
    )
    parser.add_argument(
        "--final-output-dir",
        default=str(DEFAULT_FINAL_OUTPUT_DIR),
        help="collected final USD output dir",
    )
    parser.add_argument(
        "--result-json",
        default=str(DEFAULT_RESULT_JSON),
        help="where to write the final JSON result",
    )

    parser.add_argument(
        "--face-count", type=int, default=150000, help="Hunyuan pro face count"
    )
    parser.add_argument(
        "--download-preview", action="store_true", help="download Hunyuan preview image"
    )
    parser.add_argument(
        "--sam3d-mode",
        choices=["auto", "single", "multi"],
        default="auto",
        help="SAM3D input mode; auto uses single for one image and multi for multiple images",
    )
    parser.add_argument(
        "--sam3d-prompt",
        help="SAM3 target-object segmentation prompt, for example 'metal shelves'; required with --sam3d-input",
    )
    parser.add_argument(
        "--sam3d-seed", type=int, default=42, help="SAM3D geometry reconstruction seed"
    )
    parser.add_argument(
        "--sam3d-steps",
        type=int,
        default=50,
        help="SAM3D stage-1 geometry sampling steps; stage 2 remains fixed at 25",
    )

    parser.add_argument(
        "--len-x", type=float, default=0.4, help="target X size in meters"
    )
    parser.add_argument(
        "--len-y", type=float, default=0.3, help="target Y size in meters"
    )
    parser.add_argument(
        "--len-z", type=float, default=0.3, help="target Z size in meters"
    )
    parser.add_argument(
        "--orientation", default="X=L,Y=M,Z=S", help="axis map, for example X=L,Y=M,Z=S"
    )
    parser.add_argument(
        "--cad-usd-output-dir",
        help="for --manual-stp, where Isaac Sim writes CAD-converted USD files",
    )
    parser.add_argument(
        "--cad-converter-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="for --manual-stp, pass an Omniverse CAD converter option; may be repeated",
    )
    parser.add_argument(
        "--manual-sdf-resolution",
        type=int,
        default=DEFAULT_MANUAL_SDF_RESOLUTION,
        help=(
            "SDF resolution for --manual-stp; lower values improve physics "
            f"performance (default: {DEFAULT_MANUAL_SDF_RESOLUTION})"
        ),
    )
    parser.add_argument(
        "--material", default="plastic", help="material name in materials.json"
    )
    parser.add_argument(
        "--approx",
        default="sdf",
        help="collision approximation, for example sdf or convexDecomposition",
    )
    parser.add_argument(
        "--set-mass",
        type=float,
        help="total asset mass in kg; omitted means auto estimate",
    )

    parser.add_argument("--skip-refine", action="store_true", help="skip refine mesh")
    parser.add_argument("--refine-output-dir", help="refine mesh output dir")
    parser.add_argument("--refine-config-path", help="refine mesh config path")
    parser.add_argument(
        "--refine-temp-upload",
        default=None,
        help="temporary upload provider; use none to disable",
    )
    parser.add_argument(
        "--refine-fail-on-qc-error",
        action="store_true",
        help="fail when refine QC status is fail",
    )
    return parser


def ensure_generation_source(args: argparse.Namespace) -> None:
    if args.existing_glb or args.sam3d_glb or args.manual_stp:
        return
    if args.sam3d_input:
        input_path = Path(args.sam3d_input).expanduser()
        if not input_path.exists():
            raise FileNotFoundError(f"SAM3D input does not exist: {input_path}")
        if not args.sam3d_prompt:
            raise ValueError("--sam3d-prompt is required when using --sam3d-input")
        return
    if args.prompt or args.image_url:
        return
    input_dir = Path(args.input_dir).expanduser()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input image directory does not exist: {input_dir}")


def write_result(result: dict[str, Any], path: str) -> None:
    result_path = Path(path).expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log(f"Result JSON: {result_path}")


def run_glb_refine_then_process(
    args: argparse.Namespace, input_path: str, mode: str
) -> dict[str, Any]:
    """Run the common GLB path: optional refine mesh, then Blender/Isaac postprocess."""
    process_input = input_path
    refine_result = None
    if not args.skip_refine:
        refine_result = run_refine_mesh_job(
            input_path=process_input,
            output_dir=args.refine_output_dir,
            config_path=args.refine_config_path,
            temp_upload=args.refine_temp_upload,
            fail_on_qc_error=args.refine_fail_on_qc_error,
            log_cb=log,
        )
        process_input = refine_result["postprocess_input_path"]

    postprocess_result = run_process_model_job(
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
        "runtime": runtime.runtime_summary(),
        "mode": mode,
        "generation": None,
        "refine_mesh": refine_result,
        "postprocess": postprocess_result,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Choose the source workflow and delegate to its owning module."""
    if args.manual_stp:
        postprocess_result = run_stp_physics_job(
            input_path=args.manual_stp,
            intermediate_output_dir=args.intermediate_output_dir,
            final_output_dir=args.final_output_dir,
            cad_usd_output_dir=args.cad_usd_output_dir,
            cad_converter_options=args.cad_converter_option,
            material=args.material,
            set_mass=args.set_mass,
            approx=args.approx,
            sdf_resolution=args.manual_sdf_resolution,
            log_cb=log,
        )
        return {
            "runtime": runtime.runtime_summary(),
            "mode": "manual_stp",
            "generation": None,
            "refine_mesh": None,
            "postprocess": postprocess_result,
        }

    if args.sam3d_glb:
        return run_glb_refine_then_process(args, args.sam3d_glb, "sam3d_glb")

    if args.sam3d_input:
        result = run_sam3d_image_and_process_model_job(
            input_path=args.sam3d_input,
            output_dir=args.output_dir,
            intermediate_output_dir=args.intermediate_output_dir,
            final_output_dir=args.final_output_dir,
            len_x=args.len_x,
            len_y=args.len_y,
            len_z=args.len_z,
            orientation=args.orientation,
            sam3d_mode=args.sam3d_mode,
            sam3d_prompt=args.sam3d_prompt,
            sam3d_seed=args.sam3d_seed,
            sam3d_steps=args.sam3d_steps,
            set_mass=args.set_mass,
            material=args.material,
            approx=args.approx,
            refine_mesh=not args.skip_refine,
            refine_output_dir=args.refine_output_dir,
            refine_config_path=args.refine_config_path,
            refine_temp_upload=args.refine_temp_upload,
            refine_fail_on_qc_error=args.refine_fail_on_qc_error,
            log_cb=log,
        )
        result["mode"] = "sam3d_input"
        result["runtime"] = runtime.runtime_summary()
        return result

    if args.existing_glb:
        return run_glb_refine_then_process(args, args.existing_glb, "existing_glb")

    result = run_generate_and_process_model_job(
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
    result["runtime"] = runtime.runtime_summary()
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        runtime_config = configure_runtime()
        log("Runtime:")
        for key, value in runtime_config.items():
            log(f"  {key}={value}")
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
