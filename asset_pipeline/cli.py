#!/usr/bin/env python3

"""User-facing asset pipeline CLI implementation.

Call flow:
main -> runtime.configure_runtime -> ensure_generation_source -> run

run branches:
- manual_stp -> manual_cad.run_manual_cad_workflow
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
from .jobs.cad import validate_cad_input_path
from .visual_materials import load_visual_material_config, parse_visual_references
from .jobs.refine import run_refine_mesh_job
from .manual_cad import DEFAULT_MANUAL_SDF_RESOLUTION, run_manual_cad_workflow
from .workflows import (
    run_generate_and_process_model_job,
    run_process_model_job,
    run_sam3d_image_and_process_model_job,
)


PROJECT_ROOT = runtime.project_root()

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data"
DEFAULT_GENERATION_OUTPUT_DIR = PROJECT_ROOT / "downloads"
DEFAULT_INTERMEDIATE_OUTPUT_DIR = PROJECT_ROOT / "output_intermediate"
DEFAULT_FINAL_OUTPUT_DIR = PROJECT_ROOT / "output_final"
DEFAULT_RESULT_JSON = PROJECT_ROOT / "pipeline_result.json"
DEFAULT_LEN_X = 0.4
DEFAULT_LEN_Y = 0.3
DEFAULT_LEN_Z = 0.3
DEFAULT_ORIENTATION = "X=L,Y=M,Z=S"


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
        help=(
            "hand-modeled STEP/STP CAD file or folder; preserves CAD dimensions and "
            "runs CAD -> USD -> physics preparation -> optional visual materials -> collect; "
            "automatic materials also require final delivery validation"
        ),
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
        "--sam3d-confidence-threshold",
        type=float,
        default=0.5,
        help=(
            "minimum SAM3 text-grounding confidence in [0, 1]; lower only when "
            "a correct prompt narrowly misses the default 0.5"
        ),
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
        "--len-x",
        type=float,
        default=None,
        help=f"GLB target X size in meters (default: {DEFAULT_LEN_X}); invalid with --manual-stp",
    )
    parser.add_argument(
        "--len-y",
        type=float,
        default=None,
        help=f"GLB target Y size in meters (default: {DEFAULT_LEN_Y}); invalid with --manual-stp",
    )
    parser.add_argument(
        "--len-z",
        type=float,
        default=None,
        help=f"GLB target Z size in meters (default: {DEFAULT_LEN_Z}); invalid with --manual-stp",
    )
    parser.add_argument(
        "--orientation",
        default=None,
        help=(
            f"GLB axis map (default: {DEFAULT_ORIENTATION}); invalid with "
            "--manual-stp because CAD orientation is preserved"
        ),
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
    parser.add_argument(
        "--auto-visual-materials",
        action="store_true",
        help=(
            "after STEP/STP physics geometry preparation, assign appearance "
            "materials from same-workpiece reference images before collection"
        ),
    )
    parser.add_argument(
        "--visual-reference",
        action="append",
        default=[],
        metavar="[ID=]IMAGE",
        help=(
            "photo of the same physical STEP/STP workpiece used for material "
            "assignment; repeat 2..4 times"
        ),
    )
    parser.add_argument(
        "--visual-material-output-dir",
        help="isolated Qwen/MVInverse evidence and Look USD output directory",
    )
    parser.add_argument(
        "--visual-material-config",
        help=(
            "runtime config for Qwen, MVInverse, MDL and evidence rendering; "
            "defaults to tools/qwen_material_pipeline/configs/pipeline/"
            "manual_part_id_materials.json"
        ),
    )
    parser.add_argument(
        "--visual-foreground-annotations",
        help=(
            "human-confirmed interactive SAM3 foreground JSON; generated by "
            "'python -m qwen_material_pipeline sam3-foreground-ui'"
        ),
    )
    parser.add_argument(
        "--visual-inference-mode",
        choices=("live", "auto", "bundled"),
        default="live",
        help=(
            "visual-material inference source: live runs the current models "
            "(default); auto reuses an exactly matching recorded project when "
            "available; bundled requires that exact recorded-project match"
        ),
    )
    parser.add_argument(
        "--acknowledge-mvinverse-noncommercial",
        action="store_true",
        help="confirm this MVInverse run is permitted by its non-commercial license",
    )
    parser.add_argument(
        "--allow-policy-material-fallback",
        action="store_true",
        help=(
            "for --manual-stp --auto-visual-materials only, allow the material "
            "policy to fill otherwise unresolved CAD parts"
        ),
    )
    return parser


def ensure_generation_source(args: argparse.Namespace) -> None:
    if args.manual_stp:
        validate_manual_cad_args(args)
        return
    if args.existing_glb or args.sam3d_glb:
        return
    if args.sam3d_input:
        input_path = Path(args.sam3d_input).expanduser()
        if not input_path.exists():
            raise FileNotFoundError(f"SAM3D input does not exist: {input_path}")
        if not args.sam3d_prompt:
            raise ValueError("--sam3d-prompt is required when using --sam3d-input")
        if not 0.0 <= args.sam3d_confidence_threshold <= 1.0:
            raise ValueError("--sam3d-confidence-threshold must be between 0 and 1")
        return
    if args.image_url:
        return
    input_dir = Path(args.input_dir).expanduser()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input image directory does not exist: {input_dir}")


def validate_manual_cad_args(args: argparse.Namespace) -> None:
    """Reject non-CAD and GLB-only transform options before starting a job."""

    if not args.manual_stp:
        return
    validate_cad_input_path(
        args.manual_stp, require_single=bool(args.auto_visual_materials)
    )
    supplied = [
        flag
        for flag, value in (
            ("--len-x", args.len_x),
            ("--len-y", args.len_y),
            ("--len-z", args.len_z),
            ("--orientation", args.orientation),
        )
        if value is not None
    ]
    if supplied:
        raise ValueError(
            "Manual STEP/STP preserves CAD dimensions and orientation; remove "
            + ", ".join(supplied)
        )


def resolved_glb_transform_args(
    args: argparse.Namespace,
) -> tuple[float, float, float, str]:
    """Apply legacy GLB defaults without exposing them to manual CAD."""

    return (
        DEFAULT_LEN_X if args.len_x is None else args.len_x,
        DEFAULT_LEN_Y if args.len_y is None else args.len_y,
        DEFAULT_LEN_Z if args.len_z is None else args.len_z,
        DEFAULT_ORIENTATION if args.orientation is None else args.orientation,
    )


def validate_visual_material_args(args: argparse.Namespace) -> None:
    """Reject incomplete reference-image material requests before CAD conversion."""

    if args.visual_inference_mode != "live" and not args.manual_stp:
        raise ValueError(
            "--visual-inference-mode auto/bundled is only supported with "
            "--manual-stp; generated and GLB assets always use live inference"
        )
    if args.allow_policy_material_fallback and not args.manual_stp:
        raise ValueError(
            "--allow-policy-material-fallback is only supported with --manual-stp"
        )
    if args.visual_foreground_annotations is not None and not args.manual_stp:
        raise ValueError(
            "--visual-foreground-annotations is only supported with --manual-stp"
        )
    if (
        args.visual_foreground_annotations is not None
        and args.visual_inference_mode != "live"
    ):
        raise ValueError(
            "--visual-foreground-annotations requires --visual-inference-mode live"
        )
    if args.allow_policy_material_fallback and not args.auto_visual_materials:
        raise ValueError(
            "--allow-policy-material-fallback requires --auto-visual-materials"
        )
    supplied = bool(
        args.visual_reference
        or args.visual_material_output_dir
        or args.visual_material_config
        or args.visual_foreground_annotations is not None
        or args.visual_inference_mode != "live"
        or args.acknowledge_mvinverse_noncommercial
        or args.allow_policy_material_fallback
    )
    if supplied and not args.auto_visual_materials:
        raise ValueError(
            "Reference-image material options require --auto-visual-materials"
        )
    if not args.auto_visual_materials:
        return
    if not args.acknowledge_mvinverse_noncommercial:
        raise ValueError(
            "Reference-image material assignment requires "
            "--acknowledge-mvinverse-noncommercial"
        )
    parse_visual_references(args.visual_reference)
    load_visual_material_config(args.visual_material_config)
    if args.visual_foreground_annotations is not None:
        annotation_path = Path(args.visual_foreground_annotations).expanduser()
        try:
            annotation_path = annotation_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise FileNotFoundError(
                "SAM3 foreground annotation file does not exist: "
                f"{args.visual_foreground_annotations}"
            ) from exc
        if not annotation_path.is_file():
            raise ValueError(
                f"SAM3 foreground annotations must be a file: {annotation_path}"
            )


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

    len_x, len_y, len_z, orientation = resolved_glb_transform_args(args)
    postprocess_result = run_process_model_job(
        input_path=process_input,
        len_x=len_x,
        len_y=len_y,
        len_z=len_z,
        orientation=orientation,
        intermediate_output_dir=args.intermediate_output_dir,
        final_output_dir=args.final_output_dir,
        set_mass=args.set_mass,
        material=args.material,
        approx=args.approx,
        auto_visual_materials=args.auto_visual_materials,
        visual_material_references=args.visual_reference,
        visual_material_output_dir=args.visual_material_output_dir,
        visual_material_config=args.visual_material_config,
        acknowledge_mvinverse_noncommercial=(args.acknowledge_mvinverse_noncommercial),
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
        postprocess_result = run_manual_cad_workflow(
            input_path=args.manual_stp,
            intermediate_output_dir=args.intermediate_output_dir,
            final_output_dir=args.final_output_dir,
            cad_usd_output_dir=args.cad_usd_output_dir,
            cad_converter_options=args.cad_converter_option,
            material=args.material,
            set_mass=args.set_mass,
            approx=args.approx,
            sdf_resolution=args.manual_sdf_resolution,
            auto_visual_materials=args.auto_visual_materials,
            visual_material_references=args.visual_reference,
            visual_material_output_dir=args.visual_material_output_dir,
            visual_material_config=args.visual_material_config,
            visual_foreground_annotations=args.visual_foreground_annotations,
            visual_inference_mode=args.visual_inference_mode,
            acknowledge_mvinverse_noncommercial=(
                args.acknowledge_mvinverse_noncommercial
            ),
            allow_policy_material_fallback=args.allow_policy_material_fallback,
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
        len_x, len_y, len_z, orientation = resolved_glb_transform_args(args)
        result = run_sam3d_image_and_process_model_job(
            input_path=args.sam3d_input,
            output_dir=args.output_dir,
            intermediate_output_dir=args.intermediate_output_dir,
            final_output_dir=args.final_output_dir,
            len_x=len_x,
            len_y=len_y,
            len_z=len_z,
            orientation=orientation,
            sam3d_mode=args.sam3d_mode,
            sam3d_prompt=args.sam3d_prompt,
            sam3d_seed=args.sam3d_seed,
            sam3d_steps=args.sam3d_steps,
            sam3d_confidence_threshold=args.sam3d_confidence_threshold,
            set_mass=args.set_mass,
            material=args.material,
            approx=args.approx,
            refine_mesh=not args.skip_refine,
            refine_output_dir=args.refine_output_dir,
            refine_config_path=args.refine_config_path,
            refine_temp_upload=args.refine_temp_upload,
            refine_fail_on_qc_error=args.refine_fail_on_qc_error,
            auto_visual_materials=args.auto_visual_materials,
            visual_material_references=args.visual_reference,
            visual_material_output_dir=args.visual_material_output_dir,
            visual_material_config=args.visual_material_config,
            acknowledge_mvinverse_noncommercial=(
                args.acknowledge_mvinverse_noncommercial
            ),
            log_cb=log,
        )
        result["mode"] = "sam3d_input"
        result["runtime"] = runtime.runtime_summary()
        return result

    if args.existing_glb:
        return run_glb_refine_then_process(args, args.existing_glb, "existing_glb")

    len_x, len_y, len_z, orientation = resolved_glb_transform_args(args)
    result = run_generate_and_process_model_job(
        output_dir=args.output_dir,
        intermediate_output_dir=args.intermediate_output_dir,
        final_output_dir=args.final_output_dir,
        len_x=len_x,
        len_y=len_y,
        len_z=len_z,
        orientation=orientation,
        set_mass=args.set_mass,
        material=args.material,
        approx=args.approx,
        input_dir=None if args.image_url else args.input_dir,
        image_url=args.image_url,
        face_count=args.face_count,
        download_preview=args.download_preview,
        refine_mesh=not args.skip_refine,
        refine_output_dir=args.refine_output_dir,
        refine_config_path=args.refine_config_path,
        refine_temp_upload=args.refine_temp_upload,
        refine_fail_on_qc_error=args.refine_fail_on_qc_error,
        auto_visual_materials=args.auto_visual_materials,
        visual_material_references=args.visual_reference,
        visual_material_output_dir=args.visual_material_output_dir,
        visual_material_config=args.visual_material_config,
        acknowledge_mvinverse_noncommercial=(args.acknowledge_mvinverse_noncommercial),
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
        validate_visual_material_args(args)
        result = run(args)
        write_result(result, args.result_json)
        log("Pipeline finished.")
        return 0
    except Exception as exc:
        log(f"Pipeline failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
