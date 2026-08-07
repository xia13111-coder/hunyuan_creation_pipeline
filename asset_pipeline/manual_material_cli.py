"""Small production CLI for the hand-modelled Part-ID material workflow.

This is intentionally a facade over the owning workflow modules.  It keeps
the user contract small while the CAD, SAM3, alignment, MVInverse, retrieval,
Qwen, USD and validation implementations remain independently testable.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from . import runtime
from .jobs.cad import validate_cad_input_path
from .manual_cad import DEFAULT_MANUAL_SDF_RESOLUTION, run_manual_cad_workflow
from .project_layout import SOURCE_LAYOUT
from .visual_materials import load_visual_material_config


PROJECT_ROOT = runtime.project_root()
DEFAULT_CONFIG = SOURCE_LAYOUT.manual_part_id_material_config
ANNOTATION_SCHEMA = "sam3-human-foreground-annotations/v2"
VISUAL_INFERENCE_MODES = ("live", "auto", "bundled")


def log(message: str) -> None:
    print(message, flush=True)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read {label} {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return document


def references_from_annotations(annotation_path: Path) -> list[str]:
    """Return the hash-bound reference list embedded by the SAM3 point UI."""

    document = _read_json_object(annotation_path, "SAM3 annotations")
    if document.get("schema_version") != ANNOTATION_SCHEMA:
        raise ValueError(
            "SAM3 annotations use an unsupported schema: "
            f"{document.get('schema_version')!r}; expected {ANNOTATION_SCHEMA!r}"
        )
    source_views = document.get("source_views")
    if not isinstance(source_views, list) or not 2 <= len(source_views) <= 4:
        raise ValueError("SAM3 annotations must contain 2..4 source_views")

    references: list[str] = []
    seen: set[str] = set()
    for index, view in enumerate(source_views):
        if not isinstance(view, dict):
            raise ValueError(f"SAM3 source_views[{index}] must be an object")
        view_id = view.get("id")
        raw_image = view.get("image")
        if not isinstance(view_id, str) or not view_id.strip():
            raise ValueError(f"SAM3 source_views[{index}].id is invalid")
        view_id = view_id.strip()
        if view_id in seen:
            raise ValueError(f"SAM3 source view id is duplicated: {view_id}")
        if not isinstance(raw_image, str) or not raw_image.strip():
            raise ValueError(f"SAM3 source_views[{index}].image is invalid")
        image = Path(raw_image).expanduser().resolve(strict=True)
        if not image.is_file():
            raise ValueError(f"SAM3 source image must be a file: {image}")
        seen.add(view_id)
        references.append(f"{view_id}={image}")
    return references


def default_output_root(stp: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "outputs" / "manual" / f"{stp.stem}_{timestamp}"


def _require_fresh_or_resumable_output(output: Path, *, resume: bool) -> None:
    if not output.exists():
        return
    if not output.is_dir():
        raise ValueError(f"--output must be a directory: {output}")
    if any(output.iterdir()) and not resume:
        raise FileExistsError(
            f"Output directory is not empty: {output}. Use a new directory for "
            "a from-zero run, or pass --resume for hash-verified continuation."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run STP -> USD -> physics geometry preparation -> Part-ID reference "
            "analysis -> NVIDIA Base MDL assignment -> final validation."
        )
    )
    parser.add_argument("--stp", required=True, help="one hand-modelled .stp/.step")
    parser.add_argument(
        "--sam3-annotations",
        required=True,
        help=(
            "human-confirmed SAM3 point annotation JSON; its source_views are "
            "used as the reference images"
        ),
    )
    parser.add_argument(
        "--output",
        help="run root; defaults to outputs/manual/<stp>_<timestamp>",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="local model/material configuration",
    )
    parser.add_argument(
        "--visual-inference-mode",
        choices=VISUAL_INFERENCE_MODES,
        default="live",
        help=(
            "live runs the current material workflow (default); auto reuses an "
            "exactly matching recorded project when available; bundled requires "
            "that exact project match"
        ),
    )
    parser.add_argument(
        "--material",
        default="plastic",
        help="physics material preset from materials.json",
    )
    parser.add_argument(
        "--approx",
        default="sdf",
        help="collision approximation (default: sdf)",
    )
    parser.add_argument(
        "--sdf-resolution",
        type=int,
        default=DEFAULT_MANUAL_SDF_RESOLUTION,
    )
    parser.add_argument("--set-mass", type=float, help="optional total mass in kg")
    parser.add_argument(
        "--cad-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="optional CAD converter option; may be repeated",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue only hash-verified artifacts in an existing run root",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    runtime_config = runtime.configure_runtime()
    stp = Path(args.stp).expanduser().resolve(strict=True)
    validate_cad_input_path(str(stp), require_single=True)
    annotations = Path(args.sam3_annotations).expanduser().resolve(strict=True)
    if not annotations.is_file():
        raise ValueError(f"--sam3-annotations must be a file: {annotations}")
    references = references_from_annotations(annotations)

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_output_root(stp).resolve()
    )
    _require_fresh_or_resumable_output(output, resume=args.resume)
    config = Path(args.config).expanduser().resolve(strict=True)
    load_visual_material_config(config)

    # A human SAM3 file is the authoritative reference manifest for every mode.
    # Its foreground masks are live-inference evidence only: an exact bundled
    # replay instead uses the immutable, hash-bound evidence owned by the
    # sealed project.  Passing a live annotation file into bundled mode would
    # incorrectly imply that it can change a historical material decision.
    foreground_annotations = (
        str(annotations) if args.visual_inference_mode == "live" else None
    )

    output.mkdir(parents=True, exist_ok=True)
    result = run_manual_cad_workflow(
        input_path=str(stp),
        cad_usd_output_dir=str(output / "cad_usd"),
        intermediate_output_dir=str(output / "intermediate"),
        final_output_dir=str(output / "final"),
        cad_converter_options=args.cad_option,
        material=args.material,
        set_mass=args.set_mass,
        approx=args.approx,
        sdf_resolution=args.sdf_resolution,
        auto_visual_materials=True,
        visual_material_references=references,
        visual_material_output_dir=str(output / "visual_material"),
        visual_material_config=str(config),
        visual_foreground_annotations=foreground_annotations,
        visual_inference_mode=args.visual_inference_mode,
        acknowledge_mvinverse_noncommercial=True,
        allow_policy_material_fallback=args.visual_inference_mode == "live",
        resume=args.resume,
        log_cb=log,
    )
    document = {
        "runtime": runtime_config,
        "mode": "manual_part_id_materials",
        "run_root": str(output),
        "visual_inference_mode": args.visual_inference_mode,
        "sam3_annotations": str(annotations),
        "sam3_annotations_role": (
            "foreground_evidence"
            if args.visual_inference_mode == "live"
            else "reference_manifest_only"
        ),
        "references": references,
        "result": result,
    }
    result_json = output / "pipeline_result.json"
    result_json.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log(f"Result JSON: {result_json}")
    log("Manual Part-ID material pipeline finished.")
    return document


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
        return 0
    except Exception as exc:
        log(f"Pipeline failed: {exc}")
        return 1


__all__ = [
    "build_parser",
    "default_output_root",
    "main",
    "references_from_annotations",
    "run",
    "VISUAL_INFERENCE_MODES",
]
