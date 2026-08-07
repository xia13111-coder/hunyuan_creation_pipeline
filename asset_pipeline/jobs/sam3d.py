"""SAM 3D Objects image preparation and reconstruction job."""

from __future__ import annotations

from pathlib import Path

from ..command import LogCallback, log_message, run_command, sam3d_tool_path
from ..paths import (
    list_direct_image_files,
    list_files_by_suffix,
    safe_name,
    unique_path,
)
from ..runtime import sam3d_python


def _copy_image_as_png(source: Path, destination: Path) -> None:
    from PIL import Image

    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.open(source).convert("RGB").save(destination)


def prepare_sam3d_input(
    input_path: str, output_dir: str, mode: str
) -> tuple[str, str, list[str]]:
    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"SAM3D input does not exist: {source}")

    requested_mode = mode.lower()
    if requested_mode not in {"auto", "single", "multi"}:
        raise ValueError(f"Unsupported SAM3D mode: {mode}")

    image_dir = (
        source / "images"
        if source.is_dir() and (source / "images").exists()
        else source
    )
    image_files = list_direct_image_files(image_dir)
    if not image_files:
        raise FileNotFoundError(f"No input images found for SAM3D: {source}")

    resolved_mode = requested_mode
    if resolved_mode == "auto":
        resolved_mode = "multi" if len(image_files) > 1 else "single"
    if resolved_mode == "single":
        image_files = image_files[:1]

    source_name = source.stem if source.is_file() else source.name
    work_dir = unique_path(
        Path(output_dir).expanduser().resolve() / "sam3d" / safe_name(source_name)
    )
    if resolved_mode == "single":
        _copy_image_as_png(image_files[0], work_dir / "image.png")
    else:
        for index, image_file in enumerate(image_files):
            _copy_image_as_png(image_file, work_dir / "images" / f"{index:05d}.png")
    return str(work_dir), resolved_mode, [str(path) for path in image_files]


def select_sam3d_glb(work_dir: Path) -> Path:
    for path in (
        work_dir / "scene_combined.glb",
        work_dir / "result.glb",
        work_dir / "result_obj0.glb",
    ):
        if path.exists() and path.stat().st_size > 0:
            return path

    candidates = [Path(path) for path in list_files_by_suffix(str(work_dir), {".glb"})]
    candidates = [
        path for path in candidates if path.exists() and path.stat().st_size > 0
    ]
    if not candidates:
        raise FileNotFoundError(f"SAM3D finished but no GLB was found in: {work_dir}")
    return sorted(candidates)[0]


def run_sam3d_image_job(
    *,
    input_path: str,
    output_dir: str,
    mode: str = "auto",
    prompt: str | None = None,
    seed: int = 42,
    steps: int = 50,
    confidence_threshold: float = 0.5,
    log_cb: LogCallback = None,
) -> dict:
    if not prompt or not prompt.strip():
        raise ValueError("--sam3d-prompt is required for raw SAM3D image input")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("--sam3d-confidence-threshold must be between 0 and 1")

    work_dir, resolved_mode, image_files = prepare_sam3d_input(
        input_path, output_dir, mode
    )
    python_path = sam3d_python()
    args = [
        str(python_path),
        str(sam3d_tool_path("run_reconstruct.py")),
        "--input_dir",
        work_dir,
        "--mode",
        resolved_mode,
        "--prompt",
        prompt,
        "--confidence-threshold",
        str(float(confidence_threshold)),
        "--seed",
        str(int(seed)),
        "--steps",
        str(int(steps)),
    ]
    environment = {
        "PYTHONNOUSERSITE": "1",
        "CONDA_PREFIX": str(python_path.resolve().parents[1]),
    }

    log_message(
        log_cb,
        f"SAM3D source images: {input_path} | image_count={len(image_files)} | mode={resolved_mode}",
    )
    log_message(log_cb, f"SAM3D working dir: {work_dir}")
    run_command(args, log_cb=log_cb, env_overrides=environment)

    selected_glb = select_sam3d_glb(Path(work_dir))
    model_files = list_files_by_suffix(work_dir, {".glb", ".obj", ".ply"})
    log_message(log_cb, f"SAM3D GLB ready for refine mesh: {selected_glb}")
    return {
        "input_path": input_path,
        "output_dir": str(Path(output_dir).expanduser().resolve()),
        "work_dir": work_dir,
        "mode": resolved_mode,
        "prompt": prompt,
        "confidence_threshold": confidence_threshold,
        "seed": seed,
        "steps": steps,
        "source_images": image_files,
        "model_files": model_files,
        "selected_glb": str(selected_glb),
        "postprocess_input_path": str(selected_glb),
    }
