#!/usr/bin/env python3

"""Headless SAM 3D Objects reconstruction wrapper.

This script lives in the main pipeline repo and calls the local SAM 3D Objects
checkout through SAM3D_SINGLE_VIEW_ROOT. Single-view runs use the pointmap
pipeline from that checkout. Multi-view runs delegate to the separate
sam-3d-objects-multiview checkout through SAM3D_MULTI_VIEW_ROOT.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_ROOT = Path(__file__).resolve().parent / "third_party"
SINGLE_VIEW_ROOT = (
    Path(os.getenv("SAM3D_SINGLE_VIEW_ROOT", str(THIRD_PARTY_ROOT / "sam-3d-objects")))
    .expanduser()
    .resolve()
)
MULTI_VIEW_ROOT = (
    Path(
        os.getenv(
            "SAM3D_MULTI_VIEW_ROOT", str(THIRD_PARTY_ROOT / "sam-3d-objects-multiview")
        )
    )
    .expanduser()
    .resolve()
)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("GITHUB_PROXY", "https://ghproxy.com/")

# Keep the wrapper self-contained: SAM3D keeps several import roots inside the
# checkout instead of one installable package.
for path in (
    SINGLE_VIEW_ROOT,
    SINGLE_VIEW_ROOT / "MoGe",
    SINGLE_VIEW_ROOT / "pytorch3d",
    SINGLE_VIEW_ROOT / "submodules" / "sam3",
    SINGLE_VIEW_ROOT / "submodules",
    SINGLE_VIEW_ROOT / "notebook",
):
    sys.path.insert(0, str(path))


def load_single_view_pipeline():
    print(">>> Loading single-view SAM3D pipeline...")
    from inference import Inference, make_scene

    config_path = (
        SINGLE_VIEW_ROOT
        / "checkpoints"
        / "sam-3d-objects"
        / "checkpoints"
        / "pipeline.yaml"
    )
    if not config_path.exists():
        raise FileNotFoundError(f"SAM3D pipeline config not found: {config_path}")
    return Inference(str(config_path), compile=False), make_scene


def load_rgba_mask_for_inference(path: Path):
    image = Image.open(path)
    return image.split()[-1] if image.mode == "RGBA" else image.convert("L")


def load_sam3_processor():
    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"{exc}. SAM3D dependencies are not available in this Python: {sys.executable}. "
            "Activate the unified hunyuan_sam3d environment before running the pipeline."
        ) from exc

    checkpoint_env = os.getenv("SAM3_CHECKPOINT")
    checkpoint = Path(checkpoint_env).expanduser().resolve() if checkpoint_env else None
    if checkpoint is None or not checkpoint.exists():
        checkpoint = (
            Path.home()
            / ".cache"
            / "modelscope"
            / "hub"
            / "models"
            / "facebook"
            / "sam3"
            / "sam3.pt"
        )
    if not checkpoint.exists():
        checkpoint = SINGLE_VIEW_ROOT / "checkpoints" / "sam3.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"SAM3 checkpoint not found: {checkpoint}. "
            "Set SAM3_CHECKPOINT or place sam3.pt under "
            f"{SINGLE_VIEW_ROOT / 'checkpoints' / 'sam3.pt'}."
        )

    model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False)
    return Sam3Processor(model), model


def save_mask_rgba(
    mask_array: np.ndarray, image_array: np.ndarray, output_path: Path
) -> None:
    if mask_array.max() <= 1.0:
        mask_array = (mask_array * 255).astype(np.uint8)
    rgba = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = image_array[:, :, :3]
    rgba[:, :, 3] = mask_array
    Image.fromarray(rgba, "RGBA").save(output_path)


def auto_segment(input_dir: Path, mode: str, prompt: str) -> list[str]:
    print(f">>> Segmenting images with prompt: {prompt!r}")
    processor, model = load_sam3_processor()
    valid_names: list[str] = []

    def process_one(image_path: Path, mask_path: Path, segmenter) -> bool:
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        state = segmenter.set_image(image)
        output = segmenter.set_text_prompt(state=state, prompt=prompt)
        masks = output["masks"].cpu().numpy()
        if len(masks) == 0:
            print(f"  [warn] No mask found for {image_path}")
            return False

        scores = output.get("scores")
        best_idx = (
            int(np.argmax(scores.cpu().numpy()))
            if scores is not None and len(scores) > 0
            else 0
        )
        mask = masks[best_idx]
        if len(mask.shape) == 3:
            mask = mask[0]
        if mask.shape != image_array.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (image_array.shape[1], image_array.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask_binary = (mask > 0.5).astype(np.uint8)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        save_mask_rgba(mask_binary * 255, image_array, mask_path)
        print(f"  [+] Mask: {mask_path}")
        return True

    if mode == "single":
        image_path = input_dir / "image.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Single-view input image not found: {image_path}")
        if process_one(image_path, input_dir / "0.png", processor):
            valid_names.append("0")
    else:
        images_dir = input_dir / "images"
        masks_dir = input_dir / "masks"
        for image_path in sorted(images_dir.glob("*")):
            if image_path.suffix.lower() not in {
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
                ".bmp",
            }:
                continue
            mask_path = masks_dir / f"{image_path.stem}.png"
            if process_one(image_path, mask_path, processor):
                valid_names.append(image_path.stem)

    del processor, model
    torch.cuda.empty_cache()
    return valid_names


def run_single_view(input_dir: Path, seed: int, steps: int) -> None:
    image_path = input_dir / "image.png"
    mask_paths = sorted(input_dir.glob("[0-9]*.png"))
    if not image_path.exists():
        raise FileNotFoundError(f"Single-view input image not found: {image_path}")
    if not mask_paths:
        raise FileNotFoundError(f"No single-view mask files found in: {input_dir}")

    pipeline, make_scene = load_single_view_pipeline()
    outputs = []
    for index, mask_path in enumerate(mask_paths):
        print(f">>> Reconstructing object {index}: {mask_path}")
        result = pipeline._pipeline.run(
            image=Image.open(image_path).convert("RGB"),
            mask=load_rgba_mask_for_inference(mask_path),
            seed=int(seed),
            stage1_inference_steps=int(steps),
            stage2_inference_steps=25,
            decode_formats=["gaussian", "mesh"],
            with_mesh_postprocess=False,
            with_texture_baking=False,
            use_vertex_color=True,
        )
        outputs.append(result)
        if result.get("glb"):
            result["glb"].export(input_dir / f"result_obj{index}.glb")
            result["glb"].export(input_dir / f"result_obj{index}.obj")
        if result.get("gs"):
            result["gs"].save_ply(input_dir / f"result_obj{index}.ply")

    if len(outputs) <= 1:
        return

    try:
        scene_gs = make_scene(*outputs)
        scene_gs.save_ply(input_dir / "scene_combined.ply")
    except Exception as exc:
        print(f"[warn] Combined PLY export failed: {exc}")

    meshes = [item["glb"] for item in outputs if item.get("glb") is not None]
    if meshes:
        combined = trimesh.util.concatenate(meshes)
        combined.export(input_dir / "scene_combined.glb")
        combined.export(input_dir / "scene_combined.obj")


def find_multiview_script() -> tuple[Path, Path]:
    candidates = [
        (MULTI_VIEW_ROOT / "MV-SAM3D" / "run_inference.py", MULTI_VIEW_ROOT),
        (MULTI_VIEW_ROOT / "run_inference.py", MULTI_VIEW_ROOT),
        (MULTI_VIEW_ROOT / "MV-SAM3D" / "run_inference_weighted.py", MULTI_VIEW_ROOT),
    ]
    for script, cwd in candidates:
        if script.exists():
            return script, cwd
    raise FileNotFoundError(
        "Multi-view SAM3D script not found. Set SAM3D_MULTI_VIEW_ROOT to the "
        "sam-3d-objects-multiview checkout."
    )


def run_multi_view(
    input_dir: Path, seed: int, steps: int, valid_names: list[str]
) -> None:
    script, cwd = find_multiview_script()
    conda_prefix = Path(sys.executable).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["CONDA_PREFIX"] = str(conda_prefix)
    import_roots = [
        MULTI_VIEW_ROOT,
        SINGLE_VIEW_ROOT,
        SINGLE_VIEW_ROOT / "MoGe",
        SINGLE_VIEW_ROOT / "pytorch3d",
        SINGLE_VIEW_ROOT / "submodules" / "sam3",
        SINGLE_VIEW_ROOT / "submodules",
    ]
    if env.get("PYTHONPATH"):
        import_roots.append(Path(env["PYTHONPATH"]))
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in import_roots)
    cmd = [
        sys.executable,
        str(script),
        "--input_path",
        str(input_dir.resolve()),
        "--mask_prompt",
        "masks",
        "--seed",
        str(int(seed)),
        "--stage1_steps",
        str(int(steps)),
        "--decode_formats",
        "gaussian,mesh",
    ]
    if valid_names:
        cmd.extend(["--image_names", ",".join(valid_names)])
    print(f">>> Running multi-view command: {' '.join(cmd)}")
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            subprocess.run(cmd, cwd=str(cwd), check=True, env=env)
            break
        except subprocess.CalledProcessError:
            if attempt >= max_attempts:
                raise
            print(
                f"[warn] Multi-view command failed on attempt {attempt}/{max_attempts}; retrying..."
            )
            time.sleep(2)

    model_files = []
    vis_dir = cwd / "visualization"
    for root, _, files in os.walk(vis_dir):
        for name in files:
            if name.endswith((".glb", ".ply")):
                model_files.append(Path(root) / name)
    if not model_files:
        raise FileNotFoundError(f"No multi-view model output found under: {vis_dir}")
    latest_dir = max(model_files, key=lambda path: path.stat().st_mtime).parent

    for ext in ("glb", "ply"):
        matches = sorted(latest_dir.glob(f"*.{ext}"))
        if matches:
            dst = input_dir / f"result.{ext}"
            shutil.copy2(matches[0], dst)
            print(f"  [+] {ext.upper()}: {dst}")
            if ext == "glb":
                try:
                    mesh = trimesh.load(dst)
                    mesh.export(input_dir / "result.obj")
                except Exception as exc:
                    print(f"[warn] OBJ export failed: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless SAM 3D Objects reconstruction"
    )
    parser.add_argument(
        "--input_dir", required=True, help="pipeline-prepared SAM3D input directory"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        required=True,
        help="reconstruction backend",
    )
    parser.add_argument(
        "--prompt", required=True, help="SAM3 target-object segmentation prompt"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="geometry reconstruction seed"
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="stage-1 geometry sampling steps"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    try:
        if args.mode == "multi":
            find_multiview_script()
        valid_names = auto_segment(input_dir, args.mode, args.prompt)
        if not valid_names:
            raise ValueError("SAM3 segmentation produced no valid masks")
        if args.mode == "single":
            run_single_view(input_dir, args.seed, args.steps)
        else:
            run_multi_view(input_dir, args.seed, args.steps, valid_names)
        print("SAM3D reconstruction finished.")
        return 0
    except Exception as exc:
        print(f"SAM3D reconstruction failed: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
