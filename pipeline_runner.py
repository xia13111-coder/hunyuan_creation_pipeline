#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import json
import re
import shutil
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence


LogCallback = Optional[Callable[[str], None]]


def root_dir() -> Path:
    return Path(os.getenv("ROOT_DIR", Path(__file__).resolve().parent)).expanduser().resolve()


def blender_bin() -> Path:
    env_value = os.getenv("BLENDER_BIN")
    if env_value:
        return Path(env_value).expanduser().resolve()
    for candidate in (
        shutil.which("blender"),
        "/opt/blender/blender",
        "/usr/local/bin/blender",
    ):
        if candidate and Path(candidate).expanduser().exists():
            return Path(candidate).expanduser().resolve()
    return Path("/opt/blender/blender").resolve()


def isaac_python() -> Path:
    env_value = os.getenv("ISAAC_PYTHON")
    if env_value:
        return Path(env_value).expanduser().resolve()

    root_value = os.getenv("ISAACSIM_ROOT")
    if root_value:
        return (Path(root_value).expanduser().resolve() / "python.sh").resolve()

    candidates = [
        Path("/home/user/isaacsim500/python.sh"),
        Path("/isaac-sim/python.sh"),
        Path("/opt/isaac-sim/python.sh"),
    ]
    candidates.extend(sorted(Path("/home/user/.local/share/ov/pkg").glob("isaac_sim*/python.sh")))
    for candidate in candidates:
        if candidate.exists():
            return candidate.expanduser().resolve()
    return Path("/isaac-sim/python.sh").resolve()


def runtime_summary() -> dict:
    summary = {
        "root_dir": str(root_dir()),
        "python_bin": sys.executable,
        "blender_bin": str(blender_bin()),
        "isaac_python": str(isaac_python()),
        "refine_mesh_config": str(default_refine_config_path()),
    }
    summary["blender_exists"] = blender_bin().exists()
    summary["isaac_python_exists"] = isaac_python().exists()
    summary["refine_mesh_config_exists"] = default_refine_config_path().exists()
    return summary


def materials_file() -> Path:
    return root_dir() / "materials.json"


def default_refine_config_path() -> Path:
    return root_dir() / "configs" / "hunyuan_reduce_local_postprocess.yaml"


def default_refine_temp_upload() -> str | None:
    value = os.getenv("REFINE_MESH_TEMP_UPLOAD", "uguu").strip()
    return value or None


def default_refine_output_dir(input_path: str) -> str:
    base = Path(input_path).expanduser()
    if base.suffix.lower() == ".glb":
        return str(base.with_suffix("").with_name(base.stem + "_refined_mesh"))
    return str(base.with_name(base.name + "_refined_mesh"))


def available_materials() -> list[str]:
    try:
        with open(materials_file(), "r", encoding="utf-8") as f:
            data = json.load(f)
        mats = data.get("materials", {})
        return sorted(str(name) for name in mats.keys())
    except Exception:
        return []


def available_approx_types() -> list[str]:
    return [
        "sdf",
        "convexHull",
        "convexDecomposition",
        "triangleMesh",
        "meshSimplification",
        "boundingCube",
        "boundingSphere",
        "sphereApproximation",
    ]


def _log(log_cb: LogCallback, message: str) -> None:
    if log_cb:
        log_cb(message)


def _script_path(script_name: str) -> Path:
    script_path = root_dir() / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    return script_path


def _run_command(cmd: Sequence[str], *, log_cb: LogCallback = None) -> None:
    env = os.environ.copy()
    pretty_cmd = " ".join(shlex.quote(part) for part in cmd)
    _log(log_cb, f"$ {pretty_cmd}")

    executable = Path(cmd[0])
    if not executable.exists():
        raise FileNotFoundError(
            f"Executable not found: {executable}. "
            "Set BLENDER_BIN, ISAAC_PYTHON, or ISAACSIM_ROOT for this machine."
        )
    if executable.is_dir():
        raise RuntimeError(f"Executable path is a directory: {executable}")

    try:
        process = subprocess.Popen(
            list(cmd),
            cwd=str(root_dir()),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to start command: {pretty_cmd} | {exc}") from exc

    assert process.stdout is not None
    for line in process.stdout:
        _log(log_cb, line.rstrip())

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {pretty_cmd}")


def _append_flag(args: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        args.append(flag)


def _append_option(args: list[str], name: str, value: object | None) -> None:
    if value is not None:
        args.extend([name, str(value)])


def _list_files_by_suffix(root: str, suffixes: set[str], *, name_contains: str | None = None) -> list[str]:
    base = Path(root)
    if not base.exists():
        return []
    lowered_suffixes = {suffix.lower() for suffix in suffixes}
    lowered_contains = name_contains.lower() if name_contains else None
    if base.is_file():
        if lowered_contains and lowered_contains not in base.name.lower():
            return []
        return [str(base)] if base.suffix.lower() in lowered_suffixes else []

    files = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if lowered_contains and lowered_contains not in path.name.lower():
            continue
        if path.suffix.lower() in lowered_suffixes:
            files.append(str(path))
    return sorted(files)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or "asset"


def _asset_run_name(glb_path: str, input_path: str) -> str:
    path = Path(glb_path).expanduser().resolve()
    base = Path(input_path).expanduser().resolve()
    if base.is_file():
        base = base.parent
    try:
        rel = path.relative_to(base).with_suffix("")
    except ValueError:
        rel = Path(path.stem)
    return "__".join(_safe_name(part) for part in rel.parts)


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for index in range(2, 10000):
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    return parent / f"{stem}_{int(time.time())}{suffix}"


def _refined_glb_from_output(refine_dir: Path) -> Path:
    report_path = refine_dir / "qc_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            for item in report.get("exports", []):
                export_path = item.get("path") if isinstance(item, dict) else None
                if export_path:
                    candidate = Path(export_path)
                    if not candidate.is_absolute():
                        candidate = refine_dir / candidate
                    if candidate.suffix.lower() == ".glb" and candidate.exists():
                        return candidate
        except Exception:
            pass
    candidate = refine_dir / "refined_asset.glb"
    if candidate.exists():
        return candidate
    glb_files = _list_files_by_suffix(str(refine_dir), {".glb"})
    for glb_file in glb_files:
        path = Path(glb_file)
        if "intermediate" not in path.parts:
            return path
    raise FileNotFoundError(f"Refine mesh finished but no final GLB was found in: {refine_dir}")


def _converted_usd_root(input_path: str, *, overwrite: bool, suffix: str | None) -> str:
    base = Path(input_path)
    if base.is_file():
        out_dir = base.with_suffix("")
        if not overwrite:
            out_dir = out_dir.with_name(out_dir.name + (suffix or "_zup"))
        return str(out_dir)
    return input_path


def _log_blender_preflight(input_path: str, log_cb: LogCallback = None) -> list[str]:
    blender = blender_bin()
    glb_files = _list_files_by_suffix(input_path, {".glb"})
    _log(log_cb, f"Blender binary: {blender} | exists={blender.exists()}")
    _log(log_cb, f"Postprocess input: {input_path} | glb_count={len(glb_files)}")
    if not blender.exists():
        raise FileNotFoundError(f"Blender binary not found: {blender}. Set BLENDER_BIN for this machine.")
    if blender.is_dir():
        raise RuntimeError(f"BLENDER_BIN points to a directory, not an executable: {blender}")
    for glb in glb_files[:5]:
        _log(log_cb, f"  GLB: {glb}")
    if len(glb_files) > 5:
        _log(log_cb, f"  ... {len(glb_files) - 5} more GLB files")
    if not glb_files:
        raise FileNotFoundError(f"No .glb files found for Blender input: {input_path}")

    completed = subprocess.run(
        [str(blender), "--version"],
        cwd=str(root_dir()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
        check=False,
    )
    first_line = (completed.stdout or "").splitlines()[0] if completed.stdout else ""
    _log(log_cb, f"Blender version check: {first_line or 'no output'}")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Blender version check failed with exit code {completed.returncode}: {completed.stdout}"
        )
    return glb_files


def run_hunyuan_job(
    *,
    output_dir: str,
    input_dir: str | None = None,
    prompt: str | None = None,
    image_url: str | None = None,
    result_formats: Sequence[str] = ("GLB",),
    region: str = "ap-guangzhou",
    endpoint: str = "ai3d.tencentcloudapi.com",
    pbr: bool = True,
    interval: float = 30.0,
    timeout: float = 900.0,
    retry_interval: int = 60,
    max_retry: int = 10,
    http_retries: int = 5,
    version: str = "pro",
    face_count: int | None = None,
    gen_type: str | None = None,
    resubmit: int = 1,
    resubmit_backoff: float = 30.0,
    download_preview: bool = False,
    verbose: int = 1,
    log_cb: LogCallback = None,
) -> dict:
    args = [
        sys.executable,
        str(_script_path("hunyuan_to3d_batch.py")),
        "--output",
        output_dir,
        "--region",
        region,
        "--endpoint",
        endpoint,
        "--interval",
        str(interval),
        "--timeout",
        str(timeout),
        "--retry-interval",
        str(retry_interval),
        "--max-retry",
        str(max_retry),
        "--http-retries",
        str(http_retries),
        "--version",
        version,
        "--resubmit",
        str(resubmit),
        "--resubmit-backoff",
        str(resubmit_backoff),
        "--result-formats",
        *[fmt.upper() for fmt in result_formats],
    ]

    if input_dir:
        args.extend(["--input", input_dir])
    if prompt:
        args.extend(["--prompt", prompt])
    if image_url:
        args.extend(["--image-url", image_url])
    if not pbr:
        args.append("--no-pbr")
    _append_flag(args, "--download-preview", download_preview)
    _append_option(args, "--face-count", face_count)
    _append_option(args, "--gen-type", gen_type)
    if verbose > 0:
        args.append("-" + ("v" * verbose))

    _run_command(args, log_cb=log_cb)
    return {
        "output_dir": output_dir,
        "input_dir": input_dir,
        "prompt": prompt,
        "image_url": image_url,
        "result_formats": [fmt.upper() for fmt in result_formats],
        "download_preview": download_preview,
        "model_files": _list_files_by_suffix(
            output_dir,
            {".glb", ".obj", ".fbx", ".stl", ".usdz", ".mp4"},
        ),
        "preview_files": _list_files_by_suffix(
            output_dir,
            {".png", ".jpg", ".jpeg", ".webp", ".bmp"},
            name_contains="_preview",
        ) if download_preview else [],
    }


def run_refine_mesh_job(
    *,
    input_path: str,
    output_dir: str | None = None,
    config_path: str | None = None,
    temp_upload: str | None = None,
    fail_on_qc_error: bool = False,
    log_cb: LogCallback = None,
) -> dict:
    glb_files = _list_files_by_suffix(input_path, {".glb"})
    if not glb_files:
        raise FileNotFoundError(f"No .glb files found for refine mesh input: {input_path}")

    config = Path(config_path or os.getenv("REFINE_MESH_CONFIG", str(default_refine_config_path()))).expanduser()
    if not config.is_absolute():
        config = root_dir() / config
    if not config.exists():
        raise FileNotFoundError(f"Refine mesh config not found: {config}")

    refine_root = Path(output_dir or default_refine_output_dir(input_path)).expanduser().resolve()
    refine_root.mkdir(parents=True, exist_ok=True)
    final_glbs_dir = refine_root / "postprocess_glbs"
    if final_glbs_dir.exists():
        final_glbs_dir = refine_root / f"postprocess_glbs_{int(time.time())}"
    final_glbs_dir.mkdir(parents=True, exist_ok=False)

    provider = default_refine_temp_upload() if temp_upload is None else temp_upload
    provider = provider.strip() if isinstance(provider, str) else provider
    if provider and provider.lower() in {"none", "false", "off", "0"}:
        provider = None

    _log(log_cb, f"Refine mesh input: {input_path} | glb_count={len(glb_files)}")
    _log(log_cb, f"Refine mesh output: {refine_root}")
    _log(log_cb, f"Refine mesh config: {config}")

    refined_items = []
    for glb_file in glb_files:
        asset_name = _asset_run_name(glb_file, input_path)
        asset_output_dir = refine_root / asset_name
        args = [
            sys.executable,
            "-m",
            "asset_refiner",
            "--input",
            glb_file,
            "--output",
            str(asset_output_dir),
            "--config",
            str(config),
            "--blender",
            str(blender_bin()),
            "--hunyuan-local-postprocess",
        ]
        if provider:
            args.extend(["--hunyuan-temp-upload", provider])
        if fail_on_qc_error:
            args.append("--fail-on-qc-error")

        _log(log_cb, f"Refining GLB: {glb_file}")
        _run_command(args, log_cb=log_cb)

        final_glb = _refined_glb_from_output(asset_output_dir)
        copied_glb = _unique_path(final_glbs_dir / f"{asset_name}_refined.glb")
        shutil.copy2(final_glb, copied_glb)
        refined_items.append(
            {
                "source_glb": glb_file,
                "refine_output_dir": str(asset_output_dir),
                "qc_report": str(asset_output_dir / "qc_report.json"),
                "refined_glb": str(final_glb),
                "postprocess_glb": str(copied_glb),
            }
        )
        _log(log_cb, f"Refined GLB ready for postprocess: {copied_glb}")

    return {
        "input_path": input_path,
        "output_dir": str(refine_root),
        "config_path": str(config),
        "temp_upload": provider,
        "postprocess_input_path": str(final_glbs_dir),
        "refined_files": [item["postprocess_glb"] for item in refined_items],
        "items": refined_items,
    }


def run_align_job(
    *,
    input_path: str,
    axis_map: str = "X=L,Y=M,Z=S",
    overwrite: bool = True,
    suffix: str | None = None,
    log_cb: LogCallback = None,
) -> dict:
    args = [
        str(blender_bin()),
        "-b",
        "-P",
        str(_script_path("align_glb_axis_only.py")),
        "--",
        "--input",
        input_path,
        "--axis-map",
        axis_map,
    ]
    _append_flag(args, "--overwrite", overwrite)
    _append_option(args, "--suffix", suffix if not overwrite else None)
    _run_command(args, log_cb=log_cb)
    return {"input_path": input_path, "axis_map": axis_map, "overwrite": overwrite}


def run_resize_job(
    *,
    input_path: str,
    len_x: float,
    len_y: float,
    len_z: float,
    unit: str = "m",
    overwrite: bool = True,
    suffix: str | None = None,
    log_cb: LogCallback = None,
) -> dict:
    args = [
        str(blender_bin()),
        "-b",
        "-P",
        str(_script_path("resize_glb_xyz_and_center.py")),
        "--",
        "--input",
        input_path,
        "--len-x",
        str(len_x),
        "--len-y",
        str(len_y),
        "--len-z",
        str(len_z),
        "--unit",
        unit,
    ]
    _append_flag(args, "--overwrite", overwrite)
    _append_option(args, "--suffix", suffix if not overwrite else None)
    _run_command(args, log_cb=log_cb)
    return {
        "input_path": input_path,
        "target_size": {"x": len_x, "y": len_y, "z": len_z, "unit": unit},
        "overwrite": overwrite,
    }


def run_convert_job(
    *,
    input_path: str,
    usd_format: str = "usd",
    overwrite: bool = True,
    suffix: str | None = None,
    visible_only: bool = False,
    log_cb: LogCallback = None,
) -> dict:
    args = [
        str(blender_bin()),
        "-b",
        "-P",
        str(_script_path("convert_glb_to_usd_zup.py")),
        "--",
        "--input",
        input_path,
        "--usd-format",
        usd_format,
    ]
    _append_flag(args, "--overwrite", overwrite)
    _append_option(args, "--suffix", suffix if not overwrite else None)
    _append_flag(args, "--visible-only", visible_only)
    _run_command(args, log_cb=log_cb)
    usd_root = _converted_usd_root(input_path, overwrite=overwrite, suffix=suffix)
    return {
        "input_path": input_path,
        "usd_format": usd_format,
        "overwrite": overwrite,
        "usd_root": usd_root,
        "usd_files": _list_files_by_suffix(usd_root, {".usd", ".usda", ".usdc"}),
    }


def run_add_physics_job(
    *,
    folder: str,
    out_dir: str,
    material_file: str | None = None,
    material: str = "plastic",
    set_mass: float | None = None,
    approx: str = "convexDecomposition",
    headless: bool = True,
    log_cb: LogCallback = None,
) -> dict:
    material_file = material_file or str(root_dir() / "materials.json")
    args = [
        str(isaac_python()),
        str(_script_path("add_physics.py")),
        "--folder",
        folder,
        "--material-file",
        material_file,
        "--out-dir",
        out_dir,
    ]
    _append_flag(args, "--headless", headless)
    _append_option(args, "--material", material)
    _append_option(args, "--set-mass", set_mass)
    _append_option(args, "--approx", approx)
    _run_command(args, log_cb=log_cb)
    return {
        "folder": folder,
        "out_dir": out_dir,
        "material_file": material_file,
        "material": material,
        "set_mass": set_mass,
        "approx": approx,
    }


def run_collect_job(
    *,
    folder: str,
    out_dir: str,
    headless: bool = True,
    log_cb: LogCallback = None,
) -> dict:
    args = [
        str(isaac_python()),
        str(_script_path("collect_usd_flat.py")),
        "--folder",
        folder,
        "--out-dir",
        out_dir,
    ]
    _append_flag(args, "--headless", headless)
    _run_command(args, log_cb=log_cb)
    return {"folder": folder, "out_dir": out_dir}


def run_postprocess_job(
    *,
    input_path: str,
    len_x: float,
    len_y: float,
    len_z: float,
    intermediate_output_dir: str,
    final_output_dir: str,
    axis_map: str = "X=L,Y=M,Z=S",
    unit: str = "m",
    usd_format: str = "usd",
    material_file: str | None = None,
    material: str = "plastic",
    set_mass: float | None = None,
    approx: str = "convexDecomposition",
    headless: bool = True,
    log_cb: LogCallback = None,
) -> dict:
    _log_blender_preflight(input_path, log_cb=log_cb)
    steps = []
    steps.append({"step": "align", "result": run_align_job(input_path=input_path, axis_map=axis_map, log_cb=log_cb)})
    steps.append(
        {
            "step": "resize",
            "result": run_resize_job(
                input_path=input_path,
                len_x=len_x,
                len_y=len_y,
                len_z=len_z,
                unit=unit,
                log_cb=log_cb,
            ),
        }
    )
    convert_result = run_convert_job(
        input_path=input_path,
        usd_format=usd_format,
        log_cb=log_cb,
    )
    steps.append({"step": "convert_usd", "result": convert_result})
    usd_input_root = convert_result["usd_root"]
    steps.append(
        {
            "step": "add_physics",
            "result": run_add_physics_job(
                folder=usd_input_root,
                out_dir=intermediate_output_dir,
                material_file=material_file,
                material=material,
                set_mass=set_mass,
                approx=approx,
                headless=headless,
                log_cb=log_cb,
            ),
        }
    )
    steps.append(
        {
            "step": "collect_usd",
            "result": run_collect_job(
                folder=intermediate_output_dir,
                out_dir=final_output_dir,
                headless=headless,
                log_cb=log_cb,
            ),
        }
    )
    return {
        "input_path": input_path,
        "usd_input_root": usd_input_root,
        "intermediate_output_dir": intermediate_output_dir,
        "final_output_dir": final_output_dir,
        "processed_glb_files": _list_files_by_suffix(input_path, {".glb"}),
        "steps": steps,
    }


def run_full_pipeline_job(
    *,
    generation_output_dir: str,
    postprocess_input_path: str | None,
    postprocess_params: dict,
    hunyuan_params: dict,
    log_cb: LogCallback = None,
) -> dict:
    generation_args = dict(hunyuan_params)
    generation_args["output_dir"] = generation_output_dir
    generation_result = run_hunyuan_job(log_cb=log_cb, **generation_args)
    postprocess_args = dict(postprocess_params)
    resolved_input_path = postprocess_input_path or postprocess_args.pop("input_path", None) or generation_output_dir
    postprocess_result = run_postprocess_job(
        input_path=resolved_input_path,
        log_cb=log_cb,
        **postprocess_args,
    )
    return {
        "generation": generation_result,
        "postprocess": postprocess_result,
    }


def run_generate_model_job(
    *,
    output_dir: str,
    input_dir: str | None = None,
    prompt: str | None = None,
    image_url: str | None = None,
    face_count: int | None = None,
    download_preview: bool = False,
    log_cb: LogCallback = None,
) -> dict:
    return run_hunyuan_job(
        output_dir=output_dir,
        input_dir=input_dir,
        prompt=prompt,
        image_url=image_url,
        result_formats=("GLB",),
        version="pro",
        face_count=face_count,
        download_preview=download_preview,
        verbose=1,
        log_cb=log_cb,
    )


def run_process_model_job(
    *,
    input_path: str,
    len_x: float,
    len_y: float,
    len_z: float,
    orientation: str,
    intermediate_output_dir: str,
    final_output_dir: str,
    set_mass: float | None = None,
    material: str = "plastic",
    approx: str = "convexDecomposition",
    log_cb: LogCallback = None,
) -> dict:
    return run_postprocess_job(
        input_path=input_path,
        len_x=len_x,
        len_y=len_y,
        len_z=len_z,
        intermediate_output_dir=intermediate_output_dir,
        final_output_dir=final_output_dir,
        axis_map=orientation,
        material=material,
        set_mass=set_mass,
        approx=approx,
        usd_format="usd",
        headless=True,
        log_cb=log_cb,
    )


def run_generate_and_process_model_job(
    *,
    output_dir: str,
    intermediate_output_dir: str,
    final_output_dir: str,
    len_x: float,
    len_y: float,
    len_z: float,
    orientation: str,
    set_mass: float | None = None,
    material: str = "plastic",
    approx: str = "convexDecomposition",
    input_dir: str | None = None,
    prompt: str | None = None,
    image_url: str | None = None,
    face_count: int | None = None,
    download_preview: bool = False,
    postprocess_input_path: str | None = None,
    refine_mesh: bool = True,
    refine_output_dir: str | None = None,
    refine_config_path: str | None = None,
    refine_temp_upload: str | None = None,
    refine_fail_on_qc_error: bool = False,
    log_cb: LogCallback = None,
) -> dict:
    generation_result = run_generate_model_job(
        output_dir=output_dir,
        input_dir=input_dir,
        prompt=prompt,
        image_url=image_url,
        face_count=face_count,
        download_preview=download_preview,
        log_cb=log_cb,
    )
    resolved_postprocess_input = postprocess_input_path or output_dir
    refine_result = None
    if refine_mesh:
        refine_result = run_refine_mesh_job(
            input_path=resolved_postprocess_input,
            output_dir=refine_output_dir,
            config_path=refine_config_path,
            temp_upload=refine_temp_upload,
            fail_on_qc_error=refine_fail_on_qc_error,
            log_cb=log_cb,
        )
        resolved_postprocess_input = refine_result["postprocess_input_path"]

    process_result = run_process_model_job(
        input_path=resolved_postprocess_input,
        len_x=len_x,
        len_y=len_y,
        len_z=len_z,
        orientation=orientation,
        intermediate_output_dir=intermediate_output_dir,
        final_output_dir=final_output_dir,
        set_mass=set_mass,
        material=material,
        approx=approx,
        log_cb=log_cb,
    )
    return {
        "generation": generation_result,
        "refine_mesh": refine_result,
        "postprocess": process_result,
    }


def _console_log(message: str) -> None:
    print(message, flush=True)


def _write_result(result: dict, result_json: str | None) -> None:
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if result_json:
        path = Path(result_json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(f"Result JSON: {path}", flush=True)
    else:
        print(text, flush=True)


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", help="输入图片目录")
    parser.add_argument("--prompt", help="文生 3D prompt")
    parser.add_argument("--image-url", help="图片 URL")
    parser.add_argument("--output-dir", default="./downloads", help="混元生成输出目录")
    parser.add_argument("--face-count", type=int, help="混元专业版目标面数")
    parser.add_argument("--download-preview", action="store_true", help="下载混元预览图")


def _add_postprocess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--len-x", type=float, required=True, help="目标 X 尺寸，单位 m")
    parser.add_argument("--len-y", type=float, required=True, help="目标 Y 尺寸，单位 m")
    parser.add_argument("--len-z", type=float, required=True, help="目标 Z 尺寸，单位 m")
    parser.add_argument("--orientation", default="X=L,Y=M,Z=S", help="轴向映射，例如 X=L,Y=M,Z=S")
    parser.add_argument("--intermediate-output-dir", required=True, help="加物理后的中间 USD 输出目录")
    parser.add_argument("--final-output-dir", required=True, help="collect 后最终 USD 输出目录")
    parser.add_argument("--set-mass", type=float, help="固定质量 kg；不传则自动估算")
    parser.add_argument("--material", default="plastic", help="材料标签，对应 materials.json")
    parser.add_argument("--approx", default="convexDecomposition", help="碰撞近似类型")


def _add_refine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-refine-mesh", action="store_false", dest="refine_mesh", help="跳过 refine mesh")
    parser.set_defaults(refine_mesh=True)
    parser.add_argument("--refine-output-dir", help="refine mesh 输出目录")
    parser.add_argument("--refine-config-path", help="refine mesh 配置文件")
    parser.add_argument("--refine-temp-upload", help="临时上传服务，默认 uguu；可传 none 关闭")
    parser.add_argument("--refine-fail-on-qc-error", action="store_true", help="QC fail 时让任务失败")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Hunyuan asset generation and post-processing without the HTTP API.")
    parser.add_argument("--result-json", help="把最终结果写到 JSON 文件；不传则打印到 stdout")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-model", help="只调用混元生成 GLB")
    _add_generation_args(generate)

    process = subparsers.add_parser("process-model", help="只处理已有 GLB，不调用混元生成")
    process.add_argument("--input-path", required=True, help="已有 GLB 文件或目录")
    _add_postprocess_args(process)

    refine = subparsers.add_parser("refine-mesh", help="只对已有 GLB 执行 refine mesh")
    refine.add_argument("--input-path", required=True, help="已有 GLB 文件或目录")
    refine.add_argument("--output-dir", help="refine mesh 输出目录")
    refine.add_argument("--config-path", help="refine mesh 配置文件")
    refine.add_argument("--temp-upload", help="临时上传服务，默认 uguu；可传 none 关闭")
    refine.add_argument("--fail-on-qc-error", action="store_true", help="QC fail 时让任务失败")

    full = subparsers.add_parser("generate-and-process-model", help="混元生成 -> refine mesh -> Blender/Isaac 后处理")
    _add_generation_args(full)
    full.add_argument("--postprocess-input-path", help="指定后处理输入；不传则使用混元输出目录")
    _add_refine_args(full)
    _add_postprocess_args(full)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "generate-model":
            result = run_generate_model_job(
                output_dir=args.output_dir,
                input_dir=args.input_dir,
                prompt=args.prompt,
                image_url=args.image_url,
                face_count=args.face_count,
                download_preview=args.download_preview,
                log_cb=_console_log,
            )
        elif args.command == "process-model":
            result = run_process_model_job(
                input_path=args.input_path,
                len_x=args.len_x,
                len_y=args.len_y,
                len_z=args.len_z,
                orientation=args.orientation,
                intermediate_output_dir=args.intermediate_output_dir,
                final_output_dir=args.final_output_dir,
                set_mass=args.set_mass,
                material=args.material,
                approx=args.approx,
                log_cb=_console_log,
            )
        elif args.command == "refine-mesh":
            result = run_refine_mesh_job(
                input_path=args.input_path,
                output_dir=args.output_dir,
                config_path=args.config_path,
                temp_upload=args.temp_upload,
                fail_on_qc_error=args.fail_on_qc_error,
                log_cb=_console_log,
            )
        elif args.command == "generate-and-process-model":
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
                postprocess_input_path=args.postprocess_input_path,
                refine_mesh=args.refine_mesh,
                refine_output_dir=args.refine_output_dir,
                refine_config_path=args.refine_config_path,
                refine_temp_upload=args.refine_temp_upload,
                refine_fail_on_qc_error=args.refine_fail_on_qc_error,
                log_cb=_console_log,
            )
        else:
            parser.error(f"Unknown command: {args.command}")
            return 2
    except Exception as exc:
        print(f"pipeline_runner: error: {exc}", file=sys.stderr, flush=True)
        return 1

    _write_result(result, args.result_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
