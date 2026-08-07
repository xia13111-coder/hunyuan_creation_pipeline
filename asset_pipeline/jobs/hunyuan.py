"""Tencent Hunyuan model-generation jobs."""

from __future__ import annotations

import sys
from typing import Sequence

from ..command import LogCallback, append_flag, append_option, run_command
from ..paths import list_files_by_suffix


def _validate_image_source(input_dir: str | None, image_url: str | None) -> None:
    """Require exactly one image source before starting a Hunyuan job."""

    source_count = sum(bool(value) for value in (input_dir, image_url))
    if source_count != 1:
        raise ValueError(
            "Hunyuan generation requires exactly one image source: "
            "input_dir or image_url"
        )


def run_hunyuan_job(
    *,
    output_dir: str,
    input_dir: str | None = None,
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
    _validate_image_source(input_dir, image_url)
    args = [
        sys.executable,
        "-m",
        "asset_pipeline.hunyuan_generation",
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
    if image_url:
        args.extend(["--image-url", image_url])
    if not pbr:
        args.append("--no-pbr")
    append_flag(args, "--download-preview", download_preview)
    append_option(args, "--face-count", face_count)
    append_option(args, "--gen-type", gen_type)
    if verbose > 0:
        args.append("-" + ("v" * verbose))

    run_command(args, log_cb=log_cb)
    return {
        "output_dir": output_dir,
        "input_dir": input_dir,
        "image_url": image_url,
        "result_formats": [fmt.upper() for fmt in result_formats],
        "download_preview": download_preview,
        "model_files": list_files_by_suffix(
            output_dir,
            {".glb", ".obj", ".fbx", ".stl", ".usdz", ".mp4"},
        ),
        "preview_files": (
            list_files_by_suffix(
                output_dir,
                {".png", ".jpg", ".jpeg", ".webp", ".bmp"},
                name_contains="_preview",
            )
            if download_preview
            else []
        ),
    }


def run_generate_model_job(
    *,
    output_dir: str,
    input_dir: str | None = None,
    image_url: str | None = None,
    face_count: int | None = None,
    download_preview: bool = False,
    log_cb: LogCallback = None,
) -> dict:
    return run_hunyuan_job(
        output_dir=output_dir,
        input_dir=input_dir,
        image_url=image_url,
        result_formats=("GLB",),
        version="pro",
        face_count=face_count,
        download_preview=download_preview,
        verbose=1,
        log_cb=log_cb,
    )
