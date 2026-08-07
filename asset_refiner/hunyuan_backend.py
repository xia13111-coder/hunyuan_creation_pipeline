"""Tencent Hunyuan ReduceFace backend plus local Blender postprocess.

Call flow:
runner.run_refinement
-> run_hunyuan_refinement
-> resolve input URL or temporary upload
-> submit_reduce_face -> run_job
-> download ReduceFace result
-> run_local_postprocess_worker -> blender_worker.py
"""

from __future__ import annotations

import copy
import http.client
import json
import os
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import BackendExecutionError, HunyuanApiError
from .tencent_api import TencentCloudApiClient
from .temporary_upload import upload_to_temporary_host


SUPPORTED_INPUT_TYPES = {"GLB", "OBJ", "FBX"}
FINAL_FILE_TYPES = ["GLB", "FBX", "ZIP", "OBJ"]


@dataclass(frozen=True)
class RemoteFile:
    type: str
    url: str


def is_http_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def infer_file_type(value: str) -> str:
    clean_path = urllib.parse.urlparse(value).path if is_http_url(value) else value
    suffix = Path(clean_path).suffix.lower().lstrip(".")
    if suffix == "gltf":
        return "GLB"
    if not suffix:
        raise HunyuanApiError(f"Cannot infer 3D file type from: {value}")
    return suffix.upper()


def resolve_remote_input(input_ref: str, config: dict[str, Any]) -> RemoteFile:
    hunyuan = config.get("hunyuan", {})
    input_url_env = str(hunyuan.get("input_url_env") or "HUNYUAN_INPUT_URL")
    url = hunyuan.get("input_url") or os.environ.get(input_url_env) or (input_ref if is_http_url(input_ref) else None)
    if not url:
        raise HunyuanApiError(
            "Hunyuan API requires a public or signed input URL. "
            "Set hunyuan.input_url in the config, or pass an http(s) URL as --input. "
            f"You can also set {input_url_env}. "
            "The local --input path is still used for QC when available, but Tencent's API cannot fetch it directly."
        )

    file_type = str(hunyuan.get("input_type") or infer_file_type(str(url))).upper()
    if file_type not in SUPPORTED_INPUT_TYPES:
        raise HunyuanApiError(f"Unsupported Hunyuan input type: {file_type}. Supported: {sorted(SUPPORTED_INPUT_TYPES)}")
    return RemoteFile(type=file_type, url=str(url))


def resolve_api_upload_input_ref(input_ref: str, config: dict[str, Any]) -> str:
    hunyuan = config.get("hunyuan", {})
    env_name = str(hunyuan.get("upload_input_path_env") or "HUNYUAN_UPLOAD_INPUT_PATH")
    upload_input = hunyuan.get("upload_input_path") or os.environ.get(env_name)
    return str(upload_input or input_ref)


def temp_upload_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("hunyuan", {}).get("temp_upload", {}).get("enabled", False))


def local_postprocess_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("hunyuan", {}).get("local_postprocess", {}).get("enabled", False))


def planned_temp_upload_input(input_ref: str, config: dict[str, Any]) -> RemoteFile:
    file_type = str(config.get("hunyuan", {}).get("input_type") or infer_file_type(input_ref)).upper()
    provider = str(config.get("hunyuan", {}).get("temp_upload", {}).get("provider") or "temporary-host")
    return RemoteFile(type=file_type, url=f"<{provider}-temporary-upload>/{Path(input_ref).name}")


def choose_result_file(files: list[dict[str, Any]], preferences: list[str] | None = None) -> RemoteFile:
    if not files:
        raise HunyuanApiError("Hunyuan job completed without ResultFile3Ds")
    by_type: dict[str, dict[str, Any]] = {}
    for item in files:
        file_type = str(item.get("Type") or "").upper()
        if file_type and item.get("Url"):
            by_type.setdefault(file_type, item)
    for preference in preferences or FINAL_FILE_TYPES:
        item = by_type.get(preference.upper())
        if item:
            return RemoteFile(type=str(item["Type"]).upper(), url=str(item["Url"]))
    first = next((item for item in files if item.get("Url")), None)
    if not first:
        raise HunyuanApiError("Hunyuan ResultFile3Ds did not include downloadable Url values")
    return RemoteFile(type=str(first.get("Type") or "UNKNOWN").upper(), url=str(first["Url"]))


RETRYABLE_RESULT_DOWNLOAD_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}


def is_retryable_result_download_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in RETRYABLE_RESULT_DOWNLOAD_HTTP_CODES
    return isinstance(
        exc,
        (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            ssl.SSLError,
            http.client.IncompleteRead,
        ),
    )


def download_url(
    url: str,
    destination: Path,
    *,
    max_attempts: int = 5,
    retry_interval_seconds: float = 5,
    retry_backoff_factor: float = 2.0,
    timeout_seconds: float = 120,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(f"{destination.name}.part")
    attempts = max(1, int(max_attempts))
    interval = max(0.0, float(retry_interval_seconds))
    backoff = max(1.0, float(retry_backoff_factor))

    for attempt in range(1, attempts + 1):
        part_path.unlink(missing_ok=True)
        print(
            f"Downloading Hunyuan result: attempt {attempt}/{attempts}",
            file=sys.stderr,
            flush=True,
        )
        try:
            with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
                content_length = response.headers.get("Content-Length")
                with part_path.open("wb") as handle:
                    shutil.copyfileobj(response, handle)

            if content_length:
                try:
                    expected_size = int(content_length)
                except ValueError:
                    expected_size = None
                if expected_size is not None:
                    actual_size = part_path.stat().st_size
                    if actual_size != expected_size:
                        raise urllib.error.ContentTooShortError(
                            f"Hunyuan result download was incomplete: "
                            f"expected {expected_size} bytes, got {actual_size}",
                            None,
                        )

            part_path.replace(destination)
            return destination
        except Exception as exc:
            part_path.unlink(missing_ok=True)
            if attempt >= attempts or not is_retryable_result_download_error(exc):
                raise
            print(
                f"Hunyuan result download failed: {exc}. "
                f"Retrying in {interval:.0f}s.",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(interval)
            interval *= backoff

    raise RuntimeError("Hunyuan result download retry loop exited unexpectedly")


def prepare_downloaded_model(path: Path) -> Path:
    if path.suffix.lower() != ".zip":
        return path
    extract_dir = path.with_suffix("")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as archive:
        archive.extractall(extract_dir)
    candidates = []
    for suffix in ("*.glb", "*.gltf", "*.fbx", "*.obj"):
        candidates.extend(extract_dir.rglob(suffix))
    if not candidates:
        raise HunyuanApiError(f"Downloaded ZIP did not contain an importable 3D file: {path}")
    candidates.sort(key=lambda item: ([".glb", ".gltf", ".fbx", ".obj"].index(item.suffix.lower()), len(str(item))))
    return candidates[0]


def is_retryable_describe_error(
    exc: Exception,
    retry_error_codes: list[str],
) -> bool:
    if is_retryable_result_download_error(exc):
        return True
    cause = exc.__cause__
    if isinstance(cause, Exception) and is_retryable_result_download_error(cause):
        return True

    message = str(exc)
    if "Tencent API request failed for " in message:
        return True
    return any(code and code in message for code in retry_error_codes)


def describe_job_with_retry(
    client: TencentCloudApiClient,
    *,
    describe_action: str,
    job_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    max_attempts = max(1, int(config.get("describe_max_attempts", 6) or 6))
    interval_value = config.get("describe_retry_interval_seconds", 5)
    interval = max(0.0, float(5 if interval_value is None else interval_value))
    backoff_value = config.get("describe_retry_backoff_factor", 2.0)
    backoff = max(1.0, float(2.0 if backoff_value is None else backoff_value))
    retry_codes = [
        str(code) for code in config.get("describe_retry_error_codes", [])
    ]

    for attempt in range(1, max_attempts + 1):
        try:
            return client.call(describe_action, {"JobId": job_id})
        except Exception as exc:
            if (
                attempt >= max_attempts
                or not is_retryable_describe_error(exc, retry_codes)
            ):
                raise
            print(
                f"Hunyuan status query {describe_action} failed "
                f"(attempt {attempt}/{max_attempts}): {exc}. "
                f"Retrying in {interval:.0f}s.",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(interval)
            interval *= backoff

    raise RuntimeError("Hunyuan status query retry loop exited unexpectedly")


def run_job(
    client: TencentCloudApiClient,
    *,
    submit_action: str,
    describe_action: str,
    submit_payload: dict[str, Any],
    poll_interval_seconds: int,
    timeout_seconds: int,
    stage_name: str,
) -> dict[str, Any]:
    hunyuan_cfg = submit_payload.pop("__hunyuan_retry_config", {})
    submit_response = submit_job_with_retry(
        client,
        submit_action=submit_action,
        submit_payload=submit_payload,
        config=hunyuan_cfg,
    )
    job_id = submit_response.get("JobId")
    if not job_id:
        raise HunyuanApiError(f"{submit_action} did not return JobId: {submit_response}")
    print(f"Hunyuan {stage_name} job submitted: {job_id}", file=sys.stderr, flush=True)

    deadline = time.monotonic() + timeout_seconds
    polls: list[dict[str, Any]] = []
    last_status: str | None = None
    while True:
        describe_response = describe_job_with_retry(
            client,
            describe_action=describe_action,
            job_id=str(job_id),
            config=hunyuan_cfg,
        )
        status = str(describe_response.get("Status") or "").upper()
        if status != last_status:
            print(f"Hunyuan {stage_name} job {job_id}: {status}", file=sys.stderr, flush=True)
            last_status = status
        polls.append(
            {
                "status": status,
                "request_id": describe_response.get("RequestId"),
                "error_code": describe_response.get("ErrorCode"),
                "error_message": describe_response.get("ErrorMessage"),
            }
        )
        if status == "DONE":
            return {
                "stage": stage_name,
                "submit_action": submit_action,
                "describe_action": describe_action,
                "job_id": job_id,
                "submit_response": submit_response,
                "describe_response": describe_response,
                "polls": polls,
            }
        if status == "FAIL":
            raise HunyuanApiError(
                f"Hunyuan {stage_name} job failed: "
                f"{describe_response.get('ErrorCode')} {describe_response.get('ErrorMessage')}"
            )
        if time.monotonic() > deadline:
            raise HunyuanApiError(f"Hunyuan {stage_name} job timed out after {timeout_seconds}s")
        time.sleep(poll_interval_seconds)


def should_retry_submit_error(exc: HunyuanApiError, retry_error_codes: list[str]) -> bool:
    message = str(exc)
    return any(code and code in message for code in retry_error_codes)


def is_download_error(exc: HunyuanApiError) -> bool:
    return "DownloadError" in str(exc)


def is_retryable_job_failure(
    exc: HunyuanApiError,
    retry_error_codes: list[str],
) -> bool:
    message = str(exc)
    return any(code and code in message for code in retry_error_codes)


def submit_job_with_retry(
    client: TencentCloudApiClient,
    *,
    submit_action: str,
    submit_payload: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    max_retries = max(1, int(config.get("submit_max_retries", 1) or 1))
    interval_value = config.get("submit_retry_interval_seconds", 60)
    interval = float(60 if interval_value is None else interval_value)
    backoff_value = config.get("submit_retry_backoff_factor", 1.0)
    backoff = max(1.0, float(1.0 if backoff_value is None else backoff_value))
    retry_codes = [str(code) for code in config.get("submit_retry_error_codes", [])]
    last_error: HunyuanApiError | None = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Hunyuan submit {submit_action}: attempt {attempt}/{max_retries}", file=sys.stderr, flush=True)
            return client.call(submit_action, submit_payload)
        except HunyuanApiError as exc:
            last_error = exc
            if attempt >= max_retries or not should_retry_submit_error(exc, retry_codes):
                raise
            print(
                f"Hunyuan submit {submit_action} retryable error: {exc}. "
                f"Retrying in {interval:.0f}s.",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(interval)
            interval *= backoff

    assert last_error is not None
    raise last_error


def submit_reduce_face(
    client: TencentCloudApiClient,
    input_file: RemoteFile,
    config: dict[str, Any],
) -> dict[str, Any]:
    retopo = config.get("hunyuan", {}).get("retopology", {})
    payload: dict[str, Any] = {
        "File3D": {"Type": input_file.type, "Url": input_file.url},
        "__hunyuan_retry_config": config.get("hunyuan", {}),
    }
    if retopo.get("polygon_type"):
        payload["PolygonType"] = retopo["polygon_type"]
    if retopo.get("face_level"):
        payload["FaceLevel"] = retopo["face_level"]
    return run_job(
        client,
        submit_action="SubmitReduceFaceJob",
        describe_action="DescribeReduceFaceJob",
        submit_payload=payload,
        poll_interval_seconds=int(config.get("hunyuan", {}).get("poll_interval_seconds", 10) or 10),
        timeout_seconds=int(config.get("hunyuan", {}).get("timeout_seconds", 3600) or 3600),
        stage_name="reduce_face",
    )


def submit_reduce_face_with_temporary_upload(
    client: TencentCloudApiClient,
    *,
    local_input: Path,
    input_type: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Retry temporary uploads, remote downloads, and retryable failed jobs."""
    hunyuan_cfg = config.get("hunyuan", {})
    temp_cfg = hunyuan_cfg.get("temp_upload", {})
    max_url_attempts = max(1, int(temp_cfg.get("download_error_max_retries", 4) or 4))
    url_retry_interval = float(
        temp_cfg.get("download_error_retry_interval_seconds", 10) or 10
    )
    job_retry_cfg = hunyuan_cfg.get("job_failure_retry", {})
    max_job_attempts = max(1, int(job_retry_cfg.get("max_attempts", 1) or 1))
    job_retry_interval_value = job_retry_cfg.get("retry_interval_seconds", 30)
    job_retry_interval = max(
        0.0,
        float(
            30
            if job_retry_interval_value is None
            else job_retry_interval_value
        ),
    )
    job_retry_backoff_value = job_retry_cfg.get(
        "retry_backoff_factor",
        2.0,
    )
    job_retry_backoff = max(
        1.0,
        float(
            2.0
            if job_retry_backoff_value is None
            else job_retry_backoff_value
        ),
    )
    job_retry_codes = [
        str(code) for code in job_retry_cfg.get("error_codes", [])
    ]

    for job_attempt in range(1, max_job_attempts + 1):
        print(
            f"Hunyuan reduce_face task attempt "
            f"{job_attempt}/{max_job_attempts}",
            file=sys.stderr,
            flush=True,
        )
        for upload_attempt in range(1, max_url_attempts + 1):
            print(
                f"Uploading local input to temporary public host for Hunyuan API "
                f"({upload_attempt}/{max_url_attempts})...",
                file=sys.stderr,
                flush=True,
            )
            try:
                uploaded_url = upload_to_temporary_host(local_input, config)
            except HunyuanApiError as exc:
                if upload_attempt >= max_url_attempts:
                    raise
                print(
                    f"Temporary upload failed: {exc}. "
                    f"Retrying in {url_retry_interval:.0f}s.",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(url_retry_interval)
                continue

            print(
                "Temporary upload finished; submitting uploaded URL to "
                "Hunyuan API.",
                file=sys.stderr,
                flush=True,
            )
            current_file = RemoteFile(type=input_type, url=uploaded_url)
            try:
                return submit_reduce_face(client, current_file, config)
            except HunyuanApiError as exc:
                if is_download_error(exc):
                    if upload_attempt >= max_url_attempts:
                        raise
                    print(
                        f"Hunyuan could not download that temporary URL: {exc}. "
                        f"Re-uploading in {url_retry_interval:.0f}s.",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(url_retry_interval)
                    continue
                if (
                    job_attempt >= max_job_attempts
                    or not is_retryable_job_failure(exc, job_retry_codes)
                ):
                    raise
                print(
                    f"Hunyuan reduce_face task failed with retryable error "
                    f"(attempt {job_attempt}/{max_job_attempts}): {exc}. "
                    f"Re-uploading and submitting a new job in "
                    f"{job_retry_interval:.0f}s.",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(job_retry_interval)
                job_retry_interval *= job_retry_backoff
                break

    raise HunyuanApiError(
        "Hunyuan task retry loop exited without a ReduceFace result"
    )


def local_worker_path() -> Path:
    return Path(__file__).with_name("blender_worker.py")


def blender_executable(config: dict[str, Any]) -> str:
    blender = shutil.which(config.get("backend", {}).get("blender_executable") or "blender")
    if blender:
        return blender
    candidate = Path(config.get("backend", {}).get("blender_executable") or "blender")
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError("Blender executable not found for Hunyuan API local processing")


def build_local_postprocess_config(config: dict[str, Any], target_path: Path) -> dict[str, Any]:
    local_config = copy.deepcopy(config)
    local_config.setdefault("backend", {})["name"] = "blender"
    retopo = local_config.setdefault("retopology", {})
    retopo["method"] = "external_target_project"
    retopo["target_path"] = str(target_path.resolve())
    retopo.setdefault("normalize_external_target_to_source_bbox", True)
    return local_config


def run_local_postprocess_worker(
    *,
    input_ref: str,
    target_path: Path,
    output_dir: Path,
    report_path: Path,
    log_path: Path,
    config: dict[str, Any],
) -> Path:
    """Run blender_worker.py with the ReduceFace result as the retopo target."""
    if is_http_url(input_ref) or not Path(input_ref).exists():
        raise HunyuanApiError(
            "Hunyuan local postprocess requires a local --input path so Blender can migrate textures "
            "from the original source model."
        )

    local_config = build_local_postprocess_config(config, target_path)
    local_config_path = output_dir / "resolved_local_postprocess_config.json"
    local_config_path.write_text(json.dumps(local_config, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    command = [
        blender_executable(config),
        "--background",
        "--factory-startup",
        "--python",
        str(local_worker_path()),
        "--",
        "--input",
        str(Path(input_ref).resolve()),
        "--output",
        str(output_dir),
        "--config-json",
        str(local_config_path),
        "--report",
        str(report_path),
    ]
    print("Running local Blender UV and texture postprocess...", file=sys.stderr, flush=True)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    log_path.write_text(
        (completed.stdout or "") + ("\n" if completed.stdout and completed.stderr else "") + (completed.stderr or ""),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        excerpt = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise BackendExecutionError(
            f"Hunyuan local postprocess worker failed with exit code {completed.returncode}. "
            f"Log: {log_path}\n{excerpt}"
        )
    return local_config_path


def run_hunyuan_refinement(
    *,
    input_ref: str,
    output_dir: Path,
    report_path: Path,
    config: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Submit ReduceFace, download the target, then run local Blender postprocess."""
    temp_upload_required = False
    use_local_postprocess = local_postprocess_enabled(config)
    if not use_local_postprocess:
        raise HunyuanApiError(
            "The Hunyuan backend now supports only ReduceFace plus local Blender postprocess. "
            "Set hunyuan.local_postprocess.enabled=true."
        )
    api_input_ref = resolve_api_upload_input_ref(input_ref, config)
    try:
        remote_input = resolve_remote_input(api_input_ref, config)
    except HunyuanApiError:
        if temp_upload_enabled(config) and not is_http_url(api_input_ref) and Path(api_input_ref).exists():
            temp_upload_required = True
            remote_input = planned_temp_upload_input(api_input_ref, config)
        else:
            raise
    plan = {
        "backend": "hunyuan_api",
        "source_input": str(input_ref),
        "api_upload_input": str(api_input_ref),
        "remote_input": remote_input.__dict__,
        "temporary_upload": temp_upload_required,
        "stages": ["reduce_face", "local_uv_texture_postprocess"],
    }
    if dry_run:
        return {"status": "dry_run", "plan": plan}

    client = TencentCloudApiClient.from_config(config)
    intermediate = output_dir / "intermediate" / "hunyuan_api"
    intermediate.mkdir(parents=True, exist_ok=True)
    api_stages: list[dict[str, Any]] = []

    current_file = remote_input
    if config.get("hunyuan", {}).get("retopology", {}).get("enabled", True):
        if temp_upload_required:
            reduce_result = submit_reduce_face_with_temporary_upload(
                client,
                local_input=Path(api_input_ref).resolve(),
                input_type=remote_input.type,
                config=config,
            )
        else:
            reduce_result = submit_reduce_face(client, current_file, config)
        api_stages.append(reduce_result)
        current_file = choose_result_file(
            reduce_result["describe_response"].get("ResultFile3Ds", []),
            list(config.get("hunyuan", {}).get("download_preference", [])),
        )

    api_summary = {
        "backend": "hunyuan_api",
        "source_input": str(input_ref),
        "api_upload_input": str(api_input_ref),
        "remote_input": remote_input.__dict__,
        "selected_result": current_file.__dict__,
        "stages": api_stages,
        "policy": {
            "whole_asset_processing": True,
            "semantic_segmentation": False,
            "component_generation_api_used": False,
            "local_uv_texture_postprocess": use_local_postprocess,
        },
    }
    (output_dir / "hunyuan_api_result.json").write_text(json.dumps(api_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    suffix = "." + current_file.type.lower().replace("unknown", "bin")
    download_name = "hunyuan_reduce_target"
    result_download_cfg = config.get("hunyuan", {}).get("result_download", {})
    api_result_path = prepare_downloaded_model(
        download_url(
            current_file.url,
            intermediate / f"{download_name}{suffix}",
            max_attempts=int(result_download_cfg.get("max_attempts", 5)),
            retry_interval_seconds=float(
                result_download_cfg.get("retry_interval_seconds", 5)
            ),
            retry_backoff_factor=float(
                result_download_cfg.get("retry_backoff_factor", 2.0)
            ),
            timeout_seconds=float(result_download_cfg.get("timeout_seconds", 120)),
        )
    )

    local_log_path = output_dir / "hunyuan_local_postprocess_blender.log"
    local_config_path = run_local_postprocess_worker(
        input_ref=input_ref,
        target_path=api_result_path,
        output_dir=output_dir,
        report_path=report_path,
        log_path=local_log_path,
        config=config,
    )
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    api_summary["local_postprocess"] = {
        "method": "blender_external_target_project_uv_texture_migration",
        "target_path": str(api_result_path),
        "config_path": str(local_config_path),
        "log_path": str(local_log_path),
    }
    report["hunyuan_api"] = api_summary
    report.setdefault("stages", {})["hunyuan_api"] = {
        "method": "SubmitReduceFaceJob",
        "job_ids": [stage.get("job_id") for stage in api_stages],
        "local_postprocess": True,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    (output_dir / "hunyuan_api_result.json").write_text(
        json.dumps(api_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report
