#!/usr/bin/env python3

from __future__ import annotations

import os
import threading
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import hunyuan_generation, runtime
from .jobs.hunyuan import run_generate_model_job
from .workflows import run_generate_and_process_model_job, run_process_model_job


app = FastAPI(
    title="Hunyuan Asset Pipeline API",
    version="2.0.0",
    description="Business-oriented API for model generation and post-processing.",
)

jobs_lock = threading.Lock()
jobs: dict[str, dict[str, Any]] = {}
executor = ThreadPoolExecutor(max_workers=int(os.getenv("PIPELINE_MAX_WORKERS", "1")))
max_log_lines = int(os.getenv("PIPELINE_MAX_LOG_LINES", "2000"))
tencent_key_lock = threading.Lock()
tencent_key_status: dict[str, Any] = {
    "valid": None,
    "checked_at": None,
    "code": "NotChecked",
    "message": "腾讯云密钥尚未校验",
    "account_id": None,
    "principal_id": None,
    "arn": None,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def tencent_key_region() -> str:
    return os.getenv("TENCENTCLOUD_REGION", "ap-guangzhou")


def tencent_key_sts_endpoint() -> str:
    return os.getenv("TENCENTCLOUD_STS_ENDPOINT", "sts.tencentcloudapi.com")


def update_tencent_key_status(status: dict[str, Any]) -> dict[str, Any]:
    with tencent_key_lock:
        tencent_key_status.clear()
        tencent_key_status.update(status)
        return dict(tencent_key_status)


def snapshot_tencent_key_status() -> dict[str, Any]:
    with tencent_key_lock:
        return dict(tencent_key_status)


def refresh_tencent_key_status() -> dict[str, Any]:
    checked_at = now_iso()
    try:
        identity = hunyuan_generation.validate_credentials(
            tencent_key_region(),
            tencent_key_sts_endpoint(),
        )
        status = {
            "valid": True,
            "checked_at": checked_at,
            "code": "OK",
            "message": "腾讯云密钥有效",
            "account_id": identity.get("AccountId"),
            "principal_id": identity.get("PrincipalId"),
            "arn": identity.get("Arn"),
        }
    except hunyuan_generation.TencentCloudSDKException as exc:
        status = {
            "valid": False,
            "checked_at": checked_at,
            "code": hunyuan_generation.get_sdk_error_code(exc)
            or "TencentCloudSDKException",
            "message": hunyuan_generation.get_sdk_error_message(exc),
            "account_id": None,
            "principal_id": None,
            "arn": None,
        }
    except SystemExit:
        status = {
            "valid": False,
            "checked_at": checked_at,
            "code": "MissingCredentials",
            "message": "缺少 TENCENTCLOUD_SECRET_ID 或 TENCENTCLOUD_SECRET_KEY 环境变量",
            "account_id": None,
            "principal_id": None,
            "arn": None,
        }
    except Exception as exc:
        status = {
            "valid": False,
            "checked_at": checked_at,
            "code": type(exc).__name__,
            "message": str(exc),
            "account_id": None,
            "principal_id": None,
            "arn": None,
        }
    return update_tencent_key_status(status)


def require_valid_tencent_key() -> None:
    status = snapshot_tencent_key_status()
    if status.get("valid") is None:
        status = refresh_tencent_key_status()
    if status.get("valid") is not True:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "腾讯云密钥不可用，已阻止提交混元生成任务",
                "credential_status": status,
            },
        )


def create_job(kind: str, payload: dict[str, Any]) -> str:
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "kind": kind,
            "status": "queued",
            "created_at": now_iso(),
            "started_at": None,
            "finished_at": None,
            "request": payload,
            "result": None,
            "error": None,
            "logs": deque(maxlen=max_log_lines),
        }
    return job_id


def append_log(job_id: str, message: str) -> None:
    line = message.rstrip()
    if not line:
        return
    with jobs_lock:
        jobs[job_id]["logs"].append(f"[{now_iso()}] {line}")


def snapshot_job(job_id: str, *, log_tail: Optional[int] = None) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        data = {k: v for k, v in job.items() if k != "logs"}
        logs = list(job["logs"])
    data["logs"] = logs if log_tail is None else logs[-log_tail:]
    return data


def run_async_job(job_id: str, func) -> None:
    def _wrapped() -> None:
        with jobs_lock:
            jobs[job_id]["status"] = "running"
            jobs[job_id]["started_at"] = now_iso()
        try:
            result = func(lambda message: append_log(job_id, message))
            with jobs_lock:
                jobs[job_id]["status"] = "succeeded"
                jobs[job_id]["result"] = result
                jobs[job_id]["finished_at"] = now_iso()
        except Exception as exc:
            append_log(job_id, f"ERROR: {exc}")
            with jobs_lock:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = str(exc)
                jobs[job_id]["finished_at"] = now_iso()

    executor.submit(_wrapped)


def submit_job(kind: str, payload: dict[str, Any], func) -> dict[str, Any]:
    job_id = create_job(kind, payload)
    run_async_job(job_id, func)
    return snapshot_job(job_id, log_tail=50)


class GenerateModelRequest(BaseModel):
    input_dir: str | None = None
    prompt: str | None = None
    image_url: str | None = None
    output_dir: str = "./downloads"
    face_count: int | None = Field(
        default=None, description="映射到 Hunyuan generation 模块的 --face-count"
    )
    download_preview: bool = Field(
        default=False, description="下载腾讯云返回的 PreviewImageUrl 预览图"
    )


class ProcessModelRequest(BaseModel):
    input_path: str
    intermediate_output_dir: str
    final_output_dir: str
    len_x: float = Field(description="模型 X 尺寸")
    len_y: float = Field(description="模型 Y 尺寸")
    len_z: float = Field(description="模型 Z 尺寸")
    orientation: str = Field(
        default="X=L,Y=M,Z=S", description="模型朝向，直接映射到 --axis-map"
    )
    set_mass: float | None = Field(
        default=None, description="模型质量；为空时自动按体积和材质密度计算"
    )
    material: str = Field(
        default="plastic", description="材质标签，对应 materials.json"
    )
    approx: str = Field(
        default="convexDecomposition", description="碰撞体类型，对应 --approx"
    )


class GenerateAndProcessModelRequest(BaseModel):
    input_dir: str | None = None
    prompt: str | None = None
    image_url: str | None = None
    output_dir: str = "./downloads"
    download_preview: bool = Field(
        default=False, description="下载腾讯云返回的 PreviewImageUrl 预览图"
    )
    postprocess_input_path: str | None = None
    intermediate_output_dir: str
    final_output_dir: str
    face_count: int | None = None
    refine_mesh: bool = Field(
        default=True,
        description="混元生成 GLB 后先执行 refine mesh，再进入 Blender/Isaac 后处理",
    )
    refine_output_dir: str | None = Field(
        default=None,
        description="refine mesh 输出目录；为空时自动使用 output_dir + '_refined_mesh'",
    )
    refine_config_path: str | None = Field(
        default=None,
        description="refine mesh 配置文件；为空时使用 configs/hunyuan_reduce_local_postprocess.yaml",
    )
    refine_temp_upload: str | None = Field(
        default=None,
        description="临时公网上传服务；默认取 REFINE_MESH_TEMP_UPLOAD，未设置时为 uguu",
    )
    refine_fail_on_qc_error: bool = Field(
        default=False, description="QC 状态为 fail 时是否让任务失败"
    )
    len_x: float
    len_y: float
    len_z: float
    orientation: str = "X=L,Y=M,Z=S"
    set_mass: float | None = None
    material: str = "plastic"
    approx: str = "convexDecomposition"


@app.on_event("startup")
def check_tencent_key_on_startup() -> None:
    runtime.require_unified_environment()
    refresh_tencent_key_status()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "runtime": runtime.runtime_summary(),
        "supported_materials": runtime.available_materials(),
        "supported_approx_types": runtime.available_approx_types(),
        "queue_workers": int(os.getenv("PIPELINE_MAX_WORKERS", "1")),
        "tencent_cloud_credentials": snapshot_tencent_key_status(),
    }


@app.get("/credentials/tencent-cloud")
def get_tencent_cloud_credentials() -> dict[str, Any]:
    return snapshot_tencent_key_status()


@app.post("/credentials/tencent-cloud/check")
def check_tencent_cloud_credentials() -> dict[str, Any]:
    return refresh_tencent_key_status()


@app.get("/jobs")
def list_jobs() -> list[dict[str, Any]]:
    with jobs_lock:
        job_ids = list(jobs.keys())
    return [snapshot_job(job_id, log_tail=20) for job_id in job_ids]


@app.get("/jobs/{job_id}")
def get_job(job_id: str, log_tail: int = 200) -> dict[str, Any]:
    try:
        return snapshot_job(job_id, log_tail=log_tail)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from exc


@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str, tail: int = 200) -> dict[str, Any]:
    try:
        job = snapshot_job(job_id, log_tail=tail)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from exc
    return {"job_id": job_id, "logs": job["logs"]}


@app.post("/jobs/generate-model")
def generate_model(request: GenerateModelRequest) -> dict[str, Any]:
    payload = model_dump(request)
    if not (
        payload.get("input_dir") or payload.get("prompt") or payload.get("image_url")
    ):
        raise HTTPException(
            status_code=400, detail="input_dir、prompt、image_url 至少提供一个"
        )
    require_valid_tencent_key()
    return submit_job(
        "generate_model",
        payload,
        lambda log_cb: run_generate_model_job(log_cb=log_cb, **payload),
    )


@app.post("/jobs/process-model")
def process_model(request: ProcessModelRequest) -> dict[str, Any]:
    payload = model_dump(request)
    return submit_job(
        "process_model",
        payload,
        lambda log_cb: run_process_model_job(log_cb=log_cb, **payload),
    )


@app.post("/jobs/generate-and-process-model")
def generate_and_process_model(
    request: GenerateAndProcessModelRequest,
) -> dict[str, Any]:
    payload = model_dump(request)
    if not (
        payload.get("input_dir") or payload.get("prompt") or payload.get("image_url")
    ):
        raise HTTPException(
            status_code=400, detail="input_dir、prompt、image_url 至少提供一个"
        )
    require_valid_tencent_key()
    return submit_job(
        "generate_and_process_model",
        payload,
        lambda log_cb: run_generate_and_process_model_job(log_cb=log_cb, **payload),
    )
