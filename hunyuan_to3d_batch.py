#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
from typing import List, Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Tencent Cloud SDK (Hunyuan 3D, 2025-05-13)
try:
    from tencentcloud.common import credential
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile
    from tencentcloud.ai3d.v20250513 import ai3d_client, models
    from tencentcloud.common.exception.tencent_cloud_sdk_exception import (
        TencentCloudSDKException,
    )
except ImportError:
    credential = None
    ClientProfile = None
    HttpProfile = None
    ai3d_client = None
    models = None

    class TencentCloudSDKException(Exception):
        pass

try:
    from tencentcloud.sts.v20180813 import sts_client, models as sts_models
except ImportError:
    sts_client = None
    sts_models = None

# =========================
# 基础工具
# =========================
def setup_logger(verbosity: int):
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def build_requests_session(total_retries=3, backoff=0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def require_ascii_credential(name: str, value: str):
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise SystemExit(
            f"{name} 只能包含 ASCII 字符。当前值里可能仍是中文占位符，"
            f"请重新 export 真实的腾讯云密钥。"
        ) from exc

def load_credentials():
    sid = os.getenv("TENCENTCLOUD_SECRET_ID")
    skey = os.getenv("TENCENTCLOUD_SECRET_KEY")
    if not sid or not skey:
        logging.error("找不到云凭证环境变量: TENCENTCLOUD_SECRET_ID / TENCENTCLOUD_SECRET_KEY")
        logging.error("请先执行: export TENCENTCLOUD_SECRET_ID=你的ID; export TENCENTCLOUD_SECRET_KEY=你的KEY")
        raise SystemExit(2)
    require_ascii_credential("TENCENTCLOUD_SECRET_ID", sid)
    require_ascii_credential("TENCENTCLOUD_SECRET_KEY", skey)
    if credential is None:
        raise RuntimeError("当前环境缺少腾讯云 SDK，请安装 tencentcloud-sdk-python")
    return credential.Credential(sid, skey)

def get_sdk_error_code(exc: TencentCloudSDKException) -> str:
    getter = getattr(exc, "get_code", None)
    if callable(getter):
        return getter() or ""
    return getattr(exc, "code", "") or ""

def get_sdk_error_message(exc: TencentCloudSDKException) -> str:
    getter = getattr(exc, "get_message", None)
    if callable(getter):
        return getter() or str(exc)
    return getattr(exc, "message", str(exc)) or str(exc)

def validate_credentials(region: str, endpoint: str = "sts.tencentcloudapi.com") -> dict:
    """
    通过 STS GetCallerIdentity 校验当前环境变量中的腾讯云密钥。
    该接口只查询调用者身份，不会提交混元3D生成任务。
    """
    if (
        credential is None
        or ClientProfile is None
        or HttpProfile is None
        or sts_client is None
        or sts_models is None
    ):
        raise RuntimeError("当前环境缺少腾讯云 SDK，请安装 tencentcloud-sdk-python")

    hp = HttpProfile()
    hp.endpoint = endpoint
    hp.reqMethod = "POST"
    hp.timeout = 30
    cp = ClientProfile()
    cp.httpProfile = hp
    cp.signMethod = "TC3-HMAC-SHA256"

    client = sts_client.StsClient(load_credentials(), region, cp)
    req = sts_models.GetCallerIdentityRequest()
    resp = client.GetCallerIdentity(req)
    data = json.loads(resp.to_json_string())
    return data.get("Response") or data

def init_client(region: str, endpoint: str) -> ai3d_client.Ai3dClient:
    if (
        credential is None
        or ClientProfile is None
        or HttpProfile is None
        or ai3d_client is None
        or models is None
    ):
        raise RuntimeError("当前环境缺少腾讯云 AI3D SDK，请安装 tencentcloud-sdk-python-ai3d")

    hp = HttpProfile()
    hp.endpoint = endpoint
    hp.reqMethod = "POST"
    hp.timeout = 30
    cp = ClientProfile()
    cp.httpProfile = hp
    cp.signMethod = "TC3-HMAC-SHA256"
    cred = load_credentials()
    return ai3d_client.Ai3dClient(cred, region, cp)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def get_format_field_mapping():
    # 以 ResultFile3Ds 为主（极速/专业版都兼容）
    return {
        "GLB": ["ResultFile3Ds"],
        "OBJ": ["ResultFile3Ds"],
        "FBX": ["ResultFile3Ds"],
        "STL": ["ResultFile3Ds"],
        "USDZ": ["ResultFile3Ds"],
        "MP4": ["ResultFile3Ds"],
    }

# =========================
# 请求载荷构造
# =========================
def build_payload_single_rapid(
    image_path: Optional[str],
    result_format: str,
    enable_pbr: bool,
    prompt: Optional[str] = None,
    image_url: Optional[str] = None,
):
    """
    极速版：Prompt / ImageBase64 / ImageUrl 三选一
    """
    payload = {
        "EnablePBR": bool(enable_pbr),
        "ResultFormat": result_format.upper() if result_format else None,
    }
    if prompt:
        payload["Prompt"] = prompt
    elif image_url:
        payload["ImageUrl"] = image_url
    else:
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")
        if os.path.getsize(image_path) < 1024:
            raise ValueError(f"图片过小({os.path.getsize(image_path)}B): {image_path}")
        with open(image_path, "rb") as f:
            payload["ImageBase64"] = base64.b64encode(f.read()).decode("utf-8")

    return {k: v for k, v in payload.items() if v is not None}

def build_payload_single_pro(
    image_path: Optional[str],
    result_format: str,
    enable_pbr: bool,
    prompt: Optional[str] = None,
    image_url: Optional[str] = None,
    face_count: Optional[int] = None,
    generate_type: Optional[str] = None,
):
    """
    专业版：支持可选 FaceCount / GenerateType
    """
    payload = {
        "EnablePBR": bool(enable_pbr),
        "ResultFormat": result_format.upper() if result_format else None,
    }
    if prompt:
        payload["Prompt"] = prompt
    elif image_url:
        payload["ImageUrl"] = image_url
    else:
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")
        if os.path.getsize(image_path) < 1024:
            raise ValueError(f"图片过小({os.path.getsize(image_path)}B): {image_path}")
        with open(image_path, "rb") as f:
            payload["ImageBase64"] = base64.b64encode(f.read()).decode("utf-8")

    if face_count:
        payload["FaceCount"] = face_count
    if generate_type:
        payload["GenerateType"] = generate_type

    return {k: v for k, v in payload.items() if v is not None}

def submit_job_with_retry(
    client: ai3d_client.Ai3dClient,
    payload: dict,
    result_format: str,
    version: str,
    max_retry: int = 10,
    retry_interval: int = 60,
) -> str:
    result_format = (result_format or "").upper()
    for retry_count in range(max_retry):
        try:
            if version == "rapid":
                req = models.SubmitHunyuanTo3DRapidJobRequest()
                req.from_json_string(json.dumps(payload))
                resp = client.SubmitHunyuanTo3DRapidJob(req)
            else:
                req = models.SubmitHunyuanTo3DProJobRequest()
                req.from_json_string(json.dumps(payload))
                resp = client.SubmitHunyuanTo3DProJob(req)

            data = json.loads(resp.to_json_string())
            job_id = data.get("JobId") or (data.get("Response") or {}).get("JobId")
            if not job_id:
                raise RuntimeError(f"接口未返回JobId，响应: {json.dumps(data, ensure_ascii=False)[:300]}...")
            logging.info(f"任务提交成功 | {version} | 格式={result_format} | JobId={job_id} | 重试次数={retry_count}")
            return job_id

        except TencentCloudSDKException as e:
            logging.warning(f"提交失败[{getattr(e, 'code', '')}]: {getattr(e, 'message', str(e))} | {retry_interval}s后重试（剩余{max_retry - retry_count - 1}次）")
            time.sleep(retry_interval)
            continue
        except Exception as e:
            logging.error(f"任务提交异常 | 格式={result_format} | 原因={str(e)}", exc_info=True)
            raise
    raise RuntimeError(f"超过最大重试次数（{max_retry}次）| 格式={result_format}")

def query_job(client: ai3d_client.Ai3dClient, job_id: str, version: str) -> dict:
    if version == "rapid":
        req = models.QueryHunyuanTo3DRapidJobRequest()
        req.from_json_string(json.dumps({"JobId": job_id}))
        resp = client.QueryHunyuanTo3DRapidJob(req)
    else:
        req = models.QueryHunyuanTo3DProJobRequest()
        req.from_json_string(json.dumps({"JobId": job_id}))
        resp = client.QueryHunyuanTo3DProJob(req)
    return json.loads(resp.to_json_string())

def poll_until_done(
    client: ai3d_client.Ai3dClient,
    job_id: str,
    result_format: str,
    interval: float = 30.0,
    timeout: float = 900.0,
    version: str = "rapid",
) -> dict:
    """
    仅在 WAIT/RUN 继续轮询；DONE 返回；FAIL/ERROR/CANCELLED 立即抛出
    """
    start = time.time()
    attempt = 0
    while True:
        attempt += 1
        elapsed = time.time() - start
        if elapsed > timeout:
            raise TimeoutError(f"任务超时({timeout:.0f}s) | 格式={result_format} | JobId={job_id}")

        try:
            job_info = query_job(client, job_id, version)
        except TencentCloudSDKException as e:
            logging.warning(f"[{job_id}] 查询SDK异常: {e} | {interval:.0f}s后重试")
            time.sleep(interval)
            continue

        body = job_info.get("Response") or job_info
        status = (body.get("Status") or "UNKNOWN").upper()
        logging.info(f"[{job_id}] 第{attempt}次查询 | {version} | 状态={status} | 耗时={int(elapsed)}s")

        if status in {"DONE", "SUCCEED", "SUCCESS", "FINISHED"}:
            return body
        if status in {"FAIL", "FAILED", "ERROR", "CANCELLED"}:
            reason = body.get("ErrorMessage") or body.get("Message") or body.get("ErrorCode") or "未知原因"
            raise RuntimeError(f"任务失败 | 格式={result_format} | 原因={reason}")

        time.sleep(interval)


def download_single_format_results(
    session: requests.Session,
    job_info: dict,
    result_format: str,
    output_dir: str,
    base_name: str,
) -> List[str]:
    ensure_dir(output_dir)
    saved_paths: List[str] = []

    body = job_info.get("Response") or job_info
    files = body.get("ResultFile3Ds") or []
    if not files:
        logging.warning(f"格式{result_format}没有结果文件")
        return saved_paths

    want = result_format.upper()
    for info in files:
        ftype = (info.get("Type") or want).upper()
        url = info.get("Url")
        if not url:
            logging.warning("某结果无Url，跳过")
            continue
        # 若API返回多种类型，优先匹配Type==目标格式；否则也下载
        if ftype != want and any(x.get("Type", "").upper() == want for x in files):
            continue

        path_in_url = urlparse(url).path
        ext = os.path.splitext(path_in_url)[1].lower() or f".{ftype.lower()}"
        out = os.path.join(output_dir, f"{base_name}_{ftype}{ext}")
        try:
            logging.info(f"下载 | {url} -> {out}")
            with session.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                with open(out, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
            saved_paths.append(out)
        except Exception as e:
            logging.error(f"下载失败 | 文件={out} | 原因={str(e)}")
            if os.path.exists(out):
                os.remove(out)
    return saved_paths


def download_single_format_previews(
    session: requests.Session,
    job_info: dict,
    result_format: str,
    output_dir: str,
    base_name: str,
) -> List[str]:
    ensure_dir(output_dir)
    saved_paths: List[str] = []

    body = job_info.get("Response") or job_info
    files = body.get("ResultFile3Ds") or []
    if not files:
        logging.warning(f"格式{result_format}没有结果文件，无法下载预览图")
        return saved_paths

    want = result_format.upper()
    for idx, info in enumerate(files, 1):
        ftype = (info.get("Type") or want).upper()
        url = info.get("PreviewImageUrl")
        if not url:
            logging.warning(f"格式{ftype}没有PreviewImageUrl，跳过预览图")
            continue
        # 若API返回多种类型，优先匹配Type==目标格式；否则也下载
        if ftype != want and any(x.get("Type", "").upper() == want for x in files):
            continue

        path_in_url = urlparse(url).path
        ext = os.path.splitext(path_in_url)[1].lower() or ".png"
        suffix = f"{base_name}_{ftype}_preview"
        out = os.path.join(output_dir, f"{suffix}{ext}")
        if out in saved_paths:
            out = os.path.join(output_dir, f"{suffix}_{idx}{ext}")
        try:
            logging.info(f"下载预览图 | {url} -> {out}")
            with session.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                with open(out, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
            saved_paths.append(out)
        except Exception as e:
            logging.error(f"预览图下载失败 | 文件={out} | 原因={str(e)}")
            if os.path.exists(out):
                os.remove(out)
    return saved_paths


# =========================
# 主流程（单视角，多格式）
# =========================
def process_single_view_with_multi_formats(
    client: ai3d_client.Ai3dClient,
    session: requests.Session,
    image_path: Optional[str],
    output_root: str,
    result_formats: List[str],
    enable_pbr: bool,
    poll_interval: float,
    poll_timeout: float,
    version: str,
    prompt: Optional[str] = None,
    image_url: Optional[str] = None,
    face_count: Optional[int] = None,
    generate_type: Optional[str] = None,
    resubmit: int = 1,
    resubmit_backoff: float = 30.0,
    download_preview: bool = False,
):
    img_name = os.path.basename(image_path) if image_path else (image_url or prompt or "job")
    stem = os.path.splitext(img_name)[0]
    out_dir = os.path.join(output_root, stem)
    ensure_dir(out_dir)

    for fmt in result_formats:
        logging.info(f"\n开始处理格式: {fmt} | 输入={img_name}")
        attempt = 0
        while True:
            attempt += 1
            try:
                if version == "rapid":
                    payload = build_payload_single_rapid(
                        image_path=image_path,
                        result_format=fmt,
                        enable_pbr=enable_pbr,
                        prompt=prompt,
                        image_url=image_url,
                    )
                else:
                    payload = build_payload_single_pro(
                        image_path=image_path,
                        result_format=fmt,
                        enable_pbr=enable_pbr,
                        prompt=prompt,
                        image_url=image_url,
                        face_count=face_count,
                        generate_type=generate_type,
                    )

                job_id = submit_job_with_retry(client, payload, fmt, version)
                job_info = poll_until_done(
                    client, job_id, fmt, poll_interval, poll_timeout, version
                )
                saved = download_single_format_results(session, job_info, fmt, out_dir, stem)
                preview_saved = []
                if download_preview:
                    preview_saved = download_single_format_previews(
                        session, job_info, fmt, out_dir, stem
                    )
                if saved or preview_saved:
                    logging.info(
                        f"完成 | {fmt} | 保存模型{len(saved)}个 | 预览图{len(preview_saved)}个 | 目录={out_dir}"
                    )
                else:
                    logging.warning(f"完成但无可下载文件 | {fmt}")
                break  # 当前格式成功，跳出resubmit循环

            except RuntimeError as e:
                msg = str(e)
                # 内部错误时尝试重提
                if ("服务内部错误" in msg or "Internal" in msg) and attempt <= resubmit + 1:
                    logging.warning(f"{fmt} 失败原因：{msg} | 将在 {resubmit_backoff:.0f}s 后重提（第{attempt-1}/{resubmit}次重提）")
                    time.sleep(resubmit_backoff)
                    continue
                logging.error(f"{fmt} 最终失败：{msg}")
                break  # 不再重提，进入下一格式

            except Exception as e:
                logging.error(f"{fmt} 异常：{str(e)}", exc_info=True)
                break  # 进入下一格式

# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", default="./data/", help="输入目录：放单视角图片（jpg/png/jpeg/webp）")
    parser.add_argument("--output", "-o", default="./downloads/", help="输出目录")
    parser.add_argument("--region", default="ap-guangzhou", help="腾讯云区域")
    parser.add_argument("--endpoint", default="ai3d.tencentcloudapi.com", help="API 端点")
    parser.add_argument("--check-key", action="store_true",
                        help="仅校验腾讯云密钥是否有效，不提交混元3D任务")
    parser.add_argument("--sts-endpoint", default="sts.tencentcloudapi.com",
                        help="密钥校验使用的 STS API 端点")
    parser.add_argument("--result-formats", "-f", nargs="+", default=["GLB"],
                        choices=["GLB", "OBJ", "FBX", "STL", "USDZ", "MP4"],
                        help="输出格式（多格式会分别提交多个任务）")
    parser.add_argument("--no-pbr", action="store_false", dest="pbr", default=True,
                        help="禁用PBR材质（默认启用）")
    parser.add_argument("--interval", type=float, default=30.0, help="任务查询间隔（秒）")
    parser.add_argument("--timeout", type=float, default=900.0, help="单任务超时时间（秒）")
    parser.add_argument("--retry-interval", type=int, default=60, help="提交失败时的重试间隔（秒）")
    parser.add_argument("--max-retry", type=int, default=10, help="提交失败的最大重试次数")
    parser.add_argument("--http-retries", type=int, default=5, help="HTTP下载重试次数（requests层）")
    parser.add_argument("--version", choices=["rapid", "pro"], default="pro",
                        help="选择 API 版本: rapid（极速版） / pro（专业版）")
    # Pro 版可选参数
    parser.add_argument("--face-count", type=int, default=None,
                        help="专业版面数（示例：150000）")
    parser.add_argument("--gen-type", type=str, default=None,
                        help="专业版生成类型（如：Normal/LowPoly/Geometry/Sketch）")
    # Prompt / 图片URL（可替代本地图片）
    parser.add_argument("--prompt", type=str, default=None,
                        help="文生3D提示词（提供后忽略本地图片）")
    parser.add_argument("--image-url", type=str, default=None,
                        help="图片URL（提供后忽略本地图片）")
    parser.add_argument("--download-preview", action="store_true",
                        help="下载腾讯云返回的PreviewImageUrl预览图")
    # 失败后自动重提
    parser.add_argument("--resubmit", type=int, default=1,
                        help="遇到服务内部错误时的自动重提交次数（默认1）")
    parser.add_argument("--resubmit-backoff", type=float, default=30.0,
                        help="自动重提交的退避秒数（默认30）")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="日志详细程度: -v=INFO, -vv=DEBUG")

    args = parser.parse_args()
    setup_logger(args.verbose)

    if args.check_key:
        try:
            identity = validate_credentials(args.region, args.sts_endpoint)
            result = {
                "valid": True,
                "account_id": identity.get("AccountId"),
                "principal_id": identity.get("PrincipalId"),
                "arn": identity.get("Arn"),
            }
            print(json.dumps(result, ensure_ascii=False))
            return
        except TencentCloudSDKException as e:
            result = {
                "valid": False,
                "code": get_sdk_error_code(e),
                "message": get_sdk_error_message(e),
            }
            print(json.dumps(result, ensure_ascii=False))
            sys.exit(1)
        except SystemExit as e:
            result = {
                "valid": False,
                "code": "MissingCredentials",
                "message": "缺少 TENCENTCLOUD_SECRET_ID 或 TENCENTCLOUD_SECRET_KEY 环境变量",
            }
            print(json.dumps(result, ensure_ascii=False))
            sys.exit(e.code or 2)
        except Exception as e:
            result = {
                "valid": False,
                "code": type(e).__name__,
                "message": str(e),
            }
            print(json.dumps(result, ensure_ascii=False))
            sys.exit(2)

    # 初始化
    try:
        client = init_client(args.region, args.endpoint)
        session = build_requests_session(total_retries=args.http_retries, backoff=0.5)
        logging.info(f"\n腾讯云客户端初始化成功 | 区域={args.region}")
    except Exception as e:
        logging.error(f"初始化失败: {str(e)}")
        sys.exit(2)

    # 输入来源：prompt/image-url 或 本地目录
    ensure_dir(args.output)
    if args.prompt or args.image_url:
        logging.info("\n单任务模式（Prompt/ImageUrl）启动")
        process_single_view_with_multi_formats(
            client=client,
            session=session,
            image_path=None,
            output_root=args.output,
            result_formats=[fmt.upper() for fmt in args.result_formats],
            enable_pbr=bool(args.pbr),
            poll_interval=float(args.interval),
            poll_timeout=float(args.timeout),
            version=args.version,
            prompt=args.prompt,
            image_url=args.image_url,
            face_count=args.face_count,
            generate_type=args.gen_type,
            resubmit=args.resubmit,
            resubmit_backoff=float(args.resubmit_backoff),
            download_preview=bool(args.download_preview),
        )
        logging.info(f"\n任务处理完毕！最终结果目录: {args.output}")
        return

    # 批量模式：本地目录
    if not os.path.isdir(args.input):
        logging.error(f"输入路径不存在: {args.input}")
        sys.exit(2)

    images = [
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if os.path.isfile(os.path.join(args.input, f)) and is_image_file(os.path.join(args.input, f))
    ]
    if not images:
        logging.warning(f"输入目录 {args.input} 下无有效图片")
        return

    logging.info(f"\n批量模式启动 | 发现{len(images)}张有效图片")
    for idx, img_path in enumerate(sorted(images), 1):
        logging.info(f"\n=== 开始处理单视角图片 {idx}/{len(images)} ===")
        process_single_view_with_multi_formats(
            client=client,
            session=session,
            image_path=img_path,
            output_root=args.output,
            result_formats=[fmt.upper() for fmt in args.result_formats],
            enable_pbr=bool(args.pbr),
            poll_interval=float(args.interval),
            poll_timeout=float(args.timeout),
            version=args.version,
            prompt=args.prompt,
            image_url=args.image_url,
            face_count=args.face_count,
            generate_type=args.gen_type,
            resubmit=args.resubmit,
            resubmit_backoff=float(args.resubmit_backoff),
            download_preview=bool(args.download_preview),
        )

    logging.info(f"\n所有任务处理完毕！最终结果目录: {args.output}")

if __name__ == "__main__":
    main()
