from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .exceptions import HunyuanApiError


def _sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hmac_sha256(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


@dataclass(frozen=True)
class TencentApiCredentials:
    secret_id: str
    secret_key: str
    token: str | None = None


def _require_ascii_credential(name: str, value: str | None) -> None:
    if value is None:
        return
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise HunyuanApiError(
            f"{name} must contain only ASCII characters. "
            "It looks like the environment variable may still contain a placeholder instead of the real Tencent Cloud credential."
        ) from exc


class TencentCloudApiClient:
    """Small TC3-HMAC-SHA256 JSON client for Tencent Cloud API 3.0."""

    def __init__(
        self,
        *,
        endpoint: str,
        service: str,
        version: str,
        region: str,
        credentials: TencentApiCredentials,
        timeout_seconds: int = 60,
    ) -> None:
        self.endpoint = endpoint
        self.service = service
        self.version = version
        self.region = region
        self.credentials = credentials
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TencentCloudApiClient":
        hunyuan = config.get("hunyuan", {})
        secret_id = os.environ.get(str(hunyuan.get("secret_id_env") or "TENCENTCLOUD_SECRET_ID"))
        secret_key = os.environ.get(str(hunyuan.get("secret_key_env") or "TENCENTCLOUD_SECRET_KEY"))
        token_env = str(hunyuan.get("token_env") or "TENCENTCLOUD_TOKEN")
        token = os.environ.get(token_env)
        if not secret_id or not secret_key:
            raise HunyuanApiError(
                "Missing Tencent Cloud credentials. Set "
                f"{hunyuan.get('secret_id_env') or 'TENCENTCLOUD_SECRET_ID'} and "
                f"{hunyuan.get('secret_key_env') or 'TENCENTCLOUD_SECRET_KEY'}."
            )
        _require_ascii_credential(str(hunyuan.get("secret_id_env") or "TENCENTCLOUD_SECRET_ID"), secret_id)
        _require_ascii_credential(str(hunyuan.get("secret_key_env") or "TENCENTCLOUD_SECRET_KEY"), secret_key)
        _require_ascii_credential(token_env, token)
        return cls(
            endpoint=str(hunyuan.get("endpoint") or "ai3d.tencentcloudapi.com"),
            service=str(hunyuan.get("service") or "ai3d"),
            version=str(hunyuan.get("version") or "2025-05-13"),
            region=str(hunyuan.get("region") or "ap-guangzhou"),
            credentials=TencentApiCredentials(secret_id=secret_id, secret_key=secret_key, token=token),
        )

    def call(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        timestamp = int(time.time())
        date = dt.datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")
        credential_scope = f"{date}/{self.service}/tc3_request"

        canonical_headers = (
            "content-type:application/json; charset=utf-8\n"
            f"host:{self.endpoint}\n"
        )
        signed_headers = "content-type;host"
        canonical_request = "\n".join(
            [
                "POST",
                "/",
                "",
                canonical_headers,
                signed_headers,
                _sha256_hex(body),
            ]
        )
        string_to_sign = "\n".join(
            [
                "TC3-HMAC-SHA256",
                str(timestamp),
                credential_scope,
                _sha256_hex(canonical_request.encode("utf-8")),
            ]
        )

        secret_date = _hmac_sha256(("TC3" + self.credentials.secret_key).encode("utf-8"), date)
        secret_service = _hmac_sha256(secret_date, self.service)
        secret_signing = _hmac_sha256(secret_service, "tc3_request")
        signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
        authorization = (
            "TC3-HMAC-SHA256 "
            f"Credential={self.credentials.secret_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )

        headers = {
            "Authorization": authorization,
            "Content-Type": "application/json; charset=utf-8",
            "Host": self.endpoint,
            "X-TC-Action": action,
            "X-TC-Timestamp": str(timestamp),
            "X-TC-Version": self.version,
            "X-TC-Region": self.region,
        }
        if self.credentials.token:
            headers["X-TC-Token"] = self.credentials.token

        request = urllib.request.Request(f"https://{self.endpoint}/", data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise HunyuanApiError(f"Tencent API HTTP {exc.code} for {action}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise HunyuanApiError(f"Tencent API request failed for {action}: {exc}") from exc

        data = json.loads(raw)
        response = data.get("Response", {})
        if "Error" in response:
            error = response["Error"]
            raise HunyuanApiError(f"Tencent API {action} error {error.get('Code')}: {error.get('Message')}")
        return response
