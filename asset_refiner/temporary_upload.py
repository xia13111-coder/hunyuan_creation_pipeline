from __future__ import annotations

import json
import mimetypes
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .exceptions import HunyuanApiError


def _multipart_body(path: Path, field_name: str) -> tuple[bytes, str]:
    boundary = f"asset-refiner-{uuid.uuid4().hex}"
    filename = path.name
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return head + path.read_bytes() + tail, boundary


def upload_to_uguu(path: str | Path, config: dict[str, Any]) -> str:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Temporary upload input file does not exist: {file_path}")

    upload_cfg = config.get("hunyuan", {}).get("temp_upload", {})
    endpoint = str(upload_cfg.get("endpoint") or "https://uguu.se/upload.php")
    timeout = int(upload_cfg.get("timeout_seconds") or 300)

    body, boundary = _multipart_body(file_path, "files[]")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "asset-refiner/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HunyuanApiError(f"Temporary upload HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise HunyuanApiError(f"Temporary upload failed: {exc}") from exc

    data = json.loads(raw)
    if not data.get("success") or not data.get("files"):
        raise HunyuanApiError(f"Temporary upload did not return a file URL: {raw[:500]}")
    url = data["files"][0].get("url")
    if not url:
        raise HunyuanApiError(f"Temporary upload response missing URL: {raw[:500]}")
    return str(url)


def upload_to_temporary_host(path: str | Path, config: dict[str, Any]) -> str:
    provider = str(config.get("hunyuan", {}).get("temp_upload", {}).get("provider") or "uguu").lower()
    if provider != "uguu":
        raise HunyuanApiError(f"Unsupported temporary upload provider: {provider}")
    return upload_to_uguu(path, config)
