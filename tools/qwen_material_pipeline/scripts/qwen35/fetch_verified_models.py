#!/usr/bin/env python3
"""Fetch and verify the pinned Qwen3.5/SigLIP2 runtime checkpoints.

The Hugging Face repository/revision is the canonical identity used by the
pipeline.  ModelScope is only a transport fallback for networks where the
Hugging Face/Xet connection is unreliable.  A fallback is accepted only when
every runtime-required file matches the pinned size and SHA256 manifest below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    key: str
    huggingface_repository: str
    huggingface_revision: str
    modelscope_repository: str
    modelscope_revision: str
    identity_schema: str
    files: dict[str, tuple[int, str]]


MODEL_SPECS = {
    "qwen3_5_4b": ModelSpec(
        key="qwen3_5_4b",
        huggingface_repository="Qwen/Qwen3.5-4B",
        huggingface_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        modelscope_repository="Qwen/Qwen3.5-4B",
        modelscope_revision="fcb1a040bb418b0b8add6f6f6c475386abc2cb97",
        identity_schema="qwen-local-checkpoint/v1",
        files={
            "chat_template.jinja": (
                6669,
                "04b007131663760bf3e581e5a953be77044014e87efe1d2a6ca4b72ec0eac978",
            ),
            "config.json": (
                3161,
                "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
            ),
            "merges.txt": (
                3353259,
                "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
            ),
            "model.safetensors-00001-of-00002.safetensors": (
                5329398688,
                "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
            ),
            "model.safetensors-00002-of-00002.safetensors": (
                3990429408,
                "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
            ),
            "model.safetensors.index.json": (
                76196,
                "cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d",
            ),
            "preprocessor_config.json": (
                390,
                "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
            ),
            "tokenizer.json": (
                12807982,
                "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            ),
            "tokenizer_config.json": (
                15602,
                "aefe82b6d1bfb01fcb2a889986fb99e4cec84ee84327116887d883d0b9e7f1e6",
            ),
            "video_preprocessor_config.json": (
                387,
                "a4313a685593ef3947edc895450bcb8b06c19828ed9686fbf995dd26c5fb9f7f",
            ),
            "vocab.json": (
                6722759,
                "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
            ),
        },
    ),
    "siglip2_base": ModelSpec(
        key="siglip2_base",
        huggingface_repository="google/siglip2-base-patch16-224",
        huggingface_revision="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        modelscope_repository="google/siglip2-base-patch16-224",
        modelscope_revision="153facb840ceec16916cc7ec0e3bf8757d95f7e3",
        identity_schema="retrieval-local-checkpoint/v1",
        files={
            "config.json": (
                253,
                "fe8b5fe6d5734360678fd71c11c21e1ea3364bd8598d34295d9206335973ffd7",
            ),
            "model.safetensors": (
                1500800904,
                "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b",
            ),
            "preprocessor_config.json": (
                394,
                "9b36b57ebaf20f09bf4c22100ccc21877ea6bfe5aead0c00c59f8af8ccefacfc",
            ),
            "special_tokens_map.json": (
                636,
                "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351",
            ),
            "tokenizer.json": (
                34363039,
                "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322",
            ),
            "tokenizer.model": (
                4241003,
                "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
            ),
            "tokenizer_config.json": (
                47164,
                "14afe629fe4959b9e0d51e1852b8d9f7ad074f90a1a7125a4fcdd17f06e78fc8",
            ),
        },
    ),
}


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_payload(spec: ModelSpec) -> list[dict[str, Any]]:
    return [
        {"path": path, "bytes": size, "sha256": sha256}
        for path, (size, sha256) in sorted(spec.files.items())
    ]


def _verify_destination(
    spec: ModelSpec,
    destination: Path,
    *,
    report_progress: bool,
) -> list[str]:
    errors: list[str] = []
    candidates: list[tuple[str, Path, int, str]] = []
    for relative, (expected_size, expected_sha256) in sorted(spec.files.items()):
        path = destination / relative
        if not path.is_file():
            errors.append(f"missing {relative}")
            continue
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            errors.append(
                f"size mismatch {relative}: expected={expected_size}, "
                f"actual={actual_size}"
            )
            continue
        candidates.append((relative, path, expected_size, expected_sha256))

    for index, (relative, path, _size, expected_sha256) in enumerate(
        candidates, start=1
    ):
        if report_progress:
            print(
                f"[verify {index}/{len(candidates)}] {spec.key}: {relative}",
                flush=True,
            )
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            errors.append(
                f"sha256 mismatch {relative}: expected={expected_sha256}, "
                f"actual={actual_sha256}"
            )
    return errors


def _download_huggingface(
    spec: ModelSpec,
    destination: Path,
    *,
    max_workers: int,
) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=spec.huggingface_repository,
        revision=spec.huggingface_revision,
        local_dir=destination,
        allow_patterns=sorted(spec.files),
        max_workers=max_workers,
    )


def _download_modelscope(
    spec: ModelSpec,
    destination: Path,
    *,
    max_workers: int,
) -> None:
    del max_workers
    download_environment = os.environ.copy()
    for proxy_variable in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        download_environment.pop(proxy_variable, None)
    direct_hosts = "modelscope.cn,.modelscope.cn"
    download_environment["NO_PROXY"] = direct_hosts
    download_environment["no_proxy"] = direct_hosts
    total = len(spec.files)
    for index, (relative, (expected_size, expected_sha256)) in enumerate(
        sorted(spec.files.items()), start=1
    ):
        output = destination / relative
        if (
            output.is_file()
            and output.stat().st_size == expected_size
            and _sha256_file(output) == expected_sha256
        ):
            print(
                f"[download {index}/{total}] {spec.key}: {relative} already verified",
                flush=True,
            )
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(f"{output.name}.incomplete")
        url = (
            "https://modelscope.cn/models/"
            f"{spec.modelscope_repository}/resolve/"
            f"{spec.modelscope_revision}/{relative}"
        )
        print(
            f"[download {index}/{total}] {spec.key}: {relative} "
            f"({expected_size / (1024**3):.3f} GiB)",
            flush=True,
        )
        aria2 = shutil.which("aria2c")
        if aria2 is not None and expected_size >= 1024 * 1024:
            subprocess.run(
                [
                    aria2,
                    "--allow-overwrite=true",
                    "--auto-file-renaming=false",
                    "--continue=true",
                    "--file-allocation=none",
                    "--max-connection-per-server=16",
                    "--max-tries=10",
                    "--min-split-size=1M",
                    (
                        "--no-proxy=modelscope.cn,cdn-lfs-cn-1.modelscope.cn,"
                        "www.modelscope.cn"
                    ),
                    "--retry-wait=2",
                    "--split=16",
                    "--summary-interval=10",
                    "--timeout=60",
                    f"--dir={partial.parent}",
                    f"--out={partial.name}",
                    url,
                ],
                check=True,
                env=download_environment,
            )
        else:
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--retry",
                    "10",
                    "--retry-all-errors",
                    "--connect-timeout",
                    "30",
                    "--noproxy",
                    "*",
                    "--continue-at",
                    "-",
                    "--output",
                    str(partial),
                    url,
                ],
                check=True,
                env=download_environment,
            )
        actual_size = partial.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"ModelScope size mismatch for {relative}: "
                f"expected={expected_size}, actual={actual_size}"
            )
        actual_sha256 = _sha256_file(partial)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"ModelScope SHA256 mismatch for {relative}: "
                f"expected={expected_sha256}, actual={actual_sha256}"
            )
        partial.replace(output)


def _write_identity(spec: ModelSpec, destination: Path, *, transport: str) -> Path:
    manifest = _manifest_payload(spec)
    config_path = destination / "config.json"
    identity = {
        "schema_version": spec.identity_schema,
        "repository": spec.huggingface_repository,
        "revision": spec.huggingface_revision,
        "config_sha256": _sha256_file(config_path),
        "content_manifest_sha256": _canonical_sha256(manifest),
        "runtime_files": manifest,
        "transport": {
            "source": transport,
            "content_verification": "pinned_size_and_sha256_manifest",
            "modelscope_repository": spec.modelscope_repository,
            "modelscope_revision": spec.modelscope_revision,
        },
    }
    output = destination / "checkpoint_identity.json"
    temporary = output.with_name(f"{output.name}.incomplete-{os.getpid()}")
    temporary.write_text(
        json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def _identity_errors(spec: ModelSpec, destination: Path) -> list[str]:
    identity_path = destination / "checkpoint_identity.json"
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ["missing checkpoint_identity.json"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"invalid checkpoint_identity.json: {exc}"]
    manifest = _manifest_payload(spec)
    expected = {
        "schema_version": spec.identity_schema,
        "repository": spec.huggingface_repository,
        "revision": spec.huggingface_revision,
        "config_sha256": spec.files["config.json"][1],
        "content_manifest_sha256": _canonical_sha256(manifest),
        "runtime_files": manifest,
    }
    return [
        f"checkpoint identity mismatch for {field}"
        for field, expected_value in expected.items()
        if identity.get(field) != expected_value
    ]


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--source",
        choices=("huggingface", "modelscope", "verify"),
        required=True,
    )
    parser.add_argument("--max-workers", type=_positive_int, default=2)
    parser.add_argument("--attempts", type=_positive_int, default=1)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    spec = MODEL_SPECS[args.model]
    destination = args.destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    errors = _verify_destination(spec, destination, report_progress=False)
    if not errors:
        if args.source == "verify":
            identity_errors = _identity_errors(spec, destination)
            if identity_errors:
                print(
                    f"{spec.key}: checkpoint identity verification failed:",
                    file=sys.stderr,
                )
                for error in identity_errors:
                    print(f"  - {error}", file=sys.stderr)
                return 2
            print(f"{spec.key}: content and identity verified", flush=True)
            return 0
        identity = _write_identity(
            spec,
            destination,
            transport="preexisting_verified_pinned_content",
        )
        print(f"{spec.key}: already complete and verified ({identity})", flush=True)
        return 0
    if args.source == "verify":
        print(f"{spec.key}: checkpoint verification failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2

    download = (
        _download_huggingface
        if args.source == "huggingface"
        else _download_modelscope
    )
    last_error: Exception | None = None
    for attempt in range(1, args.attempts + 1):
        print(
            f"{spec.key}: {args.source} download attempt "
            f"{attempt}/{args.attempts}",
            flush=True,
        )
        try:
            download(spec, destination, max_workers=args.max_workers)
            last_error = None
            break
        except Exception as exc:  # downloader exceptions vary by package/version
            last_error = exc
            print(
                f"{spec.key}: {args.source} attempt {attempt} failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if attempt < args.attempts:
                time.sleep(min(20, 2**attempt))
    if last_error is not None:
        return 3

    errors = _verify_destination(spec, destination, report_progress=True)
    if errors:
        print(
            f"{spec.key}: downloaded content failed the pinned manifest:",
            file=sys.stderr,
        )
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 4
    identity = _write_identity(spec, destination, transport=args.source)
    print(f"{spec.key}: complete, verified, identity={identity}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
