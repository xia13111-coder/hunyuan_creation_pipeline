from __future__ import annotations

import json
import ssl
from typing import Any
from urllib import error

import pytest

from qwen_material_pipeline.qwen.client import QwenClientError
import qwen_material_pipeline.qwen.remote_vl as remote_vl
from qwen_material_pipeline.qwen.remote_vl import (
    CONNECT_IP_ENV,
    OpenAICompatibleVisionRunner,
)


class _Response:
    def __init__(self, document: dict[str, Any]) -> None:
        self._payload = json.dumps(document).encode("utf-8")
        self.closed = False

    def read(self) -> bytes:
        return self._payload

    def close(self) -> None:
        self.closed = True


def test_remote_runner_requires_https() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        OpenAICompatibleVisionRunner(
            base_url="http://example.invalid/v1",
            model="gpt-5.6",
            api_key_env="TEST_REMOTE_KEY",
        )


def test_remote_runner_requires_only_named_environment_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEST_REMOTE_KEY", raising=False)
    runner = OpenAICompatibleVisionRunner(
        base_url="https://example.invalid/v1",
        model="gpt-5.6",
        api_key_env="TEST_REMOTE_KEY",
    )

    with pytest.raises(QwenClientError, match="TEST_REMOTE_KEY"):
        runner.preflight()
    assert "TEST_REMOTE_KEY" in json.dumps(runner.model_identity)


def test_remote_runner_translates_staged_chat_payload_without_persisting_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "not-a-real-secret"
    monkeypatch.setenv("TEST_REMOTE_KEY", secret)
    captured: dict[str, Any] = {}
    response = _Response(
        {
            "choices": [
                {
                    "message": {"content": '{"schema_version":"ok"}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 17},
        }
    )

    def opener(http_request: Any, *, timeout: float) -> _Response:
        captured["url"] = http_request.full_url
        captured["authorization"] = http_request.get_header("Authorization")
        captured["body"] = json.loads(http_request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return response

    runner = OpenAICompatibleVisionRunner(
        base_url="https://example.invalid/v1",
        model="gpt-5.6",
        api_key_env="TEST_REMOTE_KEY",
        reasoning_effort="high",
        timeout_seconds=27,
        max_new_tokens=800,
        opener=opener,
    )
    result = runner.generate_with_metadata(
        {
            "model": "legacy-local-name",
            "messages": [{"role": "user", "content": "return JSON"}],
            "response_format": {"type": "json_object"},
            "enable_thinking": False,
            "temperature": 0,
        },
        max_new_tokens=900,
    )

    assert result.text == '{"schema_version":"ok"}'
    assert result.generated_tokens == 17
    assert result.max_new_tokens == 900
    assert result.hit_token_limit is False
    assert captured["url"] == "https://example.invalid/v1/chat/completions"
    assert captured["authorization"] == f"Bearer {secret}"
    assert captured["timeout"] == 27
    assert captured["body"]["model"] == "gpt-5.6"
    assert captured["body"]["reasoning_effort"] == "high"
    assert captured["body"]["max_completion_tokens"] == 900
    assert "enable_thinking" not in captured["body"]
    assert "temperature" not in captured["body"]
    assert "top_p" not in captured["body"]
    assert secret not in json.dumps(runner.model_identity)
    assert response.closed is True


def test_remote_runner_retries_tls_handshake_without_rebuilding_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_REMOTE_KEY", "not-a-real-secret")
    attempts: list[Any] = []
    sleeps: list[float] = []
    response = _Response(
        {
            "choices": [
                {
                    "message": {"content": '{"schema_version":"ok"}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 3},
        }
    )

    def opener(http_request: Any, *, timeout: float) -> _Response:
        attempts.append(http_request)
        if len(attempts) < 3:
            raise error.URLError(
                ssl.SSLEOFError(8, "unexpected EOF during handshake")
            )
        return response

    runner = OpenAICompatibleVisionRunner(
        base_url="https://example.invalid/v1",
        model="gpt-5.6",
        api_key_env="TEST_REMOTE_KEY",
        opener=opener,
        sleeper=sleeps.append,
    )
    result = runner.generate_with_metadata(
        {"messages": [{"role": "user", "content": "return JSON"}]}
    )

    assert result.text == '{"schema_version":"ok"}'
    assert len(attempts) == 3
    assert attempts[0] is attempts[1] is attempts[2]
    assert sleeps == [1.0, 2.0]


def test_remote_runner_does_not_retry_non_transport_url_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_REMOTE_KEY", "not-a-real-secret")
    attempts = 0

    def opener(http_request: Any, *, timeout: float) -> _Response:
        nonlocal attempts
        attempts += 1
        raise error.URLError("non-connection policy failure")

    runner = OpenAICompatibleVisionRunner(
        base_url="https://example.invalid/v1",
        model="gpt-5.6",
        api_key_env="TEST_REMOTE_KEY",
        opener=opener,
        sleeper=lambda _seconds: pytest.fail("must not sleep"),
    )

    with pytest.raises(QwenClientError, match="1 transport attempt"):
        runner.generate_with_metadata(
            {"messages": [{"role": "user", "content": "return JSON"}]}
        )
    assert attempts == 1


def test_remote_runner_retries_direct_connection_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_REMOTE_KEY", "not-a-real-secret")
    attempts = 0
    sleeps: list[float] = []
    response = _Response(
        {
            "choices": [
                {
                    "message": {"content": '{"schema_version":"ok"}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 3},
        }
    )

    def opener(http_request: Any, *, timeout: float) -> _Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionResetError(104, "connection reset by peer")
        return response

    runner = OpenAICompatibleVisionRunner(
        base_url="https://example.invalid/v1",
        model="gpt-5.6",
        api_key_env="TEST_REMOTE_KEY",
        opener=opener,
        sleeper=sleeps.append,
    )

    result = runner.generate_with_metadata(
        {"messages": [{"role": "user", "content": "return JSON"}]}
    )
    assert result.text == '{"schema_version":"ok"}'
    assert attempts == 2
    assert sleeps == [1.0]


def test_remote_runner_retries_direct_tls_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_REMOTE_KEY", "not-a-real-secret")
    attempts = 0
    sleeps: list[float] = []

    def opener(http_request: Any, *, timeout: float) -> _Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ssl.SSLEOFError(8, "unexpected EOF during handshake")
        return _Response(
            {
                "choices": [
                    {
                        "message": {"content": '{"schema_version":"ok"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 3},
            }
        )

    runner = OpenAICompatibleVisionRunner(
        base_url="https://example.invalid/v1",
        model="gpt-5.6-sol",
        api_key_env="TEST_REMOTE_KEY",
        opener=opener,
        sleeper=sleeps.append,
    )
    result = runner.generate_with_metadata(
        {"messages": [{"role": "user", "content": "return JSON"}]}
    )

    assert result.text == '{"schema_version":"ok"}'
    assert attempts == 2
    assert sleeps == [1.0]


def test_remote_runner_uses_optional_direct_ip_without_changing_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_REMOTE_KEY", "not-a-real-secret")
    monkeypatch.setenv(CONNECT_IP_ENV, "203.0.113.10")
    captured: dict[str, Any] = {}

    def direct_open(
        http_request: Any,
        *,
        timeout: float,
        connect_ip: str,
    ) -> _Response:
        captured["url"] = http_request.full_url
        captured["timeout"] = timeout
        captured["connect_ip"] = connect_ip
        return _Response(
            {
                "choices": [
                    {
                        "message": {"content": '{"schema_version":"ok"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 3},
            }
        )

    monkeypatch.setattr(remote_vl, "_direct_ip_urlopen", direct_open)
    runner = OpenAICompatibleVisionRunner(
        base_url="https://example.invalid/v1",
        model="gpt-5.6-sol",
        api_key_env="TEST_REMOTE_KEY",
        timeout_seconds=31,
    )
    result = runner.generate_with_metadata(
        {"messages": [{"role": "user", "content": "return JSON"}]}
    )

    assert result.text == '{"schema_version":"ok"}'
    assert captured == {
        "url": "https://example.invalid/v1/chat/completions",
        "timeout": 31.0,
        "connect_ip": "203.0.113.10",
    }
    assert CONNECT_IP_ENV not in json.dumps(runner.model_identity)


def test_remote_runner_refreshes_fake_ip_route_after_tls_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_REMOTE_KEY", "not-a-real-secret")
    monkeypatch.delenv(CONNECT_IP_ENV, raising=False)
    for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(remote_vl, "_hostname_uses_fake_ip", lambda _host: True)
    resolved = iter(("203.0.113.10", "203.0.113.11"))
    resolve_calls: list[str] = []
    direct_calls: list[str] = []
    sleeps: list[float] = []

    def resolve(hostname: str, *, timeout: float) -> str:
        del timeout
        resolve_calls.append(hostname)
        return next(resolved)

    def direct_open(
        http_request: Any,
        *,
        timeout: float,
        connect_ip: str,
    ) -> _Response:
        del http_request, timeout
        direct_calls.append(connect_ip)
        if len(direct_calls) == 1:
            raise ssl.SSLEOFError(8, "stale gateway route")
        return _Response(
            {
                "choices": [
                    {
                        "message": {"content": '{"schema_version":"ok"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 3},
            }
        )

    monkeypatch.setattr(remote_vl, "_resolve_public_ipv4_doh", resolve)
    monkeypatch.setattr(remote_vl, "_direct_ip_urlopen", direct_open)
    runner = OpenAICompatibleVisionRunner(
        base_url="https://api.example.invalid/v1",
        model="gpt-5.6-terra",
        api_key_env="TEST_REMOTE_KEY",
        sleeper=sleeps.append,
    )
    result = runner.generate_with_metadata(
        {"messages": [{"role": "user", "content": "return JSON"}]}
    )

    assert result.text == '{"schema_version":"ok"}'
    assert resolve_calls == [
        "api.example.invalid",
        "api.example.invalid",
    ]
    assert direct_calls == ["203.0.113.10", "203.0.113.11"]
    assert sleeps == [1.0]
    assert CONNECT_IP_ENV not in json.dumps(runner.model_identity)


def test_remote_runner_prefers_configured_proxy_over_fake_ip_direct_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_REMOTE_KEY", "not-a-real-secret")
    monkeypatch.delenv(CONNECT_IP_ENV, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setattr(remote_vl, "_hostname_uses_fake_ip", lambda _host: True)
    monkeypatch.setattr(
        remote_vl,
        "_direct_ip_urlopen",
        lambda *_args, **_kwargs: pytest.fail(
            "configured HTTPS proxy must take precedence over direct-IP routing"
        ),
    )
    proxy_calls: list[str] = []

    def proxy_open(http_request: Any, *, timeout: float) -> _Response:
        del timeout
        proxy_calls.append(http_request.full_url)
        return _Response(
            {
                "choices": [
                    {
                        "message": {"content": '{"schema_version":"ok"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 3},
            }
        )

    monkeypatch.setattr(remote_vl.request, "urlopen", proxy_open)
    runner = OpenAICompatibleVisionRunner(
        base_url="https://api.example.invalid/v1",
        model="gpt-5.6-terra",
        api_key_env="TEST_REMOTE_KEY",
    )
    result = runner.generate_with_metadata(
        {"messages": [{"role": "user", "content": "return JSON"}]}
    )

    assert result.text == '{"schema_version":"ok"}'
    assert proxy_calls == [
        "https://api.example.invalid/v1/chat/completions"
    ]


def test_vectorengine_fake_ip_uses_provider_owned_cn_transport_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_REMOTE_KEY", "not-a-real-secret")
    monkeypatch.delenv(CONNECT_IP_ENV, raising=False)
    monkeypatch.setattr(remote_vl, "_hostname_uses_fake_ip", lambda _host: True)
    captured: dict[str, Any] = {}

    def alias_open(http_request: Any, *, timeout: float) -> _Response:
        captured["url"] = http_request.full_url
        captured["authorization"] = http_request.get_header("Authorization")
        captured["timeout"] = timeout
        return _Response(
            {
                "choices": [
                    {
                        "message": {"content": '{"schema_version":"ok"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 3},
            }
        )

    monkeypatch.setattr(remote_vl.request, "urlopen", alias_open)
    runner = OpenAICompatibleVisionRunner(
        base_url="https://api.vectorengine.ai/v1",
        model="gpt-5.6-terra",
        api_key_env="TEST_REMOTE_KEY",
        timeout_seconds=43,
    )
    result = runner.generate_with_metadata(
        {"messages": [{"role": "user", "content": "return JSON"}]}
    )

    assert result.text == '{"schema_version":"ok"}'
    assert captured == {
        "url": "https://api.vectorengine.cn/v1/chat/completions",
        "authorization": "Bearer not-a-real-secret",
        "timeout": 43.0,
    }
    assert runner.model_identity["base_url"] == (
        "https://api.vectorengine.ai/v1"
    )
