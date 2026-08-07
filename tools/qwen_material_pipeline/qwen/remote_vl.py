"""OpenAI-compatible remote vision-language backend.

The material workflow keeps its existing staged JSON validators and evidence
gates.  This module replaces only the model transport: it submits the bounded
multimodal payload to an OpenAI-compatible Chat Completions endpoint and
returns the assistant text plus completion telemetry.

API keys are referenced by environment-variable name.  They are never written
to checkpoints, inference ledgers, subprocess arguments, or error messages.
"""

from __future__ import annotations

import copy
import http.client
import ipaddress
import json
import os
import re
import socket
import ssl
import time
from collections.abc import Callable, Mapping
from typing import Any
from urllib import error, parse, request

from qwen_material_pipeline.qwen.client import (
    QwenClientError,
    QwenResponseError,
)
from qwen_material_pipeline.qwen.local_vl import LocalGenerationResult


REMOTE_IDENTITY_SCHEMA_VERSION = "openai-compatible-vlm-identity/v1"
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REASONING_EFFORTS = frozenset(
    {"none", "low", "medium", "high", "xhigh", "max"}
)
DEFAULT_TRANSPORT_MAX_ATTEMPTS = 5
DEFAULT_TRANSPORT_BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0)
CONNECT_IP_ENV = "OPENAI_COMPATIBLE_CONNECT_IP"
FAKE_IP_NETWORK = ipaddress.ip_network("198.18.0.0/15")
AUTO_DIRECT_IP_TTL_SECONDS = 30.0
_PROXY_ENV_NAMES = (
    "HTTPS_PROXY",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
)
_FAKE_IP_PROVIDER_HOST_ALIASES = {
    # Both hosts expose the same VectorEngine account/model inventory.  The
    # .ai edge intermittently terminates TLS or returns an Aliyun ICP block
    # when reached from Clash fake-IP networks, while the provider's .cn edge
    # is routable through the configured proxy.
    "api.vectorengine.ai": "api.vectorengine.cn",
}


class _DirectIPHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection with explicit routing and normal hostname TLS SNI."""

    def __init__(
        self,
        host: str,
        *,
        connect_ip: str,
        port: int,
        timeout: float,
    ) -> None:
        super().__init__(host, port=port, timeout=timeout)
        self._connect_ip = connect_ip

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._connect_ip, self.port),
            self.timeout,
            self.source_address,
        )
        if self._tunnel_host:
            self._tunnel()
        self.sock = self._context.wrap_socket(
            self.sock,
            server_hostname=self.host,
        )


def _direct_ip_urlopen(
    http_request: request.Request,
    *,
    timeout: float,
    connect_ip: str,
) -> Any:
    """Open one HTTPS request directly while preserving Host and TLS SNI."""

    parsed = parse.urlparse(http_request.full_url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("direct-IP transport requires an absolute HTTPS URL")
    connection = _DirectIPHTTPSConnection(
        parsed.hostname,
        connect_ip=connect_ip,
        port=parsed.port or 443,
        timeout=timeout,
    )
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    connection.request(
        http_request.get_method(),
        path,
        body=http_request.data,
        headers=dict(http_request.header_items()),
    )
    response = connection.getresponse()
    if response.status >= 400:
        raise error.HTTPError(
            http_request.full_url,
            response.status,
            response.reason,
            response.headers,
            response,
        )
    return response


def _hostname_uses_fake_ip(hostname: str) -> bool:
    """Return true only when every locally resolved address is Clash fake-IP."""

    try:
        records = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
    except OSError:
        return False
    addresses = {
        record[4][0]
        for record in records
        if record[4] and isinstance(record[4][0], str)
    }
    if not addresses:
        return False
    parsed_addresses = []
    for address in addresses:
        try:
            parsed_addresses.append(ipaddress.ip_address(address))
        except ValueError:
            return False
    return all(
        isinstance(address, ipaddress.IPv4Address)
        and address in FAKE_IP_NETWORK
        for address in parsed_addresses
    )


def _https_proxy_is_configured(hostname: str) -> bool:
    """Return whether urllib can route HTTPS without resolving the target.

    Clash-style fake-IP DNS is safe when urllib is using an HTTP(S) proxy:
    the proxy resolves the destination hostname after CONNECT.  Bypassing that
    working route in favour of a periodically resolved public IP made long
    unattended runs vulnerable to CDN edge TLS resets.
    """

    if not any(os.getenv(name, "").strip() for name in _PROXY_ENV_NAMES):
        return False
    try:
        if request.proxy_bypass(hostname):
            return False
    except OSError:
        return False
    proxies = request.getproxies()
    return bool(proxies.get("https") or proxies.get("all"))


def _resolve_public_ipv4_doh(hostname: str, *, timeout: float) -> str:
    """Resolve one current public A record without relying on fake-IP DNS."""

    query = parse.urlencode({"name": hostname, "type": "A"})
    resolvers = (
        ("dns.google", "8.8.8.8", f"/resolve?{query}"),
        (
            "cloudflare-dns.com",
            "1.1.1.1",
            f"/dns-query?{query}",
        ),
    )
    failures: list[str] = []
    for resolver_host, resolver_ip, path in resolvers:
        connection = _DirectIPHTTPSConnection(
            resolver_host,
            connect_ip=resolver_ip,
            port=443,
            timeout=min(timeout, 15.0),
        )
        try:
            connection.request(
                "GET",
                path,
                headers={"Accept": "application/dns-json"},
            )
            response = connection.getresponse()
            payload = response.read()
            if response.status != 200:
                failures.append(f"{resolver_host}: HTTP {response.status}")
                continue
            document = json.loads(payload.decode("utf-8"))
            answers = (
                document.get("Answer")
                if isinstance(document, Mapping)
                else None
            )
            if not isinstance(answers, list):
                failures.append(f"{resolver_host}: no Answer array")
                continue
            for answer in answers:
                if not isinstance(answer, Mapping) or answer.get("type") != 1:
                    continue
                candidate = answer.get("data")
                try:
                    address = ipaddress.ip_address(candidate)
                except ValueError:
                    continue
                if (
                    isinstance(address, ipaddress.IPv4Address)
                    and address.is_global
                    and address not in FAKE_IP_NETWORK
                ):
                    return str(address)
            failures.append(f"{resolver_host}: no public A record")
        except (
            OSError,
            ssl.SSLError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            failures.append(f"{resolver_host}: {type(exc).__name__}")
        finally:
            connection.close()
    raise ConnectionError(
        "could not resolve current gateway address through trusted DNS: "
        + "; ".join(failures)
    )


class _AutoDirectIPOpener:
    """Refresh a fake-IP gateway route periodically and after transport errors."""

    def __init__(self, hostname: str) -> None:
        self.hostname = hostname
        self.connect_ip: str | None = None
        self.resolved_at = 0.0

    def __call__(self, http_request: request.Request, *, timeout: float) -> Any:
        now = time.monotonic()
        if (
            self.connect_ip is None
            or now - self.resolved_at >= AUTO_DIRECT_IP_TTL_SECONDS
        ):
            self.connect_ip = _resolve_public_ipv4_doh(
                self.hostname,
                timeout=timeout,
            )
            self.resolved_at = now
        try:
            return _direct_ip_urlopen(
                http_request,
                timeout=timeout,
                connect_ip=self.connect_ip,
            )
        except (OSError, ssl.SSLError):
            self.connect_ip = None
            self.resolved_at = 0.0
            raise


def _request_with_hostname(
    http_request: request.Request,
    *,
    hostname: str,
) -> request.Request:
    """Clone a request onto one provider-owned transport hostname."""

    parsed = parse.urlparse(http_request.full_url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("provider alias transport requires an HTTPS URL")
    netloc = hostname
    if parsed.port is not None:
        netloc += f":{parsed.port}"
    alias_url = parse.urlunparse(parsed._replace(netloc=netloc))
    return request.Request(
        alias_url,
        data=http_request.data,
        method=http_request.get_method(),
        headers=dict(http_request.header_items()),
    )


class _ProviderAliasOpener:
    """Route a logical API endpoint through its provider-owned stable alias."""

    def __init__(self, hostname: str) -> None:
        self.hostname = hostname

    def __call__(self, http_request: request.Request, *, timeout: float) -> Any:
        return request.urlopen(
            _request_with_hostname(http_request, hostname=self.hostname),
            timeout=timeout,
        )


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _content_text(value: Any) -> str:
    """Extract assistant text from an OpenAI-compatible message content."""

    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            text = item.get("text")
            if item.get("type") in {"text", "output_text"} and isinstance(text, str):
                chunks.append(text)
        if chunks:
            return "".join(chunks)
    raise QwenResponseError(
        "OpenAI-compatible response contains no assistant text"
    )


class OpenAICompatibleVisionRunner:
    """Callable staged-model runner for an OpenAI-compatible HTTPS gateway."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str,
        reasoning_effort: str = "medium",
        timeout_seconds: float = 180.0,
        max_new_tokens: int = 1024,
        opener: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self.base_url = _nonempty(base_url, "base_url").rstrip("/")
        parsed = parse.urlparse(self.base_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError(
                "OpenAI-compatible base_url must be an absolute HTTPS URL"
            )
        self.model = _nonempty(model, "model")
        self.api_key_env = _nonempty(api_key_env, "api_key_env")
        if _ENV_NAME_RE.fullmatch(self.api_key_env) is None:
            raise ValueError("api_key_env must be a valid environment-variable name")
        self.reasoning_effort = _nonempty(
            reasoning_effort, "reasoning_effort"
        )
        if self.reasoning_effort not in _REASONING_EFFORTS:
            raise ValueError(
                "reasoning_effort must be one of "
                f"{sorted(_REASONING_EFFORTS)}"
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or float(timeout_seconds) <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        self.timeout_seconds = float(timeout_seconds)
        self.max_new_tokens = max_new_tokens
        connect_ip = os.getenv(CONNECT_IP_ENV, "").strip()
        if connect_ip:
            try:
                ipaddress.ip_address(connect_ip)
            except ValueError as exc:
                raise ValueError(
                    f"{CONNECT_IP_ENV} must contain one IPv4 or IPv6 address"
                ) from exc
        if opener is not None:
            self._opener = opener
        elif connect_ip:
            self._opener = lambda http_request, *, timeout: _direct_ip_urlopen(
                http_request,
                timeout=timeout,
                connect_ip=connect_ip,
            )
        elif (
            parsed.hostname in _FAKE_IP_PROVIDER_HOST_ALIASES
            and _hostname_uses_fake_ip(parsed.hostname or "")
        ):
            self._opener = _ProviderAliasOpener(
                _FAKE_IP_PROVIDER_HOST_ALIASES[parsed.hostname]
            )
        elif _https_proxy_is_configured(parsed.hostname or ""):
            # urllib performs CONNECT through the configured proxy, so local
            # fake-IP DNS must not force the less stable direct-IP transport.
            self._opener = request.urlopen
        elif _hostname_uses_fake_ip(parsed.hostname or ""):
            self._opener = _AutoDirectIPOpener(parsed.hostname or "")
        else:
            self._opener = request.urlopen
        self._sleeper = sleeper or time.sleep
        self.model_identity = {
            "schema_version": REMOTE_IDENTITY_SCHEMA_VERSION,
            "backend": "openai_compatible_chat_completions",
            "model_type": "openai_compatible",
            "model": self.model,
            "base_url": self.base_url,
            "endpoint": self.endpoint,
            "api_key_env": self.api_key_env,
            "generation": {
                "max_new_tokens": self.max_new_tokens,
                "reasoning_effort": self.reasoning_effort,
                "response_format": "json_object",
            },
        }

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    def preflight(self) -> dict[str, Any]:
        """Validate configuration and credential presence without sending data."""

        if not os.getenv(self.api_key_env):
            raise QwenClientError(
                "Missing remote vision-model credential; set the configured "
                f"environment variable {self.api_key_env}"
            )
        return copy.deepcopy(self.model_identity)

    def unload(self) -> None:
        """Remote inference owns no local GPU state."""

    def __call__(self, payload: Mapping[str, Any]) -> str:
        return self.generate_with_metadata(payload).text

    def generate_with_metadata(
        self,
        payload: Mapping[str, Any],
        *,
        max_new_tokens: int | None = None,
    ) -> LocalGenerationResult:
        if not isinstance(payload, Mapping):
            raise TypeError("remote vision-model payload must be an object")
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise QwenClientError(
                "Missing remote vision-model credential; set the configured "
                f"environment variable {self.api_key_env}"
            )
        budget = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError("max_new_tokens must be a positive integer")

        body_document = copy.deepcopy(dict(payload))
        body_document["model"] = self.model
        body_document.pop("enable_thinking", None)
        # Local Qwen payloads use sampling controls that reasoning-model
        # gateways may reject. Determinism comes from the strict JSON schema
        # and reasoning contract in this backend.
        body_document.pop("temperature", None)
        body_document.pop("top_p", None)
        body_document["reasoning_effort"] = self.reasoning_effort
        body_document["max_completion_tokens"] = budget
        body = json.dumps(
            body_document,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        http_request = request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        raw_response: bytes | None = None
        last_transport_error: BaseException | None = None
        for attempt in range(1, DEFAULT_TRANSPORT_MAX_ATTEMPTS + 1):
            try:
                response = self._opener(
                    http_request,
                    timeout=self.timeout_seconds,
                )
                try:
                    raw_response = response.read()
                finally:
                    close = getattr(response, "close", None)
                    if callable(close):
                        close()
                break
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                if api_key:
                    detail = detail.replace(api_key, "[REDACTED]")
                raise QwenClientError(
                    "OpenAI-compatible gateway returned "
                    f"HTTP {exc.code}: {detail or exc.reason}"
                ) from exc
            except (error.URLError, ConnectionError, ssl.SSLError) as exc:
                # Retry only failures known to occur before an HTTP response is
                # established.  A timeout after request upload is deliberately
                # not retried because the gateway may already be generating a
                # billable completion.
                reason = exc.reason if isinstance(exc, error.URLError) else exc
                retryable = isinstance(
                    reason,
                    (
                        ssl.SSLError,
                        ConnectionError,
                    ),
                )
                if not retryable or attempt == DEFAULT_TRANSPORT_MAX_ATTEMPTS:
                    raise QwenClientError(
                        "Could not reach OpenAI-compatible gateway after "
                        f"{attempt} transport attempt(s): {reason}"
                    ) from exc
                last_transport_error = exc
                self._sleeper(
                    DEFAULT_TRANSPORT_BACKOFF_SECONDS[attempt - 1]
                )
        if raw_response is None:
            if last_transport_error is None:
                raise AssertionError("remote transport ended without a response")
            raise QwenClientError(
                "Could not reach OpenAI-compatible gateway after bounded "
                "transport retries: "
                + str(
                    last_transport_error.reason
                    if isinstance(last_transport_error, error.URLError)
                    else last_transport_error
                )
            ) from last_transport_error

        try:
            envelope = json.loads(raw_response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QwenResponseError(
                "OpenAI-compatible response is not valid UTF-8 JSON"
            ) from exc
        if not isinstance(envelope, Mapping):
            raise QwenResponseError(
                "OpenAI-compatible response must be a JSON object"
            )
        choices = envelope.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise QwenResponseError(
                "OpenAI-compatible response must contain exactly one choice"
            )
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise QwenResponseError(
                "OpenAI-compatible response choice is invalid"
            )
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise QwenResponseError(
                "OpenAI-compatible response choice has no message"
            )
        text = _content_text(message.get("content"))
        finish_reason = choice.get("finish_reason")
        usage = envelope.get("usage")
        generated_tokens = (
            usage.get("completion_tokens")
            if isinstance(usage, Mapping)
            else None
        )
        if (
            isinstance(generated_tokens, bool)
            or not isinstance(generated_tokens, int)
            or generated_tokens < 0
        ):
            generated_tokens = budget if finish_reason == "length" else 0
        return LocalGenerationResult(
            text=text,
            generated_tokens=generated_tokens,
            max_new_tokens=budget,
            hit_token_limit=finish_reason == "length",
            eos_detected=True if finish_reason == "stop" else None,
        )


__all__ = [
    "OpenAICompatibleVisionRunner",
    "REMOTE_IDENTITY_SCHEMA_VERSION",
    "CONNECT_IP_ENV",
]
