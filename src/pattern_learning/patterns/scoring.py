from __future__ import annotations

import http.client
import json
import os
import socket
import time
from typing import Any, Protocol
from urllib import error, request

from debug import debug_log
from llm_config import GLOBAL_MIN_COMPLETION_TOKENS
from .prompts import PatternPrompt
from weave_support import call_weave_op


NO_TIMEOUT_CONNECT_RETRY_WINDOW_SECONDS = 600.0
CONNECT_RETRY_INTERVAL_SECONDS = 1.0
TRANSIENT_REQUEST_MAX_ATTEMPTS = 8
TRANSIENT_REQUEST_RETRY_BASE_SECONDS = 1.0
THROTTLED_REQUEST_MAX_ATTEMPTS = 18
THROTTLED_REQUEST_RETRY_BASE_SECONDS = 10.0
EMPTY_TEXT_RESPONSE_MAX_ATTEMPTS = 4
EMPTY_TEXT_RESPONSE_RETRY_BASE_SECONDS = 2.0
CHAT_COMPLETION_RETRY_WINDOW_SECONDS = 600.0
CHAT_COMPLETION_RETRY_INTERVAL_SECONDS = 15.0
TRANSIENT_HTTP_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
TRANSIENT_ERROR_TEXT_SNIPPETS = (
    "server disconnected without sending a response",
    "remote end closed connection without response",
    "temporary overload",
    "connection reset",
    "timed out",
    "connection aborted",
    "unexpected eof while reading",
    "eof occurred in violation of protocol",
    "peer closed connection without sending complete message body",
    "remoteprotocolerror",
)


class PatternResponder(Protocol):
    def require_healthy(self) -> dict[str, Any]: ...

    def respond(self, prompt: PatternPrompt) -> str: ...


class OllamaResponder:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str = "qwen3.5:27b",
        timeout: int = 120,
    ) -> None:
        resolved_base_url = str(base_url or "").strip()
        if not resolved_base_url:
            raise ValueError(
                "OllamaResponder requires an explicit base_url. "
                "Configure LLM_ROUTER_BASE_URL and use the OpenAI-compatible chat responder instead."
            )
        self.base_url = resolved_base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def respond(self, prompt: PatternPrompt) -> str:
        debug_log("patterns.ollama.respond.start", base_url=self.base_url, model=self.model)
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
        }
        body = call_weave_op(
            "deepphy.ollama.chat",
            _request_json,
            base_url=self.base_url,
            path="/api/chat",
            method="POST",
            payload=payload,
            timeout=self.timeout,
        )
        message = body.get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Ollama response did not contain message.content")
        debug_log("patterns.ollama.respond.done", base_url=self.base_url, model=self.model, chars=len(content.strip()))
        return content.strip()

    def require_healthy(self) -> dict[str, Any]:
        debug_log("patterns.ollama.health.start", base_url=self.base_url)
        payload = _request_json(
            base_url=self.base_url,
            path="/api/tags",
            method="GET",
            payload=None,
            timeout=self.timeout,
        )
        if not isinstance(payload.get("models"), list):
            raise RuntimeError(f"Ollama endpoint at {self.base_url} is unhealthy: {payload}")
        debug_log("patterns.ollama.health.done", base_url=self.base_url, model_count=len(payload["models"]))
        return payload


class ChatCompletionsResponder:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: int = 120,
        max_tokens: int | None = None,
        api_key: str | None = None,
    ) -> None:
        self.base_url = _normalize_chat_base_url(base_url)
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.api_key = api_key or _default_chat_api_key()

    def respond(self, prompt: PatternPrompt) -> str:
        debug_log("patterns.chat.respond.start", base_url=self.base_url, model=self.model)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body, content = request_chat_completion_with_text_retry(
            base_url=self.base_url,
            model=self.model,
            messages=[
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            timeout=self.timeout,
            headers=headers,
            max_tokens=self.max_tokens,
            temperature=0.0,
            error_prefix="Chat completions response",
            weave_op_name="deepphy.reference.chat_completion",
        )
        debug_log("patterns.chat.respond.done", base_url=self.base_url, model=self.model, chars=len(content))
        return content

    def require_healthy(self) -> dict[str, Any]:
        debug_log("patterns.chat.health.start", base_url=self.base_url, model=self.model)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = _request_json(
            base_url=self.base_url,
            path="/v1/models",
            method="GET",
            payload=None,
            timeout=self.timeout,
            headers=headers,
        )
        if not isinstance(payload.get("data"), list):
            raise RuntimeError(
                f"OpenAI-compatible endpoint at {self.base_url} is unhealthy: {payload}"
            )
        debug_log("patterns.chat.health.done", base_url=self.base_url, model_count=len(payload["data"]))
        return payload


def _default_chat_api_key() -> str | None:
    value = os.environ.get("LLM_ROUTER_API_KEY")
    if value is not None and value.strip():
        return value.strip()
    return None


def _with_default_chat_headers(headers: dict[str, str] | None) -> dict[str, str]:
    merged = dict(headers or {})
    merged.setdefault("Content-Type", "application/json")
    api_key = _default_chat_api_key()
    if api_key and "Authorization" not in merged and "X-API-Key" not in merged:
        merged["Authorization"] = f"Bearer {api_key}"
    return merged


def _normalize_chat_base_url(base_url: str) -> str:
    normalized = str(base_url).strip().rstrip("/")
    if normalized.endswith("/v1"):
        return normalized[:-3]
    return normalized


def extract_chat_completion_text(body: dict[str, Any], *, error_prefix: str) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"{error_prefix} did not contain choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError(f"{error_prefix} contained a non-dict choice payload")

    message = first.get("message", {})
    if isinstance(message, dict):
        content = _extract_content_text(message.get("content"))
        if content:
            return content
        for extra_key in ("text", "output_text", "reasoning_content", "reasoning"):
            content = _extract_content_text(message.get(extra_key))
            if content:
                return content

    text = first.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    nested_output = _extract_content_text(first.get("output"))
    if nested_output:
        return nested_output

    delta = first.get("delta", {})
    if isinstance(delta, dict):
        content = _extract_content_text(delta.get("content"))
        if content:
            return content
        for extra_key in ("text", "output_text", "reasoning_content", "reasoning"):
            content = _extract_content_text(delta.get(extra_key))
            if content:
                return content
    debug_log(
        "patterns.chat.extract_failed",
        error_prefix=error_prefix,
        choice_keys=sorted(first.keys()),
        message_keys=sorted(message.keys()) if isinstance(message, dict) else [],
        body_preview=_truncate_preview(body),
    )
    raise ValueError(f"{error_prefix} did not contain text content")


def request_chat_completion_with_text_retry(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: int | None,
    headers: dict[str, str] | None,
    error_prefix: str,
    max_tokens: int | None = None,
    temperature: float = 0.0,
    stream: bool = False,
    extra_payload: dict[str, Any] | None = None,
    weave_op_name: str | None = None,
    request_fn: Any | None = None,
) -> tuple[dict[str, Any], str]:
    base_url = _normalize_chat_base_url(base_url)
    headers = _with_default_chat_headers(headers)
    retry_deadline = time.monotonic() + CHAT_COMPLETION_RETRY_WINDOW_SECONDS
    resolved_max_tokens = GLOBAL_MIN_COMPLETION_TOKENS
    if max_tokens is not None:
        resolved_max_tokens = max(GLOBAL_MIN_COMPLETION_TOKENS, int(max_tokens))
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": resolved_max_tokens,
        "stream": stream,
        **(extra_payload or {}),
    }
    resolved_model = str(model or "").strip()
    if resolved_model:
        payload["model"] = resolved_model
    attempts = 0
    while True:
        attempts += 1
        try:
            if weave_op_name:
                body = call_weave_op(
                    weave_op_name,
                    request_fn or _request_json,
                    base_url=base_url,
                    path="/v1/chat/completions",
                    method="POST",
                    payload=payload,
                    timeout=timeout,
                    headers=headers,
                )
            else:
                body = (request_fn or _request_json)(
                    base_url=base_url,
                    path="/v1/chat/completions",
                    method="POST",
                    payload=payload,
                    timeout=timeout,
                    headers=headers,
                )
        except RuntimeError as exc:
            if not _should_retry_chat_runtime_error(exc):
                raise
            remaining_seconds = retry_deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise RuntimeError(
                    f"Chat completions request failed continuously for {int(CHAT_COMPLETION_RETRY_WINDOW_SECONDS)}s: {exc}"
                ) from exc
            delay_seconds = min(CHAT_COMPLETION_RETRY_INTERVAL_SECONDS, remaining_seconds)
            debug_log(
                "patterns.chat.long_retry",
                base_url=base_url,
                model=model,
                attempts=attempts,
                delay_seconds=delay_seconds,
                remaining_seconds=round(remaining_seconds, 3),
                error=str(exc),
            )
            time.sleep(delay_seconds)
            continue
        try:
            return body, extract_chat_completion_text(body, error_prefix=error_prefix)
        except ValueError as exc:
            if "did not contain text content" not in str(exc).lower():
                raise
            if attempts >= EMPTY_TEXT_RESPONSE_MAX_ATTEMPTS:
                raise
            delay_seconds = min(EMPTY_TEXT_RESPONSE_RETRY_BASE_SECONDS * attempts, 8.0)
            debug_log(
                "patterns.chat.empty_text_retry",
                base_url=base_url,
                model=model,
                attempts=attempts,
                delay_seconds=delay_seconds,
                error_prefix=error_prefix,
            )
            time.sleep(delay_seconds)


def _should_retry_chat_runtime_error(exc: RuntimeError) -> bool:
    normalized = str(exc).lower()
    if "missing public router api key" in normalized:
        return False
    if "http 400" in normalized or "http 401" in normalized or "http 403" in normalized or "http 404" in normalized:
        return False
    if "http 408" in normalized or "http 409" in normalized or "http 425" in normalized:
        return True
    if "http 429" in normalized or "http 500" in normalized or "http 502" in normalized or "http 503" in normalized or "http 504" in normalized:
        return True
    if "could not reach service" in normalized or "timed out waiting for service" in normalized:
        return True
    return any(snippet in normalized for snippet in TRANSIENT_ERROR_TEXT_SNIPPETS)


def _extract_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        for key in ("text", "content", "output_text", "reasoning_content", "reasoning", "value"):
            extracted = _extract_content_text(content.get(key))
            if extracted:
                return extracted
        for value in content.values():
            extracted = _extract_content_text(value)
            if extracted:
                return extracted
        return ""
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for item in content:
        extracted = _extract_content_text(item)
        if extracted:
            text_parts.append(extracted)
    return "".join(text_parts).strip()


def _truncate_preview(value: Any, *, limit: int = 1200) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError:
        text = repr(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _request_json(
    *,
    base_url: str,
    path: str,
    method: str,
    payload: dict[str, Any] | None,
    timeout: int | None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    resolved_timeout = None if timeout is None or timeout <= 0 else timeout
    connect_retry_deadline = (
        time.monotonic() + NO_TIMEOUT_CONNECT_RETRY_WINDOW_SECONDS
        if resolved_timeout is None
        else None
    )
    transient_attempts = 0
    debug_log(
        "http.request.start",
        method=method,
        url=f"{base_url}{path}",
        timeout=resolved_timeout if resolved_timeout is not None else "none",
        payload_keys=sorted(payload.keys()) if payload is not None else [],
    )
    encoded = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=f"{base_url}{path}",
        data=encoded,
        headers=headers or {"Content-Type": "application/json"},
        method=method,
    )
    while True:
        try:
            with request.urlopen(req, timeout=resolved_timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            break
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if _should_retry_transient_http_error(exc.code, detail, attempts=transient_attempts):
                transient_attempts += 1
                _sleep_before_transient_retry(
                    method=method,
                    url=f"{base_url}{path}",
                    attempts=transient_attempts,
                    reason=f"HTTP {exc.code}",
                    throttled=_is_throttled_http_error(exc.code, detail),
                )
                continue
            debug_log("http.request.http_error", method=method, url=f"{base_url}{path}", status=exc.code)
            raise RuntimeError(f"HTTP {exc.code} from {base_url}{path}: {detail}") from exc
        except error.URLError as exc:
            if _should_retry_connecting(exc, connect_retry_deadline):
                remaining_seconds = max(connect_retry_deadline - time.monotonic(), 0.0)
                debug_log(
                    "http.request.waiting_for_service",
                    method=method,
                    url=f"{base_url}{path}",
                    reason=str(exc.reason),
                    remaining_seconds=round(remaining_seconds, 3),
                )
                time.sleep(min(CONNECT_RETRY_INTERVAL_SECONDS, remaining_seconds))
                continue
            if _should_retry_transient_transport_error(exc.reason, attempts=transient_attempts):
                transient_attempts += 1
                _sleep_before_transient_retry(
                    method=method,
                    url=f"{base_url}{path}",
                    attempts=transient_attempts,
                    reason=str(exc.reason),
                    throttled=False,
                )
                continue
            debug_log("http.request.url_error", method=method, url=f"{base_url}{path}", reason=str(exc.reason))
            if connect_retry_deadline is not None:
                raise RuntimeError(
                    f"Could not reach service at {base_url}{path} after waiting "
                    f"{int(NO_TIMEOUT_CONNECT_RETRY_WINDOW_SECONDS)}s: {exc.reason}"
                ) from exc
            raise RuntimeError(f"Could not reach service at {base_url}{path}: {exc.reason}") from exc
        except (
            TimeoutError,
            socket.timeout,
            http.client.RemoteDisconnected,
            ConnectionResetError,
            ConnectionAbortedError,
        ) as exc:
            if _should_retry_transient_transport_error(exc, attempts=transient_attempts):
                transient_attempts += 1
                _sleep_before_transient_retry(
                    method=method,
                    url=f"{base_url}{path}",
                    attempts=transient_attempts,
                    reason=str(exc),
                    throttled=False,
                )
                continue
            debug_log(
                "http.request.timeout",
                method=method,
                url=f"{base_url}{path}",
                timeout=resolved_timeout if resolved_timeout is not None else "none",
            )
            raise RuntimeError(
                f"Timed out waiting for service at {base_url}{path} after {resolved_timeout}s"
            ) from exc
    if not isinstance(body, dict):
        raise ValueError(f"Expected JSON object from {base_url}{path}")
    debug_log("http.request.done", method=method, url=f"{base_url}{path}", response_keys=sorted(body.keys()))
    return body


def _sleep_before_transient_retry(
    *,
    method: str,
    url: str,
    attempts: int,
    reason: str,
    throttled: bool,
) -> None:
    if throttled:
        delay_seconds = min(THROTTLED_REQUEST_RETRY_BASE_SECONDS * attempts, 60.0)
    else:
        delay_seconds = min(TRANSIENT_REQUEST_RETRY_BASE_SECONDS * attempts, 5.0)
    debug_log(
        "http.request.transient_retry",
        method=method,
        url=url,
        attempts=attempts,
        delay_seconds=delay_seconds,
        reason=reason,
        throttled=throttled,
    )
    time.sleep(delay_seconds)


def _should_retry_transient_http_error(status_code: int, detail: str, *, attempts: int) -> bool:
    if attempts >= _max_http_retry_attempts(status_code, detail):
        return False
    if status_code not in TRANSIENT_HTTP_STATUS_CODES:
        return False
    if not detail:
        return True
    normalized = detail.lower()
    return any(snippet in normalized for snippet in TRANSIENT_ERROR_TEXT_SNIPPETS) or status_code in {429, 500, 502, 503, 504}


def _max_http_retry_attempts(status_code: int, detail: str) -> int:
    if _is_throttled_http_error(status_code, detail):
        return THROTTLED_REQUEST_MAX_ATTEMPTS
    return TRANSIENT_REQUEST_MAX_ATTEMPTS


def _is_throttled_http_error(status_code: int, detail: str) -> bool:
    if status_code == 429:
        return True
    if not detail:
        return False
    normalized = detail.lower()
    return "too many requests" in normalized or "status/429" in normalized or '"code":429' in normalized


def _should_retry_transient_transport_error(reason: BaseException | str | object, *, attempts: int) -> bool:
    if attempts >= TRANSIENT_REQUEST_MAX_ATTEMPTS:
        return False
    normalized = str(reason).lower()
    if any(snippet in normalized for snippet in TRANSIENT_ERROR_TEXT_SNIPPETS):
        return True
    return isinstance(
        reason,
        (
            TimeoutError,
            socket.timeout,
            http.client.RemoteDisconnected,
            ConnectionResetError,
            ConnectionAbortedError,
            BrokenPipeError,
        ),
    )


def _should_retry_connecting(exc: error.URLError, deadline: float | None) -> bool:
    if deadline is None:
        return False
    reason = exc.reason
    errno_value = getattr(reason, "errno", None)
    if isinstance(reason, ConnectionRefusedError):
        return time.monotonic() < deadline
    if errno_value == 111:
        return time.monotonic() < deadline
    return "connection refused" in str(reason).lower() and time.monotonic() < deadline
