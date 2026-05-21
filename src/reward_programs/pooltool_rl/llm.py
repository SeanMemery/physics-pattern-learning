from __future__ import annotations

import json
import os
import time
from typing import Any

import requests

from .schemas import LLMSettings


def chat_completion(
    *,
    settings: LLMSettings,
    messages: list[dict[str, Any]],
    max_tokens: int = 4096,
    temperature: float = 0.0,
    progress: callable | None = None,
) -> tuple[dict[str, Any], str]:
    if not str(settings.model or "").strip():
        raise ValueError("LLM model is empty. Set `llm.model` in configs/config.yml (or pass a config with a model set).")
    url = settings.base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get(settings.api_key_env, "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": settings.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            if progress is not None:
                progress(f"llm: request attempt {attempt}/4 timeout={settings.timeout_seconds}s model={settings.model}")
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=settings.timeout_seconds)
            response.raise_for_status()
            body = response.json()
            text = extract_chat_text(body)
            if text:
                return body, text
            raise RuntimeError("Chat completion returned no text")
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if progress is not None:
                progress(f"llm: attempt {attempt}/4 failed: {type(exc).__name__}: {exc}")
            time.sleep(min(8, attempt))
    assert last_error is not None
    raise RuntimeError(f"LLM request failed: {last_error}") from last_error


def extract_chat_text(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    text = first.get("text")
    return str(text or "").strip()
