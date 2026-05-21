from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


_FALSEY_DEBUG_VALUES = {"", "0", "false", "no", "off"}
_DEBUG_LOG_PATH: Path | None = None


def debug_enabled() -> bool:
    value = os.environ.get("DEBUG", "")
    return value.strip().lower() not in _FALSEY_DEBUG_VALUES


def set_debug_log_file(path: str | Path | None) -> None:
    global _DEBUG_LOG_PATH
    if path is None:
        _DEBUG_LOG_PATH = None
        return
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    _DEBUG_LOG_PATH = resolved


def get_debug_log_file() -> Path | None:
    env_override = os.environ.get("DEBUG_LOG_PATH")
    if env_override and env_override.strip():
        return Path(env_override).expanduser()
    return _DEBUG_LOG_PATH


def debug_log(event: str, /, **fields: Any) -> None:
    if not debug_enabled():
        return
    _emit_log("DEBUG", event, **fields)


def progress_log(event: str, /, **fields: Any) -> None:
    _emit_log("INFO", event, **fields)


def _emit_log(level: str, event: str, /, **fields: Any) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    console_line = _format_console(payload)
    print(console_line.replace("[DEBUG]", f"[{level}]"), file=sys.stderr, flush=True)
    log_path = get_debug_log_file()
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=_json_default, sort_keys=True))
            handle.write("\n")


def _json_default(value: Any) -> Any:
    if hasattr(value, "as_posix"):
        return value.as_posix()
    return str(value)


def _format_console(payload: dict[str, Any]) -> str:
    event = str(payload["event"])
    summary = _event_summary(event=event, payload=payload)
    if summary:
        return f"[DEBUG] {event}: {summary}"
    return f"[DEBUG] {event}"


def _event_summary(*, event: str, payload: dict[str, Any]) -> str:
    if event == "cli.start":
        argv = payload.get("argv", [])
        if isinstance(argv, list):
            shortened = [Path(str(arg)).name if index == 0 else str(arg) for index, arg in enumerate(argv[:6])]
            if len(argv) > 6:
                shortened.append("...")
            return "argv=" + " ".join(shortened)
    if event.startswith("http.request."):
        return _render_pairs(payload, ["method", "url", "status", "timeout", "reason"], max_fields=3)
    if event.startswith("patterns.chat.") or event.startswith("patterns.ollama."):
        return _render_pairs(payload, ["model", "base_url", "chars", "model_count"], max_fields=3)
    if event.startswith("patterns.vllm."):
        return _render_pairs(payload, ["base_url", "model", "request_id", "target_chars", "timeout"], max_fields=3)
    if event.startswith("patterns.learning."):
        return _render_pairs(
            payload,
            [
                "worker",
                "label",
                "index",
                "total_labels",
                "evaluation_index",
                "score",
                "validation_score",
                "sample_count",
                "validation_sample_count",
                "primitive_label_count",
                "chars",
                "event_count",
            ],
            max_fields=5,
        )
    if event.startswith("datasets.build."):
        return _render_pairs(
            payload,
            [
                "dataset_name",
                "split",
                "env_id",
                "completed_samples",
                "completed_after",
                "generated_this_pass",
                "pending_count",
                "sample_count",
                "output_root",
            ],
            max_fields=5,
        )
    if event.startswith("generation."):
        return _render_pairs(
            payload,
            [
                "env_id",
                "task_id",
                "sampling_strategy",
                "timestep_count",
                "attachment_count",
                "sample_dir",
                "error",
                "reason",
            ],
            max_fields=5,
        )
    if event.startswith("vllm.http."):
        return _render_pairs(payload, ["method", "path", "request_id", "payload_keys"], max_fields=3)
    if event.startswith("vllm.service.") or event.startswith("vllm.worker."):
        return _render_pairs(payload, ["request_id", "request_ids", "batch_size", "queue_size", "remaining_queue", "error"], max_fields=4)
    if event.startswith("cli.pattern_learn."):
        return _render_pairs(
            payload,
            ["env", "output_path", "sample_count", "validation_sample_count", "average_score", "generator_model"],
            max_fields=5,
        )
    if event.startswith("dashboard.workflows."):
        return _render_pairs(
            payload,
            ["env", "iteration_count", "reasoning_trace_count", "proposed_label_count", "run_dir", "returncode"],
            max_fields=5,
        )
    return _render_pairs(payload, ["env", "env_id", "task_id", "output_path", "output_root", "error"], max_fields=4)


def _render_pairs(payload: dict[str, Any], ordered_keys: list[str], *, max_fields: int) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for key in ordered_keys:
        if key in payload and payload[key] is not None:
            parts.append(f"{key}={_format_console_value(payload[key])}")
            seen.add(key)
        if len(parts) >= max_fields:
            return " ".join(parts)
    for key in sorted(payload):
        if key in {"event", "ts"} or key in seen or payload[key] is None:
            continue
        parts.append(f"{key}={_format_console_value(payload[key])}")
        if len(parts) >= max_fields:
            break
    return " ".join(parts)


def _format_console_value(value: Any) -> str:
    if hasattr(value, "as_posix"):
        value = value.as_posix()
    if isinstance(value, str) and (value.startswith("http://") or value.startswith("https://")):
        parsed = urlparse(value)
        suffix = parsed.path or "/"
        if parsed.query:
            suffix += "?" + parsed.query
        value = f"{parsed.netloc}{suffix}"
    if isinstance(value, list):
        rendered = ",".join(_format_console_value(item) for item in value[:4])
        if len(value) > 4:
            rendered += ",..."
        return f"[{rendered}]"
    if isinstance(value, dict):
        keys = list(value.keys())
        rendered = ",".join(str(key) for key in keys[:4])
        if len(keys) > 4:
            rendered += ",..."
        return f"{{{rendered}}}"
    text = str(value)
    if len(text) > 80:
        return text[:77] + "..."
    return text
