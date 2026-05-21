from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from debug import debug_log

try:
    import weave  # type: ignore
except ImportError:  # pragma: no cover - optional runtime dependency
    weave = None

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional runtime dependency
    wandb = None


_WEAVE_CLIENT: Any | None = None
_WEAVE_INIT_FAILED = False
_OP_CACHE: dict[str, Callable[..., Any]] = {}
DEFAULT_WEAVE_PROJECT = "pattern-learning-v2"
_FALSEY_ENV_VALUES = {"", "0", "false", "no", "off"}


def _parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_key = key.strip()
        if not env_key:
            continue
        values[env_key] = value.strip().strip("'").strip('"')
    return values


def _load_wandb_env_from_dotenv() -> str | None:
    candidates: list[Path] = []
    cwd = Path.cwd()
    candidates.append(cwd / ".env")
    for parent in cwd.parents:
        candidates.append(parent / ".env")
        if (parent / ".git").exists():
            break
    module_dir = Path(__file__).resolve().parent
    for parent in (module_dir, *module_dir.parents):
        candidates.append(parent / ".env")
        if (parent / ".git").exists():
            break
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        env_values = _parse_dotenv(candidate)
        api_key = str(env_values.get("WANDB_API_KEY") or "").strip()
        if not api_key:
            continue
        os.environ.setdefault("WANDB_API_KEY", api_key)
        project = str(env_values.get("WANDB_PROJECT") or env_values.get("WEAVE_PROJECT") or "").strip()
        if project:
            os.environ.setdefault("WANDB_PROJECT", project)
            os.environ.setdefault("WEAVE_PROJECT", project)
        return api_key
    return None


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in _FALSEY_ENV_VALUES


def observability_disabled() -> bool:
    return _env_flag_enabled("DISABLE_WANDB") or _env_flag_enabled("DISABLE_WEAVE")


def weave_project_name() -> str:
    for key in ("WEAVE_PROJECT", "WANDB_PROJECT"):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    return DEFAULT_WEAVE_PROJECT


def ensure_wandb_login() -> bool:
    if observability_disabled():
        debug_log("wandb.login.skipped", reason="disabled_by_env")
        return False
    if wandb is None:
        debug_log("wandb.login.skipped", reason="wandb_missing")
        return False
    api_key = os.environ.get("WANDB_API_KEY") or _load_wandb_env_from_dotenv()
    if api_key and api_key.strip():
        debug_log("wandb.login.available", source="env")
        return True
    debug_log("wandb.login.skipped", reason="missing_api_key")
    return False


def ensure_weave_initialized() -> Any | None:
    global _WEAVE_CLIENT, _WEAVE_INIT_FAILED

    if _WEAVE_CLIENT is not None:
        return _WEAVE_CLIENT
    if observability_disabled():
        debug_log("weave.init.skipped", reason="disabled_by_env")
        return None
    if _WEAVE_INIT_FAILED or weave is None:
        return None

    project_name = weave_project_name()
    if not ensure_wandb_login():
        return None

    try:
        _WEAVE_CLIENT = weave.init(project_name)
        debug_log("weave.init.done", project_name=project_name)
        return _WEAVE_CLIENT
    except Exception as exc:  # noqa: BLE001
        _WEAVE_INIT_FAILED = True
        debug_log("weave.init.failed", project_name=project_name, error=str(exc))
        return None


def weave_enabled() -> bool:
    return ensure_weave_initialized() is not None


def call_weave_op(name: str, fn: Callable[..., Any], /, **kwargs: Any) -> Any:
    if ensure_weave_initialized() is None or weave is None:
        return fn(**kwargs)

    op = _OP_CACHE.get(name)
    if op is None:

        @weave.op(name=name, enable_code_capture=False)
        def _wrapped(**inner_kwargs: Any) -> Any:
            return fn(**inner_kwargs)

        op = _wrapped
        _OP_CACHE[name] = op

    debug_log("weave.op.call", name=name, keys=sorted(kwargs.keys()))
    return op(**kwargs)
