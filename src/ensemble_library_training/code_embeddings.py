from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


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


def resolve_openai_embedding_config(project_root: Path) -> tuple[str | None, str, str]:
    env_from_files: dict[str, str] = {}
    for candidate in (project_root / ".env", project_root.parent / ".env"):
        env_from_files.update(_parse_dotenv(candidate))
    api_key = (
        __import__("os").environ.get("OPENAI_API_KEY")
        or __import__("os").environ.get("PATTERN_LLM_API_KEY")
        or __import__("os").environ.get("JOB_LLM_API_KEY")
        or env_from_files.get("OPENAI_API_KEY")
        or env_from_files.get("PATTERN_LLM_API_KEY")
        or env_from_files.get("JOB_LLM_API_KEY")
    )
    base_url = (
        __import__("os").environ.get("OPENAI_BASE_URL")
        or env_from_files.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    model = (
        __import__("os").environ.get("PATTERN_STATS_EMBED_MODEL")
        or env_from_files.get("PATTERN_STATS_EMBED_MODEL")
        or "text-embedding-3-small"
    )
    api_key_text = str(api_key or "").strip()
    return (api_key_text if api_key_text else None, str(base_url).rstrip("/"), str(model).strip())


def embedding_cache_dir(project_root: Path) -> Path:
    return project_root / "state" / "statistics_embedding_cache"


def embedding_cache_path(*, project_root: Path, model: str, code_hash: str) -> Path:
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(model or "").strip()) or "model"
    return embedding_cache_dir(project_root) / safe_model / f"{code_hash}.json"


def load_cached_embedding(*, project_root: Path, model: str, code_hash: str) -> list[float] | None:
    path = embedding_cache_path(project_root=project_root, model=model, code_hash=code_hash)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw_embedding = payload.get("embedding") if isinstance(payload, dict) else None
    if not isinstance(raw_embedding, list):
        return None
    vector: list[float] = []
    for value in raw_embedding:
        try:
            vector.append(float(value))
        except (TypeError, ValueError):
            return None
    return vector if vector else None


def store_cached_embedding(
    *,
    project_root: Path,
    model: str,
    code_hash: str,
    embedding: list[float],
) -> None:
    path = embedding_cache_path(project_root=project_root, model=model, code_hash=code_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "code_hash": code_hash,
        "embedding": [float(item) for item in embedding],
    }
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")


def fetch_openai_embeddings(
    *,
    inputs: list[str],
    cache_keys: list[str] | None,
    project_root: Path,
    batch_size: int = 64,
) -> list[list[float]] | None:
    if not inputs:
        return []
    api_key, base_url, model = resolve_openai_embedding_config(project_root)
    if not api_key:
        return None
    if cache_keys is not None and len(cache_keys) != len(inputs):
        return None
    output: list[list[float] | None] = [None] * len(inputs)
    missing_indices: list[int] = []
    if cache_keys is not None:
        for index, code_hash in enumerate(cache_keys):
            cached = load_cached_embedding(project_root=project_root, model=model, code_hash=str(code_hash))
            if cached is not None:
                output[index] = cached
            else:
                missing_indices.append(index)
    else:
        missing_indices = list(range(len(inputs)))

    batch_width = max(1, int(batch_size))
    for offset in range(0, len(missing_indices), batch_width):
        batch_indices = missing_indices[offset : offset + batch_width]
        request_body = json.dumps(
            {"model": model, "input": [inputs[index] for index in batch_indices]}
        ).encode("utf-8")
        request = urllib_request.Request(
            f"{base_url}/embeddings",
            data=request_body,
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError, json.JSONDecodeError):
            return None
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != len(batch_indices):
            return None
        for batch_position, item in enumerate(data):
            raw_embedding = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(raw_embedding, list):
                return None
            vector: list[float] = []
            for value in raw_embedding:
                try:
                    vector.append(float(value))
                except (TypeError, ValueError):
                    return None
            output_index = batch_indices[batch_position]
            output[output_index] = vector
            if cache_keys is not None:
                store_cached_embedding(
                    project_root=project_root,
                    model=model,
                    code_hash=str(cache_keys[output_index]),
                    embedding=vector,
                )
    if any(item is None for item in output):
        return None
    return [item for item in output if item is not None]


def code_hash_for_text(source_code: str) -> str:
    return __import__("hashlib").sha256(str(source_code).encode("utf-8")).hexdigest()


def normalize_embedding(vector: list[float]) -> list[float]:
    total = sum(float(item) * float(item) for item in vector)
    if total <= 0.0:
        return [float(item) for item in vector]
    scale = total ** 0.5
    return [float(item) / scale for item in vector]
