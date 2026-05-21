from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from deepphy_eval.io import load_trace
from pattern_library_detectors import apply_pattern_library
from pattern_library_schema import DetectorDefinition, PatternLibrary


matplotlib.use("Agg")

SUPPORTED_ENVS = {"phyre", "iphyre", "kinetix", "pooltool"}
DEFAULT_WINDOW_BIN_COUNT = 10
DEFAULT_SAMPLING_REPEATS = 64
MAX_SAMPLING_REPEATS = 32
CODE_SAMPLE_PER_LABEL = 8
TOP_K_CANDIDATES_PER_LABEL = 12
MIN_LABEL_TRAJECTORY_CANDIDATES = 25
SUMMARY_FILENAME = "summary.json"
MATRICES_FILENAME = "matrices.json"
RESUME_STATE_FILENAME = "resume_state.json"
SCHEMA_VERSION = "supervised-consistency.v7"
RESUME_SCHEMA_VERSION = "supervised-consistency.resume.v7"


@dataclass(frozen=True)
class LabelSpec:
    label_index: int
    label: str
    detector_id: str
    description: str | None


@dataclass(frozen=True)
class CandidateArtifact:
    uid: str
    label_index: int
    label: str
    candidate_dir: Path
    candidate_payload: dict[str, Any]
    source_code: str
    detector_payload: dict[str, Any]


@dataclass(frozen=True)
class CandidateRecord:
    ordinal: int
    label_index: int
    label: str
    candidate_uid: str
    score: float
    artifact: CandidateArtifact | None


@dataclass(frozen=True)
class RunState:
    run_dir: Path
    run_name: str
    label_specs: list[LabelSpec]
    candidate_records: list[CandidateRecord]
    total_candidate_count: int
    final_detectors_by_label: dict[int, DetectorDefinition]


@dataclass(frozen=True)
class CodeSample:
    label_index: int
    label: str
    candidate_uid: str
    code_hash: str
    source_code: str


@dataclass(frozen=True)
class LabelTrajectory:
    trajectory_id: str
    run_name: str
    run_dir: Path
    label_index: int
    label: str
    candidate_records: list[CandidateRecord]
    final_detector: DetectorDefinition


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log_event(event: str, **fields: object) -> None:
    bits = [f"{key}={_format_log_value(value)}" for key, value in fields.items() if value is not None]
    suffix = f": {' '.join(bits)}" if bits else ""
    print(f"[INFO] patterns.analysis.consistency.{event}{suffix}", flush=True)


def _format_log_value(value: object) -> str:
    text = str(value)
    if not text:
        return '""'
    if any(ch.isspace() for ch in text):
        return json.dumps(text)
    return text


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _slugify(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value))
    compact = "-".join(part for part in text.split("-") if part)
    return compact or "item"


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _label_spec_from_dir(label_dir: Path) -> LabelSpec | None:
    payload = _load_json_object(label_dir / "label_result.json")
    detector = payload.get("detector") if isinstance(payload.get("detector"), dict) else {}
    label_index = _coerce_int(payload.get("label_index"))
    label = str(payload.get("label") or "").strip()
    detector_id = str(detector.get("detector_id") or "").strip()
    description = str(detector.get("description") or "").strip() or None
    if label_index is None:
        parts = label_dir.name.split("_", 2)
        if len(parts) >= 2 and parts[1].isdigit():
            label_index = int(parts[1])
    if not label:
        parts = label_dir.name.split("_", 2)
        label = parts[2].replace("_", " ") if len(parts) >= 3 else label_dir.name
    if label_index is None or not label:
        return None
    if not detector_id:
        detector_id = f"primitive-{label_index:03d}"
    return LabelSpec(
        label_index=label_index,
        label=label,
        detector_id=detector_id,
        description=description,
    )


def _candidate_artifacts_by_uid(run_dir: Path) -> dict[str, CandidateArtifact]:
    artifacts: dict[str, CandidateArtifact] = {}
    for label_dir in sorted(path for path in run_dir.glob("label_*") if path.is_dir()):
        label_spec = _label_spec_from_dir(label_dir)
        if label_spec is None:
            continue
        detector_payload = (
            _load_json_object(label_dir / "label_result.json").get("detector")
            if (label_dir / "label_result.json").is_file()
            else {}
        )
        if not isinstance(detector_payload, dict):
            detector_payload = {}
        for candidate_dir in sorted(path for path in label_dir.glob("candidate_*") if path.is_dir()):
            candidate_payload = _load_json_object(candidate_dir / "candidate.json")
            candidate_uid = str(candidate_payload.get("uid") or "").strip()
            if not candidate_uid:
                continue
            source_path = next((path for path in sorted(candidate_dir.glob("*.py")) if path.is_file()), None)
            if source_path is None:
                continue
            try:
                source_code = source_path.read_text(encoding="utf-8")
            except OSError:
                continue
            artifacts[candidate_uid] = CandidateArtifact(
                uid=candidate_uid,
                label_index=label_spec.label_index,
                label=label_spec.label,
                candidate_dir=candidate_dir,
                candidate_payload=candidate_payload,
                source_code=source_code,
                detector_payload=detector_payload,
            )
    return artifacts


def _candidate_records_for_run(run_dir: Path) -> list[CandidateRecord]:
    artifacts = _candidate_artifacts_by_uid(run_dir)
    seen: set[tuple[int, str, int | None]] = set()
    records: list[CandidateRecord] = []
    ordinal = 0
    for path in (run_dir / "evolution_record.jsonl", run_dir / "evolution_live.jsonl"):
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for raw_line in lines:
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            candidate_uid = str(payload.get("candidate_uid") or "").strip()
            label_index = _coerce_int(payload.get("label_index"))
            label = str(payload.get("label") or "").strip()
            iteration = _coerce_int(payload.get("iteration"))
            if not candidate_uid or label_index is None or not label:
                continue
            dedupe_key = (label_index, candidate_uid, iteration)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            ordinal += 1
            score = _coerce_float(payload.get("score"))
            if score is None:
                score = _coerce_float(payload.get("train_score"))
            if score is None:
                score = _coerce_float(payload.get("val_score"))
            records.append(
                CandidateRecord(
                    ordinal=ordinal,
                    label_index=label_index,
                    label=label,
                    candidate_uid=candidate_uid,
                    score=float(score or 0.0),
                    artifact=artifacts.get(candidate_uid),
                )
            )
    return records


def _final_detectors_for_run(run_dir: Path, *, environment_id: str) -> dict[int, DetectorDefinition]:
    detectors: dict[int, DetectorDefinition] = {}
    for label_dir in sorted(path for path in run_dir.glob("label_*") if path.is_dir()):
        payload = _load_json_object(label_dir / "label_result.json")
        label_index = _coerce_int(payload.get("label_index"))
        detector_payload = payload.get("detector") if isinstance(payload.get("detector"), dict) else None
        if label_index is None or not isinstance(detector_payload, dict):
            continue
        environment_ids = detector_payload.get("environment_ids")
        if not isinstance(environment_ids, list) or not environment_ids:
            environment_ids = [environment_id]
        detectors[label_index] = DetectorDefinition(
            detector_id=str(detector_payload.get("detector_id") or f"primitive-{label_index:03d}"),
            label=str(detector_payload.get("label") or payload.get("label") or f"label_{label_index}"),
            description=str(detector_payload.get("description") or "").strip() or None,
            environment_ids=[str(item) for item in environment_ids if str(item).strip()],
            source_code=str(detector_payload.get("source_code") or ""),
            entrypoint=str(detector_payload.get("entrypoint") or "find_pattern"),
            input_mode="trace",
            config=dict(detector_payload.get("config")) if isinstance(detector_payload.get("config"), dict) else {},
            metadata=dict(detector_payload.get("metadata")) if isinstance(detector_payload.get("metadata"), dict) else {},
        )
    return detectors


def _load_run_state(run_dir: Path, *, environment_id: str) -> RunState:
    resolved = run_dir.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Run directory not found: {resolved}")
    label_specs = [
        spec
        for spec in (_label_spec_from_dir(path) for path in sorted(resolved.glob("label_*")))
        if spec is not None
    ]
    if not label_specs:
        raise ValueError(f"No supervised label artifacts found under {resolved}")
    candidate_records = _candidate_records_for_run(resolved)
    return RunState(
        run_dir=resolved,
        run_name=resolved.name,
        label_specs=sorted(label_specs, key=lambda item: (item.label_index, item.label)),
        candidate_records=candidate_records,
        total_candidate_count=len(candidate_records),
        final_detectors_by_label=_final_detectors_for_run(resolved, environment_id=environment_id),
    )


def _canonical_labels(run_states: list[RunState]) -> list[LabelSpec]:
    merged: dict[int, LabelSpec] = {}
    for state in run_states:
        for spec in state.label_specs:
            merged.setdefault(spec.label_index, spec)
    return [merged[index] for index in sorted(merged)]


def _detector_from_candidate(artifact: CandidateArtifact, *, environment_id: str) -> DetectorDefinition:
    detector_payload = dict(artifact.detector_payload)
    candidate_metadata = (
        dict(artifact.candidate_payload.get("metadata"))
        if isinstance(artifact.candidate_payload.get("metadata"), dict)
        else {}
    )
    metadata = (
        dict(detector_payload.get("metadata"))
        if isinstance(detector_payload.get("metadata"), dict)
        else {}
    )
    metadata.update(candidate_metadata)
    metadata.update(
        {
            "candidate_uid": artifact.uid,
            "candidate_dir": str(artifact.candidate_dir),
            "score": artifact.candidate_payload.get("score"),
            "train_score": artifact.candidate_payload.get("train_score"),
            "val_score": artifact.candidate_payload.get("val_score"),
        }
    )
    environment_ids = detector_payload.get("environment_ids")
    if not isinstance(environment_ids, list) or not environment_ids:
        environment_ids = [environment_id]
    return DetectorDefinition(
        detector_id=str(detector_payload.get("detector_id") or f"primitive-{artifact.label_index:03d}"),
        label=str(detector_payload.get("label") or artifact.label),
        description=str(detector_payload.get("description") or "").strip() or None,
        environment_ids=[str(item) for item in environment_ids if str(item).strip()],
        source_code=artifact.source_code,
        entrypoint=str(detector_payload.get("entrypoint") or "find_pattern"),
        input_mode="trace",
        config=dict(detector_payload.get("config")) if isinstance(detector_payload.get("config"), dict) else {},
        metadata=metadata,
    )


def _window_bounds(*, total_candidates: int, window_index: int, window_count: int) -> tuple[int, int]:
    total = max(0, int(total_candidates))
    count = max(1, int(window_count))
    index = max(0, min(int(window_index), count - 1))
    start = int(math.floor((index / count) * total)) + 1
    end = int(math.floor(((index + 1) / count) * total))
    return start, end


def _window_label(window_index: int, window_count: int) -> str:
    count = max(1, int(window_count))
    index = max(0, min(int(window_index), count - 1))
    lower = int(round((index / count) * 100.0))
    upper = int(round(((index + 1) / count) * 100.0))
    return f"{lower}-{upper}%"


def _analysis_project_root(output_dir: Path) -> Path:
    resolved = output_dir.resolve()
    for parent in (resolved, *resolved.parents):
        if (parent / "state").is_dir() and (parent / "data").is_dir():
            return parent
    if len(resolved.parents) >= 3:
        return resolved.parents[2]
    return resolved.parent


def _resolve_openai_embedding_config(project_root: Path) -> tuple[str | None, str, str]:
    env_from_files: dict[str, str] = {}
    for candidate in (project_root / ".env", project_root.parent / ".env"):
        env_from_files.update(_parse_dotenv(candidate))
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("PATTERN_LLM_API_KEY")
        or os.environ.get("JOB_LLM_API_KEY")
        or env_from_files.get("OPENAI_API_KEY")
        or env_from_files.get("PATTERN_LLM_API_KEY")
        or env_from_files.get("JOB_LLM_API_KEY")
    )
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or env_from_files.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    model = (
        os.environ.get("PATTERN_STATS_EMBED_MODEL")
        or env_from_files.get("PATTERN_STATS_EMBED_MODEL")
        or "text-embedding-3-small"
    )
    key_text = str(api_key or "").strip()
    return (key_text if key_text else None, str(base_url).rstrip("/"), str(model).strip())


def _embedding_cache_dir(project_root: Path) -> Path:
    return project_root / "state" / "statistics_embedding_cache"


def _embedding_cache_path(*, project_root: Path, model: str, code_hash: str) -> Path:
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(model or "").strip()) or "model"
    return _embedding_cache_dir(project_root) / safe_model / f"{code_hash}.json"


def _load_cached_embedding(*, project_root: Path, model: str, code_hash: str) -> list[float] | None:
    path = _embedding_cache_path(project_root=project_root, model=model, code_hash=code_hash)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    vector_raw = payload.get("embedding") if isinstance(payload, dict) else None
    if not isinstance(vector_raw, list):
        return None
    vector: list[float] = []
    for value in vector_raw:
        numeric = _safe_float(value)
        if numeric is None:
            return None
        vector.append(float(numeric))
    return vector if vector else None


def _store_cached_embedding(
    *,
    project_root: Path,
    model: str,
    code_hash: str,
    embedding: list[float],
) -> None:
    path = _embedding_cache_path(project_root=project_root, model=model, code_hash=code_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "code_hash": code_hash, "embedding": embedding}
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")


def _fetch_openai_embeddings(
    *,
    inputs: list[str],
    cache_keys: list[str] | None,
    project_root: Path,
    batch_size: int = 64,
) -> list[list[float]] | None:
    if not inputs:
        return []
    api_key, base_url, model = _resolve_openai_embedding_config(project_root)
    if not api_key:
        return None
    if cache_keys is not None and len(cache_keys) != len(inputs):
        return None
    output: list[list[float] | None] = [None] * len(inputs)
    missing_indices: list[int] = []
    if cache_keys is not None:
        for index, code_hash in enumerate(cache_keys):
            cached_vector = _load_cached_embedding(project_root=project_root, model=model, code_hash=str(code_hash))
            if cached_vector is not None:
                output[index] = cached_vector
            else:
                missing_indices.append(index)
    else:
        missing_indices = list(range(len(inputs)))

    for offset in range(0, len(missing_indices), max(1, int(batch_size))):
        batch_indices = missing_indices[offset:offset + max(1, int(batch_size))]
        body = json.dumps({"model": model, "input": [inputs[item] for item in batch_indices]}).encode("utf-8")
        request = urllib_request.Request(
            f"{base_url}/embeddings",
            data=body,
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
        for position, item in enumerate(data):
            embedding = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(embedding, list):
                return None
            vector: list[float] = []
            for value in embedding:
                numeric = _safe_float(value)
                if numeric is None:
                    return None
                vector.append(float(numeric))
            index = batch_indices[position]
            output[index] = vector
            if cache_keys is not None:
                _store_cached_embedding(
                    project_root=project_root,
                    model=model,
                    code_hash=str(cache_keys[index]),
                    embedding=vector,
                )
    if any(item is None for item in output):
        return None
    return [item for item in output if item is not None]


def _cosine_distance(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 1.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for av, bv in zip(a, b):
        dot += av * bv
        norm_a += av * av
        norm_b += bv * bv
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 1.0
    similarity = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    similarity = max(-1.0, min(1.0, similarity))
    return max(0.0, min(2.0, 1.0 - similarity))


def _mean_pairwise_distance(vectors: list[list[float]]) -> float:
    if len(vectors) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            total += _cosine_distance(vectors[left], vectors[right])
            pairs += 1
    return total / float(max(1, pairs))


def _new_stats_grid(size: int) -> dict[str, np.ndarray]:
    shape = (int(size), int(size))
    return {
        "count": np.zeros(shape, dtype=np.int32),
        "sum": np.zeros(shape, dtype=np.float64),
        "sum_sq": np.zeros(shape, dtype=np.float64),
    }


def _new_stats_vector(size: int) -> dict[str, np.ndarray]:
    length = int(size)
    return {
        "count": np.zeros(length, dtype=np.int32),
        "sum": np.zeros(length, dtype=np.float64),
        "sum_sq": np.zeros(length, dtype=np.float64),
    }


def _stats_grid_add(stats: dict[str, np.ndarray], row: int, col: int, value: float) -> None:
    numeric = float(value)
    stats["count"][row, col] += 1
    stats["sum"][row, col] += numeric
    stats["sum_sq"][row, col] += numeric * numeric


def _stats_vector_add(stats: dict[str, np.ndarray], index: int, value: float) -> None:
    numeric = float(value)
    stats["count"][index] += 1
    stats["sum"][index] += numeric
    stats["sum_sq"][index] += numeric * numeric


def _stats_grid_mean(stats: dict[str, np.ndarray]) -> np.ndarray:
    count = np.maximum(stats["count"].astype(np.float64), 1.0)
    return np.divide(stats["sum"], count, dtype=np.float64)


def _stats_grid_std(stats: dict[str, np.ndarray]) -> np.ndarray:
    count = stats["count"].astype(np.float64)
    safe_count = np.maximum(count, 1.0)
    mean = np.divide(stats["sum"], safe_count, dtype=np.float64)
    variance = np.divide(stats["sum_sq"], safe_count, dtype=np.float64) - np.square(mean)
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance, dtype=np.float64)
    std[count <= 0] = 0.0
    return std


def _stats_vector_mean(stats: dict[str, np.ndarray]) -> np.ndarray:
    count = np.maximum(stats["count"].astype(np.float64), 1.0)
    return np.divide(stats["sum"], count, dtype=np.float64)


def _stats_vector_std(stats: dict[str, np.ndarray]) -> np.ndarray:
    count = stats["count"].astype(np.float64)
    safe_count = np.maximum(count, 1.0)
    mean = np.divide(stats["sum"], safe_count, dtype=np.float64)
    variance = np.divide(stats["sum_sq"], safe_count, dtype=np.float64) - np.square(mean)
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance, dtype=np.float64)
    std[count <= 0] = 0.0
    return std


def _stats_grid_cell_mean(stats: dict[str, np.ndarray], row: int, col: int) -> float:
    count = int(stats["count"][row, col])
    if count <= 0:
        return 0.0
    return float(stats["sum"][row, col] / float(count))


def _stats_grid_cell_std(stats: dict[str, np.ndarray], row: int, col: int) -> float:
    count = int(stats["count"][row, col])
    if count <= 0:
        return 0.0
    mean = float(stats["sum"][row, col] / float(count))
    variance = float(stats["sum_sq"][row, col] / float(count)) - (mean * mean)
    return math.sqrt(max(0.0, variance))


def _stats_vector_item_mean(stats: dict[str, np.ndarray], index: int) -> float:
    count = int(stats["count"][index])
    if count <= 0:
        return 0.0
    return float(stats["sum"][index] / float(count))


def _stats_vector_item_std(stats: dict[str, np.ndarray], index: int) -> float:
    count = int(stats["count"][index])
    if count <= 0:
        return 0.0
    mean = float(stats["sum"][index] / float(count))
    variance = float(stats["sum_sq"][index] / float(count)) - (mean * mean)
    return math.sqrt(max(0.0, variance))


def _serialize_stats_payload(stats: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        "count": stats["count"].tolist(),
        "sum": stats["sum"].tolist(),
        "sum_sq": stats["sum_sq"].tolist(),
    }


def _deserialize_stats_grid(payload: dict[str, Any], size: int) -> dict[str, np.ndarray]:
    stats = _new_stats_grid(size)
    for key, dtype in (("count", np.int32), ("sum", np.float64), ("sum_sq", np.float64)):
        raw = payload.get(key)
        if raw is None:
            continue
        array = np.asarray(raw, dtype=dtype)
        if array.shape == (int(size), int(size)):
            stats[key] = array
    return stats


def _deserialize_stats_vector(payload: dict[str, Any], size: int) -> dict[str, np.ndarray]:
    stats = _new_stats_vector(size)
    for key, dtype in (("count", np.int32), ("sum", np.float64), ("sum_sq", np.float64)):
        raw = payload.get(key)
        if raw is None:
            continue
        array = np.asarray(raw, dtype=dtype)
        if array.shape == (int(size),):
            stats[key] = array
    return stats


def _pair_key(left_run: str, right_run: str) -> str:
    return f"{left_run}|||{right_run}"


def _resume_state_path(output_dir: Path) -> Path:
    return output_dir / RESUME_STATE_FILENAME


def _write_resume_state(output_dir: Path, payload: dict[str, object]) -> None:
    path = _resume_state_path(output_dir)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _load_resume_state(
    *,
    output_dir: Path,
    env_id: str,
    run_dirs: list[Path],
    trace_paths: list[Path],
    sampling_seed: int,
    total_repeats: int,
    window_count: int,
    run_names: list[str],
) -> dict[str, Any] | None:
    path = _resume_state_path(output_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("schema_version") or "").strip() != RESUME_SCHEMA_VERSION:
        return None
    if str(payload.get("env_id") or "").strip().lower() != str(env_id or "").strip().lower():
        return None
    expected_run_dirs = [str(Path(path).resolve()) for path in run_dirs]
    expected_trace_paths = [str(Path(path).resolve()) for path in trace_paths]
    if list(payload.get("run_dirs") or []) != expected_run_dirs:
        return None
    if list(payload.get("trace_paths") or []) != expected_trace_paths:
        return None
    if int(payload.get("sampling_seed") or 0) != int(sampling_seed):
        return None
    if int(payload.get("total_repeats") or 0) != int(total_repeats):
        return None
    if int(payload.get("window_count") or 0) != int(window_count):
        return None
    if list(payload.get("run_names") or []) != list(run_names):
        return None
    return payload


def _library_snapshot_for_window(
    state: RunState,
    *,
    environment_id: str,
    window_index: int,
    window_count: int,
    rng: random.Random,
) -> PatternLibrary:
    start_ordinal, end_ordinal = _window_bounds(
        total_candidates=state.total_candidate_count,
        window_index=window_index,
        window_count=window_count,
    )
    candidate_pool_by_label = _candidate_records_by_label_for_window(
        state,
        start_ordinal=start_ordinal,
        end_ordinal=end_ordinal,
    )
    sampled_by_label: dict[int, CandidateRecord] = {}
    for label_index, records in sorted(candidate_pool_by_label.items()):
        sampled_by_label[label_index] = records[rng.randrange(len(records))]
    detectors = [
        _detector_from_candidate(record.artifact, environment_id=environment_id)
        for _, record in sorted(sampled_by_label.items())
    ]
    return PatternLibrary(
        library_id=f"{state.run_name}-window-{window_index + 1:02d}",
        environment_ids=[environment_id],
        primitive_source="supervised",
        detectors=detectors,
        metadata={
            "source_run_dir": str(state.run_dir),
            "window_index": int(window_index),
            "window_count": int(window_count),
            "window_label": _window_label(window_index, window_count),
            "start_ordinal": int(start_ordinal),
            "end_ordinal": int(end_ordinal),
            "selected_detector_count": len(detectors),
            "sampling_mode": "monte_carlo",
        },
    )


def _candidate_records_by_label_for_window(
    state: RunState,
    *,
    start_ordinal: int,
    end_ordinal: int,
) -> dict[int, list[CandidateRecord]]:
    candidate_pool_by_label: dict[int, list[CandidateRecord]] = {}
    for record in state.candidate_records:
        if record.ordinal < start_ordinal or record.ordinal > end_ordinal:
            continue
        if record.artifact is None:
            continue
        candidate_pool_by_label.setdefault(record.label_index, []).append(record)
    return candidate_pool_by_label


def _top_candidate_records_by_label_for_window(
    state: RunState,
    *,
    window_index: int,
    window_count: int,
    top_k: int = TOP_K_CANDIDATES_PER_LABEL,
) -> dict[int, list[CandidateRecord]]:
    start_ordinal, end_ordinal = _window_bounds(
        total_candidates=state.total_candidate_count,
        window_index=window_index,
        window_count=window_count,
    )
    candidate_pool_by_label = _candidate_records_by_label_for_window(
        state,
        start_ordinal=start_ordinal,
        end_ordinal=end_ordinal,
    )
    limited: dict[int, list[CandidateRecord]] = {}
    for label_index, records in sorted(candidate_pool_by_label.items()):
        ranked = sorted(
            records,
            key=lambda item: (float(item.score), int(item.ordinal), str(item.candidate_uid)),
            reverse=True,
        )
        selected = [record for record in ranked[: max(1, int(top_k))] if record.artifact is not None]
        if selected:
            limited[label_index] = selected
    return limited


def _sample_code_candidates_for_window(
    state: RunState,
    *,
    window_index: int,
    window_count: int,
    sample_size: int = CODE_SAMPLE_PER_LABEL,
) -> dict[int, list[CodeSample]]:
    candidate_pool_by_label = _top_candidate_records_by_label_for_window(
        state,
        window_index=window_index,
        window_count=window_count,
        top_k=sample_size,
    )
    sampled: dict[int, list[CodeSample]] = {}
    for label_index, records in sorted(candidate_pool_by_label.items()):
        code_samples: list[CodeSample] = []
        for record in records:
            if record.artifact is None:
                continue
            source_code = str(record.artifact.source_code or "").strip()
            if not source_code:
                continue
            code_samples.append(
                CodeSample(
                    label_index=record.label_index,
                    label=record.label,
                    candidate_uid=record.candidate_uid,
                    code_hash=hashlib.sha256(source_code.encode("utf-8")).hexdigest(),
                    source_code=source_code,
                )
            )
        if code_samples:
            sampled[label_index] = code_samples
    return sampled


def _top_candidate_records_for_label_window(
    candidate_records: list[CandidateRecord],
    *,
    window_index: int,
    window_count: int,
    top_k: int = TOP_K_CANDIDATES_PER_LABEL,
) -> list[CandidateRecord]:
    total_candidates = len(candidate_records)
    if total_candidates <= 0:
        return []
    start_ordinal, end_ordinal = _window_bounds(
        total_candidates=total_candidates,
        window_index=window_index,
        window_count=window_count,
    )
    if end_ordinal < start_ordinal:
        return []
    selected_pool = candidate_records[max(0, start_ordinal - 1) : max(0, end_ordinal)]
    if not selected_pool:
        return []
    ranked = sorted(
        selected_pool,
        key=lambda item: (float(item.score), int(item.ordinal), str(item.candidate_uid)),
        reverse=True,
    )
    return [record for record in ranked[: max(1, int(top_k))] if record.artifact is not None]


def _code_samples_for_candidate_records(candidate_records: list[CandidateRecord]) -> list[CodeSample]:
    samples: list[CodeSample] = []
    for record in candidate_records:
        if record.artifact is None:
            continue
        source_code = str(record.artifact.source_code or "").strip()
        if not source_code:
            continue
        samples.append(
            CodeSample(
                label_index=record.label_index,
                label=record.label,
                candidate_uid=record.candidate_uid,
                code_hash=hashlib.sha256(source_code.encode("utf-8")).hexdigest(),
                source_code=source_code,
            )
        )
    return samples


def _single_detector_library(
    detector: DetectorDefinition,
    *,
    environment_id: str,
    library_id: str,
) -> PatternLibrary:
    return PatternLibrary(
        library_id=library_id,
        environment_ids=[environment_id],
        primitive_source="supervised",
        detectors=[detector],
        metadata={"selected_detector_count": 1, "sampling_mode": "top_k"},
    )


def _annotation_matrix_for_detector(
    *,
    trace,
    detector: DetectorDefinition,
    label_spec: LabelSpec,
    environment_id: str,
    library_id: str,
) -> np.ndarray:
    library = _single_detector_library(detector, environment_id=environment_id, library_id=library_id)
    return _annotation_matrix(trace=trace, library=library, label_order=[label_spec])


def _mean_matrix(matrices: list[np.ndarray]) -> np.ndarray:
    if not matrices:
        return np.zeros((0, 0), dtype=np.float32)
    stacked = np.stack([matrix.astype(np.float32) for matrix in matrices], axis=0)
    return np.mean(stacked, axis=0).astype(np.float32)


def _normalized_code_convergence(distance: float) -> float:
    return max(0.0, min(1.0, 1.0 - (max(0.0, float(distance)) / 2.0)))


def _annotation_matrix(
    *,
    trace,
    library: PatternLibrary,
    label_order: list[LabelSpec],
) -> np.ndarray:
    timestep_count = len(trace.timesteps)
    label_index_by_name = {spec.label: index for index, spec in enumerate(label_order)}
    matrix = np.zeros((timestep_count, len(label_order)), dtype=np.float32)
    for pattern in apply_pattern_library(trace, library):
        label_position = label_index_by_name.get(pattern.label)
        if label_position is None:
            continue
        start = max(0, min(int(pattern.start_timestep), timestep_count - 1))
        end = max(0, min(int(pattern.end_timestep), timestep_count - 1))
        if end < start:
            start, end = end, start
        matrix[start : end + 1, label_position] = 1.0
    return matrix


def _matrix_distance(left: np.ndarray, right: np.ndarray) -> float:
    total_cells = float(left.size or 1)
    diff = float(np.not_equal(left, right).sum())
    return diff / total_cells


def _per_label_matrix_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    if left.ndim != 2 or right.ndim != 2:
        return _matrix_distance(left, right)
    label_count = min(int(left.shape[1]), int(right.shape[1]))
    if label_count <= 0:
        return 0.0
    aligned_left = left[:, :label_count]
    aligned_right = right[:, :label_count]
    per_label_diffs = np.not_equal(aligned_left, aligned_right).mean(axis=0)
    return float(np.mean(per_label_diffs)) if per_label_diffs.size else 0.0


def _render_metric_plot(
    window_summaries: list[dict[str, object]],
    *,
    similarity_key: str,
    distance_key: str,
    title: str,
    output_path: Path,
) -> None:
    bins = [int(item["window_index"]) + 1 for item in window_summaries]
    similarities = [float(item.get(similarity_key) or 0.0) for item in window_summaries]
    distances = [float(item.get(distance_key) or 0.0) for item in window_summaries]
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.plot(bins, similarities, marker="o", label="Mean similarity")
    axis.plot(bins, distances, marker="s", label="Mean distance")
    axis.set_xlabel("Window bin")
    axis.set_ylabel("Score")
    axis.set_title(title)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend()
    axis.set_xticks(bins)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _render_distance_grid(
    grid: np.ndarray,
    *,
    title: str,
    labels: list[str],
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7.6, 6.2))
    image = axis.imshow(grid, cmap="viridis", vmin=0.0, vmax=max(1e-6, float(grid.max(initial=0.0))))
    axis.set_title(title)
    axis.set_xlabel("Right run window")
    axis.set_ylabel("Left run window")
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axis.set_yticklabels(labels, fontsize=8)
    for row in range(len(labels)):
        for col in range(len(labels)):
            axis.text(col, row, f"{grid[row, col]:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _render_distance_histogram(pairwise: list[dict[str, object]], *, output_path: Path) -> None:
    values = [float(item["distance"]) for item in pairwise]
    fig, axis = plt.subplots(figsize=(6.0, 4.0))
    axis.hist(values, bins=min(10, max(3, len(values))), color="#315c72", edgecolor="#111111")
    axis.set_title("Pairwise distances across all run-pair windows")
    axis.set_xlabel("Normalized Hamming distance")
    axis.set_ylabel("Run-pair windows")
    axis.set_xlim(0.0, 1.0)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _render_run_pair_matrix(
    run_names: list[str],
    matrix: np.ndarray,
    *,
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(5.2, 4.6))
    image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=max(1e-6, float(matrix.max(initial=0.0))))
    axis.set_title("Mean run-pair distance across all windows")
    axis.set_xticks(range(len(run_names)))
    axis.set_yticks(range(len(run_names)))
    axis.set_xticklabels(run_names, rotation=45, ha="right", fontsize=8)
    axis.set_yticklabels(run_names, fontsize=8)
    for row in range(len(run_names)):
        for col in range(len(run_names)):
            axis.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _render_snapshot_embedding(
    snapshot_rows: list[dict[str, object]],
    *,
    output_path: Path,
) -> None:
    def _short_run_label(raw_name: str) -> str:
        text = str(raw_name or "").strip()
        if not text:
            return "run"
        if ".__resume_sync__" in text:
            text = text.split(".__resume_sync__", 1)[0]
        if len(text) > 28:
            return f"{text[:25]}..."
        return text

    if len(snapshot_rows) < 2:
        fig, axis = plt.subplots(figsize=(6.0, 4.0))
        axis.text(0.5, 0.5, "Need at least two snapshots for PCA/SVD view", ha="center", va="center")
        axis.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return
    matrix = np.stack([np.asarray(item["flattened_matrix"], dtype=np.float32) for item in snapshot_rows], axis=0)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    try:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vt[:2].T
    except np.linalg.LinAlgError:
        coords = np.zeros((len(snapshot_rows), 2), dtype=np.float32)
    bins = [int(item["window_index"]) for item in snapshot_rows]
    unique_bins = sorted(set(bins))
    color_lookup = {window_index: plt.cm.viridis(index / max(1, len(unique_bins) - 1)) for index, window_index in enumerate(unique_bins)}
    fig, axis = plt.subplots(figsize=(7.2, 5.0))
    for index, item in enumerate(snapshot_rows):
        x, y = coords[index]
        window_index = int(item["window_index"])
        window_label = str(item.get("window_label") or f"bin {window_index + 1}")
        run_name = _short_run_label(str(item["run_name"]))
        axis.scatter([x], [y], color=color_lookup[window_index], s=44)
        axis.text(x, y, f"{run_name}\n{window_label}", fontsize=7, ha="left", va="bottom")
    axis.set_title("Snapshot embedding across runs and windows")
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_supervised_consistency_analysis(
    *,
    env_id: str,
    run_dirs: list[Path],
    trace_paths: list[Path],
    output_dir: Path,
    checkpoints: list[int],
    sampling_repeats: int = DEFAULT_SAMPLING_REPEATS,
    sampling_seed: int = 42,
) -> dict[str, Any]:
    resolved_env = str(env_id or "").strip().lower()
    if resolved_env not in SUPPORTED_ENVS:
        raise ValueError(f"Unsupported environment: {env_id}")
    if not run_dirs:
        raise ValueError("At least one supervised run directory is required")
    if not trace_paths:
        raise ValueError("At least one trace path is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_run_id = output_dir.name
    requested_repeats = max(1, int(sampling_repeats))
    total_repeats = min(requested_repeats, MAX_SAMPLING_REPEATS)

    resolved_trace_paths = [Path(path).resolve() for path in trace_paths]
    traces = [load_trace(path) for path in resolved_trace_paths]
    for trace in traces:
        if trace.environment != resolved_env:
            raise ValueError(f"Trace environment {trace.environment} does not match requested env {resolved_env}")
    reference_trace = traces[0]
    run_states = [_load_run_state(path, environment_id=resolved_env) for path in run_dirs]
    for state in run_states:
        _log_event("run.loaded", run_name=state.run_name, candidates=state.total_candidate_count, labels=len(state.label_specs))
    canonical_labels = _canonical_labels(run_states)
    if not canonical_labels:
        raise ValueError("No label set could be resolved from the selected runs")
    if checkpoints:
        _log_event("checkpoints.ignored", reason="window-bin mode uses fixed percentile windows")
    label_spec_by_index = {spec.label_index: spec for spec in canonical_labels}
    trajectories: list[LabelTrajectory] = []
    for state in run_states:
        for label_index, final_detector in sorted(state.final_detectors_by_label.items()):
            label_spec = label_spec_by_index.get(label_index)
            if label_spec is None:
                continue
            label_candidates = [
                record
                for record in state.candidate_records
                if record.label_index == label_index and record.artifact is not None
            ]
            if len(label_candidates) < MIN_LABEL_TRAJECTORY_CANDIDATES:
                continue
            trajectories.append(
                LabelTrajectory(
                    trajectory_id=f"{state.run_name}::{label_index}",
                    run_name=state.run_name,
                    run_dir=state.run_dir,
                    label_index=label_index,
                    label=label_spec.label,
                    candidate_records=label_candidates,
                    final_detector=final_detector,
                )
            )
    if not trajectories:
        raise ValueError("No learned label trajectories were found in the selected runs")
    trajectory_candidate_totals = [len(item.candidate_records) for item in trajectories]
    max_budget = max(trajectory_candidate_totals)
    average_budget = int(round(float(sum(trajectory_candidate_totals)) / float(len(trajectory_candidate_totals))))
    window_count = DEFAULT_WINDOW_BIN_COUNT
    window_labels = [_window_label(index, window_count) for index in range(window_count)]
    project_root = _analysis_project_root(output_dir)
    _log_event(
        "start",
        env_id=resolved_env,
        runs=len(run_dirs),
        trajectories=len(trajectories),
        windows=window_count,
        trace_count=len(traces),
        requested_sampling_repeats=requested_repeats,
        sampling_repeats=total_repeats,
        output_dir=output_dir,
    )

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    matrices_payload: dict[str, object] = {
        "trace_path": str(resolved_trace_paths[0]),
        "trace_paths": [str(path) for path in resolved_trace_paths],
        "trace_count": len(traces),
        "trace_ids": [trace.trace_id for trace in traces],
        "task_ids": [trace.task_id for trace in traces],
        "timesteps": [len(trace.timesteps) for trace in traces],
        "labels": [spec.label for spec in canonical_labels],
        "trajectory_count": len(trajectories),
        "window_count": window_count,
        "window_labels": list(window_labels),
        "windows": {},
    }
    run_names = sorted(state.run_name for state in run_states)
    code_similarity_to_final_bin_stats = _new_stats_vector(window_count)
    pattern_similarity_to_final_bin_stats = _new_stats_vector(window_count)
    code_diversity_stats = _new_stats_vector(window_count)
    trajectory_records_by_window: dict[tuple[str, int], list[CandidateRecord]] = {}
    trajectory_code_samples_by_window: dict[tuple[str, int], list[CodeSample]] = {}
    code_inputs_by_hash: dict[str, str] = {}
    for trajectory in trajectories:
        for window_index in range(window_count):
            top_records = _top_candidate_records_for_label_window(
                trajectory.candidate_records,
                window_index=window_index,
                window_count=window_count,
                top_k=TOP_K_CANDIDATES_PER_LABEL,
            )
            trajectory_records_by_window[(trajectory.trajectory_id, window_index)] = top_records
            code_samples = _code_samples_for_candidate_records(top_records)
            trajectory_code_samples_by_window[(trajectory.trajectory_id, window_index)] = code_samples
            for item in code_samples:
                code_inputs_by_hash.setdefault(item.code_hash, item.source_code)
            start_ordinal, end_ordinal = _window_bounds(
                total_candidates=len(trajectory.candidate_records),
                window_index=window_index,
                window_count=window_count,
            )
            window_payload = matrices_payload["windows"].setdefault(
                str(window_index),
                {
                    "window_index": window_index,
                    "window_label": window_labels[window_index],
                    "trajectories": {},
                },
            )
            trajectory_payload = window_payload["trajectories"]
            if isinstance(trajectory_payload, dict):
                trajectory_payload[trajectory.trajectory_id] = {
                    "run_name": trajectory.run_name,
                    "label_index": trajectory.label_index,
                    "label": trajectory.label,
                    "candidate_count": len(top_records),
                    "start_ordinal": int(start_ordinal),
                    "end_ordinal": int(end_ordinal),
                    "top_k_candidates_per_label": int(TOP_K_CANDIDATES_PER_LABEL),
                }

    code_hashes = sorted(code_inputs_by_hash)
    code_embeddings_by_hash: dict[str, list[float]] = {}
    if code_hashes:
        vectors = _fetch_openai_embeddings(
            inputs=[code_inputs_by_hash[item] for item in code_hashes],
            cache_keys=code_hashes,
            project_root=project_root,
            batch_size=64,
        )
        if vectors is None or len(vectors) != len(code_hashes):
            raise RuntimeError("Unable to compute embeddings for analysis code convergence")
        code_embeddings_by_hash = {code_hashes[index]: vectors[index] for index in range(len(code_hashes))}

    _log_event("repeat.start", repeat=1, total_repeats=1)
    candidate_detector_cache: dict[str, DetectorDefinition] = {}
    candidate_matrix_cache: dict[tuple[str, int], np.ndarray] = {}
    for window_index in range(window_count):
        for trajectory in trajectories:
            current_code_samples = trajectory_code_samples_by_window.get((trajectory.trajectory_id, window_index), [])
            final_code_samples = trajectory_code_samples_by_window.get((trajectory.trajectory_id, window_count - 1), [])
            current_vectors = [
                code_embeddings_by_hash[item.code_hash]
                for item in current_code_samples
                if item.code_hash in code_embeddings_by_hash
            ]
            final_vectors = [
                code_embeddings_by_hash[item.code_hash]
                for item in final_code_samples
                if item.code_hash in code_embeddings_by_hash
            ]
            if len(current_vectors) >= 2:
                code_diversity_stats_value = _mean_pairwise_distance(current_vectors)
                _stats_vector_add(code_diversity_stats, window_index, code_diversity_stats_value)
            if current_vectors and final_vectors:
                similarities = [
                    _normalized_code_convergence(_cosine_distance(current_vector, final_vector))
                    for current_vector in current_vectors
                    for final_vector in final_vectors
                ]
                if similarities:
                    _stats_vector_add(
                        code_similarity_to_final_bin_stats,
                        window_index,
                        float(np.mean(similarities)),
                    )
            current_records = trajectory_records_by_window.get((trajectory.trajectory_id, window_index), [])
            final_records = trajectory_records_by_window.get((trajectory.trajectory_id, window_count - 1), [])
            if not current_records or not final_records:
                continue
            trace_similarities: list[float] = []
            label_spec = label_spec_by_index.get(trajectory.label_index)
            if label_spec is None:
                continue
            for trace_index, trace in enumerate(traces):
                pairwise_similarities: list[float] = []
                for current_record in current_records:
                    if current_record.artifact is None:
                        continue
                    current_detector = candidate_detector_cache.get(current_record.candidate_uid)
                    if current_detector is None:
                        current_detector = _detector_from_candidate(current_record.artifact, environment_id=resolved_env)
                        candidate_detector_cache[current_record.candidate_uid] = current_detector
                    current_matrix_key = (current_record.candidate_uid, trace_index)
                    current_matrix = candidate_matrix_cache.get(current_matrix_key)
                    if current_matrix is None:
                        current_matrix = _annotation_matrix_for_detector(
                            trace=trace,
                            detector=current_detector,
                            label_spec=label_spec,
                            environment_id=resolved_env,
                            library_id=f"{trajectory.trajectory_id}-{window_index}-{current_record.candidate_uid}",
                        )
                        candidate_matrix_cache[current_matrix_key] = current_matrix
                    for final_record in final_records:
                        if final_record.artifact is None:
                            continue
                        final_detector = candidate_detector_cache.get(final_record.candidate_uid)
                        if final_detector is None:
                            final_detector = _detector_from_candidate(final_record.artifact, environment_id=resolved_env)
                            candidate_detector_cache[final_record.candidate_uid] = final_detector
                        final_matrix_key = (final_record.candidate_uid, trace_index)
                        final_matrix = candidate_matrix_cache.get(final_matrix_key)
                        if final_matrix is None:
                            final_matrix = _annotation_matrix_for_detector(
                                trace=trace,
                                detector=final_detector,
                                label_spec=label_spec,
                                environment_id=resolved_env,
                                library_id=f"{trajectory.trajectory_id}-final-{final_record.candidate_uid}",
                            )
                            candidate_matrix_cache[final_matrix_key] = final_matrix
                        pairwise_similarities.append(1.0 - _matrix_distance(current_matrix, final_matrix))
                if pairwise_similarities:
                    trace_similarities.append(float(np.mean(pairwise_similarities)))
            if trace_similarities:
                _stats_vector_add(
                    pattern_similarity_to_final_bin_stats,
                    window_index,
                    float(np.mean(trace_similarities)),
                )

    completed_repeats = 1
    _log_event("repeat.done", repeat=1, total_repeats=1)

    window_summaries: list[dict[str, object]] = []
    for window_index in range(window_count):
        mean_code_to_final = _stats_vector_item_mean(code_similarity_to_final_bin_stats, window_index)
        std_code_to_final = _stats_vector_item_std(code_similarity_to_final_bin_stats, window_index)
        mean_pattern_similarity = _stats_vector_item_mean(pattern_similarity_to_final_bin_stats, window_index)
        std_pattern_similarity = _stats_vector_item_std(pattern_similarity_to_final_bin_stats, window_index)
        mean_code_diversity = _stats_vector_item_mean(code_diversity_stats, window_index)
        std_code_diversity = _stats_vector_item_std(code_diversity_stats, window_index)
        mean_distance = 1.0 - mean_pattern_similarity
        window_summaries.append(
            {
                "window_index": window_index,
                "window_label": window_labels[window_index],
                "mean_distance_all_labels": mean_distance,
                "mean_distance_std_all_labels": std_pattern_similarity,
                "mean_similarity_all_labels": mean_pattern_similarity,
                "mean_similarity_std_all_labels": std_pattern_similarity,
                "mean_distance_per_label": mean_distance,
                "mean_distance_std_per_label": std_pattern_similarity,
                "mean_similarity_per_label": mean_pattern_similarity,
                "mean_similarity_std_per_label": std_pattern_similarity,
                "mean_code_convergence_per_label": 0.0,
                "mean_code_convergence_std_per_label": 0.0,
                "mean_code_to_final_per_label": mean_code_to_final,
                "mean_code_to_final_std_per_label": std_code_to_final,
                "mean_behavior_to_final_per_label": mean_pattern_similarity,
                "mean_behavior_to_final_std_per_label": std_pattern_similarity,
                "mean_code_embedding_similarity_to_final_per_label": mean_code_to_final,
                "mean_code_embedding_similarity_to_final_std_per_label": std_code_to_final,
                "mean_pattern_similarity_to_final_per_label": mean_pattern_similarity,
                "mean_pattern_similarity_to_final_std_per_label": std_pattern_similarity,
                "mean_code_diversity_per_label": mean_code_diversity,
                "mean_code_diversity_std_per_label": std_code_diversity,
                "code_diversity_run_count": int(code_diversity_stats["count"][window_index]),
                "code_diversity_label_samples": int(code_diversity_stats["count"][window_index]),
                "code_diversity_code_samples": int(code_diversity_stats["count"][window_index]),
                "code_samples_per_label_cap": TOP_K_CANDIDATES_PER_LABEL,
                "distance_gap_all_minus_per_label": 0.0,
                "mean_distance": mean_distance,
                "mean_distance_std": std_pattern_similarity,
                "mean_similarity": mean_pattern_similarity,
                "mean_similarity_std": std_pattern_similarity,
                "sample_count": int(pattern_similarity_to_final_bin_stats["count"][window_index]),
                "code_cross_run_sample_count": 0,
                "code_to_final_sample_count": int(code_similarity_to_final_bin_stats["count"][window_index]),
                "behavior_to_final_sample_count": int(pattern_similarity_to_final_bin_stats["count"][window_index]),
                "code_embedding_to_final_sample_count": int(code_similarity_to_final_bin_stats["count"][window_index]),
                "pattern_to_final_sample_count": int(pattern_similarity_to_final_bin_stats["count"][window_index]),
            }
        )

    similarity_plot_path = output_dir / "similarity_by_window_all_labels.png"
    similarity_per_label_plot_path = output_dir / "similarity_by_window_per_label.png"
    embedding_plot_path = output_dir / "snapshot_embedding.png"
    _render_metric_plot(
        window_summaries,
        similarity_key="mean_similarity_all_labels",
        distance_key="mean_distance_all_labels",
        title="Similarity to final bin across learned label trajectories",
        output_path=similarity_plot_path,
    )
    _render_metric_plot(
        window_summaries,
        similarity_key="mean_similarity_per_label",
        distance_key="mean_distance_per_label",
        title="Similarity to final bin across learned label trajectories",
        output_path=similarity_per_label_plot_path,
    )
    fig, axis = plt.subplots(figsize=(6.0, 4.0))
    axis.text(0.5, 0.5, "Label-trajectory aggregation mode", ha="center", va="center")
    axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(embedding_plot_path, dpi=160)
    plt.close(fig)

    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "analysis_kind": "supervised_consistency",
        "analysis_run_id": analysis_run_id,
        "created_at": _utc_now_text(),
        "env_id": resolved_env,
        "trace_path": str(resolved_trace_paths[0]),
        "trace_paths": [str(path) for path in resolved_trace_paths],
        "trace_count": len(traces),
        "trace_scope": "test_set" if len(traces) > 1 else "single_trace",
        "trace_id": reference_trace.trace_id if len(traces) == 1 else "multiple",
        "task_id": reference_trace.task_id if len(traces) == 1 else "multiple",
        "timestep_count": len(reference_trace.timesteps) if len(traces) == 1 else None,
        "label_count": len(canonical_labels),
        "trajectory_count": len(trajectories),
        "labels": [spec.label for spec in canonical_labels],
        "run_names": [state.run_name for state in run_states],
        "run_dirs": [str(state.run_dir) for state in run_states],
        "run_candidate_totals": {state.run_name: state.total_candidate_count for state in run_states},
        "window_count": window_count,
        "bin_labels": window_labels,
        "analysis_mode": "window_bins",
        "analysis_variant": "label_trajectory_similarity_to_final_bin_min25",
        "average_candidate_budget": average_budget,
        "max_candidate_budget": max_budget,
        "checkpoints": [],
        "sampling_repeats": int(max(1, int(sampling_repeats))),
        "completed_repeats": int(completed_repeats),
        "sampling_seed": int(sampling_seed),
        "code_samples_per_label": TOP_K_CANDIDATES_PER_LABEL,
        "top_k_candidates_per_label": TOP_K_CANDIDATES_PER_LABEL,
        "min_label_trajectory_candidates": MIN_LABEL_TRAJECTORY_CANDIDATES,
        "final_bin_index": window_count - 1,
        "final_bin_label": window_labels[window_count - 1],
        "window_summaries": window_summaries,
        "checkpoint_summaries": window_summaries,
        "pair_summaries": [],
        "average_distance_grid": [],
        "average_distance_grid_std": [],
        "average_distance_grid_all_labels": [],
        "average_distance_grid_std_all_labels": [],
        "average_distance_grid_per_label": [],
        "average_distance_grid_std_per_label": [],
        "average_distance_grid_path": "",
        "average_distance_grid_std_path": "",
        "average_distance_grid_all_labels_path": "",
        "average_distance_grid_std_all_labels_path": "",
        "average_distance_grid_per_label_path": "",
        "average_distance_grid_std_per_label_path": "",
        "two_run_distance_grid_path": "",
        "two_run_distance_grid_per_label_path": "",
        "two_run_distance_grid_pair": "",
        "pairwise_distance_histogram_path": "",
        "pairwise_distance_histogram_per_label_path": "",
        "run_pair_distance_matrix_path": "",
        "run_pair_distance_matrix_per_label_path": "",
        "similarity_plot_path": str(similarity_plot_path),
        "similarity_all_labels_plot_path": str(similarity_plot_path),
        "similarity_per_label_plot_path": str(similarity_per_label_plot_path),
        "embedding_plot_path": str(embedding_plot_path),
    }
    (output_dir / SUMMARY_FILENAME).write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / MATRICES_FILENAME).write_text(json.dumps(matrices_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    resume_state_path = _resume_state_path(output_dir)
    if resume_state_path.is_file():
        try:
            resume_state_path.unlink()
        except OSError:
            pass
    _log_event("completed", windows=window_count, runs=len(run_states), trajectories=len(trajectories), output_dir=output_dir)
    return summary_payload


def _default_output_dir(*, env_id: str) -> Path:
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{env_id}-supervised-consistency"
    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "analysis"
        / "supervised_consistency"
        / env_id
        / run_id
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m code_convergence.consistency_analysis",
        description="Compare supervised learning runs by applying percentile-window libraries to the same trace.",
    )
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--trace-path", action="append", dest="trace_paths", required=True)
    parser.add_argument("--run-dir", action="append", dest="run_dirs", required=True)
    parser.add_argument("--checkpoint", action="append", dest="checkpoints", type=int, default=[])
    parser.add_argument(
        "--sampling-repeats",
        type=int,
        default=DEFAULT_SAMPLING_REPEATS,
        help="Monte Carlo repeats per candidate window bin for average distance estimation.",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=42,
        help="Random seed used for candidate sampling.",
    )
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args(argv)

    env_id = str(args.env_id or "").strip().lower()
    output_dir = Path(str(args.output_dir).strip()) if str(args.output_dir).strip() else _default_output_dir(env_id=env_id)
    checkpoints = list(args.checkpoints or [])
    run_supervised_consistency_analysis(
        env_id=env_id,
        run_dirs=[Path(item) for item in args.run_dirs],
        trace_paths=[Path(item) for item in (args.trace_paths or [])],
        output_dir=output_dir,
        checkpoints=checkpoints,
        sampling_repeats=max(1, int(args.sampling_repeats)),
        sampling_seed=int(args.sampling_seed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
