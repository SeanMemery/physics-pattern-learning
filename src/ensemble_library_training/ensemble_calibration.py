from __future__ import annotations

import argparse
import base64
import importlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from deepphy_eval import default_run_id
from pattern_library_registry import load_pattern_library, save_pattern_library
from pattern_library_schema import LabelDetectorEnsemble, PatternLibrary
from pattern_learning.patterns.scoring import request_chat_completion_with_text_retry


CALIBRATION_SCHEMA_VERSION = "ensemble-calibration.v1"
DEFAULT_TOTAL_TRIALS = 12
DEFAULT_RANDOM_INIT_TRIALS = 4
DEFAULT_START_TIME_THRESHOLD = 5
DEFAULT_END_TIME_THRESHOLD = 5
DEFAULT_START_LOCATION_THRESHOLD = 0.05
DEFAULT_END_LOCATION_THRESHOLD = 0.05
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_ENV_CALIBRATION_THRESHOLDS = {
    "phyre": {
        "start_time_threshold": 4,
        "end_time_threshold": 4,
        "start_location_threshold": 0.05,
        "end_location_threshold": 0.05,
        "confidence_threshold": 0.234375,
    },
    "iphyre": {
        "start_time_threshold": 2,
        "end_time_threshold": 2,
        "start_location_threshold": 0.00625,
        "end_location_threshold": 0.00625,
        "confidence_threshold": 0.0875,
    },
    "kinetix": {
        "start_time_threshold": 9,
        "end_time_threshold": 9,
        "start_location_threshold": 1.25,
        "end_location_threshold": 1.25,
        "confidence_threshold": 1.5,
    },
    "pooltool": {
        "start_time_threshold": 1,
        "end_time_threshold": 1,
        "start_location_threshold": 0.005,
        "end_location_threshold": 0.005,
        "confidence_threshold": 0.35,
    },
}
DEFAULT_EVAL_WORKERS = 2
DEFAULT_TASK_LIMIT_BY_ENV = {
    "phyre": 20,
}
DEFAULT_DISCRIMINATION_OPTION_COUNT = 5
DEFAULT_DISCRIMINATION_OPTION_COUNT_BY_ENV = {
    "iphyre": 9,
    "kinetix": 9,
    "phyre": 9,
}
DEFAULT_TARGET_MAX_INITIAL_SCORE = 0.60
OPTION_TUNING_FILENAME = "option_tuning.json"
DEFAULT_SUMMARY_BATCH_SIZE = 20
DEFAULT_DISCRIMINATION_FRAME_LIMIT = 10
SUMMARY_MAX_TOKENS = 8192
CHOICE_MAX_TOKENS = 4096
SUMMARY_PROMPT_TEMPLATE = """You are writing a summary for a discrimination benchmark.

Another model will be shown the same simulation and must choose which one of several candidate summaries matches it exactly.
Write the summary so this exact simulation is easy to distinguish from similar alternatives.

Required content, in order:
1. The initial arrangement of the important objects and their relative positions.
2. The decisive causal interactions, collisions, contacts, motions, or constraint changes.
3. The final outcome, including where the important objects end up and what task-relevant result occurs.

Be concrete and specific. Mention counts, spatial relations, directions, collisions, and final state details that make this simulation unique.
Use the single initial frame to anchor the starting state, and use the human-readable annotation text as full-sequence context for the whole simulation.
Do not mention the annotation itself. Do not write generic statements that could fit many simulations.
Write 4-6 sentences."""
CHOICE_SYSTEM_PROMPT = (
    "You are matching a simulation to one of several candidate summaries. "
    "Use only the provided frames and choose the single best-matching summary. "
    "You may reason before answering if needed, but your final answer must be one fenced code block exactly like "
    "```choice\\nA\\n``` with one letter."
)

_CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _discrimination_option_count(env_id: str) -> int:
    env_key = str(env_id or "").strip().lower()
    return max(
        2,
        min(
            len(_CHOICE_LETTERS),
            int(DEFAULT_DISCRIMINATION_OPTION_COUNT_BY_ENV.get(env_key, DEFAULT_DISCRIMINATION_OPTION_COUNT)),
        ),
    )


def _target_max_initial_score() -> float:
    raw = str(os.environ.get("ENSEMBLE_BO_TARGET_MAX_INITIAL_SCORE", "") or "").strip()
    if not raw:
        return float(DEFAULT_TARGET_MAX_INITIAL_SCORE)
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return float(DEFAULT_TARGET_MAX_INITIAL_SCORE)


def _load_option_tuning(output_root: Path) -> dict[str, Any]:
    payload = _load_json(output_root / OPTION_TUNING_FILENAME)
    return payload if isinstance(payload, dict) else {}


def _suggested_min_option_count(output_root: Path) -> int:
    payload = _load_option_tuning(output_root)
    try:
        value = int(payload.get("suggested_min_option_count") or 0)
    except (TypeError, ValueError):
        value = 0
    return max(0, value)


def _run_option_count(*, env_id: str, output_root: Path, task_batch: list[dict[str, Any]]) -> int:
    base = _discrimination_option_count(env_id)
    suggested = _suggested_min_option_count(output_root)
    limit = min(len(_CHOICE_LETTERS), len(task_batch))
    return max(2, min(limit, max(base, suggested)))


def _current_option_count(output_root: Path, env_id: str) -> int:
    payload = _load_json(output_root / "question_set.json")
    try:
        value = int(payload.get("option_count") or 0)
    except (TypeError, ValueError):
        value = 0
    if value > 0:
        return value
    return _discrimination_option_count(env_id)


def _normalized_task_count(env_id: str, requested: int | None = None) -> int:
    default = int(DEFAULT_TASK_LIMIT_BY_ENV.get(str(env_id or "").strip().lower(), DEFAULT_SUMMARY_BATCH_SIZE))
    if requested is None:
        return max(1, default)
    return max(1, int(requested))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: datetime | None = None) -> str:
    return (value or _utc_now()).isoformat().replace("+00:00", "Z")


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str
    input_mode: str
    parameter_kind: str
    slot_index: int | None
    lower: float
    upper: float
    integer: bool = False


def _supported_ensembles(library: PatternLibrary) -> list[LabelDetectorEnsemble]:
    return [
        ensemble
        for ensemble in library.label_ensembles
        if ensemble.input_mode in {"trace", "supervised_evidence"} and ensemble.members
    ]


def _default_calibration_payload(library: PatternLibrary) -> dict[str, Any]:
    env_id = str((library.environment_ids or [""])[0] or "").strip().lower()
    default_thresholds = dict(DEFAULT_ENV_CALIBRATION_THRESHOLDS.get(env_id, {}))
    if not default_thresholds:
        default_thresholds = {
            "start_time_threshold": DEFAULT_START_TIME_THRESHOLD,
            "end_time_threshold": DEFAULT_END_TIME_THRESHOLD,
            "start_location_threshold": DEFAULT_START_LOCATION_THRESHOLD,
            "end_location_threshold": DEFAULT_END_LOCATION_THRESHOLD,
            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        }
    labels_payload: list[dict[str, Any]] = []
    for ensemble in _supported_ensembles(library):
        slot_weights = [1.0 for _member in ensemble.members]
        labels_payload.append(
            {
                "label": ensemble.label,
                "input_mode": ensemble.input_mode,
                "slot_weights": slot_weights,
                "thresholds": {
                    "start_time_threshold": int(default_thresholds["start_time_threshold"]),
                    "end_time_threshold": int(default_thresholds["end_time_threshold"]),
                    "start_location_threshold": float(default_thresholds["start_location_threshold"]),
                    "end_location_threshold": float(default_thresholds["end_location_threshold"]),
                    "confidence_threshold": float(default_thresholds["confidence_threshold"]),
                    "require_object_identity": True,
                },
            }
        )
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "library_id": library.library_id,
        "labels": labels_payload,
    }


def _parameter_specs(library: PatternLibrary) -> list[ParameterSpec]:
    specs: list[ParameterSpec] = []
    for ensemble in _supported_ensembles(library):
        label_key = f"{ensemble.input_mode}:{ensemble.label}"
        for member in ensemble.members:
            specs.append(
                ParameterSpec(
                    key=f"{label_key}:slot:{int(member.slot_index)}",
                    label=ensemble.label,
                    input_mode=ensemble.input_mode,
                    parameter_kind="slot_weight",
                    slot_index=int(member.slot_index),
                    lower=0.0,
                    upper=1.0,
                )
            )
    return specs


def _vector_from_payload(payload: dict[str, Any], specs: list[ParameterSpec]) -> list[float]:
    label_lookup = {
        (str(item.get("label") or ""), str(item.get("input_mode") or "")): item
        for item in payload.get("labels", [])
        if isinstance(item, dict)
    }
    vector: list[float] = []
    for spec in specs:
        label_payload = label_lookup.get((spec.label, spec.input_mode), {})
        slot_weights = label_payload.get("slot_weights")
        value = 1.0
        if (
            isinstance(slot_weights, list)
            and spec.slot_index is not None
            and 0 <= spec.slot_index < len(slot_weights)
        ):
            try:
                value = float(slot_weights[spec.slot_index])
            except (TypeError, ValueError):
                value = 1.0
        vector.append(_clip(value, spec.lower, spec.upper))
    return vector


def _payload_from_vector(
    *,
    template_payload: dict[str, Any],
    specs: list[ParameterSpec],
    vector: list[float],
) -> dict[str, Any]:
    payload = json.loads(json.dumps(template_payload))
    label_lookup = {
        (str(item.get("label") or ""), str(item.get("input_mode") or "")): item
        for item in payload.get("labels", [])
        if isinstance(item, dict)
    }
    for spec, value in zip(specs, vector):
        label_payload = label_lookup.get((spec.label, spec.input_mode))
        if not isinstance(label_payload, dict):
            continue
        slot_weights = label_payload.get("slot_weights")
        if not isinstance(slot_weights, list) or spec.slot_index is None:
            continue
        while len(slot_weights) <= spec.slot_index:
            slot_weights.append(1.0)
        slot_weights[spec.slot_index] = _clip(value, spec.lower, spec.upper)
    return payload


def _clip(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def _random_vector(specs: list[ParameterSpec], rng: random.Random) -> list[float]:
    return [rng.uniform(spec.lower, spec.upper) for spec in specs]


def _mutated_vector(base: list[float], specs: list[ParameterSpec], rng: random.Random) -> list[float]:
    values: list[float] = []
    for current, spec in zip(base, specs):
        span = float(spec.upper - spec.lower)
        proposal = float(current) + rng.gauss(0.0, max(0.01, span / 6.0))
        values.append(_clip(proposal, spec.lower, spec.upper))
    return values


def _squared_distance(left: list[float], right: list[float], specs: list[ParameterSpec]) -> float:
    total = 0.0
    for lv, rv, spec in zip(left, right, specs):
        span = max(1e-6, float(spec.upper - spec.lower))
        delta = (float(lv) - float(rv)) / span
        total += delta * delta
    return total


def _acquisition_score(
    candidate: list[float],
    history: list[dict[str, Any]],
    specs: list[ParameterSpec],
) -> float:
    if not history:
        return float("inf")
    weights: list[float] = []
    scores: list[float] = []
    for trial in history:
        vector = [float(item) for item in trial.get("vector", [])]
        score = float(trial.get("score", 0.0))
        distance = _squared_distance(candidate, vector, specs)
        weight = math.exp(-distance / 0.5)
        weights.append(weight)
        scores.append(score)
    total_weight = sum(weights)
    if total_weight <= 1e-9:
        return max(scores) + 1.0
    mean = sum(weight * score for weight, score in zip(weights, scores)) / total_weight
    uncertainty = 1.0 / (1.0 + total_weight)
    return mean + 0.25 * uncertainty


def _propose_next_vector(
    *,
    specs: list[ParameterSpec],
    history: list[dict[str, Any]],
    best_vector: list[float],
    rng: random.Random,
) -> list[float]:
    candidate_pool: list[list[float]] = []
    for _ in range(48):
        candidate_pool.append(_mutated_vector(best_vector, specs, rng))
    for _ in range(16):
        candidate_pool.append(_random_vector(specs, rng))
    return max(candidate_pool, key=lambda candidate: _acquisition_score(candidate, history, specs))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _default_eval_workers(env_id: str) -> int:
    return 1 if str(env_id or "").strip().lower() == "pooltool" else DEFAULT_EVAL_WORKERS


def _task_limit(env_id: str, requested: int | None = None) -> int:
    return _normalized_task_count(env_id, requested)


def _server_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_server_src_on_path() -> None:
    server_src = _server_project_root() / "src"
    server_src_text = str(server_src)
    if server_src_text not in sys.path:
        sys.path.insert(0, server_src_text)


def _task_summary_module():
    _ensure_server_src_on_path()
    return importlib.import_module("pattern_server.task_summaries")


def _chat_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = str(os.environ.get("LLM_ROUTER_API_KEY") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _encode_image_part(image_path: Path) -> dict[str, Any]:
    suffix = image_path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{encoded}"},
    }


def _flatten_completion_text(value: Any, *, _depth: int = 0) -> str:
    if _depth > 6:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_flatten_completion_text(item, _depth=_depth + 1) for item in value]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        parts: list[str] = []
        for key in ("text", "content", "output_text", "reasoning_content", "reasoning", "output", "value"):
            part = _flatten_completion_text(value.get(key), _depth=_depth + 1)
            if part:
                parts.append(part)
        if parts:
            return "\n".join(parts).strip()
    return ""


def _completion_text_candidates(completion_body: dict[str, Any]) -> list[str]:
    if not isinstance(completion_body, dict):
        return []
    choices = completion_body.get("choices")
    if not isinstance(choices, list):
        return []
    fields = ("content", "output_text", "text", "reasoning_content", "reasoning", "output")
    candidates: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict):
            for field in fields:
                flattened = _flatten_completion_text(message.get(field))
                if flattened:
                    candidates.append(flattened)
        for field in fields:
            flattened = _flatten_completion_text(choice.get(field))
            if flattened:
                candidates.append(flattened)
        delta = choice.get("delta")
        if isinstance(delta, dict):
            for field in fields:
                flattened = _flatten_completion_text(delta.get(field))
                if flattened:
                    candidates.append(flattened)
    return candidates


def _cleanup_completion_text(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith("'") and cleaned.endswith("'") and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _sample_paths_evenly(paths: list[Path], limit: int) -> list[Path]:
    if limit <= 0 or len(paths) <= limit:
        return list(paths)
    if limit == 1:
        return [paths[0]]
    last_index = len(paths) - 1
    sampled_indices = {
        int(round(position * last_index / float(limit - 1)))
        for position in range(limit)
    }
    return [paths[index] for index in sorted(sampled_indices)]


def _extract_fenced_block(text: str, label: str) -> str:
    pattern = re.compile(r"```([^\n`]*)\n(.*?)```", flags=re.IGNORECASE | re.DOTALL)
    for match in pattern.finditer(text):
        current_label = str(match.group(1) or "").strip().lower()
        if current_label != label.lower():
            continue
        body = _cleanup_completion_text(str(match.group(2) or ""))
        if body:
            return body
    return ""


def _extract_summary_text(primary_text: str, completion_body: dict[str, Any]) -> str:
    candidates = [_cleanup_completion_text(primary_text)]
    candidates.extend(_cleanup_completion_text(item) for item in _completion_text_candidates(completion_body))
    for candidate in candidates:
        if not candidate:
            continue
        fenced = _extract_fenced_block(candidate, "summary")
        if fenced:
            return fenced
    for candidate in candidates:
        if candidate:
            return candidate
    return ""


def _extract_choice_text(primary_text: str, completion_body: dict[str, Any]) -> str:
    candidates = [_cleanup_completion_text(primary_text)]
    candidates.extend(_cleanup_completion_text(item) for item in _completion_text_candidates(completion_body))
    for candidate in candidates:
        if not candidate:
            continue
        fenced = _extract_fenced_block(candidate, "choice")
        if fenced:
            match = re.search(r"\b([A-E])\b", fenced.upper())
            if match:
                return match.group(1)
    for candidate in candidates:
        match = re.search(r"\b([A-E])\b", candidate.upper())
        if match:
            return match.group(1)
    return ""


def _request_summary_text(
    *,
    project_root: Path,
    base_url: str,
    model: str,
    timeout: int,
    env_id: str,
    task_id: str,
    annotation_text: str,
    image_paths: list[Path],
) -> str:
    task_summaries = _task_summary_module()
    user_lines = [
        SUMMARY_PROMPT_TEMPLATE.strip(),
        "",
        f"Environment: {env_id}",
        f"Task: {task_id}",
        "The attached image is the single initial frame of the simulation.",
        "Use that initial frame to ground the starting arrangement, and use the human-readable annotation text below to understand the full simulation dynamics and outcome.",
        "Do not omit important concrete events described in the annotation if they are part of the simulation.",
        "",
        "Learned pattern annotation summary:",
        annotation_text,
        "",
        "Return only the final summary wrapped in exactly one fenced code block like this:",
        "```summary",
        "<your summary>",
        "```",
    ]
    user_content: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(user_lines).strip()}]
    for image_path in image_paths:
        user_content.append(_encode_image_part(image_path))
    with task_summaries._acquire_llm_request_slot(project_root):
        body, text = request_chat_completion_with_text_retry(
            base_url=base_url,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You summarize physics simulations for a matching benchmark. "
                        "Be concrete, specific, and grounded in the simulation. "
                        "Return only one fenced code block exactly like ```summary ... ```."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            timeout=timeout,
            headers=_chat_headers(),
            error_prefix="Ensemble calibration summary response",
            max_tokens=SUMMARY_MAX_TOKENS,
            temperature=0.0,
        )
    explicit_summary = task_summaries._extract_explicit_summary_from_completion(text, body)
    if explicit_summary:
        return explicit_summary
    reasoning_trace = task_summaries._extract_reasoning_trace_from_completion(text, body)
    if reasoning_trace:
        extracted = task_summaries._request_summary_from_reasoning_trace(
            project_root=project_root,
            base_url=base_url,
            model=model,
            timeout=timeout,
            env_id=env_id,
            task_id=task_id,
            reasoning_trace=reasoning_trace,
        )
        if extracted:
            return extracted
    return _extract_summary_text(text, body)


def _request_choice(
    *,
    project_root: Path,
    base_url: str,
    model: str,
    timeout: int,
    env_id: str,
    task_id: str,
    option_texts: list[str],
    image_paths: list[Path],
) -> tuple[str, str]:
    task_summaries = _task_summary_module()
    option_count = min(len(option_texts), _discrimination_option_count(env_id))
    option_letters = _CHOICE_LETTERS[:option_count]
    option_lines = []
    for index, summary_text in enumerate(option_texts[:option_count]):
        option_lines.append(f"{option_letters[index]}. {summary_text}")
    user_lines = [
        "You will be shown low-resolution frames sampled across the full simulation video.",
        f"Choose which one of the {option_count} candidate summaries matches this exact simulation.",
        f"One option is correct. The other {max(0, option_count - 1)} are summaries of different simulations.",
        "Pay attention to the initial arrangement, decisive interactions, and final outcome.",
        "You may reason first, but the final answer must still be a single fenced choice block.",
        "",
        f"Environment: {env_id}",
        f"Task: {task_id}",
        "",
        "Candidate summaries:",
        *option_lines,
        "",
        "Return only the matching option letter wrapped in exactly one fenced code block like this:",
        "```choice",
        "A",
        "```",
    ]
    user_content: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(user_lines).strip()}]
    for image_path in image_paths:
        user_content.append(_encode_image_part(image_path))
    with task_summaries._acquire_llm_request_slot(project_root):
        body, text = request_chat_completion_with_text_retry(
            base_url=base_url,
            model=model,
            messages=[
                {"role": "system", "content": CHOICE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            timeout=timeout,
            headers=_chat_headers(),
            error_prefix="Ensemble calibration discrimination response",
            max_tokens=CHOICE_MAX_TOKENS,
            temperature=0.0,
        )
    choice = _extract_choice_text(text, body)
    return choice, _cleanup_completion_text(text)


def _serialize_task_summary_task(task: Any, *, batch_index: int) -> dict[str, Any]:
    return {
        "batch_id": f"task-{batch_index:03d}",
        "batch_index": int(batch_index),
        "env_id": str(task.env_id),
        "source_run_id": str(task.source_run_id),
        "source_run_dir": str(task.source_run_dir),
        "source_result_path": str(task.source_result_path),
        "case_id": str(task.case_id),
        "task_id": str(task.task_id),
        "payload": dict(task.payload) if isinstance(task.payload, dict) else {},
    }


def _deserialize_task_summary_task(task_payload: dict[str, Any], *, output_root: Path):
    task_summaries = _task_summary_module()
    batch_id = str(task_payload.get("batch_id") or "").strip()
    task_dir = output_root / "media_cache" / batch_id
    return task_summaries.TaskSummaryTask(
        env_id=str(task_payload.get("env_id") or "").strip(),
        source_run_id=str(task_payload.get("source_run_id") or "").strip(),
        source_run_dir=Path(str(task_payload.get("source_run_dir") or "")).resolve(),
        source_result_path=Path(str(task_payload.get("source_result_path") or "")).resolve(),
        case_id=str(task_payload.get("case_id") or "").strip(),
        task_id=str(task_payload.get("task_id") or "").strip(),
        payload=dict(task_payload.get("payload") or {}),
        output_run_dir=output_root / "media_cache",
        output_task_dir=task_dir,
        summary_path=task_dir / "summary.json",
    )


def _find_reusable_run_bundle_dir(
    *,
    output_root: Path,
    seed: int,
) -> Path | None:
    runs_root = output_root.parent
    if not runs_root.is_dir():
        return None
    required_filenames = ("task_batch.json", "question_set.json", "media_manifest.json")
    candidates = sorted(
        (
            path.resolve()
            for path in runs_root.iterdir()
            if path.is_dir()
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    current_root = output_root.resolve()
    for candidate in candidates:
        if candidate == current_root:
            continue
        request_payload = _load_json(candidate / "request.json")
        if request_payload and int(request_payload.get("seed") or -1) != int(seed):
            continue
        if any(not (candidate / required_name).is_file() for required_name in required_filenames):
            continue
        return candidate
    return None


def _load_existing_trial_history(output_root: Path) -> list[dict[str, Any]]:
    trials_root = output_root / "trials"
    if not trials_root.is_dir():
        return []
    history: list[dict[str, Any]] = []
    for trial_index in range(100000):
        trial_path = trials_root / f"trial-{trial_index:03d}" / "trial.json"
        if not trial_path.is_file():
            break
        payload = _load_json(trial_path)
        if not isinstance(payload, dict):
            break
        history.append(payload)
    return history


def _resume_started_at(output_root: Path, history: list[dict[str, Any]]) -> datetime | None:
    progress_payload = _load_json(output_root / "progress.json")
    started_at = _parse_iso_datetime(progress_payload.get("started_at"))
    if started_at is not None:
        return started_at
    if history:
        return _parse_iso_datetime(history[0].get("started_at"))
    return None


def _load_or_create_task_batch(
    *,
    server_root: Path,
    env_id: str,
    output_root: Path,
    seed: int,
    task_count: int | None = None,
) -> list[dict[str, Any]]:
    batch_path = output_root / "task_batch.json"
    limit = _task_limit(env_id, task_count)
    existing = _load_json(batch_path)
    if isinstance(existing.get("tasks"), list):
        tasks = [item for item in existing["tasks"] if isinstance(item, dict)]
        if len(tasks) == limit:
            return tasks
    reusable_dir = _find_reusable_run_bundle_dir(output_root=output_root, seed=seed)
    if reusable_dir is not None:
        reused_payload = _load_json(reusable_dir / "task_batch.json")
        tasks = [item for item in reused_payload.get("tasks", []) if isinstance(item, dict)]
        if len(tasks) == limit:
            shutil.copy2(reusable_dir / "task_batch.json", batch_path)
            return tasks
    task_summaries = _task_summary_module()
    discovered = list(task_summaries.discover_task_summary_tasks(server_root, env_id))
    if len(discovered) < limit:
        raise ValueError(f"Need at least {limit} successful summary-source tasks for {env_id}, found {len(discovered)}")
    rng = random.Random(int(seed))
    candidate_indices = list(range(len(discovered)))
    rng.shuffle(candidate_indices)
    tasks: list[dict[str, Any]] = []
    for candidate_index in candidate_indices:
        if len(tasks) >= limit:
            break
        task = discovered[candidate_index]
        batch_index = len(tasks)
        serialized = _serialize_task_summary_task(task, batch_index=batch_index)
        candidate_task = _deserialize_task_summary_task(serialized, output_root=output_root)
        try:
            _trace_path, _video_path, _initial_image_path, image_paths = task_summaries._render_summary_media(
                server_root,
                candidate_task,
            )
        except Exception:
            continue
        if not image_paths:
            continue
        tasks.append(serialized)
    if len(tasks) < limit:
        raise ValueError(
            f"Need at least {limit} summary-source tasks with renderable media for {env_id}, found {len(tasks)}"
        )
    _write_json(
        batch_path,
        {
            "env_id": env_id,
            "seed": int(seed),
            "task_count": len(tasks),
            "tasks": tasks,
        },
    )
    return tasks


def _load_or_create_question_set(*, output_root: Path, task_batch: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    question_path = output_root / "question_set.json"
    env_id = str(task_batch[0].get("env_id") or "").strip() if task_batch else ""
    option_count = _run_option_count(env_id=env_id, output_root=output_root, task_batch=task_batch)
    expected_question_count = len(task_batch)
    expected_batch_ids = {str(item.get("batch_id") or "") for item in task_batch}
    existing = _load_json(question_path)
    if (
        isinstance(existing.get("questions"), list)
        and int(existing.get("option_count") or DEFAULT_DISCRIMINATION_OPTION_COUNT) == option_count
    ):
        questions = [item for item in existing["questions"] if isinstance(item, dict)]
        question_batch_ids = {str(item.get("target_batch_id") or "") for item in questions}
        if len(questions) == expected_question_count and question_batch_ids == expected_batch_ids:
            return questions
    reusable_dir = _find_reusable_run_bundle_dir(output_root=output_root, seed=seed)
    if reusable_dir is not None:
        reused_payload = _load_json(reusable_dir / "question_set.json")
        if int(reused_payload.get("option_count") or DEFAULT_DISCRIMINATION_OPTION_COUNT) == option_count:
            questions = [item for item in reused_payload.get("questions", []) if isinstance(item, dict)]
            question_batch_ids = {str(item.get("target_batch_id") or "") for item in questions}
            if len(questions) == expected_question_count and question_batch_ids == expected_batch_ids:
                shutil.copy2(reusable_dir / "question_set.json", question_path)
                return questions
    if len(task_batch) < option_count:
        raise ValueError(f"Need at least {option_count} tasks for discrimination evaluation")
    rng = random.Random(int(seed) + 1009)
    questions: list[dict[str, Any]] = []
    batch_ids = [str(item.get("batch_id") or "") for item in task_batch]
    for target_batch_id in batch_ids:
        distractors = [batch_id for batch_id in batch_ids if batch_id != target_batch_id]
        distractor_ids = rng.sample(distractors, option_count - 1)
        option_batch_ids = [target_batch_id, *distractor_ids]
        rng.shuffle(option_batch_ids)
        correct_index = option_batch_ids.index(target_batch_id)
        questions.append(
            {
                "target_batch_id": target_batch_id,
                "option_batch_ids": option_batch_ids,
                "correct_choice": _CHOICE_LETTERS[correct_index],
            }
        )
    _write_json(
        question_path,
        {
            "seed": int(seed),
            "option_count": option_count,
            "question_count": len(questions),
            "questions": questions,
        },
    )
    return questions


def _load_or_create_media_manifest(
    *,
    server_root: Path,
    output_root: Path,
    task_batch: list[dict[str, Any]],
    seed: int,
) -> dict[str, dict[str, Any]]:
    manifest_path = output_root / "media_manifest.json"
    expected_batch_ids = {str(item.get("batch_id") or "") for item in task_batch}
    existing = _load_json(manifest_path)
    manifest = existing.get("tasks") if isinstance(existing.get("tasks"), dict) else {}
    if isinstance(manifest, dict):
        valid = True
        if set(str(key) for key in manifest.keys()) != expected_batch_ids:
            valid = False
        for batch_id, item in manifest.items():
            if not isinstance(item, dict):
                valid = False
                break
            frame_paths = item.get("frame_paths")
            trace_path = item.get("trace_path")
            initial_image_path = item.get("initial_image_path")
            if (
                not isinstance(frame_paths, list)
                or not frame_paths
                or not isinstance(trace_path, str)
                or not isinstance(initial_image_path, str)
            ):
                valid = False
                break
            initial_image_name = Path(initial_image_path).name
            if (
                not Path(trace_path).is_file()
                or not Path(initial_image_path).is_file()
                or not initial_image_name.startswith("summary_initial")
                or any(not Path(str(path)).is_file() for path in frame_paths)
            ):
                valid = False
                break
        if valid and manifest:
            return {str(key): dict(value) for key, value in manifest.items() if isinstance(value, dict)}
    reusable_dir = _find_reusable_run_bundle_dir(output_root=output_root, seed=seed)
    if reusable_dir is not None:
        copied = _load_json(reusable_dir / "media_manifest.json")
        manifest = copied.get("tasks") if isinstance(copied.get("tasks"), dict) else {}
        valid = True
        if isinstance(manifest, dict):
            if set(str(key) for key in manifest.keys()) != expected_batch_ids:
                valid = False
            for _batch_id, item in manifest.items():
                if not isinstance(item, dict):
                    valid = False
                    break
                frame_paths = item.get("frame_paths")
                trace_path = item.get("trace_path")
                initial_image_path = item.get("initial_image_path")
                if (
                    not isinstance(frame_paths, list)
                    or not frame_paths
                    or not isinstance(trace_path, str)
                    or not isinstance(initial_image_path, str)
                ):
                    valid = False
                    break
                initial_image_name = Path(initial_image_path).name
                if (
                    not Path(trace_path).is_file()
                    or not Path(initial_image_path).is_file()
                    or not initial_image_name.startswith("summary_initial")
                    or any(not Path(str(path)).is_file() for path in frame_paths)
                ):
                    valid = False
                    break
            if valid and manifest:
                shutil.copy2(reusable_dir / "media_manifest.json", manifest_path)
                return {str(key): dict(value) for key, value in manifest.items() if isinstance(value, dict)}

    task_summaries = _task_summary_module()
    tasks_payload: dict[str, dict[str, Any]] = {}
    for task_payload in task_batch:
        task = _deserialize_task_summary_task(task_payload, output_root=output_root)
        trace_path, video_path, initial_image_path, image_paths = task_summaries._render_summary_media(server_root, task)
        if not image_paths:
            raise ValueError(f"No summary media frames available for {task.task_id}")
        tasks_payload[str(task_payload.get("batch_id") or "")] = {
            "trace_path": str(trace_path),
            "video_path": str(video_path) if isinstance(video_path, Path) else "",
            "initial_image_path": str(initial_image_path),
            "frame_paths": [str(path) for path in image_paths],
        }
    _write_json(
        manifest_path,
        {
            "task_count": len(tasks_payload),
            "tasks": tasks_payload,
        },
    )
    return tasks_payload


def _generate_trial_summaries(
    *,
    server_root: Path,
    env_id: str,
    base_url: str,
    model: str,
    timeout: int,
    workers: int,
    task_batch: list[dict[str, Any]],
    media_manifest: dict[str, dict[str, Any]],
    trial_library_path: Path,
    trial_root: Path,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, dict[str, Any]]:
    task_summaries = _task_summary_module()
    annotations_root = trial_root / "annotations"
    summaries_root = trial_root / "summaries"
    annotations_root.mkdir(parents=True, exist_ok=True)
    summaries_root.mkdir(parents=True, exist_ok=True)
    total_count = len(task_batch)
    counters_lock = threading.Lock()
    annotation_started = 0
    annotation_completed = 0
    summary_started = 0
    summary_completed = 0

    def _load_cached_summary(batch_id: str, task_payload: dict[str, Any]) -> dict[str, Any] | None:
        annotation_path = annotations_root / f"{batch_id}.json"
        summary_path = summaries_root / f"{batch_id}.json"
        if not annotation_path.is_file() or not summary_path.is_file():
            return None
        summary_payload = _load_json(summary_path)
        summary_text = str(summary_payload.get("summary_text") or "").strip()
        if not summary_text:
            return None
        annotation_pattern_count = summary_payload.get("annotation_pattern_count")
        if not isinstance(annotation_pattern_count, int):
            annotation_pattern_count = _annotation_pattern_count(_load_json(annotation_path))
        return {
            "batch_id": batch_id,
            "summary_text": summary_text,
            "annotation_path": str(annotation_path),
            "summary_path": str(summary_path),
            "annotation_pattern_count": int(annotation_pattern_count or 0),
        }

    def _emit_progress() -> None:
        if progress_callback is None:
            return
        with counters_lock:
            completed_units = annotation_completed + summary_completed
            started_units = annotation_started + summary_started
        progress_callback(
            {
                "eval_completed_tasks": completed_units,
                "eval_started_tasks": started_units,
                "eval_total_tasks": total_count * 2,
            }
        )

    def _run_one(task_payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal annotation_started, annotation_completed, summary_started, summary_completed
        batch_id = str(task_payload.get("batch_id") or "").strip()
        cached = _load_cached_summary(batch_id, task_payload)
        if cached is not None:
            with counters_lock:
                annotation_started += 1
                annotation_completed += 1
                summary_started += 1
                summary_completed += 1
            _emit_progress()
            return cached
        media_entry = dict(media_manifest.get(batch_id) or {})
        trace_path = Path(str(media_entry.get("trace_path") or "")).resolve()
        initial_image_path = Path(str(media_entry.get("initial_image_path") or "")).resolve()
        if not trace_path.is_file() or not initial_image_path.is_file():
            raise ValueError(f"Missing cached media for {batch_id}")
        with counters_lock:
            annotation_started += 1
        _emit_progress()
        trace = task_summaries._load_trace(trace_path)
        annotation_payload = task_summaries._apply_library_to_trace(server_root, trace, trial_library_path)
        annotation_pattern_count = _annotation_pattern_count(annotation_payload)
        annotation_path = annotations_root / f"{batch_id}.json"
        _write_json(annotation_path, annotation_payload)
        with counters_lock:
            annotation_completed += 1
            summary_started += 1
        _emit_progress()
        annotation_text = task_summaries._format_annotation_text(annotation_payload)
        summary_text = _request_summary_text(
            project_root=server_root,
            base_url=base_url,
            model=model,
            timeout=timeout,
            env_id=env_id,
            task_id=str(task_payload.get("task_id") or batch_id),
            annotation_text=annotation_text,
            image_paths=[initial_image_path],
        )
        summary_path = summaries_root / f"{batch_id}.json"
        _write_json(
            summary_path,
            {
                "batch_id": batch_id,
                "summary_text": summary_text,
                "source_task_id": str(task_payload.get("task_id") or ""),
                "annotation_pattern_count": int(annotation_pattern_count),
            },
        )
        with counters_lock:
            summary_completed += 1
        _emit_progress()
        return {
            "batch_id": batch_id,
            "summary_text": summary_text,
            "annotation_path": str(annotation_path),
            "summary_path": str(summary_path),
            "annotation_pattern_count": int(annotation_pattern_count),
        }

    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        future_map = {executor.submit(_run_one, task_payload): task_payload for task_payload in task_batch}
        for future in as_completed(future_map):
            item = future.result()
            results[str(item["batch_id"])] = item
    return results


def _run_discrimination_questions(
    *,
    server_root: Path,
    env_id: str,
    base_url: str,
    model: str,
    timeout: int,
    workers: int,
    question_set: list[dict[str, Any]],
    task_batch: list[dict[str, Any]],
    media_manifest: dict[str, dict[str, Any]],
    summaries_by_task: dict[str, dict[str, Any]],
    trial_root: Path,
    summary_progress_completed: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    results_root = trial_root / "discrimination"
    results_root.mkdir(parents=True, exist_ok=True)
    task_lookup = {str(item.get("batch_id") or ""): item for item in task_batch}
    total_units = len(task_batch) * 2
    completed_questions = 0

    def _load_cached_question(question_index: int) -> dict[str, Any] | None:
        result_path = results_root / f"question-{question_index:03d}.json"
        if not result_path.is_file():
            return None
        payload = _load_json(result_path)
        if not isinstance(payload, dict):
            return None
        if "correct" not in payload:
            return None
        return payload

    def _run_one(question_index: int, question: dict[str, Any]) -> dict[str, Any]:
        cached = _load_cached_question(question_index)
        if cached is not None:
            return cached
        target_batch_id = str(question.get("target_batch_id") or "").strip()
        option_batch_ids = [
            str(item).strip()
            for item in question.get("option_batch_ids") or []
            if str(item).strip()
        ]
        if target_batch_id not in task_lookup:
            raise ValueError(f"Unknown target task {target_batch_id}")
        option_count = _discrimination_option_count(env_id)
        if len(option_batch_ids) != option_count:
            raise ValueError(f"Question {question_index} does not have {option_count} options")
        option_texts = []
        option_annotation_counts: list[int] = []
        for batch_id in option_batch_ids:
            summary_entry = summaries_by_task.get(batch_id)
            if not isinstance(summary_entry, dict):
                raise ValueError(f"Missing summary for {batch_id}")
            option_texts.append(str(summary_entry.get("summary_text") or "").strip())
            option_annotation_counts.append(int(summary_entry.get("annotation_pattern_count") or 0))
        correct_choice = str(question.get("correct_choice") or "").strip().upper()
        if option_annotation_counts and all(count <= 0 for count in option_annotation_counts):
            result = {
                "question_index": question_index,
                "target_batch_id": target_batch_id,
                "option_batch_ids": option_batch_ids,
                "correct_choice": correct_choice,
                "predicted_choice": "",
                "correct": False,
                "raw_response_text": f"Skipped LLM call: all {option_count} option summaries came from zero-pattern annotations.",
                "skipped_llm": True,
                "skip_reason": "all_option_annotations_empty",
                "option_annotation_counts": option_annotation_counts,
            }
            _write_json(results_root / f"question-{question_index:03d}.json", result)
            return result
        media_entry = dict(media_manifest.get(target_batch_id) or {})
        frame_paths = [Path(str(path)).resolve() for path in media_entry.get("frame_paths") or []]
        if not frame_paths:
            raise ValueError(f"Missing frame cache for {target_batch_id}")
        frame_paths = _sample_paths_evenly(frame_paths, DEFAULT_DISCRIMINATION_FRAME_LIMIT)
        raw_choice, raw_text = _request_choice(
            project_root=server_root,
            base_url=base_url,
            model=model,
            timeout=timeout,
            env_id=env_id,
            task_id=str(task_lookup[target_batch_id].get("task_id") or target_batch_id),
            option_texts=option_texts,
            image_paths=frame_paths,
        )
        result = {
            "question_index": question_index,
            "target_batch_id": target_batch_id,
            "option_batch_ids": option_batch_ids,
            "correct_choice": correct_choice,
            "predicted_choice": raw_choice,
            "correct": raw_choice == correct_choice,
            "raw_response_text": raw_text,
            "option_annotation_counts": option_annotation_counts,
        }
        _write_json(results_root / f"question-{question_index:03d}.json", result)
        return result

    question_results: list[dict[str, Any]] = []
    cached_results: list[dict[str, Any]] = []
    pending_questions: list[tuple[int, dict[str, Any]]] = []
    for question_index, question in enumerate(question_set):
        cached = _load_cached_question(question_index)
        if cached is not None:
            cached_results.append(cached)
        else:
            pending_questions.append((question_index, question))
    completed_questions = len(cached_results)
    if progress_callback is not None and completed_questions:
        progress_callback(
            {
                "eval_completed_tasks": summary_progress_completed + completed_questions,
                "eval_started_tasks": summary_progress_completed + completed_questions,
                "eval_total_tasks": total_units,
            }
        )
    print(
        f"[ensemble-calibration] discrimination cached={len(cached_results)} pending={len(pending_questions)}",
        flush=True,
    )
    question_results.extend(cached_results)
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        future_map = {
            executor.submit(_run_one, question_index, question): question_index
            for question_index, question in pending_questions
        }
        for future in as_completed(future_map):
            result = future.result()
            question_results.append(result)
            completed_questions += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "eval_completed_tasks": summary_progress_completed + completed_questions,
                        "eval_started_tasks": summary_progress_completed + completed_questions,
                        "eval_total_tasks": total_units,
                    }
                )
    question_results.sort(key=lambda item: int(item.get("question_index") or 0))
    correct_count = sum(1 for item in question_results if bool(item.get("correct")))
    accuracy = correct_count / float(len(question_results)) if question_results else 0.0
    payload = {
        "question_count": len(question_results),
        "correct_count": correct_count,
        "accuracy": accuracy,
        "questions": question_results,
    }
    _write_json(trial_root / "discrimination_results.json", payload)
    return payload


def _run_discrimination_trial(
    *,
    server_root: Path,
    env_id: str,
    model: str,
    ollama_base_url: str,
    model_timeout: int,
    workers: int,
    trial_library_path: Path,
    output_root: Path,
    task_batch: list[dict[str, Any]],
    question_set: list[dict[str, Any]],
    media_manifest: dict[str, dict[str, Any]],
    trial_root: Path,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[int, float, dict[str, Any]]:
    summaries_by_task = _generate_trial_summaries(
        server_root=server_root,
        env_id=env_id,
        base_url=ollama_base_url,
        model=model,
        timeout=model_timeout,
        workers=workers,
        task_batch=task_batch,
        media_manifest=media_manifest,
        trial_library_path=trial_library_path,
        trial_root=trial_root,
        progress_callback=progress_callback,
    )
    print(
        f"[ensemble-calibration] summaries ready count={len(summaries_by_task)} trial_root={trial_root}",
        flush=True,
    )
    discrimination_payload = _run_discrimination_questions(
        server_root=server_root,
        env_id=env_id,
        base_url=ollama_base_url,
        model=model,
        timeout=model_timeout,
        workers=workers,
        question_set=question_set,
        task_batch=task_batch,
        media_manifest=media_manifest,
        summaries_by_task=summaries_by_task,
        trial_root=trial_root,
        summary_progress_completed=len(task_batch),
        progress_callback=progress_callback,
    )
    summary_payload = {
        "objective": "summary_discrimination_accuracy",
        "task_count": len(task_batch),
        "question_count": int(discrimination_payload.get("question_count") or 0),
        "accuracy": float(discrimination_payload.get("accuracy") or 0.0),
        "correct_count": int(discrimination_payload.get("correct_count") or 0),
        "summaries_root": str(trial_root / "summaries"),
        "annotations_root": str(trial_root / "annotations"),
        "discrimination_results_path": str(trial_root / "discrimination_results.json"),
    }
    return 0, float(summary_payload["accuracy"]), summary_payload


def _annotation_pattern_count(payload: dict[str, Any]) -> int:
    if not isinstance(payload, dict):
        return 0
    annotations = payload.get("pattern_annotations")
    if isinstance(annotations, dict):
        metadata = annotations.get("metadata")
        if isinstance(metadata, dict):
            total = metadata.get("total_pattern_count")
            if isinstance(total, int):
                return int(total)
        patterns = annotations.get("patterns")
        if isinstance(patterns, list):
            return len(patterns)
    patterns = payload.get("patterns")
    if isinstance(patterns, list):
        return len(patterns)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        total = metadata.get("total_pattern_count")
        if isinstance(total, int):
            return int(total)
    return 0


def run_local_ensemble_calibration(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ensemble_library_training.ensemble_calibration",
        description="Locally optimize ensemble detector reliabilities for summary-discrimination accuracy.",
    )
    parser.add_argument("--env", required=True)
    parser.add_argument("--library-path", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:8788")
    parser.add_argument("--model-timeout", type=int, default=120)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--trials", type=int, default=DEFAULT_TOTAL_TRIALS)
    parser.add_argument("--random-init-trials", type=int, default=DEFAULT_RANDOM_INIT_TRIALS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=0)
    parser.add_argument("--write-sidecar", action="store_true")
    args = parser.parse_args(argv)

    env_id = str(args.env).strip().lower()
    workers = max(1, int(args.workers or _default_eval_workers(env_id)))
    task_count = _normalized_task_count(env_id, int(args.task_count) if int(args.task_count or 0) > 0 else None)
    library_path = Path(args.library_path).resolve()
    library = load_pattern_library(library_path)
    if not _supported_ensembles(library):
        raise ValueError(f"Library {library_path} has no supervised/unsupervised ensembles to calibrate")

    run_id = str(args.run_id or default_run_id()).strip()
    output_root = (
        Path(args.output_root).resolve()
        if str(args.output_root or "").strip()
        else library_path.parent / "ensemble_calibration_runs" / run_id
    )
    output_root.mkdir(parents=True, exist_ok=True)
    history = _load_existing_trial_history(output_root)
    best_trial = max(history, key=lambda item: float(item.get("score") or 0.0)) if history else None
    resumed_trial_count = len(history)
    started_at = _resume_started_at(output_root, history) or _utc_now()
    _write_json(
        output_root / "request.json",
        {
            "env_id": env_id,
            "library_path": str(library_path),
            "run_id": run_id,
            "trials": int(args.trials),
            "random_init_trials": int(args.random_init_trials),
            "seed": int(args.seed),
            "task_count": task_count,
            "model": str(args.model or ""),
            "ollama_base_url": str(args.ollama_base_url),
            "model_timeout": int(args.model_timeout),
            "workers": workers,
            "objective": "summary_discrimination_accuracy",
            "task_batch_path": str(output_root / "task_batch.json"),
            "question_set_path": str(output_root / "question_set.json"),
            "resume_completed_trials": resumed_trial_count,
            "updated_at": _isoformat(),
        },
    )
    _write_json(
        output_root / "progress.json",
        {
            "run_id": run_id,
            "status": "starting" if resumed_trial_count == 0 else "resuming",
            "started_at": _isoformat(started_at),
            "updated_at": _isoformat(),
            "completed_trials": resumed_trial_count,
            "total_trials": max(1, int(args.trials)),
            "best_score": float(best_trial["score"]) if best_trial is not None else None,
            "last_score": float(history[-1]["score"]) if history else None,
            "average_trial_seconds": (
                sum(float(item.get("trial_seconds") or 0.0) for item in history) / len(history)
                if history
                else None
            ),
            "eta_seconds": (
                (
                    sum(float(item.get("trial_seconds") or 0.0) for item in history) / len(history)
                )
                * float(max(0, int(args.trials) - resumed_trial_count))
                if history
                else None
            ),
            "current_trial_index": resumed_trial_count,
            "current_trial_mode": "preparing-fixed-batch" if resumed_trial_count == 0 else "resume-scan",
            "current_eval_completed_tasks": 0,
            "current_eval_started_tasks": 0,
            "current_eval_total_tasks": 0,
        },
    )
    server_root = _server_project_root()
    task_batch = _load_or_create_task_batch(
        server_root=server_root,
        env_id=env_id,
        output_root=output_root,
        seed=int(args.seed),
        task_count=task_count,
    )
    question_set = _load_or_create_question_set(
        output_root=output_root,
        task_batch=task_batch,
        seed=int(args.seed),
    )
    media_manifest = _load_or_create_media_manifest(
        server_root=server_root,
        output_root=output_root,
        task_batch=task_batch,
        seed=int(args.seed),
    )
    _write_json(
        output_root / "progress.json",
        {
            "run_id": run_id,
            "status": "starting" if resumed_trial_count == 0 else "resuming",
            "started_at": _isoformat(started_at),
            "updated_at": _isoformat(),
            "completed_trials": resumed_trial_count,
            "total_trials": max(1, int(args.trials)),
            "best_score": float(best_trial["score"]) if best_trial is not None else None,
            "last_score": float(history[-1]["score"]) if history else None,
            "average_trial_seconds": (
                sum(float(item.get("trial_seconds") or 0.0) for item in history) / len(history)
                if history
                else None
            ),
            "eta_seconds": (
                (
                    sum(float(item.get("trial_seconds") or 0.0) for item in history) / len(history)
                )
                * float(max(0, int(args.trials) - resumed_trial_count))
                if history
                else None
            ),
            "current_trial_index": resumed_trial_count,
            "current_trial_mode": "starting" if resumed_trial_count == 0 else "resuming",
            "current_eval_completed_tasks": 0,
            "current_eval_started_tasks": 0,
            "current_eval_total_tasks": len(task_batch) * 2,
        },
    )

    rng = random.Random(int(args.seed))
    calibration_template = _default_calibration_payload(library)
    specs = _parameter_specs(library)
    initial_vector = _vector_from_payload(calibration_template, specs)
    print(
        f"[ensemble-calibration] starting env={env_id} run_id={run_id} "
        f"trials={max(1, int(args.trials))} random_init={max(1, int(args.random_init_trials))} "
        f"objective=summary_discrimination_accuracy tasks={len(task_batch)} questions={len(question_set)}",
        flush=True,
    )
    if resumed_trial_count:
        print(
            f"[ensemble-calibration] resuming env={env_id} run_id={run_id} "
            f"completed_trials={resumed_trial_count}/{max(1, int(args.trials))}",
            flush=True,
        )
    if resumed_trial_count >= max(1, int(args.trials)):
        _write_json(
            output_root / "summary.json",
            {
                "run_id": run_id,
                "env_id": env_id,
                "library_path": str(library_path),
                "trial_count": len(history),
                "best_score": float(best_trial["score"]) if best_trial is not None else None,
                "best_trial_index": int(best_trial["trial_index"]) if best_trial is not None else None,
                "best_calibration_path": (
                    str(output_root / "ensemble_calibration.json")
                    if best_trial is not None
                    else ""
                ),
                "objective": "summary_discrimination_accuracy",
                "task_batch_path": str(output_root / "task_batch.json"),
                "question_set_path": str(output_root / "question_set.json"),
                "started_at": _isoformat(started_at),
                "finished_at": _isoformat(),
            },
        )
        print(
            f"[ensemble-calibration] already complete env={env_id} run_id={run_id} "
            f"best={float(best_trial['score']) if best_trial is not None else 0.0:.4f}",
            flush=True,
        )
        return 0

    for trial_index in range(resumed_trial_count, max(1, int(args.trials))):
        trial_mode = "guided"
        if trial_index == 0:
            vector = list(initial_vector)
            trial_mode = "initial"
        elif trial_index < max(1, int(args.random_init_trials)):
            vector = _random_vector(specs, rng)
            trial_mode = "random"
        else:
            best_vector = [float(item) for item in (best_trial or {}).get("vector", initial_vector)]
            vector = _propose_next_vector(
                specs=specs,
                history=history,
                best_vector=best_vector,
                rng=rng,
            )
            trial_mode = "acquisition"
        calibration_payload = _payload_from_vector(
            template_payload=calibration_template,
            specs=specs,
            vector=vector,
        )
        trial_root = output_root / "trials" / f"trial-{trial_index:03d}"
        trial_root.mkdir(parents=True, exist_ok=True)
        trial_library_path = trial_root / "pattern_library.json"
        trial_started_at = _utc_now()
        _write_json(
            output_root / "progress.json",
            {
                "run_id": run_id,
                "status": "running",
                "started_at": _isoformat(started_at),
                "updated_at": _isoformat(trial_started_at),
                "completed_trials": trial_index,
                "total_trials": max(1, int(args.trials)),
                "best_score": float(best_trial["score"]) if best_trial is not None else None,
                "last_score": float(history[-1]["score"]) if history else None,
                "average_trial_seconds": (
                    sum(float(item.get("trial_seconds") or 0.0) for item in history) / len(history)
                    if history
                    else None
                ),
                "eta_seconds": (
                    (
                        sum(float(item.get("trial_seconds") or 0.0) for item in history) / len(history)
                    )
                    * float(max(0, int(args.trials) - trial_index))
                    if history
                    else None
                ),
                "current_trial_index": trial_index + 1,
                "current_trial_mode": trial_mode,
                "current_eval_completed_tasks": 0,
                "current_eval_started_tasks": 0,
                "current_eval_total_tasks": len(task_batch) * 2,
            },
        )
        print(
            f"[ensemble-calibration] trial {trial_index + 1}/{max(1, int(args.trials))} "
            f"mode={trial_mode} starting",
            flush=True,
        )
        save_pattern_library(library, trial_library_path)
        _write_json(trial_root / "ensemble_calibration.json", calibration_payload)
        last_logged_eval_completed = -1

        def _update_running_progress(progress_update: dict[str, Any]) -> None:
            nonlocal last_logged_eval_completed
            eval_completed = int(progress_update.get("eval_completed_tasks") or 0)
            eval_started = int(progress_update.get("eval_started_tasks") or eval_completed)
            eval_total = int(progress_update.get("eval_total_tasks") or (len(task_batch) * 2))
            progress_payload = {
                "run_id": run_id,
                "status": "running",
                "started_at": _isoformat(started_at),
                "updated_at": _isoformat(),
                "completed_trials": trial_index,
                "total_trials": max(1, int(args.trials)),
                "best_score": float(best_trial["score"]) if best_trial is not None else None,
                "last_score": float(history[-1]["score"]) if history else None,
                "average_trial_seconds": (
                    sum(float(item.get("trial_seconds") or 0.0) for item in history) / len(history)
                    if history
                    else None
                ),
                "eta_seconds": (
                    (
                        sum(float(item.get("trial_seconds") or 0.0) for item in history) / len(history)
                    )
                    * float(max(0, int(args.trials) - trial_index))
                    if history
                    else None
                ),
                "current_trial_index": trial_index + 1,
                "current_trial_mode": trial_mode,
                "current_eval_completed_tasks": eval_completed,
                "current_eval_started_tasks": eval_started,
                "current_eval_total_tasks": eval_total,
            }
            _write_json(output_root / "progress.json", progress_payload)
            if eval_total > 0 and eval_completed != last_logged_eval_completed:
                print(
                    f"[ensemble-calibration] trial {trial_index + 1}/{max(1, int(args.trials))} "
                    f"eval {eval_completed}/{eval_total} units complete",
                    flush=True,
                )
                last_logged_eval_completed = eval_completed

        return_code, score_value, summary_payload = _run_discrimination_trial(
            server_root=server_root,
            env_id=env_id,
            model=str(args.model or ""),
            ollama_base_url=str(args.ollama_base_url),
            model_timeout=int(args.model_timeout),
            workers=workers,
            trial_library_path=trial_library_path,
            output_root=output_root,
            task_batch=task_batch,
            question_set=question_set,
            media_manifest=media_manifest,
            trial_root=trial_root,
            progress_callback=_update_running_progress,
        )
        trial_finished_at = _utc_now()
        trial_seconds = max(0.0, (trial_finished_at - trial_started_at).total_seconds())
        trial_payload = {
            "trial_index": trial_index,
            "proposal_mode": trial_mode,
            "started_at": _isoformat(trial_started_at),
            "finished_at": _isoformat(trial_finished_at),
            "trial_seconds": float(trial_seconds),
            "vector": vector,
            "score": float(score_value),
            "return_code": int(return_code),
            "calibration_path": str(trial_root / "ensemble_calibration.json"),
            "evaluation_root": str(trial_root),
            "summary": summary_payload,
        }
        _write_json(trial_root / "trial.json", trial_payload)
        history.append(trial_payload)

        if trial_index == 0:
            option_count = _current_option_count(output_root, env_id)
            target = _target_max_initial_score()
            score_float = float(score_value)
            if score_float > target:
                limit = min(len(_CHOICE_LETTERS), len(task_batch))
                if option_count < limit:
                    suggested = int(min(limit, option_count + 1))
                    _write_json(
                        output_root / OPTION_TUNING_FILENAME,
                        {
                            "env_id": env_id,
                            "run_id": run_id,
                            "observed_initial_score": score_float,
                            "target_max_initial_score": float(target),
                            "current_option_count": int(option_count),
                            "suggested_min_option_count": int(suggested),
                            "updated_at": _isoformat(),
                        },
                    )
                    print(
                        f"[ensemble-calibration] initial score={score_float:.4f} is above target={target:.4f}; "
                        f"suggested_min_option_count={suggested}",
                        flush=True,
                    )
        if best_trial is None or float(trial_payload["score"]) > float(best_trial["score"]):
            best_trial = trial_payload
            _write_json(output_root / "best.json", best_trial)
            _write_json(output_root / "ensemble_calibration.json", calibration_payload)
            if args.write_sidecar:
                _write_json(library_path.parent / "ensemble_calibration.json", calibration_payload)
            print(
                f"[ensemble-calibration] new best trial={trial_index + 1} score={score_value:.4f}",
                flush=True,
            )
        average_trial_seconds = (
            sum(float(item.get("trial_seconds") or 0.0) for item in history) / len(history)
            if history
            else None
        )
        remaining_trials = max(0, int(args.trials) - (trial_index + 1))
        eta_seconds = (
            float(average_trial_seconds) * float(remaining_trials)
            if average_trial_seconds is not None
            else None
        )
        _write_json(
            output_root / "progress.json",
            {
                "run_id": run_id,
                "status": "running" if trial_index + 1 < max(1, int(args.trials)) else "completed",
                "started_at": _isoformat(started_at),
                "updated_at": _isoformat(trial_finished_at),
                "completed_trials": trial_index + 1,
                "total_trials": max(1, int(args.trials)),
                "best_score": float(best_trial["score"]) if best_trial is not None else None,
                "last_score": float(score_value),
                "average_trial_seconds": average_trial_seconds,
                "eta_seconds": eta_seconds,
                "current_trial_index": min(max(1, int(args.trials)), trial_index + 1),
                "current_trial_mode": trial_mode,
                "current_eval_completed_tasks": len(task_batch) * 2,
                "current_eval_started_tasks": len(task_batch) * 2,
                "current_eval_total_tasks": len(task_batch) * 2,
            },
        )
        print(
            f"[ensemble-calibration] trial {trial_index + 1}/{max(1, int(args.trials))} "
            f"score={score_value:.4f} best={float(best_trial['score']) if best_trial is not None else 0.0:.4f} "
            f"duration={trial_seconds:.1f}s",
            flush=True,
        )

    _write_json(
        output_root / "summary.json",
        {
            "run_id": run_id,
            "env_id": env_id,
            "library_path": str(library_path),
            "trial_count": len(history),
            "best_score": float(best_trial["score"]) if best_trial is not None else None,
            "best_trial_index": int(best_trial["trial_index"]) if best_trial is not None else None,
            "best_calibration_path": (
                str(output_root / "ensemble_calibration.json")
                if best_trial is not None
                else ""
            ),
            "objective": "summary_discrimination_accuracy",
            "task_batch_path": str(output_root / "task_batch.json"),
            "question_set_path": str(output_root / "question_set.json"),
            "started_at": _isoformat(started_at),
            "finished_at": _isoformat(),
        },
    )
    print(
        f"[ensemble-calibration] completed env={env_id} run_id={run_id} "
        f"best={float(best_trial['score']) if best_trial is not None else 0.0:.4f}",
        flush=True,
    )
    return 0


def main() -> int:
    raise SystemExit(run_local_ensemble_calibration(sys.argv[1:]))


if __name__ == "__main__":
    main()
