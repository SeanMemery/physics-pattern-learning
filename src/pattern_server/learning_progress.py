from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_LABEL_LINE_RE = re.compile(r"label=(.+?)(?: (?=[a-z_]+=)|$)")
_INT_FIELD_RE = re.compile(
    r"(?:^| )(?P<key>worker|index|total_labels|evaluation_index|active_workers|desired_workers|queued_labels)=(?P<value>\d+)"
)
_FLOAT_FIELD_RE = re.compile(r"(?:^| )(?P<key>score|validation_score)=(-?\d+(?:\.\d+)?)")
_FINGERPRINT_FIELD_RE = re.compile(r"(?:^| )candidate_fingerprint=([^ ]+)")


def summarize_learning_progress(
    *,
    run_dir: Path | None,
    lines: list[str],
    status: str,
    fallback_progress: dict[str, object] | None = None,
) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    run_payload = _load_run_payload(run_dir)
    total_labels = _coerce_int(run_payload.get("primitive_label_count")) or 0
    payload_completed_labels = _coerce_int(run_payload.get("completed_label_count")) or 0
    completed_label_results = _completed_label_result_count(run_dir)
    has_label_dirs = any(path.is_dir() for path in run_dir.glob("label_*"))
    # Prefer concrete per-label completion files when label directories exist.
    # `run.json` can become stale (for example after resume/copy operations).
    if has_label_dirs:
        completed_labels = completed_label_results
    else:
        completed_labels = max(payload_completed_labels, completed_label_results)
    if total_labels > 0:
        completed_labels = max(0, min(completed_labels, total_labels))
    parsed = _parse_learning_recent_output(lines)

    workers = parsed.get("workers", [])
    stage = parsed.get("stage")
    active_label = parsed.get("active_label")
    active_label_index = parsed.get("active_label_index")
    if status == "paused":
        stage = "paused"
        workers = [{**worker, "stage": "paused"} for worker in workers]
    elif status == "failed":
        stage = "failed"
        workers = [{**worker, "stage": "failed"} for worker in workers]
    elif status == "queued" and not stage:
        stage = "queued"
        workers = [{**worker, "stage": worker.get("stage") or "queued"} for worker in workers]

    desired_workers = (
        _coerce_int(run_payload.get("configured_workers"))
        or _coerce_int(run_payload.get("workers"))
        or 1
    )
    if not workers and active_label is not None:
        workers = _synthesized_learning_workers(
            active_label=active_label,
            active_label_index=active_label_index,
            stage=stage,
            candidate_evaluations=parsed.get("candidate_evaluations"),
            latest_score=parsed.get("latest_score"),
            latest_validation_score=parsed.get("latest_validation_score"),
            desired_workers=desired_workers,
        )

    latest_score = parsed.get("latest_score")
    latest_validation_score = parsed.get("latest_validation_score")
    candidate_evaluations = parsed.get("candidate_evaluations")
    if isinstance(fallback_progress, dict):
        latest_score = latest_score if latest_score is not None else _coerce_float(fallback_progress.get("summary"))
        latest_validation_score = latest_validation_score if latest_validation_score is not None else _coerce_float(fallback_progress.get("validation"))
        candidate_evaluations = candidate_evaluations if candidate_evaluations is not None else _coerce_int(fallback_progress.get("current"))
        stage = stage or _fallback_stage(fallback_progress)

    provisional_labels = _merge_provisional_and_completed_labels(
        run_dir,
        provisional_labels=parsed.get("provisional_labels", []),
    )

    return {
        "completed_labels": completed_labels,
        "total_labels": total_labels,
        "workers": workers,
        "active_label": active_label,
        "active_label_index": active_label_index,
        "candidate_evaluations": candidate_evaluations,
        "latest_score": latest_score,
        "latest_validation_score": latest_validation_score,
        "stage": stage,
        "provisional_labels": provisional_labels,
    }


def display_learning_stage(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return "—"
    mapping = {
        "label": "Starting label",
        "search": "Evaluating",
        "llm": "Writing code",
        "label_done": "Label done",
        "queued": "Queued",
        "paused": "Paused",
        "failed": "Failed",
    }
    return mapping.get(value, value.replace("_", " ").title())


def _load_run_payload(run_dir: Path) -> dict[str, Any]:
    run_json = run_dir / "run.json"
    if not run_json.is_file():
        return {}
    try:
        payload = json.loads(run_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _completed_label_result_count(run_dir: Path) -> int:
    count = 0
    try:
        for path in run_dir.glob("label_*/label_result.json"):
            if path.is_file():
                count += 1
    except OSError:
        return 0
    return count


def _completed_label_payloads(run_dir: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    try:
        paths = sorted(run_dir.glob("label_*/label_result.json"))
    except OSError:
        return payloads
    for path in paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _merge_provisional_and_completed_labels(
    run_dir: Path,
    *,
    provisional_labels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in provisional_labels:
        if not isinstance(item, dict):
            continue
        label_name = str(item.get("label") or "").strip()
        if not label_name:
            continue
        merged[_normalized_label_name(label_name)] = dict(item)

    for payload in _completed_label_payloads(run_dir):
        label_name = str(payload.get("label") or "").strip()
        if not label_name:
            continue
        label_index = _coerce_int(payload.get("label_index"))
        key = _normalized_label_name(label_name)
        entry = merged.get(key)
        if entry is None:
            entry = {
                "label": label_name,
                "label_index": label_index,
                "candidate_evaluations": 0,
                "event_count": 0,
                "best_score": None,
                "best_validation_score": None,
                "best_fingerprint": None,
                "backfilled_from_label_result": True,
            }
            merged[key] = entry
        if entry.get("label_index") is None and label_index is not None:
            entry["label_index"] = label_index
        candidate_evaluations = _coerce_int(payload.get("candidate_evaluations"))
        if candidate_evaluations is not None:
            entry["candidate_evaluations"] = max(
                _coerce_int(entry.get("candidate_evaluations")) or 0,
                candidate_evaluations,
            )
        best_score = _coerce_float(payload.get("best_score"))
        if best_score is not None:
            existing_score = _coerce_float(entry.get("best_score"))
            if existing_score is None or best_score >= existing_score:
                entry["best_score"] = best_score
                entry["best_validation_score"] = _coerce_float(payload.get("best_validation_score"))
        best_candidate_uid = str(payload.get("best_candidate_uid") or "").strip()
        if best_candidate_uid and not entry.get("best_fingerprint"):
            entry["best_fingerprint"] = best_candidate_uid

    return sorted(
        merged.values(),
        key=lambda item: (
            item["label_index"] if isinstance(item.get("label_index"), int) else 10_000,
            str(item.get("label") or ""),
        ),
    )


def _parse_learning_recent_output(lines: list[str]) -> dict[str, Any]:
    state: dict[str, Any] = {
        "active_label": None,
        "active_label_index": None,
        "candidate_evaluations": None,
        "latest_score": None,
        "latest_validation_score": None,
        "stage": None,
        "workers": [],
    }
    workers: dict[int, dict[str, Any]] = {}
    labels: dict[str, dict[str, Any]] = {}

    def _reset_runtime_state(*, keep_stage: bool = False) -> None:
        workers.clear()
        labels.clear()
        state["active_label"] = None
        state["active_label_index"] = None
        state["candidate_evaluations"] = None
        state["latest_score"] = None
        state["latest_validation_score"] = None
        if not keep_stage:
            state["stage"] = None

    def _worker_state(worker_id: int) -> dict[str, Any]:
        worker_state = workers.get(worker_id)
        if worker_state is None:
            worker_state = {
                "worker": worker_id,
                "label": None,
                "label_index": None,
                "candidate_evaluations": None,
                "latest_score": None,
                "latest_validation_score": None,
                "stage": None,
            }
            workers[worker_id] = worker_state
        return worker_state

    def _label_state(*, label_name: str, label_index: int | None) -> dict[str, Any]:
        label_key = _label_key(label_name, label_index)
        state_key = label_key or label_name
        label_state = labels.get(state_key)
        if label_state is None:
            label_state = {
                "label": label_name,
                "label_index": label_index,
                "candidate_evaluations": 0,
                "event_count": 0,
                "best_score": None,
                "best_validation_score": None,
                "best_fingerprint": None,
            }
            labels[state_key] = label_state
        elif label_index is not None and label_state.get("label_index") is None:
            label_state["label_index"] = label_index
        return label_state

    for line in lines:
        if "patterns.learning." not in line:
            continue
        label_match = _LABEL_LINE_RE.search(line)
        label = label_match.group(1) if label_match else None
        fingerprint_match = _FINGERPRINT_FIELD_RE.search(line)
        fingerprint = fingerprint_match.group(1) if fingerprint_match else None
        ints = {
            match.group("key"): int(match.group("value"))
            for match in _INT_FIELD_RE.finditer(line)
        }
        floats = {
            match.group("key"): float(match.group(2))
            for match in _FLOAT_FIELD_RE.finditer(line)
        }
        if "patterns.learning.start:" in line or "patterns.learning.resume:" in line:
            _reset_runtime_state()
            continue
        if "patterns.learning.workers.updated" in line:
            desired_workers = ints.get("desired_workers")
            active_workers = ints.get("active_workers")
            queued_labels = ints.get("queued_labels")
            if active_workers == 0 and (
                desired_workers is None
                or queued_labels is None
                or desired_workers <= queued_labels
            ):
                _reset_runtime_state(keep_stage=True)
            continue
        worker_id = ints.get("worker")
        worker_state = _worker_state(worker_id) if worker_id is not None else None
        if "patterns.learning.label.start" in line:
            state["stage"] = "label"
            state["active_label"] = label
            state["active_label_index"] = ints.get("index")
            state["candidate_evaluations"] = None
            if worker_state is not None:
                worker_state["stage"] = "label"
                worker_state["label"] = label
                worker_state["label_index"] = ints.get("index")
                worker_state["candidate_evaluations"] = None
            if label is not None:
                _label_state(label_name=label, label_index=ints.get("index"))
            continue
        if "patterns.learning.candidate.start" in line:
            state["stage"] = "search"
            if label is not None:
                state["active_label"] = label
            if ints.get("evaluation_index") is not None:
                state["candidate_evaluations"] = ints["evaluation_index"]
            if worker_state is not None:
                worker_state["stage"] = "search"
                if label is not None:
                    worker_state["label"] = label
                if ints.get("evaluation_index") is not None:
                    worker_state["candidate_evaluations"] = ints["evaluation_index"]
            if label is not None:
                label_state = _label_state(label_name=label, label_index=worker_state.get("label_index") if worker_state is not None else None)
                if ints.get("evaluation_index") is not None:
                    label_state["candidate_evaluations"] = max(int(label_state["candidate_evaluations"]), ints["evaluation_index"])
            continue
        if "patterns.learning.lm_request.start" in line:
            state["stage"] = "llm"
            if worker_state is not None:
                worker_state["stage"] = "llm"
            continue
        if "patterns.learning.candidate.done" in line:
            state["stage"] = "search"
            if label is not None:
                state["active_label"] = label
            if ints.get("evaluation_index") is not None:
                state["candidate_evaluations"] = ints["evaluation_index"]
            if "score" in floats:
                state["latest_score"] = floats["score"]
            if "validation_score" in floats:
                state["latest_validation_score"] = floats["validation_score"]
            if worker_state is not None:
                worker_state["stage"] = "search"
                if label is not None:
                    worker_state["label"] = label
                if ints.get("evaluation_index") is not None:
                    worker_state["candidate_evaluations"] = ints["evaluation_index"]
                if "score" in floats:
                    worker_state["latest_score"] = floats["score"]
                if "validation_score" in floats:
                    worker_state["latest_validation_score"] = floats["validation_score"]
            if label is not None:
                label_state = _label_state(label_name=label, label_index=worker_state.get("label_index") if worker_state is not None else None)
                if ints.get("evaluation_index") is not None:
                    label_state["candidate_evaluations"] = max(int(label_state["candidate_evaluations"]), ints["evaluation_index"])
                label_state["event_count"] = int(label_state["event_count"]) + 1
                score = floats.get("score")
                validation_score = floats.get("validation_score")
                best_score = label_state.get("best_score")
                if score is not None and (best_score is None or score >= float(best_score)):
                    label_state["best_score"] = score
                    label_state["best_validation_score"] = validation_score
                    label_state["best_fingerprint"] = fingerprint
            continue
        if "patterns.learning.label.done" in line:
            state["stage"] = "label_done"
            if label is not None:
                state["active_label"] = label
            if ints.get("index") is not None:
                state["active_label_index"] = ints["index"]
            if "score" in floats:
                state["latest_score"] = floats["score"]
            if "validation_score" in floats:
                state["latest_validation_score"] = floats["validation_score"]
            if worker_state is not None:
                worker_state["stage"] = "label_done"
                if label is not None:
                    worker_state["label"] = label
                if ints.get("index") is not None:
                    worker_state["label_index"] = ints["index"]
                if "score" in floats:
                    worker_state["latest_score"] = floats["score"]
                if "validation_score" in floats:
                    worker_state["latest_validation_score"] = floats["validation_score"]

    provisional_labels = sorted(
        labels.values(),
        key=lambda item: (item["label_index"] if isinstance(item.get("label_index"), int) else 10_000, item["label"]),
    )
    state["workers"] = [workers[key] for key in sorted(workers)]
    state["provisional_labels"] = provisional_labels
    return state


def _synthesized_learning_workers(
    *,
    active_label: str,
    active_label_index: int | None,
    stage: str | None,
    candidate_evaluations: int | None,
    latest_score: float | None,
    latest_validation_score: float | None,
    desired_workers: int,
) -> list[dict[str, Any]]:
    synthesized: list[dict[str, Any]] = []
    for worker in range(1, max(1, desired_workers) + 1):
        synthesized.append(
            {
                "worker": worker,
                "label": active_label,
                "label_index": active_label_index,
                "candidate_evaluations": candidate_evaluations,
                "latest_score": latest_score,
                "latest_validation_score": latest_validation_score,
                "stage": stage or "queued",
            }
        )
    return synthesized


def _label_key(label_name: str, label_index: int | None) -> str | None:
    if label_index is None:
        return label_name.strip() or None
    return f"{label_index:03d}:{label_name.strip()}"


def _normalized_label_name(label_name: str) -> str:
    return label_name.strip().casefold()


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


def _fallback_stage(payload: dict[str, object]) -> str | None:
    phase = payload.get("phase")
    if not isinstance(phase, str):
        return None
    normalized = phase.strip().lower()
    mapping = {
        "learning labels": "search",
        "llm generation": "llm",
        "paused": "paused",
        "failed": "failed",
        "queued": "queued",
    }
    return mapping.get(normalized)
