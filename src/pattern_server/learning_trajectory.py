from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_reward_trajectory(run_dir: Path) -> dict[str, Any]:
    series: list[dict[str, Any]] = []
    max_iteration = -1
    history_by_label: dict[str, list[dict[str, Any]]] = {}
    palette_by_label: dict[str, int] = {}
    seen: set[tuple[str, str | None, int | None]] = set()
    sequence = 0

    for path in _resolve_evolution_paths(run_dir):
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            label = str(payload.get("label") or "").strip()
            if not label:
                continue
            candidate_uid = _coerce_string(payload.get("candidate_uid"))
            iteration = _coerce_int(payload.get("iteration"))
            dedupe_key = (label, candidate_uid, iteration)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            sequence += 1
            record = {
                "sequence": sequence,
                "iteration_hint": iteration if isinstance(iteration, int) and iteration > 0 else None,
                "train_reward": _coerce_float(payload.get("train_score"), fallback=_coerce_float(payload.get("score"))),
                "validation_reward": _coerce_float(payload.get("val_score")),
            }
            history_by_label.setdefault(label, []).append(record)
            palette_index = _coerce_int(payload.get("label_index"))
            if palette_index is not None and label not in palette_by_label:
                palette_by_label[label] = max(0, palette_index - 1)

    # Some resumed runs can miss evolution JSONL history for labels that are
    # already complete; backfill from label_result.json so charts and label
    # completion counters stay aligned.
    for item in _label_result_fallback_records(run_dir):
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        if label in history_by_label:
            continue
        sequence += 1
        history_by_label[label] = [
            {
                "sequence": sequence,
                "iteration_hint": _coerce_int(item.get("candidate_evaluations")) or 1,
                "train_reward": _coerce_float(item.get("train_reward")),
                "validation_reward": _coerce_float(item.get("validation_reward")),
            }
        ]
        label_index = _coerce_int(item.get("label_index"))
        if label_index is not None and label not in palette_by_label:
            palette_by_label[label] = max(0, label_index - 1)

    if not history_by_label:
        return {"series": [], "max_iteration": None}

    sorted_label_names = sorted(
        history_by_label,
        key=lambda label: (palette_by_label.get(label, 10_000), label),
    )
    for fallback_index, label in enumerate(sorted_label_names):
        values: list[dict[str, Any]] = []
        for index, record in enumerate(
            sorted(history_by_label[label], key=lambda item: int(item["sequence"])),
            start=1,
        ):
            iteration = _coerce_int(record.get("iteration_hint")) or index
            train_reward = record.get("train_reward")
            validation_reward = record.get("validation_reward")
            if train_reward is None and validation_reward is None:
                continue
            values.append(
                {
                    "iteration": iteration,
                    "train_reward": train_reward,
                    "validation_reward": validation_reward,
                }
            )
        if values:
            max_iteration = max(max_iteration, max(point["iteration"] for point in values))
            series.append(
                {
                    "label": label,
                    "palette_index": palette_by_label.get(label, fallback_index),
                    "values": values,
                }
            )
    return {
        "series": series,
        "max_iteration": max_iteration if max_iteration >= 0 else None,
    }


def _resolve_evolution_paths(run_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for candidate in (
        run_dir / "evolution_record.jsonl",
        run_dir / "evolution_live.jsonl",
    ):
        if candidate.is_file() and candidate not in paths:
            paths.append(candidate)
    return paths


def _label_result_fallback_records(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        paths = sorted(run_dir.glob("label_*/label_result.json"))
    except OSError:
        return records
    for path in paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        label = str(payload.get("label") or "").strip()
        if not label:
            continue
        records.append(
            {
                "label": label,
                "label_index": _coerce_int(payload.get("label_index")),
                "candidate_evaluations": _coerce_int(payload.get("candidate_evaluations")),
                "train_reward": _coerce_float(
                    payload.get("best_train_score"),
                    fallback=_coerce_float(payload.get("best_score")),
                ),
                "validation_reward": _coerce_float(payload.get("best_validation_score")),
            }
        )
    return records


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any, *, fallback: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
