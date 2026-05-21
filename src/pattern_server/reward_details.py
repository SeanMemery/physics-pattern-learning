from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import JobRecord


@dataclass(frozen=True)
class RewardOptimizationAggregate:
    budget: int
    reward_kind: str
    scene_count: int
    success_rate: float
    success_rate_std: float
    mean_reward: float
    mean_reward_std: float
    seed_count: int


@dataclass(frozen=True)
class RewardOptimizationOutcome:
    budget: int
    reward_kind: str
    scene_id: str
    seed: int
    best_reward: float
    success: bool
    action: dict[str, float]
    trace_path: Path | None
    reward_path: Path | None


@dataclass(frozen=True)
class RewardOptimizationDetail:
    run_id: str
    root: Path
    summary_path: Path
    goal_text: str
    goal_lines: list[str]
    dsl_program: str
    source_run_dir: str
    source_run_kind: str
    source_run_id: str
    model_name: str
    scene_count: int
    seed_count: int
    budgets: list[int]
    rack_jitter: float | None
    synthesized: bool
    program_text: str
    synthesis_text: str
    aggregates: list[RewardOptimizationAggregate]
    outcomes: list[RewardOptimizationOutcome]


@dataclass(frozen=True)
class RewardValueModelPoint:
    epoch: int
    train_loss: float
    validation_mean_reward: float
    validation_mean_reward_std: float
    validation_success_rate: float
    validation_success_rate_std: float


@dataclass(frozen=True)
class RewardValueModelOutcome:
    epoch: int
    scene_id: str
    reward: float
    success: bool
    action: dict[str, float]
    trace_path: Path | None
    reward_path: Path | None


@dataclass(frozen=True)
class RewardValueModelDetail:
    run_id: str
    root: Path
    summary_path: Path
    goal_text: str
    goal_lines: list[str]
    dsl_program: str
    source_run_dir: str
    source_run_kind: str
    source_run_id: str
    model_name: str
    train_scene_count: int
    train_actions_per_scene: int
    validation_scene_count: int
    validation_candidates_per_scene: int
    epochs: int
    batch_size: int
    learning_rate: float | None
    image_size: int | None
    rack_jitter: float | None
    train_sample_count: int
    best_epoch: int
    best_validation_mean_reward: float
    best_validation_success_rate: float
    synthesized: bool
    checkpoint_path: Path | None
    program_text: str
    synthesis_text: str
    eval_curve: list[RewardValueModelPoint]
    validation_outcomes: list[RewardValueModelOutcome]


def load_reward_job_detail(job: JobRecord, *, project_root: Path) -> RewardOptimizationDetail | RewardValueModelDetail | None:
    action = str(job.spec.action or "")
    run_root = _reward_run_root(job, project_root=project_root)
    if run_root is None:
        return None
    summary_path = run_root / "summary.json"
    if not summary_path.is_file():
        return None
    payload = _load_json(summary_path)
    if action == "pooltool-reward-optimize":
        return _load_optimization_detail(run_root=run_root, summary_path=summary_path, payload=payload)
    if action == "pooltool-reward-value-model":
        return _load_value_model_detail(run_root=run_root, summary_path=summary_path, payload=payload)
    return None


def _reward_run_root(job: JobRecord, *, project_root: Path) -> Path | None:
    metadata = dict(job.spec.metadata or {})
    run_dir = str(metadata.get("run_dir") or "").strip()
    if run_dir:
        candidate = _normalize_project_path(run_dir, project_root=project_root)
        if candidate is not None and candidate.exists():
            return candidate
    run_id = str(metadata.get("reward_program_run_id") or "").strip()
    if not run_id:
        return None
    kind = "optimization" if str(job.spec.action or "") == "pooltool-reward-optimize" else "value_models"
    candidate = (
        project_root
        / "data"
        / "patterns"
        / "pooltool"
        / "reward_program_experiments"
        / kind
        / run_id
    ).resolve()
    return candidate if candidate.exists() else candidate


def _load_optimization_detail(*, run_root: Path, summary_path: Path, payload: dict[str, Any]) -> RewardOptimizationDetail:
    outcomes: list[RewardOptimizationOutcome] = []
    for item in payload.get("outcomes", []):
        if not isinstance(item, dict):
            continue
        action_raw = item.get("action")
        outcomes.append(
            RewardOptimizationOutcome(
                budget=_coerce_int(item.get("budget")),
                reward_kind=str(item.get("reward_kind") or ""),
                scene_id=str(item.get("scene_id") or ""),
                seed=_coerce_int(item.get("seed")),
                best_reward=_coerce_float(item.get("best_reward")),
                success=bool(item.get("success")),
                action=action_raw if isinstance(action_raw, dict) else {},
                trace_path=_resolve_run_path(run_root, item.get("trace_path")),
                reward_path=_resolve_run_path(run_root, item.get("trace_reward_path")),
            )
        )
    aggregates: list[RewardOptimizationAggregate] = []
    raw_aggregates = payload.get("budget_aggregates", [])
    if isinstance(raw_aggregates, list) and raw_aggregates:
        for item in raw_aggregates:
            if not isinstance(item, dict):
                continue
            budget = _coerce_int(item.get("budget"))
            reward_kind = str(item.get("reward_kind") or "")
            matching_outcomes = [
                row for row in outcomes
                if row.budget == budget and row.reward_kind == reward_kind
            ]
            reward_values = [float(row.best_reward) for row in matching_outcomes]
            success_values = [1.0 if row.success else 0.0 for row in matching_outcomes]
            aggregates.append(
                RewardOptimizationAggregate(
                    budget=budget,
                    reward_kind=reward_kind,
                    scene_count=_coerce_int(item.get("scene_count")),
                    success_rate=_coerce_float(item.get("success_rate")),
                    success_rate_std=_stddev(success_values),
                    mean_reward=_coerce_float(item.get("mean_reward")),
                    mean_reward_std=_stddev(reward_values),
                    seed_count=_coerce_int(item.get("seed_count")),
                )
            )
    elif outcomes:
        grouped: dict[tuple[int, str], list[RewardOptimizationOutcome]] = {}
        for item in outcomes:
            grouped.setdefault((item.budget, item.reward_kind), []).append(item)
        for (budget, reward_kind), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
            reward_values = [float(row.best_reward) for row in rows]
            success_values = [1.0 if row.success else 0.0 for row in rows]
            aggregates.append(
                RewardOptimizationAggregate(
                    budget=budget,
                    reward_kind=reward_kind,
                    scene_count=len({row.scene_id for row in rows}),
                    success_rate=(sum(success_values) / len(success_values)) if success_values else 0.0,
                    success_rate_std=_stddev(success_values),
                    mean_reward=(sum(reward_values) / len(reward_values)) if reward_values else 0.0,
                    mean_reward_std=_stddev(reward_values),
                    seed_count=len({row.seed for row in rows}),
                )
            )
    return RewardOptimizationDetail(
        run_id=str(payload.get("run_id") or run_root.name),
        root=run_root,
        summary_path=summary_path,
        goal_text=str(payload.get("goal_text") or ""),
        goal_lines=_goal_lines(payload),
        dsl_program=str(payload.get("dsl_program") or ""),
        source_run_dir=str(payload.get("source_run_dir") or ""),
        source_run_kind=str(payload.get("source_run_kind") or ""),
        source_run_id=str(payload.get("source_run_id") or ""),
        model_name=str(payload.get("model_name") or ""),
        scene_count=_coerce_int(payload.get("scene_count")),
        seed_count=_coerce_int(payload.get("seed_count")),
        budgets=[_coerce_int(item) for item in payload.get("budgets", []) if _coerce_int(item) > 0],
        rack_jitter=_coerce_float_or_none(payload.get("rack_jitter")),
        synthesized=bool(payload.get("synthesized")),
        program_text=_read_text(run_root / "program.dsl", fallback=str(payload.get("dsl_program") or "")),
        synthesis_text=_read_text(_resolve_run_path(run_root, payload.get("synthesis_response_path")), fallback=""),
        aggregates=aggregates,
        outcomes=outcomes,
    )


def _load_value_model_detail(*, run_root: Path, summary_path: Path, payload: dict[str, Any]) -> RewardValueModelDetail:
    validation_outcomes: list[RewardValueModelOutcome] = []
    for item in payload.get("validation_outcomes", []):
        if not isinstance(item, dict):
            continue
        action_raw = item.get("action")
        validation_outcomes.append(
            RewardValueModelOutcome(
                epoch=_coerce_int(item.get("epoch")),
                scene_id=str(item.get("scene_id") or ""),
                reward=_coerce_float(item.get("reward")),
                success=bool(item.get("success")),
                action=action_raw if isinstance(action_raw, dict) else {},
                trace_path=_resolve_run_path(run_root, item.get("trace_path")),
                reward_path=_resolve_run_path(run_root, item.get("trace_reward_path")),
            )
        )
    reward_std_by_epoch: dict[int, float] = {}
    success_std_by_epoch: dict[int, float] = {}
    reward_mean_by_epoch: dict[int, float] = {}
    success_mean_by_epoch: dict[int, float] = {}
    grouped_rewards: dict[int, list[float]] = {}
    grouped_success: dict[int, list[float]] = {}
    for item in validation_outcomes:
        grouped_rewards.setdefault(item.epoch, []).append(float(item.reward))
        grouped_success.setdefault(item.epoch, []).append(1.0 if item.success else 0.0)
    for epoch, values in grouped_rewards.items():
        reward_mean_by_epoch[epoch] = (sum(values) / len(values)) if values else 0.0
        reward_std_by_epoch[epoch] = _stddev(values)
    for epoch, values in grouped_success.items():
        success_mean_by_epoch[epoch] = (sum(values) / len(values)) if values else 0.0
        success_std_by_epoch[epoch] = _stddev(values)

    eval_curve: list[RewardValueModelPoint] = []
    raw_curve = payload.get("eval_curve", [])
    if isinstance(raw_curve, list) and raw_curve:
        for item in raw_curve:
            if not isinstance(item, dict):
                continue
            epoch = _coerce_int(item.get("epoch"))
            eval_curve.append(
                RewardValueModelPoint(
                    epoch=epoch,
                    train_loss=_coerce_float(item.get("train_loss")),
                    validation_mean_reward=_coerce_float(item.get("validation_mean_reward")),
                    validation_mean_reward_std=reward_std_by_epoch.get(epoch, 0.0),
                    validation_success_rate=_coerce_float(item.get("validation_success_rate")),
                    validation_success_rate_std=success_std_by_epoch.get(epoch, 0.0),
                )
            )
    elif validation_outcomes:
        for epoch in sorted(reward_mean_by_epoch):
            eval_curve.append(
                RewardValueModelPoint(
                    epoch=epoch,
                    train_loss=0.0,
                    validation_mean_reward=reward_mean_by_epoch.get(epoch, 0.0),
                    validation_mean_reward_std=reward_std_by_epoch.get(epoch, 0.0),
                    validation_success_rate=success_mean_by_epoch.get(epoch, 0.0),
                    validation_success_rate_std=success_std_by_epoch.get(epoch, 0.0),
                )
            )
    return RewardValueModelDetail(
        run_id=str(payload.get("run_id") or run_root.name),
        root=run_root,
        summary_path=summary_path,
        goal_text=str(payload.get("goal_text") or ""),
        goal_lines=_goal_lines(payload),
        dsl_program=str(payload.get("dsl_program") or ""),
        source_run_dir=str(payload.get("source_run_dir") or ""),
        source_run_kind=str(payload.get("source_run_kind") or ""),
        source_run_id=str(payload.get("source_run_id") or ""),
        model_name=str(payload.get("model_name") or ""),
        train_scene_count=_coerce_int(payload.get("train_scene_count")),
        train_actions_per_scene=_coerce_int(payload.get("train_actions_per_scene")),
        validation_scene_count=_coerce_int(payload.get("validation_scene_count")),
        validation_candidates_per_scene=_coerce_int(payload.get("validation_candidates_per_scene")),
        epochs=_coerce_int(payload.get("epochs")),
        batch_size=_coerce_int(payload.get("batch_size")),
        learning_rate=_coerce_float_or_none(payload.get("learning_rate")),
        image_size=_coerce_int_or_none(payload.get("image_size")),
        rack_jitter=_coerce_float_or_none(payload.get("rack_jitter")),
        train_sample_count=_coerce_int(payload.get("train_sample_count")),
        best_epoch=_coerce_int(payload.get("best_epoch")),
        best_validation_mean_reward=_coerce_float(payload.get("best_validation_mean_reward")),
        best_validation_success_rate=_coerce_float(payload.get("best_validation_success_rate")),
        synthesized=bool(payload.get("synthesized")),
        checkpoint_path=_resolve_run_path(run_root, payload.get("checkpoint_path")),
        program_text=_read_text(run_root / "program.dsl", fallback=str(payload.get("dsl_program") or "")),
        synthesis_text=_read_text(_resolve_run_path(run_root, payload.get("synthesis_response_path")), fallback=""),
        eval_curve=eval_curve,
        validation_outcomes=validation_outcomes,
    )


def _goal_lines(payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key in ("goal_texts", "goals"):
        raw = payload.get(key)
        if isinstance(raw, list):
            for item in raw:
                text = _clean_goal_line(item)
                if text and text not in lines:
                    lines.append(text)
    if lines:
        return lines
    goal_text = str(payload.get("goal_text") or "").strip()
    if not goal_text:
        return []
    split_lines = [_clean_goal_line(part) for part in goal_text.splitlines()]
    normalized = [item for item in split_lines if item]
    return normalized or [goal_text]


def _clean_goal_line(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.lstrip("-* ").strip()
    return text


def _normalize_project_path(raw: str, *, project_root: Path) -> Path | None:
    value = str(raw or "").strip()
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate.resolve()
        stack_prefix = "/opt/pattern-learning-stack/"
        if value.startswith(stack_prefix):
            mapped = (project_root / value[len(stack_prefix):]).resolve()
            if mapped.exists():
                return mapped
        return candidate
    return (project_root / candidate).resolve()


def _resolve_run_path(run_root: Path, raw: object) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = Path(raw)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate.resolve()
        stack_prefix = "/opt/pattern-learning-stack/"
        value = str(raw)
        if value.startswith(stack_prefix):
            mapped = (run_root.parents[5] / value[len(stack_prefix):]).resolve()
            return mapped if mapped.exists() else mapped
        return candidate
    return (run_root / candidate).resolve()


def _read_text(path: Path | None, *, fallback: str) -> str:
    if path is None or not path.is_file():
        return fallback
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return fallback


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _stddev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    try:
        return float(statistics.pstdev(values))
    except statistics.StatisticsError:
        return 0.0


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _coerce_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
