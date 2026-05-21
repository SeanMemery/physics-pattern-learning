from __future__ import annotations

import asyncio
from dataclasses import replace
import difflib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable
from urllib.parse import parse_qs, urlencode
from zoneinfo import ZoneInfo
import re

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from pydantic import ValidationError

from deepphy_eval import (
    default_benchmark_task_limit,
    default_run_id,
    summarize_deepphy_benchmark,
)
from datasets import load_dataset_build_config
from dashboard.actions import (
    AUTOMATIC_LEARNING_ACTION,
    DEFAULT_LIBRARY_HOLDOUT_EVALUATION_TEMPLATE,
    LIBRARY_HOLDOUT_EVALUATION_TEMPLATES,
    DatasetBuildActionRequest,
    DeepPHYEvaluationActionRequest,
    LibraryHoldoutEvaluationActionRequest,
    PatternLearningActionRequest,
    PooltoolRewardOptimizationActionRequest,
    PooltoolRewardValueModelActionRequest,
    AutomaticLearningActionRequest,
    UnsupervisedLearningActionRequest,
    UNSUPERVISED_LEARNING_ACTION,
    build_automatic_launch_argv,
    build_dataset_build_argv,
    build_deepphy_eval_argv,
    build_pattern_learning_argv,
    build_pooltool_reward_optimization_argv,
    build_pooltool_reward_value_model_argv,
    build_supervised_learning_with_auto_eval_argv,
    build_unsupervised_launch_argv,
    default_pattern_learning_output_path,
    pattern_run_dir_for_output_path,
    python_executable_for_env,
    resolve_deepphy_paths,
)
from dashboard.data import (
    LIBRARY_EVALUATIONS_DIRNAME,
    SUPPORTED_ENVS,
    load_candidate_code,
    load_dataset_detail,
    load_dataset_sample,
    load_dataset_summary,
    load_dashboard_home,
    load_evaluation_run_summary,
    load_evaluation_task_detail,
    load_environment_summary,
    load_library_evaluation_detail,
    load_automatic_library_detail,
    load_pattern_run_detail,
    load_pattern_run_detail_from_root,
    list_pooltool_reward_optimization_experiments,
    list_pooltool_reward_value_model_experiments,
    load_target_label_config,
    load_unsupervised_library_detail,
    list_library_evaluations_for_run,
)
from dashboard.jobs import QUEUEABLE_JOB_ACTIONS, DashboardJobManager, DashboardJobSnapshot
from dashboard.reward_local import (
    LocalRewardRunRecord,
    list_local_reward_runs,
    load_local_reward_run,
    launch_local_reward_run,
)
from llm_config import load_shared_llm_config
from paths import ArtifactLayout, get_artifact_layout
from pattern_learning.patterns.config import load_pattern_learning_config
from pattern_learning.patterns.learning import promote_live_best_candidate_to_checkpoint
from pattern_library_registry import load_pattern_library, save_pattern_library
from pattern_library_paths import (
    AUTOMATIC_RUNS_DIRNAME,
    UNSUPERVISED_RUNS_DIRNAME,
    default_automatic_run_dir,
    default_unsupervised_run_dir,
)

DISPLAY_TIMEZONE = ZoneInfo("Europe/London")
FUNSEARCH_TEMPLATES: dict[str, dict[str, str]] = {
    "full": {
        "label": "Full",
        "iteration_count": "1",
        "population_size": "50",
        "elite_fraction": "0.25",
        "children_per_elite": "3",
    },
    "light": {
        "label": "Light",
        "iteration_count": "1",
        "population_size": "12",
        "elite_fraction": "0.25",
        "children_per_elite": "2",
    },
}
DEFAULT_FUNSEARCH_TEMPLATE = "full"
AUTO_SOURCE_EVALUATION_RUN_ID = "__auto_source_eval__"
AUTOMATIC_ROUTE_SEGMENT = "automatic"
UNSUPERVISED_ROUTE_SEGMENT = "unsupervised"
AUTOMATIC_WORKFLOW_KIND = "automatic"
UNSUPERVISED_WORKFLOW_KIND = "unsupervised"
AUTOMATIC_LAUNCH_ACTION = AUTOMATIC_LEARNING_ACTION
UNSUPERVISED_LAUNCH_ACTION = UNSUPERVISED_LEARNING_ACTION
POOLTOOL_REWARD_OPTIMIZATION_ACTION = "pooltool-reward-optimize"
POOLTOOL_REWARD_VALUE_MODEL_ACTION = "pooltool-reward-value-model"
_TTL_CACHE_LOCK = Lock()
_TTL_CACHE: dict[tuple[Any, ...], tuple[float, Any]] = {}
_TTL_CACHE_MAX_ENTRIES = 512
_CPU_USAGE_LOCK = Lock()
_CPU_USAGE_PREVIOUS_SAMPLE: tuple[int, int] | None = None


def _is_automatic_action(action: Any) -> bool:
    return str(action or "") == AUTOMATIC_LAUNCH_ACTION


def _is_unsupervised_action(action: Any) -> bool:
    return str(action or "") == UNSUPERVISED_LAUNCH_ACTION


def _is_automatic_kind(kind: Any) -> bool:
    return str(kind or "") == AUTOMATIC_WORKFLOW_KIND


def _is_unsupervised_kind(kind: Any) -> bool:
    return str(kind or "") == UNSUPERVISED_WORKFLOW_KIND


def _automatic_run_url(env_id: str, run_id: str) -> str:
    return f"/{AUTOMATIC_ROUTE_SEGMENT}/{env_id}/{run_id}"


def _unsupervised_run_url(env_id: str, run_id: str) -> str:
    return f"/{UNSUPERVISED_ROUTE_SEGMENT}/{env_id}/{run_id}"


def _run_url_for_kind(env_id: str, run_id: str, kind: Any) -> str:
    if _is_automatic_kind(kind):
        return _automatic_run_url(env_id, run_id)
    if _is_unsupervised_kind(kind):
        return _unsupervised_run_url(env_id, run_id)
    return f"/runs/{env_id}/{run_id}"


def _display_run_kind(kind: Any) -> str:
    return AUTOMATIC_WORKFLOW_KIND if _is_automatic_kind(kind) else str(kind or "")


def _display_job_action(action: Any, fallback: str = "") -> str:
    if _is_automatic_action(action):
        return AUTOMATIC_LAUNCH_ACTION
    if _is_unsupervised_action(action):
        return UNSUPERVISED_LAUNCH_ACTION
    return str(action or fallback)


def _coerce_optional(value: Any) -> Any | None:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    return stripped if stripped else None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _cached_value(
    key: tuple[Any, ...],
    *,
    ttl_seconds: float,
    builder: Callable[[], Any],
) -> Any:
    now = time.monotonic()
    with _TTL_CACHE_LOCK:
        cached = _TTL_CACHE.get(key)
        if cached is not None and (now - cached[0]) <= ttl_seconds:
            return cached[1]
    value = builder()
    with _TTL_CACHE_LOCK:
        _TTL_CACHE[key] = (now, value)
        if len(_TTL_CACHE) > _TTL_CACHE_MAX_ENTRIES:
            expired_keys = sorted(_TTL_CACHE.items(), key=lambda item: item[1][0])[
                : len(_TTL_CACHE) - _TTL_CACHE_MAX_ENTRIES
            ]
            for stale_key, _stale_value in expired_keys:
                _TTL_CACHE.pop(stale_key, None)
    return value


def _file_signature(path: Path) -> str:
    try:
        stat_result = path.stat()
    except OSError:
        return "missing"
    return f"{stat_result.st_mtime_ns}:{stat_result.st_size}"


def _resolve_funsearch_template_name(value: Any) -> str:
    if isinstance(value, str) and value in FUNSEARCH_TEMPLATES:
        return value
    return DEFAULT_FUNSEARCH_TEMPLATE


def _funsearch_template_values(template_name: str) -> dict[str, str]:
    resolved_name = _resolve_funsearch_template_name(template_name)
    return dict(FUNSEARCH_TEMPLATES[resolved_name])


def _queue_settings_state_path(layout: ArtifactLayout) -> Path:
    return layout.data_root / "dashboard" / "queue_settings.json"


def load_queue_settings_state(layout: ArtifactLayout) -> dict[str, Any]:
    path = _queue_settings_state_path(layout)
    default = {
        "max_concurrent_jobs": 1,
        "enforce_single_pooltool_job": True,
    }
    if not path.is_file():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    if not isinstance(payload, dict):
        return default
    raw_value = payload.get("max_concurrent_jobs", 1)
    try:
        normalized_value = max(1, int(raw_value))
    except (TypeError, ValueError):
        normalized_value = 1
    raw_pooltool_value = payload.get("enforce_single_pooltool_job", True)
    normalized_pooltool_value = (
        raw_pooltool_value
        if isinstance(raw_pooltool_value, bool)
        else str(raw_pooltool_value).strip().lower() in {"1", "true", "yes", "on"}
    )
    return {
        "max_concurrent_jobs": normalized_value,
        "enforce_single_pooltool_job": normalized_pooltool_value,
    }


def save_queue_settings_state(
    layout: ArtifactLayout,
    state: dict[str, Any],
) -> dict[str, Any]:
    try:
        normalized_value = max(1, int(state.get("max_concurrent_jobs", 1)))
    except (TypeError, ValueError):
        normalized_value = 1
    raw_pooltool_value = state.get("enforce_single_pooltool_job", True)
    normalized_pooltool_value = (
        raw_pooltool_value
        if isinstance(raw_pooltool_value, bool)
        else str(raw_pooltool_value).strip().lower() in {"1", "true", "yes", "on"}
    )
    normalized = {
        "max_concurrent_jobs": normalized_value,
        "enforce_single_pooltool_job": normalized_pooltool_value,
    }
    path = _queue_settings_state_path(layout)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return normalized




def _snapshot_to_payload(snapshot: DashboardJobSnapshot) -> dict[str, Any]:
    return {
        "job_id": snapshot.job_id,
        "name": snapshot.name,
        "status": snapshot.status,
        "argv": snapshot.argv,
        "metadata": snapshot.metadata,
        "created_at": snapshot.created_at,
        "created_at_display": _format_display_time(snapshot.created_at),
        "started_at": snapshot.started_at,
        "started_at_display": _format_display_time(snapshot.started_at),
        "finished_at": snapshot.finished_at,
        "finished_at_display": _format_display_time(snapshot.finished_at),
        "exit_code": snapshot.exit_code,
        "log_path": str(snapshot.log_path),
        "stdout_path": str(snapshot.stdout_path),
        "stderr_path": str(snapshot.stderr_path),
        "snapshot_path": str(snapshot.snapshot_path),
        "recent_output": snapshot.recent_output,
    }


def _queue_order_value(snapshot: DashboardJobSnapshot) -> int:
    metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
    raw_value = metadata.get("queue_order")
    if isinstance(raw_value, bool):
        return 1_000_000_000
    if isinstance(raw_value, int) and raw_value > 0:
        return raw_value
    if isinstance(raw_value, str):
        try:
            parsed = int(raw_value)
        except ValueError:
            return 1_000_000_000
        if parsed > 0:
            return parsed
    return 1_000_000_000


def _queue_dispatch_priority(snapshot: DashboardJobSnapshot) -> int:
    metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
    action = str(metadata.get("action") or "")
    return _queue_dispatch_priority_for_action(action)


def _queue_dispatch_priority_for_action(action: str) -> int:
    if action in {"supervised-learn-and-evaluate", "evaluate-deepphy"}:
        return 0
    if action in {
        "evaluate-library-case",
        POOLTOOL_REWARD_OPTIMIZATION_ACTION,
        POOLTOOL_REWARD_VALUE_MODEL_ACTION,
    }:
        return 1
    if _is_automatic_action(action):
        return 1
    return 2


def _queue_sort_key(snapshot: DashboardJobSnapshot) -> tuple[int, int, str, str]:
    return (
        _queue_dispatch_priority(snapshot),
        _queue_order_value(snapshot),
        snapshot.created_at,
        snapshot.job_id,
    )


def _job_can_requeue(snapshot: DashboardJobSnapshot) -> bool:
    metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
    supports_resume = metadata.get("supports_pause", False) or (
        isinstance(metadata.get("resume_argv"), list)
        and bool(metadata.get("resume_argv"))
    )
    return bool(
        metadata.get("queue_requested")
        and metadata.get("action") in QUEUEABLE_JOB_ACTIONS
        and supports_resume
        and snapshot.status in {"running", "paused"}
    )


def _job_can_resume(snapshot: DashboardJobSnapshot) -> bool:
    metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
    supports_resume = metadata.get("supports_pause", False) or (
        isinstance(metadata.get("resume_argv"), list)
        and bool(metadata.get("resume_argv"))
    )
    return bool(supports_resume and snapshot.status == "paused")


def _launch_mode_from_parsed_form(parsed_form: dict[str, list[str]]) -> str:
    values = parsed_form.get("launch_mode")
    if not values:
        return "run_now"
    value = values[0].strip().lower()
    return "queue" if value == "queue" else "run_now"


def _queued_home_jobs(
    snapshots: list[DashboardJobSnapshot],
    *,
    layout: ArtifactLayout,
) -> list[dict[str, Any]]:
    def target_identity(snapshot: DashboardJobSnapshot) -> tuple[str, str] | None:
        metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
        if metadata.get("action") not in QUEUEABLE_JOB_ACTIONS:
            return None
        for key in ("evaluation_run_dir", "run_dir"):
            raw_value = metadata.get(key)
            if not isinstance(raw_value, str) or not raw_value.strip():
                continue
            return (key, str(Path(raw_value).resolve()))
        return None

    succeeded_created_at_by_target: dict[tuple[str, str], str] = {}
    for snapshot in snapshots:
        if snapshot.status != "succeeded":
            continue
        identity = target_identity(snapshot)
        if identity is None:
            continue
        created_at = snapshot.created_at or ""
        if created_at > succeeded_created_at_by_target.get(identity, ""):
            succeeded_created_at_by_target[identity] = created_at

    active_entries: list[dict[str, Any]] = []
    pending_entries: list[dict[str, Any]] = []
    for snapshot in snapshots:
        metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
        if metadata.get("action") not in QUEUEABLE_JOB_ACTIONS:
            continue
        identity = target_identity(snapshot)
        if (
            identity is not None
            and snapshot.status in {"queued", "running", "paused"}
            and succeeded_created_at_by_target.get(identity, "") > (snapshot.created_at or "")
        ):
            continue
        is_active_entry = snapshot.status == "running"
        is_paused_entry = snapshot.status == "paused"
        is_pending_entry = snapshot.status == "queued" and bool(metadata.get("queue_requested"))
        if not is_active_entry and not is_paused_entry and not is_pending_entry:
            continue
        learning_target = _learning_run_target(snapshot)
        evaluation_target = _evaluation_job_target(snapshot)
        reward_target = _pooltool_reward_job_target(snapshot)
        target_url = f"/jobs/{snapshot.job_id}"
        target_label = snapshot.name
        if learning_target is not None:
            target_url = learning_target["run_url"]
            target_label = learning_target["run_id"]
        elif evaluation_target is not None:
            target_url = evaluation_target["evaluation_url"]
            target_label = evaluation_target["run_id"]
        elif reward_target is not None:
            target_url = reward_target["url"]
            target_label = reward_target["run_id"]
        payload = {
            **_snapshot_to_payload(snapshot),
            "env_id": metadata.get("env_id"),
            "action_label": _display_job_action(metadata.get("action"), snapshot.name),
            "target_url": target_url,
            "target_label": target_label,
            "can_resume": _job_can_resume(snapshot),
            "can_requeue": _job_can_requeue(snapshot),
            "queue_order": _queue_order_value(snapshot) if is_pending_entry else None,
        }
        if is_active_entry or is_paused_entry:
            learning_progress = _learning_progress_payload(snapshot, layout=layout)
            if learning_progress is not None:
                payload["learning_progress"] = learning_progress
            evaluation_progress = _evaluation_progress_payload(snapshot, layout=layout)
            if evaluation_progress is not None:
                payload["evaluation_progress"] = evaluation_progress
        if is_active_entry:
            active_entries.append(payload)
        elif is_paused_entry:
            pending_entries.append(payload)
        else:
            pending_entries.append(payload)
    active_entries.sort(
        key=lambda item: (
            item.get("started_at") or "",
            item.get("created_at") or "",
            item.get("job_id") or "",
        )
    )
    pending_entries.sort(
        key=lambda item: (
            _queue_dispatch_priority_for_action(str(item.get("action_label") or "")),
            item.get("queue_order") or 1_000_000_000,
            item.get("created_at") or "",
            item.get("job_id") or "",
        )
    )
    ordered = [*active_entries, *pending_entries]
    for index, item in enumerate(ordered):
        item["queue_index"] = index
        item["is_queue_head"] = index == 0
    return ordered


def _queue_panel_payload(
    *,
    job_manager: DashboardJobManager,
    snapshots: list[DashboardJobSnapshot],
    layout: ArtifactLayout,
) -> dict[str, Any]:
    snapshots_signature = tuple(
        (
            snapshot.job_id,
            snapshot.status,
            snapshot.started_at,
            snapshot.finished_at,
            snapshot.exit_code,
            _file_signature(snapshot.log_path),
        )
        for snapshot in snapshots
        if isinstance(snapshot.metadata, dict)
        and snapshot.metadata.get("action") in QUEUEABLE_JOB_ACTIONS
        and (
            snapshot.status in {"running", "paused"}
            or (
                snapshot.status == "queued"
                and bool(snapshot.metadata.get("queue_requested"))
            )
        )
    )
    cache_key = (
        "queue-panel-payload",
        job_manager.max_concurrent_queue_jobs,
        job_manager.enforce_single_pooltool_queue_job,
        snapshots_signature,
    )

    def build() -> dict[str, Any]:
        queued_home_jobs = _queued_home_jobs(snapshots, layout=layout)
        active_count = sum(1 for item in queued_home_jobs if item.get("status") == "running")
        queued_count = sum(1 for item in queued_home_jobs if item.get("status") == "queued")
        return {
            "queued_home_jobs": queued_home_jobs,
            "queue_settings": {
                "max_concurrent_jobs": job_manager.max_concurrent_queue_jobs,
                "enforce_single_pooltool_job": job_manager.enforce_single_pooltool_queue_job,
                "active_count": active_count,
                "queued_count": queued_count,
            },
        }

    return _cached_value(cache_key, ttl_seconds=2.0, builder=build)


def _queue_panel_version(
    *,
    job_manager: DashboardJobManager,
    snapshots: list[DashboardJobSnapshot],
) -> str:
    parts: list[str] = [
        f"max:{job_manager.max_concurrent_queue_jobs}",
        f"pooltool:{int(job_manager.enforce_single_pooltool_queue_job)}",
    ]
    for snapshot in snapshots:
        if not isinstance(snapshot.metadata, dict):
            continue
        if snapshot.metadata.get("action") not in QUEUEABLE_JOB_ACTIONS:
            continue
        if snapshot.status not in {"queued", "running", "paused"}:
            continue
        if snapshot.status == "queued" and not snapshot.metadata.get("queue_requested"):
            continue
        parts.append(
            "|".join(
                [
                    snapshot.job_id,
                    snapshot.status,
                    snapshot.started_at or "",
                    snapshot.finished_at or "",
                    str(snapshot.exit_code),
                    str(_file_signature(snapshot.log_path)),
                ]
            )
        )
    return str(hash(tuple(parts)))




def _queue_panel_json_payload(queue_panel: dict[str, Any]) -> dict[str, Any]:
    jobs_payload: list[dict[str, Any]] = []
    for job in queue_panel.get("queued_home_jobs", []):
        progress_summary = None
        if isinstance(job.get("learning_progress"), dict):
            progress_summary = job["learning_progress"].get("progress_summary")
        evaluation_progress = job.get("evaluation_progress") if isinstance(job.get("evaluation_progress"), dict) else None
        jobs_payload.append(
            {
                "job_id": job.get("job_id"),
                "status": job.get("status"),
                "target_url": job.get("target_url"),
                "target_label": job.get("target_label"),
                "env_id": job.get("env_id"),
                "action_label": job.get("action_label"),
                "started_at_display": job.get("started_at_display"),
                "created_at_display": job.get("created_at_display"),
                "can_requeue": bool(job.get("can_requeue")),
                "is_queue_head": bool(job.get("is_queue_head")),
                "queue_index": job.get("queue_index"),
                "learning_progress": progress_summary,
                "evaluation_progress": evaluation_progress,
            }
        )
    return {
        "queued_home_jobs": jobs_payload,
        "queue_settings": dict(queue_panel.get("queue_settings") or {}),
    }


def _recent_runs_version(
    *,
    summaries: list[Any],
    job_snapshots: list[DashboardJobSnapshot],
    layout: ArtifactLayout,
) -> str:
    summary_parts = []
    for summary in summaries:
        env_id = getattr(summary, "env_id", "")
        for collection_name in (
            "datasets",
            "supervised_libraries",
            "unsupervised_libraries",
            "evaluations",
        ):
            collection = getattr(summary, collection_name, []) or []
            for item in collection:
                summary_parts.append(
                    "|".join(
                        [
                            str(env_id),
                            collection_name,
                            str(getattr(item, "run_id", "")),
                            str(getattr(item, "updated_at", "")),
                            str(getattr(item, "status", "")),
                        ]
                    )
                )
    job_parts = [
        "|".join(
            [
                snapshot.job_id,
                snapshot.name,
                snapshot.status,
                snapshot.started_at or "",
                snapshot.finished_at or "",
                str(snapshot.exit_code),
            ]
        )
        for snapshot in job_snapshots
    ]
    parts = [*summary_parts, *job_parts, str(layout.data_root)]
    return str(hash(tuple(parts)))


def _evaluation_scores_by_model(
    evaluation_runs: list[Any] | None,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for evaluation in evaluation_runs or []:
        model_name = getattr(evaluation, "model_name", None)
        normalized_model = model_name if isinstance(model_name, str) else None
        display_model = (
            normalized_model
            if normalized_model is not None and normalized_model != ""
            else "(router default)"
        )
        row = grouped.get(display_model)
        success_ratio = getattr(evaluation, "success_ratio", None)
        created_at = getattr(evaluation, "created_at", None) or ""
        if row is None:
            row = {
                "model_name": normalized_model if normalized_model is not None else "",
                "display_model": display_model,
                "evaluation_count": 0,
                "latest_run_id": getattr(evaluation, "run_id", None),
                "latest_created_at": created_at,
                "latest_status": getattr(evaluation, "status", None),
                "latest_success_ratio": success_ratio,
                "latest_success_count": getattr(evaluation, "success_count", None),
                "latest_task_count": getattr(evaluation, "task_count", None),
                "best_run_id": getattr(evaluation, "run_id", None),
                "best_success_ratio": success_ratio,
                "best_success_count": getattr(evaluation, "success_count", None),
                "best_task_count": getattr(evaluation, "task_count", None),
            }
            grouped[display_model] = row
        row["evaluation_count"] += 1
        if created_at >= str(row.get("latest_created_at") or ""):
            row["latest_run_id"] = getattr(evaluation, "run_id", None)
            row["latest_created_at"] = created_at
            row["latest_status"] = getattr(evaluation, "status", None)
            row["latest_success_ratio"] = success_ratio
            row["latest_success_count"] = getattr(evaluation, "success_count", None)
            row["latest_task_count"] = getattr(evaluation, "task_count", None)
        best_ratio = row.get("best_success_ratio")
        if isinstance(success_ratio, (int, float)) and (
            not isinstance(best_ratio, (int, float)) or success_ratio > best_ratio
        ):
            row["best_run_id"] = getattr(evaluation, "run_id", None)
            row["best_success_ratio"] = success_ratio
            row["best_success_count"] = getattr(evaluation, "success_count", None)
            row["best_task_count"] = getattr(evaluation, "task_count", None)
    return sorted(
        grouped.values(),
        key=lambda item: (
            -1 if isinstance(item.get("best_success_ratio"), (int, float)) else 0,
            -float(item.get("best_success_ratio") or 0.0),
            -(int(item.get("evaluation_count") or 0)),
            str(item.get("display_model") or ""),
        ),
    )


def _read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payloads: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _format_kib(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    kib = float(value)
    gib = kib / (1024.0 * 1024.0)
    if gib >= 1.0:
        return f"{gib:.2f} GiB"
    mib = kib / 1024.0
    if mib >= 1.0:
        return f"{mib:.0f} MiB"
    return f"{kib:.0f} KiB"


def _format_mib(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    mib = float(value)
    gib = mib / 1024.0
    if gib >= 1.0:
        return f"{gib:.2f} GiB"
    return f"{mib:.0f} MiB"


def _format_temp_c(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{float(value):.1f} C"


def _format_pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{float(value):.1f}%"


def _max_numeric_mapping_value(payload: Any) -> int | float | None:
    if not isinstance(payload, dict):
        return None
    values = [value for value in payload.values() if isinstance(value, (int, float))]
    if not values:
        return None
    return max(values)


def _summarize_job_resource_metrics(snapshot: DashboardJobSnapshot) -> dict[str, Any] | None:
    job_root = snapshot.snapshot_path.parent
    metrics_path = job_root / "system_metrics.jsonl"
    summary_path = job_root / "system_metrics_summary.json"
    summary_payload = _read_json_file(summary_path) or {}
    samples = _read_jsonl_objects(metrics_path)
    latest_sample = samples[-1] if samples else None
    if not summary_payload and latest_sample is None:
        return None

    latest_process = latest_sample.get("process") if isinstance(latest_sample, dict) else None
    latest_memory = (
        latest_sample.get("system_memory") if isinstance(latest_sample, dict) else None
    )
    latest_gpus = latest_sample.get("gpus") if isinstance(latest_sample, dict) else None
    latest_temperatures = (
        latest_sample.get("temperatures") if isinstance(latest_sample, dict) else None
    )
    latest_gpu_vram = None
    latest_gpu_temp = None
    if isinstance(latest_gpus, list):
        gpu_vram_values = [
            item.get("memory_used_mb")
            for item in latest_gpus
            if isinstance(item, dict) and isinstance(item.get("memory_used_mb"), (int, float))
        ]
        gpu_temp_values = [
            item.get("temperature_c")
            for item in latest_gpus
            if isinstance(item, dict) and isinstance(item.get("temperature_c"), (int, float))
        ]
        latest_gpu_vram = max(gpu_vram_values) if gpu_vram_values else None
        latest_gpu_temp = max(gpu_temp_values) if gpu_temp_values else None
    latest_cpu_temp = None
    if isinstance(latest_temperatures, list):
        temp_values = [
            item.get("temp_c")
            for item in latest_temperatures
            if isinstance(item, dict) and isinstance(item.get("temp_c"), (int, float))
        ]
        latest_cpu_temp = max(temp_values) if temp_values else None

    peak_gpu_vram = _max_numeric_mapping_value(summary_payload.get("max_gpu_memory_used_mb"))
    peak_gpu_temp = _max_numeric_mapping_value(summary_payload.get("max_gpu_temperature_c"))
    peak_cpu_temp = _max_numeric_mapping_value(summary_payload.get("max_temperature_c_by_label"))

    return {
        "available": True,
        "sample_count": summary_payload.get("sample_count") if summary_payload else len(samples),
        "updated_at": latest_sample.get("timestamp") if isinstance(latest_sample, dict) else None,
        "updated_at_display": _format_display_time(
            latest_sample.get("timestamp") if isinstance(latest_sample, dict) else None
        ),
        "current_rss_display": _format_kib(
            latest_process.get("rss_kb") if isinstance(latest_process, dict) else None
        ),
        "peak_rss_display": _format_kib(summary_payload.get("max_process_rss_kb")),
        "peak_hwm_display": _format_kib(summary_payload.get("max_process_hwm_kb")),
        "current_mem_available_display": _format_kib(
            latest_memory.get("mem_available_kb") if isinstance(latest_memory, dict) else None
        ),
        "min_mem_available_display": _format_kib(
            summary_payload.get("min_system_mem_available_kb")
        ),
        "current_gpu_vram_display": _format_mib(latest_gpu_vram),
        "peak_gpu_vram_display": _format_mib(peak_gpu_vram),
        "current_gpu_temp_display": _format_temp_c(latest_gpu_temp),
        "peak_gpu_temp_display": _format_temp_c(peak_gpu_temp),
        "current_cpu_temp_display": _format_temp_c(latest_cpu_temp),
        "peak_cpu_temp_display": _format_temp_c(peak_cpu_temp),
        "summary_path": str(summary_path) if summary_path.is_file() else None,
        "metrics_path": str(metrics_path) if metrics_path.is_file() else None,
    }


def _summarize_job_failure_snapshot(snapshot: DashboardJobSnapshot) -> dict[str, Any] | None:
    failure_snapshot_path = snapshot.snapshot_path.parent / "failed_resource_snapshot.json"
    payload = _read_json_file(failure_snapshot_path)
    if not isinstance(payload, dict):
        return None
    process = payload.get("process")
    system_memory = payload.get("system_memory")
    gpus = payload.get("gpus")
    gpu_vram = None
    if isinstance(gpus, list):
        gpu_vram_values = [
            item.get("memory_used_mb")
            for item in gpus
            if isinstance(item, dict) and isinstance(item.get("memory_used_mb"), (int, float))
        ]
        gpu_vram = max(gpu_vram_values) if gpu_vram_values else None
    mem_available_kb = (
        system_memory.get("mem_available_kb")
        if isinstance(system_memory, dict)
        else None
    )
    mem_total_kb = (
        system_memory.get("mem_total_kb")
        if isinstance(system_memory, dict)
        else None
    )
    mem_used_kb = None
    if isinstance(mem_total_kb, int) and isinstance(mem_available_kb, int):
        mem_used_kb = max(mem_total_kb - mem_available_kb, 0)
    return {
        "captured_at": payload.get("captured_at"),
        "captured_at_display": _format_display_time(payload.get("captured_at")),
        "process_rss_display": _format_kib(
            process.get("rss_kb") if isinstance(process, dict) else None
        ),
        "system_mem_used_display": _format_kib(mem_used_kb),
        "system_mem_available_display": _format_kib(mem_available_kb),
        "gpu_vram_display": _format_mib(gpu_vram),
        "path": str(failure_snapshot_path),
    }


def _dataset_progress_payload(
    snapshot: DashboardJobSnapshot, *, layout: ArtifactLayout
) -> dict[str, Any] | None:
    if snapshot.metadata.get("action") != "build-dataset":
        return None
    env_id = snapshot.metadata.get("env_id")
    dataset_name = snapshot.metadata.get("dataset_name")
    if not isinstance(env_id, str) or not isinstance(dataset_name, str):
        return None
    request_payload = snapshot.metadata.get("request")
    dataset_root: Path
    if isinstance(request_payload, dict) and isinstance(
        request_payload.get("output_dir"), str
    ):
        dataset_root = Path(request_payload["output_dir"])
    else:
        dataset_root = layout.data_root / "datasets" / env_id / dataset_name
    if not dataset_root.exists():
        return {
            "env_id": env_id,
            "dataset_name": dataset_name,
            "dataset_root": str(dataset_root),
            "available": False,
        }
    summary = load_dataset_summary(env_id, dataset_root)
    split_payloads = []
    total_completed_count = 0
    for split in summary.splits:
        completed_count = (
            split.manifest_completed_count
            if split.manifest_completed_count is not None
            else split.completed_count
        )
        total_completed_count += completed_count
        split_payloads.append(
            {
                "name": split.name,
                "completed_count": completed_count,
                "target_count": split.target_count,
                "manifest_completed_count": split.manifest_completed_count,
                "disk_completed_count": split.completed_count,
            }
        )
    target_total = sum(
        split.target_count or 0 for split in summary.splits if split.target_count is not None
    )
    return {
        "env_id": env_id,
        "dataset_name": summary.dataset_name,
        "dataset_root": str(summary.root),
        "manifest_path": str(summary.manifest_path) if summary.manifest_path else None,
        "available": True,
        "total_completed_count": total_completed_count,
        "target_total_count": target_total if target_total > 0 else None,
        "splits": split_payloads,
    }


def _job_payload(
    snapshot: DashboardJobSnapshot, *, layout: ArtifactLayout
) -> dict[str, Any]:
    learning_target = _learning_run_target(snapshot)
    run_signature = (
        _run_refresh_token(learning_target["run_root"])
        if learning_target is not None and learning_target["run_root"].exists()
        else "missing"
    )
    cache_key = (
        "job-payload",
        snapshot.job_id,
        snapshot.status,
        snapshot.exit_code,
        snapshot.started_at,
        snapshot.finished_at,
        len(snapshot.recent_output),
        snapshot.recent_output[-1] if snapshot.recent_output else "",
        _file_signature(snapshot.log_path),
        run_signature,
    )

    def build() -> dict[str, Any]:
        payload = _snapshot_to_payload(snapshot)
        payload["display_name"] = snapshot.name
        resource_metrics = _summarize_job_resource_metrics(snapshot)
        if resource_metrics is not None:
            payload["resource_metrics"] = resource_metrics
        failure_resource_snapshot = _summarize_job_failure_snapshot(snapshot)
        if failure_resource_snapshot is not None:
            payload["failure_resource_snapshot"] = failure_resource_snapshot
        dataset_progress = _dataset_progress_payload(snapshot, layout=layout)
        if dataset_progress is not None:
            payload["dataset_progress"] = dataset_progress
        learning_progress = _learning_progress_payload(snapshot, layout=layout)
        if learning_progress is not None:
            payload["learning_progress"] = learning_progress
            payload["display_name"] = learning_progress["run_id"]
        evaluation_progress = _evaluation_progress_payload(snapshot, layout=layout)
        if evaluation_progress is not None:
            payload["evaluation_progress"] = evaluation_progress
        return payload

    return _cached_value(cache_key, ttl_seconds=2.0, builder=build)


def _read_log_tail_lines(log_path: Path, *, limit: int = 1200) -> list[str]:
    cache_key = (
        "log-tail",
        str(log_path),
        limit,
        _file_signature(log_path),
    )

    def build() -> list[str]:
        try:
            text = log_path.read_text(encoding="utf-8")
        except OSError:
            return []
        lines = [line for line in text.splitlines() if line.strip()]
        if limit <= 0:
            return lines
        return lines[-limit:]

    return _cached_value(cache_key, ttl_seconds=2.0, builder=build)


def _run_refresh_token(run_root: Path) -> str:
    tracked_paths: list[Path] = []
    for candidate in (
        run_root / "run.json",
        run_root / "run_control.json",
        run_root / "evolution_record.jsonl",
        run_root / "evolution_live.jsonl",
        run_root / "llm_calls.jsonl",
        run_root / "pattern_library.json",
    ):
        if candidate.is_file():
            tracked_paths.append(candidate)
    tracked_paths.extend(
        path for path in sorted(run_root.glob("label_*")) if path.is_dir()
    )
    if not tracked_paths:
        return "missing"
    latest_mtime_ns = max(path.stat().st_mtime_ns for path in tracked_paths)
    return f"{len(tracked_paths)}:{latest_mtime_ns}"


def _run_has_recent_activity(
    run_root: Path, *, max_age_seconds: int = 600
) -> bool:
    tracked_paths: list[Path] = []
    for candidate in (
        run_root / "run.json",
        run_root / "run_control.json",
        run_root / "evolution_record.jsonl",
        run_root / "evolution_live.jsonl",
        run_root / "llm_calls.jsonl",
        run_root / "pattern_library.json",
    ):
        if candidate.is_file():
            tracked_paths.append(candidate)
    tracked_paths.extend(
        path for path in sorted(run_root.glob("label_*")) if path.is_dir()
    )
    if not tracked_paths:
        return False
    latest_mtime = max(path.stat().st_mtime for path in tracked_paths)
    return (time.time() - latest_mtime) <= max_age_seconds


def _learning_settings(
    *,
    run_payload: dict[str, Any] | None = None,
    request_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    def apply_loss_settings(source: dict[str, Any]) -> None:
        scorer_config = source.get("scorer_config")
        if not isinstance(scorer_config, dict):
            return
        losses = scorer_config.get("losses")
        if not isinstance(losses, list):
            return
        for loss in losses:
            if not isinstance(loss, dict):
                continue
            if loss.get("kind") != "correlation_structure":
                continue
            if loss.get("pair_limit") is not None:
                settings.setdefault("max_pairs", loss.get("pair_limit"))
            if loss.get("histogram_bins") is not None:
                settings.setdefault("histogram_bins", loss.get("histogram_bins"))
            break
    for source in [run_payload if isinstance(run_payload, dict) else None]:
        if not source:
            continue
        nested = source.get("request")
        if isinstance(nested, dict):
            for key in (
                "iteration_count",
                "workers",
                "population_size",
                "elite_fraction",
                "children_per_elite",
                "crossover_rate",
                "seed",
                "split",
                "validation_split",
            ):
                if key in nested and nested[key] is not None:
                    settings[key] = nested[key]
        for key in (
            "iteration_count",
            "workers",
            "population_size",
            "elite_fraction",
            "children_per_elite",
            "crossover_rate",
            "seed",
            "split",
            "validation_split",
        ):
            if key in source and source[key] is not None:
                settings.setdefault(key, source[key])
        apply_loss_settings(source)
    if isinstance(request_payload, dict):
        for key in (
            "iteration_count",
            "workers",
            "population_size",
            "elite_fraction",
            "children_per_elite",
            "crossover_rate",
            "seed",
            "split",
            "validation_split",
        ):
            if key in request_payload and request_payload[key] is not None:
                settings.setdefault(key, request_payload[key])
        apply_loss_settings(request_payload)
    return settings


def _read_learning_run_control(run_root: Path) -> dict[str, Any]:
    control_path = run_root / "run_control.json"
    if not control_path.is_file():
        return {}
    try:
        payload = json.loads(control_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_learning_run_control(run_root: Path, payload: dict[str, Any]) -> None:
    control_path = run_root / "run_control.json"
    control_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _update_learning_run_worker_target(run_root: Path, workers: int) -> int:
    desired_workers = max(1, int(workers))
    primitive_label_count = None
    run_json_path = run_root / "run.json"
    if run_json_path.is_file():
        try:
            run_payload = json.loads(run_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            run_payload = {}
        if isinstance(run_payload, dict):
            if str(run_payload.get("env_id") or "") == "pooltool":
                desired_workers = 1
            primitive_label_count = run_payload.get("primitive_label_count")
            try:
                max_workers = int(primitive_label_count)
            except (TypeError, ValueError):
                max_workers = None
            if max_workers is not None and max_workers > 0:
                desired_workers = min(desired_workers, max_workers)
            run_payload["workers"] = desired_workers
            run_json_path.write_text(
                json.dumps(run_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    control_payload = _read_learning_run_control(run_root)
    configured_workers = desired_workers
    if isinstance(control_payload, dict):
        try:
            configured_workers = max(
                1, int(control_payload.get("configured_workers") or desired_workers)
            )
        except (TypeError, ValueError):
            configured_workers = desired_workers
    control_payload.update(
        {
            "desired_workers": desired_workers,
            "configured_workers": configured_workers,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
    )
    _write_learning_run_control(run_root, control_payload)
    return desired_workers


def _mark_learning_label_done(
    run_root: Path,
    *,
    label_name: str,
    label_index: int | None = None,
) -> str:
    clean_label = label_name.strip()
    if not clean_label:
        raise ValueError("Label name must not be empty.")
    checkpoint = promote_live_best_candidate_to_checkpoint(
        artifact_root=run_root,
        label_name=clean_label,
        label_index=label_index,
    )
    control_payload = _read_learning_run_control(run_root)
    raw_labels = control_payload.get("manually_completed_labels")
    labels = [
        item.strip()
        for item in raw_labels
        if isinstance(item, str) and item.strip() and item.strip() != checkpoint.label
    ] if isinstance(raw_labels, list) else []
    control_payload["manually_completed_labels"] = labels
    raw_indices = control_payload.get("manually_completed_label_indices")
    indices: list[int] = []
    if isinstance(raw_indices, list):
        for item in raw_indices:
            try:
                coerced = int(item)
            except (TypeError, ValueError):
                continue
            if coerced > 0 and coerced != checkpoint.label_index:
                indices.append(coerced)
    control_payload["manually_completed_label_indices"] = sorted(set(indices))
    control_payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
    _write_learning_run_control(run_root, control_payload)
    return checkpoint.label


def _restart_learning_job_for_control_update(
    *,
    env_id: str,
    run_id: str,
    kind: str,
    job_manager: DashboardJobManager,
) -> None:
    snapshot = _find_learning_job_for_run(
        env_id=env_id,
        run_id=run_id,
        kind=kind,
        requested_job_id=None,
        job_manager=job_manager,
    )
    if snapshot is None:
        return
    if not snapshot.metadata.get("supports_pause", False):
        return
    if snapshot.status in {"queued", "running"}:
        job_manager.pause(snapshot.job_id)
        job_manager.resume(snapshot.job_id)
        return
    can_resume = snapshot.metadata.get("supports_pause", False) or (
        isinstance(snapshot.metadata.get("resume_argv"), list)
        and bool(snapshot.metadata.get("resume_argv"))
    )
    if can_resume and snapshot.status in {"paused", "failed"}:
        job_manager.resume(snapshot.job_id)


def _learning_progress_context(
    snapshot: DashboardJobSnapshot,
    *,
    layout: ArtifactLayout,
) -> dict[str, Any] | None:
    target = _learning_run_target(snapshot)
    if target is None:
        return None
    run_root = target["run_root"]
    env_id = target["env_id"]
    if not run_root.exists():
        return None
    run_refresh_token = _run_refresh_token(run_root)
    cache_key = (
        "run-detail",
        env_id,
        str(run_root),
        run_refresh_token,
    )

    def build_run_detail() -> Any | None:
        try:
            return load_pattern_run_detail_from_root(env_id, run_root)
        except FileNotFoundError:
            return None

    run_detail = _cached_value(cache_key, ttl_seconds=2.0, builder=build_run_detail)
    if run_detail is None:
        return None
    request_payload = (
        snapshot.metadata.get("request")
        if isinstance(snapshot.metadata.get("request"), dict)
        else None
    )
    return {
        "kind": target["kind"],
        "run_id": target["run_id"],
        "run_url": target["run_url"],
        "refresh_token": run_refresh_token,
        "settings": _learning_settings(
            run_payload=run_detail.raw_run_payload,
            request_payload=request_payload,
        ),
        "run_detail": run_detail,
    }


def _learning_run_target(snapshot: DashboardJobSnapshot) -> dict[str, Any] | None:
    action = snapshot.metadata.get("action")
    if action not in {"supervised-learn-and-evaluate", AUTOMATIC_LAUNCH_ACTION, UNSUPERVISED_LAUNCH_ACTION}:
        return None
    env_id = snapshot.metadata.get("env_id")
    run_dir_value = snapshot.metadata.get("run_dir")
    if not isinstance(env_id, str):
        return None
    run_root = (
        Path(run_dir_value)
        if isinstance(run_dir_value, str) and run_dir_value
        else None
    )
    if run_root is None:
        return None
    metadata_kind = snapshot.metadata.get("workflow_kind")
    if _is_automatic_action(action) or _is_automatic_kind(metadata_kind):
        kind = AUTOMATIC_WORKFLOW_KIND
    elif _is_unsupervised_action(action) or _is_unsupervised_kind(metadata_kind):
        kind = UNSUPERVISED_WORKFLOW_KIND
    else:
        kind = "supervised"
    run_url = _run_url_for_kind(env_id, run_root.name, kind)
    return {
        "kind": kind,
        "env_id": env_id,
        "run_root": run_root,
        "run_id": run_root.name,
        "run_url": run_url,
    }


def _evaluation_job_target(snapshot: DashboardJobSnapshot) -> dict[str, Any] | None:
    metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
    if metadata.get("action") != "evaluate-deepphy":
        return None
    env_id = metadata.get("env_id")
    run_id = metadata.get("run_id")
    evaluation_run_id = metadata.get("evaluation_run_id")
    if not all(isinstance(value, str) and value for value in (env_id, run_id, evaluation_run_id)):
        return None
    pattern_run_dir = metadata.get("pattern_run_dir")
    run_kind = metadata.get("run_kind")
    use_automatic = _is_automatic_kind(run_kind) or (
        isinstance(pattern_run_dir, str)
        and f"/{AUTOMATIC_RUNS_DIRNAME}/" in pattern_run_dir
    )
    use_unsupervised = _is_unsupervised_kind(run_kind) or (
        isinstance(pattern_run_dir, str)
        and f"/{UNSUPERVISED_RUNS_DIRNAME}/" in pattern_run_dir
        and not use_automatic
    )
    return {
        "env_id": env_id,
        "run_id": run_id,
        "evaluation_run_id": evaluation_run_id,
        "evaluation_url": (
            f"/automatic/{env_id}/{run_id}/evaluations/{evaluation_run_id}"
            if use_automatic
            else (
                f"/unsupervised/{env_id}/{run_id}/evaluations/{evaluation_run_id}"
                if use_unsupervised
                else f"/runs/{env_id}/{run_id}/evaluations/{evaluation_run_id}"
            )
        ),
    }


def _library_evaluation_job_target(
    snapshot: DashboardJobSnapshot,
) -> dict[str, Any] | None:
    metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
    if metadata.get("action") != "evaluate-library-case":
        return None
    env_id = metadata.get("env_id")
    run_id = metadata.get("run_id")
    library_evaluation_run_id = metadata.get("library_evaluation_run_id")
    if not all(
        isinstance(value, str) and value
        for value in (env_id, run_id, library_evaluation_run_id)
    ):
        return None
    return {
        "env_id": env_id,
        "run_id": run_id,
        "library_evaluation_run_id": library_evaluation_run_id,
        "library_evaluation_url": (
            f"/runs/{env_id}/{run_id}/library-evaluations/{library_evaluation_run_id}"
        ),
    }


def _pooltool_reward_job_target(
    snapshot: DashboardJobSnapshot,
) -> dict[str, Any] | None:
    metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
    action = metadata.get("action")
    run_id = metadata.get("reward_program_run_id") or metadata.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        return None
    if action == POOLTOOL_REWARD_OPTIMIZATION_ACTION:
        return {
            "kind": "optimization",
            "run_id": run_id,
            "url": f"/pooltool/reward-programs/optimization/{run_id}",
        }
    if action == POOLTOOL_REWARD_VALUE_MODEL_ACTION:
        return {
            "kind": "value-model",
            "run_id": run_id,
            "url": f"/pooltool/reward-programs/value-model/{run_id}",
        }
    return None


def _evaluation_refresh_token(eval_root: Path) -> str:
    tracked_paths: list[Path] = []
    for candidate in (eval_root / "summary.json", eval_root / "results.json"):
        if candidate.is_file():
            tracked_paths.append(candidate)
    tracked_paths.extend(
        path
        for path in sorted(eval_root.glob("tasks/*/*/task_result.json"))
        if path.is_file()
    )
    if not tracked_paths:
        return "missing"
    latest_mtime_ns = max(path.stat().st_mtime_ns for path in tracked_paths)
    return f"{len(tracked_paths)}:{latest_mtime_ns}"


def _load_run_payload_for_root(run_root: Path) -> dict[str, Any]:
    run_json_path = run_root / "run.json"
    if not run_json_path.is_file():
        return {}
    try:
        payload = json.loads(run_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _candidate_evaluation_roots_for_snapshot(
    snapshot: DashboardJobSnapshot,
    *,
    layout: ArtifactLayout,
) -> list[tuple[Path, dict[str, Any]]]:
    metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
    action = str(metadata.get("action") or "")
    candidates: list[tuple[Path, dict[str, Any]]] = []
    seen: set[Path] = set()

    def add_candidate(path: Path | None, payload: dict[str, Any]) -> None:
        if path is None:
            return
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append((resolved, payload))

    if action == "evaluate-deepphy":
        target = _evaluation_job_target(snapshot)
        if target is None:
            return []
        eval_root_value = metadata.get("evaluation_run_dir")
        eval_root = (
            Path(eval_root_value)
            if isinstance(eval_root_value, str) and eval_root_value.strip()
            else (
                layout.data_root
                / "patterns"
                / target["env_id"]
                / target["run_id"]
                / "deepphy_eval"
                / target["evaluation_run_id"]
            )
        )
        add_candidate(
            eval_root,
            {
                "env_id": target["env_id"],
                "run_id": target["run_id"],
                "evaluation_run_id": target["evaluation_run_id"],
                "evaluation_url": target["evaluation_url"],
                "benchmark_task_count": _coerce_int(metadata.get("benchmark_task_count")),
            },
        )
        return candidates

    if action not in {"supervised-learn-and-evaluate", AUTOMATIC_LAUNCH_ACTION, UNSUPERVISED_LAUNCH_ACTION}:
        return []
    learning_target = _learning_run_target(snapshot)
    if learning_target is None:
        return []
    run_root = learning_target["run_root"]
    env_id = learning_target["env_id"]
    run_id = learning_target["run_id"]
    run_kind = learning_target["kind"]
    run_payload = _load_run_payload_for_root(run_root)
    auto_eval_run_id = run_payload.get("auto_evaluation_run_id")
    evaluation_root = run_root / "deepphy_eval"
    evaluation_url_prefix = (
        f"{_automatic_run_url(env_id, run_id)}/evaluations"
        if run_kind == AUTOMATIC_WORKFLOW_KIND
        else f"/runs/{env_id}/{run_id}/evaluations"
    )

    metadata_eval_root = metadata.get("evaluation_run_dir")
    if isinstance(metadata_eval_root, str) and metadata_eval_root.strip():
        eval_path = Path(metadata_eval_root)
        add_candidate(
            eval_path,
            {
                "env_id": env_id,
                "run_id": run_id,
                "evaluation_run_id": eval_path.name,
                "evaluation_url": f"{evaluation_url_prefix}/{eval_path.name}",
                "benchmark_task_count": None,
            },
        )
    metadata_eval_id = metadata.get("evaluation_run_id")
    if isinstance(metadata_eval_id, str) and metadata_eval_id.strip():
        add_candidate(
            evaluation_root / metadata_eval_id.strip(),
            {
                "env_id": env_id,
                "run_id": run_id,
                "evaluation_run_id": metadata_eval_id.strip(),
                "evaluation_url": f"{evaluation_url_prefix}/{metadata_eval_id.strip()}",
                "benchmark_task_count": None,
            },
        )
    if isinstance(auto_eval_run_id, str) and auto_eval_run_id.strip():
        add_candidate(
            evaluation_root / auto_eval_run_id.strip(),
            {
                "env_id": env_id,
                "run_id": run_id,
                "evaluation_run_id": auto_eval_run_id.strip(),
                "evaluation_url": f"{evaluation_url_prefix}/{auto_eval_run_id.strip()}",
                "benchmark_task_count": summarize_deepphy_benchmark(
                    env_id,
                    preset="full",
                    task_limit=default_benchmark_task_limit(env_id, preset="full"),
                ).task_count,
            },
        )
    if evaluation_root.is_dir():
        benchmark_task_count = summarize_deepphy_benchmark(
            env_id,
            preset="full",
            task_limit=default_benchmark_task_limit(env_id, preset="full"),
        ).task_count
        for eval_path in sorted(
            (path for path in evaluation_root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        ):
            add_candidate(
                eval_path,
                {
                    "env_id": env_id,
                    "run_id": run_id,
                    "evaluation_run_id": eval_path.name,
                    "evaluation_url": f"{evaluation_url_prefix}/{eval_path.name}",
                    "benchmark_task_count": benchmark_task_count,
                },
            )
    return candidates


def _evaluation_progress_payload(
    snapshot: DashboardJobSnapshot, *, layout: ArtifactLayout
) -> dict[str, Any] | None:
    candidates = _candidate_evaluation_roots_for_snapshot(snapshot, layout=layout)
    candidate_signature = tuple(
        (
            str(eval_root),
            payload.get("evaluation_run_id"),
            payload.get("benchmark_task_count"),
            _evaluation_refresh_token(eval_root) if eval_root.is_dir() else "missing",
        )
        for eval_root, payload in candidates
    )
    cache_key = (
        "evaluation-progress",
        snapshot.job_id,
        snapshot.status,
        candidate_signature,
    )

    def build() -> dict[str, Any] | None:
        summaries: list[tuple[dict[str, Any], Any]] = []
        fallback_payload: dict[str, Any] | None = None
        for eval_root, payload in candidates:
            benchmark_task_count = _coerce_int(payload.get("benchmark_task_count"))
            if not eval_root.is_dir():
                if fallback_payload is None:
                    fallback_payload = {
                        **payload,
                        "status": "pending",
                        "completed_tasks": 0,
                        "total_tasks": benchmark_task_count,
                        "success_count": 0,
                        "success_ratio": None,
                        "latest_task_id": None,
                        "latest_task_status": None,
                        "refresh_token": "missing",
                    }
                continue
            try:
                summary = load_evaluation_run_summary(eval_root)
            except (FileNotFoundError, ValueError):
                continue
            summaries.append(
                (
                    {
                        **payload,
                        "eval_root": eval_root,
                        "benchmark_task_count": benchmark_task_count,
                    },
                    summary,
                )
            )
        if summaries:
            summaries.sort(
                key=lambda item: (
                    0 if item[1].status in {"running", "pending"} else 1,
                    item[1].run_id,
                )
            )
            payload, summary = summaries[0]
            completed_tasks = summary.task_count
            total_tasks = payload["benchmark_task_count"] or (
                summary.task_count if summary.task_count > 0 else None
            )
            latest_task = next(
                (
                    item
                    for item in reversed(summary.task_summaries)
                    if item.terminal_status or item.total_duration_seconds is not None
                ),
                None,
            )
            return {
                "run_id": summary.run_id,
                "status": summary.status,
                "completed_tasks": completed_tasks,
                "total_tasks": total_tasks,
                "success_count": summary.success_count,
                "success_ratio": summary.success_ratio,
                "latest_task_id": latest_task.task_id if latest_task is not None else None,
                "latest_task_status": (
                    latest_task.terminal_status if latest_task is not None else None
                ),
                "evaluation_url": payload["evaluation_url"],
                "refresh_token": _evaluation_refresh_token(payload["eval_root"]),
            }
        return fallback_payload

    return _cached_value(cache_key, ttl_seconds=2.0, builder=build)


def _find_learning_job_for_run(
    *,
    env_id: str,
    run_id: str,
    kind: Literal["supervised", "automatic", "unsupervised"],
    requested_job_id: str | None,
    job_manager: DashboardJobManager,
) -> DashboardJobSnapshot | None:
    if requested_job_id:
        try:
            snapshot = job_manager.get_snapshot(requested_job_id)
        except KeyError:
            return None
        target = _learning_run_target(snapshot)
        if (
            target is not None
            and target["env_id"] == env_id
            and target["run_id"] == run_id
            and target["kind"] == kind
        ):
            return snapshot
        return None
    for snapshot in job_manager.list_snapshots():
        target = _learning_run_target(snapshot)
        if (
            target is not None
            and target["env_id"] == env_id
            and target["run_id"] == run_id
            and target["kind"] == kind
        ):
            return snapshot
    return None


def _find_evaluation_job_for_run(
    *,
    env_id: str,
    run_id: str,
    evaluation_run_id: str,
    requested_job_id: str | None,
    job_manager: DashboardJobManager,
) -> DashboardJobSnapshot | None:
    def matches(snapshot: DashboardJobSnapshot) -> bool:
        target = _evaluation_job_target(snapshot)
        return (
            target is not None
            and target["env_id"] == env_id
            and target["run_id"] == run_id
            and target["evaluation_run_id"] == evaluation_run_id
        )

    if requested_job_id:
        try:
            snapshot = job_manager.get_snapshot(requested_job_id)
        except KeyError:
            return None
        return snapshot if matches(snapshot) else None
    for snapshot in job_manager.list_snapshots():
        if matches(snapshot):
            return snapshot
    return None


def _select_evolution_label(
    labels: list[Any],
    requested_label_name: str | None,
) -> Any | None:
    if not labels:
        return None
    if requested_label_name:
        for label in labels:
            if getattr(label, "name", None) == requested_label_name:
                return label
    return labels[0]


def _normalize_reward_metric(value: Any) -> str:
    if value == "validation":
        return "validation"
    return "train"


def _updated_query_url(request: Request, **updates: Any) -> str:
    params = dict(request.query_params)
    for key, value in updates.items():
        if value is None:
            params.pop(key, None)
            continue
        params[key] = str(value)
    query = urlencode(params)
    return f"{request.url.path}?{query}" if query else request.url.path


def _learning_progress_payload(
    snapshot: DashboardJobSnapshot,
    *,
    layout: ArtifactLayout,
) -> dict[str, Any] | None:
    context = _learning_progress_context(snapshot, layout=layout)
    if context is None:
        return None
    run_detail = context["run_detail"]
    progress_summary = _learning_progress_summary(
        snapshot=snapshot,
        context=context,
        layout=layout,
    )
    return {
        "kind": context["kind"],
        "run_id": context["run_id"],
        "run_url": context["run_url"],
        "refresh_token": context["refresh_token"],
        "candidate_count": run_detail.candidate_count,
        "label_count": len(run_detail.label_telemetry),
        "iteration_count": run_detail.iteration_count,
        "average_score": run_detail.average_score,
        "settings": context["settings"],
        "progress_summary": progress_summary,
    }


def _learning_progress_summary(
    *,
    snapshot: DashboardJobSnapshot,
    context: dict[str, Any],
    layout: ArtifactLayout,
) -> dict[str, Any]:
    env_id = context["run_detail"].env_id
    kind = context["kind"]
    request_payload = (
        snapshot.metadata.get("request")
        if isinstance(snapshot.metadata.get("request"), dict)
        else {}
    )
    control_payload = _read_learning_run_control(context["run_detail"].root)
    ordered_labels, completed_indices = _dashboard_completed_label_indices(
        env_id=env_id,
        kind=kind,
        request_payload=request_payload,
        run_payload=context["run_detail"].raw_run_payload,
        run_detail=context["run_detail"],
        layout=layout,
        control_payload=control_payload,
    )
    total_labels = len(ordered_labels)
    completed_labels = len(completed_indices)
    parsing_lines = _read_log_tail_lines(snapshot.log_path, limit=1200)
    if not parsing_lines:
        parsing_lines = snapshot.recent_output
    parsed = _parse_learning_recent_output(parsing_lines)
    active_label = parsed.get("active_label")
    active_label_index = parsed.get("active_label_index")
    if active_label is None and completed_labels < total_labels:
        next_index = next(
            (index for index in range(1, total_labels + 1) if index not in completed_indices),
            None,
        )
        if next_index is not None:
            active_label_index = next_index
            active_label = ordered_labels[next_index - 1]
    workers = parsed.get("workers", [])
    stage = parsed.get("stage")
    if snapshot.status == "paused":
        stage = "paused"
        workers = [{**worker, "stage": "paused"} for worker in workers]
    elif snapshot.status == "failed":
        stage = "failed"
        workers = [{**worker, "stage": "failed"} for worker in workers]
    elif snapshot.status == "queued" and not stage:
        stage = "queued"
        workers = [{**worker, "stage": worker.get("stage") or "queued"} for worker in workers]
    if not workers and active_label is not None:
        workers = _synthesized_learning_workers(
            active_label=active_label,
            active_label_index=active_label_index,
            stage=stage,
            candidate_evaluations=parsed.get("candidate_evaluations"),
            latest_score=parsed.get("latest_score"),
            latest_validation_score=parsed.get("latest_validation_score"),
            desired_workers=_desired_learning_worker_count(
                request_payload=request_payload,
                run_payload=context["run_detail"].raw_run_payload,
                control_payload=control_payload,
            ),
        )
    return {
        "completed_labels": completed_labels,
        "total_labels": total_labels,
        "workers": workers,
        "active_label": active_label,
        "active_label_index": active_label_index,
        "candidate_evaluations": parsed.get("candidate_evaluations"),
        "latest_score": parsed.get("latest_score"),
        "latest_validation_score": parsed.get("latest_validation_score"),
        "stage": stage,
    }


def _desired_learning_worker_count(
    *,
    request_payload: dict[str, Any],
    run_payload: dict[str, Any],
    control_payload: dict[str, Any],
) -> int:
    for payload, key in (
        (control_payload, "desired_workers"),
        (request_payload, "workers"),
        (run_payload, "workers"),
    ):
        try:
            value = int(payload.get(key))
        except (AttributeError, TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 1


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
    worker_count = max(1, int(desired_workers))
    return [
        {
            "worker": worker_id,
            "label": active_label,
            "label_index": active_label_index,
            "candidate_evaluations": candidate_evaluations,
            "latest_score": latest_score,
            "latest_validation_score": latest_validation_score,
            "stage": stage,
        }
        for worker_id in range(1, worker_count + 1)
    ]


def _learning_total_label_count(
    *,
    env_id: str,
    kind: str,
    request_payload: dict[str, Any],
    run_payload: dict[str, Any],
    layout: ArtifactLayout,
) -> int:
    if kind == "supervised":
        target = load_target_label_config(env_id, layout=layout)
        return target.label_count if target is not None else 0
    proposed_labels = run_payload.get("proposed_labels")
    if isinstance(proposed_labels, list):
        return len([item for item in proposed_labels if isinstance(item, str) and item.strip()])
    proposed_label_count = request_payload.get("proposed_label_count")
    return int(proposed_label_count) if isinstance(proposed_label_count, int) else 0


def _label_name_for_index(
    *,
    env_id: str,
    kind: str,
    index: int,
    request_payload: dict[str, Any],
    run_payload: dict[str, Any],
    layout: ArtifactLayout,
) -> str | None:
    if index <= 0:
        return None
    if kind == "supervised":
        target = load_target_label_config(env_id, layout=layout)
        if target is None or index > len(target.labels):
            return None
        return target.labels[index - 1]
    proposed_labels = run_payload.get("proposed_labels")
    if isinstance(proposed_labels, list) and index <= len(proposed_labels):
        item = proposed_labels[index - 1]
        return item if isinstance(item, str) else None
    return None


_LABEL_LINE_RE = re.compile(r"label=(.+?)(?: (?=[a-z_]+=)|$)")
_INT_FIELD_RE = re.compile(
    r"(?:^| )(?P<key>worker|index|total_labels|evaluation_index|active_workers|desired_workers|queued_labels)=(?P<value>\d+)"
)
_FLOAT_FIELD_RE = re.compile(r"(?:^| )(?P<key>score|validation_score)=(-?\d+(?:\.\d+)?)")
_FINGERPRINT_FIELD_RE = re.compile(r"(?:^| )candidate_fingerprint=([^ ]+)")


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
    recent_events: list[dict[str, Any]] = []

    def _reset_runtime_state(*, keep_stage: bool = False) -> None:
        workers.clear()
        labels.clear()
        recent_events.clear()
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
        label_key = _dashboard_label_key(label_name, label_index)
        state_key = label_key or label_name
        label_state = labels.get(state_key)
        if label_state is None:
            label_state = {
                "label": label_name,
                "label_index": label_index,
                "label_key": label_key,
                "candidate_evaluations": 0,
                "event_count": 0,
                "best_score": None,
                "best_validation_score": None,
                "best_fingerprint": None,
            }
            labels[state_key] = label_state
        elif label_index is not None and label_state.get("label_index") is None:
            label_state["label_index"] = label_index
            label_state["label_key"] = _dashboard_label_key(label_name, label_index)
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
        if "patterns.learning.start:" in line:
            _reset_runtime_state()
            continue
        if "patterns.learning.resume:" in line:
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
                label_state = _label_state(
                    label_name=label,
                    label_index=worker_state.get("label_index") if worker_state is not None else None,
                )
                if ints.get("evaluation_index") is not None:
                    label_state["candidate_evaluations"] = max(
                        int(label_state["candidate_evaluations"]),
                        ints["evaluation_index"],
                    )
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
                label_state = _label_state(
                    label_name=label,
                    label_index=worker_state.get("label_index") if worker_state is not None else None,
                )
                if ints.get("evaluation_index") is not None:
                    label_state["candidate_evaluations"] = max(
                        int(label_state["candidate_evaluations"]),
                        ints["evaluation_index"],
                    )
                label_state["event_count"] = int(label_state["event_count"]) + 1
                score = floats.get("score")
                validation_score = floats.get("validation_score")
                best_score = label_state.get("best_score")
                if score is not None and (
                    best_score is None or score >= float(best_score)
                ):
                    label_state["best_score"] = score
                    label_state["best_validation_score"] = validation_score
                    label_state["best_fingerprint"] = fingerprint
                recent_events.append(
                    {
                        "label": label,
                        "label_index": label_state.get("label_index"),
                        "label_key": label_state.get("label_key"),
                        "evaluation_index": ints.get("evaluation_index"),
                        "candidate_fingerprint": fingerprint,
                        "score": score,
                        "validation_score": validation_score,
                        "worker": worker_id,
                    }
                )
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
        key=lambda item: (
            item["label_index"] if isinstance(item.get("label_index"), int) else 10_000,
            item["label"],
        ),
    )
    state["workers"] = [workers[key] for key in sorted(workers)]
    state["provisional_labels"] = provisional_labels
    state["provisional_labels_by_key"] = {
        item["label_key"]: item
        for item in provisional_labels
        if isinstance(item.get("label_key"), str) and item["label_key"]
    }
    state["provisional_events"] = recent_events[-50:]
    state["provisional_candidate_count"] = sum(
        int(item["candidate_evaluations"]) for item in provisional_labels
    )
    return state


def _dashboard_label_key(label: str, label_index: int | None) -> str | None:
    if label_index is None:
        return None
    slug = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    return f"label_{label_index:03d}_{slug}" if slug else None


def _dashboard_completed_label_indices(
    *,
    env_id: str,
    kind: str,
    request_payload: dict[str, Any],
    run_payload: dict[str, Any],
    run_detail: Any,
    layout: ArtifactLayout,
    control_payload: dict[str, Any] | None = None,
) -> tuple[list[str | None], set[int]]:
    total_labels = _learning_total_label_count(
        env_id=env_id,
        kind=kind,
        request_payload=request_payload,
        run_payload=run_payload,
        layout=layout,
    )
    ordered_labels = [
        _label_name_for_index(
            env_id=env_id,
            kind=kind,
            index=index,
            request_payload=request_payload,
            run_payload=run_payload,
            layout=layout,
        )
        for index in range(1, total_labels + 1)
    ]
    label_to_indices: dict[str, set[int]] = {}
    for index, label_name in enumerate(ordered_labels, start=1):
        if label_name is None:
            continue
        label_to_indices.setdefault(label_name, set()).add(index)

    def _indices_for_label_reference(raw_label: Any) -> set[int]:
        if not isinstance(raw_label, str):
            return set()
        clean_label = raw_label.strip()
        if not clean_label:
            return set()
        direct = label_to_indices.get(clean_label)
        if direct:
            return set(direct)
        match = re.match(r"^label_(\d{3})_", clean_label)
        if match is None:
            return set()
        label_index = int(match.group(1))
        if 1 <= label_index <= total_labels:
            return {label_index}
        return set()

    explicit_completed_indices: set[int] = set()
    inferred_completed_indices: set[int] = set()
    for label in getattr(run_detail, "label_telemetry", []):
        inferred_completed_indices.update(
            _indices_for_label_reference(getattr(label, "name", None))
        )

    raw_manual_labels = run_payload.get("manually_completed_labels")
    if not isinstance(raw_manual_labels, list):
        raw_manual_labels = []
    merged_manual_labels = [*raw_manual_labels]
    if isinstance(control_payload, dict):
        control_manual_labels = control_payload.get("manually_completed_labels")
        if isinstance(control_manual_labels, list):
            merged_manual_labels.extend(control_manual_labels)
    for raw_label in merged_manual_labels:
        explicit_completed_indices.update(_indices_for_label_reference(raw_label))

    if isinstance(control_payload, dict):
        raw_manual_indices = control_payload.get("manually_completed_label_indices")
        if isinstance(raw_manual_indices, list):
            for raw_index in raw_manual_indices:
                if isinstance(raw_index, bool):
                    continue
                try:
                    label_index = int(raw_index)
                except (TypeError, ValueError):
                    continue
                if 1 <= label_index <= total_labels:
                    explicit_completed_indices.add(label_index)

    raw_completed_label_count = run_payload.get("completed_label_count")
    if isinstance(raw_completed_label_count, int) and raw_completed_label_count >= 0:
        completed_indices = set(
            index
            for index in explicit_completed_indices
            if 1 <= index <= raw_completed_label_count
        )
        for label_index in range(1, total_labels + 1):
            if len(completed_indices) >= raw_completed_label_count:
                break
            if label_index in completed_indices:
                continue
            if (
                label_index in inferred_completed_indices
                or label_index <= raw_completed_label_count
            ):
                completed_indices.add(label_index)
    else:
        completed_indices = explicit_completed_indices | inferred_completed_indices

    return ordered_labels, completed_indices


def _format_validation_error(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        for error in exc.errors()
    )


def _format_display_time(value: str | None) -> str:
    if not value:
        return "—"
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return value
    localized = parsed.astimezone(DISPLAY_TIMEZONE)
    return localized.strftime("%Y-%m-%d %H:%M:%S %Z")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_duration_seconds(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    seconds = float(value)
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    minutes = int(seconds // 60)
    remaining_seconds = seconds - (minutes * 60)
    return f"{minutes}m {remaining_seconds:.1f}s"


def _format_millions(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{float(value) / 1_000_000:.2f}M"


def _system_memory_usage_payload() -> dict[str, Any]:
    meminfo_path = Path("/proc/meminfo")
    values: dict[str, int] = {}
    try:
        lines = meminfo_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {"available": False}
    for line in lines:
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        parts = raw_value.split()
        if not parts:
            continue
        try:
            values[key] = int(parts[0])
        except ValueError:
            continue
    total_kb = values.get("MemTotal")
    available_kb = values.get("MemAvailable")
    if not isinstance(total_kb, int) or not isinstance(available_kb, int) or total_kb <= 0:
        return {"available": False}
    used_kb = max(0, total_kb - available_kb)
    used_pct = (used_kb / total_kb) * 100.0
    return {
        "available": True,
        "used_kb": used_kb,
        "total_kb": total_kb,
        "used_display": _format_kib(used_kb),
        "total_display": _format_kib(total_kb),
        "percent": round(used_pct, 1),
    }


def _system_cpu_usage_payload() -> dict[str, Any]:
    stat_path = Path("/proc/stat")
    try:
        first_line = stat_path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return {"available": False}
    parts = first_line.split()
    if len(parts) < 5 or parts[0] != "cpu":
        return {"available": False}
    try:
        values = [int(part) for part in parts[1:]]
    except ValueError:
        return {"available": False}
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    with _CPU_USAGE_LOCK:
        global _CPU_USAGE_PREVIOUS_SAMPLE
        previous = _CPU_USAGE_PREVIOUS_SAMPLE
        _CPU_USAGE_PREVIOUS_SAMPLE = (total, idle)
    if previous is None:
        return {"available": False}
    total_delta = total - previous[0]
    idle_delta = idle - previous[1]
    if total_delta <= 0:
        return {"available": False}
    used_pct = max(0.0, min(100.0, (1.0 - (idle_delta / total_delta)) * 100.0))
    return {
        "available": True,
        "percent": round(used_pct, 1),
        "percent_display": _format_pct(used_pct),
    }


def _nvidia_smi_gpu_rows() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]

    def build() -> list[dict[str, Any]]:
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        rows: list[dict[str, Any]] = []
        for raw_line in completed.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                continue
            try:
                used_mb = int(parts[0])
                total_mb = int(parts[1])
                utilization_gpu = int(parts[2])
                temperature_gpu = int(parts[3])
            except ValueError:
                continue
            rows.append(
                {
                    "used_mb": used_mb,
                    "total_mb": total_mb,
                    "utilization_gpu": utilization_gpu,
                    "temperature_gpu": temperature_gpu,
                }
            )
        return rows

    return _cached_value(("nvidia-smi-gpu-rows",), ttl_seconds=2.0, builder=build)


def _system_gpu_usage_payload() -> dict[str, Any]:
    rows = _nvidia_smi_gpu_rows()
    if not rows:
        return {"available": False}
    used_total_mb = 0
    total_total_mb = 0
    gpu_count = 0
    utilization_values: list[int] = []
    for row in rows:
        used_mb = row["used_mb"]
        total_mb = row["total_mb"]
        used_total_mb += used_mb
        total_total_mb += total_mb
        gpu_count += 1
        utilization = row.get("utilization_gpu")
        if isinstance(utilization, int):
            utilization_values.append(utilization)
    if total_total_mb <= 0:
        return {"available": False}
    used_pct = (used_total_mb / total_total_mb) * 100.0
    return {
        "available": True,
        "gpu_count": gpu_count,
        "used_mb": used_total_mb,
        "total_mb": total_total_mb,
        "used_display": _format_mib(used_total_mb),
        "total_display": _format_mib(total_total_mb),
        "percent": round(used_pct, 1),
        "utilization_percent": round(sum(utilization_values) / len(utilization_values), 1)
        if utilization_values
        else None,
        "utilization_peak_percent": max(utilization_values) if utilization_values else None,
        "utilization_display": _format_pct(
            round(sum(utilization_values) / len(utilization_values), 1)
        )
        if utilization_values
        else "—",
    }


def _system_temperature_payload() -> dict[str, Any]:
    cpu_temp_c: float | None = None
    thermal_root = Path("/sys/class/thermal")
    if thermal_root.is_dir():
        cpu_values: list[float] = []
        for zone in sorted(thermal_root.glob("thermal_zone*")):
            temp_path = zone / "temp"
            try:
                raw_value = temp_path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            try:
                temp_millideg = int(raw_value)
            except ValueError:
                continue
            cpu_values.append(temp_millideg / 1000.0)
        if cpu_values:
            cpu_temp_c = max(cpu_values)

    gpu_temp_c: int | None = None
    gpu_rows = _nvidia_smi_gpu_rows()
    if gpu_rows:
        gpu_values = [
            row["temperature_gpu"]
            for row in gpu_rows
            if isinstance(row.get("temperature_gpu"), int)
        ]
        if gpu_values:
            gpu_temp_c = max(gpu_values)

    hottest_candidates = [
        value for value in (cpu_temp_c, gpu_temp_c) if isinstance(value, (int, float))
    ]
    hottest_temp_c = max(hottest_candidates) if hottest_candidates else None
    if hottest_temp_c is None:
        return {"available": False}
    return {
        "available": True,
        "hottest_temp_c": hottest_temp_c,
        "hottest_display": _format_temp_c(hottest_temp_c),
        "cpu_temp_c": cpu_temp_c,
        "cpu_display": _format_temp_c(cpu_temp_c),
        "gpu_temp_c": gpu_temp_c,
        "gpu_display": _format_temp_c(gpu_temp_c),
    }


def _system_resource_snapshot() -> dict[str, Any]:
    gpu_memory = _system_gpu_usage_payload()
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "timestamp_display": _format_display_time(datetime.utcnow().isoformat() + "Z"),
        "cpu": _system_cpu_usage_payload(),
        "gpu": {
            "available": gpu_memory.get("utilization_percent") is not None,
            "percent": gpu_memory.get("utilization_percent"),
            "percent_display": gpu_memory.get("utilization_display", "—"),
        },
        "ram": _system_memory_usage_payload(),
        "vram": gpu_memory,
        "temperature": _system_temperature_payload(),
    }


def _display_learning_stage(value: Any) -> str:
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


def _render_trajectory_chart(
    payload: Any,
    *,
    title: str = "Reward over iterations by label",
    x_label: str = "Iteration",
    metric: str = "train",
) -> Markup:
    if not isinstance(payload, dict):
        return Markup('<p class="empty">No reward data yet.</p>')
    raw_series = payload.get("series")
    if not isinstance(raw_series, list) or not raw_series:
        return Markup('<p class="empty">No reward data yet.</p>')

    series: list[dict[str, Any]] = []
    values: list[float] = []
    max_iteration = 0
    reward_key = "validation_reward" if metric == "validation" else "train_reward"
    for item in raw_series:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        raw_values = item.get("values")
        if not isinstance(label, str) or not isinstance(raw_values, list):
            continue
        cleaned_values: list[tuple[int, float]] = []
        for point in raw_values:
            if not isinstance(point, dict):
                continue
            iteration = point.get("iteration")
            reward = point.get(reward_key)
            if not isinstance(iteration, int):
                continue
            if not isinstance(reward, (int, float)):
                continue
            reward_value = float(reward)
            if reward_value < -1.0 or reward_value > 1.0:
                continue
            cleaned_values.append((iteration, reward_value))
            values.append(reward_value)
            max_iteration = max(max_iteration, iteration)
        if cleaned_values:
            series.append({"label": label, "values": cleaned_values})

    if not series or not values:
        return Markup('<p class="empty">No reward data yet.</p>')

    width = 980
    height = 420
    margin_top = 20
    margin_right = 20
    margin_bottom = 56
    margin_left = 52
    inner_width = width - margin_left - margin_right
    inner_height = height - margin_top - margin_bottom
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0

    def x(iteration: int) -> float:
        return margin_left + ((iteration or 0) / max(max_iteration, 1)) * inner_width

    def y(value: float) -> float:
        return margin_top + ((max_value - value) / (max_value - min_value)) * inner_height

    palette = [
        "#b35c00",
        "#0d7a6f",
        "#6b3fa0",
        "#a23b5a",
        "#1f5f8b",
        "#5a7f22",
        "#8c4c16",
        "#365d73",
    ]
    tick_values = [min_value + ((max_value - min_value) * index) / 4 for index in range(5)]
    tick_count = min(max_iteration + 1, 6)
    if tick_count <= 1:
        x_ticks = [0]
    else:
        x_ticks = [round((max_iteration * index) / (tick_count - 1)) for index in range(tick_count)]

    legend_markup: list[str] = []
    line_markup: list[str] = []
    for index, item in enumerate(series):
        palette_index = item.get("palette_index")
        if not isinstance(palette_index, int):
            palette_index = index
        color = palette[palette_index % len(palette)]
        points = item["values"]
        running_best_points: list[tuple[int, float]] = []
        best_so_far: float | None = None
        for iteration, reward in points:
            if best_so_far is None or reward > best_so_far:
                best_so_far = reward
                running_best_points.append((iteration, reward))
        path_parts = [
            f'{"M" if point_index == 0 else "L"} {x(iteration):.1f} {y(reward):.1f}'
            for point_index, (iteration, reward) in enumerate(points)
        ]
        path = " ".join(path_parts)
        line_markup.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.4" stroke-opacity="0.42" />'
        )
        for iteration, reward in points:
            line_markup.append(
                f'<circle cx="{x(iteration):.1f}" cy="{y(reward):.1f}" r="2.2" fill="{color}" fill-opacity="0.34" />'
            )
        for best_iteration, best_reward in running_best_points:
            line_markup.append(
                f'<circle cx="{x(best_iteration):.1f}" cy="{y(best_reward):.1f}" r="7.2" fill="{color}" fill-opacity="0.16" />'
            )
            line_markup.append(
                f'<circle cx="{x(best_iteration):.1f}" cy="{y(best_reward):.1f}" r="4.8" fill="{color}" stroke="white" stroke-width="1.8" />'
            )
        legend_markup.append(
            f'<span><i class="legend-swatch" style="background:{color}; border-color:{color};"></i>{escape(item["label"])}</span>'
        )

    grid_markup = "".join(
        (
            f'<g><line x1="{margin_left}" y1="{y(value):.1f}" '
            f'x2="{width - margin_right}" y2="{y(value):.1f}" class="loss-grid-line" />'
            f'<text x="{margin_left - 8}" y="{y(value) + 4:.1f}" text-anchor="end" class="loss-axis-label">{value:.2f}</text></g>'
        )
        for value in tick_values
    )
    x_tick_markup = "".join(
        (
            f'<g><line x1="{x(tick):.1f}" y1="{margin_top}" x2="{x(tick):.1f}" y2="{height - margin_bottom}" class="loss-grid-line" />'
            f'<text x="{x(tick):.1f}" y="{height - margin_bottom + 18:.1f}" text-anchor="middle" class="loss-axis-label loss-x-label">{tick}</text></g>'
        )
        for tick in x_ticks
    )
    zero_line_markup = ""
    if min_value <= 0.0 <= max_value:
        zero_y = y(0.0)
        zero_line_markup = (
            f'<g><line x1="{margin_left}" y1="{zero_y:.1f}" '
            f'x2="{width - margin_right}" y2="{zero_y:.1f}" class="loss-zero-line" />'
            f'<text x="{margin_left - 8}" y="{zero_y + 4:.1f}" text-anchor="end" class="loss-axis-label loss-zero-label">0</text></g>'
        )

    html = f"""
    <div class="trajectory-chart trajectory-chart-large">
        <div class="loss-chart-frame">
        <div class="loss-chart-legend">
          {''.join(legend_markup)}
        </div>
        <svg viewBox="0 0 {width} {height}" class="loss-chart-svg" role="img" aria-label="{escape(title)}">
          {grid_markup}
          {zero_line_markup}
          {x_tick_markup}
          <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" class="loss-axis-line" />
          <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" class="loss-axis-line" />
          {''.join(line_markup)}
        </svg>
        <div class="loss-axis-caption">{escape(x_label)}</div>
      </div>
    </div>
    """
    return Markup(html)


def _evaluation_summary_metadata(summary_path: Path | None) -> dict[str, Any]:
    if summary_path is None or not summary_path.is_file():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_label_names(run: Any) -> list[str]:
    library_path = getattr(run, "library_path", None)
    if isinstance(library_path, Path) and library_path.is_file():
        try:
            library = load_pattern_library(library_path)
        except (OSError, ValueError, json.JSONDecodeError):
            library = None
        if library is not None:
            labels = sorted(
                {
                    detector.label.strip()
                    for detector in library.detectors
                    if isinstance(detector.label, str) and detector.label.strip()
                }
            )
            if labels:
                return labels
    labels: list[str] = []
    for item in getattr(run, "label_telemetry", []):
        label_name = getattr(item, "name", None)
        if isinstance(label_name, str) and label_name:
            labels.append(label_name)
    return sorted(set(labels))


def _candidate_lookup_maps(label_summary: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates_by_uid = {
        item.uid: item for item in getattr(label_summary, "candidates", []) if item.uid is not None
    }
    candidates_by_name = {
        item.name: item for item in getattr(label_summary, "candidates", []) if getattr(item, "name", None)
    }
    return candidates_by_uid, candidates_by_name


def _candidate_parent_candidates(candidate: Any, label_summary: Any) -> list[Any]:
    candidates_by_uid, _ = _candidate_lookup_maps(label_summary)
    return [
        candidates_by_uid[uid]
        for uid in [getattr(candidate, "parent_uid", None), getattr(candidate, "secondary_parent_uid", None)]
        if uid is not None and uid in candidates_by_uid
    ]


def _candidate_diff_sections(*, child_candidate: Any, child_code: str, parent_candidates: list[Any]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    child_lines = child_code.splitlines()
    for index, parent in enumerate(parent_candidates, start=1):
        if getattr(parent, "code_path", None) is None or not parent.code_path.is_file():
            sections.append(
                {
                    "title": f"Parent {index}",
                    "parent": parent,
                    "diff_text": None,
                    "error": "Parent source code is not available.",
                }
            )
            continue
        parent_code = parent.code_path.read_text(encoding="utf-8")
        diff_lines = list(
            difflib.unified_diff(
                parent_code.splitlines(),
                child_lines,
                fromfile=parent.name,
                tofile=child_candidate.name,
                lineterm="",
            )
        )
        sections.append(
            {
                "title": f"Parent {index}",
                "parent": parent,
                "diff_text": "\n".join(diff_lines) if diff_lines else "No textual differences found.",
                "error": None,
            }
        )
    return sections


def _recent_evaluation_jobs(
    *,
    env_id: str,
    run_id: str,
    run: Any | None,
    job_manager: DashboardJobManager,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for snapshot in job_manager.list_snapshots():
        metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
        if metadata.get("env_id") != env_id:
            continue
        if not _job_matches_evaluation_run(snapshot, run_id=run_id, run=run):
            continue
        action = metadata.get("action")
        jobs.append(
            {
                **_snapshot_to_payload(snapshot),
                "evaluation_origin": (
                    "workflow"
                    if action in {"supervised-learn-and-evaluate", AUTOMATIC_LAUNCH_ACTION, UNSUPERVISED_LAUNCH_ACTION}
                    else "evaluation"
                ),
            }
        )
    jobs.sort(
        key=lambda item: (
            item.get("started_at") or "",
            item.get("created_at") or "",
            item.get("job_id") or "",
        ),
        reverse=True,
    )
    return jobs


def _recent_library_evaluation_jobs(
    *,
    env_id: str,
    run_id: str,
    job_manager: DashboardJobManager,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for snapshot in job_manager.list_snapshots():
        metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
        if metadata.get("action") != "evaluate-library-case":
            continue
        if metadata.get("env_id") != env_id or metadata.get("run_id") != run_id:
            continue
        jobs.append(_snapshot_to_payload(snapshot))
    jobs.sort(
        key=lambda item: (
            item.get("started_at") or "",
            item.get("created_at") or "",
            item.get("job_id") or "",
        ),
        reverse=True,
    )
    return jobs


def _normalize_evaluation_statuses(
    *,
    run: Any,
    recent_evaluation_jobs: list[dict[str, Any]],
) -> tuple[Any, list[dict[str, Any]]]:
    evaluation_runs = list(getattr(run, "evaluation_runs", []) or [])
    if not evaluation_runs:
        return run, recent_evaluation_jobs

    direct_job_status_by_eval_id: dict[str, str] = {}
    latest_workflow_job_status: str | None = None
    latest_workflow_job_sort_key: tuple[str, str, str] | None = None
    for job in recent_evaluation_jobs:
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        evaluation_run_id = metadata.get("evaluation_run_id")
        job_status = job.get("status")
        if isinstance(evaluation_run_id, str) and isinstance(job_status, str):
            direct_job_status_by_eval_id[evaluation_run_id] = (
                "completed" if job_status == "succeeded" else job_status
            )
        if job.get("evaluation_origin") == "workflow" and isinstance(job_status, str):
            sort_key = (
                job.get("started_at") or "",
                job.get("created_at") or "",
                job.get("job_id") or "",
            )
            if latest_workflow_job_sort_key is None or sort_key > latest_workflow_job_sort_key:
                latest_workflow_job_sort_key = sort_key
                latest_workflow_job_status = (
                    "completed" if job_status == "succeeded" else job_status
                )

    incomplete_runs = [
        item for item in evaluation_runs if item.summary_path is None and item.results_path is None
    ]
    latest_incomplete_run_id = max(
        (item.run_id for item in incomplete_runs),
        default=None,
    )

    normalized_runs = []
    for item in evaluation_runs:
        status = item.status
        if item.summary_path is None and item.results_path is None:
            if item.run_id in direct_job_status_by_eval_id:
                status = direct_job_status_by_eval_id[item.run_id]
            elif item.run_id == latest_incomplete_run_id and latest_workflow_job_status is not None:
                status = latest_workflow_job_status
            else:
                status = "failed"
        normalized_runs.append(replace(item, status=status))

    selected_ids = {item.run_id for item in normalized_runs}
    normalized_jobs = []
    for job in recent_evaluation_jobs:
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        evaluation_run_id = metadata.get("evaluation_run_id")
        if (
            isinstance(evaluation_run_id, str)
            and evaluation_run_id in selected_ids
            and isinstance(job.get("status"), str)
        ):
            normalized_jobs.append(
                {
                    **job,
                    "status": "completed" if job["status"] == "succeeded" else job["status"],
                }
            )
        else:
            normalized_jobs.append(job)

    return replace(run, evaluation_runs=normalized_runs), normalized_jobs


def _job_matches_evaluation_run(
    snapshot: DashboardJobSnapshot,
    *,
    run_id: str,
    run: Any | None,
) -> bool:
    metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
    action = metadata.get("action")
    if action == "evaluate-deepphy":
        return metadata.get("run_id") == run_id or (
            run is not None and metadata.get("pattern_run_dir") == str(run.root)
        )
    if action not in {"supervised-learn-and-evaluate", AUTOMATIC_LAUNCH_ACTION, UNSUPERVISED_LAUNCH_ACTION}:
        return False
    run_dir = metadata.get("run_dir")
    if run_dir is not None and run is not None and run_dir != str(run.root):
        return False
    if run_dir is None and metadata.get("run_id") not in {None, run_id}:
        return False
    return _job_recent_output_indicates_evaluation(snapshot)


def _remove_jobs_for_evaluation(
    *,
    env_id: str,
    run_id: str,
    evaluation_run_id: str,
    job_manager: DashboardJobManager,
) -> None:
    matching_job_ids: list[str] = []
    for snapshot in job_manager.list_snapshots():
        metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
        if metadata.get("env_id") != env_id:
            continue
        if metadata.get("run_id") not in {None, run_id} and metadata.get("pattern_run_dir") is None:
            continue
        if metadata.get("evaluation_run_id") != evaluation_run_id:
            continue
        matching_job_ids.append(snapshot.job_id)
    for job_id in matching_job_ids:
        try:
            snapshot = job_manager.get_snapshot(job_id)
        except KeyError:
            continue
        if snapshot.status in {"queued", "running", "paused"}:
            try:
                job_manager.terminate_and_remove(job_id)
            except KeyError:
                continue
        else:
            try:
                job_manager.clear(job_id)
            except (KeyError, ValueError):
                try:
                    job_manager.terminate_and_remove(job_id)
                except KeyError:
                    continue


def _job_recent_output_indicates_evaluation(snapshot: DashboardJobSnapshot) -> bool:
    for line in snapshot.recent_output:
        if "dashboard.workflows.supervised.evaluation_start" in line:
            return True
        if "dashboard.workflows.child.start" in line and (
            "deepphy_eval.evaluate_deepphy" in line
            or "cli,evaluate-deepphy" in line
        ):
            return True
    return False


def _default_evaluation_values(*, layout: ArtifactLayout) -> dict[str, Any]:
    return {
        "benchmark_preset": "full",
        "prompt_format": "WM",
        "capture_visuals": True,
        "workers": "1",
        **_load_dashboard_llm_defaults(layout),
    }


def _default_image_only_evaluation_values(*, layout: ArtifactLayout) -> dict[str, Any]:
    return {
        "benchmark_preset": "quick-eval",
        "model": "",
        "workers": "1",
        **_load_dashboard_llm_defaults(layout),
    }


def _default_library_evaluation_values() -> dict[str, Any]:
    return {
        "template_name": DEFAULT_LIBRARY_HOLDOUT_EVALUATION_TEMPLATE,
        "workers": "1",
        "model": "",
    }


def _materialize_filtered_evaluation_library(
    *,
    run: Any,
    evaluation_run_id: str,
    enabled_labels: list[str],
    available_labels: list[str],
) -> Path:
    source_library_path = run.library_path if run.library_path is not None else run.root
    source_library = load_pattern_library(source_library_path)
    enabled_set = set(enabled_labels)
    filtered_detectors = [
        detector
        for detector in source_library.detectors
        if detector.label in enabled_set
    ]
    if not filtered_detectors:
        raise ValueError("At least one label must remain enabled for evaluation.")
    eval_root = run.root / "deepphy_eval" / evaluation_run_id
    eval_root.mkdir(parents=True, exist_ok=True)
    filtered_library = source_library.model_copy(
        update={
            "library_id": f"{source_library.library_id}-eval-{evaluation_run_id}",
            "detectors": filtered_detectors,
            "metadata": {
                **source_library.metadata,
                "source_library_id": source_library.library_id,
                "source_run_id": run.run_id,
                "evaluation_run_id": evaluation_run_id,
                "enabled_labels": list(enabled_labels),
                "disabled_labels": [
                    label for label in available_labels if label not in enabled_set
                ],
            },
        }
    )
    filtered_library_path = eval_root / "pattern_library.filtered.json"
    save_pattern_library(filtered_library, filtered_library_path)
    selection_path = eval_root / "label_selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "source": "manual-dashboard-selection",
                "available_labels": available_labels,
                "enabled_labels": enabled_labels,
                "disabled_labels": [
                    label for label in available_labels if label not in enabled_set
                ],
                "source_library_path": str(source_library_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return filtered_library_path


def _path_for_form(path: Path, *, layout: ArtifactLayout) -> str:
    try:
        return str(path.relative_to(layout.project_root))
    except ValueError:
        return str(path)


def _resolve_dashboard_path(value: str, *, layout: ArtifactLayout) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return layout.project_root / path


def _validate_path_segment(value: str, *, label: str) -> str:
    if value in {"", ".", ".."} or "/" in value or "\\" in value:
        raise HTTPException(status_code=404, detail=f"Invalid {label}: {value}")
    return value


def _job_path_candidates(snapshot: DashboardJobSnapshot) -> list[Path]:
    candidates: list[Path] = []
    metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
    for key in ("run_dir", "pattern_run_dir", "source_supervised_run_dir", "evaluation_run_dir"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            candidates.append(Path(value))
    request_payload = metadata.get("request")
    if isinstance(request_payload, dict):
        for key in ("output_dir",):
            value = request_payload.get(key)
            if isinstance(value, str) and value:
                candidates.append(Path(value))
    return candidates


def _resolve_job_path(path: Path, *, layout: ArtifactLayout) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (layout.project_root / path).resolve()


def _active_job_conflict(
    target: Path,
    *,
    layout: ArtifactLayout,
    job_manager: DashboardJobManager,
) -> DashboardJobSnapshot | None:
    resolved_target = target.resolve()
    for snapshot in job_manager.list_snapshots():
        if snapshot.status not in {"queued", "running", "paused"}:
            continue
        for candidate in _job_path_candidates(snapshot):
            resolved_candidate = _resolve_job_path(candidate, layout=layout)
            if (
                resolved_candidate == resolved_target
                or resolved_candidate.is_relative_to(resolved_target)
            ):
                return snapshot
    return None


def _delete_tree(
    target: Path,
    *,
    layout: ArtifactLayout,
    job_manager: DashboardJobManager,
) -> None:
    while True:
        conflict = _active_job_conflict(
            target,
            layout=layout,
            job_manager=job_manager,
        )
        if conflict is None:
            break
        try:
            job_manager.terminate_and_remove(conflict.job_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Failed to stop dashboard job {conflict.job_id} before deleting "
                    f"{target.name}: {exc}"
                ),
            ) from exc
    if not target.is_dir():
        raise HTTPException(status_code=404, detail=f"Path not found: {target}")
    shutil.rmtree(target)


def _load_dashboard_llm_defaults(layout: ArtifactLayout) -> dict[str, str]:
    llm = load_shared_llm_config(layout.config_root / "LLM.yaml")
    return {
        "model": "",
        "ollama_base_url": llm.base_url,
        "model_timeout": str(llm.timeout),
    }


def _normalize_labels_text(value: str) -> tuple[str, list[str]]:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    labels: list[str] = []
    output_lines: list[str] = []
    for raw_line in normalized.split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        labels.append(stripped)
        output_lines.append(stripped)
    if not labels:
        raise ValueError("target labels must include at least one non-empty label")
    return "\n".join(output_lines) + "\n", labels


def _default_dataset_launch_values(
    env_id: str, *, layout: ArtifactLayout
) -> dict[str, Any]:
    config_path = layout.config_root / "datasets" / "default.yaml"
    dataset_name = "default_simulator_trace_dataset"
    train_count = "1000"
    validation_count = "100"
    test_count = "100"
    train_offset = "0"
    validation_offset = "500000"
    test_offset = "1000000"
    if config_path.is_file():
        try:
            loaded_config = load_dataset_build_config(config_path)
        except Exception:
            loaded_config = None
        if loaded_config is not None and loaded_config.dataset_name:
            dataset_name = loaded_config.dataset_name
            split_map = {split.name: split for split in loaded_config.splits}
            train = split_map.get("train")
            validation = split_map.get("validation")
            test = split_map.get("test")
            if train is not None:
                train_count = str(train.sample_count)
                train_offset = str(train.sample_offset)
            if validation is not None:
                validation_count = str(validation.sample_count)
                validation_offset = str(validation.sample_offset)
            if test is not None:
                test_count = str(test.sample_count)
                test_offset = str(test.sample_offset)
    return {
        "dataset_name": dataset_name,
        "train_count": train_count,
        "validation_count": validation_count,
        "test_count": test_count,
        "train_offset": train_offset,
        "validation_offset": validation_offset,
        "test_offset": test_offset,
        "force": False,
    }


def _pattern_learning_dataset_options(summary: Any) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    for dataset in summary.datasets:
        if dataset.manifest_path is None:
            continue
        options.append(
            {
                "dataset_name": dataset.dataset_name,
                "dataset_manifest": str(dataset.manifest_path),
            }
        )
    return options


def _default_pattern_launch_values(summary: Any) -> dict[str, Any]:
    dataset_options = _pattern_learning_dataset_options(summary)
    template_name = DEFAULT_FUNSEARCH_TEMPLATE
    template_values = _funsearch_template_values(template_name)
    return {
        "dataset_name": dataset_options[0]["dataset_name"] if dataset_options else "",
        "funsearch_template": template_name,
        "iteration_count": template_values["iteration_count"],
        "workers": "1",
        "model": "",
        "seed": "",
        "population_size": template_values["population_size"],
        "elite_fraction": template_values["elite_fraction"],
        "children_per_elite": template_values["children_per_elite"],
        "crossover_rate": "",
    }


def _supervised_source_options(
    summary: Any,
    *,
    layout: ArtifactLayout,
    job_snapshots: list[DashboardJobSnapshot],
) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    seen_values: set[str] = set()
    for run in getattr(summary, "supervised_libraries", []):
        run_root = getattr(run, "root", None)
        run_id = getattr(run, "run_id", None)
        if not isinstance(run_root, Path) or not isinstance(run_id, str):
            continue
        value = _path_for_form(run_root, layout=layout)
        if value in seen_values:
            continue
        seen_values.add(value)
        options.append(
            {
                "label": run_id,
                "value": value,
                "status": "saved",
            }
        )
    for snapshot in sorted(
        job_snapshots,
        key=lambda item: (item.created_at, item.job_id),
    ):
        if snapshot.metadata.get("workflow_kind") != "supervised":
            continue
        if snapshot.metadata.get("env_id") != getattr(summary, "env_id", None):
            continue
        if snapshot.status not in {"queued", "running", "paused"}:
            continue
        run_dir_value = snapshot.metadata.get("run_dir")
        if not isinstance(run_dir_value, str) or not run_dir_value.strip():
            continue
        run_root = Path(run_dir_value)
        value = _path_for_form(run_root, layout=layout)
        if value in seen_values:
            continue
        seen_values.add(value)
        options.append(
            {
                "label": f"{run_root.name} ({snapshot.status})",
                "value": value,
                "status": snapshot.status,
            }
        )
    return options


def _default_unsupervised_launch_values(
    summary: Any,
    *,
    layout: ArtifactLayout,
    job_snapshots: list[DashboardJobSnapshot] | None = None,
) -> dict[str, Any]:
    options = _supervised_source_options(
        summary,
        layout=layout,
        job_snapshots=job_snapshots or [],
    )
    template_name = DEFAULT_FUNSEARCH_TEMPLATE
    template_values = _funsearch_template_values(template_name)
    return {
        "source_supervised_run_dir": options[0]["value"] if options else "",
        "source_evaluation_run_id": (
            AUTO_SOURCE_EVALUATION_RUN_ID if options else ""
        ),
        "reasoning_trace_count": "6",
        "proposed_label_count": "10",
        "limit": "",
        "funsearch_template": template_name,
        "iteration_count": template_values["iteration_count"],
        "workers": "1",
        "model": "",
        "seed": "",
        "population_size": template_values["population_size"],
        "elite_fraction": template_values["elite_fraction"],
        "children_per_elite": template_values["children_per_elite"],
        "crossover_rate": "",
        "run_timestamp": "",
    }


def _default_true_unsupervised_launch_values(summary: Any) -> dict[str, Any]:
    dataset_options = _pattern_learning_dataset_options(summary)
    template_name = DEFAULT_FUNSEARCH_TEMPLATE
    template_values = _funsearch_template_values(template_name)
    return {
        "dataset_name": dataset_options[0]["dataset_name"] if dataset_options else "",
        "trace_count": "10",
        "labels_per_trace": "10",
        "final_label_count": "10",
        "funsearch_template": template_name,
        "iteration_count": template_values["iteration_count"],
        "workers": "1",
        "model": "",
        "seed": "",
        "population_size": template_values["population_size"],
        "elite_fraction": template_values["elite_fraction"],
        "children_per_elite": template_values["children_per_elite"],
        "crossover_rate": "",
        "run_timestamp": "",
    }


def _pooltool_reward_program_source_options(
    summary: Any,
    *,
    layout: ArtifactLayout,
) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    for kind, runs in (
        ("supervised", getattr(summary, "supervised_libraries", [])),
        ("automatic", getattr(summary, "automatic_libraries", [])),
        ("unsupervised", getattr(summary, "unsupervised_libraries", [])),
    ):
        for run in runs:
            run_root = getattr(run, "root", None)
            run_id = getattr(run, "run_id", None)
            if not isinstance(run_root, Path) or not isinstance(run_id, str):
                continue
            library_path = run_root / "pattern_library.json"
            if not library_path.is_file():
                continue
            options.append(
                {
                    "label": f"{run_id} ({kind})",
                    "value": _path_for_form(run_root, layout=layout),
                    "run_id": run_id,
                    "kind": kind,
                }
            )
    return options


def _default_pooltool_reward_optimization_values(
    summary: Any,
    *,
    layout: ArtifactLayout,
) -> dict[str, Any]:
    source_options = _pooltool_reward_program_source_options(
        summary,
        layout=layout,
    )
    return {
        "source_run_dir": source_options[0]["value"] if source_options else "",
        "goal_text": "",
        "dsl_program": "",
        "model": "",
        "scene_count": "10",
        "seed_count": "5",
        "budgets": "10,50,100,250,500,1000",
        "rack_jitter": "0.12",
        "run_id": "",
    }


def _default_pooltool_reward_value_model_values(
    summary: Any,
    *,
    layout: ArtifactLayout,
) -> dict[str, Any]:
    source_options = _pooltool_reward_program_source_options(
        summary,
        layout=layout,
    )
    return {
        "source_run_dir": source_options[0]["value"] if source_options else "",
        "goal_text": "",
        "dsl_program": "",
        "model": "",
        "train_scene_count": "64",
        "train_actions_per_scene": "32",
        "validation_scene_count": "12",
        "validation_candidates_per_scene": "128",
        "epochs": "50",
        "batch_size": "64",
        "learning_rate": "0.001",
        "image_size": "256",
        "rack_jitter": "0.12",
        "run_id": "",
    }


def _supervised_evaluation_options(summary: Any) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    options.append(
        {
            "run_id": "",
            "value": AUTO_SOURCE_EVALUATION_RUN_ID,
            "label": "Auto from selected source run",
        }
    )
    for run in getattr(summary, "supervised_libraries", []):
        for evaluation in getattr(run, "evaluation_runs", []):
            ratio = evaluation.success_ratio
            ratio_text = (
                f"{ratio * 100:.1f}%"
                if ratio is not None
                else f"{evaluation.success_count}/{evaluation.task_count}"
            )
            options.append(
                {
                    "run_id": run.run_id,
                    "value": evaluation.run_id,
                    "label": f"{run.run_id} / {evaluation.run_id} ({ratio_text})",
                }
            )
    return options


def _dataset_jobs_for_env(
    snapshots: list[DashboardJobSnapshot],
    *,
    env_id: str,
    dataset_name: str | None = None,
) -> list[DashboardJobSnapshot]:
    return [
        snapshot
        for snapshot in snapshots
        if snapshot.metadata.get("action") == "build-dataset"
        and snapshot.metadata.get("env_id") == env_id
        and (
            dataset_name is None
            or snapshot.metadata.get("dataset_name") == dataset_name
        )
    ]


def _active_job_payloads(
    snapshots: list[DashboardJobSnapshot], *, layout: ArtifactLayout
) -> list[dict[str, Any]]:
    return [
        _job_payload(snapshot, layout=layout)
        for snapshot in snapshots
        if snapshot.status in {"queued", "running"}
    ]


def _progress_summary_from_run_detail(
    *,
    env_id: str,
    kind: Literal["supervised", "automatic", "unsupervised"],
    run_detail: Any,
    layout: ArtifactLayout,
) -> dict[str, Any]:
    request_payload = (
        run_detail.raw_run_payload.get("request")
        if isinstance(run_detail.raw_run_payload, dict)
        else {}
    )
    ordered_labels, completed_indices = _dashboard_completed_label_indices(
        env_id=env_id,
        kind=kind,
        request_payload=request_payload if isinstance(request_payload, dict) else {},
        run_payload=run_detail.raw_run_payload if isinstance(run_detail.raw_run_payload, dict) else {},
        run_detail=run_detail,
        layout=layout,
    )
    return {
        "completed_labels": len(completed_indices),
        "total_labels": len(ordered_labels),
        "workers": [],
        "active_label": None,
        "active_label_index": None,
        "candidate_evaluations": None,
        "latest_score": None,
        "latest_validation_score": None,
        "stage": None,
    }


def _active_run_payloads_from_disk(
    *,
    summaries: list[Any],
    active_jobs: list[dict[str, Any]],
    layout: ArtifactLayout,
) -> list[dict[str, Any]]:
    active_run_ids = {
        job.get("learning_progress", {}).get("run_id")
        for job in active_jobs
        if isinstance(job.get("learning_progress"), dict)
    }
    payloads: list[dict[str, Any]] = []
    for summary in summaries:
        env_id = summary.env_id
        for run in summary.supervised_libraries:
            status = (
                run.raw_run_payload.get("status")
                if isinstance(run.raw_run_payload, dict)
                else None
            )
            if status != "running" or run.run_id in active_run_ids:
                continue
            if not _run_has_recent_activity(run.root):
                continue
            run_detail = load_pattern_run_detail_from_root(env_id, run.root)
            progress_summary = _progress_summary_from_run_detail(
                env_id=env_id,
                kind="supervised",
                run_detail=run_detail,
                layout=layout,
            )
            payloads.append(
                {
                    "job_id": f"run-{run.run_id}",
                    "name": "supervised-learn-and-evaluate",
                    "status": "running",
                    "display_name": run.run_id,
                    "metadata": {
                        "env_id": env_id,
                        "action": "supervised-learn-and-evaluate",
                        "supports_pause": False,
                    },
                    "learning_progress": {
                        "kind": "supervised",
                        "run_id": run.run_id,
                        "run_url": f"/runs/{env_id}/{run.run_id}",
                        "refresh_token": _run_refresh_token(run.root),
                        "settings": _learning_settings(run_payload=run_detail.raw_run_payload),
                        "progress_summary": progress_summary,
                    },
                }
            )
        for run in summary.automatic_libraries:
            status = (
                run.raw_run_payload.get("status")
                if isinstance(run.raw_run_payload, dict)
                else None
            )
            if status != "running" or run.run_id in active_run_ids:
                continue
            if not _run_has_recent_activity(run.root):
                continue
            run_detail = load_pattern_run_detail_from_root(env_id, run.root)
            progress_summary = _progress_summary_from_run_detail(
                env_id=env_id,
                kind=AUTOMATIC_WORKFLOW_KIND,
                run_detail=run_detail,
                layout=layout,
            )
            payloads.append(
                {
                    "job_id": f"run-{run.run_id}",
                    "name": AUTOMATIC_LAUNCH_ACTION,
                    "status": "running",
                    "display_name": run.run_id,
                    "metadata": {
                        "env_id": env_id,
                        "action": AUTOMATIC_LAUNCH_ACTION,
                        "supports_pause": False,
                    },
                    "learning_progress": {
                        "kind": AUTOMATIC_WORKFLOW_KIND,
                        "run_id": run.run_id,
                        "run_url": _automatic_run_url(env_id, run.run_id),
                        "refresh_token": _run_refresh_token(run.root),
                        "settings": _learning_settings(run_payload=run_detail.raw_run_payload),
                        "progress_summary": progress_summary,
                    },
                }
            )
        for run in summary.unsupervised_libraries:
            status = (
                run.raw_run_payload.get("status")
                if isinstance(run.raw_run_payload, dict)
                else None
            )
            if status != "running" or run.run_id in active_run_ids:
                continue
            if not _run_has_recent_activity(run.root):
                continue
            run_detail = load_pattern_run_detail_from_root(env_id, run.root)
            progress_summary = _progress_summary_from_run_detail(
                env_id=env_id,
                kind=UNSUPERVISED_WORKFLOW_KIND,
                run_detail=run_detail,
                layout=layout,
            )
            payloads.append(
                {
                    "job_id": f"run-{run.run_id}",
                    "name": UNSUPERVISED_LAUNCH_ACTION,
                    "status": "running",
                    "display_name": run.run_id,
                    "metadata": {
                        "env_id": env_id,
                        "action": UNSUPERVISED_LAUNCH_ACTION,
                        "supports_pause": False,
                    },
                    "learning_progress": {
                        "kind": UNSUPERVISED_WORKFLOW_KIND,
                        "run_id": run.run_id,
                        "run_url": _unsupervised_run_url(env_id, run.run_id),
                        "refresh_token": _run_refresh_token(run.root),
                        "settings": _learning_settings(run_payload=run_detail.raw_run_payload),
                        "progress_summary": progress_summary,
                    },
                }
            )
    return payloads


def create_dashboard_app(
    *,
    layout: ArtifactLayout | None = None,
    job_manager: DashboardJobManager | None = None,
) -> FastAPI:
    resolved_layout = layout or get_artifact_layout()
    root = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(root / "templates"))
    templates.env.filters["display_time"] = _format_display_time
    templates.env.filters["duration"] = _format_duration_seconds
    templates.env.filters["millions"] = _format_millions
    templates.env.globals["display_time"] = _format_display_time
    templates.env.globals["format_duration"] = _format_duration_seconds
    templates.env.globals["format_millions"] = _format_millions
    templates.env.globals["display_learning_stage"] = _display_learning_stage
    templates.env.globals["render_trajectory_chart"] = _render_trajectory_chart
    dashboard_job_manager = job_manager or DashboardJobManager(layout=resolved_layout)
    app_queue_settings = save_queue_settings_state(
        resolved_layout,
        load_queue_settings_state(resolved_layout),
    )
    dashboard_job_manager.set_max_concurrent_queue_jobs(
        app_queue_settings["max_concurrent_jobs"]
    )
    dashboard_job_manager.set_enforce_single_pooltool_queue_job(
        app_queue_settings["enforce_single_pooltool_job"]
    )

    app = FastAPI(title="Pattern Learning Dashboard")
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)
    app.state.queue_settings = app_queue_settings
    app.mount(
        "/dashboard/static",
        StaticFiles(directory=str(root / "static")),
        name="dashboard-static",
    )

    @app.middleware("http")
    async def disable_dashboard_caching(request: Request, call_next):
        response = await call_next(request)
        if not request.url.path.startswith("/dashboard/static/"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    def _nav_links(
        *, active: str | None = None, env_id: str | None = None
    ) -> list[dict[str, object]]:
        links: list[dict[str, object]] = [
            {"href": "/", "label": "Overview", "active": active == "home"},
            {
                "href": "/pooltool/reward-programs",
                "label": "PoolTool rewards",
                "active": active == "reward-programs",
            },
        ]
        for supported_env_id in SUPPORTED_ENVS:
            links.append(
                {
                    "href": f"/env/{supported_env_id}",
                    "label": supported_env_id,
                    "active": env_id == supported_env_id and active in {"env", "libraries", "compare"},
                }
            )
        return links

    def _env_job_snapshots(env_id: str) -> list[DashboardJobSnapshot]:
        return [
            snapshot
            for snapshot in dashboard_job_manager.list_snapshots()
            if snapshot.metadata.get("env_id") == env_id
        ]

    def _env_job_payloads(env_id: str) -> list[dict[str, Any]]:
        return [
            _job_payload(snapshot, layout=resolved_layout)
            for snapshot in _env_job_snapshots(env_id)
        ]

    def _job_recent_run_item(snapshot: DashboardJobSnapshot) -> dict[str, Any] | None:
        metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
        if not bool(metadata.get("queue_requested")):
            return None
        action = metadata.get("action")
        if action not in {"supervised-learn-and-evaluate", AUTOMATIC_LAUNCH_ACTION, UNSUPERVISED_LAUNCH_ACTION, "evaluate-deepphy"}:
            return None
        if snapshot.status not in {"failed", "succeeded"}:
            return None
        env_id = metadata.get("env_id")
        if not isinstance(env_id, str) or not env_id.strip():
            env_id = "unknown"
        run_dir_raw = metadata.get("run_dir")
        evaluation_run_id = metadata.get("evaluation_run_id")
        run_id = metadata.get("run_id")
        if isinstance(run_dir_raw, str) and run_dir_raw.strip():
            display_name = Path(run_dir_raw).name
        elif isinstance(run_id, str) and run_id.strip():
            display_name = run_id
        elif isinstance(evaluation_run_id, str) and evaluation_run_id.strip():
            display_name = evaluation_run_id
        else:
            display_name = snapshot.job_id[:8]
        finished_at = snapshot.finished_at or snapshot.started_at or snapshot.created_at
        sort_key = time.time()
        if isinstance(finished_at, str) and finished_at.strip():
            try:
                sort_key = datetime.strptime(
                    finished_at, "%Y-%m-%dT%H:%M:%SZ"
                ).timestamp()
            except ValueError:
                pass
        if _is_automatic_action(action):
            kind = AUTOMATIC_WORKFLOW_KIND
        elif _is_unsupervised_action(action):
            kind = UNSUPERVISED_WORKFLOW_KIND
        else:
            kind = "evaluation" if action == "evaluate-deepphy" else "supervised"
        return {
            "env_id": env_id,
            "job": snapshot,
            "display_name": display_name,
            "sort_key": sort_key,
            "kind": kind,
            "display_status": snapshot.status,
        }

    def _recent_run_identity(item: dict[str, Any]) -> tuple[str, str, str] | None:
        env_id = item.get("env_id")
        kind = item.get("kind")
        if not isinstance(env_id, str) or not env_id.strip():
            return None
        if not isinstance(kind, str) or not kind.strip():
            return None
        run = item.get("run")
        if run is not None:
            run_id = getattr(run, "run_id", None)
            if isinstance(run_id, str) and run_id.strip():
                return (kind, env_id, run_id)
        job = item.get("job")
        if isinstance(job, DashboardJobSnapshot):
            metadata = job.metadata if isinstance(job.metadata, dict) else {}
            if kind in {"supervised", AUTOMATIC_WORKFLOW_KIND, UNSUPERVISED_WORKFLOW_KIND}:
                run_dir_raw = metadata.get("run_dir")
                if isinstance(run_dir_raw, str) and run_dir_raw.strip():
                    return (kind, env_id, Path(run_dir_raw).name)
            if kind == "evaluation":
                evaluation_run_id = metadata.get("evaluation_run_id")
                if isinstance(evaluation_run_id, str) and evaluation_run_id.strip():
                    return (kind, env_id, evaluation_run_id)
        return None

    def _recent_runs(
        summaries: list[Any],
        *,
        job_snapshots: list[DashboardJobSnapshot] | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        archive_statuses = {"completed", "succeeded", "failed", "interrupted", "paused"}
        items: list[dict[str, Any]] = []
        seen_identities: set[tuple[str, str, str]] = set()
        for summary in summaries:
            env_id = summary.env_id
            for run in summary.supervised_libraries:
                latest_eval = run.evaluation_runs[0] if run.evaluation_runs else None
                try:
                    sort_key = run.root.stat().st_mtime
                except OSError:
                    sort_key = 0.0
                raw_status = (
                    run.raw_run_payload.get("status")
                    if isinstance(run.raw_run_payload, dict)
                    else None
                )
                display_status = raw_status
                if raw_status == "running" and not _run_has_recent_activity(run.root):
                    display_status = "interrupted"
                if display_status not in archive_statuses:
                    continue
                item = {
                    "env_id": env_id,
                    "run": run,
                    "latest_eval": latest_eval,
                    "sort_key": sort_key,
                    "kind": "supervised",
                    "display_status": display_status,
                }
                identity = _recent_run_identity(item)
                if identity is not None:
                    seen_identities.add(identity)
                items.append(item)
            for run in summary.automatic_libraries:
                latest_eval = run.evaluation_runs[0] if run.evaluation_runs else None
                try:
                    sort_key = run.root.stat().st_mtime
                except OSError:
                    sort_key = 0.0
                raw_status = (
                    run.raw_run_payload.get("status")
                    if isinstance(run.raw_run_payload, dict)
                    else None
                )
                display_status = raw_status
                if raw_status == "running" and not _run_has_recent_activity(run.root):
                    display_status = "interrupted"
                if display_status not in archive_statuses:
                    continue
                item = {
                    "env_id": env_id,
                    "run": run,
                    "latest_eval": latest_eval,
                    "sort_key": sort_key,
                    "kind": AUTOMATIC_WORKFLOW_KIND,
                    "display_status": display_status,
                }
                identity = _recent_run_identity(item)
                if identity is not None:
                    seen_identities.add(identity)
                items.append(item)
            for run in summary.unsupervised_libraries:
                latest_eval = run.evaluation_runs[0] if run.evaluation_runs else None
                try:
                    sort_key = run.root.stat().st_mtime
                except OSError:
                    sort_key = 0.0
                raw_status = (
                    run.raw_run_payload.get("status")
                    if isinstance(run.raw_run_payload, dict)
                    else None
                )
                display_status = raw_status
                if raw_status == "running" and not _run_has_recent_activity(run.root):
                    display_status = "interrupted"
                if display_status not in archive_statuses:
                    continue
                item = {
                    "env_id": env_id,
                    "run": run,
                    "latest_eval": latest_eval,
                    "sort_key": sort_key,
                    "kind": UNSUPERVISED_WORKFLOW_KIND,
                    "display_status": display_status,
                }
                identity = _recent_run_identity(item)
                if identity is not None:
                    seen_identities.add(identity)
                items.append(item)
        for snapshot in job_snapshots or []:
            item = _job_recent_run_item(snapshot)
            if item is not None:
                identity = _recent_run_identity(item)
                if identity is not None and identity in seen_identities:
                    continue
                items.append(item)
        return sorted(items, key=lambda item: item["sort_key"], reverse=True)[:limit]

    def _recent_runs_json_payload(
        summaries: list[Any],
        *,
        job_snapshots: list[DashboardJobSnapshot] | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for item in _recent_runs(summaries, job_snapshots=job_snapshots, limit=limit):
            run_status = item.get("display_status")
            run_status_filter = (
                "finished"
                if run_status in {"completed", "succeeded"}
                else (run_status or "unknown")
            )
            if item.get("job"):
                job = item["job"]
                metadata = job.metadata if isinstance(job.metadata, dict) else {}
                payload.append(
                    {
                        "env_id": item.get("env_id"),
                        "kind": item.get("kind"),
                        "display_name": item.get("display_name"),
                        "route": f"/jobs/{job.job_id}",
                        "status": run_status,
                        "status_filter": run_status_filter,
                        "subtitle": f"{item.get('env_id')} | {item.get('kind')} job",
                        "meta": " | ".join(
                            part
                            for part in [
                                job.name,
                                str(metadata.get("queue_wait_reason") or "").strip(),
                            ]
                            if part
                        ),
                        "eval_display": "—",
                    }
                )
                continue
            run = item.get("run")
            if run is None:
                continue
            latest_eval = item.get("latest_eval")
            kind = item.get("kind")
            route = _run_url_for_kind(item.get("env_id"), run.run_id, kind)
            subtitle = f"{item.get('env_id')} | {_display_run_kind(kind)}"
            if kind == "supervised" and run.average_score is not None:
                subtitle += f" | reward {run.average_score:.3f}"
            if kind == "supervised":
                meta = f"{run.label_count} labels | {run.candidate_count} candidates"
            else:
                meta = f"{run.learned_label_count} learned | {len(run.proposed_labels)} proposed"
            eval_display = "—"
            if latest_eval is not None and latest_eval.success_ratio is not None:
                eval_display = f"{latest_eval.success_ratio * 100:.1f}%"
            payload.append(
                {
                    "env_id": item.get("env_id"),
                    "kind": kind,
                    "display_name": run.run_id,
                    "route": route,
                    "status": run_status,
                    "status_filter": run_status_filter,
                    "subtitle": subtitle,
                    "meta": meta,
                    "eval_display": eval_display,
                }
            )
        return payload

    def _build_supervised_compare_payload(
        env_id: str,
        *,
        requested_run_ids: list[str],
    ) -> dict[str, Any]:
        summary = load_environment_summary(env_id, layout=resolved_layout)
        available_runs = summary.supervised_libraries
        runs_by_id = {run.run_id: run for run in available_runs}
        selected_run_ids: list[str] = []
        for run_id in requested_run_ids:
            if run_id in runs_by_id and run_id not in selected_run_ids:
                selected_run_ids.append(run_id)
        if not selected_run_ids:
            selected_run_ids = [run.run_id for run in available_runs[:2]]
        selected_run_ids = selected_run_ids[:4]

        selected_runs: list[dict[str, Any]] = []
        label_maps: list[dict[str, Any]] = []
        all_label_names: set[str] = set()
        for run_id in selected_run_ids:
            detail = load_pattern_run_detail(env_id, run_id, layout=resolved_layout)
            latest_eval = detail.evaluation_runs[0] if detail.evaluation_runs else None
            selected_runs.append(
                {
                    "run_id": run_id,
                    "detail": detail,
                    "latest_eval": latest_eval,
                }
            )
            label_map = {label.name: label for label in detail.label_telemetry}
            label_maps.append(label_map)
            all_label_names.update(label_map.keys())

        label_rows: list[dict[str, Any]] = []
        for label_name in sorted(all_label_names):
            cells: list[dict[str, Any]] = []
            best_validation_run_id: str | None = None
            best_validation_score: float | None = None
            for selected in selected_runs:
                label = next(
                    (
                        item
                        for item in selected["detail"].label_telemetry
                        if item.name == label_name
                    ),
                    None,
                )
                best_candidate = label.best_candidate if label is not None else None
                validation_score = (
                    best_candidate.val_score
                    if best_candidate is not None
                    else None
                )
                if (
                    isinstance(validation_score, (int, float))
                    and (
                        best_validation_score is None
                        or validation_score > best_validation_score
                    )
                ):
                    best_validation_score = float(validation_score)
                    best_validation_run_id = selected["run_id"]
                cells.append(
                    {
                        "run_id": selected["run_id"],
                        "present": label is not None,
                        "candidate_count": label.candidate_count if label is not None else 0,
                        "best_average_score": (
                            best_candidate.average_score
                            if best_candidate is not None
                            else None
                        ),
                        "best_train_score": (
                            best_candidate.train_score
                            if best_candidate is not None
                            else None
                        ),
                        "best_validation_score": validation_score,
                        "best_candidate_name": (
                            best_candidate.name if best_candidate is not None else None
                        ),
                    }
                )
            label_rows.append(
                {
                    "label_name": label_name,
                    "cells": cells,
                    "best_validation_run_id": best_validation_run_id,
                }
            )

        return {
            "summary": summary,
            "available_runs": available_runs,
            "selected_run_ids": selected_run_ids,
            "selected_runs": selected_runs,
            "label_rows": label_rows,
        }

    def _render_environment_page(
        request: Request,
        *,
        env_id: str,
        summary: Any,
        dataset_launch_error: str | None,
        dataset_launch_values: dict[str, Any],
        label_error: str | None,
        label_success: str | None,
        label_values: dict[str, Any],
        launch_error: str | None,
        launch_values: dict[str, Any],
        unsupervised_launch_error: str | None,
        unsupervised_launch_values: dict[str, Any],
        true_unsupervised_launch_error: str | None = None,
        true_unsupervised_launch_values: dict[str, Any] | None = None,
        image_only_evaluation_error: str | None = None,
        image_only_evaluation_values: dict[str, Any] | None = None,
        status_code: int = 200,
    ) -> HTMLResponse:
        env_job_snapshots = _env_job_snapshots(env_id)
        image_only_values = _default_image_only_evaluation_values(
            layout=resolved_layout
        )
        true_unsupervised_values = _default_true_unsupervised_launch_values(summary)
        if true_unsupervised_launch_values is not None:
            true_unsupervised_values.update(true_unsupervised_launch_values)
        if image_only_evaluation_values is not None:
            image_only_values.update(image_only_evaluation_values)
        return templates.TemplateResponse(
            request,
            "environment.html",
            {
                "page_title": f"{env_id} workspace",
                "request": request,
                "summary": summary,
                "dataset_launch_error": dataset_launch_error,
                "dataset_launch_values": dataset_launch_values,
                "recent_dataset_jobs": _dataset_jobs_for_env(
                    env_job_snapshots, env_id=env_id
                ),
                "environment_jobs": _env_job_payloads(env_id),
                "label_error": label_error,
                "label_success": label_success,
                "label_values": label_values,
                "launch_error": launch_error,
                "launch_values": launch_values,
                "unsupervised_launch_error": unsupervised_launch_error,
                "unsupervised_launch_values": unsupervised_launch_values,
                "true_unsupervised_launch_error": true_unsupervised_launch_error,
                "true_unsupervised_launch_values": true_unsupervised_values,
                "image_only_evaluation_error": image_only_evaluation_error,
                "image_only_evaluation_values": image_only_values,
                "funsearch_templates": FUNSEARCH_TEMPLATES,
                "learning_dataset_options": _pattern_learning_dataset_options(summary),
                "supervised_library_options": _supervised_source_options(
                    summary,
                    layout=resolved_layout,
                    job_snapshots=env_job_snapshots,
                ),
                "supervised_evaluation_options": _supervised_evaluation_options(summary),
                "nav_links": _nav_links(active="env", env_id=env_id),
            },
            status_code=status_code,
        )

    def _reward_run_detail_url(run: LocalRewardRunRecord) -> str:
        if run.kind == "optimization":
            return f"/pooltool/reward-programs/optimization/{run.run_id}"
        return f"/pooltool/reward-programs/value-model/{run.run_id}"

    def _render_pooltool_reward_programs_page(
        request: Request,
        *,
        optimization_error: str | None = None,
        optimization_values: dict[str, Any] | None = None,
        value_model_error: str | None = None,
        value_model_values: dict[str, Any] | None = None,
        status_code: int = 200,
    ) -> HTMLResponse:
        summary = load_environment_summary("pooltool", layout=resolved_layout)
        source_options = _pooltool_reward_program_source_options(
            summary,
            layout=resolved_layout,
        )
        optimization_defaults = _default_pooltool_reward_optimization_values(
            summary,
            layout=resolved_layout,
        )
        value_model_defaults = _default_pooltool_reward_value_model_values(
            summary,
            layout=resolved_layout,
        )
        if optimization_values is not None:
            optimization_defaults.update(optimization_values)
        if value_model_values is not None:
            value_model_defaults.update(value_model_values)
        reward_runs = list_local_reward_runs(layout=resolved_layout)
        active_reward_runs = [
            run for run in reward_runs if run.status in {"starting", "running"}
        ]
        optimization_runs = list_pooltool_reward_optimization_experiments(
            layout=resolved_layout
        )
        value_model_runs = list_pooltool_reward_value_model_experiments(
            layout=resolved_layout
        )
        return templates.TemplateResponse(
            request,
            "pooltool_reward_programs.html",
            {
                "page_title": "PoolTool reward programs",
                "request": request,
                "summary": summary,
                "source_options": source_options,
                "optimization_error": optimization_error,
                "optimization_values": optimization_defaults,
                "value_model_error": value_model_error,
                "value_model_values": value_model_defaults,
                "reward_runs": reward_runs,
                "active_reward_runs": active_reward_runs,
                "reward_run_url": _reward_run_detail_url,
                "optimization_runs": optimization_runs,
                "value_model_runs": value_model_runs,
                "nav_links": _nav_links(active="reward-programs"),
            },
            status_code=status_code,
        )

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        summaries = load_dashboard_home(layout=resolved_layout)
        all_jobs = dashboard_job_manager.list_snapshots()
        queue_panel = _queue_panel_payload(
            job_manager=dashboard_job_manager,
            snapshots=all_jobs,
            layout=resolved_layout,
        )
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "page_title": "Pattern Learning Dashboard",
                "request": request,
                "summaries": summaries,
                "all_jobs": all_jobs,
                "active_jobs": (
                    _active_job_payloads(all_jobs, layout=resolved_layout)
                    + _active_run_payloads_from_disk(
                        summaries=summaries,
                        active_jobs=_active_job_payloads(all_jobs, layout=resolved_layout),
                        layout=resolved_layout,
                    )
                ),
                "recent_runs": _recent_runs(summaries, job_snapshots=dashboard_job_manager.list_snapshots()),
                "queued_home_jobs": queue_panel["queued_home_jobs"],
                "queue_settings": queue_panel["queue_settings"],
                "system_resources": _system_resource_snapshot(),
                "dataset_jobs_by_env": {
                    summary.env_id: _dataset_jobs_for_env(all_jobs, env_id=summary.env_id)
                    for summary in summaries
                },
                "nav_links": _nav_links(active="home"),
            },
        )

    @app.get("/pooltool/reward-programs", response_class=HTMLResponse)
    async def pooltool_reward_programs_view(request: Request) -> HTMLResponse:
        return _render_pooltool_reward_programs_page(request)

    @app.post("/pooltool/reward-programs/optimization/launch")
    async def launch_pooltool_reward_optimization(
        request: Request,
    ):
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        llm_defaults = _load_dashboard_llm_defaults(resolved_layout)

        def _first(name: str, default: str = "") -> str:
            values = parsed_form.get(name)
            if not values:
                return default
            return values[0]

        form_values = {
            "source_run_dir": _first("source_run_dir", ""),
            "goal_text": _first("goal_text", ""),
            "dsl_program": _first("dsl_program", ""),
            "model": _first("model", ""),
            "scene_count": _first("scene_count", "10"),
            "seed_count": _first("seed_count", "5"),
            "budgets": _first("budgets", "10,50,100,250,500,1000"),
            "rack_jitter": _first("rack_jitter", "0.12"),
            "run_id": _first("run_id", ""),
        }
        try:
            budgets = [
                int(part.strip())
                for part in form_values["budgets"].split(",")
                if part.strip()
            ]
            action_request = PooltoolRewardOptimizationActionRequest.model_validate(
                {
                    "source_run_dir": form_values["source_run_dir"],
                    "goal_text": form_values["goal_text"],
                    "dsl_program": form_values["dsl_program"],
                    "model": form_values["model"],
                    "scene_count": int(form_values["scene_count"] or "10"),
                    "seed_count": int(form_values["seed_count"] or "5"),
                    "budgets": budgets,
                    "rack_jitter": float(form_values["rack_jitter"] or "0.12"),
                    "run_id": _coerce_optional(form_values["run_id"]),
                    "ollama_base_url": llm_defaults["ollama_base_url"],
                    "model_timeout": llm_defaults["model_timeout"],
                }
            )
        except (ValidationError, ValueError) as exc:
            message = (
                _format_validation_error(exc)
                if isinstance(exc, ValidationError)
                else str(exc)
            )
            return _render_pooltool_reward_programs_page(
                request,
                optimization_error=message,
                optimization_values=form_values,
                status_code=422,
            )
        reward_run_id = action_request.run_id or default_run_id()
        action_request = action_request.model_copy(update={"run_id": reward_run_id})
        try:
            launch_local_reward_run(
                kind="optimization",
                request=action_request,
                layout=resolved_layout,
            )
        except (ValueError, FileNotFoundError) as exc:
            return _render_pooltool_reward_programs_page(
                request,
                optimization_error=str(exc),
                optimization_values=form_values,
                status_code=422,
            )
        return RedirectResponse(
            url=f"/pooltool/reward-programs/optimization/{reward_run_id}",
            status_code=303,
        )

    @app.post("/pooltool/reward-programs/value-model/launch")
    async def launch_pooltool_reward_value_model(
        request: Request,
    ):
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        llm_defaults = _load_dashboard_llm_defaults(resolved_layout)

        def _first(name: str, default: str = "") -> str:
            values = parsed_form.get(name)
            if not values:
                return default
            return values[0]

        form_values = {
            "source_run_dir": _first("source_run_dir", ""),
            "goal_text": _first("goal_text", ""),
            "dsl_program": _first("dsl_program", ""),
            "model": _first("model", ""),
            "train_scene_count": _first("train_scene_count", "64"),
            "train_actions_per_scene": _first("train_actions_per_scene", "32"),
            "validation_scene_count": _first("validation_scene_count", "12"),
            "validation_candidates_per_scene": _first(
                "validation_candidates_per_scene",
                "128",
            ),
            "epochs": _first("epochs", "50"),
            "batch_size": _first("batch_size", "64"),
            "learning_rate": _first("learning_rate", "0.001"),
            "image_size": _first("image_size", "256"),
            "rack_jitter": _first("rack_jitter", "0.12"),
            "run_id": _first("run_id", ""),
        }
        try:
            action_request = PooltoolRewardValueModelActionRequest.model_validate(
                {
                    "source_run_dir": form_values["source_run_dir"],
                    "goal_text": form_values["goal_text"],
                    "dsl_program": form_values["dsl_program"],
                    "model": form_values["model"],
                    "train_scene_count": int(form_values["train_scene_count"] or "64"),
                    "train_actions_per_scene": int(
                        form_values["train_actions_per_scene"] or "32"
                    ),
                    "validation_scene_count": int(
                        form_values["validation_scene_count"] or "12"
                    ),
                    "validation_candidates_per_scene": int(
                        form_values["validation_candidates_per_scene"] or "128"
                    ),
                    "epochs": int(form_values["epochs"] or "50"),
                    "batch_size": int(form_values["batch_size"] or "64"),
                    "learning_rate": float(form_values["learning_rate"] or "0.001"),
                    "image_size": int(form_values["image_size"] or "256"),
                    "rack_jitter": float(form_values["rack_jitter"] or "0.12"),
                    "run_id": _coerce_optional(form_values["run_id"]),
                    "ollama_base_url": llm_defaults["ollama_base_url"],
                    "model_timeout": llm_defaults["model_timeout"],
                }
            )
        except (ValidationError, ValueError) as exc:
            message = (
                _format_validation_error(exc)
                if isinstance(exc, ValidationError)
                else str(exc)
            )
            return _render_pooltool_reward_programs_page(
                request,
                value_model_error=message,
                value_model_values=form_values,
                status_code=422,
            )
        reward_run_id = action_request.run_id or default_run_id()
        action_request = action_request.model_copy(update={"run_id": reward_run_id})
        try:
            launch_local_reward_run(
                kind="value_model",
                request=action_request,
                layout=resolved_layout,
            )
        except (ValueError, FileNotFoundError) as exc:
            return _render_pooltool_reward_programs_page(
                request,
                value_model_error=str(exc),
                value_model_values=form_values,
                status_code=422,
            )
        return RedirectResponse(
            url=f"/pooltool/reward-programs/value-model/{reward_run_id}",
            status_code=303,
        )

    @app.get(
        "/pooltool/reward-programs/optimization/{run_id}",
        response_class=HTMLResponse,
    )
    async def pooltool_reward_optimization_detail(
        request: Request,
        run_id: str,
        job: str | None = None,
    ) -> HTMLResponse:
        _validate_path_segment(run_id, label="reward optimization run id")
        run_root = (
            resolved_layout.data_root
            / "patterns"
            / "pooltool"
            / "reward_program_experiments"
            / "optimization"
            / run_id
        )
        if not run_root.is_dir():
            raise HTTPException(
                status_code=404,
                detail=f"PoolTool reward optimization not found: {run_id}",
            )
        summary = next(
            (
                item
                for item in list_pooltool_reward_optimization_experiments(
                    layout=resolved_layout
                )
                if item.run_id == run_id
            ),
            None,
        )
        run_state = load_local_reward_run(
            kind="optimization",
            run_id=run_id,
            layout=resolved_layout,
        )
        program_text = (
            (run_root / "program.dsl").read_text(encoding="utf-8")
            if (run_root / "program.dsl").is_file()
            else (summary.dsl_program if summary is not None else "")
        )
        synthesis_text = (
            (run_root / "synthesis_response.txt").read_text(encoding="utf-8")
            if (run_root / "synthesis_response.txt").is_file()
            else ""
        )
        return templates.TemplateResponse(
            request,
            "pooltool_reward_optimization_detail.html",
            {
                "page_title": f"PoolTool reward optimization {run_id}",
                "request": request,
                "run_id": run_id,
                "summary": summary,
                "run_state": run_state,
                "program_text": program_text,
                "synthesis_text": synthesis_text,
                "nav_links": _nav_links(active="reward-programs"),
            },
        )

    @app.get(
        "/pooltool/reward-programs/value-model/{run_id}",
        response_class=HTMLResponse,
    )
    async def pooltool_reward_value_model_detail(
        request: Request,
        run_id: str,
        job: str | None = None,
    ) -> HTMLResponse:
        _validate_path_segment(run_id, label="reward value model run id")
        run_root = (
            resolved_layout.data_root
            / "patterns"
            / "pooltool"
            / "reward_program_experiments"
            / "value_models"
            / run_id
        )
        if not run_root.is_dir():
            raise HTTPException(
                status_code=404,
                detail=f"PoolTool reward value model not found: {run_id}",
            )
        summary = next(
            (
                item
                for item in list_pooltool_reward_value_model_experiments(
                    layout=resolved_layout
                )
                if item.run_id == run_id
            ),
            None,
        )
        run_state = load_local_reward_run(
            kind="value_model",
            run_id=run_id,
            layout=resolved_layout,
        )
        program_text = (
            (run_root / "program.dsl").read_text(encoding="utf-8")
            if (run_root / "program.dsl").is_file()
            else (summary.dsl_program if summary is not None else "")
        )
        synthesis_text = (
            (run_root / "synthesis_response.txt").read_text(encoding="utf-8")
            if (run_root / "synthesis_response.txt").is_file()
            else ""
        )
        return templates.TemplateResponse(
            request,
            "pooltool_reward_value_model_detail.html",
            {
                "page_title": f"PoolTool reward value model {run_id}",
                "request": request,
                "run_id": run_id,
                "summary": summary,
                "run_state": run_state,
                "program_text": program_text,
                "synthesis_text": synthesis_text,
                "nav_links": _nav_links(active="reward-programs"),
            },
        )

    def _render_queue_panel_response(request: Request) -> HTMLResponse:
        queue_panel = _queue_panel_payload(
            job_manager=dashboard_job_manager,
            snapshots=dashboard_job_manager.list_snapshots(),
            layout=resolved_layout,
        )
        response = templates.TemplateResponse(
            request,
            "_queue_panel.html",
            {
                "request": request,
                "queued_home_jobs": queue_panel["queued_home_jobs"],
                "queue_settings": queue_panel["queue_settings"],
            },
        )
        response.headers["cache-control"] = "no-store, max-age=0"
        response.headers["pragma"] = "no-cache"
        return response

    @app.get("/queue-panel", response_class=HTMLResponse)
    async def queue_panel_partial(request: Request) -> HTMLResponse:
        return _render_queue_panel_response(request)

    @app.get("/queue-panel-version", response_class=JSONResponse)
    async def queue_panel_version() -> JSONResponse:
        payload = {
            "version": _queue_panel_version(
                job_manager=dashboard_job_manager,
                snapshots=dashboard_job_manager.list_snapshots(),
            )
        }
        response = JSONResponse(payload)
        response.headers["cache-control"] = "no-store, max-age=0"
        response.headers["pragma"] = "no-cache"
        return response

    @app.get("/queue-panel-data", response_class=JSONResponse)
    async def queue_panel_data() -> JSONResponse:
        queue_panel = _queue_panel_payload(
            job_manager=dashboard_job_manager,
            snapshots=dashboard_job_manager.list_snapshots(),
            layout=resolved_layout,
        )
        response = JSONResponse(_queue_panel_json_payload(queue_panel))
        response.headers["cache-control"] = "no-store, max-age=0"
        response.headers["pragma"] = "no-cache"
        return response

    @app.get("/system-resources.json", response_class=JSONResponse)
    async def system_resources_json() -> JSONResponse:
        snapshot = _cached_value(
            ("system-resources-snapshot",),
            ttl_seconds=2.0,
            builder=_system_resource_snapshot,
        )
        response = JSONResponse(snapshot)
        response.headers["cache-control"] = "no-store, max-age=0"
        response.headers["pragma"] = "no-cache"
        return response

    @app.get("/recent-runs", response_class=HTMLResponse)
    async def recent_runs_panel(request: Request) -> HTMLResponse:
        summaries = _cached_value(
            ("recent-runs", str(resolved_layout.data_root)),
            ttl_seconds=8.0,
            builder=lambda: load_dashboard_home(layout=resolved_layout),
        )
        response = templates.TemplateResponse(
            request,
            "_recent_runs_panel.html",
            {
                "request": request,
                "recent_runs": _recent_runs(
                    summaries,
                    job_snapshots=dashboard_job_manager.list_snapshots(),
                ),
            },
        )
        response.headers["cache-control"] = "no-store, max-age=0"
        response.headers["pragma"] = "no-cache"
        return response

    @app.get("/recent-runs-version", response_class=JSONResponse)
    async def recent_runs_panel_version() -> JSONResponse:
        summaries = _cached_value(
            ("recent-runs", str(resolved_layout.data_root)),
            ttl_seconds=8.0,
            builder=lambda: load_dashboard_home(layout=resolved_layout),
        )
        payload = {
            "version": _recent_runs_version(
                summaries=summaries,
                job_snapshots=dashboard_job_manager.list_snapshots(),
                layout=resolved_layout,
            )
        }
        response = JSONResponse(payload)
        response.headers["cache-control"] = "no-store, max-age=0"
        response.headers["pragma"] = "no-cache"
        return response

    @app.get("/recent-runs-data", response_class=JSONResponse)
    async def recent_runs_panel_data() -> JSONResponse:
        summaries = _cached_value(
            ("recent-runs", str(resolved_layout.data_root)),
            ttl_seconds=8.0,
            builder=lambda: load_dashboard_home(layout=resolved_layout),
        )
        response = JSONResponse(
            {"recent_runs": _recent_runs_json_payload(summaries, job_snapshots=dashboard_job_manager.list_snapshots())}
        )
        response.headers["cache-control"] = "no-store, max-age=0"
        response.headers["pragma"] = "no-cache"
        return response

    @app.post("/jobs/{job_id}/evaluation-workers")
    async def update_evaluation_job_workers(
        request: Request,
        job_id: str,
    ) -> RedirectResponse:
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", [f"/jobs/{job_id}"])[0] or f"/jobs/{job_id}"
        try:
            workers = int(parsed_form.get("workers", ["1"])[0] or "1")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Workers must be an integer.") from exc
        snapshot = dashboard_job_manager.get_snapshot(job_id)
        metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
        if metadata.get("action") != "evaluate-deepphy":
            raise HTTPException(status_code=400, detail="Job is not a DeepPHY evaluation job.")
        request_payload = metadata.get("request")
        if not isinstance(request_payload, dict):
            raise HTTPException(status_code=400, detail="Evaluation job does not include a request payload.")
        env_id = metadata.get("env_id")
        run_id = metadata.get("run_id")
        evaluation_run_id = metadata.get("evaluation_run_id")
        if (not isinstance(run_id, str) or not run_id) and isinstance(
            request_payload.get("run_id"), str
        ):
            run_id = request_payload.get("run_id")
        if not all(isinstance(value, str) and value for value in (env_id, evaluation_run_id)):
            raise HTTPException(status_code=400, detail="Evaluation job metadata is incomplete.")
        request_payload = {**request_payload, "workers": max(1, int(workers))}
        try:
            action_request = DeepPHYEvaluationActionRequest.model_validate(request_payload)
        except (ValidationError, ValueError) as exc:
            message = _format_validation_error(exc) if isinstance(exc, ValidationError) else str(exc)
            raise HTTPException(status_code=400, detail=message) from exc
        try:
            eval_python = python_executable_for_env(env_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _resolved_pattern_run_dir, evaluation_run_dir, chosen_eval_run_id = resolve_deepphy_paths(
            action_request
        )
        dashboard_job_manager.terminate_and_remove(job_id)
        relaunched_argv = [str(eval_python), *build_deepphy_eval_argv(action_request)]
        new_snapshot = dashboard_job_manager.launch(
            name="evaluate-deepphy",
            argv=relaunched_argv,
            metadata={
                **metadata,
                "supports_pause": True,
                "env_id": env_id,
                "run_id": run_id,
                "evaluation_run_id": chosen_eval_run_id,
                "evaluation_run_dir": str(evaluation_run_dir)
                if evaluation_run_dir is not None
                else None,
                "benchmark_preset": action_request.benchmark_preset,
                "benchmark_task_count": summarize_deepphy_benchmark(
                    env_id,
                    preset=action_request.benchmark_preset,
                    task_limit=action_request.task_limit,
                ).task_count,
                "request": action_request.model_dump(mode="json", exclude_none=True),
                "resume_argv": relaunched_argv,
            },
        )
        pattern_run_dir = metadata.get("pattern_run_dir")
        use_unsupervised = (
            isinstance(pattern_run_dir, str)
            and f"/{UNSUPERVISED_RUNS_DIRNAME}/" in pattern_run_dir
        )
        if metadata.get("evaluation_scope") == "image-only":
            redirect_url = f"/env/{env_id}?job={new_snapshot.job_id}"
        elif use_unsupervised:
            redirect_url = (
                f"/automatic/{env_id}/{run_id}/evaluations/{evaluation_run_id}"
                f"?job={new_snapshot.job_id}"
            )
        else:
            redirect_url = (
                f"/runs/{env_id}/{run_id}/evaluations/{evaluation_run_id}"
                f"?job={new_snapshot.job_id}"
            )
        return RedirectResponse(url=redirect_url, status_code=303)

    @app.get("/env/{env_id}", response_class=HTMLResponse)
    async def environment_view(request: Request, env_id: str) -> HTMLResponse:
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        summary = load_environment_summary(env_id, layout=resolved_layout)
        return _render_environment_page(
            request,
            env_id=env_id,
            summary=summary,
            dataset_launch_error=None,
            dataset_launch_values=_default_dataset_launch_values(
                env_id, layout=resolved_layout
            ),
            label_error=None,
            label_success=None,
            label_values={
                "labels_text": (
                    summary.target_labels.raw_text
                    if summary.target_labels is not None
                    else ""
                )
            },
            launch_error=None,
            launch_values=_default_pattern_launch_values(summary),
            unsupervised_launch_error=None,
            unsupervised_launch_values=_default_unsupervised_launch_values(
                summary,
                layout=resolved_layout,
                job_snapshots=_env_job_snapshots(env_id),
            ),
        )

    @app.get("/env/{env_id}/libraries", response_class=HTMLResponse)
    async def libraries_view(request: Request, env_id: str) -> HTMLResponse:
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        summary = load_environment_summary(env_id, layout=resolved_layout)
        return templates.TemplateResponse(
            request,
            "libraries.html",
            {
                "page_title": f"{env_id} libraries",
                "request": request,
                "summary": summary,
                "nav_links": _nav_links(active="libraries", env_id=env_id),
            },
        )

    @app.get("/env/{env_id}/compare", response_class=HTMLResponse)
    async def compare_view(request: Request, env_id: str) -> HTMLResponse:
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        compare_payload = _build_supervised_compare_payload(
            env_id,
            requested_run_ids=request.query_params.getlist("run_id"),
        )
        return templates.TemplateResponse(
            request,
            "compare.html",
            {
                "page_title": f"{env_id} compare runs",
                "request": request,
                **compare_payload,
                "nav_links": _nav_links(active="compare", env_id=env_id),
            },
        )

    @app.post("/env/{env_id}/labels")
    async def update_target_labels(request: Request, env_id: str):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        labels_text = parsed_form.get("labels_text", [""])[0]
        try:
            normalized_text, _ = _normalize_labels_text(labels_text)
        except ValueError as exc:
            summary = load_environment_summary(env_id, layout=resolved_layout)
            return _render_environment_page(
                request,
                env_id=env_id,
                summary=summary,
                dataset_launch_error=None,
                dataset_launch_values=_default_dataset_launch_values(
                    env_id, layout=resolved_layout
                ),
                label_error=str(exc),
                label_success=None,
                label_values={"labels_text": labels_text},
                launch_error=None,
                launch_values=_default_pattern_launch_values(summary),
                unsupervised_launch_error=None,
                unsupervised_launch_values=_default_unsupervised_launch_values(
                    summary,
                    layout=resolved_layout,
                    job_snapshots=_env_job_snapshots(env_id),
                ),
                status_code=422,
            )

        labels_path = resolved_layout.config_root / env_id / "labels.txt"
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.write_text(normalized_text, encoding="utf-8")
        summary = load_environment_summary(env_id, layout=resolved_layout)
        return _render_environment_page(
            request,
            env_id=env_id,
            summary=summary,
            dataset_launch_error=None,
            dataset_launch_values=_default_dataset_launch_values(
                env_id, layout=resolved_layout
            ),
            label_error=None,
            label_success=f"Updated {labels_path.relative_to(resolved_layout.project_root)}",
            label_values={
                "labels_text": (
                    summary.target_labels.raw_text
                    if summary.target_labels is not None
                    else normalized_text
                )
            },
            launch_error=None,
            launch_values=_default_pattern_launch_values(summary),
            unsupervised_launch_error=None,
            unsupervised_launch_values=_default_unsupervised_launch_values(
                summary,
                layout=resolved_layout,
                job_snapshots=_env_job_snapshots(env_id),
            ),
        )

    @app.post("/env/{env_id}/datasets/build")
    async def launch_dataset_build(request: Request, env_id: str):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        defaults = _default_dataset_launch_values(env_id, layout=resolved_layout)
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        split_values = ["train", "validation", "test"]

        def _first(name: str, default: str = "") -> str:
            values = parsed_form.get(name)
            if not values:
                return default
            return values[0]

        dataset_name = (
            _coerce_optional(_first("dataset_name", defaults["dataset_name"]))
            or defaults["dataset_name"]
        )
        output_dir = resolved_layout.data_root / "datasets" / env_id / dataset_name
        dataset_launch_values = {
            "dataset_name": _first("dataset_name", defaults["dataset_name"]),
            "train_count": _first("train_count", defaults["train_count"]),
            "validation_count": _first("validation_count", defaults["validation_count"]),
            "test_count": _first("test_count", defaults["test_count"]),
            "train_offset": _first("train_offset", defaults["train_offset"]),
            "validation_offset": _first("validation_offset", defaults["validation_offset"]),
            "test_offset": _first("test_offset", defaults["test_offset"]),
            "force": bool(parsed_form.get("force")),
        }
        try:
            action_request = DatasetBuildActionRequest.model_validate(
                {
                    "env_id": env_id,
                    "output_dir": output_dir,
                    "dataset_name": dataset_name,
                    "train_count": int(_first("train_count", defaults["train_count"])),
                    "validation_count": int(
                        _first("validation_count", defaults["validation_count"])
                    ),
                    "test_count": int(_first("test_count", defaults["test_count"])),
                    "train_offset": int(_first("train_offset", defaults["train_offset"])),
                    "validation_offset": int(
                        _first("validation_offset", defaults["validation_offset"])
                    ),
                    "test_offset": int(_first("test_offset", defaults["test_offset"])),
                    "split": split_values,
                    "force": bool(parsed_form.get("force")),
                }
            )
        except ValidationError as exc:
            summary = load_environment_summary(env_id, layout=resolved_layout)
            return _render_environment_page(
                request,
                env_id=env_id,
                summary=summary,
                dataset_launch_error=_format_validation_error(exc),
                dataset_launch_values=dataset_launch_values,
                label_error=None,
                label_success=None,
                label_values={
                    "labels_text": (
                        summary.target_labels.raw_text
                        if summary.target_labels is not None
                        else ""
                    )
                },
                launch_error=None,
                launch_values=_default_pattern_launch_values(summary),
                unsupervised_launch_error=None,
                unsupervised_launch_values=_default_unsupervised_launch_values(
                    summary,
                    layout=resolved_layout,
                    job_snapshots=_env_job_snapshots(env_id),
                ),
                status_code=422,
            )

        snapshot = dashboard_job_manager.launch(
            name="build-dataset",
            argv=build_dataset_build_argv(action_request),
            cwd=resolved_layout.project_root,
            metadata={
                "action": "build-dataset",
                "env_id": env_id,
                "dataset_name": dataset_name,
                "supports_pause": True,
                "request": action_request.model_dump(mode="json", exclude_none=True),
            },
        )
        return RedirectResponse(url=f"/jobs/{snapshot.job_id}", status_code=303)

    @app.post("/env/{env_id}/learn")
    async def launch_learning_run(request: Request, env_id: str):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        enqueue = _launch_mode_from_parsed_form(parsed_form) == "queue"

        def _first(name: str, default: str = "") -> str:
            values = parsed_form.get(name)
            if not values:
                return default
            return values[0]

        summary = load_environment_summary(env_id, layout=resolved_layout)
        dataset_options = _pattern_learning_dataset_options(summary)
        manifest_by_dataset = {
            item["dataset_name"]: item["dataset_manifest"] for item in dataset_options
        }
        split_values = ["train"]
        validation_split_values = ["validation"]
        template_name = _resolve_funsearch_template_name(
            _first("funsearch_template", DEFAULT_FUNSEARCH_TEMPLATE)
        )
        template_values = _funsearch_template_values(template_name)
        launch_values = {
            "dataset_name": _first("dataset_name"),
            "limit": _first("limit"),
            "funsearch_template": template_name,
            "iteration_count": _first(
                "iteration_count", template_values["iteration_count"]
            ),
            "workers": _first("workers", "1"),
            "model": _first("model"),
            "seed": _first("seed"),
            "population_size": _first(
                "population_size", template_values["population_size"]
            ),
            "elite_fraction": _first(
                "elite_fraction", template_values["elite_fraction"]
            ),
            "children_per_elite": _first(
                "children_per_elite", template_values["children_per_elite"]
            ),
            "crossover_rate": _first("crossover_rate"),
        }
        selected_dataset_name = _coerce_optional(_first("dataset_name"))
        selected_dataset_manifest = (
            manifest_by_dataset.get(selected_dataset_name)
            if selected_dataset_name is not None
            else None
        )
        if selected_dataset_name is None or selected_dataset_manifest is None:
            return _render_environment_page(
                request,
                env_id=env_id,
                summary=summary,
                dataset_launch_error=None,
                dataset_launch_values=_default_dataset_launch_values(
                    env_id, layout=resolved_layout
                ),
                label_error=None,
                label_success=None,
                label_values={
                    "labels_text": (
                        summary.target_labels.raw_text
                        if summary.target_labels is not None
                        else ""
                    )
                },
                launch_error="dataset_name: select a dataset from the dropdown",
                launch_values=launch_values,
                unsupervised_launch_error=None,
                unsupervised_launch_values=_default_unsupervised_launch_values(
                    summary,
                    layout=resolved_layout,
                    job_snapshots=_env_job_snapshots(env_id),
                ),
                status_code=422,
            )
        try:
            pattern_learning_config_path = (
                resolved_layout.config_root / "pattern_learning.yaml"
            )
            patience_factor = 0.35
            if pattern_learning_config_path.is_file():
                patience_factor = load_pattern_learning_config(
                    pattern_learning_config_path
                ).patience_factor
            action_request = PatternLearningActionRequest.model_validate(
                {
                    "env_id": env_id,
                    "config_path": resolved_layout.config_root
                    / "pattern_learning.yaml",
                    "dataset_manifest": selected_dataset_manifest,
                    "split": split_values,
                    "validation_split": validation_split_values,
                    "limit": _coerce_optional(launch_values["limit"]),
                    "model": launch_values["model"],
                    "iteration_count": _coerce_optional(launch_values["iteration_count"])
                    or 1,
                    "workers": _coerce_optional(launch_values["workers"]) or 1,
                    "seed": _coerce_optional(launch_values["seed"]),
                    "population_size": _coerce_optional(
                        launch_values["population_size"]
                    ),
                    "elite_fraction": _coerce_optional(
                        launch_values["elite_fraction"]
                    ),
                    "children_per_elite": _coerce_optional(
                        launch_values["children_per_elite"]
                    ),
                    "patience_factor": patience_factor,
                    "crossover_rate": _coerce_optional(
                        launch_values["crossover_rate"]
                    ),
                }
            )
        except ValidationError as exc:
            return _render_environment_page(
                request,
                env_id=env_id,
                summary=summary,
                dataset_launch_error=None,
                dataset_launch_values=_default_dataset_launch_values(
                    env_id, layout=resolved_layout
                ),
                label_error=None,
                label_success=None,
                label_values={
                    "labels_text": (
                        summary.target_labels.raw_text
                        if summary.target_labels is not None
                        else ""
                    )
                },
                launch_error=_format_validation_error(exc),
                launch_values=launch_values,
                unsupervised_launch_error=None,
                unsupervised_launch_values=_default_unsupervised_launch_values(
                    summary,
                    layout=resolved_layout,
                    job_snapshots=_env_job_snapshots(env_id),
                ),
                status_code=422,
            )

        if action_request.output_path is None:
            # Compute once so argv, metadata, and downstream workflows agree on the run directory.
            action_request.output_path = default_pattern_learning_output_path(
                action_request
            )
        run_dir = pattern_run_dir_for_output_path(action_request.output_path)
        argv = build_supervised_learning_with_auto_eval_argv(action_request)
        snapshot = dashboard_job_manager.launch(
            name="supervised-learn-and-evaluate",
            argv=argv,
            cwd=resolved_layout.project_root,
            metadata={
                "action": "supervised-learn-and-evaluate",
                "env_id": env_id,
                "workflow_kind": "supervised",
                "supports_pause": True,
                "resume_argv": [*argv, "--resume"],
                "output_path": str(action_request.output_path),
                "run_dir": str(run_dir),
                "funsearch_template": template_name,
                "request": action_request.model_dump(mode="json", exclude_none=True),
            },
            enqueue=enqueue,
        )
        return RedirectResponse(url=f"/jobs/{snapshot.job_id}", status_code=303)

    @app.post("/env/{env_id}/image-only-evaluations/launch")
    async def launch_image_only_evaluation(request: Request, env_id: str):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)

        def _first(name: str, default: str = "") -> str:
            values = parsed_form.get(name)
            if not values:
                return default
            return values[0]

        summary = load_environment_summary(env_id, layout=resolved_layout)
        llm_defaults = _load_dashboard_llm_defaults(resolved_layout)
        form_values = {
            "benchmark_preset": _first("benchmark_preset", "quick-eval"),
            "model": _first("model", ""),
            "workers": _first("workers", "1"),
        }
        request_payload = {
            "env_id": env_id,
            "benchmark_preset": form_values["benchmark_preset"],
            "prompt_format": "WM",
            "model": form_values["model"],
            "description_mode": "trace_text",
            "ollama_base_url": llm_defaults["ollama_base_url"],
            "model_timeout": llm_defaults["model_timeout"],
            "capture_visuals": True,
            "workers": _coerce_optional(form_values["workers"]) or 1,
            "run_id": default_run_id(),
            "output_dir": (
                resolved_layout.data_root
                / "patterns"
                / env_id
                / "image_only_evaluations"
                / default_run_id()
            ),
        }
        request_payload["run_id"] = _coerce_optional(_first("evaluation_run_id")) or request_payload["run_id"]
        request_payload["output_dir"] = (
            resolved_layout.data_root
            / "patterns"
            / env_id
            / "image_only_evaluations"
            / request_payload["run_id"]
        )
        try:
            action_request = DeepPHYEvaluationActionRequest.model_validate(
                request_payload
            )
        except (ValidationError, ValueError) as exc:
            message = _format_validation_error(exc) if isinstance(exc, ValidationError) else str(exc)
            return _render_environment_page(
                request,
                env_id=env_id,
                summary=summary,
                dataset_launch_error=None,
                dataset_launch_values=_default_dataset_launch_values(
                    env_id, layout=resolved_layout
                ),
                label_error=None,
                label_success=None,
                label_values={
                    "labels_text": (
                        summary.target_labels.raw_text
                        if summary.target_labels is not None
                        else ""
                    )
                },
                launch_error=None,
                launch_values=_default_pattern_launch_values(summary),
                unsupervised_launch_error=None,
                unsupervised_launch_values=_default_unsupervised_launch_values(
                    summary,
                    layout=resolved_layout,
                    job_snapshots=_env_job_snapshots(env_id),
                ),
                image_only_evaluation_error=message,
                image_only_evaluation_values=form_values,
                status_code=422,
            )

        try:
            eval_python = python_executable_for_env(env_id)
        except FileNotFoundError as exc:
            return _render_environment_page(
                request,
                env_id=env_id,
                summary=summary,
                dataset_launch_error=None,
                dataset_launch_values=_default_dataset_launch_values(
                    env_id, layout=resolved_layout
                ),
                label_error=None,
                label_success=None,
                label_values={
                    "labels_text": (
                        summary.target_labels.raw_text
                        if summary.target_labels is not None
                        else ""
                    )
                },
                launch_error=None,
                launch_values=_default_pattern_launch_values(summary),
                unsupervised_launch_error=None,
                unsupervised_launch_values=_default_unsupervised_launch_values(
                    summary,
                    layout=resolved_layout,
                    job_snapshots=_env_job_snapshots(env_id),
                ),
                image_only_evaluation_error=str(exc),
                image_only_evaluation_values=form_values,
                status_code=422,
            )

        _resolved_pattern_run_dir, evaluation_run_dir, chosen_eval_run_id = resolve_deepphy_paths(
            action_request
        )
        if evaluation_run_dir is not None:
            try:
                evaluation_run_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                return _render_environment_page(
                    request,
                    env_id=env_id,
                    summary=summary,
                    dataset_launch_error=None,
                    dataset_launch_values=_default_dataset_launch_values(
                        env_id, layout=resolved_layout
                    ),
                    label_error=None,
                    label_success=None,
                    label_values={
                        "labels_text": (
                            summary.target_labels.raw_text
                            if summary.target_labels is not None
                            else ""
                        )
                    },
                    launch_error=None,
                    launch_values=_default_pattern_launch_values(summary),
                    unsupervised_launch_error=None,
                    unsupervised_launch_values=_default_unsupervised_launch_values(
                        summary,
                        layout=resolved_layout,
                        job_snapshots=_env_job_snapshots(env_id),
                    ),
                    image_only_evaluation_error=(
                        f"Image-only evaluation already exists: {chosen_eval_run_id}"
                    ),
                    image_only_evaluation_values=form_values,
                    status_code=422,
                )
            (evaluation_run_dir / "request.json").write_text(
                json.dumps(
                    action_request.model_dump(mode="json", exclude_none=True),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        snapshot = dashboard_job_manager.launch(
            name="evaluate-deepphy",
            argv=[str(eval_python), *build_deepphy_eval_argv(action_request)],
            metadata={
                "action": "evaluate-deepphy",
                "supports_pause": True,
                "env_id": env_id,
                "evaluation_scope": "image-only",
                "evaluation_run_id": chosen_eval_run_id,
                "evaluation_run_dir": str(evaluation_run_dir)
                if evaluation_run_dir is not None
                else None,
                "benchmark_preset": action_request.benchmark_preset,
                "benchmark_task_count": summarize_deepphy_benchmark(
                    env_id,
                    preset=action_request.benchmark_preset,
                    task_limit=action_request.task_limit,
                ).task_count,
                "request": action_request.model_dump(mode="json", exclude_none=True),
                "resume_argv": [str(eval_python), *build_deepphy_eval_argv(action_request)],
            },
            enqueue=True,
        )
        return RedirectResponse(url=f"/env/{env_id}?job={snapshot.job_id}", status_code=303)

    @app.post("/env/{env_id}/automatic/launch")
    async def launch_unsupervised_run(request: Request, env_id: str):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        enqueue = _launch_mode_from_parsed_form(parsed_form) == "queue"

        def _first(name: str, default: str = "") -> str:
            values = parsed_form.get(name)
            if not values:
                return default
            return values[0]

        summary = load_environment_summary(env_id, layout=resolved_layout)
        source_options = _supervised_source_options(
            summary,
            layout=resolved_layout,
            job_snapshots=_env_job_snapshots(env_id),
        )
        valid_source_values = {item["value"] for item in source_options}
        saved_source_values = {
            item["value"] for item in source_options if item.get("status") == "saved"
        }
        valid_evaluation_values = {
            item["value"] for item in _supervised_evaluation_options(summary)
        }
        split_values = ["train"]
        validation_split_values = ["validation"]
        template_name = _resolve_funsearch_template_name(
            _first("funsearch_template", DEFAULT_FUNSEARCH_TEMPLATE)
        )
        template_values = _funsearch_template_values(template_name)
        unsupervised_values = {
            "source_supervised_run_dir": _first("source_supervised_run_dir"),
            "source_evaluation_run_id": _first("source_evaluation_run_id"),
            "reasoning_trace_count": _first("reasoning_trace_count", "6"),
            "proposed_label_count": _first("proposed_label_count", "10"),
            "limit": _first("limit"),
            "funsearch_template": template_name,
            "iteration_count": _first(
                "iteration_count", template_values["iteration_count"]
            ),
            "workers": _first("workers", "1"),
            "model": _first("model"),
            "seed": _first("seed"),
            "population_size": _first(
                "population_size", template_values["population_size"]
            ),
            "elite_fraction": _first(
                "elite_fraction", template_values["elite_fraction"]
            ),
            "children_per_elite": _first(
                "children_per_elite", template_values["children_per_elite"]
            ),
            "crossover_rate": _first("crossover_rate"),
            "run_timestamp": _first("run_timestamp"),
        }

        selected_source = _coerce_optional(_first("source_supervised_run_dir"))
        selected_eval = _coerce_optional(_first("source_evaluation_run_id"))
        if selected_source is None or selected_source not in valid_source_values:
            return _render_environment_page(
                request,
                env_id=env_id,
                summary=summary,
                dataset_launch_error=None,
                dataset_launch_values=_default_dataset_launch_values(
                    env_id, layout=resolved_layout
                ),
                label_error=None,
                label_success=None,
                label_values={
                    "labels_text": (
                        summary.target_labels.raw_text
                        if summary.target_labels is not None
                        else ""
                    )
                },
                launch_error=None,
                launch_values=_default_pattern_launch_values(summary),
                unsupervised_launch_error=(
                    "source_supervised_run_dir: select a supervised library from the dropdown"
                ),
                unsupervised_launch_values=unsupervised_values,
                status_code=422,
            )
        if selected_eval is None or selected_eval not in valid_evaluation_values:
            return _render_environment_page(
                request,
                env_id=env_id,
                summary=summary,
                dataset_launch_error=None,
                dataset_launch_values=_default_dataset_launch_values(
                    env_id, layout=resolved_layout
                ),
                label_error=None,
                label_success=None,
                label_values={
                    "labels_text": (
                        summary.target_labels.raw_text
                        if summary.target_labels is not None
                        else ""
                    )
                },
                launch_error=None,
                launch_values=_default_pattern_launch_values(summary),
                unsupervised_launch_error=(
                    "source_evaluation_run_id: select a supervised evaluation run"
                ),
                unsupervised_launch_values=unsupervised_values,
                status_code=422,
            )
        if (
            selected_source not in saved_source_values
            and selected_eval != AUTO_SOURCE_EVALUATION_RUN_ID
        ):
            return _render_environment_page(
                request,
                env_id=env_id,
                summary=summary,
                dataset_launch_error=None,
                dataset_launch_values=_default_dataset_launch_values(
                    env_id, layout=resolved_layout
                ),
                label_error=None,
                label_success=None,
                label_values={
                    "labels_text": (
                        summary.target_labels.raw_text
                        if summary.target_labels is not None
                        else ""
                    )
                },
                launch_error=None,
                launch_values=_default_pattern_launch_values(summary),
                unsupervised_launch_error=(
                    "source_evaluation_run_id: queued or running supervised sources must use Auto from selected source run"
                ),
                unsupervised_launch_values=unsupervised_values,
                status_code=422,
            )

        try:
            pattern_learning_config_path = (
                resolved_layout.config_root / "pattern_learning.yaml"
            )
            patience_factor = 0.35
            if pattern_learning_config_path.is_file():
                patience_factor = load_pattern_learning_config(
                    pattern_learning_config_path
                ).patience_factor
            action_request = AutomaticLearningActionRequest.model_validate(
                {
                    "env_id": env_id,
                    "source_supervised_run_dir": _resolve_dashboard_path(
                        selected_source, layout=resolved_layout
                    ),
                    "source_evaluation_run_id": selected_eval,
                    "reasoning_trace_count": _coerce_optional(
                        unsupervised_values["reasoning_trace_count"]
                    )
                    or 6,
                    "proposed_label_count": _coerce_optional(
                        unsupervised_values["proposed_label_count"]
                    )
                    or 10,
                    "split": split_values,
                    "validation_split": validation_split_values,
                    "limit": _coerce_optional(unsupervised_values["limit"]),
                    "model": unsupervised_values["model"],
                    "iteration_count": _coerce_optional(
                        unsupervised_values["iteration_count"]
                    )
                    or 1,
                    "workers": _coerce_optional(unsupervised_values["workers"]) or 1,
                    "seed": _coerce_optional(unsupervised_values["seed"]),
                    "population_size": _coerce_optional(
                        unsupervised_values["population_size"]
                    ),
                    "elite_fraction": _coerce_optional(
                        unsupervised_values["elite_fraction"]
                    ),
                    "children_per_elite": _coerce_optional(
                        unsupervised_values["children_per_elite"]
                    ),
                    "patience_factor": patience_factor,
                    "crossover_rate": _coerce_optional(
                        unsupervised_values["crossover_rate"]
                    ),
                    "run_timestamp": _coerce_optional(
                        unsupervised_values["run_timestamp"]
                    ),
                }
            )
        except ValidationError as exc:
            return _render_environment_page(
                request,
                env_id=env_id,
                summary=summary,
                dataset_launch_error=None,
                dataset_launch_values=_default_dataset_launch_values(
                    env_id, layout=resolved_layout
                ),
                label_error=None,
                label_success=None,
                label_values={
                    "labels_text": (
                        summary.target_labels.raw_text
                        if summary.target_labels is not None
                        else ""
                    )
                },
                launch_error=None,
                launch_values=_default_pattern_launch_values(summary),
                unsupervised_launch_error=_format_validation_error(exc),
                unsupervised_launch_values=unsupervised_values,
                status_code=422,
            )

        effective_run_dir = default_automatic_run_dir(
            env_id,
            timestamp=action_request.run_timestamp,
        )
        effective_action_request = action_request.model_copy(
            update={"run_timestamp": effective_run_dir.name}
        )
        argv = build_automatic_launch_argv(effective_action_request)
        try:
            snapshot = dashboard_job_manager.launch(
                name=AUTOMATIC_LAUNCH_ACTION,
                argv=argv,
                cwd=resolved_layout.project_root,
                metadata={
                    "action": AUTOMATIC_LAUNCH_ACTION,
                    "env_id": env_id,
                    "workflow_kind": AUTOMATIC_WORKFLOW_KIND,
                    "supports_pause": True,
                    "resume_argv": [*argv, "--resume"],
                    "funsearch_template": template_name,
                    "request": effective_action_request.model_dump(mode="json", exclude_none=True),
                    "source_supervised_run_dir": str(effective_action_request.source_supervised_run_dir),
                    "source_supervised_run_id": (
                        effective_action_request.source_supervised_run_dir.name
                        if effective_action_request.source_supervised_run_dir is not None
                        else None
                    ),
                    "source_evaluation_run_id": effective_action_request.source_evaluation_run_id,
                    "run_dir": str(effective_run_dir),
                },
                enqueue=enqueue,
            )
        except ValueError as exc:
            return _render_environment_page(
                request,
                env_id=env_id,
                summary=summary,
                dataset_launch_error=None,
                dataset_launch_values=_default_dataset_launch_values(
                    env_id, layout=resolved_layout
                ),
                label_error=None,
                label_success=None,
                label_values={
                    "labels_text": (
                        summary.target_labels.raw_text
                        if summary.target_labels is not None
                        else ""
                    )
                },
                launch_error=None,
                launch_values=_default_pattern_launch_values(summary),
                unsupervised_launch_error=str(exc),
                unsupervised_launch_values=unsupervised_values,
                status_code=422,
            )
        return RedirectResponse(url=f"/jobs/{snapshot.job_id}", status_code=303)

    @app.post("/env/{env_id}/unsupervised/launch")
    async def launch_true_unsupervised_run(request: Request, env_id: str):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        enqueue = _launch_mode_from_parsed_form(parsed_form) == "queue"

        def _first(name: str, default: str = "") -> str:
            values = parsed_form.get(name)
            if not values:
                return default
            return values[0]

        summary = load_environment_summary(env_id, layout=resolved_layout)
        dataset_options = _pattern_learning_dataset_options(summary)
        dataset_by_name = {item["dataset_name"]: item for item in dataset_options}
        template_name = _resolve_funsearch_template_name(
            _first("funsearch_template", DEFAULT_FUNSEARCH_TEMPLATE)
        )
        template_values = _funsearch_template_values(template_name)
        form_values = {
            "dataset_name": _first("dataset_name"),
            "trace_count": _first("trace_count", "10"),
            "labels_per_trace": _first("labels_per_trace", "10"),
            "final_label_count": _first("final_label_count", "10"),
            "funsearch_template": template_name,
            "iteration_count": _first("iteration_count", template_values["iteration_count"]),
            "workers": _first("workers", "1"),
            "model": _first("model"),
            "seed": _first("seed"),
            "population_size": _first("population_size", template_values["population_size"]),
            "elite_fraction": _first("elite_fraction", template_values["elite_fraction"]),
            "children_per_elite": _first("children_per_elite", template_values["children_per_elite"]),
            "crossover_rate": _first("crossover_rate"),
            "run_timestamp": _first("run_timestamp"),
        }
        selected_dataset = _coerce_optional(form_values["dataset_name"])
        if selected_dataset is None or selected_dataset not in dataset_by_name:
            return _render_environment_page(
                request,
                env_id=env_id,
                summary=summary,
                dataset_launch_error=None,
                dataset_launch_values=_default_dataset_launch_values(env_id, layout=resolved_layout),
                label_error=None,
                label_success=None,
                label_values={"labels_text": summary.target_labels.raw_text if summary.target_labels is not None else ""},
                launch_error=None,
                launch_values=_default_pattern_launch_values(summary),
                unsupervised_launch_error=None,
                unsupervised_launch_values=_default_unsupervised_launch_values(
                    summary,
                    layout=resolved_layout,
                    job_snapshots=_env_job_snapshots(env_id),
                ),
                true_unsupervised_launch_error="dataset_name: select a dataset",
                true_unsupervised_launch_values=form_values,
                status_code=422,
            )
        try:
            pattern_learning_config_path = resolved_layout.config_root / "pattern_learning.yaml"
            patience_factor = 0.35
            if pattern_learning_config_path.is_file():
                patience_factor = load_pattern_learning_config(
                    pattern_learning_config_path
                ).patience_factor
            action_request = UnsupervisedLearningActionRequest.model_validate(
                {
                    "env_id": env_id,
                    "dataset_manifest": dataset_by_name[selected_dataset]["dataset_manifest"],
                    "split": ["train"],
                    "validation_split": ["validation"],
                    "model": form_values["model"],
                    "iteration_count": _coerce_optional(form_values["iteration_count"]) or 1,
                    "workers": _coerce_optional(form_values["workers"]) or 1,
                    "seed": _coerce_optional(form_values["seed"]),
                    "population_size": _coerce_optional(form_values["population_size"]),
                    "elite_fraction": _coerce_optional(form_values["elite_fraction"]),
                    "children_per_elite": _coerce_optional(form_values["children_per_elite"]),
                    "patience_factor": patience_factor,
                    "crossover_rate": _coerce_optional(form_values["crossover_rate"]),
                    "trace_count": _coerce_optional(form_values["trace_count"]) or 10,
                    "labels_per_trace": _coerce_optional(form_values["labels_per_trace"]) or 10,
                    "final_label_count": _coerce_optional(form_values["final_label_count"]) or 10,
                    "run_timestamp": _coerce_optional(form_values["run_timestamp"]),
                }
            )
        except ValidationError as exc:
            return _render_environment_page(
                request,
                env_id=env_id,
                summary=summary,
                dataset_launch_error=None,
                dataset_launch_values=_default_dataset_launch_values(env_id, layout=resolved_layout),
                label_error=None,
                label_success=None,
                label_values={"labels_text": summary.target_labels.raw_text if summary.target_labels is not None else ""},
                launch_error=None,
                launch_values=_default_pattern_launch_values(summary),
                unsupervised_launch_error=None,
                unsupervised_launch_values=_default_unsupervised_launch_values(
                    summary,
                    layout=resolved_layout,
                    job_snapshots=_env_job_snapshots(env_id),
                ),
                true_unsupervised_launch_error=_format_validation_error(exc),
                true_unsupervised_launch_values=form_values,
                status_code=422,
            )
        effective_run_dir = default_unsupervised_run_dir(
            env_id,
            timestamp=action_request.run_timestamp,
        )
        effective_action_request = action_request.model_copy(
            update={"run_timestamp": effective_run_dir.name}
        )
        argv = build_unsupervised_launch_argv(effective_action_request)
        try:
            snapshot = dashboard_job_manager.launch(
                name=UNSUPERVISED_LAUNCH_ACTION,
                argv=argv,
                cwd=resolved_layout.project_root,
                metadata={
                    "action": UNSUPERVISED_LAUNCH_ACTION,
                    "env_id": env_id,
                    "workflow_kind": UNSUPERVISED_WORKFLOW_KIND,
                    "supports_pause": True,
                    "resume_argv": [*argv, "--resume"],
                    "funsearch_template": template_name,
                    "request": effective_action_request.model_dump(mode="json", exclude_none=True),
                    "run_dir": str(effective_run_dir),
                },
                enqueue=enqueue,
            )
        except ValueError as exc:
            return _render_environment_page(
                request,
                env_id=env_id,
                summary=summary,
                dataset_launch_error=None,
                dataset_launch_values=_default_dataset_launch_values(env_id, layout=resolved_layout),
                label_error=None,
                label_success=None,
                label_values={"labels_text": summary.target_labels.raw_text if summary.target_labels is not None else ""},
                launch_error=None,
                launch_values=_default_pattern_launch_values(summary),
                unsupervised_launch_error=None,
                unsupervised_launch_values=_default_unsupervised_launch_values(
                    summary,
                    layout=resolved_layout,
                    job_snapshots=_env_job_snapshots(env_id),
                ),
                true_unsupervised_launch_error=str(exc),
                true_unsupervised_launch_values=form_values,
                status_code=422,
            )
        return RedirectResponse(url=f"/jobs/{snapshot.job_id}", status_code=303)

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    async def job_view(request: Request, job_id: str) -> HTMLResponse:
        try:
            snapshot = dashboard_job_manager.get_snapshot(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        learning_target = _learning_run_target(snapshot)
        if learning_target is not None:
            return RedirectResponse(
                url=f"{learning_target['run_url']}?job={job_id}",
                status_code=303,
            )
        evaluation_target = _evaluation_job_target(snapshot)
        if evaluation_target is not None:
            return RedirectResponse(
                url=f"{evaluation_target['evaluation_url']}?job={job_id}",
                status_code=303,
            )
        library_evaluation_target = _library_evaluation_job_target(snapshot)
        if library_evaluation_target is not None:
            return RedirectResponse(
                url=f"{library_evaluation_target['library_evaluation_url']}?job={job_id}",
                status_code=303,
            )
        dataset_progress = _dataset_progress_payload(snapshot, layout=resolved_layout)
        learning_progress = _learning_progress_context(snapshot, layout=resolved_layout)
        return templates.TemplateResponse(
            request,
            "job.html",
            {
                "page_title": f"Job {job_id}",
                "request": request,
                "job": snapshot,
                "dataset_progress": dataset_progress,
                "learning_progress": learning_progress,
                "poll_url": f"/jobs/{job_id}/status.json",
            },
        )

    @app.get("/jobs/{job_id}/status.json", response_class=JSONResponse)
    async def job_status_json(job_id: str) -> JSONResponse:
        try:
            snapshot = dashboard_job_manager.get_snapshot(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(_job_payload(snapshot, layout=resolved_layout))

    @app.post("/jobs/{job_id}/pause")
    async def job_pause(request: Request, job_id: str):
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", [f"/jobs/{job_id}"])[0] or f"/jobs/{job_id}"
        try:
            dashboard_job_manager.pause(job_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RedirectResponse(url=next_url, status_code=303)

    @app.post("/jobs/{job_id}/resume")
    async def job_resume(request: Request, job_id: str):
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", [f"/jobs/{job_id}"])[0] or f"/jobs/{job_id}"
        try:
            dashboard_job_manager.resume(job_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RedirectResponse(url=next_url, status_code=303)

    @app.post("/jobs/{job_id}/requeue")
    async def job_requeue(request: Request, job_id: str):
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", [f"/jobs/{job_id}"])[0] or f"/jobs/{job_id}"
        try:
            dashboard_job_manager.requeue(job_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RedirectResponse(url=next_url, status_code=303)

    @app.post("/jobs/queue/reorder")
    async def job_queue_reorder(request: Request):
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", ["/"])[0] or "/"
        ordered_job_ids = parsed_form.get("ordered_job_id", [])
        try:
            dashboard_job_manager.reorder_queue(ordered_job_ids)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if request.headers.get("x-requested-with") == "dashboard-queue-reorder":
            return _render_queue_panel_response(request)
        return RedirectResponse(url=next_url, status_code=303)

    @app.post("/jobs/queue/settings")
    async def job_queue_settings_update(request: Request):
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", ["/"])[0] or "/"
        raw_value = parsed_form.get("max_concurrent_jobs", ["1"])[0]
        try:
            max_concurrent_jobs = max(1, int(raw_value or "1"))
        except ValueError as exc:
            raise HTTPException(status_code=409, detail="Queue concurrency must be an integer.") from exc
        enforce_single_pooltool_job = (
            parsed_form.get("enforce_single_pooltool_job", [""])[0].strip().lower()
            in {"1", "true", "yes", "on"}
        )
        app.state.queue_settings = save_queue_settings_state(
            resolved_layout,
            {
                **dict(app.state.queue_settings or {}),
                "max_concurrent_jobs": max_concurrent_jobs,
                "enforce_single_pooltool_job": enforce_single_pooltool_job,
            },
        )
        dashboard_job_manager.set_max_concurrent_queue_jobs(
            app.state.queue_settings["max_concurrent_jobs"]
        )
        dashboard_job_manager.set_enforce_single_pooltool_queue_job(
            app.state.queue_settings["enforce_single_pooltool_job"]
        )
        if request.headers.get("x-requested-with") == "dashboard-queue-settings":
            return _render_queue_panel_response(request)
        return RedirectResponse(url=next_url, status_code=303)

    @app.post("/jobs/{job_id}/clear")
    async def job_clear(request: Request, job_id: str):
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", ["/"])[0] or "/"
        try:
            dashboard_job_manager.clear(job_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RedirectResponse(url=next_url, status_code=303)

    @app.post("/jobs/clear-failed")
    async def jobs_clear_failed(request: Request):
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", ["/"])[0] or "/"
        dashboard_job_manager.clear_all_failed()
        return RedirectResponse(url=next_url, status_code=303)

    @app.post("/jobs/{job_id}/exit")
    async def job_exit(request: Request, job_id: str):
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", ["/"])[0] or "/"
        try:
            dashboard_job_manager.exit_paused(job_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RedirectResponse(url=next_url, status_code=303)

    @app.get("/env/{env_id}/datasets/{dataset_name}", response_class=HTMLResponse)
    async def dataset_view(
        request: Request,
        env_id: str,
        dataset_name: str,
    ) -> HTMLResponse:
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        try:
            summary = load_dataset_detail(env_id, dataset_name, layout=resolved_layout)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return templates.TemplateResponse(
            request,
            "dataset.html",
            {
                "page_title": f"{env_id} {dataset_name}",
                "request": request,
                "summary": summary,
                "recent_dataset_jobs": _dataset_jobs_for_env(
                    dashboard_job_manager.list_snapshots(),
                    env_id=env_id,
                    dataset_name=dataset_name,
                ),
            },
        )

    @app.post("/env/{env_id}/datasets/{dataset_name}/delete")
    async def delete_dataset(request: Request, env_id: str, dataset_name: str):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        _validate_path_segment(dataset_name, label="dataset name")
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", [f"/env/{env_id}"])[0] or f"/env/{env_id}"
        dataset_root = resolved_layout.data_root / "datasets" / env_id / dataset_name
        _delete_tree(
            dataset_root,
            layout=resolved_layout,
            job_manager=dashboard_job_manager,
        )
        return RedirectResponse(url=next_url, status_code=303)

    @app.get(
        "/env/{env_id}/datasets/{dataset_name}/{split_name}/{sample_name}",
        response_class=HTMLResponse,
    )
    async def sample_view(
        request: Request,
        env_id: str,
        dataset_name: str,
        split_name: str,
        sample_name: str,
    ) -> HTMLResponse:
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        try:
            summary = load_dataset_sample(
                env_id,
                dataset_name,
                split_name,
                sample_name,
                layout=resolved_layout,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return templates.TemplateResponse(
            request,
            "sample.html",
            {
                "page_title": f"{env_id} {dataset_name} {sample_name}",
                "request": request,
                "summary": summary,
            },
        )

    @app.get(
        "/env/{env_id}/datasets/{dataset_name}/{split_name}/{sample_name}/media/initial_scene.png"
    )
    async def sample_scene_preview(
        env_id: str,
        dataset_name: str,
        split_name: str,
        sample_name: str,
    ) -> FileResponse:
        try:
            summary = load_dataset_sample(
                env_id,
                dataset_name,
                split_name,
                sample_name,
                layout=resolved_layout,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if summary.scene_image_path is None:
            raise HTTPException(status_code=404, detail="initial_scene.png not found")
        return FileResponse(summary.scene_image_path, media_type="image/png")

    @app.get(
        "/env/{env_id}/datasets/{dataset_name}/{split_name}/{sample_name}/media/simulation.mp4"
    )
    async def sample_simulation_preview(
        env_id: str,
        dataset_name: str,
        split_name: str,
        sample_name: str,
    ) -> FileResponse:
        try:
            summary = load_dataset_sample(
                env_id,
                dataset_name,
                split_name,
                sample_name,
                layout=resolved_layout,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if summary.simulation_path is None:
            raise HTTPException(status_code=404, detail="simulation.mp4 not found")
        return FileResponse(summary.simulation_path, media_type="video/mp4")

    @app.get(
        "/env/{env_id}/datasets/{dataset_name}/{split_name}/{sample_name}/media/{filename}"
    )
    async def sample_media_preview(
        env_id: str,
        dataset_name: str,
        split_name: str,
        sample_name: str,
        filename: str,
    ) -> FileResponse:
        if "/" in filename or filename in {".", ".."}:
            raise HTTPException(status_code=404, detail="Unknown media file")
        try:
            summary = load_dataset_sample(
                env_id,
                dataset_name,
                split_name,
                sample_name,
                layout=resolved_layout,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        allowed_paths = {
            path.name: path
            for path in [*summary.scene_image_paths, *summary.simulation_paths]
        }
        selected_path = allowed_paths.get(filename)
        if selected_path is None or not selected_path.is_file():
            raise HTTPException(status_code=404, detail=f"{filename} not found")
        media_type = "image/png" if selected_path.suffix.lower() == ".png" else "video/mp4"
        return FileResponse(selected_path, media_type=media_type)

    @app.get("/runs/{env_id}/{run_id}", response_class=HTMLResponse)
    async def run_view(
        request: Request, env_id: str, run_id: str, job: str | None = None
    ) -> HTMLResponse:
        return _render_run_page(request, env_id=env_id, run_id=run_id, job_id=job)

    @app.get("/runs/{env_id}/{run_id}/llm-calls/{sequence}", response_class=HTMLResponse)
    async def run_llm_call_view(
        request: Request,
        env_id: str,
        run_id: str,
        sequence: int,
    ) -> HTMLResponse:
        try:
            run = load_pattern_run_detail(env_id, run_id, layout=resolved_layout)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail=f"Pattern run not found: {env_id}/{run_id}"
            ) from exc
        call = next((item for item in run.llm_calls if item.sequence == sequence), None)
        if call is None:
            raise HTTPException(
                status_code=404,
                detail=f"LLM call not found for run {run_id}: {sequence}",
            )
        run_url = f"/runs/{env_id}/{run_id}"
        return templates.TemplateResponse(
            request,
            "llm_call.html",
            {
                "page_title": f"{env_id} run {run_id} llm call {sequence}",
                "request": request,
                "env_id": env_id,
                "run_id": run_id,
                "run": run,
                "run_url": run_url,
                "run_kind": "supervised",
                "call": call,
                "nav_links": _nav_links(active="env", env_id=env_id),
            },
        )

    @app.post("/runs/{env_id}/{run_id}/delete")
    async def delete_run(request: Request, env_id: str, run_id: str):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        _validate_path_segment(run_id, label="run id")
        if run_id in {UNSUPERVISED_RUNS_DIRNAME, AUTOMATIC_RUNS_DIRNAME}:
            raise HTTPException(status_code=404, detail=f"Invalid run id: {run_id}")
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", [f"/env/{env_id}"])[0] or f"/env/{env_id}"
        run_root = resolved_layout.data_root / "patterns" / env_id / run_id
        _delete_tree(
            run_root,
            layout=resolved_layout,
            job_manager=dashboard_job_manager,
        )
        return RedirectResponse(url=next_url, status_code=303)

    @app.post("/runs/{env_id}/{run_id}/workers")
    async def update_run_workers(request: Request, env_id: str, run_id: str):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        _validate_path_segment(run_id, label="run id")
        if run_id in {UNSUPERVISED_RUNS_DIRNAME, AUTOMATIC_RUNS_DIRNAME}:
            raise HTTPException(status_code=404, detail=f"Invalid run id: {run_id}")
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", [f"/runs/{env_id}/{run_id}"])[0] or f"/runs/{env_id}/{run_id}"
        try:
            workers = int(parsed_form.get("workers", ["1"])[0] or "1")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Workers must be an integer.") from exc
        run_root = resolved_layout.data_root / "patterns" / env_id / run_id
        if not run_root.is_dir():
            raise HTTPException(status_code=404, detail=f"Pattern run not found: {env_id}/{run_id}")
        _update_learning_run_worker_target(run_root, workers)
        _restart_learning_job_for_control_update(
            env_id=env_id,
            run_id=run_id,
            kind="supervised",
            job_manager=dashboard_job_manager,
        )
        return RedirectResponse(url=next_url, status_code=303)

    @app.post("/runs/{env_id}/{run_id}/mark-label-done")
    async def mark_run_label_done(request: Request, env_id: str, run_id: str):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        _validate_path_segment(run_id, label="run id")
        if run_id in {UNSUPERVISED_RUNS_DIRNAME, AUTOMATIC_RUNS_DIRNAME}:
            raise HTTPException(status_code=404, detail=f"Invalid run id: {run_id}")
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", [f"/runs/{env_id}/{run_id}"])[0] or f"/runs/{env_id}/{run_id}"
        label_name = parsed_form.get("label", [""])[0]
        raw_label_index = parsed_form.get("label_index", [""])[0]
        label_index: int | None = None
        if raw_label_index:
            try:
                label_index = int(raw_label_index)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Label index must be an integer.") from exc
        run_root = resolved_layout.data_root / "patterns" / env_id / run_id
        if not run_root.is_dir():
            raise HTTPException(status_code=404, detail=f"Pattern run not found: {env_id}/{run_id}")
        try:
            _mark_learning_label_done(
                run_root,
                label_name=label_name,
                label_index=label_index,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _restart_learning_job_for_control_update(
            env_id=env_id,
            run_id=run_id,
            kind="supervised",
            job_manager=dashboard_job_manager,
        )
        return RedirectResponse(url=next_url, status_code=303)

    @app.get("/automatic/{env_id}/{run_id}", response_class=HTMLResponse)
    async def unsupervised_run_view(
        request: Request, env_id: str, run_id: str, job: str | None = None
    ) -> HTMLResponse:
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        run_job = _find_learning_job_for_run(
            env_id=env_id,
            run_id=run_id,
            kind=AUTOMATIC_WORKFLOW_KIND,
            requested_job_id=job,
            job_manager=dashboard_job_manager,
        )
        try:
            run = load_automatic_library_detail(env_id, run_id, layout=resolved_layout)
        except FileNotFoundError:
            run = None
        if run is None and run_job is None:
            raise HTTPException(status_code=404, detail=f"Automatic run not found: {env_id}/{run_id}")
        run_json_text = (
            json.dumps(run.raw_run_payload, indent=2, sort_keys=True)
            if run is not None
            else ""
        )
        request_payload = (
            run_job.metadata.get("request")
            if run_job is not None and isinstance(run_job.metadata.get("request"), dict)
            else None
        )
        learning_settings = _learning_settings(
            run_payload=run.raw_run_payload if run is not None else None,
            request_payload=request_payload,
        )
        source_supervised_run_id = (
            run.source_supervised_run_id
            if run is not None and run.source_supervised_run_id
            else (
                (
                    str(run_job.metadata.get("source_supervised_run_id")).strip()
                    if run_job is not None
                    and isinstance(run_job.metadata.get("source_supervised_run_id"), str)
                    and str(run_job.metadata.get("source_supervised_run_id")).strip()
                    else None
                )
                if run_job is not None
                else None
            )
        )
        source_evaluation_run_id = (
            run.source_evaluation_run_id
            if run is not None and run.source_evaluation_run_id
            else (
                (
                    str(run_job.metadata.get("source_evaluation_run_id")).strip()
                    if run_job is not None
                    and isinstance(run_job.metadata.get("source_evaluation_run_id"), str)
                    and str(run_job.metadata.get("source_evaluation_run_id")).strip()
                    else None
                )
                if run_job is not None
                else None
            )
        )
        source_comparison = (
            next(
                (
                    item
                    for item in run.source_supervised_evaluation_runs
                    if item.run_id == source_evaluation_run_id
                ),
                None,
            )
            if run is not None and run.source_supervised_evaluation_runs
            else None
        )
        if (
            source_comparison is None
            and source_supervised_run_id
            and source_evaluation_run_id
        ):
            source_eval_root = (
                resolved_layout.data_root
                / "patterns"
                / env_id
                / source_supervised_run_id
                / "deepphy_eval"
                / source_evaluation_run_id
            )
            if source_eval_root.is_dir():
                try:
                    source_comparison = load_evaluation_run_summary(source_eval_root)
                except (FileNotFoundError, ValueError):
                    source_comparison = None
        combined_comparison = (
            run.evaluation_runs[0]
            if run is not None and run.evaluation_runs
            else None
        )
        requested_proposed_label_count = (
            int(request_payload.get("proposed_label_count"))
            if isinstance(request_payload, dict)
            and request_payload.get("proposed_label_count") is not None
            else None
        )
        requested_reasoning_trace_count = (
            int(request_payload.get("reasoning_trace_count"))
            if isinstance(request_payload, dict)
            and request_payload.get("reasoning_trace_count") is not None
            else None
        )
        reward_metric = _normalize_reward_metric(request.query_params.get("reward_metric"))
        return templates.TemplateResponse(
            request,
            "unsupervised_run.html",
            {
                "page_title": f"{env_id} automatic {run_id}",
                "request": request,
                "run": run,
                "run_id": run_id,
                "env_id": env_id,
                "run_json_text": run_json_text,
                "source_comparison": source_comparison,
                "combined_comparison": combined_comparison,
                "source_supervised_run_id": source_supervised_run_id,
                "source_evaluation_run_id": source_evaluation_run_id,
                "requested_proposed_label_count": requested_proposed_label_count,
                "requested_reasoning_trace_count": requested_reasoning_trace_count,
                "learning_settings": learning_settings,
                "evaluation_scores_by_model": (
                    _evaluation_scores_by_model(run.evaluation_runs)
                    if run is not None
                    else []
                ),
                "current_reward_metric": reward_metric,
                "reward_metric_toggle": {
                    "train_url": _updated_query_url(request, reward_metric="train"),
                    "validation_url": _updated_query_url(request, reward_metric="validation"),
                },
                "run_job": _job_payload(run_job, layout=resolved_layout) if run_job is not None else None,
                "run_job_poll_url": f"/jobs/{run_job.job_id}/status.json" if run_job is not None else None,
                "run_refresh_token": _run_refresh_token(run.root) if run is not None else None,
                "worker_update_action": f"/automatic/{env_id}/{run_id}/workers",
                "mark_label_done_action": f"/automatic/{env_id}/{run_id}/mark-label-done",
                "run_route_base": "/automatic",
                "run_kind_label": "automatic",
                "nav_links": _nav_links(active="env", env_id=env_id),
            },
        )

    @app.get("/automatic/{env_id}/{run_id}/llm-calls/{sequence}", response_class=HTMLResponse)
    async def unsupervised_llm_call_view(
        request: Request,
        env_id: str,
        run_id: str,
        sequence: int,
    ) -> HTMLResponse:
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        try:
            run = load_automatic_library_detail(env_id, run_id, layout=resolved_layout)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Automatic run not found: {env_id}/{run_id}",
            ) from exc
        call = next((item for item in run.llm_calls if item.sequence == sequence), None)
        if call is None:
            raise HTTPException(
                status_code=404,
                detail=f"LLM call not found for automatic run {run_id}: {sequence}",
            )
        run_url = _automatic_run_url(env_id, run_id)
        return templates.TemplateResponse(
            request,
            "llm_call.html",
            {
                "page_title": f"{env_id} automatic {run_id} llm call {sequence}",
                "request": request,
                "env_id": env_id,
                "run_id": run_id,
                "run": run,
                "run_url": run_url,
                "run_kind": AUTOMATIC_WORKFLOW_KIND,
                "call": call,
                "run_route_base": "/automatic",
                "run_kind_label": "automatic",
                "nav_links": _nav_links(active="env", env_id=env_id),
            },
        )

    @app.post("/automatic/{env_id}/{run_id}/delete")
    async def delete_unsupervised_run(request: Request, env_id: str, run_id: str):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        _validate_path_segment(run_id, label="run id")
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", [f"/env/{env_id}"])[0] or f"/env/{env_id}"
        run_root = (
            resolved_layout.data_root
            / "patterns"
            / env_id
            / AUTOMATIC_RUNS_DIRNAME
            / run_id
        )
        _delete_tree(
            run_root,
            layout=resolved_layout,
            job_manager=dashboard_job_manager,
        )
        return RedirectResponse(url=next_url, status_code=303)

    @app.post("/automatic/{env_id}/{run_id}/workers")
    async def update_unsupervised_run_workers(request: Request, env_id: str, run_id: str):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        _validate_path_segment(run_id, label="run id")
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", [f"/automatic/{env_id}/{run_id}"])[0] or f"/automatic/{env_id}/{run_id}"
        try:
            workers = int(parsed_form.get("workers", ["1"])[0] or "1")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Workers must be an integer.") from exc
        run_root = (
            resolved_layout.data_root
            / "patterns"
            / env_id
            / AUTOMATIC_RUNS_DIRNAME
            / run_id
        )
        if not run_root.is_dir():
            raise HTTPException(status_code=404, detail=f"Automatic run not found: {env_id}/{run_id}")
        _update_learning_run_worker_target(run_root, workers)
        _restart_learning_job_for_control_update(
            env_id=env_id,
            run_id=run_id,
            kind=AUTOMATIC_WORKFLOW_KIND,
            job_manager=dashboard_job_manager,
        )
        return RedirectResponse(url=next_url, status_code=303)

    @app.post("/automatic/{env_id}/{run_id}/mark-label-done")
    async def mark_unsupervised_run_label_done(
        request: Request, env_id: str, run_id: str
    ):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        _validate_path_segment(run_id, label="run id")
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", [f"/automatic/{env_id}/{run_id}"])[0] or f"/automatic/{env_id}/{run_id}"
        label_name = parsed_form.get("label", [""])[0]
        raw_label_index = parsed_form.get("label_index", [""])[0]
        label_index: int | None = None
        if raw_label_index:
            try:
                label_index = int(raw_label_index)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Label index must be an integer.") from exc
        run_root = (
            resolved_layout.data_root
            / "patterns"
            / env_id
            / AUTOMATIC_RUNS_DIRNAME
            / run_id
        )
        if not run_root.is_dir():
            raise HTTPException(status_code=404, detail=f"Automatic run not found: {env_id}/{run_id}")
        try:
            _mark_learning_label_done(
                run_root,
                label_name=label_name,
                label_index=label_index,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _restart_learning_job_for_control_update(
            env_id=env_id,
            run_id=run_id,
            kind=AUTOMATIC_WORKFLOW_KIND,
            job_manager=dashboard_job_manager,
        )
        return RedirectResponse(url=next_url, status_code=303)

    @app.get("/unsupervised/{env_id}/{run_id}", response_class=HTMLResponse)
    async def true_unsupervised_run_view(
        request: Request, env_id: str, run_id: str, job: str | None = None
    ) -> HTMLResponse:
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        run_job = _find_learning_job_for_run(
            env_id=env_id,
            run_id=run_id,
            kind=UNSUPERVISED_WORKFLOW_KIND,
            requested_job_id=job,
            job_manager=dashboard_job_manager,
        )
        try:
            run = load_unsupervised_library_detail(env_id, run_id, layout=resolved_layout)
        except FileNotFoundError:
            run = None
        if run is None and run_job is None:
            raise HTTPException(status_code=404, detail=f"Unsupervised run not found: {env_id}/{run_id}")
        run_json_text = json.dumps(run.raw_run_payload, indent=2, sort_keys=True) if run is not None else ""
        request_payload = (
            run_job.metadata.get("request")
            if run_job is not None and isinstance(run_job.metadata.get("request"), dict)
            else None
        )
        learning_settings = _learning_settings(
            run_payload=run.raw_run_payload if run is not None else None,
            request_payload=request_payload,
        )
        reward_metric = _normalize_reward_metric(request.query_params.get("reward_metric"))
        requested_final_label_count = (
            int(request_payload.get("final_label_count"))
            if isinstance(request_payload, dict)
            and request_payload.get("final_label_count") is not None
            else None
        )
        requested_trace_count = (
            int(request_payload.get("trace_count"))
            if isinstance(request_payload, dict)
            and request_payload.get("trace_count") is not None
            else None
        )
        return templates.TemplateResponse(
            request,
            "unsupervised_run.html",
            {
                "page_title": f"{env_id} unsupervised {run_id}",
                "request": request,
                "run": run,
                "run_id": run_id,
                "env_id": env_id,
                "run_json_text": run_json_text,
                "source_comparison": None,
                "combined_comparison": run.evaluation_runs[0] if run is not None and run.evaluation_runs else None,
                "source_supervised_run_id": None,
                "source_evaluation_run_id": None,
                "requested_proposed_label_count": requested_final_label_count,
                "requested_reasoning_trace_count": requested_trace_count,
                "learning_settings": learning_settings,
                "evaluation_scores_by_model": (
                    _evaluation_scores_by_model(run.evaluation_runs)
                    if run is not None
                    else []
                ),
                "current_reward_metric": reward_metric,
                "reward_metric_toggle": {
                    "train_url": _updated_query_url(request, reward_metric="train"),
                    "validation_url": _updated_query_url(request, reward_metric="validation"),
                },
                "run_job": _job_payload(run_job, layout=resolved_layout) if run_job is not None else None,
                "run_job_poll_url": f"/jobs/{run_job.job_id}/status.json" if run_job is not None else None,
                "run_refresh_token": _run_refresh_token(run.root) if run is not None else None,
                "worker_update_action": f"/unsupervised/{env_id}/{run_id}/workers",
                "mark_label_done_action": f"/unsupervised/{env_id}/{run_id}/mark-label-done",
                "run_route_base": "/unsupervised",
                "run_kind_label": "unsupervised",
                "nav_links": _nav_links(active="env", env_id=env_id),
            },
        )

    @app.get("/unsupervised/{env_id}/{run_id}/llm-calls/{sequence}", response_class=HTMLResponse)
    async def true_unsupervised_llm_call_view(
        request: Request,
        env_id: str,
        run_id: str,
        sequence: int,
    ) -> HTMLResponse:
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(status_code=404, detail=f"Unsupported environment: {env_id}")
        try:
            run = load_unsupervised_library_detail(env_id, run_id, layout=resolved_layout)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Unsupervised run not found: {env_id}/{run_id}",
            ) from exc
        call = next((item for item in run.llm_calls if item.sequence == sequence), None)
        if call is None:
            raise HTTPException(
                status_code=404,
                detail=f"LLM call not found for unsupervised run {run_id}: {sequence}",
            )
        run_url = _unsupervised_run_url(env_id, run_id)
        return templates.TemplateResponse(
            request,
            "llm_call.html",
            {
                "page_title": f"{env_id} unsupervised {run_id} llm call {sequence}",
                "request": request,
                "env_id": env_id,
                "run_id": run_id,
                "run": run,
                "run_url": run_url,
                "run_kind": UNSUPERVISED_WORKFLOW_KIND,
                "call": call,
                "nav_links": _nav_links(active="env", env_id=env_id),
            },
        )

    @app.post("/unsupervised/{env_id}/{run_id}/delete")
    async def delete_true_unsupervised_run(request: Request, env_id: str, run_id: str):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(status_code=404, detail=f"Unsupported environment: {env_id}")
        _validate_path_segment(run_id, label="run id")
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", [f"/env/{env_id}"])[0] or f"/env/{env_id}"
        run_root = (
            resolved_layout.data_root
            / "patterns"
            / env_id
            / UNSUPERVISED_RUNS_DIRNAME
            / run_id
        )
        _delete_tree(run_root, layout=resolved_layout, job_manager=dashboard_job_manager)
        return RedirectResponse(url=next_url, status_code=303)

    @app.post("/unsupervised/{env_id}/{run_id}/workers")
    async def update_true_unsupervised_run_workers(request: Request, env_id: str, run_id: str):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(status_code=404, detail=f"Unsupported environment: {env_id}")
        _validate_path_segment(run_id, label="run id")
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", [f"/unsupervised/{env_id}/{run_id}"])[0] or f"/unsupervised/{env_id}/{run_id}"
        try:
            workers = int(parsed_form.get("workers", ["1"])[0] or "1")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Workers must be an integer.") from exc
        run_root = (
            resolved_layout.data_root
            / "patterns"
            / env_id
            / UNSUPERVISED_RUNS_DIRNAME
            / run_id
        )
        if not run_root.is_dir():
            raise HTTPException(status_code=404, detail=f"Unsupervised run not found: {env_id}/{run_id}")
        _update_learning_run_worker_target(run_root, workers)
        _restart_learning_job_for_control_update(
            env_id=env_id,
            run_id=run_id,
            kind=UNSUPERVISED_WORKFLOW_KIND,
            job_manager=dashboard_job_manager,
        )
        return RedirectResponse(url=next_url, status_code=303)

    @app.post("/unsupervised/{env_id}/{run_id}/mark-label-done")
    async def mark_true_unsupervised_run_label_done(request: Request, env_id: str, run_id: str):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(status_code=404, detail=f"Unsupported environment: {env_id}")
        _validate_path_segment(run_id, label="run id")
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", [f"/unsupervised/{env_id}/{run_id}"])[0] or f"/unsupervised/{env_id}/{run_id}"
        label_name = parsed_form.get("label", [""])[0]
        raw_label_index = parsed_form.get("label_index", [""])[0]
        label_index: int | None = None
        if raw_label_index:
            try:
                label_index = int(raw_label_index)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Label index must be an integer.") from exc
        run_root = (
            resolved_layout.data_root
            / "patterns"
            / env_id
            / UNSUPERVISED_RUNS_DIRNAME
            / run_id
        )
        if not run_root.is_dir():
            raise HTTPException(status_code=404, detail=f"Unsupervised run not found: {env_id}/{run_id}")
        try:
            _mark_learning_label_done(run_root, label_name=label_name, label_index=label_index)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _restart_learning_job_for_control_update(
            env_id=env_id,
            run_id=run_id,
            kind=UNSUPERVISED_WORKFLOW_KIND,
            job_manager=dashboard_job_manager,
        )
        return RedirectResponse(url=next_url, status_code=303)

    def _render_run_evaluations_page(
        request: Request,
        *,
        env_id: str,
        run_id: str,
        run: Any | None = None,
        evaluation_error: str | None = None,
        evaluation_values: dict[str, str] | None = None,
        selected_labels: list[str] | None = None,
        status_code: int = 200,
    ) -> HTMLResponse:
        if run is None:
            try:
                run = load_pattern_run_detail(env_id, run_id, layout=resolved_layout)
            except FileNotFoundError as exc:
                raise HTTPException(
                    status_code=404, detail=f"Pattern run not found: {env_id}/{run_id}"
                ) from exc
        available_labels = _run_label_names(run)
        enabled_label_set = set(selected_labels or available_labels)
        evaluation_values_payload = _default_evaluation_values(layout=resolved_layout)
        if evaluation_values is not None:
            evaluation_values_payload.update(evaluation_values)
        elif run is not None:
            learning_workers = _coerce_int(run.raw_run_payload.get("workers"))
            if learning_workers:
                evaluation_values_payload["workers"] = str(learning_workers)
        chosen_preset = str(evaluation_values_payload.get("benchmark_preset", "full"))
        if chosen_preset not in {"full", "quick-eval", "test"}:
            chosen_preset = "full"
        evaluation_values_payload["benchmark_preset"] = chosen_preset
        benchmark = summarize_deepphy_benchmark(
            env_id,
            preset=chosen_preset,
            task_limit=default_benchmark_task_limit(env_id, preset=chosen_preset),
        )
        recent_evaluation_jobs = _recent_evaluation_jobs(
            env_id=env_id,
            run_id=run_id,
            run=run,
            job_manager=dashboard_job_manager,
        )
        run, recent_evaluation_jobs = _normalize_evaluation_statuses(
            run=run,
            recent_evaluation_jobs=recent_evaluation_jobs,
        )
        latest_evaluation = run.evaluation_runs[0] if run.evaluation_runs else None
        return templates.TemplateResponse(
            request,
            "run_evaluations.html",
            {
                "page_title": f"{env_id} run {run_id} evaluations",
                "request": request,
                "env_id": env_id,
                "run_id": run_id,
                "run": run,
                "benchmark": benchmark,
                "evaluation_error": evaluation_error,
                "evaluation_values": evaluation_values_payload,
                "recent_evaluation_jobs": recent_evaluation_jobs,
                "latest_evaluation": latest_evaluation,
                "available_labels": available_labels,
                "enabled_labels": [
                    label for label in available_labels if label in enabled_label_set
                ],
                "nav_links": _nav_links(active="env", env_id=env_id),
            },
            status_code=status_code,
        )

    def _render_library_evaluations_page(
        request: Request,
        *,
        env_id: str,
        run_id: str,
        run: Any | None = None,
        evaluation_error: str | None = None,
        evaluation_values: dict[str, str] | None = None,
        status_code: int = 200,
    ) -> HTMLResponse:
        if run is None:
            try:
                run = load_pattern_run_detail(env_id, run_id, layout=resolved_layout)
            except FileNotFoundError as exc:
                raise HTTPException(
                    status_code=404, detail=f"Pattern run not found: {env_id}/{run_id}"
                ) from exc
        evaluation_values_payload = _default_library_evaluation_values()
        if evaluation_values is not None:
            evaluation_values_payload.update(evaluation_values)
        elif run is not None:
            learning_workers = _coerce_int(run.raw_run_payload.get("workers"))
            if learning_workers:
                evaluation_values_payload["workers"] = str(learning_workers)
        chosen_template = str(
            evaluation_values_payload.get(
                "template_name", DEFAULT_LIBRARY_HOLDOUT_EVALUATION_TEMPLATE
            )
        )
        if chosen_template not in LIBRARY_HOLDOUT_EVALUATION_TEMPLATES:
            chosen_template = DEFAULT_LIBRARY_HOLDOUT_EVALUATION_TEMPLATE
        evaluation_values_payload["template_name"] = chosen_template
        template = LIBRARY_HOLDOUT_EVALUATION_TEMPLATES[chosen_template]
        saved_evaluations = list_library_evaluations_for_run(run.root)
        recent_jobs = _recent_library_evaluation_jobs(
            env_id=env_id,
            run_id=run_id,
            job_manager=dashboard_job_manager,
        )
        return templates.TemplateResponse(
            request,
            "library_evaluations.html",
            {
                "page_title": f"{env_id} run {run_id} library evaluations",
                "request": request,
                "env_id": env_id,
                "run_id": run_id,
                "run": run,
                "evaluation_error": evaluation_error,
                "evaluation_values": evaluation_values_payload,
                "library_eval_template": template,
                "template_options": LIBRARY_HOLDOUT_EVALUATION_TEMPLATES,
                "saved_library_evaluations": saved_evaluations,
                "recent_library_evaluation_jobs": recent_jobs,
                "nav_links": _nav_links(active="env", env_id=env_id),
            },
            status_code=status_code,
        )

    @app.get("/runs/{env_id}/{run_id}/evaluations", response_class=HTMLResponse)
    async def run_evaluations_view(request: Request, env_id: str, run_id: str):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        return _render_run_evaluations_page(
            request,
            env_id=env_id,
            run_id=run_id,
        )

    @app.get(
        "/runs/{env_id}/{run_id}/library-evaluations",
        response_class=HTMLResponse,
    )
    async def library_evaluations_view(request: Request, env_id: str, run_id: str):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        return _render_library_evaluations_page(
            request,
            env_id=env_id,
            run_id=run_id,
        )

    @app.get(
        "/runs/{env_id}/{run_id}/evaluations/{evaluation_run_id}",
        response_class=HTMLResponse,
    )
    async def run_evaluation_detail_view(
        request: Request,
        env_id: str,
        run_id: str,
        evaluation_run_id: str,
    ) -> HTMLResponse:
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        _validate_path_segment(run_id, label="run id")
        _validate_path_segment(evaluation_run_id, label="evaluation run id")
        try:
            run = load_pattern_run_detail(env_id, run_id, layout=resolved_layout)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail=f"Pattern run not found: {env_id}/{run_id}"
            ) from exc
        recent_evaluation_jobs = _recent_evaluation_jobs(
            env_id=env_id,
            run_id=run_id,
            run=run,
            job_manager=dashboard_job_manager,
        )
        run, recent_evaluation_jobs = _normalize_evaluation_statuses(
            run=run,
            recent_evaluation_jobs=recent_evaluation_jobs,
        )
        selected_evaluation = next(
            (item for item in run.evaluation_runs if item.run_id == evaluation_run_id),
            None,
        )
        if selected_evaluation is None:
            raise HTTPException(
                status_code=404,
                detail=f"Evaluation not found for run {run_id}: {evaluation_run_id}",
            )
        evaluation_job = _find_evaluation_job_for_run(
            env_id=env_id,
            run_id=run_id,
            evaluation_run_id=evaluation_run_id,
            requested_job_id=request.query_params.get("job"),
            job_manager=dashboard_job_manager,
        )
        evaluation_status = (
            ("completed" if evaluation_job.status == "succeeded" else evaluation_job.status)
            if evaluation_job is not None
            else selected_evaluation.status
        )
        return templates.TemplateResponse(
            request,
            "evaluation_detail.html",
            {
                "page_title": f"{env_id} run {run_id} evaluation {evaluation_run_id}",
                "request": request,
                "env_id": env_id,
                "run_id": run_id,
                "run": run,
                "selected_evaluation": selected_evaluation,
                "recent_evaluation_jobs": recent_evaluation_jobs,
                "evaluation_job": _job_payload(evaluation_job, layout=resolved_layout)
                if evaluation_job is not None
                else None,
                "evaluation_job_next_url": f"/runs/{env_id}/{run_id}/evaluations/{evaluation_run_id}",
                "evaluation_job_poll_url": f"/jobs/{evaluation_job.job_id}/status.json"
                if evaluation_job is not None
                else None,
                "evaluation_status": evaluation_status,
                "run_route_base": "/automatic",
                "run_kind_label": "automatic",
                "nav_links": _nav_links(active="env", env_id=env_id),
            },
        )

    @app.get(
        "/runs/{env_id}/{run_id}/library-evaluations/{library_evaluation_run_id}",
        response_class=HTMLResponse,
    )
    async def library_evaluation_detail_view(
        request: Request,
        env_id: str,
        run_id: str,
        library_evaluation_run_id: str,
    ):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        try:
            run = load_pattern_run_detail(env_id, run_id, layout=resolved_layout)
            evaluation = load_library_evaluation_detail(
                env_id,
                run_id,
                library_evaluation_run_id,
                layout=resolved_layout,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        active_job = None
        job_id = request.query_params.get("job")
        if job_id:
            try:
                active_job = dashboard_job_manager.get_snapshot(job_id)
            except KeyError:
                active_job = None
        return templates.TemplateResponse(
            request,
            "library_evaluation_detail.html",
            {
                "page_title": (
                    f"{env_id} run {run_id} library evaluation {library_evaluation_run_id}"
                ),
                "request": request,
                "env_id": env_id,
                "run_id": run_id,
                "run": run,
                "evaluation": evaluation,
                "active_job": active_job,
                "nav_links": _nav_links(active="env", env_id=env_id),
            },
        )

    @app.get(
        "/automatic/{env_id}/{run_id}/evaluations/{evaluation_run_id}",
        response_class=HTMLResponse,
    )
    async def unsupervised_evaluation_detail_view(
        request: Request,
        env_id: str,
        run_id: str,
        evaluation_run_id: str,
    ) -> HTMLResponse:
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        _validate_path_segment(run_id, label="run id")
        _validate_path_segment(evaluation_run_id, label="evaluation run id")
        try:
            run = load_automatic_library_detail(env_id, run_id, layout=resolved_layout)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail=f"Automatic run not found: {env_id}/{run_id}"
            ) from exc
        recent_evaluation_jobs = _recent_evaluation_jobs(
            env_id=env_id,
            run_id=run_id,
            run=run,
            job_manager=dashboard_job_manager,
        )
        run, recent_evaluation_jobs = _normalize_evaluation_statuses(
            run=run,
            recent_evaluation_jobs=recent_evaluation_jobs,
        )
        selected_evaluation = next(
            (item for item in run.evaluation_runs if item.run_id == evaluation_run_id),
            None,
        )
        if selected_evaluation is None:
            raise HTTPException(
                status_code=404,
                detail=f"Evaluation not found for automatic run {run_id}: {evaluation_run_id}",
            )
        evaluation_job = _find_evaluation_job_for_run(
            env_id=env_id,
            run_id=run_id,
            evaluation_run_id=evaluation_run_id,
            requested_job_id=request.query_params.get("job"),
            job_manager=dashboard_job_manager,
        )
        evaluation_status = (
            ("completed" if evaluation_job.status == "succeeded" else evaluation_job.status)
            if evaluation_job is not None
            else selected_evaluation.status
        )
        return templates.TemplateResponse(
            request,
            "unsupervised_evaluation_detail.html",
            {
                "page_title": f"{env_id} automatic {run_id} evaluation {evaluation_run_id}",
                "request": request,
                "env_id": env_id,
                "run_id": run_id,
                "run": run,
                "selected_evaluation": selected_evaluation,
                "recent_evaluation_jobs": recent_evaluation_jobs,
                "evaluation_job": _job_payload(evaluation_job, layout=resolved_layout)
                if evaluation_job is not None
                else None,
                "evaluation_job_next_url": f"/automatic/{env_id}/{run_id}/evaluations/{evaluation_run_id}",
                "evaluation_job_poll_url": f"/jobs/{evaluation_job.job_id}/status.json"
                if evaluation_job is not None
                else None,
                "evaluation_status": evaluation_status,
                "nav_links": _nav_links(active="env", env_id=env_id),
            },
        )

    @app.get(
        "/runs/{env_id}/{run_id}/evaluations/{evaluation_run_id}/tasks/{task_token}",
        response_class=HTMLResponse,
    )
    async def evaluation_task_view(
        request: Request,
        env_id: str,
        run_id: str,
        evaluation_run_id: str,
        task_token: str,
    ) -> HTMLResponse:
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        _validate_path_segment(run_id, label="run id")
        _validate_path_segment(evaluation_run_id, label="evaluation run id")
        _validate_path_segment(task_token, label="task token")
        try:
            run = load_pattern_run_detail(env_id, run_id, layout=resolved_layout)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail=f"Pattern run not found: {env_id}/{run_id}"
            ) from exc
        evaluation = next(
            (item for item in run.evaluation_runs if item.run_id == evaluation_run_id),
            None,
        )
        if evaluation is None:
            raise HTTPException(
                status_code=404,
                detail=f"Evaluation not found for run {run_id}: {evaluation_run_id}",
            )
        task = next(
            (
                item
                for item in evaluation.task_summaries
                if item.task_token == task_token or item.task_id == task_token
            ),
            None,
        )
        task_detail = load_evaluation_task_detail(
            evaluation.root,
            task_id=task.task_id if task is not None else None,
            task_token=task.task_token if task is not None else task_token,
        )
        if task_detail is None:
            raise HTTPException(
                status_code=404,
                detail=f"Evaluation task not found: {task_token}",
            )
        selected_attempt = None
        if task_detail.records:
            requested_attempt = request.query_params.get("attempt")
            if requested_attempt is not None:
                try:
                    requested_index = int(requested_attempt)
                except ValueError:
                    requested_index = None
                if requested_index is not None:
                    selected_attempt = next(
                        (item for item in task_detail.records if item.index == requested_index),
                        None,
                    )
            if selected_attempt is None:
                selected_attempt = task_detail.records[-1]
        return templates.TemplateResponse(
            request,
            "evaluation_task.html",
            {
                "page_title": f"{env_id} {run_id} {evaluation_run_id} task {task_detail.task_id or task_token}",
                "request": request,
                "env_id": env_id,
                "run_id": run_id,
                "run": run,
                "evaluation": evaluation,
                "task": task_detail,
                "selected_attempt": selected_attempt,
                "run_route_base": "/automatic",
                "run_kind_label": "automatic",
                "nav_links": _nav_links(active="env", env_id=env_id),
            },
        )

    @app.get(
        "/automatic/{env_id}/{run_id}/evaluations/{evaluation_run_id}/tasks/{task_token}",
        response_class=HTMLResponse,
    )
    async def unsupervised_evaluation_task_view(
        request: Request,
        env_id: str,
        run_id: str,
        evaluation_run_id: str,
        task_token: str,
    ) -> HTMLResponse:
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        _validate_path_segment(run_id, label="run id")
        _validate_path_segment(evaluation_run_id, label="evaluation run id")
        _validate_path_segment(task_token, label="task token")
        try:
            run = load_automatic_library_detail(env_id, run_id, layout=resolved_layout)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail=f"Automatic run not found: {env_id}/{run_id}"
            ) from exc
        evaluation = next(
            (item for item in run.evaluation_runs if item.run_id == evaluation_run_id),
            None,
        )
        if evaluation is None:
            raise HTTPException(
                status_code=404,
                detail=f"Evaluation not found for automatic run {run_id}: {evaluation_run_id}",
            )
        task = next(
            (
                item
                for item in evaluation.task_summaries
                if item.task_token == task_token or item.task_id == task_token
            ),
            None,
        )
        task_detail = load_evaluation_task_detail(
            evaluation.root,
            task_id=task.task_id if task is not None else None,
            task_token=task.task_token if task is not None else task_token,
        )
        if task_detail is None:
            raise HTTPException(
                status_code=404,
                detail=f"Evaluation task not found: {task_token}",
            )
        selected_attempt = None
        if task_detail.records:
            requested_attempt = request.query_params.get("attempt")
            if requested_attempt is not None:
                try:
                    requested_index = int(requested_attempt)
                except ValueError:
                    requested_index = None
                if requested_index is not None:
                    selected_attempt = next(
                        (item for item in task_detail.records if item.index == requested_index),
                        None,
                    )
            if selected_attempt is None:
                selected_attempt = task_detail.records[-1]
        return templates.TemplateResponse(
            request,
            "unsupervised_evaluation_task.html",
            {
                "page_title": f"{env_id} automatic {run_id} {evaluation_run_id} task {task_detail.task_id or task_token}",
                "request": request,
                "env_id": env_id,
                "run_id": run_id,
                "run": run,
                "evaluation": evaluation,
                "task": task_detail,
                "selected_attempt": selected_attempt,
                "nav_links": _nav_links(active="env", env_id=env_id),
            },
        )

    @app.post("/runs/{env_id}/{run_id}/evaluations/{evaluation_run_id}/delete")
    async def delete_run_evaluation(
        request: Request,
        env_id: str,
        run_id: str,
        evaluation_run_id: str,
    ):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        _validate_path_segment(run_id, label="run id")
        _validate_path_segment(evaluation_run_id, label="evaluation run id")
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = (
            parsed_form.get("next", [f"/runs/{env_id}/{run_id}/evaluations"])[0]
            or f"/runs/{env_id}/{run_id}/evaluations"
        )
        eval_root = (
            resolved_layout.data_root
            / "patterns"
            / env_id
            / run_id
            / "deepphy_eval"
            / evaluation_run_id
        )
        _remove_jobs_for_evaluation(
            env_id=env_id,
            run_id=run_id,
            evaluation_run_id=evaluation_run_id,
            job_manager=dashboard_job_manager,
        )
        _delete_tree(
            eval_root,
            layout=resolved_layout,
            job_manager=dashboard_job_manager,
        )
        return RedirectResponse(url=next_url, status_code=303)

    @app.post("/automatic/{env_id}/{run_id}/evaluations/{evaluation_run_id}/delete")
    async def delete_unsupervised_run_evaluation(
        request: Request,
        env_id: str,
        run_id: str,
        evaluation_run_id: str,
    ):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        _validate_path_segment(run_id, label="run id")
        _validate_path_segment(evaluation_run_id, label="evaluation run id")
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = (
            parsed_form.get("next", [f"/automatic/{env_id}/{run_id}"])[0]
            or f"/automatic/{env_id}/{run_id}"
        )
        eval_root = (
            resolved_layout.data_root
            / "patterns"
            / env_id
            / AUTOMATIC_RUNS_DIRNAME
            / run_id
            / "deepphy_eval"
            / evaluation_run_id
        )
        _remove_jobs_for_evaluation(
            env_id=env_id,
            run_id=run_id,
            evaluation_run_id=evaluation_run_id,
            job_manager=dashboard_job_manager,
        )
        _delete_tree(
            eval_root,
            layout=resolved_layout,
            job_manager=dashboard_job_manager,
        )
        return RedirectResponse(url=next_url, status_code=303)

    @app.get(
        "/unsupervised/{env_id}/{run_id}/evaluations/{evaluation_run_id}",
        response_class=HTMLResponse,
    )
    async def true_unsupervised_evaluation_detail_view(
        request: Request,
        env_id: str,
        run_id: str,
        evaluation_run_id: str,
    ) -> HTMLResponse:
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(status_code=404, detail=f"Unsupported environment: {env_id}")
        _validate_path_segment(run_id, label="run id")
        _validate_path_segment(evaluation_run_id, label="evaluation run id")
        try:
            run = load_unsupervised_library_detail(env_id, run_id, layout=resolved_layout)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Unsupervised run not found: {env_id}/{run_id}",
            ) from exc
        recent_evaluation_jobs = _recent_evaluation_jobs(
            env_id=env_id,
            run_id=run_id,
            run=run,
            job_manager=dashboard_job_manager,
        )
        run, recent_evaluation_jobs = _normalize_evaluation_statuses(
            run=run,
            recent_evaluation_jobs=recent_evaluation_jobs,
        )
        selected_evaluation = next(
            (item for item in run.evaluation_runs if item.run_id == evaluation_run_id),
            None,
        )
        if selected_evaluation is None:
            raise HTTPException(
                status_code=404,
                detail=f"Evaluation not found for unsupervised run {run_id}: {evaluation_run_id}",
            )
        evaluation_job = _find_evaluation_job_for_run(
            env_id=env_id,
            run_id=run_id,
            evaluation_run_id=evaluation_run_id,
            requested_job_id=request.query_params.get("job"),
            job_manager=dashboard_job_manager,
        )
        evaluation_status = (
            ("completed" if evaluation_job.status == "succeeded" else evaluation_job.status)
            if evaluation_job is not None
            else selected_evaluation.status
        )
        return templates.TemplateResponse(
            request,
            "unsupervised_evaluation_detail.html",
            {
                "page_title": f"{env_id} unsupervised {run_id} evaluation {evaluation_run_id}",
                "request": request,
                "env_id": env_id,
                "run_id": run_id,
                "run": run,
                "selected_evaluation": selected_evaluation,
                "recent_evaluation_jobs": recent_evaluation_jobs,
                "evaluation_job": _job_payload(evaluation_job, layout=resolved_layout)
                if evaluation_job is not None
                else None,
                "evaluation_job_next_url": f"/unsupervised/{env_id}/{run_id}/evaluations/{evaluation_run_id}",
                "evaluation_job_poll_url": f"/jobs/{evaluation_job.job_id}/status.json"
                if evaluation_job is not None
                else None,
                "evaluation_status": evaluation_status,
                "run_route_base": "/unsupervised",
                "run_kind_label": "unsupervised",
                "nav_links": _nav_links(active="env", env_id=env_id),
            },
        )

    @app.get(
        "/unsupervised/{env_id}/{run_id}/evaluations/{evaluation_run_id}/tasks/{task_token}",
        response_class=HTMLResponse,
    )
    async def true_unsupervised_evaluation_task_view(
        request: Request,
        env_id: str,
        run_id: str,
        evaluation_run_id: str,
        task_token: str,
    ) -> HTMLResponse:
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(status_code=404, detail=f"Unsupported environment: {env_id}")
        _validate_path_segment(run_id, label="run id")
        _validate_path_segment(evaluation_run_id, label="evaluation run id")
        _validate_path_segment(task_token, label="task token")
        try:
            run = load_unsupervised_library_detail(env_id, run_id, layout=resolved_layout)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Unsupervised run not found: {env_id}/{run_id}",
            ) from exc
        evaluation = next(
            (item for item in run.evaluation_runs if item.run_id == evaluation_run_id),
            None,
        )
        if evaluation is None:
            raise HTTPException(
                status_code=404,
                detail=f"Evaluation not found for unsupervised run {run_id}: {evaluation_run_id}",
            )
        task = next(
            (
                item
                for item in evaluation.task_summaries
                if item.task_token == task_token or item.task_id == task_token
            ),
            None,
        )
        task_detail = load_evaluation_task_detail(
            evaluation.root,
            task_id=task.task_id if task is not None else None,
            task_token=task.task_token if task is not None else task_token,
        )
        if task_detail is None:
            raise HTTPException(status_code=404, detail=f"Evaluation task not found: {task_token}")
        selected_attempt = None
        if task_detail.records:
            requested_attempt = request.query_params.get("attempt")
            if requested_attempt is not None:
                try:
                    requested_index = int(requested_attempt)
                except ValueError:
                    requested_index = None
                if requested_index is not None:
                    selected_attempt = next(
                        (item for item in task_detail.records if item.index == requested_index),
                        None,
                    )
            if selected_attempt is None:
                selected_attempt = task_detail.records[-1]
        return templates.TemplateResponse(
            request,
            "unsupervised_evaluation_task.html",
            {
                "page_title": f"{env_id} unsupervised {run_id} {evaluation_run_id} task {task_detail.task_id or task_token}",
                "request": request,
                "env_id": env_id,
                "run_id": run_id,
                "run": run,
                "evaluation": evaluation,
                "task": task_detail,
                "selected_attempt": selected_attempt,
                "run_route_base": "/unsupervised",
                "run_kind_label": "unsupervised",
                "nav_links": _nav_links(active="env", env_id=env_id),
            },
        )

    @app.post("/unsupervised/{env_id}/{run_id}/evaluations/{evaluation_run_id}/delete")
    async def delete_true_unsupervised_run_evaluation(
        request: Request,
        env_id: str,
        run_id: str,
        evaluation_run_id: str,
    ):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(status_code=404, detail=f"Unsupported environment: {env_id}")
        _validate_path_segment(run_id, label="run id")
        _validate_path_segment(evaluation_run_id, label="evaluation run id")
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = (
            parsed_form.get("next", [f"/unsupervised/{env_id}/{run_id}"])[0]
            or f"/unsupervised/{env_id}/{run_id}"
        )
        eval_root = (
            resolved_layout.data_root
            / "patterns"
            / env_id
            / UNSUPERVISED_RUNS_DIRNAME
            / run_id
            / "deepphy_eval"
            / evaluation_run_id
        )
        _remove_jobs_for_evaluation(
            env_id=env_id,
            run_id=run_id,
            evaluation_run_id=evaluation_run_id,
            job_manager=dashboard_job_manager,
        )
        _delete_tree(eval_root, layout=resolved_layout, job_manager=dashboard_job_manager)
        return RedirectResponse(url=next_url, status_code=303)

    @app.get(
        "/unsupervised/{env_id}/{run_id}/evaluations/{evaluation_run_id}/tasks/{task_token}/media/{filename}"
    )
    async def true_unsupervised_evaluation_task_media(
        env_id: str,
        run_id: str,
        evaluation_run_id: str,
        task_token: str,
        filename: str,
    ):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(status_code=404, detail=f"Unsupported environment: {env_id}")
        _validate_path_segment(run_id, label="run id")
        _validate_path_segment(evaluation_run_id, label="evaluation run id")
        _validate_path_segment(task_token, label="task token")
        if "/" in filename or "\\" in filename:
            raise HTTPException(status_code=404, detail=f"Unsupported media file: {filename}")
        task_root = (
            resolved_layout.data_root
            / "patterns"
            / env_id
            / UNSUPERVISED_RUNS_DIRNAME
            / run_id
            / "deepphy_eval"
            / evaluation_run_id
            / "tasks"
            / env_id
            / task_token
        )
        if not task_root.is_dir():
            raise HTTPException(status_code=404, detail=f"Evaluation task not found: {task_token}")
        attempt_roots = sorted(
            (
                path
                for path in task_root.iterdir()
                if path.is_dir() and (path.name.startswith("attempt_") or path.name.startswith("step_"))
            ),
            key=lambda path: path.name,
        )
        if not attempt_roots:
            raise HTTPException(status_code=404, detail=f"No attempts found for evaluation task: {task_token}")
        if filename.endswith(".png"):
            initial_dir = task_root / "initial"
            media_path = initial_dir / filename if initial_dir.is_dir() else attempt_roots[-1] / filename
        elif filename == "simulation.mp4":
            media_path = attempt_roots[-1] / filename
        else:
            raise HTTPException(status_code=404, detail=f"Unsupported media file: {filename}")
        if not media_path.is_file():
            raise HTTPException(status_code=404, detail=f"{filename} not found for evaluation task: {task_token}")
        media_type = "image/png" if filename.endswith(".png") else "video/mp4"
        return FileResponse(media_path, media_type=media_type)

    @app.post("/runs/{env_id}/{run_id}/evaluations/launch")
    async def launch_run_evaluation(request: Request, env_id: str, run_id: str):
        try:
            run = load_pattern_run_detail(env_id, run_id, layout=resolved_layout)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail=f"Pattern run not found: {env_id}/{run_id}"
            ) from exc

        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        enqueue = _launch_mode_from_parsed_form(parsed_form) == "queue"

        def _first(name: str, default: str = "") -> str:
            values = parsed_form.get(name)
            if not values:
                return default
            return values[0]

        llm_defaults = _load_dashboard_llm_defaults(resolved_layout)
        available_labels = _run_label_names(run)
        enabled_labels = sorted(
            {
                label.strip()
                for label in parsed_form.get("enabled_label", [])
                if label.strip()
            }
        )

        eval_values = {
            "benchmark_preset": _first("benchmark_preset", "full"),
            "prompt_format": "WM",
            "model": _first("model", ""),
            "ollama_base_url": _first(
                "ollama_base_url", llm_defaults["ollama_base_url"]
            ),
            "model_timeout": _first("model_timeout", llm_defaults["model_timeout"]),
            "capture_visuals": True,
        }
        invalid_labels = [label for label in enabled_labels if label not in set(available_labels)]
        if invalid_labels:
            return _render_run_evaluations_page(
                request,
                env_id=env_id,
                run_id=run_id,
                run=run,
                evaluation_error="Unknown labels selected: " + ", ".join(invalid_labels),
                evaluation_values=eval_values,
                selected_labels=enabled_labels,
                status_code=422,
            )
        if not enabled_labels:
            return _render_run_evaluations_page(
                request,
                env_id=env_id,
                run_id=run_id,
                run=run,
                evaluation_error="At least one label must remain enabled for evaluation.",
                evaluation_values=eval_values,
                selected_labels=enabled_labels,
                status_code=422,
            )

        evaluation_run_id = _coerce_optional(_first("evaluation_run_id")) or default_run_id()
        use_filtered_library = len(enabled_labels) != len(available_labels)
        filtered_library_path = None
        if use_filtered_library:
            try:
                filtered_library_path = _materialize_filtered_evaluation_library(
                    run=run,
                    evaluation_run_id=evaluation_run_id,
                    enabled_labels=enabled_labels,
                    available_labels=available_labels,
                )
            except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
                return _render_run_evaluations_page(
                    request,
                    env_id=env_id,
                    run_id=run_id,
                    run=run,
                    evaluation_error=str(exc),
                    evaluation_values=eval_values,
                    selected_labels=enabled_labels,
                    status_code=422,
                )

        request_payload = {
            "env_id": env_id,
            "benchmark_preset": _first("benchmark_preset", "full"),
            "prompt_format": "WM",
            "model": _first("model", ""),
            "description_mode": "pattern_text",
            "ollama_base_url": _coerce_optional(_first("ollama_base_url"))
            or llm_defaults["ollama_base_url"],
            "model_timeout": _coerce_optional(_first("model_timeout"))
            or llm_defaults["model_timeout"],
            "capture_visuals": True,
            "run_id": evaluation_run_id,
            "workers": _coerce_optional(_first("workers")) or 1,
        }
        if filtered_library_path is not None:
            request_payload["pattern_library"] = filtered_library_path
        else:
            request_payload["pattern_run_dir"] = run.root

        try:
            action_request = DeepPHYEvaluationActionRequest.model_validate(
                {
                    key: value
                    for key, value in request_payload.items()
                    if value is not None
                }
            )
        except (ValidationError, ValueError) as exc:
            if isinstance(exc, ValidationError):
                message = _format_validation_error(exc)
            else:
                message = str(exc)
            return _render_run_evaluations_page(
                request,
                env_id=env_id,
                run_id=run_id,
                run=run,
                evaluation_error=message,
                evaluation_values=eval_values,
                selected_labels=enabled_labels,
                status_code=422,
            )

        try:
            eval_python = python_executable_for_env(env_id)
        except FileNotFoundError as exc:
            return _render_run_evaluations_page(
                request,
                env_id=env_id,
                run_id=run_id,
                run=run,
                evaluation_error=str(exc),
                evaluation_values=eval_values,
                selected_labels=enabled_labels,
                status_code=422,
            )

        _resolved_pattern_run_dir, evaluation_run_dir, chosen_eval_run_id = resolve_deepphy_paths(
            action_request
        )
        snapshot = dashboard_job_manager.launch(
            name="evaluate-deepphy",
            argv=[str(eval_python), *build_deepphy_eval_argv(action_request)],
            metadata={
                "action": "evaluate-deepphy",
                "supports_pause": True,
                "env_id": env_id,
                "run_id": run_id,
                "evaluation_run_id": chosen_eval_run_id,
                "evaluation_run_dir": str(evaluation_run_dir)
                if evaluation_run_dir is not None
                else None,
                "pattern_run_dir": str(run.root),
                "benchmark_preset": action_request.benchmark_preset,
                "benchmark_task_count": summarize_deepphy_benchmark(
                    env_id,
                    preset=action_request.benchmark_preset,
                    task_limit=action_request.task_limit,
                ).task_count,
                "enabled_labels": enabled_labels,
                "disabled_labels": [
                    label for label in available_labels if label not in set(enabled_labels)
                ],
                "request": action_request.model_dump(mode="json", exclude_none=True),
                "resume_argv": [str(eval_python), *build_deepphy_eval_argv(action_request)],
            },
            enqueue=enqueue,
        )
        return RedirectResponse(
            url=f"/runs/{env_id}/{run_id}/evaluations?job={snapshot.job_id}",
            status_code=303,
        )

    @app.post("/runs/{env_id}/{run_id}/library-evaluations/launch")
    async def launch_library_evaluation(request: Request, env_id: str, run_id: str):
        try:
            run = load_pattern_run_detail(env_id, run_id, layout=resolved_layout)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail=f"Pattern run not found: {env_id}/{run_id}"
            ) from exc

        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)

        def _first(name: str, default: str = "") -> str:
            values = parsed_form.get(name)
            if not values:
                return default
            return values[0]

        llm_defaults = _load_dashboard_llm_defaults(resolved_layout)
        evaluation_values = {
            "template_name": _first(
                "template_name",
                DEFAULT_LIBRARY_HOLDOUT_EVALUATION_TEMPLATE,
            ),
            "workers": _first("workers", "1"),
            "model": _first("model", ""),
        }

        request_payload = {
            "env_id": env_id,
            "pattern_run_dir": run.root,
            "template_name": evaluation_values["template_name"],
            "workers": _coerce_optional(evaluation_values["workers"]) or "1",
            "model": evaluation_values["model"],
            "ollama_base_url": llm_defaults["ollama_base_url"],
            "model_timeout": llm_defaults["model_timeout"],
            "run_id": _coerce_optional(_first("library_evaluation_run_id")),
        }
        try:
            action_request = LibraryHoldoutEvaluationActionRequest.model_validate(
                request_payload
            )
        except (ValidationError, ValueError) as exc:
            message = _format_validation_error(exc) if isinstance(exc, ValidationError) else str(exc)
            return _render_library_evaluations_page(
                request,
                env_id=env_id,
                run_id=run_id,
                run=run,
                evaluation_error=message,
                evaluation_values=evaluation_values,
                status_code=422,
            )

        try:
            eval_python = python_executable_for_env(env_id)
        except FileNotFoundError as exc:
            return _render_library_evaluations_page(
                request,
                env_id=env_id,
                run_id=run_id,
                run=run,
                evaluation_error=str(exc),
                evaluation_values=evaluation_values,
                status_code=422,
            )

        library_evaluation_run_id = action_request.run_id or default_run_id()
        library_evaluation_run_dir = (
            run.root / LIBRARY_EVALUATIONS_DIRNAME / library_evaluation_run_id
        )
        if library_evaluation_run_dir.exists():
            return _render_library_evaluations_page(
                request,
                env_id=env_id,
                run_id=run_id,
                run=run,
                evaluation_error=f"Library evaluation already exists: {library_evaluation_run_id}",
                evaluation_values=evaluation_values,
                status_code=422,
            )
        library_evaluation_run_dir.mkdir(parents=True, exist_ok=False)
        source_library = load_pattern_library(run.library_path or run.root)
        available_labels = _run_label_names(run)
        from dashboard.workflows import (
            _library_holdout_manifest,
            _materialize_library_holdout_filtered_library,
            _write_json,
            _write_library_holdout_summary,
        )

        manifest = _library_holdout_manifest(
            available_labels=available_labels,
            template_name=action_request.template_name,
            benchmark_preset=action_request.benchmark_preset or "quick-eval",
            seed_count=action_request.seed_count or 5,
            single_holdout_repeat_count=action_request.single_holdout_repeat_count or 1,
            pair_holdout_size=action_request.pair_holdout_size or 2,
            pair_holdout_sample_count=action_request.pair_holdout_sample_count or 10,
        )
        _write_json(library_evaluation_run_dir / "manifest.json", manifest)
        _write_json(
            library_evaluation_run_dir / "request.json",
            action_request.model_dump(mode="json", exclude_none=True),
        )
        _write_library_holdout_summary(library_evaluation_run_dir, manifest)
        launched_snapshots: list[DashboardJobSnapshot] = []
        for case in manifest.get("cases", []):
            if not isinstance(case, dict):
                continue
            case_id = str(case.get("case_id") or "").strip()
            if not case_id:
                continue
            held_out_labels = [
                label.strip()
                for label in case.get("held_out_labels", [])
                if isinstance(label, str) and label.strip()
            ]
            enabled_labels = [
                label.strip()
                for label in case.get("enabled_labels", [])
                if isinstance(label, str) and label.strip()
            ]
            filtered_library_path = None
            if held_out_labels:
                filtered_library_path = _materialize_library_holdout_filtered_library(
                    source_run_dir=run.root,
                    source_library=source_library,
                    eval_root=library_evaluation_run_dir,
                    case_id=case_id,
                    available_labels=available_labels,
                    enabled_labels=enabled_labels,
                )
            for seed_run in case.get("seed_runs", []):
                if not isinstance(seed_run, dict):
                    continue
                seed = seed_run.get("seed")
                evaluation_run_id = seed_run.get("evaluation_run_id")
                relative_root = seed_run.get("relative_root")
                if not isinstance(seed, int):
                    continue
                if not isinstance(evaluation_run_id, str) or not evaluation_run_id.strip():
                    continue
                if not isinstance(relative_root, str) or not relative_root.strip():
                    continue
                child_root = library_evaluation_run_dir / relative_root
                child_request_payload = {
                    "env_id": env_id,
                    "benchmark_preset": action_request.benchmark_preset or "quick-eval",
                    "benchmark_seed": seed,
                    "workers": action_request.workers,
                    "prompt_format": "WM",
                    "model": action_request.model,
                    "description_mode": "pattern_text",
                    "ollama_base_url": action_request.ollama_base_url,
                    "model_timeout": action_request.model_timeout,
                    "capture_visuals": True,
                    "run_id": evaluation_run_id.strip(),
                    "output_dir": child_root,
                }
                if filtered_library_path is not None:
                    child_request_payload["pattern_library"] = filtered_library_path
                else:
                    child_request_payload["pattern_run_dir"] = run.root
                child_request = DeepPHYEvaluationActionRequest.model_validate(
                    child_request_payload
                )
                launched_snapshots.append(
                    dashboard_job_manager.launch(
                        name="evaluate-deepphy",
                        argv=[str(eval_python), *build_deepphy_eval_argv(child_request)],
                        metadata={
                            "action": "evaluate-library-case",
                            "supports_pause": True,
                            "env_id": env_id,
                            "run_id": run_id,
                            "pattern_run_dir": str(run.root),
                            "library_evaluation_run_id": library_evaluation_run_id,
                            "library_evaluation_run_dir": str(library_evaluation_run_dir),
                            "library_evaluation_case_id": case_id,
                            "library_evaluation_case_label": case.get("label"),
                            "evaluation_run_id": evaluation_run_id.strip(),
                            "evaluation_run_dir": str(child_root),
                            "benchmark_preset": child_request.benchmark_preset,
                            "benchmark_task_count": summarize_deepphy_benchmark(
                                env_id,
                                preset=child_request.benchmark_preset,
                                task_limit=child_request.task_limit,
                            ).task_count,
                            "benchmark_seed": seed,
                            "enabled_labels": enabled_labels,
                            "disabled_labels": held_out_labels,
                            "request": child_request.model_dump(mode="json", exclude_none=True),
                            "resume_argv": [str(eval_python), *build_deepphy_eval_argv(child_request)],
                        },
                        enqueue=True,
                    )
                )
        snapshot = launched_snapshots[0] if launched_snapshots else None
        return RedirectResponse(
            url=(
                f"/runs/{env_id}/{run_id}/library-evaluations"
                + (f"?job={snapshot.job_id}" if snapshot is not None else "")
            ),
            status_code=303,
        )

    @app.get(
        "/runs/{env_id}/{run_id}/evaluations/{evaluation_run_id}/tasks/{task_token}/media/{filename}"
    )
    async def evaluation_task_media(
        env_id: str,
        run_id: str,
        evaluation_run_id: str,
        task_token: str,
        filename: str,
    ):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        _validate_path_segment(run_id, label="run id")
        _validate_path_segment(evaluation_run_id, label="evaluation run id")
        _validate_path_segment(task_token, label="task token")
        if "/" in filename or "\\" in filename:
            raise HTTPException(status_code=404, detail=f"Unsupported media file: {filename}")
        task_root = (
            resolved_layout.data_root
            / "patterns"
            / env_id
            / run_id
            / "deepphy_eval"
            / evaluation_run_id
            / "tasks"
            / env_id
            / task_token
        )
        if not task_root.is_dir():
            raise HTTPException(status_code=404, detail=f"Evaluation task not found: {task_token}")
        attempt_roots = sorted(
            (
                path
                for path in task_root.iterdir()
                if path.is_dir() and (path.name.startswith("attempt_") or path.name.startswith("step_"))
            ),
            key=lambda path: path.name,
        )
        if not attempt_roots:
            raise HTTPException(status_code=404, detail=f"No attempts found for evaluation task: {task_token}")
        if filename.endswith(".png"):
            initial_dir = task_root / "initial"
            media_path = initial_dir / filename if initial_dir.is_dir() else attempt_roots[-1] / filename
        elif filename == "simulation.mp4":
            media_path = attempt_roots[-1] / filename
        else:
            raise HTTPException(status_code=404, detail=f"Unsupported media file: {filename}")
        if not media_path.is_file():
            raise HTTPException(status_code=404, detail=f"{filename} not found for evaluation task: {task_token}")
        media_type = "image/png" if filename.endswith(".png") else "video/mp4"
        return FileResponse(media_path, media_type=media_type)

    @app.get(
        "/automatic/{env_id}/{run_id}/evaluations/{evaluation_run_id}/tasks/{task_token}/media/{filename}"
    )
    async def unsupervised_evaluation_task_media(
        env_id: str,
        run_id: str,
        evaluation_run_id: str,
        task_token: str,
        filename: str,
    ):
        if env_id not in SUPPORTED_ENVS:
            raise HTTPException(
                status_code=404, detail=f"Unsupported environment: {env_id}"
            )
        _validate_path_segment(run_id, label="run id")
        _validate_path_segment(evaluation_run_id, label="evaluation run id")
        _validate_path_segment(task_token, label="task token")
        if "/" in filename or "\\" in filename:
            raise HTTPException(status_code=404, detail=f"Unsupported media file: {filename}")
        task_root = (
            resolved_layout.data_root
            / "patterns"
            / env_id
            / AUTOMATIC_RUNS_DIRNAME
            / run_id
            / "deepphy_eval"
            / evaluation_run_id
            / "tasks"
            / env_id
            / task_token
        )
        if not task_root.is_dir():
            raise HTTPException(status_code=404, detail=f"Evaluation task not found: {task_token}")
        attempt_roots = sorted(
            (
                path
                for path in task_root.iterdir()
                if path.is_dir() and (path.name.startswith("attempt_") or path.name.startswith("step_"))
            ),
            key=lambda path: path.name,
        )
        if not attempt_roots:
            raise HTTPException(status_code=404, detail=f"No attempts found for evaluation task: {task_token}")
        if filename.endswith(".png"):
            initial_dir = task_root / "initial"
            media_path = initial_dir / filename if initial_dir.is_dir() else attempt_roots[-1] / filename
        elif filename == "simulation.mp4":
            media_path = attempt_roots[-1] / filename
        else:
            raise HTTPException(status_code=404, detail=f"Unsupported media file: {filename}")
        if not media_path.is_file():
            raise HTTPException(status_code=404, detail=f"{filename} not found for evaluation task: {task_token}")
        media_type = "image/png" if filename.endswith(".png") else "video/mp4"
        return FileResponse(media_path, media_type=media_type)

    def _render_run_page(
        request: Request,
        *,
        env_id: str,
        run_id: str,
        job_id: str | None = None,
        run: Any | None = None,
        evaluation_error: str | None = None,
        evaluation_values: dict[str, str] | None = None,
        status_code: int = 200,
    ) -> HTMLResponse:
        run_job = _find_learning_job_for_run(
            env_id=env_id,
            run_id=run_id,
            kind="supervised",
            requested_job_id=job_id,
            job_manager=dashboard_job_manager,
        )
        if run is None:
            try:
                run = load_pattern_run_detail(env_id, run_id, layout=resolved_layout)
            except FileNotFoundError:
                run = None
        if run is None and run_job is None:
            raise HTTPException(
                status_code=404, detail=f"Pattern run not found: {env_id}/{run_id}"
            )

        request_payload = (
            run_job.metadata.get("request")
            if run_job is not None and isinstance(run_job.metadata.get("request"), dict)
            else None
        )
        learning_settings = _learning_settings(
            run_payload=run.raw_run_payload if run is not None else None,
            request_payload=request_payload,
        )
        reward_metric = _normalize_reward_metric(request.query_params.get("reward_metric"))
        selected_evolution_label = _select_evolution_label(
            run.label_telemetry if run is not None else [],
            request.query_params.get("evolution_label"),
        )

        return templates.TemplateResponse(
            request,
            "run.html",
            {
                "page_title": f"{env_id} run {run_id}",
                "request": request,
                "run": run,
                "run_id": run_id,
                "env_id": env_id,
                "learning_settings": learning_settings,
                "evaluation_scores_by_model": (
                    _evaluation_scores_by_model(run.evaluation_runs)
                    if run is not None
                    else []
                ),
                "current_reward_metric": reward_metric,
                "reward_metric_toggle": {
                    "train_url": _updated_query_url(request, reward_metric="train"),
                    "validation_url": _updated_query_url(request, reward_metric="validation"),
                },
                "selected_evolution_label": selected_evolution_label,
                "run_job": _job_payload(run_job, layout=resolved_layout) if run_job is not None else None,
                "run_job_poll_url": f"/jobs/{run_job.job_id}/status.json" if run_job is not None else None,
                "run_refresh_token": _run_refresh_token(run.root) if run is not None else None,
                "worker_update_action": f"/runs/{env_id}/{run_id}/workers",
                "mark_label_done_action": f"/runs/{env_id}/{run_id}/mark-label-done",
                "nav_links": _nav_links(active="env", env_id=env_id),
            },
            status_code=status_code,
        )

    @app.get(
        "/runs/{env_id}/{run_id}/labels/{label_name}",
        response_class=HTMLResponse,
    )
    async def label_view(
        request: Request,
        env_id: str,
        run_id: str,
        label_name: str,
    ) -> HTMLResponse:
        try:
            run_detail = load_pattern_run_detail(
                env_id,
                run_id,
                layout=resolved_layout,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        label_summary = next(
            (item for item in run_detail.label_telemetry if item.name == label_name),
            None,
        )
        if label_summary is None:
            raise HTTPException(
                status_code=404,
                detail=f"Label not found for run {run_id}: {label_name}",
            )
        run_job = _find_learning_job_for_run(
            env_id=env_id,
            run_id=run_id,
            kind="supervised",
            requested_job_id=None,
            job_manager=dashboard_job_manager,
        )
        progress = None
        provisional_label = None
        provisional_events: list[dict[str, Any]] = []
        if run_job is not None:
            run_job_payload = _job_payload(run_job, layout=resolved_layout)
            progress = (
                run_job_payload.get("learning_progress", {}).get("progress_summary")
                if isinstance(run_job_payload.get("learning_progress"), dict)
                else None
            )
            if isinstance(progress, dict):
                provisional_label_map = progress.get("provisional_labels_by_key", {})
                if isinstance(provisional_label_map, dict):
                    provisional_label = provisional_label_map.get(label_name)
                provisional_events = [
                    event
                    for event in progress.get("provisional_events", [])
                    if isinstance(event, dict) and event.get("label_key") == label_name
                ]
        reward_metric = _normalize_reward_metric(request.query_params.get("reward_metric"))
        return templates.TemplateResponse(
            request,
            "label.html",
            {
                "page_title": f"{env_id} {label_name}",
                "request": request,
                "env_id": env_id,
                "run_id": run_id,
                "run_detail": run_detail,
                "label_summary": label_summary,
                "provisional_label": provisional_label,
                "provisional_events": provisional_events,
                "current_reward_metric": reward_metric,
                "reward_metric_toggle": {
                    "train_url": _updated_query_url(request, reward_metric="train"),
                    "validation_url": _updated_query_url(request, reward_metric="validation"),
                },
                "nav_links": _nav_links(active="env", env_id=env_id),
            },
        )

    @app.get(
        "/runs/{env_id}/{run_id}/labels/{label_name}/candidates/{candidate_name}",
        response_class=HTMLResponse,
    )
    async def candidate_code_view(
        request: Request,
        env_id: str,
        run_id: str,
        label_name: str,
        candidate_name: str,
    ) -> HTMLResponse:
        try:
            candidate, code = load_candidate_code(
                env_id,
                run_id,
                label_name,
                candidate_name,
                layout=resolved_layout,
            )
            run_detail = load_pattern_run_detail(
                env_id,
                run_id,
                layout=resolved_layout,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        label_summary = next(
            (item for item in run_detail.label_telemetry if item.name == label_name),
            None,
        )
        if label_summary is None:
            raise HTTPException(
                status_code=404,
                detail=f"Label not found for run {run_id}: {label_name}",
            )
        parent_candidates = _candidate_parent_candidates(candidate, label_summary)
        reward_metric = _normalize_reward_metric(request.query_params.get("reward_metric"))
        return templates.TemplateResponse(
            request,
            "code.html",
            {
                "page_title": f"{env_id} {candidate_name}",
                "request": request,
                "env_id": env_id,
                "run_id": run_id,
                "label_name": label_name,
                "candidate": candidate,
                "label_summary": label_summary,
                "code": code,
                "parent_candidates": parent_candidates,
                "current_reward_metric": reward_metric,
                "reward_metric_toggle": {
                    "train_url": _updated_query_url(request, reward_metric="train"),
                    "validation_url": _updated_query_url(request, reward_metric="validation"),
                },
                "nav_links": _nav_links(active="env", env_id=env_id),
            },
        )

    @app.get(
        "/runs/{env_id}/{run_id}/labels/{label_name}/candidates/{candidate_name}/diff",
        response_class=HTMLResponse,
    )
    async def candidate_diff_view(
        request: Request,
        env_id: str,
        run_id: str,
        label_name: str,
        candidate_name: str,
    ) -> HTMLResponse:
        try:
            candidate, code = load_candidate_code(
                env_id,
                run_id,
                label_name,
                candidate_name,
                layout=resolved_layout,
            )
            run_detail = load_pattern_run_detail(
                env_id,
                run_id,
                layout=resolved_layout,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        label_summary = next(
            (item for item in run_detail.label_telemetry if item.name == label_name),
            None,
        )
        if label_summary is None:
            raise HTTPException(
                status_code=404,
                detail=f"Label not found for run {run_id}: {label_name}",
            )
        parent_candidates = _candidate_parent_candidates(candidate, label_summary)
        diff_sections = _candidate_diff_sections(
            child_candidate=candidate,
            child_code=code,
            parent_candidates=parent_candidates,
        )
        return templates.TemplateResponse(
            request,
            "candidate_diff.html",
            {
                "page_title": f"{env_id} {candidate_name} diff",
                "request": request,
                "env_id": env_id,
                "run_id": run_id,
                "label_name": label_name,
                "candidate": candidate,
                "parent_candidates": parent_candidates,
                "diff_sections": diff_sections,
                "nav_links": _nav_links(active="env", env_id=env_id),
            },
        )

    return app
