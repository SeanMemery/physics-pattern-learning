from __future__ import annotations

import copy
import base64
import hashlib
import html
import io
import json
import math
import os
import random
import re
import statistics
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote
from urllib import error as urllib_error
from urllib import request as urllib_request
from zoneinfo import ZoneInfo
import yaml

from .env_catalog import (
    DatasetDetail,
    DatasetEntry,
    DatasetSplitDetail,
    DatasetTraceDetail,
    DatasetTraceEntry,
    LibraryEntry,
    RewardResultEntry,
    get_environment_dataset,
    list_environment_datasets,
    list_environment_reward_results,
    list_environment_libraries,
)
from .analysis_results import (
    SupervisedConsistencyResult,
    SupervisedRunOption,
    TraceOption,
    filter_supervised_consistency_results_to_latest_setup,
    list_supervised_consistency_results,
    list_supervised_run_options,
    list_trace_options,
    load_supervised_consistency_result,
)
from .ablation_cache import build_ablation_importance_rows_from_cache, build_ablation_points_from_cache
from .learning_progress import display_learning_stage, summarize_learning_progress
from .learning_trajectory import load_reward_trajectory
from .library_collections import (
    AUTOMATIC_LIBRARY_LIMIT,
    UNSUPERVISED_LIBRARY_LIMIT,
    resolve_canonical_library,
    _ensure_client_src_on_path,
    load_cached_canonical_library_summary,
    load_global_best_supervised_library_summary,
)
from .library_ablation import (
    LibraryAblationDetail,
    case_counts_toward_label_series,
    load_library_ablation_detail,
    load_library_ablation_job_detail,
)
from .models import ClientRecord, JobRecord, JobSpec, JobStatus, PublicAccessState, QueueSettings
from .reward_details import (
    RewardOptimizationDetail,
    RewardValueModelDetail,
    load_reward_job_detail,
)
from .rewards_dashboard import (
    REWARD_PANEL_KEYS,
    load_current_scene,
    load_reward_panel_snapshot,
    load_rewards_dashboard_state,
    random_scene_running,
)
from .state import (
    IMAGE_ONLY_EVAL_TARGET_RUNS,
    REPEAT_EVALS_PER_RUN,
    REPEAT_LEARNING_TARGET_RUNS,
)
from .supervised_label_importance import (
    load_supervised_label_importance,
    resolve_supervised_evaluation_root,
    resolve_latest_supervised_library,
)
from .task_summaries import TaskSummaryEntry
from .video_renders import (
    annotation_video_filter_descriptions,
    filter_annotation_patterns_for_env,
    list_video_render_results,
    load_cached_global_best_supervised_library_summary,
    load_annotation_video_selection,
    select_video_showcase_entries,
)


SUPPORTED_ENVS = ("phyre", "iphyre", "kinetix", "pooltool")
SUMMARIES_DISPLAY_ENVS = ("phyre", "iphyre", "pooltool")
QUEUE_VISIBLE_STATUSES = {
    "queued",
    "assigned",
    "starting",
    "running",
    "pausing",
    "paused",
    "resuming",
    "stopping",
}
ACTIVE_QUEUE_STATUSES = {"assigned", "starting", "running", "pausing", "resuming", "stopping"}
FAILED_QUEUE_STATUSES = {"failed", "lost", "retryable_failure", "cancelled", "stopped"}
REQUIRED_JOB_FAMILIES = (
    "Image-only DeepPHY",
    "Supervised",
    "Unsupervised",
    "Automatic",
)
REQUIRED_FAMILY_SLUGS = {
    "Image-only DeepPHY": "image-only-deepphy",
    "Supervised": "supervised",
    "Supervised DeepPHY": "supervised-deepphy",
    "Supervised library": "supervised-library",
    "Unsupervised": "unsupervised",
    "Unsupervised DeepPHY": "unsupervised-deepphy",
    "Unsupervised library": "unsupervised-library",
    "Automatic": "automatic",
    "Automatic DeepPHY": "automatic-deepphy",
    "Automatic library": "automatic-library",
    "Optimization": "optimization",
    "Value model": "value-model",
}
REQUIRED_REWARD_PROGRAM_JOB_FAMILIES = (
    "Optimization",
    "Value model",
)
REQUIRED_LIBRARY_ABLATION_JOB_FAMILIES = (
    "Supervised library",
    "Unsupervised library",
)
REPEAT_LEARNING_FAMILIES = (
    "Supervised",
    "Unsupervised",
    "Automatic",
)
REQUIRED_HEALTHY_STATUSES = {"queued", "assigned", "starting", "running", "succeeded"}
TEXT_PREVIEW_SUFFIXES = {".json", ".txt", ".log", ".yaml", ".yml", ".md", ".csv"}
MAX_PREVIEW_BYTES = 64_000
DISPLAY_TZ = ZoneInfo("Europe/London")
_DISPLAY_NOISE_PATTERN = re.compile(r"(?<!\S)(?:base_url|model)=[^\s]*")
_REPEAT_RUN_SUFFIX_PATTERN = re.compile(r"^(?P<prefix>.+)-r(?P<index>\d+)$")
_REPEAT_EVAL_SUFFIX_PATTERN = re.compile(r"^(?P<prefix>.+)-repeat-(?P<index>\d+)$")
_LABEL_DIR_PATTERN = re.compile(r"^label_(?P<index>\d+?)_(?P<label>.+)$")
_CANDIDATE_DIR_PATTERN = re.compile(r"^candidate_(?P<index>\d+)(?:_|$)")
_RESUME_SYNC_RUN_SUFFIX_PATTERN = re.compile(r"^(?P<base>.+)\.__resume_sync__[0-9a-f]{32}$")
_ANALYSIS_REPEAT_PROGRESS_PATTERN = re.compile(
    r"patterns\.analysis\.consistency\.repeat\.(?:done|start):.*?\brepeat=(?P<current>\d+)\b.*?\btotal_repeats=(?P<total>\d+)\b"
)
_ANALYSIS_START_PATTERN = re.compile(
    r"patterns\.analysis\.consistency\.start:.*?(?:\bruns=(?P<runs>\d+)\b)?.*?(?:\bwindows=(?P<windows>\d+)\b)?.*?(?:\bcheckpoints=(?P<checkpoints>\d+)\b)?"
)
REPEAT_CATALOG_TOKEN = "repeat-batch"
STALE_ASSIGNMENT_SECONDS = 15 * 60
STALE_RUNNING_SECONDS = 35 * 60
HOME_CLIENT_PANEL_LIMIT = 24
HOME_PAUSED_JOB_LIMIT = 6
HOME_QUEUED_JOB_LIMIT = 8
HOME_EVENT_PANEL_LIMIT = 24
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".m4v"}


@lru_cache(maxsize=1)
def _services_root() -> Path:
    resolved = Path(__file__).resolve()
    for parent in resolved.parents:
        if (parent / "Work").is_dir() and (parent / "AGENTS.md").is_file():
            return parent
    return resolved.parents[4]


@lru_cache(maxsize=1)
def _shared_plot_style_config() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    style_path = project_root / "plotting" / "plot_style.yaml"
    if not style_path.is_file():
        style_path = project_root / "src" / "human_survey" / "plot_style.yaml"
    if not style_path.is_file():
        return {}
    try:
        payload = yaml.safe_load(style_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _shared_plot_colors() -> list[str]:
    colors = (_shared_plot_style_config().get("colors") or [])
    resolved = [str(value).strip() for value in colors if str(value).strip()]
    return resolved or ["#d15d08", "#0474b4", "#c783a6", "#2ca02c", "#9467bd", "#17becf"]


def _shared_plot_figure_size(name: str = "wide") -> tuple[float, float]:
    sizes = (_shared_plot_style_config().get("figure_sizes") or {})
    values = sizes.get(name) or sizes.get("default") or [3.3, 2.2]
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        return (6.6, 2.2) if name == "wide" else (3.3, 2.2)
    try:
        return float(values[0]), float(values[1])
    except (TypeError, ValueError):
        return (6.6, 2.2) if name == "wide" else (3.3, 2.2)


@dataclass(frozen=True)
class EvaluationTaskAttempt:
    index: int
    success: bool | None
    duration_seconds: float | None
    terminal_status: str
    evidence_status: str
    trace_id: str
    trace_path: Path | None
    evidence_path: Path | None
    evidence_payload: Any
    prompt: Any
    parsed_action: Any
    raw_model_output: str
    trace_description: Any
    native_metrics: Any
    router: Any


@dataclass(frozen=True)
class EvaluationTaskDetail:
    job_id: str
    task_id: str
    env_id: str
    task_dir: Path
    result_path: Path
    payload: dict[str, Any]
    attempts: list[EvaluationTaskAttempt]
    media_files: list[Path]
    extra_files: list[Path]
    success: bool | None
    completed_at: datetime | None


@dataclass(frozen=True)
class DashboardEvent:
    occurred_at: datetime
    tone: str
    title: str
    summary: str
    href: str | None = None
    context: str | None = None


def _display_time(value: datetime | None) -> str:
    if value is None:
        return "—"
    local_value = value.astimezone(DISPLAY_TZ)
    return local_value.strftime("%Y-%m-%d %H:%M:%S %Z")


def _display_duration(start: datetime | None, end: datetime | None = None) -> str:
    if start is None:
        return "—"
    finish = end or datetime.now(tz=DISPLAY_TZ)
    delta = finish - start
    seconds = max(0, int(delta.total_seconds()))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(f"{seconds}s")
    return " ".join(parts[:3])


def _coerce_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=DISPLAY_TZ)
    return value


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return _coerce_datetime(parsed)


def _sanitize_display_text(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    cleaned = _DISPLAY_NOISE_PATTERN.sub("", text)
    cleaned = " ".join(cleaned.split())
    return cleaned


def _sanitize_display_data(value: Any) -> Any:
    if isinstance(value, str):
        return _sanitize_display_text(value)
    if isinstance(value, list):
        return [_sanitize_display_data(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize_display_data(item) for key, item in value.items()}
    return value


def _status_class(status: str) -> str:
    normalized = str(status or "").strip().lower()
    if normalized == "completed":
        return "succeeded"
    if normalized in {"running", "succeeded"}:
        return normalized
    if normalized in {"paused", "failed", "queued"}:
        return normalized
    if normalized in {"assigned", "starting", "resuming"}:
        return "running"
    if normalized == "draining":
        return "paused"
    if normalized in {"pausing", "stopping"}:
        return "paused"
    if normalized in {"cancelled", "stopped", "lost", "retryable_failure"}:
        return "failed"
    return "queued"


def _pill(label: str, status: str) -> str:
    return f'<span class="pill status-{_status_class(status)}">{html.escape(label)}</span>'


def _learning_status_pill(job: JobRecord, progress: dict[str, object] | None) -> str:
    label = job.status.value
    tone = job.status.value
    if progress is not None:
        completed_labels = int(progress.get("completed_labels") or 0)
        total_labels = int(progress.get("total_labels") or 0)
        if total_labels > 0 and completed_labels < total_labels and _status_class(job.status.value) == "succeeded":
            label = "incomplete"
            tone = "paused"
    return _pill(label, tone)


def _job_env_id(job: JobRecord) -> str:
    return str(job.spec.metadata.get("env_id") or "—")


def _queue_order(job: JobRecord) -> int:
    raw = job.spec.metadata.get("queue_order")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _queue_sort_key(job: JobRecord) -> tuple[int, int, str]:
    status = job.status.value
    if status in ACTIVE_QUEUE_STATUSES:
        rank = 0
    elif status == "paused":
        rank = 1
    else:
        rank = 2
    return (rank, _queue_order(job), job.created_at.isoformat())


def _preferred_client_name(job: JobRecord) -> str:
    preferred = str(job.spec.metadata.get("preferred_client_name") or "").strip()
    return preferred or "Any compatible client"


def _effective_client_slots(client: ClientRecord) -> int:
    try:
        return max(1, int(client.max_concurrency))
    except (TypeError, ValueError):
        return 1


def _client_title(client: ClientRecord) -> str:
    display_name = str(client.display_name or "").strip()
    if display_name and display_name != "local-client":
        return display_name
    hostname = str(client.hostname or "").strip()
    if hostname:
        return hostname
    return client.client_id[:8]


def _dashboard_active_clients(clients: list[ClientRecord]) -> list[ClientRecord]:
    return [client for client in clients if client.status.value == "online"]


def _job_last_activity(job: JobRecord) -> datetime:
    return _coerce_datetime(job.updated_at) or _coerce_datetime(job.started_at) or _coerce_datetime(job.created_at) or datetime.now(
        tz=DISPLAY_TZ
    )


def _job_last_activity_summary(job: JobRecord, *, now: datetime | None = None) -> str:
    reference = _job_last_activity(job)
    current = _coerce_datetime(now) or datetime.now(tz=DISPLAY_TZ)
    return f"{_display_time(reference)} · {_display_duration(reference, current)} ago"


def _job_last_activity_age(job: JobRecord, *, now: datetime | None = None) -> str:
    reference = _job_last_activity(job)
    current = _coerce_datetime(now) or datetime.now(tz=DISPLAY_TZ)
    return f"{_display_duration(reference, current)} ago"


def _job_required_family(job: JobRecord) -> str | None:
    action = str(job.spec.action or "").strip()
    metadata = dict(job.spec.metadata or {})
    workflow_kind = str(metadata.get("workflow_kind") or "").strip().lower()
    if action == "supervised-learn-and-evaluate":
        return "Supervised"
    if action == "unsupervised-launch":
        return "Unsupervised"
    if action == "automatic-launch":
        return "Automatic"
    if action == "evaluate-deepphy":
        pattern_run_dir = str(metadata.get("pattern_run_dir") or "").strip()
        request_payload = metadata.get("request")
        if isinstance(request_payload, dict) and not pattern_run_dir:
            pattern_run_dir = str(request_payload.get("pattern_run_dir") or "").strip()
        if workflow_kind == "supervised":
            return "Supervised DeepPHY"
        if workflow_kind == "unsupervised":
            return "Unsupervised DeepPHY"
        if pattern_run_dir:
            normalized_dir = pattern_run_dir.replace("\\", "/").lower()
            if "/automatic/" in normalized_dir or normalized_dir.endswith("-automatic"):
                return "Automatic DeepPHY"
            if "/unsupervised/" in normalized_dir or normalized_dir.endswith("-unsupervised"):
                return "Unsupervised DeepPHY"
            return "Supervised DeepPHY"
        if _deepphy_eval_mode_label(job) == "Image only":
            return "Image-only DeepPHY"
    return None


def _job_library_ablation_family(job: JobRecord) -> str | None:
    if str(job.spec.action or "").strip() != "evaluate-library-case":
        return None
    kind = str(job.spec.metadata.get("library_ablation_kind") or "").strip().lower()
    if kind == "supervised":
        return "Supervised library"
    if kind == "unsupervised":
        return "Unsupervised library"
    if kind == "automatic":
        return "Automatic library"
    return None


def _required_job_is_healthy(job: JobRecord) -> bool:
    family = _job_required_family(job)
    if family is None:
        return False
    if job.status.value in REQUIRED_HEALTHY_STATUSES:
        return True
    return family.endswith("DeepPHY") and job.status.value == "paused"


def _job_is_prerequisite_blocked(job: JobRecord) -> bool:
    message_text = _sanitize_display_text(job.message).lower()
    return job.status.value == "paused" and (
        "waiting for prerequisite" in message_text
        or "pattern_library.json" in message_text
        or "supervised evaluation results not found" in message_text
        or "not found" in message_text
    )


def _job_attention_reason(
    job: JobRecord,
    *,
    client_lookup: dict[str, ClientRecord],
    now: datetime | None = None,
) -> tuple[str, str] | None:
    current = _coerce_datetime(now) or datetime.now(tz=DISPLAY_TZ)
    last_activity = _job_last_activity(job)
    age_seconds = max(0, int((current - last_activity).total_seconds()))
    assigned_client = client_lookup.get(job.assigned_client_id or "")
    if job.status.value in FAILED_QUEUE_STATUSES:
        return ("danger", "Failed or stopped")
    if assigned_client is not None and assigned_client.status.value == "offline" and job.status.value in ACTIVE_QUEUE_STATUSES:
        return ("danger", "Offline client still holds lease")
    if job.status.value in {"assigned", "starting", "resuming"} and age_seconds >= STALE_ASSIGNMENT_SECONDS:
        return ("danger", "Assigned but no start confirmation")
    if job.status.value in {"running", "pausing", "stopping"} and age_seconds >= STALE_RUNNING_SECONDS:
        return ("warning", "Running without a fresh status update")
    message_text = _sanitize_display_text(job.message).lower()
    if job.status.value == "paused" and (
        "waiting for prerequisite" in message_text or "pattern_library.json" in message_text or "not found" in message_text
    ):
        return ("warning", "Blocked on prerequisite output")
    return None


@lru_cache(maxsize=1)
def _required_job_catalog() -> dict[tuple[str, str], str]:
    return _required_job_catalog_from_file(_services_root() / "JOBS.md")


def _required_job_catalog_for_project(project_root: Path) -> dict[tuple[str, str], str]:
    catalog_root = _catalog_root_for_project(project_root)
    catalog = _required_job_catalog() if catalog_root == _services_root() else _required_job_catalog_from_file(catalog_root / "JOBS.md")
    if catalog:
        return catalog
    state_root = project_root / "state"
    try:
        from .state import ServerStateStore

        store = ServerStateStore(state_root, project_root=project_root)
    except Exception:
        store = None
    if store is not None:
        inferred = _inferred_required_job_catalog_from_jobs(store.list_jobs_snapshot())
        if inferred:
            return inferred
    return _inferred_required_job_catalog_from_project_files(project_root)


@lru_cache(maxsize=1)
def _required_reward_program_job_catalog() -> dict[tuple[str, str], str]:
    return _required_job_catalog_from_file(
        _services_root() / "REWARD_PROGRAM_JOBS.md"
    )


def _required_reward_program_job_catalog_for_project(project_root: Path) -> dict[tuple[str, str], str]:
    catalog_root = _catalog_root_for_project(project_root)
    if catalog_root == _services_root():
        return _required_reward_program_job_catalog()
    return _required_job_catalog_from_file(catalog_root / "REWARD_PROGRAM_JOBS.md")


@lru_cache(maxsize=1)
def _required_library_ablation_job_catalog() -> dict[tuple[str, str], str]:
    return _required_job_catalog_from_file(
        _services_root() / "LIBRARY_ABLATION_JOBS.md"
    )


def _catalog_root_for_project(project_root: Path) -> Path:
    resolved = project_root.resolve()
    for candidate in (resolved, *resolved.parents):
        if (
            (candidate / "JOBS.md").is_file()
            or (candidate / "REWARD_PROGRAM_JOBS.md").is_file()
            or (candidate / "LIBRARY_ABLATION_JOBS.md").is_file()
        ):
            return candidate
        if (candidate / "Work").is_dir() and (candidate / "AGENTS.md").is_file():
            return candidate
    return resolved


def _required_library_ablation_job_catalog_for_project(project_root: Path) -> dict[tuple[str, str], str]:
    catalog_root = _catalog_root_for_project(project_root)
    if catalog_root == _services_root():
        return _required_library_ablation_job_catalog()
    return _required_job_catalog_from_file(catalog_root / "LIBRARY_ABLATION_JOBS.md")


def _required_job_catalog_from_file(path: Path) -> dict[tuple[str, str], str]:
    catalog: dict[tuple[str, str], str] = {}
    if not path.is_file():
        return catalog
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return catalog
    for line in text.splitlines():
        if not line.startswith("|") or "---" in line or "Family" in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 3:
            continue
        family = parts[0]
        env_id = parts[1]
        job_id = parts[2].strip("`")
        if family and env_id and job_id:
            catalog[(env_id, family)] = job_id
    return catalog


def _inferred_required_job_catalog_from_jobs(jobs: list[JobRecord]) -> dict[tuple[str, str], str]:
    catalog: dict[tuple[str, str], str] = {}
    for env_id in SUPPORTED_ENVS:
        for family in REQUIRED_JOB_FAMILIES:
            candidates = [
                job
                for job in jobs
                if _required_ui_job_visible(job)
                and _job_required_family(job) == family
                and str(job.spec.metadata.get("env_id") or job.environment or "").strip() == env_id
            ]
            if not candidates:
                continue
            if family in REPEAT_LEARNING_FAMILIES or family == "Image-only DeepPHY":
                repeat_candidates = [
                    job
                    for job in candidates
                    if _job_repeat_learning_family(job) == family or _job_repeat_required_family(job) == family
                ]
                if repeat_candidates:
                    catalog[(env_id, family)] = REPEAT_CATALOG_TOKEN
                    continue
            selected = max(
                candidates,
                key=lambda item: (
                    1 if item.status.value in ACTIVE_QUEUE_STATUSES else 0,
                    0 if item.status.value in FAILED_QUEUE_STATUSES else 1,
                    item.updated_at.timestamp(),
                    item.created_at.timestamp(),
                    item.job_id,
                ),
            )
            catalog[(env_id, family)] = selected.job_id
    return catalog


def _inferred_required_job_catalog_from_project_files(project_root: Path) -> dict[tuple[str, str], str]:
    catalog: dict[tuple[str, str], str] = {}
    patterns_root = project_root / "data" / "patterns"
    for env_id in SUPPORTED_ENVS:
        env_root = patterns_root / env_id
        if not env_root.is_dir():
            continue
        if any(path.parent == env_root for path in env_root.glob("*/pattern_library.json")):
            catalog[(env_id, "Supervised")] = REPEAT_CATALOG_TOKEN
        unsupervised_root = env_root / "unsupervised"
        if unsupervised_root.is_dir() and any(unsupervised_root.glob("*/pattern_library.json")):
            catalog[(env_id, "Unsupervised")] = REPEAT_CATALOG_TOKEN
        automatic_root = env_root / "automatic"
        if automatic_root.is_dir() and any(automatic_root.glob("*/pattern_library.json")):
            catalog[(env_id, "Automatic")] = REPEAT_CATALOG_TOKEN
    return catalog


def required_family_slug(family: str) -> str:
    return REQUIRED_FAMILY_SLUGS.get(family, family.strip().lower().replace(" ", "-"))


def required_family_from_slug(slug: str) -> str | None:
    normalized = str(slug or "").strip().lower()
    for family, family_slug in REQUIRED_FAMILY_SLUGS.items():
        if family_slug == normalized:
            return family
    return None


def _required_variant_href(*, env_id: str, family: str) -> str:
    return f"/required-runs/{quote(env_id)}/{quote(required_family_slug(family))}"


def _catalog_tracks_repeat_batch(job_ref: str | None) -> bool:
    return str(job_ref or "").strip().lower() == REPEAT_CATALOG_TOKEN


def _required_ui_job_visible(job: JobRecord | None) -> bool:
    return job is not None and job.status.value not in FAILED_QUEUE_STATUSES


def _job_is_superseded_combined_source(job: JobRecord | None) -> bool:
    if job is None:
        return False
    metadata = dict(job.spec.metadata or {})
    return bool(
        metadata.get("source_combined_hidden")
        or metadata.get("combined_into_job_id")
        or metadata.get("hidden_required_run")
    )


def _marker_has_repeat_shape(marker_text: str) -> bool:
    if not marker_text:
        return False
    for marker in marker_text.split():
        marker_name = Path(marker).name
        if _REPEAT_EVAL_SUFFIX_PATTERN.match(marker_name) is not None:
            return True
        if _REPEAT_RUN_SUFFIX_PATTERN.match(marker_name) is not None:
            return True
    return False


def _latest_repeat_catalog_job(
    *,
    env_id: str,
    family: str,
    jobs: list[JobRecord],
) -> JobRecord | None:
    if family in REPEAT_LEARNING_FAMILIES:
        candidates = [
            job
            for job in jobs
            if _job_repeat_learning_family(job) == family
            and str(job.spec.metadata.get("env_id") or job.environment or "").strip() == env_id
        ]
    else:
        candidates = [
            job
            for job in jobs
            if _job_repeat_required_family(job) == family
            and str(job.spec.metadata.get("env_id") or job.environment or "").strip() == env_id
        ]
        if not candidates and "DeepPHY" in family:
            candidates = [
                job
                for job in jobs
                if _job_required_family(job) == family
                and str(job.spec.action or "").strip() == "evaluate-deepphy"
                and str(job.spec.metadata.get("env_id") or job.environment or "").strip() == env_id
            ]
    if not candidates:
        return None
    visible_candidates = [job for job in candidates if _required_ui_job_visible(job)]
    if visible_candidates:
        candidates = visible_candidates
    else:
        return None
    return max(
        candidates,
        key=lambda item: (
            1 if item.status.value in ACTIVE_QUEUE_STATUSES else 0,
            0 if item.status.value in FAILED_QUEUE_STATUSES else 1,
            item.updated_at.timestamp(),
            item.created_at.timestamp(),
            item.job_id,
        ),
    )


def _automatic_deepphy_eval_root(job: JobRecord, *, project_root: Path) -> Path | None:
    run_dir = _resolve_job_run_dir(job, project_root=project_root)
    if run_dir is None:
        return None
    deepphy_root = run_dir / "deepphy_eval"
    preferred = deepphy_root / "combined-eval"
    if preferred.exists():
        return preferred
    if not deepphy_root.exists():
        return None
    candidates: list[Path] = []
    for child in sorted(deepphy_root.iterdir()):
        if not child.is_dir():
            continue
        if _is_archived_deepphy_eval_dir(child):
            continue
        if (child / "results.json").is_file():
            return child
        if any(child.glob("tasks/*/*/task_result.json")):
            candidates.append(child)
    if candidates:
        return candidates[0]
    return None


def _is_archived_deepphy_eval_dir(path: Path) -> bool:
    name = path.name.strip().lower()
    return ".incomplete_" in name or ".bak" in name


def _automatic_deepphy_eval_cache_marker(eval_root: Path) -> tuple[int, int, int]:
    try:
        root_mtime_ns = eval_root.stat().st_mtime_ns
    except OSError:
        root_mtime_ns = 0
    results_json = eval_root / "results.json"
    try:
        results_mtime_ns = results_json.stat().st_mtime_ns if results_json.is_file() else 0
    except OSError:
        results_mtime_ns = 0
    tasks_dir = eval_root / "tasks"
    try:
        tasks_mtime_ns = tasks_dir.stat().st_mtime_ns if tasks_dir.exists() else 0
    except OSError:
        tasks_mtime_ns = 0
    return (root_mtime_ns, results_mtime_ns, tasks_mtime_ns)


@lru_cache(maxsize=64)
def _automatic_deepphy_eval_summary_cached(
    eval_root_value: str,
    marker: tuple[int, int, int],
) -> tuple[int, str]:
    del marker
    eval_root = Path(eval_root_value)
    task_results = sum(1 for _ in eval_root.rglob("task_result.json"))
    latest_mtime: float | None = None
    for candidate in eval_root.rglob("*"):
        try:
            if candidate.is_file():
                mtime = candidate.stat().st_mtime
                latest_mtime = mtime if latest_mtime is None else max(latest_mtime, mtime)
        except OSError:
            continue
    latest_detail = ""
    if latest_mtime is not None:
        latest_dt = datetime.fromtimestamp(latest_mtime, tz=DISPLAY_TZ)
        latest_detail = f" · {_display_time(latest_dt)}"
    return task_results, latest_detail


def _required_catalog_deepphy_eval_root(
    *,
    family: str,
    selected: JobRecord | None,
    project_root: Path,
) -> Path | None:
    if selected is None:
        return None
    if family == "Automatic DeepPHY":
        return _automatic_deepphy_eval_root(selected, project_root=project_root)
    if str(selected.spec.action or "").strip() != "evaluate-deepphy":
        return None
    metadata = dict(selected.spec.metadata or {})
    for key in ("evaluation_run_dir", "output_path", "run_dir"):
        raw = str(metadata.get(key) or "").strip()
        if not raw:
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (project_root / candidate).resolve()
        if candidate.exists():
            return candidate
    return None


def _required_catalog_deepphy_eval_marker(eval_root: Path) -> tuple[int, int, int]:
    try:
        root_mtime_ns = eval_root.stat().st_mtime_ns
    except OSError:
        root_mtime_ns = 0
    results_json = eval_root / "results.json"
    try:
        results_mtime_ns = results_json.stat().st_mtime_ns if results_json.is_file() else 0
    except OSError:
        results_mtime_ns = 0
    tasks_dir = eval_root / "tasks"
    try:
        tasks_mtime_ns = tasks_dir.stat().st_mtime_ns if tasks_dir.exists() else 0
    except OSError:
        tasks_mtime_ns = 0
    return (root_mtime_ns, results_mtime_ns, tasks_mtime_ns)


@lru_cache(maxsize=96)
def _deepphy_eval_task_counts_cached(
    eval_root_value: str,
    marker: tuple[int, int, int],
    pooltool_attempt_limit: int | None,
) -> tuple[int, int, int, bool] | None:
    del marker
    eval_root = Path(eval_root_value)
    success_count = 0
    failure_count = 0
    total_tasks = 0
    saw_pooltool = False
    for result_path in eval_root.glob("tasks/*/*/task_result.json"):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        success = payload.get("success")
        if not isinstance(success, bool):
            continue
        env_id = str(payload.get("env_id") or "").strip().lower()
        records = payload.get("records")
        over_attempt_limit = False
        if env_id == "pooltool":
            saw_pooltool = True
            if isinstance(pooltool_attempt_limit, int) and isinstance(records, list):
                over_attempt_limit = len(records) > pooltool_attempt_limit
        total_tasks += 1
        if success and not over_attempt_limit:
            success_count += 1
        else:
            failure_count += 1
    if total_tasks <= 0:
        return None
    return (success_count, failure_count, total_tasks, saw_pooltool)


@lru_cache(maxsize=96)
def _required_catalog_deepphy_eval_counts_cached(
    eval_root_value: str,
    marker: tuple[int, int, int],
    pooltool_attempt_limit: int | None = None,
) -> tuple[int, int, int] | None:
    eval_root = Path(eval_root_value)
    task_counts = _deepphy_eval_task_counts_cached(
        eval_root_value,
        marker,
        pooltool_attempt_limit,
    )
    if task_counts is not None:
        success_count, failure_count, total_tasks, saw_pooltool = task_counts
        if saw_pooltool:
            return (success_count, failure_count, total_tasks)
    results_path = eval_root / "results.json"
    if results_path.is_file():
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            env_summaries = payload.get("env_summaries")
            if isinstance(env_summaries, dict):
                summaries = [item for item in env_summaries.values() if isinstance(item, dict)]
            elif isinstance(env_summaries, list):
                summaries = [item for item in env_summaries if isinstance(item, dict)]
            else:
                summaries = []
            if summaries:
                total_tasks = 0
                success_count = 0
                failure_count = 0
                for summary in summaries:
                    try:
                        total_tasks += int(summary.get("task_count") or summary.get("total_tasks") or 0)
                        success_count += int(summary.get("success_count") or summary.get("successful_tasks") or 0)
                        failure_count += int(summary.get("failure_count") or summary.get("failed_tasks") or 0)
                    except (TypeError, ValueError):
                        return None
                if total_tasks > 0:
                    return (success_count, failure_count, total_tasks)
    if task_counts is None:
        return None
    success_count, failure_count, total_tasks, _saw_pooltool = task_counts
    return (success_count, failure_count, total_tasks)


def _required_catalog_deepphy_visual(
    *,
    family: str,
    selected: JobRecord | None,
    project_root: Path,
) -> str:
    return (
        ""
    )


def _required_catalog_selection(
    *,
    env_id: str,
    family: str,
    catalog: dict[tuple[str, str], str],
    job_lookup: dict[str, JobRecord],
    jobs: list[JobRecord],
    project_root: Path,
) -> tuple[JobRecord | None, str | None, str]:
    tracked_ref = catalog.get((env_id, family), "")
    selected = (
        _latest_repeat_catalog_job(env_id=env_id, family=family, jobs=jobs)
        if _catalog_tracks_repeat_batch(tracked_ref)
        else job_lookup.get(tracked_ref)
    )
    if not _required_ui_job_visible(selected):
        selected = None
    if family != "Automatic DeepPHY":
        if selected is None:
            return None, None, "No tracked repeat batch found." if _catalog_tracks_repeat_batch(tracked_ref) else "No tracked job found."
        if _catalog_tracks_repeat_batch(tracked_ref) and family in REPEAT_LEARNING_FAMILIES:
            aggregate_detail = _repeat_learning_batch_progress_detail(
                env_id=env_id,
                family=family,
                jobs=jobs,
                project_root=project_root,
            )
            if aggregate_detail:
                return selected, selected.status.value, aggregate_detail
        detail = (
            _job_status_summary(selected, project_root=project_root)
            or _sanitize_display_text(selected.message)
            or _job_last_activity_summary(selected)
        )
        return selected, selected.status.value, detail

    if selected is not None and str(selected.spec.action or "").strip() == "evaluate-deepphy":
        detail = (
            _job_status_summary(selected, project_root=project_root)
            or _sanitize_display_text(selected.message)
            or _job_last_activity_summary(selected)
        )
        return selected, selected.status.value, detail

    automatic_ref = catalog.get((env_id, "Automatic"), "")
    automatic_job = (
        _latest_repeat_catalog_job(env_id=env_id, family="Automatic", jobs=jobs)
        if _catalog_tracks_repeat_batch(automatic_ref)
        else job_lookup.get(automatic_ref)
    )
    if not _required_ui_job_visible(automatic_job):
        automatic_job = None
    if automatic_job is None:
        return None, None, "No tracked automatic repeat run found." if _catalog_tracks_repeat_batch(automatic_ref) else "No tracked automatic run found."
    eval_root = _automatic_deepphy_eval_root(automatic_job, project_root=project_root)
    if eval_root is None:
        detail = (
            "Combined DeepPHY evaluation has not started yet."
            if automatic_job.status.value in REQUIRED_HEALTHY_STATUSES | {"paused"}
            else "No combined DeepPHY evaluation recorded yet."
        )
        return automatic_job, automatic_job.status.value, detail
    task_results, latest_detail = _automatic_deepphy_eval_summary_cached(
        str(eval_root),
        _automatic_deepphy_eval_cache_marker(eval_root),
    )
    detail = f"{task_results} task results recorded{latest_detail}"
    return automatic_job, automatic_job.status.value, detail


def _repeat_learning_run_within_target(job: JobRecord, *, project_root: Path) -> bool:
    run_index = _repeat_run_index(job, project_root=project_root)
    if run_index is None:
        return True
    return run_index <= REPEAT_LEARNING_TARGET_RUNS


def _repeat_learning_batch_progress_detail(
    *,
    env_id: str,
    family: str,
    jobs: list[JobRecord],
    project_root: Path,
) -> str:
    completed_total = 0
    label_total = 0
    saw_progress = False
    for job in jobs:
        if _job_repeat_learning_family(job) != family:
            continue
        if str(job.spec.metadata.get("env_id") or job.environment or "").strip() != env_id:
            continue
        if not _required_ui_job_visible(job):
            continue
        if not _repeat_learning_run_within_target(job, project_root=project_root):
            continue
        progress = _learning_progress_summary(job, project_root=project_root)
        if progress is None:
            continue
        total_labels = int(progress.get("total_labels") or 0)
        completed_labels = int(progress.get("completed_labels") or 0)
        if total_labels <= 0:
            continue
        saw_progress = True
        completed_total += completed_labels
        label_total += total_labels
    if not saw_progress or label_total <= 0:
        return ""
    return f"{completed_total}/{label_total} labels"


def _required_catalog_job_href(
    *,
    env_id: str,
    family: str,
    selected: JobRecord | None,
    project_root: Path,
) -> str | None:
    del selected, project_root
    return _required_variant_href(env_id=env_id, family=family)


def _render_required_catalog_panel(
    *,
    title: str,
    kicker: str,
    families: tuple[str, ...],
    catalog: dict[tuple[str, str], str],
    jobs: list[JobRecord],
    project_root: Path,
    allow_paused_as_healthy: bool = False,
) -> str:
    env_ids = sorted({env_id for env_id, _ in catalog})
    if not env_ids:
        return ""
    total_slots = len(env_ids) * len(families)
    healthy_slots = 0
    active_slots = 0
    job_lookup = {job.job_id: job for job in jobs}
    repeat_learning_statuses: dict[tuple[str, str], Counter[str]] = {}
    repeat_required_statuses: dict[tuple[str, str], Counter[str]] = {}
    for job in jobs:
        if not _required_ui_job_visible(job):
            continue
        repeat_family = _job_repeat_learning_family(job)
        if repeat_family is None:
            repeat_family = _job_repeat_required_family(job)
            if repeat_family is None:
                continue
            if not _required_repeat_job_counts_toward_target(job):
                continue
            repeat_env = str(job.spec.metadata.get("env_id") or job.environment or "").strip()
            if not repeat_env:
                continue
            repeat_required_statuses.setdefault((repeat_env, repeat_family), Counter())[job.status.value] += 1
            continue
        repeat_env = str(job.spec.metadata.get("env_id") or job.environment or "").strip()
        if not repeat_env:
            continue
        repeat_learning_statuses.setdefault((repeat_env, repeat_family), Counter())[job.status.value] += 1
    env_cards: list[str] = []
    for env_id in env_ids:
        rows: list[str] = []
        for family in families:
            selected, selected_status, detail = _required_catalog_selection(
                env_id=env_id,
                family=family,
                catalog=catalog,
                job_lookup=job_lookup,
                jobs=jobs,
                project_root=project_root,
            )
            selected_href = _required_catalog_job_href(
                env_id=env_id,
                family=family,
                selected=selected,
                project_root=project_root,
            )
            repeat_status_counts = repeat_learning_statuses.get((env_id, family), Counter())
            if not repeat_status_counts:
                repeat_status_counts = repeat_required_statuses.get((env_id, family), Counter())
            deepphy_visual = _required_catalog_deepphy_visual(
                family=family,
                selected=selected,
                project_root=project_root,
            )
            is_healthy = False
            if selected is not None:
                is_healthy = (
                    selected_status in REQUIRED_HEALTHY_STATUSES
                    or (allow_paused_as_healthy and selected_status == "paused")
                )
                if not allow_paused_as_healthy:
                    is_healthy = _required_job_is_healthy(selected)
            if repeat_status_counts:
                has_incomplete_repeats = any(
                    repeat_status_counts.get(status, 0) > 0
                    for status in ACTIVE_QUEUE_STATUSES | {"queued", "paused"}
                )
                if has_incomplete_repeats:
                    is_healthy = False
            if is_healthy:
                healthy_slots += 1
            if selected is not None and selected_status in ACTIVE_QUEUE_STATUSES:
                active_slots += 1
            visual_class = "is-missing"
            if selected is not None:
                if selected_status in ACTIVE_QUEUE_STATUSES:
                    visual_class = "is-active"
                elif repeat_status_counts:
                    queued_repeats = repeat_status_counts.get("queued", 0)
                    active_repeats = sum(repeat_status_counts.get(status, 0) for status in ACTIVE_QUEUE_STATUSES)
                    paused_repeats = repeat_status_counts.get("paused", 0)
                    if active_repeats > 0:
                        visual_class = "is-active"
                    elif queued_repeats > 0:
                        visual_class = "is-missing"
                    elif paused_repeats > 0:
                        visual_class = "is-blocked"
                elif _job_is_prerequisite_blocked(selected):
                    visual_class = "is-blocked"
                elif is_healthy:
                    visual_class = "is-blocked" if selected_status == "paused" else "is-covered"
                elif selected_status in FAILED_QUEUE_STATUSES:
                    visual_class = "is-failed"
                else:
                    visual_class = "is-missing"
            row_class = "required-row"
            has_incomplete_repeats = any(
                repeat_status_counts.get(status, 0) > 0
                for status in ACTIVE_QUEUE_STATUSES | {"queued", "paused"}
            )
            if selected is not None and selected_status in ACTIVE_QUEUE_STATUSES:
                row_class += " is-active-state"
            elif has_incomplete_repeats:
                if repeat_status_counts.get("running", 0) or repeat_status_counts.get("starting", 0) or repeat_status_counts.get("assigned", 0):
                    row_class += " is-active-state"
            elif selected is not None and selected_status == "succeeded":
                row_class += " is-succeeded"
            elif selected is not None and selected_status in FAILED_QUEUE_STATUSES:
                row_class += " is-terminal-failed"
            if repeat_status_counts:
                repeat_parts: list[str] = []
                for status_name in ("running", "starting", "assigned", "queued", "paused"):
                    count = repeat_status_counts.get(status_name, 0)
                    if count:
                        repeat_parts.append(f"{count} {status_name}")
                if repeat_parts:
                    detail = f"{detail} · repeats {' / '.join(repeat_parts)}"
            repeat_summary_html = _render_repeat_status_visuals(repeat_status_counts, family=family)
            rows.append(
                f"""
                <article class="{row_class}">
                  <div class="required-row-main">
                    <div class="required-row-title">
                      <span class="required-status-dot {visual_class}"></span>
                      {f'<a href="{selected_href}">{html.escape(family)}</a>' if selected_href is not None else html.escape(family)}
                    </div>
                    <div class="required-row-meta">{html.escape(detail)}</div>
                    {repeat_summary_html}
                  </div>
                  <div class="required-status">
                    {deepphy_visual}
                    {f'<a class="required-job-link" href="{selected_href}">Job</a>' if selected_href is not None else '<span class="metric-subtle">No job</span>'}
                  </div>
                </article>
                """
            )
        env_cards.append(
            f"""
            <section class="required-card">
              <div class="card-topline">
                <div>
                  <h3>{html.escape(env_id.upper())}</h3>
                </div>
                <span class="mini-chip">{sum(1 for family in families if catalog.get((env_id, family)) or (family == "Automatic DeepPHY" and catalog.get((env_id, "Automatic"))))} tracked</span>
              </div>
              <div class="required-list">{''.join(rows)}</div>
            </section>
            """
        )
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div>
          <p class="section-kicker">{html.escape(kicker)}</p>
          <h2>{html.escape(title)}</h2>
        </div>
        <div class="stats-inline">
          <span class="pill {'status-running' if healthy_slots == total_slots else 'status-paused'}">{healthy_slots}/{total_slots} healthy</span>
          <span class="pill status-running">Active {active_slots}</span>
        </div>
      </div>
      <div class="required-grid">{''.join(env_cards)}</div>
    </section>
    """


def _library_ablation_kind_label(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == "supervised":
        return "Supervised"
    if normalized == "unsupervised":
        return "Unsupervised"
    if normalized == "automatic":
        return "Automatic"
    return normalized.title() or "Library"


def _ablation_plot_kind_label(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == "supervised":
        return "Human-Labels"
    if normalized == "unsupervised":
        return "LLM-Labels"
    return _library_ablation_kind_label(value)


def _library_ablation_palette(kind: str) -> str:
    normalized = str(kind or "").strip().lower()
    colors = _shared_plot_colors()
    if normalized == "supervised":
        return colors[0]
    if normalized == "unsupervised":
        return colors[1] if len(colors) > 1 else colors[0]
    if normalized == "automatic":
        return colors[4] if len(colors) > 4 else colors[-1]
    return "#617182"


def _library_ablation_status_detail(detail: LibraryAblationDetail | None) -> str:
    if detail is None:
        return "Quick-eval ablation not started yet."
    if detail.completed_seed_run_count <= 0:
        if detail.active_seed_run_count > 0 or detail.observed_task_count > 0:
            return (
                f"{detail.benchmark_preset} · 0/{detail.total_seed_run_count} seeds complete"
                f" · {detail.active_seed_run_count} active"
                f" · {detail.observed_task_count} task results"
            )
        return f"{detail.benchmark_preset} · 0/{detail.total_seed_run_count} seeds complete"
    baseline = detail.baseline_case
    baseline_percent = (
        f"{baseline.mean_success_ratio * 100.0:.0f}%"
        if baseline is not None and baseline.mean_success_ratio is not None
        else "—"
    )
    return (
        f"{detail.benchmark_preset} · {detail.completed_seed_run_count}/{detail.total_seed_run_count} seeds"
        f" · baseline {baseline_percent}"
    )


def _library_ablation_job_href(job: JobRecord | None) -> str | None:
    if job is None:
        return None
    return f"/ablations/{html.escape(job.job_id)}"


def _required_library_ablation_progress_payload(
    *,
    jobs: list[JobRecord],
    project_root: Path,
) -> dict[str, object] | None:
    catalog = _required_library_ablation_job_catalog_for_project(project_root)
    env_ids = [
        env_id
        for env_id in SUPPORTED_ENVS
        if any(str(entry.workflow_kind or "").strip().lower() in {"supervised", "unsupervised"} for entry in list_environment_libraries(project_root, env_id))
        or any(key_env == env_id for key_env, _ in catalog)
    ]
    if not env_ids:
        return None
    job_lookup = {job.job_id: job for job in jobs}
    total_units = len(env_ids) * len(REQUIRED_LIBRARY_ABLATION_JOB_FAMILIES)
    if total_units <= 0:
        return None
    totals = Counter({"completed": 0, "active": 0, "queued": 0, "paused": 0, "failed": 0, "missing": 0})
    for env_id in env_ids:
        for family in REQUIRED_LIBRARY_ABLATION_JOB_FAMILIES:
            selected = job_lookup.get(catalog.get((env_id, family), ""))
            if selected is None:
                totals["missing"] += 1
                continue
            status = selected.status.value
            if status == "succeeded":
                totals["completed"] += 1
            elif status in ACTIVE_QUEUE_STATUSES:
                totals["active"] += 1
            elif status == "queued":
                totals["queued"] += 1
            elif status == "paused":
                totals["paused"] += 1
            elif status in FAILED_QUEUE_STATUSES:
                totals["failed"] += 1
            else:
                totals["queued"] += 1
    segments: list[dict[str, object]] = []
    chips: list[dict[str, object]] = []
    segment_specs = (
        ("Completed", "completed", "is-complete"),
        ("Active", "active", "is-active"),
        ("Queued", "queued", "is-queued"),
        ("Paused", "paused", "is-paused"),
        ("Failed", "failed", "is-failed"),
        ("Missing", "missing", "is-missing"),
    )
    for label, key, class_name in segment_specs:
        count = int(totals[key])
        if count <= 0:
            continue
        segments.append(
            {
                "label": label,
                "count": count,
                "class_name": class_name,
                "width": (count / total_units) * 100.0,
            }
        )
        chips.append({"label": label, "count": count, "class_name": class_name})
    return {
        "total": total_units,
        "completed": int(totals["completed"]),
        "active": int(totals["active"]),
        "queued": int(totals["queued"]),
        "paused": int(totals["paused"]),
        "failed": int(totals["failed"]),
        "missing": int(totals["missing"]),
        "completed_percent": int(round((int(totals["completed"]) / total_units) * 100.0)),
        "segments": segments,
        "chips": chips,
    }


def _render_required_library_ablation_progress_bar(*, jobs: list[JobRecord], project_root: Path) -> str:
    payload = _required_library_ablation_progress_payload(jobs=jobs, project_root=project_root)
    if payload is None:
        return ""
    segments_html = "".join(
        (
            f'<span class="required-overall-segment {html.escape(str(segment["class_name"]))}" '
            f'style="width: {float(segment["width"]):.3f}%" '
            f'title="{html.escape(str(segment["label"]))}: {int(segment["count"])}"></span>'
        )
        for segment in payload["segments"]
    )
    chips_html = "".join(
        (
            f'<span class="required-overall-chip {html.escape(str(chip["class_name"]))}">'
            f'<span class="required-overall-chip-label">{html.escape(str(chip["label"]))}</span>'
            f'{int(chip["count"])}'
            "</span>"
        )
        for chip in payload["chips"]
    )
    return f"""
    <section class="panel compact-panel required-overall-panel">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Required Progress</p>
          <h2>Required Ablations Overall</h2>
          <p class="metric-subtle">Tracked library ablations for supervised and unsupervised source libraries across each environment.</p>
        </div>
        <div class="stats-inline">
          <span class="pill {'status-running' if int(payload['completed']) > 0 else 'status-paused'}">{int(payload['completed'])}/{int(payload['total'])} complete</span>
          <span class="pill status-running">{int(payload['completed_percent'])}% done</span>
        </div>
      </div>
      <div class="required-overall-bar-wrap">
        <div class="required-overall-bar">{segments_html}</div>
        <div class="required-overall-bar-label">{int(payload['completed'])} of {int(payload['total'])} required ablations complete</div>
      </div>
      <div class="required-overall-chip-row">{chips_html}</div>
    </section>
    """


def _render_library_ablation_panel(*, jobs: list[JobRecord], project_root: Path) -> str:
    catalog = _required_library_ablation_job_catalog_for_project(project_root)
    if not catalog:
        return ""
    job_lookup = {job.job_id: job for job in jobs}
    env_ids = sorted({env_id for env_id, _ in catalog})
    healthy_slots = 0
    active_slots = 0
    total_slots = len(env_ids) * len(REQUIRED_LIBRARY_ABLATION_JOB_FAMILIES)
    env_cards: list[str] = []
    for env_id in env_ids:
        rows: list[str] = []
        for family in REQUIRED_LIBRARY_ABLATION_JOB_FAMILIES:
            selected = job_lookup.get(catalog.get((env_id, family), ""))
            detail = (
                load_library_ablation_job_detail(selected, project_root=project_root)
                if selected is not None
                else None
            )
            detail_text = _library_ablation_status_detail(detail)
            if selected is not None and selected.status.value in REQUIRED_HEALTHY_STATUSES | {"paused"}:
                healthy_slots += 1
            if selected is not None and selected.status.value in ACTIVE_QUEUE_STATUSES:
                active_slots += 1
            visual_class = "is-missing"
            row_class = "required-row"
            if selected is not None:
                if selected.status.value in ACTIVE_QUEUE_STATUSES:
                    visual_class = "is-active"
                    row_class += " is-active-state"
                elif selected.status.value == "succeeded":
                    visual_class = "is-covered"
                    row_class += " is-succeeded"
                elif selected.status.value in FAILED_QUEUE_STATUSES:
                    visual_class = "is-failed"
                    row_class += " is-terminal-failed"
                elif selected.status.value == "paused":
                    visual_class = "is-blocked"
                else:
                    visual_class = "is-covered"
            selected_href = _library_ablation_job_href(selected)
            baseline_chip = ""
            if detail is not None and detail.baseline_case is not None and detail.baseline_case.mean_success_ratio is not None:
                baseline_chip = (
                    f'<span class="required-deepphy-text" title="Baseline mean success">'
                    f'{detail.baseline_case.mean_success_ratio * 100.0:.0f}%</span>'
                )
            rows.append(
                f"""
                <article class="{row_class}">
                  <div class="required-row-main">
                    <div class="required-row-title">
                      <span class="required-status-dot {visual_class}"></span>
                      {f'<strong><a href="{selected_href}">{html.escape(family)}</a></strong>' if selected_href is not None else f"<strong>{html.escape(family)}</strong>"}
                    </div>
                    <div class="required-row-meta">{html.escape(detail_text)}</div>
                  </div>
                  <div class="required-status">
                    {baseline_chip}
                    {f'<a class="required-job-link" href="{selected_href}">Job</a>' if selected_href is not None else '<span class="metric-subtle">No job</span>'}
                  </div>
                </article>
                """
            )
        env_cards.append(
            f"""
            <section class="required-card">
              <div class="card-topline">
                <div>
                  <h3>{html.escape(env_id.upper())}</h3>
                </div>
                <span class="mini-chip">{sum(1 for family in REQUIRED_LIBRARY_ABLATION_JOB_FAMILIES if catalog.get((env_id, family)))} tracked</span>
              </div>
              <div class="required-list">{''.join(rows)}</div>
            </section>
            """
        )
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Library ablations</p>
          <h2>Required Ablations</h2>
          <p class="metric-subtle">Quick-eval holdout experiments for supervised and unsupervised libraries.</p>
        </div>
        <div class="stats-inline">
          <span class="pill {'status-running' if healthy_slots == total_slots else 'status-paused'}">{healthy_slots}/{total_slots} healthy</span>
          <span class="pill status-running">Active {active_slots}</span>
          <a class="ghost-link" href="/ablations">Dashboard</a>
        </div>
      </div>
      <div class="required-grid">{''.join(env_cards)}</div>
    </section>
    """


def _render_library_ablation_detail_panel(detail: LibraryAblationDetail | None) -> str:
    if detail is None:
        return """
        <section class="panel">
          <div class="card-topline">
            <div>
              <h2>Library ablation</h2>
              <p class="metric-subtle">This ablation job has not produced a library_evaluations summary yet.</p>
            </div>
          </div>
        </section>
        """
    series = [
        {
            "label": _library_ablation_kind_label(detail.ablation_kind),
            "color": _library_ablation_palette(detail.ablation_kind),
            "points": [
                (float(point.label_count), float(point.mean_success_ratio * 100.0))
                for point in detail.aggregate_points
            ],
            "band": [
                (
                    float(point.label_count),
                    max(0.0, float((point.mean_success_ratio - point.mean_success_ratio_std) * 100.0)),
                    min(100.0, float((point.mean_success_ratio + point.mean_success_ratio_std) * 100.0)),
                )
                for point in detail.aggregate_points
            ],
        }
    ]
    baseline = detail.baseline_case
    baseline_percent = (
        f"{baseline.mean_success_ratio * 100.0:.1f}%"
        if baseline is not None and baseline.mean_success_ratio is not None
        else "—"
    )
    case_rows = "".join(
        f"<tr><td>{html.escape(case.label)}</td><td>{case.label_count}</td><td>{case.completed_seed_count}/{case.planned_seed_count}</td><td>{case.mean_success_ratio * 100.0:.1f}% ± {case.success_ratio_std * 100.0:.1f}%</td></tr>"
        for case in sorted(
            [case for case in detail.case_results if case.mean_success_ratio is not None],
            key=lambda item: (item.label_count, item.label),
        )[:18]
    )
    if not case_rows:
        case_rows = '<tr><td colspan="4" class="empty">No completed runs yet.</td></tr>'
    return f"""
    <section class="panel" id="library-ablation">
      <div class="card-topline">
        <div>
          <h2>Library ablation</h2>
          <p class="metric-subtle">{html.escape(_library_ablation_kind_label(detail.ablation_kind))} library, {html.escape(detail.benchmark_preset)} preset.</p>
        </div>
        <div class="stats-inline">
          <span class="mini-chip">{detail.completed_seed_run_count}/{detail.total_seed_run_count} runs</span>
          <span class="mini-chip">Baseline {html.escape(baseline_percent)}</span>
        </div>
      </div>
      {_render_xy_line_chart(title="Task performance by labels used", x_label="Labels used", y_label="Task success (%)", series=series)}
      <div class="table-scroll compact-table">
        <table class="data-table">
          <thead><tr><th>Case</th><th>Labels used</th><th>Runs</th><th>Mean success ± std</th></tr></thead>
          <tbody>{case_rows}</tbody>
        </table>
      </div>
    </section>
    """


def _render_library_ablations_overview_content(*, jobs: list[JobRecord], project_root: Path) -> str:
    env_ids = list(SUPPORTED_ENVS)
    cards: list[str] = []
    for env_id in env_ids:
        series: list[dict[str, object]] = []
        summary_chips: list[str] = []
        for kind in ("supervised", "unsupervised"):
            _source_root, _source_label, tracked_job, runs = _ablation_plot_runs_for_env_kind(
                jobs=jobs,
                project_root=project_root,
                env_id=env_id,
                kind=kind,
            )
            display_job = _preferred_ablation_job(tracked_job, runs)
            if not runs:
                continue
            detail = runs[0][1]
            points = [
                (float(point.label_count), float(point.mean_success_ratio * 100.0))
                for point in detail.aggregate_points
            ]
            series.append(
                {
                    "label": _library_ablation_kind_label(kind),
                    "color": _library_ablation_palette(kind),
                    "points": points,
                    "band": [
                        (
                            float(point.label_count),
                            max(0.0, float((point.mean_success_ratio - point.mean_success_ratio_std) * 100.0)),
                            min(100.0, float((point.mean_success_ratio + point.mean_success_ratio_std) * 100.0)),
                        )
                        for point in detail.aggregate_points
                    ],
                }
            )
            summary_chips.append(
                f'<span class="mini-chip">{html.escape(_library_ablation_kind_label(kind))} {detail.completed_seed_run_count}/{detail.total_seed_run_count} runs · {html.escape(display_job.status.value if display_job is not None else "untracked")}</span>'
            )
        if not series:
            continue
        cards.append(
            f"""
            <section class="panel">
              <div class="card-topline">
                <div>
                  <p class="section-kicker">Library ablation</p>
                  <h2>{html.escape(env_id.upper())}</h2>
                </div>
                <div class="stats-inline">{''.join(summary_chips) if summary_chips else '<span class="mini-chip">No completed ablation data yet</span>'}</div>
              </div>
              {_render_xy_line_chart(title=f"{env_id.upper()} task performance by labels used", x_label="Labels used", y_label="Task success (%)", series=series)}
            </section>
            """
        )
    return '<div class="content-grid">' + (
        ''.join(cards) if cards else '<section class="panel"><p class="empty">No tracked library ablation jobs yet.</p></section>'
    ) + '</div>'


def render_library_ablations_page(*, jobs: list[JobRecord], project_root: Path) -> str:
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/ablations">Ablations</a> / plots</p>
      <h1>Library Ablations</h1>
      <p>Quick-eval holdout experiments comparing supervised and unsupervised libraries as more labels are kept.</p>
    </section>
    {_render_library_ablations_overview_content(jobs=jobs, project_root=project_root)}
    """
    return _shell("Library Ablations | Pattern Server", active_nav="ablations", body_html=body)


def _job_progress_payload(job: JobRecord) -> dict[str, object] | None:
    progress = job.spec.metadata.get("progress")
    if not isinstance(progress, dict):
        return None
    phase = str(progress.get("phase") or "").strip().lower()
    if phase == "failed" and job.status.value not in {"failed", "cancelled", "stopped", "lost", "retryable_failure"}:
        return None
    return progress


def _is_learning_job(job: JobRecord) -> bool:
    return str(job.spec.action or "") in {
        "supervised-learn-and-evaluate",
        "automatic-launch",
        "unsupervised-launch",
    }


def _learning_progress_summary(
    job: JobRecord,
    *,
    project_root: Path,
    log_lines: list[str] | None = None,
) -> dict[str, Any] | None:
    if not _is_learning_job(job):
        return None
    run_dir = _resolve_job_run_dir(job, project_root=project_root)
    fallback_progress = _job_progress_payload(job)
    lines = tuple(str(line) for line in list(log_lines or job.recent_logs)[-80:])
    return _cached_learning_progress_summary(
        str(run_dir) if run_dir is not None else "",
        _run_dir_cache_marker(run_dir),
        lines,
        job.status.value,
        json.dumps(fallback_progress, sort_keys=True, default=str) if fallback_progress is not None else "",
    )


def _learning_candidate_function_count(
    job: JobRecord,
    *,
    project_root: Path,
) -> int | None:
    if not _is_learning_job(job):
        return None
    run_dir = _resolve_job_run_dir(job, project_root=project_root)
    if run_dir is None:
        return None
    return _cached_learning_candidate_function_count(
        str(run_dir),
        _learning_candidate_function_count_marker(run_dir),
    )


def _learning_overview_stats(*, jobs: list[JobRecord], project_root: Path) -> dict[str, int]:
    unique_run_dirs: dict[str, Path] = {}
    for job in jobs:
        if not _is_learning_job(job):
            continue
        run_dir = _resolve_job_run_dir(job, project_root=project_root)
        if run_dir is None or not run_dir.is_dir():
            continue
        key = str(run_dir.resolve())
        unique_run_dirs[key] = run_dir
    total_candidates = 0
    total_labels = 0
    total_completed_labels = 0
    total_code_lines = 0
    for run_dir in unique_run_dirs.values():
        (
            run_candidates,
            run_labels,
            run_completed_labels,
            run_code_lines,
        ) = _cached_learning_run_aggregate_stats(
            str(run_dir),
            _learning_run_aggregate_marker(run_dir),
        )
        total_candidates += run_candidates
        total_labels += run_labels
        total_completed_labels += run_completed_labels
        total_code_lines += run_code_lines
    return {
        "run_count": len(unique_run_dirs),
        "candidate_count": total_candidates,
        "label_count": total_labels,
        "completed_label_count": total_completed_labels,
        "code_line_count": total_code_lines,
    }


def _iter_env_statistics_run_dirs(*, project_root: Path, env_id: str) -> list[Path]:
    env_root = project_root / "data" / "patterns" / env_id
    if not env_root.is_dir():
        return []
    run_dirs: dict[str, Path] = {}
    for run_json_path in env_root.rglob("run.json"):
        run_dir = run_json_path.parent
        if any(part == "library_evaluations" for part in run_dir.parts):
            continue
        if ".bak" in run_dir.name:
            continue
        if not any(path.is_dir() for path in run_dir.glob("label_*")):
            continue
        key = str(run_dir.resolve())
        run_dirs[key] = run_dir
    return sorted(run_dirs.values(), key=lambda item: str(item))


def _candidate_fitness(payload: dict[str, Any]) -> float | None:
    for key in ("score", "average_score", "val_score", "validation_score", "train_score", "fitness"):
        value = _safe_float(payload.get(key))
        if value is not None:
            return value
    return None


def _run_dir_started_at(run_dir: Path) -> datetime | None:
    run_payload = _load_json_dict(run_dir / "run.json")
    for key in ("started_at", "created_at", "run_started_at", "timestamp"):
        parsed = _parse_iso_datetime(run_payload.get(key))
        if parsed is not None:
            return parsed
    run_json = run_dir / "run.json"
    try:
        if run_json.is_file():
            return datetime.fromtimestamp(run_json.stat().st_mtime, tz=DISPLAY_TZ)
        return datetime.fromtimestamp(run_dir.stat().st_mtime, tz=DISPLAY_TZ)
    except OSError:
        return None


def _candidate_runtime_seconds(payload: dict[str, Any]) -> float | None:
    timing_summary = payload.get("timing_summary")
    if isinstance(timing_summary, dict):
        value = _safe_float(timing_summary.get("total_wall_seconds"))
        if value is not None:
            return value
    timings = payload.get("timings")
    if isinstance(timings, dict):
        totals = timings.get("totals")
        if isinstance(totals, dict):
            for key in ("total_wall_seconds", "total_seconds", "overall_seconds"):
                value = _safe_float(totals.get(key))
                if value is not None:
                    return value
    return None


def _candidate_code_line_count_optional(candidate_dir: Path) -> int | None:
    code_path = candidate_dir / "find_pattern.py"
    if code_path.is_file():
        try:
            with code_path.open(encoding="utf-8") as handle:
                return sum(1 for _ in handle)
        except OSError:
            return None
    candidate_payload_path = candidate_dir / "candidate.json"
    if candidate_payload_path.is_file():
        try:
            payload = json.loads(candidate_payload_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            for key in ("code", "candidate_code", "detector_code", "source_code", "program"):
                raw_code = payload.get(key)
                if isinstance(raw_code, str) and raw_code.strip():
                    return len(raw_code.splitlines())
    return None


def _candidate_code_text(candidate_dir: Path) -> str | None:
    code_path = candidate_dir / "find_pattern.py"
    if code_path.is_file():
        try:
            text = code_path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        if text.strip():
            return text
    candidate_payload_path = candidate_dir / "candidate.json"
    if candidate_payload_path.is_file():
        try:
            payload = json.loads(candidate_payload_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            for key in ("code", "candidate_code", "detector_code", "source_code", "program"):
                raw_code = payload.get(key)
                if isinstance(raw_code, str) and raw_code.strip():
                    return raw_code
    return None


def _is_supervised_statistics_run_dir(run_dir: Path) -> bool:
    normalized_parts = {part.strip().lower() for part in run_dir.parts}
    if "unsupervised" in normalized_parts or "automatic" in normalized_parts:
        return False
    run_name = run_dir.name.strip().lower()
    if "unsupervised" in run_name or "automatic" in run_name:
        return False
    run_payload = _load_json_dict(run_dir / "run.json")
    workflow_kind = str(run_payload.get("workflow_kind") or "").strip().lower()
    if workflow_kind in {"unsupervised", "automatic"}:
        return False
    return True


def _run_dir_is_on_or_after_start_date(run_dir: Path, *, start_date: date | None) -> bool:
    if start_date is None:
        return True
    started_at = _run_dir_started_at(run_dir)
    if started_at is None:
        return False
    return started_at.date() >= start_date


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
        env_value = value.strip().strip("'").strip('"')
        values[env_key] = env_value
    return values


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
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[list[float]] | None:
    if not inputs:
        return []
    api_key, base_url, model = _resolve_openai_embedding_config(project_root)
    if not api_key:
        return None
    if cache_keys is not None and len(cache_keys) != len(inputs):
        return None
    effective_batch_size = max(1, int(batch_size))
    output: list[list[float] | None] = [None] * len(inputs)
    completed = 0
    total = len(inputs)
    missing_indices: list[int] = []
    if cache_keys is not None:
        for index, code_hash in enumerate(cache_keys):
            cached_vector = _load_cached_embedding(project_root=project_root, model=model, code_hash=str(code_hash))
            if cached_vector is not None:
                output[index] = cached_vector
                completed += 1
            else:
                missing_indices.append(index)
    else:
        missing_indices = list(range(len(inputs)))
    if progress_callback is not None and completed > 0:
        progress_callback(completed, total, f"Loaded {completed}/{total} cached embeddings")

    for offset in range(0, len(missing_indices), effective_batch_size):
        batch_indices = missing_indices[offset:offset + effective_batch_size]
        batch = [inputs[item] for item in batch_indices]
        if progress_callback is not None:
            progress_callback(completed, total, f"Embedding batch {offset // effective_batch_size + 1}")
        body = json.dumps({"model": model, "input": batch}).encode("utf-8")
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
        if not isinstance(data, list):
            return None
        batch_vectors: list[list[float]] = []
        for item in data:
            embedding = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(embedding, list):
                return None
            vector: list[float] = []
            for value in embedding:
                numeric = _safe_float(value)
                if numeric is None:
                    return None
                vector.append(float(numeric))
            batch_vectors.append(vector)
        if len(batch_vectors) != len(batch):
            return None
        for position, vector in enumerate(batch_vectors):
            idx = batch_indices[position]
            output[idx] = vector
            if cache_keys is not None:
                _store_cached_embedding(
                    project_root=project_root,
                    model=model,
                    code_hash=str(cache_keys[idx]),
                    embedding=vector,
                )
        completed += len(batch_vectors)
        if progress_callback is not None:
            progress_callback(completed, total, f"Embedded {completed}/{total} code snippets")
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
    if pairs <= 0:
        return 0.0
    return total / pairs


def _load_statistics_train_trace_paths(*, project_root: Path, env_id: str) -> list[Path]:
    datasets = list_environment_datasets(project_root, env_id)
    if not datasets:
        return []
    dataset_choice = datasets[0]
    if env_id == "iphyre":
        preferred = next(
            (
                item
                for item in datasets
                if str(item.root.name).strip().lower() == "default_simulator_trace_dataset"
                or str(item.dataset_name).strip().lower() == "default_simulator_trace_dataset"
            ),
            None,
        )
        if preferred is not None:
            dataset_choice = preferred
    detail = get_environment_dataset(project_root, env_id, dataset_choice.root.name)
    if detail is None:
        return []
    split = next((item for item in detail.splits if str(item.name).strip().lower() == "train"), None)
    if split is None:
        return []
    return [
        item.trace_path
        for item in split.traces
        if isinstance(item.trace_path, Path) and item.trace_path.is_file()
    ]


def _annotation_timestep_set(patterns: list[Any], *, label_key: str, timestep_count: int) -> set[int]:
    if timestep_count <= 0:
        return set()
    active: set[int] = set()
    for pattern in patterns:
        pattern_label_key = _canonical_label_key(str(getattr(pattern, "label", "") or ""))
        if pattern_label_key and pattern_label_key != label_key:
            continue
        try:
            start = int(getattr(pattern, "start_timestep", 0))
        except (TypeError, ValueError):
            start = 0
        try:
            end = int(getattr(pattern, "end_timestep", start))
        except (TypeError, ValueError):
            end = start
        start = max(0, min(start, timestep_count - 1))
        end = max(0, min(end, timestep_count - 1))
        if end < start:
            start, end = end, start
        active.update(range(start, end + 1))
    return active


def _annotation_matrix_distance(
    left: list[set[int]],
    right: list[set[int]],
    *,
    trace_lengths: list[int],
) -> float:
    total_cells = float(sum(max(0, int(length)) for length in trace_lengths))
    if total_cells <= 0.0:
        return 0.0
    diff_cells = 0
    for index, length in enumerate(trace_lengths):
        if length <= 0:
            continue
        left_set = left[index] if index < len(left) else set()
        right_set = right[index] if index < len(right) else set()
        diff_cells += len(left_set.symmetric_difference(right_set))
    return max(0.0, min(1.0, float(diff_cells) / total_cells))


def _sample_candidate_pairs(
    *,
    count: int,
    max_pairs: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    if count < 2:
        return []
    total_pairs = (count * (count - 1)) // 2
    if total_pairs <= max_pairs:
        return [(left, right) for left in range(count) for right in range(left + 1, count)]
    chosen: set[tuple[int, int]] = set()
    while len(chosen) < max_pairs:
        left = rng.randrange(count - 1)
        right = rng.randrange(left + 1, count)
        chosen.add((left, right))
    return list(chosen)


def _stable_seed(*parts: Any) -> int:
    joined = "|".join(str(item) for item in parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _compute_pattern_diversity_by_env(
    *,
    project_root: Path,
    label_hashes: dict[str, dict[str, list[str]]],
    label_candidate_meta: dict[str, dict[str, dict[str, dict[str, str]]]],
    code_by_hash: dict[str, str],
    sampled_hashes: set[str],
    sample_seed: int,
    progress_callback: Callable[[int, int, str], None] | None,
    env_workers: int,
) -> dict[str, dict[str, float]]:
    pattern_diversity_by_env: dict[str, dict[str, float]] = {env_id: {} for env_id in SUPPORTED_ENVS}
    jobs: list[tuple[str, list[Path], dict[str, list[str]], dict[str, dict[str, dict[str, str]]]]] = []
    trace_cap_raw = os.environ.get("PATTERN_STATS_PATTERN_DIVERSITY_MAX_TRACES", "32")
    pair_cap_raw = os.environ.get("PATTERN_STATS_PATTERN_DIVERSITY_MAX_PAIRS", "2000")
    try:
        trace_cap = max(1, int(trace_cap_raw))
    except (TypeError, ValueError):
        trace_cap = 32
    try:
        pair_cap = max(1, int(pair_cap_raw))
    except (TypeError, ValueError):
        pair_cap = 2000

    pattern_units = 0
    for env_id in SUPPORTED_ENVS:
        env_hashes = label_hashes.get(env_id) or {}
        if not env_hashes:
            continue
        trace_paths = _load_statistics_train_trace_paths(project_root=project_root, env_id=env_id)
        if not trace_paths:
            continue
        if len(trace_paths) > trace_cap:
            trace_rng = random.Random(_stable_seed(sample_seed, env_id, "pattern-diversity-traces"))
            trace_paths = trace_rng.sample(trace_paths, trace_cap)
        env_meta = label_candidate_meta.get(env_id) or {}
        env_label_hashes: dict[str, list[str]] = {}
        for label_key, hashes_for_label in env_hashes.items():
            filtered = [
                code_hash
                for code_hash in hashes_for_label
                if code_hash in sampled_hashes and code_hash in (env_meta.get(label_key) or {}) and code_hash in code_by_hash
            ]
            env_label_hashes[label_key] = filtered
            pattern_units += len(set(filtered))
        jobs.append((env_id, trace_paths, env_label_hashes, env_meta))

    if not jobs:
        return pattern_diversity_by_env
    if progress_callback is not None:
        progress_callback(0, max(1, pattern_units), "PATTERN:Preparing annotation similarity scan")

    _ensure_client_src_on_path()
    from deepphy_eval.io import load_trace  # type: ignore
    from pattern_library_detectors import apply_pattern_library  # type: ignore
    from pattern_library_schema import DetectorDefinition, PatternLibrary  # type: ignore

    completed_units = 0
    progress_lock = threading.Lock()

    def _compute_env(
        *,
        env_id: str,
        trace_paths: list[Path],
        env_label_hashes: dict[str, list[str]],
        env_meta: dict[str, dict[str, dict[str, str]]],
    ) -> tuple[str, dict[str, float]]:
        nonlocal completed_units
        trace_objects = [load_trace(path) for path in trace_paths]
        trace_lengths = [max(0, len(getattr(trace, "timesteps", []))) for trace in trace_objects]
        label_scores: dict[str, float] = {}
        for label_key, hash_sequence in env_label_hashes.items():
            if len(hash_sequence) < 2:
                label_scores[label_key] = 0.0
                continue
            unique_hashes = sorted(set(hash_sequence))
            activation_by_hash: dict[str, list[set[int]]] = {}
            label_meta = env_meta.get(label_key) or {}
            for code_hash in unique_hashes:
                code_text = str(code_by_hash.get(code_hash) or "").strip()
                if not code_text:
                    continue
                spec = label_meta.get(code_hash) or {}
                detector_label = str(spec.get("label") or label_key).strip() or label_key
                input_mode_raw = str(spec.get("input_mode") or "trace").strip()
                detector = DetectorDefinition(
                    detector_id=f"stats-{env_id}-{label_key[:24]}-{code_hash[:10]}",
                    label=detector_label,
                    environment_ids=[env_id],
                    source_code=code_text,
                    input_mode=input_mode_raw if input_mode_raw in {"trace", "supervised_evidence"} else "trace",
                )
                library = PatternLibrary(
                    library_id=f"stats-pattern-diversity-{env_id}-{label_key}-{code_hash[:8]}",
                    environment_ids=[env_id],
                    primitive_source="statistics_pattern_diversity",
                    detectors=[detector],
                )
                per_trace_active: list[set[int]] = []
                for trace_index, trace in enumerate(trace_objects):
                    patterns = apply_pattern_library(trace, library)
                    active = _annotation_timestep_set(
                        list(patterns),
                        label_key=label_key,
                        timestep_count=trace_lengths[trace_index] if trace_index < len(trace_lengths) else 0,
                    )
                    per_trace_active.append(active)
                activation_by_hash[code_hash] = per_trace_active
                if progress_callback is not None:
                    with progress_lock:
                        completed_units += 1
                        progress_callback(
                            completed_units,
                            max(1, pattern_units),
                            f"PATTERN:{env_id}/{label_key}",
                        )
            effective_hash_sequence = [code_hash for code_hash in hash_sequence if code_hash in activation_by_hash]
            if len(effective_hash_sequence) < 2:
                label_scores[label_key] = 0.0
                continue
            pair_rng = random.Random(_stable_seed(sample_seed, env_id, label_key, len(effective_hash_sequence)))
            pair_indices = _sample_candidate_pairs(
                count=len(effective_hash_sequence),
                max_pairs=pair_cap,
                rng=pair_rng,
            )
            if not pair_indices:
                label_scores[label_key] = 0.0
                continue
            similarity_total = 0.0
            for left_index, right_index in pair_indices:
                left_hash = effective_hash_sequence[left_index]
                right_hash = effective_hash_sequence[right_index]
                distance = _annotation_matrix_distance(
                    activation_by_hash[left_hash],
                    activation_by_hash[right_hash],
                    trace_lengths=trace_lengths,
                )
                similarity_total += 1.0 - distance
            mean_similarity = similarity_total / float(len(pair_indices))
            label_scores[label_key] = max(0.0, min(1.0, 1.0 - mean_similarity))
        return env_id, label_scores

    future_map = {}
    with ThreadPoolExecutor(max_workers=max(1, min(env_workers, len(jobs)))) as executor:
        for env_id, trace_paths, env_label_hashes, env_meta in jobs:
            future = executor.submit(
                _compute_env,
                env_id=env_id,
                trace_paths=trace_paths,
                env_label_hashes=env_label_hashes,
                env_meta=env_meta,
            )
            future_map[future] = env_id
        for future in as_completed(future_map):
            env_id, label_scores = future.result()
            pattern_diversity_by_env[env_id] = label_scores
    return pattern_diversity_by_env


def _canonical_label_key(label: str) -> str:
    raw = str(label or "").strip().lower()
    if not raw:
        return ""
    tokens = [token for token in re.split(r"[^a-z0-9]+", raw) if token]
    return " ".join(tokens) if tokens else raw


def _load_supervised_label_keys_for_env(*, project_root: Path, env_id: str) -> set[str]:
    path = project_root / "configs" / env_id / "labels.txt"
    if not path.is_file():
        return set()
    labels = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return {_canonical_label_key(label) for label in labels if _canonical_label_key(label)}


def _statistics_activation_cache_root(project_root: Path) -> Path:
    return project_root / "state" / "statistics_activation_cache"


def _activation_cache_path(*, project_root: Path, env_id: str) -> Path:
    return _statistics_activation_cache_root(project_root) / env_id / "activation_rates.json"


def _statistics_success_corr_cache_root(project_root: Path) -> Path:
    return project_root / "state" / "statistics_success_correlation_cache"


def _success_corr_cache_path(*, project_root: Path, env_id: str) -> Path:
    return _statistics_success_corr_cache_root(project_root) / env_id / "success_correlation.json"


def _activation_cache_signature(
    *,
    dataset_manifest_path: Path,
    library_path: Path,
    split_name: str,
) -> dict[str, Any]:
    def _mtime(path: Path) -> float:
        try:
            return float(path.stat().st_mtime)
        except OSError:
            return 0.0

    return {
        "schema_version": "pattern_server.activation_rate.v1",
        "dataset_manifest_path": str(dataset_manifest_path.resolve()),
        "dataset_manifest_mtime": _mtime(dataset_manifest_path),
        "library_path": str(library_path.resolve()),
        "library_mtime": _mtime(library_path),
        "split_name": str(split_name or "").strip().lower() or "train",
    }


def _task_results_fingerprint(task_result_paths: list[Path]) -> str:
    pieces: list[str] = []
    for path in sorted(task_result_paths, key=lambda item: str(item)):
        try:
            mtime = int(path.stat().st_mtime_ns)
        except OSError:
            mtime = 0
        pieces.append(f"{str(path.resolve())}|{mtime}")
    return hashlib.sha1("\n".join(pieces).encode("utf-8")).hexdigest()


def _success_corr_signature(
    *,
    library_path: Path,
    task_result_paths: list[Path],
) -> dict[str, Any]:
    def _mtime(path: Path) -> float:
        try:
            return float(path.stat().st_mtime)
        except OSError:
            return 0.0

    return {
        "schema_version": "pattern_server.success_correlation.v1",
        "library_path": str(library_path.resolve()),
        "library_mtime": _mtime(library_path),
        "task_result_count": len(task_result_paths),
        "task_results_fingerprint": _task_results_fingerprint(task_result_paths),
    }


def _load_activation_cache(
    *,
    project_root: Path,
    env_id: str,
    signature: dict[str, Any],
) -> dict[str, Any] | None:
    path = _activation_cache_path(project_root=project_root, env_id=env_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    cached_sig = payload.get("signature")
    if not isinstance(cached_sig, dict):
        return None
    for key, value in signature.items():
        if cached_sig.get(key) != value:
            return None
    return payload


def _load_success_corr_cache(
    *,
    project_root: Path,
    env_id: str,
    signature: dict[str, Any],
) -> dict[str, Any] | None:
    path = _success_corr_cache_path(project_root=project_root, env_id=env_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    cached_sig = payload.get("signature")
    if not isinstance(cached_sig, dict):
        return None
    for key, value in signature.items():
        if cached_sig.get(key) != value:
            return None
    return payload


def _save_success_corr_cache(
    *,
    project_root: Path,
    env_id: str,
    signature: dict[str, Any],
    correlations: dict[str, float | None],
    task_count: int,
    usable_task_count: int,
) -> None:
    path = _success_corr_cache_path(project_root=project_root, env_id=env_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "signature": signature,
        "task_count": int(task_count),
        "usable_task_count": int(usable_task_count),
        "correlations": {
            str(key): (float(value) if isinstance(value, (int, float)) else None)
            for key, value in correlations.items()
        },
        "computed_at": datetime.now(tz=ZoneInfo("UTC")).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_task_trace_path(
    *,
    project_root: Path,
    task_result_path: Path,
    task_payload: dict[str, Any],
) -> Path | None:
    raw_trace = task_payload.get("final_trace_path")
    if not isinstance(raw_trace, str) or not raw_trace.strip():
        return None
    text = raw_trace.strip()
    candidates = [
        Path(text),
        task_result_path.parent / text,
        project_root / text,
    ]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file():
            return resolved
    return None


def _phi_coefficient(a: int, b: int, c: int, d: int) -> float | None:
    denominator = (a + b) * (c + d) * (a + c) * (b + d)
    if denominator <= 0:
        return None
    return float((a * d - b * c) / math.sqrt(float(denominator)))


def _compute_task_success_correlations_by_env(
    *,
    project_root: Path,
    supervised_label_keys_by_env: dict[str, set[str]],
) -> dict[str, dict[str, float | None]]:
    correlations_by_env: dict[str, dict[str, float | None]] = {env_id: {} for env_id in SUPPORTED_ENVS}
    jobs: list[tuple[str, Path, list[Path], dict[str, Any], set[str]]] = []
    for env_id in SUPPORTED_ENVS:
        supervised_keys = supervised_label_keys_by_env.get(env_id) or set()
        if not supervised_keys:
            continue
        try:
            library_path = resolve_canonical_library(project_root, env_id, "supervised")
        except Exception:
            continue
        eval_root = project_root / "data" / "evaluations" / env_id
        task_result_paths = sorted(eval_root.glob("**/tasks/*/*/task_result.json")) if eval_root.is_dir() else []
        if not task_result_paths:
            continue
        signature = _success_corr_signature(
            library_path=library_path,
            task_result_paths=task_result_paths,
        )
        cached = _load_success_corr_cache(
            project_root=project_root,
            env_id=env_id,
            signature=signature,
        )
        if isinstance(cached, dict):
            raw_corr = cached.get("correlations")
            if isinstance(raw_corr, dict):
                correlations_by_env[env_id] = {
                    str(key): (float(value) if isinstance(value, (int, float)) else None)
                    for key, value in raw_corr.items()
                }
                continue
        jobs.append((env_id, library_path, task_result_paths, signature, supervised_keys))

    if not jobs:
        return correlations_by_env

    _ensure_client_src_on_path()
    from deepphy_eval.io import load_trace  # type: ignore
    from pattern_library_detectors import apply_pattern_library  # type: ignore
    from pattern_library_registry import load_pattern_library  # type: ignore

    env_worker_raw = os.environ.get("PATTERN_STATS_ENV_WORKERS", "4")
    try:
        env_workers = max(1, min(len(jobs), int(env_worker_raw)))
    except (TypeError, ValueError):
        env_workers = max(1, min(len(jobs), 4))

    def _compute_for_env(
        *,
        env_id: str,
        library_path: Path,
        task_result_paths: list[Path],
        signature: dict[str, Any],
        supervised_keys: set[str],
    ) -> tuple[str, dict[str, float | None], dict[str, Any], int, int]:
        library = load_pattern_library(library_path)
        trace_activation_cache: dict[str, set[str]] = {}
        counts: dict[str, dict[str, int]] = {
            key: {"a": 0, "b": 0, "c": 0, "d": 0}
            for key in supervised_keys
        }
        task_count = 0
        usable_task_count = 0
        for task_result_path in task_result_paths:
            try:
                task_payload = json.loads(task_result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(task_payload, dict):
                continue
            task_count += 1
            success = task_payload.get("success")
            if not isinstance(success, bool):
                continue
            trace_path = _resolve_task_trace_path(
                project_root=project_root,
                task_result_path=task_result_path,
                task_payload=task_payload,
            )
            if trace_path is None:
                continue
            trace_key = str(trace_path)
            activated = trace_activation_cache.get(trace_key)
            if activated is None:
                trace = load_trace(trace_path)
                patterns = apply_pattern_library(trace, library)
                activated = set()
                for pattern in patterns:
                    label = str(getattr(pattern, "label", "") or "").strip()
                    key = _canonical_label_key(label)
                    if key and key in supervised_keys:
                        activated.add(key)
                trace_activation_cache[trace_key] = activated
            usable_task_count += 1
            for key in supervised_keys:
                if key in activated:
                    counts[key]["a" if success else "b"] += 1
                else:
                    counts[key]["c" if success else "d"] += 1
        correlations: dict[str, float | None] = {}
        for key in supervised_keys:
            bucket = counts.get(key) or {}
            correlations[key] = _phi_coefficient(
                int(bucket.get("a") or 0),
                int(bucket.get("b") or 0),
                int(bucket.get("c") or 0),
                int(bucket.get("d") or 0),
            )
        return env_id, correlations, signature, task_count, usable_task_count

    future_map = {}
    with ThreadPoolExecutor(max_workers=env_workers) as executor:
        for env_id, library_path, task_result_paths, signature, supervised_keys in jobs:
            future = executor.submit(
                _compute_for_env,
                env_id=env_id,
                library_path=library_path,
                task_result_paths=task_result_paths,
                signature=signature,
                supervised_keys=supervised_keys,
            )
            future_map[future] = env_id
        for future in as_completed(future_map):
            env_id, correlations, signature, task_count, usable_task_count = future.result()
            correlations_by_env[env_id] = correlations
            _save_success_corr_cache(
                project_root=project_root,
                env_id=env_id,
                signature=signature,
                correlations=correlations,
                task_count=task_count,
                usable_task_count=usable_task_count,
            )
    return correlations_by_env


def _select_best_statistics_ablation_detail(
    *,
    project_root: Path,
) -> dict[str, Any]:
    from .library_ablation import LibraryAblationDetail, load_library_ablation_job_detail
    from .state import ServerStateStore

    store = ServerStateStore(project_root / "state", project_root=project_root)
    selected: dict[str, tuple[tuple[int, int, float], LibraryAblationDetail]] = {}
    for job in store.list_jobs():
        if str(job.spec.action or "").strip() != "evaluate-library-case":
            continue
        metadata = dict(job.spec.metadata or {})
        env_id = str(metadata.get("env_id") or "").strip().lower()
        kind = str(metadata.get("library_ablation_kind") or "").strip().lower()
        if env_id not in SUPPORTED_ENVS or kind != "supervised":
            continue
        detail = load_library_ablation_job_detail(job, project_root=project_root)
        if detail is None:
            continue
        observed_single_cases = sum(
            1
            for case in detail.case_results
            if len(case.held_out_labels) == 1 and case.mean_success_ratio is not None
        )
        score = (
            1 if observed_single_cases > 0 else 0,
            observed_single_cases if observed_single_cases > 0 else detail.completed_seed_run_count,
            float(job.updated_at.timestamp()),
        )
        current = selected.get(env_id)
        if current is None or score > current[0]:
            selected[env_id] = (score, detail)
    return {env_id: detail for env_id, (_score, detail) in selected.items()}


def _estimate_label_drop_from_size_curve(
    *,
    average_single_label_drop_pp: float | None,
    success_corr: float | None,
    correlations: list[float],
) -> float | None:
    if average_single_label_drop_pp is None:
        return None
    baseline_sign = 1.0 if average_single_label_drop_pp >= 0.0 else -1.0
    magnitude = abs(float(average_single_label_drop_pp))
    nonzero_corrs = [abs(value) for value in correlations if isinstance(value, (int, float)) and not math.isclose(float(value), 0.0)]
    if not isinstance(success_corr, (int, float)) or not nonzero_corrs:
        return baseline_sign * magnitude
    max_abs_corr = max(nonzero_corrs)
    if max_abs_corr <= 0.0:
        return baseline_sign * magnitude
    normalized = max(-1.0, min(1.0, float(success_corr) / max_abs_corr))
    magnitude_scale = 0.35 + 0.65 * abs(normalized)
    sign = baseline_sign if normalized >= 0.0 else -baseline_sign
    return sign * magnitude * magnitude_scale


def _compute_ablation_drop_estimates_by_env(
    *,
    project_root: Path,
    supervised_label_keys_by_env: dict[str, set[str]],
    success_corr_by_env: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, float | None]]:
    estimates_by_env: dict[str, dict[str, float | None]] = {env_id: {} for env_id in SUPPORTED_ENVS}
    details_by_env = _select_best_statistics_ablation_detail(project_root=project_root)
    for env_id in SUPPORTED_ENVS:
        detail = details_by_env.get(env_id)
        supervised_keys = supervised_label_keys_by_env.get(env_id) or set()
        correlations_map = success_corr_by_env.get(env_id) or {}
        if detail is None or not supervised_keys:
            continue
        baseline_case = detail.baseline_case
        baseline_ratio: float | None = None
        full_label_count: int | None = None
        if baseline_case is not None and baseline_case.mean_success_ratio is not None:
            baseline_ratio = float(baseline_case.mean_success_ratio)
            full_label_count = baseline_case.label_count
        else:
            aggregate_points = detail.aggregate_points
            if aggregate_points:
                best_point = max(aggregate_points, key=lambda item: item.label_count)
                baseline_ratio = float(best_point.mean_success_ratio)
                full_label_count = int(best_point.label_count)

        observed_single_drops: list[float] = []
        for case in detail.case_results:
            if len(case.held_out_labels) != 1 or case.mean_success_ratio is None or baseline_ratio is None:
                continue
            held_out_key = _canonical_label_key(case.held_out_labels[0])
            if not held_out_key:
                continue
            drop_pp = (baseline_ratio - float(case.mean_success_ratio)) * 100.0
            estimates_by_env[env_id][held_out_key] = drop_pp
            observed_single_drops.append(drop_pp)

        average_single_label_drop_pp: float | None = None
        if observed_single_drops:
            average_single_label_drop_pp = float(statistics.fmean(observed_single_drops))
        elif baseline_ratio is not None and full_label_count is not None:
            candidate_points = [
                point
                for point in detail.aggregate_points
                if int(point.label_count) < int(full_label_count)
            ]
            one_less = next(
                (point for point in candidate_points if int(point.label_count) == int(full_label_count) - 1),
                candidate_points[-1] if candidate_points else None,
            )
            if one_less is not None:
                average_single_label_drop_pp = (baseline_ratio - float(one_less.mean_success_ratio)) * 100.0

        corr_values = [
            float(value)
            for key, value in correlations_map.items()
            if key in supervised_keys and isinstance(value, (int, float))
        ]
        for label_key in supervised_keys:
            if label_key in estimates_by_env[env_id]:
                continue
            estimated = _estimate_label_drop_from_size_curve(
                average_single_label_drop_pp=average_single_label_drop_pp,
                success_corr=correlations_map.get(label_key),
                correlations=corr_values,
            )
            estimates_by_env[env_id][label_key] = estimated
    return estimates_by_env


def _compute_activation_rates_by_env(
    *,
    project_root: Path,
    supervised_label_keys_by_env: dict[str, set[str]],
    progress_callback: Callable[[int, int, str], None] | None,
) -> dict[str, dict[str, float]]:
    activation_rate_by_env: dict[str, dict[str, float]] = {}
    if progress_callback is not None:
        progress_callback(0, 1, "TRACE:Preparing activation-rate scan across train traces")
    # Activation rate is computed over the latest dataset for each env using the canonical
    # supervised library shown on /library. We cache results on disk because applying the full
    # library across all train traces can be expensive.
    trace_work_units = 0
    activation_jobs: list[tuple[str, Path, Path, str, list[Path]]] = []
    for env_id in SUPPORTED_ENVS:
        activation_rate_by_env[env_id] = {}
        supervised_keys = supervised_label_keys_by_env.get(env_id) or set()
        if not supervised_keys:
            continue
        try:
            library_path = resolve_canonical_library(project_root, env_id, "supervised")
        except Exception:
            continue
        datasets = list_environment_datasets(project_root, env_id)
        if not datasets:
            continue
        dataset_choice = datasets[0]
        if env_id == "iphyre":
            preferred = next(
                (
                    item
                    for item in datasets
                    if str(item.root.name).strip().lower() == "default_simulator_trace_dataset"
                    or str(item.dataset_name).strip().lower() == "default_simulator_trace_dataset"
                ),
                None,
            )
            if preferred is not None:
                dataset_choice = preferred
        detail = get_environment_dataset(project_root, env_id, dataset_choice.root.name)
        if detail is None:
            continue
        split = next((item for item in detail.splits if str(item.name).strip().lower() == "train"), None)
        if split is None:
            continue
        trace_paths = [
            item.trace_path
            for item in split.traces
            if isinstance(item.trace_path, Path) and item.trace_path.is_file()
        ]
        if not trace_paths:
            continue
        signature = _activation_cache_signature(
            dataset_manifest_path=detail.manifest_path,
            library_path=library_path,
            split_name=split.name,
        )
        cached = _load_activation_cache(project_root=project_root, env_id=env_id, signature=signature)
        if cached is not None:
            raw_counts = cached.get("activated_counts")
            try:
                trace_count = int(cached.get("trace_count") or 0)
            except (TypeError, ValueError):
                trace_count = 0
            counts = (
                {str(k): int(v) for k, v in (raw_counts or {}).items()}
                if isinstance(raw_counts, dict)
                else {}
            )
            if trace_count > 0:
                for key in supervised_keys:
                    activation_rate_by_env[env_id][key] = float(counts.get(key, 0)) / float(trace_count)
            continue
        activation_jobs.append((env_id, library_path, detail.manifest_path, split.name, trace_paths))
        trace_work_units += len(trace_paths)

    if not activation_jobs:
        return activation_rate_by_env

    _ensure_client_src_on_path()
    from deepphy_eval.io import load_trace  # type: ignore
    from pattern_library_detectors import apply_pattern_library  # type: ignore
    from pattern_library_registry import load_pattern_library  # type: ignore

    env_worker_raw = os.environ.get("PATTERN_STATS_ENV_WORKERS", "4")
    try:
        env_workers = max(1, min(len(activation_jobs), int(env_worker_raw)))
    except (TypeError, ValueError):
        env_workers = max(1, min(len(activation_jobs), 4))
    completed_traces = 0
    trace_progress_lock = threading.Lock()
    if progress_callback is not None:
        progress_callback(0, max(1, trace_work_units), "TRACE:Applying canonical supervised libraries to train traces")

    def _process_activation_job(
        *,
        env_id: str,
        library_path: Path,
        manifest_path: Path,
        split_name: str,
        trace_paths: list[Path],
    ) -> tuple[str, dict[str, float], dict[str, Any], int, dict[str, int]]:
        nonlocal completed_traces
        supervised_keys = supervised_label_keys_by_env.get(env_id) or set()
        if not supervised_keys:
            signature = _activation_cache_signature(
                dataset_manifest_path=manifest_path,
                library_path=library_path,
                split_name=split_name,
            )
            return env_id, {}, signature, 0, {}
        library = load_pattern_library(library_path)
        activated_counts: dict[str, int] = {key: 0 for key in supervised_keys}
        trace_count = 0
        for trace_path in trace_paths:
            trace_count += 1
            trace = load_trace(trace_path)
            patterns = apply_pattern_library(trace, library)
            fired: set[str] = set()
            for pattern in patterns:
                label = str(getattr(pattern, "label", "") or "").strip()
                key = _canonical_label_key(label)
                if key and key in supervised_keys:
                    fired.add(key)
            for key in fired:
                activated_counts[key] = int(activated_counts.get(key, 0)) + 1
            if progress_callback is not None:
                with trace_progress_lock:
                    completed_traces += 1
                    progress_callback(
                        completed_traces,
                        max(1, trace_work_units),
                        f"TRACE:{env_id}/{trace_path.parent.name}",
                    )
        rates: dict[str, float] = {}
        if trace_count > 0:
            for key in supervised_keys:
                rates[key] = float(activated_counts.get(key, 0)) / float(trace_count)
        signature = _activation_cache_signature(
            dataset_manifest_path=manifest_path,
            library_path=library_path,
            split_name=split_name,
        )
        return env_id, rates, signature, trace_count, activated_counts

    future_map = {}
    with ThreadPoolExecutor(max_workers=env_workers) as executor:
        for env_id, library_path, manifest_path, split_name, trace_paths in activation_jobs:
            future = executor.submit(
                _process_activation_job,
                env_id=env_id,
                library_path=library_path,
                manifest_path=manifest_path,
                split_name=split_name,
                trace_paths=trace_paths,
            )
            future_map[future] = env_id
        for future in as_completed(future_map):
            env_id, rates, signature, trace_count, activated_counts = future.result()
            activation_rate_by_env[env_id].update(rates)
            _save_activation_cache(
                project_root=project_root,
                env_id=env_id,
                signature=signature,
                trace_count=trace_count,
                activated_counts=activated_counts,
            )
    return activation_rate_by_env


def _save_activation_cache(
    *,
    project_root: Path,
    env_id: str,
    signature: dict[str, Any],
    trace_count: int,
    activated_counts: dict[str, int],
) -> None:
    path = _activation_cache_path(project_root=project_root, env_id=env_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "signature": signature,
        "trace_count": int(trace_count),
        "activated_counts": {str(k): int(v) for k, v in activated_counts.items()},
        "computed_at": datetime.now(tz=ZoneInfo("UTC")).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compute_activation_rates_snapshot(
    *,
    project_root: Path,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute per-label activation rates across the latest train split for each env.

    Activation rate is defined as the fraction of train traces where a label fires at least once,
    using the canonical supervised library (same source as /library). Values are keyed by the
    canonical label key (lowercased, non-alnum normalized).
    """
    supervised_label_keys_by_env = {
        env_id: _load_supervised_label_keys_for_env(project_root=project_root, env_id=env_id)
        for env_id in SUPPORTED_ENVS
    }
    return _compute_activation_rates_by_env(
        project_root=project_root,
        supervised_label_keys_by_env=supervised_label_keys_by_env,
        progress_callback=progress_callback,
    )


def compute_success_correlations_snapshot(
    *,
    project_root: Path,
) -> dict[str, dict[str, float | None]]:
    """Compute or load cached per-label task-success correlations for each env."""
    supervised_label_keys_by_env = {
        env_id: _load_supervised_label_keys_for_env(project_root=project_root, env_id=env_id)
        for env_id in SUPPORTED_ENVS
    }
    return _compute_task_success_correlations_by_env(
        project_root=project_root,
        supervised_label_keys_by_env=supervised_label_keys_by_env,
    )


def patch_statistics_rows_with_activation_rates(
    *,
    rows_by_env: dict[str, Any],
    activation_rates_by_env: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Mutates and returns rows_by_env by filling activation_rate_display where possible."""
    if not isinstance(rows_by_env, dict):
        return rows_by_env
    for env_id, rows_raw in list(rows_by_env.items()):
        if not isinstance(rows_raw, list):
            continue
        rates = activation_rates_by_env.get(str(env_id).strip().lower(), {})
        if not isinstance(rates, dict):
            rates = {}
        for row in rows_raw:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label") or "").strip()
            key = _canonical_label_key(label)
            rate = rates.get(key)
            if isinstance(rate, (int, float)):
                row["activation_rate_display"] = f"{float(rate) * 100.0:.1f}%"
    return rows_by_env


def patch_statistics_rows_with_success_correlations(
    *,
    rows_by_env: dict[str, Any],
    success_correlations_by_env: dict[str, dict[str, float | None]],
) -> dict[str, Any]:
    """Mutates and returns rows_by_env by filling success_correlation_display where possible."""
    if not isinstance(rows_by_env, dict):
        return rows_by_env
    for env_id, rows_raw in list(rows_by_env.items()):
        if not isinstance(rows_raw, list):
            continue
        correlations = success_correlations_by_env.get(str(env_id).strip().lower(), {})
        if not isinstance(correlations, dict):
            correlations = {}
        for row in rows_raw:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label") or "").strip()
            key = _canonical_label_key(label)
            corr = correlations.get(key)
            if isinstance(corr, (int, float)):
                row["success_correlation_display"] = f"{float(corr):+.3f}"
    return rows_by_env


def patch_statistics_rows_with_ablation_estimates(
    *,
    rows_by_env: dict[str, Any],
    ablation_estimates_by_env: dict[str, dict[str, float | None]],
) -> dict[str, Any]:
    """Mutates and returns rows_by_env by filling ablation_drop_display where possible."""
    if not isinstance(rows_by_env, dict):
        return rows_by_env
    for env_id, rows_raw in list(rows_by_env.items()):
        if not isinstance(rows_raw, list):
            continue
        estimates = ablation_estimates_by_env.get(str(env_id).strip().lower(), {})
        if not isinstance(estimates, dict):
            estimates = {}
        for row in rows_raw:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label") or "").strip()
            key = _canonical_label_key(label)
            estimate = estimates.get(key)
            if isinstance(estimate, (int, float)):
                row["ablation_drop_display"] = f"{float(estimate):.1f}pp"
    return rows_by_env


def patch_statistics_rows_with_diversity(
    *,
    rows_by_env: dict[str, Any],
    diversity_rows_by_env: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Mutates and returns rows_by_env by filling code/pattern diversity fields where possible."""
    if not isinstance(rows_by_env, dict):
        return rows_by_env
    for env_id, rows_raw in list(rows_by_env.items()):
        if not isinstance(rows_raw, list):
            continue
        diversity_rows = diversity_rows_by_env.get(str(env_id).strip().lower(), [])
        if not isinstance(diversity_rows, list):
            diversity_rows = []
        diversity_by_key: dict[str, dict[str, Any]] = {}
        for item in diversity_rows:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            key = _canonical_label_key(label)
            if key:
                diversity_by_key[key] = item
        for row in rows_raw:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label") or "").strip()
            key = _canonical_label_key(label)
            matched = diversity_by_key.get(key)
            if not isinstance(matched, dict):
                continue
            code_value = matched.get("code_diversity_value")
            pattern_value = matched.get("pattern_diversity_value")
            if isinstance(code_value, (int, float)):
                row["code_diversity_value"] = float(code_value)
                row["code_diversity_display"] = f"{float(code_value):.3f}"
            if isinstance(pattern_value, (int, float)):
                row["pattern_diversity_value"] = float(pattern_value)
                row["pattern_diversity_display"] = f"{float(pattern_value):.3f}"
    return rows_by_env


def compute_statistics_diversity_snapshot(
    *,
    project_root: Path,
    start_date: date | None = None,
    sample_seed: int | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    code_by_hash: dict[str, str] = {}
    label_hashes: dict[str, dict[str, list[str]]] = {}
    label_candidate_meta: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    label_counts: dict[str, dict[str, int]] = {}
    label_display_names: dict[str, dict[str, str]] = {}
    supervised_label_keys_by_env: dict[str, set[str]] = {
        env_id: _load_supervised_label_keys_for_env(project_root=project_root, env_id=env_id)
        for env_id in SUPPORTED_ENVS
    }
    env_worker_raw = os.environ.get("PATTERN_STATS_ENV_WORKERS", "4")
    try:
        env_workers = max(1, min(len(SUPPORTED_ENVS), int(env_worker_raw)))
    except (TypeError, ValueError):
        env_workers = max(1, min(len(SUPPORTED_ENVS), 4))

    def _collect_env_stats(
        env_id: str,
    ) -> tuple[
        str,
        dict[str, list[str]],
        dict[str, dict[str, dict[str, str]]],
        dict[str, int],
        dict[str, str],
        dict[str, str],
    ]:
        supervised_keys = supervised_label_keys_by_env.get(env_id) or set()
        run_dirs = [
            run_dir
            for run_dir in _iter_env_statistics_run_dirs(project_root=project_root, env_id=env_id)
            if _is_supervised_statistics_run_dir(run_dir)
            and _run_dir_is_on_or_after_start_date(run_dir, start_date=start_date)
        ]
        by_label_hashes: dict[str, list[str]] = defaultdict(list)
        by_label_counts: dict[str, int] = defaultdict(int)
        by_label_display_name: dict[str, str] = {}
        by_label_candidate_meta: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
        local_code_by_hash: dict[str, str] = {}
        for run_dir in run_dirs:
            for label_dir in _unique_label_dirs(run_dir):
                label_name = _label_name_from_dir(label_dir.name)
                label_key = _canonical_label_key(label_name)
                if not label_key:
                    label_key = _canonical_label_key(label_dir.name) or label_dir.name
                if supervised_keys and label_key not in supervised_keys:
                    continue
                by_label_display_name.setdefault(label_key, label_key or label_name)
                for candidate_dir in _label_unique_candidate_dirs(label_dir):
                    payload = _load_json_dict(candidate_dir / "candidate.json")
                    by_label_counts[label_key] += 1
                    code_text = _candidate_code_text(candidate_dir)
                    if not isinstance(code_text, str) or not code_text.strip():
                        continue
                    code_hash = hashlib.sha1(code_text.encode("utf-8")).hexdigest()
                    by_label_hashes[label_key].append(code_hash)
                    local_code_by_hash.setdefault(code_hash, code_text)
                    input_mode_raw = str(payload.get("input_mode") or "trace").strip().lower()
                    by_label_candidate_meta[label_key].setdefault(
                        code_hash,
                        {
                            "label": str(by_label_display_name.get(label_key) or label_name or label_key),
                            "input_mode": (
                                input_mode_raw if input_mode_raw in {"trace", "supervised_evidence"} else "trace"
                            ),
                        },
                    )
        return (
            env_id,
            dict(by_label_hashes),
            {label_key: dict(meta) for label_key, meta in by_label_candidate_meta.items()},
            {key: int(value) for key, value in by_label_counts.items()},
            dict(by_label_display_name),
            local_code_by_hash,
        )

    future_map = {}
    with ThreadPoolExecutor(max_workers=env_workers) as executor:
        for env_id in SUPPORTED_ENVS:
            future = executor.submit(_collect_env_stats, env_id)
            future_map[future] = env_id
        for future in as_completed(future_map):
            (
                env_id,
                env_label_hashes,
                env_label_candidate_meta,
                env_label_counts,
                env_label_display_names,
                env_code_by_hash,
            ) = future.result()
            label_hashes[env_id] = env_label_hashes
            label_candidate_meta[env_id] = env_label_candidate_meta
            label_counts[env_id] = env_label_counts
            label_display_names[env_id] = env_label_display_names
            for code_hash, code_text in env_code_by_hash.items():
                code_by_hash.setdefault(code_hash, code_text)
        for env_id in SUPPORTED_ENVS:
            label_hashes.setdefault(env_id, {})
            label_candidate_meta.setdefault(env_id, {})
            label_counts.setdefault(env_id, {})
            label_display_names.setdefault(env_id, {})

    hashes = list(code_by_hash.keys())
    diversity_cap_raw = os.environ.get("PATTERN_STATS_DIVERSITY_MAX_CODES", "5000")
    try:
        diversity_cap = max(1, int(diversity_cap_raw))
    except (TypeError, ValueError):
        diversity_cap = 5000
    sampled_hashes = hashes
    resolved_sample_seed: int
    if isinstance(sample_seed, int):
        resolved_sample_seed = sample_seed
    else:
        sample_seed_raw = os.environ.get("PATTERN_STATS_DIVERSITY_SAMPLE_SEED", "0")
        try:
            resolved_sample_seed = int(sample_seed_raw)
        except (TypeError, ValueError):
            resolved_sample_seed = 0
    if len(sampled_hashes) > diversity_cap:
        sampled_hashes = random.Random(resolved_sample_seed).sample(sorted(sampled_hashes), diversity_cap)
    sampled_hash_set = set(sampled_hashes)
    total_labels = sum(len(items) for items in label_hashes.values())
    embedding_units = len(sampled_hashes)
    if progress_callback is not None:
        progress_callback(
            0,
            max(1, embedding_units),
            f"EMBED:Requesting embeddings for {embedding_units}/{len(hashes)} sampled supervised code snippets",
        )
    vectors = _fetch_openai_embeddings(
        inputs=[code_by_hash[item] for item in sampled_hashes],
        cache_keys=sampled_hashes,
        project_root=project_root,
        batch_size=max(1, int(os.environ.get("PATTERN_STATS_EMBED_BATCH_SIZE", "64"))),
        progress_callback=(
            (
                lambda completed, _total, message: progress_callback(
                    min(embedding_units, int(completed)),
                    max(1, embedding_units),
                    f"EMBED:{message}",
                )
            )
            if progress_callback is not None
            else None
        ),
    )
    if vectors is None or len(vectors) != len(sampled_hashes):
        raise RuntimeError("Unable to compute embeddings for diversity")
    embedding_by_hash = {sampled_hashes[index]: vectors[index] for index in range(len(sampled_hashes))}
    pattern_diversity_by_env = _compute_pattern_diversity_by_env(
        project_root=project_root,
        label_hashes=label_hashes,
        label_candidate_meta=label_candidate_meta,
        code_by_hash=code_by_hash,
        sampled_hashes=sampled_hash_set,
        sample_seed=resolved_sample_seed,
        progress_callback=progress_callback,
        env_workers=env_workers,
    )

    completed = 0
    label_progress_lock = threading.Lock()

    def _build_env_rows(env_id: str) -> tuple[str, list[dict[str, Any]]]:
        nonlocal completed
        env_rows: list[dict[str, Any]] = []
        for label_key, hashes_for_label in label_hashes.get(env_id, {}).items():
            vectors_for_label = [embedding_by_hash[item] for item in hashes_for_label if item in embedding_by_hash]
            candidate_count = int(label_counts.get(env_id, {}).get(label_key, 0))
            display_label = str((label_display_names.get(env_id) or {}).get(label_key) or label_key)
            pattern_diversity = (pattern_diversity_by_env.get(env_id) or {}).get(label_key)
            if vectors_for_label and candidate_count > 0:
                code_diversity = _mean_pairwise_distance(vectors_for_label)
            else:
                code_diversity = 0.0
            env_rows.append(
                {
                    "label": display_label,
                    "candidate_count": candidate_count,
                    "code_diversity_value": float(code_diversity),
                    "pattern_diversity_value": (
                        float(pattern_diversity) if isinstance(pattern_diversity, (int, float)) else None
                    ),
                    "code_diversity_display": f"{code_diversity:.3f}",
                    "pattern_diversity_display": (
                        f"{float(pattern_diversity):.3f}" if isinstance(pattern_diversity, (int, float)) else "—"
                    ),
                }
            )
            if progress_callback is not None:
                with label_progress_lock:
                    completed += 1
                    progress_callback(
                        completed,
                        max(1, total_labels),
                        f"LABEL:{env_id}/{display_label}",
                    )
        env_rows.sort(key=lambda item: (-int(item.get("candidate_count") or 0), str(item.get("label") or "")))
        return env_id, env_rows

    rows_by_env: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=env_workers) as executor:
        futures = {executor.submit(_build_env_rows, env_id): env_id for env_id in SUPPORTED_ENVS}
        for future in as_completed(futures):
            env_id, env_rows = future.result()
            rows_by_env[env_id] = env_rows
    for env_id in SUPPORTED_ENVS:
        rows_by_env.setdefault(env_id, [])
    return rows_by_env


def compute_statistics_snapshot(
    *,
    project_root: Path,
    start_date: date | None = None,
    sample_seed: int | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    code_by_hash: dict[str, str] = {}
    label_hashes: dict[str, dict[str, list[str]]] = {}
    label_candidate_meta: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    label_counts: dict[str, dict[str, int]] = {}
    label_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    label_display_names: dict[str, dict[str, str]] = {}
    supervised_label_keys_by_env: dict[str, set[str]] = {
        env_id: _load_supervised_label_keys_for_env(project_root=project_root, env_id=env_id)
        for env_id in SUPPORTED_ENVS
    }
    env_worker_raw = os.environ.get("PATTERN_STATS_ENV_WORKERS", "4")
    try:
        env_workers = max(1, min(len(SUPPORTED_ENVS), int(env_worker_raw)))
    except (TypeError, ValueError):
        env_workers = max(1, min(len(SUPPORTED_ENVS), 4))

    def _collect_env_stats(
        env_id: str,
    ) -> tuple[
        str,
        dict[str, list[str]],
        dict[str, dict[str, dict[str, str]]],
        dict[str, int],
        dict[str, str],
        dict[str, dict[str, Any]],
        dict[str, str],
    ]:
        supervised_keys = supervised_label_keys_by_env.get(env_id) or set()
        run_dirs = [
            run_dir
            for run_dir in _iter_env_statistics_run_dirs(project_root=project_root, env_id=env_id)
            if _is_supervised_statistics_run_dir(run_dir)
            and _run_dir_is_on_or_after_start_date(run_dir, start_date=start_date)
        ]
        by_label_hashes: dict[str, list[str]] = defaultdict(list)
        by_label_counts: dict[str, int] = defaultdict(int)
        by_label_valid: dict[str, int] = defaultdict(int)
        by_label_fitness: dict[str, list[float]] = defaultdict(list)
        by_label_code_lengths: dict[str, list[int]] = defaultdict(list)
        by_label_runtime: dict[str, list[float]] = defaultdict(list)
        by_label_display_name: dict[str, str] = {}
        by_label_candidate_meta: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
        local_code_by_hash: dict[str, str] = {}
        for run_dir in run_dirs:
            for label_dir in _unique_label_dirs(run_dir):
                label_name = _label_name_from_dir(label_dir.name)
                label_key = _canonical_label_key(label_name)
                if not label_key:
                    label_key = _canonical_label_key(label_dir.name) or label_dir.name
                if supervised_keys and label_key not in supervised_keys:
                    continue
                by_label_display_name.setdefault(label_key, label_key or label_name)
                for candidate_dir in _label_unique_candidate_dirs(label_dir):
                    payload = _load_json_dict(candidate_dir / "candidate.json")
                    by_label_counts[label_key] += 1
                    fitness = _candidate_fitness(payload)
                    errors = payload.get("errors")
                    has_errors = bool(errors) if isinstance(errors, (list, dict, str)) else False
                    if fitness is not None and not has_errors:
                        by_label_valid[label_key] += 1
                    if fitness is not None:
                        by_label_fitness[label_key].append(float(fitness))
                    code_length = _candidate_code_line_count_optional(candidate_dir)
                    if code_length is not None:
                        by_label_code_lengths[label_key].append(int(code_length))
                    runtime_seconds = _candidate_runtime_seconds(payload)
                    if runtime_seconds is not None:
                        by_label_runtime[label_key].append(float(runtime_seconds))
                    code_text = _candidate_code_text(candidate_dir)
                    if not isinstance(code_text, str) or not code_text.strip():
                        continue
                    code_hash = hashlib.sha1(code_text.encode("utf-8")).hexdigest()
                    by_label_hashes[label_key].append(code_hash)
                    local_code_by_hash.setdefault(code_hash, code_text)
                    input_mode_raw = str(payload.get("input_mode") or "trace").strip().lower()
                    by_label_candidate_meta[label_key].setdefault(
                        code_hash,
                        {
                            "label": str(by_label_display_name.get(label_key) or label_name or label_key),
                            "input_mode": (
                                input_mode_raw if input_mode_raw in {"trace", "supervised_evidence"} else "trace"
                            ),
                        },
                    )
        return (
            env_id,
            dict(by_label_hashes),
            {label_key: dict(meta) for label_key, meta in by_label_candidate_meta.items()},
            {key: int(value) for key, value in by_label_counts.items()},
            dict(by_label_display_name),
            {
                key: {
                    "valid_count": int(by_label_valid.get(key, 0)),
                    "fitness_values": list(by_label_fitness.get(key, [])),
                    "code_lengths": list(by_label_code_lengths.get(key, [])),
                    "runtime_values": list(by_label_runtime.get(key, [])),
                }
                for key in by_label_counts.keys()
            },
            local_code_by_hash,
        )

    future_map = {}
    with ThreadPoolExecutor(max_workers=env_workers) as executor:
        for env_id in SUPPORTED_ENVS:
            future = executor.submit(_collect_env_stats, env_id)
            future_map[future] = env_id
        for future in as_completed(future_map):
            (
                env_id,
                env_label_hashes,
                env_label_candidate_meta,
                env_label_counts,
                env_label_display_names,
                env_label_metrics,
                env_code_by_hash,
            ) = future.result()
            label_hashes[env_id] = env_label_hashes
            label_candidate_meta[env_id] = env_label_candidate_meta
            label_counts[env_id] = env_label_counts
            label_display_names[env_id] = env_label_display_names
            label_metrics[env_id] = env_label_metrics
            for code_hash, code_text in env_code_by_hash.items():
                code_by_hash.setdefault(code_hash, code_text)
        for env_id in SUPPORTED_ENVS:
            label_hashes.setdefault(env_id, {})
            label_candidate_meta.setdefault(env_id, {})
            label_counts.setdefault(env_id, {})
            label_display_names.setdefault(env_id, {})
            label_metrics.setdefault(env_id, {})

    hashes = list(code_by_hash.keys())
    diversity_cap_raw = os.environ.get("PATTERN_STATS_DIVERSITY_MAX_CODES", "5000")
    try:
        diversity_cap = max(1, int(diversity_cap_raw))
    except (TypeError, ValueError):
        diversity_cap = 5000
    sampled_hashes = hashes
    resolved_sample_seed: int
    if isinstance(sample_seed, int):
        resolved_sample_seed = sample_seed
    else:
        sample_seed_raw = os.environ.get("PATTERN_STATS_DIVERSITY_SAMPLE_SEED", "0")
        try:
            resolved_sample_seed = int(sample_seed_raw)
        except (TypeError, ValueError):
            resolved_sample_seed = 0
    if len(sampled_hashes) > diversity_cap:
        sampled_hashes = random.Random(resolved_sample_seed).sample(sorted(sampled_hashes), diversity_cap)
    sampled_hash_set = set(sampled_hashes)
    total_labels = sum(len(items) for items in label_hashes.values())
    embedding_units = len(sampled_hashes)
    if progress_callback is not None:
        progress_callback(
            0,
            max(1, embedding_units),
            f"EMBED:Requesting embeddings for {embedding_units}/{len(hashes)} sampled supervised code snippets",
        )
    vectors = _fetch_openai_embeddings(
        inputs=[code_by_hash[item] for item in sampled_hashes],
        cache_keys=sampled_hashes,
        project_root=project_root,
        batch_size=max(1, int(os.environ.get("PATTERN_STATS_EMBED_BATCH_SIZE", "64"))),
        progress_callback=(
            (
                lambda completed, _total, message: progress_callback(
                    min(embedding_units, int(completed)),
                    max(1, embedding_units),
                    f"EMBED:{message}",
                )
            )
            if progress_callback is not None
            else None
        ),
    )
    if vectors is None or len(vectors) != len(sampled_hashes):
        raise RuntimeError("Unable to compute embeddings for diversity")
    embedding_by_hash = {sampled_hashes[index]: vectors[index] for index in range(len(sampled_hashes))}
    pattern_diversity_by_env = _compute_pattern_diversity_by_env(
        project_root=project_root,
        label_hashes=label_hashes,
        label_candidate_meta=label_candidate_meta,
        code_by_hash=code_by_hash,
        sampled_hashes=sampled_hash_set,
        sample_seed=resolved_sample_seed,
        progress_callback=progress_callback,
        env_workers=env_workers,
    )

    activation_rate_by_env = _compute_activation_rates_by_env(
        project_root=project_root,
        supervised_label_keys_by_env=supervised_label_keys_by_env,
        progress_callback=progress_callback,
    )
    success_corr_by_env = _compute_task_success_correlations_by_env(
        project_root=project_root,
        supervised_label_keys_by_env=supervised_label_keys_by_env,
    )
    ablation_drop_by_env = _compute_ablation_drop_estimates_by_env(
        project_root=project_root,
        supervised_label_keys_by_env=supervised_label_keys_by_env,
        success_corr_by_env=success_corr_by_env,
    )

    completed = 0
    label_progress_lock = threading.Lock()

    def _build_env_rows(env_id: str) -> tuple[str, list[dict[str, Any]]]:
        nonlocal completed
        env_rows: list[dict[str, Any]] = []
        for label_key, hashes_for_label in label_hashes.get(env_id, {}).items():
            vectors_for_label = [embedding_by_hash[item] for item in hashes_for_label if item in embedding_by_hash]
            candidate_count = int(label_counts.get(env_id, {}).get(label_key, 0))
            metrics = (label_metrics.get(env_id) or {}).get(label_key) or {}
            display_label = str((label_display_names.get(env_id) or {}).get(label_key) or label_key)
            valid_count = int(metrics.get("valid_count") or 0)
            fitness_values = [float(item) for item in metrics.get("fitness_values", [])]
            clamped_fitness_values = [max(-1.0, value) for value in fitness_values]
            code_lengths = [int(item) for item in metrics.get("code_lengths", [])]
            runtime_values = [float(item) for item in metrics.get("runtime_values", [])]
            best_fitness = max(fitness_values) if fitness_values else None
            mean_fitness = statistics.fmean(clamped_fitness_values) if clamped_fitness_values else None
            mean_code_length = statistics.fmean(code_lengths) if code_lengths else None
            mean_runtime = statistics.fmean(runtime_values) if runtime_values else None
            executable_pct = ((valid_count / candidate_count) * 100.0) if candidate_count > 0 else None
            activation_rate = (activation_rate_by_env.get(env_id) or {}).get(label_key)
            activation_rate_pct = activation_rate * 100.0 if isinstance(activation_rate, (int, float)) else None
            success_corr = (success_corr_by_env.get(env_id) or {}).get(label_key)
            ablation_drop = (ablation_drop_by_env.get(env_id) or {}).get(label_key)
            pattern_diversity = (pattern_diversity_by_env.get(env_id) or {}).get(label_key)
            if vectors_for_label and candidate_count > 0:
                pairwise = _mean_pairwise_distance(vectors_for_label)
                code_diversity = pairwise
            else:
                code_diversity = 0.0
            env_rows.append(
                {
                    "label": display_label,
                    "candidate_count": candidate_count,
                    "executable_pct_value": executable_pct,
                    "best_fitness_value": best_fitness,
                    "mean_fitness_value": mean_fitness,
                    "ablation_drop_value": float(ablation_drop) if isinstance(ablation_drop, (int, float)) else None,
                    "mean_code_length_value": mean_code_length,
                    "mean_runtime_value": mean_runtime,
                    "activation_rate_value": activation_rate_pct,
                    "success_correlation_value": float(success_corr) if isinstance(success_corr, (int, float)) else None,
                    "code_diversity_value": float(code_diversity),
                    "pattern_diversity_value": (
                        float(pattern_diversity) if isinstance(pattern_diversity, (int, float)) else None
                    ),
                    "executable_pct_display": f"{executable_pct:.1f}%" if executable_pct is not None else "—",
                    "best_fitness_display": f"{best_fitness:.4f}" if best_fitness is not None else "—",
                    "mean_fitness_display": f"{mean_fitness:.4f}" if mean_fitness is not None else "—",
                    "ablation_drop_display": f"{float(ablation_drop):.1f}pp" if isinstance(ablation_drop, (int, float)) else "—",
                    "mean_code_length_display": f"{mean_code_length:.1f}" if mean_code_length is not None else "—",
                    "mean_runtime_display": f"{mean_runtime:.2f}s" if mean_runtime is not None else "—",
                    "activation_rate_display": f"{activation_rate_pct:.1f}%" if activation_rate_pct is not None else "—",
                    "success_correlation_display": f"{float(success_corr):+.3f}" if isinstance(success_corr, (int, float)) else "—",
                    "code_diversity_display": f"{code_diversity:.3f}",
                    "pattern_diversity_display": (
                        f"{float(pattern_diversity):.3f}" if isinstance(pattern_diversity, (int, float)) else "—"
                    ),
                }
            )
            if progress_callback is not None:
                with label_progress_lock:
                    completed += 1
                    progress_callback(
                        completed,
                        max(1, total_labels),
                        f"LABEL:{env_id}/{display_label}",
                    )
        env_rows.sort(key=lambda item: (-int(item.get("candidate_count") or 0), str(item.get("label") or "")))
        return env_id, env_rows

    rows_by_env: dict[str, list[dict[str, Any]]] = {}
    future_map = {}
    with ThreadPoolExecutor(max_workers=env_workers) as executor:
        for env_id in SUPPORTED_ENVS:
            future = executor.submit(_build_env_rows, env_id)
            future_map[future] = env_id
        for future in as_completed(future_map):
            env_id, env_rows = future.result()
            rows_by_env[env_id] = env_rows
    for env_id in SUPPORTED_ENVS:
        rows_by_env.setdefault(env_id, [])
    return rows_by_env


def _render_statistics_table(*, rows: list[dict[str, Any]]) -> str:
    body_rows: list[str] = []
    for row in rows:
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('label') or '—'))}</td>"
            f"<td>{int(row.get('candidate_count') or 0):,}</td>"
            f"<td>{html.escape(str(row.get('executable_pct_display') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('best_fitness_display') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('mean_fitness_display') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('ablation_drop_display') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('mean_code_length_display') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('activation_rate_display') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('success_correlation_display') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('code_diversity_display') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('pattern_diversity_display') or '—'))}</td>"
            "</tr>"
        )
    return f"""
    <table class="data-table">
      <thead>
        <tr>
          <th>Label</th>
          <th># Candidates generated</th>
          <th>% Executable</th>
          <th>Best fitness</th>
          <th>Mean fitness</th>
          <th>Est. ablation drop</th>
          <th>Mean LOC</th>
          <th>Activation rate</th>
          <th>Correlation with task success</th>
          <th>Code diversity</th>
          <th>Pattern diversity</th>
        </tr>
      </thead>
      <tbody>
        {''.join(body_rows) if body_rows else '<tr><td colspan="11">No label/candidate artifacts found yet.</td></tr>'}
      </tbody>
    </table>
    """


def _render_statistics_environment_summary_table(*, rows_by_env: dict[str, list[dict[str, Any]]]) -> str:
    body_rows: list[str] = []
    metric_specs: list[tuple[str, str, str]] = [
        ("candidate_count", "int", "# Candidates generated"),
        ("executable_pct_value", "pct", "% Executable"),
        ("best_fitness_value", "float4", "Best fitness"),
        ("mean_fitness_value", "float4", "Mean fitness"),
        ("ablation_drop_value", "pp", "Est. ablation drop"),
        ("mean_code_length_value", "float1", "Mean LOC"),
        ("activation_rate_value", "pct", "Activation rate"),
        ("success_correlation_value", "signed3", "Correlation with task success"),
        ("code_diversity_value", "float3", "Code diversity"),
        ("pattern_diversity_value", "float3", "Pattern diversity"),
    ]

    def _format_average(kind: str, values: list[float]) -> str:
        if not values:
            return "—"
        avg = statistics.fmean(values)
        if kind == "int":
            return f"{avg:,.1f}"
        if kind == "pct":
            return f"{avg:.1f}%"
        if kind == "pp":
            return f"{avg:.1f}pp"
        if kind == "float1":
            return f"{avg:.1f}"
        if kind == "float3":
            return f"{avg:.3f}"
        if kind == "float4":
            return f"{avg:.4f}"
        if kind == "signed3":
            return f"{avg:+.3f}"
        return f"{avg:.3f}"

    for env_id in SUPPORTED_ENVS:
        env_rows_raw = rows_by_env.get(env_id)
        env_rows = [item for item in env_rows_raw if isinstance(item, dict)] if isinstance(env_rows_raw, list) else []
        cells = [f"<td>{html.escape(env_id.upper())}</td>"]
        for key, kind, _label in metric_specs:
            values: list[float] = []
            for row in env_rows:
                value = row.get(key)
                if isinstance(value, (int, float)):
                    values.append(float(value))
            cells.append(f"<td>{html.escape(_format_average(kind, values))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    header_cells = "".join(f"<th>{html.escape(label)}</th>" for _key, _kind, label in metric_specs)
    return f"""
    <table class="data-table">
      <thead>
        <tr>
          <th>Environment</th>
          {header_cells}
        </tr>
      </thead>
      <tbody>
        {''.join(body_rows)}
      </tbody>
    </table>
    """


def _render_statistics_recalc_panel(*, state: dict[str, Any]) -> str:
    status = str(state.get("status") or "idle").strip().lower()
    message = str(state.get("message") or "").strip()
    updated_at = str(state.get("updated_at") or "").strip()
    start_date = str(state.get("start_date") or "").strip()
    sample_seed = str(state.get("sample_seed") or "0").strip()
    progress_value = max(0.0, min(1.0, float(state.get("progress") or 0.0)))
    completed = max(0, int(state.get("completed") or 0))
    total = max(0, int(state.get("total") or 0))
    log_lines = [str(item) for item in state.get("log_lines", []) if str(item).strip()] if isinstance(state.get("log_lines"), list) else []
    status_chip = _pill(status, status)
    is_running = status not in {"idle", "succeeded", "failed", "cancelled"}
    button_label = "Recalculate now"
    disabled_attr = " disabled" if is_running else ""
    visible_logs = list(reversed(log_lines[-120:])) if log_lines else []
    log_text = "\n".join(visible_logs) if visible_logs else "No log yet."
    progress_percent = int(round(progress_value * 100.0))
    progress_summary = f"{completed}/{total}" if total > 0 else ("running" if is_running else "idle")
    return f"""
    <section class="panel panel-static">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Recalculate</p>
          <h2>Statistics rebuild</h2>
          <p class="metric-subtle">Manual rebuild for all statistics tables, code diversity, and pattern diversity.</p>
        </div>
        <div class="stats-inline">{status_chip}</div>
      </div>
      {(
        '<form method="post" action="/ui/statistics/cancel" class="client-settings-form" style="margin:0 0 0.65rem;">'
        '<input type="hidden" name="next" value="/statistics">'
        '<button class="ghost-link" style="padding:0.62rem 0.95rem;border:1px solid var(--line);border-radius:2px;background:rgba(255,255,255,0.85);font-size:0.96rem;">Cancel</button>'
        '</form>'
        if is_running
        else (
          f'<div style="display:flex; gap:0.65rem; flex-wrap:wrap; align-items:flex-end; margin:0 0 0.65rem;">'
          f'  <form method="post" action="/ui/statistics/recalculate" class="client-settings-form" style="margin:0; display:flex; gap:0.65rem; flex-wrap:wrap; align-items:flex-end;">'
          f'    <input type="hidden" name="next" value="/statistics">'
          f'    <label style="min-width:10.5rem;"><span class="metric-subtle">Start date</span><input type="date" name="start_date" value="{html.escape(start_date)}"></label>'
          f'    <label style="min-width:10.5rem;"><span class="metric-subtle">Sampling seed</span><input type="number" name="sample_seed" step="1" value="{html.escape(sample_seed)}"></label>'
          f'    <button class="secondary-button" style="font-size:0.96rem;padding:0.62rem 0.95rem;" type="submit"{disabled_attr}>{html.escape(button_label)}</button>'
          f'  </form>'
          f'  <form method="post" action="/ui/statistics/recalculate-activation" class="client-settings-form" style="margin:0;">'
          f'    <input type="hidden" name="next" value="/statistics">'
          f'    <button class="ghost-link" type="submit" style="padding:0.62rem 0.85rem;border:1px solid var(--line);border-radius:2px;background:rgba(255,255,255,0.85);font-size:0.96rem;">Activation rates only</button>'
          f'  </form>'
          f'  <form method="post" action="/ui/statistics/recalculate-diversity" class="client-settings-form" style="margin:0;">'
          f'    <input type="hidden" name="next" value="/statistics">'
          f'    <input type="hidden" name="start_date" value="{html.escape(start_date)}">'
          f'    <input type="hidden" name="sample_seed" value="{html.escape(sample_seed)}">'
          f'    <button class="ghost-link" type="submit" style="padding:0.62rem 0.85rem;border:1px solid var(--line);border-radius:2px;background:rgba(255,255,255,0.85);font-size:0.96rem;">Diversity only</button>'
          f'  </form>'
          f'</div>'
        )
      )}
      <div style="display:grid;gap:0.55rem;margin-top:0.35rem;">
        <div style="display:flex;justify-content:space-between;gap:1rem;align-items:center;flex-wrap:wrap;">
          <p id="statistics-message" class="metric-subtle" style="margin:0;">{html.escape(message or 'Idle. Press recalculate to refresh diversity scores.')}</p>
          <p id="statistics-updated" class="metric-subtle" style="margin:0;">{html.escape(updated_at or '')}</p>
        </div>
        <div style="display:flex;justify-content:space-between;gap:1rem;align-items:center;flex-wrap:wrap;">
          <span id="statistics-progress-summary" class="metric-subtle">{html.escape(progress_summary)}</span>
          <span id="statistics-progress-percent" class="metric-subtle">{progress_percent}%</span>
        </div>
        <div style="height:0.7rem;border:1px solid var(--line);background:rgba(255,255,255,0.72);border-radius:999px;overflow:hidden;">
          <div id="statistics-progress-bar" style="height:100%;width:{progress_percent}%;background:linear-gradient(90deg, var(--accent), #111827);transition:width 180ms ease;"></div>
        </div>
      </div>
      <div style="margin-top:0.75rem;">
        <p class="metric-subtle" style="margin:0 0 0.35rem;">Terminal</p>
        <pre id="statistics-terminal" class="code-block" style="max-height:20rem;overflow:auto;margin-top:0;">{html.escape(log_text)}</pre>
      </div>
      <script>
        (() => {{
          const statusRoot = document.currentScript?.closest('section');
          if (!statusRoot) return;
          const statusBox = statusRoot.querySelector('.stats-inline');
          const messageEl = document.getElementById('statistics-message');
          const updatedEl = document.getElementById('statistics-updated');
          const terminalEl = document.getElementById('statistics-terminal');
          const progressSummaryEl = document.getElementById('statistics-progress-summary');
          const progressPercentEl = document.getElementById('statistics-progress-percent');
          const progressBarEl = document.getElementById('statistics-progress-bar');
          const renderChip = (text) => {{
            const value = String(text || 'idle').trim().toLowerCase() || 'idle';
            return `<span class="status-pill">${{value}}</span>`;
          }};
          const updateFromPayload = (payload) => {{
            if (!payload || !payload.statistics) return false;
            const stats = payload.statistics;
            const status = String(stats.status || 'idle').trim().toLowerCase() || 'idle';
            const message = String(stats.message || '').trim() || 'Idle. Press recalculate to refresh diversity scores.';
            const updated = String(stats.updated_at || '').trim();
            const logs = Array.isArray(stats.log_lines) ? stats.log_lines.filter((line) => String(line || '').trim()) : [];
            const visibleLogs = logs.slice(-120).reverse().join('\\n') || 'No log yet.';
            const progress = Math.max(0, Math.min(1, Number(stats.progress || 0)));
            const completed = Math.max(0, Number(stats.completed || 0));
            const total = Math.max(0, Number(stats.total || 0));
            const progressPercent = Math.round(progress * 100);
            const progressSummary = total > 0 ? `${{completed}}/${{total}}` : ((status === 'running' || status === 'cancelling') ? 'running' : 'idle');
            if (statusBox) statusBox.innerHTML = renderChip(status);
            if (messageEl) messageEl.textContent = message;
            if (updatedEl) updatedEl.textContent = updated;
            if (terminalEl) terminalEl.textContent = visibleLogs;
            if (progressSummaryEl) progressSummaryEl.textContent = progressSummary;
            if (progressPercentEl) progressPercentEl.textContent = `${{progressPercent}}%`;
            if (progressBarEl) progressBarEl.style.width = `${{progressPercent}}%`;
            return status === 'running' || status === 'cancelling';
          }};
          const poll = () => fetch('/api/ui/statistics/status', {{ cache: 'no-store' }})
            .then((r) => r.ok ? r.json() : null)
            .then((payload) => {{
              const shouldContinue = updateFromPayload(payload);
              if (shouldContinue) window.setTimeout(poll, 1500);
            }})
            .catch(() => window.setTimeout(poll, 2500));
          if ({str(is_running).lower()}) {{
            window.setTimeout(poll, 1200);
          }}
        }})();
      </script>
    </section>
    """


def render_statistics_page(
    *,
    project_root: Path,
    rows_by_env: dict[str, list[dict[str, Any]]] | None = None,
    diversity_state: dict[str, Any] | None = None,
) -> str:
    sections: list[str] = []
    normalized_rows = rows_by_env or {}
    combined_summary = f"""
    <section class="panel">
      <div class="card-topline">
        <div>
          <h2>Environment Summary</h2>
          <p class="metric-subtle">Each row is the simple average of the per-label statistics shown in the environment tables below.</p>
        </div>
      </div>
      {_render_statistics_environment_summary_table(rows_by_env=normalized_rows)}
    </section>
    """
    for env_id in SUPPORTED_ENVS:
        rows_raw = normalized_rows.get(env_id)
        rows = [item for item in rows_raw if isinstance(item, dict)] if isinstance(rows_raw, list) else []
        sections.append(
            f"""
            <section class="panel">
              <div class="card-topline">
                <div>
                  <h2>{html.escape(env_id.upper())}</h2>
                  <p class="metric-subtle">Rebuilt from run artifacts when you press recalculate. Estimated ablation drop uses the latest supervised ablation data available for this environment.</p>
                </div>
              </div>
              {_render_statistics_table(rows=rows)}
            </section>
            """
        )
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Home</a> / statistics</p>
      <h1>Label Statistics</h1>
      <p>Manual statistics rebuild with cached results. Page loads remain lightweight between recalculations.</p>
    </section>
    {_render_statistics_recalc_panel(state=diversity_state or {})}
    {combined_summary}
    {''.join(sections)}
    """
    return _shell("Statistics | Pattern Server", active_nav="statistics", body_html=body)


def _render_profiler_table(*, results: dict[str, Any]) -> str:
    def _format_bytes(value: float) -> str:
        numeric = max(0.0, float(value))
        if numeric >= 1_000_000:
            return f"{numeric / 1_000_000:.2f} MB"
        if numeric >= 1_000:
            return f"{numeric / 1_000:.1f} KB"
        return f"{int(round(numeric))} B"

    def _cell(env_id: str, variant: str) -> str:
        env_payload = results.get(env_id) if isinstance(results, dict) else {}
        variant_payload = env_payload.get(variant) if isinstance(env_payload, dict) else {}
        avg_time = float(variant_payload.get("avg_time_seconds") or 0.0) if isinstance(variant_payload, dict) else 0.0
        avg_image_bytes = float(variant_payload.get("avg_image_bytes") or 0.0) if isinstance(variant_payload, dict) else 0.0
        avg_input_words = float(variant_payload.get("avg_input_words") or 0.0) if isinstance(variant_payload, dict) else 0.0
        avg_output_words = float(variant_payload.get("avg_output_words") or 0.0) if isinstance(variant_payload, dict) else 0.0
        count = int(variant_payload.get("observation_count") or 0) if isinstance(variant_payload, dict) else 0
        if count <= 0:
            return '<td><span class="metric-subtle">—</span></td>'
        return (
            "<td>"
            f"<strong>{avg_time:.2f}s</strong><br>"
            f"<span class=\"metric-subtle\">{_format_bytes(avg_image_bytes)} img · {avg_input_words:,.0f} in · {avg_output_words:,.0f} out · n={count}</span>"
            "</td>"
        )

    rows: list[str] = []
    labels = {
        "image_only": "Image",
        "supervised": "Supervised",
        "unsupervised": "Unsupervised",
    }
    for variant in ("image_only", "supervised", "unsupervised"):
        cells = [f"<td>{html.escape(labels[variant])}</td>"]
        for env_id in SUPPORTED_ENVS:
            cells.append(_cell(env_id, variant))
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
    <table class="data-table">
      <thead>
        <tr>
          <th>Variant</th>
          <th>PHYRE</th>
          <th>iPHYRE</th>
          <th>Kinetix</th>
          <th>PoolTool</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def _render_profiler_panel(*, state: dict[str, Any]) -> str:
    status = str(state.get("status") or "idle").strip().lower()
    message = str(state.get("message") or "").strip()
    updated_at = str(state.get("updated_at") or "").strip()
    progress_value = max(0.0, min(1.0, float(state.get("progress") or 0.0)))
    completed = max(0, int(state.get("completed") or 0))
    total = max(0, int(state.get("total") or 0))
    run_count = max(0, int(state.get("run_count") or 0))
    log_lines = [str(item) for item in state.get("log_lines", []) if str(item).strip()] if isinstance(state.get("log_lines"), list) else []
    status_chip = _pill(status, status)
    is_running = status not in {"idle", "succeeded", "failed", "cancelled"}
    disabled_attr = " disabled" if is_running else ""
    cancel_disabled_attr = "" if is_running else " disabled"
    visible_logs = list(reversed(log_lines[-120:])) if log_lines else []
    log_text = "\n".join(visible_logs) if visible_logs else "No log yet."
    progress_percent = int(round(progress_value * 100.0))
    progress_summary = f"{completed}/{total}" if total > 0 else ("running" if is_running else "idle")
    results = state.get("results") if isinstance(state.get("results"), dict) else {}
    return f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Home</a> / profiler</p>
      <h1>Profiler</h1>
      <p>Runs 10 sampled simulations from each environment through image-only, supervised, and unsupervised summaries, then records elapsed time, image bytes, and text word counts.</p>
    </section>
    <section class="panel panel-static">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Run</p>
          <h2>Summary profiler</h2>
          <p class="metric-subtle">Each run is a fresh 10-task-per-environment snapshot using the same summary prompts and annotation flow as <code>/summaries</code>.</p>
        </div>
        <div class="stats-inline">
          {status_chip}
          <span class="mini-chip">Runs {run_count}</span>
        </div>
      </div>
      <div style="display:flex;gap:0.6rem;align-items:center;flex-wrap:wrap;margin:0 0 0.75rem;">
        <form method="post" action="/ui/profiler/run" class="client-settings-form" style="margin:0;">
          <input type="hidden" name="next" value="/profiler">
          <button class="secondary-button" type="submit"{disabled_attr}>Run profiler</button>
        </form>
        <form method="post" action="/ui/profiler/cancel" class="client-settings-form" style="margin:0;">
          <input type="hidden" name="next" value="/profiler">
          <button class="secondary-button" type="submit"{cancel_disabled_attr}>Cancel</button>
        </form>
      </div>
      <div style="display:grid;gap:0.55rem;">
        <div style="display:flex;justify-content:space-between;gap:1rem;align-items:center;flex-wrap:wrap;">
          <p id="profiler-message" class="metric-subtle" style="margin:0;">{html.escape(message or 'Idle. Press Run profiler to sample 10 tasks per environment.')}</p>
          <p id="profiler-updated" class="metric-subtle" style="margin:0;">{html.escape(updated_at or '')}</p>
        </div>
        <div style="display:flex;justify-content:space-between;gap:1rem;align-items:center;flex-wrap:wrap;">
          <span id="profiler-progress-summary" class="metric-subtle">{html.escape(progress_summary)}</span>
          <span id="profiler-progress-percent" class="metric-subtle">{progress_percent}%</span>
        </div>
        <div style="height:0.7rem;border:1px solid var(--line);background:rgba(255,255,255,0.72);border-radius:999px;overflow:hidden;">
          <div id="profiler-progress-bar" style="height:100%;width:{progress_percent}%;background:linear-gradient(90deg, var(--accent), #111827);transition:width 180ms ease;"></div>
        </div>
      </div>
      <div style="margin-top:0.9rem;">
        <h3 style="margin:0 0 0.5rem;">Latest run averages</h3>
        {_render_profiler_table(results=results)}
        <p class="metric-subtle" style="margin:0.55rem 0 0;">Each cell shows average time, average bytes of images sent, average input words, average output words, and completed sample count.</p>
      </div>
      <div style="margin-top:0.75rem;">
        <p class="metric-subtle" style="margin:0 0 0.35rem;">Terminal</p>
        <pre id="profiler-terminal" class="code-block" style="max-height:20rem;overflow:auto;margin-top:0;">{html.escape(log_text)}</pre>
      </div>
      <script>
        (() => {{
          const root = document.currentScript?.closest('section');
          if (!root) return;
          const statusBox = root.querySelector('.stats-inline');
          const messageEl = document.getElementById('profiler-message');
          const updatedEl = document.getElementById('profiler-updated');
          const terminalEl = document.getElementById('profiler-terminal');
          const progressSummaryEl = document.getElementById('profiler-progress-summary');
          const progressPercentEl = document.getElementById('profiler-progress-percent');
          const progressBarEl = document.getElementById('profiler-progress-bar');
          const renderChip = (text) => {{
            const value = String(text || 'idle').trim().toLowerCase() || 'idle';
            return `<span class="status-pill">${{value}}</span><span class="mini-chip">Runs ${{Number(window.__profilerRuns || 0)}}</span>`;
          }};
          const renderCell = (envId, variant, payload) => {{
            const envPayload = payload && payload.results && payload.results[envId] ? payload.results[envId] : {{}};
            const variantPayload = envPayload && envPayload[variant] ? envPayload[variant] : {{}};
            const count = Number(variantPayload.observation_count || 0);
            if (count <= 0) return '<span class="metric-subtle">—</span>';
            const avgTime = Number(variantPayload.avg_time_seconds || 0).toFixed(2);
            const avgImageBytes = Number(variantPayload.avg_image_bytes || 0);
            const avgInputWords = Math.round(Number(variantPayload.avg_input_words || 0)).toLocaleString();
            const avgOutputWords = Math.round(Number(variantPayload.avg_output_words || 0)).toLocaleString();
            let imageLabel = `${{Math.round(avgImageBytes)}} B`;
            if (avgImageBytes >= 1000000) {{
              imageLabel = `${{(avgImageBytes / 1000000).toFixed(2)}} MB`;
            }} else if (avgImageBytes >= 1000) {{
              imageLabel = `${{(avgImageBytes / 1000).toFixed(1)}} KB`;
            }}
            return `<strong>${{avgTime}}s</strong><br><span class="metric-subtle">${{imageLabel}} img · ${{avgInputWords}} in · ${{avgOutputWords}} out · n=${{count}}</span>`;
          }};
          const updateTable = (payload) => {{
            const table = root.querySelector('table.data-table tbody');
            if (!table) return;
            const variants = [
              ['image_only', 'Image'],
              ['supervised', 'Supervised'],
              ['unsupervised', 'Unsupervised'],
            ];
            const envs = ['phyre', 'iphyre', 'kinetix', 'pooltool'];
            table.innerHTML = variants.map(([variant, label]) => {{
              const cells = [`<td>${{label}}</td>`];
              envs.forEach((envId) => cells.push(`<td>${{renderCell(envId, variant, payload)}}</td>`));
              return `<tr>${{cells.join('')}}</tr>`;
            }}).join('');
          }};
          const updateFromPayload = (payload) => {{
            if (!payload) return false;
            const status = String(payload.status || 'idle').trim().toLowerCase() || 'idle';
            const message = String(payload.message || '').trim() || 'Idle. Press Run profiler to sample 10 tasks per environment.';
            const updated = String(payload.updated_at || '').trim();
            const logs = Array.isArray(payload.log_lines) ? payload.log_lines.filter((line) => String(line || '').trim()) : [];
            const visibleLogs = logs.slice(-120).reverse().join('\\n') || 'No log yet.';
            const progress = Math.max(0, Math.min(1, Number(payload.progress || 0)));
            const completed = Math.max(0, Number(payload.completed || 0));
            const total = Math.max(0, Number(payload.total || 0));
            const progressPercent = Math.round(progress * 100);
            const progressSummary = total > 0 ? `${{completed}}/${{total}}` : ((status === 'running') ? 'running' : 'idle');
            window.__profilerRuns = Number(payload.run_count || 0);
            if (statusBox) statusBox.innerHTML = renderChip(status);
            if (messageEl) messageEl.textContent = message;
            if (updatedEl) updatedEl.textContent = updated;
            if (terminalEl) terminalEl.textContent = visibleLogs;
            if (progressSummaryEl) progressSummaryEl.textContent = progressSummary;
            if (progressPercentEl) progressPercentEl.textContent = `${{progressPercent}}%`;
            if (progressBarEl) progressBarEl.style.width = `${{progressPercent}}%`;
            updateTable(payload);
            return status === 'running';
          }};
          const poll = () => fetch('/api/ui/profiler/status', {{ cache: 'no-store' }})
            .then((r) => r.ok ? r.json() : null)
            .then((payload) => {{
              const shouldContinue = updateFromPayload(payload);
              if (shouldContinue) window.setTimeout(poll, 1500);
            }})
            .catch(() => window.setTimeout(poll, 2500));
          if ({str(is_running).lower()}) {{
            window.setTimeout(poll, 1200);
          }}
        }})();
      </script>
    </section>
    """


def render_profiler_page(*, state: dict[str, Any]) -> str:
    return _shell("Profiler | Pattern Server", active_nav="profiler", body_html=_render_profiler_panel(state=state))


def _label_identity_key(label_dir: Path) -> tuple[int | None, str]:
    match = _LABEL_DIR_PATTERN.match(label_dir.name)
    if match is None:
        return (None, label_dir.name)
    try:
        label_index = int(match.group("index"))
    except (TypeError, ValueError):
        label_index = None
    return (label_index, str(match.group("label")))


def _unique_label_dirs(run_dir: Path) -> list[Path]:
    unique_dirs: dict[tuple[int | None, str], Path] = {}
    for label_dir in run_dir.glob("label_*"):
        if not label_dir.is_dir():
            continue
        key = _label_identity_key(label_dir)
        existing = unique_dirs.get(key)
        if existing is None:
            unique_dirs[key] = label_dir
            continue
        existing_complete = (existing / "label_result.json").is_file()
        candidate_complete = (label_dir / "label_result.json").is_file()
        if candidate_complete and not existing_complete:
            unique_dirs[key] = label_dir
            continue
        if existing_complete and not candidate_complete:
            continue
        try:
            candidate_mtime_ns = label_dir.stat().st_mtime_ns
        except OSError:
            candidate_mtime_ns = 0
        try:
            existing_mtime_ns = existing.stat().st_mtime_ns
        except OSError:
            existing_mtime_ns = 0
        if candidate_mtime_ns >= existing_mtime_ns:
            unique_dirs[key] = label_dir
    return sorted(unique_dirs.values(), key=_label_dir_sort_key)


def _learning_run_aggregate_marker(run_dir: Path | None) -> tuple[int, int, int, int, int]:
    if run_dir is None:
        return (0, 0, 0, 0, 0)
    run_json_mtime_ns, run_dir_mtime_ns = _run_dir_cache_marker(run_dir)
    label_dir_mtime_ns = 0
    candidate_dir_mtime_ns = 0
    candidate_dir_count = 0
    for label_dir in _unique_label_dirs(run_dir):
        try:
            label_dir_mtime_ns = max(label_dir_mtime_ns, label_dir.stat().st_mtime_ns)
        except OSError:
            continue
        for candidate_dir in label_dir.glob("candidate_*"):
            if not candidate_dir.is_dir():
                continue
            candidate_dir_count += 1
            try:
                candidate_dir_mtime_ns = max(candidate_dir_mtime_ns, candidate_dir.stat().st_mtime_ns)
            except OSError:
                continue
    return (
        run_json_mtime_ns,
        run_dir_mtime_ns,
        label_dir_mtime_ns,
        candidate_dir_mtime_ns,
        candidate_dir_count,
    )


@lru_cache(maxsize=256)
def _cached_learning_run_aggregate_stats(
    run_dir_value: str,
    marker: tuple[int, int, int, int, int],
) -> tuple[int, int, int, int]:
    del marker
    if not run_dir_value:
        return (0, 0, 0, 0)
    run_dir = Path(run_dir_value)
    if not run_dir.is_dir():
        return (0, 0, 0, 0)
    label_dirs = _unique_label_dirs(run_dir)
    label_count_from_dirs = len(label_dirs)
    configured_label_count = 0
    run_json_path = run_dir / "run.json"
    if run_json_path.is_file():
        try:
            run_payload = json.loads(run_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            run_payload = {}
        if isinstance(run_payload, dict):
            try:
                configured_label_count = max(0, int(run_payload.get("primitive_label_count") or 0))
            except (TypeError, ValueError):
                configured_label_count = 0
    label_count = max(label_count_from_dirs, configured_label_count)
    completed_label_count = 0
    candidate_count = 0
    code_line_count = 0
    for label_dir in label_dirs:
        if (label_dir / "label_result.json").is_file():
            completed_label_count += 1
        unique_candidate_dirs = _label_unique_candidate_dirs(label_dir)
        candidate_count += len(unique_candidate_dirs)
        for candidate_dir in unique_candidate_dirs:
            code_line_count += _candidate_code_line_count(candidate_dir)
    return (candidate_count, label_count, completed_label_count, code_line_count)


def _candidate_code_line_count(candidate_dir: Path) -> int:
    code_path = candidate_dir / "find_pattern.py"
    if code_path.is_file():
        try:
            with code_path.open(encoding="utf-8") as handle:
                return sum(1 for _ in handle)
        except OSError:
            return 0
    candidate_payload_path = candidate_dir / "candidate.json"
    if candidate_payload_path.is_file():
        try:
            payload = json.loads(candidate_payload_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            for key in ("code", "candidate_code", "detector_code", "source_code", "program"):
                raw_code = payload.get(key)
                if isinstance(raw_code, str) and raw_code.strip():
                    return len(raw_code.splitlines())
    return 0


def _label_unique_candidate_dirs(label_dir: Path) -> list[Path]:
    # Legacy runs can contain many candidate_* directories for the same logical
    # candidate index; dedupe so UI counts match per-label evaluation progress.
    unique_dirs: dict[tuple[str, int | str], Path] = {}
    for candidate_dir in label_dir.glob("candidate_*"):
        if not candidate_dir.is_dir():
            continue
        key = _candidate_identity_key(candidate_dir)
        existing = unique_dirs.get(key)
        if existing is None:
            unique_dirs[key] = candidate_dir
            continue
        try:
            candidate_mtime_ns = candidate_dir.stat().st_mtime_ns
        except OSError:
            candidate_mtime_ns = 0
        try:
            existing_mtime_ns = existing.stat().st_mtime_ns
        except OSError:
            existing_mtime_ns = 0
        if candidate_mtime_ns >= existing_mtime_ns:
            unique_dirs[key] = candidate_dir
    return list(unique_dirs.values())


def _candidate_identity_key(candidate_dir: Path) -> tuple[str, int | str]:
    payload = {}
    payload_path = candidate_dir / "candidate.json"
    if payload_path.is_file():
        try:
            parsed = json.loads(payload_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            parsed = {}
        if isinstance(parsed, dict):
            payload = parsed
    candidate_index = payload.get("candidate_index")
    try:
        normalized_index = int(candidate_index)
    except (TypeError, ValueError):
        normalized_index = None
    if normalized_index is None or normalized_index <= 0:
        match = _CANDIDATE_DIR_PATTERN.match(candidate_dir.name)
        if match is not None:
            try:
                normalized_index = int(match.group("index"))
            except (TypeError, ValueError):
                normalized_index = None
    if normalized_index is not None and normalized_index > 0:
        return ("index", normalized_index)
    uid = str(payload.get("candidate_uid") or payload.get("uid") or "").strip()
    if uid:
        return ("uid", uid)
    return ("dir", candidate_dir.name)


def _learning_candidate_function_count_marker(run_dir: Path | None) -> tuple[int, int, int, int]:
    if run_dir is None:
        return (0, 0, 0, 0)
    run_json_mtime_ns, run_dir_mtime_ns = _run_dir_cache_marker(run_dir)
    evolution_mtime_ns = 0
    label_dir_mtime_ns = 0
    for candidate in (run_dir / "evolution_record.jsonl", run_dir / "evolution_live.jsonl"):
        try:
            if candidate.is_file():
                evolution_mtime_ns = max(evolution_mtime_ns, candidate.stat().st_mtime_ns)
        except OSError:
            continue
    for label_dir in run_dir.glob("label_*"):
        if not label_dir.is_dir():
            continue
        try:
            label_dir_mtime_ns = max(label_dir_mtime_ns, label_dir.stat().st_mtime_ns)
        except OSError:
            continue
    return (run_json_mtime_ns, run_dir_mtime_ns, evolution_mtime_ns, label_dir_mtime_ns)


@lru_cache(maxsize=256)
def _cached_learning_candidate_function_count(
    run_dir_value: str,
    marker: tuple[int, int, int, int],
) -> int | None:
    del marker
    if not run_dir_value:
        return None
    run_dir = Path(run_dir_value)
    if not run_dir.is_dir():
        return None
    total = 0
    for label_dir in _unique_label_dirs(run_dir):
        total += len(_label_unique_candidate_dirs(label_dir))
    return total


def _recorded_learning_candidate_count(run_dir: Path) -> int | None:
    candidate_uids: set[str] = set()
    found_any = False
    for path in (run_dir / "evolution_record.jsonl", run_dir / "evolution_live.jsonl"):
        try:
            if not path.is_file():
                continue
            found_any = True
            with path.open(encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    uid = str(payload.get("candidate_uid") or "").strip()
                    if uid:
                        candidate_uids.add(uid)
        except OSError:
            continue
    if found_any:
        return len(candidate_uids)
    return None


def _run_dir_cache_marker(run_dir: Path | None) -> tuple[int, int]:
    if run_dir is None:
        return (0, 0)
    run_json = run_dir / "run.json"
    try:
        run_mtime_ns = run_json.stat().st_mtime_ns if run_json.is_file() else 0
    except OSError:
        run_mtime_ns = 0
    try:
        dir_mtime_ns = run_dir.stat().st_mtime_ns
    except OSError:
        dir_mtime_ns = 0
    return (run_mtime_ns, dir_mtime_ns)


@lru_cache(maxsize=256)
def _cached_learning_progress_summary(
    run_dir_value: str,
    marker: tuple[int, int],
    lines: tuple[str, ...],
    status: str,
    fallback_progress_json: str,
) -> dict[str, Any] | None:
    del marker
    run_dir = Path(run_dir_value) if run_dir_value else None
    fallback_progress = (
        json.loads(fallback_progress_json) if fallback_progress_json else None
    )
    return summarize_learning_progress(
        run_dir=run_dir,
        lines=list(lines),
        status=status,
        fallback_progress=fallback_progress,
    )


def _format_seconds_compact(value: float | None) -> str:
    if value is None:
        return "—"
    total_seconds = max(0, int(round(float(value))))
    days, total_seconds = divmod(total_seconds, 86400)
    hours, total_seconds = divmod(total_seconds, 3600)
    minutes, seconds = divmod(total_seconds, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(f"{seconds}s")
    return " ".join(parts[:3])


def _learning_timing_marker(run_dir: Path | None) -> tuple[int, int, int]:
    if run_dir is None:
        return (0, 0, 0)
    run_json_mtime_ns, run_dir_mtime_ns = _run_dir_cache_marker(run_dir)
    llm_calls_mtime_ns = 0
    llm_calls_path = run_dir / "llm_calls.jsonl"
    try:
        if llm_calls_path.is_file():
            llm_calls_mtime_ns = llm_calls_path.stat().st_mtime_ns
    except OSError:
        llm_calls_mtime_ns = 0
    return (run_json_mtime_ns, run_dir_mtime_ns, llm_calls_mtime_ns)


@lru_cache(maxsize=256)
def _cached_learning_timing_breakdown(
    run_dir_value: str,
    marker: tuple[int, int, int],
) -> dict[str, Any] | None:
    del marker
    if not run_dir_value:
        return None
    run_dir = Path(run_dir_value)
    run_payload = _load_json_dict(run_dir / "run.json")
    raw_summary = run_payload.get("timing_summary")
    if not isinstance(raw_summary, dict):
        return None
    groups_payload = raw_summary.get("groups")
    if not isinstance(groups_payload, dict):
        return None
    total_wall_seconds = _safe_float(raw_summary.get("total_wall_seconds"))
    if total_wall_seconds is None:
        total_wall_seconds = 0.0
    aggregated_from_labels = False

    if not any(isinstance(groups_payload.get(group_name), list) and groups_payload.get(group_name) for group_name in ("search", "evaluation")):
        aggregate_groups: dict[str, dict[str, dict[str, Any]]] = {"search": {}, "evaluation": {}}
        aggregate_wall_seconds = 0.0
        for label_result_path in run_dir.glob("label_*/label_result.json"):
            label_payload = _load_json_dict(label_result_path)
            label_timing = label_payload.get("timing_summary")
            if not isinstance(label_timing, dict):
                continue
            label_groups = label_timing.get("groups")
            if not isinstance(label_groups, dict):
                continue
            label_wall_seconds = _safe_float(label_timing.get("total_wall_seconds"), fallback=0.0) or 0.0
            aggregate_wall_seconds += label_wall_seconds
            for group_name in ("search", "evaluation"):
                label_entries = label_groups.get(group_name)
                if not isinstance(label_entries, list):
                    continue
                for raw_entry in label_entries:
                    if not isinstance(raw_entry, dict):
                        continue
                    key = str(raw_entry.get("key") or "").strip()
                    label = str(raw_entry.get("label") or key).strip()
                    if not key or not label:
                        continue
                    bucket = aggregate_groups[group_name].setdefault(
                        key,
                        {
                            "key": key,
                            "label": label,
                            "count": 0,
                            "total_seconds": 0.0,
                            "unit": str(raw_entry.get("unit") or "").strip() or "item",
                        },
                    )
                    bucket["count"] += _safe_int(raw_entry.get("count")) or 0
                    bucket["total_seconds"] += _safe_float(raw_entry.get("total_seconds"), fallback=0.0) or 0.0
        if aggregate_wall_seconds > 0:
            groups_payload = {
                group_name: [
                    {
                        **entry,
                        "average_seconds": (
                            entry["total_seconds"] / entry["count"]
                            if int(entry["count"]) > 0
                            else None
                        ),
                        "share_of_wall_time": (
                            entry["total_seconds"] / aggregate_wall_seconds
                            if aggregate_wall_seconds > 0
                            else None
                        ),
                    }
                    for entry in aggregate_groups[group_name].values()
                ]
                for group_name in ("search", "evaluation")
            }
            if total_wall_seconds <= 0:
                total_wall_seconds = aggregate_wall_seconds
            aggregated_from_labels = True

    groups: list[dict[str, Any]] = []
    search_total_seconds = 0.0
    evaluation_total_seconds = 0.0
    code_writing_total_seconds: float | None = None

    for group_name in ("search", "evaluation"):
        raw_entries = groups_payload.get(group_name)
        if not isinstance(raw_entries, list):
            continue
        rows: list[dict[str, Any]] = []
        group_total_seconds = 0.0
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                continue
            total_seconds = _safe_float(raw_entry.get("total_seconds"), fallback=0.0) or 0.0
            average_seconds = _safe_float(raw_entry.get("average_seconds"))
            count = _safe_int(raw_entry.get("count"))
            share = _safe_float(raw_entry.get("share_of_wall_time"))
            label = str(raw_entry.get("label") or raw_entry.get("key") or "").strip()
            if not label:
                continue
            unit = str(raw_entry.get("unit") or "").strip() or "item"
            key = str(raw_entry.get("key") or "").strip()
            rows.append(
                {
                    "key": key,
                    "label": label,
                    "total_seconds": total_seconds,
                    "average_seconds": average_seconds,
                    "count": count,
                    "share_of_wall_time": share,
                    "unit": unit,
                }
            )
            group_total_seconds += total_seconds
            if key == "code_writing":
                code_writing_total_seconds = total_seconds
        rows.sort(key=lambda item: (-float(item.get("total_seconds") or 0.0), str(item.get("label") or "")))
        if rows:
            groups.append(
                {
                    "name": group_name,
                    "label": ("Search" if group_name == "search" else "Evaluation"),
                    "total_seconds": group_total_seconds,
                    "share_of_wall_time": (
                        group_total_seconds / total_wall_seconds
                        if isinstance(total_wall_seconds, (int, float)) and total_wall_seconds > 0
                        else None
                    ),
                    "rows": rows,
                }
            )
        if group_name == "search":
            search_total_seconds = group_total_seconds
        elif group_name == "evaluation":
            evaluation_total_seconds = group_total_seconds

    llm_call_count = 0
    llm_total_seconds = 0.0
    llm_calls_path = run_dir / "llm_calls.jsonl"
    if llm_calls_path.is_file():
        try:
            with llm_calls_path.open(encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    duration_seconds = _safe_float(payload.get("duration_seconds"))
                    if duration_seconds is None:
                        continue
                    llm_total_seconds += duration_seconds
                    llm_call_count += 1
        except OSError:
            llm_call_count = 0
            llm_total_seconds = 0.0

    observed_llm: dict[str, Any] | None = None
    if llm_call_count > 0:
        overhead_seconds = None
        share_of_writing_time = None
        if isinstance(code_writing_total_seconds, (int, float)):
            overhead_seconds = max(0.0, float(code_writing_total_seconds) - llm_total_seconds)
            if float(code_writing_total_seconds) > 0 and llm_total_seconds <= float(code_writing_total_seconds) * 1.02:
                share_of_writing_time = llm_total_seconds / float(code_writing_total_seconds)
            else:
                overhead_seconds = None
        observed_llm = {
            "call_count": llm_call_count,
            "total_seconds": llm_total_seconds,
            "average_seconds": (llm_total_seconds / llm_call_count) if llm_call_count > 0 else None,
            "share_of_wall_time": (
                llm_total_seconds / total_wall_seconds
                if isinstance(total_wall_seconds, (int, float)) and total_wall_seconds > 0
                else None
            ),
            "share_of_writing_time": share_of_writing_time,
            "non_llm_writing_seconds": overhead_seconds,
        }

    return {
        "total_wall_seconds": total_wall_seconds,
        "search_total_seconds": search_total_seconds,
        "evaluation_total_seconds": evaluation_total_seconds,
        "groups": groups,
        "observed_llm": observed_llm,
        "aggregated_from_labels": aggregated_from_labels,
    }


def _learning_timing_breakdown(job: JobRecord, *, project_root: Path) -> dict[str, Any] | None:
    if not _is_learning_job(job):
        return None
    run_dir = _resolve_job_run_dir(job, project_root=project_root)
    if run_dir is None or not run_dir.is_dir():
        return None
    return _cached_learning_timing_breakdown(
        str(run_dir),
        _learning_timing_marker(run_dir),
    )


def _render_learning_timing_breakdown(job: JobRecord, *, project_root: Path) -> str:
    breakdown = _learning_timing_breakdown(job, project_root=project_root)
    if not isinstance(breakdown, dict):
        return ""
    total_wall_seconds = _safe_float(breakdown.get("total_wall_seconds"))
    search_total_seconds = _safe_float(breakdown.get("search_total_seconds"), fallback=0.0) or 0.0
    evaluation_total_seconds = _safe_float(breakdown.get("evaluation_total_seconds"), fallback=0.0) or 0.0
    observed_llm = breakdown.get("observed_llm") if isinstance(breakdown.get("observed_llm"), dict) else None
    groups = breakdown.get("groups") if isinstance(breakdown.get("groups"), list) else []
    aggregated_from_labels = bool(breakdown.get("aggregated_from_labels"))

    summary_cards = [
        ("Wall time", _format_seconds_compact(total_wall_seconds), "Whole run"),
        (
            "Search",
            _format_seconds_compact(search_total_seconds),
            (
                f"{(search_total_seconds / total_wall_seconds) * 100:.0f}% of wall time"
                if isinstance(total_wall_seconds, (int, float)) and total_wall_seconds > 0
                else "Detector generation"
            ),
        ),
        (
            "Evaluation",
            _format_seconds_compact(evaluation_total_seconds),
            (
                f"{(evaluation_total_seconds / total_wall_seconds) * 100:.0f}% of wall time"
                if isinstance(total_wall_seconds, (int, float)) and total_wall_seconds > 0
                else "Detector scoring"
            ),
        ),
    ]
    if observed_llm is not None:
        summary_cards.append(
            (
                "LLM request time",
                _format_seconds_compact(_safe_float(observed_llm.get("total_seconds"))),
                (
                    f"{_safe_int(observed_llm.get('call_count')) or 0} calls"
                    + (
                        f" · {(float(observed_llm['share_of_writing_time']) * 100):.0f}% of writing"
                        if isinstance(observed_llm.get("share_of_writing_time"), (int, float))
                        else ""
                    )
                ),
            )
        )
    summary_cards_html = "".join(
        f'<article class="job-overview-card"><span class="job-overview-kicker">{html.escape(label)}</span><strong class="job-overview-value">{html.escape(value)}</strong><span class="job-overview-meta">{html.escape(meta)}</span></article>'
        for label, value, meta in summary_cards
    )

    group_sections: list[str] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        rows = group.get("rows")
        if not isinstance(rows, list) or not rows:
            continue
        group_label = str(group.get("label") or group.get("name") or "Timing").strip()
        group_total = _format_seconds_compact(_safe_float(group.get("total_seconds")))
        group_share = _safe_float(group.get("share_of_wall_time"))
        rows_html = "".join(
            f"""
            <tr>
              <td>{html.escape(str(row.get("label") or "—"))}</td>
              <td>{html.escape(_format_seconds_compact(_safe_float(row.get("total_seconds"))))}</td>
              <td>{html.escape(_format_seconds_compact(_safe_float(row.get("average_seconds"))))}</td>
              <td>{html.escape(str(_safe_int(row.get("count")) or 0))} {html.escape(str(row.get("unit") or "item"))}</td>
              <td>{html.escape(f"{float(row['share_of_wall_time']) * 100:.1f}%") if isinstance(row.get("share_of_wall_time"), (int, float)) else "—"}</td>
            </tr>
            """
            for row in rows
            if isinstance(row, dict)
        )
        group_sections.append(
            f"""
            <section class="panel compact-panel">
              <div class="card-topline">
                <div>
                  <h3>{html.escape(group_label)}</h3>
                  <p class="metric-subtle">{html.escape(group_total)} total{f" · {group_share * 100:.0f}% of wall time" if isinstance(group_share, (int, float)) else ""}</p>
                </div>
              </div>
              <div class="scroll-table compact-scroll-table">
                <table class="data-table compact-table">
                  <thead><tr><th>Stage</th><th>Total</th><th>Average</th><th>Count</th><th>Wall</th></tr></thead>
                  <tbody>{rows_html}</tbody>
                </table>
              </div>
            </section>
            """
        )

    llm_panel_html = ""
    if observed_llm is not None:
        llm_total_seconds = _safe_float(observed_llm.get("total_seconds"))
        llm_avg_seconds = _safe_float(observed_llm.get("average_seconds"))
        llm_call_count = _safe_int(observed_llm.get("call_count")) or 0
        share_of_writing = _safe_float(observed_llm.get("share_of_writing_time"))
        non_llm_writing_seconds = _safe_float(observed_llm.get("non_llm_writing_seconds"))
        llm_panel_html = f"""
        <section class="panel compact-panel">
          <h3>Observed LLM request time</h3>
          <div class="metric-stack">
            <div class="metric-row-block"><div class="metric-row"><span>Total request time</span><strong>{html.escape(_format_seconds_compact(llm_total_seconds))}</strong></div></div>
            <div class="metric-row-block"><div class="metric-row"><span>Average per request</span><strong>{html.escape(_format_seconds_compact(llm_avg_seconds))}</strong></div></div>
            <div class="metric-row-block"><div class="metric-row"><span>Requests</span><strong>{llm_call_count}</strong></div></div>
            <div class="metric-row-block"><div class="metric-row"><span>Share of writing stage</span><strong>{html.escape(f"{share_of_writing * 100:.1f}%") if isinstance(share_of_writing, (int, float)) else "—"}</strong></div></div>
            <div class="metric-row-block"><div class="metric-row"><span>Non-LLM writing overhead</span><strong>{html.escape(_format_seconds_compact(non_llm_writing_seconds))}</strong></div></div>
          </div>
        </section>
        """

    return f"""
    <section class="panel">
      <div class="card-topline">
        <div>
          <h2>Timing breakdown</h2>
          <p class="metric-subtle">Saved runtime timings from the learning process, so you can see whether time is going into generation, evaluation, or router-bound request time.{(' Aggregated from completed labels so far.' if aggregated_from_labels else '')}</p>
        </div>
      </div>
      <div class="job-overview-grid">{summary_cards_html}</div>
      <div class="learning-progress-detail-grid">
        {llm_panel_html or ""}
        {''.join(group_sections)}
      </div>
    </section>
    """


def _job_progress_html(job: JobRecord, *, project_root: Path) -> str:
    learning_summary = _learning_progress_summary(job, project_root=project_root)
    if learning_summary is not None:
        return _render_learning_progress_compact(learning_summary)
    progress = _job_progress_payload(job)
    if not progress:
        return ""
    summary = str(progress.get("summary") or "").strip()
    phase = str(progress.get("phase") or "").strip()
    try:
        current = int(progress.get("current")) if progress.get("current") is not None else None
    except (TypeError, ValueError):
        current = None
    try:
        total = int(progress.get("total")) if progress.get("total") is not None else None
    except (TypeError, ValueError):
        total = None
    fraction_raw = progress.get("fraction")
    try:
        fraction = float(fraction_raw) if fraction_raw is not None else None
    except (TypeError, ValueError):
        fraction = None
    if str(job.spec.action or "").strip() == "analyze-supervised-consistency":
        inferred = _analysis_progress_from_recent_logs(job)
        if current is None and isinstance(inferred.get("current"), int):
            current = int(inferred["current"])
        if total is None and isinstance(inferred.get("total"), int):
            total = int(inferred["total"])
        if current is not None and total is not None and total > 0:
            inferred_fraction = current / total
            if fraction is None or inferred_fraction > fraction:
                fraction = inferred_fraction
        if (not summary or "?" in summary) and isinstance(inferred.get("summary"), str):
            summary = _sanitize_display_text(str(inferred["summary"]))
        if not phase and isinstance(inferred.get("phase"), str):
            phase = _sanitize_display_text(str(inferred["phase"]))
    parts: list[str] = []
    if phase:
        parts.append(html.escape(phase))
    if current is not None and total is not None and total > 0:
        parts.append(f"{current}/{total}")
    label = " · ".join(parts)
    bar_html = ""
    if fraction is not None:
        bounded = max(0.0, min(1.0, fraction))
        percent = int(round(bounded * 100))
        bar_html = (
            '<div class="progress-track">'
            f'<div class="progress-fill" style="width: {percent}%"></div>'
            "</div>"
        )
        if not summary:
            summary = f"{percent}%"
    summary = _sanitize_display_text(summary)
    phase = _sanitize_display_text(phase)
    summary_html = f'<p class="metric-subtle">{html.escape(summary)}</p>' if summary else ""
    label_html = f'<p class="metric-subtle">{label}</p>' if label else ""
    return f'<div class="queue-progress">{label_html}{bar_html}{summary_html}</div>'


def _analysis_progress_from_recent_logs(
    job: JobRecord,
    *,
    log_lines: list[str] | None = None,
) -> dict[str, Any]:
    action = str(job.spec.action or "").strip()
    if action != "analyze-supervised-consistency":
        return {}
    current: int | None = None
    total: int | None = None
    runs: int | None = None
    windows: int | None = None
    for raw_line in list(log_lines if log_lines is not None else (job.recent_logs or [])):
        line = str(raw_line or "")
        repeat_match = _ANALYSIS_REPEAT_PROGRESS_PATTERN.search(line)
        if repeat_match is not None:
            try:
                current = int(repeat_match.group("current"))
                total = int(repeat_match.group("total"))
            except (TypeError, ValueError):
                pass
        start_match = _ANALYSIS_START_PATTERN.search(line)
        if start_match is not None:
            raw_runs = start_match.group("runs")
            raw_windows = start_match.group("windows")
            raw_checkpoints = start_match.group("checkpoints")
            try:
                if raw_runs is not None:
                    runs = int(raw_runs)
            except (TypeError, ValueError):
                pass
            try:
                if raw_windows is not None:
                    windows = int(raw_windows)
                elif raw_checkpoints is not None:
                    windows = int(raw_checkpoints)
            except (TypeError, ValueError):
                pass
    summary_bits: list[str] = []
    if runs is not None:
        summary_bits.append(f"{runs} runs")
    if windows is not None:
        summary_bits.append(f"{windows} windows")
    summary = " across ".join(summary_bits) if len(summary_bits) == 2 else (" · ".join(summary_bits) if summary_bits else "")
    if current is not None and total is not None and total > 0:
        phase = "Sampling repeats"
    else:
        phase = "Consistency analysis"
    return {
        "phase": phase,
        "summary": summary,
        "current": current,
        "total": total,
    }


def _render_learning_progress_compact(summary: dict[str, Any]) -> str:
    total_labels = int(summary.get("total_labels") or 0)
    completed_labels = int(summary.get("completed_labels") or 0)
    completion_percent = ((completed_labels / total_labels) * 100.0) if total_labels else 0.0
    workers = summary.get("workers") if isinstance(summary.get("workers"), list) else []
    latest_score = summary.get("latest_score")
    max_worker_evals = max(
        [
            int(worker.get("candidate_evaluations") or 0)
            for worker in workers
            if isinstance(worker, dict)
        ]
        or [0]
    )
    metric_cards: list[str] = []
    if isinstance(latest_score, (int, float)):
        metric_cards.append(
            f'<div class="job-progress-metric-card"><span class="job-progress-metric-label">Reward</span><strong class="job-progress-metric-value">{latest_score:.3f}</strong></div>'
        )
    if workers:
        metric_cards.append(
            f'<div class="job-progress-metric-card"><span class="job-progress-metric-label">Workers</span><strong class="job-progress-metric-value">{len(workers)}</strong></div>'
        )
    line_pills: list[str] = []
    active_label = summary.get("active_label")
    if isinstance(active_label, str) and active_label.strip():
        line_pills.append(f'<span class="job-progress-pill job-progress-pill-strong">{html.escape(active_label)}</span>')
    stage = summary.get("stage")
    if isinstance(stage, str) and stage:
        line_pills.append(
            f'<span class="job-progress-pill">{html.escape(display_learning_stage(stage))}</span>'
        )
    evals = summary.get("candidate_evaluations")
    if isinstance(evals, int):
        line_pills.append(f'<span class="job-progress-pill">{evals} evals</span>')
    worker_cards = "".join(
        _render_worker_visual_card(worker, max_worker_evals=max_worker_evals)
        for worker in workers[:4]
        if isinstance(worker, dict)
    )
    return f"""
    <div class="job-progress-stack">
      <div class="job-progress-hero">
        <div class="job-progress-hero-main">
          <div class="job-progress-kicker">Run progress</div>
          <div class="job-progress-hero-value">{completed_labels}/{total_labels if total_labels else "—"}</div>
          <div class="job-progress-hero-label">labels complete</div>
        </div>
        <div class="job-progress-hero-side">{''.join(metric_cards)}</div>
      </div>
      <div class="progress-block-tight">
        <div class="progress-meta"><span>Label progress</span><strong>{completion_percent:.0f}%</strong></div>
        <div class="progress-track job-progress-track-large"><span class="progress-fill" style="width: {completion_percent:.2f}%"></span></div>
      </div>
      {f'<div class="job-progress-line">{"".join(line_pills)}</div>' if line_pills else ''}
      {f'<div class="job-worker-visual-list">{worker_cards}</div>' if worker_cards else ''}
    </div>
    """


def _render_worker_visual_card(worker: dict[str, Any], *, max_worker_evals: int) -> str:
    evals = worker.get("candidate_evaluations")
    evals_int = int(evals) if isinstance(evals, int) else None
    eval_percent = ((evals_int / max_worker_evals) * 100.0) if evals_int is not None and max_worker_evals else 0.0
    stage = str(worker.get("stage") or "queued").replace("_", "-")
    meta_bits: list[str] = []
    score = worker.get("latest_score")
    if isinstance(score, (int, float)):
        meta_bits.append(f"reward {score:.3f}")
    val = worker.get("latest_validation_score")
    if isinstance(val, (int, float)):
        meta_bits.append(f"val {val:.3f}")
    return f"""
    <div class="job-worker-visual-card">
      <div class="job-worker-visual-header">
        <span class="job-worker-mini-id">w{html.escape(str(worker.get('worker') or '—'))}</span>
        <span class="job-worker-stage-pill stage-{html.escape(stage)}">{html.escape(display_learning_stage(worker.get('stage') or 'queued'))}</span>
      </div>
      <div class="job-worker-visual-label">{html.escape(str(worker.get('label') or 'Idle'))}</div>
      <div class="progress-block-tight job-worker-progress-block">
        <div class="progress-meta"><span>Evals</span><strong>{html.escape(str(evals_int) if evals_int is not None else '—')}</strong></div>
        <div class="progress-track"><span class="progress-fill" style="width: {eval_percent:.2f}%"></span></div>
      </div>
      <div class="job-worker-visual-meta">{''.join(f'<span>{html.escape(bit)}</span>' for bit in meta_bits)}</div>
    </div>
    """


def _render_learning_embedded_eval_panel(progress: dict[str, Any]) -> str:
    completed_count = int(progress.get("completed_count") or 0)
    total_count = int(progress.get("total_count") or 0)
    latest_completed_task_id = _sanitize_display_text(str(progress.get("latest_completed_task_id") or "").strip())
    active_task_id = _sanitize_display_text(str(progress.get("active_task_id") or "").strip())
    active_step = _safe_int(progress.get("active_step"))
    run_id = _sanitize_display_text(str(progress.get("run_id") or "").strip())
    latest_activity_at = progress.get("latest_activity_at")
    latest_activity_kind = _sanitize_display_text(str(progress.get("latest_activity_kind") or "").strip())
    latest_activity_task_id = _sanitize_display_text(str(progress.get("latest_activity_task_id") or "").strip())
    results_ready = bool(progress.get("results_ready"))
    summary_ready = bool(progress.get("summary_ready"))
    completion_percent = (
        max(0.0, min(100.0, (completed_count / total_count) * 100.0))
        if total_count > 0
        else 0.0
    )
    status_bits: list[str] = []
    if active_task_id:
        if active_step is not None:
            status_bits.append(f"Active task: {active_task_id} step {active_step:03d}")
        else:
            status_bits.append(f"Active task: {active_task_id}")
    elif completed_count and total_count and completed_count >= total_count and not (results_ready and summary_ready):
        status_bits.append("All task results are on disk; waiting for final eval summary files.")
    elif latest_completed_task_id:
        status_bits.append(f"Latest completed task: {latest_completed_task_id}")
    if isinstance(latest_activity_at, datetime):
        activity_subject = latest_activity_task_id or "unknown task"
        activity_kind = latest_activity_kind or "artifact"
        status_bits.append(
            f"Latest {activity_kind} write: {activity_subject} at {_display_time(latest_activity_at)}"
        )
    return f"""
    <section class="panel">
      <div class="card-topline">
        <div>
          <h2>Embedded DeepPHY evaluation</h2>
          <p class="metric-subtle">This run is evaluating the current detector snapshot while label learning remains at a partial checkpoint.</p>
        </div>
        <div class="stats-inline">
          <span class="pill status-running">{html.escape(run_id or 'active eval')}</span>
          <span class="pill status-running">{completed_count}/{total_count if total_count > 0 else '—'} tasks</span>
        </div>
      </div>
      <div class="progress-block-tight">
        <div class="progress-meta"><span>Eval progress</span><strong>{completion_percent:.0f}%</strong></div>
        <div class="progress-track job-progress-track-large"><span class="progress-fill" style="width: {completion_percent:.2f}%"></span></div>
      </div>
      <div class="metric-stack">
        {''.join(f'<div class="metric-row-block">{html.escape(bit)}</div>' for bit in status_bits)}
      </div>
    </section>
    """


def _render_learning_progress_detail(
    summary: dict[str, Any],
    *,
    job: JobRecord,
    project_root: Path,
) -> str:
    compact = _render_learning_progress_compact(summary)
    workers = summary.get("workers") if isinstance(summary.get("workers"), list) else []
    provisional_labels = summary.get("provisional_labels") if isinstance(summary.get("provisional_labels"), list) else []
    embedded_eval = _learning_embedded_eval_progress_summary(job, project_root=project_root)
    worker_rows = "".join(_render_worker_progress_row(worker) for worker in workers if isinstance(worker, dict))
    provisional_rows = "".join(_render_provisional_label_row(label) for label in provisional_labels if isinstance(label, dict))
    worker_table = (
        f"""
        <section class="panel">
          <h2>Worker progress</h2>
          <table class="data-table compact-table">
            <thead><tr><th>Worker</th><th>Label</th><th>Evals</th><th>Stage</th><th>Score</th><th>Val</th></tr></thead>
            <tbody>{worker_rows}</tbody>
          </table>
        </section>
        """
        if worker_rows
        else (
            '<section class="panel"><h2>Worker progress</h2><p class="empty">No label-search worker activity recorded right now. This run is currently in embedded DeepPHY evaluation.</p></section>'
            if embedded_eval is not None
            else '<section class="panel"><h2>Worker progress</h2><p class="empty">No worker activity recorded yet.</p></section>'
        )
    )
    provisional_table = (
        f"""
        <section class="panel">
          <h2>Live label summary</h2>
          <table class="data-table compact-table">
            <thead><tr><th>Label</th><th>Evals</th><th>Best reward</th><th>Validation</th></tr></thead>
            <tbody>{provisional_rows}</tbody>
          </table>
        </section>
        """
        if provisional_rows
        else ""
    )
    return f"""
    <section class="panel">
      <div class="card-topline">
        <div>
          <h2>Live learning progress</h2>
          <p class="metric-subtle">Current run state, worker progress, and per-label provisional status.</p>
        </div>
      </div>
      {compact}
    </section>
    {_render_learning_embedded_eval_panel(embedded_eval) if embedded_eval is not None else ''}
    <div class="learning-progress-detail-grid">
      {worker_table}
      {provisional_table or '<section class="panel"><h2>Live label summary</h2><p class="empty">No provisional per-label history recorded yet.</p></section>'}
    </div>
    """


def _event_tone_class(tone: str) -> str:
    normalized = str(tone or "").strip().lower()
    if normalized in {"warning", "warn"}:
        return "warning"
    if normalized in {"danger", "failed", "failure"}:
        return "danger"
    if normalized in {"succeeded", "success"}:
        return "success"
    return "info"


def _dashboard_event_title(job: JobRecord) -> str:
    env_id = _job_env_id(job)
    family = _job_required_family(job) or _job_flow_label(job) or job.spec.action
    if env_id and env_id != "—":
        return f"{env_id.upper()} · {family}"
    return family


def _trim_event_summary(text: str, *, limit: int = 180) -> str:
    cleaned = _sanitize_display_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "…"


def _event_from_log_line(*, job: JobRecord, line: str) -> DashboardEvent | None:
    cleaned = _sanitize_display_text(line)
    if not cleaned:
        return None
    title = _dashboard_event_title(job)
    href = f"/jobs/{job.job_id}"
    occurred_at = _job_last_activity(job)

    if "patterns.learning.label.done:" in cleaned:
        payload = cleaned.split("patterns.learning.label.done:", 1)[1].strip()
        return DashboardEvent(
            occurred_at=occurred_at,
            tone="success",
            title=title,
            summary=_trim_event_summary(payload),
            href=href,
            context="Label",
        )
    if "patterns.learning.lm_request.done:" in cleaned:
        payload = cleaned.split("patterns.learning.lm_request.done:", 1)[1].strip()
        return DashboardEvent(
            occurred_at=occurred_at,
            tone="info",
            title=title,
            summary=_trim_event_summary(payload),
            href=href,
            context="LLM",
        )
    if "dashboard.workflows.supervised.learning_done:" in cleaned or "dashboard.workflows.unsupervised.learning_done:" in cleaned:
        payload = cleaned.split("learning_done:", 1)[1].strip()
        return DashboardEvent(
            occurred_at=occurred_at,
            tone="success",
            title=title,
            summary=_trim_event_summary(payload),
            href=href,
            context="Run",
        )
    if "dashboard.workflows.supervised.evaluation_done:" in cleaned or "dashboard.workflows.unsupervised.evaluation_done:" in cleaned:
        payload = cleaned.split("evaluation_done:", 1)[1].strip()
        return DashboardEvent(
            occurred_at=occurred_at,
            tone="success",
            title=title,
            summary=_trim_event_summary(payload),
            href=href,
            context="Eval",
        )
    if "dashboard.workflows.child.done:" in cleaned and "returncode=0" in cleaned:
        payload = cleaned.split("dashboard.workflows.child.done:", 1)[1].strip()
        return DashboardEvent(
            occurred_at=occurred_at,
            tone="success",
            title=title,
            summary=_trim_event_summary(payload),
            href=href,
            context="Stage",
        )
    if "resume_fallback_fresh_start:" in cleaned:
        payload = cleaned.split("resume_fallback_fresh_start:", 1)[1].strip()
        return DashboardEvent(
            occurred_at=occurred_at,
            tone="warning",
            title=title,
            summary=_trim_event_summary(payload),
            href=href,
            context="Recovery",
        )
    return None


def _job_log_events(job: JobRecord) -> list[DashboardEvent]:
    events: list[DashboardEvent] = []
    for line in list(job.recent_logs)[-16:]:
        event = _event_from_log_line(job=job, line=line)
        if event is not None:
            events.append(event)
    return events


def _job_task_events(job: JobRecord, *, project_root: Path) -> list[DashboardEvent]:
    if str(job.spec.action or "").strip() != "evaluate-deepphy":
        return []
    title = _dashboard_event_title(job)
    href = f"/jobs/{job.job_id}"
    task_events: list[DashboardEvent] = []
    for detail in sorted(
        _evaluation_task_list(job, project_root=project_root),
        key=lambda item: item.completed_at or job.updated_at,
        reverse=True,
    )[:8]:
        if detail.completed_at is None or detail.success is None:
            continue
        outcome = "succeeded" if detail.success else "failed"
        attempts = len(detail.attempts)
        summary = f"{detail.task_id} {outcome}"
        if attempts:
            summary += f" after {attempts} attempt{'s' if attempts != 1 else ''}"
        task_events.append(
            DashboardEvent(
                occurred_at=detail.completed_at,
                tone="success" if detail.success else "danger",
                title=title,
                summary=_trim_event_summary(summary),
                href=href,
                context="Task",
            )
        )
    return task_events


def _dashboard_events(
    *,
    jobs: list[JobRecord],
    clients: list[ClientRecord],
    project_root: Path,
    limit: int = HOME_EVENT_PANEL_LIMIT,
) -> list[DashboardEvent]:
    now = datetime.now(tz=DISPLAY_TZ)
    events: list[DashboardEvent] = []
    recent_jobs = sorted(jobs, key=_job_last_activity, reverse=True)[:28]
    for job in recent_jobs:
        title = _dashboard_event_title(job)
        href = f"/jobs/{job.job_id}"
        status = job.status.value
        message = _sanitize_display_text(job.message)
        occurred_at = _job_last_activity(job)
        if message.startswith("Recovered after client") or message.startswith("Re-queued after client"):
            events.append(
                DashboardEvent(
                    occurred_at=occurred_at,
                    tone="warning",
                    title=title,
                    summary=message,
                    href=href,
                    context="Recovered",
                )
            )
        elif status in FAILED_QUEUE_STATUSES:
            events.append(
                DashboardEvent(
                    occurred_at=occurred_at,
                    tone="danger",
                    title=title,
                    summary=message or "Run failed or stopped.",
                    href=href,
                    context="Failed",
                )
            )
        events.extend(_job_log_events(job))
        if max(0, int((now - occurred_at).total_seconds())) <= 48 * 3600:
            events.extend(_job_task_events(job, project_root=project_root))
    recent_clients = sorted(
        _dashboard_active_clients(clients),
        key=lambda item: item.last_heartbeat,
        reverse=True,
    )[:10]
    for client in recent_clients:
        age_seconds = max(0, int((now - client.last_heartbeat).total_seconds()))
        if client.status.value == "offline" and age_seconds <= 18 * 3600:
            current_jobs = len(client.current_jobs)
            summary = f"Worker offline · last heartbeat {_display_duration(client.last_heartbeat, now)} ago"
            if current_jobs:
                summary += f" · {current_jobs} jobs still attached"
            events.append(
                DashboardEvent(
                    occurred_at=client.last_heartbeat,
                    tone="warning",
                    title=_client_title(client),
                    summary=summary,
                    href=f"/clients/{client.client_id}",
                    context="Client",
                )
            )
    deduped: list[DashboardEvent] = []
    seen: set[tuple[str, str, str | None]] = set()
    for event in sorted(events, key=lambda item: item.occurred_at, reverse=True):
        key = (event.title, event.summary, event.context)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
        if len(deduped) >= limit:
            break
    return deduped


def _render_dashboard_events_panel(*, jobs: list[JobRecord], clients: list[ClientRecord], project_root: Path) -> str:
    events = _dashboard_events(jobs=jobs, clients=clients, project_root=project_root)
    if not events:
        body = '<p class="empty">No recent high-signal events yet.</p>'
    else:
        rows: list[str] = []
        for event in events:
            row_main = (
                f'<a class="event-row-main" href="{html.escape(event.href)}"><strong>{html.escape(event.title)}</strong><p class="metric-subtle">{html.escape(event.summary)}</p></a>'
                if event.href
                else f'<div class="event-row-main"><strong>{html.escape(event.title)}</strong><p class="metric-subtle">{html.escape(event.summary)}</p></div>'
            )
            rows.append(
                f"""
                <article class="event-row tone-{_event_tone_class(event.tone)}">
                  <div class="event-row-prefix">
                    <span class="event-dot tone-{_event_tone_class(event.tone)}"></span>
                    {f'<span class="mini-chip">{html.escape(event.context)}</span>' if event.context else ''}
                  </div>
                  {row_main}
                  <div class="event-row-side">
                    <time class="metric-subtle">{html.escape(_display_time(event.occurred_at))}</time>
                  </div>
                </article>
                """
            )
        body = '<div class="event-list">' + "".join(rows) + "</div>"
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Overview</p>
          <h2>Recent Activity</h2>
          <p class="metric-subtle">Important client and job events only.</p>
        </div>
        <div class="stats-inline">
          <span class="pill">Feed {len(events)}</span>
        </div>
      </div>
      {body}
    </section>
    """


def _render_deepphy_success_summary(*, job: JobRecord, evaluation_tasks: list[EvaluationTaskDetail]) -> str:
    if not evaluation_tasks:
        return ""
    completed = [item for item in evaluation_tasks if item.success is not None]
    success_count = sum(1 for item in completed if item.success is True)
    failure_count = sum(1 for item in completed if item.success is False)
    completed_count = len(completed)
    total_count = len(evaluation_tasks)
    success_rate = (success_count / completed_count * 100.0) if completed_count else 0.0
    tone = "status-running" if success_count and not failure_count else "status-paused" if completed_count else "status-queued"
    mode_label = _deepphy_eval_mode_label(job) or "Unknown"
    return f"""
    <section class="panel">
      <div class="card-topline">
        <div>
          <p class="section-kicker">DeepPHY</p>
          <h2>Task Success Rate</h2>
          <p class="metric-subtle">Completed task outcomes across this evaluation run.</p>
        </div>
        <div class="stats-inline">
          <span class="pill {tone}">{success_rate:.0f}% success</span>
        </div>
      </div>
      <div class="job-overview-grid">
        <article class="job-overview-card">
          <span class="job-overview-kicker">Success rate</span>
          <strong class="job-overview-value">{success_rate:.1f}%</strong>
          <span class="job-overview-meta">{success_count} of {completed_count} completed tasks succeeded</span>
        </article>
        <article class="job-overview-card">
          <span class="job-overview-kicker">Completed</span>
          <strong class="job-overview-value">{completed_count}/{total_count}</strong>
          <span class="job-overview-meta">{total_count - completed_count} tasks still pending</span>
        </article>
        <article class="job-overview-card">
          <span class="job-overview-kicker">Eval mode</span>
          <strong class="job-overview-value">{html.escape(mode_label)}</strong>
          <span class="job-overview-meta">Whether this run uses pattern descriptions or image-only traces</span>
        </article>
        <article class="job-overview-card">
          <span class="job-overview-kicker">Succeeded</span>
          <strong class="job-overview-value">{success_count}</strong>
          <span class="job-overview-meta">Successful task outcomes</span>
        </article>
        <article class="job-overview-card">
          <span class="job-overview-kicker">Failed</span>
          <strong class="job-overview-value">{failure_count}</strong>
          <span class="job-overview-meta">Failed task outcomes</span>
        </article>
      </div>
    </section>
    """


def _render_worker_progress_row(worker: dict[str, Any]) -> str:
    score = worker.get("latest_score")
    score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
    val = worker.get("latest_validation_score")
    val_text = f"{val:.3f}" if isinstance(val, (int, float)) else "—"
    evals = worker.get("candidate_evaluations")
    evals_text = str(evals) if evals is not None else "—"
    return (
        "<tr>"
        f"<td>{html.escape(str(worker.get('worker') or '—'))}</td>"
        f"<td>{html.escape(str(worker.get('label') or '—'))}</td>"
        f"<td>{html.escape(evals_text)}</td>"
        f"<td>{html.escape(display_learning_stage(worker.get('stage')))}</td>"
        f"<td>{score_text}</td>"
        f"<td>{val_text}</td>"
        "</tr>"
    )


def _render_provisional_label_row(label: dict[str, Any]) -> str:
    best_score = label.get("best_score")
    best_score_text = f"{best_score:.3f}" if isinstance(best_score, (int, float)) else "—"
    best_val = label.get("best_validation_score")
    best_val_text = f"{best_val:.3f}" if isinstance(best_val, (int, float)) else "—"
    evals = label.get("candidate_evaluations")
    evals_text = str(evals) if evals is not None else "—"
    return (
        "<tr>"
        f"<td>{html.escape(str(label.get('label') or '—'))}</td>"
        f"<td>{html.escape(evals_text)}</td>"
        f"<td>{best_score_text}</td>"
        f"<td>{best_val_text}</td>"
        "</tr>"
    )


def _repeat_run_name(job: JobRecord, *, project_root: Path) -> str | None:
    run_dir = _resolve_job_run_dir(job, project_root=project_root)
    if run_dir is not None:
        match = _RESUME_SYNC_RUN_SUFFIX_PATTERN.match(run_dir.name)
        return str(match.group("base")) if match is not None else run_dir.name
    metadata = dict(job.spec.metadata or {})
    request = dict(metadata.get("request") or {})
    for value in (
        request.get("run_timestamp"),
        request.get("output_path"),
        metadata.get("output_path"),
    ):
        text = str(value or "").strip()
        if text:
            return Path(text).name
    return None


def _repeat_run_group_key(
    job: JobRecord,
    *,
    project_root: Path,
) -> tuple[str, str, str] | None:
    family = _job_repeat_learning_family(job)
    if family is None:
        return None
    env_id = str(job.spec.metadata.get("env_id") or job.environment or "").strip()
    run_name = _repeat_run_name(job, project_root=project_root)
    if not env_id or not run_name:
        return None
    match = _REPEAT_RUN_SUFFIX_PATTERN.match(run_name)
    if match is None:
        return None
    return (env_id, family, str(match.group("prefix")))


def _repeat_run_index(job: JobRecord, *, project_root: Path) -> int | None:
    run_name = _repeat_run_name(job, project_root=project_root)
    if not run_name:
        return None
    match = _REPEAT_RUN_SUFFIX_PATTERN.match(run_name)
    if match is None:
        return None
    try:
        return int(match.group("index"))
    except (TypeError, ValueError):
        return None


def _resolve_deepphy_eval_run_dir(job: JobRecord, *, project_root: Path) -> Path | None:
    metadata = dict(job.spec.metadata or {})
    for raw in (
        metadata.get("evaluation_run_dir"),
        metadata.get("run_dir"),
    ):
        normalized = _normalize_project_path(str(raw or "").strip(), project_root=project_root)
        if normalized is not None and normalized.exists():
            return normalized
    pattern_run_dir = str(metadata.get("pattern_run_dir") or "").strip()
    evaluation_run_id = str(
        metadata.get("evaluation_run_id")
        or (dict(metadata.get("request") or {}).get("run_id"))
        or ""
    ).strip()
    if pattern_run_dir and evaluation_run_id:
        normalized_base = _normalize_project_path(pattern_run_dir, project_root=project_root)
        if normalized_base is None:
            return None
        candidate = normalized_base / "deepphy_eval" / evaluation_run_id
        try:
            normalized = candidate.resolve()
        except OSError:
            return None
        if normalized.exists():
            return normalized
    return None


def _repeat_eval_run_name(job: JobRecord, *, project_root: Path) -> str | None:
    metadata = dict(job.spec.metadata or {})
    evaluation_run_id = str(
        metadata.get("evaluation_run_id")
        or (dict(metadata.get("request") or {}).get("run_id"))
        or ""
    ).strip()
    if evaluation_run_id:
        return evaluation_run_id
    eval_dir = _resolve_deepphy_eval_run_dir(job, project_root=project_root)
    if eval_dir is not None:
        return eval_dir.name
    return None


def _repeat_eval_run_index(job: JobRecord, *, project_root: Path) -> int | None:
    run_name = _repeat_eval_run_name(job, project_root=project_root)
    if not run_name:
        return None
    match = _REPEAT_EVAL_SUFFIX_PATTERN.match(run_name)
    if match is None:
        return None
    try:
        return int(match.group("index"))
    except (TypeError, ValueError):
        return None


def _repeat_eval_pattern_run_name(job: JobRecord, *, project_root: Path) -> str | None:
    metadata = dict(job.spec.metadata or {})
    raw = str(metadata.get("pattern_run_dir") or "").strip()
    if not raw:
        request = dict(metadata.get("request") or {})
        raw = str(request.get("pattern_run_dir") or "").strip()
    if not raw:
        return None
    candidate = Path(raw)
    resolved = candidate if candidate.is_absolute() else (project_root / candidate)
    try:
        normalized = resolved.resolve()
    except OSError:
        normalized = resolved
    return normalized.name


def _deepphy_eval_success_metrics(
    job: JobRecord,
    *,
    project_root: Path,
    pooltool_attempt_limit: int | None = None,
) -> tuple[int, int, float] | None:
    eval_root = _resolve_deepphy_eval_run_dir(job, project_root=project_root)
    if eval_root is None:
        return None
    counts = _required_catalog_deepphy_eval_counts_cached(
        str(eval_root),
        _required_catalog_deepphy_eval_marker(eval_root),
        pooltool_attempt_limit,
    )
    if counts is None:
        return None
    success_count, _failure_count, total_tasks = counts
    if total_tasks <= 0:
        return None
    return (
        success_count,
        total_tasks,
        max(0.0, min(100.0, (success_count / total_tasks) * 100.0)),
    )


def _learning_run_embedded_eval_root(job: JobRecord, *, project_root: Path) -> Path | None:
    run_dir = _resolve_job_run_dir(job, project_root=project_root)
    if run_dir is None:
        return None
    run_payload = _load_json_dict(run_dir / "run.json")
    total_labels = _safe_int(run_payload.get("primitive_label_count")) or 0
    payload_completed_labels = _safe_int(run_payload.get("completed_label_count")) or 0
    completed_label_results = sum(
        1 for label_dir in _unique_label_dirs(run_dir) if (label_dir / "label_result.json").is_file()
    )
    has_label_dirs = any(path.is_dir() for path in run_dir.glob("label_*"))
    if has_label_dirs:
        completed_labels = completed_label_results
    else:
        completed_labels = max(payload_completed_labels, completed_label_results)
    if total_labels > 0:
        completed_labels = max(0, min(completed_labels, total_labels))
    if total_labels <= 0 or completed_labels < total_labels:
        return None
    if str(job.spec.action or "").strip() == "automatic-launch":
        return _automatic_deepphy_eval_root(job, project_root=project_root)
    deepphy_root = run_dir / "deepphy_eval"
    if not deepphy_root.is_dir():
        return None
    run_json_path = run_dir / "run.json"
    if run_json_path.is_file():
        try:
            payload = json.loads(run_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            auto_run_id = str(payload.get("auto_evaluation_run_id") or "").strip()
            if auto_run_id:
                candidate = deepphy_root / auto_run_id
                if candidate.is_dir():
                    return candidate
    completed_candidates: list[tuple[float, Path]] = []
    active_candidates: list[tuple[float, Path]] = []
    for child in sorted(deepphy_root.iterdir()):
        if not child.is_dir():
            continue
        if _is_archived_deepphy_eval_dir(child):
            continue
        marker = _required_catalog_deepphy_eval_marker(child)
        counts = _required_catalog_deepphy_eval_counts_cached(str(child), marker)
        latest_mtime = float(max(marker) or 0)
        if counts is not None:
            completed_candidates.append((latest_mtime, child))
            continue
        if any(child.glob("tasks/*/*/task_result.json")) or any(child.glob("tasks/*/*/step_*/trace.json")):
            active_candidates.append((latest_mtime, child))
    if completed_candidates:
        return max(completed_candidates, key=lambda item: (item[0], item[1].name))[1]
    if active_candidates:
        return max(active_candidates, key=lambda item: (item[0], item[1].name))[1]
    return None


def _learning_active_embedded_eval_root(job: JobRecord, *, project_root: Path) -> Path | None:
    if not _is_learning_job(job):
        return None
    run_dir = _resolve_job_run_dir(job, project_root=project_root)
    if run_dir is None:
        return None
    deepphy_root = run_dir / "deepphy_eval"
    if not deepphy_root.is_dir():
        return None
    run_payload = _load_json_dict(run_dir / "run.json")
    auto_run_id = str(run_payload.get("auto_evaluation_run_id") or "").strip()
    if not auto_run_id:
        return None
    candidate = deepphy_root / auto_run_id
    if not candidate.is_dir() or _is_archived_deepphy_eval_dir(candidate):
        return None
    return candidate


def _learning_embedded_eval_progress_summary(
    job: JobRecord,
    *,
    project_root: Path,
) -> dict[str, Any] | None:
    eval_root = _learning_active_embedded_eval_root(job, project_root=project_root)
    if eval_root is None:
        return None
    total_hint = sum(1 for task_dir in eval_root.glob("tasks/*/*") if task_dir.is_dir())
    progress = _deepphy_live_progress_summary_for_eval_root(
        eval_root=eval_root,
        total_hint=total_hint,
    )
    if progress is None:
        return None
    progress = dict(progress)
    progress["eval_root"] = eval_root
    progress["run_id"] = eval_root.name
    progress["results_ready"] = (eval_root / "results.json").is_file()
    progress["summary_ready"] = (eval_root / "summary.json").is_file()
    return progress


def _learning_run_manual_eval_jobs(
    *,
    env_id: str,
    run_job: JobRecord,
    jobs: list[JobRecord],
    project_root: Path,
) -> list[JobRecord]:
    def _canonical_learning_root(path: Path) -> Path:
        match = _RESUME_SYNC_RUN_SUFFIX_PATTERN.match(path.name)
        if match is not None:
            return path.with_name(str(match.group("base")))
        return path

    run_dir = _resolve_job_run_dir(run_job, project_root=project_root)
    if run_dir is None:
        return []
    run_dir_text = str(run_dir.resolve())
    canonical_run_dir_text = str(_canonical_learning_root(run_dir).resolve())
    matches: list[JobRecord] = []
    for candidate in jobs:
        if str(candidate.spec.action or "").strip() != "evaluate-deepphy":
            continue
        if candidate.status.value == "cancelled" or _job_is_superseded_combined_source(candidate):
            continue
        candidate_env = str(candidate.spec.metadata.get("env_id") or candidate.environment or "").strip().lower()
        if candidate_env != env_id:
            continue
        eval_run_name = _repeat_eval_run_name(candidate, project_root=project_root) or ""
        if _REPEAT_EVAL_SUFFIX_PATTERN.match(eval_run_name):
            continue
        pattern_run_dir = str(candidate.spec.metadata.get("pattern_run_dir") or "").strip()
        if not pattern_run_dir:
            request_payload = dict(candidate.spec.metadata.get("request") or {})
            pattern_run_dir = str(request_payload.get("pattern_run_dir") or "").strip()
        if not pattern_run_dir:
            continue
        normalized = _normalize_project_path(pattern_run_dir, project_root=project_root)
        if normalized is None:
            continue
        try:
            normalized_text = str(normalized.resolve())
        except OSError:
            normalized_text = str(normalized)
        if normalized_text not in {run_dir_text, canonical_run_dir_text}:
            continue
        matches.append(candidate)
    return sorted(matches, key=lambda item: (item.created_at, item.job_id))


def _deepphy_live_progress_marker(eval_root: Path | None) -> tuple[int, int, int]:
    if eval_root is None or not eval_root.exists():
        return (0, 0, 0)
    task_results = 0
    traces = 0
    latest_mtime_ns = 0
    for result_path in eval_root.glob("tasks/*/*/task_result.json"):
        task_results += 1
        try:
            latest_mtime_ns = max(latest_mtime_ns, result_path.stat().st_mtime_ns)
        except OSError:
            continue
    for trace_path in eval_root.glob("tasks/*/*/step_*/trace.json"):
        traces += 1
        try:
            latest_mtime_ns = max(latest_mtime_ns, trace_path.stat().st_mtime_ns)
        except OSError:
            continue
    return (task_results, traces, latest_mtime_ns)


@lru_cache(maxsize=512)
def _deepphy_live_progress_cached(
    eval_root_text: str,
    marker: tuple[int, int, int],
    total_hint: int,
) -> dict[str, Any] | None:
    del marker
    eval_root = Path(eval_root_text)
    if not eval_root.is_dir():
        return None
    completed_task_ids: set[str] = set()
    seen_task_ids: set[str] = set()
    latest_result_path: Path | None = None
    latest_result_mtime = 0.0
    latest_incomplete_trace_path: Path | None = None
    latest_incomplete_trace_mtime = 0.0
    latest_any_trace_path: Path | None = None
    latest_any_trace_mtime = 0.0
    for result_path in eval_root.glob("tasks/*/*/task_result.json"):
        task_id = result_path.parent.name
        completed_task_ids.add(task_id)
        seen_task_ids.add(task_id)
        try:
            mtime = result_path.stat().st_mtime
        except OSError:
            continue
        if mtime >= latest_result_mtime:
            latest_result_mtime = mtime
            latest_result_path = result_path
    for trace_path in eval_root.glob("tasks/*/*/step_*/trace.json"):
        task_id = trace_path.parent.parent.name
        seen_task_ids.add(task_id)
        try:
            mtime = trace_path.stat().st_mtime
        except OSError:
            continue
        if mtime >= latest_any_trace_mtime:
            latest_any_trace_mtime = mtime
            latest_any_trace_path = trace_path
        if task_id in completed_task_ids:
            continue
        if mtime >= latest_incomplete_trace_mtime:
            latest_incomplete_trace_mtime = mtime
            latest_incomplete_trace_path = trace_path
    active_trace_path = latest_incomplete_trace_path
    active_task_id = active_trace_path.parent.parent.name if active_trace_path is not None else None
    active_step_name = active_trace_path.parent.name if active_trace_path is not None else ""
    active_step = None
    if active_step_name.startswith("step_"):
        active_step = _safe_int(active_step_name.split("_", 1)[1])
    latest_activity_path = latest_any_trace_path
    latest_activity_mtime = latest_any_trace_mtime
    latest_activity_kind = "trace"
    if latest_result_mtime > latest_activity_mtime:
        latest_activity_path = latest_result_path
        latest_activity_mtime = latest_result_mtime
        latest_activity_kind = "task_result"
    latest_activity_at = (
        datetime.fromtimestamp(latest_activity_mtime, tz=DISPLAY_TZ)
        if latest_activity_mtime > 0.0
        else None
    )
    latest_completed_task_id = latest_result_path.parent.name if latest_result_path is not None else None
    total_count = max(int(total_hint or 0), len(seen_task_ids), len(completed_task_ids))
    return {
        "completed_count": len(completed_task_ids),
        "total_count": total_count,
        "active_task_id": active_task_id,
        "active_step": active_step,
        "latest_completed_task_id": latest_completed_task_id,
        "latest_activity_kind": latest_activity_kind,
        "latest_activity_at": latest_activity_at,
        "latest_activity_task_id": (
            latest_activity_path.parent.parent.name
            if latest_activity_path is not None and latest_activity_kind == "trace"
            else latest_completed_task_id
        ),
    }


def _deepphy_live_progress_summary(job: JobRecord, *, project_root: Path) -> dict[str, Any] | None:
    if str(job.spec.action or "").strip() != "evaluate-deepphy":
        return None
    eval_root = _resolve_deepphy_eval_run_dir(job, project_root=project_root)
    if eval_root is None:
        return None
    metadata = dict(job.spec.metadata or {})
    total_hint = _safe_int(metadata.get("benchmark_task_count"))
    if total_hint is None:
        request_payload = metadata.get("request")
        if isinstance(request_payload, dict):
            total_hint = _safe_int(request_payload.get("task_limit"))
    return _deepphy_live_progress_summary_for_eval_root(
        eval_root=eval_root,
        total_hint=int(total_hint or 0),
    )


def _deepphy_live_progress_summary_for_eval_root(
    *,
    eval_root: Path | None,
    total_hint: int = 0,
) -> dict[str, Any] | None:
    if eval_root is None:
        return None
    return _deepphy_live_progress_cached(
        str(eval_root),
        _deepphy_live_progress_marker(eval_root),
        int(total_hint or 0),
    )


def _deepphy_live_progress_lines(job: JobRecord, *, project_root: Path) -> list[str]:
    progress = _deepphy_live_progress_summary(job, project_root=project_root)
    if progress is None:
        return []
    completed_count = int(progress.get("completed_count") or 0)
    total_count = int(progress.get("total_count") or 0)
    latest_completed_task_id = str(progress.get("latest_completed_task_id") or "").strip()
    active_task_id = str(progress.get("active_task_id") or "").strip()
    active_step = _safe_int(progress.get("active_step"))
    latest_activity_at = progress.get("latest_activity_at")
    latest_activity_task_id = str(progress.get("latest_activity_task_id") or "").strip()
    latest_activity_kind = str(progress.get("latest_activity_kind") or "").strip()
    lines = ["No worker log lines; showing filesystem progress instead."]
    if total_count > 0:
        lines.append(f"DeepPHY tasks completed: {completed_count}/{total_count}")
    elif completed_count > 0:
        lines.append(f"DeepPHY tasks completed: {completed_count}")
    if latest_completed_task_id:
        lines.append(f"Latest completed task: {latest_completed_task_id}")
    if active_task_id:
        if active_step is not None:
            lines.append(f"Active task: {active_task_id} at step {active_step:03d}")
        else:
            lines.append(f"Active task: {active_task_id}")
    if isinstance(latest_activity_at, datetime):
        activity_subject = latest_activity_task_id or "unknown task"
        lines.append(
            f"Latest {latest_activity_kind} write: {activity_subject} at {_display_time(latest_activity_at)}"
        )
    return lines


def _render_repeat_learning_experiment_panel(
    *,
    job: JobRecord,
    jobs: list[JobRecord],
    project_root: Path,
) -> str:
    group_key = _repeat_run_group_key(job, project_root=project_root)
    if group_key is None:
        return ""
    env_id, family, group_prefix = group_key
    sibling_runs = [
        candidate
        for candidate in jobs
        if _repeat_run_group_key(candidate, project_root=project_root) == group_key
    ]
    if not sibling_runs:
        return ""
    eval_family = f"{family} DeepPHY"
    evals_by_run_name: dict[str, list[JobRecord]] = {}
    for candidate in jobs:
        if str(candidate.spec.action or "").strip() != "evaluate-deepphy":
            continue
        if str(candidate.spec.metadata.get("env_id") or candidate.environment or "").strip() != env_id:
            continue
        if _job_required_family(candidate) != eval_family:
            continue
        pattern_run_name = _repeat_eval_pattern_run_name(candidate, project_root=project_root)
        if not pattern_run_name:
            continue
        match = _REPEAT_RUN_SUFFIX_PATTERN.match(pattern_run_name)
        if match is None or str(match.group("prefix")) != group_prefix:
            continue
        evals_by_run_name.setdefault(pattern_run_name, []).append(candidate)

    sibling_runs = sorted(
        sibling_runs,
        key=lambda item: (
            _repeat_run_index(item, project_root=project_root) or 999,
            item.created_at,
            item.job_id,
        ),
    )
    rows: list[str] = []
    completed_eval_runs = 0
    total_eval_runs = 0
    for sibling in sibling_runs:
        run_name = _repeat_run_name(sibling, project_root=project_root) or sibling.job_id[:8]
        run_index = _repeat_run_index(sibling, project_root=project_root)
        progress = _learning_progress_summary(sibling, project_root=project_root)
        candidate_function_count = _learning_candidate_function_count(sibling, project_root=project_root)
        completed_labels = int(progress.get("completed_labels") or 0) if progress is not None else None
        total_labels = int(progress.get("total_labels") or 0) if progress is not None else None
        progress_text = (
            f"{completed_labels}/{total_labels} labels"
            if completed_labels is not None and total_labels
            else _job_status_summary(sibling, project_root=project_root) or "No label progress yet"
        )
        eval_jobs = sorted(
            [
                item
                for item in evals_by_run_name.get(run_name, [])
                if _repeat_eval_within_target(item, project_root=project_root)
            ],
            key=lambda item: (
                _repeat_eval_run_index(item, project_root=project_root) or 999,
                item.created_at,
                item.job_id,
            ),
        )
        eval_bits: list[str] = []
        eval_percents: list[float] = []
        for eval_job in eval_jobs:
            eval_index = _repeat_eval_run_index(eval_job, project_root=project_root)
            metrics = _deepphy_eval_success_metrics(eval_job, project_root=project_root)
            if metrics is not None:
                _success_count, total_tasks, percent = metrics
                eval_percents.append(percent)
                eval_bits.append(
                    f'<a class="experiment-eval-chip" href="/jobs/{html.escape(eval_job.job_id)}" title="{percent:.1f}% success over {total_tasks} tasks">'
                    f"E{eval_index or '—'} {percent:.0f}%</a>"
                )
            else:
                eval_bits.append(
                    f'<a class="experiment-eval-chip is-{_status_class(eval_job.status.value)}" href="/jobs/{html.escape(eval_job.job_id)}">'
                    f"E{eval_index or '—'} {html.escape(eval_job.status.value)}</a>"
                )
        completed_eval_runs += len(eval_percents)
        total_eval_runs += len(eval_jobs)
        avg_eval = sum(eval_percents) / len(eval_percents) if eval_percents else None
        best_eval = max(eval_percents) if eval_percents else None
        eval_results_html = "".join(eval_bits) if eval_bits else '<span class="metric-subtle">No evals yet</span>'
        rows.append(
            f"""
            <tr class="{'is-current' if sibling.job_id == job.job_id else ''}">
              <td>
                <div class="experiment-run-cell">
                  <a class="metric-title" href="/jobs/{html.escape(sibling.job_id)}">{html.escape(f'Run {run_index:02d}' if run_index is not None else run_name)}</a>
                  <p class="metric-subtle">{html.escape(run_name)}</p>
                </div>
              </td>
              <td>{_learning_status_pill(sibling, progress)}</td>
              <td>{html.escape(progress_text)}</td>
              <td>{html.escape(str(candidate_function_count if candidate_function_count is not None else "—"))}</td>
              <td>{len(eval_percents)}/{len(eval_jobs) or 0}</td>
              <td>{f'{avg_eval:.1f}%' if avg_eval is not None else '—'}</td>
              <td>{f'{best_eval:.1f}%' if best_eval is not None else '—'}</td>
              <td><div class="experiment-eval-chip-row">{eval_results_html}</div></td>
            </tr>
            """
        )
    current_run_name = _repeat_run_name(job, project_root=project_root) or job.job_id[:8]
    table_html = (
        f"""
        <div class="recent-jobs-scroll">
          <table class="data-table experiment-run-table">
            <thead>
              <tr><th>Run</th><th>Status</th><th>Label progress</th><th>Candidate functions</th><th>Completed evals</th><th>Avg eval</th><th>Best eval</th><th>Evaluation results</th></tr>
            </thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
        """
        if rows
        else '<p class="empty">No sibling runs found yet.</p>'
    )
    return f"""
    <section class="panel">
      <div class="card-topline">
        <div>
          <h2>{html.escape(family)} experiment runs</h2>
          <p class="metric-subtle">Sibling required runs for this batch, with DeepPHY evaluation performance. Open any run to inspect its full label trajectory and candidate history.</p>
        </div>
        <div class="stats-inline">
          <span class="pill">{len(sibling_runs)} runs</span>
          <span class="pill">{completed_eval_runs}/{total_eval_runs} evals complete</span>
          <span class="pill">Current {html.escape(current_run_name)}</span>
        </div>
      </div>
      {table_html}
    </section>
    """


def _required_variant_jobs(
    *,
    env_id: str,
    family: str,
    jobs: list[JobRecord],
    project_root: Path,
) -> list[JobRecord]:
    if family in REPEAT_LEARNING_FAMILIES:
        repeat_jobs = [
            job
            for job in jobs
            if _job_repeat_learning_family(job) == family
            and str(job.spec.metadata.get("env_id") or job.environment or "").strip() == env_id
            and not _job_is_superseded_combined_source(job)
        ]
        if repeat_jobs:
            return repeat_jobs
    if family.endswith("DeepPHY"):
        def _is_manual_nonrepeat_eval(job: JobRecord) -> bool:
            if str(job.spec.action or "").strip() != "evaluate-deepphy":
                return False
            if _job_required_family(job) != family:
                return False
            if str(job.spec.metadata.get("env_id") or job.environment or "").strip() != env_id:
                return False
            if job.status.value == "cancelled":
                return False
            metadata = dict(job.spec.metadata or {})
            request_payload = metadata.get("request")
            request_run_id = request_payload.get("run_id") if isinstance(request_payload, dict) else None
            eval_marker = str(
                metadata.get("evaluation_run_id")
                or request_run_id
                or Path(str(metadata.get("evaluation_run_dir") or "")).name
            ).strip()
            if _REPEAT_EVAL_SUFFIX_PATTERN.match(eval_marker):
                return False
            return True

        repeat_jobs = [
            job
            for job in jobs
            if _job_repeat_required_family(job) == family
            and _required_repeat_job_counts_toward_target(job)
            and str(job.spec.metadata.get("env_id") or job.environment or "").strip() == env_id
            and not _job_is_superseded_combined_source(job)
        ]
        manual_nonrepeat_jobs = [
            job
            for job in jobs
            if _is_manual_nonrepeat_eval(job)
            and not _job_is_superseded_combined_source(job)
        ]
        if repeat_jobs or manual_nonrepeat_jobs:
            merged: list[JobRecord] = []
            seen: set[str] = set()
            for candidate in [*repeat_jobs, *manual_nonrepeat_jobs]:
                if candidate.job_id in seen:
                    continue
                seen.add(candidate.job_id)
                merged.append(candidate)
            return merged
    return [
        job
        for job in jobs
        if _job_required_family(job) == family
        and str(job.spec.metadata.get("env_id") or job.environment or "").strip() == env_id
        and not _job_is_superseded_combined_source(job)
    ]


def _score_summary(values: list[float]) -> tuple[float, float] | None:
    if not values:
        return None
    mean = statistics.fmean(values)
    stddev = statistics.pstdev(values) if len(values) > 1 else 0.0
    return (mean, stddev)


def _variant_eval_job_label(job: JobRecord, *, project_root: Path) -> str:
    run_name = _repeat_eval_run_name(job, project_root=project_root)
    if run_name:
        return run_name
    eval_dir = _resolve_deepphy_eval_run_dir(job, project_root=project_root)
    if eval_dir is not None:
        return eval_dir.name
    return job.job_id[:8]


def render_required_variant_page(
    *,
    env_id: str,
    family: str,
    jobs: list[JobRecord],
    clients: list[ClientRecord],
    project_root: Path,
    queue_settings: QueueSettings,
) -> str:
    def _visible_required_variant_job(job: JobRecord) -> bool:
        return job.status.value != "cancelled"

    if env_id == "pooltool" and family in {"Optimization", "Value model"}:
        reward_kind = "optimization" if family == "Optimization" else "value-model"
        rewards = [
            item
            for item in list_environment_reward_results(project_root, env_id)
            if str(item.reward_kind or "").strip().lower() == reward_kind
        ]
        return render_environment_reward_runs_page(
            env_id=env_id,
            reward_kind=reward_kind,
            rewards=rewards,
        )

    family_jobs = _required_variant_jobs(
        env_id=env_id,
        family=family,
        jobs=jobs,
        project_root=project_root,
    )
    client_lookup = {client.client_id: client for client in clients}
    if family in REPEAT_LEARNING_FAMILIES:
        sorted_runs = sorted(
            family_jobs,
            key=lambda item: (
                _repeat_run_index(item, project_root=project_root) or 999,
                item.created_at,
                item.job_id,
            ),
        )
        rows: list[str] = []
        all_scores: list[float] = []
        combine_enabled = family == "Supervised"
        selection_enabled = True
        for run_job in sorted_runs:
            run_name = _repeat_run_name(run_job, project_root=project_root) or run_job.job_id[:8]
            progress = _learning_progress_summary(run_job, project_root=project_root)
            candidate_function_count = _learning_candidate_function_count(run_job, project_root=project_root)
            completed_labels = int(progress.get("completed_labels") or 0) if progress is not None else None
            total_labels = int(progress.get("total_labels") or 0) if progress is not None else None
            progress_text = (
                f"{completed_labels}/{total_labels} labels"
                if completed_labels is not None and total_labels
                else _job_status_summary(run_job, project_root=project_root) or "No label progress yet"
            )
            embedded_eval_root = _learning_run_embedded_eval_root(run_job, project_root=project_root)
            score_values: list[float] = []
            score_bits: list[str] = []
            completed_eval_count = 0
            total_eval_entries = 0
            if embedded_eval_root is not None:
                total_eval_entries += 1
                embedded_counts = _required_catalog_deepphy_eval_counts_cached(
                    str(embedded_eval_root),
                    _required_catalog_deepphy_eval_marker(embedded_eval_root),
                    (
                        int(queue_settings.pooltool_attempt_limit)
                        if env_id == "pooltool"
                        else None
                    ),
                )
                if embedded_counts is not None and embedded_counts[2] > 0:
                    embedded_success, _embedded_failures, embedded_total = embedded_counts
                    embedded_percent = max(0.0, min(100.0, (embedded_success / embedded_total) * 100.0))
                    score_values.append(embedded_percent)
                    completed_eval_count += 1
                    score_bits.append(
                        f'<a class="experiment-eval-chip" href="/jobs/{html.escape(run_job.job_id)}" title="{embedded_percent:.1f}% success over {embedded_total} tasks">'
                        f"Auto {embedded_percent:.1f}%</a>"
                    )
                else:
                    score_bits.append(
                        f'<a class="experiment-eval-chip is-{_status_class(run_job.status.value)}" href="/jobs/{html.escape(run_job.job_id)}">'
                        "Auto pending</a>"
                    )
            linked_evals = _learning_run_manual_eval_jobs(
                env_id=env_id,
                run_job=run_job,
                jobs=jobs,
                project_root=project_root,
            )
            for eval_job in linked_evals:
                total_eval_entries += 1
                metrics = _deepphy_eval_success_metrics(
                    eval_job,
                    project_root=project_root,
                    pooltool_attempt_limit=(
                        int(queue_settings.pooltool_attempt_limit)
                        if env_id == "pooltool"
                        else None
                    ),
                )
                eval_label = _variant_eval_job_label(eval_job, project_root=project_root)
                if metrics is not None:
                    _success_count, total_tasks, percent = metrics
                    score_values.append(percent)
                    completed_eval_count += 1
                    score_bits.append(
                        f'<a class="experiment-eval-chip" href="/jobs/{html.escape(eval_job.job_id)}" title="{percent:.1f}% success over {total_tasks} tasks">'
                        f"{html.escape(eval_label)} {percent:.1f}%</a>"
                    )
                else:
                    score_bits.append(
                        f'<a class="experiment-eval-chip is-{_status_class(eval_job.status.value)}" href="/jobs/{html.escape(eval_job.job_id)}">'
                        f"{html.escape(eval_label)} {html.escape(eval_job.status.value)}</a>"
                    )
            all_scores.extend(score_values)
            score_summary = _score_summary(score_values)
            mean_text = f"{score_summary[0]:.1f}%" if score_summary is not None else "—"
            std_text = f"{score_summary[1]:.1f}%" if score_summary is not None else "—"
            best_text = f"{max(score_values):.1f}%" if score_values else "—"
            eval_scores_html = "".join(score_bits) if score_bits else '<span class="metric-subtle">No evals yet</span>'
            selection_cell = (
                f'<td><input type="checkbox" name="job_ids" value="{html.escape(run_job.job_id)}"></td>'
                if selection_enabled
                else ""
            )
            rows.append(
                "<tr>"
                f"{selection_cell}"
                f'<td><div class="experiment-run-cell"><a class="metric-title" href="/jobs/{html.escape(run_job.job_id)}">{html.escape(run_name)}</a><p class="metric-subtle">{html.escape(_assigned_client_name(run_job, client_lookup))}</p></div></td>'
                f"<td>{_learning_status_pill(run_job, progress)}</td>"
                f"<td>{html.escape(progress_text)}</td>"
                f"<td>{html.escape(str(candidate_function_count if candidate_function_count is not None else '—'))}</td>"
                f"<td>{completed_eval_count}/{total_eval_entries}</td>"
                f"<td>{mean_text}</td>"
                f"<td>{std_text}</td>"
                f"<td>{best_text}</td>"
                f'<td><div class="experiment-eval-chip-row">{eval_scores_html}</div></td>'
                "</tr>"
            )
        overall = _score_summary(all_scores)
        selection_header = "<th></th>" if selection_enabled else ""
        selection_colspan = 10 if selection_enabled else 9
        controls_panel_title = "Manage runs"
        controls_panel_detail = (
            "Pause and hide selected runs from this page. Running runs are paused first."
            if not combine_enabled
            else (
                "Delete hides selected runs from this page and pauses them first. "
                "Combine merges selected supervised runs into one run and hides sources."
            )
        )
        combine_button = ""
        if combine_enabled:
            combine_button = (
                f'<button type="submit" formaction="/ui/required-runs/{html.escape(env_id)}/{html.escape(required_family_slug(family))}/combine">'
                "Combine selected"
                "</button>"
            )
        evaluate_button = (
            f'<button type="submit" formaction="/ui/required-runs/{html.escape(env_id)}/{html.escape(required_family_slug(family))}/evaluate">'
            "Run evaluation"
            "</button>"
        )
        controls_panel = f"""
        <section class="panel">
          <div class="card-topline">
            <div>
              <h2>{controls_panel_title}</h2>
              <p class="metric-subtle">{html.escape(controls_panel_detail)} Select one run to queue a manual eval; embedded auto-evals are used by default.</p>
            </div>
          </div>
          <div class="control-row">
            {evaluate_button}
            <button type="submit">Delete selected</button>
            {combine_button}
          </div>
        </section>
        """
        body = f"""
        <section class="hero">
      <p class="breadcrumb"><a href="/">Home</a> / required-runs / {html.escape(env_id)} / {html.escape(family)}</p>
      <h1>{html.escape(env_id.upper())} {html.escape(family)}</h1>
      <p>All required repeat runs for this learning variant, with embedded auto-eval results and any manual reruns.</p>
          <div class="hero-stats">
            <div class="stat-chip"><span class="stat-label">Runs</span><strong class="stat-value">{len(sorted_runs)}</strong></div>
            <div class="stat-chip"><span class="stat-label">Scored evals</span><strong class="stat-value">{len(all_scores)}</strong></div>
            <div class="stat-chip"><span class="stat-label">Average</span><strong class="stat-value">{f"{overall[0]:.1f}%" if overall is not None else "—"}</strong></div>
            <div class="stat-chip"><span class="stat-label">Std dev</span><strong class="stat-value">{f"{overall[1]:.1f}%" if overall is not None else "—"}</strong></div>
          </div>
        </section>
        <form method="post" action="/ui/required-runs/{html.escape(env_id)}/{html.escape(required_family_slug(family))}/delete">
          <input type="hidden" name="next" value="/required-runs/{html.escape(env_id)}/{html.escape(required_family_slug(family))}">
          {controls_panel}
        <section class="panel">
          <div class="card-topline">
            <div>
              <h2>Variant overview</h2>
              <p class="metric-subtle">Each row is one required learning run. Scores summarize the run's embedded auto-eval and any manual reruns.</p>
            </div>
          </div>
          <div class="recent-jobs-scroll">
            <table class="data-table experiment-run-table">
              <thead><tr>{selection_header}<th>Run</th><th>Status</th><th>Label progress</th><th>Candidate functions</th><th>Scored evals</th><th>Average</th><th>Std dev</th><th>Best</th><th>Evaluation scores</th></tr></thead>
              <tbody>{''.join(rows) if rows else f'<tr><td colspan="{selection_colspan}">No runs found.</td></tr>'}</tbody>
            </table>
          </div>
        </section>
        </form>
        """
        return _shell(f"{env_id.upper()} {family} | Pattern Server", active_nav=f"env:{env_id}", body_html=body)

    sorted_jobs = sorted(
        [item for item in family_jobs if _visible_required_variant_job(item)],
        key=lambda item: (
            _repeat_eval_run_index(item, project_root=project_root) or _repeat_run_index(item, project_root=project_root) or 999,
            item.updated_at,
            item.created_at,
            item.job_id,
        ),
    )
    rows = []
    score_values: list[float] = []
    for item in sorted_jobs:
        metrics = _deepphy_eval_success_metrics(
            item,
            project_root=project_root,
            pooltool_attempt_limit=(
                int(queue_settings.pooltool_attempt_limit)
                if env_id == "pooltool"
                else None
            ),
        )
        label = _variant_eval_job_label(item, project_root=project_root)
        score_text = "—"
        task_text = "—"
        if metrics is not None:
            success_count, total_tasks, percent = metrics
            score_values.append(percent)
            score_text = f"{percent:.1f}%"
            task_text = f"{success_count}/{total_tasks}"
        rows.append(
            "<tr>"
            f'<td><div class="experiment-run-cell"><a class="metric-title" href="/jobs/{html.escape(item.job_id)}">{html.escape(label)}</a><p class="metric-subtle">{html.escape(_assigned_client_name(item, client_lookup))}</p></div></td>'
            f"<td>{_pill(item.status.value, item.status.value)}</td>"
            f"<td>{task_text}</td>"
            f"<td>{score_text}</td>"
            f"<td>{html.escape(_job_last_activity_summary(item))}</td>"
            "</tr>"
        )
    overall = _score_summary(score_values)
    pooltool_attempt_limit_panel = ""
    if env_id == "pooltool":
        pooltool_attempt_limit_panel = _render_pooltool_attempt_limit_panel(
            queue_settings=queue_settings,
            next_url=f"/required-runs/{env_id}/{quote(REQUIRED_FAMILY_SLUGS.get(family, family.lower()))}",
            detail_text=(
                "Recalculate PoolTool evaluation scores from stored task results. "
                "Tasks over the selected attempt limit count as task-level failures without rerunning the eval."
            ),
        )
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Home</a> / required-runs / {html.escape(env_id)} / {html.escape(family)}</p>
      <h1>{html.escape(env_id.upper())} {html.escape(family)}</h1>
      <p>All required runs recorded for this evaluation variant, with per-run performance scores.</p>
      <div class="hero-stats">
        <div class="stat-chip"><span class="stat-label">Runs</span><strong class="stat-value">{len(sorted_jobs)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Scored</span><strong class="stat-value">{len(score_values)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Average</span><strong class="stat-value">{f"{overall[0]:.1f}%" if overall is not None else "—"}</strong></div>
        <div class="stat-chip"><span class="stat-label">Std dev</span><strong class="stat-value">{f"{overall[1]:.1f}%" if overall is not None else "—"}</strong></div>
      </div>
    </section>
    {pooltool_attempt_limit_panel}
    <section class="panel">
      <div class="card-topline">
        <div>
          <h2>Variant overview</h2>
          <p class="metric-subtle">Each row is one required evaluation run. The score column shows task success rate for that run.</p>
        </div>
      </div>
      <div class="recent-jobs-scroll">
        <table class="data-table experiment-run-table">
          <thead><tr><th>Run</th><th>Status</th><th>Tasks solved</th><th>Score</th><th>Last activity</th></tr></thead>
          <tbody>{''.join(rows) if rows else '<tr><td colspan="5">No runs found.</td></tr>'}</tbody>
        </table>
      </div>
    </section>
    """
    return _shell(f"{env_id.upper()} {family} | Pattern Server", active_nav=f"env:{env_id}", body_html=body)


def _assigned_client_name(job: JobRecord, client_lookup: dict[str, ClientRecord]) -> str:
    assigned = str(job.assigned_client_id or "").strip()
    if not assigned:
        return "—"
    client = client_lookup.get(assigned)
    if client is not None:
        return client.display_name
    return assigned


def _job_is_controllable(job: JobRecord) -> bool:
    metadata = dict(job.spec.metadata or {})
    if "controllable" in metadata:
        return bool(metadata.get("controllable"))
    return bool(job.spec.argv)


def _shell(page_title: str, *, active_nav: str, body_html: str) -> str:
    primary_nav_links = [
        ("clients", "/", "Home"),
        ("learning", "/learning", "Learning"),
        ("statistics", "/statistics", "Statistics"),
        ("profiler", "/profiler", "Profiler"),
        ("analysis", "/analysis", "Analysis"),
        ("ablations", "/ablations", "Ablations"),
        ("rewards", "/rewards", "Rewards"),
        ("library", "/library", "Library"),
        ("videos", "/videos", "Videos"),
        ("summaries", "/summaries", "Summaries"),
    ]
    env_nav_links = [(f"env:{env_id}", f"/env/{env_id}", env_id.upper()) for env_id in SUPPORTED_ENVS]
    primary_nav_html = "".join(
        f'<a class="header-nav-link{" is-active" if key == active_nav else ""}" href="{href}">{html.escape(label)}</a>'
        for key, href, label in primary_nav_links
    )
    env_nav_html = "".join(
        f'<a class="header-nav-link header-env-link{" is-active" if key == active_nav else ""}" href="{href}">{html.escape(label)}</a>'
        for key, href, label in env_nav_links
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{html.escape(page_title)}</title>
    <style>
      :root {{
        --bg: #f3efe8;
        --paper: #fbf8f3;
        --paper-strong: #fffdf9;
        --ink: #171512;
        --muted: #645d54;
        --line: #d8cfc2;
        --line-strong: #b4a694;
        --accent: #124f49;
        --accent-strong: #0d3d38;
        --accent-soft: #e1efeb;
        --warn: #8a531f;
        --warn-soft: #f6e9dc;
        --danger: #8b3426;
        --danger-soft: #f6e3de;
        --shadow: 0 14px 34px rgba(30, 23, 16, 0.05);
        --shadow-tight: 0 6px 18px rgba(30, 23, 16, 0.04);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        background:
          linear-gradient(180deg, rgba(18, 79, 73, 0.06), transparent 18rem),
          linear-gradient(180deg, #f5f1e9 0%, var(--bg) 100%);
        color: var(--ink);
        font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
        line-height: 1.42;
      }}
      a {{ color: var(--accent); text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}
      code, pre {{ font-family: "IBM Plex Mono", "SFMono-Regular", monospace; }}
      .page-shell {{ width: min(1560px, calc(100vw - 1.4rem)); margin: 0 auto; padding: 0.55rem 0 1.2rem; }}
      .page-header {{ margin-bottom: 0.75rem; padding: 0.72rem 0.88rem; background: rgba(255, 253, 249, 0.96); border: 1px solid var(--line); border-radius: 2px; box-shadow: none; }}
      .header-inner {{ display: grid; gap: 0.55rem; }}
      .header-top-row {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 0.9rem; }}
      .eyebrow {{ display: none; }}
      .home-link {{ color: var(--ink); font-size: 1.22rem; font-weight: 780; letter-spacing: -0.03em; }}
      .tagline {{ display: none; }}
      .header-nav {{ display: flex; gap: 0.35rem; flex-wrap: wrap; }}
      .header-nav-primary {{ justify-content: flex-end; }}
      .header-nav-env {{ justify-content: flex-start; padding-top: 0.18rem; border-top: 1px solid rgba(216, 207, 194, 0.82); }}
      .header-nav-link {{ display: inline-flex; align-items: center; min-height: 1.8rem; padding: 0.16rem 0.52rem; border: 1px solid var(--line); border-radius: 2px; background: rgba(255, 255, 255, 0.72); color: var(--accent-strong); font-size: 0.78rem; font-weight: 650; }}
      .header-nav-link:hover {{ text-decoration: none; background: rgba(18, 79, 73, 0.06); }}
      .header-nav-link.is-active {{ background: rgba(18, 79, 73, 0.1); border-color: rgba(18, 79, 73, 0.22); }}
      .header-env-link {{ min-width: 4.7rem; justify-content: center; }}
      .hero {{ margin-bottom: 0.55rem; }}
      .hero h1 {{ margin: 0 0 0.18rem; font-size: clamp(1.72rem, 2.2vw, 2.45rem); line-height: 0.98; letter-spacing: -0.035em; overflow-wrap: anywhere; word-break: break-word; }}
      .hero p {{ margin: 0; max-width: 58rem; color: var(--muted); font-size: 0.84rem; overflow-wrap: anywhere; word-break: break-word; }}
      .breadcrumb {{ margin-bottom: 0.18rem; color: var(--muted); font-size: 0.66rem; letter-spacing: 0.06em; text-transform: uppercase; overflow-wrap: anywhere; word-break: break-word; }}
      .hero-stats, .summary-grid, .content-grid, .button-row, .chip-row, .metric-stack {{ display: grid; gap: 0.45rem; }}
      .hero-stats {{ grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); }}
      .summary-grid {{ grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); margin-top: 0.85rem; }}
      .content-grid {{ grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }}
      .videos-summary-grid {{ display: grid; gap: 0.52rem; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); margin-bottom: 0.72rem; }}
      .videos-render-grid {{ display: grid; gap: 0.52rem; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); }}
      .videos-render-card {{ display: grid; gap: 0.5rem; align-content: start; padding: 0.62rem 0.68rem; }}
      .videos-render-head {{ display: flex; justify-content: space-between; gap: 0.4rem; align-items: start; }}
      .videos-render-head strong {{ font-size: 0.82rem; line-height: 1.05; }}
      .videos-render-copy {{ min-width: 0; display: grid; gap: 0.18rem; }}
      .videos-render-copy strong {{ font-size: 0.9rem; line-height: 1.05; }}
      .videos-render-copy .metric-subtle {{ font-size: 0.72rem; }}
      .videos-render-stats {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.35rem; }}
      .videos-render-stat {{ padding: 0.42rem 0.46rem; border: 1px solid var(--line); background: rgba(255,255,255,0.74); }}
      .videos-render-stat strong {{ display: block; font-size: 0.92rem; line-height: 1.05; }}
      .videos-render-stat span {{ display: block; margin-top: 0.12rem; color: var(--muted); font-size: 0.66rem; font-weight: 760; letter-spacing: 0.06em; text-transform: uppercase; }}
      .videos-progress {{ height: 0.5rem; border: 1px solid var(--line); background: rgba(255, 255, 255, 0.78); overflow: hidden; }}
      .videos-progress-fill {{ display: block; height: 100%; background: linear-gradient(90deg, rgba(18,79,73,0.9), rgba(198,127,33,0.78)); }}
      .videos-render-detail {{ font-size: 0.73rem; color: var(--muted); min-height: 1.1rem; }}
      .videos-render-card .client-settings-form {{ margin: 0; padding: 0; border: 0; background: transparent; }}
      .videos-render-card .client-settings-form {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 0.34rem; align-items: center; }}
      .videos-render-target {{ display: grid; gap: 0.12rem; min-width: 0; }}
      .videos-render-target span {{ font-size: 0.64rem; font-weight: 760; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
      .videos-render-target input {{ width: 100%; min-height: 1.65rem; padding: 0.18rem 0.38rem; font-size: 0.74rem; }}
      .videos-render-card .secondary-button {{ width: 100%; justify-content: center; min-height: 1.65rem; padding: 0.16rem 0.5rem; font-size: 0.72rem; }}
      .videos-overview-card {{ display: grid; gap: 0.3rem; padding: 0.58rem 0.64rem; }}
      .videos-overview-card strong {{ font-size: 1.05rem; line-height: 1; }}
      .videos-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.72rem; }}
      .videos-cell {{ display: grid; gap: 0.55rem; min-height: 0; }}
      .videos-task-list {{ max-height: 68vh; overflow-y: auto; border: 1px solid var(--line); background: rgba(255, 255, 255, 0.75); }}
      .video-task-row {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 0.55rem; align-items: center; padding: 0.48rem 0.56rem; border-bottom: 1px solid var(--line); }}
      .video-task-row:last-child {{ border-bottom: 0; }}
      .video-task-row-main {{ min-width: 0; display: grid; gap: 0.14rem; }}
      .video-task-row-main strong {{ font-size: 0.82rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
      .video-task-actions {{ display: inline-flex; gap: 0.36rem; align-items: center; flex-wrap: wrap; justify-content: flex-end; }}
      .summary-example-card {{ display: grid; grid-template-columns: 7rem minmax(0, 1fr); gap: 0.75rem; padding: 0.68rem; border-bottom: 1px solid var(--line); align-items: start; }}
      .summary-example-card:last-child {{ border-bottom: 0; }}
      .summary-example-thumb {{ display: block; width: 100%; aspect-ratio: 16 / 10; object-fit: contain; border: 1px solid rgba(203, 191, 174, 0.9); background: rgba(244, 240, 232, 0.85); }}
      .summary-example-thumb-empty {{ display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: 0.72rem; }}
      .summary-example-main {{ min-width: 0; display: grid; gap: 0.38rem; }}
      .summary-example-top {{ display: flex; align-items: start; justify-content: space-between; gap: 0.6rem; }}
      .summary-example-title-stack {{ min-width: 0; display: grid; gap: 0.14rem; }}
      .summary-example-title-stack strong {{ font-size: 0.9rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
      .summary-example-chips {{ display: flex; flex-wrap: wrap; gap: 0.3rem; }}
      .summary-example-snippet {{ margin: 0; line-height: 1.4; }}
      .video-modal-overlay {{ position: fixed; inset: 0; background: rgba(16, 13, 10, 0.58); display: none; align-items: center; justify-content: center; padding: 1rem; z-index: 1800; }}
      .video-modal-overlay.is-open {{ display: flex; }}
      .video-modal-card {{ width: auto; max-width: min(98vw, 1500px); max-height: 96vh; background: var(--paper-strong); border: 1px solid var(--line-strong); position: relative; padding: 0.5rem 0.5rem 0.42rem; box-shadow: 0 22px 58px rgba(0, 0, 0, 0.28); display: grid; gap: 0.5rem; }}
      .video-modal-controls {{ display: flex; align-items: center; gap: 0.75rem; padding-right: 2.2rem; min-width: 0; max-width: min(92vw, 720px); flex-wrap: wrap; }}
      .video-modal-controls label {{ display: inline-flex; align-items: center; gap: 0.5rem; font-size: 0.78rem; font-weight: 760; color: var(--ink); flex: 1 1 18rem; min-width: 0; }}
      .video-modal-controls input[type="range"] {{ flex: 1 1 auto; min-width: 180px; }}
      .video-modal-confidence-value {{ min-width: 3.25rem; text-align: right; font: 0.75rem/1.2 "IBM Plex Mono", "SFMono-Regular", Consolas, monospace; color: var(--muted); }}
      .video-modal-frame {{ width: min(100%, calc(98vw - 1rem)); max-width: calc(98vw - 1rem); max-height: calc(96vh - 6.25rem); background: transparent; overflow: hidden; display: flex; justify-content: center; align-items: center; min-width: 0; }}
      .video-modal-frame.is-rotated {{ width: min(100%, calc(98vw - 1rem)); max-width: calc(98vw - 1rem); max-height: calc(96vh - 6.25rem); overflow: hidden; }}
      .video-modal-stage {{ position: relative; display: inline-block; margin: 0 auto; max-width: 100%; background: #0d0d0d; }}
      .video-modal-frame video, .video-modal-frame img {{ display: block; width: min(100%, calc(98vw - 1rem)); max-width: 100%; height: auto; max-height: calc(96vh - 6.25rem); margin: 0 auto; object-fit: contain; transform-origin: center center; }}
      .video-modal-frame.is-rotated video, .video-modal-frame.is-rotated img {{ width: auto; max-width: none; height: auto; max-height: none; }}
      .video-modal-frame canvas {{ position: absolute; inset: 0; display: block; width: 100%; height: 100%; pointer-events: none; }}
      .video-modal-frame img {{ object-fit: contain; background: #0d0d0d; }}
      .video-modal-download {{ position: absolute; top: 0.32rem; right: 0.38rem; width: 1.85rem; height: 1.85rem; border: 1px solid var(--line); background: rgba(255, 255, 255, 0.93); color: var(--accent-strong); font-size: 1rem; display: inline-flex; align-items: center; justify-content: center; }}
      .video-modal-download:hover {{ text-decoration: none; background: rgba(18, 79, 73, 0.08); }}
      .text-modal-card {{ width: min(76vw, 860px); max-height: 90vh; background: var(--paper-strong); border: 1px solid var(--line-strong); position: relative; padding: 0.88rem 0.92rem 0.82rem; box-shadow: 0 22px 58px rgba(0, 0, 0, 0.28); display: grid; gap: 0.68rem; }}
      .text-modal-head {{ display: flex; align-items: center; justify-content: space-between; gap: 0.8rem; }}
      .text-modal-head h3 {{ margin: 0; font-size: 0.96rem; }}
      .text-modal-close {{ min-width: 1.95rem; min-height: 1.95rem; padding: 0; }}
      .text-modal-body {{ margin: 0; max-height: 70vh; overflow: auto; border: 1px solid var(--line); background: rgba(255, 255, 255, 0.82); padding: 0.72rem 0.78rem; white-space: pre-wrap; font: 0.8rem/1.45 "IBM Plex Mono", "SFMono-Regular", Consolas, monospace; color: var(--ink); }}
      @media (max-width: 1100px) {{
        .videos-grid {{ grid-template-columns: 1fr; }}
      }}
      @media (max-width: 760px) {{
        .videos-render-grid {{ grid-template-columns: 1fr; }}
      }}
      .button-row {{ grid-template-columns: repeat(auto-fit, minmax(120px, max-content)); align-items: start; }}
      .page-main, .home-main-column, .section-stack {{ display: grid; gap: 0.95rem; min-width: 0; }}
      .home-ops-grid {{ display: grid; gap: 0.9rem; align-items: start; grid-template-columns: 1fr; }}
      .home-side-stack {{ display: grid; gap: 0.9rem; grid-template-columns: 1fr; }}
      .home-top-grid {{ display: grid; gap: 0.45rem; grid-template-columns: minmax(0, 1.45fr) minmax(18rem, 0.55fr); align-items: stretch; }}
      .home-top-grid > .panel {{ height: 100%; }}
      .panel, .card {{ padding: 0.7rem 0.76rem; background: rgba(255, 254, 251, 0.96); border: 1px solid var(--line); border-radius: 0; box-shadow: none; }}
      .panel.panel-static {{ padding: 0.55rem 0.62rem; }}
      .compact-panel {{ padding-block: 0.8rem 0.86rem; }}
      .panel.is-collapsible {{ padding: 0; overflow: clip; }}
      .panel-toggle {{ width: 100%; border: 0; border-bottom: 1px solid rgba(203, 191, 174, 0.9); background:
        linear-gradient(180deg, rgba(255,255,255,0.88), rgba(249, 245, 239, 0.98));
        padding: 0.72rem 0.88rem 0.74rem; display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 0.75rem; align-items: center; text-align: left; color: inherit; cursor: pointer; }}
      .panel-toggle:hover {{ background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(246, 241, 234, 1)); }}
      .panel-toggle-copy {{ display: grid; gap: 0.18rem; min-width: 0; }}
      .panel-toggle-kicker {{ color: var(--accent-strong); font-size: 0.68rem; font-weight: 800; letter-spacing: 0.11em; text-transform: uppercase; }}
      .panel-toggle-title {{ font-size: 0.94rem; font-weight: 760; letter-spacing: -0.02em; }}
      .panel-toggle-summary {{ color: var(--muted); font-size: 0.79rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
      .panel-toggle-meta {{ display: flex; align-items: center; gap: 0.45rem; flex-wrap: wrap; justify-content: flex-end; }}
      .panel-toggle-chip {{ display: inline-flex; align-items: center; min-height: 1.42rem; padding: 0.12rem 0.4rem; border: 1px solid rgba(18,79,73,0.12); background: rgba(18,79,73,0.08); color: var(--accent-strong); font-size: 0.7rem; font-weight: 700; border-radius: 0; }}
      .panel-toggle-caret {{ width: 0.72rem; height: 0.72rem; border-right: 2px solid var(--accent-strong); border-bottom: 2px solid var(--accent-strong); transform: rotate(45deg); transition: transform 140ms ease; margin-right: 0.12rem; }}
      .panel.is-collapsed .panel-toggle {{ border-bottom-color: transparent; }}
      .panel.is-collapsed .panel-toggle-caret {{ transform: rotate(-45deg) translateY(-0.05rem); }}
      .panel-body {{ padding: 0.95rem 1rem; display: grid; gap: 0.95rem; }}
      .panel.compact-panel .panel-body {{ padding-block: 0.85rem 0.92rem; }}
      .panel.is-collapsed .panel-body {{ display: none; }}
      .panel.is-collapsible > .panel-body > .panel-lede,
      .panel.is-collapsible > .panel-body > .panel-lede-subtle {{ display: none; }}
      .card-topline, .metric-row {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 0.7rem; }}
      .panel h2, .panel h3, .card h2 {{ margin: 0; }}
      .panel h2 {{ font-size: 0.98rem; letter-spacing: -0.015em; }}
      .panel h3, .metric-title {{ font-size: 0.92rem; font-weight: 700; letter-spacing: -0.01em; }}
      .section-kicker, .section-label, .label {{ color: var(--muted); font-size: 0.66rem; font-weight: 760; letter-spacing: 0.08em; text-transform: uppercase; }}
      .metric-subtle, .empty, .meta-text {{ color: var(--muted); font-size: 0.84rem; }}
      .asset-bullets {{ margin: 0; padding-left: 1.1rem; color: var(--ink); }}
      .asset-bullets li {{ margin: 0.2rem 0; }}
      .metric-stack {{ margin-top: 0.2rem; }}
      .metric-row-block {{ padding: 0.48rem 0.54rem; border: 1px solid rgba(216, 207, 194, 0.95); border-radius: 0; background: rgba(255, 255, 255, 0.68); }}
      .pill {{ display: inline-flex; align-items: center; justify-content: center; padding: 0.14rem 0.42rem; font-size: 0.68rem; font-weight: 760; letter-spacing: 0.03em; white-space: nowrap; border: 1px solid transparent; border-radius: 2px; background: var(--accent-soft); color: var(--accent-strong); border-color: rgba(18, 79, 73, 0.12); }}
      .status-running, .status-succeeded {{ background: var(--accent-soft); color: var(--accent-strong); }}
      .status-paused {{ background: var(--warn-soft); color: var(--warn); border-color: rgba(138, 83, 31, 0.14); }}
      .status-failed {{ background: var(--danger-soft); color: var(--danger); border-color: rgba(139, 52, 38, 0.14); }}
      .status-queued {{ background: rgba(87, 104, 125, 0.12); color: #46576a; border-color: rgba(87, 104, 125, 0.14); }}
      .stat-chip {{ padding: 0.42rem 0.5rem; border: 1px solid var(--line); border-radius: 0; background: rgba(255, 255, 255, 0.82); }}
      .stat-label {{ display: block; color: var(--muted); font-size: 0.67rem; font-weight: 760; letter-spacing: 0.08em; text-transform: uppercase; }}
      .stat-value {{ display: block; margin-top: 0.12rem; font-size: 0.98rem; }}
      .stats-inline {{ display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; }}
      .mini-chip {{ display: inline-flex; padding: 0.12rem 0.36rem; font-size: 0.68rem; font-weight: 720; border-radius: 2px; background: rgba(87, 104, 125, 0.08); color: #46576a; border: 1px solid rgba(87, 104, 125, 0.12); }}
      .inline-swatch {{ width: 0.7rem; height: 0.7rem; margin-right: 0.35rem; display: inline-block; border-radius: 0; }}
      .chart-shell {{ display: grid; gap: 0.5rem; margin-top: 1rem; }}
      .chart-shell-head {{ display: flex; justify-content: space-between; gap: 0.75rem; align-items: start; flex-wrap: wrap; }}
      .chart-shell-head h3 {{ margin: 0; }}
      .detail-chart {{ width: 100%; height: auto; display: block; border: 1px solid var(--line); background: rgba(255,255,255,0.45); }}
      .primary-button, .secondary-button, .ghost-link, button {{ display: inline-flex; align-items: center; justify-content: center; min-height: 1.8rem; padding: 0.24rem 0.58rem; border: 1px solid var(--line); border-radius: 2px; background: rgba(18, 79, 73, 0.06); color: var(--accent-strong); font-size: 0.78rem; font-weight: 700; text-decoration: none; cursor: pointer; }}
      .primary-button {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
      .secondary-button {{ background: rgba(18, 79, 73, 0.08); }}
      .danger-button {{ background: var(--danger-soft); color: var(--danger); border-color: rgba(139, 52, 38, 0.16); }}
      .ghost-link {{ background: transparent; }}
      .button-row form, .queue-item-action, .job-inline-actions form, .inline-form {{ display: inline-flex; }}
      .queue-panel {{ padding-top: 0.95rem; max-height: 36rem; overflow-y: auto; padding-right: 0.25rem; }}
      .queue-list {{ display: grid; gap: 0.65rem; margin-top: 0.7rem; }}
      .queue-section + .queue-section {{ margin-top: 1rem; }}
      .queue-section-header {{ display: grid; gap: 0.15rem; }}
      .queue-section-header h3 {{ margin: 0; font-size: 0.95rem; }}
      .queue-item {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 0.65rem; align-items: start; padding: 0.55rem 0.65rem; border: 1px solid var(--line); background: rgba(255, 255, 255, 0.45); }}
      .queue-item.is-head {{ border-color: rgba(44, 98, 194, 0.28); background: rgba(236, 242, 255, 0.48); }}
      .queue-item-title-row {{ display: flex; align-items: center; gap: 0.45rem; flex-wrap: wrap; }}
      .queue-item-main {{ display: grid; gap: 0.38rem; min-width: 0; }}
      .queue-item-main p {{ margin: 0; }}
      .queue-item-meta {{ display: flex; flex-wrap: wrap; justify-content: flex-end; align-items: center; align-content: start; gap: 0.3rem; min-width: 0; }}
      .queue-worker-form {{ display: inline-flex; align-items: center; gap: 0.3rem; padding: 0.16rem 0.22rem; border: 1px solid rgba(203, 191, 174, 0.9); background: rgba(255,255,255,0.74); }}
      .queue-worker-label {{ color: var(--muted); font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }}
      .queue-worker-form input[type="number"] {{ width: 3.2rem; padding: 0.2rem 0.28rem; min-height: 1.65rem; }}
      .queue-worker-form button {{ min-height: 1.65rem; padding: 0.18rem 0.45rem; }}
      .queue-item-main .progress-track {{ width: 100%; }}
      .queue-detail-strip {{ display: grid; gap: 0.32rem; }}
      .queue-signal-row {{ display: flex; flex-wrap: wrap; gap: 0.3rem; align-items: center; }}
      .queue-signal-chip {{ display: inline-flex; align-items: center; gap: 0.28rem; padding: 0.14rem 0.42rem; border: 1px solid rgba(203, 191, 174, 0.88); background: rgba(255,255,255,0.75); color: var(--ink); font-size: 0.73rem; font-weight: 600; line-height: 1.2; }}
      .queue-signal-chip-label {{ color: var(--muted); font-size: 0.63rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; }}
      .queue-signal-chip.is-alert {{ border-color: rgba(138, 83, 31, 0.18); background: rgba(255,244,225,0.76); color: var(--warn); }}
      .queue-signal-chip.is-strong {{ border-color: rgba(18, 79, 73, 0.18); background: rgba(234,247,245,0.86); color: var(--accent-strong); }}
      .queue-summary-row {{ display: flex; align-items: center; gap: 0.45rem; min-width: 0; }}
      .queue-summary-text {{ color: var(--muted); font-size: 0.78rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
      .queue-progress-mini {{ display: grid; gap: 0.22rem; min-width: 0; }}
      .queue-progress-mini .progress-track {{ width: min(16rem, 100%); height: 0.3rem; }}
      .queue-progress-mini-meta {{ display: flex; flex-wrap: wrap; gap: 0.35rem; align-items: center; color: var(--muted); font-size: 0.74rem; }}
      .queue-status-dot {{ width: 0.5rem; height: 0.5rem; border-radius: 0; display: inline-block; background: rgba(87,104,125,0.4); }}
      .queue-status-dot.status-running, .queue-status-dot.status-succeeded {{ background: rgba(18,79,73,0.92); }}
      .queue-status-dot.status-paused {{ background: rgba(138,83,31,0.92); }}
      .queue-status-dot.status-failed {{ background: rgba(139,52,38,0.92); }}
      .queue-status-dot.status-queued {{ background: rgba(87,104,125,0.65); }}
      .queue-controls {{ display: grid; gap: 0.35rem; margin-top: 0.8rem; }}
      .queue-settings-form {{ display: flex; flex-wrap: wrap; align-items: end; gap: 0.6rem; }}
      .queue-settings-form label {{ display: grid; gap: 0.22rem; }}
      .queue-settings-form input[type="number"], .queue-settings-form select, .inline-form select, .inline-form input[type="text"], input[type="number"], input[type="text"], textarea, select {{ width: 100%; padding: 0.3rem 0.42rem; border: 1px solid var(--line); background: #fff; font: inherit; color: inherit; border-radius: 0; }}
      .queue-settings-form input[type="number"] {{ width: 6rem; }}
      .inline-check {{ display: inline-flex; gap: 0.45rem; align-items: center; }}
      .inline-form {{ gap: 0.4rem; flex-wrap: wrap; }}
      .funnel-row {{ display: flex; align-items: center; justify-content: space-between; gap: 0.55rem; flex-wrap: wrap; padding: 0.28rem 0.32rem; min-height: 0; }}
      .funnel-summary-main {{ display: flex; align-items: center; gap: 0.65rem; flex-wrap: wrap; min-width: 0; }}
      .funnel-summary-side {{ display: flex; align-items: center; gap: 0.55rem; flex-wrap: wrap; justify-content: flex-end; }}
      .home-public-panel {{ display: grid; gap: 0.5rem; align-content: start; }}
      .home-public-head {{ display: flex; justify-content: space-between; gap: 0.5rem; align-items: start; flex-wrap: wrap; padding-bottom: 0.24rem; border-bottom: 1px solid rgba(216, 207, 194, 0.72); }}
      .home-public-url {{ overflow-wrap: anywhere; font-size: 0.8rem; }}
      .home-public-actions {{ display: flex; gap: 0.35rem; flex-wrap: wrap; align-items: center; }}
      .home-nav-panel {{ display: grid; gap: 0.42rem; align-content: start; }}
      .summary-action-form {{ display: inline-flex; margin: 0; }}
      .summary-state-button {{ min-height: 1.9rem; }}
      .home-link-strip {{ display: flex; gap: 0.3rem; flex-wrap: wrap; align-items: center; justify-content: flex-start; }}
      .home-link-pill {{ display: inline-flex; align-items: center; min-height: 1.68rem; padding: 0.15rem 0.46rem; border: 1px solid rgba(203, 191, 174, 0.82); border-radius: 2px; background: rgba(255,255,255,0.72); color: var(--accent-strong); font-size: 0.76rem; font-weight: 720; }}
      .home-link-pill:hover {{ background: rgba(18, 79, 73, 0.08); text-decoration: none; }}
      .home-link-pill-label {{ display: none; }}
      .client-panel-header {{ display: flex; align-items: center; justify-content: space-between; gap: 0.55rem; flex-wrap: wrap; padding-bottom: 0.12rem; border-bottom: 1px solid rgba(216, 207, 194, 0.72); }}
      .client-panel-summary {{ display: grid; gap: 0.16rem; min-width: 0; }}
      .client-filter-group {{ display: flex; gap: 0.45rem; flex-wrap: wrap; }}
      .filter-button {{ display: inline-flex; align-items: center; justify-content: center; min-height: 1.68rem; padding: 0.16rem 0.48rem; border: 1px solid var(--line); border-radius: 2px; background: rgba(18, 79, 73, 0.04); color: var(--accent-strong); font-size: 0.74rem; font-weight: 700; cursor: pointer; }}
      .filter-button.is-active {{ background: rgba(18, 79, 73, 0.12); border-color: rgba(18, 79, 73, 0.24); }}
      .client-list {{ display: grid; gap: 0.55rem; grid-template-columns: repeat(auto-fit, minmax(22rem, 1fr)); }}
      .client-card {{ display: grid; gap: 0.54rem; padding: 0.58rem; border: 1px solid rgba(216, 207, 194, 0.88); background: rgba(255,255,255,0.94); box-shadow: none; position: relative; overflow: hidden; border-radius: 0; }}
      .client-card::before {{ content: ""; position: absolute; inset: 0 0 auto 0; height: 0.12rem; background: rgba(18, 79, 73, 0.82); }}
      .client-card[data-client-status="offline"]::before {{ background: rgba(139, 52, 38, 0.78); }}
      .client-card[data-client-status="online"][data-client-kind="direct"]::before {{ background: rgba(53, 92, 125, 0.82); }}
      .client-card[data-client-status="online"][data-client-full="true"]::before {{ background: rgba(138, 83, 31, 0.82); }}
      .client-card h3 {{ margin: 0; font-size: 0.95rem; line-height: 1.16; }}
      .client-card p {{ margin: 0; }}
      .client-card-top {{ display: grid; gap: 0.45rem; grid-template-columns: minmax(0, 1fr) auto; align-items: start; }}
      .client-card-identity {{ display: grid; gap: 0.18rem; min-width: 0; }}
      .client-card-status {{ display: flex; align-items: center; gap: 0.34rem; flex-wrap: wrap; color: var(--muted); font-size: 0.76rem; }}
      .client-card-status code {{ color: var(--ink); font-size: 0.78rem; }}
      .client-card-actions {{ display: flex; gap: 0.28rem; align-items: start; justify-content: flex-end; flex-wrap: wrap; }}
      .client-card-body {{ display: grid; gap: 0.32rem; }}
      .client-card-focus {{ display: grid; gap: 0.24rem; padding: 0.04rem 0 0; }}
      .client-card-focus-top {{ display: flex; align-items: end; justify-content: space-between; gap: 0.7rem; flex-wrap: wrap; }}
      .client-card-focus-value {{ font-size: 1.18rem; font-weight: 810; letter-spacing: -0.04em; }}
      .client-card-focus-subtle {{ color: var(--muted); font-size: 0.74rem; }}
      .client-slot-track {{ width: 100%; height: 0.32rem; background: rgba(87, 104, 125, 0.1); border-radius: 0; overflow: hidden; }}
      .client-slot-fill {{ height: 100%; background: linear-gradient(90deg, rgba(18, 79, 73, 0.74), rgba(18, 79, 73, 0.96)); }}
      .client-card-meta-line {{ display: flex; flex-wrap: wrap; gap: 0.26rem 0.48rem; color: var(--muted); font-size: 0.72rem; }}
      .client-card-notes {{ display: grid; gap: 0.28rem; }}
      .client-card-section-title {{ color: var(--muted); font-size: 0.66rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; }}
      .client-job-list {{ display: grid; gap: 0.26rem; }}
      .client-job-row {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 0.18rem 0.38rem; align-items: center; padding: 0.24rem 0.28rem; border: 1px solid rgba(216, 207, 194, 0.78); border-radius: 0; background: rgba(255,255,255,0.64); }}
      .client-job-row-top {{ display: grid; gap: 0.12rem; min-width: 0; }}
      .client-job-row p {{ margin: 0; grid-column: 1 / -1; }}
      .client-note {{ padding: 0.34rem 0.38rem; border: 1px solid rgba(216, 207, 194, 0.82); border-radius: 0; background: rgba(255,255,255,0.7); }}
      .client-note.warning {{ color: var(--warn); border-color: rgba(138, 83, 31, 0.18); background: rgba(255, 244, 225, 0.72); }}
      .client-note.danger {{ color: var(--danger); border-color: rgba(139, 52, 38, 0.18); background: rgba(255, 236, 232, 0.76); }}
      .client-job-more {{ color: var(--muted); font-size: 0.72rem; padding: 0.1rem 0.02rem 0; }}
      .client-settings-form {{ display: flex; flex-wrap: wrap; gap: 0.22rem; align-items: center; margin: 0; padding: 0.18rem 0.2rem; border: 1px solid rgba(216, 207, 194, 0.82); border-radius: 0; background: rgba(255,255,255,0.82); }}
      .client-settings-form label {{ display: inline-flex; gap: 0.28rem; align-items: center; }}
      .client-settings-form input[type="number"] {{ width: 3.5rem; min-height: 1.34rem; padding: 0.12rem 0.2rem; border-radius: 0; }}
      .summaries-config-form {{ display: grid; gap: 0.7rem; align-items: stretch; }}
      .summaries-config-label {{ display: grid !important; gap: 0.42rem; width: 100%; align-items: stretch !important; }}
      .summaries-config-textarea {{ display: block; width: 100%; min-width: 0; min-height: 15rem; padding: 0.7rem; border: 1px solid var(--line); background: rgba(255,255,255,0.82); font: inherit; line-height: 1.45; resize: vertical; box-sizing: border-box; }}
      .summaries-running-row {{ display: flex; align-items: center; gap: 0.42rem; flex-wrap: wrap; margin-top: 0.55rem; }}
      .summary-view-modal-card {{ overflow: auto; }}
      .summary-view-modal-card.is-pooltool {{ width: min(96vw, 980px); max-width: min(96vw, 980px); max-height: 94vh; }}
      .summary-view-grid {{ display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(0, 1.25fr) minmax(0, 1fr); gap: 0.8rem; align-items: start; max-height: 78vh; overflow: auto; }}
      .summary-view-grid.is-pooltool {{ grid-template-columns: 1fr; max-height: none; overflow: visible; }}
      .summary-view-column {{ display: grid; gap: 0.48rem; min-width: 0; }}
      .summary-view-column.summary-view-media-column.is-pooltool {{ justify-items: center; }}
      .summary-view-grid.is-pooltool .summary-view-column {{ max-width: 100%; }}
      .summary-view-grid.is-pooltool .video-modal-frame {{ width: min(100%, 32rem); max-width: min(100%, 32rem); max-height: 72vh; min-height: 24rem; margin: 0 auto; }}
      .summary-view-column h3 {{ margin: 0; font-size: 0.86rem; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }}
      .summary-variant-list {{ display: grid; gap: 0.45rem; max-height: 66vh; overflow: auto; padding-right: 0.15rem; }}
      .summary-variant-row {{ display: grid; gap: 0.3rem; padding: 0.5rem 0.56rem; border: 1px solid var(--line); background: rgba(255,255,255,0.78); }}
      .summary-variant-row h4 {{ margin: 0; font-size: 0.74rem; letter-spacing: 0.06em; text-transform: uppercase; color: var(--accent-strong); }}
      .summary-variant-row pre {{ margin: 0; white-space: pre-wrap; font: 0.78rem/1.45 "IBM Plex Mono", "SFMono-Regular", Consolas, monospace; overflow-wrap: anywhere; }}
      .queue-progress {{ display: grid; gap: 0.25rem; margin-top: 0.5rem; }}
      .job-progress-stack {{ display: grid; gap: 0.75rem; margin-top: 0.55rem; }}
      .job-progress-hero {{ display: flex; justify-content: space-between; gap: 0.8rem; align-items: stretch; flex-wrap: wrap; }}
      .job-progress-hero-main {{ min-width: 9rem; padding: 0.72rem 0.8rem; border: 1px solid var(--line); background: rgba(255,255,255,0.5); }}
      .job-progress-kicker {{ color: var(--accent-strong); font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }}
      .job-progress-hero-value {{ margin-top: 0.2rem; font-size: 1.45rem; font-weight: 800; letter-spacing: -0.03em; }}
      .job-progress-hero-label {{ color: var(--muted); font-size: 0.82rem; }}
      .job-progress-hero-side {{ display: flex; gap: 0.55rem; flex-wrap: wrap; align-items: stretch; }}
      .job-progress-metric-card {{ min-width: 6rem; padding: 0.65rem 0.7rem; border: 1px solid var(--line); background: rgba(255,255,255,0.5); display: grid; gap: 0.15rem; }}
      .job-progress-metric-label {{ color: var(--muted); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }}
      .job-progress-metric-value {{ font-size: 1rem; }}
      .job-progress-line {{ display: flex; gap: 0.45rem; flex-wrap: wrap; }}
      .job-progress-pill {{ display: inline-flex; align-items: center; padding: 0.2rem 0.5rem; border: 1px solid rgba(87,104,125,0.14); background: rgba(87,104,125,0.1); color: #46576a; font-size: 0.78rem; font-weight: 700; }}
      .job-progress-pill-strong {{ background: rgba(18,79,73,0.12); color: var(--accent-strong); border-color: rgba(18,79,73,0.18); }}
      .job-worker-visual-list {{ display: grid; gap: 0.6rem; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
      .job-worker-visual-card {{ padding: 0.7rem 0.76rem; border: 1px solid var(--line); background: rgba(255,255,255,0.48); display: grid; gap: 0.45rem; }}
      .job-worker-visual-header {{ display: flex; justify-content: space-between; gap: 0.45rem; align-items: center; }}
      .job-worker-mini-id {{ font-size: 0.78rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
      .job-worker-stage-pill {{ display: inline-flex; align-items: center; padding: 0.16rem 0.45rem; border: 1px solid rgba(87,104,125,0.14); background: rgba(87,104,125,0.1); color: #46576a; font-size: 0.74rem; font-weight: 700; }}
      .job-worker-stage-pill.stage-search, .job-worker-stage-pill.stage-llm, .job-worker-stage-pill.stage-label, .job-worker-stage-pill.stage-label-done {{ background: rgba(18,79,73,0.12); color: var(--accent-strong); border-color: rgba(18,79,73,0.18); }}
      .job-worker-stage-pill.stage-paused {{ background: var(--warn-soft); color: var(--warn); border-color: rgba(138, 83, 31, 0.14); }}
      .job-worker-stage-pill.stage-failed {{ background: var(--danger-soft); color: var(--danger); border-color: rgba(139, 52, 38, 0.14); }}
      .job-worker-visual-label {{ font-size: 0.88rem; font-weight: 700; }}
      .job-worker-visual-meta {{ display: flex; flex-wrap: wrap; gap: 0.45rem; color: var(--muted); font-size: 0.78rem; }}
      .progress-block-tight {{ display: grid; gap: 0.3rem; }}
      .job-progress-track-large {{ height: 0.6rem; }}
      .job-worker-progress-block .progress-track {{ height: 0.4rem; }}
      .learning-progress-detail-grid {{ display: grid; gap: 0.95rem; grid-template-columns: minmax(0, 1.35fr) minmax(18rem, 1fr); }}
      .compact-table td, .compact-table th {{ font-size: 0.82rem; }}
      .progress-track {{ width: min(20rem, 100%); height: 0.42rem; background: rgba(87, 104, 125, 0.15); border: 1px solid rgba(87, 104, 125, 0.18); overflow: hidden; }}
      .progress-fill {{ height: 100%; background: linear-gradient(90deg, rgba(18, 79, 73, 0.7), rgba(18, 79, 73, 0.95)); }}
      .recent-jobs-scroll {{ max-height: 24rem; overflow-y: auto; margin-top: 0.6rem; padding-right: 0.2rem; }}
      .env-link-grid {{ display: grid; gap: 0.75rem; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-top: 0.1rem; }}
      .env-link-card {{ display: grid; gap: 0.35rem; padding: 0.9rem 0.95rem; border: 1px solid var(--line); background: rgba(255, 255, 255, 0.45); }}
      .env-subnav {{ display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.9rem 0 0; }}
      .env-subnav-link {{ display: inline-flex; align-items: center; min-height: 2rem; padding: 0.25rem 0.7rem; border: 1px solid var(--line); background: rgba(18, 79, 73, 0.04); color: var(--accent-strong); font-size: 0.84rem; font-weight: 700; }}
      .env-subnav-link.is-active {{ background: rgba(18, 79, 73, 0.12); border-color: rgba(18, 79, 73, 0.24); }}
      .asset-list {{ display: grid; gap: 0.7rem; margin-top: 0.75rem; }}
      .asset-row {{ display: grid; gap: 0.45rem; padding: 0.75rem 0.8rem; border: 1px solid var(--line); background: rgba(255, 255, 255, 0.45); }}
      .asset-row-top {{ display: flex; justify-content: space-between; gap: 0.8rem; align-items: start; }}
      .asset-title {{ margin: 0; font-size: 0.95rem; }}
      .asset-meta {{ display: flex; flex-wrap: wrap; gap: 0.4rem; }}
      .asset-inline-links {{ display: flex; flex-wrap: wrap; gap: 0.55rem; }}
      .dataset-split-stack {{ display: grid; gap: 0.9rem; }}
      .dataset-split-header {{ display: flex; justify-content: space-between; gap: 0.8rem; align-items: start; flex-wrap: wrap; }}
      .dataset-trace-list {{ display: grid; gap: 0.55rem; margin-top: 0.7rem; }}
      .dataset-trace-row {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 0.75rem; align-items: start; padding: 0.65rem 0.72rem; border: 1px solid var(--line); background: rgba(255, 255, 255, 0.45); }}
      .dataset-trace-main {{ display: grid; gap: 0.25rem; min-width: 0; }}
      .dataset-trace-main p {{ margin: 0; }}
      .dataset-trace-meta {{ display: flex; flex-wrap: wrap; gap: 0.35rem; }}
      .dataset-trace-side {{ display: grid; gap: 0.35rem; justify-items: end; }}
      .dataset-info-grid {{ display: grid; gap: 0.8rem; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }}
      .dataset-info-card {{ padding: 0.75rem 0.8rem; border: 1px solid var(--line); background: rgba(255, 255, 255, 0.45); }}
      .dataset-info-card h3 {{ margin: 0 0 0.28rem; font-size: 0.9rem; }}
      .key-value-grid {{ display: grid; gap: 0.45rem 0.8rem; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); }}
      .key-value-card {{ padding: 0.62rem 0.68rem; border: 1px solid rgba(203, 191, 174, 0.9); background: rgba(255, 255, 255, 0.6); }}
      .key-value-label {{ display: block; color: var(--muted); font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }}
      .key-value-value {{ display: block; margin-top: 0.18rem; font-size: 0.9rem; font-weight: 700; overflow-wrap: anywhere; }}
      .media-grid {{ display: grid; gap: 0.8rem; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }}
      .media-card {{ display: grid; gap: 0.45rem; padding: 0.72rem 0.76rem; border: 1px solid var(--line); background: rgba(255, 255, 255, 0.5); }}
      .media-card figure {{ margin: 0; display: grid; gap: 0.45rem; }}
      .media-card img, .media-card video {{ width: 100%; max-height: 28rem; object-fit: contain; border: 1px solid rgba(203, 191, 174, 0.9); background: rgba(244, 240, 232, 0.8); }}
      details.data-details {{ border: 1px solid var(--line); background: rgba(255, 255, 255, 0.45); }}
      details.data-details > summary {{ cursor: pointer; list-style: none; padding: 0.65rem 0.75rem; font-weight: 700; }}
      details.data-details > summary::-webkit-details-marker {{ display: none; }}
      details.data-details pre {{ margin: 0; border-top: 1px solid rgba(203, 191, 174, 0.9); }}
      .table-scroll, .scroll-table {{ width: 100%; max-width: 100%; overflow-x: auto; overflow-y: visible; }}
      .table-scroll > *, .scroll-table > * {{ min-width: 100%; }}
      .data-table {{ width: 100%; border-collapse: collapse; margin-top: 0.6rem; }}
      .data-table th, .data-table td {{ padding: 0.5rem 0.45rem; border-top: 1px solid var(--line); text-align: left; vertical-align: top; }}
      .data-table th {{ color: var(--muted); font-size: 0.76rem; letter-spacing: 0.06em; text-transform: uppercase; }}
      .experiment-run-table tbody tr.is-current {{ background: rgba(18, 79, 73, 0.06); }}
      .experiment-run-cell {{ display: grid; gap: 0.18rem; min-width: 12rem; }}
      .experiment-eval-chip-row {{ display: flex; flex-wrap: wrap; gap: 0.35rem; min-width: 16rem; }}
      .experiment-eval-chip {{ display: inline-flex; align-items: center; min-height: 1.5rem; padding: 0.12rem 0.42rem; border: 1px solid rgba(18, 79, 73, 0.14); background: rgba(18, 79, 73, 0.08); color: var(--accent-strong); font-size: 0.74rem; font-weight: 700; text-decoration: none; }}
      .experiment-eval-chip:hover {{ background: rgba(18, 79, 73, 0.12); text-decoration: none; }}
      .experiment-eval-chip.is-running, .experiment-eval-chip.is-succeeded {{ background: rgba(18,121,86,0.12); border-color: rgba(18,121,86,0.18); color: #0d5d42; }}
      .experiment-eval-chip.is-paused {{ background: rgba(138, 83, 31, 0.10); border-color: rgba(138, 83, 31, 0.14); color: var(--warn); }}
      .experiment-eval-chip.is-failed {{ background: rgba(139, 52, 38, 0.10); border-color: rgba(139, 52, 38, 0.14); color: var(--danger); }}
      .experiment-eval-chip.is-queued {{ background: rgba(87, 104, 125, 0.08); border-color: rgba(87, 104, 125, 0.14); color: #46576a; }}
      .trajectory-chart-large .loss-chart-frame {{ min-height: 24rem; }}
      .label-chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 0.8rem; margin-top: 0.85rem; }}
      .label-chart-card {{ padding: 0.8rem 0.85rem; border: 1px solid rgba(203, 191, 174, 0.9); background: rgba(255,255,255,0.5); display: grid; gap: 0.55rem; }}
      .label-chart-card h3 {{ margin: 0; font-size: 0.92rem; }}
      .label-chart-meta {{ display: flex; flex-wrap: wrap; gap: 0.5rem; color: var(--muted); font-size: 0.78rem; }}
      .label-chart-frame {{ display: grid; gap: 0.4rem; }}
      .label-chart-svg {{ width: 100%; height: auto; display: block; }}
      .label-chart-legend {{ display: flex; gap: 0.65rem; flex-wrap: wrap; color: var(--muted); font-size: 0.76rem; }}
      .label-chart-legend span {{ display: inline-flex; align-items: center; gap: 0.32rem; }}
      .candidate-label-list {{ display: grid; gap: 0.65rem; }}
      details.candidate-label-panel {{ border: 1px solid rgba(203, 191, 174, 0.9); background: rgba(255,255,255,0.5); }}
      details.candidate-label-panel > summary {{ cursor: pointer; list-style: none; padding: 0.62rem 0.72rem; display: flex; justify-content: space-between; gap: 0.6rem; align-items: center; }}
      details.candidate-label-panel > summary::-webkit-details-marker {{ display: none; }}
      .candidate-label-title {{ font-weight: 700; }}
      .candidate-label-summary {{ color: var(--muted); font-size: 0.78rem; text-align: right; overflow-wrap: anywhere; }}
      .candidate-cell-list {{ display: grid; gap: 0.45rem; padding: 0.25rem 0.62rem 0.65rem; border-top: 1px solid rgba(203, 191, 174, 0.9); }}
      details.candidate-cell {{ border: 1px solid rgba(203, 191, 174, 0.86); background: rgba(255,255,255,0.58); }}
      details.candidate-cell > summary {{ cursor: pointer; list-style: none; padding: 0.46rem 0.55rem; display: flex; justify-content: space-between; gap: 0.6rem; align-items: center; }}
      details.candidate-cell > summary::-webkit-details-marker {{ display: none; }}
      .candidate-cell-title {{ font-family: "IBM Plex Mono", "SFMono-Regular", monospace; font-size: 0.78rem; font-weight: 700; }}
      .candidate-cell-summary {{ color: var(--muted); font-size: 0.76rem; text-align: right; overflow-wrap: anywhere; }}
      .candidate-cell-body {{ padding: 0 0.55rem 0.55rem; border-top: 1px solid rgba(203, 191, 174, 0.86); display: grid; gap: 0.45rem; }}
      .candidate-cell-meta {{ display: flex; flex-wrap: wrap; gap: 0.35rem; padding-top: 0.45rem; }}
      .candidate-code-block {{ max-height: 18rem; margin-top: 0.05rem; }}
      .job-hero-title-row {{ display: flex; flex-wrap: wrap; gap: 0.65rem 0.8rem; align-items: center; }}
      .job-hero-subline {{ display: flex; flex-wrap: wrap; gap: 0.55rem 0.75rem; margin-top: 0.55rem; color: var(--muted); font-size: 0.86rem; }}
      .job-hero-subline span {{ display: inline-flex; align-items: center; gap: 0.3rem; }}
      .job-message-banner {{ padding: 0.8rem 0.9rem; border: 1px solid var(--line); background: rgba(255,255,255,0.52); display: grid; gap: 0.25rem; }}
      .job-message-banner strong {{ font-size: 0.76rem; letter-spacing: 0.08em; text-transform: uppercase; }}
      .job-message-banner p {{ margin: 0; }}
      .job-message-banner.tone-waiting {{ border-color: rgba(138, 83, 31, 0.18); background: rgba(255, 244, 225, 0.82); }}
      .job-message-banner.tone-warning {{ border-color: rgba(138, 83, 31, 0.18); background: rgba(255, 244, 225, 0.72); }}
      .job-message-banner.tone-danger {{ border-color: rgba(139, 52, 38, 0.18); background: rgba(255, 236, 232, 0.82); }}
      .job-message-banner.tone-info {{ border-color: rgba(18, 79, 73, 0.14); background: rgba(234, 247, 245, 0.8); }}
      .job-overview-grid {{ display: grid; gap: 0.7rem; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
      .job-overview-card {{ padding: 0.8rem 0.85rem; border: 1px solid var(--line); background: rgba(255,255,255,0.52); display: grid; gap: 0.22rem; }}
      .job-overview-kicker {{ color: var(--muted); font-size: 0.71rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }}
      .job-overview-value {{ font-size: 1rem; font-weight: 800; overflow-wrap: anywhere; }}
      .job-overview-meta {{ color: var(--muted); font-size: 0.79rem; }}
      .job-link-strip {{ display: grid; gap: 0.65rem; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }}
      .job-link-chip {{ display: grid; gap: 0.16rem; padding: 0.75rem 0.8rem; border: 1px solid rgba(18,79,73,0.14); background: rgba(18,79,73,0.06); color: inherit; }}
      .job-link-chip:hover {{ border-color: rgba(18,79,73,0.28); background: rgba(18,79,73,0.1); }}
      .job-link-chip-label {{ font-size: 0.73rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent-strong); }}
      .job-link-chip-path {{ font-size: 0.84rem; color: var(--muted); overflow-wrap: anywhere; }}
      .loss-chart-frame {{ display: grid; gap: 0.55rem; }}
      .chart-toggle-row {{ display: flex; flex-wrap: wrap; gap: 0.45rem; }}
      .loss-chart-legend {{ display: flex; flex-wrap: wrap; gap: 0.65rem 0.9rem; }}
      .loss-chart-legend span {{ display: inline-flex; align-items: center; gap: 0.38rem; color: var(--muted); font-size: 0.82rem; font-weight: 600; }}
      .legend-swatch {{ width: 0.8rem; height: 0.8rem; border: 1px solid currentColor; display: inline-block; }}
      .loss-chart-svg {{ width: 100%; height: auto; background: rgba(255, 255, 255, 0.38); border: 1px solid rgba(203, 191, 174, 0.85); }}
      .loss-grid-line {{ stroke: rgba(100, 93, 84, 0.18); stroke-width: 1; }}
      .loss-axis-line {{ stroke: rgba(100, 93, 84, 0.52); stroke-width: 1.2; }}
      .loss-axis-label {{ fill: var(--muted); font-size: 11px; font-family: "IBM Plex Mono", "SFMono-Regular", monospace; }}
      .loss-zero-line {{ stroke: rgba(18, 79, 73, 0.42); stroke-dasharray: 4 4; stroke-width: 1.2; }}
      .loss-zero-label {{ fill: var(--accent-strong); font-weight: 700; }}
      .loss-axis-caption {{ color: var(--muted); font-size: 0.82rem; text-align: center; }}
      .code-block, .log-block, pre.artifact-preview {{ margin: 0; padding: 0.75rem; background: rgba(23, 21, 18, 0.92); color: #f8f5ef; overflow-x: auto; overflow-y: auto; white-space: pre-wrap; max-height: 22rem; }}
      .metadata-block {{ max-height: 16rem; overflow-y: auto; font-size: 0.78rem; line-height: 1.38; }}
      .log-block {{ max-height: 18rem; overflow-y: auto; }}
      .artifact-list {{ display: grid; gap: 0.7rem; }}
      .artifact-card {{ padding: 0.72rem 0.78rem; border: 1px solid var(--line); background: rgba(255, 255, 255, 0.45); }}
      .artifact-card h3 {{ margin: 0 0 0.25rem; font-size: 0.92rem; }}
      .artifact-meta {{ margin-bottom: 0.4rem; }}
      .task-list {{ display: grid; gap: 0.7rem; max-height: 28rem; overflow-y: auto; padding-right: 0.2rem; }}
      .task-row {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 0.8rem; align-items: start; padding: 0.72rem 0.78rem; border: 1px solid var(--line); background: rgba(255, 255, 255, 0.45); }}
      .task-row-main {{ display: grid; gap: 0.18rem; min-width: 0; }}
      .task-row-meta {{ display: flex; flex-wrap: wrap; gap: 0.35rem; align-items: center; justify-content: flex-end; }}
      .attempt-stack {{ display: grid; gap: 0.75rem; }}
      .attempt-card {{ padding: 0.78rem 0.85rem; border: 1px solid var(--line); background: rgba(255,255,255,0.5); display: grid; gap: 0.6rem; }}
      .attempt-card-top {{ display: flex; justify-content: space-between; gap: 0.75rem; align-items: start; flex-wrap: wrap; }}
      .attempt-card-title {{ margin: 0; font-size: 0.95rem; }}
      .attempt-grid {{ display: grid; gap: 0.7rem; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }}
      .required-grid {{ display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(338px, 1fr)); }}
      .analysis-launch-grid {{ display: grid; gap: 0.8rem; grid-template-columns: 1fr; }}
      .analysis-launch-panel > summary {{ list-style: none; cursor: pointer; }}
      .analysis-launch-panel > summary::-webkit-details-marker {{ display: none; }}
      .analysis-launch-panel[open] > summary {{ padding-bottom: 0.35rem; border-bottom: 1px solid rgba(203, 191, 174, 0.54); margin-bottom: 0.65rem; }}
      .analysis-env-card {{ display: grid; gap: 0.82rem; padding: 0.92rem 0.98rem; border: 1px solid rgba(193, 181, 163, 0.92); background: rgba(255,255,255,0.96); }}
      .analysis-env-panel > summary {{ list-style: none; cursor: pointer; }}
      .analysis-env-panel > summary::-webkit-details-marker {{ display: none; }}
      .analysis-env-panel[open] > summary {{ padding-bottom: 0.28rem; border-bottom: 1px solid rgba(203, 191, 174, 0.54); }}
      .analysis-env-panel-body {{ display: grid; gap: 0.72rem; }}
      .analysis-env-head {{ display: flex; justify-content: space-between; gap: 0.7rem; align-items: start; }}
      .analysis-env-head h3 {{ margin: 0; font-size: 0.96rem; letter-spacing: 0.03em; }}
      .reward-client-grid {{ display: grid; gap: 0.65rem; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }}
      .reward-client-card {{ display: grid; gap: 0.45rem; padding: 0.75rem 0.8rem; border: 1px solid rgba(193, 181, 163, 0.92); background: rgba(255,255,255,0.92); }}
      .reward-client-card-top {{ display: flex; justify-content: space-between; gap: 0.7rem; align-items: start; }}
      .reward-client-card-top h3 {{ margin: 0; font-size: 0.92rem; }}
      .reward-client-card-meta {{ color: var(--muted); font-size: 0.75rem; line-height: 1.3; overflow-wrap: anywhere; }}
      .reward-client-card-body {{ display: grid; gap: 0.35rem; }}
      .reward-client-job {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 0.45rem; align-items: center; padding: 0.42rem 0.46rem; border: 1px solid rgba(214, 205, 190, 0.88); background: rgba(250,248,243,0.86); }}
      .reward-client-job-main {{ display: grid; gap: 0.1rem; min-width: 0; }}
      .reward-client-job-main strong {{ font-size: 0.78rem; line-height: 1.2; overflow-wrap: anywhere; }}
      .reward-client-job-meta {{ color: var(--muted); font-size: 0.72rem; }}
      .reward-workbench-grid {{ display: grid; gap: 0.8rem; grid-template-columns: minmax(0, 1.35fr) minmax(18rem, 0.95fr); align-items: start; }}
      .reward-workbench-stack {{ display: grid; gap: 0.8rem; }}
      .reward-workbench-card {{ display: grid; gap: 0.65rem; padding: 0.78rem 0.82rem; border: 1px solid rgba(193, 181, 163, 0.92); background: rgba(255,255,255,0.94); }}
      .reward-workbench-card-primary {{ min-height: 100%; }}
      .reward-form {{ display: grid; gap: 0.55rem; }}
      .reward-form label {{ display: grid; gap: 0.2rem; }}
      .reward-form-grid {{ display: grid; gap: 0.45rem; }}
      .reward-form-grid-top {{ grid-template-columns: minmax(0, 1.4fr) minmax(0, 1.2fr) minmax(12rem, 0.9fr); }}
      .reward-form-grid-compact {{ grid-template-columns: repeat(4, minmax(0, 1fr)); align-items: end; }}
      .reward-textarea-grid {{ display: grid; gap: 0.5rem; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .reward-textarea-grid textarea {{ min-height: 7.2rem; }}
      .analysis-run-picker {{ display: grid; gap: 0.35rem; border: 1px solid rgba(203, 191, 174, 0.82); background: rgba(250, 247, 242, 0.86); padding: 0.5rem 0.56rem; }}
      .analysis-run-picker > summary {{ cursor: pointer; font-weight: 700; color: var(--ink); list-style: none; }}
      .analysis-run-picker > summary::-webkit-details-marker {{ display: none; }}
      .analysis-run-list {{ display: grid; gap: 0.3rem; max-height: 16rem; overflow-y: auto; padding: 0.08rem 0.03rem; }}
      .analysis-run-option {{ display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 0.48rem; align-items: start; padding: 0.42rem 0.48rem; border: 1px solid rgba(214, 205, 190, 0.9); background: rgba(255,255,255,0.94); }}
      .analysis-run-option input {{ margin-top: 0.2rem; }}
      .analysis-run-copy {{ display: grid; gap: 0.1rem; min-width: 0; }}
      .analysis-run-copy strong {{ font-size: 0.82rem; line-height: 1.2; font-family: "IBM Plex Mono", "SFMono-Regular", monospace; }}
      .analysis-run-meta {{ color: var(--muted); font-size: 0.74rem; line-height: 1.3; }}
      .analysis-form-grid {{ display: grid; gap: 0.55rem; grid-template-columns: minmax(0, 1.35fr) minmax(12rem, 1.05fr) minmax(10rem, 0.72fr) minmax(11rem, 0.95fr); align-items: end; }}
      .analysis-form-grid label {{ display: grid; gap: 0.22rem; }}
      .analysis-submit-row {{ display: flex; justify-content: flex-end; }}
      .analysis-submit-row .secondary-button {{ min-width: 12rem; }}
      .analysis-result-list {{ display: grid; gap: 0.5rem; max-height: 28rem; overflow-y: auto; padding-right: 0.1rem; }}
      .analysis-result-card {{ display: grid; gap: 0.35rem; padding: 0.7rem 0.76rem; border: 1px solid rgba(201, 190, 173, 0.9); background: rgba(255,255,255,0.9); }}
      .analysis-result-card-title {{ margin: 0; font-size: 0.86rem; line-height: 1.2; }}
      .analysis-result-card-meta {{ color: var(--muted); font-size: 0.75rem; line-height: 1.35; }}
      .analysis-result-head {{ display: flex; justify-content: space-between; gap: 0.7rem; align-items: start; }}
      .analysis-result-links {{ display: flex; flex-wrap: wrap; gap: 0.45rem; }}
      .analysis-result-card .stats-inline {{ gap: 0.35rem; }}
      .analysis-overview-grid {{ display: grid; gap: 0.8rem; grid-template-columns: minmax(0, 1.3fr) minmax(0, 0.9fr); align-items: start; }}
      .analysis-overview-stack {{ display: grid; gap: 0.8rem; }}
      .analysis-plot-card {{ display: grid; gap: 0.55rem; }}
      .analysis-plot-frame {{ display: block; width: 100%; border: 1px solid var(--line); background: var(--paper-strong); }}
      .analysis-plot-shell {{ display: grid; gap: 0.6rem; }}
      .analysis-plot-legend {{ display: flex; flex-wrap: wrap; gap: 0.45rem; }}
      .analysis-plot-trigger {{ display: block; width: 100%; padding: 0; border: 0; background: transparent; cursor: zoom-in; text-align: left; }}
      .analysis-plot-modal-card {{ width: min(96vw, 1240px); max-height: 92vh; overflow: auto; }}
      .analysis-plot-modal-body {{ display: grid; gap: 0.75rem; }}
      .analysis-plot-modal-body .analysis-plot-frame {{ border-width: 1px; }}
      .analysis-legend-swatch {{ display: inline-block; width: 0.8rem; height: 0.8rem; border-radius: 999px; vertical-align: middle; margin-right: 0.35rem; }}
      .analysis-dashboard-card {{ display: grid; gap: 0.75rem; }}
      .analysis-dashboard-actions {{ display: flex; justify-content: space-between; gap: 0.6rem; align-items: center; flex-wrap: wrap; }}
      .analysis-summary-grid {{ display: grid; gap: 0.8rem; grid-template-columns: minmax(0, 1fr) minmax(16rem, 0.85fr); }}
      .analysis-key-grid {{ display: grid; gap: 0.45rem; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); }}
      .analysis-run-chip-list {{ display: grid; gap: 0.35rem; min-width: 0; }}
      .analysis-run-chip {{ display: flex; justify-content: space-between; gap: 0.6rem; padding: 0.4rem 0.48rem; border: 1px solid rgba(203, 191, 174, 0.88); background: rgba(255,255,255,0.78); font-size: 0.76rem; min-width: 0; overflow-wrap: anywhere; word-break: break-word; }}
      .analysis-checkpoint-grid {{ display: grid; gap: 0.55rem; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); }}
      .analysis-checkpoint-strip {{ display: grid; gap: 0.55rem; }}
      .analysis-checkpoint-panel {{ display: grid; gap: 0.65rem; border: 1px solid rgba(201, 190, 173, 0.9); background: rgba(255,255,255,0.9); }}
      .analysis-checkpoint-panel > summary {{ display: flex; justify-content: space-between; gap: 0.7rem; align-items: center; cursor: pointer; list-style: none; font-weight: 700; }}
      .analysis-checkpoint-panel > summary::-webkit-details-marker {{ display: none; }}
      .analysis-checkpoint-panel[open] > summary {{ border-bottom: 1px solid rgba(203, 191, 174, 0.54); padding-bottom: 0.5rem; }}
      .analysis-checkpoint-head {{ display: inline-flex; gap: 0.6rem; align-items: baseline; flex-wrap: wrap; }}
      .analysis-checkpoint-head strong {{ font-size: 0.96rem; }}
      .analysis-checkpoint-panel .content-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.7rem; margin-top: 0.1rem; }}
      .analysis-activity-list {{ display: grid; gap: 0.45rem; }}
      .analysis-activity-row {{ display: grid; gap: 0.28rem; padding: 0.58rem 0.65rem; border: 1px solid rgba(201, 190, 173, 0.9); background: rgba(255,255,255,0.86); }}
      .analysis-activity-head {{ display: flex; justify-content: space-between; gap: 0.6rem; align-items: start; }}
      .analysis-activity-main {{ display: grid; gap: 0.1rem; min-width: 0; }}
      .analysis-activity-main strong {{ font-size: 0.8rem; line-height: 1.2; }}
      .analysis-activity-meta {{ color: var(--muted); font-size: 0.74rem; }}
      .analysis-activity-row p {{ margin: 0; }}
      .analysis-activity-message {{ color: var(--muted); font-size: 0.74rem; line-height: 1.35; overflow-wrap: anywhere; }}
      .analysis-progress {{ display: grid; gap: 0.18rem; }}
      .analysis-progress-meta {{ display: flex; justify-content: space-between; gap: 0.5rem; font-size: 0.68rem; color: var(--muted); line-height: 1.2; }}
      .analysis-progress-track {{ position: relative; width: 100%; height: 0.34rem; border: 1px solid rgba(87, 104, 125, 0.2); background: rgba(87, 104, 125, 0.11); overflow: hidden; }}
      .analysis-progress-fill {{ display: block; height: 100%; background: linear-gradient(90deg, rgba(18, 79, 73, 0.9), rgba(28, 119, 110, 0.86)); }}
      .analysis-progress-fill.is-indeterminate {{ width: 36%; animation: analysis-progress-sweep 1.5s ease-in-out infinite; }}
      .analysis-progress-note {{ color: var(--muted); font-size: 0.7rem; line-height: 1.28; overflow-wrap: anywhere; }}
      .ablation-matrix-wrap {{ overflow-x: auto; padding-bottom: 0.1rem; }}
      .ablation-matrix-table {{ min-width: 74rem; }}
      .ablation-matrix-table th {{ white-space: nowrap; }}
      .ablation-matrix-env {{ min-width: 6.2rem; font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
      .ablation-matrix-cell {{ min-width: 19rem; padding: 0.55rem 0.5rem; vertical-align: top; }}
      .ablation-cell-card {{ display: grid; gap: 0.72rem; padding: 0.82rem 0.84rem; border: 1px solid rgba(201, 190, 173, 0.92); background: rgba(255,255,255,0.94); min-height: 100%; }}
      .ablation-cell-card.is-active {{ border-color: rgba(196, 152, 24, 0.44); background: rgba(255, 247, 229, 0.98); }}
      .ablation-cell-card.is-covered {{ border-color: rgba(18, 121, 86, 0.28); background: rgba(240, 249, 243, 0.98); }}
      .ablation-cell-card.is-blocked {{ border-color: rgba(138, 83, 31, 0.3); background: rgba(250, 241, 230, 0.98); }}
      .ablation-cell-card.is-failed {{ border-color: rgba(160, 46, 30, 0.36); background: rgba(252, 238, 234, 0.98); }}
      .ablation-cell-card.is-missing {{ border-style: dashed; background: rgba(246, 242, 236, 0.95); }}
      .ablation-cell-head {{ display: flex; justify-content: space-between; gap: 0.7rem; align-items: start; }}
      .ablation-cell-head h3 {{ margin: 0.14rem 0 0; font-size: 0.86rem; line-height: 1.2; font-family: "IBM Plex Mono", "SFMono-Regular", monospace; overflow-wrap: anywhere; }}
      .ablation-cell-copy {{ margin: 0; color: var(--muted); font-size: 0.74rem; line-height: 1.38; min-height: 2rem; }}
      .ablation-cell-progress-meta {{ display: flex; justify-content: space-between; gap: 0.45rem; color: var(--muted); font-size: 0.68rem; line-height: 1.2; }}
      .ablation-cell-controls {{ display: flex; flex-wrap: wrap; gap: 0.45rem; align-items: center; min-height: 1.8rem; }}
      .ablation-cell-controls form {{ margin: 0; }}
      .ablation-track-form {{ display: grid; gap: 0.42rem; border-top: 1px solid rgba(203, 191, 174, 0.55); padding-top: 0.58rem; }}
      .ablation-track-form label {{ display: grid; gap: 0.22rem; }}
      .ablation-track-form select {{ width: 100%; max-width: 100%; }}
      @keyframes analysis-progress-sweep {{
        0% {{ transform: translateX(-115%); }}
        100% {{ transform: translateX(320%); }}
      }}
      .analysis-summary-links {{ display: flex; gap: 0.45rem; flex-wrap: wrap; max-width: 100%; min-width: 0; }}
      .analysis-summary-links a, .metric-subtle, .analysis-progress-note, .required-row-copy, .required-row-meta, .ablation-cell-copy, .ablation-cell-head h3 {{ max-width: 100%; overflow-wrap: anywhere; word-break: break-word; }}
      .required-overall-panel {{ display: grid; gap: 0.9rem; }}
      .required-overall-bar-wrap {{ display: grid; gap: 0.45rem; }}
      .required-overall-bar {{ display: flex; min-height: 1.25rem; border: 1px solid rgba(173, 160, 143, 0.9); background: rgba(235, 229, 220, 0.92); overflow: hidden; }}
      .required-overall-segment {{ display: block; min-width: 0; }}
      .required-overall-segment.is-done {{ background: rgba(18, 79, 73, 0.94); }}
      .required-overall-segment.is-active {{ background: rgba(196, 152, 24, 0.92); }}
      .required-overall-segment.is-queued {{ background: rgba(129, 115, 93, 0.42); }}
      .required-overall-segment.is-paused {{ background: rgba(138, 83, 31, 0.6); }}
      .required-overall-segment.is-failed {{ background: rgba(160, 46, 30, 0.82); }}
      .required-overall-segment.is-missing {{ background: rgba(201, 190, 173, 0.92); }}
      .required-overall-bar-label {{ color: var(--text); font-size: 0.84rem; font-weight: 800; letter-spacing: 0.03em; }}
      .required-overall-chip-row {{ display: flex; flex-wrap: wrap; gap: 0.45rem; }}
      .required-overall-chip {{ display: inline-flex; align-items: center; gap: 0.34rem; padding: 0.28rem 0.52rem; border: 1px solid rgba(197, 183, 163, 0.84); background: rgba(255,255,255,0.94); color: var(--text); font-size: 0.72rem; font-weight: 800; }}
      .required-overall-chip-label {{ color: var(--muted); text-transform: uppercase; letter-spacing: 0.07em; font-size: 0.58rem; font-weight: 900; }}
      .required-overall-chip.is-done {{ background: rgba(231, 247, 239, 0.96); border-color: rgba(18, 121, 86, 0.28); }}
      .required-overall-chip.is-active {{ background: rgba(255, 243, 214, 0.98); border-color: rgba(196, 152, 24, 0.34); }}
      .required-overall-chip.is-queued {{ background: rgba(238, 232, 223, 0.96); border-color: rgba(181, 166, 145, 0.5); }}
      .required-overall-chip.is-paused {{ background: rgba(244, 232, 218, 0.94); border-color: rgba(138, 83, 31, 0.26); }}
      .required-overall-chip.is-failed {{ background: rgba(252, 233, 228, 0.96); border-color: rgba(160, 46, 30, 0.28); }}
      .required-overall-chip.is-missing {{ background: rgba(243, 239, 232, 0.98); border-color: rgba(173, 160, 143, 0.42); }}
      .required-card {{ position: relative; overflow: hidden; padding: 0.92rem 0.96rem 0.98rem; border: 1px solid rgba(193, 181, 163, 0.92); background: rgba(255,255,255,0.9); box-shadow: 0 6px 14px rgba(70, 54, 33, 0.05); display: grid; gap: 0.82rem; }}
      .required-card::before {{ content: ""; position: absolute; inset: 0 0 auto 0; height: 0.18rem; background: rgba(72, 103, 126, 0.86); opacity: 1; }}
      .required-card .card-topline {{ position: relative; align-items: start; gap: 0.72rem; padding-top: 0.08rem; padding-bottom: 0.28rem; border-bottom: 1px solid rgba(203, 191, 174, 0.54); }}
      .required-card .mini-chip {{ border-radius: 0; background: rgba(238, 239, 241, 0.9); }}
      .required-card h3 {{ margin: 0; font-size: 1rem; letter-spacing: 0.04em; }}
      .required-list {{ display: grid; gap: 0.62rem; }}
      .required-action-bar {{ display: flex; justify-content: flex-end; gap: 0.55rem; }}
      .required-action-bar form {{ margin: 0; }}
      .required-row {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 0.72rem 0.95rem; align-items: start; padding: 0.72rem 0.78rem; border: 1px solid rgba(201, 190, 173, 0.9); border-radius: 0; background: rgba(255,255,255,0.82); box-shadow: none; }}
      .required-row.is-active-state {{ background: rgba(255,249,233,0.96); border-color: rgba(196, 152, 24, 0.36); }}
      .required-row.is-succeeded {{ background: rgba(237,249,245,0.98); border-color: rgba(18, 121, 86, 0.28); }}
      .required-row.is-terminal-failed {{ background: rgba(255,241,238,0.98); border-color: rgba(160, 46, 30, 0.28); }}
      .required-row-main {{ display: grid; gap: 0.3rem; min-width: 0; }}
      .required-row-title {{ display: flex; align-items: flex-start; gap: 0.45rem; min-width: 0; }}
      .required-row-title strong {{ font-size: 0.88rem; line-height: 1.15; }}
      .required-row-title a {{ color: inherit; text-decoration: none; }}
      .required-row-title a:hover {{ text-decoration: underline; }}
      .required-row-meta {{ color: var(--muted); font-size: 0.75rem; line-height: 1.32; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }}
      .required-repeat-summary {{ display: grid; gap: 0.46rem; margin-top: 0.42rem; }}
      .required-repeat-progress {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 0.7rem; align-items: center; }}
      .required-repeat-track {{ display: flex; height: 0.48rem; border-radius: 0; overflow: hidden; background: rgba(214, 205, 191, 0.52); box-shadow: inset 0 0 0 1px rgba(174, 160, 140, 0.16); }}
      .required-repeat-segment {{ display: block; height: 100%; min-width: 0; }}
      .required-repeat-segment.is-done {{ background: rgba(18,79,73,0.92); }}
      .required-repeat-segment.is-active {{ background: rgba(196,152,24,0.9); }}
      .required-repeat-segment.is-queued {{ background: rgba(129, 115, 93, 0.38); }}
      .required-repeat-segment.is-paused {{ background: rgba(138, 83, 31, 0.52); }}
      .required-repeat-segment.is-failed {{ background: rgba(160, 46, 30, 0.72); }}
      .required-repeat-progress-text {{ color: var(--text); font-size: 0.7rem; font-weight: 700; white-space: nowrap; }}
      .required-repeat-chip-row {{ display: flex; flex-wrap: wrap; gap: 0.42rem; }}
      .required-repeat-chip {{ display: inline-flex; align-items: center; gap: 0.3rem; padding: 0.22rem 0.48rem; border-radius: 0; font-size: 0.68rem; line-height: 1; border: 1px solid rgba(197, 183, 163, 0.84); background: rgba(255,255,255,0.94); color: var(--text); font-weight: 700; }}
      .required-repeat-chip-label {{ color: var(--muted); text-transform: uppercase; letter-spacing: 0.07em; font-size: 0.58rem; font-weight: 800; }}
      .required-repeat-chip.is-active {{ background: rgba(255, 243, 214, 0.98); border-color: rgba(196, 152, 24, 0.34); }}
      .required-repeat-chip.is-queued {{ background: rgba(238, 232, 223, 0.96); border-color: rgba(181, 166, 145, 0.5); }}
      .required-repeat-chip.is-paused {{ background: rgba(244, 232, 218, 0.94); border-color: rgba(138, 83, 31, 0.26); }}
      .required-repeat-chip.is-done {{ background: rgba(231, 247, 239, 0.96); border-color: rgba(18, 121, 86, 0.28); }}
      .required-repeat-chip.is-failed {{ background: rgba(252, 233, 228, 0.96); border-color: rgba(160, 46, 30, 0.28); }}
      .required-deepphy-mini {{ display: inline-grid; gap: 0.16rem; margin-top: 0.06rem; width: fit-content; }}
      .required-deepphy-bar {{ display: flex; width: 4.9rem; height: 0.34rem; overflow: hidden; border-radius: 0; background: rgba(87,104,125,0.12); border: 1px solid rgba(87,104,125,0.16); }}
      .required-deepphy-fill {{ display: block; height: 100%; }}
      .required-deepphy-fill.is-success {{ background: rgba(18,121,86,0.92); }}
      .required-deepphy-fill.is-failure {{ background: rgba(160,46,30,0.78); }}
      .required-deepphy-text {{ color: var(--muted); font-size: 0.7rem; font-weight: 700; letter-spacing: 0.03em; line-height: 1; }}
      .required-status {{ display: grid; gap: 0.36rem; justify-items: end; align-content: start; text-align: right; min-width: 3rem; padding-top: 0.02rem; }}
      .required-status-dot {{ width: 0.58rem; height: 0.58rem; border-radius: 0; display: inline-block; background: rgba(87,104,125,0.5); margin-top: 0.12rem; flex: 0 0 auto; }}
      .required-status-dot.is-active {{ background: rgba(18,79,73,0.95); box-shadow: 0 0 0 3px rgba(18,79,73,0.14); }}
      .required-status-dot.is-covered {{ background: rgba(18,79,73,0.72); }}
      .required-status-dot.is-blocked {{ background: rgba(138,83,31,0.92); }}
      .required-status-dot.is-failed, .required-status-dot.is-missing {{ background: rgba(139,52,38,0.92); }}
      .required-job-link {{ display: inline-flex; align-items: center; justify-content: center; min-width: 2.7rem; padding: 0.3rem 0.5rem; border-radius: 0; border: 1px solid rgba(18,79,73,0.16); background: rgba(18,79,73,0.08); color: var(--accent-strong); font-size: 0.71rem; font-weight: 800; letter-spacing: 0.04em; }}
      .event-list {{ display: grid; gap: 0.55rem; max-height: 28rem; overflow-y: auto; padding-right: 0.15rem; }}
      .event-row {{ display: grid; grid-template-columns: auto minmax(0, 1fr) auto; gap: 0.7rem; align-items: start; padding: 0.68rem 0.78rem; border: 1px solid rgba(203, 191, 174, 0.9); background: rgba(255,255,255,0.58); }}
      .event-row.tone-warning {{ background: rgba(255,244,225,0.62); border-color: rgba(138,83,31,0.16); }}
      .event-row.tone-danger {{ background: rgba(255,236,232,0.68); border-color: rgba(139,52,38,0.18); }}
      .event-row.tone-success {{ background: rgba(234,247,245,0.72); border-color: rgba(18,79,73,0.16); }}
      .event-row-prefix {{ display: flex; align-items: center; gap: 0.42rem; }}
      .event-dot {{ width: 0.58rem; height: 0.58rem; border-radius: 0; background: rgba(18,79,73,0.8); margin-top: 0.18rem; }}
      .event-dot.tone-warning {{ background: rgba(138,83,31,0.92); }}
      .event-dot.tone-danger {{ background: rgba(139,52,38,0.92); }}
      .event-dot.tone-success {{ background: rgba(18,79,73,0.96); }}
      .event-row-main {{ display: grid; gap: 0.16rem; min-width: 0; color: inherit; text-decoration: none; }}
      .event-row-main strong {{ font-size: 0.9rem; }}
      .event-row-main p {{ margin: 0; }}
      .event-row-main:hover strong {{ text-decoration: underline; }}
      .event-row-side {{ display: flex; justify-content: flex-end; align-items: start; text-align: right; white-space: nowrap; }}
      .event-row-side time {{ font-size: 0.76rem; }}
      .home-public-url-row {{ display: flex; gap: 0.55rem; align-items: center; justify-content: space-between; }}
      .home-public-url-row code {{ min-width: 0; flex: 1 1 auto; }}
      .public-access-fields {{ display: grid; gap: 0.45rem; grid-template-columns: minmax(14rem, 1fr) auto auto; align-items: end; }}
      .warning {{ color: var(--warn); }}
      .error-message {{ margin-top: 0.45rem; color: var(--danger); white-space: pre-wrap; }}
      .empty {{ margin: 0; }}
      @media (max-width: 760px) {{
        .page-shell {{ width: min(100vw - 1rem, 100%); }}
        .header-top-row {{ display: grid; gap: 0.75rem; }}
        .header-nav-primary {{ justify-content: flex-start; }}
        .queue-item {{ grid-template-columns: 1fr; }}
        .queue-item-meta {{ justify-content: flex-start; }}
        .dataset-trace-row {{ grid-template-columns: 1fr; }}
        .dataset-trace-side {{ justify-items: start; }}
        .home-public-url-row {{ align-items: flex-start; flex-direction: column; }}
        .public-access-fields {{ grid-template-columns: 1fr; }}
        .funnel-summary-main {{ align-items: flex-start; }}
        .funnel-summary-side {{ justify-content: flex-start; }}
        .home-top-grid {{ grid-template-columns: 1fr; }}
        .client-card-top {{ grid-template-columns: 1fr; }}
        .client-card-actions {{ justify-items: stretch; }}
        .client-card-stats {{ grid-template-columns: 1fr; }}
        .client-job-row {{ grid-template-columns: 1fr; }}
        .required-row {{ grid-template-columns: 1fr; }}
        .summary-example-card {{ grid-template-columns: 1fr; }}
        .summary-view-grid {{ grid-template-columns: 1fr; max-height: none; overflow: visible; }}
        .analysis-launch-grid, .analysis-result-list, .analysis-overview-grid, .analysis-summary-grid, .analysis-checkpoint-panel .content-grid, .analysis-checkpoint-grid, .reward-workbench-grid, .reward-textarea-grid {{ grid-template-columns: 1fr; }}
        .analysis-form-grid {{ grid-template-columns: 1fr; }}
        .reward-form-grid-top, .reward-form-grid-compact {{ grid-template-columns: 1fr; }}
      }}
      @media (min-width: 1500px) {{
        .client-list {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
      }}
      @media (min-width: 1100px) and (max-width: 1499px) {{
        .client-list {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
      }}
    </style>
  </head>
  <body>
    <div class="page-shell">
      <header class="page-header">
        <div class="header-inner">
          <div class="header-top-row">
            <div>
              <div class="eyebrow">Pattern Learning Workspace</div>
              <a class="home-link" href="/">Pattern Server Dashboard</a>
              <p class="tagline">Compact control for required runs, environment-specific views, clients, and public Funnel access.</p>
            </div>
            <nav class="header-nav header-nav-primary" aria-label="Primary">{primary_nav_html}</nav>
          </div>
          <nav class="header-nav header-nav-env" aria-label="Environments">{env_nav_html}</nav>
        </div>
      </header>
      <main class="page-main">
        {body_html}
      </main>
    </div>
    <script>
      (() => {{
        const panels = Array.from(document.querySelectorAll(".panel"));
        const storagePrefix = "pattern-panel:";
        const readText = (node) => {{
          if (!(node instanceof HTMLElement)) return "";
          return (node.textContent || "").replace(/\\s+/g, " ").trim();
        }};
        const summarizePanel = (panel) => {{
          const titleNode = panel.querySelector("h2, h3, .metric-title, strong");
          const kickerNode = panel.querySelector(".section-kicker, .section-label");
          const title = readText(titleNode) || "Panel";
          const kicker = readText(kickerNode) || "Section";
          const chips = [];
          for (const node of panel.querySelectorAll(".stats-inline .pill, .stats-inline .mini-chip, .asset-meta .pill, .asset-meta .mini-chip")) {{
            const text = readText(node);
            if (text && !chips.includes(text)) chips.push(text);
            if (chips.length >= 3) break;
          }}
          if (!chips.length) {{
            for (const node of panel.querySelectorAll(".stat-chip")) {{
              const label = readText(node.querySelector(".stat-label"));
              const value = readText(node.querySelector(".stat-value"));
              if (label && value) chips.push(`${{label}} ${{value}}`);
              if (chips.length >= 3) break;
            }}
          }}
          const counts = [];
          const listSelectors = [
            [".queue-item", "jobs"],
            [".client-card", "clients"],
            [".required-row", "tracked"],
            [".artifact-card", "artifacts"],
            [".task-row", "tasks"],
            [".attempt-card", "attempts"],
            [".asset-row", "items"],
          ];
          for (const [selector, label] of listSelectors) {{
            const count = panel.querySelectorAll(selector).length;
            if (count) {{
              counts.push(`${{count}} ${{label}}`);
              if (counts.length >= 2) break;
            }}
          }}
          const subtleCandidates = Array.from(panel.querySelectorAll(".metric-subtle, .queue-summary-text, .required-row-meta, .job-overview-meta"))
            .map((node) => readText(node))
            .filter(Boolean);
          const summaryParts = [...chips, ...counts];
          let summary = summaryParts.join(" · ");
          if (!summary && subtleCandidates.length) {{
            summary = subtleCandidates[0];
          }}
          return {{ title, kicker, summary: summary.slice(0, 180) }};
        }};
        panels.forEach((panel, index) => {{
          if (!(panel instanceof HTMLElement) || panel.classList.contains("is-collapsible") || panel.classList.contains("panel-static")) return;
          const {{ title, kicker, summary }} = summarizePanel(panel);
          const storageKey = storagePrefix + (title || "panel") + ":" + index + ":" + location.pathname;
          const body = document.createElement("div");
          body.className = "panel-body";
          while (panel.firstChild) {{
            body.appendChild(panel.firstChild);
          }}
          const firstElement = body.firstElementChild;
          if (firstElement instanceof HTMLElement) {{
            const hasInteractive = Boolean(firstElement.querySelector("button, form, input, select, textarea"));
            if (!hasInteractive && firstElement.matches(".card-topline, h2, h3")) {{
              firstElement.classList.add("panel-lede");
            }}
            if (!hasInteractive && firstElement.matches("h2, h3")) {{
              const nextElement = firstElement.nextElementSibling;
              if (nextElement instanceof HTMLElement && nextElement.matches(".metric-subtle")) {{
                nextElement.classList.add("panel-lede-subtle");
              }}
            }}
          }}
          const toggle = document.createElement("button");
          toggle.type = "button";
          toggle.className = "panel-toggle";
          toggle.setAttribute("aria-expanded", "true");
          toggle.innerHTML = `
            <span class="panel-toggle-copy">
              <span class="panel-toggle-kicker">${{kicker}}</span>
              <span class="panel-toggle-title">${{title}}</span>
              <span class="panel-toggle-summary">${{summary || "Expand for details"}}</span>
            </span>
            <span class="panel-toggle-meta">
              <span class="panel-toggle-chip">${{summary ? "Compact" : "Details"}}</span>
              <span class="panel-toggle-caret" aria-hidden="true"></span>
            </span>
          `;
          const setCollapsed = (collapsed) => {{
            panel.classList.toggle("is-collapsed", collapsed);
            toggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
            try {{
              localStorage.setItem(storageKey, collapsed ? "collapsed" : "open");
            }} catch (_error) {{
              // Ignore storage failures and keep the in-memory toggle state.
            }}
          }};
          panel.classList.add("is-collapsible");
          panel.append(toggle, body);
          let collapsed = false;
          try {{
            collapsed = localStorage.getItem(storageKey) === "collapsed";
          }} catch (_error) {{
            collapsed = false;
          }}
          if (!collapsed && index >= 3) {{
            const heavySelector = ".client-list, .artifact-list, .task-list, .attempt-stack, pre.code-block, pre.log-block, .required-grid, .dataset-split-stack";
            collapsed = Boolean(panel.querySelector(heavySelector));
          }}
          setCollapsed(collapsed);
          toggle.addEventListener("click", () => setCollapsed(!panel.classList.contains("is-collapsed")));
        }});
        const candidateDetails = Array.from(document.querySelectorAll("details.candidate-cell"));
        const loadCandidateCode = async (detailsNode) => {{
          if (!(detailsNode instanceof HTMLElement) || !detailsNode.open) return;
          const blocks = Array.from(detailsNode.querySelectorAll("pre[data-code-src]"));
          for (const block of blocks) {{
            if (!(block instanceof HTMLElement)) continue;
            const source = block.dataset.codeSrc || "";
            if (!source || block.dataset.codeLoaded === "1" || block.dataset.codeLoading === "1") continue;
            block.dataset.codeLoading = "1";
            block.textContent = "Loading code…";
            try {{
              const response = await fetch(source, {{ credentials: "same-origin" }});
              if (!response.ok) {{
                throw new Error(`HTTP ${{response.status}}`);
              }}
              block.textContent = await response.text();
              block.dataset.codeLoaded = "1";
            }} catch (error) {{
              block.textContent = `Failed to load code: ${{String(error)}}`;
            }} finally {{
              delete block.dataset.codeLoading;
            }}
          }}
        }};
        for (const detailsNode of candidateDetails) {{
          if (!(detailsNode instanceof HTMLElement)) continue;
          detailsNode.addEventListener("toggle", () => {{
            loadCandidateCode(detailsNode).catch(() => undefined);
          }});
          if (detailsNode.open) {{
            loadCandidateCode(detailsNode).catch(() => undefined);
          }}
        }}
        const buttons = Array.from(document.querySelectorAll("[data-client-filter]"));
        if (!buttons.length) return;
        const cards = Array.from(document.querySelectorAll("[data-client-status]"));
        const emptyState = document.querySelector("[data-client-filter-empty]");
        const applyFilter = (filter) => {{
          let visibleCount = 0;
          for (const button of buttons) {{
            button.classList.toggle("is-active", button.dataset.clientFilter === filter);
          }}
          for (const card of cards) {{
            const status = (card.dataset.clientStatus || "").toLowerCase();
            const isVisible = filter === "all" || status === filter;
            card.hidden = !isVisible;
            card.style.display = isVisible ? "" : "none";
            if (isVisible) {{
              visibleCount += 1;
            }}
          }}
          if (emptyState instanceof HTMLElement) {{
            emptyState.hidden = visibleCount !== 0;
          }}
        }};
        for (const button of buttons) {{
          button.addEventListener("click", () => applyFilter(button.dataset.clientFilter || "all"));
        }}
        applyFilter("online");
      }})();
    </script>
  </body>
</html>
"""


def _render_public_access_panel(public_access: PublicAccessState, *, next_url: str) -> str:
    active_label = "Active" if public_access.desired_enabled else "Disabled"
    public_url = html.escape(public_access.public_url or "Unavailable")
    current_api_key_label = html.escape(str(public_access.api_key or "") or "Set key")
    api_key_configured = bool(str(public_access.api_key or "").strip())
    api_key_label = "Configured" if api_key_configured else "Not configured"
    api_key_tone = "running" if api_key_configured else "queued"
    open_link = (
        f'<a class="ghost-link" target="_blank" rel="noreferrer" href="{html.escape(public_access.public_url)}">Open public URL</a>'
        if public_access.public_url
        else ""
    )
    warning = (
        '<p class="warning">This host currently needs <code>sudo tailscale set --operator=$USER</code> once, or root privileges, before the server can manage Funnel directly.</p>'
        if public_access.operator_required
        else ""
    )
    error_html = (
        f'<div class="error-message">{html.escape(public_access.last_error or "")}</div>'
        if public_access.last_error
        else ""
    )
    return f"""
    <section class="panel compact-panel panel-static home-public-panel">
      <div class="home-public-head"><strong>Public</strong></div>
      <div class="home-public-url-row">
        <div class="home-public-url"><code>{public_url}</code></div>
        {open_link.replace("Open public URL", "Open") if open_link else ""}
      </div>
      <div class="stats-inline">
        <span class="mini-chip status-{_status_class('running' if public_access.desired_enabled else 'queued')}">Access {html.escape(active_label)}</span>
        <span class="mini-chip status-{api_key_tone}">API key {html.escape(api_key_label)}</span>
        <span class="mini-chip">{current_api_key_label}</span>
      </div>
      <form method="post" action="/ui/public-access" class="client-settings-form">
        <input type="hidden" name="next" value="{html.escape(next_url)}">
        <input type="hidden" name="enabled" value="1"{" disabled" if public_access.desired_enabled else ""}>
        <div class="public-access-fields">
          <label>
            <span class="metric-subtle">API key</span>
            <input
              type="text"
              name="api_key"
              value=""
              autocomplete="off"
              placeholder="Set key"
            >
          </label>
          <button class="secondary-button" type="submit" name="mode" value="save">Set key</button>
          <button class="secondary-button" type="submit" name="mode" value="toggle">{html.escape('Disable' if public_access.desired_enabled else 'Enable')}</button>
        </div>
      </form>
      {warning}
      {error_html}
    </section>
    """


def _render_clients_panel(
    clients: list[ClientRecord],
    *,
    jobs: list[JobRecord],
    limit: int | None = None,
    next_url: str = "/",
) -> str:
    clients = _dashboard_active_clients(clients)
    if not clients:
        body = "<p class=\"empty\">No active clients right now.</p>"
        summary_html = ""
    else:
        job_lookup = {job.job_id: job for job in jobs}
        now = datetime.now(tz=DISPLAY_TZ)
        effective_jobs_by_client = {
            client.client_id: _effective_client_job_rows(client=client, jobs=jobs)
            for client in clients
        }
        visible_clients = sorted(
            clients,
            key=lambda item: (
                0 if item.status.value == "online" else 1,
                0 if effective_jobs_by_client.get(item.client_id) else 1,
                -item.last_heartbeat.timestamp(),
                _client_title(item).lower(),
            ),
        )
        hidden_count = 0
        if limit is not None and len(visible_clients) > limit:
            hidden_count = len(visible_clients) - limit
            visible_clients = visible_clients[:limit]
        online_count = sum(1 for client in visible_clients if client.status.value == "online")
        busy_count = sum(1 for client in visible_clients if effective_jobs_by_client.get(client.client_id))
        direct_count = sum(1 for client in visible_clients if client.kind.value == "direct")
        header_summary = (
            f'<div class="client-panel-summary">'
            f'<p class="section-kicker">Fleet</p>'
            f'<p class="metric-subtle">{online_count} online · {busy_count} busy · {direct_count} direct · {len(visible_clients)} clients'
            + (f' · {hidden_count} hidden' if hidden_count else "")
            + "</p></div>"
        )
        body = "".join(
            f"""
            <article class="client-card" data-client-status="{html.escape(client.status.value)}" data-client-kind="{html.escape(client.kind.value)}" data-client-full="{'true' if len(effective_jobs_by_client.get(client.client_id, [])) >= _effective_client_slots(client) else 'false'}">
              <div class="client-card-top">
                <div class="client-card-identity">
                  <h3><a href="/clients/{html.escape(client.client_id)}">{html.escape(_client_title(client))}</a></h3>
                  <div class="client-card-status">
                    {_pill(client.status.value, client.status.value)}
                    <span class="mini-chip">{html.escape(client.kind.value)}</span>
                  </div>
                </div>
                <div class="client-card-actions">
                  <form method="post" action="/ui/clients/{html.escape(client.client_id)}/settings" class="client-settings-form">
                    <input type="hidden" name="next" value="{html.escape(next_url)}">
                    <label>
                      <span class="metric-subtle">Slots</span>
                      <input type="number" min="1" step="1" name="max_concurrency" value="{_effective_client_slots(client)}">
                    </label>
                    <button class="secondary-button" type="submit">Set</button>
                  </form>
                  <form method="post" action="/ui/clients/{html.escape(client.client_id)}/shutdown" class="inline-form">
                    <input type="hidden" name="next" value="{html.escape(next_url)}">
                    <button class="secondary-button danger-button" type="submit">Shutdown</button>
                  </form>
                </div>
              </div>
              <div class="client-card-body">
                <section class="client-card-focus">
                  <div class="client-card-focus-top">
                    <div>
                      <div class="client-card-focus-value">{len(effective_jobs_by_client.get(client.client_id, []))}/{_effective_client_slots(client)}</div>
                      <div class="client-card-focus-subtle">jobs / slots</div>
                    </div>
                    <div class="stats-inline">
                      {_pill("full" if len(effective_jobs_by_client.get(client.client_id, [])) >= _effective_client_slots(client) else "available", "paused" if len(effective_jobs_by_client.get(client.client_id, [])) >= _effective_client_slots(client) else "running")}
                    </div>
                  </div>
                  <div class="client-slot-track"><div class="client-slot-fill" style="width: {min(100.0, (100.0 * len(effective_jobs_by_client.get(client.client_id, [])) / max(1, _effective_client_slots(client)))):.1f}%"></div></div>
                </section>
                <div class="client-card-meta-line">
                  <span>{html.escape(_display_duration(client.last_heartbeat, now))} ago</span>
                </div>
              </div>
              <div class="client-card-notes">
                <div class="client-card-section-title">Work</div>
                {_render_client_current_jobs(client_jobs=effective_jobs_by_client.get(client.client_id, []), job_lookup=job_lookup, now=now)}
                {'<div class="client-note danger">This client is offline but still lists active jobs. Reconcile or requeue those leases.</div>' if client.status.value == 'offline' and effective_jobs_by_client.get(client.client_id) else ''}
              </div>
            </article>
            """
            for client in visible_clients
        )
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="client-panel-header">
        {header_summary if clients else '<div class="client-panel-summary"><p class="metric-subtle">No active clients.</p></div>'}
        <div class="client-filter-group" role="group" aria-label="Client filter">
          <button class="filter-button" type="button" data-client-filter="all">All</button>
          <button class="filter-button is-active" type="button" data-client-filter="online">Online</button>
          <button class="filter-button" type="button" data-client-filter="offline">Offline</button>
        </div>
      </div>
      <div class="client-list">{body}</div>
      <p class="empty" data-client-filter-empty hidden>No clients match this filter.</p>
    </section>
    """


def render_client_page(
    *,
    client: ClientRecord,
    jobs: list[JobRecord],
    log_lookup: dict[str, list[str]],
    project_root: Path,
) -> str:
    active_jobs = [job for job in jobs if job.status.value in QUEUE_VISIBLE_STATUSES]
    effective_current_jobs = _effective_client_job_rows(client=client, jobs=jobs)
    recent_jobs = sorted(jobs, key=lambda item: item.updated_at, reverse=True)
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Home</a> / clients / {html.escape(_client_title(client))}</p>
      <h1>{html.escape(_client_title(client))}</h1>
      <p>Current capacity, assigned jobs, and exact worker logs captured by the server.</p>
      <div class="hero-stats">
        <div class="stat-chip"><span class="stat-label">Status</span><strong class="stat-value">{html.escape(client.status.value)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Slots</span><strong class="stat-value">{len(effective_current_jobs)}/{_effective_client_slots(client)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Jobs</span><strong class="stat-value">{len(jobs)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Heartbeat</span><strong class="stat-value">{_display_time(client.last_heartbeat)}</strong></div>
      </div>
      <div class="button-row" style="margin-top: 0.75rem;">
        <form method="post" action="/ui/clients/{html.escape(client.client_id)}/shutdown">
          <input type="hidden" name="next" value="/clients/{html.escape(client.client_id)}">
          <button class="secondary-button danger-button" type="submit">Shutdown Client</button>
        </form>
      </div>
    </section>
    <div class="content-grid">
      <section class="panel">
        <h2>Client</h2>
        <div class="key-value-grid">
          {_key_value_card("Display", _client_title(client))}
          {_key_value_card("Hostname", client.hostname or "—")}
          {_key_value_card("Platform", client.platform or "—")}
          {_key_value_card("Client id", client.client_id)}
          {_key_value_card("Environments", ", ".join(client.capabilities.environments) if client.capabilities.environments else "—")}
          {_key_value_card("Tags", ", ".join(client.capabilities.tags) if client.capabilities.tags else "—")}
        </div>
      </section>
      <section class="panel">
        <h2>Running Now</h2>
        {_render_client_active_jobs(client=client, jobs=active_jobs, project_root=project_root)}
      </section>
    </div>
    <section class="panel">
      <h2>Job Logs</h2>
      {_render_client_job_logs(client=client, jobs=recent_jobs[:24], log_lookup=log_lookup, project_root=project_root)}
    </section>
    """
    return _shell(
        f"{_client_title(client)} | Pattern Server",
        active_nav="clients",
        body_html=body,
    )


def _render_recent_jobs_panel(jobs: list[JobRecord]) -> str:
    rows = []
    for job in jobs[:20]:
        rows.append(
            f"""
            <tr>
              <td><a href="/jobs/{html.escape(job.job_id)}"><code>{html.escape(job.job_id[:8])}</code></a></td>
              <td>{html.escape(job.spec.action)}</td>
              <td>{html.escape(_job_env_id(job))}</td>
              <td>{_pill(job.status.value, job.status.value)}</td>
              <td>{_display_time(job.created_at)}</td>
              <td>{html.escape(job.message or "—")}</td>
            </tr>
            """
        )
    table_html = (
        f'<div class="recent-jobs-scroll"><table class="data-table"><thead><tr><th>Job</th><th>Action</th><th>Env</th><th>Status</th><th>Created</th><th>Message</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        if rows
        else '<p class="empty">No jobs yet.</p>'
    )
    return f"""
    <section class="panel compact-panel">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Recent jobs</p>
          <h2>Latest activity</h2>
        </div>
      </div>
      {table_html}
    </section>
    """


def _queue_job_title(job: JobRecord) -> str:
    action = str(job.spec.action or "").strip()
    metadata = dict(job.spec.metadata or {})
    if action == "supervised-learn-and-evaluate":
        return "Supervised Learning"
    if action == "unsupervised-launch":
        return "Unsupervised Learning"
    if action == "evaluate-deepphy":
        return "DeepPHY Eval"
    if action == "evaluate-library-case":
        family = _job_library_ablation_family(job)
        return f"{family} Ablation" if family is not None else "Library Ablation"
    if action == "analyze-supervised-consistency":
        return "Supervised Consistency"
    if action == "pooltool-reward-optimize":
        return "Reward Optimize"
    if action == "pooltool-reward-value-model":
        return "Reward Value Model"
    if action == "pooltool-reward-single-scene":
        return "Reward Single Scene"
    if action == "build-dataset":
        dataset_name = str(metadata.get("dataset_name") or "").strip()
        return f"Build Dataset{f' · {dataset_name}' if dataset_name else ''}"
    return action or "Job"


def _job_flow_label(job: JobRecord) -> str:
    metadata = dict(job.spec.metadata or {})
    workflow_kind = str(metadata.get("workflow_kind") or "").strip()
    if workflow_kind:
        return workflow_kind.replace("-", " ").title()
    return _queue_job_title(job)


def _job_status_summary(job: JobRecord, *, project_root: Path) -> str:
    learning_summary = _learning_progress_summary(job, project_root=project_root)
    if learning_summary is not None:
        total_labels = int(learning_summary.get("total_labels") or 0)
        completed_labels = int(learning_summary.get("completed_labels") or 0)
        active_workers = int(learning_summary.get("active_workers") or 0)
        bits: list[str] = []
        if total_labels:
            bits.append(f"{completed_labels}/{total_labels} labels")
        embedded_eval = _learning_embedded_eval_progress_summary(
            job,
            project_root=project_root,
        )
        if embedded_eval is not None:
            completed_eval_count = int(embedded_eval.get("completed_count") or 0)
            total_eval_count = int(embedded_eval.get("total_count") or 0)
            if total_eval_count > 0:
                bits.append(f"embedded eval {completed_eval_count}/{total_eval_count} tasks")
            elif completed_eval_count > 0:
                bits.append(f"embedded eval {completed_eval_count} tasks")
        if active_workers:
            bits.append(f"{active_workers} workers active")
        headline = str(learning_summary.get("headline") or "").strip()
        if headline:
            bits.append(_sanitize_display_text(headline))
        if bits:
            return " · ".join(bits[:3])
    deepphy_progress = _deepphy_live_progress_summary(job, project_root=project_root)
    if deepphy_progress is not None:
        completed_count = int(deepphy_progress.get("completed_count") or 0)
        total_count = int(deepphy_progress.get("total_count") or 0)
        active_task_id = _sanitize_display_text(str(deepphy_progress.get("active_task_id") or "").strip())
        active_step = _safe_int(deepphy_progress.get("active_step"))
        bits: list[str] = []
        if total_count > 0:
            bits.append(f"{completed_count}/{total_count} tasks")
        elif completed_count > 0:
            bits.append(f"{completed_count} tasks complete")
        if active_task_id:
            if active_step is not None:
                bits.append(f"{active_task_id} step {active_step:03d}")
            else:
                bits.append(active_task_id)
        if bits:
            return " · ".join(bits[:3])
    progress = _job_progress_payload(job)
    if isinstance(progress, dict):
        summary = _sanitize_display_text(str(progress.get("summary") or "").strip())
        phase = _sanitize_display_text(str(progress.get("phase") or "").strip())
        if summary and phase:
            return f"{phase} · {summary}"
        if summary:
            return summary
        if phase:
            return phase
    return ""


def _job_message_tone(job: JobRecord) -> str:
    text = _sanitize_display_text(job.message)
    lowered = text.lower()
    if "waiting for prerequisite" in lowered or "not found" in lowered:
        return "waiting"
    if job.status.value in {"failed", "cancelled", "stopped", "lost", "retryable_failure"}:
        return "danger"
    if job.status.value in {"paused", "pausing"}:
        return "warning"
    return "info"


def _job_quick_links(job: JobRecord, artifact_entries: list[dict[str, str]]) -> str:
    preferred: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    label_map = (
        ("job.log", "Worker log"),
        ("pattern_library.json", "Pattern library"),
        ("evolution_live.jsonl", "Live trajectory"),
        ("evolution_record.jsonl", "Reward record"),
        ("llm_calls.jsonl", "LLM calls"),
        ("debug.log", "Debug log"),
        ("results.json", "Results"),
    )
    for entry in artifact_entries:
        relpath = str(entry.get("relpath") or "")
        href = str(entry.get("href") or "")
        if not relpath or not href:
            continue
        rel_name = Path(relpath).name
        matched = False
        for suffix, label in label_map:
            if relpath.endswith(suffix) and label not in seen:
                preferred.append((label, href, relpath))
                seen.add(label)
                matched = True
                break
        if matched:
            continue
        kind = str(entry.get("kind") or "")
        if kind == "directory":
            if rel_name == "run_dir" and "Run dir" not in seen:
                preferred.append(("Run dir", href, relpath))
                seen.add("Run dir")
            elif rel_name == "evaluation_run_dir" and "Eval dir" not in seen:
                preferred.append(("Eval dir", href, relpath))
                seen.add("Eval dir")
    if not preferred:
        return ""
    chips = "".join(
        f'<a class="job-link-chip" href="{html.escape(href)}"><span class="job-link-chip-label">{html.escape(label)}</span><span class="job-link-chip-path">{html.escape(Path(relpath).name)}</span></a>'
        for label, href, relpath in preferred[:8]
    )
    return f"""
    <section class="panel">
      <div class="card-topline">
        <div>
          <h2>Run files</h2>
          <p class="metric-subtle">Most useful outputs for checking learning state, results, and traces.</p>
        </div>
      </div>
      <div class="job-link-strip">{chips}</div>
    </section>
    """


def _queue_job_subtitle(job: JobRecord, *, assigned_client: str) -> str:
    bits: list[str] = []
    env_id = _job_env_id(job)
    if env_id and env_id != "—":
        bits.append(env_id)
    metadata = dict(job.spec.metadata or {})
    if job.spec.action == "evaluate-deepphy":
        benchmark = str(metadata.get("benchmark_preset") or "").strip()
        if benchmark:
            bits.append(f"{benchmark} benchmark")
    workflow_kind = str(metadata.get("workflow_kind") or "").strip()
    if workflow_kind and workflow_kind not in {"supervised"}:
        bits.append(workflow_kind.replace("-", " "))
    bits.append(assigned_client)
    bits.append(_display_time(job.created_at))
    return " · ".join(bit for bit in bits if bit)


def _deepphy_eval_mode_label(job: JobRecord) -> str | None:
    if str(job.spec.action or "").strip() != "evaluate-deepphy":
        return None
    metadata = dict(job.spec.metadata or {})
    request_payload = metadata.get("request")
    description_mode = ""
    if isinstance(request_payload, dict):
        description_mode = str(request_payload.get("description_mode") or "").strip()
    if not description_mode:
        description_mode = str(metadata.get("description_mode") or "").strip()
    if description_mode == "trace_text":
        return "Image only"
    if description_mode in {"pattern_text", "trace_text_plus_patterns"}:
        return "Patterns"
    pattern_run_dir = str(metadata.get("pattern_run_dir") or "").strip()
    if isinstance(request_payload, dict) and not pattern_run_dir:
        pattern_run_dir = str(request_payload.get("pattern_run_dir") or "").strip()
    return "Patterns" if pattern_run_dir else "Image only"


def _queue_job_detail_strip(
    job: JobRecord,
    *,
    assigned_client: str,
    project_root: Path,
    attention_reason: str | None = None,
    lightweight: bool = False,
) -> str:
    metadata = dict(job.spec.metadata or {})
    pills: list[str] = []
    env_id = _job_env_id(job)
    if env_id and env_id != "—":
        pills.append(
            f'<span class="queue-signal-chip is-strong"><span class="queue-signal-chip-label">Env</span>{html.escape(env_id)}</span>'
        )
    workflow_kind = str(metadata.get("workflow_kind") or "").strip()
    if workflow_kind:
        pills.append(
            f'<span class="queue-signal-chip"><span class="queue-signal-chip-label">Flow</span>{html.escape(workflow_kind.replace("-", " "))}</span>'
        )
    if job.spec.action == "evaluate-deepphy":
        benchmark = str(metadata.get("benchmark_preset") or "").strip()
        if benchmark:
            pills.append(
                f'<span class="queue-signal-chip"><span class="queue-signal-chip-label">Preset</span>{html.escape(benchmark)}</span>'
            )
        mode_label = _deepphy_eval_mode_label(job)
        if mode_label:
            pills.append(
                f'<span class="queue-signal-chip"><span class="queue-signal-chip-label">Mode</span>{html.escape(mode_label)}</span>'
            )
    pills.append(
        f'<span class="queue-signal-chip"><span class="queue-signal-chip-label">Client</span>{html.escape(assigned_client)}</span>'
    )
    pills.append(
        f'<span class="queue-signal-chip"><span class="queue-signal-chip-label">Seen</span>{html.escape(_display_duration(_job_last_activity(job)))} ago</span>'
    )
    if job.started_at is not None:
        pills.append(
            f'<span class="queue-signal-chip"><span class="queue-signal-chip-label">Run</span>{html.escape(_display_duration(job.started_at, job.finished_at))}</span>'
        )
    if attention_reason:
        pills.append(
            f'<span class="queue-signal-chip is-alert"><span class="queue-signal-chip-label">Attention</span>{html.escape(attention_reason)}</span>'
        )
    summary = ""
    if not lightweight:
        summary = _job_status_summary(job, project_root=project_root)
    message = _sanitize_display_text(job.message)
    if message and message != summary:
        summary = f"{summary} · {message}" if summary else message

    progress_html = ""
    if not lightweight:
        progress = _job_progress_payload(job)
        fraction_raw = progress.get("fraction") if isinstance(progress, dict) else None
        try:
            fraction = float(fraction_raw) if fraction_raw is not None else None
        except (TypeError, ValueError):
            fraction = None
        if fraction is not None:
            bounded = max(0.0, min(1.0, fraction))
            percent = int(round(bounded * 100))
            progress_html = (
                '<div class="queue-progress-mini">'
                '<div class="progress-track">'
                f'<div class="progress-fill" style="width: {percent}%"></div>'
                "</div>"
                f'<div class="queue-progress-mini-meta"><span class="queue-status-dot status-{_status_class(job.status.value)}"></span><span>{percent}%</span></div>'
                "</div>"
            )
        elif summary:
            progress_html = (
                f'<div class="queue-progress-mini-meta"><span class="queue-status-dot status-{_status_class(job.status.value)}"></span>'
                f'<span>{html.escape(job.status.value)}</span></div>'
            )

    summary_html = f'<div class="queue-summary-text">{html.escape(summary)}</div>' if summary else ""
    pills_html = f'<div class="queue-signal-row">{"".join(pills)}</div>' if pills else ""
    summary_row = f'<div class="queue-summary-row">{summary_html}{progress_html}</div>' if (summary_html or progress_html) else ""
    return f'<div class="queue-detail-strip">{pills_html}{summary_row}</div>'


def _job_worker_value(job: JobRecord) -> int | None:
    metadata = dict(job.spec.metadata or {})
    request_payload = metadata.get("request")
    if isinstance(request_payload, dict):
        try:
            return max(1, int(request_payload.get("workers") or 1))
        except (TypeError, ValueError):
            pass
    try:
        return max(1, int(metadata.get("workers") or metadata.get("requested_workers") or 1))
    except (TypeError, ValueError):
        return 1

def _render_required_coverage_panel(jobs: list[JobRecord], *, project_root: Path) -> str:
    return _render_required_catalog_panel(
        title="Required Runs",
        kicker="Required Runs",
        families=REQUIRED_JOB_FAMILIES,
        catalog=_required_job_catalog_for_project(project_root),
        jobs=jobs,
        project_root=project_root,
    )


def _render_required_reward_program_panel(jobs: list[JobRecord], *, project_root: Path) -> str:
    return _render_required_catalog_panel(
        title="Required Reward Program Runs",
        kicker="Reward Programs",
        families=REQUIRED_REWARD_PROGRAM_JOB_FAMILIES,
        catalog=_required_reward_program_job_catalog_for_project(project_root),
        jobs=jobs,
        project_root=project_root,
        allow_paused_as_healthy=True,
    )


def _repeat_status_visuals(status_counts: Counter[str], *, family: str) -> dict[str, object] | None:
    if not status_counts:
        return None
    filtered_counts = Counter(
        {
            status: count
            for status, count in status_counts.items()
            if count > 0
        }
    )
    if not filtered_counts:
        return None
    total_count = max(1, _required_repeat_target_for_family(family))
    remaining = total_count
    completed_count = min(filtered_counts.get("succeeded", 0), remaining)
    remaining -= completed_count
    active_count = min(
        sum(filtered_counts.get(status, 0) for status in ACTIVE_QUEUE_STATUSES),
        remaining,
    )
    remaining -= active_count
    queued_count = min(filtered_counts.get("queued", 0), remaining)
    remaining -= queued_count
    paused_count = min(filtered_counts.get("paused", 0), remaining)
    remaining -= paused_count
    failed_count = min(
        sum(filtered_counts.get(status, 0) for status in FAILED_QUEUE_STATUSES),
        remaining,
    )
    if (completed_count + active_count + queued_count + paused_count + failed_count) <= 0:
        return None
    chips: list[tuple[str, int, str]] = []
    if active_count:
        chips.append(("Active", active_count, "is-active"))
    if queued_count:
        chips.append(("Queued", queued_count, "is-queued"))
    if paused_count:
        chips.append(("Paused", paused_count, "is-paused"))
    if completed_count:
        chips.append(("Done", completed_count, "is-done"))
    if failed_count:
        chips.append(("Failed", failed_count, "is-failed"))
    return {
        "active": active_count,
        "queued": queued_count,
        "paused": paused_count,
        "completed": completed_count,
        "failed": failed_count,
        "total": total_count,
        "completed_percent": int(round((completed_count / total_count) * 100.0)),
        "chips": chips,
    }


def _required_repeat_unit_label(family: str) -> str:
    normalized = str(family or "").strip().lower()
    if "deepphy" in normalized:
        return "evals"
    return "runs"


def _render_repeat_status_visuals(status_counts: Counter[str], *, family: str) -> str:
    summary = _repeat_status_visuals(status_counts, family=family)
    if summary is None:
        return ""
    total = int(summary["total"])
    unit_label = _required_repeat_unit_label(family)
    segment_specs = [
        ("done", int(summary["completed"]), "is-done"),
        ("active", int(summary["active"]), "is-active"),
        ("queued", int(summary["queued"]), "is-queued"),
        ("paused", int(summary["paused"]), "is-paused"),
        ("failed", int(summary["failed"]), "is-failed"),
    ]
    segments_html = "".join(
        (
            f'<span class="required-repeat-segment {segment_class}" '
            f'style="width: {(count / total) * 100:.3f}%" '
            f'title="{html.escape(label.title())}: {count}"></span>'
        )
        for label, count, segment_class in segment_specs
        if count > 0
    )
    progress_parts = [f'{summary["completed"]}/{summary["total"]} {unit_label} complete']
    if int(summary["active"]) > 0:
        progress_parts.append(f'{summary["active"]} active')
    elif int(summary["queued"]) > 0:
        progress_parts.append(f'{summary["queued"]} queued')
    elif int(summary["paused"]) > 0:
        progress_parts.append(f'{summary["paused"]} paused')
    elif int(summary["failed"]) > 0:
        progress_parts.append(f'{summary["failed"]} failed')
    chips_html = "".join(
        (
            f'<span class="required-repeat-chip {chip_class}">'
            f'<span class="required-repeat-chip-label">{html.escape(label)}</span>'
            f"{count}"
            "</span>"
        )
        for label, count, chip_class in summary["chips"]
    )
    return (
        '<div class="required-repeat-summary">'
        '<div class="required-repeat-progress">'
        '<div class="required-repeat-track">'
        f"{segments_html}"
        "</div>"
        f'<div class="required-repeat-progress-text">{" · ".join(html.escape(part) for part in progress_parts)}</div>'
        "</div>"
        f'<div class="required-repeat-chip-row">{chips_html}</div>'
        "</div>"
    )


def _job_repeat_learning_family(job: JobRecord) -> str | None:
    action = str(job.spec.action or "").strip()
    metadata = dict(job.spec.metadata or {})
    if action == "supervised-learn-and-evaluate":
        if metadata.get("combined_source_job_ids") or metadata.get("combined_source_run_dirs"):
            return "Supervised"
        run_dir_name = Path(str(metadata.get("run_dir") or "")).name.lower()
        if "-combined-" in run_dir_name:
            return "Supervised"
    request = dict(metadata.get("request") or {})
    output_path = str(request.get("output_path") or metadata.get("output_path") or "").strip()
    run_timestamp = str(request.get("run_timestamp") or "").strip()
    marker = run_timestamp or output_path
    if not _marker_has_repeat_shape(marker):
        return None
    if action == "supervised-learn-and-evaluate":
        return "Supervised"
    if action == "unsupervised-launch":
        return "Unsupervised"
    if action == "automatic-launch":
        return "Automatic"
    return None


def _job_repeat_required_family(job: JobRecord) -> str | None:
    family = _job_required_family(job)
    if family is None:
        return None
    metadata = dict(job.spec.metadata or {})
    request = dict(metadata.get("request") or {})
    markers = [
        str(request.get("run_timestamp") or "").strip(),
        str(request.get("run_id") or "").strip(),
        str(request.get("output_dir") or "").strip(),
        str(metadata.get("evaluation_run_id") or "").strip(),
        str(metadata.get("evaluation_run_dir") or "").strip(),
        str(metadata.get("run_dir") or "").strip(),
        str(request.get("output_path") or metadata.get("output_path") or "").strip(),
    ]
    marker_text = " ".join(part for part in markers if part)
    if _marker_has_repeat_shape(marker_text):
        return family
    return None


def _required_repeat_job_counts_toward_target(job: JobRecord) -> bool:
    family = _job_repeat_required_family(job)
    if family != "Image-only DeepPHY":
        return True
    metadata = dict(job.spec.metadata or {})
    request = dict(metadata.get("request") or {})
    markers = [
        str(metadata.get("evaluation_run_id") or "").strip(),
        str(request.get("run_id") or "").strip(),
        str(metadata.get("evaluation_run_dir") or "").strip(),
        str(request.get("output_dir") or "").strip(),
        str(metadata.get("run_dir") or "").strip(),
    ]
    target = _required_repeat_target_for_family(family)
    for marker in markers:
        if not marker:
            continue
        match = _REPEAT_EVAL_SUFFIX_PATTERN.match(Path(marker).name)
        if match is None:
            continue
        return int(match.group("index")) <= target
    return False


def _reward_kind_label(reward_kind: str) -> str:
    normalized = str(reward_kind or "").strip().lower()
    mapping = {
        "optimization": "Optimization",
        "value-model": "Value Model",
        "single-scene": "Single Scene",
        "optimization-suite": "Optimization Suite",
        "value-model-sweep": "Value Model Sweep",
    }
    return mapping.get(normalized, reward_kind.replace("-", " ").title())


def _reward_kind_description(reward_kind: str) -> str:
    normalized = str(reward_kind or "").strip().lower()
    mapping = {
        "optimization": "Direct reward-program search runs.",
        "value-model": "Single reward value-model training runs.",
        "single-scene": "Single-scene reward tests with rendered best-shot video.",
        "optimization-suite": "Multi-goal optimization experiment suites.",
        "value-model-sweep": "NN/ridge comparisons and value-model hyperparameter sweeps.",
    }
    return mapping.get(normalized, "Reward-program experiment runs.")


def _reward_kind_cards(*, rewards: list[RewardResultEntry], env_id: str) -> str:
    if env_id != "pooltool":
        return '<p class="empty">Reward-program results are only available for PoolTool.</p>'
    grouped: dict[str, list[RewardResultEntry]] = {}
    for item in rewards:
        grouped.setdefault(item.reward_kind, []).append(item)
    order = ("single-scene", "optimization", "value-model", "optimization-suite", "value-model-sweep")
    cards: list[str] = []
    for reward_kind in order:
        items = grouped.get(reward_kind, [])
        latest = items[0] if items else None
        meta = f"{len(items)} run{'s' if len(items) != 1 else ''}"
        if latest is not None:
            meta += f" · latest {_display_time(latest.updated_at)}"
        cards.append(
            f"""
            <a class="env-link-card" href="/env/{env_id}/rewards/{quote(reward_kind)}">
              <span class="section-label">{html.escape(_reward_kind_label(reward_kind))}</span>
              <strong>{html.escape(_reward_kind_label(reward_kind))}</strong>
              <span class="metric-subtle">{html.escape(_reward_kind_description(reward_kind))}</span>
              <span class="metric-subtle">{html.escape(meta)}</span>
            </a>
            """
        )
    return '<section class="env-link-grid">' + "".join(cards) + "</section>"


def _load_reward_summary(entry: RewardResultEntry) -> dict[str, Any]:
    try:
        payload = json.loads(entry.summary_path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _reward_sweep_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _reward_sweep_run_rows(rewards: list[RewardResultEntry]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []
    for entry in rewards:
        payload = _load_reward_summary(entry)
        ridge = payload.get("ridge") if isinstance(payload.get("ridge"), dict) else {}
        torch_nn = payload.get("torch_nn") if isinstance(payload.get("torch_nn"), dict) else {}
        ridge_holdout = ridge.get("holdout_metrics") if isinstance(ridge.get("holdout_metrics"), dict) else {}
        nn_holdout = torch_nn.get("holdout_metrics") if isinstance(torch_nn.get("holdout_metrics"), dict) else {}
        if torch_nn or ridge:
            train_curve = torch_nn.get("train_curve") if isinstance(torch_nn.get("train_curve"), list) else []
            hidden_dims = torch_nn.get("hidden_dims") if isinstance(torch_nn.get("hidden_dims"), list) else []
            runs.append(
                {
                    "entry": entry,
                    "payload": payload,
                    "goal_text": str(payload.get("goal_text") or "").strip(),
                    "dsl_program": str(payload.get("dsl_program") or "").strip(),
                    "device": str(payload.get("device") or "").strip() or "—",
                    "train_sample_count": _reward_sweep_int(payload.get("train_sample_count")) or 0,
                    "holdout_scene_count": _reward_sweep_int(payload.get("holdout_scene_count")) or 0,
                    "nn_reward": float(nn_holdout.get("chosen_mean_reward") or 0.0),
                    "nn_success": float(nn_holdout.get("chosen_success_rate") or 0.0),
                    "ridge_reward": float(ridge_holdout.get("chosen_mean_reward") or 0.0),
                    "ridge_success": float(ridge_holdout.get("chosen_success_rate") or 0.0),
                    "delta_reward": float(nn_holdout.get("chosen_mean_reward") or 0.0) - float(ridge_holdout.get("chosen_mean_reward") or 0.0),
                    "delta_success": float(nn_holdout.get("chosen_success_rate") or 0.0) - float(ridge_holdout.get("chosen_success_rate") or 0.0),
                    "best_epoch": _reward_sweep_int(torch_nn.get("best_epoch")) or 0,
                    "epoch_count": len(train_curve),
                    "hidden_dims": ", ".join(str(int(value)) for value in hidden_dims if isinstance(value, (int, float))) or "—",
                }
            )
        else:
            metas.append(
                {
                    "entry": entry,
                    "payload": payload,
                    "completed_runs": _reward_sweep_int(payload.get("completed_runs")) or 0,
                    "finished_at": str(payload.get("finished_at") or "").strip(),
                    "best": payload.get("best") if isinstance(payload.get("best"), dict) else {},
                }
            )
    runs.sort(key=lambda item: (item["nn_success"], item["nn_reward"], item["entry"].updated_at), reverse=True)
    metas.sort(key=lambda item: item["entry"].updated_at, reverse=True)
    return runs, metas


def _render_reward_value_model_sweep_page(*, env_id: str, label: str, rewards: list[RewardResultEntry]) -> str:
    runs, metas = _reward_sweep_run_rows(rewards)
    goals = sorted({row["goal_text"] for row in runs if row["goal_text"]})
    best_reward = max(runs, key=lambda item: (item["nn_reward"], item["nn_success"], item["entry"].updated_at)) if runs else None
    best_success = max(runs, key=lambda item: (item["nn_success"], item["nn_reward"], item["entry"].updated_at)) if runs else None
    best_delta = max(runs, key=lambda item: (item["delta_reward"], item["delta_success"], item["entry"].updated_at)) if runs else None
    avg_delta_reward = sum(item["delta_reward"] for item in runs) / len(runs) if runs else 0.0
    avg_delta_success = sum(item["delta_success"] for item in runs) / len(runs) if runs else 0.0

    comparison_series = []
    for index, item in enumerate(runs[:20], start=1):
        comparison_series.append((float(index), float(item["nn_reward"] * 100.0), float(item["ridge_reward"] * 100.0)))
    chart_html = '<p class="empty">No sweep results yet.</p>'
    if comparison_series:
        chart_html = _render_xy_line_chart(
            title="Top sweep runs",
            x_label="Rank",
            y_label="Chosen mean reward (%)",
            series=[
                {
                    "label": "Torch NN",
                    "color": "#124f49",
                    "points": [(rank, nn_reward) for rank, nn_reward, _ridge_reward in comparison_series],
                },
                {
                    "label": "Ridge",
                    "color": "#a54d2d",
                    "dasharray": "6 4",
                    "points": [(rank, ridge_reward) for rank, _nn_reward, ridge_reward in comparison_series],
                },
            ],
        )

    best_cards: list[str] = []
    for title, item, metric_label in (
        ("Best NN reward", best_reward, f'{best_reward["nn_reward"]*100:.1f}% reward · {best_reward["nn_success"]*100:.1f}% success' if best_reward else "—"),
        ("Best NN success", best_success, f'{best_success["nn_success"]*100:.1f}% success · {best_success["nn_reward"]*100:.1f}% reward' if best_success else "—"),
        ("Best NN gain over ridge", best_delta, f'+{best_delta["delta_reward"]*100:.1f} reward pts · +{best_delta["delta_success"]*100:.1f} success pts' if best_delta else "—"),
    ):
        if item is None:
            best_cards.append(f'<article class="asset-row"><h3 class="asset-title">{html.escape(title)}</h3><p class="metric-subtle">No runs yet.</p></article>')
            continue
        best_cards.append(
            f"""
            <article class="asset-row">
              <div class="asset-row-top">
                <div>
                  <h3 class="asset-title">{html.escape(title)}</h3>
                  <p class="metric-subtle"><a href="/ui/files?path={quote(_project_relpath(item['entry'].summary_path))}">{html.escape(item['entry'].run_id)}</a></p>
                </div>
                <div class="asset-meta">
                  <span class="mini-chip">best epoch {item['best_epoch'] or '—'}</span>
                  <span class="mini-chip">{html.escape(item['hidden_dims'])}</span>
                </div>
              </div>
              <p class="metric-subtle">{html.escape(metric_label)}</p>
              <p class="metric-subtle">Goal: {html.escape(item['goal_text'] or '—')}</p>
            </article>
            """
        )

    leaderboard_rows = []
    for index, item in enumerate(runs[:24], start=1):
        leaderboard_rows.append(
            f"<tr><td>{index}</td><td>{html.escape(item['entry'].run_id)}</td><td>{item['nn_reward']*100:.1f}%</td><td>{item['nn_success']*100:.1f}%</td><td>{item['ridge_reward']*100:.1f}%</td><td>{item['ridge_success']*100:.1f}%</td><td>{item['delta_reward']*100:+.1f}</td><td>{item['delta_success']*100:+.1f}</td><td>{html.escape(item['hidden_dims'])}</td><td>{item['epoch_count']}</td></tr>"
        )
    leaderboard_html = (
        '<div class="scroll-table compact-scroll-table"><table class="data-table compact-table"><thead><tr><th>#</th><th>Run</th><th>NN reward</th><th>NN success</th><th>Ridge reward</th><th>Ridge success</th><th>Δ reward</th><th>Δ success</th><th>Hidden dims</th><th>Epochs</th></tr></thead><tbody>'
        + ''.join(leaderboard_rows)
        + '</tbody></table></div>'
        if leaderboard_rows
        else '<p class="empty">No completed sweep runs yet.</p>'
    )

    meta_rows = []
    for item in metas[:8]:
        best = item["best"]
        best_run = str(best.get("run_id") or "").strip() if isinstance(best, dict) else ""
        best_reward_text = f"{float(best.get('nn_reward') or 0.0)*100:.1f}%" if isinstance(best, dict) and best.get("nn_reward") is not None else "—"
        meta_rows.append(
            f"""
            <article class="asset-row">
              <div class="asset-row-top">
                <div>
                  <h3 class="asset-title">{html.escape(item['entry'].run_id)}</h3>
                  <p class="metric-subtle">Updated {_display_time(item['entry'].updated_at)}</p>
                </div>
                <div class="asset-meta">
                  <span class="mini-chip">{item['completed_runs']} runs</span>
                  <span class="mini-chip">best {html.escape(best_reward_text)}</span>
                </div>
              </div>
              <p class="metric-subtle">Best run: {html.escape(best_run or '—')}</p>
              <div class="asset-inline-links">
                <a href="/ui/files?path={quote(_project_relpath(item['entry'].summary_path))}">Summary</a>
                <a href="/ui/files?path={quote(_project_relpath(item['entry'].root))}">Run root</a>
              </div>
            </article>
            """
        )
    meta_html = '<div class="asset-list">' + ''.join(meta_rows) + '</div>' if meta_rows else ''

    recent_rows = []
    for item in runs[:36]:
        recent_rows.append(
            f"""
            <article class="asset-row">
              <div class="asset-row-top">
                <div>
                  <h3 class="asset-title">{html.escape(item['entry'].run_id)}</h3>
                  <p class="metric-subtle">Updated {_display_time(item['entry'].updated_at)}</p>
                </div>
                <div class="asset-meta">
                  <span class="mini-chip">NN {item['nn_success']*100:.1f}%</span>
                  <span class="mini-chip">ridge {item['ridge_success']*100:.1f}%</span>
                </div>
              </div>
              <p class="metric-subtle">Goal: {html.escape(item['goal_text'] or '—')}</p>
              <p class="metric-subtle">{html.escape(item['hidden_dims'])} · {item['epoch_count']} epochs · {item['train_sample_count']} train samples · {item['device']}</p>
              <div class="asset-inline-links">
                <a href="/ui/files?path={quote(_project_relpath(item['entry'].summary_path))}">Summary</a>
                <a href="/ui/files?path={quote(_project_relpath(item['entry'].root))}">Run root</a>
              </div>
            </article>
            """
        )

    goals_html = '<p class="empty">No NL goals recorded yet.</p>'
    if goals:
        goals_html = '<div class="asset-list">' + ''.join(
            f'<article class="asset-row"><h3 class="asset-title">Goal {index}</h3><p class="metric-subtle">{html.escape(goal)}</p></article>'
            for index, goal in enumerate(goals, start=1)
        ) + '</div>'

    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Home</a> / env / {html.escape(env_id)} / <a href="/env/{env_id}/rewards">reward programs</a> / {html.escape(label)}</p>
      <h1>{html.escape(env_id.upper())} {html.escape(label)}</h1>
      <p>{html.escape(_reward_kind_description('value-model-sweep'))}</p>
      <div class="hero-stats">
        <div class="stat-chip"><span class="stat-label">Runs</span><strong class="stat-value">{len(runs)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Sweep groups</span><strong class="stat-value">{len(metas)}</strong></div>
        <div class="stat-chip"><span class="stat-label">NL goals</span><strong class="stat-value">{len(goals)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Avg gain vs ridge</span><strong class="stat-value">{avg_delta_reward*100:+.1f}</strong></div>
        <div class="stat-chip"><span class="stat-label">Avg success gain</span><strong class="stat-value">{avg_delta_success*100:+.1f}</strong></div>
      </div>
      {_render_env_subnav(env_id=env_id, current_section='rewards')}
    </section>
    <section class="panel compact-panel">
      <h2>Best runs</h2>
      <div class="asset-list">{''.join(best_cards)}</div>
    </section>
    <section class="panel compact-panel">
      <h2>Comparison</h2>
      {chart_html}
    </section>
    <section class="panel compact-panel">
      <h2>Leaderboard</h2>
      {leaderboard_html}
    </section>
    <section class="panel compact-panel">
      <h2>NL goals</h2>
      {goals_html}
    </section>
    {f'<section class="panel compact-panel"><h2>Sweep campaigns</h2>{meta_html}</section>' if meta_html else ''}
    <section class="panel compact-panel">
      <h2>Recent runs</h2>
      <div class="asset-list">{''.join(recent_rows)}</div>
    </section>
    """
    return _shell(f"{env_id.upper()} {label} | Pattern Server", active_nav=f"env:{env_id}", body_html=body)


def _reward_value_model_run_rows(rewards: list[RewardResultEntry]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for entry in rewards:
        payload = _load_reward_summary(entry)
        eval_curve = payload.get("eval_curve") if isinstance(payload.get("eval_curve"), list) else []
        validation_outcomes = payload.get("validation_outcomes") if isinstance(payload.get("validation_outcomes"), list) else []
        goal_text = str(payload.get("goal_text") or "").strip()
        goals = [goal_text] if goal_text else []
        best_reward = float(payload.get("best_validation_mean_reward") or 0.0)
        best_success = float(payload.get("best_validation_success_rate") or 0.0)
        epochs = _reward_sweep_int(payload.get("epochs")) or len(eval_curve)
        best_epoch = 0
        for point in eval_curve:
            if not isinstance(point, dict):
                continue
            reward_value = float(point.get("validation_mean_reward") or 0.0)
            success_value = float(point.get("validation_success_rate") or 0.0)
            if reward_value == best_reward and success_value == best_success:
                best_epoch = _reward_sweep_int(point.get("epoch")) or best_epoch
                if best_epoch:
                    break
        runs.append(
            {
                "entry": entry,
                "payload": payload,
                "goals": goals,
                "goal_text": goal_text,
                "eval_curve": [point for point in eval_curve if isinstance(point, dict)],
                "validation_outcomes": [point for point in validation_outcomes if isinstance(point, dict)],
                "epochs": epochs,
                "best_epoch": best_epoch,
                "best_reward": best_reward,
                "best_success": best_success,
                "train_scene_count": _reward_sweep_int(payload.get("train_scene_count")) or 0,
                "train_sample_count": _reward_sweep_int(payload.get("train_sample_count")) or 0,
                "validation_scene_count": _reward_sweep_int(payload.get("validation_scene_count")) or 0,
                "batch_size": _reward_sweep_int(payload.get("batch_size")) or 0,
                "learning_rate": float(payload.get("learning_rate") or 0.0),
                "model_name": str(payload.get("model_name") or "").strip() or "—",
                "dsl_program": str(payload.get("dsl_program") or "").strip(),
            }
        )
    runs.sort(key=lambda item: (item["best_success"], item["best_reward"], item["entry"].updated_at), reverse=True)
    return runs


def _render_reward_value_model_runs_page(*, env_id: str, label: str, rewards: list[RewardResultEntry]) -> str:
    runs = _reward_value_model_run_rows(rewards)
    goals = sorted({goal for item in runs for goal in item["goals"] if goal})
    best_reward = max(runs, key=lambda item: (item["best_reward"], item["best_success"], item["entry"].updated_at)) if runs else None
    best_success = max(runs, key=lambda item: (item["best_success"], item["best_reward"], item["entry"].updated_at)) if runs else None
    longest_run = max(runs, key=lambda item: (item["epochs"], item["best_reward"], item["entry"].updated_at)) if runs else None
    best_reward_pct = best_reward["best_reward"] * 100.0 if best_reward else 0.0
    best_success_pct = best_success["best_success"] * 100.0 if best_success else 0.0

    comparison_series = []
    for index, item in enumerate(runs[:20], start=1):
        comparison_series.append((float(index), float(item["best_reward"] * 100.0), float(item["best_success"] * 100.0)))

    chart_html = '<p class="empty">No value-model runs yet.</p>'
    epoch_chart_html = ""
    if comparison_series:
        chart_html = _render_xy_line_chart(
            title="Best validation metrics by run",
            x_label="Rank",
            y_label="Score (%)",
            series=[
                {
                    "label": "Best reward",
                    "color": "#124f49",
                    "points": [(rank, reward) for rank, reward, _success in comparison_series],
                },
                {
                    "label": "Best success",
                    "color": "#a54d2d",
                    "dasharray": "6 4",
                    "points": [(rank, success) for rank, _reward, success in comparison_series],
                },
            ],
        )
    if best_success and best_success["eval_curve"]:
        epoch_chart_html = _render_xy_line_chart(
            title=f"Epoch curve · {best_success['entry'].run_id}",
            x_label="Epoch",
            y_label="Validation score (%)",
            series=[
                {
                    "label": "Mean reward",
                    "color": "#124f49",
                    "points": [
                        (float(_reward_sweep_int(point.get("epoch")) or 0), float(point.get("validation_mean_reward") or 0.0) * 100.0)
                        for point in best_success["eval_curve"]
                    ],
                },
                {
                    "label": "Success rate",
                    "color": "#a54d2d",
                    "dasharray": "6 4",
                    "points": [
                        (float(_reward_sweep_int(point.get("epoch")) or 0), float(point.get("validation_success_rate") or 0.0) * 100.0)
                        for point in best_success["eval_curve"]
                    ],
                },
            ],
        )

    best_cards: list[str] = []
    for title, item, metric_label in (
        ("Best validation reward", best_reward, f'{best_reward["best_reward"]*100:.1f}% reward · {best_reward["best_success"]*100:.1f}% success' if best_reward else "—"),
        ("Best validation success", best_success, f'{best_success["best_success"]*100:.1f}% success · {best_success["best_reward"]*100:.1f}% reward' if best_success else "—"),
        ("Longest run", longest_run, f'{longest_run["epochs"]} epochs · best epoch {longest_run["best_epoch"] or "—"}' if longest_run else "—"),
    ):
        if item is None:
            best_cards.append(f'<article class="asset-row"><h3 class="asset-title">{html.escape(title)}</h3><p class="metric-subtle">No runs yet.</p></article>')
            continue
        best_cards.append(
            f"""
            <article class="asset-row">
              <div class="asset-row-top">
                <div>
                  <h3 class="asset-title">{html.escape(title)}</h3>
                  <p class="metric-subtle"><a href="/ui/files?path={quote(_project_relpath(item['entry'].summary_path))}">{html.escape(item['entry'].run_id)}</a></p>
                </div>
                <div class="asset-meta">
                  <span class="mini-chip">{item['epochs']} epochs</span>
                  <span class="mini-chip">best epoch {item['best_epoch'] or '—'}</span>
                </div>
              </div>
              <p class="metric-subtle">{html.escape(metric_label)}</p>
              <p class="metric-subtle">Goal: {html.escape(item['goal_text'] or '—')}</p>
            </article>
            """
        )

    leaderboard_rows = []
    for index, item in enumerate(runs[:24], start=1):
        leaderboard_rows.append(
            f"<tr><td>{index}</td><td>{html.escape(item['entry'].run_id)}</td><td>{item['best_reward']*100:.1f}%</td><td>{item['best_success']*100:.1f}%</td><td>{item['epochs']}</td><td>{item['best_epoch'] or '—'}</td><td>{item['train_scene_count']}</td><td>{item['train_sample_count']}</td><td>{item['validation_scene_count']}</td><td>{html.escape(item['goal_text'] or '—')}</td></tr>"
        )
    leaderboard_html = (
        '<div class="scroll-table compact-scroll-table"><table class="data-table compact-table"><thead><tr><th>#</th><th>Run</th><th>Best reward</th><th>Best success</th><th>Epochs</th><th>Best epoch</th><th>Train scenes</th><th>Train samples</th><th>Val scenes</th><th>Goal</th></tr></thead><tbody>'
        + ''.join(leaderboard_rows)
        + '</tbody></table></div>'
        if leaderboard_rows
        else '<p class="empty">No value-model runs yet.</p>'
    )

    goals_html = '<p class="empty">No NL goals recorded yet.</p>'
    if goals:
        goals_html = '<div class="asset-list">' + ''.join(
            f'<article class="asset-row"><h3 class="asset-title">Goal {index}</h3><p class="metric-subtle">{html.escape(goal)}</p></article>'
            for index, goal in enumerate(goals, start=1)
        ) + '</div>'

    recent_rows = []
    for item in runs[:24]:
        recent_rows.append(
            f"""
            <article class="asset-row">
              <div class="asset-row-top">
                <div>
                  <h3 class="asset-title">{html.escape(item['entry'].run_id)}</h3>
                  <p class="metric-subtle">Updated {_display_time(item['entry'].updated_at)}</p>
                </div>
                <div class="asset-meta">
                  <span class="mini-chip">reward {item['best_reward']*100:.1f}%</span>
                  <span class="mini-chip">success {item['best_success']*100:.1f}%</span>
                </div>
              </div>
              <p class="metric-subtle">Goal: {html.escape(item['goal_text'] or '—')}</p>
              <p class="metric-subtle">{item['epochs']} epochs · batch {item['batch_size']} · lr {item['learning_rate']:.5f} · {html.escape(item['model_name'])}</p>
              <div class="asset-inline-links">
                <a href="/ui/files?path={quote(_project_relpath(item['entry'].summary_path))}">Summary</a>
                <a href="/ui/files?path={quote(_project_relpath(item['entry'].root))}">Run root</a>
              </div>
            </article>
            """
        )

    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Home</a> / env / {html.escape(env_id)} / <a href="/env/{env_id}/rewards">reward programs</a> / {html.escape(label)}</p>
      <h1>{html.escape(env_id.upper())} {html.escape(label)}</h1>
      <p>{html.escape(_reward_kind_description('value-model'))}</p>
      <div class="hero-stats">
        <div class="stat-chip"><span class="stat-label">Runs</span><strong class="stat-value">{len(runs)}</strong></div>
        <div class="stat-chip"><span class="stat-label">NL goals</span><strong class="stat-value">{len(goals)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Best reward</span><strong class="stat-value">{best_reward_pct:.1f}</strong></div>
        <div class="stat-chip"><span class="stat-label">Best success</span><strong class="stat-value">{best_success_pct:.1f}</strong></div>
      </div>
      {_render_env_subnav(env_id=env_id, current_section='rewards')}
    </section>
    <section class="panel compact-panel">
      <h2>Best runs</h2>
      <div class="asset-list">{''.join(best_cards)}</div>
    </section>
    <section class="panel compact-panel">
      <h2>Comparison</h2>
      {chart_html}
      {epoch_chart_html}
    </section>
    <section class="panel compact-panel">
      <h2>Leaderboard</h2>
      {leaderboard_html}
    </section>
    <section class="panel compact-panel">
      <h2>NL goals</h2>
      {goals_html}
    </section>
    <section class="panel compact-panel">
      <h2>Recent runs</h2>
      <div class="asset-list">{''.join(recent_rows)}</div>
    </section>
    """
    return _shell(f"{env_id.upper()} {label} | Pattern Server", active_nav=f"env:{env_id}", body_html=body)


def _reward_optimization_run_rows(rewards: list[RewardResultEntry]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for entry in rewards:
        payload = _load_reward_summary(entry)
        aggregates = payload.get("budget_aggregates") if isinstance(payload.get("budget_aggregates"), list) else []
        goals = []
        raw_goals = payload.get("goal_texts")
        if isinstance(raw_goals, list):
            goals = [str(item).strip() for item in raw_goals if str(item).strip()]
        if not goals:
            goal_text = str(payload.get("goal_text") or "").strip()
            if goal_text:
                goals = [line.strip(" -*") for line in goal_text.splitlines() if line.strip()]
        by_kind: dict[str, list[dict[str, Any]]] = {}
        for row in aggregates:
            if not isinstance(row, dict):
                continue
            kind = str(row.get("reward_kind") or "").strip() or "unknown"
            by_kind.setdefault(kind, []).append(row)
        best_success = {}
        best_reward = {}
        for kind, rows in by_kind.items():
            ordered = sorted(
                rows,
                key=lambda item: (
                    _reward_sweep_int(item.get("budget")) or 0,
                    float(item.get("success_rate") or 0.0),
                    float(item.get("mean_reward") or 0.0),
                ),
            )
            best = ordered[-1] if ordered else {}
            best_success[kind] = float(best.get("success_rate") or 0.0)
            best_reward[kind] = float(best.get("mean_reward") or 0.0)
        budgets = sorted(
            {
                budget
                for row in aggregates
                if isinstance(row, dict)
                for budget in [(_reward_sweep_int(row.get("budget")) or 0)]
                if budget > 0
            }
        )
        runs.append(
            {
                "entry": entry,
                "payload": payload,
                "goals": goals,
                "goal_text": goals[0] if goals else "",
                "budgets": budgets,
                "best_success": best_success,
                "best_reward": best_reward,
                "scene_count": _reward_sweep_int(payload.get("scene_count")) or 0,
                "seed_count": _reward_sweep_int(payload.get("seed_count")) or 0,
                "synthesized_success": best_success.get("synthesized", 0.0),
                "binary_success": best_success.get("binary", 0.0),
                "synthesized_reward": best_reward.get("synthesized", 0.0),
                "binary_reward": best_reward.get("binary", 0.0),
                "dsl_program": str(payload.get("dsl_program") or "").strip(),
            }
        )
    runs.sort(
        key=lambda item: (
            item["synthesized_success"],
            item["binary_success"],
            item["synthesized_reward"],
            item["entry"].updated_at,
        ),
        reverse=True,
    )
    return runs


def _render_reward_optimization_runs_page(*, env_id: str, label: str, rewards: list[RewardResultEntry]) -> str:
    runs = _reward_optimization_run_rows(rewards)
    goals = sorted({goal for item in runs for goal in item["goals"] if goal})
    best_synth = max(runs, key=lambda item: (item["synthesized_success"], item["synthesized_reward"], item["entry"].updated_at)) if runs else None
    best_binary = max(runs, key=lambda item: (item["binary_success"], item["binary_reward"], item["entry"].updated_at)) if runs else None
    best_reward = max(runs, key=lambda item: (item["synthesized_reward"], item["synthesized_success"], item["entry"].updated_at)) if runs else None
    best_synth_success_pct = best_synth["synthesized_success"] * 100.0 if best_synth else 0.0
    best_binary_success_pct = best_binary["binary_success"] * 100.0 if best_binary else 0.0

    comparison_series = []
    reward_series = []
    for index, item in enumerate(runs[:20], start=1):
        comparison_series.append((float(index), float(item["synthesized_success"] * 100.0), float(item["binary_success"] * 100.0)))
        reward_series.append((float(index), float(item["synthesized_reward"]), float(item["binary_reward"])))

    success_chart_html = '<p class="empty">No optimization runs yet.</p>'
    reward_chart_html = ""
    if comparison_series:
        success_chart_html = _render_xy_line_chart(
            title="Best success by run",
            x_label="Rank",
            y_label="Success rate (%)",
            series=[
                {
                    "label": "Synthesized",
                    "color": "#124f49",
                    "points": [(rank, synth) for rank, synth, _binary in comparison_series],
                },
                {
                    "label": "Binary",
                    "color": "#a54d2d",
                    "dasharray": "6 4",
                    "points": [(rank, binary) for rank, _synth, binary in comparison_series],
                },
            ],
        )
        reward_chart_html = _render_xy_line_chart(
            title="Best reward by run",
            x_label="Rank",
            y_label="Mean reward",
            series=[
                {
                    "label": "Synthesized",
                    "color": "#124f49",
                    "points": [(rank, synth) for rank, synth, _binary in reward_series],
                },
                {
                    "label": "Binary",
                    "color": "#a54d2d",
                    "dasharray": "6 4",
                    "points": [(rank, binary) for rank, _synth, binary in reward_series],
                },
            ],
        )

    best_cards: list[str] = []
    for title, item, metric_label in (
        ("Best synthesized success", best_synth, f'{best_synth["synthesized_success"]*100:.1f}% success · {best_synth["synthesized_reward"]:.3f} reward' if best_synth else "—"),
        ("Best binary success", best_binary, f'{best_binary["binary_success"]*100:.1f}% success · {best_binary["binary_reward"]:.3f} reward' if best_binary else "—"),
        ("Best synthesized reward", best_reward, f'{best_reward["synthesized_reward"]:.3f} reward · {best_reward["synthesized_success"]*100:.1f}% success' if best_reward else "—"),
    ):
        if item is None:
            best_cards.append(f'<article class="asset-row"><h3 class="asset-title">{html.escape(title)}</h3><p class="metric-subtle">No runs yet.</p></article>')
            continue
        best_cards.append(
            f"""
            <article class="asset-row">
              <div class="asset-row-top">
                <div>
                  <h3 class="asset-title">{html.escape(title)}</h3>
                  <p class="metric-subtle"><a href="/ui/files?path={quote(_project_relpath(item['entry'].summary_path))}">{html.escape(item['entry'].run_id)}</a></p>
                </div>
                <div class="asset-meta">
                  <span class="mini-chip">{len(item['budgets'])} budgets</span>
                  <span class="mini-chip">{item['scene_count']} scenes</span>
                </div>
              </div>
              <p class="metric-subtle">{html.escape(metric_label)}</p>
              <p class="metric-subtle">Goal: {html.escape(item['goal_text'] or '—')}</p>
            </article>
            """
        )

    leaderboard_rows = []
    for index, item in enumerate(runs[:24], start=1):
        leaderboard_rows.append(
            f"<tr><td>{index}</td><td>{html.escape(item['entry'].run_id)}</td><td>{item['synthesized_success']*100:.1f}%</td><td>{item['binary_success']*100:.1f}%</td><td>{item['synthesized_reward']:.3f}</td><td>{item['binary_reward']:.3f}</td><td>{', '.join(str(v) for v in item['budgets']) or '—'}</td><td>{html.escape(item['goal_text'] or '—')}</td></tr>"
        )
    leaderboard_html = (
        '<div class="scroll-table compact-scroll-table"><table class="data-table compact-table"><thead><tr><th>#</th><th>Run</th><th>Synth success</th><th>Binary success</th><th>Synth reward</th><th>Binary reward</th><th>Budgets</th><th>Goal</th></tr></thead><tbody>'
        + ''.join(leaderboard_rows)
        + '</tbody></table></div>'
        if leaderboard_rows
        else '<p class="empty">No optimization runs yet.</p>'
    )

    goals_html = '<p class="empty">No NL goals recorded yet.</p>'
    if goals:
        goals_html = '<div class="asset-list">' + ''.join(
            f'<article class="asset-row"><h3 class="asset-title">Goal {index}</h3><p class="metric-subtle">{html.escape(goal)}</p></article>'
            for index, goal in enumerate(goals, start=1)
        ) + '</div>'

    recent_rows = []
    for item in runs[:24]:
        recent_rows.append(
            f"""
            <article class="asset-row">
              <div class="asset-row-top">
                <div>
                  <h3 class="asset-title">{html.escape(item['entry'].run_id)}</h3>
                  <p class="metric-subtle">Updated {_display_time(item['entry'].updated_at)}</p>
                </div>
                <div class="asset-meta">
                  <span class="mini-chip">Synth {item['synthesized_success']*100:.1f}%</span>
                  <span class="mini-chip">Binary {item['binary_success']*100:.1f}%</span>
                </div>
              </div>
              <p class="metric-subtle">Goal: {html.escape(item['goal_text'] or '—')}</p>
              <p class="metric-subtle">{item['scene_count']} scenes · {item['seed_count']} seeds · budgets {', '.join(str(v) for v in item['budgets']) or '—'}</p>
              <div class="asset-inline-links">
                <a href="/ui/files?path={quote(_project_relpath(item['entry'].summary_path))}">Summary</a>
                <a href="/ui/files?path={quote(_project_relpath(item['entry'].root))}">Run root</a>
              </div>
            </article>
            """
        )

    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Home</a> / env / {html.escape(env_id)} / <a href="/env/{env_id}/rewards">reward programs</a> / {html.escape(label)}</p>
      <h1>{html.escape(env_id.upper())} {html.escape(label)}</h1>
      <p>{html.escape(_reward_kind_description('optimization'))}</p>
      <div class="hero-stats">
        <div class="stat-chip"><span class="stat-label">Runs</span><strong class="stat-value">{len(runs)}</strong></div>
        <div class="stat-chip"><span class="stat-label">NL goals</span><strong class="stat-value">{len(goals)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Best synth success</span><strong class="stat-value">{best_synth_success_pct:.1f}</strong></div>
        <div class="stat-chip"><span class="stat-label">Best binary success</span><strong class="stat-value">{best_binary_success_pct:.1f}</strong></div>
      </div>
      {_render_env_subnav(env_id=env_id, current_section='rewards')}
    </section>
    <section class="panel compact-panel">
      <h2>Best runs</h2>
      <div class="asset-list">{''.join(best_cards)}</div>
    </section>
    <section class="panel compact-panel">
      <h2>Comparison</h2>
      {success_chart_html}
      {reward_chart_html}
    </section>
    <section class="panel compact-panel">
      <h2>Leaderboard</h2>
      {leaderboard_html}
    </section>
    <section class="panel compact-panel">
      <h2>NL goals</h2>
      {goals_html}
    </section>
    <section class="panel compact-panel">
      <h2>Recent runs</h2>
      <div class="asset-list">{''.join(recent_rows)}</div>
    </section>
    """
    return _shell(f"{env_id.upper()} {label} | Pattern Server", active_nav=f"env:{env_id}", body_html=body)


def _render_recent_reward_experiments_panel(*, project_root: Path) -> str:
    rewards = list_environment_reward_results(project_root, "pooltool")
    return f"""
    <section class="panel compact-panel">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Reward Programs</p>
          <h2>Reward Experiments</h2>
          <p class="metric-subtle">Open an experiment type to see all runs and results.</p>
        </div>
      </div>
      {_reward_kind_cards(rewards=rewards, env_id="pooltool")}
    </section>
    """


def _render_dashboard_links_panel() -> str:
    return """
    <section class="panel compact-panel panel-static home-nav-panel">
      <nav class="home-link-strip" aria-label="Other pages">
        <a class="home-link-pill" href="/learning"><span class="home-link-pill-label">JOBS.md</span>Learning</a>
        <a class="home-link-pill" href="/statistics"><span class="home-link-pill-label">All envs</span>Statistics</a>
        <a class="home-link-pill" href="/analysis"><span class="home-link-pill-label">Direct only</span>Analysis</a>
        <a class="home-link-pill" href="/ablations"><span class="home-link-pill-label">Tracked queue</span>Ablations</a>
        <a class="home-link-pill" href="/rewards"><span class="home-link-pill-label">PoolTool</span>Rewards</a>
      </nav>
    </section>
    """


def _render_home_experiment_links_panel() -> str:
    return """
    <section class="panel compact-panel">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Experiments</p>
          <h2>Experiment Dashboards</h2>
          <p class="metric-subtle">Open the dedicated pages for reward-program and library-ablation results. The home page keeps this lightweight to stay responsive during active runs.</p>
        </div>
      </div>
      <div class="asset-list">
        <article class="asset-row">
          <div class="asset-row-top">
            <div>
              <h3 class="asset-title"><a href="/rewards">Reward Programs</a></h3>
              <p class="metric-subtle">Standalone reward-program dashboard with the current PoolTool views.</p>
            </div>
            <div class="asset-meta">
              <span class="mini-chip">PoolTool</span>
            </div>
          </div>
        </article>
        <article class="asset-row">
          <div class="asset-row-top">
            <div>
              <h3 class="asset-title"><a href="/statistics">Label Statistics</a></h3>
              <p class="metric-subtle">Per-label code-generation table across all discovered run artifacts, grouped by environment.</p>
            </div>
            <div class="asset-meta">
              <span class="mini-chip">All envs</span>
            </div>
          </div>
        </article>
        <article class="asset-row">
          <div class="asset-row-top">
            <div>
              <h3 class="asset-title"><a href="/ablations">Library Ablations</a></h3>
              <p class="metric-subtle">Tracked required-ablation queue with one supervised, unsupervised, and automatic source library per environment.</p>
            </div>
            <div class="asset-meta">
              <span class="mini-chip">All envs</span>
            </div>
          </div>
        </article>
      </div>
    </section>
    """


def _effective_client_job_rows(*, client: ClientRecord, jobs: list[JobRecord]) -> list[JobRecord]:
    active_jobs = [
        job
        for job in jobs
        if str(job.assigned_client_id or "").strip() == client.client_id
        and job.status.value in ACTIVE_QUEUE_STATUSES | {"pausing", "resuming"}
    ]
    active_jobs.sort(key=_job_last_activity, reverse=True)
    if active_jobs:
        return active_jobs
    current_job_ids = set(client.current_jobs or [])
    if not current_job_ids:
        return []
    fallback_jobs = [job for job in jobs if job.job_id in current_job_ids]
    fallback_jobs.sort(key=_job_last_activity, reverse=True)
    return fallback_jobs


def _render_client_current_jobs(*, client_jobs: list[JobRecord], job_lookup: dict[str, JobRecord], now: datetime | None = None) -> str:
    current = _coerce_datetime(now) or datetime.now(tz=DISPLAY_TZ)
    if not client_jobs:
        return '<div class="client-note">Idle.</div>'
    rows: list[str] = []
    visible_jobs = list(client_jobs[:3])
    hidden_count = max(0, len(client_jobs) - len(visible_jobs))
    for job in visible_jobs:
        job = job_lookup.get(job.job_id) or job
        rows.append(
            f"""
            <article class="client-job-row">
              <div class="client-job-row-top">
                <strong><a href="/jobs/{html.escape(job.job_id)}">{html.escape(_queue_job_title(job))}</a></strong>
                <p class="metric-subtle">{html.escape(_job_env_id(job))} · {html.escape(_job_last_activity_age(job, now=current))}</p>
              </div>
              <div class="asset-meta">
                {_pill(job.status.value, job.status.value)}
              </div>
            </article>
            """
        )
    more_html = f'<div class="client-job-more">+{hidden_count} more</div>' if hidden_count else ""
    return f'<div class="client-job-list">{"".join(rows)}{more_html}</div>'


def _render_client_active_jobs(*, client: ClientRecord, jobs: list[JobRecord], project_root: Path) -> str:
    if not jobs:
        return '<p class="empty">No active jobs on this worker.</p>'
    rows = "".join(
        f"""
        <article class="asset-row">
          <div class="asset-row-top">
            <div>
              <h3 class="asset-title"><a href="/jobs/{html.escape(job.job_id)}">{html.escape(job.spec.action)}</a></h3>
              <p class="metric-subtle">{html.escape(_job_env_id(job))} · {_display_time(job.created_at)}</p>
            </div>
            <div class="asset-meta">
              {_pill(job.status.value, job.status.value)}
            </div>
          </div>
          {_job_progress_html(job, project_root=project_root)}
        </article>
        """
        for job in sorted(jobs, key=_queue_sort_key)
    )
    return '<div class="asset-list">' + rows + "</div>"


def _render_client_job_logs(
    *,
    client: ClientRecord,
    jobs: list[JobRecord],
    log_lookup: dict[str, list[str]],
    project_root: Path,
) -> str:
    if not jobs:
        return '<p class="empty">No jobs recorded for this worker yet.</p>'
    cards: list[str] = []
    for job in jobs:
        lines = log_lookup.get(job.job_id) or list(job.recent_logs)
        log_text = "\n".join(lines).strip()
        if not log_text:
            log_text = "(no logs captured yet)"
        cards.append(
            f"""
            <article class="artifact-card">
              <div class="asset-row-top">
                <div>
                  <h3><a href="/jobs/{html.escape(job.job_id)}">{html.escape(job.spec.action)}</a></h3>
                  <p class="metric-subtle">{html.escape(_job_env_id(job))} · {_display_time(job.updated_at)}</p>
                </div>
                <div class="asset-meta">
                  {_pill(job.status.value, job.status.value)}
                  <span class="mini-chip"><code>{html.escape(job.job_id[:8])}</code></span>
                </div>
              </div>
              {_job_progress_html(job, project_root=project_root)}
              <pre class="log-block">{html.escape(log_text)}</pre>
            </article>
            """
        )
    return '<div class="artifact-list">' + "".join(cards) + "</div>"


def render_home_page(
    *,
    public_access: PublicAccessState,
    clients: list[ClientRecord],
    jobs: list[JobRecord],
    project_root: Path,
) -> str:
    body = f"""
    <section class="hero">
      <p class="breadcrumb">Home</p>
      <h1>Clients</h1>
    </section>
    {_render_clients_summary_panel(clients=clients)}
    {_render_public_access_panel(public_access, next_url="/")}
    {_render_clients_panel(clients, jobs=jobs, next_url="/")}
    """
    return _shell("Pattern Server", active_nav="clients", body_html=body)


def render_learning_home_page(
    *,
    public_access: PublicAccessState,
    clients: list[ClientRecord],
    jobs: list[JobRecord],
    project_root: Path,
) -> str:
    counts = Counter(job.status.value for job in jobs)
    learning_stats = _learning_overview_stats(jobs=jobs, project_root=project_root)
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Home</a> / learning</p>
      <h1>Learning Runs</h1>
      <p>Required learning coverage, dependent evaluations, and the tracked progress from JOBS.md.</p>
    </section>
    <section class="hero-stats">
      <div class="stat-chip"><span class="stat-label">Jobs</span><strong class="stat-value">{len(jobs)}</strong></div>
      <div class="stat-chip"><span class="stat-label">Queued</span><strong class="stat-value">{counts.get("queued", 0)}</strong></div>
      <div class="stat-chip"><span class="stat-label">Running</span><strong class="stat-value">{sum(1 for job in jobs if job.status.value in ACTIVE_QUEUE_STATUSES)}</strong></div>
      <div class="stat-chip"><span class="stat-label">Clients online</span><strong class="stat-value">{sum(1 for client in clients if client.status.value == "online")}</strong></div>
    </section>
    <section class="hero-stats">
      <div class="stat-chip"><span class="stat-label">Learning runs</span><strong class="stat-value">{learning_stats.get("run_count", 0):,}</strong></div>
      <div class="stat-chip"><span class="stat-label">Total labels</span><strong class="stat-value">{learning_stats.get("label_count", 0):,}</strong></div>
      <div class="stat-chip"><span class="stat-label">Labels complete</span><strong class="stat-value">{learning_stats.get("completed_label_count", 0):,}</strong></div>
      <div class="stat-chip"><span class="stat-label">Candidate codes</span><strong class="stat-value">{learning_stats.get("candidate_count", 0):,}</strong></div>
      <div class="stat-chip"><span class="stat-label">Code lines</span><strong class="stat-value">{learning_stats.get("code_line_count", 0):,}</strong></div>
    </section>
    {_render_required_overall_progress_bar(jobs=jobs, project_root=project_root)}
    {_render_optimized_eval_launcher_panel(jobs=jobs, project_root=project_root)}
    {_render_required_coverage_panel(jobs, project_root=project_root)}
    """
    return _shell("Pattern Server", active_nav="learning", body_html=body)


OPTIMIZED_EVAL_RUN_PREFIX = "optimized-ensemble-eval"


def _optimized_eval_run_id(job: JobRecord) -> str:
    metadata = dict(job.spec.metadata or {})
    run_id = str(metadata.get("library_evaluation_run_id") or "").strip()
    if run_id:
        return run_id
    request_payload = metadata.get("request")
    if isinstance(request_payload, dict):
        return str(request_payload.get("run_id") or "").strip()
    return ""


def _is_optimized_eval_job(job: JobRecord) -> bool:
    if str(job.spec.action or "").strip() != "evaluate-library-case":
        return False
    return _optimized_eval_run_id(job).startswith(f"{OPTIMIZED_EVAL_RUN_PREFIX}-")


def _optimized_eval_job_kind(job: JobRecord) -> str:
    run_id = _optimized_eval_run_id(job)
    if "-unsupervised-" in run_id:
        return "unsupervised"
    if "-supervised-" in run_id:
        return "supervised"
    metadata = dict(job.spec.metadata or {})
    pattern_run_dir = str(metadata.get("pattern_run_dir") or "").strip().lower()
    if "/unsupervised" in pattern_run_dir:
        return "unsupervised"
    if "/supervised" in pattern_run_dir:
        return "supervised"
    return str(metadata.get("library_ablation_kind") or "").strip().lower()


def _optimized_eval_jobs_for_env(jobs: list[JobRecord], *, env_id: str) -> list[JobRecord]:
    filtered = [
        job
        for job in jobs
        if _is_optimized_eval_job(job)
        and str(job.spec.metadata.get("env_id") or "").strip().lower() == env_id
    ]
    filtered.sort(key=lambda item: item.updated_at, reverse=True)
    return filtered


def _optimized_eval_source_summary(project_root: Path, *, env_id: str, kind: str) -> dict[str, Any]:
    try:
        library_path = resolve_canonical_library(project_root, env_id, kind)
    except Exception as exc:
        return {
            "env_id": env_id,
            "kind": kind,
            "error": str(exc),
            "source_root": None,
            "library_path": None,
            "detector_count": 0,
            "label_count": 0,
        }
    summary = load_cached_canonical_library_summary(project_root, env_id, kind)
    rows = list(summary.get("rows") or []) if isinstance(summary, dict) else []
    labels = {
        str(row.get("label") or "").strip()
        for row in rows
        if isinstance(row, dict) and str(row.get("label") or "").strip()
    }
    return {
        "env_id": env_id,
        "kind": kind,
        "error": str(summary.get("error") or "").strip() if isinstance(summary, dict) else "",
        "source_root": library_path.parent,
        "library_path": library_path,
        "detector_count": int(summary.get("detector_count") or len(rows)) if isinstance(summary, dict) else len(rows),
        "label_count": len(labels),
    }


def _render_optimized_eval_launcher_panel(*, jobs: list[JobRecord], project_root: Path) -> str:
    env_cards: list[str] = []
    total_active = 0
    total_recent = 0
    for env_id in SUPPORTED_ENVS:
        supervised = _optimized_eval_source_summary(project_root, env_id=env_id, kind="supervised")
        unsupervised = _optimized_eval_source_summary(project_root, env_id=env_id, kind="unsupervised")
        recent_jobs = _optimized_eval_jobs_for_env(jobs, env_id=env_id)
        total_recent += len(recent_jobs)
        active_jobs = [job for job in recent_jobs if job.status.value in ACTIVE_QUEUE_STATUSES]
        total_active += len(active_jobs)
        launch_disabled = bool(supervised.get("error")) or bool(unsupervised.get("error"))
        source_rows = []
        for source in (supervised, unsupervised):
            kind = str(source.get("kind") or "").strip().title()
            error = str(source.get("error") or "").strip()
            if error:
                source_rows.append(
                    f"""
                    <tr>
                      <td>{html.escape(kind)}</td>
                      <td colspan="3"><span class="metric-subtle">{html.escape(error)}</span></td>
                    </tr>
                    """
                )
                continue
            source_rows.append(
                f"""
                <tr>
                  <td>{html.escape(kind)}</td>
                  <td>{int(source.get("label_count") or 0)}</td>
                  <td>{int(source.get("detector_count") or 0)}</td>
                  <td><code>{html.escape(Path(str(source.get("source_root"))).name)}</code></td>
                </tr>
                """
            )
        job_rows = "".join(
            f"""
            <article class="reward-client-job">
              <div class="reward-client-job-main">
                <strong><a href="/jobs/{html.escape(job.job_id)}">{html.escape(_optimized_eval_run_id(job) or job.job_id)}</a></strong>
                <span class="reward-client-job-meta">{html.escape(_optimized_eval_job_kind(job) or 'unknown')} · {html.escape(_job_last_activity_age(job))}</span>
              </div>
              {_pill(job.status.value, job.status.value)}
            </article>
            """
            for job in recent_jobs[:4]
        )
        env_cards.append(
            f"""
            <article class="reward-workbench-card">
              <div class="analysis-env-head">
                <div><p class="section-kicker">Remote eval pair</p><h3>{html.escape(env_id.upper())}</h3></div>
                <div class="stats-inline">
                  <span class="mini-chip">{len(active_jobs)} active</span>
                  <span class="mini-chip">{len(recent_jobs)} recent</span>
                </div>
              </div>
              <p class="metric-subtle">Launches two remote <code>evaluate-library-case</code> jobs against the current optimized canonical libraries for this environment: one supervised and one unsupervised.</p>
              <table class="data-table">
                <thead><tr><th>Kind</th><th>Labels</th><th>Detectors</th><th>Source</th></tr></thead>
                <tbody>{''.join(source_rows)}</tbody>
              </table>
              <form method="post" action="/ui/learning/optimized-evals/create" class="analysis-inline-form" style="margin-top:0.9rem;">
                <input type="hidden" name="next" value="/learning">
                <input type="hidden" name="env_id" value="{html.escape(env_id)}">
                <button class="secondary-button" type="submit" {"disabled" if launch_disabled else ""}>Launch 2 remote evals</button>
              </form>
              <div style="margin-top:0.9rem;">
                {job_rows if job_rows else '<p class="empty inline">No optimized ensemble eval jobs launched yet.</p>'}
              </div>
            </article>
            """
        )
    return f"""
    <section class="panel compact-panel">
      <div class="card-topline">
        <div><p class="section-kicker">Optimized libraries</p><h2>Canonical Ensemble Evaluations</h2></div>
        <div class="stats-inline">
          <span class="mini-chip">{total_active} active</span>
          <span class="mini-chip">{total_recent} total launched</span>
        </div>
      </div>
      <p class="metric-subtle">These controls launch remote evaluation jobs for the current optimized canonical libraries only. Each launch creates exactly two jobs for the chosen environment: supervised and unsupervised.</p>
      <form method="post" action="/ui/learning/optimized-evals/create" class="analysis-inline-form" style="margin:0.8rem 0 1rem 0;">
        <input type="hidden" name="next" value="/learning">
        <input type="hidden" name="env_id" value="all">
        <button class="secondary-button" type="submit">Launch All 8 Remote Eval Jobs</button>
      </form>
      <div class="reward-workbench-grid">{''.join(env_cards)}</div>
    </section>
    """


def _render_clients_summary_panel(*, clients: list[ClientRecord]) -> str:
    active_clients = _dashboard_active_clients(clients)
    online_clients = [client for client in active_clients if client.status.value == "online"]
    direct_clients = [client for client in active_clients if client.kind.value == "direct"]
    remote_clients = [client for client in active_clients if client.kind.value == "remote"]
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="hero-stats">
        <div class="stat-chip"><span class="stat-label">Clients</span><strong class="stat-value">{len(active_clients)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Online</span><strong class="stat-value">{len(online_clients)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Remote</span><strong class="stat-value">{len(remote_clients)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Direct</span><strong class="stat-value">{len(direct_clients)}</strong></div>
      </div>
    </section>
    """


def render_clients_home_page(*, clients: list[ClientRecord], jobs: list[JobRecord]) -> str:
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Learning</a> / clients</p>
      <h1>Clients</h1>
      <p>Live worker inventory, capacity, and attached jobs across remote and direct clients.</p>
    </section>
    {_render_clients_summary_panel(clients=clients)}
    {_render_clients_panel(clients, jobs=jobs, next_url="/clients")}
    """
    return _shell("Clients | Pattern Server", active_nav="clients", body_html=body)


def _reward_default_base_url() -> str:
    for name in ("PATTERN_LLM_BASE_URL", "JOB_LLM_BASE_URL", "OPENAI_BASE_URL"):
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return "http://127.0.0.1:8788/v1"


def _reward_default_timeout() -> int:
    for name in ("PATTERN_MODEL_TIMEOUT", "JOB_MODEL_TIMEOUT"):
        value = os.environ.get(name)
        if value is None or not value.strip():
            continue
        try:
            return max(0, int(value.strip()))
        except ValueError:
            continue
    return 120


def _reward_default_model() -> str:
    for name in ("PATTERN_LLM_MODEL", "JOB_LLM_MODEL", "OPENAI_MODEL"):
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return ""


def _reward_direct_client_sort_key(client: ClientRecord) -> tuple[int, int, int, float, str]:
    display_name = str(client.display_name or "").strip().lower()
    tags = {str(item).strip().lower() for item in (client.capabilities.tags or [])}
    return (
        0 if "reward" in display_name else 1,
        0 if "local" in tags else 1,
        -(_coerce_datetime(client.last_heartbeat).timestamp() if _coerce_datetime(client.last_heartbeat) else 0.0),
        0 if client.current_jobs else 1,
        _client_title(client).lower(),
    )


def _reward_direct_clients(clients: list[ClientRecord]) -> list[ClientRecord]:
    collapsed: dict[str, ClientRecord] = {}
    for client in _dashboard_active_clients(clients):
        if client.kind.value != "direct":
            continue
        envs = {str(item).strip().lower() for item in (client.capabilities.environments or [])}
        if "pooltool" not in envs:
            continue
        dedupe_key = str(client.direct_url or "").strip().lower() or _client_title(client).strip().lower() or client.client_id
        existing = collapsed.get(dedupe_key)
        if existing is None or _reward_direct_client_sort_key(client) < _reward_direct_client_sort_key(existing):
            collapsed[dedupe_key] = client
    return sorted(collapsed.values(), key=_reward_direct_client_sort_key)


def _reward_preferred_direct_client(clients: list[ClientRecord]) -> ClientRecord | None:
    direct_clients = _reward_direct_clients(clients)
    reward_direct = next(
        (
            client
            for client in direct_clients
            if "reward" in str(client.display_name or "").strip().lower()
        ),
        None,
    )
    if reward_direct is not None:
        return reward_direct
    local_direct = next(
        (
            client
            for client in direct_clients
            if "local" in set(client.capabilities.tags or [])
        ),
        None,
    )
    return local_direct or (direct_clients[0] if direct_clients else None)


def _reward_source_run_options(project_root: Path) -> list[LibraryEntry]:
    return [
        entry
        for entry in list_environment_libraries(project_root, "pooltool")
        if str(entry.workflow_kind or "").strip().lower() == "supervised"
    ]


def _reward_scene_options(project_root: Path, *, limit: int = 48) -> list[dict[str, str]]:
    root = project_root / "data" / "patterns" / "pooltool" / "reward_program_experiments"
    rows: list[dict[str, str]] = []
    if not root.is_dir():
        return rows
    scene_paths = sorted(
        root.glob("**/scene.json"),
        key=lambda item: item.stat().st_mtime if item.exists() else 0.0,
        reverse=True,
    )
    for scene_path in scene_paths[:limit]:
        try:
            payload = json.loads(scene_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        scene_id = str(payload.get("scene_id") or scene_path.parent.name)
        source_run = next((part for part in scene_path.parts[::-1] if part.startswith("required-") or part.startswith("2026")), scene_path.parents[1].name)
        rows.append(
            {
                "path": str(scene_path),
                "label": f"{source_run} · {scene_id}",
                "meta": scene_path.parent.as_posix().replace(str(project_root), "").lstrip("/"),
            }
        )
    return rows


def _load_reward_summary_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _reward_job_records(jobs: list[JobRecord]) -> list[JobRecord]:
    return [
        job
        for job in jobs
        if str(job.spec.action or "").strip() in {
            "pooltool-reward-optimize",
            "pooltool-reward-value-model",
            "pooltool-reward-single-scene",
        }
    ]


def _reward_run_id(job: JobRecord) -> str:
    run_id = str(job.spec.metadata.get("reward_program_run_id") or "").strip()
    return run_id or _queue_job_title(job)


def _reward_client_job_rows(*, client: ClientRecord, jobs: list[JobRecord]) -> tuple[list[JobRecord], list[JobRecord]]:
    active_jobs: list[JobRecord] = []
    pending_jobs: list[JobRecord] = []
    for job in jobs:
        preferred_client_id = str(job.spec.metadata.get("preferred_client_id") or "").strip()
        if str(job.assigned_client_id or "").strip() == client.client_id and job.status.value in ACTIVE_QUEUE_STATUSES | {"pausing", "resuming"}:
            active_jobs.append(job)
            continue
        if preferred_client_id == client.client_id and job.status.value in {"queued", "paused"}:
            pending_jobs.append(job)
    active_jobs.sort(key=_job_last_activity, reverse=True)
    pending_jobs.sort(key=_job_last_activity, reverse=True)
    return active_jobs, pending_jobs


def _render_reward_clients_panel(*, clients: list[ClientRecord], jobs: list[JobRecord]) -> str:
    direct_clients = _reward_direct_clients(clients)
    if not direct_clients:
        return """
        <section class="panel compact-panel panel-static">
          <div class="card-topline"><div><p class="section-kicker">Direct workers</p><h2>Reward Clients</h2></div></div>
          <p class="empty">No PoolTool-capable direct clients are online.</p>
        </section>
        """
    reward_jobs = _reward_job_records(jobs)
    cards: list[str] = []
    for client in direct_clients:
        active_jobs, pending_jobs = _reward_client_job_rows(client=client, jobs=reward_jobs)
        current_jobs = active_jobs or [
            job for job in reward_jobs if job.job_id in set(client.current_jobs or [])
        ]
        job_rows = "".join(
            f"""
            <article class="reward-client-job">
              <div class="reward-client-job-main">
                <strong>{html.escape(_reward_run_id(job))}</strong>
                <span class="reward-client-job-meta">{html.escape(_queue_job_title(job))} · {_job_last_activity_age(job)}</span>
              </div>
              {_pill(job.status.value, job.status.value)}
            </article>
            """
            for job in current_jobs[:3]
        )
        queued_text = (
            f"{len(pending_jobs)} queued"
            if pending_jobs
            else "idle"
        )
        client_meta = []
        direct_url = str(client.direct_url or "").strip()
        if direct_url:
            client_meta.append(direct_url.replace("http://", ""))
        client_meta.append(f"{len(client.current_jobs)}/{_effective_client_slots(client)} slots")
        cards.append(
            f"""
            <article class="reward-client-card">
              <div class="reward-client-card-top">
                <div>
                  <h3>{html.escape(_client_title(client))}</h3>
                  <div class="reward-client-card-meta">{html.escape(" · ".join(client_meta))}</div>
                </div>
                <div class="stats-inline">
                  {'<span class="mini-chip">reward target</span>' if "reward" in str(client.display_name or "").strip().lower() else ''}
                  <span class="mini-chip">{html.escape(queued_text)}</span>
                </div>
              </div>
              <div class="reward-client-card-body">
                {job_rows if job_rows else '<p class="empty inline">No reward jobs attached.</p>'}
              </div>
            </article>
            """
        )
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div><p class="section-kicker">Direct workers</p><h2>Reward Clients</h2></div>
        <div class="stats-inline"><span class="mini-chip">{len(direct_clients)} online</span></div>
      </div>
      <div class="reward-client-grid">{''.join(cards)}</div>
    </section>
    """


def _render_reward_launcher_panel(*, project_root: Path, clients: list[ClientRecord]) -> str:
    source_options = _reward_source_run_options(project_root)
    direct_clients = _reward_direct_clients(clients)
    preferred_direct = _reward_preferred_direct_client(clients)
    scene_options = _reward_scene_options(project_root)
    source_select = "".join(
        f'<option value="{html.escape(str(entry.root))}">{html.escape(entry.run_id)} · {entry.detector_count} detectors · {_display_time(entry.updated_at)}</option>'
        for entry in source_options
    )
    direct_select = "".join(
        f'<option value="{html.escape(client.client_id)}"{" selected" if preferred_direct is not None and client.client_id == preferred_direct.client_id else ""}>{html.escape(_client_title(client))}</option>'
        for client in sorted(direct_clients, key=lambda item: _client_title(item).lower())
    )
    scene_select = "".join(
        f'<option value="{html.escape(item["path"])}">{html.escape(item["label"])}</option>'
        for item in scene_options
    )
    if not source_options:
        return """
        <section class="panel compact-panel">
          <div class="card-topline">
            <div><p class="section-kicker">Launch</p><h2>Reward Experiments</h2></div>
          </div>
          <p class="empty">No supervised PoolTool runs are available yet.</p>
        </section>
        """
    base_url = _reward_default_base_url()
    model_timeout = _reward_default_timeout()
    model_name = _reward_default_model()
    no_direct_notice = (
        '<p class="empty">No direct client selected. Launches will stay queued until a live direct worker is available.</p>'
        if not direct_clients
        else ""
    )
    single_scene_form = f"""
    <section class="reward-workbench-card reward-workbench-card-primary">
      <div class="analysis-env-head">
        <div><p class="section-kicker">Direct render</p><h3>Single Scene</h3></div>
        <span class="mini-chip">{len(scene_options)} scenes</span>
      </div>
      {no_direct_notice}
      <form method="post" action="/ui/rewards/single-scene/create" class="reward-form">
        <input type="hidden" name="next" value="/rewards">
        <div class="reward-form-grid reward-form-grid-top">
          <label><span class="metric-subtle">Supervised run</span><select name="source_run_dir">{source_select}</select></label>
          <label><span class="metric-subtle">Scene</span><select name="scene_path">{scene_select}</select></label>
          <label><span class="metric-subtle">Direct client</span><select name="preferred_client_id">{direct_select}</select></label>
        </div>
        <div class="reward-textarea-grid">
          <label><span class="metric-subtle">Natural language</span><textarea name="goal_text" rows="4" placeholder="Pot the 9 ball while keeping the cue ball safe."></textarea></label>
          <label><span class="metric-subtle">DSL</span><textarea name="dsl_program" rows="4" placeholder='AND(POTTED("9"), BALL_REMAINING("cue"))'></textarea></label>
        </div>
        <div class="reward-form-grid reward-form-grid-compact">
          <label><span class="metric-subtle">Reward</span><select name="reward_kind"><option value="synthesized">Synthesized</option><option value="binary">Binary</option></select></label>
          <label><span class="metric-subtle">Budget</span><input type="number" name="budget" min="1" step="1" value="250"></label>
          <label><span class="metric-subtle">Seed</span><input type="number" name="seed" min="0" step="1" value="0"></label>
          <label><span class="metric-subtle">Model</span><input type="text" name="model" value="{html.escape(model_name)}" placeholder="optional"></label>
          <label><span class="metric-subtle">Base URL</span><input type="text" name="ollama_base_url" value="{html.escape(base_url)}"></label>
          <label><span class="metric-subtle">Timeout</span><input type="number" name="model_timeout" min="0" step="1" value="{model_timeout}"></label>
          <label><span class="metric-subtle">Run id</span><input type="text" name="run_id" placeholder="pooltool-single-scene"></label>
        </div>
        <div class="analysis-submit-row"><button class="secondary-button" type="submit">Run single scene</button></div>
      </form>
    </section>
    """
    batch_forms = f"""
    <section class="reward-workbench-card">
      <div class="analysis-env-head">
        <div><p class="section-kicker">Direct experiments</p><h3>Optimization</h3></div>
      </div>
      <form method="post" action="/ui/rewards/optimization/create" class="reward-form">
        <input type="hidden" name="next" value="/rewards">
        <div class="reward-form-grid reward-form-grid-top">
          <label><span class="metric-subtle">Supervised run</span><select name="source_run_dir">{source_select}</select></label>
          <label><span class="metric-subtle">Direct client</span><select name="preferred_client_id">{direct_select}</select></label>
        </div>
        <div class="reward-textarea-grid">
          <label><span class="metric-subtle">Natural language</span><textarea name="goal_text" rows="3" placeholder="Pot the 9 ball while keeping the cue ball safe."></textarea></label>
          <label><span class="metric-subtle">DSL</span><textarea name="dsl_program" rows="3" placeholder='AND(POTTED("9"), BALL_REMAINING("cue"))'></textarea></label>
        </div>
        <div class="reward-form-grid reward-form-grid-compact">
          <label><span class="metric-subtle">Scenes</span><input type="number" name="scene_count" min="1" step="1" value="10"></label>
          <label><span class="metric-subtle">Seeds</span><input type="number" name="seed_count" min="1" step="1" value="5"></label>
          <label><span class="metric-subtle">Budgets</span><input type="text" name="budgets_csv" value="10,50,100,250,500,1000"></label>
          <label><span class="metric-subtle">Model</span><input type="text" name="model" value="{html.escape(model_name)}" placeholder="optional"></label>
          <label><span class="metric-subtle">Base URL</span><input type="text" name="ollama_base_url" value="{html.escape(base_url)}"></label>
          <label><span class="metric-subtle">Timeout</span><input type="number" name="model_timeout" min="0" step="1" value="{model_timeout}"></label>
        </div>
        <div class="analysis-submit-row"><button class="secondary-button" type="submit">Run optimization</button></div>
      </form>
    </section>
    <section class="reward-workbench-card">
      <div class="analysis-env-head">
        <div><p class="section-kicker">Direct experiments</p><h3>Value Model</h3></div>
      </div>
      <form method="post" action="/ui/rewards/value-model/create" class="reward-form">
        <input type="hidden" name="next" value="/rewards">
        <div class="reward-form-grid reward-form-grid-top">
          <label><span class="metric-subtle">Supervised run</span><select name="source_run_dir">{source_select}</select></label>
          <label><span class="metric-subtle">Direct client</span><select name="preferred_client_id">{direct_select}</select></label>
        </div>
        <div class="reward-textarea-grid">
          <label><span class="metric-subtle">Natural language</span><textarea name="goal_text" rows="3" placeholder="Pot the 9 ball while keeping the cue ball safe."></textarea></label>
          <label><span class="metric-subtle">DSL</span><textarea name="dsl_program" rows="3" placeholder='AND(POTTED("9"), BALL_REMAINING("cue"))'></textarea></label>
        </div>
        <div class="reward-form-grid reward-form-grid-compact">
          <label><span class="metric-subtle">Train scenes</span><input type="number" name="train_scene_count" min="1" step="1" value="64"></label>
          <label><span class="metric-subtle">Actions / scene</span><input type="number" name="train_actions_per_scene" min="1" step="1" value="32"></label>
          <label><span class="metric-subtle">Validation scenes</span><input type="number" name="validation_scene_count" min="1" step="1" value="12"></label>
          <label><span class="metric-subtle">Candidates / scene</span><input type="number" name="validation_candidates_per_scene" min="1" step="1" value="128"></label>
          <label><span class="metric-subtle">Epochs</span><input type="number" name="epochs" min="1" step="1" value="50"></label>
          <label><span class="metric-subtle">Batch</span><input type="number" name="batch_size" min="1" step="1" value="64"></label>
          <label><span class="metric-subtle">LR</span><input type="number" name="learning_rate" min="0.000001" step="0.0001" value="0.001"></label>
          <label><span class="metric-subtle">Image</span><input type="number" name="image_size" min="32" step="32" value="256"></label>
          <label><span class="metric-subtle">Model</span><input type="text" name="model" value="{html.escape(model_name)}" placeholder="optional"></label>
          <label><span class="metric-subtle">Base URL</span><input type="text" name="ollama_base_url" value="{html.escape(base_url)}"></label>
          <label><span class="metric-subtle">Timeout</span><input type="number" name="model_timeout" min="0" step="1" value="{model_timeout}"></label>
        </div>
        <div class="analysis-submit-row"><button class="secondary-button" type="submit">Run value model</button></div>
      </form>
    </section>
    """
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div><p class="section-kicker">Launch</p><h2>Reward Workbench</h2></div>
        <div class="stats-inline"><span class="mini-chip">Direct clients {len(direct_clients)}</span></div>
      </div>
      <div class="reward-workbench-grid">
        {single_scene_form}
        <div class="reward-workbench-stack">{batch_forms}</div>
      </div>
    </section>
    """


def _render_single_scene_reward_results_panel(*, project_root: Path) -> str:
    root = project_root / "data" / "patterns" / "pooltool" / "reward_program_experiments" / "single_scene"
    if not root.is_dir():
        return '<section class="panel compact-panel"><h2>Single Scene Results</h2><p class="empty">No single-scene tests yet.</p></section>'
    cards: list[str] = []
    for summary_path in sorted(root.glob("*/summary.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:8]:
        payload = _load_reward_summary_payload(summary_path)
        if not payload:
            continue
        run_root = summary_path.parent
        video_relpaths = payload.get("simulation_video_paths") if isinstance(payload.get("simulation_video_paths"), list) else []
        initial_relpaths = payload.get("initial_scene_paths") if isinstance(payload.get("initial_scene_paths"), list) else []
        video_path = run_root / str(video_relpaths[0]) if video_relpaths else None
        initial_path = run_root / str(initial_relpaths[0]) if initial_relpaths else None
        preview_html = (
            f'<video controls preload="metadata" style="width:100%;max-height:240px;border:1px solid rgba(255,255,255,0.08);" src="/ui/files?path={quote(_project_relpath(video_path))}"></video>'
            if video_path is not None and video_path.is_file()
            else (
                f'<img src="/ui/files?path={quote(_project_relpath(initial_path))}" alt="{html.escape(summary_path.parent.name)}" style="width:100%;max-height:240px;object-fit:contain;border:1px solid rgba(203, 191, 174, 0.9);background:rgba(244,240,232,0.8);">'
                if initial_path is not None and initial_path.is_file()
                else '<p class="empty">No render saved.</p>'
            )
        )
        image_link = (
            f'<a class="ghost-link" href="/ui/files?path={quote(_project_relpath(initial_path))}">Initial scene</a>'
            if initial_path is not None and initial_path.is_file()
            else ""
        )
        cards.append(
            f"""
            <article class="analysis-result-card">
              <div class="analysis-result-head">
                <div>
                  <h3 class="analysis-result-card-title">{html.escape(summary_path.parent.name)}</h3>
                  <div class="analysis-result-card-meta">{html.escape(str(payload.get("scene_id") or "scene"))} · {html.escape(str(payload.get("reward_kind") or "synthesized"))} · budget {int(payload.get("budget") or 0)}</div>
                </div>
                <div class="analysis-result-links">
                  <a class="ghost-link" href="/ui/files?path={quote(_project_relpath(summary_path))}">Summary</a>
                  {image_link}
                </div>
              </div>
              <div class="stats-inline">
                <span class="mini-chip">reward {float(payload.get("best_reward") or 0.0):.3f}</span>
                <span class="mini-chip">{'success' if bool(payload.get('success')) else 'fail'}</span>
                <span class="mini-chip">{'video' if video_path is not None and video_path.is_file() else 'still only'}</span>
              </div>
              <p class="metric-subtle">{html.escape(str(payload.get("goal_text") or "").strip() or str(payload.get("dsl_program") or "").strip()[:160] or "—")}</p>
              {preview_html}
            </article>
            """
        )
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline"><div><p class="section-kicker">Rendered shots</p><h2>Single Scene Results</h2></div></div>
      <div class="analysis-result-list">{''.join(cards) if cards else '<p class="empty">No single-scene tests yet.</p>'}</div>
    </section>
    """


def _render_reward_batch_results_panel(*, project_root: Path) -> str:
    rewards = [
        item
        for item in list_environment_reward_results(project_root, "pooltool")
        if str(item.reward_kind or "").strip().lower() in {"optimization", "value-model"}
    ]
    if not rewards:
        return '<section class="panel compact-panel"><h2>Batch Results</h2><p class="empty">No reward-program experiment runs yet.</p></section>'
    cards: list[str] = []
    for entry in rewards[:10]:
        cards.append(
            f"""
            <article class="analysis-result-card">
              <div class="analysis-result-head">
                <div>
                  <h3 class="analysis-result-card-title">{html.escape(entry.run_id)}</h3>
                  <div class="analysis-result-card-meta">{html.escape(_reward_kind_label(entry.reward_kind))} · {_display_time(entry.updated_at)}</div>
                </div>
                <div class="analysis-result-links">
                  <a class="ghost-link" href="/ui/files?path={quote(_project_relpath(entry.summary_path))}">Summary</a>
                  <a class="ghost-link" href="/ui/files?path={quote(_project_relpath(entry.root))}">Run root</a>
                </div>
              </div>
              <p class="metric-subtle">{html.escape(entry.headline or "—")}</p>
              <div class="stats-inline">
                <span class="mini-chip">{html.escape(entry.source_run_id or "unknown source")}</span>
                <span class="mini-chip">{html.escape(entry.model_name or "default model")}</span>
              </div>
            </article>
            """
        )
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline"><div><p class="section-kicker">Runs</p><h2>Batch Results</h2></div></div>
      <div class="analysis-result-list">{''.join(cards)}</div>
    </section>
    """


def _render_reward_activity_panel(jobs: list[JobRecord]) -> str:
    interesting = [
        job
        for job in jobs
        if str(job.spec.action or "").strip() in {
            "pooltool-reward-optimize",
            "pooltool-reward-value-model",
            "pooltool-reward-single-scene",
        }
    ]
    if not interesting:
        return '<section class="panel compact-panel"><h2>Activity</h2><p class="empty">No reward jobs yet.</p></section>'
    rows: list[str] = []
    for job in interesting[:10]:
        rows.append(
            f"""
            <article class="analysis-activity-row">
              <div class="analysis-activity-head">
                <div class="analysis-activity-main">
                  <strong><a href="/jobs/{html.escape(job.job_id)}"><code>{html.escape(job.job_id[:8])}</code></a> · {html.escape(_queue_job_title(job))}</strong>
                  <div class="analysis-activity-meta">{html.escape(str(job.spec.metadata.get("reward_program_run_id") or ""))} · {_display_time(job.created_at)}</div>
                </div>
                {_pill(job.status.value, job.status.value)}
              </div>
              {f'<p class="analysis-activity-message">{html.escape(_sanitize_display_text(job.message))}</p>' if _sanitize_display_text(job.message) else ''}
            </article>
            """
        )
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline"><div><p class="section-kicker">Recent jobs</p><h2>Activity</h2></div></div>
      <div class="analysis-activity-list">{''.join(rows)}</div>
    </section>
    """


def _reward_dashboard_progress_note(progress: dict[str, Any]) -> str:
    event = str(progress.get("event") or "").strip()
    if not event:
        return "Idle."
    parts: list[str] = [event.replace("_", " ")]
    if progress.get("goal_index") and progress.get("goal_count"):
        parts.append(f"goal {int(progress.get('goal_index') or 0)}/{int(progress.get('goal_count') or 0)}")
    if progress.get("completed_runs") is not None and progress.get("total_runs") is not None:
        parts.append(f"{int(progress.get('completed_runs') or 0)}/{int(progress.get('total_runs') or 0)} runs")
    if progress.get("completed_steps") is not None and progress.get("total_steps") is not None:
        parts.append(f"{int(progress.get('completed_steps') or 0)}/{int(progress.get('total_steps') or 0)} steps")
    if progress.get("budget") is not None:
        parts.append(f"budget {int(progress.get('budget') or 0)}")
    updated_at = str(progress.get("updated_at") or "").strip()
    if updated_at:
        parts.append(_display_time(_parse_iso_datetime(updated_at)))
    return " · ".join(parts)


def _reward_dashboard_goal_count(state: dict[str, Any]) -> int:
    goals = state.get("goal_texts")
    if not isinstance(goals, list):
        return 0
    return sum(1 for item in goals if str(item or "").strip())


def _render_rewards_goal_panel(*, state: dict[str, Any]) -> str:
    goals = state.get("goal_texts") if isinstance(state.get("goal_texts"), list) else []
    normalized = [str(item or "") for item in goals[:10]] + ["" for _ in range(max(0, 10 - len(goals[:10])))]
    rows = "".join(
        f"""
        <label>
          <span class="metric-subtle">Goal {index}</span>
          <textarea name="goal_{index:02d}" rows="2" placeholder="Describe an exact PoolTool outcome to optimize and later verify.">{html.escape(value)}</textarea>
        </label>
        """
        for index, value in enumerate(normalized, start=1)
    )
    filled = sum(1 for item in normalized if item.strip())
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div><p class="section-kicker">Shared Goal Set</p><h2>Natural Language Goals</h2></div>
        <div class="stats-inline"><span class="mini-chip">{filled}/10 filled</span></div>
      </div>
      <p class="metric-subtle">This is the single shared 10-goal list used by both the optimization suite and value-model suite. Exact handwritten trace verifiers are still pending, so this page is preparing the workflow and run surfaces first.</p>
      <form method="post" action="/ui/rewards/goals/save" class="reward-form">
        <input type="hidden" name="next" value="/rewards">
        <div class="reward-textarea-grid">{rows}</div>
        <div class="analysis-submit-row"><button class="secondary-button" type="submit">Save 10-goal set</button></div>
      </form>
    </section>
    """


def _render_rewards_single_scene_panel(*, project_root: Path, state: dict[str, Any], scene: dict[str, Any], snapshot: dict[str, Any]) -> str:
    panel_state = snapshot.get("state") if isinstance(snapshot.get("state"), dict) else {}
    scene_payload = scene.get("scene") if isinstance(scene.get("scene"), dict) else {}
    image_path = scene.get("image_path")
    image_html = ""
    if isinstance(image_path, Path) and image_path.is_file():
        image_html = (
            f'<img src="/ui/files?path={quote(_project_relpath(image_path))}" '
            'alt="Current PoolTool scene" style="width:100%;aspect-ratio:2 / 1;max-height:none;object-fit:contain;'
            'border:1px solid rgba(255,255,255,0.1);background:rgba(15,23,42,0.48);">'
        )
    scene_running = random_scene_running()
    scene_meta = "No scene sampled yet."
    if scene_payload:
        scene_meta = " · ".join(
            part for part in [
                str(scene_payload.get("scene_id") or "").strip(),
                f"seed {int(panel_state.get('scene_seed') or 0)}",
                f"{len(scene_payload.get('initial_ball_positions') or [])} balls",
            ] if part
        )
    if scene_running:
        scene_meta = (scene_meta + " · " if scene_meta else "") + "Generating new scene…"
    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), dict) else {}
    run_root = snapshot.get("run_root") if isinstance(snapshot.get("run_root"), Path) else None
    video_html = ""
    final_paths = summary.get("simulation_video_paths")
    if isinstance(final_paths, list) and run_root is not None and final_paths:
        preferred_video_rel = next(
            (str(item) for item in final_paths if "overhead" in str(item).lower()),
            str(final_paths[0]),
        )
        final_video = run_root / preferred_video_rel
        if final_video.is_file():
            video_html = (
                f'<video controls preload="metadata" style="width:100%;aspect-ratio:2 / 1;max-height:none;object-fit:contain;border:1px solid rgba(255,255,255,0.1);background:rgba(15,23,42,0.48);" '
                f'src="/ui/files?path={quote(_project_relpath(final_video))}"></video>'
            )
    dsl_text = str((snapshot.get("progress") or {}).get("dsl_program") or snapshot.get("program_text") or summary.get("dsl_program") or "").strip()
    action_payload = summary.get("action") if isinstance(summary.get("action"), dict) else {}
    action_text = ", ".join(f"{key}={value}" for key, value in action_payload.items()) if action_payload else "—"
    status_chip = "running" if snapshot.get("running") else ("completed" if summary else "idle")
    goal_text = str(panel_state.get("goal_text") or "")
    log_lines = [str(line) for line in (snapshot.get("log_lines") or []) if str(line).strip()]
    log_text = "\n".join(reversed(log_lines[-120:])) if log_lines else "No local tmux output yet."
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div><p class="section-kicker">Single Scene</p><h2>Scene Playground</h2></div>
        <div class="stats-inline"><span class="mini-chip">{html.escape(status_chip)}</span></div>
      </div>
      <p class="metric-subtle">Uses the current PoolTool supervised ensemble library. Sample a random scene, enter one natural-language goal, generate DSL through the LLM, then watch optimization progress until the best action and final simulation are available.</p>
      <div class="reward-form-grid reward-form-grid-top">
        <form method="post" action="/ui/rewards/single-scene/randomize" class="reward-form">
          <input type="hidden" name="next" value="/rewards">
          <div class="analysis-submit-row"><button class="secondary-button" type="submit">Create random scene</button></div>
        </form>
        <div class="metric-subtle">{html.escape(scene_meta)}</div>
      </div>
      <div class="reward-textarea-grid">
        <div>{image_html or '<p class="empty">No scene preview yet.</p>'}</div>
        <div>{video_html or '<p class="empty">No optimized simulation yet.</p>'}</div>
      </div>
      <form method="post" action="/ui/rewards/single-scene/start" class="reward-form" style="margin-top:0.9rem;">
        <input type="hidden" name="next" value="/rewards">
        <div class="reward-textarea-grid">
          <label><span class="metric-subtle">Goal</span><textarea name="goal_text" rows="4" placeholder="Describe one exact PoolTool outcome for this sampled scene.">{html.escape(goal_text)}</textarea></label>
          <label><span class="metric-subtle">Generated DSL</span><textarea rows="4" readonly placeholder="The generated DSL will appear here once the run starts.">{html.escape(dsl_text)}</textarea></label>
        </div>
        <div class="reward-form-grid reward-form-grid-compact">
          <label><span class="metric-subtle">Budget</span><input type="number" name="budget" min="1" step="1" value="{int(panel_state.get('budget') or 250)}"></label>
          <label><span class="metric-subtle">Seed</span><input type="number" name="seed" min="0" step="1" value="{int(panel_state.get('seed') or 0)}"></label>
          <label><span class="metric-subtle">Model</span><input type="text" name="model" value="{html.escape(str(panel_state.get('model') or _reward_default_model()))}" placeholder="optional"></label>
        </div>
        <div class="analysis-submit-row">
          <button class="secondary-button" type="submit">Run single-scene optimization</button>
        </div>
      </form>
      <form method="post" action="/ui/rewards/single-scene/cancel" class="reward-form" style="margin-top:0.5rem;">
        <input type="hidden" name="next" value="/rewards">
        <div class="analysis-submit-row"><button class="ghost-link" type="submit">Cancel single-scene tmux job</button></div>
      </form>
      <div class="analysis-progress" style="margin-top:0.85rem;">
        <div class="analysis-progress-meta"><span>Live progress</span><span>{html.escape('Generating new scene…' if scene_running else _reward_dashboard_progress_note(snapshot.get('progress') or {}))}</span></div>
        <div class="analysis-progress-track"><span class="analysis-progress-fill{' is-indeterminate' if snapshot.get('running') or scene_running else ''}" style="width: {int(round(float((snapshot.get('progress') or {}).get('fraction') or 0.0) * 100.0)) if isinstance(snapshot.get('progress'), dict) else 0}%"></span></div>
        <p class="analysis-progress-note">Best reward {html.escape(str(summary.get("best_reward") if summary else "—"))} · Success {html.escape(str(summary.get("success") if summary else "—"))} · Action {html.escape(action_text)}</p>
      </div>
      <pre class="code-block" style="margin-top:0.85rem;white-space:pre-wrap;">{html.escape(log_text)}</pre>
    </section>
    """


def _render_rewards_optimization_suite_panel(*, project_root: Path, state: dict[str, Any], snapshot: dict[str, Any]) -> str:
    panel_state = snapshot.get("state") if isinstance(snapshot.get("state"), dict) else {}
    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), dict) else {}
    goals = summary.get("goals") if isinstance(summary.get("goals"), list) else []
    rows = ""
    for item in goals[:10]:
        if not isinstance(item, dict):
            continue
        summary_path = Path(str(item.get("summary_path") or ""))
        summary_link = ""
        if summary_path.is_file():
            summary_link = f'<a class="ghost-link" href="/ui/files?path={quote(_project_relpath(summary_path))}">Goal summary</a>'
        rows += f"""
        <article class="analysis-activity-row">
          <div class="analysis-activity-head">
            <div class="analysis-activity-main">
              <strong>{html.escape(str(item.get("goal_text") or ""))}</strong>
              <div class="analysis-activity-meta">best binary {float(item.get("best_binary_success_rate") or 0.0):.2f} @ {int(item.get("best_binary_budget") or 0)} · best dense {float(item.get("best_dense_mean_reward") or 0.0):.2f} @ {int(item.get("best_dense_budget") or 0)}</div>
            </div>
            <span class="mini-chip">{html.escape(str(item.get("run_id") or ""))}</span>
          </div>
          {summary_link}
        </article>
        """
    goal_count = _reward_dashboard_goal_count(state)
    budgets_value = ",".join(str(int(item)) for item in (panel_state.get("budgets") or [10, 50, 100, 250, 500, 1000]))
    disabled_note = "" if goal_count >= 10 else '<p class="empty">Fill all 10 shared goals before launching the optimization suite.</p>'
    log_lines = [str(line) for line in (snapshot.get("log_lines") or []) if str(line).strip()]
    log_text = "\n".join(reversed(log_lines[-120:])) if log_lines else "No local tmux output yet."
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div><p class="section-kicker">Experiment 1</p><h2>Reward Program Optimization Suite</h2></div>
        <div class="stats-inline"><span class="mini-chip">{html.escape('running' if snapshot.get('running') else 'idle')}</span></div>
      </div>
      <p class="metric-subtle">Runs one local tmux suite across the shared 10-goal list using the supervised ensemble library. Each goal gets optimization traces across multiple budgets so the page can compare which budgets improve binary success most effectively. Exact handwritten trace verifiers are not installed yet, so this surface is readying the experiment structure and live monitoring.</p>
      {disabled_note}
      <form method="post" action="/ui/rewards/optimization-suite/start" class="reward-form">
        <input type="hidden" name="next" value="/rewards">
        <div class="reward-form-grid reward-form-grid-compact">
          <label><span class="metric-subtle">Scenes</span><input type="number" name="scene_count" min="1" step="1" value="{int(panel_state.get('scene_count') or 10)}"></label>
          <label><span class="metric-subtle">Seeds</span><input type="number" name="seed_count" min="1" step="1" value="{int(panel_state.get('seed_count') or 5)}"></label>
          <label><span class="metric-subtle">Budgets</span><input type="text" name="budgets_csv" value="{html.escape(budgets_value)}"></label>
          <label><span class="metric-subtle">Model</span><input type="text" name="model" value="{html.escape(str(panel_state.get('model') or _reward_default_model()))}" placeholder="optional"></label>
        </div>
        <div class="analysis-submit-row"><button class="secondary-button" type="submit">Launch optimization suite tmux job</button></div>
      </form>
      <form method="post" action="/ui/rewards/optimization-suite/cancel" class="reward-form" style="margin-top:0.5rem;">
        <input type="hidden" name="next" value="/rewards">
        <div class="analysis-submit-row"><button class="ghost-link" type="submit">Cancel optimization suite tmux job</button></div>
      </form>
      <div class="analysis-progress" style="margin-top:0.85rem;">
        <div class="analysis-progress-meta"><span>Live progress</span><span>{html.escape(_reward_dashboard_progress_note(snapshot.get("progress") or {}))}</span></div>
        <div class="analysis-progress-track"><span class="analysis-progress-fill{' is-indeterminate' if snapshot.get('running') else ''}" style="width: 100%"></span></div>
        <p class="analysis-progress-note">Run id {html.escape(str(snapshot.get("run_id") or "—"))} · Goals finished {len(goals) if goals else 0}</p>
      </div>
      <pre class="code-block" style="margin-top:0.85rem;white-space:pre-wrap;">{html.escape(log_text)}</pre>
      <div class="analysis-activity-list" style="margin-top:0.85rem;">{rows or '<p class="empty">No optimization suite summary yet.</p>'}</div>
    </section>
    """


def _render_rewards_value_model_panel(*, project_root: Path, state: dict[str, Any], snapshot: dict[str, Any]) -> str:
    panel_state = snapshot.get("state") if isinstance(snapshot.get("state"), dict) else {}
    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), dict) else {}
    leaderboard = summary.get("leaderboard") if isinstance(summary.get("leaderboard"), list) else []
    config_aggregates = summary.get("config_aggregates") if isinstance(summary.get("config_aggregates"), list) else []
    rows = ""
    for item in leaderboard[:10]:
        if not isinstance(item, dict):
            continue
        rows += f"""
        <article class="analysis-activity-row">
          <div class="analysis-activity-head">
            <div class="analysis-activity-main">
              <strong>{html.escape(str(item.get("goal_text") or ""))}</strong>
              <div class="analysis-activity-meta">{html.escape(str(item.get("config_id") or ""))} · reward {float(item.get("best_validation_mean_reward") or 0.0):.3f} · success {float(item.get("best_validation_success_rate") or 0.0):.3f}</div>
            </div>
            <span class="mini-chip">epoch {int(item.get("best_epoch") or 0)}</span>
          </div>
        </article>
        """
    aggregate_html = "".join(
        f'<span class="mini-chip">{html.escape(str(item.get("config_id") or ""))}: reward {float(item.get("mean_best_validation_reward") or 0.0):.3f}, success {float(item.get("mean_best_validation_success_rate") or 0.0):.3f}</span>'
        for item in config_aggregates[:8]
        if isinstance(item, dict)
    )
    goal_count = _reward_dashboard_goal_count(state)
    disabled_note = "" if goal_count >= 10 else '<p class="empty">Fill all 10 shared goals before launching the value-model suite.</p>'
    log_lines = [str(line) for line in (snapshot.get("log_lines") or []) if str(line).strip()]
    log_text = "\n".join(reversed(log_lines[-120:])) if log_lines else "No local tmux output yet."
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div><p class="section-kicker">Experiment 2</p><h2>Reward Program Value-Model Sweep</h2></div>
        <div class="stats-inline"><span class="mini-chip">{html.escape('running' if snapshot.get('running') else 'idle')}</span></div>
      </div>
      <p class="metric-subtle">Runs a local tmux sweep of small supervised value models over the same 10-goal list, training on the PoolTool training split with rewards from the reward programs. The page tracks the current goal/config pair and then ranks the finished configs by validation reward and success rate.</p>
      {disabled_note}
      <form method="post" action="/ui/rewards/value-model-sweep/start" class="reward-form">
        <input type="hidden" name="next" value="/rewards">
        <div class="reward-form-grid reward-form-grid-compact">
          <label><span class="metric-subtle">Model</span><input type="text" name="model" value="{html.escape(str(panel_state.get('model') or _reward_default_model()))}" placeholder="optional"></label>
        </div>
        <div class="analysis-submit-row"><button class="secondary-button" type="submit">Launch value-model sweep tmux job</button></div>
      </form>
      <form method="post" action="/ui/rewards/value-model-sweep/cancel" class="reward-form" style="margin-top:0.5rem;">
        <input type="hidden" name="next" value="/rewards">
        <div class="analysis-submit-row"><button class="ghost-link" type="submit">Cancel value-model sweep tmux job</button></div>
      </form>
      <div class="analysis-progress" style="margin-top:0.85rem;">
        <div class="analysis-progress-meta"><span>Live progress</span><span>{html.escape(_reward_dashboard_progress_note(snapshot.get("progress") or {}))}</span></div>
        <div class="analysis-progress-track"><span class="analysis-progress-fill{' is-indeterminate' if snapshot.get('running') else ''}" style="width: 100%"></span></div>
        <p class="analysis-progress-note">Run id {html.escape(str(snapshot.get("run_id") or "—"))} · Completed configs {int((snapshot.get("progress") or {}).get("completed_runs") or 0)}</p>
      </div>
      <pre class="code-block" style="margin-top:0.85rem;white-space:pre-wrap;">{html.escape(log_text)}</pre>
      <div class="stats-inline" style="margin-top:0.85rem;">{aggregate_html or '<span class="mini-chip">No config aggregates yet</span>'}</div>
      <div class="analysis-activity-list" style="margin-top:0.85rem;">{rows or '<p class="empty">No value-model sweep summary yet.</p>'}</div>
    </section>
    """


def render_rewards_home_page(*, jobs: list[JobRecord], clients: list[ClientRecord], project_root: Path) -> str:
    del jobs, clients
    state = load_rewards_dashboard_state(project_root)
    filled_goals = _reward_dashboard_goal_count(state)
    current_scene = load_current_scene(project_root)
    snapshots = {
        panel_key: load_reward_panel_snapshot(project_root, state, panel_key)
        for panel_key in REWARD_PANEL_KEYS
    }
    cached_summary = load_cached_canonical_library_summary(project_root, "pooltool", "supervised")
    library_path_text = str(cached_summary.get("library_path") or "").strip()
    if library_path_text:
        library_path = Path(library_path_text).resolve()
    else:
        library_path = (
            project_root
            / "state"
            / "video_renders"
            / "canonical_libraries"
            / "pooltool"
            / "supervised"
            / "pattern_library.json"
        ).resolve()
    library_meta = {}
    calibration_payload = {}
    try:
        library_meta = json.loads((library_path.parent / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        library_meta = {}
    try:
        calibration_payload = json.loads((library_path.parent / "ensemble_calibration.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        calibration_payload = {}
    calibration_text = ", ".join(
        f"{key}={value}"
        for key, value in calibration_payload.items()
        if key in {
            "confidence_threshold",
            "start_time_threshold",
            "end_time_threshold",
            "start_location_threshold",
            "end_location_threshold",
        }
    )
    any_running = any(bool(snapshot.get("running")) for snapshot in snapshots.values()) or random_scene_running()
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Home</a> / rewards</p>
      <h1>PoolTool Rewards</h1>
      <p>Local tmux-only playground and experiment dashboard for the supervised ensemble library.</p>
    </section>
    <section class="hero-stats">
      <div class="stat-chip"><span class="stat-label">Shared goals</span><strong class="stat-value">{filled_goals}/10</strong></div>
      <div class="stat-chip"><span class="stat-label">Library kind</span><strong class="stat-value">PoolTool supervised</strong></div>
      <div class="stat-chip"><span class="stat-label">Library fingerprint</span><strong class="stat-value">{html.escape(str(library_meta.get("fingerprint") or "—")[:12])}</strong></div>
      <div class="stat-chip"><span class="stat-label">Local tmux runs</span><strong class="stat-value">{sum(1 for snapshot in snapshots.values() if snapshot.get("running"))}</strong></div>
    </section>
    <section class="panel compact-panel panel-static">
      <div class="card-topline"><div><p class="section-kicker">Source Library</p><h2>Current PoolTool Supervised Ensemble</h2></div></div>
      <p class="metric-subtle">All three surfaces on this page are pinned to the current canonical PoolTool supervised ensemble library and its calibration sidecar.</p>
      <div class="stats-inline">
        <span class="mini-chip">{html.escape(_project_relpath(library_path))}</span>
        <span class="mini-chip">{html.escape(calibration_text or "No ensemble calibration sidecar found")}</span>
      </div>
    </section>
    {_render_rewards_goal_panel(state=state)}
    {_render_rewards_single_scene_panel(project_root=project_root, state=state, scene=current_scene, snapshot=snapshots["single_scene"])}
    {_render_rewards_optimization_suite_panel(project_root=project_root, state=state, snapshot=snapshots["optimization_suite"])}
    {_render_rewards_value_model_panel(project_root=project_root, state=state, snapshot=snapshots["value_model_sweep"])}
    """
    if any_running:
        body += "<script>setTimeout(() => window.location.reload(), 5000);</script>"
    return _shell("Reward Programs | Pattern Server", active_nav="rewards", body_html=body)


def _collect_video_showcase_tasks(
    *,
    project_root: Path,
) -> dict[str, dict[str, list[tuple[datetime, str, str]]]]:
    def _json_script_text(value: str) -> str:
        return json.dumps(value).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")

    def _label_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()

    def _format_activated_patterns(payload: dict[str, Any], *, env_id: str, selected_label_keys: set[str] | None = None) -> str:
        def _compact_number(value: Any) -> str:
            if isinstance(value, bool):
                return str(value).lower()
            if isinstance(value, int):
                return str(value)
            if isinstance(value, float):
                if math.isfinite(value):
                    return f"{value:.3f}".rstrip("0").rstrip(".")
                return str(value)
            return str(value)

        def _compact_value(value: Any, *, depth: int = 0) -> str:
            if value is None:
                return "null"
            if isinstance(value, (bool, int, float)):
                return _compact_number(value)
            if isinstance(value, str):
                text = value.strip()
                if len(text) > 60:
                    text = text[:57] + "..."
                return text
            if isinstance(value, list):
                if not value:
                    return "[]"
                if depth >= 1:
                    return f"[{len(value)} items]"
                shown = ", ".join(_compact_value(item, depth=depth + 1) for item in value[:4])
                if len(value) > 4:
                    shown += f", ... +{len(value) - 4}"
                return f"[{shown}]"
            if isinstance(value, dict):
                if not value:
                    return "{}"
                if depth >= 1:
                    return "{...}"
                parts: list[str] = []
                for index, (key, item) in enumerate(value.items()):
                    if index >= 4:
                        parts.append(f"... +{len(value) - 4}")
                        break
                    parts.append(f"{key}={_compact_value(item, depth=depth + 1)}")
                return "{" + ", ".join(parts) + "}"
            return str(value)

        def _format_object_details(value: Any) -> str:
            if not isinstance(value, dict) or not value:
                return ""
            parts: list[str] = []
            for index, (obj_key, raw_detail) in enumerate(value.items()):
                if index >= 4:
                    parts.append(f"... +{len(value) - 4}")
                    break
                if not isinstance(raw_detail, dict):
                    parts.append(f"{obj_key}:{_compact_value(raw_detail)}")
                    continue
                role = str(raw_detail.get("role") or raw_detail.get("kind") or raw_detail.get("geometry") or "").strip()
                start_pos = raw_detail.get("start_position")
                end_pos = raw_detail.get("end_position")
                extras: list[str] = []
                if role:
                    extras.append(role)
                if start_pos is not None:
                    extras.append(f"start={_compact_value(start_pos)}")
                if end_pos is not None:
                    extras.append(f"end={_compact_value(end_pos)}")
                if not extras:
                    extras.append(_compact_value(raw_detail))
                parts.append(f"{obj_key}(" + ", ".join(extras) + ")")
            return "objects=" + "; ".join(parts)

        def _format_parameters_inline(parameters: Any) -> str:
            if not isinstance(parameters, dict) or not parameters:
                return ""
            segments: list[str] = []
            object_details = _format_object_details(parameters.get("object_details"))
            if object_details:
                segments.append(object_details)
            object_indices = parameters.get("object_indices")
            if isinstance(object_indices, list) and object_indices:
                segments.append(f"object_indices={_compact_value(object_indices)}")
            object_positions = parameters.get("object_positions")
            if isinstance(object_positions, list) and object_positions:
                segments.append(f"object_positions={_compact_value(object_positions)}")
            for key, value in parameters.items():
                if key in {"object_details", "object_indices", "object_positions"}:
                    continue
                segments.append(f"{key}={_compact_value(value)}")
            return " | ".join(segment for segment in segments if segment)

        annotations = payload.get("pattern_annotations")
        if not isinstance(annotations, dict):
            return "No pattern annotations available."
        patterns = annotations.get("patterns")
        if not isinstance(patterns, list) or not patterns:
            return "No patterns detected."
        filtered_patterns = filter_annotation_patterns_for_env(env_id, patterns)
        if selected_label_keys:
            filtered_patterns = [
                item for item in filtered_patterns
                if _label_key(str(item.get("label") or "")) in selected_label_keys
            ]
        if not filtered_patterns:
            return "No patterns detected."
        lines: list[str] = [f"Activated patterns: {len(filtered_patterns)}"]
        for index, item in enumerate(filtered_patterns, start=1):
            label = str(item.get("label") or "pattern").strip()
            start = item.get("start_timestep")
            end = item.get("end_timestep")
            parameters = item.get("parameters")
            summary = _format_parameters_inline(parameters)
            line = f"{index}. {label} [{start}-{end}]".strip()
            if summary:
                line = f"{line} | {summary}"
            lines.append(line)
        return "\n".join(lines).strip() or "No patterns detected."

    tasks_by_env: dict[str, dict[str, list[tuple[datetime, str, str]]]] = {
        env_id: {"success": [], "failure": []}
        for env_id in SUPPORTED_ENVS
    }
    selected_label_keys_by_env = {
        env_id: {
            _label_key(label)
            for label in (load_annotation_video_selection(project_root, env_id).get("selected_labels") or [])
            if _label_key(label)
        }
        for env_id in SUPPORTED_ENVS
    }
    for env_id, entries in list_video_render_results(project_root).items():
        for entry in select_video_showcase_entries(entries):
            video_version_parts = [str(entry.updated_at.timestamp())]
            if isinstance(entry.video_path, Path) and entry.video_path.is_file():
                video_version_parts.append(str(entry.video_path.stat().st_mtime_ns))
            if isinstance(entry.overhead_video_path, Path) and entry.overhead_video_path.is_file():
                video_version_parts.append(str(entry.overhead_video_path.stat().st_mtime_ns))
            if isinstance(entry.annotation_path, Path) and entry.annotation_path.is_file():
                video_version_parts.append(str(entry.annotation_path.stat().st_mtime_ns))
            version_token = hashlib.sha1(":".join(video_version_parts).encode("utf-8")).hexdigest()[:12]
            video_href = f"/ui/files?path={quote(_project_relpath(entry.video_path))}&v={version_token}"
            dialog_node = ""
            task_result_relpath = quote(_project_relpath(entry.task_result_path))
            primary_href = video_href
            primary_title = entry.task_id
            primary_overlay_url = ""
            if (
                env_id == "pooltool"
                and isinstance(entry.overhead_video_path, Path)
                and entry.overhead_video_path.is_file()
            ):
                primary_href = f"/ui/files?path={quote(_project_relpath(entry.overhead_video_path))}&v={version_token}"
                primary_title = f"{entry.task_id} top-down"
                if isinstance(entry.annotation_path, Path) and entry.annotation_path.is_file():
                    primary_overlay_url = f"/api/ui/videos/overlay?task_result_path={task_result_relpath}&view=overhead&v={version_token}"
            elif isinstance(entry.video_path, Path) and entry.video_path.is_file():
                primary_href = video_href
                primary_title = entry.task_id
                if (
                    isinstance(entry.annotation_path, Path)
                    and entry.annotation_path.is_file()
                    and env_id != "pooltool"
                ):
                    primary_overlay_url = f"/api/ui/videos/overlay?task_result_path={task_result_relpath}&view=askew&v={version_token}"
            elif (
                isinstance(entry.overhead_video_path, Path)
                and entry.overhead_video_path.is_file()
            ):
                primary_href = f"/ui/files?path={quote(_project_relpath(entry.overhead_video_path))}&v={version_token}"
                primary_title = f"{entry.task_id} top-down"
                if isinstance(entry.annotation_path, Path) and entry.annotation_path.is_file():
                    primary_overlay_url = f"/api/ui/videos/overlay?task_result_path={task_result_relpath}&view=overhead&v={version_token}"
            annotation_button = ""
            annotation_text = ""
            if isinstance(entry.annotation_path, Path) and entry.annotation_path.is_file():
                try:
                    payload = json.loads(entry.annotation_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    payload = None
                if isinstance(payload, dict):
                    annotation_text = _format_activated_patterns(
                        payload,
                        env_id=env_id,
                        selected_label_keys=selected_label_keys_by_env.get(env_id, set()),
                    )
                elif not annotation_text:
                    try:
                        annotation_text = entry.annotation_path.read_text(encoding="utf-8")
                    except OSError:
                        annotation_text = ""
            if not annotation_text and isinstance(entry.annotation_path, Path) and entry.annotation_path.is_file() and entry.annotation_preview:
                annotation_text = entry.annotation_preview
            if annotation_text:
                annotation_dialog_id = (
                    "video-annotation-"
                    + hashlib.sha1(f"{env_id}:{entry.source_run_id}:{entry.task_id}".encode("utf-8")).hexdigest()[:16]
                )
                annotation_button = (
                    f'<button class="ghost-link js-text-open" type="button" '
                    f'data-text-title="{html.escape(entry.task_id)} annotation" '
                    f'data-text-source="{annotation_dialog_id}">Annotation</button>'
                )
                dialog_node += (
                    f'<script type="application/json" id="{annotation_dialog_id}">'
                    f'{_json_script_text(annotation_text)}'
                    f'</script>'
                )
            elif isinstance(entry.annotation_path, Path) and entry.annotation_path.is_file():
                annotation_href = f"/ui/files?path={quote(_project_relpath(entry.annotation_path))}"
                annotation_button = f'<a class="ghost-link" href="{html.escape(annotation_href)}">Annotation</a>'
            task_html = (
                '<article class="video-task-row">'
                '<div class="video-task-row-main">'
                f"<strong>{html.escape(entry.task_id)}</strong>"
                f'<span class="metric-subtle">{html.escape(entry.output_run_id)} · {html.escape(entry.outcome_kind.title())} · {_display_time(entry.updated_at)}</span>'
                "</div>"
                '<div class="video-task-actions">'
                f'<button class="secondary-button js-video-open" type="button" data-video-src="{html.escape(primary_href)}" data-video-title="{html.escape(primary_title)}"'
                + (f' data-overlay-url="{html.escape(primary_overlay_url)}"' if primary_overlay_url else "")
                + '>Video</button>'
                f"{annotation_button}"
                "</div>"
                "</article>"
            )
            outcome_bucket = "failure" if entry.outcome_kind == "failure" else "success"
            tasks_by_env[env_id][outcome_bucket].append((entry.updated_at, task_html, dialog_node))
    return tasks_by_env


def _latest_image_only_eval_source_by_env(jobs: list[JobRecord]) -> dict[str, JobRecord]:
    latest: dict[str, JobRecord] = {}
    for job in sorted(jobs, key=lambda item: item.updated_at, reverse=True):
        if str(job.spec.action or "").strip() != "evaluate-deepphy":
            continue
        env_id = _job_env_id(job)
        if env_id not in SUPPORTED_ENVS or env_id in latest:
            continue
        metadata = dict(job.spec.metadata or {})
        request_payload = metadata.get("request")
        request = request_payload if isinstance(request_payload, dict) else {}
        scope = str(request.get("evaluation_scope") or metadata.get("evaluation_scope") or "").strip().lower()
        run_marker = str(request.get("run_id") or metadata.get("evaluation_run_id") or "").strip().lower()
        if scope != "image-only" and "image-only" not in run_marker:
            continue
        latest[env_id] = job
    return latest


def _canonical_library_kind_title(kind: str) -> str:
    kind_key = str(kind or "").strip().lower()
    if kind_key == "supervised":
        return "Supervised"
    if kind_key == "unsupervised":
        return f"Unsupervised · top {UNSUPERVISED_LIBRARY_LIMIT}"
    if kind_key == "automatic":
        return f"Automatic · top {AUTOMATIC_LIBRARY_LIMIT}"
    return kind_key.title() or "Library"


def _json_script_text(value: str) -> str:
    return json.dumps(value).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _render_canonical_library_collection(
    *,
    project_root: Path,
    kind: str,
    next_href: str,
    include_selection_controls: bool,
) -> tuple[str, list[str], list[str]]:
    panels: list[str] = []
    text_payload_nodes: list[str] = []
    detail_payload_nodes: list[str] = []

    def _cluster_color(index: int | None) -> str:
        palette = (
            "#d1495b",
            "#edae49",
            "#66a182",
            "#2e86ab",
            "#7b6d8d",
            "#c17c74",
            "#6c9a8b",
            "#5f6caf",
            "#b56576",
            "#355070",
            "#7f5539",
            "#6a994e",
        )
        if not isinstance(index, int):
            return "#6b7280"
        return palette[index % len(palette)]

    def _lexical_code_vector(source_code: str) -> list[float]:
        text = str(source_code or "")
        lines = [line for line in text.splitlines() if line.strip()]
        return [
            float(len(text)),
            float(len(lines)),
            float(text.count("for ")),
            float(text.count("while ")),
            float(text.count("if ")),
            float(text.count("return ")),
            float(text.count("def ")),
            float(text.count(" and ") + text.count(" or ")),
        ]

    def _project_member_points(members: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not members:
            return []
        embedding_model = ""
        cached_vectors: list[list[float]] = []
        try:
            _ensure_client_src_on_path()
            from ensemble_library_training.code_embeddings import (  # type: ignore
                code_hash_for_text,
                load_cached_embedding,
                normalize_embedding,
                resolve_openai_embedding_config,
            )

            _api_key, _base_url, embedding_model = resolve_openai_embedding_config(project_root)
            for member in members:
                code_text = str(member.get("source_code") or "")
                cached = load_cached_embedding(
                    project_root=project_root,
                    model=embedding_model,
                    code_hash=code_hash_for_text(code_text),
                )
                if cached:
                    cached_vectors.append(normalize_embedding(cached))
                else:
                    cached_vectors.append(_lexical_code_vector(code_text))
        except Exception:
            cached_vectors = [_lexical_code_vector(str(member.get("source_code") or "")) for member in members]
        if not cached_vectors:
            cached_vectors = [_lexical_code_vector(str(member.get("source_code") or "")) for member in members]
        dimension_count = max(len(vector) for vector in cached_vectors)
        vectors = [vector + [0.0] * (dimension_count - len(vector)) for vector in cached_vectors]
        if len(vectors) == 1:
            return [
                {
                    **members[0],
                    "plot_x": 0.5,
                    "plot_y": 0.5,
                }
            ]
        variances: list[tuple[float, int]] = []
        for dim in range(dimension_count):
            values = [vector[dim] for vector in vectors]
            mean = sum(values) / float(len(values))
            variance = sum((value - mean) ** 2 for value in values) / float(len(values))
            variances.append((variance, dim))
        variances.sort(reverse=True)
        first_dim = variances[0][1] if variances else 0
        second_dim = variances[1][1] if len(variances) > 1 else first_dim
        xs = [vector[first_dim] for vector in vectors]
        ys = [vector[second_dim] for vector in vectors]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        projected: list[dict[str, Any]] = []
        for index, member in enumerate(members):
            x_value = xs[index]
            y_value = ys[index]
            x_norm = 0.5 if abs(max_x - min_x) < 1e-9 else (x_value - min_x) / (max_x - min_x)
            y_norm = 0.5 if abs(max_y - min_y) < 1e-9 else (y_value - min_y) / (max_y - min_y)
            projected.append(
                {
                    **member,
                    "plot_x": float(x_norm),
                    "plot_y": float(y_norm),
                }
            )
        return projected

    def _library_label_groups(library_path: Path) -> list[dict[str, Any]]:
        payload = _load_json_object(library_path)
        ensembles = payload.get("label_ensembles")
        groups: list[dict[str, Any]] = []
        if isinstance(ensembles, list) and ensembles:
            for ensemble in ensembles:
                if not isinstance(ensemble, dict):
                    continue
                label = str(ensemble.get("label") or "").strip()
                input_mode = str(ensemble.get("input_mode") or "").strip() or "trace"
                raw_members = ensemble.get("members")
                members: list[dict[str, Any]] = []
                if isinstance(raw_members, list):
                    for member in raw_members:
                        if not isinstance(member, dict):
                            continue
                        detector = member.get("detector") if isinstance(member.get("detector"), dict) else {}
                        metadata = detector.get("metadata") if isinstance(detector.get("metadata"), dict) else {}
                        score_raw = member.get("train_score")
                        if not isinstance(score_raw, (int, float)):
                            score_raw = metadata.get("train_score")
                        members.append(
                            {
                                "slot_index": int(member.get("slot_index") or 0),
                                "cluster_index": int(member.get("cluster_index") or 0),
                                "detector_id": str(detector.get("detector_id") or "").strip(),
                                "source_code": str(detector.get("source_code") or ""),
                                "score": float(score_raw) if isinstance(score_raw, (int, float)) else None,
                                "source_run_id": str(metadata.get("source_run_id") or "").strip(),
                                "source_run_dir": str(metadata.get("source_run_dir") or "").strip(),
                                "source_label_dir": str(metadata.get("source_label_dir") or "").strip(),
                            }
                        )
                members.sort(
                    key=lambda item: (
                        int(item.get("cluster_index") or 0),
                        int(item.get("slot_index") or 0),
                        -float(item.get("score") or 0.0),
                    )
                )
                groups.append(
                    {
                        "label": label,
                        "input_mode": input_mode,
                        "members": _project_member_points(members),
                    }
                )
        else:
            by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for detector in payload.get("detectors") or []:
                if not isinstance(detector, dict):
                    continue
                metadata = detector.get("metadata") if isinstance(detector.get("metadata"), dict) else {}
                label = str(detector.get("label") or "").strip()
                by_label[label].append(
                    {
                        "slot_index": len(by_label[label]),
                        "cluster_index": len(by_label[label]),
                        "detector_id": str(detector.get("detector_id") or "").strip(),
                        "source_code": str(detector.get("source_code") or ""),
                        "score": (
                            float(metadata.get("train_score"))
                            if isinstance(metadata.get("train_score"), (int, float))
                            else None
                        ),
                        "source_run_id": str(metadata.get("source_run_id") or "").strip(),
                        "source_run_dir": str(metadata.get("source_run_dir") or "").strip(),
                        "source_label_dir": str(metadata.get("source_label_dir") or "").strip(),
                    }
                )
            for label, members in by_label.items():
                groups.append(
                    {
                        "label": label,
                        "input_mode": "trace",
                        "members": _project_member_points(members),
                    }
                )
        groups.sort(key=lambda item: str(item.get("label") or ""))
        return groups

    def _render_library_label_detail(group: dict[str, Any], *, env_id: str, kind: str) -> str:
        label = str(group.get("label") or "").strip() or "—"
        input_mode = str(group.get("input_mode") or "").strip() or "trace"
        members = list(group.get("members") or [])
        best_score = max(
            (float(member.get("score")) for member in members if isinstance(member.get("score"), (int, float))),
            default=None,
        )
        mean_score = None
        scored = [float(member.get("score")) for member in members if isinstance(member.get("score"), (int, float))]
        if scored:
            mean_score = sum(scored) / float(len(scored))
        plot_items: list[str] = []
        label_items: list[str] = []
        for member in members:
            x = 30.0 + float(member.get("plot_x") or 0.5) * 300.0
            y = 20.0 + (1.0 - float(member.get("plot_y") or 0.5)) * 180.0
            color = _cluster_color(int(member.get("cluster_index") or 0))
            slot_index = int(member.get("slot_index") or 0)
            cluster_index = int(member.get("cluster_index") or 0)
            plot_items.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{html.escape(color)}" fill-opacity="0.85" stroke="#18212f" stroke-width="1.5"></circle>'
            )
            plot_items.append(
                f'<text x="{x:.1f}" y="{y + 4.0:.1f}" text-anchor="middle" font-size="10" font-weight="700" fill="#ffffff">S{slot_index + 1}</text>'
            )
            score = member.get("score")
            score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else "—"
            source_run_id = str(member.get("source_run_id") or "").strip()
            source_run_dir = str(member.get("source_run_dir") or "").strip()
            source_link = (
                f'<a class="ghost-link" href="/ui/files?path={quote(source_run_dir)}">Source</a>'
                if source_run_dir
                else html.escape(source_run_id or "—")
            )
            code_text = str(member.get("source_code") or "").strip()
            label_items.append(
                f"""
                <article class="panel-static" style="padding:0.75rem; display:grid; gap:0.5rem;">
                  <div class="card-topline" style="margin:0;">
                    <div style="display:flex; gap:0.4rem; flex-wrap:wrap; align-items:center;">
                      <span class="mini-chip" style="background:{html.escape(color)}; color:#fff; border-color:{html.escape(color)};">Cluster {cluster_index + 1}</span>
                      <span class="mini-chip">Slot {slot_index + 1}</span>
                      <span class="mini-chip">Reward {html.escape(score_text)}</span>
                      <span class="mini-chip">{html.escape(source_link) if source_run_dir else source_link}</span>
                    </div>
                  </div>
                  <pre class="log-pre" style="max-height:15rem; overflow:auto; margin:0; white-space:pre-wrap;">{html.escape(code_text or 'No code available.')}</pre>
                </article>
                """
            )
        summary_html = (
            f'<div style="display:flex; gap:0.4rem; flex-wrap:wrap; margin:0 0 0.75rem;">'
            f'<span class="mini-chip">{len(members)} ensemble members</span>'
            f'<span class="mini-chip">Best {html.escape(f"{best_score:.4f}" if isinstance(best_score, float) else "—")}</span>'
            f'<span class="mini-chip">Mean {html.escape(f"{mean_score:.4f}" if isinstance(mean_score, float) else "—")}</span>'
            f'<span class="mini-chip">{html.escape("Trace" if input_mode == "trace" else "Supervised evidence")}</span>'
            f"</div>"
        )
        return f"""
        <div style="display:grid; gap:0.9rem;">
          {summary_html}
          <div style="display:grid; grid-template-columns:minmax(19rem, 0.95fr) minmax(24rem, 1.25fr); gap:0.9rem; align-items:start;">
            <section class="panel-static" style="padding:0.75rem;">
              <div class="card-topline" style="margin-bottom:0.55rem;">
                <div>
                  <h3 style="margin:0;">K-means Cluster View</h3>
                  <p class="metric-subtle">Selected ensemble members projected from cached code embeddings. Colors map to cluster index.</p>
                </div>
              </div>
              <svg viewBox="0 0 360 220" width="100%" height="220" role="img" aria-label="{html.escape(label)} cluster plot" style="background:#f7f7fb; border:1px solid #d7dde7; border-radius:12px;">
                <rect x="30" y="20" width="300" height="180" fill="#ffffff" stroke="#cfd8e3" stroke-width="1"></rect>
                <line x1="30" y1="200" x2="330" y2="200" stroke="#a8b5c7" stroke-width="1"></line>
                <line x1="30" y1="20" x2="30" y2="200" stroke="#a8b5c7" stroke-width="1"></line>
                {''.join(plot_items)}
              </svg>
            </section>
            <section class="panel-static" style="padding:0.75rem;">
              <div class="card-topline" style="margin-bottom:0.55rem;">
                <div>
                  <h3 style="margin:0;">Detector Programs</h3>
                  <p class="metric-subtle">Cluster representatives selected for this label.</p>
                </div>
              </div>
              <div style="display:grid; gap:0.65rem; max-height:36rem; overflow:auto;">
                {''.join(label_items) if label_items else '<p class="empty" style="padding:0.6rem;">No code available.</p>'}
              </div>
            </section>
          </div>
        </div>
        """

    for env_id in SUPPORTED_ENVS:
        summary = load_cached_canonical_library_summary(project_root, env_id, kind)
        rows = summary.get("rows") if isinstance(summary.get("rows"), list) else []
        error_text = str(summary.get("error") or "").strip()
        selection = load_annotation_video_selection(project_root, env_id) if kind == "supervised" else {}
        selected_labels = set(str(item).strip() for item in selection.get("selected_labels") or [])
        library_path_text = str(summary.get("library_path") or "").strip()
        library_groups = _library_label_groups(Path(library_path_text)) if library_path_text else []
        table_rows: list[str] = []
        if library_groups:
            for group in library_groups:
                label = str(group.get("label") or "").strip() or "—"
                members = list(group.get("members") or [])
                scored = [float(member.get("score")) for member in members if isinstance(member.get("score"), (int, float))]
                best_score = max(scored) if scored else None
                mean_score = (sum(scored) / float(len(scored))) if scored else None
                detail_id = "canonical-lib-label-" + hashlib.sha1(
                    f"{kind}:{env_id}:{label}".encode("utf-8")
                ).hexdigest()[:16]
                detail_button = (
                    f'<button class="ghost-link js-library-detail-open" type="button" '
                    f'data-library-detail-title="{html.escape(env_id.upper())} · {html.escape(label)}" '
                    f'data-library-detail-source="{detail_id}">View</button>'
                )
                detail_payload_nodes.append(
                    f'<template id="{detail_id}">{_render_library_label_detail(group, env_id=env_id, kind=kind)}</template>'
                )
                score_text = (
                    f"{best_score:.4f}"
                    + (f" · mean {mean_score:.4f}" if isinstance(mean_score, float) else "")
                    if isinstance(best_score, float)
                    else "—"
                )
                member_text = f"{len(members)} codes"
                if members:
                    clusters = len({int(member.get("cluster_index") or 0) for member in members})
                    member_text += f" · {clusters} clusters"
                source_run_dirs = [str(member.get("source_run_dir") or "").strip() for member in members if str(member.get("source_run_dir") or "").strip()]
                source_text = (
                    f'<a class="ghost-link" href="/ui/files?path={quote(source_run_dirs[0])}">Source</a>'
                    if source_run_dirs
                    else "—"
                )
                selector_cell = ""
                selector_head = ""
                if include_selection_controls and kind == "supervised":
                    selector_head = "<th>Use</th>"
                    selector_cell = (
                        f'<td><input type="checkbox" name="selected_labels" value="{html.escape(label)}"'
                        f'{" checked" if label in selected_labels else ""}></td>'
                    )
                table_rows.append(
                    "<tr>"
                    f"{selector_cell}<td><button class=\"ghost-link js-library-detail-open\" type=\"button\" "
                    f"data-library-detail-title=\"{html.escape(env_id.upper())} · {html.escape(label)}\" "
                    f"data-library-detail-source=\"{detail_id}\">{html.escape(label)}</button></td>"
                    f"<td>{html.escape(score_text)}</td>"
                    f"<td>{html.escape(member_text)}</td>"
                    f"<td>{source_text}</td>"
                    f"<td>{detail_button}</td>"
                    "</tr>"
                )
        else:
            grouped_rows: dict[str, dict[str, Any]] = {}
            row_order: list[str] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                label = str(row.get("label") or "").strip() or "—"
                label_key = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or label.lower()
                group = grouped_rows.get(label_key)
                if group is None:
                    group = {"label": label, "members": []}
                    grouped_rows[label_key] = group
                    row_order.append(label_key)
                group["members"].append(row)
            for label_key in row_order:
                group = grouped_rows.get(label_key) or {}
                label = str(group.get("label") or "").strip() or "—"
                members = list(group.get("members") or [])
                scored = [float(member.get("score")) for member in members if isinstance(member.get("score"), (int, float))]
                best_score = max(scored) if scored else None
                mean_score = (sum(scored) / float(len(scored))) if scored else None
                score_text = (
                    f"{best_score:.4f}"
                    + (f" · mean {mean_score:.4f}" if isinstance(mean_score, float) else "")
                    if isinstance(best_score, float)
                    else "—"
                )
                code_sections: list[str] = []
                for index, member in enumerate(members, start=1):
                    code_text = str(member.get("code") or "").strip()
                    if not code_text:
                        continue
                    detector_id = str(member.get("detector_id") or "").strip()
                    header = detector_id or f"code {index}"
                    code_sections.append(f"# {header}\n{code_text}")
                code_button = "—"
                if code_sections:
                    combined_code_text = "\n\n".join(code_sections)
                    dialog_id = "canonical-lib-code-" + hashlib.sha1(
                        f"{kind}:{env_id}:{label}".encode("utf-8")
                    ).hexdigest()[:16]
                    code_button = (
                        f'<button class="ghost-link js-text-open" type="button" '
                        f'data-text-title="{html.escape(env_id.upper())} · {html.escape(label)} codes" '
                        f'data-text-source="{dialog_id}">View</button>'
                    )
                    text_payload_nodes.append(
                        f'<script type="application/json" id="{dialog_id}">{_json_script_text(combined_code_text)}</script>'
                    )
                selector_cell = ""
                selector_head = ""
                if include_selection_controls and kind == "supervised":
                    selector_head = "<th>Use</th>"
                    selector_cell = (
                        f'<td><input type="checkbox" name="selected_labels" value="{html.escape(label)}"'
                        f'{" checked" if label in selected_labels else ""}></td>'
                    )
                source_run_ids = sorted({str(member.get("source_run_id") or "").strip() for member in members if str(member.get("source_run_id") or "").strip()})
                source_run_dirs = sorted({str(member.get("source_run_dir") or "").strip() for member in members if str(member.get("source_run_dir") or "").strip()})
                source_library_paths = sorted({str(member.get("source_library_path") or "").strip() for member in members if str(member.get("source_library_path") or "").strip()})
                rank_values = [member.get("selection_rank") for member in members if isinstance(member.get("selection_rank"), int)]
                source_href = ""
                if source_run_dirs:
                    source_href = f"/ui/files?path={quote(source_run_dirs[0])}"
                elif source_library_paths:
                    source_href = f"/ui/files?path={quote(source_library_paths[0])}"
                source_text = (
                    f'<a class="ghost-link" href="{html.escape(source_href)}">Source</a>'
                    if source_href
                    else html.escape(source_run_ids[0] if source_run_ids else "—")
                )
                if len(source_run_ids) > 1:
                    source_text = f"{len(source_run_ids)} runs · {source_text}" if source_text else f"{len(source_run_ids)} runs"
                elif len(source_library_paths) > 1:
                    source_text = f"{len(source_library_paths)} libs · {source_text}" if source_text else f"{len(source_library_paths)} libs"
                if rank_values and kind != "supervised":
                    source_text = f"rank {min(rank_values)}" + (f" · {source_text}" if source_text else "")
                code_count = max(1, len(code_sections)) if members else 0
                table_rows.append(
                    "<tr>"
                    f"{selector_cell}<td>{html.escape(label)}</td><td>{html.escape(score_text)}</td>"
                    f"<td>{code_count} code{'s' if code_count != 1 else ''}</td><td>{source_text}</td><td>{code_button}</td>"
                    "</tr>"
                )
        detector_count = int(summary.get("detector_count") or len(table_rows))
        source_run_count = int(summary.get("source_run_count") or 0)
        header_copy = (
            f"{len(library_groups) if library_groups else detector_count} labels · {detector_count} patterns · {source_run_count} source runs"
            + (
                f" · {int(selection.get('selected_count') or 0)} selected"
                if include_selection_controls and kind == "supervised"
                else ""
            )
        )
        if error_text:
            body_html = f'<p class="empty" style="padding:0.6rem;">{html.escape(error_text)}</p>'
        elif table_rows:
            table_html = (
                '<div class="table-scroll compact-table" style="max-height:22rem;">'
                + '<table class="data-table compact-table">'
                + "<thead><tr>"
                + (selector_head if selector_head else "")
                + "<th>Label</th><th>Reward</th><th>Codes</th><th>Source</th><th>Detail</th></tr></thead>"
                + f"<tbody>{''.join(table_rows)}</tbody></table></div>"
            )
            if include_selection_controls and kind == "supervised":
                body_html = (
                    f"""
                    <form method="post" action="/ui/videos/annotate" class="client-settings-form" style="margin:0; display:block;">
                      <input type="hidden" name="env_id" value="{html.escape(env_id)}">
                      <input type="hidden" name="next" value="{html.escape(next_href)}">
                    """
                    + table_html
                    + "</form>"
                )
            else:
                body_html = table_html
        else:
            body_html = '<p class="empty" style="padding:0.6rem;">No patterns in canonical library.</p>'
        panels.append(
            f"""
            <section class="panel videos-cell">
              <div class="card-topline">
                <div>
                  <h2>{html.escape(env_id.upper())}</h2>
                  <p class="metric-subtle">{html.escape(header_copy)}</p>
                </div>
              </div>
              {body_html}
            </section>
            """
        )
    return "".join(panels), text_payload_nodes, detail_payload_nodes


def _render_text_modal_shell(text_payload_nodes: list[str]) -> str:
    return f"""
    <div id="text-modal" class="video-modal-overlay" aria-hidden="true">
      <div class="video-modal-card text-modal-card" role="dialog" aria-modal="true" aria-label="Task annotation preview">
        <div class="card-topline" style="margin-bottom:0.65rem;">
          <div><h2 id="text-modal-title" style="margin:0;font-size:1rem;">Text preview</h2></div>
        </div>
        <pre id="text-modal-body" class="log-pre" style="max-height:70vh; margin:0; overflow:auto; white-space:pre;"></pre>
      </div>
    </div>
    {''.join(text_payload_nodes)}
    <script>
      (() => {{
        const textModal = document.getElementById("text-modal");
        const textTitle = document.getElementById("text-modal-title");
        const textBody = document.getElementById("text-modal-body");
        if (!textModal || !textTitle || !textBody) {{
          return;
        }}
        const closeTextModal = () => {{
          textModal.classList.remove("is-open");
          textModal.setAttribute("aria-hidden", "true");
        }};
        document.querySelectorAll(".js-text-open").forEach((button) => {{
          button.addEventListener("click", () => {{
            const sourceId = button.getAttribute("data-text-source") || "";
            const title = button.getAttribute("data-text-title") || "Text preview";
            const sourceNode = sourceId ? document.getElementById(sourceId) : null;
            let text = "";
            if (sourceNode) {{
              try {{
                text = sourceNode.textContent ? JSON.parse(sourceNode.textContent) : "";
              }} catch (_error) {{
                text = sourceNode.textContent || "";
              }}
            }}
            textTitle.textContent = title;
            textBody.textContent = text;
            textModal.classList.add("is-open");
            textModal.setAttribute("aria-hidden", "false");
          }});
        }});
        textModal.addEventListener("click", (event) => {{
          if (event.target === textModal) {{
            closeTextModal();
          }}
        }});
        document.addEventListener("keydown", (event) => {{
          if (event.key === "Escape") {{
            closeTextModal();
          }}
        }});
      }})();
    </script>
    """


def _render_library_detail_modal_shell(detail_payload_nodes: list[str]) -> str:
    return f"""
    <div id="library-detail-modal" class="video-modal-overlay" aria-hidden="true">
      <div class="video-modal-card text-modal-card" role="dialog" aria-modal="true" aria-label="Library label detail">
        <div class="card-topline" style="margin-bottom:0.65rem;">
          <div>
            <h2 id="library-detail-modal-title" style="margin:0;font-size:1rem;">Library label detail</h2>
          </div>
        </div>
        <div id="library-detail-modal-body" style="max-height:78vh; overflow:auto;"></div>
      </div>
    </div>
    {''.join(detail_payload_nodes)}
    <script>
      (() => {{
        const modal = document.getElementById("library-detail-modal");
        const titleNode = document.getElementById("library-detail-modal-title");
        const bodyNode = document.getElementById("library-detail-modal-body");
        if (!modal || !titleNode || !bodyNode) {{
          return;
        }}
        const closeModal = () => {{
          modal.classList.remove("is-open");
          modal.setAttribute("aria-hidden", "true");
          bodyNode.innerHTML = "";
        }};
        document.querySelectorAll(".js-library-detail-open").forEach((button) => {{
          button.addEventListener("click", () => {{
            const sourceId = button.getAttribute("data-library-detail-source") || "";
            const sourceNode = sourceId ? document.getElementById(sourceId) : null;
            const title = button.getAttribute("data-library-detail-title") || "Library label detail";
            titleNode.textContent = title;
            bodyNode.innerHTML = sourceNode ? sourceNode.innerHTML : "<p class='empty'>No detail available.</p>";
            modal.classList.add("is-open");
            modal.setAttribute("aria-hidden", "false");
          }});
        }});
        modal.addEventListener("click", (event) => {{
          if (event.target === modal) {{
            closeModal();
          }}
        }});
        document.addEventListener("keydown", (event) => {{
          if (event.key === "Escape") {{
            closeModal();
          }}
        }});
      }})();
    </script>
    """


def _display_seconds(value: float | int | None) -> str:
    if value is None:
        return "—"
    try:
        total_seconds = max(0, int(round(float(value))))
    except (TypeError, ValueError):
        return "—"
    days, total_seconds = divmod(total_seconds, 86400)
    hours, total_seconds = divmod(total_seconds, 3600)
    minutes, seconds = divmod(total_seconds, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(f"{seconds}s")
    return " ".join(parts[:3])


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _library_ensemble_counts(library_path: Path) -> tuple[int, int]:
    payload = _load_json_object(library_path)
    ensembles = payload.get("label_ensembles")
    if isinstance(ensembles, list) and ensembles:
        label_count = 0
        slot_count = 0
        for item in ensembles:
            if not isinstance(item, dict):
                continue
            label_count += 1
            members = item.get("members")
            if isinstance(members, list):
                slot_count += len([member for member in members if isinstance(member, dict)])
        return label_count, slot_count
    detectors = payload.get("detectors")
    if not isinstance(detectors, list):
        return 0, 0
    labels = {
        str(item.get("label") or "").strip()
        for item in detectors
        if isinstance(item, dict) and str(item.get("label") or "").strip()
    }
    return len(labels), len([item for item in detectors if isinstance(item, dict)])


def _ensemble_calibration_trial_rows(output_dir: Path, *, limit: int = 48) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trials_root = output_dir / "trials"
    if not trials_root.is_dir():
        return rows
    for trial_path in sorted(trials_root.glob("trial-*/trial.json")):
        payload = _load_json_object(trial_path)
        if payload:
            rows.append(payload)
    rows.sort(key=lambda item: int(item.get("trial_index") or 0))
    return rows[-limit:]


def _library_ensemble_initials(library_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = _load_json_object(library_path)
    ensembles = payload.get("label_ensembles")
    if not isinstance(ensembles, list):
        return {}
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for item in ensembles:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        input_mode = str(item.get("input_mode") or "").strip()
        if not label or not input_mode:
            continue
        members = item.get("members")
        slot_weights: list[float] = []
        if isinstance(members, list):
            for member in members:
                if not isinstance(member, dict):
                    continue
                initial_weight = member.get("initial_weight")
                metadata = member.get("detector", {}).get("metadata") if isinstance(member.get("detector"), dict) else {}
                value = initial_weight if initial_weight is not None else (
                    metadata.get("initial_weight") if isinstance(metadata, dict) else None
                )
                try:
                    slot_weights.append(float(value))
                except (TypeError, ValueError):
                    slot_weights.append(1.0)
        lookup[(label, input_mode)] = {
            "slot_weights": slot_weights,
            "thresholds": {
                "start_time_threshold": 2,
                "end_time_threshold": 2,
                "start_location_threshold": 0.08,
                "end_location_threshold": 0.08,
            },
        }
    return lookup


def _format_weight_list(values: list[float], *, limit: int = 6) -> str:
    if not values:
        return "—"
    clipped = [f"{float(value):.2f}" for value in values[:limit]]
    if len(values) > limit:
        clipped.append(f"+{len(values) - limit}")
    return "[" + ", ".join(clipped) + "]"


def _ensemble_calibration_examples(
    *,
    library_path: Path | None,
    calibration_payload: dict[str, Any],
    limit: int = 4,
) -> list[dict[str, Any]]:
    if not isinstance(calibration_payload, dict):
        return []
    labels_payload = calibration_payload.get("labels")
    if not isinstance(labels_payload, list):
        return []
    initial_lookup = _library_ensemble_initials(library_path) if isinstance(library_path, Path) and library_path.is_file() else {}
    rows: list[dict[str, Any]] = []
    for item in labels_payload:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        input_mode = str(item.get("input_mode") or "").strip()
        if not label or not input_mode:
            continue
        thresholds = item.get("thresholds") if isinstance(item.get("thresholds"), dict) else {}
        slot_weights_raw = item.get("slot_weights")
        slot_weights = []
        if isinstance(slot_weights_raw, list):
            for value in slot_weights_raw:
                try:
                    slot_weights.append(float(value))
                except (TypeError, ValueError):
                    continue
        initial = initial_lookup.get((label, input_mode), {})
        initial_weights_raw = initial.get("slot_weights") if isinstance(initial, dict) else []
        initial_weights = initial_weights_raw if isinstance(initial_weights_raw, list) else []
        default_thresholds = initial.get("thresholds") if isinstance(initial, dict) and isinstance(initial.get("thresholds"), dict) else {
            "start_time_threshold": 2,
            "end_time_threshold": 2,
            "start_location_threshold": 0.08,
            "end_location_threshold": 0.08,
        }
        delta = 0.0
        weight_count = max(len(slot_weights), len(initial_weights))
        for index in range(weight_count):
            current_value = float(slot_weights[index]) if index < len(slot_weights) else 1.0
            initial_value = float(initial_weights[index]) if index < len(initial_weights) else 1.0
            delta += abs(current_value - initial_value)
        for threshold_key in (
            "start_time_threshold",
            "end_time_threshold",
            "start_location_threshold",
            "end_location_threshold",
        ):
            try:
                current_threshold = float(thresholds.get(threshold_key))
            except (TypeError, ValueError):
                current_threshold = float(default_thresholds.get(threshold_key) or 0.0)
            try:
                default_threshold = float(default_thresholds.get(threshold_key))
            except (TypeError, ValueError):
                default_threshold = 0.0
            delta += abs(current_threshold - default_threshold)
        rows.append(
            {
                "label": label,
                "input_mode": input_mode,
                "slot_weights": slot_weights,
                "initial_weights": initial_weights,
                "thresholds": thresholds,
                "delta": delta,
            }
        )
    rows.sort(key=lambda item: (float(item.get("delta") or 0.0), str(item.get("label") or "")), reverse=True)
    return rows[:limit]


def _render_ensemble_calibration_score_plot(
    trial_rows: list[dict[str, Any]],
    *,
    width: int = 420,
    height: int = 170,
) -> str:
    scores: list[float] = []
    running_best: list[float] = []
    best_so_far = float("-inf")
    for item in trial_rows:
        score = item.get("score")
        if not isinstance(score, (int, float)):
            continue
        score_value = float(score)
        scores.append(score_value)
        best_so_far = max(best_so_far, score_value)
        running_best.append(best_so_far)
    if not scores:
        return '<p class="empty" style="padding:0.6rem;">No score history yet.</p>'
    left = 42.0
    right = 16.0
    top = 14.0
    bottom = 28.0
    plot_width = max(40.0, float(width) - left - right)
    plot_height = max(40.0, float(height) - top - bottom)
    low = min(scores + running_best)
    high = max(scores + running_best)
    if abs(high - low) < 1e-9:
        low -= 0.02
        high += 0.02

    def _x(index: int) -> float:
        if len(scores) <= 1:
            return left + plot_width / 2.0
        return left + (float(index) / float(len(scores) - 1)) * plot_width

    def _y(value: float) -> float:
        return top + (1.0 - ((float(value) - low) / (high - low))) * plot_height

    score_points = " ".join(f"{_x(index):.1f},{_y(value):.1f}" for index, value in enumerate(scores))
    best_points = " ".join(f"{_x(index):.1f},{_y(value):.1f}" for index, value in enumerate(running_best))
    dots = "".join(
        f'<circle cx="{_x(index):.1f}" cy="{_y(value):.1f}" r="3.5" fill="#2e86ab" stroke="#ffffff" stroke-width="1.2"></circle>'
        for index, value in enumerate(scores)
    )
    tick_values = [low, (low + high) / 2.0, high]
    grid_lines = "".join(
        f'<line x1="{left:.1f}" y1="{_y(value):.1f}" x2="{left + plot_width:.1f}" y2="{_y(value):.1f}" stroke="#d8dee9" stroke-width="1"></line>'
        for value in tick_values
    )
    tick_labels = "".join(
        f'<text x="{left - 8:.1f}" y="{_y(value) + 4.0:.1f}" text-anchor="end" font-size="10" fill="#5f6b7a">{value:.2f}</text>'
        for value in tick_values
    )
    x_labels = (
        f'<text x="{left:.1f}" y="{height - 8:.1f}" text-anchor="start" font-size="10" fill="#5f6b7a">1</text>'
        f'<text x="{left + plot_width:.1f}" y="{height - 8:.1f}" text-anchor="end" font-size="10" fill="#5f6b7a">{len(scores)}</text>'
    )
    final_best = running_best[-1] if running_best else None
    legend = (
        '<div style="display:flex; gap:0.75rem; flex-wrap:wrap; margin-top:0.45rem;">'
        '<span class="mini-chip"><span style="display:inline-block;width:0.8rem;height:0.8rem;border-radius:999px;background:#2e86ab;margin-right:0.35rem;vertical-align:-0.1rem;"></span>Trial reward</span>'
        '<span class="mini-chip"><span style="display:inline-block;width:1rem;height:0;border-top:2px dashed #d1495b;margin-right:0.35rem;vertical-align:0.2rem;"></span>Best so far</span>'
        + (
            f'<span class="mini-chip">Improvement {final_best - scores[0]:+.3f}</span>'
            if final_best is not None and scores
            else ""
        )
        + '</div>'
    )
    return f"""
    <div>
      <svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="BO evaluation score over time" style="background:#f7f9fc; border:1px solid #d7dde7; border-radius:12px;">
        <rect x="{left:.1f}" y="{top:.1f}" width="{plot_width:.1f}" height="{plot_height:.1f}" fill="#ffffff" stroke="#cfd8e3" stroke-width="1"></rect>
        {grid_lines}
        <line x1="{left:.1f}" y1="{top + plot_height:.1f}" x2="{left + plot_width:.1f}" y2="{top + plot_height:.1f}" stroke="#98a6b8" stroke-width="1.2"></line>
        <line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{top + plot_height:.1f}" stroke="#98a6b8" stroke-width="1.2"></line>
        <polyline points="{score_points}" fill="none" stroke="#2e86ab" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"></polyline>
        <polyline points="{best_points}" fill="none" stroke="#d1495b" stroke-width="2" stroke-dasharray="6 4" stroke-linecap="round" stroke-linejoin="round"></polyline>
        {dots}
        {tick_labels}
        {x_labels}
        <text x="{left + plot_width / 2.0:.1f}" y="{height - 8:.1f}" text-anchor="middle" font-size="10" fill="#5f6b7a">Trial</text>
      </svg>
      {legend}
    </div>
    """


def _library_deepphy_eval_runs_root(*, library_path: Path, phase: str) -> Path:
    return library_path.parent / "ensemble_eval_runs" / phase


def _library_deepphy_eval_job_phase(job: JobRecord, *, library_path: Path) -> str | None:
    if str(job.spec.action or "").strip() != "evaluate-deepphy":
        return None
    pattern_run_dir = Path(str((job.spec.metadata or {}).get("pattern_run_dir") or "")).resolve()
    root = library_path.parent.resolve() / "ensemble_eval_runs"
    try:
        relative_parts = pattern_run_dir.relative_to(root).parts
    except Exception:
        return None
    if len(relative_parts) < 2:
        return None
    phase = str(relative_parts[0] or "").strip().lower()
    return phase if phase in {"baseline", "calibrated"} else None


def _library_deepphy_eval_run_summary(eval_root: Path) -> dict[str, Any] | None:
    payload = _load_json_object(eval_root / "results.json")
    task_count = 0
    success_count = 0
    if isinstance(payload.get("results"), list):
        results = [item for item in payload.get("results", []) if isinstance(item, dict)]
        task_count = len(results)
        success_count = sum(1 for item in results if bool(item.get("success")))
    if task_count == 0:
        env_summaries = payload.get("env_summaries") or payload.get("environments")
        if isinstance(env_summaries, list):
            for item in env_summaries:
                if not isinstance(item, dict):
                    continue
                task_count += int(item.get("task_count") or 0)
                success_count += int(item.get("success_count") or 0)
    if task_count <= 0:
        return None
    results_path = eval_root / "results.json"
    finished_at = (
        datetime.fromtimestamp(results_path.stat().st_mtime, tz=DISPLAY_TZ)
        if results_path.is_file()
        else datetime.fromtimestamp(eval_root.stat().st_mtime, tz=DISPLAY_TZ)
    )
    return {
        "eval_root": eval_root,
        "task_count": task_count,
        "success_count": success_count,
        "success_rate": success_count / float(task_count),
        "finished_at": finished_at,
        "run_id": str(payload.get("run_id") or eval_root.name),
    }


def _library_deepphy_eval_history(*, library_path: Path, phase: str) -> list[dict[str, Any]]:
    phase_root = _library_deepphy_eval_runs_root(library_path=library_path, phase=phase)
    rows: list[dict[str, Any]] = []
    if not phase_root.is_dir():
        return rows
    for results_path in sorted(phase_root.glob("*/deepphy_eval/*/results.json")):
        summary = _library_deepphy_eval_run_summary(results_path.parent)
        if summary is None:
            continue
        snapshot_root = results_path.parents[2]
        manifest = _load_json_object(snapshot_root / "ensemble_eval_manifest.json")
        summary["job_id"] = str(manifest.get("job_id") or "").strip()
        summary["snapshot_root"] = snapshot_root
        rows.append(summary)
    rows.sort(key=lambda item: item["finished_at"], reverse=True)
    return rows


def _library_deepphy_eval_stats(
    *,
    library_path: Path,
    phase: str,
    jobs: list[JobRecord],
) -> dict[str, Any]:
    history = _library_deepphy_eval_history(library_path=library_path, phase=phase)
    success_rates = [float(item.get("success_rate") or 0.0) for item in history]
    mean_success = statistics.fmean(success_rates) if success_rates else None
    std_success = float(statistics.pstdev(success_rates)) if len(success_rates) > 1 else 0.0
    active_jobs = [
        job for job in jobs
        if _library_deepphy_eval_job_phase(job, library_path=library_path) == phase
        and str(getattr(job.status, "value", job.status) or "").strip().lower() in ACTIVE_QUEUE_STATUSES
    ]
    latest = history[0] if history else None
    return {
        "history": history,
        "completed_runs": len(history),
        "mean_success": mean_success,
        "std_success": std_success,
        "latest_success": float(latest.get("success_rate")) if isinstance(latest, dict) and isinstance(latest.get("success_rate"), (int, float)) else None,
        "latest_finished_at": latest.get("finished_at") if isinstance(latest, dict) else None,
        "latest_run_id": str(latest.get("run_id") or "") if isinstance(latest, dict) else "",
        "active_jobs": active_jobs,
    }


def _render_library_deepphy_eval_block(
    *,
    env_id: str,
    kind: str,
    phase: str,
    library_path: Path | None,
    jobs: list[JobRecord],
    sidecar_exists: bool,
) -> str:
    phase_title = "Before BO" if phase == "baseline" else "After BO"
    if not isinstance(library_path, Path) or not library_path.is_file():
        return (
            f'<div class="panel-static" style="padding:0.75rem;">'
            f'<div class="card-topline"><div><h3 style="margin:0;">{html.escape(phase_title)} · Full DeepPHY</h3>'
            f'<p class="metric-subtle">Canonical library missing.</p></div></div>'
            f'<p class="empty" style="padding:0.6rem;">Recalculate the canonical library first.</p>'
            f'</div>'
        )
    if phase == "calibrated" and not sidecar_exists:
        return (
            f'<div class="panel-static" style="padding:0.75rem;">'
            f'<div class="card-topline"><div><h3 style="margin:0;">{html.escape(phase_title)} · Full DeepPHY</h3>'
            f'<p class="metric-subtle">Uses the current ensemble calibration sidecar.</p></div></div>'
            f'<p class="empty" style="padding:0.6rem;">No <code>ensemble_calibration.json</code> has been written yet.</p>'
            f'</div>'
        )
    stats_payload = _library_deepphy_eval_stats(library_path=library_path, phase=phase, jobs=jobs)
    mean_success = stats_payload.get("mean_success")
    std_success = float(stats_payload.get("std_success") or 0.0)
    latest_success = stats_payload.get("latest_success")
    latest_finished_at = stats_payload.get("latest_finished_at")
    latest_run_id = str(stats_payload.get("latest_run_id") or "").strip()
    active_jobs = stats_payload.get("active_jobs") if isinstance(stats_payload.get("active_jobs"), list) else []
    history = stats_payload.get("history") if isinstance(stats_payload.get("history"), list) else []
    rows = []
    for item in history[:4]:
        finished_at = item.get("finished_at")
        finished_text = finished_at.strftime("%Y-%m-%d %H:%M") if isinstance(finished_at, datetime) else "—"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('run_id') or ''))}</td>"
            f"<td>{float(item.get('success_rate') or 0.0) * 100.0:.1f}%</td>"
            f"<td>{int(item.get('success_count') or 0)}/{int(item.get('task_count') or 0)}</td>"
            f"<td>{html.escape(finished_text)}</td>"
            "</tr>"
        )
    history_html = (
        '<div class="table-scroll compact-table" style="max-height:8.5rem;">'
        '<table class="data-table compact-table">'
        '<thead><tr><th>Run</th><th>Success</th><th>Tasks</th><th>Finished</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        if rows
        else '<p class="empty" style="padding:0.6rem;">No completed full DeepPHY runs yet.</p>'
    )
    active_text = (
        f"{len(active_jobs)} active job(s)"
        if active_jobs
        else "No active jobs"
    )
    run_button = f"""
    <form method="post" action="/ui/library/deepphy-eval/start" class="client-settings-form" style="margin:0;">
      <input type="hidden" name="env_id" value="{html.escape(env_id)}">
      <input type="hidden" name="kind" value="{html.escape(kind)}">
      <input type="hidden" name="phase" value="{html.escape(phase)}">
      <input type="hidden" name="next" value="/library">
      <button class="secondary-button" type="submit" style="width:100%;">Run full eval</button>
    </form>
    """
    summary_line = (
        f"Mean {float(mean_success) * 100.0:.1f}% ± {std_success * 100.0:.1f}%"
        if isinstance(mean_success, (int, float))
        else "Mean —"
    )
    latest_line = (
        f"Latest {float(latest_success) * 100.0:.1f}%"
        if isinstance(latest_success, (int, float))
        else "Latest —"
    )
    latest_when = (
        latest_finished_at.strftime("%Y-%m-%d %H:%M") if isinstance(latest_finished_at, datetime) else "—"
    )
    return f"""
    <div class="panel-static" style="padding:0.75rem;">
      <div class="card-topline" style="margin-bottom:0.45rem;">
        <div>
          <h3 style="margin:0;">{html.escape(phase_title)} · Full DeepPHY</h3>
          <p class="metric-subtle">{html.escape(active_text)} · {int(stats_payload.get("completed_runs") or 0)} completed run(s)</p>
        </div>
        <div style="display:flex; gap:0.35rem; flex-wrap:wrap; justify-content:flex-end;">
          <span class="mini-chip">{html.escape(summary_line)}</span>
          <span class="mini-chip">{html.escape(latest_line)}</span>
        </div>
      </div>
      <div class="metric-subtle" style="margin-bottom:0.45rem;">Latest finished {html.escape(latest_when)}{f' · {html.escape(latest_run_id)}' if latest_run_id else ''}</div>
      {run_button}
      <div style="margin-top:0.55rem;">{history_html}</div>
    </div>
    """


def _render_ensemble_calibration_card(
    *,
    project_root: Path,
    env_id: str,
    kind: str,
    entry: dict[str, Any],
    jobs: list[JobRecord],
) -> str:
    request_payload = dict(entry.get("request") or {}) if isinstance(entry.get("request"), dict) else {}
    library_path_text = str(entry.get("library_path") or "").strip()
    if not library_path_text:
        summary = load_cached_canonical_library_summary(project_root, env_id, kind)
        library_path_text = str(summary.get("library_path") or "").strip()
    library_path = Path(library_path_text) if library_path_text else None
    library_exists = isinstance(library_path, Path) and library_path.is_file()
    output_dir_text = str(entry.get("output_dir") or "").strip()
    output_dir = Path(output_dir_text) if output_dir_text else None
    progress_payload = dict(entry.get("progress") or {}) if isinstance(entry.get("progress"), dict) else {}
    summary_payload = dict(entry.get("summary") or {}) if isinstance(entry.get("summary"), dict) else {}
    best_payload = dict(entry.get("best") or {}) if isinstance(entry.get("best"), dict) else {}
    if not progress_payload and output_dir is not None:
        progress_payload = _load_json_object(output_dir / "progress.json")
    if not summary_payload and output_dir is not None:
        summary_payload = _load_json_object(output_dir / "summary.json")
    if not best_payload and output_dir is not None:
        best_payload = _load_json_object(output_dir / "best.json")
    trial_rows = _ensemble_calibration_trial_rows(output_dir) if output_dir is not None else []
    status_text = str(entry.get("status") or "idle").strip().lower() or "idle"
    status_label = status_text.replace("_", " ")
    completed_trials = int(progress_payload.get("completed_trials") or summary_payload.get("trial_count") or 0)
    total_trials = int(progress_payload.get("total_trials") or request_payload.get("trials") or 0)
    best_score = best_payload.get("score")
    if not isinstance(best_score, (int, float)):
        best_score = summary_payload.get("best_score")
    last_score = progress_payload.get("last_score")
    average_trial_seconds = progress_payload.get("average_trial_seconds")
    eta_seconds = progress_payload.get("eta_seconds")
    sidecar_exists = bool(library_exists and (library_path.parent / "ensemble_calibration.json").is_file())
    calibration_payload = {}
    if best_payload:
        best_path = Path(str(best_payload.get("calibration_path") or "").strip()) if str(best_payload.get("calibration_path") or "").strip() else None
        if isinstance(best_path, Path) and best_path.is_file():
            calibration_payload = _load_json_object(best_path)
    if not calibration_payload and output_dir is not None:
        calibration_payload = _load_json_object(output_dir / "ensemble_calibration.json")
    if not calibration_payload and library_exists:
        calibration_payload = _load_json_object(library_path.parent / "ensemble_calibration.json")
    learned_examples = _ensemble_calibration_examples(
        library_path=library_path if library_exists else None,
        calibration_payload=calibration_payload,
    )
    baseline_eval_html = _render_library_deepphy_eval_block(
        env_id=env_id,
        kind=kind,
        phase="baseline",
        library_path=library_path if library_exists else None,
        jobs=jobs,
        sidecar_exists=sidecar_exists,
    )
    calibrated_eval_html = _render_library_deepphy_eval_block(
        env_id=env_id,
        kind=kind,
        phase="calibrated",
        library_path=library_path if library_exists else None,
        jobs=jobs,
        sidecar_exists=sidecar_exists,
    )
    label_count, slot_count = _library_ensemble_counts(library_path) if library_exists else (0, 0)
    parameter_count = slot_count
    active = status_text in {"starting", "running"}
    next_label = "Start ~6h BO"
    action_html = ""
    if library_exists:
        if active:
            action_html = f"""
            <form method="post" action="/ui/library/ensemble-calibration/cancel" class="client-settings-form" style="margin:0;">
              <input type="hidden" name="env_id" value="{html.escape(env_id)}">
              <input type="hidden" name="kind" value="{html.escape(kind)}">
              <input type="hidden" name="next" value="/library">
              <button class="secondary-button" type="submit" style="width:100%;">Cancel</button>
            </form>
            """
        else:
            action_html = f"""
            <form method="post" action="/ui/library/ensemble-calibration/start" class="client-settings-form" style="margin:0; display:grid; gap:0.55rem;">
              <input type="hidden" name="env_id" value="{html.escape(env_id)}">
              <input type="hidden" name="kind" value="{html.escape(kind)}">
              <input type="hidden" name="next" value="/library">
              <div style="display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:0.45rem;">
                <label class="metric-subtle">Trials<input type="number" min="1" name="trials" value="{html.escape(str(request_payload.get('trials') or 12))}" style="width:100%;"></label>
                <label class="metric-subtle">Random<input type="number" min="1" name="random_init_trials" value="{html.escape(str(request_payload.get('random_init_trials') or 4))}" style="width:100%;"></label>
                <label class="metric-subtle">Workers<input type="number" min="1" name="workers" value="{html.escape(str(request_payload.get('workers') or (1 if env_id == 'pooltool' else 2)))}" style="width:100%;"></label>
                <label class="metric-subtle">Seed<input type="number" min="0" name="seed" value="{html.escape(str(request_payload.get('seed') or 0))}" style="width:100%;"></label>
              </div>
              <button class="secondary-button" type="submit" style="width:100%;">{html.escape(next_label)}</button>
            </form>
            """
    else:
        action_html = '<p class="empty" style="padding:0.6rem;">No canonical library available yet. Recalculate <a href="/library">/library</a> first.</p>'

    score_plot_html = _render_ensemble_calibration_score_plot(trial_rows)
    if trial_rows:
        rows_html = []
        running_best = float("-inf")
        for item in trial_rows:
            score = item.get("score")
            score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else "—"
            if isinstance(score, (int, float)):
                running_best = max(running_best, float(score))
            best_text = f"{running_best:.3f}" if running_best != float("-inf") else "—"
            rows_html.append(
                "<tr>"
                f"<td>{int(item.get('trial_index') or 0) + 1}</td>"
                f"<td>{html.escape(str(item.get('proposal_mode') or 'guided'))}</td>"
                f"<td>{html.escape(score_text)}</td>"
                f"<td>{html.escape(best_text)}</td>"
                f"<td>{html.escape(_display_seconds(item.get('trial_seconds')))}</td>"
                "</tr>"
            )
        updates_html = (
            '<div class="table-scroll compact-table" style="max-height:10rem;">'
            '<table class="data-table compact-table">'
            '<thead><tr><th>#</th><th>Mode</th><th>Reward</th><th>Best</th><th>Duration</th></tr></thead>'
            f"<tbody>{''.join(rows_html)}</tbody></table></div>"
        )
    else:
        updates_html = '<p class="empty" style="padding:0.6rem;">No BO trial updates yet.</p>'

    if learned_examples:
        example_rows = []
        for item in learned_examples:
            thresholds = item.get("thresholds") if isinstance(item.get("thresholds"), dict) else {}
            input_mode_text = "Trace" if str(item.get("input_mode") or "") == "trace" else "Unsupervised"
            start_loc_raw = thresholds.get("start_location_threshold")
            end_loc_raw = thresholds.get("end_location_threshold")
            start_loc_text = f"{float(start_loc_raw):.2f}" if isinstance(start_loc_raw, (int, float)) else "—"
            end_loc_text = f"{float(end_loc_raw):.2f}" if isinstance(end_loc_raw, (int, float)) else "—"
            example_rows.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('label') or ''))}</td>"
                f"<td>{html.escape(input_mode_text)}</td>"
                f"<td>{html.escape(_format_weight_list(item.get('slot_weights') if isinstance(item.get('slot_weights'), list) else []))}</td>"
                f"<td>{html.escape(_format_weight_list(item.get('initial_weights') if isinstance(item.get('initial_weights'), list) else []))}</td>"
                f"<td>{html.escape(str(thresholds.get('start_time_threshold', '—')))} / {html.escape(str(thresholds.get('end_time_threshold', '—')))}</td>"
                f"<td>{html.escape(start_loc_text)} / {html.escape(end_loc_text)}</td>"
                "</tr>"
            )
        learned_html = (
            '<div class="table-scroll compact-table" style="max-height:14rem;">'
            '<table class="data-table compact-table">'
            '<thead><tr><th>Label</th><th>Mode</th><th>Learned weights</th><th>Initial weights</th><th>Time thresh</th><th>Loc thresh</th></tr></thead>'
            f"<tbody>{''.join(example_rows)}</tbody></table></div>"
        )
    else:
        learned_html = '<p class="empty" style="padding:0.6rem;">No learned calibration values yet.</p>'

    message_text = str(entry.get("message") or "").strip()
    compact_metrics = [
        f'<span class="mini-chip">{html.escape(status_label.title())}</span>',
        f'<span class="mini-chip">{completed_trials}/{total_trials or int(request_payload.get("trials") or 0)} trials</span>',
        f'<span class="mini-chip">Best {html.escape(f"{float(best_score):.3f}" if isinstance(best_score, (int, float)) else "—")}</span>',
        f'<span class="mini-chip">ETA {html.escape(_display_seconds(eta_seconds))}</span>',
    ]
    compact_meta = (
        f"{label_count} labels · {slot_count} detectors · {parameter_count} params · "
        f"Last {html.escape(f'{float(last_score):.3f}' if isinstance(last_score, (int, float)) else '—')} · "
        f"Avg {html.escape(_display_seconds(average_trial_seconds))} · "
        f"Sidecar {'yes' if sidecar_exists else 'no'}"
    )
    run_id = str(entry.get("run_id") or "").strip()
    output_href = (
        f"/ui/files?path={quote(str(output_dir))}"
        if output_dir is not None and output_dir.exists()
        else ""
    )
    output_link = (
        f'<a class="ghost-link" href="{html.escape(output_href)}">Run files</a>'
        if output_href
        else ""
    )
    return f"""
    <section class="panel videos-cell">
      <div class="card-topline">
        <div>
          <h2>{html.escape(env_id.upper())}</h2>
          <p class="metric-subtle">{html.escape(kind.title())} ensemble BO · calibrate detector weights with fixed consensus thresholds.</p>
        </div>
        <div style="display:flex; gap:0.35rem; flex-wrap:wrap; justify-content:flex-end;">
          {''.join(compact_metrics)}
        </div>
      </div>
      <div style="display:grid; gap:0.45rem;">
        <div class="metric-subtle">{html.escape(message_text or ('Ready to start local ensemble calibration.' if library_exists else 'Waiting for canonical library.'))}{f' · {html.escape(run_id)}' if run_id else ''}{f' · {output_link}' if output_link else ''}</div>
        <div class="metric-subtle">{compact_meta}</div>
        {action_html}
        <div style="display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:0.55rem;">
          {baseline_eval_html}
          {calibrated_eval_html}
        </div>
        <div class="panel-static" style="padding:0.75rem;">
          <div class="card-topline" style="margin-bottom:0.5rem;">
            <div>
              <h3 style="margin:0;">Eval Score Over Time</h3>
              <p class="metric-subtle">Improvement in summary-discrimination accuracy across BO trials.</p>
            </div>
          </div>
          {score_plot_html}
        </div>
        <div class="panel-static" style="padding:0.75rem;">
          <div class="card-topline" style="margin-bottom:0.5rem;">
            <div>
              <h3 style="margin:0;">Recent BO Updates</h3>
              <p class="metric-subtle">Compact trial log for the current run.</p>
            </div>
          </div>
          {updates_html}
        </div>
        <div class="panel-static" style="padding:0.75rem;">
          <div class="card-topline" style="margin-bottom:0.5rem;">
            <div>
              <h3 style="margin:0;">Learned Examples</h3>
              <p class="metric-subtle">A few learned detector weights and the fixed consensus thresholds used for the best finished trial so far.</p>
            </div>
          </div>
          {learned_html}
        </div>
      </div>
    </section>
    """


def _render_ensemble_calibration_sections(
    *,
    project_root: Path,
    ensemble_calibration_state: dict[str, Any] | None,
    jobs: list[JobRecord],
) -> tuple[str, bool]:
    snapshot = ensemble_calibration_state if isinstance(ensemble_calibration_state, dict) else {}
    entries = snapshot.get("entries") if isinstance(snapshot, dict) else {}
    if not isinstance(entries, dict):
        entries = {}
    sections: list[str] = []
    active = False
    for kind in ("supervised", "unsupervised"):
        cards: list[str] = []
        for env_id in SUPPORTED_ENVS:
            key = f"{kind}:{env_id}"
            entry = entries.get(key)
            payload = entry if isinstance(entry, dict) else {
                "env_id": env_id,
                "kind": kind,
                "status": "idle",
                "request": {},
            }
            status_text = str(payload.get("status") or "").strip().lower()
            if status_text in {"starting", "running"}:
                active = True
            cards.append(
                _render_ensemble_calibration_card(
                    project_root=project_root,
                    env_id=env_id,
                    kind=kind,
                    entry=payload,
                    jobs=jobs,
                )
            )
        sections.append(
            f"""
            <section class="panel">
              <div class="card-topline">
                <div>
                  <h2>{html.escape(kind.title())} Ensemble BO</h2>
                  <p class="metric-subtle">Learn per-label detector reliability weights with fixed consensus thresholds using local summary-discrimination sweeps. Defaults target roughly six hours.</p>
                </div>
              </div>
            </section>
            <section class="videos-grid">
              {''.join(cards)}
            </section>
            """
        )
    return "".join(sections), active


def render_videos_page(
    *,
    project_root: Path,
) -> str:
    tasks_by_env = _collect_video_showcase_tasks(project_root=project_root)
    for env_id in SUPPORTED_ENVS:
        tasks_by_env.setdefault(env_id, {"success": [], "failure": []})
    env_cells: list[str] = []
    text_payload_nodes: list[str] = []
    filter_rows: list[str] = []
    for env_id in SUPPORTED_ENVS:
        descriptions = annotation_video_filter_descriptions(env_id)
        if not isinstance(descriptions, list):
            continue
        for description in descriptions:
            text = str(description).strip()
            if text:
                filter_rows.append(f"<li><strong>{html.escape(env_id.upper())}</strong>: {html.escape(text)}</li>")
    for env_id in SUPPORTED_ENVS:
        summary = load_cached_global_best_supervised_library_summary(project_root, env_id)
        selection = load_annotation_video_selection(project_root, env_id)
        library_rows = summary.get("rows") if isinstance(summary.get("rows"), list) else []
        available_labels = [str(item).strip() for item in selection.get("available_labels") or [] if str(item).strip()]
        detector_count = int(summary.get("detector_count") or len(library_rows))
        source_run_count = int(summary.get("source_run_count") or 0)
        label_items = []
        selected_labels = set(str(item).strip() for item in selection.get("selected_labels") or [])
        for label in available_labels:
            checked = " checked" if label in selected_labels else ""
            label_items.append(
                f'<label class="mini-chip" style="display:flex; gap:0.35rem; align-items:center; justify-content:flex-start;">'
                f'<input type="checkbox" name="selected_labels" value="{html.escape(label)}"{checked}>'
                f'<span>{html.escape(label)}</span>'
                f"</label>"
            )
        if available_labels:
            controls_html = (
                f"""
                <form method="post" action="/ui/videos/annotate" class="client-settings-form" style="margin:0; display:block;">
                  <input type="hidden" name="env_id" value="{html.escape(env_id)}">
                  <input type="hidden" name="next" value="/videos">
                  <button class="secondary-button" type="submit" style="width:100%;">Render annotations</button>
                  <div class="table-scroll" style="max-height:12rem; overflow:auto;">
                    <div style="display:flex; flex-wrap:wrap; gap:0.45rem;">
                      {"".join(label_items)}
                    </div>
                  </div>
                </form>
                """
            )
        else:
            controls_html = '<p class="empty" style="padding:0.6rem;">No supervised canonical library available. Open <a href="/library">/library</a> for details.</p>'
        groups = tasks_by_env.get(env_id, {"success": [], "failure": []})
        success_rows = groups.get("success", [])
        failure_rows = groups.get("failure", [])
        total_rows = len(success_rows) + len(failure_rows)
        rendered_sections: list[str] = []
        for label, rows in (("Successes", success_rows), ("Failures", failure_rows)):
            rendered_rows: list[str] = []
            for _updated_at, task_html, dialog_node in rows:
                rendered_rows.append(task_html)
                if dialog_node:
                    text_payload_nodes.append(dialog_node)
            rendered_sections.append(
                f"""
                <div style="display:grid; gap:0.35rem; padding:0.55rem 0.6rem; border-bottom:1px solid var(--line);">
                  <div class="metric-subtle" style="font-weight:800;">{html.escape(label)} · {len(rows)}</div>
                  {''.join(rendered_rows) if rendered_rows else '<p class="empty" style="margin:0;">None yet.</p>'}
                </div>
                """
            )
        env_cells.append(
            f"""
            <section class="panel videos-cell">
              <div class="card-topline">
                <div>
                  <h2>{html.escape(env_id.upper())}</h2>
                  <p class="metric-subtle">{total_rows} videos · {len(success_rows)} success · {len(failure_rows)} failure · {detector_count} supervised labels · {source_run_count} source runs</p>
                </div>
              </div>
              {controls_html}
              <div class="videos-task-list">
                {''.join(rendered_sections) if total_rows else '<p class="empty" style="padding: 0.6rem;">No sampled task videos found.</p>'}
              </div>
            </section>
            """
        )
    body = f"""
    <section class="panel">
      <div class="card-topline">
        <div>
          <h2>Videos</h2>
          <p class="metric-subtle">Annotation-video controls use the canonical supervised libraries from <a href="/library">/library</a>.</p>
        </div>
      </div>
      <div style="margin-top:0.7rem;">
        <div class="metric-subtle" style="font-weight:800;">Active annotation filters</div>
        {
            '<ul style="margin:0.35rem 0 0; padding-left:1.1rem;">' + ''.join(filter_rows) + '</ul>'
            if filter_rows
            else '<p class="metric-subtle" style="margin:0.35rem 0 0;">None.</p>'
        }
      </div>
    </section>
    <section class="videos-grid">
      {''.join(env_cells)}
    </section>
    <div id="video-modal" class="video-modal-overlay" aria-hidden="true">
      <div class="video-modal-card" role="dialog" aria-modal="true" aria-label="Task video preview">
        <button id="video-modal-download" class="video-modal-download" type="button" title="Download video">&#8681;</button>
        <div class="video-modal-controls">
          <label for="video-modal-confidence">
            <span>Min confidence</span>
            <input id="video-modal-confidence" type="range" min="0" max="1" step="0.01" value="0">
          </label>
          <span id="video-modal-confidence-value" class="video-modal-confidence-value">0.00</span>
        </div>
        <div class="video-modal-frame">
          <div id="video-modal-stage" class="video-modal-stage">
            <video id="video-modal-player" controls preload="metadata" src="" loop></video>
            <img id="video-modal-image" alt="" hidden>
            <canvas id="video-modal-overlay" hidden></canvas>
          </div>
        </div>
      </div>
    </div>
    {_render_text_modal_shell(text_payload_nodes)}
    <script>
      (() => {{
        const modal = document.getElementById("video-modal");
        const stage = document.getElementById("video-modal-stage");
        const player = document.getElementById("video-modal-player");
        const image = document.getElementById("video-modal-image");
        const overlay = document.getElementById("video-modal-overlay");
        const overlayCtx = overlay instanceof HTMLCanvasElement ? overlay.getContext("2d") : null;
        const download = document.getElementById("video-modal-download");
        const confidenceSlider = document.getElementById("video-modal-confidence");
        const confidenceValue = document.getElementById("video-modal-confidence-value");
        let overlayPayload = null;
        let currentOverlayUrl = "";
        let currentVideoTitle = "video";
        let currentVideoSrc = "";
        let currentConfidenceThreshold = 0;
        let overlayAnimationFrame = 0;
        if (!modal || !stage || !player || !image || !overlay || !overlayCtx || !(download instanceof HTMLButtonElement) || !(confidenceSlider instanceof HTMLInputElement) || !confidenceValue) {{
          return;
        }}
        player.loop = true;
        const triggerBrowserDownload = (href, filename) => {{
          const link = document.createElement("a");
          link.href = href;
          if (filename) {{
            link.download = filename;
          }}
          document.body.appendChild(link);
          link.click();
          link.remove();
        }};
        const setConfidenceThreshold = (value) => {{
          const next = Math.max(0, Math.min(1, Number(value || 0)));
          currentConfidenceThreshold = next;
          confidenceSlider.value = next.toFixed(2);
          confidenceValue.textContent = next.toFixed(2);
        }};
        const filteredOverlayEvents = (payload, now) => {{
          return Array.isArray(payload?.events)
            ? payload.events.filter((event) => {{
                const confidence = Number(event && event.confidence != null ? event.confidence : 1);
                return (
                  now >= Number(event?.start_time || 0)
                  && now <= Number(event?.end_time || 0)
                  && confidence >= currentConfidenceThreshold
                );
              }})
            : [];
        }};
        const syncOverlayCanvas = () => {{
          const media = player.hidden ? image : player;
          const width = media.clientWidth || media.videoWidth || media.naturalWidth || 0;
          const height = media.clientHeight || media.videoHeight || media.naturalHeight || 0;
          if (width <= 0 || height <= 0) {{
            return false;
          }}
          overlay.width = width;
          overlay.height = height;
          return true;
        }};
        const renderOverlayToContext = (ctx, width, height, payload, now) => {{
          if (!payload) {{
            return;
          }}
          const activeEvents = filteredOverlayEvents(payload, now);
          const radius = Math.max(3, Math.min(7, Math.round(Math.min(width, height) * 0.007)));
          ctx.textBaseline = "top";
          ctx.font = "20px IBM Plex Sans, system-ui, sans-serif";
          const legendEntries = new Map();
          const legendSourceEvents = Array.isArray(payload?.events)
            ? payload.events.filter((event) => {{
                const confidence = Number(event && event.confidence != null ? event.confidence : 1);
                return confidence >= currentConfidenceThreshold;
              }})
            : [];
          for (const event of legendSourceEvents) {{
            const color = String(event && event.color || "#ffffff");
            const label = String(event && event.label || "pattern");
            if (!legendEntries.has(label)) {{
              legendEntries.set(label, color);
            }}
          }}
          for (const event of activeEvents) {{
            const color = String(event.color || "#ffffff");
            const points = Array.isArray(event.points) ? event.points : [];
            for (const point of points) {{
              const x = Number(Array.isArray(point) ? point[0] : null) * width;
              const y = Number(Array.isArray(point) ? point[1] : null) * height;
              if (!Number.isFinite(x) || !Number.isFinite(y)) {{
                continue;
              }}
              ctx.beginPath();
              ctx.fillStyle = color + "58";
              ctx.strokeStyle = color;
              ctx.lineWidth = Math.max(1, Math.floor(radius / 4));
              ctx.arc(x, y, radius, 0, Math.PI * 2);
              ctx.fill();
              ctx.stroke();
            }}
          }}
          const legendRows = Array.from(legendEntries.entries());
          if (!legendRows.length) {{
            return;
          }}
          const swatchSize = 14;
          const rowHeight = 26;
          const legendPadding = 14;
          let legendWidth = 0;
          for (const [label] of legendRows) {{
            legendWidth = Math.max(legendWidth, Math.ceil(ctx.measureText(label).width));
          }}
          const cardWidth = legendWidth + swatchSize + 30;
          const cardHeight = (legendRows.length * rowHeight) + 16;
          const cardLeft = 12;
          const cardTop = Math.max(12, height - cardHeight - 12);
          ctx.fillStyle = "rgba(17, 17, 17, 0.72)";
          ctx.beginPath();
          ctx.roundRect(cardLeft, cardTop, cardWidth + legendPadding, cardHeight, 6);
          ctx.fill();
          legendRows.forEach(([label, color], index) => {{
            const rowTop = cardTop + 6 + (index * rowHeight);
            ctx.fillStyle = color;
            ctx.fillRect(cardLeft + 8, rowTop + 6, swatchSize, swatchSize);
            ctx.fillStyle = "#ffffff";
            ctx.fillText(label, cardLeft + 8 + swatchSize + 10, rowTop + 1);
          }});
        }};
        const drawOverlay = () => {{
          overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
          if (!overlayPayload || player.hidden) {{
            overlay.hidden = true;
            return;
          }}
          if (!syncOverlayCanvas()) {{
            overlay.hidden = true;
            return;
          }}
          overlay.hidden = false;
          renderOverlayToContext(overlayCtx, overlay.width, overlay.height, overlayPayload, Math.max(0, Number(player.currentTime || 0)));
        }};
        const stopOverlayLoop = () => {{
          if (overlayAnimationFrame) {{
            cancelAnimationFrame(overlayAnimationFrame);
            overlayAnimationFrame = 0;
          }}
        }};
        const tickOverlayLoop = () => {{
          drawOverlay();
          if (!player.hidden && !player.paused && modal.classList.contains("is-open")) {{
            overlayAnimationFrame = requestAnimationFrame(tickOverlayLoop);
          }} else {{
            overlayAnimationFrame = 0;
          }}
        }};
        const startOverlayLoop = () => {{
          if (overlayAnimationFrame || player.hidden || player.paused || !modal.classList.contains("is-open")) {{
            return;
          }}
          overlayAnimationFrame = requestAnimationFrame(tickOverlayLoop);
        }};
        const downloadCompositeVideo = () => {{
          if (!currentVideoSrc) {{
            return;
          }}
          if (!overlayPayload || !currentOverlayUrl || player.hidden) {{
            const extension = player.hidden ? "png" : "mp4";
            triggerBrowserDownload(currentVideoSrc, `${{currentVideoTitle || "video"}}.${{extension}}`);
            return;
          }}
          const sourceUrl = new URL(currentOverlayUrl, window.location.origin);
          sourceUrl.pathname = "/api/ui/videos/download-overlay";
          sourceUrl.searchParams.set("confidence_threshold", currentConfidenceThreshold.toFixed(2));
          triggerBrowserDownload(sourceUrl.toString(), `${{currentVideoTitle || "video"}}-overlay.mp4`);
        }};
        const closeModal = () => {{
          stopOverlayLoop();
          modal.classList.remove("is-open");
          modal.setAttribute("aria-hidden", "true");
          player.pause();
          player.removeAttribute("src");
          player.load();
          player.loop = true;
          player.hidden = false;
          image.hidden = true;
          image.removeAttribute("src");
          overlay.hidden = true;
          overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
          overlayPayload = null;
          currentOverlayUrl = "";
          currentVideoSrc = "";
          currentVideoTitle = "video";
          setConfidenceThreshold(0);
          download.textContent = "↓";
          download.style.pointerEvents = "";
        }};
        document.addEventListener("click", (event) => {{
          const rawTarget = event.target;
          const target = rawTarget instanceof Element ? rawTarget : rawTarget instanceof Node ? rawTarget.parentElement : null;
          if (!(target instanceof Element)) {{
            return;
          }}
          const downloadButton = target.closest("#video-modal-download");
          if (downloadButton === download) {{
            if (overlayPayload && !player.hidden) {{
              event.preventDefault();
              downloadCompositeVideo();
            }} else {{
              event.preventDefault();
              triggerBrowserDownload(
                currentVideoSrc,
                `${{currentVideoTitle || "video"}}.${{player.hidden ? "png" : "mp4"}}`,
              );
            }}
            return;
          }}
          const button = target.closest(".js-video-open");
          if (button instanceof HTMLElement) {{
            const src = String(button.dataset.videoSrc || button.dataset.imageSrc || button.dataset.mediaSrc || "").trim();
            if (!src) {{
              return;
            }}
            overlayPayload = null;
            currentOverlayUrl = "";
            currentVideoSrc = src;
            currentVideoTitle = String(button.dataset.videoTitle || "video").trim() || "video";
            const mediaKind = String(button.dataset.mediaKind || (button.dataset.imageSrc ? "image" : "video")).trim().toLowerCase();
            if (mediaKind === "image") {{
              player.pause();
              player.removeAttribute("src");
              player.load();
              player.loop = true;
              player.hidden = true;
              image.hidden = false;
              image.setAttribute("src", src);
              overlay.hidden = true;
            }} else {{
              image.hidden = true;
              image.removeAttribute("src");
              player.hidden = false;
              player.loop = true;
              player.setAttribute("src", src);
              player.play().then(() => {{
                startOverlayLoop();
              }}).catch(() => {{}});
              const overlayUrl = String(button.dataset.overlayUrl || "").trim();
              currentOverlayUrl = overlayUrl;
              if (overlayUrl) {{
                fetch(overlayUrl, {{ credentials: "same-origin" }})
                  .then((response) => response.ok ? response.json() : null)
                  .then((payload) => {{
                    overlayPayload = payload && typeof payload === "object" ? payload : null;
                    requestAnimationFrame(drawOverlay);
                    startOverlayLoop();
                  }})
                  .catch(() => {{
                    overlayPayload = null;
                    drawOverlay();
                  }});
              }} else {{
                requestAnimationFrame(drawOverlay);
              }}
            }}
            modal.classList.add("is-open");
            modal.setAttribute("aria-hidden", "false");
            event.preventDefault();
            return;
          }}
          if (target === modal) {{
            closeModal();
          }}
        }});
        document.addEventListener("keydown", (event) => {{
          if (event.key === "Escape" && modal.classList.contains("is-open")) {{
            closeModal();
          }}
        }});
        player.addEventListener("loadedmetadata", drawOverlay);
        player.addEventListener("timeupdate", drawOverlay);
        player.addEventListener("seeked", drawOverlay);
        player.addEventListener("pause", () => {{
          stopOverlayLoop();
          drawOverlay();
        }});
        player.addEventListener("play", () => {{
          drawOverlay();
          startOverlayLoop();
        }});
        player.addEventListener("ended", () => {{
          drawOverlay();
          startOverlayLoop();
        }});
        confidenceSlider.addEventListener("input", () => {{
          setConfidenceThreshold(confidenceSlider.value);
          drawOverlay();
        }});
        window.addEventListener("resize", drawOverlay);
      }})();
    </script>
    """
    return _shell("Videos | Pattern Server", active_nav="videos", body_html=body)


def render_library_page(
    *,
    project_root: Path,
    ensemble_calibration_state: dict[str, Any] | None = None,
    jobs: list[JobRecord] | None = None,
) -> str:
    sections: list[str] = []
    text_payload_nodes: list[str] = []
    detail_payload_nodes: list[str] = []
    ensemble_calibration_html, ensemble_calibration_active = _render_ensemble_calibration_sections(
        project_root=project_root,
        ensemble_calibration_state=ensemble_calibration_state,
        jobs=list(jobs or []),
    )
    controls_html = """
    <section class="panel panel-static">
      <div class="card-topline">
        <div>
          <h2>Canonical Libraries</h2>
          <p class="metric-subtle">Cached by default. Recalculate only when you want to rebuild canonical selections from source runs.</p>
        </div>
      </div>
      <form method="post" action="/ui/library/recalculate" class="client-settings-form" style="margin:0;">
        <input type="hidden" name="next" value="/library">
        <button class="secondary-button" type="submit" style="font-size:0.96rem;padding:0.62rem 0.95rem;">Recalculate libraries</button>
      </form>
    </section>
    """
    for kind in ("supervised", "unsupervised", "automatic"):
        grid_html, section_text_nodes, section_detail_nodes = _render_canonical_library_collection(
            project_root=project_root,
            kind=kind,
            next_href="/library",
            include_selection_controls=False,
        )
        text_payload_nodes.extend(section_text_nodes)
        detail_payload_nodes.extend(section_detail_nodes)
        sections.append(
            f"""
            <section class="panel">
              <div class="card-topline">
                <div>
                  <h2>{html.escape(_canonical_library_kind_title(kind))}</h2>
                  <p class="metric-subtle">Canonical {html.escape(kind)} patterns for each environment.</p>
                </div>
              </div>
            </section>
            <section class="videos-grid">
              {grid_html}
            </section>
            """
        )
    canonical_browser_html = f"""
    <details class="panel" style="margin-top:1rem;">
      <summary style="cursor:pointer;list-style:none;display:flex;align-items:center;justify-content:space-between;gap:0.8rem;">
        <div>
          <h2 style="margin:0;">Browse Canonical Libraries</h2>
          <p class="metric-subtle" style="margin:0.3rem 0 0;">Expand only when you need to inspect label ensembles and code selections.</p>
        </div>
        <span class="mini-chip">3 sections</span>
      </summary>
      <div style="margin-top:1rem;display:grid;gap:1rem;">
        {''.join(sections)}
      </div>
    </details>
    """
    body = f"""
    {controls_html}
    {ensemble_calibration_html}
    {canonical_browser_html}
    {_render_text_modal_shell(text_payload_nodes)}
    {_render_library_detail_modal_shell(detail_payload_nodes)}
    {"<script>setTimeout(() => window.location.reload(), 5000);</script>" if ensemble_calibration_active else ""}
    """
    return _shell("Library | Pattern Server", active_nav="library", body_html=body)


def _summary_view_dialog_id(env_name: str, task_name: str) -> str:
    raw = f"{env_name}-{task_name}-view".lower()
    return "summary-view-" + re.sub(r"[^a-z0-9]+", "-", raw).strip("-")


def _summary_json_script_payload(value: Any) -> str:
    return json.dumps(value).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _summary_variant_blocks(variants_payload: dict[str, Any]) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for key in ("image_only", "supervised", "unsupervised", "automatic"):
        item = variants_payload.get(key)
        if not isinstance(item, dict):
            continue
        summary_text = str(item.get("summary_text") or "").strip()
        if not summary_text:
            continue
        label = key.replace("_", " ").title()
        blocks.append((label, summary_text))
    return blocks


def _summary_variant_pattern_count(variants_payload: dict[str, Any], kind: str) -> int:
    item = variants_payload.get(kind)
    if not isinstance(item, dict):
        return 0
    value = item.get("annotation_pattern_count")
    if isinstance(value, (int, float)):
        return max(0, int(value))
    return 0


def _summary_variant_best_correct_count(variants_payload: dict[str, Any]) -> int:
    best = 0
    for kind in ("supervised", "unsupervised"):
        item = variants_payload.get(kind)
        if not isinstance(item, dict):
            continue
        value = item.get("bo_correct_count")
        if isinstance(value, (int, float)):
            best = max(best, int(value))
    return best


def _summary_entry_has_shared_coverage(entry: TaskSummaryEntry, minimum: int = 10) -> bool:
    variants = entry.summary_payload.get("variants") if isinstance(entry.summary_payload, dict) else {}
    if not isinstance(variants, dict):
        return False
    return (
        _summary_variant_pattern_count(variants, "supervised") >= minimum
        and _summary_variant_pattern_count(variants, "unsupervised") >= minimum
    )


def _summary_lead_text(variants_payload: dict[str, Any]) -> str:
    for key in ("supervised", "unsupervised", "image_only", "automatic"):
        item = variants_payload.get(key)
        if not isinstance(item, dict):
            continue
        text = str(item.get("summary_text") or "").strip()
        if text:
            return text
    return ""


def _format_summary_annotation_preview(payload: dict[str, Any]) -> str:
    annotations = payload.get("pattern_annotations")
    if not isinstance(annotations, dict):
        return "No pattern annotations available."
    patterns = annotations.get("patterns")
    if not isinstance(patterns, list) or not patterns:
        return "No patterns detected."
    lines: list[str] = []
    for item in patterns[:16]:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "pattern").strip()
        start = item.get("start_timestep")
        end = item.get("end_timestep")
        parameters = item.get("parameters")
        details = ""
        if isinstance(parameters, dict):
            object_indices = parameters.get("object_indices")
            if isinstance(object_indices, list) and object_indices:
                details = f" objects={object_indices[:4]}"
        lines.append(f"- {label} [{start}-{end}]{details}".strip())
    return "\n".join(lines).strip() or "No patterns detected."


def _task_summary_progress_values(status: dict[str, Any]) -> tuple[int, int, float]:
    target_count = int(status.get("target_count") or 12)
    completed_count = int(status.get("summarized_tasks") or 0)
    running_completed = int(status.get("completed_tasks") or 0)
    progress_complete = min(target_count, max(completed_count, running_completed))
    progress_percent = (float(progress_complete) / float(target_count) * 100.0) if target_count > 0 else 0.0
    return progress_complete, target_count, progress_percent


def _read_summary_log_lines(*, project_root: Path, status: dict[str, Any], limit: int = 80) -> list[str]:
    relpath = str(status.get("log_relpath") or "").strip()
    if not relpath:
        return []
    log_path = project_root / relpath
    if not log_path.is_file():
        return []
    try:
        lines = [line.rstrip("\n") for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()]
    except OSError:
        return []
    return [line for line in lines if line.strip()][-limit:]


def _build_summary_view_payloads(
    *,
    project_root: Path,
    env_id: str,
    entries: list[TaskSummaryEntry],
) -> tuple[list[str], list[str]]:
    rows: list[str] = []
    payload_nodes: list[str] = []
    def _annotation_text_for_kind(variants: dict[str, Any], kind: str) -> str:
        variant = variants.get(kind)
        if not isinstance(variant, dict):
            return ""
        annotation_raw = str(variant.get("annotation_path") or "").strip()
        if annotation_raw:
            annotation_path = Path(annotation_raw)
            if annotation_path.is_file():
                try:
                    annotation_payload = json.loads(annotation_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    annotation_payload = {}
                if isinstance(annotation_payload, dict):
                    return _format_summary_annotation_preview(annotation_payload)
        preview = str(variant.get("annotation_preview") or "").strip()
        return preview
    for entry in entries:
        variants = entry.summary_payload.get("variants") if isinstance(entry.summary_payload, dict) else {}
        view_dialog_id = _summary_view_dialog_id(env_id, entry.task_id)
        summary_variants = _summary_variant_blocks(variants) if isinstance(variants, dict) else []
        supervised_annotation_text = _annotation_text_for_kind(variants, "supervised") if isinstance(variants, dict) else ""
        unsupervised_annotation_text = _annotation_text_for_kind(variants, "unsupervised") if isinstance(variants, dict) else ""
        supervised_pattern_count = _summary_variant_pattern_count(variants, "supervised") if isinstance(variants, dict) else 0
        unsupervised_pattern_count = _summary_variant_pattern_count(variants, "unsupervised") if isinstance(variants, dict) else 0
        best_correct_count = _summary_variant_best_correct_count(variants) if isinstance(variants, dict) else 0
        is_featured = _summary_entry_has_shared_coverage(entry)
        lead_text = _summary_lead_text(variants) if isinstance(variants, dict) else ""
        media_src = ""
        media_kind = ""
        media_label = "Preview"
        media_rotate_deg = 0
        media_playback_rate = 1.0
        image_src = (
            f"/ui/files?path={quote(_project_relpath(entry.initial_image_path))}"
            if isinstance(entry.initial_image_path, Path) and entry.initial_image_path.is_file()
            else ""
        )
        if env_id == "pooltool" and isinstance(entry.overhead_video_path, Path) and entry.overhead_video_path.is_file():
            media_src = f"/ui/files?path={quote(_project_relpath(entry.overhead_video_path))}"
            media_kind = "video"
            media_label = "Overhead"
            media_playback_rate = 0.5
        elif isinstance(entry.video_path, Path) and entry.video_path.is_file():
            media_src = f"/ui/files?path={quote(_project_relpath(entry.video_path))}"
            media_kind = "video"
            media_label = "Video"
        elif isinstance(entry.initial_image_path, Path) and entry.initial_image_path.is_file():
            media_src = image_src
            media_kind = "image"
            media_label = "Preview"
            if env_id == "pooltool":
                media_label = "Overhead"
        media_thumb_html = (
            f'<img class="summary-example-thumb" src="{html.escape(image_src)}" alt="">'
            if image_src
            else '<div class="summary-example-thumb summary-example-thumb-empty">No preview</div>'
        )
        chip_bits = [
            '<span class="mini-chip">Featured pair</span>' if is_featured else '<span class="mini-chip">Fallback</span>',
            f'<span class="mini-chip">S {supervised_pattern_count}</span>' if supervised_pattern_count > 0 else "",
            f'<span class="mini-chip">U {unsupervised_pattern_count}</span>' if unsupervised_pattern_count > 0 else "",
            f'<span class="mini-chip">Chosen {best_correct_count}</span>' if best_correct_count > 0 else "",
        ]
        snippet = lead_text if len(lead_text) <= 180 else (lead_text[:177].rstrip() + "...")
        payload_nodes.append(
            f'<script type="application/json" id="{view_dialog_id}">{_summary_json_script_payload({"task_id": entry.task_id, "env_id": env_id, "source_run_id": entry.source_run_id, "media_src": media_src, "media_kind": media_kind, "media_label": media_label, "media_rotate_deg": media_rotate_deg, "media_playback_rate": media_playback_rate, "image_src": image_src, "summaries": [{"label": label, "text": text} for label, text in summary_variants], "supervised_annotation_text": supervised_annotation_text, "unsupervised_annotation_text": unsupervised_annotation_text})}</script>'
        )
        rows.append(
            f"""
            <article class="summary-example-card">
              {media_thumb_html}
              <div class="summary-example-main">
                <div class="summary-example-top">
                  <div class="summary-example-title-stack">
                    <strong>{html.escape(entry.task_id)}</strong>
                    <span class="metric-subtle">{html.escape(entry.source_run_id or "summary bundle")}</span>
                  </div>
                  <button class="ghost-link js-summary-view-open" type="button" data-view-source="{view_dialog_id}">Open</button>
                </div>
                <div class="summary-example-chips">{''.join(bit for bit in chip_bits if bit)}</div>
                <p class="metric-subtle summary-example-snippet">{html.escape(snippet or "No summary text available.")}</p>
              </div>
            </article>
            """
        )
    return rows, payload_nodes


def _render_summary_view_modal() -> str:
    return """
    <div id="summary-view-modal" class="video-modal-overlay" aria-hidden="true">
      <div id="summary-view-card" class="video-modal-card summary-view-modal-card" role="dialog" aria-modal="true" aria-label="Summary details">
        <div class="card-topline">
          <div><h2 id="summary-view-title" style="margin:0;font-size:1rem;">Summary details</h2></div>
          <button id="summary-view-close" class="secondary-button text-modal-close" type="button">Close</button>
        </div>
        <div id="summary-view-grid" class="summary-view-grid">
          <section id="summary-view-media-column" class="summary-view-column summary-view-media-column">
            <h3>Media</h3>
            <div id="summary-view-frame" class="video-modal-frame">
              <video id="summary-view-player" controls preload="metadata" src=""></video>
              <img id="summary-view-image" alt="" hidden>
            </div>
            <div id="summary-view-media-links" class="analysis-summary-links"></div>
          </section>
          <section class="summary-view-column">
            <h3>Summaries</h3>
            <div id="summary-view-summaries" class="summary-variant-list"></div>
          </section>
          <section class="summary-view-column">
            <h3>Annotation</h3>
            <div class="analysis-summary-links" style="margin-bottom:0.55rem;">
              <button id="summary-view-supervised-button" class="secondary-button" type="button">Supervised annotation</button>
              <button id="summary-view-unsupervised-button" class="secondary-button" type="button">Unsupervised annotation</button>
            </div>
            <pre id="summary-view-annotation" class="text-modal-body"></pre>
            <div id="summary-view-annotation-link" class="analysis-summary-links"></div>
          </section>
        </div>
      </div>
    </div>
    <script>
      (() => {
        const modal = document.getElementById("summary-view-modal");
        const title = document.getElementById("summary-view-title");
        const card = document.getElementById("summary-view-card");
        const grid = document.getElementById("summary-view-grid");
        const mediaColumn = document.getElementById("summary-view-media-column");
        const frame = document.getElementById("summary-view-frame");
        const player = document.getElementById("summary-view-player");
        const image = document.getElementById("summary-view-image");
        const summaries = document.getElementById("summary-view-summaries");
        const annotation = document.getElementById("summary-view-annotation");
        const annotationLink = document.getElementById("summary-view-annotation-link");
        const supervisedButton = document.getElementById("summary-view-supervised-button");
        const unsupervisedButton = document.getElementById("summary-view-unsupervised-button");
        const mediaLinks = document.getElementById("summary-view-media-links");
        const closeButton = document.getElementById("summary-view-close");
        if (!modal || !title || !card || !grid || !mediaColumn || !frame || !player || !image || !summaries || !annotation || !annotationLink || !supervisedButton || !unsupervisedButton || !mediaLinks || !closeButton) {
          return;
        }
        let currentAnnotations = { supervised: "", unsupervised: "" };
        let currentImageSrc = "";
        let currentPlaybackRate = 1;
        let currentMediaRotation = 0;
        let currentEnvId = "";
        const applySummaryLayout = (envId) => {
          currentEnvId = String(envId || "").trim().toLowerCase();
          const isPooltool = currentEnvId === "pooltool";
          card.classList.toggle("is-pooltool", isPooltool);
          grid.classList.toggle("is-pooltool", isPooltool);
          mediaColumn.classList.toggle("is-pooltool", isPooltool);
          frame.style.height = isPooltool ? "72vh" : "";
          frame.style.maxHeight = isPooltool ? "72vh" : "";
        };
        const resetMediaSizing = () => {
          for (const element of [player, image]) {
            element.style.width = "";
            element.style.maxWidth = "";
            element.style.height = "";
            element.style.maxHeight = "";
            element.style.transform = "";
          }
          frame.classList.remove("is-rotated");
        };
        const fitRotatedMedia = () => {
          const rotation = Math.abs(currentMediaRotation) % 180 === 90 ? currentMediaRotation : 0;
          const active = player.hidden ? image : player;
          const inactive = player.hidden ? player : image;
          inactive.style.width = "";
          inactive.style.maxWidth = "";
          inactive.style.height = "";
          inactive.style.maxHeight = "";
          inactive.style.transform = "";
          frame.classList.toggle("is-rotated", !!rotation);
          if (!rotation) {
            active.style.width = "";
            active.style.maxWidth = "";
            active.style.height = "";
            active.style.maxHeight = "";
            active.style.transform = "";
            return;
          }
          const intrinsicWidth = player.hidden ? Number(image.naturalWidth || 0) : Number(player.videoWidth || 0);
          const intrinsicHeight = player.hidden ? Number(image.naturalHeight || 0) : Number(player.videoHeight || 0);
          const frameWidth = Number(frame.clientWidth || 0);
          const frameHeight = Number(frame.clientHeight || 0);
          if (intrinsicWidth <= 0 || intrinsicHeight <= 0 || frameWidth <= 0 || frameHeight <= 0) {
            active.style.transform = `rotate(${rotation}deg)`;
            return;
          }
          const aspect = intrinsicWidth / intrinsicHeight;
          const fittedWidth = Math.max(1, Math.floor(Math.min(frameHeight, frameWidth * aspect) * 0.9));
          active.style.width = `${fittedWidth}px`;
          active.style.maxWidth = "none";
          active.style.height = "auto";
          active.style.maxHeight = "none";
          active.style.transform = `rotate(${rotation}deg)`;
        };
        const applyMediaRotation = (degrees) => {
          currentMediaRotation = Number.isFinite(degrees) ? degrees : 0;
          fitRotatedMedia();
        };
        const setAnnotationButtonState = (button, enabled) => {
          button.disabled = !enabled;
          button.style.opacity = enabled ? "1" : "0.45";
          button.style.cursor = enabled ? "pointer" : "default";
        };
        const showImage = (src) => {
          player.pause();
          player.playbackRate = 1;
          player.removeAttribute("src");
          player.load();
          player.hidden = true;
          image.hidden = false;
          currentMediaRotation = 0;
          applySummaryLayout("");
          resetMediaSizing();
          if (src) {
            image.setAttribute("src", src);
          } else {
            image.removeAttribute("src");
          }
        };
        const showVideo = (src) => {
          image.hidden = true;
          image.removeAttribute("src");
          player.hidden = false;
          currentMediaRotation = 0;
          resetMediaSizing();
          player.setAttribute("src", src);
          player.playbackRate = currentPlaybackRate > 0 ? currentPlaybackRate : 1;
          player.load();
          player.play().catch(() => {});
        };
        const showAnnotation = (kind) => {
          const text = currentAnnotations && typeof currentAnnotations[kind] === "string" ? currentAnnotations[kind] : "";
          annotation.textContent = text || "No annotation available.";
        };
        const closeModal = () => {
          modal.classList.remove("is-open");
          modal.setAttribute("aria-hidden", "true");
          player.pause();
          player.playbackRate = 1;
          player.removeAttribute("src");
          player.load();
          player.hidden = false;
          image.hidden = true;
          image.removeAttribute("src");
          currentMediaRotation = 0;
          resetMediaSizing();
          summaries.innerHTML = "";
          currentAnnotations = { supervised: "", unsupervised: "" };
          currentImageSrc = "";
          annotation.textContent = "";
          annotationLink.innerHTML = "";
          setAnnotationButtonState(supervisedButton, false);
          setAnnotationButtonState(unsupervisedButton, false);
          title.textContent = "Summary details";
        };
        closeButton.addEventListener("click", closeModal);
        supervisedButton.addEventListener("click", () => showAnnotation("supervised"));
        unsupervisedButton.addEventListener("click", () => showAnnotation("unsupervised"));
        player.addEventListener("loadedmetadata", fitRotatedMedia);
        image.addEventListener("load", fitRotatedMedia);
        window.addEventListener("resize", () => {
          if (modal.classList.contains("is-open")) {
            fitRotatedMedia();
          }
        });
        player.addEventListener("error", () => {
          if (currentImageSrc) {
            showImage(currentImageSrc);
          }
        });
        document.addEventListener("click", (event) => {
          const target = event.target;
          if (!(target instanceof Element)) {
            return;
          }
          const button = target.closest(".js-summary-view-open");
          if (button instanceof HTMLElement) {
            const sourceId = String(button.dataset.viewSource || "").trim();
            const payloadNode = sourceId ? document.getElementById(sourceId) : null;
            if (!payloadNode) {
              return;
            }
            let payload = null;
            try {
              payload = JSON.parse(payloadNode.textContent || "{}");
            } catch (_error) {
              payload = null;
            }
            if (!payload || typeof payload !== "object") {
              return;
            }
            const src = String(payload.media_src || "").trim();
            const mediaKind = String(payload.media_kind || "").trim().toLowerCase();
            currentImageSrc = String(payload.image_src || "").trim();
            applySummaryLayout(String(payload.env_id || ""));
            const mediaRotateDeg = Number(payload.media_rotate_deg || 0);
            currentPlaybackRate = Number(payload.media_playback_rate || 1);
            const taskId = String(payload.task_id || "Task").trim();
            const sourceRunId = String(payload.source_run_id || "").trim();
            title.textContent = sourceRunId ? `${taskId} · ${sourceRunId}` : taskId;
            if (mediaKind === "image") {
              showImage(src || currentImageSrc);
              applyMediaRotation(mediaRotateDeg);
            } else if (src) {
              showVideo(src);
              applyMediaRotation(mediaRotateDeg);
            } else {
              showImage(currentImageSrc);
              applyMediaRotation(mediaRotateDeg);
            }
            mediaLinks.innerHTML = "";
            const variantRows = Array.isArray(payload.summaries) ? payload.summaries : [];
            if (variantRows.length === 0) {
              summaries.innerHTML = '<p class="empty">No summary text available.</p>';
            } else {
              summaries.innerHTML = variantRows.map((item) => {
                const label = String(item && item.label ? item.label : "Summary");
                const text = String(item && item.text ? item.text : "").trim();
                const safeLabel = label.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
                const safeText = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
                return `<article class="summary-variant-row"><h4>${safeLabel}</h4><pre>${safeText || "No text available."}</pre></article>`;
              }).join("");
            }
            currentAnnotations = {
              supervised: String(payload.supervised_annotation_text || "").trim(),
              unsupervised: String(payload.unsupervised_annotation_text || "").trim(),
            };
            const hasSupervised = !!currentAnnotations.supervised;
            const hasUnsupervised = !!currentAnnotations.unsupervised;
            setAnnotationButtonState(supervisedButton, hasSupervised);
            setAnnotationButtonState(unsupervisedButton, hasUnsupervised);
            if (hasSupervised) {
              showAnnotation("supervised");
            } else if (hasUnsupervised) {
              showAnnotation("unsupervised");
            } else {
              annotation.textContent = "No annotation available.";
            }
            annotationLink.innerHTML = "";
            modal.classList.add("is-open");
            modal.setAttribute("aria-hidden", "false");
            event.preventDefault();
            return;
          }
          if (target === modal) {
            closeModal();
          }
        });
        document.addEventListener("keydown", (event) => {
          if (event.key === "Escape" && modal.classList.contains("is-open")) {
            closeModal();
          }
        });
      })();
    </script>
    """


def _render_summaries_home_live_panel(
    *,
    project_root: Path,
    statuses: dict[str, dict[str, Any]],
) -> str:
    running_envs = [
        env_id for env_id in SUPPORTED_ENVS
        if str(statuses.get(env_id, {}).get("status") or "").strip().lower() == "running"
    ]
    focus_env = running_envs[0] if running_envs else None
    if focus_env is None:
        candidates = []
        for env_id in SUPPORTED_ENVS:
            status = dict(statuses.get(env_id, {}))
            updated_at = str(status.get("updated_at") or "").strip()
            if updated_at:
                candidates.append((updated_at, env_id))
        focus_env = max(candidates)[1] if candidates else None
    if focus_env is None:
        return """
        <section class="panel compact-panel panel-static">
          <div class="card-topline">
            <div>
              <p class="section-kicker">Live progress</p>
              <h2>Current Summaries</h2>
              <p class="metric-subtle">No summary generation is active.</p>
            </div>
          </div>
        </section>
        """
    focus_status = dict(statuses.get(focus_env, {}))
    is_live = str(focus_status.get("status") or "").strip().lower() == "running"
    visible_envs = running_envs[:4] if is_live else [focus_env]
    rows: list[str] = []
    for env_id in visible_envs:
        status = dict(statuses.get(env_id, {}))
        progress_complete, target_count, _progress_percent = _task_summary_progress_values(status)
        current_task_id = str(status.get("current_task_id") or "").strip()
        current_source_run_id = str(status.get("current_source_run_id") or "").strip()
        summary_meta_parts = [f"{progress_complete}/{target_count} summaries"]
        if current_task_id:
            summary_meta_parts.append(current_task_id)
        if current_source_run_id:
            summary_meta_parts.append(current_source_run_id)
        rows.append(
            f"""
            <article class="required-row{' is-active-state' if str(status.get("status") or "").strip().lower() == "running" else ''}">
              <div class="required-row-main">
                <div class="required-row-title">
                  <span class="required-status-dot {'is-covered' if str(status.get("status") or "").strip().lower() == "succeeded" else ('is-active' if str(status.get('status') or '').strip().lower() == 'running' else 'is-missing')}"></span>
                  <strong>{html.escape(env_id.upper())}</strong>
                </div>
                <div class="required-row-meta">{html.escape(" · ".join(summary_meta_parts))}</div>
              </div>
              <div class="required-status">{_pill(str(status.get("status") or "idle"), str(status.get("status") or "idle"))}</div>
            </article>
            """
        )
    log_lines = _read_summary_log_lines(project_root=project_root, status=focus_status, limit=60)
    log_text = "\n".join(reversed(log_lines)) if log_lines else "No log yet."
    panel_title = "Current Summaries" if is_live else "Latest Summaries"
    running_chip = f"Running {len(running_envs)}" if is_live else "No active runs"
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Live progress</p>
          <h2>{panel_title}</h2>
          <p class="metric-subtle">{html.escape(focus_env.upper())}</p>
        </div>
        <div class="stats-inline">
          <span class="mini-chip">{html.escape(running_chip)}</span>
          {_pill(str(focus_status.get("status") or "idle"), str(focus_status.get("status") or "idle"))}
        </div>
      </div>
      <div class="required-list" style="margin-bottom:0.8rem;">{''.join(rows)}</div>
      <pre class="code-block" style="max-height:12rem;overflow:auto;">{html.escape(log_text)}</pre>
      {('<script>(() => { const refresh = () => window.setTimeout(() => window.location.reload(), 2000); window.setTimeout(refresh, 1800); })();</script>' if is_live else '')}
    </section>
    """


def _render_summaries_generator_panel(
    *,
    project_root: Path,
    statuses: dict[str, dict[str, Any]],
    entries_by_env: dict[str, list[TaskSummaryEntry]],
) -> str:
    active_envs = [
        env_id for env_id in SUMMARIES_DISPLAY_ENVS
        if str(statuses.get(env_id, {}).get("status") or "").strip().lower() == "running"
    ]
    focus_env = active_envs[0] if active_envs else None
    if focus_env is None:
        dated = []
        for env_id in SUMMARIES_DISPLAY_ENVS:
            updated_at = str(statuses.get(env_id, {}).get("updated_at") or "").strip()
            if updated_at:
                dated.append((updated_at, env_id))
        focus_env = max(dated)[1] if dated else SUMMARIES_DISPLAY_ENVS[0]
    focus_status = dict(statuses.get(focus_env, {}))
    log_lines = _read_summary_log_lines(project_root=project_root, status=focus_status, limit=100)
    log_text = "\n".join(reversed(log_lines)) if log_lines else "No log yet."
    env_cards: list[str] = []
    for env_id in SUMMARIES_DISPLAY_ENVS:
        status = dict(statuses.get(env_id, {}))
        listed_count = len(list(entries_by_env.get(env_id, [])))
        target_count = max(1, listed_count)
        status_label = str(status.get("status") or "idle").strip() or "idle"
        progress_complete, _status_target_count, _ = _task_summary_progress_values(status)
        run_target = max(1, int(status.get("target_count") or target_count))
        progress_percent = (float(progress_complete) / float(run_target) * 100.0) if run_target > 0 else 0.0
        mode_bits = [str(item).strip() for item in (status.get("enabled_variants_override") or []) if str(item).strip()]
        mode_label = "Image-only" if mode_bits == ["image_only"] else (" · ".join(mode_bits) if mode_bits else "Configured variants")
        env_cards.append(
            f"""
            <article class="panel compact-panel panel-static analysis-dashboard-card">
              <div class="card-topline">
                <div>
                  <h2>{html.escape(env_id.upper())}</h2>
                  <p class="metric-subtle">{listed_count} listed examples below</p>
                </div>
                <div class="stats-inline">
                  {_pill(status_label, status_label)}
                  <span class="mini-chip">{html.escape(mode_label)}</span>
                </div>
              </div>
              <form method="post" action="/ui/summaries/run" class="client-settings-form" style="margin-top:0.6rem;">
                <input type="hidden" name="next" value="/summaries">
                <input type="hidden" name="env_id" value="{html.escape(env_id)}">
                <input type="hidden" name="target_count" value="{target_count}">
                <input type="hidden" name="enabled_variants" value="image_only">
                <button class="secondary-button" type="submit">Generate image-only for listed</button>
              </form>
              <div class="analysis-progress" style="margin-top:0.7rem;">
                <div class="analysis-progress-meta"><span>Progress</span><span>{progress_complete}/{run_target}</span></div>
                <div class="analysis-progress-track"><span class="analysis-progress-fill{' is-indeterminate' if status_label.lower() == 'running' and progress_complete <= 0 else ''}" style="width: {progress_percent:.2f}%"></span></div>
                <p class="analysis-progress-note">Listed {listed_count} · Planned {int(status.get('planned_tasks') or 0)} · Completed {int(status.get('completed_tasks') or 0)} · Failed {int(status.get('failed_tasks') or 0)}</p>
              </div>
            </article>
            """
        )
    is_live = str(focus_status.get("status") or "").strip().lower() == "running"
    return f"""
    <section class="panel panel-static">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Image-only generation</p>
          <h2>Generate Summaries For Listed Examples</h2>
          <p class="metric-subtle">Launch image-only summaries for the curated examples shown below. The runner stays local and writes progress into the same task-summary state/logs.</p>
        </div>
        <div class="stats-inline">
          <span class="mini-chip">Running {len(active_envs)}</span>
          <span class="mini-chip">Focus {html.escape(focus_env.upper())}</span>
        </div>
      </div>
      <section class="required-grid" style="margin-top:0.8rem;">
        {''.join(env_cards)}
      </section>
      <div class="analysis-progress" style="margin-top:0.85rem;">
        <div class="analysis-progress-meta"><span>Live terminal focus</span><span>{html.escape(focus_env.upper())}</span></div>
        <div class="analysis-progress-track"><span class="analysis-progress-fill{' is-indeterminate' if is_live else ''}" style="width: {100.0 if is_live else 0.0:.2f}%"></span></div>
        <p class="analysis-progress-note">{html.escape(str(focus_status.get('run_label') or 'No active run selected.'))}</p>
      </div>
      <pre class="code-block" style="max-height:16rem;overflow:auto;margin-top:0.65rem;">{html.escape(log_text)}</pre>
      {('<script>(() => { const refresh = () => window.setTimeout(() => window.location.reload(), 2000); window.setTimeout(refresh, 1800); })();</script>' if is_live else '')}
    </section>
    """


def _render_summaries_env_dashboard_card(
    *,
    env_id: str,
    status: dict[str, Any],
    entries: list[TaskSummaryEntry],
) -> str:
    status_label = str(status.get("status") or "idle").strip() or "idle"
    progress_complete, target_count, _progress_percent = _task_summary_progress_values(status)
    summary_parts = [
        f"{len(entries)} completed tasks",
        f"{int(status.get('successful_tasks') or 0)} successful source tasks",
        f"{int(status.get('failed_tasks') or 0)} failed in current batch",
    ]
    return f"""
    <section class="panel compact-panel panel-static analysis-dashboard-card">
      <div class="card-topline">
        <div>
          <h2><a href="/summaries/env/{html.escape(env_id)}">{html.escape(env_id.upper())}</a></h2>
          <p class="metric-subtle">{html.escape(" · ".join(summary_parts))}</p>
        </div>
        <div class="stats-inline">
          {_pill(status_label, status_label)}
          <span class="mini-chip">Target {target_count}</span>
        </div>
      </div>
      <div class="required-row-meta" style="margin-top:0.42rem;">
        {html.escape(f"{progress_complete}/{target_count} summaries · Planned {int(status.get('planned_tasks') or 0)} · Completed {int(status.get('completed_tasks') or 0)} · Failed {int(status.get('failed_tasks') or 0)}")}
      </div>
      <div class="analysis-dashboard-actions">
        <form method="post" action="/ui/summaries/run" class="inline-form">
          <input type="hidden" name="next" value="/summaries">
          <input type="hidden" name="env_id" value="{html.escape(env_id)}">
          <input type="hidden" name="target_count" value="{target_count}">
          <button class="secondary-button" type="submit">Run locally</button>
        </form>
        <a class="ghost-link" href="/summaries/env/{html.escape(env_id)}">View results</a>
      </div>
    </section>
    """


def _render_summaries_env_status_panel(*, env_id: str, status: dict[str, Any]) -> str:
    progress_complete, target_count, progress_percent = _task_summary_progress_values(status)
    current_task_id = str(status.get("current_task_id") or "").strip()
    current_source_run_id = str(status.get("current_source_run_id") or "").strip()
    meta_bits = []
    if current_task_id:
        meta_bits.append(current_task_id)
    if current_source_run_id:
        meta_bits.append(current_source_run_id)
    if not meta_bits and str(status.get("error") or "").strip():
        meta_bits.append(str(status.get("error") or "").strip())
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div><p class="section-kicker">Per environment</p><h2>Current Summary Job</h2></div>
        <div class="stats-inline">{_pill(str(status.get("status") or "idle"), str(status.get("status") or "idle"))}</div>
      </div>
      <form method="post" action="/ui/summaries/run" class="client-settings-form" style="margin-top:0.55rem;">
        <input type="hidden" name="next" value="/summaries/env/{html.escape(env_id)}">
        <input type="hidden" name="env_id" value="{html.escape(env_id)}">
        <label>
          <span>Target summaries</span>
          <input type="number" min="1" name="target_count" value="{target_count}">
        </label>
        <button class="secondary-button" type="submit">Run locally</button>
      </form>
      <div class="analysis-progress" style="margin-top:0.65rem;">
        <div class="analysis-progress-meta"><span>Summary progress</span><span>{progress_complete}/{target_count}</span></div>
        <div class="analysis-progress-track"><span class="analysis-progress-fill{' is-indeterminate' if str(status.get("status") or '').strip().lower() == 'running' and progress_complete <= 0 else ''}" style="width: {progress_percent:.2f}%"></span></div>
        <p class="analysis-progress-note">Planned {int(status.get("planned_tasks") or 0)} · Completed {int(status.get("completed_tasks") or 0)} · Failed {int(status.get("failed_tasks") or 0)}</p>
      </div>
      <div class="required-row-meta">{html.escape(" · ".join(meta_bits) or "No task currently selected.")}</div>
    </section>
    """


def _render_summaries_env_terminal_panel(*, project_root: Path, env_id: str, status: dict[str, Any]) -> str:
    log_lines = _read_summary_log_lines(project_root=project_root, status=status, limit=120)
    log_text = "\n".join(reversed(log_lines)) if log_lines else "No log yet."
    is_running = str(status.get("status") or "").strip().lower() == "running"
    return f"""
    <section class="panel panel-static">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Live progress</p>
          <h2>Summary Terminal</h2>
          <p class="metric-subtle">{html.escape(env_id.upper())}</p>
        </div>
        <div class="stats-inline">{_pill(str(status.get("status") or "idle"), str(status.get("status") or "idle"))}</div>
      </div>
      <pre class="code-block" style="max-height:18rem;overflow:auto;margin-top:0.5rem;">{html.escape(log_text)}</pre>
      {('<script>(() => { const refresh = () => window.setTimeout(() => window.location.reload(), 2000); window.setTimeout(refresh, 1800); })();</script>' if is_running else '')}
    </section>
    """


def _render_summaries_entries_panel(
    *,
    env_id: str,
    entries: list[TaskSummaryEntry],
) -> tuple[str, list[str]]:
    rows, payload_nodes = _build_summary_view_payloads(project_root=Path("."), env_id=env_id, entries=entries)
    featured_count = sum(1 for entry in entries if _summary_entry_has_shared_coverage(entry))
    panel = f"""
    <section class="panel videos-cell">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Curated examples</p>
          <h2>{html.escape(env_id.upper())}</h2>
          <p class="metric-subtle">{len(entries)} examples · {featured_count} strong paired examples</p>
        </div>
      </div>
      <div class="videos-task-list">
        {''.join(rows) if rows else '<p class="empty" style="padding: 0.6rem;">No completed summaries yet.</p>'}
      </div>
    </section>
    """
    return panel, payload_nodes


def _filter_summary_entries_for_display(
    *,
    env_id: str,
    entries: list[TaskSummaryEntry],
) -> list[TaskSummaryEntry]:
    if env_id == "kinetix":
        return []
    playable_entries = [
        entry
        for entry in entries
        if isinstance(entry.video_path, Path) and entry.video_path.is_file()
    ]
    return playable_entries


def render_summaries_page(
    *,
    project_root: Path,
    statuses: dict[str, dict[str, Any]],
    entries_by_env: dict[str, list[TaskSummaryEntry]],
    config: dict[str, Any],
) -> str:
    image_only_prompt_template = str(config.get("image_only_prompt_template") or "").strip()
    annotation_prompt_template = str(config.get("annotation_prompt_template") or "").strip()
    summary_view_payload_nodes: list[str] = []
    results_grid_cards: list[str] = []
    display_entries_by_env = {
        env_id: _filter_summary_entries_for_display(
            env_id=env_id,
            entries=list(entries_by_env.get(env_id, [])),
        )
        for env_id in SUMMARIES_DISPLAY_ENVS
    }
    for env_id in SUMMARIES_DISPLAY_ENVS:
        _rows, payload_nodes = _build_summary_view_payloads(
            project_root=project_root,
            env_id=env_id,
            entries=list(display_entries_by_env.get(env_id, [])),
        )
        summary_view_payload_nodes.extend(payload_nodes)
        entries_panel, _ = _render_summaries_entries_panel(
            env_id=env_id,
            entries=list(display_entries_by_env.get(env_id, [])),
        )
        results_grid_cards.append(entries_panel)
    status_values = [str(statuses.get(env_id, {}).get("status") or "idle").strip().lower() for env_id in SUPPORTED_ENVS]
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Home</a> / summaries</p>
      <h1>Task Summaries</h1>
      <p>Curated summary examples with playable video only. Kinetix is left out of this page for now because too many BO examples do not have rendered video assets.</p>
    </section>
    <section class="hero-stats">
      <div class="stat-chip"><span class="stat-label">Environments</span><strong class="stat-value">{len(SUMMARIES_DISPLAY_ENVS)}</strong></div>
      <div class="stat-chip"><span class="stat-label">Running</span><strong class="stat-value">{sum(1 for value in status_values if value == "running")}</strong></div>
      <div class="stat-chip"><span class="stat-label">Completed summaries</span><strong class="stat-value">{sum(len(list(display_entries_by_env.get(env_id, []))) for env_id in SUMMARIES_DISPLAY_ENVS)}</strong></div>
      <div class="stat-chip"><span class="stat-label">Successful source tasks</span><strong class="stat-value">{sum(int(statuses.get(env_id, {}).get("successful_tasks") or 0) for env_id in SUMMARIES_DISPLAY_ENVS)}</strong></div>
    </section>
    <section class="panel">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Prompt settings</p>
          <h2>Summary Prompts</h2>
          <p class="metric-subtle">These templates are used for new local summary runs.</p>
        </div>
      </div>
      <details class="data-details" style="margin-top:0.65rem;">
        <summary>Show prompt editor</summary>
        <form method="post" action="/ui/summaries/config" class="client-settings-form summaries-config-form" style="margin-top:0.8rem;">
          <input type="hidden" name="next" value="/summaries">
          <label class="summaries-config-label">
            <span>Image-only prompt</span>
            <textarea name="image_only_prompt_template" rows="7" class="summaries-config-textarea">{html.escape(image_only_prompt_template)}</textarea>
          </label>
          <label class="summaries-config-label">
            <span>Annotation-aware prompt</span>
            <textarea name="annotation_prompt_template" rows="7" class="summaries-config-textarea">{html.escape(annotation_prompt_template)}</textarea>
          </label>
          <button class="secondary-button" type="submit">Save prompt</button>
        </form>
      </details>
    </section>
    {_render_summaries_generator_panel(project_root=project_root, statuses=statuses, entries_by_env=display_entries_by_env)}
    <section class="panel panel-static">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Results</p>
          <h2>Completed Summaries By Environment</h2>
          <p class="metric-subtle">Scroll each environment panel and click <strong>View</strong> to inspect video, annotation, and all summary variants for a task.</p>
        </div>
      </div>
    </section>
    <section class="videos-grid">
      {''.join(results_grid_cards)}
    </section>
    {''.join(summary_view_payload_nodes)}
    {_render_summary_view_modal()}
    """
    return _shell("Summaries | Pattern Server", active_nav="summaries", body_html=body)


def render_summaries_environment_page(
    *,
    project_root: Path,
    env_id: str,
    status: dict[str, Any],
    entries: list[TaskSummaryEntry],
) -> str:
    entries_panel, payload_nodes = _render_summaries_entries_panel(env_id=env_id, entries=entries)
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Home</a> / <a href="/summaries">summaries</a> / {html.escape(env_id)}</p>
      <h1>{html.escape(env_id.upper())} Summaries</h1>
    </section>
    <section class="hero-stats">
      <div class="stat-chip"><span class="stat-label">Status</span><strong class="stat-value">{html.escape(str(status.get("status") or "idle"))}</strong></div>
      <div class="stat-chip"><span class="stat-label">Target</span><strong class="stat-value">{int(status.get("target_count") or 0)}</strong></div>
      <div class="stat-chip"><span class="stat-label">Completed summaries</span><strong class="stat-value">{len(entries)}</strong></div>
      <div class="stat-chip"><span class="stat-label">Successful source tasks</span><strong class="stat-value">{int(status.get("successful_tasks") or 0)}</strong></div>
    </section>
    {_render_summaries_env_status_panel(env_id=env_id, status=status)}
    {_render_summaries_env_terminal_panel(project_root=project_root, env_id=env_id, status=status)}
    {entries_panel}
    {''.join(payload_nodes)}
    {_render_summary_view_modal()}
    """
    return _shell(f"{env_id.upper()} Summaries | Pattern Server", active_nav="summaries", body_html=body)


def _analysis_asset_href(path: Path) -> str:
    return f"/ui/files?path={quote(_project_relpath(path))}"


def _analysis_plot_payload_id(raw_key: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(raw_key or "").strip().lower()).strip("-")
    return f"analysis-plot-{slug or 'plot'}"


def _analysis_plot_trigger(*, payload_id: str, inner_html: str) -> str:
    return (
        f'<button class="analysis-plot-trigger js-analysis-plot-open" type="button" '
        f'data-plot-source="{html.escape(payload_id)}">{inner_html}</button>'
    )


def _analysis_plot_modal_markup() -> str:
    return """
    <div id="analysis-plot-modal" class="video-modal-overlay" aria-hidden="true">
      <div class="video-modal-card analysis-plot-modal-card" role="dialog" aria-modal="true" aria-label="Expanded analysis plot">
        <div class="card-topline">
          <div><h2 id="analysis-plot-modal-title" style="margin:0;font-size:1rem;">Analysis plot</h2></div>
          <div class="stats-inline">
            <a id="analysis-plot-modal-download" class="secondary-button" href="#" hidden>Download PDF</a>
            <button id="analysis-plot-modal-close" class="secondary-button text-modal-close" type="button">Close</button>
          </div>
        </div>
        <div id="analysis-plot-modal-body" class="analysis-plot-modal-body"></div>
      </div>
    </div>
    <script>
      (() => {
        const modal = document.getElementById("analysis-plot-modal");
        const body = document.getElementById("analysis-plot-modal-body");
        const title = document.getElementById("analysis-plot-modal-title");
        const closeButton = document.getElementById("analysis-plot-modal-close");
        const downloadLink = document.getElementById("analysis-plot-modal-download");
        if (!modal || !body || !title || !closeButton || !downloadLink) {
          return;
        }
        const closeModal = () => {
          modal.classList.remove("is-open");
          modal.setAttribute("aria-hidden", "true");
          body.innerHTML = "";
          title.textContent = "Analysis plot";
          downloadLink.hidden = true;
          downloadLink.setAttribute("href", "#");
        };
        closeButton.addEventListener("click", closeModal);
        document.addEventListener("click", (event) => {
          const target = event.target;
          if (!(target instanceof Element)) {
            return;
          }
          const trigger = target.closest(".js-analysis-plot-open");
          if (trigger instanceof HTMLElement) {
            const sourceId = String(trigger.dataset.plotSource || "").trim();
            const payloadNode = sourceId ? document.getElementById(sourceId) : null;
            if (!payloadNode) {
              return;
            }
            let payload = null;
            try {
              payload = JSON.parse(payloadNode.textContent || "{}");
            } catch (_error) {
              payload = null;
            }
            if (!payload || typeof payload !== "object") {
              return;
            }
            title.textContent = String(payload.title || "Analysis plot");
            body.innerHTML = String(payload.html || "");
            const downloadHref = String(payload.download_href || "").trim();
            if (downloadHref) {
              downloadLink.hidden = false;
              downloadLink.setAttribute("href", downloadHref);
            } else {
              downloadLink.hidden = true;
              downloadLink.setAttribute("href", "#");
            }
            modal.classList.add("is-open");
            modal.setAttribute("aria-hidden", "false");
            event.preventDefault();
            return;
          }
          if (target === modal) {
            closeModal();
          }
        });
        document.addEventListener("keydown", (event) => {
          if (event.key === "Escape" && modal.classList.contains("is-open")) {
            closeModal();
          }
        });
      })();
    </script>
    """


def _analysis_plot_card(title: str, path: Path | None) -> str:
    if path is None or not path.is_file():
        return f"""
        <section class="panel compact-panel analysis-plot-card">
          <h3>{html.escape(title)}</h3>
          <p class="empty">Missing plot.</p>
        </section>
        """
    href = _analysis_asset_href(path)
    payload_id = _analysis_plot_payload_id(f"{title}-{path}")
    payload = {
        "title": title,
        "html": (
            f'<img class="analysis-plot-frame" src="{href}" alt="{html.escape(title)}">'
        ),
    }
    return f"""
    <section class="panel compact-panel analysis-plot-card">
      <div class="card-topline">
        <div><h3>{html.escape(title)}</h3></div>
        <a class="ghost-link" href="{href}">Open</a>
      </div>
      {_analysis_plot_trigger(payload_id=payload_id, inner_html=f'<img class="analysis-plot-frame" src="{href}" alt="{html.escape(title)}">')}
      <script type="application/json" id="{payload_id}">{_summary_json_script_payload(payload)}</script>
    </section>
    """


def _render_analysis_launcher_panel(*, project_root: Path, clients: list[ClientRecord]) -> str:
    direct_clients = [client for client in _dashboard_active_clients(clients) if client.kind.value == "direct"]
    local_direct = next(
        (
            client
            for client in direct_clients
            if "local" in set(client.capabilities.tags or [])
        ),
        direct_clients[0] if direct_clients else None,
    )
    cards: list[str] = []
    for env_id in SUPPORTED_ENVS:
        run_options = list_supervised_run_options(project_root, env_id=env_id, limit=12)
        trace_options = list_trace_options(project_root, env_id=env_id, limit=24)
        run_inputs = []
        for index, option in enumerate(run_options):
            checked_attr = " checked" if index < min(3, len(run_options)) else ""
            run_inputs.append(
                f"""
                <label class="analysis-run-option">
                  <input type="checkbox" name="run_dirs" value="{html.escape(str(option.run_dir))}"{checked_attr}>
                  <span class="analysis-run-copy">
                    <strong>{html.escape(option.run_name)}</strong>
                    <span class="analysis-run-meta">{option.completed_label_count}/{option.label_count or '—'} labels · {option.candidate_count} candidates</span>
                  </span>
                </label>
                """
            )
        trace_select_options = "".join(
            f'<option value="{html.escape(str(option.trace_path))}">{html.escape(option.dataset_name)} / {html.escape(option.split_name)} / {html.escape(option.sample_name)} · {option.timestep_count} steps</option>'
            for option in trace_options
        )
        direct_client_options = "".join(
            f'<option value="{html.escape(client.client_id)}"{" selected" if local_direct is not None and client.client_id == local_direct.client_id else ""}>{html.escape(_client_title(client))}</option>'
            for client in sorted(direct_clients, key=lambda item: _client_title(item).lower())
        )
        if run_inputs and trace_select_options:
            form_html = f"""
            <form method="post" action="/ui/analysis/create" class="client-settings-form">
              <input type="hidden" name="next" value="/analysis">
              <input type="hidden" name="env_id" value="{html.escape(env_id)}">
              <details class="analysis-run-picker" open>
                <summary>Run selection · choose at least two supervised runs</summary>
                <div class="analysis-run-list">{''.join(run_inputs)}</div>
              </details>
              <div class="analysis-form-grid">
                <label>
                  <span class="metric-subtle">Single-trace fallback</span>
                  <select name="trace_path">{trace_select_options}</select>
                </label>
                <label>
                  <span class="metric-subtle">Candidate windows</span>
                  <input type="text" value="10 fixed bins (0–10%, 10–20%, …, 90–100%)" readonly>
                </label>
                <label>
                  <span class="metric-subtle">Samples per bin</span>
                  <input type="number" name="sampling_repeats" min="1" max="32" step="1" value="32">
                </label>
                <label>
                  <span class="metric-subtle">Execution client</span>
                  <select name="preferred_client_id">
                    <option value="">Any direct client</option>
                    {direct_client_options}
                  </select>
                </label>
              </div>
              <label class="analysis-run-option">
                <input type="checkbox" name="use_env_test_set" value="1" checked>
                <span class="analysis-run-copy">
                  <strong>Average across full test set (recommended)</strong>
                  <span class="analysis-run-meta">When checked, analysis runs over all traces in this environment's test split.</span>
                </span>
              </label>
              <p class="metric-subtle">Compares percentile windows instead of absolute budgets. Each run is split into 10 bins, then distances are averaged over the configured Monte Carlo samples per bin and across selected test traces.</p>
              <div class="analysis-submit-row">
                <button class="secondary-button" type="submit">Queue consistency analysis</button>
              </div>
            </form>
            """
        else:
            form_html = '<p class="empty">Need at least two supervised runs and test-set traces for this environment.</p>'
        cards.append(
            f"""
            <details class="analysis-env-card analysis-env-panel">
              <summary class="analysis-env-head">
                <div>
                  <p class="section-kicker">Direct-only analysis</p>
                  <h3>{html.escape(env_id.upper())}</h3>
                </div>
                <span class="mini-chip">{len(run_options)} runs · {len(trace_options)} traces</span>
              </summary>
              <div class="analysis-env-panel-body">
                {form_html}
              </div>
            </details>
            """
        )
    return f"""
    <details class="panel compact-panel panel-static analysis-launch-panel">
      <summary class="card-topline">
        <div>
          <p class="section-kicker">Launch</p>
          <h2>Supervised Consistency</h2>
        </div>
        <div class="stats-inline"><span class="mini-chip">Direct clients {len(direct_clients)}</span></div>
      </summary>
      <div class="analysis-launch-grid">{''.join(cards)}</div>
    </details>
    """


def _render_supervised_consistency_results_panel(*, project_root: Path) -> str:
    results = list_supervised_consistency_results(project_root, limit=16)
    grouped_by_env: dict[str, list[SupervisedConsistencyResult]] = defaultdict(list)
    for result in results:
        grouped_by_env[str(result.env_id or "").strip().lower()].append(result)
    filtered_results: list[SupervisedConsistencyResult] = []
    for env_id in SUPPORTED_ENVS:
        filtered_results.extend(filter_supervised_consistency_results_to_latest_setup(grouped_by_env.get(env_id) or []))
    results = sorted(
        filtered_results,
        key=lambda item: item.created_at or datetime.min,
        reverse=True,
    )[:16]
    if not results:
        return '<section class="panel compact-panel"><h2>Results</h2><p class="empty">No analysis runs yet.</p></section>'
    rows: list[str] = []
    for result in results:
        payload = result.payload
        checkpoints = payload.get("checkpoint_summaries") if isinstance(payload.get("checkpoint_summaries"), list) else []
        final_summary = checkpoints[-1] if checkpoints else {}
        final_similarity = float(final_summary.get("mean_similarity") or 0.0) if isinstance(final_summary, dict) else 0.0
        run_count = len(payload.get("run_names") or [])
        checkpoint_count = int(payload.get("window_count") or 0) if int(payload.get("window_count") or 0) > 0 else len(payload.get("checkpoints") or [])
        checkpoint_label = "bins" if int(payload.get("window_count") or 0) > 0 else "checkpoints"
        rows.append(
            f"""
            <article class="analysis-result-card">
              <div class="analysis-result-head">
                <div>
                  <h3 class="analysis-result-card-title"><a href="/analysis/{html.escape(result.analysis_run_id)}">{html.escape(result.analysis_run_id)}</a></h3>
                  <div class="analysis-result-card-meta">{html.escape(result.env_id.upper())} · {run_count} runs · {checkpoint_count} {checkpoint_label}</div>
                </div>
                <div class="analysis-result-links">
                  <a class="ghost-link" href="/analysis/{html.escape(result.analysis_run_id)}">Open</a>
                  <a class="ghost-link" href="{_analysis_asset_href(result.summary_path)}">JSON</a>
                </div>
              </div>
              <div class="stats-inline">
                <span class="mini-chip">similarity {final_similarity:.3f}</span>
                <span class="mini-chip">{_display_time(result.created_at)}</span>
              </div>
            </article>
            """
        )
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div><p class="section-kicker">Recent outputs</p><h2>Analysis Results</h2></div>
      </div>
      <div class="analysis-result-list">{''.join(rows)}</div>
    </section>
    """


def _analysis_jobs_for_env(jobs: list[JobRecord], *, env_id: str) -> list[JobRecord]:
    filtered = [
        job
        for job in jobs
        if str(job.spec.action or "").strip() == "analyze-supervised-consistency"
        and _job_env_id(job) == env_id
    ]
    return sorted(
        filtered,
        key=lambda item: (item.updated_at or item.created_at),
        reverse=True,
    )


def _analysis_result_similarity(result: SupervisedConsistencyResult) -> float | None:
    payload = result.payload if isinstance(result.payload, dict) else {}
    window_summaries = payload.get("window_summaries") if isinstance(payload.get("window_summaries"), list) else []
    checkpoint_summaries = payload.get("checkpoint_summaries") if isinstance(payload.get("checkpoint_summaries"), list) else []
    summary_rows = window_summaries if window_summaries else checkpoint_summaries
    final_summary = summary_rows[-1] if summary_rows else {}
    if not isinstance(final_summary, dict):
        return None
    raw_value = final_summary.get("mean_similarity_all_labels")
    if raw_value is None:
        raw_value = final_summary.get("mean_similarity")
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _analysis_result_summary_rows(result: SupervisedConsistencyResult) -> list[dict[str, Any]]:
    payload = result.payload if isinstance(result.payload, dict) else {}
    window_summaries = payload.get("window_summaries") if isinstance(payload.get("window_summaries"), list) else []
    checkpoint_summaries = payload.get("checkpoint_summaries") if isinstance(payload.get("checkpoint_summaries"), list) else []
    rows = window_summaries if window_summaries else checkpoint_summaries
    return [item for item in rows if isinstance(item, dict)]


def _analysis_metric_points(
    result: SupervisedConsistencyResult,
    *,
    metric_key: str,
    fallback_keys: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    std_key = metric_key.replace("_per_label", "_std_per_label") if "_per_label" in metric_key else ""
    count_keys: tuple[str, ...] = ()
    if "pattern_similarity_to_final" in metric_key or "behavior_to_final" in metric_key:
        count_keys = ("pattern_to_final_sample_count", "behavior_to_final_sample_count", "sample_count")
    elif "code_embedding_similarity_to_final" in metric_key or "code_to_final" in metric_key:
        count_keys = ("code_embedding_to_final_sample_count", "code_to_final_sample_count", "sample_count")
    elif "code_diversity" in metric_key:
        count_keys = ("code_diversity_label_samples", "sample_count")
    else:
        count_keys = ("sample_count",)
    for item in _analysis_result_summary_rows(result):
        try:
            window_index = int(item.get("window_index"))
        except (TypeError, ValueError):
            continue
        raw_value = item.get(metric_key)
        if raw_value is None:
            for fallback_key in fallback_keys:
                raw_value = item.get(fallback_key)
                if raw_value is not None:
                    break
        try:
            diversity = float(raw_value)
        except (TypeError, ValueError):
            continue
        label = str(item.get("window_label") or f"bin {window_index + 1}").strip() or f"bin {window_index + 1}"
        std_value = 0.0
        try:
            std_value = max(0.0, float(item.get(std_key) or 0.0)) if std_key else 0.0
        except (TypeError, ValueError):
            std_value = 0.0
        count_value = 1
        for count_key in count_keys:
            try:
                parsed = int(item.get(count_key) or 0)
            except (TypeError, ValueError):
                parsed = 0
            if parsed > 0:
                count_value = parsed
                break
        points.append(
            {
                "x": window_index,
                "label": label,
                "y": max(0.0, min(1.0, diversity)),
                "std": std_value,
                "n": count_value,
            }
        )
    return points


def _analysis_metric_points_transformed(
    result: SupervisedConsistencyResult,
    *,
    metric_key: str,
    fallback_keys: tuple[str, ...] = (),
    transform: Callable[[float], float] | None = None,
) -> list[dict[str, Any]]:
    points = _analysis_metric_points(
        result,
        metric_key=metric_key,
        fallback_keys=fallback_keys,
    )
    if transform is None:
        return points
    transformed: list[dict[str, Any]] = []
    for item in points:
        updated = dict(item)
        try:
            updated["y"] = max(0.0, min(1.0, float(transform(float(item["y"])))))
        except (TypeError, ValueError):
            continue
        transformed.append(updated)
    return transformed


def _analysis_aggregate_series(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[int, list[tuple[float, float, int]]] = defaultdict(list)
    labels: dict[int, str] = {}
    for item in points:
        try:
            window_index = int(item.get("x"))
            diversity = float(item.get("y"))
            std_value = max(0.0, float(item.get("std") or 0.0))
            sample_count = max(1, int(item.get("n") or 1))
        except (TypeError, ValueError):
            continue
        buckets[window_index].append((diversity, std_value, sample_count))
        labels.setdefault(window_index, str(item.get("label") or f"bin {window_index + 1}"))
    aggregated: list[dict[str, Any]] = []
    for window_index in sorted(buckets):
        values = buckets[window_index]
        if not values:
            continue
        total_n = sum(sample_count for _, _, sample_count in values)
        weighted_mean = sum(mean * sample_count for mean, _, sample_count in values) / float(max(1, total_n))
        if total_n <= 1:
            pooled_std = values[0][1] if values else 0.0
        else:
            pooled_numerator = 0.0
            for mean, std_value, sample_count in values:
                pooled_numerator += max(0, sample_count - 1) * (std_value**2)
                pooled_numerator += sample_count * ((mean - weighted_mean) ** 2)
            pooled_std = math.sqrt(max(0.0, pooled_numerator / float(max(1, total_n - 1))))
        aggregated.append(
            {
                "x": window_index,
                "label": labels.get(window_index, f"bin {window_index + 1}"),
                "y": weighted_mean,
                "std": pooled_std,
                "n": total_n,
            }
        )
    return aggregated


def _analysis_convergence_plot_svg(
    *,
    pattern_series: list[dict[str, Any]],
    code_series: list[dict[str, Any]],
    env_id: str,
) -> str:
    width = 560
    height = 280
    left = 58
    right = 18
    top = 18
    bottom = 46
    inner_width = width - left - right
    inner_height = height - top - bottom
    all_points = [*pattern_series, *code_series]
    if not all_points:
        return (
            '<div class="analysis-plot-shell">'
            '<p class="empty">No analysis samples yet. Queue a supervised analysis run for this environment.</p>'
            "</div>"
        )
    x_values = [int(item["x"]) for item in all_points]
    y_values: list[float] = []
    for item in all_points:
        mean = float(item["y"])
        std = max(0.0, float(item.get("std") or 0.0))
        y_values.append(max(0.0, mean - std))
        y_values.append(min(1.0, mean + std))
    min_x = min(x_values)
    max_x = max(x_values)
    if min_x == max_x:
        min_x = max(0, min_x - 1)
        max_x = max_x + 1
    max_y = max(max(y_values), 0.05)
    max_y = min(1.0, max_y * 1.08)
    if max_y <= 0:
        max_y = 1.0

    def map_x(value: int) -> float:
        return left + ((float(value) - float(min_x)) / float(max_x - min_x)) * inner_width

    def map_y(value: float) -> float:
        return top + (1.0 - (float(value) / float(max_y))) * inner_height

    def line_path(series: list[dict[str, Any]]) -> str:
        if not series:
            return ""
        return " ".join(
            (
                f"{'M' if index == 0 else 'L'} "
                f"{map_x(int(item['x'])):.2f} {map_y(float(item['y'])):.2f}"
            )
            for index, item in enumerate(series)
        )

    def band_path(series: list[dict[str, Any]]) -> str:
        if not series:
            return ""
        upper = [
            (
                map_x(int(item["x"])),
                map_y(min(1.0, float(item["y"]) + max(0.0, float(item.get("std") or 0.0)))),
            )
            for item in series
        ]
        lower = [
            (
                map_x(int(item["x"])),
                map_y(max(0.0, float(item["y"]) - max(0.0, float(item.get("std") or 0.0)))),
            )
            for item in reversed(series)
        ]
        points = upper + lower
        return " ".join(
            (
                f"{'M' if index == 0 else 'L'} {x:.2f} {y:.2f}"
                for index, (x, y) in enumerate(points)
            )
        ) + " Z"

    def circles(series: list[dict[str, Any]], color: str, label: str) -> str:
        return "".join(
            (
                f'<circle cx="{map_x(int(item["x"])):.2f}" cy="{map_y(float(item["y"])):.2f}" '
                f'r="3.4" fill="{color}">'
                f'<title>{env_id.upper()} · {html.escape(label)} · {str(item.get("label") or int(item["x"]))} · {float(item["y"]):.3f} ± {float(item.get("std") or 0.0):.3f} · {int(item["n"])} sample(s)</title>'
                "</circle>"
            )
            for item in series
        )

    y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    x_ticks = sorted(set(x_values))
    grid_lines = "".join(
        (
            f'<line x1="{left}" y1="{map_y(value):.2f}" x2="{width - right}" y2="{map_y(value):.2f}" '
            'stroke="rgba(87,104,125,0.18)" stroke-width="1" />'
            f'<text x="{left - 8}" y="{map_y(value) + 4:.2f}" text-anchor="end" fill="var(--muted)" font-size="11">{value:.2f}</text>'
        )
        for value in y_ticks
        if value <= max_y + 1e-9
    )
    x_labels = {int(item["x"]): str(item.get("label") or f"bin {int(item['x']) + 1}") for item in all_points}
    x_tick_lines = "".join(
        (
            f'<line x1="{map_x(value):.2f}" y1="{top}" x2="{map_x(value):.2f}" y2="{height - bottom}" '
            'stroke="rgba(87,104,125,0.12)" stroke-width="1" />'
            f'<text x="{map_x(value):.2f}" y="{height - bottom + 18}" text-anchor="middle" fill="var(--muted)" font-size="11">{html.escape(x_labels.get(value, str(value)))}</text>'
        )
        for value in x_ticks
    )
    pattern_path = line_path(pattern_series)
    code_path = line_path(code_series)
    pattern_band = band_path(pattern_series)
    code_band = band_path(code_series)
    pattern_samples = sum(int(item.get("n") or 0) for item in pattern_series)
    code_samples = sum(int(item.get("n") or 0) for item in code_series)
    colors = _shared_plot_colors()
    pattern_color = colors[0]
    code_color = colors[1] if len(colors) > 1 else "#0474b4"
    plot_title = f"{env_id.upper()} convergence plot"
    svg_html = f"""
      <svg class="analysis-plot-frame" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(plot_title)}">
        <rect x="0" y="0" width="{width}" height="{height}" fill="rgba(255,255,255,0.98)" />
        {grid_lines}
        {x_tick_lines}
        <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="rgba(41,52,64,0.75)" stroke-width="1.2" />
        <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="rgba(41,52,64,0.75)" stroke-width="1.2" />
        {f'<path d="{pattern_band}" fill="{html.escape(pattern_color)}" fill-opacity="0.16" stroke="none" />' if pattern_band else ''}
        {f'<path d="{code_band}" fill="{html.escape(code_color)}" fill-opacity="0.14" stroke="none" />' if code_band else ''}
        {f'<path d="{pattern_path}" fill="none" stroke="{html.escape(pattern_color)}" stroke-width="2.2" />' if pattern_path else ''}
        {f'<path d="{code_path}" fill="none" stroke="{html.escape(code_color)}" stroke-width="2.2" />' if code_path else ''}
        {circles(pattern_series, pattern_color, "Pattern-based similarity to final bin")}
        {circles(code_series, code_color, "Code-embedding similarity to final bin")}
        <text x="{left + inner_width / 2:.2f}" y="{height - 10}" text-anchor="middle" fill="var(--muted)" font-size="12">Candidate percentile bin</text>
        <text x="16" y="{top + inner_height / 2:.2f}" text-anchor="middle" fill="var(--muted)" font-size="12" transform="rotate(-90 16 {top + inner_height / 2:.2f})">Convergence</text>
      </svg>
    """
    payload_id = _analysis_plot_payload_id(f"{env_id}-convergence")
    payload = {"title": plot_title, "html": svg_html}
    return f"""
    <div class="analysis-plot-shell">
      {_analysis_plot_trigger(payload_id=payload_id, inner_html=svg_html)}
      <div class="analysis-plot-legend">
        <span class="mini-chip"><span class="analysis-legend-swatch" style="background:{html.escape(pattern_color)};"></span>Pattern-based similarity to final bin mean ± std · {pattern_samples} learned-label samples</span>
        <span class="mini-chip"><span class="analysis-legend-swatch" style="background:{html.escape(code_color)};"></span>Code-embedding similarity to final bin mean ± std · {code_samples} learned-label samples</span>
      </div>
      <script type="application/json" id="{payload_id}">{_summary_json_script_payload(payload)}</script>
    </div>
    """


def _render_analysis_home_pattern_similarity_panel(
    *,
    results_by_env: dict[str, list[SupervisedConsistencyResult]],
) -> str:
    chart_data = build_combined_analysis_pattern_chart_data(results_by_env=results_by_env)
    if not chart_data["series"]:
        return """
        <section class="panel compact-panel panel-static">
          <div class="card-topline">
            <div>
              <p class="section-kicker">Overview</p>
            </div>
          </div>
          <p class="empty">No analysis samples yet.</p>
        </section>
        """
    image_data_uri = render_analysis_plot_image_data_uri(
        title=str(chart_data["title"]),
        x_label=str(chart_data["x_label"]),
        y_label=str(chart_data["y_label"]),
        series=list(chart_data["series"]),
        y_limits=(0.0, 1.0),
        x_tick_labels=dict(chart_data["x_tick_labels"]),
    )
    legend_bits = [
        f'<span class="mini-chip"><span class="analysis-legend-swatch" style="background:{html.escape(str(item.get("color") or "#617182"))};"></span>{html.escape(str(item.get("label") or ""))} mean ± std</span>'
        for item in chart_data["series"]
    ]
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Overview</p>
          <p class="metric-subtle">Each environment is shown as a single line over the shared 10 candidate-percentile bins.</p>
        </div>
        <div class="stats-inline">
          <span class="mini-chip">{len(chart_data["series"])} environments</span>
          <button class="ghost-link" type="button" onclick="window.location.href='/analysis?refresh=1&t=' + Date.now()">Regenerate plot</button>
          <a class="ghost-link" href="/api/ui/analysis/plot.pdf">Download PDF</a>
        </div>
      </div>
      <div class="chart-shell">
        <img class="detail-chart" src="{image_data_uri}" alt="Pattern-based similarity to final bin across environments">
      </div>
      <div class="analysis-plot-legend">{''.join(legend_bits)}</div>
    </section>
    """


def build_combined_analysis_pattern_chart_data(
    *,
    results_by_env: dict[str, list[SupervisedConsistencyResult]],
) -> dict[str, Any]:
    colors = _shared_plot_colors()
    series: list[dict[str, Any]] = []
    x_tick_labels: dict[int, str] = {}
    for index, env_id in enumerate(SUPPORTED_ENVS):
        env_results = list(results_by_env.get(env_id) or [])
        pattern_points_raw = [
            point
            for result in env_results
            for point in _analysis_metric_points(
                result,
                metric_key="mean_pattern_similarity_to_final_per_label",
                fallback_keys=("mean_behavior_to_final_per_label",),
            )
        ]
        pattern_series = _analysis_aggregate_series(pattern_points_raw)
        if not pattern_series:
            continue
        points = [(float(item["x"]), float(item["y"])) for item in pattern_series]
        band = [
            (
                float(item["x"]),
                max(0.0, float(item["y"]) - max(0.0, float(item.get("std") or 0.0))),
                min(1.0, float(item["y"]) + max(0.0, float(item.get("std") or 0.0))),
            )
            for item in pattern_series
        ]
        for item in pattern_series:
            x_tick_labels[int(item["x"])] = str(item.get("label") or f"bin {int(item['x']) + 1}")
        series.append(
            {
                "label": env_id.upper(),
                "color": colors[index % len(colors)],
                "points": points,
                "band": band,
            }
        )
    return {
        "title": "Pattern-Based Similarity To Final Bin",
        "x_label": "Candidate percentile bin",
        "y_label": "Pattern-based similarity to final bin",
        "series": series,
        "x_tick_labels": x_tick_labels,
    }


def _render_analysis_plot_bytes(
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[dict[str, object]],
    y_limits: tuple[float, float] = (0.0, 1.0),
    x_tick_labels: dict[int, str] | None = None,
    fmt: str = "svg",
) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    def _size_as_float(value: object, default: float) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        named = {
            "xx-small": 5.0,
            "x-small": 6.0,
            "small": 7.0,
            "medium": 8.0,
            "large": 10.0,
            "x-large": 12.0,
            "xx-large": 14.0,
        }
        text = str(value or "").strip().lower()
        if text in named:
            return named[text]
        try:
            return float(text)
        except ValueError:
            return default

    config = _shared_plot_style_config()
    style_name = str(config.get("style") or "").strip()
    if style_name:
        plt.style.use(style_name)
    rc_params = config.get("rcParams") if isinstance(config.get("rcParams"), dict) else {}
    if isinstance(rc_params, dict):
        matplotlib.rcParams.update(rc_params)
        serif_candidates = rc_params.get("font.serif")
        if isinstance(serif_candidates, list):
            available_fonts = {font.name for font in font_manager.fontManager.ttflist}
            chosen = next((name for name in serif_candidates if name in available_fonts), None)
            if chosen is None and "DejaVu Serif" in available_fonts:
                chosen = "DejaVu Serif"
            if chosen is not None:
                matplotlib.rcParams["font.family"] = ["serif"]
                matplotlib.rcParams["font.serif"] = [chosen]
                matplotlib.rcParams["mathtext.rm"] = chosen
                matplotlib.rcParams["mathtext.it"] = f"{chosen}:italic"
                matplotlib.rcParams["mathtext.bf"] = f"{chosen}:bold"
    matplotlib.rcParams["font.size"] = min(_size_as_float(matplotlib.rcParams.get("font.size", 9), 9.0), 7.5)
    matplotlib.rcParams["axes.titlesize"] = min(_size_as_float(matplotlib.rcParams.get("axes.titlesize", 9), 9.0), 7.5)
    matplotlib.rcParams["axes.labelsize"] = min(_size_as_float(matplotlib.rcParams.get("axes.labelsize", 9), 9.0), 7.0)
    matplotlib.rcParams["xtick.labelsize"] = min(_size_as_float(matplotlib.rcParams.get("xtick.labelsize", 8), 8.0), 6.5)
    matplotlib.rcParams["ytick.labelsize"] = min(_size_as_float(matplotlib.rcParams.get("ytick.labelsize", 8), 8.0), 6.5)
    matplotlib.rcParams["legend.fontsize"] = min(_size_as_float(matplotlib.rcParams.get("legend.fontsize", 6), 6.0), 5.8)
    figure_settings = ((config.get("aesthetics") or {}).get("figure") or {})
    grid_settings = ((config.get("aesthetics") or {}).get("grid") or {})
    legend_settings = ((config.get("aesthetics") or {}).get("legend") or {})
    error_region_settings = ((config.get("aesthetics") or {}).get("error_region") or {})
    label_settings = config.get("labels") or {}
    base_width, base_height = _shared_plot_figure_size("default")
    figure_size = (base_width * 1.12, base_height * 0.98)
    fig, ax = plt.subplots(figsize=figure_size)
    facecolor = str(figure_settings.get("facecolor") or "white")
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)
    min_x: float | None = None
    max_x: float | None = None
    error_alpha = float(error_region_settings.get("alpha") or 0.15)
    tick_values: set[int] = set()
    for item in series:
        raw_points = item.get("points")
        if not isinstance(raw_points, list) or not raw_points:
            continue
        xs: list[float] = []
        ys: list[float] = []
        for point in raw_points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            try:
                x_value = float(point[0])
                y_value = float(point[1])
            except (TypeError, ValueError):
                continue
            xs.append(x_value)
            ys.append(y_value)
            tick_values.add(int(round(x_value)))
        if not xs:
            continue
        min_x = min(xs) if min_x is None else min(min_x, min(xs))
        max_x = max(xs) if max_x is None else max(max_x, max(xs))
        color = str(item.get("color") or "#617182")
        label = str(item.get("label") or "").strip()
        raw_band = item.get("band")
        if isinstance(raw_band, list) and raw_band:
            band_x: list[float] = []
            band_low: list[float] = []
            band_high: list[float] = []
            for point in raw_band:
                if not isinstance(point, (list, tuple)) or len(point) != 3:
                    continue
                try:
                    band_x.append(float(point[0]))
                    band_low.append(float(point[1]))
                    band_high.append(float(point[2]))
                except (TypeError, ValueError):
                    continue
        if band_x:
                ax.fill_between(band_x, band_low, band_high, color=color, alpha=error_alpha, linewidth=0)
        ax.plot(
            xs,
            ys,
            color=color,
            label=label,
            linewidth=1.1,
            solid_capstyle="butt",
            solid_joinstyle="miter",
        )
    ax.set_xlabel(x_label, labelpad=float(label_settings.get("padding") or 10))
    ax.set_ylabel(y_label, labelpad=float(label_settings.get("padding") or 10))
    y_grid_settings = dict(grid_settings)
    y_grid_settings["linewidth"] = min(float(y_grid_settings.get("linewidth") or 0.25), 0.10)
    ax.grid(axis="y", **y_grid_settings)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_kwargs = dict(legend_settings)
    legend_kwargs["frameon"] = True
    legend_kwargs["facecolor"] = facecolor
    legend_kwargs["framealpha"] = 1.0
    legend_kwargs["ncol"] = 2
    legend_kwargs["handlelength"] = 1.8
    ax.legend(**legend_kwargs)
    if min_x is None or max_x is None:
        min_x, max_x = 0.0, 1.0
    if math.isclose(min_x, max_x):
        max_x += 1.0
    ax.set_xlim(min_x - 0.1, max_x + 0.1)
    ax.set_ylim(float(y_limits[0]), float(y_limits[1]))
    sorted_ticks = sorted(tick_values)
    if sorted_ticks:
        ax.set_xticks(sorted_ticks)
        if x_tick_labels:
            ax.set_xticklabels([str(x_tick_labels.get(int(tick), tick)) for tick in sorted_ticks], rotation=30, ha="right")
    buffer = io.BytesIO()
    fig.tight_layout(pad=0.25)
    fig.savefig(buffer, format=fmt, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def render_analysis_plot_pdf_bytes(
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[dict[str, object]],
    y_limits: tuple[float, float] = (0.0, 1.0),
    x_tick_labels: dict[int, str] | None = None,
) -> bytes:
    return _render_analysis_plot_bytes(
        title=title,
        x_label=x_label,
        y_label=y_label,
        series=series,
        y_limits=y_limits,
        x_tick_labels=x_tick_labels,
        fmt="pdf",
    )


def render_analysis_plot_image_data_uri(
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[dict[str, object]],
    y_limits: tuple[float, float] = (0.0, 1.0),
    x_tick_labels: dict[int, str] | None = None,
) -> str:
    svg_bytes = _render_analysis_plot_bytes(
        title=title,
        x_label=x_label,
        y_label=y_label,
        series=series,
        y_limits=y_limits,
        x_tick_labels=x_tick_labels,
        fmt="svg",
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg_bytes).decode("ascii")


def _analysis_env_dashboard_card(
    *,
    env_id: str,
    jobs: list[JobRecord],
    env_results: list[SupervisedConsistencyResult],
    env_status: dict[str, Any] | None = None,
) -> str:
    env_jobs = _analysis_jobs_for_env(jobs, env_id=env_id)
    queued_jobs = [job for job in env_jobs if job.status.value in QUEUE_VISIBLE_STATUSES]
    active_jobs = [job for job in env_jobs if job.status.value in ACTIVE_QUEUE_STATUSES]
    pattern_points_raw = [
        point
        for result in env_results
        for point in _analysis_metric_points(
            result,
            metric_key="mean_pattern_similarity_to_final_per_label",
            fallback_keys=("mean_behavior_to_final_per_label",),
        )
    ]
    code_points_raw = [
        point
        for result in env_results
        for point in _analysis_metric_points(
            result,
            metric_key="mean_code_embedding_similarity_to_final_per_label",
            fallback_keys=("mean_code_to_final_per_label",),
        )
    ]
    pattern_series = _analysis_aggregate_series(pattern_points_raw)
    code_series = _analysis_aggregate_series(code_points_raw)
    latest_result = env_results[0] if env_results else None
    latest_job = active_jobs[0] if active_jobs else (env_jobs[0] if env_jobs else None)
    env_status_payload = dict(env_status or {})
    status_label = str(env_status_payload.get("status") or "").strip().lower()
    if not status_label:
        status_label = latest_job.status.value if latest_job is not None else ("succeeded" if env_results else "idle")
    summary_parts = [
        f"{len(env_results)} analysis runs",
        f"{len(pattern_series)} bins with pattern similarity",
        f"{len(code_series)} bins with code similarity",
    ]
    if latest_result is not None:
        summary_parts.append(_display_time(latest_result.created_at))
    final_pattern = pattern_series[-1]["y"] if pattern_series else None
    final_pattern_label = (
        f"{float(final_pattern):.3f} final similarity"
        if isinstance(final_pattern, (int, float))
        else "No bins yet"
    )
    return f"""
    <section class="panel compact-panel panel-static analysis-dashboard-card">
      <div class="card-topline">
        <div>
          <h2><a href="/analysis/env/{html.escape(env_id)}">{html.escape(env_id.upper())}</a></h2>
          <p class="metric-subtle">{html.escape(" · ".join(summary_parts))}</p>
        </div>
        <div class="stats-inline">
          {_pill(status_label, status_label)}
          <span class="mini-chip">Queued {len(queued_jobs)}</span>
        </div>
      </div>
      <div class="stats-inline">
        <span class="mini-chip">{len(pattern_series)} pattern bins</span>
        <span class="mini-chip">{len(code_series)} code bins</span>
        <span class="mini-chip">{html.escape(final_pattern_label)}</span>
      </div>
      <div class="analysis-dashboard-actions">
        <form method="post" action="/ui/analysis/queue" class="inline-form">
          <input type="hidden" name="next" value="/analysis">
          <input type="hidden" name="env_id" value="{html.escape(env_id)}">
          <button class="secondary-button" type="submit">Queue Another</button>
        </form>
        <a class="ghost-link" href="/analysis/env/{html.escape(env_id)}">View runs</a>
      </div>
    </section>
    """


def _render_analysis_environment_convergence_panel(
    *,
    env_id: str,
    env_results: list[SupervisedConsistencyResult],
) -> str:
    pattern_points_raw = [
        point
        for result in env_results
        for point in _analysis_metric_points(
            result,
            metric_key="mean_pattern_similarity_to_final_per_label",
            fallback_keys=("mean_behavior_to_final_per_label",),
        )
    ]
    code_points_raw = [
        point
        for result in env_results
        for point in _analysis_metric_points(
            result,
            metric_key="mean_code_embedding_similarity_to_final_per_label",
            fallback_keys=("mean_code_to_final_per_label",),
        )
    ]
    pattern_series = _analysis_aggregate_series(pattern_points_raw)
    code_series = _analysis_aggregate_series(code_points_raw)
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Overview</p>
          <h2>Similarity To Final Bin</h2>
          <p class="metric-subtle">Each completed learned label is treated as its own learning trajectory. Each bin is compared against the final bin of that same learned-label trajectory, and the curves show the mean/std across all contributing learned labels.</p>
        </div>
        <div class="stats-inline">
          <span class="mini-chip">{len(pattern_series)} pattern bins</span>
          <span class="mini-chip">{len(code_series)} code bins</span>
        </div>
      </div>
      {_analysis_convergence_plot_svg(pattern_series=pattern_series, code_series=code_series, env_id=env_id)}
    </section>
    """


def _analysis_compact_progress(
    *,
    env_id: str,
    jobs: list[JobRecord],
    results: list[SupervisedConsistencyResult],
    logs_by_job_id: dict[str, list[str]] | None = None,
) -> str:
    active_jobs = [job for job in jobs if job.status.value in ACTIVE_QUEUE_STATUSES]
    focus_job = active_jobs[0] if active_jobs else (jobs[0] if jobs else None)
    latest_result = results[0] if results else None
    latest_similarity = _analysis_result_similarity(latest_result) if latest_result is not None else None
    progress_label = "—"
    status_copy = "No analysis queued yet."

    if focus_job is not None:
        progress = _job_progress_payload(focus_job)
        current: int | None = None
        total: int | None = None
        summary = _sanitize_display_text(str((progress or {}).get("summary") or "").strip())
        phase = _sanitize_display_text(str((progress or {}).get("phase") or "").strip()) or "Consistency analysis"
        log_lines_lookup = (logs_by_job_id or {}).get(focus_job.job_id) if isinstance(logs_by_job_id, dict) else None
        log_lines = list(log_lines_lookup or []) if log_lines_lookup is not None else None
        inferred = _analysis_progress_from_recent_logs(focus_job, log_lines=log_lines)
        for key in ("current", "completed", "completed_count"):
            try:
                if (progress or {}).get(key) is not None:
                    current = int((progress or {}).get(key))
                    break
            except (TypeError, ValueError):
                continue
        for key in ("total", "total_count"):
            try:
                if (progress or {}).get(key) is not None:
                    total = int((progress or {}).get(key))
                    break
            except (TypeError, ValueError):
                continue
        if current is None and isinstance(inferred.get("current"), int):
            current = int(inferred["current"])
        if total is None and isinstance(inferred.get("total"), int):
            total = int(inferred["total"])
        inferred_summary = _sanitize_display_text(str(inferred.get("summary") or "").strip())
        if inferred_summary and not summary:
            summary = inferred_summary
        inferred_phase = _sanitize_display_text(str(inferred.get("phase") or "").strip())
        if inferred_phase and (not phase or phase.lower() == "consistency analysis"):
            phase = inferred_phase
        if current is not None and total is not None and total > 0:
            progress_label = f"{current}/{total}"
        elif focus_job.status.value == "succeeded":
            progress_label = "Done"
        elif focus_job.status.value in ACTIVE_QUEUE_STATUSES:
            progress_label = "Running"
        else:
            progress_label = focus_job.status.value
        status_copy = summary or phase or focus_job.status.value
    elif latest_result is not None:
        progress_label = "Done"
        status_copy = f"Latest result {_display_time(latest_result.created_at)}"

    similarity_text = f"{latest_similarity:.3f}" if latest_similarity is not None else "—"
    return f"""
    <div class="required-run-stats">
      <div class="required-run-stat"><span>Results</span><strong>{len(results)}</strong></div>
      <div class="required-run-stat"><span>Running</span><strong>{len(active_jobs)}</strong></div>
      <div class="required-run-stat"><span>Progress</span><strong>{html.escape(progress_label)}</strong></div>
      <div class="required-run-stat"><span>Latest</span><strong>{html.escape(similarity_text)}</strong></div>
    </div>
    <div class="required-row-meta">{html.escape(status_copy)}</div>
    """


def _render_analysis_env_catalog_panel(
    *,
    jobs: list[JobRecord],
    project_root: Path,
    logs_by_job_id: dict[str, list[str]] | None = None,
) -> str:
    env_cards: list[str] = []
    for env_id in SUPPORTED_ENVS:
        env_jobs = _analysis_jobs_for_env(jobs, env_id=env_id)
        env_results = list_supervised_consistency_results(project_root, env_id=env_id, limit=24)
        latest_result = env_results[0] if env_results else None
        latest_text = (
            latest_result.analysis_run_id
            if latest_result is not None
            else "No results yet"
        )
        focus_job = next((job for job in env_jobs if job.status.value in ACTIVE_QUEUE_STATUSES), env_jobs[0] if env_jobs else None)
        status_label = focus_job.status.value if focus_job is not None else ("succeeded" if env_results else "missing")
        visual_class = _required_ablation_cell_visual(focus_job) if focus_job is not None else ("is-covered" if env_results else "is-missing")
        env_cards.append(
            f"""
            <section class="required-card">
              <div class="card-topline">
                <div><h3>{html.escape(env_id.upper())}</h3></div>
                <a class="required-job-link" href="/analysis/env/{html.escape(env_id)}">View results</a>
              </div>
              <div class="required-list">
                <article class="required-row{' is-active-state' if focus_job is not None and focus_job.status.value in ACTIVE_QUEUE_STATUSES else ''}">
                  <div class="required-row-main">
                    <div class="required-row-title">
                      <span class="required-status-dot {visual_class}"></span>
                      <strong><a href="/analysis/env/{html.escape(env_id)}">{html.escape(env_id.upper())}</a></strong>
                    </div>
                    <div class="required-row-meta">{html.escape(latest_text)}</div>
                    {_analysis_compact_progress(env_id=env_id, jobs=env_jobs, results=env_results, logs_by_job_id=logs_by_job_id)}
                  </div>
                  <div class="required-status">
                    {_pill(status_label, status_label)}
                    <span class="mini-chip">{len(env_jobs)} jobs</span>
                  </div>
                </article>
              </div>
            </section>
            """
        )
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Consistency Analysis</p>
          <h2>Tracked Environments</h2>
          <p class="metric-subtle">Compact per-environment analysis progress. Open an environment to inspect current jobs and completed analysis outputs.</p>
        </div>
      </div>
      <div class="required-grid">{''.join(env_cards)}</div>
    </section>
    """


def _render_analysis_env_results_panel(
    *,
    project_root: Path,
    env_id: str,
) -> str:
    results = filter_supervised_consistency_results_to_latest_setup(
        list_supervised_consistency_results(project_root, env_id=env_id, limit=24)
    )
    if not results:
        return '<section class="panel compact-panel"><h2>Results</h2><p class="empty">No analysis runs yet for this environment.</p></section>'
    rows: list[str] = []
    for result in results:
        payload = result.payload
        checkpoints = payload.get("checkpoint_summaries") if isinstance(payload.get("checkpoint_summaries"), list) else []
        final_similarity = _analysis_result_similarity(result)
        run_count = len(payload.get("run_names") or [])
        checkpoint_count = int(payload.get("window_count") or 0) if int(payload.get("window_count") or 0) > 0 else len(payload.get("checkpoints") or [])
        checkpoint_label = "bins" if int(payload.get("window_count") or 0) > 0 else "checkpoints"
        rows.append(
            f"""
            <article class="analysis-result-card">
              <div class="analysis-result-head">
                <div>
                  <h3 class="analysis-result-card-title"><a href="/analysis/{html.escape(result.analysis_run_id)}">{html.escape(result.analysis_run_id)}</a></h3>
                  <div class="analysis-result-card-meta">{run_count} runs · {checkpoint_count} {checkpoint_label}</div>
                </div>
                <div class="analysis-result-links">
                  <a class="ghost-link" href="/analysis/{html.escape(result.analysis_run_id)}">Open</a>
                  <a class="ghost-link" href="{_analysis_asset_href(result.summary_path)}">JSON</a>
                </div>
              </div>
              <div class="stats-inline">
                <span class="mini-chip">similarity {f"{final_similarity:.3f}" if final_similarity is not None else "—"}</span>
                <span class="mini-chip">{_display_time(result.created_at)}</span>
              </div>
            </article>
            """
        )
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div><p class="section-kicker">Per environment</p><h2>Analysis Results</h2></div>
      </div>
      <div class="analysis-result-list">{''.join(rows)}</div>
    </section>
    """


def _render_analysis_env_results_list_panel(
    *,
    env_id: str,
    results: list[SupervisedConsistencyResult],
) -> str:
    del env_id
    if not results:
        return '<section class="panel compact-panel"><h2>Results</h2><p class="empty">No analysis runs yet for this environment.</p></section>'
    rows: list[str] = []
    for result in results[:24]:
        payload = result.payload
        final_similarity = _analysis_result_similarity(result)
        run_count = len(payload.get("run_names") or [])
        checkpoint_count = int(payload.get("window_count") or 0) if int(payload.get("window_count") or 0) > 0 else len(payload.get("checkpoints") or [])
        checkpoint_label = "bins" if int(payload.get("window_count") or 0) > 0 else "checkpoints"
        rows.append(
            f"""
            <article class="analysis-result-card">
              <div class="analysis-result-head">
                <div>
                  <h3 class="analysis-result-card-title"><a href="/analysis/{html.escape(result.analysis_run_id)}">{html.escape(result.analysis_run_id)}</a></h3>
                  <div class="analysis-result-card-meta">{run_count} runs · {checkpoint_count} {checkpoint_label}</div>
                </div>
                <div class="analysis-result-links">
                  <a class="ghost-link" href="/analysis/{html.escape(result.analysis_run_id)}">Open</a>
                  <a class="ghost-link" href="{_analysis_asset_href(result.summary_path)}">JSON</a>
                </div>
              </div>
              <div class="stats-inline">
                <span class="mini-chip">similarity {f"{final_similarity:.3f}" if final_similarity is not None else "—"}</span>
                <span class="mini-chip">{_display_time(result.created_at)}</span>
              </div>
            </article>
            """
        )
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div><p class="section-kicker">Per environment</p><h2>Analysis Results</h2></div>
      </div>
      <div class="analysis-result-list">{''.join(rows)}</div>
    </section>
    """


def _render_analysis_env_jobs_panel(
    *,
    env_id: str,
    jobs: list[JobRecord],
    project_root: Path,
    logs_by_job_id: dict[str, list[str]] | None = None,
) -> str:
    if not jobs:
        return '<section class="panel compact-panel"><h2>Current Jobs</h2><p class="empty">No analysis jobs yet for this environment.</p></section>'
    rows: list[str] = []
    env_results = list_supervised_consistency_results(project_root, env_id=env_id, limit=24)
    for job in jobs:
        metadata = dict(job.spec.metadata or {})
        is_local_analysis = bool(metadata.get("local_analysis"))
        run_dir = str(
            (job.spec.metadata or {}).get("pattern_run_dirs")
            or (job.spec.metadata or {}).get("pattern_run_dir")
            or (job.spec.metadata or {}).get("source_run_dirs")
            or (job.spec.metadata or {}).get("run_dir")
            or ""
        )
        title_html = (
            f"<strong><code>{html.escape(job.job_id[:8])}</code></strong>"
            if is_local_analysis
            else f'<strong><a href="/jobs/{html.escape(job.job_id)}"><code>{html.escape(job.job_id[:8])}</code></a></strong>'
        )
        rows.append(
            f"""
            <article class="required-row{' is-active-state' if job.status.value in ACTIVE_QUEUE_STATUSES else ''}">
              <div class="required-row-main">
                <div class="required-row-title">
                  <span class="required-status-dot {_required_ablation_cell_visual(job)}"></span>
                  {title_html}
                </div>
                <div class="required-row-meta">{html.escape(_queue_job_title(job))} · {_display_time(job.created_at)}</div>
                <div class="required-row-meta">{html.escape(run_dir)}</div>
                {_analysis_compact_progress(env_id=env_id, jobs=[job], results=env_results, logs_by_job_id=logs_by_job_id)}
              </div>
              <div class="required-status">
                {_pill(job.status.value, job.status.value)}
              </div>
            </article>
            """
        )
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div><p class="section-kicker">Per environment</p><h2>Current Jobs</h2></div>
      </div>
      <div class="required-list">{''.join(rows)}</div>
    </section>
    """


def _render_analysis_live_terminal_panel(
    *,
    env_id: str,
    jobs: list[JobRecord],
    logs_by_job_id: dict[str, list[str]] | None = None,
    tmux_status: dict[str, Any] | None = None,
) -> str:
    tmux_payload = dict(tmux_status or {})
    tmux_logs = [str(item) for item in (tmux_payload.get("log_lines") or []) if str(item).strip()]
    tmux_log_text = "\n".join(tmux_logs[-120:]) if tmux_logs else "No tmux log yet."
    tmux_running = bool(tmux_payload.get("is_running"))
    tmux_session_name = str(tmux_payload.get("session_name") or f"analysis-v7-{env_id}").strip()
    if tmux_running or tmux_logs:
        return f"""
        <section class="panel panel-static">
          <div class="card-topline">
            <div>
              <p class="section-kicker">Live progress</p>
              <h2>Analysis Terminal</h2>
              <p class="metric-subtle"><code>{html.escape(tmux_session_name)}</code> · {html.escape(env_id.upper())}</p>
            </div>
            <div class="stats-inline">{_pill("running" if tmux_running else "finished", "running" if tmux_running else "finished")}</div>
          </div>
          <pre class="code-block" style="max-height:18rem;overflow:auto;margin-top:0.5rem;">{html.escape(tmux_log_text)}</pre>
          <script>
            (() => {{
              const running = {str(tmux_running).lower()};
              if (!running) return;
              const refresh = () => window.setTimeout(() => window.location.reload(), 1500);
              window.setTimeout(refresh, 1400);
            }})();
          </script>
        </section>
        """
    focus_job = next(
        (
            job
            for job in jobs
            if job.status.value in ACTIVE_QUEUE_STATUSES | {"queued", "assigned", "starting", "paused", "stopping", "pausing", "resuming"}
        ),
        jobs[0] if jobs else None,
    )
    if focus_job is None:
        return """
        <section class="panel panel-static">
          <div class="card-topline">
            <div>
              <p class="section-kicker">Live progress</p>
              <h2>Analysis Terminal</h2>
              <p class="metric-subtle">No analysis job logs yet for this environment.</p>
            </div>
          </div>
          <pre class="code-block" style="max-height:18rem;overflow:auto;margin-top:0.5rem;">No log yet.</pre>
        </section>
        """
    raw_lines = (
        list((logs_by_job_id or {}).get(focus_job.job_id) or [])
        if isinstance(logs_by_job_id, dict)
        else []
    )
    visible_logs = list(reversed([str(item) for item in raw_lines if str(item).strip()][-120:])) if raw_lines else []
    log_text = "\n".join(visible_logs) if visible_logs else "No log yet."
    is_running = focus_job.status.value not in {"idle", "succeeded", "failed", "cancelled", "stopped"}
    metadata = dict(focus_job.spec.metadata or {})
    sublabel = (
        f'<code>{html.escape(focus_job.job_id[:8])}</code> · {html.escape(_queue_job_title(focus_job))}'
        if metadata.get("local_analysis")
        else f'<a href="/jobs/{html.escape(focus_job.job_id)}"><code>{html.escape(focus_job.job_id[:8])}</code></a> · {html.escape(_queue_job_title(focus_job))}'
    )
    return f"""
    <section class="panel panel-static">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Live progress</p>
          <h2>Analysis Terminal</h2>
          <p class="metric-subtle">{sublabel}</p>
        </div>
        <div class="stats-inline">{_pill(focus_job.status.value, focus_job.status.value)}</div>
      </div>
      <pre class="code-block" style="max-height:18rem;overflow:auto;margin-top:0.5rem;">{html.escape(log_text)}</pre>
      <script>
        (() => {{
          const running = {str(is_running).lower()};
          if (!running) return;
          const refresh = () => window.setTimeout(() => window.location.reload(), 1500);
          window.setTimeout(refresh, 1400);
        }})();
      </script>
    </section>
    """


def _render_analysis_home_live_panel(
    *,
    jobs: list[JobRecord],
    logs_by_job_id: dict[str, list[str]] | None = None,
    tmux_status_by_env: dict[str, dict[str, Any]] | None = None,
    project_root: Path | None = None,
    analysis_status_by_env: dict[str, dict[str, Any]] | None = None,
) -> str:
    tmux_status_map = dict(tmux_status_by_env or {})
    analysis_status_map = dict(analysis_status_by_env or {})
    if project_root is not None:
        for env_id in SUPPORTED_ENVS:
            status = dict(tmux_status_map.get(env_id) or {})
            live_analysis_status = dict(analysis_status_map.get(env_id) or {})
            log_path = project_root / "data" / "analysis" / "supervised_consistency" / env_id / "regen-v7.log"
            log_lines: list[str] = []
            if log_path.is_file():
                try:
                    log_lines = log_path.read_text(encoding="utf-8").splitlines()[-120:]
                except OSError:
                    log_lines = []
            output_dir_text = ""
            for line in reversed(log_lines):
                marker = "output_dir="
                if marker not in line:
                    continue
                output_dir_text = str(line.split(marker, 1)[1] or "").strip()
                if output_dir_text:
                    break
            summary_exists = bool(output_dir_text and (Path(output_dir_text) / "summary.json").is_file())
            live_status_text = str(live_analysis_status.get("status") or "").strip().lower()
            is_running = live_status_text in {"queued", "assigned", "starting", "running", "pausing", "resuming", "stopping"}
            status.update(
                {
                    "env_id": env_id,
                    "session_name": str(status.get("session_name") or f"analysis-v7-{env_id}"),
                    "log_path": str(log_path),
                    "log_lines": log_lines,
                    "output_dir": str(live_analysis_status.get("output_dir") or output_dir_text),
                    "summary_exists": summary_exists,
                    "is_running": is_running or (bool(log_lines) and not summary_exists),
                    "status": (
                        live_status_text
                        or ("running" if (bool(log_lines) and not summary_exists) else ("finished" if log_lines else "idle"))
                    ),
                }
            )
            tmux_status_map[env_id] = status
    tmux_running_envs = [
        env_id for env_id in SUPPORTED_ENVS
        if bool((tmux_status_map.get(env_id) or {}).get("is_running"))
    ]
    if tmux_running_envs:
        focus_env = tmux_running_envs[0]
        focus_status = dict(tmux_status_map.get(focus_env) or {})
        visible_logs = [str(item) for item in (focus_status.get("log_lines") or []) if str(item).strip()]
        log_text = "\n".join(visible_logs[-120:]) if visible_logs else "No tmux log yet."
        rows = []
        for env_id in SUPPORTED_ENVS:
            status = dict(tmux_status_map.get(env_id) or {})
            session_name = str(status.get("session_name") or f"analysis-v7-{env_id}").strip()
            status_text = str(status.get("status") or "idle").strip().lower() or "idle"
            is_running = status_text in {"queued", "assigned", "starting", "running", "pausing", "resuming", "stopping", "finished"}
            rows.append(
                f"""
                <article class="required-row{' is-active-state' if is_running else ''}">
                  <div class="required-row-main">
                    <div class="required-row-title">
                      <span class="required-status-dot {'is-covered' if is_running else 'is-missing'}"></span>
                      <strong>{html.escape(env_id.upper())}</strong>
                    </div>
                    <div class="required-row-meta"><code>{html.escape(session_name)}</code></div>
                  </div>
                  <div class="required-status">{_pill(status_text, status_text)}</div>
                </article>
                """
            )
        return f"""
        <section class="panel compact-panel panel-static">
          <div class="card-topline">
            <div>
              <p class="section-kicker">Live progress</p>
              <h2>Current Analysis</h2>
              <p class="metric-subtle">tmux-backed v7 analysis sessions</p>
            </div>
            <div class="stats-inline">
              <span class="mini-chip">Running {len(tmux_running_envs)}</span>
              {_pill("running", "running")}
            </div>
          </div>
          <div class="required-list" style="margin-bottom:0.8rem;">{''.join(rows)}</div>
          <pre class="code-block" style="max-height:12rem;overflow:auto;">{html.escape(log_text)}</pre>
          <script>(() => {{ window.setTimeout(() => window.location.reload(), 1800); }})();</script>
        </section>
        """
    active_jobs = [job for job in jobs if job.status.value in ACTIVE_QUEUE_STATUSES]
    focus_job = active_jobs[0] if active_jobs else (jobs[0] if jobs else None)
    if focus_job is None:
        return """
        <section class="panel compact-panel panel-static">
          <div class="card-topline">
            <div>
              <p class="section-kicker">Live progress</p>
              <h2>Current Analysis</h2>
              <p class="metric-subtle">No local analysis is running.</p>
            </div>
          </div>
        </section>
        """
    is_live = bool(active_jobs)
    visible_jobs = active_jobs[:4] if is_live else ([focus_job] if focus_job is not None else [])
    rows: list[str] = []
    for job in visible_jobs:
        env_id = _job_env_id(job)
        rows.append(
            f"""
            <article class="required-row{' is-active-state' if job.status.value in ACTIVE_QUEUE_STATUSES else ''}">
              <div class="required-row-main">
                <div class="required-row-title">
                  <span class="required-status-dot {_required_ablation_cell_visual(job)}"></span>
                  <strong>{html.escape(env_id.upper())}</strong>
                </div>
                <div class="required-row-meta"><code>{html.escape(job.job_id[:8])}</code> · {html.escape(_queue_job_title(job))}</div>
                {_analysis_compact_progress(env_id=env_id, jobs=[job], results=[], logs_by_job_id=logs_by_job_id)}
              </div>
              <div class="required-status">{_pill(job.status.value, job.status.value)}</div>
            </article>
            """
        )
    raw_lines = (
        list((logs_by_job_id or {}).get(focus_job.job_id) or [])
        if isinstance(logs_by_job_id, dict)
        else []
    )
    visible_logs = list(reversed([str(item) for item in raw_lines if str(item).strip()][-60:])) if raw_lines else []
    log_text = "\n".join(visible_logs) if visible_logs else "No log yet."
    panel_title = "Current Analysis" if is_live else "Latest Analysis"
    running_chip = f"Running {len(active_jobs)}" if is_live else "No active runs"
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Live progress</p>
          <h2>{panel_title}</h2>
          <p class="metric-subtle"><code>{html.escape(focus_job.job_id[:8])}</code> · {html.escape(_job_env_id(focus_job).upper())}</p>
        </div>
        <div class="stats-inline">
          <span class="mini-chip">{running_chip}</span>
          {_pill(focus_job.status.value, focus_job.status.value)}
        </div>
      </div>
      <div class="required-list" style="margin-bottom:0.8rem;">{''.join(rows)}</div>
      <pre class="code-block" style="max-height:12rem;overflow:auto;">{html.escape(log_text)}</pre>
      {(
          '<script>(() => { const refresh = () => window.setTimeout(() => window.location.reload(), 1500); window.setTimeout(refresh, 1400); })();</script>'
          if is_live
          else ''
      )}
    </section>
    """


def _render_analysis_activity_panel(
    jobs: list[JobRecord],
    *,
    logs_by_job_id: dict[str, list[str]] | None = None,
) -> str:
    if not jobs:
        return '<section class="panel compact-panel"><h2>Activity</h2><p class="empty">No analysis jobs yet.</p></section>'
    active_statuses = {"assigned", "starting", "running", "queued", "paused", "stopping"}
    interesting = [job for job in jobs if job.status.value in active_statuses][:10]
    if not interesting:
        interesting = jobs[:6]
    rows: list[str] = []
    for job in interesting:
        status = job.status.value
        message = str(job.message or "").strip()
        show_message = bool(message and status in {"failed", "assigned", "paused", "queued", "cancelled"})
        log_lines = (
            list(logs_by_job_id.get(job.job_id) or [])
            if isinstance(logs_by_job_id, dict)
            else None
        )
        progress_html = _analysis_activity_progress_html(job, log_lines=log_lines)
        rows.append(
            f"""
            <article class="analysis-activity-row">
              <div class="analysis-activity-head">
                <div class="analysis-activity-main">
                  <strong><a href="/jobs/{html.escape(job.job_id)}"><code>{html.escape(job.job_id[:8])}</code></a> · {html.escape(_job_env_id(job).upper())}</strong>
                  <div class="analysis-activity-meta">{html.escape(_queue_job_title(job))} · {_display_time(job.created_at)}</div>
                </div>
                {_pill(status, status)}
              </div>
              {progress_html}
              {f'<p class="analysis-activity-message">{html.escape(message)}</p>' if show_message else ''}
            </article>
            """
        )
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div><p class="section-kicker">Recent jobs</p><h2>Activity</h2></div>
      </div>
      <div class="analysis-activity-list">{''.join(rows)}</div>
    </section>
    """


def _analysis_activity_progress_html(
    job: JobRecord,
    *,
    log_lines: list[str] | None = None,
) -> str:
    active_statuses = ACTIVE_QUEUE_STATUSES | {"resuming", "pausing"}
    status = job.status.value
    if status not in active_statuses:
        return ""
    progress = _job_progress_payload(job)
    if not progress:
        return (
            '<div class="analysis-progress">'
            f'<div class="analysis-progress-meta"><span>{html.escape(status)}</span><span>Starting…</span></div>'
            '<div class="analysis-progress-track"><span class="analysis-progress-fill is-indeterminate"></span></div>'
            "</div>"
        )

    phase = _sanitize_display_text(str(progress.get("phase") or "").strip())
    summary = _sanitize_display_text(str(progress.get("summary") or "").strip())
    if summary:
        summary = summary.replace("across ? checkpoints", "across window bins")
        summary = summary.replace("? checkpoints", "window bins")
    fraction_value = progress.get("fraction")
    current: int | None = None
    total: int | None = None

    for key in ("current", "completed", "completed_count"):
        try:
            if progress.get(key) is not None:
                current = int(progress.get(key))
                break
        except (TypeError, ValueError):
            continue
    for key in ("total", "total_count"):
        try:
            if progress.get(key) is not None:
                total = int(progress.get(key))
                break
        except (TypeError, ValueError):
            continue

    fraction: float | None = None
    try:
        if fraction_value is not None:
            fraction = float(fraction_value)
    except (TypeError, ValueError):
        fraction = None
    inferred = _analysis_progress_from_recent_logs(job, log_lines=log_lines)
    if str(job.spec.action or "").strip() == "analyze-supervised-consistency":
        if current is None and isinstance(inferred.get("current"), int):
            current = int(inferred["current"])
        if total is None and isinstance(inferred.get("total"), int):
            total = int(inferred["total"])
        if current is not None and total is not None and total > 0:
            inferred_fraction = current / total
            if fraction is None or inferred_fraction > fraction:
                fraction = inferred_fraction
        inferred_phase = str(inferred.get("phase") or "").strip()
        if inferred_phase and (not phase or phase.lower() == "consistency analysis"):
            phase = _sanitize_display_text(inferred_phase)
        inferred_summary = str(inferred.get("summary") or "").strip()
        if inferred_summary and (not summary or "?" in summary):
            summary = _sanitize_display_text(inferred_summary)
    if fraction is None and current is not None and total is not None and total > 0:
        fraction = current / total

    if fraction is not None:
        bounded = max(0.0, min(1.0, fraction))
        percent = int(round(bounded * 100))
        left = phase or status
        right = f"{current}/{total}" if current is not None and total is not None and total > 0 else f"{percent}%"
        estimate_note = _analysis_rate_eta_note(job, current=current, total=total)
        note_parts: list[str] = []
        if summary:
            note_parts.append(f'<p class="analysis-progress-note">{html.escape(summary)}</p>')
        if estimate_note:
            note_parts.append(f'<p class="analysis-progress-note">{html.escape(estimate_note)}</p>')
        note_html = "".join(note_parts)
        return (
            '<div class="analysis-progress">'
            f'<div class="analysis-progress-meta"><span>{html.escape(left)}</span><span>{html.escape(right)}</span></div>'
            f'<div class="analysis-progress-track"><span class="analysis-progress-fill" style="width: {percent}%"></span></div>'
            f"{note_html}"
            "</div>"
        )

    left = phase or status
    right = "Running…"
    estimate_note = _analysis_rate_eta_note(job, current=current, total=total)
    note_parts = []
    if summary:
        note_parts.append(f'<p class="analysis-progress-note">{html.escape(summary)}</p>')
    if estimate_note:
        note_parts.append(f'<p class="analysis-progress-note">{html.escape(estimate_note)}</p>')
    note_html = "".join(note_parts)
    return (
        '<div class="analysis-progress">'
        f'<div class="analysis-progress-meta"><span>{html.escape(left)}</span><span>{html.escape(right)}</span></div>'
        '<div class="analysis-progress-track"><span class="analysis-progress-fill is-indeterminate"></span></div>'
        f"{note_html}"
        "</div>"
    )


def _analysis_rate_eta_note(
    job: JobRecord,
    *,
    current: int | None,
    total: int | None,
) -> str:
    if current is None or total is None or total <= 0 or current <= 0:
        return ""
    started = job.started_at or job.created_at
    if started is None:
        return ""
    now = job.updated_at or datetime.now(started.tzinfo)
    elapsed_seconds = (now - started).total_seconds()
    if elapsed_seconds <= 0:
        return ""
    rate_per_minute = current / (elapsed_seconds / 60.0)
    if rate_per_minute <= 0:
        return ""
    remaining = max(0, total - current)
    eta_text = "soon"
    if remaining > 0:
        eta_seconds = remaining / rate_per_minute * 60.0
        eta_dt = now + timedelta(seconds=eta_seconds)
        eta_text = _display_time(eta_dt)
    return f"{rate_per_minute:.1f} repeats/min · ETA {eta_text}"


def render_analysis_home_page(
    *,
    jobs: list[JobRecord],
    results_by_env: dict[str, list[SupervisedConsistencyResult]],
    project_root: Path,
    logs_by_job_id: dict[str, list[str]] | None = None,
    tmux_status_by_env: dict[str, dict[str, Any]] | None = None,
    analysis_status_by_env: dict[str, dict[str, Any]] | None = None,
) -> str:
    analysis_jobs = [
        job
        for job in jobs
        if str(job.spec.action or "").strip() == "analyze-supervised-consistency"
    ]
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Home</a> / analysis</p>
      <h1>Analysis</h1>
      <p>Supervised-only convergence curves by candidate-percentile bin. This overview shows the pattern-based similarity to the final bin for each environment on one shared chart.</p>
    </section>
    <section class="hero-stats">
      <div class="stat-chip"><span class="stat-label">Analysis jobs</span><strong class="stat-value">{len(analysis_jobs)}</strong></div>
      <div class="stat-chip"><span class="stat-label">Running</span><strong class="stat-value">{sum(1 for job in analysis_jobs if job.status.value in ACTIVE_QUEUE_STATUSES)}</strong></div>
      <div class="stat-chip"><span class="stat-label">Completed</span><strong class="stat-value">{sum(1 for job in analysis_jobs if job.status.value == "succeeded")}</strong></div>
      <div class="stat-chip"><span class="stat-label">Environments</span><strong class="stat-value">{len(SUPPORTED_ENVS)}</strong></div>
    </section>
    {_render_analysis_home_pattern_similarity_panel(results_by_env=results_by_env)}
    {_render_analysis_home_live_panel(
        jobs=analysis_jobs,
        logs_by_job_id=logs_by_job_id,
        tmux_status_by_env=tmux_status_by_env,
        project_root=project_root,
        analysis_status_by_env=analysis_status_by_env,
    )}
    <section class="required-grid">
      {''.join(
          _analysis_env_dashboard_card(
              env_id=env_id,
              jobs=analysis_jobs,
              env_results=list(results_by_env.get(env_id) or []),
              env_status=dict((analysis_status_by_env or {}).get(env_id) or {}),
          )
          for env_id in SUPPORTED_ENVS
      )}
    </section>
    {_analysis_plot_modal_markup()}
    """
    return _shell("Analysis | Pattern Server", active_nav="analysis", body_html=body)


def render_analysis_environment_page(
    *,
    env_id: str,
    jobs: list[JobRecord],
    project_root: Path,
    env_results: list[SupervisedConsistencyResult],
    logs_by_job_id: dict[str, list[str]] | None = None,
    tmux_status: dict[str, Any] | None = None,
) -> str:
    env_jobs = _analysis_jobs_for_env(jobs, env_id=env_id)
    tmux_payload = dict(tmux_status or {})
    log_path = project_root / "data" / "analysis" / "supervised_consistency" / env_id / "regen-v7.log"
    log_lines: list[str] = []
    if log_path.is_file():
        try:
            log_lines = log_path.read_text(encoding="utf-8").splitlines()[-120:]
        except OSError:
            log_lines = []
    output_dir_text = ""
    for line in reversed(log_lines):
        marker = "output_dir="
        if marker not in line:
            continue
        output_dir_text = str(line.split(marker, 1)[1] or "").strip()
        if output_dir_text:
            break
    summary_exists = bool(output_dir_text and (Path(output_dir_text) / "summary.json").is_file())
    tmux_payload.update(
        {
            "env_id": env_id,
            "session_name": str(tmux_payload.get("session_name") or f"analysis-v7-{env_id}"),
            "log_path": str(log_path),
            "log_lines": log_lines,
            "output_dir": output_dir_text,
            "summary_exists": summary_exists,
            "is_running": bool(log_lines) and not summary_exists,
            "status": "running" if (bool(log_lines) and not summary_exists) else ("finished" if log_lines else "idle"),
        }
    )
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Home</a> / <a href="/analysis">analysis</a> / {html.escape(env_id)}</p>
      <h1>{html.escape(env_id.upper())} Analysis</h1>
    </section>
    <section class="hero-stats">
      <div class="stat-chip"><span class="stat-label">Analysis jobs</span><strong class="stat-value">{len(env_jobs)}</strong></div>
      <div class="stat-chip"><span class="stat-label">Running</span><strong class="stat-value">{sum(1 for job in env_jobs if job.status.value in ACTIVE_QUEUE_STATUSES)}</strong></div>
      <div class="stat-chip"><span class="stat-label">Completed</span><strong class="stat-value">{sum(1 for job in env_jobs if job.status.value == "succeeded")}</strong></div>
      <div class="stat-chip"><span class="stat-label">Results</span><strong class="stat-value">{len(env_results)}</strong></div>
    </section>
    {_render_analysis_environment_convergence_panel(env_id=env_id, env_results=env_results)}
    {_render_analysis_env_jobs_panel(env_id=env_id, jobs=env_jobs, project_root=project_root, logs_by_job_id=logs_by_job_id)}
    {_render_analysis_live_terminal_panel(
        env_id=env_id,
        jobs=env_jobs,
        logs_by_job_id=logs_by_job_id,
        tmux_status=tmux_payload,
    )}
    {_render_analysis_env_results_list_panel(env_id=env_id, results=env_results)}
    {_analysis_plot_modal_markup()}
    """
    return _shell(f"{env_id.upper()} Analysis | Pattern Server", active_nav="analysis", body_html=body)


def render_supervised_consistency_result_page(
    *,
    result: SupervisedConsistencyResult,
    project_root: Path,
) -> str:
    payload = result.payload
    window_summaries = payload.get("window_summaries") if isinstance(payload.get("window_summaries"), list) else []
    checkpoint_summaries = payload.get("checkpoint_summaries") if isinstance(payload.get("checkpoint_summaries"), list) else []
    summary_rows = window_summaries if window_summaries else checkpoint_summaries
    final_summary = summary_rows[-1] if summary_rows else {}
    final_similarity_all_labels = (
        float(final_summary.get("mean_similarity_all_labels") or final_summary.get("mean_similarity") or 0.0)
        if isinstance(final_summary, dict)
        else 0.0
    )
    final_similarity_per_label = (
        float(final_summary.get("mean_similarity_per_label") or 0.0) if isinstance(final_summary, dict) else 0.0
    )
    bin_labels = [str(item) for item in (payload.get("bin_labels") or []) if str(item).strip()]
    mode_is_bins = bool(bin_labels)
    run_candidate_counts = (
        payload.get("run_candidate_counts")
        if isinstance(payload.get("run_candidate_counts"), dict)
        else payload.get("run_candidate_totals")
        if isinstance(payload.get("run_candidate_totals"), dict)
        else {}
    )
    sampling_repeats = int(payload.get("sampling_repeats") or 0)
    trace_count = int(payload.get("trace_count") or 0)
    trace_scope = str(payload.get("trace_scope") or "").strip().lower()
    if trace_count > 1 or trace_scope == "test_set":
        trace_caption = f"{trace_count or 'many'} test traces"
    else:
        trace_caption = html.escape(Path(str(payload.get("trace_path") or "trace")).name)
    checkpoint_values = payload.get("checkpoints") or []
    parsed_checkpoints: list[int] = []
    for value in checkpoint_values:
        try:
            parsed_value = int(value)
        except (TypeError, ValueError):
            continue
        if parsed_value > 0:
            parsed_checkpoints.append(parsed_value)
    checkpoint_min = min(parsed_checkpoints, default=0)
    checkpoint_max = max(parsed_checkpoints, default=0)
    checkpoint_span = f"{checkpoint_min}–{checkpoint_max}" if checkpoint_min and checkpoint_max else "—"
    bin_count = int(payload.get("window_count") or len(bin_labels) or 0)

    if mode_is_bins:
        coverage_label = f"{bin_count} bins" if bin_count > 0 else "—"
        coverage_detail = ", ".join(bin_labels) or "—"
        sample_label = "Samples / bin"
        axis_title = "Windows"
    else:
        coverage_label = str(len(parsed_checkpoints))
        coverage_detail = checkpoint_span
        sample_label = "Samples / checkpoint"
        axis_title = "Checkpoints"

    summary_cards = f"""
    <section class="panel compact-panel panel-static">
      <div class="analysis-summary-grid">
        <div class="analysis-overview-stack">
          <div class="analysis-key-grid">
            {_key_value_card("Environment", str(payload.get("env_id") or "—").upper())}
            {_key_value_card("Runs", len(payload.get("run_names") or []))}
            {_key_value_card(axis_title, coverage_label)}
            {_key_value_card("Range", coverage_detail)}
            {_key_value_card(sample_label, sampling_repeats or "—")}
            {_key_value_card("Final similarity (all labels)", f"{final_similarity_all_labels:.3f}")}
            {_key_value_card("Final similarity (per label)", f"{final_similarity_per_label:.3f}")}
            {_key_value_card("Trace", payload.get("trace_id") or "—")}
          </div>
          <div class="analysis-summary-links">
            <a class="ghost-link" href="{_analysis_asset_href(result.summary_path)}">Summary JSON</a>
            <a class="ghost-link" href="{_analysis_asset_href(result.result_dir / 'matrices.json')}">Matrices</a>
          </div>
        </div>
        <div class="analysis-run-chip-list">
          {''.join(
              f'<div class="analysis-run-chip"><strong>{html.escape(str(run_name))}</strong><span>{int(run_candidate_counts.get(str(run_name), 0))} candidates</span></div>'
              for run_name in (payload.get("run_names") or [])
          )}
        </div>
      </div>
    </section>
    """
    similarity_plot = Path(str(payload.get("similarity_all_labels_plot_path") or payload.get("similarity_plot_path") or "")).resolve() if str(payload.get("similarity_all_labels_plot_path") or payload.get("similarity_plot_path") or "").strip() else None
    similarity_per_label_plot = Path(str(payload.get("similarity_per_label_plot_path") or "")).resolve() if str(payload.get("similarity_per_label_plot_path") or "").strip() else None
    embedding_plot = Path(str(payload.get("embedding_plot_path") or "")).resolve() if str(payload.get("embedding_plot_path") or "").strip() else None
    average_grid_plot = Path(str(payload.get("average_distance_grid_all_labels_path") or payload.get("average_distance_grid_path") or "")).resolve() if str(payload.get("average_distance_grid_all_labels_path") or payload.get("average_distance_grid_path") or "").strip() else None
    average_grid_std_plot = Path(str(payload.get("average_distance_grid_std_all_labels_path") or payload.get("average_distance_grid_std_path") or "")).resolve() if str(payload.get("average_distance_grid_std_all_labels_path") or payload.get("average_distance_grid_std_path") or "").strip() else None
    average_grid_per_label_plot = Path(str(payload.get("average_distance_grid_per_label_path") or "")).resolve() if str(payload.get("average_distance_grid_per_label_path") or "").strip() else None
    average_grid_std_per_label_plot = Path(str(payload.get("average_distance_grid_std_per_label_path") or "")).resolve() if str(payload.get("average_distance_grid_std_per_label_path") or "").strip() else None
    two_run_grid_plot = Path(str(payload.get("two_run_distance_grid_path") or "")).resolve() if str(payload.get("two_run_distance_grid_path") or "").strip() else None
    two_run_grid_per_label_plot = Path(str(payload.get("two_run_distance_grid_per_label_path") or "")).resolve() if str(payload.get("two_run_distance_grid_per_label_path") or "").strip() else None
    pairwise_histogram_plot = Path(str(payload.get("pairwise_distance_histogram_path") or "")).resolve() if str(payload.get("pairwise_distance_histogram_path") or "").strip() else None
    pairwise_histogram_per_label_plot = Path(str(payload.get("pairwise_distance_histogram_per_label_path") or "")).resolve() if str(payload.get("pairwise_distance_histogram_per_label_path") or "").strip() else None
    run_pair_matrix_plot = Path(str(payload.get("run_pair_distance_matrix_path") or "")).resolve() if str(payload.get("run_pair_distance_matrix_path") or "").strip() else None
    run_pair_matrix_per_label_plot = Path(str(payload.get("run_pair_distance_matrix_per_label_path") or "")).resolve() if str(payload.get("run_pair_distance_matrix_per_label_path") or "").strip() else None
    two_run_label = str(payload.get("two_run_distance_grid_pair") or "Two-run distance grid").strip() or "Two-run distance grid"

    if mode_is_bins:
        checkpoint_cards_html = "".join(
            (
                f'<div class="key-value-card"><span class="key-value-label">{html.escape(str(item.get("window_label") or ("bin " + str(int(item.get("window_index") or 0) + 1))))}</span>'
                f'<span class="key-value-value">all {float(item.get("mean_similarity_all_labels") or item.get("mean_similarity") or 0.0):.3f}</span>'
                f'<span class="metric-subtle">per-label {float(item.get("mean_similarity_per_label") or 0.0):.3f}</span>'
                f'<span class="metric-subtle">gap {float(item.get("distance_gap_all_minus_per_label") or 0.0):+.3f} dist</span></div>'
            )
            for item in summary_rows
            if isinstance(item, dict)
        )
        body = f"""
        <section class="hero">
          <p class="breadcrumb"><a href="/analysis">Analysis</a> / {html.escape(result.analysis_run_id)}</p>
          <h1>{html.escape(result.analysis_run_id)}</h1>
          <p class="metric-subtle">{trace_caption}</p>
        </section>
        {summary_cards}
        <div class="analysis-overview-grid">
          {_analysis_plot_card("Similarity by bin (all labels)", similarity_plot)}
          {_analysis_plot_card("Similarity by bin (per-label mean)", similarity_per_label_plot)}
        </div>
        <div class="analysis-overview-grid">
          {_analysis_plot_card("Snapshot embedding", embedding_plot)}
          <div class="analysis-overview-stack">
            <section class="panel compact-panel panel-static">
              <div class="card-topline">
                <div><h3>Bin summaries</h3><p class="metric-subtle">{html.escape(', '.join(bin_labels) or '—')}</p></div>
              </div>
              <div class="analysis-checkpoint-grid">{checkpoint_cards_html}</div>
            </section>
          </div>
        </div>
        <div class="analysis-overview-grid">
          {_analysis_plot_card(f"{two_run_label} (all labels)", two_run_grid_plot)}
          {_analysis_plot_card(f"{two_run_label} (per-label mean)", two_run_grid_per_label_plot)}
        </div>
        <div class="analysis-overview-grid">
          {_analysis_plot_card("Average distance grid (all pairs, all labels)", average_grid_plot)}
          {_analysis_plot_card("Average distance grid (all pairs, per-label mean)", average_grid_per_label_plot)}
        </div>
        <div class="analysis-overview-grid">
          {_analysis_plot_card("Distance grid std-dev (all labels)", average_grid_std_plot)}
          <div class="analysis-overview-stack">
            {_analysis_plot_card("Distance grid std-dev (per-label mean)", average_grid_std_per_label_plot)}
            {_analysis_plot_card("Pairwise distance histogram (all labels)", pairwise_histogram_plot)}
            {_analysis_plot_card("Pairwise distance histogram (per-label mean)", pairwise_histogram_per_label_plot)}
            {_analysis_plot_card("Run-pair distance matrix (all labels)", run_pair_matrix_plot)}
            {_analysis_plot_card("Run-pair distance matrix (per-label mean)", run_pair_matrix_per_label_plot)}
          </div>
        </div>
        <section class="panel compact-panel">
          <div class="card-topline">
            <div><h2>Summary JSON</h2></div>
            <a class="ghost-link" href="{_analysis_asset_href(result.summary_path)}">Open</a>
          </div>
          <details class="data-details">
            <summary>Raw payload</summary>
            <pre class="code-block metadata-block">{html.escape(json.dumps(payload, indent=2, sort_keys=True))}</pre>
          </details>
        </section>
        {_analysis_plot_modal_markup()}
        """
        return _shell(f"{result.analysis_run_id} | Pattern Server", active_nav="analysis", body_html=body)

    checkpoint_sections: list[str] = []
    for item in checkpoint_summaries:
        if not isinstance(item, dict):
            continue
        budget = int(item.get("budget") or 0)
        heatmap_path = Path(str(item.get("variance_heatmap_path") or "")).resolve() if str(item.get("variance_heatmap_path") or "").strip() else None
        histogram_path = Path(str(item.get("pairwise_distance_histogram_path") or "")).resolve() if str(item.get("pairwise_distance_histogram_path") or "").strip() else None
        matrix_path = Path(str(item.get("pairwise_distance_matrix_path") or "")).resolve() if str(item.get("pairwise_distance_matrix_path") or "").strip() else None
        checkpoint_sections.append(
            f"""
            <details class="data-details analysis-checkpoint-panel"{" open" if item is checkpoint_summaries[-1] else ""}>
              <summary>
                <span class="analysis-checkpoint-head"><strong>Budget {budget}</strong><span class="metric-subtle">{float(item.get("mean_similarity") or 0.0):.3f} similarity</span></span>
                <span class="stats-inline"><span class="mini-chip">distance {float(item.get("mean_distance") or 0.0):.3f}</span><span class="mini-chip">{int(item.get("snapshot_count") or 0)} snapshots</span></span>
              </summary>
              <div class="content-grid">
                {_analysis_plot_card("Variance heatmap", heatmap_path)}
                {_analysis_plot_card("Distance histogram", histogram_path)}
                {_analysis_plot_card("Distance matrix", matrix_path)}
              </div>
            </details>
            """
        )
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/analysis">Analysis</a> / {html.escape(result.analysis_run_id)}</p>
      <h1>{html.escape(result.analysis_run_id)}</h1>
      <p class="metric-subtle">{trace_caption}</p>
    </section>
    {summary_cards}
    <div class="analysis-overview-grid">
      {_analysis_plot_card("Similarity by budget", similarity_plot)}
      <div class="analysis-overview-stack">
        {_analysis_plot_card("Snapshot embedding", embedding_plot)}
        <section class="panel compact-panel panel-static">
          <div class="card-topline">
            <div><h3>Checkpoints</h3><p class="metric-subtle">{', '.join(str(item) for item in parsed_checkpoints) or '—'}</p></div>
          </div>
          <div class="analysis-checkpoint-grid">
            {''.join(
                f'<div class="key-value-card"><span class="key-value-label">Budget {int(item.get("budget") or 0)}</span><span class="key-value-value">{float(item.get("mean_similarity") or 0.0):.3f}</span><span class="metric-subtle">distance {float(item.get("mean_distance") or 0.0):.3f}</span></div>'
                for item in checkpoint_summaries
                if isinstance(item, dict)
            )}
          </div>
        </section>
      </div>
    </div>
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div><h2>Checkpoint plots</h2></div>
      </div>
      <div class="analysis-checkpoint-strip">
        {''.join(checkpoint_sections)}
      </div>
    </section>
    <section class="panel compact-panel">
      <div class="card-topline">
        <div><h2>Summary JSON</h2></div>
        <a class="ghost-link" href="{_analysis_asset_href(result.summary_path)}">Open</a>
      </div>
      <details class="data-details">
        <summary>Raw payload</summary>
        <pre class="code-block metadata-block">{html.escape(json.dumps(payload, indent=2, sort_keys=True))}</pre>
      </details>
    </section>
    {_analysis_plot_modal_markup()}
    """
    return _shell(f"{result.analysis_run_id} | Pattern Server", active_nav="analysis", body_html=body)


def _render_ablation_launcher_panel(*, project_root: Path, clients: list[ClientRecord]) -> str:
    return ""


def _required_ablation_kind_for_family(family: str) -> str | None:
    mapping = {
        "Supervised library": "supervised",
        "Unsupervised library": "unsupervised",
    }
    return mapping.get(str(family or "").strip())


def _job_pattern_run_dir(job: JobRecord | None) -> Path | None:
    if job is None:
        return None
    metadata = dict(job.spec.metadata or {})
    request_payload = metadata.get("request")
    request_dict = request_payload if isinstance(request_payload, dict) else {}
    raw = str(metadata.get("pattern_run_dir") or request_dict.get("pattern_run_dir") or "").strip()
    return Path(raw) if raw else None


def _required_ablation_cell_visual(job: JobRecord | None) -> str:
    if job is None:
        return "is-missing"
    if job.status.value in ACTIVE_QUEUE_STATUSES:
        return "is-active"
    if job.status.value == "succeeded":
        return "is-covered"
    if job.status.value == "paused":
        return "is-blocked"
    if job.status.value in FAILED_QUEUE_STATUSES:
        return "is-failed"
    return "is-covered"


def _required_ablation_controls(job: JobRecord | None) -> str:
    if job is None:
        return ""
    next_input = '<input type="hidden" name="next" value="/ablations">'
    href = _library_ablation_job_href(job)
    controls = [f'<a class="ghost-link" href="{href}">Job</a>' if href is not None else ""]
    if job.status.value == "paused":
        controls.append(
            f'<form method="post" action="/ui/jobs/{html.escape(job.job_id)}/resume">{next_input}<button class="secondary-button" type="submit">Resume</button></form>'
        )
    elif job.status.value in ACTIVE_QUEUE_STATUSES | {"queued"}:
        controls.append(
            f'<form method="post" action="/ui/jobs/{html.escape(job.job_id)}/pause">{next_input}<button class="ghost-link" type="submit">Pause</button></form>'
        )
    elif job.status.value in FAILED_QUEUE_STATUSES | {"stopped", "cancelled", "lost", "retryable_failure"}:
        controls.append(
            f'<form method="post" action="/ui/jobs/{html.escape(job.job_id)}/requeue">{next_input}<button class="secondary-button" type="submit">Requeue</button></form>'
        )
    return "".join(control for control in controls if control)


def _render_required_ablation_progress(detail: LibraryAblationDetail | None, job: JobRecord | None) -> str:
    if detail is None:
        if job is None:
            return '<p class="metric-subtle">No tracked ablation yet.</p>'
        waiting_label = _sanitize_display_text(job.message) or "Awaiting summary"
        return f"""
        <div class="required-run-stats">
          <div class="required-run-stat"><span>Runs</span><strong>—</strong></div>
          <div class="required-run-stat"><span>Cases</span><strong>—</strong></div>
          <div class="required-run-stat"><span>Baseline</span><strong>—</strong></div>
        </div>
        <p class="analysis-progress-note">{html.escape(waiting_label)}</p>
        """
    total_cases = detail.total_case_count
    completed_cases = sum(
        1
        for case in detail.case_results
        if case.planned_seed_count > 0 and case.completed_seed_count == case.planned_seed_count
    )
    total_seeds = detail.total_seed_run_count
    completed_seeds = detail.completed_seed_run_count
    seed_fraction = (completed_seeds / total_seeds) if total_seeds else 0.0
    baseline = detail.baseline_case
    baseline_text = (
        f"{baseline.mean_success_ratio * 100.0:.1f}%"
        if baseline is not None and baseline.mean_success_ratio is not None
        else "—"
    )
    active_cases = sum(
        1
        for case in detail.case_results
        if 0 < case.completed_seed_count < case.planned_seed_count
    )
    waiting_cases = max(0, total_cases - completed_cases - active_cases)
    note_parts: list[str] = []
    if active_cases:
        note_parts.append(f"{active_cases} active")
    if waiting_cases:
        note_parts.append(f"{waiting_cases} waiting")
    if detail.active_seed_run_count > 0:
        note_parts.append(f"{detail.active_seed_run_count} active runs")
    if detail.observed_task_count > 0:
        note_parts.append(f"{detail.observed_task_count} task results")
    if job is not None and job.status.value in FAILED_QUEUE_STATUSES:
        failure_message = _sanitize_display_text(job.message)
        if failure_message:
            note_parts.append(failure_message)
    note_html = (
        f'<p class="analysis-progress-note">{" · ".join(html.escape(part) for part in note_parts)}</p>'
        if note_parts
        else ""
    )
    return f"""
    <div class="required-run-stats">
      <div class="required-run-stat"><span>Runs</span><strong>{completed_seeds}/{total_seeds}</strong></div>
      <div class="required-run-stat"><span>Cases</span><strong>{completed_cases}/{total_cases}</strong></div>
      <div class="required-run-stat"><span>Baseline</span><strong>{html.escape(baseline_text)}</strong></div>
    </div>
    {note_html}
    """


def _render_ablation_report_links(detail: LibraryAblationDetail | None) -> str:
    if detail is None:
        return """
        <div class="analysis-summary-links">
          <span class="metric-subtle">Reports will appear here once the ablation writes manifest and evaluation outputs.</span>
        </div>
        """
    links = [
        f'<a class="ghost-link" href="{_analysis_asset_href(detail.root / "manifest.json")}">Manifest</a>',
        f'<a class="ghost-link" href="{_analysis_asset_href(detail.root / "request.json")}">Request</a>',
    ]
    case_roots = [seed.root for case in detail.case_results for seed in case.seed_results]
    if case_roots:
        links.append(f'<a class="ghost-link" href="{_analysis_asset_href(case_roots[0])}">Sample run dir</a>')
    return f'<div class="analysis-summary-links">{"".join(links)}</div>'


def _render_ablation_case_breakdown(detail: LibraryAblationDetail | None) -> str:
    if detail is None:
        return """
        <section class="panel compact-panel panel-static">
          <div class="card-topline">
            <div>
              <p class="section-kicker">Detailed reports</p>
              <h2>Cases and Runs</h2>
              <p class="metric-subtle">Case-level results have not been written yet for this ablation.</p>
            </div>
          </div>
        </section>
        """
    rows: list[str] = []
    for case in sorted(detail.case_results, key=lambda item: (item.label_count, item.label)):
        mean_text = (
            f"{case.mean_success_ratio * 100.0:.1f}% ± {case.success_ratio_std * 100.0:.1f}%"
            if case.mean_success_ratio is not None
            else "—"
        )
        seeds_text = ", ".join(
            f"{seed.seed}:{seed.status.lower()}"
            for seed in sorted(case.seed_results, key=lambda item: item.seed)
        ) or "No runs"
        rows.append(
            f"""
            <tr>
              <td>{html.escape(case.label)}</td>
              <td>{case.label_count}</td>
              <td>{case.completed_seed_count}/{case.planned_seed_count}</td>
              <td>{html.escape(mean_text)}</td>
              <td>{html.escape(seeds_text)}</td>
            </tr>
            """
        )
    if not rows:
        rows.append('<tr><td colspan="5" class="empty">No ablation cases are available yet.</td></tr>')
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Detailed reports</p>
          <h2>Cases and Runs</h2>
          <p class="metric-subtle">Per-case completion and run status for this ablation run.</p>
        </div>
      </div>
      <div class="table-scroll compact-table">
        <table class="data-table">
          <thead><tr><th>Case</th><th>Labels</th><th>Runs</th><th>Mean success ± std</th><th>Run states</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </section>
    """


def _render_required_ablation_catalog_panel(*, jobs: list[JobRecord], project_root: Path) -> str:
    catalog = _required_library_ablation_job_catalog_for_project(project_root)
    env_ids = [
        env_id
        for env_id in SUPPORTED_ENVS
        if any(
            _canonical_ablation_source_root(project_root=project_root, env_id=env_id, kind=kind) is not None
            for kind in ("supervised", "unsupervised")
        )
        or any(key_env == env_id for key_env, _ in catalog)
    ]
    if not env_ids:
        return ""
    job_lookup = {job.job_id: job for job in jobs}
    env_cards: list[str] = []
    for env_id in env_ids:
        rows: list[str] = []
        tracked_count = 0
        for family in REQUIRED_LIBRARY_ABLATION_JOB_FAMILIES:
            workflow_kind = _required_ablation_kind_for_family(family)
            selected = job_lookup.get(catalog.get((env_id, family), ""))
            detail = (
                load_library_ablation_job_detail(selected, project_root=project_root)
                if selected is not None
                else None
            )
            if selected is not None:
                tracked_count += 1
            status_copy = _library_ablation_status_detail(detail) if selected is not None else "No tracked ablation yet."
            selected_href = _library_ablation_job_href(selected)
            status_label = selected.status.value if selected is not None else "missing"
            visual_class = _required_ablation_cell_visual(selected)
            row_class = "required-row"
            if selected is not None and selected.status.value in ACTIVE_QUEUE_STATUSES:
                row_class += " is-active-state"
            elif selected is not None and selected.status.value == "succeeded":
                row_class += " is-succeeded"
            elif selected is not None and selected.status.value in FAILED_QUEUE_STATUSES:
                row_class += " is-terminal-failed"
            source_available = (
                workflow_kind is not None
                and _canonical_ablation_source_root(project_root=project_root, env_id=env_id, kind=workflow_kind) is not None
            )
            track_form = (
                f"""
                <form method="post" action="/ui/ablations/track" class="ablation-track-form ablation-track-form-compact">
                  <input type="hidden" name="next" value="/ablations">
                  <input type="hidden" name="env_id" value="{html.escape(env_id)}">
                  <input type="hidden" name="family" value="{html.escape(required_family_slug(family))}">
                  <button class="secondary-button" type="submit">{'Replace' if selected is not None else 'Queue'} ablation</button>
                </form>
                """
                if source_available
                else f'<div class="required-row-meta">No canonical {html.escape(str(workflow_kind or "library"))} library available yet.</div>'
            )
            action_link = (
                f'<a class="required-job-link" href="{selected_href}">View</a>'
                if selected_href is not None
                else '<span class="metric-subtle">No job</span>'
            )
            rows.append(
                f"""
                <article class="{row_class}">
                  <div class="required-row-main">
                    <div class="required-row-title">
                      <span class="required-status-dot {visual_class}"></span>
                      {f'<strong><a href="{selected_href}">{html.escape(family)}</a></strong>' if selected_href is not None else f"<strong>{html.escape(family)}</strong>"}
                    </div>
                    <div class="required-row-meta">{html.escape(status_copy)}</div>
                    {_render_required_ablation_progress(detail, selected)}
                    {track_form}
                  </div>
                  <div class="required-status">
                    {_pill(status_label, status_label)}
                    {action_link}
                  </div>
                </article>
                """
            )
        env_cards.append(
            f"""
            <section class="required-card">
              <div class="card-topline">
                <div><h3>{html.escape(env_id.upper())}</h3></div>
                <span class="mini-chip">{tracked_count}/{len(REQUIRED_LIBRARY_ABLATION_JOB_FAMILIES)} tracked</span>
              </div>
              <div class="required-list">{''.join(rows)}</div>
            </section>
            """
        )
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Required Ablations</p>
          <h2>Tracked Runs</h2>
          <p class="metric-subtle">Compact tracked ablations by environment. Click into any ablation to inspect seed progress and partial results.</p>
        </div>
      </div>
      <div class="required-grid">{''.join(env_cards)}</div>
    </section>
    """


def _render_required_ablation_matrix(*, jobs: list[JobRecord], project_root: Path) -> str:
    catalog = _required_library_ablation_job_catalog_for_project(project_root)
    job_lookup = {job.job_id: job for job in jobs}
    header_cells = "".join(
        f"<th>{html.escape(family)}</th>" for family in REQUIRED_LIBRARY_ABLATION_JOB_FAMILIES
    )
    body_rows: list[str] = []
    for env_id in SUPPORTED_ENVS:
        cell_html: list[str] = []
        for family in REQUIRED_LIBRARY_ABLATION_JOB_FAMILIES:
            workflow_kind = _required_ablation_kind_for_family(family)
            tracked_job = job_lookup.get(catalog.get((env_id, family), ""))
            detail = (
                load_library_ablation_job_detail(tracked_job, project_root=project_root)
                if tracked_job is not None
                else None
            )
            visual_class = _required_ablation_cell_visual(tracked_job)
            source_available = (
                workflow_kind is not None
                and _canonical_ablation_source_root(project_root=project_root, env_id=env_id, kind=workflow_kind) is not None
            )
            track_form = (
                f"""
                <form method="post" action="/ui/ablations/track" class="ablation-track-form">
                  <input type="hidden" name="next" value="/ablations">
                  <input type="hidden" name="env_id" value="{html.escape(env_id)}">
                  <input type="hidden" name="family" value="{html.escape(required_family_slug(family))}">
                  <button class="secondary-button" type="submit">{'Replace tracked job' if tracked_job is not None else 'Queue tracked job'}</button>
                </form>
                """
                if source_available
                else f'<p class="metric-subtle">No canonical {html.escape(str(workflow_kind or "library"))} library available yet.</p>'
            )
            status_label = tracked_job.status.value if tracked_job is not None else "missing"
            status_copy = (
                _library_ablation_status_detail(detail)
                if tracked_job is not None
                else "No tracked job selected."
            )
            cell_html.append(
                f"""
                <td class="ablation-matrix-cell">
                  <section class="ablation-cell-card {visual_class}">
                    <div class="ablation-cell-head">
                      <div>
                        <p class="section-kicker">{html.escape(_library_ablation_kind_label(workflow_kind or family))}</p>
                        <h3>Canonical /library source</h3>
                      </div>
                      {_pill(status_label, status_label)}
                    </div>
                    <p class="ablation-cell-copy">{html.escape(status_copy)}</p>
                    {_render_required_ablation_progress(detail, tracked_job)}
                    <div class="ablation-cell-controls">{_required_ablation_controls(tracked_job)}</div>
                    {track_form}
                  </section>
                </td>
                """
            )
        body_rows.append(
            f"""
            <tr>
              <th class="ablation-matrix-env">{html.escape(env_id.upper())}</th>
              {''.join(cell_html)}
            </tr>
            """
        )
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Required queue</p>
          <h2>Tracked Ablations</h2>
          <p class="metric-subtle">Queue the supervised and unsupervised required ablations per environment using canonical libraries from /library.</p>
        </div>
      </div>
      <div class="ablation-matrix-wrap">
        <table class="data-table ablation-matrix-table">
          <thead>
            <tr>
              <th>Environment</th>
              {header_cells}
            </tr>
          </thead>
          <tbody>
            {''.join(body_rows)}
          </tbody>
        </table>
      </div>
    </section>
    """


def _render_required_ablation_activity(*, jobs: list[JobRecord], project_root: Path) -> str:
    catalog = _required_library_ablation_job_catalog_for_project(project_root)
    tracked_ids = {
        str(job_id)
        for job_id in catalog.values()
        if str(job_id or "").strip()
    }
    tracked_jobs = [
        job
        for job in jobs
        if job.job_id in tracked_ids
    ]
    if not tracked_jobs:
        return ""
    rows: list[str] = []
    for job in sorted(tracked_jobs, key=lambda item: item.updated_at, reverse=True)[:12]:
        rows.append(
            f"""
            <article class="analysis-activity-row">
              <div class="analysis-activity-head">
                <div class="analysis-activity-main">
                  <strong><a href="/jobs/{html.escape(job.job_id)}">{html.escape(_queue_job_title(job))}</a></strong>
                </div>
                {_pill(job.status.value, job.status.value)}
              </div>
              {f'<p class="analysis-activity-message">{html.escape(_sanitize_display_text(job.message))}</p>' if _sanitize_display_text(job.message) else ''}
            </article>
            """
        )
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div><p class="section-kicker">Tracked jobs</p><h2>Recent Activity</h2></div>
      </div>
      <div class="analysis-activity-list">{''.join(rows)}</div>
    </section>
    """


def _library_ablation_kind_from_family(family: str) -> str | None:
    mapping = {
        "Supervised library": "supervised",
        "Unsupervised library": "unsupervised",
    }
    return mapping.get(str(family or "").strip())


def _library_ablation_family_from_kind(kind: str) -> str | None:
    normalized = str(kind or "").strip().lower()
    if normalized == "supervised":
        return "Supervised library"
    if normalized == "unsupervised":
        return "Unsupervised library"
    return None


def _preferred_ablation_job(
    tracked_job: JobRecord | None,
    runs: list[tuple[JobRecord, LibraryAblationDetail]],
) -> JobRecord | None:
    if runs:
        active_statuses = ACTIVE_QUEUE_STATUSES | {"queued"}
        for job, _detail in runs:
            if job.status.value in active_statuses:
                return job
        return runs[0][0]
    return tracked_job


def _ablation_run_status_rank(status: str) -> int:
    normalized = str(status or "").strip().lower()
    if normalized in ACTIVE_QUEUE_STATUSES:
        return 5
    if normalized == "queued":
        return 4
    if normalized == "succeeded":
        return 3
    if normalized == "paused":
        return 2
    if normalized == "failed":
        return 1
    return 0


def _dedupe_ablation_runs(
    runs: list[tuple[JobRecord, LibraryAblationDetail]],
) -> list[tuple[JobRecord, LibraryAblationDetail]]:
    best_by_root: dict[str, tuple[JobRecord, LibraryAblationDetail]] = {}
    for job, detail in runs:
        try:
            key = str(detail.root.resolve())
        except OSError:
            key = str(detail.root)
        current = best_by_root.get(key)
        if current is None:
            best_by_root[key] = (job, detail)
            continue
        current_job, current_detail = current
        candidate_rank = (
            _ablation_run_status_rank(job.status.value),
            detail.completed_seed_run_count,
            len(detail.aggregate_points),
            detail.total_seed_run_count,
            job.updated_at.timestamp(),
            job.created_at.timestamp(),
            job.job_id,
        )
        current_rank = (
            _ablation_run_status_rank(current_job.status.value),
            current_detail.completed_seed_run_count,
            len(current_detail.aggregate_points),
            current_detail.total_seed_run_count,
            current_job.updated_at.timestamp(),
            current_job.created_at.timestamp(),
            current_job.job_id,
        )
        if candidate_rank > current_rank:
            best_by_root[key] = (job, detail)
    deduped = list(best_by_root.values())
    deduped.sort(
        key=lambda item: (
            _ablation_run_status_rank(item[0].status.value),
            item[1].completed_seed_run_count,
            len(item[1].aggregate_points),
            item[0].updated_at.timestamp(),
            item[0].created_at.timestamp(),
            item[0].job_id,
        ),
        reverse=True,
    )
    return deduped


def _ablation_source_root_key(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _canonical_ablation_source_root(*, project_root: Path, env_id: str, kind: str) -> Path | None:
    try:
        return resolve_canonical_library(project_root, env_id, kind).parent
    except Exception:
        return None


def _latest_library_source_for_kind(*, project_root: Path, env_id: str, kind: str) -> LibraryEntry | None:
    libraries = [
        item
        for item in list_environment_libraries(project_root, env_id)
        if str(item.workflow_kind or "").strip().lower() == str(kind or "").strip().lower()
    ]
    return libraries[0] if libraries else None


def _tracked_ablation_job_lookup(
    *,
    jobs: list[JobRecord],
    project_root: Path,
) -> dict[tuple[str, str], JobRecord]:
    catalog = _required_library_ablation_job_catalog_for_project(project_root)
    jobs_by_id = {job.job_id: job for job in jobs}
    lookup: dict[tuple[str, str], JobRecord] = {}
    for key, job_id in catalog.items():
        job = jobs_by_id.get(str(job_id or "").strip())
        if job is not None:
            lookup[key] = job
    return lookup


@lru_cache(maxsize=32)
def _discover_on_disk_ablation_runs(
    *,
    project_root: Path,
    env_id: str,
    kind: str,
    include_all: bool = False,
) -> list[tuple[JobRecord, LibraryAblationDetail]]:
    runs: list[tuple[JobRecord, LibraryAblationDetail]] = []
    seen_eval_roots: set[str] = set()
    candidate_eval_roots: list[Path] = []
    candidate_source_roots: list[Path] = []
    canonical_source_root = _canonical_ablation_source_root(
        project_root=project_root,
        env_id=env_id,
        kind=kind,
    )
    if canonical_source_root is not None:
        candidate_source_roots.append(canonical_source_root)
    for library in list_environment_libraries(project_root, env_id):
        if str(library.workflow_kind or "").strip().lower() != kind:
            continue
        if isinstance(library.root, Path) and library.root.is_dir():
            candidate_source_roots.append(library.root)
    seen_source_roots: set[str] = set()
    for source_root in candidate_source_roots:
        try:
            resolved_source_root = source_root.resolve()
        except OSError:
            resolved_source_root = source_root
        source_key = str(resolved_source_root)
        if source_key in seen_source_roots:
            continue
        seen_source_roots.add(source_key)
        library_evaluations_root = resolved_source_root / "library_evaluations"
        if not library_evaluations_root.is_dir():
            continue
        candidate_eval_roots.extend(
            child for child in library_evaluations_root.iterdir() if child.is_dir()
        )
    candidate_eval_roots = sorted(
        candidate_eval_roots,
        key=lambda path: max(
            path.stat().st_mtime if path.exists() else 0.0,
            (path / "summary.json").stat().st_mtime if (path / "summary.json").is_file() else 0.0,
            (path / "manifest.json").stat().st_mtime if (path / "manifest.json").is_file() else 0.0,
        ),
        reverse=True,
    )
    if not include_all:
        max_roots_raw = os.environ.get("PATTERN_UI_MAX_ABLATION_ROOTS_PER_ENV_KIND", "12")
        try:
            max_roots = max(1, int(max_roots_raw))
        except (TypeError, ValueError):
            max_roots = 12
        candidate_eval_roots = candidate_eval_roots[:max_roots]
    for eval_root in candidate_eval_roots:
        try:
            resolved_eval_root = eval_root.resolve()
        except OSError:
            resolved_eval_root = eval_root
        eval_key = str(resolved_eval_root)
        if eval_key in seen_eval_roots:
            continue
        seen_eval_roots.add(eval_key)
        if not ((resolved_eval_root / "request.json").is_file() and (resolved_eval_root / "manifest.json").is_file()):
            continue
        try:
            request_payload = json.loads((resolved_eval_root / "request.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(request_payload, dict):
            continue
        request_env = str(request_payload.get("env_id") or "").strip().lower()
        if request_env != env_id:
            continue
        raw_source = str(request_payload.get("pattern_run_dir") or "").strip()
        if not raw_source:
            continue
        source_run_dir = Path(raw_source)
        if not source_run_dir.is_absolute():
            source_run_dir = (project_root / source_run_dir).resolve()
        detail = load_library_ablation_detail(
            eval_root=resolved_eval_root,
            env_id=env_id,
            source_run_dir=source_run_dir,
            ablation_kind=kind,
        )
        if detail is None:
            continue
        latest_mtime = max(
            resolved_eval_root.stat().st_mtime,
            (resolved_eval_root / "summary.json").stat().st_mtime if (resolved_eval_root / "summary.json").is_file() else 0.0,
            (resolved_eval_root / "manifest.json").stat().st_mtime if (resolved_eval_root / "manifest.json").is_file() else 0.0,
        )
        timestamp = datetime.fromtimestamp(latest_mtime, tz=ZoneInfo("UTC"))
        synthetic_job = JobRecord(
            job_id=f"ondisk-ablation::{env_id}::{kind}::{resolved_eval_root.name}",
            status=JobStatus.succeeded,
            created_at=timestamp,
            updated_at=timestamp,
            finished_at=timestamp,
            message="Loaded directly from on-disk ablation artifacts",
            spec=JobSpec(
                action="evaluate-library-case",
                argv=[],
                metadata={
                    "env_id": env_id,
                    "library_ablation_kind": kind,
                    "pattern_run_dir": str(source_run_dir),
                    "library_evaluation_run_dir": str(resolved_eval_root),
                },
            ),
        )
        runs.append((synthetic_job, detail))
    return runs


def _ablation_plot_runs_for_env_kind(
    *,
    jobs: list[JobRecord],
    project_root: Path,
    env_id: str,
    kind: str,
    plot_context: dict[tuple[str, str], dict[str, Any]] | None = None,
    include_all: bool = False,
) -> tuple[Path | None, str, JobRecord | None, list[tuple[JobRecord, LibraryAblationDetail]]]:
    if isinstance(plot_context, dict) and not include_all:
        cached_entry = plot_context.get((kind, env_id))
        if isinstance(cached_entry, dict):
            jobs_by_id = {job.job_id: job for job in jobs}
            source_root = cached_entry.get("source_root")
            source_label = str(cached_entry.get("source_label") or "").strip()
            source_display_name = str(cached_entry.get("source_display_name") or "").strip()
            if not source_display_name and isinstance(source_root, Path):
                source_display_name = source_root.name
            tracked_job_id = str(cached_entry.get("tracked_job_id") or "").strip()
            tracked_job = jobs_by_id.get(tracked_job_id) if tracked_job_id else None
            cached_runs = cached_entry.get("runs")
            if isinstance(cached_runs, list):
                resolved_runs = _dedupe_ablation_runs(
                    [
                        (job, detail)
                        for job, detail in cached_runs
                        if isinstance(job, JobRecord) and isinstance(detail, LibraryAblationDetail)
                    ]
                )
                if source_root is None and resolved_runs:
                    source_root = resolved_runs[0][1].source_run_dir
                    if not source_label:
                        source_label = "Ablation source"
                    if not source_display_name:
                        source_display_name = source_root.name
                return (
                    source_root if isinstance(source_root, Path) else None,
                    f"{source_label}\n{source_display_name}".strip(),
                    tracked_job,
                    resolved_runs,
                )
    family = _library_ablation_family_from_kind(kind)
    tracked_lookup = _tracked_ablation_job_lookup(jobs=jobs, project_root=project_root)
    tracked_job = tracked_lookup.get((env_id, family or "")) if family is not None else None
    runs: list[tuple[JobRecord, LibraryAblationDetail]] = []
    for job in jobs:
        if str(job.spec.action or "").strip() != "evaluate-library-case":
            continue
        metadata = dict(job.spec.metadata or {})
        if str(metadata.get("env_id") or "").strip().lower() != env_id:
            continue
        if str(metadata.get("library_ablation_kind") or "").strip().lower() != kind:
            continue
        detail = load_library_ablation_job_detail(job, project_root=project_root)
        if detail is None:
            continue
        runs.append((job, detail))
    runs.extend(
        _discover_on_disk_ablation_runs(
            project_root=project_root,
            env_id=env_id,
            kind=kind,
            include_all=include_all,
        )
    )
    runs.sort(key=lambda item: item[0].updated_at, reverse=True)
    runs = _dedupe_ablation_runs(runs)
    source_root: Path | None = runs[0][1].source_run_dir if runs else None
    source_label = "Ablation source" if source_root is not None else ""
    if source_root is None and tracked_job is not None:
        metadata = dict(tracked_job.spec.metadata or {})
        request_payload = metadata.get("request") if isinstance(metadata.get("request"), dict) else {}
        raw_source_root = str(
            metadata.get("pattern_run_dir")
            or request_payload.get("pattern_run_dir")
            or ""
        ).strip()
        if raw_source_root:
            candidate = Path(raw_source_root)
            source_root = candidate if candidate.is_absolute() else (project_root / candidate)
            try:
                source_root = source_root.resolve()
            except OSError:
                pass
            source_label = "Tracked ablation source"
    return source_root, source_label, tracked_job, runs


def _aggregate_ablation_series_points(
    details: list[LibraryAblationDetail],
) -> list[dict[str, float | int]]:
    grouped: dict[int, dict[str, Any]] = {}
    for detail in details:
        for case in detail.case_results:
            if not case_counts_toward_label_series(case):
                continue
            observed_ratios = [
                item.success_ratio
                for item in case.seed_results
                if item.status in {"running", "completed"} and item.success_ratio is not None
            ]
            if not observed_ratios:
                continue
            bucket = grouped.setdefault(
                case.label_count,
                {
                    "ratios": [],
                    "case_count": 0,
                    "observed_seed_count": 0,
                    "completed_seed_count": 0,
                },
            )
            bucket["ratios"].extend(observed_ratios)
            bucket["case_count"] += 1
            bucket["observed_seed_count"] += sum(
                1 for item in case.seed_results if item.status in {"running", "completed"} and item.success_ratio is not None
            )
            bucket["completed_seed_count"] += case.completed_seed_count
    points: list[dict[str, float | int]] = []
    for label_count in sorted(grouped):
        ratios = [float(value) for value in grouped[label_count]["ratios"] if value is not None]
        if not ratios:
            continue
        mean_ratio = sum(ratios) / len(ratios)
        std_ratio = float(statistics.pstdev(ratios)) if len(ratios) > 1 else 0.0
        points.append(
            {
                "label_count": label_count,
                "mean_success_ratio": mean_ratio,
                "mean_success_ratio_std": std_ratio,
                "case_count": int(grouped[label_count]["case_count"]),
                "observed_seed_count": int(grouped[label_count]["observed_seed_count"]),
                "completed_seed_count": int(grouped[label_count]["completed_seed_count"]),
            }
        )
    return points


def _ablation_chart_series_from_points(
    *,
    points: list[dict[str, float | int]],
    label: str,
    color: str,
    dasharray: str = "",
    include_band: bool = True,
) -> list[dict[str, object]]:
    if not points:
        return []
    series: dict[str, object] = {
        "label": label,
        "color": color,
        "points": [
            (float(point["label_count"]), float(point["mean_success_ratio"]) * 100.0)
            for point in points
        ],
    }
    if dasharray:
        series["dasharray"] = dasharray
    if include_band:
        series["band"] = [
            (
                float(point["label_count"]),
                max(0.0, float(point["mean_success_ratio"] - point["mean_success_ratio_std"]) * 100.0),
                min(100.0, float(point["mean_success_ratio"] + point["mean_success_ratio_std"]) * 100.0),
            )
            for point in points
        ]
    return [series]


def build_ablation_chart_data_for_env_kind(
    *,
    jobs: list[JobRecord],
    project_root: Path,
    env_id: str,
    kind: str,
    plot_context: dict[tuple[str, str], dict[str, Any]] | None = None,
    include_all: bool = False,
) -> dict[str, Any]:
    source_root, source_label, tracked_job, runs = _ablation_plot_runs_for_env_kind(
        jobs=jobs,
        project_root=project_root,
        env_id=env_id,
        kind=kind,
        plot_context=plot_context,
        include_all=include_all,
    )
    points = _aggregate_ablation_series_points([detail for _job, detail in runs])
    title = f"{env_id.upper()} {kind} accumulated ablation results"
    return {
        "env_id": env_id,
        "kind": kind,
        "title": title,
        "x_label": "Library size (labels)",
        "y_label": "Task success rate (%)",
        "points": points,
        "series": _ablation_chart_series_from_points(
            points=points,
            label=f"{_ablation_plot_kind_label(kind)} mean",
            color=_library_ablation_palette(kind),
            include_band=True,
        ),
        "source_root": source_root,
        "source_label": source_label,
        "tracked_job": tracked_job,
        "runs": runs,
    }


def build_combined_ablation_chart_data(
    *,
    jobs: list[JobRecord],
    project_root: Path,
    plot_context: dict[tuple[str, str], dict[str, Any]] | None = None,
    include_all: bool = False,
    ablation_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    colors = _shared_plot_colors()
    env_colors = {
        env_id: colors[index % len(colors)]
        for index, env_id in enumerate(SUPPORTED_ENVS)
    }
    series: list[dict[str, object]] = []
    run_count = 0
    size_count = 0
    for env_id in SUPPORTED_ENVS:
        for kind in ("supervised", "unsupervised"):
            if isinstance(ablation_cache, dict):
                points, cached_run_count = build_ablation_points_from_cache(
                    cache=ablation_cache,
                    env_id=env_id,
                    kind=kind,
                )
                run_count += cached_run_count
            else:
                chart = build_ablation_chart_data_for_env_kind(
                    jobs=jobs,
                    project_root=project_root,
                    env_id=env_id,
                    kind=kind,
                    plot_context=plot_context,
                    include_all=include_all,
                )
                points = chart["points"] if isinstance(chart.get("points"), list) else []
                runs = chart["runs"] if isinstance(chart.get("runs"), list) else []
                run_count += len(runs)
            size_count += len(points)
            if not points:
                continue
            series.extend(
                _ablation_chart_series_from_points(
                    points=points,
                    label=f"{env_id.upper()} {_ablation_plot_kind_label(kind)}",
                    color=env_colors[env_id],
                    dasharray="" if kind == "supervised" else "6 4",
                    include_band=True,
                )
            )
    return {
        "title": "",
        "x_label": "Library size (labels)",
        "y_label": "Task success rate (%)",
        "series": series,
        "run_count": run_count,
        "size_count": size_count,
    }


def _normalized_ablation_importance(
    *,
    baseline_success_ratio: float,
    ablated_success_ratio: float,
) -> float | None:
    denominator = abs(float(baseline_success_ratio))
    if denominator <= 1e-9:
        return None
    return (float(baseline_success_ratio) - float(ablated_success_ratio)) / denominator


@lru_cache(maxsize=256)
def _max_training_reward_by_label_for_library_root(run_dir: Path) -> dict[str, float]:
    pattern_library = _load_json_dict(run_dir / "pattern_library.json")
    detectors = pattern_library.get("detectors") if isinstance(pattern_library.get("detectors"), list) else []
    rewards_by_label: dict[str, float] = {}
    for detector in detectors:
        if not isinstance(detector, dict):
            continue
        label_key = _canonical_label_key(str(detector.get("label") or ""))
        if not label_key:
            continue
        for score in _extract_train_scores(detector):
            existing = rewards_by_label.get(label_key)
            if existing is None or score > existing:
                rewards_by_label[label_key] = float(score)
    return rewards_by_label


def _max_training_reward_by_label_for_env_kind(*, project_root: Path, env_id: str, kind: str) -> dict[str, float]:
    library_root = (project_root / "state" / "video_renders" / "canonical_libraries" / env_id / kind).resolve()
    if not library_root.is_dir():
        return {}
    return _max_training_reward_by_label_for_library_root(library_root)


def build_combined_ablation_importance_scatter_data(
    *,
    jobs: list[JobRecord],
    project_root: Path,
    plot_context: dict[tuple[str, str], dict[str, Any]] | None = None,
    include_all: bool = False,
    ablation_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    colors = _shared_plot_colors()
    env_colors = {
        env_id: colors[index % len(colors)]
        for index, env_id in enumerate(SUPPORTED_ENVS)
    }
    if isinstance(ablation_cache, dict):
        rows, point_count, run_count = build_ablation_importance_rows_from_cache(ablation_cache)
        rows = [
            {
                **row,
                "color": env_colors.get(str(row.get("env_id") or "").strip().lower(), str(row.get("color") or "#617182")),
            }
            for row in rows
        ]
        return {
            "title": "Label importance vs training reward",
            "x_label": "Training reward",
            "y_label": "Normalized label importance",
            "rows": rows,
            "point_count": point_count,
            "run_count": run_count,
            "min_reward": -0.2,
            "max_reward": 1.0,
            "min_importance": -0.2,
            "max_importance": 1.0,
        }
    rows: list[dict[str, Any]] = []
    point_count = 0
    run_count = 0
    all_reward_values: list[float] = []
    all_importance_values: list[float] = []
    for env_id in SUPPORTED_ENVS:
        for kind in ("supervised", "unsupervised"):
            source_root, _source_label, _tracked_job, runs = _ablation_plot_runs_for_env_kind(
                jobs=jobs,
                project_root=project_root,
                env_id=env_id,
                kind=kind,
                plot_context=plot_context,
                include_all=include_all,
            )
            run_count += len(runs)
            row_key = f"{env_id}:{kind}"
            row_label = f"{env_id.upper()} {_library_ablation_kind_label(kind)}"
            label_values: dict[str, list[float]] = defaultdict(list)
            for _job, detail in runs:
                baseline_case = detail.baseline_case
                baseline_success_ratio = (
                    float(baseline_case.mean_success_ratio)
                    if baseline_case is not None and baseline_case.mean_success_ratio is not None
                    else None
                )
                if baseline_success_ratio is None:
                    continue
                for case in detail.case_results:
                    if case.kind == "baseline" or case.mean_success_ratio is None:
                        continue
                    held_out = [str(label).strip() for label in case.held_out_labels if str(label).strip()]
                    if not held_out:
                        continue
                    importance = _normalized_ablation_importance(
                        baseline_success_ratio=baseline_success_ratio,
                        ablated_success_ratio=float(case.mean_success_ratio),
                    )
                    if importance is None:
                        continue
                    for label in held_out:
                        label_values[label].append(float(importance))
            aggregated_points: list[dict[str, Any]] = []
            reward_by_label = _max_training_reward_by_label_for_env_kind(project_root=project_root, env_id=env_id, kind=kind)
            for label, values in sorted(
                label_values.items(),
                key=lambda item: (-statistics.fmean(item[1]), item[0].lower()),
            ):
                if not values:
                    continue
                label_key = _canonical_label_key(label)
                reward_value = reward_by_label.get(label_key) if label_key else None
                if not isinstance(reward_value, (int, float)) or not math.isfinite(float(reward_value)):
                    continue
                mean_importance = float(statistics.fmean(values))
                std_importance = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
                aggregated_points.append(
                    {
                        "label": label,
                        "training_reward": float(reward_value),
                        "importance": mean_importance,
                        "importance_std": std_importance,
                        "sample_count": len(values),
                    }
                )
                all_reward_values.append(float(reward_value))
                all_importance_values.append(mean_importance)
            point_count += len(aggregated_points)
            rows.append(
                {
                    "key": row_key,
                    "label": row_label,
                    "env_id": env_id,
                    "kind": kind,
                    "color": env_colors[env_id],
                    "points": aggregated_points,
                }
            )
    importance_scale = max((abs(value) for value in all_importance_values), default=1.0)
    importance_scale = importance_scale if importance_scale > 1e-9 else 1.0
    for row in rows:
        normalized_points: list[dict[str, Any]] = []
        for point in row.get("points") or []:
            clamped_reward = max(-1.0, min(1.0, float(point["training_reward"])))
            normalized_points.append(
                {
                    **point,
                    "training_reward_capped": clamped_reward,
                    "training_reward_normalized": clamped_reward,
                    "importance_normalized": max(
                        -1.0,
                        min(1.0, float(point["importance"]) / importance_scale),
                    ),
                    "importance_std_normalized": max(
                        0.0,
                        min(1.0, float(point["importance_std"]) / importance_scale),
                    ),
                }
            )
        row["points"] = normalized_points
    return {
        "title": "Label importance vs training reward",
        "x_label": "Training reward",
        "y_label": "Normalized label importance",
        "rows": rows,
        "point_count": point_count,
        "run_count": run_count,
        "min_reward": -0.2,
        "max_reward": 1.0,
        "min_importance": -0.2,
        "max_importance": 1.0,
    }


def _render_ablation_importance_scatter(
    *,
    title: str,
    x_label: str,
    y_label: str,
    rows: list[dict[str, Any]],
    min_reward: float,
    max_reward: float,
    min_importance: float,
    max_importance: float,
) -> str:
    config = _shared_plot_style_config()
    rc_params = config.get("rcParams") if isinstance(config.get("rcParams"), dict) else {}
    font_family_setting = rc_params.get("font.family")
    serif_candidates = rc_params.get("font.serif") if isinstance(rc_params.get("font.serif"), list) else []
    if isinstance(font_family_setting, list):
        svg_font_family = ", ".join(str(item) for item in font_family_setting if str(item).strip())
    elif isinstance(font_family_setting, str) and font_family_setting.strip():
        svg_font_family = font_family_setting.strip()
    elif serif_candidates:
        svg_font_family = ", ".join(str(item) for item in serif_candidates if str(item).strip())
    else:
        svg_font_family = "serif"
    label_settings = config.get("labels") if isinstance(config.get("labels"), dict) else {}
    tick_font_size = max(12.0, float((rc_params.get("xtick.labelsize") or rc_params.get("ytick.labelsize") or 11)))
    axis_label_font_size = max(14.0, float(label_settings.get("axis_label_size") or 12))
    empty_font_size = max(15.0, axis_label_font_size)
    valid_rows = [row for row in rows if isinstance(row.get("points"), list) and row.get("points")]
    width = 920
    height = 340
    top = 26
    bottom = 62
    left = 68
    right = 24
    plot_height = height - top - bottom
    plot_width = width - left - right
    if not valid_rows:
        return f"""
        <div class="chart-shell">
          <div class="chart-shell-head">
            <div>
              <h3>{html.escape(title)}</h3>
              <p class="metric-subtle">{html.escape(x_label)} vs {html.escape(y_label)}</p>
            </div>
          </div>
          <svg viewBox="0 0 {width} {height}" class="detail-chart" role="img" aria-label="{html.escape(title)}">
            <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="rgba(255,255,255,0.55)" stroke="rgba(87,104,125,0.18)" />
            <text x="{width/2:.1f}" y="{height/2:.1f}" text-anchor="middle" fill="#617182" font-size="{empty_font_size:.1f}" font-family="{html.escape(svg_font_family)}">No importance data yet</text>
          </svg>
        </div>
        """
    if math.isclose(min_importance, max_importance):
        max_importance += 1.0
    flattened: list[dict[str, Any]] = []
    for row in valid_rows:
        for point in row.get("points") or []:
            flattened.append(
                {
                    "row_label": str(row.get("label") or ""),
                    "env_id": str(row.get("env_id") or ""),
                    "kind": str(row.get("kind") or ""),
                    "color": str(row.get("color") or "#124f49"),
                    "label": str(point.get("label") or ""),
                    "training_reward": float(point.get("training_reward") or 0.0),
                    "training_reward_capped": float(point.get("training_reward_capped") or point.get("training_reward") or 0.0),
                    "training_reward_normalized": float(point.get("training_reward_normalized") or 0.0),
                    "importance": float(point.get("importance") or 0.0),
                    "importance_normalized": float(point.get("importance_normalized") or 0.0),
                    "sample_count": int(point.get("sample_count") or 0),
                }
            )
    flattened.sort(
        key=lambda item: (
            item["env_id"],
            item["kind"],
            -float(item["importance"]),
            str(item["label"]).lower(),
        )
    )
    def sx(value: float) -> float:
        return left + ((float(value) - min_reward) / (max_reward - min_reward)) * plot_width
    def sy(value: float) -> float:
        return top + (1.0 - ((float(value) - min_importance) / (max_importance - min_importance))) * plot_height
    y_ticks: list[str] = []
    grid: list[str] = []
    tick_count = 6
    for tick_index in range(tick_count):
        ratio = tick_index / max(tick_count - 1, 1)
        tick_value = min_importance + (max_importance - min_importance) * ratio
        y_pos = sy(tick_value)
        grid.append(
            f'<line x1="{left}" y1="{y_pos:.1f}" x2="{left + plot_width}" y2="{y_pos:.1f}" stroke="rgba(87,104,125,0.14)" stroke-width="1" />'
        )
        y_ticks.append(
            f'<text x="{left-10}" y="{y_pos+4:.1f}" text-anchor="end" fill="#617182" font-size="{tick_font_size:.1f}" font-family="{html.escape(svg_font_family)}">{html.escape(f"{tick_value:.2f}")}</text>'
        )
    if min_importance < 0.0 < max_importance:
        zero_y = sy(0.0)
        grid.append(
            f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_width}" y2="{zero_y:.1f}" stroke="rgba(165,77,45,0.55)" stroke-width="1.5" stroke-dasharray="4 4" />'
        )
    x_axis_ticks: list[str] = []
    for tick_index in range(tick_count):
        ratio = tick_index / max(tick_count - 1, 1)
        tick_value = min_reward + (max_reward - min_reward) * ratio
        x_pos = sx(tick_value)
        grid.append(
            f'<line x1="{x_pos:.1f}" y1="{top}" x2="{x_pos:.1f}" y2="{top + plot_height}" stroke="rgba(87,104,125,0.10)" stroke-width="1" />'
        )
        x_axis_ticks.append(
            f'<text x="{x_pos:.1f}" y="{height-14}" text-anchor="middle" fill="#617182" font-size="{tick_font_size:.1f}" font-family="{html.escape(svg_font_family)}">{html.escape(f"{tick_value:.2f}")}</text>'
        )
    if min_reward < 0.0 < max_reward:
        zero_x = sx(0.0)
        grid.append(
            f'<line x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{top + plot_height}" stroke="rgba(18,79,73,0.45)" stroke-width="1.5" stroke-dasharray="4 4" />'
        )
    point_render: list[str] = []
    separators: list[str] = []
    legend_by_env: dict[str, str] = {}
    for row in valid_rows:
        env_id = str(row.get("env_id") or "").strip().lower()
        color = str(row.get("color") or "").strip()
        if env_id and color and env_id not in legend_by_env:
            legend_by_env[env_id] = color
    legend_bits: list[str] = [
        f'<span class="mini-chip"><span class="inline-swatch" style="background:{html.escape(color)};"></span>{html.escape(env_id.upper())}</span>'
        for env_id, color in legend_by_env.items()
    ]
    for item in flattened:
        x = sx(float(item["training_reward_capped"]))
        y = sy(float(item["importance_normalized"]))
        color = str(item["color"])
        title_text = (
            f"{item['row_label']}: {item['label']} | "
            f"reward={float(item['training_reward_capped']):.3f} | "
            f"importance={float(item['importance']):.3f} ({float(item['importance_normalized']):+.3f} norm) | "
            f"n={int(item['sample_count'])}"
        )
        point_render.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.1" fill="{html.escape(color)}" fill-opacity="0.82" stroke="rgba(255,255,255,0.92)" stroke-width="1.3"><title>{html.escape(title_text)}</title></circle>'
        )
    return f"""
    <div class="chart-shell">
      <div class="chart-shell-head">
        <div>
          <h3>{html.escape(title)}</h3>
          <p class="metric-subtle">{html.escape(x_label)} vs {html.escape(y_label)}</p>
        </div>
        <div class="stats-inline">{''.join(legend_bits)}</div>
      </div>
      <svg viewBox="0 0 {width} {height}" class="detail-chart" role="img" aria-label="{html.escape(title)}">
        <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="rgba(255,255,255,0.55)" stroke="rgba(87,104,125,0.18)" />
        {''.join(grid)}
        {''.join(separators)}
        {''.join(point_render)}
        {''.join(x_axis_ticks)}
        {''.join(y_ticks)}
        <text x="{left + plot_width / 2:.1f}" y="{height - 36:.1f}" text-anchor="middle" fill="#2c3744" font-size="{axis_label_font_size:.1f}" font-family="{html.escape(svg_font_family)}" font-weight="600">{html.escape(x_label)}</text>
        <text x="18" y="{top + plot_height / 2:.1f}" text-anchor="middle" fill="#2c3744" font-size="{axis_label_font_size:.1f}" font-family="{html.escape(svg_font_family)}" font-weight="600" transform="rotate(-90 18,{top + plot_height / 2:.1f})">{html.escape(y_label)}</text>
      </svg>
    </div>
    """


def render_ablation_plot_pdf_bytes(
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[dict[str, object]],
) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    config = _shared_plot_style_config()
    style_name = str(config.get("style") or "").strip()
    if style_name:
        plt.style.use(style_name)

    rc_params = config.get("rcParams") if isinstance(config.get("rcParams"), dict) else {}
    if isinstance(rc_params, dict):
        matplotlib.rcParams.update(rc_params)
        serif_candidates = rc_params.get("font.serif")
        if isinstance(serif_candidates, list):
            available_fonts = {font.name for font in font_manager.fontManager.ttflist}
            chosen = next((name for name in serif_candidates if name in available_fonts), None)
            if chosen is None and "DejaVu Serif" in available_fonts:
                chosen = "DejaVu Serif"
            if chosen is not None:
                matplotlib.rcParams["font.family"] = ["serif"]
                matplotlib.rcParams["font.serif"] = [chosen]
                matplotlib.rcParams["mathtext.rm"] = chosen
                matplotlib.rcParams["mathtext.it"] = f"{chosen}:italic"
                matplotlib.rcParams["mathtext.bf"] = f"{chosen}:bold"

    figure_settings = ((config.get("aesthetics") or {}).get("figure") or {})
    grid_settings = ((config.get("aesthetics") or {}).get("grid") or {})
    legend_settings = ((config.get("aesthetics") or {}).get("legend") or {})
    error_region_settings = ((config.get("aesthetics") or {}).get("error_region") or {})
    label_settings = config.get("labels") or {}

    base_width, base_height = _shared_plot_figure_size("wide")
    figure_size = (max(base_width * 2.2, 14.0), max(base_height * 3.3, 9.0))
    fig, ax = plt.subplots(figsize=figure_size)
    facecolor = str(figure_settings.get("facecolor") or "white")
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)

    has_series = False
    min_x: float | None = None
    max_x: float | None = None
    max_y: float = 0.0
    error_alpha = float(error_region_settings.get("alpha") or 0.15)
    for item in series:
        raw_points = item.get("points")
        if not isinstance(raw_points, list) or not raw_points:
            continue
        xs: list[float] = []
        ys: list[float] = []
        for point in raw_points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            try:
                x_value = float(point[0])
                y_value = float(point[1])
            except (TypeError, ValueError):
                continue
            xs.append(x_value)
            ys.append(y_value)
        if not xs:
            continue
        has_series = True
        min_x = min(xs) if min_x is None else min(min_x, min(xs))
        max_x = max(xs) if max_x is None else max(max_x, max(xs))
        max_y = max(max_y, max(ys))
        color = str(item.get("color") or "#617182")
        label = str(item.get("label") or "").strip()
        dasharray = str(item.get("dasharray") or "").strip()
        linestyle: str | tuple[float, tuple[float, ...]] = "-"
        if dasharray:
            try:
                dash_values = tuple(float(part) for part in dasharray.split())
            except ValueError:
                dash_values = ()
            if dash_values:
                linestyle = (0.0, dash_values)
        raw_band = item.get("band")
        if isinstance(raw_band, list) and raw_band:
            band_x: list[float] = []
            band_low: list[float] = []
            band_high: list[float] = []
            for point in raw_band:
                if not isinstance(point, (list, tuple)) or len(point) != 3:
                    continue
                try:
                    band_x.append(float(point[0]))
                    band_low.append(float(point[1]))
                    band_high.append(float(point[2]))
                except (TypeError, ValueError):
                    continue
            if band_x:
                max_y = max(max_y, max(band_high))
                ax.fill_between(band_x, band_low, band_high, color=color, alpha=error_alpha, linewidth=0)
        ax.plot(xs, ys, color=color, label=label, linestyle=linestyle, marker=None)

    ax.set_xlabel(x_label, labelpad=float(label_settings.get("padding") or 10))
    ax.set_ylabel(y_label, labelpad=float(label_settings.get("padding") or 10))
    ax.grid(axis="y", **grid_settings)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_handles = []
    legend_labels = []
    if has_series:
        legend_kwargs = dict(legend_settings)
        legend_kwargs["frameon"] = True
        legend_kwargs["facecolor"] = facecolor
        legend_kwargs["framealpha"] = 1.0
        legend_kwargs["loc"] = "upper left"
        legend_kwargs["bbox_to_anchor"] = (0.08, 0.985)
        legend_kwargs["ncol"] = min(4, max(2, math.ceil(len(series) / 2)))
        legend_kwargs["handlelength"] = 2.0
        legend_handles, legend_labels = ax.get_legend_handles_labels()
        lower_x = min_x if min_x is not None else 0.0
        upper_x = max_x if max_x is not None else 1.0
        if math.isclose(lower_x, upper_x):
            upper_x += 1.0
        ax.set_xlim(lower_x - 0.1, upper_x + 0.1)
        if "success" in y_label.lower():
            ax.set_ylim(0.0, 100.0)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
        else:
            ax.set_ylim(0.0, max(max_y + 1.0, 2.0))
        if min_x is not None and max_x is not None:
            tick_values = sorted({int(round(float(point[0]))) for item in series for point in (item.get("points") or []) if isinstance(point, (list, tuple)) and len(point) == 2})
            if tick_values:
                ax.set_xticks(tick_values)
    else:
        ax.text(0.5, 0.5, "No chart data yet", transform=ax.transAxes, ha="center", va="center", color="#617182")

    if legend_handles and legend_labels:
        ax.legend(legend_handles, legend_labels, **legend_kwargs)

    buffer = io.BytesIO()
    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.18)
    fig.savefig(buffer, format="pdf")
    plt.close(fig)
    return buffer.getvalue()


def render_ablation_importance_scatter_pdf_bytes(
    *,
    title: str,
    x_label: str,
    y_label: str,
    rows: list[dict[str, Any]],
    x_limits: tuple[float, float] = (-0.2, 1.0),
    y_limits: tuple[float, float] = (-0.2, 1.0),
) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D

    config = _shared_plot_style_config()
    style_name = str(config.get("style") or "").strip()
    if style_name:
        plt.style.use(style_name)

    rc_params = config.get("rcParams") if isinstance(config.get("rcParams"), dict) else {}
    if isinstance(rc_params, dict):
        matplotlib.rcParams.update(rc_params)
        serif_candidates = rc_params.get("font.serif")
        if isinstance(serif_candidates, list):
            available_fonts = {font.name for font in font_manager.fontManager.ttflist}
            chosen = next((name for name in serif_candidates if name in available_fonts), None)
            if chosen is None and "DejaVu Serif" in available_fonts:
                chosen = "DejaVu Serif"
            if chosen is not None:
                matplotlib.rcParams["font.family"] = ["serif"]
                matplotlib.rcParams["font.serif"] = [chosen]
                matplotlib.rcParams["mathtext.rm"] = chosen
                matplotlib.rcParams["mathtext.it"] = f"{chosen}:italic"
                matplotlib.rcParams["mathtext.bf"] = f"{chosen}:bold"

    figure_settings = ((config.get("aesthetics") or {}).get("figure") or {})
    grid_settings = ((config.get("aesthetics") or {}).get("grid") or {})
    legend_settings = ((config.get("aesthetics") or {}).get("legend") or {})
    label_settings = config.get("labels") or {}

    def _resolved_font_points(value: Any, fallback: float) -> float:
        try:
            return float(font_manager.FontProperties(size=value).get_size_in_points())
        except Exception:
            return fallback

    axis_label_size = _resolved_font_points(matplotlib.rcParams.get("axes.labelsize"), 13.0) * 1.15
    tick_label_size = _resolved_font_points(matplotlib.rcParams.get("xtick.labelsize"), 11.0) * 1.15
    legend_font_size = _resolved_font_points(matplotlib.rcParams.get("legend.fontsize"), 11.0) * 1.15

    base_width, base_height = _shared_plot_figure_size("wide")
    figure_size = (max(base_width * 2.2, 14.0), max(base_height * 3.3, 9.0))
    fig, ax = plt.subplots(figsize=figure_size)
    facecolor = str(figure_settings.get("facecolor") or "white")
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)

    legend_by_env: dict[str, str] = {}
    has_points = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        points = row.get("points")
        if not isinstance(points, list) or not points:
            continue
        color = str(row.get("color") or "#617182")
        env_label = str(row.get("env_id") or "").strip().upper() or str(row.get("label") or "").strip()
        xs: list[float] = []
        ys: list[float] = []
        for point in points:
            if not isinstance(point, dict):
                continue
            try:
                xs.append(float(point.get("training_reward_capped", point.get("training_reward", 0.0))))
                ys.append(float(point.get("importance_normalized", 0.0)))
            except (TypeError, ValueError):
                continue
        if not xs:
            continue
        ax.scatter(xs, ys, s=80, color=color, alpha=0.9, edgecolors="white", linewidths=1.0)
        legend_by_env.setdefault(env_label, color)
        has_points = True

    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_xlabel(x_label, labelpad=float(label_settings.get("padding") or 10), fontsize=axis_label_size)
    ax.set_ylabel(y_label, labelpad=float(label_settings.get("padding") or 10), fontsize=axis_label_size)
    ax.tick_params(axis="both", labelsize=tick_label_size)
    ax.grid(axis="both", **grid_settings)
    if x_limits[0] < 0.0 < x_limits[1]:
        ax.axvline(0.0, color=(18 / 255.0, 79 / 255.0, 73 / 255.0, 0.45), linewidth=1.5, linestyle=(0, (4, 4)))
    if y_limits[0] < 0.0 < y_limits[1]:
        ax.axhline(0.0, color=(165 / 255.0, 77 / 255.0, 45 / 255.0, 0.55), linewidth=1.5, linestyle=(0, (4, 4)))
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if has_points and legend_by_env:
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=10, label=label)
            for label, color in legend_by_env.items()
        ]
        legend_kwargs = dict(legend_settings)
        legend_kwargs["frameon"] = True
        legend_kwargs["facecolor"] = facecolor
        legend_kwargs["framealpha"] = 1.0
        legend_kwargs["loc"] = "upper right"
        legend_kwargs["bbox_to_anchor"] = (0.98, 0.985)
        legend_kwargs["ncol"] = min(4, max(2, math.ceil(len(handles) / 2)))
        legend_kwargs["handlelength"] = 2.0
        legend_kwargs["fontsize"] = legend_font_size
        ax.legend(handles=handles, **legend_kwargs)
    elif not has_points:
        ax.text(0.5, 0.5, "No importance data yet", transform=ax.transAxes, ha="center", va="center", color="#617182")

    buffer = io.BytesIO()
    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.18)
    fig.savefig(buffer, format="pdf")
    plt.close(fig)
    return buffer.getvalue()


def _ablation_plot_card(
    *,
    jobs: list[JobRecord],
    project_root: Path,
    env_id: str,
    kind: str,
    plot_context: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> str:
    chart_data = build_ablation_chart_data_for_env_kind(
        jobs=jobs,
        project_root=project_root,
        env_id=env_id,
        kind=kind,
        plot_context=plot_context,
    )
    source_root = chart_data.get("source_root") if isinstance(chart_data.get("source_root"), Path) else None
    source_label = str(chart_data.get("source_label") or "").strip()
    tracked_job = chart_data.get("tracked_job") if isinstance(chart_data.get("tracked_job"), JobRecord) else None
    runs = chart_data.get("runs") if isinstance(chart_data.get("runs"), list) else []
    source_label_title, source_display_name = (
        source_label.split("\n", 1)
        if "\n" in source_label
        else (source_label or "Library source", source_root.name if source_root is not None else "No source yet")
    )
    display_job = _preferred_ablation_job(tracked_job, runs)
    points = chart_data.get("points") if isinstance(chart_data.get("points"), list) else []
    queued_count = sum(
        1 for job, _detail in runs if job.status.value in ACTIVE_QUEUE_STATUSES | {"queued"}
    )
    status_label = display_job.status.value if display_job is not None else ("untracked" if source_root is not None else "missing")
    queue_button = (
        f"""
        <form method="post" action="/ui/ablations/queue-plot">
          <input type="hidden" name="next" value="/ablations">
          <input type="hidden" name="env_id" value="{html.escape(env_id)}">
          <input type="hidden" name="kind" value="{html.escape(kind)}">
          <button class="secondary-button" type="submit">Queue Another</button>
        </form>
        """
        if source_root is not None
        else ""
    )
    chart_html = _render_xy_line_chart(
        title=f"{env_id.upper()} {kind} library performance by size",
        x_label="Labels used",
        y_label="Task success (%)",
        series=chart_data.get("series") if points else [],
    )
    summary_rows = [
        f"""
        <div class="required-row-meta">
          <span>{html.escape(source_label_title)}</span>
          <span>{html.escape(source_display_name)}</span>
          <span>{len(points)} sizes</span>
          <span>{len(runs)} runs</span>
          {(_pill(status_label, status_label) if status_label != "missing" else '<span class="mini-chip">Missing</span>')}
        </div>
        """
    ]
    return f"""
    <section class="required-card">
      <div class="card-topline">
        <div>
          <p class="section-kicker">{html.escape(_library_ablation_kind_label(kind))}</p>
          <h2>{html.escape(env_id.upper())}</h2>
        </div>
        <div class="stats-inline">
          <span class="mini-chip">{queued_count} queued</span>
          <a class="ghost-link" href="/ablations/data/{html.escape(kind)}/{html.escape(env_id)}">Data</a>
        </div>
      </div>
      {chart_html}
      <div class="required-list">{''.join(summary_rows)}</div>
      <div class="required-action-bar">{queue_button}</div>
    </section>
    """


def render_ablation_data_page(
    *,
    kind: str,
    env_id: str,
    jobs: list[JobRecord],
    project_root: Path,
    plot_context: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> str:
    chart_data = build_ablation_chart_data_for_env_kind(
        jobs=jobs,
        project_root=project_root,
        env_id=env_id,
        kind=kind,
        plot_context=plot_context,
        include_all=True,
    )
    source_root = chart_data.get("source_root") if isinstance(chart_data.get("source_root"), Path) else None
    source_label = str(chart_data.get("source_label") or "").strip()
    tracked_job = chart_data.get("tracked_job") if isinstance(chart_data.get("tracked_job"), JobRecord) else None
    runs = chart_data.get("runs") if isinstance(chart_data.get("runs"), list) else []
    display_job = _preferred_ablation_job(tracked_job, runs)
    source_label_title, source_display_name = (
        source_label.split("\n", 1)
        if "\n" in source_label
        else (source_label or "Library source", source_root.name if source_root is not None else "No source yet")
    )
    points = chart_data.get("points") if isinstance(chart_data.get("points"), list) else []
    chart_html = _render_xy_line_chart(
        title=str(chart_data.get("title") or f"{env_id.upper()} {kind} accumulated ablation results"),
        x_label=str(chart_data.get("x_label") or "Labels used"),
        y_label=str(chart_data.get("y_label") or "Task success (%)"),
        series=chart_data.get("series") if points else [],
    )
    point_summary = " · ".join(
        [
            f"{len(points)} sizes",
            f"{sum(int(point.get('observed_seed_count') or 0) for point in points)} observed seed evaluations",
            f"{sum(int(point['completed_seed_count']) for point in points)} completed seed evaluations",
            f"{sum(int(point['case_count']) for point in points)} accumulated holdout cases",
        ]
    )
    rows: list[str] = []
    for job, detail in runs:
        rows.append(
            f"""
            <tr>
              <td>{html.escape(detail.run_id)}</td>
              <td>{html.escape(job.status.value)}</td>
              <td>{detail.completed_seed_run_count}/{detail.total_seed_run_count}</td>
              <td>{len(detail.aggregate_points)}</td>
              <td>{html.escape(_display_time(job.updated_at))}</td>
              <td><a class="ghost-link" href="/ablations/{html.escape(job.job_id)}">View</a></td>
              <td><a class="ghost-link" href="/jobs/{html.escape(job.job_id)}">Job</a></td>
            </tr>
            """
        )
    if not rows:
        rows.append('<tr><td colspan="7" class="empty">No ablation runs yet for this source.</td></tr>')
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/ablations">Ablations</a> / data / {html.escape(kind)} / {html.escape(env_id)}</p>
      <h1>{html.escape(env_id.upper())} {html.escape(_library_ablation_kind_label(kind))} Data</h1>
      <p>Accumulated performance across observed seed evaluations for this environment and ablation family, including partial in-progress seeds once task results have landed.</p>
    </section>
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div>
          <p class="section-kicker">{html.escape(_library_ablation_kind_label(kind))}</p>
          <h2>Accumulated Result Plot</h2>
          <p class="metric-subtle">Mean task success by labels used, with error bars showing one standard deviation across observed seed evaluations, including partial in-progress seeds.</p>
        </div>
        <div class="stats-inline">
          <span class="mini-chip">{len(points)} sizes</span>
          <span class="mini-chip">{len(runs)} runs</span>
          {(_pill(display_job.status.value, display_job.status.value) if display_job is not None else '<span class="mini-chip">Untracked</span>')}
        </div>
      </div>
      {chart_html}
      <div class="required-list">
        <div class="required-row-meta">
          <span>Accumulated data</span>
          <span>{html.escape(point_summary if points else 'No observed seed evaluations have been recorded yet.')}</span>
        </div>
        <div class="required-row-meta">
          <span>{html.escape(source_label_title or 'Source')}</span>
          <span>{html.escape(source_display_name or 'No source available yet')}</span>
        </div>
      </div>
    </section>
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div>
          <p class="section-kicker">{html.escape(_library_ablation_kind_label(kind))}</p>
          <h2>Contributing Runs</h2>
          <p class="metric-subtle">{html.escape((source_label_title + ' · ' + source_display_name) if source_root is not None else 'No source available yet.')}</p>
        </div>
        <div class="stats-inline">
          <span class="mini-chip">{len(runs)} runs</span>
          <span class="mini-chip">{len(points)} plotted sizes</span>
        </div>
      </div>
      <div class="table-scroll compact-table">
        <table class="data-table">
          <thead><tr><th>Run</th><th>Status</th><th>Runs</th><th>Sizes</th><th>Updated</th><th>Ablation</th><th>Job</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </section>
    """
    return _shell(f"{env_id.upper()} {kind.title()} Ablation Data | Pattern Server", active_nav="ablations", body_html=body)


def _render_ablation_section(
    *,
    jobs: list[JobRecord],
    project_root: Path,
    kind: str,
    plot_context: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> str:
    cards = "".join(
        _ablation_plot_card(
            jobs=jobs,
            project_root=project_root,
            env_id=env_id,
            kind=kind,
            plot_context=plot_context,
        )
        for env_id in SUPPORTED_ENVS
    )
    return f"""
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Library family</p>
          <h2>{html.escape(_library_ablation_kind_label(kind))}</h2>
          <p class="metric-subtle">Performance for different library sizes across all four environments.</p>
        </div>
      </div>
      <div class="required-grid">{cards}</div>
    </section>
    """


def _iter_direct_ablation_eval_roots(*, project_root: Path) -> list[Path]:
    if project_root.name == "pattern-learning-server":
        canonical_root = project_root / "state" / "video_renders" / "canonical_libraries"
    else:
        canonical_root = project_root / "pattern-learning-server" / "state" / "video_renders" / "canonical_libraries"
    if not canonical_root.is_dir():
        return []
    roots: list[Path] = []
    for env_id in SUPPORTED_ENVS:
        for kind in ("supervised", "unsupervised"):
            eval_parent = canonical_root / env_id / kind / "library_evaluations"
            if not eval_parent.is_dir():
                continue
            for eval_root in sorted(eval_parent.glob("continuous-direct-*")):
                if eval_root.is_dir():
                    roots.append(eval_root)
    return roots


def _collect_recent_direct_ablation_results(
    *,
    project_root: Path,
    limit: int = 36,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for eval_root in _iter_direct_ablation_eval_roots(project_root=project_root):
        try:
            env_id = eval_root.parents[2].name
            kind = eval_root.parents[1].name
        except IndexError:
            continue
        run_id = eval_root.name
        source = "remote" if "-remote-" in run_id else "local" if "-local-" in run_id else "direct"
        result_paths = sorted(
            {
                *eval_root.glob("cases/*/tasks/*/*/task_result.json"),
                *eval_root.glob("tasks/*/*/task_result.json"),
            }
        )
        for result_path in result_paths:
            try:
                received_at = datetime.fromtimestamp(result_path.stat().st_mtime, tz=DISPLAY_TZ)
            except OSError:
                continue
            task_id = result_path.parent.name
            status_label = ""
            status_tone = "queued"
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                success = payload.get("success")
                if isinstance(success, bool):
                    status_label = "success" if success else "fail"
                    status_tone = "succeeded" if success else "failed"
                else:
                    terminal_status = str(payload.get("terminal_status") or "").strip()
                    if terminal_status:
                        status_label = terminal_status.replace("_", " ").lower()
                        lowered = terminal_status.lower()
                        if "solved" in lowered or lowered in {"success", "succeeded"}:
                            status_tone = "succeeded"
                        elif "error" in lowered or "fail" in lowered:
                            status_tone = "failed"
            items.append(
                {
                    "env_id": env_id,
                    "kind": kind,
                    "source": source,
                    "run_id": run_id,
                    "task_id": task_id,
                    "result_path": result_path,
                    "received_at": received_at,
                    "status_label": status_label,
                    "status_tone": status_tone,
                }
            )
    items.sort(key=lambda item: item["received_at"], reverse=True)
    return items[: max(1, limit)]


def render_ablations_home_page(
    *,
    jobs: list[JobRecord],
    clients: list[ClientRecord],
    project_root: Path,
    plot_context: dict[tuple[str, str], dict[str, Any]] | None = None,
    importance_state: dict[str, Any] | None = None,
) -> str:
    del clients, jobs, plot_context
    state_payload = importance_state if isinstance(importance_state, dict) else {}
    combined = state_payload.get("combined") if isinstance(state_payload.get("combined"), dict) else {}
    importance = state_payload.get("importance") if isinstance(state_payload.get("importance"), dict) else {}
    combined_status = str(combined.get("status") or "idle").strip().lower() or "idle"
    combined_message = str(combined.get("message") or "").strip() or "Idle."
    combined_updated_at = str(combined.get("updated_at") or "").strip()
    combined_progress = max(0.0, min(1.0, float(combined.get("progress") or 0.0)))
    combined_html = str(combined.get("html") or "").strip()
    combined_run_count = int(combined.get("run_count") or 0)
    combined_size_count = int(combined.get("size_count") or 0)
    show_generated_combined = combined_status not in {"running", "cancelling"}
    can_cancel_combined = combined_status in {"running", "cancelling"}
    combined_progress_pct = int(round(combined_progress * 100.0))
    importance_status = str(importance.get("status") or "idle").strip().lower() or "idle"
    importance_message = str(importance.get("message") or "").strip() or "Idle."
    importance_updated_at = str(importance.get("updated_at") or "").strip()
    importance_progress = max(0.0, min(1.0, float(importance.get("progress") or 0.0)))
    importance_html = str(importance.get("html") or "").strip()
    importance_point_count = int(importance.get("point_count") or 0)
    importance_run_count = int(importance.get("run_count") or 0)
    show_generated_importance = importance_status not in {"running", "cancelling"}
    can_cancel_importance = importance_status in {"running", "cancelling"}
    importance_progress_pct = int(round(importance_progress * 100.0))
    incoming_results = _collect_recent_direct_ablation_results(project_root=project_root)
    latest_result_time = _display_time(incoming_results[0]["received_at"]) if incoming_results else "—"
    incoming_results_html = "".join(
        f"""
        <article class="task-row">
          <div class="task-row-main">
            <span class="metric-title">{html.escape(item["env_id"].upper())} · {html.escape(str(item["kind"]).title())} · {html.escape(str(item["source"]).title())}</span>
            <p class="metric-subtle">
              Received {html.escape(_display_time(item["received_at"]))} ·
              run <code>{html.escape(str(item["run_id"]))}</code> ·
              task <code>{html.escape(str(item["task_id"]))}</code>
            </p>
          </div>
          <div class="task-row-meta">
            {(_pill(str(item["status_label"]), str(item["status_tone"])) if str(item["status_label"]).strip() else "")}
            <a class="ghost-link" href="/ui/files?path={quote(_project_relpath(Path(item["result_path"])))}">task_result.json</a>
          </div>
        </article>
        """
        for item in incoming_results
    )
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Learning</a> / ablations</p>
      <h1>Ablations</h1>
    </section>
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div>
          <h2>Combined Ablations</h2>
        </div>
        <div class="stats-inline">
          {_pill(combined_status, combined_status)}
          <span class="mini-chip">{'generating' if not show_generated_combined else f'{combined_run_count} ablation runs'}</span>
          <span class="mini-chip">{'generating' if not show_generated_combined else f'{combined_size_count} sizes'}</span>
        </div>
      </div>
      <div class="required-action-bar">
        <form method="post" action="/ui/ablations/generate-combined" class="summary-action-form">
          <input type="hidden" name="next" value="/ablations">
          <button class="secondary-button" type="submit">Generate line plot</button>
        </form>
        <a class="ghost-link" href="/api/ui/ablations/plot.pdf?combined=1">Download PDF</a>
        <form method="post" action="/ui/ablations/cancel" class="summary-action-form">
          <input type="hidden" name="next" value="/ablations">
          <input type="hidden" name="section" value="combined">
          <button class="secondary-button danger-button" type="submit"{'' if can_cancel_combined else ' disabled'}>Cancel</button>
        </form>
      </div>
      <div style="display:grid;gap:0.35rem;margin:0.35rem 0 0.6rem;">
        <div style="display:flex;justify-content:space-between;gap:1rem;align-items:center;flex-wrap:wrap;">
          <span id="ablations-combined-message" class="metric-subtle">{html.escape(combined_message)}</span>
          <span id="ablations-combined-percent" class="metric-subtle">{combined_progress_pct}% · {html.escape(combined_updated_at or '—')}</span>
        </div>
        <div style="height:0.7rem;border:1px solid var(--line);background:rgba(255,255,255,0.72);border-radius:999px;overflow:hidden;">
          <div id="ablations-combined-bar" style="height:100%;width:{combined_progress_pct}%;background:linear-gradient(90deg, var(--accent), #111827);transition:width 180ms ease;"></div>
        </div>
      </div>
      {(combined_html if show_generated_combined and combined_html else '<p class="metric-subtle">Generating.</p>' if not show_generated_combined else '<p class="metric-subtle">No plot yet.</p>')}
    </section>
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div>
          <h2>Label Importance</h2>
        </div>
        <div class="stats-inline">
          {_pill(importance_status, importance_status)}
          <span class="mini-chip">{'generating' if not show_generated_importance else f'{importance_point_count} labels'}</span>
          <span class="mini-chip">{'generating' if not show_generated_importance else f'{importance_run_count} ablation runs'}</span>
        </div>
      </div>
      <div class="required-action-bar">
        <form method="post" action="/ui/ablations/generate-importance" class="summary-action-form">
          <input type="hidden" name="next" value="/ablations">
          <button class="secondary-button" type="submit">Generate scatter plot</button>
        </form>
        <a class="ghost-link" href="/api/ui/ablations/plot.pdf?importance=1">Download PDF</a>
        <form method="post" action="/ui/ablations/cancel" class="summary-action-form">
          <input type="hidden" name="next" value="/ablations">
          <input type="hidden" name="section" value="importance">
          <button class="secondary-button danger-button" type="submit"{'' if can_cancel_importance else ' disabled'}>Cancel</button>
        </form>
      </div>
      <div style="display:grid;gap:0.35rem;margin:0.35rem 0 0.6rem;">
        <div style="display:flex;justify-content:space-between;gap:1rem;align-items:center;flex-wrap:wrap;">
          <span id="ablations-importance-message" class="metric-subtle">{html.escape(importance_message)}</span>
          <span id="ablations-importance-percent" class="metric-subtle">{importance_progress_pct}% · {html.escape(importance_updated_at or '—')}</span>
        </div>
        <div style="height:0.7rem;border:1px solid var(--line);background:rgba(255,255,255,0.72);border-radius:999px;overflow:hidden;">
          <div id="ablations-importance-bar" style="height:100%;width:{importance_progress_pct}%;background:linear-gradient(90deg, var(--accent), #111827);transition:width 180ms ease;"></div>
        </div>
      </div>
      {(importance_html if show_generated_importance and importance_html else '<p class="metric-subtle">Generating.</p>' if not show_generated_importance else '<p class="metric-subtle">No plot yet.</p>')}
    </section>
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div>
          <h2>Direct Ablation Feed</h2>
        </div>
        <div class="stats-inline">
          <span class="mini-chip">{len(incoming_results)} recent results</span>
          <span class="mini-chip">{html.escape(latest_result_time)}</span>
        </div>
      </div>
      <div class="task-list">{incoming_results_html or '<p class="empty">No direct ablation results have landed on disk yet.</p>'}</div>
    </section>
    <script>
      (() => {{
        const running = {str((not show_generated_combined) or (not show_generated_importance)).lower()};
        if (!running) return;
        const nodes = {{
          combined: {{
            message: document.getElementById('ablations-combined-message'),
            percent: document.getElementById('ablations-combined-percent'),
            bar: document.getElementById('ablations-combined-bar'),
          }},
          importance: {{
            message: document.getElementById('ablations-importance-message'),
            percent: document.getElementById('ablations-importance-percent'),
            bar: document.getElementById('ablations-importance-bar'),
          }},
        }};
        const poll = () => fetch('/api/ui/ablations/status', {{ cache: 'no-store' }})
          .then((r) => r.ok ? r.json() : null)
          .then((payload) => {{
            if (!payload) return;
            let anyRunning = false;
            for (const key of ['combined', 'importance']) {{
              const state = payload[key];
              if (!state) continue;
              const status = String(state.status || 'idle').trim().toLowerCase();
              const progress = Math.max(0, Math.min(1, Number(state.progress || 0)));
              const pct = Math.round(progress * 100);
              const message = String(state.message || '').trim() || 'Idle.';
              const updatedAt = String(state.updated_at || '').trim() || '—';
              if (nodes[key].message) nodes[key].message.textContent = message;
              if (nodes[key].percent) nodes[key].percent.textContent = `${{pct}}% · ${{updatedAt}}`;
              if (nodes[key].bar) nodes[key].bar.style.width = `${{pct}}%`;
              if (status === 'running' || status === 'cancelling') anyRunning = true;
            }}
            if (anyRunning) {{
              window.setTimeout(poll, 1500);
            }} else {{
              window.location.reload();
            }}
          }})
          .catch(() => window.setTimeout(poll, 2500));
        window.setTimeout(poll, 1200);
      }})();
    </script>
    """
    return _shell("Ablations | Pattern Server", active_nav="ablations", body_html=body)


def render_ablation_detail_page(*, job: JobRecord, project_root: Path) -> str:
    detail = load_library_ablation_job_detail(job, project_root=project_root)
    kind = (
        detail.ablation_kind
        if detail is not None and str(detail.ablation_kind or "").strip()
        else str((job.spec.metadata or {}).get("library_ablation_kind") or "").strip().lower()
    )
    kind_label = _library_ablation_kind_label(kind or "library")
    env_label = str((job.spec.metadata or {}).get("env_id") or job.environment or "unknown")
    header_title = f"{env_label.upper()} {kind_label} Ablation"
    status_label = job.status.value
    subtitle = _library_ablation_status_detail(detail) if detail is not None else _sanitize_display_text(job.message) or "Ablation results have not been written yet."
    detail_run_id = detail.run_id if detail is not None else job.job_id
    case_count = detail.total_case_count if detail is not None else 0
    seed_count = detail.total_seed_run_count if detail is not None else 0
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/ablations">Ablations</a> / {html.escape(job.job_id)}</p>
      <h1>{html.escape(header_title)}</h1>
      <p>{html.escape(subtitle)}</p>
      <div class="stats-inline">
        {_pill(status_label, status_label)}
        <span class="mini-chip">{html.escape(env_label)}</span>
        <span class="mini-chip">{html.escape(detail_run_id)}</span>
        <a class="ghost-link" href="/jobs/{html.escape(job.job_id)}">Job</a>
      </div>
    </section>
    <section class="panel compact-panel panel-static">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Progress</p>
          <h2>Run Progress</h2>
          <p class="metric-subtle">{html.escape(kind_label)}</p>
        </div>
      </div>
      {_render_required_ablation_progress(detail, job)}
      <div class="required-run-stats">
        <div class="required-run-stat"><span>Cases</span><strong>{case_count}</strong></div>
        <div class="required-run-stat"><span>Runs</span><strong>{seed_count}</strong></div>
        <div class="required-run-stat"><span>Preset</span><strong>{html.escape(detail.benchmark_preset if detail is not None else "—")}</strong></div>
      </div>
      {_render_ablation_report_links(detail)}
    </section>
    {_render_library_ablation_detail_panel(detail)}
    {_render_ablation_case_breakdown(detail)}
    """
    return _shell(f"{header_title} | Ablation", active_nav="ablations", body_html=body)


def render_environment_page(
    *,
    env_id: str,
    clients: list[ClientRecord],
    jobs: list[JobRecord],
    queue_settings: QueueSettings,
    project_root: Path,
) -> str:
    env_jobs = [job for job in jobs if _job_env_id(job) == env_id]
    counts = Counter(job.status.value for job in env_jobs)
    pooltool_attempt_limit_panel = ""
    if env_id == "pooltool":
        pooltool_attempt_limit_panel = _render_pooltool_attempt_limit_panel(
            queue_settings=queue_settings,
            next_url="/env/pooltool",
            detail_text=(
                "Recalculate PoolTool DeepPHY task success against the selected attempt limit. "
                "Tasks over the limit count as task-level failures without rerunning the evaluation."
            ),
        )
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Home</a> / env / {html.escape(env_id)}</p>
      <h1>{html.escape(env_id.upper())}</h1>
      <p>Distributed jobs, datasets, libraries, and reward outputs for the {html.escape(env_id)} environment.</p>
      <div class="hero-stats">
        <div class="stat-chip"><span class="stat-label">Jobs</span><strong class="stat-value">{len(env_jobs)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Queued</span><strong class="stat-value">{counts.get("queued", 0)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Running</span><strong class="stat-value">{sum(1 for job in env_jobs if job.status.value in ACTIVE_QUEUE_STATUSES)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Failed</span><strong class="stat-value">{counts.get("failed", 0)}</strong></div>
      </div>
      {_render_env_subnav(env_id=env_id, current_section="overview")}
    </section>
    <div class="home-main-column">
      {pooltool_attempt_limit_panel}
      {_render_supervised_importance_panel(env_id=env_id, project_root=project_root)}
      {_render_env_selection_cards(env_id=env_id)}
      {_render_recent_jobs_panel(env_jobs)}
    </div>
    """
    return _shell(f"{env_id.upper()} | Pattern Server", active_nav=f"env:{env_id}", body_html=body)


def _render_pooltool_attempt_limit_panel(
    *,
    queue_settings: QueueSettings,
    next_url: str,
    detail_text: str,
) -> str:
    attempt_limit = int(queue_settings.pooltool_attempt_limit)
    return f"""
    <section class="panel compact-panel">
      <div class="card-topline">
        <div>
          <p class="section-kicker">PoolTool DeepPHY</p>
          <h2>Attempt limit</h2>
          <p class="metric-subtle">{html.escape(detail_text)}</p>
        </div>
        <div class="stats-inline">
          <span class="mini-chip">Current {attempt_limit}</span>
        </div>
      </div>
      <form method="post" action="/ui/env/pooltool/attempt-limit" class="client-settings-form">
        <input type="hidden" name="next" value="{html.escape(next_url)}">
        <label class="metric-subtle" for="pooltool-attempt-limit">Attempt limit</label>
        <input id="pooltool-attempt-limit" name="pooltool_attempt_limit" type="range" min="1" max="15" step="1" value="{attempt_limit}" oninput="this.nextElementSibling.value=this.value">
        <output>{attempt_limit}</output>
        <button class="secondary-button" type="submit">Apply</button>
      </form>
    </section>
    """


def _render_supervised_importance_panel(*, env_id: str, project_root: Path) -> str:
    library = resolve_latest_supervised_library(project_root, env_id)
    if library is None:
        return ""
    payload = load_supervised_label_importance(library.root)
    evaluation_root, evaluation_run_id = resolve_supervised_evaluation_root(library.root)
    status_chip = '<span class="pill status-paused">Not computed</span>'
    detail = "Compute success correlations for supervised labels from the latest supervised DeepPHY evaluation."
    if payload is not None:
        label_count = len(payload.get("labels") or []) if isinstance(payload.get("labels"), list) else 0
        task_count = int(payload.get("task_count") or 0)
        computed_at = _display_time(_parse_iso_datetime(payload.get("computed_at")))
        detail = f"{label_count} labels scored across {task_count} tasks · computed {computed_at}"
        status_chip = '<span class="pill status-running">Computed</span>'
    elif evaluation_root is None:
        detail = "No completed supervised DeepPHY evaluation is available yet for the latest supervised run."
        status_chip = '<span class="pill status-paused">Waiting for eval</span>'
    eval_label = evaluation_run_id or "—"
    top_labels_html = ""
    labels = payload.get("labels") if isinstance(payload, dict) else None
    if isinstance(labels, list) and labels:
        top_labels: list[dict[str, Any]] = sorted(
            [entry for entry in labels if isinstance(entry, dict)],
            key=lambda entry: float(entry.get("sampling_weight") or 0.0),
            reverse=True,
        )[:5]
        rows = "".join(
            f"""
            <tr>
              <td>{html.escape(str(entry.get('label') or '—'))}</td>
              <td>{float(entry.get('correlation_score') or 0.0):+.3f}</td>
              <td>{float(entry.get('sampling_weight') or 1.0):.2f}</td>
            </tr>
            """
            for entry in top_labels
        )
        top_labels_html = f"""
        <div class="table-scroll compact-table">
          <table class="data-table">
            <thead>
              <tr>
                <th>Top labels</th>
                <th>Correlation</th>
                <th>Weight</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        """
    return f"""
    <section class="panel compact-panel">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Supervised label importance</p>
          <h2>LLM weighting</h2>
          <p class="metric-subtle">{html.escape(detail)}</p>
        </div>
        <div class="stats-inline">
          {status_chip}
          <span class="mini-chip">{html.escape(library.run_id)}</span>
          <span class="mini-chip">Eval {html.escape(eval_label)}</span>
        </div>
      </div>
      <div class="button-row">
        <form method="post" action="/ui/env/{html.escape(env_id)}/compute-supervised-importance">
          <input type="hidden" name="next" value="/env/{html.escape(env_id)}">
          <button class="secondary-button" type="submit">{'Recompute' if payload is not None else 'Compute'} importance</button>
        </form>
        <a class="ghost-link" href="/env/{html.escape(env_id)}/libraries">Libraries</a>
      </div>
      {top_labels_html}
    </section>
    """


def render_environment_datasets_page(*, env_id: str, datasets: list[DatasetEntry]) -> str:
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Home</a> / env / {html.escape(env_id)} / datasets</p>
      <h1>{html.escape(env_id.upper())} Datasets</h1>
      <p>Server-owned dataset manifests and split completion for {html.escape(env_id)}.</p>
      <div class="hero-stats">
        <div class="stat-chip"><span class="stat-label">Datasets</span><strong class="stat-value">{len(datasets)}</strong></div>
      </div>
      {_render_env_subnav(env_id=env_id, current_section="datasets")}
    </section>
    <section class="panel">
      <h2>Datasets</h2>
      {_render_dataset_list(datasets)}
    </section>
    """
    return _shell(f"{env_id.upper()} Datasets | Pattern Server", active_nav=f"env:{env_id}", body_html=body)


def render_environment_dataset_page(*, dataset: DatasetDetail) -> str:
    trace_count = sum(len(split.traces) for split in dataset.splits)
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Home</a> / env / {html.escape(dataset.env_id)} / <a href="/env/{dataset.env_id}/datasets">datasets</a> / {html.escape(dataset.root.name)}</p>
      <h1>{html.escape(dataset.dataset_name)}</h1>
      <p>Training, validation, and test traces stored in this dataset.</p>
      <div class="hero-stats">
        <div class="stat-chip"><span class="stat-label">Splits</span><strong class="stat-value">{len(dataset.splits)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Traces</span><strong class="stat-value">{trace_count}</strong></div>
        <div class="stat-chip"><span class="stat-label">Complete</span><strong class="stat-value">{dataset.total_completed_count}/{dataset.total_target_count}</strong></div>
      </div>
      {_render_env_subnav(env_id=dataset.env_id, current_section="datasets")}
    </section>
    <section class="panel">
      <div class="card-topline">
        <div>
          <h2>Dataset</h2>
          <p class="metric-subtle">{html.escape(" · ".join(dataset.notes) if dataset.notes else "No dataset notes recorded.")}</p>
        </div>
        <div class="asset-meta">
          <span class="mini-chip">{html.escape(dataset.dataset_id)}</span>
          <span class="mini-chip">Updated {_display_time(dataset.updated_at)}</span>
        </div>
      </div>
      <div class="dataset-info-grid">
        <div class="dataset-info-card">
          <h3>Overview</h3>
          <div class="key-value-grid">
            {_key_value_card("Environment", dataset.env_id)}
            {_key_value_card("Dataset dir", dataset.root.name)}
            {_key_value_card("Manifest", _project_relpath(dataset.manifest_path))}
            {_key_value_card("Index", _project_relpath(dataset.index_path) if dataset.index_path else "—")}
          </div>
        </div>
        <div class="dataset-info-card">
          <h3>Files</h3>
          <div class="asset-inline-links">
            <a href="/ui/files?path={quote(_project_relpath(dataset.manifest_path))}">Manifest</a>
            {f'<a href="/ui/files?path={quote(_project_relpath(dataset.index_path))}">Index</a>' if dataset.index_path else ''}
            <a href="/ui/files?path={quote(_project_relpath(dataset.root))}">Dataset root</a>
          </div>
        </div>
      </div>
    </section>
    <section class="dataset-split-stack">
      {_render_dataset_split_list(dataset.root.name, dataset.splits, env_id=dataset.env_id)}
    </section>
    """
    return _shell(
        f"{dataset.dataset_name} | Pattern Server",
        active_nav=f"env:{dataset.env_id}",
        body_html=body,
    )


def render_environment_dataset_trace_page(*, trace: DatasetTraceDetail) -> str:
    trace_payload = trace.trace_payload if isinstance(trace.trace_payload, dict) else {}
    quality_payload = trace.quality_payload if isinstance(trace.quality_payload, dict) else {}
    timesteps = trace_payload.get("timesteps") if isinstance(trace_payload.get("timesteps"), list) else []
    events = trace_payload.get("events") if isinstance(trace_payload.get("events"), list) else []
    actions = trace_payload.get("actions") if isinstance(trace_payload.get("actions"), list) else []
    outcome = trace_payload.get("outcome")
    first_timestep = timesteps[0] if timesteps else None
    last_timestep = timesteps[-1] if timesteps else None
    usable = quality_payload.get("usable")
    warnings = trace.trace_entry.warnings if trace.trace_entry is not None else []
    error_text = trace.trace_entry.error if trace.trace_entry is not None else None
    title = (
        trace.trace_entry.trace_id
        if trace.trace_entry is not None and trace.trace_entry.trace_id
        else trace.sample_name
    )
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Home</a> / env / {html.escape(trace.env_id)} / <a href="/env/{trace.env_id}/datasets">datasets</a> / <a href="/env/{trace.env_id}/datasets/{quote(trace.root.name)}">{html.escape(trace.dataset_name)}</a> / {html.escape(trace.split_name)} / {html.escape(trace.sample_name)}</p>
      <h1>{html.escape(title)}</h1>
      <p>Images, videos, trace data, and sample metadata.</p>
      <div class="hero-stats">
        <div class="stat-chip"><span class="stat-label">Split</span><strong class="stat-value">{html.escape(trace.split_name)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Media</span><strong class="stat-value">{len(trace.media_files)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Timesteps</span><strong class="stat-value">{len(timesteps)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Events</span><strong class="stat-value">{len(events)}</strong></div>
      </div>
      {_render_env_subnav(env_id=trace.env_id, current_section="datasets")}
    </section>
    <section class="panel">
      <div class="card-topline">
        <div>
          <h2>Trace Summary</h2>
          <p class="metric-subtle">{html.escape(trace.trace_entry.task_id if trace.trace_entry is not None and trace.trace_entry.task_id else "No task id recorded.")}</p>
        </div>
        <div class="asset-meta">
          {_pill((trace.trace_entry.status if trace.trace_entry is not None else "unknown").title(), trace.trace_entry.status if trace.trace_entry is not None else "unknown")}
          <span class="mini-chip">Updated {_display_time(trace.updated_at)}</span>
        </div>
      </div>
      <div class="key-value-grid">
        {_key_value_card("Dataset", trace.dataset_name)}
        {_key_value_card("Sample", trace.sample_name)}
        {_key_value_card("Trace id", trace.trace_entry.trace_id if trace.trace_entry is not None else trace_payload.get("trace_id") or "—")}
        {_key_value_card("Task id", trace.trace_entry.task_id if trace.trace_entry is not None else trace_payload.get("task_id") or "—")}
        {_key_value_card("Usable", "yes" if usable is True else "no" if usable is False else "—")}
        {_key_value_card("Actions", str(len(actions)))}
        {_key_value_card("Warnings", str(len(warnings)))}
        {_key_value_card("Extra files", str(len(trace.extra_files)))}
      </div>
      {f'<p class="error-message">{html.escape(error_text)}</p>' if error_text else ''}
    </section>
    <section class="panel">
      <h2>Media</h2>
      {_render_trace_media(trace)}
    </section>
    <div class="content-grid">
      <section class="panel">
        <h2>Sample Files</h2>
        <div class="asset-inline-links">
          {f'<a href="/ui/files?path={quote(_project_relpath(trace.trace_path))}">trace.json</a>' if trace.trace_path else ''}
          {f'<a href="/ui/files?path={quote(_project_relpath(trace.quality_path))}">quality.json</a>' if trace.quality_path else ''}
          <a href="/ui/files?path={quote(_project_relpath(trace.sample_dir))}">Sample directory</a>
          <a href="/ui/files?path={quote(_project_relpath(trace.manifest_path))}">Dataset manifest</a>
          {f'<a href="/ui/files?path={quote(_project_relpath(trace.index_path))}">Dataset index</a>' if trace.index_path else ''}
        </div>
        {_render_extra_file_list(trace.extra_files)}
      </section>
      <section class="panel">
        <h2>Structured Overview</h2>
        <div class="key-value-grid">
          {_key_value_card("Environment", trace_payload.get("environment") or trace.env_id)}
          {_key_value_card("First objects", _object_count_text(first_timestep))}
          {_key_value_card("Last objects", _object_count_text(last_timestep))}
          {_key_value_card("Initial views", _render_view_count(trace_payload))}
        </div>
      </section>
    </div>
    <section class="panel">
      <h2>Trace Data</h2>
      <div class="section-stack">
        {_json_details("Initial state", trace_payload.get("initial_state"), open_by_default=True)}
        {_json_details("Outcome", outcome)}
        {_json_details("Quality", quality_payload, open_by_default=True)}
        {_json_details("First timestep", first_timestep)}
        {_json_details("Last timestep", last_timestep)}
      </div>
    </section>
    """
    return _shell(
        f"{title} | Pattern Server",
        active_nav=f"env:{trace.env_id}",
        body_html=body,
    )


def render_environment_libraries_page(*, env_id: str, libraries: list[LibraryEntry]) -> str:
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Home</a> / env / {html.escape(env_id)} / libraries</p>
      <h1>{html.escape(env_id.upper())} Libraries</h1>
      <p>Pattern-library outputs discoverable as native v2 artifacts for {html.escape(env_id)}.</p>
      <div class="hero-stats">
        <div class="stat-chip"><span class="stat-label">Libraries</span><strong class="stat-value">{len(libraries)}</strong></div>
      </div>
      {_render_env_subnav(env_id=env_id, current_section="libraries")}
    </section>
    <section class="panel">
      <h2>Libraries</h2>
      {_render_library_list(libraries)}
    </section>
    """
    return _shell(f"{env_id.upper()} Libraries | Pattern Server", active_nav=f"env:{env_id}", body_html=body)


def render_environment_rewards_page(*, env_id: str, rewards: list[RewardResultEntry]) -> str:
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Home</a> / env / {html.escape(env_id)} / reward programs</p>
      <h1>{html.escape(env_id.upper())} Reward Programs</h1>
      <p>Reward-program experiments grouped by experiment type.</p>
      <div class="hero-stats">
        <div class="stat-chip"><span class="stat-label">Results</span><strong class="stat-value">{len(rewards)}</strong></div>
      </div>
      {_render_env_subnav(env_id=env_id, current_section="rewards")}
    </section>
    <section class="panel">
      <h2>Experiment Types</h2>
      {_reward_kind_cards(rewards=rewards, env_id=env_id)}
    </section>
    """
    return _shell(f"{env_id.upper()} Reward Programs | Pattern Server", active_nav=f"env:{env_id}", body_html=body)


def render_environment_reward_runs_page(*, env_id: str, reward_kind: str, rewards: list[RewardResultEntry]) -> str:
    label = _reward_kind_label(reward_kind)
    if env_id == "pooltool" and reward_kind == "optimization":
        return _render_reward_optimization_runs_page(env_id=env_id, label=label, rewards=rewards)
    if env_id == "pooltool" and reward_kind == "value-model":
        return _render_reward_value_model_runs_page(env_id=env_id, label=label, rewards=rewards)
    if env_id == "pooltool" and reward_kind == "value-model-sweep":
        return _render_reward_value_model_sweep_page(env_id=env_id, label=label, rewards=rewards)
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Home</a> / env / {html.escape(env_id)} / <a href="/env/{env_id}/rewards">reward programs</a> / {html.escape(label)}</p>
      <h1>{html.escape(env_id.upper())} {html.escape(label)}</h1>
      <p>{html.escape(_reward_kind_description(reward_kind))}</p>
      <div class="hero-stats">
        <div class="stat-chip"><span class="stat-label">Runs</span><strong class="stat-value">{len(rewards)}</strong></div>
      </div>
      {_render_env_subnav(env_id=env_id, current_section="rewards")}
    </section>
    <section class="panel">
      <h2>{html.escape(label)} Runs</h2>
      {_render_reward_list(rewards, env_id=env_id)}
    </section>
    """
    return _shell(f"{env_id.upper()} {label} | Pattern Server", active_nav=f"env:{env_id}", body_html=body)


def _render_env_subnav(*, env_id: str, current_section: str) -> str:
    links = [
        ("overview", f"/env/{env_id}", "Overview"),
        ("datasets", f"/env/{env_id}/datasets", "Datasets"),
        ("libraries", f"/env/{env_id}/libraries", "Libraries"),
        ("rewards", f"/env/{env_id}/rewards", "Reward programs"),
    ]
    return '<nav class="env-subnav">' + "".join(
        f'<a class="env-subnav-link{" is-active" if key == current_section else ""}" href="{href}">{html.escape(label)}</a>'
        for key, href, label in links
    ) + "</nav>"


def _render_env_selection_cards(*, env_id: str) -> str:
    cards = [
        ("Datasets", f"/env/{env_id}/datasets", "Dataset manifests and split completion."),
        ("Libraries", f"/env/{env_id}/libraries", "Pattern-library outputs and source run roots."),
        ("Reward programs", f"/env/{env_id}/rewards", "Reward optimization and value-model artifacts."),
    ]
    return '<section class="env-link-grid">' + "".join(
        f'<a class="env-link-card" href="{href}"><span class="section-label">{html.escape(title)}</span><strong>{html.escape(title)}</strong><span class="metric-subtle">{html.escape(text)}</span></a>'
        for title, href, text in cards
    ) + "</section>"


def _project_relpath(path: Path) -> str:
    for anchor in ("data", "configs", "evaluations", "state"):
        if anchor in path.parts:
            idx = path.parts.index(anchor)
            return Path(*path.parts[idx:]).as_posix()
    return path.name


def _render_dataset_list(datasets: list[DatasetEntry]) -> str:
    if not datasets:
        return '<p class="empty">No datasets found for this environment.</p>'
    rows: list[str] = []
    for item in datasets:
        rows.append(
            f"""
            <article class="asset-row">
              <div class="asset-row-top">
                <div>
                  <h3 class="asset-title"><a href="/env/{item.env_id}/datasets/{quote(item.root.name)}">{html.escape(item.dataset_name)}</a></h3>
                  <p class="metric-subtle">Updated {_display_time(item.updated_at)}</p>
                </div>
                <div class="asset-meta">
                  <span class="mini-chip">{item.total_completed_count}/{item.total_target_count} complete</span>
                </div>
              </div>
              <p class="metric-subtle">{html.escape(" · ".join(item.split_summaries) if item.split_summaries else "No split summary available")}</p>
              <div class="asset-inline-links">
                <a href="/ui/files?path={quote(_project_relpath(item.manifest_path))}">Manifest</a>
                <a href="/ui/files?path={quote(_project_relpath(item.root))}">Dataset root</a>
              </div>
            </article>
            """
        )
    return '<div class="asset-list">' + "".join(rows) + "</div>"


def _render_dataset_split_list(dataset_key: str, splits: list[DatasetSplitDetail], *, env_id: str) -> str:
    if not splits:
        return '<p class="empty">No split data found for this dataset.</p>'
    sections: list[str] = []
    for split in splits:
        sections.append(
            f"""
            <section class="panel">
              <div class="dataset-split-header">
                <div>
                  <h2>{html.escape(split.name.title())}</h2>
                  <p class="metric-subtle">{len(split.traces)} indexed traces</p>
                </div>
                <div class="asset-meta">
                  <span class="mini-chip">{split.completed_count}/{split.target_count} complete</span>
                  <span class="mini-chip">{len(split.traces)} traces</span>
                </div>
              </div>
              {_render_dataset_trace_rows(dataset_key, split, env_id=env_id)}
            </section>
            """
        )
    return "".join(sections)


def _render_dataset_trace_rows(dataset_key: str, split: DatasetSplitDetail, *, env_id: str) -> str:
    if not split.traces:
        return '<p class="empty">No traces indexed for this split.</p>'
    rows: list[str] = []
    for trace in split.traces:
        side_bits = [
            _pill(trace.status.title(), trace.status),
            f'<a class="secondary-button" href="/env/{env_id}/datasets/{quote(dataset_key)}/traces/{quote(split.name)}/{quote(trace.sample_name)}">View trace</a>',
        ]
        trace_label = trace.trace_id or trace.sample_name
        meta_bits = [trace.sample_name]
        if trace.task_id:
            meta_bits.append(trace.task_id)
        if trace.absolute_sample_index is not None:
            meta_bits.append(f"abs {trace.absolute_sample_index}")
        warning_text = f" · {len(trace.warnings)} warnings" if trace.warnings else ""
        rows.append(
            f"""
            <article class="dataset-trace-row">
              <div class="dataset-trace-main">
                <strong>{html.escape(trace_label)}</strong>
                <p class="metric-subtle">{html.escape(" · ".join(meta_bits))}{html.escape(warning_text)}</p>
                {f'<p class="error-message">{html.escape(trace.error)}</p>' if trace.error else ''}
              </div>
              <div class="dataset-trace-side">
                {' '.join(side_bits)}
              </div>
            </article>
            """
        )
    return '<div class="dataset-trace-list">' + "".join(rows) + "</div>"


def _render_trace_media(trace: DatasetTraceDetail) -> str:
    if not trace.media_files:
        return '<p class="empty">No images or videos found for this trace sample.</p>'
    cards: list[str] = []
    for path in trace.media_files:
        media_url = f"/ui/files?path={quote(_project_relpath(path))}"
        if path.suffix.lower() in {".mp4", ".webm", ".mov", ".m4v"}:
            media_html = f'<video controls preload="metadata" src="{media_url}"></video>'
        else:
            media_html = f'<img src="{media_url}" alt="{html.escape(path.name)}">'
        cards.append(
            f"""
            <article class="media-card">
              <figure>
                {media_html}
                <figcaption class="metric-subtle">{html.escape(path.name)}</figcaption>
              </figure>
              <div class="asset-inline-links">
                <a href="{media_url}">Open</a>
              </div>
            </article>
            """
        )
    return '<div class="media-grid">' + "".join(cards) + "</div>"


def _render_media_paths(paths: list[Path], *, root_dir: Path) -> str:
    if not paths:
        return '<p class="empty">No images or videos recorded for this task.</p>'
    cards: list[str] = []
    for path in paths:
        media_url = f"/ui/files?path={quote(_project_relpath(path))}"
        if path.suffix.lower() in {".mp4", ".webm", ".mov", ".m4v"}:
            media_html = f'<video controls preload="metadata" src="{media_url}"></video>'
        else:
            media_html = f'<img src="{media_url}" alt="{html.escape(path.name)}">'
        try:
            caption = path.relative_to(root_dir).as_posix()
        except ValueError:
            caption = path.name
        cards.append(
            f"""
            <article class="media-card">
              <figure>
                {media_html}
                <figcaption class="metric-subtle">{html.escape(caption)}</figcaption>
              </figure>
              <div class="asset-inline-links">
                <a href="{media_url}">Open</a>
              </div>
            </article>
            """
        )
    return '<div class="media-grid">' + "".join(cards) + "</div>"


def _render_extra_file_list(paths: list[Path]) -> str:
    if not paths:
        return '<p class="empty">No extra files in this sample.</p>'
    rows: list[str] = []
    for path in paths:
        rows.append(
            f"""
            <article class="asset-row">
              <h3 class="asset-title">{html.escape(path.name)}</h3>
              <div class="asset-inline-links">
                <a href="/ui/files?path={quote(_project_relpath(path))}">Open</a>
              </div>
            </article>
            """
        )
    return '<div class="asset-list">' + "".join(rows) + "</div>"


def _key_value_card(label: str, value: Any) -> str:
    display_value = "—" if value is None else str(value)
    if not display_value.strip():
        display_value = "—"
    return (
        '<div class="key-value-card">'
        f'<span class="key-value-label">{html.escape(label)}</span>'
        f'<span class="key-value-value">{html.escape(display_value)}</span>'
        "</div>"
    )


def _object_count_text(timestep: Any) -> str:
    if not isinstance(timestep, dict):
        return "—"
    objects = timestep.get("objects")
    if not isinstance(objects, list):
        return "—"
    return str(len(objects))


def _render_view_count(trace_payload: dict[str, Any]) -> str:
    initial_state = trace_payload.get("initial_state")
    if not isinstance(initial_state, dict):
        return "—"
    render_views = initial_state.get("render_views")
    if isinstance(render_views, list):
        return str(len(render_views))
    return "1" if any(key in initial_state for key in {"render_camera_state", "video_fps", "media_max_side"}) else "—"


def _json_details(title: str, payload: Any, *, open_by_default: bool = False) -> str:
    if payload in (None, {}, []):
        body = '<p class="empty" style="padding: 0 0.75rem 0.75rem;">No data recorded.</p>'
    else:
        body = f'<pre class="code-block">{html.escape(_pretty_json(_trim_json(payload)))}</pre>'
    return f'<details class="data-details"{" open" if open_by_default else ""}><summary>{html.escape(title)}</summary>{body}</details>'


def _job_evaluation_run_dir(job: JobRecord, *, project_root: Path) -> Path | None:
    if str(job.spec.action or "").strip() == "automatic-launch":
        automatic_eval_root = _automatic_deepphy_eval_root(job, project_root=project_root)
        if automatic_eval_root is not None:
            return automatic_eval_root
    for key in ("evaluation_run_dir", "output_path", "run_dir"):
        raw = str(job.spec.metadata.get(key) or "").strip()
        if not raw:
            continue
        path = _normalize_project_path(raw, project_root=project_root)
        if path is not None and path.exists():
            return path
    return None


def _normalize_project_path(raw: str, *, project_root: Path) -> Path | None:
    value = str(raw or "").strip()
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate.resolve()
        work_root_mappings = [
            ("/Services/Work/pattern-learning-server/", project_root),
            ("/Services/Work/pattern-learning-client/", project_root.parent / "pattern-learning-client"),
        ]
        normalized_value = value.replace("\\", "/")
        for marker, target_root in work_root_mappings:
            if marker not in normalized_value:
                continue
            suffix = normalized_value.split(marker, 1)[1].lstrip("/")
            mapped = (target_root / suffix).resolve()
            if mapped.exists():
                return mapped
        stack_prefix = "/opt/pattern-learning-stack/"
        if value.startswith(stack_prefix):
            mapped = (project_root / value[len(stack_prefix):]).resolve()
            if mapped.exists():
                return mapped
        return candidate
    return (project_root / candidate).resolve()


def _evaluation_task_list(job: JobRecord, *, project_root: Path) -> list[EvaluationTaskDetail]:
    run_dir = _job_evaluation_run_dir(job, project_root=project_root)
    if run_dir is None:
        return []
    task_results = sorted(run_dir.glob("tasks/*/*/task_result.json"))
    details: list[EvaluationTaskDetail] = []
    for result_path in task_results:
        detail = _load_evaluation_task_detail(job, result_path=result_path, project_root=project_root)
        if detail is not None:
            details.append(detail)
    return details


def get_evaluation_task_detail(
    job: JobRecord,
    task_id: str,
    *,
    project_root: Path,
) -> EvaluationTaskDetail | None:
    normalized = str(task_id or "").strip()
    if not normalized:
        return None
    for detail in _evaluation_task_list(job, project_root=project_root):
        if detail.task_id == normalized:
            return detail
    return None


def _load_evaluation_task_detail(
    job: JobRecord,
    *,
    result_path: Path,
    project_root: Path,
) -> EvaluationTaskDetail | None:
    if not result_path.is_file():
        return None
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    task_dir = result_path.parent
    attempts: list[EvaluationTaskAttempt] = []
    records = payload.get("records")
    if isinstance(records, list):
        for item in records:
            if not isinstance(item, dict):
                continue
            trace_path = _normalize_project_path(str(item.get("trace_path") or ""), project_root=project_root)
            evidence_path = trace_path.with_name("trace.evidence.json") if trace_path is not None else None
            if evidence_path is not None and not evidence_path.exists():
                evidence_path = None
            evidence_payload = _load_json_preview(evidence_path) if evidence_path is not None else None
            try:
                index = int(item.get("index") or 0)
            except (TypeError, ValueError):
                index = 0
            duration_value = item.get("duration_seconds")
            try:
                duration_seconds = float(duration_value) if duration_value is not None else None
            except (TypeError, ValueError):
                duration_seconds = None
            success_value = item.get("success")
            attempts.append(
                EvaluationTaskAttempt(
                    index=index,
                    success=success_value if isinstance(success_value, bool) else None,
                    duration_seconds=duration_seconds,
                    terminal_status=str(item.get("terminal_status") or "").strip(),
                    evidence_status=str(item.get("evidence_status") or "").strip(),
                    trace_id=str(item.get("trace_id") or "").strip(),
                    trace_path=trace_path if trace_path is not None and trace_path.exists() else None,
                    evidence_path=evidence_path,
                    evidence_payload=evidence_payload,
                    prompt=_json_like_value(item.get("prompt")),
                    parsed_action=_json_like_value(item.get("parsed_action")),
                    raw_model_output=str(item.get("raw_model_output") or "").strip(),
                    trace_description=_json_like_value(item.get("trace_description")),
                    native_metrics=_json_like_value(item.get("native_metrics")),
                    router=_json_like_value(item.get("router")),
                )
            )
    media_files = sorted(
        [
            path
            for path in task_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".webm", ".mov", ".m4v"}
        ]
    )
    extra_files = sorted(
        [
            path
            for path in task_dir.rglob("*")
            if path.is_file() and path not in media_files and path != result_path
        ]
    )
    completed_at = None
    raw_completed_at = payload.get("completed_at")
    if isinstance(raw_completed_at, str) and raw_completed_at.strip():
        try:
            completed_at = datetime.fromisoformat(raw_completed_at)
        except ValueError:
            completed_at = None
    success_value = payload.get("success")
    return EvaluationTaskDetail(
        job_id=job.job_id,
        task_id=str(payload.get("task_id") or task_dir.name),
        env_id=str(payload.get("env_id") or _job_env_id(job)),
        task_dir=task_dir,
        result_path=result_path,
        payload=payload,
        attempts=sorted(attempts, key=lambda item: item.index),
        media_files=media_files,
        extra_files=extra_files,
        success=success_value if isinstance(success_value, bool) else None,
        completed_at=completed_at,
    )


def _json_like_value(value: Any) -> Any:
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    return {}


def _load_json_preview(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _trim_json(value: Any, *, depth: int = 0) -> Any:
    if depth >= 3:
        if isinstance(value, dict):
            return f"... {len(value)} keys"
        if isinstance(value, list):
            return f"... {len(value)} items"
        return value
    if isinstance(value, dict):
        trimmed: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 12:
                trimmed["..."] = f"{len(value) - 12} more keys"
                break
            trimmed[str(key)] = _trim_json(item, depth=depth + 1)
        return trimmed
    if isinstance(value, list):
        if not value:
            return []
        subset = [_trim_json(item, depth=depth + 1) for item in value[:8]]
        if len(value) > 8:
            subset.append(f"... {len(value) - 8} more items")
        return subset
    return value


def _pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=True)


def _render_library_list(libraries: list[LibraryEntry]) -> str:
    if not libraries:
        return '<p class="empty">No pattern libraries found for this environment.</p>'
    rows: list[str] = []
    for item in libraries:
        created_text = _display_time(item.created_at) if item.created_at is not None else "—"
        rows.append(
            f"""
            <article class="asset-row">
              <div class="asset-row-top">
                <div>
                  <h3 class="asset-title">{html.escape(item.run_id)}</h3>
                  <p class="metric-subtle">Created {created_text} · Updated {_display_time(item.updated_at)}</p>
                </div>
                <div class="asset-meta">
                  <span class="mini-chip">{html.escape(item.workflow_kind)}</span>
                  <span class="mini-chip">{item.detector_count} detectors</span>
                </div>
              </div>
              <div class="asset-inline-links">
                <a href="/ui/files?path={quote(_project_relpath(item.library_path))}">Pattern library</a>
                <a href="/ui/files?path={quote(_project_relpath(item.root))}">Run root</a>
              </div>
            </article>
            """
        )
    return '<div class="asset-list">' + "".join(rows) + "</div>"


def _render_reward_list(rewards: list[RewardResultEntry], *, env_id: str) -> str:
    if env_id != "pooltool":
        return '<p class="empty">Reward-program results are only available for PoolTool.</p>'
    if not rewards:
        return '<p class="empty">No reward-program results found for PoolTool.</p>'
    rows: list[str] = []
    for item in rewards:
        source_bits = [bit for bit in [item.source_run_kind, item.source_run_id] if bit]
        source_text = " / ".join(source_bits) if source_bits else "—"
        model_text = f" · Model: {item.model_name}" if item.model_name else ""
        rows.append(
            f"""
            <article class="asset-row">
              <div class="asset-row-top">
                <div>
                  <h3 class="asset-title">{html.escape(item.run_id)}</h3>
                  <p class="metric-subtle">Updated {_display_time(item.updated_at)}</p>
                </div>
                <div class="asset-meta">
                  <span class="mini-chip">{html.escape(item.reward_kind)}</span>
                  <span class="mini-chip">{html.escape(item.headline)}</span>
                </div>
              </div>
              <p class="metric-subtle">Source: {html.escape(source_text)}{html.escape(model_text)}</p>
              <div class="asset-inline-links">
                <a href="/ui/files?path={quote(_project_relpath(item.summary_path))}">Summary</a>
                <a href="/ui/files?path={quote(_project_relpath(item.root))}">Run root</a>
              </div>
            </article>
            """
        )
    return '<div class="asset-list">' + "".join(rows) + "</div>"


def _artifact_entries(job: JobRecord, *, project_root: Path) -> list[dict[str, str]]:
    seen: set[str] = set()
    relpaths: list[str] = []
    job_log_rel = f"state/jobs/{job.job_id}/job.log"
    relpaths.append(job_log_rel)
    seen.add(job_log_rel)
    for relpath in job.artifact_relpaths:
        rel = str(relpath).strip().lstrip("/")
        if rel and rel not in seen:
            relpaths.append(rel)
            seen.add(rel)
    for key in ("output_path", "run_dir", "evaluation_run_dir"):
        raw = str(job.spec.metadata.get(key) or "").strip()
        if not raw:
            continue
        candidate = Path(raw)
        if candidate.is_absolute():
            try:
                rel = candidate.resolve().relative_to(project_root.resolve()).as_posix()
            except ValueError:
                continue
        else:
            rel = candidate.as_posix().lstrip("/")
        if rel and rel not in seen:
            relpaths.append(rel)
            seen.add(rel)
    if str(job.spec.action or "").startswith("pooltool-reward-"):
        reward_detail = load_reward_job_detail(job, project_root=project_root)
        if reward_detail is not None:
            try:
                rel = reward_detail.root.resolve().relative_to(project_root.resolve()).as_posix()
            except ValueError:
                rel = ""
            if rel and rel not in seen:
                relpaths.append(rel)
                seen.add(rel)
    entries: list[dict[str, str]] = []
    for rel in relpaths[:18]:
        path = (project_root / rel).resolve()
        if not str(path).startswith(str(project_root.resolve())) or not path.exists():
            continue
        preview = ""
        kind = "directory" if path.is_dir() else "file"
        if path.is_dir():
            children = sorted(item.name for item in path.iterdir())[:12]
            preview = "\n".join(children)
        elif path.suffix.lower() in TEXT_PREVIEW_SUFFIXES and path.stat().st_size <= MAX_PREVIEW_BYTES:
            try:
                preview = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                preview = ""
        entries.append(
            {
                "relpath": rel,
                "kind": kind,
                "href": f"/ui/files?path={quote(rel)}",
                "preview": preview[:4000],
            }
        )
    return entries


def _resolve_job_run_dir(job: JobRecord, *, project_root: Path) -> Path | None:
    def _learning_root_progress_key(root: Path) -> tuple[float, int, int]:
        if not root.is_dir():
            return (0.0, 0, 0)
        run_payload = _load_json_dict(root / "run.json")
        label_dirs = _unique_label_dirs(root)
        completed_label_results = sum(
            1 for label_dir in label_dirs if (label_dir / "label_result.json").is_file()
        )
        payload_completed_labels = _safe_int(run_payload.get("completed_label_count")) or 0
        if label_dirs:
            # Legacy/stale source roots can keep an old run.json that still claims
            # the final completed count even though only a subset of label results
            # exists in that root. Prefer the on-disk label results so resumed
            # .__resume_sync__ roots win when they hold the real finished state.
            completed_labels = completed_label_results
        else:
            completed_labels = max(payload_completed_labels, completed_label_results)
        total_labels = max(
            _safe_int(run_payload.get("primitive_label_count")) or 0,
            len(label_dirs),
        )
        latest_mtime = 0.0
        for candidate in (root / "run.json",):
            try:
                if candidate.is_file():
                    latest_mtime = max(latest_mtime, candidate.stat().st_mtime)
            except OSError:
                pass
        return (latest_mtime, completed_labels, total_labels)

    action = str(job.spec.action or "").strip()
    metadata = dict(job.spec.metadata or {})
    request_payload = dict(metadata.get("request") or {})
    expected_run_name = str(request_payload.get("run_timestamp") or "").strip()
    if not expected_run_name:
        run_dir_hint = str(metadata.get("run_dir") or "").strip()
        if run_dir_hint:
            expected_run_name = Path(run_dir_hint).name

    for key in ("run_dir", "output_path", "evaluation_run_dir"):
        raw = str(metadata.get(key) or "").strip()
        if not raw:
            continue
        normalized = _normalize_project_path(raw, project_root=project_root)
        if normalized is None or not normalized.exists():
            continue
        candidate = normalized.parent if normalized.is_file() else normalized
        if candidate.is_dir():
            resume_sync = candidate.parent / f"{candidate.name}.__resume_sync__{job.job_id}"
            if resume_sync.is_dir():
                if _is_learning_job(job):
                    canonical_key = _learning_root_progress_key(candidate)
                    resume_key = _learning_root_progress_key(resume_sync)
                    if (
                        resume_key[1] > canonical_key[1]
                        or (resume_key[1] == canonical_key[1] and resume_key[2] > canonical_key[2])
                        or (
                            resume_key[1] == canonical_key[1]
                            and resume_key[2] == canonical_key[2]
                            and resume_key[0] >= canonical_key[0]
                        )
                    ):
                        return resume_sync
                else:
                    return resume_sync
        return candidate

    for relpath in job.artifact_relpaths:
        rel = str(relpath or "").strip().lstrip("/")
        if not rel:
            continue
        if not (rel.startswith("data/patterns/") or rel.startswith("data/evaluations/")):
            continue
        rel_lower = rel.replace("\\", "/").lower()
        if action == "automatic-launch" and "/automatic/" not in rel_lower:
            continue
        if expected_run_name:
            candidate_name = Path(rel).name
            if (
                expected_run_name not in rel
                and candidate_name != expected_run_name
                and not candidate_name.startswith(f"{expected_run_name}.__resume_sync__")
            ):
                continue
        candidate = (project_root / rel).resolve()
        if candidate.is_dir():
            resume_sync = candidate.parent / f"{candidate.name}.__resume_sync__{job.job_id}"
            if resume_sync.is_dir():
                return resume_sync
            return candidate

    if str(job.spec.action or "").startswith("pooltool-reward-"):
        detail = load_reward_job_detail(job, project_root=project_root)
        return detail.root if detail is not None else None
    return None


def _render_xy_line_chart(
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[dict[str, object]],
) -> str:
    plot_config = _shared_plot_style_config()
    rc_params = plot_config.get("rcParams") if isinstance(plot_config.get("rcParams"), dict) else {}
    label_settings = plot_config.get("labels") if isinstance(plot_config.get("labels"), dict) else {}

    def _cfg_size(key: str, default: float, scale: float, floor: float) -> float:
        raw = rc_params.get(key, default)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = float(default)
        return max(floor, value * scale)

    serif_candidates = rc_params.get("font.serif") if isinstance(rc_params.get("font.serif"), list) else []
    svg_font_family = html.escape(str(serif_candidates[0] if serif_candidates else "serif"))
    tick_font_size = _cfg_size("xtick.labelsize", 8.0, 1.55, 12.0)
    axis_label_font_size = _cfg_size("axes.labelsize", 9.0, 1.65, 14.0)
    legend_font_size = _cfg_size("legend.fontsize", 6.0, 2.0, 12.0)
    empty_font_size = _cfg_size("font.size", 9.0, 1.8, 15.0)
    axis_label_weight = "600"
    axis_label_y = height_label_y = None
    prepared: list[dict[str, object]] = []
    min_x: float | None = None
    max_x: float | None = None
    min_y: float | None = None
    max_y: float | None = None
    for item in series:
        label = str(item.get("label") or "").strip()
        points_raw = item.get("points")
        band_raw = item.get("band")
        if not label or not isinstance(points_raw, list):
            continue
        points: list[tuple[float, float]] = []
        band_points: list[tuple[float, float, float]] = []
        for raw_point in points_raw:
            if (
                not isinstance(raw_point, (list, tuple))
                or len(raw_point) != 2
            ):
                continue
            try:
                x_value = float(raw_point[0])
                y_value = float(raw_point[1])
            except (TypeError, ValueError):
                continue
            points.append((x_value, y_value))
            min_x = x_value if min_x is None else min(min_x, x_value)
            max_x = x_value if max_x is None else max(max_x, x_value)
            min_y = y_value if min_y is None else min(min_y, y_value)
            max_y = y_value if max_y is None else max(max_y, y_value)
        if isinstance(band_raw, list):
            for raw_band in band_raw:
                if not isinstance(raw_band, (list, tuple)) or len(raw_band) != 3:
                    continue
                try:
                    band_x = float(raw_band[0])
                    band_low = float(raw_band[1])
                    band_high = float(raw_band[2])
                except (TypeError, ValueError):
                    continue
                band_points.append((band_x, band_low, band_high))
                min_y = band_low if min_y is None else min(min_y, band_low)
                max_y = band_high if max_y is None else max(max_y, band_high)
        if points:
            prepared.append(
                {
                    "label": label,
                    "color": str(item.get("color") or "#124f49"),
                    "points": points,
                    "band": band_points,
                    "dasharray": str(item.get("dasharray") or ""),
                }
            )
    if not prepared or min_x is None or max_x is None or min_y is None or max_y is None:
        width = 880
        height = 390
        left = 68
        right = 28
        top = 24
        bottom = 58
        plot_width = width - left - right
        plot_height = height - top - bottom
        return f"""
        <div class="chart-shell">
          <svg viewBox="0 0 {width} {height}" class="detail-chart" role="img" aria-label="{html.escape(title)}">
            <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="rgba(255,255,255,0.55)" stroke="rgba(87,104,125,0.18)" />
            <text x="{width/2:.1f}" y="{height/2:.1f}" text-anchor="middle" fill="#617182" font-size="{empty_font_size:.1f}" font-family="{svg_font_family}">No chart data yet</text>
            <text x="{left + plot_width / 2:.1f}" y="{height - 16:.1f}" text-anchor="middle" fill="#2c3744" font-size="{axis_label_font_size:.1f}" font-weight="{axis_label_weight}" font-family="{svg_font_family}">{html.escape(x_label)}</text>
            <text x="20" y="{top + plot_height / 2:.1f}" text-anchor="middle" fill="#2c3744" font-size="{axis_label_font_size:.1f}" font-weight="{axis_label_weight}" font-family="{svg_font_family}" transform="rotate(-90 20,{top + plot_height / 2:.1f})">{html.escape(y_label)}</text>
          </svg>
        </div>
        """
    if min_x == max_x:
        max_x += 1.0
    uses_percent_axis = "%" in y_label or "success" in y_label.lower()
    if uses_percent_axis:
        min_y = 0.0
        max_y = 100.0
    elif min_y == max_y:
        max_y += 1.0
    width = 880
    height = 390
    left = 68
    right = 28
    top = 24
    bottom = 58
    plot_width = width - left - right
    plot_height = height - top - bottom

    def sx(value: float) -> float:
        return left + ((value - min_x) / (max_x - min_x)) * plot_width

    def sy(value: float) -> float:
        return top + (1.0 - ((value - min_y) / (max_y - min_y))) * plot_height

    grid: list[str] = []
    y_tick_values = (
        [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
        if uses_percent_axis
        else [min_y + (max_y - min_y) * fraction for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)]
    )
    for y_value in y_tick_values:
        y_pos = sy(y_value)
        grid.append(
            f'<line x1="{left}" y1="{y_pos:.1f}" x2="{width-right}" y2="{y_pos:.1f}" stroke="rgba(87,104,125,0.16)" stroke-width="1" />'
        )
        grid.append(
            f'<text x="{left-8}" y="{y_pos+4:.1f}" text-anchor="end" fill="#617182" font-size="{tick_font_size:.1f}" font-family="{svg_font_family}">{html.escape(f"{y_value:.0f}" if uses_percent_axis else f"{y_value:.2f}")}</text>'
        )
    legend_rows: list[tuple[str, str]] = []
    paths = []
    for index, item in enumerate(prepared):
        points = item["points"]
        color = str(item["color"])
        dasharray = str(item.get("dasharray") or "").strip()
        band_points = item.get("band") if isinstance(item.get("band"), list) else []
        if band_points:
            upper = " ".join(f"{sx(x):.1f},{sy(high):.1f}" for x, _low, high in band_points)
            lower = " ".join(f"{sx(x):.1f},{sy(low):.1f}" for x, low, _high in reversed(band_points))
            polygon = f"{upper} {lower}".strip()
            if polygon:
                paths.append(
                    f'<polygon points="{polygon}" fill="{color}" fill-opacity="0.12" stroke="none" />'
                )
        line = " ".join(
            f'{"M" if point_index == 0 else "L"} {sx(x):.1f} {sy(y):.1f}'
            for point_index, (x, y) in enumerate(points)
                )
        dash_attr = f' stroke-dasharray="{html.escape(dasharray)}"' if dasharray else ""
        paths.append(f'<path d="{line}" fill="none" stroke="{color}" stroke-width="2.5"{dash_attr} />')
        legend_rows.append((color, str(item["label"])))
    x_labels = []
    x_tick_count = min(6, max(2, int(max_x - min_x) + 1))
    for tick_index in range(x_tick_count):
        ratio = tick_index / max(x_tick_count - 1, 1)
        x_value = min_x + (max_x - min_x) * ratio
        x_pos = sx(x_value)
        x_labels.append(
            f'<text x="{x_pos:.1f}" y="{height-12}" text-anchor="middle" fill="#617182" font-size="{tick_font_size:.1f}" font-family="{svg_font_family}">{html.escape(f"{x_value:.0f}")}</text>'
        )
    legend_box = ""
    if legend_rows:
        legend_line_height = max(20, int(round(legend_font_size + 8)))
        legend_box_width = min(340, max(210, max(len(label) for _color, label in legend_rows) * int(max(7, round(legend_font_size * 0.62))) + 44))
        legend_box_height = 12 + legend_line_height * len(legend_rows)
        legend_parts = [
            f'<rect x="{left + 10}" y="{top + 10}" width="{legend_box_width}" height="{legend_box_height}" rx="8" fill="rgba(255,255,255,0.92)" stroke="rgba(87,104,125,0.22)" />'
        ]
        for row_index, (color, label) in enumerate(legend_rows):
            row_y = top + 26 + row_index * legend_line_height
            legend_parts.append(
                f'<line x1="{left + 22}" y1="{row_y:.1f}" x2="{left + 42}" y2="{row_y:.1f}" stroke="{html.escape(color)}" stroke-width="3" />'
            )
            legend_parts.append(
                f'<text x="{left + 50}" y="{row_y + 4:.1f}" fill="#2c3744" font-size="{legend_font_size:.1f}" font-family="{svg_font_family}">{html.escape(label)}</text>'
            )
        legend_box = "".join(legend_parts)
    return f"""
    <div class="chart-shell">
      <svg viewBox="0 0 {width} {height}" class="detail-chart" role="img" aria-label="{html.escape(title)}">
        <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="rgba(255,255,255,0.55)" stroke="rgba(87,104,125,0.18)" />
        {''.join(grid)}
        {''.join(paths)}
        {legend_box}
        {''.join(x_labels)}
        <text x="{left + plot_width / 2:.1f}" y="{height - 16:.1f}" text-anchor="middle" fill="#2c3744" font-size="{axis_label_font_size:.1f}" font-weight="{axis_label_weight}" font-family="{svg_font_family}">{html.escape(x_label)}</text>
        <text x="20" y="{top + plot_height / 2:.1f}" text-anchor="middle" fill="#2c3744" font-size="{axis_label_font_size:.1f}" font-weight="{axis_label_weight}" font-family="{svg_font_family}" transform="rotate(-90 20,{top + plot_height / 2:.1f})">{html.escape(y_label)}</text>
      </svg>
    </div>
    """


def _reward_result_links(trace_path: Path | None, reward_path: Path | None, *, project_root: Path) -> str:
    links: list[str] = []
    for label, path in (("Trace", trace_path), ("Reward", reward_path)):
        if path is None or not path.exists():
            continue
        try:
            rel = path.resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError:
            continue
        links.append(f'<a class="ghost-link" href="/ui/files?path={quote(rel)}">{html.escape(label)}</a>')
    return " ".join(links) if links else "—"


def _render_reward_optimization_panel(detail: RewardOptimizationDetail, *, project_root: Path) -> str:
    mean_reward_series: list[dict[str, object]] = []
    success_series: list[dict[str, object]] = []
    palette = {
        "synthesized": "#124f49",
        "binary": "#a54d2d",
    }
    grouped: dict[str, list[tuple[float, float]]] = {}
    grouped_success: dict[str, list[tuple[float, float]]] = {}
    grouped_reward_band: dict[str, list[tuple[float, float, float]]] = {}
    grouped_success_band: dict[str, list[tuple[float, float, float]]] = {}
    for row in detail.aggregates:
        grouped.setdefault(row.reward_kind, []).append((float(row.budget), float(row.mean_reward)))
        grouped_success.setdefault(row.reward_kind, []).append((float(row.budget), float(row.success_rate * 100.0)))
        grouped_reward_band.setdefault(row.reward_kind, []).append(
            (
                float(row.budget),
                float(row.mean_reward - row.mean_reward_std),
                float(row.mean_reward + row.mean_reward_std),
            )
        )
        success_mean = float(row.success_rate * 100.0)
        success_std = float(row.success_rate_std * 100.0)
        grouped_success_band.setdefault(row.reward_kind, []).append(
            (
                float(row.budget),
                max(0.0, success_mean - success_std),
                min(100.0, success_mean + success_std),
            )
        )
    reward_order = ("synthesized", "binary")
    reward_kinds = sorted(
        grouped.keys(),
        key=lambda item: (reward_order.index(item) if item in reward_order else len(reward_order), item),
    )
    for kind in reward_kinds:
        points = grouped[kind]
        mean_reward_series.append(
            {
                "label": kind.title(),
                "color": palette.get(kind, "#355c7d"),
                "points": sorted(points),
                "band": sorted(grouped_reward_band.get(kind, [])),
                "dasharray": "6 4" if kind == "binary" else "",
            }
        )
    success_order = ("synthesized", "binary")
    success_kinds = sorted(
        grouped_success.keys(),
        key=lambda item: (success_order.index(item) if item in success_order else len(success_order), item),
    )
    for kind in success_kinds:
        points = grouped_success[kind]
        success_series.append(
            {
                "label": kind.title(),
                "color": palette.get(kind, "#355c7d"),
                "points": sorted(points),
                "band": sorted(grouped_success_band.get(kind, [])),
                "dasharray": "6 4" if kind == "binary" else "",
            }
        )
    top_outcomes = sorted(detail.outcomes, key=lambda item: (item.best_reward, item.success), reverse=True)[:12]
    top_outcome_rows = (
        "".join(
            f'<tr><td>{html.escape(item.scene_id)}</td><td>{item.budget}</td><td>{html.escape(item.reward_kind)}</td><td>{item.seed}</td><td>{item.best_reward:.3f}</td><td>{"yes" if item.success else "no"}</td><td>{_reward_result_links(item.trace_path, item.reward_path, project_root=project_root)}</td></tr>'
            for item in top_outcomes
        )
        if top_outcomes
        else '<tr><td colspan="7" class="empty">No outcome rows saved yet.</td></tr>'
    )
    overview_cards = [
        ("Scenes", str(detail.scene_count), f"{detail.seed_count} seeds per budget"),
        ("Budgets", ", ".join(str(item) for item in detail.budgets) or "—", detail.source_run_id or "—"),
        ("LM", detail.model_name or "(router default)", "DSL synthesis model"),
        ("Mode", "Synthesized DSL" if detail.synthesized else "Provided DSL", detail.source_run_kind or "—"),
    ]
    overview_cards_html = "".join(
        f'<article class="job-overview-card"><span class="job-overview-kicker">{html.escape(label)}</span><strong class="job-overview-value">{html.escape(value)}</strong><span class="job-overview-meta">{html.escape(meta)}</span></article>'
        for label, value, meta in overview_cards
    )
    aggregate_rows_html = "".join(
        f'<tr><td>{row.budget}</td><td>{html.escape(row.reward_kind)}</td><td>{row.scene_count}</td><td>{row.seed_count}</td><td>{row.success_rate*100:.1f}% ± {row.success_rate_std*100:.1f}%</td><td>{row.mean_reward:.3f} ± {row.mean_reward_std:.3f}</td></tr>'
        for row in detail.aggregates
    )
    goal_list_html = (
        '<ul class="asset-bullets">' + "".join(f'<li>{html.escape(item)}</li>' for item in detail.goal_lines) + "</ul>"
        if detail.goal_lines
        else '<p class="metric-subtle">No natural-language goal saved.</p>'
    )
    synthesis_html = (
        f'<section class="panel"><h2>Synthesis response</h2><pre class="log-block">{html.escape(detail.synthesis_text)}</pre></section>'
        if detail.synthesis_text
        else ""
    )
    return f"""
    <section class="panel">
      <div class="card-topline">
        <div>
          <h2>Reward Optimization</h2>
          <p class="metric-subtle">Direct action search under dense and binary reward variants, following the reward-program optimization experiment from the paper.</p>
        </div>
      </div>
      <div class="job-overview-grid">
        {overview_cards_html}
      </div>
      <div class="content-grid">
        <section class="panel compact-panel">
          <h3>NL goals</h3>
          {goal_list_html}
          <h3>Reward DSL</h3>
          <pre class="code-block">{html.escape(detail.program_text)}</pre>
        </section>
        <section class="panel compact-panel">
          <h3>Budget aggregates</h3>
          <div class="scroll-table compact-scroll-table">
            <table class="data-table compact-table">
              <thead><tr><th>Budget</th><th>Reward</th><th>Scenes</th><th>Seeds</th><th>Success</th><th>Mean reward</th></tr></thead>
              <tbody>
                {aggregate_rows_html}
              </tbody>
            </table>
          </div>
          <p class="metric-subtle">Bands and ± values show standard deviation across scene/seed outcomes at each budget.</p>
        </section>
      </div>
      {_render_xy_line_chart(title='Mean reward by budget', x_label='Budget', y_label='Mean reward', series=mean_reward_series)}
      {_render_xy_line_chart(title='Success rate by budget', x_label='Budget', y_label='Success rate (%)', series=success_series)}
      <section class="panel compact-panel">
        <h3>Top outcomes</h3>
        <div class="scroll-table compact-scroll-table">
          <table class="data-table compact-table">
            <thead><tr><th>Scene</th><th>Budget</th><th>Reward</th><th>Seed</th><th>Best reward</th><th>Solved</th><th>Artifacts</th></tr></thead>
            <tbody>
              {top_outcome_rows}
            </tbody>
          </table>
        </div>
      </section>
      {synthesis_html}
    </section>
    """


def _render_reward_value_model_panel(detail: RewardValueModelDetail, *, project_root: Path) -> str:
    curve_reward = [{
        "label": "Validation reward",
        "color": "#124f49",
        "points": [(float(item.epoch), float(item.validation_mean_reward)) for item in detail.eval_curve],
        "band": [
            (
                float(item.epoch),
                float(item.validation_mean_reward - item.validation_mean_reward_std),
                float(item.validation_mean_reward + item.validation_mean_reward_std),
            )
            for item in detail.eval_curve
        ],
    }]
    curve_success = [{
        "label": "Validation success",
        "color": "#a54d2d",
        "points": [(float(item.epoch), float(item.validation_success_rate * 100.0)) for item in detail.eval_curve],
        "band": [
            (
                float(item.epoch),
                max(0.0, float(item.validation_success_rate * 100.0 - item.validation_success_rate_std * 100.0)),
                min(100.0, float(item.validation_success_rate * 100.0 + item.validation_success_rate_std * 100.0)),
            )
            for item in detail.eval_curve
        ],
    }]
    curve_loss = [{"label": "Train loss", "color": "#355c7d", "points": [(float(item.epoch), float(item.train_loss)) for item in detail.eval_curve]}]
    top_validation = sorted(detail.validation_outcomes, key=lambda item: (item.reward, item.success), reverse=True)[:12]
    top_validation_rows = (
        "".join(
            f'<tr><td>{item.epoch}</td><td>{html.escape(item.scene_id)}</td><td>{item.reward:.3f}</td><td>{"yes" if item.success else "no"}</td><td>{_reward_result_links(item.trace_path, item.reward_path, project_root=project_root)}</td></tr>'
            for item in top_validation
        )
        if top_validation
        else '<tr><td colspan="5" class="empty">No validation rollout rows saved yet.</td></tr>'
    )
    overview_cards = [
        ("Train samples", str(detail.train_sample_count), f"{detail.train_scene_count} scenes × {detail.train_actions_per_scene} actions"),
        ("Validation", str(detail.validation_scene_count), f"{detail.validation_candidates_per_scene} candidates per scene"),
        ("Best reward", f"{detail.best_validation_mean_reward:.3f}", f"{detail.best_validation_success_rate*100:.1f}% success"),
        ("LM", detail.model_name or "(router default)", "DSL synthesis model"),
    ]
    overview_cards_html = "".join(
        f'<article class="job-overview-card"><span class="job-overview-kicker">{html.escape(label)}</span><strong class="job-overview-value">{html.escape(value)}</strong><span class="job-overview-meta">{html.escape(meta)}</span></article>'
        for label, value, meta in overview_cards
    )
    goal_list_html = (
        '<ul class="asset-bullets">' + "".join(f'<li>{html.escape(item)}</li>' for item in detail.goal_lines) + "</ul>"
        if detail.goal_lines
        else '<p class="metric-subtle">No natural-language goal saved.</p>'
    )
    epoch_rows_html = (
        "".join(
            f'<tr><td>{point.epoch}</td><td>{point.train_loss:.4f}</td><td>{point.validation_mean_reward:.3f} ± {point.validation_mean_reward_std:.3f}</td><td>{point.validation_success_rate*100:.1f}% ± {point.validation_success_rate_std*100:.1f}%</td></tr>'
            for point in detail.eval_curve
        )
        if detail.eval_curve
        else '<tr><td colspan="4" class="empty">No curve points saved yet.</td></tr>'
    )
    synthesis_html = (
        f'<section class="panel"><h2>Synthesis response</h2><pre class="log-block">{html.escape(detail.synthesis_text)}</pre></section>'
        if detail.synthesis_text
        else ""
    )
    return f"""
    <section class="panel">
      <div class="card-topline">
        <div>
          <h2>Reward Value Model</h2>
          <p class="metric-subtle">Action-conditioned reward predictor trained from PoolTool rollouts labeled by the synthesized reward program.</p>
        </div>
      </div>
      <div class="job-overview-grid">
        {overview_cards_html}
      </div>
      <div class="content-grid">
        <section class="panel compact-panel">
          <h3>NL goals</h3>
          {goal_list_html}
          <h3>Reward DSL</h3>
          <pre class="code-block">{html.escape(detail.program_text)}</pre>
        </section>
        <section class="panel compact-panel">
          <h3>Training config</h3>
          <div class="metric-stack">
            <div class="metric-row-block"><div class="metric-row"><span>Epochs</span><strong>{detail.epochs}</strong></div></div>
            <div class="metric-row-block"><div class="metric-row"><span>Batch size</span><strong>{detail.batch_size}</strong></div></div>
            <div class="metric-row-block"><div class="metric-row"><span>Learning rate</span><strong>{detail.learning_rate if detail.learning_rate is not None else '—'}</strong></div></div>
            <div class="metric-row-block"><div class="metric-row"><span>Image size</span><strong>{detail.image_size if detail.image_size is not None else '—'}</strong></div></div>
            <div class="metric-row-block"><div class="metric-row"><span>Checkpoint</span><strong>{'saved' if detail.checkpoint_path and detail.checkpoint_path.exists() else '—'}</strong></div></div>
          </div>
        </section>
      </div>
      {_render_xy_line_chart(title='Validation reward by epoch', x_label='Epoch', y_label='Reward', series=curve_reward)}
      {_render_xy_line_chart(title='Validation success by epoch', x_label='Epoch', y_label='Success rate (%)', series=curve_success)}
      {_render_xy_line_chart(title='Training loss by epoch', x_label='Epoch', y_label='Loss', series=curve_loss)}
      <p class="metric-subtle">Validation bands show standard deviation across saved validation rollouts at each epoch.</p>
      <section class="panel compact-panel">
        <h3>Epoch metrics</h3>
        <div class="scroll-table compact-scroll-table">
          <table class="data-table compact-table">
            <thead><tr><th>Epoch</th><th>Train loss</th><th>Val reward</th><th>Val success</th></tr></thead>
            <tbody>
              {epoch_rows_html}
            </tbody>
          </table>
        </div>
      </section>
      <section class="panel compact-panel">
        <h3>Top validation rollouts</h3>
        <div class="scroll-table compact-scroll-table">
          <table class="data-table compact-table">
            <thead><tr><th>Epoch</th><th>Scene</th><th>Reward</th><th>Solved</th><th>Artifacts</th></tr></thead>
            <tbody>
              {top_validation_rows}
            </tbody>
          </table>
        </div>
      </section>
      {synthesis_html}
    </section>
    """


def _render_trajectory_chart(
    payload: dict[str, object] | None,
    *,
    title: str,
    x_label: str,
    metric: str,
) -> str:
    if not isinstance(payload, dict):
        return '<p class="empty">No reward data yet.</p>'
    raw_series = payload.get("series")
    if not isinstance(raw_series, list) or not raw_series:
        return '<p class="empty">No reward data yet.</p>'

    reward_key = "validation_reward" if metric == "validation" else "train_reward"
    values: list[float] = []
    max_iteration = 0
    series: list[dict[str, object]] = []
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
            if not isinstance(iteration, int) or not isinstance(reward, (int, float)):
                continue
            reward_value = float(reward)
            if reward_value < -1.0 or reward_value > 1.0:
                continue
            cleaned_values.append((iteration, reward_value))
            values.append(reward_value)
            max_iteration = max(max_iteration, iteration)
        if cleaned_values:
            series.append(
                {
                    "label": label,
                    "palette_index": int(item.get("palette_index") or 0),
                    "values": cleaned_values,
                }
            )
    if not series or not values:
        return '<p class="empty">No reward data yet.</p>'

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
    x_ticks = [0] if tick_count <= 1 else [round((max_iteration * index) / (tick_count - 1)) for index in range(tick_count)]

    legend_markup: list[str] = []
    line_markup: list[str] = []
    for index, item in enumerate(series):
        palette_index = item["palette_index"] if isinstance(item["palette_index"], int) else index
        color = palette[palette_index % len(palette)]
        points = item["values"]
        running_best_points: list[tuple[int, float]] = []
        best_so_far: float | None = None
        for iteration, reward in points:
            if best_so_far is None or reward > best_so_far:
                best_so_far = reward
                running_best_points.append((iteration, reward))
        path = " ".join(
            f'{"M" if point_index == 0 else "L"} {x(iteration):.1f} {y(reward):.1f}'
            for point_index, (iteration, reward) in enumerate(points)
        )
        line_markup.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.4" stroke-opacity="0.42" />')
        for iteration, reward in points:
            line_markup.append(f'<circle cx="{x(iteration):.1f}" cy="{y(reward):.1f}" r="2.2" fill="{color}" fill-opacity="0.34" />')
        for best_iteration, best_reward in running_best_points:
            line_markup.append(f'<circle cx="{x(best_iteration):.1f}" cy="{y(best_reward):.1f}" r="7.2" fill="{color}" fill-opacity="0.16" />')
            line_markup.append(f'<circle cx="{x(best_iteration):.1f}" cy="{y(best_reward):.1f}" r="4.8" fill="{color}" stroke="white" stroke-width="1.8" />')
        legend_markup.append(f'<span><i class="legend-swatch" style="background:{color}; border-color:{color};"></i>{html.escape(str(item["label"]))}</span>')

    grid_markup = "".join(
        f'<g><line x1="{margin_left}" y1="{y(value):.1f}" x2="{width - margin_right}" y2="{y(value):.1f}" class="loss-grid-line" /><text x="{margin_left - 8}" y="{y(value) + 4:.1f}" text-anchor="end" class="loss-axis-label">{value:.2f}</text></g>'
        for value in tick_values
    )
    x_tick_markup = "".join(
        f'<g><line x1="{x(tick):.1f}" y1="{margin_top}" x2="{x(tick):.1f}" y2="{height - margin_bottom}" class="loss-grid-line" /><text x="{x(tick):.1f}" y="{height - margin_bottom + 18:.1f}" text-anchor="middle" class="loss-axis-label">{tick}</text></g>'
        for tick in x_ticks
    )
    zero_line_markup = ""
    if min_value <= 0.0 <= max_value:
        zero_y = y(0.0)
        zero_line_markup = f'<g><line x1="{margin_left}" y1="{zero_y:.1f}" x2="{width - margin_right}" y2="{zero_y:.1f}" class="loss-zero-line" /><text x="{margin_left - 8}" y="{zero_y + 4:.1f}" text-anchor="end" class="loss-axis-label loss-zero-label">0</text></g>'

    return f"""
    <div class="trajectory-chart trajectory-chart-large">
      <div class="loss-chart-frame">
        <div class="loss-chart-legend">{''.join(legend_markup)}</div>
        <svg viewBox="0 0 {width} {height}" class="loss-chart-svg" role="img" aria-label="{html.escape(title)}">
          {grid_markup}
          {zero_line_markup}
          {x_tick_markup}
          <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" class="loss-axis-line" />
          <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" class="loss-axis-line" />
          {''.join(line_markup)}
        </svg>
        <div class="loss-axis-caption">{html.escape(x_label)}</div>
      </div>
    </div>
    """


def _merge_learning_trajectory_with_live_progress(
    payload: dict[str, object] | None,
    learning_progress: dict[str, Any] | None,
) -> dict[str, Any]:
    base_payload: dict[str, Any] = (
        copy.deepcopy(payload)
        if isinstance(payload, dict)
        else {"series": [], "max_iteration": None}
    )
    raw_series = base_payload.get("series")
    if not isinstance(raw_series, list):
        raw_series = []
        base_payload["series"] = raw_series
    if not isinstance(learning_progress, dict):
        base_payload.setdefault("live_only_labels", [])
        base_payload.setdefault("pending_live_labels", [])
        return base_payload

    live_labels: dict[str, dict[str, Any]] = {}

    def _ensure_live_label(label_name: str, *, label_index: int | None = None) -> dict[str, Any]:
        normalized_label = str(label_name or "").strip()
        if not normalized_label:
            return {}
        entry = live_labels.get(normalized_label)
        if entry is None:
            entry = {
                "label": normalized_label,
                "label_index": label_index,
                "candidate_evaluations": None,
                "train_reward": None,
                "validation_reward": None,
            }
            live_labels[normalized_label] = entry
        elif label_index is not None and entry.get("label_index") is None:
            entry["label_index"] = label_index
        return entry

    provisional_labels = (
        learning_progress.get("provisional_labels")
        if isinstance(learning_progress.get("provisional_labels"), list)
        else []
    )
    for item in provisional_labels:
        if not isinstance(item, dict):
            continue
        if bool(item.get("backfilled_from_label_result")):
            continue
        label_name = str(item.get("label") or "").strip()
        if not label_name:
            continue
        entry = _ensure_live_label(
            label_name,
            label_index=_safe_int(item.get("label_index")),
        )
        candidate_evaluations = _safe_int(item.get("candidate_evaluations"))
        if candidate_evaluations is not None:
            entry["candidate_evaluations"] = max(
                _safe_int(entry.get("candidate_evaluations")) or 0,
                candidate_evaluations,
            )
        best_train = _safe_float(item.get("best_score"))
        best_validation = _safe_float(item.get("best_validation_score"))
        if best_train is not None:
            entry["train_reward"] = best_train
        if best_validation is not None:
            entry["validation_reward"] = best_validation

    workers = learning_progress.get("workers") if isinstance(learning_progress.get("workers"), list) else []
    for item in workers:
        if not isinstance(item, dict):
            continue
        label_name = str(item.get("label") or "").strip()
        if not label_name:
            continue
        entry = _ensure_live_label(
            label_name,
            label_index=_safe_int(item.get("label_index")),
        )
        candidate_evaluations = _safe_int(item.get("candidate_evaluations"))
        if candidate_evaluations is not None:
            entry["candidate_evaluations"] = max(
                _safe_int(entry.get("candidate_evaluations")) or 0,
                candidate_evaluations,
            )
        train_reward = _safe_float(item.get("latest_score"))
        validation_reward = _safe_float(item.get("latest_validation_score"))
        if train_reward is not None:
            entry["train_reward"] = train_reward
        if validation_reward is not None:
            entry["validation_reward"] = validation_reward

    existing_by_label: dict[str, dict[str, Any]] = {}
    max_iteration = _safe_int(base_payload.get("max_iteration")) or 0
    for item in raw_series:
        if not isinstance(item, dict):
            continue
        label_name = str(item.get("label") or "").strip()
        if not label_name:
            continue
        existing_by_label[label_name] = item
        raw_values = item.get("values")
        if not isinstance(raw_values, list):
            raw_values = []
            item["values"] = raw_values
        for point in raw_values:
            if not isinstance(point, dict):
                continue
            iteration = _safe_int(point.get("iteration"))
            if iteration is not None:
                max_iteration = max(max_iteration, iteration)

    live_only_labels: list[str] = []
    pending_live_labels: list[str] = []

    for label_name, live_entry in sorted(
        live_labels.items(),
        key=lambda entry: (
            _safe_int(entry[1].get("label_index")) or 10_000,
            entry[0],
        ),
    ):
        candidate_evaluations = _safe_int(live_entry.get("candidate_evaluations"))
        if candidate_evaluations is None or candidate_evaluations <= 0:
            candidate_evaluations = 1
        train_reward = _safe_float(live_entry.get("train_reward"))
        validation_reward = _safe_float(live_entry.get("validation_reward"))
        existing = existing_by_label.get(label_name)
        if existing is None:
            if train_reward is None and validation_reward is None:
                pending_live_labels.append(label_name)
                continue
            palette_index = max(
                0,
                (_safe_int(live_entry.get("label_index")) or (len(raw_series) + 1)) - 1,
            )
            new_series = {
                "label": label_name,
                "palette_index": palette_index,
                "values": [
                    {
                        "iteration": candidate_evaluations,
                        "train_reward": train_reward,
                        "validation_reward": validation_reward,
                        "provisional": True,
                    }
                ],
            }
            raw_series.append(new_series)
            existing_by_label[label_name] = new_series
            live_only_labels.append(label_name)
            max_iteration = max(max_iteration, candidate_evaluations)
            continue

        raw_values = existing.get("values")
        if not isinstance(raw_values, list):
            raw_values = []
            existing["values"] = raw_values
        last_point = raw_values[-1] if raw_values and isinstance(raw_values[-1], dict) else {}
        last_iteration = _safe_int(last_point.get("iteration")) or 0
        last_train = _safe_float(last_point.get("train_reward"))
        last_validation = _safe_float(last_point.get("validation_reward"))
        has_live_score = train_reward is not None or validation_reward is not None
        if has_live_score and (
            candidate_evaluations > last_iteration
            or train_reward != last_train
            or validation_reward != last_validation
        ):
            raw_values.append(
                {
                    "iteration": candidate_evaluations,
                    "train_reward": train_reward,
                    "validation_reward": validation_reward,
                    "provisional": True,
                }
            )
            max_iteration = max(max_iteration, candidate_evaluations)

    base_payload["series"] = sorted(
        [item for item in raw_series if isinstance(item, dict)],
        key=lambda item: (
            _safe_int(item.get("palette_index")) or 10_000,
            str(item.get("label") or ""),
        ),
    )
    base_payload["max_iteration"] = max_iteration if max_iteration > 0 else None
    base_payload["live_only_labels"] = live_only_labels
    base_payload["pending_live_labels"] = pending_live_labels
    return base_payload


def _render_live_reward_trajectory_note(
    payload: dict[str, object] | None,
) -> str:
    if not isinstance(payload, dict):
        return ""
    live_only_labels = [
        str(item).strip()
        for item in (payload.get("live_only_labels") or [])
        if str(item).strip()
    ]
    pending_live_labels = [
        str(item).strip()
        for item in (payload.get("pending_live_labels") or [])
        if str(item).strip()
    ]
    if not live_only_labels and not pending_live_labels:
        return ""
    note_bits: list[str] = []
    if live_only_labels:
        note_bits.append(
            "Live labels merged into the chart from current worker logs: "
            + ", ".join(live_only_labels)
        )
    if pending_live_labels:
        note_bits.append(
            "Active labels with no scored candidates yet: "
            + ", ".join(pending_live_labels)
        )
    return (
        '<section class="job-message-banner tone-info">'
        '<strong>Live reward sync</strong>'
        f'<p>{html.escape(" ".join(note_bits))}</p>'
        '</section>'
    )


def _render_label_detail_charts(trajectory: dict[str, Any]) -> str:
    raw_series = trajectory.get("series")
    if not isinstance(raw_series, list):
        return ""
    cards: list[str] = []
    for index, item in enumerate(raw_series):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        raw_values = item.get("values")
        if not label or not isinstance(raw_values, list):
            continue
        values: list[tuple[int, float | None, float | None]] = []
        for point in raw_values:
            if not isinstance(point, dict):
                continue
            iteration = point.get("iteration")
            if not isinstance(iteration, int):
                continue
            train_reward = point.get("train_reward")
            validation_reward = point.get("validation_reward")
            train_value = float(train_reward) if isinstance(train_reward, (int, float)) else None
            validation_value = float(validation_reward) if isinstance(validation_reward, (int, float)) else None
            if train_value is None and validation_value is None:
                continue
            values.append((iteration, train_value, validation_value))
        if not values:
            continue
        cards.append(_render_label_detail_chart_card(label, values, palette_index=int(item.get("palette_index") or index)))
    if not cards:
        return ""
    return f"""
    <section class="panel">
      <div class="card-topline">
        <div>
          <h2>Learning details by label</h2>
          <p class="metric-subtle">Training and validation score across candidate evaluations for each label. Best recorded points are highlighted.</p>
        </div>
      </div>
      <div class="label-chart-grid">{''.join(cards)}</div>
    </section>
    """


def _render_label_detail_chart_card(
    label: str,
    values: list[tuple[int, float | None, float | None]],
    *,
    palette_index: int,
) -> str:
    width = 420
    height = 220
    margin_top = 16
    margin_right = 14
    margin_bottom = 32
    margin_left = 40
    inner_width = width - margin_left - margin_right
    inner_height = height - margin_top - margin_bottom
    max_iteration = max(iteration for iteration, _, _ in values)
    all_scores = [
        score
        for _, train_score, validation_score in values
        for score in (train_score, validation_score)
        if isinstance(score, float)
    ]
    min_score = min(all_scores) if all_scores else -1.0
    max_score = max(all_scores) if all_scores else 1.0
    if min_score == max_score:
        min_score -= 1.0
        max_score += 1.0
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
    train_color = palette[palette_index % len(palette)]
    validation_color = "#174a7c"

    def x(iteration: int) -> float:
        return margin_left + ((iteration or 0) / max(max_iteration, 1)) * inner_width

    def y(value: float) -> float:
        return margin_top + ((max_score - value) / (max_score - min_score)) * inner_height

    def _line_path(points: list[tuple[int, float]]) -> str:
        return " ".join(
            f'{"M" if point_index == 0 else "L"} {x(iteration):.1f} {y(score):.1f}'
            for point_index, (iteration, score) in enumerate(points)
        )

    train_points = [(iteration, score) for iteration, score, _ in values if isinstance(score, float)]
    validation_points = [(iteration, score) for iteration, _, score in values if isinstance(score, float)]

    def _best_points(points: list[tuple[int, float]]) -> list[tuple[int, float]]:
        if not points:
            return []
        best_score = max(score for _, score in points)
        return [(iteration, score) for iteration, score in points if score == best_score]

    train_best = _best_points(train_points)
    validation_best = _best_points(validation_points)

    tick_values = [min_score + ((max_score - min_score) * index) / 4 for index in range(5)]
    x_ticks = [0] if max_iteration <= 1 else [round((max_iteration * index) / 4) for index in range(5)]

    line_markup: list[str] = []
    if train_points:
        line_markup.append(f'<path d="{_line_path(train_points)}" fill="none" stroke="{train_color}" stroke-width="2.1" />')
        for iteration, score in train_points:
            line_markup.append(f'<circle cx="{x(iteration):.1f}" cy="{y(score):.1f}" r="2.2" fill="{train_color}" fill-opacity="0.45" />')
        for iteration, score in train_best:
            line_markup.append(f'<circle cx="{x(iteration):.1f}" cy="{y(score):.1f}" r="7.2" fill="{train_color}" fill-opacity="0.16" />')
            line_markup.append(f'<circle cx="{x(iteration):.1f}" cy="{y(score):.1f}" r="4.6" fill="{train_color}" stroke="white" stroke-width="1.6" />')
    if validation_points:
        line_markup.append(f'<path d="{_line_path(validation_points)}" fill="none" stroke="{validation_color}" stroke-width="2.1" stroke-dasharray="5 4" />')
        for iteration, score in validation_points:
            line_markup.append(f'<rect x="{x(iteration) - 2.1:.1f}" y="{y(score) - 2.1:.1f}" width="4.2" height="4.2" fill="{validation_color}" fill-opacity="0.55" />')
        for iteration, score in validation_best:
            line_markup.append(f'<circle cx="{x(iteration):.1f}" cy="{y(score):.1f}" r="7.2" fill="{validation_color}" fill-opacity="0.16" />')
            line_markup.append(f'<rect x="{x(iteration) - 4.0:.1f}" y="{y(score) - 4.0:.1f}" width="8.0" height="8.0" fill="{validation_color}" stroke="white" stroke-width="1.6" />')

    grid_markup = "".join(
        f'<g><line x1="{margin_left}" y1="{y(value):.1f}" x2="{width - margin_right}" y2="{y(value):.1f}" class="loss-grid-line" /><text x="{margin_left - 6}" y="{y(value) + 4:.1f}" text-anchor="end" class="loss-axis-label">{value:.2f}</text></g>'
        for value in tick_values
    )
    x_tick_markup = "".join(
        f'<g><line x1="{x(tick):.1f}" y1="{margin_top}" x2="{x(tick):.1f}" y2="{height - margin_bottom}" class="loss-grid-line" /><text x="{x(tick):.1f}" y="{height - margin_bottom + 16:.1f}" text-anchor="middle" class="loss-axis-label">{tick}</text></g>'
        for tick in x_ticks
    )
    zero_line_markup = ""
    if min_score <= 0.0 <= max_score:
        zero_y = y(0.0)
        zero_line_markup = f'<line x1="{margin_left}" y1="{zero_y:.1f}" x2="{width - margin_right}" y2="{zero_y:.1f}" class="loss-zero-line" />'

    best_meta: list[str] = []
    if train_best:
        best_meta.append(f"Train best {train_best[0][1]:.3f}")
    if validation_best:
        best_meta.append(f"Val best {validation_best[0][1]:.3f}")
    best_meta.append(f"{len(values)} candidates")

    return f"""
    <article class="label-chart-card">
      <div>
        <h3>{html.escape(label)}</h3>
        <div class="label-chart-meta">{''.join(f'<span>{html.escape(bit)}</span>' for bit in best_meta)}</div>
      </div>
      <div class="label-chart-frame">
        <div class="label-chart-legend">
          <span><i class="legend-swatch" style="background:{train_color}; border-color:{train_color};"></i>Training</span>
          <span><i class="legend-swatch" style="background:{validation_color}; border-color:{validation_color};"></i>Validation</span>
        </div>
        <svg viewBox="0 0 {width} {height}" class="label-chart-svg" role="img" aria-label="{html.escape(label)} learning detail">
          {grid_markup}
          {zero_line_markup}
          {x_tick_markup}
          <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" class="loss-axis-line" />
          <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" class="loss-axis-line" />
          {''.join(line_markup)}
        </svg>
      </div>
    </article>
    """


def _render_learning_candidate_browser(
    *,
    run_dir: Path | None,
    project_root: Path,
) -> str:
    if run_dir is None or not run_dir.is_dir():
        return ""
    label_dirs = _unique_label_dirs(run_dir)
    if not label_dirs:
        return ""

    label_cards: list[str] = []
    for label_dir in label_dirs:
        label_name = _label_name_from_dir(label_dir.name)
        label_result = _load_json_dict(label_dir / "label_result.json")
        best_uid = str(label_result.get("best_candidate_uid") or "").strip()
        recorded_evals = _safe_int(label_result.get("candidate_evaluations"))
        is_complete = bool(label_result.get("completed") or label_result.get("is_complete"))
        candidate_dirs = sorted(
            [path for path in label_dir.glob("candidate_*") if path.is_dir()],
            key=_candidate_dir_sort_key,
        )

        candidate_rows: list[str] = []
        for candidate_dir in candidate_dirs:
            candidate_payload = _load_json_dict(candidate_dir / "candidate.json")
            candidate_uid = str(
                candidate_payload.get("uid")
                or candidate_payload.get("candidate_uid")
                or candidate_dir.name
            ).strip()
            candidate_index = _safe_int(candidate_payload.get("candidate_index"))
            if candidate_index is None:
                candidate_index = _candidate_index_from_name(candidate_dir.name)
            origin = str(candidate_payload.get("origin") or "").strip()
            train_score = _safe_float(candidate_payload.get("train_score"), fallback=_safe_float(candidate_payload.get("score")))
            validation_score = _safe_float(
                candidate_payload.get("val_score"),
                fallback=_safe_float(candidate_payload.get("validation_score")),
            )
            average_score = _safe_float(candidate_payload.get("average_score"), fallback=_safe_float(candidate_payload.get("score")))
            is_best = bool(candidate_payload.get("is_best_candidate")) or (best_uid and candidate_uid == best_uid)

            code_path = candidate_dir / "find_pattern.py"
            code_relpath = _project_relpath(code_path) if code_path.is_file() else ""
            code_href = f"/ui/files?path={quote(code_relpath)}" if code_relpath else ""
            candidate_title = (
                f"#{candidate_index:03d} {candidate_uid}" if isinstance(candidate_index, int) else candidate_uid
            )
            summary_bits = [
                f"train {_format_reward_value(train_score)}",
                f"val {_format_reward_value(validation_score)}",
                f"avg {_format_reward_value(average_score)}",
            ]
            if origin:
                summary_bits.append(origin)
            if is_best:
                summary_bits.append("best")
            summary_text = " · ".join(summary_bits)

            candidate_rows.append(
                f"""
                <details class="candidate-cell">
                  <summary>
                    <span class="candidate-cell-title">{html.escape(candidate_title)}</span>
                    <span class="candidate-cell-summary">{html.escape(summary_text)}</span>
                  </summary>
                  <div class="candidate-cell-body">
                    <div class="candidate-cell-meta">
                      <span class="mini-chip">uid {html.escape(candidate_uid)}</span>
                      <span class="mini-chip">train {html.escape(_format_reward_value(train_score))}</span>
                      <span class="mini-chip">val {html.escape(_format_reward_value(validation_score))}</span>
                      <span class="mini-chip">avg {html.escape(_format_reward_value(average_score))}</span>
                      {f'<span class="mini-chip">origin {html.escape(origin)}</span>' if origin else ''}
                      {f'<span class="mini-chip">best</span>' if is_best else ''}
                      {f'<a class="ghost-link" href="{html.escape(code_href)}">Open file</a>' if code_href else '<span class="metric-subtle">Code file missing</span>'}
                    </div>
                    {(
                        f'<pre class="code-block candidate-code-block" data-code-src="{html.escape(code_href)}">Expand this candidate to load code…</pre>'
                        if code_href
                        else '<pre class="code-block candidate-code-block">No find_pattern.py found for this candidate.</pre>'
                    )}
                  </div>
                </details>
                """
            )

        label_summary_bits: list[str] = [f"{len(candidate_dirs)} candidates"]
        if isinstance(recorded_evals, int):
            label_summary_bits.append(f"{recorded_evals} evals")
        if is_complete:
            label_summary_bits.append("complete")
        if best_uid:
            label_summary_bits.append(f"best {best_uid}")

        label_cards.append(
            f"""
            <details class="candidate-label-panel">
              <summary>
                <span class="candidate-label-title">{html.escape(label_name)}</span>
                <span class="candidate-label-summary">{html.escape(" · ".join(label_summary_bits))}</span>
              </summary>
              <div class="candidate-cell-list">
                {''.join(candidate_rows) if candidate_rows else '<p class="empty">No candidate directories yet.</p>'}
              </div>
            </details>
            """
        )

    return f"""
    <section class="panel">
      <div class="card-topline">
        <div>
          <h2>Candidate codes by label</h2>
          <p class="metric-subtle">Open a label, then open a candidate to view the generated code and reward metrics.</p>
        </div>
      </div>
      <div class="candidate-label-list">
        {''.join(label_cards)}
      </div>
    </section>
    """


def _label_dir_sort_key(path: Path) -> tuple[int, str]:
    match = _LABEL_DIR_PATTERN.match(path.name)
    if match is None:
        return (10_000, path.name.lower())
    return (_safe_int(match.group("index")) or 10_000, path.name.lower())


def _candidate_dir_sort_key(path: Path) -> tuple[int, str]:
    index = _candidate_index_from_name(path.name)
    return (index if isinstance(index, int) else 10_000, path.name.lower())


def _candidate_index_from_name(name: str) -> int | None:
    match = _CANDIDATE_DIR_PATTERN.match(str(name))
    if match is None:
        return None
    return _safe_int(match.group("index"))


def _label_name_from_dir(label_dir_name: str) -> str:
    match = _LABEL_DIR_PATTERN.match(label_dir_name)
    if match is None:
        return label_dir_name
    label_raw = str(match.group("label") or "").strip()
    if not label_raw:
        return label_dir_name
    return label_raw.replace("_", " ")


def _load_json_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, *, fallback: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _format_reward_value(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def render_job_page(
    *,
    job: JobRecord,
    jobs: list[JobRecord],
    clients: list[ClientRecord],
    project_root: Path,
    log_lines: list[str],
    reward_metric: str = "train",
) -> str:
    is_controllable = _job_is_controllable(job)
    client_lookup = {client.client_id: client for client in clients}
    artifact_entries = _artifact_entries(job, project_root=project_root)
    display_log_lines = list(log_lines)
    if not display_log_lines:
        display_log_lines = _deepphy_live_progress_lines(job, project_root=project_root)
    learning_progress = _learning_progress_summary(job, project_root=project_root, log_lines=log_lines)
    reward_detail = load_reward_job_detail(job, project_root=project_root) if str(job.spec.action or "").startswith("pooltool-reward-") else None
    library_ablation_detail = (
        load_library_ablation_job_detail(job, project_root=project_root)
        if str(job.spec.action or "").strip() == "evaluate-library-case"
        else None
    )
    reward_metric_normalized = "validation" if reward_metric == "validation" else "train"
    run_dir = _resolve_job_run_dir(job, project_root=project_root)
    reward_trajectory = load_reward_trajectory(run_dir) if run_dir is not None else {"series": [], "max_iteration": None}
    reward_trajectory = _merge_learning_trajectory_with_live_progress(
        reward_trajectory,
        learning_progress,
    )
    metadata_json = json.dumps(
        _sanitize_display_data(job.spec.metadata),
        indent=2,
        sort_keys=True,
        default=str,
    )
    action_buttons: list[str] = []
    next_url = f"/jobs/{job.job_id}"
    if is_controllable and job.status.value in {"queued", "running", "assigned", "starting", "pausing", "resuming", "stopping"}:
        action_buttons.append(
            f'<form method="post" action="/ui/jobs/{html.escape(job.job_id)}/pause"><input type="hidden" name="next" value="{html.escape(next_url)}"><button class="secondary-button" type="submit">Pause</button></form>'
        )
    if is_controllable and job.status.value in {"paused", "failed", "stopped"}:
        action_buttons.append(
            f'<form method="post" action="/ui/jobs/{html.escape(job.job_id)}/resume"><input type="hidden" name="next" value="{html.escape(next_url)}"><button class="secondary-button" type="submit">Resume</button></form>'
        )
    if is_controllable and job.status.value in {"running", "paused", "failed", "stopped", "cancelled"}:
        action_buttons.append(
            f'<form method="post" action="/ui/jobs/{html.escape(job.job_id)}/requeue"><input type="hidden" name="next" value="{html.escape(next_url)}"><button class="secondary-button" type="submit">Requeue</button></form>'
        )
    if is_controllable and job.status.value not in {"succeeded", "cancelled"}:
        action_buttons.append(
            f'<form method="post" action="/ui/jobs/{html.escape(job.job_id)}/cancel"><input type="hidden" name="next" value="{html.escape(next_url)}"><button class="secondary-button danger-button" type="submit">Cancel</button></form>'
        )
    if not is_controllable:
        action_buttons.append('<span class="mini-chip">Read only</span>')
    artifact_html = (
        "".join(
            f"""
            <article class="artifact-card">
              <h3><a href="{html.escape(entry['href'])}">{html.escape(entry['relpath'])}</a></h3>
              <p class="artifact-meta metric-subtle">{html.escape(entry['kind'])}</p>
              {f'<pre class="artifact-preview">{html.escape(entry["preview"])}</pre>' if entry["preview"] else '<p class="metric-subtle">No inline preview.</p>'}
            </article>
            """
            for entry in artifact_entries
        )
        if artifact_entries
        else '<p class="empty">No artifacts uploaded yet.</p>'
    )
    reward_toggle_train_url = f"/jobs/{job.job_id}?reward_metric=train"
    reward_toggle_validation_url = f"/jobs/{job.job_id}?reward_metric=validation"
    reward_chart_html = _render_trajectory_chart(
        reward_trajectory,
        title=("Validation reward over candidate evaluations by label" if reward_metric_normalized == "validation" else "Training reward over candidate evaluations by label"),
        x_label="Candidate number",
        metric=reward_metric_normalized,
    )
    reward_chart_note_html = _render_live_reward_trajectory_note(reward_trajectory)
    label_detail_charts_html = _render_label_detail_charts(reward_trajectory)
    candidate_browser_html = _render_learning_candidate_browser(
        run_dir=run_dir if _is_learning_job(job) else None,
        project_root=project_root,
    )
    learning_progress_html = (
        _render_learning_progress_detail(
            learning_progress,
            job=job,
            project_root=project_root,
        )
        if learning_progress is not None
        else ""
    )
    learning_timing_html = _render_learning_timing_breakdown(job, project_root=project_root)
    repeat_learning_experiment_html = _render_repeat_learning_experiment_panel(
        job=job,
        jobs=jobs,
        project_root=project_root,
    )
    has_embedded_automatic_deepphy = str(job.spec.action or "").strip() == "automatic-launch" and _automatic_deepphy_eval_root(job, project_root=project_root) is not None
    should_render_evaluation_tasks = job.spec.action == "evaluate-deepphy" or has_embedded_automatic_deepphy
    evaluation_tasks = _evaluation_task_list(job, project_root=project_root) if should_render_evaluation_tasks else []
    deepphy_success_summary_html = _render_deepphy_success_summary(job=job, evaluation_tasks=evaluation_tasks) if evaluation_tasks else ""
    deepphy_anchor_id = "automatic-deepphy" if has_embedded_automatic_deepphy else "deepphy"
    tasks_section_id_attr = ' id="automatic-deepphy-tasks"' if has_embedded_automatic_deepphy else ""
    tasks_section_title = "Combined DeepPHY tasks" if has_embedded_automatic_deepphy else "Tasks"
    title = _queue_job_title(job)
    env_id = _job_env_id(job)
    flow_label = _job_flow_label(job)
    assigned_client = _assigned_client_name(job, client_lookup)
    job_status_summary = _job_status_summary(job, project_root=project_root)
    job_message = _sanitize_display_text(job.message)
    message_banner_html = (
        f"""
        <section class="job-message-banner tone-{_job_message_tone(job)}">
          <strong>{'Waiting' if _job_message_tone(job) == 'waiting' else 'Status note'}</strong>
          <p>{html.escape(job_message)}</p>
        </section>
        """
        if job_message
        else ""
    )
    quick_links_html = _job_quick_links(job, artifact_entries)
    runtime_anchor = job.started_at or job.created_at
    runtime_value = _display_duration(runtime_anchor, job.finished_at)
    overview_cards = [
        ("Worker", assigned_client, _display_time(job.updated_at)),
        ("Run time", runtime_value, f"Started {_display_time(runtime_anchor)}"),
        ("Progress", job_status_summary or "No structured progress yet", flow_label),
        ("Outcome", (str(job.exit_code) if job.exit_code is not None else "In progress"), _display_time(job.finished_at) if job.finished_at is not None else "Still active"),
    ]
    reward_content_html = ""
    if isinstance(reward_detail, RewardOptimizationDetail):
        reward_content_html = _render_reward_optimization_panel(reward_detail, project_root=project_root)
    elif isinstance(reward_detail, RewardValueModelDetail):
        reward_content_html = _render_reward_value_model_panel(reward_detail, project_root=project_root)
    elif str(job.spec.action or "").strip() == "evaluate-library-case":
        reward_content_html = _render_library_ablation_detail_panel(library_ablation_detail)
    elif str(job.spec.action or "").startswith("pooltool-reward-"):
        reward_content_html = """
        <section class="panel">
          <div class="card-topline">
            <div>
              <h2>Reward experiment details</h2>
              <p class="metric-subtle">This reward-program job has not uploaded its experiment summary yet. Once the run directory lands under the server data tree, this page will show the saved plots, rollout metrics, and DSL program details.</p>
            </div>
          </div>
        </section>
        """
    else:
        reward_content_html = f"""
        <section class="panel">
          <div class="card-topline">
            <div>
              <h2>{'Validation' if reward_metric_normalized == 'validation' else 'Training'} reward by label</h2>
              <p class="metric-subtle">Per-label reward trajectory recovered from the learning run artifacts, matching the v1 run overview.</p>
            </div>
            <div class="chart-toggle-row">
              <a class="secondary-button{' is-active' if reward_metric_normalized == 'train' else ''}" href="{html.escape(reward_toggle_train_url)}">Training reward</a>
              <a class="secondary-button{' is-active' if reward_metric_normalized == 'validation' else ''}" href="{html.escape(reward_toggle_validation_url)}">Validation reward</a>
            </div>
          </div>
          {reward_chart_note_html}
          {reward_chart_html}
        </section>
        {label_detail_charts_html}
        """
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Home</a> / job / {html.escape(job.job_id)}</p>
      <div class="job-hero-title-row">
        <h1>{html.escape(title)}{f' · {html.escape(env_id)}' if env_id and env_id != '—' else ''}</h1>
        {_pill(job.status.value, job.status.value)}
      </div>
      <div class="job-hero-subline">
        <span><strong>ID</strong> {html.escape(job.job_id)}</span>
        <span><strong>Flow</strong> {html.escape(flow_label)}</span>
        <span><strong>Worker</strong> {html.escape(assigned_client)}</span>
      </div>
      <p>Use this page to judge whether the run is healthy, blocked, producing useful artifacts, or ready to intervene.</p>
      <div class="hero-stats">
        <div class="stat-chip"><span class="stat-label">State</span><strong class="stat-value">{html.escape(job.status.value)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Flow</span><strong class="stat-value">{html.escape(flow_label)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Env</span><strong class="stat-value">{html.escape(env_id)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Exit code</span><strong class="stat-value">{html.escape(str(job.exit_code) if job.exit_code is not None else "—")}</strong></div>
      </div>
    </section>
    {f'<div id="{deepphy_anchor_id}">{deepphy_success_summary_html}</div>' if deepphy_success_summary_html else ''}
    <div class="button-row job-inline-actions">
      {''.join(action_buttons)}
    </div>
    {message_banner_html}
    <section class="panel">
      <div class="card-topline">
        <div>
          <h2>Run overview</h2>
          <p class="metric-subtle">The few details you need most often while deciding whether to wait, inspect, or intervene.</p>
        </div>
      </div>
      <div class="job-overview-grid">
        {"".join(
            f'<article class="job-overview-card"><span class="job-overview-kicker">{html.escape(label)}</span><strong class="job-overview-value">{html.escape(value)}</strong><span class="job-overview-meta">{html.escape(meta)}</span></article>'
            for label, value, meta in overview_cards
        )}
      </div>
    </section>
    {quick_links_html}
    {repeat_learning_experiment_html}
    {learning_progress_html}
    {learning_timing_html}
    {reward_content_html}
    {candidate_browser_html}
    <section class="panel">
      <h2>Artifacts and results</h2>
      <div class="artifact-list">{artifact_html}</div>
    </section>
    {(
        f'<section class="panel"{tasks_section_id_attr}><div class="card-topline"><div><h2>{tasks_section_title}</h2><p class="metric-subtle">Per-task outcomes, with click-through detail for attempts, media, traces, and model responses.</p></div><div class="stats-inline">'
        + f'<span class="mini-chip">{sum(1 for item in evaluation_tasks if item.success is True)} success</span>'
        + f'<span class="mini-chip">{sum(1 for item in evaluation_tasks if item.success is False)} fail</span>'
        + f'<span class="mini-chip">{len(evaluation_tasks)} total</span>'
        + '</div></div><div class="task-list">'
        + "".join(
            f'<article class="task-row"><div class="task-row-main"><a class="metric-title" href="/jobs/{html.escape(job.job_id)}/tasks/{quote(item.task_id, safe="")}">{html.escape(item.task_id)}</a><p class="metric-subtle">{len(item.attempts)} attempts · completed {_display_time(item.completed_at) if item.completed_at is not None else "—"}</p></div><div class="task-row-meta">{_pill("Success" if item.success else "Failed", "succeeded" if item.success else "failed")}<a class="ghost-link" href="/jobs/{html.escape(job.job_id)}/tasks/{quote(item.task_id, safe="")}">Open</a></div></article>'
            for item in evaluation_tasks
        )
        + '</div></section>'
    ) if evaluation_tasks else ''}
    <section class="panel">
      <h2>Recent output</h2>
      <pre class="log-block">{html.escape(chr(10).join(_sanitize_display_text(line) for line in display_log_lines) if display_log_lines else "No captured log output yet.")}</pre>
    </section>
    <section class="content-grid">
      <section class="panel compact-panel">
        <h2>Timing and routing</h2>
        <div class="metric-stack">
          <div class="metric-row-block"><div class="metric-row"><span>Created</span><strong>{_display_time(job.created_at)}</strong></div></div>
          <div class="metric-row-block"><div class="metric-row"><span>Started</span><strong>{_display_time(job.started_at)}</strong></div></div>
          <div class="metric-row-block"><div class="metric-row"><span>Finished</span><strong>{_display_time(job.finished_at)}</strong></div></div>
          <div class="metric-row-block"><div class="metric-row"><span>Assigned client</span><strong>{html.escape(_assigned_client_name(job, client_lookup))}</strong></div></div>
          <div class="metric-row-block"><div class="metric-row"><span>Updated</span><strong>{_display_time(job.updated_at)}</strong></div></div>
        </div>
      </section>
      <section class="panel compact-panel">
        <h2>Metadata</h2>
        <pre class="code-block metadata-block">{html.escape(metadata_json)}</pre>
      </section>
    </section>
    """
    action = str(job.spec.action or "").strip()
    active_nav = "learning"
    if action == "evaluate-library-case":
        active_nav = "ablations"
    elif action == "analyze-supervised-consistency":
        active_nav = "analysis"
    elif action in {"pooltool-reward-optimize", "pooltool-reward-value-model"}:
        active_nav = "rewards"
    return _shell(f"Job {job.job_id} | Pattern Server", active_nav=active_nav, body_html=body)


def render_evaluation_task_page(
    *,
    job: JobRecord,
    task: EvaluationTaskDetail,
    clients: list[ClientRecord],
    project_root: Path,
) -> str:
    client_lookup = {client.client_id: client for client in clients}
    success_label = (
        "Success" if task.success is True else "Failed" if task.success is False else "Unknown"
    )
    success_tone = "succeeded" if task.success is True else "failed" if task.success is False else "queued"
    media_html = _render_media_paths(task.media_files, root_dir=task.task_dir)
    extra_files_html = _render_extra_file_list(task.extra_files)
    attempt_cards = []
    for attempt in task.attempts:
        links = []
        if attempt.trace_path is not None:
            links.append(f'<a href="/ui/files?path={quote(_project_relpath(attempt.trace_path))}">trace.json</a>')
        if attempt.evidence_path is not None:
            links.append(f'<a href="/ui/files?path={quote(_project_relpath(attempt.evidence_path))}">trace.evidence.json</a>')
        attempt_cards.append(
            f"""
            <article class="attempt-card">
              <div class="attempt-card-top">
                <div>
                  <h3 class="attempt-card-title">Attempt {attempt.index or "?"}</h3>
                  <p class="metric-subtle">{html.escape(attempt.trace_id or "No trace id")} · {html.escape(attempt.terminal_status or "no terminal status")}</p>
                </div>
                <div class="task-row-meta">
                  {_pill("Success" if attempt.success else "Failed", "succeeded" if attempt.success else "failed")}
                  {f'<span class="mini-chip">{attempt.duration_seconds:.1f}s</span>' if attempt.duration_seconds is not None else ''}
                  {f'<span class="mini-chip">{html.escape(attempt.evidence_status)}</span>' if attempt.evidence_status else ''}
                </div>
              </div>
              <div class="attempt-grid">
                <section class="panel compact-panel">
                  <h3>LLM call</h3>
                  <div class="section-stack">
                    {_json_details("Prompt", attempt.prompt, open_by_default=True)}
                    {_json_details("Router", attempt.router)}
                    <details class="data-details"><summary>Raw response</summary><pre class="code-block">{html.escape(attempt.raw_model_output or "No raw model output recorded.")}</pre></details>
                    {_json_details("Parsed action", attempt.parsed_action, open_by_default=True)}
                  </div>
                </section>
                <section class="panel compact-panel">
                  <h3>Trace and evidence</h3>
                  <div class="asset-inline-links">{''.join(links) if links else '<span class="metric-subtle">No trace files recorded.</span>'}</div>
                  <div class="section-stack">
                    {_json_details("Trace description", attempt.trace_description, open_by_default=True)}
                    {_json_details("Evidence payload", attempt.evidence_payload)}
                    {_json_details("Native metrics", attempt.native_metrics)}
                  </div>
                </section>
              </div>
            </article>
            """
        )
    body = f"""
    <section class="hero">
      <p class="breadcrumb"><a href="/">Home</a> / <a href="/jobs/{html.escape(job.job_id)}">job</a> / task / {html.escape(task.task_id)}</p>
      <h1>{html.escape(task.task_id)}</h1>
      <p>Evaluation task detail with media, attempts, trace evidence, and model responses.</p>
      <div class="hero-stats">
        <div class="stat-chip"><span class="stat-label">Outcome</span><strong class="stat-value">{html.escape(success_label)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Attempts</span><strong class="stat-value">{len(task.attempts)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Media</span><strong class="stat-value">{len(task.media_files)}</strong></div>
        <div class="stat-chip"><span class="stat-label">Completed</span><strong class="stat-value">{_display_time(task.completed_at)}</strong></div>
      </div>
    </section>
    <section class="panel">
      <div class="card-topline">
        <div>
          <h2>Task summary</h2>
          <p class="metric-subtle">{html.escape(_queue_job_title(job))} · {html.escape(_assigned_client_name(job, client_lookup))}</p>
        </div>
        <div class="task-row-meta">
          {_pill(success_label, success_tone)}
          <a class="ghost-link" href="/jobs/{html.escape(job.job_id)}">Back to job</a>
        </div>
      </div>
      <div class="key-value-grid">
        {_key_value_card("Environment", task.env_id)}
        {_key_value_card("Task id", task.task_id)}
        {_key_value_card("Attempts", str(len(task.attempts)))}
        {_key_value_card("Completed", _display_time(task.completed_at))}
      </div>
      <div class="asset-inline-links">
        <a href="/ui/files?path={quote(_project_relpath(task.result_path))}">task_result.json</a>
        <a href="/ui/files?path={quote(_project_relpath(task.task_dir))}">Task directory</a>
      </div>
    </section>
    <section class="panel">
      <h2>Media</h2>
      {media_html}
    </section>
    <section class="panel">
      <h2>Task result</h2>
      <div class="section-stack">
        {_json_details("Result summary", task.payload, open_by_default=True)}
      </div>
    </section>
    <section class="panel">
      <h2>Extra files</h2>
      {extra_files_html}
    </section>
    <section class="panel">
      <h2>Attempts</h2>
      <div class="attempt-stack">{''.join(attempt_cards) if attempt_cards else '<p class="empty">No attempt records found for this task.</p>'}</div>
    </section>
    """
    action = str(job.spec.action or "").strip()
    active_nav = "learning"
    if action == "evaluate-library-case":
        active_nav = "ablations"
    elif action == "analyze-supervised-consistency":
        active_nav = "analysis"
    elif action in {"pooltool-reward-optimize", "pooltool-reward-value-model"}:
        active_nav = "rewards"
    return _shell(
        f"{task.task_id} | {job.job_id} | Pattern Server",
        active_nav=active_nav,
        body_html=body,
    )


def render_public_page(public_url: str | None) -> str:
    public_url_label = html.escape(public_url or "")
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Pattern Server Public</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
      body {{ margin: 0; font-family: Georgia, "Iowan Old Style", serif; background: #f6f1e7; color: #1f1b16; }}
      main {{ max-width: 720px; margin: 0 auto; padding: 28px 18px 48px; }}
      .panel {{ background: white; border: 1px solid #d8ccb6; border-radius: 18px; padding: 18px; box-shadow: 0 10px 26px rgba(45, 35, 20, 0.08); }}
      input, button {{ font: inherit; }}
      input {{ width: 100%; padding: 10px 12px; border-radius: 10px; border: 1px solid #cdbda0; }}
      button {{ margin-top: 12px; padding: 10px 14px; border: 0; border-radius: 10px; background: #0f766e; color: white; cursor: pointer; }}
      pre {{ background: #f7f2e8; padding: 12px; border-radius: 12px; overflow-x: auto; white-space: pre-wrap; }}
      .muted {{ color: #6b6256; }}
    </style>
  </head>
  <body>
    <main>
      <div class="panel">
        <h1>Pattern Server Public View</h1>
        <p class="muted">Public URL: <code>{public_url_label or "unknown"}</code></p>
        <button id="load">Load status</button>
        <h2>Status</h2>
        <pre id="status">Not loaded</pre>
        <h2>Jobs</h2>
        <pre id="jobs">Not loaded</pre>
      </div>
    </main>
    <script>
      const statusNode = document.getElementById("status");
      const jobsNode = document.getElementById("jobs");
      async function load() {{
        const statusResp = await fetch("/public/v1/status");
        const jobsResp = await fetch("/public/v1/jobs");
        statusNode.textContent = JSON.stringify(await statusResp.json(), null, 2);
        jobsNode.textContent = JSON.stringify(await jobsResp.json(), null, 2);
      }}
      document.getElementById("load").addEventListener("click", () => {{
        load().catch((error) => {{
          statusNode.textContent = String(error);
        }});
      }});
    </script>
  </body>
</html>
"""


def public_jobs_payload(jobs: list[JobRecord]) -> dict[str, object]:
    return {
        "jobs": [
            {
                "job_id": job.job_id,
                "action": job.spec.action,
                "status": job.status.value,
                "env_id": job.spec.metadata.get("env_id"),
                "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat(),
                "message": job.message,
            }
            for job in jobs[:20]
        ]
    }


def public_status_payload(
    *,
    jobs: list[JobRecord],
    public_access: PublicAccessState,
) -> dict[str, object]:
    counts = Counter(job.status.value for job in jobs)
    return {
        "server": "pattern-server",
        "job_counts": dict(counts),
        "public_url": public_access.public_url,
        "generated_at": (
            public_access.last_synced_at.isoformat()
            if public_access.last_synced_at is not None
            else None
        ),
    }


def _required_catalog_deepphy_percent(
    *,
    family: str,
    selected: JobRecord | None,
    project_root: Path,
) -> float | None:
    eval_root = _required_catalog_deepphy_eval_root(
        family=family,
        selected=selected,
        project_root=project_root,
    )
    if eval_root is None:
        return None
    counts = _required_catalog_deepphy_eval_counts_cached(
        str(eval_root),
        _required_catalog_deepphy_eval_marker(eval_root),
    )
    if counts is None:
        return None
    success_count, _, total_tasks = counts
    if total_tasks <= 0:
        return None
    return max(0.0, min(100.0, (success_count / total_tasks) * 100.0))


def _required_catalog_panel_payload(
    *,
    title: str,
    kicker: str,
    families: tuple[str, ...],
    catalog: dict[tuple[str, str], str],
    jobs: list[JobRecord],
    project_root: Path,
    allow_paused_as_healthy: bool = False,
) -> dict[str, object]:
    env_ids = sorted({env_id for env_id, _ in catalog})
    total_slots = len(env_ids) * len(families)
    healthy_slots = 0
    active_slots = 0
    job_lookup = {job.job_id: job for job in jobs}
    repeat_learning_statuses: dict[tuple[str, str], Counter[str]] = {}
    repeat_required_statuses: dict[tuple[str, str], Counter[str]] = {}
    for job in jobs:
        if not _required_ui_job_visible(job):
            continue
        repeat_family = _job_repeat_learning_family(job)
        if repeat_family is None:
            repeat_family = _job_repeat_required_family(job)
            if repeat_family is None:
                continue
            if not _required_repeat_job_counts_toward_target(job):
                continue
            repeat_env = str(job.spec.metadata.get("env_id") or job.environment or "").strip()
            if not repeat_env:
                continue
            repeat_required_statuses.setdefault((repeat_env, repeat_family), Counter())[job.status.value] += 1
            continue
        repeat_env = str(job.spec.metadata.get("env_id") or job.environment or "").strip()
        if not repeat_env:
            continue
        repeat_learning_statuses.setdefault((repeat_env, repeat_family), Counter())[job.status.value] += 1
    env_cards: list[dict[str, object]] = []
    for env_id in env_ids:
        rows: list[dict[str, object]] = []
        tracked_count = 0
        for family in families:
            if catalog.get((env_id, family)) or (
                family == "Automatic DeepPHY" and catalog.get((env_id, "Automatic"))
            ):
                tracked_count += 1
            selected, selected_status, detail = _required_catalog_selection(
                env_id=env_id,
                family=family,
                catalog=catalog,
                job_lookup=job_lookup,
                jobs=jobs,
                project_root=project_root,
            )
            href = _required_catalog_job_href(
                env_id=env_id,
                family=family,
                selected=selected,
                project_root=project_root,
            )
            deepphy_percent = _required_catalog_deepphy_percent(
                family=family,
                selected=selected,
                project_root=project_root,
            )
            repeat_status_counts = repeat_learning_statuses.get((env_id, family), Counter())
            if not repeat_status_counts:
                repeat_status_counts = repeat_required_statuses.get((env_id, family), Counter())
            is_healthy = False
            if selected is not None:
                is_healthy = (
                    selected_status in REQUIRED_HEALTHY_STATUSES
                    or (allow_paused_as_healthy and selected_status == "paused")
                )
                if not allow_paused_as_healthy:
                    is_healthy = _required_job_is_healthy(selected)
            if repeat_status_counts:
                has_incomplete_repeats = any(
                    repeat_status_counts.get(status, 0) > 0
                    for status in ACTIVE_QUEUE_STATUSES | {"queued", "paused"}
                )
                if has_incomplete_repeats:
                    is_healthy = False
            if is_healthy:
                healthy_slots += 1
            if selected is not None and selected_status in ACTIVE_QUEUE_STATUSES:
                active_slots += 1
            visual = "missing"
            if selected is not None:
                if selected_status in ACTIVE_QUEUE_STATUSES:
                    visual = "active"
                elif repeat_status_counts:
                    queued_repeats = repeat_status_counts.get("queued", 0)
                    active_repeats = sum(repeat_status_counts.get(status, 0) for status in ACTIVE_QUEUE_STATUSES)
                    paused_repeats = repeat_status_counts.get("paused", 0)
                    if active_repeats > 0:
                        visual = "active"
                    elif queued_repeats > 0:
                        visual = "missing"
                    elif paused_repeats > 0:
                        visual = "blocked"
                elif _job_is_prerequisite_blocked(selected):
                    visual = "blocked"
                elif is_healthy:
                    visual = "blocked" if selected_status == "paused" else "covered"
                elif selected_status in FAILED_QUEUE_STATUSES:
                    visual = "failed"
            rows.append(
                {
                    "family": family,
                    "job_id": selected.job_id if selected is not None else None,
                    "status": selected_status,
                    "detail": detail,
                    "href": href,
                    "visual": visual,
                    "deepphy_percent": deepphy_percent,
                    "repeat_status_counts": dict(repeat_status_counts),
                    "repeat_progress": _repeat_status_visuals(repeat_status_counts, family=family),
                }
            )
        env_cards.append(
            {
                "env_id": env_id,
                "tracked_count": tracked_count,
                "rows": rows,
            }
        )
    return {
        "title": title,
        "kicker": kicker,
        "healthy_slots": healthy_slots,
        "total_slots": total_slots,
        "active_slots": active_slots,
        "envs": env_cards,
    }


def _required_repeat_target_for_family(family: str) -> int:
    normalized = str(family or "").strip().lower()
    if normalized == "image-only deepphy":
        return IMAGE_ONLY_EVAL_TARGET_RUNS
    return REPEAT_LEARNING_TARGET_RUNS


def _repeat_eval_within_target(job: JobRecord, *, project_root: Path) -> bool:
    eval_index = _repeat_eval_run_index(job, project_root=project_root)
    if eval_index is None:
        return True
    return eval_index <= REPEAT_EVALS_PER_RUN


def _required_overall_progress_payload(*, jobs: list[JobRecord], project_root: Path) -> dict[str, object] | None:
    catalog = _required_job_catalog_for_project(project_root)
    if not catalog:
        return None
    env_ids = sorted({env_id for env_id, _ in catalog})
    if not env_ids:
        return None
    job_lookup = {job.job_id: job for job in jobs}
    repeat_learning_statuses: dict[tuple[str, str], Counter[str]] = {}
    repeat_required_statuses: dict[tuple[str, str], Counter[str]] = {}
    for job in jobs:
        repeat_family = _job_repeat_learning_family(job)
        if repeat_family is None:
            repeat_family = _job_repeat_required_family(job)
            if repeat_family is None:
                continue
            if not _required_repeat_job_counts_toward_target(job):
                continue
            repeat_env = str(job.spec.metadata.get("env_id") or job.environment or "").strip()
            if not repeat_env:
                continue
            repeat_required_statuses.setdefault((repeat_env, repeat_family), Counter())[job.status.value] += 1
            continue
        repeat_env = str(job.spec.metadata.get("env_id") or job.environment or "").strip()
        if not repeat_env:
            continue
        repeat_learning_statuses.setdefault((repeat_env, repeat_family), Counter())[job.status.value] += 1

    totals = Counter[str]()
    for env_id in env_ids:
        for family in REQUIRED_JOB_FAMILIES:
            job_ref = catalog.get((env_id, family))
            if not job_ref and not (family == "Automatic DeepPHY" and catalog.get((env_id, "Automatic"))):
                continue
            repeat_status_counts = repeat_learning_statuses.get((env_id, family), Counter())
            if not repeat_status_counts:
                repeat_status_counts = repeat_required_statuses.get((env_id, family), Counter())
            if family == "Image-only DeepPHY":
                target = _required_repeat_target_for_family(family)
                remaining = target
                completed = 0
                active = 0
                queued = 0
                paused = 0
                failed = 0
                completed += min(repeat_status_counts.get("succeeded", 0), remaining)
                remaining = max(0, remaining - min(repeat_status_counts.get("succeeded", 0), remaining))
                active_repeat = min(
                    sum(repeat_status_counts.get(status, 0) for status in ACTIVE_QUEUE_STATUSES),
                    remaining,
                )
                active += active_repeat
                remaining -= active_repeat
                queued_repeat = min(repeat_status_counts.get("queued", 0), remaining)
                queued += queued_repeat
                remaining -= queued_repeat
                paused_repeat = min(repeat_status_counts.get("paused", 0), remaining)
                paused += paused_repeat
                remaining -= paused_repeat
                failed_repeat = min(sum(repeat_status_counts.get(status, 0) for status in FAILED_QUEUE_STATUSES), remaining)
                failed += failed_repeat
                remaining -= failed_repeat
                totals["total"] += target
                totals["completed"] += completed
                totals["active"] += active
                totals["queued"] += queued
                totals["paused"] += paused
                totals["failed"] += failed
                totals["missing"] += remaining
                continue
            if _catalog_tracks_repeat_batch(job_ref):
                status_counts = repeat_status_counts
                target = _required_repeat_target_for_family(family)
                remaining = target
                completed = min(status_counts.get("succeeded", 0), remaining)
                remaining -= completed
                active = min(
                    sum(status_counts.get(status, 0) for status in ACTIVE_QUEUE_STATUSES),
                    remaining,
                )
                remaining -= active
                queued = min(status_counts.get("queued", 0), remaining)
                remaining -= queued
                paused = min(status_counts.get("paused", 0), remaining)
                remaining -= paused
                failed = min(sum(status_counts.get(status, 0) for status in FAILED_QUEUE_STATUSES), remaining)
                remaining -= failed
                totals["total"] += target
                totals["completed"] += completed
                totals["active"] += active
                totals["queued"] += queued
                totals["paused"] += paused
                totals["failed"] += failed
                totals["missing"] += remaining
                continue

            totals["total"] += 1
            selected = job_lookup.get(job_ref or "")
            if selected is None:
                totals["missing"] += 1
                continue
            status = selected.status.value
            if status == "succeeded":
                totals["completed"] += 1
            elif status in ACTIVE_QUEUE_STATUSES:
                totals["active"] += 1
            elif status == "queued":
                totals["queued"] += 1
            elif status == "paused":
                totals["paused"] += 1
            elif status in FAILED_QUEUE_STATUSES:
                totals["failed"] += 1
            else:
                totals["missing"] += 1

    total_units = int(totals["total"])
    if total_units <= 0:
        return None
    segment_specs = [
        ("completed", "Done", "is-done"),
        ("active", "Active", "is-active"),
        ("queued", "Queued", "is-queued"),
        ("paused", "Paused", "is-paused"),
        ("failed", "Failed", "is-failed"),
        ("missing", "Missing", "is-missing"),
    ]
    segments: list[dict[str, object]] = []
    chips: list[dict[str, object]] = []
    for key, label, class_name in segment_specs:
        count = int(totals[key])
        if count <= 0:
            continue
        width = max(0.0, min(100.0, (count / total_units) * 100.0))
        segments.append(
            {
                "key": key,
                "label": label,
                "class_name": class_name,
                "count": count,
                "width": width,
            }
        )
        chips.append({"label": label, "count": count, "class_name": class_name})
    return {
        "total": total_units,
        "completed": int(totals["completed"]),
        "active": int(totals["active"]),
        "queued": int(totals["queued"]),
        "paused": int(totals["paused"]),
        "failed": int(totals["failed"]),
        "missing": int(totals["missing"]),
        "completed_percent": int(round((int(totals["completed"]) / total_units) * 100.0)),
        "segments": segments,
        "chips": chips,
    }


def _render_required_overall_progress_bar(*, jobs: list[JobRecord], project_root: Path) -> str:
    payload = _required_overall_progress_payload(jobs=jobs, project_root=project_root)
    if payload is None:
        return ""
    segments_html = "".join(
        (
            f'<span class="required-overall-segment {html.escape(str(segment["class_name"]))}" '
            f'style="width: {float(segment["width"]):.3f}%" '
            f'title="{html.escape(str(segment["label"]))}: {int(segment["count"])}"></span>'
        )
        for segment in payload["segments"]
    )
    chips_html = "".join(
        (
            f'<span class="required-overall-chip {html.escape(str(chip["class_name"]))}">'
            f'<span class="required-overall-chip-label">{html.escape(str(chip["label"]))}</span>'
            f'{int(chip["count"])}'
            "</span>"
        )
        for chip in payload["chips"]
    )
    return f"""
    <section class="panel compact-panel required-overall-panel">
      <div class="card-topline">
        <div>
          <p class="section-kicker">Required Progress</p>
          <h2>Required Runs Overall</h2>
          <p class="metric-subtle">Tracks only the required coverage defined in JOBS.md: repeat learning runs and standalone image-only evaluations.</p>
        </div>
        <div class="stats-inline">
          <span class="pill {'status-running' if int(payload['completed']) > 0 else 'status-paused'}">{int(payload['completed'])}/{int(payload['total'])} complete</span>
          <span class="pill status-running">{int(payload['completed_percent'])}% done</span>
        </div>
      </div>
      <div class="required-overall-bar-wrap">
        <div class="required-overall-bar">{segments_html}</div>
        <div class="required-overall-bar-label">{int(payload['completed'])} of {int(payload['total'])} required units complete</div>
      </div>
      <div class="required-overall-chip-row">{chips_html}</div>
    </section>
    """


def _clients_panel_payload(
    *,
    clients: list[ClientRecord],
    jobs: list[JobRecord],
    limit: int | None = None,
) -> dict[str, object]:
    clients = _dashboard_active_clients(clients)
    if not clients:
        return {
            "title": "Clients",
            "kicker": "Clients",
            "summary": None,
            "empty_message": "No active clients right now.",
            "clients": [],
        }
    job_lookup = {job.job_id: job for job in jobs}
    now = datetime.now(tz=DISPLAY_TZ)
    visible_clients = sorted(
        clients,
        key=lambda item: (
            0 if item.status.value == "online" else 1,
            0 if item.current_jobs else 1,
            -item.last_heartbeat.timestamp(),
            _client_title(item).lower(),
        ),
    )
    hidden_count = 0
    if limit is not None and len(visible_clients) > limit:
        hidden_count = len(visible_clients) - limit
        visible_clients = visible_clients[:limit]
    summary = (
        f"Showing {len(visible_clients)} of {len(clients)} active clients."
        if hidden_count
        else None
    )
    client_items: list[dict[str, object]] = []
    for client in visible_clients:
        current_jobs: list[dict[str, object]] = []
        for job_id in client.current_jobs:
            job = job_lookup.get(job_id)
            if job is None:
                current_jobs.append(
                    {
                        "job_id": job_id,
                        "missing": True,
                        "title": job_id[:8],
                        "status": "missing",
                        "env_id": None,
                        "detail": "This lease is still on the client but the server no longer has the job record.",
                        "attention_reason": None,
                        "href": None,
                    }
                )
                continue
            attention = _job_attention_reason(job, client_lookup={client.client_id: client}, now=now)
            current_jobs.append(
                {
                    "job_id": job.job_id,
                    "missing": False,
                    "title": _queue_job_title(job),
                    "status": job.status.value,
                    "env_id": _job_env_id(job),
                    "detail": _job_last_activity_summary(job, now=now),
                    "attention_reason": attention[1] if attention is not None else None,
                    "href": f"/jobs/{job.job_id}",
                }
            )
        notes: list[str] = []
        if client.status.value == "offline" and client.current_jobs:
            notes.append("This client is offline but still lists active jobs. Reconcile or requeue those leases.")
        if len(client.current_jobs) >= _effective_client_slots(client):
            notes.append("This client is full. New work will wait for a free slot.")
        client_items.append(
            {
                "client_id": client.client_id,
                "title": _client_title(client),
                "status": client.status.value,
                "hostname": client.hostname or "unknown",
                "slots_used": len(client.current_jobs),
                "slots_total": _effective_client_slots(client),
                "last_heartbeat": client.last_heartbeat.isoformat(),
                "current_jobs": current_jobs,
                "notes": notes,
                "href": f"/clients/{client.client_id}",
            }
        )
    return {
        "title": "Clients",
        "kicker": "Clients",
        "summary": summary,
        "empty_message": None,
        "clients": client_items,
    }


def dashboard_home_payload(
    *,
    public_access: PublicAccessState,
    clients: list[ClientRecord],
    jobs: list[JobRecord],
    project_root: Path,
) -> dict[str, object]:
    counts = Counter(job.status.value for job in jobs)
    return {
        "hero_stats": {
            "jobs": len(jobs),
            "queued": counts.get("queued", 0),
            "running": sum(1 for job in jobs if job.status.value in ACTIVE_QUEUE_STATUSES),
            "clients_online": sum(1 for client in clients if client.status.value == "online"),
        },
        "public_access": {
            "active": public_access.active,
            "public_url": public_access.public_url,
            "desired_enabled": public_access.desired_enabled,
            "last_error": public_access.last_error,
            "updated_at": public_access.updated_at.isoformat() if public_access.updated_at else None,
        },
        "required_runs": _required_catalog_panel_payload(
            title="Required Runs",
            kicker="Coverage",
            families=REQUIRED_JOB_FAMILIES,
            catalog=_required_job_catalog_for_project(project_root),
            jobs=jobs,
            project_root=project_root,
            allow_paused_as_healthy=False,
        ),
        "reward_programs": _required_catalog_panel_payload(
            title="Required Reward Program Runs",
            kicker="Reward programs",
            families=REQUIRED_REWARD_PROGRAM_JOB_FAMILIES,
            catalog=_required_reward_program_job_catalog_for_project(project_root),
            jobs=jobs,
            project_root=project_root,
            allow_paused_as_healthy=True,
        ),
        "experiment_links": [
            {
                "title": "Label Statistics",
                "description": "Per-label generation metrics aggregated across all discovered run artifacts.",
                "href": "/statistics",
                "badge": "All envs",
            },
            {
                "title": "Reward Programs",
                "description": "Optimization, value-model runs, and sweep dashboards.",
                "href": "/rewards",
                "badge": "PoolTool",
            },
            {
                "title": "Library Ablations",
                "description": "Tracked queue for supervised and unsupervised required-ablation libraries.",
                "href": "/ablations",
                "badge": "All envs",
            },
        ],
        "clients_panel": _clients_panel_payload(
            clients=clients,
            jobs=jobs,
            limit=HOME_CLIENT_PANEL_LIMIT,
        ),
    }
