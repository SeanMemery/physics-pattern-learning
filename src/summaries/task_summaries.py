from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any
from PIL import Image

from pattern_server.env_catalog import list_environment_libraries
from ensemble_library_training.library_collections import resolve_canonical_library
from video_annotations.video_renders import (
    SERVER_PROJECT_ROOT,
    SUPPORTED_VIDEO_RENDER_ENVS,
    TraceMediaPaths,
    VideoRenderTask,
    _client_project_root,
    _isoformat,
    _load_state as _load_video_state,
    _normalize_target_count,
    _pid_is_alive,
    _project_relpath,
    _server_src_root,
    python_executable_for_env,
    replay_task_video,
    resolve_trace_media_paths,
)


TASK_SUMMARY_STATE_RELPATH = Path("state/task_summaries/state.json")
TASK_SUMMARY_CONFIG_RELPATH = Path("state/task_summaries/config.json")
TASK_SUMMARY_FAILURE_FILENAME = "summary_failure.json"
DEFAULT_TASK_SUMMARY_TARGET = 12
TASK_SUMMARY_MAX_PARALLEL_REQUESTS = 8
TASK_SUMMARY_LLM_SLOT_COUNT = 8
DEFAULT_TASK_SUMMARY_LLM_BASE_URL = "http://127.0.0.1:8788"
DEFAULT_TASK_SUMMARY_LLM_MODEL = "Qwen3.6-35B-A3B-UD-Q4_K_M"
DEFAULT_ENABLED_SUMMARY_VARIANTS = ("image_only", "supervised", "unsupervised", "automatic")
BO_SUMMARY_ENTRY_LIMIT = 25
SUMMARY_MIN_SHARED_PATTERN_COUNT = 10
SUMMARY_FRAME_RATE = 2
SUMMARY_FRAME_COUNT = 20
SUMMARY_IMAGE_MAX_WIDTH = 224
SUMMARY_IMAGE_JPEG_QUALITY = 40
SUMMARY_ANNOTATION_PATTERN_LIMIT = 50
SUMMARY_REASONING_EXTRACTION_MAX_TOKENS = 2048
_RESUME_SYNC_SUFFIX_PATTERN = re.compile(r"^(?P<base>.+)\.__resume_sync__[0-9a-f]{32}$")
DEFAULT_IMAGE_ONLY_SUMMARY_PROMPT = (
    "Describe the physics simulation shown in the provided frames. "
    "Focus on the initial setup, the key causal interactions, and the final outcome. "
    "Be concrete about object motion and collisions. Keep the summary to 3-5 sentences."
)
DEFAULT_ANNOTATION_SUMMARY_PROMPT = (
    "Describe the physics simulation using the provided initial frame and the provided learned pattern annotation as context. "
    "Focus on the initial setup, the key causal interactions, and the final outcome. "
    "Use the annotation to sharpen the description, but only mention patterns that are consistent with the simulation. "
    "For pool or billiards simulations, try to identify balls by their visible colors or appearance rather than by ball numbers or ids. "
    "Assume there is exactly one shot: the cue ball is struck only once, and everything that happens is the consequence of that single cue strike. "
    "If the annotations appear to imply multiple shots, turns, resets, respawns, or repeated cue strikes, ignore that and still summarize the clip as one continuous shot only. "
    "Do not use phrases such as another shot, next shot, later shot, subsequent strike, reset, respawn, or turn. "
    "Make it explicit when a ball is potted or pocketed. "
    "Be specific about which balls are pocketed and which pockets they enter. "
    "Use these pocket codes when relevant: lb=left bottom pocket, rb=right bottom pocket, lc=left centre pocket, rc=right centre pocket, lt=left top pocket, rt=right top pocket. "
    "Try not to use numbers for positions; instead describe locations on the table using relative phrases such as bottom-left, center-right, top rail, or near the upper-right corner, where the coordinate intuition is (0,0) at the bottom left and (1,2) at the top right. "
    "Keep the summary to 3-5 sentences."
)


@dataclass(frozen=True)
class TaskSummaryTask:
    env_id: str
    source_run_id: str
    source_run_dir: Path
    source_result_path: Path
    case_id: str
    task_id: str
    payload: dict[str, Any]
    output_run_dir: Path
    output_task_dir: Path
    summary_path: Path


@dataclass(frozen=True)
class TaskSummaryEntry:
    env_id: str
    source_run_id: str
    task_id: str
    task_dir: Path
    summary_path: Path
    video_path: Path | None
    overhead_video_path: Path | None
    initial_image_path: Path | None
    summary_payload: dict[str, Any]
    updated_at: datetime


def _canonical_source_run_id(raw_run_id: str) -> str:
    value = str(raw_run_id or "").strip()
    while value.startswith("video-render-"):
        value = value[len("video-render-") :]
    while True:
        match = _RESUME_SYNC_SUFFIX_PATTERN.match(value)
        if not match:
            break
        value = str(match.group("base") or "").strip()
    return value


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _state_path(project_root: Path) -> Path:
    return project_root / TASK_SUMMARY_STATE_RELPATH


def _config_path(project_root: Path) -> Path:
    return project_root / TASK_SUMMARY_CONFIG_RELPATH


def _state_root(project_root: Path) -> Path:
    return _state_path(project_root).parent


def _log_path(project_root: Path, env_id: str) -> Path:
    return _state_root(project_root) / f"{env_id}.log"


def _load_state(project_root: Path) -> dict[str, dict[str, Any]]:
    path = _state_path(project_root)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(env_id): dict(entry)
        for env_id, entry in payload.items()
        if isinstance(env_id, str) and isinstance(entry, dict)
    }


def _write_state(project_root: Path, state: dict[str, dict[str, Any]]) -> None:
    path = _state_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _merge_env_state(project_root: Path, env_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    state = _load_state(project_root)
    current = dict(state.get(env_id, {}))
    current.update(updates)
    current["env_id"] = env_id
    current["updated_at"] = _isoformat()
    state[env_id] = current
    _write_state(project_root, state)
    return current


def _default_config_payload() -> dict[str, Any]:
    return {
        "image_only_prompt_template": DEFAULT_IMAGE_ONLY_SUMMARY_PROMPT,
        "annotation_prompt_template": DEFAULT_ANNOTATION_SUMMARY_PROMPT,
        "enabled_variants": list(DEFAULT_ENABLED_SUMMARY_VARIANTS),
        "supervised_library_strategy": "latest",
    }


def load_task_summary_config(project_root: Path) -> dict[str, Any]:
    path = _config_path(project_root)
    payload = _default_config_payload()
    if not path.is_file():
        return payload
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return payload
    if isinstance(raw, dict):
        legacy_prompt = str(raw.get("prompt_template") or "").strip()
        image_prompt = str(raw.get("image_only_prompt_template") or legacy_prompt).strip()
        annotation_prompt = str(raw.get("annotation_prompt_template") or legacy_prompt).strip()
        if image_prompt:
            payload["image_only_prompt_template"] = image_prompt
        if annotation_prompt:
            payload["annotation_prompt_template"] = annotation_prompt
        raw_variants = raw.get("enabled_variants")
        if isinstance(raw_variants, list):
            variants = [
                str(item).strip()
                for item in raw_variants
                if str(item).strip() in DEFAULT_ENABLED_SUMMARY_VARIANTS
            ]
            if variants:
                payload["enabled_variants"] = variants
        strategy = str(raw.get("supervised_library_strategy") or "").strip().lower()
        if strategy in {"latest", "best"}:
            payload["supervised_library_strategy"] = strategy
    return payload


def save_task_summary_config(
    project_root: Path,
    *,
    image_only_prompt_template: str,
    annotation_prompt_template: str,
    enabled_variants: list[str] | None = None,
    supervised_library_strategy: str | None = None,
) -> dict[str, Any]:
    variants = [
        str(item).strip()
        for item in (enabled_variants or list(DEFAULT_ENABLED_SUMMARY_VARIANTS))
        if str(item).strip() in DEFAULT_ENABLED_SUMMARY_VARIANTS
    ]
    payload = {
        "image_only_prompt_template": str(image_only_prompt_template or "").strip() or DEFAULT_IMAGE_ONLY_SUMMARY_PROMPT,
        "annotation_prompt_template": str(annotation_prompt_template or "").strip() or DEFAULT_ANNOTATION_SUMMARY_PROMPT,
        "enabled_variants": variants or list(DEFAULT_ENABLED_SUMMARY_VARIANTS),
        "supervised_library_strategy": (
            str(supervised_library_strategy or "latest").strip().lower()
            if str(supervised_library_strategy or "").strip().lower() in {"latest", "best"}
            else "latest"
        ),
    }
    path = _config_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _summary_bundle_is_complete(payload: dict[str, Any]) -> bool:
    variants = payload.get("variants")
    if not isinstance(variants, dict):
        return False
    enabled_variants = payload.get("enabled_variants")
    if isinstance(enabled_variants, list):
        required_keys = [str(item).strip() for item in enabled_variants if str(item).strip()]
    else:
        required_keys = [str(key).strip() for key in variants.keys() if str(key).strip()]
    if not required_keys:
        return False
    for key in required_keys:
        item = variants.get(key)
        if not isinstance(item, dict):
            return False
        summary_text = str(item.get("summary_text") or "").strip()
        if not summary_text:
            return False
    return True


def _summary_path_is_from_backup(summary_path: Path) -> bool:
    try:
        run_dir_name = summary_path.parents[3].name
    except IndexError:
        run_dir_name = summary_path.parent.name
    return ".bak." in run_dir_name


def _task_summary_request_parallelism(request_count: int) -> int:
    return max(1, min(TASK_SUMMARY_MAX_PARALLEL_REQUESTS, int(request_count or 0)))


def _llm_slot_root(project_root: Path) -> Path:
    return _state_root(project_root) / "llm_slots"


@contextmanager
def _acquire_llm_request_slot(project_root: Path):
    slot_root = _llm_slot_root(project_root)
    slot_root.mkdir(parents=True, exist_ok=True)
    while True:
        for slot_index in range(TASK_SUMMARY_LLM_SLOT_COUNT):
            slot_path = slot_root / f"slot-{slot_index}.lock"
            handle = slot_path.open("a+b")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    handle.close()
                return
            except BlockingIOError:
                handle.close()
        time.sleep(0.05)


def discover_task_summary_tasks(project_root: Path, env_id: str) -> list[TaskSummaryTask]:
    env_root = project_root / "data" / "evaluations" / env_id
    if not env_root.is_dir():
        return []
    tasks: list[TaskSummaryTask] = []
    source_run_dirs = sorted(
        [path for path in env_root.glob("*image-only*") if path.is_dir() and ".bak." not in path.name],
        key=lambda path: (
            0 if (env_root / f"video-render-{path.name}").is_dir() else 1,
            path.name,
        ),
    )
    for source_run_dir in source_run_dirs:
        output_run_dir = env_root / f"task-summaries-{source_run_dir.name}"
        for source_result_path in sorted(source_run_dir.glob("tasks/*/*/task_result.json")):
            try:
                payload = json.loads(source_result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or payload.get("success") is not True:
                continue
            case_id = source_result_path.parent.parent.name
            task_dir_name = source_result_path.parent.name
            task_id = str(payload.get("task_id") or task_dir_name).strip() or task_dir_name
            output_task_dir = output_run_dir / "tasks" / case_id / task_dir_name
            summary_path = output_task_dir / "summary.json"
            failure_path = output_task_dir / TASK_SUMMARY_FAILURE_FILENAME
            if summary_path.is_file():
                try:
                    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    summary_payload = {}
                if isinstance(summary_payload, dict) and _summary_bundle_is_complete(summary_payload):
                    continue
            if failure_path.is_file():
                try:
                    failure_payload = json.loads(failure_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    failure_payload = {}
                if isinstance(failure_payload, dict) and str(failure_payload.get("status") or "").strip().lower() == "fatal":
                    error_text = str(failure_payload.get("error") or "").strip()
                    if not (
                        "No summary video frames available" in error_text
                        and _task_has_authoritative_summary_media(
                            project_root,
                            env_id,
                            source_run_dir.name,
                            case_id,
                            task_dir_name,
                        )
                    ):
                        continue
            tasks.append(
                TaskSummaryTask(
                    env_id=env_id,
                    source_run_id=source_run_dir.name,
                    source_run_dir=source_run_dir,
                    source_result_path=source_result_path,
                    case_id=case_id,
                    task_id=task_id,
                    payload=payload,
                    output_run_dir=output_run_dir,
                    output_task_dir=output_task_dir,
                    summary_path=summary_path,
                )
            )
    return tasks


def _write_summary_failure(task: TaskSummaryTask, *, error: str, status: str = "fatal") -> None:
    payload = {
        "env_id": task.env_id,
        "task_id": task.task_id,
        "source_run_id": task.source_run_id,
        "status": status,
        "error": str(error or "").strip(),
        "updated_at": _isoformat(),
    }
    path = task.output_task_dir / TASK_SUMMARY_FAILURE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _list_completed_task_summaries_for_env(project_root: Path, env_id: str) -> list[TaskSummaryEntry]:
    env_root = project_root / "data" / "evaluations" / env_id
    entries: list[TaskSummaryEntry] = []
    best_bo_records = _load_best_bo_summary_task_records(project_root, env_id)
    existing_summary_entries = _load_existing_summary_entries_by_task(project_root, env_id)
    config = load_task_summary_config(project_root)
    supervised_library_strategy = str(config.get("supervised_library_strategy") or "latest").strip().lower() or "latest"
    if best_bo_records:
        for record in best_bo_records.values():
            if not isinstance(record, dict):
                continue
            task_dir = record.get("task_dir")
            summary_path = record.get("summary_path")
            updated_at = record.get("updated_at")
            if not isinstance(task_dir, Path) or not isinstance(summary_path, Path) or not isinstance(updated_at, datetime):
                continue
            task_key = (
                str(record.get("source_run_id") or ""),
                str(record.get("task_id") or ""),
            )
            existing_entry = existing_summary_entries.get(task_key)
            summary_payload = {
                "schema_version": "task_summaries.v1",
                "env_id": env_id,
                "task_id": str(record.get("task_id") or ""),
                "source_run_id": str(record.get("source_run_id") or ""),
                "enabled_variants": sorted((record.get("variants") or {}).keys()),
                "variants": dict(record.get("variants") or {}),
            }
            if isinstance(existing_entry, dict):
                existing_payload = existing_entry.get("summary_payload")
                if isinstance(existing_payload, dict):
                    summary_payload = _merge_bo_variants_into_summary_payload(
                        summary_payload=existing_payload,
                        bo_variants=(record.get("variants") or {}),
                    )
                else:
                    summary_payload = _merge_summary_variants(
                        base_payload=summary_payload,
                        extra_variants=(existing_entry.get("variants") or {}),
                    )
            entries.append(
                TaskSummaryEntry(
                    env_id=env_id,
                    source_run_id=str(record.get("source_run_id") or ""),
                    task_id=str(record.get("task_id") or ""),
                    task_dir=existing_entry.get("task_dir") if isinstance(existing_entry, dict) and isinstance(existing_entry.get("task_dir"), Path) else task_dir,
                    summary_path=existing_entry.get("summary_path") if isinstance(existing_entry, dict) and isinstance(existing_entry.get("summary_path"), Path) else summary_path,
                    video_path=(
                        existing_entry.get("video_path")
                        if isinstance(existing_entry, dict) and isinstance(existing_entry.get("video_path"), Path)
                        else (record.get("video_path") if isinstance(record.get("video_path"), Path) else None)
                    ),
                    overhead_video_path=(
                        existing_entry.get("overhead_video_path")
                        if isinstance(existing_entry, dict) and isinstance(existing_entry.get("overhead_video_path"), Path)
                        else (record.get("overhead_video_path") if isinstance(record.get("overhead_video_path"), Path) else None)
                    ),
                    initial_image_path=(
                        existing_entry.get("initial_image_path")
                        if isinstance(existing_entry, dict) and isinstance(existing_entry.get("initial_image_path"), Path)
                        else (record.get("initial_image_path") if isinstance(record.get("initial_image_path"), Path) else None)
                    ),
                    summary_payload=summary_payload,
                    updated_at=(
                        existing_entry.get("updated_at")
                        if isinstance(existing_entry, dict) and isinstance(existing_entry.get("updated_at"), datetime)
                        else updated_at
                    ),
                )
            )
        return _select_summary_entries(entries)
    if env_root.is_dir():
        for summary_path in sorted(env_root.glob("task-summaries-*/tasks/*/*/summary.json")):
            if _summary_path_is_from_backup(summary_path):
                continue
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or not _summary_bundle_is_complete(payload):
                continue
            task_dir = summary_path.parent
            task_result_payload: dict[str, Any] = {}
            task_result_path = task_dir / "task_result.json"
            if task_result_path.is_file():
                try:
                    task_result_payload = json.loads(task_result_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    task_result_payload = {}
            source_run_id = _canonical_source_run_id(str(payload.get("source_run_id") or ""))
            authoritative_payload: dict[str, Any] = {}
            if source_run_id:
                authoritative_result_path = (
                    project_root
                    / "data"
                    / "evaluations"
                    / env_id
                    / f"video-render-{source_run_id}"
                    / "tasks"
                    / summary_path.parent.parent.name
                    / summary_path.parent.name
                    / "task_result.json"
                )
                if authoritative_result_path.is_file():
                    try:
                        authoritative_payload = json.loads(authoritative_result_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        authoritative_payload = {}
            media_paths = TraceMediaPaths(None, None, None, None)
            media_payload = authoritative_payload if isinstance(authoritative_payload, dict) and authoritative_payload else task_result_payload
            if isinstance(media_payload, dict):
                trace_path = _normalize_summary_asset_path(
                    project_root,
                    str(media_payload.get("final_trace_path") or ""),
                )
                media_paths = resolve_trace_media_paths(trace_path)
            video_path = media_paths.default_video_path or next(
                (
                    path
                    for path in sorted(task_dir.rglob("*"))
                    if path.is_file() and path.suffix.lower() in {".mp4", ".webm", ".mov", ".m4v"}
                ),
                None,
            )
            overhead_video_path = media_paths.overhead_video_path
            initial_image_path = media_paths.default_initial_image_path or next(
                (
                    path
                    for path in sorted(task_dir.rglob("*"))
                    if path.is_file() and path.name == "initial_scene.png"
                ),
                None,
            )
            if initial_image_path is None:
                if isinstance(media_payload, dict):
                    initial_image_path = _normalize_summary_asset_path(
                        project_root,
                        str(media_payload.get("initial_image_path") or ""),
                    )
            task_id = str(payload.get("task_id") or task_dir.name).strip() or task_dir.name
            task_key = (source_run_id, task_id)
            bo_record = best_bo_records.pop(task_key, None)
            merged_payload = _merge_bo_variants_into_summary_payload(
                summary_payload=_filter_stale_annotation_variants(
                    project_root=project_root,
                    env_id=env_id,
                    summary_payload=payload,
                    supervised_library_strategy=supervised_library_strategy,
                ),
                bo_variants=(bo_record.get("variants") if isinstance(bo_record, dict) else {}),
            )
            entries.append(
                TaskSummaryEntry(
                    env_id=env_id,
                    source_run_id=source_run_id,
                    task_id=task_id,
                    task_dir=task_dir,
                    summary_path=summary_path,
                    video_path=video_path,
                    overhead_video_path=overhead_video_path,
                    initial_image_path=initial_image_path,
                    summary_payload=merged_payload,
                    updated_at=datetime.fromtimestamp(summary_path.stat().st_mtime, tz=timezone.utc),
                )
            )
    def _entry_rank(item: TaskSummaryEntry) -> tuple[int, float, datetime, str]:
        variants = item.summary_payload.get("variants") if isinstance(item.summary_payload, dict) else {}
        best_correct = 0
        best_score = float("-inf")
        if isinstance(variants, dict):
            for kind in ("supervised", "unsupervised"):
                variant = variants.get(kind)
                if not isinstance(variant, dict):
                    continue
                correct_count = variant.get("bo_correct_count")
                score = variant.get("bo_score")
                if isinstance(correct_count, (int, float)):
                    best_correct = max(best_correct, int(correct_count))
                if isinstance(score, (int, float)):
                    best_score = max(best_score, float(score))
        return (best_correct, best_score, item.updated_at, item.task_id)

    return _select_summary_entries(entries)


def list_completed_task_summaries(project_root: Path) -> dict[str, list[TaskSummaryEntry]]:
    return {
        env_id: _list_completed_task_summaries_for_env(project_root, env_id)
        for env_id in SUPPORTED_VIDEO_RENDER_ENVS
    }


def _ensemble_calibration_run_dirs(project_root: Path, env_id: str, kind: str) -> list[Path]:
    library_root = project_root / "state" / "video_renders" / "canonical_libraries" / env_id / kind
    if not library_root.is_dir():
        return []
    run_dirs: list[Path] = []
    for runs_root in sorted(library_root.glob("ensemble_calibration_runs*")):
        if not runs_root.is_dir():
            continue
        for run_dir in sorted((path for path in runs_root.iterdir() if path.is_dir()), key=lambda p: p.name):
            run_dirs.append(run_dir)
    return run_dirs


def _load_best_bo_summary_task_records(
    project_root: Path,
    env_id: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    records_by_task: dict[tuple[str, str], dict[str, Any]] = {}
    for kind in ("supervised", "unsupervised"):
        ranked_run_dirs: list[tuple[float, Path]] = []
        for run_dir in _ensemble_calibration_run_dirs(project_root, env_id, kind):
            best_path = run_dir / "best.json"
            if not best_path.is_file():
                continue
            try:
                best_payload = json.loads(best_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(best_payload, dict):
                continue
            best_score = best_payload.get("score")
            if not isinstance(best_score, (int, float)):
                continue
            ranked_run_dirs.append((float(best_score), run_dir))
        ranked_run_dirs.sort(key=lambda item: (item[0], item[1].name), reverse=True)
        for _score_value, run_dir in ranked_run_dirs[:1]:
            best_path = run_dir / "best.json"
            best_payload = json.loads(best_path.read_text(encoding="utf-8"))
            trial_root_raw = str(best_payload.get("evaluation_root") or "").strip()
            trial_root = Path(trial_root_raw).resolve() if trial_root_raw else None
            if not isinstance(trial_root, Path) or not trial_root.is_dir():
                continue
            task_batch_path = run_dir / "task_batch.json"
            media_manifest_path = run_dir / "media_manifest.json"
            if not task_batch_path.is_file():
                continue
            try:
                batch_payload = json.loads(task_batch_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            try:
                media_manifest_payload = (
                    json.loads(media_manifest_path.read_text(encoding="utf-8"))
                    if media_manifest_path.is_file()
                    else {}
                )
            except (OSError, json.JSONDecodeError):
                media_manifest_payload = {}
            raw_tasks = batch_payload.get("tasks")
            if not isinstance(raw_tasks, list):
                continue
            batch_lookup: dict[str, dict[str, Any]] = {}
            for item in raw_tasks:
                if not isinstance(item, dict):
                    continue
                batch_id = str(item.get("batch_id") or "").strip()
                source_run_id = str(item.get("source_run_id") or "").strip()
                task_id = str(item.get("task_id") or "").strip()
                if batch_id and source_run_id and task_id:
                    batch_lookup[batch_id] = item
            if not batch_lookup:
                continue
            media_lookup = media_manifest_payload.get("tasks") if isinstance(media_manifest_payload.get("tasks"), dict) else {}
            correct_choice_counts: dict[str, int] = {}
            for question_path in run_dir.glob("trials/trial-*/discrimination/question-*.json"):
                try:
                    question_payload = json.loads(question_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(question_payload, dict):
                    continue
                if question_payload.get("correct") is not True:
                    continue
                batch_id = str(question_payload.get("target_batch_id") or "").strip()
                if not batch_id:
                    continue
                correct_choice_counts[batch_id] = correct_choice_counts.get(batch_id, 0) + 1
            summaries_root = trial_root / "summaries"
            annotations_root = trial_root / "annotations"
            if not summaries_root.is_dir():
                continue
            for summary_path in sorted(summaries_root.glob("*.json")):
                try:
                    summary_item = json.loads(summary_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(summary_item, dict):
                    continue
                batch_id = str(summary_item.get("batch_id") or "").strip()
                summary_text = str(summary_item.get("summary_text") or "").strip()
                annotation_pattern_count = summary_item.get("annotation_pattern_count")
                if (
                    not batch_id
                    or not summary_text
                    or _summary_text_is_malformed(summary_text)
                    or not isinstance(annotation_pattern_count, (int, float))
                    or float(annotation_pattern_count) <= 0
                ):
                    continue
                task_item = batch_lookup.get(batch_id)
                if not isinstance(task_item, dict):
                    continue
                raw_source_run_id = str(task_item.get("source_run_id") or "").strip()
                source_run_id = _canonical_source_run_id(raw_source_run_id)
                task_id = str(task_item.get("task_id") or "").strip()
                task_key = (source_run_id, task_id)
                annotation_path = annotations_root / f"{batch_id}.json"
                annotation_preview = ""
                if annotation_path.is_file():
                    try:
                        annotation_payload = json.loads(annotation_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        annotation_payload = {}
                    if isinstance(annotation_payload, dict):
                        annotation_preview = _format_annotation_text(_normalize_annotation_payload(annotation_payload))
                media_item = media_lookup.get(batch_id) if isinstance(media_lookup, dict) else {}
                media_entry = media_item if isinstance(media_item, dict) else {}
                task_payload = task_item.get("payload") if isinstance(task_item.get("payload"), dict) else {}
                trace_path = _normalize_summary_asset_path(project_root, str(media_entry.get("trace_path") or ""))
                if trace_path is None and isinstance(task_payload, dict):
                    trace_path = _normalize_summary_asset_path(project_root, str(task_payload.get("final_trace_path") or ""))
                media_paths = resolve_trace_media_paths(trace_path)
                if (
                    trace_path is not None
                    and media_paths.default_video_path is None
                    and not raw_source_run_id.startswith("video-render-")
                    and source_run_id
                ):
                    alternate_trace_path = _normalize_summary_asset_path(
                        project_root,
                        str(trace_path).replace(
                            f"/data/evaluations/{env_id}/{raw_source_run_id}/",
                            f"/data/evaluations/{env_id}/video-render-{source_run_id}/",
                        ),
                    )
                    if alternate_trace_path is not None:
                        alternate_media_paths = resolve_trace_media_paths(alternate_trace_path)
                        if alternate_media_paths.default_video_path is not None or alternate_media_paths.overhead_video_path is not None:
                            media_paths = alternate_media_paths
                payload_initial_image_path = (
                    _normalize_summary_asset_path(project_root, str(task_payload.get("initial_image_path") or ""))
                    if isinstance(task_payload, dict)
                    else None
                )
                video_path = (
                    _normalize_summary_asset_path(project_root, str(media_entry.get("video_path") or ""))
                    or media_paths.default_video_path
                    or media_paths.overhead_video_path
                )
                initial_image_path = (
                    _normalize_summary_asset_path(project_root, str(media_entry.get("initial_image_path") or ""))
                    or payload_initial_image_path
                    or media_paths.default_initial_image_path
                    or media_paths.overhead_initial_image_path
                )
                updated_at = datetime.fromtimestamp(summary_path.stat().st_mtime, tz=timezone.utc)
                record = records_by_task.setdefault(
                    task_key,
                    {
                        "source_run_id": source_run_id,
                        "task_id": task_id,
                        "task_dir": trial_root,
                        "summary_path": summary_path,
                        "video_path": video_path,
                        "overhead_video_path": media_paths.overhead_video_path,
                        "initial_image_path": initial_image_path,
                        "updated_at": updated_at,
                        "variants": {},
                    },
                )
                if isinstance(record.get("updated_at"), datetime) and updated_at > record["updated_at"]:
                    record["task_dir"] = trial_root
                    record["summary_path"] = summary_path
                    record["video_path"] = video_path
                    record["overhead_video_path"] = media_paths.overhead_video_path
                    record["initial_image_path"] = initial_image_path
                    record["updated_at"] = updated_at
                variants = record.get("variants")
                if not isinstance(variants, dict):
                    continue
                existing = variants.get(kind)
                existing_score = (
                    float(existing.get("bo_score"))
                    if isinstance(existing, dict) and isinstance(existing.get("bo_score"), (int, float))
                    else float("-inf")
                )
                existing_correct_count = (
                    int(existing.get("bo_correct_count"))
                    if isinstance(existing, dict) and isinstance(existing.get("bo_correct_count"), (int, float))
                    else -1
                )
                correct_count = int(correct_choice_counts.get(batch_id, 0))
                if correct_count < existing_correct_count:
                    continue
                if correct_count == existing_correct_count and float(best_score) < existing_score:
                    continue
                variants[kind] = {
                    "summary_text": summary_text,
                    "annotation_path": str(annotation_path) if annotation_path.is_file() else "",
                    "annotation_preview": annotation_preview,
                    "annotation_pattern_count": int(annotation_pattern_count),
                    "summary_source": "ensemble_bo_trial",
                    "bo_correct_count": correct_count,
                    "bo_score": float(best_score),
                    "bo_run_id": str(run_dir.name),
                    "bo_trial_root": str(trial_root),
                }
    return records_by_task


def _merge_bo_variants_into_summary_payload(
    *,
    summary_payload: dict[str, Any],
    bo_variants: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not bo_variants:
        return summary_payload
    merged = json.loads(json.dumps(summary_payload))
    variants = merged.get("variants")
    if not isinstance(variants, dict):
        variants = {}
        merged["variants"] = variants
    for kind, item in bo_variants.items():
        if kind not in {"supervised", "unsupervised"}:
            continue
        if not isinstance(item, dict):
            continue
        existing = variants.get(kind)
        replacement = dict(existing) if isinstance(existing, dict) else {}
        replacement.update(item)
        variants[kind] = replacement
    enabled = merged.get("enabled_variants")
    if isinstance(enabled, list):
        normalized_enabled = [str(item).strip() for item in enabled if str(item).strip()]
        for key in bo_variants:
            if key not in normalized_enabled:
                normalized_enabled.append(key)
        merged["enabled_variants"] = normalized_enabled
    return merged


def _merge_summary_variants(
    *,
    base_payload: dict[str, Any],
    extra_variants: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not extra_variants:
        return base_payload
    merged = json.loads(json.dumps(base_payload))
    variants = merged.get("variants")
    if not isinstance(variants, dict):
        variants = {}
        merged["variants"] = variants
    for kind, item in extra_variants.items():
        if not isinstance(item, dict):
            continue
        existing = variants.get(kind)
        replacement = dict(existing) if isinstance(existing, dict) else {}
        replacement.update(item)
        variants[kind] = replacement
    enabled = merged.get("enabled_variants")
    normalized_enabled = [str(item).strip() for item in enabled if str(item).strip()] if isinstance(enabled, list) else []
    for key in extra_variants:
        normalized_key = str(key).strip()
        if normalized_key and normalized_key not in normalized_enabled:
            normalized_enabled.append(normalized_key)
    if normalized_enabled:
        merged["enabled_variants"] = normalized_enabled
    return merged


def _load_existing_summary_entries_by_task(
    project_root: Path,
    env_id: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    env_root = project_root / "data" / "evaluations" / env_id
    if not env_root.is_dir():
        return {}
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for summary_path in sorted(env_root.glob("task-summaries-*/tasks/*/*/summary.json")):
        if _summary_path_is_from_backup(summary_path):
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        variants = payload.get("variants")
        if not isinstance(variants, dict):
            continue
        usable_variants = {
            str(kind).strip(): dict(item)
            for kind, item in variants.items()
            if isinstance(item, dict) and str(item.get("summary_text") or "").strip()
        }
        if not usable_variants:
            continue
        task_dir = summary_path.parent
        task_result_payload: dict[str, Any] = {}
        task_result_path = task_dir / "task_result.json"
        if task_result_path.is_file():
            try:
                task_result_payload = json.loads(task_result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                task_result_payload = {}
        source_run_id = _canonical_source_run_id(str(payload.get("source_run_id") or ""))
        authoritative_payload: dict[str, Any] = {}
        if source_run_id:
            authoritative_result_path = (
                project_root
                / "data"
                / "evaluations"
                / env_id
                / f"video-render-{source_run_id}"
                / "tasks"
                / summary_path.parent.parent.name
                / summary_path.parent.name
                / "task_result.json"
            )
            if authoritative_result_path.is_file():
                try:
                    authoritative_payload = json.loads(authoritative_result_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    authoritative_payload = {}
        media_payload = authoritative_payload if authoritative_payload else task_result_payload
        media_paths = TraceMediaPaths(None, None, None, None)
        if isinstance(media_payload, dict):
            trace_path = _normalize_summary_asset_path(
                project_root,
                str(media_payload.get("final_trace_path") or ""),
            )
            media_paths = resolve_trace_media_paths(trace_path)
        video_path = media_paths.default_video_path or next(
            (
                path
                for path in sorted(task_dir.rglob("*"))
                if path.is_file() and path.suffix.lower() in {".mp4", ".webm", ".mov", ".m4v"}
            ),
            None,
        )
        initial_image_path = media_paths.default_initial_image_path or next(
            (
                path
                for path in sorted(task_dir.rglob("*"))
                if path.is_file() and path.name == "initial_scene.png"
            ),
            None,
        )
        task_id = str(payload.get("task_id") or task_dir.name).strip() or task_dir.name
        task_key = (source_run_id, task_id)
        records[task_key] = {
            "summary_payload": payload,
            "variants": usable_variants,
            "video_path": video_path,
            "overhead_video_path": media_paths.overhead_video_path,
            "initial_image_path": initial_image_path,
            "updated_at": datetime.fromtimestamp(summary_path.stat().st_mtime, tz=timezone.utc),
            "task_dir": task_dir,
            "summary_path": summary_path,
        }
    return records


def _annotation_pattern_count(payload: dict[str, Any]) -> int:
    annotations = payload.get("pattern_annotations")
    if not isinstance(annotations, dict):
        return 0
    metadata = annotations.get("metadata")
    if isinstance(metadata, dict):
        total_count = metadata.get("total_pattern_count")
        if isinstance(total_count, int):
            return max(0, total_count)
    patterns = annotations.get("patterns")
    if isinstance(patterns, list):
        return len(patterns)
    return 0


def _variant_pattern_count(variant: dict[str, Any]) -> int:
    pattern_count = variant.get("annotation_pattern_count")
    if isinstance(pattern_count, (int, float)):
        return max(0, int(pattern_count))
    preview = str(variant.get("annotation_preview") or "").strip().lower()
    match = re.search(r"pattern count:\s*(\d+)", preview)
    if match:
        return int(match.group(1))
    return 0


def _normalize_annotation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    inner = payload.get("pattern_annotations")
    if isinstance(inner, dict):
        return payload
    return {"pattern_annotations": payload}


def _filter_pooltool_summary_annotation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_annotation_payload(payload)
    annotations = normalized.get("pattern_annotations")
    if not isinstance(annotations, dict):
        return normalized
    patterns = annotations.get("patterns")
    if not isinstance(patterns, list):
        return normalized
    filtered_patterns: list[Any] = []
    kept_first_cue_strike = False
    removed_count = 0
    for item in patterns:
        if not isinstance(item, dict):
            filtered_patterns.append(item)
            continue
        label_key = str(item.get("label") or "").strip().lower()
        if label_key == "cue strike":
            if kept_first_cue_strike:
                removed_count += 1
                continue
            kept_first_cue_strike = True
        filtered_patterns.append(item)
    if removed_count <= 0:
        return normalized
    updated_annotations = dict(annotations)
    updated_annotations["patterns"] = filtered_patterns
    metadata = dict(updated_annotations.get("metadata") or {})
    metadata["total_pattern_count"] = len(filtered_patterns)
    metadata["pooltool_removed_duplicate_cue_strike_count"] = removed_count
    updated_annotations["metadata"] = metadata
    updated_payload = dict(normalized)
    updated_payload["pattern_annotations"] = updated_annotations
    return updated_payload


def _summary_text_is_malformed(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return True
    lowered = normalized.lower()
    prompt_markers = (
        'be concrete, specific, and grounded in the simulation',
        'mention counts, spatial relations, directions, collisions, and final state details',
        'write 4-6 sentences',
        'return only the final summary',
        '```summary',
        '<your summary>',
    )
    return any(marker in lowered for marker in prompt_markers)


def _variant_has_detected_annotation(variant: dict[str, Any]) -> bool:
    pattern_count = variant.get("annotation_pattern_count")
    if isinstance(pattern_count, (int, float)):
        return float(pattern_count) > 0
    preview = str(variant.get("annotation_preview") or "").strip().lower()
    if not preview:
        return False
    if "pattern count: 0" in preview:
        return False
    if "no patterns detected" in preview:
        return False
    if "no pattern annotations available" in preview:
        return False
    return True


def _current_library_variant_metadata(
    project_root: Path,
    *,
    env_id: str,
    kind: str,
    strategy: str,
) -> dict[str, str]:
    library_path = _resolve_latest_library(
        project_root,
        env_id,
        kind,
        strategy=strategy if kind == "supervised" else "latest",
    )
    calibration_path = library_path.parent / "ensemble_calibration.json"
    metadata = {
        "library_path": str(library_path.resolve()),
        "calibration_path": str(calibration_path.resolve()) if calibration_path.is_file() else "",
        "library_source": "current_optimized_library",
    }
    return metadata


def _variant_uses_current_library(
    project_root: Path,
    *,
    env_id: str,
    kind: str,
    strategy: str,
    variant: dict[str, Any],
) -> bool:
    if str(variant.get("summary_source") or "").strip() == "ensemble_bo_best":
        return True
    expected = _current_library_variant_metadata(
        project_root,
        env_id=env_id,
        kind=kind,
        strategy=strategy,
    )
    library_path = str(variant.get("library_path") or "").strip()
    calibration_path = str(variant.get("calibration_path") or "").strip()
    if library_path != expected["library_path"]:
        return False
    if expected["calibration_path"]:
        return calibration_path == expected["calibration_path"]
    return not calibration_path


def _filter_stale_annotation_variants(
    *,
    project_root: Path,
    env_id: str,
    summary_payload: dict[str, Any],
    supervised_library_strategy: str,
) -> dict[str, Any]:
    merged = json.loads(json.dumps(summary_payload))
    variants = merged.get("variants")
    if not isinstance(variants, dict):
        return merged
    for kind in ("supervised", "unsupervised"):
        variant = variants.get(kind)
        if not isinstance(variant, dict):
            continue
        if not _variant_has_detected_annotation(variant):
            variants.pop(kind, None)
            continue
        if _variant_uses_current_library(
            project_root,
            env_id=env_id,
            kind=kind,
            strategy=supervised_library_strategy,
            variant=variant,
        ):
            continue
        variants.pop(kind, None)
    enabled_variants = merged.get("enabled_variants")
    if isinstance(enabled_variants, list):
        merged["enabled_variants"] = [
            item
            for item in enabled_variants
            if str(item).strip() in set(str(key).strip() for key in variants.keys())
        ]
    return merged


def _entry_has_shared_annotation_coverage(
    entry: TaskSummaryEntry,
    *,
    minimum_pattern_count: int = SUMMARY_MIN_SHARED_PATTERN_COUNT,
) -> bool:
    variants = entry.summary_payload.get("variants") if isinstance(entry.summary_payload, dict) else {}
    if not isinstance(variants, dict):
        return False
    supervised_variant = variants.get("supervised")
    unsupervised_variant = variants.get("unsupervised")
    if not isinstance(supervised_variant, dict) or not isinstance(unsupervised_variant, dict):
        return False
    return (
        _variant_pattern_count(supervised_variant) >= minimum_pattern_count
        and _variant_pattern_count(unsupervised_variant) >= minimum_pattern_count
    )


def _entry_has_any_annotation_coverage(
    entry: TaskSummaryEntry,
    *,
    minimum_pattern_count: int = 1,
) -> bool:
    variants = entry.summary_payload.get("variants") if isinstance(entry.summary_payload, dict) else {}
    if not isinstance(variants, dict):
        return False
    for kind in ("supervised", "unsupervised"):
        variant = variants.get(kind)
        if isinstance(variant, dict) and _variant_pattern_count(variant) >= minimum_pattern_count:
            return True
    return False


def _entry_rank(item: TaskSummaryEntry) -> tuple[int, int, float, datetime, str]:
    variants = item.summary_payload.get("variants") if isinstance(item.summary_payload, dict) else {}
    best_correct = 0
    best_score = float("-inf")
    total_patterns = 0
    if isinstance(variants, dict):
        for kind in ("supervised", "unsupervised"):
            variant = variants.get(kind)
            if not isinstance(variant, dict):
                continue
            total_patterns += _variant_pattern_count(variant)
            correct_count = variant.get("bo_correct_count")
            score = variant.get("bo_score")
            if isinstance(correct_count, (int, float)):
                best_correct = max(best_correct, int(correct_count))
            if isinstance(score, (int, float)):
                best_score = max(best_score, float(score))
    return (best_correct, total_patterns, best_score, item.updated_at, item.task_id)


def _select_summary_entries(entries: list[TaskSummaryEntry]) -> list[TaskSummaryEntry]:
    ranked_entries = sorted(entries, key=_entry_rank, reverse=True)
    featured: list[TaskSummaryEntry] = [
        item for item in ranked_entries if _entry_has_shared_annotation_coverage(item)
    ]
    fallback: list[TaskSummaryEntry] = [
        item
        for item in ranked_entries
        if item not in featured and _entry_has_any_annotation_coverage(item, minimum_pattern_count=SUMMARY_MIN_SHARED_PATTERN_COUNT)
    ]
    if len(featured) >= BO_SUMMARY_ENTRY_LIMIT:
        return featured[:BO_SUMMARY_ENTRY_LIMIT]
    combined = featured + fallback
    return combined[:BO_SUMMARY_ENTRY_LIMIT]


def summarize_task_summary_inventory(project_root: Path) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for env_id in SUPPORTED_VIDEO_RENDER_ENVS:
        env_root = project_root / "data" / "evaluations" / env_id
        successful = 0
        if env_root.is_dir():
            for source_run_dir in sorted(env_root.glob("*image-only*")):
                if not source_run_dir.is_dir() or ".bak." in source_run_dir.name:
                    continue
                for source_result_path in sorted(source_run_dir.glob("tasks/*/*/task_result.json")):
                    try:
                        payload = json.loads(source_result_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    if isinstance(payload, dict) and payload.get("success") is True:
                        successful += 1
        summarized = len(_list_completed_task_summaries_for_env(project_root, env_id))
        summary[env_id] = {
            "successful_tasks": successful,
            "summarized_tasks": summarized,
            "pending_tasks": max(0, successful - summarized),
        }
    return summary


def list_task_summary_statuses(project_root: Path) -> dict[str, dict[str, Any]]:
    state = _load_state(project_root)
    inventory = summarize_task_summary_inventory(project_root)
    statuses: dict[str, dict[str, Any]] = {}
    changed = False
    for env_id in SUPPORTED_VIDEO_RENDER_ENVS:
        current = dict(state.get(env_id, {}))
        if current.get("status") == "running" and not _pid_is_alive(current.get("pid")):
            current["status"] = "failed"
            current["finished_at"] = current.get("finished_at") or _isoformat()
            current["error"] = str(current.get("error") or "Summary worker exited unexpectedly.").strip()
            current["updated_at"] = _isoformat()
            state[env_id] = current
            changed = True
        merged = {
            "env_id": env_id,
            "status": str(current.get("status") or "idle"),
            "pid": current.get("pid"),
            "run_label": str(current.get("run_label") or "").strip(),
            "enabled_variants_override": list(current.get("enabled_variants_override") or []),
            "target_count": _normalize_target_count(current.get("target_count") or DEFAULT_TASK_SUMMARY_TARGET),
            "started_at": current.get("started_at"),
            "updated_at": current.get("updated_at"),
            "finished_at": current.get("finished_at"),
            "current_task_id": str(current.get("current_task_id") or "").strip(),
            "current_source_run_id": str(current.get("current_source_run_id") or "").strip(),
            "planned_tasks": int(current.get("planned_tasks") or 0),
            "completed_tasks": int(current.get("completed_tasks") or 0),
            "failed_tasks": int(current.get("failed_tasks") or 0),
            "log_relpath": str(current.get("log_relpath") or "").strip(),
            "error": str(current.get("error") or "").strip(),
            **inventory.get(env_id, {"successful_tasks": 0, "summarized_tasks": 0, "pending_tasks": 0}),
        }
        merged["processed_tasks"] = merged["completed_tasks"] + merged["failed_tasks"]
        statuses[env_id] = merged
    if changed:
        _write_state(project_root, state)
    return statuses


def start_task_summary_process(
    project_root: Path,
    env_id: str,
    *,
    target_count: int = DEFAULT_TASK_SUMMARY_TARGET,
    enabled_variants_override: list[str] | None = None,
) -> dict[str, Any]:
    env_key = str(env_id or "").strip().lower()
    if env_key not in SUPPORTED_VIDEO_RENDER_ENVS:
        raise ValueError(f"Unknown environment {env_key}")
    normalized_target = _normalize_target_count(target_count or DEFAULT_TASK_SUMMARY_TARGET)
    statuses = list_task_summary_statuses(project_root)
    current = statuses.get(env_key, {})
    if current.get("status") == "running" and _pid_is_alive(current.get("pid")):
        return current
    successful_tasks = int(current.get("successful_tasks") or 0)
    summarized_tasks = int(current.get("summarized_tasks") or 0)
    desired_total = min(normalized_target, successful_tasks) if successful_tasks > 0 else normalized_target
    outstanding_needed = max(0, desired_total - summarized_tasks)
    planned_tasks = min(int(current.get("pending_tasks") or 0), outstanding_needed)
    if planned_tasks <= 0:
        return _merge_env_state(
            project_root,
            env_key,
            {
                "status": "succeeded",
                "pid": None,
                "target_count": normalized_target,
                "planned_tasks": 0,
                "completed_tasks": 0,
                "failed_tasks": 0,
                "current_task_id": "",
                "current_source_run_id": "",
                "run_label": f"task-summary-{env_key}-noop",
                "started_at": _isoformat(),
                "finished_at": _isoformat(),
                "error": "",
                "log_relpath": _project_relpath(project_root, _log_path(project_root, env_key)),
            },
        )
    log_path = _log_path(project_root, env_key)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env_python = Path(python_executable_for_env(env_key))
    client_root = _client_project_root()
    server_src = _server_src_root()
    existing_pythonpath = str(os.environ.get("PYTHONPATH", "")).strip()
    pythonpath_entries = [str(server_src), str(client_root / "src")]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    process_env = os.environ.copy()
    process_env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    process_env.setdefault("TASK_SUMMARY_LLM_BASE_URL", DEFAULT_TASK_SUMMARY_LLM_BASE_URL)
    normalized_variants = [
        str(item).strip()
        for item in (enabled_variants_override or [])
        if str(item).strip() in DEFAULT_ENABLED_SUMMARY_VARIANTS
    ]
    command = [
        str(env_python),
        "-m",
        "pattern_server.task_summary_worker",
        "--project-root",
        str(project_root),
        "--env",
        env_key,
        "--target-count",
        str(normalized_target),
    ]
    for item in normalized_variants:
        command.extend(["--enabled-variant", item])
    with log_path.open("ab") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(client_root),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=process_env,
        )
    started_at = _isoformat()
    return _merge_env_state(
        project_root,
        env_key,
        {
            "status": "running",
            "pid": process.pid,
            "target_count": normalized_target,
            "planned_tasks": planned_tasks,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "current_task_id": "",
            "current_source_run_id": "",
            "run_label": (
                f"task-summary-{env_key}-{'-'.join(normalized_variants)}-{_utc_now().strftime('%Y%m%d-%H%M%S')}"
                if normalized_variants
                else f"task-summary-{env_key}-{_utc_now().strftime('%Y%m%d-%H%M%S')}"
            ),
            "started_at": started_at,
            "finished_at": None,
            "error": "",
            "log_relpath": _project_relpath(project_root, log_path),
            "enabled_variants_override": normalized_variants,
        },
    )


def _ensure_client_src_on_path() -> None:
    client_src = _client_project_root() / "src"
    client_src_str = str(client_src)
    if client_src_str not in sys.path:
        sys.path.insert(0, client_src_str)


def _load_shared_llm_defaults() -> tuple[str, str, int]:
    _ensure_client_src_on_path()
    from llm_config import load_shared_llm_config  # type: ignore

    config = load_shared_llm_config()
    base_url = str(os.environ.get("TASK_SUMMARY_LLM_BASE_URL") or config.base_url or "").strip()
    if not base_url:
        base_url = DEFAULT_TASK_SUMMARY_LLM_BASE_URL
    model = str(os.environ.get("TASK_SUMMARY_LLM_MODEL") or config.model or "").strip()
    timeout_value = str(os.environ.get("TASK_SUMMARY_LLM_TIMEOUT_SECONDS") or "").strip()
    if timeout_value:
        try:
            timeout = int(timeout_value)
        except ValueError:
            timeout = int(config.timeout or 120)
    else:
        timeout = int(config.timeout or 120)
    return base_url, model, timeout


def _load_trace(trace_path: Path):
    _ensure_client_src_on_path()
    from deepphy_eval.io import load_trace  # type: ignore

    return load_trace(trace_path)


def _library_requires_supervised_seed_evidence(library: Any) -> bool:
    ensembles = [ensemble for ensemble in getattr(library, "label_ensembles", []) if getattr(ensemble, "members", None)]
    if ensembles:
        input_modes = {str(getattr(ensemble, "input_mode", "") or "").strip() for ensemble in ensembles}
        return "supervised_evidence" in input_modes and "trace" not in input_modes
    detectors = [detector for detector in getattr(library, "detectors", []) if getattr(detector, "detector_id", None)]
    if detectors:
        input_modes = {str(getattr(detector, "input_mode", "") or "").strip() for detector in detectors}
        return "supervised_evidence" in input_modes and "trace" not in input_modes
    return False


def _apply_library_to_trace(project_root: Path, trace: Any, library_path: Path) -> dict[str, Any]:
    _ensure_client_src_on_path()
    from pattern_library_detectors import apply_pattern_library  # type: ignore
    from pattern_library_evidence import build_pattern_evidence  # type: ignore
    from pattern_library_registry import load_pattern_library  # type: ignore

    library = load_pattern_library(library_path)
    existing_evidence = None
    upstream_supervised_library_path = ""
    if _library_requires_supervised_seed_evidence(library):
        env_id = str(getattr(trace, "environment", "") or "")
        if env_id:
            supervised_library_path = resolve_canonical_library(project_root, env_id, "supervised")
            supervised_library = load_pattern_library(supervised_library_path)
            supervised_patterns = apply_pattern_library(trace, supervised_library)
            existing_evidence = build_pattern_evidence(
                supervised_library,
                patterns=supervised_patterns,
                metadata={
                    "source": "task-summaries-supervised-seed",
                    "trace_id": getattr(trace, "trace_id", ""),
                    "task_id": getattr(trace, "task_id", ""),
                    "supervised_library_path": str(supervised_library_path),
                },
                pattern_limit=len(supervised_patterns),
            )
            upstream_supervised_library_path = str(supervised_library_path)
    patterns = apply_pattern_library(trace, library, existing_evidence=existing_evidence)
    return build_pattern_evidence(
        library,
        patterns=patterns,
        metadata={
            "source": "task-summaries",
            "trace_id": getattr(trace, "trace_id", ""),
            "task_id": getattr(trace, "task_id", ""),
            "supervised_library_path": str(library_path),
            "seed_supervised_library_path": upstream_supervised_library_path,
        },
        pattern_limit=SUMMARY_ANNOTATION_PATTERN_LIMIT,
    )


def _request_summary_text(
    *,
    project_root: Path,
    prompt_template: str,
    env_id: str,
    task_id: str,
    annotation_kind: str | None,
    annotation_text: str | None,
    image_paths: list[Path],
) -> str:
    _ensure_client_src_on_path()
    from pattern_learning.patterns.scoring import request_chat_completion_with_text_retry  # type: ignore

    base_url, model, timeout = _load_shared_llm_defaults()
    headers = {"Content-Type": "application/json"}
    user_lines = [
        prompt_template.strip(),
        "",
        f"Environment: {env_id}",
        f"Task: {task_id}",
        f"Summary mode: {annotation_kind or 'image_only'}",
    ]
    if annotation_text:
        image_paths = image_paths[:1]
        user_lines.extend(
            [
                "",
                (
                    "Context: the provided image is the initial frame only. "
                    "The annotation text below is a human-readable learned pattern annotation for the whole simulation, "
                    "including temporal events beyond the initial frame."
                ),
                (
                    "Use the initial frame to ground the starting state, and use the annotation text "
                    "to understand the full dynamics and outcome. Preserve the concrete event details from "
                    "the annotation instead of replacing them with generic language."
                ),
                "",
                "Learned pattern annotation summary:",
                annotation_text,
            ]
        )
    user_lines.extend(
        [
            "",
            "Return only the final summary wrapped in a fenced code block exactly like ` ```summary ... ``` `:",
            "```summary",
            "<your summary>",
            "```",
        ]
    )
    user_content: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(user_lines).strip()}]
    for image_path in image_paths:
        mime = "image/png"
        if image_path.suffix.lower() in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}"},
            }
        )
    with _acquire_llm_request_slot(project_root):
        _body, text = request_chat_completion_with_text_retry(
            base_url=base_url,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You summarize physics simulations from image frames and optional structured annotations. "
                        "Return only the final concise summary. Do not include reasoning, scratch work, or analysis steps. "
                        "Always wrap the summary in exactly one fenced code block: ` ```summary ... ``` `."
                    ),
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            timeout=timeout,
            headers=headers,
            error_prefix="Task summary response",
            max_tokens=8192,
            temperature=0.0,
        )
    explicit_summary = _extract_explicit_summary_from_completion(text, _body)
    if explicit_summary:
        return explicit_summary
    reasoning_trace = _extract_reasoning_trace_from_completion(text, _body)
    if reasoning_trace:
        extracted = _request_summary_from_reasoning_trace(
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
    return _extract_summary_text_from_completion(text, _body)


def _request_summary_from_reasoning_trace(
    *,
    project_root: Path,
    base_url: str,
    model: str,
    timeout: int,
    env_id: str,
    task_id: str,
    reasoning_trace: str,
) -> str:
    _ensure_client_src_on_path()
    from pattern_learning.patterns.scoring import request_chat_completion_with_text_retry  # type: ignore

    headers = {"Content-Type": "application/json"}
    reasoning_excerpt = reasoning_trace.strip()
    if len(reasoning_excerpt) > 24000:
        reasoning_excerpt = reasoning_excerpt[:24000].rstrip()
    with _acquire_llm_request_slot(project_root):
        body, text = request_chat_completion_with_text_retry(
            base_url=base_url,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract a final concise simulation summary from another model's reasoning trace. "
                        "Ignore scratch work, repeated deliberation, and uncertainty. "
                        "Return only the grounded final summary in exactly one fenced code block: "
                        "```summary\\n...\\n```."
                    ),
                },
                {
                    "role": "user",
                    "content": "\n".join(
                        [
                            f"Environment: {env_id}",
                            f"Task: {task_id}",
                            "",
                            "The previous model failed to emit a final summary block.",
                            "Read its reasoning trace and extract the best final summary of the simulation.",
                            "Keep the summary concrete and concise, focused on the initial setup, key interactions, and final outcome.",
                            "",
                            "Reasoning trace:",
                            reasoning_excerpt,
                            "",
                            "Return only:",
                            "```summary",
                            "<your summary>",
                            "```",
                        ]
                    ).strip(),
                },
            ],
            timeout=timeout,
            headers=headers,
            error_prefix="Task summary reasoning extraction",
            max_tokens=SUMMARY_REASONING_EXTRACTION_MAX_TOKENS,
            temperature=0.0,
        )
    return _extract_summary_text_from_completion(text, body)


def _extract_summary_text_from_completion(primary_text: str, completion_body: dict[str, Any]) -> str:
    explicit = _extract_explicit_summary_from_completion(primary_text, completion_body)
    if explicit:
        return explicit
    candidates: list[str] = []
    seen: set[str] = set()

    def _push(candidate: str) -> None:
        cleaned = _cleanup_summary_text(candidate)
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(cleaned)

    _push(primary_text)
    for candidate in _completion_text_candidates(completion_body):
        _push(candidate)

    for candidate in candidates:
        if candidate:
            return candidate
    return ""


def _extract_explicit_summary_from_completion(primary_text: str, completion_body: dict[str, Any]) -> str:
    candidates: list[str] = []
    seen: set[str] = set()

    def _push(candidate: str) -> None:
        cleaned = _cleanup_summary_text(candidate)
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(cleaned)

    _push(primary_text)
    for candidate in _completion_text_candidates(completion_body):
        _push(candidate)

    for candidate in candidates:
        fenced = _extract_fenced_summary(candidate)
        if fenced:
            return fenced
    for candidate in candidates:
        fenced = _extract_any_fenced_block(candidate)
        if fenced:
            return fenced
    return ""


def _completion_text_candidates(completion_body: dict[str, Any]) -> list[str]:
    if not isinstance(completion_body, dict):
        return []
    choices = completion_body.get("choices")
    if not isinstance(choices, list):
        return []
    fields = (
        "content",
        "output_text",
        "text",
        "reasoning_content",
        "reasoning",
        "output",
    )
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


def _extract_reasoning_trace_from_completion(primary_text: str, completion_body: dict[str, Any]) -> str:
    reasoning_candidates: list[str] = []
    seen: set[str] = set()

    def _push(candidate: str) -> None:
        cleaned = _cleanup_summary_text(candidate)
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        reasoning_candidates.append(cleaned)

    if not _extract_fenced_summary(primary_text) and "```" not in primary_text:
        _push(primary_text)
    if isinstance(completion_body, dict):
        choices = completion_body.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                for payload in (choice.get("message"), choice, choice.get("delta")):
                    if not isinstance(payload, dict):
                        continue
                    for field in ("reasoning_content", "reasoning"):
                        flattened = _flatten_completion_text(payload.get(field))
                        if flattened:
                            _push(flattened)
    return "\n\n".join(reasoning_candidates).strip()


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


def _extract_fenced_summary(text: str) -> str:
    for match in re.finditer(r"```([^\n`]*)\n(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        label = str(match.group(1) or "").strip().lower()
        body = _cleanup_summary_text(str(match.group(2) or ""))
        if not body:
            continue
        if label in {"summary", "summaries"}:
            return body
    return ""


def _extract_any_fenced_block(text: str) -> str:
    for match in re.finditer(r"```([^\n`]*)\n(.*?)```", text, flags=re.DOTALL):
        body = _cleanup_summary_text(str(match.group(2) or ""))
        if body:
            return body
    return ""


def _cleanup_summary_text(text: str) -> str:
    cleaned = unescape(str(text or "")).strip()
    if not cleaned:
        return ""
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith("'") and cleaned.endswith("'") and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    if cleaned.lower().startswith("summary:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    return cleaned


def _resolve_ffmpeg_executable() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _sample_frame_indices(total_frames: int, target_frames: int) -> list[int]:
    if total_frames <= 0 or target_frames <= 0:
        return []
    if total_frames <= target_frames:
        return list(range(total_frames))
    last_index = total_frames - 1
    return sorted(
        {
            int(round(position * last_index / max(target_frames - 1, 1)))
            for position in range(target_frames)
        }
    )


def _motion_window_indices(frames: list[Any]) -> tuple[int, int] | None:
    if len(frames) < 3:
        return None
    try:
        import numpy as np  # type: ignore
    except Exception:
        return None
    base = np.asarray(frames[0]).astype(np.float32).mean(axis=2)
    scores: list[float] = []
    for frame in frames:
        gray = np.asarray(frame).astype(np.float32).mean(axis=2)
        scores.append(float(np.abs(gray - base).mean()))
    peak = max(scores[1:], default=0.0)
    if peak <= 0.0:
        return None
    threshold = max(peak * 0.25, 0.01)
    active = [index for index, score in enumerate(scores) if score >= threshold]
    if not active:
        return None
    return min(active), max(active)


def _extract_video_frames(
    video_path: Path,
    output_dir: Path,
    *,
    frame_rate: int = SUMMARY_FRAME_RATE,
    frame_count: int = SUMMARY_FRAME_COUNT,
) -> list[Path]:
    if not video_path.is_file():
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("frame_*.png"):
        path.unlink(missing_ok=True)
    for path in output_dir.glob("frame_*.jpg"):
        path.unlink(missing_ok=True)
    try:
        import imageio.v3 as iio  # type: ignore

        metadata = iio.immeta(video_path)
        duration_seconds = metadata.get("duration")
        requested_frames = frame_count
        if isinstance(duration_seconds, (int, float)) and float(duration_seconds) > 0:
            requested_frames = min(frame_count, max(1, int(round(float(duration_seconds) * max(1, int(frame_rate))))))
        frames = list(iio.imiter(video_path))
        motion_window = _motion_window_indices(frames)
        if motion_window is not None:
            motion_start, motion_end = motion_window
            active_indices = _sample_frame_indices(max(1, motion_end - motion_start + 1), max(1, requested_frames - 1))
            indices = [0] + [motion_start + index for index in active_indices]
            indices = sorted({index for index in indices if 0 <= index < len(frames)})
        else:
            indices = _sample_frame_indices(len(frames), requested_frames)
        written: list[Path] = []
        for output_index, frame_index in enumerate(indices, start=1):
            image = Image.fromarray(frames[frame_index]).convert("RGB")
            if image.width > SUMMARY_IMAGE_MAX_WIDTH:
                target_height = max(1, int(round(image.height * SUMMARY_IMAGE_MAX_WIDTH / image.width)))
                image = image.resize((SUMMARY_IMAGE_MAX_WIDTH, target_height), Image.Resampling.LANCZOS)
            output_path = output_dir / f"frame_{output_index:03d}.jpg"
            image.save(output_path, format="JPEG", quality=SUMMARY_IMAGE_JPEG_QUALITY, optimize=True)
            written.append(output_path)
        if written:
            return written[:frame_count]
    except Exception:
        pass
    ffmpeg = _resolve_ffmpeg_executable()
    if ffmpeg is None:
        return []
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={max(1, int(frame_rate))},scale=min(iw\\,{SUMMARY_IMAGE_MAX_WIDTH}):-2",
        "-q:v",
        str(SUMMARY_IMAGE_JPEG_QUALITY),
        "-frames:v",
        str(frame_count),
        str(output_dir / "frame_%03d.jpg"),
    ]
    subprocess.run(command, check=False)
    return sorted(output_dir.glob("frame_*.jpg"))[:frame_count]


def _extract_initial_video_frame(video_path: Path, output_path: Path) -> Path | None:
    if not video_path.is_file():
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio  # type: ignore

        for frame in iio.imiter(video_path):
            image = Image.fromarray(frame).convert("RGB")
            if image.width > SUMMARY_IMAGE_MAX_WIDTH:
                target_height = max(1, int(round(image.height * SUMMARY_IMAGE_MAX_WIDTH / image.width)))
                image = image.resize((SUMMARY_IMAGE_MAX_WIDTH, target_height), Image.Resampling.LANCZOS)
            image.save(output_path, format="JPEG", quality=SUMMARY_IMAGE_JPEG_QUALITY, optimize=True)
            return output_path if output_path.is_file() else None
    except Exception:
        pass
    ffmpeg = _resolve_ffmpeg_executable()
    if ffmpeg is None:
        return None
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,0),scale=min(iw\\,{SUMMARY_IMAGE_MAX_WIDTH}):-2",
        "-frames:v",
        "1",
        str(output_path),
    ]
    subprocess.run(command, check=False)
    return output_path if output_path.is_file() else None


def _prepare_summary_initial_image(
    *,
    env_id: str,
    output_dir: Path,
    default_initial_path: Path | None,
    overhead_initial_path: Path | None,
    overhead_video_path: Path | None,
) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if env_id == "pooltool":
        overhead_output = output_dir / "summary_initial_overhead.jpg"
        if isinstance(overhead_initial_path, Path) and overhead_initial_path.is_file():
            source_image_path = overhead_initial_path
        else:
            source_image_path = None
        if source_image_path is None and isinstance(overhead_video_path, Path) and overhead_video_path.is_file():
            extracted = _extract_initial_video_frame(overhead_video_path, overhead_output)
            if extracted is not None and extracted.is_file():
                return extracted
        if source_image_path is not None:
            return _downsample_summary_initial_image(source_image_path, overhead_output)

    if not isinstance(default_initial_path, Path) or not default_initial_path.is_file():
        return None
    return _downsample_summary_initial_image(default_initial_path, output_dir / "summary_initial.jpg")


def _downsample_summary_initial_image(source_path: Path, output_path: Path) -> Path | None:
    if not source_path.is_file():
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        image = Image.open(source_path).convert("RGB")
        if image.width > SUMMARY_IMAGE_MAX_WIDTH:
            target_height = max(1, int(round(image.height * SUMMARY_IMAGE_MAX_WIDTH / image.width)))
            image = image.resize((SUMMARY_IMAGE_MAX_WIDTH, target_height), Image.Resampling.LANCZOS)
        image.save(output_path, format="JPEG", quality=SUMMARY_IMAGE_JPEG_QUALITY, optimize=True)
        return output_path if output_path.is_file() else None
    except Exception:
        return None


def _format_annotation_text(payload: dict[str, Any]) -> str:
    def _strip_consensus_trace_tokens(text: str) -> str:
        return text.replace("consensus-trace-", "")

    annotations = payload.get("pattern_annotations")
    if not isinstance(annotations, dict):
        return "No pattern annotations available."
    _ensure_client_src_on_path()
    from pattern_library_schema import PatternEvidence  # type: ignore
    from pattern_library_serializer import serialize_pattern_evidence  # type: ignore

    try:
        evidence = PatternEvidence.model_validate(annotations)
    except Exception:
        patterns = annotations.get("patterns")
        if not isinstance(patterns, list) or not patterns:
            return "No patterns detected."
        lines: list[str] = []
        for item in patterns[:SUMMARY_ANNOTATION_PATTERN_LIMIT]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "pattern").strip() or "pattern"
            start = item.get("start_timestep")
            end = item.get("end_timestep")
            parameters = item.get("parameters")
            detail = ""
            if isinstance(parameters, dict) and parameters:
                preview_items: list[str] = []
                for key in sorted(parameters.keys())[:4]:
                    preview_items.append(f"{key}={parameters[key]}")
                if preview_items:
                    detail = f" ({', '.join(preview_items)})"
            lines.append(f"- [{start}-{end}] {label}{detail}".strip())
        return _strip_consensus_trace_tokens("\n".join(lines).strip() or "No patterns detected.")

    bundle = serialize_pattern_evidence(evidence)
    metadata = dict(evidence.metadata or {})
    total_count = metadata.get("total_pattern_count")
    pattern_limit = metadata.get("pattern_limit")
    truncated = bool(metadata.get("patterns_truncated"))
    header_lines = [f"Library: {bundle.library_id}"]
    if isinstance(total_count, int):
        if truncated and isinstance(pattern_limit, int):
            header_lines.append(f"Patterns shown: {bundle.pattern_count} of {total_count} (limit {pattern_limit})")
        else:
            header_lines.append(f"Pattern count: {total_count}")
    elif bundle.pattern_count:
        header_lines.append(f"Pattern count: {bundle.pattern_count}")
    body = _strip_consensus_trace_tokens(bundle.text.strip() or "No patterns detected.")
    return _strip_consensus_trace_tokens("\n".join([*header_lines, "", body]).strip())


def _variant_text_path(task_dir: Path, key: str) -> Path:
    return task_dir / f"summary_{key}.txt"


def _variant_annotation_path(task_dir: Path, key: str) -> Path:
    return task_dir / f"annotation_{key}.json"


def _normalize_summary_asset_path(project_root: Path, raw_path: str | None) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.exists():
        return candidate.resolve()
    prefixes = (
        ("/opt/pattern-learning-stack/data/evaluations", project_root / "data" / "evaluations"),
        ("/var/opt/pattern-learning-stack/data/evaluations", project_root / "data" / "evaluations"),
        ("/opt/pattern-learning-stack/data", project_root / "data"),
        ("/var/opt/pattern-learning-stack/data", project_root / "data"),
        ("/home/sean/Services/Work/pattern-learning-server/data/evaluations", project_root / "data" / "evaluations"),
        ("/var/home/sean/Services/Work/pattern-learning-server/data/evaluations", project_root / "data" / "evaluations"),
        ("/home/sean/Services/Work/pattern-learning-server/data", project_root / "data"),
        ("/var/home/sean/Services/Work/pattern-learning-server/data", project_root / "data"),
    )
    for prefix, replacement in prefixes:
        if text == prefix or text.startswith(prefix + "/"):
            suffix = text[len(prefix) :].lstrip("/")
            remapped = replacement / suffix
            if remapped.exists():
                return remapped.resolve()
    return None


def _find_first_existing_file(paths: list[Path | None]) -> Path | None:
    for path in paths:
        if isinstance(path, Path) and path.is_file():
            return path.resolve()
    return None


def _authoritative_video_render_task_dir(project_root: Path, task: TaskSummaryTask) -> Path:
    return (
        project_root
        / "data"
        / "evaluations"
        / task.env_id
        / f"video-render-{task.source_run_id}"
        / "tasks"
        / task.case_id
        / task.source_result_path.parent.name
    )


def _task_has_authoritative_summary_media(
    project_root: Path,
    env_id: str,
    source_run_id: str,
    case_id: str,
    task_dir_name: str,
) -> bool:
    task_dir = (
        project_root
        / "data"
        / "evaluations"
        / env_id
        / f"video-render-{source_run_id}"
        / "tasks"
        / case_id
        / task_dir_name
    )
    if not task_dir.is_dir():
        return False
    if any(path.is_file() and path.suffix.lower() in {".mp4", ".webm", ".mov", ".m4v"} for path in task_dir.rglob("*")):
        return True
    task_result_path = task_dir / "task_result.json"
    if not task_result_path.is_file():
        return False
    try:
        payload = json.loads(task_result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    trace_path = _normalize_summary_asset_path(project_root, str(payload.get("final_trace_path") or ""))
    if trace_path is not None and trace_path.is_file():
        return True
    initial_path = _normalize_summary_asset_path(project_root, str(payload.get("initial_image_path") or ""))
    return bool(initial_path is not None and initial_path.is_file())


def _load_authoritative_video_render_payload(project_root: Path, task: TaskSummaryTask) -> tuple[Path | None, dict[str, Any]]:
    task_dir = _authoritative_video_render_task_dir(project_root, task)
    task_result_path = task_dir / "task_result.json"
    if not task_result_path.is_file():
        return None, {}
    try:
        payload = json.loads(task_result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return task_result_path, {}
    return task_result_path, payload if isinstance(payload, dict) else {}


def _supervised_library_rank(root: Path) -> tuple[float, float, float, str]:
    run_json = root / "run.json"
    payload: dict[str, Any] = {}
    if run_json.is_file():
        try:
            payload = json.loads(run_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    validation = payload.get("validation_score")
    average = payload.get("average_score")
    completed = payload.get("completed_label_count")
    total = payload.get("primitive_label_count")
    validation_value = float(validation) if isinstance(validation, (int, float)) else float("-inf")
    average_value = float(average) if isinstance(average, (int, float)) else float("-inf")
    completion_ratio = (
        float(completed) / float(total)
        if isinstance(completed, (int, float)) and isinstance(total, (int, float)) and float(total) > 0
        else 0.0
    )
    return (validation_value, average_value, completion_ratio, root.name)


def _resolve_latest_library(project_root: Path, env_id: str, workflow_kind: str, *, strategy: str = "latest") -> Path:
    try:
        return resolve_canonical_library(project_root, env_id, workflow_kind)
    except Exception:
        pass
    candidates = [
        item
        for item in list_environment_libraries(project_root, env_id)
        if str(item.workflow_kind or "").strip().lower() == workflow_kind
        and ".bak." not in item.run_id
        and ".broken-" not in item.run_id
    ]
    if not candidates:
        raise FileNotFoundError(f"No {workflow_kind} library found for {env_id}")
    if workflow_kind == "supervised" and strategy == "best":
        candidates = sorted(candidates, key=lambda item: _supervised_library_rank(item.root), reverse=True)
    return candidates[0].library_path


def _render_summary_media(project_root: Path, task: TaskSummaryTask) -> tuple[Path, Path | None, Path, list[Path]]:
    task.output_task_dir.mkdir(parents=True, exist_ok=True)
    rendered_payload = {}
    task_result_path = task.output_task_dir / "task_result.json"
    if task_result_path.is_file():
        try:
            rendered_payload = json.loads(task_result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            rendered_payload = {}
    source_payload = task.payload if isinstance(task.payload, dict) else {}
    authoritative_result_path, authoritative_payload = _load_authoritative_video_render_payload(project_root, task)
    trace_path = _find_first_existing_file(
        [
            _normalize_summary_asset_path(project_root, str(authoritative_payload.get("final_trace_path") or "")),
            _normalize_summary_asset_path(project_root, str(rendered_payload.get("final_trace_path") or "")) if rendered_payload else None,
            _normalize_summary_asset_path(project_root, str(source_payload.get("final_trace_path") or "")),
        ]
    )
    resolved_media = resolve_trace_media_paths(trace_path)
    preferred_video_path = (
        resolved_media.overhead_video_path
        if task.env_id == "pooltool" and isinstance(resolved_media.overhead_video_path, Path)
        else resolved_media.default_video_path
    )
    video_path = preferred_video_path or _find_first_existing_file(
        [
            _normalize_summary_asset_path(project_root, str(authoritative_payload.get("simulation_video_overhead_path") or ""))
            if task.env_id == "pooltool"
            else None,
            _normalize_summary_asset_path(project_root, str(authoritative_payload.get("simulation_video_path") or "")),
            _normalize_summary_asset_path(project_root, str(rendered_payload.get("simulation_video_path") or "")) if rendered_payload else None,
        ]
    )
    if video_path is None:
        video_path = next(
            (
                path
                for path in sorted(task.output_task_dir.rglob("*"))
                if path.is_file()
                and path.suffix.lower() in {".mp4", ".webm", ".mov", ".m4v"}
                and (
                    "overhead" in path.name.lower()
                    if task.env_id == "pooltool"
                    else "overhead" not in path.name.lower()
                )
            ),
            None,
        )
    if video_path is None and authoritative_result_path is not None:
        video_task_dir = authoritative_result_path.parent
        video_path = next(
            (
                path
                for path in sorted(video_task_dir.rglob("*"))
                if path.is_file()
                and path.suffix.lower() in {".mp4", ".webm", ".mov", ".m4v"}
                and (
                    "overhead" in path.name.lower()
                    if task.env_id == "pooltool"
                    else "overhead" not in path.name.lower()
                )
            ),
            None,
        )
    if video_path is None:
        video_path = next(
            (
                path
                for path in sorted(task.source_result_path.parent.rglob("*"))
                if path.is_file() and path.suffix.lower() in {".mp4", ".webm", ".mov", ".m4v"}
            ),
            None,
        )
    overhead_initial_path = resolved_media.overhead_initial_image_path
    initial_path = resolved_media.default_initial_image_path or next(
        (
            path
            for path in sorted(task.output_task_dir.rglob("*"))
            if path.is_file() and path.name == "initial_scene.png"
        ),
        None,
    )
    if initial_path is None:
        initial_path = _find_first_existing_file(
            [
                _normalize_summary_asset_path(project_root, str(authoritative_payload.get("initial_image_path") or "")),
                _normalize_summary_asset_path(project_root, str(source_payload.get("initial_image_path") or "")),
                next((path for path in sorted(task.source_result_path.parent.rglob("*.png")) if path.is_file()), None),
            ]
        )
    if overhead_initial_path is None:
        overhead_initial_path = _find_first_existing_file(
            [
                _normalize_summary_asset_path(project_root, str(authoritative_payload.get("initial_image_overhead_path") or "")),
                _normalize_summary_asset_path(project_root, str(rendered_payload.get("initial_image_overhead_path") or "")) if rendered_payload else None,
                _normalize_summary_asset_path(project_root, str(source_payload.get("initial_image_overhead_path") or "")),
            ]
        )
    if not (trace_path and trace_path.is_file() and initial_path):
        video_task = VideoRenderTask(
            env_id=task.env_id,
            source_run_id=task.source_run_id,
            source_run_dir=task.source_run_dir,
            source_result_path=task.source_result_path,
            case_id=task.case_id,
            task_id=task.task_id,
            payload=task.payload,
            output_run_dir=task.output_run_dir,
            output_task_dir=task.output_task_dir,
            output_result_path=task.output_task_dir / "task_result.json",
        )
        prior_cwd = Path.cwd()
        try:
            os.chdir(_client_project_root())
            replay_task_video(project_root, video_task)
        finally:
            os.chdir(prior_cwd)
        rendered_payload = json.loads((task.output_task_dir / "task_result.json").read_text(encoding="utf-8"))
        trace_path = _find_first_existing_file(
            [
                _normalize_summary_asset_path(project_root, str(rendered_payload.get("final_trace_path") or "")),
                trace_path,
            ]
        )
        resolved_media = resolve_trace_media_paths(trace_path)
        preferred_video_path = (
            resolved_media.overhead_video_path
            if task.env_id == "pooltool" and isinstance(resolved_media.overhead_video_path, Path)
            else resolved_media.default_video_path
        )
        video_path = preferred_video_path or next(
            (
                path
                for path in sorted(task.output_task_dir.rglob("*"))
                if path.is_file()
                and path.suffix.lower() in {".mp4", ".webm", ".mov", ".m4v"}
                and (
                    "overhead" in path.name.lower()
                    if task.env_id == "pooltool"
                    else "overhead" not in path.name.lower()
                )
            ),
            None,
        )
        initial_path = resolved_media.default_initial_image_path or next(
            (
                path
                for path in sorted(task.output_task_dir.rglob("*"))
                if path.is_file() and path.name == "initial_scene.png"
            ),
            None,
        )
        if initial_path is None:
            initial_path = _find_first_existing_file(
                [
                    _normalize_summary_asset_path(project_root, str(rendered_payload.get("initial_image_path") or "")),
                    _normalize_summary_asset_path(project_root, str(source_payload.get("initial_image_path") or "")),
                    initial_path,
                ]
            )
        if overhead_initial_path is None:
            overhead_initial_path = resolved_media.overhead_initial_image_path or _find_first_existing_file(
                [
                    _normalize_summary_asset_path(project_root, str(rendered_payload.get("initial_image_overhead_path") or "")),
                    _normalize_summary_asset_path(project_root, str(source_payload.get("initial_image_overhead_path") or "")),
                ]
            )
    if trace_path is None or not trace_path.is_file():
        raise FileNotFoundError(f"Missing rendered trace for {task.task_id}")
    if initial_path is None or not initial_path.is_file():
        raise FileNotFoundError(f"Missing initial summary image for {task.task_id}")
    prepared_initial_path = _prepare_summary_initial_image(
        env_id=task.env_id,
        output_dir=task.output_task_dir,
        default_initial_path=initial_path,
        overhead_initial_path=overhead_initial_path,
        overhead_video_path=resolved_media.overhead_video_path,
    )
    if prepared_initial_path is None or not prepared_initial_path.is_file():
        raise FileNotFoundError(f"Missing prepared initial summary image for {task.task_id}")
    # Prefer a short low-res frame sequence from the rendered simulation.
    image_paths: list[Path] = []
    if isinstance(video_path, Path) and video_path.is_file():
        image_paths.extend(_extract_video_frames(video_path, task.output_task_dir / "summary_frames"))
    if not image_paths:
        raise FileNotFoundError(f"No summary video frames available for {task.task_id}")
    return trace_path, video_path, prepared_initial_path, image_paths[:SUMMARY_FRAME_COUNT]


def _write_summary_bundle(
    task: TaskSummaryTask,
    *,
    image_only_prompt_template: str,
    annotation_prompt_template: str,
    variants: dict[str, dict[str, Any]],
    image_paths: list[Path],
    video_path: Path | None,
) -> None:
    payload = {
        "schema_version": "task_summaries.v1",
        "env_id": task.env_id,
        "task_id": task.task_id,
        "source_run_id": task.source_run_id,
        "source_result_path": str(task.source_result_path),
        "prompt_templates": {
            "image_only": image_only_prompt_template,
            "annotation": annotation_prompt_template,
        },
        "enabled_variants": sorted(variants.keys()),
        "generated_at": _isoformat(),
        "image_paths": [str(path) for path in image_paths],
        "video_path": str(video_path) if video_path is not None else None,
        "variants": variants,
    }
    if task.summary_path.is_file():
        try:
            existing_payload = json.loads(task.summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_payload = {}
        if isinstance(existing_payload, dict):
            payload = _merge_summary_variants(
                base_payload=existing_payload,
                extra_variants=variants,
            )
            prompt_templates = payload.get("prompt_templates")
            if not isinstance(prompt_templates, dict):
                prompt_templates = {}
                payload["prompt_templates"] = prompt_templates
            prompt_templates["image_only"] = image_only_prompt_template
            prompt_templates["annotation"] = annotation_prompt_template
            payload["schema_version"] = "task_summaries.v1"
            payload["env_id"] = task.env_id
            payload["task_id"] = task.task_id
            payload["source_run_id"] = task.source_run_id
            payload["source_result_path"] = str(task.source_result_path)
            payload["generated_at"] = _isoformat()
            payload["image_paths"] = [str(path) for path in image_paths]
            payload["video_path"] = str(video_path) if video_path is not None else None
    task.summary_path.parent.mkdir(parents=True, exist_ok=True)
    task.summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _generate_task_summary(
    project_root: Path,
    task: TaskSummaryTask,
    *,
    image_only_prompt_template: str,
    annotation_prompt_template: str,
    enabled_variants: list[str],
    supervised_library_strategy: str,
) -> None:
    trace_path, video_path, initial_image_path, image_paths = _render_summary_media(project_root, task)
    trace = _load_trace(trace_path)

    summary_requests: list[tuple[str, str | None, str | None, Path | None, dict[str, Any], str, list[Path]]] = []
    requested_annotation_variants = [
        kind for kind in ("supervised", "unsupervised", "automatic") if kind in enabled_variants
    ]
    nonempty_annotation_variant_count = 0
    if "image_only" in enabled_variants:
        summary_requests.append(("image_only", None, None, None, {}, image_only_prompt_template, image_paths))
    for kind in ("supervised", "unsupervised", "automatic"):
        if kind not in enabled_variants:
            continue
        library_path = _resolve_latest_library(
            project_root,
            task.env_id,
            kind,
            strategy=supervised_library_strategy if kind == "supervised" else "latest",
        )
        annotation_payload = _apply_library_to_trace(project_root, trace, library_path)
        if task.env_id == "pooltool" and kind in {"supervised", "unsupervised"}:
            annotation_payload = _filter_pooltool_summary_annotation_payload(annotation_payload)
        pattern_count = _annotation_pattern_count(annotation_payload)
        if pattern_count <= 0:
            continue
        nonempty_annotation_variant_count += 1
        annotation_path = _variant_annotation_path(task.output_task_dir, kind)
        annotation_path.write_text(json.dumps(annotation_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        annotation_text = _format_annotation_text(annotation_payload)
        summary_requests.append(
            (
                kind,
                kind,
                annotation_text,
                annotation_path,
                {
                    "annotation_preview": annotation_text,
                    "library_path": str(library_path),
                    "calibration_path": str((library_path.parent / "ensemble_calibration.json").resolve())
                    if (library_path.parent / "ensemble_calibration.json").is_file()
                    else "",
                    "library_source": "current_optimized_library",
                    "annotation_pattern_count": pattern_count,
                },
                annotation_prompt_template,
                [initial_image_path],
            )
        )

    if requested_annotation_variants and nonempty_annotation_variant_count <= 0:
        _write_summary_failure(
            task,
            error="Skipped because no annotation-based variant produced any detected patterns.",
            status="skipped-no-annotation",
        )
        return

    if not summary_requests:
        _write_summary_failure(
            task,
            error="Skipped because there were no valid summary variants to request.",
            status="skipped-no-variants",
        )
        return

    variants: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=_task_summary_request_parallelism(len(summary_requests))) as executor:
        future_to_request = {
            executor.submit(
                _request_summary_text,
                project_root=project_root,
                prompt_template=variant_prompt_template,
                env_id=task.env_id,
                task_id=task.task_id,
                annotation_kind=annotation_kind,
                annotation_text=annotation_text,
                image_paths=variant_image_paths,
            ): (variant_key, annotation_path, variant_metadata)
            for variant_key, annotation_kind, annotation_text, annotation_path, variant_metadata, variant_prompt_template, variant_image_paths in summary_requests
        }
        for future in as_completed(future_to_request):
            variant_key, annotation_path, variant_metadata = future_to_request[future]
            summary_text = future.result()
            _variant_text_path(task.output_task_dir, variant_key).write_text(summary_text + "\n", encoding="utf-8")
            variants[variant_key] = {
                "summary_text": summary_text,
                **variant_metadata,
            }
            if annotation_path is not None:
                variants[variant_key]["annotation_path"] = str(annotation_path)
    _write_summary_bundle(
        task,
        image_only_prompt_template=image_only_prompt_template,
        annotation_prompt_template=annotation_prompt_template,
        variants=variants,
        image_paths=image_paths,
        video_path=video_path,
    )


def run_task_summary_batch(
    project_root: Path,
    env_id: str,
    *,
    target_count: int = DEFAULT_TASK_SUMMARY_TARGET,
    enabled_variants_override: list[str] | None = None,
) -> int:
    env_key = str(env_id or "").strip().lower()
    if env_key not in SUPPORTED_VIDEO_RENDER_ENVS:
        raise ValueError(f"Unknown environment {env_key}")
    config = load_task_summary_config(project_root)
    image_only_prompt_template = str(
        config.get("image_only_prompt_template") or DEFAULT_IMAGE_ONLY_SUMMARY_PROMPT
    ).strip()
    annotation_prompt_template = str(
        config.get("annotation_prompt_template") or DEFAULT_ANNOTATION_SUMMARY_PROMPT
    ).strip()
    enabled_variants = [
        str(item).strip()
        for item in (config.get("enabled_variants") or list(DEFAULT_ENABLED_SUMMARY_VARIANTS))
        if str(item).strip() in DEFAULT_ENABLED_SUMMARY_VARIANTS
    ]
    override_variants = [
        str(item).strip()
        for item in (enabled_variants_override or [])
        if str(item).strip() in DEFAULT_ENABLED_SUMMARY_VARIANTS
    ]
    if override_variants:
        enabled_variants = override_variants
    supervised_library_strategy = str(config.get("supervised_library_strategy") or "latest").strip().lower()
    normalized_target = _normalize_target_count(target_count or DEFAULT_TASK_SUMMARY_TARGET)
    inventory = summarize_task_summary_inventory(project_root).get(
        env_key,
        {"successful_tasks": 0, "summarized_tasks": 0, "pending_tasks": 0},
    )
    successful_tasks = int(inventory.get("successful_tasks") or 0)
    summarized_tasks = int(inventory.get("summarized_tasks") or 0)
    desired_total = min(normalized_target, successful_tasks) if successful_tasks > 0 else normalized_target
    outstanding_needed = max(0, desired_total - summarized_tasks)
    tasks = discover_task_summary_tasks(project_root, env_key)[:outstanding_needed]
    _merge_env_state(
        project_root,
        env_key,
        {
            "status": "running",
            "pid": os.getpid(),
            "target_count": normalized_target,
            "planned_tasks": len(tasks),
            "completed_tasks": 0,
            "failed_tasks": 0,
            "current_task_id": "",
            "current_source_run_id": "",
            "started_at": _isoformat(),
            "finished_at": None,
            "error": "",
            "log_relpath": _project_relpath(project_root, _log_path(project_root, env_key)),
            "enabled_variants_override": list(enabled_variants),
        },
    )
    failures: list[str] = []
    completed = 0
    failed = 0
    for task in tasks:
        _merge_env_state(
            project_root,
            env_key,
            {
                "current_task_id": task.task_id,
                "current_source_run_id": task.source_run_id,
                "completed_tasks": completed,
                "failed_tasks": failed,
            },
        )
        try:
            _generate_task_summary(
                project_root,
                task,
                image_only_prompt_template=image_only_prompt_template,
                annotation_prompt_template=annotation_prompt_template,
                enabled_variants=enabled_variants,
                supervised_library_strategy=supervised_library_strategy,
            )
            completed += 1
        except Exception as exc:
            failed += 1
            if isinstance(exc, FileNotFoundError) and "No summary video frames available" in str(exc):
                _write_summary_failure(task, error=str(exc), status="fatal")
            failures.append(f"{task.source_run_id}:{task.task_id}: {exc}")
        _merge_env_state(
            project_root,
            env_key,
            {
                "completed_tasks": completed,
                "failed_tasks": failed,
            },
        )
    status = "succeeded" if failed == 0 else "failed"
    _merge_env_state(
        project_root,
        env_key,
        {
            "status": status,
            "pid": None,
            "finished_at": _isoformat(),
            "current_task_id": "",
            "current_source_run_id": "",
            "error": "\n".join(failures[:6]),
        },
    )
    return completed


__all__ = [
    "DEFAULT_ANNOTATION_SUMMARY_PROMPT",
    "DEFAULT_IMAGE_ONLY_SUMMARY_PROMPT",
    "DEFAULT_TASK_SUMMARY_TARGET",
    "TaskSummaryEntry",
    "discover_task_summary_tasks",
    "list_completed_task_summaries",
    "list_task_summary_statuses",
    "load_task_summary_config",
    "run_task_summary_batch",
    "save_task_summary_config",
    "start_task_summary_process",
    "summarize_task_summary_inventory",
]
