from __future__ import annotations

import colorsys
import copy
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from pattern_server.env_catalog import list_environment_libraries
from ensemble_library_training.library_collections import (
    load_cached_canonical_library_summary as _load_cached_canonical_library_summary,
    load_global_best_supervised_library_summary as _load_global_best_supervised_library_summary,
    resolve_canonical_library,
)
from pattern_server.legacy_jobs import MONOREPO_ROOT, python_executable_for_env


SUPPORTED_VIDEO_RENDER_ENVS = ("phyre", "iphyre", "kinetix", "pooltool")
VIDEO_RENDER_STATE_RELPATH = Path("state/video_renders/state.json")
VIDEO_RENDER_RESULTS_RELPATH = Path("state/video_renders/results.json")
VIDEO_ANNOTATION_STATE_RELPATH = Path("state/video_renders/annotation_state.json")
VIDEO_ANNOTATION_CONFIG_RELPATH = Path("state/video_renders/annotation_config.json")
DEFAULT_VIDEO_RENDER_TARGET = 50
MIN_ANNOTATION_DISPLAY_SECONDS = 1.0
IPHYRE_COORD_SPACE_SIZE = 600.0
VIDEO_ANNOTATION_VERSION = 4
SERVER_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LABEL_DIR_PATTERN = re.compile(r"^label_(?P<index>\d{3})_(?P<label>.+)$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_BEST_LIBRARY_CACHE: dict[str, tuple[str, Path]] = {}
KINETIX_VIDEO_RENDER_FPS = 16
KINETIX_VIDEO_RENDER_MAX_SIDE = 1024
POOLTOOL_TABLE_WIDTH = 0.9906
POOLTOOL_TABLE_LENGTH = 1.9812
POOLTOOL_OVERHEAD_VIEWPORT_LEFT = 0.0488
POOLTOOL_OVERHEAD_VIEWPORT_TOP = 0.1208
POOLTOOL_OVERHEAD_VIEWPORT_RIGHT = 0.9408
POOLTOOL_OVERHEAD_VIEWPORT_BOTTOM = 0.8781


@dataclass(frozen=True)
class VideoRenderTask:
    env_id: str
    source_run_id: str
    source_run_dir: Path
    source_result_path: Path
    case_id: str
    task_id: str
    payload: dict[str, Any]
    output_run_dir: Path
    output_task_dir: Path
    output_result_path: Path


@dataclass(frozen=True)
class TraceMediaPaths:
    default_video_path: Path | None
    overhead_video_path: Path | None
    default_initial_image_path: Path | None
    overhead_initial_image_path: Path | None


@dataclass(frozen=True)
class VideoRenderResultEntry:
    env_id: str
    task_id: str
    source_run_id: str
    output_run_id: str
    outcome_kind: str
    task_result_path: Path
    video_path: Path
    overhead_video_path: Path | None
    annotated_video_path: Path | None
    annotated_overhead_video_path: Path | None
    annotation_path: Path | None
    annotation_preview: str
    supervised_library_path: Path | None
    updated_at: datetime


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: datetime | None = None) -> str:
    return (value or _utc_now()).isoformat()


def _state_path(project_root: Path) -> Path:
    return project_root / VIDEO_RENDER_STATE_RELPATH


def _state_root(project_root: Path) -> Path:
    return _state_path(project_root).parent


def _results_path(project_root: Path) -> Path:
    return project_root / VIDEO_RENDER_RESULTS_RELPATH


def _log_path(project_root: Path, env_id: str) -> Path:
    return _state_root(project_root) / f"{env_id}.log"


def _annotation_state_path(project_root: Path) -> Path:
    return project_root / VIDEO_ANNOTATION_STATE_RELPATH


def _annotation_config_path(project_root: Path) -> Path:
    return project_root / VIDEO_ANNOTATION_CONFIG_RELPATH


def _annotation_log_path(project_root: Path, env_id: str) -> Path:
    return _state_root(project_root) / f"annotation_{env_id}.log"


def _overlay_export_root(project_root: Path) -> Path:
    return _state_root(project_root) / "overlay_exports"


def _project_relpath(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _normalize_video_asset_path(project_root: Path, raw_path: str | None) -> Path | None:
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


def _client_project_root() -> Path:
    return SERVER_PROJECT_ROOT


def _server_src_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_client_src_on_path() -> None:
    client_src = _client_project_root() / "src"
    client_src_str = str(client_src)
    if client_src.is_dir() and client_src_str not in os.sys.path:
        os.sys.path.insert(0, client_src_str)


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


def _load_annotation_state(project_root: Path) -> dict[str, dict[str, Any]]:
    path = _annotation_state_path(project_root)
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


def _write_annotation_state(project_root: Path, state: dict[str, dict[str, Any]]) -> None:
    path = _annotation_state_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _load_annotation_config(project_root: Path) -> dict[str, dict[str, Any]]:
    path = _annotation_config_path(project_root)
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


def _write_annotation_config(project_root: Path, config: dict[str, dict[str, Any]]) -> None:
    path = _annotation_config_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _load_results(project_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    path = _results_path(project_root)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    parsed: dict[str, dict[str, dict[str, Any]]] = {}
    for env_id, env_payload in payload.items():
        if not isinstance(env_id, str) or not isinstance(env_payload, dict):
            continue
        parsed[env_id] = {
            str(key): dict(value)
            for key, value in env_payload.items()
            if isinstance(key, str) and isinstance(value, dict)
        }
    return parsed


def _write_results(project_root: Path, results: dict[str, dict[str, dict[str, Any]]]) -> None:
    path = _results_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _pid_is_alive(pid: Any) -> bool:
    try:
        pid_value = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_value <= 0:
        return False
    try:
        os.kill(pid_value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _merge_env_state(project_root: Path, env_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    path = _state_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        try:
            payload = json.load(handle)
        except json.JSONDecodeError:
            payload = {}
        state = payload if isinstance(payload, dict) else {}
        current = dict(state.get(env_id, {})) if isinstance(state.get(env_id, {}), dict) else {}
        current.update(updates)
        current["env_id"] = env_id
        current["updated_at"] = _isoformat()
        state[env_id] = current
        handle.seek(0)
        handle.truncate()
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return current


def _merge_annotation_env_state(project_root: Path, env_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    path = _annotation_state_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        try:
            payload = json.load(handle)
        except json.JSONDecodeError:
            payload = {}
        state = payload if isinstance(payload, dict) else {}
        current = dict(state.get(env_id, {})) if isinstance(state.get(env_id, {}), dict) else {}
        current.update(updates)
        current["env_id"] = env_id
        current["updated_at"] = _isoformat()
        state[env_id] = current
        handle.seek(0)
        handle.truncate()
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return current


def _task_has_video(task_dir: Path) -> bool:
    for candidate in task_dir.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in {".mp4", ".webm", ".mov", ".m4v"}:
            return True
    return False


def discover_video_render_tasks(
    project_root: Path,
    env_id: str,
    *,
    target_count: int = 20,
) -> list[VideoRenderTask]:
    env_root = project_root / "data" / "evaluations" / env_id
    if not env_root.is_dir():
        return []
    candidates: list[tuple[float, str, VideoRenderTask]] = []
    for source_run_dir in sorted(env_root.glob("*image-only*"), reverse=True):
        if not source_run_dir.is_dir() or ".bak." in source_run_dir.name:
            continue
        for source_result_path in sorted(source_run_dir.glob("tasks/*/*/task_result.json"), reverse=True):
            try:
                payload = json.loads(source_result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            outcome_kind = _result_outcome_kind(payload)
            if outcome_kind is None:
                continue
            case_id = source_result_path.parent.parent.name
            task_dir_name = source_result_path.parent.name
            task_id = str(payload.get("task_id") or task_dir_name).strip() or task_dir_name
            output_run_dir = env_root / f"video-render-{source_run_dir.name}"
            output_task_dir = output_run_dir / "tasks" / case_id / task_dir_name
            if _task_has_video(output_task_dir):
                continue
            candidates.append(
                (
                    source_result_path.stat().st_mtime,
                    outcome_kind,
                    VideoRenderTask(
                        env_id=env_id,
                        source_run_id=source_run_dir.name,
                        source_run_dir=source_run_dir,
                        source_result_path=source_result_path,
                        case_id=case_id,
                        task_id=task_id,
                        payload=payload,
                        output_run_dir=output_run_dir,
                        output_task_dir=output_task_dir,
                        output_result_path=output_task_dir / "task_result.json",
                    ),
                )
            )
    deduped: dict[tuple[str, str], tuple[float, VideoRenderTask]] = {}
    for updated_at, outcome_kind, task in candidates:
        key = (outcome_kind, task.task_id)
        current = deduped.get(key)
        if current is None or updated_at > current[0]:
            deduped[key] = (updated_at, task)

    success_target = min(15, max(0, target_count))
    failure_target = min(5, max(0, target_count - success_target))
    successes = [item for (outcome_kind, _task_id), item in deduped.items() if outcome_kind == "success"]
    failures = [item for (outcome_kind, _task_id), item in deduped.items() if outcome_kind == "failure"]
    successes.sort(key=lambda item: item[0], reverse=True)
    failures.sort(key=lambda item: item[0], reverse=True)

    selected = [task for _updated_at, task in failures[:failure_target]]
    selected.extend(task for _updated_at, task in successes[:success_target])
    if len(selected) < target_count:
        seen = {(task.payload.get("success") is True, task.task_id) for task in selected}
        remainder = successes[success_target:] + failures[failure_target:]
        for _updated_at, task in remainder:
            key = (task.payload.get("success") is True, task.task_id)
            if key in seen:
                continue
            selected.append(task)
            seen.add(key)
            if len(selected) >= target_count:
                break
    return selected[:max(1, target_count)]


def _build_video_render_task(project_root: Path, env_id: str, source_result_path: Path) -> VideoRenderTask:
    payload = _load_json_object(source_result_path)
    if not payload:
        raise FileNotFoundError(f"Missing or invalid task result: {source_result_path}")
    source_run_dir = source_result_path.parents[3]
    case_id = source_result_path.parent.parent.name
    task_dir_name = source_result_path.parent.name
    task_id = str(payload.get("task_id") or task_dir_name).strip() or task_dir_name
    env_root = project_root / "data" / "evaluations" / env_id
    output_run_dir = env_root / f"video-render-{source_run_dir.name}"
    output_task_dir = output_run_dir / "tasks" / case_id / task_dir_name
    return VideoRenderTask(
        env_id=env_id,
        source_run_id=source_run_dir.name,
        source_run_dir=source_run_dir,
        source_result_path=source_result_path,
        case_id=case_id,
        task_id=task_id,
        payload=payload,
        output_run_dir=output_run_dir,
        output_task_dir=output_task_dir,
        output_result_path=output_task_dir / "task_result.json",
    )


def replay_task_video_from_source_result(project_root: Path, env_id: str, source_result_path: Path) -> None:
    replay_task_video(project_root, _build_video_render_task(project_root, env_id, source_result_path))


def _replay_task_video_subprocess(project_root: Path, task: VideoRenderTask) -> None:
    code = (
        "from pathlib import Path; import sys; "
        "from pattern_server.video_renders import replay_task_video_from_source_result; "
        "replay_task_video_from_source_result(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(project_root), task.env_id, str(task.source_result_path)],
        cwd=str(_client_project_root()),
        env=os.environ.copy(),
        check=True,
    )


def _normalize_target_count(raw_value: Any) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_VIDEO_RENDER_TARGET
    return max(1, value)


def _video_annotation_path(task_dir: Path) -> Path:
    return task_dir / "annotation_supervised.json"


def _annotation_video_filter_descriptions_for_env(env_id: str) -> list[str]:
    env_key = str(env_id or "").strip().lower()
    shared_filters = ["Hide pattern end markers, including single-timestep terminal events, in the last 10% of the simulation."]
    if env_key == "phyre":
        return shared_filters + ["Remove bounce pattern activations unless the annotation includes a BALL object."]
    if env_key == "pooltool":
        return shared_filters + [
            "Remove cue strike begin markers that start after the first 5% of the simulation.",
            "Drop placeholder PoolTool annotation points at the world origin before overlay projection.",
        ]
    if env_key in {"iphyre", "kinetix"}:
        return shared_filters
    return []


def annotation_video_filter_descriptions(env_id: str | None = None) -> dict[str, list[str]] | list[str]:
    if env_id is not None:
        return _annotation_video_filter_descriptions_for_env(env_id)
    return {
        env_key: _annotation_video_filter_descriptions_for_env(env_key)
        for env_key in SUPPORTED_VIDEO_RENDER_ENVS
        if _annotation_video_filter_descriptions_for_env(env_key)
    }


def _annotation_shape_tree_contains_ball(value: Any) -> bool:
    if isinstance(value, dict):
        shape = str(value.get("shape") or "").strip().upper()
        if shape == "BALL":
            return True
        return any(_annotation_shape_tree_contains_ball(item) for item in value.values())
    if isinstance(value, list):
        return any(_annotation_shape_tree_contains_ball(item) for item in value)
    return False


def _pattern_includes_ball_object(item: dict[str, Any]) -> bool:
    parameters = item.get("parameters")
    if not isinstance(parameters, dict):
        return False
    object_details = parameters.get("object_details")
    if _annotation_shape_tree_contains_ball(object_details):
        return True
    return _annotation_shape_tree_contains_ball(parameters.get("object_positions"))


def _pooltool_pocket_position_from_parameters(parameters: dict[str, Any]) -> tuple[float, float] | None:
    pocket_position = _xy_point(parameters.get("pocket_position"))
    if pocket_position is not None:
        return pocket_position
    pocket_id = str(parameters.get("pocket_id") or "").strip().lower()
    if not pocket_id:
        object_details = parameters.get("object_details")
        if isinstance(object_details, dict):
            for detail in object_details.values():
                if not isinstance(detail, dict):
                    continue
                pocket_position = _xy_point(detail.get("pocket_position"))
                if pocket_position is not None:
                    return pocket_position
                nested_pocket_id = str(detail.get("pocket_id") or "").strip().lower()
                if nested_pocket_id:
                    pocket_id = nested_pocket_id
                    break
    if pocket_id not in {"lb", "lc", "lt", "rb", "rc", "rt"}:
        return None
    x_value = 0.0 if pocket_id.startswith("l") else POOLTOOL_TABLE_WIDTH
    if pocket_id.endswith("b"):
        y_value = 0.0
    elif pocket_id.endswith("c"):
        y_value = POOLTOOL_TABLE_LENGTH * 0.5
    else:
        y_value = POOLTOOL_TABLE_LENGTH
    return (float(x_value), float(y_value))


def _pooltool_ball_pocketed_pattern_is_plausible(item: dict[str, Any]) -> bool:
    parameters = item.get("parameters")
    if not isinstance(parameters, dict):
        return False
    return _pooltool_pocket_position_from_parameters(parameters) is not None


def _pooltool_gentle_touch_pattern_is_plausible(item: dict[str, Any]) -> bool:
    parameters = item.get("parameters")
    if not isinstance(parameters, dict):
        return False
    object_details = parameters.get("object_details")
    if not isinstance(object_details, dict) or len(object_details) != 2:
        return False
    try:
        start_timestep = int(item.get("start_timestep") or 0)
    except (TypeError, ValueError):
        start_timestep = 0
    # Suppress the startup grid of stale activations clustered around cue strike.
    if start_timestep <= 2:
        return False
    # Older over-firing detectors wrote these permissive fields; the stricter detector does not.
    if "interaction_kind" in parameters or "relative_velocity_mps" in parameters:
        return False
    try:
        relative_speed = float(parameters.get("relative_speed"))
    except (TypeError, ValueError):
        return False
    if relative_speed > 0.22:
        return False
    valid_roles = {"gently_touching_ball", "gently_touched_ball"}
    for detail in object_details.values():
        if not isinstance(detail, dict):
            return False
        role = str(detail.get("role") or "").strip()
        if role not in valid_roles:
            return False
    return True


def _annotation_pattern_passes_custom_filters(*, env_id: str, item: dict[str, Any]) -> bool:
    env_key = str(env_id or "").strip().lower()
    label_key = _normalized_label_key(item.get("label"))
    if env_key == "phyre" and label_key == "bounce":
        return _pattern_includes_ball_object(item)
    if env_key == "pooltool" and label_key == "ball pocketed":
        return _pooltool_ball_pocketed_pattern_is_plausible(item)
    if env_key == "pooltool" and label_key == "gentle touch":
        return _pooltool_gentle_touch_pattern_is_plausible(item)
    return True


def filter_annotation_patterns_for_env(env_id: str, patterns: list[Any]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for item in patterns:
        if not isinstance(item, dict):
            continue
        if not _annotation_pattern_passes_custom_filters(env_id=env_id, item=item):
            continue
        filtered.append(item)
    return filtered


def _annotation_event_passes_custom_filters(
    *,
    env_id: str,
    item: dict[str, Any],
    event_kind: str,
    start_timestep: int,
    end_timestep: int,
    total_timesteps: int,
) -> bool:
    env_key = str(env_id or "").strip().lower()
    label_key = _normalized_label_key(item.get("label"))
    event_kind_key = str(event_kind).strip().lower()
    if total_timesteps <= 1:
        return not (event_kind_key == "end" or (event_kind_key == "start" and end_timestep <= start_timestep))
    event_progress = float(max(start_timestep, end_timestep)) / float(max(1, total_timesteps - 1))
    if event_kind_key == "start" and end_timestep <= start_timestep and event_progress >= 0.90:
        return False
    if event_kind_key == "end":
        if event_progress >= 0.90:
            return False
    if env_key == "pooltool" and label_key == "cue strike" and event_kind_key == "start":
        if total_timesteps <= 1:
            return start_timestep <= 0
        progress = float(start_timestep) / float(max(1, total_timesteps - 1))
        return progress <= 0.05
    return True


def _pattern_event_specs(
    *,
    start_timestep: int,
    end_timestep: int,
    start_frame: int,
    end_frame: int,
) -> list[tuple[str, int]]:
    if end_timestep <= start_timestep:
        return [("start", start_frame)]
    event_specs = [("start", start_frame)]
    if end_frame != start_frame:
        event_specs.append(("end", end_frame))
    return event_specs


def _format_annotation_preview(payload: dict[str, Any], *, env_id: str | None = None) -> str:
    annotations = payload.get("pattern_annotations")
    if not isinstance(annotations, dict):
        return "No pattern annotations available."
    patterns = annotations.get("patterns")
    if not isinstance(patterns, list) or not patterns:
        return "No patterns detected."
    filtered_patterns = filter_annotation_patterns_for_env(env_id or "", patterns)
    if not filtered_patterns:
        return "No patterns detected."
    lines: list[str] = []
    for item in filtered_patterns[:12]:
        label = str(item.get("label") or "pattern").strip()
        start = item.get("start_timestep")
        end = item.get("end_timestep")
        parameters = item.get("parameters")
        detail = ""
        if isinstance(parameters, dict):
            object_details = parameters.get("object_details")
            if isinstance(object_details, dict) and object_details:
                detail = f" | objects={_format_pattern_object_details(object_details)}"
            object_indices = parameters.get("object_indices")
            if not detail and isinstance(object_indices, list) and object_indices:
                detail = f" objects={object_indices[:4]}"
        lines.append(f"- {label} [{start}-{end}]{detail}".strip())
    text = "\n".join(lines).strip() or "No patterns detected."
    if len(text) > 1200:
        text = text[:1197].rstrip() + "..."
    return text


def _object_detail_sort_key(item: Any) -> tuple[int, int | str]:
    text = str(item)
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def _format_pattern_object_details(object_details: dict[str, Any]) -> str:
    parts: list[str] = []
    for object_key, detail in sorted(object_details.items(), key=lambda item: _object_detail_sort_key(item[0])):
        if not isinstance(detail, dict):
            continue
        preview_parts: list[str] = []
        roles = str(detail.get("roles") or detail.get("role") or "").strip()
        if roles:
            preview_parts.append(f"roles={roles}")
        color = detail.get("color")
        preview_parts.append(f"color={color if color is not None else 'null'}")
        shape = detail.get("shape")
        preview_parts.append(f"shape={shape if shape is not None else 'null'}")
        start_position = detail.get("start_position")
        end_position = detail.get("end_position")
        preview_parts.append(f"start_position={_format_preview_point(start_position)}")
        preview_parts.append(f"end_position={_format_preview_point(end_position)}")
        pocket_id = str(detail.get("pocket_id") or "").strip()
        if pocket_id:
            preview_parts.append(f"pocket_id={pocket_id}")
        head = ", ".join(preview_parts[:4])
        extra = max(0, len(preview_parts) - 4)
        suffix = f", ... +{extra}" if extra else ""
        parts.append(f"{object_key}({head}{suffix})")
    return "; ".join(parts)


def _format_preview_point(value: Any) -> str:
    point = _xy_point(value)
    if point is None:
        return "null"
    if isinstance(value, dict) and isinstance(value.get("z"), (int, float)):
        try:
            return f"{{x:{float(value.get('x')):.3f}, y:{float(value.get('y')):.3f}, z:{float(value.get('z')):.3f}}}"
        except (TypeError, ValueError):
            return "null"
    return f"{{x:{point[0]:.3f}, y:{point[1]:.3f}}}"


def _result_outcome_kind(payload: dict[str, Any]) -> str | None:
    success_value = payload.get("success")
    if success_value is True:
        return "success"
    if success_value is False:
        return "failure"
    return None


def _record_index(record: dict[str, Any]) -> int:
    try:
        return int(record.get("index") or 0)
    except (TypeError, ValueError):
        return 0


def _trace_record_candidates(
    project_root: Path,
    payload: dict[str, Any],
) -> list[tuple[int, bool, dict[str, Any], Path, TraceMediaPaths]]:
    records = payload.get("records")
    if not isinstance(records, list):
        return []
    candidates: list[tuple[int, bool, dict[str, Any], Path, TraceMediaPaths]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        trace_path = _normalize_video_asset_path(project_root, str(item.get("trace_path") or ""))
        if not isinstance(trace_path, Path) or not trace_path.is_file():
            continue
        media_paths = resolve_trace_media_paths(trace_path)
        candidates.append((_record_index(item), item.get("success") is True, item, trace_path, media_paths))
    candidates.sort(key=lambda value: value[0])
    return candidates


def _select_trace_record_candidate(
    project_root: Path,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], Path, TraceMediaPaths] | None:
    candidates = _trace_record_candidates(project_root, payload)
    if not candidates:
        final_trace_path = _normalize_video_asset_path(project_root, str(payload.get("final_trace_path") or ""))
        if not isinstance(final_trace_path, Path) or not final_trace_path.is_file():
            return None
        media_paths = resolve_trace_media_paths(final_trace_path)
        return (
            {"index": 0, "success": payload.get("success") is True, "trace_path": str(final_trace_path)},
            final_trace_path,
            media_paths,
        )
    successful = [item for item in candidates if item[1]]
    chosen = successful[-1] if successful else candidates[-1]
    _index, _success, record, trace_path, media_paths = chosen
    return record, trace_path, media_paths


def select_video_showcase_entries(
    entries: list[VideoRenderResultEntry],
    *,
    success_target: int = 15,
    failure_target: int = 5,
    total_target: int = 20,
) -> list[VideoRenderResultEntry]:
    deduped: dict[tuple[str, str], VideoRenderResultEntry] = {}
    for entry in sorted(entries, key=lambda item: (item.updated_at, item.source_run_id, item.task_id), reverse=True):
        outcome_kind = entry.outcome_kind if entry.outcome_kind in {"success", "failure"} else "success"
        deduped.setdefault((outcome_kind, entry.task_id), entry)

    successes = [entry for entry in deduped.values() if entry.outcome_kind == "success"]
    failures = [entry for entry in deduped.values() if entry.outcome_kind == "failure"]
    env_id = str(entries[0].env_id if entries else "").strip().lower()

    def _annotation_count(entry: VideoRenderResultEntry) -> int:
        path = entry.annotation_path
        if not isinstance(path, Path) or not path.is_file():
            return 0
        payload = _load_json_object(path)
        annotations = payload.get("pattern_annotations")
        if not isinstance(annotations, dict):
            return 0
        patterns = annotations.get("patterns")
        if not isinstance(patterns, list):
            return 0
        return sum(1 for item in patterns if isinstance(item, dict))

    if env_id == "kinetix":
        dense_successes = [item for item in successes if _annotation_count(item) >= 3]
        dense_failures = [item for item in failures if _annotation_count(item) >= 3]
        if dense_successes or dense_failures:
            successes = dense_successes
            failures = dense_failures
        else:
            positive_successes = [item for item in successes if _annotation_count(item) > 0]
            positive_failures = [item for item in failures if _annotation_count(item) > 0]
            if positive_successes or positive_failures:
                successes = positive_successes
                failures = positive_failures
        successes.sort(key=lambda item: (_annotation_count(item), item.updated_at, item.task_id), reverse=True)
        failures.sort(key=lambda item: (_annotation_count(item), item.updated_at, item.task_id), reverse=True)
    else:
        successes.sort(key=lambda item: (item.updated_at, item.task_id), reverse=True)
        failures.sort(key=lambda item: (item.updated_at, item.task_id), reverse=True)

    selected = successes[:max(0, success_target)] + failures[:max(0, failure_target)]
    if env_id != "kinetix" and len(selected) < total_target:
        seen = {(item.outcome_kind, item.task_id) for item in selected}
        remainder = successes[max(0, success_target):] + failures[max(0, failure_target):]
        for entry in remainder:
            key = (entry.outcome_kind, entry.task_id)
            if key in seen:
                continue
            selected.append(entry)
            seen.add(key)
            if len(selected) >= total_target:
                break
    if env_id == "kinetix":
        selected.sort(
            key=lambda item: (_annotation_count(item), item.outcome_kind == "failure", item.updated_at, item.task_id),
            reverse=True,
        )
        return selected[:max(1, min(total_target, len(selected)))] if selected else []
    selected.sort(
        key=lambda item: (item.outcome_kind == "failure", item.updated_at, item.task_id),
        reverse=False,
    )
    return selected[:max(1, total_target)]


def _prefer_result_entry(left: VideoRenderResultEntry, right: VideoRenderResultEntry) -> VideoRenderResultEntry:
    def _score(entry: VideoRenderResultEntry) -> tuple[int, int, int, float]:
        return (
            1 if isinstance(entry.annotated_video_path, Path) and entry.annotated_video_path.is_file() else 0,
            1 if isinstance(entry.annotated_overhead_video_path, Path) and entry.annotated_overhead_video_path.is_file() else 0,
            1 if str(entry.output_run_id).startswith("video-render-") else 0,
            entry.updated_at.timestamp(),
        )

    return left if _score(left) >= _score(right) else right


def _merge_result_entries(entries: list[VideoRenderResultEntry]) -> list[VideoRenderResultEntry]:
    merged: dict[tuple[str, str, str], VideoRenderResultEntry] = {}
    for entry in entries:
        key = (entry.source_run_id, entry.task_id, entry.outcome_kind)
        current = merged.get(key)
        merged[key] = entry if current is None else _prefer_result_entry(current, entry)
    return sorted(merged.values(), key=lambda item: (item.updated_at, item.task_id), reverse=True)


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


def _normalized_label_key(value: str) -> str:
    return _NON_ALNUM_RE.sub(" ", str(value or "").strip().lower()).strip()


def _slugify_label(value: str) -> str:
    slug = _NON_ALNUM_RE.sub("_", str(value or "").strip().lower()).strip("_")
    return slug or "label"


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _canonical_library_fingerprint(library_path: Path | None) -> str:
    if not isinstance(library_path, Path):
        return ""
    meta_path = library_path.parent / "meta.json"
    meta = _load_json_object(meta_path)
    fingerprint = str(meta.get("fingerprint") or "").strip()
    if fingerprint:
        return fingerprint
    try:
        return hashlib.sha1(library_path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _score_value(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    if parsed != parsed:
        return float("-inf")
    return parsed


def _candidate_rank_key(payload: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        _score_value(payload.get("train_score")),
        _score_value(payload.get("average_score")),
        _score_value(payload.get("val_score")),
        str(payload.get("uid") or ""),
    )


def _best_library_cache_root(project_root: Path, env_id: str) -> Path:
    return _state_root(project_root) / "best_supervised_libraries" / env_id


def _best_library_paths(project_root: Path, env_id: str) -> tuple[Path, Path]:
    root = _best_library_cache_root(project_root, env_id)
    return root / "pattern_library.json", root / "meta.json"


def _load_supervised_labels(project_root: Path, env_id: str) -> tuple[list[str], Path]:
    candidates = [project_root / "configs" / env_id / "labels.txt"]
    for path in candidates:
        if not path.is_file():
            continue
        labels = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if labels:
            return labels, path
    raise FileNotFoundError(f"Missing supervised label list for {env_id}")


def _supervised_library_candidates(project_root: Path, env_id: str) -> list[Any]:
    candidates = [
        item
        for item in list_environment_libraries(project_root, env_id)
        if str(item.workflow_kind or "").strip().lower() == "supervised"
        and ".bak." not in item.run_id
        and ".broken-" not in item.run_id
        and not str(item.run_id).startswith("video-render-best-supervised-")
    ]
    if not candidates:
        raise FileNotFoundError(f"No supervised library found for {env_id}")
    deduped: dict[str, Any] = {}
    for item in candidates:
        deduped.setdefault(str(item.root.resolve()), item)
    return sorted(deduped.values(), key=lambda item: _supervised_library_rank(item.root), reverse=True)


def _parse_label_dir(path: Path) -> tuple[int | None, str]:
    match = _LABEL_DIR_PATTERN.match(path.name)
    if not match:
        return None, path.name
    index_raw = str(match.group("index") or "").strip()
    label_raw = str(match.group("label") or "").strip().replace("_", " ")
    try:
        index = int(index_raw)
    except ValueError:
        index = None
    return index, label_raw or path.name


def _find_label_dir(run_dir: Path, *, label_index: int, label: str) -> Path | None:
    expected = run_dir / f"label_{label_index:03d}_{_slugify_label(label)}"
    if expected.is_dir():
        return expected
    target_key = _normalized_label_key(label)
    fallback: Path | None = None
    for candidate in sorted(path for path in run_dir.glob("label_*") if path.is_dir()):
        parsed_index, parsed_label = _parse_label_dir(candidate)
        if parsed_index == label_index and target_key == _normalized_label_key(parsed_label):
            return candidate
        if fallback is None and target_key == _normalized_label_key(parsed_label):
            fallback = candidate
    return fallback


def _load_candidate_code(candidate_dir: Path, payload: dict[str, Any]) -> str | None:
    source_path = str(payload.get("source_path") or "").strip()
    if source_path:
        path = Path(source_path)
        if not path.is_file():
            path = candidate_dir / source_path
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8").strip()
            except OSError:
                text = ""
            if text:
                return text
    for path in sorted(candidate_dir.glob("*.py")):
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8").strip()
            except OSError:
                text = ""
            if text:
                return text
    return None


def _best_label_candidate_for_run(
    run_dir: Path,
    *,
    label_index: int,
    label: str,
) -> dict[str, Any] | None:
    label_dir = _find_label_dir(run_dir, label_index=label_index, label=label)
    if label_dir is None:
        return None

    best: dict[str, Any] | None = None
    for candidate_dir in sorted(path for path in label_dir.glob("candidate_*") if path.is_dir()):
        payload = _load_json_object(candidate_dir / "candidate.json")
        if not payload:
            continue
        source_code = _load_candidate_code(candidate_dir, payload)
        if not source_code:
            continue
        normalized = {
            "uid": str(payload.get("uid") or payload.get("candidate_uid") or candidate_dir.name).strip(),
            "train_score": payload.get("train_score"),
            "average_score": payload.get("average_score"),
            "val_score": payload.get("val_score"),
            "source_code": source_code,
            "input_mode": str(payload.get("input_mode") or "trace").strip() or "trace",
            "metadata": dict(payload.get("metadata") or {}),
            "source_run_dir": str(run_dir),
            "source_label_dir": str(label_dir),
        }
        if best is None or _candidate_rank_key(normalized) > _candidate_rank_key(best):
            best = normalized

    if best is not None:
        return best

    label_result = _load_json_object(label_dir / "label_result.json")
    detector_payload = label_result.get("detector")
    if not isinstance(detector_payload, dict):
        return None
    source_code = str(detector_payload.get("source_code") or "").strip()
    if not source_code:
        return None
    return {
        "uid": str(label_result.get("best_candidate_uid") or f"{run_dir.name}:{label}").strip(),
        "train_score": label_result.get("best_train_score"),
        "average_score": label_result.get("best_score"),
        "val_score": label_result.get("best_validation_score"),
        "source_code": source_code,
        "input_mode": str(detector_payload.get("input_mode") or "trace").strip() or "trace",
        "metadata": {"synthetic_from_label_result": True},
        "source_run_dir": str(run_dir),
        "source_label_dir": str(label_dir),
    }


def _best_library_fingerprint(
    *,
    labels: list[str],
    labels_path: Path,
    supervised_runs: list[Any],
) -> str:
    labels_digest = hashlib.sha1(
        "\n".join(labels).encode("utf-8")
    ).hexdigest()
    parts = [
        f"labels={labels_digest}",
        f"labels_mtime={labels_path.stat().st_mtime_ns if labels_path.exists() else 0}",
    ]
    for entry in supervised_runs:
        parts.append(
            f"{entry.run_id}|{entry.updated_at.isoformat()}|{entry.detector_count}"
        )
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()


def _resolve_best_supervised_library(project_root: Path, env_id: str) -> Path:
    labels, labels_path = _load_supervised_labels(project_root, env_id)
    supervised_runs = _supervised_library_candidates(project_root, env_id)
    fingerprint = _best_library_fingerprint(
        labels=labels,
        labels_path=labels_path,
        supervised_runs=supervised_runs,
    )
    cached = _BEST_LIBRARY_CACHE.get(env_id)
    if cached and cached[0] == fingerprint and cached[1].is_file():
        return cached[1]

    library_path, meta_path = _best_library_paths(project_root, env_id)
    if library_path.is_file() and meta_path.is_file():
        meta = _load_json_object(meta_path)
        if str(meta.get("fingerprint") or "").strip() == fingerprint:
            _BEST_LIBRARY_CACHE[env_id] = (fingerprint, library_path)
            return library_path

    _ensure_client_src_on_path()
    from pattern_library_schema import DetectorDefinition, PatternLibrary  # type: ignore

    detectors: list[Any] = []
    for label_index, label in enumerate(labels, start=1):
        best: dict[str, Any] | None = None
        for entry in supervised_runs:
            candidate = _best_label_candidate_for_run(
                entry.root,
                label_index=label_index,
                label=label,
            )
            if candidate is None:
                continue
            if best is None or _candidate_rank_key(candidate) > _candidate_rank_key(best):
                best = {
                    **candidate,
                    "source_run_id": entry.run_id,
                }
        if best is None:
            continue
        metadata = dict(best.get("metadata") or {})
        metadata.update(
            {
                "candidate_uid": best.get("uid"),
                "score": best.get("average_score"),
                "train_score": best.get("train_score"),
                "val_score": best.get("val_score"),
                "source_run_id": best.get("source_run_id"),
                "source_run_dir": best.get("source_run_dir"),
                "source_label_dir": best.get("source_label_dir"),
                "selection_strategy": "best_per_label_across_supervised_runs",
            }
        )
        metadata = {key: value for key, value in metadata.items() if value is not None}
        detectors.append(
            DetectorDefinition(
                detector_id=f"primitive-{label_index:03d}",
                label=label,
                environment_ids=[env_id],
                source_code=str(best.get("source_code") or ""),
                input_mode=(
                    str(best.get("input_mode") or "trace").strip()
                    if str(best.get("input_mode") or "trace").strip() in {"trace", "supervised_evidence"}
                    else "trace"
                ),
                metadata=metadata,
            )
        )

    if not detectors:
        raise FileNotFoundError(f"No supervised label candidates available for {env_id}")

    top_library_payload = _load_json_object(supervised_runs[0].library_path)
    library = PatternLibrary(
        library_id=f"video-render-best-supervised-{env_id}",
        environment_ids=[env_id],
        primitive_source=str(labels_path),
        scorer_config=dict(top_library_payload.get("scorer_config") or {}),
        detectors=detectors,
        metadata={
            "source": "video_renders_best_supervised_global",
            "selection": "best_per_label_across_supervised_runs",
            "source_run_dirs": [str(entry.root) for entry in supervised_runs],
            "source_run_ids": [str(entry.run_id) for entry in supervised_runs],
            "detector_count": len(detectors),
            "label_count": len(labels),
            "fingerprint": fingerprint,
            "generated_at": _isoformat(),
        },
    )
    library_path.parent.mkdir(parents=True, exist_ok=True)
    library_path.write_text(library.model_dump_json(indent=2) + "\n", encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "generated_at": _isoformat(),
                "library_path": str(library_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _BEST_LIBRARY_CACHE[env_id] = (fingerprint, library_path)
    return library_path


def load_global_best_supervised_library_summary(project_root: Path, env_id: str) -> dict[str, Any]:
    return _load_global_best_supervised_library_summary(project_root, env_id)


def load_cached_global_best_supervised_library_summary(project_root: Path, env_id: str) -> dict[str, Any]:
    return _load_cached_canonical_library_summary(project_root, env_id, "supervised")


def _available_annotation_labels(project_root: Path, env_id: str) -> list[str]:
    summary = load_cached_global_best_supervised_library_summary(project_root, env_id)
    rows = summary.get("rows")
    if not isinstance(rows, list):
        return []
    labels: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label") or "").strip()
        key = _normalized_label_key(label)
        if not key or key in seen:
            continue
        seen.add(key)
        labels.append(label)
    return labels


def _canonicalize_selected_labels(*, available_labels: list[str], selected_labels: list[str] | None) -> list[str]:
    if not available_labels:
        return []
    available_by_key = {
        _normalized_label_key(label): label
        for label in available_labels
        if _normalized_label_key(label)
    }
    if not selected_labels:
        return list(available_labels)
    chosen_keys: set[str] = set()
    chosen: list[str] = []
    for raw_label in selected_labels:
        key = _normalized_label_key(raw_label)
        canonical = available_by_key.get(key)
        if canonical is None or key in chosen_keys:
            continue
        chosen_keys.add(key)
        chosen.append(canonical)
    return chosen or list(available_labels)


def load_annotation_video_selection(project_root: Path, env_id: str) -> dict[str, Any]:
    env_key = str(env_id or "").strip().lower()
    available_labels = _available_annotation_labels(project_root, env_key)
    config = _load_annotation_config(project_root)
    entry = dict(config.get(env_key, {}))
    raw_selected = entry.get("selected_labels")
    selected = (
        [str(item).strip() for item in raw_selected if str(item).strip()]
        if isinstance(raw_selected, list)
        else None
    )
    canonical = _canonicalize_selected_labels(
        available_labels=available_labels,
        selected_labels=selected,
    )
    return {
        "env_id": env_key,
        "available_labels": available_labels,
        "selected_labels": canonical,
        "selected_count": len(canonical),
        "updated_at": str(entry.get("updated_at") or "").strip(),
    }


def save_annotation_video_selection(project_root: Path, env_id: str, selected_labels: list[str]) -> dict[str, Any]:
    env_key = str(env_id or "").strip().lower()
    payload = load_annotation_video_selection(project_root, env_key)
    canonical = _canonicalize_selected_labels(
        available_labels=list(payload.get("available_labels") or []),
        selected_labels=selected_labels,
    )
    config = _load_annotation_config(project_root)
    config[env_key] = {
        "selected_labels": canonical,
        "updated_at": _isoformat(),
    }
    _write_annotation_config(project_root, config)
    return {
        **payload,
        "selected_labels": canonical,
        "selected_count": len(canonical),
        "updated_at": config[env_key]["updated_at"],
    }


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
                    "source": "videos-supervised-seed",
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
            "source": "videos",
            "trace_id": getattr(trace, "trace_id", ""),
            "task_id": getattr(trace, "task_id", ""),
            "supervised_library_path": str(library_path),
            "supervised_library_fingerprint": _canonical_library_fingerprint(library_path),
            "seed_supervised_library_path": upstream_supervised_library_path,
            "videos_annotation_version": VIDEO_ANNOTATION_VERSION,
        },
        pattern_limit=len(patterns),
    )


def _annotation_matches_library(payload: dict[str, Any], library_path: Path, *, trace_path: Path | None = None) -> bool:
    annotations = payload.get("pattern_annotations")
    if not isinstance(annotations, dict):
        return False
    metadata = annotations.get("metadata")
    if not isinstance(metadata, dict):
        return False
    try:
        version = int(metadata.get("videos_annotation_version") or 0)
    except (TypeError, ValueError):
        version = 0
    if version < VIDEO_ANNOTATION_VERSION:
        return False
    existing = str(metadata.get("supervised_library_path") or "").strip()
    if not existing:
        return False
    existing_fingerprint = str(metadata.get("supervised_library_fingerprint") or "").strip()
    expected_fingerprint = _canonical_library_fingerprint(library_path)
    if not existing_fingerprint or not expected_fingerprint:
        return False
    if isinstance(trace_path, Path) and trace_path.is_file():
        existing_trace_path = str(metadata.get("videos_trace_path") or "").strip()
        existing_trace_fingerprint = str(metadata.get("videos_trace_fingerprint") or "").strip()
        expected_trace_fingerprint = _trace_payload_fingerprint(trace_path)
        if not existing_trace_path or not existing_trace_fingerprint or not expected_trace_fingerprint:
            return False
        try:
            if Path(existing_trace_path).resolve() != trace_path.resolve():
                return False
        except OSError:
            if existing_trace_path != str(trace_path):
                return False
        if existing_trace_fingerprint != expected_trace_fingerprint:
            return False
    try:
        return (
            Path(existing).resolve() == library_path.resolve()
            and existing_fingerprint == expected_fingerprint
        )
    except OSError:
        return existing == str(library_path) and existing_fingerprint == expected_fingerprint


def _ensure_best_supervised_annotation(
    project_root: Path,
    env_id: str,
    *,
    trace_path: Path | None,
    task_dir: Path,
) -> tuple[Path | None, str, Path | None]:
    if not isinstance(trace_path, Path) or not trace_path.is_file():
        return None, "", None
    library_path = resolve_canonical_library(project_root, env_id, "supervised")
    annotation_path = _video_annotation_path(task_dir)
    payload: dict[str, Any] | None = None
    if annotation_path.is_file():
        try:
            existing = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and _annotation_matches_library(existing, library_path, trace_path=trace_path):
            payload = existing
    if payload is None:
        _ensure_client_src_on_path()
        from deepphy_eval.io import load_trace  # type: ignore

        trace = load_trace(trace_path)
        payload = _apply_library_to_trace(project_root, trace, library_path)
        annotations = payload.get("pattern_annotations")
        if isinstance(annotations, dict):
            metadata = annotations.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                annotations["metadata"] = metadata
            metadata["videos_trace_path"] = str(trace_path)
            metadata["videos_trace_fingerprint"] = _trace_payload_fingerprint(trace_path)
        annotation_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return annotation_path, _format_annotation_preview(payload, env_id=env_id), library_path


def _load_existing_best_supervised_annotation(
    project_root: Path,
    env_id: str,
    *,
    task_dir: Path,
    trace_path: Path | None = None,
) -> tuple[Path | None, str, Path | None]:
    library_path = resolve_canonical_library(project_root, env_id, "supervised")
    annotation_path = _video_annotation_path(task_dir)
    if not annotation_path.is_file():
        return None, "", library_path
    try:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "", library_path
    if not isinstance(payload, dict):
        return None, "", library_path
    if not _annotation_matches_library(payload, library_path, trace_path=trace_path):
        return None, "", library_path
    return annotation_path, _format_annotation_preview(payload, env_id=env_id), library_path


def _annotated_video_paths(task_dir: Path) -> tuple[Path, Path, Path]:
    return (
        task_dir / "simulation_annotated.mp4",
        task_dir / "simulation_overhead_annotated.mp4",
        task_dir / "annotation_video_meta.json",
    )


def _annotation_payload_fingerprint(annotation_path: Path | None) -> str:
    if not isinstance(annotation_path, Path) or not annotation_path.is_file():
        return ""
    try:
        return hashlib.sha1(annotation_path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _trace_payload_fingerprint(trace_path: Path | None) -> str:
    if not isinstance(trace_path, Path) or not trace_path.is_file():
        return ""
    try:
        return hashlib.sha1(trace_path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _annotated_video_meta_matches_current_state(
    *,
    project_root: Path,
    env_id: str,
    task_dir: Path,
    annotation_path: Path | None,
    trace_path: Path | None = None,
) -> bool:
    if not isinstance(annotation_path, Path) or not annotation_path.is_file():
        return False
    _annotated_video_path, _annotated_overhead_path, meta_path = _annotated_video_paths(task_dir)
    if not meta_path.is_file():
        return False
    meta = _load_json_object(meta_path)
    if not meta:
        return False
    current_selection = load_annotation_video_selection(project_root, env_id)
    current_labels = [
        str(label).strip()
        for label in (current_selection.get("selected_labels") or [])
        if str(label).strip()
    ]
    meta_labels = [
        str(label).strip()
        for label in (meta.get("selected_labels") or [])
        if str(label).strip()
    ] if isinstance(meta.get("selected_labels"), list) else []
    if meta_labels != current_labels:
        return False
    try:
        current_mtime_ns = int(annotation_path.stat().st_mtime_ns)
    except OSError:
        return False
    if int(meta.get("annotation_mtime_ns") or 0) != current_mtime_ns:
        return False
    current_fingerprint = _annotation_payload_fingerprint(annotation_path)
    meta_fingerprint = str(meta.get("annotation_fingerprint") or "").strip()
    if not current_fingerprint or meta_fingerprint != current_fingerprint:
        return False
    if isinstance(trace_path, Path) and trace_path.is_file():
        current_trace_fingerprint = _trace_payload_fingerprint(trace_path)
        meta_trace_fingerprint = str(meta.get("trace_fingerprint") or "").strip()
        meta_trace_path = str(meta.get("trace_path") or "").strip()
        if not current_trace_fingerprint or meta_trace_fingerprint != current_trace_fingerprint:
            return False
        try:
            if Path(meta_trace_path).resolve() != trace_path.resolve():
                return False
        except OSError:
            if meta_trace_path != str(trace_path):
                return False
    return True


def _load_trace_payload(trace_path: Path | None) -> dict[str, Any]:
    if not isinstance(trace_path, Path) or not trace_path.is_file():
        return {}
    return _load_json_object(trace_path)


def _trace_video_attachment_metadata(trace_payload: dict[str, Any], *, view: str) -> dict[str, Any]:
    attachments = trace_payload.get("attachments")
    if not isinstance(attachments, list):
        return {}
    fallback: dict[str, Any] = {}
    for item in attachments:
        if not isinstance(item, dict):
            continue
        if str(item.get("kind") or "").strip().lower() != "simulation_video":
            continue
        metadata = item.get("metadata")
        meta = metadata if isinstance(metadata, dict) else {}
        attachment_view = str(meta.get("view") or "").strip().lower()
        if attachment_view == str(view or "").strip().lower():
            return meta
        if not attachment_view and not fallback:
            fallback = meta
    return fallback


def _entry_source_video_path(entry: VideoRenderResultEntry, *, view: str) -> Path | None:
    view_key = str(view or "").strip().lower()
    if view_key == "overhead" and isinstance(entry.overhead_video_path, Path) and entry.overhead_video_path.is_file():
        return entry.overhead_video_path
    if isinstance(entry.video_path, Path) and entry.video_path.is_file():
        return entry.video_path
    return None


def _video_file_metrics(source_path: Path) -> tuple[float | None, int | None]:
    if not isinstance(source_path, Path) or not source_path.is_file():
        return None, None
    reader = imageio.get_reader(str(source_path))
    try:
        meta = reader.get_meta_data() or {}
        fps = None
        raw_fps = meta.get("fps")
        if isinstance(raw_fps, (int, float)) and float(raw_fps) > 0:
            fps = float(raw_fps)
        frame_count = None
        try:
            counted = int(reader.count_frames())
            if counted > 0:
                frame_count = counted
        except Exception:
            raw_count = meta.get("nframes")
            if isinstance(raw_count, (int, float)) and int(raw_count) > 0:
                frame_count = int(raw_count)
        return fps, frame_count
    finally:
        reader.close()


def _xy_point(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        x = value.get("x")
        y = value.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return float(x), float(y)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x, y = value[0], value[1]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return float(x), float(y)
    return None


def _pooltool_trace_bounds(trace_payload: dict[str, Any]) -> tuple[float, float, float, float]:
    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")
    initial_state = trace_payload.get("initial_state")
    initial = initial_state if isinstance(initial_state, dict) else {}
    initial_ball_positions = initial.get("initial_ball_positions")
    if isinstance(initial_ball_positions, list):
        for item in initial_ball_positions:
            point = _xy_point(item)
            if point is None:
                continue
            min_x = min(min_x, point[0])
            max_x = max(max_x, point[0])
            min_y = min(min_y, point[1])
            max_y = max(max_y, point[1])
    timesteps = trace_payload.get("timesteps")
    if isinstance(timesteps, list):
        for timestep in timesteps:
            if not isinstance(timestep, dict):
                continue
            objects = timestep.get("objects")
            if not isinstance(objects, list):
                continue
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                point = _xy_point(obj.get("position"))
                if point is None:
                    point = _xy_point({"x": obj.get("x"), "y": obj.get("y")})
                if point is None:
                    continue
                min_x = min(min_x, point[0])
                max_x = max(max_x, point[0])
                min_y = min(min_y, point[1])
                max_y = max(max_y, point[1])
    if not all(value == value for value in (min_x, max_x, min_y, max_y)):
        return (0.0, 1.0, 0.0, 1.0)
    if max_x <= min_x:
        max_x = min_x + 1.0
    if max_y <= min_y:
        max_y = min_y + 1.0
    padding_x = (max_x - min_x) * 0.08
    padding_y = (max_y - min_y) * 0.08
    return (min_x - padding_x, max_x + padding_x, min_y - padding_y, max_y + padding_y)


def _project_video_point(
    *,
    env_id: str,
    point: tuple[float, float] | None,
    width: int,
    height: int,
    trace_payload: dict[str, Any],
    view: str | None = None,
) -> tuple[float, float] | None:
    if point is None or width <= 0 or height <= 0:
        return None
    x_value, y_value = point
    if env_id == "pooltool":
        # Some PoolTool annotations emit placeholder cue points at the world origin.
        # Treat those as missing so they do not pin overlays to the top-left corner.
        if abs(x_value) <= 1e-6 and abs(y_value) <= 1e-6:
            return None
        initial_state = trace_payload.get("initial_state")
        initial = initial_state if isinstance(initial_state, dict) else {}
        try:
            table_width = max(float(initial.get("table_width") or POOLTOOL_TABLE_WIDTH), 1e-6)
        except (TypeError, ValueError):
            table_width = POOLTOOL_TABLE_WIDTH
        try:
            table_length = max(
                float(initial.get("table_length") or initial.get("table_height") or POOLTOOL_TABLE_LENGTH),
                1e-6,
            )
        except (TypeError, ValueError):
            table_length = POOLTOOL_TABLE_LENGTH
        normalized_x = x_value / table_width
        normalized_y = 1.0 - (y_value / table_length)
        pooltool_view = str(view or "").strip().lower()
        if pooltool_view in {"", "overhead"}:
            normalized_x = POOLTOOL_OVERHEAD_VIEWPORT_LEFT + normalized_x * (
                POOLTOOL_OVERHEAD_VIEWPORT_RIGHT - POOLTOOL_OVERHEAD_VIEWPORT_LEFT
            )
            normalized_y = POOLTOOL_OVERHEAD_VIEWPORT_TOP + normalized_y * (
                POOLTOOL_OVERHEAD_VIEWPORT_BOTTOM - POOLTOOL_OVERHEAD_VIEWPORT_TOP
            )
    elif env_id == "iphyre":
        normalized_x = x_value / IPHYRE_COORD_SPACE_SIZE
        normalized_y = y_value / IPHYRE_COORD_SPACE_SIZE
    elif env_id == "kinetix":
        initial_state = trace_payload.get("initial_state")
        state = initial_state if isinstance(initial_state, dict) else {}
        try:
            world_width = max(float(state.get("render_world_width") or 0.0), 1e-6)
        except (TypeError, ValueError):
            world_width = 5.0
        try:
            world_height = max(float(state.get("render_world_height") or 0.0), 1e-6)
        except (TypeError, ValueError):
            world_height = 5.0
        normalized_x = x_value / world_width
        normalized_y = 1.0 - (y_value / world_height)
    else:
        normalized_x = x_value
        normalized_y = 1.0 - y_value if env_id == "phyre" else y_value
    normalized_x = min(1.0, max(0.0, normalized_x))
    normalized_y = min(1.0, max(0.0, normalized_y))
    return normalized_x * width, normalized_y * height


def _pattern_activation_key(pattern: dict[str, Any]) -> str:
    label = _normalized_label_key(str(pattern.get("label") or "").strip())
    detector_id = str(pattern.get("detector_id") or "").strip()
    parameters = pattern.get("parameters")
    params = parameters if isinstance(parameters, dict) else {}
    participant_tokens: list[str] = []
    object_positions = params.get("object_positions")
    if isinstance(object_positions, list):
        for item in object_positions:
            if not isinstance(item, dict):
                continue
            object_index = item.get("object_index")
            if isinstance(object_index, (int, str)):
                participant_tokens.append(f"pos:{object_index}")
    object_indices = params.get("object_indices")
    if isinstance(object_indices, list):
        for value in object_indices:
            if isinstance(value, (int, str)):
                participant_tokens.append(f"idx:{value}")
    object_details = params.get("object_details")
    if isinstance(object_details, dict):
        for object_key, raw_detail in sorted(object_details.items()):
            if isinstance(raw_detail, dict):
                role = str(raw_detail.get("role") or raw_detail.get("roles") or "").strip()
                participant_tokens.append(f"detail:{object_key}:{role}")
            else:
                participant_tokens.append(f"detail:{object_key}")
    if not participant_tokens:
        description = str(params.get("description") or "").strip()
        if description:
            participant_tokens.append(f"description:{description}")
    payload = {
        "label": label,
        "detector_id": detector_id,
        "participants": sorted(set(token for token in participant_tokens if token)),
    }
    try:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except TypeError:
        raw = str(payload)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _pattern_event_points(pattern: dict[str, Any], *, event_kind: str) -> list[tuple[float, float]]:
    parameters = pattern.get("parameters")
    if not isinstance(parameters, dict):
        return []
    if _normalized_label_key(pattern.get("label")) == "ball pocketed":
        pocket_point = _pooltool_pocket_position_from_parameters(parameters)
        if pocket_point is not None:
            return [pocket_point]
    points: list[tuple[float, float]] = []
    use_end = str(event_kind).strip().lower() == "end"
    object_positions = parameters.get("object_positions")
    if isinstance(object_positions, list):
        for item in object_positions:
            if not isinstance(item, dict):
                continue
            point = _xy_point(item.get("end" if use_end else "start"))
            if point is not None:
                points.append(point)
    object_details = parameters.get("object_details")
    if isinstance(object_details, dict):
        for item in object_details.values():
            if not isinstance(item, dict):
                continue
            point = _xy_point(item.get("end_position" if use_end else "start_position"))
            if point is not None:
                points.append(point)
    direct_point = _xy_point(parameters.get("end_position" if use_end else "start_position"))
    if direct_point is not None:
        points.append(direct_point)
    deduped: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()
    for point in points:
        key = (round(point[0] * 1000), round(point[1] * 1000))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(point)
    return deduped


def _pooltool_ball_pocketed_matches_trace(pattern: dict[str, Any], trace_payload: dict[str, Any]) -> bool:
    parameters = pattern.get("parameters")
    if not isinstance(parameters, dict):
        return False
    candidate_ball_ids: set[str] = set()
    ball_id = parameters.get("ball_id")
    if isinstance(ball_id, (str, int)):
        token = str(ball_id).strip()
        if token:
            candidate_ball_ids.add(token)
    object_details = parameters.get("object_details")
    if isinstance(object_details, dict):
        for object_key, detail in object_details.items():
            token = str(object_key).strip()
            if token and token.lower() not in {"lb", "lc", "lt", "rb", "rc", "rt"}:
                candidate_ball_ids.add(token)
            if isinstance(detail, dict):
                identity = detail.get("ball_identity")
                if isinstance(identity, (str, int)):
                    identity_token = str(identity).strip()
                    if identity_token and identity_token.lower() not in {"lb", "lc", "lt", "rb", "rc", "rt"}:
                        candidate_ball_ids.add(identity_token)
    if not candidate_ball_ids:
        return False
    timesteps = trace_payload.get("timesteps")
    if not isinstance(timesteps, list) or not timesteps:
        return False
    try:
        start_timestep = int(pattern.get("start_timestep") or 0)
    except (TypeError, ValueError):
        start_timestep = 0
    start_index = max(0, min(len(timesteps) - 1, start_timestep))
    end_index = min(len(timesteps) - 1, start_index + 4)
    for timestep in timesteps[start_index : end_index + 1]:
        if not isinstance(timestep, dict):
            continue
        objects = timestep.get("objects")
        if not isinstance(objects, list):
            continue
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            if str(obj.get("ball_id") or "").strip() not in candidate_ball_ids:
                continue
            if str(obj.get("motion_state") or "").strip().lower() == "pocketed":
                return True
    return False


def _pooltool_ball_pocketed_timestep_from_trace(pattern: dict[str, Any], trace_payload: dict[str, Any]) -> int | None:
    parameters = pattern.get("parameters")
    if not isinstance(parameters, dict):
        return None
    candidate_ball_ids: set[str] = set()
    ball_id = parameters.get("ball_id")
    if isinstance(ball_id, (str, int)):
        token = str(ball_id).strip()
        if token:
            candidate_ball_ids.add(token)
    object_details = parameters.get("object_details")
    if isinstance(object_details, dict):
        for object_key, detail in object_details.items():
            token = str(object_key).strip()
            if token and token.lower() not in {"lb", "lc", "lt", "rb", "rc", "rt"}:
                candidate_ball_ids.add(token)
            if isinstance(detail, dict):
                identity = detail.get("ball_identity")
                if isinstance(identity, (str, int)):
                    identity_token = str(identity).strip()
                    if identity_token and identity_token.lower() not in {"lb", "lc", "lt", "rb", "rc", "rt"}:
                        candidate_ball_ids.add(identity_token)
    if not candidate_ball_ids:
        return None
    timesteps = trace_payload.get("timesteps")
    if not isinstance(timesteps, list) or not timesteps:
        return None
    try:
        start_timestep = int(pattern.get("start_timestep") or 0)
    except (TypeError, ValueError):
        start_timestep = 0
    start_index = max(0, min(len(timesteps) - 1, start_timestep - 2))
    end_index = min(len(timesteps) - 1, max(start_timestep, 0) + 6)
    for index in range(start_index, end_index + 1):
        timestep = timesteps[index]
        if not isinstance(timestep, dict):
            continue
        objects = timestep.get("objects")
        if not isinstance(objects, list):
            continue
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            if str(obj.get("ball_id") or "").strip() not in candidate_ball_ids:
                continue
            if str(obj.get("motion_state") or "").strip().lower() == "pocketed":
                return index
    return None


def _label_color(label: str) -> tuple[int, int, int]:
    digest = hashlib.sha1(str(label or "").encode("utf-8")).digest()
    hue = digest[0] / 255.0
    saturation = 0.62 + ((digest[1] / 255.0) * 0.18)
    value = 0.82 + ((digest[2] / 255.0) * 0.12)
    red, green, blue = colorsys.hsv_to_rgb(hue, min(1.0, saturation), min(1.0, value))
    return (int(red * 255), int(green * 255), int(blue * 255))


def _active_pattern_events_for_frame(
    annotation_payload: dict[str, Any],
    *,
    env_id: str,
    selected_labels: list[str],
    frame_index: int,
    total_frames: int,
    total_timesteps: int,
    min_output_frames: int,
) -> list[dict[str, Any]]:
    annotations = annotation_payload.get("pattern_annotations")
    if not isinstance(annotations, dict):
        return []
    patterns = annotations.get("patterns")
    if not isinstance(patterns, list):
        return []
    selected_keys = {_normalized_label_key(label) for label in selected_labels if _normalized_label_key(label)}
    active_by_key: dict[str, tuple[int, int, str, dict[str, Any]]] = {}
    for item in patterns:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if selected_keys and _normalized_label_key(label) not in selected_keys:
            continue
        start_timestep = int(item.get("start_timestep") or 0)
        end_timestep = int(item.get("end_timestep") or start_timestep)
        if total_timesteps <= 1 or total_frames <= 1:
            start_frame = 0
            end_frame = 0
        else:
            start_frame = int(
                round((start_timestep * max(0, total_frames - 1)) / max(1, total_timesteps - 1))
            )
            end_frame = int(
                round((max(start_timestep, end_timestep) * max(0, total_frames - 1)) / max(1, total_timesteps - 1))
            )
        event_specs = _pattern_event_specs(
            start_timestep=start_timestep,
            end_timestep=end_timestep,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        for event_kind, event_frame in event_specs:
            if not _annotation_event_passes_custom_filters(
                env_id=env_id,
                item=item,
                event_kind=event_kind,
                start_timestep=start_timestep,
                end_timestep=end_timestep,
                total_timesteps=total_timesteps,
            ):
                continue
            event_end_frame = min(max(0, total_frames - 1), event_frame + max(0, min_output_frames - 1))
            if frame_index < event_frame or frame_index > event_end_frame:
                continue
            activation_key = f"{_pattern_activation_key(item)}:{event_kind}"
            current = active_by_key.get(activation_key)
            if current is None or (event_frame, event_end_frame) >= (current[0], current[1]):
                active_by_key[activation_key] = (event_frame, event_end_frame, event_kind, item)
    ordered = sorted(active_by_key.values(), key=lambda value: (value[0], value[1], value[2]))
    return [(event_kind, item) for _start_frame, _end_frame, event_kind, item in ordered]


def _render_annotation_frame(
    *,
    frame: Any,
    env_id: str,
    trace_payload: dict[str, Any],
    annotation_payload: dict[str, Any],
    selected_labels: list[str],
    frame_index: int,
    total_frames: int,
    total_timesteps: int,
    min_output_frames: int,
    view: str | None = None,
) -> Any:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    font = _overlay_font(15)
    width, height = image.size
    circle_radius = max(4, int(round(max(6, min(14, int(round(min(width, height) * 0.014)))) * 0.75)))
    legend_entries: dict[str, tuple[int, int, int]] = {}
    annotations = annotation_payload.get("pattern_annotations")
    patterns = annotations.get("patterns") if isinstance(annotations, dict) else None
    if isinstance(patterns, list):
        selected_keys = {_normalized_label_key(label) for label in selected_labels if _normalized_label_key(label)}
        for pattern in patterns:
            if not isinstance(pattern, dict):
                continue
            label = str(pattern.get("label") or "").strip()
            if not label:
                continue
            if selected_keys and _normalized_label_key(label) not in selected_keys:
                continue
            if label not in legend_entries:
                legend_entries[label] = _label_color(label)
    for event_kind, pattern in _active_pattern_events_for_frame(
        annotation_payload,
        env_id=env_id,
        selected_labels=selected_labels,
        frame_index=frame_index,
        total_frames=total_frames,
        total_timesteps=total_timesteps,
        min_output_frames=min_output_frames,
    ):
        label = str(pattern.get("label") or "pattern").strip()
        points = _pattern_event_points(pattern, event_kind=event_kind)
        if not points:
            continue
        color = _label_color(label)
        projected = [
            _project_video_point(
                env_id=env_id,
                point=point,
                width=width,
                height=height,
                trace_payload=trace_payload,
                view=view,
            )
            for point in points
        ]
        projected = [point for point in projected if point is not None]
        if not projected:
            continue
        for point in projected:
            x_value, y_value = point
            draw.ellipse(
                (
                    x_value - circle_radius,
                    y_value - circle_radius,
                    x_value + circle_radius,
                    y_value + circle_radius,
                ),
                fill=(color[0], color[1], color[2], 88),
                outline=(color[0], color[1], color[2], 220),
                width=max(1, circle_radius // 4),
            )
    if legend_entries:
        swatch_size = 15
        row_height = 24
        legend_width = 0
        for legend_label in legend_entries:
            text_box = draw.textbbox((0, 0), legend_label, font=font)
            legend_width = max(legend_width, int(text_box[2] - text_box[0]))
        card_width = legend_width + swatch_size + 30
        card_height = (len(legend_entries) * row_height) + 14
        card_left = max(8, width - card_width - 8)
        card_top = max(8, height - card_height - 8)
        draw.rounded_rectangle(
            (card_left, card_top, card_left + card_width, card_top + card_height),
            radius=4,
            fill=(17, 17, 17, 176),
        )
        for index, (legend_label, legend_color) in enumerate(legend_entries.items()):
            row_top = card_top + 5 + (index * row_height)
            draw.rectangle(
                (card_left + 7, row_top + 5, card_left + 7 + swatch_size, row_top + 5 + swatch_size),
                fill=(legend_color[0], legend_color[1], legend_color[2], 255),
            )
            draw.text(
                (card_left + 7 + swatch_size + 8, row_top + 1),
                legend_label,
                fill=(255, 255, 255, 255),
                font=font,
            )
    rendered = Image.alpha_composite(image, overlay).convert("RGB")
    return np.asarray(rendered)


def _render_annotated_video(
    *,
    project_root: Path,
    env_id: str,
    trace_path: Path,
    annotation_path: Path,
    source_video_path: Path,
    output_video_path: Path,
    selected_labels: list[str],
    view: str,
    task_id: str,
    progress_index: int,
    progress_total: int,
) -> None:
    _merge_annotation_env_state(
        project_root,
        env_id,
        {
            "current_task_id": task_id,
            "current_view": view,
            "current_video_frame": 0,
            "current_video_total_frames": 0,
            "completed_tasks": max(0, progress_index - 1),
            "planned_tasks": progress_total,
        },
    )
    trace_payload = _load_trace_payload(trace_path)
    annotation_payload = _load_json_object(annotation_path)
    if not annotation_payload:
        raise FileNotFoundError(f"Missing annotation payload for {task_id}")
    video_attachment = _trace_video_attachment_metadata(trace_payload, view=view)
    total_timesteps = 0
    timesteps = trace_payload.get("timesteps")
    if isinstance(timesteps, list):
        total_timesteps = len(timesteps)
    if total_timesteps <= 0:
        total_timesteps = int(video_attachment.get("dense_timestep_count") or 0)
    _ensure_client_src_on_path()
    import imageio.v2 as imageio  # type: ignore

    reader = imageio.get_reader(str(source_video_path))
    try:
        meta = reader.get_meta_data() or {}
        source_fps = float(video_attachment.get("fps") or meta.get("fps") or 10.0)
        output_fps = source_fps
        output_fps = max(0.1, output_fps)
        total_frames = int(video_attachment.get("frame_count") or 0)
        if total_frames <= 0:
            try:
                total_frames = int(reader.count_frames())
            except Exception:
                total_frames = 0
        min_output_frames = max(1, int(round(MIN_ANNOTATION_DISPLAY_SECONDS * output_fps)))
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(output_video_path),
            fps=output_fps,
            codec="libx264",
            macro_block_size=None,
        )
        try:
            for frame_index, frame in enumerate(reader):
                if total_frames <= 0:
                    total_frames = frame_index + 1
                if frame_index == 0 or frame_index % 10 == 0:
                    _merge_annotation_env_state(
                        project_root,
                        env_id,
                        {
                            "current_task_id": task_id,
                            "current_view": view,
                            "current_video_frame": frame_index + 1,
                            "current_video_total_frames": total_frames,
                            "completed_tasks": max(0, progress_index - 1),
                            "planned_tasks": progress_total,
                        },
                    )
                writer.append_data(
                    _render_annotation_frame(
                        frame=frame,
                        env_id=env_id,
                        trace_payload=trace_payload,
                        annotation_payload=annotation_payload,
                        selected_labels=selected_labels,
                        frame_index=frame_index,
                        total_frames=max(total_frames, frame_index + 1),
                        total_timesteps=max(total_timesteps, frame_index + 1),
                        min_output_frames=min_output_frames,
                        view=view,
                    )
                )
        finally:
            writer.close()
    finally:
        reader.close()


def _rgb_hex(color: tuple[int, int, int]) -> str:
    red, green, blue = color
    return f"#{red:02x}{green:02x}{blue:02x}"


def build_annotation_overlay_payload(
    *,
    project_root: Path,
    entry: VideoRenderResultEntry,
    view: str,
    selected_labels: list[str],
) -> dict[str, Any] | None:
    if not isinstance(entry.annotation_path, Path) or not entry.annotation_path.is_file():
        return None
    rendered_payload = _load_json_object(entry.task_result_path)
    trace_path = _normalize_video_asset_path(project_root, str(rendered_payload.get("final_trace_path") or ""))
    if not isinstance(trace_path, Path) or not trace_path.is_file():
        return None
    trace_payload = _load_trace_payload(trace_path)
    annotation_payload = _load_json_object(entry.annotation_path)
    if not annotation_payload:
        return None
    video_attachment = _trace_video_attachment_metadata(trace_payload, view=view)
    source_path = _entry_source_video_path(entry, view=view)
    source_fps, source_frame_count = _video_file_metrics(source_path) if isinstance(source_path, Path) else (None, None)
    fps = float(source_fps or video_attachment.get("fps") or 0.0)
    if fps <= 0:
        fps = 10.0
    frame_count = int(source_frame_count or video_attachment.get("frame_count") or 0)
    if frame_count <= 0:
        return None
    total_timesteps = 0
    timesteps = trace_payload.get("timesteps")
    if isinstance(timesteps, list):
        total_timesteps = len(timesteps)
    if total_timesteps <= 0:
        total_timesteps = int(video_attachment.get("dense_timestep_count") or 0)
    if total_timesteps <= 0:
        total_timesteps = frame_count
    min_output_frames = max(1, int(round(MIN_ANNOTATION_DISPLAY_SECONDS * fps)))
    annotations = annotation_payload.get("pattern_annotations")
    if not isinstance(annotations, dict):
        return None
    patterns = annotations.get("patterns")
    if not isinstance(patterns, list) or not patterns:
        return None
    filtered_patterns = filter_annotation_patterns_for_env(entry.env_id, patterns)
    if not filtered_patterns:
        return None
    if entry.env_id == "pooltool":
        filtered_patterns = [
            item
            for item in filtered_patterns
            if _normalized_label_key(item.get("label")) != "ball pocketed"
            or _pooltool_ball_pocketed_matches_trace(item, trace_payload)
        ]
        if not filtered_patterns:
            return None
    selected_keys = {_normalized_label_key(label) for label in selected_labels if _normalized_label_key(label)}
    events: list[dict[str, Any]] = []
    for item in filtered_patterns:
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        if selected_keys and _normalized_label_key(label) not in selected_keys:
            continue
        start_timestep = int(item.get("start_timestep") or 0)
        end_timestep = int(item.get("end_timestep") or start_timestep)
        if entry.env_id == "pooltool" and _normalized_label_key(label) == "ball pocketed":
            pocketed_timestep = _pooltool_ball_pocketed_timestep_from_trace(item, trace_payload)
            if pocketed_timestep is not None:
                start_timestep = pocketed_timestep
                end_timestep = pocketed_timestep
        if total_timesteps <= 1 or frame_count <= 1:
            start_frame = 0
            end_frame = 0
        else:
            start_frame = int(round((start_timestep * max(0, frame_count - 1)) / max(1, total_timesteps - 1)))
            end_frame = int(round((max(start_timestep, end_timestep) * max(0, frame_count - 1)) / max(1, total_timesteps - 1)))
        event_specs = _pattern_event_specs(
            start_timestep=start_timestep,
            end_timestep=end_timestep,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        color = _rgb_hex(_label_color(label))
        for event_kind, event_frame in event_specs:
            if not _annotation_event_passes_custom_filters(
                env_id=entry.env_id,
                item=item,
                event_kind=event_kind,
                start_timestep=start_timestep,
                end_timestep=end_timestep,
                total_timesteps=total_timesteps,
            ):
                continue
            event_end_frame = min(max(0, frame_count - 1), event_frame + max(0, min_output_frames - 1))
            points = _pattern_event_points(item, event_kind=event_kind)
            projected = [
                _project_video_point(
                    env_id=entry.env_id,
                    point=point,
                    width=1,
                    height=1,
                    trace_payload=trace_payload,
                    view=view,
                )
                for point in points
            ]
            normalized_points = [
                [float(point[0]), float(point[1])]
                for point in projected
                if point is not None
            ]
            metadata = item.get("metadata")
            confidence = 1.0
            if isinstance(metadata, dict):
                try:
                    confidence = float(metadata.get("confidence") or 0.0)
                except (TypeError, ValueError):
                    confidence = 0.0
            events.append(
                {
                    "label": label,
                    "event_kind": event_kind,
                    "color": color,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "start_time": event_frame / fps,
                    "end_time": event_end_frame / fps,
                    "points": normalized_points,
                }
            )
    if not events:
        return None
    return {
        "env_id": entry.env_id,
        "task_id": entry.task_id,
        "view": str(view or "").strip().lower(),
        "fps": fps,
        "frame_count": frame_count,
        "events": events,
    }


def _overlay_active_events(payload: dict[str, Any], now_seconds: float, confidence_threshold: float) -> list[dict[str, Any]]:
    events = payload.get("events")
    if not isinstance(events, list):
        return []
    active: list[dict[str, Any]] = []
    threshold = max(0.0, min(1.0, float(confidence_threshold or 0.0)))
    for event in events:
        if not isinstance(event, dict):
            continue
        try:
            confidence = float(event.get("confidence") if event.get("confidence") is not None else 1.0)
            start_time = float(event.get("start_time") or 0.0)
            end_time = float(event.get("end_time") or 0.0)
        except (TypeError, ValueError):
            continue
        if confidence < threshold:
            continue
        if now_seconds < start_time or now_seconds > end_time:
            continue
        active.append(event)
    return active


def _overlay_font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _clamp_box(left: float, top: float, box_width: float, box_height: float, width: int, height: int) -> tuple[float, float, float, float]:
    max_left = max(0.0, width - box_width)
    max_top = max(0.0, height - box_height)
    next_left = max(0.0, min(max_left, left))
    next_top = max(0.0, min(max_top, top))
    return (next_left, next_top, next_left + box_width, next_top + box_height)


def _boxes_overlap(left: tuple[float, float, float, float], right: tuple[float, float, float, float], padding: float = 0.0) -> bool:
    return not (
        left[2] + padding <= right[0]
        or left[0] >= right[2] + padding
        or left[3] + padding <= right[1]
        or left[1] >= right[3] + padding
    )


def _point_penalty(box: tuple[float, float, float, float], point_boxes: list[tuple[float, float, float, float]]) -> int:
    penalty = 0
    for point_box in point_boxes:
        if _boxes_overlap(box, point_box, 4.0):
            penalty += 1
    return penalty


def _select_label_box(
    *,
    anchor_x: float,
    anchor_y: float,
    radius: float,
    box_width: float,
    box_height: float,
    width: int,
    height: int,
    occupied_label_boxes: list[tuple[float, float, float, float]],
    point_boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    candidates = [
        _clamp_box(anchor_x + radius + 6.0, anchor_y - radius - 4.0, box_width, box_height, width, height),
        _clamp_box(anchor_x + radius + 6.0, anchor_y + radius + 6.0, box_width, box_height, width, height),
        _clamp_box(anchor_x - box_width - radius - 6.0, anchor_y - radius - 4.0, box_width, box_height, width, height),
        _clamp_box(anchor_x - box_width - radius - 6.0, anchor_y + radius + 6.0, box_width, box_height, width, height),
        _clamp_box(anchor_x - (box_width / 2.0), anchor_y - box_height - radius - 8.0, box_width, box_height, width, height),
        _clamp_box(anchor_x - (box_width / 2.0), anchor_y + radius + 8.0, box_width, box_height, width, height),
    ]
    for box in candidates:
        blocked = False
        for occupied in occupied_label_boxes:
            if _boxes_overlap(box, occupied, 4.0):
                blocked = True
                break
        if not blocked and _point_penalty(box, point_boxes) == 0:
            return box
    best_box = candidates[0]
    best_penalty = float("inf")
    for box in candidates:
        penalty = sum(2 for occupied in occupied_label_boxes if _boxes_overlap(box, occupied, 4.0))
        penalty += _point_penalty(box, point_boxes)
        if penalty < best_penalty:
            best_penalty = penalty
            best_box = box
    return best_box


def _draw_overlay_frame(frame: Any, payload: dict[str, Any], now_seconds: float, confidence_threshold: float) -> Any:
    image = Image.fromarray(frame).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    radius = max(2, int(round(max(3, min(7, round(min(width, height) * 0.007))) * 0.75)))
    font = _overlay_font(20)
    legend_entries: dict[str, tuple[int, int, int]] = {}
    all_events = payload.get("events")
    if isinstance(all_events, list):
        for event in all_events:
            if not isinstance(event, dict):
                continue
            try:
                confidence = float(event.get("confidence") if event.get("confidence") is not None else 1.0)
            except (TypeError, ValueError):
                confidence = 1.0
            if confidence < confidence_threshold:
                continue
            try:
                rgba = ImageColor.getrgb(str(event.get("color") or "#ffffff"))
            except ValueError:
                rgba = (255, 255, 255)
            label = str(event.get("label") or "pattern")
            if label and label not in legend_entries:
                legend_entries[label] = rgba
    for event in _overlay_active_events(payload, now_seconds, confidence_threshold):
        try:
            rgba = ImageColor.getrgb(str(event.get("color") or "#ffffff"))
        except ValueError:
            rgba = (255, 255, 255)
        points = event.get("points")
        if not isinstance(points, list):
            continue
        for point in points:
            if not isinstance(point, list) or len(point) < 2:
                continue
            try:
                x = float(point[0]) * width
                y = float(point[1]) * height
            except (TypeError, ValueError):
                continue
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(rgba[0], rgba[1], rgba[2], 0x58),
                outline=(rgba[0], rgba[1], rgba[2], 255),
                width=max(1, radius // 4),
            )
    if legend_entries:
        swatch_size = 14
        row_height = 26
        legend_width = 0
        for label in legend_entries:
            text_box = draw.textbbox((0, 0), label, font=font)
            legend_width = max(legend_width, int(text_box[2] - text_box[0]))
        card_width = legend_width + swatch_size + 36
        card_height = (len(legend_entries) * row_height) + 16
        card_left = max(12, width - card_width - 12)
        card_top = max(12, height - card_height - 12)
        draw.rounded_rectangle(
            (card_left, card_top, card_left + card_width, card_top + card_height),
            radius=6,
            fill=(17, 17, 17, 184),
        )
        for index, (label, rgba) in enumerate(legend_entries.items()):
            row_top = card_top + 6 + (index * row_height)
            draw.rectangle(
                (card_left + 8, row_top + 6, card_left + 8 + swatch_size, row_top + 6 + swatch_size),
                fill=(rgba[0], rgba[1], rgba[2], 255),
            )
            draw.text((card_left + 8 + swatch_size + 10, row_top + 1), label, fill=(255, 255, 255, 255), font=font)
    return image.convert("RGB")


def export_annotation_overlay_video(
    *,
    project_root: Path,
    entry: VideoRenderResultEntry,
    view: str,
    selected_labels: list[str],
    confidence_threshold: float,
) -> Path:
    payload = build_annotation_overlay_payload(
        project_root=project_root,
        entry=entry,
        view=view,
        selected_labels=selected_labels,
    )
    if not payload:
        raise FileNotFoundError("No overlay payload available.")
    source_path = (
        entry.overhead_video_path
        if str(view or "").strip().lower() == "overhead" and isinstance(entry.overhead_video_path, Path) and entry.overhead_video_path.is_file()
        else entry.video_path
    )
    if not isinstance(source_path, Path) or not source_path.is_file():
        raise FileNotFoundError("Source video not found.")
    threshold = max(0.0, min(1.0, float(confidence_threshold or 0.0)))
    export_root = _overlay_export_root(project_root)
    export_root.mkdir(parents=True, exist_ok=True)
    payload_hash = hashlib.sha1(
        json.dumps(
            {
                "task_result": _project_relpath(project_root, entry.task_result_path),
                "view": str(view or "").strip().lower(),
                "threshold": round(threshold, 4),
                "selected_labels": list(selected_labels),
                "annotation_mtime": entry.annotation_path.stat().st_mtime_ns if isinstance(entry.annotation_path, Path) and entry.annotation_path.is_file() else 0,
                "source_mtime": source_path.stat().st_mtime_ns,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    export_path = export_root / f"{entry.env_id}-{_slugify_label(entry.task_id)}-{payload_hash[:12]}.mp4"
    if export_path.is_file() and export_path.stat().st_size > 0:
        return export_path
    reader = imageio.get_reader(str(source_path))
    try:
        meta = reader.get_meta_data()
        fps = float(meta.get("fps") or payload.get("fps") or 30.0)
        writer = imageio.get_writer(str(export_path), fps=max(1.0, fps), codec="libx264", quality=8)
        try:
            for index, frame in enumerate(reader):
                now_seconds = index / max(1.0, fps)
                composited = _draw_overlay_frame(frame, payload, now_seconds, threshold)
                writer.append_data(np.asarray(composited))
        finally:
            writer.close()
    finally:
        reader.close()
    return export_path


def list_annotation_video_statuses(project_root: Path) -> dict[str, dict[str, Any]]:
    state = _load_annotation_state(project_root)
    statuses: dict[str, dict[str, Any]] = {}
    changed = False
    for env_id in SUPPORTED_VIDEO_RENDER_ENVS:
        current = dict(state.get(env_id, {}))
        if current.get("status") == "running" and not _pid_is_alive(current.get("pid")):
            current["status"] = "failed"
            current["finished_at"] = current.get("finished_at") or _isoformat()
            current["error"] = str(current.get("error") or "Annotation renderer exited unexpectedly.").strip()
            current["updated_at"] = _isoformat()
            state[env_id] = current
            changed = True
        selection = load_annotation_video_selection(project_root, env_id)
        log_path = _annotation_log_path(project_root, env_id)
        log_excerpt = ""
        if log_path.is_file():
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                log_excerpt = "\n".join(reversed(lines[-24:]))
            except OSError:
                log_excerpt = ""
        statuses[env_id] = {
            "env_id": env_id,
            "status": str(current.get("status") or "idle"),
            "pid": current.get("pid"),
            "planned_tasks": int(current.get("planned_tasks") or 0),
            "completed_tasks": int(current.get("completed_tasks") or 0),
            "failed_tasks": int(current.get("failed_tasks") or 0),
            "current_task_id": str(current.get("current_task_id") or "").strip(),
            "current_view": str(current.get("current_view") or "").strip(),
            "current_video_frame": int(current.get("current_video_frame") or 0),
            "current_video_total_frames": int(current.get("current_video_total_frames") or 0),
            "started_at": current.get("started_at"),
            "updated_at": current.get("updated_at"),
            "finished_at": current.get("finished_at"),
            "error": str(current.get("error") or "").strip(),
            "log_relpath": _project_relpath(project_root, log_path),
            "log_excerpt": log_excerpt,
            "selected_labels": list(selection.get("selected_labels") or []),
            "selected_count": int(selection.get("selected_count") or 0),
            "available_labels": list(selection.get("available_labels") or []),
        }
    if changed:
        _write_annotation_state(project_root, state)
    return statuses


def start_annotation_video_process(project_root: Path, env_id: str, *, selected_labels: list[str] | None = None) -> dict[str, Any]:
    env_key = str(env_id or "").strip().lower()
    if env_key not in SUPPORTED_VIDEO_RENDER_ENVS:
        raise ValueError(f"Unknown environment {env_key}")
    if selected_labels is not None:
        save_annotation_video_selection(project_root, env_key, selected_labels)
    statuses = list_annotation_video_statuses(project_root)
    current = statuses.get(env_key, {})
    if current.get("status") == "running" and _pid_is_alive(current.get("pid")):
        return current
    entries = select_video_showcase_entries(list_video_render_results(project_root).get(env_key, []))
    if not entries:
        return _merge_annotation_env_state(
            project_root,
            env_key,
            {
                "status": "succeeded",
                "pid": None,
                "planned_tasks": 0,
                "completed_tasks": 0,
                "failed_tasks": 0,
                "current_task_id": "",
                "current_view": "",
                "current_video_frame": 0,
                "current_video_total_frames": 0,
                "started_at": _isoformat(),
                "finished_at": _isoformat(),
                "error": "",
            },
        )
    log_path = _annotation_log_path(project_root, env_key)
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
    command = [
        str(env_python),
        "-m",
        "pattern_server.video_annotation_worker",
        "--project-root",
        str(project_root),
        "--env",
        env_key,
    ]
    with log_path.open("wb") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(client_root),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=process_env,
        )
    return _merge_annotation_env_state(
        project_root,
        env_key,
        {
            "status": "running",
            "pid": process.pid,
            "planned_tasks": len(entries),
            "completed_tasks": 0,
            "failed_tasks": 0,
            "current_task_id": "",
            "current_view": "",
            "current_video_frame": 0,
            "current_video_total_frames": 0,
            "started_at": _isoformat(),
            "finished_at": None,
            "error": "",
        },
    )


def cancel_annotation_video_process(project_root: Path, env_id: str) -> dict[str, Any]:
    env_key = str(env_id or "").strip().lower()
    if env_key not in SUPPORTED_VIDEO_RENDER_ENVS:
        raise ValueError(f"Unknown environment {env_key}")
    current = dict(_load_annotation_state(project_root).get(env_key, {}))
    pid = current.get("pid")
    if _pid_is_alive(pid):
        try:
            os.killpg(int(pid), signal.SIGTERM)
        except Exception:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except Exception:
                pass
    return _merge_annotation_env_state(
        project_root,
        env_key,
        {
            "status": "cancelled",
            "pid": None,
            "finished_at": _isoformat(),
            "current_task_id": "",
            "current_view": "",
            "current_video_frame": 0,
            "current_video_total_frames": 0,
            "error": str(current.get("error") or "").strip(),
        },
    )


def summarize_video_render_inventory(project_root: Path) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for env_id in SUPPORTED_VIDEO_RENDER_ENVS:
        env_root = project_root / "data" / "evaluations" / env_id
        successful = 0
        rendered = 0
        if env_root.is_dir():
            for source_run_dir in sorted(env_root.glob("*image-only*")):
                if not source_run_dir.is_dir() or ".bak." in source_run_dir.name:
                    continue
                output_run_dir = env_root / f"video-render-{source_run_dir.name}"
                for source_result_path in sorted(source_run_dir.glob("tasks/*/*/task_result.json")):
                    try:
                        payload = json.loads(source_result_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    if not isinstance(payload, dict) or payload.get("success") is not True:
                        continue
                    successful += 1
                    case_id = source_result_path.parent.parent.name
                    task_dir_name = source_result_path.parent.name
                    output_task_dir = output_run_dir / "tasks" / case_id / task_dir_name
                    if _task_has_video(output_task_dir):
                        rendered += 1
        summary[env_id] = {
            "successful_tasks": successful,
            "rendered_tasks": rendered,
            "pending_tasks": max(0, successful - rendered),
        }
    return summary


def list_video_render_statuses(project_root: Path) -> dict[str, dict[str, Any]]:
    state = _load_state(project_root)
    inventory = summarize_video_render_inventory(project_root)
    statuses: dict[str, dict[str, Any]] = {}
    changed = False
    for env_id in SUPPORTED_VIDEO_RENDER_ENVS:
        current = dict(state.get(env_id, {}))
        if current.get("status") == "running" and not _pid_is_alive(current.get("pid")):
            current["status"] = "failed"
            current["finished_at"] = current.get("finished_at") or _isoformat()
            current["error"] = str(current.get("error") or "Renderer exited unexpectedly.").strip()
            current["updated_at"] = _isoformat()
            state[env_id] = current
            changed = True
        merged = {
            "env_id": env_id,
            "status": str(current.get("status") or "idle"),
            "pid": current.get("pid"),
            "run_label": str(current.get("run_label") or "").strip(),
            "target_count": _normalize_target_count(current.get("target_count")),
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
            **inventory.get(env_id, {"successful_tasks": 0, "rendered_tasks": 0, "pending_tasks": 0}),
        }
        merged["processed_tasks"] = merged["completed_tasks"] + merged["failed_tasks"]
        statuses[env_id] = merged
    if changed:
        _write_state(project_root, state)
    return statuses


def start_video_render_process(project_root: Path, env_id: str, *, target_count: int = DEFAULT_VIDEO_RENDER_TARGET) -> dict[str, Any]:
    env_key = str(env_id or "").strip().lower()
    if env_key not in SUPPORTED_VIDEO_RENDER_ENVS:
        raise ValueError(f"Unknown environment {env_key}")
    normalized_target = _normalize_target_count(target_count)
    statuses = list_video_render_statuses(project_root)
    current = statuses.get(env_key, {})
    if current.get("status") == "running" and _pid_is_alive(current.get("pid")):
        return current
    successful_tasks = int(current.get("successful_tasks") or 0)
    rendered_tasks = int(current.get("rendered_tasks") or 0)
    pending_tasks = int(current.get("pending_tasks") or 0)
    desired_total = min(normalized_target, successful_tasks) if successful_tasks > 0 else normalized_target
    outstanding_needed = max(0, desired_total - rendered_tasks)
    planned_tasks = min(pending_tasks, outstanding_needed)
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
                "run_label": f"video-render-{env_key}-noop",
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
    command = [
        str(env_python),
        "-m",
        "pattern_server.video_render_worker",
        "--project-root",
        str(project_root),
        "--env",
        env_key,
        "--target-count",
        str(normalized_target),
    ]
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
            "run_label": f"video-render-{env_key}-{_utc_now().strftime('%Y%m%d-%H%M%S')}",
            "started_at": started_at,
            "finished_at": None,
            "error": "",
            "log_relpath": _project_relpath(project_root, log_path),
        },
    )


def _copy_source_payload(payload: dict[str, Any], *, output_task_dir: Path, trace_relpath: str | None) -> dict[str, Any]:
    copied = copy.deepcopy(payload)
    copied["output_root"] = str(output_task_dir)
    copied["video_render_only"] = True
    copied["video_rendered_at"] = _isoformat()
    if trace_relpath is not None:
        trace_path = output_task_dir / trace_relpath
        copied["final_trace_path"] = str(trace_path)
        media_paths = resolve_trace_media_paths(trace_path)
        if media_paths.default_video_path is not None:
            copied["simulation_video_path"] = str(media_paths.default_video_path)
        if media_paths.overhead_video_path is not None:
            copied["simulation_video_overhead_path"] = str(media_paths.overhead_video_path)
        if media_paths.default_initial_image_path is not None:
            copied["initial_image_path"] = str(media_paths.default_initial_image_path)
        if media_paths.overhead_initial_image_path is not None:
            copied["initial_image_overhead_path"] = str(media_paths.overhead_initial_image_path)
    return copied


def _pooltool_ball_color(ball_id: str) -> tuple[int, int, int]:
    palette = {
        "cue": (245, 245, 245),
        "1": (245, 209, 66),
        "2": (54, 97, 214),
        "3": (199, 47, 47),
        "4": (106, 61, 186),
        "5": (232, 134, 44),
        "6": (54, 140, 73),
        "7": (120, 33, 24),
        "8": (28, 28, 28),
        "9": (245, 209, 66),
    }
    return palette.get(str(ball_id), (200, 200, 200))


def _normalize_trace_attachment_path(trace_path: Path, raw_path: str | None) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_file():
        return candidate.resolve()
    prefixes = (
        ("/opt/pattern-learning-stack/data/evaluations", SERVER_PROJECT_ROOT / "data" / "evaluations"),
        ("/var/opt/pattern-learning-stack/data/evaluations", SERVER_PROJECT_ROOT / "data" / "evaluations"),
        ("/opt/pattern-learning-stack/data", SERVER_PROJECT_ROOT / "data"),
        ("/var/opt/pattern-learning-stack/data", SERVER_PROJECT_ROOT / "data"),
        ("/home/sean/Services/Work/pattern-learning-server/data/evaluations", SERVER_PROJECT_ROOT / "data" / "evaluations"),
        ("/var/home/sean/Services/Work/pattern-learning-server/data/evaluations", SERVER_PROJECT_ROOT / "data" / "evaluations"),
        ("/home/sean/Services/Work/pattern-learning-server/data", SERVER_PROJECT_ROOT / "data"),
        ("/var/home/sean/Services/Work/pattern-learning-server/data", SERVER_PROJECT_ROOT / "data"),
    )
    for prefix, replacement in prefixes:
        if text == prefix or text.startswith(prefix + "/"):
            suffix = text[len(prefix) :].lstrip("/")
            remapped = replacement / suffix
            if remapped.is_file():
                return remapped.resolve()
    candidate = (trace_path.parent / text).resolve()
    if candidate.is_file():
        return candidate
    return None


def resolve_trace_media_paths(trace_path: Path | None) -> TraceMediaPaths:
    if not isinstance(trace_path, Path) or not trace_path.is_file():
        return TraceMediaPaths(None, None, None, None)
    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return TraceMediaPaths(None, None, None, None)
    attachments = payload.get("attachments")
    if not isinstance(attachments, list):
        attachments = payload.get("assets")
    if not isinstance(attachments, list):
        return TraceMediaPaths(None, None, None, None)

    default_video_path: Path | None = None
    overhead_video_path: Path | None = None
    default_initial_image_path: Path | None = None
    overhead_initial_image_path: Path | None = None

    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        kind = str(attachment.get("kind") or "").strip().lower()
        metadata = attachment.get("metadata")
        meta = metadata if isinstance(metadata, dict) else {}
        view = str(meta.get("view") or "").strip().lower()
        attachment_path = _normalize_trace_attachment_path(trace_path, str(attachment.get("path") or ""))
        if attachment_path is None:
            continue
        if kind == "simulation_video":
            if view == "overhead":
                overhead_video_path = overhead_video_path or attachment_path
            elif view == "askew":
                default_video_path = default_video_path or attachment_path
            else:
                default_video_path = default_video_path or attachment_path
        elif kind == "initial_scene":
            if view == "overhead":
                overhead_initial_image_path = overhead_initial_image_path or attachment_path
            elif view == "askew":
                default_initial_image_path = default_initial_image_path or attachment_path
            else:
                default_initial_image_path = default_initial_image_path or attachment_path

    default_video_path = default_video_path or overhead_video_path
    default_initial_image_path = default_initial_image_path or overhead_initial_image_path
    return TraceMediaPaths(
        default_video_path=default_video_path,
        overhead_video_path=overhead_video_path,
        default_initial_image_path=default_initial_image_path,
        overhead_initial_image_path=overhead_initial_image_path,
    )


def _upsert_video_render_result(project_root: Path, task: VideoRenderTask, rendered_payload: dict[str, Any]) -> None:
    trace_path = _normalize_video_asset_path(project_root, str(rendered_payload.get("final_trace_path") or ""))
    media_paths = resolve_trace_media_paths(trace_path)
    if media_paths.default_video_path is None:
        return
    annotation_path, annotation_preview, library_path = _ensure_best_supervised_annotation(
        project_root,
        task.env_id,
        trace_path=trace_path,
        task_dir=task.output_task_dir,
    )
    results = _load_results(project_root)
    env_results = dict(results.get(task.env_id, {}))
    key = f"{task.source_run_id}:{task.task_id}"
    updated_at = str(rendered_payload.get("video_rendered_at") or _isoformat()).strip() or _isoformat()
    annotated_video_path, annotated_overhead_video_path, _meta_path = _annotated_video_paths(task.output_task_dir)
    env_results[key] = {
        "env_id": task.env_id,
        "task_id": task.task_id,
        "source_run_id": task.source_run_id,
        "output_run_id": task.output_run_dir.name,
        "outcome_kind": _result_outcome_kind(task.payload) or "success",
        "task_result_path": _project_relpath(project_root, task.output_result_path),
        "video_path": _project_relpath(project_root, media_paths.default_video_path),
        "overhead_video_path": (
            _project_relpath(project_root, media_paths.overhead_video_path)
            if media_paths.overhead_video_path is not None
            else ""
        ),
        "annotated_video_path": (
            _project_relpath(project_root, annotated_video_path)
            if annotated_video_path.is_file()
            else ""
        ),
        "annotated_overhead_video_path": (
            _project_relpath(project_root, annotated_overhead_video_path)
            if annotated_overhead_video_path.is_file()
            else ""
        ),
        "annotation_path": (
            _project_relpath(project_root, annotation_path)
            if annotation_path is not None
            else ""
        ),
        "annotation_preview": annotation_preview,
        "supervised_library_path": (
            _project_relpath(project_root, library_path)
            if library_path is not None
            else ""
        ),
        "updated_at": updated_at,
    }
    results[task.env_id] = env_results
    _write_results(project_root, results)


def list_video_render_results(
    project_root: Path,
    *,
    _allow_rebuild: bool = True,
) -> dict[str, list[VideoRenderResultEntry]]:
    raw_results = _load_results(project_root)
    parsed: dict[str, list[VideoRenderResultEntry]] = {env_id: [] for env_id in SUPPORTED_VIDEO_RENDER_ENVS}
    for env_id, env_results in raw_results.items():
        if env_id not in parsed:
            continue
        for item in env_results.values():
            try:
                task_result_path = (project_root / str(item.get("task_result_path") or "")).resolve()
                task_dir = task_result_path.parent
                video_path = (project_root / str(item.get("video_path") or "")).resolve()
                if not video_path.is_file():
                    fallback_trace_path = _normalize_video_asset_path(project_root, str(_load_json_object(task_result_path).get("final_trace_path") or ""))
                    fallback_media_paths = resolve_trace_media_paths(fallback_trace_path)
                    if isinstance(fallback_media_paths.default_video_path, Path) and fallback_media_paths.default_video_path.is_file():
                        video_path = fallback_media_paths.default_video_path.resolve()
                    else:
                        continue
                trace_path = _normalize_video_asset_path(project_root, str(_load_json_object(task_result_path).get("final_trace_path") or ""))
                overhead_raw = str(item.get("overhead_video_path") or "").strip()
                overhead_video_path = (project_root / overhead_raw).resolve() if overhead_raw else None
                annotated_raw = str(item.get("annotated_video_path") or "").strip()
                annotated_video_path = (project_root / annotated_raw).resolve() if annotated_raw else None
                annotated_overhead_raw = str(item.get("annotated_overhead_video_path") or "").strip()
                annotated_overhead_video_path = (project_root / annotated_overhead_raw).resolve() if annotated_overhead_raw else None
                annotation_raw = str(item.get("annotation_path") or "").strip()
                annotation_path = (project_root / annotation_raw).resolve() if annotation_raw else None
                library_raw = str(item.get("supervised_library_path") or "").strip()
                library_path = (project_root / library_raw).resolve() if library_raw else None
                updated_at = datetime.fromisoformat(str(item.get("updated_at") or "").replace("Z", "+00:00"))
            except Exception:
                continue
            if not (isinstance(annotation_path, Path) and annotation_path.is_file()):
                derived_annotation_path = _video_annotation_path(task_dir)
                if derived_annotation_path.is_file():
                    annotation_path = derived_annotation_path.resolve()
            if isinstance(annotation_path, Path) and annotation_path.is_file():
                try:
                    annotation_payload = json.loads(annotation_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    annotation_payload = None
                if not (
                    isinstance(annotation_payload, dict)
                    and isinstance(library_path, Path)
                    and library_path.exists()
                    and _annotation_matches_library(annotation_payload, library_path, trace_path=trace_path)
                ):
                    annotation_path = None
                    annotation_payload = None
            else:
                annotation_payload = None
            annotation_preview = (
                _format_annotation_preview(annotation_payload, env_id=env_id)
                if isinstance(annotation_payload, dict) and isinstance(annotation_path, Path) and annotation_path.is_file()
                else ""
            )
            overlay_is_current = _annotated_video_meta_matches_current_state(
                project_root=project_root,
                env_id=env_id,
                task_dir=task_dir,
                annotation_path=annotation_path if isinstance(annotation_path, Path) and annotation_path.is_file() else None,
                trace_path=trace_path,
            )
            derived_annotated_video_path, derived_annotated_overhead_video_path, _derived_meta_path = _annotated_video_paths(task_dir)
            if overlay_is_current and not (isinstance(annotated_video_path, Path) and annotated_video_path.is_file()) and derived_annotated_video_path.is_file():
                annotated_video_path = derived_annotated_video_path.resolve()
            if overlay_is_current and not (isinstance(annotated_overhead_video_path, Path) and annotated_overhead_video_path.is_file()) and derived_annotated_overhead_video_path.is_file():
                annotated_overhead_video_path = derived_annotated_overhead_video_path.resolve()
            if not overlay_is_current:
                annotated_video_path = None
                annotated_overhead_video_path = None
            parsed[env_id].append(
                VideoRenderResultEntry(
                    env_id=env_id,
                    task_id=str(item.get("task_id") or "").strip(),
                    source_run_id=str(item.get("source_run_id") or "").strip(),
                    output_run_id=str(item.get("output_run_id") or "").strip(),
                    outcome_kind=str(item.get("outcome_kind") or "success").strip().lower() or "success",
                    task_result_path=task_result_path,
                    video_path=video_path,
                    overhead_video_path=overhead_video_path if isinstance(overhead_video_path, Path) and overhead_video_path.is_file() else None,
                    annotated_video_path=annotated_video_path if isinstance(annotated_video_path, Path) and annotated_video_path.is_file() else None,
                    annotated_overhead_video_path=(
                        annotated_overhead_video_path
                        if isinstance(annotated_overhead_video_path, Path) and annotated_overhead_video_path.is_file()
                        else None
                    ),
                    annotation_path=annotation_path if isinstance(annotation_path, Path) and annotation_path.is_file() else None,
                    annotation_preview=annotation_preview,
                    supervised_library_path=library_path if isinstance(library_path, Path) and library_path.exists() else None,
                    updated_at=updated_at,
                )
            )
        parsed[env_id] = _merge_result_entries(parsed[env_id])
    if _allow_rebuild:
        for env_id in SUPPORTED_VIDEO_RENDER_ENVS:
            if parsed.get(env_id):
                continue
            env_root = project_root / "data" / "evaluations" / env_id
            if not env_root.is_dir():
                continue
            if any(env_root.glob("video-render-*/tasks/*/*/task_result.json")):
                parsed[env_id] = _scan_video_render_results_for_env(project_root, env_id)
    return parsed


def _scan_video_render_results_for_env(project_root: Path, env_id: str) -> list[VideoRenderResultEntry]:
    env_root = project_root / "data" / "evaluations" / env_id
    if not env_root.is_dir():
        return []
    entries: list[VideoRenderResultEntry] = []
    for task_result_path in sorted(env_root.glob("video-render-*/tasks/*/*/task_result.json")):
        try:
            payload = json.loads(task_result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        trace_path = _normalize_video_asset_path(project_root, str(payload.get("final_trace_path") or ""))
        media_paths = resolve_trace_media_paths(trace_path)
        if media_paths.default_video_path is None:
            continue
        annotation_path, annotation_preview, library_path = _load_existing_best_supervised_annotation(
            project_root,
            env_id,
            task_dir=task_result_path.parent,
            trace_path=trace_path,
        )
        source_run_id = task_result_path.parents[3].name.removeprefix("video-render-")
        task_id = str(payload.get("task_id") or task_result_path.parent.name).strip() or task_result_path.parent.name
        annotated_video_path, annotated_overhead_video_path, _meta_path = _annotated_video_paths(task_result_path.parent)
        overlay_is_current = _annotated_video_meta_matches_current_state(
            project_root=project_root,
            env_id=env_id,
            task_dir=task_result_path.parent,
            annotation_path=annotation_path if isinstance(annotation_path, Path) and annotation_path.is_file() else None,
            trace_path=trace_path,
        )
        updated_raw = payload.get("video_rendered_at")
        if isinstance(updated_raw, str) and updated_raw.strip():
            try:
                updated_at = datetime.fromisoformat(updated_raw.replace("Z", "+00:00"))
            except ValueError:
                updated_at = datetime.fromtimestamp(task_result_path.stat().st_mtime, tz=timezone.utc)
        else:
            updated_at = datetime.fromtimestamp(task_result_path.stat().st_mtime, tz=timezone.utc)
        entries.append(
            VideoRenderResultEntry(
                env_id=env_id,
                task_id=task_id,
                source_run_id=source_run_id,
                output_run_id=task_result_path.parents[3].name,
                outcome_kind=str(payload.get("outcome_kind") or "success").strip().lower() or "success",
                task_result_path=task_result_path.resolve(),
                video_path=media_paths.default_video_path.resolve(),
                overhead_video_path=(
                    media_paths.overhead_video_path.resolve()
                    if isinstance(media_paths.overhead_video_path, Path) and media_paths.overhead_video_path.is_file()
                    else None
                ),
                annotated_video_path=(
                    annotated_video_path.resolve()
                    if overlay_is_current and annotated_video_path.is_file()
                    else None
                ),
                annotated_overhead_video_path=(
                    annotated_overhead_video_path.resolve()
                    if overlay_is_current and annotated_overhead_video_path.is_file()
                    else None
                ),
                annotation_path=annotation_path.resolve() if isinstance(annotation_path, Path) and annotation_path.is_file() else None,
                annotation_preview=annotation_preview,
                supervised_library_path=library_path.resolve() if isinstance(library_path, Path) and library_path.exists() else None,
                updated_at=updated_at,
            )
        )
    entries.sort(key=lambda entry: (entry.updated_at, entry.task_id), reverse=True)
    return entries


def _scan_source_video_results_for_env(project_root: Path, env_id: str) -> list[VideoRenderResultEntry]:
    env_root = project_root / "data" / "evaluations" / env_id
    if not env_root.is_dir():
        return []
    entries: list[VideoRenderResultEntry] = []
    target_counts = {"success": 15, "failure": 5}
    seen_tasks: dict[str, set[str]] = {"success": set(), "failure": set()}
    gathered = {"success": 0, "failure": 0}
    for source_run_dir in sorted(env_root.glob("*image-only*"), reverse=True):
        if not source_run_dir.is_dir() or ".bak." in source_run_dir.name:
            continue
        for task_result_path in sorted(source_run_dir.glob("tasks/*/*/task_result.json"), reverse=True):
            payload = _load_json_object(task_result_path)
            if not payload:
                continue
            outcome_kind = _result_outcome_kind(payload)
            if outcome_kind is None:
                continue
            if gathered[outcome_kind] >= target_counts[outcome_kind]:
                continue
            task_id = str(payload.get("task_id") or task_result_path.parent.name).strip() or task_result_path.parent.name
            if task_id in seen_tasks[outcome_kind]:
                continue
            selected = _select_trace_record_candidate(project_root, payload)
            if selected is None:
                continue
            _record, _trace_path, media_paths = selected
            if media_paths.default_video_path is None:
                continue
            annotation_file = _video_annotation_path(task_result_path.parent)
            annotation_path = annotation_file.resolve() if annotation_file.is_file() else None
            annotation_payload = _load_json_object(annotation_file) if annotation_file.is_file() else {}
            annotation_preview = _format_annotation_preview(annotation_payload, env_id=env_id) if annotation_payload else ""
            library_path = None
            annotated_video_path, annotated_overhead_video_path, _meta_path = _annotated_video_paths(task_result_path.parent)
            updated_at = datetime.fromtimestamp(task_result_path.stat().st_mtime, tz=timezone.utc)
            entries.append(
                VideoRenderResultEntry(
                    env_id=env_id,
                    task_id=task_id,
                    source_run_id=source_run_dir.name,
                    output_run_id=source_run_dir.name,
                    outcome_kind=outcome_kind,
                    task_result_path=task_result_path.resolve(),
                    video_path=media_paths.default_video_path.resolve(),
                    overhead_video_path=(
                        media_paths.overhead_video_path.resolve()
                        if isinstance(media_paths.overhead_video_path, Path) and media_paths.overhead_video_path.is_file()
                        else None
                    ),
                    annotated_video_path=annotated_video_path.resolve() if annotated_video_path.is_file() else None,
                    annotated_overhead_video_path=(
                        annotated_overhead_video_path.resolve()
                        if annotated_overhead_video_path.is_file()
                        else None
                    ),
                    annotation_path=annotation_path.resolve() if isinstance(annotation_path, Path) and annotation_path.is_file() else None,
                    annotation_preview=annotation_preview,
                    supervised_library_path=library_path.resolve() if isinstance(library_path, Path) and library_path.exists() else None,
                    updated_at=updated_at,
                )
            )
            seen_tasks[outcome_kind].add(task_id)
            gathered[outcome_kind] += 1
        if all(gathered[key] >= target_counts[key] for key in target_counts):
            break
    return entries


def rebuild_video_render_results_index(project_root: Path) -> dict[str, list[VideoRenderResultEntry]]:
    rebuilt: dict[str, dict[str, dict[str, Any]]] = {env_id: {} for env_id in SUPPORTED_VIDEO_RENDER_ENVS}
    for env_id in SUPPORTED_VIDEO_RENDER_ENVS:
        env_root = project_root / "data" / "evaluations" / env_id
        if not env_root.is_dir():
            continue
        for task_result_path in sorted(env_root.glob("video-render-*/tasks/*/*/task_result.json")):
            try:
                payload = json.loads(task_result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            trace_path = _normalize_video_asset_path(project_root, str(payload.get("final_trace_path") or ""))
            media_paths = resolve_trace_media_paths(trace_path)
            if media_paths.default_video_path is None:
                continue
            annotation_path, annotation_preview, library_path = _load_existing_best_supervised_annotation(
                project_root,
                env_id,
                task_dir=task_result_path.parent,
                trace_path=trace_path,
            )
            source_run_id = task_result_path.parents[3].name.removeprefix("video-render-")
            task_id = str(payload.get("task_id") or task_result_path.parent.name).strip() or task_result_path.parent.name
            key = f"{source_run_id}:{task_id}"
            annotated_video_path, annotated_overhead_video_path, _meta_path = _annotated_video_paths(task_result_path.parent)
            overlay_is_current = _annotated_video_meta_matches_current_state(
                project_root=project_root,
                env_id=env_id,
                task_dir=task_result_path.parent,
                annotation_path=annotation_path if annotation_path is not None and annotation_path.is_file() else None,
                trace_path=trace_path,
            )
            rebuilt[env_id][key] = {
                "env_id": env_id,
                "task_id": task_id,
                "source_run_id": source_run_id,
                "output_run_id": task_result_path.parents[3].name,
                "outcome_kind": _result_outcome_kind(payload) or "success",
                "task_result_path": _project_relpath(project_root, task_result_path),
                "video_path": _project_relpath(project_root, media_paths.default_video_path),
                "overhead_video_path": (
                    _project_relpath(project_root, media_paths.overhead_video_path)
                    if media_paths.overhead_video_path is not None
                    else ""
                ),
                "annotated_video_path": (
                    _project_relpath(project_root, annotated_video_path)
                    if overlay_is_current and annotated_video_path.is_file()
                    else ""
                ),
                "annotated_overhead_video_path": (
                    _project_relpath(project_root, annotated_overhead_video_path)
                    if overlay_is_current and annotated_overhead_video_path.is_file()
                    else ""
                ),
                "annotation_path": (
                    _project_relpath(project_root, annotation_path)
                    if annotation_path is not None
                    else ""
                ),
                "annotation_preview": annotation_preview,
                "supervised_library_path": (
                    _project_relpath(project_root, library_path)
                    if library_path is not None
                    else ""
                ),
                "updated_at": str(payload.get("video_rendered_at") or _isoformat(datetime.fromtimestamp(task_result_path.stat().st_mtime, tz=timezone.utc))),
            }
    _write_results(project_root, rebuilt)
    return list_video_render_results(project_root, _allow_rebuild=False)


def _select_replay_record(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Missing records in task result")
    replayable = [record for record in records if isinstance(record, dict) and isinstance(record.get("parsed_action"), dict)]
    if not replayable:
        raise ValueError("No replayable record found")
    successful = [record for record in replayable if record.get("success") is True]
    if successful:
        return successful[-1]
    traced = [record for record in replayable if str(record.get("trace_path") or "").strip()]
    if traced:
        return traced[-1]
    return replayable[-1]


def _select_replay_sequence(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Missing records in task result")
    sequence: list[dict[str, Any]] = []
    top_level_success = payload.get("success") is True
    for record in records:
        if not isinstance(record, dict):
            continue
        sequence.append(record)
        if top_level_success and record.get("success") is True:
            return sequence
    if sequence:
        return sequence
    raise ValueError("No replayable record found")


def _import_replay_runtime() -> tuple[Any, Any]:
    from adapters.registry import AdapterRegistry
    from deepphy_eval.io import write_trace

    return AdapterRegistry, write_trace


def _replay_phyre(task: VideoRenderTask, adapter: Any, write_trace: Any) -> str:
    record = _select_replay_record(task.payload)
    parsed_action = record.get("parsed_action")
    action = parsed_action.get("action") if isinstance(parsed_action, dict) else None
    if not isinstance(action, list):
        raise ValueError("PHYRE result has no replayable action")
    attempt_root = task.output_task_dir / "attempt_001"
    last_error: Exception | None = None
    for eval_setup in ("ball_within_template", "ball_cross_template"):
        try:
            trace = adapter.generate_trace(
                task.task_id,
                action=action,
                need_images=True,
                capture_visuals=True,
                output_dir=attempt_root,
                eval_setup=eval_setup,
                fold_id=0,
                eval_type="test",
            )
            trace_relpath = Path("attempt_001") / "trace.json"
            write_trace(task.output_task_dir / trace_relpath, trace)
            return trace_relpath.as_posix()
        except Exception as exc:  # pragma: no cover - exact adapter failure depends on env data
            last_error = exc
    raise RuntimeError(f"Unable to replay PHYRE task {task.task_id}: {last_error}")


def _replay_iphyre(task: VideoRenderTask, adapter: Any, write_trace: Any) -> str:
    record = _select_replay_record(task.payload)
    parsed_action = record.get("parsed_action")
    actions = parsed_action.get("parsed_actions") if isinstance(parsed_action, dict) else None
    if not isinstance(actions, list) or not actions:
        raise ValueError("I-PHYRE result has no replayable action sequence")
    attempt_root = task.output_task_dir / "attempt_001"
    trace = adapter.generate_trace(
        task.task_id,
        actions=actions,
        capture_visuals=True,
        output_dir=attempt_root,
    )
    trace_relpath = Path("attempt_001") / "trace.json"
    write_trace(task.output_task_dir / trace_relpath, trace)
    return trace_relpath.as_posix()


def _replay_kinetix(task: VideoRenderTask, adapter: Any, write_trace: Any) -> str:
    sequence = _select_replay_sequence(task.payload)
    action_sequence: list[list[int]] = []
    success_index = 0
    for record in sequence:
        parsed_action = record.get("parsed_action")
        padded_action = parsed_action.get("padded_action") if isinstance(parsed_action, dict) else None
        if isinstance(padded_action, list):
            action_sequence.append(padded_action)
        if task.payload.get("success") is True and record.get("success") is True:
            success_index = int(record.get("index") or len(action_sequence) or 1)
            break
    if not action_sequence:
        raise ValueError("Kinetix result has no replayable action sequence")
    if success_index <= 0:
        success_index = len(action_sequence)
    step_root = task.output_task_dir / f"step_{success_index:03d}"
    trace = adapter.generate_trace(
        task.task_id,
        action_sequence=action_sequence,
        annotation_mode="anno",
        capture_frames=True,
        output_dir=step_root,
        video_fps=KINETIX_VIDEO_RENDER_FPS,
        media_max_side=KINETIX_VIDEO_RENDER_MAX_SIDE,
    )
    trace_relpath = Path(f"step_{success_index:03d}") / "trace.json"
    write_trace(task.output_task_dir / trace_relpath, trace)
    return trace_relpath.as_posix()


def _replay_pooltool(task: VideoRenderTask, adapter: Any, write_trace: Any) -> str:
    sequence = _select_replay_sequence(task.payload)
    shots: list[dict[str, str]] = []
    success_index = 0
    for record in sequence:
        parsed_action = record.get("parsed_action")
        speed = parsed_action.get("speed") if isinstance(parsed_action, dict) else None
        strikespot = parsed_action.get("strikespot") if isinstance(parsed_action, dict) else None
        if speed and strikespot:
            shots.append({"speed": str(speed), "strikespot": str(strikespot)})
        if task.payload.get("success") is True and record.get("success") is True:
            success_index = int(record.get("index") or len(shots) or 1)
            break
    if not shots:
        raise ValueError("Pooltool result has no replayable shots")
    if success_index <= 0:
        success_index = len(shots)
    attempt_root = task.output_task_dir / f"attempt_{success_index:03d}"
    trace = adapter.generate_trace(
        task.task_id,
        shots=shots,
        capture_frames=True,
        output_dir=attempt_root,
    )
    trace_relpath = Path(f"attempt_{success_index:03d}") / "trace.json"
    write_trace(task.output_task_dir / trace_relpath, trace)
    return trace_relpath.as_posix()


def replay_task_video(project_root: Path, task: VideoRenderTask) -> None:
    AdapterRegistry, write_trace = _import_replay_runtime()
    client_root = _client_project_root()
    task.output_task_dir.mkdir(parents=True, exist_ok=True)
    registry = AdapterRegistry(project_root=client_root)
    adapter = registry.load(task.env_id)
    if task.env_id == "phyre":
        trace_relpath = _replay_phyre(task, adapter, write_trace)
    elif task.env_id == "iphyre":
        trace_relpath = _replay_iphyre(task, adapter, write_trace)
    elif task.env_id == "kinetix":
        trace_relpath = _replay_kinetix(task, adapter, write_trace)
    elif task.env_id == "pooltool":
        trace_relpath = _replay_pooltool(task, adapter, write_trace)
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"Unsupported environment {task.env_id}")
    rendered_payload = _copy_source_payload(task.payload, output_task_dir=task.output_task_dir, trace_relpath=trace_relpath)
    task.output_result_path.write_text(json.dumps(rendered_payload, indent=2), encoding="utf-8")
    _upsert_video_render_result(project_root, task, rendered_payload)


def run_video_render_batch(project_root: Path, env_id: str, *, target_count: int = DEFAULT_VIDEO_RENDER_TARGET) -> int:
    env_key = str(env_id or "").strip().lower()
    if env_key not in SUPPORTED_VIDEO_RENDER_ENVS:
        raise ValueError(f"Unknown environment {env_key}")
    normalized_target = _normalize_target_count(target_count)
    tasks = discover_video_render_tasks(project_root, env_key, target_count=normalized_target)
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
            if env_key == "kinetix":
                _replay_task_video_subprocess(project_root, task)
            else:
                replay_task_video(project_root, task)
            completed += 1
        except Exception as exc:
            failed += 1
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
    error_text = "\n".join(failures[:6])
    _merge_env_state(
        project_root,
        env_key,
        {
            "status": status,
            "pid": None,
            "finished_at": _isoformat(),
            "current_task_id": "",
            "current_source_run_id": "",
            "completed_tasks": completed,
            "failed_tasks": failed,
            "error": error_text,
        },
    )
    return 0 if failed == 0 else 1


def run_annotation_video_batch(project_root: Path, env_id: str) -> int:
    env_key = str(env_id or "").strip().lower()
    if env_key not in SUPPORTED_VIDEO_RENDER_ENVS:
        raise ValueError(f"Unknown environment {env_key}")
    selection = load_annotation_video_selection(project_root, env_key)
    selected_labels = list(selection.get("selected_labels") or [])
    entries = select_video_showcase_entries(list_video_render_results(project_root).get(env_key, []))
    _merge_annotation_env_state(
        project_root,
        env_key,
        {
            "status": "running",
            "pid": os.getpid(),
            "planned_tasks": len(entries),
            "completed_tasks": 0,
            "failed_tasks": 0,
            "current_task_id": "",
            "current_view": "",
            "current_video_frame": 0,
            "current_video_total_frames": 0,
            "started_at": _isoformat(),
            "finished_at": None,
            "error": "",
        },
    )
    failures: list[str] = []
    completed = 0
    failed = 0
    for index, entry in enumerate(entries, start=1):
        _merge_annotation_env_state(
            project_root,
            env_key,
            {
                "current_task_id": entry.task_id,
                "completed_tasks": completed,
                "failed_tasks": failed,
                "current_video_frame": 0,
                "current_video_total_frames": 0,
            },
        )
        try:
            trace_payload = _load_json_object(entry.task_result_path)
            trace_path = _normalize_video_asset_path(
                project_root,
                str(trace_payload.get("final_trace_path") or ""),
            )
            if not trace_path.is_file():
                raise FileNotFoundError(f"Missing trace for {entry.task_id}")
            annotation_path, _annotation_preview, _library_path = _ensure_best_supervised_annotation(
                project_root,
                env_key,
                trace_path=trace_path,
                task_dir=entry.task_result_path.parent,
            )
            if not isinstance(annotation_path, Path) or not annotation_path.is_file():
                raise FileNotFoundError(f"Missing annotation for {entry.task_id}")
            annotated_video_path, annotated_overhead_video_path, meta_path = _annotated_video_paths(entry.task_result_path.parent)
            generated_views: list[str] = []
            if env_key != "pooltool" and isinstance(entry.video_path, Path) and entry.video_path.is_file():
                _render_annotated_video(
                    project_root=project_root,
                    env_id=env_key,
                    trace_path=trace_path,
                    annotation_path=annotation_path,
                    source_video_path=entry.video_path,
                    output_video_path=annotated_video_path,
                    selected_labels=selected_labels,
                    view="askew",
                    task_id=entry.task_id,
                    progress_index=index,
                    progress_total=len(entries),
                )
                generated_views.append("askew")
            if isinstance(entry.overhead_video_path, Path) and entry.overhead_video_path.is_file():
                _render_annotated_video(
                    project_root=project_root,
                    env_id=env_key,
                    trace_path=trace_path,
                    annotation_path=annotation_path,
                    source_video_path=entry.overhead_video_path,
                    output_video_path=annotated_overhead_video_path,
                    selected_labels=selected_labels,
                    view="overhead",
                    task_id=entry.task_id,
                    progress_index=index,
                    progress_total=len(entries),
                )
                generated_views.append("overhead")
            meta_path.write_text(
                json.dumps(
                    {
                        "env_id": env_key,
                        "task_id": entry.task_id,
                        "trace_path": str(trace_path),
                        "trace_fingerprint": _trace_payload_fingerprint(trace_path),
                        "annotation_mtime_ns": int(annotation_path.stat().st_mtime_ns),
                        "annotation_fingerprint": _annotation_payload_fingerprint(annotation_path),
                        "selected_labels": selected_labels,
                        "selected_count": len(selected_labels),
                        "generated_views": generated_views,
                        "generated_at": _isoformat(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            completed += 1
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            failed += 1
            failures.append(f"{entry.task_id}: {exc}")
        _merge_annotation_env_state(
            project_root,
            env_key,
            {
                "completed_tasks": completed,
                "failed_tasks": failed,
                "current_video_frame": 0,
                "current_video_total_frames": 0,
            },
        )
    rebuild_video_render_results_index(project_root)
    status = "succeeded" if failed == 0 else "failed"
    _merge_annotation_env_state(
        project_root,
        env_key,
        {
            "status": status,
            "pid": None,
            "finished_at": _isoformat(),
            "current_task_id": "",
            "current_view": "",
            "current_video_frame": 0,
            "current_video_total_frames": 0,
            "completed_tasks": completed,
            "failed_tasks": failed,
            "error": "\n".join(failures[:8]),
        },
    )
    return 0 if failed == 0 else 1


__all__ = [
    "SUPPORTED_VIDEO_RENDER_ENVS",
    "TraceMediaPaths",
    "VideoRenderResultEntry",
    "VIDEO_RENDER_STATE_RELPATH",
    "VIDEO_RENDER_RESULTS_RELPATH",
    "VIDEO_ANNOTATION_STATE_RELPATH",
    "VIDEO_ANNOTATION_CONFIG_RELPATH",
    "annotation_video_filter_descriptions",
    "cancel_annotation_video_process",
    "build_annotation_overlay_payload",
    "DEFAULT_VIDEO_RENDER_TARGET",
    "discover_video_render_tasks",
    "filter_annotation_patterns_for_env",
    "list_annotation_video_statuses",
    "load_cached_global_best_supervised_library_summary",
    "load_global_best_supervised_library_summary",
    "load_annotation_video_selection",
    "list_video_render_results",
    "list_video_render_statuses",
    "rebuild_video_render_results_index",
    "resolve_trace_media_paths",
    "run_annotation_video_batch",
    "save_annotation_video_selection",
    "start_annotation_video_process",
    "run_video_render_batch",
    "start_video_render_process",
    "summarize_video_render_inventory",
]
