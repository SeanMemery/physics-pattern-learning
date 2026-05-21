from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import parse_qs
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response, FileResponse

from .env_catalog import (
    get_environment_dataset,
    get_environment_dataset_trace,
    list_environment_datasets,
    list_environment_libraries,
    list_environment_reward_results,
)
from .main import _resolve_project_file
from public_connection_public_access import PublicAccessController
from .state import ServerStateStore
from .ui import (
    REQUIRED_LIBRARY_ABLATION_JOB_FAMILIES,
    REPEAT_LEARNING_FAMILIES,
    SUPPORTED_ENVS,
    _resolve_job_run_dir,
    _required_library_ablation_job_catalog,
    get_evaluation_task_detail,
    required_family_from_slug,
    build_ablation_chart_data_for_env_kind,
    build_combined_ablation_chart_data,
    build_combined_ablation_importance_scatter_data,
    render_ablation_importance_scatter_pdf_bytes,
    render_ablation_plot_pdf_bytes,
    render_ablations_home_page,
    render_analysis_home_page,
    build_combined_analysis_pattern_chart_data,
    render_analysis_plot_pdf_bytes,
    render_analysis_environment_page,
    render_ablation_data_page,
    render_ablation_detail_page,
    render_client_page,
    _discover_on_disk_ablation_runs,
    render_environment_dataset_page,
    render_environment_datasets_page,
    render_environment_dataset_trace_page,
    render_environment_libraries_page,
    render_environment_page,
    render_environment_rewards_page,
    render_environment_reward_runs_page,
    render_evaluation_task_page,
    render_home_page,
    render_learning_home_page,
    render_job_page,
    render_library_page,
    render_required_variant_page,
    render_rewards_home_page,
    compute_statistics_snapshot,
    _canonical_label_key,
    _compute_ablation_drop_estimates_by_env,
    compute_activation_rates_snapshot,
    compute_success_correlations_snapshot,
    patch_statistics_rows_with_ablation_estimates,
    patch_statistics_rows_with_activation_rates,
    patch_statistics_rows_with_success_correlations,
    render_profiler_page,
    render_statistics_page,
    render_summaries_page,
    render_summaries_environment_page,
    render_supervised_consistency_result_page,
    render_videos_page,
    _render_ablation_importance_scatter,
    _render_xy_line_chart,
)
from .summary_profiler import load_profiler_state, save_profiler_state, PROFILER_STATE_RELPATH
from .library_collections import (
    load_cached_canonical_library_summary,
    resolve_canonical_library,
    recalculate_canonical_libraries,
    _client_project_root,
)
from .ablation_cache import load_ablation_plot_cache
from .library_ablation import load_library_ablation_job_detail
from .analysis_results import (
    filter_supervised_consistency_results_to_latest_setup,
    list_supervised_consistency_results,
    list_supervised_run_options,
    load_supervised_consistency_result,
)
from .models import CreateJobRequest, JobConstraints, JobRecord, JobSpec, JobStatus
from .video_renders import (
    DEFAULT_VIDEO_RENDER_TARGET,
    build_annotation_overlay_payload,
    cancel_annotation_video_process,
    export_annotation_overlay_video,
    list_video_render_results,
    load_annotation_video_selection,
    save_annotation_video_selection,
    select_video_showcase_entries,
    start_video_render_process,
)
from .task_summaries import (
    DEFAULT_TASK_SUMMARY_TARGET,
    list_completed_task_summaries,
    list_task_summary_statuses,
    load_task_summary_config,
    save_task_summary_config,
    start_task_summary_process,
)
from .rewards_dashboard import (
    REWARD_PANEL_KEYS,
    load_rewards_dashboard_state,
    RANDOM_SCENE_SESSION,
    reward_panel_exists,
    reward_panel_running,
    reward_panel_session_name,
    reward_run_root,
    random_scene_log_path,
    random_scene_exists,
    random_scene_running,
    rewards_dashboard_log_path,
    rewards_dashboard_progress_path,
    rewards_dashboard_root,
    rewards_dashboard_scene_dir,
    save_rewards_dashboard_state,
)

try:
    from tqdm import tqdm as _tqdm_cls
except Exception:  # pragma: no cover - fallback when tqdm is not installed
    _tqdm_cls = None


def _resolve_next_url(raw_value: str | None, *, default: str = "/") -> str:
    candidate = str(raw_value or "").strip()
    if not candidate.startswith("/") or candidate.startswith("//"):
        return default
    return candidate


def _coerce_form_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "on", "yes"}


def _cached_canonical_library_path(*, project_root: Path, env_id: str, kind: str) -> Path:
    summary = load_cached_canonical_library_summary(project_root, env_id, kind)
    library_path_text = str(summary.get("library_path") or "").strip()
    if not library_path_text:
        raise FileNotFoundError(
            f"No cached canonical {kind} library for {env_id}. Use Recalculate libraries first."
        )
    path = Path(library_path_text).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Cached canonical {kind} library for {env_id} is missing on disk. Use Recalculate libraries first."
        )
    return path


def _canonical_ablation_pattern_run_dir(*, project_root: Path, env_id: str, kind: str) -> str:
    canonical_library_path = _cached_canonical_library_path(project_root=project_root, env_id=env_id, kind=kind)
    return str(canonical_library_path.parent)


def _resolve_ablation_source_run_dir(*, project_root: Path, env_id: str, kind: str) -> str:
    try:
        return _canonical_ablation_pattern_run_dir(project_root=project_root, env_id=env_id, kind=kind)
    except Exception:
        libraries = [
            item
            for item in list_environment_libraries(project_root, env_id)
            if str(item.workflow_kind or "").strip().lower() == str(kind or "").strip().lower()
        ]
        if libraries:
            return str(libraries[0].root)
        raise


LIGHTWEIGHT_ABLATION_QUEUE_DEFAULTS: dict[str, int] = {
    "seed_count": 2,
    "holdout_sample_count": 2,
    "single_holdout_repeat_count": 2,
    "pair_holdout_size": 2,
    "pair_holdout_sample_count": 2,
}


def _continuous_ablation_loop_run_id(*, env_id: str, kind: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"continuous-{env_id}-{kind}-{stamp}"


def _services_root_for_project(project_root: Path) -> Path:
    resolved = project_root.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / "Work").is_dir() and (candidate / "AGENTS.md").is_file():
            return candidate
    return resolved


def _library_ablation_catalog_path(project_root: Path) -> Path:
    return _services_root_for_project(project_root) / "LIBRARY_ABLATION_JOBS.md"


def _read_required_library_ablation_catalog(project_root: Path) -> dict[tuple[str, str], str]:
    path = _library_ablation_catalog_path(project_root)
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


def _job_pattern_run_dir_for_catalog(payload: dict[str, object] | None) -> str:
    raw_payload = payload if isinstance(payload, dict) else {}
    spec = raw_payload.get("spec")
    spec_payload = spec if isinstance(spec, dict) else raw_payload
    metadata = spec_payload.get("metadata")
    metadata_payload = metadata if isinstance(metadata, dict) else {}
    request_payload = (
        metadata_payload.get("request")
        if isinstance(metadata_payload.get("request"), dict)
        else {}
    )
    return str(
        metadata_payload.get("pattern_run_dir")
        or request_payload.get("pattern_run_dir")
        or ""
    ).strip()


def _write_required_library_ablation_catalog(
    project_root: Path,
    catalog: dict[tuple[str, str], str],
    *,
    live_jobs: list,
    created_job_payload: dict[str, object] | None = None,
) -> None:
    path = _library_ablation_catalog_path(project_root)
    jobs_by_id = {str(job.job_id): job for job in live_jobs}
    created_job_id = str((created_job_payload or {}).get("job_id") or "").strip()
    lines = [
        "# Required Library Ablation Jobs",
        "",
        "This file tracks the required env-by-family library-ablation set for the pattern-learning server.",
        "",
        "Rule:",
        "- Each env should track one supervised and one unsupervised library ablation.",
        "- The dashboard rewrites this file when tracked ablations are changed from `/ablations`.",
        "- `succeeded`, `queued`, `assigned`, `starting`, `running`, and intentionally staged `paused` all count as healthy coverage.",
        "- These jobs use the smaller `quick-eval` holdout configuration.",
        "",
        "| Family | Env | Job ID | Status | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for env_id in SUPPORTED_ENVS:
        for family in REQUIRED_LIBRARY_ABLATION_JOB_FAMILIES:
            job_id = str(catalog.get((env_id, family), "") or "").strip()
            job = jobs_by_id.get(job_id)
            status = ""
            note = "No tracked job selected yet."
            pattern_run_dir = ""
            if job is not None:
                status = job.status.value
                metadata = dict(job.spec.metadata or {})
                request_payload = metadata.get("request")
                request_dict = request_payload if isinstance(request_payload, dict) else {}
                pattern_run_dir = str(
                    metadata.get("pattern_run_dir") or request_dict.get("pattern_run_dir") or ""
                ).strip()
            elif job_id and created_job_id and job_id == created_job_id:
                status = str((created_job_payload or {}).get("status") or "queued").strip()
                pattern_run_dir = _job_pattern_run_dir_for_catalog(created_job_payload)
            elif job_id:
                status = "missing"
                note = "Tracked job id is not present in current server state."
            if pattern_run_dir:
                note = f"Tracked source `{Path(pattern_run_dir).name}`"
            lines.append(
                f"| {family} | {env_id} | {f'`{job_id}`' if job_id else ''} | {f'`{status}`' if status else ''} | {note} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_dashboard_app(
    project_root: Path | None = None,
    *,
    backend_base_url: str = "http://127.0.0.1:8791",
    backend_private_port: int = 8791,
    backend_public_https_port: int = 10000,
) -> FastAPI:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    state_store = ServerStateStore(project_root / "state", project_root=project_root)
    public_local_target_url = str(
        os.environ.get("PATTERN_PUBLIC_LOCAL_TARGET_URL", f"http://127.0.0.1:{int(backend_private_port)}")
    ).strip()
    public_access = PublicAccessController(
        project_root / "state",
        local_target_url=public_local_target_url,
        https_port=int(backend_public_https_port),
    )
    backend_root = str(backend_base_url).rstrip("/")
    app = FastAPI(title="Pattern Server Dashboard")

    home_cache: dict[str, object] = {"html": None, "expires_at": 0.0}
    learning_cache: dict[str, object] = {"html": None, "expires_at": 0.0}
    ablations_cache: dict[str, object] = {"html": None, "expires_at": 0.0}
    ablation_data_cache: dict[tuple[str, str], dict[str, object]] = {}
    ablation_plot_context_cache: dict[str, object] = {"entries": None, "expires_at": 0.0}
    rewards_cache: dict[str, object] = {"html": None, "expires_at": 0.0}
    statistics_cache: dict[str, object] = {"html": None, "expires_at": 0.0, "state_key": None}
    analysis_cache: dict[str, object] = {"html": None, "expires_at": 0.0}
    analysis_results_cache: dict[str, object] = {
        "results_by_env": {env_id: [] for env_id in SUPPORTED_ENVS},
        "expires_at": 0.0,
    }
    library_cache: dict[str, object] = {"html": None, "expires_at": 0.0}
    videos_cache: dict[str, object] = {"html": None, "expires_at": 0.0}
    summaries_cache: dict[str, object] = {"html": None, "expires_at": 0.0}
    ensemble_calibration_state_lock = threading.Lock()
    analysis_state_lock = threading.Lock()
    statistics_state_lock = threading.Lock()
    ablation_importance_state_lock = threading.Lock()
    statistics_cancel_event = threading.Event()
    ENSEMBLE_CALIBRATION_KINDS = ("supervised", "unsupervised")
    ENSEMBLE_CALIBRATION_ACTIVE_STATUSES = {"starting", "running"}
    DEFAULT_ENSEMBLE_CALIBRATION_TARGET_HOURS = 6.0
    DEFAULT_ENSEMBLE_CALIBRATION_TRIAL_SECONDS = {
        "phyre": 1800,
        "iphyre": 900,
        "kinetix": 1500,
        "pooltool": 1200,
    }

    def _reward_default_base_url_local() -> str:
        for name in ("PATTERN_LLM_BASE_URL", "JOB_LLM_BASE_URL", "OPENAI_BASE_URL"):
            value = os.environ.get(name)
            if value is not None and value.strip():
                return value.strip()
        return "http://127.0.0.1:8788/v1"

    def _reward_default_timeout_local() -> int:
        for name in ("PATTERN_MODEL_TIMEOUT", "JOB_MODEL_TIMEOUT"):
            value = os.environ.get(name)
            if value is None or not value.strip():
                continue
            try:
                return max(0, int(value.strip()))
            except ValueError:
                continue
        return 120

    def _reward_client_python() -> Path:
        python_path = _client_project_root() / "venv" / "pooltool" / "bin" / "python"
        if not python_path.is_file():
            raise FileNotFoundError(f"PoolTool client Python not found: {python_path}")
        return python_path

    def _reward_pythonpath() -> str:
        return str(project_root / "src")

    def _reward_source_run_dir() -> Path:
        return resolve_canonical_library(project_root, "pooltool", "supervised").parent

    def _reward_make_run_id(prefix: str) -> str:
        return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    def _reward_load_state() -> dict[str, object]:
        return load_rewards_dashboard_state(project_root)

    def _reward_save_state(payload: dict[str, object]) -> None:
        save_rewards_dashboard_state(project_root, payload)

    def _reward_goal_texts(state: dict[str, object]) -> list[str]:
        payload = state.get("goal_texts")
        if not isinstance(payload, list):
            return []
        return [str(item or "").strip() for item in payload if str(item or "").strip()]

    def _reward_require_full_goal_set(state: dict[str, object]) -> list[str]:
        goal_texts = _reward_goal_texts(state)
        if len(goal_texts) != 10:
            raise ValueError("Fill all 10 shared PoolTool goals before starting this experiment.")
        return goal_texts

    def _reward_cleanup_dead_session(panel_key: str) -> None:
        if reward_panel_exists(panel_key) and not reward_panel_running(panel_key):
            subprocess.run(
                ["tmux", "kill-session", "-t", reward_panel_session_name(panel_key)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

    def _reward_launch_tmux(panel_key: str, argv: list[str]) -> bool:
        _reward_cleanup_dead_session(panel_key)
        if reward_panel_running(panel_key):
            return False
        root = rewards_dashboard_root(project_root)
        root.mkdir(parents=True, exist_ok=True)
        log_path = rewards_dashboard_log_path(project_root, panel_key)
        progress_path = rewards_dashboard_progress_path(project_root, panel_key)
        try:
            log_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            progress_path.unlink(missing_ok=True)
        except OSError:
            pass
        command = (
            f"cd {shlex.quote(str(project_root.parent))} && "
            f"export PYTHONPATH={shlex.quote(_reward_pythonpath())} && "
            "export PYTHONUNBUFFERED=1 && "
            f"export PATTERN_REWARD_PROGRESS_PATH={shlex.quote(str(progress_path))} && "
            f"{shlex.quote(str(_reward_client_python()))} {' '.join(shlex.quote(part) for part in argv)} "
            f"> {shlex.quote(str(log_path))} 2>&1"
        )
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", reward_panel_session_name(panel_key), command],
            check=True,
        )
        return True

    def _reward_cancel_tmux(panel_key: str) -> None:
        subprocess.run(
            ["tmux", "kill-session", "-t", reward_panel_session_name(panel_key)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def _reward_cleanup_dead_random_scene_session() -> None:
        if random_scene_exists() and not random_scene_running():
            subprocess.run(
                ["tmux", "kill-session", "-t", RANDOM_SCENE_SESSION],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

    def _reward_start_random_scene(scene_seed: int) -> str:
        if random_scene_running():
            raise ValueError("Random scene generation is already running.")
        _reward_cleanup_dead_random_scene_session()
        scene_dir = rewards_dashboard_scene_dir(project_root)
        pending_scene_dir = scene_dir.parent / f"{scene_dir.name}.pending-{int(scene_seed):04d}"
        log_path = random_scene_log_path(project_root)
        try:
            log_path.unlink(missing_ok=True)
        except OSError:
            pass
        scene_id = f"scene-{scene_seed:04d}"
        repair_command = (
            "import json; "
            "from pathlib import Path; "
            f"scene_dir = Path({str(scene_dir)!r}); "
            "scene_path = scene_dir / 'scene.json'; "
            "payload = json.loads(scene_path.read_text(encoding='utf-8')); "
            "payload['image_path'] = str((scene_dir / 'initial_scene_overhead.png').resolve()); "
            "scene_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n', encoding='utf-8')"
        )
        command = (
            f"cd {shlex.quote(str(project_root.parent))} && "
            f"export PYTHONPATH={shlex.quote(_reward_pythonpath())} && "
            "{ "
            f"find {shlex.quote(str(scene_dir.parent))} -maxdepth 1 -type d -name {shlex.quote(scene_dir.name + '.pending-*')} -exec rm -rf {{}} + && "
            f"rm -rf {shlex.quote(str(pending_scene_dir))} && "
            f"mkdir -p {shlex.quote(str(pending_scene_dir))} && "
            f"{shlex.quote(str(_reward_client_python()))} -m dashboard.workflows pooltool-reward-random-scene "
            f"--output-dir {shlex.quote(str(pending_scene_dir))} "
            f"--scene-id {shlex.quote(scene_id)} "
            f"--seed {shlex.quote(str(int(scene_seed)))} "
            f"&& rm -rf {shlex.quote(str(scene_dir))} "
            f"&& mv {shlex.quote(str(pending_scene_dir))} {shlex.quote(str(scene_dir))} "
            f"&& {shlex.quote(str(_reward_client_python()))} -c {shlex.quote(repair_command)}; "
            "} "
            f"> {shlex.quote(str(log_path))} 2>&1"
        )
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", RANDOM_SCENE_SESSION, command],
            check=True,
        )
        return str((scene_dir / "scene.json").resolve())

    def _ensemble_calibration_state_path() -> Path:
        return project_root / "state" / "ensemble_calibration_state.json"

    def _ensemble_calibration_key(*, env_id: str, kind: str) -> str:
        return f"{str(kind).strip().lower()}:{str(env_id).strip().lower()}"

    def _default_ensemble_calibration_trials(env_id: str) -> int:
        trial_seconds = int(
            DEFAULT_ENSEMBLE_CALIBRATION_TRIAL_SECONDS.get(
                str(env_id or "").strip().lower(),
                1200,
            )
        )
        target_seconds = int(DEFAULT_ENSEMBLE_CALIBRATION_TARGET_HOURS * 3600.0)
        return max(8, min(32, int(round(float(target_seconds) / float(max(1, trial_seconds))))))

    def _default_ensemble_calibration_random_init(env_id: str) -> int:
        default_trials = _default_ensemble_calibration_trials(env_id)
        return max(4, min(8, int(round(default_trials / 4.0))))

    def _default_ensemble_calibration_workers(env_id: str) -> int:
        return 1 if str(env_id or "").strip().lower() == "pooltool" else 2

    def _default_ensemble_calibration_entry(env_id: str, kind: str) -> dict[str, object]:
        return {
            "env_id": env_id,
            "kind": kind,
            "status": "idle",
            "run_id": "",
            "library_path": "",
            "output_dir": "",
            "created_at": "",
            "updated_at": "",
            "started_at": "",
            "finished_at": "",
            "message": "",
            "pid": None,
            "request": {
                "env_id": env_id,
                "kind": kind,
                "trials": _default_ensemble_calibration_trials(env_id),
                "random_init_trials": _default_ensemble_calibration_random_init(env_id),
                "workers": _default_ensemble_calibration_workers(env_id),
                "seed": 0,
                "model": "",
                "ollama_base_url": "http://127.0.0.1:8788",
                "model_timeout": 120,
            },
        }

    def _default_ensemble_calibration_state() -> dict[str, object]:
        entries: dict[str, object] = {}
        for kind in ENSEMBLE_CALIBRATION_KINDS:
            for env_id in SUPPORTED_ENVS:
                entries[_ensemble_calibration_key(env_id=env_id, kind=kind)] = _default_ensemble_calibration_entry(
                    env_id,
                    kind,
                )
        return {"entries": entries}

    def _load_ensemble_calibration_state() -> dict[str, object]:
        path = _ensemble_calibration_state_path()
        if not path.is_file():
            return _default_ensemble_calibration_state()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        entries_raw = payload.get("entries")
        merged: dict[str, object] = {}
        for kind in ENSEMBLE_CALIBRATION_KINDS:
            for env_id in SUPPORTED_ENVS:
                key = _ensemble_calibration_key(env_id=env_id, kind=kind)
                base = _default_ensemble_calibration_entry(env_id, kind)
                current = entries_raw.get(key) if isinstance(entries_raw, dict) else {}
                if isinstance(current, dict):
                    base.update(current)
                    request_payload = current.get("request")
                    if isinstance(request_payload, dict):
                        merged_request = dict(base.get("request") or {})
                        merged_request.update(request_payload)
                        base["request"] = merged_request
                merged[key] = base
        return {"entries": merged}

    def _save_ensemble_calibration_state(payload: dict[str, object]) -> None:
        path = _ensemble_calibration_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ensemble_calibration_state: dict[str, object] = _load_ensemble_calibration_state()
    _save_ensemble_calibration_state(ensemble_calibration_state)

    def _snapshot_ensemble_calibration_state() -> dict[str, object]:
        with ensemble_calibration_state_lock:
            return json.loads(json.dumps(ensemble_calibration_state))

    def _ensemble_calibration_entry(env_id: str, kind: str) -> dict[str, object]:
        snapshot = _snapshot_ensemble_calibration_state()
        entries = snapshot.get("entries") if isinstance(snapshot, dict) else {}
        key = _ensemble_calibration_key(env_id=env_id, kind=kind)
        if isinstance(entries, dict):
            current = entries.get(key)
            if isinstance(current, dict):
                return current
        return _default_ensemble_calibration_entry(env_id, kind)

    def _update_ensemble_calibration_entry(env_id: str, kind: str, **fields: object) -> None:
        key = _ensemble_calibration_key(env_id=env_id, kind=kind)
        with ensemble_calibration_state_lock:
            entries = ensemble_calibration_state.setdefault("entries", {})
            current = entries.get(key) if isinstance(entries, dict) else None
            payload = dict(current) if isinstance(current, dict) else _default_ensemble_calibration_entry(env_id, kind)
            payload.update(fields)
            request_payload = payload.get("request")
            if not isinstance(request_payload, dict):
                payload["request"] = dict(_default_ensemble_calibration_entry(env_id, kind).get("request") or {})
            payload["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            entries[key] = payload
            ensemble_calibration_state["entries"] = entries
            _save_ensemble_calibration_state(ensemble_calibration_state)

    def _load_json_object(path: Path) -> dict[str, object]:
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _pid_is_alive(pid: int | None) -> bool:
        if not isinstance(pid, int) or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _local_calibration_python() -> str:
        candidates = [
            project_root / "venv" / "learning" / "bin" / "python",
            project_root / ".venv" / "bin" / "python",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return "python3"

    def _reconcile_ensemble_calibration_entry(env_id: str, kind: str) -> dict[str, object]:
        current = _ensemble_calibration_entry(env_id, kind)
        output_dir_text = str(current.get("output_dir") or "").strip()
        output_dir = Path(output_dir_text).resolve() if output_dir_text else None
        progress_payload = _load_json_object(output_dir / "progress.json") if output_dir is not None else {}
        summary_payload = _load_json_object(output_dir / "summary.json") if output_dir is not None else {}
        best_payload = _load_json_object(output_dir / "best.json") if output_dir is not None else {}
        pid_value = current.get("pid")
        pid = int(pid_value) if isinstance(pid_value, int) or (isinstance(pid_value, str) and str(pid_value).isdigit()) else None
        pid_alive = _pid_is_alive(pid)
        current_status = str(current.get("status") or "").strip().lower()
        next_status = current_status or "idle"
        next_message = str(current.get("message") or "").strip()
        finished_at = str(current.get("finished_at") or "").strip()
        if summary_payload:
            next_status = "succeeded"
            next_message = "Calibration complete."
            if not finished_at:
                finished_at = str(summary_payload.get("finished_at") or "").strip() or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            pid = None
        elif current_status == "cancelled":
            pid = None
        elif pid_alive:
            next_status = "running" if progress_payload else "starting"
            if progress_payload:
                completed = int(progress_payload.get("completed_trials") or 0)
                total = int(progress_payload.get("total_trials") or 0)
                eval_completed = int(progress_payload.get("current_eval_completed_tasks") or 0)
                eval_total = int(progress_payload.get("current_eval_total_tasks") or 0)
                next_message = f"Running calibration: {completed}/{total} trials complete."
                if eval_total > 0:
                    next_message += f" Trial eval {eval_completed}/{eval_total} tasks complete."
            else:
                next_message = "Starting calibration..."
        elif current_status in ENSEMBLE_CALIBRATION_ACTIVE_STATUSES:
            next_status = "failed"
            next_message = "Calibration process exited unexpectedly."
            if not finished_at:
                finished_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            pid = None
        if progress_payload:
            request_payload = dict(current.get("request") or {})
            if not isinstance(request_payload, dict):
                request_payload = {}
            request_payload.setdefault("trials", int(progress_payload.get("total_trials") or 0))
            current["request"] = request_payload
        next_fields = {
            "status": next_status,
            "message": next_message,
            "pid": pid,
        }
        if finished_at:
            next_fields["finished_at"] = finished_at
        if progress_payload:
            next_fields["progress"] = progress_payload
        if summary_payload:
            next_fields["summary"] = summary_payload
        if best_payload:
            next_fields["best"] = best_payload
        _update_ensemble_calibration_entry(env_id, kind, **next_fields)
        return _ensemble_calibration_entry(env_id, kind)

    def _reconcile_all_ensemble_calibrations() -> dict[str, object]:
        refreshed: dict[str, object] = {}
        for kind in ENSEMBLE_CALIBRATION_KINDS:
            for env_id in SUPPORTED_ENVS:
                key = _ensemble_calibration_key(env_id=env_id, kind=kind)
                refreshed[key] = _reconcile_ensemble_calibration_entry(env_id, kind)
        return {"entries": refreshed}

    def _has_active_ensemble_calibrations(snapshot: dict[str, object] | None = None) -> bool:
        current = snapshot if isinstance(snapshot, dict) else _snapshot_ensemble_calibration_state()
        entries = current.get("entries") if isinstance(current, dict) else {}
        if not isinstance(entries, dict):
            return False
        return any(
            str(payload.get("status") or "").strip().lower() in ENSEMBLE_CALIBRATION_ACTIVE_STATUSES
            for payload in entries.values()
            if isinstance(payload, dict)
        )

    def _build_ensemble_calibration_request(
        *,
        env_id: str,
        kind: str,
        trials: int | None = None,
        random_init_trials: int | None = None,
        workers: int | None = None,
        seed: int | None = None,
        model: str | None = None,
    ) -> dict[str, object]:
        library_path = _cached_canonical_library_path(project_root=project_root, env_id=env_id, kind=kind)
        run_id = (
            f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}"
            f"-{env_id}-{kind}-ensemble-calibration"
        )
        output_dir = library_path.parent / "ensemble_calibration_runs" / run_id
        trial_count = max(1, int(trials if trials is not None else _default_ensemble_calibration_trials(env_id)))
        random_trials = max(1, int(random_init_trials if random_init_trials is not None else _default_ensemble_calibration_random_init(env_id)))
        random_trials = min(random_trials, trial_count)
        return {
            "env_id": env_id,
            "kind": kind,
            "library_path": str(library_path),
            "output_dir": str(output_dir),
            "run_id": run_id,
            "trials": trial_count,
            "random_init_trials": random_trials,
            "workers": max(1, int(workers if workers is not None else _default_ensemble_calibration_workers(env_id))),
            "seed": max(0, int(seed if seed is not None else 0)),
            "model": str(model or "").strip(),
            "ollama_base_url": "http://127.0.0.1:8788",
            "model_timeout": 120,
        }

    def _start_ensemble_calibration(env_id: str, kind: str, *, request_payload: dict[str, object] | None = None) -> bool:
        current = _reconcile_ensemble_calibration_entry(env_id, kind)
        if str(current.get("status") or "").strip().lower() in ENSEMBLE_CALIBRATION_ACTIVE_STATUSES:
            return False
        request_payload = dict(request_payload or _build_ensemble_calibration_request(env_id=env_id, kind=kind))
        output_dir = Path(str(request_payload.get("output_dir") or "").strip()).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = output_dir / "stdout.log"
        stderr_path = output_dir / "stderr.log"
        argv = [
            _local_calibration_python(),
            "-m",
            "ensemble_library_training.ensemble_calibration",
            "--env",
            str(env_id),
            "--library-path",
            str(request_payload.get("library_path") or ""),
            "--output-root",
            str(output_dir),
            "--run-id",
            str(request_payload.get("run_id") or ""),
            "--ollama-base-url",
            str(request_payload.get("ollama_base_url") or "http://127.0.0.1:8788"),
            "--model-timeout",
            str(int(request_payload.get("model_timeout") or 120)),
            "--workers",
            str(int(request_payload.get("workers") or 1)),
            "--trials",
            str(int(request_payload.get("trials") or _default_ensemble_calibration_trials(env_id))),
            "--random-init-trials",
            str(int(request_payload.get("random_init_trials") or _default_ensemble_calibration_random_init(env_id))),
            "--seed",
            str(int(request_payload.get("seed") or 0)),
            "--write-sidecar",
        ]
        model_name = str(request_payload.get("model") or "").strip()
        if model_name:
            argv.extend(["--model", model_name])
        env = os.environ.copy()
        client_src = str(project_root / "src")
        existing_pythonpath = str(env.get("PYTHONPATH") or "").strip()
        env["PYTHONPATH"] = f"{client_src}:{existing_pythonpath}" if existing_pythonpath else client_src
        now_text = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open("a", encoding="utf-8") as stderr_handle:
            process = subprocess.Popen(
                argv,
                cwd=str(project_root.parent),
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                start_new_session=True,
            )
        _update_ensemble_calibration_entry(
            env_id,
            kind,
            status="running",
            run_id=str(request_payload.get("run_id") or ""),
            library_path=str(request_payload.get("library_path") or ""),
            output_dir=str(output_dir),
            pid=int(process.pid),
            created_at=now_text,
            started_at=now_text,
            finished_at="",
            message="Running local ensemble calibration...",
            request=request_payload,
            progress={},
            summary={},
            best={},
        )
        _invalidate_cache(library_cache)
        return True

    def _cancel_ensemble_calibration(env_id: str, kind: str) -> bool:
        current = _reconcile_ensemble_calibration_entry(env_id, kind)
        pid_value = current.get("pid")
        pid = int(pid_value) if isinstance(pid_value, int) or (isinstance(pid_value, str) and str(pid_value).isdigit()) else None
        cancelled = False
        if isinstance(pid, int) and pid > 0:
            try:
                os.killpg(pid, signal.SIGTERM)
                cancelled = True
            except OSError:
                cancelled = False
        _update_ensemble_calibration_entry(
            env_id,
            kind,
            status="cancelled",
            pid=None,
            finished_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            message="Calibration cancelled.",
        )
        _invalidate_cache(library_cache)
        return cancelled

    def _default_library_deepphy_eval_workers(env_id: str) -> int:
        return 1 if env_id == "pooltool" else 2

    def _library_deepphy_eval_runs_root(*, library_path: Path, phase: str) -> Path:
        return library_path.parent / "ensemble_eval_runs" / phase

    def _library_deepphy_eval_snapshot_dir(*, library_path: Path, phase: str, env_id: str, kind: str) -> Path:
        run_id = (
            f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}"
            f"-{env_id}-{kind}-{phase}-deepphy"
        )
        return _library_deepphy_eval_runs_root(library_path=library_path, phase=phase) / run_id

    def _prepare_library_deepphy_eval_snapshot(*, env_id: str, kind: str, phase: str) -> Path:
        if phase not in {"baseline", "calibrated"}:
            raise ValueError(f"Unsupported library eval phase {phase}")
        library_path = _cached_canonical_library_path(project_root=project_root, env_id=env_id, kind=kind)
        snapshot_dir = _library_deepphy_eval_snapshot_dir(
            library_path=library_path,
            phase=phase,
            env_id=env_id,
            kind=kind,
        )
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(library_path, snapshot_dir / "pattern_library.json")
        run_json = library_path.parent / "run.json"
        if run_json.is_file():
            shutil.copy2(run_json, snapshot_dir / "run.json")
        source_calibration = library_path.parent / "ensemble_calibration.json"
        target_calibration = snapshot_dir / "ensemble_calibration.json"
        if phase == "calibrated":
            if not source_calibration.is_file():
                raise FileNotFoundError(
                    f"No ensemble_calibration.json found for canonical {kind} library in {env_id}"
                )
            shutil.copy2(source_calibration, target_calibration)
        elif target_calibration.exists():
            target_calibration.unlink()
        manifest = {
            "env_id": env_id,
            "kind": kind,
            "phase": phase,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_library_path": str(library_path),
            "source_library_dir": str(library_path.parent),
            "uses_calibration": phase == "calibrated",
        }
        (snapshot_dir / "ensemble_eval_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return snapshot_dir

    def _start_library_deepphy_eval(*, env_id: str, kind: str, phase: str) -> dict[str, Any]:
        snapshot_dir = _prepare_library_deepphy_eval_snapshot(env_id=env_id, kind=kind, phase=phase)
        payload = {
            "enqueue": True,
            "env_id": env_id,
            "pattern_run_dir": str(snapshot_dir),
            "benchmark_preset": "full",
            "workers": _default_library_deepphy_eval_workers(env_id),
        }
        request = urllib_request.Request(
            f"{backend_root}/api/v1/submit/deepphy-eval",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(request, timeout=30) as response:
            raw = response.read()
        response_payload = json.loads(raw.decode("utf-8")) if raw else {}
        job_payload = response_payload.get("job") if isinstance(response_payload.get("job"), dict) else {}
        if job_payload:
            manifest_path = snapshot_dir / "ensemble_eval_manifest.json"
            manifest = _load_json_object(manifest_path) if manifest_path.is_file() else {}
            manifest["job_id"] = str(job_payload.get("job_id") or "")
            manifest["job_status"] = str(job_payload.get("status") or "")
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return response_payload if isinstance(response_payload, dict) else {}

    def _has_active_library_deepphy_eval_jobs(jobs: list) -> bool:
        active_statuses = {"queued", "assigned", "starting", "running", "pausing", "resuming", "stopping"}
        for job in jobs:
            if str(job.spec.action or "").strip() != "evaluate-deepphy":
                continue
            status_value = str(getattr(job.status, "value", job.status) or "").strip().lower()
            if status_value not in active_statuses:
                continue
            pattern_run_dir = str((job.spec.metadata or {}).get("pattern_run_dir") or "").strip()
            if "/ensemble_eval_runs/" in pattern_run_dir:
                return True
        return False

    def _analysis_state_path() -> Path:
        return project_root / "state" / "analysis_state.json"

    def _default_analysis_env_state(env_id: str) -> dict[str, object]:
        return {
            "env_id": env_id,
            "status": "idle",
            "run_id": "",
            "output_dir": "",
            "created_at": "",
            "updated_at": "",
            "started_at": "",
            "finished_at": "",
            "message": "",
            "run_dirs": [],
            "trace_path": "",
            "trace_scope": "test_set",
            "checkpoints": [],
            "sampling_repeats": 32,
            "pending_runs": [],
            "log_lines": [],
            "progress": {},
        }

    def _default_analysis_state() -> dict[str, object]:
        return {
            "envs": {
                env_id: _default_analysis_env_state(env_id)
                for env_id in SUPPORTED_ENVS
            }
        }

    def _load_analysis_state() -> dict[str, object]:
        path = _analysis_state_path()
        if not path.is_file():
            return _default_analysis_state()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        envs_raw = payload.get("envs")
        merged_envs: dict[str, object] = {}
        for env_id in SUPPORTED_ENVS:
            base = _default_analysis_env_state(env_id)
            current = envs_raw.get(env_id) if isinstance(envs_raw, dict) else {}
            if isinstance(current, dict):
                base.update(current)
            pending_runs = base.get("pending_runs")
            if not isinstance(pending_runs, list):
                base["pending_runs"] = []
            merged_envs[env_id] = base
        payload["envs"] = merged_envs
        return payload

    def _save_analysis_state(payload: dict[str, object]) -> None:
        path = _analysis_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_path.replace(path)

    analysis_state: dict[str, object] = _load_analysis_state()
    _save_analysis_state(analysis_state)

    def _refresh_analysis_state_from_disk() -> None:
        with analysis_state_lock:
            analysis_state.clear()
            analysis_state.update(_load_analysis_state())

    def _snapshot_analysis_state() -> dict[str, object]:
        _refresh_analysis_state_from_disk()
        with analysis_state_lock:
            return json.loads(json.dumps(analysis_state))

    def _analysis_env_state(env_id: str) -> dict[str, object]:
        snapshot = _snapshot_analysis_state()
        envs = snapshot.get("envs") if isinstance(snapshot, dict) else {}
        if isinstance(envs, dict):
            current = envs.get(env_id)
            if isinstance(current, dict):
                return current
        return _default_analysis_env_state(env_id)

    def _update_analysis_env_state(env_id: str, **fields: object) -> None:
        with analysis_state_lock:
            envs = analysis_state.setdefault("envs", {})
            current = envs.get(env_id) if isinstance(envs, dict) else None
            payload = dict(current) if isinstance(current, dict) else _default_analysis_env_state(env_id)
            payload.update(fields)
            payload["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            envs[env_id] = payload
            analysis_state["envs"] = envs
            _save_analysis_state(analysis_state)

    def _append_analysis_log(env_id: str, message: str) -> None:
        with analysis_state_lock:
            envs = analysis_state.setdefault("envs", {})
            current = envs.get(env_id) if isinstance(envs, dict) else None
            payload = dict(current) if isinstance(current, dict) else _default_analysis_env_state(env_id)
            lines_raw = payload.get("log_lines")
            lines = [str(item) for item in lines_raw] if isinstance(lines_raw, list) else []
            lines.append(str(message).rstrip("\n"))
            payload["log_lines"] = lines[-240:]
            payload["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            envs[env_id] = payload
            analysis_state["envs"] = envs
            _save_analysis_state(analysis_state)

    def _canonical_analysis_run_name(value: str) -> str:
        text = str(value or "").strip()
        marker = ".__resume_sync__"
        if marker not in text:
            return text
        return text.split(marker, 1)[0]

    def _select_local_analysis_run_dirs(env_id: str) -> list[str]:
        run_options = [
            option
            for option in list_supervised_run_options(project_root, env_id=env_id, limit=40)
            if option.run_dir.is_dir() and option.label_count > 0 and option.completed_label_count > 0
        ]
        chosen: list[str] = []
        seen: set[str] = set()
        for option in run_options:
            canonical = _canonical_analysis_run_name(option.run_name)
            if canonical in seen:
                continue
            chosen.append(str(option.run_dir))
            seen.add(canonical)
        return chosen

    def _resolve_local_analysis_trace_paths(*, env_id: str, use_env_test_set: bool, trace_path: str | None) -> list[str]:
        if not use_env_test_set and str(trace_path or "").strip():
            resolved = Path(str(trace_path).strip()).resolve()
            if not resolved.is_file():
                raise ValueError(f"Trace file not found: {resolved}")
            return [str(resolved)]
        dataset_root = project_root / "data" / "datasets" / env_id / "default_simulator_trace_dataset" / "test"
        traces = [str(path.resolve()) for path in sorted(dataset_root.glob("**/trace.json")) if path.is_file()]
        if not traces:
            raise ValueError(f"No test traces found for {env_id}")
        return traces

    def _local_analysis_python() -> str:
        candidates = [
            project_root / "venv" / "learning" / "bin" / "python",
            project_root / ".venv" / "bin" / "python",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return "python3"

    def _analysis_job_request_from_payload(
        env_id: str,
        current: dict[str, object],
        *,
        reuse_existing_run: bool = False,
    ) -> CreateJobRequest:
        run_dirs = [
            str(path).strip()
            for path in (current.get("run_dirs") if isinstance(current.get("run_dirs"), list) else [])
            if str(path).strip() and Path(str(path).strip()).is_dir()
        ]
        if len(run_dirs) < 2:
            run_dirs = _select_local_analysis_run_dirs(env_id)
        if len(run_dirs) < 2:
            raise ValueError("Need at least two supervised runs to queue analysis")
        use_env_test_set = str(current.get("trace_scope") or "test_set").strip().lower() != "single_trace"
        trace_path = str(current.get("trace_path") or "").strip() or None
        trace_paths = _resolve_local_analysis_trace_paths(
            env_id=env_id,
            use_env_test_set=use_env_test_set,
            trace_path=trace_path,
        )
        checkpoints: list[int] = []
        raw_checkpoints = current.get("checkpoints")
        if isinstance(raw_checkpoints, list):
            for value in raw_checkpoints:
                try:
                    parsed_value = int(value)
                except (TypeError, ValueError):
                    continue
                if parsed_value > 0:
                    checkpoints.append(parsed_value)
        try:
            sampling_repeats = min(32, max(1, int(current.get("sampling_repeats") or 32)))
        except (TypeError, ValueError):
            sampling_repeats = 32
        existing_run_id = str(current.get("run_id") or "").strip()
        existing_output_dir = Path(str(current.get("output_dir") or "").strip()).resolve() if str(current.get("output_dir") or "").strip() else None
        analysis_root = (project_root / "data" / "analysis" / "supervised_consistency" / env_id).resolve()
        use_existing_run = (
            reuse_existing_run
            and bool(existing_run_id)
            and existing_output_dir is not None
            and (existing_output_dir == analysis_root or analysis_root in existing_output_dir.parents)
        )
        if use_existing_run:
            run_id = existing_run_id
            output_dir = existing_output_dir
        else:
            run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}-{env_id}-supervised-consistency"
            output_dir = analysis_root / run_id
        argv = [
            _local_analysis_python(),
            "-m",
            "code_convergence.consistency_analysis",
            "--env-id",
            env_id,
            "--output-dir",
            str(output_dir),
        ]
        for resolved_trace_path in trace_paths:
            argv.extend(["--trace-path", resolved_trace_path])
        for run_dir in run_dirs:
            argv.extend(["--run-dir", str(run_dir)])
        for checkpoint in checkpoints:
            argv.extend(["--checkpoint", str(int(checkpoint))])
        argv.extend(["--sampling-repeats", str(int(sampling_repeats))])
        return CreateJobRequest(
            enqueue=False,
            spec=JobSpec(
                action="analyze-supervised-consistency",
                argv=argv,
                cwd=str(project_root.parent),
                metadata={
                    "action": "analyze-supervised-consistency",
                    "env_id": env_id,
                    "analysis_kind": "supervised_consistency",
                    "supports_pause": True,
                    "analysis_run_id": run_id,
                    "run_dir": str(output_dir),
                    "trace_path": str(trace_paths[0]),
                    "trace_paths": trace_paths,
                    "trace_scope": "test_set" if use_env_test_set else "single_trace",
                    "source_run_dirs": run_dirs,
                    "artifact_paths": [str(output_dir)],
                    "request": {
                        "env_id": env_id,
                        "trace_path": trace_path,
                        "use_env_test_set": use_env_test_set,
                        "run_dirs": run_dirs,
                        "checkpoints": checkpoints,
                        "sampling_repeats": sampling_repeats,
                        "run_id": run_id,
                        "output_dir": str(output_dir),
                    },
                },
                constraints=JobConstraints(
                    required_environments=["learning", env_id],
                    gpu_required=False,
                    required_tags=[],
                ),
            ),
        )

    def _analysis_job_request_from_state(env_id: str) -> CreateJobRequest:
        return _analysis_job_request_from_payload(env_id, _analysis_env_state(env_id))

    def _analysis_resume_request_from_state(env_id: str) -> CreateJobRequest:
        return _analysis_job_request_from_payload(
            env_id,
            _analysis_env_state(env_id),
            reuse_existing_run=True,
        )

    def _analysis_queue_entry_from_request(request: CreateJobRequest) -> dict[str, object]:
        metadata = dict(request.spec.metadata or {})
        return {
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "argv": list(request.spec.argv),
            "cwd": request.spec.cwd or "",
            "env": dict(request.spec.env or {}),
            "metadata": metadata,
        }

    def _analysis_job_request_from_queue_entry(entry: dict[str, object]) -> CreateJobRequest:
        spec_payload = {
            "action": "analyze-supervised-consistency",
            "argv": [str(item) for item in list(entry.get("argv") or []) if str(item).strip()],
            "cwd": str(entry.get("cwd") or "").strip() or None,
            "env": {
                str(key): str(value)
                for key, value in dict(entry.get("env") or {}).items()
            },
            "metadata": dict(entry.get("metadata") or {}),
            "constraints": {
                "required_environments": [],
                "gpu_required": False,
                "required_tags": [],
            },
        }
        return CreateJobRequest.model_validate({"enqueue": False, "spec": spec_payload})

    def _analysis_job_record_from_state(env_id: str) -> JobRecord | None:
        current = _analysis_env_state(env_id)
        status_text = str(current.get("status") or "").strip().lower()
        run_id = str(current.get("run_id") or "").strip()
        if not run_id or status_text == "idle":
            return None
        try:
            status = JobStatus(status_text)
        except ValueError:
            status = JobStatus.failed
        created_at_text = str(current.get("created_at") or current.get("updated_at") or "").strip()
        updated_at_text = str(current.get("updated_at") or created_at_text).strip()
        started_at_text = str(current.get("started_at") or "").strip()
        finished_at_text = str(current.get("finished_at") or "").strip()

        def _parse_iso(value: str) -> datetime:
            text = value.replace("Z", "+00:00")
            return datetime.fromisoformat(text)

        created_at = _parse_iso(created_at_text) if created_at_text else datetime.now(timezone.utc)
        updated_at = _parse_iso(updated_at_text) if updated_at_text else created_at
        started_at = _parse_iso(started_at_text) if started_at_text else None
        finished_at = _parse_iso(finished_at_text) if finished_at_text else None
        metadata = {
            "env_id": env_id,
            "analysis_kind": "supervised_consistency",
            "analysis_run_id": run_id,
            "run_dir": str(current.get("output_dir") or "").strip(),
            "source_run_dirs": [
                str(path).strip()
                for path in (current.get("run_dirs") if isinstance(current.get("run_dirs"), list) else [])
                if str(path).strip()
            ],
            "trace_path": str(current.get("trace_path") or "").strip(),
            "trace_scope": str(current.get("trace_scope") or "test_set").strip(),
            "checkpoints": list(current.get("checkpoints") or []) if isinstance(current.get("checkpoints"), list) else [],
            "sampling_repeats": int(current.get("sampling_repeats") or 32),
            "progress": dict(current.get("progress") or {}) if isinstance(current.get("progress"), dict) else {},
            "local_analysis": True,
        }
        return JobRecord(
            job_id=run_id,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
            started_at=started_at,
            finished_at=finished_at,
            assigned_client_id=None,
            exit_code=None if status not in {JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled} else (0 if status == JobStatus.succeeded else 1),
            message=str(current.get("message") or "").strip() or None,
            spec=JobSpec(
                action="analyze-supervised-consistency",
                argv=[],
                cwd=str(project_root),
                metadata=metadata,
                constraints=JobConstraints(
                    required_environments=["learning", env_id],
                    gpu_required=False,
                    required_tags=[],
                ),
            ),
            recent_logs=[str(item) for item in (current.get("log_lines") if isinstance(current.get("log_lines"), list) else [])][-120:],
            artifact_relpaths=[],
            pending_commands=[],
        )

    def _analysis_queued_job_records(env_id: str) -> list[JobRecord]:
        current = _analysis_env_state(env_id)
        pending_runs = current.get("pending_runs")
        if not isinstance(pending_runs, list):
            return []

        def _parse_iso(value: str) -> datetime:
            text = str(value or "").strip().replace("Z", "+00:00")
            return datetime.fromisoformat(text)

        jobs: list[JobRecord] = []
        for index, raw_entry in enumerate(pending_runs, start=1):
            if not isinstance(raw_entry, dict):
                continue
            metadata = dict(raw_entry.get("metadata") or {})
            run_id = str(metadata.get("analysis_run_id") or "").strip()
            if not run_id:
                run_id = f"{env_id}-queued-{index:03d}"
            created_at_text = str(raw_entry.get("created_at") or current.get("updated_at") or "").strip()
            created_at = _parse_iso(created_at_text) if created_at_text else datetime.now(timezone.utc)
            source_run_dirs = [
                str(path).strip()
                for path in (metadata.get("source_run_dirs") if isinstance(metadata.get("source_run_dirs"), list) else [])
                if str(path).strip()
            ]
            spec = JobSpec(
                action="analyze-supervised-consistency",
                argv=[str(item) for item in list(raw_entry.get("argv") or []) if str(item).strip()],
                cwd=str(raw_entry.get("cwd") or "").strip() or str(project_root),
                env={str(key): str(value) for key, value in dict(raw_entry.get("env") or {}).items()},
                metadata={
                    **metadata,
                    "env_id": env_id,
                    "analysis_kind": "supervised_consistency",
                    "local_analysis": True,
                },
                constraints=JobConstraints(
                    required_environments=["learning", env_id],
                    gpu_required=False,
                    required_tags=[],
                ),
            )
            jobs.append(
                JobRecord(
                    job_id=run_id,
                    status=JobStatus.queued,
                    created_at=created_at,
                    updated_at=created_at,
                    message="Queued local analysis.",
                    spec=spec,
                    recent_logs=[],
                    artifact_relpaths=[],
                    pending_commands=[],
                )
            )
        return jobs

    def _local_analysis_jobs() -> list[JobRecord]:
        jobs: list[JobRecord] = []
        for env_id in SUPPORTED_ENVS:
            job = _analysis_job_record_from_state(env_id)
            if job is not None:
                jobs.append(job)
            jobs.extend(_analysis_queued_job_records(env_id))
        return jobs

    def _start_local_analysis(env_id: str, *, request: CreateJobRequest | None = None) -> bool:
        current = _analysis_env_state(env_id)
        if str(current.get("status") or "").strip().lower() in {"queued", "starting", "running"}:
            return False
        request = request or _analysis_job_request_from_state(env_id)
        run_id = str((request.spec.metadata or {}).get("analysis_run_id") or "").strip()
        output_dir = str((request.spec.metadata or {}).get("run_dir") or "").strip()
        now_text = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        source_run_dirs = [
            str(path).strip()
            for path in ((request.spec.metadata or {}).get("source_run_dirs") or [])
            if str(path).strip()
        ]
        trace_scope = str((request.spec.metadata or {}).get("trace_scope") or "test_set").strip() or "test_set"
        trace_path = str((request.spec.metadata or {}).get("trace_path") or "").strip()
        checkpoints = list((request.spec.metadata or {}).get("request", {}).get("checkpoints") or [])
        sampling_repeats = int((request.spec.metadata or {}).get("request", {}).get("sampling_repeats") or 32)
        _update_analysis_env_state(
            env_id,
            status="queued",
            run_id=run_id,
            output_dir=output_dir,
            created_at=now_text,
            started_at="",
            finished_at="",
            message="Queued local analysis.",
            run_dirs=source_run_dirs,
            trace_path=trace_path,
            trace_scope=trace_scope,
            checkpoints=checkpoints,
            sampling_repeats=sampling_repeats,
            log_lines=[],
            progress={},
        )
        _invalidate_cache(analysis_cache)

        def _start_next_pending_run() -> None:
            next_request: CreateJobRequest | None = None
            with analysis_state_lock:
                envs = analysis_state.setdefault("envs", {})
                current_payload = envs.get(env_id) if isinstance(envs, dict) else None
                payload = dict(current_payload) if isinstance(current_payload, dict) else _default_analysis_env_state(env_id)
                raw_pending = payload.get("pending_runs")
                pending_runs = list(raw_pending) if isinstance(raw_pending, list) else []
                while pending_runs:
                    candidate = pending_runs.pop(0)
                    if not isinstance(candidate, dict):
                        continue
                    payload["pending_runs"] = pending_runs
                    envs[env_id] = payload
                    analysis_state["envs"] = envs
                    _save_analysis_state(analysis_state)
                    next_request = _analysis_job_request_from_queue_entry(candidate)
                    break
                else:
                    payload["pending_runs"] = pending_runs
                    envs[env_id] = payload
                    analysis_state["envs"] = envs
                    _save_analysis_state(analysis_state)
            if next_request is not None:
                _start_local_analysis(env_id, request=next_request)

        def _worker() -> None:
            try:
                _update_analysis_env_state(
                    env_id,
                    status="starting",
                    started_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    message="Starting local analysis...",
                )
                _invalidate_cache(analysis_cache)
                env = os.environ.copy()
                env.update({str(key): str(value) for key, value in request.spec.env.items()})
                existing_pythonpath = str(env.get("PYTHONPATH") or "").strip()
                client_src = str(project_root / "src")
                env["PYTHONPATH"] = (
                    f"{client_src}:{existing_pythonpath}"
                    if existing_pythonpath
                    else client_src
                )
                process = subprocess.Popen(
                    request.spec.argv,
                    cwd=request.spec.cwd or str(project_root),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                try:
                    _update_analysis_env_state(env_id, status="running", message="Running local analysis...")
                    if process.stdout is not None:
                        for raw_line in process.stdout:
                            line = raw_line.rstrip("\n")
                            if line:
                                _append_analysis_log(env_id, line)
                    return_code = process.wait()
                finally:
                    if process.stdout is not None:
                        process.stdout.close()
                result_path = Path(output_dir) / "summary.json"
                if return_code == 0 and result_path.is_file():
                    _update_analysis_env_state(
                        env_id,
                        status="succeeded",
                        finished_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        message="Local analysis complete.",
                    )
                else:
                    _update_analysis_env_state(
                        env_id,
                        status="failed",
                        finished_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        message=f"Local analysis failed with exit code {return_code}.",
                    )
            except Exception as exc:
                _append_analysis_log(env_id, f"[dashboard] local analysis failed: {exc}")
                _update_analysis_env_state(
                    env_id,
                    status="failed",
                    finished_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    message=f"Local analysis failed: {exc}",
                )
            finally:
                _invalidate_cache(analysis_cache, analysis_results_cache)
                _start_next_pending_run()

        thread = threading.Thread(target=_worker, daemon=True, name=f"analysis-{env_id}")
        thread.start()
        return True

    def _enqueue_local_analysis(env_id: str) -> str:
        request = _analysis_job_request_from_state(env_id)
        current = _analysis_env_state(env_id)
        is_active = str(current.get("status") or "").strip().lower() in {"queued", "starting", "running"}
        if not is_active:
            _start_local_analysis(env_id, request=request)
            return "started"
        queue_entry = _analysis_queue_entry_from_request(request)
        with analysis_state_lock:
            envs = analysis_state.setdefault("envs", {})
            current_payload = envs.get(env_id) if isinstance(envs, dict) else None
            payload = dict(current_payload) if isinstance(current_payload, dict) else _default_analysis_env_state(env_id)
            raw_pending = payload.get("pending_runs")
            pending_runs = list(raw_pending) if isinstance(raw_pending, list) else []
            pending_runs.append(queue_entry)
            payload["pending_runs"] = pending_runs
            payload["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            envs[env_id] = payload
            analysis_state["envs"] = envs
            _save_analysis_state(analysis_state)
        _invalidate_cache(analysis_cache)
        return "queued"

    def _resume_interrupted_local_analyses() -> None:
        snapshot = _snapshot_analysis_state()
        envs = snapshot.get("envs") if isinstance(snapshot, dict) else {}
        if not isinstance(envs, dict):
            return
        for env_id in SUPPORTED_ENVS:
            current = envs.get(env_id)
            if not isinstance(current, dict):
                continue
            status_text = str(current.get("status") or "").strip().lower()
            run_id = str(current.get("run_id") or "").strip()
            output_dir = str(current.get("output_dir") or "").strip()
            if status_text != "interrupted" or not run_id or not output_dir:
                continue
            try:
                request = _analysis_resume_request_from_state(env_id)
            except Exception as exc:
                _append_analysis_log(env_id, f"[dashboard] failed to prepare analysis resume: {exc}")
                _update_analysis_env_state(
                    env_id,
                    status="failed",
                    finished_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    message=f"Failed to prepare interrupted analysis resume: {exc}",
                )
                continue
            started = _start_local_analysis(env_id, request=request)
            if started:
                _append_analysis_log(env_id, "[dashboard] resuming interrupted local analysis in existing output directory.")

    def _statistics_state_path() -> Path:
        return project_root / "state" / "statistics_state.json"

    def _default_statistics_recalc_state() -> dict[str, object]:
        return {
            "status": "idle",
            "progress": 0.0,
            "completed": 0,
            "total": 0,
            "embed_progress": 0.0,
            "embed_completed": 0,
            "embed_total": 0,
            "trace_progress": 0.0,
            "trace_completed": 0,
            "trace_total": 0,
            "label_progress": 0.0,
            "label_completed": 0,
            "label_total": 0,
            "message": "",
            "updated_at": "",
            "start_date": "",
            "sample_seed": "0",
            "rows_by_env": {},
            "log_lines": [],
        }

    def _load_statistics_state() -> dict[str, object]:
        path = _statistics_state_path()
        if not path.is_file():
            return {"statistics": _default_statistics_recalc_state()}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        stats = payload.get("statistics")
        if not isinstance(stats, dict):
            stats = _default_statistics_recalc_state()
        merged = _default_statistics_recalc_state()
        merged.update(stats)
        payload["statistics"] = merged
        return payload

    def _save_statistics_state(payload: dict[str, object]) -> None:
        path = _statistics_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_path.replace(path)

    statistics_state: dict[str, object] = _load_statistics_state()
    statistics_tmux_session = "statistics-recalc"
    statistics_runner_script = project_root / "scripts" / "run_statistics_recalc.py"
    statistics_log_path = project_root / "state" / "statistics_recalc_runner.log"
    statistics_cancel_flag_path = project_root / "state" / "statistics_cancel.flag"
    profiler_tmux_session = "summary-profiler"
    profiler_runner_script = project_root / "scripts" / "run_summary_profiler.py"
    profiler_state_path = project_root / PROFILER_STATE_RELPATH
    analysis_tmux_session_prefix = "analysis-v7-"

    def _refresh_statistics_state_from_disk() -> None:
        with statistics_state_lock:
            statistics_state.clear()
            statistics_state.update(_load_statistics_state())

    def _snapshot_statistics_state() -> dict[str, object]:
        _refresh_statistics_state_from_disk()
        with statistics_state_lock:
            return json.loads(json.dumps(statistics_state))

    def _append_statistics_log(message: str) -> None:
        with statistics_state_lock:
            current = statistics_state.get("statistics")
            payload = current if isinstance(current, dict) else _default_statistics_recalc_state()
            payload = dict(payload)
            log_lines = payload.get("log_lines")
            logs = [str(item) for item in log_lines] if isinstance(log_lines, list) else []
            stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
            logs.append(f"[{stamp}] {message}")
            payload["log_lines"] = logs[-240:]
            payload["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            statistics_state["statistics"] = payload
            _save_statistics_state(statistics_state)

    def _make_statistics_progress_callback(*, started: float):
        def _progress(completed: int, total: int, label_ref: str) -> None:
            if statistics_cancel_event.is_set():
                raise RuntimeError("cancelled")
            phase = "label"
            detail = str(label_ref)
            if detail.startswith("EMBED:"):
                phase = "embed"
                detail = detail[len("EMBED:"):]
            elif detail.startswith("TRACE:"):
                phase = "trace"
                detail = detail[len("TRACE:"):]
            elif detail.startswith("LABEL:"):
                phase = "label"
                detail = detail[len("LABEL:"):]
            current_snapshot = _snapshot_statistics_state()
            current_stats = current_snapshot.get("statistics") if isinstance(current_snapshot, dict) else {}
            embed_total = int((current_stats or {}).get("embed_total") or 0)
            trace_total = int((current_stats or {}).get("trace_total") or 0)
            label_total = int((current_stats or {}).get("label_total") or 0)
            embed_completed = int((current_stats or {}).get("embed_completed") or 0)
            trace_completed = int((current_stats or {}).get("trace_completed") or 0)
            label_completed = int((current_stats or {}).get("label_completed") or 0)
            if phase == "embed":
                embed_total = max(embed_total, int(total))
                embed_completed = int(completed)
            elif phase == "trace":
                trace_total = max(trace_total, int(total))
                trace_completed = int(completed)
            else:
                label_total = max(label_total, int(total))
                label_completed = int(completed)
            overall_completed = embed_completed + trace_completed + label_completed
            overall_total = max(1, embed_total + trace_total + label_total)
            _update_statistics_state(
                status="running",
                progress=(overall_completed / overall_total),
                completed=int(overall_completed),
                total=int(overall_total),
                embed_progress=(embed_completed / max(1, embed_total)),
                embed_completed=int(embed_completed),
                embed_total=int(embed_total),
                trace_progress=(trace_completed / max(1, trace_total)),
                trace_completed=int(trace_completed),
                trace_total=int(trace_total),
                label_progress=(label_completed / max(1, label_total)),
                label_completed=int(label_completed),
                label_total=int(label_total),
                message=f"Processing {detail}",
            )
            _append_statistics_log(
                _format_tqdm_meter(
                    completed=int(completed),
                    total=max(1, int(total)),
                    elapsed_seconds=(time.monotonic() - started),
                    postfix=f"[{phase}] {detail}",
                )
            )
            _invalidate_cache(statistics_cache)
            _invalidate_cache(analysis_cache)

        return _progress

    def _format_tqdm_meter(*, completed: int, total: int, elapsed_seconds: float, postfix: str = "") -> str:
        if _tqdm_cls is not None:
            try:
                meter = _tqdm_cls.format_meter(
                    n=completed,
                    total=max(1, total),
                    elapsed=max(0.0, elapsed_seconds),
                )
            except Exception:
                meter = f"{completed}/{max(1, total)}"
        else:
            safe_total = max(1, total)
            pct = (completed / safe_total) * 100.0
            bar_width = 20
            filled = int((completed / safe_total) * bar_width)
            bar = ("#" * filled) + ("-" * (bar_width - filled))
            meter = f"{pct:5.1f}%|{bar}| {completed}/{safe_total}"
        return f"{meter} {postfix}".strip()

    def _update_statistics_state(**fields: object) -> None:
        with statistics_state_lock:
            current = statistics_state.get("statistics")
            payload = current if isinstance(current, dict) else _default_statistics_recalc_state()
            payload = dict(payload)
            payload.update(fields)
            payload["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            statistics_state["statistics"] = payload
            _save_statistics_state(statistics_state)

    def _statistics_tmux_running() -> bool:
        result = subprocess.run(
            ["tmux", "has-session", "-t", statistics_tmux_session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0

    def _snapshot_profiler_state() -> dict[str, object]:
        return json.loads(json.dumps(load_profiler_state(project_root)))

    def _profiler_tmux_running() -> bool:
        result = subprocess.run(
            ["tmux", "has-session", "-t", profiler_tmux_session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0

    def _start_profiler_run() -> bool:
        snapshot = _snapshot_profiler_state()
        if _profiler_tmux_running() or str(snapshot.get("status") or "").lower() == "running":
            return False
        profiler_state_path.parent.mkdir(parents=True, exist_ok=True)
        starting_state = load_profiler_state(project_root)
        starting_state.update(
            {
                "status": "starting",
                "message": "Starting summary profiler...",
                "progress": 0.0,
                "completed": 0,
                "total": 0,
                "active_run_id": "",
                "current_env": "",
                "current_variant": "",
                "current_task_id": "",
                "started_at": None,
                "finished_at": None,
                "log_lines": [],
            }
        )
        save_profiler_state(project_root, starting_state)
        command = [
            "tmux",
            "new-session",
            "-d",
            "-s",
            profiler_tmux_session,
            (
                f"cd {shlex.quote(str(project_root.parent))} && "
                f"export PYTHONPATH={shlex.quote(str(project_root / 'src'))}:{shlex.quote(str(project_root / 'src'))} && "
                f"{shlex.quote(str(project_root / 'venv' / 'learning' / 'bin' / 'python'))} "
                f"{shlex.quote(str(profiler_runner_script))} --project-root {shlex.quote(str(project_root))}"
            ),
        ]
        subprocess.run(command, check=True)
        return True

    def _cancel_profiler_run() -> bool:
        had_running = _profiler_tmux_running()
        subprocess.run(
            ["tmux", "kill-session", "-t", profiler_tmux_session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        subprocess.run(
            ["pkill", "-f", str(profiler_runner_script)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        state = load_profiler_state(project_root)
        state.update(
            {
                "status": "cancelled",
                "message": "Summary profiler cancelled.",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "current_env": "",
                "current_variant": "",
                "current_task_id": "",
            }
        )
        log_lines = [str(item) for item in (state.get("log_lines") or []) if str(item).strip()]
        log_lines.append(f"[{datetime.now(timezone.utc).isoformat()}] Cancellation requested.")
        state["log_lines"] = log_lines[-240:]
        save_profiler_state(project_root, state)
        return had_running

    def _analysis_tmux_status_by_env() -> dict[str, dict[str, object]]:
        tmux_env = dict(os.environ)
        tmux_env["TMUX"] = ""
        active_sessions: set[str] = set()
        list_result = subprocess.run(
            ["tmux", "ls"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            env=tmux_env,
        )
        if list_result.returncode == 0:
            for line in str(list_result.stdout or "").splitlines():
                session_name = str(line.split(":", 1)[0] or "").strip()
                if session_name:
                    active_sessions.add(session_name)
        statuses: dict[str, dict[str, object]] = {}
        for env_id in SUPPORTED_ENVS:
            session_name = f"{analysis_tmux_session_prefix}{env_id}"
            log_path = project_root / "data" / "analysis" / "supervised_consistency" / env_id / "regen-v7.log"
            is_running = session_name in active_sessions
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
            output_dir = Path(output_dir_text) if output_dir_text else None
            summary_exists = bool(output_dir and (output_dir / "summary.json").is_file())
            if not is_running and output_dir is not None and output_dir.exists() and not summary_exists:
                is_running = True
            statuses[env_id] = {
                "env_id": env_id,
                "session_name": session_name,
                "status": "running" if is_running else ("idle" if not log_lines else "finished"),
                "log_path": str(log_path),
                "log_lines": log_lines,
                "is_running": is_running,
                "output_dir": str(output_dir) if output_dir else "",
                "summary_exists": summary_exists,
            }
        return statuses

    def _start_statistics_recalculation(*, start_date: str = "", sample_seed: str = "0") -> bool:
        snapshot = _snapshot_statistics_state()
        current = snapshot.get("statistics")
        if _statistics_tmux_running() or (
            isinstance(current, dict) and str(current.get("status") or "").lower() in {"running", "cancelling"}
        ):
            return False
        start_date_text = str(start_date or "").strip()
        sample_seed_text = str(sample_seed or "0").strip() or "0"
        try:
            statistics_cancel_flag_path.unlink(missing_ok=True)
        except OSError:
            pass
        statistics_log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "tmux",
            "new-session",
            "-d",
            "-s",
            statistics_tmux_session,
            (
                f"cd {shlex.quote(str(project_root.parent))} && "
                f"export PYTHONPATH={shlex.quote(str(project_root / 'src'))}:{shlex.quote(str(project_root / 'src'))} && "
                f"{shlex.quote(str(project_root / 'venv' / 'learning' / 'bin' / 'python'))} "
                f"{shlex.quote(str(statistics_runner_script))} --mode full "
                f"--start-date {shlex.quote(start_date_text)} --sample-seed {shlex.quote(sample_seed_text)} "
                f"> {shlex.quote(str(statistics_log_path))} 2>&1"
            ),
        ]
        subprocess.run(command, check=True)
        _invalidate_cache(statistics_cache)
        _invalidate_cache(analysis_cache, analysis_results_cache)
        return True

    def _start_statistics_activation_recalculation() -> bool:
        snapshot = _snapshot_statistics_state()
        current = snapshot.get("statistics")
        if _statistics_tmux_running() or (
            isinstance(current, dict) and str(current.get("status") or "").lower() in {"running", "cancelling"}
        ):
            return False
        try:
            statistics_cancel_flag_path.unlink(missing_ok=True)
        except OSError:
            pass
        statistics_log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "tmux",
            "new-session",
            "-d",
            "-s",
            statistics_tmux_session,
            (
                f"cd {shlex.quote(str(project_root.parent))} && "
                f"export PYTHONPATH={shlex.quote(str(project_root / 'src'))}:{shlex.quote(str(project_root / 'src'))} && "
                f"{shlex.quote(str(project_root / 'venv' / 'learning' / 'bin' / 'python'))} "
                f"{shlex.quote(str(statistics_runner_script))} --mode activation "
                f"> {shlex.quote(str(statistics_log_path))} 2>&1"
            ),
        ]
        subprocess.run(command, check=True)
        _invalidate_cache(statistics_cache)
        _invalidate_cache(analysis_cache, analysis_results_cache)
        return True

    def _start_statistics_diversity_recalculation(*, start_date: str = "", sample_seed: str = "0") -> bool:
        snapshot = _snapshot_statistics_state()
        current = snapshot.get("statistics")
        if _statistics_tmux_running() or (
            isinstance(current, dict) and str(current.get("status") or "").lower() in {"running", "cancelling"}
        ):
            return False
        start_date_text = str(start_date or "").strip()
        sample_seed_text = str(sample_seed or "0").strip() or "0"
        try:
            statistics_cancel_flag_path.unlink(missing_ok=True)
        except OSError:
            pass
        statistics_log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "tmux",
            "new-session",
            "-d",
            "-s",
            statistics_tmux_session,
            (
                f"cd {shlex.quote(str(project_root.parent))} && "
                f"export PYTHONPATH={shlex.quote(str(project_root / 'src'))}:{shlex.quote(str(project_root / 'src'))} && "
                f"{shlex.quote(str(project_root / 'venv' / 'learning' / 'bin' / 'python'))} "
                f"{shlex.quote(str(statistics_runner_script))} --mode diversity "
                f"--start-date {shlex.quote(start_date_text)} --sample-seed {shlex.quote(sample_seed_text)} "
                f"> {shlex.quote(str(statistics_log_path))} 2>&1"
            ),
        ]
        subprocess.run(command, check=True)
        _invalidate_cache(statistics_cache)
        _invalidate_cache(analysis_cache, analysis_results_cache)
        return True

    ABLATION_IMPORTANCE_STALE_SECONDS = 10 * 60
    ablation_combined_tmux_session = "ablation-combined-plot"
    ablation_importance_tmux_session = "ablation-importance-plot"
    ablation_runner_script = project_root / "scripts" / "recalculate_ablations.py"

    def _ablation_importance_state_path() -> Path:
        return project_root / "state" / "ablation_importance_state.json"

    def _default_ablation_generation_state(*, idle_message: str = "Idle.") -> dict[str, object]:
        return {
            "status": "idle",
            "progress": 0.0,
            "message": idle_message,
            "updated_at": "",
            "started_at": "",
            "finished_at": "",
            "point_count": 0,
            "run_count": 0,
            "size_count": 0,
            "html": "",
            "log_lines": [],
        }

    def _load_ablation_importance_state() -> dict[str, object]:
        path = _ablation_importance_state_path()
        if not path.is_file():
            return {
                "importance": _default_ablation_generation_state(),
                "combined": _default_ablation_generation_state(),
            }
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        for key in ("importance", "combined"):
            current = payload.get(key)
            merged = _default_ablation_generation_state()
            if isinstance(current, dict):
                merged.update(current)
            payload[key] = merged
        return payload

    def _save_ablation_importance_state(payload: dict[str, object]) -> None:
        path = _ablation_importance_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_path.replace(path)

    def _parse_state_timestamp(value: object) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _ablation_importance_is_stale(payload: dict[str, object] | None) -> bool:
        state = payload if isinstance(payload, dict) else {}
        status = str(state.get("status") or "").strip().lower()
        if status not in {"running", "cancelling"}:
            return False
        latest_state_time = (
            _parse_state_timestamp(state.get("updated_at"))
            or _parse_state_timestamp(state.get("started_at"))
        )
        if latest_state_time is None:
            return True
        age_seconds = (datetime.now(timezone.utc) - latest_state_time).total_seconds()
        return age_seconds >= float(ABLATION_IMPORTANCE_STALE_SECONDS)

    def _ablation_generation_label(section: str) -> str:
        return "importance scatter" if section == "importance" else "combined line plot"

    def _ablation_tmux_session_name(section: str) -> str:
        return ablation_importance_tmux_session if section == "importance" else ablation_combined_tmux_session

    def _ablation_tmux_exists(section: str) -> bool:
        result = subprocess.run(
            ["tmux", "has-session", "-t", _ablation_tmux_session_name(section)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0

    def _ablation_tmux_running(section: str) -> bool:
        session_name = _ablation_tmux_session_name(section)
        if not _ablation_tmux_exists(section):
            return False
        result = subprocess.run(
            ["tmux", "list-panes", "-t", session_name, "-F", "#{pane_dead}"],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return False
        pane_values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return any(value == "0" for value in pane_values)

    ablation_importance_state: dict[str, object] = _load_ablation_importance_state()

    def _refresh_ablation_importance_state_from_disk() -> None:
        with ablation_importance_state_lock:
            ablation_importance_state.clear()
            ablation_importance_state.update(_load_ablation_importance_state())

    def _snapshot_ablation_importance_state() -> dict[str, object]:
        _refresh_ablation_importance_state_from_disk()
        with ablation_importance_state_lock:
            changed = False
            for key in ("importance", "combined"):
                current = ablation_importance_state.get(key)
                current_dict = current if isinstance(current, dict) else None
                if _ablation_importance_is_stale(current_dict) or (
                    isinstance(current_dict, dict)
                    and str(current_dict.get("status") or "").strip().lower() in {"running", "cancelling"}
                    and not _ablation_tmux_running(key)
                ):
                    payload = dict(current) if isinstance(current, dict) else _default_ablation_generation_state()
                    payload.update(
                        {
                            "status": "failed",
                            "progress": 0.0,
                            "message": f"Previous ablation {_ablation_generation_label(key)} generation became stale. Re-run it manually from /ablations.",
                            "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        }
                    )
                    payload["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                    ablation_importance_state[key] = payload
                    changed = True
            if changed:
                _save_ablation_importance_state(ablation_importance_state)
            return json.loads(json.dumps(ablation_importance_state))

    def _update_ablation_importance_state(*, section: str, **fields: object) -> None:
        with ablation_importance_state_lock:
            current = ablation_importance_state.get(section)
            payload = current if isinstance(current, dict) else _default_ablation_generation_state()
            payload = dict(payload)
            payload.update(fields)
            payload["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            ablation_importance_state[section] = payload
            _save_ablation_importance_state(ablation_importance_state)

    def _start_ablation_importance_generation() -> bool:
        snapshot = _snapshot_ablation_importance_state()
        current = snapshot.get("importance")
        if (
            isinstance(current, dict)
            and str(current.get("status") or "").lower() == "running"
            and not _ablation_importance_is_stale(current)
        ):
            return False
        if _ablation_tmux_running("importance"):
            return False
        if _ablation_tmux_exists("importance") and not _ablation_tmux_running("importance"):
            subprocess.run(
                ["tmux", "kill-session", "-t", ablation_importance_tmux_session],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        command = [
            "tmux",
            "new-session",
            "-d",
            "-s",
            ablation_importance_tmux_session,
            (
                f"cd {shlex.quote(str(project_root.parent))} && "
                f"export PYTHONPATH={shlex.quote(str(project_root / 'src'))}:{shlex.quote(str(project_root / 'src'))} && "
                f"{shlex.quote(str(project_root / 'venv' / 'learning' / 'bin' / 'python'))} "
                f"{shlex.quote(str(ablation_runner_script))} --project-root {shlex.quote(str(project_root))} --mode importance"
            ),
        ]
        subprocess.run(command, check=True)
        _invalidate_cache(ablations_cache)
        return True

    def _start_ablation_combined_generation() -> bool:
        snapshot = _snapshot_ablation_importance_state()
        current = snapshot.get("combined")
        if (
            isinstance(current, dict)
            and str(current.get("status") or "").lower() == "running"
            and not _ablation_importance_is_stale(current)
        ):
            return False
        if _ablation_tmux_running("combined"):
            return False
        if _ablation_tmux_exists("combined") and not _ablation_tmux_running("combined"):
            subprocess.run(
                ["tmux", "kill-session", "-t", ablation_combined_tmux_session],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

        command = [
            "tmux",
            "new-session",
            "-d",
            "-s",
            ablation_combined_tmux_session,
            (
                f"cd {shlex.quote(str(project_root.parent))} && "
                f"export PYTHONPATH={shlex.quote(str(project_root / 'src'))}:{shlex.quote(str(project_root / 'src'))} && "
                f"{shlex.quote(str(project_root / 'venv' / 'learning' / 'bin' / 'python'))} "
                f"{shlex.quote(str(ablation_runner_script))} --project-root {shlex.quote(str(project_root))} --mode combined"
            ),
        ]
        subprocess.run(command, check=True)
        _invalidate_cache(ablations_cache)
        return True

    def _cancel_ablation_generation(section: str) -> bool:
        normalized = str(section or "").strip().lower()
        if normalized not in {"combined", "importance"}:
            return False
        snapshot = _snapshot_ablation_importance_state()
        current = snapshot.get(normalized)
        current_dict = current if isinstance(current, dict) else {}
        is_running = _ablation_tmux_running(normalized) or (
            isinstance(current_dict, dict)
            and str(current_dict.get("status") or "").strip().lower() in {"running", "cancelling"}
        )
        if not is_running:
            return False
        subprocess.run(
            ["tmux", "kill-session", "-t", _ablation_tmux_session_name(normalized)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        label = _ablation_generation_label(normalized)
        _update_ablation_importance_state(
            section=normalized,
            status="cancelled",
            progress=0.0,
            message=f"{label.title()} generation cancelled.",
            finished_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        _update_ablation_importance_state(
            section=normalized,
            log_lines=[
                *(
                    [str(line) for line in current_dict.get("log_lines", [])][-239:]
                    if isinstance(current_dict.get("log_lines"), list)
                    else []
                ),
                f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Cancellation requested from /ablations.",
            ],
        )
        _invalidate_cache(ablations_cache)
        return True

    def _invalidate_cache(*caches: dict[str, object]) -> None:
        for cache in caches:
            cache["html"] = None
            cache["expires_at"] = 0.0

    _resume_interrupted_local_analyses()

    def _refresh_live_state() -> tuple[list, list]:
        return state_store.list_jobs_snapshot(), state_store.list_clients_snapshot()

    def _all_jobs():
        jobs, _clients = _refresh_live_state()
        return jobs

    def _all_clients():
        _jobs, clients = _refresh_live_state()
        return clients

    def _cached_analysis_results_by_env(*, now: float | None = None, limit_per_env: int = 64) -> dict[str, list]:
        import time

        current_time = time.monotonic() if now is None else now
        cached = analysis_results_cache.get("results_by_env")
        expires_at = float(analysis_results_cache.get("expires_at") or 0.0)
        if isinstance(cached, dict) and current_time < expires_at:
            return cached
        grouped: dict[str, list] = {env_id: [] for env_id in SUPPORTED_ENVS}
        all_results = list_supervised_consistency_results(
            project_root,
            limit=max(limit_per_env * len(SUPPORTED_ENVS), 256),
        )
        by_env_raw: dict[str, list] = {env_id: [] for env_id in SUPPORTED_ENVS}
        for result in all_results:
            env_key = str(result.env_id or "").strip().lower()
            if env_key not in by_env_raw:
                continue
            by_env_raw[env_key].append(result)
        for env_key, env_results in by_env_raw.items():
            env_rows = grouped[env_key]
            if env_results:
                env_rows.append(env_results[0])
        analysis_results_cache["results_by_env"] = grouped
        analysis_results_cache["expires_at"] = current_time + 30.0
        return grouped

    def _cached_ablation_plot_context(*, jobs: list, now: float | None = None) -> dict[tuple[str, str], dict[str, object]]:
        import time

        current_time = time.monotonic() if now is None else now
        cached = ablation_plot_context_cache.get("entries")
        expires_at = float(ablation_plot_context_cache.get("expires_at") or 0.0)
        if isinstance(cached, dict) and current_time < expires_at:
            return cached
        family_to_kind = {
            "supervised": "supervised",
            "unsupervised": "unsupervised",
        }
        tracked_catalog = _read_required_library_ablation_catalog(project_root)
        entries: dict[tuple[str, str], dict[str, object]] = {}
        for kind in ("supervised", "unsupervised"):
            for env_id in SUPPORTED_ENVS:
                source_root: Path | None
                source_label = ""
                source_display_name = ""
                libraries = [
                    item
                    for item in list_environment_libraries(project_root, env_id)
                    if str(item.workflow_kind or "").strip().lower() == kind
                ]
                if libraries:
                    source_display_name = libraries[0].run_id
                try:
                    source_root = Path(_canonical_ablation_pattern_run_dir(project_root=project_root, env_id=env_id, kind=kind))
                    source_label = "Canonical /library source"
                except Exception:
                    source_root = None
                if source_root is None:
                    try:
                        source_root = Path(_resolve_ablation_source_run_dir(project_root=project_root, env_id=env_id, kind=kind))
                    except Exception:
                        source_root = None
                    if source_root is not None:
                        source_label = "Latest library source"
                if not source_display_name and source_root is not None:
                    source_display_name = source_root.name
                tracked_job_id = ""
                family = family_to_kind.get(kind, "")
                if family:
                    tracked_job_id = str(tracked_catalog.get((env_id, family), "") or "").strip()
                entries[(kind, env_id)] = {
                    "source_root": source_root,
                    "source_label": source_label,
                    "source_display_name": source_display_name,
                    "tracked_job_id": tracked_job_id,
                    "runs": [],
                }
        for job in jobs:
            if str(job.spec.action or "").strip() != "evaluate-library-case":
                continue
            metadata = dict(job.spec.metadata or {})
            env_id = str(metadata.get("env_id") or "").strip().lower()
            kind = str(metadata.get("library_ablation_kind") or "").strip().lower()
            key = (kind, env_id)
            entry = entries.get(key)
            if entry is None:
                continue
            detail = load_library_ablation_job_detail(job, project_root=project_root)
            if detail is None:
                continue
            entry_runs = entry.get("runs")
            if isinstance(entry_runs, list):
                entry_runs.append((job, detail))
        for (kind, env_id), entry in entries.items():
            entry_runs = entry.get("runs")
            if isinstance(entry_runs, list):
                entry_runs.extend(
                    _discover_on_disk_ablation_runs(
                        project_root=project_root,
                        env_id=env_id,
                        kind=kind,
                    )
                )
        for entry in entries.values():
            entry_runs = entry.get("runs")
            if isinstance(entry_runs, list):
                entry_runs.sort(key=lambda item: item[0].updated_at, reverse=True)
        ablation_plot_context_cache["entries"] = entries
        ablation_plot_context_cache["expires_at"] = current_time + 30.0
        return entries

    def _job_by_id(job_id: str):
        for job in _all_jobs():
            if job.job_id == job_id:
                return job
        raise KeyError(job_id)

    def _client_by_id(client_id: str):
        for client in _all_clients():
            if client.client_id == client_id:
                return client
        raise KeyError(client_id)

    @app.get("/", response_class=HTMLResponse)
    async def home() -> HTMLResponse:
        body = render_home_page(
            public_access=public_access.get_state(),
            clients=_all_clients(),
            jobs=_all_jobs(),
            project_root=project_root,
        )
        return HTMLResponse(
            body,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    @app.get("/learning", response_class=HTMLResponse)
    async def learning_home() -> HTMLResponse:
        import time

        now = time.monotonic()
        cached_html = learning_cache.get("html")
        expires_at = float(learning_cache.get("expires_at") or 0.0)
        if isinstance(cached_html, str) and now < expires_at:
            return HTMLResponse(cached_html)
        body = render_learning_home_page(
            public_access=public_access.get_state(),
            clients=_all_clients(),
            jobs=_all_jobs(),
            project_root=project_root,
        )
        learning_cache["html"] = body
        learning_cache["expires_at"] = now + 300.0
        return HTMLResponse(body)

    @app.get("/clients", response_class=HTMLResponse)
    async def clients_home() -> HTMLResponse:
        return RedirectResponse(url="/", status_code=307)

    @app.get("/ablations", response_class=HTMLResponse)
    async def ablations_home() -> HTMLResponse:
        import time

        now = time.monotonic()
        cached_html = ablations_cache.get("html")
        expires_at = float(ablations_cache.get("expires_at") or 0.0)
        if isinstance(cached_html, str) and now < expires_at:
            return HTMLResponse(cached_html)
        jobs = _all_jobs()
        importance_snapshot = _snapshot_ablation_importance_state()
        body = render_ablations_home_page(
            jobs=jobs,
            clients=_all_clients(),
            project_root=project_root,
            plot_context=None,
            importance_state=importance_snapshot if isinstance(importance_snapshot, dict) else None,
        )
        ablations_cache["html"] = body
        ablations_cache["expires_at"] = now + 5.0
        return HTMLResponse(body)

    @app.get("/ablations/data/{kind}/{env_id}", response_class=HTMLResponse)
    async def ablation_data_page(kind: str, env_id: str) -> HTMLResponse:
        normalized_kind = str(kind or "").strip().lower()
        normalized_env_id = str(env_id or "").strip().lower()
        if normalized_kind not in {"supervised", "unsupervised"}:
            raise HTTPException(status_code=404, detail=f"Unknown ablation family {kind}")
        if normalized_env_id not in SUPPORTED_ENVS:
            raise HTTPException(status_code=404, detail=f"Unknown ablation environment {env_id}")
        import time

        cache_key = (normalized_kind, normalized_env_id)
        cache_entry = ablation_data_cache.get(cache_key) or {}
        cached_html = cache_entry.get("html")
        expires_at = float(cache_entry.get("expires_at") or 0.0)
        now = time.monotonic()
        if isinstance(cached_html, str) and now < expires_at:
            return HTMLResponse(cached_html)
        jobs = _all_jobs()
        body = render_ablation_data_page(
            kind=normalized_kind,
            env_id=normalized_env_id,
            jobs=jobs,
            project_root=project_root,
            plot_context=None,
        )
        ablation_data_cache[cache_key] = {"html": body, "expires_at": now + 30.0}
        return HTMLResponse(
            body
        )

    @app.get("/api/ui/ablations/plot.pdf")
    async def ablation_plot_pdf(kind: str | None = None, env_id: str | None = None, combined: str | None = None, importance: str | None = None) -> Response:
        combined_requested = str(combined or "").strip().lower() in {"1", "true", "yes", "on"}
        importance_requested = str(importance or "").strip().lower() in {"1", "true", "yes", "on"}
        jobs = _all_jobs()
        if importance_requested:
            ablation_cache = load_ablation_plot_cache(project_root)
            chart_data = build_combined_ablation_importance_scatter_data(
                jobs=jobs,
                project_root=project_root,
                plot_context=None,
                include_all=True,
                ablation_cache=ablation_cache if isinstance(ablation_cache, dict) else None,
            )
            pdf_bytes = render_ablation_importance_scatter_pdf_bytes(
                title=str(chart_data.get("title") or "Label importance vs training reward"),
                x_label=str(chart_data.get("x_label") or "Training reward"),
                y_label=str(chart_data.get("y_label") or "Normalized label importance"),
                rows=chart_data.get("rows") if isinstance(chart_data.get("rows"), list) else [],
                x_limits=(float(chart_data.get("min_reward") or -0.2), float(chart_data.get("max_reward") or 1.0)),
                y_limits=(float(chart_data.get("min_importance") or -0.2), float(chart_data.get("max_importance") or 1.0)),
            )
            filename = "ablations-importance.pdf"
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        if combined_requested:
            ablation_cache = load_ablation_plot_cache(project_root)
            chart_data = build_combined_ablation_chart_data(
                jobs=jobs,
                project_root=project_root,
                plot_context=None,
                include_all=True,
                ablation_cache=ablation_cache if isinstance(ablation_cache, dict) else None,
            )
            filename = "ablations-combined.pdf"
        else:
            normalized_kind = str(kind or "").strip().lower()
            normalized_env_id = str(env_id or "").strip().lower()
            if normalized_kind not in {"supervised", "unsupervised"}:
                raise HTTPException(status_code=404, detail=f"Unknown ablation family {kind}")
            if normalized_env_id not in SUPPORTED_ENVS:
                raise HTTPException(status_code=404, detail=f"Unknown ablation environment {env_id}")
            chart_data = build_ablation_chart_data_for_env_kind(
                jobs=jobs,
                project_root=project_root,
                env_id=normalized_env_id,
                kind=normalized_kind,
                plot_context=None,
                include_all=True,
            )
            filename = f"ablations-{normalized_env_id}-{normalized_kind}.pdf"
        pdf_bytes = render_ablation_plot_pdf_bytes(
            title=str(chart_data.get("title") or "Ablation plot"),
            x_label=str(chart_data.get("x_label") or "Labels used"),
            y_label=str(chart_data.get("y_label") or "Task success (%)"),
            series=chart_data.get("series") if isinstance(chart_data.get("series"), list) else [],
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/ui/analysis/plot.pdf")
    async def analysis_plot_pdf() -> Response:
        results_by_env: dict[str, list[object]] = {}
        for env_id in SUPPORTED_ENVS:
            results = list_supervised_consistency_results(project_root, env_id=env_id, limit=16)
            results_by_env[env_id] = results[:1]
        chart_data = build_combined_analysis_pattern_chart_data(results_by_env=results_by_env)
        pdf_bytes = render_analysis_plot_pdf_bytes(
            title=str(chart_data.get("title") or "Pattern-Based Similarity To Final Bin"),
            x_label=str(chart_data.get("x_label") or "Candidate percentile bin"),
            y_label=str(chart_data.get("y_label") or "Pattern-based similarity to final bin"),
            series=chart_data.get("series") if isinstance(chart_data.get("series"), list) else [],
            y_limits=(0.0, 1.0),
            x_tick_labels=chart_data.get("x_tick_labels") if isinstance(chart_data.get("x_tick_labels"), dict) else {},
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="analysis-pattern-similarity.pdf"'},
        )

    @app.get("/ablations/{job_id}", response_class=HTMLResponse)
    async def ablation_detail_page(job_id: str) -> HTMLResponse:
        job = state_store.get_job(str(job_id or "").strip())
        if job is None or str(job.spec.action or "").strip() != "evaluate-library-case":
            raise HTTPException(status_code=404, detail=f"Unknown ablation {job_id}")
        return HTMLResponse(render_ablation_detail_page(job=job, project_root=project_root))

    @app.get("/analysis", response_class=HTMLResponse)
    async def analysis_home(request: Request) -> HTMLResponse:
        import time

        now = time.monotonic()
        refresh_requested = str(request.query_params.get("refresh") or "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        cached_html = analysis_cache.get("html")
        expires_at = float(analysis_cache.get("expires_at") or 0.0)
        response_headers = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
        if not refresh_requested and isinstance(cached_html, str) and now < expires_at:
            return HTMLResponse(cached_html, headers=response_headers)
        analysis_jobs = _local_analysis_jobs()
        logs_by_job_id = {
            job.job_id: list(job.recent_logs or [])
            for job in analysis_jobs[:8]
        }
        results_by_env = _cached_analysis_results_by_env(now=now, limit_per_env=64)
        tmux_status_by_env = _analysis_tmux_status_by_env()
        analysis_snapshot = _snapshot_analysis_state()
        analysis_status_by_env = (
            analysis_snapshot.get("envs")
            if isinstance(analysis_snapshot, dict)
            else {}
        )
        body = render_analysis_home_page(
            jobs=analysis_jobs,
            results_by_env=results_by_env,
            project_root=project_root,
            logs_by_job_id=logs_by_job_id,
            tmux_status_by_env=tmux_status_by_env,
            analysis_status_by_env=analysis_status_by_env if isinstance(analysis_status_by_env, dict) else None,
        )
        analysis_cache["html"] = body
        analysis_cache["expires_at"] = now + (
            5.0
            if any(job.status.value in {"queued", "assigned", "starting", "running", "pausing", "resuming", "stopping"} for job in analysis_jobs)
            or any(bool(item.get("is_running")) for item in tmux_status_by_env.values())
            else 30.0
        )
        return HTMLResponse(body, headers=response_headers)

    @app.get("/analysis/env/{env_id}", response_class=HTMLResponse)
    async def analysis_environment_page(env_id: str) -> HTMLResponse:
        normalized_env = str(env_id or "").strip().lower()
        if normalized_env not in SUPPORTED_ENVS:
            raise HTTPException(status_code=404, detail=f"Unknown analysis environment {env_id}")
        jobs = _local_analysis_jobs()
        logs_by_job_id: dict[str, list[str]] = {}
        for job in jobs:
            if str((job.spec.metadata or {}).get("env_id") or "").strip().lower() != normalized_env:
                continue
            logs_by_job_id[job.job_id] = list(job.recent_logs or [])
        results_by_env = _cached_analysis_results_by_env(limit_per_env=24)
        tmux_status_by_env = _analysis_tmux_status_by_env()
        return HTMLResponse(
            render_analysis_environment_page(
                env_id=normalized_env,
                jobs=jobs,
                project_root=project_root,
                env_results=list(results_by_env.get(normalized_env) or []),
                logs_by_job_id=logs_by_job_id,
                tmux_status=dict(tmux_status_by_env.get(normalized_env) or {}),
            )
        )

    @app.get("/statistics", response_class=HTMLResponse)
    async def statistics_home() -> HTMLResponse:
        response_headers = {"Cache-Control": "no-store, max-age=0"}
        stats_state = _snapshot_statistics_state()
        recalc_state = stats_state.get("statistics") if isinstance(stats_state, dict) else {}
        rows_by_env = {}
        if isinstance(recalc_state, dict):
            maybe_rows = recalc_state.get("rows_by_env")
            if isinstance(maybe_rows, dict):
                rows_by_env = maybe_rows
        body = render_statistics_page(
            project_root=project_root,
            rows_by_env=rows_by_env if isinstance(rows_by_env, dict) else {},
            diversity_state=recalc_state if isinstance(recalc_state, dict) else {},
        )
        return HTMLResponse(body, headers=response_headers)

    @app.get("/profiler", response_class=HTMLResponse)
    async def profiler_home() -> HTMLResponse:
        return HTMLResponse(
            render_profiler_page(
                state=_snapshot_profiler_state(),
            ),
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/api/ui/statistics/status", response_class=JSONResponse)
    async def statistics_status() -> JSONResponse:
        return JSONResponse(_snapshot_statistics_state())

    @app.get("/api/ui/profiler/status", response_class=JSONResponse)
    async def profiler_status() -> JSONResponse:
        return JSONResponse(_snapshot_profiler_state())

    @app.get("/api/ui/ablations/status", response_class=JSONResponse)
    async def ablations_status() -> JSONResponse:
        return JSONResponse(_snapshot_ablation_importance_state())

    @app.get("/analysis/{analysis_run_id}", response_class=HTMLResponse)
    async def analysis_result_page(analysis_run_id: str) -> HTMLResponse:
        result = load_supervised_consistency_result(project_root, analysis_run_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Unknown analysis run {analysis_run_id}")
        return HTMLResponse(
            render_supervised_consistency_result_page(
                result=result,
                project_root=project_root,
            )
        )

    @app.get("/rewards", response_class=HTMLResponse)
    async def rewards_home() -> HTMLResponse:
        import time

        now = time.monotonic()
        cached_html = rewards_cache.get("html")
        expires_at = float(rewards_cache.get("expires_at") or 0.0)
        if isinstance(cached_html, str) and now < expires_at:
            return HTMLResponse(cached_html)
        body = render_rewards_home_page(
            jobs=[],
            clients=[],
            project_root=project_root,
        )
        rewards_cache["html"] = body
        rewards_cache["expires_at"] = now + 5.0
        return HTMLResponse(body)

    @app.get("/videos", response_class=HTMLResponse)
    async def videos_page() -> HTMLResponse:
        import time

        now = time.monotonic()
        cached_html = videos_cache.get("html")
        expires_at = float(videos_cache.get("expires_at") or 0.0)
        if isinstance(cached_html, str) and now < expires_at:
            return HTMLResponse(cached_html)
        body = render_videos_page(project_root=project_root)
        videos_cache["html"] = body
        videos_cache["expires_at"] = now + 300.0
        return HTMLResponse(body)

    @app.get("/api/ui/videos/overlay", response_class=JSONResponse)
    async def video_overlay_payload(task_result_path: str, view: str = "askew") -> JSONResponse:
        target = _resolve_project_file(project_root, task_result_path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Unknown task result")
        view_key = str(view or "askew").strip().lower() or "askew"
        for env_id, entries in list_video_render_results(project_root).items():
            for entry in select_video_showcase_entries(entries):
                if entry.task_result_path.resolve() != target.resolve():
                    continue
                selection = load_annotation_video_selection(project_root, env_id)
                payload = build_annotation_overlay_payload(
                    project_root=project_root,
                    entry=entry,
                    view=view_key,
                    selected_labels=list(selection.get("selected_labels") or []),
                )
                return JSONResponse(payload or {}, headers={"Cache-Control": "no-store, max-age=0"})
        raise HTTPException(status_code=404, detail="Overlay payload not available")

    @app.get("/api/ui/videos/download-overlay", response_class=FileResponse)
    async def video_overlay_download(
        task_result_path: str,
        view: str = "askew",
        confidence_threshold: float = 0.0,
    ) -> FileResponse:
        target = _resolve_project_file(project_root, task_result_path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Unknown task result")
        view_key = str(view or "askew").strip().lower() or "askew"
        for env_id, entries in list_video_render_results(project_root).items():
            for entry in select_video_showcase_entries(entries):
                if entry.task_result_path.resolve() != target.resolve():
                    continue
                selection = load_annotation_video_selection(project_root, env_id)
                export_path = export_annotation_overlay_video(
                    project_root=project_root,
                    entry=entry,
                    view=view_key,
                    selected_labels=list(selection.get("selected_labels") or []),
                    confidence_threshold=float(confidence_threshold or 0.0),
                )
                filename = f"{entry.task_id}-overlay.mp4".replace("/", "_")
                return FileResponse(export_path, filename=filename, media_type="video/mp4")
        raise HTTPException(status_code=404, detail="Overlay payload not available")

    @app.get("/library", response_class=HTMLResponse)
    async def library_page() -> HTMLResponse:
        import time

        now = time.monotonic()
        calibration_snapshot = _reconcile_all_ensemble_calibrations()
        jobs = _all_jobs()
        cached_html = library_cache.get("html")
        expires_at = float(library_cache.get("expires_at") or 0.0)
        if isinstance(cached_html, str) and now < expires_at:
            return HTMLResponse(cached_html)
        body = render_library_page(
            project_root=project_root,
            ensemble_calibration_state=calibration_snapshot,
            jobs=jobs,
        )
        library_cache["html"] = body
        library_cache["expires_at"] = now + (
            5.0
            if (
                _has_active_ensemble_calibrations(calibration_snapshot)
                or _has_active_library_deepphy_eval_jobs(jobs)
            )
            else 60.0
        )
        return HTMLResponse(body)

    @app.get("/summaries", response_class=HTMLResponse)
    async def summaries_page() -> HTMLResponse:
        import time

        now = time.monotonic()
        cached_html = summaries_cache.get("html")
        expires_at = float(summaries_cache.get("expires_at") or 0.0)
        if isinstance(cached_html, str) and now < expires_at:
            return HTMLResponse(cached_html)
        statuses = list_task_summary_statuses(project_root)
        entries_by_env = list_completed_task_summaries(project_root)
        body = render_summaries_page(
            project_root=project_root,
            statuses=statuses,
            entries_by_env=entries_by_env,
            config=load_task_summary_config(project_root),
        )
        summaries_cache["html"] = body
        summaries_cache["expires_at"] = now + (
            5.0
            if any(str(statuses.get(env_id, {}).get("status") or "").strip().lower() == "running" for env_id in SUPPORTED_ENVS)
            else 300.0
        )
        return HTMLResponse(body)

    @app.get("/summaries/env/{env_id}", response_class=HTMLResponse)
    async def summaries_environment_page(env_id: str) -> HTMLResponse:
        normalized_env = str(env_id or "").strip().lower()
        if normalized_env not in SUPPORTED_ENVS:
            raise HTTPException(status_code=404, detail=f"Unknown summaries environment {env_id}")
        statuses = list_task_summary_statuses(project_root)
        entries_by_env = list_completed_task_summaries(project_root)
        return HTMLResponse(
            render_summaries_environment_page(
                project_root=project_root,
                env_id=normalized_env,
                status=dict(statuses.get(normalized_env, {})),
                entries=list(entries_by_env.get(normalized_env, [])),
            )
        )

    @app.get("/env/{env_id}", response_class=HTMLResponse)
    async def environment_page(env_id: str) -> HTMLResponse:
        normalized = str(env_id or "").strip().lower()
        if normalized not in SUPPORTED_ENVS:
            raise HTTPException(status_code=404, detail=f"Unknown environment {env_id}")
        return HTMLResponse(
            render_environment_page(
                env_id=normalized,
                clients=_all_clients(),
                jobs=_all_jobs(),
                queue_settings=state_store.get_queue_settings(),
                project_root=project_root,
            )
        )

    @app.get("/library-ablations", response_class=HTMLResponse)
    async def library_ablations_page() -> HTMLResponse:
        return RedirectResponse(url="/ablations", status_code=307)

    @app.get("/env/{env_id}/datasets", response_class=HTMLResponse)
    async def environment_datasets_page(env_id: str) -> HTMLResponse:
        normalized = str(env_id or "").strip().lower()
        if normalized not in SUPPORTED_ENVS:
            raise HTTPException(status_code=404, detail=f"Unknown environment {env_id}")
        return HTMLResponse(
            render_environment_datasets_page(
                env_id=normalized,
                datasets=list_environment_datasets(project_root, normalized),
            )
        )

    @app.get("/env/{env_id}/datasets/{dataset_key}", response_class=HTMLResponse)
    async def environment_dataset_page(env_id: str, dataset_key: str) -> HTMLResponse:
        normalized = str(env_id or "").strip().lower()
        if normalized not in SUPPORTED_ENVS:
            raise HTTPException(status_code=404, detail=f"Unknown environment {env_id}")
        detail = get_environment_dataset(project_root, normalized, dataset_key)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"Unknown dataset {dataset_key}")
        return HTMLResponse(render_environment_dataset_page(dataset=detail))

    @app.get("/env/{env_id}/datasets/{dataset_key}/traces/{split_name}/{sample_name}", response_class=HTMLResponse)
    async def environment_dataset_trace_page(
        env_id: str,
        dataset_key: str,
        split_name: str,
        sample_name: str,
    ) -> HTMLResponse:
        normalized = str(env_id or "").strip().lower()
        if normalized not in SUPPORTED_ENVS:
            raise HTTPException(status_code=404, detail=f"Unknown environment {env_id}")
        detail = get_environment_dataset_trace(project_root, normalized, dataset_key, split_name, sample_name)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"Unknown trace {dataset_key}/{split_name}/{sample_name}")
        return HTMLResponse(render_environment_dataset_trace_page(trace=detail))

    @app.get("/env/{env_id}/libraries", response_class=HTMLResponse)
    async def environment_libraries_page(env_id: str) -> HTMLResponse:
        normalized = str(env_id or "").strip().lower()
        if normalized not in SUPPORTED_ENVS:
            raise HTTPException(status_code=404, detail=f"Unknown environment {env_id}")
        return HTMLResponse(
            render_environment_libraries_page(
                env_id=normalized,
                libraries=list_environment_libraries(project_root, normalized),
            )
        )

    @app.get("/env/{env_id}/rewards", response_class=HTMLResponse)
    async def environment_rewards_page(env_id: str) -> HTMLResponse:
        normalized = str(env_id or "").strip().lower()
        if normalized not in SUPPORTED_ENVS:
            raise HTTPException(status_code=404, detail=f"Unknown environment {env_id}")
        return HTMLResponse(
            render_environment_rewards_page(
                env_id=normalized,
                rewards=list_environment_reward_results(project_root, normalized),
            )
        )

    @app.get("/env/{env_id}/rewards/{reward_kind}", response_class=HTMLResponse)
    async def environment_reward_runs_page(env_id: str, reward_kind: str) -> HTMLResponse:
        normalized = str(env_id or "").strip().lower()
        normalized_kind = str(reward_kind or "").strip().lower()
        if normalized not in SUPPORTED_ENVS:
            raise HTTPException(status_code=404, detail=f"Unknown environment {env_id}")
        rewards = [
            item
            for item in list_environment_reward_results(project_root, normalized)
            if str(item.reward_kind or "").strip().lower() == normalized_kind
        ]
        return HTMLResponse(
            render_environment_reward_runs_page(
                env_id=normalized,
                reward_kind=normalized_kind,
                rewards=rewards,
            )
        )

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    async def job_page(job_id: str, reward_metric: str = "train") -> HTMLResponse:
        try:
            job = _job_by_id(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id}") from exc
        return HTMLResponse(
            render_job_page(
                job=job,
                jobs=_all_jobs(),
                clients=_all_clients(),
                project_root=project_root,
                log_lines=state_store.read_job_logs(job_id, limit=400),
                reward_metric=reward_metric,
            )
        )

    @app.get("/required-runs/{env_id}/{family_slug}", response_class=HTMLResponse)
    async def required_variant_overview_page(env_id: str, family_slug: str) -> HTMLResponse:
        normalized_env = str(env_id or "").strip().lower()
        if normalized_env not in SUPPORTED_ENVS:
            raise HTTPException(status_code=404, detail=f"Unknown environment {env_id}")
        family = required_family_from_slug(family_slug)
        if family is None:
            raise HTTPException(status_code=404, detail=f"Unknown required family {family_slug}")
        return HTMLResponse(
            render_required_variant_page(
                env_id=normalized_env,
                family=family,
                jobs=_all_jobs(),
                clients=_all_clients(),
                project_root=project_root,
                queue_settings=state_store.get_queue_settings(),
            )
        )

    @app.get("/jobs/{job_id}/tasks/{task_id:path}", response_class=HTMLResponse)
    async def evaluation_task_page(job_id: str, task_id: str) -> HTMLResponse:
        try:
            job = _job_by_id(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id}") from exc
        task = get_evaluation_task_detail(job, task_id, project_root=project_root)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Unknown task {task_id} for job {job_id}")
        return HTMLResponse(
            render_evaluation_task_page(
                job=job,
                task=task,
                clients=_all_clients(),
                project_root=project_root,
            )
        )

    @app.get("/clients/{client_id}", response_class=HTMLResponse)
    async def client_page(client_id: str) -> HTMLResponse:
        try:
            client = _client_by_id(client_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown client {client_id}") from exc
        jobs = _all_jobs()
        client_jobs = [job for job in jobs if job.assigned_client_id == client_id]
        log_lookup = {job.job_id: state_store.read_job_logs(job.job_id, limit=500) for job in client_jobs[:24]}
        return HTMLResponse(
            render_client_page(
                client=client,
                jobs=client_jobs,
                log_lookup=log_lookup,
                project_root=project_root,
            )
        )

    @app.get("/ui/files", response_model=None)
    async def ui_file(path: str):
        target = _resolve_project_file(project_root, path)
        if target.is_dir():
            children = sorted(item.name for item in target.iterdir())
            return PlainTextResponse(
                "\n".join(children) if children else "(empty directory)",
                headers={"Cache-Control": "no-store, max-age=0"},
            )
        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return FileResponse(target, headers={"Cache-Control": "no-store, max-age=0"})
        return PlainTextResponse(content, headers={"Cache-Control": "no-store, max-age=0"})

    @app.post("/ui/{path:path}")
    async def proxy_ui_action(path: str, request: Request) -> Response:
        body = await request.body()
        body_text = body.decode("utf-8", errors="ignore")
        parsed = parse_qs(body_text, keep_blank_values=True)
        next_url = _resolve_next_url(parsed.get("next", ["/"])[0], default="/")

        def _backend_headers() -> dict[str, str]:
            headers = {"content-type": "application/json"}
            api_key = str(public_access.get_state().api_key or "").strip()
            if api_key:
                headers["authorization"] = f"Bearer {api_key}"
            return headers

        def _post_json(
            url: str,
            payload: dict | None = None,
            *,
            timeout_seconds: float = 30.0,
        ) -> dict[str, object] | None:
            raw = json.dumps(payload or {}).encode("utf-8")
            req = urllib_request.Request(
                url,
                data=raw,
                headers=_backend_headers(),
                method="POST",
            )
            with urllib_request.urlopen(req, timeout=timeout_seconds) as response:
                response_body = response.read()
            if not response_body:
                return None
            try:
                parsed_body = json.loads(response_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None
            return parsed_body if isinstance(parsed_body, dict) else None

        segments = [segment for segment in str(path or "").split("/") if segment]
        try:
            if segments == ["public-access"]:
                mode = (parsed.get("mode", [""])[0] or "").strip().lower()
                current = public_access.get_state()
                enabled = current.desired_enabled
                if mode == "toggle":
                    enabled = not current.desired_enabled
                elif mode == "save":
                    enabled = current.desired_enabled
                else:
                    enabled = _coerce_form_bool(parsed.get("enabled", [""])[0] if parsed.get("enabled") else "")
                api_key = (parsed.get("api_key", [""])[0] or "").strip()
                payload: dict[str, object] = {"enabled": enabled}
                if api_key:
                    payload["api_key"] = api_key
                _post_json(f"{backend_root}/api/v1/public-access", payload)
            elif len(segments) == 3 and segments[0] == "jobs" and segments[2] == "target-client":
                _post_json(
                    f"{backend_root}/api/v1/jobs/{segments[1]}/preferred-client",
                    {"preferred_client_id": (parsed.get("preferred_client_id", [""])[0] or "").strip() or None},
                )
            elif len(segments) == 3 and segments[0] == "clients" and segments[2] == "settings":
                raw = (parsed.get("max_concurrency", ["4"])[0] or "4").strip()
                _post_json(
                    f"{backend_root}/api/v1/clients/{segments[1]}/settings",
                    {"max_concurrency": max(1, int(raw))},
                )
            elif len(segments) == 3 and segments[0] == "clients" and segments[2] == "shutdown":
                _post_json(f"{backend_root}/api/v1/clients/{segments[1]}/shutdown")
            elif len(segments) == 3 and segments[0] == "jobs" and segments[2] in {"pause", "resume", "requeue", "cancel"}:
                _post_json(f"{backend_root}/api/v1/jobs/{segments[1]}/{segments[2]}")
            elif len(segments) == 3 and segments[0] == "jobs" and segments[2] == "workers":
                raw = (parsed.get("workers", ["1"])[0] or "1").strip()
                _post_json(
                    f"{backend_root}/api/v1/jobs/{segments[1]}/workers",
                    {"workers": max(1, int(raw))},
                )
            elif len(segments) == 3 and segments[0] == "env" and segments[2] == "compute-supervised-importance":
                _post_json(f"{backend_root}/api/v1/env/{segments[1]}/compute-supervised-importance")
            elif len(segments) == 3 and segments[0] == "env" and segments[2] == "attempt-limit":
                raw = (parsed.get("pooltool_attempt_limit", ["8"])[0] or "8").strip()
                _post_json(
                    f"{backend_root}/api/v1/queue/settings",
                    {"pooltool_attempt_limit": max(1, int(raw))},
                )
            elif segments == ["statistics", "recalculate"]:
                start_date = (parsed.get("start_date", [""])[0] or "").strip()
                sample_seed = (parsed.get("sample_seed", ["0"])[0] or "0").strip() or "0"
                if start_date:
                    try:
                        datetime.strptime(start_date, "%Y-%m-%d")
                    except ValueError as exc:
                        raise ValueError("start_date must be YYYY-MM-DD") from exc
                try:
                    int(sample_seed)
                except ValueError as exc:
                    raise ValueError("sample_seed must be an integer") from exc
                _start_statistics_recalculation(start_date=start_date, sample_seed=sample_seed)
            elif segments == ["statistics", "recalculate-activation"]:
                _start_statistics_activation_recalculation()
            elif segments == ["statistics", "recalculate-diversity"]:
                start_date = (parsed.get("start_date", [""])[0] or "").strip()
                sample_seed = (parsed.get("sample_seed", ["0"])[0] or "0").strip() or "0"
                if start_date:
                    try:
                        datetime.strptime(start_date, "%Y-%m-%d")
                    except ValueError as exc:
                        raise ValueError("start_date must be YYYY-MM-DD") from exc
                try:
                    int(sample_seed)
                except ValueError as exc:
                    raise ValueError("sample_seed must be an integer") from exc
                _start_statistics_diversity_recalculation(start_date=start_date, sample_seed=sample_seed)
            elif segments == ["statistics", "cancel"]:
                snapshot = _snapshot_statistics_state()
                current = snapshot.get("statistics")
                is_running = _statistics_tmux_running() or (
                    isinstance(current, dict) and str(current.get("status") or "").lower() in {"running", "cancelling"}
                )
                if is_running:
                    statistics_cancel_flag_path.parent.mkdir(parents=True, exist_ok=True)
                    statistics_cancel_flag_path.write_text("cancel\n", encoding="utf-8")
                    subprocess.run(
                        ["tmux", "kill-session", "-t", statistics_tmux_session],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                    _update_statistics_state(
                        status="cancelled",
                        progress=0.0,
                        message="Statistics recalculation cancelled.",
                    )
                    _append_statistics_log("Cancellation requested.")
                    _invalidate_cache(statistics_cache)
                    _invalidate_cache(analysis_cache)
            elif segments == ["profiler", "run"]:
                _start_profiler_run()
            elif segments == ["profiler", "cancel"]:
                _cancel_profiler_run()
            elif segments == ["ablations", "generate-importance"]:
                _start_ablation_importance_generation()
                _invalidate_cache(ablations_cache)
            elif segments == ["ablations", "generate-combined"]:
                _start_ablation_combined_generation()
                _invalidate_cache(ablations_cache)
            elif segments == ["ablations", "cancel"]:
                section = (parsed.get("section", [""])[0] or "").strip().lower()
                _cancel_ablation_generation(section)
                _invalidate_cache(ablations_cache)
            elif segments == ["library", "recalculate"]:
                recalculate_canonical_libraries(project_root)
                _invalidate_cache(library_cache)
            elif segments == ["library", "ensemble-calibration", "start"]:
                env_id = (parsed.get("env_id", [""])[0] or "").strip().lower()
                kind = (parsed.get("kind", [""])[0] or "").strip().lower()
                if env_id not in SUPPORTED_ENVS:
                    raise ValueError(f"Unknown environment {env_id}")
                if kind not in ENSEMBLE_CALIBRATION_KINDS:
                    raise ValueError(f"Unsupported ensemble calibration kind {kind}")
                trials = max(1, int((parsed.get("trials", ["0"])[0] or "0").strip() or "0"))
                random_init_trials = max(
                    1,
                    int((parsed.get("random_init_trials", ["0"])[0] or "0").strip() or "0"),
                )
                workers_raw = (parsed.get("workers", [""])[0] or "").strip()
                workers = max(
                    1,
                    int(
                        workers_raw
                        or str(_default_ensemble_calibration_workers(env_id))
                    ),
                )
                seed = max(0, int((parsed.get("seed", ["0"])[0] or "0").strip()))
                model = (parsed.get("model", [""])[0] or "").strip()
                request_payload = _build_ensemble_calibration_request(
                    env_id=env_id,
                    kind=kind,
                    trials=trials,
                    random_init_trials=random_init_trials,
                    workers=workers,
                    seed=seed,
                    model=model,
                )
                _start_ensemble_calibration(env_id, kind, request_payload=request_payload)
                _invalidate_cache(library_cache)
            elif segments == ["library", "ensemble-calibration", "cancel"]:
                env_id = (parsed.get("env_id", [""])[0] or "").strip().lower()
                kind = (parsed.get("kind", [""])[0] or "").strip().lower()
                if env_id not in SUPPORTED_ENVS:
                    raise ValueError(f"Unknown environment {env_id}")
                if kind not in ENSEMBLE_CALIBRATION_KINDS:
                    raise ValueError(f"Unsupported ensemble calibration kind {kind}")
                _cancel_ensemble_calibration(env_id, kind)
                _invalidate_cache(library_cache)
            elif segments == ["library", "deepphy-eval", "start"]:
                env_id = (parsed.get("env_id", [""])[0] or "").strip().lower()
                kind = (parsed.get("kind", [""])[0] or "").strip().lower()
                phase = (parsed.get("phase", ["baseline"])[0] or "baseline").strip().lower()
                if env_id not in SUPPORTED_ENVS:
                    raise ValueError(f"Unknown environment {env_id}")
                if kind not in ENSEMBLE_CALIBRATION_KINDS:
                    raise ValueError(f"Unsupported library kind {kind}")
                if phase not in {"baseline", "calibrated"}:
                    raise ValueError(f"Unsupported eval phase {phase}")
                _start_library_deepphy_eval(env_id=env_id, kind=kind, phase=phase)
                _invalidate_cache(library_cache)
            elif segments == ["rewards", "goals", "save"]:
                state = _reward_load_state()
                state["goal_texts"] = [
                    (parsed.get(f"goal_{index:02d}", [""])[0] or "").strip()
                    for index in range(1, 11)
                ]
                _reward_save_state(state)
                _invalidate_cache(rewards_cache)
            elif segments == ["rewards", "single-scene", "randomize"]:
                state = _reward_load_state()
                current = dict(state.get("single_scene") or {}) if isinstance(state.get("single_scene"), dict) else {}
                next_seed = max(0, int(current.get("scene_seed") or 0)) + 1
                scene_path = _reward_start_random_scene(next_seed)
                current["scene_seed"] = next_seed
                current["scene_path"] = scene_path
                state["single_scene"] = current
                _reward_save_state(state)
                _invalidate_cache(rewards_cache)
            elif segments == ["rewards", "single-scene", "start"]:
                state = _reward_load_state()
                current = dict(state.get("single_scene") or {}) if isinstance(state.get("single_scene"), dict) else {}
                scene_path = str(current.get("scene_path") or "").strip()
                if not scene_path or not Path(scene_path).is_file():
                    raise ValueError("Create a random scene first.")
                goal_text = (parsed.get("goal_text", [""])[0] or "").strip()
                if not goal_text:
                    raise ValueError("Single-scene goal text is required.")
                budget = max(1, int((parsed.get("budget", ["250"])[0] or "250").strip()))
                seed = max(0, int((parsed.get("seed", ["0"])[0] or "0").strip()))
                model = (parsed.get("model", [""])[0] or "").strip()
                run_id = _reward_make_run_id("reward-single-scene")
                argv = [
                    "-m",
                    "dashboard.workflows",
                    "pooltool-reward-single-scene",
                    "--source-run-dir",
                    str(_reward_source_run_dir()),
                    "--scene-path",
                    scene_path,
                    "--goal-text",
                    goal_text,
                    "--budget",
                    str(budget),
                    "--seed",
                    str(seed),
                    "--ollama-base-url",
                    _reward_default_base_url_local(),
                    "--model-timeout",
                    str(_reward_default_timeout_local()),
                    "--run-id",
                    run_id,
                ]
                if model:
                    argv.extend(["--model", model])
                if _reward_launch_tmux("single_scene", argv):
                    current.update({
                        "goal_text": goal_text,
                        "budget": budget,
                        "seed": seed,
                        "model": model,
                        "run_id": run_id,
                    })
                    state["single_scene"] = current
                    _reward_save_state(state)
                _invalidate_cache(rewards_cache)
            elif segments == ["rewards", "single-scene", "cancel"]:
                _reward_cancel_tmux("single_scene")
                _invalidate_cache(rewards_cache)
            elif segments == ["rewards", "optimization-suite", "start"]:
                state = _reward_load_state()
                goals = _reward_require_full_goal_set(state)
                current = dict(state.get("optimization_suite") or {}) if isinstance(state.get("optimization_suite"), dict) else {}
                scene_count = max(1, int((parsed.get("scene_count", ["10"])[0] or "10").strip()))
                seed_count = max(1, int((parsed.get("seed_count", ["5"])[0] or "5").strip()))
                budgets_csv = (parsed.get("budgets_csv", ["10,50,100,250,500,1000"])[0] or "").strip()
                budgets: list[int] = []
                for part in budgets_csv.split(","):
                    text = part.strip()
                    if not text:
                        continue
                    budgets.append(max(1, int(text)))
                model = (parsed.get("model", [""])[0] or "").strip()
                run_id = _reward_make_run_id("reward-opt-suite")
                argv = [
                    "-m",
                    "dashboard.workflows",
                    "pooltool-reward-optimize-suite",
                    "--source-run-dir",
                    str(_reward_source_run_dir()),
                    "--scene-count",
                    str(scene_count),
                    "--seed-count",
                    str(seed_count),
                    "--ollama-base-url",
                    _reward_default_base_url_local(),
                    "--model-timeout",
                    str(_reward_default_timeout_local()),
                    "--run-id",
                    run_id,
                ]
                for budget in (budgets or [10, 50, 100, 250, 500, 1000]):
                    argv.extend(["--budget", str(int(budget))])
                for goal_text in goals:
                    argv.extend(["--goal-text", goal_text])
                if model:
                    argv.extend(["--model", model])
                if _reward_launch_tmux("optimization_suite", argv):
                    current.update({
                        "scene_count": scene_count,
                        "seed_count": seed_count,
                        "budgets": budgets or [10, 50, 100, 250, 500, 1000],
                        "model": model,
                        "run_id": run_id,
                    })
                    state["optimization_suite"] = current
                    _reward_save_state(state)
                _invalidate_cache(rewards_cache)
            elif segments == ["rewards", "optimization-suite", "cancel"]:
                _reward_cancel_tmux("optimization_suite")
                _invalidate_cache(rewards_cache)
            elif segments == ["rewards", "value-model-sweep", "start"]:
                state = _reward_load_state()
                goals = _reward_require_full_goal_set(state)
                current = dict(state.get("value_model_sweep") or {}) if isinstance(state.get("value_model_sweep"), dict) else {}
                model = (parsed.get("model", [""])[0] or "").strip()
                run_id = _reward_make_run_id("reward-value-sweep")
                argv = [
                    "-m",
                    "dashboard.workflows",
                    "pooltool-reward-value-model-sweep",
                    "--source-run-dir",
                    str(_reward_source_run_dir()),
                    "--ollama-base-url",
                    _reward_default_base_url_local(),
                    "--model-timeout",
                    str(_reward_default_timeout_local()),
                    "--run-id",
                    run_id,
                ]
                for goal_text in goals:
                    argv.extend(["--goal-text", goal_text])
                if model:
                    argv.extend(["--model", model])
                if _reward_launch_tmux("value_model_sweep", argv):
                    current.update({
                        "model": model,
                        "run_id": run_id,
                    })
                    state["value_model_sweep"] = current
                    _reward_save_state(state)
                _invalidate_cache(rewards_cache)
            elif segments == ["rewards", "value-model-sweep", "cancel"]:
                _reward_cancel_tmux("value_model_sweep")
                _invalidate_cache(rewards_cache)
            elif segments == ["ablations", "create"]:
                env_id = (parsed.get("env_id", [""])[0] or "").strip().lower()
                requested_kind = (parsed.get("library_ablation_kind", [""])[0] or "").strip().lower()
                if env_id not in SUPPORTED_ENVS:
                    raise ValueError(f"Unknown environment {env_id}")
                if requested_kind not in {"supervised", "unsupervised"}:
                    requested_kind = "supervised"
                pattern_run_dir = _resolve_ablation_source_run_dir(
                    project_root=project_root,
                    env_id=env_id,
                    kind=requested_kind,
                )
                payload: dict[str, object] = {
                    "enqueue": True,
                    "targets": [
                        {
                            "env_id": env_id,
                            "pattern_run_dir": pattern_run_dir,
                            "run_id": _continuous_ablation_loop_run_id(
                                env_id=env_id,
                                kind=requested_kind,
                            ),
                        }
                    ],
                    "template_name": (parsed.get("template_name", ["quick-eval"])[0] or "quick-eval").strip() or "quick-eval",
                }
                response_payload = _post_json(f"{backend_root}/api/v1/submit/library-holdout-loop", payload) or {}
                job_payload = response_payload.get("job")
                if not isinstance(job_payload, dict):
                    raise ValueError("Backend did not return ablation job payload")
                job_id = str(job_payload.get("job_id") or "").strip()
                if not job_id:
                    raise ValueError("Backend did not return ablation job id")
                preferred_client_id = (parsed.get("preferred_client_id", [""])[0] or "").strip()
                if preferred_client_id:
                    _post_json(
                        f"{backend_root}/api/v1/jobs/{job_id}/preferred-client",
                        {"preferred_client_id": preferred_client_id},
                    )
                _post_json(f"{backend_root}/api/v1/jobs/{job_id}/pause")
            elif segments == ["ablations", "track"]:
                env_id = (parsed.get("env_id", [""])[0] or "").strip().lower()
                family_slug = (parsed.get("family", [""])[0] or "").strip().lower()
                family = required_family_from_slug(family_slug)
                if env_id not in SUPPORTED_ENVS:
                    raise ValueError(f"Unknown environment {env_id}")
                if family not in REQUIRED_LIBRARY_ABLATION_JOB_FAMILIES:
                    raise ValueError("Unknown required ablation family")
                family_kind = {
                    "Supervised library": "supervised",
                    "Unsupervised library": "unsupervised",
                }[family]
                pattern_run_dir = _resolve_ablation_source_run_dir(
                    project_root=project_root,
                    env_id=env_id,
                    kind=family_kind,
                )
                response_payload = _post_json(
                    f"{backend_root}/api/v1/submit/library-holdout-loop",
                    {
                        "enqueue": True,
                        "targets": [
                            {
                                "env_id": env_id,
                                "pattern_run_dir": pattern_run_dir,
                                "run_id": _continuous_ablation_loop_run_id(
                                    env_id=env_id,
                                    kind=family_kind,
                                ),
                            }
                        ],
                        "template_name": "quick-eval",
                    },
                ) or {}
                job_payload = response_payload.get("job")
                if not isinstance(job_payload, dict):
                    raise ValueError("Backend did not return ablation job payload")
                job_id = str(job_payload.get("job_id") or "").strip()
                if not job_id:
                    raise ValueError("Backend did not return ablation job id")
                catalog = _read_required_library_ablation_catalog(project_root)
                catalog[(env_id, family)] = job_id
                _write_required_library_ablation_catalog(
                    project_root,
                    catalog,
                    live_jobs=_all_jobs(),
                    created_job_payload=job_payload,
                )
                _required_library_ablation_job_catalog.cache_clear()
                ablations_cache["html"] = None
                ablations_cache["expires_at"] = 0.0
                ablation_plot_context_cache["entries"] = None
                ablation_plot_context_cache["expires_at"] = 0.0
                ablation_data_cache.clear()
            elif segments == ["ablations", "queue-plot"]:
                env_id = (parsed.get("env_id", [""])[0] or "").strip().lower()
                kind = (parsed.get("kind", [""])[0] or "").strip().lower()
                if env_id not in SUPPORTED_ENVS:
                    raise ValueError(f"Unknown environment {env_id}")
                if kind not in {"supervised", "unsupervised"}:
                    raise ValueError(f"Unknown ablation family {kind}")
                catalog = _read_required_library_ablation_catalog(project_root)
                family = {
                    "supervised": "Supervised library",
                    "unsupervised": "Unsupervised library",
                }[kind]
                pattern_run_dir = _resolve_ablation_source_run_dir(
                    project_root=project_root,
                    env_id=env_id,
                    kind=kind,
                )
                response_payload = _post_json(
                    f"{backend_root}/api/v1/submit/library-holdout-loop",
                    {
                        "enqueue": True,
                        "targets": [
                            {
                                "env_id": env_id,
                                "pattern_run_dir": pattern_run_dir,
                                "run_id": _continuous_ablation_loop_run_id(
                                    env_id=env_id,
                                    kind=kind,
                                ),
                            }
                        ],
                        "template_name": "quick-eval",
                    },
                ) or {}
                job_payload = response_payload.get("job")
                if not isinstance(job_payload, dict):
                    raise ValueError("Backend did not return ablation job payload")
                job_id = str(job_payload.get("job_id") or "").strip()
                if not job_id:
                    raise ValueError("Backend did not return ablation job id")
                catalog[(env_id, family)] = job_id
                _write_required_library_ablation_catalog(
                    project_root,
                    catalog,
                    live_jobs=_all_jobs(),
                    created_job_payload=job_payload,
                )
                _required_library_ablation_job_catalog.cache_clear()
                ablations_cache["html"] = None
                ablations_cache["expires_at"] = 0.0
                ablation_plot_context_cache["entries"] = None
                ablation_plot_context_cache["expires_at"] = 0.0
                ablation_data_cache.clear()
            elif segments == ["learning", "optimized-evals", "create"]:
                requested_env = (parsed.get("env_id", ["all"])[0] or "all").strip().lower()
                target_envs = list(SUPPORTED_ENVS) if requested_env == "all" else [requested_env]
                invalid_envs = [env_id for env_id in target_envs if env_id not in SUPPORTED_ENVS]
                if invalid_envs:
                    raise ValueError(f"Unknown environment {invalid_envs[0]}")
                launch_stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
                for env_id in target_envs:
                    for kind in ("supervised", "unsupervised"):
                        source_run_dir = resolve_canonical_library(project_root, env_id, kind).parent
                        run_id = f"optimized-ensemble-eval-{env_id}-{kind}-{launch_stamp}"
                        _post_json(
                            f"{backend_root}/api/v1/submit/library-holdout-eval",
                            {
                                "enqueue": True,
                                "env_id": env_id,
                                "pattern_run_dir": str(source_run_dir),
                                "template_name": "quick-eval",
                                "workers": 1,
                                "run_id": run_id,
                            },
                        )
                _invalidate_cache(learning_cache)
            elif segments == ["analysis", "create"]:
                env_id = (parsed.get("env_id", [""])[0] or "").strip().lower()
                trace_path = (parsed.get("trace_path", [""])[0] or "").strip()
                use_env_test_set = (parsed.get("use_env_test_set", [""])[0] or "").strip().lower() in {"1", "true", "on", "yes"}
                run_dirs = [(value or "").strip() for value in parsed.get("run_dirs", []) if (value or "").strip()]
                checkpoints_csv = (parsed.get("checkpoints_csv", [""])[0] or "").strip()
                sampling_repeats_raw = (parsed.get("sampling_repeats", ["32"])[0] or "32").strip()
                if env_id not in SUPPORTED_ENVS:
                    raise ValueError(f"Unknown environment {env_id}")
                if not use_env_test_set and not trace_path:
                    raise ValueError("trace_path is required")
                if len(run_dirs) < 2:
                    raise ValueError("Select at least two supervised runs")
                sampling_repeats = min(32, max(1, int(sampling_repeats_raw)))
                checkpoints: list[int] = []
                if checkpoints_csv:
                    for part in checkpoints_csv.split(","):
                        text = part.strip()
                        if not text:
                            continue
                        checkpoints.append(max(1, int(text)))
                _update_analysis_env_state(
                    env_id,
                    trace_path=trace_path,
                    trace_scope="test_set" if use_env_test_set else "single_trace",
                    run_dirs=run_dirs,
                    checkpoints=checkpoints,
                    sampling_repeats=sampling_repeats,
                )
                _enqueue_local_analysis(env_id)
                _invalidate_cache(analysis_cache)
            elif segments == ["analysis", "queue"]:
                env_id = (parsed.get("env_id", [""])[0] or "").strip().lower()
                if env_id not in SUPPORTED_ENVS:
                    raise ValueError(f"Unknown environment {env_id}")
                latest_result = next(
                    iter(list_supervised_consistency_results(project_root, env_id=env_id, limit=1)),
                    None,
                )
                run_dirs: list[str] = []
                trace_path = ""
                use_env_test_set = True
                checkpoints: list[int] = []
                sampling_repeats = 32
                if latest_result is not None and isinstance(latest_result.payload, dict):
                    payload = latest_result.payload
                    raw_run_dirs = payload.get("run_dirs")
                    if isinstance(raw_run_dirs, list):
                        run_dirs = [
                            str(item).strip()
                            for item in raw_run_dirs
                            if str(item).strip() and Path(str(item).strip()).is_dir()
                        ]
                    trace_scope = str(payload.get("trace_scope") or "").strip().lower()
                    use_env_test_set = trace_scope != "single_trace"
                    if not use_env_test_set:
                        trace_path = str(payload.get("trace_path") or "").strip()
                    raw_checkpoints = payload.get("checkpoints")
                    if isinstance(raw_checkpoints, list):
                        for value in raw_checkpoints:
                            try:
                                parsed_value = int(value)
                            except (TypeError, ValueError):
                                continue
                            if parsed_value > 0:
                                checkpoints.append(parsed_value)
                    try:
                        sampling_repeats = min(32, max(1, int(payload.get("sampling_repeats") or 32)))
                    except (TypeError, ValueError):
                        sampling_repeats = 32
                if len(run_dirs) < 2:
                    run_dirs = _select_local_analysis_run_dirs(env_id)
                    use_env_test_set = True
                    trace_path = ""
                    checkpoints = []
                    sampling_repeats = 32
                if len(run_dirs) < 2:
                    raise ValueError("Need at least two supervised runs to queue analysis")
                _update_analysis_env_state(
                    env_id,
                    trace_path=trace_path,
                    trace_scope="test_set" if use_env_test_set else "single_trace",
                    run_dirs=run_dirs,
                    checkpoints=checkpoints,
                    sampling_repeats=sampling_repeats,
                )
                _enqueue_local_analysis(env_id)
                _invalidate_cache(analysis_cache)
            elif len(segments) == 4 and segments[0] == "required-runs" and segments[3] == "combine":
                env_id = str(segments[1] or "").strip().lower()
                family_slug = str(segments[2] or "").strip().lower()
                family = required_family_from_slug(family_slug)
                if env_id not in SUPPORTED_ENVS:
                    raise ValueError(f"Unknown environment {env_id}")
                if family != "Supervised":
                    raise ValueError("Supervised run combination is only available for supervised runs")
                job_ids = [(value or "").strip() for value in parsed.get("job_ids", []) if (value or "").strip()]
                if len(job_ids) < 2:
                    raise ValueError("Select at least two supervised runs")
                _post_json(
                    f"{backend_root}/api/v1/supervised-runs/combine",
                    {
                        "env_id": env_id,
                        "job_ids": job_ids,
                    },
                    timeout_seconds=1800.0,
                )
            elif len(segments) == 4 and segments[0] == "required-runs" and segments[3] == "delete":
                env_id = str(segments[1] or "").strip().lower()
                family_slug = str(segments[2] or "").strip().lower()
                family = required_family_from_slug(family_slug)
                if env_id not in SUPPORTED_ENVS:
                    raise ValueError(f"Unknown environment {env_id}")
                if family not in REPEAT_LEARNING_FAMILIES:
                    raise ValueError("Run deletion is only available for learning runs")
                job_ids = [(value or "").strip() for value in parsed.get("job_ids", []) if (value or "").strip()]
                if not job_ids:
                    raise ValueError("Select at least one run")
                _post_json(
                    f"{backend_root}/api/v1/repeat-learning-runs/hide",
                    {
                        "job_ids": job_ids,
                    },
                )
            elif len(segments) == 4 and segments[0] == "required-runs" and segments[3] == "evaluate":
                env_id = str(segments[1] or "").strip().lower()
                family_slug = str(segments[2] or "").strip().lower()
                family = required_family_from_slug(family_slug)
                if env_id not in SUPPORTED_ENVS:
                    raise ValueError(f"Unknown environment {env_id}")
                if family not in REPEAT_LEARNING_FAMILIES:
                    raise ValueError("Manual evaluation is only available for learning runs")
                job_ids = [(value or "").strip() for value in parsed.get("job_ids", []) if (value or "").strip()]
                if len(job_ids) != 1:
                    raise ValueError("Select exactly one run to evaluate")
                try:
                    run_job = _job_by_id(job_ids[0])
                except KeyError as exc:
                    raise ValueError(f"Unknown run {job_ids[0]}") from exc
                run_dir_path = _resolve_job_run_dir(run_job, project_root=project_root)
                if run_dir_path is None:
                    raise ValueError("Selected run does not expose a run_dir")
                _post_json(
                    f"{backend_root}/api/v1/submit/deepphy-eval",
                    {
                        "enqueue": True,
                        "env_id": env_id,
                        "pattern_run_dir": str(run_dir_path),
                        "benchmark_preset": "full",
                        "workers": 1 if env_id == "pooltool" else 2,
                    },
                )
            elif segments == ["videos", "render"]:
                env_id = (parsed.get("env_id", [""])[0] or "").strip().lower()
                if env_id not in SUPPORTED_ENVS:
                    raise ValueError(f"Unknown environment {env_id}")
                raw_target_count = (parsed.get("target_count", [str(DEFAULT_VIDEO_RENDER_TARGET)])[0] or str(DEFAULT_VIDEO_RENDER_TARGET)).strip()
                try:
                    target_count = max(1, int(raw_target_count))
                except ValueError as exc:
                    raise ValueError(f"Invalid target count {raw_target_count}") from exc
                start_video_render_process(project_root, env_id, target_count=target_count)
                _invalidate_cache(videos_cache)
            elif segments == ["videos", "annotate"]:
                env_id = (parsed.get("env_id", [""])[0] or "").strip().lower()
                if env_id not in SUPPORTED_ENVS:
                    raise ValueError(f"Unknown environment {env_id}")
                selected_labels = [(value or "").strip() for value in parsed.get("selected_labels", []) if (value or "").strip()]
                save_annotation_video_selection(project_root, env_id, selected_labels=selected_labels)
                _invalidate_cache(videos_cache)
            elif segments == ["videos", "annotate", "cancel"]:
                env_id = (parsed.get("env_id", [""])[0] or "").strip().lower()
                if env_id not in SUPPORTED_ENVS:
                    raise ValueError(f"Unknown environment {env_id}")
                cancel_annotation_video_process(project_root, env_id)
                _invalidate_cache(videos_cache)
            elif segments == ["summaries", "run"]:
                env_id = (parsed.get("env_id", [""])[0] or "").strip().lower()
                if env_id not in SUPPORTED_ENVS:
                    raise ValueError(f"Unknown environment {env_id}")
                raw_target_count = (
                    parsed.get("target_count", [str(DEFAULT_TASK_SUMMARY_TARGET)])[0]
                    or str(DEFAULT_TASK_SUMMARY_TARGET)
                ).strip()
                try:
                    target_count = max(1, int(raw_target_count))
                except ValueError as exc:
                    raise ValueError(f"Invalid target count {raw_target_count}") from exc
                enabled_variants_override = [
                    str(item).strip()
                    for item in parsed.get("enabled_variants", [])
                    if str(item).strip()
                ]
                start_task_summary_process(
                    project_root,
                    env_id,
                    target_count=target_count,
                    enabled_variants_override=enabled_variants_override or None,
                )
                _invalidate_cache(summaries_cache)
            elif segments == ["summaries", "config"]:
                image_only_prompt_template = (
                    parsed.get("image_only_prompt_template", [""])[0] or ""
                ).strip()
                annotation_prompt_template = (
                    parsed.get("annotation_prompt_template", [""])[0] or ""
                ).strip()
                current_config = load_task_summary_config(project_root)
                save_task_summary_config(
                    project_root,
                    image_only_prompt_template=image_only_prompt_template,
                    annotation_prompt_template=annotation_prompt_template,
                    enabled_variants=list(current_config.get("enabled_variants") or []),
                    supervised_library_strategy=str(
                        current_config.get("supervised_library_strategy") or "latest"
                    ).strip().lower()
                    or "latest",
                )
                _invalidate_cache(summaries_cache)
            else:
                raise HTTPException(status_code=404, detail=f"Unknown dashboard action /ui/{path}")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return PlainTextResponse(detail, status_code=exc.code)
        except (TimeoutError, socket.timeout) as exc:
            raise HTTPException(status_code=504, detail=f"Backend UI request timed out: {exc}") from exc
        except urllib_error.URLError as exc:
            raise HTTPException(status_code=502, detail=f"Backend UI request failed: {exc.reason}") from exc
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return RedirectResponse(url=next_url, status_code=303)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"ok": True})

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5174)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8791")
    parser.add_argument("--backend-port", type=int, default=8791)
    parser.add_argument("--backend-public-https-port", type=int, default=10000)
    args = parser.parse_args()

    import uvicorn

    app = create_dashboard_app(
        backend_base_url=args.backend_url,
        backend_private_port=args.backend_port,
        backend_public_https_port=args.backend_public_https_port,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
