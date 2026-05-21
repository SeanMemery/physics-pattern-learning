from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import JobConstraints, JobRecord, JobSpec, JobStatus


SPECIAL_ENV_DIRS = {
    "unsupervised",
    "automatic",
    "image_only_evaluations",
    "reward_program_experiments",
}
RUN_ACTIONS = {
    "supervised": "supervised-learn-and-evaluate",
    "unsupervised": "unsupervised-launch",
    "automatic": "automatic-launch",
    "deepphy-eval": "evaluate-deepphy",
    "image-only-eval": "evaluate-image-only",
    "reward-optimize": "pooltool-reward-optimize",
}
TEXT_PATH_CANDIDATES = (
    "run.json",
    "results.json",
    "summary.json",
    "request.json",
    "debug.log",
    "synthesis_response.txt",
)
MIGRATION_KEY = "copied_runs_to_native_v2"


@dataclass
class ImportedJob:
    record: JobRecord
    log_lines: list[str]


def _utc_from_timestamp(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def _safe_load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _stable_job_id(kind: str, relpath: str) -> str:
    digest = hashlib.sha1(f"{kind}:{relpath}".encode("utf-8")).hexdigest()
    return f"job-{digest[:24]}"


def _relpath(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _tail_lines(path: Path, *, limit: int = 200) -> list[str]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return []
    return lines[-limit:]


def _status_from_payload(status_value: Any, *, has_manifest: bool) -> JobStatus:
    normalized = str(status_value or "").strip().lower()
    if normalized in {"completed", "complete", "succeeded", "success"}:
        return JobStatus.succeeded
    if normalized in {"failed", "error"}:
        return JobStatus.failed
    if normalized == "paused":
        return JobStatus.paused
    if normalized in {"cancelled", "canceled"}:
        return JobStatus.cancelled
    if normalized in {"running", "assigned", "starting", "resuming", "stopping", "queued"}:
        return JobStatus.failed
    if has_manifest:
        return JobStatus.succeeded
    return JobStatus.failed


def _message_for_payload(kind: str, payload: dict[str, Any], *, status: JobStatus) -> str:
    run_id = str(payload.get("run_id") or payload.get("evaluation_run_id") or "").strip()
    average_score = payload.get("average_score")
    validation_score = payload.get("validation_score")
    outcome_count = payload.get("outcome_count")
    parts: list[str] = []
    if run_id:
        parts.append(run_id)
    if isinstance(average_score, (int, float)):
        parts.append(f"score {average_score:.4f}")
    elif isinstance(validation_score, (int, float)):
        parts.append(f"val {validation_score:.4f}")
    elif isinstance(outcome_count, int):
        parts.append(f"{outcome_count} outcomes")
    if status == JobStatus.failed and not parts:
        parts.append(f"incomplete {kind} run")
    return " · ".join(parts) or f"{kind} run"


def _env_id_from_payload(payload: dict[str, Any], fallback: str) -> str:
    env_id = str(payload.get("env_id") or "").strip()
    if env_id:
        return env_id
    environment_ids = payload.get("environment_ids")
    if isinstance(environment_ids, list) and environment_ids:
        return str(environment_ids[0])
    return fallback


def _normalize_path_value(project_root: Path, value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        candidate = Path(text)
        if candidate.is_absolute():
            local_candidate = project_root / candidate.relative_to(candidate.anchor)
            if local_candidate.exists():
                return str(local_candidate)
        return value
    if isinstance(value, list):
        return [_normalize_path_value(project_root, item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _normalize_path_value(project_root, item)
            for key, item in value.items()
        }
    return value


def _imported_job(
    *,
    project_root: Path,
    kind: str,
    env_id: str,
    run_dir: Path,
    manifest_path: Path | None,
    payload: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
    log_path: Path | None = None,
) -> ImportedJob:
    stat_path = manifest_path or log_path or run_dir
    created_at = _utc_from_timestamp(stat_path.stat().st_mtime)
    updated_at = created_at
    if manifest_path is not None:
        updated_at = _utc_from_timestamp(manifest_path.stat().st_mtime)
    control_path = run_dir / "run_control.json"
    control_payload = _safe_load_json(control_path) if control_path.is_file() else {}
    updated_at = _parse_iso_datetime(control_payload.get("updated_at")) or updated_at
    status = _status_from_payload(payload.get("status"), has_manifest=manifest_path is not None)
    terminal = status in {
        JobStatus.succeeded,
        JobStatus.failed,
        JobStatus.cancelled,
        JobStatus.paused,
        JobStatus.stopped,
    }
    rel_run_dir = _relpath(project_root, run_dir)
    artifact_relpaths = [rel_run_dir]
    for relname in TEXT_PATH_CANDIDATES:
        candidate = run_dir / relname
        if candidate.exists():
            artifact_relpaths.append(_relpath(project_root, candidate))
    metadata: dict[str, Any] = {
        "action": RUN_ACTIONS[kind],
        "env_id": env_id,
        "run_dir": str(run_dir),
        "controllable": False,
        "imported": True,
        "source_format": "v1-copied",
        "migration_key": MIGRATION_KEY,
    }
    normalized_request = _normalize_path_value(project_root, payload.get("request"))
    if normalized_request:
        metadata["request"] = normalized_request
    for key in (
        "auto_evaluation_run_id",
        "average_score",
        "validation_score",
        "train_score",
        "run_id",
        "source_run_dir",
        "source_run_id",
        "source_run_kind",
        "benchmark_preset",
        "workers",
        "configured_workers",
        "pattern_run_dir",
        "evaluation_run_dir",
        "evaluation_run_id",
        "output_path",
    ):
        value = payload.get(key)
        if value is not None:
            metadata[key] = _normalize_path_value(project_root, value)
    if extra_metadata:
        metadata.update(_normalize_path_value(project_root, extra_metadata))
    return ImportedJob(
        record=JobRecord(
            job_id=_stable_job_id(kind, rel_run_dir),
            status=status,
            created_at=created_at,
            updated_at=updated_at,
            started_at=created_at,
            finished_at=updated_at if terminal else None,
            message=_message_for_payload(kind, payload, status=status),
            spec=JobSpec(
                action=RUN_ACTIONS[kind],
                argv=[],
                cwd=str(project_root),
                metadata=metadata,
                constraints=JobConstraints(required_environments=[env_id] if env_id else []),
            ),
            recent_logs=_tail_lines(log_path) if log_path else [],
            artifact_relpaths=artifact_relpaths,
        ),
        log_lines=_tail_lines(log_path) if log_path else [],
    )


def _import_top_level_runs(project_root: Path) -> list[ImportedJob]:
    records: list[ImportedJob] = []
    patterns_root = project_root / "data" / "patterns"
    if not patterns_root.exists():
        return records
    for env_dir in sorted(patterns_root.iterdir()):
        if not env_dir.is_dir():
            continue
        env_id = env_dir.name
        for run_dir in sorted(env_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name in SPECIAL_ENV_DIRS:
                continue
            manifest_path = run_dir / "run.json"
            debug_log = run_dir / "debug.log"
            control_path = run_dir / "run_control.json"
            if not (manifest_path.exists() or debug_log.exists() or control_path.exists()):
                continue
            payload = _safe_load_json(manifest_path) if manifest_path.exists() else {}
            records.append(
                _imported_job(
                    project_root=project_root,
                    kind="supervised",
                    env_id=_env_id_from_payload(payload, env_id),
                    run_dir=run_dir,
                    manifest_path=manifest_path if manifest_path.exists() else None,
                    payload=payload,
                    log_path=debug_log if debug_log.exists() else None,
                )
            )
    return records


def _import_nested_run_kind(project_root: Path, nested_name: str, kind: str) -> list[ImportedJob]:
    records: list[ImportedJob] = []
    patterns_root = project_root / "data" / "patterns"
    if not patterns_root.exists():
        return records
    for run_dir in sorted(patterns_root.glob(f"*/{nested_name}/*")):
        if not run_dir.is_dir():
            continue
        env_id = run_dir.parents[1].name
        manifest_path = run_dir / "run.json"
        debug_log = run_dir / "debug.log"
        control_path = run_dir / "run_control.json"
        if not (manifest_path.exists() or debug_log.exists() or control_path.exists()):
            continue
        payload = _safe_load_json(manifest_path) if manifest_path.exists() else {}
        records.append(
            _imported_job(
                project_root=project_root,
                kind=kind,
                env_id=_env_id_from_payload(payload, env_id),
                run_dir=run_dir,
                manifest_path=manifest_path if manifest_path.exists() else None,
                payload=payload,
                log_path=debug_log if debug_log.exists() else None,
            )
        )
    return records


def _import_deepphy_evals(project_root: Path) -> list[ImportedJob]:
    records: list[ImportedJob] = []
    patterns_root = project_root / "data" / "patterns"
    if not patterns_root.exists():
        return records
    for results_path in sorted(patterns_root.glob("*/**/deepphy_eval/*/results.json")):
        eval_dir = results_path.parent
        env_id = results_path.parts[-5]
        payload = _safe_load_json(results_path)
        request_path = eval_dir / "request.json"
        records.append(
            _imported_job(
                project_root=project_root,
                kind="deepphy-eval",
                env_id=env_id,
                run_dir=eval_dir,
                manifest_path=results_path,
                payload=payload,
                extra_metadata={
                    "evaluation_run_dir": str(eval_dir),
                    "evaluation_run_id": eval_dir.name,
                    "pattern_run_dir": str(eval_dir.parents[1]),
                    "request": _safe_load_json(request_path) if request_path.exists() else {},
                },
            )
        )
    return records


def _import_image_only_evals(project_root: Path) -> list[ImportedJob]:
    records: list[ImportedJob] = []
    patterns_root = project_root / "data" / "patterns"
    if not patterns_root.exists():
        return records
    for results_path in sorted(patterns_root.glob("*/image_only_evaluations/*/results.json")):
        eval_dir = results_path.parent
        request_path = eval_dir / "request.json"
        request_payload = _safe_load_json(request_path) if request_path.exists() else {}
        env_id = str(request_payload.get("env_id") or eval_dir.parents[1].name)
        payload = _safe_load_json(results_path)
        records.append(
            _imported_job(
                project_root=project_root,
                kind="image-only-eval",
                env_id=env_id,
                run_dir=eval_dir,
                manifest_path=results_path,
                payload=payload,
                extra_metadata={
                    "output_path": str(eval_dir),
                    "request": request_payload,
                },
            )
        )
    return records


def _import_reward_optimizations(project_root: Path) -> list[ImportedJob]:
    records: list[ImportedJob] = []
    optimization_root = project_root / "data" / "patterns" / "pooltool" / "reward_program_experiments" / "optimization"
    if not optimization_root.exists():
        return records
    for run_dir in sorted(optimization_root.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        request_path = run_dir / "synthesis_request.json"
        response_log = run_dir / "synthesis_response.txt"
        if not (summary_path.exists() or request_path.exists() or response_log.exists()):
            continue
        payload = _safe_load_json(summary_path) if summary_path.exists() else {}
        records.append(
            _imported_job(
                project_root=project_root,
                kind="reward-optimize",
                env_id="pooltool",
                run_dir=run_dir,
                manifest_path=summary_path if summary_path.exists() else (request_path if request_path.exists() else None),
                payload=payload,
                extra_metadata={
                    "output_path": str(run_dir),
                    "request": _safe_load_json(request_path) if request_path.exists() else {},
                    "source_run_dir": payload.get("source_run_dir"),
                    "source_run_id": payload.get("source_run_id"),
                },
                log_path=response_log if response_log.exists() else None,
            )
        )
    return records


def discover_imported_jobs(project_root: Path) -> list[ImportedJob]:
    jobs: dict[str, ImportedJob] = {}
    for imported in (
        _import_top_level_runs(project_root)
        + _import_nested_run_kind(project_root, "unsupervised", "unsupervised")
        + _import_nested_run_kind(project_root, "automatic", "automatic")
        + _import_deepphy_evals(project_root)
        + _import_image_only_evals(project_root)
        + _import_reward_optimizations(project_root)
    ):
        jobs[imported.record.job_id] = imported
    return sorted(jobs.values(), key=lambda item: item.record.created_at, reverse=True)
