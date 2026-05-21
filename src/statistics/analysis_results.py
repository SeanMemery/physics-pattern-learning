from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SupervisedRunOption:
    env_id: str
    run_dir: Path
    run_name: str
    label_count: int
    completed_label_count: int
    candidate_count: int
    dataset_manifest: Path | None


@dataclass(frozen=True)
class TraceOption:
    env_id: str
    trace_path: Path
    dataset_name: str
    split_name: str
    sample_name: str
    task_id: str | None
    timestep_count: int


@dataclass(frozen=True)
class SupervisedConsistencyResult:
    analysis_run_id: str
    env_id: str
    result_dir: Path
    summary_path: Path
    payload: dict[str, Any]
    created_at: datetime | None


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _coerce_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def list_supervised_run_options(project_root: Path, *, env_id: str, limit: int = 24) -> list[SupervisedRunOption]:
    env_root = project_root / "data" / "patterns" / env_id
    if not env_root.is_dir():
        return []
    options: list[SupervisedRunOption] = []
    seen: set[str] = set()
    for run_json in sorted(env_root.glob("**/run.json"), key=lambda path: path.parent.as_posix()):
        run_dir = run_json.parent
        parts = {part.lower() for part in run_dir.parts}
        if "unsupervised" in parts or "automatic" in parts:
            continue
        if not any(run_dir.glob("label_*")):
            continue
        key = str(run_dir.resolve())
        if key in seen:
            continue
        seen.add(key)
        payload = _load_json_object(run_json)
        scorer_config = payload.get("scorer_config") if isinstance(payload.get("scorer_config"), dict) else {}
        dataset_manifest = None
        raw_manifest = scorer_config.get("dataset_manifest")
        if isinstance(raw_manifest, str) and raw_manifest.strip():
            candidate = Path(raw_manifest.strip())
            if not candidate.is_absolute():
                candidate = (project_root / candidate).resolve()
            if candidate.exists():
                dataset_manifest = candidate
        candidate_count = 0
        try:
            candidate_count = sum(1 for _ in (run_dir / "evolution_record.jsonl").open("r", encoding="utf-8"))
        except OSError:
            candidate_count = 0
        options.append(
            SupervisedRunOption(
                env_id=env_id,
                run_dir=run_dir.resolve(),
                run_name=run_dir.name,
                label_count=int(payload.get("primitive_label_count") or 0),
                completed_label_count=max(
                    int(payload.get("completed_label_count") or 0),
                    sum(1 for _ in run_dir.glob("label_*/label_result.json")),
                ),
                candidate_count=candidate_count,
                dataset_manifest=dataset_manifest,
            )
        )
    options.sort(key=lambda item: item.run_name, reverse=True)
    return options[:limit]


def list_trace_options(project_root: Path, *, env_id: str, limit: int = 32) -> list[TraceOption]:
    dataset_root = project_root / "data" / "datasets" / env_id
    if not dataset_root.is_dir():
        return []
    options: list[TraceOption] = []
    for trace_path in sorted(dataset_root.glob("**/trace.json"), key=lambda path: path.as_posix()):
        try:
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        try:
            dataset_name = trace_path.parents[2].name
            split_name = trace_path.parents[1].name
            sample_name = trace_path.parent.name
        except IndexError:
            continue
        options.append(
            TraceOption(
                env_id=env_id,
                trace_path=trace_path.resolve(),
                dataset_name=dataset_name,
                split_name=split_name,
                sample_name=sample_name,
                task_id=str(payload.get("task_id") or "").strip() or None,
                timestep_count=len(payload.get("timesteps") or []),
            )
        )
        if len(options) >= limit:
            break
    return options


def list_supervised_consistency_results(
    project_root: Path,
    *,
    env_id: str | None = None,
    limit: int = 24,
) -> list[SupervisedConsistencyResult]:
    root = project_root / "data" / "analysis" / "supervised_consistency"
    if env_id:
        root = root / env_id
    if not root.exists():
        return []
    results: list[SupervisedConsistencyResult] = []
    for summary_path in sorted(root.glob("**/summary.json"), key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True):
        payload = _load_json_object(summary_path)
        if payload.get("analysis_kind") != "supervised_consistency":
            continue
        result_env_id = str(payload.get("env_id") or "").strip().lower()
        if env_id and result_env_id != env_id:
            continue
        result_dir = summary_path.parent.resolve()
        analysis_run_id = str(payload.get("analysis_run_id") or result_dir.name).strip() or result_dir.name
        results.append(
            SupervisedConsistencyResult(
                analysis_run_id=analysis_run_id,
                env_id=result_env_id,
                result_dir=result_dir,
                summary_path=summary_path.resolve(),
                payload=payload,
                created_at=_coerce_datetime(payload.get("created_at")),
            )
        )
        if len(results) >= limit:
            break
    return results


def filter_supervised_consistency_results_to_latest_setup(
    results: list[SupervisedConsistencyResult],
) -> list[SupervisedConsistencyResult]:
    if not results:
        return []
    current_variant = "label_trajectory_similarity_to_final_bin_min25"
    current_mode = "window_bins"
    current_setup_results = [
        result
        for result in results
        if isinstance(result.payload, dict)
        and str(result.payload.get("analysis_variant") or "").strip() == current_variant
        and str(result.payload.get("analysis_mode") or "").strip() == current_mode
    ]
    return current_setup_results


def load_supervised_consistency_result(project_root: Path, analysis_run_id: str) -> SupervisedConsistencyResult | None:
    for result in list_supervised_consistency_results(project_root, limit=200):
        if result.analysis_run_id == analysis_run_id or result.result_dir.name == analysis_run_id:
            return result
    return None
