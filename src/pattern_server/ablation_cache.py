from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .library_ablation import case_counts_toward_label_series, load_library_ablation_detail
from .library_collections import _load_supervised_labels


SUPPORTED_ENVS: tuple[str, ...] = ("phyre", "iphyre", "kinetix", "pooltool")
SUPPORTED_KINDS: tuple[str, ...] = ("supervised", "unsupervised")
CACHE_VERSION = 5
_LABEL_DIR_PATTERN = re.compile(r"^label_(?P<index>\d+?)_(?P<label>.+)$")
_CANDIDATE_DIR_PATTERN = re.compile(r"^candidate_(?P<index>\d+)(?:_|$)")


def ablation_plot_cache_path(project_root: Path) -> Path:
    return project_root / "state" / "ablation_plot_cache.json"


def load_ablation_plot_cache(project_root: Path) -> dict[str, Any]:
    path = ablation_plot_cache_path(project_root)
    if not path.is_file():
        return {"version": CACHE_VERSION, "updated_at": "", "series": [], "importance_rows": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    series = payload.get("series")
    if not isinstance(series, list):
        series = []
    importance_rows = payload.get("importance_rows")
    if not isinstance(importance_rows, list):
        importance_rows = []
    return {
        "version": int(payload.get("version") or CACHE_VERSION),
        "updated_at": str(payload.get("updated_at") or ""),
        "series": series,
        "importance_rows": importance_rows,
    }


def save_ablation_plot_cache(project_root: Path, payload: dict[str, Any]) -> Path:
    path = ablation_plot_cache_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def refresh_ablation_plot_cache(project_root: Path) -> tuple[dict[str, Any], dict[str, int]]:
    discovered_roots = list(_iter_ablation_eval_roots(project_root))
    total = len(discovered_roots)
    refreshed = 0
    skipped = 0
    deduped_runs: dict[tuple[str, str, str], dict[str, Any]] = {}
    deduped_importance_runs: dict[tuple[str, str, str], dict[str, Any]] = {}
    deduped = 0
    for env_id, kind, eval_root in discovered_roots:
        latest_mtime = _latest_eval_root_mtime(eval_root)
        request_payload = _load_json(eval_root / "request.json")
        raw_source = str(request_payload.get("pattern_run_dir") or "").strip() if isinstance(request_payload, dict) else ""
        if not raw_source:
            skipped += 1
            continue
        source_run_dir = Path(raw_source)
        if not source_run_dir.is_absolute():
            source_run_dir = (project_root / source_run_dir).resolve()
        detail = load_library_ablation_detail(
            eval_root=eval_root,
            env_id=env_id,
            source_run_dir=source_run_dir,
            ablation_kind=kind,
        )
        if detail is None:
            skipped += 1
            continue
        serialized = _serialize_run_entry(
            detail=detail,
            env_id=env_id,
            kind=kind,
            project_root=project_root,
            latest_mtime=latest_mtime,
        )
        run_identity = (
            str(serialized.get("env_id") or ""),
            str(serialized.get("kind") or ""),
            str(serialized.get("run_id") or ""),
        )
        existing_entry = deduped_runs.get(run_identity)
        if existing_entry is not None:
            existing_mtime = float(existing_entry.get("latest_mtime") or 0.0)
            if existing_mtime >= latest_mtime:
                deduped += 1
            else:
                deduped_runs[run_identity] = serialized
                deduped += 1
        else:
            deduped_runs[run_identity] = serialized
        existing_importance_entry = deduped_importance_runs.get(run_identity)
        if existing_importance_entry is not None:
            if _prefer_importance_run(existing_importance_entry, serialized):
                deduped += 1
            else:
                deduped_importance_runs[run_identity] = serialized
                deduped += 1
        else:
            deduped_importance_runs[run_identity] = serialized
        refreshed += 1
    aggregated_series = _aggregate_series_from_runs(list(deduped_runs.values()))
    aggregated_importance_rows = _aggregate_importance_rows_from_runs(
        runs=list(deduped_importance_runs.values()),
        project_root=project_root,
    )
    payload = {
        "version": CACHE_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "series": aggregated_series,
        "importance_rows": aggregated_importance_rows,
    }
    save_ablation_plot_cache(project_root, payload)
    return payload, {"total": total, "reused": 0, "refreshed": refreshed, "skipped": skipped, "deduped": deduped}


def build_ablation_points_from_cache(
    *,
    cache: dict[str, Any],
    env_id: str,
    kind: str,
) -> tuple[list[dict[str, float | int]], int]:
    series = cache.get("series") if isinstance(cache.get("series"), list) else []
    for entry in series:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("env_id") or "").strip().lower() != env_id:
            continue
        if str(entry.get("kind") or "").strip().lower() != kind:
            continue
        bins = entry.get("bins") if isinstance(entry.get("bins"), list) else []
        return ([dict(item) for item in bins if isinstance(item, dict)], int(entry.get("run_count") or 0))
    return [], 0


def build_ablation_importance_rows_from_cache(cache: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    rows = cache.get("importance_rows") if isinstance(cache.get("importance_rows"), list) else []
    point_count = 0
    run_count = 0
    result: list[dict[str, Any]] = []
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        points = entry.get("points") if isinstance(entry.get("points"), list) else []
        copied_points = [dict(point) for point in points if isinstance(point, dict)]
        point_count += len(copied_points)
        run_count += int(entry.get("run_count") or 0)
        result.append({**entry, "points": copied_points})
    return result, point_count, run_count


def _iter_ablation_eval_roots(project_root: Path):
    seen: set[str] = set()
    for manifest_path in sorted(project_root.glob("**/library_evaluations/*/manifest.json")):
        eval_root = manifest_path.parent
        request_path = eval_root / "request.json"
        if not request_path.is_file():
            continue
        request_payload = _load_json(request_path)
        env_id = str(request_payload.get("env_id") or "").strip().lower()
        if env_id not in SUPPORTED_ENVS:
            continue
        raw_source = str(request_payload.get("pattern_run_dir") or "").strip().lower()
        if "/unsupervised/" in raw_source or raw_source.endswith("-unsupervised"):
            kind = "unsupervised"
        else:
            kind = "supervised"
        if kind not in SUPPORTED_KINDS:
            continue
        key = str(eval_root.resolve())
        if key in seen:
            continue
        seen.add(key)
        yield env_id, kind, eval_root.resolve()


def _latest_eval_root_mtime(eval_root: Path) -> float:
    latest = 0.0
    for candidate in [
        eval_root / "request.json",
        eval_root / "manifest.json",
        eval_root / "summary.json",
        *eval_root.glob("cases/*/results.json"),
        *eval_root.glob("cases/*/summary.json"),
        *eval_root.glob("cases/*/tasks/*/*/task_result.json"),
        *eval_root.glob("tasks/*/*/task_result.json"),
    ]:
        try:
            latest = max(latest, candidate.stat().st_mtime)
        except OSError:
            continue
    return latest


def _serialize_run_entry(
    *,
    detail: Any,
    env_id: str,
    kind: str,
    project_root: Path,
    latest_mtime: float,
) -> dict[str, Any]:
    target_count = _target_label_budget(project_root, env_id)
    source_count = max(1, len(getattr(detail, "available_labels", []) or []))
    cases: list[dict[str, Any]] = []
    baseline_success_ratio = _resolve_baseline_success_ratio(detail)
    for case in detail.case_results:
        observed_seed_ratios = [
            float(item.success_ratio)
            for item in case.seed_results
            if item.status in {"running", "completed"} and item.success_ratio is not None
        ]
        if not observed_seed_ratios:
            continue
        case_mean_success_ratio = _case_mean_success_ratio(case, observed_seed_ratios)
        normalized_label_count = _normalized_label_count(
            target_count=target_count,
            source_count=max(source_count, int(getattr(case, "label_count", 0) or 0), len(getattr(case, "enabled_labels", []) or [])),
            enabled_count=max(1, len(getattr(case, "enabled_labels", []) or []) or int(getattr(case, "label_count", 0) or 0)),
            current_count=int(getattr(case, "label_count", 0) or 0),
        )
        cases.append(
            {
                "case_id": case.case_id,
                "kind": case.kind,
                "counts_toward_label_series": bool(case_counts_toward_label_series(case)),
                "label_count": normalized_label_count,
                "held_out_labels": [str(label).strip() for label in getattr(case, "held_out_labels", []) if str(label).strip()],
                "mean_success_ratio": case_mean_success_ratio,
                "observed_seed_ratios": observed_seed_ratios,
                "observed_seed_count": len(observed_seed_ratios),
                "completed_seed_count": int(case.completed_seed_count),
            }
        )
    return {
        "cache_version": CACHE_VERSION,
        "env_id": env_id,
        "kind": kind,
        "run_id": str(detail.run_id),
        "latest_mtime": latest_mtime,
        "source_run_dir": str(getattr(detail, "source_run_dir", "") or ""),
        "baseline_success_ratio": baseline_success_ratio,
        "importance_case_count": sum(
            1
            for case in cases
            if isinstance(case.get("mean_success_ratio"), (int, float))
            and bool(case.get("held_out_labels"))
        ),
        "cases": cases,
    }


def _aggregate_series_from_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_series: dict[tuple[str, str], dict[str, Any]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        env_id = str(run.get("env_id") or "").strip().lower()
        kind = str(run.get("kind") or "").strip().lower()
        if not env_id or not kind:
            continue
        row = by_series.setdefault(
            (env_id, kind),
            {"env_id": env_id, "kind": kind, "run_count": 0, "bins": {}},
        )
        row["run_count"] += 1
        for case in run.get("cases", []):
            if not isinstance(case, dict):
                continue
            if not bool(case.get("counts_toward_label_series")):
                continue
            label_count = int(case.get("label_count") or 0)
            ratios = case.get("observed_seed_ratios")
            if label_count <= 0 or not isinstance(ratios, list) or not ratios:
                continue
            bucket = row["bins"].setdefault(
                label_count,
                {"ratios": [], "case_count": 0, "observed_seed_count": 0, "completed_seed_count": 0},
            )
            bucket["ratios"].extend(float(value) for value in ratios if isinstance(value, (int, float)))
            bucket["case_count"] += 1
            bucket["observed_seed_count"] += int(case.get("observed_seed_count") or 0)
            bucket["completed_seed_count"] += int(case.get("completed_seed_count") or 0)
    result: list[dict[str, Any]] = []
    for env_id in SUPPORTED_ENVS:
        for kind in SUPPORTED_KINDS:
            row = by_series.get((env_id, kind))
            if row is None:
                continue
            bins: list[dict[str, Any]] = []
            for label_count in sorted(row["bins"]):
                ratios = [float(value) for value in row["bins"][label_count]["ratios"]]
                if not ratios:
                    continue
                mean_ratio = sum(ratios) / len(ratios)
                if len(ratios) > 1:
                    mean_square = sum(value * value for value in ratios) / len(ratios)
                    variance = max(0.0, mean_square - (mean_ratio * mean_ratio))
                    std_ratio = variance ** 0.5
                else:
                    std_ratio = 0.0
                bins.append(
                    {
                        "label_count": label_count,
                        "mean_success_ratio": mean_ratio,
                        "mean_success_ratio_std": std_ratio,
                        "case_count": int(row["bins"][label_count]["case_count"]),
                        "observed_seed_count": int(row["bins"][label_count]["observed_seed_count"]),
                        "completed_seed_count": int(row["bins"][label_count]["completed_seed_count"]),
                    }
                )
            result.append(
                {
                    "env_id": env_id,
                    "kind": kind,
                    "run_count": int(row["run_count"]),
                    "bins": bins,
                }
            )
    return result


def _aggregate_importance_rows_from_runs(
    *,
    runs: list[dict[str, Any]],
    project_root: Path,
) -> list[dict[str, Any]]:
    by_series: dict[tuple[str, str], dict[str, list[float]]] = {}
    all_reward_values: list[float] = []
    all_importance_values: list[float] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        env_id = str(run.get("env_id") or "").strip().lower()
        kind = str(run.get("kind") or "").strip().lower()
        baseline_success_ratio = run.get("baseline_success_ratio")
        if env_id not in SUPPORTED_ENVS or kind not in SUPPORTED_KINDS:
            continue
        if not isinstance(baseline_success_ratio, (int, float)) or not math.isfinite(float(baseline_success_ratio)):
            continue
        series_key = (env_id, kind)
        label_values = by_series.setdefault(series_key, defaultdict(list))
        for case in run.get("cases") or []:
            if not isinstance(case, dict):
                continue
            mean_success_ratio = case.get("mean_success_ratio")
            if not isinstance(mean_success_ratio, (int, float)) or not math.isfinite(float(mean_success_ratio)):
                continue
            held_out_labels = [str(label).strip() for label in (case.get("held_out_labels") or []) if str(label).strip()]
            if not held_out_labels:
                continue
            importance = _normalized_ablation_importance(
                baseline_success_ratio=float(baseline_success_ratio),
                ablated_success_ratio=float(mean_success_ratio),
            )
            if importance is None:
                continue
            for label in held_out_labels:
                label_values[label].append(float(importance))
    rows: list[dict[str, Any]] = []
    for env_id in SUPPORTED_ENVS:
        for kind in SUPPORTED_KINDS:
            reward_by_label = _max_training_reward_by_label_for_env_kind(project_root=project_root, env_id=env_id, kind=kind)
            label_values = by_series.get((env_id, kind)) or {}
            points: list[dict[str, Any]] = []
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
                points.append(
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
            rows.append(
                {
                    "key": f"{env_id}:{kind}",
                    "label": f"{env_id.upper()} {'Human-Labels' if kind == 'supervised' else 'LLM-Labels'}",
                    "env_id": env_id,
                    "kind": kind,
                    "color": _library_ablation_palette(kind),
                    "run_count": _run_count_for_series(runs, env_id=env_id, kind=kind),
                    "points": points,
                }
            )
    importance_scale = max((abs(value) for value in all_importance_values), default=1.0)
    importance_scale = importance_scale if importance_scale > 1e-9 else 1.0
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized_points: list[dict[str, Any]] = []
        for point in row.get("points") or []:
            clamped_reward = max(-1.0, min(1.0, float(point["training_reward"])))
            normalized_points.append(
                {
                    **point,
                    "training_reward_capped": clamped_reward,
                    "training_reward_normalized": clamped_reward,
                    "importance_normalized": max(-1.0, min(1.0, float(point["importance"]) / importance_scale)),
                    "importance_std_normalized": max(0.0, min(1.0, float(point["importance_std"]) / importance_scale)),
                }
            )
        normalized_rows.append({**row, "points": normalized_points})
    return normalized_rows


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_count_for_series(runs: list[dict[str, Any]], *, env_id: str, kind: str) -> int:
    return sum(
        1
        for run in runs
        if isinstance(run, dict)
        and str(run.get("env_id") or "").strip().lower() == env_id
        and str(run.get("kind") or "").strip().lower() == kind
    )


def _case_mean_success_ratio(case: Any, observed_seed_ratios: list[float]) -> float | None:
    mean_success_ratio = getattr(case, "mean_success_ratio", None)
    if isinstance(mean_success_ratio, (int, float)) and math.isfinite(float(mean_success_ratio)):
        return float(mean_success_ratio)
    if observed_seed_ratios:
        return float(statistics.fmean(observed_seed_ratios))
    return None


def _resolve_baseline_success_ratio(detail: Any) -> float | None:
    baseline_case = getattr(detail, "baseline_case", None)
    if baseline_case is not None:
        observed = [
            float(item.success_ratio)
            for item in getattr(baseline_case, "seed_results", [])
            if getattr(item, "status", None) in {"running", "completed"} and getattr(item, "success_ratio", None) is not None
        ]
        resolved = _case_mean_success_ratio(baseline_case, observed)
        if resolved is not None:
            return resolved
    fallback_cases: list[tuple[int, float]] = []
    for case in getattr(detail, "case_results", []) or []:
        observed = [
            float(item.success_ratio)
            for item in getattr(case, "seed_results", [])
            if getattr(item, "status", None) in {"running", "completed"} and getattr(item, "success_ratio", None) is not None
        ]
        resolved = _case_mean_success_ratio(case, observed)
        if resolved is None:
            continue
        label_count = int(getattr(case, "label_count", 0) or len(getattr(case, "enabled_labels", []) or []) or 0)
        if str(getattr(case, "kind", "") or "").strip().lower() == "baseline":
            return resolved
        fallback_cases.append((label_count, resolved))
    if not fallback_cases:
        return None
    fallback_cases.sort(key=lambda item: item[0], reverse=True)
    return float(fallback_cases[0][1])


def _prefer_importance_run(existing_run: dict[str, Any], candidate_run: dict[str, Any]) -> bool:
    existing_count = int(existing_run.get("importance_case_count") or 0)
    candidate_count = int(candidate_run.get("importance_case_count") or 0)
    if existing_count != candidate_count:
        return existing_count > candidate_count
    existing_mtime = float(existing_run.get("latest_mtime") or 0.0)
    candidate_mtime = float(candidate_run.get("latest_mtime") or 0.0)
    return existing_mtime >= candidate_mtime


def _target_label_budget(project_root: Path, env_id: str) -> int:
    try:
        labels, _source = _load_supervised_labels(project_root, env_id)
        if labels:
            return max(1, len(labels))
    except Exception:
        pass
    defaults = {"phyre": 10, "iphyre": 12, "kinetix": 10, "pooltool": 12}
    return int(defaults.get(str(env_id or "").strip().lower(), 12))


def _normalized_label_count(*, target_count: int, source_count: int, enabled_count: int, current_count: int) -> int:
    target = max(1, int(target_count or 1))
    source = max(1, int(source_count or 1))
    enabled = max(1, int(enabled_count or 1))
    current = max(1, int(current_count or enabled))
    if current <= target and enabled <= target:
        return current
    scaled = int(round((float(enabled) / float(source)) * float(target)))
    return max(1, min(target, scaled))


def _normalize_source_run_dir(project_root: Path, source_run_dir: Path) -> Path | None:
    candidate = source_run_dir
    try:
        if candidate.exists():
            return candidate.resolve()
    except OSError:
        pass
    normalized = str(source_run_dir).replace("\\", "/")
    project_root = project_root.resolve()
    local_server_root = project_root
    local_services_root = project_root.parent
    local_client_root = (local_services_root / "pattern-learning-client").resolve()
    remaps = [
        ("/opt/pattern-learning-stack/pattern-learning-server/data/", (local_server_root / "data").resolve()),
        ("/var/opt/pattern-learning-stack/pattern-learning-server/data/", (local_server_root / "data").resolve()),
        ("/opt/pattern-learning-stack/pattern-learning-server/", local_server_root),
        ("/var/opt/pattern-learning-stack/pattern-learning-server/", local_server_root),
        ("/opt/pattern-learning-stack/pattern-learning-client/data/", (local_client_root / "data").resolve()),
        ("/var/opt/pattern-learning-stack/pattern-learning-client/data/", (local_client_root / "data").resolve()),
        ("/opt/pattern-learning-stack/pattern-learning-client/", local_client_root),
        ("/var/opt/pattern-learning-stack/pattern-learning-client/", local_client_root),
        ("/opt/pattern-learning-stack/data/", (local_server_root / "data").resolve()),
        ("/var/opt/pattern-learning-stack/data/", (local_server_root / "data").resolve()),
        ("/opt/pattern-learning-stack/", local_client_root),
        ("/var/opt/pattern-learning-stack/", local_client_root),
    ]
    for prefix, target_root in remaps:
        if not normalized.startswith(prefix):
            continue
        suffix = normalized[len(prefix):].lstrip("/")
        mapped = (target_root / suffix).resolve()
        if mapped.exists():
            return mapped
    work_root_mappings = [
        ("/Services/Work/pattern-learning-server/data/", (local_server_root / "data").resolve()),
        ("/Services/Work/pattern-learning-server/", local_server_root),
        ("/Services/Work/pattern-learning-client/data/", (local_client_root / "data").resolve()),
        ("/Services/Work/pattern-learning-client/", local_client_root),
    ]
    for marker, target_root in work_root_mappings:
        if marker not in normalized:
            continue
        suffix = normalized.split(marker, 1)[1].lstrip("/")
        mapped = (target_root / suffix).resolve()
        if mapped.exists():
            return mapped
    return None


def _canonical_library_root(project_root: Path, *, env_id: str, kind: str) -> Path | None:
    candidate = project_root / "state" / "video_renders" / "canonical_libraries" / env_id / kind
    try:
        return candidate.resolve() if candidate.is_dir() else None
    except OSError:
        return None


def _normalized_ablation_importance(*, baseline_success_ratio: float, ablated_success_ratio: float) -> float | None:
    denominator = abs(float(baseline_success_ratio))
    if denominator <= 1e-9:
        return None
    return (float(baseline_success_ratio) - float(ablated_success_ratio)) / denominator


def _library_ablation_palette(kind: str) -> str:
    return "#124f49" if str(kind or "").strip().lower() == "supervised" else "#a54d2d"


def _canonical_label_key(label: str) -> str:
    raw = str(label or "").strip().lower()
    if not raw:
        return ""
    tokens = [token for token in re.split(r"[^a-z0-9]+", raw) if token]
    return " ".join(tokens) if tokens else raw


def _max_training_reward_by_label_for_env_kind(*, project_root: Path, env_id: str, kind: str) -> dict[str, float]:
    canonical_root = _canonical_library_root(project_root, env_id=env_id, kind=kind)
    if canonical_root is None:
        return {}
    return _max_training_reward_by_label_for_library_root(canonical_root)


def _max_training_reward_by_label_for_library_root(run_dir: Path) -> dict[str, float]:
    rewards_by_label: dict[str, float] = {}
    pattern_library_path = run_dir / "pattern_library.json"
    pattern_library = _load_json(pattern_library_path)
    detectors = pattern_library.get("detectors") if isinstance(pattern_library.get("detectors"), list) else []
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


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None




def _extract_train_scores(payload: Any) -> list[float]:
    scores: list[float] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "train_score":
                numeric = _safe_float(value)
                if numeric is not None:
                    scores.append(numeric)
                    continue
            scores.extend(_extract_train_scores(value))
    elif isinstance(payload, list):
        for item in payload:
            scores.extend(_extract_train_scores(item))
    return scores
