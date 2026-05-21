from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import JobRecord


@dataclass(frozen=True)
class LibraryAblationSeedResult:
    seed: int
    evaluation_run_id: str
    root: Path
    status: str
    task_count: int
    success_count: int
    enabled_labels: list[str] = field(default_factory=list)
    held_out_labels: list[str] = field(default_factory=list)

    @property
    def success_ratio(self) -> float | None:
        if self.task_count <= 0:
            return None
        return self.success_count / self.task_count


@dataclass(frozen=True)
class LibraryAblationCaseResult:
    case_id: str
    label: str
    kind: str
    enabled_labels: list[str]
    held_out_labels: list[str]
    explicit_label_count: int | None = None
    seed_results: list[LibraryAblationSeedResult] = field(default_factory=list)

    @property
    def label_count(self) -> int:
        if self.explicit_label_count is not None and self.explicit_label_count > 0:
            return self.explicit_label_count
        return len(self.enabled_labels)

    @property
    def planned_seed_count(self) -> int:
        return len(self.seed_results)

    @property
    def completed_seed_count(self) -> int:
        return sum(1 for item in self.seed_results if item.status == "completed" and item.success_ratio is not None)

    @property
    def active_seed_count(self) -> int:
        return sum(1 for item in self.seed_results if item.status == "running")

    @property
    def observed_task_count(self) -> int:
        return sum(item.task_count for item in self.seed_results if item.status in {"running", "completed"})

    @property
    def mean_success_ratio(self) -> float | None:
        values = [
            item.success_ratio
            for item in self.seed_results
            if item.status == "completed" and item.success_ratio is not None
        ]
        if not values:
            return None
        return sum(values) / len(values)

    @property
    def observed_mean_success_ratio(self) -> float | None:
        values = [
            item.success_ratio
            for item in self.seed_results
            if item.status in {"running", "completed"} and item.success_ratio is not None
        ]
        if not values:
            return None
        return sum(values) / len(values)

    @property
    def success_ratio_std(self) -> float:
        values = [
            item.success_ratio
            for item in self.seed_results
            if item.status == "completed" and item.success_ratio is not None
        ]
        if len(values) <= 1:
            return 0.0
        try:
            return float(statistics.pstdev(values))
        except statistics.StatisticsError:
            return 0.0


def case_counts_toward_label_series(case: LibraryAblationCaseResult) -> bool:
    normalized = str(case.kind or "").strip().lower()
    return normalized in {"baseline", "label_count", "random_holdout"}


@dataclass(frozen=True)
class LibraryAblationPoint:
    label_count: int
    mean_success_ratio: float
    mean_success_ratio_std: float
    case_count: int
    completed_seed_count: int


@dataclass(frozen=True)
class LibraryAblationDetail:
    env_id: str
    ablation_kind: str
    source_run_dir: Path
    source_run_id: str
    run_id: str
    root: Path
    template_name: str
    benchmark_preset: str
    available_labels: list[str]
    case_results: list[LibraryAblationCaseResult] = field(default_factory=list)
    raw_request_payload: dict[str, Any] = field(default_factory=dict)
    raw_manifest_payload: dict[str, Any] = field(default_factory=dict)

    @property
    def total_case_count(self) -> int:
        return len(self.case_results)

    @property
    def total_seed_run_count(self) -> int:
        return sum(case.planned_seed_count for case in self.case_results)

    @property
    def completed_seed_run_count(self) -> int:
        return sum(case.completed_seed_count for case in self.case_results)

    @property
    def active_seed_run_count(self) -> int:
        return sum(case.active_seed_count for case in self.case_results)

    @property
    def observed_task_count(self) -> int:
        return sum(case.observed_task_count for case in self.case_results)

    @property
    def baseline_case(self) -> LibraryAblationCaseResult | None:
        return next((item for item in self.case_results if item.kind == "baseline"), None)

    @property
    def aggregate_points(self) -> list[LibraryAblationPoint]:
        by_label_count: dict[int, list[LibraryAblationCaseResult]] = {}
        for case in self.case_results:
            if case.mean_success_ratio is None:
                continue
            by_label_count.setdefault(case.label_count, []).append(case)
        points: list[LibraryAblationPoint] = []
        for label_count, cases in sorted(by_label_count.items()):
            seed_ratios = [
                item.success_ratio
                for case in cases
                for item in case.seed_results
                if item.status == "completed" and item.success_ratio is not None
            ]
            if not seed_ratios:
                continue
            points.append(
                LibraryAblationPoint(
                    label_count=label_count,
                    mean_success_ratio=sum(seed_ratios) / len(seed_ratios),
                    mean_success_ratio_std=(
                        float(statistics.pstdev(seed_ratios))
                        if len(seed_ratios) > 1
                        else 0.0
                    ),
                    case_count=len(cases),
                    completed_seed_count=sum(case.completed_seed_count for case in cases),
                )
            )
        return points


def load_library_ablation_job_detail(
    job: JobRecord,
    *,
    project_root: Path,
) -> LibraryAblationDetail | None:
    if str(job.spec.action or "").strip() != "evaluate-library-case":
        return None
    metadata = dict(job.spec.metadata or {})
    env_id = str(metadata.get("env_id") or "").strip().lower()
    if not env_id:
        return None
    ablation_kind = str(metadata.get("library_ablation_kind") or "").strip().lower()
    source_run_dir = _resolve_source_run_dir(job, project_root=project_root)
    if source_run_dir is None:
        return None
    eval_root = _resolve_eval_root(job, project_root=project_root, source_run_dir=source_run_dir)
    if eval_root is None:
        return None
    return load_library_ablation_detail(
        eval_root=eval_root,
        env_id=env_id,
        source_run_dir=source_run_dir,
        ablation_kind=ablation_kind or _infer_ablation_kind(source_run_dir),
    )


def load_library_ablation_detail(
    *,
    eval_root: Path,
    env_id: str,
    source_run_dir: Path,
    ablation_kind: str,
) -> LibraryAblationDetail | None:
    request_path = eval_root / "request.json"
    manifest_path = eval_root / "manifest.json"
    if not request_path.is_file() and not manifest_path.is_file():
        return None
    request_payload = _load_json(request_path) if request_path.is_file() else {}
    manifest_payload = _load_json(manifest_path) if manifest_path.is_file() else {}
    manifest_payloads = _collect_manifest_payloads(eval_root, manifest_payload)
    available_labels = _merged_available_labels(manifest_payloads)
    case_results: list[LibraryAblationCaseResult] = []
    for payload in manifest_payloads:
        case_results.extend(_load_manifest_case_results(eval_root, payload))
    return LibraryAblationDetail(
        env_id=env_id,
        ablation_kind=ablation_kind,
        source_run_dir=source_run_dir,
        source_run_id=source_run_dir.name,
        run_id=eval_root.name,
        root=eval_root,
        template_name=_coerce_str(request_payload.get("template_name")) or "quick-eval",
        benchmark_preset=_coerce_str(request_payload.get("benchmark_preset")) or "quick-eval",
        available_labels=available_labels,
        case_results=case_results,
        raw_request_payload=request_payload if isinstance(request_payload, dict) else {},
        raw_manifest_payload=manifest_payload if isinstance(manifest_payload, dict) else {},
    )


def _collect_manifest_payloads(
    eval_root: Path,
    manifest_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()

    def _append_path(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen_paths or not resolved.is_file():
            return
        seen_paths.add(resolved)
        payload = _load_json(resolved)
        if payload:
            payloads.append(payload)

    if manifest_payload:
        payloads.append(manifest_payload)
    for relative_path in _coerce_str_list(manifest_payload.get("legacy_manifest_paths")):
        _append_path(eval_root / relative_path)
    for candidate in sorted(eval_root.glob("manifest.legacy*.json")):
        _append_path(candidate)
    return payloads


def _merged_available_labels(manifest_payloads: list[dict[str, Any]]) -> list[str]:
    labels: set[str] = set()
    for payload in manifest_payloads:
        labels.update(_coerce_str_list(payload.get("available_labels")))
    return sorted(labels)


def _load_manifest_case_results(
    eval_root: Path,
    manifest_payload: dict[str, Any],
) -> list[LibraryAblationCaseResult]:
    case_results: list[LibraryAblationCaseResult] = []
    raw_cases = manifest_payload.get("cases")
    if not isinstance(raw_cases, list):
        return case_results
    for item in raw_cases:
        if not isinstance(item, dict):
            continue
        case_id = _coerce_str(item.get("case_id"))
        if not case_id:
            continue
        seed_results: list[LibraryAblationSeedResult] = []
        raw_seed_runs = item.get("seed_runs")
        if isinstance(raw_seed_runs, list):
            for raw_seed_run in raw_seed_runs:
                if not isinstance(raw_seed_run, dict):
                    continue
                seed = _coerce_int(raw_seed_run.get("seed"))
                evaluation_run_id = _coerce_str(raw_seed_run.get("evaluation_run_id"))
                relative_root = _coerce_str(raw_seed_run.get("relative_root"))
                if seed is None or not evaluation_run_id or not relative_root:
                    continue
                child_root = (eval_root / relative_root).resolve()
                status, task_count, success_count = _load_child_eval_metrics(child_root)
                seed_results.append(
                    LibraryAblationSeedResult(
                        seed=seed,
                        evaluation_run_id=evaluation_run_id,
                        root=child_root,
                        status=status,
                        task_count=task_count,
                        success_count=success_count,
                        enabled_labels=_coerce_str_list(raw_seed_run.get("enabled_labels")),
                        held_out_labels=_coerce_str_list(raw_seed_run.get("held_out_labels")),
                    )
                )
        explicit_label_count = _coerce_int(item.get("label_count"))
        enabled_labels = _coerce_str_list(item.get("enabled_labels"))
        if explicit_label_count is None and enabled_labels:
            explicit_label_count = len(enabled_labels)
        case_results.append(
            LibraryAblationCaseResult(
                case_id=case_id,
                label=_coerce_str(item.get("label")) or case_id,
                kind=_coerce_str(item.get("kind")) or "baseline",
                enabled_labels=enabled_labels,
                held_out_labels=_coerce_str_list(item.get("held_out_labels")),
                explicit_label_count=explicit_label_count,
                seed_results=seed_results,
            )
        )
    return case_results


def _resolve_source_run_dir(job: JobRecord, *, project_root: Path) -> Path | None:
    artifact_relpaths = [
        str(item).strip()
        for item in list(getattr(job, "artifact_relpaths", []) or [])
        if str(item).strip()
    ]
    for relpath in artifact_relpaths:
        if not relpath.startswith("data/patterns/"):
            continue
        candidate = _normalize_project_path(relpath, project_root=project_root)
        if candidate is not None and candidate.is_dir():
            return candidate
    metadata = dict(job.spec.metadata or {})
    request_payload = metadata.get("request") if isinstance(metadata.get("request"), dict) else {}
    raw = str(metadata.get("pattern_run_dir") or request_payload.get("pattern_run_dir") or "").strip()
    if not raw:
        return None
    return _normalize_project_path(raw, project_root=project_root)


def _resolve_eval_root(
    job: JobRecord,
    *,
    project_root: Path,
    source_run_dir: Path,
) -> Path | None:
    metadata = dict(job.spec.metadata or {})
    request_payload = metadata.get("request") if isinstance(metadata.get("request"), dict) else {}
    for key in ("library_evaluation_run_dir", "run_dir"):
        raw = str(metadata.get(key) or "").strip()
        if not raw:
            continue
        candidate = _normalize_project_path(raw, project_root=project_root)
        if candidate is not None:
            return candidate
    run_id = str(
        metadata.get("library_evaluation_run_id")
        or request_payload.get("run_id")
        or ""
    ).strip()
    library_evaluations_root = (source_run_dir / "library_evaluations").resolve()
    if run_id:
        return (library_evaluations_root / run_id).resolve()
    if not library_evaluations_root.is_dir():
        return None
    candidates = [
        child
        for child in library_evaluations_root.iterdir()
        if child.is_dir() and ((child / "manifest.json").is_file() or (child / "summary.json").is_file())
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda child: max(
            (child / "summary.json").stat().st_mtime if (child / "summary.json").exists() else 0.0,
            (child / "manifest.json").stat().st_mtime if (child / "manifest.json").exists() else 0.0,
            child.stat().st_mtime,
        ),
        reverse=True,
    )
    return candidates[0]


def _infer_ablation_kind(source_run_dir: Path) -> str:
    normalized = source_run_dir.as_posix().lower()
    if "/automatic/" in normalized or normalized.endswith("-automatic"):
        return "automatic"
    if "/unsupervised/" in normalized or normalized.endswith("-unsupervised"):
        return "unsupervised"
    return "supervised"


def _load_child_eval_metrics(child_root: Path) -> tuple[str, int, int]:
    if not child_root.is_dir():
        return ("pending", 0, 0)
    results_path = child_root / "results.json"
    summary_path = child_root / "summary.json"
    if results_path.is_file():
        payload = _load_json(results_path)
        metrics = _extract_eval_metrics(payload)
        if metrics is not None:
            return ("completed", metrics[0], metrics[1])
    if summary_path.is_file():
        payload = _load_json(summary_path)
        metrics = _extract_eval_metrics(payload)
        if metrics is not None:
            return ("completed", metrics[0], metrics[1])
    partial_metrics = _extract_partial_eval_metrics(child_root)
    if partial_metrics is not None:
        return ("running", partial_metrics[0], partial_metrics[1])
    return ("pending", 0, 0)


def _extract_partial_eval_metrics(child_root: Path) -> tuple[int, int] | None:
    tasks_root = child_root / "tasks"
    if not tasks_root.is_dir():
        return None
    task_count = 0
    success_count = 0
    for result_path in tasks_root.rglob("task_result.json"):
        payload = _load_json(result_path)
        if not isinstance(payload, dict):
            continue
        success = payload.get("success")
        if isinstance(success, bool):
            task_count += 1
            success_count += 1 if success else 0
    if task_count <= 0:
        return None
    return (task_count, success_count)


def _extract_eval_metrics(payload: dict[str, Any]) -> tuple[int, int] | None:
    if not isinstance(payload, dict):
        return None
    results = payload.get("results")
    if isinstance(results, list):
        task_count = 0
        success_count = 0
        for item in results:
            if not isinstance(item, dict):
                continue
            success = item.get("success")
            if isinstance(success, bool):
                task_count += 1
                success_count += 1 if success else 0
        if task_count > 0:
            return (task_count, success_count)
    env_summaries = payload.get("env_summaries")
    if isinstance(env_summaries, dict):
        summaries = [item for item in env_summaries.values() if isinstance(item, dict)]
    elif isinstance(env_summaries, list):
        summaries = [item for item in env_summaries if isinstance(item, dict)]
    else:
        summaries = []
    if summaries:
        task_count = 0
        success_count = 0
        for item in summaries:
            task_count += _coerce_int(item.get("task_count") or item.get("total_tasks")) or 0
            success_count += _coerce_int(item.get("success_count") or item.get("successful_tasks")) or 0
        if task_count > 0:
            return (task_count, success_count)
    task_count = _coerce_int(payload.get("task_count") or payload.get("total_tasks"))
    success_count = _coerce_int(payload.get("success_count") or payload.get("successful_tasks"))
    if task_count and success_count is not None:
        return (task_count, success_count)
    return None


def _normalize_project_path(raw: str, *, project_root: Path) -> Path | None:
    candidate = Path(str(raw).strip())
    if not candidate:
        return None
    resolved = candidate if candidate.is_absolute() else (project_root / candidate)
    try:
        return resolved.resolve()
    except OSError:
        return None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _coerce_str(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
