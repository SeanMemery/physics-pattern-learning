from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPECIAL_PATTERN_DIRS = {
    "unsupervised",
    "automatic",
    "image_only_evaluations",
    "reward_program_experiments",
}


@dataclass(frozen=True)
class DatasetEntry:
    env_id: str
    dataset_name: str
    root: Path
    manifest_path: Path
    total_target_count: int
    total_completed_count: int
    split_summaries: list[str]
    updated_at: datetime


@dataclass(frozen=True)
class DatasetTraceEntry:
    split_name: str
    sample_name: str
    sample_dir: Path
    trace_path: Path | None
    quality_path: Path | None
    status: str
    trace_id: str
    task_id: str
    sample_id: int | None
    absolute_sample_index: int | None
    warnings: list[str]
    error: str | None
    updated_at: datetime


@dataclass(frozen=True)
class DatasetSplitDetail:
    name: str
    target_count: int
    completed_count: int
    traces: list[DatasetTraceEntry]


@dataclass(frozen=True)
class DatasetDetail:
    env_id: str
    dataset_name: str
    dataset_id: str
    root: Path
    manifest_path: Path
    index_path: Path | None
    total_target_count: int
    total_completed_count: int
    split_summaries: list[str]
    updated_at: datetime
    notes: list[str]
    env_ids: list[str]
    splits: list[DatasetSplitDetail]


@dataclass(frozen=True)
class DatasetTraceDetail:
    env_id: str
    dataset_name: str
    dataset_id: str
    root: Path
    manifest_path: Path
    index_path: Path | None
    split_name: str
    sample_name: str
    sample_dir: Path
    trace_entry: DatasetTraceEntry | None
    trace_path: Path | None
    quality_path: Path | None
    trace_payload: dict[str, Any]
    quality_payload: dict[str, Any]
    media_files: list[Path]
    extra_files: list[Path]
    updated_at: datetime


@dataclass(frozen=True)
class LibraryEntry:
    env_id: str
    run_id: str
    root: Path
    library_path: Path
    workflow_kind: str
    detector_count: int
    created_at: datetime | None
    updated_at: datetime


@dataclass(frozen=True)
class RewardResultEntry:
    env_id: str
    run_id: str
    root: Path
    summary_path: Path
    reward_kind: str
    source_run_id: str
    source_run_kind: str
    model_name: str
    headline: str
    updated_at: datetime


def list_environment_datasets(project_root: Path, env_id: str) -> list[DatasetEntry]:
    dataset_root = project_root / "data" / "datasets" / env_id
    if not dataset_root.is_dir():
        return []
    entries: list[DatasetEntry] = []
    for manifest_path in sorted(dataset_root.glob("*/dataset_manifest.json")):
        dataset_dir = manifest_path.parent
        payload = _load_json(manifest_path)
        splits_payload = payload.get("splits")
        split_summaries: list[str] = []
        total_target_count = 0
        total_completed_count = 0
        if isinstance(splits_payload, list):
            for raw_split in splits_payload:
                if not isinstance(raw_split, dict):
                    continue
                split_name = str(raw_split.get("name") or "").strip()
                target_count = _coerce_int(raw_split.get("sample_count"))
                completed_count = _coerce_int(raw_split.get("completed_samples"))
                total_target_count += target_count or 0
                total_completed_count += completed_count or 0
                if split_name:
                    split_summaries.append(
                        f"{split_name}: {completed_count or 0}/{target_count or 0}"
                    )
        entries.append(
            DatasetEntry(
                env_id=env_id,
                dataset_name=str(payload.get("dataset_name") or dataset_dir.name),
                root=dataset_dir,
                manifest_path=manifest_path,
                total_target_count=total_target_count,
                total_completed_count=total_completed_count,
                split_summaries=split_summaries,
                updated_at=_mtime(manifest_path),
            )
        )
    return sorted(entries, key=lambda item: (item.updated_at, item.dataset_name), reverse=True)


def get_environment_dataset(project_root: Path, env_id: str, dataset_key: str) -> DatasetDetail | None:
    dataset_dir = project_root / "data" / "datasets" / env_id / dataset_key
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        return None

    manifest_payload = _load_json(manifest_path)
    index_path = dataset_dir / "dataset_index.json"
    index_payload = _load_json(index_path) if index_path.is_file() else {}
    traces_by_split: dict[str, list[DatasetTraceEntry]] = {}

    raw_entries = index_payload.get("entries")
    if isinstance(raw_entries, list):
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                continue
            split_name = str(raw_entry.get("split_name") or "").strip() or "unknown"
            sample_dir_rel = str(raw_entry.get("sample_dir") or "").strip()
            sample_name = Path(sample_dir_rel).name if sample_dir_rel else _fallback_sample_name(raw_entry)
            sample_dir = dataset_dir / sample_dir_rel if sample_dir_rel else dataset_dir / split_name / sample_name
            trace_path = _dataset_relpath(dataset_dir, raw_entry.get("trace_relpath"))
            quality_path = _dataset_relpath(dataset_dir, raw_entry.get("quality_relpath"))
            updated_candidates = [
                candidate
                for candidate in [trace_path, quality_path, sample_dir]
                if isinstance(candidate, Path) and candidate.exists()
            ]
            updated_at = _mtime(updated_candidates[0]) if updated_candidates else _mtime(manifest_path)
            traces_by_split.setdefault(split_name, []).append(
                DatasetTraceEntry(
                    split_name=split_name,
                    sample_name=sample_name,
                    sample_dir=sample_dir,
                    trace_path=trace_path if trace_path and trace_path.exists() else None,
                    quality_path=quality_path if quality_path and quality_path.exists() else None,
                    status=str(raw_entry.get("status") or "unknown"),
                    trace_id=str(raw_entry.get("trace_id") or ""),
                    task_id=str(raw_entry.get("task_id") or ""),
                    sample_id=_coerce_int(raw_entry.get("sample_id")),
                    absolute_sample_index=_coerce_int(raw_entry.get("absolute_sample_index")),
                    warnings=[str(item) for item in raw_entry.get("warnings", []) if str(item).strip()]
                    if isinstance(raw_entry.get("warnings"), list)
                    else [],
                    error=_coerce_str(raw_entry.get("error")),
                    updated_at=updated_at,
                )
            )

    split_summaries: list[str] = []
    total_target_count = 0
    total_completed_count = 0
    split_details: list[DatasetSplitDetail] = []
    raw_splits = manifest_payload.get("splits")
    seen_splits: set[str] = set()
    if isinstance(raw_splits, list):
        for raw_split in raw_splits:
            if not isinstance(raw_split, dict):
                continue
            split_name = str(raw_split.get("name") or "").strip()
            if not split_name:
                continue
            seen_splits.add(split_name)
            target_count = _coerce_int(raw_split.get("sample_count")) or 0
            completed_count = _coerce_int(raw_split.get("completed_samples")) or 0
            total_target_count += target_count
            total_completed_count += completed_count
            split_summaries.append(f"{split_name}: {completed_count}/{target_count}")
            traces = sorted(
                traces_by_split.get(split_name, []),
                key=lambda item: (
                    item.sample_id if item.sample_id is not None else 10**9,
                    item.sample_name,
                ),
            )
            split_details.append(
                DatasetSplitDetail(
                    name=split_name,
                    target_count=target_count,
                    completed_count=completed_count,
                    traces=traces,
                )
            )

    for split_name, traces in sorted(traces_by_split.items()):
        if split_name in seen_splits:
            continue
        split_details.append(
            DatasetSplitDetail(
                name=split_name,
                target_count=len(traces),
                completed_count=sum(1 for item in traces if item.status == "completed"),
                traces=sorted(
                    traces,
                    key=lambda item: (
                        item.sample_id if item.sample_id is not None else 10**9,
                        item.sample_name,
                    ),
                ),
            )
        )

    notes = [str(item) for item in manifest_payload.get("notes", []) if str(item).strip()] if isinstance(manifest_payload.get("notes"), list) else []
    env_ids = [str(item) for item in manifest_payload.get("env_ids", []) if str(item).strip()] if isinstance(manifest_payload.get("env_ids"), list) else [env_id]

    return DatasetDetail(
        env_id=env_id,
        dataset_name=str(manifest_payload.get("dataset_name") or dataset_dir.name),
        dataset_id=str(manifest_payload.get("dataset_id") or dataset_dir.name),
        root=dataset_dir,
        manifest_path=manifest_path,
        index_path=index_path if index_path.is_file() else None,
        total_target_count=total_target_count,
        total_completed_count=total_completed_count,
        split_summaries=split_summaries,
        updated_at=_mtime(manifest_path),
        notes=notes,
        env_ids=env_ids,
        splits=split_details,
    )


def get_environment_dataset_trace(
    project_root: Path,
    env_id: str,
    dataset_key: str,
    split_name: str,
    sample_name: str,
) -> DatasetTraceDetail | None:
    detail = get_environment_dataset(project_root, env_id, dataset_key)
    if detail is None:
        return None
    sample_dir = detail.root / split_name / sample_name
    if not sample_dir.is_dir():
        return None

    trace_entry = next(
        (
            item
            for split in detail.splits
            if split.name == split_name
            for item in split.traces
            if item.sample_name == sample_name
        ),
        None,
    )
    trace_path = sample_dir / "trace.json"
    quality_path = sample_dir / "quality.json"
    media_files: list[Path] = []
    extra_files: list[Path] = []
    for child in sorted(sample_dir.iterdir()):
        if not child.is_file():
            continue
        if child.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".mp4", ".webm", ".mov", ".m4v"}:
            media_files.append(child)
        elif child.name not in {"trace.json", "quality.json"}:
            extra_files.append(child)

    updated_candidates = [
        candidate
        for candidate in [trace_path if trace_path.exists() else None, quality_path if quality_path.exists() else None, sample_dir]
        if isinstance(candidate, Path) and candidate.exists()
    ]

    return DatasetTraceDetail(
        env_id=env_id,
        dataset_name=detail.dataset_name,
        dataset_id=detail.dataset_id,
        root=detail.root,
        manifest_path=detail.manifest_path,
        index_path=detail.index_path,
        split_name=split_name,
        sample_name=sample_name,
        sample_dir=sample_dir,
        trace_entry=trace_entry,
        trace_path=trace_path if trace_path.is_file() else None,
        quality_path=quality_path if quality_path.is_file() else None,
        trace_payload=_load_json(trace_path) if trace_path.is_file() else {},
        quality_payload=_load_json(quality_path) if quality_path.is_file() else {},
        media_files=media_files,
        extra_files=extra_files,
        updated_at=_mtime(updated_candidates[0]) if updated_candidates else detail.updated_at,
    )


def list_environment_libraries(project_root: Path, env_id: str) -> list[LibraryEntry]:
    pattern_root = project_root / "data" / "patterns" / env_id
    if not pattern_root.is_dir():
        return []
    entries: list[LibraryEntry] = []
    for library_path in sorted(pattern_root.glob("**/pattern_library.json")):
        run_root = library_path.parent
        if any(part == "reward_program_experiments" for part in run_root.parts):
            continue
        payload = _load_json(library_path)
        detectors = payload.get("detectors")
        detector_count = len(detectors) if isinstance(detectors, list) else 0
        created_at = _parse_iso_datetime(payload.get("created_at"))
        entries.append(
            LibraryEntry(
                env_id=env_id,
                run_id=run_root.name,
                root=run_root,
                library_path=library_path,
                workflow_kind=_library_workflow_kind(pattern_root, run_root),
                detector_count=detector_count,
                created_at=created_at,
                updated_at=_mtime(library_path),
            )
        )
    return sorted(entries, key=lambda item: (item.updated_at, item.run_id), reverse=True)


def list_environment_reward_results(project_root: Path, env_id: str) -> list[RewardResultEntry]:
    if env_id != "pooltool":
        return []
    root = project_root / "data" / "patterns" / "pooltool" / "reward_program_experiments"
    if not root.is_dir():
        return []
    entries: list[RewardResultEntry] = []

    optimization_root = root / "optimization"
    if optimization_root.is_dir():
        for summary_path in sorted(optimization_root.glob("*/summary.json")):
            payload = _load_json(summary_path)
            entries.append(
                RewardResultEntry(
                    env_id=env_id,
                    run_id=summary_path.parent.name,
                    root=summary_path.parent,
                    summary_path=summary_path,
                    reward_kind="optimization",
                    source_run_id=str(payload.get("source_run_id") or ""),
                    source_run_kind=str(payload.get("source_run_kind") or ""),
                    model_name=str(payload.get("model_name") or ""),
                    headline=_reward_headline(payload),
                    updated_at=_mtime(summary_path),
                )
            )

    value_model_root = root / "value_models"
    if value_model_root.is_dir():
        for summary_path in sorted(value_model_root.glob("*/summary.json")):
            payload = _load_json(summary_path)
            entries.append(
                RewardResultEntry(
                    env_id=env_id,
                    run_id=summary_path.parent.name,
                    root=summary_path.parent,
                    summary_path=summary_path,
                    reward_kind="value-model",
                    source_run_id=str(payload.get("source_run_id") or ""),
                    source_run_kind=str(payload.get("source_run_kind") or ""),
                    model_name=str(payload.get("model_name") or ""),
                    headline=_reward_headline(payload),
                    updated_at=_mtime(summary_path),
                )
            )

    optimization_suite_root = root / "optimization_suites"
    if optimization_suite_root.is_dir():
        for summary_path in sorted(optimization_suite_root.glob("*/summary.json")):
            payload = _load_json(summary_path)
            entries.append(
                RewardResultEntry(
                    env_id=env_id,
                    run_id=summary_path.parent.name,
                    root=summary_path.parent,
                    summary_path=summary_path,
                    reward_kind="optimization-suite",
                    source_run_id=str(payload.get("source_run_id") or ""),
                    source_run_kind=str(payload.get("source_run_kind") or ""),
                    model_name=str(payload.get("model_name") or ""),
                    headline=_reward_headline(payload),
                    updated_at=_mtime(summary_path),
                )
            )

    value_model_sweep_root = root / "value_model_sweeps"
    if value_model_sweep_root.is_dir():
        for summary_path in sorted(value_model_sweep_root.glob("*/summary.json")):
            payload = _load_json(summary_path)
            entries.append(
                RewardResultEntry(
                    env_id=env_id,
                    run_id=summary_path.parent.name,
                    root=summary_path.parent,
                    summary_path=summary_path,
                    reward_kind="value-model-sweep",
                    source_run_id=str(payload.get("source_run_id") or ""),
                    source_run_kind=str(payload.get("source_run_kind") or ""),
                    model_name=str(payload.get("model_name") or ""),
                    headline=_reward_headline(payload),
                    updated_at=_mtime(summary_path),
                )
            )

    single_scene_root = root / "single_scene"
    if single_scene_root.is_dir():
        for summary_path in sorted(single_scene_root.glob("*/summary.json")):
            payload = _load_json(summary_path)
            entries.append(
                RewardResultEntry(
                    env_id=env_id,
                    run_id=summary_path.parent.name,
                    root=summary_path.parent,
                    summary_path=summary_path,
                    reward_kind="single-scene",
                    source_run_id=str(payload.get("source_run_id") or ""),
                    source_run_kind=str(payload.get("source_run_kind") or ""),
                    model_name=str(payload.get("model_name") or ""),
                    headline=_reward_headline(payload),
                    updated_at=_mtime(summary_path),
                )
            )

    return sorted(entries, key=lambda item: (item.updated_at, item.run_id), reverse=True)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _dataset_relpath(dataset_dir: Path, raw_value: Any) -> Path | None:
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    return dataset_dir / raw_value.strip()


def _fallback_sample_name(payload: dict[str, Any]) -> str:
    sample_id = _coerce_int(payload.get("sample_id"))
    if sample_id is not None:
        return f"sample_{sample_id:03d}"
    return "sample"


def _coerce_str(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


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


def _library_workflow_kind(pattern_root: Path, run_root: Path) -> str:
    try:
        relative = run_root.relative_to(pattern_root)
    except ValueError:
        return "library"
    parts = relative.parts
    if len(parts) > 1 and parts[0] in SPECIAL_PATTERN_DIRS:
        return parts[0]
    return "supervised"


def _reward_headline(payload: dict[str, Any]) -> str:
    scene_count = _coerce_int(payload.get("scene_count")) or _coerce_int(payload.get("validation_scene_count"))
    outcome_count = _coerce_int(payload.get("outcome_count"))
    best_success = payload.get("best_validation_success_rate")
    if isinstance(best_success, (int, float)):
        return f"best success {float(best_success):.3f}"
    budget_rows = payload.get("budget_aggregates")
    if isinstance(budget_rows, list):
        best_rate: float | None = None
        for item in budget_rows:
            if not isinstance(item, dict):
                continue
            candidate = item.get("success_rate")
            if isinstance(candidate, (int, float)):
                best_rate = max(best_rate or float(candidate), float(candidate))
        if best_rate is not None:
            return f"best success {best_rate:.3f}"
    if scene_count is not None:
        return f"{scene_count} scenes"
    if outcome_count is not None:
        return f"{outcome_count} outcomes"
    return "reward program result"
