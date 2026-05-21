from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets.index import load_dataset_index
from dashboard.models import (
    CandidateSummary,
    DatasetDetailSummary,
    DatasetIndexedSampleSummary,
    DatasetSampleSummary,
    DatasetSplitDetailSummary,
    DatasetSplitSummary,
    DatasetSummary,
    EnvironmentDashboardSummary,
    EvaluationLabelSelectionSummary,
    EvaluationMetricSummary,
    EvaluationPromptImageSummary,
    EvaluationRunSummary,
    EvaluationTraceAnnotationRow,
    EvaluationTaskDetailSummary,
    EvaluationTaskRecordSummary,
    EvaluationTaskSummary,
    LibraryEvaluationCaseSummary,
    LibraryEvaluationSeedSummary,
    LibraryEvaluationSummary,
    LLMCallSummary,
    LabelSummary,
    PatternEvolutionRecord,
    PatternEvolutionCandidateSummary,
    PatternEvolutionIterationSummary,
    PatternEvolutionLabelSummary,
    PatternLabelTelemetrySummary,
    PatternRunDetailSummary,
    PatternRunSummary,
    PooltoolRewardOptimizationBudgetSummary,
    PooltoolRewardOptimizationSummary,
    PooltoolRewardValueModelPointSummary,
    PooltoolRewardValueModelSummary,
    ProposedLabelSummary,
    SupervisedLibraryDetailSummary,
    SupervisedLibrarySummary,
    TargetLabelConfigSummary,
    UnsupervisedLibraryDetailSummary,
    UnsupervisedLibrarySummary,
)
from paths import ArtifactLayout, get_artifact_layout
from pattern_library_registry import load_pattern_library
from pattern_library_paths import (
    AUTOMATIC_RUNS_DIRNAME,
    PATTERN_LIBRARY_FILENAME,
    UNSUPERVISED_RUNS_DIRNAME,
)


SUPPORTED_ENVS = ("phyre", "iphyre", "kinetix", "pooltool")
DATASET_SPLITS = ("train", "validation", "test")
TRACE_FILENAME = "trace.json"
QUALITY_FILENAME = "quality.json"
EVIDENCE_FILENAME = "trace.evidence.json"
SCENE_IMAGE_FILENAME = "initial_scene.png"
SIMULATION_VIDEO_FILENAME = "simulation.mp4"
RESERVED_PATTERN_RUN_DIRNAMES = {AUTOMATIC_RUNS_DIRNAME, UNSUPERVISED_RUNS_DIRNAME}
LIBRARY_EVALUATIONS_DIRNAME = "library_evaluations"
IMAGE_ONLY_EVALUATIONS_DIRNAME = "image_only_evaluations"
REWARD_PROGRAM_EXPERIMENTS_DIRNAME = "reward_program_experiments"
_PATTERN_ANNOTATION_LINE_RE = re.compile(
    r"^\[(step|steps)=(\d+)(?:-(\d+))?\]\s+(.+?)(?:\s+\((.*)\))?$"
)


def _is_saved_pattern_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / PATTERN_LIBRARY_FILENAME).is_file():
        return True
    run_json_path = path / "run.json"
    return run_json_path.is_file()


def _env_id_for_run_root(run_root: Path) -> str:
    return run_root.parent.name


def load_dashboard_home(
    *, layout: ArtifactLayout | None = None
) -> list[EnvironmentDashboardSummary]:
    resolved_layout = layout or get_artifact_layout()
    return [
        load_environment_summary(env_id, layout=resolved_layout)
        for env_id in SUPPORTED_ENVS
    ]


def load_environment_summary(
    env_id: str,
    *,
    layout: ArtifactLayout | None = None,
) -> EnvironmentDashboardSummary:
    resolved_layout = layout or get_artifact_layout()
    supervised_libraries = list_supervised_libraries(env_id, layout=resolved_layout)
    automatic_libraries = list_automatic_libraries(env_id, layout=resolved_layout)
    unsupervised_libraries = list_unsupervised_libraries(env_id, layout=resolved_layout)
    return EnvironmentDashboardSummary(
        env_id=env_id,
        datasets=list_environment_datasets(env_id, layout=resolved_layout),
        pattern_runs=supervised_libraries,
        target_labels=load_target_label_config(env_id, layout=resolved_layout),
        supervised_libraries=supervised_libraries,
        automatic_libraries=automatic_libraries,
        unsupervised_libraries=unsupervised_libraries,
        image_only_evaluations=list_image_only_evaluations(
            env_id, layout=resolved_layout
        ),
    )


def load_target_label_config(
    env_id: str,
    *,
    layout: ArtifactLayout | None = None,
) -> TargetLabelConfigSummary | None:
    resolved_layout = layout or get_artifact_layout()
    path = resolved_layout.config_root / env_id / "labels.txt"
    if not path.is_file():
        return None
    raw_text = path.read_text(encoding="utf-8")
    labels = _parse_target_labels(raw_text)
    return TargetLabelConfigSummary(
        env_id=env_id,
        path=path,
        labels=labels,
        raw_text=raw_text,
    )


def list_environment_datasets(
    env_id: str,
    *,
    layout: ArtifactLayout | None = None,
) -> list[DatasetSummary]:
    resolved_layout = layout or get_artifact_layout()
    env_root = resolved_layout.data_root / "datasets" / env_id
    if not env_root.is_dir():
        return []
    summaries: list[DatasetSummary] = []
    for dataset_root in sorted(
        (path for path in env_root.iterdir() if path.is_dir()), key=lambda p: p.name
    ):
        summaries.append(load_dataset_summary(env_id, dataset_root))
    return summaries


def list_image_only_evaluations(
    env_id: str,
    *,
    layout: ArtifactLayout | None = None,
) -> list[EvaluationRunSummary]:
    resolved_layout = layout or get_artifact_layout()
    eval_root = (
        resolved_layout.data_root
        / "patterns"
        / env_id
        / IMAGE_ONLY_EVALUATIONS_DIRNAME
    )
    if not eval_root.is_dir():
        return []
    return [
        load_evaluation_run_summary(item)
        for item in sorted((path for path in eval_root.iterdir() if path.is_dir()), key=lambda p: p.name, reverse=True)
    ]


def list_pooltool_reward_optimization_experiments(
    *,
    layout: ArtifactLayout | None = None,
) -> list[PooltoolRewardOptimizationSummary]:
    resolved_layout = layout or get_artifact_layout()
    root = (
        resolved_layout.data_root
        / "patterns"
        / "pooltool"
        / REWARD_PROGRAM_EXPERIMENTS_DIRNAME
        / "optimization"
    )
    if not root.is_dir():
        return []
    summaries: list[PooltoolRewardOptimizationSummary] = []
    for run_root in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name, reverse=True):
        summary_path = run_root / "summary.json"
        if not summary_path.is_file():
            continue
        payload = _load_json(summary_path)
        budget_rows: list[PooltoolRewardOptimizationBudgetSummary] = []
        raw_rows = payload.get("budget_aggregates") if isinstance(payload, dict) else None
        if isinstance(raw_rows, list):
            for item in raw_rows:
                if not isinstance(item, dict):
                    continue
                budget_rows.append(
                    PooltoolRewardOptimizationBudgetSummary(
                        budget=_coerce_int(item.get("budget")) or 0,
                        reward_kind=str(item.get("reward_kind") or ""),
                        scene_count=_coerce_int(item.get("scene_count")) or 0,
                        success_rate=float(item.get("success_rate") or 0.0),
                        mean_reward=float(item.get("mean_reward") or 0.0),
                        seed_count=_coerce_int(item.get("seed_count")) or 0,
                    )
                )
        summaries.append(
            PooltoolRewardOptimizationSummary(
                run_id=run_root.name,
                root=run_root,
                goal_text=str(payload.get("goal_text") or ""),
                dsl_program=str(payload.get("dsl_program") or ""),
                model_name=str(payload.get("model_name") or ""),
                source_run_id=str(payload.get("source_run_id") or ""),
                source_run_kind=str(payload.get("source_run_kind") or ""),
                scene_count=_coerce_int(payload.get("scene_count")) or 0,
                seed_count=_coerce_int(payload.get("seed_count")) or 0,
                synthesized=bool(payload.get("synthesized")),
                budget_summaries=budget_rows,
            )
        )
    return summaries


def list_pooltool_reward_value_model_experiments(
    *,
    layout: ArtifactLayout | None = None,
) -> list[PooltoolRewardValueModelSummary]:
    resolved_layout = layout or get_artifact_layout()
    root = (
        resolved_layout.data_root
        / "patterns"
        / "pooltool"
        / REWARD_PROGRAM_EXPERIMENTS_DIRNAME
        / "value_models"
    )
    if not root.is_dir():
        return []
    summaries: list[PooltoolRewardValueModelSummary] = []
    for run_root in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name, reverse=True):
        summary_path = run_root / "summary.json"
        if not summary_path.is_file():
            continue
        payload = _load_json(summary_path)
        curve_rows: list[PooltoolRewardValueModelPointSummary] = []
        raw_curve = payload.get("eval_curve") if isinstance(payload, dict) else None
        if isinstance(raw_curve, list):
            for item in raw_curve:
                if not isinstance(item, dict):
                    continue
                curve_rows.append(
                    PooltoolRewardValueModelPointSummary(
                        epoch=_coerce_int(item.get("epoch")) or 0,
                        train_loss=float(item.get("train_loss") or 0.0),
                        validation_mean_reward=float(item.get("validation_mean_reward") or 0.0),
                        validation_success_rate=float(item.get("validation_success_rate") or 0.0),
                    )
                )
        summaries.append(
            PooltoolRewardValueModelSummary(
                run_id=run_root.name,
                root=run_root,
                goal_text=str(payload.get("goal_text") or ""),
                dsl_program=str(payload.get("dsl_program") or ""),
                model_name=str(payload.get("model_name") or ""),
                source_run_id=str(payload.get("source_run_id") or ""),
                source_run_kind=str(payload.get("source_run_kind") or ""),
                train_sample_count=_coerce_int(payload.get("train_sample_count")) or 0,
                validation_scene_count=_coerce_int(payload.get("validation_scene_count")) or 0,
                best_validation_mean_reward=float(payload.get("best_validation_mean_reward") or 0.0),
                best_validation_success_rate=float(payload.get("best_validation_success_rate") or 0.0),
                synthesized=bool(payload.get("synthesized")),
                eval_curve=curve_rows,
            )
        )
    return summaries


def load_dataset_summary(env_id: str, dataset_root: Path) -> DatasetSummary:
    manifest_path = dataset_root / "dataset_manifest.json"
    payload = _load_json(manifest_path) if manifest_path.is_file() else {}
    manifest_splits = (
        _manifest_splits_map(payload.get("splits", {})) if isinstance(payload, dict) else {}
    )
    splits: list[DatasetSplitSummary] = []
    total_completed = 0
    for split_name in DATASET_SPLITS:
        split_root = dataset_root / split_name
        manifest_split = (
            manifest_splits.get(split_name, {})
            if isinstance(manifest_splits, dict)
            else {}
        )
        completed_count = _count_completed_samples(split_root)
        total_completed += completed_count
        splits.append(
            DatasetSplitSummary(
                name=split_name,
                target_count=_coerce_int(manifest_split.get("sample_count"))
                if isinstance(manifest_split, dict)
                else None,
                manifest_completed_count=_coerce_int(
                    manifest_split.get("completed_samples")
                )
                if isinstance(manifest_split, dict)
                else None,
                completed_count=completed_count,
            )
        )
    return DatasetSummary(
        env_id=env_id,
        dataset_name=dataset_root.name,
        root=dataset_root,
        splits=splits,
        manifest_path=manifest_path if manifest_path.is_file() else None,
        total_completed_count=total_completed,
    )


def load_dataset_detail(
    env_id: str,
    dataset_name: str,
    *,
    layout: ArtifactLayout | None = None,
) -> DatasetDetailSummary:
    resolved_layout = layout or get_artifact_layout()
    dataset_root = resolved_layout.data_root / "datasets" / env_id / dataset_name
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest_payload = _load_json(manifest_path) if manifest_path.is_file() else {}
    manifest_splits = (
        _manifest_splits_map(manifest_payload.get("splits", {}))
        if isinstance(manifest_payload, dict)
        else {}
    )
    index_path = dataset_root / "dataset_index.json"
    entries_by_split: dict[str, list[DatasetIndexedSampleSummary]] = {
        split_name: [] for split_name in DATASET_SPLITS
    }
    if index_path.is_file():
        index_payload = load_dataset_index(index_path)
        for entry in index_payload.entries:
            if entry.env_id != env_id:
                continue
            sample_root = dataset_root / entry.sample_dir
            split_name = entry.split_name
            if split_name not in entries_by_split:
                entries_by_split[split_name] = []
            entries_by_split[split_name].append(
                _dataset_index_entry_summary(
                    sample_root=sample_root, entry_payload=entry
                )
            )
    splits: list[DatasetSplitDetailSummary] = []
    for split_name in DATASET_SPLITS:
        split_root = dataset_root / split_name
        manifest_split = (
            manifest_splits.get(split_name, {})
            if isinstance(manifest_splits, dict)
            else {}
        )
        split_entries = sorted(
            entries_by_split.get(split_name, []),
            key=lambda item: (item.absolute_sample_index, item.sample_name),
        )
        splits.append(
            DatasetSplitDetailSummary(
                name=split_name,
                target_count=_coerce_int(manifest_split.get("sample_count"))
                if isinstance(manifest_split, dict)
                else None,
                manifest_completed_count=_coerce_int(
                    manifest_split.get("completed_samples")
                )
                if isinstance(manifest_split, dict)
                else None,
                completed_count=_count_completed_samples(split_root),
                indexed_samples=split_entries,
            )
        )
    return DatasetDetailSummary(
        env_id=env_id,
        dataset_name=dataset_name,
        root=dataset_root,
        manifest_path=manifest_path if manifest_path.is_file() else None,
        index_path=index_path if index_path.is_file() else None,
        splits=splits,
        raw_manifest_payload=manifest_payload
        if isinstance(manifest_payload, dict)
        else {},
    )


def load_dataset_sample(
    env_id: str,
    dataset_name: str,
    split_name: str,
    sample_name: str,
    *,
    layout: ArtifactLayout | None = None,
) -> DatasetSampleSummary:
    resolved_layout = layout or get_artifact_layout()
    dataset_root = resolved_layout.data_root / "datasets" / env_id / dataset_name
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")
    sample_root = _resolve_dataset_sample_root(
        dataset_root=dataset_root,
        split_name=split_name,
        sample_name=sample_name,
    )
    trace_path = sample_root / TRACE_FILENAME
    quality_path = sample_root / QUALITY_FILENAME
    evidence_path = sample_root / EVIDENCE_FILENAME
    scene_image_paths = _discover_media_paths(
        sample_root,
        canonical_name=SCENE_IMAGE_FILENAME,
        pattern="initial_scene*.png",
    )
    simulation_paths = _discover_media_paths(
        sample_root,
        canonical_name=SIMULATION_VIDEO_FILENAME,
        pattern="simulation*.mp4",
    )
    return DatasetSampleSummary(
        env_id=env_id,
        dataset_name=dataset_name,
        split_name=split_name,
        sample_name=sample_root.name,
        sample_root=sample_root,
        trace_path=trace_path if trace_path.is_file() else None,
        quality_path=quality_path if quality_path.is_file() else None,
        evidence_path=evidence_path if evidence_path.is_file() else None,
        scene_image_path=scene_image_paths[0] if scene_image_paths else None,
        simulation_path=simulation_paths[0] if simulation_paths else None,
        scene_image_paths=scene_image_paths,
        simulation_paths=simulation_paths,
    )


def list_pattern_runs(
    env_id: str,
    *,
    layout: ArtifactLayout | None = None,
) -> list[PatternRunSummary]:
    return list_supervised_libraries(env_id, layout=layout)


def list_supervised_libraries(
    env_id: str,
    *,
    layout: ArtifactLayout | None = None,
) -> list[SupervisedLibrarySummary]:
    resolved_layout = layout or get_artifact_layout()
    env_root = resolved_layout.data_root / "patterns" / env_id
    if not env_root.is_dir():
        return []
    runs: list[SupervisedLibrarySummary] = []
    for run_root in sorted(
        (
            path
            for path in env_root.iterdir()
            if path.is_dir()
            and path.name not in RESERVED_PATTERN_RUN_DIRNAMES
            and _is_saved_pattern_run_dir(path)
        ),
        key=lambda p: p.name,
        reverse=True,
    ):
        runs.append(load_pattern_run_summary(env_id, run_root))
    return runs


def _is_automatic_run_payload(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    run_kind = _coerce_str(payload.get("run_kind"))
    if run_kind == "automatic":
        return True
    return bool(
        _coerce_str(payload.get("source_supervised_run_id"))
        or _coerce_str(payload.get("source_library_run_id"))
        or _coerce_str(payload.get("source_run_id"))
        or _coerce_str(payload.get("combined_library_path"))
        or _coerce_str(payload.get("unsupervised_library_path"))
    )


def _load_unsupervised_summary_from_root(
    env_id: str,
    run_root: Path,
    *,
    run_kind: str,
) -> UnsupervisedLibrarySummary:
    run_json_path = run_root / "run.json"
    run_payload = _load_json(run_json_path) if run_json_path.is_file() else {}
    source_run_id = None
    source_evaluation_run_id = None
    proposed_labels: list[str] = []
    proposed_label_details: list[ProposedLabelSummary] = []
    if isinstance(run_payload, dict):
        source_run_id = _coerce_str(
            run_payload.get("source_supervised_run_id")
            or run_payload.get("source_run_id")
            or run_payload.get("source_library_run_id")
        )
        source_evaluation_run_id = _coerce_str(run_payload.get("source_evaluation_run_id"))
        raw_labels = run_payload.get("proposed_labels")
        if isinstance(raw_labels, list):
            proposed_labels = [str(item) for item in raw_labels if str(item).strip()]
        raw_label_details = run_payload.get("proposed_label_details")
        if isinstance(raw_label_details, list):
            proposed_label_details = _coerce_proposed_label_details(raw_label_details)
    if not proposed_label_details:
        proposed_label_details_path = run_root / "proposed_labels.json"
        if proposed_label_details_path.is_file():
            proposed_label_details = _load_proposed_label_details_summary(
                proposed_label_details_path
            )
    if not proposed_labels:
        if proposed_label_details:
            proposed_labels = [item.label for item in proposed_label_details]
        else:
            proposed_labels_path = run_root / "proposed_labels.txt"
            if proposed_labels_path.is_file():
                proposed_labels = [
                    line.strip()
                    for line in proposed_labels_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
    if not proposed_label_details and proposed_labels:
        proposed_label_details = [
            ProposedLabelSummary(label=label, description=None)
            for label in proposed_labels
        ]
    evaluation_runs = (
        [
            load_evaluation_run_summary(eval_root)
            for eval_root in sorted((run_root / "deepphy_eval").glob("*"), key=lambda p: p.name, reverse=True)
            if eval_root.is_dir()
        ]
        if (run_root / "deepphy_eval").is_dir()
        else []
    )
    return UnsupervisedLibrarySummary(
        env_id=env_id,
        run_id=run_root.name,
        root=run_root,
        run_json_path=run_json_path if run_json_path.is_file() else None,
        run_kind=run_kind,
        source_supervised_run_id=source_run_id,
        source_evaluation_run_id=source_evaluation_run_id,
        proposed_labels=proposed_labels,
        proposed_label_details=proposed_label_details,
        learned_label_count=len([path for path in run_root.glob("label_*") if path.is_dir()]),
        evaluation_runs=evaluation_runs,
        created_at=run_root.name,
        raw_run_payload=run_payload if isinstance(run_payload, dict) else {},
    )


def list_automatic_libraries(
    env_id: str,
    *,
    layout: ArtifactLayout | None = None,
) -> list[UnsupervisedLibrarySummary]:
    resolved_layout = layout or get_artifact_layout()
    env_root = resolved_layout.data_root / "patterns" / env_id
    candidate_roots: list[Path] = []
    for dirname in (AUTOMATIC_RUNS_DIRNAME, UNSUPERVISED_RUNS_DIRNAME):
        root = env_root / dirname
        if not root.is_dir():
            continue
        candidate_roots.extend(
            path for path in root.iterdir() if path.is_dir() and _is_saved_pattern_run_dir(path)
        )
    runs: list[UnsupervisedLibrarySummary] = []
    for run_root in sorted(candidate_roots, key=lambda p: p.name, reverse=True):
        run_json_path = run_root / "run.json"
        run_payload = _load_json(run_json_path) if run_json_path.is_file() else {}
        if not _is_automatic_run_payload(run_payload):
            continue
        runs.append(
            _load_unsupervised_summary_from_root(env_id, run_root, run_kind="automatic")
        )
    return runs


def list_unsupervised_libraries(
    env_id: str,
    *,
    layout: ArtifactLayout | None = None,
) -> list[UnsupervisedLibrarySummary]:
    resolved_layout = layout or get_artifact_layout()
    env_root = resolved_layout.data_root / "patterns" / env_id / UNSUPERVISED_RUNS_DIRNAME
    if not env_root.is_dir():
        return []
    runs: list[UnsupervisedLibrarySummary] = []
    for run_root in sorted(
        (
            path
            for path in env_root.iterdir()
            if path.is_dir() and _is_saved_pattern_run_dir(path)
        ),
        key=lambda p: p.name,
        reverse=True,
    ):
        run_json_path = run_root / "run.json"
        run_payload = _load_json(run_json_path) if run_json_path.is_file() else {}
        if _is_automatic_run_payload(run_payload):
            continue
        runs.append(
            _load_unsupervised_summary_from_root(env_id, run_root, run_kind="unsupervised")
        )
    return runs


def load_supervised_library_detail(
    env_id: str,
    run_id: str,
    *,
    layout: ArtifactLayout | None = None,
) -> SupervisedLibraryDetailSummary:
    return load_pattern_run_detail(env_id, run_id, layout=layout)


def load_automatic_library_summary(env_id: str, run_root: Path) -> UnsupervisedLibrarySummary:
    return _load_unsupervised_summary_from_root(env_id, run_root, run_kind="automatic")


def load_unsupervised_library_summary(env_id: str, run_root: Path) -> UnsupervisedLibrarySummary:
    return _load_unsupervised_summary_from_root(env_id, run_root, run_kind="unsupervised")


def _load_unsupervised_library_detail_from_root(
    env_id: str,
    run_root: Path,
    *,
    run_kind: str,
    layout: ArtifactLayout | None = None,
) -> UnsupervisedLibraryDetailSummary:
    if not run_root.is_dir():
        raise FileNotFoundError(f"{run_kind.title()} library not found: {run_root}")
    summary = _load_unsupervised_summary_from_root(env_id, run_root, run_kind=run_kind)
    resolved_layout = layout or get_artifact_layout()
    source_library_path = None
    source_evaluation_path = None
    if summary.source_supervised_run_id:
        candidate = (
            resolved_layout.data_root
            / "patterns"
            / env_id
            / summary.source_supervised_run_id
            / PATTERN_LIBRARY_FILENAME
        )
        if candidate.is_file():
            source_library_path = candidate
        if summary.source_evaluation_run_id:
            evaluation_candidate = (
                resolved_layout.data_root
                / "patterns"
                / env_id
                / summary.source_supervised_run_id
                / "deepphy_eval"
                / summary.source_evaluation_run_id
            )
            if evaluation_candidate.is_dir():
                source_evaluation_path = evaluation_candidate
    pattern_detail = _load_pattern_run_detail_from_root(
        env_id=env_id,
        run_id=summary.run_id,
        run_root=run_root,
    )
    source_supervised_evaluation_runs = (
        [
            load_evaluation_run_summary(eval_root)
            for eval_root in sorted(source_evaluation_path.parent.glob("*"), key=lambda p: p.name, reverse=True)
            if eval_root.is_dir()
        ]
        if source_evaluation_path is not None and source_evaluation_path.parent.is_dir()
        else []
    )
    combined_library_path = run_root / PATTERN_LIBRARY_FILENAME
    unsupervised_library_path = _coerce_str(summary.raw_run_payload.get("unsupervised_library_path"))
    sampled_reasoning_traces_path_value = _coerce_str(
        summary.raw_run_payload.get("sampled_reasoning_traces_path")
        or summary.raw_run_payload.get("sampled_traces_path")
    )
    sampled_reasoning_traces_path = (
        Path(sampled_reasoning_traces_path_value)
        if isinstance(sampled_reasoning_traces_path_value, str)
        and sampled_reasoning_traces_path_value
        else (
            run_root / "sampled_reasoning_traces.json"
            if run_kind == "automatic"
            else run_root / "sampled_traces.json"
        )
    )
    label_proposal_response_path_value = _coerce_str(
        summary.raw_run_payload.get("label_proposal_response_path")
        or summary.raw_run_payload.get("final_label_selection_response_path")
    )
    label_proposal_response_path = (
        Path(label_proposal_response_path_value)
        if isinstance(label_proposal_response_path_value, str)
        and label_proposal_response_path_value
        else (
            run_root / "label_proposal_response.txt"
            if run_kind == "automatic"
            else run_root / "final_label_selection_response.txt"
        )
    )
    sampled_reasoning_trace_count = _coerce_int(
        summary.raw_run_payload.get("sampled_reasoning_trace_count")
    )
    if sampled_reasoning_trace_count is None and sampled_reasoning_traces_path.is_file():
        sampled_payload = _load_json(sampled_reasoning_traces_path)
        records = sampled_payload.get("records") if isinstance(sampled_payload, dict) else None
        if isinstance(records, list):
            sampled_reasoning_trace_count = len(records)
    label_proposal_response_text = (
        label_proposal_response_path.read_text(encoding="utf-8")
        if label_proposal_response_path.is_file()
        else None
    )
    return UnsupervisedLibraryDetailSummary(
        env_id=env_id,
        run_id=summary.run_id,
        root=summary.root,
        run_json_path=summary.run_json_path,
        run_kind=run_kind,
        source_supervised_run_id=summary.source_supervised_run_id,
        source_evaluation_run_id=summary.source_evaluation_run_id,
        source_supervised_library_path=source_library_path,
        source_supervised_evaluation_path=source_evaluation_path,
        combined_library_path=combined_library_path if combined_library_path.is_file() else None,
        unsupervised_library_path=Path(unsupervised_library_path)
        if isinstance(unsupervised_library_path, str) and unsupervised_library_path
        else None,
        proposed_labels=list(summary.proposed_labels),
        proposed_label_details=list(summary.proposed_label_details),
        sampled_reasoning_trace_count=sampled_reasoning_trace_count,
        sampled_reasoning_traces_path=(
            sampled_reasoning_traces_path
            if sampled_reasoning_traces_path.is_file()
            else None
        ),
        label_proposal_response_path=(
            label_proposal_response_path if label_proposal_response_path.is_file() else None
        ),
        label_proposal_response_text=label_proposal_response_text,
        label_telemetry=pattern_detail.label_telemetry,
        average_score=pattern_detail.average_score,
        sample_count=pattern_detail.sample_count,
        iteration_count=pattern_detail.iteration_count,
        detector_count=pattern_detail.detector_count,
        candidate_count=pattern_detail.candidate_count,
        evaluation_runs=pattern_detail.evaluation_runs,
        source_supervised_evaluation_runs=source_supervised_evaluation_runs,
        evolution_history=pattern_detail.evolution_history,
        llm_calls_path=pattern_detail.llm_calls_path,
        llm_calls=pattern_detail.llm_calls,
        created_at=summary.created_at,
        raw_run_payload=summary.raw_run_payload,
    )


def load_automatic_library_detail(
    env_id: str,
    run_id: str,
    *,
    layout: ArtifactLayout | None = None,
) -> UnsupervisedLibraryDetailSummary:
    resolved_layout = layout or get_artifact_layout()
    automatic_root = (
        resolved_layout.data_root
        / "patterns"
        / env_id
        / AUTOMATIC_RUNS_DIRNAME
        / run_id
    )
    if automatic_root.is_dir():
        return _load_unsupervised_library_detail_from_root(
            env_id,
            automatic_root,
            run_kind="automatic",
            layout=resolved_layout,
        )
    legacy_root = (
        resolved_layout.data_root
        / "patterns"
        / env_id
        / UNSUPERVISED_RUNS_DIRNAME
        / run_id
    )
    if legacy_root.is_dir():
        payload = _load_json(legacy_root / "run.json")
        if _is_automatic_run_payload(payload):
            return _load_unsupervised_library_detail_from_root(
                env_id,
                legacy_root,
                run_kind="automatic",
                layout=resolved_layout,
            )
    raise FileNotFoundError(f"Automatic library not found: {run_id}")


def load_unsupervised_library_detail(
    env_id: str,
    run_id: str,
    *,
    layout: ArtifactLayout | None = None,
) -> UnsupervisedLibraryDetailSummary:
    resolved_layout = layout or get_artifact_layout()
    run_root = (
        resolved_layout.data_root
        / "patterns"
        / env_id
        / UNSUPERVISED_RUNS_DIRNAME
        / run_id
    )
    if not run_root.is_dir():
        raise FileNotFoundError(f"Unsupervised library not found: {run_root}")
    payload = _load_json(run_root / "run.json")
    if _is_automatic_run_payload(payload):
        raise FileNotFoundError(f"Unsupervised library not found: {run_root}")
    return _load_unsupervised_library_detail_from_root(
        env_id,
        run_root,
        run_kind="unsupervised",
        layout=resolved_layout,
    )


def _coerce_proposed_label_details(values: list[Any]) -> list[ProposedLabelSummary]:
    details: list[ProposedLabelSummary] = []
    for item in values:
        if isinstance(item, dict):
            label = _coerce_str(item.get("label"))
            if not label:
                continue
            details.append(
                ProposedLabelSummary(
                    label=label,
                    description=_coerce_str(item.get("description")),
                )
            )
        elif isinstance(item, str) and item.strip():
            details.append(ProposedLabelSummary(label=item.strip(), description=None))
    return details


def _load_proposed_label_details_summary(path: Path) -> list[ProposedLabelSummary]:
    payload = _load_json(path)
    values = payload.get("labels") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        return []
    return _coerce_proposed_label_details(values)


def load_pattern_run_summary(env_id: str, run_root: Path) -> PatternRunSummary:
    run_json_path = run_root / "run.json"
    run_payload = _load_json(run_json_path) if run_json_path.is_file() else {}
    library_path = run_root / PATTERN_LIBRARY_FILENAME
    source_labels = _load_pattern_library_labels(
        library_path if library_path.is_file() else None
    )
    label_summaries = [
        load_label_summary(label_root)
        for label_root in sorted(run_root.glob("label_*"))
        if label_root.is_dir()
    ]
    evaluation_runs = (
        [
            load_evaluation_run_summary(eval_root, source_labels=source_labels)
            for eval_root in sorted(
                (run_root / "deepphy_eval").glob("*"),
                key=lambda p: p.name,
                reverse=True,
            )
            if eval_root.is_dir()
        ]
        if (run_root / "deepphy_eval").is_dir()
        else []
    )
    return PatternRunSummary(
        env_id=env_id,
        run_id=run_root.name,
        root=run_root,
        library_path=library_path if library_path.is_file() else None,
        run_json_path=run_json_path if run_json_path.is_file() else None,
        average_score=_coerce_float(run_payload.get("average_score"))
        if isinstance(run_payload, dict)
        else None,
        sample_count=_coerce_int(run_payload.get("sample_count"))
        if isinstance(run_payload, dict)
        else None,
        iteration_count=_coerce_int(run_payload.get("iteration_count"))
        if isinstance(run_payload, dict)
        else None,
        label_summaries=label_summaries,
        evaluation_runs=evaluation_runs,
        created_at=run_root.name,
        raw_run_payload=run_payload if isinstance(run_payload, dict) else {},
    )


def load_pattern_run_detail(
    env_id: str,
    run_id: str,
    *,
    layout: ArtifactLayout | None = None,
) -> PatternRunDetailSummary:
    resolved_layout = layout or get_artifact_layout()
    run_root = resolved_layout.data_root / "patterns" / env_id / run_id
    if not run_root.is_dir():
        raise FileNotFoundError(f"Pattern run not found: {run_root}")

    return _load_pattern_run_detail_from_root(env_id=env_id, run_id=run_id, run_root=run_root)


def load_pattern_run_detail_from_root(
    env_id: str,
    run_root: Path,
) -> PatternRunDetailSummary:
    if not run_root.is_dir():
        raise FileNotFoundError(f"Pattern run not found: {run_root}")
    return _load_pattern_run_detail_from_root(
        env_id=env_id,
        run_id=run_root.name,
        run_root=run_root,
    )


def list_library_evaluations_for_run(run_root: Path) -> list[LibraryEvaluationSummary]:
    root = run_root / LIBRARY_EVALUATIONS_DIRNAME
    if not root.is_dir():
        return []
    return [
        load_library_evaluation_summary(eval_root, env_id=_env_id_for_run_root(run_root), source_run_id=run_root.name)
        for eval_root in sorted(root.glob("*"), key=lambda p: p.name, reverse=True)
        if eval_root.is_dir()
    ]


def load_library_evaluation_detail(
    env_id: str,
    run_id: str,
    library_evaluation_run_id: str,
    *,
    layout: ArtifactLayout | None = None,
) -> LibraryEvaluationSummary:
    resolved_layout = layout or get_artifact_layout()
    eval_root = (
        resolved_layout.data_root
        / "patterns"
        / env_id
        / run_id
        / LIBRARY_EVALUATIONS_DIRNAME
        / library_evaluation_run_id
    )
    if not eval_root.is_dir():
        raise FileNotFoundError(f"Library evaluation not found: {eval_root}")
    return load_library_evaluation_summary(
        eval_root,
        env_id=env_id,
        source_run_id=run_id,
    )


def load_library_evaluation_summary(
    eval_root: Path,
    *,
    env_id: str,
    source_run_id: str,
) -> LibraryEvaluationSummary:
    request_path = eval_root / "request.json"
    manifest_path = eval_root / "manifest.json"
    request_payload = _load_json(request_path) if request_path.is_file() else {}
    manifest_payload = _load_json(manifest_path) if manifest_path.is_file() else {}
    available_labels = _coerce_str_list(manifest_payload.get("available_labels"))
    case_summaries: list[LibraryEvaluationCaseSummary] = []
    raw_cases = manifest_payload.get("cases")
    if isinstance(raw_cases, list):
        for item in raw_cases:
            if not isinstance(item, dict):
                continue
            case_id = _coerce_str(item.get("case_id"))
            if not case_id:
                continue
            seed_runs: list[LibraryEvaluationSeedSummary] = []
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
                    child_root = eval_root / relative_root
                    evaluation = (
                        load_evaluation_run_summary(child_root)
                        if child_root.is_dir()
                        else None
                    )
                    seed_runs.append(
                        LibraryEvaluationSeedSummary(
                            seed=seed,
                            evaluation_run_id=evaluation_run_id,
                            root=child_root,
                            status=evaluation.status if evaluation is not None else "pending",
                            evaluation=evaluation,
                        )
                    )
            case_summaries.append(
                LibraryEvaluationCaseSummary(
                    case_id=case_id,
                    label=_coerce_str(item.get("label")) or case_id,
                    kind=_coerce_str(item.get("kind")) or "baseline",
                    held_out_labels=_coerce_str_list(item.get("held_out_labels")),
                    enabled_labels=_coerce_str_list(item.get("enabled_labels")),
                    seed_runs=seed_runs,
                )
            )
    template_name = _coerce_str(request_payload.get("template_name")) or "quick-eval"
    benchmark_preset = _coerce_str(request_payload.get("benchmark_preset")) or "quick-eval"
    return LibraryEvaluationSummary(
        env_id=env_id,
        source_run_id=source_run_id,
        run_id=eval_root.name,
        root=eval_root,
        created_at=eval_root.name,
        template_name=template_name,
        benchmark_preset=benchmark_preset,
        request_path=request_path if request_path.is_file() else None,
        manifest_path=manifest_path if manifest_path.is_file() else None,
        available_labels=available_labels,
        case_summaries=case_summaries,
        raw_request_payload=request_payload if isinstance(request_payload, dict) else {},
        raw_manifest_payload=manifest_payload if isinstance(manifest_payload, dict) else {},
    )


def _load_pattern_run_detail_from_root(
    *,
    env_id: str,
    run_id: str,
    run_root: Path,
) -> PatternRunDetailSummary:
    run_json_path = run_root / "run.json"
    run_payload = _load_json(run_json_path) if run_json_path.is_file() else {}
    library_path = run_root / PATTERN_LIBRARY_FILENAME
    source_labels = _load_pattern_library_labels(
        library_path if library_path.is_file() else None
    )
    evaluation_runs = (
        [
            load_evaluation_run_summary(eval_root, source_labels=source_labels)
            for eval_root in sorted(
                (run_root / "deepphy_eval").glob("*"),
                key=lambda p: p.name,
                reverse=True,
            )
            if eval_root.is_dir()
        ]
        if (run_root / "deepphy_eval").is_dir()
        else []
    )

    label_loss_map: dict[str, list[dict[str, Any]]] = {}
    label_runs_payload = (
        run_payload.get("label_runs") if isinstance(run_payload, dict) else None
    )
    if isinstance(label_runs_payload, list):
        for item in label_runs_payload:
            if not isinstance(item, dict):
                continue
            label_name = item.get("label")
            if not isinstance(label_name, str) or not label_name:
                continue
            raw_breakdown = item.get("loss_breakdown")
            if isinstance(raw_breakdown, list):
                loss_breakdown = [
                    entry for entry in raw_breakdown if isinstance(entry, dict)
                ]
            else:
                loss_breakdown = []
            label_loss_map[label_name] = loss_breakdown

    evolution_record_paths = _resolve_evolution_record_paths(
        run_root=run_root, run_payload=run_payload
    )
    evolution_record_path = evolution_record_paths[0] if evolution_record_paths else None
    evolution_history = _load_evolution_history(evolution_record_paths)
    evolution_by_label: dict[str, list[PatternEvolutionRecord]] = {}
    for record in evolution_history:
        evolution_by_label.setdefault(record.label_name, []).append(record)

    label_telemetry: list[PatternLabelTelemetrySummary] = []
    for label_root in sorted(run_root.glob("label_*")):
        if not label_root.is_dir():
            continue
        candidates = [
            load_candidate_summary(candidate_root)
            for candidate_root in sorted(label_root.glob("candidate_*"))
            if candidate_root.is_dir()
        ]
        best_candidate = _best_candidate(candidates)
        label_telemetry.append(
            PatternLabelTelemetrySummary(
                name=label_root.name,
                root=label_root,
                candidates=candidates,
                best_candidate=best_candidate,
                loss_breakdown=label_loss_map.get(label_root.name, []),
                iteration_history=sorted(
                    evolution_by_label.get(label_root.name, []),
                    key=lambda item: (
                        item.iteration if item.iteration is not None else -1,
                        item.candidate_uid or "",
                    ),
                ),
                evolution_summary=_build_label_evolution_summary(
                    label_name=label_root.name,
                    candidates=candidates,
                    records=evolution_by_label.get(label_root.name, []),
                    best_candidate=best_candidate,
                ),
            )
        )

    candidate_name_by_label_and_uid: dict[tuple[str, str], str] = {}
    for label in label_telemetry:
        for candidate in label.candidates:
            if candidate.uid:
                candidate_name_by_label_and_uid[(label.name, candidate.uid)] = candidate.name

    evolution_history = [
        PatternEvolutionRecord(
            label_name=record.label_name,
            candidate_uid=record.candidate_uid,
            candidate_name=(
                candidate_name_by_label_and_uid.get((record.label_name, record.candidate_uid))
                if record.candidate_uid is not None
                else None
            ),
            sequence=record.sequence,
            iteration=record.iteration,
            origin=record.origin,
            score=record.score,
            train_score=record.train_score,
            val_score=record.val_score,
            parent_uid=record.parent_uid,
            secondary_parent_uid=record.secondary_parent_uid,
        )
        for record in evolution_history
    ]

    raw_score_details = (
        run_payload.get("score_details") if isinstance(run_payload, dict) else []
    )
    score_details = (
        [entry for entry in raw_score_details if isinstance(entry, dict)]
        if isinstance(raw_score_details, list)
        else []
    )
    raw_loss_breakdown = (
        run_payload.get("loss_breakdown") if isinstance(run_payload, dict) else []
    )
    loss_breakdown = (
        [entry for entry in raw_loss_breakdown if isinstance(entry, dict)]
        if isinstance(raw_loss_breakdown, list)
        else []
    )
    llm_calls_path = _resolve_llm_calls_path(run_root=run_root, run_payload=run_payload)
    llm_calls = _load_llm_calls(llm_calls_path)

    return PatternRunDetailSummary(
        env_id=env_id,
        run_id=run_id,
        root=run_root,
        library_path=library_path if library_path.is_file() else None,
        run_json_path=run_json_path if run_json_path.is_file() else None,
        evolution_record_path=evolution_record_path
        if evolution_record_path is not None and evolution_record_path.is_file()
        else None,
        average_score=_coerce_float(run_payload.get("average_score"))
        if isinstance(run_payload, dict)
        else None,
        sample_count=_coerce_int(run_payload.get("sample_count"))
        if isinstance(run_payload, dict)
        else None,
        train_sample_count=_coerce_int(run_payload.get("train_sample_count"))
        if isinstance(run_payload, dict)
        else None,
        validation_sample_count=_coerce_int(run_payload.get("validation_sample_count"))
        if isinstance(run_payload, dict)
        else None,
        iteration_count=_coerce_int(run_payload.get("iteration_count"))
        if isinstance(run_payload, dict)
        else None,
        detector_count=_coerce_int(run_payload.get("detector_count"))
        if isinstance(run_payload, dict)
        else None,
        loss_breakdown=loss_breakdown,
        score_details=score_details,
        label_telemetry=label_telemetry,
        evaluation_runs=evaluation_runs,
        evolution_history=evolution_history,
        llm_calls_path=llm_calls_path if llm_calls_path and llm_calls_path.is_file() else None,
        llm_calls=llm_calls,
        raw_run_payload=run_payload if isinstance(run_payload, dict) else {},
    )


def load_label_summary(label_root: Path) -> LabelSummary:
    candidates = [
        load_candidate_summary(candidate_root)
        for candidate_root in sorted(label_root.glob("candidate_*"))
        if candidate_root.is_dir()
    ]
    return LabelSummary(name=label_root.name, root=label_root, candidates=candidates)


def load_candidate_summary(candidate_root: Path) -> CandidateSummary:
    metadata_path = candidate_root / "candidate.json"
    payload = _load_json(metadata_path) if metadata_path.is_file() else {}
    code_path = next(
        (path for path in sorted(candidate_root.glob("*.py")) if path.is_file()),
        None,
    )
    return CandidateSummary(
        name=candidate_root.name,
        root=candidate_root,
        code_path=code_path,
        metadata_path=metadata_path if metadata_path.is_file() else None,
        average_score=_coerce_float(payload.get("average_score"))
        if isinstance(payload, dict)
        else None,
        train_score=_coerce_float(payload.get("train_score"))
        if isinstance(payload, dict)
        else None,
        val_score=_coerce_float(payload.get("val_score"))
        if isinstance(payload, dict)
        else None,
        uid=str(payload.get("uid"))
        if isinstance(payload, dict) and payload.get("uid") is not None
        else None,
        origin=str(payload.get("origin"))
        if isinstance(payload, dict) and payload.get("origin") is not None
        else None,
        parent_uid=str(payload.get("parent_uid"))
        if isinstance(payload, dict) and payload.get("parent_uid") is not None
        else None,
        secondary_parent_uid=str(payload.get("secondary_parent_uid"))
        if isinstance(payload, dict) and payload.get("secondary_parent_uid") is not None
        else None,
        iteration=_coerce_int(payload.get("iteration"))
        if isinstance(payload, dict)
        else None,
        is_best_candidate=bool(payload.get("is_best_candidate"))
        if isinstance(payload, dict)
        else False,
        loss_breakdown=[
            item for item in payload.get("loss_breakdown", []) if isinstance(item, dict)
        ]
        if isinstance(payload, dict) and isinstance(payload.get("loss_breakdown"), list)
        else [],
        validation_loss_breakdown=[
            item
            for item in payload.get("validation_loss_breakdown", [])
            if isinstance(item, dict)
        ]
        if isinstance(payload, dict)
        and isinstance(payload.get("validation_loss_breakdown"), list)
        else [],
        score_details=[
            item for item in payload.get("score_details", []) if isinstance(item, dict)
        ]
        if isinstance(payload, dict) and isinstance(payload.get("score_details"), list)
        else [],
        errors=[str(item) for item in payload.get("errors", [])]
        if isinstance(payload, dict) and isinstance(payload.get("errors"), list)
        else [],
    )


def load_evaluation_run_summary(
    eval_root: Path,
    *,
    source_labels: list[str] | None = None,
) -> EvaluationRunSummary:
    summary_path = eval_root / "summary.json"
    results_path = eval_root / "results.json"
    request_path = eval_root / "request.json"
    summary_payload = _load_json(summary_path) if summary_path.is_file() else {}
    results_payload = _load_json(results_path) if results_path.is_file() else {}
    request_payload = _load_json(request_path) if request_path.is_file() else {}
    results = (
        results_payload.get("results")
        if isinstance(results_payload, dict)
        and isinstance(results_payload.get("results"), list)
        else []
    )
    status = "completed" if (summary_path.is_file() or results_path.is_file()) else "pending"
    task_count = 0
    success_count = 0
    if results:
        task_count = len(results)
        success_count = sum(
            1
            for item in results
            if isinstance(item, dict) and bool(item.get("success"))
        )
    if task_count == 0 and isinstance(summary_payload, dict):
        envs = summary_payload.get("env_summaries") or summary_payload.get("environments")
        if isinstance(envs, list):
            for item in envs:
                if isinstance(item, dict):
                    task_count += _coerce_int(item.get("task_count")) or 0
                    success_count += _coerce_int(item.get("success_count")) or 0
    first_result = next((item for item in results if isinstance(item, dict)), {})
    metric_summaries = _build_evaluation_metric_summaries(results)
    task_summaries = _build_evaluation_task_summaries(results, eval_root=eval_root)
    terminal_status_counts = _count_terminal_statuses(results)
    if not results and not summary_path.is_file():
        task_summaries = _build_live_evaluation_task_summaries(eval_root)
        if task_summaries:
            completed_task_summaries = [
                item
                for item in task_summaries
                if _live_evaluation_task_summary_is_completed(item)
            ]
            status = "running"
            task_count = len(completed_task_summaries)
            success_count = sum(
                1 for item in completed_task_summaries if item.success is True
            )
            metric_summaries = _build_metric_summaries_from_task_summaries(
                completed_task_summaries
            )
            terminal_status_counts = _count_terminal_statuses_from_task_summaries(
                completed_task_summaries
            )
    label_selection = _resolve_evaluation_label_selection(
        eval_root=eval_root,
        summary_payload=summary_payload,
        results=results,
        source_labels=source_labels or [],
    )
    env_summaries_payload = (
        summary_payload.get("env_summaries") or summary_payload.get("environments")
        if isinstance(summary_payload, dict)
        else None
    )
    first_env_summary = (
        env_summaries_payload[0]
        if isinstance(env_summaries_payload, list)
        and env_summaries_payload
        and isinstance(env_summaries_payload[0], dict)
        else {}
    )
    return EvaluationRunSummary(
        run_id=eval_root.name,
        root=eval_root,
        summary_path=summary_path if summary_path.is_file() else None,
        results_path=results_path if results_path.is_file() else None,
        status=status,
        task_count=task_count,
        success_count=success_count,
        created_at=eval_root.name,
        benchmark_preset=_coerce_str(
            request_payload.get("benchmark_preset")
            if isinstance(request_payload, dict)
            else None
        ),
        total_duration_seconds=(
            _coerce_float(summary_payload.get("total_duration_seconds"))
            if isinstance(summary_payload, dict)
            else None
        )
        or sum(
            float(item.total_duration_seconds)
            for item in task_summaries
            if item.total_duration_seconds is not None
        )
        or None,
        average_task_duration_seconds=_coerce_float(
            first_env_summary.get("average_task_duration_seconds")
        )
        or (
            sum(
                float(item.total_duration_seconds)
                for item in task_summaries
                if item.total_duration_seconds is not None
            )
            / len([item for item in task_summaries if item.total_duration_seconds is not None])
            if [item for item in task_summaries if item.total_duration_seconds is not None]
            else None
        ),
        model_name=_coerce_str(first_result.get("model_name"))
        if isinstance(first_result, dict)
        else None,
        prompt_format=_coerce_str(first_result.get("prompt_format"))
        if isinstance(first_result, dict)
        else None,
        description_mode=_coerce_str(first_result.get("description_mode"))
        if isinstance(first_result, dict)
        else None,
        metric_summaries=metric_summaries,
        task_summaries=task_summaries,
        terminal_status_counts=terminal_status_counts,
        label_selection=label_selection,
    )


def load_evaluation_task_detail(
    eval_root: Path,
    *,
    task_id: str | None = None,
    task_token: str | None = None,
) -> EvaluationTaskDetailSummary | None:
    if not task_id and not task_token:
        return None
    result_payload = _load_evaluation_task_result_payload(
        eval_root,
        task_id=task_id,
        task_token=task_token,
    )
    if isinstance(result_payload, dict):
        return _build_evaluation_task_detail_from_payload(
            result_payload,
            eval_root=eval_root,
        )
    live_summary = _build_live_evaluation_task_detail(
        eval_root,
        task_id=task_id,
        task_token=task_token,
    )
    return live_summary


def load_candidate_code(
    env_id: str,
    run_id: str,
    label_name: str,
    candidate_name: str,
    *,
    layout: ArtifactLayout | None = None,
) -> tuple[CandidateSummary, str]:
    resolved_layout = layout or get_artifact_layout()
    candidate_root = (
        resolved_layout.data_root
        / "patterns"
        / env_id
        / run_id
        / label_name
        / candidate_name
    )
    if not candidate_root.is_dir():
        candidate_root = (
            resolved_layout.data_root
            / "patterns"
            / env_id
            / UNSUPERVISED_RUNS_DIRNAME
            / run_id
            / label_name
            / candidate_name
        )
    if not candidate_root.is_dir():
        raise FileNotFoundError(f"Pattern candidate not found: {candidate_root}")
    candidate = load_candidate_summary(candidate_root)
    if candidate.code_path is None:
        raise FileNotFoundError(f"Pattern candidate code not found: {candidate_root}")
    return candidate, candidate.code_path.read_text(encoding="utf-8")


def _dataset_index_entry_summary(
    *, sample_root: Path, entry_payload: Any
) -> DatasetIndexedSampleSummary:
    trace_path = sample_root / TRACE_FILENAME
    quality_path = sample_root / QUALITY_FILENAME
    evidence_path = sample_root / EVIDENCE_FILENAME
    scene_image_paths = _discover_media_paths(
        sample_root,
        canonical_name=SCENE_IMAGE_FILENAME,
        pattern="initial_scene*.png",
    )
    simulation_paths = _discover_media_paths(
        sample_root,
        canonical_name=SIMULATION_VIDEO_FILENAME,
        pattern="simulation*.mp4",
    )
    success = _sample_success_from_trace(trace_path) if trace_path.is_file() else None
    return DatasetIndexedSampleSummary(
        split_name=entry_payload.split_name,
        sample_name=sample_root.name,
        sample_root=sample_root,
        env_id=entry_payload.env_id,
        sample_id=entry_payload.sample_id,
        absolute_sample_index=entry_payload.absolute_sample_index,
        status=entry_payload.status,
        success=success,
        trace_id=entry_payload.trace_id,
        task_id=entry_payload.task_id,
        trace_path=trace_path if trace_path.is_file() else None,
        quality_path=quality_path if quality_path.is_file() else None,
        evidence_path=evidence_path if evidence_path.is_file() else None,
        scene_image_path=scene_image_paths[0] if scene_image_paths else None,
        simulation_path=simulation_paths[0] if simulation_paths else None,
    )


def _discover_media_paths(
    sample_root: Path, *, canonical_name: str, pattern: str
) -> list[Path]:
    seen: set[Path] = set()
    discovered: list[Path] = []
    canonical_path = sample_root / canonical_name
    if canonical_path.is_file():
        discovered.append(canonical_path)
        seen.add(canonical_path)
    for path in sorted(sample_root.glob(pattern)):
        if not path.is_file() or path in seen:
            continue
        discovered.append(path)
        seen.add(path)
    return discovered


def _resolve_dataset_sample_root(
    *, dataset_root: Path, split_name: str, sample_name: str
) -> Path:
    direct = dataset_root / split_name / sample_name
    if direct.is_dir():
        return direct
    nested_matches = (
        sorted(
            path
            for path in (dataset_root / split_name).glob(f"**/{sample_name}")
            if path.is_dir()
        )
        if (dataset_root / split_name).is_dir()
        else []
    )
    if nested_matches:
        return nested_matches[0]
    raise FileNotFoundError(
        f"Dataset sample not found: {dataset_root}/{split_name}/{sample_name}"
    )


def _resolve_evolution_record_paths(
    *, run_root: Path, run_payload: dict[str, Any] | Any
) -> list[Path]:
    paths: list[Path] = []
    if isinstance(run_payload, dict):
        explicit = run_payload.get("evolution_record_path")
        if isinstance(explicit, str) and explicit:
            explicit_path = Path(explicit)
            candidate = (
                explicit_path
                if explicit_path.is_absolute()
                else (run_root / explicit_path)
            )
            if candidate.is_file():
                paths.append(candidate)
    for fallback in (
        run_root / "evolution_record.jsonl",
        run_root / "evolution_live.jsonl",
    ):
        if fallback.is_file() and fallback not in paths:
            paths.append(fallback)
    return paths


def _resolve_llm_calls_path(
    *,
    run_root: Path,
    run_payload: dict[str, Any] | Any,
) -> Path | None:
    if isinstance(run_payload, dict):
        explicit = run_payload.get("llm_calls_path")
        if isinstance(explicit, str) and explicit:
            explicit_path = Path(explicit)
            candidate = (
                explicit_path
                if explicit_path.is_absolute()
                else (run_root / explicit_path)
            )
            if candidate.is_file():
                return candidate
    fallback = run_root / "llm_calls.jsonl"
    return fallback if fallback.is_file() else None


def _load_llm_calls(path: Path | None) -> list[LLMCallSummary]:
    if path is None or not path.is_file():
        return []
    records: list[LLMCallSummary] = []
    with path.open("r", encoding="utf-8") as handle:
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
            request_payload = payload.get("request")
            inference_payload = payload.get("inference")
            response_payload = payload.get("response")
            records.append(
                LLMCallSummary(
                    sequence=_coerce_int(payload.get("sequence")) or 0,
                    created_at=_coerce_str(payload.get("created_at")),
                    label=_coerce_str(payload.get("label")),
                    label_index=_coerce_int(payload.get("label_index")),
                    worker=_coerce_int(payload.get("worker")),
                    model=_coerce_str(
                        inference_payload.get("model")
                        if isinstance(inference_payload, dict)
                        else None
                    ),
                    temperature=_coerce_float(
                        inference_payload.get("temperature")
                        if isinstance(inference_payload, dict)
                        else None
                    ),
                    max_tokens=_coerce_int(
                        inference_payload.get("max_tokens")
                        if isinstance(inference_payload, dict)
                        else None
                    ),
                    timeout_seconds=_coerce_int(
                        inference_payload.get("timeout_seconds")
                        if isinstance(inference_payload, dict)
                        else None
                    ),
                    duration_seconds=_coerce_float(payload.get("duration_seconds")),
                    request_messages=[
                        item
                        for item in request_payload.get("messages", [])
                        if isinstance(item, dict)
                    ]
                    if isinstance(request_payload, dict)
                    and isinstance(request_payload.get("messages"), list)
                    else [],
                    request_body=dict(request_payload.get("body") or {})
                    if isinstance(request_payload, dict)
                    else {},
                    request_headers=dict(request_payload.get("headers") or {})
                    if isinstance(request_payload, dict)
                    else {},
                    response_text=_coerce_str(
                        response_payload.get("text")
                        if isinstance(response_payload, dict)
                        else None
                    ),
                    finish_reason=_coerce_str(
                        response_payload.get("finish_reason")
                        if isinstance(response_payload, dict)
                        else None
                    ),
                    usage=dict(response_payload.get("usage") or {})
                    if isinstance(response_payload, dict)
                    else {},
                    router=dict(response_payload.get("router") or {})
                    if isinstance(response_payload, dict)
                    else {},
                    raw_response=dict(response_payload.get("raw") or {})
                    if isinstance(response_payload, dict)
                    else {},
                )
            )
    return sorted(
        records,
        key=lambda item: (
            item.created_at or "",
            item.sequence,
        ),
        reverse=True,
    )


def _load_evolution_history(paths: Path | list[Path] | None) -> list[PatternEvolutionRecord]:
    resolved_paths = (
        [paths]
        if isinstance(paths, Path)
        else [path for path in (paths or []) if isinstance(path, Path)]
    )
    if not resolved_paths:
        return []
    records: list[PatternEvolutionRecord] = []
    seen_keys: set[tuple[str, str | None, int | None]] = set()
    sequence = 0
    for path in resolved_paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
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
                label_name = (
                    str(payload.get("label"))
                    if payload.get("label") is not None
                    else ""
                )
                candidate_uid = _coerce_str(payload.get("candidate_uid"))
                iteration = _coerce_int(payload.get("iteration"))
                dedupe_key = (label_name, candidate_uid, iteration)
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                sequence += 1
                records.append(
                    PatternEvolutionRecord(
                        label_name=label_name,
                        candidate_uid=candidate_uid,
                        candidate_name=None,
                        sequence=sequence,
                        iteration=iteration,
                        origin=_coerce_str(payload.get("origin")),
                        score=_coerce_float(payload.get("score")),
                        train_score=_coerce_float(payload.get("train_score")),
                        val_score=_coerce_float(payload.get("val_score")),
                        parent_uid=_coerce_str(payload.get("parent_uid")),
                        secondary_parent_uid=_coerce_str(
                            payload.get("secondary_parent_uid")
                        ),
                    )
                )
    return records


def _best_candidate(candidates: list[CandidateSummary]) -> CandidateSummary | None:
    explicit = next(
        (candidate for candidate in candidates if candidate.is_best_candidate), None
    )
    if explicit is not None:
        return explicit
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda item: (
            item.average_score if item.average_score is not None else float("-inf"),
            item.name,
        ),
        reverse=True,
    )
    return ranked[0]


def _build_label_evolution_summary(
    *,
    label_name: str,
    candidates: list[CandidateSummary],
    records: list[PatternEvolutionRecord],
    best_candidate: CandidateSummary | None,
) -> PatternEvolutionLabelSummary | None:
    if not candidates and not records:
        return None

    candidate_by_uid = {
        candidate.uid: candidate for candidate in candidates if candidate.uid is not None
    }

    candidate_rows: list[PatternEvolutionCandidateSummary] = []
    for record in records:
        candidate = candidate_by_uid.get(record.candidate_uid)
        candidate_rows.append(
            PatternEvolutionCandidateSummary(
                candidate_name=candidate.name if candidate is not None else None,
                candidate_uid=record.candidate_uid,
                iteration=record.iteration,
                origin=record.origin,
                score=record.score,
                train_score=record.train_score,
                val_score=record.val_score,
                parent_uid=record.parent_uid,
                secondary_parent_uid=record.secondary_parent_uid,
                is_best_candidate=(
                    best_candidate is not None
                    and record.candidate_uid is not None
                    and record.candidate_uid == best_candidate.uid
                ),
            )
        )

    if not candidate_rows:
        for candidate in candidates:
            candidate_rows.append(
                PatternEvolutionCandidateSummary(
                    candidate_name=candidate.name,
                    candidate_uid=candidate.uid,
                    iteration=candidate.iteration,
                    origin=candidate.origin,
                    score=candidate.average_score,
                    train_score=candidate.train_score,
                    val_score=candidate.val_score,
                    parent_uid=None,
                    secondary_parent_uid=None,
                    is_best_candidate=candidate.is_best_candidate,
                )
            )

    rows_by_iteration: dict[int, list[PatternEvolutionCandidateSummary]] = {}
    for row in candidate_rows:
        iteration = row.iteration if row.iteration is not None else 0
        rows_by_iteration.setdefault(iteration, []).append(row)

    iteration_summaries: list[PatternEvolutionIterationSummary] = []
    for iteration, rows in sorted(rows_by_iteration.items()):
        ranked = sorted(
            rows,
            key=lambda item: (
                item.score if item.score is not None else float("-inf"),
                item.candidate_name or "",
                item.candidate_uid or "",
            ),
            reverse=True,
        )
        best_row = ranked[0] if ranked else None
        origin_counts: dict[str, int] = {}
        for row in rows:
            origin_key = row.origin or "unknown"
            origin_counts[origin_key] = origin_counts.get(origin_key, 0) + 1
        iteration_summaries.append(
            PatternEvolutionIterationSummary(
                iteration=iteration,
                candidate_count=len(rows),
                best_score=best_row.score if best_row is not None else None,
                best_train_score=best_row.train_score if best_row is not None else None,
                best_val_score=best_row.val_score if best_row is not None else None,
                best_candidate_uid=best_row.candidate_uid if best_row is not None else None,
                best_candidate_name=best_row.candidate_name if best_row is not None else None,
                origin_counts=origin_counts,
                top_candidates=ranked[:4],
            )
        )

    recent_lineage = sorted(
        candidate_rows,
        key=lambda item: (
            item.iteration if item.iteration is not None else -1,
            item.score if item.score is not None else float("-inf"),
            item.candidate_name or "",
        ),
        reverse=True,
    )[:12]

    best_row = next((row for row in candidate_rows if row.is_best_candidate), None)
    if best_row is None and best_candidate is not None:
        best_row = PatternEvolutionCandidateSummary(
            candidate_name=best_candidate.name,
            candidate_uid=best_candidate.uid,
            iteration=best_candidate.iteration,
            origin=best_candidate.origin,
            score=best_candidate.average_score,
            train_score=best_candidate.train_score,
            val_score=best_candidate.val_score,
            parent_uid=None,
            secondary_parent_uid=None,
            is_best_candidate=True,
        )
    if best_row is None and candidate_rows:
        best_row = max(
            candidate_rows,
            key=lambda item: item.score if item.score is not None else float("-inf"),
        )

    return PatternEvolutionLabelSummary(
        label_name=label_name,
        best_score=best_row.score if best_row is not None else None,
        best_train_score=best_row.train_score if best_row is not None else None,
        best_val_score=best_row.val_score if best_row is not None else None,
        best_candidate_uid=best_row.candidate_uid if best_row is not None else None,
        best_candidate_name=best_row.candidate_name if best_row is not None else None,
        explored_iterations=len(iteration_summaries),
        iteration_summaries=iteration_summaries,
        recent_lineage=recent_lineage,
    )


def _count_completed_samples(split_root: Path) -> int:
    if not split_root.is_dir():
        return 0
    count = 0
    for sample_root in split_root.iterdir():
        if not sample_root.is_dir():
            continue
        if (sample_root / TRACE_FILENAME).is_file() and (
            sample_root / QUALITY_FILENAME
        ).is_file():
            count += 1
    return count


def _parse_target_labels(raw_text: str) -> list[str]:
    labels: list[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        labels.append(line)
    return labels


def _manifest_splits_map(raw_splits: Any) -> dict[str, dict[str, Any]]:
    if isinstance(raw_splits, dict):
        return {
            str(name): payload
            for name, payload in raw_splits.items()
            if isinstance(payload, dict)
        }
    if isinstance(raw_splits, list):
        mapped: dict[str, dict[str, Any]] = {}
        for item in raw_splits:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if isinstance(name, str):
                mapped[name] = item
        return mapped
    return {}


def _load_pattern_library_labels(path: Path | None) -> list[str]:
    if path is None or not path.is_file():
        return []
    try:
        library = load_pattern_library(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    labels = {
        detector.label.strip()
        for detector in library.detectors
        if isinstance(detector.label, str) and detector.label.strip()
    }
    return sorted(labels)


def _build_evaluation_metric_summaries(
    results: list[Any],
) -> list[EvaluationMetricSummary]:
    values_by_name: dict[str, list[float]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        native_metrics = item.get("native_metrics")
        if not isinstance(native_metrics, dict):
            continue
        for name, value in native_metrics.items():
            numeric = _coerce_float(value)
            if numeric is None:
                continue
            values_by_name.setdefault(str(name), []).append(numeric)
    summaries: list[EvaluationMetricSummary] = []
    for name in sorted(values_by_name):
        values = values_by_name[name]
        if not values:
            continue
        summaries.append(
            EvaluationMetricSummary(
                name=name,
                mean=sum(values) / len(values),
                minimum=min(values),
                maximum=max(values),
                count=len(values),
            )
        )
    return summaries


def _build_evaluation_task_summaries(
    results: list[Any],
    *,
    eval_root: Path | None = None,
) -> list[EvaluationTaskSummary]:
    summaries: list[EvaluationTaskSummary] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        records = item.get("records")
        record_count = len(records) if isinstance(records, list) else 0
        native_metrics = item.get("native_metrics")
        task_id = _coerce_str(item.get("task_id"))
        media = _evaluation_task_media(eval_root, task_id) if eval_root is not None else {}
        summaries.append(
            EvaluationTaskSummary(
                task_id=task_id,
                success=item.get("success")
                if isinstance(item.get("success"), bool)
                else None,
                terminal_status=_coerce_str(item.get("terminal_status")),
                record_count=record_count,
                total_duration_seconds=_coerce_float(item.get("total_duration_seconds")),
                completed_at=_coerce_str(item.get("completed_at")),
                native_metrics=native_metrics if isinstance(native_metrics, dict) else {},
                task_token=media.get("task_token"),
                initial_scene_path=media.get("initial_scene_path"),
                prompt_images=media.get("prompt_images", []),
                simulation_path=media.get("simulation_path"),
            )
        )
    return summaries


def _build_evaluation_task_detail_from_payload(
    payload: dict[str, Any],
    *,
    eval_root: Path,
) -> EvaluationTaskDetailSummary:
    task_id = _coerce_str(payload.get("task_id"))
    media = _evaluation_task_media(eval_root, task_id)
    records_payload = payload.get("records")
    records: list[EvaluationTaskRecordSummary] = []
    if isinstance(records_payload, list):
        for item in records_payload:
            if not isinstance(item, dict):
                continue
            prompt = item.get("prompt")
            prompt_payload = prompt if isinstance(prompt, dict) else {}
            parsed_action = item.get("parsed_action")
            native_metrics = item.get("native_metrics")
            trace_annotation = _parse_trace_annotation_text(
                _coerce_str(item.get("trace_description"))
            )
            records.append(
                EvaluationTaskRecordSummary(
                    index=_coerce_int(item.get("index")) or 0,
                    record_kind=_coerce_str(item.get("record_kind")),
                    prompt_format=_coerce_str(prompt_payload.get("prompt_format")),
                    description_mode=_coerce_str(prompt_payload.get("description_mode")),
                    system_prompt=_coerce_str(prompt_payload.get("system_prompt")),
                    user_prompt=_coerce_str(prompt_payload.get("user_prompt")),
                    trace_description=_coerce_str(item.get("trace_description")),
                    raw_model_output=_coerce_str(item.get("raw_model_output")),
                    duration_seconds=_coerce_float(item.get("duration_seconds")),
                    parsed_action=parsed_action if isinstance(parsed_action, dict) else {},
                    success=item.get("success") if isinstance(item.get("success"), bool) else None,
                    terminal_status=_coerce_str(item.get("terminal_status")),
                    native_metrics=native_metrics if isinstance(native_metrics, dict) else {},
                    trace_annotation_overview=trace_annotation["overview"],
                    trace_annotation_notes=trace_annotation["notes"],
                    trace_annotation_rows=trace_annotation["rows"],
                )
            )
    native_metrics = payload.get("native_metrics")
    return EvaluationTaskDetailSummary(
        task_id=task_id,
        task_token=media.get("task_token"),
        model_name=_coerce_str(payload.get("model_name")),
        prompt_format=_coerce_str(payload.get("prompt_format")),
        description_mode=_coerce_str(payload.get("description_mode")),
        success=payload.get("success") if isinstance(payload.get("success"), bool) else None,
        terminal_status=_coerce_str(payload.get("terminal_status")),
        record_count=len(records),
        total_duration_seconds=_coerce_float(payload.get("total_duration_seconds")),
        native_metrics=native_metrics if isinstance(native_metrics, dict) else {},
        initial_scene_path=media.get("initial_scene_path"),
        prompt_images=media.get("prompt_images", []),
        simulation_path=media.get("simulation_path"),
        records=records,
    )


def _parse_trace_annotation_text(
    raw_text: str | None,
) -> dict[str, str | list[str] | list[EvaluationTraceAnnotationRow] | None]:
    text = (raw_text or "").strip()
    if not text:
        return {"overview": None, "notes": [], "rows": []}
    notes: list[str] = []
    pattern_text = text
    if "\n\nLearned pattern annotations:\n" in text:
        overview, pattern_text = text.split("\n\nLearned pattern annotations:\n", 1)
        overview = overview.strip()
    elif "\nLearned pattern annotations:\n" in text:
        overview, pattern_text = text.split("\nLearned pattern annotations:\n", 1)
        overview = overview.strip()
    else:
        overview = None
    rows: list[EvaluationTraceAnnotationRow] = []
    normalized_lines = [line.strip() for line in pattern_text.splitlines() if line.strip()]
    if normalized_lines == ["No learned patterns were activated."]:
        notes.append("No learned patterns were activated.")
        return {"overview": overview or text, "notes": notes, "rows": rows}
    unmatched_lines: list[str] = []
    for line in normalized_lines:
        match = _PATTERN_ANNOTATION_LINE_RE.match(line)
        if match is None:
            unmatched_lines.append(line)
            continue
        span_kind, start_step, end_step, label, parameter_text = match.groups()
        if span_kind == "step" or end_step is None or start_step == end_step:
            step_text = f"step {start_step}"
        else:
            step_text = f"steps {start_step}-{end_step}"
        rows.append(
            EvaluationTraceAnnotationRow(
                step_text=step_text,
                label=label.strip(),
                parameter_text=parameter_text.strip() if parameter_text else None,
            )
        )
    if unmatched_lines:
        notes.extend(unmatched_lines)
    if overview is None and not rows:
        overview = text
    return {"overview": overview, "notes": notes, "rows": rows}


def _build_live_evaluation_task_summaries(eval_root: Path) -> list[EvaluationTaskSummary]:
    tasks_root = eval_root / "tasks"
    if not tasks_root.is_dir():
        return []
    def _timestamp_from_path(path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        except OSError:
            return None

    summaries: list[EvaluationTaskSummary] = []
    for env_root in sorted(path for path in tasks_root.iterdir() if path.is_dir()):
        for task_root in sorted(path for path in env_root.iterdir() if path.is_dir()):
            attempt_roots = sorted(
                (
                    path
                    for path in task_root.iterdir()
                    if path.is_dir() and path.name.startswith("attempt_")
                ),
                key=lambda path: path.name,
            )
            latest_trace_path = None
            if attempt_roots:
                latest_trace = attempt_roots[-1] / TRACE_FILENAME
                if latest_trace.is_file():
                    latest_trace_path = latest_trace
            task_id = _task_id_from_task_root(task_root, latest_trace_path)
            success = (
                _sample_success_from_trace(latest_trace_path)
                if latest_trace_path is not None
                else None
            )
            terminal_status = (
                _terminal_status_from_trace(latest_trace_path)
                if latest_trace_path is not None
                else None
            )
            native_metrics = (
                _native_metrics_from_trace(latest_trace_path)
                if latest_trace_path is not None
                else {}
            )
            media = _evaluation_task_media(eval_root, task_id, task_root=task_root)
            summaries.append(
                EvaluationTaskSummary(
                    task_id=task_id,
                    success=success,
                    terminal_status=terminal_status,
                    record_count=len(attempt_roots),
                    total_duration_seconds=_coerce_float(
                        (_load_evaluation_task_result_payload(eval_root, task_id=task_id) or {}).get(
                            "total_duration_seconds"
                        )
                    ),
                    completed_at=_timestamp_from_path(latest_trace_path)
                    or _timestamp_from_path(task_root),
                    native_metrics=native_metrics,
                    task_token=media.get("task_token"),
                    initial_scene_path=media.get("initial_scene_path"),
                    prompt_images=media.get("prompt_images", []),
                    simulation_path=media.get("simulation_path"),
                )
            )
    return summaries


def _live_evaluation_task_summary_is_completed(
    task_summary: EvaluationTaskSummary,
) -> bool:
    return bool(task_summary.terminal_status) or (
        task_summary.total_duration_seconds is not None
    )


def _build_live_evaluation_task_detail(
    eval_root: Path,
    *,
    task_id: str | None = None,
    task_token: str | None = None,
) -> EvaluationTaskDetailSummary | None:
    resolved_task_id = task_id
    resolved_task_token = task_token or _evaluation_task_token(task_id)
    task_root = _resolve_evaluation_task_root(
        eval_root,
        task_id=resolved_task_id,
        task_token=resolved_task_token,
    )
    if task_root is None:
        return None
    attempt_roots = sorted(
        (
            path
            for path in task_root.iterdir()
            if path.is_dir() and (path.name.startswith("attempt_") or path.name.startswith("step_"))
        ),
        key=lambda path: path.name,
    )
    latest_trace_path = None
    if attempt_roots:
        candidate = attempt_roots[-1] / TRACE_FILENAME
        if candidate.is_file():
            latest_trace_path = candidate
    if resolved_task_id is None:
        resolved_task_id = _task_id_from_task_root(task_root, latest_trace_path)
    media = _evaluation_task_media(eval_root, resolved_task_id, task_root=task_root)
    return EvaluationTaskDetailSummary(
        task_id=resolved_task_id,
        task_token=media.get("task_token"),
        model_name=None,
        prompt_format=None,
        description_mode=None,
        success=_sample_success_from_trace(latest_trace_path) if latest_trace_path is not None else None,
        terminal_status=_terminal_status_from_trace(latest_trace_path) if latest_trace_path is not None else None,
        record_count=len(attempt_roots),
        total_duration_seconds=_coerce_float(
            (_load_evaluation_task_result_payload(
                eval_root,
                task_id=resolved_task_id,
                task_token=resolved_task_token,
            ) or {}).get("total_duration_seconds")
        ),
        native_metrics=_native_metrics_from_trace(latest_trace_path) if latest_trace_path is not None else {},
        initial_scene_path=media.get("initial_scene_path"),
        prompt_images=media.get("prompt_images", []),
        simulation_path=media.get("simulation_path"),
        records=[],
    )


def _load_evaluation_task_result_payload(
    eval_root: Path,
    *,
    task_id: str | None = None,
    task_token: str | None = None,
) -> dict[str, Any] | None:
    results_path = eval_root / "results.json"
    if results_path.is_file():
        payload = _load_json(results_path)
        results = (
            payload.get("results")
            if isinstance(payload, dict) and isinstance(payload.get("results"), list)
            else []
        )
        for item in results:
            if not isinstance(item, dict):
                continue
            candidate_task_id = _coerce_str(item.get("task_id"))
            if task_id and candidate_task_id == task_id:
                return item
            if task_token and _evaluation_task_token(candidate_task_id) == task_token:
                return item
    task_root = _resolve_evaluation_task_root(eval_root, task_id=task_id, task_token=task_token)
    if task_root is None:
        return None
    task_result_path = task_root / "task_result.json"
    if not task_result_path.is_file():
        return None
    payload = _load_json(task_result_path)
    return payload if isinstance(payload, dict) else None


def _resolve_evaluation_task_root(
    eval_root: Path,
    *,
    task_id: str | None = None,
    task_token: str | None = None,
) -> Path | None:
    resolved_task_token = task_token or _evaluation_task_token(task_id)
    if resolved_task_token is None:
        return None
    tasks_root = eval_root / "tasks"
    if not tasks_root.is_dir():
        return None
    for env_root in sorted(path for path in tasks_root.iterdir() if path.is_dir()):
        candidate = env_root / resolved_task_token
        if candidate.is_dir():
            return candidate
    return None


def _build_metric_summaries_from_task_summaries(
    task_summaries: list[EvaluationTaskSummary],
) -> list[EvaluationMetricSummary]:
    values_by_name: dict[str, list[float]] = {}
    for item in task_summaries:
        for name, value in item.native_metrics.items():
            numeric = _coerce_float(value)
            if numeric is None:
                continue
            values_by_name.setdefault(str(name), []).append(numeric)
    summaries: list[EvaluationMetricSummary] = []
    for name in sorted(values_by_name):
        values = values_by_name[name]
        if not values:
            continue
        summaries.append(
            EvaluationMetricSummary(
                name=name,
                mean=sum(values) / len(values),
                minimum=min(values),
                maximum=max(values),
                count=len(values),
            )
        )
    return summaries


def _count_terminal_statuses_from_task_summaries(
    task_summaries: list[EvaluationTaskSummary],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in task_summaries:
        if not item.terminal_status:
            continue
        counts[item.terminal_status] = counts.get(item.terminal_status, 0) + 1
    return counts


def _count_terminal_statuses(results: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        terminal_status = _coerce_str(item.get("terminal_status"))
        if terminal_status is None:
            continue
        counts[terminal_status] = counts.get(terminal_status, 0) + 1
    return counts


def _resolve_evaluation_label_selection(
    *,
    eval_root: Path,
    summary_payload: Any,
    results: list[Any],
    source_labels: list[str],
) -> EvaluationLabelSelectionSummary:
    selection_path = eval_root / "label_selection.json"
    available_labels = list(source_labels)
    enabled_labels: list[str] = []
    disabled_labels: list[str] = []
    source = "source-run"

    if selection_path.is_file():
        payload = _load_json(selection_path)
        if isinstance(payload, dict):
            available_labels = _coerce_str_list(payload.get("available_labels")) or available_labels
            enabled_labels = _coerce_str_list(payload.get("enabled_labels"))
            disabled_labels = _coerce_str_list(payload.get("disabled_labels"))
            source = _coerce_str(payload.get("source")) or "saved-selection"
    else:
        pattern_library_path = None
        first_result = next((item for item in results if isinstance(item, dict)), None)
        if isinstance(first_result, dict):
            pattern_library_path = _coerce_str(first_result.get("pattern_library_path"))
        if pattern_library_path is None and isinstance(summary_payload, dict):
            pattern_library_path = _coerce_str(summary_payload.get("pattern_library_path"))
        enabled_labels = _load_pattern_library_labels(Path(pattern_library_path)) if pattern_library_path else []
        if enabled_labels:
            source = "filtered-library"

    if not enabled_labels and available_labels:
        enabled_labels = list(available_labels)
    if not available_labels:
        available_labels = list(enabled_labels)
    if not disabled_labels and available_labels:
        enabled_set = set(enabled_labels)
        disabled_labels = [label for label in available_labels if label not in enabled_set]
    if enabled_labels and available_labels:
        enabled_set = set(enabled_labels)
        enabled_labels = [label for label in available_labels if label in enabled_set]
    return EvaluationLabelSelectionSummary(
        available_labels=available_labels,
        enabled_labels=enabled_labels,
        disabled_labels=disabled_labels,
        source=source,
    )


def _evaluation_task_token(task_id: str | None) -> str | None:
    if not task_id:
        return None
    return task_id.replace("/", "__").replace(":", "__")


def _evaluation_task_media(
    eval_root: Path | None,
    task_id: str | None,
    *,
    task_root: Path | None = None,
) -> dict[str, Any]:
    task_token = _evaluation_task_token(task_id)
    if eval_root is None or task_token is None:
        return {"task_token": task_token}
    resolved_task_root = task_root
    if resolved_task_root is None:
        tasks_root = eval_root / "tasks"
        if tasks_root.is_dir():
            for env_root in sorted(path for path in tasks_root.iterdir() if path.is_dir()):
                candidate = env_root / task_token
                if candidate.is_dir():
                    resolved_task_root = candidate
                    break
    if resolved_task_root is None or not resolved_task_root.is_dir():
        return {"task_token": task_token}
    attempt_roots = sorted(
        (
            path
            for path in resolved_task_root.iterdir()
            if path.is_dir() and (path.name.startswith("attempt_") or path.name.startswith("step_"))
        ),
        key=lambda path: path.name,
    )
    latest_root = attempt_roots[-1] if attempt_roots else None
    initial_dir = resolved_task_root / "initial"
    prompt_images: list[EvaluationPromptImageSummary] = []
    preferred_initial_candidates = []
    if initial_dir.is_dir():
        all_initial_candidates = sorted(initial_dir.glob("*.png"))
        for path in all_initial_candidates:
            prompt_images.append(
                EvaluationPromptImageSummary(
                    filename=path.name,
                    path=path,
                    label=path.stem.replace("_", " "),
                )
            )
        preferred_initial_candidates.extend(sorted(initial_dir.glob("*grid*.png")))
        preferred_initial_candidates.extend(all_initial_candidates)
    initial_scene_path = next(
        (path for path in preferred_initial_candidates if path.is_file()),
        (latest_root / "initial_scene.png" if latest_root is not None else None),
    )
    if not prompt_images and initial_scene_path is not None and initial_scene_path.is_file():
        prompt_images.append(
            EvaluationPromptImageSummary(
                filename=initial_scene_path.name,
                path=initial_scene_path,
                label=initial_scene_path.stem.replace("_", " "),
            )
        )
    simulation_path = latest_root / "simulation.mp4" if latest_root is not None else None
    return {
        "task_token": task_token,
        "initial_scene_path": initial_scene_path if initial_scene_path is not None and initial_scene_path.is_file() else None,
        "prompt_images": prompt_images,
        "simulation_path": simulation_path if simulation_path is not None and simulation_path.is_file() else None,
    }


def _task_id_from_task_root(task_root: Path, trace_path: Path | None) -> str | None:
    if trace_path is not None:
        try:
            payload = _load_json(trace_path)
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            task_id = _coerce_str(payload.get("task_id"))
            if task_id:
                return task_id
    name = task_root.name
    if "__" in name:
        return name.replace("__", ":")
    return name or None


def _terminal_status_from_trace(path: Path) -> str | None:
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    outcome = payload.get("outcome")
    if not isinstance(outcome, dict):
        return None
    terminal_status = _coerce_str(outcome.get("terminal_status"))
    if terminal_status:
        return terminal_status
    metrics = outcome.get("metrics")
    if isinstance(metrics, dict):
        return _coerce_str(metrics.get("status_name"))
    return None


def _native_metrics_from_trace(path: Path) -> dict[str, Any]:
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    outcome = payload.get("outcome")
    if not isinstance(outcome, dict):
        return {}
    metrics = outcome.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sample_success_from_trace(path: Path) -> bool | None:
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    env_id = _coerce_str(payload.get("environment"))
    trace_id = _coerce_str(payload.get("trace_id"))
    outcome = payload.get("outcome")
    if isinstance(outcome, dict):
        success = outcome.get("success")
        if isinstance(success, bool):
            return success
        metrics = outcome.get("metrics")
        terminal_status = _coerce_str(outcome.get("terminal_status"))
        if isinstance(metrics, dict):
            inferred = _infer_success_from_metrics(
                env_id=env_id,
                trace_id=trace_id,
                terminal_status=terminal_status,
                metrics=metrics,
            )
            if inferred is not None:
                return inferred
    inferred = _infer_success_from_trace_id(env_id=env_id, trace_id=trace_id)
    if inferred is not None:
        return inferred
    return None


def _infer_success_from_metrics(
    *,
    env_id: str | None,
    trace_id: str | None,
    terminal_status: str | None,
    metrics: dict[str, Any],
) -> bool | None:
    if env_id == "phyre":
        is_solved = metrics.get("is_solved")
        if isinstance(is_solved, bool):
            return is_solved
        status_name = _coerce_str(metrics.get("status_name")) or terminal_status
        if status_name is not None:
            return status_name.upper() == "SOLVED"
    if env_id == "iphyre":
        total_reward = _coerce_float(metrics.get("total_reward"))
        solved_threshold = _coerce_float(metrics.get("solved_threshold"))
        if total_reward is not None and solved_threshold is not None:
            return total_reward >= solved_threshold
        if terminal_status is not None:
            return terminal_status == "solved_threshold_reached"
    if env_id == "kinetix":
        task_success = metrics.get("task_success")
        if isinstance(task_success, bool):
            return task_success
        goal_reached = metrics.get("goal_reached")
        if isinstance(goal_reached, bool):
            return goal_reached
        info = metrics.get("info")
        if isinstance(info, dict):
            info_goal_reached = info.get("GoalR")
            if isinstance(info_goal_reached, bool):
                return info_goal_reached
        if terminal_status is not None:
            return terminal_status == "goal_reached"
        final_done = metrics.get("final_done")
        if isinstance(final_done, bool):
            return final_done
    if env_id == "pooltool":
        nine_ball_success = metrics.get("nine_ball_success")
        if isinstance(nine_ball_success, bool):
            return nine_ball_success
        if terminal_status is not None:
            return terminal_status == "SUCCESS"
    return _infer_success_from_trace_id(env_id=env_id, trace_id=trace_id)


def _infer_success_from_trace_id(*, env_id: str | None, trace_id: str | None) -> bool | None:
    if trace_id is None:
        return None
    if env_id == "phyre":
        parts = trace_id.split(":")
        if len(parts) >= 4:
            if parts[3] == "solved":
                return True
            if parts[3] == "not_solved":
                return False
    if env_id == "iphyre":
        parts = trace_id.split(":")
        if len(parts) >= 3:
            if parts[2] == "solved_threshold_reached":
                return True
            if parts[2] == "threshold_not_reached":
                return False
    if env_id == "kinetix":
        parts = trace_id.split(":")
        if len(parts) >= 3:
            if parts[2] == "goal_reached":
                return True
            if parts[2] == "max_steps_reached":
                return False
    if env_id == "pooltool":
        parts = trace_id.split(":")
        if len(parts) >= 3:
            if parts[2] == "success":
                return True
            if parts[2] in {"no_balls_potted", "other_balls_potted", "scratch", "target_ball_missed"}:
                return False
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = _coerce_str(item)
        if text is None:
            continue
        stripped = text.strip()
        if stripped:
            items.append(stripped)
    return items
