from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re


_LABEL_INDEX_PATTERN = re.compile(r"^label_(\d+)_")


def _label_palette_index(label_name: str, fallback_index: int) -> int:
    match = _LABEL_INDEX_PATTERN.match(label_name)
    if match is not None:
        return max(int(match.group(1)) - 1, 0)
    return fallback_index


@dataclass(frozen=True)
class DatasetSplitSummary:
    name: str
    target_count: int | None
    manifest_completed_count: int | None
    completed_count: int


@dataclass(frozen=True)
class DatasetSummary:
    env_id: str
    dataset_name: str
    root: Path
    splits: list[DatasetSplitSummary]
    manifest_path: Path | None
    total_completed_count: int


@dataclass(frozen=True)
class TargetLabelConfigSummary:
    env_id: str
    path: Path
    labels: list[str]
    raw_text: str

    @property
    def label_count(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class EvaluationRunSummary:
    run_id: str
    root: Path
    summary_path: Path | None
    results_path: Path | None
    status: str
    task_count: int
    success_count: int
    created_at: str
    benchmark_preset: str | None = None
    total_duration_seconds: float | None = None
    average_task_duration_seconds: float | None = None
    model_name: str | None = None
    prompt_format: str | None = None
    description_mode: str | None = None
    metric_summaries: list["EvaluationMetricSummary"] = field(default_factory=list)
    task_summaries: list["EvaluationTaskSummary"] = field(default_factory=list)
    terminal_status_counts: dict[str, int] = field(default_factory=dict)
    label_selection: "EvaluationLabelSelectionSummary" = field(
        default_factory=lambda: EvaluationLabelSelectionSummary(
            available_labels=[],
            enabled_labels=[],
            disabled_labels=[],
        )
    )

    @property
    def success_ratio(self) -> float | None:
        if self.task_count <= 0:
            return None
        return self.success_count / self.task_count


@dataclass(frozen=True)
class LibraryEvaluationSeedSummary:
    seed: int
    evaluation_run_id: str
    root: Path
    status: str
    evaluation: EvaluationRunSummary | None = None

    @property
    def success_ratio(self) -> float | None:
        if self.evaluation is None:
            return None
        return self.evaluation.success_ratio


@dataclass(frozen=True)
class LibraryEvaluationCaseSummary:
    case_id: str
    label: str
    kind: str
    held_out_labels: list[str]
    enabled_labels: list[str]
    seed_runs: list[LibraryEvaluationSeedSummary] = field(default_factory=list)

    @property
    def planned_seed_count(self) -> int:
        return len(self.seed_runs)

    @property
    def completed_seed_count(self) -> int:
        return sum(
            1
            for item in self.seed_runs
            if item.evaluation is not None and item.evaluation.status == "completed"
        )

    @property
    def mean_success_ratio(self) -> float | None:
        values = [
            item.success_ratio
            for item in self.seed_runs
            if item.success_ratio is not None
        ]
        if not values:
            return None
        return sum(values) / len(values)

    @property
    def min_success_ratio(self) -> float | None:
        values = [
            item.success_ratio
            for item in self.seed_runs
            if item.success_ratio is not None
        ]
        return min(values) if values else None

    @property
    def max_success_ratio(self) -> float | None:
        values = [
            item.success_ratio
            for item in self.seed_runs
            if item.success_ratio is not None
        ]
        return max(values) if values else None


@dataclass(frozen=True)
class LibraryEvaluationSummary:
    env_id: str
    source_run_id: str
    run_id: str
    root: Path
    created_at: str
    template_name: str
    benchmark_preset: str
    request_path: Path | None
    manifest_path: Path | None
    available_labels: list[str]
    case_summaries: list[LibraryEvaluationCaseSummary] = field(default_factory=list)
    raw_request_payload: dict[str, Any] = field(default_factory=dict)
    raw_manifest_payload: dict[str, Any] = field(default_factory=dict)

    @property
    def total_case_count(self) -> int:
        return len(self.case_summaries)

    @property
    def total_seed_run_count(self) -> int:
        return sum(case.planned_seed_count for case in self.case_summaries)

    @property
    def completed_seed_run_count(self) -> int:
        return sum(case.completed_seed_count for case in self.case_summaries)

    @property
    def baseline_case(self) -> LibraryEvaluationCaseSummary | None:
        return next((item for item in self.case_summaries if item.kind == "baseline"), None)

    @property
    def single_holdout_cases(self) -> list[LibraryEvaluationCaseSummary]:
        return [item for item in self.case_summaries if item.kind == "single"]

    @property
    def pair_holdout_cases(self) -> list[LibraryEvaluationCaseSummary]:
        return [item for item in self.case_summaries if item.kind == "pair"]


@dataclass(frozen=True)
class EvaluationMetricSummary:
    name: str
    mean: float | None
    minimum: float | None
    maximum: float | None
    count: int


@dataclass(frozen=True)
class EvaluationTaskSummary:
    task_id: str | None
    success: bool | None
    terminal_status: str | None
    record_count: int
    total_duration_seconds: float | None = None
    completed_at: str | None = None
    native_metrics: dict[str, Any] = field(default_factory=dict)
    task_token: str | None = None
    initial_scene_path: Path | None = None
    prompt_images: list["EvaluationPromptImageSummary"] = field(default_factory=list)
    simulation_path: Path | None = None


@dataclass(frozen=True)
class EvaluationTaskRecordSummary:
    index: int
    record_kind: str | None
    prompt_format: str | None
    description_mode: str | None
    system_prompt: str | None
    user_prompt: str | None
    trace_description: str | None
    raw_model_output: str | None
    duration_seconds: float | None = None
    parsed_action: dict[str, Any] = field(default_factory=dict)
    success: bool | None = None
    terminal_status: str | None = None
    native_metrics: dict[str, Any] = field(default_factory=dict)
    trace_annotation_overview: str | None = None
    trace_annotation_notes: list[str] = field(default_factory=list)
    trace_annotation_rows: list["EvaluationTraceAnnotationRow"] = field(default_factory=list)


@dataclass(frozen=True)
class EvaluationPromptImageSummary:
    filename: str
    path: Path
    label: str


@dataclass(frozen=True)
class EvaluationTraceAnnotationRow:
    step_text: str
    label: str
    parameter_text: str | None = None


@dataclass(frozen=True)
class EvaluationTaskDetailSummary:
    task_id: str | None
    task_token: str | None
    model_name: str | None
    prompt_format: str | None
    description_mode: str | None
    success: bool | None
    terminal_status: str | None
    record_count: int
    total_duration_seconds: float | None = None
    native_metrics: dict[str, Any] = field(default_factory=dict)
    initial_scene_path: Path | None = None
    prompt_images: list[EvaluationPromptImageSummary] = field(default_factory=list)
    simulation_path: Path | None = None
    records: list[EvaluationTaskRecordSummary] = field(default_factory=list)


@dataclass(frozen=True)
class ProposedLabelSummary:
    label: str
    description: str | None = None


@dataclass(frozen=True)
class EvaluationLabelSelectionSummary:
    available_labels: list[str]
    enabled_labels: list[str]
    disabled_labels: list[str]
    source: str | None = None

    @property
    def enabled_count(self) -> int:
        return len(self.enabled_labels)

    @property
    def disabled_count(self) -> int:
        return len(self.disabled_labels)


@dataclass(frozen=True)
class CandidateSummary:
    name: str
    root: Path
    code_path: Path | None
    metadata_path: Path | None
    average_score: float | None
    train_score: float | None
    val_score: float | None
    uid: str | None
    origin: str | None
    parent_uid: str | None = None
    secondary_parent_uid: str | None = None
    iteration: int | None = None
    is_best_candidate: bool = False
    loss_breakdown: list[dict[str, Any]] = field(default_factory=list)
    validation_loss_breakdown: list[dict[str, Any]] = field(default_factory=list)
    score_details: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.errors)


@dataclass(frozen=True)
class LabelSummary:
    name: str
    root: Path
    candidates: list[CandidateSummary]

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)


@dataclass(frozen=True)
class PatternEvolutionRecord:
    label_name: str
    candidate_uid: str | None
    candidate_name: str | None
    sequence: int
    iteration: int | None
    origin: str | None
    score: float | None
    train_score: float | None
    val_score: float | None
    parent_uid: str | None
    secondary_parent_uid: str | None


@dataclass(frozen=True)
class PatternEvolutionCandidateSummary:
    candidate_name: str | None
    candidate_uid: str | None
    iteration: int | None
    origin: str | None
    score: float | None
    train_score: float | None
    val_score: float | None
    parent_uid: str | None
    secondary_parent_uid: str | None
    is_best_candidate: bool = False


@dataclass(frozen=True)
class PatternEvolutionIterationSummary:
    iteration: int
    candidate_count: int
    best_score: float | None
    best_train_score: float | None
    best_val_score: float | None
    best_candidate_uid: str | None
    best_candidate_name: str | None
    origin_counts: dict[str, int] = field(default_factory=dict)
    top_candidates: list[PatternEvolutionCandidateSummary] = field(default_factory=list)


@dataclass(frozen=True)
class PatternEvolutionLabelSummary:
    label_name: str
    best_score: float | None
    best_train_score: float | None
    best_val_score: float | None
    best_candidate_uid: str | None
    best_candidate_name: str | None
    explored_iterations: int
    iteration_summaries: list[PatternEvolutionIterationSummary] = field(default_factory=list)
    recent_lineage: list[PatternEvolutionCandidateSummary] = field(default_factory=list)


@dataclass(frozen=True)
class LLMCallSummary:
    sequence: int
    created_at: str | None
    label: str | None
    label_index: int | None
    worker: int | None
    model: str | None
    temperature: float | None
    max_tokens: int | None
    timeout_seconds: int | None
    duration_seconds: float | None
    request_messages: list[dict[str, Any]] = field(default_factory=list)
    request_body: dict[str, Any] = field(default_factory=dict)
    request_headers: dict[str, Any] = field(default_factory=dict)
    response_text: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    router: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PatternRunSummary:
    env_id: str
    run_id: str
    root: Path
    library_path: Path | None
    run_json_path: Path | None
    average_score: float | None
    sample_count: int | None
    iteration_count: int | None
    label_summaries: list[LabelSummary]
    evaluation_runs: list[EvaluationRunSummary]
    created_at: str
    raw_run_payload: dict[str, Any] = field(default_factory=dict)

    @property
    def label_count(self) -> int:
        return len(self.label_summaries)

    @property
    def candidate_count(self) -> int:
        return sum(label.candidate_count for label in self.label_summaries)


@dataclass(frozen=True)
class PatternLabelTelemetrySummary:
    name: str
    root: Path
    candidates: list[CandidateSummary]
    best_candidate: CandidateSummary | None
    loss_breakdown: list[dict[str, Any]]
    iteration_history: list[PatternEvolutionRecord]
    evolution_summary: PatternEvolutionLabelSummary | None = None

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def evolution_loss_series(self) -> list[dict[str, Any]]:
        if self.evolution_summary is None:
            return []
        return [
            {
                "iteration": item.iteration,
                "train_score": item.best_train_score,
                "val_score": item.best_val_score,
            }
            for item in self.evolution_summary.iteration_summaries
            if item.best_train_score is not None or item.best_val_score is not None
        ]


@dataclass(frozen=True)
class PatternRunDetailSummary:
    env_id: str
    run_id: str
    root: Path
    library_path: Path | None
    run_json_path: Path | None
    evolution_record_path: Path | None
    average_score: float | None
    sample_count: int | None
    train_sample_count: int | None
    validation_sample_count: int | None
    iteration_count: int | None
    detector_count: int | None
    loss_breakdown: list[dict[str, Any]]
    score_details: list[dict[str, Any]]
    label_telemetry: list[PatternLabelTelemetrySummary]
    evaluation_runs: list[EvaluationRunSummary]
    evolution_history: list[PatternEvolutionRecord]
    llm_calls_path: Path | None
    llm_calls: list[LLMCallSummary]
    raw_run_payload: dict[str, Any] = field(default_factory=dict)

    @property
    def candidate_count(self) -> int:
        return sum(label.candidate_count for label in self.label_telemetry)

    @property
    def best_candidates(self) -> list[CandidateSummary]:
        return [
            label.best_candidate
            for label in self.label_telemetry
            if label.best_candidate is not None
        ]

    @property
    def mean_train_score(self) -> float | None:
        values = [
            candidate.train_score
            for candidate in self.best_candidates
            if candidate.train_score is not None
        ]
        if not values:
            return None
        return sum(values) / len(values)

    @property
    def mean_validation_score(self) -> float | None:
        values = [
            candidate.val_score
            for candidate in self.best_candidates
            if candidate.val_score is not None
        ]
        if not values:
            return None
        return sum(values) / len(values)

    @property
    def library_loss_series(self) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        for label in self.label_telemetry:
            if label.best_candidate is None:
                continue
            points.append(
                {
                    "label": label.name,
                    "train_score": label.best_candidate.train_score,
                    "val_score": label.best_candidate.val_score,
                }
            )
        return points

    @property
    def library_reward_trajectory(self) -> dict[str, Any]:
        series: list[dict[str, Any]] = []
        max_iteration = -1

        history_by_label: dict[str, list[PatternEvolutionRecord]] = {}
        for record in self.evolution_history:
            if (
                not record.label_name
            ):
                continue
            history_by_label.setdefault(record.label_name, []).append(record)

        if history_by_label:
            sorted_label_names = sorted(
                history_by_label,
                key=lambda label_name: (
                    _label_palette_index(label_name, 10_000),
                    label_name,
                ),
            )
            for fallback_index, label_name in enumerate(sorted_label_names):
                label_records = sorted(
                    history_by_label[label_name],
                    key=lambda item: item.sequence,
                )
                values = []
                for index, record in enumerate(label_records, start=1):
                    train_reward = (
                        record.train_score
                        if record.train_score is not None
                        else record.score
                    )
                    validation_reward = record.val_score
                    values.append(
                        {
                            "iteration": index,
                            "train_reward": train_reward,
                            "validation_reward": validation_reward,
                        }
                    )
                if values:
                    max_iteration = max(max_iteration, len(values))
                    series.append(
                        {
                            "label": label_name,
                            "palette_index": _label_palette_index(
                                label_name, fallback_index
                            ),
                            "values": values,
                        }
                    )
            return {
                "series": series,
                "max_iteration": max_iteration if max_iteration >= 0 else None,
            }

        for fallback_index, label in enumerate(self.label_telemetry):
            values: list[dict[str, Any]] = []
            ranked_candidates = sorted(
                label.candidates,
                key=lambda item: item.name,
            )
            for index, candidate in enumerate(ranked_candidates, start=1):
                train_reward = (
                    candidate.train_score
                    if candidate.train_score is not None
                    else candidate.average_score
                )
                validation_reward = candidate.val_score
                if train_reward is None and validation_reward is None:
                    continue
                values.append(
                    {
                        "iteration": index,
                        "train_reward": train_reward,
                        "validation_reward": validation_reward,
                    }
                )
                max_iteration = max(max_iteration, index)
            if values:
                series.append(
                    {
                        "label": label.name,
                        "palette_index": _label_palette_index(
                            label.name, fallback_index
                        ),
                        "values": values,
                    }
                )
        return {
            "series": series,
            "max_iteration": max_iteration if max_iteration >= 0 else None,
        }


# The dashboard workflow distinguishes supervised libraries from two derived-library modes:
# automatic runs built on top of supervised libraries, and true unsupervised runs proposed
# from trace-only observation.
SupervisedLibrarySummary = PatternRunSummary
SupervisedLibraryDetailSummary = PatternRunDetailSummary


@dataclass(frozen=True)
class UnsupervisedLibrarySummary:
    env_id: str
    run_id: str
    root: Path
    run_json_path: Path | None
    run_kind: str
    source_supervised_run_id: str | None
    source_evaluation_run_id: str | None
    created_at: str
    proposed_labels: list[str] = field(default_factory=list)
    proposed_label_details: list[ProposedLabelSummary] = field(default_factory=list)
    learned_label_count: int = 0
    evaluation_runs: list[EvaluationRunSummary] = field(default_factory=list)
    raw_run_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UnsupervisedLibraryDetailSummary:
    env_id: str
    run_id: str
    root: Path
    run_json_path: Path | None
    run_kind: str
    source_supervised_run_id: str | None
    source_evaluation_run_id: str | None
    source_supervised_library_path: Path | None
    source_supervised_evaluation_path: Path | None
    combined_library_path: Path | None
    unsupervised_library_path: Path | None
    created_at: str
    proposed_labels: list[str]
    proposed_label_details: list[ProposedLabelSummary]
    sampled_reasoning_trace_count: int | None
    sampled_reasoning_traces_path: Path | None
    label_proposal_response_path: Path | None
    label_proposal_response_text: str | None
    label_telemetry: list[PatternLabelTelemetrySummary]
    average_score: float | None
    sample_count: int | None
    iteration_count: int | None
    detector_count: int | None
    candidate_count: int
    evaluation_runs: list[EvaluationRunSummary]
    source_supervised_evaluation_runs: list[EvaluationRunSummary]
    evolution_history: list[PatternEvolutionRecord]
    llm_calls_path: Path | None
    llm_calls: list[LLMCallSummary]
    raw_run_payload: dict[str, Any] = field(default_factory=dict)

    @property
    def library_loss_series(self) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        for label in self.label_telemetry:
            if label.best_candidate is None:
                continue
            points.append(
                {
                    "label": label.name,
                    "train_score": label.best_candidate.train_score,
                    "val_score": label.best_candidate.val_score,
                }
            )
        return points

    @property
    def library_reward_trajectory(self) -> dict[str, Any]:
        series: list[dict[str, Any]] = []
        max_iteration = -1

        history_by_label: dict[str, list[PatternEvolutionRecord]] = {}
        for record in self.evolution_history:
            if not record.label_name:
                continue
            history_by_label.setdefault(record.label_name, []).append(record)

        if history_by_label:
            sorted_label_names = sorted(
                history_by_label,
                key=lambda label_name: (
                    _label_palette_index(label_name, 10_000),
                    label_name,
                ),
            )
            for fallback_index, label_name in enumerate(sorted_label_names):
                label_records = sorted(
                    history_by_label[label_name],
                    key=lambda item: item.sequence,
                )
                values = []
                for index, record in enumerate(label_records, start=1):
                    train_reward = (
                        record.train_score
                        if record.train_score is not None
                        else record.score
                    )
                    validation_reward = record.val_score
                    values.append(
                        {
                            "iteration": index,
                            "train_reward": train_reward,
                            "validation_reward": validation_reward,
                        }
                    )
                if values:
                    max_iteration = max(max_iteration, len(values))
                    series.append(
                        {
                            "label": label_name,
                            "palette_index": _label_palette_index(
                                label_name, fallback_index
                            ),
                            "values": values,
                        }
                    )
            return {
                "series": series,
                "max_iteration": max_iteration if max_iteration >= 0 else None,
            }

        for fallback_index, label in enumerate(self.label_telemetry):
            values: list[dict[str, Any]] = []
            ranked_candidates = sorted(
                label.candidates,
                key=lambda item: item.name,
            )
            for index, candidate in enumerate(ranked_candidates, start=1):
                train_reward = (
                    candidate.train_score
                    if candidate.train_score is not None
                    else candidate.average_score
                )
                validation_reward = candidate.val_score
                if train_reward is None and validation_reward is None:
                    continue
                values.append(
                    {
                        "iteration": index,
                        "train_reward": train_reward,
                        "validation_reward": validation_reward,
                    }
                )
                max_iteration = max(max_iteration, index)
            if values:
                series.append(
                    {
                        "label": label.name,
                        "palette_index": _label_palette_index(
                            label.name, fallback_index
                        ),
                        "values": values,
                    }
                )
        return {
            "series": series,
            "max_iteration": max_iteration if max_iteration >= 0 else None,
        }


@dataclass(frozen=True)
class DatasetIndexedSampleSummary:
    split_name: str
    sample_name: str
    sample_root: Path
    env_id: str
    sample_id: int
    absolute_sample_index: int
    status: str
    success: bool | None
    trace_id: str | None
    task_id: str | None
    trace_path: Path | None
    quality_path: Path | None
    evidence_path: Path | None
    scene_image_path: Path | None
    simulation_path: Path | None

    @property
    def has_evidence(self) -> bool:
        return self.evidence_path is not None

    @property
    def has_media(self) -> bool:
        return self.scene_image_path is not None or self.simulation_path is not None


@dataclass(frozen=True)
class DatasetSplitDetailSummary:
    name: str
    target_count: int | None
    manifest_completed_count: int | None
    completed_count: int
    indexed_samples: list[DatasetIndexedSampleSummary]

    @property
    def indexed_count(self) -> int:
        return len(self.indexed_samples)


@dataclass(frozen=True)
class DatasetDetailSummary:
    env_id: str
    dataset_name: str
    root: Path
    manifest_path: Path | None
    index_path: Path | None
    splits: list[DatasetSplitDetailSummary]
    raw_manifest_payload: dict[str, Any] = field(default_factory=dict)

    @property
    def total_completed_count(self) -> int:
        return sum(split.completed_count for split in self.splits)

    @property
    def total_indexed_count(self) -> int:
        return sum(split.indexed_count for split in self.splits)


@dataclass(frozen=True)
class DatasetSampleSummary:
    env_id: str
    dataset_name: str
    split_name: str
    sample_name: str
    sample_root: Path
    trace_path: Path | None
    quality_path: Path | None
    evidence_path: Path | None
    scene_image_path: Path | None
    simulation_path: Path | None
    scene_image_paths: list[Path] = field(default_factory=list)
    simulation_paths: list[Path] = field(default_factory=list)

    @property
    def has_evidence(self) -> bool:
        return self.evidence_path is not None

    @property
    def has_media(self) -> bool:
        return bool(self.scene_image_paths or self.simulation_paths)


@dataclass(frozen=True)
class EnvironmentDashboardSummary:
    env_id: str
    datasets: list[DatasetSummary]
    pattern_runs: list[PatternRunSummary]
    target_labels: TargetLabelConfigSummary | None = None
    supervised_libraries: list[SupervisedLibrarySummary] = field(default_factory=list)
    automatic_libraries: list[UnsupervisedLibrarySummary] = field(default_factory=list)
    unsupervised_libraries: list[UnsupervisedLibrarySummary] = field(default_factory=list)
    image_only_evaluations: list[EvaluationRunSummary] = field(default_factory=list)

    @property
    def latest_run(self) -> PatternRunSummary | None:
        return self.pattern_runs[0] if self.pattern_runs else None


@dataclass(frozen=True)
class PooltoolRewardOptimizationBudgetSummary:
    budget: int
    reward_kind: str
    scene_count: int
    success_rate: float
    mean_reward: float
    seed_count: int


@dataclass(frozen=True)
class PooltoolRewardOptimizationSummary:
    run_id: str
    root: Path
    goal_text: str
    dsl_program: str
    model_name: str
    source_run_id: str
    source_run_kind: str
    scene_count: int
    seed_count: int
    synthesized: bool
    budget_summaries: list[PooltoolRewardOptimizationBudgetSummary] = field(default_factory=list)


@dataclass(frozen=True)
class PooltoolRewardValueModelPointSummary:
    epoch: int
    train_loss: float
    validation_mean_reward: float
    validation_success_rate: float


@dataclass(frozen=True)
class PooltoolRewardValueModelSummary:
    run_id: str
    root: Path
    goal_text: str
    dsl_program: str
    model_name: str
    source_run_id: str
    source_run_kind: str
    train_sample_count: int
    validation_scene_count: int
    best_validation_mean_reward: float
    best_validation_success_rate: float
    synthesized: bool
    eval_curve: list[PooltoolRewardValueModelPointSummary] = field(default_factory=list)
