from __future__ import annotations

import json
import math
from queue import Empty, Queue
import re
import shutil
import time
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from debug import debug_log, progress_log
from deepphy_eval.io import load_trace
from datasets.manifest import load_dataset_manifest, resolve_dataset_root
from llm_config import GLOBAL_MIN_COMPLETION_TOKENS
from .config import load_pattern_labels
from pattern_library_detectors import apply_pattern_library
from pattern_library_evidence import build_pattern_evidence
from .losses import (
    LossComponentScore,
    LossConfig,
    compute_correlation_structure_score,
    default_loss_configs,
    select_correlation_pair_indices,
)
from .prompts import (
    FunSearchCandidateSummary,
    build_funsearch_prompt,
)
from pattern_library_registry import save_pattern_library
from pattern_library_schema import DetectorDefinition, PatternLibrary
from .scoring import _request_json, extract_chat_completion_text, request_chat_completion_with_text_retry
from predictions import (
    StoredPredictionTarget,
    load_learning_samples_from_prediction_manifest as _load_learning_samples_from_prediction_manifest,
    truncate_trace_for_cutoff,
)
from schema import CanonicalTrace
from weave_support import call_weave_op


INVALID_CANDIDATE_SCORE = -1_000_000.0
RUN_CONTROL_FILENAME = "run_control.json"
TIMING_BREAKDOWN_GROUPS: dict[str, list[str]] = {
    "search": [
        "code_writing",
        "code_evaluation",
        "search_overhead",
    ],
    "evaluation": [
        "train_pattern_application",
        "validation_pattern_application",
        "train_scoring",
        "validation_scoring",
    ],
}
TIMING_BREAKDOWN_META: dict[str, dict[str, str | int]] = {
    "code_writing": {
        "label": "Code writing",
        "unit": "request",
        "group": "search",
        "order": 10,
    },
    "code_evaluation": {
        "label": "Code evaluation",
        "unit": "candidate",
        "group": "search",
        "order": 20,
    },
    "search_overhead": {
        "label": "Search overhead",
        "unit": "label",
        "group": "search",
        "order": 30,
    },
    "train_pattern_application": {
        "label": "Train pattern application",
        "unit": "candidate",
        "group": "evaluation",
        "order": 40,
    },
    "validation_pattern_application": {
        "label": "Validation pattern application",
        "unit": "candidate",
        "group": "evaluation",
        "order": 50,
    },
    "train_scoring": {
        "label": "Train scoring",
        "unit": "candidate",
        "group": "evaluation",
        "order": 60,
    },
    "validation_scoring": {
        "label": "Validation scoring",
        "unit": "candidate",
        "group": "evaluation",
        "order": 70,
    },
}

PatternLearningSample = StoredPredictionTarget


@dataclass(frozen=True)
class PreparedLearningSample:
    sample: PatternLearningSample
    trace: CanonicalTrace
    reference_prediction: str
    existing_evidence: dict[str, Any] | None = None


@dataclass
class PatternLearningRun:
    output_path: Path
    library: PatternLibrary
    iteration_count: int
    workers: int
    sample_count: int
    train_sample_count: int
    validation_sample_count: int
    scorer_config: dict[str, Any]
    dataset_id: str | None = None
    dataset_version: str | None = None
    average_score: float | None = None
    train_score: float | None = None
    validation_score: float | None = None
    score_details: list[dict[str, Any]] = field(default_factory=list)
    loss_breakdown: list[dict[str, Any]] = field(default_factory=list)
    timing_summary: dict[str, Any] = field(default_factory=dict)
    readiness_notes: list[str] = field(default_factory=list)
    artifact_root: Path | None = None
    evolution_record_path: Path | None = None


@dataclass
class LabelLearningResult:
    label: str
    label_index: int
    worker: int
    detector: DetectorDefinition
    best_score: float | None
    best_train_score: float | None
    best_validation_score: float | None
    best_score_details: list[dict[str, Any]]
    best_loss_breakdown: list[dict[str, Any]]
    best_validation_loss_breakdown: list[dict[str, Any]]
    timings: TimingAccumulator
    timing_summary: dict[str, Any]
    payload: dict[str, Any]
    artifacts: dict[str, Any]
    candidate_scores: dict[str, dict[str, Any]]
    stopped_early: bool = False
    stop_reason: str | None = None
    patience_evaluations: int | None = None
    candidate_evaluations: int | None = None


@dataclass
class LLMCallRecorder:
    path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _next_sequence: int = 1

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._next_sequence = self._discover_next_sequence()

    def _discover_next_sequence(self) -> int:
        if not self.path.is_file():
            return 1
        max_seen = 0
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sequence = payload.get("sequence")
                    if isinstance(sequence, int):
                        max_seen = max(max_seen, sequence)
        except OSError:
            return 1
        return max_seen + 1

    def append(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            record = {
                "sequence": self._next_sequence,
                "created_at": datetime.now(timezone.utc).isoformat(),
                **payload,
            }
            self._next_sequence += 1
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True))
                handle.write("\n")
            return record


@dataclass(frozen=True)
class CompletedLabelCheckpoint:
    label: str
    label_index: int
    worker: int
    detector: DetectorDefinition
    best_score: float | None
    best_train_score: float | None
    best_validation_score: float | None
    best_score_details: list[dict[str, Any]]
    best_loss_breakdown: list[dict[str, Any]]
    best_validation_loss_breakdown: list[dict[str, Any]]
    timing_summary: dict[str, Any]
    timings: TimingAccumulator
    artifacts: dict[str, Any]
    stopped_early: bool = False
    stop_reason: str | None = None
    patience_evaluations: int | None = None
    candidate_evaluations: int | None = None


@dataclass(frozen=True)
class LossRuntimeContext:
    correlation_pair_indices: list[tuple[int, int]] | None = None
    correlation_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimingAccumulator:
    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, key: str, duration_seconds: float, *, count: int = 1) -> None:
        if duration_seconds < 0:
            duration_seconds = 0.0
        self.totals[key] = self.totals.get(key, 0.0) + duration_seconds
        self.counts[key] = self.counts.get(key, 0) + count

    def total(self, key: str) -> float:
        return self.totals.get(key, 0.0)

    def count(self, key: str) -> int:
        return self.counts.get(key, 0)

    def merge(self, other: TimingAccumulator) -> None:
        for key, total in other.totals.items():
            self.totals[key] = self.totals.get(key, 0.0) + total
        for key, count in other.counts.items():
            self.counts[key] = self.counts.get(key, 0) + count


def load_learning_samples_from_prediction_manifest(
    manifest_path: Path,
    *,
    split_names: list[str] | None = None,
    limit: int | None = None,
) -> tuple[list[PatternLearningSample], dict[str, Any]]:
    return _load_learning_samples_from_prediction_manifest(
        manifest_path,
        split_names=split_names,
        limit=limit,
    )


def load_learning_samples_from_dataset_manifest(
    manifest_path: Path,
    *,
    split_names: list[str] | None = None,
    limit: int | None = None,
) -> tuple[list[PatternLearningSample], dict[str, Any]]:
    manifest = load_dataset_manifest(manifest_path)
    selected = set(split_names or [])
    source_root = resolve_dataset_root(manifest_path, manifest)
    samples: list[PatternLearningSample] = []
    for split in manifest.splits:
        if selected and split.name not in selected:
            continue
        split_root = source_root / split.name
        sample_dirs = sorted(path for path in split_root.glob("sample_*") if path.is_dir())
        for sample_dir in sample_dirs:
            trace_path = sample_dir / "trace.json"
            if not trace_path.is_file():
                continue
            trace = load_trace(trace_path)
            total_timestep_count = len(trace.timesteps)
            cutoff_index = max(total_timestep_count - 1, 0)
            cutoff_t = trace.timesteps[cutoff_index].t if total_timestep_count else None
            cutoff_fraction = 1.0 if total_timestep_count > 1 else 0.0
            samples.append(
                PatternLearningSample(
                    trace_path=trace_path,
                    prediction_path=trace_path,
                    environment_id=trace.environment,
                    task_id=trace.task_id,
                    prediction_text="",
                    cutoff_index=cutoff_index,
                    cutoff_t=cutoff_t,
                    cutoff_fraction=cutoff_fraction,
                    observed_timestep_count=total_timestep_count,
                    total_timestep_count=total_timestep_count,
                    task_context={},
                )
            )
            if limit is not None and len(samples) >= limit:
                break
        if limit is not None and len(samples) >= limit:
            break
    return samples, {
        "dataset_id": manifest.dataset_id,
        "dataset_version": manifest.trace_schema_version,
        "source": "dataset_manifest",
    }


def _resolve_early_stop_thresholds(
    *,
    min_candidate_evaluations: int,
    no_improvement_patience_evaluations: int,
) -> tuple[int, int]:
    return (
        max(1, int(min_candidate_evaluations)),
        max(1, int(no_improvement_patience_evaluations)),
    )


def run_pattern_learning(
    *,
    environment_id: str,
    primitive_labels: list[str],
    label_descriptions: dict[str, str] | None = None,
    samples: list[PatternLearningSample],
    validation_samples: list[PatternLearningSample],
    output_path: Path,
    scorer_config: dict[str, Any] | None = None,
    iteration_count: int = 8,
    label_workers: int = 1,
    primitive_source: str | None = None,
    dataset_id: str | None = None,
    dataset_version: str | None = None,
    generator_model: str | None = None,
    generator_base_url: str | None = None,
    generator_timeout: int = 600,
    generator_retries: int = 3,
    seed: int = 42,
    population_size: int = 64,
    elite_fraction: float = 0.25,
    children_per_elite: int = 2,
    crossover_rate: float = 1.0,
    patience_factor: float = 0.35,
    min_candidate_evaluations: int = 100,
    no_improvement_patience_evaluations: int = 15,
    max_candidate_evaluations: int = 250,
    accept_margin: float = 0.001,
    max_repairs: int = 2,
    mutate_temperature: float = 0.7,
    mutate_max_tokens: int = GLOBAL_MIN_COMPLETION_TOKENS,
    losses: list[LossConfig] | None = None,
    artifact_root: Path | None = None,
    base_library: PatternLibrary | None = None,
    detector_input_mode: Literal["trace", "supervised_evidence"] = "trace",
    resume: bool = False,
) -> PatternLearningRun:
    run_started_at = time.perf_counter()
    clean_labels = [label.strip() for label in primitive_labels if label.strip()]
    label_descriptions = dict(label_descriptions or {})
    if not clean_labels:
        raise ValueError("primitive_labels must contain at least one non-empty label")
    if not samples:
        raise ValueError("samples must not be empty")
    if generator_model is None or not generator_base_url:
        raise ValueError("generator_model and generator_base_url are required for FunSearch")
    active_losses = list(losses or default_loss_configs())
    if not active_losses:
        raise ValueError("losses must contain at least one configured loss")
    min_candidate_evaluations, patience_evaluations = _resolve_early_stop_thresholds(
        min_candidate_evaluations=min_candidate_evaluations,
        no_improvement_patience_evaluations=no_improvement_patience_evaluations,
    )
    artifact_root = artifact_root or (output_path.parent / output_path.stem)
    requested_label_workers = max(1, min(label_workers, len(clean_labels)))
    worker_control_path = artifact_root / RUN_CONTROL_FILENAME
    if resume:
        # Resume should not silently downshift below the currently requested
        # worker target just because run_control.json is stale from an older run.
        persisted_workers = _read_desired_worker_count(
            worker_control_path,
            fallback=requested_label_workers,
            max_workers=len(clean_labels),
        )
        effective_label_workers = max(requested_label_workers, persisted_workers)
    else:
        effective_label_workers = requested_label_workers

    debug_log(
        "patterns.learning.start",
        environment_id=environment_id,
        sample_count=len(samples),
        primitive_label_count=len(clean_labels),
        output_path=output_path,
        iteration_count=iteration_count,
        workers=effective_label_workers,
        generator_model=generator_model,
        generator_base_url=generator_base_url,
        generator_timeout=generator_timeout,
        generator_retries=generator_retries,
        losses=[loss.kind for loss in active_losses],
        crossover_rate=crossover_rate,
    )
    progress_log(
        "patterns.learning.start",
        environment_id=environment_id,
        sample_count=len(samples),
        validation_sample_count=len(validation_samples),
        primitive_label_count=len(clean_labels),
        iteration_count=iteration_count,
        workers=effective_label_workers,
    )
    prepared_samples, readiness_notes = _prepare_learning_samples(
        environment_id=environment_id,
        samples=samples,
        base_library=base_library,
    )
    if not prepared_samples:
        raise ValueError(f"No learning samples matched environment {environment_id}")
    prepared_validation_samples, validation_readiness_notes = _prepare_learning_samples(
        environment_id=environment_id,
        samples=validation_samples,
        base_library=base_library,
    )
    if not prepared_validation_samples:
        raise ValueError(
            f"No validation samples matched environment {environment_id}"
        )

    if resume:
        if not artifact_root.exists():
            raise FileNotFoundError(
                f"Cannot resume pattern learning; artifact root does not exist: {artifact_root}"
            )
    else:
        if artifact_root.exists():
            shutil.rmtree(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    _write_worker_control(
        worker_control_path,
        desired_workers=effective_label_workers,
        configured_workers=requested_label_workers,
    )
    evolution_record_path = artifact_root / "evolution_record.jsonl"
    live_evolution_record_path = artifact_root / "evolution_live.jsonl"
    llm_calls_path = artifact_root / "llm_calls.jsonl"
    llm_call_recorder = LLMCallRecorder(llm_calls_path)
    train_loss_runtime_contexts = _build_loss_runtime_contexts(
        losses=active_losses,
        sample_count=len(prepared_samples),
        seed=seed,
        split_name="train",
    )
    validation_loss_runtime_contexts = _build_loss_runtime_contexts(
        losses=active_losses,
        sample_count=len(prepared_validation_samples),
        seed=seed,
        split_name="validation",
    )

    completed_checkpoints = _load_completed_label_checkpoints(
        artifact_root=artifact_root,
        environment_id=environment_id,
        expected_labels=clean_labels,
    )
    manually_completed_label_indices = _read_manually_completed_label_indices(
        worker_control_path,
        expected_labels=clean_labels,
    )
    manually_completed_label_indices.difference_update(completed_checkpoints)
    manually_completed_labels = [
        clean_labels[index - 1]
        for index in sorted(manually_completed_label_indices)
        if 1 <= index <= len(clean_labels)
    ]
    if resume:
        progress_log(
            "patterns.learning.resume",
            environment_id=environment_id,
            resumed_label_count=len(completed_checkpoints),
            remaining_label_count=max(
                len(clean_labels)
                - len(completed_checkpoints)
                - len(manually_completed_label_indices),
                0,
            ),
        )
    score_details = [
        detail
        for checkpoint in completed_checkpoints.values()
        for detail in checkpoint.best_score_details
    ]
    detectors = [
        checkpoint.detector
        for _, checkpoint in sorted(completed_checkpoints.items())
    ]
    label_summaries = [
        _label_summary_payload(artifact_root=artifact_root, checkpoint=checkpoint)
        for _, checkpoint in sorted(completed_checkpoints.items())
    ]
    run_timings = TimingAccumulator()
    for checkpoint in completed_checkpoints.values():
        run_timings.merge(checkpoint.timings)
    progress_log(
        "patterns.learning.runtime_ready",
        environment_id=environment_id,
        train_sample_count=len(prepared_samples),
        validation_sample_count=len(prepared_validation_samples),
    )
    _write_run_artifacts(
        artifact_root=artifact_root,
        library_id=output_path.stem or "learned-pattern-library",
        environment_ids=[environment_id],
        sample_count=len(prepared_samples),
        train_sample_count=len(prepared_samples),
        validation_sample_count=len(prepared_validation_samples),
        iteration_count=iteration_count,
        workers=effective_label_workers,
        primitive_label_count=len(clean_labels),
        average_score=_aggregate_label_metric(label_summaries, "train_score"),
        train_score=_aggregate_label_metric(label_summaries, "train_score"),
        validation_score=_aggregate_label_metric(label_summaries, "validation_score"),
        score_details=score_details,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        primitive_source=primitive_source,
        scorer_config=scorer_config or {},
        label_summaries=label_summaries,
        evolution_record_path=evolution_record_path,
        loss_breakdown=_aggregate_label_loss_breakdown(label_summaries),
        timing_summary=_build_timing_summary(run_timings, total_wall_seconds=0.0),
        detector_count=len(detectors),
        patience_factor=patience_factor,
        min_candidate_evaluations=min_candidate_evaluations,
        patience_evaluations=patience_evaluations,
        max_candidate_evaluations=max_candidate_evaluations,
        detector_input_mode=detector_input_mode,
        status="running",
        llm_calls_path=llm_calls_path,
        configured_workers=requested_label_workers,
        worker_control_path=worker_control_path,
        manually_completed_labels=manually_completed_labels,
    )

    persistence_lock = threading.Lock()

    def persist_label_result(label_result: LabelLearningResult) -> None:
        checkpoint = _completed_label_checkpoint_from_result(label_result)
        with persistence_lock:
            completed_checkpoints[label_result.label_index] = checkpoint
            run_timings.merge(label_result.timings)
            detectors[:] = [
                item.detector
                for _, item in sorted(completed_checkpoints.items())
            ]
            score_details[:] = [
                detail
                for _, item in sorted(completed_checkpoints.items())
                for detail in item.best_score_details
            ]
            _write_label_candidate_artifacts(
                artifact_root=artifact_root,
                label_index=label_result.label_index,
                label_result=label_result,
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                primitive_source=primitive_source,
                scorer_config=scorer_config or {},
            )
            _append_evolution_record(
                path=evolution_record_path,
                label_index=label_result.label_index,
                label=label_result.label,
                payload=label_result.payload,
            )
            label_summaries[:] = [
                _label_summary_payload(artifact_root=artifact_root, checkpoint=item)
                for _, item in sorted(completed_checkpoints.items())
            ]
            _write_run_artifacts(
                artifact_root=artifact_root,
                library_id=output_path.stem or "learned-pattern-library",
                environment_ids=[environment_id],
                sample_count=len(prepared_samples),
                train_sample_count=len(prepared_samples),
                validation_sample_count=len(prepared_validation_samples),
                iteration_count=iteration_count,
                workers=effective_label_workers,
                primitive_label_count=len(clean_labels),
                average_score=_aggregate_label_metric(label_summaries, "train_score"),
                train_score=_aggregate_label_metric(label_summaries, "train_score"),
                validation_score=_aggregate_label_metric(label_summaries, "validation_score"),
                score_details=score_details,
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                primitive_source=primitive_source,
                scorer_config=scorer_config or {},
                label_summaries=label_summaries,
                evolution_record_path=evolution_record_path,
                loss_breakdown=_aggregate_label_loss_breakdown(label_summaries),
                timing_summary=_build_timing_summary(
                    run_timings,
                    total_wall_seconds=max(time.perf_counter() - run_started_at, 0.0),
                ),
                detector_count=len(detectors),
                patience_factor=patience_factor,
                min_candidate_evaluations=min_candidate_evaluations,
                patience_evaluations=patience_evaluations,
                max_candidate_evaluations=max_candidate_evaluations,
                detector_input_mode=detector_input_mode,
                status="running",
                llm_calls_path=llm_calls_path,
                configured_workers=requested_label_workers,
                worker_control_path=worker_control_path,
                manually_completed_labels=manually_completed_labels,
            )

    label_results = _run_label_learning_workers(
        environment_id=environment_id,
        clean_labels=clean_labels,
        prepared_samples=prepared_samples,
        prepared_validation_samples=prepared_validation_samples,
        output_root=artifact_root,
        generator_model=generator_model,
        generator_base_url=generator_base_url,
        generator_timeout=generator_timeout,
        generator_retries=generator_retries,
        iteration_count=iteration_count,
        seed=seed,
        population_size=population_size,
        elite_fraction=elite_fraction,
        children_per_elite=children_per_elite,
        patience_factor=patience_factor,
        min_candidate_evaluations=min_candidate_evaluations,
        no_improvement_patience_evaluations=patience_evaluations,
        max_candidate_evaluations=max_candidate_evaluations,
        crossover_rate=crossover_rate,
        accept_margin=accept_margin,
        max_repairs=max_repairs,
        mutate_temperature=mutate_temperature,
        mutate_max_tokens=mutate_max_tokens,
        losses=active_losses,
        train_loss_runtime_contexts=train_loss_runtime_contexts,
        validation_loss_runtime_contexts=validation_loss_runtime_contexts,
        base_library=base_library,
        detector_input_mode=detector_input_mode,
        label_descriptions=label_descriptions,
        label_workers=effective_label_workers,
        worker_control_path=worker_control_path,
        skipped_label_indices=set(completed_checkpoints) | manually_completed_label_indices,
        on_label_complete=persist_label_result,
        llm_call_recorder=llm_call_recorder,
    )
    if len(completed_checkpoints) + len(manually_completed_label_indices) != len(clean_labels):
        raise RuntimeError(
            "Pattern learning completed without producing checkpoints for all labels"
        )

    run_loss_breakdown = _aggregate_label_loss_breakdown(label_summaries)
    train_score = _aggregate_label_metric(label_summaries, "train_score")
    validation_score = _aggregate_label_metric(label_summaries, "validation_score")
    average_score = train_score
    run_total_wall_seconds = max(time.perf_counter() - run_started_at, 0.0)
    timing_summary = _build_timing_summary(
        run_timings,
        total_wall_seconds=run_total_wall_seconds,
    )

    library = PatternLibrary(
        library_id=output_path.stem or "learned-pattern-library",
        environment_ids=[environment_id],
        primitive_source=primitive_source,
        scorer_config=scorer_config or {},
        detectors=detectors,
        metadata={
            "iteration_count": iteration_count,
            "workers": _read_desired_worker_count(
                worker_control_path,
                fallback=effective_label_workers,
                max_workers=len(clean_labels),
            ),
            "configured_workers": requested_label_workers,
            "generator_model": generator_model,
            "generator_base_url": generator_base_url,
            "generator_timeout": generator_timeout,
            "generator_retries": generator_retries,
            "seed": seed,
            "population_size": population_size,
            "elite_fraction": elite_fraction,
            "children_per_elite": children_per_elite,
            "patience_factor": patience_factor,
            "min_candidate_evaluations": min_candidate_evaluations,
            "patience_evaluations": patience_evaluations,
            "no_improvement_patience_evaluations": patience_evaluations,
            "max_candidate_evaluations": max_candidate_evaluations,
            "crossover_rate": crossover_rate,
            "accept_margin": accept_margin,
            "max_repairs": max_repairs,
            "losses": [loss.model_dump(mode="json") for loss in active_losses],
            "label_runs": label_summaries,
            "manually_completed_labels": manually_completed_labels,
        },
    )

    output_path = save_pattern_library(library, output_path)
    _write_run_artifacts(
        artifact_root=artifact_root,
        library_id=library.library_id,
        environment_ids=list(library.environment_ids),
        sample_count=len(prepared_samples),
        train_sample_count=len(prepared_samples),
        validation_sample_count=len(prepared_validation_samples),
        iteration_count=iteration_count,
        workers=effective_label_workers,
        primitive_label_count=len(clean_labels),
        average_score=average_score,
        train_score=train_score,
        validation_score=validation_score,
        score_details=score_details,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        primitive_source=primitive_source,
        scorer_config=scorer_config or {},
        label_summaries=label_summaries,
        evolution_record_path=evolution_record_path,
        loss_breakdown=run_loss_breakdown,
        timing_summary=timing_summary,
        detector_count=len(library.detectors),
        patience_factor=patience_factor,
        min_candidate_evaluations=min_candidate_evaluations,
        patience_evaluations=patience_evaluations,
        max_candidate_evaluations=max_candidate_evaluations,
        detector_input_mode=detector_input_mode,
        status="completed",
        llm_calls_path=llm_calls_path,
        configured_workers=requested_label_workers,
        worker_control_path=worker_control_path,
        manually_completed_labels=manually_completed_labels,
    )
    debug_log("patterns.learning.done", average_score=average_score, scored_count=len(score_details))
    progress_log(
        "patterns.learning.done",
        average_score=average_score,
        detector_count=len(detectors),
        scored_count=len(score_details),
    )

    return PatternLearningRun(
        output_path=output_path,
        library=library,
        iteration_count=iteration_count,
        workers=effective_label_workers,
        sample_count=len(prepared_samples),
        train_sample_count=len(prepared_samples),
        validation_sample_count=len(prepared_validation_samples),
        scorer_config=scorer_config or {},
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        average_score=average_score,
        train_score=train_score,
        validation_score=validation_score,
        score_details=score_details,
        loss_breakdown=run_loss_breakdown,
        timing_summary=timing_summary,
        readiness_notes=[
            *readiness_notes,
            *validation_readiness_notes,
        ],
        artifact_root=artifact_root,
        evolution_record_path=evolution_record_path,
    )


def _prepare_learning_samples(
    *,
    environment_id: str,
    samples: list[PatternLearningSample],
    base_library: PatternLibrary | None = None,
) -> tuple[list[PreparedLearningSample], list[str]]:
    prepared: list[PreparedLearningSample] = []
    readiness_notes: list[str] = []
    for sample in samples:
        trace = load_trace(sample.trace_path)
        if trace.environment != environment_id:
            readiness_notes.append(
                f"Skipped {sample.trace_path} because environment {trace.environment} != {environment_id}"
            )
            continue
        truncated_trace = truncate_trace_for_cutoff(trace, sample.cutoff_index)
        prepared.append(
            PreparedLearningSample(
                sample=sample,
                trace=truncated_trace,
                reference_prediction=sample.prediction_text,
                existing_evidence=_existing_evidence_for_trace(
                    trace=truncated_trace,
                    base_library=base_library,
                ),
            )
        )
    return prepared, readiness_notes


def _run_label_learning_workers(
    *,
    environment_id: str,
    clean_labels: list[str],
    prepared_samples: list[PreparedLearningSample],
    prepared_validation_samples: list[PreparedLearningSample],
    output_root: Path,
    generator_model: str,
    generator_base_url: str,
    generator_timeout: int,
    generator_retries: int,
    iteration_count: int,
    seed: int,
    population_size: int,
    elite_fraction: float,
    children_per_elite: int,
    patience_factor: float,
    min_candidate_evaluations: int,
    no_improvement_patience_evaluations: int,
    max_candidate_evaluations: int,
    crossover_rate: float,
    accept_margin: float,
    max_repairs: int,
    mutate_temperature: float,
    mutate_max_tokens: int,
    losses: list[LossConfig],
    train_loss_runtime_contexts: list[LossRuntimeContext],
    validation_loss_runtime_contexts: list[LossRuntimeContext],
    base_library: PatternLibrary | None,
    detector_input_mode: Literal["trace", "supervised_evidence"],
    label_descriptions: dict[str, str],
    label_workers: int,
    worker_control_path: Path | None = None,
    skipped_label_indices: set[int] | None = None,
    on_label_complete: Any | None = None,
    llm_call_recorder: LLMCallRecorder | None = None,
) -> list[LabelLearningResult]:
    queue: Queue[tuple[int, str]] = Queue()
    skipped = skipped_label_indices or set()
    for label_index, label in enumerate(clean_labels, start=1):
        if label_index in skipped:
            continue
        queue.put((label_index, label))

    results: dict[int, LabelLearningResult] = {}
    errors: list[BaseException] = []
    result_lock = threading.Lock()
    stop_event = threading.Event()
    desired_workers_lock = threading.Lock()

    def desired_worker_count() -> int:
        with desired_workers_lock:
            return _read_desired_worker_count(
                worker_control_path,
                fallback=label_workers,
                max_workers=queued_label_count,
            )

    def worker_loop(worker_id: int) -> None:
        while not stop_event.is_set():
            if worker_id > desired_worker_count():
                return
            try:
                label_index, label = queue.get(timeout=0.1)
            except Empty:
                if queue.empty():
                    return
                continue
            try:
                debug_log(
                    "patterns.learning.label.start",
                    label=label,
                    index=label_index,
                    total_labels=len(clean_labels),
                    worker=worker_id,
                )
                progress_log(
                    "patterns.learning.label.start",
                    label=label,
                    index=label_index,
                    total_labels=len(clean_labels),
                    worker=worker_id,
                )
                label_result = _run_label_funsearch(
                    environment_id=environment_id,
                    label=label,
                    label_index=label_index,
                    worker=worker_id,
                    prepared_samples=prepared_samples,
                    prepared_validation_samples=prepared_validation_samples,
                    output_root=output_root,
                    generator_model=generator_model,
                    generator_base_url=generator_base_url,
                    generator_timeout=generator_timeout,
                    generator_retries=generator_retries,
                    iteration_count=iteration_count,
                    seed=seed,
                    population_size=population_size,
                    elite_fraction=elite_fraction,
                    children_per_elite=children_per_elite,
                    patience_factor=patience_factor,
                    min_candidate_evaluations=min_candidate_evaluations,
                    patience_evaluations=no_improvement_patience_evaluations,
                    max_candidate_evaluations=max_candidate_evaluations,
                    crossover_rate=crossover_rate,
                    accept_margin=accept_margin,
                    max_repairs=max_repairs,
                    mutate_temperature=mutate_temperature,
                    mutate_max_tokens=mutate_max_tokens,
                    losses=losses,
                    train_loss_runtime_contexts=train_loss_runtime_contexts,
                    validation_loss_runtime_contexts=validation_loss_runtime_contexts,
                    base_library=base_library,
                    detector_input_mode=detector_input_mode,
                    label_description=label_descriptions.get(label),
                    llm_call_recorder=llm_call_recorder,
                )
                with result_lock:
                    results[label_index] = label_result
                if on_label_complete is not None:
                    on_label_complete(label_result)
                debug_log(
                    "patterns.learning.label.done",
                    label=label,
                    worker=worker_id,
                    index=label_index,
                    best_candidate_uid=label_result.payload["best_candidate"]["uid"],
                    best_score=label_result.best_score,
                    validation_score=label_result.best_validation_score,
                    stopped_early=label_result.stopped_early,
                    stop_reason=label_result.stop_reason,
                    candidate_evaluations=label_result.candidate_evaluations,
                )
                progress_log(
                    "patterns.learning.label.done",
                    label=label,
                    worker=worker_id,
                    index=label_index,
                    best_candidate_uid=label_result.payload["best_candidate"]["uid"],
                    best_score=label_result.best_score,
                    validation_score=label_result.best_validation_score,
                    stopped_early=label_result.stopped_early,
                    stop_reason=label_result.stop_reason,
                    candidate_evaluations=label_result.candidate_evaluations,
                )
            except BaseException as exc:  # noqa: BLE001
                with result_lock:
                    errors.append(exc)
                stop_event.set()
            finally:
                queue.task_done()

    queued_label_count = max(len(clean_labels) - len(skipped), 0)
    if queued_label_count == 0:
        return []
    threads: list[threading.Thread] = []
    next_worker_id = 1
    last_reported_desired_workers: int | None = None

    def alive_threads() -> list[threading.Thread]:
        return [thread for thread in threads if thread.is_alive()]

    def spawn_workers() -> None:
        nonlocal next_worker_id, last_reported_desired_workers
        desired = desired_worker_count()
        alive = alive_threads()
        if desired != last_reported_desired_workers:
            progress_log(
                "patterns.learning.workers.updated",
                desired_workers=desired,
                active_workers=len(alive),
                queued_labels=queue.qsize(),
            )
            last_reported_desired_workers = desired
        target = min(desired, queued_label_count)
        while len(alive) < target and not queue.empty():
            thread = threading.Thread(
                target=worker_loop,
                args=(next_worker_id,),
                daemon=False,
            )
            next_worker_id += 1
            threads.append(thread)
            thread.start()
            alive.append(thread)

    spawn_workers()
    while True:
        if errors:
            break
        active = alive_threads()
        if not active and queue.empty():
            break
        spawn_workers()
        time.sleep(0.1)
    for thread in threads:
        thread.join()

    if errors:
        raise errors[0]
    return [results[index] for index in sorted(results)]


def _run_label_funsearch(
    *,
    environment_id: str,
    label: str,
    label_index: int,
    worker: int,
    prepared_samples: list[PreparedLearningSample],
    prepared_validation_samples: list[PreparedLearningSample],
    output_root: Path,
    generator_model: str,
    generator_base_url: str,
    generator_timeout: int,
    generator_retries: int,
    iteration_count: int,
    seed: int,
    population_size: int,
    elite_fraction: float,
    children_per_elite: int,
    patience_factor: float,
    min_candidate_evaluations: int,
    patience_evaluations: int,
    max_candidate_evaluations: int,
    crossover_rate: float,
    accept_margin: float,
    max_repairs: int,
    mutate_temperature: float,
    mutate_max_tokens: int,
    losses: list[LossConfig],
    train_loss_runtime_contexts: list[LossRuntimeContext],
    validation_loss_runtime_contexts: list[LossRuntimeContext],
    base_library: PatternLibrary | None,
    detector_input_mode: Literal["trace", "supervised_evidence"],
    label_description: str | None,
    llm_call_recorder: LLMCallRecorder | None,
) -> LabelLearningResult:
    score_cache: dict[str, dict[str, Any]] = {}
    evaluation_count = 0
    label_timings = TimingAccumulator()
    label_started_at = time.perf_counter()
    label_dir = output_root / f"label_{label_index:03d}_{_slugify(label)}"

    def evaluator(candidate_code: str, *, settings: Any) -> float:
        nonlocal evaluation_count
        evaluation_count += 1
        candidate_started_at = time.perf_counter()
        candidate_timings = TimingAccumulator()
        candidate_fingerprint = f"{abs(hash(candidate_code)) & 0xFFFFFFFF:08x}"
        debug_log(
            "patterns.learning.candidate.start",
            label=label,
            worker=worker,
            evaluation_index=evaluation_count,
            candidate_fingerprint=candidate_fingerprint,
        )
        progress_log(
            "patterns.learning.candidate.start",
            label=label,
            worker=worker,
            evaluation_index=evaluation_count,
            candidate_fingerprint=candidate_fingerprint,
        )
        errors: list[str] = []
        detector = _detector_from_candidate(
            label=label,
            label_index=label_index,
            environment_id=environment_id,
            candidate_code=candidate_code,
            metadata={
                "generated_by": "funsearch",
                "label_description": label_description,
            },
            input_mode=detector_input_mode,
        )
        candidate_detectors = [*base_library.detectors] if base_library is not None else []
        candidate_detectors.append(detector)
        library = PatternLibrary(
            library_id=f"{_slugify(label)}-candidate",
            environment_ids=[environment_id],
            detectors=candidate_detectors,
        )
        train_materialize_started_at = time.perf_counter()
        train_patterns, train_errors = _materialize_candidate_patterns(
            prepared_samples=prepared_samples,
            library=library,
        )
        train_materialize_seconds = time.perf_counter() - train_materialize_started_at
        label_timings.add("train_pattern_application", train_materialize_seconds)
        candidate_timings.add("train_pattern_application", train_materialize_seconds)
        validation_materialize_started_at = time.perf_counter()
        validation_patterns, validation_errors = _materialize_candidate_patterns(
            prepared_samples=prepared_validation_samples,
            library=library,
        )
        validation_materialize_seconds = (
            time.perf_counter() - validation_materialize_started_at
        )
        label_timings.add(
            "validation_pattern_application",
            validation_materialize_seconds,
        )
        candidate_timings.add(
            "validation_pattern_application",
            validation_materialize_seconds,
        )
        errors.extend(train_errors)
        errors.extend(validation_errors)
        live_candidate = _live_candidate_record(
            label=label,
            label_index=label_index,
            evaluation_index=evaluation_count,
            candidate_code=candidate_code,
            candidate_fingerprint=candidate_fingerprint,
            settings=settings,
        )
        if errors:
            candidate_total_seconds = time.perf_counter() - candidate_started_at
            label_timings.add("code_evaluation", candidate_total_seconds)
            candidate_timings.add("code_evaluation", candidate_total_seconds)
            cached_candidate = {
                "average_score": INVALID_CANDIDATE_SCORE,
                "train_score": INVALID_CANDIDATE_SCORE,
                "validation_score": INVALID_CANDIDATE_SCORE,
                "score_details": [],
                "loss_breakdown": [],
                "validation_loss_breakdown": [],
                "timing_summary": _build_timing_summary(
                    candidate_timings,
                    total_wall_seconds=candidate_total_seconds,
                    groups={"evaluation": TIMING_BREAKDOWN_GROUPS["evaluation"]},
                ),
                "errors": errors,
            }
            score_cache[candidate_code] = cached_candidate
            _write_live_candidate_artifact(
                label_dir=label_dir,
                candidate=live_candidate,
                score_details=cached_candidate,
            )
            _append_live_evolution_record(
                path=output_root / "evolution_live.jsonl",
                label=label,
                label_index=label_index,
                candidate=live_candidate,
                score_details=cached_candidate,
            )
            debug_log(
                "patterns.learning.candidate.done",
                label=label,
                worker=worker,
                evaluation_index=evaluation_count,
                candidate_fingerprint=candidate_fingerprint,
                score=INVALID_CANDIDATE_SCORE,
                error_count=len(errors),
            )
            progress_log(
                "patterns.learning.candidate.done",
                label=label,
                worker=worker,
                evaluation_index=evaluation_count,
                candidate_fingerprint=candidate_fingerprint,
                score=INVALID_CANDIDATE_SCORE,
                error_count=len(errors),
            )
            return {"train": {"score": INVALID_CANDIDATE_SCORE}, "val": {"score": INVALID_CANDIDATE_SCORE}}
        train_scoring_started_at = time.perf_counter()
        train_total_score, train_details, train_loss_breakdown = _score_candidate_split(
            label=label,
            prepared_samples=prepared_samples,
            candidate_patterns=train_patterns,
            library=library,
            losses=losses,
            loss_runtime_contexts=train_loss_runtime_contexts,
        )
        train_scoring_seconds = time.perf_counter() - train_scoring_started_at
        label_timings.add("train_scoring", train_scoring_seconds)
        candidate_timings.add("train_scoring", train_scoring_seconds)
        validation_scoring_started_at = time.perf_counter()
        validation_total_score, _, validation_loss_breakdown = _score_candidate_split(
            label=label,
            prepared_samples=prepared_validation_samples,
            candidate_patterns=validation_patterns,
            library=library,
            losses=losses,
            loss_runtime_contexts=validation_loss_runtime_contexts,
        )
        validation_scoring_seconds = time.perf_counter() - validation_scoring_started_at
        label_timings.add("validation_scoring", validation_scoring_seconds)
        candidate_timings.add("validation_scoring", validation_scoring_seconds)
        candidate_total_seconds = time.perf_counter() - candidate_started_at
        label_timings.add("code_evaluation", candidate_total_seconds)
        candidate_timings.add("code_evaluation", candidate_total_seconds)
        cached_candidate = {
            "average_score": train_total_score,
            "train_score": train_total_score,
            "validation_score": validation_total_score,
            "score_details": list(train_details),
            "loss_breakdown": train_loss_breakdown,
            "validation_loss_breakdown": validation_loss_breakdown,
            "timing_summary": _build_timing_summary(
                candidate_timings,
                total_wall_seconds=candidate_total_seconds,
                groups={"evaluation": TIMING_BREAKDOWN_GROUPS["evaluation"]},
            ),
            "errors": errors,
        }
        score_cache[candidate_code] = cached_candidate
        _write_live_candidate_artifact(
            label_dir=label_dir,
            candidate=live_candidate,
            score_details=cached_candidate,
        )
        _append_live_evolution_record(
            path=output_root / "evolution_live.jsonl",
            label=label,
            label_index=label_index,
            candidate=live_candidate,
            score_details=cached_candidate,
        )
        debug_log(
            "patterns.learning.candidate.done",
            label=label,
            worker=worker,
            evaluation_index=evaluation_count,
            candidate_fingerprint=candidate_fingerprint,
            score=train_total_score,
            validation_score=validation_total_score,
            error_count=len(errors),
        )
        progress_log(
            "patterns.learning.candidate.done",
            label=label,
            worker=worker,
            evaluation_index=evaluation_count,
            candidate_fingerprint=candidate_fingerprint,
            score=train_total_score,
            validation_score=validation_total_score,
            error_count=len(errors),
        )
        return {
            "train": {"score": train_total_score},
            "val": {"score": validation_total_score},
        }

    def prompt(*, action: str, settings: Any, config: Any, **context: Any) -> str:
        parent_code = context.get("parent_code")
        parent_a_code = context.get("parent_a_code")
        parent_b_code = context.get("parent_b_code")
        debug_log(
            "patterns.learning.prompt",
            label=label,
            worker=worker,
            action=action,
            step=context.get("step"),
            parent_uid=context.get("parent_uid"),
            parent_a_uid=context.get("parent_a_uid"),
            parent_b_uid=context.get("parent_b_uid"),
            error_category=context.get("error_category"),
        )
        return build_funsearch_prompt(
            environment_id=environment_id,
            label=label,
            label_description=label_description,
            action=action,
            settings=settings,
            config=config,
            input_mode=detector_input_mode,
            parent_code=parent_code,
            parent_metadata=_candidate_prompt_metadata(
                candidate_code=parent_code,
                uid=context.get("parent_uid"),
                origin=context.get("parent_origin"),
                explicit_metadata=context.get("parent_metadata"),
                score=context.get("parent_score"),
                score_cache=score_cache,
            ),
            parent_a_code=parent_a_code,
            parent_b_code=parent_b_code,
            parent_a_metadata=_candidate_prompt_metadata(
                candidate_code=parent_a_code,
                uid=context.get("parent_a_uid"),
                origin=context.get("parent_a_origin"),
                explicit_metadata=context.get("parent_a_metadata"),
                score=context.get("parent_a_score"),
                score_cache=score_cache,
            ),
            parent_b_metadata=_candidate_prompt_metadata(
                candidate_code=parent_b_code,
                uid=context.get("parent_b_uid"),
                origin=context.get("parent_b_origin"),
                explicit_metadata=context.get("parent_b_metadata"),
                score=context.get("parent_b_score"),
                score_cache=score_cache,
            ),
            broken_code=context.get("broken_code"),
            error_message=context.get("error_message"),
            error_category=context.get("error_category"),
            error_line=context.get("error_line"),
            error_column=context.get("error_column"),
            step=context.get("step"),
        )

    algorithm = _construct_funsearch_algorithm(
        seed=seed,
        iteration_count=iteration_count,
        population_size=population_size,
        elite_fraction=elite_fraction,
        children_per_elite=children_per_elite,
        min_candidate_evaluations=min_candidate_evaluations,
        patience_evaluations=patience_evaluations,
        max_candidate_evaluations=max_candidate_evaluations,
        crossover_rate=crossover_rate,
        accept_margin=accept_margin,
        max_repairs=max_repairs,
        evaluator=evaluator,
        prompt=prompt,
        generator_model=generator_model,
        generator_base_url=generator_base_url,
        generator_timeout=generator_timeout,
        generator_retries=generator_retries,
        mutate_temperature=mutate_temperature,
        mutate_max_tokens=mutate_max_tokens,
        timings=label_timings,
        worker=worker,
        label=label,
        label_index=label_index,
        llm_call_recorder=llm_call_recorder,
    )
    result = algorithm.run(
        seed_candidate=_seed_candidate(label, input_mode=detector_input_mode),
        log_root=str(label_dir),
        run_id="funsearch_run",
    )
    payload = dict(result.payload)
    best_candidate = dict(payload["best_candidate"])
    best_detector = _detector_from_candidate(
        label=label,
        label_index=label_index,
        environment_id=environment_id,
        candidate_code=best_candidate["code"],
        metadata={
            "generated_by": "funsearch",
            "label_description": label_description,
            "candidate_uid": best_candidate["uid"],
            "parent_uid": best_candidate.get("parent_uid"),
            "secondary_parent_uid": best_candidate.get("secondary_parent_uid"),
            "origin": best_candidate.get("origin"),
            "score": best_candidate.get("score"),
            "train_score": best_candidate.get("train_score"),
            "val_score": best_candidate.get("val_score"),
            "artifacts": payload.get("artifacts", {}),
        },
        input_mode=detector_input_mode,
    )
    cached_best = score_cache.get(best_candidate["code"], {})
    label_total_seconds = max(time.perf_counter() - label_started_at, 0.0)
    residual_overhead_seconds = max(
        label_total_seconds
        - label_timings.total("code_writing")
        - label_timings.total("code_evaluation"),
        0.0,
    )
    label_timings.add("search_overhead", residual_overhead_seconds)
    return LabelLearningResult(
        label=label,
        label_index=label_index,
        worker=worker,
        detector=best_detector,
        best_score=cached_best.get("average_score"),
        best_train_score=cached_best.get("train_score"),
        best_validation_score=cached_best.get("validation_score"),
        best_score_details=list(cached_best.get("score_details", [])),
        best_loss_breakdown=list(cached_best.get("loss_breakdown", [])),
        best_validation_loss_breakdown=list(
            cached_best.get("validation_loss_breakdown", [])
        ),
        timings=label_timings,
        timing_summary=_build_timing_summary(
            label_timings,
            total_wall_seconds=label_total_seconds,
        ),
        payload=payload,
        artifacts=dict(payload.get("artifacts", {})),
        candidate_scores=score_cache,
        stopped_early=bool(payload.get("stopped_early")),
        stop_reason=(
            str(payload.get("stop_reason"))
            if isinstance(payload.get("stop_reason"), str)
            else None
        ),
        patience_evaluations=(
            int(payload["patience_evaluations"])
            if isinstance(payload.get("patience_evaluations"), int)
            else None
        ),
        candidate_evaluations=(
            int(payload["candidate_evaluations"])
            if isinstance(payload.get("candidate_evaluations"), int)
            else evaluation_count
        ),
    )


def _materialize_candidate_patterns(
    *,
    prepared_samples: list[PreparedLearningSample],
    library: PatternLibrary,
) -> tuple[list[tuple[PreparedLearningSample, list[Any]]], list[str]]:
    materialized: list[tuple[PreparedLearningSample, list[Any]]] = []
    errors: list[str] = []
    for item in prepared_samples:
        try:
            patterns = apply_pattern_library(
                item.trace,
                library,
                existing_evidence=item.existing_evidence,
            )
            materialized.append((item, patterns))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{item.sample.task_id}: {exc}")
            debug_log(
                "patterns.learning.candidate_error",
                task_id=item.sample.task_id,
                error=str(exc),
            )
    return materialized, errors


def _score_candidate_split(
    *,
    label: str,
    prepared_samples: list[PreparedLearningSample],
    candidate_patterns: list[tuple[PreparedLearningSample, list[Any]]],
    library: PatternLibrary,
    losses: list[LossConfig],
    loss_runtime_contexts: list[LossRuntimeContext],
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    total_score = 0.0
    details: list[dict[str, Any]] = []
    loss_breakdown: list[dict[str, Any]] = []
    candidate_traces = [item.trace for item, _ in candidate_patterns]
    candidate_pattern_sets = [patterns for _, patterns in candidate_patterns]
    for loss_index, loss in enumerate(losses):
        runtime_context = loss_runtime_contexts[loss_index]
        if loss.kind == "correlation_structure":
            debug_log(
                "patterns.learning.correlation_start",
                label=label,
                sample_count=len(candidate_traces),
            )
            correlation_value, correlation_meta = compute_correlation_structure_score(
                traces=candidate_traces,
                pattern_sets=candidate_pattern_sets,
                histogram_bins=loss.histogram_bins,
                pair_indices=runtime_context.correlation_pair_indices,
            )
            debug_log(
                "patterns.learning.correlation_done",
                label=label,
                correlation=correlation_value,
                pair_count=correlation_meta.get("pair_count"),
            )
            component = LossComponentScore(
                kind=loss.kind,
                weight=loss.weight,
                raw_score=correlation_value,
                weighted_score=loss.weight * correlation_value,
                metadata={
                    **runtime_context.correlation_metadata,
                    **correlation_meta,
                },
            )
        else:
            raise ValueError(f"Unsupported loss kind: {loss.kind}")
        loss_breakdown.append(_loss_component_payload(component))
        total_score += component.weighted_score
    return total_score, details, loss_breakdown


def _build_loss_runtime_contexts(
    *,
    losses: list[LossConfig],
    sample_count: int,
    seed: int,
    split_name: Literal["train", "validation"],
) -> list[LossRuntimeContext]:
    contexts: list[LossRuntimeContext] = []
    for loss_index, loss in enumerate(losses):
        if loss.kind != "correlation_structure":
            contexts.append(LossRuntimeContext())
            continue
        pair_indices, metadata = select_correlation_pair_indices(
            sample_count=sample_count,
            pair_limit=loss.pair_limit,
            seed=_derive_correlation_pair_seed(
                seed=seed,
                split_name=split_name,
                loss_index=loss_index,
            ),
        )
        contexts.append(
            LossRuntimeContext(
                correlation_pair_indices=pair_indices,
                correlation_metadata={
                    **metadata,
                    "pair_selection_loss_index": loss_index,
                    "pair_selection_split": split_name,
                },
            )
        )
    return contexts


def _derive_correlation_pair_seed(
    *,
    seed: int,
    split_name: Literal["train", "validation"],
    loss_index: int,
) -> int:
    split_offset = 0 if split_name == "train" else 1_000_003
    return seed + split_offset + (loss_index * 10_007)


def _construct_funsearch_algorithm(
    *,
    seed: int,
    iteration_count: int,
    population_size: int,
    elite_fraction: float,
    children_per_elite: int,
    min_candidate_evaluations: int,
    patience_evaluations: int,
    max_candidate_evaluations: int,
    crossover_rate: float,
    accept_margin: float,
    max_repairs: int,
    evaluator: Any,
    prompt: Any,
    generator_model: str,
    generator_base_url: str,
    generator_timeout: int,
    generator_retries: int,
    mutate_temperature: float,
    mutate_max_tokens: int,
    timings: TimingAccumulator,
    worker: int,
    label: str,
    label_index: int,
    llm_call_recorder: LLMCallRecorder | None,
) -> Any:
    try:
        from api import construct_algorithm
        from config import FunSearchConfig
        from lm import LMChatResponse
    except ImportError as exc:
        raise RuntimeError(
            "evo-programming-lib is not installed in the active environment. "
            "Install it with `python -m pip install -e deps/evolutionary-programming-algorithms "
            "--no-deps --no-build-isolation`."
        ) from exc

    class HostLMClient:
        def complete_chat(self, request: Any) -> Any:
            request_started_at = time.perf_counter()
            request_messages = request.normalized_messages()
            progress_log(
                "patterns.learning.lm_request.start",
                worker=worker,
                label=label,
                model=request.model,
                base_url=request.api_base,
            )
            payload = {
                "model": request.model,
                "messages": request_messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": False,
                **dict(request.extra),
            }
            last_error: Exception | None = None
            body: dict[str, Any] | None = None
            for attempt in range(1, generator_retries + 1):
                try:
                    body, resolved_content = request_chat_completion_with_text_retry(
                        base_url=request.api_base,
                        model=request.model,
                        messages=request_messages,
                        timeout=generator_timeout,
                        headers={"Content-Type": "application/json"},
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        stream=False,
                        extra_payload=dict(request.extra),
                        error_prefix="FunSearch LM response",
                        weave_op_name="deepphy.funsearch.chat_completion",
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    progress_log(
                        "patterns.learning.lm_request.retry",
                        worker=worker,
                        label=label,
                        model=request.model,
                        attempt=attempt,
                        max_attempts=generator_retries,
                        error=str(exc),
                    )
                    if attempt >= generator_retries:
                        raise
                    time.sleep(min(2.0, 0.5 * attempt))
            if body is None:
                raise RuntimeError(
                    f"FunSearch LM request failed after {generator_retries} attempts"
                ) from last_error
            choices = body.get("choices")
            router_info = body.get("router") if isinstance(body.get("router"), dict) else {}
            request_duration_seconds = time.perf_counter() - request_started_at
            progress_log(
                "patterns.learning.lm_request.done",
                worker=worker,
                label=label,
                model=body.get("model"),
                chars=len(resolved_content),
                router_connection_id=router_info.get("connection_id"),
                router_base_url=router_info.get("base_url"),
                router_selected_model=router_info.get("selected_model"),
            )
            timings.add("code_writing", request_duration_seconds)
            if llm_call_recorder is not None:
                llm_call_recorder.append(
                    {
                        "label": label,
                        "label_index": label_index,
                        "worker": worker,
                        "request": {
                            "base_url": request.api_base,
                            "path": "/v1/chat/completions",
                            "method": "POST",
                            "headers": {"Content-Type": "application/json"},
                            "body": payload,
                            "messages": request_messages,
                        },
                        "inference": {
                            "model": request.model,
                            "temperature": request.temperature,
                            "max_tokens": request.max_tokens,
                            "timeout_seconds": generator_timeout,
                            "retry_count": generator_retries,
                            "extra": dict(request.extra),
                        },
                        "response": {
                            "model": body.get("model"),
                            "text": resolved_content,
                            "finish_reason": (
                                choices[0].get("finish_reason")
                                if isinstance(choices, list) and choices
                                else None
                            ),
                            "usage": body.get("usage", {}),
                            "router": dict(router_info),
                            "raw": body,
                        },
                        "duration_seconds": request_duration_seconds,
                    }
                )
            return LMChatResponse(
                text=resolved_content,
                raw=body,
                model=body.get("model"),
                finish_reason=choices[0].get("finish_reason") if isinstance(choices, list) and choices else None,
                usage=body.get("usage", {}),
            )

    return construct_algorithm(
        "funsearch",
        config=FunSearchConfig(
            seed=seed,
            iterations=iteration_count,
            population_size=population_size,
            elite_fraction=elite_fraction,
            children_per_elite=children_per_elite,
            min_evaluations_before_patience=min_candidate_evaluations,
            patience_evaluations=patience_evaluations,
            max_candidate_evaluations=max_candidate_evaluations,
            crossover_rate=crossover_rate,
            accept_margin=accept_margin,
            max_repairs=max_repairs,
        ),
        evaluator=evaluator,
        prompt=prompt,
        api_base=generator_base_url,
        model=generator_model,
        lm_client=HostLMClient(),
        mutate_temperature=mutate_temperature,
        mutate_max_tokens=mutate_max_tokens,
    )


def _detector_from_candidate(
    *,
    label: str,
    label_index: int,
    environment_id: str,
    candidate_code: str,
    metadata: dict[str, Any],
    input_mode: Literal["trace", "supervised_evidence"] = "trace",
) -> DetectorDefinition:
    return DetectorDefinition(
        detector_id=f"primitive-{label_index:03d}",
        label=label,
        environment_ids=[environment_id],
        source_code=candidate_code,
        input_mode=input_mode,
        metadata=metadata,
    )


def _seed_candidate(label: str, *, input_mode: Literal["trace", "supervised_evidence"] = "trace") -> str:
    safe_label = label.replace("\\", "\\\\").replace('"', '\\"')
    if input_mode == "supervised_evidence":
        return (
            "def find_pattern(supervised_patterns):\n"
            f"    label = \"{safe_label}\"\n"
            "    return []\n"
        )
    return (
        "def find_pattern(trace):\n"
        f"    label = \"{safe_label}\"\n"
        "    patterns = []\n"
        "    # Emit dicts with label, timestep span, and a parameters dict that includes object_details.\n"
        "    return patterns\n"
    )


def _existing_evidence_for_trace(
    *,
    trace: CanonicalTrace,
    base_library: PatternLibrary | None,
) -> dict[str, Any] | None:
    if base_library is None:
        return None
    patterns = apply_pattern_library(trace, base_library)
    return build_pattern_evidence(
        base_library,
        patterns=patterns,
        metadata={"source": "supervised-library"},
    )


def _write_label_candidate_artifacts(
    *,
    artifact_root: Path,
    label_index: int,
    label_result: LabelLearningResult,
    dataset_id: str | None,
    dataset_version: str | None,
    primitive_source: str | None,
    scorer_config: dict[str, Any],
) -> None:
    label_dir = artifact_root / f"label_{label_index:03d}_{_slugify(label_result.label)}"
    label_dir.mkdir(parents=True, exist_ok=True)
    for existing_candidate_dir in sorted(label_dir.glob("candidate_*")):
        if existing_candidate_dir.is_dir():
            shutil.rmtree(existing_candidate_dir)
    candidates = _collect_candidate_records(label_result.payload)
    best_uid = label_result.payload["best_candidate"]["uid"]
    for candidate_index, candidate in enumerate(candidates, start=1):
        candidate_dir = label_dir / f"candidate_{candidate_index:03d}_{candidate['uid']}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        source_path = candidate_dir / "find_pattern.py"
        source_path.write_text(candidate["code"].rstrip() + "\n", encoding="utf-8")
        score_details = label_result.candidate_scores.get(candidate["code"], {})
        candidate_payload = {
            "label": label_result.label,
            "candidate_index": candidate_index,
            "uid": candidate["uid"],
            "parent_uid": candidate.get("parent_uid"),
            "secondary_parent_uid": candidate.get("secondary_parent_uid"),
            "origin": candidate.get("origin"),
            "iteration": candidate.get("iteration"),
            "score": candidate.get("score"),
            "train_score": candidate.get("train_score"),
            "val_score": candidate.get("val_score"),
            "fingerprint": candidate.get("fingerprint"),
            "metadata": candidate.get("metadata", {}),
            "library_id": f"{_slugify(label_result.label)}-candidate",
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "primitive_source": primitive_source,
            "scorer_config": scorer_config,
            "source_path": str(source_path),
            "is_best_candidate": candidate["uid"] == best_uid,
            "average_score": score_details.get("average_score"),
            "validation_score": score_details.get("validation_score"),
            "loss_breakdown": score_details.get("loss_breakdown", []),
            "validation_loss_breakdown": score_details.get(
                "validation_loss_breakdown", []
            ),
            "timing_summary": score_details.get("timing_summary", {}),
            "score_details": score_details.get("score_details", []),
            "errors": score_details.get("errors", []),
        }
        _write_json(candidate_dir / "candidate.json", candidate_payload)
    _write_json(
        label_dir / "label_result.json",
        {
            "label": label_result.label,
            "label_index": label_result.label_index,
            "worker": label_result.worker,
            "candidate_evaluations": label_result.candidate_evaluations,
            "stopped_early": label_result.stopped_early,
            "stop_reason": label_result.stop_reason,
            "patience_evaluations": label_result.patience_evaluations,
            "detector": label_result.detector.model_dump(mode="json"),
            "best_score": label_result.best_score,
            "best_train_score": label_result.best_train_score,
            "best_validation_score": label_result.best_validation_score,
            "best_score_details": label_result.best_score_details,
            "best_loss_breakdown": label_result.best_loss_breakdown,
            "best_validation_loss_breakdown": label_result.best_validation_loss_breakdown,
            "timing_summary": label_result.timing_summary,
            "timings": {
                "totals": label_result.timings.totals,
                "counts": label_result.timings.counts,
            },
            "artifacts": label_result.artifacts,
            "best_candidate_uid": label_result.payload["best_candidate"]["uid"],
            "run_dir": label_result.artifacts.get("run_dir"),
        },
    )


def _write_run_artifacts(
    *,
    artifact_root: Path,
    library_id: str,
    environment_ids: list[str],
    sample_count: int,
    train_sample_count: int,
    validation_sample_count: int,
    iteration_count: int,
    workers: int,
    primitive_label_count: int,
    average_score: float | None,
    train_score: float | None,
    validation_score: float | None,
    score_details: list[dict[str, Any]],
    dataset_id: str | None,
    dataset_version: str | None,
    primitive_source: str | None,
    scorer_config: dict[str, Any],
    label_summaries: list[dict[str, Any]],
    evolution_record_path: Path,
    loss_breakdown: list[dict[str, Any]],
    timing_summary: dict[str, Any],
    detector_count: int,
    patience_factor: float,
    min_candidate_evaluations: int,
    patience_evaluations: int,
    max_candidate_evaluations: int,
    detector_input_mode: Literal["trace", "supervised_evidence"],
    status: Literal["running", "completed"],
    llm_calls_path: Path,
    configured_workers: int,
    worker_control_path: Path,
    manually_completed_labels: list[str] | None = None,
) -> None:
    current_workers = _read_desired_worker_count(
        worker_control_path,
        fallback=workers,
        max_workers=primitive_label_count,
    )
    resolved_manually_completed_labels = [
        item.strip()
        for item in (manually_completed_labels or [])
        if isinstance(item, str) and item.strip()
    ]
    _write_json(
        artifact_root / "run.json",
        {
            "library_id": library_id,
            "environment_ids": environment_ids,
            "status": status,
            "sample_count": sample_count,
            "train_sample_count": train_sample_count,
            "validation_sample_count": validation_sample_count,
            "iteration_count": iteration_count,
            "workers": current_workers,
            "configured_workers": configured_workers,
            "primitive_label_count": primitive_label_count,
            "completed_label_count": len(label_summaries)
            + len(resolved_manually_completed_labels),
            "manually_completed_labels": resolved_manually_completed_labels,
            "average_score": average_score,
            "train_score": train_score,
            "validation_score": validation_score,
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "primitive_source": primitive_source,
            "scorer_config": scorer_config,
            "detector_count": detector_count,
            "patience_factor": patience_factor,
            "min_candidate_evaluations": min_candidate_evaluations,
            "patience_evaluations": patience_evaluations,
            "no_improvement_patience_evaluations": patience_evaluations,
            "max_candidate_evaluations": max_candidate_evaluations,
            "detector_input_mode": detector_input_mode,
            "llm_calls_path": str(llm_calls_path),
            "worker_control_path": str(worker_control_path),
            "loss_breakdown": loss_breakdown,
            "timing_summary": timing_summary,
            "score_details": score_details,
            "label_runs": label_summaries,
            "evolution_record_path": str(evolution_record_path),
        },
    )


def _append_evolution_record(
    *,
    path: Path,
    label_index: int,
    label: str,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for candidate in _collect_candidate_records(payload):
            record = {
                "label_index": label_index,
                "label": label,
                "candidate_uid": candidate["uid"],
                "parent_uid": candidate.get("parent_uid"),
                "secondary_parent_uid": candidate.get("secondary_parent_uid"),
                "origin": candidate.get("origin"),
                "iteration": candidate.get("iteration"),
                "score": candidate.get("score"),
                "train_score": candidate.get("train_score"),
                "val_score": candidate.get("val_score"),
            }
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def _live_candidate_record(
    *,
    label: str,
    label_index: int,
    evaluation_index: int,
    candidate_code: str,
    candidate_fingerprint: str,
    settings: Any,
) -> dict[str, Any]:
    metadata = _settings_candidate_metadata(settings)
    uid = _coerce_candidate_text(metadata.get("uid")) or f"live_{evaluation_index:03d}_{candidate_fingerprint}"
    return {
        "label": label,
        "label_index": label_index,
        "candidate_index": evaluation_index,
        "uid": uid,
        "parent_uid": _coerce_candidate_text(metadata.get("parent_uid")),
        "secondary_parent_uid": _coerce_candidate_text(metadata.get("secondary_parent_uid")),
        "origin": _coerce_candidate_text(metadata.get("origin")) or "candidate",
        "iteration": _coerce_candidate_int(metadata.get("iteration")),
        "fingerprint": _coerce_candidate_text(metadata.get("fingerprint")) or candidate_fingerprint,
        "metadata": metadata,
        "code": candidate_code,
    }


def _write_live_candidate_artifact(
    *,
    label_dir: Path,
    candidate: dict[str, Any],
    score_details: dict[str, Any],
) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    candidate_slug = _slugify(str(candidate["uid"]))
    candidate_dir = label_dir / f"candidate_{int(candidate['candidate_index']):03d}_{candidate_slug}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    source_path = candidate_dir / "find_pattern.py"
    source_path.write_text(str(candidate["code"]).rstrip() + "\n", encoding="utf-8")
    _write_json(
        candidate_dir / "candidate.json",
        {
            "label": candidate["label"],
            "candidate_index": candidate["candidate_index"],
            "uid": candidate["uid"],
            "parent_uid": candidate.get("parent_uid"),
            "secondary_parent_uid": candidate.get("secondary_parent_uid"),
            "origin": candidate.get("origin"),
            "iteration": candidate.get("iteration"),
            "score": score_details.get("average_score"),
            "train_score": score_details.get("train_score"),
            "val_score": score_details.get("validation_score"),
            "fingerprint": candidate.get("fingerprint"),
            "metadata": candidate.get("metadata", {}),
            "source_path": str(source_path),
            "is_best_candidate": False,
            "average_score": score_details.get("average_score"),
            "validation_score": score_details.get("validation_score"),
            "loss_breakdown": score_details.get("loss_breakdown", []),
            "validation_loss_breakdown": score_details.get(
                "validation_loss_breakdown", []
            ),
            "timing_summary": score_details.get("timing_summary", {}),
            "score_details": score_details.get("score_details", []),
            "errors": score_details.get("errors", []),
        },
    )


def _append_live_evolution_record(
    *,
    path: Path,
    label: str,
    label_index: int,
    candidate: dict[str, Any],
    score_details: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "label_index": label_index,
                    "label": label,
                    "candidate_uid": candidate.get("uid"),
                    "parent_uid": candidate.get("parent_uid"),
                    "secondary_parent_uid": candidate.get("secondary_parent_uid"),
                    "origin": candidate.get("origin"),
                    "iteration": candidate.get("iteration"),
                    "score": score_details.get("average_score"),
                    "train_score": score_details.get("train_score"),
                    "val_score": score_details.get("validation_score"),
                },
                sort_keys=True,
            )
        )
        handle.write("\n")


def _settings_candidate_metadata(settings: Any) -> dict[str, Any]:
    if isinstance(settings, dict):
        return dict(settings)
    if hasattr(settings, "__dict__"):
        raw = getattr(settings, "__dict__")
        if isinstance(raw, dict):
            return dict(raw)
    metadata: dict[str, Any] = {}
    for key in (
        "uid",
        "parent_uid",
        "secondary_parent_uid",
        "origin",
        "iteration",
        "fingerprint",
    ):
        value = getattr(settings, key, None)
        if value is not None:
            metadata[key] = value
    return metadata


def _coerce_candidate_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_candidate_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _collect_candidate_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    records: list[dict[str, Any]] = []

    def maybe_add(candidate: dict[str, Any], *, iteration: int) -> None:
        uid = str(candidate["uid"])
        if uid in seen:
            return
        seen.add(uid)
        records.append({**candidate, "iteration": iteration})

    for candidate in payload.get("initial_candidates", []):
        if isinstance(candidate, dict):
            maybe_add(candidate, iteration=0)
    for iteration_payload in payload.get("iterations", []):
        iteration = int(iteration_payload.get("iteration", 0))
        for candidate in iteration_payload.get("candidates", []):
            if isinstance(candidate, dict):
                maybe_add(candidate, iteration=iteration)
    return records


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_worker_control(
    path: Path,
    *,
    desired_workers: int,
    configured_workers: int,
) -> None:
    payload = _read_run_control_payload(path)
    payload.update(
        {
            "desired_workers": max(1, int(desired_workers)),
            "configured_workers": max(1, int(configured_workers)),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    _write_json(path, payload)


def _read_desired_worker_count(
    path: Path | None,
    *,
    fallback: int,
    max_workers: int,
) -> int:
    resolved_fallback = max(1, min(int(fallback), max_workers))
    payload = _read_run_control_payload(path)
    if not isinstance(payload, dict):
        return resolved_fallback
    desired = payload.get("desired_workers")
    try:
        desired_int = int(desired)
    except (TypeError, ValueError):
        return resolved_fallback
    return max(1, min(desired_int, max_workers))


def _read_run_control_payload(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_manually_completed_label_indices(
    path: Path | None,
    *,
    expected_labels: list[str],
) -> set[int]:
    payload = _read_run_control_payload(path)
    if not payload:
        return set()
    expected_by_name = {
        label.strip(): index
        for index, label in enumerate(expected_labels, start=1)
        if label.strip()
    }
    resolved: set[int] = set()
    raw_indices = payload.get("manually_completed_label_indices")
    if isinstance(raw_indices, list):
        for item in raw_indices:
            if isinstance(item, bool):
                continue
            try:
                index = int(item)
            except (TypeError, ValueError):
                continue
            if 1 <= index <= len(expected_labels):
                resolved.add(index)
    raw_labels = payload.get("manually_completed_labels")
    if isinstance(raw_labels, list):
        for item in raw_labels:
            if not isinstance(item, str):
                continue
            mapped = expected_by_name.get(item.strip())
            if mapped is not None:
                resolved.add(mapped)
    return resolved


def promote_live_best_candidate_to_checkpoint(
    *,
    artifact_root: Path,
    label_name: str,
    label_index: int | None = None,
) -> CompletedLabelCheckpoint:
    run_payload = _load_json(artifact_root / "run.json")
    if not isinstance(run_payload, dict):
        raise ValueError(f"Malformed run.json under {artifact_root}")
    expected_labels = _expected_labels_for_run_payload(run_payload, artifact_root=artifact_root)
    resolved_label_index = _resolve_label_index_for_checkpoint(
        label_name=label_name,
        label_index=label_index,
        expected_labels=expected_labels,
    )
    resolved_label = expected_labels[resolved_label_index - 1]
    label_dir = artifact_root / f"label_{resolved_label_index:03d}_{_slugify(resolved_label)}"
    label_dir.mkdir(parents=True, exist_ok=True)
    if (label_dir / "label_result.json").is_file():
        checkpoints = _load_completed_label_checkpoints(
            artifact_root=artifact_root,
            environment_id=_environment_id_for_run_payload(run_payload, artifact_root=artifact_root),
            expected_labels=expected_labels,
        )
        checkpoint = checkpoints.get(resolved_label_index)
        if checkpoint is None:
            raise ValueError(f"Completed checkpoint not found for {resolved_label}")
        return checkpoint

    candidate_dir, candidate_payload = _best_live_candidate_artifact(label_dir)
    if candidate_dir is None or candidate_payload is None:
        raise ValueError(f"No live candidate artifacts found for label {resolved_label!r}")

    source_path = Path(str(candidate_payload.get("source_path") or candidate_dir / "find_pattern.py"))
    if not source_path.is_file():
        raise ValueError(f"Candidate source file not found for label {resolved_label!r}: {source_path}")
    source_code = source_path.read_text(encoding="utf-8")
    environment_id = _environment_id_for_run_payload(run_payload, artifact_root=artifact_root)
    detector_input_mode = str(run_payload.get("detector_input_mode") or "trace")
    artifacts = _artifacts_for_promoted_label_candidate(label_dir)
    detector = _detector_from_candidate(
        label=resolved_label,
        label_index=resolved_label_index,
        environment_id=environment_id,
        candidate_code=source_code,
        metadata={
            "generated_by": "funsearch",
            "label_description": None,
            "candidate_uid": candidate_payload.get("uid"),
            "parent_uid": candidate_payload.get("parent_uid"),
            "secondary_parent_uid": candidate_payload.get("secondary_parent_uid"),
            "origin": candidate_payload.get("origin"),
            "score": candidate_payload.get("average_score"),
            "train_score": candidate_payload.get("train_score"),
            "val_score": candidate_payload.get("validation_score"),
            "artifacts": artifacts,
        },
        input_mode=(
            detector_input_mode
            if detector_input_mode in {"trace", "supervised_evidence"}
            else "trace"
        ),
    )
    _mark_candidate_artifact_as_best(label_dir=label_dir, chosen_candidate_dir=candidate_dir)
    label_result_payload = {
        "label": resolved_label,
        "label_index": resolved_label_index,
        "worker": int(run_payload.get("workers") or run_payload.get("configured_workers") or 0),
        "candidate_evaluations": _count_live_candidates(label_dir),
        "stopped_early": True,
        "stop_reason": "marked_done",
        "patience_evaluations": None,
        "detector": detector.model_dump(mode="json"),
        "best_score": candidate_payload.get("average_score"),
        "best_train_score": candidate_payload.get("train_score"),
        "best_validation_score": candidate_payload.get("validation_score"),
        "best_score_details": list(candidate_payload.get("score_details") or []),
        "best_loss_breakdown": list(candidate_payload.get("loss_breakdown") or []),
        "best_validation_loss_breakdown": list(candidate_payload.get("validation_loss_breakdown") or []),
        "timing_summary": dict(candidate_payload.get("timing_summary") or {}),
        "timings": {"totals": {}, "counts": {}},
        "artifacts": artifacts,
        "best_candidate_uid": candidate_payload.get("uid"),
        "run_dir": artifacts.get("run_dir"),
    }
    _write_json(label_dir / "label_result.json", label_result_payload)
    checkpoints = _load_completed_label_checkpoints(
        artifact_root=artifact_root,
        environment_id=environment_id,
        expected_labels=expected_labels,
    )
    checkpoint = checkpoints.get(resolved_label_index)
    if checkpoint is None:
        raise ValueError(f"Failed to checkpoint promoted candidate for {resolved_label}")
    _refresh_run_outputs_from_checkpoints(
        artifact_root=artifact_root,
        run_payload=run_payload,
        expected_labels=expected_labels,
        checkpoints=checkpoints,
        clear_manual_label=resolved_label,
    )
    return checkpoint


def _expected_labels_for_run_payload(
    run_payload: dict[str, Any],
    *,
    artifact_root: Path,
) -> list[str]:
    proposed = run_payload.get("proposed_labels")
    if isinstance(proposed, list):
        labels = [str(item).strip() for item in proposed if str(item).strip()]
        if labels:
            return labels
    primitive_source = run_payload.get("primitive_source")
    if isinstance(primitive_source, str) and primitive_source.strip():
        return load_pattern_labels(Path(primitive_source))
    raise ValueError(f"Unable to determine expected labels for run {artifact_root}")


def _resolve_label_index_for_checkpoint(
    *,
    label_name: str,
    label_index: int | None,
    expected_labels: list[str],
) -> int:
    if label_index is not None:
        if 1 <= label_index <= len(expected_labels):
            expected = expected_labels[label_index - 1].strip()
            if label_name.strip() and expected != label_name.strip():
                raise ValueError(
                    f"Label mismatch for index {label_index}: expected {expected!r}, got {label_name.strip()!r}"
                )
            return label_index
        raise ValueError(f"Label index out of range: {label_index}")
    clean_label = label_name.strip()
    if not clean_label:
        raise ValueError("Label name must not be empty.")
    for index, expected in enumerate(expected_labels, start=1):
        if expected.strip() == clean_label:
            return index
    raise ValueError(f"Unknown label: {clean_label}")


def _environment_id_for_run_payload(
    run_payload: dict[str, Any],
    *,
    artifact_root: Path,
) -> str:
    raw_envs = run_payload.get("environment_ids")
    if isinstance(raw_envs, list):
        for item in raw_envs:
            text = str(item).strip()
            if text:
                return text
    if artifact_root.parent.name:
        return artifact_root.parent.name
    raise ValueError(f"Unable to determine environment id for run {artifact_root}")


def _best_live_candidate_artifact(label_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    explicit: tuple[Path, dict[str, Any]] | None = None
    ranked: list[tuple[float, str, Path, dict[str, Any]]] = []
    for candidate_dir in sorted(path for path in label_dir.glob("candidate_*") if path.is_dir()):
        payload = _load_json(candidate_dir / "candidate.json")
        if not isinstance(payload, dict):
            continue
        if bool(payload.get("is_best_candidate")) and explicit is None:
            explicit = (candidate_dir, payload)
        average_score = payload.get("average_score")
        score = float(average_score) if isinstance(average_score, (int, float)) else float("-inf")
        ranked.append((score, candidate_dir.name, candidate_dir, payload))
    if explicit is not None:
        return explicit
    if not ranked:
        return None, None
    _score, _name, candidate_dir, payload = sorted(ranked, reverse=True)[0]
    return candidate_dir, payload


def _count_live_candidates(label_dir: Path) -> int:
    return sum(1 for path in label_dir.glob("candidate_*") if path.is_dir())


def _artifacts_for_promoted_label_candidate(label_dir: Path) -> dict[str, Any]:
    run_dir = next(
        (str(path) for path in sorted(label_dir.glob("funsearch_run*")) if path.is_dir()),
        None,
    )
    return {
        "log_root": str(label_dir),
        "run_dir": run_dir,
    }


def _mark_candidate_artifact_as_best(*, label_dir: Path, chosen_candidate_dir: Path) -> None:
    for candidate_dir in sorted(path for path in label_dir.glob("candidate_*") if path.is_dir()):
        payload = _load_json(candidate_dir / "candidate.json")
        if not isinstance(payload, dict):
            continue
        payload["is_best_candidate"] = candidate_dir == chosen_candidate_dir
        _write_json(candidate_dir / "candidate.json", payload)


def _refresh_run_outputs_from_checkpoints(
    *,
    artifact_root: Path,
    run_payload: dict[str, Any],
    expected_labels: list[str],
    checkpoints: dict[int, CompletedLabelCheckpoint],
    clear_manual_label: str | None = None,
) -> None:
    label_summaries = [
        _label_summary_payload(artifact_root=artifact_root, checkpoint=item)
        for _, item in sorted(checkpoints.items())
    ]
    score_details = [
        detail
        for _, checkpoint in sorted(checkpoints.items())
        for detail in checkpoint.best_score_details
    ]
    loss_breakdown = _aggregate_label_loss_breakdown(label_summaries)
    train_score = _aggregate_label_metric(label_summaries, "train_score")
    validation_score = _aggregate_label_metric(label_summaries, "validation_score")
    average_score = train_score
    manual_labels = [
        item.strip()
        for item in (run_payload.get("manually_completed_labels") or [])
        if isinstance(item, str) and item.strip()
    ]
    if clear_manual_label is not None:
        manual_labels = [item for item in manual_labels if item != clear_manual_label]
    timing_summary = (
        dict(run_payload.get("timing_summary"))
        if isinstance(run_payload.get("timing_summary"), dict)
        else {"total_wall_seconds": 0.0, "groups": {}}
    )
    library = PatternLibrary(
        library_id=str(run_payload.get("library_id") or artifact_root.name),
        environment_ids=list(run_payload.get("environment_ids") or [_environment_id_for_run_payload(run_payload, artifact_root=artifact_root)]),
        primitive_source=(
            str(run_payload.get("primitive_source"))
            if isinstance(run_payload.get("primitive_source"), str)
            else None
        ),
        scorer_config=(
            dict(run_payload.get("scorer_config"))
            if isinstance(run_payload.get("scorer_config"), dict)
            else {}
        ),
        detectors=[checkpoint.detector for _, checkpoint in sorted(checkpoints.items())],
        metadata={
            "iteration_count": run_payload.get("iteration_count"),
            "workers": run_payload.get("workers"),
            "configured_workers": run_payload.get("configured_workers"),
            "manually_completed_labels": manual_labels,
            "label_runs": label_summaries,
        },
    )
    save_pattern_library(library, artifact_root / "pattern_library.json")
    _write_run_artifacts(
        artifact_root=artifact_root,
        library_id=library.library_id,
        environment_ids=list(library.environment_ids),
        sample_count=int(run_payload.get("sample_count") or 0),
        train_sample_count=int(run_payload.get("train_sample_count") or 0),
        validation_sample_count=int(run_payload.get("validation_sample_count") or 0),
        iteration_count=int(run_payload.get("iteration_count") or 1),
        workers=int(run_payload.get("workers") or run_payload.get("configured_workers") or 1),
        primitive_label_count=len(expected_labels),
        average_score=average_score,
        train_score=train_score,
        validation_score=validation_score,
        score_details=score_details,
        dataset_id=run_payload.get("dataset_id"),
        dataset_version=run_payload.get("dataset_version"),
        primitive_source=library.primitive_source,
        scorer_config=library.scorer_config,
        label_summaries=label_summaries,
        evolution_record_path=Path(str(run_payload.get("evolution_record_path") or artifact_root / "evolution_record.jsonl")),
        loss_breakdown=loss_breakdown,
        timing_summary=timing_summary,
        detector_count=len(library.detectors),
        patience_factor=float(run_payload.get("patience_factor") or 0.35),
        min_candidate_evaluations=int(
            run_payload.get("min_candidate_evaluations") or 100
        ),
        patience_evaluations=(
            int(run_payload["no_improvement_patience_evaluations"])
            if isinstance(run_payload.get("no_improvement_patience_evaluations"), int)
            else (
                int(run_payload["patience_evaluations"])
                if isinstance(run_payload.get("patience_evaluations"), int)
                else 15
            )
        ),
        max_candidate_evaluations=int(
            run_payload.get("max_candidate_evaluations") or 250
        ),
        detector_input_mode=str(run_payload.get("detector_input_mode") or "trace"),
        status="running",
        llm_calls_path=Path(str(run_payload.get("llm_calls_path") or artifact_root / "llm_calls.jsonl")),
        configured_workers=int(run_payload.get("configured_workers") or run_payload.get("workers") or 1),
        worker_control_path=Path(str(run_payload.get("worker_control_path") or artifact_root / RUN_CONTROL_FILENAME)),
        manually_completed_labels=manual_labels,
    )


def _loss_component_payload(component: LossComponentScore) -> dict[str, Any]:
    return {
        "kind": component.kind,
        "weight": component.weight,
        "raw_score": component.raw_score,
        "weighted_score": component.weighted_score,
        "metadata": component.metadata,
    }


def _build_timing_summary(
    timings: TimingAccumulator,
    *,
    total_wall_seconds: float,
    groups: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    resolved_groups = groups or TIMING_BREAKDOWN_GROUPS
    group_payloads: dict[str, list[dict[str, Any]]] = {}
    for group_name, keys in resolved_groups.items():
        entries: list[dict[str, Any]] = []
        for key in keys:
            meta = TIMING_BREAKDOWN_META[key]
            count = timings.count(key)
            total_seconds = timings.total(key)
            if count <= 0 and total_seconds <= 0:
                continue
            entries.append(
                {
                    "key": key,
                    "label": meta["label"],
                    "unit": meta["unit"],
                    "group": group_name,
                    "order": meta["order"],
                    "count": count,
                    "total_seconds": total_seconds,
                    "average_seconds": (total_seconds / count) if count > 0 else None,
                    "share_of_wall_time": (
                        total_seconds / total_wall_seconds
                        if total_wall_seconds > 0
                        else None
                    ),
                }
            )
        group_payloads[group_name] = sorted(
            entries,
            key=lambda item: (int(item["order"]), str(item["label"])),
        )
    return {
        "total_wall_seconds": total_wall_seconds,
        "groups": group_payloads,
    }


def _label_summary_payload(
    *,
    artifact_root: Path,
    checkpoint: CompletedLabelCheckpoint,
) -> dict[str, Any]:
    return {
        "label": checkpoint.label,
        "best_candidate_uid": checkpoint.detector.metadata.get("candidate_uid"),
        "best_score": checkpoint.best_score,
        "train_score": checkpoint.best_train_score,
        "validation_score": checkpoint.best_validation_score,
        "loss_breakdown": checkpoint.best_loss_breakdown,
        "validation_loss_breakdown": checkpoint.best_validation_loss_breakdown,
        "timing_summary": checkpoint.timing_summary,
        "worker": checkpoint.worker,
        "candidate_evaluations": checkpoint.candidate_evaluations,
        "stopped_early": checkpoint.stopped_early,
        "stop_reason": checkpoint.stop_reason,
        "patience_evaluations": checkpoint.patience_evaluations,
        "artifact_dir": str(
            artifact_root / f"label_{checkpoint.label_index:03d}_{_slugify(checkpoint.label)}"
        ),
        "run_dir": checkpoint.artifacts.get("run_dir"),
    }


def _completed_label_checkpoint_from_result(
    label_result: LabelLearningResult,
) -> CompletedLabelCheckpoint:
    return CompletedLabelCheckpoint(
        label=label_result.label,
        label_index=label_result.label_index,
        worker=label_result.worker,
        detector=label_result.detector,
        best_score=label_result.best_score,
        best_train_score=label_result.best_train_score,
        best_validation_score=label_result.best_validation_score,
        best_score_details=list(label_result.best_score_details),
        best_loss_breakdown=list(label_result.best_loss_breakdown),
        best_validation_loss_breakdown=list(
            label_result.best_validation_loss_breakdown
        ),
        timing_summary=dict(label_result.timing_summary),
        timings=label_result.timings,
        artifacts=dict(label_result.artifacts),
        stopped_early=label_result.stopped_early,
        stop_reason=label_result.stop_reason,
        patience_evaluations=label_result.patience_evaluations,
        candidate_evaluations=label_result.candidate_evaluations,
    )


def _load_completed_label_checkpoints(
    *,
    artifact_root: Path,
    environment_id: str,
    expected_labels: list[str],
) -> dict[int, CompletedLabelCheckpoint]:
    expected_by_index = {
        index: label.strip() for index, label in enumerate(expected_labels, start=1)
    }
    checkpoints: dict[int, CompletedLabelCheckpoint] = {}
    for label_result_path in sorted(artifact_root.glob("label_*/label_result.json")):
        payload = _load_json(label_result_path)
        if not isinstance(payload, dict):
            continue
        label_index = payload.get("label_index")
        label = payload.get("label")
        detector_payload = payload.get("detector")
        if not isinstance(label_index, int) or not isinstance(label, str):
            continue
        expected_label = expected_by_index.get(label_index)
        if expected_label is None or expected_label != label.strip():
            raise ValueError(
                f"Resume checkpoint label mismatch at {label_result_path}: expected "
                f"{expected_label!r}, found {label!r}"
            )
        if not isinstance(detector_payload, dict):
            continue
        detector = DetectorDefinition.model_validate(detector_payload)
        if environment_id not in detector.environment_ids:
            detector.environment_ids = [environment_id]
        checkpoints[label_index] = CompletedLabelCheckpoint(
            label=label.strip(),
            label_index=label_index,
            worker=int(payload.get("worker") or 0),
            detector=detector,
            best_score=_coerce_optional_float(payload.get("best_score")),
            best_train_score=_coerce_optional_float(payload.get("best_train_score")),
            best_validation_score=_coerce_optional_float(
                payload.get("best_validation_score")
            ),
            best_score_details=_dict_list(payload.get("best_score_details")),
            best_loss_breakdown=_dict_list(payload.get("best_loss_breakdown")),
            best_validation_loss_breakdown=_dict_list(
                payload.get("best_validation_loss_breakdown")
            ),
            timing_summary=dict(payload.get("timing_summary") or {}),
            timings=_timings_from_payload(payload.get("timings")),
            artifacts=dict(payload.get("artifacts") or {}),
            stopped_early=bool(payload.get("stopped_early")),
            stop_reason=(
                str(payload.get("stop_reason"))
                if isinstance(payload.get("stop_reason"), str)
                else None
            ),
            patience_evaluations=(
                int(payload["patience_evaluations"])
                if isinstance(payload.get("patience_evaluations"), int)
                else None
            ),
            candidate_evaluations=(
                int(payload["candidate_evaluations"])
                if isinstance(payload.get("candidate_evaluations"), int)
                else None
            ),
        )
    return checkpoints


def _timings_from_payload(payload: Any) -> TimingAccumulator:
    if not isinstance(payload, dict):
        return TimingAccumulator()
    accumulator = TimingAccumulator()
    totals = payload.get("totals")
    counts = payload.get("counts")
    if isinstance(totals, dict):
        for key, value in totals.items():
            if isinstance(key, str) and isinstance(value, (int, float)):
                accumulator.totals[key] = float(value)
    if isinstance(counts, dict):
        for key, value in counts.items():
            if isinstance(key, str) and isinstance(value, int):
                accumulator.counts[key] = value
    return accumulator


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _candidate_prompt_metadata(
    *,
    candidate_code: str | None,
    uid: str | None,
    origin: str | None,
    explicit_metadata: dict[str, Any] | None,
    score: float | None,
    score_cache: dict[str, dict[str, Any]],
) -> FunSearchCandidateSummary | None:
    if candidate_code is None and uid is None and explicit_metadata is None:
        return None
    cached = score_cache.get(candidate_code or "", {})
    metadata = dict(explicit_metadata or {})
    evaluation = metadata.get("evaluation", {}) if isinstance(metadata.get("evaluation"), dict) else {}
    return FunSearchCandidateSummary(
        uid=uid,
        origin=origin,
        score=score,
        train_score=evaluation.get("train_score"),
        val_score=evaluation.get("val_score"),
        average_score=cached.get("average_score"),
        loss_breakdown=list(cached.get("loss_breakdown", [])),
        errors=list(cached.get("errors", [])),
    )


def _aggregate_label_loss_breakdown(label_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accum: dict[str, dict[str, Any]] = {}
    for summary in label_summaries:
        for component in summary.get("loss_breakdown", []):
            bucket = accum.setdefault(
                component["kind"],
                {
                    "kind": component["kind"],
                    "weight": component["weight"],
                    "raw_scores": [],
                    "weighted_scores": [],
                },
            )
            bucket["raw_scores"].append(component["raw_score"])
            bucket["weighted_scores"].append(component["weighted_score"])
    output: list[dict[str, Any]] = []
    for kind, bucket in sorted(accum.items()):
        output.append(
            {
                "kind": kind,
                "weight": bucket["weight"],
                "mean_raw_score": sum(bucket["raw_scores"]) / len(bucket["raw_scores"]),
                "mean_weighted_score": sum(bucket["weighted_scores"]) / len(bucket["weighted_scores"]),
            }
        )
    return output


def _aggregate_label_metric(
    label_summaries: list[dict[str, Any]], key: str
) -> float | None:
    values = [
        float(summary[key])
        for summary in label_summaries
        if isinstance(summary.get(key), (int, float))
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _trace_summary(trace: CanonicalTrace) -> str:
    metrics = ", ".join(f"{key}={value}" for key, value in _trace_metrics(trace).items()) or "none"
    events = ", ".join(event.kind for event in trace.events[:5]) or "no events"
    return (
        f"timesteps={len(trace.timesteps)}; "
        f"events={len(trace.events)} ({events}); "
        f"terminal_status={trace.outcome.terminal_status}; "
        f"success={trace.outcome.success}; "
        f"metrics={metrics}"
    )


def _trace_metrics(trace: CanonicalTrace) -> dict[str, Any]:
    return {
        key: value
        for key, value in sorted(trace.outcome.metrics.items())
        if isinstance(value, (str, int, float, bool))
    }


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "label"
