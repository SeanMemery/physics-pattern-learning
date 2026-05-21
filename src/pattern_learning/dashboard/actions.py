from __future__ import annotations

import json
from pathlib import Path
import os
from typing import Any, Literal

from deepphy_eval import default_benchmark_task_limit, default_run_id
from llm_config import load_shared_llm_config
from paths import get_artifact_layout
from pattern_library_paths import (
    default_pattern_evaluation_root,
    default_pattern_output_path,
    resolve_pattern_library_path,
    resolve_pattern_run_dir,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator


_SHARED_LLM = load_shared_llm_config()
AUTOMATIC_LEARNING_ACTION = "automatic-launch"
UNSUPERVISED_LEARNING_ACTION = "unsupervised-launch"

LIBRARY_HOLDOUT_EVALUATION_TEMPLATES: dict[str, dict[str, Any]] = {
    "quick-eval": {
        "label": "Quick eval holdouts",
        "benchmark_preset": "quick-eval",
        "seed_count": 2,
        "holdout_sample_count": 2,
    }
}
DEFAULT_LIBRARY_HOLDOUT_EVALUATION_TEMPLATE = "quick-eval"


def _append_model_arg(argv: list[str], model: str) -> None:
    value = str(model or "").strip()
    if value:
        argv.extend(["--model", value])


def _default_llm_api_key() -> str:
    return str(os.environ.get("LLM_ROUTER_API_KEY", "") or "").strip()


def python_executable_for_env(env_id: str) -> Path:
    path = get_artifact_layout().project_root / "venv" / env_id / "bin" / "python"
    if not path.is_file():
        raise FileNotFoundError(
            f"Environment Python not found for {env_id}: {path}"
        )
    return path


class DatasetBuildActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: Literal["phyre", "iphyre", "kinetix", "pooltool"]
    output_dir: Path
    dataset_name: str = Field(min_length=1)
    train_count: int = Field(default=1000, ge=1)
    validation_count: int = Field(default=100, ge=1)
    test_count: int = Field(default=100, ge=1)
    train_offset: int = Field(default=0, ge=0)
    validation_offset: int = Field(default=500000, ge=0)
    test_offset: int = Field(default=1000000, ge=0)
    split: list[str] = Field(default_factory=list)
    force: bool = False


class PatternLearningActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: str = Field(min_length=1)
    config_path: Path | None = None
    labels_path: Path | None = None
    output_path: Path | None = None
    dataset_manifest: Path | None = None
    split: list[str] = Field(default_factory=list)
    validation_split: list[str] = Field(default_factory=list)
    limit: int | None = Field(default=None, ge=1)
    model: str = ""
    iteration_count: int = Field(default=8, ge=1)
    workers: int = Field(default=1, ge=1)
    seed: int | None = None
    population_size: int | None = Field(default=64, ge=1)
    elite_fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    children_per_elite: int | None = Field(default=None, ge=0)
    patience_factor: float | None = Field(default=0.35, gt=0.0, le=1.0)
    crossover_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    run_timestamp: str | None = None

    @model_validator(mode="after")
    def _validate_split_inputs(self) -> "PatternLearningActionRequest":
        overlap = sorted(set(self.split) & set(self.validation_split))
        if overlap:
            raise ValueError(
                "split and validation_split must not overlap: " + ", ".join(overlap)
            )
        if self.env_id == "pooltool":
            self.workers = 1
        return self


class DeepPHYEvaluationActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: Literal["phyre", "iphyre", "kinetix", "pooltool"]
    benchmark_preset: Literal["full", "quick-eval", "test"] = "full"
    task_limit: int | None = Field(default=None, ge=1)
    benchmark_seed: int | None = None
    workers: int = Field(default=1, ge=1)
    prompt_format: Literal["VLA", "WM"] = "WM"
    model: str = ""
    description_mode: Literal[
        "trace_text", "pattern_text", "trace_text_plus_patterns"
    ] = "pattern_text"
    pattern_run_dir: Path | None = None
    pattern_library: Path | None = None
    evidence_relpath: str | None = None
    ollama_base_url: str = _SHARED_LLM.base_url
    llm_api_key: str = Field(default_factory=_default_llm_api_key)
    model_timeout: int = Field(default=_SHARED_LLM.timeout, ge=0)
    capture_visuals: bool = False
    run_id: str | None = None
    output_dir: Path | None = None
    max_attempts: int | None = Field(default=None, ge=1)
    max_steps: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_pattern_inputs(self) -> "DeepPHYEvaluationActionRequest":
        self.prompt_format = "WM"
        if self.pattern_run_dir is not None and self.pattern_library is not None:
            raise ValueError(
                "pattern_run_dir and pattern_library are mutually exclusive"
            )
        if self.env_id == "pooltool":
            # PoolTool rendering goes through Panda3D, which installs signal handlers
            # and is not safe inside the benchmark thread pool. Force single-worker evals.
            self.workers = 1
        if self.task_limit is None:
            self.task_limit = default_benchmark_task_limit(
                self.env_id,
                preset=self.benchmark_preset,
            )
        return self


class LibraryHoldoutEvaluationActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: Literal["phyre", "iphyre", "kinetix", "pooltool"]
    pattern_run_dir: Path
    template_name: Literal["quick-eval"] = DEFAULT_LIBRARY_HOLDOUT_EVALUATION_TEMPLATE
    workers: int = Field(default=1, ge=1)
    model: str = ""
    ollama_base_url: str = _SHARED_LLM.base_url
    model_timeout: int = Field(default=_SHARED_LLM.timeout, ge=0)
    run_id: str | None = None
    benchmark_preset: Literal["quick-eval"] | None = None
    seed_count: int | None = Field(default=None, ge=1)
    holdout_sample_count: int | None = Field(default=None, ge=1)
    single_holdout_repeat_count: int | None = Field(default=None, ge=1)
    pair_holdout_size: int | None = Field(default=None, ge=1)
    pair_holdout_sample_count: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _apply_template_defaults(self) -> "LibraryHoldoutEvaluationActionRequest":
        template = LIBRARY_HOLDOUT_EVALUATION_TEMPLATES[self.template_name]
        self.pattern_run_dir = resolve_pattern_run_dir(self.pattern_run_dir)
        self.benchmark_preset = "quick-eval"
        if self.seed_count is None:
            self.seed_count = int(template["seed_count"])
        if self.holdout_sample_count is None:
            legacy_counts = [
                int(value)
                for value in (
                    self.single_holdout_repeat_count,
                    self.pair_holdout_sample_count,
                )
                if value is not None
            ]
            if legacy_counts:
                self.holdout_sample_count = max(legacy_counts)
            else:
                self.holdout_sample_count = int(template["holdout_sample_count"])
        if self.single_holdout_repeat_count is None:
            self.single_holdout_repeat_count = int(self.holdout_sample_count)
        if self.pair_holdout_size is None:
            self.pair_holdout_size = 2
        if self.pair_holdout_sample_count is None:
            self.pair_holdout_sample_count = int(self.holdout_sample_count)
        return self


class ContinuousLibraryHoldoutTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: Literal["phyre", "iphyre", "kinetix", "pooltool"]
    pattern_run_dir: Path
    run_id: str | None = None

    @model_validator(mode="after")
    def _normalize_target(self) -> "ContinuousLibraryHoldoutTarget":
        self.pattern_run_dir = resolve_pattern_run_dir(self.pattern_run_dir)
        return self


class ContinuousLibraryHoldoutLoopActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    targets: list[ContinuousLibraryHoldoutTarget] = Field(min_length=1)
    template_name: Literal["quick-eval"] = DEFAULT_LIBRARY_HOLDOUT_EVALUATION_TEMPLATE
    workers: int = Field(default=1, ge=1)
    model: str = ""
    ollama_base_url: str = _SHARED_LLM.base_url
    model_timeout: int = Field(default=_SHARED_LLM.timeout, ge=0)
    baseline_interval: int = Field(default=4, ge=0)
    sleep_seconds: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _apply_template_defaults(self) -> "ContinuousLibraryHoldoutLoopActionRequest":
        if any(target.env_id == "pooltool" for target in self.targets):
            self.workers = 1
        return self


class AutomaticLearningActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: str = Field(min_length=1)
    source_supervised_run_dir: Path | None = None
    source_pattern_library: Path | None = None
    source_evaluation_run_id: str = Field(min_length=1)
    reasoning_trace_count: int = Field(default=6, ge=1)
    proposed_label_count: int = Field(default=10, ge=1)
    split: list[str] = Field(default_factory=list)
    validation_split: list[str] = Field(default_factory=list)
    limit: int | None = Field(default=None, ge=1)
    model: str = ""
    iteration_count: int = Field(default=8, ge=1)
    workers: int = Field(default=1, ge=1)
    seed: int | None = None
    population_size: int | None = Field(default=64, ge=1)
    elite_fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    children_per_elite: int | None = Field(default=None, ge=0)
    patience_factor: float | None = Field(default=0.35, gt=0.0, le=1.0)
    crossover_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    run_timestamp: str | None = None

    @model_validator(mode="after")
    def _validate_source_inputs(self) -> "AutomaticLearningActionRequest":
        if self.source_supervised_run_dir is None and self.source_pattern_library is None:
            raise ValueError(
                "One of source_supervised_run_dir or source_pattern_library is required"
            )
        if self.source_supervised_run_dir is not None and self.source_pattern_library is not None:
            raise ValueError(
                "source_supervised_run_dir and source_pattern_library are mutually exclusive"
            )
        overlap = sorted(set(self.split) & set(self.validation_split))
        if overlap:
            raise ValueError(
                "split and validation_split must not overlap: " + ", ".join(overlap)
            )
        if self.env_id == "pooltool":
            self.workers = 1
        return self


class UnsupervisedLearningActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: str = Field(min_length=1)
    dataset_manifest: Path
    split: list[str] = Field(default_factory=list)
    validation_split: list[str] = Field(default_factory=list)
    limit: int | None = Field(default=None, ge=1)
    model: str = ""
    iteration_count: int = Field(default=8, ge=1)
    workers: int = Field(default=1, ge=1)
    seed: int | None = None
    population_size: int | None = Field(default=64, ge=1)
    elite_fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    children_per_elite: int | None = Field(default=None, ge=0)
    patience_factor: float | None = Field(default=0.35, gt=0.0, le=1.0)
    crossover_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    trace_count: int = Field(default=10, ge=1)
    labels_per_trace: int = Field(default=10, ge=1)
    final_label_count: int = Field(default=10, ge=1)
    run_timestamp: str | None = None

    @model_validator(mode="after")
    def _validate_split_inputs(self) -> "UnsupervisedLearningActionRequest":
        overlap = sorted(set(self.split) & set(self.validation_split))
        if overlap:
            raise ValueError(
                "split and validation_split must not overlap: " + ", ".join(overlap)
            )
        if self.final_label_count > (self.trace_count * self.labels_per_trace):
            raise ValueError(
                "final_label_count must be less than or equal to trace_count * labels_per_trace"
            )
        if self.env_id == "pooltool":
            self.workers = 1
        return self


class PooltoolRewardOptimizationActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: Literal["pooltool"] = "pooltool"
    source_run_dir: Path
    goal_text: str = ""
    dsl_program: str = ""
    model: str = ""
    scene_count: int = Field(default=10, ge=1)
    seed_count: int = Field(default=5, ge=1)
    budgets: list[int] = Field(default_factory=lambda: [10, 50, 100, 250, 500, 1000])
    rack_jitter: float = Field(default=0.12, ge=0.0)
    ollama_base_url: str = _SHARED_LLM.base_url
    model_timeout: int = Field(default=_SHARED_LLM.timeout, ge=0)
    run_id: str | None = None

    @model_validator(mode="after")
    def _validate_goal_or_dsl(self) -> "PooltoolRewardOptimizationActionRequest":
        self.source_run_dir = resolve_pattern_run_dir(self.source_run_dir)
        if not self.goal_text.strip() and not self.dsl_program.strip():
            raise ValueError("One of goal_text or dsl_program is required")
        normalized_budgets = sorted({int(item) for item in self.budgets if int(item) > 0})
        if not normalized_budgets:
            raise ValueError("budgets must contain at least one positive integer")
        self.budgets = normalized_budgets
        return self


class PooltoolRewardValueModelActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: Literal["pooltool"] = "pooltool"
    source_run_dir: Path
    goal_text: str = ""
    dsl_program: str = ""
    model: str = ""
    train_scene_count: int = Field(default=64, ge=1)
    train_actions_per_scene: int = Field(default=32, ge=1)
    validation_scene_count: int = Field(default=12, ge=1)
    validation_candidates_per_scene: int = Field(default=128, ge=1)
    epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=64, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    image_size: int = Field(default=256, ge=32)
    rack_jitter: float = Field(default=0.12, ge=0.0)
    ollama_base_url: str = _SHARED_LLM.base_url
    model_timeout: int = Field(default=_SHARED_LLM.timeout, ge=0)
    run_id: str | None = None

    @model_validator(mode="after")
    def _validate_goal_or_dsl(self) -> "PooltoolRewardValueModelActionRequest":
        self.source_run_dir = resolve_pattern_run_dir(self.source_run_dir)
        if not self.goal_text.strip() and not self.dsl_program.strip():
            raise ValueError("One of goal_text or dsl_program is required")
        return self


def default_pattern_learning_output_path(request: PatternLearningActionRequest) -> Path:
    return default_pattern_output_path(request.env_id, timestamp=request.run_timestamp)


def resolve_deepphy_paths(
    request: DeepPHYEvaluationActionRequest,
) -> tuple[Path | None, Path | None, str]:
    chosen_run_id = request.run_id or default_run_id()
    if request.pattern_run_dir is not None:
        run_dir = resolve_pattern_run_dir(request.pattern_run_dir)
        report_root = default_pattern_evaluation_root(
            pattern_run_dir=run_dir, evaluation_run_id=chosen_run_id
        )
        report_dir = request.output_dir or report_root
        return run_dir, report_dir, chosen_run_id
    if request.output_dir is not None:
        return None, request.output_dir, chosen_run_id
    return None, None, chosen_run_id


def build_pattern_learning_argv(request: PatternLearningActionRequest) -> list[str]:
    argv = ["-m", "cli", "pattern-learn", "--env", request.env_id]

    if request.config_path is not None:
        argv.extend(["--config", str(request.config_path)])
    if request.labels_path is not None:
        argv.extend(["--labels-file", str(request.labels_path)])

    output_path = request.output_path or default_pattern_learning_output_path(request)
    argv.extend(["--output-path", str(output_path)])

    if request.dataset_manifest is not None:
        argv.extend(["--dataset-manifest", str(request.dataset_manifest)])
    argv.extend(["--generator-model", request.model])

    for split_name in request.split:
        argv.extend(["--split", split_name])
    for split_name in request.validation_split:
        argv.extend(["--validation-split", split_name])

    if request.limit is not None:
        argv.extend(["--limit", str(request.limit)])
    if request.iteration_count != 8:
        argv.extend(["--iteration-count", str(request.iteration_count)])
    if request.workers != 1:
        argv.extend(["--workers", str(request.workers)])
    if request.seed is not None:
        argv.extend(["--seed", str(request.seed)])
    if request.population_size is not None:
        argv.extend(["--population-size", str(request.population_size)])
    if request.elite_fraction is not None:
        argv.extend(["--elite-fraction", str(request.elite_fraction)])
    if request.children_per_elite is not None:
        argv.extend(["--children-per-elite", str(request.children_per_elite)])
    if request.patience_factor is not None and request.patience_factor != 0.35:
        argv.extend(["--patience-factor", str(request.patience_factor)])
    if request.crossover_rate is not None:
        argv.extend(["--crossover-rate", str(request.crossover_rate)])

    return argv


def pattern_run_dir_for_output_path(output_path: Path) -> Path:
    # Mirrors `cli.pattern-learn` effective_artifact_root derivation.
    if output_path.name == "pattern_library.json":
        return output_path.parent
    return output_path.parent / output_path.stem


def build_supervised_learning_with_auto_eval_argv(
    request: PatternLearningActionRequest,
) -> list[str]:
    argv = [
        "-m",
        "dashboard.workflows",
        "supervised-learn-and-evaluate",
        "--env",
        request.env_id,
    ]

    if request.config_path is not None:
        argv.extend(["--config", str(request.config_path)])
    if request.labels_path is not None:
        argv.extend(["--labels-file", str(request.labels_path)])

    output_path = request.output_path or default_pattern_learning_output_path(request)
    argv.extend(["--output-path", str(output_path)])

    if request.dataset_manifest is not None:
        argv.extend(["--dataset-manifest", str(request.dataset_manifest)])
    _append_model_arg(argv, request.model)

    for split_name in request.split:
        argv.extend(["--split", split_name])
    for split_name in request.validation_split:
        argv.extend(["--validation-split", split_name])

    if request.limit is not None:
        argv.extend(["--limit", str(request.limit)])
    if request.iteration_count != 8:
        argv.extend(["--iteration-count", str(request.iteration_count)])
    if request.workers != 1:
        argv.extend(["--workers", str(request.workers)])
    if request.seed is not None:
        argv.extend(["--seed", str(request.seed)])
    if request.population_size is not None:
        argv.extend(["--population-size", str(request.population_size)])
    if request.elite_fraction is not None:
        argv.extend(["--elite-fraction", str(request.elite_fraction)])
    if request.children_per_elite is not None:
        argv.extend(["--children-per-elite", str(request.children_per_elite)])
    if request.patience_factor is not None and request.patience_factor != 0.35:
        argv.extend(["--patience-factor", str(request.patience_factor)])
    if request.crossover_rate is not None:
        argv.extend(["--crossover-rate", str(request.crossover_rate)])

    return argv


def build_dataset_build_argv(request: DatasetBuildActionRequest) -> list[str]:
    argv = [
        "-m",
        "cli",
        "build-random-trace-dataset",
        "--env",
        request.env_id,
        "--output-dir",
        str(request.output_dir),
        "--dataset-name",
        request.dataset_name,
        "--train-count",
        str(request.train_count),
        "--validation-count",
        str(request.validation_count),
        "--test-count",
        str(request.test_count),
        "--train-offset",
        str(request.train_offset),
        "--validation-offset",
        str(request.validation_offset),
        "--test-offset",
        str(request.test_offset),
    ]
    for split_name in request.split:
        argv.extend(["--split", split_name])
    if request.force:
        argv.append("--force")
    return argv


def resolve_dataset_name_from_config(request: DatasetBuildActionRequest) -> str:
    return request.dataset_name


def build_deepphy_eval_argv(request: DeepPHYEvaluationActionRequest) -> list[str]:
    run_dir, report_root, chosen_run_id = resolve_deepphy_paths(request)
    argv = [
        "-m",
        "deepphy_eval.evaluate_deepphy",
        "--env",
        request.env_id,
        "--benchmark-preset",
        request.benchmark_preset,
        "--prompt-format",
        request.prompt_format,
        "--description-mode",
        request.description_mode,
        "--ollama-base-url",
        request.ollama_base_url,
        "--model-timeout",
        str(request.model_timeout),
        "--run-id",
        chosen_run_id,
    ]
    _append_model_arg(argv, request.model)
    if request.workers != 1:
        argv.extend(["--workers", str(request.workers)])
    if request.task_limit is not None:
        argv.extend(["--task-limit", str(request.task_limit)])
    if request.benchmark_seed is not None:
        argv.extend(["--benchmark-seed", str(request.benchmark_seed)])

    if run_dir is not None:
        argv.extend(["--pattern-run-dir", str(run_dir)])
    elif request.pattern_library is not None:
        argv.extend(
            [
                "--pattern-library",
                str(resolve_pattern_library_path(request.pattern_library)),
            ]
        )

    if request.evidence_relpath is not None:
        argv.extend(["--evidence-relpath", request.evidence_relpath])
    if report_root is not None:
        argv.extend(["--report-root", str(report_root)])
    if request.capture_visuals:
        argv.append("--capture-visuals")
    if request.max_attempts is not None:
        argv.extend(["--max-attempts", str(request.max_attempts)])
    if request.max_steps is not None:
        argv.extend(["--max-steps", str(request.max_steps)])

    return argv


def build_library_holdout_eval_argv(
    request: LibraryHoldoutEvaluationActionRequest,
) -> list[str]:
    argv = [
        "-m",
        "dashboard.workflows",
        "library-evaluate-holdouts",
        "--env",
        request.env_id,
        "--pattern-run-dir",
        str(resolve_pattern_run_dir(request.pattern_run_dir)),
        "--template-name",
        request.template_name,
        "--ollama-base-url",
        request.ollama_base_url,
        "--model-timeout",
        str(request.model_timeout),
        "--workers",
        str(request.workers),
        "--seed-count",
        str(request.seed_count),
        "--holdout-sample-count",
        str(request.holdout_sample_count),
        "--single-holdout-repeat-count",
        str(request.single_holdout_repeat_count),
        "--pair-holdout-size",
        str(request.pair_holdout_size),
        "--pair-holdout-sample-count",
        str(request.pair_holdout_sample_count),
    ]
    _append_model_arg(argv, request.model)
    if request.run_id is not None:
        argv.extend(["--run-id", request.run_id])
    return argv


def build_continuous_library_holdout_loop_argv(
    request: ContinuousLibraryHoldoutLoopActionRequest,
) -> list[str]:
    targets_payload = [
        {
            "env_id": target.env_id,
            "pattern_run_dir": str(resolve_pattern_run_dir(target.pattern_run_dir)),
            "run_id": target.run_id,
        }
        for target in request.targets
    ]
    argv = [
        "-m",
        "dashboard.workflows",
        "library-evaluate-holdouts-continuous",
        "--targets-json",
        json.dumps(targets_payload),
        "--template-name",
        request.template_name,
        "--ollama-base-url",
        request.ollama_base_url,
        "--model-timeout",
        str(request.model_timeout),
        "--workers",
        str(request.workers),
        "--baseline-interval",
        str(request.baseline_interval),
        "--sleep-seconds",
        str(request.sleep_seconds),
    ]
    _append_model_arg(argv, request.model)
    return argv


def build_automatic_launch_argv(request: AutomaticLearningActionRequest) -> list[str]:
    argv = [
        "-m",
        "dashboard.workflows",
        AUTOMATIC_LEARNING_ACTION,
        "--env",
        request.env_id,
    ]
    _append_model_arg(argv, request.model)
    if request.source_supervised_run_dir is not None:
        argv.extend(["--source-supervised-run-dir", str(request.source_supervised_run_dir)])
    elif request.source_pattern_library is not None:
        argv.extend(["--source-pattern-library", str(request.source_pattern_library)])
    argv.extend(["--source-evaluation-run-id", request.source_evaluation_run_id])
    if request.reasoning_trace_count != 6:
        argv.extend(["--reasoning-trace-count", str(request.reasoning_trace_count)])
    if request.proposed_label_count != 10:
        argv.extend(["--proposed-label-count", str(request.proposed_label_count)])
    for split_name in request.split:
        argv.extend(["--split", split_name])
    for split_name in request.validation_split:
        argv.extend(["--validation-split", split_name])
    if request.limit is not None:
        argv.extend(["--limit", str(request.limit)])
    if request.iteration_count != 8:
        argv.extend(["--iteration-count", str(request.iteration_count)])
    if request.workers != 1:
        argv.extend(["--workers", str(request.workers)])
    if request.seed is not None:
        argv.extend(["--seed", str(request.seed)])
    if request.population_size is not None:
        argv.extend(["--population-size", str(request.population_size)])
    if request.elite_fraction is not None:
        argv.extend(["--elite-fraction", str(request.elite_fraction)])
    if request.children_per_elite is not None:
        argv.extend(["--children-per-elite", str(request.children_per_elite)])
    if request.patience_factor is not None and request.patience_factor != 0.35:
        argv.extend(["--patience-factor", str(request.patience_factor)])
    if request.crossover_rate is not None:
        argv.extend(["--crossover-rate", str(request.crossover_rate)])
    if request.run_timestamp is not None:
        argv.extend(["--run-timestamp", request.run_timestamp])
    return argv


def build_unsupervised_launch_argv(request: UnsupervisedLearningActionRequest) -> list[str]:
    argv = [
        "-m",
        "dashboard.workflows",
        UNSUPERVISED_LEARNING_ACTION,
        "--env",
        request.env_id,
        "--dataset-manifest",
        str(request.dataset_manifest),
    ]
    _append_model_arg(argv, request.model)
    for split_name in request.split:
        argv.extend(["--split", split_name])
    for split_name in request.validation_split:
        argv.extend(["--validation-split", split_name])
    if request.limit is not None:
        argv.extend(["--limit", str(request.limit)])
    if request.iteration_count != 8:
        argv.extend(["--iteration-count", str(request.iteration_count)])
    if request.workers != 1:
        argv.extend(["--workers", str(request.workers)])
    if request.seed is not None:
        argv.extend(["--seed", str(request.seed)])
    if request.population_size is not None:
        argv.extend(["--population-size", str(request.population_size)])
    if request.elite_fraction is not None:
        argv.extend(["--elite-fraction", str(request.elite_fraction)])
    if request.children_per_elite is not None:
        argv.extend(["--children-per-elite", str(request.children_per_elite)])
    if request.patience_factor is not None and request.patience_factor != 0.35:
        argv.extend(["--patience-factor", str(request.patience_factor)])
    if request.crossover_rate is not None:
        argv.extend(["--crossover-rate", str(request.crossover_rate)])
    if request.trace_count != 10:
        argv.extend(["--trace-count", str(request.trace_count)])
    if request.labels_per_trace != 10:
        argv.extend(["--labels-per-trace", str(request.labels_per_trace)])
    if request.final_label_count != 10:
        argv.extend(["--final-label-count", str(request.final_label_count)])
    if request.run_timestamp is not None:
        argv.extend(["--run-timestamp", request.run_timestamp])
    return argv


def build_pooltool_reward_optimization_argv(
    request: PooltoolRewardOptimizationActionRequest,
) -> list[str]:
    argv = [
        "-m",
        "dashboard.workflows",
        "pooltool-reward-optimize",
        "--source-run-dir",
        str(resolve_pattern_run_dir(request.source_run_dir)),
        "--goal-text",
        request.goal_text,
        "--dsl-program",
        request.dsl_program,
        "--scene-count",
        str(request.scene_count),
        "--seed-count",
        str(request.seed_count),
        "--rack-jitter",
        str(request.rack_jitter),
        "--ollama-base-url",
        request.ollama_base_url,
        "--model-timeout",
        str(request.model_timeout),
    ]
    _append_model_arg(argv, request.model)
    for budget in request.budgets:
        argv.extend(["--budget", str(int(budget))])
    if request.run_id is not None:
        argv.extend(["--run-id", request.run_id])
    return argv


def build_pooltool_reward_value_model_argv(
    request: PooltoolRewardValueModelActionRequest,
) -> list[str]:
    argv = [
        "-m",
        "dashboard.workflows",
        "pooltool-reward-value-model",
        "--source-run-dir",
        str(resolve_pattern_run_dir(request.source_run_dir)),
        "--goal-text",
        request.goal_text,
        "--dsl-program",
        request.dsl_program,
        "--train-scene-count",
        str(request.train_scene_count),
        "--train-actions-per-scene",
        str(request.train_actions_per_scene),
        "--validation-scene-count",
        str(request.validation_scene_count),
        "--validation-candidates-per-scene",
        str(request.validation_candidates_per_scene),
        "--epochs",
        str(request.epochs),
        "--batch-size",
        str(request.batch_size),
        "--learning-rate",
        str(request.learning_rate),
        "--image-size",
        str(request.image_size),
        "--rack-jitter",
        str(request.rack_jitter),
        "--ollama-base-url",
        request.ollama_base_url,
        "--model-timeout",
        str(request.model_timeout),
    ]
    _append_model_arg(argv, request.model)
    if request.run_id is not None:
        argv.extend(["--run-id", request.run_id])
    return argv
