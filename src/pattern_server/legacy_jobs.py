from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import CreateJobRequest, JobConstraints, JobSpec


def _services_root() -> Path:
    resolved = Path(__file__).resolve()
    for parent in resolved.parents:
        if (parent / "Work").is_dir() and (parent / "AGENTS.md").is_file():
            return parent
    return resolved.parents[4]


def _client_project_root_candidates() -> list[Path]:
    return [SERVER_PROJECT_ROOT]


MONOREPO_ROOT = _services_root()
SERVER_PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_RUNTIME_SRC_CANDIDATES = [
    *(root / "src" for root in _client_project_root_candidates()),
    MONOREPO_ROOT / "src",
]
AVAILABLE_LEGACY_RUNTIME_SRCS = [
    candidate for candidate in LEGACY_RUNTIME_SRC_CANDIDATES if candidate.is_dir()
]
for candidate in reversed(AVAILABLE_LEGACY_RUNTIME_SRCS):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

LEGACY_RUNTIME_IMPORT_ERROR: ModuleNotFoundError | None = None

try:
    from dashboard.actions import (  # noqa: E402
        AUTOMATIC_LEARNING_ACTION,
        UNSUPERVISED_LEARNING_ACTION,
        ContinuousLibraryHoldoutLoopActionRequest,
        DatasetBuildActionRequest,
        DeepPHYEvaluationActionRequest,
        LibraryHoldoutEvaluationActionRequest,
        PatternLearningActionRequest,
        PooltoolRewardOptimizationActionRequest,
        PooltoolRewardValueModelActionRequest,
        AutomaticLearningActionRequest,
        UnsupervisedLearningActionRequest,
        build_automatic_launch_argv,
        build_continuous_library_holdout_loop_argv,
        build_dataset_build_argv,
        build_deepphy_eval_argv,
        build_library_holdout_eval_argv,
        build_pooltool_reward_optimization_argv,
        build_pooltool_reward_value_model_argv,
        build_supervised_learning_with_auto_eval_argv,
        build_unsupervised_launch_argv,
        default_pattern_learning_output_path,
        pattern_run_dir_for_output_path,
        python_executable_for_env,
        resolve_deepphy_paths,
    )
    from deepphy_eval import summarize_deepphy_benchmark  # noqa: E402
    from pattern_library_paths import default_automatic_run_dir, default_unsupervised_run_dir  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in live env
    LEGACY_RUNTIME_IMPORT_ERROR = exc
    AUTOMATIC_LEARNING_ACTION = "automatic"
    UNSUPERVISED_LEARNING_ACTION = "unsupervised"


def _require_legacy_runtime() -> None:
    if LEGACY_RUNTIME_IMPORT_ERROR is None:
        return
    missing = getattr(LEGACY_RUNTIME_IMPORT_ERROR, "name", "legacy runtime")
    expected_locations = ", ".join(str(path) for path in LEGACY_RUNTIME_SRC_CANDIDATES)
    raise RuntimeError(
        "Legacy Pattern Learning runtime modules are unavailable "
        f"({missing}). Expected repo support under one of: {expected_locations}."
    )


DEFAULT_FUNSEARCH_TEMPLATE = "full"
AUTOMATIC_WORKFLOW_KIND = "automatic"
UNSUPERVISED_WORKFLOW_KIND = "unsupervised"
POOLTOOL_REWARD_OPTIMIZATION_ACTION = "pooltool-reward-optimize"
POOLTOOL_REWARD_VALUE_MODEL_ACTION = "pooltool-reward-value-model"
POOLTOOL_REWARD_SINGLE_SCENE_ACTION = "pooltool-reward-single-scene"
SUPERVISED_CONSISTENCY_ANALYSIS_ACTION = "analyze-supervised-consistency"
SERVER_MIN_CANDIDATE_EVALUATIONS = 100
SERVER_NO_IMPROVEMENT_PATIENCE_EVALUATIONS = 25
SERVER_MAX_CANDIDATE_EVALUATIONS = 100
SERVER_POPULATION_SIZE = 25
SERVER_POLICY_CONFIG_RELATIVE_PATH = Path(
    "configs/generated/pattern_learning_server_policy.yaml"
)
REWARD_RUNS_LOCAL_ONLY_ERROR = (
    "Reward experiments are local-only. Launch them from the local reward dashboard "
    "instead of creating a server job."
)


def _default_llm_api_key() -> str:
    for name in ("PATTERN_LLM_API_KEY", "JOB_LLM_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return ""


def _learning_python() -> str:
    _require_legacy_runtime()
    return str(python_executable_for_env("learning"))


def _learning_python_fallback_ok() -> str:
    if LEGACY_RUNTIME_IMPORT_ERROR is None:
        return str(python_executable_for_env("learning"))
    candidate_paths = [
        root / "venv" / "learning" / "bin" / "python"
        for root in _client_project_root_candidates()
    ]
    for candidate in candidate_paths:
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError(
        "Learning Python runtime is unavailable. "
        f"Expected one of {', '.join(str(path) for path in candidate_paths)} or a working legacy runtime import."
    )


def _env_python_fallback_ok(env_id: str) -> str:
    env_name = str(env_id or "").strip()
    if LEGACY_RUNTIME_IMPORT_ERROR is None:
        return str(python_executable_for_env(env_name))
    candidate_paths = [
        root / "venv" / env_name / "bin" / "python"
        for root in _client_project_root_candidates()
    ]
    for candidate in candidate_paths:
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError(
        f"{env_name} Python runtime is unavailable. "
        f"Expected one of {', '.join(str(path) for path in candidate_paths)} or a working legacy runtime import."
    )


def _env_constraints(env_id: str, *, needs_learning: bool) -> JobConstraints:
    required = [str(env_id)]
    if needs_learning:
        required.insert(0, "learning")
    return JobConstraints(required_environments=required)


def _multi_env_learning_constraints(env_ids: list[str]) -> JobConstraints:
    required = ["learning", *sorted({str(env_id).strip() for env_id in env_ids if str(env_id).strip()})]
    return JobConstraints(required_environments=required)


def _coerce_pathish(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _server_pattern_data_root() -> Path:
    return SERVER_PROJECT_ROOT / "data" / "patterns"


def _automatic_source_dir_local_candidates(raw_path: str) -> list[Path]:
    candidate = Path(str(raw_path).strip())
    if not str(candidate).strip():
        return []
    try:
        patterns_index = candidate.parts.index("patterns")
    except ValueError:
        return []
    if len(candidate.parts) <= patterns_index + 2:
        return []
    env_id = candidate.parts[patterns_index + 1]
    relative_parts = list(candidate.parts[patterns_index + 2 :])
    if not relative_parts:
        return []
    local_env_root = _server_pattern_data_root() / env_id
    relative_options: list[Path] = [Path(*relative_parts)]
    if relative_parts[0] == "supervised" and len(relative_parts) > 1:
        relative_options.append(Path(*relative_parts[1:]))
    discovered: list[Path] = []
    seen: set[str] = set()
    for relative in relative_options:
        base = local_env_root / relative
        for entry in [base, *sorted(base.parent.glob(f"{base.name}.__resume_sync__*"))]:
            key = str(entry)
            if key in seen:
                continue
            seen.add(key)
            discovered.append(entry)
    return discovered


def _automatic_source_dir_score(path: Path) -> tuple[int, int, int, int]:
    results_count = sum(
        1 for item in (path / "deepphy_eval").glob("*/results.json") if item.is_file()
    )
    has_library = int((path / "pattern_library.json").is_file())
    has_run_json = int((path / "run.json").is_file())
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    return (int(results_count > 0), has_library, has_run_json, mtime_ns)


def _project_automatic_source_dir(raw_path: str, chosen_local_path: Path) -> str:
    raw = Path(str(raw_path).strip())
    try:
        raw_patterns_index = raw.parts.index("patterns")
        local_patterns_index = chosen_local_path.parts.index("patterns")
    except ValueError:
        return str(raw)
    if len(raw.parts) <= raw_patterns_index + 2 or len(chosen_local_path.parts) <= local_patterns_index + 2:
        return str(raw)
    raw_env_root = Path(*raw.parts[: raw_patterns_index + 2])
    local_relative = Path(*chosen_local_path.parts[local_patterns_index + 2 :])
    return str(raw_env_root / local_relative)


def _normalize_automatic_source_supervised_run_dir(raw_path: str | None) -> str | None:
    normalized = str(raw_path or "").strip()
    if not normalized:
        return None
    candidates = _automatic_source_dir_local_candidates(normalized)
    if not candidates:
        return normalized
    usable = [path for path in candidates if _automatic_source_dir_score(path)[:3] > (0, 0, 0)]
    if not usable:
        return normalized
    best = max(usable, key=_automatic_source_dir_score)
    return _project_automatic_source_dir(normalized, best)


def _pooltool_reward_run_relpath(*, kind: str, run_id: str) -> str:
    return (
        Path("data")
        / "patterns"
        / "pooltool"
        / "reward_program_experiments"
        / kind
        / run_id
    ).as_posix()


def _replace_or_add_flag(argv: list[str], flag: str, value: str) -> list[str]:
    rewritten = [str(part) for part in argv]
    index = 0
    while index < len(rewritten):
        if rewritten[index] == flag:
            if index + 1 < len(rewritten):
                rewritten[index + 1] = value
            else:
                rewritten.append(value)
            return rewritten
        index += 1
    rewritten.extend([flag, value])
    return rewritten


def _remove_flag(argv: list[str], flag: str) -> list[str]:
    rewritten: list[str] = []
    normalized = [str(part) for part in argv]
    index = 0
    while index < len(normalized):
        if normalized[index] == flag:
            index += 2
            continue
        rewritten.append(normalized[index])
        index += 1
    return rewritten


def _server_learning_stop_policy_config_path() -> Path:
    target = (SERVER_PROJECT_ROOT / SERVER_POLICY_CONFIG_RELATIVE_PATH).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(
            [
                'config_version: "1"',
                "split:",
                "  - train",
                "validation_split:",
                "  - validation",
                "generator_retries: 3",
                "iteration_count: 8",
                "seed: 42",
                f"population_size: {SERVER_POPULATION_SIZE}",
                "elite_fraction: 0.25",
                "children_per_elite: 2",
                "crossover_rate: 1.0",
                f"min_candidate_evaluations: {SERVER_MIN_CANDIDATE_EVALUATIONS}",
                (
                    "no_improvement_patience_evaluations: "
                    f"{SERVER_NO_IMPROVEMENT_PATIENCE_EVALUATIONS}"
                ),
                f"max_candidate_evaluations: {SERVER_MAX_CANDIDATE_EVALUATIONS}",
                "accept_margin: 0.001",
                "max_repairs: 2",
                "mutate_temperature: 0.7",
                "mutate_max_tokens: 8192",
                "losses:",
                "  - kind: correlation_structure",
                "    weight: 1.0",
                "    histogram_bins: 12",
                "    pair_limit: 500",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return target


def _apply_server_learning_stop_policy_to_argv(
    argv: list[str],
    *,
    action: str | None = None,
) -> list[str]:
    rewritten = _remove_flag(argv, "--patience-factor")
    rewritten = _remove_flag(rewritten, "--population-size")
    rewritten = _remove_flag(rewritten, "--min-candidate-evaluations")
    rewritten = _remove_flag(rewritten, "--no-improvement-patience-evaluations")
    rewritten = _remove_flag(rewritten, "--max-candidate-evaluations")
    if str(action or "").strip() == "automatic-launch":
        return _remove_flag(rewritten, "--config")
    rewritten = _replace_or_add_flag(
        rewritten,
        "--config",
        str(_server_learning_stop_policy_config_path()),
    )
    return rewritten


def _apply_server_learning_stop_policy_to_request(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.pop("patience_factor", None)
    normalized["population_size"] = SERVER_POPULATION_SIZE
    normalized["min_candidate_evaluations"] = SERVER_MIN_CANDIDATE_EVALUATIONS
    normalized["no_improvement_patience_evaluations"] = SERVER_NO_IMPROVEMENT_PATIENCE_EVALUATIONS
    normalized["max_candidate_evaluations"] = SERVER_MAX_CANDIDATE_EVALUATIONS
    normalized["config_path"] = str(_server_learning_stop_policy_config_path())
    return normalized


def _append_defer_evaluation_flag(argv: list[str]) -> list[str]:
    if "--defer-evaluation" in argv:
        return [str(part) for part in argv]
    return [*map(str, argv), "--defer-evaluation"]


class PooltoolRewardSingleSceneActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: str = Field(default="pooltool", min_length=1)
    source_run_dir: Path
    scene_path: Path
    goal_text: str = ""
    dsl_program: str = ""
    model: str = ""
    reward_kind: str = "synthesized"
    budget: int = Field(default=250, ge=1)
    seed: int = Field(default=0, ge=0)
    ollama_base_url: str = ""
    model_timeout: int = Field(default=0, ge=0)
    run_id: str | None = None

    @model_validator(mode="after")
    def _validate_inputs(self) -> "PooltoolRewardSingleSceneActionRequest":
        self.env_id = str(self.env_id or "").strip().lower()
        if self.env_id != "pooltool":
            raise ValueError(f"Unsupported environment {self.env_id}")
        if not self.goal_text.strip() and not self.dsl_program.strip():
            raise ValueError("One of goal_text or dsl_program is required")
        reward_kind = str(self.reward_kind or "").strip().lower()
        if reward_kind not in {"synthesized", "binary"}:
            raise ValueError(f"Unsupported reward kind {self.reward_kind}")
        self.reward_kind = reward_kind
        self.source_run_dir = Path(self.source_run_dir).resolve()
        self.scene_path = Path(self.scene_path).resolve()
        if not self.scene_path.is_file():
            raise ValueError(f"Scene file not found: {self.scene_path}")
        return self


def datetime_now_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _supervised_consistency_run_relpath(*, env_id: str, run_id: str) -> str:
    return (
        Path("data")
        / "analysis"
        / "supervised_consistency"
        / env_id
        / run_id
    ).as_posix()


def _resolve_supervised_consistency_trace_paths(*, env_id: str, use_env_test_set: bool, trace_path: Path | None) -> list[Path]:
    if not use_env_test_set and trace_path is not None:
        resolved = Path(trace_path).resolve()
        if not resolved.is_file():
            raise ValueError(f"Trace file not found: {resolved}")
        return [resolved]

    dataset_root = SERVER_PROJECT_ROOT / "data" / "datasets" / env_id
    traces: list[Path] = []
    canonical_root = dataset_root / "default_simulator_trace_dataset" / "test"
    if canonical_root.is_dir():
        for candidate in sorted(canonical_root.glob("**/trace.json")):
            if candidate.is_file():
                traces.append(candidate.resolve())
    if not traces:
        for candidate in sorted(dataset_root.glob("**/test/**/trace.json")):
            if candidate.is_file():
                traces.append(candidate.resolve())
    if not traces:
        raise ValueError(f"No test traces found under {dataset_root}")
    return traces


class SupervisedConsistencyAnalysisActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: str = Field(min_length=1)
    run_dirs: list[Path] = Field(min_length=2)
    trace_path: Path | None = None
    use_env_test_set: bool = True
    checkpoints: list[int] = Field(default_factory=list)
    sampling_repeats: int = Field(default=32, ge=1)
    run_id: str | None = None
    output_dir: Path | None = None

    @model_validator(mode="after")
    def _validate_inputs(self) -> "SupervisedConsistencyAnalysisActionRequest":
        env_id = str(self.env_id or "").strip().lower()
        if env_id not in {"phyre", "iphyre", "kinetix", "pooltool"}:
            raise ValueError(f"Unsupported environment {self.env_id}")
        normalized_run_dirs: list[Path] = []
        seen: set[str] = set()
        for path in self.run_dirs:
            resolved = Path(path).resolve()
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            normalized_run_dirs.append(resolved)
        if len(normalized_run_dirs) < 2:
            raise ValueError("At least two distinct supervised run directories are required")
        self.env_id = env_id
        self.run_dirs = normalized_run_dirs
        self.trace_path = Path(self.trace_path).resolve() if self.trace_path is not None else None
        self.use_env_test_set = bool(self.use_env_test_set)
        self.checkpoints = [max(1, int(value)) for value in self.checkpoints]
        self.sampling_repeats = min(32, max(1, int(self.sampling_repeats)))
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir).resolve()
        return self


def create_supervised_job(
    payload: dict[str, Any],
) -> CreateJobRequest:
    _require_legacy_runtime()
    body = dict(payload)
    enqueue = bool(body.pop("enqueue", True))
    funsearch_template = str(body.pop("funsearch_template", DEFAULT_FUNSEARCH_TEMPLATE))
    action_request = PatternLearningActionRequest.model_validate(body)
    if action_request.output_path is None:
        action_request.output_path = default_pattern_learning_output_path(action_request)
    run_dir = pattern_run_dir_for_output_path(action_request.output_path)
    workflow_argv = build_supervised_learning_with_auto_eval_argv(action_request)
    argv = _apply_server_learning_stop_policy_to_argv([_learning_python(), *workflow_argv])
    request_payload = _apply_server_learning_stop_policy_to_request(
        action_request.model_dump(mode="json", exclude_none=True)
    )
    return CreateJobRequest(
        enqueue=enqueue,
        spec=JobSpec(
            action="supervised-learn-and-evaluate",
            argv=argv,
            cwd=str(MONOREPO_ROOT),
            metadata={
                "action": "supervised-learn-and-evaluate",
                "env_id": action_request.env_id,
                "workflow_kind": "supervised",
                "supports_pause": True,
                "resume_argv": [*argv, "--resume"],
                "output_path": str(action_request.output_path),
                "run_dir": str(run_dir),
                "funsearch_template": funsearch_template,
                "request": request_payload,
            },
            constraints=_env_constraints(action_request.env_id, needs_learning=True),
        ),
    )


def create_dataset_build_job(payload: dict[str, Any]) -> CreateJobRequest:
    _require_legacy_runtime()
    body = dict(payload)
    enqueue = bool(body.pop("enqueue", False))
    action_request = DatasetBuildActionRequest.model_validate(body)
    argv = [_learning_python(), *build_dataset_build_argv(action_request)]
    return CreateJobRequest(
        enqueue=enqueue,
        spec=JobSpec(
            action="build-dataset",
            argv=argv,
            cwd=str(MONOREPO_ROOT),
            metadata={
                "action": "build-dataset",
                "env_id": action_request.env_id,
                "dataset_name": action_request.dataset_name,
                "supports_pause": True,
                "request": action_request.model_dump(mode="json", exclude_none=True),
            },
            constraints=_env_constraints(action_request.env_id, needs_learning=True),
        ),
    )


def create_automatic_job(payload: dict[str, Any]) -> CreateJobRequest:
    _require_legacy_runtime()
    body = dict(payload)
    enqueue = bool(body.pop("enqueue", True))
    funsearch_template = str(body.pop("funsearch_template", DEFAULT_FUNSEARCH_TEMPLATE))
    normalized_source_run_dir = _normalize_automatic_source_supervised_run_dir(
        _coerce_pathish(body.get("source_supervised_run_dir"))
    )
    if normalized_source_run_dir is not None:
        body["source_supervised_run_dir"] = normalized_source_run_dir
    action_request = AutomaticLearningActionRequest.model_validate(body)
    effective_run_dir = default_automatic_run_dir(
        action_request.env_id,
        timestamp=action_request.run_timestamp,
    )
    effective_request = action_request.model_copy(
        update={"run_timestamp": effective_run_dir.name}
    )
    workflow_argv = build_automatic_launch_argv(effective_request)
    argv = [_learning_python(), *workflow_argv]
    request_payload = _apply_server_learning_stop_policy_to_request(
        effective_request.model_dump(mode="json", exclude_none=True)
    )
    return CreateJobRequest(
        enqueue=enqueue,
        spec=JobSpec(
            action=AUTOMATIC_LEARNING_ACTION,
            argv=argv,
            cwd=str(MONOREPO_ROOT),
            metadata={
                "action": AUTOMATIC_LEARNING_ACTION,
                "env_id": effective_request.env_id,
                "workflow_kind": AUTOMATIC_WORKFLOW_KIND,
                "supports_pause": True,
                "resume_argv": [*argv, "--resume"],
                "funsearch_template": funsearch_template,
                "request": request_payload,
                "source_supervised_run_dir": _coerce_pathish(
                    effective_request.source_supervised_run_dir
                ),
                "source_supervised_run_id": (
                    effective_request.source_supervised_run_dir.name
                    if effective_request.source_supervised_run_dir is not None
                    else None
                ),
                "source_evaluation_run_id": effective_request.source_evaluation_run_id,
                "run_dir": str(effective_run_dir),
            },
            constraints=_env_constraints(effective_request.env_id, needs_learning=True),
        ),
    )


def create_unsupervised_job(payload: dict[str, Any]) -> CreateJobRequest:
    _require_legacy_runtime()
    body = dict(payload)
    enqueue = bool(body.pop("enqueue", True))
    funsearch_template = str(body.pop("funsearch_template", DEFAULT_FUNSEARCH_TEMPLATE))
    action_request = UnsupervisedLearningActionRequest.model_validate(body)
    effective_run_dir = default_unsupervised_run_dir(
        action_request.env_id,
        timestamp=action_request.run_timestamp,
    )
    effective_request = action_request.model_copy(
        update={"run_timestamp": effective_run_dir.name}
    )
    workflow_argv = build_unsupervised_launch_argv(effective_request)
    argv = _apply_server_learning_stop_policy_to_argv([_learning_python(), *workflow_argv])
    request_payload = _apply_server_learning_stop_policy_to_request(
        effective_request.model_dump(mode="json", exclude_none=True)
    )
    return CreateJobRequest(
        enqueue=enqueue,
        spec=JobSpec(
            action=UNSUPERVISED_LEARNING_ACTION,
            argv=argv,
            cwd=str(MONOREPO_ROOT),
            metadata={
                "action": UNSUPERVISED_LEARNING_ACTION,
                "env_id": effective_request.env_id,
                "workflow_kind": UNSUPERVISED_WORKFLOW_KIND,
                "supports_pause": True,
                "resume_argv": [*argv, "--resume"],
                "funsearch_template": funsearch_template,
                "request": request_payload,
                "run_dir": str(effective_run_dir),
            },
            constraints=_env_constraints(effective_request.env_id, needs_learning=True),
        ),
    )


def create_deepphy_eval_job(payload: dict[str, Any]) -> CreateJobRequest:
    _require_legacy_runtime()
    body = dict(payload)
    enqueue = bool(body.pop("enqueue", True))
    source_run_id = body.pop("source_run_id", None)
    enabled_labels = body.pop("enabled_labels", None)
    disabled_labels = body.pop("disabled_labels", None)
    evaluation_scope = body.pop("evaluation_scope", None)
    if not str(body.get("llm_api_key") or "").strip():
        default_llm_api_key = _default_llm_api_key()
        if default_llm_api_key:
            body["llm_api_key"] = default_llm_api_key
    action_request = DeepPHYEvaluationActionRequest.model_validate(body)
    eval_python = str(python_executable_for_env(action_request.env_id))
    workflow_argv = build_deepphy_eval_argv(action_request)
    argv = [eval_python, *workflow_argv]
    resolved_pattern_run_dir, evaluation_run_dir, chosen_eval_run_id = resolve_deepphy_paths(
        action_request
    )
    metadata: dict[str, Any] = {
        "action": "evaluate-deepphy",
        "supports_pause": True,
        "env_id": action_request.env_id,
        "evaluation_run_id": chosen_eval_run_id,
        "evaluation_run_dir": _coerce_pathish(evaluation_run_dir),
        "benchmark_preset": action_request.benchmark_preset,
        "benchmark_task_count": summarize_deepphy_benchmark(
            action_request.env_id,
            preset=action_request.benchmark_preset,
            task_limit=action_request.task_limit,
        ).task_count,
        "request": action_request.model_dump(mode="json", exclude_none=True),
        "resume_argv": argv,
    }
    if resolved_pattern_run_dir is not None:
        metadata["pattern_run_dir"] = str(resolved_pattern_run_dir)
    if source_run_id is not None:
        metadata["run_id"] = str(source_run_id)
    if enabled_labels is not None:
        metadata["enabled_labels"] = list(enabled_labels)
    if disabled_labels is not None:
        metadata["disabled_labels"] = list(disabled_labels)
    if evaluation_scope is not None:
        metadata["evaluation_scope"] = str(evaluation_scope)
    return CreateJobRequest(
        enqueue=enqueue,
        spec=JobSpec(
            action="evaluate-deepphy",
            argv=argv,
            cwd=str(MONOREPO_ROOT),
            metadata=metadata,
            constraints=JobConstraints(required_environments=[action_request.env_id]),
        ),
    )


def create_library_holdout_job(payload: dict[str, Any]) -> CreateJobRequest:
    _require_legacy_runtime()
    body = dict(payload)
    enqueue = bool(body.pop("enqueue", True))
    preferred_client_id = str(body.pop("preferred_client_id", "") or "").strip() or None
    library_ablation_kind = str(body.pop("library_ablation_kind", "") or "").strip().lower()
    action_request = LibraryHoldoutEvaluationActionRequest.model_validate(body)
    base_argv = [_learning_python(), *build_library_holdout_eval_argv(action_request)]
    argv = [*base_argv, "--resume"]
    library_evaluation_run_id = str(action_request.run_id) if action_request.run_id is not None else None
    library_evaluation_run_dir = (
        Path(action_request.pattern_run_dir).resolve() / "library_evaluations" / library_evaluation_run_id
        if library_evaluation_run_id
        else None
    )
    if library_ablation_kind not in {"supervised", "unsupervised", "automatic"}:
        normalized_run_dir = str(Path(action_request.pattern_run_dir).resolve()).replace("\\", "/").lower()
        if "/automatic/" in normalized_run_dir or normalized_run_dir.endswith("-automatic"):
            library_ablation_kind = "automatic"
        elif "/unsupervised/" in normalized_run_dir or normalized_run_dir.endswith("-unsupervised"):
            library_ablation_kind = "unsupervised"
        else:
            library_ablation_kind = "supervised"
    return CreateJobRequest(
        enqueue=enqueue,
        spec=JobSpec(
            action="evaluate-library-case",
            argv=argv,
            cwd=str(MONOREPO_ROOT),
            metadata={
                "action": "evaluate-library-case",
                "env_id": action_request.env_id,
                "supports_pause": True,
                "pattern_run_dir": str(action_request.pattern_run_dir),
                "library_ablation_kind": library_ablation_kind,
                "library_evaluation_run_id": library_evaluation_run_id,
                "library_evaluation_run_dir": str(library_evaluation_run_dir)
                if library_evaluation_run_dir is not None
                else "",
                "preferred_client_id": preferred_client_id,
                "request": action_request.model_dump(mode="json", exclude_none=True),
                "resume_argv": argv,
            },
            constraints=_env_constraints(action_request.env_id, needs_learning=True),
        ),
    )


def create_continuous_library_holdout_loop_job(
    payload: dict[str, Any],
) -> CreateJobRequest:
    _require_legacy_runtime()
    body = dict(payload)
    enqueue = bool(body.pop("enqueue", True))
    preferred_client_id = str(body.pop("preferred_client_id", "") or "").strip() or None
    action_request = ContinuousLibraryHoldoutLoopActionRequest.model_validate(body)
    argv = [_learning_python(), *build_continuous_library_holdout_loop_argv(action_request)]
    target_env_ids = [target.env_id for target in action_request.targets]
    target_pattern_run_dirs = [str(target.pattern_run_dir) for target in action_request.targets]
    return CreateJobRequest(
        enqueue=enqueue,
        spec=JobSpec(
            action="evaluate-library-case-loop",
            argv=argv,
            cwd=str(MONOREPO_ROOT),
            metadata={
                "action": "evaluate-library-case-loop",
                "env_ids": target_env_ids,
                "pattern_run_dirs": target_pattern_run_dirs,
                "preferred_client_id": preferred_client_id,
                "supports_pause": True,
                "request": action_request.model_dump(mode="json", exclude_none=True),
            },
            constraints=_multi_env_learning_constraints(target_env_ids),
        ),
    )


def create_supervised_consistency_analysis_job(payload: dict[str, Any]) -> CreateJobRequest:
    body = dict(payload)
    enqueue = bool(body.pop("enqueue", True))
    preferred_client_id = str(body.pop("preferred_client_id", "") or "").strip() or None
    action_request = SupervisedConsistencyAnalysisActionRequest.model_validate(body)
    run_id = (
        str(action_request.run_id).strip()
        if action_request.run_id is not None and str(action_request.run_id).strip()
        else f"{datetime_now_utc_compact()}-{action_request.env_id}-supervised-consistency"
    )
    output_dir = (
        action_request.output_dir
        if action_request.output_dir is not None
        else (MONOREPO_ROOT / _supervised_consistency_run_relpath(env_id=action_request.env_id, run_id=run_id))
    )
    trace_paths = _resolve_supervised_consistency_trace_paths(
        env_id=action_request.env_id,
        use_env_test_set=bool(action_request.use_env_test_set),
        trace_path=action_request.trace_path,
    )
    argv = [
        _learning_python_fallback_ok(),
        "-m",
        "code_convergence.consistency_analysis",
        "--env-id",
        action_request.env_id,
        "--output-dir",
        str(output_dir),
    ]
    for trace_path in trace_paths:
        argv.extend(["--trace-path", str(trace_path)])
    for run_dir in action_request.run_dirs:
        argv.extend(["--run-dir", str(run_dir)])
    for checkpoint in action_request.checkpoints:
        argv.extend(["--checkpoint", str(int(checkpoint))])
    argv.extend(["--sampling-repeats", str(int(action_request.sampling_repeats))])
    return CreateJobRequest(
        enqueue=enqueue,
        spec=JobSpec(
            action=SUPERVISED_CONSISTENCY_ANALYSIS_ACTION,
            argv=argv,
            cwd=str(MONOREPO_ROOT),
            metadata={
                "action": SUPERVISED_CONSISTENCY_ANALYSIS_ACTION,
                "env_id": action_request.env_id,
                "analysis_kind": "supervised_consistency",
                "supports_pause": True,
                "analysis_run_id": run_id,
                "run_dir": str(output_dir),
                "trace_path": str(trace_paths[0]),
                "trace_paths": [str(path) for path in trace_paths],
                "trace_scope": "test_set" if bool(action_request.use_env_test_set) else "single_trace",
                "source_run_dirs": [str(path) for path in action_request.run_dirs],
                "artifact_paths": [str(output_dir)],
                "preferred_client_id": preferred_client_id,
                "request": action_request.model_dump(mode="json", exclude_none=True),
                "resume_argv": argv,
            },
            constraints=_env_constraints(action_request.env_id, needs_learning=True),
        ),
    )


def create_pooltool_reward_optimization_job(
    payload: dict[str, Any],
) -> CreateJobRequest:
    raise ValueError(REWARD_RUNS_LOCAL_ONLY_ERROR)


def create_pooltool_reward_value_model_job(
    payload: dict[str, Any],
) -> CreateJobRequest:
    raise ValueError(REWARD_RUNS_LOCAL_ONLY_ERROR)


def create_pooltool_reward_single_scene_job(
    payload: dict[str, Any],
) -> CreateJobRequest:
    raise ValueError(REWARD_RUNS_LOCAL_ONLY_ERROR)
