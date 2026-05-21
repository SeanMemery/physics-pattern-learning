from __future__ import annotations

import json
import random
from dataclasses import dataclass
import concurrent.futures
from datetime import datetime, timezone
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Literal

from adapters.registry import AdapterRegistry
from deepphy_eval.deepphy_text.models import (
    DeepPHYTextDescriptionMode,
    DeepPHYTextEvaluationReport,
    DeepPHYTextEvaluationRequest,
    DeepPHYTextEvaluationResult,
    DeepPHYTextPromptFormat,
)
from deepphy_eval.deepphy_text.reporting import build_deepphy_text_report
from deepphy_eval.deepphy_text.runner import (
    ModelResponder,
    TraceDescriber,
    evaluate_deepphy_text_request,
)
from deepphy_eval.logging import default_run_id
from pattern_library_paths import resolve_pattern_library_path, resolve_pattern_run_dir
from pydantic import BaseModel, ConfigDict, Field, model_validator
from runtime.envs import default_interpreter_relpath


@dataclass(frozen=True)
class DeepPHYBenchmarkTask:
    task_id: str
    adapter_kwargs: dict[str, Any]


@dataclass(frozen=True)
class DeepPHYBenchmarkOverview:
    env_id: Literal["phyre", "iphyre", "kinetix", "pooltool"]
    task_count: int
    label: str
    paper_note: str


@dataclass(frozen=True)
class DeepPHYBenchmarkSpec:
    env_id: Literal["phyre", "iphyre", "kinetix", "pooltool"]
    task_count: int
    benchmark_kwargs: dict[str, Any]
    tasks: tuple[DeepPHYBenchmarkTask, ...]
    label: str
    paper_note: str


_BENCHMARK_OVERVIEWS: dict[str, DeepPHYBenchmarkOverview] = {
    "phyre": DeepPHYBenchmarkOverview(
        env_id="phyre",
        task_count=1000,
        label="Combined PHYRE-B test set",
        paper_note="500 within-template + 500 cross-template PHYRE-B test tasks (1,000 total), as specified in Appendix B/Table 5.",
    ),
    "iphyre": DeepPHYBenchmarkOverview(
        env_id="iphyre",
        task_count=40,
        label="I-PHYRE scenario set",
        paper_note="All 40 I-PHYRE benchmark scenarios with up to 10 attempts, as specified in Appendix C.",
    ),
    "kinetix": DeepPHYBenchmarkOverview(
        env_id="kinetix",
        task_count=74,
        label="Kinetix fixed handcrafted test set",
        paper_note="74 fixed handcrafted Kinetix test levels (S: 10, M: 24, L: 40), as specified in Appendix D.",
    ),
    "pooltool": DeepPHYBenchmarkOverview(
        env_id="pooltool",
        task_count=100,
        label="Pooltool 100-experiment protocol",
        paper_note="100 Pooltool experiments with up to 8 attempts each.",
    ),
}

_TASK_RESULT_FILENAME = "task_result.json"
_DEFAULT_FULL_TASK_LIMITS: dict[str, int] = {
    "phyre": 125,
    "kinetix": 50,
}
_DEFAULT_QUICK_EVAL_TASK_LIMITS: dict[str, int] = {
    "phyre": 50,
    "iphyre": 10,
    "kinetix": 15,
    "pooltool": 20,
}
_SUBPROCESS_ISOLATED_ENVS = frozenset({"kinetix"})
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DeepPHYEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: Literal["phyre", "iphyre", "kinetix", "pooltool"]
    model_name: str = "mock"
    benchmark_preset: Literal["full", "quick-eval", "test"] = "full"
    task_limit: int | None = Field(default=None, ge=1)
    benchmark_seed: int | None = None
    prompt_format: DeepPHYTextPromptFormat = "WM"
    description_mode: DeepPHYTextDescriptionMode = "pattern_text"
    pattern_run_dir: str | None = None
    pattern_library_path: str | None = None
    evidence_relpath: str | None = None
    run_id: str | None = None
    report_root: str | None = None
    capture_visuals: bool = False
    workers: int = Field(default=1, ge=1)
    benchmark_kwargs: dict[str, Any] = Field(default_factory=dict)
    adapter_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_pattern_inputs(self) -> "DeepPHYEvaluationRequest":
        self.prompt_format = "WM"
        if self.pattern_run_dir is not None and self.pattern_library_path is not None:
            raise ValueError(
                "pattern_run_dir and pattern_library_path are mutually exclusive"
            )
        if self.task_limit is None:
            self.task_limit = default_benchmark_task_limit(
                self.env_id,
                preset=self.benchmark_preset,
            )
        return self


def default_benchmark_task_limit(
    env_id: Literal["phyre", "iphyre", "kinetix", "pooltool"],
    *,
    preset: Literal["full", "quick-eval", "test"] = "full",
) -> int | None:
    if preset == "full":
        return _DEFAULT_FULL_TASK_LIMITS.get(env_id)
    if preset == "quick-eval":
        return _DEFAULT_QUICK_EVAL_TASK_LIMITS.get(env_id)
    return None


def load_deepphy_request(path: Path) -> DeepPHYEvaluationRequest:
    import json

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return DeepPHYEvaluationRequest.model_validate(payload)


def summarize_deepphy_benchmark(
    env_id: Literal["phyre", "iphyre", "kinetix", "pooltool"],
    *,
    preset: Literal["full", "quick-eval", "test"] = "full",
    task_limit: int | None = None,
) -> DeepPHYBenchmarkOverview:
    try:
        overview = _BENCHMARK_OVERVIEWS[env_id]
    except KeyError as error:
        raise ValueError(f"unsupported environment: {env_id}") from error
    if preset in {"full", "quick-eval"}:
        if isinstance(task_limit, int) and task_limit > 0:
            limited_count = min(overview.task_count, task_limit)
            if limited_count != overview.task_count:
                note_prefix = (
                    "Dashboard default currently caps this run to"
                    if preset == "full"
                    else "Quick eval currently caps this run to"
                )
                return DeepPHYBenchmarkOverview(
                    env_id=overview.env_id,
                    task_count=limited_count,
                    label=overview.label,
                    paper_note=(
                        f"{overview.paper_note} {note_prefix} {limited_count} tasks."
                    ),
                )
        return overview
    if preset == "test":
        return DeepPHYBenchmarkOverview(
            env_id=overview.env_id,
            task_count=1,
            label=f"{overview.label} · single sampled task",
            paper_note=f"Single sampled task from the full {overview.task_count}-task benchmark inventory.",
        )
    raise ValueError(f"unsupported benchmark preset: {preset}")


def describe_deepphy_benchmark(
    env_id: Literal["phyre", "iphyre", "kinetix", "pooltool"],
    *,
    preset: Literal["full", "quick-eval", "test"] = "full",
    task_limit: int | None = None,
    benchmark_seed: int | None = None,
    registry: AdapterRegistry | None = None,
) -> DeepPHYBenchmarkSpec:
    adapter_registry = registry or AdapterRegistry()
    adapter = adapter_registry.load(env_id)

    if env_id == "phyre":
        within_tasks = adapter.list_tasks(
            eval_setup="ball_within_template",
            fold_id=0,
            eval_type="test",
        )
        cross_tasks = adapter.list_tasks(
            eval_setup="ball_cross_template",
            fold_id=0,
            eval_type="test",
        )
        if len(within_tasks) != 500 or len(cross_tasks) != 500:
            raise RuntimeError(
                "PHYRE benchmark inventory does not match the paper-defined 500 within-template + 500 cross-template test tasks"
            )
        tasks = tuple(
            [DeepPHYBenchmarkTask(task_id=item, adapter_kwargs={"eval_setup": "ball_within_template", "fold_id": 0, "eval_type": "test"}) for item in within_tasks]
            + [DeepPHYBenchmarkTask(task_id=item, adapter_kwargs={"eval_setup": "ball_cross_template", "fold_id": 0, "eval_type": "test"}) for item in cross_tasks]
        )
        spec = DeepPHYBenchmarkSpec(
            env_id="phyre",
            task_count=len(tasks),
            benchmark_kwargs={"max_attempts": 10},
            tasks=tasks,
            label=summarize_deepphy_benchmark("phyre", preset="full").label,
            paper_note=summarize_deepphy_benchmark("phyre", preset="full").paper_note,
        )
        return _apply_benchmark_preset(
            spec,
            preset,
            task_limit=task_limit,
            benchmark_seed=benchmark_seed,
        )

    if env_id == "iphyre":
        tasks = tuple(
            DeepPHYBenchmarkTask(task_id=item, adapter_kwargs={})
            for item in adapter.list_tasks()
        )
        if len(tasks) != 40:
            raise RuntimeError(
                "I-PHYRE benchmark inventory does not match the paper-defined 40 scenarios"
            )
        spec = DeepPHYBenchmarkSpec(
            env_id="iphyre",
            task_count=len(tasks),
            benchmark_kwargs={"max_attempts": 10},
            tasks=tasks,
            label=summarize_deepphy_benchmark("iphyre", preset="full").label,
            paper_note=summarize_deepphy_benchmark("iphyre", preset="full").paper_note,
        )
        return _apply_benchmark_preset(
            spec,
            preset,
            task_limit=task_limit,
            benchmark_seed=benchmark_seed,
        )

    if env_id == "kinetix":
        task_ids = adapter.list_tasks()
        if len(task_ids) != 74:
            raise RuntimeError(
                "Kinetix benchmark inventory does not match the paper-defined 74 fixed handcrafted test levels"
            )
        tasks = tuple(
            DeepPHYBenchmarkTask(task_id=item, adapter_kwargs={})
            for item in task_ids
        )
        spec = DeepPHYBenchmarkSpec(
            env_id="kinetix",
            task_count=len(tasks),
            benchmark_kwargs={"annotation_mode": "anno", "max_steps": 16},
            tasks=tasks,
            label=summarize_deepphy_benchmark("kinetix", preset="full").label,
            paper_note=summarize_deepphy_benchmark("kinetix", preset="full").paper_note,
        )
        return _apply_benchmark_preset(
            spec,
            preset,
            task_limit=task_limit,
            benchmark_seed=benchmark_seed,
        )

    if env_id == "pooltool":
        tasks = tuple(
            DeepPHYBenchmarkTask(task_id=f"game-{index:03d}", adapter_kwargs={})
            for index in range(100)
        )
        spec = DeepPHYBenchmarkSpec(
            env_id="pooltool",
            task_count=len(tasks),
            benchmark_kwargs={"max_attempts": 8},
            tasks=tasks,
            label=summarize_deepphy_benchmark("pooltool", preset="full").label,
            paper_note=summarize_deepphy_benchmark("pooltool", preset="full").paper_note,
        )
        return _apply_benchmark_preset(
            spec,
            preset,
            task_limit=task_limit,
            benchmark_seed=benchmark_seed,
        )

    raise ValueError(f"unsupported environment: {env_id}")


def evaluate_deepphy_request(
    request: DeepPHYEvaluationRequest,
    *,
    registry: AdapterRegistry | None = None,
    model_responder: ModelResponder | None = None,
    trace_describer: TraceDescriber | None = None,
) -> DeepPHYTextEvaluationReport:
    adapter_registry = registry or AdapterRegistry()
    spec = describe_deepphy_benchmark(
        request.env_id,
        preset=request.benchmark_preset,
        task_limit=request.task_limit,
        benchmark_seed=request.benchmark_seed,
        registry=adapter_registry,
    )
    run_id = request.run_id or default_run_id()
    report_root = Path(request.report_root) if request.report_root else None

    pattern_run_dir = (
        str(resolve_pattern_run_dir(request.pattern_run_dir))
        if request.pattern_run_dir is not None
        else None
    )
    pattern_library_path = (
        str(resolve_pattern_library_path(request.pattern_library_path))
        if request.pattern_library_path is not None
        else None
    )

    results: list[DeepPHYTextEvaluationResult | None] = [None] * len(spec.tasks)
    benchmark_kwargs = {**spec.benchmark_kwargs, **request.benchmark_kwargs}
    tasks_to_run: list[tuple[int, DeepPHYBenchmarkTask, Path | None, str | None]] = []
    for index, item in enumerate(spec.tasks):
        output_root = None
        output_root_path: Path | None = None
        if report_root is not None:
            safe_task_id = item.task_id.replace("/", "__").replace(":", "__")
            output_root_path = report_root / "tasks" / request.env_id / safe_task_id
            output_root = str(output_root_path)
            resumed_result = _load_completed_task_result(
                output_root_path,
                env_id=request.env_id,
                benchmark_kwargs=benchmark_kwargs,
            )
            if resumed_result is not None:
                results[index] = resumed_result
                continue
        tasks_to_run.append((index, item, output_root_path, output_root))

    def _run_task(
        index: int,
        task: DeepPHYBenchmarkTask,
        output_root_path: Path | None,
        output_root: str | None,
    ) -> tuple[int, DeepPHYTextEvaluationResult]:
        single_task_request = DeepPHYTextEvaluationRequest(
            env_id=request.env_id,
            task_id=task.task_id,
            model_name=request.model_name,
            prompt_format=request.prompt_format,
            description_mode=request.description_mode,
            pattern_run_dir=pattern_run_dir,
            pattern_library_path=pattern_library_path,
            evidence_relpath=request.evidence_relpath,
            run_id=run_id,
            output_root=output_root,
            report_root=str(report_root) if report_root is not None else None,
            capture_visuals=request.capture_visuals,
            benchmark_kwargs=benchmark_kwargs,
            adapter_kwargs={**request.adapter_kwargs, **task.adapter_kwargs},
        )
        try:
            if request.env_id in _SUBPROCESS_ISOLATED_ENVS:
                task_result = _run_task_in_subprocess(single_task_request)
            else:
                task_result = evaluate_deepphy_text_request(
                    single_task_request,
                    registry=adapter_registry,
                    model_responder=model_responder,
                    trace_describer=trace_describer,
                )
        except Exception as exc:
            task_result = DeepPHYTextEvaluationResult(
                run_id=run_id,
                env_id=request.env_id,
                task_id=task.task_id,
                model_name=request.model_name,
                prompt_format=request.prompt_format,
                description_mode=request.description_mode,
                initial_image_path="",
                pattern_run_dir=pattern_run_dir,
                pattern_library_path=pattern_library_path,
                records=[],
                success=False,
                terminal_status="BENCHMARK_ERROR",
                native_metrics={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                output_root=output_root,
                report_root=str(report_root) if report_root is not None else None,
                total_duration_seconds=None,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
        if output_root_path is not None:
            _write_completed_task_result(output_root_path, task_result)
        return index, task_result

    max_workers = max(1, int(request.workers or 1))
    if request.env_id in _SUBPROCESS_ISOLATED_ENVS:
        # Kinetix retains substantial JAX/XLA state over the life of a process.
        # Run one benchmark task at a time in a short-lived subprocess so memory
        # is returned to the OS after every task instead of accumulating.
        max_workers = 1
    if tasks_to_run:
        if max_workers <= 1:
            for index, item, output_root_path, output_root in tasks_to_run:
                _, task_result = _run_task(index, item, output_root_path, output_root)
                results[index] = task_result
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_run_task, index, item, output_root_path, output_root)
                    for index, item, output_root_path, output_root in tasks_to_run
                ]
                for future in concurrent.futures.as_completed(futures):
                    index, task_result = future.result()
                    results[index] = task_result

    return build_deepphy_text_report(
        run_id=run_id,
        results=[result for result in results if result is not None],
        report_root=str(report_root) if report_root is not None else None,
        pattern_run_dir=pattern_run_dir,
    )


def _run_task_in_subprocess(
    request: DeepPHYTextEvaluationRequest,
) -> DeepPHYTextEvaluationResult:
    worker_env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    existing_pythonpath = worker_env.get("PYTHONPATH", "").strip()
    worker_env["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else source_root
    )
    worker_python = _PROJECT_ROOT / default_interpreter_relpath(request.env_id)
    if not worker_python.is_file():
        worker_python = Path(sys.executable)
    with tempfile.TemporaryDirectory(prefix="deepphy-task-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        request_path = temp_dir / "request.json"
        result_path = temp_dir / "result.json"
        request_path.write_text(
            json.dumps(request.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                str(worker_python),
                "-m",
                "deepphy_eval.task_worker",
                "--request",
                str(request_path),
                "--result",
                str(result_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=worker_env,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            raise RuntimeError(
                "Kinetix task subprocess failed"
                + (f": {stderr}" if stderr else "")
                + (f" | stdout: {stdout}" if stdout else "")
            )
        return DeepPHYTextEvaluationResult.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )


def _apply_benchmark_preset(
    spec: DeepPHYBenchmarkSpec,
    preset: Literal["full", "quick-eval", "test"],
    *,
    task_limit: int | None = None,
    benchmark_seed: int | None = None,
) -> DeepPHYBenchmarkSpec:
    if preset in {"full", "quick-eval"}:
        if isinstance(task_limit, int) and task_limit > 0:
            if preset == "quick-eval" and benchmark_seed is not None:
                sample_size = min(task_limit, len(spec.tasks))
                limited_tasks = tuple(
                    random.Random(benchmark_seed).sample(list(spec.tasks), sample_size)
                )
            else:
                limited_tasks = spec.tasks[:task_limit]
            if len(limited_tasks) != len(spec.tasks):
                return DeepPHYBenchmarkSpec(
                    env_id=spec.env_id,
                    task_count=len(limited_tasks),
                    benchmark_kwargs=dict(spec.benchmark_kwargs),
                    tasks=tuple(limited_tasks),
                    label=spec.label,
                    paper_note=(
                        f"{spec.paper_note} This run is capped to the first "
                        f"{len(limited_tasks)} benchmark tasks."
                    ),
                )
        return spec
    if preset == "test":
        sampled_task = random.Random(0).choice(list(spec.tasks))
        overview = summarize_deepphy_benchmark(spec.env_id, preset="test")
        return DeepPHYBenchmarkSpec(
            env_id=spec.env_id,
            task_count=1,
            benchmark_kwargs=dict(spec.benchmark_kwargs),
            tasks=(sampled_task,),
            label=overview.label,
            paper_note=overview.paper_note,
        )
    raise ValueError(f"unsupported benchmark preset: {preset}")


def _task_result_path(output_root: Path) -> Path:
    return output_root / _TASK_RESULT_FILENAME


def _load_completed_task_result(
    output_root: Path,
    *,
    env_id: str,
    benchmark_kwargs: dict[str, Any],
) -> DeepPHYTextEvaluationResult | None:
    result_path = _task_result_path(output_root)
    if not result_path.is_file():
        return None
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed DeepPHY task result at {result_path}: {exc}") from exc
    result = DeepPHYTextEvaluationResult.model_validate(payload)
    if not _is_resume_complete_result(result, env_id=env_id, benchmark_kwargs=benchmark_kwargs):
        return None
    return result


def _is_resume_complete_result(
    result: DeepPHYTextEvaluationResult,
    *,
    env_id: str,
    benchmark_kwargs: dict[str, Any],
) -> bool:
    if bool(result.success):
        return True

    terminal_status = str(result.terminal_status or "").strip().upper()
    record_count = len(result.records or [])
    normalized_env = str(env_id or "").strip().lower()

    if normalized_env in {"phyre", "iphyre"}:
        max_attempts = max(1, int(benchmark_kwargs.get("max_attempts", 10)))
        return record_count >= max_attempts

    if normalized_env == "pooltool":
        max_attempts = max(1, int(benchmark_kwargs.get("max_attempts", 8)))
        return terminal_status == "NO_TARGET_BALLS_LEFT" or record_count >= max_attempts

    if normalized_env == "kinetix":
        max_steps = max(1, int(benchmark_kwargs.get("max_steps", 16)))
        return record_count >= max_steps

    # Unknown envs: preserve historical behavior (presence of task_result.json implies complete).
    return True


def _write_completed_task_result(
    output_root: Path,
    result: DeepPHYTextEvaluationResult,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    def _json_fallback(value: object) -> object:
        if value is Ellipsis:
            return "..."
        return repr(value)
    _task_result_path(output_root).write_text(
        json.dumps(
            result.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            default=_json_fallback,
        )
        + "\n",
        encoding="utf-8",
    )
