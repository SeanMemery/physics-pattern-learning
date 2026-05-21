from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from shutil import rmtree
from typing import Any

from adapters.registry import AdapterRegistry
from debug import debug_log, set_debug_log_file
from media import write_image_artifact, write_video_artifact
from paths import get_artifact_layout
from phyre_video import PHYRE_SIMULATION_FPS, phyre_video_fps
from runtime.envs import resolve_manifest_interpreter
from runtime.isolation import pooltool_source_root
from runtime.manifests import load_manifest_map
from runtime.models import AvailabilityStatus
from schema import CanonicalTrace
from traces import make_attachment
from traces.quality import TraceQualityReport, validate_trace_quality


ACTIVE_ENVS = ("phyre", "iphyre", "kinetix", "pooltool")
HELPER_ENVS = {"iphyre", "pooltool"}
PHYRE_REVIEW_VIDEO_SECONDS = 5.0
PHYRE_REVIEW_VIDEO_MAX_FRAMES = 50
PHYRE_HELPER_TIMEOUT_SECONDS = 300
VENV_ROOT_RELPATH = Path("venv")
KINETIX_WORKER_BATCH_SIZE = 4
KINETIX_WORKER_ENV = {
    "JAX_PLATFORMS": "cpu",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "JAX_ENABLE_COMPILATION_CACHE": "true",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "all",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
    "MALLOC_ARENA_MAX": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_force_host_platform_device_count=1",
    "DEEPPHY_KINETIX_MAX_MEMORY_BYTES": str(4 * 1024 * 1024 * 1024),
}


@dataclass
class EnvironmentGenerationReport:
    env_id: str
    status: str
    generated: int = 0
    saved_dir: str | None = None
    blocked_reason: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class TraceGenerationRun:
    output_root: Path
    per_environment: list[EnvironmentGenerationReport]


@dataclass(frozen=True)
class TraceSampleRequest:
    sample_id: int
    absolute_sample_index: int


def default_output_root() -> Path:
    layout = get_artifact_layout()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return layout.data_root / "generated_traces" / timestamp


def generate_traces(
    *,
    limit: int = 100,
    env_ids: list[str] | None = None,
    output_root: Path | None = None,
    force: bool = False,
    sample_offset: int = 0,
) -> TraceGenerationRun:
    root = output_root or default_output_root()
    root.mkdir(parents=True, exist_ok=True)
    set_debug_log_file(root / "debug.log")
    debug_log(
        "generation.generate_traces.start",
        limit=limit,
        env_ids=env_ids,
        output_root=output_root,
        force=force,
        sample_offset=sample_offset,
    )
    registry = AdapterRegistry()
    chosen_envs = env_ids or list(ACTIVE_ENVS)
    reports: list[EnvironmentGenerationReport] = []
    project_root = get_artifact_layout().project_root

    for env_id in chosen_envs:
        debug_log("generation.env.start", env_id=env_id, root=root)
        env_dir = root if len(chosen_envs) == 1 else root / env_id
        env_dir.mkdir(parents=True, exist_ok=True)
        adapter = None if env_id == "phyre" else registry.load(env_id)

        if env_id == "phyre":
            try:
                tasks = _list_phyre_tasks(project_root)
            except Exception as error:
                debug_log("generation.env.list_tasks_failed", env_id=env_id, error=str(error))
                reports.append(
                    EnvironmentGenerationReport(
                        env_id=env_id,
                        status="blocked",
                        saved_dir=str(env_dir),
                        blocked_reason=f"Failed to prepare PHYRE helper runtime: {type(error).__name__}: {error}",
                    )
                )
                continue
        else:
            assert adapter is not None
            probe = adapter.probe()
            debug_log("generation.env.probe", env_id=env_id, availability=probe.availability.value)
            if probe.availability is not AvailabilityStatus.READY and not force:
                reports.append(
                    EnvironmentGenerationReport(
                        env_id=env_id,
                        status="blocked",
                        saved_dir=str(env_dir),
                        blocked_reason=_format_probe_blockers(probe),
                    )
                )
                continue

            try:
                if env_id in HELPER_ENVS:
                    tasks = _list_helper_tasks(project_root=project_root, env_id=env_id)
                else:
                    tasks = adapter.list_tasks()
            except Exception as error:
                debug_log("generation.env.list_tasks_failed", env_id=env_id, error=str(error))
                reports.append(
                    EnvironmentGenerationReport(
                        env_id=env_id,
                        status="blocked",
                        saved_dir=str(env_dir),
                        blocked_reason=f"Failed to list tasks: {type(error).__name__}: {error}",
                    )
                )
                continue

        saved = 0
        env_report = EnvironmentGenerationReport(env_id=env_id, status="completed", saved_dir=str(env_dir))
        debug_log("generation.env.tasks_loaded", env_id=env_id, task_count=len(tasks))
        if not tasks:
            env_report.status = "failed"
            env_report.errors.append("Environment returned no tasks to generate.")
            reports.append(env_report)
            continue

        if env_id == "kinetix":
            requests = [
                TraceSampleRequest(sample_id=index, absolute_sample_index=sample_offset + index)
                for index in range(limit)
            ]
            _generate_kinetix_batches(
                project_root=project_root,
                tasks=tasks,
                requests=requests,
                env_dir=env_dir,
                env_report=env_report,
                reset_existing=True,
            )
            if env_report.generated < limit:
                env_report.warnings.append(
                    f"saved {env_report.generated} of requested {limit} traces via bounded Kinetix workers"
                )
            reports.append(env_report)
            continue

        ordered_tasks = _ordered_task_pool(tasks, env_id=env_id, split_name="adhoc")
        max_attempt_cursor = max(limit * 5, len(ordered_tasks))
        attempt_cursor = 0
        while saved < limit and attempt_cursor < max_attempt_cursor:
            absolute_sample_index = sample_offset + attempt_cursor
            sample_dir = env_dir / f"sample_{saved:03d}"
            work_dir = _prepare_sample_work_dir(sample_dir)
            try:
                task_id, trace, quality = _generate_candidate_trace(
                    env_id=env_id,
                    project_root=project_root,
                    adapter=adapter,
                    ordered_tasks=ordered_tasks,
                    absolute_sample_index=absolute_sample_index,
                    sample_index=saved,
                    sample_dir=work_dir,
                )
                suffix = f":sample-{saved:03d}"
                if not trace.trace_id.endswith(suffix):
                    trace = trace.model_copy(update={"trace_id": f"{trace.trace_id}{suffix}"})
                _write_quality_report(work_dir / "quality.json", quality, trace=trace)
                env_report.warnings.extend(_prefix_warnings(task_id, saved, quality))
                trace_path = work_dir / "trace.json"
                with trace_path.open("w", encoding="utf-8") as handle:
                    json.dump(trace.model_dump(mode="json"), handle, indent=2)
                _commit_sample_work_dir(work_dir, sample_dir)
                saved += 1
                debug_log("generation.sample.saved", env_id=env_id, task_id=task_id, sample_dir=sample_dir)
            except Exception as error:
                _cleanup_sample_work_dir(work_dir)
                debug_log("generation.sample.failed", env_id=env_id, task_id=task_id, error=str(error))
                env_report.errors.append(
                    f"attempt_{absolute_sample_index:06d} {task_id}: {type(error).__name__}: {error}"
                )
            attempt_cursor += 1
        env_report.generated = saved
        if saved == 0 and env_report.errors:
            env_report.status = "failed"
        if saved < limit:
            env_report.warnings.append(
                f"saved {saved} of requested {limit} traces after {attempt_cursor} attempts"
            )
        reports.append(env_report)

    report_path = root / "generation_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "output_root": str(root),
                "per_environment": [report.__dict__ for report in reports],
            },
            handle,
            indent=2,
        )
    return TraceGenerationRun(output_root=root, per_environment=reports)


def generate_trace_samples(
    *,
    sample_requests: dict[str, list[TraceSampleRequest]],
    output_root: Path | None = None,
    force: bool = False,
    reset_existing: bool = False,
    flatten_single_env: bool = False,
) -> TraceGenerationRun:
    debug_log(
        "generation.generate_trace_samples.start",
        env_ids=list(sample_requests),
        output_root=output_root,
        force=force,
        reset_existing=reset_existing,
        flatten_single_env=flatten_single_env,
    )
    registry = AdapterRegistry()
    root = output_root or default_output_root()
    root.mkdir(parents=True, exist_ok=True)
    reports: list[EnvironmentGenerationReport] = []
    project_root = get_artifact_layout().project_root

    for env_id, requests in sample_requests.items():
        env_dir = root if flatten_single_env and len(sample_requests) == 1 else root / env_id
        env_dir.mkdir(parents=True, exist_ok=True)
        adapter = None if env_id == "phyre" else registry.load(env_id)

        if env_id == "phyre":
            try:
                tasks = _list_phyre_tasks(project_root)
            except Exception as error:
                reports.append(
                    EnvironmentGenerationReport(
                        env_id=env_id,
                        status="blocked",
                        saved_dir=str(env_dir),
                        blocked_reason=f"Failed to prepare PHYRE helper runtime: {type(error).__name__}: {error}",
                    )
                )
                continue
        else:
            assert adapter is not None
            probe = adapter.probe()
            if probe.availability is not AvailabilityStatus.READY and not force:
                reports.append(
                    EnvironmentGenerationReport(
                        env_id=env_id,
                        status="blocked",
                        saved_dir=str(env_dir),
                        blocked_reason=_format_probe_blockers(probe),
                    )
                )
                continue
            try:
                if env_id in HELPER_ENVS:
                    tasks = _list_helper_tasks(project_root=project_root, env_id=env_id)
                else:
                    tasks = adapter.list_tasks()
            except Exception as error:
                reports.append(
                    EnvironmentGenerationReport(
                        env_id=env_id,
                        status="blocked",
                        saved_dir=str(env_dir),
                        blocked_reason=f"Failed to list tasks: {type(error).__name__}: {error}",
                    )
                )
                continue

        env_report = EnvironmentGenerationReport(env_id=env_id, status="completed", saved_dir=str(env_dir))
        if not tasks:
            env_report.status = "failed"
            env_report.errors.append("Environment returned no tasks to generate.")
            reports.append(env_report)
            continue

        ordered_tasks = _ordered_task_pool(tasks, env_id=env_id, split_name=root.name)

        if env_id == "kinetix":
            pending_requests: list[TraceSampleRequest] = []
            for request in sorted(requests, key=lambda item: item.sample_id):
                sample_dir = env_dir / f"sample_{request.sample_id:03d}"
                if sample_dir.exists() and not reset_existing:
                    env_report.warnings.append(f"sample_{request.sample_id:03d} already exists; skipped")
                    continue
                pending_requests.append(request)
            _generate_kinetix_batches(
                project_root=project_root,
                tasks=tasks,
                requests=pending_requests,
                env_dir=env_dir,
                env_report=env_report,
                reset_existing=reset_existing,
            )
            if env_report.generated == 0 and env_report.errors:
                env_report.status = "failed"
            reports.append(env_report)
            continue

        for request in sorted(requests, key=lambda item: item.sample_id):
            sample_dir = env_dir / f"sample_{request.sample_id:03d}"
            if sample_dir.exists() and not reset_existing and _sample_dir_is_complete(sample_dir):
                env_report.warnings.append(f"sample_{request.sample_id:03d} already exists; skipped")
                continue
            work_dir = _prepare_sample_work_dir(sample_dir)

            try:
                task_id, trace, quality = _generate_candidate_trace(
                    env_id=env_id,
                    project_root=project_root,
                    adapter=adapter,
                    ordered_tasks=ordered_tasks,
                    absolute_sample_index=request.absolute_sample_index,
                    sample_index=request.sample_id,
                    sample_dir=work_dir,
                )
                suffix = f":sample-{request.sample_id:03d}"
                if not trace.trace_id.endswith(suffix):
                    trace = trace.model_copy(update={"trace_id": f"{trace.trace_id}{suffix}"})
                _write_quality_report(work_dir / "quality.json", quality, trace=trace)

                env_report.warnings.extend(_prefix_warnings(task_id, request.sample_id, quality))
                with (work_dir / "trace.json").open("w", encoding="utf-8") as handle:
                    json.dump(trace.model_dump(mode="json"), handle, indent=2)
                _commit_sample_work_dir(work_dir, sample_dir)
                env_report.generated += 1
                debug_log(
                    "generation.sample.saved",
                    env_id=env_id,
                    task_id=task_id,
                    sample_dir=sample_dir,
                    sampling_strategy=(
                        trace.initial_state.get("sampling_strategy")
                        if isinstance(trace.initial_state, dict)
                        else None
                    ),
                    timestep_count=len(trace.timesteps),
                    attachment_count=len(trace.attachments),
                    usable=quality.usable,
                )
            except Exception as error:
                _cleanup_sample_work_dir(work_dir)
                failed_task_id = ordered_tasks[request.absolute_sample_index % len(ordered_tasks)]
                debug_log(
                    "generation.sample.failed",
                    env_id=env_id,
                    task_id=failed_task_id,
                    sample_dir=sample_dir,
                    error=str(error),
                )
                env_report.errors.append(
                    f"attempt_{request.absolute_sample_index:06d} {failed_task_id}: {type(error).__name__}: {error}"
                )

        if env_report.generated == 0 and env_report.errors:
            env_report.status = "failed"
        reports.append(env_report)

    report_path = root / "generation_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "output_root": str(root),
                "per_environment": [report.__dict__ for report in reports],
            },
            handle,
            indent=2,
        )
    return TraceGenerationRun(output_root=root, per_environment=reports)


def heuristic_trace_params(
    env_id: str,
    task_id: str,
    *,
    sample_index: int,
    output_dir: Path,
) -> dict[str, Any]:
    if env_id == "phyre":
        sampling_strategy = "seek_solution" if sample_index % 2 else "informative"
        return {
            "sample_seed": sample_index,
            "sampling_strategy": sampling_strategy,
            "min_dense_frames": 96,
            "max_sample_attempts": 16,
            "solve_sample_attempts": 1024,
            "need_images": True,
        }
    if env_id == "iphyre":
        return {
            "sample_seed": sample_index,
            "sampling_strategy": "seek_solution" if sample_index % 2 else "random",
            "solve_search_attempts": 1024,
            "random_action_count_min": 1,
            "random_action_count_max": 3,
            "capture_visuals": True,
            "output_dir": output_dir,
        }
    if env_id == "kinetix":
        joint_phase = sample_index % 3
        thruster_phase = (sample_index // 3) % 2
        action_sequence = []
        for index in range(12):
            action_sequence.append(
                [
                    (joint_phase + index) % 3,
                    (index + 1) % 3,
                    0,
                    0,
                    thruster_phase if index % 2 == 0 else 0,
                    1 if index % 3 == 0 else 0,
                ]
            )
        return {
            "action_sequence": action_sequence,
            "repeat_count": 8,
            "capture_frames": True,
            "render_stride": 1,
            "video_fps": 8,
            "target_video_seconds": 10.0,
            "media_max_side": 512,
            "output_dir": output_dir,
        }
    if env_id == "pooltool":
        speeds = ["Low", "Medium", "High"]
        strikes = [
            "Middle Spin",
            "Top Spin",
            "Bottom Spin",
            "Top Left Spin",
            "Top Right Spin",
            "Bottom Left Spin",
            "Bottom Right Spin",
        ]
        shot_count = 1 + (sample_index % 3)
        return {
            "shots": [
                {
                    "speed_name": speeds[(sample_index + shot_index) % len(speeds)],
                    "strikespot_name": strikes[(sample_index * 2 + shot_index) % len(strikes)],
                }
                for shot_index in range(shot_count)
            ],
            "continuous_dt": 0.03,
            "capture_frames": False,
            "output_dir": output_dir,
            "layout_seed": sample_index,
            "rack_jitter": 0.08,
        }
    raise KeyError(f"Unsupported environment id for heuristic generation: {env_id}")


def _format_probe_blockers(probe) -> str:
    unmet = [requirement.detail for requirement in probe.requirements if requirement.status.value == "unmet"]
    if unmet:
        return "; ".join(unmet)
    return f"Availability is {probe.availability.value}"


def _safe_name(task_id: str) -> str:
    return task_id.replace("/", "__").replace(":", "__")


def _ordered_task_pool(tasks: list[str], *, env_id: str, split_name: str) -> list[str]:
    if env_id != "phyre" or len(tasks) <= 1:
        return list(tasks)
    seed_material = f"{env_id}:{split_name}:task-pool-v2".encode("utf-8")
    seed = int(hashlib.sha256(seed_material).hexdigest()[:16], 16)
    ordered = list(tasks)
    random.Random(seed).shuffle(ordered)
    return ordered


def _prefix_warnings(task_id: str, sample_index: int, quality: TraceQualityReport) -> list[str]:
    return [f"sample_{sample_index:03d} {task_id}: {warning}" for warning in quality.warnings]


def _write_quality_report(
    path: Path,
    quality: TraceQualityReport,
    *,
    trace: CanonicalTrace | None = None,
) -> None:
    payload = dict(quality.__dict__)
    if trace is not None:
        initial_state = trace.initial_state if isinstance(trace.initial_state, dict) else {}
        sampling_strategy = initial_state.get("sampling_strategy")
        payload["sampling_strategy"] = sampling_strategy
        payload["attempted_solution_search"] = sampling_strategy == "seek_solution"
        if "search_attempts" in initial_state:
            payload["search_attempts"] = initial_state.get("search_attempts")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _reset_output_dir(path: Path) -> None:
    if path.exists():
        rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _sample_dir_is_complete(path: Path) -> bool:
    return (path / "trace.json").is_file() and (path / "quality.json").is_file()


def _prepare_sample_work_dir(sample_dir: Path) -> Path:
    sample_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = sample_dir.parent / f".{sample_dir.name}.tmp-{uuid.uuid4().hex[:8]}"
    _cleanup_sample_work_dir(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def _cleanup_sample_work_dir(path: Path) -> None:
    if path.exists():
        rmtree(path)


def _commit_sample_work_dir(work_dir: Path, sample_dir: Path) -> None:
    if sample_dir.exists():
        rmtree(sample_dir)
    work_dir.rename(sample_dir)


def _generate_candidate_trace(
    *,
    env_id: str,
    project_root: Path,
    adapter: Any,
    ordered_tasks: list[str],
    absolute_sample_index: int,
    sample_index: int,
    sample_dir: Path,
) -> tuple[str, CanonicalTrace, TraceQualityReport]:
    if not ordered_tasks:
        raise RuntimeError(f"No tasks available for {env_id}")

    base_task_index = absolute_sample_index % len(ordered_tasks)
    seed_params = heuristic_trace_params(
        env_id,
        ordered_tasks[base_task_index],
        sample_index=absolute_sample_index,
        output_dir=sample_dir,
    )
    seek_across_tasks = (
        env_id in {"phyre", "iphyre"}
        and str(seed_params.get("sampling_strategy", "random")) == "seek_solution"
    )
    errors: list[str] = []

    max_task_attempts = len(ordered_tasks) if seek_across_tasks else 1
    for task_offset in range(max_task_attempts):
        task_index = (base_task_index + task_offset) % len(ordered_tasks)
        task_id = ordered_tasks[task_index]
        params = heuristic_trace_params(
            env_id,
            task_id,
            sample_index=absolute_sample_index,
            output_dir=sample_dir,
        )
        sampling_strategy = str(params.get("sampling_strategy", "random"))
        if sampling_strategy == "seek_solution":
            params["sample_seed"] = int(params.get("sample_seed", absolute_sample_index)) + task_offset * 1_000_003

        try:
            trace = _run_trace_generation(
                env_id=env_id,
                project_root=project_root,
                adapter=adapter,
                task_id=task_id,
                sample_index=sample_index,
                sample_dir=sample_dir,
                params=params,
            )
        except Exception as error:
            _reset_output_dir(sample_dir)
            errors.append(f"{task_id}: {type(error).__name__}: {error}")
            if sampling_strategy == "seek_solution" and seek_across_tasks:
                continue
            raise

        quality = validate_trace_quality(trace)
        requires_solution = sampling_strategy == "seek_solution" and seek_across_tasks
        if quality.usable and (not requires_solution or bool(trace.outcome.success)):
            return task_id, trace, quality

        _reset_output_dir(sample_dir)
        reason = []
        if not quality.usable:
            reason.extend(quality.errors)
        if requires_solution and not trace.outcome.success:
            reason.append("trace did not solve task")
        debug_log(
            "generation.sample.rejected",
            env_id=env_id,
            task_id=task_id,
            sample_dir=sample_dir,
            sampling_strategy=sampling_strategy,
            quality_errors=quality.errors,
            quality_warnings=quality.warnings,
            outcome_success=trace.outcome.success,
            reason=reason,
        )
        errors.append(f"{task_id}: {'; '.join(reason) if reason else 'rejected trace'}")
        if requires_solution and seek_across_tasks:
            continue
        raise RuntimeError(f"Rejected generated trace for {task_id}: {'; '.join(reason)}")

    raise RuntimeError(
        f"Failed to generate a solved usable {env_id} trace after trying {max_task_attempts} tasks: "
        + " | ".join(errors[:8])
    )


def _run_trace_generation(
    *,
    env_id: str,
    project_root: Path,
    adapter: Any,
    task_id: str,
    sample_index: int,
    sample_dir: Path,
    params: dict[str, Any],
) -> CanonicalTrace:
    if env_id == "phyre":
        debug_log("generation.sample.generate", env_id=env_id, task_id=task_id, sample_dir=sample_dir)
        return _generate_phyre_trace(
            project_root=project_root,
            task_id=task_id,
            sample_index=sample_index,
            output_dir=sample_dir,
            params=params,
        )
    assert adapter is not None
    if env_id in HELPER_ENVS:
        debug_log("generation.sample.generate_helper", env_id=env_id, task_id=task_id, sample_dir=sample_dir)
        return _generate_helper_trace(
            project_root=project_root,
            env_id=env_id,
            task_id=task_id,
            params=params,
        )
    debug_log("generation.sample.generate_direct", env_id=env_id, task_id=task_id, sample_dir=sample_dir)
    return adapter.generate_trace(task_id, **params)


def _list_phyre_tasks(project_root: Path) -> list[str]:
    payload = _run_phyre_helper(
        project_root,
        [
            "list-tasks",
            "--eval-setup",
            "ball_within_template",
            "--fold-id",
            "0",
            "--eval-type",
            "all",
        ],
    )
    return list(json.loads(payload))


def _generate_phyre_trace(
    *,
    project_root: Path,
    task_id: str,
    sample_index: int,
    output_dir: Path,
    params: dict[str, Any],
) -> CanonicalTrace:
    payload = _run_phyre_helper(
        project_root,
        [
            "generate",
            "--task-id",
            task_id,
            "--output-dir",
            str(output_dir),
            "--sample-index",
            str(sample_index),
            "--sample-seed",
            str(int(params.get("sample_seed", sample_index))),
            "--sampling-strategy",
            str(params.get("sampling_strategy", "random")),
            "--eval-setup",
            "ball_within_template",
            "--fold-id",
            "0",
            "--eval-type",
            "all",
            "--min-dense-frames",
            str(int(params.get("min_dense_frames", 64))),
            "--max-sample-attempts",
            str(int(params.get("max_sample_attempts", 8))),
            "--solve-sample-attempts",
            str(int(params.get("solve_sample_attempts", 256))),
        ],
    )
    trace = CanonicalTrace.model_validate(json.loads(payload))
    return _attach_phyre_visual_artifacts(trace, output_dir=output_dir)


def _run_phyre_helper(project_root: Path, args: list[str]) -> str:
    helper_python = resolve_phyre_helper_python(project_root)
    helper_module = "tooling.phyre_trace_helper_py39"
    env = os.environ.copy()
    env["PYTHONPATH"] = _merge_pythonpath(project_root / "src", env.get("PYTHONPATH"))
    env["MPLCONFIGDIR"] = str(project_root / ".runtime-home" / "matplotlib")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    debug_log("generation.phyre_helper.exec", python=helper_python, module=helper_module, args=args)
    try:
        result = subprocess.run(
            [str(helper_python), "-m", helper_module, *args],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(project_root),
            env=env,
            timeout=PHYRE_HELPER_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        debug_log(
            "generation.phyre_helper.timeout",
            args=args,
            timeout_seconds=PHYRE_HELPER_TIMEOUT_SECONDS,
            stdout=(error.stdout or "").strip(),
            stderr=(error.stderr or "").strip(),
        )
        raise RuntimeError(
            "PHYRE helper timed out after "
            f"{PHYRE_HELPER_TIMEOUT_SECONDS}s for args {args}"
        ) from error
    if result.returncode != 0:
        debug_log(
            "generation.phyre_helper.failed",
            args=args,
            returncode=result.returncode,
            stdout=(result.stdout or "").strip(),
            stderr=(result.stderr or "").strip(),
        )
        raise RuntimeError((result.stderr or result.stdout).strip() or "PHYRE helper failed")
    debug_log("generation.phyre_helper.ok", args=args)
    return result.stdout.strip()


def _list_helper_tasks(*, project_root: Path, env_id: str) -> list[str]:
    debug_log("generation.sim_helper.list_tasks.start", env_id=env_id)
    payload = _run_simulator_helper(
        project_root=project_root,
        env_id=env_id,
        args=["list-tasks", "--env", env_id],
    )
    debug_log("generation.sim_helper.list_tasks.done", env_id=env_id)
    return list(json.loads(payload))


def _generate_helper_trace(
    *,
    project_root: Path,
    env_id: str,
    task_id: str,
    params: dict[str, Any],
) -> CanonicalTrace:
    debug_log("generation.sim_helper.generate.start", env_id=env_id, task_id=task_id)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
        json.dump(_json_ready(params), handle)
        payload_path = Path(handle.name)
    try:
        payload = _run_simulator_helper(
            project_root=project_root,
            env_id=env_id,
            args=[
                "generate",
                "--env",
                env_id,
                "--task-id",
                task_id,
                "--params-json",
                str(payload_path),
            ],
        )
    finally:
        payload_path.unlink(missing_ok=True)
    debug_log("generation.sim_helper.generate.done", env_id=env_id, task_id=task_id)
    return CanonicalTrace.model_validate(json.loads(payload))


def _run_simulator_helper(
    *,
    project_root: Path,
    env_id: str,
    args: list[str],
) -> str:
    helper_python = resolve_environment_python(project_root, env_id)
    helper_module = "tooling.simulator_trace_helper"
    env = os.environ.copy()
    python_paths = [project_root / "src"]
    if env_id == "pooltool":
        python_paths.append(pooltool_source_root(project_root))
    env["PYTHONPATH"] = _merge_pythonpath(python_paths, env.get("PYTHONPATH"))
    if env_id == "iphyre":
        env.setdefault("SDL_VIDEODRIVER", "dummy")
        env.setdefault("SDL_AUDIODRIVER", "dummy")
        env.setdefault("XDG_RUNTIME_DIR", "/tmp")
    debug_log("generation.sim_helper.exec", env_id=env_id, python=helper_python, module=helper_module, args=args)
    result = subprocess.run(
        [str(helper_python), "-m", helper_module, *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(project_root),
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip() or f"{env_id} helper failed")
    debug_log("generation.sim_helper.ok", env_id=env_id, args=args)
    return result.stdout.strip()


def resolve_phyre_helper_python(project_root: Path) -> Path:
    helper_python = resolve_environment_python(project_root, "phyre")
    if not helper_python.is_file():
        raise FileNotFoundError(
            "Missing PHYRE helper interpreter at "
            f"{helper_python}. Expected the PHYRE virtualenv at venv/phyre to provide Python 3.9."
        )
    return helper_python


def resolve_environment_python(project_root: Path, env_id: str) -> Path:
    manifests = load_manifest_map(root=project_root / "configs")
    manifest = manifests[env_id]
    interpreter = resolve_manifest_interpreter(manifest, project_root=project_root)
    if interpreter is None:
        return project_root / VENV_ROOT_RELPATH / env_id / "bin" / "python"
    return interpreter


def _attach_phyre_visual_artifacts(trace: CanonicalTrace, *, output_dir: Path) -> CanonicalTrace:
    attachments = []
    initial_scene_path = output_dir / "initial_scene.png"
    initial_metadata = None
    if initial_scene_path.is_file():
        initial_metadata = write_image_artifact(initial_scene_path, initial_scene_path, max_side=512)

    for attachment in trace.attachments:
        if attachment.kind == "rendered_frame":
            continue
        if attachment.kind == "initial_scene" and initial_scene_path.is_file():
            attachments.append(
                make_attachment("initial_scene", initial_scene_path, label=attachment.label, metadata=initial_metadata)
            )
        else:
            attachments.append(attachment)

    if initial_scene_path.is_file() and not any(attachment.kind == "initial_scene" for attachment in attachments):
        attachments.append(
            make_attachment("initial_scene", initial_scene_path, label="Initial scene", metadata=initial_metadata)
        )

    frame_paths = sorted(output_dir.glob("frame_*.png"))
    if frame_paths:
        selected_frame_paths = select_review_video_frames(
            frame_paths,
            target_frame_count=PHYRE_REVIEW_VIDEO_MAX_FRAMES,
        )
        video_path = output_dir / "simulation.mp4"
        simulation_fps = int(
            trace.initial_state.get("simulation_fps", PHYRE_SIMULATION_FPS)
            if isinstance(trace.initial_state, dict)
            else PHYRE_SIMULATION_FPS
        )
        simulation_frame_count = max(1, len(trace.timesteps))
        fps = phyre_video_fps(
            len(selected_frame_paths),
            simulation_frame_count=simulation_frame_count,
            simulation_fps=simulation_fps,
            success=trace.outcome.success,
        )
        metadata = write_video_artifact(selected_frame_paths, video_path, fps=fps, max_side=512)
        metadata.update(
            {
                "duration_seconds": round(metadata["frame_count"] / max(metadata["fps"], 1), 3),
                "target_duration_seconds": round(simulation_frame_count / max(simulation_fps, 1), 3),
                "frame_selection": "uniform_subsample",
            }
        )
        attachments.append(
            make_attachment("simulation_video", video_path, label="Simulation video", metadata=metadata)
        )
        for path in frame_paths:
            path.unlink(missing_ok=True)

    return trace.model_copy(update={"attachments": attachments})


def select_review_video_frames(items: list[Any], *, target_frame_count: int) -> list[Any]:
    if len(items) <= target_frame_count:
        return list(items)
    if target_frame_count <= 1:
        return [items[0]]

    last_index = len(items) - 1
    selected: list[Any] = []
    seen_indices: set[int] = set()
    for output_index in range(target_frame_count):
        source_index = round(output_index * last_index / (target_frame_count - 1))
        if source_index in seen_indices:
            continue
        seen_indices.add(source_index)
        selected.append(items[source_index])
    return selected


def review_video_fps(frame_count: int, *, target_duration_seconds: float) -> int:
    return max(1, int(round(frame_count / max(target_duration_seconds, 1.0))))


def _generate_kinetix_batches(
    *,
    project_root: Path,
    tasks: list[str],
    requests: list[TraceSampleRequest],
    env_dir: Path,
    env_report: EnvironmentGenerationReport,
    reset_existing: bool,
) -> None:
    for batch in _chunk_requests(requests, batch_size=KINETIX_WORKER_BATCH_SIZE):
        payload_samples: list[dict[str, Any]] = []
        for request in batch:
            sample_dir = env_dir / f"sample_{request.sample_id:03d}"
            if reset_existing:
                _reset_output_dir(sample_dir)
            else:
                sample_dir.mkdir(parents=True, exist_ok=True)
            task_id = tasks[request.absolute_sample_index % len(tasks)]
            params = heuristic_trace_params(
                "kinetix",
                task_id,
                sample_index=request.absolute_sample_index,
                output_dir=sample_dir,
            )
            payload_samples.append(
                {
                    "sample_id": request.sample_id,
                    "absolute_sample_index": request.absolute_sample_index,
                    "task_id": task_id,
                    "sample_dir": str(sample_dir),
                    "params": _json_ready(params),
                }
            )

        payload = _run_kinetix_helper_batch(project_root=project_root, samples=payload_samples)
        for sample in payload.get("samples", []):
            warnings = list(sample.get("warnings", []))
            error = sample.get("error")
            if sample.get("status") == "generated":
                env_report.generated += 1
                env_report.warnings.extend(warnings)
            else:
                task_id = sample.get("task_id", "unknown-task")
                sample_id = int(sample.get("sample_id", -1))
                if warnings:
                    env_report.warnings.extend(warnings)
                env_report.errors.append(
                    f"attempt_{int(sample.get('absolute_sample_index', sample_id)):06d} "
                    f"{task_id}: {error or 'unknown Kinetix helper failure'}"
                )


def _chunk_requests(
    requests: list[TraceSampleRequest],
    *,
    batch_size: int,
) -> list[list[TraceSampleRequest]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [
        requests[index : index + batch_size]
        for index in range(0, len(requests), batch_size)
    ]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _run_kinetix_helper_batch(
    *,
    project_root: Path,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    helper_module = "tooling.kinetix_trace_helper"
    env = os.environ.copy()
    env.update(KINETIX_WORKER_ENV)
    env["PYTHONPATH"] = _merge_pythonpath(project_root / "src", env.get("PYTHONPATH"))
    env["JAX_COMPILATION_CACHE_DIR"] = str(project_root / ".runtime-home" / "jax-cache" / "kinetix")

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
        json.dump({"samples": samples}, handle)
        payload_path = Path(handle.name)

    try:
        helper_python = resolve_environment_python(project_root, "kinetix")
        result = subprocess.run(
            [str(helper_python), "-m", helper_module, "generate-batch", "--batch-json", str(payload_path)],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(project_root),
            env=env,
        )
    finally:
        payload_path.unlink(missing_ok=True)

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        message = stderr or stdout or "Kinetix helper failed"
        raise RuntimeError(message)

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Kinetix helper returned invalid JSON: "
            + ((result.stdout or result.stderr or "").strip()[:400] or "<empty>")
        ) from error


def _merge_pythonpath(project_paths: Path | list[Path], existing: str | None) -> str:
    if isinstance(project_paths, Path):
        paths = [project_paths]
    else:
        paths = list(project_paths)
    parts = [str(path) for path in paths]
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)
