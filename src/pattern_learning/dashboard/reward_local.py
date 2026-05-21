from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from deepphy_eval import default_run_id
from dashboard.actions import (
    PooltoolRewardOptimizationActionRequest,
    PooltoolRewardValueModelActionRequest,
    python_executable_for_env,
)
from paths import ArtifactLayout, get_artifact_layout
from pattern_library_paths import resolve_pattern_run_dir
from reward_programs.pooltool import (
    OPTIMIZATION_EXPERIMENT_DIRNAME,
    REWARD_PROGRAM_ROOT_DIRNAME,
    VALUE_MODEL_EXPERIMENT_DIRNAME,
    run_pooltool_optimization_experiment,
    run_pooltool_value_model_experiment,
)


RewardRunKind = Literal["optimization", "value_model"]
RewardRunStatus = Literal["starting", "running", "succeeded", "failed"]

LOCAL_RUN_FILENAME = "local_run.json"
PROGRESS_FILENAME = "progress.json"
STDOUT_FILENAME = "stdout.log"
STDERR_FILENAME = "stderr.log"
REWARD_LOCK_FILENAME = ".lock"
REWARD_PROGRESS_PATH_ENV = "PATTERN_REWARD_PROGRESS_PATH"
REWARD_RUN_ROOT_ENV = "PATTERN_REWARD_RUN_ROOT"
REWARD_LOCK_PATH_ENV = "PATTERN_REWARD_LOCK_PATH"
KIND_TO_DIRNAME: dict[RewardRunKind, str] = {
    "optimization": OPTIMIZATION_EXPERIMENT_DIRNAME,
    "value_model": VALUE_MODEL_EXPERIMENT_DIRNAME,
}
KIND_TO_CLI_NAME: dict[RewardRunKind, str] = {
    "optimization": "optimization",
    "value_model": "value-model",
}
_MANUAL_STDOUT_TEE: "_TeeStream | None" = None
_MANUAL_STDERR_TEE: "_TeeStream | None" = None


@dataclass(frozen=True)
class LocalRewardRunRecord:
    run_id: str
    kind: RewardRunKind
    root: Path
    status: str
    started_at: str | None
    updated_at: str | None
    finished_at: str | None
    pid: int | None
    source_run_dir: str
    request: dict[str, Any]
    stdout_path: Path
    stderr_path: Path
    progress_path: Path
    summary_path: Path | None
    recent_output: list[str]
    progress: dict[str, Any]
    error: str | None


class _TeeStream:
    def __init__(self, primary, secondary) -> None:  # type: ignore[no-untyped-def]
        self._primary = primary
        self._secondary = secondary

    def write(self, data: str) -> int:
        self._primary.write(data)
        self._secondary.write(data)
        return len(data)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._primary, "isatty", lambda: False)())

    def fileno(self) -> int:
        return int(self._primary.fileno())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def reward_program_root(*, layout: ArtifactLayout | None = None) -> Path:
    resolved_layout = layout or get_artifact_layout()
    return resolved_layout.data_root / "patterns" / "pooltool" / REWARD_PROGRAM_ROOT_DIRNAME


def reward_kind_root(kind: RewardRunKind, *, layout: ArtifactLayout | None = None) -> Path:
    return reward_program_root(layout=layout) / KIND_TO_DIRNAME[kind]


def reward_run_root(
    kind: RewardRunKind,
    run_id: str,
    *,
    layout: ArtifactLayout | None = None,
) -> Path:
    return reward_kind_root(kind, layout=layout) / str(run_id).strip()


def reward_lock_path(*, layout: ArtifactLayout | None = None) -> Path:
    return reward_program_root(layout=layout) / REWARD_LOCK_FILENAME


def local_run_manifest_path(run_root: Path) -> Path:
    return run_root / LOCAL_RUN_FILENAME


def local_progress_path(run_root: Path) -> Path:
    return run_root / PROGRESS_FILENAME


def _pid_is_alive(pid: int | None) -> bool:
    if pid is None or int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _tail_lines(path: Path, *, limit: int = 120) -> list[str]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-limit:]


def _active_lock_payload(*, layout: ArtifactLayout | None = None) -> dict[str, Any] | None:
    payload = _load_json_file(reward_lock_path(layout=layout))
    if not payload:
        return None
    pid = payload.get("pid")
    try:
        normalized_pid = int(pid)
    except (TypeError, ValueError):
        normalized_pid = 0
    if _pid_is_alive(normalized_pid):
        payload["pid"] = normalized_pid
        return payload
    try:
        reward_lock_path(layout=layout).unlink()
    except OSError:
        pass
    return None


def ensure_reward_launch_available(
    *,
    layout: ArtifactLayout | None = None,
    allow_pid: int | None = None,
) -> None:
    active = _active_lock_payload(layout=layout)
    if active is None:
        return
    if allow_pid is not None and int(active.get("pid") or 0) == int(allow_pid):
        return
    run_id = str(active.get("run_id") or "").strip() or "unknown"
    kind = str(active.get("kind") or "").strip() or "reward"
    raise ValueError(
        f"Reward experiments are local-only and limited to one run at a time. "
        f"Active {kind} run: {run_id}."
    )


def _write_lock(*, kind: RewardRunKind, run_id: str, pid: int, layout: ArtifactLayout | None = None) -> None:
    _write_json(
        reward_lock_path(layout=layout),
        {
            "kind": kind,
            "pid": int(pid),
            "run_id": run_id,
            "updated_at": _utc_now(),
        },
    )


def _clear_lock_if_owned(*, pid: int, layout: ArtifactLayout | None = None) -> None:
    path = reward_lock_path(layout=layout)
    payload = _load_json_file(path)
    try:
        owner_pid = int(payload.get("pid") or 0)
    except (TypeError, ValueError):
        owner_pid = 0
    if owner_pid != int(pid):
        return
    try:
        path.unlink()
    except OSError:
        pass


def _manifest_payload(
    *,
    kind: RewardRunKind,
    run_id: str,
    status: RewardRunStatus,
    source_run_dir: str,
    request: dict[str, Any],
    pid: int | None,
    started_at: str | None,
    updated_at: str | None,
    finished_at: str | None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "kind": kind,
        "status": status,
        "started_at": started_at,
        "updated_at": updated_at,
        "finished_at": finished_at,
        "pid": int(pid) if pid is not None else None,
        "source_run_dir": source_run_dir,
        "request": request,
        "error": error,
    }


def _write_manifest(
    run_root: Path,
    *,
    kind: RewardRunKind,
    run_id: str,
    status: RewardRunStatus,
    source_run_dir: str,
    request: dict[str, Any],
    pid: int | None,
    started_at: str | None,
    finished_at: str | None,
    error: str | None = None,
) -> None:
    now = _utc_now()
    _write_json(
        local_run_manifest_path(run_root),
        _manifest_payload(
            kind=kind,
            run_id=run_id,
            status=status,
            source_run_dir=source_run_dir,
            request=request,
            pid=pid,
            started_at=started_at,
            updated_at=now,
            finished_at=finished_at,
            error=error,
        ),
    )


def _build_reward_local_argv(
    kind: RewardRunKind,
    request: PooltoolRewardOptimizationActionRequest | PooltoolRewardValueModelActionRequest,
) -> list[str]:
    argv = ["-m", "dashboard.reward_local", KIND_TO_CLI_NAME[kind]]
    argv.extend(["--source-run-dir", str(resolve_pattern_run_dir(request.source_run_dir))])
    argv.extend(["--goal-text", str(request.goal_text or "")])
    argv.extend(["--dsl-program", str(request.dsl_program or "")])
    argv.extend(["--ollama-base-url", str(request.ollama_base_url)])
    argv.extend(["--model-timeout", str(int(request.model_timeout))])
    if str(request.model or "").strip():
        argv.extend(["--model", str(request.model).strip()])
    if request.run_id is not None:
        argv.extend(["--run-id", str(request.run_id)])
    if kind == "optimization":
        optimization_request = request
        assert isinstance(optimization_request, PooltoolRewardOptimizationActionRequest)
        argv.extend(["--scene-count", str(int(optimization_request.scene_count))])
        argv.extend(["--seed-count", str(int(optimization_request.seed_count))])
        argv.extend(["--rack-jitter", str(float(optimization_request.rack_jitter))])
        for budget in optimization_request.budgets:
            argv.extend(["--budget", str(int(budget))])
        return argv
    value_request = request
    assert isinstance(value_request, PooltoolRewardValueModelActionRequest)
    argv.extend(["--train-scene-count", str(int(value_request.train_scene_count))])
    argv.extend(["--train-actions-per-scene", str(int(value_request.train_actions_per_scene))])
    argv.extend(["--validation-scene-count", str(int(value_request.validation_scene_count))])
    argv.extend(
        [
            "--validation-candidates-per-scene",
            str(int(value_request.validation_candidates_per_scene)),
        ]
    )
    argv.extend(["--epochs", str(int(value_request.epochs))])
    argv.extend(["--batch-size", str(int(value_request.batch_size))])
    argv.extend(["--learning-rate", str(float(value_request.learning_rate))])
    argv.extend(["--image-size", str(int(value_request.image_size))])
    argv.extend(["--rack-jitter", str(float(value_request.rack_jitter))])
    return argv


def launch_local_reward_run(
    *,
    kind: RewardRunKind,
    request: PooltoolRewardOptimizationActionRequest | PooltoolRewardValueModelActionRequest,
    layout: ArtifactLayout | None = None,
) -> LocalRewardRunRecord:
    resolved_layout = layout or get_artifact_layout()
    ensure_reward_launch_available(layout=resolved_layout)
    run_id = str(request.run_id or default_run_id()).strip()
    if not run_id:
        raise ValueError("Reward run id must not be blank.")
    run_root = reward_run_root(kind, run_id, layout=resolved_layout)
    if run_root.exists() and any(run_root.iterdir()):
        raise ValueError(f"Reward run already exists: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    stdout_path = run_root / STDOUT_FILENAME
    stderr_path = run_root / STDERR_FILENAME
    progress_path = local_progress_path(run_root)
    request_payload = request.model_dump(mode="json", exclude_none=True)
    started_at = _utc_now()
    source_run_dir = str(resolve_pattern_run_dir(request.source_run_dir))
    _write_manifest(
        run_root,
        kind=kind,
        run_id=run_id,
        status="starting",
        source_run_dir=source_run_dir,
        request=request_payload,
        pid=None,
        started_at=started_at,
        finished_at=None,
    )
    env = dict(os.environ)
    env[REWARD_PROGRESS_PATH_ENV] = str(progress_path)
    env[REWARD_RUN_ROOT_ENV] = str(run_root)
    env[REWARD_LOCK_PATH_ENV] = str(reward_lock_path(layout=resolved_layout))
    argv = [str(python_executable_for_env("pooltool")), *_build_reward_local_argv(kind, request)]
    stdout_handle = stdout_path.open("a", encoding="utf-8")
    stderr_handle = stderr_path.open("a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(resolved_layout.project_root),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
            text=True,
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()
    _write_lock(kind=kind, run_id=run_id, pid=int(proc.pid), layout=resolved_layout)
    _write_manifest(
        run_root,
        kind=kind,
        run_id=run_id,
        status="running",
        source_run_dir=source_run_dir,
        request=request_payload,
        pid=int(proc.pid),
        started_at=started_at,
        finished_at=None,
    )
    return load_local_reward_run(kind=kind, run_id=run_id, layout=resolved_layout)


def _prepare_run_context_for_request(
    *,
    kind: RewardRunKind,
    request: PooltoolRewardOptimizationActionRequest | PooltoolRewardValueModelActionRequest,
) -> tuple[
    PooltoolRewardOptimizationActionRequest | PooltoolRewardValueModelActionRequest,
    str,
    Path,
    ArtifactLayout,
]:
    resolved_layout = get_artifact_layout()
    run_id = str(request.run_id or default_run_id()).strip()
    request = request.model_copy(update={"run_id": run_id})
    run_root = reward_run_root(kind, run_id, layout=resolved_layout)
    ensure_reward_launch_available(layout=resolved_layout, allow_pid=os.getpid())
    manual_cli = not str(os.environ.get(REWARD_RUN_ROOT_ENV, "") or "").strip()
    os.environ[REWARD_RUN_ROOT_ENV] = str(run_root)
    os.environ[REWARD_PROGRESS_PATH_ENV] = str(local_progress_path(run_root))
    os.environ[REWARD_LOCK_PATH_ENV] = str(reward_lock_path(layout=resolved_layout))
    request_payload = request.model_dump(mode="json", exclude_none=True)
    source_run_dir = str(resolve_pattern_run_dir(request.source_run_dir))
    if not local_run_manifest_path(run_root).is_file():
        run_root.mkdir(parents=True, exist_ok=True)
        _write_manifest(
            run_root,
            kind=kind,
            run_id=run_id,
            status="starting",
            source_run_dir=source_run_dir,
            request=request_payload,
            pid=os.getpid(),
            started_at=_utc_now(),
            finished_at=None,
        )
    if manual_cli:
        global _MANUAL_STDOUT_TEE, _MANUAL_STDERR_TEE
        if _MANUAL_STDOUT_TEE is None:
            _MANUAL_STDOUT_TEE = _TeeStream(
                sys.stdout,
                (run_root / STDOUT_FILENAME).open("a", encoding="utf-8"),
            )
            sys.stdout = _MANUAL_STDOUT_TEE
        if _MANUAL_STDERR_TEE is None:
            _MANUAL_STDERR_TEE = _TeeStream(
                sys.stderr,
                (run_root / STDERR_FILENAME).open("a", encoding="utf-8"),
            )
            sys.stderr = _MANUAL_STDERR_TEE
    _write_lock(kind=kind, run_id=run_id, pid=os.getpid(), layout=resolved_layout)
    _update_current_manifest(status="running")
    return request, run_id, run_root, resolved_layout


def load_local_reward_run(
    *,
    kind: RewardRunKind,
    run_id: str,
    layout: ArtifactLayout | None = None,
) -> LocalRewardRunRecord:
    resolved_layout = layout or get_artifact_layout()
    run_root = reward_run_root(kind, run_id, layout=resolved_layout)
    manifest = _load_json_file(local_run_manifest_path(run_root))
    summary_path = run_root / "summary.json"
    progress = _load_json_file(local_progress_path(run_root))
    stdout_path = run_root / STDOUT_FILENAME
    stderr_path = run_root / STDERR_FILENAME
    pid = manifest.get("pid")
    try:
        normalized_pid = int(pid) if pid is not None else None
    except (TypeError, ValueError):
        normalized_pid = None
    status = str(manifest.get("status") or ("succeeded" if summary_path.is_file() else "unknown"))
    if status in {"starting", "running"} and not _pid_is_alive(normalized_pid):
        status = "succeeded" if summary_path.is_file() else "failed"
    recent_output = _tail_lines(stdout_path, limit=80)
    recent_output.extend(_tail_lines(stderr_path, limit=40))
    recent_output = recent_output[-120:]
    return LocalRewardRunRecord(
        run_id=run_id,
        kind=kind,
        root=run_root,
        status=status,
        started_at=manifest.get("started_at"),
        updated_at=manifest.get("updated_at"),
        finished_at=manifest.get("finished_at"),
        pid=normalized_pid,
        source_run_dir=str(manifest.get("source_run_dir") or ""),
        request=dict(manifest.get("request") or {}),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        progress_path=local_progress_path(run_root),
        summary_path=summary_path if summary_path.is_file() else None,
        recent_output=recent_output,
        progress=progress,
        error=str(manifest.get("error") or "").strip() or None,
    )


def list_local_reward_runs(
    *,
    layout: ArtifactLayout | None = None,
) -> list[LocalRewardRunRecord]:
    resolved_layout = layout or get_artifact_layout()
    records: list[LocalRewardRunRecord] = []
    for kind in ("optimization", "value_model"):
        typed_kind = kind  # type: ignore[assignment]
        root = reward_kind_root(typed_kind, layout=resolved_layout)
        if not root.is_dir():
            continue
        for run_root in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name, reverse=True):
            try:
                records.append(load_local_reward_run(kind=typed_kind, run_id=run_root.name, layout=resolved_layout))
            except Exception:
                continue
    records.sort(key=lambda item: ((item.updated_at or ""), item.run_id), reverse=True)
    return records


def _run_cli_optimization(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m dashboard.reward_local optimization")
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--goal-text", default="")
    parser.add_argument("--dsl-program", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--scene-count", type=int, default=10)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--budget", action="append", type=int, default=[])
    parser.add_argument("--rack-jitter", type=float, default=0.12)
    parser.add_argument("--ollama-base-url", required=True)
    parser.add_argument("--model-timeout", type=int, required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    request = PooltoolRewardOptimizationActionRequest.model_validate(
        {
            "source_run_dir": args.source_run_dir,
            "goal_text": args.goal_text,
            "dsl_program": args.dsl_program,
            "model": args.model,
            "scene_count": args.scene_count,
            "seed_count": args.seed_count,
            "budgets": args.budget or [10, 50, 100, 250, 500, 1000],
            "rack_jitter": args.rack_jitter,
            "ollama_base_url": args.ollama_base_url,
            "model_timeout": args.model_timeout,
            "run_id": args.run_id,
        }
    )
    request, run_id, _run_root, _layout = _prepare_run_context_for_request(
        kind="optimization",
        request=request,
    )
    try:
        run_pooltool_optimization_experiment(
            run_id=run_id,
            source_run_dir=resolve_pattern_run_dir(request.source_run_dir),
            goal_text=request.goal_text,
            dsl_program=request.dsl_program,
            model_name=request.model,
            llm_base_url=request.ollama_base_url,
            llm_timeout=request.model_timeout,
            scene_count=request.scene_count,
            seed_count=request.seed_count,
            budgets=request.budgets,
            rack_jitter=request.rack_jitter,
        )
    except Exception as exc:
        _finalize_current_run("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    _finalize_current_run("succeeded")
    return 0


def _run_cli_value_model(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m dashboard.reward_local value-model")
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--goal-text", default="")
    parser.add_argument("--dsl-program", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--train-scene-count", type=int, default=64)
    parser.add_argument("--train-actions-per-scene", type=int, default=32)
    parser.add_argument("--validation-scene-count", type=int, default=12)
    parser.add_argument("--validation-candidates-per-scene", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--rack-jitter", type=float, default=0.12)
    parser.add_argument("--ollama-base-url", required=True)
    parser.add_argument("--model-timeout", type=int, required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    request = PooltoolRewardValueModelActionRequest.model_validate(
        {
            "source_run_dir": args.source_run_dir,
            "goal_text": args.goal_text,
            "dsl_program": args.dsl_program,
            "model": args.model,
            "train_scene_count": args.train_scene_count,
            "train_actions_per_scene": args.train_actions_per_scene,
            "validation_scene_count": args.validation_scene_count,
            "validation_candidates_per_scene": args.validation_candidates_per_scene,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "image_size": args.image_size,
            "rack_jitter": args.rack_jitter,
            "ollama_base_url": args.ollama_base_url,
            "model_timeout": args.model_timeout,
            "run_id": args.run_id,
        }
    )
    request, run_id, _run_root, _layout = _prepare_run_context_for_request(
        kind="value_model",
        request=request,
    )
    try:
        run_pooltool_value_model_experiment(
            run_id=run_id,
            source_run_dir=resolve_pattern_run_dir(request.source_run_dir),
            goal_text=request.goal_text,
            dsl_program=request.dsl_program,
            model_name=request.model,
            llm_base_url=request.ollama_base_url,
            llm_timeout=request.model_timeout,
            train_scene_count=request.train_scene_count,
            train_actions_per_scene=request.train_actions_per_scene,
            validation_scene_count=request.validation_scene_count,
            validation_candidates_per_scene=request.validation_candidates_per_scene,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            image_size=request.image_size,
            rack_jitter=request.rack_jitter,
        )
    except Exception as exc:
        _finalize_current_run("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    _finalize_current_run("succeeded")
    return 0


def _current_run_context() -> tuple[RewardRunKind, str, Path, Path, ArtifactLayout]:
    run_root_text = str(os.environ.get(REWARD_RUN_ROOT_ENV, "") or "").strip()
    if not run_root_text:
        raise RuntimeError(f"{REWARD_RUN_ROOT_ENV} is required for standalone reward runs.")
    run_root = Path(run_root_text).resolve()
    local_manifest = _load_json_file(local_run_manifest_path(run_root))
    kind_text = str(local_manifest.get("kind") or "").strip()
    if kind_text not in KIND_TO_DIRNAME:
        if run_root.parent.name == OPTIMIZATION_EXPERIMENT_DIRNAME:
            kind_text = "optimization"
        elif run_root.parent.name == VALUE_MODEL_EXPERIMENT_DIRNAME:
            kind_text = "value_model"
    if kind_text not in KIND_TO_DIRNAME:
        raise RuntimeError(f"Unable to determine reward run kind for {run_root}")
    kind: RewardRunKind = kind_text  # type: ignore[assignment]
    run_id = str(local_manifest.get("run_id") or run_root.name).strip() or run_root.name
    layout = get_artifact_layout()
    return kind, run_id, run_root, local_run_manifest_path(run_root), layout


def _update_current_manifest(
    *,
    status: RewardRunStatus,
    error: str | None = None,
) -> None:
    kind, run_id, run_root, manifest_path, _layout = _current_run_context()
    current = _load_json_file(manifest_path)
    request = dict(current.get("request") or {})
    source_run_dir = str(current.get("source_run_dir") or "")
    started_at = str(current.get("started_at") or "").strip() or _utc_now()
    finished_at = _utc_now() if status in {"succeeded", "failed"} else None
    _write_manifest(
        run_root,
        kind=kind,
        run_id=run_id,
        status=status,
        source_run_dir=source_run_dir,
        request=request,
        pid=os.getpid(),
        started_at=started_at,
        finished_at=finished_at,
        error=error,
    )


def _finalize_current_run(status: RewardRunStatus, *, error: str | None = None) -> None:
    kind, _run_id, _run_root, _manifest_path, layout = _current_run_context()
    _update_current_manifest(status=status, error=error)
    _clear_lock_if_owned(pid=os.getpid(), layout=layout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dashboard.reward_local",
        description="Standalone local runner for PoolTool reward-program experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    optimization = subparsers.add_parser("optimization")
    optimization.set_defaults(_handler=_run_cli_optimization)
    value_model = subparsers.add_parser("value-model")
    value_model.set_defaults(_handler=_run_cli_value_model)
    parsed, remaining = parser.parse_known_args(argv)
    return int(parsed._handler(remaining))


if __name__ == "__main__":
    raise SystemExit(main())
