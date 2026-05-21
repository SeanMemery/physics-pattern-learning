from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
import warnings

from .artifacts import append_log, timestamp_run_id, write_json
from .config import load_config, load_goals, project_root
from .dsl import synthesize_reward_program
from .library import annotate_trace_with_supervised_library
from .optimize import copy_initial_scene, optimize_scene
from .pooltool_env import PooltoolEngine
from .reward_factory import build_reward_program_for_goal
from .scenes import sample_random_scene
from .schemas import RewardProgram, SceneSpec
from .dsl import evaluate_reward_program


@contextmanager
def redirect_std_fds(path: Path):
    """
    Redirect OS-level stdout/stderr (fd 1/2) to a file.

    This catches noisy native libraries (e.g. Panda3D) that bypass `sys.stderr`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.flush()
        orig_out = os.dup(1)
        orig_err = os.dup(2)
        try:
            os.dup2(handle.fileno(), 1)
            os.dup2(handle.fileno(), 2)
            yield
        finally:
            try:
                os.dup2(orig_out, 1)
                os.dup2(orig_err, 2)
            finally:
                os.close(orig_out)
                os.close(orig_err)


def single_scene_runs_root() -> Path:
    """
    Store single-scene runs under repo-local, inspectable paths.

    We intentionally keep this independent of the main `artifacts/` tree so the
    single-scene workflow can be broken into multiple commands that all write
    into the same run directory.
    """
    root = project_root() / "single_scene" / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def latest_single_scene_run_dir() -> Path:
    root = single_scene_runs_root()
    runs = sorted([path for path in root.iterdir() if path.is_dir()], key=lambda p: p.name)
    if not runs:
        raise FileNotFoundError(f"No single-scene runs found under {root}")
    return runs[-1]


def resolve_single_scene_run_dir(run_dir: Path | None) -> Path:
    # Common footgun: shell var expands to empty string, which Typer/Path turns into "."
    if run_dir is None or run_dir == Path("."):
        return latest_single_scene_run_dir()
    return run_dir


def create_single_scene_run_dir(*, run_id: str | None = None) -> Path:
    final_run_id = run_id or timestamp_run_id("single_scene")
    run_dir = single_scene_runs_root() / final_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def log_dir(run_dir: Path) -> Path:
    path = run_dir / "log"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_run_path(run_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else (run_dir / path)


def load_scene_from_run_dir(run_dir: Path) -> SceneSpec:
    scene_path = run_dir / "scene.json"
    if not scene_path.is_file():
        raise FileNotFoundError(
            f"Missing {scene_path}. Run `pooltool-rl single-scene sample` first and pass its printed path via --run-dir."
        )
    payload = json.loads(scene_path.read_text(encoding="utf-8"))
    scene = SceneSpec.model_validate(payload)
    # Stored paths are typically run-relative; resolve before use.
    return scene.model_copy(
        update={
            "trace_path": _resolve_run_path(run_dir, scene.trace_path),
            "initial_scene_overhead_path": _resolve_run_path(run_dir, scene.initial_scene_overhead_path),
            "initial_scene_askew_path": (
                _resolve_run_path(run_dir, scene.initial_scene_askew_path) if scene.initial_scene_askew_path is not None else None
            ),
        }
    )


def load_reward_program_from_run_dir(run_dir: Path) -> RewardProgram:
    # New layout: keep the DSL at the top level and the full synthesis artifacts under log/.
    if (run_dir / "program.dsl").is_file() and (run_dir / "goal.json").is_file():
        goal_payload = json.loads((run_dir / "goal.json").read_text(encoding="utf-8"))
        goal_text = str(goal_payload.get("goal_text") or "")
        dsl_program = (run_dir / "program.dsl").read_text(encoding="utf-8").strip()
        return RewardProgram(goal_text=goal_text, dsl_program=dsl_program, synthesized=True)

    # Back-compat: older runs stored reward_program.json at the top level.
    program_path = run_dir / "reward_program.json"
    if not program_path.is_file():
        raise FileNotFoundError(
            f"Missing {run_dir / 'program.dsl'} or {program_path}. Run `pooltool-rl single-scene synthesize --run-dir <RUN_DIR> --goal-text ...` first."
        )
    payload = json.loads(program_path.read_text(encoding="utf-8"))
    program = RewardProgram.model_validate(payload)
    # Resolve any stored request/response paths.
    return program.model_copy(
        update={
            "synthesis_request_path": (
                _resolve_run_path(run_dir, program.synthesis_request_path) if program.synthesis_request_path is not None else None
            ),
            "synthesis_response_path": (
                _resolve_run_path(run_dir, program.synthesis_response_path) if program.synthesis_response_path is not None else None
            ),
        }
    )


def sample(
    *,
    seed: int = 0,
    run_id: str | None = None,
) -> Path:
    run_dir = create_single_scene_run_dir(run_id=run_id)
    sample_into(run_dir=run_dir, seed=seed)
    return run_dir


def sample_into(*, run_dir: Path, seed: int = 0) -> SceneSpec:
    logs = log_dir(run_dir)
    append_log(logs / "run.log", f"start sample seed={seed}")
    scene = sample_random_scene(seed=seed)
    append_log(logs / "run.log", f"scene {scene.sample_id}")

    # Copy dataset artifacts into the run dir so later steps are self-contained.
    copy_initial_scene(scene, run_dir / "initial_scene_overhead.png")
    if scene.initial_scene_askew_path is not None and scene.initial_scene_askew_path.is_file():
        (logs / "initial_scene_askew.png").write_bytes(scene.initial_scene_askew_path.read_bytes())
    (logs / "trace.json").write_bytes(scene.trace_path.read_bytes())

    stored_scene = scene.model_copy(
        update={
            "trace_path": Path("log") / "trace.json",
            "initial_scene_overhead_path": Path("initial_scene_overhead.png"),
            "initial_scene_askew_path": (Path("log") / "initial_scene_askew.png") if (logs / "initial_scene_askew.png").is_file() else None,
        }
    )
    write_json(
        logs / "sample_request.json",
        {
            "seed": seed,
            "source_sample_id": scene.sample_id,
        },
    )
    write_json(run_dir / "scene.json", stored_scene)
    write_json(
        run_dir / "initial_state.json",
        {
            "sample_id": stored_scene.sample_id,
            "table_width": stored_scene.table_width,
            "table_length": stored_scene.table_length,
            "initial_ball_positions": stored_scene.initial_ball_positions,
            "speed_mapping": stored_scene.speed_mapping,
            "strikespot_mapping": stored_scene.strikespot_mapping,
        },
    )
    return stored_scene


def synthesize(
    *,
    run_dir: Path,
    goal_text: str,
    config_file: Path | None = None,
    use_images: bool = True,
    progress: bool = False,
) -> RewardProgram:
    config = load_config(config_file)
    logs = log_dir(run_dir)
    scene = load_scene_from_run_dir(run_dir)
    write_json(run_dir / "goal.json", {"goal_text": goal_text})
    append_log(logs / "run.log", "start synthesize")
    progress_fn = (lambda msg: print(msg, flush=True)) if progress else None
    reward_program = synthesize_reward_program(
        goal_text=goal_text,
        scene=scene,
        llm_settings=config.llm,
        artifact_root=logs,
        use_images=use_images,
        progress=progress_fn,
    )
    (run_dir / "program.dsl").write_text(reward_program.dsl_program + "\n", encoding="utf-8")
    # Keep the full RewardProgram under log/; keep the headline DSL at the top level.
    write_json(logs / "reward_program.json", reward_program)
    append_log(logs / "run.log", "dsl synthesized")
    return reward_program


def optimize(
    *,
    run_dir: Path,
    config_file: Path | None = None,
    budget: int | None = None,
    seed: int = 0,
    render_best: bool = True,
    until_success: bool = False,
    max_attempts: int = 1,
    optimization_id: str | None = None,
) -> Path:
    config = load_config(config_file)
    logs = log_dir(run_dir)
    scene = load_scene_from_run_dir(run_dir)
    reward_program = load_reward_program_from_run_dir(run_dir)
    total_budget = budget or config.optimization.search_budget
    final_max_attempts = max(1, int(max_attempts))
    request = {
        "budget": total_budget,
        "seed": seed,
        "render_best": render_best,
        "until_success": until_success,
        "max_attempts": final_max_attempts,
        "config_file": config_file,
    }
    write_json(logs / "optimization_request.json", request)

    engine = PooltoolEngine()
    best_outcome = None
    best_attempt_dir = None
    best_attempt_index = None

    if optimization_id is None:
        attempts_root = logs / "attempts"
    else:
        attempts_root = logs / "optimizations" / optimization_id / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, final_max_attempts + 1):
        attempt_seed = seed + (attempt - 1) * 10_000
        attempt_dir = attempts_root / f"attempt_{attempt:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        append_log(attempt_dir / "run.log", f"start optimize budget={total_budget} seed={attempt_seed} attempt={attempt}/{final_max_attempts}")
        # PoolTool/Panda3D can be noisy on stdout/stderr; keep the CLI output clean by
        # redirecting that noise into a per-attempt log file.
        noise_log_path = attempt_dir / "stdout_stderr.log"
        noise_log_path.write_text("", encoding="utf-8")
        with redirect_std_fds(noise_log_path):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outcome = optimize_scene(
                    scene=scene,
                    reward_program=reward_program,
                    engine=engine,
                    budget=total_budget,
                    run_dir=attempt_dir,
                    seed=attempt_seed,
                    render_best=render_best,
                )
        append_log(attempt_dir / "run.log", f"done best_reward={outcome.best_reward:.4f} success={outcome.success}")
        write_json(attempt_dir / "optimization_result.json", outcome)
        write_json(
            attempt_dir / "result.json",
            {
                "experiment": "single_scene",
                "scene": scene,
                "reward_program": reward_program,
                "optimization": outcome,
                "attempt": {"index": attempt, "seed": attempt_seed, "budget": total_budget},
            },
        )

        if best_outcome is None or outcome.best_reward > best_outcome.best_reward:
            best_outcome = outcome
            best_attempt_dir = attempt_dir
            best_attempt_index = attempt

        if until_success and outcome.success:
            break

    assert best_outcome is not None and best_attempt_dir is not None and best_attempt_index is not None
    append_log(logs / "run.log", f"optimize best_attempt={best_attempt_index} best_reward={best_outcome.best_reward:.4f} success={best_outcome.success}")

    # Promote best attempt "headline" outputs to the run root.
    best_video = best_attempt_dir / "simulation_overhead.mp4"
    if best_video.is_file():
        (run_dir / "simulation_overhead.mp4").write_bytes(best_video.read_bytes())
    # Some engines render an overhead still; promote if present.
    best_png = best_attempt_dir / "initial_scene_overhead.png"
    if best_png.is_file():
        (run_dir / "optimized_scene_overhead.png").write_bytes(best_png.read_bytes())

    shot_payload = {
        "goal_text": reward_program.goal_text,
        "dsl_program": reward_program.dsl_program,
        "best_attempt_index": best_attempt_index,
        "best_attempt_dir": str(best_attempt_dir),
        "budget": total_budget,
        "seed_base": seed,
        "best_action": best_outcome.action,
        "best_reward": best_outcome.best_reward,
        "success": best_outcome.success,
        "video_path": "simulation_overhead.mp4" if (run_dir / "simulation_overhead.mp4").is_file() else None,
        "trace_path": str(best_outcome.trace_path),
        "reward_path": str(best_outcome.reward_path),
    }
    try:
        trace_payload = json.loads(Path(best_outcome.trace_path).read_text(encoding="utf-8"))
        library_annotation, library_errors = annotate_trace_with_supervised_library(trace_payload)
        # Also expose the simplicity bonus components for the best trace.
        pattern_item_count = 0
        for match in (library_annotation.get("matches") or []):
            if isinstance(match, dict) and isinstance(match.get("patterns"), list):
                pattern_item_count += len(match["patterns"])
        shot_payload["dsl_reward"] = evaluate_reward_program(
            reward_program.dsl_program,
            trace_payload,
            table_width=scene.table_width,
            table_length=scene.table_length,
        )
        shot_payload["pattern_count"] = pattern_item_count
        shot_payload["pattern_bonus"] = 1.0 / (1.0 + float(pattern_item_count))
        shot_payload["library_patterns"] = library_annotation
        if library_errors:
            write_json(logs / "library_annotation_errors.json", library_errors)
    except Exception as exc:  # pragma: no cover - depends on filesystem and library code
        write_json(logs / "library_annotation_errors.json", [{"error": f"{type(exc).__name__}: {exc}"}])
    write_json(run_dir / "shot.json", shot_payload)
    best_payload = {"best_attempt_index": best_attempt_index, "best_attempt_dir": str(best_attempt_dir), "best_outcome": best_outcome}
    result_payload = {"experiment": "single_scene", "scene": scene, "reward_program": reward_program, "optimization": best_outcome}
    if optimization_id is None:
        write_json(logs / "optimization_best.json", best_payload)
        write_json(logs / "result.json", result_payload)
    else:
        opt_root = attempts_root.parent
        write_json(opt_root / "optimization_best.json", best_payload)
        write_json(opt_root / "result.json", result_payload)
    return run_dir


def optimize_repeat(
    *,
    run_dir: Path,
    config_file: Path | None = None,
    budget: int | None = None,
    seed: int = 0,
    render_best: bool = True,
    until_success: bool = False,
    max_attempts: int = 1,
) -> Path:
    """
    Rerun optimization using the saved DSL/goal in `run_dir` and write results into
    a new timestamped optimization session under `log/optimizations/`.
    """
    opt_id = timestamp_run_id("opt")
    return optimize(
        run_dir=run_dir,
        config_file=config_file,
        budget=budget,
        seed=seed,
        render_best=render_best,
        until_success=until_success,
        max_attempts=max_attempts,
        optimization_id=opt_id,
    )


def run(
    *,
    goal_id: str,
    goals_file: Path,
    config_file: Path | None = None,
    budget: int | None = None,
    seed: int = 0,
) -> Path:
    config = load_config(config_file)
    goals = {goal.id: goal for goal in load_goals(goals_file)}
    if goal_id not in goals:
        raise KeyError(f"Unknown goal id: {goal_id}")
    goal = goals[goal_id]
    run_dir = create_single_scene_run_dir()
    write_json(
        run_dir / "request.json",
        {
            "mode": "run",
            "goal_id": goal.id,
            "goal_text": goal.text,
            "budget": budget or config.optimization.search_budget,
            "seed": seed,
            "goals_file": goals_file,
            "config_file": config_file,
        },
    )
    scene = sample_into(run_dir=run_dir, seed=seed)
    reward_program = build_reward_program_for_goal(
        goal=goal,
        scene=scene,
        llm_settings=config.llm,
        artifact_root=log_dir(run_dir),
    )
    write_json(run_dir / "goal.json", {"goal_text": goal.text})
    (run_dir / "program.dsl").write_text(reward_program.dsl_program + "\n", encoding="utf-8")
    write_json(log_dir(run_dir) / "reward_program.json", reward_program)
    optimize(run_dir=run_dir, config_file=config_file, budget=budget, seed=seed, render_best=True)
    return run_dir
