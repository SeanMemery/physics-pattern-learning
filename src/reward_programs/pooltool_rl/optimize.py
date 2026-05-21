from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import write_json
from .dsl import evaluate_reward_program
from .library import count_supervised_library_patterns
from .pooltool_env import PooltoolEngine
from .schemas import ActionSpec, OptimizationOutcome, RewardProgram, RewardValue, SceneSpec


def copy_initial_scene(scene: SceneSpec, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(scene.initial_scene_overhead_path.read_bytes())
    return destination


def optimize_scene(
    *,
    scene: SceneSpec,
    reward_program: RewardProgram,
    engine: PooltoolEngine,
    budget: int,
    run_dir: Path,
    seed: int = 0,
    render_best: bool = True,
) -> OptimizationOutcome:
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_trace: dict[str, Any] | None = None
    best_reward: RewardValue | None = None
    best_action: ActionSpec | None = None
    best_step = -1
    for step in range(budget):
        action = engine.sample_action(scene, seed=seed + step)
        trace = engine.simulate_action(scene, action)
        base_reward = evaluate_reward_program(
            reward_program.dsl_program,
            trace,
            table_width=scene.table_width,
            table_length=scene.table_length,
        )
        pattern_count: int | None = None
        pattern_bonus = 0.0
        reward = base_reward
        # If the DSL succeeds, prefer "simpler" traces with fewer library patterns firing.
        if base_reward.success:
            pattern_count = count_supervised_library_patterns(trace)
            pattern_bonus = 1.0 / (1.0 + float(pattern_count))
            reward = RewardValue(
                score=float(base_reward.score) + pattern_bonus,
                success=base_reward.success,
                event_time=base_reward.event_time,
            )
        step_result = {
            "step": step + 1,
            "action": action,
            "reward": reward,
            "base_reward": base_reward,
            "pattern_count": pattern_count,
            "pattern_bonus": pattern_bonus,
            "terminal_status": ((trace.get("outcome") or {}).get("terminal_status") if isinstance(trace, dict) else None),
        }
        history.append(step_result)
        if best_reward is None or reward.score > best_reward.score:
            best_reward = reward
            best_trace = trace
            best_action = action
            best_step = step + 1
            write_json(candidates_dir / "best_so_far.json", {"step": best_step, "action": best_action, "reward": best_reward})
            write_json(candidates_dir / "best_trace.json", best_trace)
        if step == 0 or (step + 1) == budget or (step + 1) % max(1, budget // 20) == 0:
            write_progress(
                path=run_dir / "progress.json",
                completed_steps=step + 1,
                total_steps=budget,
                best_action=best_action,
                best_reward=best_reward,
                best_step=best_step,
            )
    assert best_reward is not None and best_trace is not None and best_action is not None
    trace_path = write_json(run_dir / "best_trace.json", best_trace)
    reward_path = write_json(run_dir / "best_reward.json", best_reward)
    write_json(run_dir / "search_history.json", history)
    if render_best:
        render_paths = engine.render_shot(scene, best_action, run_dir)
        write_json(run_dir / "render_outputs.json", render_paths)
    return OptimizationOutcome(
        action=best_action,
        best_reward=best_reward.score,
        success=best_reward.success,
        trace_path=trace_path,
        reward_path=reward_path,
    )


def write_progress(
    *,
    path: Path,
    completed_steps: int,
    total_steps: int,
    best_action: ActionSpec | None,
    best_reward: RewardValue | None,
    best_step: int,
) -> Path:
    payload = {
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "fraction": (completed_steps / total_steps) if total_steps else 0.0,
        "best_step": best_step if best_step >= 0 else None,
        "best_action": best_action,
        "best_reward": best_reward,
    }
    return write_json(path, payload)


def load_result_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
