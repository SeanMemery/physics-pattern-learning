from __future__ import annotations

from pathlib import Path

from .artifacts import append_log, create_run_dir, write_json
from .config import load_config, load_goals
from .optimize import optimize_scene
from .pooltool_env import PooltoolEngine
from .reward_factory import build_reward_program_for_goal
from .scenes import list_scene_dirs, load_scene


def run(
    *,
    goals_file: Path,
    config_file: Path | None = None,
    scene_count: int | None = None,
    seed_count: int | None = None,
    budgets: list[int] | None = None,
    seed: int = 0,
) -> Path:
    config = load_config(config_file)
    goals = load_goals(goals_file)
    run_dir = create_run_dir("unverified_opt")
    effective_scene_count = scene_count or config.optimization.scene_count
    effective_seed_count = seed_count or config.optimization.seed_count
    effective_budgets = budgets or list(config.optimization.budgets)
    write_json(
        run_dir / "request.json",
        {
            "goals_file": goals_file,
            "scene_count": effective_scene_count,
            "seed_count": effective_seed_count,
            "budgets": effective_budgets,
            "seed": seed,
            "experiment": "unverified_opt",
        },
    )
    append_log(run_dir / "run.log", f"start scene_count={effective_scene_count} seed_count={effective_seed_count} budgets={effective_budgets}")
    all_scene_dirs = list_scene_dirs()[:effective_scene_count]
    engine = PooltoolEngine()
    aggregate: dict[str, list[dict[str, float | int]]] = {}
    write_json(run_dir / "result.json", {"experiment": "unverified_opt", "status": "running", "summary": aggregate})
    for goal_index, goal in enumerate(goals):
        goal_dir = run_dir / goal.id
        goal_dir.mkdir(parents=True, exist_ok=True)
        representative_scene = load_scene(all_scene_dirs[goal_index % len(all_scene_dirs)])
        reward_program = build_reward_program_for_goal(
            goal=goal,
            scene=representative_scene,
            llm_settings=config.llm,
            artifact_root=goal_dir,
        )
        (goal_dir / "program.dsl").write_text(reward_program.dsl_program + "\n", encoding="utf-8")
        write_json(goal_dir / "reward_program.json", reward_program)
        append_log(run_dir / "run.log", f"goal {goal.id} prepared")

        budget_rows: list[dict[str, float | int]] = []
        for budget in effective_budgets:
            reward_successes = 0
            rewards: list[float] = []
            for scene_index, scene_dir in enumerate(all_scene_dirs):
                scene = load_scene(scene_dir)
                for local_seed in range(effective_seed_count):
                    case_dir = goal_dir / f"budget_{budget:04d}" / f"{scene.sample_id}_seed_{local_seed:02d}"
                    case_dir.mkdir(parents=True, exist_ok=True)
                    outcome = optimize_scene(
                        scene=scene,
                        reward_program=reward_program,
                        engine=engine,
                        budget=budget,
                        run_dir=case_dir,
                        seed=seed + goal_index * 100_000 + scene_index * 1_000 + local_seed,
                        render_best=False,
                    )
                    rewards.append(float(outcome.best_reward))
                    reward_successes += 1 if outcome.success else 0
            total_cases = max(1, len(all_scene_dirs) * effective_seed_count)
            row = {
                "budget": budget,
                "case_count": total_cases,
                "reward_success_count": reward_successes,
                "reward_success_rate": reward_successes / total_cases,
                "avg_best_reward": sum(rewards) / max(1, len(rewards)),
            }
            budget_rows.append(row)
            write_json(goal_dir / f"budget_{budget:04d}_summary.json", row)
            append_log(run_dir / "run.log", f"goal {goal.id} budget={budget} reward_success={row['reward_success_rate']:.3f} avg_best_reward={row['avg_best_reward']:.3f}")
            aggregate[goal.id] = budget_rows
            write_json(
                run_dir / "result.json",
                {
                    "experiment": "unverified_opt",
                    "status": "running",
                    "current_goal": goal.id,
                    "current_budget": budget,
                    "summary": aggregate,
                },
            )
        aggregate[goal.id] = budget_rows
        write_json(goal_dir / "result.json", {"goal": goal, "reward_program": reward_program, "budget_summary": budget_rows})
        write_json(
            run_dir / "result.json",
            {
                "experiment": "unverified_opt",
                "status": "running",
                "current_goal": goal.id,
                "summary": aggregate,
            },
        )
    write_json(run_dir / "result.json", {"experiment": "unverified_opt", "status": "done", "summary": aggregate})
    append_log(run_dir / "run.log", "done")
    return run_dir
