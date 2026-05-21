from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from .artifacts import append_log, create_run_dir, write_json
from .config import load_config, load_goals
from .dsl import evaluate_reward_program
from .plots import write_line_plot_svg
from .pooltool_env import PooltoolEngine
from .reward_factory import build_reward_program_for_goal
from .scenes import list_scene_dirs, load_scene
from .schemas import ActionSpec
from .verifiers import evaluate_verifier


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def _stderr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return _sample_std(values) / math.sqrt(len(values))


def _compute_iteration_budget_summary(
    aggregate: dict[str, list[dict[str, float | int | None]]],
    budgets: list[int],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for budget in budgets:
        matching = [row for goal_rows in aggregate.values() for row in goal_rows if int(row["budget"]) == int(budget)]
        if not matching:
            continue
        reward_rates = [float(row["reward_optimizer_verifier_success_rate"]) for row in matching]
        verifier_rates = [float(row["verifier_optimizer_verifier_success_rate"]) for row in matching]
        reward_scores = [float(row["reward_optimizer_avg_score"]) for row in matching]
        verifier_scores = [float(row["verifier_optimizer_avg_score"]) for row in matching]
        rows.append(
            {
                "budget": int(budget),
                "goal_count": len(matching),
                "reward_optimizer_avg_verifier_success_rate": _mean(reward_rates),
                "verifier_optimizer_avg_verifier_success_rate": _mean(verifier_rates),
                "reward_optimizer_avg_score": _mean(reward_scores),
                "verifier_optimizer_avg_score": _mean(verifier_scores),
            }
        )
    return rows


def _compute_aggregate_budget_summary(
    iteration_results: list[dict[str, object]],
    budgets: list[int],
) -> list[dict[str, float | int | bool]]:
    rows: list[dict[str, float | int | bool]] = []
    for budget in budgets:
        budget_rows = [
            budget_row
            for iteration in iteration_results
            for budget_row in iteration.get("budget_summary", [])
            if int(budget_row["budget"]) == int(budget)
        ]
        if not budget_rows:
            continue
        reward_rates = [float(row["reward_optimizer_avg_verifier_success_rate"]) for row in budget_rows]
        verifier_rates = [float(row["verifier_optimizer_avg_verifier_success_rate"]) for row in budget_rows]
        reward_scores = [float(row["reward_optimizer_avg_score"]) for row in budget_rows]
        verifier_scores = [float(row["verifier_optimizer_avg_score"]) for row in budget_rows]
        advantage_rates = [reward - verifier for reward, verifier in zip(reward_rates, verifier_rates)]
        rows.append(
            {
                "budget": int(budget),
                "iteration_count": len(budget_rows),
                "reward_optimizer_avg_verifier_success_rate_mean": _mean(reward_rates),
                "reward_optimizer_avg_verifier_success_rate_std": _sample_std(reward_rates),
                "reward_optimizer_avg_verifier_success_rate_stderr": _stderr(reward_rates),
                "reward_optimizer_avg_verifier_success_rate_ci95": 1.96 * _stderr(reward_rates),
                "verifier_optimizer_avg_verifier_success_rate_mean": _mean(verifier_rates),
                "verifier_optimizer_avg_verifier_success_rate_std": _sample_std(verifier_rates),
                "verifier_optimizer_avg_verifier_success_rate_stderr": _stderr(verifier_rates),
                "verifier_optimizer_avg_verifier_success_rate_ci95": 1.96 * _stderr(verifier_rates),
                "reward_optimizer_avg_score_mean": _mean(reward_scores),
                "verifier_optimizer_avg_score_mean": _mean(verifier_scores),
                "reward_minus_verifier_verifier_success_mean": _mean(advantage_rates),
                "reward_minus_verifier_verifier_success_std": _sample_std(advantage_rates),
                "reward_minus_verifier_verifier_success_stderr": _stderr(advantage_rates),
                "reward_minus_verifier_verifier_success_ci95": 1.96 * _stderr(advantage_rates),
                "reward_optimizer_outperforms_verifier_optimizer": _mean(advantage_rates) > 0.0,
            }
        )
    return rows


def _cem_optimize(
    *,
    scene,
    engine: PooltoolEngine,
    budget: int,
    seed: int,
    score_trace,
) -> tuple[float, ActionSpec, dict]:
    speed_min, speed_max, a_min, a_max, b_min, b_max = engine.action_bounds(scene)
    rng = random.Random(seed)
    bounds = [
        (speed_min, speed_max),
        (a_min, a_max),
        (b_min, b_max),
    ]

    best_score = float("-inf")
    best_action = None
    best_trace = None
    evaluated = 0
    candidates: list[tuple[float, ActionSpec, dict, list[float]]] = []
    seed_pool = min(budget, max(8, budget // 3))

    def evaluate_vector(vector: list[float]) -> tuple[float, ActionSpec, dict, list[float]]:
        action = ActionSpec(speed=vector[0], strikespot_a=vector[1], strikespot_b=vector[2])
        trace = engine.simulate_action(scene, action)
        score = float(score_trace(trace))
        return score, action, trace, vector

    for sample_index in range(seed_pool):
        action = engine.sample_action(scene, seed=seed + sample_index)
        vector = [action.speed, action.strikespot_a, action.strikespot_b]
        score, evaluated_action, trace, evaluated_vector = evaluate_vector(vector)
        candidates.append((score, evaluated_action, trace, evaluated_vector))
        if score > best_score:
            best_score = score
            best_action = evaluated_action
            best_trace = trace
    evaluated += seed_pool

    elite_size = max(4, min(8, max(1, budget // 8)))
    mutation_batch = min(12, max(4, budget // 6 if budget >= 6 else budget))

    while evaluated < budget:
        elites = sorted(candidates, key=lambda item: item[0], reverse=True)[: min(elite_size, len(candidates))]
        progress = evaluated / max(1, budget)
        scale_factor = max(0.08, 0.35 * (1.0 - progress))
        current_batch_size = min(mutation_batch, budget - evaluated)
        for sample_index in range(current_batch_size):
            use_random_restart = sample_index == 0 and progress > 0.4
            if use_random_restart:
                action = engine.sample_action(scene, seed=seed + evaluated + sample_index)
                vector = [action.speed, action.strikespot_a, action.strikespot_b]
            else:
                parent = elites[sample_index % len(elites)]
                parent_vector = parent[3]
                vector = []
                for dim, ((lower, upper), center) in enumerate(zip(bounds, parent_vector)):
                    sigma = max((upper - lower) * scale_factor, 1e-3)
                    if dim == 0:
                        sigma *= 1.15
                    value = rng.gauss(center, sigma)
                    vector.append(min(upper, max(lower, value)))
            score, evaluated_action, trace, evaluated_vector = evaluate_vector(vector)
            candidates.append((score, evaluated_action, trace, evaluated_vector))
            if score > best_score:
                best_score = score
                best_action = evaluated_action
                best_trace = trace
        evaluated += current_batch_size
    assert best_action is not None and best_trace is not None
    return best_score, best_action, best_trace


def _random_optimize(
    *,
    scene,
    engine: PooltoolEngine,
    budget: int,
    seed: int,
    score_trace,
) -> tuple[float, ActionSpec, dict]:
    best_score = float("-inf")
    best_action = None
    best_trace = None
    for step in range(budget):
        action = engine.sample_action(scene, seed=seed + step)
        trace = engine.simulate_action(scene, action)
        score = float(score_trace(trace))
        if score > best_score:
            best_score = score
            best_action = action
            best_trace = trace
    assert best_action is not None and best_trace is not None
    return best_score, best_action, best_trace


def _run_iteration(
    *,
    iteration_dir: Path,
    goals_file: Path,
    config_file: Path | None,
    scene_count: int | None,
    seed_count: int | None,
    budgets: list[int] | None,
    seed: int,
    workers: int,
    budget_parallelism: int = 1,
    progress_callback=None,
) -> dict[str, object]:
    config = load_config(config_file)
    goals = load_goals(goals_file)
    effective_scene_count = scene_count or config.optimization.scene_count
    effective_seed_count = seed_count or config.optimization.seed_count
    effective_budgets = budgets or list(config.optimization.budgets)
    write_json(
        iteration_dir / "request.json",
        {
            "goals_file": goals_file,
            "scene_count": effective_scene_count,
            "seed_count": effective_seed_count,
            "budgets": effective_budgets,
            "seed": seed,
            "workers": workers,
            "budget_parallelism": budget_parallelism,
            "experiment": "verifier_opt",
        },
    )
    append_log(
        iteration_dir / "run.log",
        f"start scene_count={effective_scene_count} seed_count={effective_seed_count} budgets={effective_budgets} workers={workers}",
    )
    all_scene_dirs = list_scene_dirs()[:effective_scene_count]
    aggregate: dict[str, list[dict[str, float | int | None]]] = {}
    lock = Lock()

    def write_outputs(*, status: str, current_goal: str | None = None, current_budget: int | None = None) -> None:
        budget_summary = _compute_iteration_budget_summary(aggregate, effective_budgets)
        write_json(iteration_dir / "budget_summary.json", budget_summary)
        write_json(
            iteration_dir / "result.json",
            {
                "experiment": "verifier_opt",
                "status": status,
                "current_goal": current_goal,
                "current_budget": current_budget,
                "summary": aggregate,
                "budget_summary": budget_summary,
            },
        )
        write_line_plot_svg(
            iteration_dir / "avg_verifier_success_by_budget.pdf",
            title=None,
            x_label="Optimization Budget",
            y_label="Success Rate",
            series={
                "Reward Programs": [
                    (float(row["budget"]), float(row["reward_optimizer_avg_verifier_success_rate"])) for row in budget_summary
                ],
                "Binary Verifiers": [
                    (float(row["budget"]), float(row["verifier_optimizer_avg_verifier_success_rate"])) for row in budget_summary
                ],
            },
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "iteration_dir": iteration_dir,
                    "status": status,
                    "current_goal": current_goal,
                    "current_budget": current_budget,
                    "budget_summary": budget_summary,
                }
            )

    write_outputs(status="running")

    prepared: list[tuple[int, object, Path, object]] = []
    for goal_index, goal in enumerate(goals):
        goal_dir = iteration_dir / goal.id
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
        append_log(iteration_dir / "run.log", f"goal {goal.id} prepared")
        prepared.append((goal_index, goal, goal_dir, reward_program))

    def run_case(
        *,
        goal_index: int,
        goal,
        goal_dir: Path,
        reward_program,
        budget: int,
        scene_index: int,
        scene_dir: Path,
        local_seed: int,
    ) -> dict[str, float | int | bool]:
        scene = load_scene(scene_dir)
        engine = PooltoolEngine()
        case_dir = goal_dir / f"budget_{budget:04d}" / f"{scene.sample_id}_seed_{local_seed:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        case_seed = seed + goal_index * 100_000 + scene_index * 1_000 + local_seed
        reward_best_score, reward_best_action, reward_best_trace = _cem_optimize(
            scene=scene,
            engine=engine,
            budget=budget,
            seed=case_seed,
            score_trace=lambda trace: evaluate_reward_program(
                reward_program.dsl_program,
                trace,
                table_width=scene.table_width,
                table_length=scene.table_length,
            ).score,
        )
        verifier_best_score, verifier_best_action, verifier_best_trace = _random_optimize(
            scene=scene,
            engine=engine,
            budget=budget,
            seed=case_seed + 1_000_000,
            score_trace=lambda trace: (
                1.0
                if (goal.verifier_id and evaluate_verifier(goal.verifier_id, trace))
                else evaluate_reward_program(
                    goal.verifier_dsl_program or reward_program.dsl_program,
                    trace,
                    table_width=scene.table_width,
                    table_length=scene.table_length,
                ).score
            ),
        )
        reward_success = bool(evaluate_verifier(goal.verifier_id, reward_best_trace)) if goal.verifier_id else False
        verifier_success = bool(evaluate_verifier(goal.verifier_id, verifier_best_trace)) if goal.verifier_id else False
        write_json(
            case_dir / "comparison.json",
            {
                "reward_optimizer": {
                    "score": reward_best_score,
                    "verifier_success": reward_success,
                    "action": reward_best_action,
                },
                "verifier_optimizer": {
                    "score": verifier_best_score,
                    "verifier_success": verifier_success,
                    "action": verifier_best_action,
                },
            },
        )
        return {
            "reward_best_score": reward_best_score,
            "verifier_best_score": verifier_best_score,
            "reward_success": reward_success,
            "verifier_success": verifier_success,
        }

    def run_goal(goal_index: int, goal, goal_dir: Path, reward_program):
        budget_rows_by_budget: dict[int, dict[str, float | int | None]] = {}

        def current_goal_rows() -> list[dict[str, float | int | None]]:
            return [budget_rows_by_budget[budget] for budget in effective_budgets if budget in budget_rows_by_budget]

        total_cases = max(1, len(all_scene_dirs) * effective_seed_count)
        budget_states: dict[int, dict[str, object]] = {
            budget: {
                "pending_cases": [
                    (scene_index, scene_dir, local_seed)
                    for scene_index, scene_dir in enumerate(all_scene_dirs)
                    for local_seed in range(effective_seed_count)
                ],
                "active_workers": 0,
                "completed_cases": 0,
                "reward_optimizer_verifier_successes": 0,
                "verifier_optimizer_verifier_successes": 0,
                "reward_optimizer_scores": [],
                "verifier_optimizer_scores": [],
            }
            for budget in effective_budgets
        }

        def build_row(budget: int, *, complete: bool) -> dict[str, float | int | None]:
            state = budget_states[budget]
            completed_cases = int(state["completed_cases"])
            reward_scores = list(state["reward_optimizer_scores"])
            verifier_scores = list(state["verifier_optimizer_scores"])
            reward_successes = int(state["reward_optimizer_verifier_successes"])
            verifier_successes = int(state["verifier_optimizer_verifier_successes"])
            row = {
                "budget": budget,
                "case_count": total_cases if complete else completed_cases,
                "reward_optimizer_verifier_success_count": reward_successes,
                "reward_optimizer_verifier_success_rate": reward_successes / max(1, total_cases if complete else completed_cases),
                "verifier_optimizer_verifier_success_count": verifier_successes,
                "verifier_optimizer_verifier_success_rate": verifier_successes / max(1, total_cases if complete else completed_cases),
                "reward_optimizer_avg_score": _mean(reward_scores),
                "verifier_optimizer_avg_score": _mean(verifier_scores),
            }
            if not complete:
                row["total_case_count"] = total_cases
                row["complete"] = False
            return row

        def checkpoint_budget(budget: int, *, complete: bool) -> None:
            state = budget_states[budget]
            completed_cases = int(state["completed_cases"])
            if completed_cases == 0:
                return
            append_log(
                iteration_dir / "run.log",
                f"goal {goal.id} budget={budget} cases={completed_cases}/{total_cases}",
            )
            row = build_row(budget, complete=complete)
            if complete:
                write_json(goal_dir / f"budget_{budget:04d}_summary.json", row)
                append_log(
                    iteration_dir / "run.log",
                    f"goal {goal.id} budget={budget} reward_opt_verifier={row['reward_optimizer_verifier_success_rate']:.3f} verifier_opt_verifier={row['verifier_optimizer_verifier_success_rate']:.3f}",
                )
            with lock:
                budget_rows_by_budget[budget] = row
                aggregate[goal.id] = current_goal_rows()
                write_json(goal_dir / "result.json", {"goal": goal, "reward_program": reward_program, "budget_summary": current_goal_rows()})
                write_outputs(status="running", current_goal=goal.id, current_budget=budget)

        def choose_next_budget() -> int | None:
            unfinished = [
                budget
                for budget in effective_budgets
                if budget_states[budget]["pending_cases"]
            ]
            if not unfinished:
                return None
            zero_active = [budget for budget in unfinished if int(budget_states[budget]["active_workers"]) == 0]
            if zero_active:
                return min(zero_active)
            return max(
                unfinished,
                key=lambda budget: (
                    budget * len(budget_states[budget]["pending_cases"]) / (int(budget_states[budget]["active_workers"]) + 1)
                ),
            )

        def submit_one(executor, inflight: dict, budget: int) -> bool:
            state = budget_states[budget]
            pending_cases = state["pending_cases"]
            if not pending_cases:
                return False
            scene_index, scene_dir, local_seed = pending_cases.pop(0)
            state["active_workers"] = int(state["active_workers"]) + 1
            future = executor.submit(
                run_case,
                goal_index=goal_index,
                goal=goal,
                goal_dir=goal_dir,
                reward_program=reward_program,
                budget=budget,
                scene_index=scene_index,
                scene_dir=scene_dir,
                local_seed=local_seed,
            )
            inflight[future] = budget
            return True

        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as scheduler_executor:
            inflight: dict[object, int] = {}
            for budget in effective_budgets:
                if len(inflight) >= max(1, int(workers)):
                    break
                submit_one(scheduler_executor, inflight, budget)
            while len(inflight) < max(1, int(workers)):
                next_budget = choose_next_budget()
                if next_budget is None or not submit_one(scheduler_executor, inflight, next_budget):
                    break

            while inflight:
                future = next(iter(as_completed(inflight)))
                budget = inflight.pop(future)
                case_result = future.result()
                state = budget_states[budget]
                state["active_workers"] = max(0, int(state["active_workers"]) - 1)
                state["reward_optimizer_scores"].append(float(case_result["reward_best_score"]))
                state["verifier_optimizer_scores"].append(float(case_result["verifier_best_score"]))
                state["reward_optimizer_verifier_successes"] = int(state["reward_optimizer_verifier_successes"]) + (1 if bool(case_result["reward_success"]) else 0)
                state["verifier_optimizer_verifier_successes"] = int(state["verifier_optimizer_verifier_successes"]) + (1 if bool(case_result["verifier_success"]) else 0)
                state["completed_cases"] = int(state["completed_cases"]) + 1
                completed_cases = int(state["completed_cases"])
                if completed_cases == 1 or completed_cases == total_cases or completed_cases % max(1, total_cases // 5) == 0:
                    checkpoint_budget(budget, complete=completed_cases == total_cases)
                next_budget = choose_next_budget()
                if next_budget is not None:
                    submit_one(scheduler_executor, inflight, next_budget)

        with lock:
            aggregate[goal.id] = current_goal_rows()
            write_json(goal_dir / "result.json", {"goal": goal, "reward_program": reward_program, "budget_summary": current_goal_rows()})
        return goal.id

    for goal_index, goal, goal_dir, reward_program in prepared:
        run_goal(goal_index, goal, goal_dir, reward_program)

    budget_summary = _compute_iteration_budget_summary(aggregate, effective_budgets)
    write_outputs(status="done")
    append_log(iteration_dir / "run.log", "done")
    return {
        "iteration_dir": iteration_dir,
        "summary": aggregate,
        "budget_summary": budget_summary,
    }


def run(
    *,
    goals_file: Path,
    config_file: Path | None = None,
    scene_count: int | None = None,
    seed_count: int | None = None,
    budgets: list[int] | None = None,
    seed: int = 0,
    workers: int = 4,
    iterations: int = 3,
) -> Path:
    run_dir = create_run_dir("verifier_opt")
    effective_iterations = max(1, int(iterations))
    config = load_config(config_file)
    effective_budgets = budgets or list(config.optimization.budgets)
    iteration_parallelism = 1
    budget_parallelism = min(len(effective_budgets), max(1, int(workers)))
    case_workers = max(1, int(workers))
    write_json(
        run_dir / "request.json",
        {
            "goals_file": goals_file,
            "scene_count": scene_count,
            "seed_count": seed_count,
            "budgets": effective_budgets,
            "seed": seed,
            "workers": workers,
            "iteration_parallelism": iteration_parallelism,
            "budget_parallelism": budget_parallelism,
            "case_workers": case_workers,
            "iterations": effective_iterations,
            "experiment": "verifier_opt",
        },
    )
    append_log(
        run_dir / "run.log",
        f"start iterations={effective_iterations} workers={workers} iteration_parallelism={iteration_parallelism} budget_parallelism={budget_parallelism} case_workers={case_workers}",
    )
    iteration_results: list[dict[str, object]] = []
    root_lock = Lock()

    def write_root_outputs(*, status: str, current_iteration: int | None = None) -> None:
        budget_summary = _compute_aggregate_budget_summary(iteration_results, effective_budgets)
        write_json(run_dir / "iterations.json", iteration_results)
        write_json(
            run_dir / "result.json",
            {
                "experiment": "verifier_opt",
                "status": status,
                "current_iteration": current_iteration,
                "iteration_count": effective_iterations,
                "completed_iterations": len(iteration_results),
                "budget_summary": budget_summary,
                "iterations": iteration_results,
            },
        )
        write_json(run_dir / "budget_summary.json", budget_summary)
        write_line_plot_svg(
            run_dir / "avg_verifier_success_by_budget.pdf",
            title=None,
            x_label="Optimization Budget",
            y_label="Success Rate",
            series={
                "Reward Programs": [
                    (float(row["budget"]), float(row["reward_optimizer_avg_verifier_success_rate_mean"])) for row in budget_summary
                ],
                "Binary Verifiers": [
                    (float(row["budget"]), float(row["verifier_optimizer_avg_verifier_success_rate_mean"])) for row in budget_summary
                ],
            },
            error_series={
                "Reward Programs": [
                    (float(row["budget"]), float(row["reward_optimizer_avg_verifier_success_rate_std"])) for row in budget_summary
                ],
                "Binary Verifiers": [
                    (float(row["budget"]), float(row["verifier_optimizer_avg_verifier_success_rate_std"])) for row in budget_summary
                ],
            },
        )

    write_root_outputs(status="running")

    def run_single_iteration(iteration_index: int) -> dict[str, object]:
        iteration_dir = run_dir / f"iteration_{iteration_index + 1:02d}"
        iteration_seed = seed + iteration_index * 10_000_000
        append_log(run_dir / "run.log", f"iteration={iteration_index + 1} seed={iteration_seed} start")
        with root_lock:
            iteration_results.append(
                {
                    "iteration": iteration_index + 1,
                    "seed": iteration_seed,
                    "iteration_dir": iteration_dir,
                    "budget_summary": [],
                    "status": "running",
                }
            )
            iteration_results.sort(key=lambda item: int(item["iteration"]))
            write_root_outputs(status="running", current_iteration=iteration_index + 1)

        def on_iteration_progress(progress: dict[str, object]) -> None:
            with root_lock:
                for item in iteration_results:
                    if int(item["iteration"]) == iteration_index + 1:
                        item["budget_summary"] = progress.get("budget_summary", [])
                        item["status"] = progress.get("status", "running")
                        break
                write_root_outputs(status="running", current_iteration=iteration_index + 1)

        result = _run_iteration(
            iteration_dir=iteration_dir,
            goals_file=goals_file,
            config_file=config_file,
            scene_count=scene_count,
            seed_count=seed_count,
            budgets=effective_budgets,
            seed=iteration_seed,
            workers=case_workers,
            budget_parallelism=budget_parallelism,
            progress_callback=on_iteration_progress,
        )
        return {
            "iteration": iteration_index + 1,
            "seed": iteration_seed,
            "iteration_dir": iteration_dir,
            "budget_summary": result["budget_summary"],
            "status": "done",
        }

    with ThreadPoolExecutor(max_workers=iteration_parallelism) as executor:
        future_to_index = {
            executor.submit(run_single_iteration, iteration_index): iteration_index
            for iteration_index in range(effective_iterations)
        }
        for future in as_completed(future_to_index):
            iteration_index = future_to_index[future]
            iteration_result = future.result()
            with root_lock:
                for idx, item in enumerate(iteration_results):
                    if int(item["iteration"]) == iteration_index + 1:
                        iteration_results[idx] = iteration_result
                        break
                else:
                    iteration_results.append(iteration_result)
                iteration_results.sort(key=lambda item: int(item["iteration"]))
                write_root_outputs(status="running", current_iteration=iteration_index + 1)
            append_log(run_dir / "run.log", f"iteration={iteration_index + 1} done")
    write_root_outputs(status="done", current_iteration=effective_iterations)
    append_log(run_dir / "run.log", "done")
    return run_dir
