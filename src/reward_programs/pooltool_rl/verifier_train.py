from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from .artifacts import append_log, create_run_dir, write_json
from .config import load_config, load_goals
from .plots import write_line_plot_svg
from .pooltool_env import PooltoolEngine
from .reward_factory import build_reward_program_for_goal
from .scenes import list_scene_dirs, load_scene
from .value_model import SmallValueModel, build_training_examples, evaluate_model, write_training_summary


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


def _write_progress(run_dir: Path, payload: dict) -> None:
    write_json(run_dir / "progress.json", payload)


def _compute_iteration_budget_summary(
    aggregate: dict[str, list[dict[str, object]]],
    budgets: list[int],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for budget in budgets:
        matching = [row for goal_rows in aggregate.values() for row in goal_rows if int(row["budget"]) == int(budget)]
        if not matching:
            continue
        reward_rates = [float(row["reward_model"]["avg_chosen_verifier_success"] or 0.0) for row in matching]
        verifier_rates = [float(row["verifier_model"]["avg_chosen_verifier_success"] or 0.0) for row in matching]
        rows.append(
            {
                "budget": int(budget),
                "goal_count": len(matching),
                "reward_model_avg_verifier_success": _mean(reward_rates),
                "verifier_model_avg_verifier_success": _mean(verifier_rates),
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
        reward_rates = [float(row["reward_model_avg_verifier_success"]) for row in budget_rows]
        verifier_rates = [float(row["verifier_model_avg_verifier_success"]) for row in budget_rows]
        advantage_rates = [reward - verifier for reward, verifier in zip(reward_rates, verifier_rates)]
        rows.append(
            {
                "budget": int(budget),
                "iteration_count": len(budget_rows),
                "reward_model_avg_verifier_success_mean": _mean(reward_rates),
                "reward_model_avg_verifier_success_stderr": _stderr(reward_rates),
                "reward_model_avg_verifier_success_ci95": 1.96 * _stderr(reward_rates),
                "verifier_model_avg_verifier_success_mean": _mean(verifier_rates),
                "verifier_model_avg_verifier_success_stderr": _stderr(verifier_rates),
                "verifier_model_avg_verifier_success_ci95": 1.96 * _stderr(verifier_rates),
                "reward_minus_verifier_verifier_success_mean": _mean(advantage_rates),
                "reward_minus_verifier_verifier_success_stderr": _stderr(advantage_rates),
                "reward_minus_verifier_verifier_success_ci95": 1.96 * _stderr(advantage_rates),
                "reward_model_outperforms_verifier_model": _mean(advantage_rates) > 0.0,
            }
        )
    return rows


def _train_one_model(
    *,
    run_dir: Path,
    goal_id: str,
    training_budget: int,
    goal_dir: Path,
    mode_name: str,
    train_scenes,
    validation_scenes,
    reward_program,
    engine,
    config,
    goal_index: int,
    seed: int,
    verifier_id: str | None,
    progress_lock: Lock | None = None,
):
    model_dir = goal_dir / mode_name
    model_dir.mkdir(parents=True, exist_ok=True)

    def update_progress(stage: str, **extra) -> None:
        payload = {
            "experiment": "verifier_train",
            "status": "running",
            "current_goal": goal_id,
            "current_budget": training_budget,
            "current_mode": mode_name,
            "current_stage": stage,
            **extra,
        }
        write_json(model_dir / "progress.json", payload)
        if progress_lock is None:
            _write_progress(run_dir, payload)
        else:
            with progress_lock:
                _write_progress(run_dir, payload)

    update_progress(
        "build_training_examples",
        scene_index=0,
        scene_count=len(train_scenes),
        examples_built=0,
        example_budget=training_budget,
    )
    x_train, y_train = build_training_examples(
        scenes=train_scenes,
        reward_program=reward_program,
        engine=engine,
        actions_per_scene=config.value_model.actions_per_scene,
        seed=seed + goal_index * 100_000,
        verifier_id=verifier_id,
        target_mode=mode_name,
        example_budget=training_budget,
        progress_callback=lambda info: update_progress("build_training_examples", **info),
    )
    append_log(run_dir / "run.log", f"goal {goal_id} mode={mode_name} examples_built={len(y_train)}")
    model = SmallValueModel(input_dim=x_train.shape[1], hidden_size=config.value_model.hidden_size, rng_seed=seed + goal_index)
    update_progress("train", epoch=0, epochs=config.value_model.epochs, mse=None, example_count=int(x_train.shape[0]))
    train_history = model.train(
        x=x_train,
        y=y_train,
        epochs=config.value_model.epochs,
        learning_rate=config.value_model.learning_rate,
        batch_size=config.value_model.batch_size,
        progress_callback=lambda info: (
            update_progress("train", **info, example_count=int(x_train.shape[0])),
            append_log(run_dir / "run.log", f"goal {goal_id} mode={mode_name} epoch={info['epoch']}/{info['epochs']} mse={info['mse']:.6f}"),
        )[-1],
    )
    model_path = model.save(model_dir / "model.npz")
    update_progress(
        "evaluate",
        scene_index=0,
        scene_count=len(validation_scenes),
        validation_candidates=config.value_model.validation_candidates,
    )
    evaluation = evaluate_model(
        model=model,
        scenes=validation_scenes,
        reward_program=reward_program,
        engine=engine,
        validation_candidates=config.value_model.validation_candidates,
        seed=seed + goal_index * 100_000 + 50_000,
        verifier_id=verifier_id,
        progress_callback=lambda info: update_progress("evaluate", **info),
    )
    write_training_summary(model_dir / "result.json", train_history=train_history, evaluation=evaluation)
    update_progress(
        "done",
        avg_chosen_reward=evaluation["avg_chosen_reward"],
        avg_chosen_verifier_success=evaluation["avg_chosen_verifier_success"],
    )
    return model_path, evaluation


def _run_iteration(
    *,
    iteration_dir: Path,
    goals_file: Path,
    config_file: Path | None,
    seed: int,
    budgets: list[int] | None,
    workers: int,
) -> dict[str, object]:
    config = load_config(config_file)
    goals = load_goals(goals_file)
    effective_budgets = budgets or list(config.value_model.budgets)
    write_json(
        iteration_dir / "request.json",
        {
            "goals_file": goals_file,
            "seed": seed,
            "config_file": config_file,
            "budgets": effective_budgets,
            "workers": workers,
            "experiment": "verifier_train",
        },
    )
    append_log(iteration_dir / "run.log", f"start budgets={effective_budgets} workers={workers}")
    scene_dirs = list_scene_dirs()
    train_scenes = [load_scene(path) for path in scene_dirs[: config.value_model.train_scene_count]]
    validation_start = config.value_model.train_scene_count
    validation_end = validation_start + config.value_model.validation_scene_count
    validation_scenes = [load_scene(path) for path in scene_dirs[validation_start:validation_end]]
    aggregate: dict[str, list[dict[str, object]]] = {}
    lock = Lock()

    def write_outputs(*, status: str, current_goal: str | None = None, current_budget: int | None = None) -> None:
        budget_summary = _compute_iteration_budget_summary(aggregate, effective_budgets)
        write_json(iteration_dir / "budget_summary.json", budget_summary)
        write_json(
            iteration_dir / "result.json",
            {
                "experiment": "verifier_train",
                "status": status,
                "current_goal": current_goal,
                "current_budget": current_budget,
                "summary": aggregate,
                "budget_summary": budget_summary,
            },
        )
        _write_progress(
            iteration_dir,
            {
                "experiment": "verifier_train",
                "status": status,
                "current_goal": current_goal,
                "current_budget": current_budget,
                "summary": aggregate,
                "budget_summary": budget_summary,
            },
        )
        write_line_plot_svg(
            iteration_dir / "avg_verifier_success_by_budget.svg",
            title="Verifier Train: Average Verifier Success vs Training Budget",
            x_label="Training Budget",
            y_label="Average Verifier Success Rate",
            series={
                "reward-trained model": [
                    (float(row["budget"]), float(row["reward_model_avg_verifier_success"])) for row in budget_summary
                ],
                "verifier-trained model": [
                    (float(row["budget"]), float(row["verifier_model_avg_verifier_success"])) for row in budget_summary
                ],
            },
        )

    write_outputs(status="running")
    prepared: list[tuple[int, object, Path, object]] = []
    for goal_index, goal in enumerate(goals):
        goal_dir = iteration_dir / goal.id
        goal_dir.mkdir(parents=True, exist_ok=True)
        representative_scene = train_scenes[goal_index % len(train_scenes)]
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

    def run_goal(goal_index: int, goal, goal_dir: Path, reward_program):
        engine = PooltoolEngine()
        goal_rows: list[dict[str, object]] = []
        for training_budget in effective_budgets:
            budget_dir = goal_dir / f"budget_{training_budget:04d}"
            budget_dir.mkdir(parents=True, exist_ok=True)
            reward_model_path, reward_eval = _train_one_model(
                run_dir=iteration_dir,
                goal_id=goal.id,
                training_budget=training_budget,
                goal_dir=budget_dir,
                mode_name="reward",
                train_scenes=train_scenes,
                validation_scenes=validation_scenes,
                reward_program=reward_program,
                engine=engine,
                config=config,
                goal_index=goal_index,
                seed=seed,
                verifier_id=goal.verifier_id,
                progress_lock=lock,
            )
            verifier_model_path, verifier_eval = _train_one_model(
                run_dir=iteration_dir,
                goal_id=goal.id,
                training_budget=training_budget,
                goal_dir=budget_dir,
                mode_name="verifier",
                train_scenes=train_scenes,
                validation_scenes=validation_scenes,
                reward_program=reward_program,
                engine=engine,
                config=config,
                goal_index=goal_index,
                seed=seed,
                verifier_id=goal.verifier_id,
                progress_lock=lock,
            )
            append_log(
                iteration_dir / "run.log",
                f"goal {goal.id} budget={training_budget} reward_model_verifier={reward_eval['avg_chosen_verifier_success']} verifier_model_verifier={verifier_eval['avg_chosen_verifier_success']}",
            )
            goal_rows.append(
                {
                    "budget": training_budget,
                    "reward_model_path": reward_model_path,
                    "verifier_model_path": verifier_model_path,
                    "reward_model": reward_eval,
                    "verifier_model": verifier_eval,
                }
            )
            write_json(goal_dir / "result.json", {"goal": goal, "budget_results": goal_rows})
            with lock:
                aggregate[goal.id] = list(goal_rows)
                write_outputs(status="running", current_goal=goal.id, current_budget=training_budget)
        return goal.id

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = [executor.submit(run_goal, goal_index, goal, goal_dir, reward_program) for goal_index, goal, goal_dir, reward_program in prepared]
        for future in as_completed(futures):
            future.result()

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
    seed: int = 0,
    budgets: list[int] | None = None,
    workers: int = 4,
    iterations: int = 3,
) -> Path:
    run_dir = create_run_dir("verifier_train")
    config = load_config(config_file)
    effective_budgets = budgets or list(config.value_model.budgets)
    effective_iterations = max(1, int(iterations))
    write_json(
        run_dir / "request.json",
        {
            "goals_file": goals_file,
            "seed": seed,
            "config_file": config_file,
            "budgets": effective_budgets,
            "workers": workers,
            "iterations": effective_iterations,
            "experiment": "verifier_train",
        },
    )
    append_log(run_dir / "run.log", f"start budgets={effective_budgets} workers={workers} iterations={effective_iterations}")
    iteration_results: list[dict[str, object]] = []

    def write_root_outputs(*, status: str, current_iteration: int | None = None) -> None:
        budget_summary = _compute_aggregate_budget_summary(iteration_results, effective_budgets)
        write_json(run_dir / "iterations.json", iteration_results)
        write_json(
            run_dir / "result.json",
            {
                "experiment": "verifier_train",
                "status": status,
                "current_iteration": current_iteration,
                "iteration_count": effective_iterations,
                "completed_iterations": len(iteration_results),
                "budget_summary": budget_summary,
                "iterations": iteration_results,
            },
        )
        write_json(run_dir / "budget_summary.json", budget_summary)
        _write_progress(
            run_dir,
            {
                "experiment": "verifier_train",
                "status": status,
                "current_iteration": current_iteration,
                "iteration_count": effective_iterations,
                "completed_iterations": len(iteration_results),
                "budget_summary": budget_summary,
                "iterations": iteration_results,
            },
        )
        write_line_plot_svg(
            run_dir / "avg_verifier_success_by_budget.svg",
            title="Verifier Train: Average Verifier Success vs Training Budget",
            x_label="Training Budget",
            y_label="Average Verifier Success Rate",
            series={
                "reward-trained model": [
                    (float(row["budget"]), float(row["reward_model_avg_verifier_success_mean"])) for row in budget_summary
                ],
                "verifier-trained model": [
                    (float(row["budget"]), float(row["verifier_model_avg_verifier_success_mean"])) for row in budget_summary
                ],
            },
        )

    write_root_outputs(status="running")
    for iteration_index in range(effective_iterations):
        iteration_dir = run_dir / f"iteration_{iteration_index + 1:02d}"
        iteration_seed = seed + iteration_index * 10_000_000
        append_log(run_dir / "run.log", f"iteration={iteration_index + 1} seed={iteration_seed} start")
        result = _run_iteration(
            iteration_dir=iteration_dir,
            goals_file=goals_file,
            config_file=config_file,
            seed=iteration_seed,
            budgets=effective_budgets,
            workers=workers,
        )
        iteration_results.append(
            {
                "iteration": iteration_index + 1,
                "seed": iteration_seed,
                "iteration_dir": iteration_dir,
                "budget_summary": result["budget_summary"],
            }
        )
        write_root_outputs(status="running", current_iteration=iteration_index + 1)
        append_log(run_dir / "run.log", f"iteration={iteration_index + 1} done")

    write_root_outputs(status="done", current_iteration=effective_iterations)
    append_log(run_dir / "run.log", "done")
    return run_dir
