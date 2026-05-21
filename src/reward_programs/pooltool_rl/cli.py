from __future__ import annotations

import json
from pathlib import Path

import typer

from .config import config_path, project_root
from .annotation_video import (
    default_overlay_output_path,
    export_annotation_overlay_video,
    export_single_scene_annotation_overlay,
)
from . import single_scene, unverified_opt, verifier_opt, verifier_train


app = typer.Typer(help="Standalone PoolTool reward-program experiments.")
single_scene_app = typer.Typer(help="Run a single random-scene optimization.")
verifier_opt_app = typer.Typer(help="Run verifier-backed optimization experiments.")
verifier_train_app = typer.Typer(help="Train and compare verifier-backed value models.")
unverified_opt_app = typer.Typer(help="Run unverified optimization experiments.")
annotations_app = typer.Typer(help="Render supervised-library annotation overlays on PoolTool videos.")

app.add_typer(single_scene_app, name="single-scene")
app.add_typer(verifier_opt_app, name="verifier-opt")
app.add_typer(verifier_train_app, name="verifier-train")
app.add_typer(unverified_opt_app, name="unverified-opt")
app.add_typer(annotations_app, name="annotations")


def default_goals_file() -> Path:
    return project_root() / "configs" / "goals.unverified.json"


def default_verified_goals_file() -> Path:
    return project_root() / "configs" / "goals.verified.json"


def default_verified_train_goals_file() -> Path:
    return project_root() / "configs" / "goals.verified.train.json"


def default_unverified_goals_file() -> Path:
    return project_root() / "configs" / "goals.unverified.json"


@single_scene_app.command("run")
def run_single_scene(
    goal_id: str = typer.Option(..., "--goal-id"),
    goals_file: Path = typer.Option(default_goals_file(), "--goals-file"),
    config_file: Path = typer.Option(config_path(), "--config"),
    budget: int | None = typer.Option(None, "--budget"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    run_dir = single_scene.run(goal_id=goal_id, goals_file=goals_file, config_file=config_file, budget=budget, seed=seed)
    typer.echo(run_dir)


@single_scene_app.command("sample")
def sample_single_scene(
    seed: int = typer.Option(0, "--seed"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """
    Sample one random scene and write its copied assets + metadata into a new timestamped run dir.
    """
    run_dir = single_scene.sample(seed=seed, run_id=run_id)
    state_path = run_dir / "initial_state.json"
    if state_path.is_file():
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        typer.echo(f"RUN_DIR: {run_dir}")
        typer.echo(f"SCENE: {payload.get('sample_id')}")
        typer.echo(f"IMAGE: {run_dir / 'initial_scene_overhead.png'}")
        typer.echo("INITIAL_STATE:")
        typer.echo(state_path.read_text(encoding="utf-8").rstrip())
    else:
        typer.echo(run_dir)


@single_scene_app.command("synthesize")
def synthesize_single_scene(
    run_dir: Path | None = typer.Option(None, "--run-dir"),
    goal_text: str = typer.Option(..., "--goal-text"),
    config_file: Path = typer.Option(config_path(), "--config"),
    use_images: bool = typer.Option(True, "--use-images/--no-use-images"),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
) -> None:
    """
    Synthesize a reward DSL program for the sampled scene given a natural-language goal.
    """
    resolved = single_scene.resolve_single_scene_run_dir(run_dir)
    single_scene.synthesize(
        run_dir=resolved,
        goal_text=goal_text,
        config_file=config_file,
        use_images=use_images,
        progress=progress,
    )
    dsl_path = resolved / "program.dsl"
    typer.echo(f"RUN_DIR: {resolved}")
    typer.echo(f"GOAL: {goal_text}")
    typer.echo(f"DSL_PATH: {dsl_path}")
    typer.echo("DSL:")
    if dsl_path.is_file():
        typer.echo(dsl_path.read_text(encoding="utf-8").rstrip())
    else:
        typer.echo("(missing program.dsl)")


@single_scene_app.command("optimize")
def optimize_single_scene(
    run_dir: Path | None = typer.Option(None, "--run-dir"),
    config_file: Path = typer.Option(config_path(), "--config"),
    budget: int | None = typer.Option(None, "--budget"),
    seed: int = typer.Option(0, "--seed"),
    render_best: bool = typer.Option(True, "--render-best/--no-render-best"),
    until_success: bool = typer.Option(False, "--until-success/--no-until-success"),
    max_attempts: int = typer.Option(1, "--max-attempts"),
) -> None:
    """
    Optimize an action for the synthesized reward program on the sampled scene.
    """
    resolved = single_scene.resolve_single_scene_run_dir(run_dir)
    single_scene.optimize(
        run_dir=resolved,
        config_file=config_file,
        budget=budget,
        seed=seed,
        render_best=render_best,
        until_success=until_success,
        max_attempts=max_attempts,
    )
    shot_path = resolved / "shot.json"
    video_path = resolved / "simulation_overhead.mp4"
    typer.echo(f"RUN_DIR: {resolved}")
    typer.echo(f"SHOT_PATH: {shot_path}")
    if shot_path.is_file():
        shot = json.loads(shot_path.read_text(encoding="utf-8"))
        typer.echo("SHOT:")
        typer.echo(shot_path.read_text(encoding="utf-8").rstrip())
        if video_path.is_file():
            typer.echo(f"VIDEO: {video_path}")
    else:
        typer.echo("(missing shot.json)")


@single_scene_app.command("optimize-repeat")
def optimize_repeat_single_scene(
    run_dir: Path | None = typer.Option(None, "--run-dir"),
    config_file: Path = typer.Option(config_path(), "--config"),
    budget: int | None = typer.Option(None, "--budget"),
    seed: int = typer.Option(0, "--seed"),
    render_best: bool = typer.Option(True, "--render-best/--no-render-best"),
    until_success: bool = typer.Option(False, "--until-success/--no-until-success"),
    max_attempts: int = typer.Option(1, "--max-attempts"),
) -> None:
    """
    Repeat optimization using the saved DSL program in the run dir (no resynthesis),
    writing a new timestamped optimization session under log/optimizations/.
    """
    resolved = single_scene.resolve_single_scene_run_dir(run_dir)
    single_scene.optimize_repeat(
        run_dir=resolved,
        config_file=config_file,
        budget=budget,
        seed=seed,
        render_best=render_best,
        until_success=until_success,
        max_attempts=max_attempts,
    )
    # Reuse the same clean reporting as optimize.
    shot_path = resolved / "shot.json"
    video_path = resolved / "simulation_overhead.mp4"
    typer.echo(f"RUN_DIR: {resolved}")
    typer.echo(f"SHOT_PATH: {shot_path}")
    if shot_path.is_file():
        typer.echo("SHOT:")
        typer.echo(shot_path.read_text(encoding="utf-8").rstrip())
        if video_path.is_file():
            typer.echo(f"VIDEO: {video_path}")
    else:
        typer.echo("(missing shot.json)")


@single_scene_app.command("annotate-video")
def annotate_single_scene_video(
    run_dir: Path | None = typer.Option(None, "--run-dir"),
    output: Path | None = typer.Option(None, "--output"),
    annotation_file: Path | None = typer.Option(None, "--annotation-file"),
    label: list[str] | None = typer.Option(None, "--label"),
    confidence_threshold: float = typer.Option(0.0, "--confidence-threshold"),
) -> None:
    resolved = single_scene.resolve_single_scene_run_dir(run_dir)
    destination = export_single_scene_annotation_overlay(
        run_dir=resolved,
        output_path=output,
        annotation_path=annotation_file,
        selected_labels=label,
        confidence_threshold=confidence_threshold,
    )
    typer.echo(destination)


@verifier_opt_app.command("run")
def run_verifier_opt(
    goals_file: Path = typer.Option(default_verified_goals_file(), "--goals-file"),
    config_file: Path = typer.Option(config_path(), "--config"),
    scene_count: int | None = typer.Option(None, "--scene-count"),
    seed_count: int | None = typer.Option(None, "--seed-count"),
    budgets: list[int] | None = typer.Option(None, "--budget"),
    seed: int = typer.Option(0, "--seed"),
    workers: int = typer.Option(4, "--workers"),
    iterations: int = typer.Option(3, "--iterations"),
) -> None:
    run_dir = verifier_opt.run(
        goals_file=goals_file,
        config_file=config_file,
        scene_count=scene_count,
        seed_count=seed_count,
        budgets=budgets,
        seed=seed,
        workers=workers,
        iterations=iterations,
    )
    typer.echo(run_dir)


@verifier_train_app.command("run")
def run_verifier_train(
    goals_file: Path = typer.Option(default_verified_train_goals_file(), "--goals-file"),
    config_file: Path = typer.Option(config_path(), "--config"),
    seed: int = typer.Option(0, "--seed"),
    budgets: list[int] | None = typer.Option(None, "--budget"),
    workers: int = typer.Option(4, "--workers"),
    iterations: int = typer.Option(3, "--iterations"),
) -> None:
    run_dir = verifier_train.run(
        goals_file=goals_file,
        config_file=config_file,
        seed=seed,
        budgets=budgets,
        workers=workers,
        iterations=iterations,
    )
    typer.echo(run_dir)


@unverified_opt_app.command("run")
def run_unverified_opt(
    goals_file: Path = typer.Option(default_unverified_goals_file(), "--goals-file"),
    config_file: Path = typer.Option(config_path(), "--config"),
    scene_count: int | None = typer.Option(None, "--scene-count"),
    seed_count: int | None = typer.Option(None, "--seed-count"),
    budgets: list[int] | None = typer.Option(None, "--budget"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    run_dir = unverified_opt.run(
        goals_file=goals_file,
        config_file=config_file,
        scene_count=scene_count,
        seed_count=seed_count,
        budgets=budgets,
        seed=seed,
    )
    typer.echo(run_dir)


@annotations_app.command("render")
def render_annotation_video(
    trace_path: Path = typer.Option(..., "--trace-path"),
    video_path: Path = typer.Option(..., "--video-path"),
    output: Path | None = typer.Option(None, "--output"),
    annotation_file: Path | None = typer.Option(None, "--annotation-file"),
    label: list[str] | None = typer.Option(None, "--label"),
    confidence_threshold: float = typer.Option(0.0, "--confidence-threshold"),
) -> None:
    destination = export_annotation_overlay_video(
        trace_path=trace_path,
        video_path=video_path,
        output_path=output or default_overlay_output_path(video_path),
        annotation_path=annotation_file,
        selected_labels=label,
        confidence_threshold=confidence_threshold,
    )
    typer.echo(destination)


def main() -> None:
    app()
