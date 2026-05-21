from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import typer

from adapters.registry import AdapterRegistry
from debug import debug_log, progress_log, set_debug_log_file
from datasets.validation import (
    build_dataset_readiness_report,
    render_dataset_readiness_payload,
)
from datasets import (
    DatasetBuildConfig,
    DatasetSplitConfig,
    build_trace_dataset,
    load_dataset_build_config,
)
from datasets.repair import repair_trace_dataset
from dataset import build_random_trace_dataset
from deepphy_eval import (
    DeepPHYEvaluationRequest,
    DeepPHYTextEvaluationRequest,
    build_evaluation_report,
    build_deepphy_text_report,
    default_run_id,
    evaluate_deepphy_request,
    evaluate_deepphy_text_request,
    evaluate_dataset_manifest,
    evaluate_live_request,
    EvaluationReport,
    load_deepphy_request,
    load_deepphy_text_request,
    load_trace,
    evaluate_trace_root,
    load_live_request,
    render_evaluation_payload,
    resolve_dataset_identity_from_manifest,
    resolve_live_request_identity,
    resolve_trace_root_identity,
    run_parity,
    summarize_deepphy_benchmark,
    write_deepphy_text_artifacts,
    write_evaluation_artifacts,
)
from generation import ACTIVE_ENVS, generate_traces
from llm_config import load_shared_llm_config
from paths import get_artifact_layout
from pattern_library_detectors import apply_pattern_library
from pattern_learning.patterns.config import (
    default_pattern_learning_config_path,
    load_pattern_labels,
    load_pattern_learning_config,
)
from pattern_library_evidence import write_pattern_evidence
from pattern_learning.patterns.losses import default_loss_configs
from pattern_learning.patterns.learning import (
    load_learning_samples_from_dataset_manifest,
    run_pattern_learning,
)
from pattern_library_paths import (
    default_pattern_evaluation_root,
    default_pattern_output_path,
    resolve_pattern_library_path,
    resolve_pattern_run_dir,
)
from pattern_library_registry import load_pattern_library
from pattern_learning.patterns.scoring import ChatCompletionsResponder
from predictions import (
    build_prediction_dataset,
    default_prediction_dataset_config_path,
    load_prediction_dataset_build_config,
)

app = typer.Typer(add_completion=False, help="DeepPHY rework environment tooling.")
_SHARED_LLM = load_shared_llm_config()


@app.callback()
def main_callback() -> None:
    """Root command group for DeepPHY rework tooling."""
    debug_log("cli.start", argv=list(__import__("sys").argv))


def _dataset_run_payload(run: object) -> dict[str, object]:
    return {
        "output_root": str(run.output_root),
        "dataset_name": run.dataset_name,
        "config_version": run.config_version,
        "config_path": str(run.config_path) if getattr(run, "config_path", None) else None,
        "env_ids": run.env_ids,
        "splits": [split.__dict__ for split in run.splits],
        "split_reports": {
            name: {
                "output_root": str(split_run.output_root),
                "per_environment": [report.__dict__ for report in split_run.per_environment],
            }
            for name, split_run in run.split_runs.items()
        },
    }


def _pattern_learning_payload(run: object) -> dict[str, Any]:
    return {
        "output_path": str(run.output_path),
        "artifact_root": str(run.artifact_root) if getattr(run, "artifact_root", None) else None,
        "evolution_record_path": str(run.evolution_record_path) if getattr(run, "evolution_record_path", None) else None,
        "library_id": run.library.library_id,
        "environment_ids": list(run.library.environment_ids),
        "iteration_count": run.iteration_count,
        "workers": getattr(run, "workers", None),
        "sample_count": run.sample_count,
        "train_sample_count": getattr(run, "train_sample_count", None),
        "validation_sample_count": getattr(run, "validation_sample_count", None),
        "dataset_id": run.dataset_id,
        "dataset_version": run.dataset_version,
        "scorer_config": dict(run.scorer_config),
        "average_score": run.average_score,
        "train_score": getattr(run, "train_score", None),
        "validation_score": getattr(run, "validation_score", None),
        "loss_breakdown": list(getattr(run, "loss_breakdown", [])),
        "score_details": [item.__dict__ for item in run.score_details],
        "readiness_notes": list(run.readiness_notes),
    }


def _prediction_dataset_payload(run: object) -> dict[str, Any]:
    return {
        "output_root": str(run.output_root),
        "dataset_name": run.dataset_name,
        "dataset_id": run.dataset_id,
        "config_version": run.config_version,
        "source_dataset_manifest": str(run.source_dataset_manifest),
        "env_ids": run.env_ids,
        "split_reports": run.split_reports,
        "config_path": str(run.config_path) if getattr(run, "config_path", None) else None,
    }


def _default_labels_path_for_env(env_id: str | None) -> Path:
    return get_artifact_layout().config_root / (env_id or "phyre") / "labels.txt"


def _default_dataset_manifest_for_env(env_id: str) -> Path:
    return Path(f"data/datasets/{env_id}/default_simulator_trace_dataset/dataset_manifest.json")


def _default_prediction_manifest_for_env(env_id: str) -> Path:
    return Path(f"data/predictions/{env_id}/pattern_learning_small/prediction_manifest.json")


def _default_pattern_output_for_env(env_id: str) -> Path:
    return default_pattern_output_path(env_id)


def _default_prediction_output_for_env(env_id: str) -> Path:
    return Path(f"data/predictions/{env_id}/pattern_learning_small")


def _default_prediction_dataset_name_for_env(env_id: str) -> str:
    return f"{env_id}_prediction_small"


def _derive_dataset_manifest_from_prediction_manifest(path: Path) -> Path:
    parts = list(path.parts)
    try:
        prediction_index = parts.index("predictions")
    except ValueError:
        return path.parent / "dataset_manifest.json"
    if len(parts) >= prediction_index + 3:
        dataset_parts = [
            *parts[:prediction_index],
            "datasets",
            parts[prediction_index + 1],
            parts[prediction_index + 2],
            "dataset_manifest.json",
        ]
        return Path(*dataset_parts)
    return path.parent / "dataset_manifest.json"


def _load_compare_request(path: Path) -> tuple[str, list[DeepPHYTextEvaluationRequest], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    requests_payload = payload.get("requests")
    if not isinstance(requests_payload, list) or not requests_payload:
        raise typer.BadParameter("compare request must include a non-empty requests list")
    requests = [
        DeepPHYTextEvaluationRequest.model_validate(item)
        for item in requests_payload
    ]
    seen_modes = {item.description_mode for item in requests}
    required_modes = {"trace_text", "pattern_text", "trace_text_plus_patterns"}
    missing_modes = sorted(required_modes - seen_modes)
    if missing_modes:
        raise typer.BadParameter(
            f"compare request must include {', '.join(sorted(required_modes))}; missing {', '.join(missing_modes)}"
        )
    compare_run_id = str(payload.get("run_id") or default_run_id())
    return compare_run_id, requests, payload


@app.command("probe-envs")
def probe_envs(output: str = typer.Option("table", "--output", help="table or json")) -> None:
    debug_log("cli.probe_envs.start", output=output)
    registry = AdapterRegistry()
    probes = registry.probe_all()

    if output == "json":
        typer.echo(json.dumps([probe.model_dump(mode="json") for probe in probes], indent=2))
        return

    if output != "table":
        raise typer.BadParameter("output must be 'table' or 'json'")

    typer.echo("env_id\texecution_mode\tavailability\tnative_metric")
    for probe in probes:
        typer.echo(
            f"{probe.env_id}\t{probe.execution_mode.value}\t{probe.availability.value}\t{probe.native_metric}"
        )


@app.command("serve-dashboard")
def serve_dashboard_command(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8787, "--port", min=1, max=65535),
) -> None:
    debug_log("cli.serve_dashboard.start", host=host, port=port)
    from dashboard import create_dashboard_app

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to serve the dashboard") from exc
    app_instance = create_dashboard_app()
    uvicorn.run(app_instance, host=host, port=port, log_level="info")


@app.command("generate-traces")
def generate_traces_command(
    limit: int = typer.Option(100, "--limit", min=1),
    env: list[str] = typer.Option([], "--env"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    debug_log("cli.generate_traces.start", limit=limit, env=list(env), output_dir=output_dir, force=force)
    run = generate_traces(
        limit=limit,
        env_ids=list(env) or None,
        output_root=output_dir,
        force=force,
    )
    typer.echo(json.dumps(
        {
            "output_root": str(run.output_root),
            "per_environment": [report.__dict__ for report in run.per_environment],
        },
        indent=2,
    ))


@app.command("build-random-trace-dataset")
def build_random_trace_dataset_command(
    train_count: int = typer.Option(1000, "--train-count", min=1),
    validation_count: int | None = typer.Option(None, "--validation-count", min=1),
    test_count: int = typer.Option(100, "--test-count", min=1),
    dataset_name: str | None = typer.Option(None, "--dataset-name"),
    train_offset: int | None = typer.Option(None, "--train-offset", min=0),
    validation_offset: int | None = typer.Option(None, "--validation-offset", min=0),
    test_offset: int | None = typer.Option(None, "--test-offset", min=0),
    env: list[str] = typer.Option([], "--env"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
    force: bool = typer.Option(False, "--force"),
    config: Optional[Path] = typer.Option(None, "--config"),
    split: list[str] = typer.Option([], "--split"),
) -> None:
    debug_log(
        "cli.build_random_trace_dataset.start",
        train_count=train_count,
        validation_count=validation_count,
        test_count=test_count,
        dataset_name=dataset_name,
        train_offset=train_offset,
        validation_offset=validation_offset,
        test_offset=test_offset,
        env=list(env),
        output_dir=output_dir,
        force=force,
        config=config,
        split=list(split),
    )
    if config is not None:
        explicit_flags_used = (
            train_count != 1000
            or validation_count is not None
            or test_count != 100
            or dataset_name is not None
            or train_offset is not None
            or validation_offset is not None
            or test_offset is not None
        )
        if explicit_flags_used:
            raise typer.BadParameter(
                "--config cannot be combined with explicit dataset-name/count/offset flags"
            )
        loaded_config = load_dataset_build_config(config)
        effective_config = (
            loaded_config.model_copy(update={"env_ids": list(env)})
            if env
            else loaded_config
        )
        run = build_trace_dataset(
            config=effective_config,
            output_root=output_dir,
            force=force,
            selected_splits=list(split) or None,
            config_path=config,
        )
    else:
        if (
            validation_count is not None
            or dataset_name is not None
            or train_offset is not None
            or validation_offset is not None
            or test_offset is not None
            or split
        ):
            effective_config = DatasetBuildConfig(
                dataset_name=dataset_name or "default_simulator_trace_dataset",
                env_ids=list(env) or list(ACTIVE_ENVS),
                splits=[
                    DatasetSplitConfig(
                        name="train",
                        sample_count=train_count,
                        sample_offset=train_offset if train_offset is not None else 0,
                    ),
                    DatasetSplitConfig(
                        name="validation",
                        sample_count=validation_count if validation_count is not None else 100,
                        sample_offset=validation_offset if validation_offset is not None else 500000,
                    ),
                    DatasetSplitConfig(
                        name="test",
                        sample_count=test_count,
                        sample_offset=test_offset if test_offset is not None else 1000000,
                    ),
                ],
                notes=["CLI-generated dataset config from explicit dashboard/command flags."],
            )
            run = build_trace_dataset(
                config=effective_config,
                output_root=output_dir,
                force=force,
                selected_splits=list(split) or None,
            )
        else:
            run = build_random_trace_dataset(
                train_count=train_count,
                test_count=test_count,
                env_ids=list(env) or None,
                output_root=output_dir,
                force=force,
            )
    typer.echo(json.dumps(_dataset_run_payload(run), indent=2))


@app.command("validate-trace-dataset")
def validate_trace_dataset_command(
    manifest: Path = typer.Option(..., "--manifest"),
) -> None:
    report = build_dataset_readiness_report(manifest)
    typer.echo(json.dumps(render_dataset_readiness_payload(report), indent=2))


@app.command("build-prediction-dataset")
def build_prediction_dataset_command(
    config: Path = typer.Option(default_prediction_dataset_config_path(), "--config"),
    env: Optional[str] = typer.Option(None, "--env"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    loaded_config = load_prediction_dataset_build_config(config)
    effective_env = env or loaded_config.env_id
    if effective_env is None and (
        loaded_config.output_path is None
        or loaded_config.source_dataset_manifest is None
        or loaded_config.dataset_name is None
    ):
        raise typer.BadParameter("--env is required when the config omits env-specific dataset paths")
    effective_output_root = output_dir or Path(
        loaded_config.output_path or _default_prediction_output_for_env(effective_env)
    )
    effective_source_dataset_manifest = Path(
        loaded_config.source_dataset_manifest or _default_dataset_manifest_for_env(effective_env)
    )
    effective_dataset_name = loaded_config.dataset_name or _default_prediction_dataset_name_for_env(effective_env)
    effective_config = loaded_config.model_copy(
        update={
            "env_id": effective_env,
            "output_path": str(effective_output_root),
            "source_dataset_manifest": str(effective_source_dataset_manifest),
            "dataset_name": effective_dataset_name,
        }
    )
    set_debug_log_file(effective_output_root / "debug.log")
    debug_log(
        "cli.build_prediction_dataset.start",
        config=config,
        env=effective_env,
        output_root=effective_output_root,
        force=force,
        source_dataset_manifest=effective_config.source_dataset_manifest,
        cutoff_count=effective_config.cutoff_count,
    )
    run = build_prediction_dataset(
        config=effective_config,
        responder=ChatCompletionsResponder(
            base_url=effective_config.prediction_base_url,
            model=effective_config.prediction_model,
            timeout=effective_config.prediction_timeout,
        ),
        output_root=output_dir,
        force=force,
        config_path=config,
    )
    debug_log(
        "cli.build_prediction_dataset.done",
        output_root=run.output_root,
        dataset_name=run.dataset_name,
    )
    typer.echo(json.dumps(_prediction_dataset_payload(run), indent=2))


@app.command("repair-trace-dataset")
def repair_trace_dataset_command(
    dataset_root: Path = typer.Option(..., "--dataset-root"),
    env: list[str] = typer.Option([], "--env"),
) -> None:
    env_ids = list(env) or ["iphyre", "pooltool", "phyre"]
    report = repair_trace_dataset(dataset_root, env_ids=env_ids)
    typer.echo(json.dumps(report.to_payload(), indent=2))


@app.command("evaluate")
def evaluate_command(
    dataset_manifest: Optional[Path] = typer.Option(None, "--dataset-manifest"),
    trace_root: Optional[Path] = typer.Option(None, "--trace-root"),
    live_request: Optional[Path] = typer.Option(None, "--live-request"),
    env: list[str] = typer.Option([], "--env"),
    split: list[str] = typer.Option([], "--split"),
    limit: Optional[int] = typer.Option(None, "--limit", min=1),
    output: str = typer.Option("json", "--output", help="json or table"),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
) -> None:
    source_count = sum(
        1 for value in [dataset_manifest, trace_root, live_request] if value is not None
    )
    if source_count != 1:
        raise typer.BadParameter(
            "exactly one source selector is required: --dataset-manifest OR --trace-root OR --live-request"
        )
    if split and dataset_manifest is None:
        raise typer.BadParameter("--split is valid only with --dataset-manifest")
    if output not in {"json", "table"}:
        raise typer.BadParameter("output must be 'json' or 'table'")

    chosen_run_id = run_id or default_run_id()
    if dataset_manifest is not None:
        results = evaluate_dataset_manifest(
            dataset_manifest,
            env_ids=list(env) or None,
            split_names=list(split) or None,
            limit=limit,
        )
        dataset_id, dataset_version, config_identity = resolve_dataset_identity_from_manifest(dataset_manifest)
        report = build_evaluation_report(
            run_id=chosen_run_id,
            source_kind="dataset_manifest",
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            config_identity=config_identity,
            results=results,
        )
    elif trace_root is not None:
        results = evaluate_trace_root(trace_root, env_ids=list(env) or None, limit=limit)
        dataset_id, dataset_version, config_identity = resolve_trace_root_identity(trace_root)
        report = build_evaluation_report(
            run_id=chosen_run_id,
            source_kind="trace_root",
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            config_identity=config_identity,
            results=results,
        )
    else:
        request = load_live_request(live_request)
        if env and request.env_id not in set(env):
            raise typer.BadParameter("--env filter excluded the provided live request env_id")
        results = [evaluate_live_request(request)]
        dataset_id, dataset_version, config_identity = resolve_live_request_identity(live_request)
        report = build_evaluation_report(
            run_id=chosen_run_id,
            source_kind="live_request",
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            config_identity=config_identity,
            results=results[:limit] if limit is not None else results,
        )

    report = write_evaluation_artifacts(report)
    if output == "json":
        typer.echo(json.dumps(render_evaluation_payload(report), indent=2))
        return

    typer.echo(
        "env_id\ttrace_count\tsuccess\tfailure\tunknown"
    )
    for summary in report.env_summaries:
        typer.echo(
            f"{summary.env_id}\t{summary.trace_count}\t{summary.success_count}\t{summary.failure_count}\t{summary.unknown_count}"
        )


@app.command("evaluate-parity")
def evaluate_parity_command(
    evaluation_report: Path = typer.Option(..., "--evaluation-report"),
    mode: str = typer.Option("reference", "--mode", help="reference or integration"),
    output: str = typer.Option("json", "--output", help="json or table"),
    expectations: Path = typer.Option(
        Path("configs"),
        "--expectations",
    ),
) -> None:
    if mode not in {"reference", "integration"}:
        raise typer.BadParameter("mode must be 'reference' or 'integration'")
    if output not in {"json", "table"}:
        raise typer.BadParameter("output must be 'json' or 'table'")

    with evaluation_report.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    report = EvaluationReport.model_validate(payload)
    parity = run_parity(
        report,
        mode=mode,
        expectations_path=expectations if mode == "reference" else None,
    )
    if output == "json":
        typer.echo(json.dumps(parity.model_dump(mode="json"), indent=2))
        return

    typer.echo("env_id\ttask_id\tstatus\tmessage")
    for check in parity.checks:
        typer.echo(f"{check.env_id}\t{check.task_id}\t{check.status}\t{check.message or ''}")


@app.command("evaluate-deepphy")
def evaluate_deepphy_command(
    request: Optional[Path] = typer.Option(None, "--request"),
    env: Optional[str] = typer.Option(None, "--env"),
    benchmark_preset: str = typer.Option(
        "full",
        "--benchmark-preset",
        help="full, quick-eval, or test",
    ),
    prompt_format: str = typer.Option("WM", "--prompt-format", help="WM only"),
    model: str = typer.Option(_SHARED_LLM.model, "--model"),
    description_mode: str = typer.Option(
        "pattern_text",
        "--description-mode",
        help="trace_text, pattern_text, or trace_text_plus_patterns",
    ),
    pattern_run_dir: Optional[Path] = typer.Option(None, "--pattern-run-dir"),
    pattern_library: Optional[Path] = typer.Option(None, "--pattern-library"),
    evidence_relpath: Optional[str] = typer.Option(None, "--evidence-relpath"),
    ollama_base_url: str = typer.Option(_SHARED_LLM.base_url, "--ollama-base-url"),
    llm_api_key: str = typer.Option(
        os.environ.get("LLM_ROUTER_API_KEY", ""),
        "--llm-api-key",
    ),
    model_timeout: int = typer.Option(_SHARED_LLM.timeout, "--model-timeout", min=0),
    capture_visuals: bool = typer.Option(
        False,
        "--capture-visuals/--no-capture-visuals",
        help="Capture initial-scene images and replay videos for attempts when supported.",
    ),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
    report_root: Optional[Path] = typer.Option(None, "--report-root"),
    output: str = typer.Option("json", "--output", help="json or table"),
) -> None:
    if output not in {"json", "table"}:
        raise typer.BadParameter("output must be 'json' or 'table'")
    if request is not None:
        req = load_deepphy_request(request)
    else:
        if pattern_run_dir is not None and pattern_library is not None:
            raise typer.BadParameter("--pattern-run-dir and --pattern-library are mutually exclusive")
        if env is None:
            raise typer.BadParameter("--env is required when --request is not provided")
        if benchmark_preset not in {"full", "quick-eval", "test"}:
            raise typer.BadParameter(
                "benchmark-preset must be 'full', 'quick-eval', or 'test'"
            )
        if description_mode not in {"trace_text", "pattern_text", "trace_text_plus_patterns"}:
            raise typer.BadParameter(
                "description-mode must be 'trace_text', 'pattern_text', or 'trace_text_plus_patterns'"
            )
        chosen_run_id = run_id or default_run_id()
        effective_pattern_run_dir = resolve_pattern_run_dir(pattern_run_dir) if pattern_run_dir is not None else None
        effective_pattern_library = (
            None
            if pattern_run_dir is not None
            else (
                str(resolve_pattern_library_path(pattern_library))
                if pattern_library is not None
                else None
            )
        )
        effective_report_root = (
            default_pattern_evaluation_root(pattern_run_dir=effective_pattern_run_dir, evaluation_run_id=chosen_run_id)
            if effective_pattern_run_dir is not None
            else report_root
        )
        req = DeepPHYEvaluationRequest(
            env_id=env,
            model_name=model,
            benchmark_preset=benchmark_preset,
            prompt_format="WM",
            description_mode=description_mode,
            pattern_run_dir=str(effective_pattern_run_dir) if effective_pattern_run_dir is not None else None,
            pattern_library_path=effective_pattern_library,
            evidence_relpath=evidence_relpath,
            run_id=chosen_run_id,
            report_root=str(effective_report_root) if effective_report_root is not None else None,
            capture_visuals=capture_visuals,
            benchmark_kwargs={"base_url": ollama_base_url, "timeout": model_timeout, "api_key": llm_api_key},
        )

    report = evaluate_deepphy_request(req)
    report = write_deepphy_text_artifacts(report)

    if output == "json":
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
        return

    benchmark = summarize_deepphy_benchmark(req.env_id, preset=req.benchmark_preset)
    env_summary = report.env_summaries[0] if report.env_summaries else None
    success_count = env_summary.success_count if env_summary is not None else 0
    task_count = env_summary.task_count if env_summary is not None else benchmark.task_count
    typer.echo("env_id\tbenchmark\ttask_count\tsuccess_count\tsuccess_rate")
    success_rate = (success_count / task_count) if task_count else 0.0
    typer.echo(
        f"{req.env_id}\t{benchmark.label}\t{task_count}\t{success_count}\t{success_rate:.4f}"
    )


@app.command("pattern-learn")
def pattern_learn_command(
    env: Optional[str] = typer.Option(None, "--env"),
    primitive_label: list[str] = typer.Option([], "--primitive-label"),
    labels_file: Optional[Path] = typer.Option(None, "--labels-file"),
    config: Optional[Path] = typer.Option(None, "--config"),
    output_path: Optional[Path] = typer.Option(None, "--output-path"),
    dataset_manifest: Optional[Path] = typer.Option(None, "--dataset-manifest"),
    generator_model: Optional[str] = typer.Option(None, "--generator-model"),
    split: list[str] = typer.Option([], "--split"),
    validation_split: list[str] = typer.Option([], "--validation-split"),
    limit: Optional[int] = typer.Option(None, "--limit", min=1),
    iteration_count: int = typer.Option(8, "--iteration-count", min=1),
    workers: int = typer.Option(1, "--workers", min=1),
    seed: Optional[int] = typer.Option(None, "--seed"),
    population_size: Optional[int] = typer.Option(64, "--population-size", min=1),
    patience_factor: float = typer.Option(0.35, "--patience-factor", min=0.000001, max=1.0),
    min_candidate_evaluations: int = typer.Option(100, "--min-candidate-evaluations", min=1),
    no_improvement_patience_evaluations: int = typer.Option(15, "--no-improvement-patience-evaluations", min=1),
    max_candidate_evaluations: int = typer.Option(250, "--max-candidate-evaluations", min=1),
    elite_fraction: Optional[float] = typer.Option(None, "--elite-fraction", min=0.000001, max=1.0),
    children_per_elite: Optional[int] = typer.Option(None, "--children-per-elite", min=0),
    crossover_rate: Optional[float] = typer.Option(None, "--crossover-rate", min=0.0, max=1.0),
    primitive_source: Optional[str] = typer.Option(None, "--primitive-source"),
    resume: bool = typer.Option(False, "--resume"),
) -> None:
    loaded_config = load_pattern_learning_config(config) if config is not None else None
    effective_env = env or (loaded_config.env_id if loaded_config is not None else None)
    effective_output_path = output_path or (
        Path(loaded_config.output_path) if loaded_config is not None and loaded_config.output_path else None
    )
    effective_dataset_manifest = dataset_manifest or (
        Path(loaded_config.dataset_manifest) if loaded_config is not None and loaded_config.dataset_manifest else None
    )
    effective_split = list(split) or (list(loaded_config.split) if loaded_config is not None else [])
    effective_validation_split = list(validation_split) or (
        list(loaded_config.validation_split) if loaded_config is not None else []
    )
    effective_limit = limit if limit is not None else (loaded_config.limit if loaded_config is not None else None)
    effective_generator_model = (
        generator_model
        if generator_model is not None
        else (
            loaded_config.generator_model
            if loaded_config is not None
            else _SHARED_LLM.model
        )
    )
    effective_generator_base_url = loaded_config.generator_base_url if loaded_config is not None else _SHARED_LLM.base_url
    effective_generator_timeout = loaded_config.generator_timeout if loaded_config is not None else _SHARED_LLM.timeout
    effective_generator_retries = loaded_config.generator_retries if loaded_config is not None else 3
    effective_iteration_count = (
        iteration_count if iteration_count != 8 or loaded_config is None else loaded_config.iteration_count
    )
    effective_workers = (
        workers if workers != 1 or loaded_config is None else loaded_config.workers
    )
    effective_primitive_source = primitive_source or (
        loaded_config.primitive_source if loaded_config is not None else None
    )
    effective_seed = seed if seed is not None else (loaded_config.seed if loaded_config is not None else 42)
    effective_population_size = (
        population_size if population_size is not None else (loaded_config.population_size if loaded_config is not None else 64)
    )
    effective_elite_fraction = (
        elite_fraction if elite_fraction is not None else (loaded_config.elite_fraction if loaded_config is not None else 0.25)
    )
    effective_children_per_elite = (
        children_per_elite
        if children_per_elite is not None
        else (loaded_config.children_per_elite if loaded_config is not None else 2)
    )
    effective_patience_factor = (
        patience_factor if patience_factor != 0.35 or loaded_config is None else loaded_config.patience_factor
    )
    effective_min_candidate_evaluations = (
        min_candidate_evaluations
        if min_candidate_evaluations != 100 or loaded_config is None
        else loaded_config.min_candidate_evaluations
    )
    effective_no_improvement_patience_evaluations = (
        no_improvement_patience_evaluations
        if no_improvement_patience_evaluations != 15 or loaded_config is None
        else loaded_config.no_improvement_patience_evaluations
    )
    effective_max_candidate_evaluations = (
        max_candidate_evaluations
        if max_candidate_evaluations != 250 or loaded_config is None
        else loaded_config.max_candidate_evaluations
    )
    effective_crossover_rate = (
        crossover_rate if crossover_rate is not None else (loaded_config.crossover_rate if loaded_config is not None else 1.0)
    )
    effective_accept_margin = loaded_config.accept_margin if loaded_config is not None else 0.001
    effective_max_repairs = loaded_config.max_repairs if loaded_config is not None else 2
    effective_mutate_temperature = loaded_config.mutate_temperature if loaded_config is not None else 0.7
    effective_mutate_max_tokens = (
        loaded_config.mutate_max_tokens if loaded_config is not None else 8192
    )
    effective_losses = loaded_config.losses if loaded_config is not None else None
    if effective_env is None:
        raise typer.BadParameter("--env is required when --config is not provided or does not set env_id")
    overlap = sorted(set(effective_split) & set(effective_validation_split))
    if overlap:
        raise typer.BadParameter(
            "--split and --validation-split must not overlap: " + ", ".join(overlap)
        )
    if effective_output_path is None:
        effective_output_path = _default_pattern_output_for_env(effective_env)
    effective_artifact_root = (
        effective_output_path.parent
        if effective_output_path.name == "pattern_library.json"
        else effective_output_path.parent / effective_output_path.stem
    )
    if effective_dataset_manifest is None:
        effective_dataset_manifest = _default_dataset_manifest_for_env(effective_env)
    effective_labels_file = labels_file or (
        Path(loaded_config.labels_path) if loaded_config is not None and loaded_config.labels_path else _default_labels_path_for_env(effective_env)
    )
    effective_primitive_labels = list(primitive_label) or load_pattern_labels(effective_labels_file)
    if effective_output_path is not None:
        set_debug_log_file(effective_output_path.parent / "debug.log")

    debug_log(
        "cli.pattern_learn.start",
        env=env,
        labels_file=labels_file,
        config=config,
        output_path=effective_output_path or output_path,
        dataset_manifest=effective_dataset_manifest or dataset_manifest,
        split=effective_split or list(split),
        validation_split=effective_validation_split or list(validation_split),
        limit=effective_limit if effective_limit is not None else limit,
        generator_model=effective_generator_model,
        generator_base_url=effective_generator_base_url,
        iteration_count=effective_iteration_count,
        workers=effective_workers,
        seed=effective_seed,
        population_size=effective_population_size,
        elite_fraction=effective_elite_fraction,
        children_per_elite=effective_children_per_elite,
        patience_factor=effective_patience_factor,
        min_candidate_evaluations=effective_min_candidate_evaluations,
        no_improvement_patience_evaluations=effective_no_improvement_patience_evaluations,
        max_candidate_evaluations=effective_max_candidate_evaluations,
        crossover_rate=effective_crossover_rate,
        primitive_source=effective_primitive_source,
        resume=resume,
        losses=[loss.kind for loss in effective_losses] if effective_losses is not None else None,
    )

    active_losses = list(effective_losses) if effective_losses is not None else default_loss_configs()
    if effective_dataset_manifest is None:
        raise typer.BadParameter("--dataset-manifest is required")
    if not effective_dataset_manifest.is_file():
        raise FileNotFoundError(f"Pattern learning dataset manifest not found: {effective_dataset_manifest}")
    progress_log(
        "cli.pattern_learn.start",
        env=effective_env,
        output_path=effective_output_path,
        dataset_manifest=effective_dataset_manifest,
        iteration_count=effective_iteration_count,
        workers=effective_workers,
        primitive_label_count=len(effective_primitive_labels),
        resume=resume,
    )

    debug_log(
        "cli.pattern_learn.resolved",
        env=effective_env,
        output_path=effective_output_path,
        dataset_manifest=effective_dataset_manifest,
        split=effective_split,
        validation_split=effective_validation_split,
        limit=effective_limit,
        generator_model=effective_generator_model,
        generator_base_url=effective_generator_base_url,
    )
    samples, dataset_meta = load_learning_samples_from_dataset_manifest(
        effective_dataset_manifest,
        split_names=effective_split or None,
        limit=effective_limit,
    )
    validation_samples, validation_meta = load_learning_samples_from_dataset_manifest(
        effective_dataset_manifest,
        split_names=effective_validation_split or None,
        limit=effective_limit,
    )
    debug_log(
        "cli.pattern_learn.samples_loaded",
        sample_count=len(samples),
        validation_sample_count=len(validation_samples),
        dataset_id=dataset_meta.get("dataset_id"),
        dataset_version=dataset_meta.get("dataset_version"),
    )
    progress_log(
        "cli.pattern_learn.samples_loaded",
        sample_count=len(samples),
        validation_sample_count=len(validation_samples),
        dataset_id=dataset_meta.get("dataset_id"),
    )

    run = run_pattern_learning(
        environment_id=effective_env,
        primitive_labels=effective_primitive_labels,
        samples=samples,
        validation_samples=validation_samples,
        output_path=effective_output_path,
        scorer_config={
            "dataset_manifest": str(effective_dataset_manifest) if effective_dataset_manifest is not None else None,
            "train_split": list(effective_split),
            "validation_split": list(effective_validation_split),
            "generator_model": effective_generator_model,
            "generator_base_url": effective_generator_base_url,
            "generator_timeout": effective_generator_timeout,
            "generator_retries": effective_generator_retries,
            "score_metric": "correlation_structure",
            "patience_factor": effective_patience_factor,
            "min_candidate_evaluations": effective_min_candidate_evaluations,
            "no_improvement_patience_evaluations": effective_no_improvement_patience_evaluations,
            "max_candidate_evaluations": effective_max_candidate_evaluations,
            "losses": [loss.model_dump(mode="json") for loss in active_losses],
            "labels_file": str(effective_labels_file),
            "config_path": str(config) if config is not None else None,
        },
        iteration_count=effective_iteration_count,
        label_workers=effective_workers,
        primitive_source=effective_primitive_source or str(effective_labels_file),
        dataset_id=dataset_meta.get("dataset_id"),
        dataset_version=dataset_meta.get("dataset_version"),
        generator_model=effective_generator_model,
        generator_base_url=effective_generator_base_url,
        generator_timeout=effective_generator_timeout,
        generator_retries=effective_generator_retries,
        seed=effective_seed,
        population_size=effective_population_size,
        elite_fraction=effective_elite_fraction,
        children_per_elite=effective_children_per_elite,
        patience_factor=effective_patience_factor,
        min_candidate_evaluations=effective_min_candidate_evaluations,
        no_improvement_patience_evaluations=effective_no_improvement_patience_evaluations,
        max_candidate_evaluations=effective_max_candidate_evaluations,
        crossover_rate=effective_crossover_rate,
        accept_margin=effective_accept_margin,
        max_repairs=effective_max_repairs,
        mutate_temperature=effective_mutate_temperature,
        mutate_max_tokens=effective_mutate_max_tokens,
        losses=active_losses,
        artifact_root=effective_artifact_root,
        resume=resume,
    )
    debug_log(
        "cli.pattern_learn.completed",
        output_path=run.output_path,
        average_score=run.average_score,
        sample_count=run.sample_count,
    )
    progress_log(
        "cli.pattern_learn.completed",
        output_path=run.output_path,
        average_score=run.average_score,
        sample_count=run.sample_count,
    )
    typer.echo(json.dumps(_pattern_learning_payload(run), indent=2))


@app.command("pattern-annotate")
def pattern_annotate_command(
    pattern_library: Path = typer.Option(..., "--pattern-library"),
    trace_root: Optional[Path] = typer.Option(None, "--trace-root"),
    sample_dir: Optional[Path] = typer.Option(None, "--sample-dir"),
    evidence_relpath: str = typer.Option("trace.evidence.json", "--evidence-relpath"),
    limit: Optional[int] = typer.Option(None, "--limit", min=1),
) -> None:
    source_count = sum(1 for value in [trace_root, sample_dir] if value is not None)
    if source_count != 1:
        raise typer.BadParameter("exactly one source selector is required: --trace-root OR --sample-dir")

    library = load_pattern_library(pattern_library)
    trace_paths = []
    if trace_root is not None:
        trace_paths = sorted(trace_root.rglob("trace.json"), key=lambda path: path.as_posix())
    else:
        trace_paths = [sample_dir / "trace.json"]
    if limit is not None:
        trace_paths = trace_paths[:limit]
    if not trace_paths:
        raise typer.BadParameter("no trace.json files were found for annotation")

    annotations: list[dict[str, Any]] = []
    for trace_path in trace_paths:
        trace = load_trace(trace_path)
        patterns = apply_pattern_library(trace, library)
        evidence_path = write_pattern_evidence(
            trace_path.parent,
            library,
            patterns=patterns,
            evidence_relpath=evidence_relpath,
        )
        annotations.append(
            {
                "trace_path": str(trace_path),
                "evidence_path": str(evidence_path),
                "pattern_count": len(patterns),
                "task_id": trace.task_id,
                "env_id": trace.environment,
            }
        )

    typer.echo(
        json.dumps(
            {
                "pattern_library_path": str(pattern_library),
                "library_id": library.library_id,
                "annotated_count": len(annotations),
                "evidence_relpath": evidence_relpath,
                "annotations": annotations,
            },
            indent=2,
        )
    )


@app.command("pattern-compare")
def pattern_compare_command(
    request: Path = typer.Option(..., "--request"),
    output: str = typer.Option("json", "--output", help="json or table"),
) -> None:
    if output not in {"json", "table"}:
        raise typer.BadParameter("output must be 'json' or 'table'")

    compare_run_id, requests, payload = _load_compare_request(request)
    layout = get_artifact_layout()
    results = []
    for item in requests:
        isolated_output_root = (
            layout.evaluations_root
            / compare_run_id
            / "deepphy_text_compare"
            / item.description_mode
            / item.env_id
            / item.task_id.replace("/", "__").replace(":", "__")
        )
        run_request = item.model_copy(
            update={
                "run_id": compare_run_id,
                "output_root": str(isolated_output_root),
            }
        )
        results.append(evaluate_deepphy_text_request(run_request))

    report = build_deepphy_text_report(run_id=compare_run_id, results=results)
    report = write_deepphy_text_artifacts(report)
    response_payload = {
        "run_id": compare_run_id,
        "comparison_name": payload.get("comparison_name"),
        "notes": payload.get("notes", []),
        "report": report.model_dump(mode="json"),
    }
    if output == "json":
        typer.echo(json.dumps(response_payload, indent=2))
        return

    typer.echo("env_id\ttask_id\tdescription_mode\tsuccess\tterminal_status")
    for item in report.results:
        typer.echo(
            f"{item.env_id}\t{item.task_id}\t{item.description_mode}\t{item.success}\t{item.terminal_status or ''}"
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
