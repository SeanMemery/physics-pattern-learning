from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from deepphy_eval import (
    DeepPHYEvaluationRequest,
    default_run_id,
    evaluate_deepphy_request,
    load_deepphy_request,
    summarize_deepphy_benchmark,
    write_deepphy_text_artifacts,
)
from llm_config import load_shared_llm_config
from pattern_library_paths import (
    default_pattern_evaluation_root,
    resolve_pattern_library_path,
    resolve_pattern_run_dir,
)


_SHARED_LLM = load_shared_llm_config()


def _default_llm_api_key() -> str:
    value = str(__import__("os").environ.get("LLM_ROUTER_API_KEY", "") or "").strip()
    if value:
        return value
    try:
        return str(getattr(_SHARED_LLM, "api_key", "") or "").strip()
    except Exception:
        return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m deepphy_eval.evaluate_deepphy",
        description="Run the fixed DeepPHY benchmark evaluation without the Typer CLI.",
    )
    parser.add_argument("--request", type=Path, default=None)
    parser.add_argument("--env", default=None)
    parser.add_argument("--benchmark-preset", default="full")
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--benchmark-seed", type=int, default=None)
    parser.add_argument("--prompt-format", default="WM")
    parser.add_argument("--model", default=_SHARED_LLM.model)
    parser.add_argument("--description-mode", default="pattern_text")
    parser.add_argument("--pattern-run-dir", type=Path, default=None)
    parser.add_argument("--pattern-library", type=Path, default=None)
    parser.add_argument("--evidence-relpath", default=None)
    parser.add_argument("--ollama-base-url", default=_SHARED_LLM.base_url)
    parser.add_argument("--llm-api-key", default=_default_llm_api_key())
    parser.add_argument("--model-timeout", type=int, default=_SHARED_LLM.timeout)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--capture-visuals",
        action="store_true",
        help="Capture initial-scene images and attempt replay videos when the adapter supports them.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--report-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", default="json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.output not in {"json", "table"}:
        parser.error("output must be 'json' or 'table'")
    if args.request is not None:
        req = load_deepphy_request(args.request)
    else:
        if args.pattern_run_dir is not None and args.pattern_library is not None:
            parser.error("--pattern-run-dir and --pattern-library are mutually exclusive")
        if not args.env:
            parser.error("--env is required when --request is not provided")
        if args.benchmark_preset not in {"full", "quick-eval", "test"}:
            parser.error("benchmark-preset must be 'full', 'quick-eval', or 'test'")
        if args.description_mode not in {
            "trace_text",
            "pattern_text",
            "trace_text_plus_patterns",
        }:
            parser.error(
                "description-mode must be 'trace_text', 'pattern_text', or 'trace_text_plus_patterns'"
            )
        chosen_run_id = args.run_id or default_run_id()
        effective_pattern_run_dir = (
            resolve_pattern_run_dir(args.pattern_run_dir)
            if args.pattern_run_dir is not None
            else None
        )
        effective_pattern_library = (
            None
            if effective_pattern_run_dir is not None
            else (
                str(resolve_pattern_library_path(args.pattern_library))
                if args.pattern_library is not None
                else None
            )
        )
        effective_report_root = (
            args.report_root
            if args.report_root is not None
            else (
                default_pattern_evaluation_root(
                    pattern_run_dir=effective_pattern_run_dir,
                    evaluation_run_id=chosen_run_id,
                )
                if effective_pattern_run_dir is not None
                else None
            )
        )
        if effective_report_root is not None and effective_report_root.exists() and not args.resume:
            shutil.rmtree(effective_report_root)
        benchmark_kwargs = {
            "base_url": args.ollama_base_url,
            "api_key": args.llm_api_key,
            "timeout": args.model_timeout,
        }
        if args.max_attempts is not None:
            benchmark_kwargs["max_attempts"] = int(args.max_attempts)
        if args.max_steps is not None:
            benchmark_kwargs["max_steps"] = int(args.max_steps)
        req = DeepPHYEvaluationRequest(
            env_id=args.env,
            model_name=args.model,
            benchmark_preset=args.benchmark_preset,
            task_limit=args.task_limit,
            benchmark_seed=args.benchmark_seed,
            prompt_format="WM",
            description_mode=args.description_mode,
            pattern_run_dir=(
                str(effective_pattern_run_dir)
                if effective_pattern_run_dir is not None
                else None
            ),
            pattern_library_path=effective_pattern_library,
            evidence_relpath=args.evidence_relpath,
            run_id=chosen_run_id,
            report_root=(
                str(effective_report_root)
                if effective_report_root is not None
                else None
            ),
            capture_visuals=bool(args.capture_visuals),
            workers=max(1, int(args.workers)),
            benchmark_kwargs=benchmark_kwargs,
        )

    report = evaluate_deepphy_request(req)
    report = write_deepphy_text_artifacts(report)

    if args.output == "json":
        print(json.dumps(report.model_dump(mode="json"), indent=2))
        return 0

    benchmark = summarize_deepphy_benchmark(
        req.env_id,
        preset=req.benchmark_preset,
        task_limit=req.task_limit,
    )
    env_summary = report.env_summaries[0] if report.env_summaries else None
    success_count = env_summary.success_count if env_summary is not None else 0
    task_count = env_summary.task_count if env_summary is not None else benchmark.task_count
    success_rate = (success_count / task_count) if task_count else 0.0
    print("env_id\tbenchmark\ttask_count\tsuccess_count\tsuccess_rate")
    print(
        f"{req.env_id}\t{benchmark.label}\t{task_count}\t{success_count}\t{success_rate:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
