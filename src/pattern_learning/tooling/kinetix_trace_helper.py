from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adapters.registry import AdapterRegistry
from generation import _prefix_warnings, _write_quality_report
from traces.quality import validate_trace_quality


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded Kinetix trace generation helper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_batch = subparsers.add_parser("generate-batch")
    generate_batch.add_argument("--batch-json", required=True)
    generate_batch.set_defaults(func=generate_batch_command)

    args = parser.parse_args()
    args.func(args)


def generate_batch_command(args: argparse.Namespace) -> None:
    batch_path = Path(args.batch_json)
    payload = json.loads(batch_path.read_text(encoding="utf-8"))
    samples = list(payload.get("samples", []))

    _apply_resource_limits()
    _configure_jax_cache()
    registry = AdapterRegistry()
    adapter = registry.load("kinetix")
    results: list[dict[str, object]] = []

    for sample in samples:
        sample_dir = Path(str(sample["sample_dir"]))
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)

        params = dict(sample.get("params", {}))
        params["output_dir"] = sample_dir
        sample_id = int(sample["sample_id"])
        task_id = str(sample["task_id"])
        absolute_sample_index = int(sample["absolute_sample_index"])

        log_capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(log_capture):
                trace = adapter.generate_trace(task_id, **params)
            suffix = f":sample-{sample_id:03d}"
            if not trace.trace_id.endswith(suffix):
                trace = trace.model_copy(update={"trace_id": f"{trace.trace_id}{suffix}"})

            quality = validate_trace_quality(trace)
            _write_quality_report(sample_dir / "quality.json", quality)
            if not quality.usable:
                results.append(
                    {
                        "sample_id": sample_id,
                        "absolute_sample_index": absolute_sample_index,
                        "task_id": task_id,
                        "status": "failed",
                        "warnings": _prefix_warnings(task_id, sample_id, quality),
                        "error": "; ".join(quality.errors),
                    }
                )
                continue

            (sample_dir / "trace.json").write_text(
                json.dumps(trace.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )
            results.append(
                {
                    "sample_id": sample_id,
                    "absolute_sample_index": absolute_sample_index,
                    "task_id": task_id,
                    "status": "generated",
                    "warnings": _prefix_warnings(task_id, sample_id, quality),
                    "error": None,
                }
            )
        except Exception as error:
            captured = log_capture.getvalue().strip()
            details = f"{type(error).__name__}: {error}"
            if captured:
                details = f"{details} | log: {captured.splitlines()[-1]}"
            results.append(
                {
                    "sample_id": sample_id,
                    "absolute_sample_index": absolute_sample_index,
                    "task_id": task_id,
                    "status": "failed",
                    "warnings": [],
                    "error": details,
                }
            )

    print(json.dumps({"samples": results}))


def _configure_jax_cache() -> None:
    cache_dir = ROOT / ".runtime-home" / "jax-cache" / "kinetix"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        from jax import config as jax_config
        from jax.experimental.compilation_cache import compilation_cache as cc

        jax_config.update("jax_enable_compilation_cache", True)
        jax_config.update("jax_compilation_cache_dir", str(cache_dir))
        jax_config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax_config.update("jax_persistent_cache_enable_xla_caches", "all")
        cc.set_cache_dir(str(cache_dir))
    except Exception:
        return


def _apply_resource_limits() -> None:
    requested = os.environ.get("DEEPPHY_KINETIX_MAX_MEMORY_BYTES")
    if not requested:
        return
    try:
        limit_bytes = int(requested)
    except ValueError:
        return
    if limit_bytes <= 0:
        return
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except Exception:
        return


if __name__ == "__main__":
    main()
