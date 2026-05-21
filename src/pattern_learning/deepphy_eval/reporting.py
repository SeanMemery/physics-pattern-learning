from __future__ import annotations

from pathlib import Path
from typing import Any

from deepphy_eval.io import (
    normalized_json_sha256_prefix,
    read_json_bytes,
    read_json_payload,
)
from deepphy_eval.models import (
    EnvironmentEvaluationSummary,
    EvaluationReport,
    EvaluationTraceResult,
)


def build_evaluation_report(
    *,
    run_id: str,
    source_kind: str,
    dataset_id: str,
    dataset_version: str,
    config_identity: str,
    results: list[EvaluationTraceResult],
) -> EvaluationReport:
    env_summaries = _build_env_summaries(results)
    warnings = _report_warnings(results)
    return EvaluationReport(
        run_id=run_id,
        source_kind=source_kind,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        config_identity=config_identity,
        results=results,
        env_summaries=env_summaries,
        warnings=warnings,
    )


def resolve_dataset_identity_from_manifest(manifest_path: Path) -> tuple[str, str, str]:
    payload = read_json_payload(manifest_path)
    manifest_bytes = read_json_bytes(manifest_path)
    dataset_id = str(payload.get("dataset_id") or manifest_path.parent.name)
    dataset_version = str(payload.get("dataset_version") or _sha_from_bytes(manifest_bytes))
    config_identity = str(
        payload.get("config_identity")
        or payload.get("config_ref")
        or payload.get("config_source")
        or normalized_json_sha256_prefix(
            {
                "env_ids": payload.get("env_ids", []),
                "splits": payload.get("splits", []),
            }
        )
    )
    return dataset_id, dataset_version, config_identity


def resolve_trace_root_identity(trace_root: Path) -> tuple[str, str, str]:
    resolved = trace_root.resolve()
    digest = normalized_json_sha256_prefix({"trace_root": str(resolved)})
    return "adhoc-trace-root", digest, digest


def resolve_live_request_identity(live_request_path: Path) -> tuple[str, str, str]:
    payload = read_json_payload(live_request_path)
    digest = normalized_json_sha256_prefix(payload)
    return "adhoc-live-request", digest, digest


def render_evaluation_payload(report: EvaluationReport) -> dict[str, Any]:
    return report.model_dump(mode="json")


def _build_env_summaries(results: list[EvaluationTraceResult]) -> list[EnvironmentEvaluationSummary]:
    grouped: dict[str, list[EvaluationTraceResult]] = {}
    for result in results:
        grouped.setdefault(result.env_id, []).append(result)

    summaries: list[EnvironmentEvaluationSummary] = []
    for env_id in sorted(grouped):
        env_results = grouped[env_id]
        success_count = sum(1 for result in env_results if result.success is True)
        failure_count = sum(1 for result in env_results if result.success is False)
        unknown_count = sum(1 for result in env_results if result.success is None)
        summaries.append(
            EnvironmentEvaluationSummary(
                env_id=env_id,
                trace_count=len(env_results),
                success_count=success_count,
                failure_count=failure_count,
                unknown_count=unknown_count,
                metric_summaries=_metric_summaries(env_results),
                warnings=sorted({warning for result in env_results for warning in result.warnings}),
            )
        )
    return summaries


def _metric_summaries(results: list[EvaluationTraceResult]) -> dict[str, dict[str, Any]]:
    numeric_values: dict[str, list[float]] = {}
    boolean_counts: dict[str, dict[str, int]] = {}
    string_counts: dict[str, dict[str, int]] = {}

    for result in results:
        for key, value in result.native_metrics.items():
            if isinstance(value, bool):
                bucket = boolean_counts.setdefault(key, {"true_count": 0, "false_count": 0})
                bucket["true_count" if value else "false_count"] += 1
            elif isinstance(value, (int, float)):
                numeric_values.setdefault(key, []).append(float(value))
            elif isinstance(value, str):
                counts = string_counts.setdefault(key, {})
                counts[value] = counts.get(value, 0) + 1

    summaries: dict[str, dict[str, Any]] = {}
    for key, values in numeric_values.items():
        summaries[key] = {
            "kind": "numeric",
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": round(sum(values) / len(values), 6),
        }
    for key, counts in boolean_counts.items():
        summaries[key] = {"kind": "boolean", **counts}
    for key, counts in string_counts.items():
        summaries[key] = {"kind": "categorical", "value_counts": counts}
    return summaries


def _report_warnings(results: list[EvaluationTraceResult]) -> list[str]:
    warnings: list[str] = []
    for result in results:
        if result.provenance_config_ref is None:
            warnings.append(f"{result.trace_id}: missing provenance.config_ref")
        if result.provenance_runner_ref is None:
            warnings.append(f"{result.trace_id}: missing provenance.runner_ref")
    return sorted(set(warnings))


def _sha_from_bytes(payload: bytes) -> str:
    from hashlib import sha256

    return sha256(payload).hexdigest()[:12]
