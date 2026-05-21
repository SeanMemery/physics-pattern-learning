from __future__ import annotations

from pathlib import Path

from adapters.registry import AdapterRegistry
from datasets.index import index_path, load_dataset_index
from datasets.manifest import load_dataset_manifest, resolve_dataset_root
from deepphy_eval.io import inspect_evidence_sidecar, load_trace, write_trace
from deepphy_eval.models import (
    EvaluationEnvId,
    EvaluationLiveRequest,
    EvaluationTraceResult,
)
from schema import CanonicalTrace


SUPPORTED_EVALUATION_ENVS: tuple[EvaluationEnvId, ...] = (
    "phyre",
    "iphyre",
    "kinetix",
    "pooltool",
)


def evaluate_trace(
    trace: CanonicalTrace,
    *,
    registry: AdapterRegistry | None = None,
    trace_path: Path | None = None,
    source_kind: str = "stored",
    evidence_relpath: str | None = None,
) -> EvaluationTraceResult:
    env_id = trace.environment
    _require_supported_env(env_id)
    adapter_registry = registry or AdapterRegistry()
    adapter = adapter_registry.load(env_id)
    native_metrics = adapter.evaluate_trace(trace)
    sample_dir = trace_path.parent if trace_path is not None else None
    resolved_evidence_relpath = evidence_relpath
    evidence_status = "missing"
    if sample_dir is not None:
        resolved_evidence_relpath, evidence_status = inspect_evidence_sidecar(
            sample_dir,
            evidence_relpath=evidence_relpath,
        )

    warnings: list[str] = []
    if trace.provenance.config_ref is None:
        warnings.append("missing provenance.config_ref")
    if trace.provenance.runner_ref is None:
        warnings.append("missing provenance.runner_ref")

    return EvaluationTraceResult(
        env_id=env_id,
        task_id=trace.task_id,
        trace_id=trace.trace_id,
        source_kind=source_kind,
        task_name=trace.outcome.task_name,
        native_metrics=native_metrics,
        success=trace.outcome.success,
        terminal_status=trace.outcome.terminal_status,
        trace_path=str(trace_path) if trace_path is not None else None,
        provenance_config_ref=trace.provenance.config_ref,
        provenance_runner_ref=trace.provenance.runner_ref,
        evidence_relpath=resolved_evidence_relpath or "trace.evidence.json",
        evidence_status=evidence_status,
        warnings=warnings,
    )


def evaluate_stored_trace(
    trace_path: Path,
    *,
    registry: AdapterRegistry | None = None,
    evidence_relpath: str | None = None,
) -> EvaluationTraceResult:
    trace = load_trace(trace_path)
    return evaluate_trace(
        trace,
        registry=registry,
        trace_path=trace_path,
        source_kind="stored",
        evidence_relpath=evidence_relpath,
    )


def evaluate_live_request(
    request: EvaluationLiveRequest,
    *,
    registry: AdapterRegistry | None = None,
) -> EvaluationTraceResult:
    _require_supported_env(request.env_id)
    adapter_registry = registry or AdapterRegistry()
    adapter = adapter_registry.load(request.env_id)

    adapter_kwargs = dict(request.adapter_kwargs)
    trace_output_dir = Path(request.trace_output_dir) if request.trace_output_dir else None
    if trace_output_dir is not None:
        trace_output_dir.mkdir(parents=True, exist_ok=True)
        adapter_kwargs["output_dir"] = trace_output_dir

    trace = adapter.generate_trace(request.task_id, **adapter_kwargs)
    trace_path = None
    if trace_output_dir is not None:
        trace_path = trace_output_dir / "trace.json"
        write_trace(trace_path, trace)

    return evaluate_trace(
        trace,
        registry=adapter_registry,
        trace_path=trace_path,
        source_kind="live",
        evidence_relpath=request.evidence_relpath,
    )


def evaluate_trace_root(
    trace_root: Path,
    *,
    env_ids: list[str] | None = None,
    limit: int | None = None,
    registry: AdapterRegistry | None = None,
) -> list[EvaluationTraceResult]:
    chosen_envs = set(_normalize_env_ids(env_ids))
    results: list[EvaluationTraceResult] = []
    for trace_path in sorted(trace_root.rglob("trace.json"), key=lambda path: path.as_posix()):
        trace = load_trace(trace_path)
        if chosen_envs and trace.environment not in chosen_envs:
            continue
        results.append(
            evaluate_trace(
                trace,
                registry=registry,
                trace_path=trace_path,
                source_kind="stored",
            )
        )
        if limit is not None and len(results) >= limit:
            break
    return results


def evaluate_dataset_manifest(
    manifest_path: Path,
    *,
    env_ids: list[str] | None = None,
    split_names: list[str] | None = None,
    limit: int | None = None,
    registry: AdapterRegistry | None = None,
) -> list[EvaluationTraceResult]:
    chosen_envs = set(_normalize_env_ids(env_ids))
    chosen_splits = set(split_names or [])
    manifest = load_dataset_manifest(manifest_path)
    dataset_root = resolve_dataset_root(manifest_path, manifest)
    dataset_index = load_dataset_index(index_path(dataset_root))

    entries = sorted(
        dataset_index.entries,
        key=lambda entry: (entry.split_name, entry.env_id, entry.sample_id),
    )

    results: list[EvaluationTraceResult] = []
    for entry in entries:
        if entry.status != "completed":
            continue
        if chosen_envs and entry.env_id not in chosen_envs:
            continue
        if chosen_splits and entry.split_name not in chosen_splits:
            continue
        trace_path = dataset_root / entry.trace_relpath
        if not trace_path.is_file():
            continue
        results.append(evaluate_stored_trace(trace_path, registry=registry))
        if limit is not None and len(results) >= limit:
            break
    return results


def _normalize_env_ids(env_ids: list[str] | None) -> list[EvaluationEnvId]:
    if not env_ids:
        return list(SUPPORTED_EVALUATION_ENVS)
    normalized: list[EvaluationEnvId] = []
    for env_id in env_ids:
        _require_supported_env(env_id)
        normalized.append(env_id)
    return normalized


def _require_supported_env(env_id: str) -> None:
    if env_id not in SUPPORTED_EVALUATION_ENVS:
        supported = ", ".join(SUPPORTED_EVALUATION_ENVS)
        raise ValueError(f"unsupported evaluation environment: {env_id}. expected one of: {supported}")
