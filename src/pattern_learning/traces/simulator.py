from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from adapters.base import EnvironmentAdapter
from schema import (
    CanonicalTrace,
    TaskOutcome,
    TraceAttachment,
    TraceEvent,
    TraceProvenance,
    TraceStep,
)


def coerce_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): coerce_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [coerce_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return coerce_jsonable(value.model_dump())
    if hasattr(value, "tolist"):
        return coerce_jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def summarize_shape(value: Any) -> dict[str, Any]:
    if value is None or not hasattr(value, "shape"):
        return {}
    return {"shape": list(getattr(value, "shape"))}


def make_attachment(
    kind: str,
    path: str | Path,
    *,
    label: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TraceAttachment:
    return TraceAttachment(
        kind=kind,
        path=str(path),
        label=label,
        metadata=coerce_jsonable(metadata or {}),
    )


def make_event(
    kind: str,
    *,
    t: float | None = None,
    payload: dict[str, Any] | None = None,
) -> TraceEvent:
    return TraceEvent(kind=kind, t=t, payload=coerce_jsonable(payload or {}))


def make_step(
    index: int,
    *,
    t: float | None = None,
    state: dict[str, Any] | None = None,
    objects: list[dict[str, Any]] | None = None,
    actions: list[dict[str, Any]] | None = None,
    attachments: list[TraceAttachment] | None = None,
) -> TraceStep:
    return TraceStep(
        index=index,
        t=t,
        state=coerce_jsonable(state or {}),
        objects=coerce_jsonable(objects or []),
        actions=coerce_jsonable(actions or []),
        attachments=attachments or [],
    )


def build_provenance(
    adapter: EnvironmentAdapter,
    *,
    environment_version: str | None = None,
    seed: int | None = None,
    notes: list[str] | None = None,
) -> TraceProvenance:
    manifest = adapter.get_manifest()
    return TraceProvenance(
        source="deepphy-rework",
        environment_version=environment_version,
        config_ref=manifest.upstream.config,
        seed=seed,
        runner_ref=manifest.upstream.runner,
        notes=notes or [],
    )


def build_simulator_trace(
    *,
    env_id: str,
    task_id: str,
    task_name: str,
    initial_state: dict[str, Any],
    actions: list[dict[str, Any]],
    timesteps: list[TraceStep],
    events: list[TraceEvent],
    attachments: list[TraceAttachment],
    metrics: dict[str, Any],
    success: bool,
    terminal_status: str,
    provenance: TraceProvenance,
) -> CanonicalTrace:
    return CanonicalTrace(
        trace_id=f"{env_id}:{task_id}:{terminal_status.lower()}",
        environment=env_id,
        task_id=task_id,
        initial_state=coerce_jsonable(initial_state),
        actions=coerce_jsonable(actions),
        timesteps=timesteps,
        events=events,
        attachments=attachments,
        outcome=TaskOutcome(
            task_name=task_name,
            metrics=coerce_jsonable(metrics),
            success=success,
            terminal_status=terminal_status,
        ),
        provenance=provenance,
    )
