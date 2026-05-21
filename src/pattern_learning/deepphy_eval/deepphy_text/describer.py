from __future__ import annotations

from typing import Protocol

from pattern_library_schema import PatternEvidence
from pattern_library_serializer import serialize_pattern_evidence
from schema import CanonicalTrace


class TraceDescriber(Protocol):
    def describe(
        self,
        trace: CanonicalTrace,
        *,
        env_id: str,
        task_id: str,
        index: int,
        record_kind: str,
        description_mode: str,
        evidence: PatternEvidence | None = None,
    ) -> str: ...


def dummy_trace_description(
    trace: CanonicalTrace,
    *,
    env_id: str,
    task_id: str,
    index: int,
    record_kind: str,
) -> str:
    metrics = ", ".join(
        f"{key}={value}"
        for key, value in sorted(trace.outcome.metrics.items())
        if isinstance(value, (str, int, float, bool))
    )
    if not metrics:
        metrics = "no scalar metrics recorded"
    return (
        f"Dummy trace description for {env_id}:{task_id} {record_kind} {index}. "
        f"Timesteps={len(trace.timesteps)}. Events={len(trace.events)}. "
        f"Terminal status={trace.outcome.terminal_status}. Success={trace.outcome.success}. "
        f"Metrics: {metrics}."
    )


def describe_trace_with_mode(
    trace: CanonicalTrace,
    *,
    env_id: str,
    task_id: str,
    index: int,
    record_kind: str,
    description_mode: str,
    evidence: PatternEvidence | None = None,
) -> str:
    trace_text = dummy_trace_description(
        trace,
        env_id=env_id,
        task_id=task_id,
        index=index,
        record_kind=record_kind,
    )
    if description_mode == "trace_text":
        return trace_text
    if evidence is None:
        raise ValueError(f"description_mode={description_mode} requires pattern evidence")
    pattern_text = serialize_pattern_evidence(evidence).text
    if description_mode == "pattern_text":
        return pattern_text
    if description_mode == "trace_text_plus_patterns":
        return f"{trace_text}\n\nLearned pattern annotations:\n{pattern_text}"
    raise ValueError(f"Unsupported description mode: {description_mode}")
