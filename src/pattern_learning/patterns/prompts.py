from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

from paths import get_artifact_layout
from pattern_library_schema import PatternEvidence
from pattern_library_serializer import serialize_pattern_evidence
from schema import CanonicalTrace


@dataclass(frozen=True)
class PatternPrompt:
    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FunSearchCandidateSummary:
    uid: str | None = None
    origin: str | None = None
    score: float | None = None
    train_score: float | None = None
    val_score: float | None = None
    average_score: float | None = None
    loss_breakdown: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def build_trace_prediction_prompt(
    trace: CanonicalTrace,
    *,
    task_context: dict[str, Any] | None = None,
) -> PatternPrompt:
    task_context = task_context or {}
    return _render_pattern_prompt(
        environment_id=trace.environment,
        template_name="trace_prediction",
        template_vars={
            "environment_id": trace.environment,
            "task_id": trace.task_id,
            "task_context": task_context,
            "task_context_json": json.dumps(task_context, sort_keys=True),
            "trace_summary": _trace_summary(trace),
            "trace_metrics": _trace_metrics(trace),
            "event_kinds": [event.kind for event in trace.events],
            "outcome_terminal_status": trace.outcome.terminal_status,
            "outcome_success": trace.outcome.success,
        },
        metadata={"mode": "trace_text", "task_context": task_context},
    )


def build_prediction_request_prompt(
    trace: CanonicalTrace,
    *,
    cutoff_index: int,
    task_context: dict[str, Any] | None = None,
) -> PatternPrompt:
    task_context = task_context or {}
    if cutoff_index < 0 or cutoff_index >= len(trace.timesteps):
        raise ValueError(f"cutoff_index {cutoff_index} is out of range for trace {trace.trace_id}")
    observed_trace = _truncate_trace_for_prompt(trace, cutoff_index)
    observed_count = len(observed_trace.timesteps)
    total_count = len(trace.timesteps)
    cutoff_step = observed_trace.timesteps[-1]
    cutoff_fraction = cutoff_index / max(total_count - 1, 1)
    return _render_pattern_prompt(
        environment_id=trace.environment,
        template_name="prediction_request",
        template_vars={
            "environment_id": trace.environment,
            "task_id": trace.task_id,
            "task_context": task_context,
            "task_context_json": json.dumps(task_context, sort_keys=True),
            "trace_summary": _trace_summary(observed_trace),
            "trace_metrics": _trace_metrics(observed_trace),
            "event_kinds": [event.kind for event in observed_trace.events],
            "outcome_terminal_status": trace.outcome.terminal_status,
            "outcome_success": trace.outcome.success,
            "cutoff_index": cutoff_index,
            "cutoff_t": cutoff_step.t,
            "cutoff_fraction": cutoff_fraction,
            "observed_timestep_count": observed_count,
            "total_timestep_count": total_count,
            "future_timestep_count": total_count - observed_count,
        },
        metadata={
            "mode": "prediction_request",
            "task_context": task_context,
            "cutoff_index": cutoff_index,
            "cutoff_fraction": cutoff_fraction,
        },
    )


def build_pattern_prediction_prompt(
    trace: CanonicalTrace,
    evidence: PatternEvidence,
    *,
    task_context: dict[str, Any] | None = None,
) -> PatternPrompt:
    task_context = task_context or {}
    bundle = serialize_pattern_evidence(evidence)
    return _render_pattern_prompt(
        environment_id=trace.environment,
        template_name="pattern_prediction",
        template_vars={
            "environment_id": trace.environment,
            "task_id": trace.task_id,
            "task_context": task_context,
            "task_context_json": json.dumps(task_context, sort_keys=True),
            "trace_summary": _trace_summary(trace),
            "trace_metrics": _trace_metrics(trace),
            "event_kinds": [event.kind for event in trace.events],
            "outcome_terminal_status": trace.outcome.terminal_status,
            "outcome_success": trace.outcome.success,
            "library_id": evidence.library_id,
            "pattern_evidence_text": bundle.text,
        },
        metadata={
            "mode": "pattern_text",
            "library_id": evidence.library_id,
            "task_context": task_context,
        },
    )


def build_funsearch_prompt(
    *,
    environment_id: str,
    label: str,
    label_description: str | None = None,
    action: str,
    settings: Any,
    config: Any,
    input_mode: str = "trace",
    parent_code: str | None = None,
    parent_metadata: FunSearchCandidateSummary | dict[str, Any] | None = None,
    parent_a_code: str | None = None,
    parent_b_code: str | None = None,
    parent_a_metadata: FunSearchCandidateSummary | dict[str, Any] | None = None,
    parent_b_metadata: FunSearchCandidateSummary | dict[str, Any] | None = None,
    broken_code: str | None = None,
    error_message: str | None = None,
    error_category: str | None = None,
    error_line: int | None = None,
    error_column: int | None = None,
    step: int | None = None,
) -> str:
    template_name = {
        "mutate": "funsearch_mutate",
        "crossover": "funsearch_crossover",
        "repair": "funsearch_repair",
    }.get(action)
    if template_name is None:
        raise ValueError(f"Unsupported FunSearch prompt action: {action}")

    prompt = _render_single_template(
        environment_id=environment_id,
        template_name=template_name,
        template_vars={
            "environment_id": environment_id,
            "label": label,
            "settings": settings,
            "config": config,
            "step": step,
            "parent_code": parent_code,
            "parent_metadata": _candidate_summary_payload(parent_metadata),
            "parent_a_code": parent_a_code,
            "parent_b_code": parent_b_code,
            "parent_a_metadata": _candidate_summary_payload(parent_a_metadata),
            "parent_b_metadata": _candidate_summary_payload(parent_b_metadata),
            "broken_code": broken_code,
            "error_message": error_message,
            "error_category": error_category,
            "error_line": error_line,
            "error_column": error_column,
        },
    )
    if input_mode == "supervised_evidence":
        semantic_target = (
            "- Semantic target description: "
            f"{label_description}\n"
            "- This sentence explains what the pattern is, not how to detect it.\n"
            "- Learn from the examples what works; do not convert the description into a brittle rule list.\n"
            if label_description
            else ""
        )
        contract = (
            "IMPORTANT INPUT CONTRACT:\n"
            "- This detector is automatic and must operate on supervised pattern detections only.\n"
            "- The function input is the list of supervised pattern detections.\n"
            "- The caller will pass `existing_evidence['pattern_annotations']['patterns']` as `supervised_patterns`.\n"
            "- Do not rely on raw simulator state, object positions, or full trace internals.\n"
            f"- The target automatic label is `{label}`.\n"
            f"{semantic_target}"
            "- Required signature:\n"
            "  def find_pattern(supervised_patterns):\n"
            "      ...\n"
        )
        adapted_prompt = prompt
        adapted_prompt = adapted_prompt.replace(
            "def find_pattern(trace):",
            "def find_pattern(supervised_patterns):",
        )
        adapted_prompt = adapted_prompt.replace(
            "- inspect only the provided `trace` dict\n",
            "- inspect only the provided `supervised_patterns` list\n",
        )
        adapted_prompt = re.sub(
            r"Compact trace example:\n```python\n.*?```",
            (
                "Compact supervised pattern example:\n"
                "```python\n"
                "[\n"
                "  {\"label\": \"object collision\", \"start_timestep\": 2, \"end_timestep\": 5, \"parameters\": {}},\n"
                "  {\"label\": \"support contact\", \"start_timestep\": 4, \"end_timestep\": 8}\n"
                "]\n"
                "```"
            ),
            adapted_prompt,
            flags=re.DOTALL,
        )
        return f"{contract}\n\n{adapted_prompt}"
    return prompt


def pattern_prompt_template_root() -> Path:
    return get_artifact_layout().config_root


def _render_pattern_prompt(
    *,
    environment_id: str,
    template_name: str,
    template_vars: dict[str, Any],
    metadata: dict[str, Any],
) -> PatternPrompt:
    engine, env_root = _prompt_engine(environment_id)
    try:
        system_prompt = engine.get_template(f"{template_name}.system.j2").render(**template_vars).strip()
        user_prompt = engine.get_template(f"{template_name}.user.j2").render(**template_vars).strip()
    except TemplateNotFound as exc:
        raise FileNotFoundError(
            f"Pattern prompt template missing for environment '{environment_id}': {env_root / exc.name}"
        ) from exc

    if not system_prompt:
        raise ValueError(f"Rendered system prompt is empty for {environment_id}/{template_name}")
    if not user_prompt:
        raise ValueError(f"Rendered user prompt is empty for {environment_id}/{template_name}")
    return PatternPrompt(system_prompt=system_prompt, user_prompt=user_prompt, metadata=metadata)


def _render_single_template(
    *,
    environment_id: str,
    template_name: str,
    template_vars: dict[str, Any],
) -> str:
    engine, env_root = _prompt_engine(environment_id)
    try:
        rendered = engine.get_template(f"{template_name}.j2").render(**template_vars).strip()
    except TemplateNotFound as exc:
        raise FileNotFoundError(
            f"Pattern prompt template missing for environment '{environment_id}': {env_root / exc.name}"
        ) from exc
    if not rendered:
        raise ValueError(f"Rendered prompt is empty for {environment_id}/{template_name}")
    return rendered


def _prompt_engine(environment_id: str) -> tuple[Environment, Path]:
    root = pattern_prompt_template_root()
    env_root = root / environment_id / "prompts"
    if not env_root.is_dir():
        raise FileNotFoundError(
            f"Pattern prompt templates not found for environment '{environment_id}': {env_root}"
        )
    loader = FileSystemLoader(str(env_root))
    engine = Environment(
        loader=loader,
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return engine, env_root


def _trace_summary(trace: CanonicalTrace) -> str:
    metrics = ", ".join(f"{key}={value}" for key, value in _trace_metrics(trace).items()) or "none"
    events = ", ".join(event.kind for event in trace.events[:5]) or "no events"
    return (
        f"timesteps={len(trace.timesteps)}; "
        f"events={len(trace.events)} ({events}); "
        f"terminal_status={trace.outcome.terminal_status}; "
        f"success={trace.outcome.success}; "
        f"metrics={metrics}"
    )


def _trace_metrics(trace: CanonicalTrace) -> dict[str, Any]:
    return {
        key: value
        for key, value in sorted(trace.outcome.metrics.items())
        if isinstance(value, (str, int, float, bool))
    }


def _candidate_summary_payload(summary: FunSearchCandidateSummary | dict[str, Any] | None) -> dict[str, Any]:
    payload = {
        "uid": None,
        "origin": None,
        "score": None,
        "train_score": None,
        "val_score": None,
        "average_score": None,
        "loss_breakdown": [],
        "errors": [],
    }
    if summary is None:
        return payload
    if isinstance(summary, FunSearchCandidateSummary):
        payload.update(summary.__dict__)
        return payload
    payload.update(dict(summary))
    return payload


def _truncate_trace_for_prompt(trace: CanonicalTrace, cutoff_index: int) -> CanonicalTrace:
    cutoff_step = trace.timesteps[cutoff_index]
    cutoff_t = cutoff_step.t
    events = trace.events
    if cutoff_t is not None:
        events = [event for event in trace.events if event.t is None or event.t <= cutoff_t]
    return CanonicalTrace.model_validate(
        {
            **trace.model_dump(mode="json"),
            "timesteps": [step.model_dump(mode="json") for step in trace.timesteps[: cutoff_index + 1]],
            "events": [event.model_dump(mode="json") for event in events],
        }
    )
