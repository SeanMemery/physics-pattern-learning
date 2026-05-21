from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class NamedMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionSpaceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    format: str
    shape: list[int] = Field(default_factory=list)
    description: Optional[str] = None


class ObservationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    format: str
    description: Optional[str] = None


class NativeMetricSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str
    success_field: str
    notes: Optional[str] = None


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: str
    display_name: str
    execution_mode: str
    action_space: ActionSpaceSpec
    observation: ObservationSpec
    native_metric: NativeMetricSpec
    trace_step_unit: Optional[str] = None
    attachment_kinds: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class TraceAttachment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    path: str
    label: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: Optional[float] = None
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)


class TraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    t: Optional[float] = None
    state: dict[str, Any] = Field(default_factory=dict)
    objects: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    attachments: list[TraceAttachment] = Field(default_factory=list)


class TraceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    environment_version: Optional[str] = None
    config_ref: Optional[str] = None
    seed: Optional[int] = None
    runner_ref: Optional[str] = None
    notes: list[str] = Field(default_factory=list)


class TaskOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    success: Optional[bool] = None
    terminal_status: Optional[str] = None


class CanonicalTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    environment: str
    task_id: str
    initial_state: dict[str, Any] = Field(default_factory=dict)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    timesteps: list[TraceStep] = Field(default_factory=list)
    events: list[TraceEvent] = Field(default_factory=list)
    attachments: list[TraceAttachment] = Field(default_factory=list)
    outcome: TaskOutcome
    provenance: TraceProvenance


__all__ = [
    "ActionSpaceSpec",
    "CanonicalTrace",
    "NamedMetadata",
    "NativeMetricSpec",
    "ObservationSpec",
    "TaskOutcome",
    "TaskSpec",
    "TraceAttachment",
    "TraceEvent",
    "TraceProvenance",
    "TraceStep",
]
