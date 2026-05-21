from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExecutionMode(str, Enum):
    SIMULATOR = "simulator"
    GUI = "gui"


class RequirementKind(str, Enum):
    PYTHON = "python"
    MODULE = "module"
    PLATFORM = "platform"
    DIRECTORY = "directory"
    FILE = "file"
    APPLICATION = "application"
    MANUAL = "manual"


class CapabilityStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    EXTERNAL = "external"


class MetricKind(str, Enum):
    SUCCESS = "success"
    REWARD = "reward"
    DISTANCE = "distance"
    SCORE = "score"
    COMPOSITE = "composite"


class ProbeStatus(str, Enum):
    MET = "met"
    UNMET = "unmet"
    EXTERNAL = "external"
    MANUAL = "manual"


class AvailabilityStatus(str, Enum):
    READY = "ready"
    NEEDS_EXTERNAL = "needs_external"
    BLOCKED = "blocked"


class UpstreamPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: str
    runner: str
    config: str
    assets: list[str] = Field(default_factory=list)


class RuntimeRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: RequirementKind
    name: str
    value: str | None = None
    path: str | None = None
    interpreter_path: str | None = None
    required: bool = True
    notes: str | None = None


class NativeMetricSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: MetricKind
    success_field: str
    notes: str | None = None


class CapabilityFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: CapabilityStatus
    detail: str


class EnvironmentManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: str
    display_name: str
    execution_mode: ExecutionMode
    upstream: UpstreamPaths
    runtime: list[RuntimeRequirement]
    native_metric: NativeMetricSpec
    capabilities: list[CapabilityFlag]
    task_contract: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class RequirementProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: RequirementKind
    status: ProbeStatus
    detail: str
    required: bool = True


class EnvironmentProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: str
    display_name: str
    execution_mode: ExecutionMode
    native_metric: str
    availability: AvailabilityStatus
    requirements: list[RequirementProbe]
    capabilities: list[CapabilityFlag]
