from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    created = "created"
    queued = "queued"
    waiting_for_dependencies = "waiting_for_dependencies"
    assigned = "assigned"
    starting = "starting"
    running = "running"
    pausing = "pausing"
    paused = "paused"
    resuming = "resuming"
    stopping = "stopping"
    stopped = "stopped"
    succeeded = "succeeded"
    failed = "failed"
    lost = "lost"
    retryable_failure = "retryable_failure"
    cancelled = "cancelled"


class ClientStatus(str, Enum):
    online = "online"
    offline = "offline"
    draining = "draining"


class ClientKind(str, Enum):
    remote = "remote"
    direct = "direct"


class JobCommandType(str, Enum):
    pause = "pause"
    cancel = "cancel"
    requeue = "requeue"


class ClientCommandType(str, Enum):
    shutdown = "shutdown"


class QueueSettings(BaseModel):
    # Queue-wide concurrency controls are intentionally minimal. Dispatch is
    # governed by each client's own max_concurrency, with a small set of
    # server-side overrides exposed here for operational tuning.
    pooltool_attempt_limit: int = Field(default=8, ge=1, le=32)


class ClientCapabilities(BaseModel):
    environments: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    gpu: bool = False
    max_concurrency: int = 4
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClientRecord(BaseModel):
    client_id: str
    display_name: str
    kind: ClientKind = ClientKind.remote
    direct_url: str | None = None
    status: ClientStatus = ClientStatus.online
    version: str = ""
    hostname: str = ""
    platform: str = ""
    capabilities: ClientCapabilities = Field(default_factory=ClientCapabilities)
    max_concurrency: int = Field(default=4, ge=1)
    last_heartbeat: datetime
    last_poll: datetime | None = None
    current_jobs: list[str] = Field(default_factory=list)


class JobConstraints(BaseModel):
    required_tags: list[str] = Field(default_factory=list)
    required_environments: list[str] = Field(default_factory=list)
    gpu_required: bool = False


class JobSpec(BaseModel):
    action: str
    argv: list[str]
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    constraints: JobConstraints = Field(default_factory=JobConstraints)


class JobCommand(BaseModel):
    command_id: str
    job_id: str
    command: JobCommandType
    created_at: datetime


class ClientCommand(BaseModel):
    command_id: str
    command: ClientCommandType
    created_at: datetime


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    assigned_client_id: str | None = None
    exit_code: int | None = None
    message: str | None = None
    spec: JobSpec
    recent_logs: list[str] = Field(default_factory=list)
    artifact_relpaths: list[str] = Field(default_factory=list)
    pending_commands: list[JobCommand] = Field(default_factory=list)


class RegisterClientRequest(BaseModel):
    display_name: str
    kind: ClientKind = ClientKind.remote
    direct_url: str | None = None
    version: str = ""
    hostname: str = ""
    platform: str = ""
    capabilities: ClientCapabilities = Field(default_factory=ClientCapabilities)


class RegisterClientResponse(BaseModel):
    client: ClientRecord


class HeartbeatRequest(BaseModel):
    current_jobs: list[str] = Field(default_factory=list)


class PollResponse(BaseModel):
    assignments: list[JobRecord] = Field(default_factory=list)
    commands: list[JobCommand] = Field(default_factory=list)
    client_commands: list[ClientCommand] = Field(default_factory=list)
    queue_settings: QueueSettings = Field(default_factory=QueueSettings)
    client_max_concurrency: int = Field(default=1, ge=1)
    suggested_poll_delay_seconds: float | None = Field(default=None, ge=0.0)


class CreateJobRequest(BaseModel):
    spec: JobSpec
    enqueue: bool = True


class CreateJobResponse(BaseModel):
    job: JobRecord


class JobUpdateRequest(BaseModel):
    client_id: str | None = None
    status: JobStatus
    exit_code: int | None = None
    message: str | None = None
    metadata_patch: dict[str, Any] = Field(default_factory=dict)


class JobLogChunkRequest(BaseModel):
    client_id: str | None = None
    lines: list[str] = Field(default_factory=list)


class JobArtifactUploadResponse(BaseModel):
    job: JobRecord
    extracted_paths: list[str] = Field(default_factory=list)


class QueueSettingsUpdateRequest(BaseModel):
    pooltool_attempt_limit: int | None = Field(default=None, ge=1, le=32)


class ClientSettingsUpdateRequest(BaseModel):
    max_concurrency: int = Field(ge=1)


class JobWorkersUpdateRequest(BaseModel):
    workers: int = Field(ge=1)


class PreferredClientUpdateRequest(BaseModel):
    preferred_client_id: str | None = None


class PublicAccessState(BaseModel):
    desired_enabled: bool = False
    active: bool = False
    api_key: str = ""
    https_port: int = Field(default=8443, ge=1)
    path_prefix: str = "/public"
    local_target_url: str = ""
    public_url: str | None = None
    dns_name: str | None = None
    last_error: str | None = None
    operator_required: bool = False
    last_synced_at: datetime | None = None
    updated_at: datetime | None = None


class PublicAccessUpdateRequest(BaseModel):
    enabled: bool
    api_key: str | None = None


class CombineSupervisedRunsRequest(BaseModel):
    env_id: str = Field(min_length=1)
    job_ids: list[str] = Field(min_length=2)


class HideRepeatLearningRunsRequest(BaseModel):
    job_ids: list[str] = Field(min_length=1)
