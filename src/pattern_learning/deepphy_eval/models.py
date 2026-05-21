from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


EvaluationEnvId = Literal["phyre", "iphyre", "kinetix", "pooltool"]
EvaluationSourceKind = Literal["stored", "live"]
EvaluationRunSourceKind = Literal["dataset_manifest", "trace_root", "live_request"]
EvidenceStatus = Literal["missing", "present", "invalid"]
ParityMode = Literal["reference", "integration"]
ParityStatus = Literal["passed", "failed", "skipped", "blocked"]

DEFAULT_EVIDENCE_RELPATH = "trace.evidence.json"


class EvaluationLiveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: EvaluationEnvId
    task_id: str
    adapter_kwargs: dict[str, Any] = Field(default_factory=dict)
    trace_output_dir: str | None = None
    evidence_relpath: str | None = None


class EvaluationTraceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: EvaluationEnvId
    task_id: str
    trace_id: str
    source_kind: EvaluationSourceKind
    task_name: str
    native_metrics: dict[str, Any] = Field(default_factory=dict)
    success: bool | None = None
    terminal_status: str | None = None
    trace_path: str | None = None
    provenance_config_ref: str | None = None
    provenance_runner_ref: str | None = None
    evidence_relpath: str = DEFAULT_EVIDENCE_RELPATH
    evidence_status: EvidenceStatus = "missing"
    warnings: list[str] = Field(default_factory=list)


class EnvironmentEvaluationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: EvaluationEnvId
    trace_count: int
    success_count: int
    failure_count: int
    unknown_count: int
    metric_summaries: dict[str, dict[str, Any]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ParityReferenceFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: EvaluationEnvId
    task_id: str
    expected_outcome: dict[str, Any] = Field(default_factory=dict)
    expected_metrics: dict[str, Any] = Field(default_factory=dict)


class ParityExpectations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expectations_version: str
    fixtures: list[ParityReferenceFixture]


class ParityCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: EvaluationEnvId
    task_id: str
    trace_id: str | None = None
    mode: ParityMode
    status: ParityStatus
    drifts: list[str] = Field(default_factory=list)
    message: str | None = None


class ParityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: ParityMode
    checks: list[ParityCheck] = Field(default_factory=list)
    expectations_version: str | None = None


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    source_kind: EvaluationRunSourceKind
    dataset_id: str
    dataset_version: str
    config_identity: str
    results: list[EvaluationTraceResult] = Field(default_factory=list)
    env_summaries: list[EnvironmentEvaluationSummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    run_dir: str | None = None
    report_dir: str | None = None
    results_path: str | None = None
    report_path: str | None = None
    evidence_index_path: str | None = None
    parity_report_path: str | None = None
    parity: ParityReport | None = None
