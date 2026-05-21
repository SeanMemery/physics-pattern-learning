from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DeepPHYTextEnvId = Literal["phyre", "iphyre", "kinetix", "pooltool"]
DeepPHYTextPromptFormat = Literal["VLA", "WM"]
DeepPHYTextRecordKind = Literal["attempt", "step"]
DeepPHYTextDescriptionMode = Literal["trace_text", "pattern_text", "trace_text_plus_patterns"]


def normalize_deepphy_prompt_format(value: str | None = None) -> DeepPHYTextPromptFormat:
    del value
    return "WM"


class DeepPHYTextEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: DeepPHYTextEnvId
    task_id: str
    model_name: str = "mock"
    prompt_format: DeepPHYTextPromptFormat = "WM"
    description_mode: DeepPHYTextDescriptionMode = "trace_text"
    pattern_run_dir: str | None = None
    pattern_library_path: str | None = None
    evidence_relpath: str | None = None
    run_id: str | None = None
    output_root: str | None = None
    report_root: str | None = None
    capture_visuals: bool = False
    benchmark_kwargs: dict[str, Any] = Field(default_factory=dict)
    adapter_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize_prompt_format(self) -> "DeepPHYTextEvaluationRequest":
        self.prompt_format = normalize_deepphy_prompt_format(self.prompt_format)
        return self


class DeepPHYTextPromptImage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    label: str | None = None


class DeepPHYTextPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: DeepPHYTextEnvId
    task_id: str
    record_kind: DeepPHYTextRecordKind
    index: int
    prompt_format: DeepPHYTextPromptFormat
    description_mode: DeepPHYTextDescriptionMode
    system_prompt: str
    user_prompt: str
    initial_image_path: str
    image_inputs: list[DeepPHYTextPromptImage] = Field(default_factory=list)
    trace_description: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize_prompt_format(self) -> "DeepPHYTextPrompt":
        self.prompt_format = normalize_deepphy_prompt_format(self.prompt_format)
        return self


class DeepPHYTextRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    record_kind: DeepPHYTextRecordKind
    prompt: DeepPHYTextPrompt
    raw_model_output: str
    parsed_action: dict[str, Any] = Field(default_factory=dict)
    trace_description: str
    evidence_status: str | None = None
    trace_path: str | None = None
    trace_id: str | None = None
    success: bool | None = None
    terminal_status: str | None = None
    native_metrics: dict[str, Any] = Field(default_factory=dict)
    router: dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float | None = None


class DeepPHYTextEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    env_id: DeepPHYTextEnvId
    task_id: str
    model_name: str
    prompt_format: DeepPHYTextPromptFormat
    description_mode: DeepPHYTextDescriptionMode
    initial_image_path: str
    pattern_run_dir: str | None = None
    pattern_library_path: str | None = None
    records: list[DeepPHYTextRecord] = Field(default_factory=list)
    success: bool | None = None
    terminal_status: str | None = None
    native_metrics: dict[str, Any] = Field(default_factory=dict)
    final_trace_path: str | None = None
    output_root: str | None = None
    report_root: str | None = None
    total_duration_seconds: float | None = None
    completed_at: str | None = None


class DeepPHYTextEnvironmentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: DeepPHYTextEnvId
    task_count: int
    success_count: int
    failure_count: int
    description_modes: list[DeepPHYTextDescriptionMode] = Field(default_factory=list)
    evidence_status_counts: dict[str, int] = Field(default_factory=dict)
    total_duration_seconds: float | None = None
    average_task_duration_seconds: float | None = None


class DeepPHYTextEvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    report_root: str | None = None
    pattern_run_dir: str | None = None
    results: list[DeepPHYTextEvaluationResult] = Field(default_factory=list)
    env_summaries: list[DeepPHYTextEnvironmentSummary] = Field(default_factory=list)
    run_dir: str | None = None
    report_dir: str | None = None
    results_path: str | None = None
    report_path: str | None = None
    total_duration_seconds: float | None = None
