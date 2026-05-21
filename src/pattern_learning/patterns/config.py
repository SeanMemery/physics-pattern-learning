from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from config_loading import load_yaml_config
from llm_config import GLOBAL_MIN_COMPLETION_TOKENS, apply_shared_llm_defaults
from paths import get_artifact_layout
from .losses import LossConfig, default_loss_configs


PATTERN_LEARNING_CONFIG_VERSION = "1"


class PatternLearningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_version: str = PATTERN_LEARNING_CONFIG_VERSION
    env_id: str | None = None
    output_path: str | None = None
    prediction_manifest: str | None = None
    dataset_manifest: str | None = None
    labels_path: str | None = None
    split: list[str] = Field(default_factory=list)
    validation_split: list[str] = Field(default_factory=list)
    limit: int | None = Field(default=None, ge=1)
    generator_model: str | None = None
    generator_base_url: str | None = None
    generator_timeout: int = Field(default=0, ge=0)
    generator_retries: int = Field(default=3, ge=1)
    iteration_count: int = Field(default=8, ge=1)
    workers: int = Field(default=1, ge=1)
    seed: int = Field(default=42)
    population_size: int = Field(default=64, ge=1)
    elite_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    children_per_elite: int = Field(default=2, ge=0)
    crossover_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    patience_factor: float = Field(default=0.35, gt=0.0, le=1.0)
    min_candidate_evaluations: int = Field(default=100, ge=1)
    no_improvement_patience_evaluations: int = Field(default=15, ge=1)
    max_candidate_evaluations: int = Field(default=250, ge=1)
    accept_margin: float = Field(default=0.001, ge=0.0)
    max_repairs: int = Field(default=2, ge=0)
    mutate_temperature: float = Field(default=0.7)
    mutate_max_tokens: int = Field(default=GLOBAL_MIN_COMPLETION_TOKENS, ge=1)
    losses: list[LossConfig] | None = None
    primitive_source: str | None = None

    @model_validator(mode="after")
    def validate_sources(self) -> "PatternLearningConfig":
        if self.generator_model is None:
            raise ValueError("generator_model must be set")
        if not self.generator_base_url or not self.generator_base_url.strip():
            raise ValueError("generator_base_url must be non-empty")
        overlap = sorted(set(self.split) & set(self.validation_split))
        if overlap:
            raise ValueError(
                "pattern learning train and validation splits must not overlap: "
                + ", ".join(overlap)
            )
        resolved_losses = self.losses if self.losses is not None else default_loss_configs()
        object.__setattr__(self, "losses", resolved_losses)
        if not self.losses:
            raise ValueError("pattern learning config must define at least one loss")
        return self


def default_pattern_learning_config_path() -> Path:
    return get_artifact_layout().config_root / "pattern_learning.yaml"


def load_pattern_learning_config(path: Path) -> PatternLearningConfig:
    data = load_yaml_config(path)
    data = apply_shared_llm_defaults(
        data,
        model_key="generator_model",
        base_url_key="generator_base_url",
        timeout_key="generator_timeout",
    )
    return PatternLearningConfig.model_validate(data)


def load_pattern_labels(path: Path) -> list[str]:
    labels: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            labels.append(line)
    if not labels:
        raise ValueError(f"Pattern labels file is empty: {path}")
    return labels
