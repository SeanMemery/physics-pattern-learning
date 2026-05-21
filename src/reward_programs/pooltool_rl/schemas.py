from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class LLMSettings(BaseModel):
    base_url: str
    model: str
    timeout_seconds: int = 120
    api_key_env: str = "LLM_ROUTER_API_KEY"


class OptimizationDefaults(BaseModel):
    budgets: list[int] = Field(default_factory=lambda: [10, 50, 100, 250, 500, 1000])
    scene_count: int = 10
    seed_count: int = 5
    search_budget: int = 250


class ValueModelDefaults(BaseModel):
    budgets: list[int] = Field(default_factory=lambda: [128, 256, 512, 1024])
    train_scene_count: int = 64
    validation_scene_count: int = 16
    actions_per_scene: int = 24
    validation_candidates: int = 64
    hidden_size: int = 64
    epochs: int = 25
    learning_rate: float = 1e-3
    batch_size: int = 32


class AppConfig(BaseModel):
    llm: LLMSettings
    optimization: OptimizationDefaults = Field(default_factory=OptimizationDefaults)
    value_model: ValueModelDefaults = Field(default_factory=ValueModelDefaults)


class GoalSpec(BaseModel):
    id: str
    text: str
    dsl_program: str | None = None
    verifier_id: str | None = None
    verifier_dsl_program: str | None = None


class BallPosition(BaseModel):
    ball_id: str
    x: float
    y: float


class SceneSpec(BaseModel):
    sample_id: str
    trace_path: Path
    initial_scene_overhead_path: Path
    initial_scene_askew_path: Path | None = None
    table_width: float
    table_length: float
    speed_mapping: dict[str, float]
    strikespot_mapping: dict[str, dict[str, float]]
    initial_ball_positions: list[BallPosition]


class ActionSpec(BaseModel):
    speed: float
    strikespot_a: float
    strikespot_b: float


class RewardValue(BaseModel):
    score: float
    success: bool
    event_time: float | None = None


class RewardProgram(BaseModel):
    goal_text: str
    dsl_program: str
    synthesized: bool
    synthesis_request_path: Path | None = None
    synthesis_response_path: Path | None = None


class OptimizationOutcome(BaseModel):
    action: ActionSpec
    best_reward: float
    success: bool
    trace_path: Path
    reward_path: Path


def to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value
