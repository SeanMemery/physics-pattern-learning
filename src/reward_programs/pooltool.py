from __future__ import annotations

import ast
import base64
import json
import math
import hashlib
import mimetypes
import os
import random
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw

from adapters.pooltool import PooltoolAdapter
from adapters.registry import AdapterRegistry
from debug import progress_log
from llm_config import GLOBAL_MIN_COMPLETION_TOKENS
from pattern_library_detectors import apply_pattern_library
from pattern_library_registry import PatternLibrary, load_pattern_library
from pattern_library_schema import Pattern
from pattern_learning.patterns.scoring import (
    _request_json,
    extract_chat_completion_text,
    request_chat_completion_with_text_retry,
)
from schema import CanonicalTrace
from traces.simulator import build_provenance, build_simulator_trace, make_event, make_step


REWARD_PROGRAM_ROOT_DIRNAME = "reward_program_experiments"
OPTIMIZATION_EXPERIMENT_DIRNAME = "optimization"
VALUE_MODEL_EXPERIMENT_DIRNAME = "value_models"
OPTIMIZATION_SUITE_DIRNAME = "optimization_suites"
VALUE_MODEL_SWEEP_DIRNAME = "value_model_sweeps"
SINGLE_SCENE_EXPERIMENT_DIRNAME = "single_scene"
DEFAULT_OPTIMIZATION_BUDGETS = (10, 50, 100, 250, 500, 1000)
DEFAULT_OPTIMIZATION_SEED_COUNT = 5
DEFAULT_OPTIMIZATION_SCENE_COUNT = 10
DEFAULT_POOLTOOL_RACK_JITTER = 0.12
DEFAULT_VALUE_MODEL_TRAIN_SCENE_COUNT = 64
DEFAULT_VALUE_MODEL_TRAIN_ACTIONS_PER_SCENE = 32
DEFAULT_VALUE_MODEL_VALIDATION_SCENE_COUNT = 12
DEFAULT_VALUE_MODEL_VALIDATION_CANDIDATES = 128
DEFAULT_VALUE_MODEL_EPOCHS = 50
DEFAULT_VALUE_MODEL_BATCH_SIZE = 64
DEFAULT_VALUE_MODEL_LEARNING_RATE = 1e-3
DEFAULT_VALUE_MODEL_IMAGE_SIZE = 256
DEFAULT_VALUE_MODEL_SELECTION_MULTIPLIER = 1
DSL_CODE_FENCE_RE = re.compile(r"```(?:dsl|text)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
DEFAULT_POOLTOOL_REWARD_GOALS: tuple[tuple[str, str], ...] = (
    ("pot-nine-keep-cue", 'Pot the 9 ball while keeping the cue ball on the table.'),
    ("pot-nine-after-clear", 'Pot the 9 ball after clearing a path through the rack with an earlier collision.'),
    ("soft-pot-nine", 'Pot the 9 ball with a soft, controlled shot that avoids losing the cue ball.'),
    ("bank-or-carom-nine", 'Use a bank or carom style shot if needed to pot the 9 ball while keeping the cue ball safe.'),
    ("open-and-finish", 'Break up congestion around the 9 ball and still finish by potting the 9 ball with the cue ball remaining.'),
)
DEFAULT_VALUE_MODEL_SWEEP_PRESET: tuple[dict[str, Any], ...] = (
    {
        "config_id": "baseline-64x32",
        "train_scene_count": 64,
        "train_actions_per_scene": 32,
        "validation_scene_count": 12,
        "validation_candidates_per_scene": 128,
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "image_size": 256,
    },
    {
        "config_id": "lower-ridge-64x32",
        "train_scene_count": 64,
        "train_actions_per_scene": 32,
        "validation_scene_count": 12,
        "validation_candidates_per_scene": 128,
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 5e-4,
        "image_size": 256,
    },
    {
        "config_id": "more-data-80x32",
        "train_scene_count": 80,
        "train_actions_per_scene": 32,
        "validation_scene_count": 12,
        "validation_candidates_per_scene": 128,
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "image_size": 256,
    },
    {
        "config_id": "more-actions-64x48",
        "train_scene_count": 64,
        "train_actions_per_scene": 48,
        "validation_scene_count": 12,
        "validation_candidates_per_scene": 128,
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "image_size": 256,
    },
    {
        "config_id": "larger-image-64x32-320",
        "train_scene_count": 64,
        "train_actions_per_scene": 32,
        "validation_scene_count": 12,
        "validation_candidates_per_scene": 128,
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "image_size": 320,
    },
)


class RewardProgramError(RuntimeError):
    pass


def _reward_progress(event: str, **fields: Any) -> None:
    progress_log(f"dashboard.workflows.pooltool_reward.{event}", **fields)
    progress_path = str(os.environ.get("PATTERN_REWARD_PROGRESS_PATH", "") or "").strip()
    if not progress_path:
        return
    payload = {
        "event": event,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    try:
        path = Path(progress_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError:
        return


@dataclass(frozen=True)
class RewardProgramDefinition:
    goal_text: str
    dsl_program: str
    source_run_dir: str
    source_run_kind: str
    source_run_id: str
    model_name: str
    synthesized: bool
    synthesis_response_path: str | None = None


@dataclass(frozen=True)
class PooltoolRewardGoal:
    goal_id: str
    goal_text: str


@dataclass(frozen=True)
class SceneDefinition:
    scene_id: str
    seed: int
    task_id: str
    table_width: float
    table_length: float
    initial_ball_positions: list[dict[str, Any]]
    image_path: str
    rack_jitter: float


@dataclass(frozen=True)
class ContinuousPooltoolAction:
    speed: float
    strikespot_a: float
    strikespot_b: float

    def to_json(self) -> dict[str, float]:
        return {
            "speed": float(self.speed),
            "strikespot_a": float(self.strikespot_a),
            "strikespot_b": float(self.strikespot_b),
        }


@dataclass(frozen=True)
class OptimizationBudgetOutcome:
    budget: int
    seed: int
    scene_id: str
    reward_kind: str
    best_reward: float
    success: bool
    action: ContinuousPooltoolAction
    trace_path: str
    trace_reward_path: str


@dataclass(frozen=True)
class BudgetAggregate:
    budget: int
    reward_kind: str
    scene_count: int
    success_rate: float
    mean_reward: float
    seed_count: int


@dataclass(frozen=True)
class PooltoolOptimizationExperimentRequest:
    run_id: str
    output_root: str
    source_run_dir: str
    goal_text: str
    dsl_program: str
    model_name: str
    scene_count: int = DEFAULT_OPTIMIZATION_SCENE_COUNT
    seed_count: int = DEFAULT_OPTIMIZATION_SEED_COUNT
    budgets: tuple[int, ...] = DEFAULT_OPTIMIZATION_BUDGETS
    rack_jitter: float = DEFAULT_POOLTOOL_RACK_JITTER


@dataclass(frozen=True)
class PooltoolOptimizationExperimentResult:
    request: PooltoolOptimizationExperimentRequest
    reward_program: RewardProgramDefinition
    scene_summaries: list[SceneDefinition]
    budget_aggregates: list[BudgetAggregate]
    outcomes: list[OptimizationBudgetOutcome]
    outcome_count: int
    summary_path: str


@dataclass(frozen=True)
class PooltoolSingleSceneExperimentRequest:
    run_id: str
    output_root: str
    source_run_dir: str
    scene_path: str
    goal_text: str
    dsl_program: str
    model_name: str
    reward_kind: str = "synthesized"
    budget: int = 250
    seed: int = 0


@dataclass(frozen=True)
class PooltoolSingleSceneExperimentResult:
    request: PooltoolSingleSceneExperimentRequest
    reward_program: RewardProgramDefinition
    scene_summary: SceneDefinition
    outcome: OptimizationBudgetOutcome
    initial_scene_paths: list[str]
    simulation_video_paths: list[str]
    summary_path: str


@dataclass(frozen=True)
class ValueModelEvalPoint:
    epoch: int
    train_loss: float
    validation_mean_reward: float
    validation_success_rate: float


@dataclass(frozen=True)
class ValueModelValidationOutcome:
    epoch: int
    scene_id: str
    reward: float
    success: bool
    action: ContinuousPooltoolAction
    trace_path: str
    trace_reward_path: str


@dataclass(frozen=True)
class PooltoolValueModelExperimentRequest:
    run_id: str
    output_root: str
    source_run_dir: str
    goal_text: str
    dsl_program: str
    model_name: str
    train_scene_count: int = DEFAULT_VALUE_MODEL_TRAIN_SCENE_COUNT
    train_actions_per_scene: int = DEFAULT_VALUE_MODEL_TRAIN_ACTIONS_PER_SCENE
    validation_scene_count: int = DEFAULT_VALUE_MODEL_VALIDATION_SCENE_COUNT
    validation_candidates_per_scene: int = DEFAULT_VALUE_MODEL_VALIDATION_CANDIDATES
    epochs: int = DEFAULT_VALUE_MODEL_EPOCHS
    batch_size: int = DEFAULT_VALUE_MODEL_BATCH_SIZE
    learning_rate: float = DEFAULT_VALUE_MODEL_LEARNING_RATE
    image_size: int = DEFAULT_VALUE_MODEL_IMAGE_SIZE
    rack_jitter: float = DEFAULT_POOLTOOL_RACK_JITTER


@dataclass(frozen=True)
class PooltoolValueModelExperimentResult:
    request: PooltoolValueModelExperimentRequest
    reward_program: RewardProgramDefinition
    train_sample_count: int
    validation_scene_count: int
    eval_curve: list[ValueModelEvalPoint]
    validation_outcomes: list[ValueModelValidationOutcome]
    best_epoch: int
    best_validation_mean_reward: float
    best_validation_success_rate: float
    summary_path: str


@dataclass(frozen=True)
class PooltoolOptimizationSuiteGoalResult:
    goal_id: str
    goal_text: str
    run_id: str
    summary_path: str
    best_binary_success_rate: float
    best_binary_mean_reward: float
    best_binary_budget: int
    best_dense_mean_reward: float
    best_dense_budget: int


@dataclass(frozen=True)
class PooltoolOptimizationSuiteResult:
    run_id: str
    goals: list[PooltoolOptimizationSuiteGoalResult]
    summary_path: str


@dataclass(frozen=True)
class PooltoolValueModelSweepConfig:
    config_id: str
    train_scene_count: int = DEFAULT_VALUE_MODEL_TRAIN_SCENE_COUNT
    train_actions_per_scene: int = DEFAULT_VALUE_MODEL_TRAIN_ACTIONS_PER_SCENE
    validation_scene_count: int = DEFAULT_VALUE_MODEL_VALIDATION_SCENE_COUNT
    validation_candidates_per_scene: int = DEFAULT_VALUE_MODEL_VALIDATION_CANDIDATES
    epochs: int = DEFAULT_VALUE_MODEL_EPOCHS
    batch_size: int = DEFAULT_VALUE_MODEL_BATCH_SIZE
    learning_rate: float = DEFAULT_VALUE_MODEL_LEARNING_RATE
    image_size: int = DEFAULT_VALUE_MODEL_IMAGE_SIZE


@dataclass(frozen=True)
class PooltoolValueModelSweepRunResult:
    goal_id: str
    goal_text: str
    config_id: str
    run_id: str
    summary_path: str
    best_epoch: int
    best_validation_mean_reward: float
    best_validation_success_rate: float


@dataclass(frozen=True)
class PooltoolValueModelSweepResult:
    run_id: str
    goals: list[PooltoolRewardGoal]
    configs: list[PooltoolValueModelSweepConfig]
    runs: list[PooltoolValueModelSweepRunResult]
    summary_path: str


@dataclass
class _RewardProgramContext:
    trace: CanonicalTrace
    patterns: list[Pattern]
    table_width: float
    table_length: float
    pattern_labels: list[str] = field(default_factory=list)
    patterns_by_label: dict[str, list[Pattern]] = field(default_factory=dict)


@dataclass
class _ActionBounds:
    speed_min: float
    speed_max: float
    a_min: float
    a_max: float
    b_min: float
    b_max: float


@dataclass
class _SampleRecord:
    image_path: Path
    scene_summary: SceneDefinition
    action: ContinuousPooltoolAction
    reward: float
    success: bool
    scene_id: str


@dataclass(frozen=True)
class _FeatureStats:
    mean: np.ndarray
    scale: np.ndarray


class _PooltoolValueModel:
    def __init__(
        self,
        *,
        stats: _FeatureStats,
        reward_weights: np.ndarray,
        success_weights: np.ndarray,
        blend_alpha: float,
        image_size: int,
    ) -> None:
        self.stats = stats
        self.reward_weights = reward_weights.astype(np.float32)
        self.success_weights = success_weights.astype(np.float32)
        self.blend_alpha = float(blend_alpha)
        self.image_size = int(image_size)

    @classmethod
    def fit(
        cls,
        *,
        rows: list[_SampleRecord],
        image_size: int,
        ridge_alpha: float,
    ) -> tuple["_PooltoolValueModel", float]:
        if not rows:
            raise RewardProgramError("No reward-program samples available for value-model fitting.")
        feature_matrix = np.stack(
            [
                _extract_value_features(
                    image_path=row.image_path,
                    scene_summary=row.scene_summary,
                    action=row.action,
                    image_size=image_size,
                )
                for row in rows
            ],
            axis=0,
        ).astype(np.float32)
        reward_targets = np.asarray([row.reward for row in rows], dtype=np.float32)
        success_targets = np.asarray([1.0 if row.success else 0.0 for row in rows], dtype=np.float32)
        mean = feature_matrix.mean(axis=0)
        scale = feature_matrix.std(axis=0)
        scale[scale < 1e-6] = 1.0
        normalized = (feature_matrix - mean) / scale
        design = np.concatenate(
            [normalized, np.ones((normalized.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        success_count = max(float(success_targets.sum()), 1.0)
        failure_count = max(float(len(rows)) - success_count, 1.0)
        positive_weight = max(1.0, failure_count / success_count)
        success_weights_diag = np.where(success_targets > 0.5, positive_weight, 1.0).astype(np.float32)
        reward_weights_diag = (1.0 + success_targets + reward_targets).astype(np.float32)
        regularizer = np.eye(design.shape[1], dtype=np.float32) * max(float(ridge_alpha), 1e-6)
        regularizer[-1, -1] = 0.0

        weighted_reward_design = design * reward_weights_diag[:, None]
        weighted_success_design = design * success_weights_diag[:, None]
        reward_weights = np.linalg.solve(
            design.T @ weighted_reward_design + regularizer,
            design.T @ (reward_targets * reward_weights_diag),
        )
        success_weights = np.linalg.solve(
            design.T @ weighted_success_design + regularizer,
            design.T @ (success_targets * success_weights_diag),
        )
        model = cls(
            stats=_FeatureStats(mean=mean.astype(np.float32), scale=scale.astype(np.float32)),
            reward_weights=reward_weights.astype(np.float32),
            success_weights=success_weights.astype(np.float32),
            blend_alpha=0.75,
            image_size=image_size,
        )
        predictions = model.predict_reward_rows(rows)
        train_loss = float(np.mean((predictions - reward_targets) ** 2))
        return model, train_loss

    def predict_reward_rows(self, rows: Sequence[_SampleRecord]) -> np.ndarray:
        features = np.stack(
            [
                _extract_value_features(
                    image_path=row.image_path,
                    scene_summary=row.scene_summary,
                    action=row.action,
                    image_size=self.image_size,
                )
                for row in rows
            ],
            axis=0,
        ).astype(np.float32)
        return self.predict_reward_features(features)

    def predict_success_rows(self, rows: Sequence[_SampleRecord]) -> np.ndarray:
        features = np.stack(
            [
                _extract_value_features(
                    image_path=row.image_path,
                    scene_summary=row.scene_summary,
                    action=row.action,
                    image_size=self.image_size,
                )
                for row in rows
            ],
            axis=0,
        ).astype(np.float32)
        return self.predict_success_features(features)

    def predict_actions(
        self,
        *,
        image_path: Path,
        scene_summary: SceneDefinition,
        actions: Sequence[ContinuousPooltoolAction],
    ) -> np.ndarray:
        features = np.stack(
            [
                _extract_value_features(
                    image_path=image_path,
                    scene_summary=scene_summary,
                    action=action,
                    image_size=self.image_size,
                )
                for action in actions
            ],
            axis=0,
        ).astype(np.float32)
        return self.predict_features(features)

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        reward_scores = self.predict_reward_features(features)
        success_scores = self.predict_success_features(features)
        return np.clip(
            ((1.0 - self.blend_alpha) * reward_scores) + (self.blend_alpha * success_scores),
            0.0,
            1.0,
        )

    def predict_reward_features(self, features: np.ndarray) -> np.ndarray:
        normalized = (features - self.stats.mean) / self.stats.scale
        design = np.concatenate(
            [normalized, np.ones((normalized.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        scores = design @ self.reward_weights
        return np.clip(scores, 0.0, 1.0)

    def predict_success_features(self, features: np.ndarray) -> np.ndarray:
        normalized = (features - self.stats.mean) / self.stats.scale
        design = np.concatenate(
            [normalized, np.ones((normalized.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        scores = design @ self.success_weights
        return np.clip(scores, 0.0, 1.0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.stats.mean,
            scale=self.stats.scale,
            reward_weights=self.reward_weights,
            success_weights=self.success_weights,
            blend_alpha=np.asarray([self.blend_alpha], dtype=np.float32),
            image_size=np.asarray([self.image_size], dtype=np.int32),
        )

    @classmethod
    def load(cls, path: Path) -> "_PooltoolValueModel":
        payload = np.load(path)
        return cls(
            stats=_FeatureStats(
                mean=np.asarray(payload["mean"], dtype=np.float32),
                scale=np.asarray(payload["scale"], dtype=np.float32),
            ),
            reward_weights=np.asarray(payload["reward_weights"], dtype=np.float32),
            success_weights=np.asarray(payload["success_weights"], dtype=np.float32),
            blend_alpha=float(np.asarray(payload["blend_alpha"]).reshape(-1)[0]),
            image_size=int(np.asarray(payload["image_size"]).reshape(-1)[0]),
        )


def _feature_side(image_size: int) -> int:
    if image_size <= 64:
        return max(8, image_size // 2)
    return 24


def _extract_scene_features(*, image_path: Path, image_size: int) -> np.ndarray:
    side = _feature_side(image_size)
    image = Image.open(image_path).convert("L").resize((side, side))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    y_axis = np.linspace(0.0, 1.0, side, dtype=np.float32)
    x_axis = np.linspace(0.0, 1.0, side, dtype=np.float32)
    mass = float(arr.sum()) + 1e-6
    x_center = float((arr.sum(axis=0) @ x_axis) / mass)
    y_center = float((arr.sum(axis=1) @ y_axis) / mass)
    row_profile = arr.mean(axis=1)
    col_profile = arr.mean(axis=0)
    stats = np.asarray(
        [
            float(arr.mean()),
            float(arr.std()),
            float(arr.min()),
            float(arr.max()),
            x_center,
            y_center,
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [arr.reshape(-1), row_profile.astype(np.float32), col_profile.astype(np.float32), stats],
        axis=0,
    )


def _extract_scene_layout_features(scene_summary: SceneDefinition) -> np.ndarray:
    width = max(float(scene_summary.table_width), 1e-6)
    length = max(float(scene_summary.table_length), 1e-6)
    balls = {
        str(item.get("ball_id")): (
            float(item.get("x", 0.0)) / width,
            float(item.get("y", 0.0)) / length,
        )
        for item in scene_summary.initial_ball_positions
        if item.get("ball_id") is not None
    }
    ball_order = ["cue"] + [str(index) for index in range(1, 10)]
    coords: list[float] = []
    present_flags: list[float] = []
    for ball_id in ball_order:
        xy = balls.get(ball_id)
        if xy is None:
            coords.extend([0.0, 0.0])
            present_flags.append(0.0)
        else:
            coords.extend([xy[0], xy[1]])
            present_flags.append(1.0)
    cue_x, cue_y = balls.get("cue", (0.0, 0.0))
    nine_x, nine_y = balls.get("9", (0.0, 0.0))
    cue_to_nine = math.hypot(cue_x - nine_x, cue_y - nine_y)
    pockets = [
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 1.0),
        (1.0, 1.0),
    ]
    nine_to_pocket = min(math.hypot(nine_x - px, nine_y - py) for px, py in pockets)
    cue_to_pocket = min(math.hypot(cue_x - px, cue_y - py) for px, py in pockets)
    cluster_center_x = sum(x for x, _y in balls.values()) / max(len(balls), 1)
    cluster_center_y = sum(y for _x, y in balls.values()) / max(len(balls), 1)
    stats = [
        width,
        length,
        float(len(balls)),
        cue_x,
        cue_y,
        nine_x,
        nine_y,
        cue_to_nine,
        nine_to_pocket,
        cue_to_pocket,
        cluster_center_x,
        cluster_center_y,
    ]
    return np.asarray([*coords, *present_flags, *stats], dtype=np.float32)


def _action_features(action: ContinuousPooltoolAction) -> np.ndarray:
    speed = float(action.speed)
    spot_a = float(action.strikespot_a)
    spot_b = float(action.strikespot_b)
    return np.asarray(
        [
            speed,
            spot_a,
            spot_b,
            speed * speed,
            spot_a * spot_a,
            spot_b * spot_b,
            speed * spot_a,
            speed * spot_b,
            spot_a * spot_b,
        ],
        dtype=np.float32,
    )


def _extract_value_features(
    *,
    image_path: Path,
    scene_summary: SceneDefinition,
    action: ContinuousPooltoolAction,
    image_size: int,
) -> np.ndarray:
    scene = _extract_scene_features(image_path=image_path, image_size=image_size)
    layout = _extract_scene_layout_features(scene_summary)
    action_vec = _action_features(action)
    scene_stats = scene[-6:]
    layout_stats = layout[-12:]
    interactions = np.concatenate(
        [
            scene_stats[:3] * action_vec[:3],
            scene_stats[3:] * action_vec[:3],
            layout_stats[:3] * action_vec[:3],
            layout_stats[3:6] * action_vec[:3],
            layout_stats[6:9] * action_vec[:3],
        ],
        axis=0,
    )
    return np.concatenate([scene, layout, action_vec, interactions], axis=0)


def _stable_subset_rows(
    rows: Sequence[_SampleRecord],
    *,
    subset_size: int,
    seed_label: str,
) -> list[_SampleRecord]:
    decorated: list[tuple[str, _SampleRecord]] = []
    for row in rows:
        key = f"{seed_label}|{row.scene_id}|{row.image_path}|{row.action.speed:.6f}|{row.action.strikespot_a:.6f}|{row.action.strikespot_b:.6f}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        decorated.append((digest, row))
    decorated.sort(key=lambda item: item[0])
    return [row for _digest, row in decorated[:subset_size]]


def _refine_actions_with_model(
    *,
    model: _PooltoolValueModel,
    image_path: Path,
    scene_summary: SceneDefinition,
    candidate_actions: Sequence[ContinuousPooltoolAction],
    bounds: _ActionBounds,
    rng: random.Random,
    keep_top_k: int = 8,
    refinement_steps: int = 24,
) -> ContinuousPooltoolAction:
    if not candidate_actions:
        raise RewardProgramError("No candidate actions available for value-model refinement.")
    predicted = model.predict_actions(
        image_path=image_path,
        scene_summary=scene_summary,
        actions=candidate_actions,
    )
    ranked_indices = np.argsort(predicted)[::-1][: max(1, min(len(candidate_actions), keep_top_k))]
    incumbent_actions = [candidate_actions[int(index)] for index in ranked_indices]
    incumbent_scores = [float(predicted[int(index)]) for index in ranked_indices]
    for action_index, incumbent in enumerate(list(incumbent_actions)):
        current_action = incumbent
        current_score = incumbent_scores[action_index]
        best_action = incumbent
        best_score = current_score
        for step in range(refinement_steps):
            temperature = max(0.05, 1.0 - (step / max(refinement_steps - 1, 1)))
            proposal = _perturb_action(
                current_action,
                rng=rng,
                bounds=bounds,
                temperature=temperature,
            )
            proposal_score = float(
                model.predict_actions(
                    image_path=image_path,
                    scene_summary=scene_summary,
                    actions=[proposal],
                )[0]
            )
            if proposal_score >= current_score or rng.random() < math.exp((proposal_score - current_score) / max(temperature, 1e-6)):
                current_action = proposal
                current_score = proposal_score
            if proposal_score > best_score:
                best_action = proposal
                best_score = proposal_score
        incumbent_actions[action_index] = best_action
        incumbent_scores[action_index] = best_score
    best_index = max(range(len(incumbent_actions)), key=lambda index: incumbent_scores[index])
    return incumbent_actions[best_index]


def _reward_program_root() -> Path:
    from paths import get_artifact_layout

    layout = get_artifact_layout()
    return (
        layout.data_root
        / "patterns"
        / "pooltool"
        / REWARD_PROGRAM_ROOT_DIRNAME
    )


def _optimization_root(*, run_id: str) -> Path:
    return _reward_program_root() / OPTIMIZATION_EXPERIMENT_DIRNAME / run_id


def _value_model_root(*, run_id: str) -> Path:
    return _reward_program_root() / VALUE_MODEL_EXPERIMENT_DIRNAME / run_id


def _optimization_suite_root(*, run_id: str) -> Path:
    return _reward_program_root() / OPTIMIZATION_SUITE_DIRNAME / run_id


def _value_model_sweep_root(*, run_id: str) -> Path:
    return _reward_program_root() / VALUE_MODEL_SWEEP_DIRNAME / run_id


def _single_scene_root(*, run_id: str) -> Path:
    return _reward_program_root() / SINGLE_SCENE_EXPERIMENT_DIRNAME / run_id


def _ensure_run_root(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slugify_goal_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return token or "goal"


def default_pooltool_reward_goals() -> list[PooltoolRewardGoal]:
    return [
        PooltoolRewardGoal(goal_id=goal_id, goal_text=goal_text)
        for goal_id, goal_text in DEFAULT_POOLTOOL_REWARD_GOALS
    ]


def default_pooltool_value_model_sweep_configs() -> list[PooltoolValueModelSweepConfig]:
    return [PooltoolValueModelSweepConfig(**payload) for payload in DEFAULT_VALUE_MODEL_SWEEP_PRESET]


def generate_pooltool_random_scene(
    *,
    output_root: Path,
    scene_id: str,
    seed: int,
    rack_jitter: float = DEFAULT_POOLTOOL_RACK_JITTER,
) -> SceneDefinition:
    adapter = _pooltool_adapter()
    _system, summary = _create_random_scene(
        adapter=adapter,
        scene_id=scene_id,
        seed=seed,
        output_root=output_root,
        rack_jitter=rack_jitter,
    )
    return summary


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalize_dsl_program(program_text: str) -> str:
    stripped = str(program_text or "").strip()
    if not stripped:
        return ""
    fenced = DSL_CODE_FENCE_RE.search(stripped)
    if fenced is not None:
        stripped = fenced.group(1).strip()
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", stripped, re.DOTALL | re.IGNORECASE)
    if answer_match is not None:
        stripped = answer_match.group(1).strip()
        fenced = DSL_CODE_FENCE_RE.search(stripped)
        if fenced is not None:
            stripped = fenced.group(1).strip()
    return stripped


def _dsl_reference_text() -> str:
    return "\n".join(
        [
            'AND(expr1, expr2, ...) -> all clauses must hold; dense reward is the mean clause score.',
            'OR(expr1, expr2, ...) -> any clause may hold; dense reward is the max clause score.',
            'NOT(expr) -> negation.',
            'EVENT("pattern label", {"param": value}) -> a learned pattern label fires with optional parameter matching.',
            'AFTER("first label", "second label", first_params={...}, second_params={...}) -> ordered pattern events.',
            'COUNT("pattern label", target, {"param": value}) -> count matching pattern activations and reward closeness to target.',
            'GT("pattern label", threshold, {"param": value}) -> reward higher counts above threshold.',
            'POTTED("9") -> whether the named ball was potted.',
            'BALL_REMAINING("cue") -> whether the named ball remains on table at the end.',
            'NEARBY_AT("9", x, y, t_fraction, threshold_strength=0.2) -> ball position near a target coordinate at fractional time.',
            'Use only a single expression. Build the reward as a sum of positive terms with AND where possible.',
        ]
    )


def _build_multimodal_prompt_content(*, text: str, image_paths: Sequence[Path]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for index, image_path in enumerate(image_paths, start=1):
        if not isinstance(image_path, Path) or not image_path.is_file():
            continue
        content.append({"type": "text", "text": f"Scene image {index}."})
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
        image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
            }
        )
    return content


def _resize_prompt_image_in_place(path: Path, *, max_side: int = 512) -> None:
    with Image.open(path) as image:
        width, height = image.size
        longest = max(width, height)
        if longest <= max_side:
            image.save(path)
            return
        scale = max_side / float(longest)
        resized = image.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            Image.Resampling.LANCZOS,
        )
        resized.save(path)


def _prepare_prompt_image_copy(source_path: Path, *, output_dir: Path) -> Path | None:
    if not source_path.is_file():
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = output_dir / source_path.name
    copied.write_bytes(source_path.read_bytes())
    _resize_prompt_image_in_place(copied, max_side=512)
    return copied


def _summarize_pattern_library(library: PatternLibrary) -> str:
    lines: list[str] = []
    for detector in library.detectors:
        label = detector.label.strip()
        if not label:
            continue
        description = (detector.description or "").strip()
        parameter_keys = []
        metadata = detector.metadata if isinstance(detector.metadata, dict) else {}
        if isinstance(metadata.get("parameter_keys"), list):
            parameter_keys = [
                str(item).strip()
                for item in metadata.get("parameter_keys")
                if str(item).strip()
            ]
        suffix = f" Parameters: {', '.join(parameter_keys)}." if parameter_keys else ""
        if description:
            lines.append(f"- {label}: {description}.{suffix}".replace("..", "."))
        else:
            lines.append(f"- {label}.{suffix}".replace("..", "."))
    return "\n".join(lines[:60]) if lines else "- No learned detectors are available."


def _scene_summary(scene: SceneDefinition) -> str:
    lines = [
        f"Scene id: {scene.scene_id}",
        f"Table size: {scene.table_width:.3f} x {scene.table_length:.3f}",
        "Initial balls:",
    ]
    for item in scene.initial_ball_positions:
        lines.append(
            f"- Ball {item.get('ball_id')}: x={float(item.get('x', 0.0)):.3f}, y={float(item.get('y', 0.0)):.3f}"
        )
    return "\n".join(lines)


def synthesize_pooltool_reward_program(
    *,
    goal_text: str,
    source_run_dir: Path,
    model_name: str,
    llm_base_url: str,
    llm_timeout: int,
    artifact_root: Path,
    representative_scene: SceneDefinition,
) -> RewardProgramDefinition:
    if not goal_text.strip():
        raise ValueError("Goal text is required when no DSL program is provided.")
    library_path = source_run_dir / "pattern_library.json"
    if not library_path.is_file():
        raise FileNotFoundError(f"Pattern library not found: {library_path}")
    library = load_pattern_library(library_path)
    system_prompt = (
        "Given the following goal, scene, and DSL reference, propose one DSL expression.\n"
        "This expression defines a reward function. This reward function will be maximised by an optimisation process.\n"
        "First infer the intended shot plan from the scene image: which ball the cue ball should strike, what intermediate contacts should happen, and how the target outcome should unfold.\n"
        "Then encode that intended sequence in the DSL so the reward favours the planned shot, not just the final outcome.\n"
        "If the goal is to pot a specific ball, prefer a reward that includes the cue ball striking the right object ball and that object ball then doing the intended thing.\n"
        "Use the learned pattern labels if they help express the goal.\n"
        "Do not include comments or explanation in the DSL code.\n"
        "Respond with only the DSL inside ```dsl fences```."
    )
    user_prompt = "\n\n".join(
        [
            f"Goal:\n{goal_text.strip()}",
            f"DSL reference:\n{_dsl_reference_text()}",
            f"Library summary:\n{_summarize_pattern_library(library)}",
            f"Representative scene summary:\n{_scene_summary(representative_scene)}",
            (
                "Shot-planning instruction:\n"
                "Look at the scene image and choose a plausible shot trajectory for satisfying the goal. "
                "The DSL should reward the sequence of events that would make that shot succeed, such as the cue ball contacting a specific target ball before that ball is potted or moved into place."
            ),
        ]
    )
    image_paths: list[Path] = []
    representative_image = Path(str(representative_scene.image_path or "")).resolve()
    if representative_image.is_file():
        prepared_image = _prepare_prompt_image_copy(
            representative_image,
            output_dir=artifact_root / "prompt_media",
        )
        if prepared_image is not None:
            image_paths.append(prepared_image)
    body, response_text = request_chat_completion_with_text_retry(
        base_url=llm_base_url,
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": _build_multimodal_prompt_content(text=user_prompt, image_paths=image_paths),
            },
        ],
        timeout=llm_timeout,
        headers={"Content-Type": "application/json"},
        max_tokens=GLOBAL_MIN_COMPLETION_TOKENS,
        temperature=0.0,
        stream=False,
        extra_payload={"chat_template_kwargs": {"enable_thinking": False}},
        error_prefix="PoolTool reward program synthesis response",
        request_fn=_request_json,
    )
    dsl_program = _normalize_dsl_program(
        extract_chat_completion_text(
            body,
            error_prefix="PoolTool reward program synthesis response",
        )
        or response_text
    )
    if not dsl_program:
        raise RewardProgramError("Reward program synthesis returned no DSL program.")
    _parse_dsl_expression(dsl_program)
    response_path = artifact_root / "synthesis_response.txt"
    request_path = artifact_root / "synthesis_request.json"
    response_path.write_text(response_text, encoding="utf-8")
    _write_json(
        request_path,
        {
            "goal_text": goal_text,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "image_paths": [str(path) for path in image_paths],
            "response_body": body,
        },
    )
    return RewardProgramDefinition(
        goal_text=goal_text.strip(),
        dsl_program=dsl_program,
        source_run_dir=str(source_run_dir),
        source_run_kind=_source_run_kind(source_run_dir),
        source_run_id=source_run_dir.name,
        model_name=model_name,
        synthesized=True,
        synthesis_response_path=str(response_path),
    )


def _source_run_kind(source_run_dir: Path) -> str:
    path_text = str(source_run_dir)
    if "/automatic/" in path_text:
        return "automatic"
    if "/unsupervised/" in path_text:
        return "unsupervised"
    return "supervised"


def _explicit_reward_program(
    *,
    goal_text: str,
    dsl_program: str,
    source_run_dir: Path,
    model_name: str,
) -> RewardProgramDefinition:
    normalized = _normalize_dsl_program(dsl_program)
    if not normalized:
        raise ValueError("DSL program is required.")
    _parse_dsl_expression(normalized)
    return RewardProgramDefinition(
        goal_text=goal_text.strip(),
        dsl_program=normalized,
        source_run_dir=str(source_run_dir),
        source_run_kind=_source_run_kind(source_run_dir),
        source_run_id=source_run_dir.name,
        model_name=model_name,
        synthesized=False,
        synthesis_response_path=None,
    )


def _load_reward_program_definition(
    *,
    goal_text: str,
    dsl_program: str,
    source_run_dir: Path,
    model_name: str,
    llm_base_url: str,
    llm_timeout: int,
    artifact_root: Path,
    representative_scene: SceneDefinition,
) -> RewardProgramDefinition:
    if dsl_program.strip():
        return _explicit_reward_program(
            goal_text=goal_text,
            dsl_program=dsl_program,
            source_run_dir=source_run_dir,
            model_name=model_name,
        )
    return synthesize_pooltool_reward_program(
        goal_text=goal_text,
        source_run_dir=source_run_dir,
        model_name=model_name,
        llm_base_url=llm_base_url,
        llm_timeout=llm_timeout,
        artifact_root=artifact_root,
        representative_scene=representative_scene,
    )


def _parse_dsl_expression(program_text: str) -> ast.AST:
    normalized = _normalize_dsl_program(program_text)
    if not normalized:
        raise RewardProgramError("Empty reward DSL program.")
    try:
        node = ast.parse(normalized, mode="eval").body
    except SyntaxError as exc:
        raise RewardProgramError(f"Invalid reward DSL syntax: {exc}") from exc
    _validate_dsl_node(node)
    return node


_ALLOWED_DSL_NAMES = {
    "AND",
    "OR",
    "NOT",
    "EVENT",
    "AFTER",
    "COUNT",
    "GT",
    "NEARBY_AT",
    "POTTED",
    "BALL_REMAINING",
}


def _validate_dsl_node(node: ast.AST) -> None:
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_DSL_NAMES:
            raise RewardProgramError("Unsupported DSL function.")
        for arg in node.args:
            _validate_dsl_node(arg)
        for keyword in node.keywords:
            if keyword.arg is None:
                raise RewardProgramError("Starred DSL arguments are not supported.")
            _validate_dsl_node(keyword.value)
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (str, int, float, bool)) or node.value is None:
            return
        raise RewardProgramError("Unsupported DSL constant.")
    if isinstance(node, (ast.List, ast.Tuple)):
        for item in node.elts:
            _validate_dsl_node(item)
        return
    if isinstance(node, ast.Dict):
        for key in node.keys:
            if key is not None:
                _validate_dsl_node(key)
        for value in node.values:
            _validate_dsl_node(value)
        return
    raise RewardProgramError(f"Unsupported DSL node: {type(node).__name__}")


def evaluate_reward_program(
    program_text: str,
    *,
    trace: CanonicalTrace,
    library: PatternLibrary | None = None,
    patterns: list[Pattern] | None = None,
    binary_mode: bool = False,
) -> dict[str, Any]:
    parsed = _parse_dsl_expression(program_text)
    resolved_patterns = list(patterns or [])
    if not resolved_patterns and library is not None:
        resolved_patterns = apply_pattern_library(trace, library)
    context = _build_reward_context(trace=trace, patterns=resolved_patterns)
    score = _evaluate_dsl_node(parsed, context=context, binary_mode=binary_mode)
    score = min(1.0, max(0.0, float(score)))
    return {
        "score": score,
        "success": bool(score >= 0.999 if binary_mode else _evaluate_dsl_node(parsed, context=context, binary_mode=True) >= 0.999),
        "pattern_count": len(resolved_patterns),
        "pattern_labels": context.pattern_labels,
    }


def _build_reward_context(*, trace: CanonicalTrace, patterns: list[Pattern]) -> _RewardProgramContext:
    patterns_by_label: dict[str, list[Pattern]] = {}
    pattern_labels: list[str] = []
    for pattern in patterns:
        label = pattern.label.strip()
        if not label:
            continue
        pattern_labels.append(label)
        patterns_by_label.setdefault(label.lower(), []).append(pattern)
    initial_state = trace.initial_state if isinstance(trace.initial_state, dict) else {}
    return _RewardProgramContext(
        trace=trace,
        patterns=patterns,
        table_width=float(initial_state.get("table_width", initial_state.get("table_w", 1.0)) or 1.0),
        table_length=float(initial_state.get("table_length", initial_state.get("table_l", 2.0)) or 2.0),
        pattern_labels=sorted(set(pattern_labels)),
        patterns_by_label=patterns_by_label,
    )


def _evaluate_dsl_node(node: ast.AST, *, context: _RewardProgramContext, binary_mode: bool) -> float:
    if isinstance(node, ast.Call):
        fn_name = node.func.id
        args = [_literal_value(item) if not isinstance(item, ast.Call) else item for item in node.args]
        kwargs = {item.arg: (_literal_value(item.value) if not isinstance(item.value, ast.Call) else item.value) for item in node.keywords}
        if fn_name == "AND":
            child_scores = [
                _evaluate_dsl_node(item, context=context, binary_mode=binary_mode)
                for item in node.args
            ]
            if not child_scores:
                return 0.0
            if binary_mode:
                return 1.0 if all(score >= 0.999 for score in child_scores) else 0.0
            return float(sum(child_scores) / len(child_scores))
        if fn_name == "OR":
            child_scores = [
                _evaluate_dsl_node(item, context=context, binary_mode=binary_mode)
                for item in node.args
            ]
            if not child_scores:
                return 0.0
            value = max(child_scores)
            return 1.0 if binary_mode and value >= 0.999 else (0.0 if binary_mode else value)
        if fn_name == "NOT":
            if len(node.args) != 1:
                raise RewardProgramError("NOT expects exactly one argument.")
            child = _evaluate_dsl_node(node.args[0], context=context, binary_mode=binary_mode)
            if binary_mode:
                return 1.0 if child < 0.999 else 0.0
            return 1.0 - child
        if fn_name == "EVENT":
            label = str(args[0]) if args else ""
            params = kwargs.get("params", args[1] if len(args) > 1 and isinstance(args[1], dict) else {})
            return _event_score(context, label=label, params=params, binary_mode=binary_mode)
        if fn_name == "AFTER":
            first_label = str(args[0]) if args else ""
            second_label = str(args[1]) if len(args) > 1 else ""
            return _after_score(
                context,
                first_label=first_label,
                second_label=second_label,
                first_params=kwargs.get("first_params", {}),
                second_params=kwargs.get("second_params", {}),
                binary_mode=binary_mode,
            )
        if fn_name == "COUNT":
            label = str(args[0]) if args else ""
            target = int(args[1]) if len(args) > 1 else 1
            params = kwargs.get("params", args[2] if len(args) > 2 and isinstance(args[2], dict) else {})
            return _count_score(context, label=label, target=target, params=params, binary_mode=binary_mode)
        if fn_name == "GT":
            label = str(args[0]) if args else ""
            threshold = float(args[1]) if len(args) > 1 else 0.0
            params = kwargs.get("params", args[2] if len(args) > 2 and isinstance(args[2], dict) else {})
            return _gt_score(context, label=label, threshold=threshold, params=params, binary_mode=binary_mode)
        if fn_name == "NEARBY_AT":
            ball_ref = args[0] if args else "9"
            x_value = float(args[1]) if len(args) > 1 else 0.0
            y_value = float(args[2]) if len(args) > 2 else 0.0
            t_fraction = float(args[3]) if len(args) > 3 else 1.0
            threshold_strength = float(kwargs.get("threshold_strength", 0.2))
            return _nearby_at_score(
                context,
                ball_ref=ball_ref,
                x_value=x_value,
                y_value=y_value,
                t_fraction=t_fraction,
                threshold_strength=threshold_strength,
                binary_mode=binary_mode,
            )
        if fn_name == "POTTED":
            ball_ref = args[0] if args else "9"
            score = 1.0 if _ball_potted(context.trace, str(ball_ref)) else 0.0
            return score
        if fn_name == "BALL_REMAINING":
            ball_ref = args[0] if args else "cue"
            score = 1.0 if _ball_remaining(context.trace, str(ball_ref)) else 0.0
            return score
    raise RewardProgramError("Unexpected DSL evaluation node.")


def _literal_value(node: ast.AST) -> Any:
    return ast.literal_eval(node)


def _matching_patterns(context: _RewardProgramContext, *, label: str, params: dict[str, Any] | None) -> list[Pattern]:
    if not label.strip():
        return []
    matches = list(context.patterns_by_label.get(label.strip().lower(), []))
    if not params:
        return matches
    filtered: list[Pattern] = []
    for pattern in matches:
        values = pattern.parameters if isinstance(pattern.parameters, dict) else {}
        if all(str(values.get(key)) == str(expected) for key, expected in params.items()):
            filtered.append(pattern)
    return filtered


def _event_score(context: _RewardProgramContext, *, label: str, params: dict[str, Any] | None, binary_mode: bool) -> float:
    matches = _matching_patterns(context, label=label, params=params)
    score = 1.0 if matches else 0.0
    return score


def _after_score(
    context: _RewardProgramContext,
    *,
    first_label: str,
    second_label: str,
    first_params: dict[str, Any] | None,
    second_params: dict[str, Any] | None,
    binary_mode: bool,
) -> float:
    first_matches = _matching_patterns(context, label=first_label, params=first_params)
    second_matches = _matching_patterns(context, label=second_label, params=second_params)
    if not first_matches or not second_matches:
        return 0.0
    valid = any(
        first.end_timestep <= second.start_timestep
        for first in first_matches
        for second in second_matches
    )
    if binary_mode:
        return 1.0 if valid else 0.0
    exists_score = 1.0
    ordering_score = 1.0 if valid else 0.25
    return 0.5 * exists_score + 0.5 * ordering_score


def _count_score(
    context: _RewardProgramContext,
    *,
    label: str,
    target: int,
    params: dict[str, Any] | None,
    binary_mode: bool,
) -> float:
    count = len(_matching_patterns(context, label=label, params=params))
    if binary_mode:
        return 1.0 if count == target else 0.0
    return _deviation_score(float(abs(count - target)))


def _gt_score(
    context: _RewardProgramContext,
    *,
    label: str,
    threshold: float,
    params: dict[str, Any] | None,
    binary_mode: bool,
) -> float:
    count = float(len(_matching_patterns(context, label=label, params=params)))
    if binary_mode:
        return 1.0 if count > threshold else 0.0
    if threshold <= 0:
        return 1.0
    if count > threshold:
        return 1.0
    return max(0.0, count / max(threshold, 1.0))


def _deviation_score(delta: float) -> float:
    if delta <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 / (1.0 + math.log1p(delta))))


def _nearby_at_score(
    context: _RewardProgramContext,
    *,
    ball_ref: Any,
    x_value: float,
    y_value: float,
    t_fraction: float,
    threshold_strength: float,
    binary_mode: bool,
) -> float:
    position = _ball_position_at_fraction(context.trace, str(ball_ref), t_fraction)
    if position is None:
        return 0.0
    dx = float(position[0]) - x_value
    dy = float(position[1]) - y_value
    distance = math.sqrt(dx * dx + dy * dy)
    threshold = max(0.01, threshold_strength * max(context.table_width, context.table_length))
    if binary_mode:
        return 1.0 if distance <= threshold else 0.0
    return max(0.0, min(1.0, 1.0 - math.log1p(distance / threshold) / math.log(2.0 + 1e-6)))


def _ball_position_at_fraction(trace: CanonicalTrace, ball_id: str, t_fraction: float) -> tuple[float, float] | None:
    timesteps = trace.timesteps
    if not timesteps:
        return None
    clamped_fraction = min(1.0, max(0.0, float(t_fraction)))
    index = min(len(timesteps) - 1, max(0, int(round(clamped_fraction * (len(timesteps) - 1)))))
    for obj in timesteps[index].objects:
        if str(obj.get("ball_id")) == str(ball_id):
            if isinstance(obj.get("x"), (int, float)) and isinstance(obj.get("y"), (int, float)):
                return float(obj["x"]), float(obj["y"])
            position = obj.get("position")
            if (
                isinstance(position, list)
                and len(position) >= 2
                and isinstance(position[0], (int, float))
                and isinstance(position[1], (int, float))
            ):
                return float(position[0]), float(position[1])
    return None


def _ball_potted(trace: CanonicalTrace, ball_id: str) -> bool:
    metrics = trace.outcome.metrics if isinstance(trace.outcome.metrics, dict) else {}
    potted = metrics.get("potted_balls")
    if isinstance(potted, list):
        return str(ball_id) in {str(item) for item in potted}
    for event in trace.events:
        if str(event.kind) == "shot_outcome":
            payload = event.payload if isinstance(event.payload, dict) else {}
            potted_balls = payload.get("potted_balls")
            if isinstance(potted_balls, list) and str(ball_id) in {str(item) for item in potted_balls}:
                return True
    return False


def _ball_remaining(trace: CanonicalTrace, ball_id: str) -> bool:
    if not trace.timesteps:
        return False
    for obj in trace.timesteps[-1].objects:
        if str(obj.get("ball_id")) == str(ball_id):
            return True
    return False


def _pooltool_adapter() -> PooltoolAdapter:
    adapter = AdapterRegistry().load("pooltool")
    if not isinstance(adapter, PooltoolAdapter):
        raise RewardProgramError("Failed to load PoolTool adapter from registry.")
    return adapter


def _create_random_scene(
    *,
    adapter: PooltoolAdapter,
    scene_id: str,
    seed: int,
    output_root: Path,
    rack_jitter: float,
) -> tuple[Any, SceneDefinition]:
    backend = adapter._load_backend()
    system = adapter._copy_system(adapter._create_initial_system(backend))
    adapter._randomize_initial_layout(
        system,
        seed=seed,
        rack_jitter=rack_jitter,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    image_path = output_root / "initial_scene_overhead.png"
    try:
        initial_paths, _video_paths = _render_pooltool_action_media(
            adapter=adapter,
            system=system,
            action=ContinuousPooltoolAction(speed=0.1, strikespot_a=0.0, strikespot_b=0.0),
            output_root=output_root,
        )
        overhead_initial = next((Path(path) for path in initial_paths if "overhead" in str(path).lower()), None)
        if isinstance(overhead_initial, Path) and overhead_initial.is_file():
            image_path = overhead_initial
        elif initial_paths:
            candidate = Path(initial_paths[0])
            if candidate.is_file():
                image_path = candidate
        elif not image_path.is_file():
            _render_pooltool_system_image(adapter, system, image_path)
    except Exception:
        _render_pooltool_system_image(adapter, system, image_path)
    summary = SceneDefinition(
        scene_id=scene_id,
        seed=seed,
        task_id=scene_id,
        table_width=float(getattr(system.table, "w", 1.0)),
        table_length=float(getattr(system.table, "l", 2.0)),
        initial_ball_positions=adapter._ball_objects(system),
        image_path=str(image_path),
        rack_jitter=rack_jitter,
    )
    _write_json(output_root / "scene.json", asdict(summary))
    return system, summary


def _load_scene_definition(scene_path: Path) -> SceneDefinition:
    try:
        payload = json.loads(scene_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RewardProgramError(f"Failed to read PoolTool scene: {scene_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RewardProgramError(f"Malformed PoolTool scene JSON: {scene_path}")
    return SceneDefinition(**payload)


def _build_system_from_scene_summary(
    adapter: PooltoolAdapter,
    *,
    scene_summary: SceneDefinition,
) -> Any:
    backend = adapter._load_backend()
    system = adapter._copy_system(adapter._create_initial_system(backend))
    ball_map = {str(ball_id): ball for ball_id, ball in getattr(system, "balls", {}).items()}
    for item in scene_summary.initial_ball_positions:
        if not isinstance(item, dict):
            continue
        ball_id = str(item.get("ball_id") or "").strip()
        ball = ball_map.get(ball_id)
        if ball is None:
            continue
        try:
            x_value = float(item.get("x"))
            y_value = float(item.get("y"))
        except (TypeError, ValueError):
            continue
        adapter._set_ball_xy(ball, x_value, y_value)
    adapter._ensure_cue_ball_present(backend, system)
    return system


def _render_pooltool_system_image(
    adapter: PooltoolAdapter,
    system: Any,
    output_path: Path,
    *,
    width_px: int = DEFAULT_VALUE_MODEL_IMAGE_SIZE,
) -> Path:
    table = system.table
    height_px = width_px
    margin = 24
    image = Image.new("RGB", (width_px, height_px), color=(14, 35, 27))
    draw = ImageDraw.Draw(image)
    table_left = margin
    table_top = margin
    table_right = width_px - margin
    table_bottom = height_px - margin
    draw.rounded_rectangle(
        [table_left, table_top, table_right, table_bottom],
        radius=14,
        fill=(33, 110, 63),
        outline=(153, 111, 62),
        width=8,
    )

    def map_xy(x_value: float, y_value: float) -> tuple[float, float]:
        norm_x = float(x_value) / max(float(table.w), 1e-6)
        norm_y = float(y_value) / max(float(table.l), 1e-6)
        px = table_left + norm_x * (table_right - table_left)
        py = table_bottom - norm_y * (table_bottom - table_top)
        return px, py

    pocket_radius = 10
    for pocket_x, pocket_y in (
        (table_left, table_top),
        ((table_left + table_right) / 2.0, table_top),
        (table_right, table_top),
        (table_left, table_bottom),
        ((table_left + table_right) / 2.0, table_bottom),
        (table_right, table_bottom),
    ):
        draw.ellipse(
            [pocket_x - pocket_radius, pocket_y - pocket_radius, pocket_x + pocket_radius, pocket_y + pocket_radius],
            fill=(20, 20, 20),
        )
    ball_colors = {
        "cue": (245, 245, 245),
        "1": (245, 209, 66),
        "2": (54, 97, 214),
        "3": (199, 47, 47),
        "4": (106, 61, 186),
        "5": (232, 134, 44),
        "6": (54, 140, 73),
        "7": (120, 33, 24),
        "8": (28, 28, 28),
        "9": (245, 209, 66),
    }
    outline = (240, 240, 240)
    for ball_id, ball in system.balls.items():
        x_value, y_value = adapter._ball_xy(ball)
        px, py = map_xy(x_value, y_value)
        radius_m = float(getattr(getattr(ball, "params", None), "R", 0.028575))
        radius_px = max(8, int(round(radius_m / max(float(table.w), 1e-6) * (table_right - table_left) * 1.2)))
        fill = ball_colors.get(str(ball_id), (200, 200, 200))
        draw.ellipse(
            [px - radius_px, py - radius_px, px + radius_px, py + radius_px],
            fill=fill,
            outline=outline,
            width=2,
        )
        if str(ball_id) != "cue":
            text = str(ball_id)
            bbox = draw.textbbox((0, 0), text)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            text_fill = (255, 255, 255) if fill == (28, 28, 28) else (20, 20, 20)
            draw.text((px - tw / 2, py - th / 2), text, fill=text_fill)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def _render_pooltool_action_media(
    *,
    adapter: PooltoolAdapter,
    system: Any,
    action: ContinuousPooltoolAction,
    output_root: Path,
) -> tuple[list[str], list[str]]:
    backend = adapter._load_backend()
    current_system = adapter._copy_system(system)
    cue_ball_id = current_system.cue.cue_ball_id
    adapter._ensure_cue_ball_present(backend, current_system)
    if hasattr(current_system, "stop_balls"):
        current_system.stop_balls()
    if hasattr(current_system, "reset_history"):
        current_system.reset_history()
    target_ball_id = adapter._select_target_ball(current_system, cue_ball_id=cue_ball_id)
    if target_ball_id is None:
        raise RewardProgramError("No legal target ball remained for PoolTool media rendering.")
    current_system.cue.set_state(
        V0=float(action.speed),
        phi=adapter._aim_at_ball(backend, current_system, target_ball_id),
        a=float(action.strikespot_a),
        b=float(action.strikespot_b),
    )
    simulated_system = adapter._simulate(backend, current_system)
    output_root.mkdir(parents=True, exist_ok=True)
    attachments = adapter._render_media_artifacts(
        backend,
        simulated_system,
        output_dir=output_root,
        continuous_dt=0.05,
    )
    initial_paths: list[str] = []
    video_paths: list[str] = []
    for attachment in attachments:
        path_value = str(getattr(attachment, "path", "") or "")
        if not path_value:
            continue
        kind = str(getattr(attachment, "kind", "") or "")
        if kind == "initial_scene":
            initial_paths.append(path_value)
        elif kind == "simulation_video":
            video_paths.append(path_value)
    return initial_paths, video_paths


def _pooltool_action_bounds(adapter: PooltoolAdapter) -> _ActionBounds:
    settings = adapter._load_runtime_settings()
    speeds = [float(value) for value in settings["speed_mapping"].values()]
    a_values = [float(item.get("a", 0.0)) for item in settings["strikespot_mapping"].values()]
    b_values = [float(item.get("b", 0.0)) for item in settings["strikespot_mapping"].values()]
    return _ActionBounds(
        speed_min=min(speeds),
        speed_max=max(speeds),
        a_min=min(a_values),
        a_max=max(a_values),
        b_min=min(b_values),
        b_max=max(b_values),
    )


def _sample_action(rng: random.Random, bounds: _ActionBounds) -> ContinuousPooltoolAction:
    return ContinuousPooltoolAction(
        speed=rng.uniform(bounds.speed_min, bounds.speed_max),
        strikespot_a=rng.uniform(bounds.a_min, bounds.a_max),
        strikespot_b=rng.uniform(bounds.b_min, bounds.b_max),
    )


def _perturb_action(
    action: ContinuousPooltoolAction,
    *,
    rng: random.Random,
    bounds: _ActionBounds,
    temperature: float,
) -> ContinuousPooltoolAction:
    speed_span = max(bounds.speed_max - bounds.speed_min, 1e-3)
    a_span = max(bounds.a_max - bounds.a_min, 1e-3)
    b_span = max(bounds.b_max - bounds.b_min, 1e-3)
    return ContinuousPooltoolAction(
        speed=min(bounds.speed_max, max(bounds.speed_min, action.speed + rng.gauss(0.0, 0.15 * speed_span * temperature))),
        strikespot_a=min(bounds.a_max, max(bounds.a_min, action.strikespot_a + rng.gauss(0.0, 0.20 * a_span * temperature))),
        strikespot_b=min(bounds.b_max, max(bounds.b_min, action.strikespot_b + rng.gauss(0.0, 0.20 * b_span * temperature))),
    )


def _simulate_pooltool_action(
    *,
    adapter: PooltoolAdapter,
    system: Any,
    action: ContinuousPooltoolAction,
    task_id: str,
) -> CanonicalTrace:
    backend = adapter._load_backend()
    current_system = adapter._copy_system(system)
    cue_ball_id = current_system.cue.cue_ball_id
    adapter._ensure_cue_ball_present(backend, current_system)
    if hasattr(current_system, "stop_balls"):
        current_system.stop_balls()
    if hasattr(current_system, "reset_history"):
        current_system.reset_history()
    target_ball_id = adapter._select_target_ball(current_system, cue_ball_id=cue_ball_id)
    if target_ball_id is None:
        raise RewardProgramError("No legal target ball remained for PoolTool rollout.")
    current_system.cue.set_state(
        V0=float(action.speed),
        phi=adapter._aim_at_ball(backend, current_system, target_ball_id),
        a=float(action.strikespot_a),
        b=float(action.strikespot_b),
    )
    try:
        simulated_system = adapter._simulate(backend, current_system)
        dense_timesteps = adapter._extract_dense_timesteps(
            backend,
            simulated_system,
            shot=action.to_json(),
            attempt_index=0,
            time_offset=0.0,
            continuous_dt=0.05,
        )
        dense_events = adapter._extract_dense_events(
            simulated_system,
            attempt_index=0,
            time_offset=0.0,
        )
        potted_balls = adapter._extract_potted_balls(backend, simulated_system)
        if "9" in potted_balls:
            terminal_status = "9_BALL_POTTED"
            success = True
        elif cue_ball_id in potted_balls:
            terminal_status = "CUE_BALL_POTTED"
            success = False
        elif potted_balls:
            terminal_status = "OTHER_BALLS_POTTED"
            success = False
        else:
            terminal_status = "NO_BALLS_POTTED"
            success = False
        adapter._remove_potted_balls(simulated_system, potted_balls)
    except ValueError as exc:
        terminal_status = "SIMULATION_CRASHED_ROOT_SOLVER_ISSUE"
        success = False
        dense_timesteps = []
        dense_events = [
            make_event(
                "simulation_error",
                payload={"message": str(exc)},
            )
        ]
        potted_balls = []
        simulated_system = current_system

    if not dense_timesteps:
        dense_timesteps = [
            make_step(
                0,
                t=0.0,
                state={"attempt_index": 0, "warning": terminal_status},
                objects=adapter._ball_objects(simulated_system),
                actions=[action.to_json()],
            )
        ]
    metrics = {
        "attempts": 1,
        "potted_balls": list(potted_balls),
        "terminal_status": terminal_status,
        "nine_ball_success": success,
    }
    return build_simulator_trace(
        env_id="pooltool",
        task_id=task_id,
        task_name=adapter.get_manifest().native_metric.name,
        initial_state={
            "table_width": float(getattr(simulated_system.table, "w", 1.0)),
            "table_length": float(getattr(simulated_system.table, "l", 2.0)),
            "initial_ball_positions": adapter._ball_objects(system),
        },
        actions=[action.to_json()],
        timesteps=dense_timesteps,
        events=dense_events,
        attachments=[],
        metrics=metrics,
        success=success,
        terminal_status=terminal_status,
        provenance=build_provenance(adapter),
    )


def run_pooltool_optimization_experiment(
    *,
    run_id: str,
    source_run_dir: Path,
    goal_text: str,
    dsl_program: str,
    model_name: str,
    llm_base_url: str,
    llm_timeout: int,
    scene_count: int = DEFAULT_OPTIMIZATION_SCENE_COUNT,
    seed_count: int = DEFAULT_OPTIMIZATION_SEED_COUNT,
    budgets: Sequence[int] = DEFAULT_OPTIMIZATION_BUDGETS,
    rack_jitter: float = DEFAULT_POOLTOOL_RACK_JITTER,
) -> PooltoolOptimizationExperimentResult:
    _reward_progress(
        "optimization.start",
        run_id=run_id,
        scene_count=scene_count,
        seed_count=seed_count,
        budgets=",".join(str(int(item)) for item in budgets),
    )
    run_root = _ensure_run_root(_optimization_root(run_id=run_id))
    adapter = _pooltool_adapter()
    representative_system, representative_scene = _create_random_scene(
        adapter=adapter,
        scene_id="scene-000",
        seed=0,
        output_root=run_root / "scenes" / "scene-000",
        rack_jitter=rack_jitter,
    )
    reward_program = _load_reward_program_definition(
        goal_text=goal_text,
        dsl_program=dsl_program,
        source_run_dir=source_run_dir,
        model_name=model_name,
        llm_base_url=llm_base_url,
        llm_timeout=llm_timeout,
        artifact_root=run_root,
        representative_scene=representative_scene,
    )
    (run_root / "program.dsl").write_text(reward_program.dsl_program + "\n", encoding="utf-8")
    _reward_progress(
        "optimization.dsl_ready",
        run_id=run_id,
        synthesized=bool(reward_program.synthesized),
        goal_text=reward_program.goal_text,
        dsl_program=reward_program.dsl_program,
    )
    library = load_pattern_library(source_run_dir / "pattern_library.json")
    scene_records: list[SceneDefinition] = [representative_scene]
    systems_by_scene: dict[str, Any] = {"scene-000": representative_system}
    for scene_index in range(1, scene_count):
        scene_id = f"scene-{scene_index:03d}"
        system, scene_summary = _create_random_scene(
            adapter=adapter,
            scene_id=scene_id,
            seed=scene_index,
            output_root=run_root / "scenes" / scene_id,
            rack_jitter=rack_jitter,
        )
        systems_by_scene[scene_id] = system
        scene_records.append(scene_summary)
    _reward_progress(
        "optimization.scenes_ready",
        run_id=run_id,
        scene_count=len(scene_records),
    )
    budgets = tuple(int(item) for item in budgets if int(item) > 0)
    bounds = _pooltool_action_bounds(adapter)
    outcomes: list[OptimizationBudgetOutcome] = []
    for scene_summary in scene_records:
        scene_system = systems_by_scene[scene_summary.scene_id]
        for reward_kind in ("synthesized", "binary"):
            for budget in budgets:
                _reward_progress(
                    "optimization.bucket_start",
                    run_id=run_id,
                    scene_id=scene_summary.scene_id,
                    reward_kind=reward_kind,
                    budget=budget,
                    seed_count=seed_count,
                )
                for seed in range(seed_count):
                    result = _optimize_scene_action(
                        adapter=adapter,
                        base_system=scene_system,
                        scene_summary=scene_summary,
                        reward_program=reward_program,
                        library=library,
                        budget=budget,
                        reward_kind=reward_kind,
                        seed=seed,
                        bounds=bounds,
                        run_root=run_root,
                    )
                    outcomes.append(result)
                _reward_progress(
                    "optimization.bucket_done",
                    run_id=run_id,
                    scene_id=scene_summary.scene_id,
                    reward_kind=reward_kind,
                    budget=budget,
                )
    aggregates = _aggregate_budget_outcomes(outcomes)
    summary_path = run_root / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": run_id,
            "goal_text": reward_program.goal_text,
            "dsl_program": reward_program.dsl_program,
            "source_run_dir": reward_program.source_run_dir,
            "source_run_kind": reward_program.source_run_kind,
            "source_run_id": reward_program.source_run_id,
            "model_name": reward_program.model_name,
            "scene_count": scene_count,
            "seed_count": seed_count,
            "budgets": list(budgets),
            "rack_jitter": rack_jitter,
            "scene_summaries": [asdict(item) for item in scene_records],
            "budget_aggregates": [asdict(item) for item in aggregates],
            "outcomes": [asdict(item) for item in outcomes],
            "outcome_count": len(outcomes),
            "synthesized": reward_program.synthesized,
            "synthesis_response_path": reward_program.synthesis_response_path,
        },
    )
    _reward_progress(
        "optimization.done",
        run_id=run_id,
        outcome_count=len(outcomes),
        summary_path=summary_path,
    )
    return PooltoolOptimizationExperimentResult(
        request=PooltoolOptimizationExperimentRequest(
            run_id=run_id,
            output_root=str(run_root),
            source_run_dir=str(source_run_dir),
            goal_text=goal_text,
            dsl_program=dsl_program,
            model_name=model_name,
            scene_count=scene_count,
            seed_count=seed_count,
            budgets=tuple(budgets),
            rack_jitter=rack_jitter,
        ),
        reward_program=reward_program,
        scene_summaries=scene_records,
        budget_aggregates=aggregates,
        outcomes=outcomes,
        outcome_count=len(outcomes),
        summary_path=str(summary_path),
    )


def run_pooltool_single_scene_experiment(
    *,
    run_id: str,
    source_run_dir: Path,
    scene_path: Path,
    goal_text: str,
    dsl_program: str,
    model_name: str,
    llm_base_url: str,
    llm_timeout: int,
    reward_kind: str = "synthesized",
    budget: int = 250,
    seed: int = 0,
) -> PooltoolSingleSceneExperimentResult:
    normalized_reward_kind = str(reward_kind or "synthesized").strip().lower()
    if normalized_reward_kind not in {"synthesized", "binary"}:
        raise RewardProgramError(f"Unsupported reward kind: {reward_kind}")
    _reward_progress(
        "single_scene.start",
        run_id=run_id,
        scene_path=str(scene_path),
        reward_kind=normalized_reward_kind,
        budget=int(budget),
        seed=int(seed),
    )
    run_root = _ensure_run_root(_single_scene_root(run_id=run_id))
    request = PooltoolSingleSceneExperimentRequest(
        run_id=run_id,
        output_root=str(run_root),
        source_run_dir=str(source_run_dir),
        scene_path=str(scene_path),
        goal_text=goal_text,
        dsl_program=dsl_program,
        model_name=model_name,
        reward_kind=normalized_reward_kind,
        budget=max(1, int(budget)),
        seed=max(0, int(seed)),
    )
    adapter = _pooltool_adapter()
    scene_summary = _load_scene_definition(scene_path)
    scene_root = run_root / "scene"
    scene_root.mkdir(parents=True, exist_ok=True)
    _write_json(scene_root / "scene.json", asdict(scene_summary))
    initial_image_path = Path(str(scene_summary.image_path))
    if initial_image_path.is_file():
        copied_image_path = scene_root / initial_image_path.name
        copied_image_path.write_bytes(initial_image_path.read_bytes())
        scene_summary = SceneDefinition(
            scene_id=scene_summary.scene_id,
            seed=scene_summary.seed,
            task_id=scene_summary.task_id,
            table_width=scene_summary.table_width,
            table_length=scene_summary.table_length,
            initial_ball_positions=scene_summary.initial_ball_positions,
            image_path=str(copied_image_path),
            rack_jitter=scene_summary.rack_jitter,
        )
        _write_json(scene_root / "scene.json", asdict(scene_summary))
    reward_program = _load_reward_program_definition(
        goal_text=goal_text,
        dsl_program=dsl_program,
        source_run_dir=source_run_dir,
        model_name=model_name,
        llm_base_url=llm_base_url,
        llm_timeout=llm_timeout,
        artifact_root=run_root,
        representative_scene=scene_summary,
    )
    (run_root / "program.dsl").write_text(reward_program.dsl_program + "\n", encoding="utf-8")
    _reward_progress(
        "single_scene.dsl_ready",
        run_id=run_id,
        synthesized=bool(reward_program.synthesized),
        goal_text=reward_program.goal_text,
        dsl_program=reward_program.dsl_program,
    )
    library = load_pattern_library(source_run_dir / "pattern_library.json")
    base_system = _build_system_from_scene_summary(adapter, scene_summary=scene_summary)
    outcome = _optimize_scene_action(
        adapter=adapter,
        base_system=base_system,
        scene_summary=scene_summary,
        reward_program=reward_program,
        library=library,
        budget=request.budget,
        reward_kind=normalized_reward_kind,
        seed=request.seed,
        bounds=_pooltool_action_bounds(adapter),
        run_root=run_root,
        progress_callback=(
            lambda completed, total: _reward_progress(
                "single_scene.optimization_progress",
                run_id=run_id,
                completed_steps=int(completed),
                total_steps=int(total),
                fraction=(float(completed) / float(total)) if total > 0 else 0.0,
                budget=int(request.budget),
            )
        ),
    )
    initial_scene_paths: list[str] = []
    simulation_video_paths: list[str] = []
    try:
        initial_scene_paths, simulation_video_paths = _render_pooltool_action_media(
            adapter=adapter,
            system=base_system,
            action=outcome.action,
            output_root=run_root / "rendered" / "best_shot",
        )
    except Exception as exc:  # pragma: no cover - rendering environment dependent
        _reward_progress("single_scene.render_failed", run_id=run_id, error=str(exc))
    summary_payload = {
        "run_id": run_id,
        "goal_text": reward_program.goal_text,
        "dsl_program": reward_program.dsl_program,
        "source_run_dir": reward_program.source_run_dir,
        "source_run_kind": reward_program.source_run_kind,
        "source_run_id": reward_program.source_run_id,
        "model_name": reward_program.model_name,
        "synthesized": reward_program.synthesized,
        "synthesis_response_path": reward_program.synthesis_response_path,
        "scene_path": str(scene_path),
        "scene_id": scene_summary.scene_id,
        "scene_summary": asdict(scene_summary),
        "reward_kind": normalized_reward_kind,
        "budget": request.budget,
        "seed": request.seed,
        "best_reward": float(outcome.best_reward),
        "success": bool(outcome.success),
        "action": outcome.action.to_json(),
        "trace_path": str(Path(outcome.trace_path).relative_to(run_root)),
        "trace_reward_path": str(Path(outcome.trace_reward_path).relative_to(run_root)),
        "initial_scene_paths": [
            str(Path(path).relative_to(run_root))
            for path in initial_scene_paths
            if Path(path).is_file() and str(Path(path)).startswith(str(run_root))
        ],
        "simulation_video_paths": [
            str(Path(path).relative_to(run_root))
            for path in simulation_video_paths
            if Path(path).is_file() and str(Path(path)).startswith(str(run_root))
        ],
    }
    summary_path = run_root / "summary.json"
    _write_json(summary_path, summary_payload)
    _reward_progress(
        "single_scene.done",
        run_id=run_id,
        summary_path=str(summary_path),
        reward=float(outcome.best_reward),
        success=bool(outcome.success),
    )
    return PooltoolSingleSceneExperimentResult(
        request=request,
        reward_program=reward_program,
        scene_summary=scene_summary,
        outcome=outcome,
        initial_scene_paths=summary_payload["initial_scene_paths"],
        simulation_video_paths=summary_payload["simulation_video_paths"],
        summary_path=str(summary_path),
    )


def run_pooltool_optimization_goal_suite(
    *,
    run_id: str,
    source_run_dir: Path,
    goals: Sequence[PooltoolRewardGoal] | None,
    model_name: str,
    llm_base_url: str,
    llm_timeout: int,
    scene_count: int = DEFAULT_OPTIMIZATION_SCENE_COUNT,
    seed_count: int = DEFAULT_OPTIMIZATION_SEED_COUNT,
    budgets: Sequence[int] = DEFAULT_OPTIMIZATION_BUDGETS,
    rack_jitter: float = DEFAULT_POOLTOOL_RACK_JITTER,
) -> PooltoolOptimizationSuiteResult:
    suite_root = _ensure_run_root(_optimization_suite_root(run_id=run_id))
    goal_specs = list(goals or default_pooltool_reward_goals())
    goal_results: list[PooltoolOptimizationSuiteGoalResult] = []
    _reward_progress(
        "optimization_suite.start",
        run_id=run_id,
        goal_count=len(goal_specs),
        scene_count=int(scene_count),
        seed_count=int(seed_count),
        budgets=",".join(str(int(item)) for item in budgets),
    )
    for index, goal in enumerate(goal_specs, start=1):
        goal_slug = _slugify_goal_token(goal.goal_id or goal.goal_text)
        child_run_id = f"{run_id}--goal-{index:02d}-{goal_slug}"
        _reward_progress(
            "optimization_suite.goal_start",
            run_id=run_id,
            goal_index=index,
            goal_count=len(goal_specs),
            goal_id=goal.goal_id,
            goal_text=goal.goal_text,
            child_run_id=child_run_id,
        )
        result = run_pooltool_optimization_experiment(
            run_id=child_run_id,
            source_run_dir=source_run_dir,
            goal_text=goal.goal_text,
            dsl_program="",
            model_name=model_name,
            llm_base_url=llm_base_url,
            llm_timeout=llm_timeout,
            scene_count=scene_count,
            seed_count=seed_count,
            budgets=tuple(int(item) for item in budgets),
            rack_jitter=rack_jitter,
        )
        best_binary = max(
            (row for row in result.budget_aggregates if row.reward_kind == "binary"),
            key=lambda row: (row.success_rate, row.mean_reward, row.budget),
        )
        best_dense = max(
            (row for row in result.budget_aggregates if row.reward_kind == "synthesized"),
            key=lambda row: (row.mean_reward, row.success_rate, row.budget),
        )
        goal_results.append(
            PooltoolOptimizationSuiteGoalResult(
                goal_id=goal.goal_id,
                goal_text=goal.goal_text,
                run_id=result.request.run_id,
                summary_path=result.summary_path,
                best_binary_success_rate=best_binary.success_rate,
                best_binary_mean_reward=best_binary.mean_reward,
                best_binary_budget=best_binary.budget,
                best_dense_mean_reward=best_dense.mean_reward,
                best_dense_budget=best_dense.budget,
            )
        )
        _reward_progress(
            "optimization_suite.goal_done",
            run_id=run_id,
            goal_index=index,
            goal_count=len(goal_specs),
            goal_id=goal.goal_id,
            goal_text=goal.goal_text,
            child_run_id=child_run_id,
            best_binary_success_rate=float(best_binary.success_rate),
            best_binary_budget=int(best_binary.budget),
            best_dense_mean_reward=float(best_dense.mean_reward),
            best_dense_budget=int(best_dense.budget),
        )
    summary_path = suite_root / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": run_id,
            "source_run_dir": str(source_run_dir),
            "model_name": model_name,
            "scene_count": scene_count,
            "seed_count": seed_count,
            "budgets": [int(item) for item in budgets],
            "rack_jitter": rack_jitter,
            "goals": [asdict(item) for item in goal_results],
        },
    )
    _reward_progress(
        "optimization_suite.done",
        run_id=run_id,
        goal_count=len(goal_results),
        summary_path=str(summary_path),
    )
    return PooltoolOptimizationSuiteResult(
        run_id=run_id,
        goals=goal_results,
        summary_path=str(summary_path),
    )


def _optimize_scene_action(
    *,
    adapter: PooltoolAdapter,
    base_system: Any,
    scene_summary: SceneDefinition,
    reward_program: RewardProgramDefinition,
    library: PatternLibrary,
    budget: int,
    reward_kind: str,
    seed: int,
    bounds: _ActionBounds,
    run_root: Path,
    progress_callback: Callable[[int, int], None] | None = None,
) -> OptimizationBudgetOutcome:
    rng = random.Random((seed + 1) * 10_000 + budget + hash(scene_summary.scene_id) % 1000)
    current_action = _sample_action(rng, bounds)
    current_trace = _simulate_pooltool_action(
        adapter=adapter,
        system=base_system,
        action=current_action,
        task_id=scene_summary.scene_id,
    )
    current_score = evaluate_reward_program(
        reward_program.dsl_program,
        trace=current_trace,
        library=library,
        binary_mode=reward_kind == "binary",
    )
    best_action = current_action
    best_trace = current_trace
    best_score = current_score["score"]
    current_value = current_score["score"]
    if progress_callback is not None:
        progress_callback(1, max(1, budget))
    for step in range(max(1, budget - 1)):
        temperature = max(0.05, 1.0 - (step / max(budget - 1, 1)))
        proposal = _perturb_action(
            current_action,
            rng=rng,
            bounds=bounds,
            temperature=temperature,
        )
        proposal_trace = _simulate_pooltool_action(
            adapter=adapter,
            system=base_system,
            action=proposal,
            task_id=scene_summary.scene_id,
        )
        proposal_score = evaluate_reward_program(
            reward_program.dsl_program,
            trace=proposal_trace,
            library=library,
            binary_mode=reward_kind == "binary",
        )
        delta = float(proposal_score["score"]) - float(current_value)
        accept = delta >= 0.0 or rng.random() < math.exp(delta / max(temperature, 1e-6))
        if accept:
            current_action = proposal
            current_trace = proposal_trace
            current_value = float(proposal_score["score"])
        if float(proposal_score["score"]) > float(best_score):
            best_action = proposal
            best_trace = proposal_trace
            best_score = float(proposal_score["score"])
        if progress_callback is not None:
            step_index = step + 2
            if step_index >= budget or step_index == 2 or step_index % max(10, budget // 10) == 0:
                progress_callback(step_index, max(1, budget))
    output_root = run_root / "results" / scene_summary.scene_id / reward_kind / f"budget_{budget:04d}" / f"seed_{seed:02d}"
    output_root.mkdir(parents=True, exist_ok=True)
    trace_path = output_root / "trace.json"
    trace_path.write_text(
        best_trace.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    reward_path = output_root / "reward.json"
    reward_payload = evaluate_reward_program(
        reward_program.dsl_program,
        trace=best_trace,
        library=library,
        binary_mode=reward_kind == "binary",
    )
    reward_payload["action"] = best_action.to_json()
    _write_json(reward_path, reward_payload)
    return OptimizationBudgetOutcome(
        budget=budget,
        seed=seed,
        scene_id=scene_summary.scene_id,
        reward_kind=reward_kind,
        best_reward=float(reward_payload["score"]),
        success=bool(best_trace.outcome.success),
        action=best_action,
        trace_path=str(trace_path),
        trace_reward_path=str(reward_path),
    )


def _aggregate_budget_outcomes(outcomes: Iterable[OptimizationBudgetOutcome]) -> list[BudgetAggregate]:
    groups: dict[tuple[int, str], list[OptimizationBudgetOutcome]] = {}
    for item in outcomes:
        groups.setdefault((item.budget, item.reward_kind), []).append(item)
    aggregates: list[BudgetAggregate] = []
    for (budget, reward_kind), rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        success_rate = sum(1 for row in rows if row.success) / max(len(rows), 1)
        mean_reward = sum(float(row.best_reward) for row in rows) / max(len(rows), 1)
        scene_count = len({row.scene_id for row in rows})
        seed_count = len({row.seed for row in rows})
        aggregates.append(
            BudgetAggregate(
                budget=budget,
                reward_kind=reward_kind,
                scene_count=scene_count,
                success_rate=success_rate,
                mean_reward=mean_reward,
                seed_count=seed_count,
            )
        )
    return aggregates

def run_pooltool_value_model_experiment(
    *,
    run_id: str,
    source_run_dir: Path,
    goal_text: str,
    dsl_program: str,
    model_name: str,
    llm_base_url: str,
    llm_timeout: int,
    train_scene_count: int = DEFAULT_VALUE_MODEL_TRAIN_SCENE_COUNT,
    train_actions_per_scene: int = DEFAULT_VALUE_MODEL_TRAIN_ACTIONS_PER_SCENE,
    validation_scene_count: int = DEFAULT_VALUE_MODEL_VALIDATION_SCENE_COUNT,
    validation_candidates_per_scene: int = DEFAULT_VALUE_MODEL_VALIDATION_CANDIDATES,
    epochs: int = DEFAULT_VALUE_MODEL_EPOCHS,
    batch_size: int = DEFAULT_VALUE_MODEL_BATCH_SIZE,
    learning_rate: float = DEFAULT_VALUE_MODEL_LEARNING_RATE,
    image_size: int = DEFAULT_VALUE_MODEL_IMAGE_SIZE,
    rack_jitter: float = DEFAULT_POOLTOOL_RACK_JITTER,
) -> PooltoolValueModelExperimentResult:
    _reward_progress(
        "value_model.start",
        run_id=run_id,
        train_scene_count=train_scene_count,
        validation_scene_count=validation_scene_count,
        epochs=epochs,
        batch_size=batch_size,
    )
    run_root = _ensure_run_root(_value_model_root(run_id=run_id))
    adapter = _pooltool_adapter()
    representative_system, representative_scene = _create_random_scene(
        adapter=adapter,
        scene_id="scene-000",
        seed=0,
        output_root=run_root / "scenes" / "train" / "scene-000",
        rack_jitter=rack_jitter,
    )
    del representative_system
    reward_program = _load_reward_program_definition(
        goal_text=goal_text,
        dsl_program=dsl_program,
        source_run_dir=source_run_dir,
        model_name=model_name,
        llm_base_url=llm_base_url,
        llm_timeout=llm_timeout,
        artifact_root=run_root,
        representative_scene=representative_scene,
    )
    library = load_pattern_library(source_run_dir / "pattern_library.json")
    bounds = _pooltool_action_bounds(adapter)
    train_samples = _build_reward_training_samples(
        adapter=adapter,
        library=library,
        reward_program=reward_program,
        split_name="train",
        scene_count=train_scene_count,
        actions_per_scene=train_actions_per_scene,
        run_root=run_root,
        bounds=bounds,
        rack_jitter=rack_jitter,
    )
    _reward_progress(
        "value_model.training_samples_ready",
        run_id=run_id,
        sample_count=len(train_samples),
    )
    validation_scenes = _build_validation_scenes(
        adapter=adapter,
        run_root=run_root,
        scene_count=validation_scene_count,
        rack_jitter=rack_jitter,
    )
    _reward_progress(
        "value_model.validation_scenes_ready",
        run_id=run_id,
        scene_count=len(validation_scenes),
    )
    eval_curve: list[ValueModelEvalPoint] = []
    validation_outcomes: list[ValueModelValidationOutcome] = []
    best_epoch = 0
    best_mean_reward = 0.0
    best_success_rate = 0.0
    best_model: _PooltoolValueModel | None = None
    for epoch in range(1, epochs + 1):
        subset_size = min(
            len(train_samples),
            max(batch_size, math.ceil(len(train_samples) * epoch / max(epochs, 1))),
        )
        epoch_rows = _stable_subset_rows(
            train_samples,
            subset_size=subset_size,
            seed_label=f"epoch-{epoch}",
        )
        model, train_loss = _PooltoolValueModel.fit(
            rows=epoch_rows,
            image_size=image_size,
            ridge_alpha=max(learning_rate, 1e-6),
        )
        mean_reward, success_rate, epoch_outcomes = _evaluate_value_model(
            adapter=adapter,
            model=model,
            library=library,
            reward_program=reward_program,
            validation_scenes=validation_scenes,
            bounds=bounds,
            candidate_count=validation_candidates_per_scene,
            image_size=image_size,
            run_root=run_root,
            epoch=epoch,
        )
        validation_outcomes.extend(epoch_outcomes)
        eval_point = ValueModelEvalPoint(
            epoch=epoch,
            train_loss=train_loss,
            validation_mean_reward=mean_reward,
            validation_success_rate=success_rate,
        )
        eval_curve.append(eval_point)
        _reward_progress(
            "value_model.epoch_done",
            run_id=run_id,
            epoch=epoch,
            epochs=epochs,
            train_loss=f"{train_loss:.6f}",
            validation_mean_reward=f"{mean_reward:.6f}",
            validation_success_rate=f"{success_rate:.4f}",
        )
        if (
            mean_reward > best_mean_reward
            or (
                math.isclose(mean_reward, best_mean_reward)
                and success_rate >= best_success_rate
            )
        ):
            best_epoch = epoch
            best_mean_reward = mean_reward
            best_success_rate = success_rate
            best_model = model
    checkpoint_path = run_root / "best_model.npz"
    if best_model is not None:
        best_model.save(checkpoint_path)
    summary_path = run_root / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": run_id,
            "goal_text": reward_program.goal_text,
            "dsl_program": reward_program.dsl_program,
            "source_run_dir": reward_program.source_run_dir,
            "source_run_kind": reward_program.source_run_kind,
            "source_run_id": reward_program.source_run_id,
            "model_name": reward_program.model_name,
            "train_scene_count": train_scene_count,
            "train_actions_per_scene": train_actions_per_scene,
            "validation_scene_count": validation_scene_count,
            "validation_candidates_per_scene": validation_candidates_per_scene,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "image_size": image_size,
            "rack_jitter": rack_jitter,
            "train_sample_count": len(train_samples),
            "eval_curve": [asdict(item) for item in eval_curve],
            "validation_outcomes": [asdict(item) for item in validation_outcomes],
            "best_epoch": best_epoch,
            "best_validation_mean_reward": best_mean_reward,
            "best_validation_success_rate": best_success_rate,
            "synthesized": reward_program.synthesized,
            "synthesis_response_path": reward_program.synthesis_response_path,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path.is_file() else None,
        },
    )
    (run_root / "program.dsl").write_text(reward_program.dsl_program + "\n", encoding="utf-8")
    _reward_progress(
        "value_model.done",
        run_id=run_id,
        best_epoch=best_epoch,
        best_validation_mean_reward=f"{best_mean_reward:.6f}",
        best_validation_success_rate=f"{best_success_rate:.4f}",
        summary_path=summary_path,
    )
    return PooltoolValueModelExperimentResult(
        request=PooltoolValueModelExperimentRequest(
            run_id=run_id,
            output_root=str(run_root),
            source_run_dir=str(source_run_dir),
            goal_text=goal_text,
            dsl_program=dsl_program,
            model_name=model_name,
            train_scene_count=train_scene_count,
            train_actions_per_scene=train_actions_per_scene,
            validation_scene_count=validation_scene_count,
            validation_candidates_per_scene=validation_candidates_per_scene,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            image_size=image_size,
            rack_jitter=rack_jitter,
        ),
        reward_program=reward_program,
        train_sample_count=len(train_samples),
        validation_scene_count=validation_scene_count,
        eval_curve=eval_curve,
        validation_outcomes=validation_outcomes,
        best_epoch=best_epoch,
        best_validation_mean_reward=best_mean_reward,
        best_validation_success_rate=best_success_rate,
        summary_path=str(summary_path),
    )


def run_pooltool_value_model_sweep(
    *,
    run_id: str,
    source_run_dir: Path,
    goals: Sequence[PooltoolRewardGoal] | None,
    configs: Sequence[PooltoolValueModelSweepConfig] | None,
    model_name: str,
    llm_base_url: str,
    llm_timeout: int,
    rack_jitter: float = DEFAULT_POOLTOOL_RACK_JITTER,
) -> PooltoolValueModelSweepResult:
    sweep_root = _ensure_run_root(_value_model_sweep_root(run_id=run_id))
    goal_specs = list(goals or default_pooltool_reward_goals())
    config_specs = list(configs or default_pooltool_value_model_sweep_configs())
    run_results: list[PooltoolValueModelSweepRunResult] = []
    total_runs = len(goal_specs) * len(config_specs)
    completed_runs = 0
    _reward_progress(
        "value_model_sweep.start",
        run_id=run_id,
        goal_count=len(goal_specs),
        config_count=len(config_specs),
        total_runs=total_runs,
    )
    for goal_index, goal in enumerate(goal_specs, start=1):
        goal_slug = _slugify_goal_token(goal.goal_id or goal.goal_text)
        for config in config_specs:
            child_run_id = f"{run_id}--goal-{goal_index:02d}-{goal_slug}--{_slugify_goal_token(config.config_id)}"
            _reward_progress(
                "value_model_sweep.run_start",
                run_id=run_id,
                completed_runs=completed_runs,
                total_runs=total_runs,
                goal_index=goal_index,
                goal_count=len(goal_specs),
                goal_id=goal.goal_id,
                goal_text=goal.goal_text,
                config_id=config.config_id,
                child_run_id=child_run_id,
            )
            result = run_pooltool_value_model_experiment(
                run_id=child_run_id,
                source_run_dir=source_run_dir,
                goal_text=goal.goal_text,
                dsl_program="",
                model_name=model_name,
                llm_base_url=llm_base_url,
                llm_timeout=llm_timeout,
                train_scene_count=config.train_scene_count,
                train_actions_per_scene=config.train_actions_per_scene,
                validation_scene_count=config.validation_scene_count,
                validation_candidates_per_scene=config.validation_candidates_per_scene,
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                image_size=config.image_size,
                rack_jitter=rack_jitter,
            )
            run_results.append(
                PooltoolValueModelSweepRunResult(
                    goal_id=goal.goal_id,
                    goal_text=goal.goal_text,
                    config_id=config.config_id,
                    run_id=result.request.run_id,
                    summary_path=result.summary_path,
                    best_epoch=result.best_epoch,
                    best_validation_mean_reward=result.best_validation_mean_reward,
                    best_validation_success_rate=result.best_validation_success_rate,
                )
            )
            completed_runs += 1
            _reward_progress(
                "value_model_sweep.run_done",
                run_id=run_id,
                completed_runs=completed_runs,
                total_runs=total_runs,
                goal_index=goal_index,
                goal_count=len(goal_specs),
                goal_id=goal.goal_id,
                goal_text=goal.goal_text,
                config_id=config.config_id,
                child_run_id=child_run_id,
                best_epoch=int(result.best_epoch),
                best_validation_mean_reward=float(result.best_validation_mean_reward),
                best_validation_success_rate=float(result.best_validation_success_rate),
            )
    summary_path = sweep_root / "summary.json"
    leaderboard = sorted(
        run_results,
        key=lambda item: (
            item.best_validation_mean_reward,
            item.best_validation_success_rate,
            -item.best_epoch,
        ),
        reverse=True,
    )
    by_config: list[dict[str, Any]] = []
    for config in config_specs:
        rows = [row for row in run_results if row.config_id == config.config_id]
        if not rows:
            continue
        by_config.append(
            {
                "config_id": config.config_id,
                "goal_count": len(rows),
                "mean_best_validation_reward": float(
                    sum(row.best_validation_mean_reward for row in rows) / len(rows)
                ),
                "mean_best_validation_success_rate": float(
                    sum(row.best_validation_success_rate for row in rows) / len(rows)
                ),
                "runs": [asdict(row) for row in rows],
            }
        )
    _write_json(
        summary_path,
        {
            "run_id": run_id,
            "source_run_dir": str(source_run_dir),
            "model_name": model_name,
            "rack_jitter": rack_jitter,
            "goals": [asdict(item) for item in goal_specs],
            "configs": [asdict(item) for item in config_specs],
            "runs": [asdict(item) for item in run_results],
            "leaderboard": [asdict(item) for item in leaderboard],
            "config_aggregates": by_config,
        },
    )
    _reward_progress(
        "value_model_sweep.done",
        run_id=run_id,
        total_runs=completed_runs,
        summary_path=str(summary_path),
    )
    return PooltoolValueModelSweepResult(
        run_id=run_id,
        goals=goal_specs,
        configs=config_specs,
        runs=run_results,
        summary_path=str(summary_path),
    )


def _build_reward_training_samples(
    *,
    adapter: PooltoolAdapter,
    library: PatternLibrary,
    reward_program: RewardProgramDefinition,
    split_name: str,
    scene_count: int,
    actions_per_scene: int,
    run_root: Path,
    bounds: _ActionBounds,
    rack_jitter: float,
) -> list[_SampleRecord]:
    samples: list[_SampleRecord] = []
    for scene_index in range(scene_count):
        scene_id = f"scene-{scene_index:03d}"
        scene_root = run_root / "scenes" / split_name / scene_id
        scene_path = scene_root / "scene.json"
        if scene_path.is_file():
            summary = SceneDefinition(**json.loads(scene_path.read_text(encoding="utf-8")))
            system = None
        else:
            system, summary = _create_random_scene(
                adapter=adapter,
                scene_id=scene_id,
                seed=10_000 + scene_index if split_name == "validation" else 1_000 + scene_index,
                output_root=scene_root,
                rack_jitter=rack_jitter,
            )
        rng = random.Random(summary.seed)
        for action_index in range(actions_per_scene):
            trace_root = run_root / "training_rollouts" / split_name / scene_id / f"action_{action_index:03d}"
            reward_path = trace_root / "reward.json"
            if reward_path.is_file():
                reward_payload = json.loads(reward_path.read_text(encoding="utf-8"))
                action_payload = reward_payload.get("action") or {}
                action = ContinuousPooltoolAction(
                    speed=float(action_payload.get("speed", 0.0)),
                    strikespot_a=float(action_payload.get("strikespot_a", 0.0)),
                    strikespot_b=float(action_payload.get("strikespot_b", 0.0)),
                )
                reward_score = float(reward_payload.get("score", 0.0))
                success = bool(reward_payload.get("success", False))
            else:
                if system is None:
                    system, summary = _create_random_scene(
                        adapter=adapter,
                        scene_id=scene_id,
                        seed=summary.seed,
                        output_root=scene_root,
                        rack_jitter=rack_jitter,
                    )
                action = _sample_action(rng, bounds)
                trace = _simulate_pooltool_action(
                    adapter=adapter,
                    system=system,
                    action=action,
                    task_id=summary.scene_id,
                )
                reward = evaluate_reward_program(
                    reward_program.dsl_program,
                    trace=trace,
                    library=library,
                    binary_mode=False,
                )
                trace_root.mkdir(parents=True, exist_ok=True)
                (trace_root / "trace.json").write_text(
                    trace.model_dump_json(indent=2) + "\n",
                    encoding="utf-8",
                )
                _write_json(
                    reward_path,
                    {
                        **reward,
                        "action": action.to_json(),
                    },
                )
                reward_score = float(reward["score"])
                success = bool(trace.outcome.success)
            samples.append(
                _SampleRecord(
                    image_path=Path(summary.image_path),
                    scene_summary=summary,
                    action=action,
                    reward=reward_score,
                    success=success,
                    scene_id=summary.scene_id,
                )
            )
    return samples


def _build_validation_scenes(
    *,
    adapter: PooltoolAdapter,
    run_root: Path,
    scene_count: int,
    rack_jitter: float,
) -> list[tuple[Any, SceneDefinition]]:
    scenes: list[tuple[Any, SceneDefinition]] = []
    for scene_index in range(scene_count):
        scene_id = f"scene-{scene_index:03d}"
        system, summary = _create_random_scene(
            adapter=adapter,
            scene_id=scene_id,
            seed=50_000 + scene_index,
            output_root=run_root / "validation_scenes" / scene_id,
            rack_jitter=rack_jitter,
        )
        scenes.append((system, summary))
    return scenes


def _evaluate_value_model(
    *,
    adapter: PooltoolAdapter,
    model: _PooltoolValueModel,
    library: PatternLibrary,
    reward_program: RewardProgramDefinition,
    validation_scenes: list[tuple[Any, SceneDefinition]],
    bounds: _ActionBounds,
    candidate_count: int,
    image_size: int,
    run_root: Path,
    epoch: int,
    selection_multiplier: int = DEFAULT_VALUE_MODEL_SELECTION_MULTIPLIER,
) -> tuple[float, float, list[ValueModelValidationOutcome]]:
    rewards: list[float] = []
    successes: list[float] = []
    outcomes: list[ValueModelValidationOutcome] = []
    for scene_system, scene_summary in validation_scenes:
        rng = random.Random(scene_summary.seed + epoch * 1000)
        expanded_count = max(candidate_count, candidate_count * max(1, int(selection_multiplier)))
        candidate_actions = [_sample_action(rng, bounds) for _ in range(expanded_count)]
        predicted_scores = model.predict_actions(
            image_path=Path(scene_summary.image_path),
            scene_summary=scene_summary,
            actions=candidate_actions,
        )
        ranked_indices = np.argsort(predicted_scores)[::-1]
        shortlisted_actions = [
            candidate_actions[int(index)]
            for index in ranked_indices[: max(1, min(len(candidate_actions), candidate_count))]
        ]
        best_action = _refine_actions_with_model(
            model=model,
            image_path=Path(scene_summary.image_path),
            scene_summary=scene_summary,
            candidate_actions=shortlisted_actions,
            bounds=bounds,
            rng=rng,
        )
        trace = _simulate_pooltool_action(
            adapter=adapter,
            system=scene_system,
            action=best_action,
            task_id=scene_summary.scene_id,
        )
        reward = evaluate_reward_program(
            reward_program.dsl_program,
            trace=trace,
            library=library,
            binary_mode=False,
        )
        rewards.append(float(reward["score"]))
        successes.append(1.0 if trace.outcome.success else 0.0)
        trace_root = run_root / "validation_rollouts" / f"epoch_{epoch:03d}" / scene_summary.scene_id
        trace_root.mkdir(parents=True, exist_ok=True)
        (trace_root / "trace.json").write_text(
            trace.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        _write_json(
            trace_root / "reward.json",
            {
                **reward,
                "action": best_action.to_json(),
            },
        )
        outcomes.append(
            ValueModelValidationOutcome(
                epoch=epoch,
                scene_id=scene_summary.scene_id,
                reward=float(reward["score"]),
                success=bool(trace.outcome.success),
                action=best_action,
                trace_path=str(trace_root / "trace.json"),
                trace_reward_path=str(trace_root / "reward.json"),
            )
        )
    mean_reward = sum(rewards) / max(len(rewards), 1)
    success_rate = sum(successes) / max(len(successes), 1)
    return mean_reward, success_rate, outcomes
