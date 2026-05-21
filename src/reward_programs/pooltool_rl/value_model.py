from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import write_json
from .dsl import evaluate_reward_program
from .pooltool_env import PooltoolEngine
from .schemas import ActionSpec, RewardProgram, SceneSpec
from .verifiers import evaluate_verifier


def _find_ball(scene: SceneSpec, ball_id: str) -> tuple[float, float] | None:
    for item in scene.initial_ball_positions:
        if item.ball_id == ball_id:
            return float(item.x), float(item.y)
    return None


def _extract_nearby_targets(dsl_program: str, *, limit: int = 4) -> list[tuple[str, float, float, float]]:
    targets: list[tuple[str, float, float, float]] = []
    try:
        root = ast.parse(dsl_program, mode="eval")
    except SyntaxError:
        return targets
    for node in ast.walk(root):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "NEARBY_AT":
            continue
        if len(node.args) < 4:
            continue
        try:
            ball_id = str(ast.literal_eval(node.args[0]))
            x = float(ast.literal_eval(node.args[1]))
            y = float(ast.literal_eval(node.args[2]))
            t_fraction = float(ast.literal_eval(node.args[3]))
        except Exception:
            continue
        targets.append((ball_id, x, y, t_fraction))
        if len(targets) >= limit:
            break
    return targets


def scene_action_features(scene: SceneSpec, action: ActionSpec, *, reward_program: RewardProgram | None = None) -> np.ndarray:
    ordered = sorted(scene.initial_ball_positions, key=lambda item: (item.ball_id != "cue", item.ball_id))
    coords: list[float] = []
    for item in ordered[:10]:
        coords.extend([item.x, item.y])
    while len(coords) < 20:
        coords.append(0.0)
    cue = _find_ball(scene, "cue") or (0.0, 0.0)
    green = _find_ball(scene, "6") or (0.0, 0.0)
    eight = _find_ball(scene, "8") or (0.0, 0.0)

    def pair_features(a: tuple[float, float], b: tuple[float, float]) -> list[float]:
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])
        dist = math.sqrt(dx * dx + dy * dy)
        return [dx, dy, dist]

    derived = [
        cue[0],
        cue[1],
        green[0],
        green[1],
        eight[0],
        eight[1],
        *pair_features(cue, green),
        *pair_features(green, eight),
        *pair_features(cue, eight),
    ]

    target_features: list[float] = []
    targets = _extract_nearby_targets(reward_program.dsl_program) if reward_program is not None else []
    for ball_id, target_x, target_y, t_fraction in targets[:4]:
        ball = _find_ball(scene, ball_id) or (0.0, 0.0)
        dx = float(target_x - ball[0])
        dy = float(target_y - ball[1])
        dist = math.sqrt(dx * dx + dy * dy)
        target_features.extend([ball[0], ball[1], target_x, target_y, dx, dy, dist, t_fraction])
    while len(target_features) < 32:
        target_features.append(0.0)

    features = np.array(coords + derived + target_features + [action.speed, action.strikespot_a, action.strikespot_b], dtype=np.float32)
    return features


@dataclass
class SmallValueModel:
    input_dim: int
    hidden_size: int
    rng_seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.rng_seed)
        self.w1 = rng.normal(scale=0.1, size=(self.input_dim, self.hidden_size)).astype(np.float32)
        self.b1 = np.zeros((self.hidden_size,), dtype=np.float32)
        self.w2 = rng.normal(scale=0.1, size=(self.hidden_size, 1)).astype(np.float32)
        self.b2 = np.zeros((1,), dtype=np.float32)
        self.x_mean = np.zeros((self.input_dim,), dtype=np.float32)
        self.x_std = np.ones((self.input_dim,), dtype=np.float32)

    def fit_normalization(self, x: np.ndarray) -> None:
        mean = x.mean(axis=0).astype(np.float32)
        std = x.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        self.x_mean = mean
        self.x_std = std

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_std

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_norm = self._normalize(x)
        hidden = np.maximum(0.0, x_norm @ self.w1 + self.b1)
        return (hidden @ self.w2 + self.b2).reshape(-1)

    def train(
        self,
        *,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        progress_callback: Any | None = None,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        self.fit_normalization(x)
        x_norm = self._normalize(x)
        n = x.shape[0]
        for epoch in range(epochs):
            order = np.arange(n)
            np.random.default_rng(epoch).shuffle(order)
            x_epoch = x_norm[order]
            y_epoch = y[order]
            for start in range(0, n, batch_size):
                xb = x_epoch[start : start + batch_size]
                yb = y_epoch[start : start + batch_size]
                hidden_pre = xb @ self.w1 + self.b1
                hidden = np.maximum(0.0, hidden_pre)
                pred = (hidden @ self.w2 + self.b2).reshape(-1)
                error = pred - yb
                grad_pred = (2.0 / max(1, len(xb))) * error
                grad_w2 = hidden.T @ grad_pred[:, None]
                grad_b2 = np.array([grad_pred.sum()], dtype=np.float32)
                grad_hidden = grad_pred[:, None] @ self.w2.T
                grad_hidden[hidden_pre <= 0.0] = 0.0
                grad_w1 = xb.T @ grad_hidden
                grad_b1 = grad_hidden.sum(axis=0)
                self.w2 -= learning_rate * grad_w2.astype(np.float32)
                self.b2 -= learning_rate * grad_b2.astype(np.float32)
                self.w1 -= learning_rate * grad_w1.astype(np.float32)
                self.b1 -= learning_rate * grad_b1.astype(np.float32)
            pred_all = self.predict(x)
            mse = float(np.mean((pred_all - y) ** 2))
            history.append({"epoch": float(epoch + 1), "mse": mse})
            if progress_callback is not None:
                progress_callback(
                    {
                        "epoch": epoch + 1,
                        "epochs": epochs,
                        "mse": mse,
                    }
                )
        return history

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2, x_mean=self.x_mean, x_std=self.x_std)
        return path


def build_training_examples(
    *,
    scenes: list[SceneSpec],
    reward_program: RewardProgram,
    engine: PooltoolEngine,
    actions_per_scene: int,
    seed: int,
    verifier_id: str | None = None,
    target_mode: str = "reward",
    example_budget: int | None = None,
    progress_callback: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    targets: list[float] = []
    if example_budget is not None:
        total_examples = max(0, int(example_budget))
        if total_examples == 0:
            raise ValueError("example_budget must be positive")
        base_per_scene = total_examples // max(1, len(scenes))
        remainder = total_examples % max(1, len(scenes))
        global_example_index = 0
        for scene_index, scene in enumerate(scenes):
            scene_budget = base_per_scene + (1 if scene_index < remainder else 0)
            if scene_budget <= 0:
                continue
            speed_min, speed_max, a_min, a_max, b_min, b_max = engine.action_bounds(scene)
            bounds = [
                (float(speed_min), float(speed_max)),
                (float(a_min), float(a_max)),
                (float(b_min), float(b_max)),
            ]
            mean = np.array([(lower + upper) / 2.0 for lower, upper in bounds], dtype=np.float32)
            std = np.array([max((upper - lower) / 3.0, 1e-3) for lower, upper in bounds], dtype=np.float32)
            min_std = np.array([max((upper - lower) * 0.05, 1e-3) for lower, upper in bounds], dtype=np.float32)
            rng = np.random.default_rng(seed + scene_index * 10_000)
            batch_size = min(12, max(4, scene_budget // 6 if scene_budget >= 6 else scene_budget))
            elite_count = max(2, batch_size // 4)
            scene_candidates: list[tuple[float, np.ndarray]] = []
            for local_index in range(scene_budget):
                if local_index < batch_size:
                    action = engine.sample_action(scene, seed=seed + scene_index * 10_000 + local_index)
                else:
                    values: list[float] = []
                    for dim, (lower, upper) in enumerate(bounds):
                        sampled = float(rng.normal(float(mean[dim]), float(std[dim])))
                        values.append(min(upper, max(lower, sampled)))
                    action = ActionSpec(
                        speed=values[0],
                        strikespot_a=values[1],
                        strikespot_b=values[2],
                    )
                trace = engine.simulate_action(scene, action)
                reward = evaluate_reward_program(
                    reward_program.dsl_program,
                    trace,
                    table_width=scene.table_width,
                    table_length=scene.table_length,
                )
                verifier_success = evaluate_verifier(verifier_id, trace) if verifier_id else None
                rows.append(scene_action_features(scene, action, reward_program=reward_program))
                if target_mode == "reward":
                    targets.append(float(reward.score))
                elif target_mode == "verifier":
                    if verifier_success is None:
                        raise ValueError("verifier target mode requires verifier_id")
                    targets.append(1.0 if verifier_success else 0.0)
                else:
                    raise ValueError(f"Unknown target_mode: {target_mode}")
                vector = np.array([action.speed, action.strikespot_a, action.strikespot_b], dtype=np.float32)
                scene_candidates.append((float(reward.score), vector))
                global_example_index += 1
                if len(scene_candidates) >= batch_size:
                    elites = sorted(scene_candidates, key=lambda item: item[0], reverse=True)[: min(elite_count, len(scene_candidates))]
                    elite_vectors = np.stack([item[1] for item in elites])
                    mean = elite_vectors.mean(axis=0)
                    std = np.maximum(min_std, elite_vectors.std(axis=0).astype(np.float32))
                    scene_candidates = []
                if progress_callback is not None and (
                    local_index + 1 == scene_budget or global_example_index == total_examples
                ):
                    progress_callback(
                        {
                            "scene_index": scene_index + 1,
                            "scene_count": len(scenes),
                            "scene_id": scene.sample_id,
                            "examples_built": len(rows),
                            "example_budget": total_examples,
                            "target_mode": target_mode,
                        }
                    )
    else:
        for scene_index, scene in enumerate(scenes):
            for action_index in range(actions_per_scene):
                action = engine.sample_action(scene, seed=seed + scene_index * 10_000 + action_index)
                trace = engine.simulate_action(scene, action)
                reward = evaluate_reward_program(
                    reward_program.dsl_program,
                    trace,
                    table_width=scene.table_width,
                    table_length=scene.table_length,
                )
                verifier_success = evaluate_verifier(verifier_id, trace) if verifier_id else None
                rows.append(scene_action_features(scene, action, reward_program=reward_program))
                if target_mode == "reward":
                    targets.append(float(reward.score))
                elif target_mode == "verifier":
                    if verifier_success is None:
                        raise ValueError("verifier target mode requires verifier_id")
                    targets.append(1.0 if verifier_success else 0.0)
                else:
                    raise ValueError(f"Unknown target_mode: {target_mode}")
            if progress_callback is not None:
                progress_callback(
                    {
                        "scene_index": scene_index + 1,
                        "scene_count": len(scenes),
                        "scene_id": scene.sample_id,
                        "examples_built": len(rows),
                        "target_mode": target_mode,
                    }
                )
    return np.stack(rows), np.array(targets, dtype=np.float32)


def evaluate_model(
    *,
    model: SmallValueModel,
    scenes: list[SceneSpec],
    reward_program: RewardProgram,
    engine: PooltoolEngine,
    validation_candidates: int,
    seed: int,
    verifier_id: str | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    per_scene: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(scenes):
        candidates: list[tuple[ActionSpec, float, float, bool | None]] = []
        speed_min, speed_max, a_min, a_max, b_min, b_max = engine.action_bounds(scene)
        bounds = [
            (float(speed_min), float(speed_max)),
            (float(a_min), float(a_max)),
            (float(b_min), float(b_max)),
        ]
        rng = np.random.default_rng(seed + scene_index * 10_000)
        batch_size = min(8, max(4, validation_candidates // 6 if validation_candidates >= 6 else validation_candidates))
        elite_count = max(2, batch_size // 4)
        mean = np.array([(lower + upper) / 2.0 for lower, upper in bounds], dtype=np.float32)
        std = np.array([max((upper - lower) / 3.0, 1e-3) for lower, upper in bounds], dtype=np.float32)
        min_std = np.array([max((upper - lower) * 0.05, 1e-3) for lower, upper in bounds], dtype=np.float32)
        for action_index in range(validation_candidates):
            if action_index < batch_size:
                action = engine.sample_action(scene, seed=seed + scene_index * 10_000 + action_index)
            else:
                action_values: list[float] = []
                for dim, (lower, upper) in enumerate(bounds):
                    sampled = float(rng.normal(float(mean[dim]), float(std[dim])))
                    action_values.append(min(upper, max(lower, sampled)))
                action = ActionSpec(
                    speed=action_values[0],
                    strikespot_a=action_values[1],
                    strikespot_b=action_values[2],
                )
            predicted = float(model.predict(scene_action_features(scene, action, reward_program=reward_program)[None, :])[0])
            trace = engine.simulate_action(scene, action)
            reward = evaluate_reward_program(
                reward_program.dsl_program,
                trace,
                table_width=scene.table_width,
                table_length=scene.table_length,
            )
            verifier_success = evaluate_verifier(verifier_id, trace) if verifier_id else None
            candidates.append((action, predicted, float(reward.score), verifier_success))
            if len(candidates) >= batch_size:
                elites = sorted(candidates, key=lambda item: item[1], reverse=True)[: min(elite_count, len(candidates))]
                elite_vectors = np.array(
                    [[item[0].speed, item[0].strikespot_a, item[0].strikespot_b] for item in elites],
                    dtype=np.float32,
                )
                if elite_vectors.size:
                    mean = elite_vectors.mean(axis=0)
                    elite_std = elite_vectors.std(axis=0)
                    std = np.maximum(min_std, elite_std.astype(np.float32))
        candidates.sort(key=lambda item: item[1], reverse=True)
        chosen = candidates[0]
        actual_best = max(candidates, key=lambda item: item[2])
        per_scene.append(
            {
                "scene_id": scene.sample_id,
                "chosen_action": chosen[0],
                "chosen_predicted_score": chosen[1],
                "chosen_actual_reward": chosen[2],
                "chosen_verifier_success": chosen[3],
                "oracle_actual_reward": actual_best[2],
                "oracle_verifier_success": actual_best[3],
                "regret": actual_best[2] - chosen[2],
            }
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "scene_index": scene_index + 1,
                    "scene_count": len(scenes),
                    "scene_id": scene.sample_id,
                    "validation_candidates": validation_candidates,
                }
            )
    avg_reward = float(np.mean([item["chosen_actual_reward"] for item in per_scene])) if per_scene else 0.0
    avg_oracle = float(np.mean([item["oracle_actual_reward"] for item in per_scene])) if per_scene else 0.0
    avg_regret = float(np.mean([item["regret"] for item in per_scene])) if per_scene else 0.0
    verifier_rows = [item for item in per_scene if item["chosen_verifier_success"] is not None]
    avg_chosen_verifier_success = float(np.mean([1.0 if item["chosen_verifier_success"] else 0.0 for item in verifier_rows])) if verifier_rows else None
    avg_oracle_verifier_success = float(np.mean([1.0 if item["oracle_verifier_success"] else 0.0 for item in verifier_rows])) if verifier_rows else None
    return {
        "scene_count": len(per_scene),
        "avg_chosen_reward": avg_reward,
        "avg_oracle_reward": avg_oracle,
        "avg_regret": avg_regret,
        "avg_chosen_verifier_success": avg_chosen_verifier_success,
        "avg_oracle_verifier_success": avg_oracle_verifier_success,
        "per_scene": per_scene,
    }


def write_training_summary(path: Path, *, train_history: list[dict[str, float]], evaluation: dict[str, Any]) -> Path:
    return write_json(path, {"train_history": train_history, "evaluation": evaluation})
