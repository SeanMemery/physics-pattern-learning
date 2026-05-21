from __future__ import annotations

import importlib
import importlib.util
import os
import random
from pathlib import Path
from shutil import rmtree
from typing import Any

from adapters.base import EnvironmentAdapter
from media import write_image_artifact, write_video_artifact
from schema import CanonicalTrace, TaskSpec
from traces import (
    build_provenance,
    build_simulator_trace,
    coerce_jsonable,
    make_attachment,
    make_event,
    make_step,
)


class IPHYREAdapter(EnvironmentAdapter):
    DEFAULT_MAX_TIME = 15.0

    def get_task_spec(self) -> TaskSpec:
        return self.build_task_spec()

    def list_tasks(self, **kwargs: object) -> list[str]:
        return list(self._load_games())

    def generate_trace(self, task_id: str, **kwargs: object) -> CanonicalTrace:
        requested_actions = kwargs.get("actions")
        settings = self._load_runtime_settings()
        simulation_fps = int(kwargs.get("simulation_fps", settings["simulation_fps"]))
        trace_fps = int(kwargs.get("trace_fps", settings["trace_fps"]))
        video_fps = int(kwargs.get("video_fps", kwargs.get("fps", settings["video_fps"])))
        solved_threshold = float(kwargs.get("solved_threshold", settings["solved_threshold"]))
        capture_visuals = bool(kwargs.get("capture_visuals", False))
        output_dir = kwargs.get("output_dir")
        max_time = float(kwargs.get("max_time", self.DEFAULT_MAX_TIME))
        sampling_strategy = str(kwargs.get("sampling_strategy", "explicit_actions"))
        sample_seed = int(kwargs.get("sample_seed", 0))
        solve_search_attempts = int(kwargs.get("solve_search_attempts", 256))
        random_action_count_min = int(kwargs.get("random_action_count_min", 1))
        random_action_count_max = int(kwargs.get("random_action_count_max", 3))
        self._configure_runtime_environment(capture_visuals=capture_visuals)

        games = self.list_tasks()
        if task_id not in games:
            raise KeyError(f"Unknown I-PHYRE task id: {task_id}")

        sampling_details: dict[str, Any]
        if requested_actions is None:
            requested_actions, sampling_details = self._sample_requested_actions(
                task_id=task_id,
                simulation_fps=simulation_fps,
                solved_threshold=solved_threshold,
                max_time=max_time,
                sampling_strategy=sampling_strategy,
                sample_seed=sample_seed,
                solve_search_attempts=solve_search_attempts,
                random_action_count_min=random_action_count_min,
                random_action_count_max=random_action_count_max,
            )
        else:
            sampling_details = {
                "sampling_strategy": "explicit_actions",
                "search_attempts": 0,
            }

        env = self._create_env(task_id, fps=simulation_fps)
        translated_actions = self._translate_actions(requested_actions, env, fps=simulation_fps)
        initial_state = self._reset_env(env)
        state_capture_interval = max(1, round(simulation_fps / max(1, trace_fps)))

        attachments = self._capture_initial_visuals(
            env,
            task_id=task_id,
            capture_visuals=capture_visuals,
            output_dir=output_dir,
        )
        timesteps = []
        events = [
            make_event(
                "scheduled_action",
                t=float(action["time"]),
                payload={"index": action["index"]},
            )
            for action in self._normalize_requested_actions(requested_actions)
        ]

        time_count = 0.0
        cursor = 0
        total_reward = 0.0
        done = False
        step_index = 0

        while time_count < max_time:
            executed_action: dict[str, Any]
            if cursor < len(translated_actions) and time_count >= translated_actions[cursor][2]:
                x, y, action_time = translated_actions[cursor]
                state, reward, done = self._step_env(env, [x, y])
                executed_action = {
                    "kind": "eliminate_block",
                    "position": [x, y],
                    "time": action_time,
                }
                cursor += 1
            else:
                state, reward, done = self._step_env(env, [0.0, 0.0])
                executed_action = {"kind": "noop"}

            total_reward += float(reward)
            objects = self._properties_to_objects(state)
            should_record_step = (
                executed_action["kind"] != "noop"
                or step_index % state_capture_interval == 0
                or done
            )
            if should_record_step:
                timesteps.append(
                    make_step(
                        index=len(timesteps),
                        t=round(time_count, 6),
                        state={
                            "reward": float(reward),
                            "done": bool(done),
                            "object_count": len(objects),
                        },
                        objects=objects,
                        actions=[executed_action],
                    )
                )
            time_count += 1 / simulation_fps
            step_index += 1
            if done:
                break

        attachments.extend(
            self._capture_sequence_visuals(
                env,
                task_id=task_id,
                capture_visuals=capture_visuals,
                output_dir=output_dir,
                translated_actions=translated_actions,
                fps=video_fps,
            )
        )
        attachments.extend(
            self._write_visual_artifacts(
                task_id=task_id,
                capture_visuals=capture_visuals,
                output_dir=output_dir,
                fps=video_fps,
            )
        )
        success = total_reward >= solved_threshold
        terminal_status = "solved_threshold_reached" if success else "threshold_not_reached"

        return build_simulator_trace(
            env_id=self.env_id,
            task_id=task_id,
            task_name=self.get_manifest().native_metric.name,
            initial_state={
                "game": task_id,
                "simulation_fps": simulation_fps,
                "trace_fps": trace_fps,
                "video_fps": video_fps,
                "solved_threshold": solved_threshold,
                "initial_object_count": len(self._properties_to_objects(initial_state)),
                **sampling_details,
            },
            actions=[
                {
                    "requested": self._normalize_requested_actions(requested_actions),
                    "translated": translated_actions,
                }
            ],
            timesteps=timesteps,
            events=events,
            attachments=attachments,
            metrics={
                "total_reward": round(total_reward, 6),
                "solved_threshold": solved_threshold,
                "translated_actions": translated_actions,
                "simulation_fps": simulation_fps,
                "trace_fps": trace_fps,
                "video_fps": video_fps,
                **sampling_details,
            },
            success=success,
            terminal_status=terminal_status,
            provenance=build_provenance(
                self,
                notes=["requires patched IPHYRE simulator for full DeepPHY parity"],
            ),
        )

    def evaluate_trace(self, trace: CanonicalTrace) -> dict[str, object]:
        metrics = dict(trace.outcome.metrics)
        metrics["success"] = bool(trace.outcome.success)
        metrics["terminal_status"] = trace.outcome.terminal_status
        return metrics

    def _load_runtime_settings(self) -> dict[str, Any]:
        config = self.load_upstream_config()
        video_fps = int(config.get("fps", 10))
        return {
            "simulation_fps": max(60, video_fps),
            "trace_fps": max(20, video_fps * 2),
            "video_fps": video_fps,
            "solved_threshold": float(config.get("solved_threshold", 900)),
        }

    def _load_games(self) -> list[str]:
        games_module = importlib.import_module("iphyre.games")
        return list(games_module.GAMES)

    def _create_env(self, task_id: str, *, fps: int) -> Any:
        simulator_module = self._load_patched_simulator_module()
        return simulator_module.IPHYRE(task_id, fps=fps)

    def _configure_runtime_environment(self, *, capture_visuals: bool) -> None:
        if capture_visuals:
            return
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    def _load_patched_simulator_module(self) -> Any:
        simulator_path = self.resolve_project_path("src/vendor/iphyre/simulator.py")
        spec = importlib.util.spec_from_file_location("iphyre_simulator_module", simulator_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load patched simulator from {simulator_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _reset_env(self, env: Any) -> Any:
        try:
            return env.reset(use_images=False)
        except TypeError:
            return env.reset()

    def _step_env(self, env: Any, position: list[float]) -> tuple[Any, float, bool]:
        try:
            state, reward, done = env.step(position, use_images=False)
        except TypeError:
            state, reward, done = env.step(position)
        return state, float(reward), bool(done)

    def _normalize_requested_actions(self, requested_actions: object) -> list[dict[str, float | int]]:
        if requested_actions is None:
            return []

        raw_actions: list[object]
        if isinstance(requested_actions, list):
            raw_actions = list(requested_actions)
        elif (
            isinstance(requested_actions, tuple)
            and len(requested_actions) == 2
            and not isinstance(requested_actions[0], (dict, list, tuple))
        ):
            raw_actions = [requested_actions]
        elif isinstance(requested_actions, tuple):
            raw_actions = list(requested_actions)
        else:
            raw_actions = [requested_actions]

        normalized: list[dict[str, float | int]] = []
        for position, action in enumerate(raw_actions):
            try:
                if isinstance(action, dict):
                    index = self._get_action_value(
                        action,
                        keys=("index", "block_index", "object_index", "idx", "action_index"),
                    )
                    time_value = self._get_action_value(
                        action,
                        keys=("time", "t", "timestamp", "action_time", "at"),
                    )
                elif isinstance(action, (list, tuple)) and len(action) >= 2:
                    index = action[0]
                    time_value = action[1]
                else:
                    raise ValueError("expected dict with index/time keys or sequence [index, time]")
                normalized.append({"index": int(index), "time": float(time_value)})
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid I-PHYRE action at position {position}: {action!r} ({exc})"
                ) from exc
        return normalized

    @staticmethod
    def _get_action_value(action: dict[str, object], *, keys: tuple[str, ...]) -> object:
        for key in keys:
            if key in action:
                return action[key]
        raise ValueError(f"missing required key; expected one of {keys}")

    def _translate_actions(self, requested_actions: object, env: Any, *, fps: int) -> list[list[float]]:
        translated = []
        action_space = list(env.get_action_space()[1:])
        requested = self._normalize_requested_actions(requested_actions)

        for action in requested:
            index = action["index"]
            if not action_space:
                break
            if index >= len(action_space):
                index = len(action_space) - 1
            x, y = action_space[index]
            translated.append([float(x), float(y), float(action["time"])])

        translated.sort(key=lambda item: item[2])
        for action in translated:
            if action[2] <= 0:
                action[2] += 1 / fps
        return translated

    def _sample_requested_actions(
        self,
        *,
        task_id: str,
        simulation_fps: int,
        solved_threshold: float,
        max_time: float,
        sampling_strategy: str,
        sample_seed: int,
        solve_search_attempts: int,
        random_action_count_min: int,
        random_action_count_max: int,
    ) -> tuple[list[dict[str, float | int]], dict[str, Any]]:
        self._configure_runtime_environment(capture_visuals=False)
        probe_env = self._create_env(task_id, fps=simulation_fps)
        action_space = list(probe_env.get_action_space()[1:])
        if not action_space:
            raise RuntimeError(f"I-PHYRE task {task_id} does not expose eliminable actions")

        rng = random.Random(sample_seed)
        if sampling_strategy == "random":
            actions = self._sample_random_requested_actions(
                action_space=action_space,
                rng=rng,
                max_time=max_time,
                random_action_count_min=random_action_count_min,
                random_action_count_max=random_action_count_max,
            )
            return actions, {"sampling_strategy": sampling_strategy, "search_attempts": 1}

        if sampling_strategy != "seek_solution":
            raise ValueError(f"Unsupported I-PHYRE sampling strategy: {sampling_strategy}")

        for attempt in range(1, solve_search_attempts + 1):
            actions = self._sample_random_requested_actions(
                action_space=action_space,
                rng=rng,
                max_time=max_time,
                random_action_count_min=random_action_count_min,
                random_action_count_max=random_action_count_max,
            )
            total_reward, success = self._evaluate_requested_actions(
                task_id=task_id,
                requested_actions=actions,
                simulation_fps=simulation_fps,
                solved_threshold=solved_threshold,
                max_time=max_time,
            )
            if success:
                return actions, {
                    "sampling_strategy": sampling_strategy,
                    "search_attempts": attempt,
                    "search_reward": round(total_reward, 6),
                }

        raise RuntimeError(
            f"Failed to sample a solved I-PHYRE action schedule after {solve_search_attempts} attempts"
        )

    def _sample_random_requested_actions(
        self,
        *,
        action_space: list[Any],
        rng: random.Random,
        max_time: float,
        random_action_count_min: int,
        random_action_count_max: int,
    ) -> list[dict[str, float | int]]:
        upper_index = len(action_space) - 1
        action_count = rng.randint(
            max(1, random_action_count_min),
            max(max(1, random_action_count_min), random_action_count_max),
        )
        latest_time = max(0.35, max_time - 0.35)
        actions: list[dict[str, float | int]] = []
        for _ in range(action_count):
            actions.append(
                {
                    "index": rng.randint(0, upper_index),
                    "time": round(rng.uniform(0.05, latest_time), 3),
                }
            )
        actions.sort(key=lambda item: float(item["time"]))
        return actions

    def _evaluate_requested_actions(
        self,
        *,
        task_id: str,
        requested_actions: list[dict[str, float | int]],
        simulation_fps: int,
        solved_threshold: float,
        max_time: float,
    ) -> tuple[float, bool]:
        self._configure_runtime_environment(capture_visuals=False)
        env = self._create_env(task_id, fps=simulation_fps)
        translated_actions = self._translate_actions(requested_actions, env, fps=simulation_fps)
        self._reset_env(env)

        time_count = 0.0
        cursor = 0
        total_reward = 0.0
        done = False

        while time_count < max_time:
            if cursor < len(translated_actions) and time_count >= translated_actions[cursor][2]:
                x, y, _action_time = translated_actions[cursor]
                _state, reward, done = self._step_env(env, [x, y])
                cursor += 1
            else:
                _state, reward, done = self._step_env(env, [0.0, 0.0])
            total_reward += float(reward)
            time_count += 1 / simulation_fps
            if done:
                break

        return total_reward, total_reward >= solved_threshold

    def _properties_to_objects(self, state: Any) -> list[dict[str, Any]]:
        rows = coerce_jsonable(state) or []
        objects = []
        for index, row in enumerate(rows):
            if not isinstance(row, list) or not row or not any(row):
                continue
            objects.append(
                {
                    "index": index,
                    "x1": row[0],
                    "y1": row[1],
                    "x2": row[2],
                    "y2": row[3],
                    "radius": row[4],
                    "eliminable": row[5],
                    "dynamic": row[6],
                    "joint": row[7],
                    "spring": row[8],
                }
            )
        return objects

    def _capture_initial_visuals(
        self,
        env: Any,
        *,
        task_id: str,
        capture_visuals: bool,
        output_dir: object,
    ) -> list:
        if not capture_visuals or output_dir is None or not hasattr(env, "collect_initial_data"):
            return []
        initial_dir = Path(str(output_dir)) / f"{task_id}_initial"
        initial_dir.mkdir(parents=True, exist_ok=True)
        env.collect_initial_data(save_path=str(initial_dir))
        return []

    def _capture_sequence_visuals(
        self,
        env: Any,
        *,
        task_id: str,
        capture_visuals: bool,
        output_dir: object,
        translated_actions: list[list[float]],
        fps: int,
    ) -> list:
        if not capture_visuals or output_dir is None or not hasattr(env, "collect_seq_data"):
            return []
        sequence_dir = Path(str(output_dir)) / f"{task_id}_sequence"
        sequence_dir.mkdir(parents=True, exist_ok=True)
        env.collect_seq_data(save_path=str(sequence_dir), act_lists=[translated_actions], fps=fps)
        return []

    def _attachments_from_directory(self, root: Path, *, kind: str) -> list:
        attachments = []
        for index, path in enumerate(sorted(root.rglob("*.png"))):
            attachments.append(make_attachment(kind, path, label=f"{kind} {index}"))
        return attachments

    def _write_visual_artifacts(
        self,
        *,
        task_id: str,
        capture_visuals: bool,
        output_dir: object,
        fps: int,
    ) -> list:
        if not capture_visuals or output_dir is None:
            return []

        root = Path(str(output_dir))
        initial_frames = sorted((root / f"{task_id}_initial").rglob("*.png"))
        sequence_frames = sorted((root / f"{task_id}_sequence").rglob("*.png"))
        attachments = []

        if initial_frames:
            initial_scene_path = root / "initial_scene.png"
            metadata = write_image_artifact(initial_frames[0], initial_scene_path, max_side=512)
            attachments.append(
                make_attachment("initial_scene", initial_scene_path, label="Initial scene", metadata=metadata)
            )

        video_frames = sequence_frames or initial_frames
        if video_frames:
            video_path = root / "simulation.mp4"
            metadata = write_video_artifact(video_frames, video_path, fps=fps, max_side=512)
            attachments.append(
                make_attachment("simulation_video", video_path, label="Simulation video", metadata=metadata)
            )

        for temp_dir in [root / f"{task_id}_initial", root / f"{task_id}_sequence"]:
            if temp_dir.exists():
                rmtree(temp_dir)

        return attachments
