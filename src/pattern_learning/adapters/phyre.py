from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from adapters.base import EnvironmentAdapter
from media import write_video_artifact
from PIL import Image
from phyre_video import PHYRE_SIMULATION_FPS, phyre_video_fps
from schema import CanonicalTrace, TaskSpec
from traces import (
    build_provenance,
    build_simulator_trace,
    make_attachment,
    make_event,
    make_step,
    summarize_shape,
)


class PhyreAdapter(EnvironmentAdapter):
    DEFAULT_EVAL_SETUP = "ball_within_template"
    DEFAULT_FOLD_ID = 0
    DEFAULT_EVAL_TYPE = "test"
    DEFAULT_SIMULATION_FPS = PHYRE_SIMULATION_FPS
    OUTPUT_VISUAL_SIZE = 1024

    def get_task_spec(self) -> TaskSpec:
        return self.build_task_spec()

    def list_tasks(
        self,
        *,
        eval_setup: str = DEFAULT_EVAL_SETUP,
        fold_id: int = DEFAULT_FOLD_ID,
        eval_type: str = DEFAULT_EVAL_TYPE,
        **kwargs: object,
    ) -> list[str]:
        context = self._load_simulator_context(
            eval_setup=eval_setup,
            fold_id=fold_id,
            eval_type=eval_type,
        )
        return list(context["task_ids"])

    def generate_trace(
        self,
        task_id: str,
        **kwargs: object,
    ) -> CanonicalTrace:
        eval_setup = str(kwargs.get("eval_setup", self.DEFAULT_EVAL_SETUP))
        fold_id = int(kwargs.get("fold_id", self.DEFAULT_FOLD_ID))
        eval_type = str(kwargs.get("eval_type", self.DEFAULT_EVAL_TYPE))
        capture_visuals = bool(kwargs.get("capture_visuals", False))
        output_dir = kwargs.get("output_dir")
        need_images = bool(kwargs.get("need_images", False)) or capture_visuals
        simulation_fps = self.DEFAULT_SIMULATION_FPS
        stride = int(kwargs.get("stride", 1))
        min_dense_frames = int(kwargs.get("min_dense_frames", 64))
        max_sample_attempts = int(kwargs.get("max_sample_attempts", 8))
        sample_seed = int(kwargs.get("sample_seed", 0))

        context = self._load_simulator_context(
            eval_setup=eval_setup,
            fold_id=fold_id,
            eval_type=eval_type,
        )
        task_ids = list(context["task_ids"])
        if task_id not in task_ids:
            raise KeyError(f"Unknown PHYRE task id: {task_id}")

        task_index = task_ids.index(task_id)
        simulator = context["simulator"]
        if "action" in kwargs:
            action = kwargs["action"]
            simulation = self._simulate_action(
                simulator,
                task_index=task_index,
                action=action,
                need_images=need_images,
                stride=stride,
            )
        else:
            action, simulation = self._sample_informative_action(
                simulator,
                task_index=task_index,
                need_images=need_images,
                stride=stride,
                sample_seed=sample_seed,
                min_dense_frames=min_dense_frames,
                max_sample_attempts=max_sample_attempts,
            )
        status = simulation.status
        status_name = getattr(status, "name", str(status))
        success = bool(status.is_solved()) if hasattr(status, "is_solved") else False
        images_value = getattr(simulation, "images", None)
        images = list(images_value) if images_value is not None else []
        initial_scene = None
        if task_index < len(context["initial_scenes"]):
            initial_scene = context["initial_scenes"][task_index]

        attachments = []
        timesteps = []
        events = [
            make_event(
                "simulation_status",
                t=0.0,
                payload={"status_name": status_name, "is_solved": success},
            )
        ]
        action_payload = {"placement": self._normalize_action(action)}

        if initial_scene is not None:
            if capture_visuals and output_dir is not None:
                initial_scene_path = Path(str(output_dir)) / "initial_scene.png"
                initial_scene_path.parent.mkdir(parents=True, exist_ok=True)
                phyre_module = self._load_phyre_module()
                rgb = phyre_module.observations_to_uint8_rgb(initial_scene)
                self._resize_rgb_frame(rgb).save(initial_scene_path)
                attachments.append(
                    make_attachment(
                        "initial_scene",
                        initial_scene_path,
                        label="Initial scene",
                        metadata=summarize_shape(initial_scene),
                    )
                )
            else:
                attachments.append(
                    make_attachment(
                        "initial_scene",
                        f"memory://phyre/{task_id}/initial_scene.png",
                        label="Initial scene",
                        metadata=summarize_shape(initial_scene),
                    )
                )

        featurized_objects = getattr(simulation, "featurized_objects", None)
        dense_timesteps = self._build_dense_timesteps(
            featurized_objects,
            action_payload,
            simulation_fps=simulation_fps,
        )
        if dense_timesteps:
            timesteps.extend(dense_timesteps)
            if timesteps:
                events.append(
                    make_event(
                        "motion_sequence",
                        t=float(timesteps[-1].t or 0.0),
                        payload={"frame_count": len(timesteps), "object_count": len(timesteps[0].objects)},
                    )
                )
        elif images:
            for index, image in enumerate(images[:5]):
                attachment = make_attachment(
                    "rendered_frame",
                    f"memory://phyre/{task_id}/frame_{index:03d}.png",
                    label=f"Frame {index}",
                    metadata=summarize_shape(image),
                )
                attachments.append(attachment)
                timesteps.append(
                    make_step(
                        index=index,
                        t=float(index),
                        state={"frame_index": index, **summarize_shape(image)},
                        actions=[action_payload] if index == 0 else [],
                        attachments=[attachment],
                    )
                )
        if capture_visuals and output_dir is not None and images:
            output_root = Path(str(output_dir))
            output_root.mkdir(parents=True, exist_ok=True)
            phyre_module = self._load_phyre_module()
            effective_frame_count = self._simulation_frame_count(simulation)
            sampled_images = self._resample_images(
                images,
                target_frame_count=max(2, effective_frame_count),
            )
            video_frames = [
                self._resize_rgb_frame(phyre_module.observations_to_uint8_rgb(image))
                for image in sampled_images
            ]
            video_fps = phyre_video_fps(
                len(video_frames),
                simulation_frame_count=effective_frame_count,
                simulation_fps=simulation_fps,
                success=success,
            )
            video_path = output_root / "simulation.mp4"
            video_metadata = write_video_artifact(
                video_frames,
                video_path,
                fps=video_fps,
                max_side=self.OUTPUT_VISUAL_SIZE,
            )
            video_metadata["duration_seconds"] = round(
                video_metadata["frame_count"] / max(video_metadata["fps"], 1),
                3,
            )
            attachments.append(
                make_attachment(
                    "simulation_video",
                    video_path,
                    label="Simulation video",
                    metadata=video_metadata,
                )
            )
        else:
            timesteps.append(
                make_step(
                    index=0,
                    t=0.0,
                    state={"simulation_status": status_name},
                    actions=[action_payload],
                )
            )

        return build_simulator_trace(
            env_id=self.env_id,
            task_id=task_id,
            task_name=self.get_manifest().native_metric.name,
            initial_state={
                "eval_setup": eval_setup,
                "fold_id": fold_id,
                "eval_type": eval_type,
                "task_index": task_index,
                "action_tier": context["action_tier"],
                "initial_scene": summarize_shape(initial_scene),
                "simulation_fps": simulation_fps,
                "video_fps": video_fps if capture_visuals and output_dir is not None and images else simulation_fps,
                "stride": stride,
                "min_dense_frames": min_dense_frames,
            },
            actions=[action_payload],
            timesteps=timesteps,
            events=events,
            attachments=attachments,
            metrics={
                "status_name": status_name,
                "is_solved": success,
                "action_tier": context["action_tier"],
                "frame_count": len(timesteps),
            },
            success=success,
            terminal_status=status_name,
            provenance=build_provenance(
                self,
                notes=[
                    f"eval_setup={eval_setup}",
                    f"fold_id={fold_id}",
                    f"eval_type={eval_type}",
                ],
            ),
        )

    def evaluate_trace(self, trace: CanonicalTrace) -> dict[str, object]:
        return {
            "task_name": trace.outcome.task_name,
            "success": bool(trace.outcome.success),
            "terminal_status": trace.outcome.terminal_status,
            **trace.outcome.metrics,
        }

    def _load_simulator_context(
        self,
        *,
        eval_setup: str,
        fold_id: int,
        eval_type: str,
    ) -> dict[str, Any]:
        phyre_module = self._load_phyre_module()
        train_tasks, _, test_tasks = phyre_module.get_fold(eval_setup, fold_id)
        selected_tasks = list(train_tasks if eval_type == "train" else test_tasks)
        action_tier = phyre_module.eval_setup_to_action_tier(eval_setup)
        simulator = phyre_module.initialize_simulator(selected_tasks, action_tier)
        return {
            "task_ids": list(getattr(simulator, "task_ids", selected_tasks)),
            "simulator": simulator,
            "action_tier": action_tier,
            "initial_scenes": list(getattr(simulator, "initial_scenes", [])),
        }

    def _resize_rgb_frame(self, pixels: Any) -> Image.Image:
        image = Image.fromarray(pixels).convert("RGB")
        if image.size == (self.OUTPUT_VISUAL_SIZE, self.OUTPUT_VISUAL_SIZE):
            return image
        return image.resize(
            (self.OUTPUT_VISUAL_SIZE, self.OUTPUT_VISUAL_SIZE),
            Image.Resampling.LANCZOS,
        )

    def _load_phyre_module(self) -> Any:
        return importlib.import_module("phyre")

    def _simulate_action(
        self,
        simulator: Any,
        *,
        task_index: int,
        action: object,
        need_images: bool,
        stride: int,
    ) -> Any:
        try:
            return simulator.simulate_action(
                task_index,
                action,
                need_images=need_images,
                need_featurized_objects=True,
                stride=stride,
            )
        except TypeError:
            return simulator.simulate_action(task_index, action, need_images=need_images)

    def _sample_informative_action(
        self,
        simulator: Any,
        *,
        task_index: int,
        need_images: bool,
        stride: int,
        sample_seed: int,
        min_dense_frames: int,
        max_sample_attempts: int,
    ) -> tuple[Any, Any]:
        numpy = importlib.import_module("numpy")

        best_action = None
        best_simulation = None
        best_frame_count = -1
        for offset in range(max_sample_attempts):
            rng = numpy.random.RandomState(sample_seed + offset)
            action = simulator.sample(valid_only=True, rng=rng)
            simulation = self._simulate_action(
                simulator,
                task_index=task_index,
                action=action,
                need_images=need_images,
                stride=stride,
            )
            frame_count = self._simulation_frame_count(simulation)
            if frame_count > best_frame_count:
                best_action = action
                best_simulation = simulation
                best_frame_count = frame_count
            if frame_count >= min_dense_frames:
                return action, simulation

        if best_action is None or best_simulation is None:
            raise RuntimeError("Failed to sample a valid PHYRE action.")
        return best_action, best_simulation

    def _simulation_frame_count(self, simulation: Any) -> int:
        featurized_objects = getattr(simulation, "featurized_objects", None)
        if featurized_objects is None or not hasattr(featurized_objects, "states"):
            images_value = getattr(simulation, "images", None)
            return len(images_value) if images_value is not None else 0
        return self._effective_frame_count(featurized_objects.states)

    def _resample_images(self, images: list[Any], *, target_frame_count: int) -> list[Any]:
        if len(images) <= target_frame_count:
            return list(images)
        if target_frame_count <= 1:
            return [images[0]]
        last_index = len(images) - 1
        selected: list[Any] = []
        for output_index in range(target_frame_count):
            source_index = round(output_index * last_index / max(target_frame_count - 1, 1))
            selected.append(images[source_index])
        return selected

    def _build_dense_timesteps(
        self,
        featurized_objects: Any,
        action_payload: dict[str, Any],
        *,
        simulation_fps: int,
    ) -> list:
        if featurized_objects is None or not hasattr(featurized_objects, "states"):
            return []

        states = featurized_objects.states
        frame_count = self._effective_frame_count(states)
        timesteps = []
        for frame_index in range(frame_count):
            objects = []
            for object_index in range(featurized_objects.num_objects):
                angle = float(states[frame_index][object_index][2])
                previous_angle = float(states[max(frame_index - 1, 0)][object_index][2])
                angle_delta = self._normalized_angle_delta(angle, previous_angle)
                x = float(states[frame_index][object_index][0])
                y = float(states[frame_index][object_index][1])
                previous_x = float(states[max(frame_index - 1, 0)][object_index][0])
                previous_y = float(states[max(frame_index - 1, 0)][object_index][1])
                objects.append(
                    {
                        "object_index": object_index,
                        "shape": featurized_objects.shapes[object_index],
                        "color": featurized_objects.colors[object_index],
                        "diameter": float(featurized_objects.diameters[object_index]),
                        "x": x,
                        "y": y,
                        "angle": angle,
                        "vx": x - previous_x,
                        "vy": y - previous_y,
                        "angular_velocity": angle_delta,
                        "is_user_input": object_index >= featurized_objects.num_scene_objects,
                    }
                )

            timesteps.append(
                make_step(
                    index=frame_index,
                    t=float(frame_index) / max(simulation_fps, 1),
                    state={
                        "frame_index": frame_index,
                        "object_count": len(objects),
                    },
                    objects=objects,
                    actions=[action_payload] if frame_index == 0 else [],
                )
            )
        return timesteps

    def _effective_frame_count(self, states: Any) -> int:
        last_dynamic_index = 0
        for index in range(1, len(states)):
            frame_delta = 0.0
            for object_index in range(len(states[index])):
                for component_index in range(len(states[index][object_index])):
                    frame_delta += abs(
                        float(states[index][object_index][component_index])
                        - float(states[index - 1][object_index][component_index])
                    )
            if frame_delta > 1e-9:
                last_dynamic_index = index
        return max(last_dynamic_index + 1, 2)

    def _normalized_angle_delta(self, angle: float, previous_angle: float) -> float:
        delta = angle - previous_angle
        while delta > 0.5:
            delta -= 1.0
        while delta < -0.5:
            delta += 1.0
        return delta

    def _normalize_action(self, action: object) -> Any:
        if hasattr(action, "tolist"):
            return action.tolist()
        if isinstance(action, tuple):
            return list(action)
        return action
