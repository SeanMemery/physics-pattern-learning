from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from adapters.base import EnvironmentAdapter
from media import write_image_artifact, write_video_artifact
from schema import CanonicalTrace, TaskSpec
from traces import (
    build_provenance,
    build_simulator_trace,
    make_attachment,
    make_event,
    make_step,
    summarize_shape,
)


class KinetixAdapter(EnvironmentAdapter):
    MIN_TRACE_TIMESTEPS = 24
    DEFAULT_VIDEO_FPS = 8
    DEFAULT_VIDEO_SECONDS = 10.0
    DEFAULT_MEDIA_MAX_SIDE = 768

    def _configure_runtime_environment(self, *, capture_visuals: bool = False) -> None:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    def get_task_spec(self) -> TaskSpec:
        return self.build_task_spec()

    def list_tasks(self, **kwargs: object) -> list[str]:
        return [self._task_id_from_path(path) for path in self._level_paths()]

    def generate_trace(self, task_id: str, **kwargs: object) -> CanonicalTrace:
        self._configure_runtime_environment(capture_visuals=bool(kwargs.get("capture_frames", False)))
        action_sequence = kwargs.get("action_sequence")
        if action_sequence is None:
            raise ValueError("Kinetix trace generation requires an 'action_sequence' argument.")

        settings = self._load_runtime_settings()
        repeat_count = int(kwargs.get("repeat_count", settings["repeat_actions"]))
        annotation_mode = str(kwargs.get("annotation_mode", "unanno"))
        capture_frames = bool(kwargs.get("capture_frames", False))
        render_stride = max(1, int(kwargs.get("render_stride", 1)))
        video_fps = max(1, int(kwargs.get("video_fps", self.DEFAULT_VIDEO_FPS)))
        target_video_seconds = max(0.1, float(kwargs.get("target_video_seconds", self.DEFAULT_VIDEO_SECONDS)))
        media_max_side = max(1, int(kwargs.get("media_max_side", self.DEFAULT_MEDIA_MAX_SIDE)))
        output_dir = kwargs.get("output_dir")

        context = self._load_level_context(
            task_id,
            annotation_mode=annotation_mode,
            render_pixel_size=media_max_side,
        )
        actions = self._normalize_action_sequence(action_sequence)
        timesteps = [
            make_step(
                index=0,
                t=0.0,
                state=self._summarize_env_state(
                    context["env_state"],
                    reward=0.0,
                    done=False,
                    distance=None,
                    info={},
                    step_index=0,
                    repeat_index=-1,
                ),
                objects=self._extract_env_objects(context["env_state"]),
                actions=[],
            )
        ]
        attachments = [
            make_attachment(
                "level_spec",
                context["level_path"],
                label="Source level specification",
            )
        ]
        events = []

        if capture_frames:
            attachments.extend(
                self._render_attachments(
                    task_id,
                    index=0,
                    renderer=context["renderer"],
                    env_state=context["env_state"],
                    annotation_mode=annotation_mode,
                    output_dir=output_dir,
                    media_max_side=media_max_side,
                )
            )

        env_state = context["env_state"]
        reward = 0.0
        done = False
        info: dict[str, Any] = {}
        step_counter = 0

        for index, action in enumerate(actions):
            for repeat_index in range(repeat_count):
                env_state, reward, done, info = self._step_runtime(
                    context["env"],
                    env_state,
                    action,
                    context["env_params"],
                    step_index=index,
                    repeat_index=repeat_index,
                )
                step_counter += 1

                step_attachments = []
                is_last_step = index == len(actions) - 1 and repeat_index == repeat_count - 1
                if capture_frames and (
                    step_counter % render_stride == 0 or done or is_last_step
                ):
                    step_attachments = self._render_attachments(
                        task_id,
                        index=step_counter,
                        renderer=context["renderer"],
                        env_state=env_state,
                        annotation_mode=annotation_mode,
                        output_dir=output_dir,
                        media_max_side=media_max_side,
                    )
                    attachments.extend(step_attachments)

                distance_value = self._extract_distance(info)
                events.append(
                    make_event(
                        "environment_feedback",
                        t=float(step_counter),
                        payload={
                            "step_index": index,
                            "repeat_index": repeat_index,
                            "reward": float(reward),
                            "done": bool(done),
                            "distance": distance_value,
                            "repeat_count": repeat_count,
                        },
                    )
                )
                timesteps.append(
                    make_step(
                        index=len(timesteps),
                        t=float(step_counter),
                        state=self._summarize_env_state(
                            env_state,
                            reward=reward,
                            done=done,
                            distance=distance_value,
                            info=info,
                            step_index=index,
                            repeat_index=repeat_index,
                        ),
                        objects=self._extract_env_objects(env_state),
                        actions=[{"multi_discrete": action, "repeat_index": repeat_index}],
                        attachments=step_attachments,
                    )
                )
                if done:
                    break
            if done:
                break

        level_path = context["level_path"]
        if capture_frames:
            attachments.extend(
                self._write_visual_artifacts(
                    task_id=task_id,
                    output_dir=output_dir,
                    annotation_mode=annotation_mode,
                    video_fps=video_fps,
                    target_video_seconds=target_video_seconds,
                    media_max_side=media_max_side,
                )
            )
        padding_event = self._pad_terminal_timesteps(
            timesteps,
            minimum_count=self.MIN_TRACE_TIMESTEPS,
        )
        if padding_event is not None:
            events.append(padding_event)
        success, terminal_status = self._resolve_outcome(
            reward=reward,
            done=done,
            info=info,
            env_state=env_state,
            env_params=context["env_params"],
        )
        metrics = {
            "final_reward": float(reward),
            "final_done": bool(done),
            "goal_reached": self._extract_goal_reached(info),
            "forbidden_contact": bool(done and reward < 0.0),
            "task_success": success,
            "distance": self._extract_distance(info),
            "info": info,
            "dense_timestep_count": len(timesteps),
        }

        return build_simulator_trace(
            env_id=self.env_id,
            task_id=task_id,
            task_name=self.get_manifest().native_metric.name,
            initial_state={
                "level_path": str(level_path),
                "level_bucket": level_path.parent.name,
                "repeat_count": repeat_count,
                "annotation_mode": annotation_mode,
                "active_joints": context["num_active_joints"],
                "active_thrusters": context["num_active_thrusters"],
                "video_fps": video_fps,
                "target_video_seconds": target_video_seconds,
                "media_max_side": media_max_side,
                "render_screen_dim": list(map(int, context["render_static_env_params"].screen_dim)),
                "render_pixels_per_unit": float(context["render_env_params"].pixels_per_unit),
                "render_world_width": (
                    float(context["render_static_env_params"].screen_dim[0])
                    / max(float(context["render_env_params"].pixels_per_unit), 1e-6)
                ),
                "render_world_height": (
                    float(context["render_static_env_params"].screen_dim[1])
                    / max(float(context["render_env_params"].pixels_per_unit), 1e-6)
                ),
            },
            actions=[{"multi_discrete": action} for action in actions],
            timesteps=timesteps,
            events=events,
            attachments=attachments,
            metrics=metrics,
            success=success,
            terminal_status=terminal_status,
            provenance=build_provenance(self),
        )

    def evaluate_trace(self, trace: CanonicalTrace) -> dict[str, object]:
        metrics = dict(trace.outcome.metrics)
        metrics["success"] = bool(trace.outcome.success)
        metrics["terminal_status"] = trace.outcome.terminal_status
        return metrics

    def _load_runtime_settings(self) -> dict[str, Any]:
        config = self.load_upstream_config()
        evaluation_settings = config.get("evaluation_settings", {})
        return {"repeat_actions": int(evaluation_settings.get("repeat_actions", 16))}

    def _levels_root(self) -> Path:
        return self.resolve_project_path("src/vendor/kinetix_suite/levels")

    def _level_paths(self) -> list[Path]:
        root = self._levels_root()
        return sorted(root.rglob("*.json"))

    def _task_id_from_path(self, path: Path) -> str:
        root = self._levels_root()
        return path.relative_to(root).with_suffix("").as_posix()

    def _path_from_task_id(self, task_id: str) -> Path:
        direct = self._levels_root() / f"{task_id}.json"
        if direct.is_file():
            return direct

        matches = [path for path in self._level_paths() if path.stem == task_id]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(f"Unknown Kinetix task id: {task_id}")
        raise ValueError(f"Ambiguous Kinetix task id '{task_id}'; use a bucket-prefixed id.")

    def _normalize_action_sequence(self, action_sequence: object) -> list[list[int]]:
        if isinstance(action_sequence, list) and action_sequence and all(
            isinstance(item, int) for item in action_sequence
        ):
            return [list(action_sequence)]
        return [list(map(int, action)) for action in list(action_sequence)]

    def _load_level_context(
        self,
        task_id: str,
        *,
        annotation_mode: str,
        render_pixel_size: int | None = None,
    ) -> dict[str, Any]:
        self._configure_runtime_environment(capture_visuals=False)
        level_path = self._path_from_task_id(task_id)
        with level_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        env_state = data.get("env_state", {})
        num_active_joints = sum(
            1 for joint in env_state.get("joint", []) if joint.get("motor_on", False)
        )
        num_active_thrusters = sum(
            1 for thruster in env_state.get("thruster", []) if thruster.get("active", False)
        )

        self._ensure_wandb_stub()
        jax = importlib.import_module("jax")
        env_module = importlib.import_module("vendor.kinetix_suite.environment")
        utils_module = importlib.import_module("vendor.kinetix_suite.environment.utils")
        render_module = importlib.import_module("vendor.kinetix_suite.render")
        saving_module = self._load_saving_module()

        level_rel = level_path.relative_to(self._levels_root()).as_posix()
        level, static_env_params, env_params = saving_module.load_from_json_file(level_rel)
        env = env_module.make_kinetix_env(
            action_type=utils_module.ActionType.MULTI_DISCRETE,
            observation_type=utils_module.ObservationType.PIXELS,
            reset_fn=lambda rng: level,
            env_params=env_params,
            static_env_params=static_env_params,
        )
        render_env_params = env_params
        render_static_env_params = static_env_params
        if render_pixel_size is not None:
            target_size = max(64, int(render_pixel_size))
            source_width, source_height = tuple(map(int, static_env_params.screen_dim))
            scale = target_size / float(max(source_width, source_height))
            render_downscale = int(static_env_params.downscale)
            if annotation_mode == "anno":
                # The vendored Kinetix pixel renderers re-magnify by 4x internally.
                # For DeepPHY prompt capture we only need a 512px artifact, so raise
                # the renderer downscale enough to keep its transient full-screen
                # work buffers bounded instead of exploding to multi-gigapixel sizes.
                render_downscale = max(render_downscale, 8)
            render_static_env_params = static_env_params.replace(
                screen_dim=(
                    max(1, int(round(source_width * scale))),
                    max(1, int(round(source_height * scale))),
                ),
                # Kinetix renderers derive an internal downscale by dividing the
                # static downscale by their magnification factor. For DeepPHY
                # initial-image capture we keep that invariant, but we also raise
                # the effective downscale so the internal working surface stays
                # near the requested prompt size instead of the original level size.
                downscale=render_downscale,
            )
            render_env_params = env_params.replace(
                pixels_per_unit=max(1, int(round(float(env_params.pixels_per_unit) * scale)))
            )

        renderer = (
            render_module.make_render_pixels_anno(render_env_params, render_static_env_params)
            if annotation_mode == "anno"
            else render_module.make_render_pixels(render_env_params, render_static_env_params)
        )
        _, rng_reset = jax.random.split(jax.random.PRNGKey(0))
        _, initial_env_state = env.reset(rng_reset, env_params)

        return {
            "level_path": level_path,
            "env": env,
            "env_state": initial_env_state,
            "env_params": env_params,
            "render_env_params": render_env_params,
            "render_static_env_params": render_static_env_params,
            "renderer": renderer,
            "jax": jax,
            "num_active_joints": num_active_joints,
            "num_active_thrusters": num_active_thrusters,
        }

    def _summarize_env_state(
        self,
        env_state: Any,
        *,
        reward: float,
        done: bool,
        distance: float | None,
        info: dict[str, Any],
        step_index: int,
        repeat_index: int,
    ) -> dict[str, Any]:
        if not hasattr(env_state, "polygon"):
            return {
                "reward": float(reward),
                "done": bool(done),
                "distance": distance,
                "step_index": step_index,
                "repeat_index": repeat_index,
            }

        return {
            "reward": float(reward),
            "done": bool(done),
            "distance": distance,
            "step_index": step_index,
            "repeat_index": repeat_index,
            "env_timestep": int(getattr(env_state, "timestep", 0)),
            "last_distance": float(getattr(env_state, "last_distance", -1.0)),
            "active_polygon_count": int(np.asarray(env_state.polygon.active).sum()),
            "active_circle_count": int(np.asarray(env_state.circle.active).sum()),
            "active_joint_count": int(np.asarray(env_state.joint.active).sum()),
            "active_thruster_count": int(np.asarray(env_state.thruster.active).sum()),
            "info": info,
        }

    def _extract_env_objects(self, env_state: Any) -> list[dict[str, Any]]:
        if not hasattr(env_state, "polygon"):
            return []

        objects: list[dict[str, Any]] = []
        objects.extend(self._extract_rigid_body_objects(env_state, body_name="polygon"))
        objects.extend(self._extract_rigid_body_objects(env_state, body_name="circle"))
        objects.extend(self._extract_joint_objects(env_state))
        objects.extend(self._extract_thruster_objects(env_state))
        return objects

    def _extract_rigid_body_objects(self, env_state: Any, *, body_name: str) -> list[dict[str, Any]]:
        body = getattr(env_state, body_name)
        active = np.asarray(getattr(body, "active"))
        positions = np.asarray(getattr(body, "position"))
        velocities = np.asarray(getattr(body, "velocity"))
        rotations = np.asarray(getattr(body, "rotation"))
        angular_velocities = np.asarray(getattr(body, "angular_velocity"))
        radii = np.asarray(getattr(body, "radius"))
        inverse_mass = np.asarray(getattr(body, "inverse_mass"))
        inverse_inertia = np.asarray(getattr(body, "inverse_inertia"))
        friction = np.asarray(getattr(body, "friction"))
        restitution = np.asarray(getattr(body, "restitution"))
        collision_mode = np.asarray(getattr(body, "collision_mode"))
        vertices = np.asarray(getattr(body, "vertices"))
        n_vertices = np.asarray(getattr(body, "n_vertices"))

        roles = np.asarray(getattr(env_state, f"{body_name}_shape_roles", np.zeros(len(active), dtype=int)))
        highlighted = np.asarray(getattr(env_state, f"{body_name}_highlighted", np.zeros(len(active), dtype=int)))
        densities = np.asarray(getattr(env_state, f"{body_name}_densities", np.zeros(len(active), dtype=float)))

        objects = []
        for index, is_active in enumerate(active):
            if not bool(is_active):
                continue
            vertex_count = int(n_vertices[index]) if index < len(n_vertices) else 0
            objects.append(
                {
                    "kind": body_name,
                    "object_index": index,
                    "position": positions[index].tolist(),
                    "velocity": velocities[index].tolist(),
                    "rotation": float(rotations[index]),
                    "angular_velocity": float(angular_velocities[index]),
                    "radius": float(radii[index]),
                    "inverse_mass": float(inverse_mass[index]),
                    "inverse_inertia": float(inverse_inertia[index]),
                    "friction": float(friction[index]),
                    "restitution": float(restitution[index]),
                    "collision_mode": int(collision_mode[index]),
                    "shape_role": int(roles[index]),
                    "highlighted": bool(highlighted[index]),
                    "density": float(densities[index]),
                    "vertices": vertices[index][:vertex_count].tolist(),
                }
            )
        return objects

    def _extract_joint_objects(self, env_state: Any) -> list[dict[str, Any]]:
        joint = env_state.joint
        active = np.asarray(getattr(joint, "active"))
        objects = []
        for index, is_active in enumerate(active):
            if not bool(is_active):
                continue
            objects.append(
                {
                    "kind": "joint",
                    "object_index": index,
                    "a_index": int(np.asarray(joint.a_index)[index]),
                    "b_index": int(np.asarray(joint.b_index)[index]),
                    "global_position": np.asarray(joint.global_position)[index].tolist(),
                    "rotation": float(np.asarray(joint.rotation)[index]),
                    "motor_on": bool(np.asarray(joint.motor_on)[index]),
                    "motor_speed": float(np.asarray(joint.motor_speed)[index]),
                    "motor_power": float(np.asarray(joint.motor_power)[index]),
                    "is_fixed_joint": bool(np.asarray(joint.is_fixed_joint)[index]),
                    "min_rotation": float(np.asarray(joint.min_rotation)[index]),
                    "max_rotation": float(np.asarray(joint.max_rotation)[index]),
                }
            )
        return objects

    def _extract_thruster_objects(self, env_state: Any) -> list[dict[str, Any]]:
        thruster = env_state.thruster
        active = np.asarray(getattr(thruster, "active"))
        objects = []
        for index, is_active in enumerate(active):
            if not bool(is_active):
                continue
            objects.append(
                {
                    "kind": "thruster",
                    "object_index": index,
                    "parent_object_index": int(np.asarray(thruster.object_index)[index]),
                    "global_position": np.asarray(thruster.global_position)[index].tolist(),
                    "rotation": float(np.asarray(thruster.rotation)[index]),
                    "power": float(np.asarray(thruster.power)[index]),
                }
            )
        return objects

    def _load_saving_module(self) -> Any:
        self._ensure_wandb_stub()
        return importlib.import_module("vendor.kinetix_suite.util.saving")

    def _ensure_wandb_stub(self) -> None:
        for module_name in [name for name in sys.modules if name == "wandb" or name.startswith("wandb.")]:
            sys.modules.pop(module_name, None)

        stub = types.ModuleType("wandb")
        stub.run = object
        stub.init = lambda *args, **kwargs: None
        stub.define_metric = lambda *args, **kwargs: None
        stub.Api = lambda *args, **kwargs: None
        stub.Artifact = lambda *args, **kwargs: None
        sys.modules["wandb"] = stub

    def _step_runtime(
        self,
        env: Any,
        env_state: Any,
        action: list[int],
        env_params: Any,
        *,
        step_index: int,
        repeat_index: int,
    ) -> tuple[Any, float, bool, dict[str, Any]]:
        jax = importlib.import_module("jax")
        jnp = importlib.import_module("jax.numpy")
        _, next_env_state, reward, done, info = env.step(
            jax.random.PRNGKey(step_index * 1024 + repeat_index),
            env_state,
            jnp.array(action, dtype=jnp.int32),
            env_params,
        )
        return next_env_state, float(reward), bool(done), self._normalize_info(info)

    def _render_attachments(
        self,
        task_id: str,
        *,
        index: int,
        renderer: Any,
        env_state: Any,
        annotation_mode: str,
        output_dir: object,
        media_max_side: int,
    ) -> list:
        rendered = renderer(env_state)
        attachments = []
        if annotation_mode == "anno" and isinstance(rendered, tuple):
            pixels = rendered[0]
            attachment_path = self._write_render_output(
                pixels,
                output_dir=output_dir,
                task_id=task_id,
                index=index,
                suffix="annotated",
                media_max_side=media_max_side,
            )
            if output_dir is not None:
                return []
            attachments.append(
                make_attachment(
                    "annotated_frame",
                    attachment_path or f"memory://kinetix/{task_id}/frame_{index:03d}_annotated.png",
                    label=f"Annotated frame {index}",
                    metadata=summarize_shape(pixels),
                )
            )
        else:
            attachment_path = self._write_render_output(
                rendered,
                output_dir=output_dir,
                task_id=task_id,
                index=index,
                suffix="rendered",
                media_max_side=media_max_side,
            )
            if output_dir is not None:
                return []
            attachments.append(
                make_attachment(
                    "rendered_frame",
                    attachment_path or f"memory://kinetix/{task_id}/frame_{index:03d}.png",
                    label=f"Frame {index}",
                    metadata=summarize_shape(rendered),
                )
            )
        return attachments

    def _extract_distance(self, info: dict[str, Any]) -> float | None:
        if "distance" not in info:
            return None
        distance = info["distance"]
        if hasattr(distance, "item"):
            distance = distance.item()
        return float(distance)

    def _extract_goal_reached(self, info: dict[str, Any]) -> bool | None:
        goal_reached = info.get("GoalR")
        if isinstance(goal_reached, bool):
            return goal_reached
        if hasattr(goal_reached, "item"):
            goal_reached = goal_reached.item()
        if isinstance(goal_reached, bool):
            return goal_reached
        return None

    def _resolve_outcome(
        self,
        *,
        reward: float,
        done: bool,
        info: dict[str, Any],
        env_state: Any,
        env_params: Any,
    ) -> tuple[bool, str]:
        goal_reached = self._extract_goal_reached(info)
        if goal_reached is True or reward > 0.0:
            return True, "goal_reached"
        if done and reward < 0.0:
            return False, "forbidden_contact"

        max_timesteps = getattr(env_params, "max_timesteps", None)
        timestep = getattr(env_state, "timestep", None)
        if (
            done
            and max_timesteps is not None
            and timestep is not None
            and int(timestep) >= int(max_timesteps)
        ):
            return False, "max_steps_reached"
        if done:
            return False, "terminated_without_goal"
        return False, "max_steps_reached"

    def _normalize_info(self, info: Any) -> dict[str, Any]:
        if isinstance(info, dict):
            normalized = {}
            for key, value in info.items():
                if hasattr(value, "item"):
                    value = value.item()
                normalized[str(key)] = value
            return normalized
        return {}

    def _write_render_output(
        self,
        rendered: Any,
        *,
        output_dir: object,
        task_id: str,
        index: int,
        suffix: str,
        media_max_side: int,
    ) -> str | None:
        if output_dir is None:
            return None

        output_path = Path(str(output_dir)) / f"{task_id.replace('/', '__')}_frame_{index:03d}_{suffix}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            image = self._prepare_render_image(rendered, media_max_side=media_max_side)
            image.save(output_path)
            return str(output_path)
        except Exception:
            fallback_path = output_path.with_suffix(".npy")
            np.save(fallback_path, np.asarray(rendered))
            return str(fallback_path)

    def _write_visual_artifacts(
        self,
        *,
        task_id: str,
        output_dir: object,
        annotation_mode: str,
        video_fps: int,
        target_video_seconds: float,
        media_max_side: int,
    ) -> list:
        if output_dir is None:
            return []

        root = Path(str(output_dir))
        task_prefix = task_id.replace("/", "__")
        preferred_suffix = "annotated" if annotation_mode == "anno" else "rendered"
        frame_paths = sorted(root.glob(f"{task_prefix}_frame_*_{preferred_suffix}.png"))
        if not frame_paths and preferred_suffix != "rendered":
            frame_paths = sorted(root.glob(f"{task_prefix}_frame_*_rendered.png"))
        if not frame_paths:
            return []

        initial_scene_path = root / "initial_scene.png"
        video_path = root / "simulation.mp4"
        target_frame_count = max(2, int(round(video_fps * target_video_seconds)))
        video_frames = self._resample_video_frames(frame_paths, target_frame_count=target_frame_count)
        initial_metadata = write_image_artifact(frame_paths[0], initial_scene_path, max_side=media_max_side)
        video_metadata = write_video_artifact(video_frames, video_path, fps=video_fps, max_side=media_max_side)
        video_metadata.update(
            {
                "duration_seconds": round(video_metadata["frame_count"] / max(video_metadata["fps"], 1), 3),
                "target_duration_seconds": round(target_video_seconds, 3),
                "frame_selection": "uniform_resample_with_repeats",
            }
        )
        for path in frame_paths:
            path.unlink(missing_ok=True)
        return [
            make_attachment("initial_scene", initial_scene_path, label="Initial scene", metadata=initial_metadata),
            make_attachment("simulation_video", video_path, label="Simulation video", metadata=video_metadata),
        ]

    def _prepare_render_image(self, rendered: Any, *, media_max_side: int) -> Image.Image:
        pixels = np.asarray(rendered)
        if pixels.dtype != np.uint8:
            pixels = np.clip(pixels, 0, 255).astype("uint8")
        if pixels.ndim == 3:
            pixels = pixels.transpose(1, 0, 2)[::-1]
        if pixels.ndim == 2:
            pixels = np.stack([pixels] * 3, axis=-1)
        if pixels.ndim != 3:
            raise ValueError(f"unsupported render shape: {pixels.shape}")
        if pixels.shape[-1] == 4:
            pixels = pixels[..., :3]

        image = Image.fromarray(pixels, mode="RGB")
        if image.size != (media_max_side, media_max_side):
            image = image.resize((media_max_side, media_max_side), Image.Resampling.NEAREST)
        return image

    def _resample_video_frames(
        self,
        frame_paths: list[Path],
        *,
        target_frame_count: int,
    ) -> list[Path]:
        if len(frame_paths) == target_frame_count:
            return list(frame_paths)
        if len(frame_paths) == 1:
            return [frame_paths[0]] * target_frame_count

        last_index = len(frame_paths) - 1
        selected: list[Path] = []
        for output_index in range(target_frame_count):
            source_index = round(output_index * last_index / max(target_frame_count - 1, 1))
            selected.append(frame_paths[source_index])
        return selected

    def _pad_terminal_timesteps(
        self,
        timesteps: list,
        *,
        minimum_count: int,
    ):
        if len(timesteps) >= minimum_count or not timesteps:
            return None

        last_step = timesteps[-1]
        last_t = float(last_step.t) if last_step.t is not None else float(len(timesteps) - 1)
        state = dict(last_step.state)
        state["is_terminal_summary"] = True
        state["padding_origin"] = "kinetix_terminal_summary"

        original_count = len(timesteps)
        while len(timesteps) < minimum_count:
            timesteps.append(
                make_step(
                    index=len(timesteps),
                    t=last_t + float(len(timesteps) - original_count + 1),
                    state=state,
                    objects=list(last_step.objects),
                    actions=[],
                )
            )

        return make_event(
            "terminal_padding",
            t=float(timesteps[-1].t) if timesteps[-1].t is not None else float(len(timesteps) - 1),
            payload={"added_timesteps": len(timesteps) - original_count},
        )
