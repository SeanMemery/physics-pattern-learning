from __future__ import annotations

import importlib
import json
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

from adapters.base import EnvironmentAdapter
from debug import debug_log
from media import write_image_artifact, write_video_artifact
from runtime.isolation import isolated_pooltool_home, pooltool_source_root
from schema import CanonicalTrace, TaskSpec
from traces import (
    build_provenance,
    build_simulator_trace,
    make_attachment,
    make_event,
    make_step,
)


class PooltoolAdapter(EnvironmentAdapter):
    DEFAULT_NUM_GAMES = 100
    ASKEW_CAMERA_STATE = "7_foot_offcenter"
    OVERHEAD_CAMERA_STATE = "7_foot_overhead"
    REVIEW_VIDEO_FPS = 20
    REVIEW_VIDEO_SECONDS = 5.0
    RENDER_IMAGE_MAX_SIDE = 1536
    INITIAL_SCENE_WIDTH_PX = 1920
    OFFSCREEN_RENDER_WIDTH = 1920
    OFFSCREEN_RENDER_HEIGHT = 1200
    TABLE_MESH_PITCH_DEGREES = 90.0
    def get_task_spec(self) -> TaskSpec:
        return self.build_task_spec()

    def list_tasks(self, **kwargs: object) -> list[str]:
        num_games = int(kwargs.get("num_games", self.DEFAULT_NUM_GAMES))
        return [f"game-{index:03d}" for index in range(num_games)]

    def generate_trace(self, task_id: str, **kwargs: object) -> CanonicalTrace:
        debug_log("pooltool.generate_trace.start", task_id=task_id, kwargs=kwargs)
        shots = kwargs.get("shots")
        if shots is None:
            raise ValueError("Pooltool trace generation requires a 'shots' argument.")

        backend = self._load_backend()
        settings = self._load_runtime_settings()
        shot_list = self._normalize_shots(shots)
        max_attempts = min(int(kwargs.get("max_attempts", settings["max_attempts"])), len(shot_list))
        capture_frames = bool(kwargs.get("capture_frames", False))
        continuous_dt = float(kwargs.get("continuous_dt", 0.05))
        output_dir = kwargs.get("output_dir")
        layout_seed = kwargs.get("layout_seed")
        rack_jitter = float(kwargs.get("rack_jitter", 0.0))

        current_system = self._copy_system(self._create_initial_system(backend))
        layout_randomized = False
        if layout_seed is not None or rack_jitter > 0:
            layout_randomized = self._randomize_initial_layout(
                current_system,
                seed=int(layout_seed or 0),
                rack_jitter=rack_jitter,
            )
        initial_layout = self._ball_objects(current_system)
        initial_ball_count = len(self._ball_objects(current_system))
        timesteps = []
        events = []
        attachments = []
        success = False
        terminal_status = "max_attempts_exhausted"
        attempts_used = 0
        media_render_system = None
        attachments.extend(
            self._capture_table_snapshot(
                current_system,
                task_id=task_id,
                attempt_index=0,
                output_dir=output_dir,
                capture_frames=capture_frames,
                label="Initial rack state",
            )
        )

        for attempt_index, shot in enumerate(shot_list[:max_attempts]):
            debug_log("pooltool.attempt.start", task_id=task_id, attempt_index=attempt_index, shot=shot)
            attempts_used += 1
            cue_ball_id = current_system.cue.cue_ball_id
            repositioned = self._ensure_cue_ball_present(backend, current_system)
            if repositioned:
                events.append(
                    make_event(
                        "cue_ball_repositioned",
                        t=float(attempt_index),
                        payload={"cue_ball_id": cue_ball_id},
                    )
                )

            if hasattr(current_system, "stop_balls"):
                current_system.stop_balls()
            if hasattr(current_system, "reset_history"):
                current_system.reset_history()

            action_params = self._resolve_shot_parameters(shot, settings)
            potted_balls: list[str] = []
            warning: str | None = None

            if action_params is None:
                target_ball_id = None
                outcome = "INVALID_VLM_ACTION"
            else:
                target_ball_id = self._select_target_ball(current_system, cue_ball_id=cue_ball_id)
                if target_ball_id is None:
                    success = "9" not in current_system.balls
                    outcome = "NO_TARGET_BALLS_LEFT"
                else:
                    V0, a, b = action_params
                    current_system.cue.set_state(
                        V0=V0,
                        phi=self._aim_at_ball(backend, current_system, target_ball_id),
                        a=a,
                        b=b,
                    )
                    try:
                        attempt_time_offset = (
                            float(timesteps[-1].t + continuous_dt)
                            if timesteps and timesteps[-1].t is not None
                            else 0.0
                        )
                        simulated_system = self._simulate(backend, current_system)
                        dense_timesteps = self._extract_dense_timesteps(
                            backend,
                            simulated_system,
                            shot=shot,
                            attempt_index=attempt_index,
                            time_offset=attempt_time_offset,
                            continuous_dt=continuous_dt,
                        )
                        timesteps.extend(dense_timesteps)
                        events.extend(
                            self._extract_dense_events(
                                simulated_system,
                                attempt_index=attempt_index,
                                time_offset=attempt_time_offset,
                            )
                        )
                        potted_balls = self._extract_potted_balls(backend, simulated_system)
                        if "9" in potted_balls:
                            success = True
                            outcome = "9_BALL_POTTED"
                        elif cue_ball_id in potted_balls:
                            outcome = "CUE_BALL_POTTED"
                        elif potted_balls:
                            outcome = "OTHER_BALLS_POTTED"
                        else:
                            outcome = "NO_BALLS_POTTED"
                        # Keep the fully simulated attempt with its motion history for video rendering.
                        media_render_system = simulated_system
                        current_system = self._copy_system(simulated_system)
                        self._remove_potted_balls(current_system, potted_balls)
                        debug_log(
                            "pooltool.attempt.simulated",
                            task_id=task_id,
                            attempt_index=attempt_index,
                            outcome=outcome,
                            potted_balls=potted_balls,
                        )
                    except ValueError as error:
                        if "cannot assign slice of shape (2,) from input of shape (4,)" in str(error):
                            outcome = "SIMULATION_CRASHED_ROOT_SOLVER_ISSUE"
                            warning = str(error)
                        else:
                            outcome = "UNEXPECTED_SIMULATION_ERROR"
                            warning = str(error)
                    except Exception as error:
                        outcome = "UNEXPECTED_SIMULATION_ERROR"
                        warning = str(error)
                        debug_log(
                            "pooltool.attempt.failed",
                            task_id=task_id,
                            attempt_index=attempt_index,
                            error=str(error),
                        )

            step_attachments = self._capture_table_snapshot(
                current_system,
                task_id=task_id,
                attempt_index=attempt_index + 1,
                output_dir=output_dir,
                capture_frames=capture_frames,
                label=f"Attempt {attempt_index + 1} table state",
            )
            attachments.extend(step_attachments)

            remaining_balls = self._ball_objects(current_system)
            events.append(
                make_event(
                    "shot_prepared",
                    t=float(attempt_index),
                    payload={
                        "target_ball_id": target_ball_id,
                        "shot": shot,
                        "resolved_action_params": action_params,
                    },
                )
            )
            events.append(
                make_event(
                    "shot_outcome",
                    t=float(timesteps[-1].t) if timesteps and timesteps[-1].t is not None else float(attempt_index),
                    payload={
                        "outcome": outcome,
                        "potted_balls": potted_balls,
                        "warning": warning,
                    },
                )
            )
            if not timesteps or timesteps[-1].state.get("attempt_index") != attempt_index:
                timesteps.append(
                    make_step(
                        index=len(timesteps),
                        t=float(attempt_index),
                        state={
                            "attempt_index": attempt_index,
                            "outcome": outcome,
                            "warning": warning,
                            "remaining_ball_count": len(remaining_balls),
                            "target_ball_id": target_ball_id,
                            "potted_balls": potted_balls,
                        },
                        objects=remaining_balls,
                        actions=[shot],
                        attachments=step_attachments,
                    )
                )
            elif step_attachments:
                timesteps[-1].attachments.extend(step_attachments)

            terminal_status = outcome
            if success or outcome == "NO_TARGET_BALLS_LEFT":
                break

        render_media_error = None
        media_attachments, render_media_error = self._optional_render_media_artifacts(
            backend,
            media_render_system if media_render_system is not None else current_system,
            output_dir=output_dir,
            continuous_dt=continuous_dt,
            capture_frames=capture_frames,
        )
        attachments.extend(media_attachments)
        if render_media_error is not None:
            events.append(
                make_event(
                    "render_media_failed",
                    t=float(timesteps[-1].t) if timesteps and timesteps[-1].t is not None else None,
                    payload={"error": render_media_error},
                )
            )

        if len(timesteps) == 1:
            last_state = dict(timesteps[-1].state)
            last_state["is_terminal_summary"] = True
            timesteps.append(
                make_step(
                    index=1,
                    t=float(timesteps[-1].t or 0.0) + 1.0,
                    state=last_state,
                    objects=self._ball_objects(current_system),
                    actions=[],
                )
            )

        return build_simulator_trace(
            env_id=self.env_id,
            task_id=task_id,
            task_name=self.get_manifest().native_metric.name,
            initial_state={
                "speed_mapping": settings["speed_mapping"],
                "strikespot_mapping": settings["strikespot_mapping"],
                "max_attempts": settings["max_attempts"],
                "initial_ball_count": initial_ball_count,
                "continuous_dt": continuous_dt,
                "layout_seed": int(layout_seed) if layout_seed is not None else None,
                "rack_jitter": rack_jitter,
                "layout_randomized": layout_randomized,
                "initial_ball_positions": initial_layout,
                "render_camera_state": self.OVERHEAD_CAMERA_STATE,
                "render_views": [
                    {"view": view_name, "camera_state": camera_state}
                    for view_name, camera_state in self._render_views()
                ],
            },
            actions=shot_list[:max_attempts],
            timesteps=timesteps,
            events=events,
            attachments=attachments,
            metrics={
                "attempts": attempts_used,
                "dense_timestep_count": len(timesteps),
                "terminal_status": terminal_status,
                "nine_ball_success": success,
                "render_media_requested": capture_frames,
                **(
                    {"render_media_error": render_media_error}
                    if render_media_error is not None
                    else {}
                ),
            },
            success=success,
            terminal_status=terminal_status,
            provenance=build_provenance(
                self,
                notes=["surfaces solver instability as explicit outcome metadata"],
            ),
        )

    def evaluate_trace(self, trace: CanonicalTrace) -> dict[str, object]:
        metrics = dict(trace.outcome.metrics)
        metrics["success"] = bool(trace.outcome.success)
        return metrics

    def _load_runtime_settings(self) -> dict[str, Any]:
        config = self.load_upstream_config()
        return {
            "max_attempts": int(config.get("max_attempts_per_game", 15)),
            "speed_mapping": config.get("speed_mapping", {}),
            "strikespot_mapping": config.get("strikespot_mapping", {}),
        }

    def _normalize_shots(self, shots: object) -> list[dict[str, str]]:
        if isinstance(shots, dict):
            return [shots]
        return [dict(shot) for shot in list(shots)]

    def _load_backend(self) -> Any:
        with isolated_pooltool_home(self.resolve_project_path("")):
            source_root = pooltool_source_root(self.resolve_project_path(""))
            if source_root.is_dir():
                source_path = str(source_root)
                if source_path not in sys.path:
                    sys.path.insert(0, source_path)
            debug_log("pooltool.backend.load", source_root=source_root)
            backend = importlib.import_module("pooltool")
            self._patch_ball_render_texture_lookup()
            self._patch_table_render_model_transform()
            return backend

    def _patch_ball_render_texture_lookup(self) -> None:
        render_module = importlib.import_module("pooltool.objects.ball.render")
        original = render_module.BallRender.init_sphere
        if getattr(original, "_deepphy_texture_guard", False):
            return

        def patched(ball_render_self) -> None:
            position = (
                render_module.Global.render.find("scene")
                .find("table")
                .attachNewNode(f"ball_{ball_render_self._ball.id}_position")
            )
            ball_node = position.attachNewNode(f"ball_{ball_render_self._ball.id}")

            sphere_node = render_module.Global.loader.loadModel(
                render_module.panda_path(ball_render_self.model_path)
            )
            sphere_node.reparentTo(position)

            if ball_render_self._ball.ballset is None:
                tex = sphere_node.find_texture(render_module.FALLBACK_ID)
            else:
                tex = sphere_node.find_texture(ball_render_self.model_path.stem)
            if tex is None:
                textures = sphere_node.findAllTextures()
                if len(textures) > 0:
                    tex = textures[0]
            if tex is None:
                tex = sphere_node.getTexture()
            if tex is not None:
                tex.set_minfilter(render_module.SamplerState.FT_linear)

            sphere_node.setScale(ball_render_self.get_scale_factor(sphere_node))
            position.setPos(*ball_render_self._ball.state.rvw[0, :])

            ball_render_self.nodes["sphere"] = sphere_node
            ball_render_self.nodes["ball"] = ball_node
            ball_render_self.nodes["pos"] = position
            ball_render_self.nodes["shadow"] = ball_render_self.init_shadow()

            ball_render_self.set_orientation(ball_render_self._ball.initial_orientation)

        patched._deepphy_texture_guard = True
        render_module.BallRender.init_sphere = patched

    def _patch_table_render_model_transform(self) -> None:
        render_module = importlib.import_module("pooltool.objects.table.render")
        table_datatypes = importlib.import_module("pooltool.objects.table.datatypes")
        globals_module = importlib.import_module("pooltool.ani.globals")
        original = render_module.TableRender.init_table
        if getattr(original, "_deepphy_table_mesh_fix", False):
            return

        def patched(table_render_self) -> None:
            table = table_render_self._table
            graphics_settings = self._load_pooltool_graphics_settings()
            if (
                not table.model_descr
                or table.model_descr == table_datatypes.TableModelDescr.null()
                or not self._get_pooltool_graphics_flag(graphics_settings, "table", default=True)
            ):
                return original(table_render_self)

            model_descr = table.model_descr
            model_path: str | None = None
            get_path = getattr(model_descr, "get_path", None)
            if callable(get_path):
                model_path = str(
                    get_path(
                        self._get_pooltool_graphics_flag(
                            graphics_settings,
                            "physical_based_rendering",
                            default=False,
                        )
                    )
                )
            else:
                raw_path = getattr(model_descr, "path", None)
                if callable(raw_path):
                    raw_path = raw_path()
                if raw_path:
                    model_path = str(raw_path)
            if not model_path:
                return original(table_render_self)

            parent = globals_module.Global.render.find("scene").attachNewNode("table")
            model = globals_module.Global.loader.loadModel(model_path)
            model.reparentTo(parent)
            model.setP(self.TABLE_MESH_PITCH_DEGREES)

            table_render_self.nodes["table"] = parent
            table_render_self.collision_nodes = {}

        patched._deepphy_table_mesh_fix = True
        render_module.TableRender.init_table = patched

    def _configure_headless_render_settings(self, backend: Any) -> None:
        graphics_settings = getattr(getattr(getattr(backend, "config", None), "settings", None), "graphics", None)
        if graphics_settings is None:
            graphics_settings = self._load_pooltool_graphics_settings()
        self._set_pooltool_graphics_flag(graphics_settings, "room", False)
        self._set_pooltool_graphics_flag(graphics_settings, "floor", False)

    def _load_pooltool_graphics_settings(self) -> Any:
        try:
            config_module = importlib.import_module("pooltool.config")
        except ModuleNotFoundError:
            config_module = None
        settings = getattr(config_module, "settings", None)
        if settings is None:
            ani_module = importlib.import_module("pooltool.ani")
            settings = getattr(ani_module, "settings", None)
        if isinstance(settings, dict):
            return settings.get("graphics")
        graphics_settings = getattr(settings, "graphics", None)
        if graphics_settings is not None:
            return graphics_settings
        if hasattr(settings, "get"):
            return settings.get("graphics")
        return None

    def _get_pooltool_graphics_flag(self, graphics_settings: Any, name: str, *, default: bool) -> bool:
        if graphics_settings is None:
            return default
        if isinstance(graphics_settings, dict):
            return bool(graphics_settings.get(name, default))
        if hasattr(graphics_settings, name):
            return bool(getattr(graphics_settings, name))
        if hasattr(graphics_settings, "get"):
            return bool(graphics_settings.get(name, default))
        return default

    def _set_pooltool_graphics_flag(self, graphics_settings: Any, name: str, value: bool) -> None:
        if graphics_settings is None:
            return
        if isinstance(graphics_settings, dict):
            graphics_settings[name] = value
            return
        if hasattr(graphics_settings, name):
            setattr(graphics_settings, name, value)
            return
        if hasattr(graphics_settings, "__setitem__"):
            graphics_settings[name] = value

    def _create_initial_system(self, backend: Any) -> Any:
        table = backend.Table.default()
        balls = backend.get_rack(backend.GameType.NINEBALL, table)
        cue = backend.Cue(cue_ball_id="cue")
        return backend.System(table=table, balls=OrderedDict(balls), cue=cue)

    def _randomize_initial_layout(
        self,
        system: Any,
        *,
        seed: int,
        rack_jitter: float,
    ) -> bool:
        if rack_jitter <= 0:
            return False

        rng = random.Random(seed)
        table_width = float(getattr(system.table, "w", 1.0))
        table_length = float(getattr(system.table, "l", 2.0))
        min_spacing = max(0.06, 0.08 * min(table_width, table_length))
        placed_positions: list[tuple[float, float]] = []
        randomized = False

        cue_ball_id = getattr(system.cue, "cue_ball_id", "cue")
        cue_ball = system.balls.get(cue_ball_id)
        if cue_ball is not None:
            cue_x = rng.uniform(0.25 * table_width, 0.75 * table_width)
            cue_y = rng.uniform(0.15 * table_length, 0.30 * table_length)
            self._set_ball_xy(cue_ball, cue_x, cue_y)
            placed_positions.append((cue_x, cue_y))
            randomized = True

        for ball_id, ball in system.balls.items():
            if ball_id == cue_ball_id:
                continue
            placed = False
            for _ in range(64):
                x = rng.uniform(0.15 * table_width, 0.85 * table_width)
                y = rng.uniform(0.45 * table_length, 0.85 * table_length)
                if all(((x - px) ** 2 + (y - py) ** 2) ** 0.5 >= min_spacing for px, py in placed_positions):
                    self._set_ball_xy(ball, x, y)
                    placed_positions.append((x, y))
                    randomized = True
                    placed = True
                    break
            if not placed:
                x, y = self._ball_xy(ball)
                jitter_scale_x = rack_jitter * table_width
                jitter_scale_y = rack_jitter * table_length
                x = min(max(x + rng.uniform(-jitter_scale_x, jitter_scale_x), 0.1 * table_width), 0.9 * table_width)
                y = min(max(y + rng.uniform(-jitter_scale_y, jitter_scale_y), 0.35 * table_length), 0.9 * table_length)
                self._set_ball_xy(ball, x, y)
                placed_positions.append((x, y))
                randomized = True

        return randomized

    def _copy_system(self, system: Any) -> Any:
        return system.copy() if hasattr(system, "copy") else system

    def _ensure_cue_ball_present(self, backend: Any, system: Any) -> bool:
        cue_ball_id = system.cue.cue_ball_id
        if cue_ball_id in system.balls:
            return False
        x_reposition = system.table.w / 2
        y_reposition = system.table.l / 4
        new_ball = backend.Ball.create(cue_ball_id, xy=[x_reposition, y_reposition])
        system.balls[cue_ball_id] = new_ball
        return True

    def _resolve_shot_parameters(
        self,
        shot: dict[str, str],
        settings: dict[str, Any],
    ) -> tuple[float, float, float] | None:
        speed_name = shot.get("speed_name") or shot.get("speed")
        strikespot_name = shot.get("strikespot_name") or shot.get("strikespot")
        if speed_name not in settings["speed_mapping"]:
            return None
        if strikespot_name not in settings["strikespot_mapping"]:
            return None
        strikespot = settings["strikespot_mapping"][strikespot_name]
        return (
            float(settings["speed_mapping"][speed_name]),
            float(strikespot["a"]),
            float(strikespot["b"]),
        )

    def _select_target_ball(self, system: Any, *, cue_ball_id: str) -> str | None:
        candidates = []
        for ball_id in system.balls.keys():
            if ball_id == cue_ball_id:
                continue
            try:
                numeric = int(ball_id)
            except (TypeError, ValueError):
                continue
            if 1 <= numeric <= 15:
                candidates.append((numeric, ball_id))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    def _aim_at_ball(self, backend: Any, system: Any, target_ball_id: str) -> Any:
        return backend.aim.at_ball(system, target_ball_id)

    def _simulate(self, backend: Any, system: Any) -> Any:
        simulated = backend.simulate(system, inplace=False)
        return system if simulated is None else simulated

    def _extract_potted_balls(self, backend: Any, system: Any) -> list[str]:
        potted = []
        for ball_id in list(system.balls.keys()):
            events = backend.events.filter_events(
                system.events,
                backend.events.by_type(backend.EventType.BALL_POCKET),
                backend.events.by_ball(ball_id),
            )
            if events:
                potted.append(ball_id)
        return potted

    def _remove_potted_balls(self, system: Any, potted_balls: list[str]) -> None:
        remaining = OrderedDict()
        for ball_id, ball in system.balls.items():
            if ball_id not in potted_balls:
                remaining[ball_id] = ball
        system.balls = remaining

    def _ball_objects(self, system: Any) -> list[dict[str, Any]]:
        objects = []
        for ball_id, ball in system.balls.items():
            coords = getattr(ball, "xyz", [None, None])
            objects.append({"ball_id": ball_id, "x": coords[0], "y": coords[1]})
        objects.sort(key=lambda item: (item["ball_id"].isalpha(), item["ball_id"]))
        return objects

    def _ball_xy(self, ball: Any) -> tuple[float, float]:
        coords = getattr(ball, "xyz", [0.0, 0.0, 0.0])
        return float(coords[0]), float(coords[1])

    def _set_ball_xy(self, ball: Any, x: float, y: float) -> None:
        if hasattr(ball, "set_state"):
            try:
                ball.set_state(x=x, y=y)
                return
            except TypeError:
                pass

        state = getattr(ball, "state", None)
        rvw = getattr(state, "rvw", None)
        if rvw is not None:
            rvw[0][0] = float(x)
            rvw[0][1] = float(y)
            return

        coords = list(getattr(ball, "xyz", [x, y, 0.0]))
        while len(coords) < 3:
            coords.append(0.0)
        coords[0] = float(x)
        coords[1] = float(y)
        ball.xyz = coords

    def _extract_dense_timesteps(
        self,
        backend: Any,
        simulated_system: Any,
        *,
        shot: dict[str, str],
        attempt_index: int,
        time_offset: float,
        continuous_dt: float,
    ) -> list:
        if not hasattr(backend, "continuize"):
            return []

        dense_system = backend.continuize(simulated_system, dt=continuous_dt)
        if dense_system is None:
            dense_system = simulated_system
        histories = {
            ball_id: self._ball_history(ball)
            for ball_id, ball in dense_system.balls.items()
            if self._ball_history(ball)
        }
        if not histories:
            return []

        step_count = max(len(history) for history in histories.values())
        state_labels = getattr(getattr(backend, "constants", None), "state_dict", {})
        timesteps = []
        for local_index in range(step_count):
            objects = []
            pocketed_balls = []
            moving_ball_count = 0
            step_time = None

            for ball_id, ball in dense_system.balls.items():
                history = histories.get(ball_id)
                if not history:
                    continue
                state = history[min(local_index, len(history) - 1)]
                if step_time is None:
                    step_time = float(getattr(state, "t", 0.0))
                ball_object = self._ball_object_from_state(ball_id, ball, state, state_labels)
                objects.append(ball_object)
                if ball_object["motion_state"] == "pocketed":
                    pocketed_balls.append(ball_id)
                elif ball_object["motion_state"] != "stationary":
                    moving_ball_count += 1

            objects.sort(key=lambda item: (item["ball_id"].isalpha(), item["ball_id"]))
            timesteps.append(
                make_step(
                    index=len(timesteps),
                    t=(step_time or 0.0) + time_offset,
                    state={
                        "attempt_index": attempt_index,
                        "simulation_time": step_time,
                        "moving_ball_count": moving_ball_count,
                        "pocketed_balls": pocketed_balls,
                    },
                    objects=objects,
                    actions=[shot] if local_index == 0 else [],
                )
            )

        return timesteps

    def _extract_dense_events(
        self,
        simulated_system: Any,
        *,
        attempt_index: int,
        time_offset: float,
    ) -> list:
        dense_events = []
        for event in getattr(simulated_system, "events", []):
            event_type = getattr(event, "event_type", "unknown")
            event_name = getattr(event_type, "value", str(event_type))
            event_time = getattr(event, "time", None)
            dense_events.append(
                make_event(
                    f"physics_{event_name}",
                    t=(float(event_time) + time_offset) if event_time is not None else None,
                    payload={
                        "attempt_index": attempt_index,
                        "ids": list(getattr(event, "ids", ()) or ()),
                    },
                )
            )
        return dense_events

    def _ball_history(self, ball: Any) -> Any:
        continuous_history = getattr(ball, "history_cts", None)
        if continuous_history and len(continuous_history):
            return continuous_history
        discrete_history = getattr(ball, "history", None)
        if discrete_history and len(discrete_history):
            return discrete_history
        return None

    def _ball_object_from_state(
        self,
        ball_id: str,
        ball: Any,
        state: Any,
        state_labels: dict[int, str],
    ) -> dict[str, Any]:
        rvw = getattr(state, "rvw", None)
        position = rvw[0].tolist() if rvw is not None else [None, None, None]
        velocity = rvw[1].tolist() if rvw is not None else [None, None, None]
        angular_velocity = rvw[2].tolist() if rvw is not None else [None, None, None]
        motion_state_code = int(getattr(state, "s", -1))
        return {
            "kind": "ball",
            "ball_id": ball_id,
            "position": position,
            "velocity": velocity,
            "angular_velocity": angular_velocity,
            "motion_state_code": motion_state_code,
            "motion_state": state_labels.get(motion_state_code, str(motion_state_code)),
            "t": float(getattr(state, "t", 0.0)),
            "radius": float(getattr(getattr(ball, "params", None), "R", 0.0)),
        }

    def _capture_table_snapshot(
        self,
        system: Any,
        *,
        task_id: str,
        attempt_index: int,
        output_dir: object,
        capture_frames: bool,
        label: str,
    ) -> list:
        if not capture_frames:
            return []

        snapshot = {
            "attempt_index": attempt_index,
            "balls": self._ball_objects(system),
        }
        if output_dir is None:
            return [
                make_attachment(
                    "table_state",
                    f"memory://pooltool/{task_id}/attempt_{attempt_index:03d}.json",
                    label=label,
                    metadata={"ball_count": len(snapshot["balls"])},
                )
            ]

        snapshot_path = Path(str(output_dir)) / f"attempt_{attempt_index:03d}_table_state.json"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with snapshot_path.open("w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, indent=2)
        return [
            make_attachment(
                "table_state",
                snapshot_path,
                label=label,
                metadata={"ball_count": len(snapshot["balls"])},
            )
        ]

    def render_initial_scene_image(self, output_path: Path, *, view_name: str = "overhead") -> Path:
        from PIL import Image, ImageDraw, ImageFont

        backend = self._load_backend()
        system = self._create_initial_system(backend)
        table = system.table
        width_px = self.INITIAL_SCENE_WIDTH_PX
        height_px = max(384, int(round(width_px * float(table.l) / max(float(table.w), 1e-6))))
        margin = 36

        image = Image.new("RGB", (width_px, height_px), color=(22, 54, 34))
        draw = ImageDraw.Draw(image)

        table_left = margin
        table_top = margin
        table_right = width_px - margin
        table_bottom = height_px - margin
        draw.rounded_rectangle(
            [table_left, table_top, table_right, table_bottom],
            radius=16,
            fill=(33, 110, 63),
            outline=(153, 111, 62),
            width=10,
        )

        def map_xy(x: float, y: float) -> tuple[float, float]:
            norm_x = float(x) / max(float(table.w), 1e-6)
            norm_y = float(y) / max(float(table.l), 1e-6)
            px = table_left + norm_x * (table_right - table_left)
            py = table_bottom - norm_y * (table_bottom - table_top)
            return px, py

        pocket_radius = 14
        for pocket_x, pocket_y in (
            (table_left, table_top),
            (table_right, table_top),
            (table_left, (table_top + table_bottom) / 2),
            (table_right, (table_top + table_bottom) / 2),
            (table_left, table_bottom),
            (table_right, table_bottom),
        ):
            draw.ellipse(
                [pocket_x - pocket_radius, pocket_y - pocket_radius, pocket_x + pocket_radius, pocket_y + pocket_radius],
                fill=(18, 18, 18),
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
        label_font = self._load_pooltool_ball_label_font()
        for ball_id, ball in system.balls.items():
            state = getattr(ball, "state", None)
            rvw = getattr(state, "rvw", None)
            if rvw is None:
                continue
            x, y = float(rvw[0][0]), float(rvw[0][1])
            px, py = map_xy(x, y)
            radius_m = float(getattr(getattr(ball, "params", None), "R", 0.028575))
            radius_px = max(9, int(round(radius_m / max(float(table.w), 1e-6) * (table_right - table_left) * 1.05)))
            fill = ball_colors.get(str(ball_id), (200, 200, 200))
            draw.ellipse([px - radius_px, py - radius_px, px + radius_px, py + radius_px], fill=fill, outline=outline, width=2)
            if str(ball_id) != "cue":
                text = str(ball_id)
                bbox = draw.textbbox((0, 0), text, font=label_font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                text_fill = (255, 255, 255) if fill == (28, 28, 28) else (20, 20, 20)
                draw.text((px - tw / 2, py - th / 2), text, fill=text_fill, font=label_font)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return output_path

    def _load_pooltool_ball_label_font(self):
        from PIL import ImageFont

        font_candidates = (
            "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        )
        for candidate in font_candidates:
            font_path = Path(candidate)
            if not font_path.is_file():
                continue
            try:
                return ImageFont.truetype(str(font_path), size=17)
            except OSError:
                continue
        return ImageFont.load_default()

    def _render_media_artifacts(
        self,
        backend: Any,
        system: Any,
        *,
        output_dir: object,
        continuous_dt: float,
    ) -> list:
        if output_dir is None:
            return []

        del continuous_dt
        self._configure_headless_render_settings(backend)
        output_fps = self.REVIEW_VIDEO_FPS
        target_frame_count = max(1, int(round(self.REVIEW_VIDEO_SECONDS * output_fps)))
        render_fps = self._review_render_fps(system, target_frame_count=target_frame_count, output_fps=output_fps)
        root = Path(str(output_dir))
        attachments = []
        for view_name, camera_state in self._render_views():
            frame_stack, effective_render_fps = self._render_review_frame_stack(
                backend=backend,
                system=system,
                preferred_fps=render_fps,
                fallback_fps=float(output_fps),
                camera_state=self._camera_state(backend, camera_state),
                target_frame_count=target_frame_count,
            )
            initial_scene_path = root / f"initial_scene_{view_name}.png"
            video_path = root / f"simulation_{view_name}.mp4"
            initial_metadata = write_image_artifact(frame_stack[0], initial_scene_path, max_side=self.RENDER_IMAGE_MAX_SIDE)
            initial_metadata.update({"view": view_name, "camera_state": camera_state})
            video_metadata = write_video_artifact(frame_stack, video_path, fps=output_fps, max_side=self.RENDER_IMAGE_MAX_SIDE)
            video_metadata.update(
                {
                    "view": view_name,
                    "camera_state": camera_state,
                    "duration_seconds": round(video_metadata["frame_count"] / max(video_metadata["fps"], 1), 3),
                    "render_fps": round(effective_render_fps, 3),
                    "target_duration_seconds": self.REVIEW_VIDEO_SECONDS,
                }
            )
            attachments.append(
                make_attachment(
                    "initial_scene",
                    initial_scene_path,
                    label=f"Initial scene ({view_name})",
                    metadata=initial_metadata,
                )
            )
            attachments.append(
                make_attachment(
                    "simulation_video",
                    video_path,
                    label=f"Simulation video ({view_name})",
                    metadata=video_metadata,
                )
            )
        return attachments

    def _optional_render_media_artifacts(
        self,
        backend: Any,
        system: Any,
        *,
        output_dir: object,
        continuous_dt: float,
        capture_frames: bool,
    ) -> tuple[list, str | None]:
        if not capture_frames:
            return [], None
        try:
            return (
                self._render_media_artifacts(
                    backend,
                    system,
                    output_dir=output_dir,
                    continuous_dt=continuous_dt,
                ),
                None,
            )
        except Exception as error:
            debug_log(
                "pooltool.render_media.failed",
                error=str(error),
            )
            return [], str(error)

    def _render_frame_stack(
        self,
        backend: Any,
        system: Any,
        *,
        fps: int,
        camera_state: Any,
    ):
        stepper = self._get_render_stepper(backend)
        try:
            frames = backend.image.image_stack(
                system,
                stepper,
                size=(512, 320),
                fps=float(fps),
                camera_state=camera_state,
                show_hud=False,
            )
        finally:
            if hasattr(stepper, "close_scene"):
                try:
                    stepper.close_scene()
                except Exception:
                    pass
            self._destroy_active_showbase()
        return frames

    def _render_initial_frame(
        self,
        backend: Any,
        system: Any,
        *,
        camera_state: Any,
    ):
        stepper = self._get_render_stepper(backend)
        self._configure_headless_render_settings(backend)
        tex = backend.image.get_graphics_texture()
        multisystem = getattr(backend, "multisystem", None)
        if multisystem is None:
            system_module = getattr(getattr(backend, "system", None), "datatypes", None)
            multisystem = getattr(system_module, "multisystem", None)
        if multisystem is None:
            raise RuntimeError("Pooltool backend did not expose a multisystem singleton")
        try:
            multisystem.reset()
            multisystem.append(system)
            stepper.create_scene()
            backend.ani.hud.hud.destroy()
            backend.ani.camera.cam.load_state(camera_state)
            backend.ani.animate.boop(1)
            return backend.image.image_array_from_texture(tex)
        finally:
            if hasattr(stepper, "close_scene"):
                try:
                    stepper.close_scene()
                except Exception:
                    pass
            self._destroy_active_showbase()

    def _render_review_frame_stack(
        self,
        *,
        backend: Any,
        system: Any,
        preferred_fps: float,
        fallback_fps: float,
        camera_state: Any,
        target_frame_count: int,
    ) -> tuple[Any, float]:
        last_error: Exception | None = None
        for fps in (preferred_fps, fallback_fps):
            try:
                frame_stack = self._render_frame_stack(
                    backend,
                    system,
                    fps=fps,
                    camera_state=camera_state,
                )
                frame_stack = self._select_review_frames(frame_stack, target_frame_count=target_frame_count)
                if len(frame_stack) == 0:
                    raise ValueError("pooltool renderer returned no frames")
                return frame_stack, float(fps)
            except (AssertionError, IndexError, ValueError) as error:
                last_error = error
                continue
        assert last_error is not None
        raise last_error

    def _review_render_fps(
        self,
        system: Any,
        *,
        target_frame_count: int,
        output_fps: int,
    ) -> float:
        total_duration = self._simulation_duration_seconds(system)
        if total_duration <= 0:
            return float(output_fps)
        return max(1.0, min(float(output_fps), target_frame_count / total_duration))

    def _simulation_duration_seconds(self, system: Any) -> float:
        events = getattr(system, "events", None) or []
        if not events:
            return 0.0
        event_time = getattr(events[-1], "time", None)
        if event_time is None:
            return 0.0
        return max(0.0, float(event_time))

    def _select_review_frames(self, frame_stack: Any, *, target_frame_count: int):
        frame_count = len(frame_stack)
        if frame_count <= target_frame_count:
            return frame_stack
        last_index = frame_count - 1
        selected_indices: list[int] = []
        seen: set[int] = set()
        for output_index in range(target_frame_count):
            source_index = round(output_index * last_index / (target_frame_count - 1))
            if source_index in seen:
                continue
            selected_indices.append(source_index)
            seen.add(source_index)
        return frame_stack[selected_indices]

    def _get_render_stepper(self, backend: Any):
        from panda3d.core import loadPrcFileData

        render_width = self.OFFSCREEN_RENDER_WIDTH
        render_height = self.OFFSCREEN_RENDER_HEIGHT

        loadPrcFileData(
            "",
            "window-type offscreen\n"
            "audio-library-name null\n"
            "show-frame-rate-meter #f\n"
            "load-display eglGraphicsPipe\n"
            "aux-display eglGraphicsPipe\n",
        )
        loadPrcFileData("", f"win-size {render_width} {render_height}\n")

        with isolated_pooltool_home(self.resolve_project_path("")):
            os.environ.setdefault("MPLCONFIGDIR", str(self.resolve_project_path(".runtime-home") / "matplotlib"))
            os.environ.pop("DISPLAY", None)
            self._destroy_active_showbase()
            backend.ani.animate._resize_offscreen_window = lambda size: None
            stepper = backend.ani.animate.FrameStepper()
        return stepper

    def _destroy_active_showbase(self) -> None:
        try:
            from direct.showbase import ShowBaseGlobal
        except Exception:
            return
        base = getattr(ShowBaseGlobal, "base", None)
        if base is None:
            return
        try:
            base.destroy()
        except Exception:
            pass
        for attr_name in ("base",):
            if getattr(ShowBaseGlobal, attr_name, None) is base:
                setattr(ShowBaseGlobal, attr_name, None)

    def _render_views(self) -> tuple[tuple[str, str], ...]:
        return (
            ("askew", self.ASKEW_CAMERA_STATE),
            ("overhead", self.OVERHEAD_CAMERA_STATE),
        )

    def _camera_state(self, backend: Any, camera_state_name: str):
        camera_states = getattr(backend.ani.camera, "camera_states", {})
        return camera_states.get(camera_state_name, backend.ani.image.interface.DEFAULT_CAMERA)
