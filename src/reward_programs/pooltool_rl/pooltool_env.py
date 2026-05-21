from __future__ import annotations

import importlib
import json
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .media import write_image_artifact, write_video_artifact
from .schemas import ActionSpec, SceneSpec


class PooltoolEngine:
    ASKEW_CAMERA_STATE = "7_foot_offcenter"
    OVERHEAD_CAMERA_STATE = "7_foot_overhead"
    REVIEW_VIDEO_FPS = 20
    REVIEW_VIDEO_SECONDS = 5.0
    RENDER_IMAGE_MAX_SIDE = 1536
    OFFSCREEN_RENDER_WIDTH = 1920
    OFFSCREEN_RENDER_HEIGHT = 1200
    TABLE_MESH_PITCH_DEGREES = 90.0
    def __init__(self) -> None:
        self.backend = self._load_backend()

    def create_system_from_scene(self, scene: SceneSpec) -> Any:
        system = self._copy_system(self._create_initial_system())
        ball_map = {str(ball_id): ball for ball_id, ball in getattr(system, "balls", {}).items()}
        for item in scene.initial_ball_positions:
            ball = ball_map.get(item.ball_id)
            if ball is not None:
                self._set_ball_xy(ball, item.x, item.y)
        self._ensure_cue_ball_present(system)
        return system

    def action_bounds(self, scene: SceneSpec) -> tuple[float, float, float, float, float, float]:
        speeds = [float(value) for value in scene.speed_mapping.values()] or [0.2, 5.0]
        a_values = [float(item.get("a", 0.0)) for item in scene.strikespot_mapping.values()] or [-0.5, 0.5]
        b_values = [float(item.get("b", 0.0)) for item in scene.strikespot_mapping.values()] or [-0.5, 0.5]
        return min(speeds), max(speeds), min(a_values), max(a_values), min(b_values), max(b_values)

    def sample_action(self, scene: SceneSpec, *, seed: int) -> ActionSpec:
        speed_min, speed_max, a_min, a_max, b_min, b_max = self.action_bounds(scene)
        rng = random.Random(seed)
        return ActionSpec(
            speed=rng.uniform(speed_min, speed_max),
            strikespot_a=rng.uniform(a_min, a_max),
            strikespot_b=rng.uniform(b_min, b_max),
        )

    def simulate_action(self, scene: SceneSpec, action: ActionSpec) -> dict[str, Any]:
        system = self.create_system_from_scene(scene)
        cue_ball_id = system.cue.cue_ball_id
        self._ensure_cue_ball_present(system)
        if hasattr(system, "stop_balls"):
            system.stop_balls()
        if hasattr(system, "reset_history"):
            system.reset_history()
        target_ball_id = self._select_target_ball(system, cue_ball_id=cue_ball_id)
        if target_ball_id is None:
            return self._fallback_trace(scene, action, terminal_status="NO_TARGET_BALLS_LEFT")
        system.cue.set_state(
            V0=float(action.speed),
            phi=self._aim_at_ball(system, target_ball_id),
            a=float(action.strikespot_a),
            b=float(action.strikespot_b),
        )
        try:
            simulated_system = self._simulate(system)
            timesteps = self._extract_dense_timesteps(simulated_system, action=action.model_dump())
            events = self._extract_dense_events(simulated_system)
            potted_balls = self._extract_potted_balls(simulated_system)
            if "9" in potted_balls:
                success = True
                terminal_status = "9_BALL_POTTED"
            elif cue_ball_id in potted_balls:
                success = False
                terminal_status = "CUE_BALL_POTTED"
            elif potted_balls:
                success = False
                terminal_status = "OTHER_BALLS_POTTED"
            else:
                success = False
                terminal_status = "NO_BALLS_POTTED"
            trace = {
                "environment": "pooltool",
                "task_id": scene.sample_id,
                "initial_state": {
                    "initial_ball_positions": [item.model_dump() for item in scene.initial_ball_positions],
                    "table_width": scene.table_width,
                    "table_length": scene.table_length,
                },
                "actions": [action.model_dump()],
                "timesteps": timesteps,
                "events": events,
                "outcome": {
                    "metrics": {
                        "attempts": 1,
                        "potted_balls": potted_balls,
                        "terminal_status": terminal_status,
                        "nine_ball_success": success,
                    },
                    "success": success,
                    "terminal_status": terminal_status,
                },
            }
            return trace
        except Exception as exc:
            return self._fallback_trace(scene, action, terminal_status=f"SIMULATION_ERROR: {exc}")

    def render_shot(self, scene: SceneSpec, action: ActionSpec, output_dir: Path) -> dict[str, Path]:
        system = self.create_system_from_scene(scene)
        cue_ball_id = system.cue.cue_ball_id
        self._ensure_cue_ball_present(system)
        target_ball_id = self._select_target_ball(system, cue_ball_id=cue_ball_id)
        if target_ball_id is None:
            raise RuntimeError("No target ball remained for rendering")
        system.cue.set_state(
            V0=float(action.speed),
            phi=self._aim_at_ball(system, target_ball_id),
            a=float(action.strikespot_a),
            b=float(action.strikespot_b),
        )
        simulated_system = self._simulate(system)
        attachments = self._render_media_artifacts(simulated_system, output_dir=output_dir)
        mapping = {item["kind"] + ":" + item["view"]: item["path"] for item in attachments}
        return {
            "initial_overhead": mapping["initial_scene:overhead"],
            "simulation_overhead": mapping["simulation_video:overhead"],
        }

    def _load_backend(self) -> Any:
        backend = importlib.import_module("pooltool")
        self._patch_ball_render_texture_lookup()
        self._patch_table_render_model_transform()
        return backend

    def _patch_ball_render_texture_lookup(self) -> None:
        render_module = importlib.import_module("pooltool.objects.ball.render")
        original = render_module.BallRender.init_sphere
        if getattr(original, "_pooltool_rl_texture_guard", False):
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

        patched._pooltool_rl_texture_guard = True
        render_module.BallRender.init_sphere = patched

    def _patch_table_render_model_transform(self) -> None:
        render_module = importlib.import_module("pooltool.objects.table.render")
        table_datatypes = importlib.import_module("pooltool.objects.table.datatypes")
        globals_module = importlib.import_module("pooltool.ani.globals")
        original = render_module.TableRender.init_table
        if getattr(original, "_pooltool_rl_table_mesh_fix", False):
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

        patched._pooltool_rl_table_mesh_fix = True
        render_module.TableRender.init_table = patched

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

    def _create_initial_system(self) -> Any:
        table = self.backend.Table.default()
        balls = self.backend.get_rack(self.backend.GameType.NINEBALL, table)
        cue = self.backend.Cue(cue_ball_id="cue")
        return self.backend.System(table=table, balls=OrderedDict(balls), cue=cue)

    def _copy_system(self, system: Any) -> Any:
        return system.copy() if hasattr(system, "copy") else system

    def _ensure_cue_ball_present(self, system: Any) -> bool:
        cue_ball_id = system.cue.cue_ball_id
        if cue_ball_id in system.balls:
            return False
        new_ball = self.backend.Ball.create(cue_ball_id, xy=[system.table.w / 2, system.table.l / 4])
        system.balls[cue_ball_id] = new_ball
        return True

    def _set_ball_xy(self, ball: Any, x: float, y: float) -> None:
        if hasattr(ball, "set_state"):
            try:
                ball.set_state(x=float(x), y=float(y))
                return
            except TypeError:
                pass
        state = getattr(ball, "state", None)
        rvw = getattr(state, "rvw", None)
        if rvw is not None:
            rvw[0][0] = float(x)
            rvw[0][1] = float(y)

    def _select_target_ball(self, system: Any, *, cue_ball_id: str) -> str | None:
        candidates = []
        for ball_id in system.balls:
            if ball_id == cue_ball_id:
                continue
            try:
                numeric = int(ball_id)
            except (TypeError, ValueError):
                continue
            if 1 <= numeric <= 15:
                candidates.append((numeric, str(ball_id)))
        candidates.sort()
        return candidates[0][1] if candidates else None

    def _aim_at_ball(self, system: Any, target_ball_id: str) -> Any:
        return self.backend.aim.at_ball(system, target_ball_id)

    def _simulate(self, system: Any) -> Any:
        simulated = self.backend.simulate(system, inplace=False)
        return system if simulated is None else simulated

    def _extract_potted_balls(self, system: Any) -> list[str]:
        potted = []
        for ball_id in list(system.balls.keys()):
            events = self.backend.events.filter_events(
                system.events,
                self.backend.events.by_type(self.backend.EventType.BALL_POCKET),
                self.backend.events.by_ball(ball_id),
            )
            if events:
                potted.append(ball_id)
        return potted

    def _extract_dense_timesteps(self, simulated_system: Any, *, action: dict[str, Any], continuous_dt: float = 0.05) -> list[dict[str, Any]]:
        dense_system = self.backend.continuize(simulated_system, dt=continuous_dt)
        dense_system = simulated_system if dense_system is None else dense_system
        histories = {
            ball_id: history
            for ball_id, history in ((ball_id, getattr(ball, "history_cts", None) or getattr(ball, "history", None)) for ball_id, ball in dense_system.balls.items())
            if history
        }
        if not histories:
            return []
        state_labels = getattr(getattr(self.backend, "constants", None), "state_dict", {})
        step_count = max(len(history) for history in histories.values())
        timesteps: list[dict[str, Any]] = []
        for local_index in range(step_count):
            objects = []
            step_time = None
            for ball_id, ball in dense_system.balls.items():
                history = histories.get(ball_id)
                if not history:
                    continue
                state = history[min(local_index, len(history) - 1)]
                rvw = getattr(state, "rvw", None)
                position = rvw[0].tolist() if rvw is not None else [None, None, None]
                motion_state_code = int(getattr(state, "s", -1))
                if step_time is None:
                    step_time = float(getattr(state, "t", 0.0))
                objects.append(
                    {
                        "ball_id": str(ball_id),
                        "position": position,
                        "motion_state_code": motion_state_code,
                        "motion_state": state_labels.get(motion_state_code, str(motion_state_code)),
                        "t": float(getattr(state, "t", 0.0)),
                    }
                )
            objects.sort(key=lambda item: (item["ball_id"].isalpha(), item["ball_id"]))
            timesteps.append(
                {
                    "index": len(timesteps),
                    "t": step_time or 0.0,
                    "state": {"attempt_index": 0},
                    "objects": objects,
                    "actions": [action] if local_index == 0 else [],
                    "attachments": [],
                }
            )
        return timesteps

    def _extract_dense_events(self, simulated_system: Any) -> list[dict[str, Any]]:
        dense_events = []
        for event in getattr(simulated_system, "events", []):
            event_type = getattr(event, "event_type", "unknown")
            event_name = getattr(event_type, "value", str(event_type))
            event_time = getattr(event, "time", None)
            dense_events.append(
                {
                    "t": float(event_time) if event_time is not None else None,
                    "kind": f"physics_{event_name}",
                    "payload": {"attempt_index": 0, "ids": list(getattr(event, "ids", ()) or ())},
                }
            )
        return dense_events

    def _render_media_artifacts(self, system: Any, *, output_dir: Path) -> list[dict[str, Any]]:
        self._configure_headless_render_settings()
        output_fps = self.REVIEW_VIDEO_FPS
        target_frame_count = max(1, int(round(self.REVIEW_VIDEO_SECONDS * output_fps)))
        render_fps = self._review_render_fps(system, target_frame_count=target_frame_count, output_fps=output_fps)
        attachments: list[dict[str, Any]] = []
        for view_name, camera_state in self._render_views():
            frame_stack, effective_render_fps = self._render_review_frame_stack(
                system=system,
                preferred_fps=render_fps,
                fallback_fps=float(output_fps),
                camera_state=self._camera_state(camera_state),
                target_frame_count=target_frame_count,
            )
            initial_scene_path = output_dir / f"initial_scene_{view_name}.png"
            video_path = output_dir / f"simulation_{view_name}.mp4"
            write_image_artifact(frame_stack[0], initial_scene_path, max_side=self.RENDER_IMAGE_MAX_SIDE)
            write_video_artifact(frame_stack, video_path, fps=output_fps, max_side=self.RENDER_IMAGE_MAX_SIDE)
            attachments.append({"kind": "initial_scene", "view": view_name, "path": initial_scene_path})
            attachments.append({"kind": "simulation_video", "view": view_name, "path": video_path, "render_fps": effective_render_fps})
        return attachments

    def _configure_headless_render_settings(self) -> None:
        graphics_settings = getattr(getattr(getattr(self.backend, "config", None), "settings", None), "graphics", None)
        if graphics_settings is None:
            graphics_settings = self._load_pooltool_graphics_settings()
        if graphics_settings is None:
            return
        if isinstance(graphics_settings, dict):
            graphics_settings["room"] = False
            graphics_settings["floor"] = False
            return
        for name, value in {"room": False, "floor": False}.items():
            if hasattr(graphics_settings, name):
                setattr(graphics_settings, name, value)

    def _render_review_frame_stack(self, *, system: Any, preferred_fps: float, fallback_fps: float, camera_state: Any, target_frame_count: int) -> tuple[Any, float]:
        last_error: Exception | None = None
        for fps in (preferred_fps, fallback_fps):
            try:
                frame_stack = self._render_frame_stack(system=system, fps=fps, camera_state=camera_state)
                frame_stack = self._select_review_frames(frame_stack, target_frame_count=target_frame_count)
                if len(frame_stack) == 0:
                    raise ValueError("renderer returned no frames")
                return frame_stack, float(fps)
            except Exception as error:
                last_error = error
                continue
        assert last_error is not None
        raise last_error

    def _review_render_fps(self, system: Any, *, target_frame_count: int, output_fps: int) -> float:
        events = getattr(system, "events", None) or []
        total_duration = max(0.0, float(getattr(events[-1], "time", 0.0))) if events else 0.0
        if total_duration <= 0:
            return float(output_fps)
        return max(1.0, min(float(output_fps), target_frame_count / total_duration))

    def _render_frame_stack(self, *, system: Any, fps: float, camera_state: Any):
        from panda3d.core import loadPrcFileData

        loadPrcFileData("", "window-type offscreen\naudio-library-name null\nshow-frame-rate-meter #f\nload-display eglGraphicsPipe\naux-display eglGraphicsPipe\n")
        loadPrcFileData("", f"win-size {self.OFFSCREEN_RENDER_WIDTH} {self.OFFSCREEN_RENDER_HEIGHT}\n")
        self._destroy_active_showbase()
        self.backend.ani.animate._resize_offscreen_window = lambda size: None
        stepper = self.backend.ani.animate.FrameStepper()
        try:
            frames = self.backend.image.image_stack(
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

    def _select_review_frames(self, frame_stack: Any, *, target_frame_count: int):
        frame_count = len(frame_stack)
        if frame_count <= target_frame_count:
            return frame_stack
        last_index = frame_count - 1
        selected_indices = []
        seen: set[int] = set()
        for output_index in range(target_frame_count):
            source_index = round(output_index * last_index / (target_frame_count - 1))
            if source_index in seen:
                continue
            selected_indices.append(source_index)
            seen.add(source_index)
        return frame_stack[selected_indices]

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
        if getattr(ShowBaseGlobal, "base", None) is base:
            setattr(ShowBaseGlobal, "base", None)

    def _render_views(self) -> tuple[tuple[str, str], ...]:
        return (("overhead", self.OVERHEAD_CAMERA_STATE),)

    def _camera_state(self, camera_state_name: str):
        camera_states = getattr(self.backend.ani.camera, "camera_states", {})
        return camera_states.get(camera_state_name, self.backend.ani.image.interface.DEFAULT_CAMERA)

    def _fallback_trace(self, scene: SceneSpec, action: ActionSpec, *, terminal_status: str) -> dict[str, Any]:
        return {
            "environment": "pooltool",
            "task_id": scene.sample_id,
            "initial_state": {
                "initial_ball_positions": [item.model_dump() for item in scene.initial_ball_positions],
                "table_width": scene.table_width,
                "table_length": scene.table_length,
            },
            "actions": [action.model_dump()],
            "timesteps": [],
            "events": [],
            "outcome": {"metrics": {"terminal_status": terminal_status}, "success": False, "terminal_status": terminal_status},
        }
