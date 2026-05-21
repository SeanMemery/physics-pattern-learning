from __future__ import annotations

from dataclasses import dataclass, field

from schema import CanonicalTrace


@dataclass
class TraceQualityReport:
    trace_id: str
    usable: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_trace_quality(trace: CanonicalTrace) -> TraceQualityReport:
    errors: list[str] = []
    warnings: list[str] = []

    if not trace.initial_state:
        errors.append("initial_state is empty")
    if not trace.actions:
        errors.append("actions are empty")
    if not trace.timesteps:
        errors.append("timesteps are empty")
    if not trace.outcome.metrics:
        errors.append("outcome metrics are empty")
    if not trace.provenance.config_ref:
        errors.append("provenance.config_ref is missing")
    if not trace.provenance.runner_ref:
        errors.append("provenance.runner_ref is missing")

    if len(trace.timesteps) < 2:
        warnings.append("trace has fewer than 2 timesteps")
    if not trace.events:
        warnings.append("trace has no events")
    if not trace.attachments:
        warnings.append("trace has no top-level attachments")

    if trace.timesteps and not any(step.actions for step in trace.timesteps):
        warnings.append("trace records no step-level actions")

    for index, step in enumerate(trace.timesteps):
        if not step.state and not step.objects:
            warnings.append(f"step {index} has no state or object detail")

    _apply_environment_detail_rules(trace, errors, warnings)

    return TraceQualityReport(
        trace_id=trace.trace_id,
        usable=not errors,
        errors=errors,
        warnings=warnings,
    )


def _apply_environment_detail_rules(
    trace: CanonicalTrace,
    errors: list[str],
    warnings: list[str],
) -> None:
    detail_thresholds = {
        "phyre": {"min_timesteps": 64, "min_objects": 1, "min_motion": 1e-4},
        "iphyre": {"min_timesteps": 12, "min_objects": 1, "min_motion": 1e-4},
        "kinetix": {"min_timesteps": 24, "min_objects": 4, "min_motion": 1e-4},
        "pooltool": {"min_timesteps": 40, "min_objects": 2, "min_motion": 1e-4},
    }
    thresholds = detail_thresholds.get(trace.environment)
    if thresholds is None:
        return

    max_objects = max((len(step.objects) for step in trace.timesteps), default=0)
    steps_with_objects = sum(1 for step in trace.timesteps if step.objects)
    max_motion = _max_object_displacement(trace)

    if len(trace.timesteps) < thresholds["min_timesteps"]:
        errors.append(
            f"{trace.environment} trace has fewer than {thresholds['min_timesteps']} timesteps"
        )
    if max_objects < thresholds["min_objects"]:
        errors.append(
            f"{trace.environment} trace never exposes at least {thresholds['min_objects']} objects"
        )
    if steps_with_objects < 2:
        errors.append(f"{trace.environment} trace does not contain multi-step object state")
    if max_motion <= thresholds["min_motion"]:
        errors.append(f"{trace.environment} trace shows no meaningful object motion")
    if trace.environment == "iphyre":
        _apply_iphyre_fidelity_rules(trace, errors, warnings)


def _apply_iphyre_fidelity_rules(
    trace: CanonicalTrace,
    errors: list[str],
    warnings: list[str],
) -> None:
    initial_state = trace.initial_state
    simulation_fps = initial_state.get("simulation_fps")
    trace_fps = initial_state.get("trace_fps")
    video_fps = initial_state.get("video_fps")

    if not isinstance(simulation_fps, (int, float)):
        errors.append("iphyre trace is missing initial_state.simulation_fps")
        return
    if not isinstance(trace_fps, (int, float)):
        errors.append("iphyre trace is missing initial_state.trace_fps")
    if not isinstance(video_fps, (int, float)):
        errors.append("iphyre trace is missing initial_state.video_fps")

    minimum_simulation_fps = max(60, int(video_fps) if isinstance(video_fps, (int, float)) else 0)
    if float(simulation_fps) < minimum_simulation_fps:
        errors.append(
            f"iphyre trace uses simulation_fps={simulation_fps}, expected at least {minimum_simulation_fps}"
        )

    if isinstance(trace_fps, (int, float)) and float(simulation_fps) < float(trace_fps):
        errors.append(
            f"iphyre trace uses simulation_fps={simulation_fps} below trace_fps={trace_fps}"
        )

    persistent_overlap_steps = _count_iphyre_overlap_steps(trace)
    if persistent_overlap_steps >= 5:
        warnings.append(
            f"iphyre trace shows persistent object overlap in {persistent_overlap_steps} timesteps"
        )


def _count_iphyre_overlap_steps(trace: CanonicalTrace) -> int:
    overlap_steps = 0
    for step in trace.timesteps:
        objects = [_iphyre_bounds(obj) for obj in step.objects]
        objects = [bounds for bounds in objects if bounds is not None]
        found_overlap = False
        for index, first in enumerate(objects):
            for second in objects[index + 1 :]:
                overlap_ratio = _bounds_overlap_ratio(first, second)
                if overlap_ratio >= 0.35:
                    found_overlap = True
                    break
            if found_overlap:
                break
        if found_overlap:
            overlap_steps += 1
    return overlap_steps


def _iphyre_bounds(obj: dict[str, object]) -> tuple[float, float, float, float] | None:
    values = [obj.get("x1"), obj.get("y1"), obj.get("x2"), obj.get("y2")]
    if not all(isinstance(value, (int, float)) for value in values):
        return None
    x1, y1, x2, y2 = (float(value) for value in values)
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _bounds_overlap_ratio(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    overlap_area = (right - left) * (bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return overlap_area / max(min(first_area, second_area), 1e-9)


def _max_object_displacement(trace: CanonicalTrace) -> float:
    by_object: dict[tuple[object, ...], list[tuple[float, float]]] = {}
    for step in trace.timesteps:
        for obj in step.objects:
            xy = _extract_xy(obj)
            if xy is None:
                continue
            key = (
                obj.get("kind"),
                obj.get("object_index"),
                obj.get("ball_id"),
                obj.get("shape"),
                obj.get("color"),
            )
            by_object.setdefault(key, []).append(xy)

    max_displacement = 0.0
    for positions in by_object.values():
        if len(positions) < 2:
            continue
        origin_x, origin_y = positions[0]
        for x, y in positions[1:]:
            displacement = ((x - origin_x) ** 2 + (y - origin_y) ** 2) ** 0.5
            if displacement > max_displacement:
                max_displacement = displacement
    return max_displacement


def _extract_xy(obj: dict[str, object]) -> tuple[float, float] | None:
    position = obj.get("position")
    if isinstance(position, list) and len(position) >= 2:
        x = position[0]
        y = position[1]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return float(x), float(y)

    x = obj.get("x")
    y = obj.get("y")
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return float(x), float(y)

    x1 = obj.get("x1")
    y1 = obj.get("y1")
    x2 = obj.get("x2")
    y2 = obj.get("y2")
    if all(isinstance(value, (int, float)) for value in [x1, y1, x2, y2]):
        return (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0

    return None
