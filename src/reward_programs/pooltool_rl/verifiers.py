from __future__ import annotations

from typing import Any, Callable

from .library import query_trace_patterns


def _first_event_time(trace: dict[str, Any], kind: str, ids: set[str]) -> float | None:
    for event in trace.get("events", []):
        if not isinstance(event, dict) or str(event.get("kind") or "") != kind:
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        event_ids = {str(item) for item in payload.get("ids", [])}
        if ids.issubset(event_ids):
            t = event.get("t")
            return float(t) if t is not None else None
    return None


def _ball_present_at_end(trace: dict[str, Any], ball_id: str) -> bool:
    timesteps = trace.get("timesteps", [])
    if not timesteps:
        return False
    last = timesteps[-1] if isinstance(timesteps[-1], dict) else {}
    for obj in last.get("objects", []):
        if isinstance(obj, dict) and str(obj.get("ball_id") or "") == ball_id:
            return True
    return False


def _table_center(trace: dict[str, Any]) -> tuple[float, float]:
    initial_state = trace.get("initial_state") if isinstance(trace.get("initial_state"), dict) else {}
    table_width = float(initial_state.get("table_width", 0.9906))
    table_length = float(initial_state.get("table_length", 1.9812))
    return table_width / 2.0, table_length / 2.0


def _table_dimensions(trace: dict[str, Any]) -> tuple[float, float]:
    initial_state = trace.get("initial_state") if isinstance(trace.get("initial_state"), dict) else {}
    table_width = float(initial_state.get("table_width", 0.9906))
    table_length = float(initial_state.get("table_length", 1.9812))
    return table_width, table_length


def _ball_position_at_fraction(trace: dict[str, Any], ball_id: str, t_fraction: float) -> tuple[float, float] | None:
    timesteps = [item for item in trace.get("timesteps", []) if isinstance(item, dict)]
    if not timesteps:
        return None
    index = max(0, min(len(timesteps) - 1, int(round(t_fraction * (len(timesteps) - 1)))))
    target_step = timesteps[index]
    for obj in target_step.get("objects", []):
        if not isinstance(obj, dict) or str(obj.get("ball_id") or "") != ball_id:
            continue
        position = obj.get("position", [obj.get("x", 0.0), obj.get("y", 0.0), 0.0])
        return float(position[0]), float(position[1])
    return None


def _ball_near_at_end(trace: dict[str, Any], ball_id: str, x: float, y: float, radius: float) -> bool:
    position = _ball_position_at_fraction(trace, ball_id, 1.0)
    if position is None:
        return False
    dx = float(position[0]) - x
    dy = float(position[1]) - y
    return (dx * dx + dy * dy) ** 0.5 <= radius


def _pattern(trace: dict[str, Any], label: str, **criteria: Any) -> bool:
    return bool(query_trace_patterns(trace, label=label, criteria=criteria)["success"])


def verify_pot_green_clean(trace: dict[str, Any]) -> bool:
    cue_hit = _first_event_time(trace, "physics_ball_ball", {"cue", "6"})
    pot = _first_event_time(trace, "physics_ball_pocket", {"6"})
    return cue_hit is not None and pot is not None and pot > cue_hit and _ball_present_at_end(trace, "cue")


def verify_green_then_cannon_eight_then_pot(trace: dict[str, Any]) -> bool:
    cue_hit = _first_event_time(trace, "physics_ball_ball", {"cue", "6"})
    cannon = _first_event_time(trace, "physics_ball_ball", {"6", "8"})
    pot = _first_event_time(trace, "physics_ball_pocket", {"6"})
    return cue_hit is not None and cannon is not None and pot is not None and cue_hit < cannon < pot and _ball_present_at_end(trace, "cue")


def verify_green_cushion_then_pot(trace: dict[str, Any]) -> bool:
    cue_hit = _first_event_time(trace, "physics_ball_ball", {"cue", "6"})
    pot = _first_event_time(trace, "physics_ball_pocket", {"6"})
    return cue_hit is not None and pot is not None and cue_hit < pot and _pattern(trace, "cushion rebound", ball="6")


def verify_gentle_green_keep_cue_and_green(trace: dict[str, Any]) -> bool:
    cue_hit = _first_event_time(trace, "physics_ball_ball", {"cue", "6"})
    return cue_hit is not None and _pattern(trace, "gentle touch", contains=["6"]) and _ball_present_at_end(trace, "cue") and _ball_present_at_end(trace, "6")


def verify_high_angle_green_eight(trace: dict[str, Any]) -> bool:
    cue_hit = _first_event_time(trace, "physics_ball_ball", {"cue", "6"})
    return cue_hit is not None and _pattern(trace, "high angle collision", contains=["6", "8"]) and _ball_present_at_end(trace, "6") and _ball_present_at_end(trace, "8")


def verify_green_to_eight_pot_eight(trace: dict[str, Any]) -> bool:
    cue_hit = _first_event_time(trace, "physics_ball_ball", {"cue", "6"})
    cannon = _first_event_time(trace, "physics_ball_ball", {"6", "8"})
    pot = _first_event_time(trace, "physics_ball_pocket", {"8"})
    return cue_hit is not None and cannon is not None and pot is not None and cue_hit < cannon < pot and _ball_present_at_end(trace, "cue")


def verify_green_then_cue_centers(trace: dict[str, Any]) -> bool:
    cue_hit = _first_event_time(trace, "physics_ball_ball", {"cue", "6"})
    cx, cy = _table_center(trace)
    return cue_hit is not None and _pattern(trace, "ball separation", contains=["cue", "6"]) and _ball_near_at_end(trace, "cue", cx, cy, radius=0.18)


def verify_green_rolls_to_lower_half(trace: dict[str, Any]) -> bool:
    initial_state = trace.get("initial_state") if isinstance(trace.get("initial_state"), dict) else {}
    table_length = float(initial_state.get("table_length", 1.9812))
    target_y = table_length * 0.78
    return _pattern(trace, "rolling ball", ball="6") and _ball_near_at_end(trace, "6", 0.4953, target_y, radius=0.28) and _ball_present_at_end(trace, "6")


def verify_green_slides_then_slows(trace: dict[str, Any]) -> bool:
    return _pattern(trace, "sliding ball", ball="6") and _pattern(trace, "ball slowing", ball="6") and _ball_present_at_end(trace, "6")


def verify_spin_into_green(trace: dict[str, Any]) -> bool:
    cue_hit = _first_event_time(trace, "physics_ball_ball", {"cue", "6"})
    return cue_hit is not None and _pattern(trace, "spinning ball", contains=["cue"]) and _ball_present_at_end(trace, "cue")


def verify_simple_hit_green(trace: dict[str, Any]) -> bool:
    return _first_event_time(trace, "physics_ball_ball", {"cue", "6"}) is not None


def verify_simple_pot_green(trace: dict[str, Any]) -> bool:
    return _first_event_time(trace, "physics_ball_pocket", {"6"}) is not None


def verify_simple_keep_cue(trace: dict[str, Any]) -> bool:
    return _ball_present_at_end(trace, "cue")


def verify_simple_hit_green_keep_cue(trace: dict[str, Any]) -> bool:
    return verify_simple_hit_green(trace) and verify_simple_keep_cue(trace)


def verify_simple_green_hits_eight(trace: dict[str, Any]) -> bool:
    return _first_event_time(trace, "physics_ball_ball", {"6", "8"}) is not None


def verify_simple_hit_green_then_eight(trace: dict[str, Any]) -> bool:
    cue_hit = _first_event_time(trace, "physics_ball_ball", {"cue", "6"})
    second_hit = _first_event_time(trace, "physics_ball_ball", {"6", "8"})
    return cue_hit is not None and second_hit is not None and cue_hit < second_hit


def verify_simple_pot_green_keep_cue(trace: dict[str, Any]) -> bool:
    return verify_simple_pot_green(trace) and verify_simple_keep_cue(trace)


def verify_simple_hit_green_then_pot_green(trace: dict[str, Any]) -> bool:
    cue_hit = _first_event_time(trace, "physics_ball_ball", {"cue", "6"})
    pot_green = _first_event_time(trace, "physics_ball_pocket", {"6"})
    return cue_hit is not None and pot_green is not None and cue_hit < pot_green


def verify_simple_hit_green_then_pot_eight(trace: dict[str, Any]) -> bool:
    cue_hit = _first_event_time(trace, "physics_ball_ball", {"cue", "6"})
    pot_eight = _first_event_time(trace, "physics_ball_pocket", {"8"})
    return cue_hit is not None and pot_eight is not None and cue_hit < pot_eight


def verify_simple_hit_green_then_pot_green_keep_cue(trace: dict[str, Any]) -> bool:
    return verify_simple_hit_green_then_pot_green(trace) and verify_simple_keep_cue(trace)


def verify_simple_hit_green_then_pot_eight_keep_cue(trace: dict[str, Any]) -> bool:
    return verify_simple_hit_green_then_pot_eight(trace) and verify_simple_keep_cue(trace)


def verify_simple_green_hits_eight_keep_cue(trace: dict[str, Any]) -> bool:
    return verify_simple_green_hits_eight(trace) and verify_simple_keep_cue(trace)


def verify_simple_hit_green_then_eight_keep_cue(trace: dict[str, Any]) -> bool:
    return verify_simple_hit_green_then_eight(trace) and verify_simple_keep_cue(trace)


def verify_simple_hit_green_then_cue_near_center(trace: dict[str, Any]) -> bool:
    cue_hit = _first_event_time(trace, "physics_ball_ball", {"cue", "6"})
    cx, cy = _table_center(trace)
    return cue_hit is not None and _ball_near_at_end(trace, "cue", cx, cy, radius=0.18)


def verify_simple_hit_green_then_green_near_center(trace: dict[str, Any]) -> bool:
    cue_hit = _first_event_time(trace, "physics_ball_ball", {"cue", "6"})
    cx, cy = _table_center(trace)
    return cue_hit is not None and _ball_present_at_end(trace, "6") and _ball_near_at_end(trace, "6", cx, cy, radius=0.18)


def verify_simple_hit_green_then_green_center_tight(trace: dict[str, Any]) -> bool:
    cue_hit = _first_event_time(trace, "physics_ball_ball", {"cue", "6"})
    cx, cy = _table_center(trace)
    return cue_hit is not None and _ball_present_at_end(trace, "6") and _ball_near_at_end(trace, "6", cx, cy, radius=0.14)


def verify_simple_hit_green_then_green_lower_half(trace: dict[str, Any]) -> bool:
    cue_hit = _first_event_time(trace, "physics_ball_ball", {"cue", "6"})
    table_width, table_length = _table_dimensions(trace)
    target_x = table_width * 0.5
    target_y = table_length * 0.78
    return cue_hit is not None and _ball_present_at_end(trace, "6") and _ball_near_at_end(trace, "6", target_x, target_y, radius=0.22)


def verify_simple_hit_green_then_green_lower_half_tight(trace: dict[str, Any]) -> bool:
    cue_hit = _first_event_time(trace, "physics_ball_ball", {"cue", "6"})
    table_width, table_length = _table_dimensions(trace)
    target_x = table_width * 0.5
    target_y = table_length * 0.78
    return cue_hit is not None and _ball_present_at_end(trace, "6") and _ball_near_at_end(trace, "6", target_x, target_y, radius=0.14)


def verify_simple_hit_green_then_cue_left_center(trace: dict[str, Any]) -> bool:
    cue_hit = _first_event_time(trace, "physics_ball_ball", {"cue", "6"})
    table_width, table_length = _table_dimensions(trace)
    target_x = table_width * 0.25
    target_y = table_length * 0.5
    return cue_hit is not None and _ball_near_at_end(trace, "cue", target_x, target_y, radius=0.18)


def verify_simple_hit_green_then_green_left_center(trace: dict[str, Any]) -> bool:
    cue_hit = _first_event_time(trace, "physics_ball_ball", {"cue", "6"})
    table_width, table_length = _table_dimensions(trace)
    target_x = table_width * 0.25
    target_y = table_length * 0.5
    return cue_hit is not None and _ball_present_at_end(trace, "6") and _ball_near_at_end(trace, "6", target_x, target_y, radius=0.22)


def verify_simple_hit_green_then_green_right_center(trace: dict[str, Any]) -> bool:
    cue_hit = _first_event_time(trace, "physics_ball_ball", {"cue", "6"})
    table_width, table_length = _table_dimensions(trace)
    target_x = table_width * 0.75
    target_y = table_length * 0.5
    return cue_hit is not None and _ball_present_at_end(trace, "6") and _ball_near_at_end(trace, "6", target_x, target_y, radius=0.22)


def verify_simple_hit_green_then_cue_left_half(trace: dict[str, Any]) -> bool:
    cue_hit = _first_event_time(trace, "physics_ball_ball", {"cue", "6"})
    table_width, _ = _table_dimensions(trace)
    cue_position = _ball_position_at_fraction(trace, "cue", 1.0)
    return cue_hit is not None and cue_position is not None and float(cue_position[0]) < table_width * 0.5


def verify_simple_hit_green_then_cue_lower_half(trace: dict[str, Any]) -> bool:
    cue_hit = _first_event_time(trace, "physics_ball_ball", {"cue", "6"})
    _, table_length = _table_dimensions(trace)
    cue_position = _ball_position_at_fraction(trace, "cue", 1.0)
    return cue_hit is not None and cue_position is not None and float(cue_position[1]) >= table_length * 0.5


def verify_simple_hit_green_then_green_right_half(trace: dict[str, Any]) -> bool:
    cue_hit = _first_event_time(trace, "physics_ball_ball", {"cue", "6"})
    table_width, _ = _table_dimensions(trace)
    green_position = _ball_position_at_fraction(trace, "6", 1.0)
    return cue_hit is not None and green_position is not None and float(green_position[0]) >= table_width * 0.5


def verify_simple_hit_green_then_green_lower_half_wide(trace: dict[str, Any]) -> bool:
    cue_hit = _first_event_time(trace, "physics_ball_ball", {"cue", "6"})
    _, table_length = _table_dimensions(trace)
    green_position = _ball_position_at_fraction(trace, "6", 1.0)
    return cue_hit is not None and green_position is not None and float(green_position[1]) >= table_length * 0.5


VERIFIERS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "simple_hit_green": verify_simple_hit_green,
    "simple_keep_cue": verify_simple_keep_cue,
    "simple_hit_green_keep_cue": verify_simple_hit_green_keep_cue,
    "simple_pot_green": verify_simple_pot_green,
    "simple_green_hits_eight": verify_simple_green_hits_eight,
    "simple_hit_green_then_eight": verify_simple_hit_green_then_eight,
    "simple_pot_green_keep_cue": verify_simple_pot_green_keep_cue,
    "simple_hit_green_then_pot_green": verify_simple_hit_green_then_pot_green,
    "simple_hit_green_then_pot_green_keep_cue": verify_simple_hit_green_then_pot_green_keep_cue,
    "simple_hit_green_then_pot_eight": verify_simple_hit_green_then_pot_eight,
    "simple_hit_green_then_pot_eight_keep_cue": verify_simple_hit_green_then_pot_eight_keep_cue,
    "simple_green_hits_eight_keep_cue": verify_simple_green_hits_eight_keep_cue,
    "simple_hit_green_then_eight_keep_cue": verify_simple_hit_green_then_eight_keep_cue,
    "simple_hit_green_then_cue_near_center": verify_simple_hit_green_then_cue_near_center,
    "simple_hit_green_then_green_near_center": verify_simple_hit_green_then_green_near_center,
    "simple_hit_green_then_green_center_tight": verify_simple_hit_green_then_green_center_tight,
    "simple_hit_green_then_green_lower_half": verify_simple_hit_green_then_green_lower_half,
    "simple_hit_green_then_green_lower_half_tight": verify_simple_hit_green_then_green_lower_half_tight,
    "simple_hit_green_then_cue_left_center": verify_simple_hit_green_then_cue_left_center,
    "simple_hit_green_then_green_left_center": verify_simple_hit_green_then_green_left_center,
    "simple_hit_green_then_green_right_center": verify_simple_hit_green_then_green_right_center,
    "simple_hit_green_then_cue_left_half": verify_simple_hit_green_then_cue_left_half,
    "simple_hit_green_then_cue_lower_half": verify_simple_hit_green_then_cue_lower_half,
    "simple_hit_green_then_green_right_half": verify_simple_hit_green_then_green_right_half,
    "simple_hit_green_then_green_lower_half_wide": verify_simple_hit_green_then_green_lower_half_wide,
    "pot_green_clean": verify_pot_green_clean,
    "green_then_cannon_eight_then_pot": verify_green_then_cannon_eight_then_pot,
    "green_cushion_then_pot": verify_green_cushion_then_pot,
    "gentle_green_keep_cue_and_green": verify_gentle_green_keep_cue_and_green,
    "high_angle_green_eight": verify_high_angle_green_eight,
    "green_to_eight_pot_eight": verify_green_to_eight_pot_eight,
    "green_then_cue_centers": verify_green_then_cue_centers,
    "green_rolls_to_lower_half": verify_green_rolls_to_lower_half,
    "green_slides_then_slows": verify_green_slides_then_slows,
    "spin_into_green": verify_spin_into_green,
}


def evaluate_verifier(verifier_id: str | None, trace: dict[str, Any]) -> bool | None:
    if verifier_id is None:
        return None
    fn = VERIFIERS.get(verifier_id)
    if fn is None:
        raise KeyError(f"Unknown verifier id: {verifier_id}")
    return bool(fn(trace))
