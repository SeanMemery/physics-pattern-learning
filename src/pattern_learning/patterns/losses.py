from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pattern_library_schema import Pattern
from schema import CanonicalTrace


LossKind = Literal["correlation_structure"]


class LossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: LossKind
    weight: float = Field(default=1.0)
    histogram_bins: int = Field(default=12, ge=1)
    pair_limit: int | None = Field(default=500, ge=1)

@dataclass(frozen=True)
class LossComponentScore:
    kind: str
    weight: float
    raw_score: float
    weighted_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def default_loss_configs() -> list[LossConfig]:
    return [LossConfig(kind="correlation_structure", weight=1.0, histogram_bins=12, pair_limit=500)]


def compute_correlation_structure_score(
    *,
    traces: list[CanonicalTrace],
    pattern_sets: list[list[Pattern]],
    histogram_bins: int = 12,
    pair_limit: int | None = None,
    pair_indices: list[tuple[int, int]] | None = None,
) -> tuple[float, dict[str, Any]]:
    if len(traces) != len(pattern_sets):
        raise ValueError("traces and pattern_sets must have the same length")
    if len(traces) < 2:
        return 0.0, {"pair_count": 0, "reason": "not_enough_samples"}
    if pair_indices is not None and pair_limit is not None:
        raise ValueError("pair_indices and pair_limit are mutually exclusive")
    resolved_pair_indices = (
        list(pair_indices)
        if pair_indices is not None
        else list(combinations(range(len(traces)), 2))
    )
    if pair_limit is not None:
        resolved_pair_indices = resolved_pair_indices[:pair_limit]
    raw_distances: list[float] = []
    annotation_distances: list[float] = []
    for left_index, right_index in resolved_pair_indices:
        raw_distances.append(compute_trace_distance(traces[left_index], traces[right_index]))
        annotation_distances.append(
            compute_annotation_distance(
                pattern_sets[left_index],
                pattern_sets[right_index],
                histogram_bins=histogram_bins,
                max_timestep_count=max(len(traces[left_index].timesteps), len(traces[right_index].timesteps)),
            )
        )
    corr = _pearson_correlation(raw_distances, annotation_distances)
    return corr, {
        "pair_count": len(resolved_pair_indices),
        "raw_distance_mean": _mean(raw_distances),
        "annotation_distance_mean": _mean(annotation_distances),
        "histogram_bins": histogram_bins,
    }


def select_correlation_pair_indices(
    *,
    sample_count: int,
    pair_limit: int | None,
    seed: int,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    pair_indices = list(combinations(range(sample_count), 2))
    total_pair_count = len(pair_indices)
    if pair_limit is None or pair_limit >= total_pair_count:
        return pair_indices, {
            "pair_limit": pair_limit,
            "pair_selection": "all_pairs",
            "pair_selection_count": total_pair_count,
            "pair_selection_total_count": total_pair_count,
        }
    sampled_pairs = sorted(random.Random(seed).sample(pair_indices, pair_limit))
    return sampled_pairs, {
        "pair_limit": pair_limit,
        "pair_selection": "seeded_sample",
        "pair_selection_count": len(sampled_pairs),
        "pair_selection_seed": seed,
        "pair_selection_total_count": total_pair_count,
    }


def compute_trace_distance(left: CanonicalTrace, right: CanonicalTrace) -> float:
    if left.environment != right.environment:
        raise ValueError(
            f"trace distance requires matching environments, got {left.environment} and {right.environment}"
        )
    if left.environment == "phyre":
        return compute_phyre_trace_distance(left, right)
    if left.environment == "iphyre":
        return compute_iphyre_trace_distance(left, right)
    if left.environment == "kinetix":
        return compute_kinetix_trace_distance(left, right)
    if left.environment == "pooltool":
        return compute_pooltool_trace_distance(left, right)
    raise ValueError(f"unsupported environment for trace distance: {left.environment}")


def compute_phyre_trace_distance(left: CanonicalTrace, right: CanonicalTrace) -> float:
    return _compute_grouped_trace_distance_with_rules(
        left,
        right,
        signature_fn=_phyre_object_signature,
        position_fn=_xy_position,
    )


def compute_iphyre_trace_distance(left: CanonicalTrace, right: CanonicalTrace) -> float:
    return _compute_grouped_trace_distance_with_rules(
        left,
        right,
        signature_fn=_iphyre_object_signature,
        position_fn=_iphyre_center_position,
    )


def compute_kinetix_trace_distance(left: CanonicalTrace, right: CanonicalTrace) -> float:
    return _compute_grouped_trace_distance_with_rules(
        left,
        right,
        signature_fn=_kinetix_object_signature,
        position_fn=_kinetix_position,
    )


def compute_pooltool_trace_distance(left: CanonicalTrace, right: CanonicalTrace) -> float:
    return _compute_trace_distance_with_rules(
        left,
        right,
        object_key_fn=_pooltool_object_key,
        position_fn=_pooltool_position,
    )


def _compute_trace_distance_with_rules(
    left: CanonicalTrace,
    right: CanonicalTrace,
    *,
    object_key_fn,
    position_fn,
) -> float:
    common_steps = min(len(left.timesteps), len(right.timesteps))
    if common_steps == 0:
        return 0.0

    all_positions: list[tuple[float, float]] = []
    step_distances: list[float] = []
    for step_index in range(common_steps):
        left_objects = _object_map(
            left.timesteps[step_index].objects,
            object_key_fn=object_key_fn,
            position_fn=position_fn,
        )
        right_objects = _object_map(
            right.timesteps[step_index].objects,
            object_key_fn=object_key_fn,
            position_fn=position_fn,
        )
        shared_ids = sorted(set(left_objects) & set(right_objects))
        frame_distances: list[float] = []
        for object_id in shared_ids:
            left_pos = left_objects[object_id]
            right_pos = right_objects[object_id]
            all_positions.extend([left_pos, right_pos])
            frame_distances.append(math.dist(left_pos, right_pos))
        if frame_distances:
            step_distances.append(sum(frame_distances) / len(frame_distances))

    if not step_distances:
        return 0.0
    xs = [pos[0] for pos in all_positions]
    ys = [pos[1] for pos in all_positions]
    denom = math.dist((min(xs), min(ys)), (max(xs), max(ys))) if all_positions else 1.0
    if denom <= 1e-9:
        denom = 1.0
    return sum(step_distances) / len(step_distances) / denom


def _compute_grouped_trace_distance_with_rules(
    left: CanonicalTrace,
    right: CanonicalTrace,
    *,
    signature_fn,
    position_fn,
) -> float:
    common_steps = min(len(left.timesteps), len(right.timesteps))
    if common_steps == 0:
        return 0.0

    all_positions: list[tuple[float, float]] = []
    step_distances: list[float] = []
    for step_index in range(common_steps):
        left_groups = _grouped_object_positions(
            left.timesteps[step_index].objects,
            signature_fn=signature_fn,
            position_fn=position_fn,
        )
        right_groups = _grouped_object_positions(
            right.timesteps[step_index].objects,
            signature_fn=signature_fn,
            position_fn=position_fn,
        )
        shared_signatures = sorted(set(left_groups) & set(right_groups))
        frame_distances: list[float] = []
        for signature in shared_signatures:
            left_positions = sorted(left_groups[signature])
            right_positions = sorted(right_groups[signature])
            pair_count = min(len(left_positions), len(right_positions))
            for pair_index in range(pair_count):
                left_pos = left_positions[pair_index]
                right_pos = right_positions[pair_index]
                all_positions.extend([left_pos, right_pos])
                frame_distances.append(math.dist(left_pos, right_pos))
        if frame_distances:
            step_distances.append(sum(frame_distances) / len(frame_distances))

    if not step_distances:
        return 0.0
    xs = [pos[0] for pos in all_positions]
    ys = [pos[1] for pos in all_positions]
    denom = math.dist((min(xs), min(ys)), (max(xs), max(ys))) if all_positions else 1.0
    if denom <= 1e-9:
        denom = 1.0
    return sum(step_distances) / len(step_distances) / denom


def compute_annotation_distance(
    left_patterns: list[Pattern],
    right_patterns: list[Pattern],
    *,
    histogram_bins: int,
    max_timestep_count: int,
) -> float:
    left_hist = _event_histogram(
        left_patterns,
        histogram_bins=histogram_bins,
        max_timestep_count=max_timestep_count,
    )
    right_hist = _event_histogram(
        right_patterns,
        histogram_bins=histogram_bins,
        max_timestep_count=max_timestep_count,
    )
    return 0.5 * (_cross_entropy(left_hist, right_hist) + _cross_entropy(right_hist, left_hist))


def _object_map(
    objects: list[dict[str, Any]],
    *,
    object_key_fn,
    position_fn,
) -> dict[tuple[Any, ...], tuple[float, float]]:
    mapped: dict[tuple[Any, ...], tuple[float, float]] = {}
    for index, obj in enumerate(objects):
        key = object_key_fn(obj, index)
        if key is None:
            continue
        pos = position_fn(obj)
        if pos is None:
            continue
        mapped[key] = pos
    return mapped


def _grouped_object_positions(
    objects: list[dict[str, Any]],
    *,
    signature_fn,
    position_fn,
) -> dict[tuple[Any, ...], list[tuple[float, float]]]:
    grouped: dict[tuple[Any, ...], list[tuple[float, float]]] = {}
    for index, obj in enumerate(objects):
        signature = signature_fn(obj, index)
        if signature is None:
            continue
        pos = position_fn(obj)
        if pos is None:
            continue
        grouped.setdefault(signature, []).append(pos)
    return grouped


def _phyre_object_signature(obj: dict[str, Any], index: int) -> tuple[Any, ...]:
    return (
        "phyre",
        str(obj.get("shape", "unknown")),
        str(obj.get("color", "unknown")),
        bool(obj.get("is_user_input", False)),
    )


def _iphyre_object_signature(obj: dict[str, Any], index: int) -> tuple[Any, ...]:
    return (
        "iphyre",
        round(float(obj.get("radius", 0.0)), 3),
        bool(obj.get("dynamic", False)),
        bool(obj.get("eliminable", False)),
        bool(obj.get("joint", False)),
        bool(obj.get("spring", False)),
    )


def _kinetix_object_signature(obj: dict[str, Any], index: int) -> tuple[Any, ...]:
    return (
        "kinetix",
        str(obj.get("kind", "unknown")),
        int(obj.get("shape_role", -1)),
        bool(obj.get("highlighted", False)),
        round(float(obj.get("inverse_mass", 0.0)), 3),
        round(float(obj.get("radius", 0.0)), 3),
        len(obj.get("vertices", [])) if isinstance(obj.get("vertices"), list) else 0,
    )


def _pooltool_object_key(obj: dict[str, Any], index: int) -> tuple[Any, ...]:
    return ("pooltool", str(obj.get("ball_id", index)))


def _xy_position(obj: dict[str, Any]) -> tuple[float, float] | None:
    x = obj.get("x")
    y = obj.get("y")
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return float(x), float(y)
    return None


def _iphyre_center_position(obj: dict[str, Any]) -> tuple[float, float] | None:
    x1 = obj.get("x1")
    y1 = obj.get("y1")
    x2 = obj.get("x2")
    y2 = obj.get("y2")
    if all(isinstance(value, (int, float)) for value in [x1, y1, x2, y2]):
        return (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0
    return None


def _kinetix_position(obj: dict[str, Any]) -> tuple[float, float] | None:
    position = obj.get("position")
    if isinstance(position, list) and len(position) >= 2:
        x, y = position[0], position[1]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return float(x), float(y)
    global_position = obj.get("global_position")
    if isinstance(global_position, list) and len(global_position) >= 2:
        x, y = global_position[0], global_position[1]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return float(x), float(y)
    return None


def _pooltool_position(obj: dict[str, Any]) -> tuple[float, float] | None:
    if "x" in obj and "y" in obj:
        return _xy_position(obj)
    position = obj.get("position")
    if isinstance(position, list) and len(position) >= 2:
        x, y = position[0], position[1]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return float(x), float(y)
    return None


def _event_histogram(
    patterns: list[Pattern],
    *,
    histogram_bins: int,
    max_timestep_count: int,
) -> list[float]:
    counts = [0.0 for _ in range(histogram_bins)]
    if max_timestep_count <= 0:
        raise ValueError("max_timestep_count must be > 0 for span-aware annotation distance")
    for pattern in patterns:
        start_step, end_step = _pattern_step_span(pattern, max_timestep_count=max_timestep_count)
        for bucket in _overlapping_buckets(
            start=start_step,
            end=end_step + 1.0,
            bucket_count=histogram_bins,
            domain_size=float(max(max_timestep_count, 1)),
        ):
            counts[bucket] += 1.0
    smoothing = 1e-6
    total = sum(counts) + smoothing * histogram_bins
    return [(count + smoothing) / total for count in counts]


def _pattern_step_span(pattern: Pattern, *, max_timestep_count: int) -> tuple[float, float]:
    start = float(pattern.start_timestep)
    end = float(pattern.end_timestep) + 1.0
    return _clamp_interval(start, end, domain_size=float(max(max_timestep_count, 1)))


def _overlapping_buckets(
    *,
    start: float,
    end: float,
    bucket_count: int,
    domain_size: float,
) -> list[int]:
    if bucket_count <= 0:
        return [0]
    domain_size = max(domain_size, 1e-6)
    start = min(max(start, 0.0), domain_size)
    end = min(max(end, start + 1e-6), domain_size)
    bucket_width = domain_size / bucket_count
    start_bucket = min(int(start / bucket_width), bucket_count - 1)
    end_bucket = min(int((end - 1e-9) / bucket_width), bucket_count - 1)
    return list(range(start_bucket, end_bucket + 1))


def _clamp_interval(start: float, end: float, *, domain_size: float) -> tuple[float, float]:
    domain_size = max(domain_size, 1e-6)
    start = min(max(start, 0.0), domain_size)
    end = min(max(end, start + 1e-6), domain_size)
    return start, end


def _cross_entropy(p: list[float], q: list[float]) -> float:
    return -sum(pi * math.log(qi) for pi, qi in zip(p, q))


def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denom_x = math.sqrt(sum(value * value for value in centered_x))
    denom_y = math.sqrt(sum(value * value for value in centered_y))
    if denom_x <= 1e-9 or denom_y <= 1e-9:
        return 0.0
    numer = sum(x * y for x, y in zip(centered_x, centered_y))
    return numer / (denom_x * denom_y)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
