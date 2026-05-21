from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import project_root


def supervised_library_path() -> Path:
    return project_root() / "data" / "libraries" / "pooltool" / "supervised" / "pattern_library.json"


def _speed_sq(obj: dict[str, Any]) -> float:
    velocity = obj.get("velocity", [0.0, 0.0, 0.0])
    if not isinstance(velocity, list) or len(velocity) < 3:
        return 0.0
    return float(velocity[0]) ** 2 + float(velocity[1]) ** 2 + float(velocity[2]) ** 2


def _iter_ball_timesteps(trace: dict[str, Any]) -> dict[str, list[tuple[Any, dict[str, Any]]]]:
    balls: dict[str, list[tuple[Any, dict[str, Any]]]] = {}
    for i, frame in enumerate(trace.get("timesteps", [])):
        if not isinstance(frame, dict):
            continue
        idx = frame.get("index", frame.get("t", i))
        for obj in frame.get("objects", []):
            if not isinstance(obj, dict):
                continue
            bid = obj.get("ball_id")
            if bid is None:
                continue
            balls.setdefault(str(bid), []).append((idx, obj))
    return balls


def _repair_detector_primitive_005_s03(trace: dict[str, Any]) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for ball_id, seq in _iter_ball_timesteps(trace).items():
        seq.sort(key=lambda item: item[0])
        start_idx = None
        start_obj = None
        last_idx = None
        last_obj = None
        for idx, obj in seq:
            is_rolling = str(obj.get("motion_state") or "").lower() == "rolling"
            if is_rolling:
                if start_idx is None:
                    start_idx = idx
                    start_obj = obj
                last_idx = idx
                last_obj = obj
                continue
            if start_idx is not None and last_obj is not None:
                patterns.append(
                    {
                        "label": "rolling ball",
                        "start_timestep": start_idx,
                        "end_timestep": last_idx,
                        "parameters": {
                            "object_details": {
                                ball_id: {
                                    "role": "rolling ball",
                                    "identity": f"ball {ball_id}",
                                    "kind": "billiard ball",
                                    "start_timestep": start_idx,
                                    "end_timestep": last_idx,
                                    "position_start": (start_obj or {}).get("position"),
                                    "position_end": last_obj.get("position"),
                                }
                            }
                        },
                    }
                )
                start_idx = None
                start_obj = None
                last_idx = None
                last_obj = None
        if start_idx is not None and last_obj is not None:
            patterns.append(
                {
                    "label": "rolling ball",
                    "start_timestep": start_idx,
                    "end_timestep": last_idx,
                    "parameters": {
                        "object_details": {
                            ball_id: {
                                "role": "rolling ball",
                                "identity": f"ball {ball_id}",
                                "kind": "billiard ball",
                                "start_timestep": start_idx,
                                "end_timestep": last_idx,
                                "position_start": (start_obj or {}).get("position"),
                                "position_end": last_obj.get("position"),
                            }
                        }
                    },
                }
            )
    return patterns


def _repair_detector_primitive_008_s07(trace: dict[str, Any]) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for ball_id, seq in _iter_ball_timesteps(trace).items():
        seq.sort(key=lambda item: item[0])
        start_idx = None
        start_obj = None
        last_idx = None
        last_obj = None
        start_speed_sq = None
        last_speed_sq = None
        for idx, obj in seq:
            motion_state = str(obj.get("motion_state") or "").lower()
            moving = motion_state in {"sliding", "rolling", "spinning"}
            speed_sq = _speed_sq(obj)
            if moving and speed_sq > 1e-6:
                if start_idx is None:
                    start_idx = idx
                    start_obj = obj
                    start_speed_sq = speed_sq
                last_idx = idx
                last_obj = obj
                last_speed_sq = speed_sq
                continue
            if start_idx is not None and last_obj is not None and start_speed_sq is not None and last_speed_sq is not None:
                if last_speed_sq <= start_speed_sq and start_idx != last_idx:
                    patterns.append(
                        {
                            "label": "ball slowing",
                            "start_timestep": start_idx,
                            "end_timestep": last_idx,
                            "parameters": {
                                "object_details": {
                                    ball_id: {
                                        "role": "slowing ball",
                                        "kind": "ball",
                                        "start_end_timestep": [start_idx, last_idx],
                                        "location_details": {
                                            "start_position": list((start_obj or {}).get("position", [0.0, 0.0, 0.0])),
                                            "end_position": list(last_obj.get("position", [0.0, 0.0, 0.0])),
                                        },
                                    }
                                },
                                "provenance": "Continuous motion interval detected with non-increasing velocity magnitude.",
                            },
                        }
                    )
                start_idx = None
                start_obj = None
                last_idx = None
                last_obj = None
                start_speed_sq = None
                last_speed_sq = None
        if start_idx is not None and last_obj is not None and start_speed_sq is not None and last_speed_sq is not None:
            if last_speed_sq <= start_speed_sq and start_idx != last_idx:
                patterns.append(
                    {
                        "label": "ball slowing",
                        "start_timestep": start_idx,
                        "end_timestep": last_idx,
                        "parameters": {
                            "object_details": {
                                ball_id: {
                                    "role": "slowing ball",
                                    "kind": "ball",
                                    "start_end_timestep": [start_idx, last_idx],
                                    "location_details": {
                                        "start_position": list((start_obj or {}).get("position", [0.0, 0.0, 0.0])),
                                        "end_position": list(last_obj.get("position", [0.0, 0.0, 0.0])),
                                    },
                                }
                            },
                            "provenance": "Continuous motion interval detected with non-increasing velocity magnitude.",
                        },
                    }
                )
    return patterns


def _repair_detector_primitive_010_s04(trace: dict[str, Any]) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    balls = _iter_ball_timesteps(trace)
    stationary_threshold_sq = 1e-6
    proximity_sq = 0.18 ** 2
    min_turn_cos = 0.92

    for ball_id, seq in balls.items():
        seq.sort(key=lambda item: item[0])
        if len(seq) < 3:
            continue
        for i in range(2, len(seq)):
            idx0, obj0 = seq[i - 2]
            idx1, obj1 = seq[i - 1]
            idx2, obj2 = seq[i]
            p0 = obj0.get("position", [0.0, 0.0, 0.0])
            p1 = obj1.get("position", [0.0, 0.0, 0.0])
            p2 = obj2.get("position", [0.0, 0.0, 0.0])
            v1 = [float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])]
            v2 = [float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1])]
            mag1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
            mag2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5
            if mag1 < 1e-6 or mag2 < 1e-6:
                continue
            cos_turn = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)
            if cos_turn > min_turn_cos:
                continue

            nearby_ball_id = None
            for other_ball_id, other_seq in balls.items():
                if other_ball_id == ball_id or i >= len(other_seq):
                    continue
                _, other_obj = other_seq[i]
                if _speed_sq(other_obj) > stationary_threshold_sq:
                    continue
                op = other_obj.get("position", [0.0, 0.0, 0.0])
                dist_sq = (float(p1[0]) - float(op[0])) ** 2 + (float(p1[1]) - float(op[1])) ** 2
                if dist_sq <= proximity_sq:
                    nearby_ball_id = other_ball_id
                    break
            if nearby_ball_id is None:
                continue

            patterns.append(
                {
                    "label": "ball curves around ball",
                    "start_timestep": idx0,
                    "end_timestep": idx2,
                    "parameters": {
                        "object_details": {
                            ball_id: {
                                "role": "curving_ball",
                                "ball_identity": ball_id,
                                "start_timestep": idx0,
                                "end_timestep": idx2,
                            },
                            nearby_ball_id: {
                                "role": "nearby_ball",
                                "ball_identity": nearby_ball_id,
                                "start_timestep": idx1,
                                "end_timestep": idx1,
                            },
                        },
                        "turn_cosine": cos_turn,
                    },
                }
            )
            break
    return patterns


def _repair_detector_fn(detector_id: str | None, fn: Any) -> Any:
    repairs = {
        "primitive-005-s03": _repair_detector_primitive_005_s03,
        "primitive-008-s07": _repair_detector_primitive_008_s07,
        "primitive-010-s04": _repair_detector_primitive_010_s04,
    }
    return repairs.get(detector_id, fn)


def normalize_trace_for_detectors(trace: dict[str, Any]) -> dict[str, Any]:
    """
    Best-effort normalization for third-party detector code.

    The supervised library is inconsistent about expected trace schema. Some detectors
    assume timestep objects always include `kind`, `velocity`, or `radius`, and a few
    are brittle around non-dict payload items. We normalize a copy of the trace here
    rather than mutating upstream simulation output.
    """
    def normalize_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: normalize_value(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [normalize_value(item) for item in value]
        if isinstance(value, list):
            return [normalize_value(item) for item in value]
        return value

    normalized = normalize_value(copy.deepcopy(trace))

    for ts in normalized.get("timesteps", []):
        if not isinstance(ts, dict):
            continue
        objects = ts.get("objects")
        if not isinstance(objects, list):
            continue
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            if "ball_id" in obj:
                obj.setdefault("kind", "ball")
                obj.setdefault("velocity", [0.0, 0.0, 0.0])
                obj.setdefault("angular_velocity", [0.0, 0.0, 0.0])
                obj.setdefault("radius", 0.028575)
            if "position" not in obj:
                obj["position"] = [float(obj.get("x", 0.0)), float(obj.get("y", 0.0)), 0.0]

    for event in normalized.get("events", []):
        if not isinstance(event, dict):
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            payload = {}
            event["payload"] = payload
        ids = payload.get("ids")
        if ids is None:
            payload["ids"] = []
        elif isinstance(ids, tuple):
            payload["ids"] = list(ids)

    return normalized


@lru_cache(maxsize=4)
def _compiled_trace_detectors(library_path: str) -> list[dict[str, Any]]:
    """
    Compile trace-mode detector source_code into callables once per process.
    """
    path = Path(library_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    detectors = payload.get("detectors") if isinstance(payload, dict) else None
    if not isinstance(detectors, list):
        return []

    compiled: list[dict[str, Any]] = []
    for detector in detectors:
        if not isinstance(detector, dict):
            continue
        input_mode = str(detector.get("input_mode") or "").strip() or "trace"
        if input_mode != "trace":
            continue
        source_code = str(detector.get("source_code") or "")
        if not source_code.strip():
            continue
        entrypoint = str(detector.get("entrypoint") or "").strip() or "find_pattern"
        scope: dict[str, Any] = {}
        try:
            exec(source_code, scope, scope)  # noqa: S102 - local offline library code
            fn = scope.get(entrypoint) or scope.get("find_pattern")
            if not callable(fn):
                continue
            detector_id = str(detector.get("detector_id") or "").strip() or None
            compiled.append(
                {
                    "detector_id": detector_id,
                    "label": str(detector.get("label") or "").strip() or None,
                    "entrypoint": entrypoint,
                    "fn": _repair_detector_fn(detector_id, fn),
                }
            )
        except Exception:
            # Compilation failures are handled at call sites for better error payloads.
            continue
    return compiled


def count_supervised_library_patterns(
    trace: dict[str, Any],
    *,
    library_path: Path | None = None,
) -> int:
    """
    Return the total number of pattern items emitted by the supervised library across all detectors.
    """
    path = library_path or supervised_library_path()
    normalized_trace = normalize_trace_for_detectors(trace)
    count = 0
    for detector in _compiled_trace_detectors(str(path)):
        fn = detector.get("fn")
        if not callable(fn):
            continue
        try:
            result = fn(normalized_trace)
        except Exception:
            continue
        if not result:
            continue
        if isinstance(result, list):
            count += len(result)
        else:
            count += 1
    return count


def _add_field(fields: dict[str, list[str]], key: str, value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if not text:
        return
    fields.setdefault(key.lower(), []).append(text)


def _collect_pattern_scalars(value: Any, *, fields: dict[str, list[str]], values: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "object_details" and isinstance(item, dict):
                for object_key, object_value in item.items():
                    _add_field(fields, "object_key", object_key)
                    values.add(str(object_key).strip())
                    _collect_pattern_scalars(object_value, fields=fields, values=values)
                continue
            if isinstance(item, (dict, list)):
                _collect_pattern_scalars(item, fields=fields, values=values)
                continue
            _add_field(fields, str(key), item)
            if item is not None and str(item).strip():
                values.add(str(item).strip())
        return
    if isinstance(value, list):
        for item in value:
            _collect_pattern_scalars(item, fields=fields, values=values)
        return
    if value is not None and str(value).strip():
        values.add(str(value).strip())


def normalize_pattern(pattern: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, list[str]] = {}
    values: set[str] = set()
    _collect_pattern_scalars(pattern.get("parameters"), fields=fields, values=values)
    _collect_pattern_scalars(pattern.get("provenance"), fields=fields, values=values)

    # Provide a few ergonomic aliases for the most likely DSL queries.
    for alias_source in ("ball_id", "ball_identity", "identity"):
        for value in fields.get(alias_source, []):
            _add_field(fields, "ball", value)
    for value in fields.get("object_key", []):
        _add_field(fields, "object", value)

    start = pattern.get("start_timestep")
    end = pattern.get("end_timestep")
    event_time = None
    if start is not None:
        event_time = float(start)
    elif end is not None:
        event_time = float(end)

    return {
        "label": str(pattern.get("label") or "").strip(),
        "fields": fields,
        "values": {str(item).strip() for item in values if str(item).strip()},
        "event_time": event_time,
        "raw": pattern,
    }


def query_trace_patterns(
    trace: dict[str, Any],
    *,
    label: str,
    criteria: dict[str, Any] | None = None,
    annotation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Query the supervised pattern library for a particular label plus optional params.

    Matching semantics:
    - `PATTERN("label")` succeeds if any emitted pattern has that label.
    - For keyword params, all criteria must match a single emitted pattern.
    - `contains=[...]` checks that every expected value appears anywhere in the
      normalized scalar values for that pattern.
    - Other params match against normalized leaf fields, with `ball` as a friendly
      alias for detector keys like `ball_id`, `ball_identity`, and `identity`.
    """
    final_annotation = annotation
    if final_annotation is None:
        final_annotation, _ = annotate_trace_with_supervised_library(trace)

    label_text = str(label or "").strip().lower()
    normalized_patterns: list[dict[str, Any]] = []
    for match in final_annotation.get("matches", []):
        if not isinstance(match, dict):
            continue
        for pattern in match.get("patterns", []):
            if not isinstance(pattern, dict):
                continue
            normalized = normalize_pattern(pattern)
            if normalized["label"].lower() == label_text:
                normalized_patterns.append(normalized)

    if not normalized_patterns:
        return {"score": 0.0, "success": False, "event_time": None, "match_count": 0, "matches": []}

    final_criteria = criteria or {}
    if not final_criteria:
        best_event_time = min((item["event_time"] for item in normalized_patterns if item["event_time"] is not None), default=None)
        return {
            "score": 1.0,
            "success": True,
            "event_time": best_event_time,
            "match_count": len(normalized_patterns),
            "matches": [item["raw"] for item in normalized_patterns],
        }

    def expected_values(value: Any) -> list[str]:
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip().lower() for item in value if str(item).strip()]
        text = str(value).strip().lower()
        return [text] if text else []

    best_score = 0.0
    best_success = False
    best_event_time = None
    best_raw: dict[str, Any] | None = None

    for pattern in normalized_patterns:
        matched = 0
        total = 0
        success = True
        for key, raw_expected in final_criteria.items():
            total += 1
            values = expected_values(raw_expected)
            if not values:
                continue

            pattern_values: set[str]
            if key == "contains":
                pattern_values = {item.lower() for item in pattern["values"]}
            else:
                pattern_values = {item.lower() for item in pattern["fields"].get(str(key).lower(), [])}

            ok = all(item in pattern_values for item in values)
            matched += 1 if ok else 0
            success = success and ok

        score = (matched / total) if total else 1.0
        if success:
            score = 1.0
        if score > best_score or (score == best_score and success and not best_success):
            best_score = score
            best_success = success
            best_event_time = pattern["event_time"]
            best_raw = pattern["raw"]

    return {
        "score": best_score,
        "success": best_success,
        "event_time": best_event_time,
        "match_count": len(normalized_patterns),
        "matches": [best_raw] if best_raw is not None else [],
    }


def filter_trace_patterns(
    trace: dict[str, Any],
    *,
    label: str,
    criteria: dict[str, Any] | None = None,
    annotation: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Return all normalized patterns matching `label` and (optionally) satisfying all `criteria`.

    This is a lower-level helper than `query_trace_patterns`. It's primarily used by
    DSL helpers that need counts or ordering (e.g. "at least N occurrences").
    """
    final_annotation = annotation
    if final_annotation is None:
        final_annotation, _ = annotate_trace_with_supervised_library(trace)

    label_text = str(label or "").strip().lower()
    normalized_patterns: list[dict[str, Any]] = []
    for match in final_annotation.get("matches", []):
        if not isinstance(match, dict):
            continue
        for pattern in match.get("patterns", []):
            if not isinstance(pattern, dict):
                continue
            normalized = normalize_pattern(pattern)
            if normalized["label"].lower() == label_text:
                normalized_patterns.append(normalized)

    if not normalized_patterns:
        return []

    final_criteria = criteria or {}
    if not final_criteria:
        return normalized_patterns

    def expected_values(value: Any) -> list[str]:
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip().lower() for item in value if str(item).strip()]
        text = str(value).strip().lower()
        return [text] if text else []

    matches: list[dict[str, Any]] = []
    for pattern in normalized_patterns:
        ok = True
        for key, raw_expected in final_criteria.items():
            values = expected_values(raw_expected)
            if not values:
                continue

            pattern_values: set[str]
            if key == "contains":
                pattern_values = {item.lower() for item in pattern["values"]}
            else:
                pattern_values = {item.lower() for item in pattern["fields"].get(str(key).lower(), [])}

            if not all(item in pattern_values for item in values):
                ok = False
                break
        if ok:
            matches.append(pattern)

    return matches


def annotate_trace_with_supervised_library(
    trace: dict[str, Any],
    *,
    library_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Execute the local supervised pattern library against a trace.

    Returns:
      - a compact annotation payload suitable for embedding in `shot.json`
      - a list of detector execution errors (for writing into log/)
    """
    path = library_path or supervised_library_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    normalized_trace = normalize_trace_for_detectors(trace)
    matches: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for detector in _compiled_trace_detectors(str(path)):
        fn = detector.get("fn")
        if not callable(fn):
            continue
        try:
            result = fn(normalized_trace)
        except Exception as exc:  # pragma: no cover - depends on library code
            errors.append(
                {
                    "detector_id": detector.get("detector_id"),
                    "label": detector.get("label"),
                    "entrypoint": detector.get("entrypoint"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if not result:
            continue
        if not isinstance(result, list):
            result = [{"value": result}]
        matches.append(
            {
                "detector_id": detector.get("detector_id"),
                "label": detector.get("label"),
                "patterns": result,
            }
        )

    annotation = {
        "schema_version": payload.get("schema_version"),
        "library_id": payload.get("library_id"),
        "library_path": str(path),
        "match_count": len(matches),
        "matches": matches,
    }
    return annotation, errors


def load_library_summary(path: Path | None = None) -> str:
    library_path = path or supervised_library_path()
    payload = json.loads(library_path.read_text(encoding="utf-8"))
    detectors = payload.get("detectors") if isinstance(payload, dict) else None
    if not isinstance(detectors, list):
        return "- No learned detectors are available."
    lines: list[str] = []
    for detector in detectors:
        if not isinstance(detector, dict):
            continue
        label = str(detector.get("label") or "").strip()
        if not label:
            continue
        description = str(detector.get("description") or "").strip()
        metadata = detector.get("metadata") if isinstance(detector.get("metadata"), dict) else {}
        parameter_keys = metadata.get("parameter_keys") if isinstance(metadata, dict) else []
        if not isinstance(parameter_keys, list):
            parameter_keys = []
        params = ", ".join(str(item).strip() for item in parameter_keys if str(item).strip())
        suffix = f" Parameters: {params}." if params else ""
        lines.append((f"- {label}: {description}.{suffix}" if description else f"- {label}.{suffix}").replace("..", "."))
    return "\n".join(lines[:60]) if lines else "- No learned detectors are available."
