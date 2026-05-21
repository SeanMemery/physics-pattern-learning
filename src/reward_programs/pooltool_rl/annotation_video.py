from __future__ import annotations

import colorsys
import hashlib
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from .library import annotate_trace_with_supervised_library

MIN_ANNOTATION_DISPLAY_SECONDS = 1.0
POOLTOOL_TABLE_WIDTH = 0.9906
POOLTOOL_TABLE_LENGTH = 1.9812
POOLTOOL_OVERHEAD_VIEWPORT_LEFT = 0.0488
POOLTOOL_OVERHEAD_VIEWPORT_TOP = 0.1208
POOLTOOL_OVERHEAD_VIEWPORT_RIGHT = 0.9408
POOLTOOL_OVERHEAD_VIEWPORT_BOTTOM = 0.8781


def _normalized_label_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _xy_point(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        x_value = value.get("x")
        y_value = value.get("y")
        if isinstance(x_value, (int, float)) and isinstance(y_value, (int, float)):
            return float(x_value), float(y_value)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x_value, y_value = value[0], value[1]
        if isinstance(x_value, (int, float)) and isinstance(y_value, (int, float)):
            return float(x_value), float(y_value)
    return None


def _annotation_shape_tree_contains_ball(value: Any) -> bool:
    if isinstance(value, dict):
        shape = str(value.get("shape") or "").strip().upper()
        if shape == "BALL":
            return True
        return any(_annotation_shape_tree_contains_ball(item) for item in value.values())
    if isinstance(value, list):
        return any(_annotation_shape_tree_contains_ball(item) for item in value)
    return False


def _pooltool_pocket_position_from_parameters(parameters: dict[str, Any]) -> tuple[float, float] | None:
    pocket_position = _xy_point(parameters.get("pocket_position"))
    if pocket_position is not None:
        return pocket_position
    pocket_id = str(parameters.get("pocket_id") or "").strip().lower()
    if not pocket_id:
        object_details = parameters.get("object_details")
        if isinstance(object_details, dict):
            for detail in object_details.values():
                if not isinstance(detail, dict):
                    continue
                pocket_position = _xy_point(detail.get("pocket_position"))
                if pocket_position is not None:
                    return pocket_position
                nested_pocket_id = str(detail.get("pocket_id") or "").strip().lower()
                if nested_pocket_id:
                    pocket_id = nested_pocket_id
                    break
    if pocket_id not in {"lb", "lc", "lt", "rb", "rc", "rt"}:
        return None
    x_value = 0.0 if pocket_id.startswith("l") else POOLTOOL_TABLE_WIDTH
    if pocket_id.endswith("b"):
        y_value = 0.0
    elif pocket_id.endswith("c"):
        y_value = POOLTOOL_TABLE_LENGTH * 0.5
    else:
        y_value = POOLTOOL_TABLE_LENGTH
    return (float(x_value), float(y_value))


def _pooltool_ball_pocketed_pattern_is_plausible(item: dict[str, Any]) -> bool:
    parameters = item.get("parameters")
    if not isinstance(parameters, dict):
        return False
    return _pooltool_pocket_position_from_parameters(parameters) is not None


def _pooltool_gentle_touch_pattern_is_plausible(item: dict[str, Any]) -> bool:
    parameters = item.get("parameters")
    if not isinstance(parameters, dict):
        return False
    object_details = parameters.get("object_details")
    if not isinstance(object_details, dict) or len(object_details) != 2:
        return False
    try:
        start_timestep = int(item.get("start_timestep") or 0)
    except (TypeError, ValueError):
        start_timestep = 0
    if start_timestep <= 2:
        return False
    if "interaction_kind" in parameters or "relative_velocity_mps" in parameters:
        return False
    try:
        relative_speed = float(parameters.get("relative_speed"))
    except (TypeError, ValueError):
        return False
    if relative_speed > 0.22:
        return False
    valid_roles = {"gently_touching_ball", "gently_touched_ball"}
    for detail in object_details.values():
        if not isinstance(detail, dict):
            return False
        role = str(detail.get("role") or "").strip()
        if role not in valid_roles:
            return False
    return True


def _annotation_pattern_passes_custom_filters(item: dict[str, Any]) -> bool:
    label_key = _normalized_label_key(item.get("label"))
    if label_key == "ball pocketed":
        return _pooltool_ball_pocketed_pattern_is_plausible(item)
    if label_key == "gentle touch":
        return _pooltool_gentle_touch_pattern_is_plausible(item)
    return True


def _annotation_event_passes_custom_filters(
    *,
    item: dict[str, Any],
    event_kind: str,
    start_timestep: int,
    end_timestep: int,
    total_timesteps: int,
) -> bool:
    label_key = _normalized_label_key(item.get("label"))
    event_kind_key = str(event_kind).strip().lower()
    if total_timesteps <= 1:
        return not (event_kind_key == "end" or (event_kind_key == "start" and end_timestep <= start_timestep))
    event_progress = float(max(start_timestep, end_timestep)) / float(max(1, total_timesteps - 1))
    if event_kind_key == "start" and end_timestep <= start_timestep and event_progress >= 0.90:
        return False
    if event_kind_key == "end" and event_progress >= 0.90:
        return False
    if label_key == "cue strike" and event_kind_key == "start":
        progress = float(start_timestep) / float(max(1, total_timesteps - 1))
        return progress <= 0.05
    return True


def _pattern_event_specs(
    *,
    start_timestep: int,
    end_timestep: int,
    start_frame: int,
    end_frame: int,
) -> list[tuple[str, int]]:
    if end_timestep <= start_timestep:
        return [("start", start_frame)]
    specs = [("start", start_frame)]
    if end_frame != start_frame:
        specs.append(("end", end_frame))
    return specs


def _pooltool_ball_pocketed_matches_trace(pattern: dict[str, Any], trace_payload: dict[str, Any]) -> bool:
    parameters = pattern.get("parameters")
    if not isinstance(parameters, dict):
        return False
    candidate_ball_ids: set[str] = set()
    ball_id = parameters.get("ball_id")
    if isinstance(ball_id, (str, int)):
        token = str(ball_id).strip()
        if token:
            candidate_ball_ids.add(token)
    object_details = parameters.get("object_details")
    if isinstance(object_details, dict):
        for object_key, detail in object_details.items():
            token = str(object_key).strip()
            if token and token.lower() not in {"lb", "lc", "lt", "rb", "rc", "rt"}:
                candidate_ball_ids.add(token)
            if isinstance(detail, dict):
                identity = detail.get("ball_identity")
                if isinstance(identity, (str, int)):
                    identity_token = str(identity).strip()
                    if identity_token and identity_token.lower() not in {"lb", "lc", "lt", "rb", "rc", "rt"}:
                        candidate_ball_ids.add(identity_token)
    if not candidate_ball_ids:
        return False
    timesteps = trace_payload.get("timesteps")
    if not isinstance(timesteps, list) or not timesteps:
        return False
    try:
        start_timestep = int(pattern.get("start_timestep") or 0)
    except (TypeError, ValueError):
        start_timestep = 0
    start_index = max(0, min(len(timesteps) - 1, start_timestep))
    end_index = min(len(timesteps) - 1, start_index + 4)
    for timestep in timesteps[start_index : end_index + 1]:
        if not isinstance(timestep, dict):
            continue
        objects = timestep.get("objects")
        if not isinstance(objects, list):
            continue
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            if str(obj.get("ball_id") or "").strip() not in candidate_ball_ids:
                continue
            if str(obj.get("motion_state") or "").strip().lower() == "pocketed":
                return True
    return False


def _pooltool_ball_pocketed_timestep_from_trace(pattern: dict[str, Any], trace_payload: dict[str, Any]) -> int | None:
    parameters = pattern.get("parameters")
    if not isinstance(parameters, dict):
        return None
    candidate_ball_ids: set[str] = set()
    ball_id = parameters.get("ball_id")
    if isinstance(ball_id, (str, int)):
        token = str(ball_id).strip()
        if token:
            candidate_ball_ids.add(token)
    object_details = parameters.get("object_details")
    if isinstance(object_details, dict):
        for object_key, detail in object_details.items():
            token = str(object_key).strip()
            if token and token.lower() not in {"lb", "lc", "lt", "rb", "rc", "rt"}:
                candidate_ball_ids.add(token)
            if isinstance(detail, dict):
                identity = detail.get("ball_identity")
                if isinstance(identity, (str, int)):
                    identity_token = str(identity).strip()
                    if identity_token and identity_token.lower() not in {"lb", "lc", "lt", "rb", "rc", "rt"}:
                        candidate_ball_ids.add(identity_token)
    if not candidate_ball_ids:
        return None
    timesteps = trace_payload.get("timesteps")
    if not isinstance(timesteps, list) or not timesteps:
        return None
    try:
        start_timestep = int(pattern.get("start_timestep") or 0)
    except (TypeError, ValueError):
        start_timestep = 0
    start_index = max(0, min(len(timesteps) - 1, start_timestep - 2))
    end_index = min(len(timesteps) - 1, max(start_timestep, 0) + 6)
    for index in range(start_index, end_index + 1):
        timestep = timesteps[index]
        if not isinstance(timestep, dict):
            continue
        objects = timestep.get("objects")
        if not isinstance(objects, list):
            continue
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            if str(obj.get("ball_id") or "").strip() not in candidate_ball_ids:
                continue
            if str(obj.get("motion_state") or "").strip().lower() == "pocketed":
                return index
    return None


def _label_color(label: str) -> tuple[int, int, int]:
    digest = hashlib.sha1(str(label or "").encode("utf-8")).digest()
    hue = digest[0] / 255.0
    saturation = 0.62 + ((digest[1] / 255.0) * 0.18)
    value = 0.82 + ((digest[2] / 255.0) * 0.12)
    red, green, blue = colorsys.hsv_to_rgb(hue, min(1.0, saturation), min(1.0, value))
    return (int(red * 255), int(green * 255), int(blue * 255))


def _flatten_annotation_patterns(annotation_payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(annotation_payload.get("library_patterns"), dict):
        return _flatten_annotation_patterns(annotation_payload["library_patterns"])
    annotations = annotation_payload.get("pattern_annotations")
    if isinstance(annotations, dict):
        patterns = annotations.get("patterns")
        if isinstance(patterns, list):
            return [item for item in patterns if isinstance(item, dict)]
    matches = annotation_payload.get("matches")
    flattened: list[dict[str, Any]] = []
    if isinstance(matches, list):
        for match in matches:
            if not isinstance(match, dict):
                continue
            label = str(match.get("label") or "").strip()
            detector_id = str(match.get("detector_id") or "").strip()
            patterns = match.get("patterns")
            if not isinstance(patterns, list):
                continue
            for pattern in patterns:
                if not isinstance(pattern, dict):
                    continue
                item = dict(pattern)
                if label and not str(item.get("label") or "").strip():
                    item["label"] = label
                if detector_id and not str(item.get("detector_id") or "").strip():
                    item["detector_id"] = detector_id
                flattened.append(item)
    return flattened


def _pattern_event_points(pattern: dict[str, Any], *, event_kind: str) -> list[tuple[float, float]]:
    parameters = pattern.get("parameters")
    if not isinstance(parameters, dict):
        return []
    if _normalized_label_key(pattern.get("label")) == "ball pocketed":
        pocket_point = _pooltool_pocket_position_from_parameters(parameters)
        if pocket_point is not None:
            return [pocket_point]
    points: list[tuple[float, float]] = []
    use_end = str(event_kind).strip().lower() == "end"
    object_positions = parameters.get("object_positions")
    if isinstance(object_positions, list):
        for item in object_positions:
            if not isinstance(item, dict):
                continue
            point = _xy_point(item.get("end" if use_end else "start"))
            if point is not None:
                points.append(point)
    object_details = parameters.get("object_details")
    if isinstance(object_details, dict):
        for item in object_details.values():
            if not isinstance(item, dict):
                continue
            point = _xy_point(item.get("end_position" if use_end else "start_position"))
            if point is not None:
                points.append(point)
    direct_point = _xy_point(parameters.get("end_position" if use_end else "start_position"))
    if direct_point is not None:
        points.append(direct_point)
    deduped: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()
    for point in points:
        key = (round(point[0] * 1000), round(point[1] * 1000))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(point)
    return deduped


def _project_video_point(
    *,
    point: tuple[float, float] | None,
    width: int,
    height: int,
    trace_payload: dict[str, Any],
) -> tuple[float, float] | None:
    if point is None or width <= 0 or height <= 0:
        return None
    x_value, y_value = point
    if abs(x_value) <= 1e-6 and abs(y_value) <= 1e-6:
        return None
    initial_state = trace_payload.get("initial_state")
    initial = initial_state if isinstance(initial_state, dict) else {}
    try:
        table_width = max(float(initial.get("table_width") or POOLTOOL_TABLE_WIDTH), 1e-6)
    except (TypeError, ValueError):
        table_width = POOLTOOL_TABLE_WIDTH
    try:
        table_length = max(
            float(initial.get("table_length") or initial.get("table_height") or POOLTOOL_TABLE_LENGTH),
            1e-6,
        )
    except (TypeError, ValueError):
        table_length = POOLTOOL_TABLE_LENGTH
    normalized_x = x_value / table_width
    normalized_y = 1.0 - (y_value / table_length)
    normalized_x = POOLTOOL_OVERHEAD_VIEWPORT_LEFT + normalized_x * (
        POOLTOOL_OVERHEAD_VIEWPORT_RIGHT - POOLTOOL_OVERHEAD_VIEWPORT_LEFT
    )
    normalized_y = POOLTOOL_OVERHEAD_VIEWPORT_TOP + normalized_y * (
        POOLTOOL_OVERHEAD_VIEWPORT_BOTTOM - POOLTOOL_OVERHEAD_VIEWPORT_TOP
    )
    normalized_x = min(1.0, max(0.0, normalized_x))
    normalized_y = min(1.0, max(0.0, normalized_y))
    return normalized_x * width, normalized_y * height


def _load_annotation_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if isinstance(payload.get("library_patterns"), dict):
            return payload["library_patterns"]
        return payload
    raise ValueError(f"Malformed annotation payload: {path}")


def build_annotation_overlay_payload(
    *,
    trace_payload: dict[str, Any],
    annotation_payload: dict[str, Any],
    source_video_path: Path,
    selected_labels: list[str] | None = None,
) -> dict[str, Any] | None:
    reader = imageio.get_reader(str(source_video_path))
    try:
        meta = reader.get_meta_data() or {}
        fps = float(meta.get("fps") or 10.0)
        try:
            frame_count = int(reader.count_frames())
        except Exception:
            frame_count = 0
    finally:
        reader.close()
    if frame_count <= 0:
        return None
    timesteps = trace_payload.get("timesteps")
    total_timesteps = len(timesteps) if isinstance(timesteps, list) else frame_count
    min_output_frames = max(1, int(round(MIN_ANNOTATION_DISPLAY_SECONDS * fps)))
    selected_keys = {
        _normalized_label_key(label)
        for label in (selected_labels or [])
        if _normalized_label_key(label)
    }
    patterns = [
        item
        for item in _flatten_annotation_patterns(annotation_payload)
        if _annotation_pattern_passes_custom_filters(item)
    ]
    if not patterns:
        return None
    filtered_patterns: list[dict[str, Any]] = []
    for item in patterns:
        if _normalized_label_key(item.get("label")) == "ball pocketed" and not _pooltool_ball_pocketed_matches_trace(item, trace_payload):
            continue
        filtered_patterns.append(item)
    if not filtered_patterns:
        return None
    events: list[dict[str, Any]] = []
    for item in filtered_patterns:
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        if selected_keys and _normalized_label_key(label) not in selected_keys:
            continue
        start_timestep = int(item.get("start_timestep") or 0)
        end_timestep = int(item.get("end_timestep") or start_timestep)
        if _normalized_label_key(label) == "ball pocketed":
            pocketed_timestep = _pooltool_ball_pocketed_timestep_from_trace(item, trace_payload)
            if pocketed_timestep is not None:
                start_timestep = pocketed_timestep
                end_timestep = pocketed_timestep
        if total_timesteps <= 1 or frame_count <= 1:
            start_frame = 0
            end_frame = 0
        else:
            start_frame = int(round((start_timestep * max(0, frame_count - 1)) / max(1, total_timesteps - 1)))
            end_frame = int(round((max(start_timestep, end_timestep) * max(0, frame_count - 1)) / max(1, total_timesteps - 1)))
        for event_kind, event_frame in _pattern_event_specs(
            start_timestep=start_timestep,
            end_timestep=end_timestep,
            start_frame=start_frame,
            end_frame=end_frame,
        ):
            if not _annotation_event_passes_custom_filters(
                item=item,
                event_kind=event_kind,
                start_timestep=start_timestep,
                end_timestep=end_timestep,
                total_timesteps=total_timesteps,
            ):
                continue
            event_end_frame = min(max(0, frame_count - 1), event_frame + max(0, min_output_frames - 1))
            projected = [
                _project_video_point(point=point, width=1, height=1, trace_payload=trace_payload)
                for point in _pattern_event_points(item, event_kind=event_kind)
            ]
            normalized_points = [
                [float(point[0]), float(point[1])]
                for point in projected
                if point is not None
            ]
            metadata = item.get("metadata")
            confidence = 1.0
            if isinstance(metadata, dict):
                try:
                    confidence = float(metadata.get("confidence") or 0.0)
                except (TypeError, ValueError):
                    confidence = 0.0
            events.append(
                {
                    "label": label,
                    "event_kind": event_kind,
                    "color": f"#{_label_color(label)[0]:02x}{_label_color(label)[1]:02x}{_label_color(label)[2]:02x}",
                    "confidence": max(0.0, min(1.0, confidence)),
                    "start_time": event_frame / fps,
                    "end_time": event_end_frame / fps,
                    "points": normalized_points,
                }
            )
    if not events:
        return None
    return {
        "fps": fps,
        "frame_count": frame_count,
        "events": events,
    }


def _overlay_active_events(payload: dict[str, Any], now_seconds: float, confidence_threshold: float) -> list[dict[str, Any]]:
    events = payload.get("events")
    if not isinstance(events, list):
        return []
    threshold = max(0.0, min(1.0, float(confidence_threshold or 0.0)))
    active: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        try:
            confidence = float(event.get("confidence") if event.get("confidence") is not None else 1.0)
            start_time = float(event.get("start_time") or 0.0)
            end_time = float(event.get("end_time") or 0.0)
        except (TypeError, ValueError):
            continue
        if confidence < threshold:
            continue
        if now_seconds < start_time or now_seconds > end_time:
            continue
        active.append(event)
    return active


def _overlay_font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _draw_overlay_frame(frame: Any, payload: dict[str, Any], now_seconds: float, confidence_threshold: float) -> Any:
    image = Image.fromarray(frame).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    radius = max(3, min(7, round(min(width, height) * 0.007)))
    font = _overlay_font(20)
    legend_entries: dict[str, tuple[int, int, int]] = {}
    all_events = payload.get("events")
    if isinstance(all_events, list):
        for event in all_events:
            if not isinstance(event, dict):
                continue
            try:
                confidence = float(event.get("confidence") if event.get("confidence") is not None else 1.0)
            except (TypeError, ValueError):
                confidence = 1.0
            if confidence < confidence_threshold:
                continue
            try:
                rgba = ImageColor.getrgb(str(event.get("color") or "#ffffff"))
            except ValueError:
                rgba = (255, 255, 255)
            label = str(event.get("label") or "pattern")
            if label and label not in legend_entries:
                legend_entries[label] = rgba
    for event in _overlay_active_events(payload, now_seconds, confidence_threshold):
        try:
            rgba = ImageColor.getrgb(str(event.get("color") or "#ffffff"))
        except ValueError:
            rgba = (255, 255, 255)
        points = event.get("points")
        if not isinstance(points, list):
            continue
        for point in points:
            if not isinstance(point, list) or len(point) < 2:
                continue
            try:
                x_value = float(point[0]) * width
                y_value = float(point[1]) * height
            except (TypeError, ValueError):
                continue
            draw.ellipse(
                (x_value - radius, y_value - radius, x_value + radius, y_value + radius),
                fill=(rgba[0], rgba[1], rgba[2], 0x58),
                outline=(rgba[0], rgba[1], rgba[2], 255),
                width=max(1, radius // 4),
            )
    if legend_entries:
        swatch_size = 14
        row_height = 26
        legend_width = 0
        for label in legend_entries:
            text_box = draw.textbbox((0, 0), label, font=font)
            legend_width = max(legend_width, int(text_box[2] - text_box[0]))
        card_width = legend_width + swatch_size + 36
        card_height = (len(legend_entries) * row_height) + 16
        card_left = 12
        card_top = max(12, height - card_height - 12)
        draw.rounded_rectangle(
            (card_left, card_top, card_left + card_width, card_top + card_height),
            radius=6,
            fill=(17, 17, 17, 184),
        )
        for index, (label, rgba) in enumerate(legend_entries.items()):
            row_top = card_top + 6 + (index * row_height)
            draw.rectangle(
                (card_left + 8, row_top + 6, card_left + 8 + swatch_size, row_top + 6 + swatch_size),
                fill=(rgba[0], rgba[1], rgba[2], 255),
            )
            draw.text((card_left + 8 + swatch_size + 10, row_top + 1), label, fill=(255, 255, 255, 255), font=font)
    return image.convert("RGB")


def export_annotation_overlay_video(
    *,
    trace_path: Path,
    video_path: Path,
    output_path: Path,
    annotation_path: Path | None = None,
    selected_labels: list[str] | None = None,
    confidence_threshold: float = 0.0,
) -> Path:
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    if annotation_path is not None:
        annotation_payload = _load_annotation_payload(annotation_path)
    else:
        annotation_payload, _ = annotate_trace_with_supervised_library(trace_payload)
    payload = build_annotation_overlay_payload(
        trace_payload=trace_payload,
        annotation_payload=annotation_payload,
        source_video_path=video_path,
        selected_labels=selected_labels,
    )
    if not payload:
        raise FileNotFoundError("No overlay payload available.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path = output_path.with_suffix(".overlay.json")
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reader = imageio.get_reader(str(video_path))
    try:
        meta = reader.get_meta_data() or {}
        fps = float(meta.get("fps") or payload.get("fps") or 30.0)
        writer = imageio.get_writer(str(output_path), fps=max(1.0, fps), codec="libx264", quality=8)
        try:
            for index, frame in enumerate(reader):
                now_seconds = index / max(1.0, fps)
                composited = _draw_overlay_frame(frame, payload, now_seconds, confidence_threshold)
                writer.append_data(np.asarray(composited))
        finally:
            writer.close()
    finally:
        reader.close()
    return output_path


def default_overlay_output_path(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}.annotated.mp4")


def export_single_scene_annotation_overlay(
    *,
    run_dir: Path,
    output_path: Path | None = None,
    annotation_path: Path | None = None,
    selected_labels: list[str] | None = None,
    confidence_threshold: float = 0.0,
) -> Path:
    if annotation_path is None and (run_dir / "shot.json").is_file():
        annotation_path = run_dir / "shot.json"
    trace_path = run_dir / "best_trace.json"
    if not trace_path.is_file() and (run_dir / "shot.json").is_file():
        shot_payload = json.loads((run_dir / "shot.json").read_text(encoding="utf-8"))
        trace_hint = str(shot_payload.get("trace_path") or "").strip()
        if trace_hint:
            candidate = Path(trace_hint)
            if candidate.is_file():
                trace_path = candidate
    if not trace_path.is_file():
        raise FileNotFoundError(f"Missing best trace for run: {run_dir}")
    video_path = run_dir / "simulation_overhead.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing simulation_overhead.mp4 for run: {run_dir}")
    destination = output_path or (run_dir / "simulation_overhead.annotated.mp4")
    return export_annotation_overlay_video(
        trace_path=trace_path,
        video_path=video_path,
        output_path=destination,
        annotation_path=annotation_path,
        selected_labels=selected_labels,
        confidence_threshold=confidence_threshold,
    )
