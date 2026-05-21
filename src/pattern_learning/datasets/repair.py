from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio

from adapters.registry import AdapterRegistry
from generation import (
    PHYRE_REVIEW_VIDEO_MAX_FRAMES,
    select_review_video_frames,
)
from media import write_video_artifact
from phyre_video import (
    PHYRE_MAX_VIDEO_DURATION_SECONDS,
    PHYRE_SIMULATION_FPS,
    phyre_video_fps,
    phyre_video_matches_simulation,
)
from traces.quality import validate_trace_quality


POOLTOOL_REQUIRED_VIEWS = ("askew", "overhead")
IPHYRE_MIN_SIMULATION_FPS = 60


@dataclass
class DatasetRepairReport:
    dataset_root: str
    env_ids: list[str]
    reports: list["EnvironmentRepairReport"] = field(default_factory=list)

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset_root": self.dataset_root,
            "env_ids": self.env_ids,
            "reports": [asdict(report) for report in self.reports],
        }


@dataclass
class EnvironmentRepairReport:
    env_id: str
    scanned: int = 0
    repaired: int = 0
    skipped: int = 0
    invalidated: int = 0
    errors: list[str] = field(default_factory=list)


def sample_dirs(dataset_root: Path) -> Iterable[tuple[str, Path]]:
    for split in ("train", "test"):
        split_root = dataset_root / split
        if not split_root.is_dir():
            continue
        for sample_dir in sorted(split_root.glob("sample_*")):
            if sample_dir.is_dir():
                yield split, sample_dir


def sample_index_from_dir(sample_dir: Path) -> int:
    return int(sample_dir.name.split("_")[-1])


def load_trace_payload(sample_dir: Path) -> dict | None:
    trace_path = sample_dir / "trace.json"
    if not trace_path.is_file():
        return None
    with trace_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_trace_and_quality(sample_dir: Path, trace) -> None:
    quality = validate_trace_quality(trace)
    with (sample_dir / "trace.json").open("w", encoding="utf-8") as handle:
        json.dump(trace.model_dump(mode="json"), handle, indent=2)
    with (sample_dir / "quality.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(quality), handle, indent=2)
    if not quality.usable:
        raise RuntimeError(f"{sample_dir} regenerated an unusable trace: {'; '.join(quality.errors)}")


def ensure_sample_suffix(trace, sample_dir: Path):
    suffix = f":sample-{sample_index_from_dir(sample_dir):03d}"
    if trace.trace_id.endswith(suffix):
        return trace
    return trace.model_copy(update={"trace_id": f"{trace.trace_id}{suffix}"})


def current_iphyre_contract(payload: dict) -> bool:
    initial_state = payload.get("initial_state", {})
    simulation_fps = initial_state.get("simulation_fps")
    trace_fps = initial_state.get("trace_fps")
    video_fps = initial_state.get("video_fps")
    if not isinstance(simulation_fps, (int, float)):
        return False
    if simulation_fps < IPHYRE_MIN_SIMULATION_FPS:
        return False
    if isinstance(trace_fps, (int, float)) and simulation_fps < trace_fps:
        return False
    if isinstance(video_fps, (int, float)) and simulation_fps < video_fps:
        return False
    return True


def current_pooltool_contract(payload: dict) -> bool:
    initial_state = payload.get("initial_state", {})
    render_views = initial_state.get("render_views")
    if not isinstance(render_views, list):
        return False
    defined_views = {item.get("view") for item in render_views if isinstance(item, dict)}
    if not set(POOLTOOL_REQUIRED_VIEWS).issubset(defined_views):
        return False

    for kind in ("initial_scene", "simulation_video"):
        if _attachment_views(payload, kind) != set(POOLTOOL_REQUIRED_VIEWS):
            return False
    return True


def current_phyre_contract(payload: dict) -> bool:
    video_attachment = _first_attachment(payload, "simulation_video")
    if video_attachment is None:
        return False
    duration_seconds = _attachment_duration_seconds(video_attachment)
    if duration_seconds is None:
        return False
    success = _phyre_success(payload)
    if success is not False and duration_seconds > PHYRE_MAX_VIDEO_DURATION_SECONDS:
        return False
    simulation_frame_count = _phyre_simulation_frame_count(payload)
    if simulation_frame_count is None:
        return True
    return phyre_video_matches_simulation(
        duration_seconds,
        simulation_frame_count=simulation_frame_count,
        simulation_fps=_phyre_simulation_fps(payload),
        success=success,
    )


def repair_trace_dataset(dataset_root: Path, *, env_ids: list[str]) -> DatasetRepairReport:
    dataset_root = dataset_root.resolve()
    report = DatasetRepairReport(dataset_root=str(dataset_root), env_ids=list(env_ids))
    for env_id in env_ids:
        if env_id == "iphyre":
            report.reports.append(repair_iphyre(dataset_root))
        elif env_id == "pooltool":
            report.reports.append(repair_pooltool(dataset_root))
        elif env_id == "phyre":
            report.reports.append(repair_phyre(dataset_root))
        else:
            report.reports.append(
                EnvironmentRepairReport(env_id=env_id, errors=[f"unsupported repair environment: {env_id}"])
            )
    return report


def repair_iphyre(dataset_root: Path) -> EnvironmentRepairReport:
    registry = AdapterRegistry()
    adapter = registry.load("iphyre")
    report = EnvironmentRepairReport(env_id="iphyre")

    for _, sample_dir in sample_dirs(dataset_root):
        report.scanned += 1
        payload = load_trace_payload(sample_dir)
        if payload is None:
            report.skipped += 1
            continue
        if current_iphyre_contract(payload):
            report.skipped += 1
            continue
        try:
            requested_actions = payload["actions"][0]["requested"]
            trace = adapter.generate_trace(
                payload["task_id"],
                actions=requested_actions,
                capture_visuals=True,
                output_dir=sample_dir,
            )
            write_trace_and_quality(sample_dir, ensure_sample_suffix(trace, sample_dir))
            report.repaired += 1
        except Exception as error:
            report.errors.append(f"{sample_dir}: {type(error).__name__}: {error}")
    return report


def repair_pooltool(dataset_root: Path) -> EnvironmentRepairReport:
    registry = AdapterRegistry()
    adapter = registry.load("pooltool")
    report = EnvironmentRepairReport(env_id="pooltool")

    for _, sample_dir in sample_dirs(dataset_root):
        report.scanned += 1
        payload = load_trace_payload(sample_dir)
        if payload is None:
            report.skipped += 1
            continue
        if current_pooltool_contract(payload):
            report.skipped += 1
            continue
        try:
            initial_state = payload.get("initial_state", {})
            trace = adapter.generate_trace(
                payload["task_id"],
                shots=payload["actions"],
                continuous_dt=float(initial_state.get("continuous_dt", 0.03)),
                capture_frames=False,
                output_dir=sample_dir,
                layout_seed=initial_state.get("layout_seed"),
                rack_jitter=float(initial_state.get("rack_jitter", 0.0)),
            )
            write_trace_and_quality(sample_dir, ensure_sample_suffix(trace, sample_dir))
            report.repaired += 1
        except Exception as error:
            _invalidate_sample(sample_dir)
            report.invalidated += 1
            report.errors.append(f"{sample_dir}: {type(error).__name__}: {error}")
    return report


def repair_phyre(dataset_root: Path) -> EnvironmentRepairReport:
    report = EnvironmentRepairReport(env_id="phyre")

    for _, sample_dir in sample_dirs(dataset_root):
        report.scanned += 1
        payload = load_trace_payload(sample_dir)
        if payload is None:
            report.skipped += 1
            continue
        if current_phyre_contract(payload):
            report.skipped += 1
            continue

        video_path = sample_dir / "simulation.mp4"
        if not video_path.is_file():
            _invalidate_sample(sample_dir)
            report.invalidated += 1
            report.errors.append(f"{sample_dir}: missing simulation.mp4")
            continue

        reader = imageio.get_reader(video_path)
        try:
            frames = [frame for frame in reader]
        finally:
            reader.close()
        if not frames:
            _invalidate_sample(sample_dir)
            report.invalidated += 1
            report.errors.append(f"{sample_dir}: simulation.mp4 contained no frames")
            continue

        selected = select_review_video_frames(frames, target_frame_count=PHYRE_REVIEW_VIDEO_MAX_FRAMES)
        simulation_frame_count = max(len(selected), _phyre_simulation_frame_count(payload) or 0)
        simulation_fps = _phyre_simulation_fps(payload)
        fps = phyre_video_fps(
            len(selected),
            simulation_frame_count=simulation_frame_count,
            simulation_fps=simulation_fps,
            success=_phyre_success(payload),
        )
        temp_path = sample_dir / "simulation.reencoded.mp4"
        metadata = write_video_artifact(selected, temp_path, fps=fps, max_side=512)
        metadata.update(
            {
                "duration_seconds": round(metadata["frame_count"] / max(metadata["fps"], 1), 3),
                "target_duration_seconds": round(simulation_frame_count / max(simulation_fps, 1), 3),
                "frame_selection": "uniform_subsample",
            }
        )
        temp_path.replace(video_path)

        for attachment in payload.get("attachments", []):
            if attachment.get("kind") == "simulation_video":
                attachment["metadata"] = metadata
        with (sample_dir / "trace.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        report.repaired += 1
    return report


def _attachment_views(payload: dict, kind: str) -> set[str]:
    return {
        str(attachment.get("metadata", {}).get("view"))
        for attachment in payload.get("attachments", [])
        if attachment.get("kind") == kind and attachment.get("metadata", {}).get("view")
    }


def _first_attachment(payload: dict, kind: str) -> dict | None:
    for attachment in payload.get("attachments", []):
        if attachment.get("kind") == kind:
            return attachment
    return None


def _attachment_duration_seconds(attachment: dict) -> float | None:
    metadata = attachment.get("metadata", {})
    duration_seconds = metadata.get("duration_seconds")
    if isinstance(duration_seconds, (int, float)):
        return float(duration_seconds)
    frame_count = metadata.get("frame_count")
    fps = metadata.get("fps")
    if isinstance(frame_count, (int, float)) and isinstance(fps, (int, float)) and fps:
        return float(frame_count) / float(fps)
    return None


def _phyre_simulation_frame_count(payload: dict) -> int | None:
    timesteps = payload.get("timesteps")
    if isinstance(timesteps, list) and timesteps:
        return len(timesteps)
    return None


def _phyre_simulation_fps(payload: dict) -> int:
    initial_state = payload.get("initial_state", {})
    value = initial_state.get("simulation_fps") if isinstance(initial_state, dict) else None
    if isinstance(value, (int, float)):
        return max(1, int(value))
    return PHYRE_SIMULATION_FPS


def _phyre_success(payload: dict) -> bool | None:
    outcome = payload.get("outcome")
    if isinstance(outcome, dict):
        value = outcome.get("success")
        if isinstance(value, bool):
            return value
    value = payload.get("success")
    if isinstance(value, bool):
        return value
    return None


def _invalidate_sample(sample_dir: Path) -> None:
    marker = sample_dir / ".repair-invalidated"
    marker.write_text("repair failed; sample must be regenerated\n", encoding="utf-8")
