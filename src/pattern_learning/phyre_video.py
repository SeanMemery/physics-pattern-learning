from __future__ import annotations

PHYRE_SIMULATION_FPS = 60
PHYRE_TARGET_VIDEO_DURATION_SECONDS = 10.0
PHYRE_MAX_VIDEO_DURATION_SECONDS = PHYRE_TARGET_VIDEO_DURATION_SECONDS
PHYRE_VIDEO_DURATION_TOLERANCE_SECONDS = 0.25


def phyre_simulation_duration_seconds(
    simulation_frame_count: int,
    *,
    simulation_fps: int = PHYRE_SIMULATION_FPS,
) -> float:
    safe_frame_count = max(1, int(simulation_frame_count))
    safe_simulation_fps = max(1, int(simulation_fps))
    return safe_frame_count / safe_simulation_fps


def phyre_video_fps(
    video_frame_count: int,
    *,
    simulation_frame_count: int,
    simulation_fps: int = PHYRE_SIMULATION_FPS,
    success: bool | None = None,
) -> int:
    duration_seconds = PHYRE_TARGET_VIDEO_DURATION_SECONDS
    return max(1, int(round(max(1, int(video_frame_count)) / duration_seconds)))


def phyre_video_matches_simulation(
    duration_seconds: float,
    *,
    simulation_frame_count: int,
    simulation_fps: int = PHYRE_SIMULATION_FPS,
    tolerance_seconds: float = PHYRE_VIDEO_DURATION_TOLERANCE_SECONDS,
    success: bool | None = None,
) -> bool:
    expected_duration_seconds = PHYRE_TARGET_VIDEO_DURATION_SECONDS
    return abs(float(duration_seconds) - expected_duration_seconds) <= max(
        1.0 / max(1, int(simulation_fps)),
        float(tolerance_seconds),
    )
