from __future__ import annotations

import json
import random
from pathlib import Path

from .config import project_root
from .schemas import BallPosition, SceneSpec


def dataset_root() -> Path:
    return project_root() / "data" / "scenes" / "test_pooltool_100"


def list_scene_dirs() -> list[Path]:
    return sorted(path for path in dataset_root().iterdir() if path.is_dir() and path.name.startswith("sample_"))


def load_scene(scene_dir: Path) -> SceneSpec:
    trace_path = scene_dir / "trace.json"
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    initial_state = payload.get("initial_state") if isinstance(payload, dict) else {}
    ball_positions = [
        BallPosition(ball_id=str(item.get("ball_id")), x=float(item.get("x")), y=float(item.get("y")))
        for item in initial_state.get("initial_ball_positions", [])
        if isinstance(item, dict)
    ]
    return SceneSpec(
        sample_id=scene_dir.name,
        trace_path=trace_path,
        initial_scene_overhead_path=scene_dir / "initial_scene_overhead.png",
        initial_scene_askew_path=(scene_dir / "initial_scene_askew.png") if (scene_dir / "initial_scene_askew.png").is_file() else None,
        table_width=float(initial_state.get("table_width", 0.9906) if isinstance(initial_state, dict) else 0.9906),
        table_length=float(initial_state.get("table_length", 1.9812) if isinstance(initial_state, dict) else 1.9812),
        speed_mapping={str(k): float(v) for k, v in (initial_state.get("speed_mapping") or {}).items()},
        strikespot_mapping={
            str(k): {"a": float(v.get("a", 0.0)), "b": float(v.get("b", 0.0))}
            for k, v in (initial_state.get("strikespot_mapping") or {}).items()
            if isinstance(v, dict)
        },
        initial_ball_positions=ball_positions,
    )


def sample_random_scene(*, seed: int | None = None) -> SceneSpec:
    choices = list_scene_dirs()
    if not choices:
        raise RuntimeError("No scenes found in test_pooltool_100 dataset")
    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    return load_scene(rng.choice(choices))
