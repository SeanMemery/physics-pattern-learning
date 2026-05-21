from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


REWARD_PANEL_KEYS = ("single_scene", "optimization_suite", "value_model_sweep")
REWARD_PANEL_SESSIONS = {
    "single_scene": "reward-single-scene",
    "optimization_suite": "reward-opt-suite",
    "value_model_sweep": "reward-value-sweep",
}
RANDOM_SCENE_SESSION = "reward-random-scene"
REWARD_PANEL_DIRS = {
    "single_scene": "single_scene",
    "optimization_suite": "optimization_suites",
    "value_model_sweep": "value_model_sweeps",
}


def rewards_dashboard_root(project_root: Path) -> Path:
    return project_root / "state" / "rewards_dashboard"


def rewards_dashboard_state_path(project_root: Path) -> Path:
    return rewards_dashboard_root(project_root) / "state.json"


def rewards_dashboard_scene_dir(project_root: Path) -> Path:
    return rewards_dashboard_root(project_root) / "current_scene"


def rewards_dashboard_log_path(project_root: Path, panel_key: str) -> Path:
    return rewards_dashboard_root(project_root) / f"{panel_key}.log"


def rewards_dashboard_progress_path(project_root: Path, panel_key: str) -> Path:
    return rewards_dashboard_root(project_root) / f"{panel_key}_progress.json"


def random_scene_log_path(project_root: Path) -> Path:
    return rewards_dashboard_root(project_root) / "random_scene.log"


def rewards_dashboard_default_state() -> dict[str, Any]:
    return {
        "goal_texts": ["" for _ in range(10)],
        "single_scene": {
            "goal_text": "",
            "model": "",
            "budget": 250,
            "seed": 0,
            "reward_kind": "synthesized",
            "run_id": "",
            "scene_path": "",
            "scene_seed": 0,
        },
        "optimization_suite": {
            "model": "",
            "scene_count": 10,
            "seed_count": 5,
            "budgets": [10, 50, 100, 250, 500, 1000],
            "run_id": "",
        },
        "value_model_sweep": {
            "model": "",
            "run_id": "",
        },
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_rewards_dashboard_state(project_root: Path) -> dict[str, Any]:
    payload = _load_json_object(rewards_dashboard_state_path(project_root))
    base = rewards_dashboard_default_state()
    goal_texts = payload.get("goal_texts")
    if isinstance(goal_texts, list):
        normalized = [str(item or "") for item in goal_texts[:10]]
        base["goal_texts"] = normalized + ["" for _ in range(max(0, 10 - len(normalized)))]
    for panel_key in REWARD_PANEL_KEYS:
        current = payload.get(panel_key)
        if isinstance(current, dict):
            merged = dict(base[panel_key])
            merged.update(current)
            base[panel_key] = merged
    return base


def save_rewards_dashboard_state(project_root: Path, payload: dict[str, Any]) -> None:
    path = rewards_dashboard_state_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def reward_run_root(project_root: Path, panel_key: str, run_id: str) -> Path:
    run_id_text = str(run_id).strip()
    candidates = [
        project_root
        / "data"
        / "patterns"
        / "pooltool"
        / "reward_program_experiments"
        / REWARD_PANEL_DIRS[panel_key]
        / run_id_text,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def reward_panel_session_name(panel_key: str) -> str:
    return REWARD_PANEL_SESSIONS[panel_key]


def _session_exists(session_name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _session_running(session_name: str) -> bool:
    if not _session_exists(session_name):
        return False
    result = subprocess.run(
        ["tmux", "list-panes", "-t", session_name, "-F", "#{pane_dead}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    pane_values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return any(value == "0" for value in pane_values)


def reward_panel_exists(panel_key: str) -> bool:
    return _session_exists(reward_panel_session_name(panel_key))


def reward_panel_running(panel_key: str) -> bool:
    return _session_running(reward_panel_session_name(panel_key))


def random_scene_exists() -> bool:
    return _session_exists(RANDOM_SCENE_SESSION)


def random_scene_running() -> bool:
    return _session_running(RANDOM_SCENE_SESSION)


def tail_lines(path: Path, *, limit: int = 40) -> list[str]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    filtered: list[str] = []
    suppressed_prefixes = (
        ":display(warning): Unable to load libeglGraphicsPipe.so",
        "Known pipe types:",
        "(all display modules loaded.)",
        ":display:gsg:glgsg(error): GL error 0x502 : invalid operation",
        ":display:gsg:glgsg(error): An OpenGL error has occurred.",
    )
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in suppressed_prefixes):
            continue
        filtered.append(line)
    return filtered[-limit:]


def _repair_current_scene_payload(scene_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    image_value = str(payload.get("image_path") or "").strip()
    image_path = Path(image_value).resolve() if image_value else None
    if image_path is not None and image_path.is_file():
        return payload
    candidates = [
        scene_dir / "initial_scene_overhead.png",
        scene_dir / "initial_scene_askew.png",
        scene_dir / "initial_scene.png",
    ]
    repaired_image = next((candidate for candidate in candidates if candidate.is_file()), None)
    if repaired_image is None:
        return payload
    repaired = dict(payload)
    repaired["image_path"] = str(repaired_image.resolve())
    scene_path = scene_dir / "scene.json"
    try:
        scene_path.write_text(json.dumps(repaired, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        pass
    return repaired


def load_current_scene(project_root: Path) -> dict[str, Any]:
    scene_dir = rewards_dashboard_scene_dir(project_root)
    scene_payload = _load_json_object(scene_dir / "scene.json")
    if scene_payload:
        scene_payload = _repair_current_scene_payload(scene_dir, scene_payload)
    result = {
        "root": scene_dir,
        "scene": scene_payload,
        "image_path": None,
    }
    image_path = Path(str(scene_payload.get("image_path") or "")).resolve() if scene_payload else None
    if image_path is not None and image_path.is_file():
        result["image_path"] = image_path
    return result


def load_reward_panel_snapshot(project_root: Path, state: dict[str, Any], panel_key: str) -> dict[str, Any]:
    panel_state = dict(state.get(panel_key) or {}) if isinstance(state.get(panel_key), dict) else {}
    run_id = str(panel_state.get("run_id") or "").strip()
    progress_path = rewards_dashboard_progress_path(project_root, panel_key)
    log_path = rewards_dashboard_log_path(project_root, panel_key)
    summary = {}
    program_text = ""
    run_root = reward_run_root(project_root, panel_key, run_id) if run_id else None
    summary_path = (run_root / "summary.json") if run_root is not None else None
    program_path = (run_root / "program.dsl") if run_root is not None else None
    if summary_path is not None and summary_path.is_file():
        summary = _load_json_object(summary_path)
    if program_path is not None and program_path.is_file():
        try:
            program_text = program_path.read_text(encoding="utf-8").strip()
        except OSError:
            program_text = ""
    return {
        "panel_key": panel_key,
        "session_name": reward_panel_session_name(panel_key),
        "running": reward_panel_running(panel_key),
        "exists": reward_panel_exists(panel_key),
        "state": panel_state,
        "run_id": run_id,
        "run_root": run_root,
        "progress_path": progress_path,
        "progress": _load_json_object(progress_path),
        "log_path": log_path,
        "log_lines": tail_lines(log_path),
        "summary_path": summary_path,
        "summary": summary,
        "program_text": program_text,
    }
