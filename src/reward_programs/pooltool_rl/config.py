from __future__ import annotations

import json
from pathlib import Path

import yaml

from .schemas import AppConfig, GoalSpec


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_path() -> Path:
    return project_root() / "configs" / "config.yml"


def load_config(path: Path | None = None) -> AppConfig:
    target = path or config_path()
    payload = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(payload)


def load_goals(path: Path) -> list[GoalSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    goals = payload.get("goals") if isinstance(payload, dict) else None
    if not isinstance(goals, list):
        raise ValueError(f"Malformed goals file: {path}")
    return [GoalSpec.model_validate(item) for item in goals]
