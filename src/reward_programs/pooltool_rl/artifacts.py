from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import project_root
from .schemas import to_jsonable


def artifacts_root() -> Path:
    root = project_root() / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def experiment_runs_root(experiment_name: str) -> Path:
    root = project_root() / "experiments" / experiment_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def timestamp_run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def create_run_dir(experiment_name: str, run_id: str | None = None) -> Path:
    final_run_id = run_id or timestamp_run_id(experiment_name)
    run_dir = experiment_runs_root(experiment_name) / final_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def append_log(path: Path, line: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")
    return path
