from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adapters.registry import AdapterRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description="Generic simulator trace helper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_tasks = subparsers.add_parser("list-tasks")
    list_tasks.add_argument("--env", required=True)
    list_tasks.set_defaults(func=list_tasks_command)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--env", required=True)
    generate.add_argument("--task-id", required=True)
    generate.add_argument("--params-json", required=True)
    generate.set_defaults(func=generate_command)

    args = parser.parse_args()
    args.func(args)
    return 0


def list_tasks_command(args: argparse.Namespace) -> None:
    registry = AdapterRegistry()
    adapter = registry.load(str(args.env))
    with contextlib.redirect_stdout(io.StringIO()):
        tasks = adapter.list_tasks()
    print(json.dumps(tasks))


def generate_command(args: argparse.Namespace) -> None:
    registry = AdapterRegistry()
    adapter = registry.load(str(args.env))
    params = json.loads(Path(args.params_json).read_text(encoding="utf-8"))
    params = _materialize_paths(params)
    with contextlib.redirect_stdout(io.StringIO()):
        trace = adapter.generate_trace(str(args.task_id), **params)
    print(json.dumps(trace.model_dump(mode="json")))


def _materialize_paths(value: Any) -> Any:
    if isinstance(value, dict):
        materialized: dict[str, Any] = {}
        for key, item in value.items():
            if key == "output_dir" and isinstance(item, str):
                materialized[key] = Path(item)
            else:
                materialized[key] = _materialize_paths(item)
        return materialized
    if isinstance(value, list):
        return [_materialize_paths(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
