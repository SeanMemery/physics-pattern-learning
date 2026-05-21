from __future__ import annotations

import argparse
import signal
from pathlib import Path

from .video_renders import run_annotation_video_batch


_CANCELLED = False


def _handle_signal(_signum: int, _frame: object) -> None:
    global _CANCELLED
    _CANCELLED = True
    raise KeyboardInterrupt()


def main() -> int:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--env", required=True)
    args = parser.parse_args()
    try:
        return run_annotation_video_batch(Path(args.project_root), args.env)
    except KeyboardInterrupt:
        return 130 if _CANCELLED else 1


if __name__ == "__main__":
    raise SystemExit(main())
