from __future__ import annotations

import argparse
from pathlib import Path

from .video_renders import run_video_render_batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--target-count", type=int, default=50)
    args = parser.parse_args()
    return run_video_render_batch(Path(args.project_root), args.env, target_count=args.target_count)


if __name__ == "__main__":
    raise SystemExit(main())
