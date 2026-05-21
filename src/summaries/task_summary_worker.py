from __future__ import annotations

import argparse
from pathlib import Path

from .task_summaries import run_task_summary_batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--target-count", type=int, default=12)
    parser.add_argument(
        "--enabled-variant",
        action="append",
        dest="enabled_variants",
        default=[],
    )
    args = parser.parse_args()
    enabled_variants = [str(item).strip() for item in (args.enabled_variants or []) if str(item).strip()]
    return run_task_summary_batch(
        Path(args.project_root),
        args.env,
        target_count=args.target_count,
        enabled_variants_override=enabled_variants or None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
