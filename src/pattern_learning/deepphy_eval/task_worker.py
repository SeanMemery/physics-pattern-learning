from __future__ import annotations

import argparse
from pathlib import Path

from deepphy_eval.deepphy_text.models import (
    DeepPHYTextEvaluationRequest,
    DeepPHYTextEvaluationResult,
)
from deepphy_eval.deepphy_text.runner import evaluate_deepphy_text_request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m deepphy_eval.task_worker",
        description="Run one DeepPHY task request in an isolated subprocess.",
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    request = DeepPHYTextEvaluationRequest.model_validate_json(
        args.request.read_text(encoding="utf-8")
    )
    result = evaluate_deepphy_text_request(request)
    args.result.write_text(
        DeepPHYTextEvaluationResult.model_validate(result).model_dump_json(indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
