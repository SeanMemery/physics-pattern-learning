from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deepphy_eval.deepphy_text.models import (
    DeepPHYTextEnvironmentSummary,
    DeepPHYTextEvaluationReport,
    DeepPHYTextEvaluationResult,
)
from paths import get_artifact_layout


def build_deepphy_text_report(
    *,
    run_id: str,
    results: list[DeepPHYTextEvaluationResult],
    report_root: str | None = None,
    pattern_run_dir: str | None = None,
) -> DeepPHYTextEvaluationReport:
    grouped: dict[str, list[DeepPHYTextEvaluationResult]] = {}
    for result in results:
        grouped.setdefault(result.env_id, []).append(result)

    env_summaries = [
        DeepPHYTextEnvironmentSummary(
            env_id=env_id,
            task_count=len(items),
            success_count=sum(1 for item in items if item.success is True),
            failure_count=sum(1 for item in items if item.success is not True),
            description_modes=sorted({item.description_mode for item in items}),
            evidence_status_counts=_count_evidence_statuses(items),
            total_duration_seconds=_sum_optional(
                item.total_duration_seconds for item in items
            ),
            average_task_duration_seconds=_average_optional(
                item.total_duration_seconds for item in items
            ),
        )
        for env_id, items in sorted(grouped.items())
    ]
    return DeepPHYTextEvaluationReport(
        run_id=run_id,
        report_root=report_root,
        pattern_run_dir=pattern_run_dir,
        results=results,
        env_summaries=env_summaries,
        total_duration_seconds=_sum_optional(
            result.total_duration_seconds for result in results
        ),
    )


def write_deepphy_text_artifacts(report: DeepPHYTextEvaluationReport) -> DeepPHYTextEvaluationReport:
    layout = get_artifact_layout()
    run_dir = Path(report.report_root) if report.report_root else (layout.evaluations_root / report.run_id)
    report_dir = run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.json"
    report_path = report_dir / "summary.json"

    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(report.model_dump(mode="json"), handle, indent=2)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report.model_dump(mode="json"), handle, indent=2)

    return report.model_copy(
        update={
            "run_dir": str(run_dir),
            "report_dir": str(report_dir),
            "results_path": str(results_path),
            "report_path": str(report_path),
        }
    )


def _count_evidence_statuses(results: list[DeepPHYTextEvaluationResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for record in result.records:
            if record.evidence_status is None:
                continue
            counts[record.evidence_status] = counts.get(record.evidence_status, 0) + 1
    return counts


def _sum_optional(values: Any) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return sum(numbers)


def _average_optional(values: Any) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)
