from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from deepphy_eval.models import EvaluationReport
from paths import get_artifact_layout


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("eval-%Y%m%dT%H%M%SZ")


def write_evaluation_artifacts(report: EvaluationReport) -> EvaluationReport:
    layout = get_artifact_layout()
    run_dir = layout.evaluations_root / report.run_id
    report_dir = run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.json"
    report_path = report_dir / "summary.json"
    evidence_index_path = report_dir / "evidence_index.json"

    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(report.model_dump(mode="json"), handle, indent=2)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report.model_dump(mode="json"), handle, indent=2)
    with evidence_index_path.open("w", encoding="utf-8") as handle:
        json.dump(build_evidence_index(report), handle, indent=2)

    parity_report_path = None
    if report.parity is not None:
        parity_report_path = report_dir / "parity.json"
        with parity_report_path.open("w", encoding="utf-8") as handle:
            json.dump(report.parity.model_dump(mode="json"), handle, indent=2)

    return report.model_copy(
        update={
            "run_dir": str(run_dir),
            "report_dir": str(report_dir),
            "results_path": str(results_path),
            "report_path": str(report_path),
            "evidence_index_path": str(evidence_index_path),
            "parity_report_path": str(parity_report_path) if parity_report_path is not None else None,
        }
    )


def build_evidence_index(report: EvaluationReport) -> dict[str, object]:
    return {
        result.trace_id: {
            "env_id": result.env_id,
            "task_id": result.task_id,
            "trace_path": result.trace_path,
            "evidence_relpath": result.evidence_relpath,
            "evidence_status": result.evidence_status,
        }
        for result in report.results
    }
