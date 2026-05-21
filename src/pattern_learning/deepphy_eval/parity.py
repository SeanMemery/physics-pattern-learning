from __future__ import annotations

import json
from pathlib import Path

from adapters.registry import AdapterRegistry
from deepphy_eval.io import load_trace
from deepphy_eval.models import (
    EvaluationReport,
    EvaluationTraceResult,
    ParityCheck,
    ParityExpectations,
    ParityMode,
    ParityReport,
)
from runtime.models import AvailabilityStatus


def load_parity_expectations(path: Path) -> ParityExpectations:
    if path.is_dir():
        fixtures: list[dict[str, object]] = []
        expectations_version = "1"
        for child in sorted(path.glob("*/parity_expectations.json")):
            with child.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            validated = ParityExpectations.model_validate(payload)
            expectations_version = validated.expectations_version
            fixtures.extend(item.model_dump(mode="json") for item in validated.fixtures)
        return ParityExpectations.model_validate(
            {
                "expectations_version": expectations_version,
                "fixtures": fixtures,
            }
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ParityExpectations.model_validate(payload)


def run_reference_parity(
    report: EvaluationReport,
    *,
    expectations_path: Path,
) -> ParityReport:
    expectations = load_parity_expectations(expectations_path)
    fixtures = {(fixture.env_id, fixture.task_id): fixture for fixture in expectations.fixtures}
    checks: list[ParityCheck] = []

    for result in report.results:
        fixture = fixtures.get((result.env_id, result.task_id))
        if fixture is None:
            checks.append(
                ParityCheck(
                    env_id=result.env_id,
                    task_id=result.task_id,
                    trace_id=result.trace_id,
                    mode="reference",
                    status="skipped",
                    message="no checked-in parity fixture for this env/task pair",
                )
            )
            continue

        drifts = _reference_drifts(result, fixture.expected_outcome, fixture.expected_metrics)
        checks.append(
            ParityCheck(
                env_id=result.env_id,
                task_id=result.task_id,
                trace_id=result.trace_id,
                mode="reference",
                status="passed" if not drifts else "failed",
                drifts=drifts,
            )
        )

    return ParityReport(
        mode="reference",
        checks=checks,
        expectations_version=expectations.expectations_version,
    )


def run_integration_parity(
    report: EvaluationReport,
    *,
    registry: AdapterRegistry | None = None,
) -> ParityReport:
    adapter_registry = registry or AdapterRegistry()
    checks: list[ParityCheck] = []

    for result in report.results:
        adapter = adapter_registry.load(result.env_id)
        probe = adapter.probe()
        if probe.availability is not AvailabilityStatus.READY:
            availability_value = (
                probe.availability.value
                if hasattr(probe.availability, "value")
                else str(probe.availability)
            )
            checks.append(
                ParityCheck(
                    env_id=result.env_id,
                    task_id=result.task_id,
                    trace_id=result.trace_id,
                    mode="integration",
                    status="blocked",
                    message=f"runtime not ready: {availability_value}",
                )
            )
            continue
        if not result.trace_path:
            checks.append(
                ParityCheck(
                    env_id=result.env_id,
                    task_id=result.task_id,
                    trace_id=result.trace_id,
                    mode="integration",
                    status="skipped",
                    message="integration parity needs a saved trace path",
                )
            )
            continue

        trace = load_trace(Path(result.trace_path))
        runtime_metrics = adapter.evaluate_trace(trace)
        drifts = _reference_drifts(
            result,
            {
                "task_name": trace.outcome.task_name,
                "success": trace.outcome.success,
                "terminal_status": trace.outcome.terminal_status,
            },
            runtime_metrics,
        )
        checks.append(
            ParityCheck(
                env_id=result.env_id,
                task_id=result.task_id,
                trace_id=result.trace_id,
                mode="integration",
                status="passed" if not drifts else "failed",
                drifts=drifts,
            )
        )

    return ParityReport(mode="integration", checks=checks)


def run_parity(
    report: EvaluationReport,
    *,
    mode: ParityMode,
    expectations_path: Path | None = None,
    registry: AdapterRegistry | None = None,
) -> ParityReport:
    if mode == "reference":
        if expectations_path is None:
            raise ValueError("reference parity requires expectations_path")
        return run_reference_parity(report, expectations_path=expectations_path)
    return run_integration_parity(report, registry=registry)


def _reference_drifts(
    result: EvaluationTraceResult,
    expected_outcome: dict[str, object],
    expected_metrics: dict[str, object],
) -> list[str]:
    drifts: list[str] = []
    for key, expected_value in expected_outcome.items():
        actual_value = getattr(result, key, None)
        if actual_value != expected_value:
            drifts.append(f"outcome.{key}: expected {expected_value!r}, got {actual_value!r}")

    for key, expected_value in expected_metrics.items():
        actual_value = result.native_metrics.get(key)
        if actual_value != expected_value:
            drifts.append(f"metric.{key}: expected {expected_value!r}, got {actual_value!r}")

    return drifts
