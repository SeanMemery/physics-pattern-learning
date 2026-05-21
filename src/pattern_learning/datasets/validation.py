from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from datasets.index import DatasetIndexEntry, index_path, load_dataset_index
from datasets.manifest import (
    DATASET_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION,
    load_dataset_manifest,
    resolve_dataset_root,
)
from datasets.repair import (
    current_iphyre_contract,
    current_phyre_contract,
    current_pooltool_contract,
)


@dataclass
class ValidationIssue:
    level: str
    code: str
    message: str


@dataclass
class SplitValidationReport:
    name: str
    ready: bool
    expected_samples: int
    completed_samples: int
    env_counts: dict[str, int]
    outcome_summary: dict[str, dict[str, int]]
    issues: list[ValidationIssue] = field(default_factory=list)


@dataclass
class DatasetValidationReport:
    dataset_name: str
    dataset_id: str
    ready: bool
    dataset_schema_version: str
    trace_schema_version: str
    annotation_sidecar_relpath: str
    issues: list[ValidationIssue] = field(default_factory=list)
    splits: list[SplitValidationReport] = field(default_factory=list)


def build_dataset_readiness_report(manifest_file: Path) -> "DatasetValidationReport":
    return validate_dataset(manifest_file)


def render_dataset_readiness_payload(report: "DatasetValidationReport") -> dict[str, object]:
    return {
        "dataset_name": report.dataset_name,
        "dataset_id": report.dataset_id,
        "ready": report.ready,
        "dataset_schema_version": report.dataset_schema_version,
        "trace_schema_version": report.trace_schema_version,
        "annotation_sidecar_relpath": report.annotation_sidecar_relpath,
        "issues": [issue.__dict__ for issue in report.issues],
        "splits": [
            {
                "name": split.name,
                "ready": split.ready,
                "expected_samples": split.expected_samples,
                "completed_samples": split.completed_samples,
                "env_counts": split.env_counts,
                "outcome_summary": split.outcome_summary,
                "issues": [issue.__dict__ for issue in split.issues],
            }
            for split in report.splits
        ],
    }


def write_dataset_readiness_report(output_path: Path, report: "DatasetValidationReport") -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(render_dataset_readiness_payload(report), handle, indent=2)


def validate_dataset(manifest_file: Path) -> DatasetValidationReport:
    manifest = load_dataset_manifest(manifest_file)
    dataset_root = resolve_dataset_root(manifest_file, manifest)
    index = load_dataset_index(index_path(dataset_root))

    issues: list[ValidationIssue] = []
    if manifest.dataset_schema_version != DATASET_SCHEMA_VERSION:
        issues.append(
            ValidationIssue(
                level="error",
                code="dataset_schema_version",
                message=(
                    f"dataset_schema_version {manifest.dataset_schema_version} does not match "
                    f"expected {DATASET_SCHEMA_VERSION}"
                ),
            )
        )
    if manifest.trace_schema_version != TRACE_SCHEMA_VERSION:
        issues.append(
            ValidationIssue(
                level="error",
                code="trace_schema_version",
                message=(
                    f"trace_schema_version {manifest.trace_schema_version} does not match "
                    f"expected {TRACE_SCHEMA_VERSION}"
                ),
            )
        )

    split_reports = [
        _validate_split(
            dataset_root=dataset_root,
            split=split,
            env_ids=manifest.env_ids,
            entries=[entry for entry in index.entries if entry.split_name == split.name],
        )
        for split in manifest.splits
    ]

    ready = not issues and all(split.ready for split in split_reports)
    return DatasetValidationReport(
        dataset_name=manifest.dataset_name,
        dataset_id=manifest.dataset_id,
        ready=ready,
        dataset_schema_version=manifest.dataset_schema_version,
        trace_schema_version=manifest.trace_schema_version,
        annotation_sidecar_relpath="trace.evidence.json",
        issues=issues,
        splits=split_reports,
    )


def _validate_split(
    *,
    dataset_root: Path,
    split,
    env_ids: list[str],
    entries: list[DatasetIndexEntry],
) -> SplitValidationReport:
    issues: list[ValidationIssue] = []
    expected_samples = split.sample_count * len(env_ids)
    completed_samples = 0
    env_counts = {env_id: 0 for env_id in env_ids}
    outcome_summary = {env_id: {"success": 0, "failure": 0, "unknown": 0} for env_id in env_ids}

    if len(entries) != expected_samples:
        issues.append(
            ValidationIssue(
                level="error",
                code="split_entry_count",
                message=(
                    f"split {split.name} expected {expected_samples} indexed samples but found {len(entries)}"
                ),
            )
        )

    for entry in entries:
        trace_path = dataset_root / entry.trace_relpath
        quality_path = dataset_root / entry.quality_relpath
        if entry.status != "completed":
            issues.append(
                ValidationIssue(
                    level="error",
                    code="incomplete_entry",
                    message=f"{entry.split_name}/{entry.env_id}/sample_{entry.sample_id:03d} is {entry.status}",
                )
            )
            continue
        if not trace_path.is_file() or not quality_path.is_file():
            issues.append(
                ValidationIssue(
                    level="error",
                    code="missing_artifact",
                    message=f"missing trace or quality file for {entry.sample_dir}",
                )
            )
            continue

        completed_samples += 1
        env_counts[entry.env_id] = env_counts.get(entry.env_id, 0) + 1
        trace_payload = _read_json(trace_path)
        provenance = trace_payload.get("provenance", {})
        if not provenance.get("config_ref"):
            issues.append(
                ValidationIssue(
                    level="error",
                    code="missing_provenance_config_ref",
                    message=f"trace {entry.trace_relpath} is missing provenance.config_ref",
                )
            )
        if not provenance.get("runner_ref"):
            issues.append(
                ValidationIssue(
                    level="error",
                    code="missing_provenance_runner_ref",
                    message=f"trace {entry.trace_relpath} is missing provenance.runner_ref",
                )
            )

        success = trace_payload.get("outcome", {}).get("success")
        if success is True:
            outcome_summary[entry.env_id]["success"] += 1
        elif success is False:
            outcome_summary[entry.env_id]["failure"] += 1
        else:
            outcome_summary[entry.env_id]["unknown"] += 1

        issues.extend(_environment_contract_issues(entry.env_id, entry.trace_relpath, trace_payload))

    for env_id, counts in outcome_summary.items():
        if completed_samples and counts["success"] == 0 and counts["failure"] == 0:
            issues.append(
                ValidationIssue(
                    level="warning",
                    code="outcome_distribution_unknown",
                    message=f"{split.name}/{env_id} has no usable success/failure outcomes",
                )
            )

    ready = not any(issue.level == "error" for issue in issues)
    return SplitValidationReport(
        name=split.name,
        ready=ready,
        expected_samples=expected_samples,
        completed_samples=completed_samples,
        env_counts=env_counts,
        outcome_summary=outcome_summary,
        issues=issues,
    )


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _environment_contract_issues(
    env_id: str,
    trace_relpath: str,
    trace_payload: dict[str, object],
) -> list[ValidationIssue]:
    if env_id == "iphyre" and not current_iphyre_contract(trace_payload):
        return [
            ValidationIssue(
                level="error",
                code="stale_iphyre_fidelity",
                message=f"trace {trace_relpath} uses a stale I-PHYRE fidelity contract",
            )
        ]
    if env_id == "pooltool" and not current_pooltool_contract(trace_payload):
        return [
            ValidationIssue(
                level="error",
                code="stale_pooltool_render_views",
                message=f"trace {trace_relpath} is missing askew and overhead Pooltool media",
            )
        ]
    if env_id == "phyre" and not current_phyre_contract(trace_payload):
        return [
            ValidationIssue(
                level="error",
                code="stale_phyre_video_duration",
                message=f"trace {trace_relpath} does not satisfy the short PHYRE video contract",
            )
        ]
    return []
