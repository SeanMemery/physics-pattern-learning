from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from datasets.index import DatasetIndex, DatasetIndexEntry


@dataclass(frozen=True)
class PendingDatasetSample:
    split_name: str
    env_id: str
    sample_id: int
    absolute_sample_index: int


def pending_samples_for_split(
    *,
    index: DatasetIndex,
    output_root: Path,
    split_name: str,
    force: bool = False,
) -> list[PendingDatasetSample]:
    pending: list[PendingDatasetSample] = []
    for entry in index.entries:
        if entry.split_name != split_name:
            continue
        if force or not _entry_is_complete(entry, output_root):
            pending.append(
                PendingDatasetSample(
                    split_name=entry.split_name,
                    env_id=entry.env_id,
                    sample_id=entry.sample_id,
                    absolute_sample_index=entry.absolute_sample_index,
                )
            )
    return pending


def refresh_index_entries(
    *,
    index: DatasetIndex,
    output_root: Path,
    split_name: str,
    env_id: str,
) -> None:
    for entry in index.entries:
        if entry.split_name != split_name or entry.env_id != env_id:
            continue
        _refresh_entry(entry, output_root)


def _entry_is_complete(entry: DatasetIndexEntry, output_root: Path) -> bool:
    trace_path = output_root / entry.trace_relpath
    quality_path = output_root / entry.quality_relpath
    return entry.status == "completed" and trace_path.is_file() and quality_path.is_file()


def _refresh_entry(entry: DatasetIndexEntry, output_root: Path) -> None:
    trace_path = output_root / entry.trace_relpath
    quality_path = output_root / entry.quality_relpath

    if not trace_path.is_file() or not quality_path.is_file():
        entry.status = "pending"
        entry.trace_id = None
        entry.task_id = None
        entry.provenance_config_ref = None
        entry.provenance_runner_ref = None
        entry.warnings = []
        entry.error = None
        return

    with trace_path.open("r", encoding="utf-8") as handle:
        trace_payload = json.load(handle)
    with quality_path.open("r", encoding="utf-8") as handle:
        quality_payload = json.load(handle)

    usable = bool(quality_payload.get("usable", False))
    entry.status = "completed" if usable else "failed"
    entry.trace_id = trace_payload.get("trace_id")
    entry.task_id = trace_payload.get("task_id")
    provenance = trace_payload.get("provenance", {})
    entry.provenance_config_ref = provenance.get("config_ref")
    entry.provenance_runner_ref = provenance.get("runner_ref")
    entry.warnings = list(quality_payload.get("warnings", []))
    errors = list(quality_payload.get("errors", []))
    entry.error = "; ".join(errors) if errors else None
