from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


DATASET_SCHEMA_VERSION = "1"
TRACE_SCHEMA_VERSION = "1"


class DatasetSplitManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    sample_count: int
    sample_offset: int
    env_counts: dict[str, int] = Field(default_factory=dict)
    completed_samples: int = 0


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_schema_version: str = DATASET_SCHEMA_VERSION
    trace_schema_version: str = TRACE_SCHEMA_VERSION
    output_root: str | None = None
    dataset_name: str
    dataset_id: str
    config_version: str
    config_source: str | None = None
    env_ids: list[str]
    splits: list[DatasetSplitManifest]
    notes: list[str] = Field(default_factory=list)
    config: dict[str, object] = Field(default_factory=dict)
    split_reports: dict[str, object] = Field(default_factory=dict)


def manifest_path(output_root: Path) -> Path:
    return output_root / "dataset_manifest.json"


def write_dataset_manifest(path: Path, manifest: DatasetManifest) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.model_dump(mode="json"), handle, indent=2)


def load_dataset_manifest(path: Path) -> DatasetManifest:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return DatasetManifest.model_validate(payload)


def resolve_dataset_root(manifest_path: Path, manifest: DatasetManifest) -> Path:
    raw_output_root = str(manifest.output_root or "").strip()
    if raw_output_root:
        candidate = Path(raw_output_root)
        if candidate.exists():
            return candidate.resolve()
    return manifest_path.parent.resolve()
