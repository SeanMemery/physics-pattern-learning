from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from datasets.config import DatasetBuildConfig
from datasets.manifest import DATASET_SCHEMA_VERSION, TRACE_SCHEMA_VERSION


class DatasetIndexEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split_name: str
    env_id: str
    sample_id: int
    absolute_sample_index: int
    sample_dir: str
    trace_relpath: str
    quality_relpath: str
    status: str = "pending"
    trace_id: str | None = None
    task_id: str | None = None
    provenance_config_ref: str | None = None
    provenance_runner_ref: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class DatasetIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_schema_version: str = DATASET_SCHEMA_VERSION
    trace_schema_version: str = TRACE_SCHEMA_VERSION
    dataset_id: str
    entries: list[DatasetIndexEntry]


def index_path(output_root: Path) -> Path:
    return output_root / "dataset_index.json"


def load_dataset_index(path: Path) -> DatasetIndex:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return DatasetIndex.model_validate(payload)


def write_dataset_index(path: Path, index: DatasetIndex) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(index.model_dump(mode="json"), handle, indent=2)


def build_dataset_index(
    *,
    config: DatasetBuildConfig,
    dataset_id: str,
) -> DatasetIndex:
    entries: list[DatasetIndexEntry] = []
    for split in config.splits:
        for env_id in config.env_ids:
            for sample_id in range(split.sample_count):
                sample_dir = dataset_sample_dir(
                    split_name=split.name,
                    env_ids=config.env_ids,
                    env_id=env_id,
                    sample_id=sample_id,
                )
                entries.append(
                    DatasetIndexEntry(
                        split_name=split.name,
                        env_id=env_id,
                        sample_id=sample_id,
                        absolute_sample_index=split.sample_offset + sample_id,
                        sample_dir=sample_dir.as_posix(),
                        trace_relpath=(sample_dir / "trace.json").as_posix(),
                        quality_relpath=(sample_dir / "quality.json").as_posix(),
                    )
                )
    return DatasetIndex(dataset_id=dataset_id, entries=entries)


def dataset_sample_dir(
    *,
    split_name: str,
    env_ids: list[str],
    env_id: str,
    sample_id: int,
) -> Path:
    if len(env_ids) == 1:
        return Path(split_name) / f"sample_{sample_id:03d}"
    return Path(split_name) / env_id / f"sample_{sample_id:03d}"
