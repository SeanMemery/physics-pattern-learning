from __future__ import annotations

from pathlib import Path

from datasets.builder import (
    DatasetBuildRun,
    DatasetSplitSpec,
    build_trace_dataset,
    default_dataset_output_root as _default_dataset_output_root,
)
from datasets.config import (
    DatasetBuildConfig,
    build_legacy_random_dataset_config,
)
from generation import ACTIVE_ENVS


TRAIN_SAMPLE_OFFSET = 0
TEST_SAMPLE_OFFSET = 1_000_003


def default_dataset_output_root() -> Path:
    config = _legacy_dataset_config()
    return _default_dataset_output_root(config)


def default_dataset_splits(
    *,
    train_count: int = 1000,
    test_count: int = 100,
) -> list[DatasetSplitSpec]:
    return [
        DatasetSplitSpec(name="train", sample_count=train_count, sample_offset=TRAIN_SAMPLE_OFFSET),
        DatasetSplitSpec(name="test", sample_count=test_count, sample_offset=TEST_SAMPLE_OFFSET),
    ]


def build_random_trace_dataset(
    *,
    train_count: int = 1000,
    test_count: int = 100,
    env_ids: list[str] | None = None,
    output_root: Path | None = None,
    force: bool = False,
) -> DatasetBuildRun:
    config = _legacy_dataset_config(
        train_count=train_count,
        test_count=test_count,
        env_ids=env_ids,
    )
    return build_trace_dataset(config=config, output_root=output_root, force=force)


def _legacy_dataset_config(
    *,
    train_count: int = 1000,
    test_count: int = 100,
    env_ids: list[str] | None = None,
) -> DatasetBuildConfig:
    return build_legacy_random_dataset_config(
        train_count=train_count,
        test_count=test_count,
        env_ids=list(env_ids or ACTIVE_ENVS),
    )
