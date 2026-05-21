from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from datasets.config import DatasetBuildConfig
from debug import debug_log, set_debug_log_file
from datasets.index import (
    DatasetIndex,
    build_dataset_index,
    index_path,
    load_dataset_index,
    write_dataset_index,
)
from datasets.manifest import (
    DatasetManifest,
    DatasetSplitManifest,
    load_dataset_manifest,
    manifest_path,
    write_dataset_manifest,
)
from datasets.resume import pending_samples_for_split, refresh_index_entries
from datasets.validation import validate_dataset
from generation import (
    EnvironmentGenerationReport,
    TraceGenerationRun,
    TraceSampleRequest,
    generate_trace_samples,
)
from paths import get_artifact_layout


@dataclass(frozen=True)
class DatasetSplitSpec:
    name: str
    sample_count: int
    sample_offset: int


@dataclass
class DatasetBuildRun:
    output_root: Path
    env_ids: list[str]
    splits: list[DatasetSplitSpec]
    split_runs: dict[str, TraceGenerationRun]
    dataset_name: str
    config_version: str
    config_path: Path | None = None


def default_dataset_output_root(config: DatasetBuildConfig | None = None) -> Path:
    layout = get_artifact_layout()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dataset_name = config.dataset_name if config is not None else "random_trace_dataset"
    slug = _slugify(dataset_name)
    if config is not None and len(config.env_ids) == 1:
        return layout.data_root / "datasets" / config.env_ids[0] / f"{slug}_{timestamp}"
    return layout.data_root / "datasets" / f"{slug}_{timestamp}"


def build_trace_dataset(
    *,
    config: DatasetBuildConfig,
    output_root: Path | None = None,
    force: bool = False,
    selected_splits: list[str] | None = None,
    config_path: Path | None = None,
) -> DatasetBuildRun:
    root = output_root or default_dataset_output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    set_debug_log_file(root / "debug.log")
    chosen_splits = _selected_splits(config=config, selected_splits=selected_splits)
    dataset_id = root.name
    split_specs = [
        DatasetSplitSpec(
            name=split.name,
            sample_count=split.sample_count,
            sample_offset=split.sample_offset,
        )
        for split in chosen_splits
    ]
    debug_log(
        "datasets.build.start",
        dataset_name=config.dataset_name,
        env_ids=config.env_ids,
        output_root=output_root,
        force=force,
        selected_splits=selected_splits,
        config_path=config_path,
    )
    manifest, index = _ensure_dataset_metadata(
        config=config,
        output_root=root,
        dataset_id=dataset_id,
        chosen_splits=chosen_splits,
        split_specs=split_specs,
        config_path=config_path,
    )
    if not force:
        validation = validate_dataset(manifest_path(root))
        if validation.ready:
            debug_log("datasets.build.skip_existing_ready", output_root=root, dataset_name=validation.dataset_name)
            return _build_existing_dataset_run(
                root=root,
                config_path=config_path,
                chosen_splits=chosen_splits,
            )

    chosen_splits = _prioritize_splits(
        chosen_splits=chosen_splits,
        index=index,
        env_ids=config.env_ids,
    )

    split_runs: dict[str, TraceGenerationRun] = {}

    for split in chosen_splits:
        debug_log("datasets.build.split.start", split=split.name, sample_count=split.sample_count)
        split_root = root / split.name
        split_root.mkdir(parents=True, exist_ok=True)
        per_environment = []
        for env_id in config.env_ids:
            env_report = _build_split_environment(
                index=index,
                output_root=root,
                split_root=split_root,
                split_name=split.name,
                env_id=env_id,
                force=force,
                flatten_single_env=len(config.env_ids) == 1,
            )
            if env_report is None:
                continue
            per_environment.append(env_report)

        split_runs[split.name] = TraceGenerationRun(output_root=split_root, per_environment=per_environment)

    _write_dataset_manifest_with_reports(
        output_root=root,
        manifest=manifest,
        config=config,
        split_specs=split_specs,
        split_runs=split_runs,
        index=index,
    )
    debug_log("datasets.build.done", output_root=root, dataset_name=config.dataset_name)

    return DatasetBuildRun(
        output_root=root,
        env_ids=list(config.env_ids),
        splits=split_specs,
        split_runs=split_runs,
        dataset_name=config.dataset_name,
        config_version=config.config_version,
        config_path=config_path,
    )


def _build_split_environment(
    *,
    index: DatasetIndex,
    output_root: Path,
    split_root: Path,
    split_name: str,
    env_id: str,
    force: bool,
    flatten_single_env: bool,
) -> EnvironmentGenerationReport | None:
    env_pending = [
        item
        for item in pending_samples_for_split(
            index=index,
            output_root=output_root,
            split_name=split_name,
            force=force,
        )
        if item.env_id == env_id
    ]
    if not env_pending:
        return None

    total_generated = 0
    collected_errors: list[str] = []
    collected_warnings: list[str] = []
    collected_status = "completed"
    saved_dir = str(split_root if flatten_single_env else split_root / env_id)
    blocked_reason: str | None = None
    pass_index = 0

    while env_pending:
        pass_index += 1
        completed_before = _completed_count(index=index, split_name=split_name, env_id=env_id)
        env_requests = [
            TraceSampleRequest(
                sample_id=item.sample_id,
                absolute_sample_index=item.absolute_sample_index,
            )
            for item in env_pending
        ]
        debug_log(
            "datasets.build.split.env.pass.start",
            split=split_name,
            env_id=env_id,
            pass_index=pass_index,
            pending_count=len(env_pending),
        )
        env_run = generate_trace_samples(
            sample_requests={env_id: env_requests},
            output_root=split_root,
            force=force,
            reset_existing=force,
            flatten_single_env=flatten_single_env,
        )
        refresh_index_entries(index=index, output_root=output_root, split_name=split_name, env_id=env_id)
        write_dataset_index(index_path(output_root), index)

        env_reports = [
            report for report in env_run.per_environment if report.env_id == env_id
        ]
        if env_reports:
            total_generated += sum(report.generated for report in env_reports)
            for report in env_reports:
                collected_errors.extend(report.errors)
                collected_warnings.extend(report.warnings)
                if report.status != "completed":
                    collected_status = report.status
                if report.blocked_reason:
                    blocked_reason = report.blocked_reason
        completed_after = _completed_count(index=index, split_name=split_name, env_id=env_id)
        env_pending = [
            item
            for item in pending_samples_for_split(
                index=index,
                output_root=output_root,
                split_name=split_name,
                force=force,
            )
            if item.env_id == env_id
        ]
        debug_log(
            "datasets.build.split.env.pass.done",
            split=split_name,
            env_id=env_id,
            pass_index=pass_index,
            generated_this_pass=max(0, completed_after - completed_before),
            completed_after=completed_after,
            pending_count=len(env_pending),
        )
        debug_log(
            "datasets.build.split.env.index_refreshed",
            split=split_name,
            env_id=env_id,
            completed_before=completed_before,
            completed_after=completed_after,
            pending_count=len(env_pending),
            index_path=index_path(output_root),
        )
        if env_id == "phyre" and env_pending and completed_after == completed_before:
            raise RuntimeError(
                f"PHYRE dataset build made no progress for split {split_name} with {len(env_pending)} samples still pending"
            )
        if env_id != "phyre":
            break

    debug_log(
        "datasets.build.split.env.done",
        split=split_name,
        env_id=env_id,
        generated=total_generated,
        pending_count=len(env_pending),
        pass_count=pass_index,
    )
    return EnvironmentGenerationReport(
        env_id=env_id,
        status=collected_status,
        generated=total_generated,
        saved_dir=saved_dir,
        blocked_reason=blocked_reason,
        errors=collected_errors,
        warnings=collected_warnings,
    )


def _completed_count(*, index: DatasetIndex, split_name: str, env_id: str) -> int:
    return sum(
        1
        for entry in index.entries
        if entry.split_name == split_name
        and entry.env_id == env_id
        and entry.status == "completed"
    )


def _build_existing_dataset_run(
    *,
    root: Path,
    config_path: Path | None,
    chosen_splits: list,
) -> DatasetBuildRun:
    manifest = load_dataset_manifest(manifest_path(root))
    split_runs = {
        split.name: TraceGenerationRun(output_root=root / split.name, per_environment=[])
        for split in chosen_splits
    }
    split_specs = [
        DatasetSplitSpec(
            name=split.name,
            sample_count=split.sample_count,
            sample_offset=split.sample_offset,
        )
        for split in chosen_splits
    ]
    return DatasetBuildRun(
        output_root=root,
        env_ids=list(manifest.env_ids),
        splits=split_specs,
        split_runs=split_runs,
        dataset_name=manifest.dataset_name,
        config_version=manifest.config_version,
        config_path=config_path,
    )


def _ensure_dataset_metadata(
    *,
    config: DatasetBuildConfig,
    output_root: Path,
    dataset_id: str,
    chosen_splits: list,
    split_specs: list[DatasetSplitSpec],
    config_path: Path | None,
) -> tuple[DatasetManifest, DatasetIndex]:
    manifest_file = manifest_path(output_root)
    if manifest_file.is_file():
        manifest = load_dataset_manifest(manifest_file)
    else:
        manifest = _build_manifest(
            config=config,
            chosen_splits=chosen_splits,
            dataset_id=dataset_id,
            config_path=config_path,
        )
        write_dataset_manifest(manifest_file, manifest)

    index = _load_or_create_index(config=config, output_root=output_root, dataset_id=dataset_id)

    for split in chosen_splits:
        for env_id in config.env_ids:
            refresh_index_entries(index=index, output_root=output_root, split_name=split.name, env_id=env_id)
    write_dataset_index(index_path(output_root), index)

    _write_dataset_manifest_with_reports(
        output_root=output_root,
        manifest=manifest,
        config=config,
        split_specs=split_specs,
        split_runs={},
        index=index,
    )
    return manifest, index


def _selected_splits(
    *,
    config: DatasetBuildConfig,
    selected_splits: list[str] | None,
) -> list:
    if not selected_splits:
        return list(config.splits)

    split_map = {split.name: split for split in config.splits}
    unknown = [name for name in selected_splits if name not in split_map]
    if unknown:
        raise ValueError("unknown dataset splits requested: " + ", ".join(sorted(set(unknown))))
    return [split_map[name] for name in selected_splits]


def _prioritize_splits(*, chosen_splits: list, index: DatasetIndex, env_ids: list[str]) -> list:
    expected_per_split = {split.name: split.sample_count * len(env_ids) for split in chosen_splits}
    completed_per_split = {
        split.name: sum(
            1
            for entry in index.entries
            if entry.split_name == split.name and entry.status == "completed"
        )
        for split in chosen_splits
    }

    def key(split: object) -> tuple[float, int, str]:
        split_name = split.name
        expected = expected_per_split.get(split_name, 0)
        completed = completed_per_split.get(split_name, 0)
        ratio = (completed / expected) if expected else 1.0
        return (ratio, completed, split_name)

    return sorted(chosen_splits, key=key)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "dataset"


def _build_manifest(
    *,
    config: DatasetBuildConfig,
    chosen_splits: list,
    dataset_id: str,
    config_path: Path | None,
) -> DatasetManifest:
    return DatasetManifest(
        output_root=None,
        dataset_name=config.dataset_name,
        dataset_id=dataset_id,
        config_version=config.config_version,
        config_source=str(config_path) if config_path is not None else None,
        env_ids=list(config.env_ids),
        splits=[
            DatasetSplitManifest(
                name=split.name,
                sample_count=split.sample_count,
                sample_offset=split.sample_offset,
            )
            for split in chosen_splits
        ],
        notes=list(config.notes),
    )


def _load_or_create_index(
    *,
    config: DatasetBuildConfig,
    output_root: Path,
    dataset_id: str,
) -> DatasetIndex:
    path = index_path(output_root)
    if path.is_file():
        return load_dataset_index(path)
    return build_dataset_index(config=config, dataset_id=dataset_id)


def _write_dataset_manifest_with_reports(
    *,
    output_root: Path,
    manifest: DatasetManifest,
    config: DatasetBuildConfig,
    split_specs: list[DatasetSplitSpec],
    split_runs: dict[str, TraceGenerationRun],
    index: DatasetIndex,
) -> None:
    for split in manifest.splits:
        entries = [entry for entry in index.entries if entry.split_name == split.name]
        split.completed_samples = sum(1 for entry in entries if entry.status == "completed")
        split.env_counts = {
            env_id: sum(
                1 for entry in entries if entry.env_id == env_id and entry.status == "completed"
            )
            for env_id in config.env_ids
        }

    manifest.output_root = str(output_root)
    manifest.config = config.model_dump(mode="json")
    manifest.split_reports = {
        name: {
            "output_root": str(run.output_root),
            "per_environment": [report.__dict__ for report in run.per_environment],
        }
        for name, run in split_runs.items()
    }
    path = manifest_path(output_root)
    write_dataset_manifest(path, manifest)
    for split in manifest.splits:
        debug_log(
            "datasets.build.manifest.updated",
            dataset_name=manifest.dataset_name,
            split=split.name,
            completed_samples=split.completed_samples,
            sample_count=split.sample_count,
            env_counts=split.env_counts,
            manifest_path=path,
        )
