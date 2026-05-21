from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from config_loading import load_yaml_config
from datasets.manifest import DatasetManifest, load_dataset_manifest, resolve_dataset_root
from debug import debug_log, set_debug_log_file
from llm_config import apply_shared_llm_defaults
from pattern_learning.patterns.prompts import build_prediction_request_prompt
from pattern_learning.patterns.scoring import PatternResponder
from schema import CanonicalTrace
from deepphy_eval.io import load_trace


PREDICTION_CONFIG_VERSION = "1"
PREDICTION_SCHEMA_VERSION = "1"


class PredictionSplitManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    sample_count: int
    prediction_count: int
    completed_samples: int = 0


class PredictionDatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_schema_version: str = PREDICTION_SCHEMA_VERSION
    output_root: str
    dataset_name: str
    dataset_id: str
    config_version: str
    config_source: Optional[str] = None
    source_dataset_manifest: str
    source_dataset_id: str
    source_dataset_version: str
    env_ids: list[str]
    splits: list[PredictionSplitManifest]
    notes: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    split_reports: dict[str, Any] = Field(default_factory=dict)


class PredictionDatasetBuildConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_version: str = PREDICTION_CONFIG_VERSION
    env_id: str | None = None
    dataset_name: str | None = None
    source_dataset_manifest: str | None = None
    output_path: str | None = None
    split: list[str] = Field(default_factory=list)
    limit: Optional[int] = Field(default=None, ge=1)
    cutoff_count: int = Field(default=5, ge=1)
    cutoff_fractions: list[float] = Field(default_factory=list)
    prediction_model: str | None = None
    prediction_base_url: str | None = None
    prediction_timeout: int = Field(default=120, ge=1)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_schedule(self) -> "PredictionDatasetBuildConfig":
        if self.cutoff_fractions:
            normalized = [float(value) for value in self.cutoff_fractions]
            if any(value <= 0.0 or value >= 1.0 for value in normalized):
                raise ValueError("cutoff_fractions must all be strictly between 0 and 1")
        return self


class PredictionArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: str
    source_environment: str
    source_task_id: str
    source_trace_path: str
    cutoff_index: int
    cutoff_t: Optional[float] = None
    cutoff_fraction: float
    observed_timestep_count: int
    total_timestep_count: int
    model: str
    base_url: str
    generated_at: str
    prediction: str


@dataclass(frozen=True)
class PredictionDatasetRun:
    output_root: Path
    dataset_name: str
    dataset_id: str
    config_version: str
    source_dataset_manifest: Path
    env_ids: list[str]
    split_reports: dict[str, Any]
    config_path: Path | None = None


@dataclass(frozen=True)
class StoredPredictionTarget:
    trace_path: Path
    prediction_path: Path
    environment_id: str
    task_id: str
    prediction_text: str
    cutoff_index: int
    cutoff_t: Optional[float]
    cutoff_fraction: float
    observed_timestep_count: int
    total_timestep_count: int
    task_context: dict[str, Any]


def default_prediction_dataset_config_path() -> Path:
    from paths import get_artifact_layout

    return get_artifact_layout().config_root / "prediction_dataset.yaml"


def load_prediction_dataset_build_config(path: Path) -> PredictionDatasetBuildConfig:
    data = load_yaml_config(path)
    data = apply_shared_llm_defaults(
        data,
        model_key="prediction_model",
        base_url_key="prediction_base_url",
    )
    return PredictionDatasetBuildConfig.model_validate(data)


def prediction_manifest_path(output_root: Path) -> Path:
    return output_root / "prediction_manifest.json"


def load_prediction_dataset_manifest(path: Path) -> PredictionDatasetManifest:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return PredictionDatasetManifest.model_validate(payload)


def build_prediction_dataset(
    *,
    config: PredictionDatasetBuildConfig,
    responder: PatternResponder,
    output_root: Path | None = None,
    force: bool = False,
    config_path: Path | None = None,
) -> PredictionDatasetRun:
    root = output_root or Path(config.output_path)
    root.mkdir(parents=True, exist_ok=True)
    set_debug_log_file(root / "debug.log")
    debug_log(
        "predictions.build.start",
        source_dataset_manifest=config.source_dataset_manifest,
        output_root=root,
        cutoff_count=config.cutoff_count,
        split=config.split,
        limit=config.limit,
        prediction_model=config.prediction_model,
    )

    if not force and _prediction_dataset_ready(root):
        manifest = load_prediction_dataset_manifest(prediction_manifest_path(root))
        debug_log("predictions.build.skip_existing_ready", output_root=root, dataset_name=manifest.dataset_name)
        return PredictionDatasetRun(
            output_root=root,
            dataset_name=manifest.dataset_name,
            dataset_id=manifest.dataset_id,
            config_version=manifest.config_version,
            source_dataset_manifest=Path(manifest.source_dataset_manifest),
            env_ids=list(manifest.env_ids),
            split_reports=dict(manifest.split_reports),
            config_path=config_path,
        )

    source_manifest_path = Path(config.source_dataset_manifest)
    source_manifest = load_dataset_manifest(source_manifest_path)
    responder_health = responder.require_healthy()

    split_names = set(config.split)
    selected_splits = [
        split for split in source_manifest.splits if not split_names or split.name in split_names
    ]
    if split_names:
        unknown = split_names - {split.name for split in source_manifest.splits}
        if unknown:
            raise ValueError("unknown prediction dataset splits requested: " + ", ".join(sorted(unknown)))

    split_reports: dict[str, Any] = {}
    manifest_splits: list[PredictionSplitManifest] = []
    source_root = resolve_dataset_root(source_manifest_path, source_manifest)
    dataset_id = root.name

    for split in selected_splits:
        split_root = root / split.name
        split_root.mkdir(parents=True, exist_ok=True)
        source_split_root = source_root / split.name
        sample_dirs = sorted(path for path in source_split_root.glob("sample_*") if path.is_dir())
        if config.limit is not None:
            sample_dirs = sample_dirs[: config.limit]
        prediction_count = 0
        for sample_dir in sample_dirs:
            trace_path = sample_dir / "trace.json"
            trace = load_trace(trace_path)
            cutoff_indices = derive_cutoff_indices(
                trace=trace,
                cutoff_count=config.cutoff_count,
                cutoff_fractions=config.cutoff_fractions,
            )
            target_sample_dir = split_root / sample_dir.name
            target_prediction_root = target_sample_dir / "predictions"
            target_prediction_root.mkdir(parents=True, exist_ok=True)
            for prediction_index, cutoff_index in enumerate(cutoff_indices, start=1):
                truncated = truncate_trace_for_cutoff(trace, cutoff_index)
                cutoff_fraction = cutoff_index / max(len(trace.timesteps) - 1, 1)
                prompt = build_prediction_request_prompt(
                    trace=trace,
                    cutoff_index=cutoff_index,
                )
                prediction_text = responder.respond(prompt)
                prediction_payload = PredictionArtifact(
                    prediction_id=f"prediction_{prediction_index:03d}",
                    source_environment=trace.environment,
                    source_task_id=trace.task_id,
                    source_trace_path=str(trace_path),
                    cutoff_index=cutoff_index,
                    cutoff_t=truncated.timesteps[-1].t if truncated.timesteps else None,
                    cutoff_fraction=cutoff_fraction,
                    observed_timestep_count=len(truncated.timesteps),
                    total_timestep_count=len(trace.timesteps),
                    model=config.prediction_model,
                    base_url=config.prediction_base_url,
                    generated_at=datetime.now(timezone.utc).isoformat(),
                    prediction=prediction_text,
                )
                prediction_dir = target_prediction_root / prediction_payload.prediction_id
                prediction_dir.mkdir(parents=True, exist_ok=True)
                with (prediction_dir / "prediction.json").open("w", encoding="utf-8") as handle:
                    json.dump(prediction_payload.model_dump(mode="json"), handle, indent=2)
                    handle.write("\n")
                prediction_count += 1
                debug_log(
                    "predictions.build.prediction.done",
                    split=split.name,
                    sample_dir=sample_dir.name,
                    prediction_id=prediction_payload.prediction_id,
                    cutoff_index=cutoff_index,
                )
        manifest_splits.append(
            PredictionSplitManifest(
                name=split.name,
                sample_count=len(sample_dirs),
                prediction_count=prediction_count,
                completed_samples=len(sample_dirs),
            )
        )
        split_reports[split.name] = {
            "source_split_root": str(source_split_root),
            "output_root": str(split_root),
            "sample_count": len(sample_dirs),
            "prediction_count": prediction_count,
        }

    manifest = PredictionDatasetManifest(
        output_root=str(root),
        dataset_name=config.dataset_name,
        dataset_id=dataset_id,
        config_version=config.config_version,
        config_source=str(config_path) if config_path is not None else None,
        source_dataset_manifest=str(source_manifest_path),
        source_dataset_id=source_manifest.dataset_id,
        source_dataset_version=source_manifest.config_version,
        env_ids=list(source_manifest.env_ids),
        splits=manifest_splits,
        notes=list(config.notes),
        config={
            **config.model_dump(mode="json"),
            "responder_health": responder_health,
        },
        split_reports=split_reports,
    )
    with prediction_manifest_path(root).open("w", encoding="utf-8") as handle:
        json.dump(manifest.model_dump(mode="json"), handle, indent=2)
        handle.write("\n")
    debug_log("predictions.build.done", output_root=root, dataset_name=config.dataset_name)
    return PredictionDatasetRun(
        output_root=root,
        dataset_name=config.dataset_name,
        dataset_id=dataset_id,
        config_version=config.config_version,
        source_dataset_manifest=source_manifest_path,
        env_ids=list(source_manifest.env_ids),
        split_reports=split_reports,
        config_path=config_path,
    )


def load_learning_samples_from_prediction_manifest(
    manifest_path: Path,
    *,
    split_names: list[str] | None = None,
    limit: int | None = None,
) -> tuple[list[StoredPredictionTarget], dict[str, Any]]:
    manifest = load_prediction_dataset_manifest(manifest_path)
    source_manifest = load_dataset_manifest(Path(manifest.source_dataset_manifest))
    prediction_root = Path(manifest.output_root)
    source_root = resolve_dataset_root(Path(manifest.source_dataset_manifest), source_manifest)
    allowed_splits = set(split_names or [])
    items: list[StoredPredictionTarget] = []
    sample_count = 0
    for split in manifest.splits:
        if allowed_splits and split.name not in allowed_splits:
            continue
        split_root = prediction_root / split.name
        for sample_dir in sorted(path for path in split_root.glob("sample_*") if path.is_dir()):
            sample_count += 1
            trace_path = source_root / split.name / sample_dir.name / "trace.json"
            trace = load_trace(trace_path)
            for prediction_path in sorted(sample_dir.glob("predictions/*/prediction.json")):
                with prediction_path.open("r", encoding="utf-8") as handle:
                    payload = PredictionArtifact.model_validate(json.load(handle))
                items.append(
                    StoredPredictionTarget(
                        trace_path=trace_path,
                        prediction_path=prediction_path,
                        environment_id=trace.environment,
                        task_id=trace.task_id,
                        prediction_text=payload.prediction,
                        cutoff_index=payload.cutoff_index,
                        cutoff_t=payload.cutoff_t,
                        cutoff_fraction=payload.cutoff_fraction,
                        observed_timestep_count=payload.observed_timestep_count,
                        total_timestep_count=payload.total_timestep_count,
                        task_context={},
                    )
                )
            if limit is not None and sample_count >= limit:
                return items, {
                    "dataset_id": manifest.dataset_id,
                    "dataset_version": manifest.config_version,
                    "source_dataset_id": manifest.source_dataset_id,
                    "source_dataset_version": manifest.source_dataset_version,
                }
    return items, {
        "dataset_id": manifest.dataset_id,
        "dataset_version": manifest.config_version,
        "source_dataset_id": manifest.source_dataset_id,
        "source_dataset_version": manifest.source_dataset_version,
    }


def truncate_trace_for_cutoff(trace: CanonicalTrace, cutoff_index: int) -> CanonicalTrace:
    if cutoff_index < 0 or cutoff_index >= len(trace.timesteps):
        raise ValueError(f"cutoff_index {cutoff_index} is out of range for trace with {len(trace.timesteps)} timesteps")
    cutoff_step = trace.timesteps[cutoff_index]
    cutoff_t = cutoff_step.t
    events = trace.events
    if cutoff_t is not None:
        events = [
            event for event in trace.events
            if event.t is None or event.t <= cutoff_t
        ]
    return CanonicalTrace.model_validate(
        {
            **trace.model_dump(mode="json"),
            "timesteps": [step.model_dump(mode="json") for step in trace.timesteps[: cutoff_index + 1]],
            "events": [event.model_dump(mode="json") for event in events],
        }
    )


def derive_cutoff_indices(
    *,
    trace: CanonicalTrace,
    cutoff_count: int,
    cutoff_fractions: list[float],
) -> list[int]:
    total_steps = len(trace.timesteps)
    max_cutoff_index = total_steps - 2
    if total_steps < 2:
        raise ValueError(f"Trace {trace.trace_id} does not have enough timesteps for prediction cutoffs")
    if cutoff_fractions:
        indices = [
            min(max(int(math.floor(value * (total_steps - 1))), 0), max_cutoff_index)
            for value in cutoff_fractions
        ]
    else:
        indices = [
            min(max(int(math.floor(((idx + 1) / (cutoff_count + 1)) * (total_steps - 1))), 0), max_cutoff_index)
            for idx in range(cutoff_count)
        ]
    unique = sorted(dict.fromkeys(indices))
    expected = len(cutoff_fractions) if cutoff_fractions else cutoff_count
    if len(unique) != expected:
        raise ValueError(
            f"Trace {trace.trace_id} could not satisfy {expected} distinct cutoff points; got {unique}"
        )
    return unique


def _prediction_dataset_ready(output_root: Path) -> bool:
    manifest_file = prediction_manifest_path(output_root)
    if not manifest_file.is_file():
        return False
    try:
        manifest = load_prediction_dataset_manifest(manifest_file)
    except Exception:  # noqa: BLE001
        return False
    for split in manifest.splits:
        split_root = output_root / split.name
        sample_dirs = sorted(path for path in split_root.glob("sample_*") if path.is_dir())
        if len(sample_dirs) != split.sample_count:
            return False
        observed_predictions = 0
        for sample_dir in sample_dirs:
            observed_predictions += sum(1 for _ in sample_dir.glob("predictions/*/prediction.json"))
        if observed_predictions != split.prediction_count:
            return False
    return True
