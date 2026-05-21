from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _services_root() -> Path:
    resolved = Path(__file__).resolve()
    for parent in resolved.parents:
        if (parent / "Work").is_dir() and (parent / "AGENTS.md").is_file():
            return parent
    return resolved.parents[4]


def _client_project_root_candidates() -> list[Path]:
    return [Path(__file__).resolve().parents[2]]


MONOREPO_ROOT = _services_root()
LEGACY_RUNTIME_SRC_CANDIDATES = [
    *(root / "src" for root in _client_project_root_candidates()),
    MONOREPO_ROOT / "src",
]
AVAILABLE_LEGACY_RUNTIME_SRCS = [
    candidate for candidate in LEGACY_RUNTIME_SRC_CANDIDATES if candidate.is_dir()
]
for candidate in reversed(AVAILABLE_LEGACY_RUNTIME_SRCS):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from pattern_library_schema import DetectorDefinition, PatternLibrary  # noqa: E402


_LABEL_DIR_PATTERN = re.compile(r"^label_(?P<index>\d{3})_(?P<label>.+)$")


@dataclass(frozen=True)
class CombinedLabelSummary:
    label: str
    label_index: int
    candidate_count: int
    completed: bool
    best_candidate_uid: str | None
    best_train_score: float | None


@dataclass(frozen=True)
class CombinedSupervisedRun:
    run_dir: Path
    run_name: str
    completed_label_count: int
    total_label_count: int
    candidate_count: int
    fully_completed: bool
    label_summaries: list[CombinedLabelSummary]


def combine_supervised_run_artifacts(
    *,
    project_root: Path,
    env_id: str,
    source_run_dirs: list[Path],
    target_run_dir: Path,
    max_candidate_evaluations: int = 200,
    min_candidate_evaluations: int = 100,
    no_improvement_patience_evaluations: int = 10,
) -> CombinedSupervisedRun:
    resolved_sources = [path.resolve() for path in source_run_dirs]
    if len(resolved_sources) < 2:
        raise ValueError("Need at least two supervised run directories to combine")
    if len({str(path) for path in resolved_sources}) != len(resolved_sources):
        raise ValueError("Cannot combine the same supervised run more than once")

    run_payloads = {path: _load_json_object(path / "run.json") for path in resolved_sources}
    expected_labels = _resolve_expected_labels(
        project_root=project_root,
        env_id=env_id,
        source_run_dirs=resolved_sources,
        run_payloads=run_payloads,
    )
    if not expected_labels:
        raise ValueError("Unable to determine supervised label set for the selected runs")

    if target_run_dir.exists():
        raise FileExistsError(f"Target combined run directory already exists: {target_run_dir}")
    target_run_dir.mkdir(parents=True, exist_ok=False)

    base_payload = next((payload for payload in run_payloads.values() if payload), {})
    evolution_records: list[dict[str, Any]] = []
    evolution_live_records: list[dict[str, Any]] = []
    detectors: list[DetectorDefinition] = []
    label_run_payloads: list[dict[str, Any]] = []
    label_summaries: list[CombinedLabelSummary] = []
    total_candidate_count = 0

    detector_input_mode = str(base_payload.get("detector_input_mode") or "trace").strip() or "trace"
    primitive_source = _resolve_primitive_source(
        project_root=project_root,
        env_id=env_id,
        run_payloads=run_payloads,
    )

    for label_index, label in enumerate(expected_labels, start=1):
        label_dir = target_run_dir / f"label_{label_index:03d}_{_slugify(label)}"
        label_dir.mkdir(parents=True, exist_ok=True)
        merged_candidates: list[dict[str, Any]] = []
        source_label_results: list[dict[str, Any]] = []

        for source_run_dir in resolved_sources:
            source_label_dir = source_run_dir / f"label_{label_index:03d}_{_slugify(label)}"
            if not source_label_dir.is_dir():
                matched = _find_label_dir(source_run_dir, label_index=label_index, label=label)
                if matched is not None:
                    source_label_dir = matched
            if not source_label_dir.is_dir():
                continue

            label_result_payload = _load_json_object(source_label_dir / "label_result.json")
            if label_result_payload:
                source_label_results.append(label_result_payload)

            merged_candidates.extend(
                _copy_candidate_artifacts(
                    source_run_name=source_run_dir.name,
                    source_label_dir=source_label_dir,
                    target_label_dir=label_dir,
                )
            )

            if not merged_candidates and label_result_payload:
                synthesized = _synthesize_candidate_from_label_result(
                    source_run_name=source_run_dir.name,
                    source_label_dir=source_label_dir,
                    target_label_dir=label_dir,
                    label=label,
                    label_index=label_index,
                )
                if synthesized is not None:
                    merged_candidates.append(synthesized)

        merged_candidates = _limit_combined_label_candidates(
            label_dir=label_dir,
            candidates=merged_candidates,
            max_candidate_evaluations=int(max_candidate_evaluations),
        )
        total_candidate_count += len(merged_candidates)
        best_candidate = _best_candidate_by_train_score(merged_candidates)
        is_complete = bool(source_label_results) or len(merged_candidates) >= int(max_candidate_evaluations)

        if best_candidate is not None:
            _mark_best_candidate(label_dir=label_dir, chosen_uid=str(best_candidate.get("uid") or ""))
        if is_complete and best_candidate is not None:
            detector = _detector_from_candidate_payload(
                label=label,
                label_index=label_index,
                environment_id=env_id,
                candidate_payload=best_candidate,
                detector_input_mode=detector_input_mode,
            )
            detectors.append(detector)
            label_result_payload = _build_label_result_payload(
                label=label,
                label_index=label_index,
                best_candidate=best_candidate,
                detector=detector,
                candidate_count=len(merged_candidates),
                stop_reason=(
                    "combined_candidate_cap_reached"
                    if len(merged_candidates) >= int(max_candidate_evaluations)
                    else "combined_from_completed_source"
                ),
                patience_evaluations=no_improvement_patience_evaluations,
                label_dir=label_dir,
                source_label_results=source_label_results,
            )
            _write_json(label_dir / "label_result.json", label_result_payload)
            label_run_payloads.append(
                {
                    "label": label,
                    "best_candidate_uid": best_candidate.get("uid"),
                    "best_score": best_candidate.get("average_score"),
                    "train_score": best_candidate.get("train_score"),
                    "validation_score": best_candidate.get("val_score"),
                    "loss_breakdown": list(best_candidate.get("loss_breakdown") or []),
                    "validation_loss_breakdown": list(best_candidate.get("validation_loss_breakdown") or []),
                    "timing_summary": dict(best_candidate.get("timing_summary") or {}),
                    "worker": 0,
                    "candidate_evaluations": len(merged_candidates),
                    "stopped_early": True,
                    "stop_reason": label_result_payload["stop_reason"],
                    "patience_evaluations": no_improvement_patience_evaluations,
                    "artifact_dir": str(label_dir),
                    "run_dir": label_result_payload.get("run_dir"),
                }
            )

        for candidate in merged_candidates:
            record = {
                "label_index": label_index,
                "label": label,
                "candidate_uid": candidate.get("uid"),
                "parent_uid": candidate.get("parent_uid"),
                "secondary_parent_uid": candidate.get("secondary_parent_uid"),
                "origin": candidate.get("origin"),
                "iteration": candidate.get("iteration"),
                "score": candidate.get("average_score"),
                "train_score": candidate.get("train_score"),
                "val_score": candidate.get("val_score"),
            }
            evolution_records.append(record)
            evolution_live_records.append(dict(record))

        label_summaries.append(
            CombinedLabelSummary(
                label=label,
                label_index=label_index,
                candidate_count=len(merged_candidates),
                completed=is_complete and best_candidate is not None,
                best_candidate_uid=(
                    str(best_candidate.get("uid") or "").strip() or None
                    if best_candidate is not None
                    else None
                ),
                best_train_score=_coerce_optional_float(
                    best_candidate.get("train_score") if best_candidate is not None else None
                ),
            )
        )

    _write_jsonl(target_run_dir / "evolution_record.jsonl", evolution_records)
    _write_jsonl(target_run_dir / "evolution_live.jsonl", evolution_live_records)
    (target_run_dir / "llm_calls.jsonl").touch()
    _write_json(
        target_run_dir / "run_control.json",
        {
            "desired_workers": 1,
            "configured_workers": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    train_score = _aggregate_metric(label_run_payloads, "train_score")
    validation_score = _aggregate_metric(label_run_payloads, "validation_score")
    average_score = train_score
    loss_breakdown = _aggregate_loss_breakdown(label_run_payloads)
    timing_summary = _aggregate_timing_summary(label_run_payloads)

    library = PatternLibrary(
        library_id=str(base_payload.get("library_id") or target_run_dir.name),
        environment_ids=[env_id],
        primitive_source=primitive_source,
        scorer_config=_scorer_config_with_server_policy(
            dict(base_payload.get("scorer_config") or {}),
            min_candidate_evaluations=min_candidate_evaluations,
            no_improvement_patience_evaluations=no_improvement_patience_evaluations,
            max_candidate_evaluations=max_candidate_evaluations,
        ),
        detectors=detectors,
        metadata={
            "combined_from_run_dirs": [str(path) for path in resolved_sources],
            "combined_at": datetime.now(timezone.utc).isoformat(),
            "label_runs": label_run_payloads,
            "completed_label_count": len(detectors),
        },
    )
    _write_library_json(target_run_dir / "pattern_library.json", library)

    run_payload = {
        "library_id": library.library_id,
        "environment_ids": [env_id],
        "status": "completed" if len(detectors) >= len(expected_labels) else "running",
        "sample_count": int(base_payload.get("sample_count") or 0),
        "train_sample_count": int(base_payload.get("train_sample_count") or 0),
        "validation_sample_count": int(base_payload.get("validation_sample_count") or 0),
        "iteration_count": max(
            [int(payload.get("iteration_count") or 0) for payload in run_payloads.values()] or [0]
        ),
        "workers": 1,
        "configured_workers": 1,
        "primitive_label_count": len(expected_labels),
        "completed_label_count": len(detectors),
        "manually_completed_labels": [],
        "average_score": average_score,
        "train_score": train_score,
        "validation_score": validation_score,
        "dataset_id": base_payload.get("dataset_id"),
        "dataset_version": base_payload.get("dataset_version"),
        "primitive_source": primitive_source,
        "scorer_config": library.scorer_config,
        "detector_count": len(detectors),
        "min_candidate_evaluations": int(min_candidate_evaluations),
        "patience_evaluations": int(no_improvement_patience_evaluations),
        "no_improvement_patience_evaluations": int(no_improvement_patience_evaluations),
        "max_candidate_evaluations": int(max_candidate_evaluations),
        "detector_input_mode": detector_input_mode if detector_input_mode in {"trace", "supervised_evidence"} else "trace",
        "llm_calls_path": str(target_run_dir / "llm_calls.jsonl"),
        "worker_control_path": str(target_run_dir / "run_control.json"),
        "loss_breakdown": loss_breakdown,
        "timing_summary": timing_summary,
        "score_details": [],
        "label_runs": label_run_payloads,
        "evolution_record_path": str(target_run_dir / "evolution_record.jsonl"),
    }
    _write_json(target_run_dir / "run.json", run_payload)

    return CombinedSupervisedRun(
        run_dir=target_run_dir,
        run_name=target_run_dir.name,
        completed_label_count=len(detectors),
        total_label_count=len(expected_labels),
        candidate_count=total_candidate_count,
        fully_completed=len(detectors) >= len(expected_labels),
        label_summaries=label_summaries,
    )


def _resolve_expected_labels(
    *,
    project_root: Path,
    env_id: str,
    source_run_dirs: list[Path],
    run_payloads: dict[Path, dict[str, Any]],
) -> list[str]:
    for payload in run_payloads.values():
        primitive_source = str(payload.get("primitive_source") or "").strip()
        if primitive_source:
            candidate = Path(primitive_source)
            if candidate.is_file():
                labels = _load_labels_file(candidate)
                if labels:
                    return labels
    for labels_path in (
        root / "configs" / env_id / "labels.txt"
        for root in _client_project_root_candidates()
    ):
        if labels_path.is_file():
            labels = _load_labels_file(labels_path)
            if labels:
                return labels
    indexed: dict[int, str] = {}
    for run_dir in source_run_dirs:
        for label_dir in sorted(path for path in run_dir.glob("label_*") if path.is_dir()):
            parsed = _parse_label_dir_name(label_dir.name)
            if parsed is None:
                continue
            label_index, label = parsed
            indexed.setdefault(label_index, label)
    return [label for _, label in sorted(indexed.items())]


def _resolve_primitive_source(
    *,
    project_root: Path,
    env_id: str,
    run_payloads: dict[Path, dict[str, Any]],
) -> str | None:
    for payload in run_payloads.values():
        primitive_source = str(payload.get("primitive_source") or "").strip()
        if primitive_source:
            return primitive_source
    for candidate in (
        root / "configs" / env_id / "labels.txt"
        for root in _client_project_root_candidates()
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _copy_candidate_artifacts(
    *,
    source_run_name: str,
    source_label_dir: Path,
    target_label_dir: Path,
) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    existing = sorted(path for path in target_label_dir.glob("candidate_*") if path.is_dir())
    next_index = len(existing) + 1
    for source_candidate_dir in sorted(path for path in source_label_dir.glob("candidate_*") if path.is_dir()):
        candidate_payload = _load_json_object(source_candidate_dir / "candidate.json")
        if not candidate_payload:
            continue
        code_path = next((path for path in sorted(source_candidate_dir.glob("*.py")) if path.is_file()), None)
        if code_path is None:
            continue
        original_uid = str(
            candidate_payload.get("uid")
            or candidate_payload.get("candidate_uid")
            or source_candidate_dir.name
        ).strip()
        new_uid = f"{source_run_name}__{original_uid}"
        rewritten = dict(candidate_payload)
        rewritten["uid"] = new_uid
        rewritten["candidate_uid"] = new_uid
        if isinstance(candidate_payload.get("parent_uid"), str) and candidate_payload.get("parent_uid"):
            rewritten["parent_uid"] = f"{source_run_name}__{candidate_payload['parent_uid']}"
        if isinstance(candidate_payload.get("secondary_parent_uid"), str) and candidate_payload.get("secondary_parent_uid"):
            rewritten["secondary_parent_uid"] = f"{source_run_name}__{candidate_payload['secondary_parent_uid']}"
        rewritten["candidate_index"] = next_index
        rewritten["is_best_candidate"] = False
        metadata = dict(rewritten.get("metadata") or {})
        metadata["combined_source_run"] = source_run_name
        metadata["combined_from_candidate_dir"] = str(source_candidate_dir)
        rewritten["metadata"] = metadata
        target_candidate_dir = target_label_dir / f"candidate_{next_index:03d}_{_slugify(new_uid)}"
        target_candidate_dir.mkdir(parents=True, exist_ok=False)
        target_code_path = target_candidate_dir / code_path.name
        shutil.copy2(code_path, target_code_path)
        rewritten["source_path"] = str(target_code_path)
        _write_json(target_candidate_dir / "candidate.json", rewritten)
        copied.append(rewritten)
        next_index += 1
    return copied


def _synthesize_candidate_from_label_result(
    *,
    source_run_name: str,
    source_label_dir: Path,
    target_label_dir: Path,
    label: str,
    label_index: int,
) -> dict[str, Any] | None:
    label_result_payload = _load_json_object(source_label_dir / "label_result.json")
    detector_payload = label_result_payload.get("detector")
    if not isinstance(detector_payload, dict):
        return None
    source_code = detector_payload.get("source_code")
    if not isinstance(source_code, str) or not source_code.strip():
        return None
    uid = f"{source_run_name}__{str(label_result_payload.get('best_candidate_uid') or 'synthetic_best').strip()}"
    candidate_dir = target_label_dir / f"candidate_001_{_slugify(uid)}"
    if candidate_dir.exists():
        return _load_json_object(candidate_dir / "candidate.json") or None
    candidate_dir.mkdir(parents=True, exist_ok=False)
    code_path = candidate_dir / "find_pattern.py"
    code_path.write_text(source_code.rstrip() + "\n", encoding="utf-8")
    candidate_payload = {
        "label": label,
        "candidate_index": 1,
        "uid": uid,
        "parent_uid": None,
        "secondary_parent_uid": None,
        "origin": "combined-synthesized",
        "iteration": 0,
        "score": label_result_payload.get("best_score"),
        "train_score": label_result_payload.get("best_train_score"),
        "val_score": label_result_payload.get("best_validation_score"),
        "fingerprint": None,
        "metadata": {"combined_source_run": source_run_name, "synthetic": True},
        "source_path": str(code_path),
        "is_best_candidate": False,
        "average_score": label_result_payload.get("best_score"),
        "validation_score": label_result_payload.get("best_validation_score"),
        "loss_breakdown": list(label_result_payload.get("best_loss_breakdown") or []),
        "validation_loss_breakdown": list(label_result_payload.get("best_validation_loss_breakdown") or []),
        "timing_summary": dict(label_result_payload.get("timing_summary") or {}),
        "score_details": list(label_result_payload.get("best_score_details") or []),
        "errors": [],
        "label_index": label_index,
    }
    _write_json(candidate_dir / "candidate.json", candidate_payload)
    return candidate_payload


def _detector_from_candidate_payload(
    *,
    label: str,
    label_index: int,
    environment_id: str,
    candidate_payload: dict[str, Any],
    detector_input_mode: str,
) -> DetectorDefinition:
    source_path = Path(str(candidate_payload.get("source_path") or ""))
    source_code = source_path.read_text(encoding="utf-8") if source_path.is_file() else None
    metadata = dict(candidate_payload.get("metadata") or {})
    metadata.update(
        {
            "candidate_uid": candidate_payload.get("uid"),
            "parent_uid": candidate_payload.get("parent_uid"),
            "secondary_parent_uid": candidate_payload.get("secondary_parent_uid"),
            "origin": candidate_payload.get("origin"),
            "score": candidate_payload.get("average_score"),
            "train_score": candidate_payload.get("train_score"),
            "val_score": candidate_payload.get("val_score"),
        }
    )
    metadata = {key: value for key, value in metadata.items() if value is not None}
    return DetectorDefinition(
        detector_id=f"primitive-{label_index:03d}",
        label=label,
        environment_ids=[environment_id],
        source_code=source_code,
        input_mode=(
            detector_input_mode if detector_input_mode in {"trace", "supervised_evidence"} else "trace"
        ),
        metadata=metadata,
    )


def _build_label_result_payload(
    *,
    label: str,
    label_index: int,
    best_candidate: dict[str, Any],
    detector: DetectorDefinition,
    candidate_count: int,
    stop_reason: str,
    patience_evaluations: int,
    label_dir: Path,
    source_label_results: list[dict[str, Any]],
) -> dict[str, Any]:
    artifacts = {
        "log_root": str(label_dir),
        "run_dir": next(
            (str(path) for path in sorted(label_dir.glob("funsearch_run*")) if path.is_dir()),
            None,
        ),
    }
    best_worker = 0
    for item in source_label_results:
        if isinstance(item.get("worker"), int):
            best_worker = max(best_worker, int(item["worker"]))
    return {
        "label": label,
        "label_index": label_index,
        "worker": best_worker,
        "candidate_evaluations": candidate_count,
        "stopped_early": True,
        "stop_reason": stop_reason,
        "patience_evaluations": patience_evaluations,
        "detector": detector.model_dump(mode="json"),
        "best_score": best_candidate.get("average_score"),
        "best_train_score": best_candidate.get("train_score"),
        "best_validation_score": best_candidate.get("val_score"),
        "best_score_details": list(best_candidate.get("score_details") or []),
        "best_loss_breakdown": list(best_candidate.get("loss_breakdown") or []),
        "best_validation_loss_breakdown": list(best_candidate.get("validation_loss_breakdown") or []),
        "timing_summary": dict(best_candidate.get("timing_summary") or {}),
        "timings": {"totals": {}, "counts": {}},
        "artifacts": artifacts,
        "best_candidate_uid": best_candidate.get("uid"),
        "run_dir": artifacts.get("run_dir"),
    }


def _best_candidate_by_train_score(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=_candidate_rank_key,
    )


def _limit_combined_label_candidates(
    *,
    label_dir: Path,
    candidates: list[dict[str, Any]],
    max_candidate_evaluations: int,
) -> list[dict[str, Any]]:
    if max_candidate_evaluations <= 0:
        return []
    if len(candidates) <= max_candidate_evaluations:
        return list(candidates)
    ranked = sorted(candidates, key=_candidate_rank_key, reverse=True)
    kept_ranked = ranked[:max_candidate_evaluations]
    kept_source_paths = {
        str(item.get("source_path") or "").strip()
        for item in kept_ranked
        if str(item.get("source_path") or "").strip()
    }
    for payload in candidates:
        source_path = str(payload.get("source_path") or "").strip()
        if not source_path or source_path in kept_source_paths:
            continue
        candidate_dir = Path(source_path).parent
        try:
            if candidate_dir.is_dir() and candidate_dir.parent == label_dir:
                shutil.rmtree(candidate_dir, ignore_errors=True)
        except OSError:
            continue
    return sorted(
        kept_ranked,
        key=lambda payload: int(payload.get("candidate_index") or 0),
    )


def _candidate_rank_key(payload: dict[str, Any]) -> tuple[float, float, float, int, str]:
    return (
        _score_value(payload.get("train_score")),
        _score_value(payload.get("average_score")),
        _score_value(payload.get("val_score")),
        -int(payload.get("candidate_index") or 0),
        str(payload.get("uid") or ""),
    )


def _mark_best_candidate(*, label_dir: Path, chosen_uid: str) -> None:
    for candidate_dir in sorted(path for path in label_dir.glob("candidate_*") if path.is_dir()):
        payload = _load_json_object(candidate_dir / "candidate.json")
        if not payload:
            continue
        payload["is_best_candidate"] = str(payload.get("uid") or "") == chosen_uid
        _write_json(candidate_dir / "candidate.json", payload)


def _aggregate_metric(label_run_payloads: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(item[key])
        for item in label_run_payloads
        if isinstance(item.get(key), (int, float))
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _aggregate_loss_breakdown(label_run_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accum: dict[str, dict[str, Any]] = {}
    for item in label_run_payloads:
        for component in item.get("loss_breakdown", []):
            if not isinstance(component, dict):
                continue
            kind = str(component.get("kind") or "").strip()
            if not kind:
                continue
            bucket = accum.setdefault(
                kind,
                {"kind": kind, "weight": component.get("weight"), "raw_scores": [], "weighted_scores": []},
            )
            if isinstance(component.get("raw_score"), (int, float)):
                bucket["raw_scores"].append(float(component["raw_score"]))
            if isinstance(component.get("weighted_score"), (int, float)):
                bucket["weighted_scores"].append(float(component["weighted_score"]))
    output: list[dict[str, Any]] = []
    for kind, bucket in sorted(accum.items()):
        raw_scores = bucket["raw_scores"] or [0.0]
        weighted_scores = bucket["weighted_scores"] or [0.0]
        output.append(
            {
                "kind": kind,
                "weight": bucket.get("weight"),
                "mean_raw_score": sum(raw_scores) / len(raw_scores),
                "mean_weighted_score": sum(weighted_scores) / len(weighted_scores),
            }
        )
    return output


def _aggregate_timing_summary(label_run_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    total_wall_seconds = 0.0
    for item in label_run_payloads:
        timing_summary = item.get("timing_summary")
        if isinstance(timing_summary, dict) and isinstance(timing_summary.get("total_wall_seconds"), (int, float)):
            total_wall_seconds += float(timing_summary["total_wall_seconds"])
    return {"total_wall_seconds": total_wall_seconds, "groups": {}}


def _scorer_config_with_server_policy(
    scorer_config: dict[str, Any],
    *,
    min_candidate_evaluations: int,
    no_improvement_patience_evaluations: int,
    max_candidate_evaluations: int,
) -> dict[str, Any]:
    rewritten = dict(scorer_config or {})
    rewritten["min_candidate_evaluations"] = int(min_candidate_evaluations)
    rewritten["no_improvement_patience_evaluations"] = int(no_improvement_patience_evaluations)
    rewritten["max_candidate_evaluations"] = int(max_candidate_evaluations)
    rewritten.pop("patience_factor", None)
    return rewritten


def _find_label_dir(run_dir: Path, *, label_index: int, label: str) -> Path | None:
    normalized_target = _normalize_label_key(label)
    for path in sorted(run_dir.glob("label_*")):
        if not path.is_dir():
            continue
        parsed = _parse_label_dir_name(path.name)
        if parsed is None:
            continue
        parsed_index, parsed_label = parsed
        if parsed_index != label_index:
            continue
        if _normalize_label_key(parsed_label) == normalized_target:
            return path
    return None


def _parse_label_dir_name(name: str) -> tuple[int, str] | None:
    match = _LABEL_DIR_PATTERN.match(name)
    if match is None:
        return None
    return int(match.group("index")), match.group("label")


def _load_labels_file(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [line.strip() for line in lines if line.strip()]


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_library_json(path: Path, library: PatternLibrary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(library.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _score_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float("-inf")


def _coerce_optional_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip())
    text = text.strip("-._")
    return text or "candidate"


def _normalize_label_key(value: str) -> str:
    return re.sub(r"[-_\s]+", "-", str(value or "").strip().lower()).strip("-")
