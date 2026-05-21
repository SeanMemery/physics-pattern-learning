from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pattern_server.env_catalog import LibraryEntry, list_environment_libraries


SUPERVISED_LABEL_IMPORTANCE_FILENAME = "supervised_label_importance.json"
SUPERVISED_LABEL_IMPORTANCE_SCHEMA_VERSION = "supervised_label_importance.v1"


def supervised_label_importance_path(run_dir: Path) -> Path:
    return run_dir / SUPERVISED_LABEL_IMPORTANCE_FILENAME


def load_supervised_label_importance(run_dir: Path) -> dict[str, Any] | None:
    path = supervised_label_importance_path(run_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def resolve_latest_supervised_library(project_root: Path, env_id: str) -> LibraryEntry | None:
    libraries = [
        item
        for item in list_environment_libraries(project_root, env_id)
        if item.workflow_kind == "supervised"
    ]
    return libraries[0] if libraries else None


def resolve_supervised_evaluation_root(run_dir: Path) -> tuple[Path | None, str | None]:
    run_json_path = run_dir / "run.json"
    requested_run_id: str | None = None
    if run_json_path.is_file():
        try:
            payload = json.loads(run_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            raw = payload.get("auto_evaluation_run_id")
            if isinstance(raw, str) and raw.strip():
                requested_run_id = raw.strip()
    deepphy_root = run_dir / "deepphy_eval"
    if requested_run_id:
        candidate = deepphy_root / requested_run_id
        if (candidate / "results.json").is_file() and _evaluation_root_is_usable(candidate):
            return candidate, requested_run_id
    if deepphy_root.is_dir():
        candidates = sorted(
            (path.parent for path in deepphy_root.glob("*/results.json") if path.is_file()),
            key=lambda item: (item.stat().st_mtime_ns, item.name),
            reverse=True,
        )
        for candidate in candidates:
            if _evaluation_root_is_usable(candidate):
                return candidate, candidate.name
        if candidates:
            return candidates[0], candidates[0].name
    return None, None


def compute_supervised_label_importance(run_dir: Path) -> dict[str, Any]:
    library_path = run_dir / "pattern_library.json"
    if not library_path.is_file():
        raise FileNotFoundError(f"Pattern library not found: {library_path}")
    try:
        library_payload = json.loads(library_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed pattern library: {library_path}") from exc
    detectors_payload = library_payload.get("detectors")
    if not isinstance(detectors_payload, list):
        raise ValueError(f"Pattern library detectors missing from {library_path}")

    labels: list[dict[str, Any]] = []
    counts_by_key: dict[str, dict[str, int]] = {}
    for detector in detectors_payload:
        if not isinstance(detector, dict):
            continue
        input_mode = str(detector.get("input_mode") or "trace").strip()
        if input_mode != "trace":
            continue
        detector_id = str(detector.get("detector_id") or "").strip()
        label = str(detector.get("label") or "").strip()
        if not detector_id or not label:
            continue
        key = _importance_key(input_mode=input_mode, detector_id=detector_id, label=label)
        labels.append(
            {
                "detector_key": key,
                "detector_id": detector_id,
                "label": label,
                "input_mode": input_mode,
            }
        )
        counts_by_key[key] = {
            "present_success": 0,
            "present_failure": 0,
            "absent_success": 0,
            "absent_failure": 0,
        }
    if not labels:
        raise ValueError(f"No supervised trace detectors found in {library_path}")

    evaluation_root, evaluation_run_id = resolve_supervised_evaluation_root(run_dir)
    if evaluation_root is None or evaluation_run_id is None:
        raise FileNotFoundError(f"No completed DeepPHY evaluation found under {run_dir / 'deepphy_eval'}")

    task_results = sorted(evaluation_root.glob("tasks/*/*/task_result.json"))
    if not task_results:
        raise FileNotFoundError(f"No task_result.json files found under {evaluation_root / 'tasks'}")

    task_count = 0
    success_task_count = 0
    for task_result_path in task_results:
        try:
            task_payload = json.loads(task_result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(task_payload, dict):
            continue
        task_success = bool(task_payload.get("success"))
        task_count += 1
        if task_success:
            success_task_count += 1
        present_keys = _present_label_keys_for_task(task_payload, task_result_path=task_result_path)
        for key, bucket in counts_by_key.items():
            if key in present_keys:
                bucket["present_success" if task_success else "present_failure"] += 1
            else:
                bucket["absent_success" if task_success else "absent_failure"] += 1
    if task_count <= 0:
        raise ValueError(f"No usable task results found under {evaluation_root}")

    summary_labels: list[dict[str, Any]] = []
    for entry in labels:
        key = entry["detector_key"]
        bucket = counts_by_key[key]
        a = bucket["present_success"]
        b = bucket["present_failure"]
        c = bucket["absent_success"]
        d = bucket["absent_failure"]
        present_total = a + b
        absent_total = c + d
        success_rate_when_present = (a / present_total) if present_total > 0 else None
        success_rate_when_absent = (c / absent_total) if absent_total > 0 else None
        phi = _phi_coefficient(a, b, c, d)
        summary_labels.append(
            {
                **entry,
                **bucket,
                "task_count": task_count,
                "presence_rate": present_total / task_count,
                "success_rate_when_present": success_rate_when_present,
                "success_rate_when_absent": success_rate_when_absent,
                "success_rate_delta": (
                    (success_rate_when_present - success_rate_when_absent)
                    if success_rate_when_present is not None and success_rate_when_absent is not None
                    else None
                ),
                "correlation_score": phi,
                "sampling_weight": 1.0 + max(0.0, phi) * 4.0,
            }
        )
    summary_labels.sort(
        key=lambda item: (
            float(item.get("correlation_score") or 0.0),
            float(item.get("success_rate_delta") or 0.0),
            str(item.get("label") or ""),
        ),
        reverse=True,
    )
    payload = {
        "schema_version": SUPERVISED_LABEL_IMPORTANCE_SCHEMA_VERSION,
        "computed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "library_path": str(library_path),
        "evaluation_run_id": evaluation_run_id,
        "evaluation_run_dir": str(evaluation_root),
        "task_count": task_count,
        "success_task_count": success_task_count,
        "failure_task_count": task_count - success_task_count,
        "labels": summary_labels,
    }
    output_path = supervised_label_importance_path(run_dir)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _present_label_keys_for_task(task_payload: dict[str, Any], *, task_result_path: Path) -> set[str]:
    present: set[str] = set()
    records = task_payload.get("records")
    if not isinstance(records, list):
        return present
    for record in records:
        if not isinstance(record, dict):
            continue
        trace_path_raw = record.get("trace_path")
        if not isinstance(trace_path_raw, str) or not trace_path_raw.strip():
            continue
        evidence_path = _resolve_evidence_path(
            trace_path_raw=trace_path_raw,
            record=record,
            task_dir=task_result_path.parent,
        )
        if evidence_path is None or not evidence_path.is_file():
            continue
        try:
            evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        annotations = evidence_payload.get("pattern_annotations")
        if not isinstance(annotations, dict):
            continue
        patterns = annotations.get("patterns")
        if not isinstance(patterns, list):
            continue
        for item in patterns:
            if not isinstance(item, dict):
                continue
            detector_id = str(item.get("detector_id") or "").strip()
            label = str(item.get("label") or "").strip()
            if not detector_id or not label:
                continue
            present.add(_importance_key(input_mode="trace", detector_id=detector_id, label=label))
    return present


def _resolve_evidence_path(
    *,
    trace_path_raw: str,
    record: dict[str, Any],
    task_dir: Path,
) -> Path | None:
    direct_candidates: list[Path] = []
    evidence_path_raw = record.get("evidence_path")
    if isinstance(evidence_path_raw, str) and evidence_path_raw.strip():
        direct_candidates.append(Path(evidence_path_raw))
    direct_candidates.append(Path(trace_path_raw).with_name("trace.evidence.json"))
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    record_index = record.get("index")
    if isinstance(record_index, int) and record_index > 0:
        candidate = task_dir / f"attempt_{record_index:03d}" / "trace.evidence.json"
        if candidate.is_file():
            return candidate
        candidate = task_dir / f"step_{record_index:03d}" / "trace.evidence.json"
        if candidate.is_file():
            return candidate

    trace_name = Path(trace_path_raw).parent.name
    if trace_name:
        candidate = task_dir / trace_name / "trace.evidence.json"
        if candidate.is_file():
            return candidate

    for attempt_dir in sorted(task_dir.glob("attempt_*")):
        candidate = attempt_dir / "trace.evidence.json"
        if candidate.is_file():
            return candidate
    for step_dir in sorted(task_dir.glob("step_*")):
        candidate = step_dir / "trace.evidence.json"
        if candidate.is_file():
            return candidate
    return None


def _evaluation_root_is_usable(eval_root: Path) -> bool:
    task_results = sorted(eval_root.glob("tasks/*/*/task_result.json"))
    if not task_results:
        return False
    for task_result_path in task_results[:20]:
        try:
            payload = json.loads(task_result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records = payload.get("records")
        if isinstance(records, list) and records:
            return True
    return any(eval_root.rglob("trace.evidence.json"))


def _importance_key(*, input_mode: str, detector_id: str, label: str) -> str:
    return f"{input_mode.strip()}::{detector_id.strip()}::{label.strip()}"


def _phi_coefficient(a: int, b: int, c: int, d: int) -> float:
    numerator = (a * d) - (b * c)
    denominator = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denominator <= 0:
        return 0.0
    return numerator / denominator
