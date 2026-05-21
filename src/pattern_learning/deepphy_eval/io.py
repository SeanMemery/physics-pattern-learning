from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from deepphy_eval.models import (
    DEFAULT_EVIDENCE_RELPATH,
    EvaluationLiveRequest,
)
from pattern_library_evidence import load_pattern_evidence
from schema import CanonicalTrace


def load_trace(trace_path: Path) -> CanonicalTrace:
    with trace_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return CanonicalTrace.model_validate(payload)


def write_trace(trace_path: Path, trace: CanonicalTrace) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as handle:
        json.dump(trace.model_dump(mode="json"), handle, indent=2)


def load_live_request(path: Path) -> EvaluationLiveRequest:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return EvaluationLiveRequest.model_validate(payload)


def read_json_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_json_bytes(path: Path) -> bytes:
    return path.read_bytes()


def json_sha256_prefix(payload: bytes, *, size: int = 12) -> str:
    return sha256(payload).hexdigest()[:size]


def normalized_json_sha256_prefix(payload: dict[str, Any], *, size: int = 12) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return json_sha256_prefix(normalized, size=size)


def iter_trace_paths(trace_root: Path) -> list[Path]:
    return sorted(trace_root.rglob("trace.json"), key=lambda path: path.as_posix())


def inspect_evidence_sidecar(
    sample_dir: Path,
    *,
    evidence_relpath: str | None = None,
) -> tuple[str, str]:
    relpath = evidence_relpath or DEFAULT_EVIDENCE_RELPATH
    evidence_path = sample_dir / relpath
    if not evidence_path.is_file():
        return relpath, "missing"
    try:
        with evidence_path.open("r", encoding="utf-8") as handle:
            json.load(handle)
    except Exception:
        return relpath, "invalid"
    return relpath, "present"


def load_pattern_evidence_sidecar(
    sample_dir: Path,
    *,
    evidence_relpath: str | None = None,
):
    relpath = evidence_relpath or DEFAULT_EVIDENCE_RELPATH
    return load_pattern_evidence(sample_dir, evidence_relpath=relpath)
