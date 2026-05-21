from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pattern_server.env_catalog import list_environment_libraries
from pattern_server.legacy_jobs import MONOREPO_ROOT


SUPPORTED_LIBRARY_COLLECTION_ENVS = ("phyre", "iphyre", "kinetix", "pooltool")
SUPPORTED_LIBRARY_COLLECTION_KINDS = ("supervised", "unsupervised", "automatic")
UNSUPERVISED_LIBRARY_LIMIT = 10
AUTOMATIC_LIBRARY_LIMIT = 10
ENSEMBLE_LIBRARY_K = 10

_LABEL_DIR_PATTERN = re.compile(r"^label_(?P<index>\d{3})_(?P<label>.+)$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_COLLECTION_CACHE: dict[tuple[str, str], tuple[str, Path]] = {}
_COLLECTION_FINGERPRINT_VERSION = "v8-ensemble-kmeans-k10-unsupervised-trace-supervised-label-cap"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: datetime | None = None) -> str:
    return (value or _utc_now()).isoformat()


def _state_root(project_root: Path) -> Path:
    return project_root / "state" / "video_renders"


def _client_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_client_src_on_path() -> None:
    client_src = _client_project_root() / "src"
    client_src_str = str(client_src)
    if client_src.is_dir() and client_src_str not in __import__("sys").path:
        __import__("sys").path.insert(0, client_src_str)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalized_label_key(value: str) -> str:
    return _NON_ALNUM_RE.sub(" ", str(value or "").strip().lower()).strip()


def _slugify_label(value: str) -> str:
    slug = _NON_ALNUM_RE.sub("_", str(value or "").strip().lower()).strip("_")
    return slug or "label"


def _score_value(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    if parsed != parsed:
        return float("-inf")
    return parsed


def _candidate_rank_key(payload: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        _score_value(payload.get("train_score")),
        _score_value(payload.get("average_score")),
        _score_value(payload.get("val_score")),
        str(payload.get("uid") or ""),
    )


def _library_rank(root: Path) -> tuple[float, float, float, str]:
    payload = _load_json_object(root / "pattern_library.json")
    detectors = payload.get("detectors")
    if not isinstance(detectors, list):
        detectors = []
    train_scores: list[float] = []
    validation_scores: list[float] = []
    generic_scores: list[float] = []
    for detector in detectors:
        if not isinstance(detector, dict):
            continue
        metadata = detector.get("metadata")
        meta = metadata if isinstance(metadata, dict) else {}
        train = _score_value(meta.get("train_score"))
        val = _score_value(meta.get("val_score"))
        score = _score_value(meta.get("score"))
        if train != float("-inf"):
            train_scores.append(train)
        if val != float("-inf"):
            validation_scores.append(val)
        if score != float("-inf"):
            generic_scores.append(score)
    train_mean = sum(train_scores) / len(train_scores) if train_scores else float("-inf")
    val_mean = sum(validation_scores) / len(validation_scores) if validation_scores else float("-inf")
    score_mean = sum(generic_scores) / len(generic_scores) if generic_scores else float("-inf")
    return (train_mean, score_mean, val_mean, str(root))


def _collection_cache_root(project_root: Path, env_id: str, kind: str) -> Path:
    return _state_root(project_root) / "canonical_libraries" / env_id / kind


def _collection_cache_paths(project_root: Path, env_id: str, kind: str) -> tuple[Path, Path]:
    root = _collection_cache_root(project_root, env_id, kind)
    return root / "pattern_library.json", root / "meta.json"


def _load_supervised_labels(project_root: Path, env_id: str) -> tuple[list[str], Path]:
    candidates = [project_root / "configs" / env_id / "labels.txt"]
    for path in candidates:
        if not path.is_file():
            continue
        labels = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if labels:
            return labels, path
    raise FileNotFoundError(f"Missing supervised label list for {env_id}")


def _supervised_label_limit(project_root: Path, env_id: str) -> int:
    labels, _labels_path = _load_supervised_labels(project_root, env_id)
    return max(1, len(labels))


def _library_candidates(project_root: Path, env_id: str, kind: str) -> list[Any]:
    candidates = [
        item
        for item in list_environment_libraries(project_root, env_id)
        if str(item.workflow_kind or "").strip().lower() == kind
        and ".bak." not in item.run_id
        and ".broken-" not in item.run_id
        and not str(item.run_id).startswith("video-render-best-supervised-")
        and not str(item.run_id).startswith("canonical-library-")
    ]
    if not candidates:
        raise FileNotFoundError(f"No {kind} library found for {env_id}")
    deduped: dict[str, Any] = {}
    for item in candidates:
        deduped.setdefault(str(item.root.resolve()), item)
    return sorted(deduped.values(), key=lambda item: _library_rank(item.root), reverse=True)


def _parse_label_dir(path: Path) -> tuple[int | None, str]:
    match = _LABEL_DIR_PATTERN.match(path.name)
    if not match:
        return None, path.name
    index_raw = str(match.group("index") or "").strip()
    label_raw = str(match.group("label") or "").strip().replace("_", " ")
    try:
        index = int(index_raw)
    except ValueError:
        index = None
    return index, label_raw or path.name


def _find_label_dir(run_dir: Path, *, label_index: int, label: str) -> Path | None:
    expected = run_dir / f"label_{label_index:03d}_{_slugify_label(label)}"
    if expected.is_dir():
        return expected
    target_key = _normalized_label_key(label)
    fallback: Path | None = None
    for candidate in sorted(path for path in run_dir.glob("label_*") if path.is_dir()):
        parsed_index, parsed_label = _parse_label_dir(candidate)
        if parsed_index == label_index and target_key == _normalized_label_key(parsed_label):
            return candidate
        if fallback is None and target_key == _normalized_label_key(parsed_label):
            fallback = candidate
    return fallback


def _load_candidate_code(candidate_dir: Path, payload: dict[str, Any]) -> str | None:
    source_path = str(payload.get("source_path") or "").strip()
    if source_path:
        path = Path(source_path)
        if not path.is_file():
            path = candidate_dir / source_path
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8").strip()
            except OSError:
                text = ""
            if text:
                return text
    for path in sorted(candidate_dir.glob("*.py")):
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8").strip()
            except OSError:
                text = ""
            if text:
                return text
    return None


def _label_candidates_for_run(run_dir: Path, *, label_index: int, label: str) -> list[dict[str, Any]]:
    label_dir = _find_label_dir(run_dir, label_index=label_index, label=label)
    if label_dir is None:
        return []

    rows: list[dict[str, Any]] = []
    for candidate_dir in sorted(path for path in label_dir.glob("candidate_*") if path.is_dir()):
        payload = _load_json_object(candidate_dir / "candidate.json")
        if not payload:
            continue
        source_code = _load_candidate_code(candidate_dir, payload)
        if not source_code:
            continue
        rows.append(
            {
            "uid": str(payload.get("uid") or payload.get("candidate_uid") or candidate_dir.name).strip(),
            "train_score": payload.get("train_score"),
            "average_score": payload.get("average_score"),
            "val_score": payload.get("val_score"),
            "source_code": source_code,
            "input_mode": str(payload.get("input_mode") or "trace").strip() or "trace",
            "metadata": dict(payload.get("metadata") or {}),
            "source_run_dir": str(run_dir),
            "source_label_dir": str(label_dir),
            "label": label,
            "label_index": label_index,
            }
        )

    if rows:
        return rows

    label_result = _load_json_object(label_dir / "label_result.json")
    detector_payload = label_result.get("detector")
    if not isinstance(detector_payload, dict):
        return []
    source_code = str(detector_payload.get("source_code") or "").strip()
    if not source_code:
        return []
    return [
        {
            "uid": str(label_result.get("best_candidate_uid") or f"{run_dir.name}:{label}").strip(),
            "train_score": label_result.get("best_train_score"),
            "average_score": label_result.get("best_score"),
            "val_score": label_result.get("best_validation_score"),
            "source_code": source_code,
            "input_mode": str(detector_payload.get("input_mode") or "trace").strip() or "trace",
            "metadata": {"synthetic_from_label_result": True},
            "source_run_dir": str(run_dir),
            "source_label_dir": str(label_dir),
            "label": label,
            "label_index": label_index,
        }
    ]


def _all_label_candidates_for_run(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label_dir in sorted(path for path in run_dir.glob("label_*") if path.is_dir()):
        parsed = _parse_label_dir(label_dir)
        if parsed is None:
            continue
        label_index, label = parsed
        if label_index is None:
            continue
        rows.extend(_label_candidates_for_run(run_dir, label_index=label_index, label=label))
    return rows


def _code_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _train_score_value(item: dict[str, Any]) -> float:
    return _score_value(item.get("train_score"))


def _mean_candidate_train_score(rows: list[dict[str, Any]]) -> float:
    values = [
        _train_score_value(item)
        for item in rows
        if _train_score_value(item) != float("-inf")
    ]
    if not values:
        return float("-inf")
    return sum(values) / float(len(values))


def _candidate_base_reward(item: dict[str, Any]) -> float:
    for key in ("train_score", "average_score", "val_score", "score"):
        value = _score_value(item.get(key))
        if value != float("-inf"):
            return max(0.0, min(1.0, value))
    metadata = item.get("metadata")
    meta = metadata if isinstance(metadata, dict) else {}
    for key in ("base_reward", "train_score", "score", "val_score", "initial_weight"):
        value = _score_value(meta.get(key))
        if value != float("-inf"):
            return max(0.0, min(1.0, value))
    return 1.0


def _vector_sq_distance(left: list[float], right: list[float]) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))


def _average_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    if width <= 0:
        return []
    return [
        sum(float(vector[index]) for vector in vectors) / float(len(vectors))
        for index in range(width)
    ]


def _cluster_assignments(vectors: list[list[float]], *, cluster_count: int, seed_text: str) -> list[int]:
    total = len(vectors)
    if total <= 0:
        return []
    effective_clusters = max(1, min(int(cluster_count), total))
    if effective_clusters == 1:
        return [0] * total
    rng = __import__("random").Random(seed_text)
    center_indices = [rng.randrange(total)]
    while len(center_indices) < effective_clusters:
        distances: list[float] = []
        for vector in vectors:
            nearest = min(_vector_sq_distance(vector, vectors[index]) for index in center_indices)
            distances.append(max(0.0, nearest))
        total_distance = sum(distances)
        if total_distance <= 0.0:
            for index in range(total):
                if index not in center_indices:
                    center_indices.append(index)
                    break
            continue
        draw = rng.random() * total_distance
        cumulative = 0.0
        chosen_index = 0
        for index, distance in enumerate(distances):
            cumulative += distance
            if cumulative >= draw:
                chosen_index = index
                break
        if chosen_index in center_indices:
            for index in range(total):
                if index not in center_indices:
                    chosen_index = index
                    break
        center_indices.append(chosen_index)
    centers = [list(vectors[index]) for index in center_indices]
    assignments = [0] * total
    for _ in range(24):
        changed = False
        for index, vector in enumerate(vectors):
            best_cluster = min(
                range(len(centers)),
                key=lambda cluster_index: _vector_sq_distance(vector, centers[cluster_index]),
            )
            if assignments[index] != best_cluster:
                assignments[index] = best_cluster
                changed = True
        next_centers: list[list[float]] = []
        for cluster_index in range(len(centers)):
            members = [vectors[index] for index, assignment in enumerate(assignments) if assignment == cluster_index]
            if members:
                next_centers.append(_average_vector(members))
            else:
                next_centers.append(list(centers[cluster_index]))
        centers = next_centers
        if not changed:
            break
    return assignments


def _select_cluster_representatives(
    *,
    candidates: list[dict[str, Any]],
    embeddings_by_hash: dict[str, list[float]],
    cluster_count: int,
    seed_text: str,
) -> list[dict[str, Any]]:
    indexed_candidates: list[tuple[dict[str, Any], list[float], str]] = []
    seen_hashes: set[str] = set()
    for candidate in candidates:
        source_code = str(candidate.get("source_code") or "").strip()
        if not source_code:
            continue
        code_hash = _code_hash(source_code)
        if code_hash in seen_hashes:
            continue
        vector = embeddings_by_hash.get(code_hash)
        if not vector:
            continue
        seen_hashes.add(code_hash)
        indexed_candidates.append((candidate, vector, code_hash))
    if not indexed_candidates:
        return []
    assignments = _cluster_assignments(
        [vector for _candidate, vector, _code_hash_value in indexed_candidates],
        cluster_count=cluster_count,
        seed_text=seed_text,
    )
    by_cluster: dict[int, list[dict[str, Any]]] = {}
    for (candidate, _vector, code_hash), cluster_index in zip(indexed_candidates, assignments):
        row = dict(candidate)
        row["code_hash"] = code_hash
        row["cluster_index"] = int(cluster_index)
        by_cluster.setdefault(int(cluster_index), []).append(row)
    chosen: list[dict[str, Any]] = []
    for cluster_index in sorted(by_cluster):
        rows = sorted(by_cluster[cluster_index], key=_candidate_rank_key, reverse=True)
        if rows:
            chosen.append(rows[0])
    return chosen


def _collection_fingerprint(
    *,
    env_id: str,
    kind: str,
    labels: list[str] | None,
    labels_path: Path | None,
    runs: list[Any],
) -> str:
    parts = [f"env={env_id}", f"kind={kind}", f"version={_COLLECTION_FINGERPRINT_VERSION}"]
    if labels is not None:
        labels_digest = hashlib.sha1("\n".join(labels).encode("utf-8")).hexdigest()
        parts.append(f"labels={labels_digest}")
        parts.append(f"labels_mtime={labels_path.stat().st_mtime_ns if labels_path and labels_path.exists() else 0}")
    for entry in runs:
        parts.append(f"{entry.run_id}|{entry.updated_at.isoformat()}|{entry.detector_count}")
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()


def _detector_score(item: dict[str, Any]) -> float:
    metadata = item.get("metadata")
    meta = metadata if isinstance(metadata, dict) else {}
    for key in ("train_score", "score", "val_score"):
        value = _score_value(meta.get(key))
        if value != float("-inf"):
            return value
    return float("-inf")


def _resolve_supervised_collection(project_root: Path, env_id: str) -> Path:
    labels, labels_path = _load_supervised_labels(project_root, env_id)
    runs = _library_candidates(project_root, env_id, "supervised")
    fingerprint = _collection_fingerprint(
        env_id=env_id,
        kind="supervised",
        labels=labels,
        labels_path=labels_path,
        runs=runs,
    )
    cache_key = (env_id, "supervised")
    cached = _COLLECTION_CACHE.get(cache_key)
    if cached and cached[0] == fingerprint and cached[1].is_file():
        return cached[1]

    library_path, meta_path = _collection_cache_paths(project_root, env_id, "supervised")
    if library_path.is_file() and meta_path.is_file():
        meta = _load_json_object(meta_path)
        if str(meta.get("fingerprint") or "").strip() == fingerprint:
            _COLLECTION_CACHE[cache_key] = (fingerprint, library_path)
            return library_path

    _ensure_client_src_on_path()
    from ensemble_library_training.code_embeddings import fetch_openai_embeddings, normalize_embedding  # type: ignore
    from pattern_library_schema import (  # type: ignore
        DetectorDefinition,
        DetectorSlotDefinition,
        LabelDetectorEnsemble,
        PatternLibrary,
    )

    label_candidate_pool: dict[tuple[int, str], list[dict[str, Any]]] = {}
    code_inputs_by_hash: dict[str, str] = {}
    for label_index, label in enumerate(labels, start=1):
        for entry in runs:
            for candidate in _label_candidates_for_run(entry.root, label_index=label_index, label=label):
                row = {**candidate, "source_run_id": entry.run_id}
                label_candidate_pool.setdefault((label_index, label), []).append(row)
                source_code = str(row.get("source_code") or "").strip()
                if source_code:
                    code_inputs_by_hash.setdefault(_code_hash(source_code), source_code)

    code_hashes = sorted(code_inputs_by_hash)
    vectors = fetch_openai_embeddings(
        inputs=[code_inputs_by_hash[item] for item in code_hashes],
        cache_keys=code_hashes,
        project_root=project_root,
        batch_size=64,
    )
    if code_hashes and (vectors is None or len(vectors) != len(code_hashes)):
        raise RuntimeError(f"Unable to compute code embeddings for canonical supervised library {env_id}")
    embeddings_by_hash = {
        code_hashes[index]: normalize_embedding(vectors[index])
        for index in range(len(code_hashes))
    } if vectors is not None else {}

    detectors: list[Any] = []
    label_ensembles: list[Any] = []
    for label_index, label in enumerate(labels, start=1):
        candidate_pool = label_candidate_pool.get((label_index, label), [])
        selected = _select_cluster_representatives(
            candidates=candidate_pool,
            embeddings_by_hash=embeddings_by_hash,
            cluster_count=ENSEMBLE_LIBRARY_K,
            seed_text=f"{env_id}:supervised:{label_index}:{label}",
        )
        if not selected:
            continue
        members: list[Any] = []
        for slot_index, candidate in enumerate(selected):
            base_reward = _candidate_base_reward(candidate)
            metadata = dict(candidate.get("metadata") or {})
            metadata.update(
                {
                    "candidate_uid": candidate.get("uid"),
                    "score": candidate.get("average_score"),
                    "train_score": candidate.get("train_score"),
                    "val_score": candidate.get("val_score"),
                    "source_run_id": candidate.get("source_run_id"),
                    "source_run_dir": candidate.get("source_run_dir"),
                    "source_label_dir": candidate.get("source_label_dir"),
                    "selection_strategy": "kmeans_cluster_best_per_label",
                    "slot_index": slot_index,
                    "cluster_index": candidate.get("cluster_index"),
                    "base_reward": base_reward,
                    "initial_weight": 1.0,
                }
            )
            metadata = {key: value for key, value in metadata.items() if value is not None}
            detector = DetectorDefinition(
                detector_id=f"primitive-{label_index:03d}-s{slot_index:02d}",
                label=label,
                environment_ids=[env_id],
                source_code=str(candidate.get("source_code") or ""),
                input_mode=(
                    str(candidate.get("input_mode") or "trace").strip()
                    if str(candidate.get("input_mode") or "trace").strip() in {"trace", "supervised_evidence"}
                    else "trace"
                ),
                metadata=metadata,
            )
            detectors.append(detector)
            members.append(
                DetectorSlotDefinition(
                    slot_index=slot_index,
                    detector=detector,
                    train_score=_train_score_value(candidate) if _train_score_value(candidate) != float("-inf") else None,
                    cluster_index=int(candidate.get("cluster_index") or 0),
                    initial_weight=1.0,
                    metadata={},
                )
            )
        label_ensembles.append(
            LabelDetectorEnsemble(
                label=label,
                input_mode="trace",
                members=members,
                metadata={"label_index": label_index},
            )
        )

    if not label_ensembles:
        raise FileNotFoundError(f"No supervised label candidates available for {env_id}")

    top_library_payload = _load_json_object(runs[0].library_path)
    library = PatternLibrary(
        library_id=f"canonical-library-supervised-{env_id}",
        environment_ids=[env_id],
        primitive_source=str(labels_path),
        scorer_config=dict(top_library_payload.get("scorer_config") or {}),
        detectors=detectors,
        label_ensembles=label_ensembles,
        metadata={
            "source": "canonical_library",
            "selection": "kmeans_cluster_best_per_label",
            "annotation_strategy": "ensemble_consensus",
            "workflow_kind": "supervised",
            "source_run_dirs": [str(entry.root) for entry in runs],
            "source_run_ids": [str(entry.run_id) for entry in runs],
            "detector_count": len(detectors),
            "label_count": len(labels),
            "ensemble_k": ENSEMBLE_LIBRARY_K,
            "fingerprint": fingerprint,
            "generated_at": _isoformat(),
        },
    )
    library_path.parent.mkdir(parents=True, exist_ok=True)
    library_path.write_text(library.model_dump_json(indent=2) + "\n", encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "generated_at": _isoformat(),
                "library_path": str(library_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _COLLECTION_CACHE[cache_key] = (fingerprint, library_path)
    return library_path


def _resolve_ranked_collection(project_root: Path, env_id: str, kind: str, *, limit: int) -> Path:
    runs = _library_candidates(project_root, env_id, kind)
    fingerprint_labels: list[str] | None = None
    fingerprint_labels_path: Path | None = None
    supervised_label_keys: set[str] = set()
    effective_limit = max(1, int(limit))
    if kind == "automatic":
        try:
            labels, labels_path = _load_supervised_labels(project_root, env_id)
            fingerprint_labels = labels
            fingerprint_labels_path = labels_path
            supervised_label_keys = {_normalized_label_key(item) for item in labels if _normalized_label_key(item)}
        except FileNotFoundError:
            supervised_label_keys = set()
    elif kind == "unsupervised":
        labels, labels_path = _load_supervised_labels(project_root, env_id)
        fingerprint_labels = labels
        fingerprint_labels_path = labels_path
        effective_limit = _supervised_label_limit(project_root, env_id)
    fingerprint = _collection_fingerprint(
        env_id=env_id,
        kind=kind,
        labels=fingerprint_labels,
        labels_path=fingerprint_labels_path,
        runs=runs,
    )
    cache_key = (env_id, kind)
    cached = _COLLECTION_CACHE.get(cache_key)
    if cached and cached[0] == fingerprint and cached[1].is_file():
        return cached[1]

    library_path, meta_path = _collection_cache_paths(project_root, env_id, kind)
    if library_path.is_file() and meta_path.is_file():
        meta = _load_json_object(meta_path)
        if str(meta.get("fingerprint") or "").strip() == fingerprint:
            _COLLECTION_CACHE[cache_key] = (fingerprint, library_path)
            return library_path

    _ensure_client_src_on_path()
    from ensemble_library_training.code_embeddings import fetch_openai_embeddings, normalize_embedding  # type: ignore
    from pattern_library_schema import (  # type: ignore
        DetectorDefinition,
        DetectorSlotDefinition,
        LabelDetectorEnsemble,
        PatternLibrary,
    )

    detector_rows: list[dict[str, Any]] = []
    for entry in runs:
        payload = _load_json_object(entry.library_path)
        raw_detectors = payload.get("detectors")
        detectors = raw_detectors if isinstance(raw_detectors, list) else []
        for index, detector in enumerate(detectors):
            if not isinstance(detector, dict):
                continue
            detector_rows.append(
                {
                    "run_id": entry.run_id,
                    "run_root": str(entry.root),
                    "library_path": str(entry.library_path),
                    "index": index,
                    "detector": detector,
                }
            )
    if kind != "unsupervised" and not detector_rows:
        raise FileNotFoundError(f"No detector candidates available for {env_id} {kind}")

    top_payload = _load_json_object(runs[0].library_path)
    detectors_out: list[Any] = []
    label_ensembles: list[Any] = []
    if kind == "unsupervised":
        code_inputs_by_hash: dict[str, str] = {}
        candidate_rows_by_label: dict[str, list[dict[str, Any]]] = {}
        for entry in runs:
            for candidate in _all_label_candidates_for_run(entry.root):
                label = str(candidate.get("label") or "").strip()
                label_key = _normalized_label_key(label)
                if not label_key or label_key in supervised_label_keys:
                    continue
                normalized_row = {
                    **candidate,
                    "source_run_id": entry.run_id,
                    "source_library_path": str(entry.library_path),
                }
                candidate_rows_by_label.setdefault(label_key, []).append(normalized_row)
                source_code = str(normalized_row.get("source_code") or "").strip()
                if source_code:
                    code_inputs_by_hash.setdefault(_code_hash(source_code), source_code)
        ranked_label_keys = sorted(
            candidate_rows_by_label,
            key=lambda label_key: (
                _mean_candidate_train_score(candidate_rows_by_label[label_key]),
                label_key,
            ),
            reverse=True,
        )
        ranked_label_keys = ranked_label_keys[:effective_limit]
        code_hashes = sorted(code_inputs_by_hash)
        vectors = fetch_openai_embeddings(
            inputs=[code_inputs_by_hash[item] for item in code_hashes],
            cache_keys=code_hashes,
            project_root=project_root,
            batch_size=64,
        )
        if code_hashes and (vectors is None or len(vectors) != len(code_hashes)):
            raise RuntimeError(f"Unable to compute code embeddings for canonical {kind} library {env_id}")
        embeddings_by_hash = {
            code_hashes[index]: normalize_embedding(vectors[index])
            for index in range(len(code_hashes))
        } if vectors is not None else {}
        for label_index, label_key in enumerate(ranked_label_keys, start=1):
            candidate_pool = candidate_rows_by_label[label_key]
            label = str(candidate_pool[0].get("label") or label_key).strip() or label_key
            selected = _select_cluster_representatives(
                candidates=candidate_pool,
                embeddings_by_hash=embeddings_by_hash,
                cluster_count=ENSEMBLE_LIBRARY_K,
                seed_text=f"{env_id}:{kind}:{label}",
            )
            if not selected:
                continue
            members: list[Any] = []
            for slot_index, candidate in enumerate(selected):
                base_reward = _candidate_base_reward(candidate)
                metadata = dict(candidate.get("metadata") or {})
                metadata.update(
                    {
                        "candidate_uid": candidate.get("uid"),
                        "score": candidate.get("average_score"),
                        "train_score": candidate.get("train_score"),
                        "val_score": candidate.get("val_score"),
                        "source_run_id": candidate.get("source_run_id"),
                        "source_run_dir": candidate.get("source_run_dir"),
                        "source_library_path": candidate.get("source_library_path"),
                        "selection_strategy": f"kmeans_cluster_best_per_label_{kind}",
                        "slot_index": slot_index,
                        "cluster_index": candidate.get("cluster_index"),
                        "base_reward": base_reward,
                        "initial_weight": 1.0,
                    }
                )
                metadata = {key: value for key, value in metadata.items() if value is not None}
                detector = DetectorDefinition(
                    detector_id=str(candidate.get("uid") or f"{kind}-{label_index:03d}-s{slot_index:02d}").strip() or f"{kind}-{label_index:03d}-s{slot_index:02d}",
                    label=label,
                    environment_ids=[env_id],
                    source_code=str(candidate.get("source_code") or ""),
                    input_mode="trace",
                    metadata=metadata,
                )
                detectors_out.append(detector)
                members.append(
                    DetectorSlotDefinition(
                        slot_index=slot_index,
                        detector=detector,
                        train_score=_train_score_value(candidate) if _train_score_value(candidate) != float("-inf") else None,
                        cluster_index=int(candidate.get("cluster_index") or 0),
                        initial_weight=1.0,
                        metadata={},
                    )
                )
            label_ensembles.append(
                LabelDetectorEnsemble(
                    label=label,
                    input_mode="trace",
                    members=members,
                    metadata={},
                )
            )
    else:
        detector_rows.sort(
            key=lambda row: (
                _detector_score(dict(row.get("detector") or {})),
                str((dict(row.get("detector") or {})).get("detector_id") or ""),
                str((dict(row.get("detector") or {})).get("label") or ""),
                str(row.get("run_id") or ""),
            ),
            reverse=True,
        )
        chosen: list[dict[str, Any]] = []
        seen_label_keys: set[str] = set()
        for row in detector_rows:
            detector = dict(row.get("detector") or {})
            label = str(detector.get("label") or "").strip()
            label_key = _normalized_label_key(label)
            if kind == "automatic" and label_key and label_key in supervised_label_keys:
                continue
            if label_key and label_key in seen_label_keys:
                continue
            if label_key:
                seen_label_keys.add(label_key)
            chosen.append(row)
            if len(chosen) >= limit:
                break
        for index, row in enumerate(chosen, start=1):
            detector = dict(row.get("detector") or {})
            metadata = dict(detector.get("metadata") or {})
            metadata.update(
                {
                    "source_run_id": row.get("run_id"),
                    "source_run_dir": row.get("run_root"),
                    "source_library_path": row.get("library_path"),
                    "selection_rank": index,
                    "selection_strategy": f"top_scoring_{kind}_detectors",
                }
            )
            metadata = {key: value for key, value in metadata.items() if value is not None}
            detectors_out.append(
                DetectorDefinition(
                    detector_id=str(detector.get("detector_id") or f"{kind}-{index:03d}").strip() or f"{kind}-{index:03d}",
                    label=str(detector.get("label") or "").strip() or f"{kind} {index}",
                    environment_ids=[env_id],
                    source_code=str(detector.get("source_code") or ""),
                    input_mode=(
                        str(detector.get("input_mode") or "trace").strip()
                        if str(detector.get("input_mode") or "trace").strip() in {"trace", "supervised_evidence"}
                        else "trace"
                    ),
                    config=dict(detector.get("config") or {}),
                    entrypoint=str(detector.get("entrypoint") or "").strip() or None,
                    description=str(detector.get("description") or "").strip() or None,
                    metadata=metadata,
                )
            )

    if not detectors_out:
        raise FileNotFoundError(f"No detector candidates available for {env_id} {kind}")

    library = PatternLibrary(
        library_id=f"canonical-library-{kind}-{env_id}",
        environment_ids=[env_id],
        primitive_source="",
        scorer_config=dict(top_payload.get("scorer_config") or {}),
        detectors=detectors_out,
        label_ensembles=label_ensembles,
        metadata={
            "source": "canonical_library",
            "selection": (
                f"kmeans_cluster_best_per_label_{kind}"
                if kind == "unsupervised"
                else f"top_scoring_{kind}_detectors"
            ),
            "annotation_strategy": "ensemble_consensus" if kind == "unsupervised" else "",
            "workflow_kind": kind,
            "source_run_dirs": [str(entry.root) for entry in runs],
            "source_run_ids": [str(entry.run_id) for entry in runs],
            "detector_count": len(detectors_out),
            "fingerprint": fingerprint,
            "generated_at": _isoformat(),
            "limit": effective_limit,
            "ensemble_k": ENSEMBLE_LIBRARY_K if kind == "unsupervised" else None,
        },
    )
    library_path.parent.mkdir(parents=True, exist_ok=True)
    library_path.write_text(library.model_dump_json(indent=2) + "\n", encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "generated_at": _isoformat(),
                "library_path": str(library_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _COLLECTION_CACHE[cache_key] = (fingerprint, library_path)
    return library_path


def resolve_canonical_library(project_root: Path, env_id: str, kind: str) -> Path:
    env_key = str(env_id or "").strip().lower()
    kind_key = str(kind or "").strip().lower()
    if env_key not in SUPPORTED_LIBRARY_COLLECTION_ENVS:
        raise ValueError(f"Unknown environment {env_key}")
    if kind_key not in SUPPORTED_LIBRARY_COLLECTION_KINDS:
        raise ValueError(f"Unknown library kind {kind_key}")
    if kind_key == "supervised":
        return _resolve_supervised_collection(project_root, env_key)
    if kind_key == "unsupervised":
        return _resolve_ranked_collection(project_root, env_key, kind_key, limit=UNSUPERVISED_LIBRARY_LIMIT)
    return _resolve_ranked_collection(project_root, env_key, kind_key, limit=AUTOMATIC_LIBRARY_LIMIT)


def recalculate_canonical_libraries(project_root: Path) -> dict[str, Any]:
    cache_root = _state_root(project_root) / "canonical_libraries"
    if cache_root.exists():
        shutil.rmtree(cache_root, ignore_errors=True)
    _COLLECTION_CACHE.clear()
    rebuilt: dict[str, str] = {}
    errors: dict[str, str] = {}
    for env_id in SUPPORTED_LIBRARY_COLLECTION_ENVS:
        for kind in SUPPORTED_LIBRARY_COLLECTION_KINDS:
            key = f"{env_id}:{kind}"
            try:
                rebuilt[key] = str(resolve_canonical_library(project_root, env_id, kind))
            except Exception as exc:
                errors[key] = str(exc)
    return {"rebuilt": rebuilt, "errors": errors}


def _load_canonical_library_summary_from_path(
    *,
    env_key: str,
    kind_key: str,
    library_path: Path,
) -> dict[str, Any]:
    payload = _load_json_object(library_path)
    metadata = payload.get("metadata")
    metadata_obj = metadata if isinstance(metadata, dict) else {}
    raw_detectors = payload.get("detectors")
    detectors = raw_detectors if isinstance(raw_detectors, list) else []
    rows: list[dict[str, Any]] = []
    for item in detectors:
        if not isinstance(item, dict):
            continue
        row_meta = item.get("metadata")
        row_meta_obj = row_meta if isinstance(row_meta, dict) else {}
        score = row_meta_obj.get("train_score")
        if not isinstance(score, (int, float)):
            score = row_meta_obj.get("score")
        if not isinstance(score, (int, float)):
            score = row_meta_obj.get("val_score")
        rows.append(
            {
                "detector_id": str(item.get("detector_id") or "").strip(),
                "label": str(item.get("label") or "").strip(),
                "score": float(score) if isinstance(score, (int, float)) else None,
                "code": str(item.get("source_code") or ""),
                "source_run_id": str(row_meta_obj.get("source_run_id") or "").strip(),
                "source_run_dir": str(row_meta_obj.get("source_run_dir") or "").strip(),
                "source_library_path": str(row_meta_obj.get("source_library_path") or "").strip(),
                "selection_rank": row_meta_obj.get("selection_rank"),
            }
        )
    rows.sort(
        key=lambda row: (
            _score_value(row.get("score")),
            -int(row.get("selection_rank") or 0) if isinstance(row.get("selection_rank"), int) else 0,
            str(row.get("detector_id") or ""),
            str(row.get("label") or ""),
        ),
        reverse=True,
    )
    source_run_ids = metadata_obj.get("source_run_ids")
    source_runs = [str(item).strip() for item in source_run_ids if str(item).strip()] if isinstance(source_run_ids, list) else []
    return {
        "env_id": env_key,
        "kind": kind_key,
        "library_path": str(library_path),
        "rows": rows,
        "detector_count": len(rows),
        "source_run_count": len(source_runs),
        "error": "",
    }


def load_cached_canonical_library_summary(project_root: Path, env_id: str, kind: str) -> dict[str, Any]:
    env_key = str(env_id or "").strip().lower()
    kind_key = str(kind or "").strip().lower()
    if env_key not in SUPPORTED_LIBRARY_COLLECTION_ENVS:
        return {"env_id": env_key, "kind": kind_key, "library_path": None, "rows": [], "error": f"Unknown environment {env_key}"}
    if kind_key not in SUPPORTED_LIBRARY_COLLECTION_KINDS:
        return {"env_id": env_key, "kind": kind_key, "library_path": None, "rows": [], "error": f"Unknown library kind {kind_key}"}
    library_path, _meta_path = _collection_cache_paths(project_root, env_key, kind_key)
    if not library_path.is_file():
        return {
            "env_id": env_key,
            "kind": kind_key,
            "library_path": None,
            "rows": [],
            "error": "No cached canonical library yet. Use Recalculate libraries.",
        }
    return _load_canonical_library_summary_from_path(
        env_key=env_key,
        kind_key=kind_key,
        library_path=library_path,
    )


def load_canonical_library_summary(project_root: Path, env_id: str, kind: str) -> dict[str, Any]:
    env_key = str(env_id or "").strip().lower()
    kind_key = str(kind or "").strip().lower()
    if env_key not in SUPPORTED_LIBRARY_COLLECTION_ENVS:
        return {"env_id": env_key, "kind": kind_key, "library_path": None, "rows": [], "error": f"Unknown environment {env_key}"}
    if kind_key not in SUPPORTED_LIBRARY_COLLECTION_KINDS:
        return {"env_id": env_key, "kind": kind_key, "library_path": None, "rows": [], "error": f"Unknown library kind {kind_key}"}
    try:
        library_path = resolve_canonical_library(project_root, env_key, kind_key)
    except Exception as exc:
        return {"env_id": env_key, "kind": kind_key, "library_path": None, "rows": [], "error": str(exc)}
    return _load_canonical_library_summary_from_path(
        env_key=env_key,
        kind_key=kind_key,
        library_path=library_path,
    )


def load_global_best_supervised_library_summary(project_root: Path, env_id: str) -> dict[str, Any]:
    return load_canonical_library_summary(project_root, env_id, "supervised")


__all__ = [
    "SUPPORTED_LIBRARY_COLLECTION_ENVS",
    "SUPPORTED_LIBRARY_COLLECTION_KINDS",
    "UNSUPERVISED_LIBRARY_LIMIT",
    "AUTOMATIC_LIBRARY_LIMIT",
    "load_canonical_library_summary",
    "load_cached_canonical_library_summary",
    "load_global_best_supervised_library_summary",
    "recalculate_canonical_libraries",
    "resolve_canonical_library",
]
