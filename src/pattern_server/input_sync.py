from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from .legacy_jobs import MONOREPO_ROOT
from .models import ClientRecord, JobRecord


TEXT_REWRITE_SUFFIXES = {".json", ".jsonl", ".yaml", ".yml", ".txt"}


def build_job_input_plan(
    job: JobRecord,
    *,
    project_root: Path,
    assigned_client: ClientRecord | None = None,
) -> dict[str, Any]:
    bundle_items: list[dict[str, str]] = []
    seen_bundle_entries: set[tuple[str, str]] = set()
    dynamic_rewrite_prefixes: list[dict[str, str]] = []
    seen_rewrite_prefixes: set[tuple[str, str]] = set()
    preloaded_dataset_roots = _client_preloaded_dataset_roots(assigned_client, project_root=project_root)
    refresh_marker_source = project_root / "configs" / "generated" / "pattern_learning_server_policy.yaml"
    refresh_revision = _job_refresh_revision(job)
    metadata = dict(job.spec.metadata or {})
    request_payload = metadata.get("request") if isinstance(metadata, dict) else None
    request_dict = request_payload if isinstance(request_payload, dict) else {}
    resume_requested = bool(request_dict.get("resume")) or bool(metadata.get("resume_argv"))

    def add_rewrite_prefix(source_prefix: str, *, target_relpath: Path) -> None:
        normalized_source = source_prefix.strip().rstrip("/")
        if not normalized_source:
            return
        normalized_target = target_relpath.as_posix().rstrip("/")
        key = (normalized_source, normalized_target)
        if key in seen_rewrite_prefixes:
            return
        dynamic_rewrite_prefixes.append(
            {
                "source_prefix": normalized_source,
                "target_relpath": normalized_target,
            }
        )
        seen_rewrite_prefixes.add(key)

    def add_file(path: Path, *, target_relpath: Path, target_kind: str = "file") -> None:
        resolved = _resolve_source_for_bundle(path, project_root=project_root)
        if not resolved.is_file():
            return
        source_key = str(resolved)
        entry_key = (source_key, target_relpath.as_posix())
        if entry_key in seen_bundle_entries:
            return
        bundle_items.append(
            {
                "source_path": source_key,
                "target_relpath": target_relpath.as_posix(),
                "target_kind": target_kind,
            }
        )
        seen_bundle_entries.add(entry_key)

    def add_source(
        path: Path,
        *,
        target_kind: str,
        include_associated_dataset: bool = False,
        force_canonical_target: bool = False,
    ) -> None:
        raw_source_prefix = str(path)
        resolved = _resolve_source_for_bundle(path, project_root=project_root)
        if not resolved.exists():
            return
        target_relpath = _target_relpath_for_source(resolved, project_root=project_root)
        if target_relpath is None:
            return
        effective_target_relpath = target_relpath
        use_resume_sync = (
            target_kind == "run-dir"
            and resume_requested
            and not force_canonical_target
        )
        if use_resume_sync:
            effective_target_relpath = Path(f"{target_relpath.as_posix()}.__resume_sync__{job.job_id}")
            add_rewrite_prefix(raw_source_prefix, target_relpath=effective_target_relpath)
            add_rewrite_prefix(str(resolved), target_relpath=effective_target_relpath)
            add_rewrite_prefix(
                f"/opt/pattern-learning-stack/{target_relpath.as_posix()}",
                target_relpath=effective_target_relpath,
            )
        source_key = str(resolved)
        entry_key = (source_key, effective_target_relpath.as_posix())
        if entry_key not in seen_bundle_entries:
            bundle_items.append(
                {
                    "source_path": source_key,
                    "target_relpath": effective_target_relpath.as_posix(),
                    "target_kind": target_kind,
                }
            )
            seen_bundle_entries.add(entry_key)
            if use_resume_sync and refresh_marker_source.is_file():
                add_file(
                    refresh_marker_source,
                    target_relpath=effective_target_relpath
                    / f".server_bundle_refresh_{job.job_id}_{refresh_revision}.txt",
                    target_kind="refresh-marker",
                )
        if include_associated_dataset and resolved.is_dir():
            dataset_manifest = _dataset_manifest_from_run_dir(resolved)
            if dataset_manifest is not None:
                dataset_root = _dataset_root_for_manifest(dataset_manifest)
                if not _dataset_root_is_preloaded(dataset_root, preloaded_dataset_roots):
                    add_source(dataset_root, target_kind="dataset-root")

    def add_deepphy_eval_run_inputs(run_dir: Path) -> None:
        resolved_run_dir = run_dir.resolve()
        target_root = _target_relpath_for_source(resolved_run_dir, project_root=project_root)
        if target_root is None or not resolved_run_dir.is_dir():
            return
        add_file(resolved_run_dir / "pattern_library.json", target_relpath=target_root / "pattern_library.json")
        add_file(resolved_run_dir / "run.json", target_relpath=target_root / "run.json")
        direct_importance = resolved_run_dir / "supervised_label_importance.json"
        if direct_importance.is_file():
            add_file(
                direct_importance,
                target_relpath=target_root / "supervised_label_importance.json",
            )
            return
        run_payload = _load_json_object(resolved_run_dir / "run.json")
        source_supervised_run_dir = run_payload.get("source_supervised_run_dir")
        if not isinstance(source_supervised_run_dir, str) or not source_supervised_run_dir.strip():
            return
        source_importance = Path(source_supervised_run_dir.strip()) / "supervised_label_importance.json"
        add_file(
            source_importance,
            target_relpath=target_root / "supervised_label_importance.json",
        )

    def add_automatic_source_run_inputs(run_dir: Path) -> None:
        resolved_run_dir = _resolve_source_for_bundle(run_dir, project_root=project_root)
        target_root = _target_relpath_for_source(resolved_run_dir, project_root=project_root)
        if target_root is None or not resolved_run_dir.is_dir():
            return
        add_file(resolved_run_dir / "pattern_library.json", target_relpath=target_root / "pattern_library.json")
        add_file(resolved_run_dir / "run.json", target_relpath=target_root / "run.json")
        add_file(
            resolved_run_dir / "supervised_label_importance.json",
            target_relpath=target_root / "supervised_label_importance.json",
        )

        eval_root = resolved_run_dir / "deepphy_eval"
        requested_eval_id = _first_string(metadata, request_dict, key="source_evaluation_run_id")
        candidate_eval_ids: list[str] = []
        seen_eval_ids: set[str] = set()

        def add_eval_id(eval_id: str) -> None:
            cleaned = str(eval_id).strip()
            if not cleaned or cleaned in seen_eval_ids:
                return
            seen_eval_ids.add(cleaned)
            candidate_eval_ids.append(cleaned)

        if requested_eval_id:
            add_eval_id(requested_eval_id)
        run_payload = _load_json_object(resolved_run_dir / "run.json")
        auto_eval_id = run_payload.get("auto_evaluation_run_id")
        if isinstance(auto_eval_id, str) and auto_eval_id.strip():
            add_eval_id(auto_eval_id)
        if eval_root.is_dir():
            for results_path in sorted(eval_root.glob("*/results.json")):
                if results_path.is_file():
                    add_eval_id(results_path.parent.name)

        for eval_id in candidate_eval_ids:
            add_file(
                eval_root / eval_id / "results.json",
                target_relpath=target_root / "deepphy_eval" / eval_id / "results.json",
            )
            add_file(
                eval_root / eval_id / "summary.json",
                target_relpath=target_root / "deepphy_eval" / eval_id / "summary.json",
            )

        dataset_manifest = _dataset_manifest_from_run_dir(resolved_run_dir)
        if dataset_manifest is not None:
            add_dataset_inputs(dataset_manifest)

    action_name = _first_string(metadata, request_dict, key="action") or ""

    if action_name.startswith("pooltool-reward-"):
        source_run_dir_value = _first_string(metadata, request_dict, key="source_run_dir")
        if source_run_dir_value:
            source_run_dir = Path(source_run_dir_value)
            for filename in ("pattern_library.json", "run.json"):
                source_path = source_run_dir / filename
                if not source_path.is_file():
                    continue
                target_root = _target_relpath_for_source(source_run_dir, project_root=project_root)
                if target_root is None:
                    continue
                bundle_items.append(
                    {
                        "source_path": str(source_path.resolve()),
                        "target_relpath": (target_root / filename).as_posix(),
                        "target_kind": "file",
                    }
                )
        rewrite_prefixes = [
            {
                "source_prefix": str((project_root / "data").resolve()),
                "target_relpath": "data",
            },
            {
                "source_prefix": str((project_root / "configs").resolve()),
                "target_relpath": "configs",
            },
            {
                "source_prefix": str((project_root / "evaluations").resolve()),
                "target_relpath": "evaluations",
            },
            {
                "source_prefix": str((MONOREPO_ROOT / "data").resolve()),
                "target_relpath": "data",
            },
            {
                "source_prefix": str((MONOREPO_ROOT / "configs").resolve()),
                "target_relpath": "configs",
            },
        ]
        return {
            "bundle_items": bundle_items,
            "rewrite_prefixes": dynamic_rewrite_prefixes + rewrite_prefixes,
            "text_rewrite_suffixes": sorted(TEXT_REWRITE_SUFFIXES),
        }

    def add_dataset_inputs(manifest_path: Path) -> None:
        resolved_manifest = _resolve_source_for_bundle(manifest_path, project_root=project_root)
        dataset_root = _dataset_root_for_manifest(resolved_manifest)
        if _dataset_root_is_preloaded(dataset_root, preloaded_dataset_roots):
            return
        manifest_target = _target_relpath_for_source(resolved_manifest, project_root=project_root)
        if manifest_target is not None:
            add_file(
                resolved_manifest,
                target_relpath=manifest_target,
                target_kind="dataset-manifest",
            )
        requested_splits = _requested_dataset_splits(request_dict)
        if not requested_splits:
            add_source(dataset_root, target_kind="dataset-root")
            return
        split_targets_added = False
        for split_dir in _dataset_split_dirs_from_manifest(
            resolved_manifest,
            requested_splits=requested_splits,
        ):
            add_source(split_dir, target_kind="dataset-split")
            split_targets_added = True
        if not split_targets_added:
            add_source(dataset_root, target_kind="dataset-root")

    dataset_manifest_value = _first_string(metadata, request_dict, key="dataset_manifest")
    if dataset_manifest_value:
        add_dataset_inputs(Path(dataset_manifest_value))

    for key in ("source_supervised_run_dir", "pattern_run_dir", "source_run_dir", "run_dir"):
        raw = _first_string(metadata, request_dict, key=key)
        if raw:
            if action_name == "unsupervised-launch" and key == "run_dir" and resume_requested:
                _normalize_unsupervised_resume_run_dir(
                    Path(raw),
                    project_root=project_root,
                )
            if action_name == "evaluate-deepphy" and key == "pattern_run_dir":
                add_deepphy_eval_run_inputs(Path(raw))
                continue
            if action_name == "automatic-launch" and key == "source_supervised_run_dir":
                add_automatic_source_run_inputs(Path(raw))
                continue
            add_source(
                Path(raw),
                target_kind="run-dir",
                include_associated_dataset=True,
                force_canonical_target=(
                    resume_requested
                    and key == "run_dir"
                    and action_name in {"unsupervised-launch", "automatic-launch"}
                ),
            )

    targets_payload = request_dict.get("targets")
    if isinstance(targets_payload, list):
        for raw_target in targets_payload:
            if not isinstance(raw_target, dict):
                continue
            target_run_dir = raw_target.get("pattern_run_dir")
            if not isinstance(target_run_dir, str) or not target_run_dir.strip():
                continue
            add_source(
                Path(target_run_dir.strip()),
                target_kind="run-dir",
                include_associated_dataset=True,
            )

    for key in ("source_pattern_library", "pattern_library"):
        raw = _first_string(metadata, request_dict, key=key)
        if not raw:
            continue
        source_path = Path(raw)
        if source_path.name == "pattern_library.json":
            add_source(source_path.parent, target_kind="run-dir", include_associated_dataset=True)
        else:
            add_source(source_path, target_kind="file")

    for key in ("config_path",):
        # Prefer the nested request payload here. The server may intentionally
        # override a stale top-level config_path with a generated policy config
        # stored under metadata["request"]["config_path"].
        raw = _first_string(request_dict, metadata, key=key)
        if raw:
            add_source(Path(raw), target_kind="file")

    raw_trace_path = _first_string(metadata, request_dict, key="trace_path")
    if raw_trace_path:
        add_source(Path(raw_trace_path), target_kind="file")

    raw_run_dirs = request_dict.get("run_dirs")
    if isinstance(raw_run_dirs, list):
        for item in raw_run_dirs:
            if isinstance(item, str) and item.strip():
                add_source(Path(item.strip()), target_kind="run-dir", include_associated_dataset=True)

    rewrite_prefixes = [
        {
            "source_prefix": "/home/sean/Services/pattern-learning-server/configs",
            "target_relpath": "configs",
        },
        {
            "source_prefix": "/home/sean/Services/pattern-learning-client/configs",
            "target_relpath": "configs",
        },
        {
            "source_prefix": "/home/sean/Services/pattern-learning-server/data",
            "target_relpath": "data",
        },
        {
            "source_prefix": "/home/sean/Services/pattern-learning-client/data",
            "target_relpath": "data",
        },
        {
            "source_prefix": "/home/sean/Services",
            "target_relpath": ".",
        },
        {
            "source_prefix": "/home/sean",
            "target_relpath": ".",
        },
        {
            "source_prefix": str((project_root / "data").resolve()),
            "target_relpath": "data",
        },
        {
            "source_prefix": str((project_root / "configs").resolve()),
            "target_relpath": "configs",
        },
        {
            "source_prefix": str((project_root / "evaluations").resolve()),
            "target_relpath": "evaluations",
        },
        {
            "source_prefix": str((MONOREPO_ROOT / "data").resolve()),
            "target_relpath": "data",
        },
        {
            "source_prefix": str((MONOREPO_ROOT / "configs").resolve()),
            "target_relpath": "configs",
        },
    ]
    return {
        "bundle_items": bundle_items,
        "rewrite_prefixes": dynamic_rewrite_prefixes + rewrite_prefixes,
        "text_rewrite_suffixes": sorted(TEXT_REWRITE_SUFFIXES),
    }


def build_job_input_bundle(job: JobRecord, *, project_root: Path) -> bytes:
    plan = build_job_input_plan(job, project_root=project_root)
    return build_job_input_bundle_from_plan(plan)


def build_job_input_bundle_from_plan(plan: dict[str, Any]) -> bytes:
    bundle = io.BytesIO()
    added: set[str] = set()
    with tarfile.open(fileobj=bundle, mode="w:gz") as archive:
        for item in plan["bundle_items"]:
            source_path = Path(str(item["source_path"]))
            target_relpath = str(item["target_relpath"])
            if target_relpath in added:
                continue
            archive.add(source_path, arcname=target_relpath, recursive=True)
            added.add(target_relpath)
    return bundle.getvalue()


def _client_preloaded_dataset_roots(
    client: ClientRecord | None,
    *,
    project_root: Path,
) -> set[str]:
    if client is None:
        return set()
    metadata = client.capabilities.metadata or {}
    raw_roots = metadata.get("preloaded_dataset_roots")
    if not isinstance(raw_roots, list):
        return set()
    normalized: set[str] = set()
    for item in raw_roots:
        if not isinstance(item, str) or not item.strip():
            continue
        resolved = _resolve_source_for_bundle(Path(item.strip()), project_root=project_root)
        if resolved.exists():
            normalized.add(str(resolved.resolve()))
    return normalized


def _dataset_root_is_preloaded(dataset_root: Path, preloaded_roots: set[str]) -> bool:
    if not preloaded_roots:
        return False
    try:
        resolved = dataset_root.resolve()
    except OSError:
        return False
    return str(resolved) in preloaded_roots


def _first_string(*sources: dict[str, Any], key: str) -> str | None:
    for source in sources:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _job_refresh_revision(job: JobRecord) -> str:
    raw_value = job.updated_at or job.created_at or "unknown"
    raw = str(raw_value).strip()
    return (
        raw.replace(":", "")
        .replace("-", "")
        .replace(".", "")
        .replace("T", "T")
        .replace("Z", "Z")
    )


def _resolve_source_for_bundle(path: Path, *, project_root: Path) -> Path:
    raw_original = str(path)
    resolved = path.resolve(strict=False)
    if resolved.exists():
        return resolved
    raw_candidates = [raw_original, str(resolved)]
    rewrites = [
        ("/home/sean/Services/pattern-learning-server/configs", project_root / "configs"),
        ("/home/sean/Services/pattern-learning-server/data", project_root / "data"),
        ("/home/sean/Services/pattern-learning-server/evaluations", project_root / "evaluations"),
        ("/home/sean/Services/pattern-learning-client/configs", project_root / "configs"),
        ("/home/sean/Services/pattern-learning-client/data", project_root / "data"),
        ("/home/sean/Services/pattern-learning-client/evaluations", project_root / "evaluations"),
        ("/opt/pattern-learning-stack/configs", project_root / "configs"),
        ("/opt/pattern-learning-stack/data", project_root / "data"),
        ("/opt/pattern-learning-stack/evaluations", project_root / "evaluations"),
        ("/var/opt/pattern-learning-stack/configs", project_root / "configs"),
        ("/var/opt/pattern-learning-stack/data", project_root / "data"),
        ("/var/opt/pattern-learning-stack/evaluations", project_root / "evaluations"),
        (
            "/var/home/sean/Services/Work/pattern-learning-server/configs",
            project_root / "configs",
        ),
        (
            "/var/home/sean/Services/Work/pattern-learning-server/data",
            project_root / "data",
        ),
        (
            "/var/home/sean/Services/Work/pattern-learning-server/evaluations",
            project_root / "evaluations",
        ),
        (
            "/var/home/sean/Services/Work/pattern-learning-client/configs",
            project_root / "configs",
        ),
        (
            "/var/home/sean/Services/Work/pattern-learning-client/data",
            project_root / "data",
        ),
        (
            "/var/home/sean/Services/Work/pattern-learning-client/evaluations",
            project_root / "evaluations",
        ),
    ]
    for raw in raw_candidates:
        for prefix, target_root in rewrites:
            if raw == prefix or raw.startswith(prefix + "/"):
                suffix = raw[len(prefix):].lstrip("/")
                candidate = (target_root / suffix).resolve(strict=False)
                if candidate.exists():
                    return candidate
    return resolved


def _dataset_manifest_from_run_dir(run_dir: Path) -> Path | None:
    run_json = run_dir / "run.json"
    if not run_json.is_file():
        return None
    try:
        payload = json.loads(run_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    scorer_config = payload.get("scorer_config")
    if not isinstance(scorer_config, dict):
        return None
    dataset_manifest = scorer_config.get("dataset_manifest")
    if not isinstance(dataset_manifest, str) or not dataset_manifest.strip():
        return None
    manifest_path = Path(dataset_manifest.strip())
    if manifest_path.exists():
        return manifest_path
    dataset_id = payload.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        return None
    try:
        env_id = run_dir.resolve().parent.name
        project_root = run_dir.resolve().parents[3]
    except IndexError:
        return None
    fallback = (
        project_root
        / "data"
        / "datasets"
        / env_id
        / dataset_id.strip()
        / "dataset_manifest.json"
    )
    return fallback if fallback.exists() else None


def _dataset_root_for_manifest(manifest_path: Path) -> Path:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return manifest_path.parent
    output_root = payload.get("output_root")
    if isinstance(output_root, str) and output_root.strip():
        candidate = Path(output_root.strip())
        if candidate.exists():
            return candidate
    return manifest_path.parent


def _normalize_unsupervised_resume_run_dir(
    run_dir: Path,
    *,
    project_root: Path,
) -> None:
    resolved_run_dir = _resolve_source_for_bundle(run_dir, project_root=project_root)
    if not resolved_run_dir.is_dir():
        return
    proposed_labels_path = resolved_run_dir / "proposed_labels.json"
    proposed_labels_txt_path = resolved_run_dir / "proposed_labels.txt"
    run_json_path = resolved_run_dir / "run.json"
    run_payload = _load_json_object(run_json_path)
    raw_labels: list[dict[str, Any]] = []

    proposed_payload = _load_json_object(proposed_labels_path)
    proposed_payload_labels = proposed_payload.get("labels")
    if isinstance(proposed_payload_labels, list):
        raw_labels.extend(item for item in proposed_payload_labels if isinstance(item, dict))

    if not raw_labels:
        payload_label_details = run_payload.get("proposed_label_details")
        if isinstance(payload_label_details, list):
            raw_labels.extend(item for item in payload_label_details if isinstance(item, dict))

    if not raw_labels:
        payload_labels = run_payload.get("proposed_labels")
        if isinstance(payload_labels, list):
            for item in payload_labels:
                if not isinstance(item, str):
                    continue
                label = item.strip()
                if not label:
                    continue
                raw_labels.append({"label": label, "description": label.replace("_", " ")})

    if not raw_labels and proposed_labels_txt_path.is_file():
        try:
            txt_labels = proposed_labels_txt_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            txt_labels = []
        for item in txt_labels:
            label = str(item).strip()
            if not label:
                continue
            raw_labels.append({"label": label, "description": label.replace("_", " ")})

    if not raw_labels:
        return

    proposed_labels: list[dict[str, str]] = []
    descriptions: dict[str, str] = {}
    for item in raw_labels:
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        description = str(item.get("description") or "").strip() or label.replace("_", " ")
        proposed_labels.append({"label": label, "description": description})
        descriptions[label] = description
    if not proposed_labels:
        return

    missing_resume_label_files = (not proposed_labels_path.is_file()) or (not proposed_labels_txt_path.is_file())

    # Keep the latest completed checkpoint for each label index and archive older
    # duplicates so resume does not pick stale label_result.json files.
    checkpoints_by_index: dict[int, list[tuple[Path, str, int]]] = {}
    for result_path in resolved_run_dir.glob("label_*/label_result.json"):
        payload = _load_json_object(result_path)
        label_index = payload.get("label_index")
        label = payload.get("label")
        if not isinstance(label_index, int) or not isinstance(label, str):
            continue
        try:
            mtime_ns = result_path.stat().st_mtime_ns
        except OSError:
            continue
        checkpoints_by_index.setdefault(label_index, []).append(
            (result_path, label.strip(), int(mtime_ns))
        )

    canonical_labels: dict[int, str] = {}
    duplicates_found = False
    for label_index, entries in checkpoints_by_index.items():
        entries.sort(key=lambda item: (item[2], str(item[0])), reverse=True)
        keep_path, keep_label, _ = entries[0]
        canonical_labels[label_index] = keep_label
        for stale_path, _, _ in entries[1:]:
            duplicates_found = True
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            archive_path = stale_path.with_name(f"label_result.stale_{timestamp}.json")
            suffix = 1
            while archive_path.exists():
                archive_path = stale_path.with_name(f"label_result.stale_{timestamp}_{suffix}.json")
                suffix += 1
            try:
                stale_path.rename(archive_path)
            except OSError:
                # If rename fails, keep going; this normalizer is best-effort.
                continue

    if not canonical_labels and not duplicates_found and not missing_resume_label_files:
        return

    remaining = list(proposed_labels)
    rebuilt: list[dict[str, str]] = []
    for index in range(1, len(proposed_labels) + 1):
        canonical_label = canonical_labels.get(index)
        if canonical_label:
            description = descriptions.get(canonical_label) or canonical_label.replace("_", " ")
            rebuilt.append({"label": canonical_label, "description": description})
            remaining = [item for item in remaining if item["label"] != canonical_label]
            continue
        if remaining:
            rebuilt.append(remaining.pop(0))

    if len(rebuilt) != len(proposed_labels):
        return
    if rebuilt == proposed_labels and not duplicates_found and not missing_resume_label_files:
        return

    proposed_labels_path.write_text(
        json.dumps({"labels": rebuilt}, indent=2) + "\n",
        encoding="utf-8",
    )
    proposed_labels_txt_path.write_text(
        "".join(f"{item['label']}\n" for item in rebuilt),
        encoding="utf-8",
    )
    if run_payload:
        run_payload["proposed_labels"] = [item["label"] for item in rebuilt]
        run_payload["proposed_label_details"] = rebuilt
        run_json_path.write_text(json.dumps(run_payload, indent=2) + "\n", encoding="utf-8")


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _requested_dataset_splits(request_payload: dict[str, Any]) -> list[str]:
    split_names: list[str] = []
    for key in ("split", "validation_split"):
        raw_value = request_payload.get(key)
        if isinstance(raw_value, str) and raw_value.strip():
            split_names.append(raw_value.strip())
            continue
        if not isinstance(raw_value, list):
            continue
        for item in raw_value:
            if isinstance(item, str) and item.strip():
                split_names.append(item.strip())
    seen: set[str] = set()
    ordered: list[str] = []
    for name in split_names:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def _dataset_split_dirs_from_manifest(
    manifest_path: Path,
    *,
    requested_splits: list[str],
) -> list[Path]:
    dataset_root = _dataset_root_for_manifest(manifest_path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        payload = {}
    split_dirs: list[Path] = []
    seen: set[str] = set()
    split_payloads = payload.get("splits")
    if isinstance(split_payloads, list):
        for item in split_payloads:
            if not isinstance(item, dict):
                continue
            split_name = item.get("name")
            if not isinstance(split_name, str) or split_name not in requested_splits:
                continue
            relative_path = item.get("relative_path")
            candidate = (
                dataset_root / relative_path
                if isinstance(relative_path, str) and relative_path.strip()
                else dataset_root / split_name
            )
            if candidate.exists():
                key = str(candidate.resolve())
                if key not in seen:
                    seen.add(key)
                    split_dirs.append(candidate)
    if split_dirs:
        return split_dirs
    for split_name in requested_splits:
        candidate = dataset_root / split_name
        if not candidate.exists():
            continue
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        split_dirs.append(candidate)
    return split_dirs


def _target_relpath_for_source(source: Path, *, project_root: Path) -> Path | None:
    mappings = [
        (project_root / "data", Path("data")),
        (project_root / "configs", Path("configs")),
        (project_root / "evaluations", Path("evaluations")),
        (MONOREPO_ROOT / "data", Path("data")),
        (MONOREPO_ROOT / "configs", Path("configs")),
        (MONOREPO_ROOT, Path(".")),
    ]
    resolved = source.resolve()
    for source_root, target_root in mappings:
        source_root_resolved = source_root.resolve()
        try:
            rel = resolved.relative_to(source_root_resolved)
        except ValueError:
            continue
        if target_root == Path("."):
            return rel
        return target_root / rel
    return None
