from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import imageio.v2 as imageio
from PIL import Image
from adapters.registry import AdapterRegistry
from debug import progress_log
from deepphy_eval import default_benchmark_task_limit, default_run_id
from dashboard.actions import python_executable_for_env
from dashboard.actions import (
    DEFAULT_LIBRARY_HOLDOUT_EVALUATION_TEMPLATE,
    LIBRARY_HOLDOUT_EVALUATION_TEMPLATES,
    DeepPHYEvaluationActionRequest,
    LibraryHoldoutEvaluationActionRequest,
    build_deepphy_eval_argv,
)
from llm_config import GLOBAL_MIN_COMPLETION_TOKENS, load_shared_llm_config
from pattern_learning.patterns.config import load_pattern_learning_config
from pattern_learning.patterns.learning import load_learning_samples_from_dataset_manifest, run_pattern_learning
from pattern_learning.patterns.losses import default_loss_configs
from pattern_library_paths import (
    default_automatic_run_dir,
    PATTERN_LIBRARY_FILENAME,
    default_unsupervised_run_dir,
    pattern_library_path_for_run_dir,
    resolve_pattern_library_path,
    resolve_pattern_run_dir,
)
from pattern_library_registry import load_pattern_library, save_pattern_library
from pattern_library_schema import PatternLibrary
from pattern_learning.patterns.scoring import (
    _request_json,
    extract_chat_completion_text,
    request_chat_completion_with_text_retry,
)
from paths import get_artifact_layout
from reward_programs.pooltool import (
    PooltoolRewardGoal,
    default_pooltool_reward_goals,
    generate_pooltool_random_scene,
    run_pooltool_optimization_experiment,
    run_pooltool_optimization_goal_suite,
    run_pooltool_single_scene_experiment,
    run_pooltool_value_model_experiment,
    run_pooltool_value_model_sweep,
)

LABEL_PROPOSAL_MAX_ATTEMPTS = 3
AUTO_SOURCE_EVALUATION_RUN_ID = "__auto_source_eval__"
LABEL_PROPOSAL_ASSISTANT_PREFILL = "LABEL: "
LABEL_PROPOSAL_TOTAL_TOKEN_BUDGET = 20_096
LABEL_PROPOSAL_MAX_OUTPUT_TOKENS = GLOBAL_MIN_COMPLETION_TOKENS
LABEL_PROPOSAL_TOKEN_OVERHEAD = 512
LABEL_PROPOSAL_BATCH_TRACE_COUNT = 3
LABEL_PROPOSAL_BATCH_LABEL_COUNT = 3
LIBRARY_EVALUATIONS_DIRNAME = "library_evaluations"
LIBRARY_ABLATION_MANIFEST_VERSION = 2
LIBRARY_ABLATION_LEGACY_MANIFEST_NAME = "manifest.legacy_v1.json"
LABEL_PROPOSAL_INPUT_TOKEN_BUDGET = max(
    1,
    LABEL_PROPOSAL_TOTAL_TOKEN_BUDGET
    - LABEL_PROPOSAL_MAX_OUTPUT_TOKENS
    - LABEL_PROPOSAL_TOKEN_OVERHEAD,
)
LABEL_PROPOSAL_INPUT_WORD_BUDGET = max(
    1,
    int(LABEL_PROPOSAL_INPUT_TOKEN_BUDGET * 3 / 4),
)
TRUE_UNSUPERVISED_TRACE_COUNT = 10
TRUE_UNSUPERVISED_LABELS_PER_TRACE = 10
TRUE_UNSUPERVISED_FINAL_LABEL_COUNT = 10
TRUE_UNSUPERVISED_MAX_OUTPUT_TOKENS = GLOBAL_MIN_COMPLETION_TOKENS
TRUE_UNSUPERVISED_FRAME_COUNT = 4


@dataclass(frozen=True)
class ProposedUnsupervisedLabel:
    label: str
    description: str


def _normalize_labels(lines: Iterable[str]) -> list[str]:
    labels: list[str] = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        labels.append(stripped)
    return labels


def _expected_labels_for_env(env_id: str) -> list[str]:
    layout = get_artifact_layout()
    labels_path = layout.config_root / env_id / "labels.txt"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Expected labels file not found: {labels_path}")
    return _normalize_labels(labels_path.read_text(encoding="utf-8").splitlines())


def _normalized_unsupervised_ablation_label_count(
    *,
    env_id: str,
    available_labels: list[str],
    enabled_labels: list[str],
) -> int:
    source_count = max(1, len(_normalize_labels(available_labels)))
    enabled_count = max(1, len(_normalize_labels(enabled_labels)))
    try:
        target_count = max(1, len(_expected_labels_for_env(env_id)))
    except Exception:
        return enabled_count
    if source_count <= target_count:
        return enabled_count
    scaled = int(round((float(enabled_count) / float(source_count)) * float(target_count)))
    return max(1, min(target_count, scaled))


def _load_run_payload(run_dir: Path) -> dict[str, Any]:
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"Expected run.json not found: {run_path}")
    try:
        payload = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed run.json at {run_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed run.json at {run_path}: expected JSON object")
    return payload


def _derive_run_dir_from_output_path(output_path: Path) -> Path:
    # Mirrors `cli.pattern-learn` effective_artifact_root derivation.
    if output_path.name == "pattern_library.json":
        return output_path.parent
    return output_path.parent / output_path.stem


def _write_runtime_dataset_manifest(
    *,
    source_manifest: Path,
    target_manifest: Path,
    train_sample_count: int | None = None,
    validation_sample_count: int | None = None,
) -> Path:
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset manifest must be a JSON object: {source_manifest}")

    splits = payload.get("splits")
    if isinstance(splits, list):
        for split in splits:
            if not isinstance(split, dict):
                continue
            name = str(split.get("name") or "").strip().lower()
            if name == "train" and train_sample_count is not None:
                split["sample_count"] = int(train_sample_count)
                if isinstance(split.get("completed_samples"), int):
                    split["completed_samples"] = min(int(split["completed_samples"]), int(train_sample_count))
                env_counts = split.get("env_counts")
                if isinstance(env_counts, dict):
                    split["env_counts"] = {
                        key: min(int(value), int(train_sample_count))
                        for key, value in env_counts.items()
                        if isinstance(value, int)
                    }
            elif name == "validation" and validation_sample_count is not None:
                split["sample_count"] = int(validation_sample_count)
                if isinstance(split.get("completed_samples"), int):
                    split["completed_samples"] = min(int(split["completed_samples"]), int(validation_sample_count))
                env_counts = split.get("env_counts")
                if isinstance(env_counts, dict):
                    split["env_counts"] = {
                        key: min(int(value), int(validation_sample_count))
                        for key, value in env_counts.items()
                        if isinstance(value, int)
                    }

    config = payload.get("config")
    if isinstance(config, dict):
        config_splits = config.get("splits")
        if isinstance(config_splits, list):
            for split in config_splits:
                if not isinstance(split, dict):
                    continue
                name = str(split.get("name") or "").strip().lower()
                if name == "train" and train_sample_count is not None:
                    split["sample_count"] = int(train_sample_count)
                elif name == "validation" and validation_sample_count is not None:
                    split["sample_count"] = int(validation_sample_count)

    notes = payload.get("notes")
    note_text = (
        "runtime-derived reduced manifest for targeted supervised learning "
        f"(train={train_sample_count if train_sample_count is not None else 'default'}, "
        f"validation={validation_sample_count if validation_sample_count is not None else 'default'})"
    )
    if isinstance(notes, list):
        notes.append(note_text)
    else:
        payload["notes"] = [note_text]

    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    target_manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target_manifest


def _learned_labels_from_run_json(run_dir: Path) -> set[str]:
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        return set()
    payload = _load_run_payload(run_dir)
    label_runs = payload.get("label_runs")
    if not isinstance(label_runs, list):
        return set()
    learned: set[str] = set()
    for item in label_runs:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        if isinstance(label, str) and label.strip():
            learned.add(label.strip())
    return learned


def _learned_labels_from_artifacts(run_dir: Path) -> set[str]:
    learned: set[str] = set()
    for label_dir in sorted(run_dir.glob("label_*")):
        if not label_dir.is_dir():
            continue
        for candidate_json in sorted(label_dir.glob("candidate_*/candidate.json")):
            try:
                payload = json.loads(candidate_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            label = payload.get("label")
            if isinstance(label, str) and label.strip():
                learned.add(label.strip())
                break
    return learned


def _learned_labels_from_library(run_dir: Path) -> set[str]:
    library_path = pattern_library_path_for_run_dir(run_dir)
    if not library_path.is_file():
        return set()
    try:
        library = load_pattern_library(library_path)
    except Exception:
        return set()
    learned: set[str] = set()
    for detector in library.detectors:
        label = getattr(detector, "label", None)
        if isinstance(label, str) and label.strip():
            learned.add(label.strip())
    return learned


def _require_complete_labels(*, env_id: str, run_dir: Path) -> None:
    expected = _expected_labels_for_env(env_id)
    _require_complete_labels_for_expected(run_dir=run_dir, expected_labels=expected)


def _require_complete_labels_for_expected(
    *,
    run_dir: Path,
    expected_labels: list[str],
) -> None:
    expected = [label.strip() for label in expected_labels if label.strip()]
    if not expected:
        raise ValueError("Expected labels list produced no labels")

    learned = (
        _learned_labels_from_run_json(run_dir)
        | _learned_labels_from_artifacts(run_dir)
        | _learned_labels_from_library(run_dir)
    )

    if not learned:
        raise FileNotFoundError(
            f"No learned label artifacts found under {run_dir} "
            "(missing run.json, label_* artifacts, and pattern_library.json detectors)"
        )

    missing = sorted(set(expected) - learned)
    if missing:
        raise RuntimeError(
            "Pattern learning did not produce all expected labels; "
            f"missing {len(missing)}: {', '.join(missing)}"
        )


def _library_labels_from_path(library_path: Path) -> set[str]:
    if not library_path.is_file():
        raise FileNotFoundError(f"Expected pattern library not found: {library_path}")
    try:
        library = load_pattern_library(library_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load pattern library at {library_path}: {exc}") from exc
    labels = {
        detector.label.strip()
        for detector in library.detectors
        if isinstance(detector.label, str) and detector.label.strip()
    }
    if not labels:
        raise RuntimeError(f"Pattern library contains no detectors: {library_path}")
    return labels


def _require_current_library_labels(
    *,
    library_path: Path,
    expected_labels: list[str],
) -> None:
    expected = [label.strip() for label in expected_labels if label.strip()]
    if not expected:
        raise ValueError("Expected labels list produced no labels")
    library_labels = _library_labels_from_path(library_path)
    missing = sorted(set(expected) - library_labels)
    if missing:
        raise RuntimeError(
            "Current pattern library is incomplete; "
            f"missing {len(missing)} expected labels: {', '.join(missing)}"
        )


def _completed_labels_for_run(run_dir: Path) -> list[str]:
    learned = (
        _learned_labels_from_run_json(run_dir)
        | _learned_labels_from_artifacts(run_dir)
        | _learned_labels_from_library(run_dir)
    )
    return sorted(label for label in learned if label.strip())


def _resume_can_skip_supervised_learning(*, env_id: str, run_dir: Path) -> bool:
    try:
        _require_complete_labels(env_id=env_id, run_dir=run_dir)
        _require_current_library_labels(
            library_path=pattern_library_path_for_run_dir(run_dir),
            expected_labels=_expected_labels_for_env(env_id),
        )
        return True
    except (FileNotFoundError, RuntimeError) as exc:
        progress_log(
            "dashboard.workflows.supervised.learning_resume_requires_remaining_labels",
            env=env_id,
            run_dir=run_dir,
            reason=str(exc),
        )
        return False


def _run_child(argv: list[str]) -> int:
    # Inherit stdout/stderr so dashboard job logs include child output.
    progress_log("dashboard.workflows.child.start", argv=argv)
    proc = subprocess.run(argv, check=False)
    progress_log(
        "dashboard.workflows.child.done",
        argv=argv[:4],
        returncode=proc.returncode,
    )
    return int(proc.returncode)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _library_evaluation_root(*, run_dir: Path, evaluation_run_id: str) -> Path:
    return run_dir / LIBRARY_EVALUATIONS_DIRNAME / evaluation_run_id


def _slugify_library_eval_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return token or "label"


def _load_library_labels(run_dir: Path) -> list[str]:
    library = load_pattern_library(pattern_library_path_for_run_dir(run_dir))
    labels = sorted(
        {
            detector.label.strip()
            for detector in library.detectors
            if isinstance(detector.label, str) and detector.label.strip()
        }
    )
    if not labels:
        raise ValueError(f"No detectors found in library for {run_dir}")
    return labels


def _library_ablation_case_id(label_count: int) -> str:
    return f"label-count-{int(label_count):03d}"


def _library_ablation_case_label(*, label_count: int, total_label_count: int) -> str:
    if label_count >= total_label_count:
        return "Full library"
    noun = "label" if label_count == 1 else "labels"
    return f"{label_count} {noun}"


def _library_ablation_select_enabled_labels(
    *,
    available_labels: list[str],
    label_count: int,
    repetition_index: int,
    source_run_token: str,
) -> list[str]:
    if label_count >= len(available_labels):
        return list(available_labels)
    seed_text = f"{source_run_token}:{label_count}:{repetition_index}"
    return sorted(random.Random(seed_text).sample(list(available_labels), label_count))


def _materialize_library_holdout_filtered_library(
    *,
    source_run_dir: Path,
    source_library: PatternLibrary,
    eval_root: Path,
    case_id: str,
    artifact_id: str,
    available_labels: list[str],
    enabled_labels: list[str],
) -> Path:
    enabled_set = set(enabled_labels)
    filtered_detectors = [
        detector
        for detector in source_library.detectors
        if detector.label in enabled_set
    ]
    if not filtered_detectors:
        raise ValueError("At least one label must remain enabled for evaluation.")
    filtered_library = source_library.model_copy(
        update={
            "library_id": f"{source_library.library_id}-holdout-{case_id}",
            "detectors": filtered_detectors,
            "metadata": {
                **source_library.metadata,
                "source_library_id": source_library.library_id,
                "source_run_id": source_run_dir.name,
                "library_holdout_case_id": case_id,
                "library_holdout_artifact_id": artifact_id,
                "enabled_labels": list(enabled_labels),
                "disabled_labels": [
                    label for label in available_labels if label not in enabled_set
                ],
            },
        }
    )
    filtered_library_path = eval_root / "filtered_libraries" / f"{artifact_id}.json"
    save_pattern_library(filtered_library, filtered_library_path)
    return filtered_library_path


def _library_ablation_manifest_v2(
    *,
    available_labels: list[str],
    template_name: str,
    benchmark_preset: str,
    source_run_id: str,
) -> dict[str, Any]:
    manifest_cases: list[dict[str, Any]] = []
    total_label_count = len(available_labels)
    for label_count in range(1, total_label_count + 1):
        manifest_cases.append(
            {
                "case_id": _library_ablation_case_id(label_count),
                "label": _library_ablation_case_label(
                    label_count=label_count,
                    total_label_count=total_label_count,
                ),
                "kind": "baseline" if label_count == total_label_count else "label_count",
                "label_count": label_count,
                "seed_runs": [],
            }
        )
    return {
        "format_version": LIBRARY_ABLATION_MANIFEST_VERSION,
        "strategy": "label_count_sample",
        "template_name": template_name,
        "benchmark_preset": benchmark_preset,
        "source_run_id": source_run_id,
        "available_labels": list(available_labels),
        "cases": manifest_cases,
        "legacy_manifest_paths": [],
    }


def _continuous_library_ablation_manifest(
    *,
    available_labels: list[str],
    template_name: str,
    benchmark_preset: str,
    source_run_id: str,
    baseline_interval: int,
) -> dict[str, Any]:
    return {
        "format_version": LIBRARY_ABLATION_MANIFEST_VERSION,
        "strategy": "continuous_random_holdout",
        "template_name": template_name,
        "benchmark_preset": benchmark_preset,
        "source_run_id": source_run_id,
        "available_labels": list(available_labels),
        "cases": [],
        "legacy_manifest_paths": [],
        "loop_state": {
            "mode": "continuous_random_holdout",
            "baseline_interval": max(0, int(baseline_interval)),
            "next_iteration": 1,
            "total_iterations": 0,
        },
    }


def _library_ablation_manifest_is_v2(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    return (
        int(payload.get("format_version") or 0) >= LIBRARY_ABLATION_MANIFEST_VERSION
        and str(payload.get("strategy") or "").strip().lower() == "label_count_sample"
    )


def _archive_legacy_library_ablation_manifest(
    *,
    eval_root: Path,
    manifest_payload: dict[str, Any],
) -> Path:
    legacy_path = eval_root / LIBRARY_ABLATION_LEGACY_MANIFEST_NAME
    if not legacy_path.is_file():
        _write_json(legacy_path, manifest_payload)
    return legacy_path


def _load_or_initialize_library_ablation_manifest(
    *,
    eval_root: Path,
    manifest_path: Path,
    available_labels: list[str],
    template_name: str,
    benchmark_preset: str,
    source_run_id: str,
) -> dict[str, Any]:
    if manifest_path.is_file():
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest_payload = {}
    else:
        manifest_payload = {}
    manifest = manifest_payload if isinstance(manifest_payload, dict) else {}
    if _library_ablation_manifest_is_v2(manifest):
        manifest.setdefault("available_labels", list(available_labels))
        manifest.setdefault("legacy_manifest_paths", [])
        return manifest
    legacy_paths: list[str] = []
    if manifest:
        legacy_path = _archive_legacy_library_ablation_manifest(
            eval_root=eval_root,
            manifest_payload=manifest,
        )
        legacy_paths.append(legacy_path.name)
    fresh_manifest = _library_ablation_manifest_v2(
        available_labels=available_labels,
        template_name=template_name,
        benchmark_preset=benchmark_preset,
        source_run_id=source_run_id,
    )
    fresh_manifest["legacy_manifest_paths"] = legacy_paths
    return fresh_manifest


def _ensure_library_ablation_pass_planned(
    *,
    eval_root: Path,
    manifest: dict[str, Any],
    available_labels: list[str],
    source_run_id: str,
) -> bool:
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list):
        return False
    planned_any = False
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            continue
        case_id = str(raw_case.get("case_id") or "").strip()
        try:
            label_count = int(raw_case.get("label_count") or 0)
        except (TypeError, ValueError):
            label_count = 0
        if not case_id or label_count < 1 or label_count > len(available_labels):
            continue
        raw_seed_runs = raw_case.get("seed_runs")
        if not isinstance(raw_seed_runs, list):
            raw_seed_runs = []
            raw_case["seed_runs"] = raw_seed_runs
        has_pending = False
        max_seed = 0
        for raw_seed_run in raw_seed_runs:
            if not isinstance(raw_seed_run, dict):
                continue
            try:
                max_seed = max(max_seed, int(raw_seed_run.get("seed") or 0))
            except (TypeError, ValueError):
                pass
            relative_root = str(raw_seed_run.get("relative_root") or "").strip()
            if relative_root and not (eval_root / relative_root / "results.json").is_file():
                has_pending = True
        if has_pending:
            continue
        repetition_index = max_seed + 1
        enabled_labels = _library_ablation_select_enabled_labels(
            available_labels=available_labels,
            label_count=label_count,
            repetition_index=repetition_index,
            source_run_token=source_run_id,
        )
        enabled_set = set(enabled_labels)
        raw_seed_runs.append(
            {
                "seed": repetition_index,
                "evaluation_run_id": f"{case_id}-run-{repetition_index:03d}",
                "relative_root": str(
                    Path("cases") / case_id / f"run-{repetition_index:03d}"
                ),
                "enabled_labels": enabled_labels,
                "held_out_labels": [
                    label for label in available_labels if label not in enabled_set
                ],
            }
        )
        planned_any = True
    return planned_any


def _default_continuous_library_ablation_run_id() -> str:
    return "continuous-ablation-001"


def _next_continuous_library_ablation_iteration_index(
    manifest: dict[str, Any],
) -> int:
    loop_state = manifest.get("loop_state")
    if isinstance(loop_state, dict):
        try:
            next_iteration = int(loop_state.get("next_iteration") or 0)
        except (TypeError, ValueError):
            next_iteration = 0
        if next_iteration >= 1:
            return next_iteration
    highest_iteration = 0
    raw_cases = manifest.get("cases")
    if isinstance(raw_cases, list):
        for raw_case in raw_cases:
            if not isinstance(raw_case, dict):
                continue
            case_id = str(raw_case.get("case_id") or "").strip()
            match = re.fullmatch(r"iteration-(\d{6})", case_id)
            if match is not None:
                highest_iteration = max(highest_iteration, int(match.group(1)))
    return highest_iteration + 1


def _append_continuous_library_ablation_case(
    *,
    env_id: str,
    manifest: dict[str, Any],
    eval_root: Path,
    available_labels: list[str],
    source_run_id: str,
    run_token: str,
    baseline_interval: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        cases = []
        manifest["cases"] = cases
    total_label_count = len(available_labels)
    if total_label_count < 1:
        raise ValueError("At least one label is required for continuous ablations.")
    iteration_index = _next_continuous_library_ablation_iteration_index(manifest)
    seed_text = f"{source_run_id}:{run_token}:continuous:{iteration_index}"
    rng = random.Random(seed_text)
    if total_label_count <= 1:
        held_out_count = 0
    elif baseline_interval > 0 and iteration_index % baseline_interval == 0:
        held_out_count = 0
    else:
        held_out_count = rng.randint(1, total_label_count - 1)
    held_out_labels = (
        sorted(rng.sample(list(available_labels), held_out_count))
        if held_out_count > 0
        else []
    )
    held_out_set = set(held_out_labels)
    enabled_labels = [
        label for label in available_labels if label not in held_out_set
    ]
    if not enabled_labels:
        enabled_labels = list(available_labels)
        held_out_labels = []
        held_out_set = set()
        held_out_count = 0
    case_id = f"iteration-{iteration_index:06d}"
    seed_run = {
        "seed": iteration_index,
        "evaluation_run_id": case_id,
        "relative_root": str(Path("cases") / case_id),
        "enabled_labels": list(enabled_labels),
        "held_out_labels": [
            label for label in available_labels if label not in set(enabled_labels)
        ],
    }
    case_payload = {
        "case_id": case_id,
        "label": (
            "Full library baseline"
            if held_out_count == 0
            else f"Hold out {held_out_count} label"
            + ("" if held_out_count == 1 else "s")
        ),
        "kind": "baseline" if held_out_count == 0 else "random_holdout",
        "label_count": _normalized_unsupervised_ablation_label_count(
            env_id=env_id,
            available_labels=available_labels,
            enabled_labels=enabled_labels,
        ),
        "raw_label_count": len(enabled_labels),
        "enabled_labels": list(enabled_labels),
        "held_out_labels": [
            label for label in available_labels if label not in set(enabled_labels)
        ],
        "seed_runs": [seed_run],
    }
    cases.append(case_payload)
    manifest["format_version"] = max(
        LIBRARY_ABLATION_MANIFEST_VERSION,
        int(manifest.get("format_version") or LIBRARY_ABLATION_MANIFEST_VERSION),
    )
    manifest["strategy"] = "continuous_random_holdout"
    manifest["available_labels"] = list(available_labels)
    loop_state = (
        manifest.get("loop_state")
        if isinstance(manifest.get("loop_state"), dict)
        else {}
    )
    loop_state.update(
        {
            "mode": "continuous_random_holdout",
            "baseline_interval": max(0, int(baseline_interval)),
            "next_iteration": iteration_index + 1,
            "latest_case_id": case_id,
            "total_iterations": iteration_index,
            "latest_eval_root": str((eval_root / seed_run["relative_root"]).resolve()),
        }
    )
    manifest["loop_state"] = loop_state
    return case_payload, seed_run


def _write_library_holdout_request(
    *,
    eval_root: Path,
    action_request: LibraryHoldoutEvaluationActionRequest,
    run_dir: Path,
    manifest: dict[str, Any],
    library_evaluation_run_id: str,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "env_id": action_request.env_id,
        "pattern_run_dir": str(run_dir),
        "format_version": int(manifest.get("format_version") or 1),
        "strategy": str(manifest.get("strategy") or "").strip() or None,
        "template_name": action_request.template_name,
        "benchmark_preset": action_request.benchmark_preset,
        "workers": action_request.workers,
        "model": action_request.model,
        "ollama_base_url": action_request.ollama_base_url,
        "model_timeout": action_request.model_timeout,
        "run_id": library_evaluation_run_id,
        "seed_count": action_request.seed_count,
        "holdout_sample_count": action_request.holdout_sample_count,
        "single_holdout_repeat_count": action_request.single_holdout_repeat_count,
        "pair_holdout_size": action_request.pair_holdout_size,
        "pair_holdout_sample_count": action_request.pair_holdout_sample_count,
    }
    if extra_fields:
        payload.update(extra_fields)
    _write_json(eval_root / "request.json", payload)


def _write_library_holdout_summary(eval_root: Path, manifest: dict[str, Any]) -> None:
    cases = manifest.get("cases") if isinstance(manifest, dict) else []
    total_seed_runs = 0
    completed_seed_runs = 0
    for case in cases if isinstance(cases, list) else []:
        if not isinstance(case, dict):
            continue
        raw_seed_runs = case.get("seed_runs")
        if not isinstance(raw_seed_runs, list):
            continue
        total_seed_runs += len(raw_seed_runs)
        for seed_run in raw_seed_runs:
            if not isinstance(seed_run, dict):
                continue
            relative_root = seed_run.get("relative_root")
            if not isinstance(relative_root, str) or not relative_root.strip():
                continue
            if (eval_root / relative_root / "results.json").is_file():
                completed_seed_runs += 1
    _write_json(
        eval_root / "summary.json",
        {
            "format_version": int(manifest.get("format_version") or 1),
            "strategy": str(manifest.get("strategy") or "").strip() or None,
            "template_name": manifest.get("template_name"),
            "benchmark_preset": manifest.get("benchmark_preset"),
            "total_case_count": len(cases) if isinstance(cases, list) else 0,
            "total_seed_run_count": total_seed_runs,
            "completed_seed_run_count": completed_seed_runs,
            "total_evaluation_run_count": total_seed_runs,
            "completed_evaluation_run_count": completed_seed_runs,
        },
    )


def _supervised_auto_evaluation_run_id(run_dir: Path) -> str:
    run_path = run_dir / "run.json"
    if run_path.is_file():
        try:
            payload = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            raw_value = payload.get("auto_evaluation_run_id")
            if isinstance(raw_value, str) and raw_value.strip():
                return raw_value.strip()
    return "auto-eval-001"


def _persist_auto_evaluation_run_id(run_dir: Path, evaluation_run_id: str) -> None:
    payload = _load_run_payload(run_dir)
    payload["auto_evaluation_run_id"] = evaluation_run_id
    _write_json(run_dir / "run.json", payload)


def _record_historical_auto_evaluation_run_id(run_dir: Path, evaluation_run_id: str) -> None:
    payload = _load_run_payload(run_dir)
    history_raw = payload.get("historical_auto_evaluation_run_ids")
    history = [str(item).strip() for item in history_raw] if isinstance(history_raw, list) else []
    normalized = evaluation_run_id.strip()
    if normalized and normalized not in history:
        history.append(normalized)
        payload["historical_auto_evaluation_run_ids"] = history
        _write_json(run_dir / "run.json", payload)


def _next_supervised_auto_evaluation_run_id(run_dir: Path) -> str:
    deepphy_root = run_dir / "deepphy_eval"
    highest_index = 0
    if deepphy_root.is_dir():
        for child in deepphy_root.iterdir():
            if not child.is_dir():
                continue
            match = re.fullmatch(r"auto-eval-(\d{3})", child.name)
            if match is None:
                continue
            highest_index = max(highest_index, int(match.group(1)))
    return f"auto-eval-{highest_index + 1:03d}"


def _latest_learning_artifact_mtime(run_dir: Path) -> float:
    latest_mtime = 0.0
    for candidate in (
        run_dir / "pattern_library.json",
        run_dir / "run.json",
    ):
        try:
            if candidate.is_file():
                latest_mtime = max(latest_mtime, candidate.stat().st_mtime)
        except OSError:
            continue
    try:
        for label_result_path in run_dir.glob("label_*/label_result.json"):
            if not label_result_path.is_file():
                continue
            latest_mtime = max(latest_mtime, label_result_path.stat().st_mtime)
    except OSError:
        return latest_mtime
    return latest_mtime


def _auto_evaluation_results_exist(run_dir: Path, evaluation_run_id: str) -> bool:
    results_path = run_dir / "deepphy_eval" / evaluation_run_id / "results.json"
    if not results_path.is_file():
        return False
    try:
        results_mtime = results_path.stat().st_mtime
    except OSError:
        return False
    if _latest_learning_artifact_mtime(run_dir) > results_mtime:
        return False
    return True


def _prepare_supervised_auto_evaluation_run_id(run_dir: Path) -> tuple[str, bool]:
    evaluation_run_id = _supervised_auto_evaluation_run_id(run_dir)
    _persist_auto_evaluation_run_id(run_dir, evaluation_run_id)
    if _auto_evaluation_results_exist(run_dir, evaluation_run_id):
        return evaluation_run_id, True
    stale_results_path = run_dir / "deepphy_eval" / evaluation_run_id / "results.json"
    if stale_results_path.is_file():
        _record_historical_auto_evaluation_run_id(run_dir, evaluation_run_id)
        replacement_run_id = _next_supervised_auto_evaluation_run_id(run_dir)
        _persist_auto_evaluation_run_id(run_dir, replacement_run_id)
        progress_log(
            "dashboard.workflows.supervised.evaluation_results_stale",
            run_dir=run_dir,
            stale_evaluation_run_id=evaluation_run_id,
            replacement_evaluation_run_id=replacement_run_id,
        )
        return replacement_run_id, False
    return evaluation_run_id, False


def _run_supervised_auto_evaluation(
    *,
    env_id: str,
    run_dir: Path,
    model_name: str | None = None,
) -> int:
    evaluation_run_id, already_complete = _prepare_supervised_auto_evaluation_run_id(run_dir)
    if already_complete:
        progress_log(
            "dashboard.workflows.supervised.evaluation_already_complete",
            env=env_id,
            run_dir=run_dir,
            evaluation_run_id=evaluation_run_id,
        )
        return 0

    layout = get_artifact_layout()
    llm = load_shared_llm_config(layout.config_root / "LLM.yaml")
    action_request = DeepPHYEvaluationActionRequest.model_validate(
        {
            "env_id": env_id,
            "benchmark_preset": "full",
            "prompt_format": "WM",
            "description_mode": "pattern_text",
            "pattern_run_dir": run_dir,
            "model": model_name if model_name is not None else "",
            "ollama_base_url": llm.base_url,
            "model_timeout": llm.timeout,
            "capture_visuals": True,
            "workers": 1,
            "run_id": evaluation_run_id,
        }
    )
    progress_log(
        "dashboard.workflows.supervised.evaluation_start",
        env=env_id,
        run_dir=run_dir,
        evaluation_run_id=evaluation_run_id,
    )
    eval_code = _run_child(
        [
            str(python_executable_for_env(env_id)),
            *build_deepphy_eval_argv(action_request),
        ]
    )
    if eval_code == 0:
        progress_log(
            "dashboard.workflows.supervised.evaluation_done",
            env=env_id,
            run_dir=run_dir,
            evaluation_run_id=evaluation_run_id,
        )
    return eval_code


def _results_path_for_evaluation_run(
    *,
    source_run_dir: Path,
    source_evaluation_run_id: str,
) -> Path:
    results_path = source_run_dir / "deepphy_eval" / source_evaluation_run_id / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(
            f"Supervised evaluation results not found: {results_path}"
        )
    return results_path


def _resolve_source_evaluation_run_id(
    *,
    source_run_dir: Path,
    requested_run_id: str,
) -> str:
    if requested_run_id != AUTO_SOURCE_EVALUATION_RUN_ID:
        return requested_run_id
    run_json_path = source_run_dir / "run.json"
    if run_json_path.is_file():
        try:
            payload = json.loads(run_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            auto_run_id = payload.get("auto_evaluation_run_id")
            if isinstance(auto_run_id, str) and auto_run_id.strip():
                candidate = source_run_dir / "deepphy_eval" / auto_run_id / "results.json"
                if candidate.is_file():
                    return auto_run_id
    deepphy_eval_root = source_run_dir / "deepphy_eval"
    if deepphy_eval_root.is_dir():
        candidates = sorted(
            (
                path
                for path in deepphy_eval_root.glob("*/results.json")
                if path.is_file()
            ),
            key=lambda path: (path.stat().st_mtime_ns, path.parent.name),
            reverse=True,
        )
        if candidates:
            return candidates[0].parent.name
    raise FileNotFoundError(
        "No supervised evaluation results are available yet for "
        f"{source_run_dir}. Wait for the source supervised run evaluation to complete."
    )


def _flatten_reasoning_trace_records_from_payload(payload: Any) -> list[dict[str, Any]]:
    containers: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            containers.extend(item for item in results if isinstance(item, dict))
        records = payload.get("records")
        if isinstance(records, list):
            containers.append(payload)
    flattened: list[dict[str, Any]] = []
    for result in containers:
        env_id = result.get("env_id")
        task_id = result.get("task_id")
        description_mode = result.get("description_mode")
        records = result.get("records")
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            prompt = record.get("prompt")
            if not isinstance(prompt, dict):
                continue
            flattened.append(
                {
                    "env_id": env_id,
                    "task_id": task_id,
                    "description_mode": description_mode,
                    "record_kind": record.get("record_kind"),
                    "index": record.get("index"),
                    "trace_description": record.get("trace_description"),
                    "raw_model_output": record.get("raw_model_output"),
                    "parsed_action": record.get("parsed_action"),
                    "success": record.get("success"),
                    "terminal_status": record.get("terminal_status"),
                    "prompt": {
                        "prompt_format": prompt.get("prompt_format"),
                        "description_mode": prompt.get("description_mode"),
                        "system_prompt": prompt.get("system_prompt"),
                        "user_prompt": prompt.get("user_prompt"),
                    },
                }
            )
    return flattened


def _load_reasoning_trace_records(results_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed evaluation results payload: {results_path}")
    flattened = _flatten_reasoning_trace_records_from_payload(payload)
    if flattened:
        return flattened
    tasks_root = results_path.parent / "tasks"
    for task_result_path in sorted(tasks_root.glob("*/*/task_result.json")):
        try:
            task_payload = json.loads(task_result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        flattened.extend(_flatten_reasoning_trace_records_from_payload(task_payload))
    return flattened


def _sample_reasoning_traces(
    records: list[dict[str, Any]],
    *,
    reasoning_trace_count: int,
) -> list[dict[str, Any]]:
    sampled = [
        record
        for record in records
        if isinstance(record.get("raw_model_output"), str)
        and isinstance(record.get("trace_description"), str)
    ]
    sampled = sampled[:reasoning_trace_count]
    if len(sampled) < reasoning_trace_count:
        raise RuntimeError(
            f"Requested {reasoning_trace_count} reasoning traces, but only {len(sampled)} were available"
        )
    return sampled


def _build_label_proposal_prompt(
    *,
    env_id: str,
    sampled_reasoning_traces: list[dict[str, Any]],
    proposed_label_count: int,
    forbidden_labels: Iterable[str] = (),
) -> tuple[str, str]:
    system_prompt = (
        "You propose low-level automatic pattern labels for a physics-simulation reasoning library.\n"
        "Return exactly the requested number of labels.\n"
        "For each label, output exactly two lines in this format:\n"
        "LABEL: <short label>\n"
        "DESCRIPTION: <one-sentence description of what the pattern is>\n"
        "Each label must be short, a few words at most, and describe a recurring low-level pattern.\n"
        "The description must explain what the pattern is conceptually, not how to detect it.\n"
        "Do not number the labels. Do not include reasoning or headings."
    )
    header_sections = [
        f"Environment: {env_id}",
        f"Propose exactly {proposed_label_count} new automatic labels from these evaluation reasoning traces.",
        "The automatic layer only sees supervised pattern detections, not raw simulation state.",
        "Each description must be a single sentence about the meaning of the pattern, never a detection recipe.",
    ]
    blocked = [label.strip() for label in forbidden_labels if isinstance(label, str) and label.strip()]
    if blocked:
        header_sections.append(
            "Do not propose any of these labels: " + ", ".join(sorted(blocked))
        )
    trace_sections: list[str] = []
    for index, item in enumerate(sampled_reasoning_traces, start=1):
        pattern_summary = _summarize_pattern_trace_labels(item.get("trace_description"))
        model_output_summary = _compact_text_block(item.get("raw_model_output"), limit=240)
        trace_sections.append(
            "\n".join(
                [
                f"Reasoning trace {index}",
                f"Task: {item.get('task_id')}",
                f"Record kind: {item.get('record_kind')} #{item.get('index')}",
                f"Observed supervised patterns: {pattern_summary}",
                f"Model output summary: {model_output_summary}",
                f"Parsed action: {json.dumps(item.get('parsed_action') or {}, sort_keys=True)}",
                f"Outcome: success={item.get('success')} terminal_status={item.get('terminal_status')}",
                ]
            )
        )
    user_prompt = _fit_label_proposal_prompt(
        header_sections=header_sections,
        trace_sections=trace_sections,
        word_budget=LABEL_PROPOSAL_INPUT_WORD_BUDGET,
    )
    return system_prompt, user_prompt


def _fit_label_proposal_prompt(
    *,
    header_sections: list[str],
    trace_sections: list[str],
    word_budget: int,
) -> str:
    sections = list(header_sections)
    current = "\n\n".join(sections)
    for trace_section in trace_sections:
        candidate = current + "\n\n" + trace_section
        if _word_count(candidate) <= word_budget:
            sections.append(trace_section)
            current = candidate
            continue
        available_words = max(0, word_budget - _word_count(current) - 8)
        if available_words <= 0:
            break
        trimmed = _truncate_to_word_budget(trace_section, available_words)
        if trimmed:
            sections.append(trimmed)
        break
    prompt = "\n\n".join(sections)
    return _truncate_to_word_budget(prompt, word_budget)


def _word_count(text: str) -> int:
    return len(text.split())


def _truncate_to_word_budget(text: str, word_budget: int) -> str:
    words = text.split()
    if len(words) <= word_budget:
        return text
    truncated = " ".join(words[:word_budget]).rstrip()
    if not truncated.endswith("…"):
        truncated += " …"
    return truncated


def _compact_text_block(value: Any, *, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    text = re.sub(r"\s+", " ", value).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _summarize_pattern_trace_labels(trace_description: Any) -> str:
    if not isinstance(trace_description, str) or not trace_description.strip():
        return "unknown"
    counts: dict[str, int] = {}
    for raw_line in trace_description.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^\[[^\]]+\]\s+([^(\n]+)", line)
        if match is None:
            continue
        label = match.group(1).strip()
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return _compact_text_block(trace_description, limit=240)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{label} x{count}" for label, count in ranked[:10])


def _propose_unsupervised_labels(
    *,
    env_id: str,
    sampled_reasoning_traces: list[dict[str, Any]],
    proposed_label_count: int,
    llm_model: str,
    llm_base_url: str,
    llm_timeout: int,
    artifact_root: Path | None = None,
) -> tuple[list[ProposedUnsupervisedLabel], str]:
    try:
        labels, response_text = _propose_unsupervised_labels_once(
            env_id=env_id,
            sampled_reasoning_traces=sampled_reasoning_traces,
            proposed_label_count=proposed_label_count,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_timeout=llm_timeout,
            artifact_root=artifact_root,
        )
        if labels:
            return labels, response_text
    except RuntimeError:
        pass

    return _propose_unsupervised_labels_batched(
        env_id=env_id,
        sampled_reasoning_traces=sampled_reasoning_traces,
        proposed_label_count=proposed_label_count,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_timeout=llm_timeout,
        artifact_root=artifact_root,
    )


def _propose_unsupervised_labels_once(
    *,
    env_id: str,
    sampled_reasoning_traces: list[dict[str, Any]],
    proposed_label_count: int,
    llm_model: str,
    llm_base_url: str,
    llm_timeout: int,
    artifact_root: Path | None = None,
    forbidden_labels: Iterable[str] = (),
    artifact_prefix: str = "label_proposal_response",
) -> tuple[list[ProposedUnsupervisedLabel], str]:
    system_prompt, user_prompt = _build_label_proposal_prompt(
        env_id=env_id,
        sampled_reasoning_traces=sampled_reasoning_traces,
        proposed_label_count=proposed_label_count,
        forbidden_labels=forbidden_labels,
    )
    last_error: Exception | None = None
    last_response_text: str | None = None
    last_finish_reason: str | None = None
    for attempt in range(1, LABEL_PROPOSAL_MAX_ATTEMPTS + 1):
        retry_suffix = ""
        assistant_prefill = ""
        if attempt > 1:
            retry_suffix = (
                "\n\nYour previous reply was invalid for automatic parsing.\n"
                f"Return exactly {proposed_label_count} label blocks.\n"
                "Each block must be exactly two lines:\n"
                "LABEL: <short label>\n"
                "DESCRIPTION: <one-sentence description of what the pattern is>\n"
                "The description must not explain how to detect the pattern.\n"
                "Do not include headings, analysis, numbering, bullets, code fences, or any extra text."
            )
            if last_finish_reason == "length":
                retry_suffix += "\nContinue directly with the required LABEL and DESCRIPTION lines only."
                assistant_prefill = LABEL_PROPOSAL_ASSISTANT_PREFILL
            elif last_response_text:
                retry_suffix += (
                    "\n\nPrevious invalid response:\n"
                    f"{last_response_text[:2000]}"
                )
                assistant_prefill = LABEL_PROPOSAL_ASSISTANT_PREFILL
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + retry_suffix},
        ]
        if assistant_prefill:
            messages.append({"role": "assistant", "content": assistant_prefill})
        body, response_text = request_chat_completion_with_text_retry(
            base_url=llm_base_url,
            model=llm_model,
            messages=messages,
            timeout=llm_timeout,
            headers={"Content-Type": "application/json"},
            max_tokens=LABEL_PROPOSAL_MAX_OUTPUT_TOKENS,
            temperature=0.0,
            stream=False,
            extra_payload={"chat_template_kwargs": {"enable_thinking": False}},
            error_prefix="Automatic label proposal response",
            request_fn=_request_json,
        )
        choices = body.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            finish_reason = choices[0].get("finish_reason")
            last_finish_reason = finish_reason if isinstance(finish_reason, str) else None
        else:
            last_finish_reason = None
        last_response_text = response_text
        if artifact_root is not None:
            artifact_root.mkdir(parents=True, exist_ok=True)
            (artifact_root / f"{artifact_prefix}_attempt_{attempt}.txt").write_text(
                response_text,
                encoding="utf-8",
            )
            (artifact_root / f"{artifact_prefix}.txt").write_text(
                response_text,
                encoding="utf-8",
            )
        try:
            labels = _parse_proposed_labels(response_text, expected_count=proposed_label_count)
            return labels, response_text
        except RuntimeError as exc:
            last_error = exc
            continue
    message = (
        f"Label proposal returned no usable labels after {LABEL_PROPOSAL_MAX_ATTEMPTS} attempts, "
        f"expected {proposed_label_count}."
    )
    if last_response_text:
        message += f" Last response excerpt: {last_response_text[:300]!r}"
    raise RuntimeError(message) from last_error


def _propose_unsupervised_labels_batched(
    *,
    env_id: str,
    sampled_reasoning_traces: list[dict[str, Any]],
    proposed_label_count: int,
    llm_model: str,
    llm_base_url: str,
    llm_timeout: int,
    artifact_root: Path | None = None,
) -> tuple[list[ProposedUnsupervisedLabel], str]:
    if not sampled_reasoning_traces:
        raise RuntimeError("No reasoning traces available for batched label proposal.")
    rng = random.Random(
        "|".join(str(item.get("task_id") or "") for item in sampled_reasoning_traces)
    )
    traces = list(sampled_reasoning_traces)
    rng.shuffle(traces)
    collected: list[ProposedUnsupervisedLabel] = []
    seen: set[str] = set()
    response_chunks: list[str] = []
    for batch_index, start in enumerate(range(0, len(traces), LABEL_PROPOSAL_BATCH_TRACE_COUNT), start=1):
        if len(collected) >= proposed_label_count:
            break
        batch = traces[start : start + LABEL_PROPOSAL_BATCH_TRACE_COUNT]
        request_count = min(
            LABEL_PROPOSAL_BATCH_LABEL_COUNT,
            proposed_label_count - len(collected),
        )
        if request_count <= 0:
            break
        labels, response_text = _propose_unsupervised_labels_once(
            env_id=env_id,
            sampled_reasoning_traces=batch,
            proposed_label_count=request_count,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_timeout=llm_timeout,
            artifact_root=artifact_root,
            forbidden_labels=[item.label for item in collected],
            artifact_prefix=f"label_proposal_batch_{batch_index}",
        )
        response_chunks.append(response_text)
        for label in labels:
            normalized = label.label.strip().lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            collected.append(label)
            if len(collected) >= proposed_label_count:
                break
    if len(collected) < proposed_label_count:
        raise RuntimeError(
            "Batched label proposal returned only "
            f"{len(collected)} usable labels, expected {proposed_label_count}"
        )
    combined_response = "\n\n".join(response_chunks).strip()
    if artifact_root is not None and combined_response:
        (artifact_root / "label_proposal_response_batched.txt").write_text(
            combined_response,
            encoding="utf-8",
        )
        (artifact_root / "label_proposal_response.txt").write_text(
            combined_response,
            encoding="utf-8",
        )
    return collected, combined_response


def _parse_proposed_labels(
    response_text: str, *, expected_count: int, allow_partial: bool = False
) -> list[ProposedUnsupervisedLabel]:
    labels: list[ProposedUnsupervisedLabel] = []
    seen: set[str] = set()
    for candidate in _parse_proposed_labels_from_json(response_text):
        label = _normalize_proposed_label(candidate.get("label"))
        description = _normalize_proposed_label_description(
            candidate.get("description")
        )
        if not _is_valid_proposed_label(label) or not _is_valid_proposed_description(description):
            continue
        normalized_key = label.lower()
        if normalized_key in seen:
            continue
        seen.add(normalized_key)
        labels.append(
            ProposedUnsupervisedLabel(label=label, description=description)
        )
        if len(labels) == expected_count:
            return labels
    pending_label: str | None = None
    for raw_line in response_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", stripped)
        stripped = stripped.strip(" \t\"'")
        if not stripped:
            continue
        match = re.match(r"(?i)^label\s*:\s*(.+)$", stripped)
        if match is not None:
            candidate = _normalize_proposed_label(match.group(1))
            pending_label = candidate if _is_valid_proposed_label(candidate) else None
            continue
        description_match = re.match(r"(?i)^description\s*:\s*(.+)$", stripped)
        if description_match is not None and pending_label is not None:
            description = _normalize_proposed_label_description(
                description_match.group(1)
            )
            if not _is_valid_proposed_description(description):
                pending_label = None
                continue
            normalized_key = pending_label.lower()
            if normalized_key not in seen:
                seen.add(normalized_key)
                labels.append(
                    ProposedUnsupervisedLabel(
                        label=pending_label,
                        description=description,
                    )
                )
                if len(labels) == expected_count:
                    break
            pending_label = None
    if len(labels) < expected_count and not allow_partial:
        raise RuntimeError(
            f"Label proposal returned only {len(labels)} usable labels, expected {expected_count}"
        )
    return labels


def _parse_proposed_labels_from_json(response_text: str) -> list[dict[str, str]]:
    stripped = response_text.strip()
    if not stripped:
        return []
    candidates = [stripped]
    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match is not None:
        candidates.insert(0, fenced_match.group(1).strip())
    for payload_text in candidates:
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            values: list[dict[str, str]] = []
            for item in payload:
                if isinstance(item, dict):
                    label = _normalize_proposed_label(item.get("label"))
                    description = _normalize_proposed_label_description(
                        item.get("description")
                    )
                    if label and description:
                        values.append({"label": label, "description": description})
            return values
        if isinstance(payload, dict):
            labels = payload.get("labels")
            if isinstance(labels, list):
                values: list[dict[str, str]] = []
                for item in labels:
                    if isinstance(item, dict):
                        label = _normalize_proposed_label(item.get("label"))
                        description = _normalize_proposed_label_description(
                            item.get("description")
                        )
                        if label and description:
                            values.append({"label": label, "description": description})
                return values
    return []


def _is_valid_proposed_label(value: str) -> bool:
    candidate = value.strip()
    if not candidate:
        return False
    if candidate.endswith(":"):
        return False
    if any(marker in candidate for marker in ("```", "{", "}", "[", "]")):
        return False
    normalized = candidate.lower()
    if normalized in {
        "thinking process",
        "reasoning",
        "analysis",
        "proposed labels",
        "labels",
    }:
        return False
    words = [word for word in re.split(r"\s+", candidate) if word]
    if not 1 <= len(words) <= 5:
        return False
    if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9 _'/-]*", candidate):
        return False
    return True


def _normalize_proposed_label(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip(" \t\"'")


def _normalize_proposed_label_description(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip(" \t\"'")


def _is_valid_proposed_description(value: str) -> bool:
    candidate = value.strip()
    if not candidate:
        return False
    lowered = candidate.lower()
    if lowered.startswith(("thinking process", "analysis", "reasoning", "label:", "description:")):
        return False
    if "how to detect" in lowered or "detect " in lowered or "detector" in lowered:
        return False
    return 3 <= len(candidate.split()) <= 40


def _load_proposed_label_details(path: Path) -> list[ProposedUnsupervisedLabel]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("labels") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        raise ValueError(f"Malformed proposed label details payload: {path}")
    details: list[ProposedUnsupervisedLabel] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        label = _normalize_proposed_label(item.get("label"))
        description = _normalize_proposed_label_description(item.get("description"))
        if not label:
            continue
        details.append(ProposedUnsupervisedLabel(label=label, description=description))
    if not details:
        raise ValueError(f"No proposed labels found in {path}")
    return details


def _build_multimodal_user_content(
    *,
    text: str,
    image_items: list[tuple[str, Path]],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for index, (label, image_path) in enumerate(image_items, start=1):
        content.append({"type": "text", "text": f"{label or f'Image {index}'}."})
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
        image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
            }
        )
    return content


def _sample_unsupervised_traces(
    *,
    samples: list[Any],
    trace_count: int,
    seed: int | None,
) -> list[Any]:
    if len(samples) < trace_count:
        raise RuntimeError(
            f"Requested {trace_count} traces for unsupervised label proposal, but only {len(samples)} dataset samples were available"
        )
    ordered = sorted(
        samples,
        key=lambda item: (
            str(getattr(item, "task_id", "") or ""),
            str(getattr(item, "trace_path", "") or ""),
        ),
    )
    rng = random.Random(seed if seed is not None else 0)
    rng.shuffle(ordered)
    return ordered[:trace_count]


def _resize_image_in_place(path: Path, *, max_side: int = 512) -> None:
    with Image.open(path) as image:
        width, height = image.size
        longest = max(width, height)
        if longest <= max_side:
            image.save(path)
            return
        scale = max_side / float(longest)
        resized = image.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            Image.Resampling.LANCZOS,
        )
        resized.save(path)


def _extract_video_frame_paths(
    video_path: Path,
    *,
    output_dir: Path,
    frame_count: int,
) -> list[Path]:
    if frame_count <= 0 or not video_path.is_file():
        return []
    reader = imageio.get_reader(video_path)
    try:
        total_frames = reader.count_frames()
    except Exception:
        total_frames = None
    indices: list[int]
    if isinstance(total_frames, int) and total_frames > 0:
        if total_frames <= frame_count:
            indices = list(range(total_frames))
        else:
            indices = sorted(
                {
                    min(
                        total_frames - 1,
                        max(
                            0,
                            int(round(position * (total_frames - 1) / max(frame_count - 1, 1))),
                        ),
                    )
                    for position in range(frame_count)
                }
            )
    else:
        indices = list(range(frame_count))
    paths: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for out_index, frame_index in enumerate(indices, start=1):
            try:
                frame = reader.get_data(frame_index)
            except Exception:
                continue
            frame_path = output_dir / f"frame_{out_index:02d}.png"
            Image.fromarray(frame).save(frame_path)
            _resize_image_in_place(frame_path)
            paths.append(frame_path)
    finally:
        reader.close()
    return paths


def _trace_prompt_images(
    *,
    sample: Any,
    output_root: Path,
    frame_count: int = TRUE_UNSUPERVISED_FRAME_COUNT,
) -> list[tuple[str, Path]]:
    trace_path = Path(getattr(sample, "trace_path"))
    sample_root = trace_path.parent
    output_root.mkdir(parents=True, exist_ok=True)
    image_items: list[tuple[str, Path]] = []
    initial_candidates = sorted(sample_root.glob("initial_scene*.png"))
    for path in initial_candidates[:2]:
        copied = output_root / path.name
        copied.write_bytes(path.read_bytes())
        _resize_image_in_place(copied)
        label = "Initial scene" if len(image_items) == 0 else f"Initial scene view {len(image_items) + 1}"
        image_items.append((label, copied))
    simulation_candidates = sorted(sample_root.glob("simulation*.mp4"))
    if simulation_candidates:
        frame_dir = output_root / "frames"
        for index, frame_path in enumerate(
            _extract_video_frame_paths(
                simulation_candidates[0],
                output_dir=frame_dir,
                frame_count=frame_count,
            ),
            start=1,
        ):
            image_items.append((f"Trace frame {index}", frame_path))
    if not image_items:
        raise RuntimeError(f"No usable trace images found for {sample_root}")
    return image_items


def _task_spec_summary(env_id: str) -> str:
    adapter = AdapterRegistry().load(env_id)
    task_spec = adapter.get_task_spec()
    parts = [
        f"Environment: {task_spec.display_name} ({task_spec.env_id})",
        f"Execution mode: {task_spec.execution_mode}",
        f"Observation: {task_spec.observation.kind} / {task_spec.observation.format}",
        f"Action space: {task_spec.action_space.kind} / {task_spec.action_space.format}",
        f"Task metric: {task_spec.native_metric.name}",
    ]
    if task_spec.observation.description:
        parts.append(f"Observation details: {task_spec.observation.description}")
    if task_spec.action_space.description:
        parts.append(f"Action details: {task_spec.action_space.description}")
    if task_spec.notes:
        parts.append("Environment notes: " + "; ".join(task_spec.notes))
    return "\n".join(parts)


def _build_trace_label_proposal_prompt(
    *,
    env_id: str,
    sample: Any,
    labels_per_trace: int,
    forbidden_labels: Iterable[str],
) -> tuple[str, str]:
    trace_path = Path(getattr(sample, "trace_path"))
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    task_id = trace_payload.get("task_id") or getattr(sample, "task_id", "") or "unknown"
    outcome = trace_payload.get("outcome") if isinstance(trace_payload, dict) else {}
    system_prompt = (
        "You are proposing low-level unsupervised pattern labels for a physics-simulation library.\n"
        f"Return exactly {labels_per_trace} label blocks.\n"
        "Each block must be exactly two lines:\n"
        "LABEL: <short label>\n"
        "DESCRIPTION: <one-sentence description of the recurring visual-physical pattern>\n"
        "Each label must be short, a few words at most, and describe a reusable low-level physical pattern.\n"
        "The description must explain what the pattern means conceptually, not how to detect it.\n"
        "Base your suggestions only on the environment details and the observed trace images.\n"
        "Do not include headings, bullets, numbering, code fences, or extra commentary."
    )
    blocked = [item.strip() for item in forbidden_labels if isinstance(item, str) and item.strip()]
    sections = [
        _task_spec_summary(env_id),
        f"Trace task id: {task_id}",
        f"Trace success: {outcome.get('success')}",
        f"Trace terminal status: {outcome.get('terminal_status')}",
        f"Trace timestep count: {len(trace_payload.get('timesteps') or [])}",
        f"Propose exactly {labels_per_trace} new labels from this single observed trace.",
        "The labels should capture recurring physical events, interactions, object states, or spatial relations that could generalize across traces.",
    ]
    if blocked:
        sections.append("Do not propose any of these labels: " + ", ".join(sorted(blocked)))
    user_prompt = "\n\n".join(sections)
    return system_prompt, user_prompt


def _propose_trace_labels(
    *,
    env_id: str,
    sample: Any,
    labels_per_trace: int,
    llm_model: str,
    llm_base_url: str,
    llm_timeout: int,
    forbidden_labels: Iterable[str],
    artifact_root: Path,
    trace_index: int,
    require_full_count: bool = True,
) -> tuple[list[ProposedUnsupervisedLabel], str, list[str]]:
    prompt_dir = artifact_root / f"trace_{trace_index:02d}"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    image_items = _trace_prompt_images(sample=sample, output_root=prompt_dir / "prompt_media")
    accumulated_labels: list[ProposedUnsupervisedLabel] = []
    seen_labels = {
        item.strip().lower()
        for item in forbidden_labels
        if isinstance(item, str) and item.strip()
    }
    response_texts: list[str] = []
    request_payloads: list[dict[str, Any]] = []
    max_attempts = max(8, labels_per_trace)
    for attempt in range(1, max_attempts + 1):
        remaining = labels_per_trace - len(accumulated_labels)
        if remaining <= 0:
            break
        system_prompt, user_prompt = _build_trace_label_proposal_prompt(
            env_id=env_id,
            sample=sample,
            labels_per_trace=remaining,
            forbidden_labels=sorted(seen_labels),
        )
        if attempt > 1:
            user_prompt += (
                "\n\nThe previous response did not yield enough usable labels after validation. "
                f"Return exactly {remaining} additional valid labels that are distinct from every blocked label."
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": _build_multimodal_user_content(text=user_prompt, image_items=image_items),
            },
        ]
        try:
            body, response_text = request_chat_completion_with_text_retry(
                base_url=llm_base_url,
                model=llm_model,
                messages=messages,
                timeout=llm_timeout,
                headers={"Content-Type": "application/json"},
                max_tokens=TRUE_UNSUPERVISED_MAX_OUTPUT_TOKENS,
                temperature=0.0,
                stream=False,
                extra_payload={"chat_template_kwargs": {"enable_thinking": False}},
                error_prefix="Unsupervised trace label proposal response",
                request_fn=_request_json,
            )
        except RuntimeError as exc:
            progress_log(
                "dashboard.workflows.unsupervised.trace_labels_retryable_error",
                env=env_id,
                trace_index=trace_index,
                attempt=attempt,
                error=str(exc),
            )
            request_payloads.append(
                {
                    "attempt": attempt,
                    "task_id": getattr(sample, "task_id", None),
                    "trace_path": str(getattr(sample, "trace_path", "")),
                    "messages": messages,
                    "error": str(exc),
                }
            )
            response_texts.append(f"ERROR: {exc}")
            continue
        response_texts.append(response_text)
        request_payloads.append(
            {
                "attempt": attempt,
                "task_id": getattr(sample, "task_id", None),
                "trace_path": str(getattr(sample, "trace_path", "")),
                "messages": messages,
                "response_body": body,
            }
        )
        partial_labels = _parse_proposed_labels(
            response_text,
            expected_count=remaining,
            allow_partial=True,
        )
        for label in partial_labels:
            normalized_key = label.label.strip().lower()
            if normalized_key in seen_labels:
                continue
            seen_labels.add(normalized_key)
            accumulated_labels.append(label)
            if len(accumulated_labels) >= labels_per_trace:
                break
    combined_response = "\n\n=== RETRY ===\n\n".join(response_texts)
    (prompt_dir / "response.txt").write_text(combined_response, encoding="utf-8")
    (prompt_dir / "request.json").write_text(
        json.dumps(request_payloads, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    if len(accumulated_labels) < labels_per_trace and require_full_count:
        raise RuntimeError(
            "Label proposal returned only "
            f"{len(accumulated_labels)} usable labels after {max_attempts} retries, expected {labels_per_trace}"
        )
    labels = accumulated_labels[:labels_per_trace]
    return labels, combined_response, [str(path) for _, path in image_items]


def _select_final_unsupervised_labels(
    *,
    env_id: str,
    candidate_labels: list[dict[str, Any]],
    final_label_count: int,
    llm_model: str,
    llm_base_url: str,
    llm_timeout: int,
    artifact_root: Path,
) -> tuple[list[ProposedUnsupervisedLabel], str]:
    unique_candidates: list[ProposedUnsupervisedLabel] = []
    seen_candidate_labels: set[str] = set()
    for item in candidate_labels:
        if not isinstance(item, dict):
            continue
        label = _normalize_proposed_label(item.get("label"))
        description = _normalize_proposed_label_description(item.get("description"))
        if not _is_valid_proposed_label(label) or not _is_valid_proposed_description(description):
            continue
        normalized_key = label.lower()
        if normalized_key in seen_candidate_labels:
            continue
        seen_candidate_labels.add(normalized_key)
        unique_candidates.append(
            ProposedUnsupervisedLabel(label=label, description=description)
        )
    if len(unique_candidates) < final_label_count:
        raise RuntimeError(
            "Unsupervised final label selection received only "
            f"{len(unique_candidates)} unique usable candidates, expected at least {final_label_count}"
        )
    system_prompt = (
        "You are selecting the best low-level unsupervised pattern labels for a physics-simulation library.\n"
        f"Choose exactly {final_label_count} labels from the provided candidates.\n"
        "Return exactly the chosen labels using this format:\n"
        "LABEL: <short label>\n"
        "DESCRIPTION: <one-sentence description>\n"
        "Do not invent new labels. Do not rename labels. Do not include commentary."
    )
    lines = [
        _task_spec_summary(env_id),
        f"Choose the best {final_label_count} labels from these candidates.",
        "Prefer labels that are low-level, visually grounded, physically meaningful, distinct from one another, and likely to transfer across many tasks.",
    ]
    for index, item in enumerate(candidate_labels, start=1):
        lines.append(
            f"{index}. LABEL: {item.get('label')}\nDESCRIPTION: {item.get('description')}\nSOURCE_TRACE: {item.get('trace_index')}"
        )
    user_prompt = "\n\n".join(lines)
    try:
        body, response_text = request_chat_completion_with_text_retry(
            base_url=llm_base_url,
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=llm_timeout,
            headers={"Content-Type": "application/json"},
            max_tokens=TRUE_UNSUPERVISED_MAX_OUTPUT_TOKENS,
            temperature=0.0,
            stream=False,
            extra_payload={"chat_template_kwargs": {"enable_thinking": False}},
            error_prefix="Unsupervised final label selection response",
            request_fn=_request_json,
        )
    except RuntimeError as exc:
        progress_log(
            "dashboard.workflows.unsupervised.final_label_selection_fallback",
            env=env_id,
            candidate_label_count=len(unique_candidates),
            error=str(exc),
        )
        labels = unique_candidates[:final_label_count]
        response_text = (
            "FALLBACK: selected first unique candidates after transient router failure.\n"
            + "\n".join(f"LABEL: {item.label}\nDESCRIPTION: {item.description}" for item in labels)
        )
        body = {
            "fallback": True,
            "error": str(exc),
            "selected_labels": [
                {"label": item.label, "description": item.description} for item in labels
            ],
        }
        (artifact_root / "final_label_selection_response.txt").write_text(
            response_text,
            encoding="utf-8",
        )
        (artifact_root / "final_label_selection_request.json").write_text(
            json.dumps(
                {"candidate_labels": candidate_labels, "response_body": body},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return labels, response_text
    labels = _parse_proposed_labels(
        response_text,
        expected_count=final_label_count,
        allow_partial=True,
    )
    selected_label_keys = {item.label.strip().lower() for item in labels}
    if len(labels) < final_label_count:
        for candidate in unique_candidates:
            normalized_key = candidate.label.strip().lower()
            if normalized_key in selected_label_keys:
                continue
            labels.append(candidate)
            selected_label_keys.add(normalized_key)
            if len(labels) >= final_label_count:
                break
    if len(labels) < final_label_count:
        raise RuntimeError(
            "Unsupervised final label selection returned only "
            f"{len(labels)} usable labels after fallback, expected {final_label_count}"
        )
    (artifact_root / "final_label_selection_response.txt").write_text(
        response_text,
        encoding="utf-8",
    )
    (artifact_root / "final_label_selection_request.json").write_text(
        json.dumps(
            {"candidate_labels": candidate_labels, "response_body": body},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return labels, response_text


def _dataset_manifest_from_supervised_run(source_run_dir: Path) -> Path:
    run_payload = _load_run_payload(source_run_dir)
    scorer_config = run_payload.get("scorer_config")
    if not isinstance(scorer_config, dict):
        raise ValueError(
            f"Supervised run does not include scorer_config in run.json: {source_run_dir / 'run.json'}"
        )
    dataset_manifest = scorer_config.get("dataset_manifest")
    if not isinstance(dataset_manifest, str) or not dataset_manifest.strip():
        raise ValueError(
            f"Supervised run does not record dataset_manifest in scorer_config: {source_run_dir / 'run.json'}"
        )
    manifest_path = Path(dataset_manifest)
    if manifest_path.is_file():
        return manifest_path

    env_ids = run_payload.get("environment_ids")
    env_id = ""
    if isinstance(env_ids, list) and env_ids:
        env_id = str(env_ids[0] or "").strip().lower()
    if not env_id:
        env_id = str(run_payload.get("env_id") or "").strip().lower()
    if not env_id:
        env_id = source_run_dir.parent.name.strip().lower()

    fallback_candidates: list[Path] = []
    if env_id:
        fallback_candidates.extend(
            [
                Path(f"data/datasets/{env_id}/default_simulator_trace_dataset/dataset_manifest.json"),
                get_artifact_layout().project_root / "data" / "datasets" / env_id / "default_simulator_trace_dataset" / "dataset_manifest.json",
            ]
        )

    for candidate in fallback_candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")


def _combine_libraries(
    *,
    env_id: str,
    source_library: PatternLibrary,
    unsupervised_library: PatternLibrary,
    run_id: str,
) -> PatternLibrary:
    return PatternLibrary(
        library_id=f"{env_id}-{run_id}-combined",
        environment_ids=[env_id],
        primitive_source=source_library.primitive_source,
        scorer_config=dict(unsupervised_library.scorer_config),
        detectors=[*source_library.detectors, *unsupervised_library.detectors],
        metadata={
            "run_kind": "automatic",
            "source_supervised_library_id": source_library.library_id,
            "unsupervised_library_id": unsupervised_library.library_id,
            "source_supervised_detector_count": len(source_library.detectors),
            "unsupervised_detector_count": len(unsupervised_library.detectors),
        },
    )


def supervised_learn_and_evaluate(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dashboard.workflows supervised-learn-and-evaluate",
        description=(
            "Run supervised pattern learning via `-m cli pattern-learn` and stop after "
            "learning completes."
        ),
    )
    parser.add_argument("--env", required=True, help="Environment id (e.g. phyre)")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--labels-file", type=Path, default=None)
    parser.add_argument("--primitive-label", action="append", default=[])
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, default=None)
    parser.add_argument("--train-sample-count", type=int, default=None)
    parser.add_argument("--validation-sample-count", type=int, default=None)
    parser.add_argument("--model", default="")
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--validation-split", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--iteration-count", type=int, default=8)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--patience-factor", type=float, default=0.35)
    parser.add_argument("--elite-fraction", type=float, default=None)
    parser.add_argument("--children-per-elite", type=int, default=None)
    parser.add_argument("--crossover-rate", type=float, default=None)
    parser.add_argument("--primitive-source", default=None)
    parser.add_argument("--defer-evaluation", action="store_true")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args(argv)
    run_dir = _derive_run_dir_from_output_path(args.output_path)
    progress_log(
        "dashboard.workflows.supervised.start",
        env=args.env,
        output_path=args.output_path,
        dataset_manifest=args.dataset_manifest,
        train_sample_count=args.train_sample_count,
        validation_sample_count=args.validation_sample_count,
        iteration_count=args.iteration_count,
        workers=args.workers,
        resume=args.resume,
    )

    effective_dataset_manifest = args.dataset_manifest
    if (
        effective_dataset_manifest is not None
        and (args.train_sample_count is not None or args.validation_sample_count is not None)
    ):
        runtime_manifest = run_dir.parent / f"{run_dir.name}.runtime_dataset_manifest.json"
        effective_dataset_manifest = _write_runtime_dataset_manifest(
            source_manifest=effective_dataset_manifest,
            target_manifest=runtime_manifest,
            train_sample_count=args.train_sample_count,
            validation_sample_count=args.validation_sample_count,
        )
        progress_log(
            "dashboard.workflows.supervised.runtime_manifest",
            source_manifest=args.dataset_manifest,
            runtime_manifest=effective_dataset_manifest,
            train_sample_count=args.train_sample_count,
            validation_sample_count=args.validation_sample_count,
        )

    learn_argv: list[str] = [
        sys.executable,
        "-m",
        "cli",
        "pattern-learn",
        "--env",
        args.env,
    ]
    if args.config is not None:
        learn_argv.extend(["--config", str(args.config)])
    if args.labels_file is not None:
        learn_argv.extend(["--labels-file", str(args.labels_file)])
    for primitive_label in args.primitive_label:
        if primitive_label:
            learn_argv.extend(["--primitive-label", primitive_label])
    learn_argv.extend(["--output-path", str(args.output_path)])
    if effective_dataset_manifest is not None:
        learn_argv.extend(["--dataset-manifest", str(effective_dataset_manifest)])
    learn_argv.extend(["--generator-model", args.model])
    for split_name in args.split:
        if split_name:
            learn_argv.extend(["--split", split_name])
    for split_name in args.validation_split:
        if split_name:
            learn_argv.extend(["--validation-split", split_name])
    if args.limit is not None:
        learn_argv.extend(["--limit", str(args.limit)])
    if args.iteration_count != 8:
        learn_argv.extend(["--iteration-count", str(args.iteration_count)])
    if args.workers is not None:
        learn_argv.extend(["--workers", str(args.workers)])
    if args.seed is not None:
        learn_argv.extend(["--seed", str(args.seed)])
    if args.population_size is not None:
        learn_argv.extend(["--population-size", str(args.population_size)])
    if args.elite_fraction is not None:
        learn_argv.extend(["--elite-fraction", str(args.elite_fraction)])
    if args.children_per_elite is not None:
        learn_argv.extend(["--children-per-elite", str(args.children_per_elite)])
    if args.patience_factor is not None and args.patience_factor != 0.35:
        learn_argv.extend(["--patience-factor", str(args.patience_factor)])
    if args.crossover_rate is not None:
        learn_argv.extend(["--crossover-rate", str(args.crossover_rate)])
    if args.primitive_source is not None:
        learn_argv.extend(["--primitive-source", args.primitive_source])

    if args.resume:
        learn_argv.append("--resume")
    learning_already_complete = False
    if args.resume and (run_dir / PATTERN_LIBRARY_FILENAME).is_file():
        learning_already_complete = _resume_can_skip_supervised_learning(
            env_id=args.env,
            run_dir=run_dir,
        )
        if learning_already_complete:
            progress_log(
                "dashboard.workflows.supervised.learning_resumed",
                env=args.env,
                run_dir=run_dir,
            )
    if not learning_already_complete:
        learn_code = _run_child(learn_argv)
        if learn_code != 0:
            return learn_code

    progress_log(
        "dashboard.workflows.supervised.learning_done",
        env=args.env,
        run_dir=run_dir,
    )
    if args.defer_evaluation:
        run_payload = _load_run_payload(run_dir)
        run_payload["defer_evaluation"] = True
        run_payload["evaluation_strategy"] = "scheduled"
        _write_json(run_dir / "run.json", run_payload)
        progress_log(
            "dashboard.workflows.supervised.evaluation_deferred",
            env=args.env,
            run_dir=run_dir,
        )
        return 0
    return _run_supervised_auto_evaluation(
        env_id=args.env,
        run_dir=run_dir,
        model_name=args.model,
    )


def automatic_launch(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dashboard.workflows automatic-launch",
        description=(
            "Learn an automatic pattern library on top of a supervised library by "
            "proposing labels from DeepPHY reasoning traces, then evaluate the combined library."
        ),
    )
    parser.add_argument("--env", required=True, help="Environment id (e.g. phyre)")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source-supervised-run-dir",
        type=Path,
        default=None,
        help="Path to the supervised run directory containing pattern_library.json",
    )
    source_group.add_argument(
        "--source-pattern-library",
        type=Path,
        default=None,
        help="Path to the supervised pattern_library.json (or its containing run directory).",
    )
    parser.add_argument(
        "--source-evaluation-run-id",
        required=True,
        help="Supervised DeepPHY evaluation run id used to source reasoning traces.",
    )
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--reasoning-trace-count",
        type=int,
        default=6,
        help="Number of reasoning traces to sample from the supervised evaluation results.",
    )
    parser.add_argument(
        "--proposed-label-count",
        type=int,
        default=5,
        help="Number of automatic labels to request from the label-proposal LLM.",
    )
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--validation-split", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--iteration-count", type=int, default=8)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--patience-factor", type=float, default=0.35)
    parser.add_argument("--elite-fraction", type=float, default=None)
    parser.add_argument("--children-per-elite", type=int, default=None)
    parser.add_argument("--crossover-rate", type=float, default=None)
    parser.add_argument(
        "--run-timestamp",
        default=None,
        help="Optional timestamp override used to name the automatic run record directory.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    progress_log(
        "dashboard.workflows.automatic.start",
        env=args.env,
        source_evaluation_run_id=args.source_evaluation_run_id,
        iteration_count=args.iteration_count,
        workers=args.workers,
        resume=args.resume,
    )

    if args.source_supervised_run_dir is not None:
        source_run_dir = resolve_pattern_run_dir(args.source_supervised_run_dir)
        if not source_run_dir.is_dir():
            raise FileNotFoundError(f"Supervised run directory not found: {source_run_dir}")
        source_library_path = source_run_dir / PATTERN_LIBRARY_FILENAME
    else:
        assert args.source_pattern_library is not None
        source_library_path = resolve_pattern_library_path(args.source_pattern_library)
        source_run_dir = source_library_path.parent

    resolved_source_evaluation_run_id = _resolve_source_evaluation_run_id(
        source_run_dir=source_run_dir,
        requested_run_id=args.source_evaluation_run_id,
    )
    results_path = _results_path_for_evaluation_run(
        source_run_dir=source_run_dir,
        source_evaluation_run_id=resolved_source_evaluation_run_id,
    )
    reasoning_records = _load_reasoning_trace_records(results_path)
    try:
        sampled_reasoning_traces = _sample_reasoning_traces(
            reasoning_records,
            reasoning_trace_count=args.reasoning_trace_count,
        )
    except RuntimeError as exc:
        raise FileNotFoundError(
            "No usable reasoning traces were found in the supervised DeepPHY evaluation "
            f"results at {results_path}. Rerun the supervised DeepPHY evaluation with "
            "successful trace records before launching the automatic workflow."
        ) from exc
    progress_log(
        "dashboard.workflows.automatic.reasoning_traces_loaded",
        env=args.env,
        reasoning_trace_count=len(sampled_reasoning_traces),
    )

    run_root = default_automatic_run_dir(args.env, timestamp=args.run_timestamp)
    if args.resume:
        if not run_root.is_dir():
            run_root.mkdir(parents=True, exist_ok=False)
            progress_log(
                "dashboard.workflows.automatic.resume_fallback_fresh_start",
                env=args.env,
                run_dir=run_root,
            )
    else:
        if run_root.is_dir():
            progress_log(
                "dashboard.workflows.automatic.fresh_start_reusing_existing_dir",
                env=args.env,
                run_dir=run_root,
            )
        else:
            run_root.mkdir(parents=True, exist_ok=False)

    layout = get_artifact_layout()
    llm = load_shared_llm_config(layout.config_root / "LLM.yaml")
    proposed_labels_path = run_root / "proposed_labels.txt"
    proposed_label_details_path = run_root / "proposed_labels.json"
    sampled_reasoning_traces_path = run_root / "sampled_reasoning_traces.json"
    proposal_response_path = run_root / "label_proposal_response.txt"
    _write_json(
        sampled_reasoning_traces_path,
        {"records": sampled_reasoning_traces},
    )
    if (
        args.resume
        and proposed_labels_path.is_file()
        and sampled_reasoning_traces_path.is_file()
        and proposal_response_path.is_file()
    ):
        if proposed_label_details_path.is_file():
            proposed_label_details = _load_proposed_label_details(proposed_label_details_path)
            proposed_labels = [item.label for item in proposed_label_details]
        else:
            proposed_labels = _normalize_labels(
                proposed_labels_path.read_text(encoding="utf-8").splitlines()
            )
            proposed_label_details = [
                ProposedUnsupervisedLabel(label=label, description="")
                for label in proposed_labels
            ]
            _write_json(
                proposed_label_details_path,
                {
                    "labels": [
                        {"label": item.label, "description": item.description}
                        for item in proposed_label_details
                    ]
                },
            )
        sampled_payload = json.loads(
            sampled_reasoning_traces_path.read_text(encoding="utf-8")
        )
        sampled_reasoning_traces = list(sampled_payload.get("records") or [])
        proposal_response = proposal_response_path.read_text(encoding="utf-8")
        progress_log(
            "dashboard.workflows.automatic.labels_resumed",
            env=args.env,
            proposed_label_count=len(proposed_labels),
        )
    else:
        proposed_label_details, proposal_response = _propose_unsupervised_labels(
            env_id=args.env,
            sampled_reasoning_traces=sampled_reasoning_traces,
            proposed_label_count=args.proposed_label_count,
            llm_model=args.model,
            llm_base_url=llm.base_url,
            llm_timeout=llm.timeout,
            artifact_root=run_root,
        )
        proposed_labels = [item.label for item in proposed_label_details]
        progress_log(
            "dashboard.workflows.automatic.labels_proposed",
            env=args.env,
            proposed_label_count=len(proposed_labels),
        )
        _write_json(
            proposed_label_details_path,
            {
                "labels": [
                    {"label": item.label, "description": item.description}
                    for item in proposed_label_details
                ]
            },
        )
        proposed_labels_path.write_text(
            "".join(f"{label}\n" for label in proposed_labels), encoding="utf-8"
        )
        proposal_response_path.write_text(proposal_response, encoding="utf-8")
    if "proposed_label_details" not in locals():
        if proposed_label_details_path.is_file():
            proposed_label_details = _load_proposed_label_details(proposed_label_details_path)
        else:
            proposed_label_details = [
                ProposedUnsupervisedLabel(label=label, description="")
                for label in proposed_labels
            ]
    label_descriptions = {
        item.label: item.description
        for item in proposed_label_details
        if item.description
    }

    supervised_library = load_pattern_library(source_library_path)
    dataset_manifest = _dataset_manifest_from_supervised_run(source_run_dir)
    learning_config = load_pattern_learning_config(layout.config_root / "pattern_learning.yaml")
    label_workers = (
        args.workers if args.workers is not None else learning_config.workers
    )
    active_losses = list(learning_config.losses or default_loss_configs())
    samples, dataset_meta = load_learning_samples_from_dataset_manifest(
        dataset_manifest,
        split_names=args.split or list(learning_config.split) or None,
        limit=args.limit if args.limit is not None else learning_config.limit,
    )
    if not samples:
        raise RuntimeError("Automatic learning did not find any dataset samples to train on")
    validation_samples, _ = load_learning_samples_from_dataset_manifest(
        dataset_manifest,
        split_names=args.validation_split or list(learning_config.validation_split) or None,
        limit=args.limit if args.limit is not None else learning_config.limit,
    )
    if not validation_samples:
        raise RuntimeError(
            "Automatic learning did not find any validation dataset samples"
        )

    unsupervised_library_path = run_root / "unsupervised_library.json"
    learning_already_complete = False
    if args.resume and unsupervised_library_path.is_file():
        _require_complete_labels_for_expected(
            run_dir=run_root, expected_labels=proposed_labels
        )
        _require_current_library_labels(
            library_path=unsupervised_library_path,
            expected_labels=proposed_labels,
        )
        learning_already_complete = True
        progress_log(
            "dashboard.workflows.automatic.learning_resumed",
            env=args.env,
            run_dir=run_root,
        )
    if not learning_already_complete:
        run_pattern_learning(
            environment_id=args.env,
            primitive_labels=proposed_labels,
            label_descriptions=label_descriptions,
            samples=samples,
            validation_samples=validation_samples,
            output_path=unsupervised_library_path,
            scorer_config={
                "dataset_manifest": str(dataset_manifest),
                "train_split": list(args.split or list(learning_config.split)),
                "validation_split": list(
                    args.validation_split or list(learning_config.validation_split)
                ),
                "generator_model": args.model,
                "generator_base_url": learning_config.generator_base_url,
                "generator_timeout": learning_config.generator_timeout,
                "generator_retries": learning_config.generator_retries,
                "losses": [loss.model_dump(mode="json") for loss in active_losses],
                "config_path": str(layout.config_root / "pattern_learning.yaml"),
            },
            iteration_count=args.iteration_count or learning_config.iteration_count,
            label_workers=label_workers,
            primitive_source=str(run_root / "proposed_labels.txt"),
            dataset_id=dataset_meta.get("dataset_id"),
            dataset_version=dataset_meta.get("dataset_version"),
            generator_model=args.model,
            generator_base_url=learning_config.generator_base_url,
            generator_timeout=learning_config.generator_timeout,
            generator_retries=learning_config.generator_retries,
            seed=args.seed if args.seed is not None else learning_config.seed,
            population_size=args.population_size
            if args.population_size is not None
            else learning_config.population_size,
            elite_fraction=args.elite_fraction
            if args.elite_fraction is not None
            else learning_config.elite_fraction,
            children_per_elite=args.children_per_elite
            if args.children_per_elite is not None
            else learning_config.children_per_elite,
            patience_factor=args.patience_factor if args.patience_factor is not None else learning_config.patience_factor,
            min_candidate_evaluations=learning_config.min_candidate_evaluations,
            no_improvement_patience_evaluations=learning_config.no_improvement_patience_evaluations,
            crossover_rate=args.crossover_rate
            if args.crossover_rate is not None
            else learning_config.crossover_rate,
            accept_margin=learning_config.accept_margin,
            max_repairs=learning_config.max_repairs,
            mutate_temperature=learning_config.mutate_temperature,
            mutate_max_tokens=learning_config.mutate_max_tokens,
            losses=active_losses,
            artifact_root=run_root,
            base_library=supervised_library,
            detector_input_mode="supervised_evidence",
            resume=args.resume,
        )
    progress_log(
        "dashboard.workflows.automatic.learning_done",
        env=args.env,
        run_dir=run_root,
        proposed_label_count=len(proposed_labels),
    )
    _require_complete_labels_for_expected(run_dir=run_root, expected_labels=proposed_labels)
    _require_current_library_labels(
        library_path=unsupervised_library_path,
        expected_labels=proposed_labels,
    )

    unsupervised_library = load_pattern_library(unsupervised_library_path)
    combined_library = _combine_libraries(
        env_id=args.env,
        source_library=supervised_library,
        unsupervised_library=unsupervised_library,
        run_id=run_root.name,
    )
    combined_library_path = pattern_library_path_for_run_dir(run_root)
    save_pattern_library(combined_library, combined_library_path)

    run_payload = _load_run_payload(run_root)
    run_payload.update(
        {
            "run_kind": "automatic",
            "env_id": args.env,
            "run_id": run_root.name,
            "source_supervised_run_id": source_run_dir.name,
            "source_supervised_run_dir": str(source_run_dir),
            "source_supervised_library_path": str(source_library_path),
            "source_evaluation_run_id": resolved_source_evaluation_run_id,
            "source_evaluation_results_path": str(results_path),
            "proposed_labels": proposed_labels,
            "proposed_label_details": [
                {"label": item.label, "description": item.description}
                for item in proposed_label_details
            ],
            "proposed_labels_path": str(run_root / "proposed_labels.txt"),
            "proposed_label_details_path": str(proposed_label_details_path),
            "sampled_reasoning_trace_count": len(sampled_reasoning_traces),
            "sampled_reasoning_traces_path": str(run_root / "sampled_reasoning_traces.json"),
            "label_proposal_response_path": str(run_root / "label_proposal_response.txt"),
            "unsupervised_library_path": str(unsupervised_library_path),
            "combined_library_path": str(combined_library_path),
            "combined_detector_count": len(combined_library.detectors),
            "request": {
                "env_id": args.env,
                "source_supervised_run_dir": str(source_run_dir),
                "source_pattern_library": str(source_library_path),
                "source_evaluation_run_id": resolved_source_evaluation_run_id,
                "requested_source_evaluation_run_id": args.source_evaluation_run_id,
                "reasoning_trace_count": args.reasoning_trace_count,
                "proposed_label_count": args.proposed_label_count,
                "split": list(args.split),
                "limit": args.limit,
                "iteration_count": args.iteration_count,
                "workers": args.workers,
                "seed": args.seed,
                "population_size": args.population_size,
                "patience_factor": args.patience_factor,
                "elite_fraction": args.elite_fraction,
                "children_per_elite": args.children_per_elite,
                "crossover_rate": args.crossover_rate,
                "run_timestamp": args.run_timestamp,
            },
        }
    )
    _write_json(run_root / "run.json", run_payload)

    eval_run_id = "combined-eval"
    eval_workers = 1 if args.env in {"kinetix", "pooltool"} else label_workers
    eval_task_limit = default_benchmark_task_limit(args.env, preset="full")
    eval_argv = [
        str(python_executable_for_env(args.env)),
        "-m",
        "deepphy_eval.evaluate_deepphy",
        "--env",
        args.env,
        "--prompt-format",
        "WM",
        "--model",
        args.model,
        "--description-mode",
        "trace_text_plus_patterns",
        "--ollama-base-url",
        llm.base_url,
        "--model-timeout",
        str(llm.timeout),
        "--pattern-run-dir",
        str(run_root),
        "--run-id",
        eval_run_id,
    ]
    if eval_task_limit is not None:
        eval_argv.extend(["--task-limit", str(eval_task_limit)])
    eval_argv.extend(["--workers", str(eval_workers)])
    eval_code = _run_child(eval_argv)
    if eval_code != 0:
        print(
            "[dashboard.workflows] Combined automatic evaluation failed.",
            file=sys.stderr,
        )
        return eval_code

    refreshed_payload = _load_run_payload(run_root)
    refreshed_payload["auto_evaluation_run_id"] = eval_run_id
    _write_json(run_root / "run.json", refreshed_payload)
    progress_log(
        "dashboard.workflows.automatic.done",
        env=args.env,
        run_dir=run_root,
        eval_run_id=eval_run_id,
    )
    return 0


def unsupervised_launch(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dashboard.workflows unsupervised-launch",
        description=(
            "Learn a real unsupervised pattern library by proposing labels from trace images only, "
            "then run the same learning/evaluation flow as a supervised job."
        ),
    )
    parser.add_argument("--env", required=True, help="Environment id (e.g. phyre)")
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        required=True,
        help="Dataset manifest used both for trace-only label proposal and pattern learning.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional pattern-learning config override.",
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--validation-split", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--iteration-count", type=int, default=8)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--patience-factor", type=float, default=0.35)
    parser.add_argument("--elite-fraction", type=float, default=None)
    parser.add_argument("--children-per-elite", type=int, default=None)
    parser.add_argument("--crossover-rate", type=float, default=None)
    parser.add_argument("--trace-count", type=int, default=TRUE_UNSUPERVISED_TRACE_COUNT)
    parser.add_argument(
        "--labels-per-trace",
        type=int,
        default=TRUE_UNSUPERVISED_LABELS_PER_TRACE,
    )
    parser.add_argument(
        "--final-label-count",
        type=int,
        default=TRUE_UNSUPERVISED_FINAL_LABEL_COUNT,
    )
    parser.add_argument(
        "--run-timestamp",
        default=None,
        help="Optional timestamp override used to name the unsupervised run record directory.",
    )
    parser.add_argument("--defer-evaluation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    progress_log(
        "dashboard.workflows.unsupervised.start",
        env=args.env,
        config=args.config,
        dataset_manifest=args.dataset_manifest,
        iteration_count=args.iteration_count,
        workers=args.workers,
        trace_count=args.trace_count,
        labels_per_trace=args.labels_per_trace,
        final_label_count=args.final_label_count,
        resume=args.resume,
    )

    run_root = default_unsupervised_run_dir(args.env, timestamp=args.run_timestamp)
    if args.resume:
        if not run_root.is_dir():
            run_root.mkdir(parents=True, exist_ok=False)
            progress_log(
                "dashboard.workflows.unsupervised.resume_fallback_fresh_start",
                env=args.env,
                run_dir=run_root,
            )
    else:
        if run_root.is_dir():
            progress_log(
                "dashboard.workflows.unsupervised.fresh_start_reusing_existing_dir",
                env=args.env,
                run_dir=run_root,
            )
        else:
            run_root.mkdir(parents=True, exist_ok=False)

    layout = get_artifact_layout()
    llm = load_shared_llm_config(layout.config_root / "LLM.yaml")
    config_path = args.config or (layout.config_root / "pattern_learning.yaml")
    learning_config = load_pattern_learning_config(config_path)
    label_workers = args.workers if args.workers is not None else learning_config.workers
    active_losses = list(learning_config.losses or default_loss_configs())

    samples, dataset_meta = load_learning_samples_from_dataset_manifest(
        args.dataset_manifest,
        split_names=args.split or list(learning_config.split) or None,
        limit=args.limit if args.limit is not None else learning_config.limit,
    )
    if not samples:
        raise RuntimeError("Unsupervised learning did not find any dataset samples to train on")
    validation_samples, _ = load_learning_samples_from_dataset_manifest(
        args.dataset_manifest,
        split_names=args.validation_split or list(learning_config.validation_split) or None,
        limit=args.limit if args.limit is not None else learning_config.limit,
    )
    if not validation_samples:
        raise RuntimeError("Unsupervised learning did not find any validation dataset samples")

    proposed_labels_path = run_root / "proposed_labels.txt"
    proposed_label_details_path = run_root / "proposed_labels.json"
    candidate_labels_path = run_root / "candidate_labels.json"
    sampled_traces_path = run_root / "sampled_traces.json"

    if args.resume and proposed_labels_path.is_file() and proposed_label_details_path.is_file():
        proposed_label_details = _load_proposed_label_details(proposed_label_details_path)
        proposed_labels = [item.label for item in proposed_label_details]
        progress_log(
            "dashboard.workflows.unsupervised.labels_resumed",
            env=args.env,
            proposed_label_count=len(proposed_labels),
        )
    else:
        sampled_traces = _sample_unsupervised_traces(
            samples=samples,
            trace_count=args.trace_count,
            seed=args.seed,
        )
        candidate_labels: list[dict[str, Any]] = []
        trace_records: list[dict[str, Any]] = []
        seen_labels: set[str] = set()
        for trace_index, sample in enumerate(sampled_traces, start=1):
            labels, _response_text, image_paths = _propose_trace_labels(
                env_id=args.env,
                sample=sample,
                labels_per_trace=args.labels_per_trace,
                llm_model=args.model,
                llm_base_url=llm.base_url,
                llm_timeout=llm.timeout,
                forbidden_labels=sorted(seen_labels),
                artifact_root=run_root / "trace_label_proposals",
                trace_index=trace_index,
                require_full_count=False,
            )
            if not labels:
                progress_log(
                    "dashboard.workflows.unsupervised.trace_labels_skipped",
                    env=args.env,
                    trace_index=trace_index,
                    task_id=getattr(sample, "task_id", None),
                )
                continue
            label_items = [
                {
                    "label": item.label,
                    "description": item.description,
                    "trace_index": trace_index,
                    "task_id": getattr(sample, "task_id", None),
                    "trace_path": str(getattr(sample, "trace_path", "")),
                }
                for item in labels
            ]
            candidate_labels.extend(label_items)
            seen_labels.update(item["label"].strip().lower() for item in label_items)
            trace_records.append(
                {
                    "trace_index": trace_index,
                    "task_id": getattr(sample, "task_id", None),
                    "trace_path": str(getattr(sample, "trace_path", "")),
                    "image_paths": image_paths,
                    "proposed_labels": label_items,
                }
            )
        _write_json(sampled_traces_path, {"records": trace_records})
        _write_json(candidate_labels_path, {"labels": candidate_labels})
        if len(candidate_labels) < args.final_label_count:
            raise RuntimeError(
                "Unsupervised label proposal produced only "
                f"{len(candidate_labels)} candidate labels across {len(trace_records)} traces, "
                f"expected at least {args.final_label_count} before final selection."
            )
        proposed_label_details, _selection_response = _select_final_unsupervised_labels(
            env_id=args.env,
            candidate_labels=candidate_labels,
            final_label_count=args.final_label_count,
            llm_model=args.model,
            llm_base_url=llm.base_url,
            llm_timeout=llm.timeout,
            artifact_root=run_root,
        )
        proposed_labels = [item.label for item in proposed_label_details]
        _write_json(
            proposed_label_details_path,
            {
                "labels": [
                    {"label": item.label, "description": item.description}
                    for item in proposed_label_details
                ]
            },
        )
        proposed_labels_path.write_text(
            "".join(f"{label}\n" for label in proposed_labels), encoding="utf-8"
        )
        progress_log(
            "dashboard.workflows.unsupervised.labels_proposed",
            env=args.env,
            proposed_label_count=len(proposed_labels),
            candidate_label_count=len(candidate_labels),
        )

    label_descriptions = {
        item.label: item.description for item in proposed_label_details if item.description
    }
    library_path = run_root / PATTERN_LIBRARY_FILENAME
    learning_already_complete = False
    if args.resume and library_path.is_file():
        _require_complete_labels_for_expected(
            run_dir=run_root,
            expected_labels=proposed_labels,
        )
        _require_current_library_labels(
            library_path=library_path,
            expected_labels=proposed_labels,
        )
        learning_already_complete = True
        progress_log(
            "dashboard.workflows.unsupervised.learning_resumed",
            env=args.env,
            run_dir=run_root,
        )
    if not learning_already_complete:
        run_pattern_learning(
            environment_id=args.env,
            primitive_labels=proposed_labels,
            label_descriptions=label_descriptions,
            samples=samples,
            validation_samples=validation_samples,
            output_path=library_path,
            scorer_config={
                "dataset_manifest": str(args.dataset_manifest),
                "train_split": list(args.split or list(learning_config.split)),
                "validation_split": list(args.validation_split or list(learning_config.validation_split)),
                "generator_model": args.model,
                "generator_base_url": learning_config.generator_base_url,
                "generator_timeout": learning_config.generator_timeout,
                "generator_retries": learning_config.generator_retries,
                "losses": [loss.model_dump(mode="json") for loss in active_losses],
                "config_path": str(config_path),
            },
            iteration_count=args.iteration_count or learning_config.iteration_count,
            label_workers=label_workers,
            primitive_source=str(proposed_labels_path),
            dataset_id=dataset_meta.get("dataset_id"),
            dataset_version=dataset_meta.get("dataset_version"),
            generator_model=args.model,
            generator_base_url=learning_config.generator_base_url,
            generator_timeout=learning_config.generator_timeout,
            generator_retries=learning_config.generator_retries,
            seed=args.seed if args.seed is not None else learning_config.seed,
            population_size=args.population_size if args.population_size is not None else learning_config.population_size,
            elite_fraction=args.elite_fraction if args.elite_fraction is not None else learning_config.elite_fraction,
            children_per_elite=args.children_per_elite if args.children_per_elite is not None else learning_config.children_per_elite,
            patience_factor=args.patience_factor if args.patience_factor is not None else learning_config.patience_factor,
            min_candidate_evaluations=learning_config.min_candidate_evaluations,
            no_improvement_patience_evaluations=learning_config.no_improvement_patience_evaluations,
            crossover_rate=args.crossover_rate if args.crossover_rate is not None else learning_config.crossover_rate,
            accept_margin=learning_config.accept_margin,
            max_repairs=learning_config.max_repairs,
            mutate_temperature=learning_config.mutate_temperature,
            mutate_max_tokens=learning_config.mutate_max_tokens,
            losses=active_losses,
            artifact_root=run_root,
            detector_input_mode="trace",
            resume=args.resume,
        )
    progress_log(
        "dashboard.workflows.unsupervised.learning_done",
        env=args.env,
        run_dir=run_root,
        proposed_label_count=len(proposed_labels),
    )
    _require_complete_labels_for_expected(run_dir=run_root, expected_labels=proposed_labels)
    _require_current_library_labels(
        library_path=library_path,
        expected_labels=proposed_labels,
    )

    run_payload = _load_run_payload(run_root)
    run_payload.update(
        {
            "run_kind": "unsupervised",
            "env_id": args.env,
            "run_id": run_root.name,
            "dataset_manifest": str(args.dataset_manifest),
            "proposed_labels": proposed_labels,
            "proposed_label_details": [
                {"label": item.label, "description": item.description}
                for item in proposed_label_details
            ],
            "proposed_labels_path": str(proposed_labels_path),
            "proposed_label_details_path": str(proposed_label_details_path),
            "candidate_labels_path": str(candidate_labels_path) if candidate_labels_path.is_file() else None,
            "sampled_traces_path": str(sampled_traces_path) if sampled_traces_path.is_file() else None,
            "sampled_reasoning_trace_count": args.trace_count,
            "final_label_selection_response_path": (
                str(run_root / "final_label_selection_response.txt")
                if (run_root / "final_label_selection_response.txt").is_file()
                else None
            ),
            "trace_count": args.trace_count,
            "labels_per_trace": args.labels_per_trace,
            "final_label_count": args.final_label_count,
            "request": {
                "env_id": args.env,
                "dataset_manifest": str(args.dataset_manifest),
                "split": list(args.split),
                "validation_split": list(args.validation_split),
                "limit": args.limit,
                "iteration_count": args.iteration_count,
                "workers": args.workers,
                "seed": args.seed,
                "population_size": args.population_size,
                "patience_factor": args.patience_factor,
                "elite_fraction": args.elite_fraction,
                "children_per_elite": args.children_per_elite,
                "crossover_rate": args.crossover_rate,
                "trace_count": args.trace_count,
                "labels_per_trace": args.labels_per_trace,
                "final_label_count": args.final_label_count,
                "run_timestamp": args.run_timestamp,
                "model": args.model,
                "defer_evaluation": args.defer_evaluation,
            },
        }
    )
    _write_json(run_root / "run.json", run_payload)
    if args.defer_evaluation:
        refreshed_payload = _load_run_payload(run_root)
        refreshed_payload["defer_evaluation"] = True
        refreshed_payload["evaluation_strategy"] = "scheduled"
        _write_json(run_root / "run.json", refreshed_payload)
        progress_log(
            "dashboard.workflows.unsupervised.evaluation_deferred",
            env=args.env,
            run_dir=run_root,
        )
        return 0
    return _run_supervised_auto_evaluation(
        env_id=args.env,
        run_dir=run_root,
        model_name=args.model,
    )


def library_evaluate_holdouts(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dashboard.workflows library-evaluate-holdouts",
        description=(
            "Run one sampled library evaluation per enabled-label count from 1 through |L|, "
            "and append a new pass on resume so ablation evidence can be layered over time."
        ),
    )
    parser.add_argument("--env", required=True)
    parser.add_argument("--pattern-run-dir", type=Path, required=True)
    parser.add_argument(
        "--template-name",
        default=DEFAULT_LIBRARY_HOLDOUT_EVALUATION_TEMPLATE,
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--ollama-base-url", required=True)
    parser.add_argument("--model-timeout", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--holdout-sample-count", type=int, default=None)
    parser.add_argument("--single-holdout-repeat-count", type=int, default=None)
    parser.add_argument("--pair-holdout-size", type=int, default=None)
    parser.add_argument("--pair-holdout-sample-count", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    action_request = LibraryHoldoutEvaluationActionRequest.model_validate(
        {
            "env_id": args.env,
            "pattern_run_dir": args.pattern_run_dir,
            "template_name": args.template_name,
            "workers": args.workers,
            "model": args.model,
            "ollama_base_url": args.ollama_base_url,
            "model_timeout": args.model_timeout,
            "run_id": args.run_id,
            "seed_count": args.seed_count,
            "holdout_sample_count": args.holdout_sample_count,
            "single_holdout_repeat_count": args.single_holdout_repeat_count,
            "pair_holdout_size": args.pair_holdout_size,
            "pair_holdout_sample_count": args.pair_holdout_sample_count,
        }
    )
    run_dir = resolve_pattern_run_dir(action_request.pattern_run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Pattern run directory not found: {run_dir}")
    source_library_path = pattern_library_path_for_run_dir(run_dir)
    if not source_library_path.is_file():
        raise FileNotFoundError(f"Pattern library not found: {source_library_path}")
    source_library = load_pattern_library(source_library_path)
    available_labels = _load_library_labels(run_dir)
    library_evaluation_run_id = action_request.run_id
    if args.resume and not library_evaluation_run_id:
        library_evaluation_run_id = _resume_library_evaluation_run_id(run_dir)
    library_evaluation_run_id = library_evaluation_run_id or default_run_id()
    eval_root = _library_evaluation_root(
        run_dir=run_dir,
        evaluation_run_id=library_evaluation_run_id,
    )
    if eval_root.is_dir() and not args.resume:
        shutil.rmtree(eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)

    manifest_path = eval_root / "manifest.json"
    manifest = _load_or_initialize_library_ablation_manifest(
        eval_root=eval_root,
        manifest_path=manifest_path,
        available_labels=available_labels,
        template_name=action_request.template_name,
        benchmark_preset=action_request.benchmark_preset or "quick-eval",
        source_run_id=run_dir.name,
    )
    _ensure_library_ablation_pass_planned(
        eval_root=eval_root,
        manifest=manifest,
        available_labels=available_labels,
        source_run_id=run_dir.name,
    )
    _write_json(manifest_path, manifest)

    _write_library_holdout_request(
        eval_root=eval_root,
        action_request=action_request,
        run_dir=run_dir,
        manifest=manifest,
        library_evaluation_run_id=library_evaluation_run_id,
    )
    _write_library_holdout_summary(eval_root, manifest)
    progress_log(
        "dashboard.workflows.library_eval.start",
        env=action_request.env_id,
        run_dir=run_dir,
        library_evaluation_run_id=library_evaluation_run_id,
        template_name=action_request.template_name,
    )

    for case in manifest.get("cases", []):
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("case_id") or "").strip()
        if not case_id:
            continue
        for seed_run in case.get("seed_runs", []):
            if not isinstance(seed_run, dict):
                continue
            raw_seed = seed_run.get("seed")
            seed = int(raw_seed) if isinstance(raw_seed, int) else None
            raw_evaluation_run_id = seed_run.get("evaluation_run_id")
            evaluation_run_id = (
                raw_evaluation_run_id.strip()
                if isinstance(raw_evaluation_run_id, str)
                and raw_evaluation_run_id.strip()
                else None
            )
            raw_relative_root = seed_run.get("relative_root")
            relative_root = (
                raw_relative_root.strip()
                if isinstance(raw_relative_root, str) and raw_relative_root.strip()
                else None
            )
            if seed is None or not evaluation_run_id or not relative_root:
                continue
            child_root = eval_root / relative_root
            if (child_root / "results.json").is_file():
                continue
            enabled_labels = _normalize_labels(seed_run.get("enabled_labels") or [])
            held_out_labels = _normalize_labels(seed_run.get("held_out_labels") or [])
            if not enabled_labels:
                enabled_labels = _normalize_labels(case.get("enabled_labels") or [])
            if not held_out_labels and enabled_labels:
                enabled_set = set(enabled_labels)
                held_out_labels = [
                    label for label in available_labels if label not in enabled_set
                ]
            pattern_library_override: Path | None = None
            if held_out_labels:
                pattern_library_override = _materialize_library_holdout_filtered_library(
                    source_run_dir=run_dir,
                    source_library=source_library,
                    eval_root=eval_root,
                    case_id=case_id,
                    artifact_id=_slugify_library_eval_token(evaluation_run_id),
                    available_labels=available_labels,
                    enabled_labels=enabled_labels,
                )
            attempt_overrides = _library_ablation_benchmark_overrides(action_request.env_id)
            child_request_payload: dict[str, Any] = {
                "env_id": action_request.env_id,
                "benchmark_preset": action_request.benchmark_preset,
                "benchmark_seed": seed,
                "prompt_format": "WM",
                "model": action_request.model,
                "description_mode": "pattern_text",
                "ollama_base_url": action_request.ollama_base_url,
                "llm_api_key": "",
                "model_timeout": action_request.model_timeout,
                "capture_visuals": True,
                "workers": action_request.workers,
                "run_id": evaluation_run_id,
                "output_dir": child_root,
                **attempt_overrides,
            }
            if pattern_library_override is not None:
                child_request_payload["pattern_library"] = pattern_library_override
            else:
                child_request_payload["pattern_run_dir"] = run_dir
            child_request = DeepPHYEvaluationActionRequest.model_validate(
                child_request_payload
            )
            progress_log(
                "dashboard.workflows.library_eval.case_start",
                env=action_request.env_id,
                run_dir=run_dir,
                library_evaluation_run_id=library_evaluation_run_id,
                case_id=case_id,
                seed=seed,
                enabled_label_count=len(enabled_labels),
                held_out_labels=held_out_labels,
            )
            exit_code = _run_child(
                [
                    str(python_executable_for_env(action_request.env_id)),
                    *build_deepphy_eval_argv(child_request),
                ]
            )
            _write_library_holdout_summary(eval_root, manifest)
            if exit_code != 0:
                return exit_code

    _write_library_holdout_summary(eval_root, manifest)
    progress_log(
        "dashboard.workflows.library_eval.done",
        env=action_request.env_id,
        run_dir=run_dir,
        library_evaluation_run_id=library_evaluation_run_id,
    )
    return 0


def library_evaluate_holdouts_continuous(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dashboard.workflows library-evaluate-holdouts-continuous",
        description=(
            "Append one small held-out library evaluation at a time and keep looping "
            "forever so ablation evidence accumulates incrementally."
        ),
    )
    parser.add_argument("--targets-json", required=True)
    parser.add_argument(
        "--template-name",
        default=DEFAULT_LIBRARY_HOLDOUT_EVALUATION_TEMPLATE,
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--ollama-base-url", required=True)
    parser.add_argument("--model-timeout", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--baseline-interval", type=int, default=4)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-iterations", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        raw_targets = json.loads(args.targets_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --targets-json payload: {exc}") from exc
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("--targets-json must be a non-empty JSON list.")

    targets: list[tuple[LibraryHoldoutEvaluationActionRequest, str]] = []
    for raw_target in raw_targets:
        if not isinstance(raw_target, dict):
            continue
        action_request = LibraryHoldoutEvaluationActionRequest.model_validate(
            {
                "env_id": raw_target.get("env_id"),
                "pattern_run_dir": raw_target.get("pattern_run_dir"),
                "template_name": args.template_name,
                "workers": args.workers,
                "model": args.model,
                "ollama_base_url": args.ollama_base_url,
                "model_timeout": args.model_timeout,
            }
        )
        run_id = (
            str(raw_target.get("run_id") or "").strip()
            or _default_continuous_library_ablation_run_id()
        )
        targets.append((action_request, run_id))
    if not targets:
        raise ValueError("No valid holdout targets were supplied.")

    baseline_interval = max(0, int(args.baseline_interval))
    max_iterations = (
        max(1, int(args.max_iterations))
        if args.max_iterations is not None
        else None
    )
    completed_iterations = 0
    while max_iterations is None or completed_iterations < max_iterations:
        for action_request, library_evaluation_run_id in targets:
            run_dir = resolve_pattern_run_dir(action_request.pattern_run_dir)
            if not run_dir.is_dir():
                progress_log(
                    "dashboard.workflows.library_eval.loop_target_missing",
                    env=action_request.env_id,
                    run_dir=run_dir,
                    library_evaluation_run_id=library_evaluation_run_id,
                )
                continue
            source_library_path = pattern_library_path_for_run_dir(run_dir)
            if not source_library_path.is_file():
                progress_log(
                    "dashboard.workflows.library_eval.loop_library_missing",
                    env=action_request.env_id,
                    run_dir=run_dir,
                    library_evaluation_run_id=library_evaluation_run_id,
                )
                continue
            source_library = load_pattern_library(source_library_path)
            available_labels = _load_library_labels(run_dir)
            eval_root = _library_evaluation_root(
                run_dir=run_dir,
                evaluation_run_id=library_evaluation_run_id,
            )
            eval_root.mkdir(parents=True, exist_ok=True)
            manifest_path = eval_root / "manifest.json"
            if manifest_path.is_file():
                manifest = _load_or_initialize_library_ablation_manifest(
                    eval_root=eval_root,
                    manifest_path=manifest_path,
                    available_labels=available_labels,
                    template_name=action_request.template_name,
                    benchmark_preset=action_request.benchmark_preset or "quick-eval",
                    source_run_id=run_dir.name,
                )
            else:
                manifest = _continuous_library_ablation_manifest(
                    available_labels=available_labels,
                    template_name=action_request.template_name,
                    benchmark_preset=action_request.benchmark_preset or "quick-eval",
                    source_run_id=run_dir.name,
                    baseline_interval=baseline_interval,
                )
            case, seed_run = _append_continuous_library_ablation_case(
                env_id=action_request.env_id,
                manifest=manifest,
                eval_root=eval_root,
                available_labels=available_labels,
                source_run_id=run_dir.name,
                run_token=library_evaluation_run_id,
                baseline_interval=baseline_interval,
            )
            _write_json(manifest_path, manifest)
            _write_library_holdout_request(
                eval_root=eval_root,
                action_request=action_request,
                run_dir=run_dir,
                manifest=manifest,
                library_evaluation_run_id=library_evaluation_run_id,
                extra_fields={
                    "continuous": True,
                    "baseline_interval": baseline_interval,
                    "targets_json": raw_targets,
                },
            )
            _write_library_holdout_summary(eval_root, manifest)
            case_id = str(case.get("case_id") or "").strip()
            enabled_labels = _normalize_labels(case.get("enabled_labels") or [])
            held_out_labels = _normalize_labels(case.get("held_out_labels") or [])
            relative_root = str(seed_run.get("relative_root") or "").strip()
            evaluation_run_id = str(seed_run.get("evaluation_run_id") or "").strip()
            seed = int(seed_run.get("seed") or 0)
            child_root = eval_root / relative_root
            pattern_library_override: Path | None = None
            if held_out_labels:
                pattern_library_override = _materialize_library_holdout_filtered_library(
                    source_run_dir=run_dir,
                    source_library=source_library,
                    eval_root=eval_root,
                    case_id=case_id,
                    artifact_id=_slugify_library_eval_token(evaluation_run_id),
                    available_labels=available_labels,
                    enabled_labels=enabled_labels,
                )
            attempt_overrides = _library_ablation_benchmark_overrides(action_request.env_id)
            child_request_payload: dict[str, Any] = {
                "env_id": action_request.env_id,
                "benchmark_preset": action_request.benchmark_preset,
                "benchmark_seed": seed,
                "prompt_format": "WM",
                "model": action_request.model,
                "description_mode": "pattern_text",
                "ollama_base_url": action_request.ollama_base_url,
                "llm_api_key": "",
                "model_timeout": action_request.model_timeout,
                "capture_visuals": True,
                "workers": action_request.workers,
                "run_id": evaluation_run_id,
                "output_dir": child_root,
                **attempt_overrides,
            }
            if pattern_library_override is not None:
                child_request_payload["pattern_library"] = pattern_library_override
            else:
                child_request_payload["pattern_run_dir"] = run_dir
            child_request = DeepPHYEvaluationActionRequest.model_validate(
                child_request_payload
            )
            progress_log(
                "dashboard.workflows.library_eval.loop_case_start",
                env=action_request.env_id,
                run_dir=run_dir,
                library_evaluation_run_id=library_evaluation_run_id,
                case_id=case_id,
                seed=seed,
                enabled_label_count=len(enabled_labels),
                held_out_labels=held_out_labels,
            )
            exit_code = _run_child(
                [
                    str(python_executable_for_env(action_request.env_id)),
                    *build_deepphy_eval_argv(child_request),
                ]
            )
            _write_library_holdout_summary(eval_root, manifest)
            progress_log(
                "dashboard.workflows.library_eval.loop_case_done",
                env=action_request.env_id,
                run_dir=run_dir,
                library_evaluation_run_id=library_evaluation_run_id,
                case_id=case_id,
                exit_code=exit_code,
            )
            completed_iterations += 1
            if exit_code != 0:
                continue
            if max_iterations is not None and completed_iterations >= max_iterations:
                break
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
        if max_iterations is not None and completed_iterations >= max_iterations:
            break
    return 0


def _library_ablation_benchmark_overrides(env_id: str) -> dict[str, int]:
    normalized = str(env_id or "").strip().lower()
    if normalized in {"phyre", "iphyre"}:
        return {"max_attempts": 6}
    if normalized == "pooltool":
        return {"max_attempts": 5}
    if normalized == "kinetix":
        return {"max_steps": 10}
    return {}


def _resume_library_evaluation_run_id(run_dir: Path) -> str | None:
    evaluations_root = run_dir / LIBRARY_EVALUATIONS_DIRNAME
    if not evaluations_root.is_dir():
        return None
    best_run_id: str | None = None
    best_completed = -1
    best_total = -1
    best_mtime_ns = -1
    for candidate in evaluations_root.iterdir():
        if not candidate.is_dir():
            continue
        summary_path = candidate / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        completed = int(payload.get("completed_seed_run_count") or 0)
        total = int(payload.get("total_seed_run_count") or 0)
        if total <= 0 or completed >= total:
            continue
        mtime_ns = summary_path.stat().st_mtime_ns
        if (
            completed > best_completed
            or (completed == best_completed and total > best_total)
            or (completed == best_completed and total == best_total and mtime_ns > best_mtime_ns)
        ):
            best_completed = completed
            best_total = total
            best_mtime_ns = mtime_ns
            best_run_id = candidate.name
    return best_run_id


def pooltool_reward_optimize(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dashboard.workflows pooltool-reward-optimize",
        description=(
            "Synthesize or use a PoolTool reward DSL program, then optimize actions "
            "over random scenes with simulated annealing under dense and binary rewards."
        ),
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--goal-text", default="")
    parser.add_argument("--dsl-program", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--scene-count", type=int, default=10)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--budget", action="append", type=int, default=[])
    parser.add_argument("--rack-jitter", type=float, default=0.12)
    parser.add_argument("--ollama-base-url", required=True)
    parser.add_argument("--model-timeout", type=int, required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    run_id = args.run_id or default_run_id()
    result = run_pooltool_optimization_experiment(
        run_id=run_id,
        source_run_dir=resolve_pattern_run_dir(args.source_run_dir),
        goal_text=args.goal_text,
        dsl_program=args.dsl_program,
        model_name=args.model,
        llm_base_url=args.ollama_base_url,
        llm_timeout=args.model_timeout,
        scene_count=args.scene_count,
        seed_count=args.seed_count,
        budgets=args.budget or [10, 50, 100, 250, 500, 1000],
        rack_jitter=args.rack_jitter,
    )
    progress_log(
        "dashboard.workflows.pooltool_reward_optimize.done",
        run_id=run_id,
        summary_path=result.summary_path,
    )
    return 0


def _goal_specs_from_args(goal_texts: list[str]) -> list[PooltoolRewardGoal]:
    normalized = [item.strip() for item in goal_texts if item.strip()]
    if not normalized:
        return default_pooltool_reward_goals()
    return [
        PooltoolRewardGoal(
            goal_id=f"custom-{index:02d}",
            goal_text=goal_text,
        )
        for index, goal_text in enumerate(normalized, start=1)
    ]


def pooltool_reward_optimize_suite(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dashboard.workflows pooltool-reward-optimize-suite",
        description=(
            "Run the PoolTool reward-program optimization experiment across a goal suite."
        ),
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--goal-text", action="append", default=[])
    parser.add_argument("--model", default="")
    parser.add_argument("--scene-count", type=int, default=10)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--budget", action="append", type=int, default=[])
    parser.add_argument("--rack-jitter", type=float, default=0.12)
    parser.add_argument("--ollama-base-url", required=True)
    parser.add_argument("--model-timeout", type=int, required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    run_id = args.run_id or default_run_id()
    result = run_pooltool_optimization_goal_suite(
        run_id=run_id,
        source_run_dir=resolve_pattern_run_dir(args.source_run_dir),
        goals=_goal_specs_from_args(args.goal_text),
        model_name=args.model,
        llm_base_url=args.ollama_base_url,
        llm_timeout=args.model_timeout,
        scene_count=args.scene_count,
        seed_count=args.seed_count,
        budgets=args.budget or [10, 50, 100, 250, 500, 1000],
        rack_jitter=args.rack_jitter,
    )
    progress_log(
        "dashboard.workflows.pooltool_reward_optimize_suite.done",
        run_id=run_id,
        summary_path=result.summary_path,
        goal_count=len(result.goals),
    )
    return 0


def pooltool_reward_value_model(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dashboard.workflows pooltool-reward-value-model",
        description=(
            "Train and evaluate an action-conditioned PoolTool value model from a reward DSL program."
        ),
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--goal-text", default="")
    parser.add_argument("--dsl-program", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--train-scene-count", type=int, default=64)
    parser.add_argument("--train-actions-per-scene", type=int, default=32)
    parser.add_argument("--validation-scene-count", type=int, default=12)
    parser.add_argument("--validation-candidates-per-scene", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--rack-jitter", type=float, default=0.12)
    parser.add_argument("--ollama-base-url", required=True)
    parser.add_argument("--model-timeout", type=int, required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    run_id = args.run_id or default_run_id()
    result = run_pooltool_value_model_experiment(
        run_id=run_id,
        source_run_dir=resolve_pattern_run_dir(args.source_run_dir),
        goal_text=args.goal_text,
        dsl_program=args.dsl_program,
        model_name=args.model,
        llm_base_url=args.ollama_base_url,
        llm_timeout=args.model_timeout,
        train_scene_count=args.train_scene_count,
        train_actions_per_scene=args.train_actions_per_scene,
        validation_scene_count=args.validation_scene_count,
        validation_candidates_per_scene=args.validation_candidates_per_scene,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        rack_jitter=args.rack_jitter,
    )
    progress_log(
        "dashboard.workflows.pooltool_reward_value_model.done",
        run_id=run_id,
        summary_path=result.summary_path,
    )
    return 0


def pooltool_reward_single_scene(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dashboard.workflows pooltool-reward-single-scene",
        description=(
            "Optimize a single stored PoolTool scene against a DSL or natural-language reward program "
            "and render the best shot as video."
        ),
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--scene-path", type=Path, required=True)
    parser.add_argument("--goal-text", default="")
    parser.add_argument("--dsl-program", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--reward-kind", default="synthesized")
    parser.add_argument("--budget", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ollama-base-url", required=True)
    parser.add_argument("--model-timeout", type=int, required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    run_id = args.run_id or default_run_id()
    result = run_pooltool_single_scene_experiment(
        run_id=run_id,
        source_run_dir=resolve_pattern_run_dir(args.source_run_dir),
        scene_path=Path(args.scene_path).resolve(),
        goal_text=args.goal_text,
        dsl_program=args.dsl_program,
        model_name=args.model,
        llm_base_url=args.ollama_base_url,
        llm_timeout=args.model_timeout,
        reward_kind=args.reward_kind,
        budget=args.budget,
        seed=args.seed,
    )
    progress_log(
        "dashboard.workflows.pooltool_reward_single_scene.done",
        run_id=run_id,
        summary_path=result.summary_path,
    )
    return 0


def pooltool_reward_random_scene(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dashboard.workflows pooltool-reward-random-scene",
        description="Create a random PoolTool scene and render a still image.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-id", default="scene-random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rack-jitter", type=float, default=0.12)
    args = parser.parse_args(argv)

    summary = generate_pooltool_random_scene(
        output_root=args.output_dir.resolve(),
        scene_id=str(args.scene_id or "scene-random").strip() or "scene-random",
        seed=int(args.seed),
        rack_jitter=float(args.rack_jitter),
    )
    progress_log(
        "dashboard.workflows.pooltool_reward_random_scene.done",
        scene_id=summary.scene_id,
        scene_path=args.output_dir.resolve() / "scene.json",
    )
    return 0


def pooltool_reward_value_model_sweep(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dashboard.workflows pooltool-reward-value-model-sweep",
        description=(
            "Run the PoolTool reward-program value-model experiment across a goal suite and hyperparameter preset."
        ),
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--goal-text", action="append", default=[])
    parser.add_argument("--model", default="")
    parser.add_argument("--rack-jitter", type=float, default=0.12)
    parser.add_argument("--ollama-base-url", required=True)
    parser.add_argument("--model-timeout", type=int, required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    run_id = args.run_id or default_run_id()
    result = run_pooltool_value_model_sweep(
        run_id=run_id,
        source_run_dir=resolve_pattern_run_dir(args.source_run_dir),
        goals=_goal_specs_from_args(args.goal_text),
        configs=None,
        model_name=args.model,
        llm_base_url=args.ollama_base_url,
        llm_timeout=args.model_timeout,
        rack_jitter=args.rack_jitter,
    )
    progress_log(
        "dashboard.workflows.pooltool_reward_value_model_sweep.done",
        run_id=run_id,
        summary_path=result.summary_path,
        goal_count=len(result.goals),
        config_count=len(result.configs),
    )
    return 0


def ensemble_calibrate_library(argv: list[str]) -> int:
    from ensemble_library_training.ensemble_calibration import run_local_ensemble_calibration

    return int(run_local_ensemble_calibration(argv))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dashboard.workflows",
        description="Dashboard workflow runners used by the web UI job manager.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    supervised = subparsers.add_parser(
        "supervised-learn-and-evaluate",
        help="Run supervised learning then DeepPHY evaluation (if all labels are learned).",
    )
    supervised.set_defaults(_handler=supervised_learn_and_evaluate)

    automatic = subparsers.add_parser(
        "automatic-launch",
        help=(
            "Launch automatic label proposal, learning, combined-library creation, "
            "and DeepPHY evaluation."
        ),
    )
    automatic.set_defaults(_handler=automatic_launch)
    unsupervised = subparsers.add_parser(
        "unsupervised-launch",
        help=(
            "Launch a trace-only unsupervised label proposal run, then learn and evaluate the resulting library."
        ),
    )
    unsupervised.set_defaults(_handler=unsupervised_launch)

    library_eval = subparsers.add_parser(
        "library-evaluate-holdouts",
        help=(
            "Run repeated quick-eval sweeps for a supervised library with held-out labels."
        ),
    )
    library_eval.set_defaults(_handler=library_evaluate_holdouts)
    library_eval_continuous = subparsers.add_parser(
        "library-evaluate-holdouts-continuous",
        help=(
            "Continuously append small held-out library evaluations in a round-robin loop."
        ),
    )
    library_eval_continuous.set_defaults(
        _handler=library_evaluate_holdouts_continuous
    )
    ensemble_calibrate = subparsers.add_parser(
        "ensemble-calibrate-library",
        help="Locally optimize ensemble detector reliabilities and clustering thresholds.",
    )
    ensemble_calibrate.set_defaults(_handler=ensemble_calibrate_library)
    reward_optimize = subparsers.add_parser(
        "pooltool-reward-optimize",
        help="Run the PoolTool reward-program optimization experiment.",
    )
    reward_optimize.set_defaults(_handler=pooltool_reward_optimize)
    reward_value = subparsers.add_parser(
        "pooltool-reward-value-model",
        help="Run the PoolTool reward-program value-model experiment.",
    )
    reward_value.set_defaults(_handler=pooltool_reward_value_model)
    reward_single_scene = subparsers.add_parser(
        "pooltool-reward-single-scene",
        help="Run a single-scene PoolTool reward optimization and render the best shot.",
    )
    reward_single_scene.set_defaults(_handler=pooltool_reward_single_scene)
    reward_random_scene = subparsers.add_parser(
        "pooltool-reward-random-scene",
        help="Create a random PoolTool scene and render its still image.",
    )
    reward_random_scene.set_defaults(_handler=pooltool_reward_random_scene)

    reward_opt_suite = subparsers.add_parser(
        "pooltool-reward-optimize-suite",
        help="Run the PoolTool reward optimization experiment across multiple goals.",
    )
    reward_opt_suite.set_defaults(_handler=pooltool_reward_optimize_suite)

    reward_value_sweep = subparsers.add_parser(
        "pooltool-reward-value-model-sweep",
        help="Run the PoolTool reward value-model experiment across multiple goals and sweep configs.",
    )
    reward_value_sweep.set_defaults(_handler=pooltool_reward_value_model_sweep)

    parsed, rest = parser.parse_known_args(argv)
    handler = getattr(parsed, "_handler", None)
    if handler is None:
        raise RuntimeError(f"Unknown command: {parsed.command}")
    return int(handler(rest))


if __name__ == "__main__":
    raise SystemExit(main())
