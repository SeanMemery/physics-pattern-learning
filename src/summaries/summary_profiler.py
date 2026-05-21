from __future__ import annotations

import base64
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import task_summaries as ts


PROFILER_STATE_RELPATH = Path("state/profiler_state.json")
PROFILER_SAMPLE_COUNT = 10
PROFILED_VARIANTS = ("image_only", "supervised", "unsupervised")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat() -> str:
    return _utc_now().isoformat()


def _state_path(project_root: Path) -> Path:
    return project_root / PROFILER_STATE_RELPATH


def _default_results() -> dict[str, dict[str, dict[str, float | int]]]:
    results: dict[str, dict[str, dict[str, float | int]]] = {}
    for env_id in ts.SUPPORTED_VIDEO_RENDER_ENVS:
        results[env_id] = {}
        for variant in PROFILED_VARIANTS:
            results[env_id][variant] = {
                "observation_count": 0,
                "total_time_seconds": 0.0,
                "total_image_bytes": 0,
                "total_input_words": 0,
                "total_output_words": 0,
                "avg_time_seconds": 0.0,
                "avg_image_bytes": 0.0,
                "avg_input_words": 0.0,
                "avg_output_words": 0.0,
            }
    return results


def _default_state() -> dict[str, Any]:
    return {
        "status": "idle",
        "message": "",
        "progress": 0.0,
        "completed": 0,
        "total": 0,
        "run_count": 0,
        "active_run_id": "",
        "current_env": "",
        "current_variant": "",
        "current_task_id": "",
        "started_at": None,
        "finished_at": None,
        "updated_at": _isoformat(),
        "log_lines": [],
        "results": _default_results(),
        "sample_count_per_env": PROFILER_SAMPLE_COUNT,
    }


def load_profiler_state(project_root: Path) -> dict[str, Any]:
    path = _state_path(project_root)
    if not path.is_file():
        return _default_state()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_state()
    if not isinstance(payload, dict):
        return _default_state()
    base = _default_state()
    base.update(payload)
    results = payload.get("results")
    if isinstance(results, dict):
        merged = _default_results()
        for env_id, env_payload in results.items():
            if env_id not in merged or not isinstance(env_payload, dict):
                continue
            for variant, variant_payload in env_payload.items():
                if variant not in merged[env_id] or not isinstance(variant_payload, dict):
                    continue
                merged[env_id][variant].update(variant_payload)
        base["results"] = merged
    else:
        base["results"] = _default_results()
    return base


def save_profiler_state(project_root: Path, payload: dict[str, Any]) -> None:
    path = _state_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = _isoformat()
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _append_log(project_root: Path, message: str) -> None:
    payload = load_profiler_state(project_root)
    log_lines = [str(item) for item in (payload.get("log_lines") or []) if str(item).strip()]
    log_lines.append(f"[{_isoformat()}] {message}")
    payload["log_lines"] = log_lines[-240:]
    save_profiler_state(project_root, payload)


def _set_state(project_root: Path, **fields: Any) -> dict[str, Any]:
    payload = load_profiler_state(project_root)
    payload.update(fields)
    save_profiler_state(project_root, payload)
    return payload


def _count_words(text: str) -> int:
    cleaned = str(text or "").strip()
    if not cleaned:
        return 0
    return len(cleaned.split())


def _build_user_content(
    *,
    prompt_template: str,
    env_id: str,
    task_id: str,
    annotation_kind: str | None,
    annotation_text: str | None,
    image_paths: list[Path],
) -> tuple[list[dict[str, Any]], list[Path], str]:
    user_lines = [
        prompt_template.strip(),
        "",
        f"Environment: {env_id}",
        f"Task: {task_id}",
        f"Summary mode: {annotation_kind or 'image_only'}",
    ]
    effective_images = list(image_paths)
    if annotation_text:
        effective_images = effective_images[:1]
        user_lines.extend(
            [
                "",
                (
                    "Context: the provided image is the initial frame only. "
                    "The annotation text below is a human-readable learned pattern annotation for the whole simulation, "
                    "including temporal events beyond the initial frame."
                ),
                (
                    "Use the initial frame to ground the starting state, and use the annotation text "
                    "to understand the full dynamics and outcome. Preserve the concrete event details from "
                    "the annotation instead of replacing them with generic language."
                ),
                "",
                "Learned pattern annotation summary:",
                annotation_text,
            ]
        )
    user_lines.extend(
        [
            "",
            "Return only the final summary wrapped in a fenced code block exactly like ` ```summary ... ``` `:",
            "```summary",
            "<your summary>",
            "```",
        ]
    )
    user_text = "\n".join(user_lines).strip()
    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for image_path in effective_images:
        mime = "image/png"
        if image_path.suffix.lower() in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}"},
            }
        )
    return user_content, effective_images, user_text


def _request_summary_with_metrics(
    *,
    project_root: Path,
    prompt_template: str,
    env_id: str,
    task_id: str,
    annotation_kind: str | None,
    annotation_text: str | None,
    image_paths: list[Path],
) -> dict[str, Any]:
    ts._ensure_client_src_on_path()
    from pattern_learning.patterns.scoring import request_chat_completion_with_text_retry  # type: ignore

    base_url, model, timeout = ts._load_shared_llm_defaults()
    headers = {"Content-Type": "application/json"}
    user_content, effective_images, user_text = _build_user_content(
        prompt_template=prompt_template,
        env_id=env_id,
        task_id=task_id,
        annotation_kind=annotation_kind,
        annotation_text=annotation_text,
        image_paths=image_paths,
    )
    system_text = (
        "You summarize physics simulations from image frames and optional structured annotations. "
        "Return only the final concise summary. Do not include reasoning, scratch work, or analysis steps. "
        "Always wrap the summary in exactly one fenced code block: ` ```summary ... ``` `."
    )
    elapsed_seconds = 0.0

    def _call(messages: list[dict[str, Any]], *, max_tokens: int, error_prefix: str) -> tuple[dict[str, Any], str, float]:
        start = time.perf_counter()
        with ts._acquire_llm_request_slot(project_root):
            body, text = request_chat_completion_with_text_retry(
                base_url=base_url,
                model=model,
                messages=messages,
                timeout=timeout,
                headers=headers,
                error_prefix=error_prefix,
                max_tokens=max_tokens,
                temperature=0.0,
            )
        return body, text, time.perf_counter() - start

    body, text, elapsed = _call(
        [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_content},
        ],
        max_tokens=8192,
        error_prefix="Task summary response",
    )
    elapsed_seconds += elapsed
    explicit_summary = ts._extract_explicit_summary_from_completion(text, body)
    if explicit_summary:
        summary_text = explicit_summary
    else:
        reasoning_trace = ts._extract_reasoning_trace_from_completion(text, body)
        if reasoning_trace:
            reasoning_excerpt = reasoning_trace.strip()
            if len(reasoning_excerpt) > 24000:
                reasoning_excerpt = reasoning_excerpt[:24000].rstrip()
            body2, text2, elapsed2 = _call(
                [
                    {
                        "role": "system",
                        "content": (
                            "You extract a final concise simulation summary from another model's reasoning trace. "
                            "Ignore scratch work, repeated deliberation, and uncertainty. "
                            "Return only the grounded final summary in exactly one fenced code block: "
                            "```summary\\n...\\n```."
                        ),
                    },
                    {
                        "role": "user",
                        "content": "\n".join(
                            [
                                f"Environment: {env_id}",
                                f"Task: {task_id}",
                                "",
                                "The previous model failed to emit a final summary block.",
                                "Read its reasoning trace and extract the best final summary of the simulation.",
                                "Keep the summary concrete and concise, focused on the initial setup, key interactions, and final outcome.",
                                "",
                                "Reasoning trace:",
                                reasoning_excerpt,
                                "",
                                "Return only:",
                                "```summary",
                                "<your summary>",
                                "```",
                            ]
                        ).strip(),
                    },
                ],
                max_tokens=ts.SUMMARY_REASONING_EXTRACTION_MAX_TOKENS,
                error_prefix="Task summary reasoning extraction",
            )
            elapsed_seconds += elapsed2
            summary_text = ts._extract_summary_text_from_completion(text2, body2)
        else:
            summary_text = ts._extract_summary_text_from_completion(text, body)
    return {
        "summary_text": summary_text,
        "elapsed_seconds": elapsed_seconds,
        "image_bytes": sum(image_path.stat().st_size for image_path in effective_images if image_path.is_file()),
        "input_word_count": _count_words(system_text) + _count_words(user_text),
        "output_word_count": _count_words(summary_text),
    }


def _sample_tasks_for_env(project_root: Path, env_id: str, *, run_index: int) -> list[ts.TaskSummaryTask]:
    tasks = list(ts.discover_task_summary_tasks(project_root, env_id))
    if not tasks:
        return []
    sample_count = min(PROFILER_SAMPLE_COUNT, len(tasks))
    seed = f"summary-profiler:{env_id}:{run_index}:{len(tasks)}"
    rng = random.Random(seed)
    shuffled = list(tasks)
    rng.shuffle(shuffled)
    selected: list[ts.TaskSummaryTask] = []
    for task in shuffled:
        try:
            ts._render_summary_media(project_root, task)
        except Exception:
            continue
        selected.append(task)
        if len(selected) >= sample_count:
            break
    return sorted(selected, key=lambda item: (item.source_run_id, item.task_id))


def _accumulate_observation(
    payload: dict[str, Any],
    *,
    env_id: str,
    variant: str,
    elapsed_seconds: float,
    image_bytes: int,
    input_word_count: int,
    output_word_count: int,
) -> None:
    results = payload.setdefault("results", _default_results())
    env_bucket = results.setdefault(env_id, {})
    variant_bucket = env_bucket.setdefault(
        variant,
        {
            "observation_count": 0,
            "total_time_seconds": 0.0,
            "total_image_bytes": 0,
            "total_input_words": 0,
            "total_output_words": 0,
            "avg_time_seconds": 0.0,
            "avg_image_bytes": 0.0,
            "avg_input_words": 0.0,
            "avg_output_words": 0.0,
        },
    )
    variant_bucket["observation_count"] = int(variant_bucket.get("observation_count") or 0) + 1
    variant_bucket["total_time_seconds"] = float(variant_bucket.get("total_time_seconds") or 0.0) + float(elapsed_seconds)
    variant_bucket["total_image_bytes"] = int(variant_bucket.get("total_image_bytes") or 0) + int(image_bytes)
    variant_bucket["total_input_words"] = int(variant_bucket.get("total_input_words") or 0) + int(input_word_count)
    variant_bucket["total_output_words"] = int(variant_bucket.get("total_output_words") or 0) + int(output_word_count)
    count = max(1, int(variant_bucket["observation_count"]))
    variant_bucket["avg_time_seconds"] = float(variant_bucket["total_time_seconds"]) / count
    variant_bucket["avg_image_bytes"] = float(variant_bucket["total_image_bytes"]) / count
    variant_bucket["avg_input_words"] = float(variant_bucket["total_input_words"]) / count
    variant_bucket["avg_output_words"] = float(variant_bucket["total_output_words"]) / count


def run_summary_profiler(project_root: Path) -> int:
    state = load_profiler_state(project_root)
    run_index = int(state.get("run_count") or 0) + 1
    run_id = f"profiler-run-{run_index:03d}"
    config = ts.load_task_summary_config(project_root)
    image_only_prompt_template = str(
        config.get("image_only_prompt_template") or ts.DEFAULT_IMAGE_ONLY_SUMMARY_PROMPT
    ).strip()
    annotation_prompt_template = str(
        config.get("annotation_prompt_template") or ts.DEFAULT_ANNOTATION_SUMMARY_PROMPT
    ).strip()
    supervised_library_strategy = str(config.get("supervised_library_strategy") or "latest").strip().lower()
    started_at = _isoformat()
    state.update(
        {
            "status": "running",
            "message": "Sampling profiler tasks...",
            "progress": 0.0,
            "completed": 0,
            "total": 0,
            "active_run_id": run_id,
            "current_env": "",
            "current_variant": "",
            "current_task_id": "",
            "started_at": started_at,
            "finished_at": None,
            "log_lines": [],
            "results": _default_results(),
        }
    )
    save_profiler_state(project_root, state)
    _append_log(project_root, f"Profiler run {run_id} started.")
    _append_log(project_root, f"Using {PROFILER_SAMPLE_COUNT} sampled tasks per environment and variants: {', '.join(PROFILED_VARIANTS)}.")
    sampled_tasks: dict[str, list[ts.TaskSummaryTask]] = {}
    for env_id in ts.SUPPORTED_VIDEO_RENDER_ENVS:
        sampled_tasks[env_id] = _sample_tasks_for_env(project_root, env_id, run_index=run_index)
        _append_log(project_root, f"{env_id}: selected {len(sampled_tasks[env_id])} tasks with summary media.")
    total = sum(len(tasks) * len(PROFILED_VARIANTS) for tasks in sampled_tasks.values())
    state = load_profiler_state(project_root)
    state["message"] = "Starting summary profiler..."
    state["total"] = total
    state["started_at"] = started_at
    save_profiler_state(project_root, state)
    completed = 0
    try:
        for env_id in ts.SUPPORTED_VIDEO_RENDER_ENVS:
            tasks = sampled_tasks.get(env_id) or []
            for task in tasks:
                try:
                    trace_path, _video_path, initial_image_path, image_paths = ts._render_summary_media(project_root, task)
                    trace = ts._load_trace(trace_path)
                except Exception as exc:
                    _append_log(project_root, f"{env_id} {task.task_id}: skipped (media unavailable: {exc})")
                    completed += len(PROFILED_VARIANTS)
                    payload = load_profiler_state(project_root)
                    payload["completed"] = completed
                    payload["progress"] = (completed / total) if total > 0 else 1.0
                    save_profiler_state(project_root, payload)
                    continue
                for variant in PROFILED_VARIANTS:
                    payload = load_profiler_state(project_root)
                    payload["current_env"] = env_id
                    payload["current_variant"] = variant
                    payload["current_task_id"] = task.task_id
                    payload["message"] = f"{env_id} · {variant} · {task.task_id}"
                    payload["completed"] = completed
                    payload["progress"] = (completed / total) if total > 0 else 0.0
                    save_profiler_state(project_root, payload)
                    try:
                        if variant == "image_only":
                            metrics = _request_summary_with_metrics(
                                project_root=project_root,
                                prompt_template=image_only_prompt_template,
                                env_id=env_id,
                                task_id=task.task_id,
                                annotation_kind=None,
                                annotation_text=None,
                                image_paths=image_paths,
                            )
                        else:
                            library_path = ts._resolve_latest_library(
                                project_root,
                                env_id,
                                variant,
                                strategy=supervised_library_strategy if variant == "supervised" else "latest",
                            )
                            annotation_payload = ts._apply_library_to_trace(project_root, trace, library_path)
                            if env_id == "pooltool" and variant in {"supervised", "unsupervised"}:
                                annotation_payload = ts._filter_pooltool_summary_annotation_payload(annotation_payload)
                            pattern_count = ts._annotation_pattern_count(annotation_payload)
                            if pattern_count <= 0:
                                _append_log(project_root, f"{env_id} {variant} {task.task_id}: skipped (no detected patterns).")
                                completed += 1
                                continue
                            annotation_text = ts._format_annotation_text(annotation_payload)
                            metrics = _request_summary_with_metrics(
                                project_root=project_root,
                                prompt_template=annotation_prompt_template,
                                env_id=env_id,
                                task_id=task.task_id,
                                annotation_kind=variant,
                                annotation_text=annotation_text,
                                image_paths=[initial_image_path],
                            )
                        payload = load_profiler_state(project_root)
                        _accumulate_observation(
                            payload,
                            env_id=env_id,
                            variant=variant,
                            elapsed_seconds=float(metrics["elapsed_seconds"]),
                            image_bytes=int(metrics["image_bytes"]),
                            input_word_count=int(metrics["input_word_count"]),
                            output_word_count=int(metrics["output_word_count"]),
                        )
                        save_profiler_state(project_root, payload)
                        _append_log(
                            project_root,
                            (
                                f"{env_id} {variant} {task.task_id}: "
                                f"{float(metrics['elapsed_seconds']):.2f}s, "
                                f"{int(metrics['image_bytes'])} image bytes, "
                                f"{int(metrics['input_word_count'])} input words, "
                                f"{int(metrics['output_word_count'])} output words."
                            ),
                        )
                    except Exception as exc:
                        _append_log(project_root, f"{env_id} {variant} {task.task_id}: failed: {exc}")
                    completed += 1
                    payload = load_profiler_state(project_root)
                    payload["completed"] = completed
                    payload["progress"] = (completed / total) if total > 0 else 1.0
                    save_profiler_state(project_root, payload)

        payload = load_profiler_state(project_root)
        payload["status"] = "succeeded"
        payload["message"] = f"Completed profiler run {run_id}."
        payload["progress"] = 1.0
        payload["completed"] = completed
        payload["total"] = total
        payload["current_env"] = ""
        payload["current_variant"] = ""
        payload["current_task_id"] = ""
        payload["finished_at"] = _isoformat()
        payload["run_count"] = run_index
        save_profiler_state(project_root, payload)
        _append_log(project_root, f"Profiler run {run_id} finished.")
        return 0
    except Exception as exc:
        payload = load_profiler_state(project_root)
        payload["status"] = "failed"
        payload["message"] = f"Profiler run {run_id} failed."
        payload["completed"] = completed
        payload["total"] = total
        payload["progress"] = (completed / total) if total > 0 else 0.0
        payload["finished_at"] = _isoformat()
        save_profiler_state(project_root, payload)
        _append_log(project_root, f"Profiler run {run_id} failed: {exc}")
        return 1


__all__ = [
    "PROFILED_VARIANTS",
    "PROFILER_SAMPLE_COUNT",
    "PROFILER_STATE_RELPATH",
    "load_profiler_state",
    "run_summary_profiler",
    "save_profiler_state",
]
