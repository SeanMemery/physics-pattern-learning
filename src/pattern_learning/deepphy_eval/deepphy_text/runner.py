from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from adapters.registry import AdapterRegistry
from deepphy_eval.deepphy_text.describer import (
    TraceDescriber,
    describe_trace_with_mode,
    dummy_trace_description,
)
from deepphy_eval.deepphy_text.models import (
    DeepPHYTextEvaluationRequest,
    DeepPHYTextEvaluationResult,
    DeepPHYTextPrompt,
    DeepPHYTextPromptImage,
    DeepPHYTextRecord,
)
from deepphy_eval.deepphy_text.prompts import (
    build_iphyre_prompt,
    build_kinetix_prompt,
    build_phyre_prompt,
    build_pooltool_prompt,
)
from deepphy_eval.io import write_trace
from llm_config import GLOBAL_MIN_COMPLETION_TOKENS, load_shared_llm_config
from media import write_image_artifact
from paths import get_artifact_layout
from pattern_library_detectors import apply_pattern_library
from pattern_library_evidence import build_pattern_evidence, write_pattern_evidence
from pattern_library_paths import resolve_pattern_library_path
from pattern_library_registry import load_pattern_library
from pattern_learning.patterns.scoring import (
    ChatCompletionsResponder,
    _request_json,
    extract_chat_completion_text,
    request_chat_completion_with_text_retry,
)
from vendor.deepphy_support.img_utils import add_grid_with_matplotlib
from schema import CanonicalTrace
from pattern_library_schema import PatternEvidence, PatternLibrary


class ModelResponder(Protocol):
    def respond(self, prompt: DeepPHYTextPrompt) -> str: ...


def default_deepphy_text_run_id() -> str:
    return datetime.now(timezone.utc).strftime("deepphy-text-%Y%m%dT%H%M%SZ")


def load_deepphy_text_request(path: Path) -> DeepPHYTextEvaluationRequest:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return DeepPHYTextEvaluationRequest.model_validate(payload)


def evaluate_deepphy_text_request(
    request: DeepPHYTextEvaluationRequest,
    *,
    registry: AdapterRegistry | None = None,
    model_responder: ModelResponder | None = None,
    trace_describer: TraceDescriber | None = None,
) -> DeepPHYTextEvaluationResult:
    adapter_registry = registry or AdapterRegistry()
    responder = model_responder or _build_model_responder(request)
    describer = trace_describer or _DefaultTraceDescriber()
    run_id = request.run_id or default_deepphy_text_run_id()
    output_root = Path(request.output_root) if request.output_root else _default_output_root(run_id, request.env_id, request.task_id)
    output_root.mkdir(parents=True, exist_ok=True)
    adapter = adapter_registry.load(request.env_id)
    pattern_library = _load_pattern_library_for_request(request)

    if request.env_id == "phyre":
        return _evaluate_phyre(
            request=request,
            run_id=run_id,
            output_root=output_root,
            adapter=adapter,
            responder=responder,
            describer=describer,
            pattern_library=pattern_library,
        )
    if request.env_id == "iphyre":
        return _evaluate_iphyre(
            request=request,
            run_id=run_id,
            output_root=output_root,
            adapter=adapter,
            responder=responder,
            describer=describer,
            pattern_library=pattern_library,
        )
    if request.env_id == "kinetix":
        return _evaluate_kinetix(
            request=request,
            run_id=run_id,
            output_root=output_root,
            adapter=adapter,
            responder=responder,
            describer=describer,
            pattern_library=pattern_library,
        )
    if request.env_id == "pooltool":
        return _evaluate_pooltool(
            request=request,
            run_id=run_id,
            output_root=output_root,
            adapter=adapter,
            responder=responder,
            describer=describer,
            pattern_library=pattern_library,
        )
    raise ValueError(f"unsupported environment: {request.env_id}")


def _write_partial_task_result(
    *,
    run_id: str,
    request: DeepPHYTextEvaluationRequest,
    output_root: Path,
    initial_image_path: str,
    records: list[DeepPHYTextRecord],
) -> None:
    if request.output_root is None:
        return
    result = _build_result(
        run_id=run_id,
        request=request,
        output_root=output_root,
        initial_image_path=initial_image_path,
        records=records,
    )
    task_result_path = output_root / "task_result.json"
    def _json_fallback(value: Any) -> Any:
        if value is Ellipsis:
            return "..."
        return repr(value)
    task_result_path.write_text(
        json.dumps(
            result.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            default=_json_fallback,
        )
        + "\n",
        encoding="utf-8",
    )


class MockModelResponder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def respond(self, prompt: DeepPHYTextPrompt) -> str:
        env_id = prompt.env_id
        if self.model_name.lower() != "mock":
            raise NotImplementedError(
                "Only model_name='mock' is wired in Phase 8. Inject a custom model_responder for real LM calls."
            )
        if env_id == "phyre":
            return '{"cell": 1, "radius": 1}'
        if env_id == "iphyre":
            payload = '[{"time": 0.5, "index": 0}]'
            return f"Prediction: placeholder.\nAction:\n```json\n{payload}\n```" if prompt.prompt_format == "WM" else payload
        if env_id == "pooltool":
            action = "Speed: Medium, Strikespot: Middle Spin."
            return f"Prediction: placeholder.\nAction: {action}" if prompt.prompt_format == "WM" else action
        if env_id == "kinetix":
            total = int(prompt.metadata.get("num_active_joints", 0)) + int(prompt.metadata.get("num_active_thrusters", 0))
            vector = "[" + ", ".join(["0"] * total) + "]"
            return f"Prediction: placeholder.\nAction: {vector}" if prompt.prompt_format == "WM" else vector
        raise ValueError(f"unsupported mock environment: {env_id}")


class ChatCompletionsDeepPHYResponder:
    def __init__(self, *, model_name: str, base_url: str, timeout: int, max_tokens: int, api_key: str | None = None) -> None:
        self.client = ChatCompletionsResponder(
            base_url=base_url,
            model=model_name,
            timeout=timeout,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        self.last_router_info: dict[str, Any] = {}
        self.last_raw_response: dict[str, Any] = {}

    def respond(self, prompt: DeepPHYTextPrompt) -> str:
        headers = {"Content-Type": "application/json"}
        if self.client.api_key:
            headers["Authorization"] = f"Bearer {self.client.api_key}"
        user_content: str | list[dict[str, Any]]
        if prompt.image_inputs or prompt.initial_image_path:
            user_content = _build_multimodal_user_content(
                text=prompt.user_prompt,
                image_inputs=prompt.image_inputs
                or [DeepPHYTextPromptImage(path=prompt.initial_image_path, label="Initial scene")],
            )
        else:
            user_content = prompt.user_prompt
        body, content = request_chat_completion_with_text_retry(
            base_url=self.client.base_url,
            model=self.client.model,
            messages=[
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": user_content},
            ],
            timeout=self.client.timeout,
            headers=headers,
            max_tokens=self.client.max_tokens,
            temperature=0.0,
            stream=False,
            extra_payload={"chat_template_kwargs": {"enable_thinking": False}},
            error_prefix="Chat completions response",
            request_fn=_request_json,
        )
        self.last_raw_response = dict(body)
        self.last_router_info = (
            dict(body.get("router"))
            if isinstance(body.get("router"), dict)
            else {}
        )
        return content

    def repair_phyre_action(self, *, prompt: DeepPHYTextPrompt, raw_output: str) -> str | None:
        deterministic = _extract_repairable_phyre_cell(raw_output)
        if deterministic is not None:
            return deterministic
        headers = {"Content-Type": "application/json"}
        if self.client.api_key:
            headers["Authorization"] = f"Bearer {self.client.api_key}"
        body, content = request_chat_completion_with_text_retry(
            base_url=self.client.base_url,
            model=self.client.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the draft into the exact PHYRE action JSON. "
                        "Return exactly one JSON object in the form {\"cell\": <int 1-64>, \"radius\": <int 1-8>} and nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Draft output:\n"
                        f"{raw_output}\n\n"
                        "If the draft already names a cell or radius, preserve them. "
                        "Otherwise infer the most likely action and return only the JSON object."
                    ),
                },
            ],
            timeout=self.client.timeout,
            headers=headers,
            max_tokens=GLOBAL_MIN_COMPLETION_TOKENS,
            temperature=0.0,
            stream=False,
            extra_payload={"chat_template_kwargs": {"enable_thinking": False}},
            error_prefix="PHYRE action repair response",
            request_fn=_request_json,
        )
        self.last_raw_response = dict(body)
        self.last_router_info = (
            dict(body.get("router"))
            if isinstance(body.get("router"), dict)
            else {}
        )
        repaired = content.strip()
        if not repaired:
            return None
        coerced = _coerce_json_action(env_id="phyre", raw_output=repaired)
        return coerced or repaired

    def repair_action_output(
        self,
        *,
        prompt: DeepPHYTextPrompt,
        raw_output: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str | None:
        headers = {"Content-Type": "application/json"}
        if self.client.api_key:
            headers["Authorization"] = f"Bearer {self.client.api_key}"
        body, content = request_chat_completion_with_text_retry(
            base_url=self.client.base_url,
            model=self.client.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=self.client.timeout,
            headers=headers,
            max_tokens=GLOBAL_MIN_COMPLETION_TOKENS,
            temperature=0.0,
            stream=False,
            extra_payload={
                "chat_template_kwargs": {"enable_thinking": False},
            },
            error_prefix=f"{prompt.env_id.upper()} action repair response",
            request_fn=_request_json,
        )
        self.last_raw_response = dict(body)
        self.last_router_info = (
            dict(body.get("router"))
            if isinstance(body.get("router"), dict)
            else {}
        )
        repaired = content.strip()
        return repaired or None


class DeepPHYAttemptFailure(RuntimeError):
    def __init__(self, terminal_status: str, message: str, *, native_metrics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.terminal_status = terminal_status
        self.native_metrics = dict(native_metrics or {})


def _responder_router_info(responder: ModelResponder) -> dict[str, Any]:
    router_info = getattr(responder, "last_router_info", None)
    return dict(router_info) if isinstance(router_info, dict) else {}


class _DefaultTraceDescriber:
    def describe(
        self,
        trace: CanonicalTrace,
        *,
        env_id: str,
        task_id: str,
        index: int,
        record_kind: str,
        description_mode: str,
        evidence: PatternEvidence | None = None,
    ) -> str:
        return describe_trace_with_mode(
            trace,
            env_id=env_id,
            task_id=task_id,
            index=index,
            record_kind=record_kind,
            description_mode=description_mode,
            evidence=evidence,
        )


def _default_output_root(run_id: str, env_id: str, task_id: str) -> Path:
    layout = get_artifact_layout()
    safe_task_id = task_id.replace("/", "__").replace(":", "__")
    return layout.evaluations_root / run_id / "tasks" / env_id / safe_task_id


def _evaluate_phyre(
    *,
    request: DeepPHYTextEvaluationRequest,
    run_id: str,
    output_root: Path,
    adapter: Any,
    responder: ModelResponder,
    describer: TraceDescriber,
    pattern_library: PatternLibrary | None,
) -> DeepPHYTextEvaluationResult:
    settings = adapter.load_upstream_config()
    max_attempts = int(request.benchmark_kwargs.get("max_attempts", settings.get("max_attempts", 10)))
    initial_image_path, initial_meta = _normalize_initial_image_payload(
        _prepare_initial_image("phyre", adapter, request, output_root)
    )
    prompt_image_inputs = list(initial_meta.get("prompt_image_inputs", []))
    attempt_history: list[dict[str, Any]] = []
    records: list[DeepPHYTextRecord] = []
    trace_description = "No prior simulation trace description is available yet."
    seen_actions: list[tuple[int, int]] = []

    for attempt_index in range(1, max_attempts + 1):
        attempt_started_at = time.perf_counter()
        prompt, strategy = build_phyre_prompt(
            project_root=adapter.resolve_project_path(""),
            prompt_format=request.prompt_format,
            task_id=request.task_id,
            initial_image_path=initial_image_path,
            image_inputs=prompt_image_inputs,
            trace_description=trace_description,
            description_mode=request.description_mode,
            attempt_index=attempt_index,
            attempt_history=attempt_history,
        )
        raw_output = ""
        parsed: dict[str, Any] = {}
        repaired_output = None
        trace = None
        trace_path = None
        trace_id = None
        feedback_trace_text = ""
        feedback_pattern_text = ""
        terminal_status = "INVALID_ACTION_FORMAT"
        success = False
        metrics: dict[str, Any] = {}
        try:
            raw_output = responder.respond(prompt)
            parsed = strategy.parse_response(raw_output)
            coerced = _coerce_json_action(env_id="phyre", raw_output=raw_output)
            if request.prompt_format == "VLA" and coerced is not None:
                raw_output = coerced
                parsed = strategy.parse_response(raw_output)
            elif request.prompt_format == "WM" and coerced is not None and parsed.get("action") is None:
                parsed = strategy.parse_response(coerced)
            elif request.prompt_format == "WM" and parsed.get("action") is None:
                deterministic_action = _extract_repairable_phyre_cell(raw_output)
                if deterministic_action is not None:
                    parsed = strategy.parse_response(
                        _build_wm_action_response(
                            prediction=parsed.get("prediction"),
                            action_text=deterministic_action,
                        )
                    )
            action = parsed.get("action")
            if action is None:
                repaired_output = _repair_phyre_action_output(
                    responder=responder,
                    prompt=prompt,
                    raw_output=raw_output,
                )
                if repaired_output is not None:
                    repaired_parsed = strategy.parse_response(repaired_output)
                    if repaired_parsed.get("action") is not None:
                        parsed = repaired_parsed
                        action = parsed.get("action")
            parsed_action = {
                "cell": parsed.get("parsed_cell"),
                "radius_size": parsed.get("parsed_radius_size"),
                "action": action.tolist() if action is not None else None,
                **(
                    {"format_repair_output": repaired_output, "format_repair_used": True}
                    if repaired_output is not None
                    else {}
                ),
            }
            cell = parsed.get("parsed_cell")
            radius = parsed.get("parsed_radius_size")
            if cell is not None and radius is not None:
                duplicate_reason = _phyre_duplicate_reason(
                    cell=int(cell),
                    radius=int(radius),
                    seen_actions=seen_actions,
                )
                if duplicate_reason is not None:
                    raise DeepPHYAttemptFailure(
                        "REPEATED_ACTION",
                        duplicate_reason,
                        native_metrics={"cell": int(cell), "radius_size": int(radius), "duplicate_reason": duplicate_reason},
                    )
            if action is not None:
                attempt_root = output_root / f"attempt_{attempt_index:03d}"
                trace = adapter.generate_trace(
                    request.task_id,
                    action=action,
                    need_images=request.capture_visuals,
                    capture_visuals=request.capture_visuals,
                    output_dir=attempt_root,
                    video_fps=int(request.benchmark_kwargs.get("video_fps", 10)),
                    **request.adapter_kwargs,
                )
                trace_path = attempt_root / "trace.json"
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                write_trace(trace_path, trace)
                trace_id = trace.trace_id
                terminal_status = trace.outcome.terminal_status or terminal_status
                success = bool(trace.outcome.success)
                metrics = dict(trace.outcome.metrics)
                trace_description, evidence_status, feedback_details = _describe_trace_record_detail(
                    trace=trace,
                    trace_path=trace_path,
                    request=request,
                    describer=describer,
                    pattern_library=pattern_library,
                    env_id="phyre",
                    task_id=request.task_id,
                    index=attempt_index,
                    record_kind="attempt",
                )
                feedback_trace_text = str(feedback_details.get("trace_text") or "").strip()
                feedback_pattern_text = str(feedback_details.get("pattern_text") or "").strip()
                if cell is not None and radius is not None:
                    seen_actions.append((int(cell), int(radius)))
            else:
                parsed_action = {
                    "cell": parsed.get("parsed_cell"),
                    "radius_size": parsed.get("parsed_radius_size"),
                    "action": None,
                    **(
                        {"format_repair_output": repaired_output, "format_repair_used": True}
                        if repaired_output is not None
                        else {}
                    ),
                }
                trace_description = "No simulation was run because the action format was invalid."
                evidence_status = None
        except DeepPHYAttemptFailure as exc:
            parsed_action = {
                "cell": parsed.get("parsed_cell"),
                "radius_size": parsed.get("parsed_radius_size"),
                "action": parsed.get("action").tolist() if parsed.get("action") is not None else None,
            }
            terminal_status = exc.terminal_status
            success = False
            metrics = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                **exc.native_metrics,
            }
            trace_description = f"Attempt failed before simulation: {exc}"
            evidence_status = None
        except Exception as exc:
            parsed_action = {
                "cell": parsed.get("parsed_cell"),
                "radius_size": parsed.get("parsed_radius_size"),
                "action": parsed.get("action").tolist() if parsed.get("action") is not None else None,
            }
            terminal_status = "MODEL_REQUEST_ERROR"
            success = False
            metrics = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            trace_description = f"Attempt failed before simulation: {exc}"
            evidence_status = None

        base_attempt_data = {
            "attempt_number": attempt_index,
            "simulation_status": terminal_status,
            "parsed_cell": parsed.get("parsed_cell"),
            "parsed_radius_size": parsed.get("parsed_radius_size"),
            "trace_summary": trace_description,
            "outcome_summary": _summarize_attempt_outcome(trace_description),
            "feedback_trace_text": feedback_trace_text or trace_description,
            "feedback_pattern_text": feedback_pattern_text,
        }
        attempt_history.append(strategy.update_attempt_data(base_attempt_data, parsed))
        records.append(
            DeepPHYTextRecord(
                index=attempt_index,
                record_kind="attempt",
                prompt=prompt,
                raw_model_output=raw_output,
                parsed_action=parsed_action,
                trace_description=trace_description,
                evidence_status=evidence_status,
                trace_path=str(trace_path) if trace_path is not None else None,
                trace_id=trace_id,
                success=success,
                terminal_status=terminal_status,
                native_metrics=metrics,
                router=_responder_router_info(responder),
                duration_seconds=max(time.perf_counter() - attempt_started_at, 0.0),
            )
        )
        _write_partial_task_result(
            run_id=run_id,
            request=request,
            output_root=output_root,
            initial_image_path=initial_image_path,
            records=records,
        )
        if success:
            break

    return _build_result(
        run_id=run_id,
        request=request,
        output_root=output_root,
        initial_image_path=initial_image_path,
        records=records,
    )


def _evaluate_iphyre(
    *,
    request: DeepPHYTextEvaluationRequest,
    run_id: str,
    output_root: Path,
    adapter: Any,
    responder: ModelResponder,
    describer: TraceDescriber,
    pattern_library: PatternLibrary | None,
) -> DeepPHYTextEvaluationResult:
    upstream = adapter.load_upstream_config()
    max_attempts = int(request.benchmark_kwargs.get("max_attempts", upstream.get("max_attempts", 10)))
    runtime = adapter._load_runtime_settings()
    solved_threshold = float(request.benchmark_kwargs.get("solved_threshold", runtime["solved_threshold"]))
    initial_image_path, initial_meta = _normalize_initial_image_payload(
        _prepare_initial_image("iphyre", adapter, request, output_root, extra={"runtime": runtime})
    )
    attempt_history: list[dict[str, Any]] = []
    records: list[DeepPHYTextRecord] = []
    trace_description = "No prior simulation trace description is available yet."
    num_eliminable_blocks = int(initial_meta.get("num_eliminable_blocks", 0))
    seen_action_sequences: list[tuple[tuple[int, float], ...]] = []

    for attempt_index in range(1, max_attempts + 1):
        attempt_started_at = time.perf_counter()
        prompt, strategy = build_iphyre_prompt(
            project_root=adapter.resolve_project_path(""),
            prompt_format=request.prompt_format,
            task_id=request.task_id,
            initial_image_path=initial_image_path,
            image_inputs=list(initial_meta.get("prompt_image_inputs", [])),
            trace_description=trace_description,
            description_mode=request.description_mode,
            attempt_index=attempt_index,
            attempt_history=attempt_history,
            solved_threshold=solved_threshold,
            num_eliminable_blocks=num_eliminable_blocks,
        )
        raw_output = ""
        parsed: dict[str, Any] = {}
        parsed_actions: list[dict[str, Any]] = []
        repaired_output = None
        trace = None
        trace_path = None
        trace_id = None
        feedback_trace_text = ""
        feedback_pattern_text = ""
        terminal_status = "INVALID_ACTION_FORMAT"
        success = False
        metrics: dict[str, Any] = {}
        try:
            raw_output = responder.respond(prompt)
            parsed = strategy.parse_response(raw_output)
            coerced = _coerce_json_action(env_id="iphyre", raw_output=raw_output)
            if request.prompt_format == "VLA" and coerced is not None:
                raw_output = coerced
                parsed = strategy.parse_response(raw_output)
            elif request.prompt_format == "WM" and coerced is not None and not (parsed.get("parsed_actions") or []):
                parsed = strategy.parse_response(
                    _build_wm_action_response(
                        prediction=parsed.get("prediction"),
                        action_text=coerced,
                        multiline_action=True,
                    )
                )
            parsed_actions = parsed.get("parsed_actions") or []
            if not parsed_actions:
                repaired_output = _repair_iphyre_action_output(
                    responder=responder,
                    prompt=prompt,
                    raw_output=raw_output,
                )
                if repaired_output is not None:
                    repaired_parsed = strategy.parse_response(repaired_output)
                    repaired_actions = repaired_parsed.get("parsed_actions") or []
                    if repaired_actions:
                        parsed = repaired_parsed
                        parsed_actions = repaired_actions
            if parsed_actions:
                duplicate_reason = _iphyre_duplicate_reason(
                    parsed_actions=parsed_actions,
                    seen_action_sequences=seen_action_sequences,
                )
                if duplicate_reason is not None:
                    terminal_status = "REPEATED_ACTION"
                    trace_description = duplicate_reason
                    metrics = {"duplicate_reason": duplicate_reason}
                    evidence_status = None
                else:
                    attempt_root = output_root / f"attempt_{attempt_index:03d}"
                    trace = adapter.generate_trace(
                        request.task_id,
                        actions=parsed_actions,
                        simulation_fps=runtime["simulation_fps"],
                        trace_fps=runtime["trace_fps"],
                        video_fps=runtime["video_fps"],
                        solved_threshold=solved_threshold,
                        capture_visuals=request.capture_visuals,
                        output_dir=attempt_root,
                        **request.adapter_kwargs,
                    )
                    trace_path = attempt_root / "trace.json"
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    write_trace(trace_path, trace)
                    trace_id = trace.trace_id
                    terminal_status = trace.outcome.terminal_status or terminal_status
                    success = bool(trace.outcome.success)
                    metrics = dict(trace.outcome.metrics)
                    trace_description, evidence_status, feedback_details = _describe_trace_record_detail(
                        trace=trace,
                        trace_path=trace_path,
                        request=request,
                        describer=describer,
                        pattern_library=pattern_library,
                        env_id="iphyre",
                        task_id=request.task_id,
                        index=attempt_index,
                        record_kind="attempt",
                    )
                    feedback_trace_text = str(feedback_details.get("trace_text") or "").strip()
                    feedback_pattern_text = str(feedback_details.get("pattern_text") or "").strip()
                    seen_action_sequences.append(_normalize_iphyre_actions(parsed_actions))
            else:
                trace_description = "Dummy trace description: no simulation was run because the action format was invalid."
                evidence_status = None
        except ValueError as exc:
            terminal_status = "INVALID_ACTION_FORMAT"
            success = False
            metrics = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            trace_description = f"Attempt failed before simulation: {exc}"
            evidence_status = None
        except Exception as exc:
            terminal_status = "MODEL_REQUEST_ERROR"
            success = False
            metrics = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            trace_description = f"Attempt failed before simulation: {exc}"
            evidence_status = None

        base_attempt_data = {
            "attempt_number": attempt_index,
            "vlm_response": raw_output,
            "sim_actions": metrics.get("translated_actions", []),
            "reward": str(metrics.get("total_reward", -1)),
            "simulation_frames": [],
            "simulation_status": terminal_status,
            "trace_summary": _summarize_attempt_outcome(trace_description),
            "outcome_summary": _summarize_attempt_outcome(trace_description),
            "feedback_trace_text": feedback_trace_text or trace_description,
            "feedback_pattern_text": feedback_pattern_text,
        }
        attempt_history.append(strategy.update_attempt_data(base_attempt_data, parsed))
        records.append(
            DeepPHYTextRecord(
                index=attempt_index,
                record_kind="attempt",
                prompt=prompt,
                raw_model_output=raw_output,
                parsed_action={
                    "parsed_actions": parsed_actions,
                    "prediction": parsed.get("prediction"),
                    **(
                        {"format_repair_output": repaired_output, "format_repair_used": True}
                        if repaired_output is not None
                        else {}
                    ),
                },
                trace_description=trace_description,
                evidence_status=evidence_status,
                trace_path=str(trace_path) if trace_path is not None else None,
                trace_id=trace_id,
                success=success,
                terminal_status=terminal_status,
                native_metrics=metrics,
                router=_responder_router_info(responder),
                duration_seconds=max(time.perf_counter() - attempt_started_at, 0.0),
            )
        )
        _write_partial_task_result(
            run_id=run_id,
            request=request,
            output_root=output_root,
            initial_image_path=initial_image_path,
            records=records,
        )
        if success:
            break

    return _build_result(
        run_id=run_id,
        request=request,
        output_root=output_root,
        initial_image_path=initial_image_path,
        records=records,
    )


def _evaluate_pooltool(
    *,
    request: DeepPHYTextEvaluationRequest,
    run_id: str,
    output_root: Path,
    adapter: Any,
    responder: ModelResponder,
    describer: TraceDescriber,
    pattern_library: PatternLibrary | None,
) -> DeepPHYTextEvaluationResult:
    settings = adapter._load_runtime_settings()
    max_attempts = int(request.benchmark_kwargs.get("max_attempts", settings["max_attempts"]))
    initial_image_path, initial_meta = _normalize_initial_image_payload(
        _prepare_initial_image("pooltool", adapter, request, output_root)
    )
    attempt_history: list[dict[str, Any]] = []
    records: list[DeepPHYTextRecord] = []
    trace_description = "No prior simulation trace description is available yet."
    applied_shots: list[dict[str, str]] = []

    for attempt_index in range(1, max_attempts + 1):
        attempt_started_at = time.perf_counter()
        prompt, strategy = build_pooltool_prompt(
            project_root=adapter.resolve_project_path(""),
            prompt_format=request.prompt_format,
            task_id=request.task_id,
            initial_image_path=initial_image_path,
            image_inputs=list(initial_meta.get("prompt_image_inputs", [])),
            trace_description=trace_description,
            description_mode=request.description_mode,
            attempt_index=attempt_index,
            attempt_history=attempt_history,
        )
        raw_output = ""
        parsed: dict[str, Any] = {}
        chosen_speed_name = None
        chosen_strikespot_name = None
        repaired_output = None
        trace = None
        trace_path = None
        trace_id = None
        terminal_status = "INVALID_VLM_ACTION"
        success = False
        metrics: dict[str, Any] = {}
        try:
            raw_output = responder.respond(prompt)
            parsed = strategy.parse_response(raw_output)
            coerced = _coerce_json_action(env_id="pooltool", raw_output=raw_output)
            if request.prompt_format == "VLA" and coerced is not None:
                raw_output = coerced
                parsed = strategy.parse_response(raw_output)
            elif request.prompt_format == "WM" and coerced is not None and not (
                parsed.get("parsed_speed_name") and parsed.get("parsed_strikespot_name")
            ):
                parsed = strategy.parse_response(
                    _build_wm_action_response(
                        prediction=parsed.get("prediction"),
                        action_text=coerced,
                    )
                )
            chosen_speed_name = parsed.get("parsed_speed_name")
            chosen_strikespot_name = parsed.get("parsed_strikespot_name")
            if not (chosen_speed_name and chosen_strikespot_name):
                repaired_output = _repair_pooltool_action_output(
                    responder=responder,
                    prompt=prompt,
                    raw_output=raw_output,
                )
                if repaired_output is not None:
                    repaired_parsed = strategy.parse_response(repaired_output)
                    repaired_speed_name = repaired_parsed.get("parsed_speed_name")
                    repaired_strikespot_name = repaired_parsed.get("parsed_strikespot_name")
                    if repaired_speed_name and repaired_strikespot_name:
                        parsed = repaired_parsed
                        chosen_speed_name = repaired_speed_name
                        chosen_strikespot_name = repaired_strikespot_name
            if chosen_speed_name and chosen_strikespot_name:
                attempt_root = output_root / f"attempt_{attempt_index:03d}"
                candidate_shot = {
                    "speed": chosen_speed_name,
                    "strikespot": chosen_strikespot_name,
                }
                trace = adapter.generate_trace(
                    request.task_id,
                    shots=applied_shots + [candidate_shot],
                    capture_frames=request.capture_visuals,
                    output_dir=attempt_root,
                    **request.adapter_kwargs,
                )
                trace_path = attempt_root / "trace.json"
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                write_trace(trace_path, trace)
                trace_id = trace.trace_id
                terminal_status = trace.outcome.terminal_status or terminal_status
                success = bool(trace.outcome.success)
                metrics = dict(trace.outcome.metrics)
                trace_description, evidence_status = _describe_trace_record(
                    trace=trace,
                    trace_path=trace_path,
                    request=request,
                    describer=describer,
                    pattern_library=pattern_library,
                    env_id="pooltool",
                    task_id=request.task_id,
                    index=attempt_index,
                    record_kind="attempt",
                )
                if terminal_status not in {
                    "SIMULATION_CRASHED_ROOT_SOLVER_ISSUE",
                    "UNEXPECTED_SIMULATION_ERROR",
                    "INVALID_VLM_ACTION",
                }:
                    applied_shots.append(candidate_shot)
            else:
                trace_description = "Dummy trace description: no simulation was run because the action format was invalid."
                evidence_status = None
        except Exception as exc:
            terminal_status = "MODEL_REQUEST_ERROR"
            success = False
            metrics = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            trace_description = f"Attempt failed before simulation: {exc}"
            evidence_status = None

        parsed_action = {
            "speed": chosen_speed_name,
            "strikespot": chosen_strikespot_name,
            "prediction": parsed.get("prediction"),
            **(
                {"format_repair_output": repaired_output, "format_repair_used": True}
                if repaired_output is not None
                else {}
            ),
        }

        base_attempt_data = {
            "attempt_number": attempt_index,
            "vlm_response": raw_output,
            "chosen_speed_name": chosen_speed_name,
            "chosen_strikespot_name": chosen_strikespot_name,
            "parsed_action_params": parsed.get("action"),
            "simulation_outcome": terminal_status,
            "potted_balls_in_this_run": [],
            "is_game_solved_in_this_attempt": success,
        }
        attempt_history.append(strategy.update_attempt_data(base_attempt_data, parsed))
        records.append(
            DeepPHYTextRecord(
                index=attempt_index,
                record_kind="attempt",
                prompt=prompt,
                raw_model_output=raw_output,
                parsed_action=parsed_action,
                trace_description=trace_description,
                evidence_status=evidence_status,
                trace_path=str(trace_path) if trace_path is not None else None,
                trace_id=trace_id,
                success=success,
                terminal_status=terminal_status,
                native_metrics=metrics,
                router=_responder_router_info(responder),
                duration_seconds=max(time.perf_counter() - attempt_started_at, 0.0),
            )
        )
        _write_partial_task_result(
            run_id=run_id,
            request=request,
            output_root=output_root,
            initial_image_path=initial_image_path,
            records=records,
        )
        if success or terminal_status == "NO_TARGET_BALLS_LEFT":
            break

    return _build_result(
        run_id=run_id,
        request=request,
        output_root=output_root,
        initial_image_path=initial_image_path,
        records=records,
    )


def _evaluate_kinetix(
    *,
    request: DeepPHYTextEvaluationRequest,
    run_id: str,
    output_root: Path,
    adapter: Any,
    responder: ModelResponder,
    describer: TraceDescriber,
    pattern_library: PatternLibrary | None,
) -> DeepPHYTextEvaluationResult:
    settings = adapter._load_runtime_settings()
    repeat_count = int(request.benchmark_kwargs.get("repeat_count", settings["repeat_actions"]))
    annotation_mode = str(request.benchmark_kwargs.get("annotation_mode", "anno"))
    initial_image_path, meta = _normalize_initial_image_payload(
        _prepare_initial_image("kinetix", adapter, request, output_root, extra={"annotation_mode": annotation_mode})
    )
    num_active_joints = int(meta["num_active_joints"])
    num_active_thrusters = int(meta["num_active_thrusters"])
    max_steps = int(request.benchmark_kwargs.get("max_steps", meta["max_steps"]))
    history_length = int(request.benchmark_kwargs.get("history_length", 5))

    records: list[DeepPHYTextRecord] = []
    trace_description = "No prior simulation trace description is available yet."
    action_history: list[np.ndarray] = []
    applied_actions: list[list[int]] = []
    distance_history: list[float] = []
    last_action = np.zeros(6, dtype=np.int32)

    for step_index in range(1, max_steps + 1):
        step_started_at = time.perf_counter()
        prompt = build_kinetix_prompt(
            project_root=adapter.resolve_project_path(""),
            prompt_format=request.prompt_format,
            task_id=request.task_id,
            initial_image_path=initial_image_path,
            image_inputs=list(meta.get("prompt_image_inputs", [])),
            trace_description=trace_description,
            description_mode=request.description_mode,
            step_index=step_index,
            num_active_joints=num_active_joints,
            num_active_thrusters=num_active_thrusters,
            previous_actions=action_history[-history_length:],
            distances=distance_history[-history_length:],
            last_action=last_action,
            is_annotated=annotation_mode == "anno",
        )
        raw_output = responder.respond(prompt)
        parsed = _parse_kinetix_response(
            raw_output,
            prompt_format=request.prompt_format,
            num_active_joints=num_active_joints,
            num_active_thrusters=num_active_thrusters,
        )
        coerced = _coerce_json_action(env_id="kinetix", raw_output=raw_output)
        if request.prompt_format == "VLA" and coerced is not None:
            raw_output = coerced
            parsed = _parse_kinetix_response(
                raw_output,
                prompt_format=request.prompt_format,
                num_active_joints=num_active_joints,
                num_active_thrusters=num_active_thrusters,
            )
        elif request.prompt_format == "WM" and coerced is not None and not parsed.get("action_format_valid", False):
            parsed = _parse_kinetix_response(
                _build_wm_action_response(
                    prediction=parsed.get("prediction"),
                    action_text=coerced,
                ),
                prompt_format=request.prompt_format,
                num_active_joints=num_active_joints,
                num_active_thrusters=num_active_thrusters,
            )
        repaired_output = None
        if not parsed.get("action_format_valid", False):
            repaired_output = _repair_kinetix_action_output(
                responder=responder,
                prompt=prompt,
                raw_output=raw_output,
                num_active_joints=num_active_joints,
                num_active_thrusters=num_active_thrusters,
            )
            if repaired_output is not None:
                repaired_parsed = _parse_kinetix_response(
                    repaired_output,
                    prompt_format=request.prompt_format,
                    num_active_joints=num_active_joints,
                    num_active_thrusters=num_active_thrusters,
                )
                if repaired_parsed.get("action_format_valid", False):
                    parsed = repaired_parsed
        padded_action = parsed["padded_action"]
        applied_actions.append(padded_action)
        step_root = output_root / f"step_{step_index:03d}"
        trace = adapter.generate_trace(
            request.task_id,
            action_sequence=applied_actions,
            repeat_count=repeat_count,
            annotation_mode=annotation_mode,
            capture_frames=request.capture_visuals,
            output_dir=step_root,
            **request.adapter_kwargs,
        )
        trace_path = step_root / "trace.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        write_trace(trace_path, trace)
        trace_description, evidence_status = _describe_trace_record(
            trace=trace,
            trace_path=trace_path,
            request=request,
            describer=describer,
            pattern_library=pattern_library,
            env_id="kinetix",
            task_id=request.task_id,
            index=step_index,
            record_kind="step",
        )
        distance = float(trace.outcome.metrics.get("distance", 0.0) or 0.0)
        action_array = np.array(padded_action, dtype=np.int32)
        action_history.append(action_array)
        distance_history.append(distance)
        last_action = action_array
        records.append(
            DeepPHYTextRecord(
                index=step_index,
                record_kind="step",
                prompt=prompt,
                raw_model_output=raw_output,
                parsed_action={
                    **parsed,
                    **(
                        {"format_repair_output": repaired_output, "format_repair_used": True}
                        if repaired_output is not None
                        else {}
                    ),
                },
                trace_description=trace_description,
                evidence_status=evidence_status,
                trace_path=str(trace_path),
                trace_id=trace.trace_id,
                success=trace.outcome.success,
                terminal_status=trace.outcome.terminal_status,
                native_metrics=dict(trace.outcome.metrics),
                router=_responder_router_info(responder),
                duration_seconds=max(time.perf_counter() - step_started_at, 0.0),
            )
        )
        _write_partial_task_result(
            run_id=run_id,
            request=request,
            output_root=output_root,
            initial_image_path=initial_image_path,
            records=records,
        )
        if trace.outcome.success:
            break

    return _build_result(
        run_id=run_id,
        request=request,
        output_root=output_root,
        initial_image_path=initial_image_path,
        records=records,
    )


def _build_result(
    *,
    run_id: str,
    request: DeepPHYTextEvaluationRequest,
    output_root: Path,
    initial_image_path: str,
    records: list[DeepPHYTextRecord],
) -> DeepPHYTextEvaluationResult:
    final_record = records[-1] if records else None
    total_duration_seconds = sum(
        float(record.duration_seconds)
        for record in records
        if record.duration_seconds is not None
    )
    completed_at = datetime.now(timezone.utc).isoformat()
    return DeepPHYTextEvaluationResult(
        run_id=run_id,
        env_id=request.env_id,
        task_id=request.task_id,
        model_name=request.model_name,
        prompt_format=request.prompt_format,
        description_mode=request.description_mode,
        initial_image_path=initial_image_path,
        pattern_run_dir=request.pattern_run_dir,
        pattern_library_path=request.pattern_library_path,
        records=records,
        success=final_record.success if final_record is not None else None,
        terminal_status=final_record.terminal_status if final_record is not None else None,
        native_metrics=final_record.native_metrics if final_record is not None else {},
        final_trace_path=final_record.trace_path if final_record is not None else None,
        output_root=str(output_root),
        report_root=request.report_root,
        total_duration_seconds=total_duration_seconds if records else None,
        completed_at=completed_at,
    )


def _phyre_duplicate_reason(*, cell: int, radius: int, seen_actions: list[tuple[int, int]]) -> str | None:
    for prev_cell, prev_radius in seen_actions:
        if prev_cell == cell and prev_radius == radius:
            return f"Repeated exact previous action (cell {cell}, radius {radius})."
        if abs(prev_cell - cell) <= 1 and prev_radius == radius:
            return f"Near-duplicate previous action in the same local region (cell {cell}, radius {radius})."
        if prev_cell == cell and abs(prev_radius - radius) <= 1:
            return f"Near-duplicate previous action with almost the same size (cell {cell}, radius {radius})."
    return None


def _normalize_iphyre_actions(parsed_actions: list[dict[str, Any]]) -> tuple[tuple[int, float], ...]:
    normalized: list[tuple[int, float]] = []
    for action in parsed_actions:
        if not isinstance(action, dict):
            continue
        index = int(action.get("index", -1))
        time_value = round(float(action.get("time", 0.0)), 2)
        normalized.append((index, time_value))
    return tuple(normalized)


def _iphyre_duplicate_reason(
    *,
    parsed_actions: list[dict[str, Any]],
    seen_action_sequences: list[tuple[tuple[int, float], ...]],
) -> str | None:
    normalized = _normalize_iphyre_actions(parsed_actions)
    if not normalized:
        return None
    for previous in seen_action_sequences:
        if previous == normalized:
            return f"Repeated exact previous removal sequence {list(normalized)}."
        if len(previous) != len(normalized):
            continue
        same_indices = all(prev_index == cur_index for (prev_index, _), (cur_index, _) in zip(previous, normalized))
        if same_indices and all(abs(prev_time - cur_time) <= 0.15 for (_, prev_time), (_, cur_time) in zip(previous, normalized)):
            return f"Near-duplicate previous removal sequence with the same block indices and nearly identical timing {list(normalized)}."
    return None


def _summarize_attempt_outcome(text: str) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= 180:
        return compact
    return compact[:177].rstrip() + "..."


def _build_model_responder(request: DeepPHYTextEvaluationRequest) -> ModelResponder:
    if request.model_name.lower() == "mock":
        return MockModelResponder(request.model_name)
    shared_llm = load_shared_llm_config()
    base_url = str(
        request.benchmark_kwargs.get("base_url")
        or request.benchmark_kwargs.get("ollama_base_url")
        or shared_llm.base_url
    )
    timeout = int(request.benchmark_kwargs.get("timeout", 120))
    max_tokens = int(
        request.benchmark_kwargs.get("max_tokens", GLOBAL_MIN_COMPLETION_TOKENS)
    )
    api_key = str(request.benchmark_kwargs.get("api_key") or "").strip() or None
    return ChatCompletionsDeepPHYResponder(
        model_name=request.model_name,
        base_url=base_url,
        timeout=timeout,
        max_tokens=max_tokens,
        api_key=api_key,
    )


def _load_pattern_library_for_request(
    request: DeepPHYTextEvaluationRequest,
) -> PatternLibrary | None:
    if request.description_mode == "trace_text":
        return None
    library_path = request.pattern_library_path
    if not library_path and request.pattern_run_dir:
        library_path = str(resolve_pattern_library_path(request.pattern_run_dir))
    if not library_path:
        raise ValueError("pattern_library_path is required when description_mode uses patterns")
    return load_pattern_library(library_path)


def _describe_trace_record(
    *,
    trace: CanonicalTrace,
    trace_path: Path,
    request: DeepPHYTextEvaluationRequest,
    describer: TraceDescriber,
    pattern_library: PatternLibrary | None,
    env_id: str,
    task_id: str,
    index: int,
    record_kind: str,
) -> tuple[str, str | None]:
    description, evidence_status, _feedback_details = _describe_trace_record_detail(
        trace=trace,
        trace_path=trace_path,
        request=request,
        describer=describer,
        pattern_library=pattern_library,
        env_id=env_id,
        task_id=task_id,
        index=index,
        record_kind=record_kind,
    )
    return description, evidence_status


def _describe_trace_record_detail(
    *,
    trace: CanonicalTrace,
    trace_path: Path,
    request: DeepPHYTextEvaluationRequest,
    describer: TraceDescriber,
    pattern_library: PatternLibrary | None,
    env_id: str,
    task_id: str,
    index: int,
    record_kind: str,
) -> tuple[str, str | None, dict[str, str | None]]:
    evidence = None
    evidence_status = None
    trace_text = describer.describe(
        trace,
        env_id=env_id,
        task_id=task_id,
        index=index,
        record_kind=record_kind,
        description_mode="trace_text",
        evidence=None,
    )
    pattern_text = None
    if request.description_mode != "trace_text":
        if pattern_library is None:
            raise ValueError("pattern_library must be loaded for pattern description modes")
        patterns = apply_pattern_library(trace, pattern_library)
        evidence_payload = build_pattern_evidence(pattern_library, patterns=patterns)
        evidence = PatternEvidence.model_validate(evidence_payload["pattern_annotations"])
        write_pattern_evidence(
            trace_path.parent,
            pattern_library,
            patterns=patterns,
            evidence_relpath=request.evidence_relpath or "trace.evidence.json",
        )
        evidence_status = "present"
        pattern_text = describer.describe(
            trace,
            env_id=env_id,
            task_id=task_id,
            index=index,
            record_kind=record_kind,
            description_mode="pattern_text",
            evidence=evidence,
        )
    description = describer.describe(
        trace,
        env_id=env_id,
        task_id=task_id,
        index=index,
        record_kind=record_kind,
        description_mode=request.description_mode,
        evidence=evidence,
    )
    return (
        description,
        evidence_status,
        {"trace_text": trace_text, "pattern_text": pattern_text},
    )


def _prepare_initial_image(
    env_id: str,
    adapter: Any,
    request: DeepPHYTextEvaluationRequest,
    output_root: Path,
    *,
    extra: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]] | Path:
    extra = dict(extra or {})
    initial_dir = output_root / "initial"
    initial_dir.mkdir(parents=True, exist_ok=True)

    if env_id == "phyre":
        grid_size = (8, 8)
        if hasattr(adapter, "load_upstream_config"):
            upstream_config = adapter.load_upstream_config()
            raw_grid_size = upstream_config.get("grid_size") if isinstance(upstream_config, dict) else None
            if (
                isinstance(raw_grid_size, (list, tuple))
                and len(raw_grid_size) == 2
                and all(isinstance(value, (int, float)) for value in raw_grid_size)
            ):
                grid_size = (int(raw_grid_size[0]), int(raw_grid_size[1]))
        context = adapter._load_simulator_context(
            eval_setup=str(request.adapter_kwargs.get("eval_setup", adapter.DEFAULT_EVAL_SETUP)),
            fold_id=int(request.adapter_kwargs.get("fold_id", adapter.DEFAULT_FOLD_ID)),
            eval_type=str(request.adapter_kwargs.get("eval_type", adapter.DEFAULT_EVAL_TYPE)),
        )
        task_index = context["task_ids"].index(request.task_id)
        initial_scene = context["initial_scenes"][task_index]
        phyre_module = adapter._load_phyre_module()
        pixels = phyre_module.observations_to_uint8_rgb(initial_scene)
        raw_path = initial_dir / "initial_scene.png"
        write_image_artifact(pixels, raw_path, max_side=512)
        gridded_path = Path(
            add_grid_with_matplotlib(
                str(raw_path),
                grid_size,
            )
            or raw_path
        )
        grid_label = f"Initial scene with {grid_size[0]}x{grid_size[1]} grid"
        return gridded_path, {
            "prompt_image_inputs": [
                DeepPHYTextPromptImage(path=str(raw_path), label="Initial scene"),
                DeepPHYTextPromptImage(path=str(gridded_path), label=grid_label),
            ]
        }

    if env_id == "iphyre":
        runtime = extra.get("runtime") or adapter._load_runtime_settings()
        if hasattr(adapter, "_configure_runtime_environment"):
            adapter._configure_runtime_environment(capture_visuals=False)
        env = adapter._create_env(request.task_id, fps=runtime["simulation_fps"])
        sequence_root = initial_dir / f"{request.task_id}_initial"
        if hasattr(env, "collect_initial_data"):
            env.collect_initial_data(save_path=str(sequence_root))
        frames = sorted(sequence_root.rglob("*.png"))
        if not frames:
            fallback = initial_dir / "initial_scene.png"
            Image.new("RGB", (512, 512), color="white").save(fallback)
            return fallback, {
                "num_eliminable_blocks": 0,
                "prompt_image_inputs": [DeepPHYTextPromptImage(path=str(fallback), label="Initial scene")],
            }
        action_space = list(env.get_action_space()[1:])
        valid_action_space = [pos for pos in action_space if list(pos) != [0.0, 0.0]]
        return frames[0], {
            "num_eliminable_blocks": len(valid_action_space),
            "prompt_image_inputs": [DeepPHYTextPromptImage(path=str(frames[0]), label="Initial scene")],
        }

    if env_id == "kinetix":
        annotation_mode = str(extra.get("annotation_mode", "anno"))
        context = adapter._load_level_context(request.task_id, annotation_mode=annotation_mode, render_pixel_size=512)
        attachments = adapter._render_attachments(
            request.task_id,
            index=0,
            renderer=context["renderer"],
            env_state=context["env_state"],
            annotation_mode=annotation_mode,
            output_dir=initial_dir,
            media_max_side=512,
        )
        if attachments:
            path = Path(attachments[0].path)
        else:
            rendered_files = sorted(initial_dir.glob(f"{request.task_id.replace('/', '__')}_frame_000_*"))
            if not rendered_files:
                raise RuntimeError(f"Kinetix initial render did not produce any files for task {request.task_id}")
            path = rendered_files[0]
        prompt_image_inputs = [DeepPHYTextPromptImage(path=str(path), label="Initial scene")]
        if annotation_mode == "anno":
            unannotated_context = adapter._load_level_context(
                request.task_id,
                annotation_mode="unanno",
                render_pixel_size=512,
            )
            unannotated_attachments = adapter._render_attachments(
                request.task_id,
                index=0,
                renderer=unannotated_context["renderer"],
                env_state=unannotated_context["env_state"],
                annotation_mode="unanno",
                output_dir=initial_dir,
                media_max_side=512,
            )
            if unannotated_attachments:
                unannotated_path = Path(unannotated_attachments[0].path)
            else:
                unannotated_files = sorted(initial_dir.glob(f"{request.task_id.replace('/', '__')}_frame_000_*rendered.png"))
                unannotated_path = unannotated_files[0] if unannotated_files else path
            annotated_files = sorted(initial_dir.glob(f"{request.task_id.replace('/', '__')}_frame_000_*annotated.png"))
            annotated_path = annotated_files[0] if annotated_files else path
            prompt_image_inputs = [
                DeepPHYTextPromptImage(path=str(unannotated_path), label="Current scene"),
                DeepPHYTextPromptImage(path=str(annotated_path), label="Annotated scene with motor and thruster labels"),
            ]
        max_steps = (int(context["env_params"].max_timesteps) + 1) // int(extra.get("repeat_count", adapter._load_runtime_settings()["repeat_actions"]))
        return path, {
            "num_active_joints": context["num_active_joints"],
            "num_active_thrusters": context["num_active_thrusters"],
            "max_steps": max_steps,
            "prompt_image_inputs": prompt_image_inputs,
        }

    if env_id == "pooltool":
        path = initial_dir / "initial_scene_overhead.png"
        adapter.render_initial_scene_image(path, view_name="overhead")
        return path, {
            "prompt_image_inputs": [DeepPHYTextPromptImage(path=str(path), label="Current table state")]
        }

    raise ValueError(f"unsupported environment: {env_id}")


def _build_multimodal_user_content(
    *,
    text: str,
    image_inputs: list[DeepPHYTextPromptImage],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for index, item in enumerate(image_inputs, start=1):
        image_path = Path(item.path)
        label = (item.label or f"Image {index}").strip()
        content.append({"type": "text", "text": f"{label}."})
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
        image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}
        )
    return content


def _normalize_initial_image_payload(
    payload: tuple[Path, dict[str, Any]] | Path,
) -> tuple[str, dict[str, Any]]:
    if isinstance(payload, tuple):
        path, meta = payload
        return str(path), dict(meta)
    return str(payload), {}


def _repair_phyre_action_output(
    *,
    responder: ModelResponder,
    prompt: DeepPHYTextPrompt,
    raw_output: str,
) -> str | None:
    if hasattr(responder, "repair_phyre_action"):
        return responder.repair_phyre_action(prompt=prompt, raw_output=raw_output)
    deterministic = _extract_repairable_phyre_cell(raw_output)
    if deterministic is not None:
        return deterministic
    repair_prompt = DeepPHYTextPrompt(
        env_id=prompt.env_id,
        task_id=prompt.task_id,
        record_kind=prompt.record_kind,
        index=prompt.index,
        prompt_format=prompt.prompt_format,
        description_mode=prompt.description_mode,
        system_prompt=(
            "You convert verbose PHYRE solver outputs into the exact required action format. "
            "Return JSON only: {\"cell\": <int 1-64>, \"radius\": <int 1-8>} and nothing else."
        ),
        user_prompt=(
            "Given the model draft below, produce the final action line only.\n\n"
            "Model draft:\n"
            f"{raw_output}\n\n"
            "If the draft already names a cell or radius, preserve them; otherwise choose radius 1. "
            "Return JSON only. Start with: {\"cell\":"
        ),
        initial_image_path=prompt.initial_image_path,
        image_inputs=prompt.image_inputs,
        trace_description=prompt.trace_description,
        metadata={**prompt.metadata, "repair_kind": "phyre_action_format"},
    )
    repaired_output = responder.respond(repair_prompt).strip()
    coerced = _coerce_json_action(env_id="phyre", raw_output=repaired_output)
    return coerced or repaired_output or None


def _repair_iphyre_action_output(
    *,
    responder: ModelResponder,
    prompt: DeepPHYTextPrompt,
    raw_output: str,
) -> str | None:
    system_prompt = (
        "Rewrite the draft into the exact I-PHYRE action format. "
        "Return only valid JSON: [{\"time\": <float 0-1>, \"index\": <int>}]. "
        "No markdown, no extra text."
    )
    user_prompt = (
        "Given the draft below, return only the final JSON action array.\n\n"
        f"Draft output:\n{raw_output}\n\n"
        "If multiple actions are present, keep them all in order. "
        "Return JSON only. Start with: ["
    )
    if hasattr(responder, "repair_action_output"):
        return responder.repair_action_output(
            prompt=prompt,
            raw_output=raw_output,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    repair_prompt = DeepPHYTextPrompt(
        env_id=prompt.env_id,
        task_id=prompt.task_id,
        record_kind=prompt.record_kind,
        index=prompt.index,
        prompt_format=prompt.prompt_format,
        description_mode=prompt.description_mode,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        initial_image_path=prompt.initial_image_path,
        image_inputs=prompt.image_inputs,
        trace_description=prompt.trace_description,
        metadata={**prompt.metadata, "repair_kind": "iphyre_action_format"},
    )
    repaired_output = responder.respond(repair_prompt).strip()
    coerced = _coerce_json_action(env_id="iphyre", raw_output=repaired_output)
    return coerced or repaired_output or None


def _repair_pooltool_action_output(
    *,
    responder: ModelResponder,
    prompt: DeepPHYTextPrompt,
    raw_output: str,
) -> str | None:
    system_prompt = (
        "Rewrite the draft into the exact PoolTool action format. "
        "Return exactly one line and nothing else: "
        "`Speed: <Speed Name>, Strikespot: <Strikespot Name>.`"
    )
    user_prompt = (
        "Given the draft below, return only the final PoolTool action line.\n\n"
        f"Draft output:\n{raw_output}\n\n"
        "Preserve any valid speed and strikespot names if they are already present. "
        "Return JSON only. Start with: {\"speed\":"
    )
    if hasattr(responder, "repair_action_output"):
        return responder.repair_action_output(
            prompt=prompt,
            raw_output=raw_output,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    repair_prompt = DeepPHYTextPrompt(
        env_id=prompt.env_id,
        task_id=prompt.task_id,
        record_kind=prompt.record_kind,
        index=prompt.index,
        prompt_format=prompt.prompt_format,
        description_mode=prompt.description_mode,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        initial_image_path=prompt.initial_image_path,
        image_inputs=prompt.image_inputs,
        trace_description=prompt.trace_description,
        metadata={**prompt.metadata, "repair_kind": "pooltool_action_format"},
    )
    repaired_output = responder.respond(repair_prompt).strip()
    coerced = _coerce_json_action(env_id="pooltool", raw_output=repaired_output)
    return coerced or repaired_output or None


def _repair_kinetix_action_output(
    *,
    responder: ModelResponder,
    prompt: DeepPHYTextPrompt,
    raw_output: str,
    num_active_joints: int,
    num_active_thrusters: int,
) -> str | None:
    total = num_active_joints + num_active_thrusters
    system_prompt = (
        "Rewrite the draft into the exact Kinetix action format. "
        "Return only the integer action vector list and nothing else, "
        f"with exactly {total} integers."
    )
    user_prompt = (
        "Given the draft below, return only the final integer action vector list.\n\n"
        f"Draft output:\n{raw_output}\n\n"
        f"The list must contain exactly {total} integers."
    )
    if hasattr(responder, "repair_action_output"):
        return responder.repair_action_output(
            prompt=prompt,
            raw_output=raw_output,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    repair_prompt = DeepPHYTextPrompt(
        env_id=prompt.env_id,
        task_id=prompt.task_id,
        record_kind=prompt.record_kind,
        index=prompt.index,
        prompt_format=prompt.prompt_format,
        description_mode=prompt.description_mode,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        initial_image_path=prompt.initial_image_path,
        image_inputs=prompt.image_inputs,
        trace_description=prompt.trace_description,
        metadata={**prompt.metadata, "repair_kind": "kinetix_action_format"},
    )
    repaired_output = responder.respond(repair_prompt).strip()
    coerced = _coerce_json_action(env_id="kinetix", raw_output=repaired_output)
    return coerced or repaired_output or None


def _extract_repairable_phyre_cell(raw_output: str) -> str | None:
    for pattern in (
        r"Cell\s*:\s*(\d{1,2})\s*,\s*Radius\s*:\s*(\d{1,2})",
        r"\bcell\s+(\d{1,2})\b\s*,?\s*radius\s*:?\s*(\d{1,2})",
        r"Cell\s*:\s*(\d{1,2})",
        r"\bcell\s+(\d{1,2})\b",
    ):
        match = re.search(pattern, raw_output, re.IGNORECASE)
        if match is None:
            continue
        cell = int(match.group(1))
        radius = (
            int(match.group(2))
            if match.lastindex is not None and match.lastindex >= 2 and match.group(2)
            else 1
        )
        cell = min(max(cell, 1), 64)
        radius = min(max(radius, 1), 8)
        return json.dumps({"cell": cell, "radius": radius})
    return None


def _build_wm_action_response(
    *,
    prediction: str | None,
    action_text: str,
    multiline_action: bool = False,
) -> str:
    prediction_text = (prediction or "N/A").strip() or "N/A"
    if multiline_action:
        return f"Prediction: {prediction_text}\nAction:\n{action_text}"
    return f"Prediction: {prediction_text}\nAction: {action_text}"


def _coerce_json_action(*, env_id: str, raw_output: str) -> str | None:
    import json
    candidates = []
    text = raw_output.strip()
    for start in range(len(text)):
        ch = text[start]
        if ch not in "[{":
            continue
        for end in range(start + 1, min(len(text), start + 12000)):
            if text[end] not in "]}":
                continue
            snippet = text[start : end + 1]
            try:
                obj = json.loads(snippet)
                candidates.append(obj)
                break
            except Exception:
                continue
    if not candidates:
        return None
    obj = candidates[-1]
    try:
        if env_id == "phyre":
            if isinstance(obj, dict) and "cell" in obj:
                cell = int(obj.get("cell"))
                radius = int(obj.get("radius", 1))
                cell = min(max(cell, 1), 64)
                radius = min(max(radius, 1), 8)
                return json.dumps({"cell": cell, "radius": radius})
        if env_id == "iphyre":
            if isinstance(obj, dict) and "actions" in obj:
                return json.dumps(obj["actions"])
        if env_id == "pooltool":
            if isinstance(obj, dict) and "speed" in obj and "strikespot" in obj:
                return f"Speed: {obj['speed']}, Strikespot: {obj['strikespot']}"
        if env_id == "kinetix":
            if isinstance(obj, dict) and isinstance(obj.get("action"), list):
                return json.dumps(obj["action"])
            if isinstance(obj, list):
                return json.dumps(obj)
    except Exception:
        return None
    return None


def _parse_kinetix_response(
    raw_output: str,
    *,
    prompt_format: str,
    num_active_joints: int,
    num_active_thrusters: int,
) -> dict[str, Any]:
    prediction = None
    action_text = raw_output
    if prompt_format == "WM":
        prediction_match = re.search(r"Prediction\s*:\s*(.*?)(?=\nAction\s*:|\Z)", raw_output, re.IGNORECASE | re.DOTALL)
        if prediction_match:
            prediction = prediction_match.group(1).strip()
        action_match = re.search(r"Action\s*:\s*(.*)", raw_output, re.IGNORECASE | re.DOTALL)
        if action_match:
            action_text = action_match.group(1).strip()
    parsed_action = _parse_kinetix_action_list(action_text)
    format_valid = parsed_action is not None
    if parsed_action is None:
        parsed_action = [0] * (num_active_joints + num_active_thrusters)
    joints = parsed_action[:num_active_joints]
    thrusters = parsed_action[num_active_joints:num_active_joints + num_active_thrusters]
    padded = list(joints) + [0] * (4 - num_active_joints) + list(thrusters) + [0] * (2 - num_active_thrusters)
    return {
        "prediction": prediction,
        "action_format_valid": format_valid,
        "parsed_action": parsed_action,
        "padded_action": padded[:6],
    }


def _parse_kinetix_action_list(action_text: str) -> list[int] | None:
    try:
        start_index = action_text.rfind("[")
        end_index = action_text.rfind("]")
        if start_index != -1 and end_index != -1 and start_index < end_index:
            payload = action_text[start_index:end_index + 1]
            return [int(value) for value in json.loads(payload)]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    cleaned = action_text.strip().strip("[]` ")
    if not cleaned:
        return None
    try:
        return [int(value.strip()) for value in cleaned.split(",") if value.strip()]
    except ValueError:
        return None
