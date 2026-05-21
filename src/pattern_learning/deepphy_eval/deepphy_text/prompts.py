from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np

from deepphy_eval.deepphy_text.models import (
    DeepPHYTextDescriptionMode,
    DeepPHYTextPrompt,
    DeepPHYTextPromptImage,
    DeepPHYTextPromptFormat,
    normalize_deepphy_prompt_format,
)

_PROMPT_MODULES = {
    "phyre": "vendor.deepphy_prompts.prompt_strategy_01_phyre",
    "iphyre": "vendor.deepphy_prompts.prompt_strategy_02_iphyre",
    "kinetix": "vendor.deepphy_prompts.prompt_strategy_03_kinetix",
    "pooltool": "vendor.deepphy_prompts.prompt_strategy_04_pooltool",
}


def _load_prompt_module(env_id: str) -> Any:
    return importlib.import_module(_PROMPT_MODULES[env_id])


def build_phyre_prompt(
    *,
    project_root: Path,
    prompt_format: DeepPHYTextPromptFormat,
    task_id: str,
    initial_image_path: str,
    image_inputs: list[DeepPHYTextPromptImage] | None,
    trace_description: str,
    description_mode: DeepPHYTextDescriptionMode,
    attempt_index: int,
    attempt_history: list[dict[str, Any]],
) -> tuple[DeepPHYTextPrompt, Any]:
    del project_root
    prompt_format = normalize_deepphy_prompt_format(prompt_format)
    module = _load_prompt_module("phyre")
    strategy = module.get_prompt_strategy(prompt_format)
    prompt_images = list(image_inputs or [DeepPHYTextPromptImage(path=initial_image_path, label="Initial scene")])
    system_prompt, user_prompt = strategy.generate_prompts(
        [],
        [{"path": item.path, "label": item.label or "Initial scene"} for item in prompt_images],
    )
    user_prompt = _augment_phyre_attempt_history(user_prompt, attempt_history)
    user_prompt = _normalize_vla_user_prompt(user_prompt, prompt_format)
    json_hint = _json_instruction("phyre")
    user_prompt = f"{json_hint}\n\n{user_prompt}\n\n{json_hint}"
    prompt = _wrap_prompt(
        env_id="phyre",
        task_id=task_id,
        record_kind="attempt",
        index=attempt_index,
        prompt_format=prompt_format,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        initial_image_path=initial_image_path,
        image_inputs=prompt_images,
        trace_description=trace_description,
        description_mode=description_mode,
        metadata={"attempt_number": attempt_index, "inline_feedback_block": True},
    )
    return prompt, strategy


def build_iphyre_prompt(
    *,
    project_root: Path,
    prompt_format: DeepPHYTextPromptFormat,
    task_id: str,
    initial_image_path: str,
    image_inputs: list[DeepPHYTextPromptImage] | None,
    trace_description: str,
    description_mode: DeepPHYTextDescriptionMode,
    attempt_index: int,
    attempt_history: list[dict[str, Any]],
    solved_threshold: float,
    num_eliminable_blocks: int,
) -> tuple[DeepPHYTextPrompt, Any]:
    del project_root
    prompt_format = normalize_deepphy_prompt_format(prompt_format)
    module = _load_prompt_module("iphyre")
    strategy = module.get_prompt_strategy(prompt_format, solved_threshold)
    prompt_images = list(image_inputs or [DeepPHYTextPromptImage(path=initial_image_path, label="Initial scene")])
    system_prompt, user_prompt = strategy.generate_prompts(
        [],
        [{"path": item.path, "label": item.label or "Initial scene"} for item in prompt_images],
        num_eliminable_blocks,
    )
    user_prompt = _augment_iphyre_attempt_history(user_prompt, attempt_history)
    user_prompt = _normalize_vla_user_prompt(user_prompt, prompt_format)
    json_hint = _json_instruction("iphyre")
    user_prompt = f"{json_hint}\n\n{user_prompt}\n\n{json_hint}"
    prompt = _wrap_prompt(
        env_id="iphyre",
        task_id=task_id,
        record_kind="attempt",
        index=attempt_index,
        prompt_format=prompt_format,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        initial_image_path=initial_image_path,
        image_inputs=prompt_images,
        trace_description=trace_description,
        description_mode=description_mode,
        metadata={
            "attempt_number": attempt_index,
            "solved_threshold": solved_threshold,
            "num_eliminable_blocks": num_eliminable_blocks,
            "inline_feedback_block": True,
        },
    )
    return prompt, strategy


def build_pooltool_prompt(
    *,
    project_root: Path,
    prompt_format: DeepPHYTextPromptFormat,
    task_id: str,
    initial_image_path: str,
    image_inputs: list[DeepPHYTextPromptImage] | None,
    trace_description: str,
    description_mode: DeepPHYTextDescriptionMode,
    attempt_index: int,
    attempt_history: list[dict[str, Any]],
) -> tuple[DeepPHYTextPrompt, Any]:
    del project_root
    prompt_format = normalize_deepphy_prompt_format(prompt_format)
    module = _load_prompt_module("pooltool")
    strategy = module.get_prompt_strategy(prompt_format)
    prompt_images = list(
        image_inputs
        or ([DeepPHYTextPromptImage(path=initial_image_path, label="Initial table state")] if initial_image_path else [])
    )
    system_prompt, user_prompt = strategy.generate_prompts(
        attempt_history,
        [{"path": item.path, "label": item.label or "Initial table state"} for item in prompt_images],
    )
    user_prompt = _normalize_vla_user_prompt(user_prompt, prompt_format)
    json_hint = _json_instruction("pooltool")
    user_prompt = f"{json_hint}\n\n{user_prompt}\n\n{json_hint}"
    prompt = _wrap_prompt(
        env_id="pooltool",
        task_id=task_id,
        record_kind="attempt",
        index=attempt_index,
        prompt_format=prompt_format,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        initial_image_path=initial_image_path,
        image_inputs=prompt_images,
        trace_description=trace_description,
        description_mode=description_mode,
        metadata={"attempt_number": attempt_index},
    )
    return prompt, strategy


def build_kinetix_prompt(
    *,
    project_root: Path,
    prompt_format: DeepPHYTextPromptFormat,
    task_id: str,
    initial_image_path: str,
    image_inputs: list[DeepPHYTextPromptImage] | None,
    trace_description: str,
    description_mode: DeepPHYTextDescriptionMode,
    step_index: int,
    num_active_joints: int,
    num_active_thrusters: int,
    previous_actions: list[np.ndarray],
    distances: list[float],
    last_action: np.ndarray,
    is_annotated: bool,
) -> DeepPHYTextPrompt:
    del project_root
    prompt_format = normalize_deepphy_prompt_format(prompt_format)
    module = _load_prompt_module("kinetix")
    prompt_images = list(image_inputs or [DeepPHYTextPromptImage(path=initial_image_path, label="Initial scene")])
    current_image_paths = {"unannotated": initial_image_path, "annotated": initial_image_path}
    for item in prompt_images:
        label = (item.label or "").lower()
        if "annotated" in label or "labeled" in label:
            current_image_paths["annotated"] = item.path
        else:
            current_image_paths["unannotated"] = item.path
    system_prompt, user_prompt, _ = module.generate_prompts_and_images(
        num_active_joints,
        num_active_thrusters,
        previous_actions,
        distances,
        [],
        step_index,
        current_image_paths,
        last_action,
        prompt_format,
        is_annotated,
    )
    json_hint = _json_instruction("kinetix")
    user_prompt = f"{json_hint}\n\n{user_prompt}\n\n{json_hint}"
    return _wrap_prompt(
        env_id="kinetix",
        task_id=task_id,
        record_kind="step",
        index=step_index,
        prompt_format=prompt_format,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        initial_image_path=initial_image_path,
        image_inputs=prompt_images,
        trace_description=trace_description,
        description_mode=description_mode,
        metadata={
            "step_index": step_index,
            "num_active_joints": num_active_joints,
            "num_active_thrusters": num_active_thrusters,
        },
    )


def _wrap_prompt(
    *,
    env_id: str,
    task_id: str,
    record_kind: str,
    index: int,
    prompt_format: DeepPHYTextPromptFormat,
    system_prompt: str,
    user_prompt: str,
    initial_image_path: str,
    image_inputs: list[DeepPHYTextPromptImage],
    trace_description: str,
    description_mode: DeepPHYTextDescriptionMode,
    metadata: dict[str, Any],
) -> DeepPHYTextPrompt:
    evidence_text = trace_description.strip() or _empty_description_text(description_mode)
    description_label = _description_label(description_mode)
    evidence_intro = _description_intro(
        env_id=env_id,
        description_mode=description_mode,
        metadata=metadata,
    )
    format_note = _output_format_note(env_id=env_id, prompt_format=prompt_format)
    has_initial_image = bool(image_inputs or initial_image_path)
    inline_feedback_block = bool(metadata.get("inline_feedback_block"))
    wrapped_system_prompt = system_prompt
    if has_initial_image:
        wrapped_system_prompt = (
            f"{wrapped_system_prompt}\n\n"
            "Benchmark note: initial-state image inputs are attached to the user message. "
            "Use the attached images together with the simulation description below. "
            "Your output is parsed by an environment-specific strict action parser, so format violations count as failure."
        )
    else:
        wrapped_system_prompt = (
            f"{wrapped_system_prompt}\n\n"
            "Benchmark note: no initial-state image is attached for this request. "
            "Use the simulation description below as the full input. "
            "Your output is parsed by an environment-specific strict action parser, so format violations count as failure."
        )
    if inline_feedback_block:
        wrapped_user_prompt = f"{user_prompt}\n\n{format_note}"
    else:
        wrapped_user_prompt = (
            f"{user_prompt}\n\n"
            f"{evidence_intro}\n\n"
            f"{description_label}:\n"
            f"{evidence_text}\n\n"
            f"{format_note}"
        )
    return DeepPHYTextPrompt(
        env_id=env_id,
        task_id=task_id,
        record_kind=record_kind,
        index=index,
        prompt_format=prompt_format,
        description_mode=description_mode,
        system_prompt=wrapped_system_prompt,
        user_prompt=wrapped_user_prompt,
        initial_image_path=initial_image_path,
        image_inputs=image_inputs,
        trace_description=evidence_text,
        metadata=metadata,
    )


def _augment_phyre_attempt_history(user_prompt: str, attempt_history: list[dict[str, Any]]) -> str:
    if not attempt_history:
        return user_prompt
    attempt_lines: list[str] = []
    for item in attempt_history:
        cell = item.get("parsed_cell")
        radius = item.get("parsed_radius_size")
        status = str(item.get("simulation_status") or "UNKNOWN")
        summary = str(item.get("outcome_summary") or item.get("trace_summary") or "").strip()
        if not summary:
            summary = "No useful effect; choose a meaningfully different action."
        attempt_lines.append(
            f"- Attempt {item.get('attempt_number', '?')}: Cell {cell}, Radius {radius}. Result: {status}. {summary}"
        )
    latest = attempt_history[-1]
    latest_trace = str(latest.get("feedback_trace_text") or latest.get("trace_summary") or "").strip()
    latest_pattern = str(latest.get("feedback_pattern_text") or "").strip()
    if not latest_trace:
        latest_trace = "No simulation description is available for the most recent attempt."
    if not latest_pattern:
        latest_pattern = "No learned pattern annotation is available for the most recent attempt."
    guidance = (
        "Previous attempts summary:\n"
        + "\n".join(attempt_lines)
        + "\n\n"
        "Use a new legal action. Do not repeat an earlier cell/radius pair.\n\n"
        "Detailed feedback from the most recent attempt:\n"
        f"Action: Cell {latest.get('parsed_cell')}, Radius {latest.get('parsed_radius_size')}\n\n"
        "Full simulation description:\n"
        f"{latest_trace}\n\n"
        "Full learned pattern annotation:\n"
        f"{latest_pattern}\n\n"
    )
    return f"{guidance}{user_prompt}"


def _augment_iphyre_attempt_history(user_prompt: str, attempt_history: list[dict[str, Any]]) -> str:
    if not attempt_history:
        return user_prompt
    attempt_lines: list[str] = []
    for item in attempt_history:
        parsed_actions = item.get("parsed_actions") or []
        status = str(item.get("simulation_status") or "UNKNOWN")
        summary = str(item.get("outcome_summary") or item.get("trace_summary") or "").strip()
        if not summary:
            summary = "No useful effect; choose a materially different removal sequence."
        attempt_lines.append(
            f"- Attempt {item.get('attempt_number', '?')}: actions={parsed_actions}. Result: {status}. {summary}"
        )
    latest = attempt_history[-1]
    latest_trace = str(latest.get("feedback_trace_text") or latest.get("trace_summary") or "").strip()
    latest_pattern = str(latest.get("feedback_pattern_text") or "").strip()
    if not latest_trace:
        latest_trace = "No simulation description is available for the most recent attempt."
    if not latest_pattern:
        latest_pattern = "No learned pattern annotation is available for the most recent attempt."
    guidance = (
        "Previous attempts summary:\n"
        + "\n".join(attempt_lines)
        + "\n\n"
        "Use a new action sequence. Do not repeat an earlier exact sequence or a near-duplicate with the same blocks at nearly identical times.\n\n"
        "Detailed feedback from the most recent attempt:\n"
        f"Action: {latest.get('parsed_actions')}\n\n"
        "Full simulation description:\n"
        f"{latest_trace}\n\n"
        "Full learned pattern annotation:\n"
        f"{latest_pattern}\n\n"
    )
    return f"{guidance}{user_prompt}"


def _json_instruction(env_id: str) -> str:
    if env_id == "phyre":
        return (
            "Think briefly in plain text about the task, then end with exactly one final JSON object on the last line: "
            "{\"cell\": <int 1-64>, \"radius\": <int 1-8>}. "
            "Radius 1 is small, radius 3 is still modest, and radii 4-8 are progressively much larger. "
            "Prefer radius 1-3 unless a larger ball is genuinely needed and the chosen cell is clearly wide open."
        )
    if env_id == "iphyre":
        return (
            "Think briefly in plain text about the task, then end with exactly one final JSON object on the last line: "
            "{\"actions\": [{\"time\": <float 0-1>, \"index\": <int>} ...]}."
        )
    if env_id == "pooltool":
        return (
            "Think briefly in plain text about the shot, then end with exactly one final JSON object on the last line: "
            "{\"speed\": \"Low|Medium|High\", \"strikespot\": \"Top Spin|Middle Spin|Back Spin|Left|Right\"}."
        )
    if env_id == "kinetix":
        return (
            "Think briefly in plain text about the control decision, then end with exactly one final JSON object on the last line: "
            "{\"action\": [numbers]} where the list length matches the action dimension."
        )
    return (
        "Think briefly in plain text, then end with exactly one final JSON object on the last line "
        "using the required action fields."
    )


def _description_label(description_mode: DeepPHYTextDescriptionMode) -> str:
    if description_mode == "trace_text":
        return "Trace-derived simulation description"
    if description_mode == "pattern_text":
        return "Learned pattern annotation summary"
    if description_mode == "trace_text_plus_patterns":
        return "Trace and learned pattern summary"
    raise ValueError(f"Unsupported description mode: {description_mode}")


def _description_intro(
    *,
    env_id: str,
    description_mode: DeepPHYTextDescriptionMode,
    metadata: dict[str, Any],
) -> str:
    if env_id == "phyre":
        attempt_number = int(metadata.get("attempt_number") or 0)
        if attempt_number <= 1:
            base = (
                "The attached image shows the current PHYRE scene before your first action. "
                "There is no prior simulation feedback yet, so any description below refers only to the initial scene before the first attempt."
            )
        else:
            base = (
                "The attached image shows the current PHYRE scene you are acting from now. "
                "Any description below is feedback from the previous simulated attempt in this same task. "
                "It describes what happened after the last action and what that outcome implies for the current decision."
            )
        if description_mode == "trace_text":
            return f"{base} The simulation description is environment feedback, not a new instruction."
        if description_mode == "pattern_text":
            return f"{base} The learned pattern annotations describe events detected in that previous simulation outcome."
        if description_mode == "trace_text_plus_patterns":
            return f"{base} The trace text summarizes the previous simulation outcome, and the learned pattern annotations extract structured events from that same outcome."
    if env_id == "pooltool":
        attempt_number = int(metadata.get("attempt_number") or 0)
        if attempt_number <= 1:
            base = (
                "The attached image shows the current pool table before your next shot. "
                "There is no prior-shot simulation feedback yet, so any annotation summary below refers only to the initial simulation state before the first shot."
            )
        else:
            base = (
                "The attached image shows the current pool table before your next shot. "
                "Any annotation summary below is feedback from the previous simulated shot in this same episode. "
                "It describes what happened after the last action and what that outcome implies for the table state you are acting from now."
            )
        if description_mode == "trace_text":
            return (
                f"{base} The simulation description is environment feedback, not a new instruction."
            )
        if description_mode == "pattern_text":
            return (
                f"{base} The learned pattern annotations describe events detected in that simulation outcome, such as collisions, spins, pocketing, or stagnation."
            )
        if description_mode == "trace_text_plus_patterns":
            return (
                f"{base} The trace text summarizes the simulated outcome, and the learned pattern annotations extract structured events from that same outcome."
            )
    if env_id == "iphyre":
        attempt_number = int(metadata.get("attempt_number") or 0)
        if attempt_number <= 1:
            base = (
                "The attached image shows the current iPHYRE scene before your first removal attempt. "
                "There is no prior simulation feedback yet, so any description below refers only to the initial scene before the first attempt."
            )
        else:
            base = (
                "The attached image shows the current iPHYRE scene you are acting from now. "
                "Any description below is feedback from the previous simulated removal attempt in this same task. "
                "It describes what happened after the last action sequence and what that outcome implies for the current decision."
            )
        if description_mode == "trace_text":
            return f"{base} The simulation description is environment feedback, not a new instruction."
        if description_mode == "pattern_text":
            return f"{base} The learned pattern annotations describe events detected in that previous simulation outcome."
        if description_mode == "trace_text_plus_patterns":
            return f"{base} The trace text summarizes the previous simulation outcome, and the learned pattern annotations extract structured events from that same outcome."
    if env_id == "kinetix":
        step_index = int(metadata.get("step_index") or 0)
        if step_index <= 1:
            base = (
                "The attached image shows the current Kinetix state before the first control step. "
                "There is no prior simulation feedback yet, so any description below refers only to the initial state before the first step."
            )
        else:
            base = (
                "The attached image shows the current Kinetix state you are controlling now. "
                "Any description below is feedback from the previous simulated control step in this same rollout. "
                "It describes what happened after the last action and what that outcome implies for the next control decision."
            )
        if description_mode == "trace_text":
            return f"{base} The simulation description is environment feedback, not a new instruction."
        if description_mode == "pattern_text":
            return f"{base} The learned pattern annotations describe events detected in that previous simulation outcome."
        if description_mode == "trace_text_plus_patterns":
            return f"{base} The trace text summarizes the previous simulation outcome, and the learned pattern annotations extract structured events from that same outcome."
    if description_mode == "trace_text":
        return "Use the simulation description below as environment feedback relevant to the current decision."
    if description_mode == "pattern_text":
        return "Use the learned pattern annotations below as environment feedback relevant to the current decision."
    if description_mode == "trace_text_plus_patterns":
        return "Use the simulation description and learned pattern annotations below as environment feedback relevant to the current decision."
    raise ValueError(f"Unsupported description mode: {description_mode}")


def _empty_description_text(description_mode: DeepPHYTextDescriptionMode) -> str:
    if description_mode == "pattern_text":
        return "No learned pattern annotations are available yet."
    if description_mode == "trace_text_plus_patterns":
        return "No trace or learned pattern description is available yet."
    return "No prior simulation trace description is available yet."


def _output_format_note(*, env_id: str, prompt_format: DeepPHYTextPromptFormat) -> str:
    if prompt_format == "WM" and env_id == "phyre":
        return (
            "Reason briefly in plain text if needed. "
            "Then end with exactly one final JSON object on the last line in this schema: "
            "{\"cell\": <int 1-64>, \"radius\": <int 1-8>}. "
            "Radius 1 is small, radius 3 is still modest, and radii 4-8 are progressively much larger. "
            "Prefer radius 1-3, but you may still choose a larger radius when it is genuinely needed and there is obvious clearance. "
            "Avoid edge cells for medium or large radii unless there is obvious clearance. "
            "Do not use `Prediction:` or `Action:` labels, and do not put any extra JSON anywhere else."
        )
    if prompt_format == "WM" and env_id == "iphyre":
        return (
            "Reason briefly under `Prediction:`. "
            "Then provide the final answer under `Action:` as JSON only in this schema: "
            "{\"actions\": [{\"time\": <float 0-1>, \"index\": <int>} ...]}. "
            "Do not put any extra JSON anywhere else."
        )
    if prompt_format == "WM" and env_id == "pooltool":
        return (
            "Reason briefly under `Prediction:`. "
            "Then provide the final answer under `Action:` as JSON only in this schema: "
            "{\"speed\": \"Low|Medium|High\", \"strikespot\": \"Top Spin|Middle Spin|Back Spin|Left|Right\"}. "
            "Do not put any extra JSON anywhere else."
        )
    if prompt_format == "WM" and env_id == "kinetix":
        return (
            "Reason briefly under `Prediction:`. "
            "Then provide the final answer under `Action:` as JSON only in this schema: "
            "{\"action\": [numbers]}. "
            "Do not put any extra JSON anywhere else."
        )
    if env_id == "phyre":
        return (
            "You should reason briefly in plain text first. "
            "Then end with exactly one JSON object on the final line and nothing after it: "
            "{\"cell\": <int 1-64>, \"radius\": <int 1-8>}. "
            "Radius 1 is small, radius 3 is still modest, and radii 4-8 are progressively much larger. "
            "Prefer radius 1-3, but you may still choose a larger radius when it is genuinely needed and the selected cell is clearly wide open. "
            "Do not include any other JSON blocks."
        )
    if env_id == "iphyre":
        return (
            "You should reason briefly in plain text first. "
            "Then end with exactly one JSON object on the final line and nothing after it: "
            "{\"actions\": [{\"time\": <float 0-1>, \"index\": <int>} ...]}. "
            "Do not include any other JSON blocks."
        )
    if env_id == "pooltool":
        return (
            "You should reason briefly in plain text first. "
            "Then end with exactly one JSON object on the final line and nothing after it: "
            "{\"speed\": \"Low|Medium|High\", \"strikespot\": \"Top Spin|Middle Spin|Back Spin|Left|Right\"}. "
            "Do not include any other JSON blocks."
        )
    if env_id == "kinetix":
        return (
            "You should reason briefly in plain text first. "
            "Then end with exactly one JSON object on the final line and nothing after it: "
            "{\"action\": [numbers]}. "
            "Do not include any other JSON blocks."
        )
    raise ValueError(f"Unsupported environment: {env_id}")


def _normalize_vla_user_prompt(user_prompt: str, prompt_format: DeepPHYTextPromptFormat) -> str:
    prompt_format = normalize_deepphy_prompt_format(prompt_format)
    if prompt_format != "VLA":
        return user_prompt
    normalized = user_prompt.replace(
        "Analyze the scene, formulate a prediction about how to solve it, and then provide the corresponding action.",
        "Analyze the scene and provide the corresponding action immediately.",
    )
    normalized = normalized.replace(
        "Analyze the scene, formulate a prediction about how to solve it, and then provide the corresponding action sequence.",
        "Analyze the scene and provide the corresponding action sequence immediately.",
    )
    return normalized
