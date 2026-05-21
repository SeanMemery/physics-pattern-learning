from __future__ import annotations

import ast
import base64
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from .artifacts import write_json
from .library import load_library_summary, annotate_trace_with_supervised_library, query_trace_patterns, filter_trace_patterns
from .llm import chat_completion
from .schemas import LLMSettings, RewardProgram, RewardValue, SceneSpec


DSL_CODE_FENCE_RE = re.compile(r"```(?:dsl|text)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class EvalContext:
    trace: dict[str, Any]
    table_width: float
    table_length: float
    pattern_annotation: dict[str, Any] | None = None


def synthesize_reward_program(
    *,
    goal_text: str,
    scene: SceneSpec,
    llm_settings: LLMSettings,
    artifact_root: Path,
    library_summary: str | None = None,
    use_images: bool = True,
    progress: callable | None = None,
) -> RewardProgram:
    system_prompt = (
        "Given the following goal, scene, and DSL reference, propose one DSL expression.\n"
        "This expression defines a reward function. This reward function will be maximised by an optimisation process.\n"
        "First infer the intended shot plan from the scene image: which ball the cue ball should strike, what intermediate contacts should happen, and how the target outcome should unfold.\n"
        "Then encode that intended sequence in the DSL so the reward favours the planned shot, not just the final outcome.\n"
        "If the goal is to pot a specific ball, prefer a reward that includes the cue ball striking the right object ball and that object ball then doing the intended thing.\n"
        "Use PATTERN(...) when the goal is better expressed in terms of learned library patterns such as gentle touch, cushion rebound, or high angle collision.\n"
        "Prefer the direct PoolTool event primitives when they are simpler and more precise.\n"
        "Do not include comments or explanation in the DSL code.\n"
        "Respond with only the DSL inside ```dsl fences```."
    )
    user_prompt = "\n\n".join(
        [
            f"Goal:\n{goal_text.strip()}",
            f"DSL reference:\n{dsl_reference_text()}",
            f"Library summary:\n{library_summary or load_library_summary()}",
            f"Representative scene summary:\n{scene_summary_text(scene)}",
            (
                "Shot-planning instruction:\n"
                "Look at the scene image and choose a plausible shot trajectory for satisfying the goal. "
                "The DSL should reward the sequence of events that would make that shot succeed, such as the cue ball contacting a specific target ball before that ball is potted or moved into place."
            ),
        ]
    )
    prompt_media_dir = artifact_root / "prompt_media"
    image_paths = []
    if use_images:
        if progress is not None:
            progress("synthesize: preparing prompt image")
        prepared = prepare_prompt_image_copy(scene.initial_scene_overhead_path, output_dir=prompt_media_dir)
        if prepared is not None:
            image_paths.append(prepared)
    if progress is not None:
        progress(f"synthesize: calling llm (images={len(image_paths)})")
    body, response_text = chat_completion(
        settings=llm_settings,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_multimodal_prompt_content(text=user_prompt, image_paths=image_paths)},
        ],
        max_tokens=512,
        progress=progress,
    )
    dsl_program = normalize_dsl_program(response_text)
    parse_expression(dsl_program)
    if progress is not None:
        progress("synthesize: received response; writing artifacts")
    request_path = write_json(
        artifact_root / "synthesis_request.json",
        {
            "goal_text": goal_text,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "image_paths": [str(path) for path in image_paths],
            "response_body": body,
        },
    )
    response_path = artifact_root / "synthesis_response.txt"
    response_path.write_text(response_text, encoding="utf-8")
    return RewardProgram(
        goal_text=goal_text,
        dsl_program=dsl_program,
        synthesized=True,
        synthesis_request_path=request_path,
        synthesis_response_path=response_path,
    )


def explicit_reward_program(*, goal_text: str, dsl_program: str) -> RewardProgram:
    normalized = normalize_dsl_program(dsl_program)
    parse_expression(normalized)
    return RewardProgram(goal_text=goal_text, dsl_program=normalized, synthesized=False)


def normalize_dsl_program(program_text: str) -> str:
    stripped = str(program_text or "").strip()
    if not stripped:
        return ""
    fenced = DSL_CODE_FENCE_RE.search(stripped)
    if fenced is not None:
        stripped = fenced.group(1).strip()
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", stripped, re.DOTALL | re.IGNORECASE)
    if answer_match is not None:
        stripped = answer_match.group(1).strip()
    return stripped


def dsl_reference_text() -> str:
    return "\n".join(
        [
            'AND(expr1, expr2, ...) -> all clauses must hold; dense reward is the mean clause score.',
            'OR(expr1, expr2, ...) -> any clause may hold; dense reward is the max clause score.',
            'NOT(expr) -> negation.',
            'AFTER(expr1, expr2) -> both event clauses occur and expr2 happens after expr1.',
            'POTTED("red") -> whether the named ball was potted.',
            'BALL_REMAINING("cue") -> whether the named ball remains on table at the end.',
            'CUE_HIT("red") -> cue ball collides with the named ball.',
            'BALL_HIT("red", "black") -> the first named ball collides with the second named ball.',
            'NEARBY_AT("red", x, y, t_fraction, threshold_strength=0.2) -> ball position near a target coordinate at a fractional time.',
            'PATTERN("gentle touch") -> whether the supervised pattern library emits a pattern with that label.',
            'PATTERN("ball pocketed", ball="6") -> require a labeled pattern plus matching normalized params.',
            'PATTERN("ball collision", contains=["6", "3"]) -> require that all listed scalar values appear in one emitted pattern.',
            'PATTERN_COUNT("cushion rebound", ball="cue", min_count=2) -> require at least N matching patterns; event_time is the Nth match time.',
            'Use only a single expression. Build the reward as a sum of positive terms with AND where possible.',
        ]
    )


def scene_summary_text(scene: SceneSpec) -> str:
    lines = [
        f"Scene id: {scene.sample_id}",
        f"Table size: {scene.table_width:.3f} x {scene.table_length:.3f}",
        "Initial balls:",
    ]
    for item in scene.initial_ball_positions:
        lines.append(f"- Ball {item.ball_id}: x={item.x:.3f}, y={item.y:.3f}")
    return "\n".join(lines)


def build_multimodal_prompt_content(*, text: str, image_paths: Sequence[Path]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for index, image_path in enumerate(image_paths, start=1):
        content.append({"type": "text", "text": f"Scene image {index}."})
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
        image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}})
    return content


def resize_prompt_image_in_place(path: Path, *, max_side: int = 512) -> None:
    with Image.open(path) as image:
        width, height = image.size
        longest = max(width, height)
        if longest <= max_side:
            image.save(path)
            return
        scale = max_side / float(longest)
        resized = image.resize((max(1, int(round(width * scale))), max(1, int(round(height * scale)))), Image.Resampling.LANCZOS)
        resized.save(path)


def prepare_prompt_image_copy(source_path: Path, *, output_dir: Path) -> Path | None:
    if not source_path.is_file():
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = output_dir / source_path.name
    copied.write_bytes(source_path.read_bytes())
    resize_prompt_image_in_place(copied, max_side=512)
    return copied


def parse_expression(program_text: str) -> ast.AST:
    return ast.parse(program_text, mode="eval").body


def evaluate_reward_program(program_text: str, trace: dict[str, Any], *, table_width: float, table_length: float) -> RewardValue:
    node = parse_expression(program_text)
    pattern_annotation = None
    if "PATTERN(" in program_text or "PATTERN_COUNT(" in program_text:
        pattern_annotation, _ = annotate_trace_with_supervised_library(trace)
    context = EvalContext(trace=trace, table_width=table_width, table_length=table_length, pattern_annotation=pattern_annotation)
    return _eval_node(node, context)


def _eval_node(node: ast.AST, context: EvalContext) -> RewardValue:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError(f"Unsupported DSL node: {ast.dump(node)}")
    name = node.func.id
    if name == "AND":
        values = [_eval_node(arg, context) for arg in node.args]
        return RewardValue(score=sum(item.score for item in values) / max(len(values), 1), success=all(item.success for item in values), event_time=max((item.event_time for item in values if item.event_time is not None), default=None))
    if name == "OR":
        values = [_eval_node(arg, context) for arg in node.args]
        best = max(values, key=lambda item: item.score)
        return RewardValue(score=best.score, success=any(item.success for item in values), event_time=best.event_time)
    if name == "NOT":
        value = _eval_node(node.args[0], context)
        return RewardValue(score=1.0 - value.score, success=not value.success, event_time=value.event_time)
    if name == "AFTER":
        first = _eval_node(node.args[0], context)
        second = _eval_node(node.args[1], context)
        success = bool(first.success and second.success and first.event_time is not None and second.event_time is not None and second.event_time > first.event_time)
        return RewardValue(score=1.0 if success else 0.0, success=success, event_time=second.event_time if success else None)
    if name == "POTTED":
        ball_id = _literal(node.args[0])
        event_time = _first_event_time(context.trace, "physics_ball_pocket", {ball_id})
        return RewardValue(score=1.0 if event_time is not None else 0.0, success=event_time is not None, event_time=event_time)
    if name == "BALL_REMAINING":
        ball_id = _literal(node.args[0])
        success = _ball_present_at_end(context.trace, ball_id)
        return RewardValue(score=1.0 if success else 0.0, success=success)
    if name == "CUE_HIT":
        ball_id = _literal(node.args[0])
        event_time = _first_event_time(context.trace, "physics_ball_ball", {"cue", ball_id})
        return RewardValue(score=1.0 if event_time is not None else 0.0, success=event_time is not None, event_time=event_time)
    if name == "BALL_HIT":
        first_ball = _literal(node.args[0])
        second_ball = _literal(node.args[1])
        event_time = _first_event_time(context.trace, "physics_ball_ball", {first_ball, second_ball})
        return RewardValue(score=1.0 if event_time is not None else 0.0, success=event_time is not None, event_time=event_time)
    if name == "NEARBY_AT":
        ball_id = _literal(node.args[0])
        x = float(_literal(node.args[1]))
        y = float(_literal(node.args[2]))
        t_fraction = float(_literal(node.args[3]))
        threshold_strength = 0.2
        for kw in node.keywords:
            if kw.arg == "threshold_strength":
                threshold_strength = float(_literal(kw.value))
        score = _nearby_score(context.trace, ball_id, x, y, t_fraction, threshold_strength=threshold_strength, table_width=context.table_width, table_length=context.table_length)
        return RewardValue(score=score, success=score >= 0.999)
    if name == "PATTERN":
        label = str(_literal(node.args[0]))
        criteria = {str(kw.arg): _literal(kw.value) for kw in node.keywords if kw.arg}
        result = query_trace_patterns(
            context.trace,
            label=label,
            criteria=criteria,
            annotation=context.pattern_annotation,
        )
        return RewardValue(
            score=float(result["score"]),
            success=bool(result["success"]),
            event_time=result["event_time"],
        )
    if name == "PATTERN_COUNT":
        label = str(_literal(node.args[0]))
        criteria: dict[str, Any] = {}
        min_count = 1
        for kw in node.keywords:
            if not kw.arg:
                continue
            if kw.arg == "min_count":
                min_count = int(_literal(kw.value))
            else:
                criteria[str(kw.arg)] = _literal(kw.value)
        min_count = max(1, min_count)
        matches = filter_trace_patterns(
            context.trace,
            label=label,
            criteria=criteria,
            annotation=context.pattern_annotation,
        )
        count = len(matches)
        score = min(1.0, float(count) / float(min_count))
        times = sorted([item["event_time"] for item in matches if item.get("event_time") is not None])
        event_time: float | None = None
        if times:
            event_time = times[min_count - 1] if len(times) >= min_count else times[-1]
        return RewardValue(score=score, success=count >= min_count, event_time=event_time)
    raise ValueError(f"Unsupported DSL function: {name}")


def _literal(node: ast.AST) -> Any:
    return ast.literal_eval(node)


def _first_event_time(trace: dict[str, Any], kind: str, ids: set[str]) -> float | None:
    for event in trace.get("events", []):
        if not isinstance(event, dict) or str(event.get("kind") or "") != kind:
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        event_ids = {str(item) for item in payload.get("ids", [])}
        if ids.issubset(event_ids):
            t = event.get("t")
            return float(t) if t is not None else None
    return None


def _ball_present_at_end(trace: dict[str, Any], ball_id: str) -> bool:
    timesteps = trace.get("timesteps", [])
    if not timesteps:
        return False
    last = timesteps[-1] if isinstance(timesteps[-1], dict) else {}
    for obj in last.get("objects", []):
        if isinstance(obj, dict) and str(obj.get("ball_id") or "") == ball_id:
            return True
    return False


def _nearby_score(trace: dict[str, Any], ball_id: str, x: float, y: float, t_fraction: float, *, threshold_strength: float, table_width: float, table_length: float) -> float:
    timesteps = [item for item in trace.get("timesteps", []) if isinstance(item, dict)]
    if not timesteps:
        return 0.0
    index = max(0, min(len(timesteps) - 1, int(round(t_fraction * (len(timesteps) - 1)))))
    target_step = timesteps[index]
    for obj in target_step.get("objects", []):
        if not isinstance(obj, dict) or str(obj.get("ball_id") or "") != ball_id:
            continue
        if "position" in obj and isinstance(obj.get("position"), list):
            ox, oy = float(obj["position"][0]), float(obj["position"][1])
        else:
            ox, oy = float(obj.get("x", 0.0)), float(obj.get("y", 0.0))
        scale = max(min(table_width, table_length) * max(threshold_strength, 1e-3), 1e-6)
        distance = ((ox - x) ** 2 + (oy - y) ** 2) ** 0.5
        return max(0.0, 1.0 - (distance / scale))
    return 0.0
