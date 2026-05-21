from __future__ import annotations

from pathlib import Path

from .dsl import explicit_reward_program, synthesize_reward_program
from .schemas import GoalSpec, LLMSettings, RewardProgram, SceneSpec


def build_reward_program_for_goal(
    *,
    goal: GoalSpec,
    scene: SceneSpec,
    llm_settings: LLMSettings | None,
    artifact_root: Path,
) -> RewardProgram:
    if goal.dsl_program:
        return explicit_reward_program(goal_text=goal.text, dsl_program=goal.dsl_program)
    if llm_settings is None:
        raise ValueError(f"Goal {goal.id} has no dsl_program and no llm_settings were provided.")
    return synthesize_reward_program(
        goal_text=goal.text,
        scene=scene,
        llm_settings=llm_settings,
        artifact_root=artifact_root,
    )
