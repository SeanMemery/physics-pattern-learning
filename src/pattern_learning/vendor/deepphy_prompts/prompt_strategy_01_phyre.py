from __future__ import annotations

import re
import json
import numpy as np

from vendor.deepphy_prompts._config import load_prompt_config
from vendor.deepphy_support.img_utils import (
    convert_radius_size_to_normalized,
    get_normalized_center_coords,
)

config = load_prompt_config("env_config_01_phyre.json")


def _extract_final_json_action(text_output: str) -> dict | None:
    text = str(text_output).strip()
    candidates: list[dict] = []
    for start in range(len(text)):
        if text[start] != "{":
            continue
        for end in range(start + 1, min(len(text), start + 256)):
            if text[end] != "}":
                continue
            snippet = text[start : end + 1]
            try:
                obj = json.loads(snippet)
            except Exception:
                continue
            if isinstance(obj, dict) and "cell" in obj:
                candidates.append(obj)
    return candidates[-1] if candidates else None

def parse_and_convert_vlm_output(text_output: str) -> tuple:
    """"VLA Prompt Format Output Parser and Converter, extracts Cell and Radius from text output and converts to [x, y, r] coordinates."""
    text_output = str(text_output)
    obj = _extract_final_json_action(text_output)
    if obj is not None:
        try:
            cell_num = int(obj.get("cell"))
            radius_size = int(obj.get("radius", 1))
            coords = get_normalized_center_coords(cell_num, config.grid_size)
            if coords is None: return None, cell_num, radius_size
            normalized_r = convert_radius_size_to_normalized(radius_size, config.radius_levels )
            if normalized_r is None: return None, cell_num, radius_size
            action = np.array([coords[0], coords[1], normalized_r])
            return action, cell_num, radius_size
        except (TypeError, ValueError, IndexError):
            pass
    match = re.search(r"Cell\s*[:=\s]*(\d+).*Radius\s*[:=\s]*(\d+)", text_output, re.IGNORECASE | re.DOTALL)
    if not match:
        return None, None, None
    try:
        cell_num = int(match.group(1))
        radius_size = int(match.group(2))
        coords = get_normalized_center_coords(cell_num, config.grid_size)
        if coords is None: return None, cell_num, radius_size
        normalized_r = convert_radius_size_to_normalized(radius_size, config.radius_levels )
        if normalized_r is None: return None, cell_num, radius_size
        action = np.array([coords[0], coords[1], normalized_r])
        return action, cell_num, radius_size
    except (ValueError, IndexError):
        return None, None, None

def parse_wm_vlm_output(text_output: str) -> tuple:
    """ WM Prompt Format Output Parser and Converter, extracts Prediction, Cell and Radius from text output and converts to [x, y, r] coordinates.  """
    text_output = str(text_output)
    obj = _extract_final_json_action(text_output)
    if obj is not None:
        prediction_text = text_output.split("{", 1)[0].strip() or "The model did not provide additional reasoning."
        try:
            cell_num = int(obj.get("cell"))
            radius_size = int(obj.get("radius", 1))
            coords = get_normalized_center_coords(cell_num, config.grid_size)
            if coords is None: return None, cell_num, radius_size, prediction_text
            normalized_r = convert_radius_size_to_normalized(radius_size, config.radius_levels)
            if normalized_r is None: return None, cell_num, radius_size, prediction_text
            action = np.array([coords[0], coords[1], normalized_r])
            return action, cell_num, radius_size, prediction_text
        except (TypeError, ValueError, IndexError):
            pass
    prediction_match = re.search(r"Prediction\s*:\s*(.*?)Action\s*:", text_output, re.IGNORECASE | re.DOTALL)
    if not prediction_match:
        prediction_match = re.search(r"Prediction\s*:\s*(.*)", text_output, re.IGNORECASE)
    prediction_text = prediction_match.group(1).strip() if prediction_match else "The model failed to provide a prediction."

    action_match = re.search(r"Action\s*:\s*Cell\s*[:=\s]*(\d+).*Radius\s*[:=\s]*(\d+)", text_output, re.IGNORECASE | re.DOTALL)
    if not action_match:
        return None, None, None, prediction_text

    try:
        cell_num = int(action_match.group(1))
        radius_size = int(action_match.group(2))
        coords = get_normalized_center_coords(cell_num, config.grid_size)
        if coords is None: return None, cell_num, radius_size, prediction_text
        normalized_r = convert_radius_size_to_normalized(radius_size, config.radius_levels)
        if normalized_r is None: return None, cell_num, radius_size, prediction_text
        action = np.array([coords[0], coords[1], normalized_r])
        return action, cell_num, radius_size, prediction_text
    except (ValueError, IndexError):
        return None, None, None, prediction_text

# --- 策略类定义 ---

class PromptStrategy:
    """ Prompt Strategy Interface """
    def __init__(self):
        # 从常量模块获取配置
        self.grid_size = config.grid_size
        self.radius_levels = config.radius_levels

    def generate_prompts(self, attempt_history, image_descriptions):
        """ Generate system and user prompts based on attempt history and image descriptions. """
        raise NotImplementedError

    def parse_response(self, vlm_response: str) -> dict:
        """ Parse the VLM response and return a dictionary with parsed fields. """
        raise NotImplementedError

    def update_attempt_data(self, base_data: dict, parsed_output: dict) -> dict:
        """ Update the base attempt data with additional fields from parsed output. """
        return base_data

class VlaStrategy(PromptStrategy):
    """ VLA Prompt Format Implementation. """
    def generate_prompts(self, attempt_history, image_descriptions):
        system_prompt = (
            f"""You are a premier AI agent, an expert in solving physics-based puzzles. Your task is to analyze a given scene and determine the single optimal placement of a new object to achieve a specific goal.
## Scene Legend:
*   Red Ball: The dynamic ball you will place.
*   Green Ball: The primary dynamic object you must manipulate.
*   Blue Ball: A dynamic target.
*   Purple Ball: A static (immovable) target.
*   Grey Ball: Other dynamic objects.
*   Black Object: Static obstacles.
## Mission Objective:
Strategically place a single red ball to trigger a chain reaction. The successful outcome is achieved when the green ball makes physical contact with either the blue ball or the purple ball.
## Action Parameters:
*   `Cell`: A grid cell number from 1 to {self.grid_size[0] * self.grid_size[1]}.
*   `Radius`: A size level from 1 (smallest) to {self.radius_levels} (largest).
*   Size intuition: radius 1 is a small drop, radius 3 is still modest, and radii 4-8 become progressively much larger.
## Placement Constraints:
Your action is only valid if it meets these strict conditions:
*   No Collisions: The red ball cannot overlap with any existing objects.
*   In-Bounds: The red ball must be placed entirely within the puzzle's boundaries.
*   Prefer small radii: `1-3` are the safest choices.
*   You may still choose radii `4+` when a larger ball is genuinely needed, but only if the chosen cell is clearly in wide open space.
*   Avoid edge cells for medium or large radii unless there is obvious clearance.
## Required Output Format:
Return exactly one JSON object and nothing else:
{{"cell": [NUMBER], "radius": [NUMBER]}}"""
        )

        if not attempt_history:
            user_prompt = "Analyze the scene, formulate a prediction about how to solve it, and then provide the corresponding action."
        else:
            feedback_summary = "You have made previous attempts that failed. Here is a summary:\n\n"
            for past_attempt in attempt_history:
                status = "INVALID ACTION" if past_attempt['simulation_status'] in ["INVALID_INPUT", "INVALID_ACTION_FORMAT"] else "FAILED"
                feedback_summary += f"Attempt {past_attempt['attempt_number']}: You chose (Cell: {past_attempt['parsed_cell']}, Radius: {past_attempt['parsed_radius_size']}), which {status}.\n"

            user_prompt = (
                f"{feedback_summary}Based on this complete history and the scene images, analyze your mistakes and propose a new, better action. "
                "Use the full structured history of previous attempts below. You MUST propose a NEW action. DO NOT repeat any of the previous actions. "
                "If a previous action was invalid, first shrink the radius and move away from occupied or edge cells. "
                f"Your response MUST be ONLY this JSON object: {{\"cell\": [NUMBER], \"radius\": [NUMBER]}}."
            )
        return system_prompt, user_prompt

    def parse_response(self, vlm_response: str) -> dict:
        action, cell, radius = parse_and_convert_vlm_output(vlm_response)
        return {"action": action, "parsed_cell": cell, "parsed_radius_size": radius}

class WmStrategy(PromptStrategy):
    """ WM Prompt Format Implementation. """
    def generate_prompts(self, attempt_history, image_descriptions):
        system_prompt = (
            f"""You are a premier AI agent, an expert in solving physics-based puzzles. Your task is to analyze a given scene and decide one legal action.
## Scene Legend:
*   Red Ball: The dynamic ball you will place.
*   Green Ball: The primary dynamic object you must manipulate.
*   Blue Ball: A dynamic target.
*   Purple Ball: A static (immovable) target.
*   Grey Ball: Other dynamic objects.
*   Black Object: Static obstacles.
## Mission Objective:
Strategically place a single red ball to trigger a chain reaction. The successful outcome is achieved when the green ball makes physical contact with either the blue ball or the purple ball.
## Action Parameters:
*   `Cell`: A grid cell number from 1 to {self.grid_size[0] * self.grid_size[1]}.
*   `Radius`: A size level from 1 (smallest) to {self.radius_levels} (largest).
*   Size intuition: radius 1 is a small drop, radius 3 is still modest, and radii 4-8 become progressively much larger.
## Placement Constraints:
Your action is only valid if it meets these strict conditions:
*   No Collisions: The red ball cannot overlap with any existing objects.
*   In-Bounds: The red ball must be placed entirely within the puzzle's boundaries.
*   Prefer small radii: `1-3` are the safest choices.
*   You may still choose radii `4+` when a larger ball is genuinely needed, but only if the chosen cell is clearly in wide open space.
*   Avoid edge cells for medium or large radii unless there is obvious clearance.
## Required Output Format:
You may think briefly in plain text, but your final line must be exactly one JSON object in this schema and nothing else:
{{"cell": [NUMBER], "radius": [NUMBER]}}"""
        )

        if not attempt_history:
            user_prompt = "Analyze the scene, formulate a prediction about how to solve it, and then provide the corresponding action."
        else:
            feedback_summary = "You have made previous attempts that failed. Here is a summary of your predictions and the actual outcomes:\n\n"
            for past_attempt in attempt_history:
                outcome = "Your action was INVALID." if past_attempt['simulation_status'] in ["INVALID_INPUT", "INVALID_ACTION_FORMAT"] else "The action FAILED."
                feedback_summary += (
                    f"--- Attempt {past_attempt['attempt_number']} ---\n"
                    f"Your Action: Cell: {past_attempt['parsed_cell']}, Radius: {past_attempt['parsed_radius_size']}\n"
                    f"Your Prediction: {past_attempt.get('prediction', 'N/A')}\n"
                    f"Actual Outcome: {outcome}\n\n"
                )

            user_prompt = (
                f"{feedback_summary}Analyze why your previous predictions were wrong by comparing them to the actual outcomes. Use the structured history below instead of repeating old actions. "
                "Propose a new, better legal action. If a previous action was invalid, first shrink the radius and move away from occupied or edge cells. "
                "Your response MUST end with exactly one JSON object in the form {\"cell\": [NUMBER], \"radius\": [NUMBER]}."
            )
        return system_prompt, user_prompt

    def parse_response(self, vlm_response: str) -> dict:
        action, cell, radius, prediction = parse_wm_vlm_output(vlm_response)
        return {"action": action, "parsed_cell": cell, "parsed_radius_size": radius, "prediction": prediction}

    def update_attempt_data(self, base_data: dict, parsed_output: dict) -> dict:
        base_data['prediction'] = parsed_output.get('prediction', 'N/A')
        return base_data

def get_prompt_strategy(format_name: str) -> PromptStrategy:
    """ Factory function to get the appropriate prompt strategy instance based on the format name. """
    strategies = {
        "VLA": VlaStrategy,
        "WM": WmStrategy,
    }
    strategy_class = strategies.get(format_name)
    if not strategy_class:
        raise ValueError(f"Unknown prompt strategy format: {format_name}")
    print(f"Using prompt strategy: {format_name}")
    return strategy_class()
