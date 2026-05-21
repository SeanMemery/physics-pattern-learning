#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import phyre
from media import write_video_artifact
from phyre_video import phyre_video_fps


CONFIG_REF = "src/vendor/deepphy_prompts/configs/env_config_01_phyre.json"
RUNNER_REF = "src/tooling/phyre_trace_helper_py39.py"
PHYRE_SIMULATION_FPS = 60


def summarize_shape(value):
    if value is None or not hasattr(value, "shape"):
        return {}
    return {"shape": list(getattr(value, "shape"))}


def normalize_action(action):
    if hasattr(action, "tolist"):
        return action.tolist()
    if isinstance(action, tuple):
        return list(action)
    return action


def save_rgb_observation(observation, destination):
    rgb = phyre.observations_to_uint8_rgb(observation)
    Image.fromarray(rgb).resize((1024, 1024), Image.Resampling.LANCZOS).save(destination)


def effective_frame_count(states):
    last_dynamic_index = 0
    for index in range(1, len(states)):
        frame_delta = 0.0
        for object_index in range(len(states[index])):
            for component_index in range(len(states[index][object_index])):
                frame_delta += abs(
                    float(states[index][object_index][component_index])
                    - float(states[index - 1][object_index][component_index])
                )
        if frame_delta > 1e-9:
            last_dynamic_index = index
    return max(last_dynamic_index + 1, 2)


def normalized_angle_delta(angle, previous_angle):
    delta = angle - previous_angle
    while delta > 0.5:
        delta -= 1.0
    while delta < -0.5:
        delta += 1.0
    return delta


def build_dense_timesteps(featurized_objects, action_payload, simulation_fps):
    if featurized_objects is None:
        return []

    states = featurized_objects.states
    frame_count = effective_frame_count(states)
    timesteps = []
    for frame_index in range(frame_count):
        objects = []
        for object_index in range(featurized_objects.num_objects):
            angle = float(states[frame_index][object_index][2])
            previous_angle = float(states[max(frame_index - 1, 0)][object_index][2])
            x = float(states[frame_index][object_index][0])
            y = float(states[frame_index][object_index][1])
            previous_x = float(states[max(frame_index - 1, 0)][object_index][0])
            previous_y = float(states[max(frame_index - 1, 0)][object_index][1])
            objects.append(
                {
                    "object_index": object_index,
                    "shape": featurized_objects.shapes[object_index],
                    "color": featurized_objects.colors[object_index],
                    "diameter": float(featurized_objects.diameters[object_index]),
                    "x": x,
                    "y": y,
                    "angle": angle,
                    "vx": x - previous_x,
                    "vy": y - previous_y,
                    "angular_velocity": normalized_angle_delta(angle, previous_angle),
                    "is_user_input": object_index >= featurized_objects.num_scene_objects,
                }
            )

        timesteps.append(
            {
                "index": frame_index,
                "t": float(frame_index) / max(simulation_fps, 1),
                "state": {"frame_index": frame_index, "object_count": len(objects)},
                "objects": objects,
                "actions": [action_payload] if frame_index == 0 else [],
            }
        )
    return timesteps


def load_context(eval_setup, fold_id, eval_type):
    train_tasks, _, test_tasks = phyre.get_fold(eval_setup, fold_id)
    if eval_type == "train":
        selected_tasks = list(train_tasks)
    elif eval_type == "test":
        selected_tasks = list(test_tasks)
    else:
        selected_tasks = list(dict.fromkeys([*train_tasks, *test_tasks]))
    action_tier = phyre.eval_setup_to_action_tier(eval_setup)
    simulator = phyre.initialize_simulator(selected_tasks, action_tier)
    return {
        "task_ids": list(getattr(simulator, "task_ids", selected_tasks)),
        "simulator": simulator,
        "action_tier": action_tier,
        "initial_scenes": list(getattr(simulator, "initial_scenes", [])),
    }


def list_tasks(args):
    context = load_context(args.eval_setup, args.fold_id, args.eval_type)
    print(json.dumps(context["task_ids"]))


def generate_trace(args):
    context = load_context(args.eval_setup, args.fold_id, args.eval_type)
    task_ids = context["task_ids"]
    if args.task_id not in task_ids:
        raise KeyError("Unknown PHYRE task id: %s" % args.task_id)

    task_index = task_ids.index(args.task_id)
    simulator = context["simulator"]
    action, simulation, search_attempts = sample_action(
        simulator,
        task_index=task_index,
        sample_seed=args.sample_seed,
        sampling_strategy=args.sampling_strategy,
        min_dense_frames=args.min_dense_frames,
        max_sample_attempts=args.max_sample_attempts,
        solve_sample_attempts=args.solve_sample_attempts,
    )
    status = simulation.status
    status_name = getattr(status, "name", str(status))
    success = bool(status.is_solved()) if hasattr(status, "is_solved") else False
    images_value = getattr(simulation, "images", None)
    images = list(images_value) if images_value is not None else []

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attachments = []
    timesteps = []
    events = [
        {
            "kind": "simulation_status",
            "t": 0.0,
            "payload": {"status_name": status_name, "is_solved": success},
        }
    ]
    action_payload = {"placement": normalize_action(action)}

    initial_scene = None
    if task_index < len(context["initial_scenes"]):
        initial_scene = context["initial_scenes"][task_index]
    if initial_scene is not None:
        initial_scene_path = output_dir / "initial_scene.png"
        save_rgb_observation(initial_scene, initial_scene_path)
        attachments.append(
            {
                "kind": "initial_scene",
                "path": str(initial_scene_path),
                "label": "Initial scene",
                "metadata": summarize_shape(initial_scene),
            }
        )

    timesteps = build_dense_timesteps(
        getattr(simulation, "featurized_objects", None),
        action_payload,
        PHYRE_SIMULATION_FPS,
    )
    if not timesteps:
        timesteps = [
            {
                "index": 0,
                "t": 0.0,
                "state": {"simulation_status": status_name},
                "actions": [action_payload],
            },
            {
                "index": 1,
                "t": 1.0,
                "state": {"simulation_status": status_name, "is_terminal_summary": True},
                "actions": [],
            },
        ]
    if images:
        for frame_index, image in enumerate(images[: len(timesteps)]):
            frame_path = output_dir / ("frame_%03d.png" % frame_index)
            save_rgb_observation(image, frame_path)
            attachments.append(
                {
                    "kind": "rendered_frame",
                    "path": str(frame_path),
                    "label": "Frame %d" % frame_index,
                    "metadata": summarize_shape(image),
                }
            )
        simulation_frame_count = max(1, len(timesteps))
        video_fps = phyre_video_fps(
            len(images[: len(timesteps)]),
            simulation_frame_count=simulation_frame_count,
            simulation_fps=PHYRE_SIMULATION_FPS,
            success=success,
        )
        video_metadata = write_video_artifact(
            [output_dir / ("frame_%03d.png" % frame_index) for frame_index in range(min(len(images), len(timesteps)))],
            output_dir / "simulation.mp4",
            fps=video_fps,
            max_side=1024,
        )
        video_metadata["duration_seconds"] = round(
            video_metadata["frame_count"] / max(video_metadata["fps"], 1),
            3,
        )
        attachments.append(
            {
                "kind": "simulation_video",
                "path": str(output_dir / "simulation.mp4"),
                "label": "Simulation video",
                "metadata": video_metadata,
            }
        )
    events.append(
        {
            "kind": "motion_sequence",
            "t": float(timesteps[-1]["t"]),
            "payload": {"frame_count": len(timesteps)},
        }
    )

    trace = {
        "trace_id": "%s:%s:%s:sample-%03d" % (
            "phyre",
            args.task_id,
            status_name.lower(),
            args.sample_index,
        ),
        "environment": "phyre",
        "task_id": args.task_id,
        "initial_state": {
            "eval_setup": args.eval_setup,
            "fold_id": args.fold_id,
            "eval_type": args.eval_type,
            "task_index": task_index,
            "action_tier": context["action_tier"],
            "initial_scene": summarize_shape(initial_scene),
            "simulation_fps": PHYRE_SIMULATION_FPS,
            "video_fps": video_metadata["fps"] if images else PHYRE_SIMULATION_FPS,
            "min_dense_frames": args.min_dense_frames,
            "sampling_strategy": args.sampling_strategy,
            "search_attempts": search_attempts,
        },
        "actions": [action_payload],
        "timesteps": timesteps,
        "events": events,
        "attachments": attachments,
        "outcome": {
            "task_name": "solved_status",
            "metrics": {
                "status_name": status_name,
                "is_solved": success,
                "action_tier": context["action_tier"],
                "frame_count": len(timesteps),
                "sampling_strategy": args.sampling_strategy,
                "search_attempts": search_attempts,
            },
            "success": success,
            "terminal_status": status_name,
        },
        "provenance": {
            "source": "deepphy-rework",
            "config_ref": CONFIG_REF,
            "runner_ref": RUNNER_REF,
            "notes": [
                "eval_setup=%s" % args.eval_setup,
                "fold_id=%d" % args.fold_id,
                "eval_type=%s" % args.eval_type,
                "generated via dedicated Python 3.9 PHYRE helper",
            ],
        },
    }
    print(json.dumps(trace))


def simulate_action(simulator, task_index, action):
    return simulator.simulate_action(
        task_index,
        action,
        need_images=True,
        need_featurized_objects=True,
        stride=1,
    )


def sample_action(
    simulator,
    task_index,
    sample_seed,
    sampling_strategy,
    min_dense_frames,
    max_sample_attempts,
    solve_sample_attempts,
):
    if sampling_strategy == "random":
        rng = np.random.RandomState(sample_seed)
        action = simulator.sample(valid_only=True, rng=rng)
        return action, simulate_action(simulator, task_index, action), 1
    if sampling_strategy == "informative":
        action, simulation, search_attempts = sample_informative_action(
            simulator,
            task_index=task_index,
            sample_seed=sample_seed,
            min_dense_frames=min_dense_frames,
            max_sample_attempts=max_sample_attempts,
        )
        return action, simulation, search_attempts
    if sampling_strategy == "seek_solution":
        return sample_solved_action(
            simulator,
            task_index=task_index,
            sample_seed=sample_seed,
            solve_sample_attempts=solve_sample_attempts,
        )
    raise ValueError("Unsupported PHYRE sampling strategy: %s" % sampling_strategy)


def sample_solved_action(simulator, task_index, sample_seed, solve_sample_attempts):
    for offset in range(solve_sample_attempts):
        rng = np.random.RandomState(sample_seed + offset)
        action = simulator.sample(valid_only=True, rng=rng)
        simulation = simulate_action(simulator, task_index, action)
        status = getattr(simulation, "status", None)
        if status is not None and hasattr(status, "is_solved") and status.is_solved():
            return action, simulation, offset + 1
    raise RuntimeError(
        "Failed to sample a solved PHYRE action after %d attempts" % solve_sample_attempts
    )


def sample_informative_action(simulator, task_index, sample_seed, min_dense_frames, max_sample_attempts):
    best_action = None
    best_simulation = None
    best_frame_count = -1
    for offset in range(max_sample_attempts):
        rng = np.random.RandomState(sample_seed + offset)
        action = simulator.sample(valid_only=True, rng=rng)
        simulation = simulate_action(simulator, task_index, action)
        featurized_objects = getattr(simulation, "featurized_objects", None)
        frame_count = 0
        if featurized_objects is not None and getattr(featurized_objects, "states", None) is not None:
            frame_count = effective_frame_count(featurized_objects.states)
        if frame_count > best_frame_count:
            best_action = action
            best_simulation = simulation
            best_frame_count = frame_count
        if frame_count >= min_dense_frames:
            return action, simulation, offset + 1

    if best_action is None or best_simulation is None:
        raise RuntimeError("Failed to sample a valid PHYRE action")
    return best_action, best_simulation, max_sample_attempts


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-tasks")
    list_parser.add_argument("--eval-setup", default="ball_within_template")
    list_parser.add_argument("--fold-id", type=int, default=0)
    list_parser.add_argument("--eval-type", default="all", choices=["train", "test", "all"])
    list_parser.set_defaults(func=list_tasks)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--task-id", required=True)
    generate_parser.add_argument("--output-dir", required=True)
    generate_parser.add_argument("--sample-index", type=int, default=0)
    generate_parser.add_argument("--sample-seed", type=int, default=0)
    generate_parser.add_argument(
        "--sampling-strategy",
        default="random",
        choices=["random", "informative", "seek_solution"],
    )
    generate_parser.add_argument("--min-dense-frames", type=int, default=64)
    generate_parser.add_argument("--max-sample-attempts", type=int, default=8)
    generate_parser.add_argument("--solve-sample-attempts", type=int, default=256)
    generate_parser.add_argument("--eval-setup", default="ball_within_template")
    generate_parser.add_argument("--fold-id", type=int, default=0)
    generate_parser.add_argument("--eval-type", default="all", choices=["train", "test", "all"])
    generate_parser.set_defaults(func=generate_trace)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
