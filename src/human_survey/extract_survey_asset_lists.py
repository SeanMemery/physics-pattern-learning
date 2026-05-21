#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from PIL import Image
from PIL import ImageDraw


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_ROOT = ROOT / "data"
MANIFEST_PATH = DATA_ROOT / "manifest.json"

INITIAL_IMAGES_PATH = DATA_ROOT / "initial_images.json"
SUPERVISED_ANNOTATIONS_PATH = DATA_ROOT / "supervised_annotations.json"
SUPERVISED_SUMMARIES_PATH = DATA_ROOT / "supervised_summaries.json"
COMBINED_PATH = DATA_ROOT / "survey_asset_lists.json"
REPORT_PATH = DATA_ROOT / "survey_asset_lists.md"
SUPERVISED_FLAT_DIR = DATA_ROOT / "flat_exports" / "supervised"
IMAGE_ONLY_FLAT_DIR = DATA_ROOT / "flat_exports" / "image_only"
EXAMPLES_PER_ENV = 4


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _normalize_relpath(path: Path) -> str:
    for anchor in (ROOT, PROJECT_ROOT):
        try:
            return str(path.relative_to(anchor)).replace("\\", "/")
        except ValueError:
            continue
    return str(path)


def _task_summary_root(env_id: str, source_run_id: str, task_id: str) -> Path:
    task_slug = task_id.replace(":", "__")
    return (
        PROJECT_ROOT
        / "data"
        / "evaluations"
        / env_id
        / f"task-summaries-{source_run_id}"
        / "tasks"
        / env_id
        / task_slug
    )


def _video_render_task_root(env_id: str, source_run_id: str, task_id: str) -> Path:
    task_slug = task_id.replace(":", "__")
    return (
        PROJECT_ROOT
        / "data"
        / "evaluations"
        / env_id
        / f"video-render-{source_run_id}"
        / "tasks"
        / env_id
        / task_slug
    )


def _select_export_image_source(item: dict[str, Any]) -> Path:
    initial_image_path = ROOT / str(item["initial_image_path"])
    if not initial_image_path.is_file():
        raise ValueError(f"Missing initial image file: {initial_image_path}")

    env_id = str(item["env_id"])
    if env_id != "pooltool":
        return initial_image_path

    task_root = _video_render_task_root(
        env_id=env_id,
        source_run_id=str(item["source_run_id"]),
        task_id=str(item["task_id"]),
    )
    task_result_path = task_root / "task_result.json"
    if task_result_path.is_file():
        payload = _load_json(task_result_path)
        rendered_initial = str(payload.get("initial_image_path") or "").strip()
        if rendered_initial:
            askew_path = Path(rendered_initial.replace("/var/home/sean/Services/Work/pattern-learning-server", str(PROJECT_ROOT)))
            overhead_path = askew_path.with_name("initial_scene_overhead.png")
            if overhead_path.is_file():
                return overhead_path

    for candidate in sorted(task_root.glob("attempt_*/initial_scene_overhead.png")):
        if candidate.is_file():
            return candidate
    return initial_image_path


def _resolve_supervised_variant(item: dict[str, Any]) -> dict[str, Any]:
    task_root = _task_summary_root(
        env_id=str(item["env_id"]),
        source_run_id=str(item["source_run_id"]),
        task_id=str(item["task_id"]),
    )
    summary_path = task_root / "summary.json"
    annotation_preview = ""
    library_path = ""
    recorded_annotation_path = ""
    if summary_path.is_file():
        payload = _load_json(summary_path)
        variants = payload.get("variants")
        if isinstance(variants, dict):
            supervised = variants.get("supervised")
            if isinstance(supervised, dict):
                annotation_preview = str(supervised.get("annotation_preview") or "").strip()
                library_path = str(supervised.get("library_path") or "").strip()
                recorded_annotation_path = str(supervised.get("annotation_path") or "").strip()
    video_render_root = _video_render_task_root(
        env_id=str(item["env_id"]),
        source_run_id=str(item["source_run_id"]),
        task_id=str(item["task_id"]),
    )
    annotation_path = video_render_root / "annotation_supervised.json"
    if not annotation_path.is_file():
        raise ValueError(f"Missing supervised annotation file in {video_render_root}")
    return {
        "task_summary_path": _normalize_relpath(summary_path) if summary_path.is_file() else "",
        "annotation_path": _normalize_relpath(annotation_path) if annotation_path.is_file() else "",
        "annotation_abs_path": str(annotation_path),
        "annotation_preview": annotation_preview,
        "library_path": library_path,
        "recorded_annotation_path": recorded_annotation_path,
    }


def _format_supervised_annotation(annotation_path: Path) -> str:
    payload = _load_json(annotation_path)
    pattern_annotations = payload.get("pattern_annotations")
    if not isinstance(pattern_annotations, dict):
        return json.dumps(payload, indent=2)

    library_id = str(pattern_annotations.get("library_id") or "").strip()
    metadata = pattern_annotations.get("metadata")
    patterns = pattern_annotations.get("patterns")
    lines: list[str] = []
    if library_id:
        lines.append(f"Library: {library_id}")
    if isinstance(metadata, dict):
        total_pattern_count = metadata.get("total_pattern_count")
        if total_pattern_count is not None:
            lines.append(f"Pattern count: {total_pattern_count}")
    if lines:
        lines.append("")

    if isinstance(patterns, list):
        for pattern in patterns:
            if not isinstance(pattern, dict):
                continue
            label = str(pattern.get("label") or "unknown").strip()
            start_timestep = pattern.get("start_timestep")
            end_timestep = pattern.get("end_timestep")
            if start_timestep is not None and end_timestep is not None:
                if start_timestep == end_timestep:
                    prefix = f"[step={start_timestep}]"
                else:
                    prefix = f"[steps={start_timestep}-{end_timestep}]"
            else:
                prefix = "[timesteps=unknown]"
            params = pattern.get("parameters")
            params_text = ""
            if isinstance(params, dict) and params:
                params_text = json.dumps(params, sort_keys=True)
            line = f"{prefix} {label}"
            if params_text:
                line = f"{line} ({params_text})"
            lines.append(line)

    if not lines:
        return json.dumps(payload, indent=2)
    return "\n".join(lines).strip() + "\n"


def _copy_flat_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _write_png_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image.save(dest, format="PNG")


def _render_phyre_action_image(src: Path, trace_path: Path, dest: Path) -> None:
    payload = _load_json(trace_path)
    timesteps = payload.get("timesteps")
    if not isinstance(timesteps, list) or not timesteps:
        _write_png_copy(src, dest)
        return
    first_timestep = timesteps[0]
    if not isinstance(first_timestep, dict):
        _write_png_copy(src, dest)
        return
    objects = first_timestep.get("objects")
    if not isinstance(objects, list):
        _write_png_copy(src, dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src).convert("RGBA") as image:
        draw = ImageDraw.Draw(image, "RGBA")
        width, height = image.size
        for item in objects:
            if not isinstance(item, dict):
                continue
            if item.get("is_user_input") is not True:
                continue
            try:
                x_value = float(item.get("x"))
                y_value = float(item.get("y"))
                diameter = float(item.get("diameter"))
            except (TypeError, ValueError):
                continue
            center_x = min(max(x_value, 0.0), 1.0) * width
            center_y = (1.0 - min(max(y_value, 0.0), 1.0)) * height
            radius = max(1.0, (diameter * width) / 2.0)
            bbox = [
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ]
            draw.ellipse(bbox, fill=(220, 35, 28, 255), outline=(140, 0, 0, 255), width=max(1, int(round(radius * 0.12))))
        image.save(dest, format="PNG")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _export_examples(
    supervised_manifest: list[dict[str, Any]],
    image_only_manifest: list[dict[str, Any]],
) -> None:
    supervised_examples_dir = SUPERVISED_FLAT_DIR / "examples"
    image_only_examples_dir = IMAGE_ONLY_FLAT_DIR / "examples"
    _reset_dir(supervised_examples_dir)
    _reset_dir(image_only_examples_dir)

    image_only_by_trace = {str(item["trace_name"]): item for item in image_only_manifest}
    selected: list[dict[str, Any]] = []

    def _scene_key(item: dict[str, Any]) -> str:
        env_id = str(item["env_id"])
        task_id = str(item["task_id"])
        if env_id == "phyre":
            return task_id.split(":", 1)[0]
        return task_id

    for env_id in sorted({str(item["env_id"]) for item in supervised_manifest}):
        candidates = [item for item in supervised_manifest if str(item["env_id"]) == env_id]
        candidates.sort(
            key=lambda item: len((SUPERVISED_FLAT_DIR / str(item["annotation_file"])).read_text(encoding="utf-8")),
            reverse=True,
        )
        seen_scene_keys: set[str] = set()
        env_selected: list[dict[str, Any]] = []
        for item in candidates:
            scene_key = _scene_key(item)
            if scene_key in seen_scene_keys:
                continue
            seen_scene_keys.add(scene_key)
            env_selected.append(item)
            if len(env_selected) >= EXAMPLES_PER_ENV:
                break
        selected.extend(env_selected)

    supervised_index: list[dict[str, Any]] = []
    image_only_index: list[dict[str, Any]] = []
    for item in selected:
        trace_name = str(item["trace_name"])
        for key in ("image_file", "annotation_file", "summary_file"):
            src = SUPERVISED_FLAT_DIR / str(item[key])
            dst = supervised_examples_dir / src.name
            shutil.copy2(src, dst)
        supervised_index.append(item)

        image_item = image_only_by_trace[trace_name]
        for key in ("image_file", "summary_file"):
            src = IMAGE_ONLY_FLAT_DIR / str(image_item[key])
            dst = image_only_examples_dir / src.name
            shutil.copy2(src, dst)
        image_only_index.append(image_item)

    (supervised_examples_dir / "manifest.json").write_text(
        json.dumps(
            {
                "generated_at": _utc_now_iso(),
                "examples_per_env": EXAMPLES_PER_ENV,
                "item_count": len(supervised_index),
                "items": supervised_index,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (image_only_examples_dir / "manifest.json").write_text(
        json.dumps(
            {
                "generated_at": _utc_now_iso(),
                "examples_per_env": EXAMPLES_PER_ENV,
                "item_count": len(image_only_index),
                "items": image_only_index,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _export_flat_bundles(items: list[dict[str, Any]]) -> None:
    _reset_dir(SUPERVISED_FLAT_DIR)
    _reset_dir(IMAGE_ONLY_FLAT_DIR)

    supervised_manifest: list[dict[str, Any]] = []
    image_only_manifest: list[dict[str, Any]] = []

    for index, item in enumerate(items, start=1):
        trace_name = f"trace{index}"
        initial_image_path = ROOT / str(item["initial_image_path"])
        image_source_path = _select_export_image_source(item)
        trace_path = (
            _video_render_task_root(
                env_id=str(item["env_id"]),
                source_run_id=str(item["source_run_id"]),
                task_id=str(item["task_id"]),
            )
            / "attempt_001"
            / "trace.json"
        )

        supervised_info = _resolve_supervised_variant(item)
        supervised_summary = dict(item.get("summaries", {}).get("supervised") or {})
        image_only_summary = dict(item.get("summaries", {}).get("image_only") or {})

        supervised_image_dest = SUPERVISED_FLAT_DIR / f"{trace_name}_image.png"
        supervised_annotation_dest = SUPERVISED_FLAT_DIR / f"{trace_name}_annotation.txt"
        supervised_summary_dest = SUPERVISED_FLAT_DIR / f"{trace_name}_summary.txt"

        if str(item["env_id"]) == "phyre" and trace_path.is_file():
            _render_phyre_action_image(image_source_path, trace_path, supervised_image_dest)
        else:
            _write_png_copy(image_source_path, supervised_image_dest)
        annotation_preview = str(supervised_info.get("annotation_preview") or "").strip()
        if annotation_preview:
            annotation_text = annotation_preview + "\n"
        else:
            annotation_text = _format_supervised_annotation(Path(str(supervised_info["annotation_abs_path"])))
        _write_text(supervised_annotation_dest, annotation_text)
        _write_text(
            supervised_summary_dest,
            str(supervised_summary.get("summary_text") or "").strip(),
        )

        supervised_manifest.append(
            {
                "trace_name": trace_name,
                "item_id": str(item["item_id"]),
                "env_id": str(item["env_id"]),
                "task_id": str(item["task_id"]),
                "image_file": supervised_image_dest.name,
                "image_source_path": _normalize_relpath(image_source_path),
                "annotation_file": supervised_annotation_dest.name,
                "summary_file": supervised_summary_dest.name,
            }
        )

        image_only_image_dest = IMAGE_ONLY_FLAT_DIR / f"{trace_name}_image.png"
        image_only_summary_dest = IMAGE_ONLY_FLAT_DIR / f"{trace_name}_summary.txt"
        if str(item["env_id"]) == "phyre" and trace_path.is_file():
            _render_phyre_action_image(image_source_path, trace_path, image_only_image_dest)
        else:
            _write_png_copy(image_source_path, image_only_image_dest)
        _write_text(
            image_only_summary_dest,
            str(image_only_summary.get("summary_text") or "").strip(),
        )
        image_only_manifest.append(
            {
                "trace_name": trace_name,
                "item_id": str(item["item_id"]),
                "env_id": str(item["env_id"]),
                "task_id": str(item["task_id"]),
                "image_file": image_only_image_dest.name,
                "image_source_path": _normalize_relpath(image_source_path),
                "summary_file": image_only_summary_dest.name,
            }
        )

    (SUPERVISED_FLAT_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "generated_at": _utc_now_iso(),
                "item_count": len(supervised_manifest),
                "items": supervised_manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (IMAGE_ONLY_FLAT_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "generated_at": _utc_now_iso(),
                "item_count": len(image_only_manifest),
                "items": image_only_manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _export_examples(supervised_manifest, image_only_manifest)


def main() -> None:
    manifest = _load_json(MANIFEST_PATH)
    items = manifest.get("items")
    if not isinstance(items, list):
        raise ValueError(f"Manifest items payload is missing or malformed: {MANIFEST_PATH}")

    initial_images: list[dict[str, Any]] = []
    supervised_annotations: list[dict[str, Any]] = []
    supervised_summaries: list[dict[str, Any]] = []
    combined_items: list[dict[str, Any]] = []

    for raw_item in items:
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        supervised = _resolve_supervised_variant(item)

        base_fields = {
            "item_id": str(item["item_id"]),
            "env_id": str(item["env_id"]),
            "task_id": str(item["task_id"]),
            "source_run_id": str(item["source_run_id"]),
        }

        initial_record = {
            **base_fields,
            "initial_image_path": str(item["initial_image_path"]),
            "video_path": str(item["video_path"]),
        }
        annotation_record = {
            **base_fields,
            "annotation_path": supervised["annotation_path"],
            "annotation_preview": supervised["annotation_preview"],
            "library_path": supervised["library_path"],
            "task_summary_path": supervised["task_summary_path"],
            "recorded_annotation_path": supervised["recorded_annotation_path"],
        }
        supervised_summary_info = dict(item.get("summaries", {}).get("supervised") or {})
        summary_record = {
            **base_fields,
            "summary_path": str(supervised_summary_info.get("path") or ""),
            "summary_text": str(supervised_summary_info.get("summary_text") or "").strip(),
            "summary_source": str(item.get("summary_sources", {}).get("supervised") or ""),
            "annotation_pattern_count": item.get("annotation_pattern_counts", {}).get("supervised"),
            "task_summary_path": supervised["task_summary_path"],
        }

        initial_images.append(initial_record)
        supervised_annotations.append(annotation_record)
        supervised_summaries.append(summary_record)
        combined_items.append(
            {
                **base_fields,
                "initial_image_path": initial_record["initial_image_path"],
                "video_path": initial_record["video_path"],
                "supervised_annotation_path": annotation_record["annotation_path"],
                "supervised_annotation_preview": annotation_record["annotation_preview"],
                "supervised_summary_path": summary_record["summary_path"],
                "supervised_summary_text": summary_record["summary_text"],
                "summary_source": summary_record["summary_source"],
                "annotation_pattern_count": summary_record["annotation_pattern_count"],
            }
        )

    generated_at = _utc_now_iso()
    common_header = {
        "generated_at": generated_at,
        "item_count": len(combined_items),
    }

    INITIAL_IMAGES_PATH.write_text(
        json.dumps({**common_header, "items": initial_images}, indent=2) + "\n",
        encoding="utf-8",
    )
    SUPERVISED_ANNOTATIONS_PATH.write_text(
        json.dumps({**common_header, "items": supervised_annotations}, indent=2) + "\n",
        encoding="utf-8",
    )
    SUPERVISED_SUMMARIES_PATH.write_text(
        json.dumps({**common_header, "items": supervised_summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    COMBINED_PATH.write_text(
        json.dumps({**common_header, "items": combined_items}, indent=2) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# Survey Asset Lists",
        "",
        f"Generated: `{generated_at}`",
        "",
        f"Items: `{len(combined_items)}`",
        "",
        "Files:",
        f"- `initial_images.json`: `{len(initial_images)}` rows",
        f"- `supervised_annotations.json`: `{len(supervised_annotations)}` rows",
        f"- `supervised_summaries.json`: `{len(supervised_summaries)}` rows",
        f"- `survey_asset_lists.json`: `{len(combined_items)}` rows",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    _export_flat_bundles(items)
    print(f"wrote survey asset lists for {len(combined_items)} items to {DATA_ROOT}")


if __name__ == "__main__":
    main()
