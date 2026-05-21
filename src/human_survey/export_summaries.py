#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SERVER_ROOT = ROOT.parents[1]
SERVER_SRC = SERVER_ROOT / "src"
DATA_ROOT = ROOT / "data"
ITEMS_ROOT = DATA_ROOT / "items"
REPORT_PATH = DATA_ROOT / "extraction_report.md"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
TARGET_ENVS = ("phyre", "iphyre", "pooltool")
REQUIRED_VARIANTS = ("image_only", "supervised", "unsupervised")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip()).strip("-._").lower()
    return slug or "item"


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ExportedItem:
    item_id: str
    env_id: str
    task_id: str
    source_run_id: str
    video_path: str
    initial_image_path: str
    summaries: dict[str, dict[str, Any]]
    available_summary_variants: list[str]
    annotation_pattern_counts: dict[str, int | None]
    summary_sources: dict[str, str | None]


def main() -> None:
    server_python = SERVER_ROOT / ".venv" / "bin" / "python"
    if os.environ.get("HUMAN_SURVEY_EXPORT_BOOTSTRAPPED") != "1" and server_python.is_file():
        try:
            import pydantic  # type: ignore  # noqa: F401
        except ModuleNotFoundError:
            env = dict(os.environ)
            env["HUMAN_SURVEY_EXPORT_BOOTSTRAPPED"] = "1"
            os.execve(str(server_python), [str(server_python), str(Path(__file__).resolve()), *sys.argv[1:]], env)

    sys.path.insert(0, str(SERVER_SRC))
    from pattern_server.task_summaries import list_completed_task_summaries

    if DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)
    ITEMS_ROOT.mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "responses").mkdir(parents=True, exist_ok=True)

    entries_by_env = list_completed_task_summaries(SERVER_ROOT)
    exported_items: list[ExportedItem] = []
    report_lines: list[str] = [
        "# Extraction Report",
        "",
        f"Generated: `{_utc_now_iso()}`",
        "",
        "Only entries with a working video plus `image_only`, `supervised`, and `unsupervised` summaries are included.",
        "",
    ]

    env_counts: dict[str, int] = {}
    for env_id in TARGET_ENVS:
        entries = list(entries_by_env.get(env_id, []))
        included = 0
        skipped: list[tuple[str, str]] = []
        report_lines.append(f"## {env_id.upper()}")
        report_lines.append("")
        report_lines.append("| Task | Status | Notes |")
        report_lines.append("| --- | --- | --- |")
        for entry in entries:
            variants = entry.summary_payload.get("variants") if isinstance(entry.summary_payload, dict) else {}
            if not isinstance(variants, dict):
                skipped.append((entry.task_id, "missing variants payload"))
                report_lines.append(f"| `{entry.task_id}` | skipped | missing variants payload |")
                continue
            video_src = entry.overhead_video_path if env_id == "pooltool" and entry.overhead_video_path and entry.overhead_video_path.is_file() else entry.video_path
            if not (video_src and video_src.is_file()):
                skipped.append((entry.task_id, "missing video"))
                report_lines.append(f"| `{entry.task_id}` | skipped | missing video |")
                continue
            if not (entry.initial_image_path and entry.initial_image_path.is_file()):
                skipped.append((entry.task_id, "missing initial image"))
                report_lines.append(f"| `{entry.task_id}` | skipped | missing initial image |")
                continue
            missing_variants = [
                key
                for key in REQUIRED_VARIANTS
                if not (
                    isinstance(variants.get(key), dict)
                    and str(variants.get(key, {}).get("summary_text") or "").strip()
                )
            ]
            if missing_variants:
                skipped.append((entry.task_id, f"missing variants: {', '.join(missing_variants)}"))
                report_lines.append(f"| `{entry.task_id}` | skipped | missing variants: {', '.join(missing_variants)} |")
                continue

            item_id = f"{env_id}-{_slugify(entry.task_id)}"
            item_dir = ITEMS_ROOT / env_id / item_id
            item_dir.mkdir(parents=True, exist_ok=True)

            video_dest = item_dir / "video.mp4"
            image_dest = item_dir / f"initial_image{entry.initial_image_path.suffix.lower()}"
            _copy_file(video_src, video_dest)
            _copy_file(entry.initial_image_path, image_dest)

            summaries: dict[str, dict[str, Any]] = {}
            annotation_pattern_counts: dict[str, int | None] = {}
            summary_sources: dict[str, str | None] = {}
            for variant_key in REQUIRED_VARIANTS:
                payload = dict(variants[variant_key])
                summary_text = str(payload.get("summary_text") or "").strip()
                summary_path = item_dir / f"summary_{variant_key}.json"
                summary_json = {
                    "item_id": item_id,
                    "env_id": env_id,
                    "task_id": entry.task_id,
                    "source_run_id": entry.source_run_id,
                    "summary_variant": variant_key,
                    "summary_text": summary_text,
                    "annotation_pattern_count": payload.get("annotation_pattern_count"),
                    "summary_source": payload.get("summary_source"),
                }
                summary_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
                summaries[variant_key] = {
                    "path": str(summary_path.relative_to(ROOT)).replace("\\", "/"),
                    "summary_text": summary_text,
                    "annotation_pattern_count": payload.get("annotation_pattern_count"),
                    "summary_source": payload.get("summary_source"),
                }
                annotation_pattern_counts[variant_key] = payload.get("annotation_pattern_count")
                summary_sources[variant_key] = payload.get("summary_source")

            item_payload = ExportedItem(
                item_id=item_id,
                env_id=env_id,
                task_id=entry.task_id,
                source_run_id=entry.source_run_id,
                video_path=str(video_dest.relative_to(ROOT)).replace("\\", "/"),
                initial_image_path=str(image_dest.relative_to(ROOT)).replace("\\", "/"),
                summaries=summaries,
                available_summary_variants=list(REQUIRED_VARIANTS),
                annotation_pattern_counts=annotation_pattern_counts,
                summary_sources=summary_sources,
            )
            (item_dir / "item.json").write_text(json.dumps(asdict(item_payload), indent=2), encoding="utf-8")
            exported_items.append(item_payload)
            included += 1
            report_lines.append(f"| `{entry.task_id}` | included | video + image + all required summaries present |")

        env_counts[env_id] = included
        report_lines.append("")
        report_lines.append(f"Included: `{included}`")
        report_lines.append("")
        if skipped:
            report_lines.append(f"Skipped: `{len(skipped)}`")
            report_lines.append("")

    manifest = {
        "schema_version": "human_survey_manifest.v1",
        "generated_at": _utc_now_iso(),
        "questions_per_user": 12,
        "seed_source": "PROLIFIC_PID (fallback: generated participant id)",
        "environments": list(TARGET_ENVS),
        "summary_variants": [
            {"key": "image_only", "label": "Image summary"},
            {"key": "supervised", "label": "Supervised summary"},
            {"key": "unsupervised", "label": "Unsupervised summary"},
        ],
        "item_count": len(exported_items),
        "env_counts": env_counts,
        "items": [asdict(item) for item in exported_items],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"wrote {len(exported_items)} items to {DATA_ROOT}")


if __name__ == "__main__":
    main()
