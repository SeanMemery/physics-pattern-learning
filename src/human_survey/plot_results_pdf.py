#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from json import JSONDecoder
from pathlib import Path

import matplotlib.pyplot as plt

from plot_style import apply_plot_config, color_cycle, figure_size


ROOT = Path(__file__).resolve().parent
RESPONSES_ROOT = ROOT / "data" / "responses"
DEFAULT_OUTPUT = ROOT / "survey_results_plot.pdf"
DEFAULT_COMMENTS_OUTPUT = ROOT / "survey_comments.txt"
VARIANT_ORDER = ["image_only", "supervised", "unsupervised"]
VARIANT_LABELS = {
    "image_only": "Image only",
    "supervised": "Human-labels",
    "unsupervised": "LLM-labels",
}


def _decode_maybe_concatenated_json(text: str) -> list[object]:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        return [json.loads(stripped)]
    except json.JSONDecodeError:
        decoder = JSONDecoder()
        items: list[object] = []
        index = 0
        length = len(stripped)
        while index < length:
            while index < length and stripped[index].isspace():
                index += 1
            if index >= length:
                break
            try:
                item, next_index = decoder.raw_decode(stripped, index)
            except json.JSONDecodeError:
                break
            items.append(item)
            index = next_index
        return items


def _load_response_records(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    raw_values = _decode_maybe_concatenated_json(text)
    records: list[dict] = []
    for value in raw_values:
        if isinstance(value, list):
            records.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict):
            records.append(value)
    if any(isinstance(item, dict) and item.get("task") == "rating" for item in records):
        return records
    return _extract_records_from_broken_text(text)


def _extract_records_from_broken_text(text: str) -> list[dict]:
    records: list[dict] = []
    seen_ranges: set[tuple[int, int]] = set()
    marker = '"task": "rating"'
    index = 0
    while True:
        hit = text.find(marker, index)
        if hit == -1:
            break
        start = text.rfind("{", 0, hit)
        if start == -1:
            index = hit + len(marker)
            continue
        end = _find_matching_brace(text, start)
        if end is None:
            index = hit + len(marker)
            continue
        span = (start, end)
        if span in seen_ranges:
            index = end
            continue
        seen_ranges.add(span)
        snippet = text[start:end]
        try:
            item = json.loads(snippet)
        except Exception:
            index = end
            continue
        if isinstance(item, dict):
            records.append(item)
        index = end
    return records


def _find_matching_brace(text: str, start: int) -> int | None:
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def collect_variant_ratings(responses_root: Path) -> dict[str, list[int]]:
    by_variant: dict[str, list[int]] = defaultdict(list)
    for path in sorted(responses_root.glob("user_*.json")):
        for item in _load_response_records(path):
            if item.get("task") != "rating":
                continue
            rating = item.get("rating")
            variant = str(item.get("summary_variant") or "").strip()
            if isinstance(rating, int) and 1 <= rating <= 7 and variant in VARIANT_ORDER:
                by_variant[variant].append(rating)
    return by_variant


def collect_comments(responses_root: Path) -> list[dict[str, str]]:
    comments: list[dict[str, str]] = []
    for path in sorted(responses_root.glob("user_*.json")):
        for item in _load_response_records(path):
            if item.get("task") != "comment":
                continue
            comment = str(item.get("comment") or "").strip()
            if not comment:
                continue
            comments.append(
                {
                    "participant_id": str(item.get("participant_id") or ""),
                    "prolific_pid": str(item.get("prolific_pid") or ""),
                    "study_id": str(item.get("study_id") or ""),
                    "session_id": str(item.get("session_id") or ""),
                    "comment": comment,
                }
            )
    return comments


def build_plot(by_variant: dict[str, list[int]], output_path: Path) -> None:
    config = apply_plot_config()
    colors = color_cycle(config)
    variant_colors = {
        variant: colors[index % len(colors)]
        for index, variant in enumerate(VARIANT_ORDER)
    }
    scores = list(range(1, 8))
    wide_width, wide_height = figure_size(config, "wide")
    fig, ax = plt.subplots(figsize=(wide_width * 0.82, wide_height))

    grid_settings = ((config.get("aesthetics") or {}).get("grid") or {})
    legend_settings = ((config.get("aesthetics") or {}).get("legend") or {})
    label_settings = config.get("labels") or {}
    figure_settings = ((config.get("aesthetics") or {}).get("figure") or {})

    fig.patch.set_facecolor(str(figure_settings.get("facecolor") or "white"))

    ymax = 0
    for variant in VARIANT_ORDER:
        ratings = by_variant.get(variant, [])
        counts = [ratings.count(score) for score in scores]
        ymax = max(ymax, max(counts, default=0))
        ax.plot(
            scores,
            counts,
            color=variant_colors[variant],
            label=VARIANT_LABELS[variant],
            linewidth=2.4,
            solid_capstyle="butt",
            solid_joinstyle="miter",
        )
        if ratings:
            average = sum(ratings) / len(ratings)
            ax.axvline(
                average,
                linestyle="--",
                color=variant_colors[variant],
                alpha=0.9,
                linewidth=2.0,
                dash_capstyle="butt",
                dash_joinstyle="miter",
            )

    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_xticks(scores)
    ax.set_xlim(0.9, 7.1)
    ax.set_ylim(0, max(ymax + 1.0, 2))
    ax.grid(axis="y", **grid_settings)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.legend(**legend_settings, title="Variant")
    ax.set_facecolor(str(figure_settings.get("facecolor") or "white"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.labelpad = float(label_settings.get("padding") or 10)
    ax.yaxis.labelpad = float(label_settings.get("padding") or 10)
    fig.subplots_adjust(left=0.11, right=0.995, bottom=0.16, top=0.995)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_comments(comments: list[dict[str, str]], output_path: Path) -> None:
    lines: list[str] = []
    lines.append(f"Comments exported: {len(comments)}")
    lines.append("")
    for index, item in enumerate(comments, start=1):
        lines.append(f"Comment {index}")
        if item["participant_id"]:
            lines.append(f"participant_id: {item['participant_id']}")
        if item["prolific_pid"]:
            lines.append(f"prolific_pid: {item['prolific_pid']}")
        if item["study_id"]:
            lines.append(f"study_id: {item['study_id']}")
        if item["session_id"]:
            lines.append(f"session_id: {item['session_id']}")
        lines.append("comment:")
        lines.append(item["comment"])
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot survey rating distributions to PDF.")
    parser.add_argument(
        "--responses-root",
        type=Path,
        default=RESPONSES_ROOT,
        help="Directory containing saved response JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PDF path.",
    )
    parser.add_argument(
        "--comments-output",
        type=Path,
        default=DEFAULT_COMMENTS_OUTPUT,
        help="Output text file for survey comments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    by_variant = collect_variant_ratings(args.responses_root)
    comments = collect_comments(args.responses_root)
    build_plot(by_variant, args.output)
    write_comments(comments, args.comments_output)
    total = sum(len(values) for values in by_variant.values())
    print(f"Wrote {args.output}")
    print(f"Wrote {args.comments_output}")
    print(f"Ratings counted: {total}")
    print(f"Comments counted: {len(comments)}")
    for variant in VARIANT_ORDER:
        ratings = by_variant.get(variant, [])
        avg = sum(ratings) / len(ratings) if ratings else 0.0
        print(f"{variant}: n={len(ratings)} avg={avg:.2f}")


if __name__ == "__main__":
    main()
