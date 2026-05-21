#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESPONSES_ROOT = ROOT / "data" / "responses"


def main() -> None:
    by_variant = defaultdict(list)
    by_env = defaultdict(list)
    response_files = sorted(RESPONSES_ROOT.glob("user_*.json"))
    for path in response_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict) or item.get("task") != "rating":
                continue
            rating = item.get("rating")
            if not isinstance(rating, int):
                continue
            variant = str(item.get("summary_variant") or "").strip()
            env_id = str(item.get("env_id") or "").strip()
            if variant:
                by_variant[variant].append(rating)
            if env_id:
                by_env[env_id].append(rating)

    print("Response files:", len(response_files))
    print("\nBy summary variant")
    for key, values in sorted(by_variant.items()):
        avg = sum(values) / len(values) if values else 0.0
        print(f"- {key}: n={len(values)} avg={avg:.2f}")
    print("\nBy environment")
    for key, values in sorted(by_env.items()):
        avg = sum(values) / len(values) if values else 0.0
        print(f"- {key}: n={len(values)} avg={avg:.2f}")


if __name__ == "__main__":
    main()
