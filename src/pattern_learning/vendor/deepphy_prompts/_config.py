from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def load_prompt_config(filename: str) -> SimpleNamespace:
    config_path = Path(__file__).resolve().parent / "configs" / filename
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SimpleNamespace(**payload)
