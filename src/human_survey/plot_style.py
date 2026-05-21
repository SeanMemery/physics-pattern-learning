from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import yaml


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_STYLE_PATH = PROJECT_ROOT / "plotting" / "plot_style.yaml"
if not DEFAULT_STYLE_PATH.is_file():
    DEFAULT_STYLE_PATH = ROOT / "plot_style.yaml"


def load_plot_config(path: Path | None = None) -> dict[str, Any]:
    style_path = path or DEFAULT_STYLE_PATH
    payload = yaml.safe_load(style_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Plot config at {style_path} must be a mapping.")
    return payload


def apply_plot_config(path: Path | None = None) -> dict[str, Any]:
    config = load_plot_config(path)
    style_name = str(config.get("style") or "").strip()
    if style_name:
        plt.style.use(style_name)

    rc_params = config.get("rcParams") or {}
    if isinstance(rc_params, dict):
        mpl.rcParams.update(rc_params)
        _resolve_serif_fonts(rc_params)

    figure_settings = ((config.get("aesthetics") or {}).get("figure") or {})
    if isinstance(figure_settings, dict):
        facecolor = figure_settings.get("facecolor")
        dpi = figure_settings.get("dpi")
        if facecolor is not None:
            mpl.rcParams["figure.facecolor"] = facecolor
            mpl.rcParams["savefig.facecolor"] = facecolor
        if dpi is not None:
            mpl.rcParams["figure.dpi"] = dpi
            mpl.rcParams["savefig.dpi"] = dpi
    return config


def _resolve_serif_fonts(rc_params: dict[str, Any]) -> None:
    serif_candidates = rc_params.get("font.serif")
    if not isinstance(serif_candidates, list):
        return
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next(
        (font_name for font_name in serif_candidates if font_name in available_fonts),
        None,
    )
    if chosen is None:
        chosen = "DejaVu Serif" if "DejaVu Serif" in available_fonts else None
    if chosen is None:
        return
    mpl.rcParams["font.family"] = ["serif"]
    mpl.rcParams["font.serif"] = [chosen]
    mpl.rcParams["mathtext.rm"] = chosen
    mpl.rcParams["mathtext.it"] = f"{chosen}:italic"
    mpl.rcParams["mathtext.bf"] = f"{chosen}:bold"


def figure_size(config: dict[str, Any], name: str = "default") -> tuple[float, float]:
    sizes = config.get("figure_sizes") or {}
    values = sizes.get(name) or sizes.get("default") or [3.3, 2.2]
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        return (3.3, 2.2)
    return float(values[0]), float(values[1])


def color_cycle(config: dict[str, Any]) -> list[str]:
    colors = config.get("colors") or []
    return [str(item) for item in colors if isinstance(item, str)] or [
        "#d15d08",
        "#0474b4",
        "#c783a6",
    ]
