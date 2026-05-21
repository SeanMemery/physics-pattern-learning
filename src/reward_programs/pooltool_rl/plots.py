from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

from .config import project_root


def write_line_plot_svg(
    path: Path,
    *,
    title: str | None,
    x_label: str,
    y_label: str,
    series: dict[str, list[tuple[float, float]]],
    error_series: dict[str, list[tuple[float, float]]] | None = None,
    figure_size: str = "default",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = _load_plot_config()
    plt.style.use(config.get("style", "default"))
    for key, value in (config.get("rcParams") or {}).items():
        plt.rcParams[key] = value
    _apply_font_fallbacks(plt.rcParams)

    figure_sizes = config.get("figure_sizes") or {}
    figsize = figure_sizes.get(figure_size) or figure_sizes.get("default") or [3.3, 2.2]
    fig, ax = plt.subplots(figsize=figsize, dpi=((config.get("aesthetics") or {}).get("figure") or {}).get("dpi", 300))

    aesthetics = config.get("aesthetics") or {}
    figure_cfg = aesthetics.get("figure") or {}
    grid_cfg = aesthetics.get("grid") or {}
    errorbar_cfg = aesthetics.get("errorbar") or {}
    error_region_cfg = aesthetics.get("error_region") or {}
    legend_cfg = aesthetics.get("legend") or {}
    colors = config.get("colors") or ["#0b6e4f", "#b02e0c", "#1d4ed8", "#a16207", "#be185d", "#374151"]
    markers = ((config.get("markers") or {}).get("scatter")) or ["o", "s", "D", "^", "v", ">", "<", "p", "h"]

    fig.patch.set_facecolor(figure_cfg.get("facecolor", "white"))
    ax.grid(
        True,
        linestyle=grid_cfg.get("linestyle", "-"),
        linewidth=grid_cfg.get("linewidth", 0.25),
        alpha=grid_cfg.get("alpha", 0.2),
        color=grid_cfg.get("color", "#bbbbbb"),
    )

    all_x = [x for points in series.values() for x, _ in points]
    if not all_x:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        for index, (name, points) in enumerate(series.items()):
            if not points:
                continue
            ordered = sorted(points, key=lambda item: item[0])
            xs = [float(x) for x, _ in ordered]
            ys = [float(y) for _, y in ordered]
            errs = _ordered_errors(error_series, name, xs)
            color = colors[index % len(colors)]
            marker = markers[index % len(markers)]
            ax.plot(xs, ys, color=color, marker=marker, label=name)
            if errs is not None:
                lower = [max(0.0, y - e) for y, e in zip(ys, errs)]
                upper = [min(1.0, y + e) for y, e in zip(ys, errs)]
                ax.fill_between(xs, lower, upper, color=color, alpha=error_region_cfg.get("alpha", 0.15))
        ax.set_ylim(0.0, 1.0)
        configured_xticks = _configured_xticks(config, x_label)
        ax.set_xticks(configured_xticks if configured_xticks is not None else sorted(set(float(x) for x in all_x)))

    labels_cfg = config.get("labels") or {}
    ax.set_xlabel(x_label, labelpad=labels_cfg.get("padding", 10))
    ax.set_ylabel(y_label, labelpad=labels_cfg.get("padding", 10))
    if title:
        ax.set_title(title, pad=labels_cfg.get("title_padding", 20))
    if any(points for points in series.values()):
        ax.legend(
            frameon=legend_cfg.get("frameon", True),
            facecolor=legend_cfg.get("facecolor", "white"),
            framealpha=legend_cfg.get("framealpha", 1.0),
            shadow=legend_cfg.get("shadow", False),
            fancybox=legend_cfg.get("fancybox", True),
        )
    fig.tight_layout()
    output_format = path.suffix.lstrip(".").lower() or "pdf"
    fig.savefig(path, format=output_format)
    plt.close(fig)
    return path


def _load_plot_config() -> dict:
    target = project_root() / "plot.yml"
    if not target.is_file():
        return {}
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _ordered_errors(
    error_series: dict[str, list[tuple[float, float]]] | None,
    name: str,
    xs: Iterable[float],
) -> list[float] | None:
    if not error_series or name not in error_series:
        return None
    mapping = {float(x): float(err) for x, err in error_series[name]}
    return [mapping.get(float(x), 0.0) for x in xs]


def _configured_xticks(config: dict, x_label: str) -> list[float] | None:
    x_ticks = config.get("x_ticks") or {}
    values = x_ticks.get(x_label)
    if not isinstance(values, list) or not values:
        return None
    return [float(value) for value in values]


def _apply_font_fallbacks(rc_params) -> None:
    try:
        from matplotlib import font_manager
    except Exception:
        return
    available = {font.name for font in font_manager.fontManager.ttflist}
    configured_serif = rc_params.get("font.serif") or []
    if isinstance(configured_serif, str):
        configured_serif = [configured_serif]
    resolved_serif = [name for name in configured_serif if name in available]
    if not resolved_serif:
        resolved_serif = ["DejaVu Serif"]
    rc_params["font.serif"] = resolved_serif
    preferred_math_font = resolved_serif[0]
    if preferred_math_font in available:
        rc_params["mathtext.rm"] = preferred_math_font
        rc_params["mathtext.it"] = f"{preferred_math_font}:italic"
        rc_params["mathtext.bf"] = f"{preferred_math_font}:bold"
