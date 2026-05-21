from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def add_grid_with_matplotlib(
    image_path: str,
    grid_size: tuple[int, int] | list[int],
    line_color: str = "red",
    line_width: float = 1.5,
    fontsize: int = 10,
) -> str | None:
    if not os.path.exists(image_path):
        return None
    try:
        img = mpimg.imread(image_path)
        height, width, _ = img.shape
        dpi = 100
        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax.imshow(img)
        rows, cols = tuple(grid_size)
        for i in range(1, cols):
            ax.axvline(x=(width / cols) * i, color=line_color, linewidth=line_width)
        for i in range(1, rows):
            ax.axhline(y=(height / rows) * i, color=line_color, linewidth=line_width)

        cell_number = 1
        cell_height = height / rows
        cell_width = width / cols
        for r in range(rows):
            for c in range(cols):
                ax.text(
                    cell_width * (c + 0.5),
                    cell_height * (r + 0.5),
                    str(cell_number),
                    color=line_color,
                    fontsize=fontsize,
                    fontweight="bold",
                    ha="center",
                    va="center",
                )
                cell_number += 1

        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_{rows}x{cols}_grid{ext}"
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
        plt.close(fig)
        return output_path
    except Exception:
        return None


def get_normalized_center_coords(cell_number: int, grid_size: tuple[int, int] | list[int]) -> tuple[float, float] | None:
    rows, cols = tuple(grid_size)
    if not (1 <= cell_number <= rows * cols):
        return None
    index = cell_number - 1
    row_from_top = index // cols
    col_from_left = index % cols
    center_x = (col_from_left + 0.5) / cols
    center_y = ((rows - 1 - row_from_top) + 0.5) / rows
    return (center_x, center_y)


def convert_radius_size_to_normalized(radius_size: int, max_level: int) -> float | None:
    if not (1 <= radius_size <= max_level):
        return None

    min_output_r = 0.0625
    # PHYRE rejects many large placements when the radius is scaled too
    # aggressively. Keep level 1 unchanged, but cap the largest discrete size to
    # a smaller normalized radius so level 8 remains usable.
    max_output_r = 0.5

    if max_level == 1:
        return max_output_r

    ratio = (radius_size - 1) / (max_level - 1)
    output_range_span = max_output_r - min_output_r
    normalized_r = min_output_r + ratio * output_range_span
    return normalized_r
