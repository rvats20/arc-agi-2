"""Grid rendering: ASCII for debugging, PIL image for Qwen2.5-VL input.

ARC grids are list-of-lists of ints 0-9. We render each color as a distinct
filled cell so the vision model can read them. 0 is treated as background.
"""
from __future__ import annotations

from PIL import Image, ImageDraw

# Distinct, high-contrast colors for 0..9 (RGB). Index 0 = black background.
ARC_COLORS = [
    (0, 0, 0),        # 0 black
    (255, 0, 0),      # 1 red
    (0, 200, 0),      # 2 green
    (0, 100, 255),    # 3 blue
    (255, 255, 0),    # 4 yellow
    (255, 0, 255),    # 5 magenta
    (0, 255, 255),    # 6 cyan
    (180, 180, 180),  # 7 grey
    (150, 75, 0),     # 8 brown
    (255, 255, 255),  # 9 white
]

ASCII_CHARS = " .123456789"  # index aligns with color value 0..9


def grid_to_ascii(grid: list[list[int]], sep: str = " ") -> str:
    rows = []
    for r in grid:
        rows.append(sep.join(ASCII_CHARS[v] if 0 <= v < 10 else "?" for v in r))
    return "\n".join(rows)


def grid_to_image(grid: list[list[int]], cell: int = 32, border: int = 2) -> Image.Image:
    """Render a grid as a clean PIL image with colored cells.

    cell  - pixel size of each grid cell
    border- black gap between cells so the model can see grid lines
    """
    h = len(grid)
    w = len(grid[0]) if h else 0
    if w == 0:
        return Image.new("RGB", (cell, cell), (0, 0, 0))
    img = Image.new("RGB", (w * cell, h * cell), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    for y, row in enumerate(grid):
        for x, v in enumerate(row):
            color = ARC_COLORS[v] if 0 <= v < 10 else (80, 80, 80)
            draw.rectangle(
                [x * cell + border, y * cell + border,
                 (x + 1) * cell - border, (y + 1) * cell - border],
                fill=color,
            )
    return img


def render_task_thumbnails(task, cell: int = 24) -> list[Image.Image]:
    """Render every train/test pair (input+output) as images for the model prompt."""
    imgs: list[Image.Image] = []
    for i, pair in enumerate(task.train):
        imgs.append(grid_to_image(pair["input"], cell))
        if "output" in pair:
            imgs.append(grid_to_image(pair["output"], cell))
    for pair in task.test:
        imgs.append(grid_to_image(pair["input"], cell))
    return imgs
