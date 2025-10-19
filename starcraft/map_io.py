from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


PASSABLE_TILES = {".", "G", "S"}


def load_grid(map_path: Path) -> np.ndarray:
    """
    Load a Moving AI StarCraft map and return a boolean walkability grid.

    Args:
        map_path: Path to the `.map` file (Moving AI format).

    Returns:
        A numpy array of shape (height, width) with True for walkable tiles.
    """
    with map_path.open("r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle]

    try:
        idx = lines.index("map")
    except ValueError as exc:
        raise ValueError(f"Map file {map_path} is missing the 'map' header") from exc

    grid_lines = lines[idx + 1 :]
    if not grid_lines:
        raise ValueError(f"Map file {map_path} does not contain any grid rows")

    width = len(grid_lines[0])
    height = len(grid_lines)

    grid: List[List[bool]] = []
    for row in grid_lines:
        if len(row) != width:
            raise ValueError(f"Row length mismatch in {map_path}: expected {width}, got {len(row)}")

        grid_row = []
        for char in row:
            if char == ".":
                grid_row.append(True)
            elif char in {"@", "T"}:
                grid_row.append(False)
            else:
                # Fallback to passable for unknown terrain to avoid silently blocking maps.
                grid_row.append(char in PASSABLE_TILES)
        grid.append(grid_row)

    return np.array(grid, dtype=bool)
