"""Rasterize building polygons into localization and damage masks.

Pixel convention: pixel (row, column) belongs to a polygon when its center
(column + 0.5, row + 0.5) lies inside the polygon or on its boundary. Whether
this matches the official xView2 target PNGs is checked by the data audit.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import Polygon

from aeris.data.labels import IGNORE, Building


@dataclass(frozen=True, slots=True)
class Raster:
    mask: np.ndarray
    """uint8 array of shape (height, width)."""
    empty_buildings: tuple[int, ...]
    """Indices of buildings that cover no pixel center."""
    overlap_pixels: int
    """Pixels covered by more than one building."""


def building_pixels(
    polygon: Polygon, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (rows, columns) of the pixels inside the image that belong to ``polygon``."""
    min_x, min_y, max_x, max_y = polygon.bounds
    first_column = max(0, math.ceil(min_x - 0.5))
    last_column = min(width - 1, math.floor(max_x - 0.5))
    first_row = max(0, math.ceil(min_y - 0.5))
    last_row = min(height - 1, math.floor(max_y - 0.5))
    if first_column > last_column or first_row > last_row:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty

    columns, rows = np.meshgrid(
        np.arange(first_column, last_column + 1),
        np.arange(first_row, last_row + 1),
    )
    inside = shapely.intersects_xy(polygon, columns + 0.5, rows + 0.5)
    return rows[inside], columns[inside]


def rasterize_localization(
    buildings: Sequence[Building], width: int, height: int
) -> Raster:
    """Building pixels are 1, background is 0."""
    mask = np.zeros((height, width), dtype=np.uint8)
    empty, overlap, pixels = _building_pixels(buildings, width, height)
    for _, rows, columns in pixels:
        mask[rows, columns] = 1
    return Raster(mask, empty, overlap)


def rasterize_damage(buildings: Sequence[Building], width: int, height: int) -> Raster:
    """Damage classes 1-4 and :data:`IGNORE`; overlaps take the most severe class.

    A classified building wins over an ignored one where they overlap.
    """
    if any(building.damage_class is None for building in buildings):
        raise ValueError("Damage masks need post-disaster buildings with damage labels.")

    mask = np.zeros((height, width), dtype=np.uint8)
    empty, overlap, pixels = _building_pixels(buildings, width, height)
    for building, rows, columns in sorted(pixels, key=lambda item: _priority(item[0])):
        mask[rows, columns] = building.damage_class
    return Raster(mask, empty, overlap)


def _priority(building: Building) -> int:
    return -1 if building.damage_class == IGNORE else building.damage_class


def _building_pixels(
    buildings: Sequence[Building], width: int, height: int
) -> tuple[tuple[int, ...], int, list[tuple[Building, np.ndarray, np.ndarray]]]:
    coverage = np.zeros((height, width), dtype=np.uint16)
    empty: list[int] = []
    pixels: list[tuple[Building, np.ndarray, np.ndarray]] = []
    for index, building in enumerate(buildings):
        rows, columns = building_pixels(building.polygon, width, height)
        if rows.size == 0:
            empty.append(index)
            continue
        coverage[rows, columns] += 1
        pixels.append((building, rows, columns))
    return tuple(empty), int(np.count_nonzero(coverage > 1)), pixels
