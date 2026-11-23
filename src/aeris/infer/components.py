"""Deterministic connected-component footprints on the original pixel grid."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class Footprint:
    building_id: str
    bbox: tuple[int, int, int, int]
    pixels: np.ndarray


def label_components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mask.ndim != 2 or not mask.size or np.any((mask != 0) & (mask != 1)):
        raise ValueError("A nonempty 2D binary localization mask is required")
    _, labels, stats, _ = cv2.connectedComponentsWithStatsWithAlgorithm(
        mask.astype(np.uint8, copy=False), 8, cv2.CV_32S, cv2.CCL_SAUF,
    )
    return labels, stats[1:]


def candidate_footprints(
    mask: np.ndarray, *, minimum_area: int, minimum_side: int,
) -> tuple[list[Footprint], int]:
    """Use half-open bounds and keep each pixel assigned to its component."""
    if minimum_area < 1 or minimum_side < 1:
        raise ValueError("Minimum area and side must be positive")
    labels, stats = label_components(mask)
    candidates = []
    for index, (left, top, width, height, area) in enumerate(stats.tolist(), start=1):
        if area < minimum_area or min(width, height) < minimum_side:
            continue
        right, bottom = left + width, top + height
        rows, columns = np.nonzero(labels[top:bottom, left:right] == index)
        pixels = (rows + top) * mask.shape[1] + columns + left
        candidates.append(Footprint(f"c{index:05d}", (left, top, right, bottom), pixels))
    return candidates, len(stats) - len(candidates)
