"""Compare AERIS masks with the official xView2 target PNGs."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from aeris.data.labels import IGNORE


@dataclass(frozen=True, slots=True)
class TargetComparison:
    compared_pixels: int
    mismatched_pixels: int
    ignore_pixels: int
    """Pixels AERIS marks as :data:`IGNORE`; excluded from the comparison."""
    target_values_on_ignore: dict[int, int]
    """How the official target labels the pixels AERIS ignores."""


def target_path(image_path: Path) -> Path:
    """``images/<stem>.png`` → ``targets/<stem>_target.png``."""
    return image_path.parent.parent / "targets" / f"{image_path.stem}_target.png"


def compare_target(ours: np.ndarray, path: Path, *, binary: bool) -> TargetComparison:
    """Compare a mask with a target PNG. ``binary`` compares building vs background."""
    with Image.open(path) as image:
        target = np.asarray(image)
    if target.shape != ours.shape:
        raise ValueError(f"{path}: target shape {target.shape}, expected {ours.shape}")

    if binary:
        mismatched = np.count_nonzero((target > 0) != (ours > 0))
        return TargetComparison(int(ours.size), int(mismatched), 0, {})

    ignored = ours == IGNORE
    compared = ~ignored
    mismatched = np.count_nonzero(target[compared] != ours[compared])
    values, counts = np.unique(target[ignored], return_counts=True)
    return TargetComparison(
        int(np.count_nonzero(compared)),
        int(mismatched),
        int(np.count_nonzero(ignored)),
        {int(value): int(count) for value, count in zip(values, counts)},
    )
