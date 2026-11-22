"""Local pixel evaluation compatible with the xView2 reference scorer."""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from aeris.data.labels import DAMAGE_CLASS_NAMES, IGNORE

_IMAGE_SHAPE = (1024, 1024)
_CLASS_COUNT = 5  # Background and the four damage classes.
_DAMAGE_EPSILON = 1e-6
_TARGET_NAME = re.compile(r"(test|hold)_(localization|damage)_([^_]+)_target\.png")


@dataclass(frozen=True, slots=True)
class MaskPair:
    localization_prediction: np.ndarray
    damage_prediction: np.ndarray
    localization_target: np.ndarray
    damage_target: np.ndarray
    sample_id: str = ""


@dataclass(frozen=True, slots=True)
class PixelMetrics:
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def support(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def precision(self) -> float:
        predicted = self.true_positive + self.false_positive
        return self.true_positive / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        return self.true_positive / self.support if self.support else 0.0

    @property
    def f1(self) -> float:
        union = self.support + self.false_positive
        denominator = self.true_positive + union
        return 2 * self.true_positive / denominator if denominator else 0.0

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "support": self.support,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True, slots=True)
class XView2Score:
    image_count: int
    localization: PixelMetrics
    damage_by_class: dict[str, PixelMetrics]
    """Damage metrics for the four classes, keyed by their project names."""
    damage_confusion: tuple[tuple[int, ...], ...]
    """Rows are target IDs and columns are prediction IDs, ordered 0 through 4."""

    @property
    def damage_f1(self) -> float:
        reciprocals = sum(
            1 / (metric.f1 + _DAMAGE_EPSILON)
            for metric in self.damage_by_class.values()
        )
        return len(self.damage_by_class) / reciprocals

    @property
    def score(self) -> float:
        return 0.3 * self.localization.f1 + 0.7 * self.damage_f1

    def to_dict(self) -> dict[str, object]:
        """Include the reference JSON keys plus counts and diagnostic metrics."""
        localization = self.localization.to_dict()
        union = self.localization.support + self.localization.false_positive
        localization["iou"] = self.localization.true_positive / union if union else 0.0

        damage_by_class = {}
        for name, metric in self.damage_by_class.items():
            row = metric.to_dict()
            if metric.support == 0:
                row["recall"] = None
                row["f1"] = None
            damage_by_class[name] = row

        return {
            "score": self.score,
            "damage_f1": self.damage_f1,
            "localization_f1": self.localization.f1,
            **{
                f"damage_f1_{name}": metric.f1
                for name, metric in self.damage_by_class.items()
            },
            "image_count": self.image_count,
            "localization": localization,
            "damage_by_class": damage_by_class,
            "damage_confusion": self.damage_confusion,
        }


def score_mask_pairs(
    pairs: Iterable[MaskPair], *, allow_ignored_targets: bool = True
) -> XView2Score:
    """Pool pixel counts before scoring. Internal damage targets may use IGNORE."""
    localization_tp = 0
    localization_fp = 0
    localization_fn = 0
    damage = np.zeros((_CLASS_COUNT, _CLASS_COUNT), dtype=np.int64)
    image_count = 0

    for pair in pairs:
        image_count += 1
        label = pair.sample_id or f"image {image_count}"
        _validate_mask(pair.localization_prediction, f"{label}: localization prediction")
        _validate_mask(pair.localization_target, f"{label}: localization target")
        _validate_mask(pair.damage_prediction, f"{label}: damage prediction")
        _validate_mask(
            pair.damage_target,
            f"{label}: damage target",
            allow_ignore=allow_ignored_targets,
        )

        located_prediction = pair.localization_prediction != 0
        located_target = pair.localization_target != 0
        true_positive = int(np.count_nonzero(located_prediction & located_target))
        localization_tp += true_positive
        localization_fp += int(np.count_nonzero(located_prediction)) - true_positive
        localization_fn += int(np.count_nonzero(located_target)) - true_positive

        scored = (pair.damage_target != 0) & (pair.damage_target != IGNORE)
        damage_predictions = np.where(
            located_prediction[scored], pair.damage_prediction[scored], np.uint8(0)
        )
        encoded = _CLASS_COUNT * pair.damage_target[scored] + damage_predictions
        damage += np.bincount(encoded, minlength=damage.size).reshape(damage.shape)

    if image_count == 0:
        raise ValueError("No mask pairs to score")

    damage_metrics = {}
    for class_id, name in DAMAGE_CLASS_NAMES.items():
        tp = int(damage[class_id, class_id])
        damage_metrics[name] = PixelMetrics(
            true_positive=tp,
            false_positive=int(damage[:, class_id].sum()) - tp,
            false_negative=int(damage[class_id].sum()) - tp,
        )
    return XView2Score(
        image_count=image_count,
        localization=PixelMetrics(localization_tp, localization_fp, localization_fn),
        damage_by_class=damage_metrics,
        damage_confusion=tuple(tuple(int(count) for count in row) for row in damage),
    )


def _validate_mask(mask: np.ndarray, label: str, *, allow_ignore: bool = False) -> None:
    if mask.shape != _IMAGE_SHAPE or mask.dtype != np.uint8:
        raise ValueError(f"{label}: expected a 1024×1024 uint8 mask")
    invalid = mask >= _CLASS_COUNT
    if allow_ignore:
        invalid &= mask != IGNORE
    if np.any(invalid):
        allowed = "0–4 or 255" if allow_ignore else "0–4"
        raise ValueError(f"{label}: pixel values must be {allowed}")


def score_directories(predictions: Path, targets: Path) -> XView2Score:
    """Score complete pairs in the target directory using xView2 PNG filenames."""
    for directory in (predictions, targets):
        if not directory.is_dir():
            raise ValueError(f"Mask directory not found: {directory}")
    return score_mask_pairs(_read_pairs(predictions, targets), allow_ignored_targets=False)


def _read_pairs(predictions: Path, targets: Path) -> Iterator[MaskPair]:
    grouped: dict[tuple[str, str], dict[str, Path]] = {}
    for path in sorted(targets.glob("*.png")):
        match = _TARGET_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Invalid xView2 target filename: {path.name}")
        split, task, image_id = match.groups()
        grouped.setdefault((split, image_id), {})[task] = path

    for (split, image_id), paths in grouped.items():
        label = f"{split}/{image_id}"
        missing = {"localization", "damage"} - paths.keys()
        if missing:
            raise ValueError(f"{label}: missing {' and '.join(sorted(missing))} target PNG")
        yield MaskPair(
            localization_prediction=_read_png(
                predictions / f"{split}_localization_{image_id}_prediction.png"
            ),
            damage_prediction=_read_png(
                predictions / f"{split}_damage_{image_id}_prediction.png"
            ),
            localization_target=_read_png(paths["localization"]),
            damage_target=_read_png(paths["damage"]),
            sample_id=label,
        )


def _read_png(path: Path) -> np.ndarray:
    if not path.is_file():
        raise ValueError(f"Mask PNG not found: {path}")
    with Image.open(path, formats=["PNG"]) as image:
        return np.asarray(image)
