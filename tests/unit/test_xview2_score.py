import numpy as np
import pytest
from shapely.geometry import box

from aeris.data.labels import IGNORE, Building
from aeris.data.raster import rasterize_damage, rasterize_localization
from aeris.eval.xview2 import MaskPair, PixelMetrics, score_mask_pairs


def test_score_masks_localizes_damage_and_preserves_ignored_building() -> None:
    localization_prediction, damage_prediction, localization_target, damage_target = (
        np.zeros((1024, 1024), dtype=np.uint8) for _ in range(4)
    )
    localization_prediction[0, :9] = [1, 0, 1, 1, 1, 1, 1, 1, 1]
    localization_target[0, :9] = [1, 1, 1, 1, 1, 1, 1, 0, 1]
    damage_prediction[0, :9] = [1, 1, 2, 3, 2, 3, 4, 3, 1]
    damage_target[0, :9] = [1, 1, 2, 2, 3, 3, 4, 0, IGNORE]

    report = score_mask_pairs(
        [MaskPair(localization_prediction, damage_prediction, localization_target, damage_target)]
    )

    assert report.localization == PixelMetrics(7, 1, 1)
    assert report.damage_confusion == (
        (0, 0, 0, 0, 0),
        (1, 1, 0, 0, 0),
        (0, 0, 1, 1, 0),
        (0, 0, 1, 1, 0),
        (0, 0, 0, 0, 1),
    )
    expected_damage_f1 = 4 / sum(1 / (f1 + 1e-6) for f1 in (2 / 3, 0.5, 0.5, 1))
    assert report.localization.f1 == pytest.approx(0.875)
    assert report.damage_f1 == pytest.approx(expected_damage_f1)
    assert report.score == pytest.approx(0.3 * 0.875 + 0.7 * expected_damage_f1)
    assert report.to_dict()["localization"]["iou"] == pytest.approx(7 / 9)


def test_rasterized_targets_are_pooled_before_scoring() -> None:
    major = Building("major", box(0, 0, 2, 2), 3, "major_damage")
    ignored = Building("ignored", box(2, 0, 3, 1), IGNORE, "un_classified")
    buildings = [major, ignored]
    localization = rasterize_localization(buildings, 1024, 1024).mask
    damage = rasterize_damage(buildings, 1024, 1024).mask
    prediction = damage.copy()
    prediction[prediction == IGNORE] = 1
    empty = np.zeros_like(localization)

    report = score_mask_pairs(
        [
            MaskPair(localization, prediction, localization, damage),
            MaskPair(empty, prediction, localization, damage),
        ]
    )

    assert report.image_count == 2
    assert report.localization == PixelMetrics(5, 0, 5)
    assert report.damage_by_class["major_damage"] == PixelMetrics(4, 0, 4)
    assert report.to_dict()["damage_f1_minor_damage"] == 0.0
    assert report.to_dict()["damage_by_class"]["minor_damage"]["f1"] is None
    assert report.to_dict()["damage_by_class"]["major_damage"]["support"] == 8


def test_predictions_cannot_use_the_internal_ignore_label() -> None:
    empty = np.zeros((1024, 1024), dtype=np.uint8)
    invalid = empty.copy()
    invalid[0, 0] = IGNORE

    with pytest.raises(ValueError, match="damage prediction: pixel values must be 0–4"):
        score_mask_pairs([MaskPair(empty, invalid, empty, empty)])


@pytest.mark.parametrize(
    "invalid",
    [np.zeros((8, 8), dtype=np.uint8), np.zeros((1024, 1024), dtype=np.int64)],
)
def test_mask_shape_and_dtype_are_checked(invalid: np.ndarray) -> None:
    empty = np.zeros((1024, 1024), dtype=np.uint8)

    with pytest.raises(ValueError, match="localization target: expected a 1024×1024 uint8 mask"):
        score_mask_pairs([MaskPair(empty, empty, invalid, empty)])
