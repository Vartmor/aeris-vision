import numpy as np
import pytest
from PIL import Image
from shapely.geometry import box

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from aeris.data.labels import IGNORE, Building
from aeris.data.raster import rasterize_damage, rasterize_localization
from aeris.eval.pipeline import _write_masks, evaluate_tile
from aeris.eval.xview2 import score_directories
from aeris.infer.pipeline import InferencePipeline, InferenceSettings


class _WhitePixels(torch.nn.Module):
    def forward(self, images):
        return images[:, :1]


class _MinorDamage(torch.nn.Module):
    def forward(self, images):
        return images.new_tensor((0, 5, 1, 0)).expand(len(images), -1)


def test_end_to_end_pngs_and_building_errors_on_known_geometry(tmp_path):
    torch.set_num_threads(2)
    pipeline = InferencePipeline(
        _WhitePixels(), _MinorDamage(), InferenceSettings(256, 0.5, 4, 64, 0.25, 2),
        torch.device("cpu"), amp=False,
    )
    pre = np.zeros((1024, 1024, 3), dtype=np.uint8)
    pre[10:14, 10:18] = 255  # Two separate reference buildings in one detection.
    pre[30:34, 40:44] = 255  # False positive.
    pre[50:54, 50:54] = 255
    pre[70:74, 70:74] = 255  # Unclassified reference still eligible for localization.
    pre[100, 100] = 255  # Removed before classification.
    buildings = [
        Building("a", box(10, 10, 14, 14), 3, "major_damage"),
        Building("b", box(14, 10, 18, 14), 4, "destroyed"),
        Building("c", box(20, 20, 24, 24), 3, "major_damage"),
        Building("d", box(50, 50, 54, 54), 4, "destroyed"),
        Building("e", box(70, 70, 74, 74), IGNORE, "un_classified"),
        Building("f", box(90.1, 90.1, 90.2, 90.2), 3, "major_damage"),
    ]

    predicted, oracle, report, unrasterizable = evaluate_tile(
        pipeline, Image.fromarray(pre), Image.new("RGB", (1024, 1024)), buildings,
    )

    assert predicted.filtered_components == 1
    assert len(predicted.detections) == 4
    assert predicted.damage[10, 10] == 2
    assert predicted.damage[100, 100] == 0
    assert predicted.localization[30, 40] == 1
    assert oracle[0][70, 70] == 1 and oracle[1][70, 70] == 0
    assert unrasterizable == 1
    assert (report.matches[0].reference_id, report.matches[0].iou) == ("r00003", 1.0)
    merged_match = next(match for match in report.matches if match.iou == 0.5)
    assert merged_match.reference_id == "r00000"
    assert report.localization_recall == 3 / 6
    assert report.unmatched_reference_ids == ("r00001", "r00002", "r00005")
    assert len(report.unmatched_candidate_ids) == len(report.merged_candidates) == 1
    assert report.severe_damage.in_merged_candidate == 2
    assert report.severe_damage.not_detected == 2
    assert report.severe_damage.minor_damage == 1
    assert report.severe_damage.total == 5

    loc_target = rasterize_localization(buildings, 1024, 1024).mask
    dmg_target = rasterize_damage(buildings, 1024, 1024).mask
    dmg_target[dmg_target == IGNORE] = 0
    prediction_dir, target_dir = tmp_path / "predictions", tmp_path / "targets"
    prediction_dir.mkdir()
    target_dir.mkdir()
    _write_masks(prediction_dir, "00000", "prediction", predicted.localization, predicted.damage)
    _write_masks(target_dir, "00000", "target", loc_target, dmg_target)
    score = score_directories(prediction_dir, target_dir)
    assert score.localization.f1 == pytest.approx(0.8)
    assert score.damage_f1 == pytest.approx(1e-6)
    assert score.damage_confusion[3] == (16, 0, 16, 0, 0)
    assert score.damage_confusion[4] == (0, 0, 32, 0, 0)


def test_empty_prediction_keeps_original_dimensions():
    torch.set_num_threads(2)
    pipeline = InferencePipeline(
        _WhitePixels(), _MinorDamage(), InferenceSettings(64, 0.5, 4, 64, 0.25, 2),
        torch.device("cpu"), amp=False,
    )
    image = Image.new("RGB", (78, 97))

    predicted = pipeline.predict(image, image)

    assert predicted.detections == ()
    assert predicted.damage.shape == (97, 78)
    assert not predicted.damage.any() and not predicted.localization.any()
