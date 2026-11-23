import json
import math

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from aeris.cli import main
from aeris.infer import output
from aeris.infer.pipeline import InferencePipeline, InferenceSettings


class _Localization(torch.nn.Module):
    def forward(self, images):
        return images[:, :1]


class _Damage(torch.nn.Module):
    def forward(self, images):
        return images.new_tensor((-5, 1, 2, 4)).expand(len(images), -1)


@pytest.fixture
def cpu_pipeline(monkeypatch):
    pipeline = InferencePipeline(
        _Localization(), _Damage(), InferenceSettings(64, 0.5, 4, 64, 0.25, 2),
        torch.device("cpu"), amp=False,
    )
    monkeypatch.setattr(output, "load_pipeline", lambda *args: (pipeline, {"device": "cpu", "amp": False}))


@pytest.mark.parametrize("has_building", [True, False])
def test_infer_cli_exports_prediction_and_aligned_masks(tmp_path, cpu_pipeline, has_building):
    image = np.zeros((72, 81, 3), dtype=np.uint8)
    if has_building:
        image[14:19, 63:68] = 255  # One footprint across two localization patches.
    pre, post = tmp_path / "example_pre_disaster.png", tmp_path / "post.png"
    Image.fromarray(image).save(pre)
    Image.new("RGB", (81, 72)).save(post)
    destination = tmp_path / "out"
    args = ["infer", "--pre", str(pre), "--post", str(post), "--output", str(destination)]

    assert main(args) == 0

    json_path = destination / "prediction.json"
    content = json_path.read_bytes()
    result = json.loads(content)
    assert result["schema_version"] == 1
    assert result["sample_id"] == "example"
    assert result["coordinate_system"] == "image_pixels"
    assert result["image_size"] == {"width": 81, "height": 72}
    images = {}
    for key, name in result["masks"].items():
        with Image.open(destination / name) as mask:
            images[key] = np.asarray(mask)
            assert mask.size == (81, 72)
    assert np.array_equal(images["localization"], images["damage"] > 0)
    if has_building:
        (row,) = result["buildings"]
        assert row["bbox_xyxy"] == [63, 14, 68, 19]
        assert row["area_pixels"] == np.count_nonzero(images["damage"]) == 25
        assert row["predicted_class"] == "destroyed"
        assert row["damage_class_id"] == images["damage"][14, 63] == 4
        assert math.isclose(sum(row["class_probabilities"].values()), 1.0)
        assert row["confidence"] == row["class_probabilities"]["destroyed"]
        assert row["manual_review"] is None
    else:
        assert result["buildings"] == []
        assert not images["damage"].any()
    assert main(args) == 2
    assert json_path.read_bytes() == content


def test_invalid_pair_leaves_no_partial_export(tmp_path, capsys, cpu_pipeline):
    pre, post = tmp_path / "pre.png", tmp_path / "post.png"
    Image.new("RGB", (65, 64)).save(pre)
    Image.new("RGB", (64, 64)).save(post)
    destination = tmp_path / "out"

    assert main(["infer", "--pre", str(pre), "--post", str(post), "--output", str(destination)]) == 2
    assert "same-size RGB" in capsys.readouterr().err
    assert not destination.exists()
