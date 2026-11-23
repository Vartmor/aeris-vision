import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from aeris.cli import main
from aeris.data.damage import DamageCrop, DamageCrops, cache_damage_crops, damage_crop_box
from aeris.data.manifest import build_manifest, write_manifest
from aeris.eval.damage import damage_metrics
from aeris.models.damage import build_damage_model
from aeris.splits.storage import freeze_split, plan_from_manifest


def test_damage_crop_clips_expands_and_matches_cache(tmp_path):
    image = tmp_path / "post.png"
    Image.fromarray(np.random.default_rng(4).integers(0, 256, (90, 110, 3), dtype=np.uint8)).save(image)
    assert damage_crop_box((-3, 10, 8, 18), (110, 90)) == (0, 8, 10, 20)
    assert damage_crop_box((19, 21, 20, 23), (110, 90)) is None
    assert damage_crop_box((111, 21, 120, 28), (110, 90)) is None
    assert damage_crop_box((2, 2, float("inf"), 9), (110, 90)) is None
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    crop = DamageCrop("sample", image, digest, 0, "building", (0, 8, 10, 20), 3)
    cached_path = cache_damage_crops([crop], 64, tmp_path / "cache", "manifest", "train")

    direct_image, direct_class = DamageCrops([crop], 64)[0]
    cached_image, cached_class = DamageCrops([crop], 64, cached_path)[0]

    torch.testing.assert_close(direct_image, cached_image, rtol=0, atol=0)
    assert direct_image.shape == (3, 64, 64)
    assert direct_class == cached_class == 2
    Image.new("RGB", (110, 90)).save(image)
    with pytest.raises(ValueError, match="source changed"):
        cache_damage_crops([crop], 64, tmp_path / "cache", "manifest", "train")


def test_damage_cli_trains_only_frozen_train_and_records_validation_logits(synthetic_xbd, tmp_path: Path):
    buildings = [
        synthetic_xbd.rectangle(name, 12 * i, 8, 12 * i + 8, 18, name)
        for i, name in enumerate(("no-damage", "minor-damage", "major-damage", "destroyed", "un-classified"))
    ]
    for index in range(5):
        event = "train-wind" if index < 4 else "held-wind"
        sample = f"{event}_{index:08d}"
        package = synthetic_xbd.write_pair(
            sample, buildings, size=(64, 64), lng_lat_origin=(index * 0.01, 40.0),
            metadata={"pre": {"catalog_id": f"pre-{index}"}, "post": {"catalog_id": f"post-{index}"}},
        )
        for phase, offset in (("pre", 0), ("post", 1)):
            Image.new("RGB", (64, 64), (index + 1, offset, 0)).save(
                package / "images" / f"{sample}_{phase}_disaster.png",
            )

    manifest, split = tmp_path / "manifest.jsonl", tmp_path / "splits.json"
    write_manifest(build_manifest(synthetic_xbd.root, "challenge_train"), manifest, tmp_path / "audit.json")
    fractions = {"train": 0.7, "validation": 0.1, "calibration": 0.1, "test": 0.1}
    groups = {"validation": ["train-wind_00000001"], "calibration": ["train-wind_00000002"],
              "test": ["train-wind_00000003"]}
    plan = plan_from_manifest(manifest, fractions, ood_event="held-wind", id_evaluation_groups=groups)
    freeze_split(split, plan)
    data = tmp_path / "data.yaml"
    data.write_text(yaml.safe_dump({
        "dataset": {"raw_dir": str(synthetic_xbd.root), "manifest_path": str(manifest)},
        "split": {"output_path": str(split), "id_fractions": fractions, "ood_event": "held-wind",
                  "id_evaluation_groups": groups},
    }))
    for partition in ("calibration", "test", "ood_test"):
        sample = plan["partitions"][partition][0]
        (synthetic_xbd.root / "train" / "labels" / f"{sample}_post_disaster.json").unlink()

    config = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/damage_post.yaml").read_text())
    config["model"]["encoder_weights"] = None
    config["training"].update(crop_size=64, batch_size=4)
    settings = tmp_path / "damage.yaml"
    settings.write_text(yaml.safe_dump(config))
    run = tmp_path / "run"
    assert main(["train", "--task", "damage", "--config", str(settings), "--data-config", str(data),
                 "--run-dir", str(run), "--device", "cpu", "--workers", "0", "--max-crops-per-class", "1",
                 "--cache-dir", str(tmp_path / "cache")]) == 0

    report = json.loads((run / "metrics.json").read_text())
    assert report["train_crops"] == report["evaluation_crops"] == 4
    assert report["class_counts"] == dict.fromkeys(report["classes"], 1)
    assert report["class_weights"] == dict.fromkeys(report["classes"], 1.0)
    assert report["train_filter_counts"] == {"unclassified": 1}
    assert report["subset"]["max_crops_per_class"] == 1
    with np.load(run / "validation_logits.npz", allow_pickle=False) as data:
        assert set(data["sample_id"]) == set(plan["partitions"]["validation"])
        assert data["label_id"].tolist() == [1, 2, 3, 4]
        assert data["logits"].shape == (4, 4)
        assert damage_metrics(data["label_id"] - 1, data["logits"]) == report["best_checkpoint"]["metrics"]
    saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    model = build_damage_model().eval()
    model.load_state_dict(saved["model"])
    assert saved["split_sha256"] == plan["split_sha256"]
    with torch.inference_mode():
        assert model(torch.zeros((1, 3, 64, 64))).shape == (1, 4)
