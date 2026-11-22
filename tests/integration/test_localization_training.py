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
from aeris.data.localization import LocalizationPatches, Patch, cache_localization_tiles
from aeris.data.manifest import build_manifest, write_manifest
from aeris.models.localization import LocalizationUNet
from aeris.splits.storage import freeze_split, plan_from_manifest
from aeris.train.localization import evaluate_localization


def test_localization_trains_from_frozen_split_and_reloads_checkpoint(synthetic_xbd, tmp_path: Path):
    buildings = [
        synthetic_xbd.rectangle(subtype, 26 * index, 4, 26 * index + 8, 16, subtype)
        for index, subtype in enumerate(("no-damage", "minor-damage", "major-damage", "destroyed"))
    ]
    for index in range(5):
        event = "train-wind" if index < 4 else "held-wind"
        sample = f"{event}_{index:08d}"
        package = synthetic_xbd.write_pair(
            sample, buildings, size=(128, 64), lng_lat_origin=(index * 0.01, 40.0),
            metadata={"pre": {"catalog_id": f"pre-{index}"}, "post": {"catalog_id": f"post-{index}"}},
        )
        for phase, offset in (("pre", 0), ("post", 1)):
            image = Image.new("RGB", (128, 64), (index + 1, offset, 0))
            image.save(package / "images" / f"{sample}_{phase}_disaster.png")
        if index < 2:
            mask = np.zeros((64, 128), dtype=np.uint8)
            for building in range(4):
                mask[4:16, 26 * building:26 * building + 8] = 1
            synthetic_xbd.write_target(sample, "pre", mask)

    manifest = tmp_path / "manifest.jsonl"
    write_manifest(build_manifest(synthetic_xbd.root, "challenge_train"), manifest, tmp_path / "audit.json")
    split_path = tmp_path / "splits.json"
    fractions = {"train": 0.7, "validation": 0.1, "calibration": 0.1, "test": 0.1}
    groups = {"validation": ["train-wind_00000001"], "calibration": ["train-wind_00000002"],
              "test": ["train-wind_00000003"]}
    plan = plan_from_manifest(manifest, fractions, ood_event="held-wind", id_evaluation_groups=groups)
    freeze_split(split_path, plan)
    data_config = tmp_path / "data.yaml"
    data_config.write_text(yaml.safe_dump({
        "dataset": {"raw_dir": str(synthetic_xbd.root), "manifest_path": str(manifest)},
        "split": {"output_path": str(split_path), "id_fractions": fractions,
                  "ood_event": "held-wind", "id_evaluation_groups": groups},
    }))
    model_config = tmp_path / "localization.yaml"
    model_config.write_text(yaml.safe_dump({
        "model": {"architecture": "unet", "encoder": "resnet18", "input": "pre_rgb"},
        "training": {"crop_size": 64, "batch_size": 2, "loss": "bce_dice", "optimizer": "adamw",
                     "learning_rate": 0.0001, "amp": "off", "checkpoint_metric": "validation_localization_f1"},
    }))
    run = tmp_path / "run"
    args = ["train", "--task", "localization", "--config", str(model_config),
            "--data-config", str(data_config), "--run-dir", str(run), "--device", "cpu", "--workers", "0",
            "--train-tiles", "1", "--thresholds", "0.4", "0.5", "0.6", "--validation-masks"]

    assert main(args) == 0
    report = json.loads((run / "metrics.json").read_text())
    assert report["evaluation_partition"] == "validation"
    assert report["train_patches"] == report["evaluation_patches"] == 2
    assert report["train_tiles"] == plan["partitions"]["train"]
    assert report["split_sha256"] == plan["split_sha256"]
    checkpoint_file = run / "best.pt"
    assert hashlib.sha256(checkpoint_file.read_bytes()).hexdigest() == report["best_checkpoint"]["sha256"]
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    assert checkpoint["threshold"] == report["best_checkpoint"]["metrics"]["threshold"]
    exported = report["validation_masks"]
    assert exported["count"] == 1
    assert exported["tiles"][0]["sample_id"] == plan["partitions"]["validation"][0]
    mask_path = run / exported["directory"] / exported["tiles"][0]["filename"]
    with Image.open(mask_path) as mask:
        assert mask.size == (128, 64)
        assert set(np.unique(np.asarray(mask))) <= {0, 1}
    model = LocalizationUNet().eval()
    model.load_state_dict(checkpoint["model"])
    with torch.inference_mode():
        logits = model(torch.zeros((1, 3, 64, 64)))
    assert logits.shape == (1, 1, 64, 64)
    assert torch.isfinite(logits).all()
    assert main(args) == 2


def test_validation_threshold_uses_pooled_pixel_counts():
    probabilities = torch.tensor([[[[0.42, 0.32]]], [[[0.43, 0.48]]]])
    targets = torch.tensor([[[[1.0, 0.0]]], [[[1.0, 0.0]]]])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(probabilities.logit(), targets), batch_size=1,
    )

    result = evaluate_localization(
        torch.nn.Identity(), loader, torch.device("cpu"), False, (0.3, 0.4, 0.5),
    )

    assert result["threshold"] == 0.4
    assert result["f1"] == 0.8
    assert result["iou"] == 2 / 3
    assert [row["f1"] for row in result["threshold_curve"]] == [2 / 3, 0.8, 0.0]


def test_validation_threshold_prefers_earlier_cutoff_at_equal_distance():
    data = torch.utils.data.TensorDataset(torch.ones((1, 1, 1, 1)), torch.ones((1, 1, 1, 1)))
    result = evaluate_localization(
        torch.nn.Identity(), torch.utils.data.DataLoader(data), torch.device("cpu"), False, (0.3, 0.7),
    )

    assert result["f1"] == 1.0
    assert result["threshold"] == 0.3


def test_cached_crop_matches_aligned_pngs_and_tracks_content_changes(tmp_path: Path):
    rgb = np.zeros((90, 90, 3), dtype=np.uint8)
    rgb[21, 17] = (150, 170, 190)
    mask = np.zeros((90, 90), dtype=np.uint8)
    mask[21, 17] = 1
    image_path, target_path = tmp_path / "image.png", tmp_path / "target.png"
    Image.fromarray(rgb).save(image_path)
    Image.fromarray(mask).save(target_path)
    patch = Patch("tile", image_path, target_path, (10, 20, 74, 84))
    directory = tmp_path / "cache"

    cache = cache_localization_tiles([patch], directory, "audited-manifest")
    direct = LocalizationPatches([patch])[0]
    cached = LocalizationPatches([patch], cache)[0]
    for expected, actual in zip(direct, cached):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert cached[1][0, 1, 7] == 1 and cached[1].sum() == 1

    Image.fromarray(np.zeros_like(mask)).save(target_path)
    fresh_cache = cache_localization_tiles([patch], directory, "audited-manifest")
    assert fresh_cache[image_path] != cache[image_path]
    assert LocalizationPatches([patch], fresh_cache)[0][1].sum() == 0
