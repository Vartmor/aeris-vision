import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def mask_dirs(tmp_path: Path) -> tuple[Path, Path]:
    predictions, targets = tmp_path / "predictions", tmp_path / "targets"
    damage = np.zeros((1024, 1024), dtype=np.uint8)
    damage[0, :4] = [1, 2, 3, 4]
    localization = (damage > 0).astype(np.uint8)

    for role, folder in (("prediction", predictions), ("target", targets)):
        folder.mkdir()
        Image.fromarray(damage).save(folder / f"test_damage_00001_{role}.png")
        Image.fromarray(localization).save(folder / f"test_localization_00001_{role}.png")
    return predictions, targets


def run_evaluate(predictions: Path, targets: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, "-m", "aeris", "evaluate",
            "--predictions", str(predictions), "--targets", str(targets), *args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_evaluate_writes_reference_scores_for_official_png_layout(
    mask_dirs: tuple[Path, Path], tmp_path: Path
) -> None:
    predictions, targets = mask_dirs
    output = tmp_path / "scores.json"

    result = run_evaluate(predictions, targets, "--output", str(output))

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["image_count"] == 1
    assert report["localization_f1"] == 1.0
    assert report["damage_f1"] == pytest.approx(1.0 + 1e-6)
    assert report["score"] == pytest.approx(1.0 + 0.7e-6)


def test_evaluate_reports_missing_prediction(mask_dirs: tuple[Path, Path]) -> None:
    predictions, targets = mask_dirs
    (predictions / "test_damage_00001_prediction.png").unlink()

    result = run_evaluate(predictions, targets)

    assert result.returncode == 2
    assert "test_damage_00001_prediction.png" in result.stderr


def test_evaluate_rejects_ignore_in_exported_target_png(mask_dirs: tuple[Path, Path]) -> None:
    predictions, targets = mask_dirs
    path = targets / "test_damage_00001_target.png"
    with Image.open(path) as image:
        damage = np.array(image)
    damage[0, 4] = 255
    Image.fromarray(damage).save(path)

    result = run_evaluate(predictions, targets)

    assert result.returncode == 2
    assert "damage target: pixel values must be 0–4" in result.stderr
