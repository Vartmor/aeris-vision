import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "aeris", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_split_cli_freezes_and_verifies_an_audited_xbd_manifest(synthetic_xbd, tmp_path: Path) -> None:
    buildings = [
        synthetic_xbd.rectangle(subtype, index * 3, 0, index * 3 + 2, 2, subtype)
        for index, subtype in enumerate(("no-damage", "minor-damage", "major-damage", "destroyed"))
    ]
    for index in range(9):
        event = "held-wind" if index == 8 else "training-wind"
        sample_id = f"{event}_{index:08d}"
        package = synthetic_xbd.write_pair(
            sample_id,
            buildings,
            metadata={
                "pre": {"catalog_id": f"pre-{index}"},
                "post": {"catalog_id": f"post-{index}"},
            },
            lng_lat_origin=(10.0 + index * 0.01, 20.0),
        )
        for phase_id, phase in enumerate(("pre", "post")):
            path = package / "images" / f"{sample_id}_{phase}_disaster.png"
            Image.new("RGB", (16, 16), (index + 1, phase_id, 0)).save(path)
            target = np.zeros((16, 16), dtype=np.uint8)
            for building in range(4):
                target[:2, building * 3 : building * 3 + 2] = 1 if phase == "pre" else building + 1
            synthetic_xbd.write_target(sample_id, phase, target)

    archive = tmp_path / "challenge-train.tar.gz"
    with tarfile.open(archive, "w:gz") as package:
        package.add(synthetic_xbd.root / "train", arcname="train")
    expected_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    raw_dir = tmp_path / "raw" / "xbd"
    manifest = tmp_path / "manifest.jsonl"
    audit = tmp_path / "audit.json"
    output = tmp_path / "frozen" / "splits.json"
    config_path = tmp_path / "data.yaml"
    config_path.write_text(yaml.safe_dump({
        "dataset": {
            "raw_dir": str(raw_dir),
            "manifest_path": str(manifest),
            "audit_path": str(audit),
            "source_package": "challenge_train",
        },
        "split": {
            "output_path": str(output),
            "ood_event": "held-wind",
            "id_evaluation_groups": {
                "validation": ["training-wind_00000005"],
                "calibration": ["training-wind_00000006"],
                "test": ["training-wind_00000007"],
            },
            "id_fractions": {"train": 0.7, "validation": 0.1, "calibration": 0.1, "test": 0.1},
        },
    }))
    prepared = _run(
        "data", "prepare", "--config", str(config_path),
        "--archive", str(archive), "--sha256", expected_sha256,
    )
    assert prepared.returncode == 0, prepared.stderr
    receipt = json.loads((raw_dir / ".aeris-import.json").read_text())
    assert receipt["archive_sha256"] == expected_sha256

    audited = _run("data", "audit", "--config", str(config_path), "--check-targets")
    assert audited.returncode == 0, audited.stderr
    audit_report = json.loads(audit.read_text())
    assert audit_report["target_check"]["pre"]["samples_exact"] == 9
    assert audit_report["target_check"]["post"]["samples_exact"] == 9
    original_manifest = manifest.read_bytes()

    verified = _run("data", "verify", "--config", str(config_path))
    assert verified.returncode == 0, verified.stderr
    target = raw_dir / "train" / "targets" / "held-wind_00000008_post_disaster_target.png"
    original_target = target.read_bytes()
    with Image.open(target) as image:
        modified = image.copy()
    modified.putpixel((0, 0), 4)
    modified.save(target)
    stale = _run("data", "verify", "--config", str(config_path))
    assert stale.returncode == 2
    assert "kaydedilen manifestle uyuşmuyor" in stale.stderr
    assert manifest.read_bytes() == original_manifest
    target.write_bytes(original_target)

    made = _run("split", "make", "--config", str(config_path))
    assert made.returncode == 0, made.stderr
    frozen = json.loads(output.read_text())
    assert frozen["manifest_sha256"] == audit_report["manifest_sha256"]
    assert frozen["partitions"]["ood_test"] == ["held-wind_00000008"]
    assert frozen["partitions"]["test"] == ["training-wind_00000007"]
    assert frozen["same_event_type_in_train"] is True
    assert len(frozen["split_sha256"]) == 64

    checked = _run("split", "verify", "--config", str(config_path))
    assert checked.returncode == 0, checked.stderr

    frozen["partitions"]["train"].append("fake-id")
    output.write_text(json.dumps(frozen))
    rejected = _run("split", "verify", "--config", str(config_path))
    assert rejected.returncode == 2
    assert "frozen split differs" in rejected.stderr
