import json
import subprocess
import sys
from pathlib import Path


def run_audit(config: Path, cwd: Path, *options: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "aeris", "data", "audit", "--config", str(config), *options],
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
    )


def write_config(path: Path, raw_dir: str) -> Path:
    path.write_text(
        "dataset:\n"
        "  name: xbd\n"
        "  source_package: challenge_train\n"
        f"  raw_dir: {raw_dir}\n"
        "  manifest_path: processed/manifest.jsonl\n"
        "  audit_path: processed/audit.json\n",
        encoding="utf-8",
    )
    return path


def test_data_audit_writes_manifest_and_report_relative_to_cwd(synthetic_xbd, tmp_path) -> None:
    rectangle = synthetic_xbd.rectangle
    synthetic_xbd.write_pair(
        "hurricane-a_00000001",
        [rectangle("a", 0, 0, 2, 2, "destroyed"), rectangle("b", 4, 4, 6, 6, "minor-damage")],
    )
    config = write_config(tmp_path / "data.yaml", synthetic_xbd.root.name)

    result = run_audit(config, tmp_path, "--check-targets")

    assert result.returncode == 0, result.stderr
    assert "1 örnek, 0 uyarılı, 0 geçersiz" in result.stdout
    assert "hurricane-a" in result.stdout
    records = (tmp_path / "processed" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(records[0])["sample_id"] == "hurricane-a_00000001"
    summary = json.loads((tmp_path / "processed" / "audit.json").read_text(encoding="utf-8"))
    assert summary["events"]["hurricane-a"]["damage_label_counts"]["destroyed"] == 1
    assert summary["target_check"]["pre"]["not_compared"] == {"missing": 1}


def test_data_audit_fails_without_data(tmp_path) -> None:
    config = write_config(tmp_path / "data.yaml", "missing")

    result = run_audit(config, tmp_path)

    assert result.returncode == 2
    assert "Veri dizini bulunamadı" in result.stderr


def test_data_audit_fails_when_no_labels_are_found(tmp_path) -> None:
    (tmp_path / "empty").mkdir()
    config = write_config(tmp_path / "data.yaml", "empty")

    result = run_audit(config, tmp_path)

    assert result.returncode == 2
    assert "Etiket dosyası bulunamadı" in result.stderr
    assert not (tmp_path / "processed").exists()


def test_data_audit_rejects_incomplete_dataset_config(tmp_path) -> None:
    config = tmp_path / "data.yaml"
    config.write_text("dataset:\n  raw_dir: data\n", encoding="utf-8")

    result = run_audit(config, tmp_path)

    assert result.returncode == 2
    assert "dataset.manifest_path" in result.stderr
