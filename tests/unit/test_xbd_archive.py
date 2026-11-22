import io
import tarfile
from pathlib import Path

import pytest

from aeris.data.archive import ArchiveError, prepare_archive


def _archive_with_member(archive: Path, member_name: str) -> None:
    data = b"test entry"
    with tarfile.open(archive, "w:gz") as package:
        member = tarfile.TarInfo(member_name)
        member.size = len(data)
        package.addfile(member, io.BytesIO(data))


def test_archive_rejects_checksum_mismatch_before_extraction(tmp_path: Path) -> None:
    archive = tmp_path / "training.tar.gz"
    _archive_with_member(archive, "train/labels/example.json")
    raw_dir = tmp_path / "raw" / "xbd"

    with pytest.raises(ArchiveError, match="SHA-256 mismatch"):
        prepare_archive(archive, raw_dir, source_package="challenge_train", expected_sha256="0" * 64)

    assert not raw_dir.exists()
    assert not list(raw_dir.parent.iterdir())


def test_incomplete_layout_is_not_published(tmp_path: Path) -> None:
    archive = tmp_path / "training.tar.gz"
    _archive_with_member(archive, "train/labels/area_0001_pre_disaster.json")
    raw_dir = tmp_path / "raw" / "xbd"

    with pytest.raises(ArchiveError, match="containing an xBD pre/post pair"):
        prepare_archive(archive, raw_dir, source_package="challenge_train")

    assert not raw_dir.exists()
    assert not list(raw_dir.parent.iterdir())


def test_existing_raw_data_is_preserved(tmp_path: Path) -> None:
    raw_dir = tmp_path / "xbd"
    raw_dir.mkdir()
    marker = raw_dir / "existing.txt"
    marker.write_text("existing dataset")

    with pytest.raises(ArchiveError, match="Raw directory already exists"):
        prepare_archive(tmp_path / "training.tar.gz", raw_dir, source_package="challenge_train")

    assert marker.read_text() == "existing dataset"



def test_test_package_does_not_enter_training_root(tmp_path: Path) -> None:
    archive = tmp_path / "different-split.tar.gz"
    _archive_with_member(archive, "test/labels/area_0001_pre_disaster.json")
    raw_dir = tmp_path / "raw" / "xbd"

    with pytest.raises(ArchiveError, match="test or holdout labels"):
        prepare_archive(archive, raw_dir, source_package="challenge_train")

    assert not raw_dir.exists()
