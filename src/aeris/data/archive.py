"""Prepare a locally obtained xBD challenge training archive."""

import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

# Published for train_images_labels_targets.tar.gz by TorchGeo v0.10.1.
REFERENCE_TRAIN_SHA256 = "a5941b7a3e523eafc4aeaa740a1c83f1af6a18c894e7e8c62dd830a76921ecd4"
ARCHIVE_RECEIPT = ".aeris-import.json"


class ArchiveError(ValueError):
    """The archive cannot be safely prepared as an xBD package."""


def prepare_archive(
    archive: Path, raw_dir: Path, *, source_package: str, expected_sha256: str | None = None
) -> dict[str, object]:
    """Hash and stage a local tar archive before making it visible to the audit."""
    if expected_sha256 is not None:
        expected_sha256 = expected_sha256.casefold()
        if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
            raise ArchiveError("Expected SHA-256 must contain 64 hexadecimal characters")
    if raw_dir.is_symlink() or raw_dir.exists():
        raise ArchiveError(f"Raw directory already exists: {raw_dir}; use data audit for extracted data")
    raw_dir.parent.mkdir(parents=True, exist_ok=True)

    with archive.open("rb") as source, TemporaryDirectory(
        dir=raw_dir.parent, prefix=".xbd-import-"
    ) as temporary:
        sha256 = hashlib.file_digest(source, "sha256").hexdigest()
        if expected_sha256 is not None and sha256 != expected_sha256:
            raise ArchiveError(f"Archive SHA-256 mismatch: {archive.name}")

        staged = Path(temporary) / "package"
        staged.mkdir()
        source.seek(0)
        with tarfile.open(fileobj=source, mode="r|*") as package:
            _unpack(package, staged)
        _validate_layout(staged, source_package)

        receipt = {
            "source_package": source_package,
            "archive_name": archive.name,
            "archive_sha256": sha256,
            "matches_torchgeo_challenge_train": sha256 == REFERENCE_TRAIN_SHA256,
        }
        with (staged / ARCHIVE_RECEIPT).open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")

        # Same-filesystem rename publishes a fully extracted directory.
        staged.rename(raw_dir)
    return receipt


def _unpack(package: tarfile.TarFile, staged: Path) -> None:
    for member in package:
        relative = PurePosixPath(member.name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in member.name:
            raise ArchiveError(f"Unexpected archive member path: {member.name!r}")
        destination = staged.joinpath(*relative.parts)

        if member.isdir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        if not member.isreg() or not relative.parts:
            raise ArchiveError(f"Expected a regular file or directory: {member.name!r}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        data = package.extractfile(member)
        if data is None:
            raise ArchiveError(f"Could not read archive member: {member.name!r}")
        with data, destination.open("xb") as target:
            shutil.copyfileobj(data, target, length=1024 * 1024)


def _validate_layout(root: Path, source_package: str) -> None:
    found_pair = False
    for labels in root.rglob("labels"):
        images = labels.parent / "images"
        for pre in labels.glob("*_pre_disaster.json"):
            layout = labels.parent.relative_to(root)
            if source_package == "challenge_train" and {"test", "hold", "holdout"} & set(layout.parts):
                raise ArchiveError(f"Challenge test or holdout labels in training archive: {layout}")
            stem = pre.name.removesuffix("_pre_disaster.json")
            if (
                (labels / f"{stem}_post_disaster.json").is_file()
                and (images / f"{stem}_pre_disaster.png").is_file()
                and (images / f"{stem}_post_disaster.png").is_file()
            ):
                found_pair = True
                break
    if not found_pair:
        raise ArchiveError("Expected images/ and labels/ containing an xBD pre/post pair")
