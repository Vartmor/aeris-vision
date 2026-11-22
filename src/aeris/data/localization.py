"""Aligned RGB and binary localization patches from audited xBD tiles."""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from aeris.data.manifest import Record
from aeris.data.targets import target_path

_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


@dataclass(frozen=True, slots=True)
class Patch:
    sample_id: str
    image: Path
    target: Path
    box: tuple[int, int, int, int]


def tile_patches(raw: Path, row: Record, size: int) -> list[Patch]:
    width, height = row["width"], row["height"]
    if width < size or height < size or width % size or height % size:
        raise ValueError(f"{row['sample_id']}: crop size needs to divide tile width and height")
    image = raw / row["pre_image_path"]
    mask = target_path(image)
    return [
        Patch(row["sample_id"], image, mask, (left, top, left + size, top + size))
        for top in range(0, height, size)
        for left in range(0, width, size)
    ]


def tiny_patches(raw: Path, rows: Sequence[Record], size: int, seed: int) -> list[Patch]:
    """Select stable training crops with 24 building and 8 empty targets."""
    rng = Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    buildings: list[Patch] = []
    empty: list[Patch] = []
    for row in shuffled:
        patches = tile_patches(raw, row, size)
        rng.shuffle(patches)
        with Image.open(patches[0].target, formats=["PNG"]) as mask:
            for patch in patches:
                coverage = np.asarray(mask.crop(patch.box)).mean()
                if 0.02 <= coverage <= 0.55 and len(buildings) < 24:
                    buildings.append(patch)
                    break
                if coverage == 0 and len(empty) < 8:
                    empty.append(patch)
                    break
        if len(buildings) == 24 and len(empty) == 8:
            return buildings + empty
    raise ValueError("Need 24 training tiles with suitable foreground and 8 with empty crops")


class LocalizationPatches(Dataset):
    """Load each aligned crop on demand; ImageNet normalization preserves geometry."""

    def __init__(self, patches: Sequence[Patch], cached_tiles: Mapping[Path, Path] | None = None) -> None:
        self.patches = patches
        self.cached_tiles = cached_tiles or {}

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch = self.patches[index]
        cached = self.cached_tiles.get(patch.image)
        if cached is None:
            rgb, target = _read_source(patch, patch.box)
        else:
            left, top, right, bottom = patch.box
            tile = np.load(cached, mmap_mode="r", allow_pickle=False)
            crop = tile[top:bottom, left:right]
            if crop.shape != (bottom - top, right - left, 4) or crop.dtype != np.uint8:
                raise ValueError(f"{cached}: invalid cached patch for {patch.sample_id}")
            rgb, target = crop[:, :, :3].copy(), crop[:, :, 3].copy()
        image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255)
        image_tensor.sub_(_MEAN).div_(_STD)
        return image_tensor, torch.from_numpy(target).unsqueeze(0).float()


def _read_source(patch: Patch, box: tuple[int, int, int, int] | None) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(patch.image, formats=["PNG"]) as image:
        if image.mode != "RGB":
            raise ValueError(f"{patch.image}: expected RGB imagery")
        size = image.size
        if box is not None and (box[2] > size[0] or box[3] > size[1]):
            raise ValueError(f"{patch.image}: patch exceeds image bounds")
        rgb = np.array(image if box is None else image.crop(box))
    with Image.open(patch.target, formats=["PNG"]) as mask:
        if mask.size != size:
            raise ValueError(f"{patch.target}: target size does not match the image")
        target = np.array(mask if box is None else mask.crop(box))
    if target.ndim != 2 or not np.isin(target, (0, 1)).all():
        raise ValueError(f"{patch.target}: expected a binary 0/1 localization mask")
    return rgb, target.astype(np.uint8, copy=False)


def cache_localization_tiles(
    patches: Sequence[Patch], directory: Path, manifest_sha256: str,
) -> dict[Path, Path]:
    """Decode each source tile once into an atomic, content-addressed array."""
    root = directory / "rgb-mask-v1" / manifest_sha256
    root.mkdir(parents=True, exist_ok=True)
    cached: dict[Path, Path] = {}
    for patch in patches:
        if patch.image in cached:
            continue
        hashes = []
        for path in (patch.image, patch.target):
            with path.open("rb") as stream:
                hashes.append(hashlib.file_digest(stream, "sha256").digest())
        key = hashlib.sha256(b"".join(hashes)).hexdigest()
        path = root / f"{key}.npy"
        if not path.is_file():
            rgb, mask = _read_source(patch, None)
            tile = np.dstack((rgb, mask))
            with NamedTemporaryFile(dir=root, prefix=".tile-", delete=False) as sink:
                temporary = Path(sink.name)
                try:
                    np.save(sink, tile, allow_pickle=False)
                    sink.close()
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
        cached[patch.image] = path
    return cached
