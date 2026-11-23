"""Post-disaster building crops, shared with candidate-based inference."""

import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from aeris.data.images import normalize_rgb
from aeris.data.labels import DAMAGE_CLASS_NAMES, parse_label
from aeris.data.manifest import Record


@dataclass(frozen=True, slots=True)
class DamageCrop:
    sample_id: str
    image: Path
    sha256: str | None
    building_index: int
    uid: str | None
    box: tuple[int, int, int, int]
    label: int  # xView2 damage ID 1–4.


def damage_crop_box(
    bounds: tuple[float, float, float, float], image_size: tuple[int, int],
    *, padding: float = 0.25, minimum_side: int = 2,
) -> tuple[int, int, int, int] | None:
    """Clip the visible footprint, then add context within the image."""
    if not all(math.isfinite(v) for v in bounds):
        return None
    x0, y0, x1, y1 = bounds
    width, height = image_size
    left, top = max(0, x0), max(0, y0)
    right, bottom = min(width, x1), min(height, y1)
    visible_width, visible_height = right - left, bottom - top
    if min(visible_width, visible_height) < minimum_side:
        return None
    xpad, ypad = visible_width * padding, visible_height * padding
    return (
        max(0, math.floor(left - xpad)), max(0, math.floor(top - ypad)),
        min(width, math.ceil(right + xpad)), min(height, math.ceil(bottom + ypad)),
    )


def read_damage_crops(
    raw: Path, records: Sequence[Record], *, padding: float, minimum_side: int,
) -> tuple[list[DamageCrop], Counter]:
    crops = []
    skipped = Counter()
    for record in records:
        size = record["width"], record["height"]
        label_path = raw / record["post_label_path"]
        hashes = record.get("sha256") or {}
        if label_digest := hashes.get("post_label"):
            with label_path.open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() != label_digest:
                    raise ValueError(f"{label_path}: label changed since the frozen manifest")
        labels = parse_label(label_path, "post", size)
        image = raw / record["post_image_path"]
        digest = hashes.get("post_image")
        for index, building in enumerate(labels.buildings):
            if building.damage_class not in DAMAGE_CLASS_NAMES:
                skipped["unclassified"] += 1
                continue
            if not building.polygon.is_valid:
                skipped["invalid_polygon"] += 1
                continue
            box = damage_crop_box(building.polygon.bounds, size, padding=padding, minimum_side=minimum_side)
            if box is None:
                skipped["too_small"] += 1
                continue
            crops.append(DamageCrop(record["sample_id"], image, digest, index, building.uid, box,
                                    building.damage_class))
    return crops, skipped


def damage_crop_rgb(image: Image.Image, box: tuple[int, int, int, int], size: int) -> np.ndarray:
    left, top, right, bottom = box
    if image.mode != "RGB" or not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
        raise ValueError("Damage crop needs RGB imagery and in-bounds coordinates")
    scale = size / max(right - left, bottom - top)
    width = max(1, round((right - left) * scale))
    height = max(1, round((bottom - top) * scale))
    patch = image.crop(box).resize((width, height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (124, 116, 104))
    canvas.paste(patch, ((size - width) // 2, (size - height) // 2))
    return np.array(canvas)


class DamageCrops(Dataset):
    def __init__(self, crops: Sequence[DamageCrop], size: int, cache: Path | None = None) -> None:
        self.crops = crops
        self.size = size
        self.cache = cache
        self._pixels = None

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, index: int):
        crop = self.crops[index]
        if self.cache is None:
            with Image.open(crop.image, formats=["PNG"]) as image:
                pixels = damage_crop_rgb(image, crop.box, self.size)
        else:
            if self._pixels is None:
                self._pixels = np.load(self.cache, mmap_mode="r", allow_pickle=False)
            pixels = self._pixels[index].copy()
        return normalize_rgb(pixels), crop.label - 1


def cache_damage_crops(
    crops: Sequence[DamageCrop], size: int, root: Path, manifest_sha: str, partition: str,
) -> Path:
    """Store the ordered crop tensor using exact source digests and output geometry."""
    grouped: dict[Path, list[tuple[int, DamageCrop]]] = {}
    for index, crop in enumerate(crops):
        grouped.setdefault(crop.image, []).append((index, crop))
    fingerprint = hashlib.sha256(str(size).encode())
    for image, items in grouped.items():
        with image.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if items[0][1].sha256 is not None and items[0][1].sha256 != digest:
            raise ValueError(f"{image}: source changed since the frozen manifest")
        fingerprint.update(digest.encode())
        fingerprint.update(json.dumps([(index, crop.box) for index, crop in items]).encode())
    directory = root / "rgb-crops-v1" / manifest_sha / partition
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{fingerprint.hexdigest()}.npy"
    shape = (len(crops), size, size, 3)
    if destination.is_file():
        pixels = np.load(destination, mmap_mode="r", allow_pickle=False)
        if pixels.shape != shape or pixels.dtype != np.uint8:
            raise ValueError(f"Invalid crop cache {destination}")
        return destination

    with NamedTemporaryFile(dir=directory, prefix=".crops-", delete=False) as temporary:
        pending = Path(temporary.name)
    try:
        pixels = np.lib.format.open_memmap(pending, mode="w+", dtype=np.uint8, shape=shape)
        for image, items in grouped.items():
            with Image.open(image, formats=["PNG"]) as source:
                source.load()
                for index, crop in items:
                    pixels[index] = damage_crop_rgb(source, crop.box, size)
        pixels.flush()
        del pixels
        pending.replace(destination)
    finally:
        pending.unlink(missing_ok=True)
    return destination
