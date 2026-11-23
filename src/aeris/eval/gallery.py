"""Choose reproducible error examples and draw local xBD comparison crops."""

from dataclasses import asdict, dataclass
from math import ceil, floor
from pathlib import Path

from PIL import Image, ImageDraw

from aeris.calibration.frozen import sha256
from aeris.calibration.metrics import probabilities

TITLES = {
    "minor-no-damage": "Hafif hasar → hasarsız",
    "major-minor": "Ağır hasar → hafif hasar",
    "destroyed-no-damage": "Yıkılmış → hasarsız",
    "merged": "Bir adayda birleşen binalar",
    "missed-small": "Bulunamayan küçük bina",
    "low-correct": "Düşük güvenle doğru tahmin",
    "high-wrong": "Yüksek güvenle yanlış tahmin",
}
LABELS = {1: "hasarsız", 2: "hafif hasarlı", 3: "ağır hasarlı", 4: "yıkılmış", None: "hasar sınıfı yok"}
_TRANSITIONS = {(2, 1): "minor-no-damage", (3, 2): "major-minor", (4, 1): "destroyed-no-damage"}


@dataclass(frozen=True, slots=True)
class Example:
    partition: str
    sample_id: str
    category: str
    references: tuple[dict, ...]
    candidate: dict | None = None
    predicted_class: int | None = None
    confidence: float | None = None
    iou: float | None = None


def choose_examples(partitions: dict[str, list[dict]], temperature: float, oracle_tau: float) -> dict[str, Example]:
    selected: dict[str, tuple[tuple, Example]] = {}

    def consider(example: Example, priority: float = 0) -> None:
        key = priority, example.partition, example.sample_id, tuple(row["building_id"] for row in example.references)
        previous = selected.get(example.category)
        if previous is None or key < previous[0]:
            selected[example.category] = (key, example)

    for partition, tiles in partitions.items():
        for tile in tiles:
            sample = tile["sample_id"]
            references = {row["building_id"]: row for row in tile["references"]}
            candidates = {row["building_id"]: row for row in tile["candidates"]}
            for match in tile["matches"]:
                reference = references[match["reference_id"]]
                candidate = candidates[match["candidate_id"]]
                label, prediction = reference["damage_class"], candidate["damage_class"]
                category = _TRANSITIONS.get((label, prediction))
                if category:
                    confidence = float(probabilities([candidate["logits"]], temperature).max())
                    consider(Example(partition, sample, category, (reference,), candidate,
                                     prediction, confidence, match["iou"]))

            for merge in tile["merged_candidates"]:
                members = tuple(references[key] for key in merge["reference_ids"])
                consider(Example(partition, sample, "merged", members, candidates[merge["candidate_id"]]))

            genuinely_missed = set(tile["unmatched_reference_ids"]) - set(tile["merged_reference_ids"])
            for key in sorted(genuinely_missed & references.keys()):
                row = references[key]
                consider(Example(partition, sample, "missed-small", (row,)), row["area_pixels"])

            for row in tile["oracle"]:
                truth, prediction = row["damage_class"], row["predicted_class"]
                confidence = float(probabilities([row["logits"]], temperature).max())
                reference = (references[row["building_id"]],)
                if truth == prediction and confidence < oracle_tau:
                    consider(Example(partition, sample, "low-correct", reference,
                                     predicted_class=prediction, confidence=confidence), confidence)
                elif truth != prediction and confidence >= oracle_tau:
                    consider(Example(partition, sample, "high-wrong", reference,
                                     predicted_class=prediction, confidence=confidence), -confidence)
    return {category: selected[category][1] for category in TITLES if category in selected}


def _crop_bounds(boxes: list[tuple], width: int, height: int) -> tuple[int, int, int, int]:
    left, top = min(box[0] for box in boxes), min(box[1] for box in boxes)
    right, bottom = max(box[2] for box in boxes), max(box[3] for box in boxes)
    padding = max(12, (max(right - left, bottom - top) * 0.28))
    return (
        max(0, floor(left - padding)), max(0, floor(top - padding)),
        min(width, ceil(right + padding)), min(height, ceil(bottom + padding)),
    )


def _panel(image: Image.Image, crop: tuple, references: list[tuple], candidate: tuple | None) -> Image.Image:
    image = image.crop(crop).convert("RGB")
    scale = min(440 / image.width, 360 / image.height)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (440, 360), "#281a24")
    offset_x, offset_y = (canvas.width - image.width) // 2, (canvas.height - image.height) // 2
    canvas.paste(image, (offset_x, offset_y))
    draw = ImageDraw.Draw(canvas)
    boxes = [(box, "#65f7d0") for box in references]
    if candidate is not None:
        boxes.append((candidate, "#ffb17a"))
    for box, color in boxes:
        left, top, right, bottom = box
        draw.rectangle(
            ((left - crop[0]) * scale + offset_x, (top - crop[1]) * scale + offset_y,
             (right - crop[0]) * scale + offset_x, (bottom - crop[1]) * scale + offset_y),
            outline=color, width=3,
        )
    return canvas


def write_example(example: Example, raw: Path, rows: dict, destination: Path) -> dict:
    row = rows[example.sample_id]
    references = [tuple(item["bbox_xyxy"]) for item in example.references]
    candidate = tuple(example.candidate["bbox_xyxy"]) if example.candidate else None
    boxes = [*references, candidate] if candidate is not None else references
    crop = _crop_bounds(boxes, row["width"], row["height"])
    result = Image.new("RGB", (890, 360), "#281a24")
    hashes = row["sha256"] or {}
    for offset, phase in ((0, "pre"), (450, "post")):
        source = raw / row[f"{phase}_image_path"]
        digest = hashes.get(f"{phase}_image")
        if digest and sha256(source) != digest:
            raise ValueError(f"{source}: gallery image differs from the frozen manifest")
        with Image.open(source, formats=["PNG"]) as image:
            panel = _panel(image, crop, references, candidate)
        result.paste(panel, (offset, 0))
    result.save(destination)
    return {**asdict(example), "image": destination.name, "crop_xyxy": crop}
