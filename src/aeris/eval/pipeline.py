"""Oracle and predicted-footprint evaluation on frozen data partitions."""

import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch
from PIL import Image

from aeris.calibration.frozen import sha256
from aeris.config import ConfigError, load_config
from aeris.data.damage import damage_crop_box
from aeris.data.labels import DAMAGE_CLASS_NAMES, parse_label
from aeris.data.raster import building_pixels
from aeris.data.targets import target_path
from aeris.eval.buildings import BuildingEvaluation, BuildingMask, evaluate_buildings
from aeris.eval.xview2 import score_directories
from aeris.infer.pipeline import InferencePipeline, Prediction, load_pipeline
from aeris.train.common import load_training_sources


@dataclass(frozen=True, slots=True)
class Evaluation:
    raw: Path
    records: dict
    split: dict
    pipeline: InferencePipeline
    identity: dict
    rules: dict


@dataclass(frozen=True, slots=True)
class _Tile:
    predicted: Prediction
    oracle: tuple[np.ndarray, np.ndarray]
    buildings: BuildingEvaluation
    unrasterizable: list[tuple[str, int | None]]
    references: list[tuple[BuildingMask, tuple[float, ...]]]
    oracle_rows: list[dict]


def load_evaluation(config_path: Path, data_config: Path, device: str | None = None) -> Evaluation:
    config = load_config(config_path)
    rules = config.get("evaluation")
    if not isinstance(rules, dict) or (
        rules.get("match_iou_minimum"), rules.get("merge_gt_coverage_threshold"), rules.get("merge_min_buildings")
    ) != (0.5, 0.5, 2):
        raise ConfigError("Evaluation requires the configured v1.1 matching and merge rules")
    raw, records, split = load_training_sources(data_config)
    pipeline, identity = load_pipeline(config_path, device)
    if identity["split_sha256"] != split["split_sha256"]:
        raise ValueError("The selected checkpoints differ from the frozen split")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    cv2.setNumThreads(1)
    return Evaluation(raw, records, split, pipeline, identity, rules)


def _checked(path: Path, digest: str | None) -> Path:
    if digest is not None and sha256(path) != digest:
        raise ValueError(f"{path}: source changed since the frozen manifest")
    return path


def _reference_masks(buildings, width: int, height: int):
    references, unmatched = [], []
    for index, building in enumerate(buildings):
        building_id = f"r{index:05d}"
        damage = building.damage_class if building.damage_class in DAMAGE_CLASS_NAMES else None
        if building.polygon.is_valid:
            rows, columns = building_pixels(building.polygon, width, height)
            if rows.size:
                pixels = frozenset((rows * width + columns).tolist())
                references.append((BuildingMask(building_id, pixels, damage), building.polygon.bounds))
                continue
        unmatched.append((building_id, damage))
    return references, unmatched


def _oracle_masks(pipeline: InferencePipeline, post: Image.Image, references, capture: bool):
    localization = np.zeros((post.height, post.width), dtype=np.uint8)
    damage = np.zeros_like(localization)
    loc_flat, damage_flat = localization.ravel(), damage.ravel()
    candidates = []
    boxes = []
    for reference, bounds in references:
        pixels = np.fromiter(reference.pixels, dtype=np.intp, count=len(reference.pixels))
        loc_flat[pixels] = 1
        if reference.damage_class is not None:
            box = damage_crop_box(
                bounds, post.size, padding=pipeline.settings.context_padding,
                minimum_side=pipeline.settings.minimum_side,
            )
            if box is not None:
                candidates.append((reference, pixels))
                boxes.append(box)
    scores = pipeline.classify(post, boxes)
    rows = []
    for (reference, pixels), box, logits in zip(candidates, boxes, scores, strict=True):
        predicted = int(logits.argmax()) + 1
        damage_flat[pixels] = np.maximum(damage_flat[pixels], predicted)
        if capture:
            rows.append({
                "building_id": reference.building_id, "crop_xyxy": box,
                "damage_class": reference.damage_class, "predicted_class": predicted,
                "logits": logits.tolist(),
            })
    return (localization, damage), rows


def _evaluate_tile(pipeline: InferencePipeline, pre: Image.Image, post: Image.Image, buildings, capture: bool) -> _Tile:
    predicted = pipeline.predict(pre, post)
    references, unrasterizable = _reference_masks(buildings, pre.width, pre.height)
    candidates = [
        BuildingMask(item.footprint.building_id, frozenset(item.footprint.pixels.tolist()), item.damage_class)
        for item in predicted.detections
    ]
    report = evaluate_buildings(
        (reference for reference, _ in references), candidates, image_width=pre.width, image_height=pre.height,
    )
    if unrasterizable:
        severe = report.severe_damage
        report = replace(
            report,
            unmatched_reference_ids=tuple(sorted((*report.unmatched_reference_ids,
                                                  *(key for key, _ in unrasterizable)))),
            severe_damage=replace(severe, not_detected=severe.not_detected
                                  + sum(label in (3, 4) for _, label in unrasterizable)),
        )
    oracle, rows = _oracle_masks(pipeline, post, references, capture)
    return _Tile(predicted, oracle, report, unrasterizable, references, rows)


def evaluate_tile(pipeline: InferencePipeline, pre: Image.Image, post: Image.Image, buildings):
    tile = _evaluate_tile(pipeline, pre, post, buildings, False)
    return tile.predicted, tile.oracle, tile.buildings, len(tile.unrasterizable)


def _write_masks(directory: Path, image_id: str, suffix: str, localization: np.ndarray, damage: np.ndarray) -> None:
    Image.fromarray(localization).save(directory / f"test_localization_{image_id}_{suffix}.png")
    Image.fromarray(damage).save(directory / f"test_damage_{image_id}_{suffix}.png")


def _score_directory(predictions: Path, targets: Path) -> dict:
    score = score_directories(predictions, targets)
    macro = sum(row.f1 for row in score.damage_by_class.values()) / len(DAMAGE_CLASS_NAMES)
    return {**score.to_dict(), "damage_macro_f1": macro}


def _read_tile(evaluation: Evaluation, row: dict, capture: bool) -> tuple[_Tile, Path, Path]:
    hashes = row["sha256"] or {}
    pre = _checked(evaluation.raw / row["pre_image_path"], hashes.get("pre_image"))
    post = _checked(evaluation.raw / row["post_image_path"], hashes.get("post_image"))
    labels = _checked(evaluation.raw / row["post_label_path"], hashes.get("post_label"))
    buildings = parse_label(labels, "post", (row["width"], row["height"])).buildings
    with Image.open(pre, formats=["PNG"]) as pre_image, Image.open(post, formats=["PNG"]) as post_image:
        tile = _evaluate_tile(evaluation.pipeline, pre_image, post_image, buildings, capture)
    return tile, pre, post


def run_partition(
    evaluation: Evaluation, partition: str, stage: Path, *, capture: bool = False, score_pixels: bool = True,
) -> dict:
    """Shared runner. Callers are responsible for enforcing partition access."""
    samples = evaluation.split["partitions"][partition]
    if not samples:
        raise ValueError(f"The frozen {partition} partition is empty")
    predictions = stage / "predictions"
    oracle = stage / "oracle_predictions"
    targets = stage / "targets"
    if score_pixels:
        for directory in (predictions, oracle, targets):
            directory.mkdir()
    counts: Counter = Counter()
    severe_counts: Counter = Counter()
    with (stage / "buildings.jsonl").open("w", encoding="utf-8") as stream:
        for index, sample_id in enumerate(samples):
            row = evaluation.records[sample_id]
            tile, pre, post = _read_tile(evaluation, row, capture)
            report, predicted = tile.buildings, tile.predicted
            image_id = f"{index:05d}"
            if score_pixels:
                hashes = row["sha256"] or {}
                pre_target = _checked(target_path(pre), hashes.get("pre_target"))
                post_target = _checked(target_path(post), hashes.get("post_target"))
                _write_masks(predictions, image_id, "prediction", predicted.localization, predicted.damage)
                _write_masks(oracle, image_id, "prediction", *tile.oracle)
                with Image.open(pre_target, formats=["PNG"]) as loc, Image.open(post_target, formats=["PNG"]) as dmg:
                    _write_masks(targets, image_id, "target", np.asarray(loc), np.asarray(dmg))
            tile_counts = {
                "references": len(report.matches) + len(report.unmatched_reference_ids),
                "candidates": len(predicted.detections), "matches": len(report.matches),
                "unmatched_references": len(report.unmatched_reference_ids),
                "unmatched_candidates": len(report.unmatched_candidate_ids),
                "merged_candidates": len(report.merged_candidates),
                "merged_references": len(report.merged_reference_ids),
                "unrasterizable_references": len(tile.unrasterizable),
                "filtered_components": predicted.filtered_components,
            }
            counts.update(tile_counts)
            severe_counts.update(asdict(report.severe_damage))
            detections = []
            for item in predicted.detections:
                detection = {
                    "building_id": item.footprint.building_id, "bbox_xyxy": item.footprint.bbox,
                    "damage_class": item.damage_class, "logits": item.logits,
                }
                if capture:
                    detection["area_pixels"] = int(item.footprint.pixels.size)
                detections.append(detection)
            entry = {
                "image_id": image_id, "sample_id": sample_id, "counts": tile_counts,
                "candidates": detections, **asdict(report),
            }
            if capture:
                entry.update(
                    group_id=evaluation.split["group_by_sample"][sample_id],
                    spatial_group_id=row["spatial_group_id"], scene_id=row["scene_id"], event_id=row["event_id"],
                    references=[
                        {"building_id": item.building_id, "bbox_xyxy": bounds,
                         "damage_class": item.damage_class, "area_pixels": len(item.pixels)}
                        for item, bounds in tile.references
                    ],
                    unrasterizable=[
                        {"building_id": building_id, "damage_class": label}
                        for building_id, label in tile.unrasterizable
                    ],
                    oracle=tile.oracle_rows,
                )
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
            if (index + 1) % 50 == 0 or index + 1 == len(samples):
                print(f"{partition}: {index + 1}/{len(samples)}", flush=True)
    total_references = counts["references"]
    building_summary = {
        **counts, "localization_recall": counts["matches"] / total_references if total_references else None,
        "severe_damage": {**severe_counts, "total": sum(severe_counts.values())},
    }
    summary = {
        **evaluation.identity, "partition": partition, "images": len(samples),
        "settings": asdict(evaluation.pipeline.settings), "matching": evaluation.rules,
    }
    if score_pixels:
        summary.update(predicted=_score_directory(predictions, targets), oracle=_score_directory(oracle, targets))
    summary["buildings"] = building_summary
    (stage / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def evaluate_validation(
    config_path: Path, data_config: Path, output: Path, device: str | None = None,
) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Evaluation directory already exists: {output}")
    evaluation = load_evaluation(config_path, data_config, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".evaluation-") as working:
        stage = Path(working)
        summary = run_partition(evaluation, "validation", stage)
        stage.rename(output)
    return summary
