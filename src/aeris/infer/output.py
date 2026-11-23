"""Run a labeled-free pair and write atomically published JSON and masks."""

import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch
from PIL import Image

from aeris.calibration.frozen import FrozenCalibration, load_freeze
from aeris.calibration.metrics import probabilities
from aeris.data.labels import DAMAGE_CLASS_NAMES
from aeris.infer.pipeline import Detection, load_pipeline


def _building_record(detection: Detection, calibration: FrozenCalibration | None) -> dict:
    logits = np.asarray([detection.logits], dtype=np.float64)
    scores = probabilities(logits, calibration.temperature if calibration else 1.0)[0]
    confidence = float(scores[detection.damage_class - 1])
    footprint = detection.footprint
    return {
        "building_id": footprint.building_id,
        "bbox_xyxy": footprint.bbox,
        "area_pixels": int(footprint.pixels.size),
        "damage_class_id": detection.damage_class,
        "predicted_class": DAMAGE_CLASS_NAMES[detection.damage_class],
        "confidence": confidence,
        "class_probabilities": dict(zip(DAMAGE_CLASS_NAMES.values(), scores.tolist(), strict=True)),
        "logits": detection.logits,
        "manual_review": confidence < calibration.candidate_threshold if calibration else None,
    }


def infer_pair(
    pre: Path, post: Path, config: Path, output: Path, *,
    device: str | None = None, sample_id: str | None = None, calibration_path: Path | None = None,
) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Inference directory already exists: {output}")
    if sample_id is None:
        sample_id = pre.stem.removesuffix("_pre_disaster") or pre.stem
    if not sample_id.strip():
        raise ValueError("Sample ID must not be empty")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    cv2.setNumThreads(1)
    with Image.open(pre) as pre_image, Image.open(post) as post_image:
        pipeline, identity = load_pipeline(config, device)
        calibration = load_freeze(calibration_path) if calibration_path else None
        if calibration:
            calibration.check_model(identity, pipeline.settings)
        prediction = pipeline.predict(pre_image, post_image)
    height, width = prediction.damage.shape
    masks = {"localization": "localization.png", "damage": "damage.png"}
    result = {
        "schema_version": 1,
        "sample_id": sample_id,
        "model_version": version("aeris-vision"),
        "coordinate_system": "image_pixels",
        "image_size": {"width": width, "height": height},
        "confidence_method": "temperature_scaled_softmax" if calibration else "raw_softmax",
        "class_order": list(DAMAGE_CLASS_NAMES.values()),
        "provenance": identity,
        "localization_threshold": pipeline.settings.localization_threshold,
        "minimum_component_area": pipeline.settings.minimum_component_area,
        "filtered_components": prediction.filtered_components,
        "masks": masks,
        "buildings": [_building_record(detection, calibration) for detection in prediction.detections],
    }
    if calibration:
        result["calibration"] = {
            "freeze_sha256": calibration.sha256, "temperature": calibration.temperature,
            "candidate_threshold": calibration.candidate_threshold,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".inference-") as temporary:
        stage = Path(temporary)
        Image.fromarray(prediction.localization).save(stage / masks["localization"])
        Image.fromarray(prediction.damage).save(stage / masks["damage"])
        (stage / "prediction.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.rename(output)
    return result
