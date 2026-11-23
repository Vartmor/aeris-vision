"""Two-stage inference from a co-registered RGB pair."""

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from aeris.config import ConfigError, load_config
from aeris.data.damage import damage_crop_box, damage_crop_rgb
from aeris.data.images import normalize_rgb
from aeris.data.labels import DAMAGE_CLASS_NAMES
from aeris.eval.localization import logit_cutoffs
from aeris.infer.components import Footprint, candidate_footprints
from aeris.models.damage import build_damage_model
from aeris.models.localization import LocalizationUNet


@dataclass(frozen=True, slots=True)
class InferenceSettings:
    localization_crop_size: int
    localization_threshold: float
    minimum_component_area: int
    damage_crop_size: int
    context_padding: float
    minimum_side: int
    localization_batch: int = 2
    damage_batch: int = 8


@dataclass(frozen=True, slots=True)
class Detection:
    footprint: Footprint
    damage_class: int
    logits: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Prediction:
    localization: np.ndarray
    damage: np.ndarray
    detections: tuple[Detection, ...]
    filtered_components: int


class InferencePipeline:
    def __init__(
        self, localization: nn.Module, damage: nn.Module, settings: InferenceSettings,
        device: torch.device, *, amp: bool,
    ) -> None:
        self.localization = localization.to(device).eval()
        self.damage = damage.to(device).eval()
        self.settings = settings
        self.device = device
        self.amp = amp
        self.cutoff = logit_cutoffs((settings.localization_threshold,), device)[0]

    @torch.inference_mode()
    def localize(self, image: Image.Image) -> np.ndarray:
        patch_size = self.settings.localization_crop_size
        width, height = image.size
        positions = [(left, top) for top in range(0, height, patch_size)
                     for left in range(0, width, patch_size)]
        mask = np.zeros((height, width), dtype=np.uint8)
        batch_size = self.settings.localization_batch
        for start in range(0, len(positions), batch_size):
            batch = positions[start:start + batch_size]
            inputs = torch.stack([
                normalize_rgb(np.array(image.crop((x, y, x + patch_size, y + patch_size)))) for x, y in batch
            ]).to(self.device)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                logits = self.localization(inputs)
            bits = (logits.float() >= self.cutoff)[:, 0].cpu().numpy()
            for (x, y), bit in zip(batch, bits, strict=True):
                right, bottom = min(x + patch_size, width), min(y + patch_size, height)
                mask[y:bottom, x:right] = bit[:bottom - y, :right - x]
        return mask

    @torch.inference_mode()
    def classify(self, image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
        size, batch_size = self.settings.damage_crop_size, self.settings.damage_batch
        result = np.empty((len(boxes), len(DAMAGE_CLASS_NAMES)), dtype=np.float32)
        for start in range(0, len(boxes), batch_size):
            batch = boxes[start:start + batch_size]
            inputs = torch.stack([normalize_rgb(damage_crop_rgb(image, box, size)) for box in batch])
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                scores = self.damage(inputs.to(self.device))
            result[start:start + len(batch)] = scores.float().cpu().numpy()
        if not np.isfinite(result).all():
            raise RuntimeError("Damage classifier produced non-finite logits")
        return result

    def predict(self, pre: Image.Image, post: Image.Image) -> Prediction:
        if pre.mode != "RGB" or post.mode != "RGB" or pre.size != post.size:
            raise ValueError("Inference needs same-size RGB pre and post images")
        buildings, dropped = candidate_footprints(
            self.localize(pre), minimum_area=self.settings.minimum_component_area,
            minimum_side=self.settings.minimum_side,
        )
        boxes = []
        for item in buildings:
            box = damage_crop_box(
                item.bbox, post.size,
                padding=self.settings.context_padding, minimum_side=self.settings.minimum_side,
            )
            if box is None:
                raise RuntimeError(f"Retained component {item.building_id} has no classifiable bounding box")
            boxes.append(box)
        scores = self.classify(post, boxes)
        width, height = pre.size
        damage = np.zeros((height, width), dtype=np.uint8)
        flattened = damage.ravel()
        detections = []
        for building, logits in zip(buildings, scores, strict=True):
            predicted = int(logits.argmax()) + 1
            flattened[building.pixels] = predicted
            detections.append(Detection(building, predicted, tuple(logits.tolist())))
        return Prediction((damage != 0).astype(np.uint8), damage, tuple(detections), dropped)


def _read_checkpoint(path: Path) -> tuple[dict, str]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"{path}: expected checkpoint metadata and a model state dictionary")
    return payload, digest


def load_pipeline(path: Path, device_override: str | None = None) -> tuple[InferencePipeline, dict]:
    config = load_config(path)
    runtime, checkpoints, settings = (config.get(key) for key in ("runtime", "checkpoints", "localization"))
    if not all(isinstance(value, dict) for value in (runtime, checkpoints, settings)):
        raise ConfigError("Inference needs runtime, checkpoints and localization mappings")
    if runtime.get("backend") != "pytorch" or runtime.get("amp", "auto") not in ("auto", "off"):
        raise ConfigError("Inference expects a PyTorch backend and amp auto or off")
    if not all(isinstance(checkpoints.get(kind), str) and checkpoints[kind] for kind in ("localization", "damage")):
        raise ConfigError("Both checkpoint paths are required")
    loc, loc_sha = _read_checkpoint(Path(checkpoints["localization"]))
    damage, damage_sha = _read_checkpoint(Path(checkpoints["damage"]))
    if not loc.get("split_sha256") or loc["split_sha256"] != damage.get("split_sha256"):
        raise ValueError("Localization and damage checkpoints must use the same frozen split")
    if loc.get("evaluation_partition") != "validation" or damage.get("evaluation_partition") != "validation":
        raise ValueError("Inference needs checkpoints selected on validation")
    if damage.get("classes") != list(DAMAGE_CLASS_NAMES.values()):
        raise ValueError("Damage checkpoint class order differs from the xView2 order")
    threshold, patch_size, min_area = loc.get("threshold"), settings.get("crop_size"), settings.get("minimum_component_area")
    if type(threshold) not in (int, float) or not 0 < threshold < 1:
        raise ValueError("Localization checkpoint needs a valid probability threshold")
    if type(patch_size) is not int or patch_size < 32 or patch_size % 32:
        raise ConfigError("Localization crop size must be a positive multiple of 32")
    if type(min_area) is not int or min_area < 1:
        raise ConfigError("Minimum component area must be a positive integer")
    policy = damage.get("crop_policy") or {}
    crop_size, padding, minimum_side = damage.get("crop_size"), policy.get("padding"), policy.get("minimum_side")
    if type(crop_size) is not int or crop_size < 32 or type(minimum_side) is not int or minimum_side < 1:
        raise ValueError("Invalid damage checkpoint crop geometry")
    if type(padding) not in (int, float) or not math.isfinite(padding) or not 0 <= padding <= 1:
        raise ValueError("Invalid damage checkpoint crop padding")
    device_name = device_override or runtime.get("device", "cpu")
    if device_name not in ("cpu", "cuda", "auto"):
        raise ConfigError("Inference device must be cpu, cuda or auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot initialize a GPU")
    device = torch.device(device_name)
    localization_model = LocalizationUNet()
    damage_model = build_damage_model()
    localization_model.load_state_dict(loc["model"])
    damage_model.load_state_dict(damage["model"])
    amp = device.type == "cuda" and runtime.get("amp", "auto") == "auto"
    parameters = InferenceSettings(patch_size, threshold, min_area, crop_size, padding, minimum_side)
    return InferencePipeline(localization_model, damage_model, parameters, device, amp=amp), {
        "split_sha256": loc["split_sha256"], "localization_checkpoint_sha256": loc_sha,
        "damage_checkpoint_sha256": damage_sha, "device": str(device), "amp": amp,
    }
