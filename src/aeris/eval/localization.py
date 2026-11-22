"""Binary localization metrics and full-tile validation mask export."""

from collections.abc import Iterator, Sequence
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader


def localization_metrics(tp: int, fp: int, fn: int) -> dict:
    union = tp + fp + fn
    return {
        "f1": 2 * tp / (tp + union) if union else 0.0,
        "iou": tp / union if union else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
    }


def logit_cutoffs(thresholds: Sequence[float], device: torch.device) -> torch.Tensor:
    return torch.tensor(thresholds, dtype=torch.float64).logit().to(device, dtype=torch.float32)


def _predict_patches(
    model: nn.Module, loader: DataLoader, device: torch.device, amp: bool, threshold: float,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    cutoff = logit_cutoffs((threshold,), device)[0]
    for inputs, targets in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(inputs.to(device, non_blocking=True))
        masks = (logits.float() >= cutoff).cpu().numpy()[:, 0]
        yield from zip(masks, targets.numpy()[:, 0].astype(bool), strict=True)


@torch.inference_mode()
def write_validation_masks(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    threshold: float,
    output: Path,
) -> dict:
    """Reassemble ordered validation crops and score the emitted masks in tile space."""
    model.eval()
    output.mkdir(parents=True, exist_ok=False)
    patches = loader.dataset.patches
    batches = _predict_patches(model, loader, device, amp, threshold)
    by_tile = groupby(zip(patches, batches, strict=True), key=lambda pair: pair[0].sample_id)
    tile_results = []
    for sample_id, group in by_tile:
        regions = list(group)
        width = max(patch.box[2] for patch, _ in regions)
        height = max(patch.box[3] for patch, _ in regions)
        mask = np.zeros((height, width), dtype=np.uint8)
        target = np.zeros((height, width), dtype=bool)
        for patch, (predicted, actual) in regions:
            left, top, right, bottom = patch.box
            mask[top:bottom, left:right] = predicted
            target[top:bottom, left:right] = actual
        tp = int(np.count_nonzero(mask & target))
        fp = int(np.count_nonzero(mask)) - tp
        fn = int(np.count_nonzero(target)) - tp
        name = f"{sample_id}_localization_prediction.png"
        Image.fromarray(mask).save(output / name)
        tile_results.append({"sample_id": sample_id, "filename": name, **localization_metrics(tp, fp, fn)})

    tp, fp, fn = (sum(row[key] for row in tile_results) for key in ("tp", "fp", "fn"))
    return {
        "directory": output.name, "count": len(tile_results), "threshold": threshold,
        **localization_metrics(tp, fp, fn), "tiles": tile_results,
    }
