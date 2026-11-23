"""Train and measure the building-localization baseline."""

import hashlib
import math
from argparse import Namespace
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from time import perf_counter

import torch
import torchvision
from torch.utils.data import DataLoader

from aeris.config import ConfigError, load_config
from aeris.data.localization import (
    LocalizationPatches,
    cache_localization_tiles,
    tile_patches,
    tiny_patches,
)
from aeris.eval.localization import localization_metrics, logit_cutoffs, write_validation_masks
from aeris.models.localization import LocalizationUNet, bce_dice_loss
from aeris.train.common import git_commit, load_training_sources, training_loader, write_run_report


def _settings(path: Path) -> tuple[dict, dict]:
    config = load_config(path)
    model, training = config.get("model"), config.get("training")
    if not isinstance(model, dict) or not isinstance(training, dict):
        raise ConfigError(f"{path}: model and training mappings are required")
    required = (
        (model, "architecture", "unet"),
        (model, "encoder", "resnet18"),
        (model, "input", "pre_rgb"),
        (training, "loss", "bce_dice"),
        (training, "optimizer", "adamw"),
        (training, "checkpoint_metric", "validation_localization_f1"),
    )
    for section, key, value in required:
        if section.get(key) != value:
            raise ConfigError(f"{path}: {key} must be {value}")
    if model.get("encoder_weights") not in ("imagenet", None):
        raise ConfigError(f"{path}: encoder_weights must be imagenet or null")
    size, batch = training.get("crop_size"), training.get("batch_size")
    if type(size) is not int or size < 64 or size % 32:
        raise ConfigError(f"{path}: crop_size must be a multiple of 32 and at least 64")
    if type(batch) is not int or batch < 1:
        raise ConfigError(f"{path}: batch_size must be a positive integer")
    rate = training.get("learning_rate")
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ConfigError(f"{path}: learning_rate must be positive and finite")
    if training.get("amp") not in ("auto", "off"):
        raise ConfigError(f"{path}: amp must be auto or off")
    return model, training


def _to_device(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images, targets = batch
    return images.to(device, non_blocking=True), targets.to(device, non_blocking=True)


def _training_epoch(model, loader, optimizer, scaler, device, amp) -> dict:
    model.train()
    total_loss = torch.zeros((), device=device)
    wait = 0.0
    batches = iter(loader)
    start = perf_counter()
    for _ in range(len(loader)):
        before_fetch = perf_counter()
        inputs, targets = _to_device(next(batches), device)
        wait += perf_counter() - before_fetch
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            loss = bce_dice_loss(model(inputs), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.detach() * len(inputs)
    loss_value = total_loss.item() / len(loader.dataset)
    duration = perf_counter() - start
    if not math.isfinite(loss_value):
        raise RuntimeError("Training produced non-finite loss")
    return {
        "loss": loss_value,
        "epoch_seconds": duration,
        "batch_input_seconds": wait,
        "patches_per_second": len(loader.dataset) / duration,
        "batches": len(loader),
    }


@torch.inference_mode()
def evaluate_localization(
    model, loader, device: torch.device, amp: bool, thresholds: Sequence[float] = (0.5,),
) -> dict:
    """Pool validation pixel counts at each cutoff from one forward pass."""
    model.eval()
    total_loss = torch.zeros((), device=device)
    cutoffs = logit_cutoffs(thresholds, device)[:, None, None, None, None]
    found = torch.zeros(len(thresholds), dtype=torch.long, device=device)
    correct = torch.zeros_like(found)
    positive = torch.zeros((), dtype=torch.long, device=device)
    for batch in loader:
        inputs, targets = _to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(inputs)
            total_loss += bce_dice_loss(logits, targets) * len(inputs)
        predictions = logits.float().unsqueeze(0) >= cutoffs
        actual = targets.bool()
        found += predictions.flatten(1).sum(dim=1, dtype=torch.int)
        correct += (predictions & actual).flatten(1).sum(dim=1, dtype=torch.int)
        positive += actual.sum()
    count = positive.item()
    curve = [
        {"threshold": threshold, **localization_metrics(tp, predicted - tp, count - tp)}
        for threshold, tp, predicted in zip(thresholds, correct.tolist(), found.tolist(), strict=True)
    ]
    best = min(
        curve,
        key=lambda row: (-row["f1"], round(abs(row["threshold"] - 0.5), 12), row["threshold"]),
    )
    return {"loss": total_loss.item() / len(loader.dataset), **best, "threshold_curve": curve}


def train_localization(args: Namespace) -> None:
    if args.run_dir.exists() or args.run_dir.is_symlink():
        raise FileExistsError(f"Run directory already exists: {args.run_dir}")
    model_settings, training = _settings(args.config)
    raw, records, plan = load_training_sources(args.data_config)
    train_rows = [records[sample] for sample in plan["partitions"]["train"]]
    if args.train_tiles is not None:
        if args.train_tiles > len(train_rows):
            raise ConfigError("train-tiles exceeds the frozen training partition")
        train_rows = Random(args.seed).sample(train_rows, args.train_tiles)
    size, batch = training["crop_size"], training["batch_size"]
    thresholds = sorted(set(args.thresholds))
    if args.tiny:
        train = tiny_patches(raw, train_rows, size, args.seed)
        evaluation = train
    else:
        train = [patch for row in train_rows for patch in tile_patches(raw, row, size)]
        evaluation = [
            patch for sample in plan["partitions"]["validation"]
            for patch in tile_patches(raw, records[sample], size)
        ]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot initialize a GPU")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    cuda = device.type == "cuda"
    amp = cuda and training["amp"] == "auto"
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    cache = None
    cache_seconds = None
    if args.cache_dir is not None:
        start = perf_counter()
        cache = cache_localization_tiles(train + evaluation, args.cache_dir, plan["manifest_sha256"])
        cache_seconds = perf_counter() - start
    train_loader = training_loader(LocalizationPatches(train, cache), batch, args.workers, cuda, args.seed)
    eval_loader = training_loader(LocalizationPatches(evaluation, cache), batch, args.workers, cuda, None)
    model = LocalizationUNet(pretrained=model_settings.get("encoder_weights") == "imagenet").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.0)

    args.run_dir.mkdir(parents=True, exist_ok=False)
    run_started = perf_counter()
    report = {
        "run_id": args.run_dir.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda_build": torch.version.cuda,
        "device": torch.cuda.get_device_name() if cuda else "cpu",
        "cuda_total_bytes": torch.cuda.get_device_properties(0).total_memory if cuda else None,
        "amp": amp,
        "seed": args.seed,
        "workers": args.workers,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "manifest_sha256": plan["manifest_sha256"],
        "split_sha256": plan["split_sha256"],
        "model": model_settings,
        "training": training,
        "cache": {"tiles": len(cache), "prepare_seconds": cache_seconds} if cache is not None else None,
        "train_patches": len(train),
        "evaluation_patches": len(evaluation),
        "evaluation_partition": "train" if args.tiny else "validation",
        "threshold_grid": thresholds,
        "history": [],
    }
    if args.tiny:
        report["tiny_patches"] = [
            {"sample_id": patch.sample_id, "box": patch.box} for patch in train
        ]
        start = perf_counter()
        for index in range(len(train)):
            train_loader.dataset[index]
        elapsed = perf_counter() - start
        report["serial_input"] = {"seconds": elapsed, "patches_per_second": len(train) / elapsed}
    else:
        report["train_tiles"] = [row["sample_id"] for row in train_rows]
        report["train_event_tiles"] = dict(Counter(row["event_id"] for row in train_rows))

    print(f"Train: {len(train)} patches | {report['evaluation_partition']}: {len(evaluation)} patches", flush=True)
    report["initial_evaluation"] = evaluate_localization(model, eval_loader, device, amp, thresholds)
    write_run_report(args.run_dir / "metrics.json", report)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        stats = _training_epoch(model, train_loader, optimizer, scaler, device, amp)
        result = evaluate_localization(model, eval_loader, device, amp, thresholds)
        entry = {"epoch": epoch, "training": stats, "evaluation": result}
        if cuda:
            entry["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            entry["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        report["history"].append(entry)
        if result["f1"] > best:
            path = args.run_dir / "best.pt"
            pending = path.with_suffix(".tmp")
            torch.save({
                "model": {name: param.cpu() for name, param in model.state_dict().items()},
                "model_settings": model_settings,
                "epoch": epoch,
                "threshold": result["threshold"],
                "split_sha256": plan["split_sha256"],
                "evaluation_partition": report["evaluation_partition"],
            }, pending)
            pending.replace(path)
            best = result["f1"]
            report["threshold"] = result["threshold"]
            with path.open("rb") as checkpoint:
                checkpoint_sha = hashlib.file_digest(checkpoint, "sha256").hexdigest()
            report["best_checkpoint"] = {
                "filename": path.name, "sha256": checkpoint_sha, "epoch": epoch, "metrics": result,
            }
        report["duration_seconds"] = perf_counter() - run_started
        write_run_report(args.run_dir / "metrics.json", report)
        print(
            f"Epoch {epoch}/{args.epochs} | loss={stats['loss']:.4f} | "
            f"{report['evaluation_partition']} F1={result['f1']:.4f} @ {result['threshold']:.2f} | "
            f"{stats['epoch_seconds']:.2f}s | {stats['patches_per_second']:.1f} patch/s",
            flush=True,
        )
    if args.validation_masks:
        checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        exported = write_validation_masks(
            model, eval_loader, device, amp, checkpoint["threshold"], args.run_dir / "validation_masks",
        )
        expected = report["best_checkpoint"]["metrics"]
        if any(exported[key] != expected[key] for key in ("tp", "fp", "fn")):
            raise RuntimeError("Exported validation masks differ from the selected checkpoint metrics")
        report["validation_masks"] = exported
        report["duration_seconds"] = perf_counter() - run_started
        write_run_report(args.run_dir / "metrics.json", report)
