"""Train and validate the oracle-footprint post-only damage baseline."""

import hashlib
import math
from argparse import Namespace
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from time import perf_counter

import numpy as np
import torch
import torchvision
from torch.nn.functional import cross_entropy

from aeris.config import ConfigError, load_config
from aeris.data.damage import DamageCrop, DamageCrops, cache_damage_crops, read_damage_crops
from aeris.data.labels import DAMAGE_CLASS_NAMES
from aeris.eval.damage import damage_metrics
from aeris.models.damage import build_damage_model
from aeris.train.common import git_commit, load_training_sources, training_loader, write_run_report


def _settings(path: Path) -> tuple[dict, dict]:
    config = load_config(path)
    model, training = config.get("model"), config.get("training")
    if not isinstance(model, dict) or not isinstance(training, dict):
        raise ConfigError("Damage configuration needs model and training mappings")
    if (model.get("architecture"), model.get("input"), model.get("num_classes")) != ("resnet18", "post_rgb", 4):
        raise ConfigError("Damage model must be post-only ResNet18 with four damage classes")
    if model.get("encoder_weights") not in ("imagenet", None):
        raise ConfigError("encoder_weights must be imagenet or null")
    if training.get("loss") != "weighted_cross_entropy" or training.get("class_weights") != {
        "method": "inverse_square_root_frequency", "maximum": 4.0,
    }:
        raise ConfigError("Damage training expects the v1.1 weighted cross entropy policy")
    if training.get("checkpoint_metric") != "validation_oracle_harmonic_damage_f1":
        raise ConfigError("Damage checkpoint selection requires validation oracle harmonic F1")
    for key, minimum in (("crop_size", 64), ("batch_size", 1), ("min_visible_side", 1)):
        value = training.get(key)
        if type(value) is not int or value < minimum:
            raise ConfigError(f"{key} must be an integer of at least {minimum}")
    for key, minimum, maximum in (("context_padding", 0.0, 1.0), ("learning_rate", 0.0, 1.0)):
        value = training.get(key)
        if type(value) not in (int, float) or not minimum <= value <= maximum:
            raise ConfigError(f"{key} must be between {minimum} and {maximum}")
    if training["learning_rate"] == 0 or training.get("amp") not in ("auto", "off"):
        raise ConfigError("Damage training needs a positive learning rate and amp auto or off")
    return model, training


def _class_counts(crops: list[DamageCrop]) -> dict[str, int]:
    counts = Counter(crop.label for crop in crops)
    return {name: counts[label] for label, name in DAMAGE_CLASS_NAMES.items()}


def _train_epoch(model, loader, device, amp, optimizer, scaler, weights) -> dict:
    model.train()
    total_loss = torch.zeros((), device=device)
    total_weight = torch.zeros_like(total_loss)
    start = perf_counter()
    for step, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            loss = cross_entropy(model(images), labels, weight=weights)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        weight = weights[labels].sum()
        total_loss += loss.detach() * weight
        total_weight += weight
        if step % 500 == 0 and step < len(loader):
            print(f"  Batch {step}/{len(loader)}", flush=True)
    loss_value = (total_loss / total_weight).item()
    seconds = perf_counter() - start
    if not math.isfinite(loss_value):
        raise RuntimeError("Damage training produced non-finite loss")
    return {"loss": loss_value, "epoch_seconds": seconds, "crops_per_second": len(loader.dataset) / seconds}


@torch.inference_mode()
def score_damage(model, loader, device: torch.device, amp: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, truth = [], []
    for images, labels in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            outputs = model(images.to(device, non_blocking=True))
        scores.append(outputs.float().cpu().numpy())
        truth.append(labels.numpy())
    return np.concatenate(scores), np.concatenate(truth)


def _save_logits(path: Path, scores: np.ndarray, labels: np.ndarray, crops: list[DamageCrop]) -> None:
    pending = path.with_suffix(".tmp")
    with pending.open("wb") as file:
        np.savez_compressed(
            file, logits=scores, label_id=labels + 1,
            sample_id=np.array([crop.sample_id for crop in crops]),
            building_uid=np.array([crop.uid or "" for crop in crops]),
            building_index=np.array([crop.building_index for crop in crops]),
            box=np.array([crop.box for crop in crops]),
        )
    pending.replace(path)


def _read_rate(dataset: DamageCrops, seed: int) -> float:
    indices = Random(seed).sample(range(len(dataset)), min(32, len(dataset)))
    start = perf_counter()
    for index in indices:
        dataset[index]
    return len(indices) / (perf_counter() - start)


def train_damage(args: Namespace) -> None:
    if args.run_dir.exists() or args.run_dir.is_symlink():
        raise FileExistsError(f"Run directory already exists: {args.run_dir}")
    model_config, training = _settings(args.config)
    root, records, plan = load_training_sources(args.data_config)
    policy = {"padding": training["context_padding"], "minimum_side": training["min_visible_side"]}
    train, skipped = read_damage_crops(root, [records[key] for key in plan["partitions"]["train"]], **policy)
    eligible_counts = _class_counts(train)
    rng = Random(args.seed)
    if args.tiny:
        train = [crop for label in DAMAGE_CLASS_NAMES for crop in rng.sample(
            [crop for crop in train if crop.label == label], 8,
        )]
    elif args.train_buildings is not None:
        if args.train_buildings > len(train):
            raise ConfigError("train-buildings exceeds the eligible training crop count")
        indices = sorted(rng.sample(range(len(train)), args.train_buildings))
        train = [train[index] for index in indices]
    elif args.max_crops_per_class is not None:
        groups = {label: [] for label in DAMAGE_CLASS_NAMES}
        for index, crop in enumerate(train):
            groups[crop.label].append(index)
        selected = [index for indices in groups.values() for index in rng.sample(
            indices, min(args.max_crops_per_class, len(indices)),
        )]
        train = [train[index] for index in sorted(selected)]
    counts = _class_counts(train)
    if not all(counts.values()):
        raise ValueError("Selected training crop set needs support for all four classes")
    weighted = {name: min(4.0, math.sqrt(max(counts.values()) / count)) for name, count in counts.items()}
    partition = "train" if args.tiny else "validation"
    validation, val_skipped = (train, Counter()) if args.tiny else read_damage_crops(
        root, [records[key] for key in plan["partitions"]["validation"]], **policy,
    )
    if not validation:
        raise ValueError("No eligible evaluation crops")

    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    cuda = device.type == "cuda"
    amp = cuda and training["amp"] == "auto"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot initialize a GPU")
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    size, batch = training["crop_size"], training["batch_size"]
    train_cache, val_cache, cache_stats = None, None, None
    if args.cache_dir is not None:
        uncached_rate = _read_rate(DamageCrops(train, size), args.seed)
        start = perf_counter()
        train_cache = cache_damage_crops(train, size, args.cache_dir, plan["manifest_sha256"], "train")
        val_cache = train_cache if args.tiny else cache_damage_crops(
            validation, size, args.cache_dir, plan["manifest_sha256"], "validation",
        )
        cache_stats = {"prepare_seconds": perf_counter() - start, "png_crops_per_second": uncached_rate,
                       "cached_crops_per_second": _read_rate(DamageCrops(train, size, train_cache), args.seed)}

    train_loader = training_loader(DamageCrops(train, size, train_cache), batch, args.workers, cuda, args.seed)
    val_loader = training_loader(DamageCrops(validation, size, val_cache), batch, args.workers, cuda, None)
    model = build_damage_model(pretrained=model_config.get("encoder_weights") == "imagenet").to(device)
    weights = torch.tensor(list(weighted.values()), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.0)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    report = {
        "run_id": args.run_dir.name, "timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(),
        "torch": str(torch.__version__), "torchvision": str(torchvision.__version__), "cuda_build": torch.version.cuda,
        "device": torch.cuda.get_device_name() if cuda else "cpu", "amp": amp, "seed": args.seed, "workers": args.workers,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "model": model_config, "training": training,
        "manifest_sha256": plan["manifest_sha256"], "split_sha256": plan["split_sha256"],
        "classes": list(DAMAGE_CLASS_NAMES.values()), "class_counts": counts, "class_weights": weighted,
        "subset": {"train_buildings": args.train_buildings, "max_crops_per_class": args.max_crops_per_class,
                   "tiny": args.tiny},
        "eligible_train_counts": eligible_counts, "train_crops": len(train), "train_tiles": len({c.sample_id for c in train}),
        "train_filter_counts": dict(skipped), "evaluation_filter_counts": dict(val_skipped),
        "evaluation_crops": len(validation), "evaluation_partition": partition,
        "evaluation_counts": _class_counts(validation), "optimizer": "adamw", "cache": cache_stats, "history": [],
    }
    print(f"Train: {len(train)} crops | {partition}: {len(validation)} crops | class counts {counts}", flush=True)
    write_run_report(args.run_dir / "metrics.json", report)
    best = (-1.0, -1.0)
    for epoch in range(1, args.epochs + 1):
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        stats = _train_epoch(model, train_loader, device, amp, optimizer, scaler, weights)
        scores, labels = score_damage(model, val_loader, device, amp)
        metrics = damage_metrics(labels, scores)
        entry = {"epoch": epoch, "training": stats, "evaluation": metrics}
        if cuda:
            entry["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            entry["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        report["history"].append(entry)
        rank = metrics["harmonic_f1"], metrics["macro_f1"]
        if rank > best:
            best = rank
            pending = args.run_dir / "checkpoint.tmp"
            checkpoint = args.run_dir / "best.pt"
            torch.save({
                "model": {name: value.cpu() for name, value in model.state_dict().items()},
                "model_config": model_config, "crop_size": size, "crop_policy": policy, "epoch": epoch,
                "classes": report["classes"], "split_sha256": plan["split_sha256"],
                "evaluation_partition": partition,
            }, pending)
            pending.replace(checkpoint)
            logits_path = args.run_dir / f"{partition}_logits.npz"
            _save_logits(logits_path, scores, labels, validation)
            with checkpoint.open("rb") as file:
                checkpoint_sha = hashlib.file_digest(file, "sha256").hexdigest()
            with logits_path.open("rb") as file:
                logits_sha = hashlib.file_digest(file, "sha256").hexdigest()
            report["best_checkpoint"] = {
                "epoch": epoch, "filename": checkpoint.name, "sha256": checkpoint_sha,
                "logits": logits_path.name, "logits_sha256": logits_sha, "metrics": metrics,
            }
        report["duration_seconds"] = perf_counter() - started
        write_run_report(args.run_dir / "metrics.json", report)
        print(f"Epoch {epoch}/{args.epochs} | loss={stats['loss']:.4f} | {partition} harmonic F1={rank[0]:.4f} | "
              f"macro F1={rank[1]:.4f} | {stats['crops_per_second']:.1f} crop/s", flush=True)
