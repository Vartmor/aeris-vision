"""Frozen data sources and run utilities shared by training tasks."""

import json
import subprocess
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from aeris.config import ConfigError, load_config
from aeris.data.manifest import read_manifest
from aeris.splits.storage import plan_from_manifest, verify_split


def load_training_sources(path: Path) -> tuple[Path, dict, dict]:
    config = load_config(path)
    dataset, split = config.get("dataset"), config.get("split")
    if not isinstance(dataset, dict) or not isinstance(split, dict):
        raise ConfigError(f"{path}: dataset and split mappings are required")
    for section, key in (
        (dataset, "raw_dir"), (dataset, "manifest_path"), (split, "output_path"),
    ):
        if not isinstance(section.get(key), str) or not section[key]:
            raise ConfigError(f"{path}: {key} must be a nonempty path")
    fractions = split.get("id_fractions")
    groups = split.get("id_evaluation_groups")
    event = split.get("ood_event")
    if not isinstance(fractions, dict) or (groups is not None and not isinstance(groups, dict)):
        raise ConfigError(f"{path}: invalid id_fractions or id_evaluation_groups")
    if event is not None and (not isinstance(event, str) or not event.strip()):
        raise ConfigError(f"{path}: ood_event must be a nonempty name or null")
    manifest_path = Path(dataset["manifest_path"])
    plan = plan_from_manifest(manifest_path, fractions, ood_event=event, id_evaluation_groups=groups)
    verify_split(Path(split["output_path"]), plan)
    records = {record["sample_id"]: record for record in read_manifest(manifest_path)}
    return Path(dataset["raw_dir"]), records, plan


def training_loader(dataset: Dataset, batch: int, workers: int, cuda: bool, seed: int | None) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(
        dataset, batch_size=batch, shuffle=seed is not None,
        num_workers=workers, persistent_workers=workers > 0, pin_memory=cuda, generator=generator,
    )


def write_run_report(path: Path, report: dict) -> None:
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()
