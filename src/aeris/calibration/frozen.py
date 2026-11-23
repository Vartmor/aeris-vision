"""Read the immutable calibration artifact shared by final tests and inference."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from aeris.calibration.metrics import ECE_BINS, TEMPERATURE_BOUNDS

SCHEMA_VERSION = 1
MODEL_KEYS = ("localization_checkpoint_sha256", "damage_checkpoint_sha256", "split_sha256")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def protocol(target: float) -> dict:
    return {
        "temperature_bounds": list(TEMPERATURE_BOUNDS),
        "ece_bins": ECE_BINS, "ece_binning": "equal_width_0_to_1_inclusive_last",
        "target_coverage": target, "threshold_comparison": "greater_than_or_equal",
        "coverage_tie_break": "prefer_higher_coverage",
        "candidate_risk": "unmatched_is_error_matched_unknown_excluded",
        "bootstrap": "skip_insufficient_frozen_groups",
    }


@dataclass(frozen=True, slots=True)
class FrozenCalibration:
    sha256: str
    record: dict
    temperature: float
    oracle_threshold: float
    candidate_threshold: float

    def check_model(self, identity: dict, settings: object) -> None:
        if any(self.record["model"].get(key) != identity.get(key) for key in MODEL_KEYS):
            raise ValueError("Frozen calibration belongs to different checkpoints or split")
        if self.record["settings"] != asdict(settings):
            raise ValueError("Inference settings differ from the frozen calibration")


def load_freeze(path: Path) -> FrozenCalibration:
    raw = path.read_bytes()
    record = json.loads(raw)
    if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported frozen calibration schema")
    if record.get("partition") != "calibration":
        raise ValueError("Calibration must originate in the ID calibration partition")
    for key in ("model", "settings", "matching", "protocol", "source_sha256", "thresholds"):
        if not isinstance(record.get(key), dict):
            raise ValueError(f"Frozen calibration is missing {key}")
    for key in ("manifest_sha256", "config_sha256", "calibration_scores_sha256"):
        digest = record.get(key)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Frozen calibration is missing {key}")
    if any(not isinstance(record["model"].get(key), str) for key in MODEL_KEYS):
        raise ValueError("Frozen calibration is missing the checkpoint or split hashes")
    target = record["protocol"].get("target_coverage")
    if type(target) not in (int, float) or not 0 < target <= 1 or record["protocol"] != protocol(target):
        raise ValueError("Unknown frozen calibration evaluation protocol")
    temperature = record.get("temperature")
    if type(temperature) not in (int, float) or not TEMPERATURE_BOUNDS[0] <= temperature <= TEMPERATURE_BOUNDS[1]:
        raise ValueError("Invalid frozen temperature")
    oracle = record["thresholds"].get("oracle")
    candidate = record["thresholds"].get("candidate")
    if any(type(value) not in (int, float) or not 0 <= value <= math.nextafter(1.0, math.inf)
           for value in (oracle, candidate)):
        raise ValueError("Invalid frozen review threshold")
    return FrozenCalibration(hashlib.sha256(raw).hexdigest(), record, temperature, oracle, candidate)
