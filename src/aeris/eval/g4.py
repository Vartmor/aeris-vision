"""Fit on ID calibration, freeze the protocol, then evaluate both final tests."""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from aeris.calibration.frozen import SCHEMA_VERSION, load_freeze, protocol, sha256
from aeris.calibration.metrics import SelectiveScores, calibration_metrics, fit_temperature, probabilities, select_threshold
from aeris.config import ConfigError, load_config
from aeris.eval.damage import damage_metrics
from aeris.eval.pipeline import Evaluation, load_evaluation, run_partition

FINAL_PARTITIONS = ("test", "ood_test")
_CODE_FILES = (
    "calibration/frozen.py", "calibration/metrics.py", "data/damage.py", "data/images.py",
    "data/labels.py", "data/raster.py", "eval/buildings.py", "eval/damage.py", "eval/g4.py",
    "eval/pipeline.py", "eval/xview2.py", "infer/components.py", "infer/pipeline.py",
    "models/damage.py", "models/localization.py",
)


def _source_sha256() -> dict:
    source = Path(__file__).resolve().parents[1]
    return {name: sha256(source / name) for name in _CODE_FILES}


def _target_coverage(config_path: Path) -> float:
    section = load_config(config_path).get("calibration")
    target = section.get("target_coverage") if isinstance(section, dict) else None
    if type(target) not in (int, float) or not 0 < target <= 1:
        raise ConfigError("calibration.target_coverage must be in (0, 1]")
    return float(target)


def _new_directory(output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Run directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_score_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _arrays(rows: list[dict]) -> np.ndarray:
    return np.asarray([row["logits"] for row in rows], dtype=np.float64).reshape((-1, 4))


def oracle_arrays(tiles: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    rows = [row for tile in tiles for row in tile["oracle"]]
    return _arrays(rows), np.asarray([row["damage_class"] for row in rows], dtype=np.intp)


def candidate_scores(tiles: list[dict], temperature: float) -> SelectiveScores:
    rows, outcome = [], []
    for tile in tiles:
        references = {row["building_id"]: row["damage_class"] for row in tile["references"]}
        matched = {match["candidate_id"]: match["reference_id"] for match in tile["matches"]}
        for row in tile["candidates"]:
            rows.append(row)
            reference = matched.get(row["building_id"])
            if reference is None:
                outcome.append(1)
            else:
                label = references[reference]
                outcome.append(-1 if label is None else int(row["damage_class"] != label))
    return SelectiveScores(probabilities(_arrays(rows), temperature).max(axis=1), np.asarray(outcome, dtype=np.int8))


def oracle_scores(logits: np.ndarray, labels: np.ndarray, temperature: float) -> SelectiveScores:
    prediction = logits.argmax(axis=1) + 1
    return SelectiveScores(probabilities(logits, temperature).max(axis=1), (prediction != labels).astype(np.int8))


def calibrate(
    config_path: Path, data_config: Path, output: Path, device: str | None = None,
) -> dict:
    _new_directory(output)
    evaluation = load_evaluation(config_path, data_config, device)
    target = _target_coverage(config_path)
    source_sha256 = _source_sha256()
    config_sha256 = sha256(config_path)
    with TemporaryDirectory(dir=output.parent, prefix=".calibration-") as working:
        stage = Path(working)
        run_partition(evaluation, "calibration", stage, capture=True, score_pixels=False)
        scores_path = stage / "buildings.jsonl"
        tiles = read_score_rows(scores_path)
        logits, labels = oracle_arrays(tiles)
        temperature = fit_temperature(logits, labels)
        oracle = oracle_scores(logits, labels, temperature)
        candidate = candidate_scores(tiles, temperature)
        thresholds = {
            "oracle": select_threshold(oracle.confidence, target),
            "candidate": select_threshold(candidate.confidence, target),
        }
        record = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "partition": "calibration", "images": len(tiles),
            "events": evaluation.split["summary"]["calibration"]["events"],
            "manifest_sha256": evaluation.split["manifest_sha256"],
            "calibration_scores_sha256": sha256(scores_path),
            "config_sha256": config_sha256, "source_sha256": source_sha256,
            "model": evaluation.identity, "settings": asdict(evaluation.pipeline.settings),
            "matching": evaluation.rules, "protocol": protocol(target),
            "temperature": temperature, "thresholds": thresholds,
            "achieved": {
                "oracle": oracle.summary(thresholds["oracle"]),
                "candidate": candidate.summary(thresholds["candidate"]),
            },
            "diagnostics": {
                "raw": calibration_metrics(logits, labels),
                "calibrated": calibration_metrics(logits, labels, temperature),
            },
        }
        _write_json(stage / "freeze.json", record)
        stage.rename(output)
    return record


def _verify_protocol(evaluation: Evaluation, config_path: Path, freeze_path: Path):
    frozen = load_freeze(freeze_path)
    record = frozen.record
    frozen.check_model(evaluation.identity, evaluation.pipeline.settings)
    if (
        record["model"] != evaluation.identity
        or record["manifest_sha256"] != evaluation.split["manifest_sha256"]
        or record["matching"] != evaluation.rules
        or record["config_sha256"] != sha256(config_path)
        or record["source_sha256"] != _source_sha256()
    ):
        raise ValueError("Runtime, data, settings, or evaluator differ from the frozen protocol")
    if record["calibration_scores_sha256"] != sha256(freeze_path.with_name("buildings.jsonl")):
        raise ValueError("The saved ID calibration scores differ from the frozen record")
    return frozen


def _score_final_partition(
    evaluation: Evaluation, stage: Path, partition: str, temperature: float, oracle_tau: float, candidate_tau: float,
) -> dict:
    stage.mkdir()
    core = run_partition(evaluation, partition, stage, capture=True)
    scores_path = stage / "buildings.jsonl"
    tiles = read_score_rows(scores_path)
    logits, labels = oracle_arrays(tiles)
    oracle = oracle_scores(logits, labels, temperature)
    candidate = candidate_scores(tiles, temperature)
    curves = {"oracle": oracle.curve(), "candidate": candidate.curve()}
    _write_json(stage / "risk_coverage.json", curves)
    counts = core["buildings"]
    eligible = damage_metrics(labels - 1, logits)
    for row in eligible["by_class"].values():
        if not row["support"]:
            row["recall"] = None
            row["f1"] = None
    result = {
        **core, "raw_logits_sha256": sha256(scores_path),
        "risk_coverage_sha256": sha256(stage / "risk_coverage.json"),
        "source_split": evaluation.split["summary"][partition],
        "unmatched_building_rate": counts["unmatched_references"] / counts["references"]
        if counts["references"] else None,
        "oracle_building_damage": eligible,
        "oracle_calibration": {
            "raw": calibration_metrics(logits, labels),
            "calibrated": calibration_metrics(logits, labels, temperature),
        },
        "review": {"oracle": oracle.summary(oracle_tau), "candidate": candidate.summary(candidate_tau)},
    }
    _write_json(stage / "results.json", result)
    return result


def evaluate_final(
    config_path: Path, data_config: Path, freeze_path: Path, output: Path, device: str | None = None,
) -> dict:
    _new_directory(output)
    if not freeze_path.is_file():
        raise FileNotFoundError(f"Run calibration and freeze its settings first: {freeze_path}")
    evaluation = load_evaluation(config_path, data_config, device)
    frozen = _verify_protocol(evaluation, config_path, freeze_path)
    with TemporaryDirectory(dir=output.parent, prefix=".final-evaluation-") as working:
        stage = Path(working)
        results = {
            partition: _score_final_partition(
                evaluation, stage / partition, partition, frozen.temperature,
                frozen.oracle_threshold, frozen.candidate_threshold,
            )
            for partition in FINAL_PARTITIONS
        }
        record = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "freeze_sha256": frozen.sha256, "freeze": frozen.record,
            "thresholds": frozen.record["thresholds"], "temperature": frozen.temperature,
            "bootstrap": {
                "status": "insufficient_frozen_groups",
                "groups": {partition: evaluation.split["summary"][partition]["groups"] for partition in FINAL_PARTITIONS},
            },
            "partitions": results,
        }
        _write_json(stage / "final.json", record)
        stage.rename(output)
    return record
