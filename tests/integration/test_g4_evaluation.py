import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from aeris.eval import g4


def test_matching_unknown_reference_and_unmatched_candidate_have_distinct_risk_roles():
    tiles = [{
        "references": [{"building_id": "r", "damage_class": None}],
        "matches": [{"candidate_id": "c", "reference_id": "r"}],
        "candidates": [
            {"building_id": "c", "damage_class": 1, "logits": [4, 0, 0, 0]},
            {"building_id": "extra", "damage_class": 1, "logits": [4, 0, 0, 0]},
        ],
    }]
    fixed = g4.candidate_scores(tiles, 1).summary(0.5)
    assert fixed["coverage"] == 1
    assert fixed["accepted_unscored"] == 1
    assert fixed["scored"] == fixed["errors"] == 1
    assert fixed["risk"] == 1


def test_final_partition_access_requires_existing_and_verified_freeze(monkeypatch, tmp_path):
    freeze = tmp_path / "freeze.json"
    forbidden = lambda *args, **kwargs: pytest.fail("Opened an ID/OOD partition without a valid freeze")
    monkeypatch.setattr(g4, "_score_final_partition", forbidden)
    monkeypatch.setattr(g4, "load_evaluation", forbidden)
    with pytest.raises(FileNotFoundError, match="freeze"):
        g4.evaluate_final(Path("inference.yaml"), Path("data.yaml"), freeze, tmp_path / "final")

    freeze.write_text(json.dumps({"schema_version": 1, "partition": "test"}), encoding="utf-8")
    monkeypatch.setattr(g4, "load_evaluation", lambda *args: object())
    with pytest.raises(ValueError, match="ID calibration"):
        g4.evaluate_final(Path("inference.yaml"), Path("data.yaml"), freeze, tmp_path / "final")
    assert not (tmp_path / "final").exists()
