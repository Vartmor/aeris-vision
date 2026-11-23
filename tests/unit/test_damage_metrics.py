import numpy as np
import pytest

from aeris.eval.damage import damage_metrics


def test_damage_metrics_use_four_sorted_classes_and_zero_harmonic_for_missed_class():
    truth = np.arange(4)
    scores = np.eye(4)[[0, 0, 2, 3]]

    metrics = damage_metrics(truth, scores)

    assert metrics["confusion"] == [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    assert metrics["harmonic_f1"] == 0.0
    assert metrics["macro_f1"] == pytest.approx(2 / 3)
    assert metrics["by_class"]["minor_damage"]["support"] == 1
    assert damage_metrics(truth, np.eye(4))["harmonic_f1"] == 1.0
