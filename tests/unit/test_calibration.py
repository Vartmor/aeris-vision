import numpy as np
import pytest

from aeris.calibration.metrics import SelectiveScores, calibration_metrics, fit_temperature, probabilities, select_threshold


def test_temperature_fits_nll_without_changing_argmax():
    logits = np.tile([5.0, 0.0, 0.0, 0.0], (4, 1))
    labels = np.array([1, 1, 1, 2])
    temperature = fit_temperature(logits, labels)
    assert temperature == pytest.approx(5 / np.log(9))
    raw = calibration_metrics(logits, labels)
    calibrated = calibration_metrics(logits, labels, temperature)
    assert calibrated["nll"] < raw["nll"]
    assert calibrated["ece"] == pytest.approx(0.0, abs=1e-12)
    assert calibrated["accuracy"] == raw["accuracy"] == 0.75
    np.testing.assert_array_equal(probabilities(logits, temperature).argmax(axis=1), logits.argmax(axis=1))


def test_extreme_logits_keep_finite_nll_and_top_bin():
    result = calibration_metrics(np.array([[1000.0, -1000.0, 0.0, 0.0]]), np.array([2]))
    assert result["nll"] == pytest.approx(2000.0)
    assert result["ece"] == 1.0
    assert result["bins"][-1]["count"] == 1


def test_threshold_keeps_ties_and_prefers_higher_realized_coverage():
    confidence = np.array([0.9, 0.9, 0.7, 0.7, 0.7])
    tau = select_threshold(confidence, 0.7)
    assert tau == 0.7
    assert select_threshold(confidence, 0.8) == 0.7
    assert select_threshold(confidence, 0.1) > confidence.max()
    scores = SelectiveScores(confidence, np.array([0, -1, 1, -1, 0]))
    assert scores.summary(0.9)["coverage"] == 0.4
    assert scores.summary(0.9)["accepted_unscored"] == 1
    assert scores.summary(tau)["risk"] == pytest.approx(1 / 3)
    assert [point["coverage"] for point in scores.curve()] == [0.4, 1.0]


def test_no_scored_acceptances_has_undefined_risk():
    scores = SelectiveScores(np.array([0.9, 0.7]), np.array([-1, 1]))
    summary = scores.summary(0.9)
    assert summary["coverage"] == 0.5
    assert summary["risk"] is None
    assert summary["full_coverage_risk"] == 1.0
