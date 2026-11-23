"""Temperature scaling, calibration diagnostics, and selective-classification metrics."""

from dataclasses import dataclass

import numpy as np

ECE_BINS = 15
TEMPERATURE_BOUNDS = (0.01, 100.0)


def _logits(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 4 or not np.isfinite(scores).all():
        raise ValueError("Expected four finite logits per building")
    return scores


def _classified(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores, labels = _logits(scores), np.asarray(labels)
    if labels.shape != (len(scores),) or not len(labels) or labels.dtype.kind not in "iu":
        raise ValueError("Expected one integer label per building")
    if np.any((labels < 1) | (labels > 4)):
        raise ValueError("Damage labels must be in the range 1–4")
    return scores, labels.astype(np.intp) - 1


def _centered_scores(scores: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    return (scores - scores.max(axis=1, keepdims=True)) / temperature


def probabilities(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    exp = np.exp(_centered_scores(_logits(scores), temperature))
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(scores: np.ndarray, labels: np.ndarray) -> float:
    """Minimize unweighted NLL by bisection in inverse temperature.

    NLL is convex in inverse temperature. Keep 1 when all four logits are
    identical for every example and the objective is flat.
    """
    scores, indices = _classified(scores, labels)
    centered = scores - scores.max(axis=1, keepdims=True)
    if not np.any(centered):
        return 1.0
    truth = centered[np.arange(len(indices)), indices]

    def derivative(scale: float) -> float:
        exp = np.exp(centered * scale)
        expectation = (exp * centered).sum(axis=1) / exp.sum(axis=1)
        return float(np.mean(expectation - truth))

    minimum, maximum = TEMPERATURE_BOUNDS
    lower, upper = 1.0 / maximum, 1.0 / minimum
    if derivative(lower) >= 0:
        return maximum
    if derivative(upper) <= 0:
        return minimum
    for _ in range(64):
        middle = (lower + upper) / 2
        if derivative(middle) > 0:
            upper = middle
        else:
            lower = middle
    return 1.0 / ((lower + upper) / 2)


def calibration_metrics(scores: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> dict:
    scores, indices = _classified(scores, labels)
    scaled = _centered_scores(scores, temperature)
    exp = np.exp(scaled)
    normalizer = exp.sum(axis=1)
    confidence = exp.max(axis=1) / normalizer
    correct = scores.argmax(axis=1) == indices
    # [0,1/15), ... [14/15,1]. A probability of exactly one is in the last bin.
    bin_ids = np.minimum((confidence * ECE_BINS).astype(int), ECE_BINS - 1)
    count = np.bincount(bin_ids, minlength=ECE_BINS)
    correct_sum = np.bincount(bin_ids, weights=correct, minlength=ECE_BINS)
    confidence_sum = np.bincount(bin_ids, weights=confidence, minlength=ECE_BINS)
    return {
        "samples": len(indices),
        "accuracy": float(correct.mean()),
        "ece": float(np.abs(correct_sum - confidence_sum).sum() / len(indices)),
        "nll": float((np.log(normalizer) - scaled[np.arange(len(indices)), indices]).mean()),
        "bins": [
            {
                "lower": index / ECE_BINS, "upper": (index + 1) / ECE_BINS,
                "count": int(size),
                "accuracy": float(correct_sum[index] / size) if size else None,
                "mean_confidence": float(confidence_sum[index] / size) if size else None,
            }
            for index, size in enumerate(count)
        ],
    }


def _confidence(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Expected a vector of probabilities")
    return values


def select_threshold(confidence: np.ndarray, target_coverage: float) -> float:
    """Use inclusive ties and prefer higher coverage at equal distance."""
    confidence = _confidence(confidence)
    if not len(confidence) or not 0 < target_coverage <= 1:
        raise ValueError("Threshold selection needs scores and a coverage target in (0, 1]")
    ordered = np.sort(confidence)[::-1]
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]) + 1, len(ordered)]
    # Zero accepted is possible just above the highest observed score.
    counts = np.r_[0, ends]
    delta = np.abs(counts - target_coverage * len(ordered))
    tolerance = np.spacing(max(1.0, float(len(ordered))))
    best = int(np.flatnonzero(delta <= delta.min() + tolerance)[-1])
    return float(ordered[ends[best - 1] - 1] if best else np.nextafter(ordered[0], np.inf))


@dataclass(frozen=True, slots=True)
class SelectiveScores:
    """An outcome of -1 is unscored, 0 is correct, and 1 is an error."""

    confidence: np.ndarray
    outcome: np.ndarray

    def __post_init__(self) -> None:
        confidence, outcome = _confidence(self.confidence), np.asarray(self.outcome)
        if outcome.shape != confidence.shape or not np.isin(outcome, (-1, 0, 1)).all():
            raise ValueError("Expected outcomes aligned with confidence scores")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "outcome", outcome)

    def summary(self, threshold: float) -> dict:
        accepted = self.confidence >= threshold
        total = len(accepted)
        covered = int(accepted.sum())
        evaluated = int(np.count_nonzero(accepted & (self.outcome >= 0)))
        errors = int(np.count_nonzero(accepted & (self.outcome == 1)))
        full_evaluated = int(np.count_nonzero(self.outcome >= 0))
        full_errors = int(np.count_nonzero(self.outcome == 1))
        return {
            "threshold": threshold, "total": total, "accepted": covered, "review": total - covered,
            "coverage": covered / total if total else None,
            "scored": evaluated, "errors": errors, "accepted_unscored": covered - evaluated,
            "total_unscored": total - full_evaluated,
            "risk": errors / evaluated if evaluated else None,
            "full_coverage_risk": full_errors / full_evaluated if full_evaluated else None,
        }

    def curve(self) -> list[dict]:
        order = np.argsort(-self.confidence, kind="stable")
        confidence, outcome = self.confidence[order], self.outcome[order]
        if not len(confidence):
            return []
        ends = np.r_[np.flatnonzero(confidence[:-1] != confidence[1:]), len(confidence) - 1]
        scored = np.cumsum(outcome >= 0)
        errors = np.cumsum(outcome == 1)
        return [{
            "threshold": float(confidence[end]), "coverage": float((end + 1) / len(order)),
            "risk": float(errors[end] / scored[end]) if scored[end] else None,
        } for end in ends]
