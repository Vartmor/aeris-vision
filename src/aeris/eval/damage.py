"""Building-level validation scores for the four ordered damage categories."""

import numpy as np

from aeris.data.labels import DAMAGE_CLASS_NAMES


def damage_metrics(labels: np.ndarray, logits: np.ndarray) -> dict:
    names = list(DAMAGE_CLASS_NAMES.values())
    classes = len(names)
    if labels.ndim != 1 or logits.shape != (len(labels), classes) or not len(labels):
        raise ValueError("Expected one label and four logits per building")
    if not np.isfinite(logits).all() or np.any((labels < 0) | (labels >= classes)):
        raise ValueError("Invalid damage label or non-finite logits")
    predicted = logits.argmax(axis=1)
    matrix = np.bincount(classes * labels + predicted, minlength=classes**2).reshape(classes, classes)
    by_class = {}
    f1s = []
    for index, name in enumerate(names):
        tp = int(matrix[index, index])
        actual, suggested = int(matrix[index].sum()), int(matrix[:, index].sum())
        f1 = 2 * tp / (actual + suggested) if actual + suggested else 0.0
        f1s.append(f1)
        by_class[name] = {
            "support": actual, "predicted": suggested,
            "precision": tp / suggested if suggested else 0.0,
            "recall": tp / actual if actual else 0.0, "f1": f1,
        }
    return {
        "harmonic_f1": classes / sum(1 / f1 for f1 in f1s) if all(f1s) else 0.0,
        "macro_f1": sum(f1s) / classes,
        "accuracy": float(matrix.trace() / len(labels)),
        "by_class": by_class,
        "confusion": matrix.tolist(),
    }
