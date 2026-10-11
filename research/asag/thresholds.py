"""
Maps a 0-100 score to the three labels with two cut-offs tuned on the dev split only.

Cut-offs are chosen to maximise QWK on dev (ties: higher accuracy, then the lowest cut-offs).
Candidates are the midpoints between neighbouring distinct dev scores, plus one point below the
lowest and one above the highest, so every distinct way of splitting the dev scores is tried.
"""
import itertools
import math
from typing import Dict, Sequence, Tuple

import numpy as np

from research.evaluate import accuracy, qwk

LABELS = ("incorrect", "partial", "correct")
# The app's fixed grade cut-offs (Poor < 50 <= Moderate < 80 <= Strong), for the untuned variants
APP_CUTOFFS = (50.0, 80.0)


def apply(scores: Sequence[float], cutoffs: Tuple[float, float]) -> np.ndarray:
    """0 below the first cut-off, 1 between, 2 at or above the second."""
    low, high = cutoffs
    s = np.asarray(scores, dtype=float)
    return np.where(s >= high, 2, np.where(s >= low, 1, 0))


def labels_for(scores: Sequence[float], cutoffs: Tuple[float, float]) -> list:
    return [LABELS[i] for i in apply(scores, cutoffs)]


def candidates(scores: Sequence[float]) -> np.ndarray:
    values = np.unique(np.asarray(scores, dtype=float))
    middle = (values[:-1] + values[1:]) / 2
    return np.concatenate([[values[0] - 1.0], middle, [values[-1] + 1.0]])


def tune(dev_scores: Sequence[float], dev_labels: Sequence[str]) -> Dict[str, float]:
    """Returns {"low", "high", "dev_qwk", "dev_accuracy"} for the best cut-offs on dev."""
    index = {label: i for i, label in enumerate(LABELS)}
    y = np.array([index[label] for label in dev_labels])
    best = None
    for low, high in itertools.combinations_with_replacement(candidates(dev_scores), 2):
        pred = apply(dev_scores, (low, high))
        k = qwk(y, pred, len(LABELS))
        k = -math.inf if math.isnan(k) else k
        key = (k, accuracy(y, pred), -low, -high)
        if best is None or key > best[0]:
            best = (key, low, high)
    (k, acc, _, _), low, high = best
    return {"low": float(low), "high": float(high), "dev_qwk": float(k), "dev_accuracy": float(acc)}
