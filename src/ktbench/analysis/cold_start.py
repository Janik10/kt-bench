"""AUC split by how many answers the student had given before the one being predicted."""

from __future__ import annotations

import numpy as np

from ktbench import metrics

# (label, lowest prior-answer count, highest) inclusive.
BUCKETS = [("1-5", 1, 5), ("6-20", 6, 20), ("21+", 21, None)]


def by_history_length(t: np.ndarray, y: np.ndarray, p: np.ndarray) -> dict[str, dict[str, float]]:
    out = {}
    for label, lo, hi in BUCKETS:
        sel = (t >= lo) & (t <= hi if hi is not None else True)
        out[label] = {"auc": metrics.auc(y[sel], p[sel]), "n": int(sel.sum())}
    return out
