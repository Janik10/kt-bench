"""One scorer for every model.

Feeds whole test sequences to ``model.predict`` and scores one prediction per
question. Following pyKT, each student's first answer is not scored by
default: no model has any history for it, and leaving it in only blurs
comparisons with published numbers.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ktbench import metrics
from ktbench.analysis import cold_start
from ktbench.models.base import KTModel


def evaluate(model: KTModel, df: pd.DataFrame, skip_first: bool = True) -> tuple[dict, np.ndarray]:
    t0 = time.perf_counter()
    p = np.asarray(model.predict(df), dtype=np.float64)
    secs = time.perf_counter() - t0
    if p.shape != (len(df),):
        raise ValueError(f"{model.name}.predict returned shape {p.shape}, expected ({len(df)},)")

    keep = df["t"].to_numpy() > 0 if skip_first else np.ones(len(df), bool)
    y, t = df["correct"].to_numpy()[keep], df["t"].to_numpy()[keep]
    result = metrics.all_metrics(y, p[keep])
    result["cold_start"] = cold_start.by_history_length(t, y, p[keep])
    result["infer_secs"] = secs
    result["infer_us_per_answer"] = 1e6 * secs / max(len(df), 1)
    return result, p
