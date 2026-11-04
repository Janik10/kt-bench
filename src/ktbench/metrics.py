"""Scoring metrics. Every model is scored by these same functions."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-7


def auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def accuracy(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    return float(np.mean((p >= threshold) == (y == 1)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    """Equal-width bins on [0, 1]. Returns (mean predicted, observed rate, count) per bin."""
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    sum_p = np.bincount(idx, weights=p, minlength=n_bins)
    sum_y = np.bincount(idx, weights=y, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sum_p / count, sum_y / count, count


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: count-weighted |mean predicted - observed| over bins."""
    mean_p, rate, count = calibration_bins(y, p, n_bins)
    nonempty = count > 0
    return float(np.sum(count[nonempty] * np.abs(mean_p[nonempty] - rate[nonempty])) / count.sum())


def all_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if not np.all(np.isfinite(p)) or p.min() < 0 or p.max() > 1:
        raise ValueError("predictions must be finite probabilities")
    return {
        "auc": auc(y, p),
        "acc": accuracy(y, p),
        "brier": brier(y, p),
        "ece": ece(y, p),
        "logloss": log_loss(y, p),
        "n": int(len(y)),
    }
