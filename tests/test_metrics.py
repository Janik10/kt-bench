import numpy as np
import pytest

from ktbench import metrics


def test_calibrated_predictions_have_low_ece():
    p = np.repeat([0.05, 0.35, 0.65, 0.95], 1000)
    y = (np.random.default_rng(0).random(len(p)) < p).astype(float)
    assert metrics.ece(y, p) < 0.03


def test_overconfident_predictions_have_high_ece():
    y = np.array([0, 1] * 500, dtype=float)
    p = np.full(len(y), 0.99)
    assert metrics.ece(y, p) == pytest.approx(0.49)


def test_basic_values():
    y = np.array([0, 0, 1, 1], dtype=float)
    p = np.array([0.1, 0.4, 0.35, 0.8])
    assert metrics.auc(y, p) == pytest.approx(0.75)
    assert metrics.accuracy(y, p) == pytest.approx(0.75)
    assert metrics.brier(y, p) == pytest.approx(np.mean((p - y) ** 2))


def test_rejects_non_probabilities():
    with pytest.raises(ValueError):
        metrics.all_metrics(np.array([0, 1]), np.array([0.5, 1.2]))
