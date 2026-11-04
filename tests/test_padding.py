"""Padded positions must never count toward the loss or the metrics."""

import numpy as np
import pytest
import torch

from ktbench import models
from ktbench.data import sequences
from tests.conftest import TINY


@pytest.mark.parametrize("name", ["dkt", "sakt"])
def test_prediction_independent_of_batch_padding(dataset, split, name):
    _, meta = dataset
    model = models.build(name, meta.n_skills, meta.n_items, seed=0, **TINY[name]).fit(split.train)
    net = model.net.eval()
    windows = sequences.eval_windows(split.test, model.eval_max_len, None)
    short = min(windows, key=lambda w: len(w.skill))
    long = max(windows, key=lambda w: len(w.skill))
    assert len(short.skill) < len(long.skill)
    with torch.no_grad():
        alone = sequences.collate([short])
        padded = sequences.collate([short, long])
        a = net(alone["skill"], alone["correct"], alone["ctx"])[0]
        b = net(padded["skill"], padded["correct"], padded["ctx"])[0, : len(short.skill)]
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_collate_masks_padding():
    w1 = sequences.Window(np.array([1, 2]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1]))
    w2 = sequences.Window(np.array([3, 4, 5]), np.array([1, 1, 0]), np.array([0, 1, 1]), np.array([2, 3, 4]))
    b = sequences.collate([w1, w2])
    assert b["mask"].tolist() == [[True, True, False], [True, True, True]]
    assert b["target"][0, 2].item() == -1


@pytest.mark.parametrize("max_len,stride", [(None, None), (10, 5), (10, 10), (7, 3)])
def test_eval_windows_score_every_position_once(split, max_len, stride):
    windows = sequences.eval_windows(split.test, max_len, stride)
    targets = np.concatenate([w.target[w.target >= 0] for w in windows])
    n = int(split.test["skills"].map(len).sum())
    assert np.array_equal(np.sort(targets), np.arange(n))


def test_padding_excluded_from_training_loss(dataset, split):
    """Flipping the labels on padded positions must not change the masked loss."""
    _, meta = dataset
    model = models.build("dkt", meta.n_skills, meta.n_items, seed=0, **TINY["dkt"]).fit(split.train)
    model.net.eval()  # no dropout, so the two passes are comparable
    batch = sequences.collate(sequences.train_windows(split.train, 200)[:4])
    m = batch["mask"]
    assert (~m).any()
    noisy = batch["correct"].clone()
    noisy[~m] = 1 - noisy[~m]
    loss = torch.nn.BCEWithLogitsLoss(reduction="sum")
    with torch.no_grad():
        clean = loss(model.net(batch["skill"], batch["correct"], batch["ctx"])[m], batch["correct"][m].float())
        dirty = loss(model.net(batch["skill"], noisy, batch["ctx"])[m], noisy[m].float())
    torch.testing.assert_close(clean, dirty)
