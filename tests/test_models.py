import numpy as np
import pandas as pd
import pytest

from ktbench import models
from ktbench.evaluate import evaluate
from tests.conftest import TINY


@pytest.mark.parametrize("name", list(models.REGISTRY))
def test_every_model_fits_and_beats_chance(dataset, split, name):
    _, meta = dataset
    params = TINY.get(name, {})
    if name in ("dkt", "sakt"):
        params = {**params, "max_epochs": 15, "lr": 0.01}  # tiny data needs a few epochs to learn
    m = models.build(name, meta.n_skills, meta.n_items, seed=0, **params)
    m.fit(split.train, split.valid)
    result, p = evaluate(m, split.test)
    assert p.shape == (len(split.test),)
    assert np.all((p >= 0) & (p <= 1))
    assert result["auc"] > 0.55


def _one_skill(correct, timestamp=None):
    n = len(correct)
    return pd.DataFrame(
        {"user_id": 0, "t": range(n), "item_id": 0, "skills": [[0]] * n, "correct": correct,
         "timestamp": timestamp if timestamp is not None else [float("nan")] * n}
    )


def test_beta_production_matches_hand_computed_posterior():
    df = _one_skill([1, 1, 0, 1])
    m = models.build("beta", n_skills=1, n_items=1).fit(df)
    # Beta(1,1), then +1 alpha, +1 alpha, +1 beta: the posterior mean before each answer.
    np.testing.assert_allclose(m.predict(df), [1 / 2, 2 / 3, 3 / 4, 3 / 5])


def test_beta_decay_halves_old_evidence():
    df = _one_skill([1, 1], timestamp=[0.0, 10 * 86400.0])
    m = models.build("beta", n_skills=1, n_items=1, half_life=10, decay_unit="days").fit(df)
    # Ten days later, with a ten-day half-life, the first success counts 0.5.
    np.testing.assert_allclose(m.predict(df)[1], 1.5 / 2.5)


def test_pfa_counts_only_earlier_answers():
    from ktbench.models.base import explode_skills, prior_counts

    df = _one_skill([1, 0, 1, 1])
    succ, fail = prior_counts(explode_skills(df))
    assert succ.tolist() == [0, 1, 1, 2]
    assert fail.tolist() == [0, 0, 1, 1]
