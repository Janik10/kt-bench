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


def test_bkt_chunking_does_not_change_the_fit(dataset, split, monkeypatch):
    from ktbench.models import bkt

    _, meta = dataset
    one = models.build("bkt", meta.n_skills, meta.n_items, seed=0, num_fits=1).fit(split.train)
    monkeypatch.setattr(bkt, "MAX_CELLS", 50)  # force many small chunks per skill
    many = models.build("bkt", meta.n_skills, meta.n_items, seed=0, num_fits=1).fit(split.train)
    for skill, params in one.skill_params.items():
        np.testing.assert_allclose(many.skill_params[skill], params, rtol=1e-6)
    np.testing.assert_allclose(many.predict(split.test), one.predict(split.test), rtol=1e-6)


def test_bkt_forward_stays_a_probability_with_tiny_guess_and_slip():
    """Regression for the EdNet BKT crash. With guess and slip ~1e-5 (fit on some EdNet skills),
    computing P(wrong) as 1 - P(correct) loses ~12 digits, the posterior P(known) after a wrong
    answer rounds above 1 (by 2e-13 at step 4 here), and the error compounds until P(correct) > 1."""
    from ktbench.models import bkt

    answers = (  # the shortest failing prefix found by a random search
        "1110101111111111111111111111111010111111111111110111111111111111"
        "0111110111111111111111001011111111111101111111111111110110001"
    )
    obs = np.array([[int(c) for c in answers]], dtype=float)
    mask = np.ones_like(obs, dtype=bool)
    p_known, p_correct, loglik = bkt.forward(
        obs, mask, prior=0.8326691199192928, learn=0.6055340626775021,
        guess=9.370144637718133e-06, slip=1.793683401796543e-05,
    )
    assert np.all((p_known >= 0) & (p_known <= 1))
    assert np.all((p_correct >= 0) & (p_correct <= 1))
    assert np.isfinite(loglik)
