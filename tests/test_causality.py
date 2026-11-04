"""Changing a student's answers from step t on must not move any prediction up to step t.

That includes row t's own answer, so a model that peeks at the label it is
predicting fails, and so does a multi-skill question whose second skill sees
the first skill's answer.
"""

import numpy as np
import pytest
import torch

from ktbench import models
from ktbench.data import sequences
from tests.conftest import TINY

ALL = ["skill_avg", "pfa", "beta", "bkt", "dkt", "sakt"]


@pytest.fixture(scope="module")
def fitted(dataset, split):
    _, meta = dataset
    out = {}
    for name in ALL:
        params = {"half_life": 5} if name == "beta" else TINY.get(name, {})
        model = models.build(name, meta.n_skills, meta.n_items, seed=0, **params)
        out[name] = model.fit(split.train, split.valid)
    return out


@pytest.mark.parametrize("name", ALL)
def test_future_answers_do_not_change_past_predictions(fitted, split, name):
    model = fitted[name]
    test = split.test
    p0 = model.predict(test)
    users = test["user_id"].to_numpy()
    rng = np.random.default_rng(0)
    for u in rng.choice(np.unique(users), size=5, replace=False):
        rows = np.flatnonzero(users == u)
        cut = rows[len(rows) // 2]
        changed = test.copy()
        flip = rows[rows >= cut]
        changed.loc[flip, "correct"] = 1 - changed.loc[flip, "correct"]
        p1 = model.predict(changed)
        upto = rows[rows <= cut]
        np.testing.assert_allclose(p1[upto], p0[upto], atol=1e-6, err_msg=f"{name} saw the future")
        np.testing.assert_allclose(p1[users != u], p0[users != u], atol=1e-6, err_msg=f"{name} mixed students")


@pytest.mark.parametrize("name", ["dkt", "sakt"])
def test_multi_skill_question_does_not_see_its_own_answer(fitted, split, name):
    """The leakage the pyKT paper warns about: skill 2 of a question predicted after seeing skill 1's answer."""
    L = 16  # fits SAKT's tiny position table; truncating is harmless for a causal model
    test = split.test
    net = fitted[name].net.eval()
    checked = 0
    for _, one in test.groupby("user_id"):
        one = one.reset_index(drop=True)
        question_of_token = np.repeat(np.arange(len(one)), one["skills"].map(len).to_numpy())[:L]
        for q in np.flatnonzero(one["skills"].map(len).to_numpy() > 1):
            pos = np.flatnonzero(question_of_token == q)
            if q == 0 or len(pos) < 2:
                continue
            batch = {k: v[:, :L] for k, v in sequences.collate(sequences.eval_windows(one, None, None)).items()}
            flipped = batch["correct"].clone()
            flipped[0, pos[0]] = 1 - flipped[0, pos[0]]  # change the answer to the question's first skill
            with torch.no_grad():
                base = net(batch["skill"], batch["correct"], batch["ctx"])
                after = net(batch["skill"], flipped, batch["ctx"])
            torch.testing.assert_close(after[0, pos], base[0, pos])
            checked += 1
    assert checked >= 3, "the fixture should contain multi-skill questions"
