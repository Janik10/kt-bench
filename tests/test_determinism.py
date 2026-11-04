import pytest

from ktbench import models
from ktbench.evaluate import evaluate
from tests.conftest import TINY


@pytest.mark.parametrize("name", ["dkt", "sakt", "pfa", "bkt"])
def test_same_seed_same_auc(dataset, split, name):
    _, meta = dataset
    aucs = []
    for _ in range(2):
        m = models.build(name, meta.n_skills, meta.n_items, seed=3, **TINY.get(name, {}))
        m.fit(split.train, split.valid)
        aucs.append(round(evaluate(m, split.test)[0]["auc"], 4))
    assert aucs[0] == aucs[1]
