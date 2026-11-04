import pytest

from ktbench.data import splits, synthetic


@pytest.fixture(scope="session")
def dataset():
    return synthetic.generate(n_users=120, max_len=40, seed=0)


@pytest.fixture(scope="session")
def split(dataset):
    df, _ = dataset
    folds = splits.assign(df["user_id"].to_numpy(), seed=42)
    return splits.make_split(df, folds, valid_fold=0)


FAST_NEURAL = {"max_epochs": 2, "device": "cpu", "verbose": False, "batch_size": 32}
TINY = {
    "dkt": {"emb_size": 16, "hidden_size": 16, **FAST_NEURAL},
    "sakt": {"emb_size": 16, "n_heads": 2, "max_len": 16, **FAST_NEURAL},
    "bkt": {"num_fits": 1},
}
