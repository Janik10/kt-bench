import numpy as np

from ktbench.data import splits


def test_no_student_leakage(dataset):
    df, _ = dataset
    folds = splits.assign(df["user_id"].to_numpy(), seed=42)
    for k in range(5):
        s = splits.make_split(df, folds, valid_fold=k)
        train, valid, test = (set(x["user_id"]) for x in (s.train, s.valid, s.test))
        assert not train & test
        assert not valid & test
        assert not train & valid
        assert train | valid | test == set(df["user_id"])


def test_test_set_is_fixed_and_twenty_percent(dataset):
    df, _ = dataset
    folds = splits.assign(df["user_id"].to_numpy(), seed=42)
    tests = {frozenset(splits.make_split(df, folds, k).test["user_id"]) for k in range(5)}
    assert len(tests) == 1
    assert (folds["fold"] == splits.TEST).sum() == round(0.2 * df["user_id"].nunique())


def test_split_is_deterministic(dataset):
    df, _ = dataset
    a = splits.assign(df["user_id"].to_numpy(), seed=42)
    b = splits.assign(df["user_id"].to_numpy(), seed=42)
    c = splits.assign(df["user_id"].to_numpy(), seed=7)
    assert a.equals(b)
    assert not np.array_equal(a["fold"], c["fold"])


def test_folds_are_balanced(dataset):
    df, _ = dataset
    folds = splits.assign(df["user_id"].to_numpy(), seed=42)
    sizes = folds[folds["fold"] != splits.TEST]["fold"].value_counts()
    assert sizes.max() - sizes.min() <= 1
