"""Student-level splits: 20% of students held out as test, the rest in 5 folds.

Splitting by student (never by row) measures how well a model predicts a
student it has never seen, which is the situation a deployed tutor is in. The split is a
pure function of the student ids and the seed, and is saved next to the data
so every model and every run sees exactly the same students.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TEST = -1


def assign(user_ids: np.ndarray, test_frac: float = 0.2, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Return a frame ``user_id -> fold`` where fold is ``TEST`` (-1) or 0..n_folds-1."""
    users = np.unique(user_ids)
    rng = np.random.default_rng(seed)
    users = users[rng.permutation(len(users))]
    n_test = int(round(test_frac * len(users)))
    folds = np.empty(len(users), dtype=np.int64)
    folds[:n_test] = TEST
    folds[n_test:] = np.arange(len(users) - n_test) % n_folds
    return pd.DataFrame({"user_id": users, "fold": folds}).sort_values("user_id").reset_index(drop=True)


@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def make_split(df: pd.DataFrame, folds: pd.DataFrame, valid_fold: int) -> Split:
    """Train on every non-test fold except ``valid_fold``, validate on ``valid_fold``."""
    fold_of = df["user_id"].map(folds.set_index("user_id")["fold"])
    if fold_of.isna().any():
        raise ValueError("some students have no fold assignment")
    return Split(
        train=df[(fold_of != TEST) & (fold_of != valid_fold)].reset_index(drop=True),
        valid=df[fold_of == valid_fold].reset_index(drop=True),
        test=df[fold_of == TEST].reset_index(drop=True),
    )


DEFAULT_SEED = 42


def load_or_create(df: pd.DataFrame, data_dir: Path, seed: int = DEFAULT_SEED, **kwargs) -> pd.DataFrame:
    """The benchmark split (seed 42) lives in ``folds.parquet``; other seeds, used only to
    measure how much the test-set draw moves the numbers, in ``folds_seed<s>.parquet``."""
    path = data_dir / ("folds.parquet" if seed == DEFAULT_SEED else f"folds_seed{seed}.parquet")
    if path.exists():
        folds = pd.read_parquet(path)
        if set(folds["user_id"]) != set(df["user_id"].unique()):
            raise ValueError(f"{path} does not match the dataset; delete it to regenerate")
        return folds
    folds = assign(df["user_id"].to_numpy(), seed=seed, **kwargs)
    folds.to_parquet(path, index=False)
    return folds
