"""The one interface every model implements.

``fit(train, valid)`` learns from canonical interaction tables (see
``ktbench.data.schema``). ``predict(df)`` returns, for every row of ``df``,
the probability that the answer is correct, using only that student's
*earlier* rows. The evaluator feeds whole test sequences through ``predict``,
so a model that peeks at row t's own answer, or at any later row, fails the
causality tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class KTModel(ABC):
    name: str = "base"

    def __init__(self, n_skills: int, n_items: int, **params):
        self.n_skills = n_skills
        self.n_items = n_items
        self.params = params

    @abstractmethod
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> KTModel: ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray: ...


def explode_skills(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (question, skill). ``row`` points back at the question row in ``df``.

    The skills of one question share the same ``row``; per-skill models must
    treat them as simultaneous, never as one after the other.
    """
    lengths = df["skills"].map(len).to_numpy()
    out = pd.DataFrame(
        {
            "row": np.repeat(np.arange(len(df)), lengths),
            "user_id": np.repeat(df["user_id"].to_numpy(), lengths),
            "t": np.repeat(df["t"].to_numpy(), lengths),
            "skill": np.concatenate(df["skills"].to_numpy()).astype(np.int64)
            if len(df)
            else np.empty(0, np.int64),
            "correct": np.repeat(df["correct"].to_numpy(), lengths),
            "timestamp": np.repeat(df["timestamp"].to_numpy(), lengths),
        }
    )
    return out


def fuse(row: np.ndarray, p: np.ndarray, n_rows: int) -> np.ndarray:
    """Average per-skill predictions back to one prediction per question ("late fusion, mean")."""
    total = np.bincount(row, weights=p, minlength=n_rows)
    count = np.bincount(row, minlength=n_rows)
    return total / np.maximum(count, 1)


def prior_counts(ex: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """For each exploded row: the student's earlier successes and failures on that skill.

    Within one question a skill appears once, so "earlier rows in the same
    (student, skill) group" is exactly "earlier questions".
    """
    g = ex.groupby(["user_id", "skill"], sort=False)["correct"]
    succ = g.cumsum().to_numpy() - ex["correct"].to_numpy()
    attempts = g.cumcount().to_numpy()
    return succ.astype(np.float64), (attempts - succ).astype(np.float64)
