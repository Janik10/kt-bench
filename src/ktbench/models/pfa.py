"""Model 2: Performance Factors Analysis (Pavlik, Cen & Koedinger, 2009).

logit P(correct) = sum over the question's skills k of
    beta_k + gamma_k * successes_k + rho_k * failures_k
where successes_k / failures_k are the student's earlier right / wrong answers
on skill k. Fit as one logistic regression over a sparse design matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from ktbench.models.base import KTModel, explode_skills, prior_counts


class PFA(KTModel):
    name = "pfa"

    def _features(self, df: pd.DataFrame) -> sparse.csr_matrix:
        ex = explode_skills(df)
        succ, fail = prior_counts(ex)
        if self.params.get("count_transform", "raw") == "log1p":
            succ, fail = np.log1p(succ), np.log1p(fail)
        row, skill, S = ex["row"].to_numpy(), ex["skill"].to_numpy(), self.n_skills
        return sparse.coo_matrix(
            (
                np.concatenate([np.ones(len(ex)), succ, fail]),
                (np.tile(row, 3), np.concatenate([skill, S + skill, 2 * S + skill])),
            ),
            shape=(len(df), 3 * S),
        ).tocsr()

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> PFA:
        self.clf = LogisticRegression(C=float(self.params.get("C", 1.0)), max_iter=1000)
        self.clf.fit(self._features(train), train["correct"].to_numpy())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self._features(df))[:, 1]
