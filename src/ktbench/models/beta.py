"""Model 3: a Bayesian Beta-posterior baseline.

Per student and skill, mastery is the mean of a Beta(alpha, beta) posterior.
It starts from a prior and every answer adds ``weight * recency`` to alpha if
right and to beta if wrong, where ``recency = 2 ** (-age / half_life)``
discounts old evidence. The posterior mean alpha / (alpha + beta) before an
answer is the predicted chance of getting it right, and a multi-skill
question averages its skills. It is a counting model with forgetting: no
hidden state, no learning-rate parameter, and almost nothing to fit.

Parameters
----------
prior        "fixed" uses Beta(prior_alpha, prior_beta) for every skill (default 1, 1).
             "skill" centres each skill's prior on its training-set correct rate
             with total pseudo-count ``prior_strength``.
half_life    evidence half-life; null means no decay.
decay_unit   "days" (needs timestamps) or "steps" (the student's answer count,
             for datasets without a clock such as ASSISTments 2009).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ktbench.models.base import KTModel, explode_skills, fuse


class BetaPosterior(KTModel):
    name = "beta"

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> BetaPosterior:
        mode = self.params.get("prior", "fixed")
        if mode == "fixed":
            self.a0 = np.full(self.n_skills, float(self.params.get("prior_alpha", 1.0)))
            self.b0 = np.full(self.n_skills, float(self.params.get("prior_beta", 1.0)))
        elif mode == "skill":
            ex = explode_skills(train)
            g = float(train["correct"].mean())
            stats = ex.groupby("skill")["correct"].agg(["sum", "count"])
            mean = np.full(self.n_skills, g)
            mean[stats.index] = (stats["sum"] + g) / (stats["count"] + 1.0)
            s = float(self.params.get("prior_strength", 2.0))
            self.a0, self.b0 = s * mean, s * (1.0 - mean)
        else:
            raise ValueError(f"unknown prior mode {mode!r}")
        return self

    def _clock(self, ex: pd.DataFrame) -> np.ndarray:
        unit = self.params.get("decay_unit", "steps")
        if unit == "steps":
            return ex["t"].to_numpy(np.float64)
        if unit == "days":
            ts = ex["timestamp"].to_numpy(np.float64)
            if np.isnan(ts).any():
                raise ValueError("decay_unit='days' needs timestamps")
            return ts / 86400.0
        raise ValueError(f"unknown decay_unit {unit!r}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        ex = explode_skills(df)
        half_life = self.params.get("half_life")
        lam = 0.0 if not half_life else math.log(2) / float(half_life)
        clock = self._clock(ex)
        skill = ex["skill"].to_numpy()
        correct = ex["correct"].to_numpy(np.float64)
        w = float(self.params.get("weight", 1.0))

        # Walk each (student, skill) history once, carrying decayed evidence.
        order = np.lexsort((ex["t"].to_numpy(), skill, ex["user_id"].to_numpy()))
        key = (ex["user_id"].to_numpy()[order] * (self.n_skills + 1) + skill[order]).tolist()
        # Plain Python lists: indexing numpy scalars in a loop is ~10x slower.
        a0, b0 = self.a0.tolist(), self.b0.tolist()
        clock_l, skill_l, correct_l = clock.tolist(), skill.tolist(), correct.tolist()
        p = [0.0] * len(ex)
        a = b = last = 0.0
        prev_key = -1
        for j, i in enumerate(order.tolist()):
            if key[j] != prev_key:
                a = b = 0.0
                prev_key = key[j]
            elif lam:
                decay = math.exp(-lam * (clock_l[i] - last))
                a *= decay
                b *= decay
            k = skill_l[i]
            p[i] = (a0[k] + a) / (a0[k] + b0[k] + a + b)
            a += w * correct_l[i]
            b += w * (1.0 - correct_l[i])
            last = clock_l[i]
        return fuse(ex["row"].to_numpy(), np.asarray(p), len(df))
