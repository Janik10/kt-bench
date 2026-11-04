"""Model 0: each skill's overall correct rate. A sanity floor every model must beat."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ktbench.models.base import KTModel, explode_skills, fuse


class SkillAverage(KTModel):
    name = "skill_avg"

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> SkillAverage:
        ex = explode_skills(train)
        self.global_rate = float(train["correct"].mean())
        # Light smoothing toward the global rate so rare skills don't get 0 or 1.
        k = float(self.params.get("smoothing", 1.0))
        stats = ex.groupby("skill")["correct"].agg(["sum", "count"])
        self.rate = np.full(self.n_skills, self.global_rate)
        self.rate[stats.index] = (stats["sum"] + k * self.global_rate) / (stats["count"] + k)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        ex = explode_skills(df)
        return fuse(ex["row"].to_numpy(), self.rate[ex["skill"].to_numpy()], len(df))
