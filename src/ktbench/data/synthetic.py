"""A small synthetic dataset with real learning in it, for tests and CI.

Each student has a hidden known/unknown state per skill that follows BKT
dynamics, so sequence models have signal to find while the whole thing still
trains in seconds on a CPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ktbench.data import schema


def generate(
    n_users: int = 200,
    n_skills: int = 8,
    items_per_skill: int = 6,
    min_len: int = 5,
    max_len: int = 60,
    multi_skill_frac: float = 0.2,
    seed: int = 0,
) -> tuple[pd.DataFrame, schema.DatasetMeta]:
    rng = np.random.default_rng(seed)
    p_init = rng.uniform(0.1, 0.5, n_skills)
    p_learn = rng.uniform(0.05, 0.3, n_skills)
    p_guess = rng.uniform(0.1, 0.3, n_skills)
    p_slip = rng.uniform(0.05, 0.15, n_skills)

    n_items = n_skills * items_per_skill
    item_skills = []
    for i in range(n_items):
        first = i // items_per_skill
        if rng.random() < multi_skill_frac:
            second = int(rng.integers(n_skills - 1))
            second += second >= first
            item_skills.append(sorted([first, second]))
        else:
            item_skills.append([first])

    rows = []
    for u in range(n_users):
        known = rng.random(n_skills) < p_init
        ability = rng.normal(0, 0.5)
        t0 = 1.7e9 + rng.uniform(0, 1e7)
        for t in range(int(rng.integers(min_len, max_len + 1))):
            item = int(rng.integers(n_items))
            skills = item_skills[item]
            p_skill = np.where(known[skills], 1 - p_slip[skills], p_guess[skills])
            p = float(np.clip(np.prod(p_skill) + 0.05 * ability, 0.01, 0.99))
            correct = int(rng.random() < p)
            rows.append((u, item, skills, correct, t0 + 3600.0 * t))
            learned = rng.random(len(skills)) < p_learn[skills]
            known[skills] |= learned
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "skills", "correct", "timestamp"])
    df = schema.finalize(df)
    meta = schema.DatasetMeta(
        name="synthetic",
        n_users=n_users,
        n_items=int(df["item_id"].nunique()),
        n_skills=n_skills,
        n_interactions=len(df),
        has_timestamps=True,
    )
    return df, meta
