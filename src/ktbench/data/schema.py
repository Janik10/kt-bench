"""The canonical interaction table every dataset is converted into.

One row is one student answering one question. Multi-skill questions keep a
single row with a list of skills, so every model is scored once per question
(the pyKT "question-level" protocol). Models that work per skill expand the
list themselves and must never let one skill of a question see the answer to
another skill of the same question.

Columns
-------
user_id   int64         student id, dense 0..n_users-1
t         int64         0-based position of the answer within the student's sequence
item_id   int64         question id, dense 0..n_items-1
skills    list[int64]   skill ids, dense 0..n_skills-1, at least one per row
correct   int8          1 if the answer was right, else 0
timestamp float64       seconds since epoch, NaN when the dataset has no clock
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["user_id", "t", "item_id", "skills", "correct", "timestamp"]


@dataclass
class DatasetMeta:
    name: str
    n_users: int
    n_items: int
    n_skills: int
    n_interactions: int
    has_timestamps: bool

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> DatasetMeta:
        return cls(**json.loads(path.read_text()))


def validate(df: pd.DataFrame) -> None:
    """Raise if ``df`` breaks the canonical schema. Cheap enough to call on every load."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if not df["correct"].isin([0, 1]).all():
        raise ValueError("correct must be 0/1")
    if (df["skills"].map(len) == 0).any():
        raise ValueError("every row needs at least one skill")
    # Rows must be sorted by student, then time, with t = 0..n-1 per student.
    if not df["user_id"].is_monotonic_increasing:
        raise ValueError("rows must be sorted by user_id")
    expected_t = df.groupby("user_id", sort=False).cumcount().to_numpy()
    if not np.array_equal(df["t"].to_numpy(), expected_t):
        raise ValueError("t must be 0..n-1 within each student")


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, densify ids and assign ``t``. Input needs the canonical columns minus ``t``.

    Ties in time are broken by the incoming row order, so callers should sort
    by their own ordering key before calling.
    """
    df = df.reset_index(drop=True)
    df["_row"] = np.arange(len(df))
    df["user_id"] = pd.factorize(df["user_id"], sort=True)[0].astype(np.int64)
    df["item_id"] = pd.factorize(df["item_id"], sort=True)[0].astype(np.int64)
    df = df.sort_values(["user_id", "_row"], kind="stable").drop(columns="_row")
    df["t"] = df.groupby("user_id").cumcount().astype(np.int64)
    df["correct"] = df["correct"].astype(np.int8)
    df["timestamp"] = df["timestamp"].astype(np.float64)
    df = df[COLUMNS].reset_index(drop=True)
    validate(df)
    return df


def densify_skills(skill_lists: pd.Series) -> tuple[pd.Series, list[str]]:
    """Map raw skill labels to dense ints. Returns the new lists and the vocabulary."""
    vocab = sorted({s for skills in skill_lists for s in skills})
    index = {s: i for i, s in enumerate(vocab)}
    return skill_lists.map(lambda skills: [index[s] for s in skills]), vocab


def drop_short_students(df: pd.DataFrame, min_len: int) -> pd.DataFrame:
    counts = df.groupby("user_id")["user_id"].transform("size")
    return df[counts >= min_len]


def save(df: pd.DataFrame, meta: DatasetMeta, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "interactions.parquet", index=False)
    meta.save(out_dir / "meta.json")


def load(out_dir: Path) -> tuple[pd.DataFrame, DatasetMeta]:
    df = pd.read_parquet(out_dir / "interactions.parquet")
    df["skills"] = df["skills"].map(lambda a: [int(x) for x in a])
    validate(df)
    return df, DatasetMeta.load(out_dir / "meta.json")
