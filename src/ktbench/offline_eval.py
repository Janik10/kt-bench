"""Offline evaluation of mastery models on a small deployment's own data.

    python -m ktbench.offline_eval /secure/path/graded_answers.csv

Run this where the data lives. The data must never leave that machine: no
Colab, no copying into this repo (``*.csv`` is git-ignored), and only the
aggregate numbers this prints go in any write-up. Check that what students
and instructors agreed to covers this use before running it.

Input: one row per graded answer with columns
``user_id, skill_id, correct, timestamp`` (``correct`` is 0/1, ``timestamp``
is ISO 8601 or epoch seconds). Choosing which answers count as graded is done
when the export is made, not here.

Protocol (deployments are small, tens of students, so this is not a
student-level split): each student's first 70% of answers by time are
training data and the last 30% are the test set. Models see the whole
sequence but are fitted on the first 70% only and scored on the last 30%.
Hyperparameters are picked by the same 70/30 cut inside the training part.
Confidence intervals resample students (1,000 bootstrap draws), and the
difference against the current production model is paired.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from ktbench import metrics, models
from ktbench.data import schema

CURRENT = ("beta", {"prior": "fixed", "prior_alpha": 1.0, "prior_beta": 1.0,
                          "half_life": 23.1, "decay_unit": "days"})
CANDIDATES = {
    "skill_avg": ("skill_avg", {}),
    "pfa": ("pfa", {"C": [0.01, 0.1, 1.0], "count_transform": ["raw", "log1p"]}),
    "beta_tuned": ("beta", {"prior": ["fixed", "skill"], "prior_strength": [1.0, 2.0, 4.0],
                                        "half_life": [None, 3, 10, 23.1, 60], "decay_unit": ["days"]}),
    "bkt": ("bkt", {"num_fits": [3]}),
}


def load_export(path: Path) -> tuple[pd.DataFrame, schema.DatasetMeta]:
    raw = pd.read_csv(path, dtype={"skill_id": str})
    ts = raw["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        raw["ts"] = ts.astype(float)
    else:
        raw["ts"] = (pd.to_datetime(ts, utc=True) - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    raw = raw.sort_values(["user_id", "ts"], kind="stable")
    df = pd.DataFrame({
        "user_id": raw["user_id"].to_numpy(),
        "item_id": raw["skill_id"].to_numpy(),
        "skills": raw["skill_id"].map(lambda t: [t]).to_numpy(),
        "correct": raw["correct"].astype(np.int8).to_numpy(),
        "timestamp": raw["ts"].to_numpy(),
    })
    df["skills"], vocab = schema.densify_skills(df["skills"])
    df = schema.finalize(df)
    meta = schema.DatasetMeta("offline", int(df["user_id"].nunique()), int(df["item_id"].nunique()),
                              len(vocab), len(df), True)
    return df, meta


def temporal_mask(df: pd.DataFrame, frac: float = 0.7) -> np.ndarray:
    """True for each student's first ``frac`` of answers (at least one answer on each side)."""
    n = df.groupby("user_id")["t"].transform("size").to_numpy()
    cut = np.clip(np.floor(frac * n), 1, n - 1)
    return df["t"].to_numpy() < cut


def _fit_predict(name: str, params: dict, df: pd.DataFrame, train_mask: np.ndarray, meta) -> np.ndarray:
    model = models.build(name, meta.n_skills, meta.n_items, seed=0, **params)
    model.fit(df[train_mask].reset_index(drop=True))
    return model.predict(df)  # causal, so test rows see the training answers as history


def select(name: str, space: dict, df: pd.DataFrame, train_mask: np.ndarray, meta) -> dict:
    """Pick params by a 70/30 time cut inside the training part only."""
    grid = [dict(zip(space, v, strict=True)) for v in itertools.product(*space.values())] if space else [{}]
    if len(grid) == 1:
        return grid[0]
    inner = df[train_mask].reset_index(drop=True)
    inner_train = temporal_mask(inner)
    y = inner["correct"].to_numpy()
    scores = [metrics.auc(y[~inner_train], _fit_predict(name, g, inner, inner_train, meta)[~inner_train])
              for g in grid]
    return grid[int(np.nanargmax(scores))]


def bootstrap(users: np.ndarray, y: np.ndarray, preds: dict[str, np.ndarray], n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(users)
    rows_of = {u: np.flatnonzero(users == u) for u in uniq}
    draws = {k: {"auc": [], "brier": []} for k in preds}
    for _ in range(n):
        idx = np.concatenate([rows_of[u] for u in rng.choice(uniq, size=len(uniq))])
        for k, p in preds.items():
            draws[k]["auc"].append(metrics.auc(y[idx], p[idx]))
            draws[k]["brier"].append(metrics.brier(y[idx], p[idx]))
    return {k: {m: np.asarray(v) for m, v in d.items()} for k, d in draws.items()}


def evaluate(df: pd.DataFrame, meta, n_boot: int = 1000) -> pd.DataFrame:
    train_mask = temporal_mask(df)
    test = ~train_mask
    y, users = df["correct"].to_numpy()[test], df["user_id"].to_numpy()[test]
    preds, chosen = {"current": _fit_predict(*CURRENT, df, train_mask, meta)[test]}, {"current": CURRENT[1]}
    for label, (name, space) in CANDIDATES.items():
        params = select(name, space, df, train_mask, meta)
        chosen[label] = params
        preds[label] = _fit_predict(name, params, df, train_mask, meta)[test]
    boot = bootstrap(users, y, preds, n_boot)
    rows = []
    for k, p in preds.items():
        d_auc = boot[k]["auc"] - boot["current"]["auc"]
        rows.append({
            "model": k,
            "auc": metrics.auc(y, p),
            "auc_95ci": tuple(np.nanpercentile(boot[k]["auc"], [2.5, 97.5]).round(4)),
            "brier": metrics.brier(y, p),
            "brier_95ci": tuple(np.percentile(boot[k]["brier"], [2.5, 97.5]).round(4)),
            "auc_minus_current_95ci": tuple(np.nanpercentile(d_auc, [2.5, 97.5]).round(4)),
            "params": chosen[k],
        })
    out = pd.DataFrame(rows).sort_values("auc", ascending=False)
    out.attrs.update(n_test=int(test.sum()), n_train=int(train_mask.sum()), n_students=int(len(np.unique(users))))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", type=Path)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args(argv)
    df, meta = load_export(args.export)
    table = evaluate(df, meta, args.boot)
    a = table.attrs
    print(f"{a['n_students']} students, {a['n_train']} train / {a['n_test']} test answers")
    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
