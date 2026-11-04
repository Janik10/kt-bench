"""ASSISTments 2009-2010 Skill Builder, corrected and collapsed version.

Source: https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010
The corrected file fixes the duplicated rows of the original release and
holds one row per student-problem, with multi-skill problems written as
``skill1_skill2``.

Cleaning follows pyKT (``assist2009_preprocess.py``) so the numbers are
comparable with the pyKT paper: drop rows missing user, problem, skill,
correct or order_id; keep only 0/1 responses; order each student's answers by
``order_id``; drop students with fewer than 3 answers. The dataset has no
wall-clock timestamps, so ``timestamp`` is NaN.

The raw file matches pyKT's Table 1 exactly (4,217 students, 346,860 rows,
26,688 questions, 123 skills). After cleaning: 3,852 students, 282,619
interactions (338k skill-level rows), 17,737 questions, 123 skills. Most of
the drop is the 63,755 rows with no skill tag.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ktbench.data import schema

NAME = "assist2009"
DRIVE_ID = "1NNXHFRxcArrU0ZJSb9BIL56vmUt5FhlE"
URL = f"https://drive.usercontent.google.com/download?id={DRIVE_ID}&export=download&confirm=t"
RAW_FILE = "skill_builder_data_corrected_collapsed.csv"
MIN_SEQ_LEN = 3


def download(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / RAW_FILE
    if path.exists():
        return path
    print(f"downloading {RAW_FILE} ...")
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = path.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    with open(tmp, "rb") as f:
        head = f.read(64)
    if b"order_id" not in head:
        tmp.unlink()
        raise RuntimeError(
            "download did not return the CSV (Google Drive may need a manual download). "
            f"Get it from the ASSISTments page and save it as {path}"
        )
    tmp.rename(path)
    return path


def preprocess(raw_path: Path) -> tuple[pd.DataFrame, schema.DatasetMeta]:
    cols = ["order_id", "user_id", "problem_id", "skill_id", "correct"]
    raw = pd.read_csv(raw_path, usecols=cols, dtype=str, encoding="latin-1")
    raw["_row"] = np.arange(len(raw))
    raw = raw.dropna(subset=cols)
    raw = raw[raw["correct"].isin(["0", "1"])]

    raw["order_id"] = raw["order_id"].astype(np.int64)
    raw = raw.sort_values(["user_id", "order_id", "_row"], kind="stable")

    df = pd.DataFrame(
        {
            "user_id": raw["user_id"].astype(np.int64).to_numpy(),
            "item_id": raw["problem_id"].astype(np.int64).to_numpy(),
            "skills": raw["skill_id"].str.split("_").to_numpy(),
            "correct": raw["correct"].astype(np.int8).to_numpy(),
            "timestamp": np.nan,
        }
    )
    df = schema.drop_short_students(df, MIN_SEQ_LEN)
    df["skills"], _vocab = schema.densify_skills(df["skills"])
    df = schema.finalize(df)

    meta = schema.DatasetMeta(
        name=NAME,
        n_users=int(df["user_id"].nunique()),
        n_items=int(df["item_id"].nunique()),
        n_skills=len(_vocab),
        n_interactions=len(df),
        has_timestamps=False,
    )
    return df, meta


def build(data_dir: Path) -> Path:
    raw_path = download(data_dir / "raw" / NAME)
    df, meta = preprocess(raw_path)
    out = data_dir / "processed" / NAME
    schema.save(df, meta, out)
    print(
        f"{NAME}: {meta.n_users:,} students, {meta.n_interactions:,} interactions, "
        f"{meta.n_items:,} questions, {meta.n_skills} skills -> {out}"
    )
    return out
