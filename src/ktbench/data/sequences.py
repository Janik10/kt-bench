"""Turn interaction tables into padded skill-level windows for sequence models.

Sequence models (DKT, SAKT) read one token per (question, skill). Every
window carries a ``ctx`` index per position: the window position where that
position's *question* starts. A model predicts position i from the state
built from tokens strictly before ``ctx[i]``, so the second skill of a
question never sees the answer to the first. This is the label-leakage guard
the pyKT paper warns about, enforced by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ktbench.models.base import explode_skills


@dataclass
class Window:
    skill: np.ndarray  # (L,) int64
    correct: np.ndarray  # (L,) int64
    ctx: np.ndarray  # (L,) int64, window index where this position's question starts
    target: np.ndarray  # (L,) int64, exploded-row index, -1 where not scored


def student_arrays(df: pd.DataFrame):
    """Yield (skill, correct, row, exploded_index) arrays per student, in time order."""
    ex = explode_skills(df)
    users = ex["user_id"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, users[1:] != users[:-1], True])
    skill, correct = ex["skill"].to_numpy(), ex["correct"].to_numpy().astype(np.int64)
    row = ex["row"].to_numpy()
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        yield skill[a:b], correct[a:b], row[a:b], np.arange(a, b)


def question_ctx(row: np.ndarray) -> np.ndarray:
    """For each position, the first position in ``row`` belonging to the same question."""
    idx = np.arange(len(row))
    start = np.ones(len(row), dtype=bool)
    start[1:] = row[1:] != row[:-1]
    return np.maximum.accumulate(np.where(start, idx, 0))


def train_windows(df: pd.DataFrame, max_len: int) -> list[Window]:
    """Non-overlapping windows of ``max_len`` skill tokens; every position is a target."""
    out = []
    for skill, correct, row, ex_idx in student_arrays(df):
        for s in range(0, len(skill), max_len):
            sl = slice(s, s + max_len)
            out.append(Window(skill[sl], correct[sl], question_ctx(row[sl]), ex_idx[sl]))
    return out


def eval_windows(df: pd.DataFrame, max_len: int | None, stride: int | None) -> list[Window]:
    """Windows that score every position exactly once.

    ``max_len=None`` gives one window per student (fine for an LSTM). With a
    length limit, windows start every ``stride`` tokens and each one scores
    only its last ``stride`` positions, so every scored position still sees
    at least ``max_len - stride`` tokens of context.
    """
    out = []
    for skill, correct, row, ex_idx in student_arrays(df):
        n = len(skill)
        if max_len is None or n <= max_len:
            out.append(Window(skill, correct, question_ctx(row), ex_idx))
            continue
        stride = stride or max_len
        s, scored_to = 0, 0
        while scored_to < n:
            e = min(s + max_len, n)
            target = ex_idx[s:e].copy()
            target[: scored_to - s] = -1
            out.append(Window(skill[s:e], correct[s:e], question_ctx(row[s:e]), target))
            scored_to = e
            s += stride
    return out


def collate(windows: list[Window]) -> dict[str, torch.Tensor]:
    L = max(len(w.skill) for w in windows)
    B = len(windows)
    skill = np.zeros((B, L), np.int64)
    correct = np.zeros((B, L), np.int64)
    ctx = np.zeros((B, L), np.int64)
    target = np.full((B, L), -1, np.int64)
    mask = np.zeros((B, L), bool)
    for i, w in enumerate(windows):
        n = len(w.skill)
        skill[i, :n], correct[i, :n], ctx[i, :n], target[i, :n] = w.skill, w.correct, w.ctx, w.target
        mask[i, :n] = True
    return {
        "skill": torch.from_numpy(skill),
        "correct": torch.from_numpy(correct),
        "ctx": torch.from_numpy(ctx),
        "target": torch.from_numpy(target),
        "mask": torch.from_numpy(mask),
    }
