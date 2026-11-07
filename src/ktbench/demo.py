"""Watch a model's predicted mastery change as a student answers.

    python -m ktbench.demo runs/assist2009/dkt/model_run0.pkl --history "10:1 10:0 10:1 10:1 45:0 45:1" --probe 10,45

``--history`` is a list of ``skill:correct`` answers (skills joined with ``+``
for multi-skill questions). After each answer, the model is asked for the
chance of a right answer on each probe skill. Every probe is scored as its own
one-off continuation of the history, so probes never see each other, and
this works for any model behind the shared interface.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ktbench.train import load_model


def parse_history(text: str) -> list[tuple[list[int], int]]:
    out = []
    for tok in text.split():
        skills, correct = tok.split(":")
        out.append(([int(s) for s in skills.split("+")], int(correct)))
    return out


def trajectory(model, history: list[tuple[list[int], int]], probes: list[int]) -> np.ndarray:
    """P(correct) on each probe skill before any answer and after each one: shape (len(history)+1, len(probes))."""
    rows = []
    user = 0
    for k in range(len(history) + 1):
        for skill in probes:
            for t, (skills, correct) in enumerate(history[:k]):
                rows.append((user, t, 0, skills, correct))
            rows.append((user, k, 0, [skill], 0))  # the probe; its label is never read
            user += 1
    df = pd.DataFrame(rows, columns=["user_id", "t", "item_id", "skills", "correct"])
    df["timestamp"] = np.nan
    df["correct"] = df["correct"].astype(np.int8)
    p = model.predict(df)
    last = df.groupby("user_id").tail(1).index.to_numpy()
    return p[last].reshape(len(history) + 1, len(probes))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path, help="a model_run0.pkl written by ktbench.train")
    ap.add_argument("--history", required=True, help='e.g. "10:1 10:0 45+10:1"')
    ap.add_argument("--probe", default=None, help="comma-separated skills to track (default: skills in the history)")
    args = ap.parse_args(argv)

    model = load_model(args.model)
    history = parse_history(args.history or "")
    probes = [int(s) for s in args.probe.split(",")] if args.probe else sorted({s for sk, _ in history for s in sk})

    traj = trajectory(model, history, probes)
    print("step  answer        " + "  ".join(f"skill {s:>4}" for s in probes))
    print("0     (start)       " + "  ".join(f"{p:10.3f}" for p in traj[0]))
    for i, (skills, correct) in enumerate(history, start=1):
        ans = f"{'+'.join(map(str, skills))}:{'right' if correct else 'wrong'}"
        print(f"{i:<5} {ans:<13} " + "  ".join(f"{p:10.3f}" for p in traj[i]))


if __name__ == "__main__":
    main()
