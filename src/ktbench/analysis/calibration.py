"""Reliability diagram: predicted probability vs. observed correct rate, one line per model.

    python -m ktbench.analysis.calibration assist2009
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ktbench import metrics
from ktbench.report import LABELS, ORDER


def plot(dataset: str, preds_dir: Path, out: Path, n_bins: int = 10) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = preds_dir / dataset
    names = sorted((p.name for p in base.iterdir() if p.is_dir()),
                   key=lambda n: (ORDER.index(n) if n in ORDER else len(ORDER), n))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="perfect calibration")
    for name in names:
        runs = [pd.read_parquet(f) for f in sorted((base / name).glob("run*.parquet"))]
        d = pd.concat(runs)
        d = d[d["t"] > 0]
        mean_p, rate, count = metrics.calibration_bins(d["correct"].to_numpy(float), d["p"].to_numpy(), n_bins)
        ok = count > 0
        ece = metrics.ece(d["correct"].to_numpy(float), d["p"].to_numpy(), n_bins)
        ax.plot(mean_p[ok], rate[ok], marker="o", ms=3, lw=1.5, label=f"{LABELS.get(name, name)} (ECE {ece:.3f})")
    ax.set_xlabel("Predicted probability correct")
    ax.set_ylabel("Observed correct rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Calibration on held-out students: {dataset}")
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.grid(alpha=0.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--preds-dir", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.out or Path("report/figures") / f"calibration_{args.dataset}.png"
    print(plot(args.dataset, args.preds_dir, out))


if __name__ == "__main__":
    main()
