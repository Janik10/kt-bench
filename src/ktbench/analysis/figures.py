"""The report figures, as PNGs under report/figures/.

    python -m ktbench.analysis.figures

- calibration_<dataset>.png: predicted probability vs. observed correct rate per model
- cold_start.png: AUC by number of prior answers (1–5, 6–20, 21+), one panel per dataset
- cost_vs_auc.png: training time vs. AUC, one panel per dataset

Every model keeps one color and one marker in every figure (validated categorical
palette; the marker is the secondary encoding the lighter hues need).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ktbench import metrics
from ktbench.analysis.cold_start import BUCKETS
from ktbench.report import DATASETS, LABELS, ORDER, SPLIT_SEEDS, collect, collect_split_seeds

LABEL_OFFSET = {"beta_default": (6, -11)}  # points; keeps neighbouring labels apart
COLORS = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"],
                  strict=True))
MARKERS = dict(zip(ORDER, ["o", "s", "D", "^", "v", "P", "X"], strict=True))
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
    })
    return plt


def _order(names) -> list[str]:
    return sorted(names, key=lambda n: (ORDER.index(n) if n in ORDER else len(ORDER), n))


def calibration(dataset: str, preds_dir: Path, out: Path, n_bins: int = 10) -> Path:
    plt = _plt()
    base = preds_dir / dataset
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls="--", label="perfect calibration")
    for name in _order(p.name for p in base.iterdir() if p.is_dir()):
        files = sorted((base / name).glob("run*.parquet"))
        if not files:
            continue
        d = pd.concat(pd.read_parquet(f) for f in files)
        d = d[d["t"] > 0]
        y, p = d["correct"].to_numpy(float), d["p"].to_numpy()
        mean_p, rate, count = metrics.calibration_bins(y, p, n_bins)
        ok = count >= 50  # skip near-empty bins: their rate is noise
        ax.plot(mean_p[ok], rate[ok], color=COLORS.get(name), marker=MARKERS.get(name), ms=5, lw=2,
                label=f"{LABELS.get(name, name)} (ECE {metrics.ece(y, p, n_bins):.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted probability of a right answer",
           ylabel="Observed share of right answers")
    ax.set_title(f"Calibration on held-out students: {DATASETS.get(dataset, dataset)}", loc="left")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _cold_start_rows(results_dir: Path, dataset: str) -> tuple[dict[str, list[dict]], str]:
    draws = collect_split_seeds(results_dir / "split_seeds", dataset, SPLIT_SEEDS)
    if draws:
        return draws, f"{len(SPLIT_SEEDS)} test draws"
    return collect(results_dir).get(dataset, {}), "5 training seeds"


def cold_start(results_dir: Path, out: Path) -> Path:
    plt = _plt()
    datasets = [d for d in DATASETS if collect(results_dir).get(d)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.8), squeeze=False, sharey=True)
    for ax, ds in zip(axes[0], datasets, strict=True):
        rows, source = _cold_start_rows(results_dir, ds)
        names = _order(rows)
        width = 0.8 / len(names)
        for i, n in enumerate(names):
            aucs = np.array([[r["cold_start"][b]["auc"] for b, _, _ in BUCKETS] for r in rows[n]])
            x = np.arange(len(BUCKETS)) + (i - (len(names) - 1) / 2) * width
            ax.bar(x, aucs.mean(0), width * 0.9, color=COLORS.get(n), label=LABELS.get(n, n),
                   yerr=aucs.std(0) if len(aucs) > 1 else None, error_kw={"ecolor": INK2, "lw": 0.8})
        ax.set_xticks(np.arange(len(BUCKETS)), [f"{b} prior answers" for b, _, _ in BUCKETS])
        ax.set_ylim(0.5, None)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{DATASETS[ds]} (mean over {source})", loc="left")
    axes[0][0].set_ylabel("AUC")
    handles = {}  # one legend for every model drawn in any panel, in the fixed model order
    for ax in axes[0]:
        for h, label in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(label, h)
    labels = [LABELS.get(n, n) for n in ORDER if LABELS.get(n, n) in handles]
    axes[0][-1].legend([handles[lb] for lb in labels], labels, fontsize=7.5, loc="upper left",
                       bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("AUC by how many answers the student had already given", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def cost_vs_auc(results_dir: Path, out: Path) -> Path:
    plt = _plt()
    data = collect(results_dir)
    datasets = [d for d in DATASETS if data.get(d)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.8), squeeze=False)
    for ax, ds in zip(axes[0], datasets, strict=True):
        for n in _order(data[ds]):
            runs = data[ds][n]
            secs = max(np.mean([r["train_secs"] for r in runs]), 0.01)  # log axis needs > 0
            auc = np.mean([r["auc"] for r in runs])
            ax.scatter(secs, auc, s=60, color=COLORS.get(n), marker=MARKERS.get(n), edgecolor=SURFACE,
                       linewidth=1.5, zorder=3)
            ax.annotate(LABELS.get(n, n), (secs, auc), xytext=LABEL_OFFSET.get(n, (6, 4)), textcoords="offset points",
                        fontsize=7.5, color=INK2)
        ax.set_xscale("log")
        ax.set_xlabel("Training time per run (s, log scale; under 0.01 s drawn at 0.01)")
        ax.set_title(f"{DATASETS[ds]}: fixed test set, mean of 5 runs", loc="left")
    axes[0][0].set_ylabel("Test AUC")
    fig.suptitle("What each point of AUC costs to train", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--preds-dir", type=Path, default=Path("runs"))
    ap.add_argument("--out-dir", type=Path, default=Path("report/figures"))
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        if (args.preds_dir / ds).exists():
            print(calibration(ds, args.preds_dir, args.out_dir / f"calibration_{ds}.png"))
    print(cold_start(args.results_dir, args.out_dir / "cold_start.png"))
    print(cost_vs_auc(args.results_dir, args.out_dir / "cost_vs_auc.png"))


if __name__ == "__main__":
    main()
