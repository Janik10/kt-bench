"""Collect run results into the README tables.

    python -m ktbench.report            # print the tables
    python -m ktbench.report --readme   # also rewrite the block between the README markers

Headline: ASSISTments 2009 over 5 different draws of the test students
(``results/split_seeds/seed0..4``, config defaults). Secondary: the fixed
seed-42 test set with tuned params, 5 training seeds (``results/<dataset>``).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

START, END = "<!-- results:start -->", "<!-- results:end -->"
ORDER = ["skill_avg", "bkt", "pfa", "beta_default", "beta", "dkt", "sakt"]
LABELS = {
    "skill_avg": "Skill average",
    "bkt": "BKT",
    "pfa": "PFA",
    "beta_default": "Bayesian Beta baseline (default)",
    "beta": "Bayesian Beta baseline (tuned)",
    "dkt": "DKT",
    "sakt": "SAKT",
}
DATASETS = {"assist2009": "ASSISTments 2009", "ednet": "EdNet KT1 (sample)"}
# pyKT (Liu et al., 2022), question-level, all-in-one, late-fusion mean, ASSISTments 2009.
PYKT = {"assist2009": {"dkt": 0.7541, "sakt": 0.7246}}
SPLIT_SEEDS = [0, 1, 2, 3, 4]
FIXED_SEED = 42


def collect(results_dir: Path) -> dict[str, dict[str, list[dict]]]:
    out: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for ds_dir in sorted(p for p in results_dir.iterdir() if p.is_dir() and p.name != "split_seeds"):
        for cfg_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            runs = [json.loads(f.read_text()) for f in sorted(cfg_dir.glob("run*.json"))]
            if runs:
                out[ds_dir.name][cfg_dir.name] = runs
    return out


def collect_split_seeds(base: Path, dataset: str, seeds: list[int]) -> dict[str, list[dict]]:
    """config -> run-0 results, one per split seed that has it."""
    out: dict[str, list[dict]] = defaultdict(list)
    for s in seeds:
        for f in sorted((base / f"seed{s}" / dataset).glob("*/run0.json")):
            out[f.parent.name].append(json.loads(f.read_text()))
    return dict(out)


def _order(names) -> list[str]:
    return sorted(names, key=lambda n: (ORDER.index(n) if n in ORDER else len(ORDER), n))


def _ms(values: list[float], digits: int = 4) -> str:
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.{digits}f} ± {v.std():.{digits}f}"


def _pykt(dataset: str, name: str, auc: float) -> str:
    ref = PYKT.get(dataset, {}).get(name)
    return f"{ref:.4f} ({auc - ref:+.4f})" if ref is not None else "–"


def headline_table(dataset: str, by_cfg: dict[str, list[dict]]) -> str:
    lines = [
        f"**{DATASETS.get(dataset, dataset)}: {len(SPLIT_SEEDS)} draws of the 20% test students**, "
        "config defaults, mean ± std over draws",
        "",
        "| Model | AUC | pyKT published AUC (ours − pyKT) | Accuracy | Brier | ECE | AUC, 1–5 prior answers | Draws |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n in _order(by_cfg):
        runs = by_cfg[n]
        auc = float(np.mean([r["auc"] for r in runs]))
        lines.append(
            f"| {LABELS.get(n, n)} | {_ms([r['auc'] for r in runs])} | {_pykt(dataset, n, auc)} "
            f"| {_ms([r['acc'] for r in runs])} | {_ms([r['brier'] for r in runs])} | {_ms([r['ece'] for r in runs])} "
            f"| {_ms([r['cold_start']['1-5']['auc'] for r in runs])} | {len(runs)} |"
        )
    return "\n".join(lines)


def fixed_table(dataset: str, configs: dict[str, list[dict]], title: str) -> str:
    lines = [
        title,
        "",
        "| Model | AUC | pyKT published AUC (ours − pyKT) | Accuracy | Brier | ECE | AUC, 1–5 prior answers "
        "| Train (s) | Inference (µs/answer) | Runs |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n in _order(configs):
        runs = configs[n]
        auc = float(np.mean([r["auc"] for r in runs]))
        lines.append(
            f"| {LABELS.get(n, n)} | {_ms([r['auc'] for r in runs])} | {_pykt(dataset, n, auc)} "
            f"| {_ms([r['acc'] for r in runs])} "
            f"| {_ms([r['brier'] for r in runs])} | {_ms([r['ece'] for r in runs])} "
            f"| {_ms([r['cold_start']['1-5']['auc'] for r in runs])} "
            f"| {np.mean([r['train_secs'] for r in runs]):.1f} "
            f"| {np.mean([r['infer_us_per_answer'] for r in runs]):.1f} | {len(runs)} |"
        )
    return "\n".join(lines)


def fixed_seed_offset(base: Path, dataset: str) -> tuple[float, int] | None:
    """Mean over models of (seed-42 AUC − mean AUC over the other draws), all with config defaults."""
    draws = collect_split_seeds(base, dataset, SPLIT_SEEDS)
    fixed = collect_split_seeds(base, dataset, [FIXED_SEED])
    diffs = [fixed[n][0]["auc"] - np.mean([r["auc"] for r in draws[n]]) for n in fixed if n in draws]
    return (float(np.mean(diffs)), len(diffs)) if diffs else None


def render(results_dir: Path) -> str:
    data = collect(results_dir)
    split_base = results_dir / "split_seeds"
    parts = []
    for ds in [d for d in DATASETS if d in data] + [d for d in data if d not in DATASETS]:
        draws = collect_split_seeds(split_base, ds, SPLIT_SEEDS) if split_base.exists() else {}
        if draws:
            parts.append(headline_table(ds, draws))
            offset = fixed_seed_offset(split_base, ds)
            note = (f"The fixed seed-{FIXED_SEED} test set is {offset[0]:+.4f} AUC easier than an average draw "
                    f"(mean over {offset[1]} models, same default params), so read the table below as "
                    "a within-split comparison, not an estimate of absolute AUC.") if offset else ""
            parts.append(fixed_table(
                ds, data[ds],
                f"**{DATASETS.get(ds, ds)}: fixed seed-{FIXED_SEED} test set**, tuned params, "
                "mean ± std over 5 training seeds" + (f"\n\n{note}" if note else "")))
        else:
            parts.append(fixed_table(
                ds, data[ds],
                f"**{DATASETS.get(ds, ds)}**, held-out test students, mean ± std over 5 training seeds"))
    return "\n\n".join(parts) or "_No results yet._"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--readme", type=Path, nargs="?", const=Path("README.md"), default=None)
    args = ap.parse_args(argv)
    text = render(args.results_dir)
    print(text)
    if args.readme:
        readme = args.readme.read_text(encoding="utf-8")
        a, b = readme.index(START) + len(START), readme.index(END)
        args.readme.write_text(readme[:a] + "\n" + text + "\n" + readme[b:], encoding="utf-8")


if __name__ == "__main__":
    main()
