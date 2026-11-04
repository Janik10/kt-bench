"""Collect run results into the README tables.

    python -m ktbench.report            # print the tables
    python -m ktbench.report --readme   # also rewrite the block between the README markers
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


def collect(results_dir: Path) -> dict[str, dict[str, list[dict]]]:
    out: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for ds_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        for cfg_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            runs = [json.loads(f.read_text()) for f in sorted(cfg_dir.glob("run*.json"))]
            if runs:
                out[ds_dir.name][cfg_dir.name] = runs
    return out


def _ms(values: list[float], digits: int = 4) -> str:
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.{digits}f} ± {v.std():.{digits}f}"


def table(dataset: str, configs: dict[str, list[dict]]) -> str:
    names = sorted(configs, key=lambda n: (ORDER.index(n) if n in ORDER else len(ORDER), n))
    lines = [
        f"**{DATASETS.get(dataset, dataset)}**, held-out test students, mean ± std over runs",
        "",
        "| Model | AUC | Accuracy | Brier | ECE | AUC, 1–5 prior answers | Train (s) | Inference (µs/answer) | Runs |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for n in names:
        runs = configs[n]
        lines.append(
            f"| {LABELS.get(n, n)} | {_ms([r['auc'] for r in runs])} | {_ms([r['acc'] for r in runs])} "
            f"| {_ms([r['brier'] for r in runs])} | {_ms([r['ece'] for r in runs])} "
            f"| {_ms([r['cold_start']['1-5']['auc'] for r in runs])} "
            f"| {np.mean([r['train_secs'] for r in runs]):.1f} "
            f"| {np.mean([r['infer_us_per_answer'] for r in runs]):.1f} | {len(runs)} |"
        )
    return "\n".join(lines)


def render(results_dir: Path) -> str:
    data = collect(results_dir)
    order = [d for d in DATASETS if d in data] + [d for d in data if d not in DATASETS]
    return "\n\n".join(table(d, data[d]) for d in order) or "_No results yet._"


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
