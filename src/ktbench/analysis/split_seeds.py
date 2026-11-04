"""How much does the draw of the 20% test students move each model's score?

    bash scripts/split_seeds.sh assist2009          # trains run 0 of every config on 5 test-set draws
    python -m ktbench.analysis.split_seeds assist2009

Each split seed draws a different set of test students. Models use their
config defaults (not the tuned params, which were tuned on a validation fold
that overlaps other seeds' test sets), seed 0 and validation fold 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ktbench.report import LABELS, ORDER


def collect(base: Path, dataset: str) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for seed_dir in sorted(base.glob("seed*")):
        seed = int(seed_dir.name[4:])
        for f in sorted((seed_dir / dataset).glob("*/run0.json")):
            out.setdefault(f.parent.name, {})[seed] = json.loads(f.read_text())
    return out


def table(runs: dict[str, dict[int, dict]]) -> str:
    names = sorted(runs, key=lambda n: (ORDER.index(n) if n in ORDER else len(ORDER), n))
    seeds = sorted({s for r in runs.values() for s in r})
    head = "| Model | " + " | ".join(f"seed {s}" for s in seeds) + " | Mean ± std |"
    lines = [head, "|---" * (len(seeds) + 2) + "|"]
    for n in names:
        aucs = [runs[n].get(s, {}).get("auc") for s in seeds]
        vals = np.array([a for a in aucs if a is not None])
        cells = " | ".join(f"{a:.4f}" if a is not None else "–" for a in aucs)
        lines.append(f"| {LABELS.get(n, n)} | {cells} | {vals.mean():.4f} ± {vals.std():.4f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--base", type=Path, default=Path("results/split_seeds"))
    args = ap.parse_args(argv)
    print(table(collect(args.base, args.dataset)))


if __name__ == "__main__":
    main()
