"""Random hyperparameter search on the validation fold only.

    python -m ktbench.tune configs/assist2009/pfa.yaml --trials 12

Every model gets the same budget of ``--trials`` sampled configurations from
the ``search:`` block of its config. Trials train on folds 1-4 and are scored
by AUC on fold 0; the test students are never loaded into a model. The best
params go to ``results/<dataset>/<config>/tuning.json`` and ``train`` picks
them up automatically.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from ktbench import data as datasets
from ktbench import metrics, models
from ktbench.data import splits
from ktbench.seed import set_seed
from ktbench.train import load_config

VALID_FOLD = 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    space = cfg.get("search") or {}
    if not space:
        raise SystemExit(f"{args.config} has no search: block")
    df, meta, data_path = datasets.load(cfg["dataset"], args.data_dir)
    split = splits.make_split(df, splits.load_or_create(df, data_path), valid_fold=VALID_FOLD)
    del df  # nothing below may touch the test students

    keys = sorted(space)
    grid = list(itertools.product(*(space[k] for k in keys)))
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(grid), size=min(args.trials, len(grid)), replace=False)

    trials = []
    for i, g in enumerate(picks):
        trial = dict(zip(keys, grid[g], strict=True))
        params = {**cfg["params"], **trial, "verbose": False}
        set_seed(args.seed)
        t0 = time.perf_counter()
        model = models.build(cfg["model"], meta.n_skills, meta.n_items, seed=args.seed, **params)
        model.fit(split.train, split.valid)
        auc = metrics.auc(split.valid["correct"].to_numpy(), model.predict(split.valid))
        trials.append({"params": trial, "valid_auc": auc, "secs": time.perf_counter() - t0})
        print(f"[{i + 1}/{len(picks)}] valid auc={auc:.4f} {trial}")

    best = max(trials, key=lambda r: r["valid_auc"])
    out = args.results_dir / cfg["dataset"] / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "tuning.json").write_text(json.dumps(
        {"valid_fold": VALID_FOLD, "budget": args.trials, "best_params": best["params"],
         "best_valid_auc": best["valid_auc"], "trials": trials}, indent=2))
    print(f"best valid auc={best['valid_auc']:.4f} {best['params']}")


if __name__ == "__main__":
    main()
