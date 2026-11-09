"""Train and test any model from a config.

    python -m ktbench.train configs/assist2009/dkt.yaml
    python -m ktbench.train configs/assist2009/dkt.yaml --runs 0 --set max_epochs=3

Run r uses seed r and validation fold r (early stopping and tuning only ever
see the validation fold), then scores the untouched test students. Each run
writes ``results/<dataset>/<config>/run<r>.json``; test predictions go to
``runs/`` (git-ignored) for the calibration plots.
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ktbench import data as datasets
from ktbench import models
from ktbench.data import splits
from ktbench.evaluate import evaluate
from ktbench.seed import set_seed


def load_config(path: Path, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(path.read_text())
    cfg.setdefault("name", path.stem)
    cfg.setdefault("params", {})
    for item in overrides or []:
        key, value = item.split("=", 1)
        cfg["params"][key] = yaml.safe_load(value)
    return cfg


def tuned_params(cfg: dict, results_dir: Path) -> dict:
    path = results_dir / cfg["dataset"] / cfg["name"] / "tuning.json"
    return json.loads(path.read_text())["best_params"] if path.exists() else {}


def run_one(cfg: dict, run: int, df: pd.DataFrame, meta, folds: pd.DataFrame, results_dir: Path,
            preds_dir: Path, params: dict) -> dict:
    set_seed(run)
    valid_fold = run % int(folds["fold"].max() + 1)
    split = splits.make_split(df, folds, valid_fold=valid_fold)
    model = models.build(cfg["model"], meta.n_skills, meta.n_items, seed=run, **params)
    t0 = time.perf_counter()
    model.fit(split.train, split.valid)
    train_secs = time.perf_counter() - t0
    result, p = evaluate(model, split.test)
    result.update(
        dataset=cfg["dataset"], config=cfg["name"], model=cfg["model"], run=run, seed=run,
        valid_fold=valid_fold, train_secs=train_secs, params=params,
        n_train_students=int(split.train["user_id"].nunique()),
        n_test_students=int(split.test["user_id"].nunique()),
        history=getattr(model, "history", None), python=platform.python_version(),
    )
    out = results_dir / cfg["dataset"] / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"run{run}.json").write_text(json.dumps(result, indent=2))
    pdir = preds_dir / cfg["dataset"] / cfg["name"]
    pdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"user_id": split.test["user_id"], "t": split.test["t"], "correct": split.test["correct"], "p": p}
    ).to_parquet(pdir / f"run{run}.parquet", index=False)
    if run == 0:
        save_model(model, pdir / "model_run0.pkl")
    return result


def save_model(model, path: Path) -> None:
    """Pickle a fitted model (moved to CPU so it loads anywhere) for the demo."""
    if hasattr(model, "net"):
        import torch

        model.net.cpu()
        model.device = torch.device("cpu")
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--runs", default=None, help="comma-separated run ids (default: 0..runs-1 from config)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a model param")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--preds-dir", type=Path, default=Path("runs"))
    ap.add_argument("--no-tuned", action="store_true", help="ignore tuning.json even if present")
    ap.add_argument("--split-seed", type=int, default=splits.DEFAULT_SEED,
                    help="draw a different 20%% test set (for split-variance checks only)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    df, meta, data_path = datasets.load(cfg["dataset"], args.data_dir)
    folds = splits.load_or_create(df, data_path, seed=args.split_seed)
    params = dict(cfg["params"])
    if not args.no_tuned:
        params.update(tuned_params(cfg, args.results_dir))
        params.update({k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)})
    runs = [int(r) for r in args.runs.split(",")] if args.runs else list(range(int(cfg.get("runs", 5))))

    aucs = []
    for run in runs:
        print(f"[{cfg['dataset']}/{cfg['name']}] run {run}", flush=True)
        r = run_one(cfg, run, df, meta, folds, args.results_dir, args.preds_dir, params)
        aucs.append(r["auc"])
        print(f"  test auc={r['auc']:.4f} acc={r['acc']:.4f} brier={r['brier']:.4f} "
              f"ece={r['ece']:.4f} train={r['train_secs']:.1f}s", flush=True)
    print(f"[{cfg['dataset']}/{cfg['name']}] AUC {np.mean(aucs):.4f} +/- {np.std(aucs):.4f} over {len(aucs)} runs",
          flush=True)


if __name__ == "__main__":
    main()
