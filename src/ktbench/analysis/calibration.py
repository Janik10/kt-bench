"""Reliability diagram for one dataset. Thin wrapper; the plot lives in ``figures``.

    python -m ktbench.analysis.calibration assist2009
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ktbench.analysis.figures import calibration


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--preds-dir", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.out or Path("report/figures") / f"calibration_{args.dataset}.png"
    print(calibration(args.dataset, args.preds_dir, out))


if __name__ == "__main__":
    main()
