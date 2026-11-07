#!/usr/bin/env bash
# Reproduce every number in the README.
#
#   bash scripts/reproduce_all.sh               # ASSISTments 2009: every model, 5 runs each,
#                                               # plus the 5-draw split study (the headline table)
#   bash scripts/reproduce_all.sh assist2009 ednet
#   bash scripts/reproduce_all.sh synthetic     # tiny fixture, what CI runs
#
# Set TUNE=1 to rerun the validation-only hyperparameter search first
# (TRIALS=n, default 12). Without it, the committed tuning.json files are used.
# Set SPLITS=0 to skip the ASSISTments split study (about 30 minutes on a laptop GPU).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
DATASETS=${*:-assist2009}
TRIALS=${TRIALS:-12}

for ds in $DATASETS; do
  results=results
  preds=runs
  if [ "$ds" = synthetic ]; then
    # Keep CI output out of the committed results.
    results=runs/ci/results
    preds=runs/ci/preds
  fi
  for cfg in configs/"$ds"/*.yaml; do
    if [ "${TUNE:-0}" = 1 ] && grep -q '^search:' "$cfg"; then
      $PY -m ktbench.tune "$cfg" --trials "$TRIALS" --results-dir "$results"
    fi
    $PY -m ktbench.train "$cfg" --results-dir "$results" --preds-dir "$preds"
  done
  if [ "$ds" = synthetic ]; then
    $PY -m ktbench.analysis.calibration "$ds" --preds-dir "$preds" --out runs/ci/calibration_$ds.png
  fi
  if [ "$ds" = assist2009 ] && [ "${SPLITS:-1}" = 1 ]; then
    SEEDS="0 1 2 3 4 42" PYTHON="$PY" bash scripts/split_seeds.sh assist2009
  fi
  $PY -m ktbench.report --results-dir "$results"
done
if [ "$DATASETS" != synthetic ]; then
  $PY -m ktbench.analysis.figures   # report/figures/*.png
fi
