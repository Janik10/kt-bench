#!/usr/bin/env bash
# Train run 0 of every config on 5 different draws of the test students and
# print AUC per draw with mean ± std. See ktbench/analysis/split_seeds.py.
#
#   bash scripts/split_seeds.sh assist2009
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
DS=${1:-assist2009}
SEEDS=${SEEDS:-"0 1 2 3 4"}

for s in $SEEDS; do
  for cfg in configs/"$DS"/*.yaml; do
    $PY -m ktbench.train "$cfg" --runs 0 --no-tuned --split-seed "$s" \
      --results-dir "results/split_seeds/seed$s" --preds-dir "runs/split_seeds/seed$s"
  done
done
$PY -m ktbench.analysis.split_seeds "$DS"
