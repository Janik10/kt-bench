# Changelog

## Unreleased

### Added
- Project scaffolding: pyproject with pinned versions, ruff, pytest, GitHub Actions CI.
- ASSISTments 2009 (corrected, collapsed) download and pyKT-style cleaning to Parquet.
- Student-level 80/20 test split with 5 validation folds, saved next to the data.
- One evaluator for every model: AUC, accuracy, Brier, ECE, log loss, cold-start buckets.
- Baselines: skill average, BKT (EM, implemented here), PFA, a Bayesian Beta baseline.
- DKT and SAKT in PyTorch with a per-token question-start index that stops a
  multi-skill question from seeing its own answer.
- Tests for student leakage, causality, determinism and padding.
- `ktbench.train`, validation-only `ktbench.tune`, `scripts/reproduce_all.sh`, README table generator.
- EdNet KT1 pipeline: answers joined to the correct letter and tags, Parquet parts, 50k-student sample.
