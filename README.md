# kt-bench

[![CI](https://github.com/Janik10/kt-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/Janik10/kt-bench/actions/workflows/ci.yml)

A reproducible benchmark of knowledge-tracing models: given a student's answer
history, predict whether they get the next question right.

**The question it answers:** does a neural model (DKT, SAKT) predict student
answers better than a Bayesian mastery model and the classic baselines, and by
how much? This holds in particular for students with very little history.

## Results

<!-- results:start -->
_No results yet._
<!-- results:end -->

All numbers are on held-out **students** (never seen in training), with one
prediction per question, so multi-skill questions are scored once. Each run r
uses seed r and validation fold r, and the test set is the same 20% of
students in every run. "AUC, 1–5 prior answers" is the cold-start slice: answers
where the student had given only 1–5 answers before.

## Models

All six sit behind one interface: `fit(train, valid)` and `predict(df) -> P(correct)`,
where each prediction may only use that student's earlier answers. One scorer
(`ktbench.evaluate`) scores them all.

| # | Model | What it is | Code |
|---|---|---|---|
| 0 | Skill average | Each skill's overall correct rate; the floor | [skill_avg.py](src/ktbench/models/skill_avg.py) |
| 1 | BKT | Hidden known/unknown state per skill, fit by EM | [bkt.py](src/ktbench/models/bkt.py) |
| 2 | PFA | Logistic regression on per-skill prior successes/failures | [pfa.py](src/ktbench/models/pfa.py) |
| 3 | Bayesian Beta baseline | Per-skill Beta(α, β) counts of right/wrong answers with a decaying half-life | [beta.py](src/ktbench/models/beta.py) |
| 4 | DKT | LSTM over the answer history, written from scratch | [dkt.py](src/ktbench/models/dkt.py) |
| 5 | SAKT | Self-attention over past answers, written from scratch | [sakt.py](src/ktbench/models/sakt.py) |

The Bayesian Beta baseline is reported twice: with default settings, and with the
prior and evidence half-life tuned on the validation fold.

## Protocol

Follows [pyKT](https://arxiv.org/abs/2206.11460) so the numbers compare with published ones.

- **Splits by student.** 20% of students are the test set; the other 80% are split
  into 5 folds. Hyperparameters and early stopping only ever see the validation fold.
- **One-step-ahead.** Predict answer t+1 from answers 1..t. Each student's first
  answer is not scored.
- **Question-level, no label leakage.** Skill-level models (BKT, DKT, SAKT) score
  every skill of a question from the state *before* the question and average them.
  The second skill of a question never sees the answer to the first: sequence
  models carry a per-token `ctx` index that enforces this (see
  [sequences.py](src/ktbench/data/sequences.py)).
- **Equal tuning budget.** Every tuned model gets the same number of random-search
  trials (`TRIALS`, default 12) over its `search:` block.

Reference AUCs on ASSISTments 2009 (pyKT paper): DKT 0.7541, SAKT 0.7246, AKT 0.7853.

## Tests that guard the protocol

| Test | What it catches |
|---|---|
| [test_splits.py](tests/test_splits.py) | A test student appearing in train or validation |
| [test_causality.py](tests/test_causality.py) | A prediction moving when that student's current or future answers change, including the multi-skill leak |
| [test_determinism.py](tests/test_determinism.py) | The same seed giving a different AUC (4 decimal places) |
| [test_padding.py](tests/test_padding.py) | Padded positions affecting predictions, the loss or the metrics |

CI runs lint, the tests and an end-to-end run on a small synthetic dataset on every push.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.11
pip install torch --index-url https://download.pytorch.org/whl/cu126   # or /whl/cpu
pip install -e ".[dev]"
bash scripts/reproduce_all.sh                # downloads ASSISTments 2009, trains everything, prints the table
python -m ktbench.report --readme            # writes the table above
```

One model, one run, with a quick override:

```bash
python -m ktbench.train configs/assist2009/dkt.yaml --runs 0 --set max_epochs=5
python -m ktbench.tune  configs/assist2009/pfa.yaml --trials 12
```

ASSISTments 2009 trains on a laptop CPU; DKT and SAKT use a GPU when one is available.

## Data

Datasets are downloaded by the code and never committed.

- **ASSISTments 2009–2010 Skill Builder**, the *corrected, collapsed* release (the
  original has duplicate rows). The raw file matches pyKT's statistics (4,217 students,
  346,860 rows, 26,688 questions, 123 skills); after pyKT's cleaning (drop rows with no skill,
  drop students with fewer than 3 answers) it is 3,852 students, 282,619 interactions,
  17,737 questions, 123 skills.
  Source: [ASSISTments data page](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010).
  Please cite the dataset and the ASSISTments paper if you use it.
- **EdNet KT1** ([repository](https://github.com/riiid/ednet)), CC BY-NC 4.0, research and
  non-commercial use only.

## Layout

```
configs/<dataset>/<model>.yaml   one config per model x dataset
src/ktbench/data/                download, clean, split, Parquet, sequence windows
src/ktbench/models/              the six models behind one interface
src/ktbench/train.py             trains + tests any config
src/ktbench/tune.py              validation-only random search
src/ktbench/evaluate.py          the one scorer
src/ktbench/analysis/            cold-start buckets, calibration plot
results/                         per-run JSON (committed); predictions go to runs/ (ignored)
```

## License

Code: MIT. Datasets keep their own terms (see above).
