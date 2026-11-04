import numpy as np
import pandas as pd

from ktbench import offline_eval
from ktbench.data import synthetic


def fake_export(tmp_path, n_users=30, iso_times=True):
    """A graded-answer export in the documented format, built from synthetic data."""
    df, _ = synthetic.generate(n_users=n_users, n_skills=4, max_len=25, multi_skill_frac=0.0, seed=1)
    export = pd.DataFrame({
        "user_id": df["user_id"] + 100,
        "skill_id": "topic-" + df["skills"].map(lambda s: str(s[0])),
        "correct": df["correct"],
        "timestamp": pd.to_datetime(df["timestamp"], unit="s", utc=True).astype(str) if iso_times
        else df["timestamp"],
    })
    path = tmp_path / "export.csv"
    export.to_csv(path, index=False)
    return path, len(export)


def test_load_export_reads_iso_and_epoch_times(tmp_path):
    for iso in (True, False):
        path, n = fake_export(tmp_path, iso_times=iso)
        df, meta = offline_eval.load_export(path)
        assert len(df) == n
        assert meta.n_skills == 4
        assert not np.isnan(df["timestamp"]).any()


def test_temporal_mask_splits_each_student_70_30(tmp_path):
    df, _ = offline_eval.load_export(fake_export(tmp_path)[0])
    train = offline_eval.temporal_mask(df)
    for _, g in df.assign(train=train).groupby("user_id"):
        # Every training answer comes before every test answer.
        assert g["train"].iloc[0] and not g["train"].iloc[-1]
        assert (np.diff(g["train"].astype(int)) <= 0).all()


def test_offline_evaluation_runs_and_reports_cis(tmp_path):
    df, meta = offline_eval.load_export(fake_export(tmp_path)[0])
    table = offline_eval.evaluate(df, meta, n_boot=50)
    assert set(table["model"]) == {"current", *offline_eval.CANDIDATES}
    for _, row in table.iterrows():
        lo, hi = row["auc_95ci"]
        assert lo <= row["auc"] + 0.05 and hi >= row["auc"] - 0.05
