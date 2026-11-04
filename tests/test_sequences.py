import numpy as np

from ktbench.data import sequences


def test_question_ctx_points_at_question_start():
    row = np.array([0, 1, 1, 2, 3, 3, 3])
    assert sequences.question_ctx(row).tolist() == [0, 1, 1, 3, 4, 4, 4]


def test_train_windows_cover_every_token(split):
    windows = sequences.train_windows(split.train, 16)
    assert max(len(w.skill) for w in windows) <= 16
    assert sum(len(w.skill) for w in windows) == int(split.train["skills"].map(len).sum())
