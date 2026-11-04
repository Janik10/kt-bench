import pandas as pd

from ktbench.data import assist2009


def test_preprocess_follows_pykt_rules(tmp_path):
    raw = pd.DataFrame(
        {
            "order_id": [3, 1, 2, 10, 11, 12, 13, 20, 21],
            "user_id": [7, 7, 7, 8, 8, 8, 8, 9, 9],
            "problem_id": [100, 101, 102, 100, 103, 104, 105, 100, 101],
            "skill_id": ["5", "5_9", "9", "5", None, "9", "5", "5", "9"],
            "correct": ["1", "0", "1", "0", "1", "1", "0.5", "1", "1"],
            "other": "x",
        }
    )
    path = tmp_path / "raw.csv"
    raw.to_csv(path)
    df, meta = assist2009.preprocess(path)
    # User 9 has only 2 answers and is dropped. User 8 loses the row with no
    # skill and the non-0/1 row, leaving 2 answers, so is dropped too.
    assert meta.n_users == 1
    assert len(df) == 3
    assert meta.n_skills == 2
    # Ordered by order_id (101, 102, 100), and "5_9" is split into two skills.
    assert df["item_id"].tolist() == [1, 2, 0]
    assert df["skills"].tolist() == [[0, 1], [1], [0]]
