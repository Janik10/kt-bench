import numpy as np

from ktbench import demo, models
from ktbench.train import load_model, save_model
from tests.conftest import TINY


def test_trajectory_shape_and_direction(dataset, split, tmp_path):
    _, meta = dataset
    m = models.build("beta", meta.n_skills, meta.n_items).fit(split.train)
    history = demo.parse_history("1:1 1:1 1:1 2:0 2:0")
    traj = demo.trajectory(m, history, [1, 2])
    assert traj.shape == (6, 2)
    assert traj[3, 0] > traj[0, 0]  # three right answers on skill 1 raise its mastery
    assert traj[5, 1] < traj[3, 1]  # two wrong answers on skill 2 lower it


def test_saved_neural_model_round_trips(dataset, split, tmp_path):
    _, meta = dataset
    m = models.build("dkt", meta.n_skills, meta.n_items, seed=0, **TINY["dkt"]).fit(split.train)
    before = m.predict(split.test)
    save_model(m, tmp_path / "m.pkl")
    np.testing.assert_allclose(load_model(tmp_path / "m.pkl").predict(split.test), before, rtol=1e-5)
