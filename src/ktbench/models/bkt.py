"""Model 1: Bayesian Knowledge Tracing (Corbett & Anderson, 1995).

A two-state hidden Markov model per skill: the student either knows the skill
or not. Parameters per skill: prior P(L0), learn P(T), guess P(G), slip P(S)
and optionally forget P(F). Fit per skill by EM (Baum-Welch) with random
restarts, keeping the restart with the highest likelihood, the same scheme
pyBKT uses.

Implemented here rather than wrapping pyBKT: pyBKT 1.4.3, the latest
release, fails under numpy 2 and scikit-learn >= 1.8.

Prediction is forward filtering, so the prediction for an answer uses only
earlier answers on that skill. Multi-skill questions are split into one row
per skill (each skill is its own independent HMM, so no skill sees another's
answer) and averaged back. Skills absent from training fall back to the
training correct rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ktbench.models.base import KTModel, explode_skills, fuse

EPS = 1e-6
TINY = np.finfo(np.float64).tiny
MAX_CELLS = 2_000_000  # padded (sequences x steps) per chunk; bounds memory on EdNet


def pad_by_skill(ex: pd.DataFrame, max_cells: int | None = None):
    """Yield (skill, chunks) where each chunk is (obs, mask, ex_index), one padded row per
    (student, skill) sequence. Sequences are sorted by length and grouped so that no chunk
    pads more than ``max_cells`` cells."""
    max_cells = max_cells or MAX_CELLS
    ex = ex.assign(_i=np.arange(len(ex)))
    for skill, g in ex.groupby("skill", sort=True):
        seq = pd.factorize(g["user_id"])[0]
        pos = g.groupby("user_id", sort=False).cumcount().to_numpy()
        lengths = np.bincount(seq)
        order = np.argsort(-lengths, kind="stable")
        chunks, start = [], 0
        while start < len(order):
            width = lengths[order[start]]
            stop = start + max(1, max_cells // width)
            members = order[start:stop]
            slot = np.full(len(lengths), -1)
            slot[members] = np.arange(len(members))
            sel = slot[seq] >= 0
            obs = np.zeros((len(members), width))
            mask = np.zeros_like(obs, dtype=bool)
            idx = np.full(obs.shape, -1, dtype=np.int64)
            r, c = slot[seq[sel]], pos[sel]
            obs[r, c] = g["correct"].to_numpy()[sel]
            mask[r, c] = True
            idx[r, c] = g["_i"].to_numpy()[sel]
            chunks.append((obs, mask, idx))
            start = stop
        yield int(skill), chunks


def forward(obs: np.ndarray, mask: np.ndarray, prior, learn, guess, slip, forget=0.0):
    """Predictive P(known) and P(correct) before each observation, plus the log-likelihood."""
    n, T = obs.shape
    known = np.full(n, prior)
    p_known = np.empty((n, T))
    p_correct = np.empty((n, T))
    loglik = 0.0
    for t in range(T):
        p_known[:, t] = known
        pc = known * (1 - slip) + (1 - known) * guess
        p_correct[:, t] = pc
        o, m = obs[:, t], mask[:, t]
        # Likelihood of the observed answer from each state, with no 1 - P(correct): when guess
        # and slip are ~1e-5 that subtraction loses ~12 digits and pushed P(known) above 1.
        from_known = known * np.where(o == 1, 1 - slip, slip)
        from_unknown = (1 - known) * np.where(o == 1, guess, 1 - guess)
        like = from_known + from_unknown
        loglik += np.log(np.maximum(like[m], TINY)).sum()
        post = from_known / np.maximum(like, TINY)  # a / (a + b), so never above 1
        nxt = post * (1 - forget) + (1 - post) * learn
        known = np.where(m, nxt, known)
    return p_known, p_correct, loglik


def e_step(obs, mask, prior, learn, guess, slip, forget):
    """Baum-Welch expected counts for one chunk, with per-step scaling."""
    n, T = obs.shape
    A = np.array([[1 - learn, learn], [forget, 1 - forget]])  # rows: from unknown, known
    emit1 = np.array([guess, 1 - slip])  # P(correct | state)

    def emission(t):
        o = obs[:, t][:, None]
        e = np.where(o == 1, emit1[None, :], 1 - emit1[None, :])
        return np.where(mask[:, t][:, None], e, 1.0)  # padding emits nothing

    alpha = np.empty((n, T, 2))
    scale = np.ones((n, T))
    a = np.stack([np.full(n, 1 - prior), np.full(n, prior)], axis=1) * emission(0)
    scale[:, 0] = a.sum(1)
    alpha[:, 0] = a / scale[:, :1]
    for t in range(1, T):
        a = (alpha[:, t - 1] @ A) * emission(t)
        s = a.sum(1)
        active = mask[:, t]
        alpha[:, t] = np.where(active[:, None], a / s[:, None], alpha[:, t - 1])
        scale[:, t] = np.where(active, s, 1.0)

    beta = np.ones((n, T, 2))
    trans = np.zeros((2, 2))
    for t in range(T - 2, -1, -1):
        active = mask[:, t + 1]
        eb = emission(t + 1) * beta[:, t + 1]
        b = (eb @ A.T) / scale[:, t + 1][:, None]
        beta[:, t] = np.where(active[:, None], b, 1.0)
        xi = alpha[:, t, :, None] * A[None] * eb[:, None, :] / scale[:, t + 1][:, None, None]
        trans += xi[active].sum(0)

    gamma = alpha * beta
    gamma /= gamma.sum(-1, keepdims=True)
    g, o = gamma[mask], obs[mask]
    has_next = np.zeros_like(mask)
    has_next[:, :-1] = mask[:, 1:]
    return {
        "first_known": gamma[:, 0, 1].sum(),
        "n_seq": n,
        "trans": trans,
        "from_state": gamma[has_next].sum(0),  # expected time in each state before a next step
        "guess": np.array([(g[:, 0] * o).sum(), g[:, 0].sum()]),
        "slip": np.array([(g[:, 1] * (1 - o)).sum(), g[:, 1].sum()]),
        "loglik": np.log(scale[mask]).sum(),
    }


def m_step(stats: list[dict], fit_forget: bool):
    s = {k: sum(d[k] for d in stats) for k in stats[0]}
    prior = s["first_known"] / s["n_seq"]
    learn = s["trans"][0, 1] / max(s["from_state"][0], EPS)
    forget = s["trans"][1, 0] / max(s["from_state"][1], EPS) if fit_forget else 0.0
    guess = s["guess"][0] / max(s["guess"][1], EPS)
    slip = s["slip"][0] / max(s["slip"][1], EPS)
    return [prior, learn, guess, slip, forget], s["loglik"]


def fit_skill(chunks, rng, num_fits=5, max_iter=100, tol=1e-4, fit_forget=False, max_guess=0.5, max_slip=0.5):
    best = None
    for _ in range(num_fits):
        params = [rng.uniform(0.1, 0.8), rng.uniform(0.05, 0.4), rng.uniform(0.05, 0.3),
                  rng.uniform(0.05, 0.3), rng.uniform(0.0, 0.1) if fit_forget else 0.0]
        prev = -np.inf
        for _ in range(max_iter):
            params, ll = m_step([e_step(obs, mask, *params) for obs, mask, _ in chunks], fit_forget)
            params = [float(np.clip(params[0], EPS, 1 - EPS)), float(np.clip(params[1], EPS, 1 - EPS)),
                      float(np.clip(params[2], EPS, max_guess)), float(np.clip(params[3], EPS, max_slip)),
                      float(np.clip(params[4], 0.0, 1 - EPS))]
            if ll - prev < tol * max(1.0, abs(ll)):
                break
            prev = ll
        ll = sum(forward(obs, mask, *params)[2] for obs, mask, _ in chunks)
        if best is None or ll > best[1]:
            best = (params, ll)
    return best[0]


class BKT(KTModel):
    name = "bkt"

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> BKT:
        p = self.params
        rng = np.random.default_rng(int(p.get("seed", 0)))
        self.global_rate = float(train["correct"].mean())
        self.skill_params: dict[int, list[float]] = {}
        for skill, chunks in pad_by_skill(explode_skills(train)):
            self.skill_params[skill] = fit_skill(
                chunks, rng,
                num_fits=int(p.get("num_fits", 5)),
                max_iter=int(p.get("max_iter", 100)),
                fit_forget=bool(p.get("forgets", False)),
                max_guess=float(p.get("max_guess", 0.5)),
                max_slip=float(p.get("max_slip", 0.5)),
            )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        ex = explode_skills(df)
        out = np.full(len(ex), self.global_rate)
        for skill, chunks in pad_by_skill(ex):
            if skill not in self.skill_params:
                continue
            for obs, mask, idx in chunks:
                _, pc, _ = forward(obs, mask, *self.skill_params[skill])
                out[idx[mask]] = pc[mask]
        return fuse(ex["row"].to_numpy(), out, len(df))
