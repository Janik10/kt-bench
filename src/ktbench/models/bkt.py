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


def pad_by_skill(ex: pd.DataFrame):
    """Yield (skill, obs, mask, ex_index) with one padded row per (student, skill) sequence."""
    ex = ex.assign(_i=np.arange(len(ex)))
    for skill, g in ex.groupby("skill", sort=True):
        seq = pd.factorize(g["user_id"])[0]
        pos = g.groupby("user_id", sort=False).cumcount().to_numpy()
        obs = np.zeros((seq.max() + 1, pos.max() + 1))
        mask = np.zeros_like(obs, dtype=bool)
        idx = np.full(obs.shape, -1, dtype=np.int64)
        obs[seq, pos] = g["correct"].to_numpy()
        mask[seq, pos] = True
        idx[seq, pos] = g["_i"].to_numpy()
        yield int(skill), obs, mask, idx


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
        like = np.where(o == 1, pc, 1 - pc)
        loglik += np.log(np.clip(like[m], EPS, None)).sum()
        post = np.where(o == 1, known * (1 - slip), known * slip) / np.clip(like, EPS, None)
        nxt = post * (1 - forget) + (1 - post) * learn
        known = np.where(m, nxt, known)
    return p_known, p_correct, loglik


def em_step(obs, mask, prior, learn, guess, slip, forget, fit_forget):
    """One Baum-Welch iteration with per-step scaling. Returns new params and the log-likelihood."""
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
    g = gamma[mask]  # (n_obs, 2)
    o = obs[mask]
    # Expected time in each state at steps that have a next step.
    has_next = np.zeros_like(mask)
    has_next[:, :-1] = mask[:, 1:]
    from_state = gamma[has_next].sum(0)

    new_prior = gamma[:, 0, 1].mean()
    new_learn = trans[0, 1] / max(from_state[0], EPS)
    new_forget = trans[1, 0] / max(from_state[1], EPS) if fit_forget else 0.0
    new_guess = (g[:, 0] * o).sum() / max(g[:, 0].sum(), EPS)
    new_slip = (g[:, 1] * (1 - o)).sum() / max(g[:, 1].sum(), EPS)
    loglik = np.log(scale[mask]).sum()
    return new_prior, new_learn, new_guess, new_slip, new_forget, loglik


def fit_skill(obs, mask, rng, num_fits=5, max_iter=100, tol=1e-4, fit_forget=False,
              max_guess=0.5, max_slip=0.5):
    best = None
    for _ in range(num_fits):
        params = [rng.uniform(0.1, 0.8), rng.uniform(0.05, 0.4), rng.uniform(0.05, 0.3),
                  rng.uniform(0.05, 0.3), rng.uniform(0.0, 0.1) if fit_forget else 0.0]
        prev = -np.inf
        for _ in range(max_iter):
            *params, ll = em_step(obs, mask, *params, fit_forget)
            params[0] = float(np.clip(params[0], EPS, 1 - EPS))
            params[1] = float(np.clip(params[1], EPS, 1 - EPS))
            params[2] = float(np.clip(params[2], EPS, max_guess))
            params[3] = float(np.clip(params[3], EPS, max_slip))
            params[4] = float(np.clip(params[4], 0.0, 1 - EPS))
            if ll - prev < tol * max(1.0, abs(ll)):
                break
            prev = ll
        _, _, ll = forward(obs, mask, *params)
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
        for skill, obs, mask, _ in pad_by_skill(explode_skills(train)):
            self.skill_params[skill] = fit_skill(
                obs, mask, rng,
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
        for skill, obs, mask, idx in pad_by_skill(ex):
            if skill in self.skill_params:
                _, pc, _ = forward(obs, mask, *self.skill_params[skill])
                out[idx[mask]] = pc[mask]
        return fuse(ex["row"].to_numpy(), out, len(df))
