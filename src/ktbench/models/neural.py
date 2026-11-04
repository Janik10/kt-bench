"""Shared training and prediction loop for the PyTorch sequence models.

A subclass only builds the network. The network maps a padded batch
``(skill, correct, ctx)`` to one logit per position, where position i may
depend only on tokens before ``ctx[i]`` (see ``ktbench.data.sequences``).
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from ktbench import metrics
from ktbench.data import sequences
from ktbench.models.base import KTModel, explode_skills, fuse
from ktbench.seed import set_seed


def interaction_tokens(skill: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
    """Token for "answered skill s, right/wrong", shifted right by one with BOS (= 0) in front.

    Position i of the result describes the answer at position i-1, so a model
    reading it at position i has only seen answers before i.
    """
    tok = 1 + 2 * skill + correct
    bos = torch.zeros_like(tok[:, :1])
    return torch.cat([bos, tok[:, :-1]], dim=1)


class SequenceModel(KTModel):
    name = "sequence"
    eval_max_len: int | None = None  # None: whole sequence in one window

    def build_net(self) -> nn.Module:
        raise NotImplementedError

    def _device(self) -> torch.device:
        want = self.params.get("device", "auto")
        if want == "auto":
            want = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(want)

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> SequenceModel:
        p = self.params
        set_seed(int(p.get("seed", 0)))
        self.device = self._device()
        self.net = self.build_net().to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=float(p.get("lr", 1e-3)))
        loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        max_len = int(p.get("max_len", 200))
        batch_size = int(p.get("batch_size", 64))
        max_epochs = int(p.get("max_epochs", 100))
        patience = int(p.get("patience", 10))
        windows = sequences.train_windows(train, max_len)
        rng = np.random.default_rng(int(p.get("seed", 0)))

        best_auc, best_state, bad_epochs = -1.0, None, 0
        self.history = []
        for epoch in range(max_epochs):
            t0 = time.perf_counter()
            self.net.train()
            total, count = 0.0, 0
            order = rng.permutation(len(windows))
            for b in range(0, len(order), batch_size):
                batch = sequences.collate([windows[i] for i in order[b : b + batch_size]])
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.net(batch["skill"], batch["correct"], batch["ctx"])
                m = batch["mask"]
                loss = loss_fn(logits[m], batch["correct"][m].float())
                opt.zero_grad()
                (loss / m.sum()).backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), float(p.get("grad_clip", 1.0)))
                opt.step()
                total += loss.item()
                count += int(m.sum())
            entry = {"epoch": epoch, "train_loss": total / count, "secs": time.perf_counter() - t0}
            if valid is not None and len(valid):
                entry["valid_auc"] = metrics.auc(valid["correct"].to_numpy(), self.predict(valid))
                if entry["valid_auc"] > best_auc:
                    best_auc, bad_epochs = entry["valid_auc"], 0
                    best_state = copy.deepcopy(self.net.state_dict())
                else:
                    bad_epochs += 1
            self.history.append(entry)
            if p.get("verbose", True):
                print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in entry.items()))
            if bad_epochs >= patience:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self.net.eval()
        stride = self.params.get("eval_stride") or (self.eval_max_len // 2 if self.eval_max_len else None)
        windows = sequences.eval_windows(df, self.eval_max_len, stride)
        windows.sort(key=lambda w: len(w.skill))
        n_ex = int(df["skills"].map(len).sum())
        p_ex = np.full(n_ex, np.nan)
        batch_size = int(self.params.get("eval_batch_size", 128))
        for b in range(0, len(windows), batch_size):
            batch = sequences.collate(windows[b : b + batch_size])
            logits = self.net(
                batch["skill"].to(self.device), batch["correct"].to(self.device), batch["ctx"].to(self.device)
            )
            prob = torch.sigmoid(logits).cpu().numpy()
            tgt = batch["target"].numpy()
            keep = tgt >= 0
            p_ex[tgt[keep]] = prob[keep]
        if np.isnan(p_ex).any():
            raise RuntimeError("some positions were never scored")
        return fuse(explode_skills(df)["row"].to_numpy(), p_ex, len(df))
