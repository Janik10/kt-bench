"""Model 4: Deep Knowledge Tracing (Piech et al., 2015).

An LSTM reads the answer history (one embedded "skill s, right/wrong" token
per step) and a linear layer maps its state to a logit for every skill. The
logit for the skill actually asked is the prediction. Position i reads the
LSTM output at ``ctx[i]``, which has seen only answers to earlier questions.
"""

from __future__ import annotations

import torch
from torch import nn

from ktbench.models.neural import SequenceModel, interaction_tokens


class DKTNet(nn.Module):
    def __init__(self, n_skills: int, emb_size: int, hidden_size: int, n_layers: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(2 * n_skills + 1, emb_size)  # 0 = BOS
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, n_skills)

    def forward(self, skill: torch.Tensor, correct: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(self.emb(interaction_tokens(skill, correct)))
        h = self.drop(h)
        # h[:, j] has read BOS and answers 0..j-1; take it at each position's question start.
        h = torch.gather(h, 1, ctx.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        # Equivalent to out(h)[..., skill] without materialising the (B, L, n_skills) tensor.
        w = self.out.weight[skill]
        return (h * w).sum(-1) + self.out.bias[skill]


class DKT(SequenceModel):
    name = "dkt"
    eval_max_len = None  # an LSTM can read each test student's whole history

    def build_net(self) -> nn.Module:
        p = self.params
        return DKTNet(
            self.n_skills,
            emb_size=int(p.get("emb_size", 200)),
            hidden_size=int(p.get("hidden_size", 200)),
            n_layers=int(p.get("n_layers", 1)),
            dropout=float(p.get("dropout", 0.2)),
        )
