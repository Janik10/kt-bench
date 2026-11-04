"""Model 5: Self-Attentive Knowledge Tracing (Pandey & Karypis, 2019).

Queries are embeddings of the skill being asked; keys and values are
embeddings of past "skill s, right/wrong" answers plus a position embedding.
The attention mask lets position i attend only to key positions <= ctx[i]:
key 0 is BOS and key j+1 is the answer at position j, so position i sees BOS
and the answers to every earlier question, and nothing from its own question
or later.
"""

from __future__ import annotations

import torch
from torch import nn

from ktbench.models.neural import SequenceModel, interaction_tokens


class SAKTBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.norm2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, blocked: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(q, kv, kv, attn_mask=blocked, need_weights=False)
        x = self.norm1(q + self.drop(a))
        return self.norm2(x + self.drop(self.ffn(x)))


class SAKTNet(nn.Module):
    def __init__(self, n_skills: int, d: int, n_heads: int, n_blocks: int, dropout: float, max_len: int):
        super().__init__()
        self.n_heads = n_heads
        self.inter = nn.Embedding(2 * n_skills + 1, d)  # 0 = BOS
        self.query = nn.Embedding(n_skills, d)
        self.pos = nn.Embedding(max_len, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(SAKTBlock(d, n_heads, dropout) for _ in range(n_blocks))
        self.out = nn.Linear(d, 1)

    def forward(self, skill: torch.Tensor, correct: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, L = skill.shape
        pos = torch.arange(L, device=skill.device)
        kv = self.drop(self.inter(interaction_tokens(skill, correct)) + self.pos(pos))
        x = self.query(skill)
        blocked = pos.view(1, 1, L) > ctx.unsqueeze(-1)  # (B, L, L), True = may not attend
        blocked = blocked.repeat_interleave(self.n_heads, dim=0)
        for block in self.blocks:
            x = block(x, kv, blocked)
        return self.out(x).squeeze(-1)


class SAKT(SequenceModel):
    name = "sakt"

    @property
    def eval_max_len(self) -> int:
        return int(self.params.get("max_len", 200))

    def build_net(self) -> nn.Module:
        p = self.params
        return SAKTNet(
            self.n_skills,
            d=int(p.get("emb_size", 128)),
            n_heads=int(p.get("n_heads", 8)),
            n_blocks=int(p.get("n_blocks", 1)),
            dropout=float(p.get("dropout", 0.2)),
            max_len=int(p.get("max_len", 200)),
        )
