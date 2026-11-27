"""Learned-Score Token Selection (LSTS).

Scores every spatial token with a lightweight MLP, selects the top-k tokens by
score and passes those through to the next stage. The straight-through estimator
of Bengio, Léonard & Courville (2013) is used to propagate gradients through the
hard selection mask so the scoring MLP can be trained end-to-end.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTS(nn.Module):
    """Learned-Score Token Selection.

    Inputs:
        tokens : (B, N, C)
    Returns:
        kept_tokens : (B, k, C)   # k = ceil(keep_ratio * N)
        indices     : (B, k)      # indices into the original token sequence
        scores      : (B, N)      # per-token score, for analysis/regularisation
    """

    def __init__(self, dim: int, keep_ratio: float = 0.7, hidden: int = 64):
        super().__init__()
        if not 0 < keep_ratio <= 1:
            raise ValueError(f"keep_ratio must be in (0, 1]; got {keep_ratio}")
        self.keep_ratio = keep_ratio
        self.scorer = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = tokens.shape
        k = max(1, int(round(self.keep_ratio * N)))

        logits = self.scorer(tokens).squeeze(-1)   # (B, N)
        scores = torch.sigmoid(logits)

        # Hard top-k mask with a straight-through estimator.
        topk_vals, topk_idx = torch.topk(scores, k=k, dim=-1)
        hard_mask = torch.zeros_like(scores)
        hard_mask.scatter_(1, topk_idx, 1.0)
        mask_ste = hard_mask + scores - scores.detach()      # straight-through
        gated = tokens * mask_ste.unsqueeze(-1)              # (B, N, C)

        # Gather the kept tokens in their original order.
        topk_idx_sorted, _ = topk_idx.sort(dim=-1)
        gather_idx = topk_idx_sorted.unsqueeze(-1).expand(-1, -1, C)
        kept = torch.gather(gated, 1, gather_idx)
        return kept, topk_idx_sorted, scores
