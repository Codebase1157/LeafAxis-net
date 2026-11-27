"""Cross-Branch Fusion (CBF).

Fuses two parallel feature streams (CNN branch and ViT branch) at a given stage
via a learned per-channel gate. The gate is a sigmoid over a small linear
projection of the concatenated global-average-pooled features; it allows the
model to shift weight between branches without constraining either.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CBF(nn.Module):
    """Cross-Branch Fusion.

    Inputs:
        cnn_feat  : (B, C, H, W)
        vit_feat  : (B, C, H, W)   # must share spatial size with cnn_feat
    Output:
        fused     : (B, C, H, W) = g * cnn_feat + (1 - g) * vit_feat
    where g has shape (B, C, 1, 1) and comes from a lightweight gate MLP.
    """

    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )

    def forward(self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor) -> torch.Tensor:
        if cnn_feat.shape != vit_feat.shape:
            raise ValueError(
                f"CBF expects matching shapes; got {tuple(cnn_feat.shape)} and {tuple(vit_feat.shape)}"
            )
        B, C, _, _ = cnn_feat.shape
        gap_cnn = cnn_feat.mean(dim=(-2, -1))
        gap_vit = vit_feat.mean(dim=(-2, -1))
        g = self.gate(torch.cat([gap_cnn, gap_vit], dim=-1)).view(B, C, 1, 1)
        return g * cnn_feat + (1 - g) * vit_feat

    def gate_magnitude(self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor) -> torch.Tensor:
        """For analysis: mean |g - 0.5| across the batch and channels."""
        B, C, _, _ = cnn_feat.shape
        gap_cnn = cnn_feat.mean(dim=(-2, -1))
        gap_vit = vit_feat.mean(dim=(-2, -1))
        g = self.gate(torch.cat([gap_cnn, gap_vit], dim=-1)).view(B, C)
        return (g - 0.5).abs().mean()
