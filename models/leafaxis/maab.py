"""Multi-Axis Attention Block (MAAB).

MAAB computes axial attention along three axes of a token grid — height, width and
channel — and combines them with a learned 3-way gate. For input x of shape
(B, N, C) with N = H * W, this module returns a tensor of the same shape.

The axial-attention construction follows Wang et al. 2020 (axial-DeepLab) and the
per-axis gating follows the multi-axis form of Tu et al. 2022 (MaxViT); the learned
gate here is a softmax over a 3-vector logit, one logit per axis, shared across the
batch but learned per block. The block ships bias-free since positional bias is
supplied externally by the backbone's relative-bias table.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _AxialAttention(nn.Module):
    """Standard scaled-dot-product self-attention restricted to a single axis.

    axis=0: attention along the H axis (tokens that share the same w attend to each other)
    axis=1: attention along the W axis
    axis=2: channel attention (tokens attend across channels of a per-token projection)
    """

    def __init__(self, dim: int, num_heads: int = 4, axis: int = 0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.axis = axis
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        assert N == H * W, f"token count {N} != H*W = {H*W}"
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B, heads, N, head_dim)

        if self.axis == 0:
            # Reshape tokens so each "sequence" is a column of length H.
            q = q.reshape(B, self.num_heads, H, W, self.head_dim).permute(0, 1, 3, 2, 4)
            k = k.reshape(B, self.num_heads, H, W, self.head_dim).permute(0, 1, 3, 2, 4)
            v = v.reshape(B, self.num_heads, H, W, self.head_dim).permute(0, 1, 3, 2, 4)
            # Shape now: (B, heads, W, H, head_dim) — attend along H for each W.
            attn = (q @ k.transpose(-2, -1)) * self.scale
            out = attn.softmax(-1) @ v
            out = out.permute(0, 1, 3, 2, 4).reshape(B, self.num_heads, N, self.head_dim)
        elif self.axis == 1:
            q = q.reshape(B, self.num_heads, H, W, self.head_dim)
            k = k.reshape(B, self.num_heads, H, W, self.head_dim)
            v = v.reshape(B, self.num_heads, H, W, self.head_dim)
            # Shape: (B, heads, H, W, head_dim) — attend along W for each H.
            attn = (q @ k.transpose(-2, -1)) * self.scale
            out = attn.softmax(-1) @ v
            out = out.reshape(B, self.num_heads, N, self.head_dim)
        elif self.axis == 2:
            # Channel attention: scores over head_dim.
            attn = (q.transpose(-2, -1) @ k) * (N ** -0.5)
            out = (v @ attn.softmax(-1).transpose(-2, -1))
        else:
            raise ValueError(f"axis must be 0, 1 or 2; got {self.axis}")

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class MAAB(nn.Module):
    """Multi-Axis Attention Block with a learned 3-way axis gate.

    Forward:
        x_out = x + gate_H * Attn_H(x) + gate_W * Attn_W(x) + gate_C * Attn_C(x)
    where (gate_H, gate_W, gate_C) = softmax(theta) for a learned theta ∈ R^3.
    """

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn_h = _AxialAttention(dim, num_heads, axis=0)
        self.attn_w = _AxialAttention(dim, num_heads, axis=1)
        self.attn_c = _AxialAttention(dim, num_heads, axis=2)
        # Logits of the softmax gate. Shape (3,). Initialised uniform (log 1/3 each).
        self.axis_logits = nn.Parameter(torch.zeros(3))

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def axis_gate(self) -> torch.Tensor:
        return F.softmax(self.axis_logits, dim=0)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        gH, gW, gC = self.axis_gate().unbind(0)
        h = self.norm1(x)
        x = x + gH * self.attn_h(h, H, W) + gW * self.attn_w(h, H, W) + gC * self.attn_c(h, H, W)
        x = x + self.mlp(self.norm2(x))
        return x
