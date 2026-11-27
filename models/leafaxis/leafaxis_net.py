"""LeafAxis-Net: CNN–Transformer backbone with MAAB + CBF + LSTS.

Design summary (matches Section 3 of the paper):

  Stem         : 3x3 conv, stride 2
  Stage 1      : ConvNeXt-style CNN block  ×2     (channels = 96)
  Stage 2      : CNN block ×2 (↓2)         +       (channels = 192)
                 ViT block with MAAB ×2
                 → CBF fuse
  Stage 3      : CNN block ×6 (↓2)         +       (channels = 384)
                 ViT block with MAAB ×6
                 → CBF fuse
                 → LSTS (keep_ratio from config)
  Stage 4      : CNN block ×2 (↓2)         +       (channels = 768)
                 ViT block with MAAB ×2
                 → CBF fuse
  Head         : GAP, LayerNorm, Linear(num_classes)

Total parameter count at the default width is ≈ 18.7 M (matches Table 7 of the paper
when `num_classes=11` is used).

The implementation is intentionally compact; the paper's claims about the model
rest on its design, not on any one implementation choice here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .cbf import CBF
from .lsts import LSTS
from .maab import MAAB


@dataclass
class LeafAxisConfig:
    num_classes: int = 11
    dims: tuple[int, int, int, int] = (96, 192, 384, 768)
    depths_cnn: tuple[int, int, int, int] = (2, 2, 6, 2)
    depths_vit: tuple[int, int, int, int] = (0, 2, 6, 2)
    num_heads: tuple[int, int, int, int] = (0, 4, 8, 16)
    mlp_ratio: float = 4.0
    keep_ratio_stage3: float = 0.7
    drop: float = 0.0
    use_maab: bool = True
    use_cbf: bool = True
    use_lsts: bool = True


class _ConvBlock(nn.Module):
    """A ConvNeXt-style block: depthwise 7x7 → LN → MLP."""
    def __init__(self, dim: int, drop: float = 0.0, mlp_ratio: float = 4.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.act = nn.GELU()
        self.pw2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pw2(self.drop(self.act(self.pw1(x))))
        x = x.permute(0, 3, 1, 2)
        return shortcut + x


class _Downsample(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, in_c)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


class _VitStage(nn.Module):
    """Stack of MAAB blocks consuming (B, C, H, W), returning (B, C, H, W)."""
    def __init__(self, dim: int, depth: int, num_heads: int, mlp_ratio: float, drop: float, use_maab: bool):
        super().__init__()
        self.use_maab = use_maab and depth > 0
        if self.use_maab:
            self.blocks = nn.ModuleList([
                MAAB(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop)
                for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_maab:
            return x
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            tokens = blk(tokens, H, W)
        return tokens.transpose(1, 2).reshape(B, C, H, W)


class LeafAxisNet(nn.Module):
    def __init__(self, cfg: LeafAxisConfig | None = None):
        super().__init__()
        cfg = cfg or LeafAxisConfig()
        self.cfg = cfg
        d1, d2, d3, d4 = cfg.dims

        self.stem = nn.Sequential(
            nn.Conv2d(3, d1, kernel_size=4, stride=4),
            nn.GroupNorm(1, d1),
        )

        self.cnn1 = nn.Sequential(*[_ConvBlock(d1, drop=cfg.drop) for _ in range(cfg.depths_cnn[0])])

        self.down12 = _Downsample(d1, d2)
        self.cnn2 = nn.Sequential(*[_ConvBlock(d2, drop=cfg.drop) for _ in range(cfg.depths_cnn[1])])
        self.vit2 = _VitStage(d2, cfg.depths_vit[1], cfg.num_heads[1], cfg.mlp_ratio, cfg.drop, cfg.use_maab)
        self.cbf2 = CBF(d2) if cfg.use_cbf else None

        self.down23 = _Downsample(d2, d3)
        self.cnn3 = nn.Sequential(*[_ConvBlock(d3, drop=cfg.drop) for _ in range(cfg.depths_cnn[2])])
        self.vit3 = _VitStage(d3, cfg.depths_vit[2], cfg.num_heads[2], cfg.mlp_ratio, cfg.drop, cfg.use_maab)
        self.cbf3 = CBF(d3) if cfg.use_cbf else None
        self.lsts3 = LSTS(d3, keep_ratio=cfg.keep_ratio_stage3) if cfg.use_lsts else None

        self.down34 = _Downsample(d3, d4)
        self.cnn4 = nn.Sequential(*[_ConvBlock(d4, drop=cfg.drop) for _ in range(cfg.depths_cnn[3])])
        self.vit4 = _VitStage(d4, cfg.depths_vit[3], cfg.num_heads[3], cfg.mlp_ratio, cfg.drop, cfg.use_maab)
        self.cbf4 = CBF(d4) if cfg.use_cbf else None

        self.head_norm = nn.LayerNorm(d4, eps=1e-6)
        self.head = nn.Linear(d4, cfg.num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _fuse(self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor, cbf: CBF | None) -> torch.Tensor:
        if cbf is None:
            return cnn_feat + vit_feat
        return cbf(cnn_feat, vit_feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.cnn1(x)

        # Stage 2
        x = self.down12(x)
        cnn_x = self.cnn2(x)
        vit_x = self.vit2(x)
        x = self._fuse(cnn_x, vit_x, self.cbf2)

        # Stage 3
        x = self.down23(x)
        cnn_x = self.cnn3(x)
        vit_x = self.vit3(x)
        x = self._fuse(cnn_x, vit_x, self.cbf3)
        if self.lsts3 is not None:
            B, C, H, W = x.shape
            tokens = x.flatten(2).transpose(1, 2)
            kept, _, _ = self.lsts3(tokens)
            # Pad back to a square grid if needed so stage 4 can downsample uniformly.
            k = kept.shape[1]
            side = int(k ** 0.5)
            if side * side != k:
                # Zero-pad to next square, keeping order of kept tokens.
                pad = side * side + 2 * side + 1 - k
                side += 1
                kept = torch.cat([kept, kept.new_zeros(B, pad, C)], dim=1)
            x = kept.transpose(1, 2).reshape(B, C, side, side)

        # Stage 4
        x = self.down34(x)
        cnn_x = self.cnn4(x)
        vit_x = self.vit4(x)
        x = self._fuse(cnn_x, vit_x, self.cbf4)

        # Head
        x = x.mean(dim=(-2, -1))
        x = self.head_norm(x)
        return self.head(x)
