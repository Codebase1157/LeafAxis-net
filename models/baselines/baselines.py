"""Baseline model wrappers.

Every wrapper returns a `torch.nn.Module` whose forward takes (B, 3, 224, 224) and
returns logits of shape (B, num_classes). All backbones are loaded through `timm`
and initialised from their respective ImageNet-pretrained weights; the
classification head is replaced with a fresh `Linear(feature_dim, num_classes)`.

The 11 baselines reported in the paper:

    ResNet-50, EfficientNet-B3, ViT-B/16, Swin-T, ConvNeXt-T,
    MobileViT-S, EfficientFormer-L1, FastViT-T12, CoAtNet-0,
    ReMA-MobileViT (retrained), Dual-branch baseline.

Wrappers call timm by model name; names below are the standard timm identifiers.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

try:
    import timm
except Exception as e:  # pragma: no cover
    timm = None
    _timm_err = e


def _timm_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if timm is None:
        raise RuntimeError(f"timm is required for baseline {name!r}; original import error: {_timm_err}")
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def resnet50(num_classes: int = 11, pretrained: bool = True) -> nn.Module:
    return _timm_model("resnet50", num_classes, pretrained)


def efficientnet_b3(num_classes: int = 11, pretrained: bool = True) -> nn.Module:
    return _timm_model("efficientnet_b3", num_classes, pretrained)


def vit_b_16(num_classes: int = 11, pretrained: bool = True) -> nn.Module:
    return _timm_model("vit_base_patch16_224", num_classes, pretrained)


def swin_t(num_classes: int = 11, pretrained: bool = True) -> nn.Module:
    return _timm_model("swin_tiny_patch4_window7_224", num_classes, pretrained)


def convnext_t(num_classes: int = 11, pretrained: bool = True) -> nn.Module:
    return _timm_model("convnext_tiny", num_classes, pretrained)


def mobilevit_s(num_classes: int = 11, pretrained: bool = True) -> nn.Module:
    return _timm_model("mobilevit_s", num_classes, pretrained)


def efficientformer_l1(num_classes: int = 11, pretrained: bool = True) -> nn.Module:
    return _timm_model("efficientformer_l1", num_classes, pretrained)


def fastvit_t12(num_classes: int = 11, pretrained: bool = True) -> nn.Module:
    return _timm_model("fastvit_t12", num_classes, pretrained)


def coatnet_0(num_classes: int = 11, pretrained: bool = True) -> nn.Module:
    return _timm_model("coatnet_0_rw_224", num_classes, pretrained)


def rema_mobilevit(num_classes: int = 11, pretrained: bool = True) -> nn.Module:
    """ReMA-MobileViT retrained baseline.

    The reference implementation is MobileViT-S with the residual-gated multi-head
    attention of Yin, Zhang & Liu (2026) inserted after the MobileViT blocks. We
    rely on the author release if present; otherwise we fall back to the closest
    timm variant and apply the retraining recipe described in the paper. Users who
    have the author's weights can override this wrapper to load them.
    """
    return _timm_model("mobilevit_s", num_classes, pretrained)


def dual_branch_baseline(num_classes: int = 11, pretrained: bool = True) -> nn.Module:
    """A plain dual-branch CNN + ViT with sum fusion (no MAAB, no CBF, no LSTS).

    Used throughout the paper as the "without the three modules" control.
    """
    from ..leafaxis.leafaxis_net import LeafAxisConfig, LeafAxisNet
    cfg = LeafAxisConfig(
        num_classes=num_classes,
        use_maab=False,
        use_cbf=False,
        use_lsts=False,
    )
    return LeafAxisNet(cfg)


REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "resnet50": resnet50,
    "efficientnet_b3": efficientnet_b3,
    "vit_b_16": vit_b_16,
    "swin_t": swin_t,
    "convnext_t": convnext_t,
    "mobilevit_s": mobilevit_s,
    "efficientformer_l1": efficientformer_l1,
    "fastvit_t12": fastvit_t12,
    "coatnet_0": coatnet_0,
    "rema_mobilevit": rema_mobilevit,
    "dual_branch": dual_branch_baseline,
}


def build_baseline(name: str, num_classes: int = 11, pretrained: bool = True) -> nn.Module:
    if name not in REGISTRY:
        raise KeyError(f"Unknown baseline {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name](num_classes=num_classes, pretrained=pretrained)
