"""Profile parameters, GFLOPs and 224×224 batch-1 latency for each trained model.

Writes results/tables/table13.csv. Latency is the median of 100 forward passes
after 10 warm-up iterations on the machine running the script; move to the same
hardware the paper uses before reporting.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import LeafAxisNet, LeafAxisConfig, build_baseline  # noqa: E402

try:
    from fvcore.nn import FlopCountAnalysis
except Exception:
    FlopCountAnalysis = None


MODELS_NUMCLASSES = {
    "resnet50": 5, "efficientnet_b3": 5, "vit_b_16": 5, "swin_t": 5,
    "convnext_t": 5, "mobilevit_s": 5, "efficientformer_l1": 5, "fastvit_t12": 5,
    "coatnet_0": 5, "rema_mobilevit": 5, "leafaxis_net": 5,
}


def profile(model: torch.nn.Module, device: torch.device) -> tuple[float, float, float]:
    model.eval().to(device)
    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(100):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    latency_ms = sorted(times)[len(times) // 2] * 1000
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    if FlopCountAnalysis is not None:
        flops = FlopCountAnalysis(model, x).total() / 1e9
    else:
        flops = float("nan")
    return params_m, flops, latency_ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=Path("results/tables/table13.csv"), type=Path)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for name, nc in MODELS_NUMCLASSES.items():
        if name == "leafaxis_net":
            m = LeafAxisNet(LeafAxisConfig(num_classes=nc))
        else:
            m = build_baseline(name, num_classes=nc, pretrained=False)
        try:
            p, f, l = profile(m, device)
        except Exception as e:
            print(f"[{name}] profile failed: {e}")
            p = f = l = float("nan")
        rows.append({"method": name, "params_m": p, "gflops": f, "latency_ms": l})
        print(f"{name:24s}  params={p:.2f}M  GFLOPs={f:.2f}  latency={l:.2f} ms")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
