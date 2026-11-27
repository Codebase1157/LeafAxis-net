"""PlantVillage pre-training entry point.

Reads a YAML config, builds the model with its PlantVillage class count, trains
for the configured number of epochs with AdamW + cosine schedule, and writes a
checkpoint to `checkpoint.save_path`. The checkpoint stores:
    {
        'state_dict': model.state_dict(),
        'config': cfg,
        'val_top1': best_val_top1,
    }
`scripts/hash_artifacts.py` hashes the file bytes and records the SHA-256 in
REPRODUCE.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.leafaxis import LeafAxisConfig, LeafAxisNet  # noqa: E402
from preprocessing.dataset import train_transform, eval_transform  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 0))

    root = Path(cfg["data"]["root"])
    train_ds = ImageFolder(root / "train", transform=train_transform(cfg["data"]["image_size"]))
    val_ds = ImageFolder(root / "val", transform=eval_transform(cfg["data"]["image_size"]))
    train_dl = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
                          num_workers=cfg["data"]["num_workers"], pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
                        num_workers=cfg["data"]["num_workers"], pin_memory=True)

    model = LeafAxisNet(LeafAxisConfig(num_classes=cfg["model"]["num_classes"])).to(device)
    opt = AdamW(model.parameters(), lr=cfg["optim"]["lr"],
                weight_decay=cfg["optim"]["weight_decay"],
                betas=tuple(cfg["optim"]["betas"]))
    sched = CosineAnnealingLR(opt, T_max=cfg["schedule"]["epochs"],
                              eta_min=cfg["schedule"]["cosine_eta_min"])
    ce = nn.CrossEntropyLoss()

    best = 0.0
    for epoch in range(cfg["schedule"]["epochs"]):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            ce(model(xb), yb).backward()
            opt.step()
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(1) == yb).sum().item()
                total += yb.size(0)
        acc = correct / max(total, 1)
        print(f"epoch {epoch + 1:3d}  val_top1={acc:.4f}")

        if acc > best:
            best = acc
            save_path = Path(cfg["checkpoint"]["save_path"])
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "config": cfg, "val_top1": best}, save_path)

    print(f"best val top-1 = {best:.4f}; checkpoint at {cfg['checkpoint']['save_path']}")


if __name__ == "__main__":
    main()
