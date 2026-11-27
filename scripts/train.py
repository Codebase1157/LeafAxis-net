"""Train one (model, dataset, seed) combination and save per-seed predictions.

This is the single-run entry point. `scripts/train_all.sh` loops over all
12 models × 3 datasets × 5 seeds (180 runs) and calls this script.

For each run, we write:
    checkpoints/<model>_<dataset>_seed<seed>.pth    best val top-1 checkpoint
    logs/<model>_<dataset>_seed<seed>.jsonl         one epoch per line
    predictions/<model>_<dataset>_seed<seed>.npz    final test predictions
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import LeafAxisNet, LeafAxisConfig, build_baseline, BASELINES  # noqa: E402
from preprocessing.dataset import ManifestDataset, train_transform, eval_transform  # noqa: E402


def _load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    inc = cfg.pop("include", None)
    if inc:
        base = yaml.safe_load((path.parent / inc).read_text())
        # Shallow-merge overrides.
        def merge(a: dict, b: dict) -> dict:
            out = dict(a)
            for k, v in b.items():
                if isinstance(v, dict) and isinstance(out.get(k), dict):
                    out[k] = merge(out[k], v)
                else:
                    out[k] = v
            return out
        cfg = merge(base, cfg)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict) -> nn.Module:
    name = cfg["model"]["arch"]
    nc = cfg["data"]["num_classes"]
    if name == "leafaxis_net":
        return LeafAxisNet(LeafAxisConfig(num_classes=nc))
    if name in BASELINES:
        return build_baseline(name, num_classes=nc, pretrained=cfg["model"].get("pretrained", True))
    raise KeyError(f"Unknown model arch {name!r}")


def class_balanced_weights(labels: np.ndarray, beta: float) -> np.ndarray:
    """Cui et al. 2019 effective-number class weights."""
    counts = np.bincount(labels)
    eff = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / np.maximum(eff, 1e-12)
    return w / w.mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--seed", required=True, type=int)
    args = ap.parse_args()
    cfg = _load_config(args.config)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_name = cfg["data"]["dataset"]
    size = cfg["data"]["image_size"]

    train_ds = ManifestDataset(cfg["data"]["manifest"], ds_name, "train", transform=train_transform(size))
    val_ds = ManifestDataset(cfg["data"]["manifest"], ds_name, "val", transform=eval_transform(size))
    test_ds = ManifestDataset(cfg["data"]["manifest"], ds_name, "test", transform=eval_transform(size))

    kw = dict(batch_size=cfg["data"]["batch_size"], num_workers=cfg["data"]["num_workers"], pin_memory=True)
    train_dl = DataLoader(train_ds, shuffle=True, **kw)
    val_dl = DataLoader(val_ds, shuffle=False, **kw)
    test_dl = DataLoader(test_ds, shuffle=False, **kw)

    model = build_model(cfg).to(device)

    # Load PlantVillage pre-trained weights if present (ignore head).
    pre = cfg.get("pretrained_checkpoint")
    if pre and Path(pre).exists():
        state = torch.load(pre, map_location="cpu")["state_dict"]
        state = {k: v for k, v in state.items() if not k.startswith("head.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"loaded pretrained from {pre} (missing={len(missing)}, unexpected={len(unexpected)})")

    opt = AdamW(model.parameters(), lr=cfg["optim"]["lr"],
                weight_decay=cfg["optim"]["weight_decay"], betas=tuple(cfg["optim"]["betas"]))
    sched = CosineAnnealingLR(opt, T_max=cfg["schedule"]["epochs"],
                              eta_min=cfg["schedule"]["cosine_eta_min"])

    labels = np.array(train_ds.df["label"].to_list())
    cb_w = torch.tensor(class_balanced_weights(labels, cfg["loss"]["class_balanced_beta"]),
                        dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cb_w)

    run_tag = f"{cfg['model']['arch']}_{ds_name}_seed{args.seed}"
    log_path = Path("logs") / f"{run_tag}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path("checkpoints") / f"{run_tag}.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = 0.0
    with log_path.open("w") as logf:
        for epoch in range(cfg["schedule"]["epochs"]):
            model.train()
            running = 0.0
            for xb, yb, _ in train_dl:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                opt.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                opt.step()
                running += loss.item() * xb.size(0)
            sched.step()

            model.eval()
            v_correct = v_total = 0
            with torch.no_grad():
                for xb, yb, _ in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    v_correct += (model(xb).argmax(1) == yb).sum().item()
                    v_total += yb.size(0)
            val_acc = v_correct / max(v_total, 1)

            logf.write(json.dumps({"epoch": epoch + 1, "train_loss": running / len(train_ds),
                                   "val_top1": val_acc, "lr": sched.get_last_lr()[0]}) + "\n")
            if val_acc > best_val:
                best_val = val_acc
                torch.save({"state_dict": model.state_dict(), "config": cfg,
                            "seed": args.seed, "val_top1": val_acc}, ckpt_path)

    # Test with the best checkpoint.
    state = torch.load(ckpt_path, map_location=device)["state_dict"]
    model.load_state_dict(state)
    model.eval()
    all_logits = []
    all_labels = []
    all_components = []
    with torch.no_grad():
        for xb, yb, comp in test_dl:
            xb = xb.to(device)
            all_logits.append(model(xb).cpu().numpy())
            all_labels.append(yb.numpy())
            all_components.append(comp.numpy())
    logits = np.concatenate(all_logits)
    labels_arr = np.concatenate(all_labels)
    comps_arr = np.concatenate(all_components)
    preds = logits.argmax(1)
    correct = int((preds == labels_arr).sum())
    total = int(labels_arr.size)

    pred_path = Path(cfg["eval"]["predictions_dir"]) / f"{run_tag}.npz"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(pred_path, logits=logits, labels=labels_arr, components=comps_arr,
             preds=preds, correct=correct, total=total, seed=args.seed)
    print(f"[{run_tag}] test correct/total = {correct}/{total} = {correct/total:.4f}")


if __name__ == "__main__":
    main()
