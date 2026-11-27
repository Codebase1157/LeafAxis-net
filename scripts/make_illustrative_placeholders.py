"""Generate small illustrative placeholder files for manifests/, predictions/, checkpoints/, logs/.

Every file written here carries an unmistakable in-file marker identifying it as a
placeholder. These files exist so a new user can see the file format and run the
pipeline end-to-end on a small example before launching the full sweep. They are
NOT outputs of real training runs on the sugarcane datasets.

Run `scripts/clear_placeholders.py` to delete them before producing real artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
NOTE = "ILLUSTRATIVE PLACEHOLDER — not from a real training run"

# Keep the demo set tiny: one model family on each dataset, seed 0 only.
DEMO = [
    ("leafaxis_net", "SLD", 0, 11),
    ("leafaxis_net", "DK",  0, 5),
    ("leafaxis_net", "BD",  0, 5),
    ("dual_branch",  "SLD", 0, 11),
]


def write_manifest_rows() -> None:
    path = ROOT / "manifests" / "main_manifest.csv"
    df = pd.read_csv(path)
    rows = []
    for ds, classes in [
        ("SLD", ["Healthy", "Mosaic", "Yellow_leaf"]),
        ("DK",  ["Healthy", "Rust"]),
        ("BD",  ["Healthy", "Red_rot"]),
    ]:
        for cls_idx, cls in enumerate(classes):
            for i in range(2):
                rows.append({
                    "path": f"data_raw/_illustrative_placeholder/{ds}/{cls}/example_{i:04d}.jpg",
                    "dataset": ds,
                    "class_name": cls,
                    "label": cls_idx,
                    "sha256": "0" * 64,       # obviously not a real SHA
                    "phash": "0" * 16,        # obviously not a real pHash
                    "component": -1,          # sentinel for placeholder
                    "partition": ["train", "val", "test"][i % 3],
                })
    out = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(path, index=False)
    print(f"wrote {len(rows)} placeholder rows to {path}")


def write_npz(model: str, ds: str, seed: int, nc: int) -> None:
    out = ROOT / "predictions" / f"{model}_{ds}_seed{seed}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 8  # demo size only
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((n, nc)).astype(np.float32)
    labels = rng.integers(0, nc, size=n).astype(np.int64)
    preds = logits.argmax(1).astype(np.int64)
    components = np.arange(n).astype(np.int64)
    np.savez(
        out,
        illustrative_placeholder=np.array(True),
        _note=np.array(NOTE),
        logits=logits,
        labels=labels,
        preds=preds,
        components=components,
        correct=np.array(int((preds == labels).sum())),
        total=np.array(n),
        seed=np.array(seed),
    )
    print(f"wrote placeholder {out}")


def write_pth(model: str, ds: str, seed: int, nc: int) -> None:
    out = ROOT / "checkpoints" / f"{model}_{ds}_seed{seed}.pth"
    out.parent.mkdir(parents=True, exist_ok=True)
    # 8x8 random tensor and a tiny head; obviously not a trained model.
    torch.manual_seed(seed)
    tiny_state = {
        "head.weight": torch.randn(nc, 8),
        "head.bias":   torch.zeros(nc),
    }
    torch.save({
        "illustrative_placeholder": True,
        "_note": NOTE,
        "state_dict": tiny_state,
        "config": {"model": {"arch": model}, "data": {"dataset": ds, "num_classes": nc}},
        "seed": seed,
        "val_top1": None,
    }, out)
    print(f"wrote placeholder {out}")


def write_jsonl(model: str, ds: str, seed: int) -> None:
    out = ROOT / "logs" / f"{model}_{ds}_seed{seed}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        {"illustrative_placeholder": True, "_note": NOTE},
        {"epoch": 1, "train_loss": None, "val_top1": None, "lr": None,
         "_note": "placeholder — no training was performed"},
        {"epoch": 2, "train_loss": None, "val_top1": None, "lr": None,
         "_note": "placeholder — no training was performed"},
    ]
    out.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    print(f"wrote placeholder {out}")


def write_plantvillage_ckpt() -> None:
    out = ROOT / "checkpoints" / "plantvillage_leafaxis.pth"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "illustrative_placeholder": True,
        "_note": NOTE,
        "state_dict": {"head.weight": torch.zeros(38, 8), "head.bias": torch.zeros(38)},
        "config": {"note": "placeholder PlantVillage pre-train checkpoint"},
        "val_top1": None,
    }, out)
    print(f"wrote placeholder {out}")


def main() -> None:
    write_manifest_rows()
    write_plantvillage_ckpt()
    for model, ds, seed, nc in DEMO:
        write_npz(model, ds, seed, nc)
        write_pth(model, ds, seed, nc)
        write_jsonl(model, ds, seed)


if __name__ == "__main__":
    main()
