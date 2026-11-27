"""Evaluate a model trained on source dataset against the target dataset's test set.

Produces predictions/transfer/<model>_<src>_to_<tgt>_seed<s>.npz with keys:
    logits, labels, preds, components, macro_recall_restricted, macro_recall_unrestricted

Restricted variant: logits are projected to the shared label set before argmax.
Unrestricted variant: argmax over the full source label space; a prediction that
falls outside the shared set counts as an error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import LeafAxisNet, LeafAxisConfig, build_baseline, BASELINES  # noqa: E402
from preprocessing.dataset import ManifestDataset, eval_transform  # noqa: E402


SHARED_CONFIRMED = ("Healthy", "Mosaic", "Yellow")   # confirmed shared categories
SHARED_WITH_PROV = SHARED_CONFIRMED + ("Rust",)      # + provisional rust


def build(cfg: dict, nc: int):
    arch = cfg["model"]["arch"]
    if arch == "leafaxis_net":
        return LeafAxisNet(LeafAxisConfig(num_classes=nc))
    return build_baseline(arch, num_classes=nc, pretrained=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--source-config", required=True, type=Path,
                    help="Training config that produced the checkpoint (for arch + label map).")
    ap.add_argument("--target-dataset", required=True, choices=("SLD", "DK", "BD"))
    ap.add_argument("--manifest", default=Path("manifests/main_manifest.csv"), type=Path)
    ap.add_argument("--out-dir", default=Path("predictions/transfer"), type=Path)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.source_config.read_text())
    src_ds = cfg["data"]["dataset"]
    src_nc = cfg["data"]["num_classes"]

    model = build(cfg, src_nc)
    state = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    manifest = pd.read_csv(args.manifest)
    src_map = (manifest[manifest["dataset"] == src_ds]
               .drop_duplicates("label").set_index("label")["class_name"].to_dict())
    tgt_sub = manifest[(manifest["dataset"] == args.target_dataset) & (manifest["partition"] == "test")]
    tgt_map = tgt_sub.drop_duplicates("label").set_index("label")["class_name"].to_dict()

    ds = ManifestDataset(args.manifest, args.target_dataset, "test", eval_transform(224))
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    all_logits, all_labels, all_comp = [], [], []
    with torch.no_grad():
        for xb, yb, comp in dl:
            all_logits.append(model(xb.to(device)).cpu().numpy())
            all_labels.append(yb.numpy())
            all_comp.append(comp.numpy())
    logits = np.concatenate(all_logits); labels = np.concatenate(all_labels); comps = np.concatenate(all_comp)

    # Map target class names to their source indices.
    def _recall(pred_class_names: np.ndarray, label_class_names: np.ndarray, set_names: tuple[str, ...]) -> float:
        recs = []
        for name in set_names:
            mask = label_class_names == name
            if mask.sum() == 0:
                continue
            recs.append(float((pred_class_names[mask] == name).mean()))
        return float(np.mean(recs)) if recs else float("nan")

    label_names = np.array([tgt_map.get(int(l), "?") for l in labels])
    # Restricted: project logits onto shared classes present in source.
    shared_in_src = {name: idx for idx, name in src_map.items() if name in SHARED_WITH_PROV}
    if not shared_in_src:
        print(f"[warn] source dataset has no shared classes with target; skipping")
        return
    shared_idx = np.array([shared_in_src[n] for n in sorted(shared_in_src, key=lambda n: shared_in_src[n])])
    shared_names_order = np.array(sorted(shared_in_src, key=lambda n: shared_in_src[n]))

    pred_restricted_idx = logits[:, shared_idx].argmax(1)
    pred_restricted_names = shared_names_order[pred_restricted_idx]

    pred_full_idx = logits.argmax(1)
    pred_full_names = np.array([src_map.get(int(i), "__OOV__") for i in pred_full_idx])

    rec_restricted = _recall(pred_restricted_names, label_names, SHARED_CONFIRMED)
    rec_unrestricted = _recall(pred_full_names, label_names, SHARED_CONFIRMED)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{cfg['model']['arch']}_{src_ds}_to_{args.target_dataset}_seed{args.seed}.npz"
    np.savez(out_path, logits=logits, labels=labels, components=comps,
             preds_restricted=pred_restricted_idx, preds_full=pred_full_idx,
             macro_recall_restricted=rec_restricted, macro_recall_unrestricted=rec_unrestricted)
    print(f"wrote {out_path}  restricted={rec_restricted:.4f}  unrestricted={rec_unrestricted:.4f}")


if __name__ == "__main__":
    main()
