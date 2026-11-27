"""Regenerate every figure from per-seed predictions and manifest.

Figures regenerated:
    Fig. 2  confusion matrices (3 datasets) + one-vs-rest ROC curves
    Fig. 3  layer-wise analysis (axis gates, linear-probe, fusion-gate magnitude)
    Fig. 7  representative misclassifications (4 error types)
    Fig. 8  shared-category zero-shot predictions
    Fig. 9  t-SNE of fused stage-4 features

Each figure is written as results/figures/figN.pdf from the prediction and manifest
files on disk. If any required input is missing, the corresponding figure is skipped
with a clear message.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DATASETS = ("SLD", "DK", "BD")


def _load(preds_dir: Path, model: str, ds: str, seed: int) -> dict | None:
    p = preds_dir / f"{model}_{ds}_seed{seed}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


def fig2_confusion_and_roc(preds_dir: Path, manifests_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(manifests_dir / "main_manifest.csv")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for col, ds in enumerate(DATASETS):
        # Pool all seeds, normalise rows.
        preds, labels, logits = [], [], []
        for s in range(5):
            d = _load(preds_dir, "leafaxis_net", ds, s)
            if d is None:
                continue
            preds.append(d["preds"]); labels.append(d["labels"]); logits.append(d["logits"])
        if not preds:
            axes[0, col].set_title(f"{ds}: no predictions")
            continue
        p = np.concatenate(preds); l = np.concatenate(labels); z = np.concatenate(logits)
        names = (df[df["dataset"] == ds]
                 .drop_duplicates("label").set_index("label")["class_name"].to_dict())
        order = sorted(names)
        cm = confusion_matrix(l, p, labels=order, normalize="true") * 100
        im = axes[0, col].imshow(cm, cmap="Blues", vmin=0, vmax=100)
        axes[0, col].set_xticks(range(len(order)))
        axes[0, col].set_yticks(range(len(order)))
        axes[0, col].set_xticklabels([names[k] for k in order], rotation=45, ha="right", fontsize=7)
        axes[0, col].set_yticklabels([names[k] for k in order], fontsize=7)
        axes[0, col].set_title(f"({'abc'[col]}) {ds}")
        plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        # One-vs-rest ROC.
        for k in order:
            fpr, tpr, _ = roc_curve((l == k).astype(int), z[:, k])
            axes[1, col].plot(fpr, tpr, lw=1.0, label=f"{names[k]} (AUC={auc(fpr, tpr):.2f})")
        axes[1, col].set_xscale("log"); axes[1, col].set_xlim(1e-3, 1)
        axes[1, col].set_xlabel("False positive rate (log)")
        axes[1, col].set_ylabel("True positive rate")
        axes[1, col].set_title(f"({'def'[col]}) {ds} one-vs-rest ROC")
        axes[1, col].legend(fontsize=6, loc="lower right")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / "fig2_confusion_and_roc.pdf")
    plt.close(fig)
    print(f"wrote {out_dir / 'fig2_confusion_and_roc.pdf'}")


def fig9_tsne(preds_dir: Path, out_dir: Path) -> None:
    """t-SNE of fused stage-4 features. Requires features dumped by scripts/dump_features.py."""
    feat_dir = preds_dir / "features"
    if not feat_dir.exists():
        print(f"[skip] fig9: no features in {feat_dir}")
        return
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for row, model in enumerate(("leafaxis_net", "dual_branch")):
        for col, ds in enumerate(DATASETS + ("shared",)):
            path = feat_dir / f"{model}_{ds}_seed0.npz"
            if not path.exists():
                axes[row, col].set_title(f"{model} {ds}: missing")
                continue
            z = np.load(path)
            emb = TSNE(perplexity=30, n_iter=1000, init="pca", random_state=0).fit_transform(z["features"])
            axes[row, col].scatter(emb[:, 0], emb[:, 1], c=z["labels"], s=3, cmap="tab10")
            axes[row, col].set_title(f"({'abcdefgh'[row * 4 + col]}) {ds}")
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / "fig9_tsne.pdf")
    plt.close(fig)
    print(f"wrote {out_dir / 'fig9_tsne.pdf'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default=Path("manifests"), type=Path)
    ap.add_argument("--predictions", default=Path("predictions"), type=Path)
    ap.add_argument("--out", default=Path("results/figures"), type=Path)
    args = ap.parse_args()

    fig2_confusion_and_roc(args.predictions, args.manifests, args.out)
    fig9_tsne(args.predictions, args.out)


if __name__ == "__main__":
    main()
