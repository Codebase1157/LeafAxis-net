"""Regenerate every supplementary table from the manifest and per-seed predictions.

Outputs are written to results/tables/. The CSVs below are what the paper's
Supplementary Tables S1-S4 should display; any discrepancy with the typeset paper
means the typeset version is out of date and must be updated from these files.

Dependencies:
    manifests/main_manifest.csv              (from scripts/make_splits.py)
    manifests/resplit_<seed>.csv             (from scripts/make_splits.py)
    predictions/<model>_<ds>_seed<seed>.npz  (from scripts/train.py)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from statistical_analysis import bootstrap_f1_ci  # noqa: E402


DATASETS = ("SLD", "DK", "BD")
TEST_N = {"SLD": 1139, "DK": 490, "BD": 126}  # for sanity-check printouts only
SEEDS = (0, 1, 2, 3, 4)

MODELS = (
    "resnet50", "efficientnet_b3", "vit_b_16", "swin_t", "convnext_t",
    "mobilevit_s", "efficientformer_l1", "fastvit_t12", "coatnet_0",
    "rema_mobilevit", "dual_branch", "leafaxis_net",
)


def _resplit_files(manifests_dir: Path) -> list[Path]:
    return sorted(manifests_dir.glob("resplit_*.csv"))


def supp_S1(manifests_dir: Path, out_dir: Path) -> None:
    """Per-class and per-component counts for the main partition of each dataset,
    plus spread of test-image counts across the five re-splits."""
    df = pd.read_csv(manifests_dir / "main_manifest.csv")
    rs_dfs = [pd.read_csv(p) for p in _resplit_files(manifests_dir)]

    rows = []
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        for cls_name, cls_sub in sub.groupby("class_name"):
            parts = cls_sub["partition"].value_counts().to_dict()
            comp_parts = (
                cls_sub.groupby("component")["partition"].first().value_counts().to_dict()
            )
            test_counts_rs = []
            for rs in rs_dfs:
                rs_sub = rs[(rs["dataset"] == ds) & (rs["class_name"] == cls_name)]
                test_counts_rs.append(int((rs_sub["partition"] == "test").sum()))
            rows.append({
                "dataset": ds,
                "class": cls_name,
                "train_images": parts.get("train", 0),
                "val_images": parts.get("val", 0),
                "test_images": parts.get("test", 0),
                "train_components": comp_parts.get("train", 0),
                "val_components": comp_parts.get("val", 0),
                "test_components": comp_parts.get("test", 0),
                "test_images_resplit_mean": float(np.mean(test_counts_rs)) if test_counts_rs else float("nan"),
                "test_images_resplit_sd": float(np.std(test_counts_rs, ddof=1)) if len(test_counts_rs) > 1 else float("nan"),
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "supp_S1.csv", index=False)
    print(f"wrote {out_dir / 'supp_S1.csv'}")


def _load_pred(preds_dir: Path, model: str, ds: str, seed: int) -> dict | None:
    p = preds_dir / f"{model}_{ds}_seed{seed}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def supp_S2(preds_dir: Path, manifests_dir: Path, out_dir: Path) -> None:
    """Per-class support/precision/recall/F1 of LeafAxis-Net with bootstrap 95% CI on F1."""
    df = pd.read_csv(manifests_dir / "main_manifest.csv")
    rows = []
    for ds in DATASETS:
        # Mean per-class precision/recall/F1 across seeds, and bootstrap CI on F1.
        pred_arrs: list[np.ndarray] = []
        label_arrs: list[np.ndarray] = []
        for s in SEEDS:
            d = _load_pred(preds_dir, "leafaxis_net", ds, s)
            if d is None:
                continue
            pred_arrs.append(d["preds"])
            label_arrs.append(d["labels"])
        if not pred_arrs:
            print(f"[skip] S2 {ds}: no predictions found")
            continue

        sub = df[df["dataset"] == ds]
        name_of = sub.drop_duplicates("label").set_index("label")["class_name"].to_dict()
        classes = sorted(name_of)

        for c in classes:
            precs, recs, f1s = [], [], []
            for p, l in zip(pred_arrs, label_arrs):
                tp = int(((p == c) & (l == c)).sum())
                fp = int(((p == c) & (l != c)).sum())
                fn = int(((p != c) & (l == c)).sum())
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
                precs.append(prec); recs.append(rec); f1s.append(f1)
            # Bootstrap CI on F1 using the first seed's predictions.
            f1_point, f1_lo, f1_hi = bootstrap_f1_ci(label_arrs[0], pred_arrs[0], class_id=c)
            support = int((label_arrs[0] == c).sum())
            rows.append({
                "dataset": ds,
                "class": name_of[c],
                "support": support,
                "precision_mean": 100 * float(np.mean(precs)),
                "precision_sd": 100 * float(np.std(precs, ddof=1)),
                "recall_mean": 100 * float(np.mean(recs)),
                "recall_sd": 100 * float(np.std(recs, ddof=1)),
                "f1_mean": 100 * float(np.mean(f1s)),
                "f1_sd": 100 * float(np.std(f1s, ddof=1)),
                "f1_ci_low": 100 * f1_lo,
                "f1_ci_high": 100 * f1_hi,
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "supp_S2.csv", index=False)
    print(f"wrote {out_dir / 'supp_S2.csv'}")


def supp_S3(preds_dir: Path, out_dir: Path) -> None:
    """Per-seed test accuracy (%) for every method on every dataset.

    Each cell is derived from the integer count of correct predictions in the
    .npz prediction file; S3 cannot disagree with any per-seed count reported
    in the main text because both come from this same file.
    """
    rows = []
    for model in MODELS:
        for ds in DATASETS:
            row = {"method": model, "dataset": ds}
            for s in SEEDS:
                d = _load_pred(preds_dir, model, ds, s)
                if d is None:
                    row[f"seed{s}_acc"] = None
                    row[f"seed{s}_correct"] = None
                    row[f"seed{s}_total"] = None
                else:
                    correct = int(d["correct"])
                    total = int(d["total"])
                    row[f"seed{s}_acc"] = 100 * correct / total if total else None
                    row[f"seed{s}_correct"] = correct
                    row[f"seed{s}_total"] = total
            accs = [row[f"seed{s}_acc"] for s in SEEDS if row[f"seed{s}_acc"] is not None]
            if accs:
                row["mean"] = float(np.mean(accs))
                row["sd"] = float(np.std(accs, ddof=1)) if len(accs) > 1 else float("nan")
            rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "supp_S3.csv", index=False)
    print(f"wrote {out_dir / 'supp_S3.csv'}")


def supp_S4(preds_dir: Path, out_dir: Path) -> None:
    """Cross-dataset transfer macro recall, restricted vs unrestricted variants.

    Requires per-seed predictions obtained by running each model trained on one
    dataset against the other dataset's test partition; see scripts/evaluate_transfer.py.
    """
    transfer_dir = preds_dir / "transfer"
    if not transfer_dir.exists():
        print(f"[skip] S4: {transfer_dir} does not exist (run scripts/evaluate_transfer.py first)")
        return
    # Expected files: <model>_<src>_to_<tgt>_seed<s>.npz with keys macro_recall_restricted,
    # macro_recall_unrestricted. Aggregate.
    rows: dict[tuple[str, str], dict] = {}
    for f in transfer_dir.glob("*.npz"):
        name = f.stem
        try:
            model, _rest = name.split("_", 1)
            src, _, after = _rest.partition("_to_")
            tgt, _, seed_tag = after.partition("_seed")
        except ValueError:
            continue
        d = dict(np.load(f))
        key = (model, f"{src}_to_{tgt}")
        rows.setdefault(key, {"restricted": [], "unrestricted": []})
        rows[key]["restricted"].append(float(d["macro_recall_restricted"]))
        rows[key]["unrestricted"].append(float(d["macro_recall_unrestricted"]))

    out_rows = []
    for (model, direction), d in rows.items():
        out_rows.append({
            "method": model,
            "direction": direction,
            "restricted_mean": 100 * float(np.mean(d["restricted"])),
            "restricted_sd": 100 * float(np.std(d["restricted"], ddof=1)) if len(d["restricted"]) > 1 else float("nan"),
            "unrestricted_mean": 100 * float(np.mean(d["unrestricted"])),
            "unrestricted_sd": 100 * float(np.std(d["unrestricted"], ddof=1)) if len(d["unrestricted"]) > 1 else float("nan"),
            "delta": 100 * (float(np.mean(d["unrestricted"])) - float(np.mean(d["restricted"]))),
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_dir / "supp_S4.csv", index=False)
    print(f"wrote {out_dir / 'supp_S4.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default=Path("manifests"), type=Path)
    ap.add_argument("--predictions", default=Path("predictions"), type=Path)
    ap.add_argument("--out", default=Path("results/tables"), type=Path)
    args = ap.parse_args()

    supp_S1(args.manifests, args.out)
    supp_S2(args.predictions, args.manifests, args.out)
    supp_S3(args.predictions, args.out)
    supp_S4(args.predictions, args.out)


if __name__ == "__main__":
    main()
