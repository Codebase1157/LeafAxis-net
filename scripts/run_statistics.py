"""Run every paired statistical analysis reported in the paper's Table 10.

Block A: seed-paired t-tests with Holm correction, full model vs each of
         9 controls on each dataset (3 datasets × 9 controls = 27 tests).
Block B: re-split 95% CI of the difference across the 5 re-splits.
Block C: paired cluster bootstrap over the test components (10,000 draws).

Outputs results/tables/table10.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from statistical_analysis import paired_t, holm_correction, cluster_bootstrap_diff  # noqa: E402


DATASETS = ("SLD", "DK", "BD")
SEEDS = (0, 1, 2, 3, 4)
CONTROLS = (
    "width_matched", "cbam_for_maab", "coord_attn_for_maab",
    "random_token_dropping", "magnitude_pruning",
    "fixed_gate_third", "no_positional_prior", "single_scale_prior",
    "learned_bias_table",
)


def _acc(preds_dir: Path, model: str, ds: str, seed: int) -> float | None:
    p = preds_dir / f"{model}_{ds}_seed{seed}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    total = int(z["total"])
    return float(z["correct"]) / total if total else None


def block_a(preds_dir: Path) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        diffs_by_control: dict[str, list[float]] = {}
        for c in CONTROLS:
            diffs = []
            for s in SEEDS:
                full = _acc(preds_dir, "leafaxis_net", ds, s)
                ctrl = _acc(preds_dir, c, ds, s)
                if full is not None and ctrl is not None:
                    diffs.append(full - ctrl)
            if len(diffs) >= 2:
                diffs_by_control[c] = diffs

        # Paired t per control, then Holm across the family.
        results = {c: paired_t(d) for c, d in diffs_by_control.items()}
        if results:
            adj = holm_correction([r.p_raw for r in results.values()])
            for (c, r), p_adj in zip(results.items(), adj):
                rows.append({
                    "block": "A",
                    "dataset": ds,
                    "comparison": f"leafaxis_net - {c}",
                    "mean_diff_pts": 100 * r.mean_diff,
                    "p_raw": r.p_raw,
                    "p_holm": p_adj,
                    "t_stat": r.t_stat,
                })
    return pd.DataFrame(rows)


def block_b(preds_dir: Path, manifests_dir: Path) -> pd.DataFrame:
    """Re-split 95% CI of the paired difference for the strongest control.

    Each re-split has its own predictions directory under predictions/resplit_<seed>/.
    """
    rows = []
    for ds in DATASETS:
        for c in ("strongest_control", "coatnet_0"):  # paper reports these two
            diffs = []
            for rs_seed in (100, 200, 300, 400, 500):
                full = _acc(preds_dir / f"resplit_{rs_seed}", "leafaxis_net", ds, 0)
                ctrl = _acc(preds_dir / f"resplit_{rs_seed}", c, ds, 0)
                if full is not None and ctrl is not None:
                    diffs.append(full - ctrl)
            if len(diffs) >= 2:
                r = paired_t(diffs)
                rows.append({
                    "block": "B",
                    "dataset": ds,
                    "comparison": f"leafaxis_net - {c}",
                    "mean_diff_pts": 100 * r.mean_diff,
                    "ci_low_pts": 100 * r.ci_low,
                    "ci_high_pts": 100 * r.ci_high,
                })
    return pd.DataFrame(rows)


def block_c(preds_dir: Path) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for c in ("strongest_control", "coatnet_0"):
            diffs_ci_lo, diffs_ci_hi, diff_points = [], [], []
            full = np.load(preds_dir / f"leafaxis_net_{ds}_seed0.npz") if (preds_dir / f"leafaxis_net_{ds}_seed0.npz").exists() else None
            ctrl_p = preds_dir / f"{c}_{ds}_seed0.npz"
            if full is None or not ctrl_p.exists():
                continue
            ctrl = np.load(ctrl_p)
            res = cluster_bootstrap_diff(
                full["preds"], ctrl["preds"], full["labels"], full["components"]
            )
            rows.append({
                "block": "C",
                "dataset": ds,
                "comparison": f"leafaxis_net - {c}",
                "mean_diff_pts": 100 * res.mean_diff,
                "ci_low_pts": 100 * res.ci_low,
                "ci_high_pts": 100 * res.ci_high,
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=Path("predictions"), type=Path)
    ap.add_argument("--manifests", default=Path("manifests"), type=Path)
    ap.add_argument("--out", default=Path("results/tables"), type=Path)
    args = ap.parse_args()

    a = block_a(args.predictions)
    b = block_b(args.predictions, args.manifests)
    c = block_c(args.predictions)
    all_rows = pd.concat([a, b, c], ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(args.out / "table10.csv", index=False)
    print(f"wrote {args.out / 'table10.csv'} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
