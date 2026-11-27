"""Hard check that no connected component crosses train / val / test partitions.

Exits 0 on clean split, 1 otherwise. Also prints the per-dataset per-partition
image and component counts used in the paper's Table 5 and Supplementary Table S1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def check(df: pd.DataFrame) -> int:
    cross = 0
    for comp, sub in df.groupby("component"):
        parts = sub["partition"].nunique()
        if parts > 1:
            print(f"[leak] component {comp} spans {parts} partitions: "
                  f"{sub['partition'].value_counts().to_dict()}")
            cross += 1
    return cross


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--resplits", nargs="*", type=Path, default=[])
    args = ap.parse_args()

    status = 0

    df = pd.read_csv(args.manifest)
    print(f"=== {args.manifest} ===")
    if check(df) > 0:
        status = 1
    for dataset, sub in df.groupby("dataset"):
        img = sub["partition"].value_counts().to_dict()
        comp = (
            sub.groupby("component")["partition"].first()
            .value_counts().to_dict()
        )
        print(f"  {dataset}  images train/val/test = "
              f"{img.get('train', 0)} / {img.get('val', 0)} / {img.get('test', 0)}")
        print(f"  {dataset}  comps  train/val/test = "
              f"{comp.get('train', 0)} / {comp.get('val', 0)} / {comp.get('test', 0)}")

    for rs in args.resplits:
        print(f"\n=== {rs} ===")
        df_rs = pd.read_csv(rs)
        if check(df_rs) > 0:
            status = 1

    print("\nOK" if status == 0 else "\nLEAKAGE FOUND")
    return status


if __name__ == "__main__":
    sys.exit(main())
