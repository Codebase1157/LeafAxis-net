"""Delete every illustrative placeholder file under manifests/, predictions/, checkpoints/, logs/.

A file is a placeholder iff it carries the `illustrative_placeholder` marker written
by `scripts/make_illustrative_placeholders.py`:
    .npz          has `illustrative_placeholder` key set to True
    .pth          loads to a dict with `illustrative_placeholder` set to True
    .jsonl        first line parses to {"illustrative_placeholder": true, ...}
    manifest row  has `component == -1`

Real training artifacts never carry these markers, so this script will not touch
any file produced by `scripts/train.py` or `scripts/build_manifest.py`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def is_placeholder_npz(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as z:
            if "illustrative_placeholder" in z.files:
                return bool(z["illustrative_placeholder"].item())
    except Exception:
        return False
    return False


def is_placeholder_pth(path: Path) -> bool:
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
        return bool(d.get("illustrative_placeholder", False)) if isinstance(d, dict) else False
    except Exception:
        return False


def is_placeholder_jsonl(path: Path) -> bool:
    try:
        first = path.open().readline()
        obj = json.loads(first)
        return bool(obj.get("illustrative_placeholder", False))
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=Path(__file__).resolve().parent.parent, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    removed, kept = 0, 0
    for p in (args.root / "predictions").rglob("*.npz"):
        if is_placeholder_npz(p):
            print(f"[remove] {p}")
            if not args.dry_run: p.unlink()
            removed += 1
        else:
            kept += 1
    for p in (args.root / "checkpoints").rglob("*.pth"):
        if is_placeholder_pth(p):
            print(f"[remove] {p}")
            if not args.dry_run: p.unlink()
            removed += 1
        else:
            kept += 1
    for p in (args.root / "logs").rglob("*.jsonl"):
        if is_placeholder_jsonl(p):
            print(f"[remove] {p}")
            if not args.dry_run: p.unlink()
            removed += 1
        else:
            kept += 1

    manifest = args.root / "manifests" / "main_manifest.csv"
    if manifest.exists():
        df = pd.read_csv(manifest)
        n_before = len(df)
        df = df[df.get("component", 0) != -1]
        if len(df) != n_before:
            print(f"[manifest] removed {n_before - len(df)} placeholder rows from {manifest}")
            if not args.dry_run:
                df.to_csv(manifest, index=False)

    print(f"\n{'DRY RUN — nothing deleted' if args.dry_run else 'done'}: "
          f"removed={removed} files, kept={kept} real files")


if __name__ == "__main__":
    main()
