"""Build the group-aware 70/10/20 split and five re-splits.

Input:  manifests/main_manifest.csv (from build_manifest.py), embeddings/<dataset>.npy
Output: adds `component` + `partition` columns to the main manifest, writes
        manifests/resplit_<seed>.csv for each re-split seed.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from splitting.groupaware_split import split_manifest  # noqa: E402
from splitting.union_find import UnionFind  # noqa: E402


# Thresholds used throughout the paper for near-duplicate edge construction.
PHASH_HAMMING_MAX = 4
COSINE_MIN = 0.990


def _phash_edges(df: pd.DataFrame) -> list[tuple[int, int]]:
    """pHash Hamming distance ≤ PHASH_HAMMING_MAX within each dataset."""
    edges: list[tuple[int, int]] = []
    for _, sub in df.groupby("dataset"):
        idx = sub.index.to_list()
        hashes = [imagehash.hex_to_hash(h) for h in sub["phash"].to_list()]
        for i, j in combinations(range(len(hashes)), 2):
            if (hashes[i] - hashes[j]) <= PHASH_HAMMING_MAX:
                edges.append((idx[i], idx[j]))
    return edges


def _sha_edges(df: pd.DataFrame) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for _, sub in df.groupby("dataset"):
        by_sha: dict[str, list[int]] = {}
        for i, sha in zip(sub.index.to_list(), sub["sha256"].to_list()):
            by_sha.setdefault(sha, []).append(i)
        for ids in by_sha.values():
            if len(ids) > 1:
                anchor = ids[0]
                for other in ids[1:]:
                    edges.append((anchor, other))
    return edges


def _cosine_edges(df: pd.DataFrame, embeddings_dir: Path) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for dataset, sub in df.groupby("dataset"):
        emb_path = embeddings_dir / f"{dataset}.npy"
        if not emb_path.exists():
            continue
        embs = np.load(emb_path)
        # Embeddings are already L2-normalised in build_manifest.py, so dot = cos.
        sims = embs @ embs.T
        idx = sub.index.to_list()
        n = len(idx)
        for i in range(n):
            # Upper triangle only to avoid duplicates.
            js = np.where(sims[i, i + 1:] >= COSINE_MIN)[0]
            for j in js:
                edges.append((idx[i], idx[i + 1 + j]))
    return edges


def _assign_components(df: pd.DataFrame, all_edges: list[tuple[int, int]]) -> pd.DataFrame:
    """Assign a global `component` id (unique across datasets) from Union-Find."""
    n = len(df)
    uf = UnionFind(n)
    for a, b in all_edges:
        uf.union(a, b)
    comp_of = [uf.find(i) for i in range(n)]
    # Re-number to contiguous ids 0..k-1.
    remap: dict[int, int] = {}
    contig = []
    for c in comp_of:
        if c not in remap:
            remap[c] = len(remap)
        contig.append(remap[c])
    df = df.copy()
    df["component"] = contig
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--embeddings-dir", default=Path("embeddings"), type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 200, 300, 400, 500])
    ap.add_argument("--main-seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    print(f"Loaded {len(df)} rows from {args.manifest}")

    sha = _sha_edges(df)
    ph = _phash_edges(df)
    cos = _cosine_edges(df, args.embeddings_dir)
    print(f"Edges: SHA-256={len(sha)}, pHash≤{PHASH_HAMMING_MAX}={len(ph)}, cos≥{COSINE_MIN}={len(cos)}")

    df = _assign_components(df, sha + ph + cos)
    print(f"Components: {df['component'].nunique()}")

    # Main partition.
    df_main, main_report = split_manifest(df, edge_pairs=[], seed=args.main_seed)
    assert main_report.cross_partition_components == 0, "Leakage in main split"
    df_main.to_csv(args.manifest, index=False)

    # Five re-splits.
    out_dir = Path(args.manifest).parent
    for seed in args.seeds:
        df_rs, rep = split_manifest(df, edge_pairs=[], seed=seed)
        assert rep.cross_partition_components == 0, f"Leakage in re-split seed={seed}"
        rs_path = out_dir / f"resplit_{seed}.csv"
        df_rs.to_csv(rs_path, index=False)
        print(f"Wrote {rs_path} ({len(df_rs)} rows)")

    for dataset, sub in df_main.groupby("dataset"):
        parts = sub["partition"].value_counts().to_dict()
        print(f"  {dataset}  train/val/test images = "
              f"{parts.get('train', 0)} / {parts.get('val', 0)} / {parts.get('test', 0)}")


if __name__ == "__main__":
    main()
