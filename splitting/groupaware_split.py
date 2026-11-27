"""Group-aware 70/10/20 split with leakage control.

Every image is assigned to a connected component via the Union-Find over the three
near-duplicate edge sets (exact SHA-256, pHash Hamming distance ≤ 4, cosine ≥ 0.990
on a fixed-feature backbone). The split assigns whole components to one partition,
never splitting a component across partitions. A component is placed in `train`,
`val` or `test` by a random draw weighted by the remaining per-partition capacity
of its class, so the realised ratios approximate 70 : 10 : 20 for every class.

Any component with internally conflicting labels is listed in the audit output and
is placed in the partition where the majority label falls.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .union_find import UnionFind


TARGET_RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}


@dataclass
class SplitReport:
    partition_counts: dict[str, dict[str, int]]   # dataset -> partition -> image count
    component_counts: dict[str, dict[str, int]]   # dataset -> partition -> component count
    conflicting_components: list[tuple[int, dict[str, int]]]  # (component, {label: n})
    cross_partition_components: int               # must be 0 at the end


def _assign_components_to_partitions(
    components: dict[int, list[int]],
    labels: list[int],
    target_per_class: dict[int, dict[str, int]],
    rng: random.Random,
) -> dict[int, str]:
    """Greedy largest-component-first assignment weighted by remaining capacity."""
    comp_order = sorted(components.items(), key=lambda kv: -len(kv[1]))
    assignment: dict[int, str] = {}
    remaining = {c: {p: v for p, v in d.items()} for c, d in target_per_class.items()}

    for comp_id, members in comp_order:
        # Majority class of the component.
        class_counts: dict[int, int] = defaultdict(int)
        for idx in members:
            class_counts[labels[idx]] += 1
        cls = max(class_counts.items(), key=lambda kv: kv[1])[0]

        weights = [max(remaining[cls].get(p, 0), 0) for p in ("train", "val", "test")]
        if sum(weights) == 0:
            # Every bucket for this class is full; put it in the largest-remaining overall.
            weights = [1, 1, 1]
        choice = rng.choices(("train", "val", "test"), weights=weights, k=1)[0]
        assignment[comp_id] = choice
        remaining[cls][choice] = remaining[cls].get(choice, 0) - len(members)
    return assignment


def split_manifest(
    df: pd.DataFrame,
    edge_pairs: list[tuple[int, int]],
    seed: int,
) -> tuple[pd.DataFrame, SplitReport]:
    """Add a `partition` column to `df` based on the given near-duplicate edges.

    `df` must carry integer columns `component_seed` (0..n-1, reset per call),
    `label`, `dataset`. `edge_pairs` are (i, j) index pairs that should be merged.
    """
    df = df.reset_index(drop=True).copy()
    rng = random.Random(seed)

    reports: dict[str, SplitReport] = {}
    all_assignments = [None] * len(df)

    # Split dataset-by-dataset so no component can cross a dataset boundary.
    for dataset, sub in df.groupby("dataset", sort=False):
        sub_idx = sub.index.to_list()
        local_of_global = {g: i for i, g in enumerate(sub_idx)}
        n = len(sub_idx)
        uf = UnionFind(n)
        for a, b in edge_pairs:
            if a in local_of_global and b in local_of_global:
                uf.union(local_of_global[a], local_of_global[b])

        comps = uf.components()
        labels = sub["label"].to_list()

        # Per-class target image counts for this dataset.
        per_class_n = sub["label"].value_counts().to_dict()
        target_per_class = {
            cls: {
                "train": int(round(n_ * TARGET_RATIOS["train"])),
                "val":   int(round(n_ * TARGET_RATIOS["val"])),
                "test":  n_ - int(round(n_ * TARGET_RATIOS["train"])) - int(round(n_ * TARGET_RATIOS["val"])),
            }
            for cls, n_ in per_class_n.items()
        }

        assignment = _assign_components_to_partitions(comps, labels, target_per_class, rng)

        for comp_id, members in comps.items():
            part = assignment[comp_id]
            for local_i in members:
                all_assignments[sub_idx[local_i]] = part

    df["partition"] = all_assignments
    return df, _build_report(df)


def _build_report(df: pd.DataFrame) -> SplitReport:
    pc = (
        df.groupby(["dataset", "partition"]).size().unstack(fill_value=0).to_dict(orient="index")
    )
    comps = (
        df.groupby(["dataset", "component"])["partition"].first()
        .reset_index()
        .groupby(["dataset", "partition"]).size().unstack(fill_value=0).to_dict(orient="index")
    )
    # Check for conflicting labels within a component.
    conflicts: list[tuple[int, dict[str, int]]] = []
    for comp, sub in df.groupby("component"):
        counts = sub["label"].value_counts().to_dict()
        if len(counts) > 1:
            conflicts.append((int(comp), {int(k): int(v) for k, v in counts.items()}))
    # Check no component spans partitions.
    cross = sum(
        1 for _, sub in df.groupby("component") if sub["partition"].nunique() > 1
    )
    return SplitReport(
        partition_counts=pc,
        component_counts=comps,
        conflicting_components=conflicts,
        cross_partition_components=cross,
    )
