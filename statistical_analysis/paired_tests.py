"""Paired statistical comparisons.

Two families, matching the paper:

1. Seed-paired: five seeds per (model, dataset). For each pair (full model vs control)
   the per-seed accuracy differences form a vector of length 5. A two-sided paired
   t-test on this vector is then Holm-corrected across the family of control
   comparisons for that dataset.

2. Re-split-paired: five independent re-splits. Each re-split's per-image prediction
   file for each model is used to compute an accuracy, and the full model vs control
   difference across the five re-splits is summarised by a 95% CI.

Also implements a cluster bootstrap over components for inference that respects
the leakage-controlled partitioning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass
class PairedResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    t_stat: float
    p_raw: float


def paired_t(diffs: Sequence[float], alpha: float = 0.05) -> PairedResult:
    d = np.asarray(diffs, dtype=float)
    n = d.size
    mean = float(d.mean())
    if n < 2:
        return PairedResult(mean, mean, mean, float("nan"), float("nan"))
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    tstat, pval = stats.ttest_rel(d, np.zeros_like(d))
    return PairedResult(mean, mean - tcrit * se, mean + tcrit * se, float(tstat), float(pval))


def holm_correction(p_values: Sequence[float]) -> list[float]:
    """Holm step-down p-value adjustment."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def cluster_bootstrap_diff(
    predA: np.ndarray, predB: np.ndarray, labels: np.ndarray, components: np.ndarray,
    n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05,
) -> PairedResult:
    """95% CI of (acc_A - acc_B) via resampling components, not images."""
    rng = np.random.default_rng(seed)
    unique = np.unique(components)
    idx_by_comp = {int(c): np.where(components == c)[0] for c in unique}

    correctA = (predA == labels).astype(np.int64)
    correctB = (predB == labels).astype(np.int64)

    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([idx_by_comp[int(c)] for c in sample])
        diffs[b] = correctA[idx].mean() - correctB[idx].mean()

    mean = float((correctA.mean() - correctB.mean()))
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    return PairedResult(mean, lo, hi, float("nan"), float("nan"))


def bootstrap_f1_ci(
    labels: np.ndarray, preds: np.ndarray, class_id: int,
    n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Nonparametric bootstrap 95% CI of per-class F1.

    Resamples (label, pred) pairs with replacement; computes precision, recall and
    F1 for the target class on each resample; returns (point_estimate, lo, hi).
    """
    rng = np.random.default_rng(seed)
    n = labels.size

    def _f1(l: np.ndarray, p: np.ndarray) -> float:
        tp = int(((p == class_id) & (l == class_id)).sum())
        fp = int(((p == class_id) & (l != class_id)).sum())
        fn = int(((p != class_id) & (l == class_id)).sum())
        if tp == 0:
            return 0.0
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        return 2 * prec * rec / (prec + rec)

    point = _f1(labels, preds)
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = _f1(labels[idx], preds[idx])
    lo = float(np.quantile(draws, alpha / 2))
    hi = float(np.quantile(draws, 1 - alpha / 2))
    return point, lo, hi
