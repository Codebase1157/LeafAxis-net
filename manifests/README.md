# Manifests

These files are produced by the splitting scripts; do not edit by hand.

- `main_manifest.csv` — one row per retained image (after deduplication), with the columns:
  - `path, dataset, class_name, label, sha256, phash, component, partition`
  - Written by `scripts/build_manifest.py`, extended with `component` and `partition` columns by `scripts/make_splits.py`.
- `resplit_100.csv` … `resplit_500.csv` — the five independent re-splits used for seed-independent variance.

The "frozen" nature of the main manifest is what Table 5, Supplementary Table S1 and the paired-split analyses depend on. `scripts/verify_no_leakage.py` asserts that no component crosses a train / val / test partition in any of these files; run it after every split and before every evaluation.

If you want to re-freeze with a different split seed, change `--main-seed` on `scripts/make_splits.py` and all downstream tables will be rebuilt from the new manifest the next time `make_tables.py` runs.
