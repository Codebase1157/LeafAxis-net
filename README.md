# LeafAxis-Net

Reproducibility code for *LeafAxis-Net: A Multi-Axis Attention CNN–Transformer Network for Sugarcane Leaf Disease Classification under a Unified Leakage-Controlled Protocol*.

## Status of this repository


```
LeafAxis-Net/
├── README.md
├── requirements.txt
├── REPRODUCE.json              # table/figure → script/config/predictions/SHA map
│
├── datasets/                   # source dataset documentation
├── preprocessing/              # image loading, resizing, normalization
├── splitting/                  # Union-Find group-aware splitter
├── manifests/                  # frozen manifest + 5 re-splits (filled by splitting/)
├── models/
│   ├── leafaxis/               # MAAB, CBF, LSTS, LeafAxis-Net forward pass
│   └── baselines/              # 11 baseline wrappers
├── pretraining/                # PlantVillage pre-training
├── configs/
│   ├── models/                 # per-model architecture configs
│   └── training/               # per-run training configs (12 models × 3 datasets × 5 seeds)
├── checkpoints/                # filled by training; empty otherwise
├── logs/                       # training logs; empty otherwise
├── predictions/                # per-seed test predictions; empty otherwise
├── scripts/                    # top-level entry points (train, eval, hash, etc.)
├── statistical_analysis/       # paired t-test, Holm, bootstrap, CIs
└── results/
    ├── tables/                 # Supp S1–S4 regenerated from manifests + predictions
    └── figures/                # Figs 2, 3, 7, 8, 9 regenerated from predictions
```

## End-to-end reproduction

```bash
# 0. Install
pip install -r requirements.txt

# 1. Fetch the three source datasets to ./data_raw/{SLD,DK,BD}/
#    See datasets/download_instructions.md

# 2. Build the frozen manifest (SHA-256 + 64-bit pHash + components)
python scripts/build_manifest.py --root ./data_raw --out manifests/main_manifest.csv

# 3. Build the group-aware 70/10/20 split and five re-splits
python scripts/make_splits.py --manifest manifests/main_manifest.csv --seeds 100 200 300 400 500

# 4. Verify no duplicate component crosses a partition
python scripts/verify_no_leakage.py --manifest manifests/main_manifest.csv

# 5. Pre-train on PlantVillage (or point to an existing checkpoint)
python scripts/pretrain.py --config pretraining/plantvillage.yaml

# 6. Train every model × dataset × seed (writes checkpoints + per-seed predictions)
bash scripts/train_all.sh

# 7. Hash every checkpoint and prediction file, write hashes into REPRODUCE.json
python scripts/hash_artifacts.py

# 8. Regenerate every table and figure from predictions + manifest
python scripts/make_tables.py
python scripts/make_figures.py

# 9. Run the paired statistical analyses
python scripts/run_statistics.py
```


## License

Code: MIT. The three source datasets have their own licenses — see `datasets/dataset_information.md`.
