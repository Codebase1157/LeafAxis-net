#!/usr/bin/env bash
# Train every (model, dataset, seed) combination used in the paper.
# 12 models * 3 datasets * 5 seeds = 180 runs. Edit MODELS / DATASETS / SEEDS below to
# reduce scope while iterating.

set -euo pipefail

MODELS=(
  resnet50 efficientnet_b3 vit_b_16 swin_t convnext_t
  mobilevit_s efficientformer_l1 fastvit_t12 coatnet_0
  rema_mobilevit dual_branch leafaxis_net
)
DATASETS=(SLD DK BD)
SEEDS=(0 1 2 3 4)

for model in "${MODELS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for s in "${SEEDS[@]}"; do
      cfg="configs/training/${ds}/${model}.yaml"
      echo "=== ${model} on ${ds}, seed=${s} ==="
      python scripts/train.py --config "${cfg}" --seed "${s}"
    done
  done
done
