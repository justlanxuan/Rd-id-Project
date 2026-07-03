#!/usr/bin/env bash
# Experiment Note: E1-A3-train-source-global-all
# Train Model-G on EgoHumans for all seeds in parallel.
set -euo pipefail

REPO=/home/fzliang/Autism-project
E1_DIR="$REPO/experiments/G5:cross_dataset_transfer/E1:egohumans_dual_embedding_pretrain"
SCRIPT="$E1_DIR/scripts/A3_train_source_global_one_seed.sh"

SEEDS=(0 42 123 1 2 3)

for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    DEVICE="cuda:$i"
    echo "[Launch] Model-G seed=$SEED on $DEVICE"
    bash "$SCRIPT" "$SEED" "$DEVICE" > "$E1_DIR/results/A3_global_seed${SEED}.log" 2>&1 &
done

wait
echo "=== All Model-G seeds complete ==="
