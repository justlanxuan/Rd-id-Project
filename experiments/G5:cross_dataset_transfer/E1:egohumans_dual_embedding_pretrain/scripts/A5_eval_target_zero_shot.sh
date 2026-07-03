#!/usr/bin/env bash
# Experiment Note: E1-A5-eval-target-zero-shot
# Zero-shot evaluate source pre-trained dual-embedding on custom test.
# Usage: A5_eval_target_zero_shot.sh [sim_norm]
set -euo pipefail

REPO=/home/fzliang/Autism-project
E1_DIR="$REPO/experiments/G5:cross_dataset_transfer/E1:egohumans_dual_embedding_pretrain"
ARTIFACTS_DIR="$E1_DIR/artifacts"
A5="$REPO/experiments/G4:mobind_single_imu_adaptation/E11:dual_embedding_local_global/scripts/A5_eval_fusion.py"

SIM_NORM=${1:-none}
SEEDS=(0 42 123 1 2 3)

for SEED in "${SEEDS[@]}"; do
    LOCAL_EXP=$(cat "$ARTIFACTS_DIR/source_w24_seed${SEED}_local/stage2_exp_path.txt" 2>/dev/null || echo "")
    GLOBAL_EXP=$(cat "$ARTIFACTS_DIR/source_w24_seed${SEED}_global/stage2_exp_path.txt" 2>/dev/null || echo "")
    OUT="$ARTIFACTS_DIR/source_w24_seed${SEED}_global/results_zero_shot_${SIM_NORM}.json"
    if [[ -z "$LOCAL_EXP" || -z "$GLOBAL_EXP" ]]; then
        echo "[seed=$SEED] Skipping (missing source checkpoint)"
        continue
    fi
    echo "[seed=$SEED] Zero-shot eval (norm=$SIM_NORM)..."
    conda run --no-capture-output -n mobind_repro python "$A5" \
      --window w24 \
      --seed "$SEED" \
      --local_exp "$LOCAL_EXP" \
      --global_exp "$GLOBAL_EXP" \
      --sim_norm "$SIM_NORM" \
      --out_json "$OUT" \
      --device cuda:0
    echo "  -> $OUT"
done
