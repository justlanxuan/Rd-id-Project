#!/usr/bin/env bash
# Experiment Note: E1-A5-eval-target-finetuned-seed0
# Evaluate fine-tuned seed0 local branch on custom test.
# For alpha=1.0, the global branch score weight is 0, so its checkpoint only needs to exist.
set -euo pipefail

REPO=/home/fzliang/Autism-project
E1_DIR="$REPO/experiments/G5:cross_dataset_transfer/E1:egohumans_dual_embedding_pretrain"
ARTIFACTS_DIR="$E1_DIR/artifacts"
A5="$REPO/experiments/G4:mobind_single_imu_adaptation/E11:dual_embedding_local_global/scripts/A5_eval_fusion.py"

SIM_NORM=${1:-none}
SEED=0
DEVICE=${2:-cuda:0}

LOCAL_EXP=$(cat "$ARTIFACTS_DIR/target_w24_seed${SEED}_local/stage2_exp_path.txt" 2>/dev/null || echo "")
GLOBAL_EXP=$(cat "$ARTIFACTS_DIR/source_w24_seed${SEED}_global/stage2_exp_path.txt" 2>/dev/null || echo "")
OUT="$ARTIFACTS_DIR/target_w24_seed${SEED}_local/results_finetune_${SIM_NORM}.json"

if [[ -z "$LOCAL_EXP" || -z "$GLOBAL_EXP" ]]; then
    echo "[ERROR] Missing checkpoint for seed=$SEED"
    exit 1
fi

echo "=== E1 Fine-tuned eval seed=$SEED (norm=$SIM_NORM) on $DEVICE ==="
echo "Local (fine-tuned): $LOCAL_EXP"
CUDA_VISIBLE_DEVICES="${DEVICE##*:}" conda run --no-capture-output -n mobind_repro python "$A5" \
  --window w24 \
  --seed "$SEED" \
  --local_exp "$LOCAL_EXP" \
  --global_exp "$GLOBAL_EXP" \
  --sim_norm "$SIM_NORM" \
  --out_json "$OUT" \
  --device cuda:0

echo "=== Saved $OUT ==="
