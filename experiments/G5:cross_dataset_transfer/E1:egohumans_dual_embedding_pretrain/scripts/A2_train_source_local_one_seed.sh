#!/usr/bin/env bash
# Experiment Note: E1-A2-train-source-local-one-seed
# Train Model-L (local) on EgoHumans for one seed.
set -euo pipefail

REPO=/home/fzliang/Autism-project
E1_DIR="$REPO/experiments/G5:cross_dataset_transfer/E1:egohumans_dual_embedding_pretrain"
CONFIG_DIR="$E1_DIR/config"
ARTIFACTS_DIR="$E1_DIR/artifacts"
MOBIND_ROOT=/home/fzliang/MoBind

SEED=${1:-0}
DEVICE=${2:-cuda:0}

gpu_idx=${DEVICE##*:}
SEED_OUT="$ARTIFACTS_DIR/source_w24_seed${SEED}_local"
mkdir -p "$SEED_OUT"

echo "=== E1 Source Model-L seed=$SEED on GPU $gpu_idx ==="

# ---------- Stage 1 ----------
STAGE1_CFG="$SEED_OUT/stage1.yaml"
conda run --no-capture-output -n mobind_repro python "$REPO/experiments/G3:custom_failure_diagnosis/E2:mobind_on_custom_same_split/scripts/generate_config.py" \
  --base "$CONFIG_DIR/stage1_local_base.yaml" \
  --seed "$SEED" \
  --output_dir "$SEED_OUT/stage1" \
  --data_root "/data/lyxie/ReID/Data/egohumans/EgoHumans" \
  --out_path "$STAGE1_CFG"

echo "[Stage1] Training..."
CUDA_VISIBLE_DEVICES="$gpu_idx" WANDB_MODE=disabled conda run --no-capture-output -n mobind_repro \
  python "$MOBIND_ROOT/train_contrastive.py" --config "$STAGE1_CFG" \
  > "$SEED_OUT/stage1_train.log" 2>&1

STAGE1_EXP=$(ls -td "$SEED_OUT/stage1/EgoHumans/"* 2>/dev/null | head -1)
if [[ -z "$STAGE1_EXP" ]]; then
  echo "[ERROR] Stage1 output directory not found under $SEED_OUT/stage1/EgoHumans"
  exit 1
fi
echo "[Stage1] Output: $STAGE1_EXP"

# ---------- Stage 2 ----------
STAGE2_CFG="$SEED_OUT/stage2.yaml"
conda run --no-capture-output -n mobind_repro python "$REPO/experiments/G3:custom_failure_diagnosis/E2:mobind_on_custom_same_split/scripts/generate_config.py" \
  --base "$CONFIG_DIR/stage2_local_base.yaml" \
  --seed "$SEED" \
  --output_dir "$SEED_OUT/stage2" \
  --data_root "/data/lyxie/ReID/Data/egohumans/EgoHumans" \
  --stage1_exp "$STAGE1_EXP" \
  --out_path "$STAGE2_CFG"

echo "[Stage2] Training..."
CUDA_VISIBLE_DEVICES="$gpu_idx" WANDB_MODE=disabled conda run --no-capture-output -n mobind_repro \
  python "$MOBIND_ROOT/train_contrastive.py" --config "$STAGE2_CFG" \
  > "$SEED_OUT/stage2_train.log" 2>&1

STAGE2_EXP=$(ls -td "$SEED_OUT/stage2/EgoHumans/"* 2>/dev/null | head -1)
echo "[Stage2] Output: $STAGE2_EXP"
echo "$STAGE2_EXP" > "$SEED_OUT/stage2_exp_path.txt"

echo "=== E1 Source Model-L seed=$SEED complete ==="
