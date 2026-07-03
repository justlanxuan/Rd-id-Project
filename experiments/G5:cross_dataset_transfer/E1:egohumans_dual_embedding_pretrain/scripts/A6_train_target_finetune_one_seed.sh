#!/usr/bin/env bash
# Experiment Note: E1-A6-train-target-finetune-one-seed
# Fine-tune a source pre-trained branch on custom target data.
# Usage: A6_train_target_finetune_one_seed.sh <local|global> <seed> [device]
set -euo pipefail

REPO=/home/fzliang/Autism-project
E1_DIR="$REPO/experiments/G5:cross_dataset_transfer/E1:egohumans_dual_embedding_pretrain"
ARTIFACTS_DIR="$E1_DIR/artifacts"
E2_DIR="$REPO/experiments/G3:custom_failure_diagnosis/E2:mobind_on_custom_same_split"
E11_DIR="$REPO/experiments/G4:mobind_single_imu_adaptation/E11:dual_embedding_local_global"
MOBIND_ROOT=/home/fzliang/MoBind

BRANCH=${1:-local}
SEED=${2:-0}
DEVICE=${3:-cuda:0}

gpu_idx=${DEVICE##*:}

if [[ "$BRANCH" == "local" ]]; then
    SOURCE_DIR="$ARTIFACTS_DIR/source_w24_seed${SEED}_local"
    TARGET_BASE="$E2_DIR/config/stage2_w24_base.yaml"
    TARGET_DIR="$ARTIFACTS_DIR/target_w24_seed${SEED}_local"
elif [[ "$BRANCH" == "global" ]]; then
    SOURCE_DIR="$ARTIFACTS_DIR/source_w24_seed${SEED}_global"
    TARGET_BASE="$E11_DIR/config/stage2_w24_global_base.yaml"
    TARGET_DIR="$ARTIFACTS_DIR/target_w24_seed${SEED}_global"
else
    echo "Unknown branch $BRANCH (use local or global)"
    exit 1
fi

SOURCE_STAGE2_EXP=$(cat "$SOURCE_DIR/stage2_exp_path.txt" 2>/dev/null || echo "")
SOURCE_STAGE1_EXP=$(cat "$SOURCE_DIR/stage1_exp_path.txt" 2>/dev/null || echo "")
if [[ -z "$SOURCE_STAGE2_EXP" || -z "$SOURCE_STAGE1_EXP" ]]; then
    echo "[ERROR] Source checkpoint not found for $BRANCH seed=$SEED"
    exit 1
fi

mkdir -p "$TARGET_DIR"

echo "=== E1 Target fine-tune $BRANCH seed=$SEED on GPU $gpu_idx ==="
echo "Source Stage2: $SOURCE_STAGE2_EXP"

# ---------- Stage 2 fine-tune ----------
TARGET_CFG="$TARGET_DIR/stage2.yaml"
conda run --no-capture-output -n mobind_repro python "$E2_DIR/scripts/generate_config.py" \
  --base "$TARGET_BASE" \
  --seed "$SEED" \
  --output_dir "$TARGET_DIR/stage2" \
  --data_root "$E2_DIR/data" \
  --stage1_exp "$SOURCE_STAGE1_EXP" \
  --out_path "$TARGET_CFG"

echo "[Stage2] Fine-tuning..."
CUDA_VISIBLE_DEVICES="$gpu_idx" WANDB_MODE=disabled conda run --no-capture-output -n mobind_repro \
  python "$MOBIND_ROOT/train_contrastive.py" --config "$TARGET_CFG" \
  --pretrained "$SOURCE_STAGE2_EXP/checkpoints/best.pt" \
  > "$TARGET_DIR/stage2_finetune.log" 2>&1

TARGET_STAGE2_EXP=$(ls -td "$TARGET_DIR/stage2/Custom/"* 2>/dev/null | head -1)
echo "[Stage2] Output: $TARGET_STAGE2_EXP"
echo "$TARGET_STAGE2_EXP" > "$TARGET_DIR/stage2_exp_path.txt"

echo "=== E1 Target fine-tune $BRANCH seed=$SEED complete ==="
