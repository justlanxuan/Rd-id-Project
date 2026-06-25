#!/bin/bash
# Experiment Note: A1-train-mobind
# Train MoBInd Stage1 and Stage2 with single RightWrist IMU and 24-frame window.
set -e

cd /home/fzliang/MoBind
export WANDB_MODE=disabled

echo "[A1] Stage1 training..."
conda run -n mobind_repro python train_contrastive.py \
  --config /home/fzliang/Autism-project/experiments/G_egohumans/E6:fair_single_imu_same_window/config/MoBind_stage1_w24.yaml

echo "[A1] Stage2 training..."
conda run -n mobind_repro python train_contrastive.py \
  --config /home/fzliang/Autism-project/experiments/G_egohumans/E6:fair_single_imu_same_window/config/MoBind_stage2_w24.yaml

echo "[A1] Done."
