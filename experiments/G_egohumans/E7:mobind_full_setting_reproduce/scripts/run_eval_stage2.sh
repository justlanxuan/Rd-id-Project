#!/usr/bin/env bash
# E7: Evaluate retrained MoBInd Stage2 (official 5-IMU / 100-frame setting)
# on the 24 official MoBInd test sequences.
set -euo pipefail

STAGE2_EXP="/home/fzliang/MoBind/outputs/EgoHumans/stage2_E7_repro/EgoHumans/06-26-2026:11:04:59"
OUT_JSON="/home/fzliang/Autism-project/experiments/G_egohumans/E7:mobind_full_setting_reproduce/results/mobind_retrained_frameacc_aligned_test.json"

cd /home/fzliang/Autism-project
conda run -n mobind_repro python \
  experiments/G_egohumans/E5:mobind_aligned_splits/scripts/A5_eval_mobind_aligned_test.py \
  --mobind_exp_dir "${STAGE2_EXP}" \
  --window_size 100 \
  --stride 16 \
  --output_json "${OUT_JSON}"
