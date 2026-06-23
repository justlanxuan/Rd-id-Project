#!/bin/bash
set -e
LOG=/home/fzliang/Autism-project/experiments/G_egohumans/E1:dataset_integration/logs/extract_minimal.log
exec > "$LOG" 2>&1
eval "$(conda shell.bash hook)"
conda activate autism_test
cd /home/fzliang/Autism-project
python scripts/egohumans/extract_egohumans_minimal.py \
  --data_dir /data/lyxie/ReID/Data/egohumans
echo "Extraction complete."
