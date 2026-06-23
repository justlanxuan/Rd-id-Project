#!/bin/bash
set -e
LOG=/home/fzliang/Autism-project/experiments/G_egohumans/E1:dataset_integration/logs/mobind_extract_imu.log
exec > "$LOG" 2>&1
eval "$(conda shell.bash hook)"
conda activate autism_test
cd /home/fzliang/MoBind
python preprocess/EgoHumans/extract_data.py \
  --data_dir /data/lyxie/ReID/Data/egohumans \
  --mode extract_imu
echo "MoBInd IMU extraction complete."
